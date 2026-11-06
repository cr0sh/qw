// SPDX-License-Identifier: Apache-2.0
//
//! The D-LSA forward of one full-attention layer (`docs/dlsa.md` §7.1, §12): the index keys and
//! queries are captured after q_norm/k_norm and before RoPE, the read is restricted to each
//! row's list, and everything else (projections, RoPE, the gate, o_proj) is the dense code.

use std::time::Instant;

use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Qwen3NextAttention;
use crate::dlsa::{
    DeviceSelection, DlsaBackend, DlsaBucketCache, DlsaConfig, DlsaGeometry, DlsaLayerMaps,
    DlsaSelection, DlsaStats, causal_attention, gather_attention, index_keys, index_queries,
    masked_attention, profile, sparse_read,
};
use profile::{DlsaStage, lap};

/// What one layer's D-LSA forward reads and updates besides its KV cache.
pub(crate) struct DlsaLayerStep<'a> {
    pub maps: &'a DlsaLayerMaps,
    pub geometry: &'a DlsaGeometry,
    pub config: &'a DlsaConfig,
    pub cache: &'a mut DlsaBucketCache,
    pub stats: &'a mut DlsaStats,
    /// Commit the index at the end of this forward (prefill, decode). A verify forward passes
    /// `false`, so a rollback can still reopen a block it closed; every forward first commits
    /// what the previous one left after any rollback.
    pub commit: bool,
}

/// A forward's read lists: on the host (task 3 and the parity mode) or on the device.
enum Selected {
    Host(DlsaSelection),
    Device(DeviceSelection),
}

impl Qwen3NextAttention {
    /// [`forward_with_position_ids`](Self::forward_with_position_ids) under D-LSA (no mask: the
    /// read is causal by construction), including `o_proj`.
    ///
    /// Panics on a broken invariant: a batch other than 1, an index cache out of step with the
    /// KV cache, a KV cache whose slots are not positions, or a non-finite index key (under
    /// the Metal backend, found when the deferred keys are read back).
    pub(crate) fn forward_dlsa(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        position_ids: Option<&MlxArray>,
        step: DlsaLayerStep<'_>,
    ) -> UniquePtr<MlxArray> {
        let heads = self.project_heads(x);
        let (b, l) = (heads.b, heads.l);
        assert_eq!(b, 1, "D-LSA runs one sequence (batch 1), got batch {b}");
        let offset = cache.offset;
        assert_eq!(
            cache.live_len(),
            offset,
            "D-LSA needs KV slots equal to positions (no front trim)"
        );
        let start = offset as usize;
        assert_eq!(
            step.cache.len(),
            start,
            "the D-LSA index holds {} keys but the KV cache {start}: a restore or rollback bypassed it",
            step.cache.len()
        );
        // Accept what the previous forward left (after any rollback)
        step.cache.commit();
        let rows = l as usize;

        let backend = step.config.backend;
        let fenced = profile::begin(start, &[x]);
        lap(
            DlsaStage::Projection,
            &[&heads.queries, &heads.keys, &heads.values, &heads.gate],
        );
        let started = Instant::now();
        let ik = index_keys(&heads.keys, &step.maps.w_k_idx, step.geometry)
            .expect("D-LSA index keys of a full-attention layer");
        lap(DlsaStage::IndexKeys, &[&ik]);
        let ward_before = fenced.then(|| step.cache.stats().cluster_time);
        // The Metal backend keeps keys that close no block on the device (no sync)
        let appended = if backend == DlsaBackend::Metal {
            step.cache.append_deferred(&ik)
        } else {
            step.cache.append_mlx(&ik)
        };
        appended.expect("D-LSA bucket cache append");
        if step.cache.deferred_keys() == 0 {
            // The append synced, so the counts of earlier forwards are ready
            step.stats.keys_read += step.cache.take_keys_read();
        }
        step.stats.index_time += started.elapsed();
        if let Some(before) = ward_before {
            let ward = step.cache.stats().cluster_time - before;
            profile::lap_split(DlsaStage::KeyReadback, DlsaStage::Ward, ward, &[]);
        }

        // The policy is per row: rows below `dense_below` read every causal key. A forward
        // entirely in that range (or on the dense key set) runs the unchanged dense attention;
        // one that crosses it reads its lower rows causally and selects for the rest.
        let dense_rows = step.config.dense_rows(start, rows);
        let selection = (dense_rows < rows).then(|| {
            let started = Instant::now();
            let iq = index_queries(&heads.queries, &step.maps.w_q_idx, step.geometry)
                .expect("D-LSA index queries of a full-attention layer");
            let iq = if dense_rows == 0 {
                iq
            } else {
                let width = mlxcel_core::array_shape(&iq)[1];
                mlxcel_core::slice(&iq, &[dense_rows as i32, 0], &[rows as i32, width])
            };
            lap(DlsaStage::IndexQueries, &[&iq]);
            let positions: Vec<usize> = (start + dense_rows..start + rows).collect();
            let sel = match backend {
                DlsaBackend::Mlx => step.cache.select(&iq, &positions).map(Selected::Host),
                DlsaBackend::Metal => step
                    .cache
                    .select_device(&iq, &positions)
                    .map(Selected::Device),
                // The parity mode: device lists, read back for the task-3 read
                DlsaBackend::MetalSelect => step
                    .cache
                    .select_device(&iq, &positions)
                    .and_then(|sel| sel.to_host())
                    .map(Selected::Host),
            }
            .expect("D-LSA selection");
            if backend == DlsaBackend::MetalSelect {
                lap(DlsaStage::ScoresReadback, &[]);
            }
            step.stats.select_time += started.elapsed();
            sel
        });
        if step.commit {
            step.cache.commit();
        }

        let (queries, keys, values) = self.rotate(
            &heads.queries,
            &heads.keys,
            &heads.values,
            offset,
            position_ids,
            b,
            l,
        );
        lap(DlsaStage::Rotate, &[&queries, &keys, &values]);
        let attn_out = match selection {
            None => {
                step.stats.dense_key_set_forwards += 1;
                let out = self.dense_attention(&queries, keys, values, cache, None, l);
                lap(DlsaStage::DenseRead, &[&out]);
                out
            }
            Some(sel) => {
                let sparse = rows - dense_rows;
                step.stats.sparse_forwards += 1;
                step.stats.sparse_rows += sparse as u64;
                let (cache_k, cache_v) = cache.update_and_fetch(keys, values);
                lap(DlsaStage::KvFetch, &[&cache_k, &cache_v]);
                let q_rows = |from: usize, to: usize| {
                    let shape = mlxcel_core::array_shape(&queries);
                    mlxcel_core::slice(
                        &queries,
                        &[0, 0, from as i32, 0],
                        &[shape[0], shape[1], to as i32, shape[3]],
                    )
                };
                let sparse_q = if dense_rows == 0 {
                    mlxcel_core::share(&queries)
                } else {
                    q_rows(dense_rows, rows)
                };
                let out = match sel {
                    Selected::Device(sel) => {
                        step.cache.count_keys_read(&sel);
                        sparse_read(&sparse_q, &cache_k, &cache_v, &sel, self.scale)
                    }
                    Selected::Host(sel) => {
                        step.stats.keys_read += sel.lengths.iter().map(|&n| n as u64).sum::<u64>();
                        if sparse == 1 {
                            gather_attention(&sparse_q, &cache_k, &cache_v, sel.list(0), self.scale)
                        } else {
                            masked_attention(
                                &sparse_q,
                                &cache_k,
                                &cache_v,
                                &sel,
                                start + dense_rows,
                                step.config.prefill_mask_rows,
                                self.scale,
                            )
                        }
                    }
                };
                lap(DlsaStage::Read, &[&out]);
                if dense_rows == 0 {
                    out
                } else {
                    // The rows below `dense_below` of a forward that crosses it (not counted
                    // in `sparse_rows` or `keys_read`)
                    let below = causal_attention(
                        &q_rows(0, dense_rows),
                        &cache_k,
                        &cache_v,
                        start,
                        self.scale,
                    );
                    mlxcel_core::concatenate(&below, &out, 2)
                }
            }
        };
        let output = self.finish(&attn_out, &heads.gate, b, l);
        let output = self.o_proj.forward(&output);
        lap(DlsaStage::Finish, &[&output]);
        profile::end();
        output
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mlxcel_core::cache::KVCacheMode;
    use mlxcel_core::weights::WeightMap;

    use super::*;
    use crate::dlsa::test_model::{
        TEST_HEAD_DIM, TEST_HEADS, TEST_INDEX, TEST_KV_HEADS, max_abs_diff, noise_array,
        test_layer_maps,
    };
    use crate::dlsa::{DlsaMapDtype, to_host_f32};
    use crate::qwen3_next::Qwen3NextConfig;

    const HIDDEN: i32 = 64;

    fn test_attention() -> Qwen3NextAttention {
        let config = Qwen3NextConfig {
            num_attention_heads: TEST_HEADS,
            num_key_value_heads: TEST_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
            quantization: None,
            mrope_section: vec![2, 1, 1],
        };
        let (hq, hkv, d) = (
            TEST_HEADS as i32,
            TEST_KV_HEADS as i32,
            TEST_HEAD_DIM as i32,
        );
        let mut weights = WeightMap::new();
        for (seed, (name, shape)) in [
            ("q_proj", [2 * hq * d, HIDDEN]),
            ("k_proj", [hkv * d, HIDDEN]),
            ("v_proj", [hkv * d, HIDDEN]),
            ("o_proj", [HIDDEN, hq * d]),
        ]
        .into_iter()
        .enumerate()
        {
            weights.insert(
                format!("attn.{name}.weight"),
                noise_array(&shape, seed as u64, 0.25),
            );
        }
        for norm in ["q_norm", "k_norm"] {
            weights.insert(
                format!("attn.{norm}.weight"),
                mlxcel_core::full_f32(&[d], 1.0, mlxcel_core::dtype::FLOAT32),
            );
        }
        Qwen3NextAttention::from_weights(&weights, &config, "attn").expect("test attention")
    }

    #[test]
    fn layer_sparse_read_matches_host_mask() {
        for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
            for backend in [
                DlsaBackend::Mlx,
                DlsaBackend::Metal,
                DlsaBackend::MetalSelect,
            ] {
                layer_sparse_read_matches_host_mask_on(backend, mode);
            }
        }
    }

    /// The sparse read of one layer equals dense attention under a mask built on the host
    /// from `select_host` lists of an independent bucket cache fed the same index keys.
    fn layer_sparse_read_matches_host_mask_on(backend: DlsaBackend, mode: KVCacheMode) {
        let attention = test_attention();
        let geometry = DlsaGeometry {
            query_heads: TEST_HEADS,
            kv_heads: TEST_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            rotary_dim: 8,
        };
        let maps = test_layer_maps(1);
        let config = DlsaConfig {
            map_dtype: DlsaMapDtype::F32,
            index: TEST_INDEX,
            prefill_mask_rows: 16,
            dense_below: 0,
            backend,
            ..DlsaConfig::new(PathBuf::new())
        };
        let mut kv = KVCache::new_with_mode(mode);
        let mut index = DlsaBucketCache::new(TEST_INDEX).unwrap();
        let mut stats = DlsaStats::default();
        let mut ref_kv = KVCache::new_with_mode(mode);
        let mut ref_index = DlsaBucketCache::new(TEST_INDEX).unwrap();
        let (mut expanded, mut partial) = (false, false);
        let mut start = 0usize;
        let schedule = [120usize, 37, 1, 1, 37, 1];
        for (step_no, &rows) in schedule.iter().enumerate() {
            let x = noise_array(&[1, rows as i32, HIDDEN], 500 + step_no as u64, 2.0);
            let got = attention.forward_dlsa(
                &x,
                &mut kv,
                None,
                DlsaLayerStep {
                    maps: &maps,
                    geometry: &geometry,
                    config: &config,
                    cache: &mut index,
                    stats: &mut stats,
                    commit: true,
                },
            );

            // The reference: host selection, host-built additive mask, dense SDPA
            let heads = attention.project_heads(&x);
            let ik = index_keys(&heads.keys, &maps.w_k_idx, &geometry).unwrap();
            ref_index.append(&to_host_f32(&ik)).unwrap();
            let iq = index_queries(&heads.queries, &maps.w_q_idx, &geometry).unwrap();
            let positions: Vec<usize> = (start..start + rows).collect();
            let sel = ref_index
                .select_host(&to_host_f32(&iq), &positions)
                .unwrap();
            ref_index.commit();
            let total = start + rows;
            let mut mask = vec![f32::NEG_INFINITY; rows * total];
            for r in 0..rows {
                let p = start + r;
                let list = sel.list(r);
                for &k in list {
                    mask[r * total + k as usize] = 0.0;
                }
                let window_start = (p + 1).saturating_sub(TEST_INDEX.window_size);
                expanded |= list
                    .iter()
                    .any(|&k| k as usize >= TEST_INDEX.sink_size && (k as usize) < window_start);
                partial |= list.len() < p + 1;
            }
            let mask = mlxcel_core::from_slice_f32(&mask, &[1, 1, rows as i32, total as i32]);
            let (q, k, v) = attention.rotate(
                &heads.queries,
                &heads.keys,
                &heads.values,
                start as i32,
                None,
                1,
                rows as i32,
            );
            let (ck, cv) = ref_kv.update_and_fetch(k, v);
            // SAFETY: the mask outlives the call.
            let out = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &ck,
                    &cv,
                    attention.scale,
                    &*mask as *const MlxArray,
                    0.0,
                    0,
                )
            };
            let want =
                attention
                    .o_proj
                    .forward(&attention.finish(&out, &heads.gate, 1, rows as i32));
            let diff = max_abs_diff(&to_host_f32(&got), &to_host_f32(&want));
            assert!(
                diff < 1e-5,
                "{backend:?} {mode} forward {step_no} ({rows} rows at {start}): sparse read vs host mask {diff:e}"
            );
            start = total;
        }
        assert!(expanded, "no list read a key outside its window and sinks");
        assert!(partial, "every list covered its whole causal context");
        assert_eq!(stats.sparse_forwards, schedule.len() as u64);
        assert_eq!(stats.dense_key_set_forwards, 0);
        assert_eq!(index.len(), start);
        assert_eq!(index.members(), ref_index.members());
        // Deferred keys read back to the same host keys
        assert_eq!(index.tail_keys(), ref_index.tail_keys(), "{backend:?}");
    }

    /// A forward that reads the dense key set under the fallback is the dense attention itself.
    #[test]
    fn dense_key_set_forward_is_bitwise_dense() {
        let attention = test_attention();
        let geometry = DlsaGeometry {
            query_heads: TEST_HEADS,
            kv_heads: TEST_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            rotary_dim: 8,
        };
        let maps = test_layer_maps(1);
        for backend in [
            DlsaBackend::Mlx,
            DlsaBackend::Metal,
            DlsaBackend::MetalSelect,
        ] {
            let config = DlsaConfig {
                map_dtype: DlsaMapDtype::F32,
                index: TEST_INDEX,
                backend,
                ..DlsaConfig::new(PathBuf::new())
            };
            let mut kv = KVCache::new_with_mode(KVCacheMode::Fp16);
            let mut dense_kv = KVCache::new_with_mode(KVCacheMode::Fp16);
            let mut index = DlsaBucketCache::new(TEST_INDEX).unwrap();
            let mut stats = DlsaStats::default();
            for (step_no, rows) in [30usize, 1, 17].into_iter().enumerate() {
                let x = noise_array(&[1, rows as i32, HIDDEN], 900 + step_no as u64, 2.0);
                let got = attention.forward_dlsa(
                    &x,
                    &mut kv,
                    None,
                    DlsaLayerStep {
                        maps: &maps,
                        geometry: &geometry,
                        config: &config,
                        cache: &mut index,
                        stats: &mut stats,
                        commit: true,
                    },
                );
                let want = attention.forward_with_position_ids(&x, &mut dense_kv, None, None);
                assert_eq!(
                    to_host_f32(&got),
                    to_host_f32(&want),
                    "{backend:?} forward {step_no}"
                );
            }
            assert_eq!(stats.dense_key_set_forwards, 3);
            assert_eq!(index.len(), 48);
        }
    }

    /// `dense_below`: forwards ending below it run the dense attention bitwise although their
    /// rows would read sparse lists; a forward that crosses it reads its rows below causally
    /// (dense up to rounding) and its other rows sparse; the index is built throughout.
    #[test]
    fn dense_below_runs_dense_until_the_threshold() {
        let attention = test_attention();
        let geometry = DlsaGeometry {
            query_heads: TEST_HEADS,
            kv_heads: TEST_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            rotary_dim: 8,
        };
        let maps = test_layer_maps(1);
        for backend in [
            DlsaBackend::Mlx,
            DlsaBackend::Metal,
            DlsaBackend::MetalSelect,
        ] {
            let config = DlsaConfig {
                map_dtype: DlsaMapDtype::F32,
                index: TEST_INDEX,
                prefill_mask_rows: 16,
                dense_below: 160,
                backend,
                ..DlsaConfig::new(PathBuf::new())
            };
            assert!(!config.reads_dense(160) && config.reads_dense(159));
            assert!(
                !crate::dlsa::reads_dense_key_set(&TEST_INDEX, 119),
                "the schedule must reach sparse depths before the threshold"
            );
            let mut kv = KVCache::new_with_mode(KVCacheMode::Fp16);
            let mut dense_kv = KVCache::new_with_mode(KVCacheMode::Fp16);
            let mut index = DlsaBucketCache::new(TEST_INDEX).unwrap();
            let mut stats = DlsaStats::default();
            let mut start = 0;
            // Ends at 119, 149, 150, 151 (dense); then 152..=195 crosses the threshold (rows
            // 152..=159 read every causal key, the rest sparse lists); then 196 (sparse)
            for (step_no, rows) in [120usize, 30, 1, 1, 44, 1].into_iter().enumerate() {
                let x = noise_array(&[1, rows as i32, HIDDEN], 700 + step_no as u64, 2.0);
                let got = attention.forward_dlsa(
                    &x,
                    &mut kv,
                    None,
                    DlsaLayerStep {
                        maps: &maps,
                        geometry: &geometry,
                        config: &config,
                        cache: &mut index,
                        stats: &mut stats,
                        commit: true,
                    },
                );
                let want = attention.forward_with_position_ids(&x, &mut dense_kv, None, None);
                let (got, want) = (to_host_f32(&got), to_host_f32(&want));
                let below = config.dense_rows(start, rows);
                let split = below * HIDDEN as usize;
                if below == rows {
                    assert_eq!(got, want, "{backend:?} forward {step_no}: dense below");
                } else {
                    // The causal read of the rows below equals dense up to rounding
                    let diff = max_abs_diff(&got[..split], &want[..split]);
                    assert!(
                        diff < 1e-5,
                        "{backend:?} forward {step_no}: rows below {diff:e}"
                    );
                    assert_ne!(
                        got[split..],
                        want[split..],
                        "{backend:?} forward {step_no}: sparse past it"
                    );
                }
                start += rows;
            }
            assert_eq!(config.dense_rows(152, 44), 8);
            assert_eq!(stats.dense_key_set_forwards, 4, "{backend:?}");
            assert_eq!(stats.sparse_forwards, 2, "{backend:?}");
            assert_eq!(stats.sparse_rows, 36 + 1, "{backend:?}");
            assert_eq!(index.len(), start);
        }
    }
}
