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
    DlsaBucketCache, DlsaConfig, DlsaGeometry, DlsaLayerMaps, DlsaShortContext, DlsaStats,
    gather_attention, index_keys, index_queries, masked_attention, reads_dense_key_set,
};

/// What one layer's D-LSA forward reads and updates besides its KV cache.
pub(crate) struct DlsaLayerStep<'a> {
    pub maps: &'a DlsaLayerMaps,
    pub geometry: &'a DlsaGeometry,
    pub config: &'a DlsaConfig,
    pub cache: &'a mut DlsaBucketCache,
    pub stats: &'a mut DlsaStats,
}

impl Qwen3NextAttention {
    /// [`forward_with_position_ids`](Self::forward_with_position_ids) under D-LSA (no mask: the
    /// read is causal by construction), including `o_proj`.
    ///
    /// Panics on a broken invariant: a batch other than 1, an index cache out of step with the
    /// KV cache, a KV cache whose slots are not positions, or a non-finite index key.
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
        let rows = l as usize;

        let started = Instant::now();
        let ik = index_keys(&heads.keys, &step.maps.w_k_idx, step.geometry)
            .expect("D-LSA index keys of a full-attention layer");
        step.cache
            .append_mlx(&ik)
            .expect("D-LSA bucket cache append");
        step.stats.index_time += started.elapsed();

        let index = &step.config.index;
        let dense = step.config.short_context == DlsaShortContext::DenseFallback
            && reads_dense_key_set(index, start + rows - 1);
        let selection = (!dense).then(|| {
            let started = Instant::now();
            let iq = index_queries(&heads.queries, &step.maps.w_q_idx, step.geometry)
                .expect("D-LSA index queries of a full-attention layer");
            let positions: Vec<usize> = (start..start + rows).collect();
            let sel = step.cache.select(&iq, &positions).expect("D-LSA selection");
            step.stats.select_time += started.elapsed();
            sel
        });
        // No speculation in this path: every forward is accepted
        step.cache.commit();

        let (queries, keys, values) = self.rotate(
            &heads.queries,
            &heads.keys,
            &heads.values,
            offset,
            position_ids,
            b,
            l,
        );
        let attn_out = match selection {
            None => {
                step.stats.dense_key_set_forwards += 1;
                self.dense_attention(&queries, keys, values, cache, None, l)
            }
            Some(sel) => {
                step.stats.sparse_forwards += 1;
                step.stats.sparse_rows += rows as u64;
                step.stats.keys_read += sel.lengths.iter().map(|&n| n as u64).sum::<u64>();
                let (cache_k, cache_v) = cache.update_and_fetch(keys, values);
                if rows == 1 {
                    gather_attention(&queries, &cache_k, &cache_v, sel.list(0), self.scale)
                } else {
                    masked_attention(
                        &queries,
                        &cache_k,
                        &cache_v,
                        &sel,
                        start,
                        step.config.prefill_mask_rows,
                        self.scale,
                    )
                }
            }
        };
        let output = self.finish(&attn_out, &heads.gate, b, l);
        self.o_proj.forward(&output)
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

    /// The sparse read of one layer equals dense attention under a mask built on the host
    /// from `select_host` lists of an independent bucket cache fed the same index keys.
    #[test]
    fn layer_sparse_read_matches_host_mask() {
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
            ..DlsaConfig::new(PathBuf::new())
        };
        let mut kv = KVCache::new_with_mode(KVCacheMode::Fp16);
        let mut index = DlsaBucketCache::new(TEST_INDEX).unwrap();
        let mut stats = DlsaStats::default();
        let mut ref_kv = KVCache::new_with_mode(KVCacheMode::Fp16);
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
                "forward {step_no} ({rows} rows at {start}): sparse read vs host mask {diff:e}"
            );
            start = total;
        }
        assert!(expanded, "no list read a key outside its window and sinks");
        assert!(partial, "every list covered its whole causal context");
        assert_eq!(stats.sparse_forwards, schedule.len() as u64);
        assert_eq!(stats.dense_key_set_forwards, 0);
        assert_eq!(index.len(), start);
        assert_eq!(index.members(), ref_index.members());
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
        let config = DlsaConfig {
            map_dtype: DlsaMapDtype::F32,
            index: TEST_INDEX,
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
                },
            );
            let want = attention.forward_with_position_ids(&x, &mut dense_kv, None, None);
            assert_eq!(to_host_f32(&got), to_host_f32(&want), "forward {step_no}");
        }
        assert_eq!(stats.dense_key_set_forwards, 3);
        assert_eq!(index.len(), 48);
    }
}
