// SPDX-License-Identifier: Apache-2.0
//
//! The model-level D-LSA switch of Qwen3.5 and its backbone forward (`docs/dlsa.md` §7, §12).
//! D-LSA is a property of the loaded model and its one sequence: enabling or disabling it resets
//! the sequence, and its index caches follow the KV caches (reset, one key per position).

use anyhow::{Result, bail};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{
    Qwen35AttentionVariant, Qwen35DecoderLayer, Qwen35Model, push_snapshot_i32, snapshot_i32,
};
use crate::dlsa::{
    DlsaBucketCache, DlsaConfig, DlsaIndexConfig, DlsaIndexSnapshot, DlsaRuntime, DlsaStats,
};
use crate::qwen3_next::{DlsaLayerStep, Qwen3NextAttention, Qwen3NextCache};
use mlxcel_core::generate::ModelStateSnapshot;

impl Qwen35Model {
    /// Turn D-LSA on for this model: load the map, check it against the checkpoint, and start
    /// a fresh sequence. Only Fp16 and Int8 KV caches are supported.
    pub(crate) fn enable_dlsa(&mut self, config: DlsaConfig) -> Result<()> {
        self.check_dlsa_cache_mode()?;
        config.validate()?;
        let runtime = DlsaRuntime::load(config, &self.config)?;
        self.install_dlsa(runtime);
        Ok(())
    }

    /// [`enable_dlsa`](Self::enable_dlsa) with maps already in memory.
    #[cfg(test)]
    pub(crate) fn enable_dlsa_with_maps(
        &mut self,
        config: DlsaConfig,
        maps: crate::dlsa::DlsaMaps,
    ) -> Result<()> {
        self.check_dlsa_cache_mode()?;
        let runtime = DlsaRuntime::from_maps(config, maps, &self.config)?;
        self.install_dlsa(runtime);
        Ok(())
    }

    fn check_dlsa_cache_mode(&self) -> Result<()> {
        match self.kv_cache_mode {
            KVCacheMode::Fp16 | KVCacheMode::Int8 => Ok(()),
            mode => bail!(
                "D-LSA needs an fp16 or int8 KV cache, but the model was loaded with kv-cache-mode {mode}: \
                 sparse reads of packed Turbo caches are not implemented yet (D-LSA tasks 4/5). \
                 Load the model with KVCacheMode::Fp16 (`qw generate` does)"
            ),
        }
    }

    fn install_dlsa(&mut self, runtime: DlsaRuntime) {
        tracing::info!(
            map = %runtime.config.map_path.display(),
            map_dtype = ?runtime.config.map_dtype,
            index = ?runtime.config.index,
            short_context = ?runtime.config.short_context,
            backend = runtime.config.backend.name(),
            layers = runtime.layer_ordinal.len(),
            "D-LSA enabled"
        );
        self.dlsa = Some(runtime);
        self.reset_runtime_state();
    }

    /// Back to dense attention, with a fresh sequence.
    pub(crate) fn disable_dlsa(&mut self) {
        if self.dlsa.take().is_some() {
            self.reset_runtime_state();
        }
    }

    pub(crate) fn dlsa_enabled(&self) -> bool {
        self.dlsa.is_some()
    }

    pub(crate) fn dlsa_stats(&self) -> Option<DlsaStats> {
        self.dlsa.as_ref().map(DlsaRuntime::stats)
    }

    /// Bytes the D-LSA bucket caches of the current sequence hold.
    pub(crate) fn dlsa_memory_bytes(&self) -> usize {
        self.dlsa.as_ref().map_or(0, DlsaRuntime::memory_bytes)
    }

    /// `Err` when D-LSA is on, for entry points whose positions are not token positions (image
    /// MRoPE prompts): the index would be built from the wrong keys.
    pub(crate) fn dlsa_refusal(&self, what: &str) -> std::result::Result<(), String> {
        if self.dlsa.is_some() {
            return Err(format!(
                "{what} is not supported under D-LSA: its cache slots or positions are not the \
                 prompt's token positions, so the sparse index would be built from the wrong keys; \
                 disable D-LSA"
            ));
        }
        Ok(())
    }

    /// The decoder stack under D-LSA: full-attention layers through
    /// [`Qwen3NextAttention::forward_dlsa`] (committing the index), linear layers unchanged.
    pub(super) fn forward_backbone_dlsa(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen3NextCache],
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut hidden = input_embeddings
            .map(mlxcel_core::copy)
            .unwrap_or_else(|| self.embed_tokens.forward(input_ids));
        for (layer_idx, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            hidden = match self.forward_layer_dlsa(layer_idx, &hidden, cache, position_ids, true) {
                Some(hidden) => hidden,
                None => layer.forward(&hidden, None, cache, position_ids),
            };
        }
        hidden
    }

    /// One decoder layer under D-LSA: `Some` for a full-attention layer while D-LSA is on,
    /// through [`Qwen3NextAttention::forward_dlsa`]; `None` otherwise, and the caller runs its
    /// own dense code for the layer. The verify loops pass `commit = false`, so
    /// [`rollback_dlsa`](Self::rollback_dlsa) can reopen a block the verify closed.
    pub(super) fn forward_layer_dlsa(
        &self,
        layer_idx: usize,
        hidden: &MlxArray,
        cache: &mut Qwen3NextCache,
        position_ids: Option<&MlxArray>,
        commit: bool,
    ) -> Option<UniquePtr<MlxArray>> {
        let dlsa = self.dlsa.as_ref()?;
        let &ordinal = dlsa.layer_ordinal.get(&layer_idx)?;
        let layer = &self.layers[layer_idx];
        let (Qwen35AttentionVariant::FullAttention(attention), Qwen3NextCache::Attention(kv)) =
            (&layer.attention, cache)
        else {
            return None;
        };
        let mut state = dlsa.state.borrow_mut();
        let (index_caches, stats) = &mut *state;
        let step = DlsaLayerStep {
            maps: dlsa
                .maps
                .layer(layer_idx)
                .expect("D-LSA maps were checked against the layers"),
            geometry: &dlsa.geometry,
            config: &dlsa.config,
            cache: &mut index_caches[ordinal],
            stats,
            commit,
        };
        Some(layer.forward_dlsa(attention, hidden, kv, position_ids, step))
    }

    /// Roll every index back with its KV cache after a verify (`rollback_mtp_verify`): to
    /// `final_offset` keys, reopening a block the rejected rows had closed. Panics when an index
    /// and its KV cache end out of step.
    pub(super) fn rollback_dlsa(&self, caches: &[Qwen3NextCache], final_offset: i32) {
        let Some(dlsa) = &self.dlsa else {
            return;
        };
        let keep = usize::try_from(final_offset).expect("a rollback offset is non-negative");
        let mut state = dlsa.state.borrow_mut();
        let (index_caches, stats) = &mut *state;
        let (mut trimmed, mut reopened) = (false, false);
        for (&layer, &ordinal) in &dlsa.layer_ordinal {
            let index = &mut index_caches[ordinal];
            let (len, rows) = (index.len(), index.rows());
            index
                .trim(keep)
                .unwrap_or_else(|e| panic!("D-LSA rollback of layer {layer} to {keep}: {e:#}"));
            trimmed |= index.len() < len;
            reopened |= index.rows() < rows;
            let Some(Qwen3NextCache::Attention(kv)) = caches.get(layer) else {
                panic!("D-LSA layer {layer} has no attention cache");
            };
            assert_eq!(
                index.len(),
                kv.offset as usize,
                "D-LSA layer {layer}: the index and the KV cache are out of step after a rollback"
            );
        }
        stats.rollbacks += u64::from(trimmed);
        stats.reopened_blocks += u64::from(reopened);
    }

    /// Append the index state of every full-attention layer to a snapshot of `token_len`
    /// tokens, as dense tensors (never paged: the bucket rows are per block, not per token, and
    /// the codecs reject zero-sized dims, so the tables are present only when non-empty):
    ///
    /// - `layer.{i}.dlsa.len` i32 `[1]`, always;
    /// - `layer.{i}.dlsa.centroids` f32 `[rows, index_dim]`, `.members` i32 `[rows, cap]` and
    ///   `.sizes` i32 `[rows]` when `rows > 0`;
    /// - `layer.{i}.dlsa.open_keys` f32 `[open, index_dim]` when the open block holds keys;
    /// - `meta.dlsa.index` i32 `[8]`: the index config ([`index_meta`]).
    ///
    /// `false` when an index is out of step with `token_len` (capture then gives `None`).
    /// Nothing is pushed while D-LSA is off.
    pub(super) fn push_dlsa_snapshot(
        &self,
        snapshot: &mut ModelStateSnapshot,
        token_len: usize,
    ) -> bool {
        let Some(dlsa) = &self.dlsa else {
            return true;
        };
        let Ok(len) = i32::try_from(token_len) else {
            return false;
        };
        let cfg = &dlsa.config.index;
        let (d, cap) = (cfg.index_dim as i32, cfg.bucket_cap as i32);
        let state = dlsa.state.borrow();
        for (&layer, &ordinal) in &dlsa.layer_ordinal {
            let index = &state.0[ordinal];
            if index.len() != token_len {
                return false;
            }
            let snap = index.snapshot();
            push_snapshot_i32(snapshot, &dlsa_name(layer, "len"), len);
            let rows = snap.sizes.len() as i32;
            if rows > 0 {
                snapshot.push_tensor(
                    dlsa_name(layer, "centroids"),
                    &mlxcel_core::from_slice_f32(&snap.centroids, &[rows, d]),
                );
                snapshot.push_tensor(
                    dlsa_name(layer, "members"),
                    &mlxcel_core::from_slice_i32(&snap.members, &[rows, cap]),
                );
                snapshot.push_tensor(
                    dlsa_name(layer, "sizes"),
                    &mlxcel_core::from_slice_i32(&snap.sizes, &[rows]),
                );
            }
            let open = (snap.open_keys.len() / cfg.index_dim) as i32;
            if open > 0 {
                snapshot.push_tensor(
                    dlsa_name(layer, "open_keys"),
                    &mlxcel_core::from_slice_f32(&snap.open_keys, &[open, d]),
                );
            }
        }
        snapshot.push_tensor(
            DLSA_INDEX_META,
            &mlxcel_core::from_slice_i32(&index_meta(cfg), &[INDEX_META_LEN as i32]),
        );
        true
    }

    /// The D-LSA tensor names a snapshot of `token_len` tokens must hold on this model (none
    /// while D-LSA is off), for the exact name check of `restore_sequence_state`. Errs at once
    /// when the snapshot's D-LSA state does not match whether D-LSA is on.
    pub(super) fn dlsa_snapshot_names(
        &self,
        snapshot: &ModelStateSnapshot,
        token_len: usize,
    ) -> std::result::Result<Vec<String>, String> {
        let has_meta = snapshot.tensor(DLSA_INDEX_META).is_some();
        let Some(dlsa) = &self.dlsa else {
            if has_meta {
                return Err(
                    "the Qwen3.5 snapshot carries D-LSA index state, but D-LSA is off on this model"
                        .to_string(),
                );
            }
            return Ok(Vec::new());
        };
        if !has_meta {
            return Err(
                "the Qwen3.5 snapshot carries no D-LSA index state (taken with D-LSA off), but \
                 D-LSA is on on this model"
                    .to_string(),
            );
        }
        let cfg = &dlsa.config.index;
        let closed = token_len / cfg.window_size;
        let open = token_len - closed * cfg.window_size;
        let mut names = vec![DLSA_INDEX_META.to_string()];
        for &layer in dlsa.layer_ordinal.keys() {
            names.push(dlsa_name(layer, "len"));
            if closed > 0 {
                for part in ["centroids", "members", "sizes"] {
                    names.push(dlsa_name(layer, part));
                }
            }
            if open > 0 {
                names.push(dlsa_name(layer, "open_keys"));
            }
        }
        Ok(names)
    }

    /// Rebuild every layer's index from a snapshot (all or nothing), checked against the
    /// model's index config and the snapshot's length. `None` while D-LSA is off. Install the
    /// result with [`install_dlsa_caches`](Self::install_dlsa_caches) once the KV restore
    /// succeeded.
    pub(super) fn restore_dlsa_caches(
        &self,
        snapshot: &ModelStateSnapshot,
    ) -> std::result::Result<Option<Vec<DlsaBucketCache>>, String> {
        let Some(dlsa) = &self.dlsa else {
            return Ok(None);
        };
        let cfg = dlsa.config.index;
        let meta = host_i32(snapshot, DLSA_INDEX_META, &[INDEX_META_LEN as i32])?;
        if meta != index_meta(&cfg) {
            return Err(format!(
                "the Qwen3.5 snapshot's D-LSA index was built under another index config \
                 (window, sink, top_k, top_p, pool, bucket, cap, index_dim = {meta:?}), not {:?}",
                index_meta(&cfg)
            ));
        }
        let token_len = snapshot.token_len();
        let (d, cap) = (cfg.index_dim, cfg.bucket_cap);
        let closed = token_len / cfg.window_size;
        let rows = closed * cfg.buckets_per_block();
        let open = token_len - closed * cfg.window_size;
        let mut caches: Vec<Option<DlsaBucketCache>> =
            (0..dlsa.layer_ordinal.len()).map(|_| None).collect();
        for (&layer, &ordinal) in &dlsa.layer_ordinal {
            let len = snapshot_i32(snapshot, &dlsa_name(layer, "len"))?;
            if usize::try_from(len).ok() != Some(token_len) {
                return Err(format!(
                    "Qwen3.5 snapshot layer {layer}: the D-LSA index holds {len} keys, the \
                     snapshot {token_len} tokens"
                ));
            }
            let (centroids, members, sizes) = if rows > 0 {
                let shape = |w: usize| [rows as i32, w as i32];
                (
                    host_f32(snapshot, &dlsa_name(layer, "centroids"), &shape(d))?,
                    host_i32(snapshot, &dlsa_name(layer, "members"), &shape(cap))?,
                    host_i32(snapshot, &dlsa_name(layer, "sizes"), &[rows as i32])?,
                )
            } else {
                Default::default()
            };
            let open_keys = if open > 0 {
                host_f32(
                    snapshot,
                    &dlsa_name(layer, "open_keys"),
                    &[open as i32, d as i32],
                )?
            } else {
                Vec::new()
            };
            let snap = DlsaIndexSnapshot {
                cfg,
                len: token_len,
                centroids,
                members,
                sizes,
                open_keys,
            };
            let cache = DlsaBucketCache::restore(cfg, &snap).map_err(|e| {
                format!("Qwen3.5 snapshot layer {layer}: invalid D-LSA index: {e:#}")
            })?;
            caches[ordinal] = Some(cache);
        }
        Ok(Some(
            caches
                .into_iter()
                .map(|cache| cache.expect("every D-LSA layer restored"))
                .collect(),
        ))
    }

    /// Install the indexes [`restore_dlsa_caches`](Self::restore_dlsa_caches) rebuilt. The
    /// counters are kept: they cover the model's forwards, not the sequence.
    pub(super) fn install_dlsa_caches(&self, caches: Option<Vec<DlsaBucketCache>>) {
        if let (Some(dlsa), Some(caches)) = (&self.dlsa, caches) {
            let mut state = dlsa.state.borrow_mut();
            let (old, stats) = &mut *state;
            // Keep the replaced indexes' lazy key counts (one sync, at a restore)
            for cache in old.iter_mut() {
                stats.keys_read += cache.take_keys_read();
            }
            *old = caches;
        }
    }
}

/// The config-identity tensor of a D-LSA snapshot.
const DLSA_INDEX_META: &str = "meta.dlsa.index";
const INDEX_META_LEN: usize = 8;

fn dlsa_name(layer: usize, part: &str) -> String {
    format!("layer.{layer}.dlsa.{part}")
}

/// window, sink, top_k, top_p, pool, bucket, cap and index_dim, as stored in `meta.dlsa.index`.
fn index_meta(cfg: &DlsaIndexConfig) -> [i32; INDEX_META_LEN] {
    [
        cfg.window_size,
        cfg.sink_size,
        cfg.top_k,
        cfg.top_p,
        cfg.pool_size,
        cfg.bucket_size,
        cfg.bucket_cap,
        cfg.index_dim,
    ]
    .map(|v| i32::try_from(v).unwrap_or(i32::MAX))
}

/// The host bytes of a dense snapshot tensor of exactly `shape` and `dtype`.
fn host_bytes(
    snapshot: &ModelStateSnapshot,
    name: &str,
    shape: &[i32],
    dtype: i32,
) -> std::result::Result<Vec<u8>, String> {
    let tensor = snapshot
        .tensor(name)
        .ok_or_else(|| format!("Qwen3.5 snapshot is missing {name}"))?;
    if mlxcel_core::array_shape(tensor) != shape || mlxcel_core::array_dtype(tensor) != dtype {
        return Err(format!(
            "Qwen3.5 snapshot tensor {name} has an invalid layout: {:?} of dtype {}, expected \
             {shape:?} of dtype {dtype}",
            mlxcel_core::array_shape(tensor),
            mlxcel_core::array_dtype(tensor)
        ));
    }
    mlxcel_core::eval(tensor);
    Ok(mlxcel_core::array_to_raw_bytes(tensor))
}

fn host_f32(
    snapshot: &ModelStateSnapshot,
    name: &str,
    shape: &[i32],
) -> std::result::Result<Vec<f32>, String> {
    let bytes = host_bytes(snapshot, name, shape, mlxcel_core::dtype::FLOAT32)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect())
}

fn host_i32(
    snapshot: &ModelStateSnapshot,
    name: &str,
    shape: &[i32],
) -> std::result::Result<Vec<i32>, String> {
    let bytes = host_bytes(snapshot, name, shape, mlxcel_core::dtype::INT32)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect())
}

impl Qwen35DecoderLayer {
    /// [`forward`](Self::forward) of a full-attention layer under D-LSA.
    fn forward_dlsa(
        &self,
        attention: &Qwen3NextAttention,
        x: &MlxArray,
        cache: &mut KVCache,
        position_ids: Option<&MlxArray>,
        step: DlsaLayerStep<'_>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let residual = attention.forward_dlsa(&normed, cache, position_ids, step);
        let hidden = mlxcel_core::add(x, &residual);
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden));
        mlxcel_core::add(&hidden, &mlp)
    }
}
