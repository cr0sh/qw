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

use super::{Qwen35AttentionVariant, Qwen35DecoderLayer, Qwen35Model};
use crate::dlsa::{DlsaConfig, DlsaRuntime, DlsaStats};
use crate::qwen3_next::{DlsaLayerStep, Qwen3NextAttention, Qwen3NextCache};

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

    fn dlsa_refusal_message(what: &str) -> String {
        format!(
            "{what} does not support D-LSA yet (D-LSA task 5): its multi-row verification, rollback \
             and position-remapped caches would leave the index out of step; disable D-LSA"
        )
    }

    /// `Err` when D-LSA is on, for entry points that return a `Result`.
    pub(crate) fn dlsa_refusal(&self, what: &str) -> std::result::Result<(), String> {
        if self.dlsa.is_some() {
            return Err(Self::dlsa_refusal_message(what));
        }
        Ok(())
    }

    /// Panic when D-LSA is on, for entry points that cannot return an error (the provider
    /// refuses these modes before reaching them).
    pub(crate) fn refuse_dlsa(&self, what: &str) {
        assert!(self.dlsa.is_none(), "{}", Self::dlsa_refusal_message(what));
    }

    /// The decoder stack under D-LSA: full-attention layers through
    /// [`Qwen3NextAttention::forward_dlsa`], linear layers unchanged.
    pub(super) fn forward_backbone_dlsa(
        &self,
        dlsa: &DlsaRuntime,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen3NextCache],
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut hidden = input_embeddings
            .map(mlxcel_core::copy)
            .unwrap_or_else(|| self.embed_tokens.forward(input_ids));
        let mut state = dlsa.state.borrow_mut();
        let (index_caches, stats) = &mut *state;
        for (layer_idx, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            hidden = match (&layer.attention, cache, dlsa.layer_ordinal.get(&layer_idx)) {
                (
                    Qwen35AttentionVariant::FullAttention(attention),
                    Qwen3NextCache::Attention(kv),
                    Some(&ordinal),
                ) => {
                    let step = DlsaLayerStep {
                        maps: dlsa
                            .maps
                            .layer(layer_idx)
                            .expect("D-LSA maps were checked against the layers"),
                        geometry: &dlsa.geometry,
                        config: &dlsa.config,
                        cache: &mut index_caches[ordinal],
                        stats: &mut *stats,
                    };
                    layer.forward_dlsa(attention, &hidden, kv, position_ids, step)
                }
                (_, cache, _) => layer.forward(&hidden, None, cache, position_ids),
            };
        }
        hidden
    }
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
