// SPDX-License-Identifier: Apache-2.0
//
//! The model-level D-LSA switch: its configuration, the per-layer bucket caches of the one
//! sequence a Qwen3.5 model holds, and counters (`docs/dlsa.md` §7, §12).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, ensure};

use super::INDEX_DIM;
use super::cache::DlsaBucketCache;
use super::config::{DlsaGeometry, DlsaIndexConfig};
use super::loader::{DlsaLayout, DlsaMapDtype, DlsaMaps};
use super::resolve::{DlsaMapKind, resolve_dlsa_map_path};
use crate::qwen3_5::Qwen35Config;

/// What a full-attention forward whose rows all read the dense key set (`docs/dlsa.md` §4.7) runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DlsaShortContext {
    /// The unchanged dense attention: bit-identical to D-LSA off. The key set is the same.
    #[default]
    DenseFallback,
    /// The sparse read even then (tests and parity runs only).
    Sparse,
}

/// How D-LSA runs on a loaded model.
#[derive(Debug, Clone, PartialEq)]
pub struct DlsaConfig {
    /// A converted map file (`qw download --dlsa-map`).
    pub map_path: PathBuf,
    /// Bf16 is what upstream serves (§4.1); F32 reproduces the reference.
    pub map_dtype: DlsaMapDtype,
    pub index: DlsaIndexConfig,
    pub short_context: DlsaShortContext,
    /// Query rows one additive read mask covers in a prefill chunk (`rows x T` mask bytes).
    pub prefill_mask_rows: usize,
}

impl DlsaConfig {
    /// Defaults: bf16 maps, the Qwen indexer defaults, the dense fallback, 256-row masks.
    pub fn new(map_path: PathBuf) -> Self {
        Self {
            map_path,
            map_dtype: DlsaMapDtype::Bf16,
            index: DlsaIndexConfig::QWEN_DEFAULTS,
            short_context: DlsaShortContext::DenseFallback,
            prefill_mask_rows: 256,
        }
    }

    /// Defaults on the map that `resolve_dlsa_map_path` finds (CLI override, env, cache).
    pub fn resolve(kind: DlsaMapKind, map_override: Option<&Path>) -> Result<Self> {
        Ok(Self::new(resolve_dlsa_map_path(kind, map_override)?))
    }

    pub fn validate(&self) -> Result<()> {
        self.index.validate()?;
        ensure!(
            self.index.index_dim == INDEX_DIM,
            "index_dim must be {INDEX_DIM}, the width of the released maps, got {}",
            self.index.index_dim
        );
        ensure!(
            self.prefill_mask_rows > 0,
            "prefill_mask_rows must be at least 1"
        );
        Ok(())
    }
}

/// Counters over the full-attention layer forwards since D-LSA was enabled or reset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DlsaStats {
    /// Layer forwards that read the dense key set through the unchanged dense attention.
    pub dense_key_set_forwards: u64,
    /// Layer forwards through the sparse read.
    pub sparse_forwards: u64,
    /// Query rows of the sparse layer forwards.
    pub sparse_rows: u64,
    /// Keys those rows read (sum of list lengths).
    pub keys_read: u64,
    /// Index key/query projection and the bucket cache append (incl. Ward), host-synced.
    pub index_time: Duration,
    /// Scoring, top-p pick and expansion.
    pub select_time: Duration,
    /// Verify rollbacks that dropped index keys (model-level events, not per layer).
    pub rollbacks: u64,
    /// Of those, rollbacks that reopened a closed cluster block (its keys rejected).
    pub reopened_blocks: u64,
}

/// D-LSA state of one loaded model: maps, and one bucket cache per full-attention layer for the
/// model's one sequence (the same lifecycle as its KV caches).
pub(crate) struct DlsaRuntime {
    pub(crate) config: DlsaConfig,
    pub(crate) geometry: DlsaGeometry,
    pub(crate) maps: DlsaMaps,
    /// Decoder layer index -> slot in `state.0`.
    pub(crate) layer_ordinal: BTreeMap<usize, usize>,
    pub(crate) state: RefCell<(Vec<DlsaBucketCache>, DlsaStats)>,
}

/// The attention geometry of a Qwen3.5 checkpoint's full-attention layers.
pub(crate) fn qwen35_geometry(model: &Qwen35Config) -> DlsaGeometry {
    let n = model.to_qwen3next_config();
    DlsaGeometry {
        query_heads: n.num_attention_heads,
        kv_heads: n.num_key_value_heads,
        head_dim: n.head_dim,
        rotary_dim: n.rope_dims() as usize,
    }
}

impl DlsaRuntime {
    /// A runtime over maps already in memory (validated against `model`).
    pub(crate) fn from_maps(
        config: DlsaConfig,
        maps: DlsaMaps,
        model: &Qwen35Config,
    ) -> Result<Self> {
        config.validate()?;
        let geometry = qwen35_geometry(model);
        geometry.validate()?;
        ensure!(
            geometry.input_dim() == maps.input_dim,
            "the map reads {} position-free key dims, the checkpoint has {} ({} kv heads x {})",
            maps.input_dim,
            geometry.input_dim(),
            geometry.kv_heads,
            geometry.free_dim()
        );
        let layout = DlsaLayout::from_qwen35_config(model)?;
        let mut layer_ordinal = BTreeMap::new();
        for (ordinal, &layer) in layout.full_attention_layers.iter().enumerate() {
            let maps = maps.layer(layer);
            ensure!(
                maps.is_some(),
                "the map has no projections for full-attention layer {layer}"
            );
            layer_ordinal.insert(layer, ordinal);
        }
        let caches = (0..layer_ordinal.len())
            .map(|_| DlsaBucketCache::new(config.index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            geometry,
            maps,
            layer_ordinal,
            state: RefCell::new((caches, DlsaStats::default())),
        })
    }

    /// Load `config.map_path` for `model` and build the runtime.
    pub(crate) fn load(config: DlsaConfig, model: &Qwen35Config) -> Result<Self> {
        config.validate()?;
        let layout = DlsaLayout::from_qwen35_config(model)?;
        let maps = DlsaMaps::load(&config.map_path, &layout, config.map_dtype)?;
        Self::from_maps(config, maps, model)
    }

    /// Back to an empty sequence: every bucket cache emptied, counters zeroed.
    pub(crate) fn reset(&self) {
        let mut state = self.state.borrow_mut();
        for cache in state.0.iter_mut() {
            cache.reset();
        }
        state.1 = DlsaStats::default();
    }

    pub(crate) fn stats(&self) -> DlsaStats {
        self.state.borrow().1
    }

    /// Bytes the bucket caches hold (host allocations plus device centroids); maps excluded.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.state
            .borrow()
            .0
            .iter()
            .map(DlsaBucketCache::nbytes)
            .sum()
    }

    /// Keys each layer's bucket cache holds, by layer ordinal.
    #[cfg(test)]
    pub(crate) fn cache_lens(&self) -> Vec<usize> {
        self.state
            .borrow()
            .0
            .iter()
            .map(DlsaBucketCache::len)
            .collect()
    }
}
