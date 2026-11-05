// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/sglang/settings/design.py (IndexerConfig defaults and validation).
//
//! Flat-design indexer knobs and the attention geometry the index reads (`docs/dlsa.md` §3, §11).

use anyhow::{Result, ensure};

/// Upstream's top-k engine limit on `top_p` (`design.py`, `TOPK_LIMIT`).
pub const TOP_P_LIMIT: usize = 2048;

/// The knobs of the flat design (`top_level` 0). The cluster block size is `window_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlsaIndexConfig {
    /// Recent keys always read, and the keys a cluster block holds.
    pub window_size: usize,
    /// First keys always read.
    pub sink_size: usize,
    /// Long-range keys read besides the sink and the window.
    pub top_k: usize,
    /// Buckets a query picks for expansion.
    pub top_p: usize,
    /// Width of index keys and queries.
    pub index_dim: usize,
    /// Consecutive keys clustered as one pool (never split across buckets).
    pub pool_size: usize,
    /// Mean keys a bucket; `window_size / bucket_size` buckets a block.
    pub bucket_size: usize,
    /// Most keys a bucket holds; the row width of the member table.
    pub bucket_cap: usize,
}

impl DlsaIndexConfig {
    /// What upstream serves for Qwen3.8-27B (`IndexerConfig()` defaults, `docs/dlsa.md` §3).
    pub const QWEN_DEFAULTS: Self = Self {
        window_size: 512,
        sink_size: 4,
        top_k: 4096,
        top_p: 1024,
        index_dim: 64,
        pool_size: 4,
        bucket_size: 8,
        bucket_cap: 16,
    };

    /// The checks of `IndexerConfig.__post_init__` that apply to the flat design (`design.py:257-275`).
    pub fn validate(&self) -> Result<()> {
        for (name, value, least) in [
            ("window_size", self.window_size, 1),
            ("top_k", self.top_k, 1),
            ("top_p", self.top_p, 1),
            ("index_dim", self.index_dim, 1),
            ("pool_size", self.pool_size, 1),
            ("bucket_size", self.bucket_size, 1),
            ("bucket_cap", self.bucket_cap, 1),
        ] {
            ensure!(
                value >= least,
                "{name} must be at least {least}, got {value}"
            );
        }
        ensure!(
            self.bucket_cap >= 2 * self.bucket_size,
            "bucket_cap must be at least twice bucket_size, or Ward's merge can stop short: got bucket_cap {} against bucket_size {}",
            self.bucket_cap,
            self.bucket_size
        );
        for (name, value) in [
            ("window_size", self.window_size),
            ("bucket_size", self.bucket_size),
            ("bucket_cap", self.bucket_cap),
        ] {
            ensure!(
                value.is_multiple_of(self.pool_size),
                "pool_size must divide {name}, since a pool is never split: {} does not divide {value}",
                self.pool_size
            );
        }
        ensure!(
            self.window_size.is_multiple_of(self.bucket_size),
            "bucket_size must divide window_size, the cluster block: {} does not divide {}",
            self.bucket_size,
            self.window_size
        );
        ensure!(
            self.top_p <= TOP_P_LIMIT,
            "top_p is at most {TOP_P_LIMIT}, got {}",
            self.top_p
        );
        ensure!(
            self.top_p * self.bucket_cap >= self.top_k,
            "top_p buckets of at most bucket_cap keys must be able to fill top_k: {} x {} < {}",
            self.top_p,
            self.bucket_cap,
            self.top_k
        );
        // Member ids and list entries are i32
        ensure!(
            self.list_width() <= i32::MAX as usize,
            "list width overflows i32"
        );
        Ok(())
    }

    /// Buckets one closed cluster block yields: `m = window_size / bucket_size`.
    pub fn buckets_per_block(&self) -> usize {
        self.window_size / self.bucket_size
    }

    /// Pools of one cluster block: `window_size / pool_size`.
    pub fn pools_per_block(&self) -> usize {
        self.window_size / self.pool_size
    }

    /// Entries of one read list: `top_k + window_size + sink_size` (4612 for Qwen).
    pub fn list_width(&self) -> usize {
        self.top_k + self.window_size + self.sink_size
    }

    /// Buckets a query at `position` may score: those of the blocks closed before its own block
    /// (`visible(p) = floor(p / W) * m`, `docs/dlsa.md` §4.4).
    pub fn visible_buckets(&self, position: usize) -> usize {
        position / self.window_size * self.buckets_per_block()
    }
}

/// The attention geometry of one full-attention layer, as the index reads it. The rotated dims
/// are a prefix of each head (`rotary_end: prefix`); the index reads dims `rotary_dim..head_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlsaGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

impl DlsaGeometry {
    /// Qwen3.8-27B full-attention layers (`docs/dlsa.md` §1).
    pub const QWEN38_27B: Self = Self {
        query_heads: 24,
        kv_heads: 4,
        head_dim: 256,
        rotary_dim: 64,
    };

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.kv_heads > 0 && self.query_heads.is_multiple_of(self.kv_heads),
            "query_heads {} must be a multiple of kv_heads {}",
            self.query_heads,
            self.kv_heads
        );
        ensure!(
            self.rotary_dim < self.head_dim,
            "rotary_dim {} leaves no position-free dims of head_dim {}",
            self.rotary_dim,
            self.head_dim
        );
        Ok(())
    }

    /// Query heads a KV head serves (G = 6 for Qwen).
    pub fn group(&self) -> usize {
        self.query_heads / self.kv_heads
    }

    /// Position-free dims of one head (192 for Qwen).
    pub fn free_dim(&self) -> usize {
        self.head_dim - self.rotary_dim
    }

    /// Rows of the index maps: `kv_heads * free_dim` (768 for Qwen).
    pub fn input_dim(&self) -> usize {
        self.kv_heads * self.free_dim()
    }
}
