// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/core/reference.py (build_buckets) and src/lsa/core/ward.py (members_from_labels).
//
//! One closed cluster block's buckets: pool, Ward, then per-bucket centroid and members
//! (`docs/dlsa.md` §4.3).

use anyhow::{Result, ensure};

use super::config::DlsaIndexConfig;
use super::ward::{pool_keys, ward_labels};

/// Norm floor of a bucket centroid (`reference.build_buckets`: `clamp_min(1e-30)`).
pub const CENTROID_NORM_FLOOR: f32 = 1e-30;

/// The `m = window_size / bucket_size` buckets of one closed block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockBuckets {
    /// `[m, index_dim]` f32: `normalize(sum of the bucket's index keys)`.
    pub centroids: Vec<f32>,
    /// `[m, bucket_cap]` i32: absolute positions of the bucket's keys, ascending, `-1` padded.
    pub members: Vec<i32>,
    /// `[m]` i32: keys in each bucket.
    pub sizes: Vec<i32>,
    /// Ward rounds the block took.
    pub ward_rounds: usize,
}

/// Cluster one closed block: `keys` are its `window_size` index keys (`[W, index_dim]` f32,
/// unit rows), `base` the position of its first key (a multiple of `window_size`).
pub fn cluster_block(keys: &[f32], base: usize, cfg: &DlsaIndexConfig) -> Result<BlockBuckets> {
    let (w, d, cap) = (cfg.window_size, cfg.index_dim, cfg.bucket_cap);
    let m = cfg.buckets_per_block();
    ensure!(
        keys.len() == w * d,
        "a cluster block holds {w} keys of {d} dims, got {} values",
        keys.len()
    );
    let pools = pool_keys(keys, d, cfg.pool_size);
    let pool_sizes = vec![cfg.pool_size as f32; cfg.pools_per_block()];
    let ward = ward_labels(&pools, &pool_sizes, d, m, cap)?;
    let mut sums = vec![0.0f32; m * d];
    let mut members = vec![-1i32; m * cap];
    let mut sizes = vec![0i32; m];
    // Keys in position order: the sums accumulate as `index_add_` does, and members come out ascending
    for (t, key) in keys.chunks_exact(d).enumerate() {
        let b = ward.labels[t / cfg.pool_size] as usize;
        for (s, &v) in sums[b * d..(b + 1) * d].iter_mut().zip(key) {
            *s += v;
        }
        let n = sizes[b] as usize;
        ensure!(
            n < cap,
            "bucket {b} of block at {base} exceeds the cap {cap}"
        );
        members[b * cap + n] = (base + t) as i32;
        sizes[b] += 1;
    }
    for row in sums.chunks_exact_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm = norm.max(CENTROID_NORM_FLOOR);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    Ok(BlockBuckets {
        centroids: sums,
        members,
        sizes,
        ward_rounds: ward.rounds,
    })
}
