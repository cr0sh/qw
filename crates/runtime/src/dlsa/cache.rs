// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/core/reference.py (build_buckets: blocks clustered once, never again) and the flat
// bucket cache of src/lsa/sglang/index/bucket_cache.py (append-only rows, block after block).
//
//! The per-layer, per-sequence bucket cache with incremental growth, commit and rollback
//! (`docs/dlsa.md` §4.4, §7.3, §11).

use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

use super::buckets::cluster_block;
use super::config::DlsaIndexConfig;
use super::project::to_host_f32;

/// Counters of the clustering work a cache did (for benchmarks; not part of its state).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DlsaCacheStats {
    pub blocks_clustered: usize,
    pub ward_rounds: usize,
    pub cluster_time: Duration,
}

/// Bytes a cache holds, by part: logical sizes (`len`) and host allocations (`capacity`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DlsaCacheMemory {
    /// Host bucket tables in use: centroids f32 + members i32 + sizes i32 (`rows * (4 d + 4 cap + 4)`).
    pub host_tables: usize,
    /// Host bytes allocated for the bucket tables (`>= host_tables`; vectors grow geometrically).
    pub host_tables_allocated: usize,
    /// Retained index keys `[tail_start, len)` f32 in use.
    pub tail_keys: usize,
    /// Host bytes allocated for retained index keys. After `commit` this is bounded by
    /// `max(tail_keys, 2 * window_size * index_dim * 4)` (256 KiB a layer for Qwen).
    pub tail_keys_allocated: usize,
    /// The device copy of the centroids used for scoring, f32 (0 until the first `select`).
    pub device_centroids: usize,
}

impl DlsaCacheMemory {
    /// Logical bytes: tables, retained keys and the device copy.
    pub fn total(&self) -> usize {
        self.host_tables + self.tail_keys + self.device_centroids
    }

    /// Bytes actually held: host allocations plus the device copy.
    pub fn allocated(&self) -> usize {
        self.host_tables_allocated + self.tail_keys_allocated + self.device_centroids
    }
}

/// The bucket cache of one sequence in one full-attention layer.
///
/// Keys are appended in position order (`len` of them, positions `0..len`). Appending the last
/// key of block `b` (position `W b + W - 1`) clusters that block at once, so the cache always
/// holds `floor(len / W)` closed blocks and `m` bucket rows for each; a trailing partial block
/// has no rows. The index keys of `[tail_start, len)` are retained, where `tail_start` is the
/// start of the block that holds the last commit point: [`trim`](Self::trim) can roll back to
/// any `n >= tail_start`, reopening the blocks that ended past `n`, and
/// [`commit`](Self::commit) frees the keys of blocks closed before it.
pub struct DlsaBucketCache {
    cfg: DlsaIndexConfig,
    len: usize,
    committed: usize,
    tail_start: usize,
    centroids: Vec<f32>,
    members: Vec<i32>,
    sizes: Vec<i32>,
    tail_keys: Vec<f32>,
    device: Option<(UniquePtr<MlxArray>, usize)>,
    stats: DlsaCacheStats,
}

/// A cache's persistent state, for snapshot/restore (layout: `docs/dlsa.md` §11.5).
#[derive(Debug, Clone, PartialEq)]
pub struct DlsaIndexSnapshot {
    pub cfg: DlsaIndexConfig,
    /// Keys appended.
    pub len: usize,
    /// `[rows, index_dim]` f32, `rows = m * floor(len / W)`.
    pub centroids: Vec<f32>,
    /// `[rows, bucket_cap]` i32, `-1` padded.
    pub members: Vec<i32>,
    /// `[rows]` i32.
    pub sizes: Vec<i32>,
    /// Index keys of the open block, `[len - W * floor(len / W), index_dim]` f32.
    pub open_keys: Vec<f32>,
}

impl DlsaBucketCache {
    pub fn new(cfg: DlsaIndexConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            len: 0,
            committed: 0,
            tail_start: 0,
            centroids: Vec::new(),
            members: Vec::new(),
            sizes: Vec::new(),
            tail_keys: Vec::new(),
            device: None,
            stats: DlsaCacheStats::default(),
        })
    }

    /// A committed cache over `index_keys` (`[T, index_dim]` f32), built in one batch.
    pub fn build(cfg: DlsaIndexConfig, index_keys: &[f32]) -> Result<Self> {
        let mut cache = Self::new(cfg)?;
        cache.append(index_keys)?;
        cache.commit();
        Ok(cache)
    }

    pub fn config(&self) -> &DlsaIndexConfig {
        &self.cfg
    }
    /// Keys appended (logical positions `0..len`).
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The last commit point.
    pub fn committed(&self) -> usize {
        self.committed
    }
    /// The first position whose index key is retained; [`trim`](Self::trim) can reach it.
    pub fn tail_start(&self) -> usize {
        self.tail_start
    }
    pub fn closed_blocks(&self) -> usize {
        self.len / self.cfg.window_size
    }
    /// Bucket rows: `m * closed_blocks`.
    pub fn rows(&self) -> usize {
        self.sizes.len()
    }
    /// `[rows, index_dim]` f32.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
    /// `[rows, bucket_cap]` i32.
    pub fn members(&self) -> &[i32] {
        &self.members
    }
    /// `[rows]` i32.
    pub fn sizes(&self) -> &[i32] {
        &self.sizes
    }
    /// Retained index keys `[tail_start, len)`, `[len - tail_start, index_dim]` f32.
    pub fn tail_keys(&self) -> &[f32] {
        &self.tail_keys
    }
    pub fn stats(&self) -> DlsaCacheStats {
        self.stats
    }

    /// Buckets a query at `position` scores: `floor(p / W) * m`, never more than the cache holds.
    pub fn visible_buckets(&self, position: usize) -> usize {
        self.cfg.visible_buckets(position).min(self.rows())
    }

    pub fn memory(&self) -> DlsaCacheMemory {
        DlsaCacheMemory {
            host_tables: 4 * (self.centroids.len() + self.members.len() + self.sizes.len()),
            host_tables_allocated: 4
                * (self.centroids.capacity() + self.members.capacity() + self.sizes.capacity()),
            tail_keys: 4 * self.tail_keys.len(),
            tail_keys_allocated: 4 * self.tail_keys.capacity(),
            device_centroids: self
                .device
                .as_ref()
                .map_or(0, |(_, rows)| rows * self.cfg.index_dim * 4),
        }
    }

    /// Bytes actually held (host allocations plus the device copy), [`DlsaCacheMemory::allocated`].
    pub fn nbytes(&self) -> usize {
        self.memory().allocated()
    }

    /// Capacity of the retained-key buffer kept across commits: two blocks' worth, so decode
    /// appends do not reallocate every step while a large prefill's buffer is released.
    fn tail_capacity_bound(&self) -> usize {
        2 * self.cfg.window_size * self.cfg.index_dim
    }

    /// Append the index keys of the next `L` positions (`[L, index_dim]` f32, unit rows), and
    /// cluster every block they complete, including one completed mid-append.
    pub fn append(&mut self, index_keys: &[f32]) -> Result<()> {
        let d = self.cfg.index_dim;
        ensure!(
            index_keys.len().is_multiple_of(d),
            "index keys must be [L, {d}], got {} values",
            index_keys.len()
        );
        ensure!(
            index_keys.iter().all(|v| v.is_finite()),
            "index keys hold a non-finite value"
        );
        ensure!(
            self.len + index_keys.len() / d <= i32::MAX as usize,
            "positions overflow i32"
        );
        self.tail_keys.extend_from_slice(index_keys);
        let new_len = self.len + index_keys.len() / d;
        let w = self.cfg.window_size;
        let mut closed = self.closed_blocks();
        let close = new_len / w;
        let (m, cap) = (self.cfg.buckets_per_block(), self.cfg.bucket_cap);
        let new_rows = (close - closed) * m;
        self.centroids.reserve(new_rows * d);
        self.members.reserve(new_rows * cap);
        self.sizes.reserve(new_rows);
        while closed < close {
            let start = w * closed - self.tail_start;
            let started = Instant::now();
            let block = cluster_block(
                &self.tail_keys[start * d..(start + w) * d],
                w * closed,
                &self.cfg,
            );
            let block = match block {
                Ok(block) => block,
                Err(e) => {
                    // Leave the cache as it was before this append
                    self.centroids
                        .truncate(self.cfg.buckets_per_block() * self.closed_blocks() * d);
                    let m = self.cfg.buckets_per_block() * self.closed_blocks();
                    self.members.truncate(m * self.cfg.bucket_cap);
                    self.sizes.truncate(m);
                    self.tail_keys.truncate((self.len - self.tail_start) * d);
                    return Err(e);
                }
            };
            self.stats.blocks_clustered += 1;
            self.stats.ward_rounds += block.ward_rounds;
            self.stats.cluster_time += started.elapsed();
            self.centroids.extend_from_slice(&block.centroids);
            self.members.extend_from_slice(&block.members);
            self.sizes.extend_from_slice(&block.sizes);
            closed += 1;
        }
        self.len = new_len;
        Ok(())
    }

    /// [`append`](Self::append) from an MLX array `[L, index_dim]` (evaluated and read back).
    pub fn append_mlx(&mut self, index_keys: &MlxArray) -> Result<()> {
        let shape = mlxcel_core::array_shape(index_keys);
        ensure!(
            shape.len() == 2 && shape[1] as usize == self.cfg.index_dim,
            "index keys must be [L, {}], got {shape:?}",
            self.cfg.index_dim
        );
        self.append(&to_host_f32(index_keys))
    }

    /// Accept everything appended: later rollbacks cannot go below the block holding `len`,
    /// and the keys of blocks closed before it are freed.
    pub fn commit(&mut self) {
        self.committed = self.len;
        let new_start = self.cfg.window_size * self.closed_blocks();
        let drop = (new_start - self.tail_start) * self.cfg.index_dim;
        self.tail_keys.drain(..drop);
        self.tail_start = new_start;
        // Release what a large forward (a prefill chunk, a batch build) left allocated
        let bound = self.tail_capacity_bound();
        if self.tail_keys.capacity() > bound {
            self.tail_keys.shrink_to(bound);
        }
    }

    /// Roll back to the first `new_len` keys, as if only they had been appended: drops the
    /// rows of every block that ends past `new_len` (reopening it) and the keys past it. Errors
    /// if `new_len > len` or `new_len < tail_start` (those keys are no longer retained).
    pub fn trim(&mut self, new_len: usize) -> Result<()> {
        ensure!(
            new_len <= self.len,
            "cannot trim a cache of {} keys to {new_len}",
            self.len
        );
        ensure!(
            new_len >= self.tail_start,
            "cannot roll back to {new_len}: index keys before {} were freed by a commit",
            self.tail_start
        );
        let rows = self.cfg.buckets_per_block() * (new_len / self.cfg.window_size);
        self.centroids.truncate(rows * self.cfg.index_dim);
        self.members.truncate(rows * self.cfg.bucket_cap);
        self.sizes.truncate(rows);
        self.tail_keys
            .truncate((new_len - self.tail_start) * self.cfg.index_dim);
        self.len = new_len;
        self.committed = self.committed.min(new_len);
        if self.device.as_ref().is_some_and(|(_, r)| *r > rows) {
            // A reopened block closes again with other rows under the same row numbers
            self.device = None;
        }
        Ok(())
    }

    /// Back to an empty cache, releasing every allocation.
    pub fn reset(&mut self) {
        self.len = 0;
        self.committed = 0;
        self.tail_start = 0;
        self.centroids = Vec::new();
        self.members = Vec::new();
        self.sizes = Vec::new();
        self.tail_keys = Vec::new();
        self.device = None;
    }

    /// The persistent state: bucket tables plus the open block's keys. Uncommitted state is
    /// captured as if committed (a restored cache cannot roll back past its open block).
    pub fn snapshot(&self) -> DlsaIndexSnapshot {
        let open_start = self.cfg.window_size * self.closed_blocks();
        let d = self.cfg.index_dim;
        DlsaIndexSnapshot {
            cfg: self.cfg,
            len: self.len,
            centroids: self.centroids.clone(),
            members: self.members.clone(),
            sizes: self.sizes.clone(),
            open_keys: self.tail_keys[(open_start - self.tail_start) * d..].to_vec(),
        }
    }

    /// A committed cache from a snapshot taken under the same config.
    pub fn restore(cfg: DlsaIndexConfig, snap: &DlsaIndexSnapshot) -> Result<Self> {
        ensure!(
            snap.cfg == cfg,
            "the snapshot was taken under {:?}, not {cfg:?}",
            snap.cfg
        );
        let mut cache = Self::new(cfg)?;
        let (w, d, cap) = (cfg.window_size, cfg.index_dim, cfg.bucket_cap);
        let closed = snap.len / w;
        let rows = closed * cfg.buckets_per_block();
        ensure!(
            snap.centroids.len() == rows * d
                && snap.members.len() == rows * cap
                && snap.sizes.len() == rows
                && snap.open_keys.len() == (snap.len - closed * w) * d,
            "snapshot tables do not match {} keys",
            snap.len
        );
        ensure!(
            snap.sizes.iter().all(|&s| s >= 0 && s as usize <= cap)
                && snap
                    .members
                    .iter()
                    .all(|&m| m >= -1 && (m as i64) < (closed * w) as i64),
            "snapshot bucket tables are out of range"
        );
        cache.len = snap.len;
        cache.committed = snap.len;
        cache.tail_start = closed * w;
        cache.centroids = snap.centroids.clone();
        cache.members = snap.members.clone();
        cache.sizes = snap.sizes.clone();
        cache.tail_keys = snap.open_keys.clone();
        Ok(cache)
    }

    /// The device copy of the first `rows` centroids (`[rows, index_dim]` f32), synced from the
    /// host tables: grown by the rows appended since the last sync, cut after a trim.
    pub(super) fn device_centroids(&mut self, rows: usize) -> Option<UniquePtr<MlxArray>> {
        let total = self.rows();
        if total == 0 || rows == 0 {
            return None;
        }
        let d = self.cfg.index_dim;
        let synced = match self.device.take() {
            Some((arr, r)) if r == total => arr,
            Some((arr, r)) if r < total => {
                let new = mlxcel_core::from_slice_f32(
                    &self.centroids[r * d..total * d],
                    &[(total - r) as i32, d as i32],
                );
                mlxcel_core::concatenate(&arr, &new, 0)
            }
            // After a trim (which also drops the copy) or a restore: rebuild from the host
            _ => mlxcel_core::from_slice_f32(&self.centroids, &[total as i32, d as i32]),
        };
        mlxcel_core::eval(&synced);
        let view = mlxcel_core::slice(&synced, &[0, 0], &[rows as i32, d as i32]);
        self.device = Some((synced, total));
        Some(view)
    }
}
