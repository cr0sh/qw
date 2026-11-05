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
use super::device::{DeferredKeys, DeviceCount, DeviceMirror};
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
    /// Retained index keys `[tail_start, len)` f32 in use on the host (deferred keys excluded).
    pub tail_keys: usize,
    /// Host bytes allocated for retained index keys. After `commit` this is bounded by
    /// `max(tail_keys, 2 * window_size * index_dim * 4)` (256 KiB a layer for Qwen).
    pub tail_keys_allocated: usize,
    /// The device mirror of the bucket tables (f32 centroids + i32 members, allocated rows), for
    /// scoring and the K1 select (0 until the first `select`).
    pub device_centroids: usize,
    /// Index keys appended with `append_deferred` and not read back yet, f32 on the device
    /// (`deferred_keys() * index_dim * 4`; 0 after a flush).
    pub deferred_keys: usize,
}

impl DlsaCacheMemory {
    /// Logical bytes: tables, retained keys (host and deferred) and the device mirror.
    pub fn total(&self) -> usize {
        self.host_tables + self.tail_keys + self.device_centroids + self.deferred_keys
    }

    /// Bytes actually held: host allocations plus the device mirror and deferred keys.
    pub fn allocated(&self) -> usize {
        self.host_tables_allocated
            + self.tail_keys_allocated
            + self.device_centroids
            + self.deferred_keys
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
    mirror: DeviceMirror,
    /// Keys appended on the device and not read back yet (`append_deferred`).
    deferred: DeferredKeys,
    /// Lazy count of the keys the Metal reads visited (`DlsaStats::keys_read`).
    pub(super) keys_read: DeviceCount,
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
            mirror: DeviceMirror::default(),
            deferred: DeferredKeys::default(),
            keys_read: DeviceCount::default(),
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
    /// Retained index keys `[tail_start, len)`, `[len - tail_start, index_dim]` f32. Reads back
    /// deferred keys first ([`flush`](Self::flush)); panics if they hold a non-finite value.
    pub fn tail_keys(&mut self) -> &[f32] {
        self.flush().expect("D-LSA deferred index keys");
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
            device_centroids: {
                let (c, m) = self.mirror.nbytes(self.cfg.index_dim, self.cfg.bucket_cap);
                c + m
            },
            deferred_keys: 4 * self.deferred.rows() * self.cfg.index_dim,
        }
    }

    /// Bytes actually held (host allocations, the device mirror and deferred keys),
    /// [`DlsaCacheMemory::allocated`].
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
        self.flush()?;
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

    /// Append the index keys of the next `L` positions without reading them back, when they
    /// close no block: they stay on the device (deferred) until something needs the host copy
    /// ([`flush`](Self::flush): closing a block, `tail_keys`, `snapshot`, `trim`). Keys that
    /// close a block go through [`append_mlx`](Self::append_mlx). Scoring never needs them:
    /// only closed blocks have buckets. Their finiteness is checked at the flush.
    pub fn append_deferred(&mut self, index_keys: &MlxArray) -> Result<()> {
        let shape = mlxcel_core::array_shape(index_keys);
        ensure!(
            shape.len() == 2 && shape[1] as usize == self.cfg.index_dim,
            "index keys must be [L, {}], got {shape:?}",
            self.cfg.index_dim
        );
        let rows = shape[0] as usize;
        ensure!(
            self.len + rows <= i32::MAX as usize,
            "positions overflow i32"
        );
        if (self.len + rows) / self.cfg.window_size > self.closed_blocks() {
            return self.append_mlx(index_keys);
        }
        self.deferred.push(index_keys, rows);
        self.len += rows;
        Ok(())
    }

    /// Keys appended with [`append_deferred`](Self::append_deferred) and not read back yet.
    pub fn deferred_keys(&self) -> usize {
        self.deferred.rows()
    }

    /// Read the deferred keys back into the host tables (one sync; nothing when there are
    /// none). Errors on a non-finite key, dropping the deferred keys (`len` falls back by them).
    pub fn flush(&mut self) -> Result<()> {
        let Some((keys, rows)) = self.deferred.take() else {
            return Ok(());
        };
        // They lie in the open block, so this append closes nothing
        self.len -= rows;
        self.append(&to_host_f32(&keys))
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
        self.flush()?;
        let rows = self.cfg.buckets_per_block() * (new_len / self.cfg.window_size);
        self.centroids.truncate(rows * self.cfg.index_dim);
        self.members.truncate(rows * self.cfg.bucket_cap);
        self.sizes.truncate(rows);
        self.tail_keys
            .truncate((new_len - self.tail_start) * self.cfg.index_dim);
        self.len = new_len;
        self.committed = self.committed.min(new_len);
        // A reopened block closes again with other rows under the same row numbers
        self.mirror.truncate(rows);
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
        self.mirror.clear();
        self.deferred.clear();
        self.keys_read.clear();
    }

    /// The persistent state: bucket tables plus the open block's keys. Uncommitted state is
    /// captured as if committed (a restored cache cannot roll back past its open block).
    /// Deferred keys are read back into it (the cache keeps them deferred); panics if they hold
    /// a non-finite value.
    pub fn snapshot(&self) -> DlsaIndexSnapshot {
        let open_start = self.cfg.window_size * self.closed_blocks();
        let d = self.cfg.index_dim;
        let mut open_keys = self.tail_keys[(open_start - self.tail_start) * d..].to_vec();
        if let Some(keys) = self.deferred.get() {
            let keys = to_host_f32(keys);
            assert!(
                keys.iter().all(|v| v.is_finite()),
                "D-LSA deferred index keys hold a non-finite value"
            );
            open_keys.extend_from_slice(&keys);
        }
        DlsaIndexSnapshot {
            cfg: self.cfg,
            len: self.len,
            centroids: self.centroids.clone(),
            members: self.members.clone(),
            sizes: self.sizes.clone(),
            open_keys,
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

    /// The device mirror of the bucket tables, synced from the host tables first.
    pub(super) fn device_mirror(&mut self) -> &DeviceMirror {
        let (d, cap) = (self.cfg.index_dim, self.cfg.bucket_cap);
        self.mirror.sync(&self.centroids, &self.members, d, cap);
        &self.mirror
    }

    /// The device copy of the first `rows` centroids (`[rows, index_dim]` f32, a view of the
    /// mirror), or `None` when there are none.
    pub(super) fn device_centroids(&mut self, rows: usize) -> Option<UniquePtr<MlxArray>> {
        if self.rows() == 0 || rows == 0 {
            return None;
        }
        self.device_mirror().centroids(rows)
    }
}
