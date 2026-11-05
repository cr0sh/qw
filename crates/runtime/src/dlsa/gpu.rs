// SPDX-License-Identifier: Apache-2.0
//
//! The Metal backend of the sparse forward (`docs/dlsa.md` §14.1): the read lists selected on the
//! device by K1 (`mlxcel_core::dlsa_kernels::dlsa_select`) from the same MLX score matmul as
//! task 3, and the read through K2 (`mlxcel_core::dlsa_attention::sparse_attention`). Nothing
//! is read back: the lists stay on the device between the two kernels.

use anyhow::{Result, anyhow, ensure};
use mlxcel_core::dlsa_kernels::{DLSA_TOP_P_MAX, DlsaSelectParams, STATUS_NAN, dlsa_select};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::cache::DlsaBucketCache;
use super::config::DlsaIndexConfig;
use super::profile::{DlsaStage, lap};
use super::select::DlsaSelection;

/// The read lists of `rows` queries on the device ([`DlsaBucketCache::select_device`]), lazy:
/// the same lists, lengths and picks as the host [`DlsaSelection`], bit for bit.
pub struct DeviceSelection {
    pub rows: usize,
    /// `top_k + window_size + sink_size`.
    pub width: usize,
    pub top_p: usize,
    /// `[rows, width]` i32, `-1` padded.
    pub lists: UniquePtr<MlxArray>,
    /// `[rows]` i32.
    pub lengths: UniquePtr<MlxArray>,
    /// `[rows, top_p]` i32, best first, `-1` padded.
    pub picked: UniquePtr<MlxArray>,
    /// `[rows]` i32: 0, or K1's status bits (a NaN score, which task 2 refuses, ranks last).
    pub status: UniquePtr<MlxArray>,
    /// `[rows]` buckets each row could score (host-known).
    pub visible: Vec<i32>,
}

fn to_host_i32(x: &MlxArray) -> Vec<i32> {
    let x = mlxcel_core::astype(x, dtype::INT32);
    mlxcel_core::eval(&x);
    mlxcel_core::array_to_raw_bytes(&x)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect()
}

impl DeviceSelection {
    /// Read the selection back (one sync). Errors when a row saw a NaN score, as the host
    /// select does.
    pub fn to_host(&self) -> Result<DlsaSelection> {
        let status = to_host_i32(&self.status);
        ensure!(
            status.iter().all(|&s| s & STATUS_NAN == 0),
            "a bucket score is NaN"
        );
        ensure!(
            status.iter().all(|&s| s == 0),
            "the device select reported status {status:?}"
        );
        Ok(DlsaSelection {
            rows: self.rows,
            width: self.width,
            top_p: self.top_p,
            lists: to_host_i32(&self.lists),
            lengths: to_host_i32(&self.lengths),
            picked: to_host_i32(&self.picked),
            visible: self.visible.clone(),
        })
    }
}

fn select_params(cfg: &DlsaIndexConfig) -> DlsaSelectParams {
    DlsaSelectParams {
        window: cfg.window_size as i32,
        sink: cfg.sink_size as i32,
        top_k: cfg.top_k as i32,
        top_p: cfg.top_p as i32,
        bucket_cap: cfg.bucket_cap as i32,
        buckets_per_block: cfg.buckets_per_block() as i32,
    }
}

impl DlsaBucketCache {
    /// [`select`](Self::select) on the device: the same MLX score matmul, then K1 over the
    /// device mirror of the member table. Nothing is read back.
    pub fn select_device(
        &mut self,
        index_queries: &MlxArray,
        positions: &[usize],
    ) -> Result<DeviceSelection> {
        let cfg = *self.config();
        ensure!(
            cfg.top_p as i32 <= DLSA_TOP_P_MAX,
            "the device select takes top_p <= {DLSA_TOP_P_MAX}, got {}",
            cfg.top_p
        );
        let rows = positions.len();
        let scores = match self.score_device(index_queries, positions)? {
            Some((scores, _)) => scores,
            None => mlxcel_core::zeros(&[rows as i32, 0], dtype::FLOAT32),
        };
        let total_rows = self.rows();
        let visible = positions
            .iter()
            .map(|&p| cfg.visible_buckets(p).min(total_rows) as i32)
            .collect();
        let pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let pos = mlxcel_core::from_slice_i32(&pos, &[rows as i32]);
        let empty;
        let members = if total_rows == 0 {
            empty = mlxcel_core::zeros(&[0, cfg.bucket_cap as i32], dtype::INT32);
            &*empty
        } else {
            self.device_mirror()
                .members()
                .expect("a synced mirror holds its tables")
        };
        let out = dlsa_select(
            &scores,
            members,
            &pos,
            total_rows as i32,
            &select_params(&cfg),
        )
        .map_err(|e| anyhow!("dlsa_select: {e}"))?;
        lap(DlsaStage::DeviceSelect, &[&out.lists, &out.lengths]);
        Ok(DeviceSelection {
            rows,
            width: cfg.list_width(),
            top_p: cfg.top_p,
            lists: out.lists,
            lengths: out.lengths,
            picked: out.picked,
            status: out.status,
            visible,
        })
    }

    /// Count `sel`'s list lengths into `DlsaStats::keys_read`, lazily (no sync).
    pub(crate) fn count_keys_read(&mut self, sel: &DeviceSelection) {
        self.keys_read.add(&sel.lengths);
    }

    /// The keys counted by [`count_keys_read`](Self::count_keys_read) since the last call,
    /// evaluated: call it only where the forward already synced.
    pub(crate) fn take_keys_read(&mut self) -> u64 {
        self.keys_read.take()
    }
}

/// Every query row of `q` (`[1, Hq, L, D]`, rotated) over its own device list: K2, reading the
/// listed rows of `keys`/`values` (`[1, Hkv, T, D]`, the KV cache views, read in place).
pub fn sparse_read(
    q: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    sel: &DeviceSelection,
    scale: f32,
) -> UniquePtr<MlxArray> {
    mlxcel_core::dlsa_attention::sparse_attention(q, keys, values, &sel.lists, &sel.lengths, scale)
}
