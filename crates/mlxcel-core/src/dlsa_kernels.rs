// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! D-LSA K1: the fused top-p bucket select + expansion + window + sinks
//! (`cpp/mlx_cxx_dlsa.cpp`, `docs/dlsa.md` §14.1).
//!
//! [`dlsa_select`] turns each query row's centroid scores into its read list on
//! the device. Lists, lengths and picked buckets are bit-identical, in order, to
//! qw-runtime's host reference (`dlsa/select.rs` `select_from_scores`) on the
//! same scores, with one deviation: the host refuses a NaN score, while the
//! kernel ranks a NaN below every number and sets [`STATUS_NAN`] in the row's
//! `status`. Read `status` only where the caller already syncs.

use crate::{MlxArray, UniquePtr, ffi};

/// `status` bit: the row had a NaN score (ranked below every number).
pub const STATUS_NAN: i32 = 1;
/// `status` bit: the row's visible buckets exceeded the scores' columns; only the columns
/// present were ranked.
pub const STATUS_SHORT_SCORES: i32 = 2;
/// Most buckets a row may pick (the kernel's threadgroup key buffer).
pub const DLSA_TOP_P_MAX: i32 = 2048;

/// The flat-design knobs the select reads (qw-runtime's `DlsaIndexConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlsaSelectParams {
    /// Window and cluster block size `W`.
    pub window: i32,
    pub sink: i32,
    pub top_k: i32,
    /// At most [`DLSA_TOP_P_MAX`].
    pub top_p: i32,
    /// Row width of the member table.
    pub bucket_cap: i32,
    /// Buckets one closed block yields (`m = W / bucket_size`).
    pub buckets_per_block: i32,
}

impl DlsaSelectParams {
    /// Entries of one read list: `top_k + window + sink`.
    pub fn list_width(&self) -> i32 {
        self.top_k + self.window + self.sink
    }
}

/// The device outputs of [`dlsa_select`], all i32 and lazy (not evaluated).
pub struct DlsaSelectOut {
    /// `[R, top_k + window + sink]`: the expansion (picked buckets best first, members
    /// ascending, cut at `top_k`), the window ascending, the sinks ascending; `-1` padded.
    pub lists: UniquePtr<MlxArray>,
    /// `[R]` valid entries of each list.
    pub lengths: UniquePtr<MlxArray>,
    /// `[R, top_p]` picked bucket rows, best first, `-1` padded.
    pub picked: UniquePtr<MlxArray>,
    /// `[R]` 0, or [`STATUS_NAN`] | [`STATUS_SHORT_SCORES`] bits.
    pub status: UniquePtr<MlxArray>,
}

/// Select the read lists of `R` query rows on the device.
///
/// - `scores` `[R, S]` f32: row `r` ranks its first `visible(p_r)` columns, with
///   `visible(p) = min(floor(p / window) * buckets_per_block, total_rows)` (`S` must cover
///   it, else [`STATUS_SHORT_SCORES`]).
/// - `members` `[N, bucket_cap]` i32: each bucket's key positions, ascending, `-1` padded;
///   only its first `total_rows <= N` rows are buckets (a preallocated device mirror).
/// - `positions` `[R]` i32: each row's absolute position, `>= 0`.
///
/// Errors on bad shapes, dtypes or knobs, and on a back-end without Metal.
pub fn dlsa_select(
    scores: &MlxArray,
    members: &MlxArray,
    positions: &MlxArray,
    total_rows: i32,
    params: &DlsaSelectParams,
) -> Result<DlsaSelectOut, cxx::Exception> {
    dlsa_select_with(scores, members, positions, total_rows, params, 0)
}

/// [`dlsa_select`] with an explicit threadgroup size (`0` = auto, else a power of two in
/// `[64, 1024]`), for tests and benches. The outputs do not depend on it.
pub fn dlsa_select_with(
    scores: &MlxArray,
    members: &MlxArray,
    positions: &MlxArray,
    total_rows: i32,
    params: &DlsaSelectParams,
    threads: i32,
) -> Result<DlsaSelectOut, cxx::Exception> {
    let mut out = DlsaSelectOut {
        lists: UniquePtr::null(),
        lengths: UniquePtr::null(),
        picked: UniquePtr::null(),
        status: UniquePtr::null(),
    };
    ffi::dlsa_select(
        scores,
        members,
        positions,
        total_rows,
        params.window,
        params.sink,
        params.top_k,
        params.top_p,
        params.bucket_cap,
        params.buckets_per_block,
        threads,
        &mut out.lists,
        &mut out.lengths,
        &mut out.picked,
        &mut out.status,
    )?;
    Ok(out)
}

#[cfg(test)]
#[path = "dlsa_kernels_tests.rs"]
mod tests;
