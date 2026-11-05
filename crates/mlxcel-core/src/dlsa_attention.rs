// SPDX-License-Identifier: Apache-2.0
//
//! D-LSA fused sparse gather-attention (K2, `docs/dlsa.md` §14.1): every query row attends over
//! its own read list of cached keys only, in one Metal kernel (`cpp/mlx_cxx_dlsa_attention.cpp`).
//!
//! - `q` `[1, Hq, L, D]` (f16/bf16/f32), `k`/`v` `[1, Hkv, T, D]` (one dtype; strided views of a
//!   larger cache buffer are read in place, never copied), `lists` `[L, W]` i32 (any order, no
//!   duplicates, `-1` pad past `lengths[r]`), `lengths` `[L]` i32. Output `[1, Hq, L, D]` in q's
//!   dtype; head `h` reads KV head `h / (Hq / Hkv)`. `D` ∈ {32, 64, 128, 256}, `Hq / Hkv ≤ 8`.
//! - No mask and no causality check: the lists are causal by construction. A list entry outside
//!   `[0, T)` is skipped; a row with no valid entry gets zeros.
//! - Rows with many `L · Hkv` threadgroups run one pass; few rows (decode, verify) split each list
//!   into partitions and merge them (split-K), see [`SparseAttentionLaunch`].
//! - Deterministic for fixed shapes and launch knobs; independent of list order up to f32 rounding.

use cxx::UniquePtr;

use crate::MlxArray;

/// Launch knobs of [`sparse_attention_with`] (tests and benches). `0` means automatic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseAttentionLaunch {
    /// `1` = one pass, `> 1` = split-K partitions of each list (at most 128).
    pub parts: i32,
    /// Simdgroups per threadgroup (1..=16).
    pub simdgroups: i32,
}

/// The fused sparse read with the automatic launch policy. Panics on invalid shapes or dtypes
/// (a programming error at the call site).
pub fn sparse_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    lists: &MlxArray,
    lengths: &MlxArray,
    scale: f32,
) -> UniquePtr<MlxArray> {
    crate::ffi::dlsa_sparse_attention(q, k, v, lists, lengths, scale)
        .unwrap_or_else(|e| panic!("dlsa_sparse_attention: {e}"))
}

/// The fused sparse read with explicit launch knobs.
pub fn sparse_attention_with(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    lists: &MlxArray,
    lengths: &MlxArray,
    scale: f32,
    launch: SparseAttentionLaunch,
) -> Result<UniquePtr<MlxArray>, cxx::Exception> {
    crate::ffi::dlsa_sparse_attention_with(
        q,
        k,
        v,
        lists,
        lengths,
        scale,
        launch.parts,
        launch.simdgroups,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array_to_raw_bytes, astype, dtype, eval, from_slice_f32, from_slice_i32};

    fn host(x: &MlxArray) -> Vec<f32> {
        let x = astype(x, dtype::FLOAT32);
        eval(&x);
        array_to_raw_bytes(&x)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect()
    }

    /// Deterministic pseudo-random values in about [-1.5, 1.5].
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.0
            })
            .collect()
    }

    struct Case {
        hq: usize,
        hkv: usize,
        d: usize,
        t: usize,
        rows: usize,
        width: usize,
        lists: Vec<i32>,
        lengths: Vec<i32>,
    }

    /// Row r sits at position t - rows + r and reads a scattered, unsorted subset of its causal
    /// keys (always itself), `-1` padded to `width`.
    fn case(hq: usize, hkv: usize, d: usize, t: usize, rows: usize, seed: u64) -> Case {
        let mut lists = Vec::new();
        let mut lens = Vec::new();
        let mut raw = Vec::new();
        for r in 0..rows {
            let p = t - rows + r;
            let mut l: Vec<i32> = (0..p)
                .filter(|k| (k * 7 + p + seed as usize).is_multiple_of(3))
                .map(|k| k as i32)
                .collect();
            l.push(p as i32);
            // Unsorted: rotate and reverse halves
            let mid = l.len() / 2;
            l[..mid].reverse();
            l.rotate_left(mid / 3);
            raw.push(l);
        }
        let width = raw.iter().map(Vec::len).max().unwrap() + 5;
        for l in &raw {
            lens.push(l.len() as i32);
            let mut row = l.clone();
            row.resize(width, -1);
            lists.extend(row);
        }
        Case {
            hq,
            hkv,
            d,
            t,
            rows,
            width,
            lists,
            lengths: lens,
        }
    }

    fn host_attention(c: &Case, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
        let (d, g) = (c.d, c.hq / c.hkv);
        let mut out = vec![0f32; c.hq * c.rows * d];
        for h in 0..c.hq {
            let kh = h / g;
            for r in 0..c.rows {
                let qr = &q[(h * c.rows + r) * d..][..d];
                let list = &c.lists[r * c.width..][..c.lengths[r] as usize];
                let s: Vec<f64> = list
                    .iter()
                    .map(|&p| {
                        let kr = &k[(kh * c.t + p as usize) * d..][..d];
                        qr.iter()
                            .zip(kr)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for j in 0..d {
                    let acc: f64 = list
                        .iter()
                        .zip(&w)
                        .map(|(&p, wi)| wi * v[(kh * c.t + p as usize) * d + j] as f64)
                        .sum();
                    out[(h * c.rows + r) * d + j] = (acc / z) as f32;
                }
            }
        }
        out
    }

    fn run_case(c: &Case, launch: SparseAttentionLaunch) -> f32 {
        let scale = 1.0 / (c.d as f32).sqrt();
        let qh = values(c.hq * c.rows * c.d, 1);
        let kh = values(c.hkv * c.t * c.d, 2);
        let vh = values(c.hkv * c.t * c.d, 3);
        let (hq, hkv, t, rows, d) = (
            c.hq as i32,
            c.hkv as i32,
            c.t as i32,
            c.rows as i32,
            c.d as i32,
        );
        let q = from_slice_f32(&qh, &[1, hq, rows, d]);
        let k = from_slice_f32(&kh, &[1, hkv, t, d]);
        let v = from_slice_f32(&vh, &[1, hkv, t, d]);
        let lists = from_slice_i32(&c.lists, &[rows, c.width as i32]);
        let lengths = from_slice_i32(&c.lengths, &[rows]);
        let got = sparse_attention_with(&q, &k, &v, &lists, &lengths, scale, launch).unwrap();
        assert_eq!(crate::array_shape(&got), vec![1, hq, rows, d]);
        let got = host(&got);
        let want = host_attention(c, &qh, &kh, &vh, scale);
        // A NaN anywhere is a failure (f32::max would skip it)
        got.iter()
            .zip(&want)
            .map(|(a, b)| {
                if a.is_finite() {
                    (a - b).abs()
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn f32_matches_host_softmax_across_launches() {
        if !crate::metal_is_available() {
            return;
        }
        let mut failures = Vec::new();
        for (hq, hkv, d) in [
            (6, 2, 32),
            (24, 4, 256),
            (8, 8, 64),
            (16, 2, 128),
            (6, 1, 32),
        ] {
            for (t, rows) in [(40, 1), (97, 3), (200, 8), (300, 37)] {
                let c = case(hq, hkv, d, t, rows, (t + rows) as u64);
                for launch in [
                    SparseAttentionLaunch::default(),
                    SparseAttentionLaunch {
                        parts: 1,
                        simdgroups: 1,
                    },
                    SparseAttentionLaunch {
                        parts: 1,
                        simdgroups: 16,
                    },
                    SparseAttentionLaunch {
                        parts: 7,
                        simdgroups: 4,
                    },
                    SparseAttentionLaunch {
                        parts: 64,
                        simdgroups: 2,
                    },
                ] {
                    let err = run_case(&c, launch);
                    // run_case maps a non-finite output to +inf, so this catches NaN too
                    if err >= 1e-5 {
                        failures.push(format!(
                            "{hq}/{hkv}/{d} t={t} rows={rows} {launch:?}: {err:e}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
