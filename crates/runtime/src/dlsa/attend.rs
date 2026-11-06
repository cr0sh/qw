// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/core/attention.py (the read mask: one key list per query, shared by all heads).
//
//! Attention restricted to D-LSA read lists, in plain MLX ops (`docs/dlsa.md` §4.6, §7.1.5):
//! a gathered read for decode rows, an additive list mask over the cached keys for prefill
//! chunks. Inputs are the model's own rotated q/k/v, `[1, H, L, D]` / `[1, Hkv, T, D]`.

use std::ops::Range;

use mlxcel_core::{MlxArray, UniquePtr};

use super::config::DlsaIndexConfig;
use super::profile::{DlsaStage, lap};
use super::select::DlsaSelection;

/// Whether a query at `last_position` (and so every query before it) reads every causal key:
/// all non-window, non-sink keys fit in `top_k` and the visible buckets fit in `top_p`
/// (`docs/dlsa.md` §4.7). 4612 keys under the Qwen defaults.
pub fn reads_dense_key_set(cfg: &DlsaIndexConfig, last_position: usize) -> bool {
    last_position < cfg.list_width() && cfg.visible_buckets(last_position) <= cfg.top_p
}

fn sdpa(
    q: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let mask_ptr = mask.map_or(std::ptr::null(), |m| m as *const MlxArray);
    // SAFETY: `mask_ptr` is null or borrows `mask`, which outlives the call.
    unsafe { mlxcel_core::layers::attention_from_ptr(q, keys, values, scale, mask_ptr, 0.0, 0) }
}

/// One query row (`q` `[1, Hq, 1, D]`) over the keys of `list` only: the listed rows of
/// `keys`/`values` (`[1, Hkv, T, D]`) are gathered in ascending position order, then plain SDPA
/// (GQA by head grouping, no mask: every listed key is causal).
pub fn gather_attention(
    q: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    list: &[i32],
    scale: f32,
) -> UniquePtr<MlxArray> {
    let total = mlxcel_core::array_shape(keys)[2];
    let mut sorted = list.to_vec();
    sorted.sort_unstable();
    assert!(
        sorted.first().is_some_and(|&p| p >= 0) && sorted.last().is_some_and(|&p| p < total),
        "a D-LSA read list must be non-empty and within the {total} cached keys"
    );
    let idx = mlxcel_core::from_slice_i32(&sorted, &[sorted.len() as i32]);
    lap(DlsaStage::ListUpload, &[&idx]);
    let k = mlxcel_core::take(keys, &idx, 2);
    let v = mlxcel_core::take(values, &idx, 2);
    lap(DlsaStage::Gather, &[&k, &v]);
    sdpa(q, &k, &v, scale, None)
}

/// The additive read mask `[1, 1, rows, total_keys]` (0 on listed keys, -inf elsewhere) of
/// `sel`'s rows `rows`, whose queries sit at `positions_start + r`. A `-1` pad is replaced by
/// the row's own position, which every list holds, so no row is fully masked.
pub(crate) fn list_mask(
    sel: &DlsaSelection,
    positions_start: usize,
    rows: Range<usize>,
    total_keys: usize,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let rb = rows.len();
    // Only the block's longest list needs scattering
    let width = rows
        .clone()
        .map(|r| sel.lengths[r] as usize)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut idx = Vec::with_capacity(rb * width);
    for r in rows {
        let own = (positions_start + r) as i32;
        let row = &sel.lists[r * sel.width..r * sel.width + width];
        idx.extend(row.iter().map(|&p| if p < 0 { own } else { p }));
    }
    let (rb, width, total) = (rb as i32, width as i32, total_keys as i32);
    let idx = mlxcel_core::from_slice_i32(&idx, &[rb, width]);
    let blocked = mlxcel_core::full_f32(&[rb, total], f32::NEG_INFINITY, dtype);
    let open = mlxcel_core::zeros(&[rb, width], dtype);
    let mask = mlxcel_core::put_along_axis(&blocked, &idx, &open, 1);
    mlxcel_core::reshape(&mask, &[1, 1, rb, total])
}

/// Every query row of `q` (`[1, Hq, L, D]`, positions `positions_start..+L`, `sel.rows == L`)
/// over its own list, as SDPA over all cached keys under [`list_mask`], `row_block` rows a mask.
pub fn masked_attention(
    q: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    sel: &DlsaSelection,
    positions_start: usize,
    row_block: usize,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let q_shape = mlxcel_core::array_shape(q);
    let rows = q_shape[2] as usize;
    assert_eq!(sel.rows, rows, "one D-LSA read list per query row");
    let total = mlxcel_core::array_shape(keys)[2] as usize;
    let dtype = mlxcel_core::array_dtype(q);
    let row_block = row_block.max(1);
    let mut parts = Vec::with_capacity(rows.div_ceil(row_block));
    for start in (0..rows).step_by(row_block) {
        let stop = (start + row_block).min(rows);
        let q_block = if start == 0 && stop == rows {
            mlxcel_core::share(q)
        } else {
            mlxcel_core::slice(
                q,
                &[0, 0, start as i32, 0],
                &[q_shape[0], q_shape[1], stop as i32, q_shape[3]],
            )
        };
        let mask = list_mask(sel, positions_start, start..stop, total, dtype);
        lap(DlsaStage::MaskBuild, &[&mask]);
        parts.push(sdpa(&q_block, keys, values, scale, Some(&mask)));
        lap(DlsaStage::Read, &[parts.last().expect("just pushed")]);
    }
    let mut out = parts.remove(0);
    for part in parts {
        out = mlxcel_core::concatenate(&out, &part, 2);
    }
    out
}

/// Every query row of `q` (`[1, Hq, L, D]`, positions `positions_start..+L`) over all its causal
/// keys: SDPA over the cached keys under an additive causal mask. The rows of a forward that
/// crosses `DlsaConfig::dense_below` below it; equal to the dense attention up to rounding.
pub fn causal_attention(
    q: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    positions_start: usize,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let rows = mlxcel_core::array_shape(q)[2];
    let total = mlxcel_core::array_shape(keys)[2];
    let dtype = mlxcel_core::array_dtype(q);
    let start = positions_start as i32;
    let query = mlxcel_core::reshape(&mlxcel_core::arange_i32(start, start + rows, 1), &[rows, 1]);
    let key = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, total, 1), &[1, total]);
    let future = mlxcel_core::greater(&key, &query);
    let mask = mlxcel_core::where_cond(
        &future,
        &mlxcel_core::full_f32(&[rows, total], f32::NEG_INFINITY, dtype),
        &mlxcel_core::zeros(&[rows, total], dtype),
    );
    let mask = mlxcel_core::reshape(&mask, &[1, 1, rows, total]);
    lap(DlsaStage::MaskBuild, &[&mask]);
    let out = sdpa(q, keys, values, scale, Some(&mask));
    lap(DlsaStage::DenseRead, &[&out]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlsa::test_model::max_abs_diff;
    use crate::dlsa::to_host_f32;

    const HQ: usize = 6;
    const HKV: usize = 2;
    const D: usize = 32;

    fn tensor(shape: &[usize], seed: f32) -> (Vec<f32>, UniquePtr<MlxArray>) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.7310 + seed).sin() * 1.3 + (i as f32 * 0.123).cos() * 0.4)
            .collect();
        let shape32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let arr = mlxcel_core::from_slice_f32(&data, &shape32);
        (data, arr)
    }

    /// A selection over `rows` queries at `start..`, row r's list given by `pick(position)`.
    fn selection(start: usize, rows: usize, pick: impl Fn(usize) -> Vec<i32>) -> DlsaSelection {
        let lists: Vec<Vec<i32>> = (0..rows).map(|r| pick(start + r)).collect();
        let width = lists.iter().map(Vec::len).max().unwrap() + 3;
        let mut flat = vec![-1; rows * width];
        for (r, l) in lists.iter().enumerate() {
            flat[r * width..r * width + l.len()].copy_from_slice(l);
        }
        DlsaSelection {
            rows,
            width,
            top_p: 1,
            lists: flat,
            lengths: lists.iter().map(|l| l.len() as i32).collect(),
            picked: vec![-1; rows],
            visible: vec![0; rows],
        }
    }

    /// A sparse, unordered list for a query at `p`: itself, some far keys, a short window.
    fn sparse_pick(p: usize) -> Vec<i32> {
        let mut l: Vec<i32> = (0..p)
            .filter(|k| (k * 7 + p).is_multiple_of(5))
            .map(|k| k as i32)
            .collect();
        l.reverse();
        l.extend((p.saturating_sub(2)..=p).map(|k| k as i32));
        l.push(0);
        l.dedup();
        let mut seen = std::collections::BTreeSet::new();
        l.retain(|k| seen.insert(*k));
        l
    }

    /// f64 softmax attention of every row over its own list.
    fn host_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rows: usize,
        total: usize,
        sel: &DlsaSelection,
        scale: f32,
    ) -> Vec<f32> {
        let g = HQ / HKV;
        let mut out = vec![0.0f32; HQ * rows * D];
        for h in 0..HQ {
            let kh = h / g;
            for r in 0..rows {
                let qr = &q[(h * rows + r) * D..(h * rows + r + 1) * D];
                let list = sel.list(r);
                let scores: Vec<f64> = list
                    .iter()
                    .map(|&p| {
                        let kr = &k[(kh * total + p as usize) * D..][..D];
                        qr.iter()
                            .zip(kr)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in 0..D {
                    let acc: f64 = list
                        .iter()
                        .zip(&w)
                        .map(|(&p, wi)| wi * v[(kh * total + p as usize) * D + d] as f64)
                        .sum();
                    out[(h * rows + r) * D + d] = (acc / z) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn dense_key_set_predicate_matches_the_qwen_boundary() {
        let cfg = DlsaIndexConfig::QWEN_DEFAULTS;
        assert!(reads_dense_key_set(&cfg, 0));
        assert!(reads_dense_key_set(&cfg, 4611));
        assert!(!reads_dense_key_set(&cfg, 4612));
        // top_p binds first when it is small
        let cfg = DlsaIndexConfig {
            top_p: 64,
            ..DlsaIndexConfig::QWEN_DEFAULTS
        };
        assert!(reads_dense_key_set(&cfg, 1023));
        assert!(!reads_dense_key_set(&cfg, 1024));
    }

    #[test]
    fn gathered_and_masked_reads_match_host_softmax() {
        let scale = 1.0 / (D as f32).sqrt();
        let (o, rows) = (29usize, 11usize);
        let total = o + rows;
        let (qh, q) = tensor(&[1, HQ, rows, D], 0.3);
        let (kh, k) = tensor(&[1, HKV, total, D], 1.1);
        let (vh, v) = tensor(&[1, HKV, total, D], 2.7);
        let sel = selection(o, rows, sparse_pick);
        let expected = host_attention(&qh, &kh, &vh, rows, total, &sel, scale);
        let masked = to_host_f32(&masked_attention(&q, &k, &v, &sel, o, 4, scale));
        let diff = max_abs_diff(&masked, &expected);
        assert!(diff < 1e-5, "masked read vs host: {diff:e}");

        // Each row through the gathered decode read
        for r in [0, 5, rows - 1] {
            let q_row = mlxcel_core::slice(
                &q,
                &[0, 0, r as i32, 0],
                &[1, HQ as i32, r as i32 + 1, D as i32],
            );
            let got = to_host_f32(&gather_attention(&q_row, &k, &v, sel.list(r), scale));
            let want: Vec<f32> = (0..HQ)
                .flat_map(|h| expected[(h * rows + r) * D..(h * rows + r + 1) * D].to_vec())
                .collect();
            let diff = max_abs_diff(&got, &want);
            assert!(diff < 1e-5, "gathered read of row {r} vs host: {diff:e}");
        }
    }

    #[test]
    fn full_causal_lists_equal_dense_causal_attention() {
        let scale = 1.0 / (D as f32).sqrt();
        let (o, rows) = (13usize, 9usize);
        let total = o + rows;
        let (_, q) = tensor(&[1, HQ, rows, D], 0.9);
        let (_, k) = tensor(&[1, HKV, total, D], 1.7);
        let (_, v) = tensor(&[1, HKV, total, D], 3.1);
        // Window first, then sinks, as the reference lays lists out: order must not matter
        let sel = selection(o, rows, |p| {
            let mut l: Vec<i32> = (2..=p as i32).collect();
            l.extend([0, 1]);
            l
        });
        let dense = mlxcel_core::causal_attention(&q, &k, &v, scale, 0.0, 0);
        let dense = to_host_f32(&dense);
        let masked = to_host_f32(&masked_attention(&q, &k, &v, &sel, o, 256, scale));
        let diff = max_abs_diff(&masked, &dense);
        assert!(diff < 1e-5, "full lists vs dense causal: {diff:e}");

        let q_last = mlxcel_core::slice(
            &q,
            &[0, 0, rows as i32 - 1, 0],
            &[1, HQ as i32, rows as i32, D as i32],
        );
        let gathered = to_host_f32(&gather_attention(
            &q_last,
            &k,
            &v,
            sel.list(rows - 1),
            scale,
        ));
        let dense_last = to_host_f32(&sdpa(&q_last, &k, &v, scale, None));
        let diff = max_abs_diff(&gathered, &dense_last);
        assert!(diff < 1e-5, "full decode list vs dense: {diff:e}");
    }

    #[test]
    fn mask_row_blocks_do_not_change_the_output() {
        let scale = 1.0 / (D as f32).sqrt();
        let (o, rows) = (40usize, 23usize);
        let total = o + rows;
        let (_, q) = tensor(&[1, HQ, rows, D], 4.2);
        let (_, k) = tensor(&[1, HKV, total, D], 5.3);
        let (_, v) = tensor(&[1, HKV, total, D], 6.4);
        let sel = selection(o, rows, sparse_pick);
        let whole = to_host_f32(&masked_attention(&q, &k, &v, &sel, o, 256, scale));
        for block in [1, 7] {
            let got = to_host_f32(&masked_attention(&q, &k, &v, &sel, o, block, scale));
            let diff = max_abs_diff(&got, &whole);
            assert!(diff < 1e-6, "row block {block}: {diff:e}");
        }
    }

    #[test]
    fn pads_become_the_rows_own_position() {
        let sel = selection(5, 2, |p| vec![p as i32]);
        let mask = to_host_f32(&list_mask(&sel, 5, 0..2, 8, mlxcel_core::dtype::FLOAT32));
        for (r, row) in mask.chunks(8).enumerate() {
            for (key, &m) in row.iter().enumerate() {
                assert_eq!(m == 0.0, key == 5 + r, "row {r} key {key}: {m}");
            }
        }
    }
}
