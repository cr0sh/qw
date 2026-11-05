// SPDX-License-Identifier: Apache-2.0
//
//! Shared pieces of the synthetic D-LSA model tests (`qwen3_next::dlsa`, `qwen3_5::dlsa_tests`):
//! deterministic noise, a small but really sparse index config, and maps for a tiny geometry.

use std::collections::BTreeMap;

use mlxcel_core::{MlxArray, UniquePtr};

use super::{DlsaIndexConfig, DlsaLayerMaps, DlsaMapDtype, DlsaMaps, INDEX_DIM};

/// Window 16, sink 2, top_k 32, top_p 8, 4-key buckets of at most 8: lists of at most 50 keys,
/// and every forward past position 47 is sparse.
pub(crate) const TEST_INDEX: DlsaIndexConfig = DlsaIndexConfig {
    window_size: 16,
    sink_size: 2,
    top_k: 32,
    top_p: 8,
    index_dim: INDEX_DIM,
    pool_size: 2,
    bucket_size: 4,
    bucket_cap: 8,
};

/// The tiny geometry: 4 query heads, 2 KV heads, head_dim 32, rotary 8, so 48 index input dims.
pub(crate) const TEST_HEADS: usize = 4;
pub(crate) const TEST_KV_HEADS: usize = 2;
pub(crate) const TEST_HEAD_DIM: usize = 32;
pub(crate) const TEST_INPUT_DIM: usize = 48;

/// Uniform noise in `[-0.5, 0.5)` from a counter and a seed (splitmix64).
pub(crate) fn noise(i: u64, seed: u64) -> f32 {
    let mut z = i
        .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

/// An f32 array of `shape` filled with `scale * noise(i, seed)`.
pub(crate) fn noise_array(shape: &[i32], seed: u64, scale: f32) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|&s| s as usize).product();
    let data: Vec<f32> = (0..n as u64).map(|i| scale * noise(i, seed)).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// The maps of one layer, `[48, 64]` f32 each.
pub(crate) fn test_layer_maps(layer: usize) -> DlsaLayerMaps {
    let shape = [TEST_INPUT_DIM as i32, INDEX_DIM as i32];
    DlsaLayerMaps {
        w_q_idx: noise_array(&shape, 1000 + 2 * layer as u64, 1.0),
        w_k_idx: noise_array(&shape, 1001 + 2 * layer as u64, 1.0),
    }
}

/// F32 maps for `layers`.
pub(crate) fn test_maps(layers: &[usize]) -> DlsaMaps {
    DlsaMaps {
        layers: layers.iter().map(|&l| (l, test_layer_maps(l))).collect(),
        input_dim: TEST_INPUT_DIM,
        dtype: DlsaMapDtype::F32,
        metadata: BTreeMap::new(),
    }
}

/// Both sides the same length and every value finite: `f32::max` would drop a NaN, so a
/// non-finite output is rejected here instead of comparing as close.
#[track_caller]
fn check_comparable(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (side, values) in [("actual", a), ("expected", b)] {
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            panic!("{side} value {i} is not finite: {}", values[i]);
        }
    }
}

/// Max of `|a - b| - (atol + rtol |b|)` over the elements (`<= 0` means allclose). Panics on
/// a non-finite value in either side.
#[track_caller]
pub(crate) fn allclose_excess(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> f32 {
    check_comparable(a, b);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() - (atol + rtol * y.abs()))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Max `|a - b|`. Panics on a non-finite value in either side.
#[track_caller]
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    check_comparable(a, b);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_reject_non_finite_values() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (a, b) in [([bad, 1.0], [1.0, 1.0]), ([1.0, 1.0], [1.0, bad])] {
                let diff = std::panic::catch_unwind(|| max_abs_diff(&a, &b));
                assert!(diff.is_err(), "max_abs_diff accepted {a:?} vs {b:?}");
                let close = std::panic::catch_unwind(|| allclose_excess(&a, &b, 1e-4, 1e-5));
                assert!(close.is_err(), "allclose_excess accepted {a:?} vs {b:?}");
            }
        }
        assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.0, 2.5]), 0.5);
        assert!(allclose_excess(&[1.0], &[1.0], 1e-4, 1e-5) <= 0.0);
    }
}
