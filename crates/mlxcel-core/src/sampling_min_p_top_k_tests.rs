// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Fused sampler filter regressions:
//!
//! - min-p must filter on every dtype. The compiled min-p filter returned its
//!   input unfiltered on Metal because the compiled-kernel overlay cast the
//!   comparison operands to bool (upstream mlxcel 6ecf756e, d702f4e5).
//! - `top_k >= vocab` means "no top-k" instead of aborting the process in
//!   `argpartition` (upstream mlxcel d078601b).

use super::*;

const LOGITS: [f32; 8] = [2.0, 1.875, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0];

fn logits(dt: i32) -> UniquePtr<MlxArray> {
    astype(&from_slice_f32(&LOGITS, &[1, 8]), dt)
}

fn probs(dt: i32, top_k: i32, top_p: f32, min_p: f32) -> Vec<f32> {
    let p = fused_sample_probs(&logits(dt), 1.0, top_k, top_p, min_p);
    eval(&p);
    array_to_raw_bytes(&p)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Host reference: softmax, drop `p < min_p * p_max`, renormalise.
fn min_p_reference(dt_rounded: &[f32], min_p: f32) -> Vec<f32> {
    let m = dt_rounded.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = dt_rounded.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    let p: Vec<f32> = e.iter().map(|x| x / s).collect();
    let pmax = p.iter().cloned().fold(0.0, f32::max);
    let kept: Vec<f32> = p
        .iter()
        .map(|&x| if x >= min_p * pmax { x } else { 0.0 })
        .collect();
    let ks: f32 = kept.iter().sum();
    kept.iter().map(|x| x / ks).collect()
}

#[test]
fn min_p_filters_on_every_dtype() {
    for dt in [dtype::FLOAT32, dtype::BFLOAT16, dtype::FLOAT16] {
        let got = probs(dt, 0, 1.0, 0.5);
        // The logits are exact in bf16/f16, so one host reference serves all.
        let want = min_p_reference(&LOGITS, 0.5);
        assert_eq!(got.len(), 8);
        // Tokens 2.. are below half the max probability and must be removed.
        for (i, g) in got.iter().enumerate().skip(2) {
            assert_eq!(*g, 0.0, "dtype {dt}: token {i} survived min_p");
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-2, "dtype {dt} token {i}: {g} vs {w}");
        }
        assert!((got.iter().sum::<f32>() - 1.0).abs() < 1e-3);
        // And min_p == 0 leaves the plain softmax.
        let unfiltered = probs(dt, 0, 1.0, 0.0);
        assert!(unfiltered.iter().all(|&p| p > 0.0), "dtype {dt}");
    }
}

#[test]
fn top_k_at_or_above_vocab_equals_no_top_k() {
    for dt in [dtype::FLOAT32, dtype::BFLOAT16] {
        let none = probs(dt, 0, 1.0, 0.0);
        for k in [8, 9, 1000, i32::MAX] {
            assert_eq!(probs(dt, k, 1.0, 0.0), none, "dtype {dt} top_k {k}");
            // Combined with the other filters, still equal to top_k off.
            assert_eq!(
                probs(dt, k, 0.9, 0.1),
                probs(dt, 0, 0.9, 0.1),
                "dtype {dt} top_k {k} with top_p/min_p"
            );
        }
        // A real top_k still filters.
        assert_eq!(
            probs(dt, 2, 1.0, 0.0).iter().filter(|&&p| p > 0.0).count(),
            2
        );
    }
}

#[test]
fn sampling_entry_points_survive_top_k_above_vocab() {
    let x = logits(dtype::FLOAT32);
    let greedy = fused_sample(&x, 0.0, 1000, 1.0, 0.0);
    eval(&greedy);
    assert_eq!(item_i32(&greedy), 0);
    for f in [fused_sample, fused_sample_categorical] {
        let t = f(&x, 0.7, 1000, 0.9, 0.05);
        eval(&t);
        assert!((0..8).contains(&item_i32(&t)));
    }
}

fn f32_values(arr: &MlxArray) -> Vec<f32> {
    let f = astype(arr, dtype::FLOAT32);
    eval(&f);
    array_to_raw_bytes(&f)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compiled and compiled-shapeless graphs equal eager. On Metal the overlay
/// used to cast comparison operands to bool, so compiled `a < b` / `>=` masks
/// were wrong (upstream mlxcel d702f4e5, #2325).
#[test]
fn compiled_comparisons_equal_eager_on_every_dtype() {
    let a_vals = [0.75f32, 0.25, -0.5, 3.0, f32::NAN, 0.0, 0.25, 1.5];
    let b_vals = [0.25f32, 0.75, -0.25, 3.0, 1.0, -1.0, 0.5, 1.5];
    for dt in [dtype::FLOAT32, dtype::BFLOAT16, dtype::FLOAT16] {
        let a = astype(&from_slice_f32(&a_vals, &[2, 4]), dt);
        let b = astype(&from_slice_f32(&b_vals, &[2, 4]), dt);
        let m = astype(&from_slice_f32(&LOGITS, &[1, 8]), dt);
        for (graph, x, y) in [(0, &a, &b), (1, &m, &m), (2, &a, &b)] {
            let eager = f32_values(&compile_dtype_probe_for_test(graph, x, y, 0));
            for mode in [1, 2] {
                let got = f32_values(&compile_dtype_probe_for_test(graph, x, y, mode));
                assert_eq!(got.len(), eager.len());
                for (i, (g, e)) in got.iter().zip(&eager).enumerate() {
                    assert!(
                        // The filtered sentinel (f32 lowest) may differ by a rounding
                        // step between compiled and eager; both are "dropped".
                        (g.is_nan() && e.is_nan()) || g == e || (*g < -1e30 && *e < -1e30),
                        "dtype {dt} graph {graph} mode {mode} [{i}]: {g} vs eager {e}"
                    );
                }
            }
        }
        // Anchor eager: min-p graph 1 actually filters (0.3 * p_max cut).
        let eager = f32_values(&compile_dtype_probe_for_test(1, &m, &m, 0));
        assert_eq!(eager[0], 2.0);
        assert!(
            eager[7] < -1e30,
            "dtype {dt}: lowest tail kept: {}",
            eager[7]
        );
        // Anchor eager: where(a < b, a, b) = min(a, b) for non-NaN lanes.
        let eager0 = f32_values(&compile_dtype_probe_for_test(0, &a, &b, 0));
        assert_eq!(eager0[0], 0.25);
        assert_eq!(eager0[1], 0.25);
    }
}
