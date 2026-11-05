// SPDX-License-Identifier: Apache-2.0
//
//! The fused sparse read (K2, `mlxcel_core::dlsa_attention`) against task 3's MLX reads
//! (`attend.rs`: `masked_attention` for prefill rows, `gather_attention` for decode rows) and
//! dense causal attention, at the Qwen geometry (Hq 24, Hkv 4, D 256, fp16) and the list layout
//! of the D-LSA defaults (expansion, then window, then sinks; `docs/dlsa.md` §4.5).
//!
//! GPU-only (Metal); returns early elsewhere. Run under `./gpu-lock`. The microbenchmark is
//! `#[ignore]`d: `QW_K2_BENCH_CONTEXTS=32768,131072,262144 cargo test --release -p qw-runtime
//! --features dlsa --lib gpu_attend_tests::bench -- --ignored --nocapture`.

use std::collections::HashSet;
use std::time::Instant;

use mlxcel_core::cache::{KVCache, KVCacheMode};
use mlxcel_core::dlsa_attention::{SparseAttentionLaunch, sparse_attention, sparse_attention_with};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::DlsaIndexConfig;
use super::select::DlsaSelection;
use super::test_model::max_abs_diff;
use super::{gather_attention, masked_attention, to_host_f32};

const HQ: i32 = 24;
const HKV: i32 = 4;
const D: i32 = 256;
const CFG: DlsaIndexConfig = DlsaIndexConfig::QWEN_DEFAULTS;
/// task 3's default prefill mask block (`DlsaConfig::prefill_mask_rows`).
const MASK_ROWS: usize = 256;

fn gpu() -> bool {
    mlxcel_core::metal_is_available()
}

fn scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

/// Standard-normal `shape` in `dt`, times `mul`.
fn randn(shape: &[i32], dt: i32, seed: u64, mul: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::random_seed(seed);
    // SAFETY: a null key means the global PRNG state.
    let x = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let x = if mul == 1.0 {
        x
    } else {
        mlxcel_core::multiply(&x, &mlxcel_core::full_f32(&[1], mul, dtype::FLOAT32))
    };
    let x = mlxcel_core::astype(&x, dt);
    mlxcel_core::eval(&x);
    x
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// How the long-range parts of a block of rows relate.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expansion {
    /// Every row draws its own keys (worst cache reuse).
    Independent,
    /// Every row of the block reads the same long-range keys (best cache reuse).
    Shared,
}

/// D-LSA-shaped lists for rows at positions `start..start + rows`: `top_k` distinct long-range
/// keys from `[sink, win_start)` in random order, then the window ascending, then the sinks;
/// `-1` padded to the list width. Short contexts read everything they can.
fn selection(start: usize, rows: usize, top_k: usize, exp: Expansion, seed: u64) -> DlsaSelection {
    let width = top_k + CFG.window_size + CFG.sink_size;
    let mut lists = vec![-1i32; rows * width];
    let mut lengths = Vec::with_capacity(rows);
    for r in 0..rows {
        let p = start + r;
        let win_start = (p + 1).saturating_sub(CFG.window_size);
        let sinks = CFG.sink_size.min(win_start).min(p + 1);
        let pool = win_start.saturating_sub(sinks);
        let mut list: Vec<i32> = Vec::with_capacity(width);
        let want = top_k.min(pool);
        if want == pool {
            let mut all: Vec<i32> = (sinks..win_start).map(|k| k as i32).collect();
            // A fixed shuffle, so the lists are unsorted
            for i in (1..all.len()).rev() {
                all.swap(i, (mix(seed ^ (i as u64) << 1) % (i as u64 + 1)) as usize);
            }
            list.extend(all);
        } else {
            let row_seed = match exp {
                Expansion::Independent => seed ^ (p as u64).wrapping_mul(0x9e37_79b9),
                Expansion::Shared => seed,
            };
            let mut seen = HashSet::with_capacity(want * 2);
            let mut i = 0u64;
            while list.len() < want {
                let k = sinks + (mix(row_seed.wrapping_add(i)) % pool as u64) as usize;
                i += 1;
                if seen.insert(k) {
                    list.push(k as i32);
                }
            }
        }
        list.extend((win_start..=p).map(|k| k as i32));
        list.extend((0..sinks).map(|k| k as i32));
        lengths.push(list.len() as i32);
        lists[r * width..r * width + list.len()].copy_from_slice(&list);
    }
    DlsaSelection {
        rows,
        width,
        top_p: CFG.top_p,
        lists,
        lengths,
        picked: vec![-1; rows * CFG.top_p],
        visible: vec![0; rows],
    }
}

fn lengths_mlx(sel: &DlsaSelection) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&sel.lengths, &[sel.rows as i32])
}

fn k2(q: &MlxArray, k: &MlxArray, v: &MlxArray, sel: &DlsaSelection) -> UniquePtr<MlxArray> {
    sparse_attention(q, k, v, &sel.lists_mlx(), &lengths_mlx(sel), scale())
}

fn k2_with(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    sel: &DlsaSelection,
    parts: i32,
    simdgroups: i32,
) -> UniquePtr<MlxArray> {
    let launch = SparseAttentionLaunch { parts, simdgroups };
    sparse_attention_with(
        q,
        k,
        v,
        &sel.lists_mlx(),
        &lengths_mlx(sel),
        scale(),
        launch,
    )
    .unwrap()
}

fn rows_of(q: &MlxArray, r0: usize, r1: usize) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(q);
    mlxcel_core::slice(q, &[0, 0, r0 as i32, 0], &[s[0], s[1], r1 as i32, s[3]])
}

fn raw(x: &MlxArray) -> Vec<u8> {
    mlxcel_core::eval(x);
    mlxcel_core::array_to_raw_bytes(x)
}

/// f64 reference of row `r` of `q` (host copy `qh`, `rows` rows) over `list`.
fn host_ref(
    qh: &[f32],
    rows: usize,
    r: usize,
    k: &MlxArray,
    v: &MlxArray,
    list: &[i32],
) -> Vec<f32> {
    let idx = mlxcel_core::from_slice_i32(list, &[list.len() as i32]);
    let kl = to_host_f32(&mlxcel_core::take(k, &idx, 2));
    let vl = to_host_f32(&mlxcel_core::take(v, &idx, 2));
    host_rows(qh, rows, r, &kl, &vl, list.len())
}

/// Row `r` of a `[1, HQ, rows, D]` host output, heads concatenated.
fn row_of(out: &[f32], rows: usize, r: usize) -> Vec<f32> {
    (0..HQ as usize)
        .flat_map(|h| out[(h * rows + r) * D as usize..][..D as usize].to_vec())
        .collect()
}

/// One f16 ulp at 2.0 is 2e-3; a correctly rounded f16 output of a value below 2 is within half
/// of that of the f64 result. K2 accumulates in f32, so it stays near that bound.
const F16_OUT_TOL: f32 = 1.5e-3;

/// fp16 K2 vs task 3's reads (masked SDPA, and the gathered decode read row by row), both
/// measured against an f64 host softmax on sampled rows.
#[test]
fn fp16_matches_task3_reads() {
    if !gpu() {
        return;
    }
    let t = 6000usize;
    let k = randn(&[1, HKV, t as i32, D], dtype::FLOAT16, 11, 1.0);
    let v = randn(&[1, HKV, t as i32, D], dtype::FLOAT16, 12, 1.0);
    let (mut k2_host, mut task3_host, mut k2_task3) = (0f32, 0f32, 0f32);
    for rows in [1usize, 3, 8, 37, 1536] {
        // Peaked scores (std ~3), so the read is not a near-uniform average
        let q = randn(
            &[1, HQ, rows as i32, D],
            dtype::FLOAT16,
            13 + rows as u64,
            3.0,
        );
        let qh = to_host_f32(&q);
        let start = t - rows;
        let sel = selection(start, rows, CFG.top_k, Expansion::Independent, rows as u64);
        let got = to_host_f32(&k2(&q, &k, &v, &sel));
        if rows > 1 {
            let want = to_host_f32(&masked_attention(
                &q,
                &k,
                &v,
                &sel,
                start,
                MASK_ROWS,
                scale(),
            ));
            let diff = max_abs_diff(&got, &want);
            eprintln!("K2 vs masked_attention, L={rows}: max abs {diff:e} (all rows)");
            k2_task3 = k2_task3.max(diff);
            for r in [0, rows / 2, rows - 1] {
                let href = host_ref(&qh, rows, r, &k, &v, sel.list(r));
                task3_host = task3_host.max(max_abs_diff(&row_of(&want, rows, r), &href));
            }
        }
        for r in [0, rows / 2, rows - 1] {
            let href = host_ref(&qh, rows, r, &k, &v, sel.list(r));
            let qr = rows_of(&q, r, r + 1);
            let gathered = to_host_f32(&gather_attention(&qr, &k, &v, sel.list(r), scale()));
            let got_r = to_host_f32(&k2(
                &qr,
                &k,
                &v,
                &DlsaSelection {
                    rows: 1,
                    lists: sel.lists[r * sel.width..(r + 1) * sel.width].to_vec(),
                    lengths: vec![sel.lengths[r]],
                    ..sel.clone()
                },
            ));
            let in_block = row_of(&got, rows, r);
            k2_task3 = k2_task3.max(max_abs_diff(&got_r, &gathered));
            task3_host = task3_host.max(max_abs_diff(&gathered, &href));
            k2_host = k2_host
                .max(max_abs_diff(&got_r, &href))
                .max(max_abs_diff(&in_block, &href));
        }
    }
    eprintln!(
        "fp16: K2 vs f64 host {k2_host:e}, task 3 vs f64 host {task3_host:e}, K2 vs task 3 {k2_task3:e}"
    );
    assert!(k2_host < F16_OUT_TOL, "fp16 K2 vs host: {k2_host:e}");
    // task 3's masked SDPA materializes fp16 scores, so its own error sets the gap
    assert!(k2_task3 < 1e-2, "fp16 K2 vs task 3: {k2_task3:e}");
}

/// Lists covering every causal key (in any order) equal dense causal attention.
#[test]
fn full_coverage_equals_dense_causal() {
    if !gpu() {
        return;
    }
    let t = 900usize;
    for (dt, tol) in [(dtype::FLOAT32, 1e-5f32), (dtype::FLOAT16, 1e-2)] {
        let k = randn(&[1, HKV, t as i32, D], dt, 21, 1.0);
        let v = randn(&[1, HKV, t as i32, D], dt, 22, 1.0);
        for rows in [1usize, 3, 37] {
            let q = randn(&[1, HQ, rows as i32, D], dt, 23, 3.0);
            let qh = to_host_f32(&q);
            let start = t - rows;
            // top_k larger than any pool: every causal key, the expansion shuffled
            let sel = selection(start, rows, t, Expansion::Independent, 5);
            for r in 0..rows {
                assert_eq!(sel.lengths[r] as usize, start + r + 1);
            }
            let dense = to_host_f32(&mlxcel_core::causal_attention(&q, &k, &v, scale(), 0.0, 0));
            for (parts, sg) in [(0, 0), (1, 8), (16, 4)] {
                let got = to_host_f32(&k2_with(&q, &k, &v, &sel, parts, sg));
                let diff = max_abs_diff(&got, &dense);
                let mut vs_host = 0f32;
                for r in [0, rows - 1] {
                    let href = host_ref(&qh, rows, r, &k, &v, sel.list(r));
                    vs_host = vs_host.max(max_abs_diff(&row_of(&got, rows, r), &href));
                }
                eprintln!(
                    "full coverage dtype {dt} L={rows} parts={parts}: vs dense {diff:e}, vs f64 host {vs_host:e}"
                );
                assert!(
                    diff < tol,
                    "full coverage dtype {dt} L={rows} parts={parts}: {diff:e}"
                );
                let host_tol = if dt == dtype::FLOAT32 {
                    1e-5
                } else {
                    F16_OUT_TOL
                };
                assert!(
                    vs_host < host_tol,
                    "dtype {dt} L={rows} parts={parts} vs host: {vs_host:e}"
                );
            }
        }
    }
}

/// Length-1 lists, `-1` pads, a list entry repeated in the padding, and rows of different lengths
/// in one launch.
#[test]
fn short_and_padded_lists() {
    if !gpu() {
        return;
    }
    let t = 64usize;
    let k = randn(&[1, HKV, t as i32, D], dtype::FLOAT32, 31, 1.0);
    let v = randn(&[1, HKV, t as i32, D], dtype::FLOAT32, 32, 1.0);
    let q = randn(&[1, HQ, 3, D], dtype::FLOAT32, 33, 1.0);
    let width = 6;
    let lists = vec![
        17, -1, -1, -1, -1, -1, 63, 2, 40, -1, -1, -1, 5, 9, 1, 0, 33, 62,
    ];
    let sel = DlsaSelection {
        rows: 3,
        width,
        top_p: 1,
        lists,
        lengths: vec![1, 3, 6],
        picked: vec![-1; 3],
        visible: vec![0; 3],
    };
    let vh = to_host_f32(&v);
    for (parts, sg) in [(0, 0), (1, 1), (4, 2)] {
        let got = to_host_f32(&k2_with(&q, &k, &v, &sel, parts, sg));
        // A single key reads its value row exactly
        for h in 0..HQ as usize {
            let kh = h / (HQ / HKV) as usize;
            let want = &vh[(kh * t + 17) * D as usize..][..D as usize];
            let row = &got[(h * 3) * D as usize..][..D as usize];
            assert!(max_abs_diff(row, want) < 1e-6, "length-1 list, head {h}");
        }
        for r in 1..3 {
            let qr = rows_of(&q, r, r + 1);
            let want = to_host_f32(&gather_attention(&qr, &k, &v, sel.list(r), scale()));
            let got_r: Vec<f32> = (0..HQ as usize)
                .flat_map(|h| got[(h * 3 + r) * D as usize..][..D as usize].to_vec())
                .collect();
            let diff = max_abs_diff(&got_r, &want);
            assert!(diff < 1e-5, "row {r} parts={parts}: {diff:e}");
        }
    }
}

/// The cache's fetched view (a strided slice of its preallocated buffer, after several appends)
/// reads bitwise the same as the same data made contiguous, under every launch.
#[test]
fn strided_cache_view_equals_contiguous_bitwise() {
    if !gpu() {
        return;
    }
    let mut cache = KVCache::new_with_mode(KVCacheMode::Fp16);
    let mut fetched = None;
    for (i, n) in [300i32, 17, 1, 250, 1].into_iter().enumerate() {
        let k = randn(&[1, HKV, n, D], dtype::FLOAT16, 40 + i as u64, 1.0);
        let v = randn(&[1, HKV, n, D], dtype::FLOAT16, 50 + i as u64, 1.0);
        fetched = Some(cache.update_and_fetch(k, v));
    }
    let (ck, cv) = fetched.unwrap();
    let t = mlxcel_core::array_shape(&ck)[2] as usize;
    assert_eq!(t, 569);
    let kc = mlxcel_core::contiguous(
        &mlxcel_core::add(&ck, &mlxcel_core::zeros(&[1], dtype::FLOAT16)),
        false,
    );
    let vc = mlxcel_core::contiguous(
        &mlxcel_core::add(&cv, &mlxcel_core::zeros(&[1], dtype::FLOAT16)),
        false,
    );
    assert_eq!(
        raw(&ck),
        raw(&kc),
        "the view and its copy hold the same keys"
    );
    for rows in [1usize, 8, 37] {
        let q = randn(&[1, HQ, rows as i32, D], dtype::FLOAT16, 60, 3.0);
        let sel = selection(t - rows, rows, 40, Expansion::Independent, 7);
        for (parts, sg) in [(0, 0), (1, 8), (8, 4)] {
            let a = raw(&k2_with(&q, &ck, &cv, &sel, parts, sg));
            let b = raw(&k2_with(&q, &kc, &vc, &sel, parts, sg));
            assert!(
                a == b,
                "strided vs contiguous differ, L={rows} parts={parts}"
            );
        }
    }
}

/// f64 softmax of row `r` over `list`, with the listed K/V rows (`[Hkv, n, D]`, host).
fn host_rows(q: &[f32], rows: usize, r: usize, kl: &[f32], vl: &[f32], n: usize) -> Vec<f32> {
    let (g, d) = ((HQ / HKV) as usize, D as usize);
    let mut out = vec![0f32; HQ as usize * d];
    for h in 0..HQ as usize {
        let kh = h / g;
        let qr = &q[(h * rows + r) * d..][..d];
        let s: Vec<f64> = (0..n)
            .map(|j| {
                let kr = &kl[(kh * n + j) * d..][..d];
                qr.iter()
                    .zip(kr)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * scale() as f64
            })
            .collect();
        let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
        let z: f64 = w.iter().sum();
        for e in 0..d {
            let acc: f64 = (0..n).map(|j| w[j] * vl[(kh * n + j) * d + e] as f64).sum();
            out[h * d + e] = (acc / z) as f32;
        }
    }
    out
}

/// T = 262,144 through a cache-like view (a slice of a larger buffer): the index arithmetic of
/// the last head's last keys, for one pass and split-K.
#[test]
fn index_arithmetic_at_262144() {
    if !gpu() {
        return;
    }
    let t = 262_144usize;
    let cap = t as i32 + 512;
    let kbuf = randn(&[1, HKV, cap, D], dtype::FLOAT16, 71, 1.0);
    let vbuf = randn(&[1, HKV, cap, D], dtype::FLOAT16, 72, 1.0);
    let k = mlxcel_core::slice(&kbuf, &[0, 0, 0, 0], &[1, HKV, t as i32, D]);
    let v = mlxcel_core::slice(&vbuf, &[0, 0, 0, 0], &[1, HKV, t as i32, D]);
    let rows = 3usize;
    let q = randn(&[1, HQ, rows as i32, D], dtype::FLOAT16, 73, 3.0);
    let qh = to_host_f32(&q);
    let sel = selection(t - rows, rows, CFG.top_k, Expansion::Independent, 9);
    for (parts, sg) in [(0, 0), (1, 8), (64, 4)] {
        let got = to_host_f32(&k2_with(&q, &k, &v, &sel, parts, sg));
        let mut worst = 0f32;
        for r in 0..rows {
            let list = sel.list(r);
            let idx = mlxcel_core::from_slice_i32(list, &[list.len() as i32]);
            let kl = to_host_f32(&mlxcel_core::take(&k, &idx, 2));
            let vl = to_host_f32(&mlxcel_core::take(&v, &idx, 2));
            let want = host_rows(&qh, rows, r, &kl, &vl, list.len());
            let got_r: Vec<f32> = (0..HQ as usize)
                .flat_map(|h| got[(h * rows + r) * D as usize..][..D as usize].to_vec())
                .collect();
            worst = worst.max(max_abs_diff(&got_r, &want));
        }
        eprintln!("T=262144 parts={parts}: max abs vs f64 host {worst:e}");
        assert!(worst < 2e-3, "T=262144 parts={parts}: {worst:e}");
    }
}

fn median_ms(reps: usize, mut f: impl FnMut() -> UniquePtr<MlxArray>) -> f64 {
    mlxcel_core::eval(&f());
    let mut ms: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            let out = f();
            mlxcel_core::eval(&out);
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse().expect("a number"))
        .collect()
}

/// K2 vs task 3 at the Qwen geometry: one 1536-row prefill chunk at depth T (masked SDPA vs K2),
/// and decode / verify rows (take + SDPA vs K2 under several split-K settings). Prints a table
/// and, with `QW_K2_BENCH_OUT`, writes JSON lines there. `QW_K2_BENCH_PHASES` (default
/// `prefill,decode`) picks phases; `QW_K2_BENCH_SKIP_MASKED=1` skips the slow task-3 and dense
/// baselines.
#[test]
#[ignore = "GPU microbenchmark; run under gpu-lock with --ignored --nocapture"]
fn bench() {
    if !gpu() {
        return;
    }
    let contexts = env_list("QW_K2_BENCH_CONTEXTS", "32768,131072,262144");
    let skip_masked = std::env::var("QW_K2_BENCH_SKIP_MASKED").is_ok_and(|v| v == "1");
    let phases = std::env::var("QW_K2_BENCH_PHASES").unwrap_or_else(|_| "prefill,decode".into());
    let mut json = Vec::new();
    let mut record = |line: String| {
        eprintln!("{line}");
        json.push(line);
    };
    for &t in &contexts {
        // A cache-like view: a slice of a larger buffer
        let cap = t as i32 + 256;
        let kbuf = randn(&[1, HKV, cap, D], dtype::FLOAT16, 81, 1.0);
        let vbuf = randn(&[1, HKV, cap, D], dtype::FLOAT16, 82, 1.0);
        let k = mlxcel_core::slice(&kbuf, &[0, 0, 0, 0], &[1, HKV, t as i32, D]);
        let v = mlxcel_core::slice(&vbuf, &[0, 0, 0, 0], &[1, HKV, t as i32, D]);

        // Prefill: one 1536-row chunk ending at depth t
        let rows = 1536usize;
        let q = randn(&[1, HQ, rows as i32, D], dtype::FLOAT16, 83, 1.0);
        let prefill_lists: &[Expansion] = if phases.contains("prefill") {
            &[Expansion::Independent, Expansion::Shared]
        } else {
            &[]
        };
        for &exp in prefill_lists {
            let sel = selection(t - rows, rows, CFG.top_k, exp, 3);
            let (lists, lens) = (sel.lists_mlx(), lengths_mlx(&sel));
            mlxcel_core::eval(&lists);
            mlxcel_core::eval(&lens);
            for sg in [2, 4, 8, 16] {
                let launch = SparseAttentionLaunch {
                    parts: 1,
                    simdgroups: sg,
                };
                let ms = median_ms(5, || {
                    sparse_attention_with(&q, &k, &v, &lists, &lens, scale(), launch).unwrap()
                });
                record(format!(
                    "{{\"t\":{t},\"phase\":\"prefill\",\"rows\":{rows},\"lists\":\"{exp:?}\",\"impl\":\"k2\",\"parts\":1,\"simdgroups\":{sg},\"ms\":{ms:.3}}}"
                ));
            }
            let ms = median_ms(5, || sparse_attention(&q, &k, &v, &lists, &lens, scale()));
            record(format!(
                "{{\"t\":{t},\"phase\":\"prefill\",\"rows\":{rows},\"lists\":\"{exp:?}\",\"impl\":\"k2_auto\",\"ms\":{ms:.3}}}"
            ));
            if !skip_masked && exp == Expansion::Independent {
                let ms = median_ms(2, || {
                    masked_attention(&q, &k, &v, &sel, t - rows, MASK_ROWS, scale())
                });
                record(format!(
                    "{{\"t\":{t},\"phase\":\"prefill\",\"rows\":{rows},\"lists\":\"{exp:?}\",\"impl\":\"task3_masked\",\"ms\":{ms:.3}}}"
                ));
                let ms = median_ms(2, || {
                    mlxcel_core::causal_attention(&q, &k, &v, scale(), 0.0, 0)
                });
                record(format!(
                    "{{\"t\":{t},\"phase\":\"prefill\",\"rows\":{rows},\"impl\":\"dense_causal\",\"ms\":{ms:.3}}}"
                ));
            }
        }

        // Decode and verify rows
        let decode_rows: &[usize] = if phases.contains("decode") {
            &[1, 3, 8]
        } else {
            &[]
        };
        for &rows in decode_rows {
            let q = randn(&[1, HQ, rows as i32, D], dtype::FLOAT16, 84, 1.0);
            let sel = selection(t - rows, rows, CFG.top_k, Expansion::Independent, 4);
            let (lists, lens) = (sel.lists_mlx(), lengths_mlx(&sel));
            mlxcel_core::eval(&lists);
            mlxcel_core::eval(&lens);
            for (parts, sg) in [
                (1, 8),
                (8, 4),
                (16, 4),
                (32, 4),
                (64, 4),
                (32, 2),
                (64, 2),
                (128, 2),
            ] {
                let launch = SparseAttentionLaunch {
                    parts,
                    simdgroups: sg,
                };
                let ms = median_ms(25, || {
                    sparse_attention_with(&q, &k, &v, &lists, &lens, scale(), launch).unwrap()
                });
                record(format!(
                    "{{\"t\":{t},\"phase\":\"decode\",\"rows\":{rows},\"impl\":\"k2\",\"parts\":{parts},\"simdgroups\":{sg},\"ms\":{ms:.4}}}"
                ));
            }
            let ms = median_ms(25, || sparse_attention(&q, &k, &v, &lists, &lens, scale()));
            record(format!(
                "{{\"t\":{t},\"phase\":\"decode\",\"rows\":{rows},\"impl\":\"k2_auto\",\"ms\":{ms:.4}}}"
            ));
            if rows == 1 {
                // task 3's decode read, including its host sort and list upload
                let list = sel.list(0).to_vec();
                let ms = median_ms(25, || gather_attention(&q, &k, &v, &list, scale()));
                record(format!(
                    "{{\"t\":{t},\"phase\":\"decode\",\"rows\":1,\"impl\":\"task3_take_sdpa\",\"ms\":{ms:.4}}}"
                ));
                let ms = median_ms(25, || {
                    mlxcel_core::causal_attention(&q, &k, &v, scale(), 0.0, 0)
                });
                record(format!(
                    "{{\"t\":{t},\"phase\":\"decode\",\"rows\":1,\"impl\":\"dense\",\"ms\":{ms:.4}}}"
                ));
            } else if !skip_masked {
                let ms = median_ms(5, || {
                    masked_attention(&q, &k, &v, &sel, t - rows, MASK_ROWS, scale())
                });
                record(format!(
                    "{{\"t\":{t},\"phase\":\"decode\",\"rows\":{rows},\"impl\":\"task3_masked\",\"ms\":{ms:.4}}}"
                ));
            }
        }
    }
    if let Ok(path) = std::env::var("QW_K2_BENCH_OUT") {
        std::fs::write(&path, json.join("\n") + "\n").expect("write the bench output");
    }
}
