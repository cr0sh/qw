// SPDX-License-Identifier: Apache-2.0
//! K1 `dlsa_select` (mlxcel-core `dlsa_kernels`) against task 2's host pick and expansion
//! (`select_from_scores`): on the same scores, lists, lengths and picked buckets must be
//! bit-identical, in order (`docs/dlsa.md` §14.2).
//!
//! The scores are one MLX f32 matmul against the centroids, as `score_mlx` runs it; the
//! kernel reads that device array and the host reads it back, so both see the same bits.
//!
//! Metal only: run under `./gpu-lock -- cargo test -p qw-runtime --features dlsa --lib
//! gpu_select`. `bench_gpu_select_vs_host` is `#[ignore]`d (a microbench).

use std::time::{Duration, Instant};

use mlxcel_core::dlsa_kernels::{DlsaSelectOut, DlsaSelectParams, dlsa_select};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::fixtures::Fixture;
use super::*;

// ------------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------------

fn params(cfg: &DlsaIndexConfig) -> DlsaSelectParams {
    DlsaSelectParams {
        window: cfg.window_size as i32,
        sink: cfg.sink_size as i32,
        top_k: cfg.top_k as i32,
        top_p: cfg.top_p as i32,
        bucket_cap: cfg.bucket_cap as i32,
        buckets_per_block: cfg.buckets_per_block() as i32,
    }
}

fn host_i32(a: &MlxArray) -> Vec<i32> {
    mlxcel_core::eval(a);
    mlxcel_core::array_to_raw_bytes(a)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect()
}

fn positions_mlx(positions: &[usize]) -> UniquePtr<MlxArray> {
    let p: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    mlxcel_core::from_slice_i32(&p, &[p.len() as i32])
}

fn members_mlx(cache: &DlsaBucketCache) -> UniquePtr<MlxArray> {
    let cap = cache.config().bucket_cap;
    mlxcel_core::from_slice_i32(cache.members(), &[cache.rows() as i32, cap as i32])
}

/// `[R, stride]` f32 scores of `queries` (`[R, d]` f32 host) against the first `stride`
/// centroids, as one MLX matmul (the op of `score_mlx`), on the device.
fn device_scores(cache: &DlsaBucketCache, queries: &[f32], stride: usize) -> UniquePtr<MlxArray> {
    let d = cache.config().index_dim;
    let rows = queries.len() / d;
    if stride == 0 {
        return mlxcel_core::from_slice_f32(&[], &[rows as i32, 0]);
    }
    let q = mlxcel_core::from_slice_f32(queries, &[rows as i32, d as i32]);
    let c =
        mlxcel_core::from_slice_f32(&cache.centroids()[..stride * d], &[stride as i32, d as i32]);
    let q = mlxcel_core::astype(&q, dtype::FLOAT32);
    let s = mlxcel_core::matmul(&q, &mlxcel_core::transpose(&c));
    mlxcel_core::eval(&s);
    s
}

fn run_kernel(cache: &DlsaBucketCache, scores: &MlxArray, positions: &[usize]) -> DlsaSelectOut {
    dlsa_select(
        scores,
        &members_mlx(cache),
        &positions_mlx(positions),
        cache.rows() as i32,
        &params(cache.config()),
    )
    .expect("dlsa_select")
}

/// The kernel on `scores` (device) == `select_from_scores` on the same scores read back.
#[track_caller]
fn assert_kernel_matches_host(
    cache: &DlsaBucketCache,
    scores: &MlxArray,
    positions: &[usize],
    what: &str,
) -> DlsaSelection {
    let stride = mlxcel_core::array_shape(scores)[1] as usize;
    let host_scores = to_host_f32(scores);
    let host = cache
        .select_from_scores(&host_scores, stride, positions)
        .unwrap();
    let out = run_kernel(cache, scores, positions);
    let lists = host_i32(&out.lists);
    let lengths = host_i32(&out.lengths);
    let picked = host_i32(&out.picked);
    let status = host_i32(&out.status);
    for (r, &p) in positions.iter().enumerate() {
        assert_eq!(
            &picked[r * host.top_p..(r + 1) * host.top_p],
            &host.picked[r * host.top_p..(r + 1) * host.top_p],
            "{what}: picked of row {r} (p {p})"
        );
        assert_eq!(
            lengths[r], host.lengths[r],
            "{what}: length of row {r} (p {p})"
        );
        assert_eq!(
            &lists[r * host.width..(r + 1) * host.width],
            &host.lists[r * host.width..(r + 1) * host.width],
            "{what}: list of row {r} (p {p})"
        );
    }
    assert!(status.iter().all(|&s| s == 0), "{what}: status {status:?}");
    host
}

/// The kernel == host on host-made scores `[R, stride]` (ties and ±0 made on purpose).
#[track_caller]
fn assert_on_scores(cache: &DlsaBucketCache, scores: &[f32], positions: &[usize], what: &str) {
    let stride = cache.max_visible(positions);
    let s = mlxcel_core::from_slice_f32(scores, &[positions.len() as i32, stride as i32]);
    assert_kernel_matches_host(cache, &s, positions, what);
}

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// `n` unit vectors of `d` dims in topical runs (as `indexer_tests.rs` makes them);
/// `quantized` draws components from {-1, 0, 1}, so scores are exact in any summation order
/// and full of exact ties.
fn synthetic_vectors(rng: &mut Rng, n: usize, d: usize, quantized: bool) -> Vec<f32> {
    let topics: Vec<Vec<f32>> = (0..6)
        .map(|_| (0..d).map(|_| rng.normal()).collect())
        .collect();
    let mut out = Vec::with_capacity(n * d);
    let mut topic = 0;
    for i in 0..n {
        if i == 0 || rng.below(4) == 0 {
            topic = rng.below(topics.len());
        }
        let mut v: Vec<f32> = if quantized {
            (0..d).map(|_| rng.below(3) as f32 - 1.0).collect()
        } else {
            topics[topic]
                .iter()
                .map(|c| 0.8 * c + 0.6 * rng.normal())
                .collect()
        };
        if v.iter().all(|&x| x == 0.0) {
            v[0] = 1.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(NORM_EPS);
        out.extend(v.iter().map(|x| x / norm));
    }
    out
}

fn random_config(rng: &mut Rng) -> DlsaIndexConfig {
    let pool_size = [1, 2, 4][rng.below(3)];
    let bucket_size = pool_size * rng.range(1, 3);
    let window_size = bucket_size * rng.range(1, 6);
    let bucket_cap = pool_size * (2 * bucket_size / pool_size + rng.range(0, 3));
    let top_p = rng.range(1, 8);
    let cfg = DlsaIndexConfig {
        window_size,
        sink_size: rng.range(0, 4),
        top_k: rng.range(1, top_p * bucket_cap),
        top_p,
        index_dim: rng.range(2, 8),
        pool_size,
        bucket_size,
        bucket_cap,
    };
    cfg.validate().unwrap();
    cfg
}

fn rows_of(data: &[f32], d: usize, positions: &[usize]) -> Vec<f32> {
    positions
        .iter()
        .flat_map(|&p| data[p * d..(p + 1) * d].to_vec())
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Fixtures
// ------------------------------------------------------------------------------------------------

/// Cases with read lists, and the tensor prefix of each map (as `indexer_tests.rs`).
const LIST_CASES: &[(&str, &str)] = &[
    ("tiny", ""),
    ("tiny_ties", ""),
    ("short_dense", ""),
    ("qwen_layer", "random."),
    ("qwen_layer", "real."),
];

/// Every fixture case run as a forward would: prefill chunks (append, select the chunk's
/// rows, commit), then one key a step; `qwen_index_defaults` grows to each sparse position.
/// Each select: the kernel == the host on the same matmul scores, and both == the fixture.
#[test]
fn fixtures_kernel_equals_host() {
    let mut cases: Vec<(Fixture, &str)> = LIST_CASES
        .iter()
        .map(|(n, p)| (Fixture::load(n), *p))
        .collect();
    cases.push((Fixture::load("qwen_index_defaults"), ""));
    for (f, prefix) in cases {
        let cfg = f.cfg();
        let d = cfg.index_dim;
        let keys = f.f32(&format!("{prefix}index_keys"));
        let queries = f.f32(&format!("{prefix}index_queries"));
        let selected = f.i32(&format!("{prefix}selected"));
        let width = cfg.list_width();
        let what = format!("{} {prefix}", f.name);
        let mut cache = DlsaBucketCache::new(cfg).unwrap();
        if f.name == "qwen_index_defaults" {
            let positions: Vec<usize> = f.i32("positions").iter().map(|&p| p as usize).collect();
            for (r, &p) in positions.iter().enumerate() {
                let at = cache.len();
                cache.append(&keys[at * d..(p + 1) * d]).unwrap();
                cache.commit();
                let q = &queries[r * d..(r + 1) * d];
                let s = device_scores(&cache, q, cache.max_visible(&[p]));
                let host = assert_kernel_matches_host(&cache, &s, &[p], &format!("{what} p {p}"));
                assert_eq!(&host.lists[..], &selected[r * width..(r + 1) * width]);
            }
            continue;
        }
        let t = f.key_count();
        let prefill = f.prefill_len().unwrap();
        let chunk = 2 * cfg.window_size + 3;
        let mut at = 0;
        while at < t {
            let step = if at < prefill {
                chunk.min(prefill - at)
            } else {
                1
            };
            cache.append(&keys[at * d..(at + step) * d]).unwrap();
            let rows: Vec<usize> = (at..at + step).collect();
            let s = device_scores(
                &cache,
                &queries[at * d..(at + step) * d],
                cache.max_visible(&rows),
            );
            let host =
                assert_kernel_matches_host(&cache, &s, &rows, &format!("{what} rows {at}.."));
            assert_eq!(&host.lists[..], &selected[at * width..(at + step) * width]);
            cache.commit();
            at += step;
        }
    }
}

/// The fixtures' own reference scores (not recomputed): the kernel == the host.
#[test]
fn fixture_reference_scores_kernel_equals_host() {
    for (name, prefix) in LIST_CASES {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        let keys = f.f32(&format!("{prefix}index_keys"));
        let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
        let positions: Vec<usize> = (0..f.key_count()).collect();
        let scores = f.f32(&format!("{prefix}bucket_scores"));
        let rows = cache.rows();
        let stride = cache.max_visible(&positions);
        let s: Vec<f32> = (0..positions.len())
            .flat_map(|r| scores[r * rows..r * rows + stride].to_vec())
            .collect();
        assert_on_scores(&cache, &s, &positions, &format!("{name} {prefix}"));
    }
}

// ------------------------------------------------------------------------------------------------
// Seeded random configs
// ------------------------------------------------------------------------------------------------

/// 72 random flat configs (a third with {-1,0,1} vectors), grown in random chunks; every
/// chunk's rows are selected after its append, as a forward does.
#[test]
fn random_configs_kernel_equals_host() {
    for seed in 0..72u64 {
        let mut rng = Rng(seed);
        let cfg = random_config(&mut rng);
        let d = cfg.index_dim;
        let n = rng.range(1, 6 * cfg.window_size + 5);
        let quantized = seed % 3 == 0;
        let keys = synthetic_vectors(&mut rng, n, d, quantized);
        let queries = synthetic_vectors(&mut rng, n, d, quantized);
        let mut cache = DlsaBucketCache::new(cfg).unwrap();
        let mut at = 0;
        while at < n {
            let step = rng.range(1, 2 * cfg.window_size + 3).min(n - at);
            cache.append(&keys[at * d..(at + step) * d]).unwrap();
            let rows: Vec<usize> = (at..at + step).collect();
            let s = device_scores(
                &cache,
                &queries[at * d..(at + step) * d],
                cache.max_visible(&rows),
            );
            assert_kernel_matches_host(&cache, &s, &rows, &format!("seed {seed} {cfg:?} at {at}"));
            cache.commit();
            at += step;
        }
    }
}

/// Host-made scores full of exact ties: {-1, 0, 1}, ±0 only, one value, and a few values,
/// so the bucket-index tie rule and `-0.0 == 0.0` decide most picks.
#[test]
fn tie_heavy_scores_kernel_equals_host() {
    let cfg = DlsaIndexConfig {
        top_k: 96,
        top_p: 24,
        ..DlsaIndexConfig::QWEN_DEFAULTS
    };
    let mut rng = Rng(9);
    let n = 12 * 512 + 77;
    let keys = synthetic_vectors(&mut rng, n, 64, false);
    let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
    let positions: Vec<usize> = [0, 511, 512, 1023, 1024, 4000, 6143, 6144, n - 1]
        .into_iter()
        .chain((0..40).map(|_| rng.below(n)))
        .collect();
    let stride = cache.max_visible(&positions);
    let draws: [&[f32]; 4] = [
        &[-1.0, 0.0, 1.0],
        &[-0.0, 0.0],
        &[0.5],
        &[0.25, -0.0, 0.0, 0.75],
    ];
    for (k, vals) in draws.iter().enumerate() {
        let s: Vec<f32> = (0..positions.len() * stride)
            .map(|_| vals[rng.below(vals.len())])
            .collect();
        assert_on_scores(&cache, &s, &positions, &format!("draw {k} {vals:?}"));
    }
    // The same with the full defaults (top_p 1024 of 768 visible: every bucket a candidate)
    let cache = DlsaBucketCache::build(DlsaIndexConfig::QWEN_DEFAULTS, &keys).unwrap();
    let s: Vec<f32> = (0..positions.len() * stride)
        .map(|_| [-0.0, 0.0, 1.0][rng.below(3)])
        .collect();
    assert_on_scores(&cache, &s, &positions, "defaults ±0");
}

/// top_k binding (few long-range keys), top_p binding (top_p * cap == top_k), V <= top_p and
/// V > top_p, block boundaries, the window overlapping the last closed block.
#[test]
fn binding_knobs_and_block_edges() {
    let base = DlsaIndexConfig::QWEN_DEFAULTS;
    let mut rng = Rng(21);
    let n = 9 * 512 + 300;
    let keys = synthetic_vectors(&mut rng, n, 64, false);
    let queries = synthetic_vectors(&mut rng, n, 64, false);
    let positions: Vec<usize> = (0..9)
        .flat_map(|b| [b * 512, b * 512 + 1, b * 512 + 511])
        .chain([n - 1, 4611, 4612, 4613])
        .collect();
    for (k, cfg) in [
        base,
        DlsaIndexConfig { top_k: 40, ..base },
        DlsaIndexConfig {
            top_p: 64,
            top_k: 1024,
            ..base
        },
        DlsaIndexConfig {
            top_p: 100,
            top_k: 1600,
            ..base
        },
        DlsaIndexConfig {
            top_p: 2048,
            ..base
        },
        DlsaIndexConfig {
            top_p: 1,
            top_k: 16,
            sink_size: 0,
            ..base
        },
    ]
    .into_iter()
    .enumerate()
    {
        let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
        let s = device_scores(
            &cache,
            &rows_of(&queries, 64, &positions),
            cache.max_visible(&positions),
        );
        assert_kernel_matches_host(&cache, &s, &positions, &format!("cfg {k} {cfg:?}"));
    }
}

/// No closed block yet: no buckets, S = 0, the kernel reads only the window and the sinks.
#[test]
fn empty_cache_kernel_equals_host() {
    let cfg = DlsaIndexConfig::QWEN_DEFAULTS;
    let mut rng = Rng(3);
    let keys = synthetic_vectors(&mut rng, 300, 64, false);
    let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
    assert_eq!(cache.rows(), 0);
    let positions: Vec<usize> = vec![0, 1, 3, 4, 5, 299];
    let s = device_scores(&cache, &rows_of(&keys, 64, &positions), 0);
    assert_kernel_matches_host(&cache, &s, &positions, "empty cache");
}

// ------------------------------------------------------------------------------------------------
// Qwen defaults at depth
// ------------------------------------------------------------------------------------------------

/// A Qwen-defaults cache over `t` synthetic topical keys, 64 sampled rows (the first and
/// last rows of blocks, the last row of the context, random rows): kernel == host.
fn qwen_at_depth(t: usize, seed: u64) {
    let cfg = DlsaIndexConfig::QWEN_DEFAULTS;
    let mut rng = Rng(seed);
    let keys = synthetic_vectors(&mut rng, t, 64, false);
    let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
    let blocks = t / 512;
    let mut positions = vec![t - 1, 0, 511, 512];
    for _ in 0..10 {
        let b = rng.below(blocks.max(1));
        positions.push(b * 512);
        positions.push((b * 512 + 511).min(t - 1));
    }
    while positions.len() < 64 {
        positions.push(rng.below(t));
    }
    let queries = synthetic_vectors(&mut rng, positions.len(), 64, false);
    let s = device_scores(&cache, &queries, cache.max_visible(&positions));
    assert_kernel_matches_host(&cache, &s, &positions, &format!("qwen T {t}"));
}

#[test]
fn qwen_defaults_6k_and_32k() {
    qwen_at_depth(6000, 60);
    qwen_at_depth(32_768, 61);
}

#[test]
fn qwen_defaults_131k_and_262k() {
    qwen_at_depth(131_072, 62);
    qwen_at_depth(262_144, 63);
}

// ------------------------------------------------------------------------------------------------
// Microbench
// ------------------------------------------------------------------------------------------------

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// The kernel (scores already on device; lists eval'd) vs the host path it replaces (scores
/// readback + `select_from_scores`), at Qwen defaults, L = 1536 and L = 1, at
/// `QW_DLSA_SELECT_BENCH_CONTEXTS` (default 32768,131072,262144).
#[test]
#[ignore = "microbench; run with --ignored --nocapture under ./gpu-lock"]
fn bench_gpu_select_vs_host() {
    let contexts: Vec<usize> = std::env::var("QW_DLSA_SELECT_BENCH_CONTEXTS")
        .unwrap_or_else(|_| "32768,131072,262144".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let cfg = DlsaIndexConfig::QWEN_DEFAULTS;
    for t in contexts {
        let mut rng = Rng(t as u64);
        let keys = synthetic_vectors(&mut rng, t, 64, false);
        let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
        for l in [1536usize, 1] {
            // The last L rows of the context (a prefill chunk at depth / a decode step)
            let positions: Vec<usize> = (t - l..t).collect();
            let queries = synthetic_vectors(&mut rng, l, 64, false);
            let stride = cache.max_visible(&positions);
            let s = device_scores(&cache, &queries, stride);
            let members = members_mlx(&cache);
            let pos = positions_mlx(&positions);
            mlxcel_core::eval(&members);
            mlxcel_core::eval(&pos);
            let p = params(&cfg);
            let reps = if l == 1 { 25 } else { 5 };
            let mut kernel = Vec::new();
            for _ in 0..reps + 1 {
                let start = Instant::now();
                let out = dlsa_select(&s, &members, &pos, cache.rows() as i32, &p).unwrap();
                mlxcel_core::eval(&out.lists);
                mlxcel_core::eval(&out.lengths);
                kernel.push(start.elapsed());
            }
            kernel.remove(0);
            // GPU time without the per-launch sync: `batch` launches in one eval
            let batch = if l == 1 { 64 } else { 8 };
            let mut amortized = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                let mut acc = dlsa_select(&s, &members, &pos, cache.rows() as i32, &p)
                    .unwrap()
                    .lengths;
                for _ in 1..batch {
                    let out = dlsa_select(&s, &members, &pos, cache.rows() as i32, &p).unwrap();
                    acc = mlxcel_core::add(&acc, &out.lengths);
                }
                mlxcel_core::eval(&acc);
                amortized.push(start.elapsed() / batch);
            }
            // One eval round trip of a trivial op: the sync floor inside `kernel`
            let tiny = mlxcel_core::from_slice_i32(&[1], &[1]);
            let mut floor = Vec::new();
            for _ in 0..reps {
                let start = Instant::now();
                let x = mlxcel_core::add(&tiny, &tiny);
                mlxcel_core::eval(&x);
                floor.push(start.elapsed());
            }
            let mut readback = Vec::new();
            let mut pick = Vec::new();
            for _ in 0..reps {
                // A fresh lazy copy so the readback really moves the [L, V] scores
                let fresh = mlxcel_core::multiply_scalar(&s, 1.0);
                mlxcel_core::eval(&fresh);
                let start = Instant::now();
                let host = to_host_f32(&fresh);
                readback.push(start.elapsed());
                let start = Instant::now();
                let sel = cache.select_from_scores(&host, stride, &positions).unwrap();
                pick.push(start.elapsed());
                std::hint::black_box(sel);
            }
            let (k, r, h) = (median(kernel), median(readback), median(pick));
            let (a, f) = (median(amortized), median(floor));
            println!(
                "dlsa_select bench: T {t} L {l} V {stride}: kernel {:.3} ms (with its eval; {:.3} ms amortized over {batch} launches; eval floor {:.3} ms) | host readback {:.3} ms + pick {:.3} ms = {:.3} ms | speedup {:.1}x ({:.1}x amortized)",
                k.as_secs_f64() * 1e3,
                a.as_secs_f64() * 1e3,
                f.as_secs_f64() * 1e3,
                r.as_secs_f64() * 1e3,
                h.as_secs_f64() * 1e3,
                (r + h).as_secs_f64() * 1e3,
                (r + h).as_secs_f64() / k.as_secs_f64(),
                (r + h).as_secs_f64() / a.as_secs_f64()
            );
        }
    }
}
