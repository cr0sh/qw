// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! K1 `dlsa_select` against a host oracle written here (a comparator sort, independent of
//! the kernel's packed keys). Metal-only: every test returns early without Metal.
//!
//! Run: `cargo test -p mlxcel-core --lib dlsa_kernels`

use std::cmp::Ordering;

use super::*;

fn gpu() -> bool {
    crate::metal_is_available()
}

/// splitmix64, so the tests need no RNG crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug)]
enum ScoreKind {
    /// Uniform floats in [-1, 1).
    Float,
    /// {-1, 0, 1}: ties everywhere.
    Ternary,
    /// {-0.0, +0.0, 0.5}: ±0 must tie.
    SignedZeros,
    /// A handful of values: long runs of equal scores.
    FewValues,
}

fn make_scores(rng: &mut Rng, kind: ScoreKind, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| match kind {
            ScoreKind::Float => rng.unit(),
            ScoreKind::Ternary => [-1.0, 0.0, 1.0][rng.below(3)],
            ScoreKind::SignedZeros => [-0.0, 0.0, 0.5][rng.below(3)],
            ScoreKind::FewValues => [0.25, -0.125, 0.75, 0.75, 1e-30][rng.below(5)],
        })
        .collect()
}

/// `[rows, cap]`: bucket b of block b / m holds an ascending random subset of that block's
/// positions, `-1` padded.
fn make_members(rng: &mut Rng, p: &DlsaSelectParams, rows: usize) -> Vec<i32> {
    let (w, m, cap) = (
        p.window as usize,
        p.buckets_per_block as usize,
        p.bucket_cap as usize,
    );
    let mut out = vec![-1i32; rows * cap];
    for b in 0..rows {
        let base = (b / m) * w;
        let n = rng.below(cap.min(w) + 1);
        let mut picks: Vec<usize> = Vec::with_capacity(n);
        while picks.len() < n {
            let x = base + rng.below(w);
            if !picks.contains(&x) {
                picks.push(x);
            }
        }
        picks.sort_unstable();
        for (j, &x) in picks.iter().enumerate() {
            out[b * cap + j] = x as i32;
        }
    }
    out
}

/// Best first: score descending (`-0.0 == 0.0`, a NaN below every number), then bucket
/// ascending.
fn oracle_rank(s: &[f32], a: usize, b: usize) -> Ordering {
    let (x, y) = (s[a], s[b]);
    let by_score = match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => y.partial_cmp(&x).unwrap(),
    };
    by_score.then(a.cmp(&b))
}

struct Expected {
    lists: Vec<i32>,
    lengths: Vec<i32>,
    picked: Vec<i32>,
    status: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
fn oracle(
    scores: &[f32],
    stride: usize,
    members: &[i32],
    positions: &[i32],
    total_rows: usize,
    p: &DlsaSelectParams,
) -> Expected {
    let (w, m, cap) = (
        p.window as usize,
        p.buckets_per_block as usize,
        p.bucket_cap as usize,
    );
    let (top_k, top_p, sink) = (p.top_k as usize, p.top_p as usize, p.sink as usize);
    let width = p.list_width() as usize;
    let rows = positions.len();
    let mut e = Expected {
        lists: vec![-1; rows * width],
        lengths: vec![0; rows],
        picked: vec![-1; rows * top_p],
        status: vec![0; rows],
    };
    for r in 0..rows {
        let pos = positions[r] as usize;
        let mut v = (pos / w * m).min(total_rows);
        if v > stride {
            v = stride;
            e.status[r] |= STATUS_SHORT_SCORES;
        }
        let row = &scores[r * stride..r * stride + v];
        if row.iter().any(|s| s.is_nan()) {
            e.status[r] |= STATUS_NAN;
        }
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by(|&a, &b| oracle_rank(row, a, b));
        order.truncate(top_p);
        for (i, &b) in order.iter().enumerate() {
            e.picked[r * top_p + i] = b as i32;
        }
        let ws = (pos + 1).saturating_sub(w);
        let mut list = Vec::with_capacity(width);
        'walk: for &b in &order {
            for &x in &members[b * cap..(b + 1) * cap] {
                if x >= 0 && (x as usize) >= sink && (x as usize) < ws {
                    if list.len() == top_k {
                        break 'walk;
                    }
                    list.push(x);
                }
            }
        }
        list.extend((ws..=pos).map(|x| x as i32));
        list.extend((0..sink.min(ws).min(pos + 1)).map(|x| x as i32));
        e.lists[r * width..r * width + list.len()].copy_from_slice(&list);
        e.lengths[r] = list.len() as i32;
    }
    e
}

fn host_i32(a: &MlxArray) -> Vec<i32> {
    crate::eval(a);
    crate::array_to_raw_bytes(a)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect()
}

/// Run the kernel and compare every output with the oracle.
#[allow(clippy::too_many_arguments)]
fn check(
    label: &str,
    scores: &[f32],
    stride: usize,
    members: &[i32],
    positions: &[i32],
    total_rows: usize,
    p: &DlsaSelectParams,
    threads: i32,
) {
    let rows = positions.len();
    let exp = oracle(scores, stride, members, positions, total_rows, p);
    let sc = crate::from_slice_f32(scores, &[rows as i32, stride as i32]);
    let mrows = members.len() / p.bucket_cap as usize;
    let mem = crate::from_slice_i32(members, &[mrows as i32, p.bucket_cap]);
    let pos = crate::from_slice_i32(positions, &[rows as i32]);
    let out = dlsa_select_with(&sc, &mem, &pos, total_rows as i32, p, threads)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let (lists, lengths) = (host_i32(&out.lists), host_i32(&out.lengths));
    let (picked, status) = (host_i32(&out.picked), host_i32(&out.status));
    let width = p.list_width() as usize;
    for r in 0..rows {
        assert_eq!(
            picked[r * p.top_p as usize..(r + 1) * p.top_p as usize],
            exp.picked[r * p.top_p as usize..(r + 1) * p.top_p as usize],
            "{label}: picked of row {r} (position {})",
            positions[r]
        );
        assert_eq!(lengths[r], exp.lengths[r], "{label}: length of row {r}");
        assert_eq!(
            lists[r * width..(r + 1) * width],
            exp.lists[r * width..(r + 1) * width],
            "{label}: list of row {r} (position {})",
            positions[r]
        );
        assert_eq!(status[r], exp.status[r], "{label}: status of row {r}");
    }
}

const QWEN: DlsaSelectParams = DlsaSelectParams {
    window: 512,
    sink: 4,
    top_k: 4096,
    top_p: 1024,
    bucket_cap: 16,
    buckets_per_block: 64,
};

const SMALL: DlsaSelectParams = DlsaSelectParams {
    window: 16,
    sink: 2,
    top_k: 20,
    top_p: 6,
    bucket_cap: 8,
    buckets_per_block: 4,
};

/// One random case: `rows` positions below `t` (plus the block edges and `t - 1`).
fn random_case(seed: u64, p: &DlsaSelectParams, t: usize, rows: usize, kind: ScoreKind) {
    let mut rng = Rng(seed);
    let w = p.window as usize;
    let total_rows = t / w * p.buckets_per_block as usize;
    let members = make_members(&mut rng, p, total_rows.max(1));
    let mut positions: Vec<i32> = (0..rows).map(|_| rng.below(t) as i32).collect();
    for edge in [0, w - 1, w, 2 * w - 1, 2 * w, t - 1] {
        if edge < t {
            positions.push(edge as i32);
        }
    }
    let stride = positions
        .iter()
        .map(|&x| (x as usize / w * p.buckets_per_block as usize).min(total_rows))
        .max()
        .unwrap_or(0);
    let scores = make_scores(&mut rng, kind, positions.len() * stride);
    for threads in [0, 256] {
        check(
            &format!("seed {seed} {kind:?} t {t} threads {threads} {p:?}"),
            &scores,
            stride,
            &members,
            &positions,
            total_rows,
            p,
            threads,
        );
    }
}

#[test]
fn small_configs_match_oracle() {
    if !gpu() {
        return;
    }
    let kinds = [
        ScoreKind::Float,
        ScoreKind::Ternary,
        ScoreKind::SignedZeros,
        ScoreKind::FewValues,
    ];
    let mut seed = 1;
    for p in [
        SMALL,
        // top_k binding long before top_p
        DlsaSelectParams { top_k: 3, ..SMALL },
        // top_p = 1, no sinks
        DlsaSelectParams {
            top_p: 1,
            sink: 0,
            ..SMALL
        },
        // more sinks than the window
        DlsaSelectParams { sink: 40, ..SMALL },
        // top_p binding (top_p * cap far above top_k is fine, below is the binding case)
        DlsaSelectParams {
            top_p: 2,
            top_k: 64,
            ..SMALL
        },
    ] {
        for t in [1, 15, 16, 17, 33, 100, 400] {
            for kind in kinds {
                random_case(seed, &p, t, 24, kind);
                seed += 1;
            }
        }
    }
}

#[test]
fn qwen_defaults_match_oracle_up_to_40k_buckets() {
    if !gpu() {
        return;
    }
    let mut seed = 1000;
    // V = 64 * floor(T / 512): 0, 64, 1024 (= top_p), 1088, 11 712, 40 000
    for t in [511, 1024, 8704, 9216, 93_696, 320_000] {
        for kind in [ScoreKind::Float, ScoreKind::Ternary, ScoreKind::SignedZeros] {
            random_case(seed, &QWEN, t, 12, kind);
            seed += 1;
        }
    }
}

#[test]
fn top_p_limit_and_wide_lists() {
    if !gpu() {
        return;
    }
    let p = DlsaSelectParams {
        top_p: DLSA_TOP_P_MAX,
        top_k: 8192,
        ..QWEN
    };
    for (seed, kind) in [(7, ScoreKind::Float), (8, ScoreKind::Ternary)] {
        random_case(seed, &p, 70_000, 6, kind);
    }
}

#[test]
fn threadgroup_size_does_not_change_outputs() {
    if !gpu() {
        return;
    }
    let mut rng = Rng(42);
    let t = 40_000;
    let total_rows = t / 512 * 64;
    let members = make_members(&mut rng, &QWEN, total_rows);
    let positions: Vec<i32> = (0..8).map(|_| rng.below(t) as i32).collect();
    let scores = make_scores(&mut rng, ScoreKind::Ternary, positions.len() * total_rows);
    for threads in [64, 128, 512, 1024] {
        check(
            &format!("threads {threads}"),
            &scores,
            total_rows,
            &members,
            &positions,
            total_rows,
            &QWEN,
            threads,
        );
    }
}

#[test]
fn nan_ranks_last_and_sets_status() {
    if !gpu() {
        return;
    }
    let mut rng = Rng(5);
    let t = 4096;
    let total_rows = t / 16 * 4;
    let members = make_members(&mut rng, &SMALL, total_rows);
    let positions = vec![4095, 200, 15];
    let mut scores = make_scores(&mut rng, ScoreKind::Float, positions.len() * total_rows);
    scores[3] = f32::NAN;
    scores[total_rows + 1] = -f32::NAN;
    scores[total_rows + 2] = f32::NEG_INFINITY;
    // Row 2 sees no buckets: its NaN is never read
    scores[2 * total_rows] = f32::NAN;
    check(
        "nan", &scores, total_rows, &members, &positions, total_rows, &SMALL, 0,
    );
}

#[test]
fn short_scores_and_mirror_capacity() {
    if !gpu() {
        return;
    }
    let mut rng = Rng(6);
    let total_rows = 64;
    // The member table has spare rows past total_rows (a preallocated mirror)
    let members = make_members(&mut rng, &SMALL, total_rows + 40);
    let positions = vec![1000, 100];
    // stride 20 < visible(1000) = 64: status bit 1 on row 0
    let scores = make_scores(&mut rng, ScoreKind::Float, positions.len() * 20);
    check(
        "short", &scores, 20, &members, &positions, total_rows, &SMALL, 0,
    );
}

#[test]
fn empty_cache_and_no_rows() {
    if !gpu() {
        return;
    }
    // No bucket yet: S = 0 and N = 0
    let positions = vec![0, 5, 300];
    check("empty", &[], 0, &[], &positions, 0, &SMALL, 0);
    // R = 0
    let sc = crate::from_slice_f32(&[], &[0, 4]);
    let mem = crate::from_slice_i32(&[-1; 8], &[1, 8]);
    let pos = crate::from_slice_i32(&[], &[0]);
    let out = dlsa_select(&sc, &mem, &pos, 1, &SMALL).unwrap();
    assert_eq!(crate::array_shape(&out.lists), vec![0, SMALL.list_width()]);
    assert_eq!(crate::array_shape(&out.picked), vec![0, SMALL.top_p]);
}

#[test]
fn bad_arguments_are_errors() {
    let sc = crate::from_slice_f32(&[0.0; 4], &[1, 4]);
    let mem = crate::from_slice_i32(&[-1; 32], &[4, 8]);
    let pos = crate::from_slice_i32(&[100], &[1]);
    let too_many = DlsaSelectParams {
        top_p: DLSA_TOP_P_MAX + 1,
        ..SMALL
    };
    assert!(dlsa_select(&sc, &mem, &pos, 4, &too_many).is_err());
    let wrong_cap = DlsaSelectParams {
        bucket_cap: 4,
        ..SMALL
    };
    assert!(dlsa_select(&sc, &mem, &pos, 4, &wrong_cap).is_err());
    assert!(dlsa_select(&sc, &mem, &pos, 5, &SMALL).is_err());
    assert!(dlsa_select_with(&sc, &mem, &pos, 4, &SMALL, 100).is_err());
}
