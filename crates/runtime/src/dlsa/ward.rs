// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/core/ward.py (pool_keys, ward_rounds) with the tie policy of
// src/lsa/core/kernels/triton/ward.py:154-204, as tests/fixtures/dlsa/gen_fixtures.py
// (`ward_labels_kernel_policy`) states it.
//
//! Pooling and Ward clustering of one closed cluster block, on the host in f32 (`docs/dlsa.md` §4.2-4.3).

use anyhow::{Result, bail, ensure};

/// Rounds of mutual-cheapest merges at most (`ward.ward_rounds`, `max_rounds`).
pub const WARD_MAX_ROUNDS: usize = 256;

/// Sum each run of `pool_size` consecutive keys (`[n, dim]` row-major) into a pool key, not
/// normalized: `[n / pool_size, dim]`. Sums run in position order, in f32.
pub fn pool_keys(keys: &[f32], dim: usize, pool_size: usize) -> Vec<f32> {
    assert!(dim > 0 && pool_size > 0);
    assert_eq!(
        keys.len() % (dim * pool_size),
        0,
        "keys are not whole pools"
    );
    let pools = keys.len() / (dim * pool_size);
    let mut out = vec![0.0f32; pools * dim];
    for (p, pool) in out.chunks_exact_mut(dim).enumerate() {
        for key in keys[p * pool_size * dim..(p + 1) * pool_size * dim].chunks_exact(dim) {
            for (s, &v) in pool.iter_mut().zip(key) {
                *s += v;
            }
        }
    }
    out
}

/// The result of Ward over one block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WardOutcome {
    /// Bucket of each pool, `0..bucket_count`, numbered in the order of each bucket's first pool.
    pub labels: Vec<u32>,
    /// Rounds that merged at least one pair.
    pub rounds: usize,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// Merge `pool_keys` (`[P, dim]`, with `pool_sizes` keys each) down to exactly `bucket_count`
/// buckets of at most `bucket_cap` keys. Errors if the cap leaves a block short of its count
/// (impossible under `DlsaIndexConfig::validate`).
///
/// Every round (`ward.py:60-90`): Gram `G = S S^T` (f32, symmetric bitwise), merge cost
/// `sqrt(G_ii) + sqrt(G_jj) - sqrt(G_ii + G_jj + 2 G_ij)`; each live bucket takes its cheapest
/// partner under the cap, the **lowest index** on an exact tie; mutual pairs `(i, j)`, `i < j`,
/// merge `j` into `i`, and when more mutual pairs exist than merges are left, the pairs ranked
/// first by `(cost, partner index)` merge (the served kernel's policy, not the reference's
/// unstable argsort; `docs/dlsa.md` §4.3).
pub fn ward_labels(
    pool_keys: &[f32],
    pool_sizes: &[f32],
    dim: usize,
    bucket_count: usize,
    bucket_cap: usize,
) -> Result<WardOutcome> {
    ward_labels_with_rounds(
        pool_keys,
        pool_sizes,
        dim,
        bucket_count,
        bucket_cap,
        WARD_MAX_ROUNDS,
    )
}

/// [`ward_labels`] with an explicit round limit.
pub fn ward_labels_with_rounds(
    pool_keys: &[f32],
    pool_sizes: &[f32],
    dim: usize,
    bucket_count: usize,
    bucket_cap: usize,
    max_rounds: usize,
) -> Result<WardOutcome> {
    ensure!(dim > 0, "dim must be positive");
    ensure!(
        pool_keys.len().is_multiple_of(dim),
        "pool keys are not [pools, {dim}]"
    );
    let pools = pool_keys.len() / dim;
    ensure!(
        pool_sizes.len() == pools,
        "{} pool sizes for {pools} pools",
        pool_sizes.len()
    );
    ensure!(bucket_count > 0, "bucket_count must be positive");
    ensure!(
        pool_keys.iter().all(|v| v.is_finite()),
        "pool keys hold a non-finite value"
    );
    let cap = bucket_cap as f32;
    let mut sums = pool_keys.to_vec();
    let mut sizes = pool_sizes.to_vec();
    let mut alive = vec![true; pools];
    let mut parent: Vec<usize> = (0..pools).collect();
    let mut alive_count = pools;
    let mut rounds = 0;
    let mut gram = vec![0.0f32; pools * pools];
    let mut best: Vec<Option<(usize, f32)>> = vec![None; pools];
    let mut mutual: Vec<(f32, usize, usize)> = Vec::with_capacity(pools / 2);
    for _ in 0..max_rounds {
        if alive_count <= bucket_count {
            break;
        }
        let merges_left = alive_count - bucket_count;
        let live: Vec<usize> = (0..pools).filter(|&i| alive[i]).collect();
        // One dot per unordered pair, mirrored, so cost(i, j) == cost(j, i) bitwise
        for (a, &i) in live.iter().enumerate() {
            let si = &sums[i * dim..(i + 1) * dim];
            for &j in &live[a..] {
                let g = dot(si, &sums[j * dim..(j + 1) * dim]);
                gram[i * pools + j] = g;
                gram[j * pools + i] = g;
            }
        }
        for &i in &live {
            let sq_i = gram[i * pools + i];
            let align_i = sq_i.max(0.0).sqrt();
            let mut choice: Option<(usize, f32)> = None;
            for &j in &live {
                if j == i || sizes[i] + sizes[j] > cap {
                    continue;
                }
                let sq_j = gram[j * pools + j];
                let merged = (sq_i + sq_j + 2.0 * gram[i * pools + j]).max(0.0).sqrt();
                let cost = align_i + sq_j.max(0.0).sqrt() - merged;
                // Ascending j with a strict `<` keeps the lowest index on an exact tie
                if choice.is_none_or(|(_, c)| cost < c) {
                    choice = Some((j, cost));
                }
            }
            best[i] = choice;
        }
        mutual.clear();
        for &i in &live {
            if let Some((j, cost)) = best[i]
                && i < j
                && best[j].is_some_and(|(k, _)| k == i)
            {
                mutual.push((cost, j, i));
            }
        }
        if mutual.is_empty() {
            break;
        }
        // Rank by (cost, partner index); partners are distinct, so the order is total
        // (`partial_cmp`, as torch compares: -0.0 == 0.0; costs of finite keys are never NaN)
        mutual.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("a Ward merge cost is NaN")
                .then(a.1.cmp(&b.1))
        });
        mutual.truncate(merges_left);
        for &(_, j, i) in &mutual {
            for t in 0..dim {
                let v = sums[j * dim + t];
                sums[i * dim + t] += v;
            }
            sizes[i] += sizes[j];
            alive[j] = false;
            parent[j] = i;
        }
        alive_count -= mutual.len();
        rounds += 1;
    }
    if alive_count > bucket_count {
        bail!(
            "Ward stopped at {alive_count} buckets, short of {bucket_count}, under the cap {bucket_cap}"
        );
    }
    let mut number = vec![0u32; pools];
    let mut next = 0u32;
    for i in 0..pools {
        if alive[i] {
            number[i] = next;
            next += 1;
        }
    }
    let labels = (0..pools)
        .map(|mut i| {
            while parent[i] != i {
                i = parent[i];
            }
            number[i]
        })
        .collect();
    Ok(WardOutcome { labels, rounds })
}
