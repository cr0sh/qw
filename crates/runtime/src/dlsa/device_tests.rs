// SPDX-License-Identifier: Apache-2.0
//
//! The device mirror and deferred index keys against the host tables and an eager cache
//! (`docs/dlsa.md` §14.2). These tests need Metal (`./gpu-lock -- cargo test ...`).

use mlxcel_core::dtype;

use crate::dlsa::test_model::{TEST_INDEX, noise};
use crate::dlsa::{DlsaBucketCache, DlsaIndexConfig, to_host_f32};

fn to_host_i32(x: &mlxcel_core::MlxArray) -> Vec<i32> {
    let x = mlxcel_core::astype(x, dtype::INT32);
    mlxcel_core::eval(&x);
    mlxcel_core::array_to_raw_bytes(&x)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_ne_bytes(*c))
        .collect()
}

/// `rows` unit index keys (`[rows, d]` f32), deterministic in `seed`.
fn unit_keys(rows: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut out: Vec<f32> = (0..(rows * d) as u64).map(|i| noise(i, seed)).collect();
    for row in out.chunks_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
    }
    out
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

/// The mirror equals the host tables row for row.
fn assert_mirror_matches_host(cache: &mut DlsaBucketCache, what: &str) {
    let (rows, d, cap) = (
        cache.rows(),
        cache.config().index_dim,
        cache.config().bucket_cap,
    );
    let host_c = cache.centroids().to_vec();
    let host_m = cache.members().to_vec();
    let mirror = cache.device_mirror();
    assert_eq!(mirror.rows(), rows, "{what}: mirror rows");
    assert!(mirror.capacity() >= rows, "{what}: capacity");
    if rows == 0 {
        return;
    }
    let c = to_host_f32(&mirror.centroids(rows).unwrap());
    assert_eq!(c, host_c, "{what}: mirror centroids");
    let m = mlxcel_core::slice(
        mirror.members().unwrap(),
        &[0, 0],
        &[rows as i32, cap as i32],
    );
    assert_eq!(to_host_i32(&m), host_m, "{what}: mirror members");
    assert_eq!(c.len(), rows * d);
}

#[test]
fn mirror_grows_trims_and_scores_like_a_fresh_upload() {
    let cfg = TEST_INDEX;
    let d = cfg.index_dim;
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    let keys = unit_keys(3000, d, 11);
    let mut at = 0;
    // 3000 keys = 187 blocks = 748 rows: grows past the 256-row minimum twice
    for (i, chunk) in [100usize, 700, 37, 1, 1, 400, 900, 861].iter().enumerate() {
        cache.append(&keys[at * d..(at + chunk) * d]).unwrap();
        at += chunk;
        assert_mirror_matches_host(&mut cache, &format!("append {i}"));
        if i == 3 {
            // Reopen two blocks (not committed since 800), then append other keys over them
            let back = at - 37;
            cache.trim(back).unwrap();
            assert_mirror_matches_host(&mut cache, "after trim");
            cache.append(&unit_keys(37, d, 99)).unwrap();
            assert_mirror_matches_host(&mut cache, "after reappend");
        }
        if i != 2 {
            cache.commit();
        }
    }
    assert!(cache.device_mirror().capacity() >= 748);

    // Scores through the mirror view are bit-identical to scores over a fresh contiguous copy
    let q = mlxcel_core::from_slice_f32(&unit_keys(5, d, 3), &[5, d as i32]);
    for stride in [1usize, 100, 513, cache.rows()] {
        let view = cache.device_centroids(stride).unwrap();
        let fresh = mlxcel_core::from_slice_f32(
            &cache.centroids()[..stride * d],
            &[stride as i32, d as i32],
        );
        let a = to_host_f32(&mlxcel_core::matmul(&q, &mlxcel_core::transpose(&view)));
        let b = to_host_f32(&mlxcel_core::matmul(&q, &mlxcel_core::transpose(&fresh)));
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "stride {stride}: mirror scores differ"
        );
    }

    cache.reset();
    assert_eq!(cache.memory().device_centroids, 0);
    cache.append(&keys[..64 * d]).unwrap();
    assert_mirror_matches_host(&mut cache, "after reset");
}

/// Everything host-visible of two caches is equal, bit for bit (without flushing).
fn assert_same_state(eager: &DlsaBucketCache, deferred: &DlsaBucketCache, what: &str) {
    assert_eq!(eager.len(), deferred.len(), "{what}: len");
    assert_eq!(eager.committed(), deferred.committed(), "{what}: committed");
    assert_eq!(
        eager.tail_start(),
        deferred.tail_start(),
        "{what}: tail_start"
    );
    assert_eq!(eager.rows(), deferred.rows(), "{what}: rows");
    assert_eq!(eager.centroids(), deferred.centroids(), "{what}: centroids");
    assert_eq!(eager.members(), deferred.members(), "{what}: members");
    assert_eq!(eager.sizes(), deferred.sizes(), "{what}: sizes");
    assert_eq!(eager.snapshot(), deferred.snapshot(), "{what}: snapshot");
}

fn check_schedule(cfg: DlsaIndexConfig, seed: u64, ops: usize) -> (usize, usize) {
    let d = cfg.index_dim;
    let mut rng = Rng(seed);
    let mut eager = DlsaBucketCache::new(cfg).unwrap();
    let mut deferred = DlsaBucketCache::new(cfg).unwrap();
    let (mut max_deferred, mut selects) = (0, 0);
    for op in 0..ops {
        let what = format!("seed {seed} op {op}");
        match rng.below(20) {
            // A forward: decode-sized mostly, sometimes a chunk crossing 1-2 blocks
            0..=13 => {
                let rows = match rng.below(10) {
                    0..=5 => 1,
                    6..=8 => 2 + rng.below(7),
                    _ => 9 + rng.below(32),
                };
                let keys = unit_keys(rows, d, seed * 1000 + op as u64);
                eager.append(&keys).unwrap();
                deferred
                    .append_deferred(&mlxcel_core::from_slice_f32(
                        &keys,
                        &[rows as i32, d as i32],
                    ))
                    .unwrap();
                max_deferred = max_deferred.max(deferred.deferred_keys());
                // The forward's own selection, host vs device, and through the MLX path
                let positions: Vec<usize> = (eager.len() - rows..eager.len()).collect();
                let q = unit_keys(rows, d, seed * 7777 + op as u64);
                let q_mlx = mlxcel_core::from_slice_f32(&q, &[rows as i32, d as i32]);
                // The same MLX scores on both sides, so the lists must be equal exactly
                let want = eager.select(&q_mlx, &positions).unwrap();
                let device = deferred.select_device(&q_mlx, &positions).unwrap();
                assert_eq!(device.to_host().unwrap(), want, "{what}: device select");
                let mlx = deferred.select(&q_mlx, &positions).unwrap();
                assert_eq!(mlx, want, "{what}: mlx select");
                selects += 1;
            }
            14..=16 => {
                eager.commit();
                deferred.commit();
            }
            17 | 18 => {
                let lo = eager.tail_start();
                let n = lo + rng.below(eager.len() - lo + 1);
                eager.trim(n).unwrap();
                deferred.trim(n).unwrap();
                assert_eq!(deferred.deferred_keys(), 0, "{what}: trim flushes");
            }
            _ => {
                // Snapshot (deferred keys included), restore both, carry on
                let snap = deferred.snapshot();
                assert_eq!(snap, eager.snapshot(), "{what}: snapshot");
                eager = DlsaBucketCache::restore(cfg, &snap).unwrap();
                deferred = DlsaBucketCache::restore(cfg, &snap).unwrap();
            }
        }
        assert_same_state(&eager, &deferred, &what);
        if rng.below(5) == 0 {
            assert_eq!(eager.tail_keys(), deferred.tail_keys(), "{what}: tail keys");
            assert_eq!(deferred.deferred_keys(), 0, "{what}: tail_keys flushes");
        }
    }
    deferred.flush().unwrap();
    assert_eq!(
        eager.tail_keys(),
        deferred.tail_keys(),
        "seed {seed}: final tail keys"
    );
    assert_same_state(&eager, &deferred, &format!("seed {seed}: final"));
    (max_deferred, selects)
}

#[test]
fn deferred_keys_equal_eager_over_random_schedules() {
    let mut max_deferred = 0;
    for seed in 1..=24 {
        let (deferred, selects) = check_schedule(TEST_INDEX, seed, 60);
        assert!(selects > 0);
        max_deferred = max_deferred.max(deferred);
    }
    assert!(
        max_deferred >= 8,
        "keys were hardly ever deferred ({max_deferred})"
    );
    // A small top_p binds, so the pick itself matters
    let tight = DlsaIndexConfig {
        top_p: 3,
        top_k: 9,
        ..TEST_INDEX
    };
    for seed in 100..108 {
        check_schedule(tight, seed, 60);
    }
}

#[test]
fn deferred_keys_flush_on_block_close_and_check_finiteness() {
    let cfg = TEST_INDEX;
    let (w, d) = (cfg.window_size, cfg.index_dim);
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    let keys = unit_keys(w, d, 5);
    for p in 0..w - 1 {
        cache
            .append_deferred(&mlxcel_core::from_slice_f32(
                &keys[p * d..(p + 1) * d],
                &[1, d as i32],
            ))
            .unwrap();
    }
    assert_eq!((cache.deferred_keys(), cache.rows()), (w - 1, 0));
    // The key that closes the block reads everything back and clusters it
    cache
        .append_deferred(&mlxcel_core::from_slice_f32(
            &keys[(w - 1) * d..],
            &[1, d as i32],
        ))
        .unwrap();
    assert_eq!(cache.deferred_keys(), 0);
    assert_eq!(cache.rows(), cfg.buckets_per_block());
    let eager = DlsaBucketCache::build(cfg, &keys).unwrap();
    assert_eq!(cache.members(), eager.members());
    assert_eq!(cache.centroids(), eager.centroids());

    // A non-finite key is found at the flush
    let mut bad = vec![0.5f32; d];
    bad[3] = f32::NAN;
    cache
        .append_deferred(&mlxcel_core::from_slice_f32(&bad, &[1, d as i32]))
        .unwrap();
    assert_eq!(cache.deferred_keys(), 1);
    let err = cache.flush().expect_err("a NaN key").to_string();
    assert!(err.contains("non-finite"), "{err}");
    assert_eq!(cache.len(), w);
}

/// `memory()` and `nbytes()` count deferred keys (device f32) through every transition.
#[test]
fn memory_accounts_for_deferred_keys() {
    let cfg = TEST_INDEX;
    let (w, d) = (cfg.window_size, cfg.index_dim);
    let row = d * 4;
    let keys = unit_keys(2 * w, d, 21);
    let key = |p: usize| mlxcel_core::from_slice_f32(&keys[p * d..(p + 1) * d], &[1, d as i32]);
    #[track_caller]
    fn check(cache: &DlsaBucketCache, deferred: usize, host_keys: usize, what: &str) {
        let mem = cache.memory();
        let row = cache.config().index_dim * 4;
        assert_eq!(mem.deferred_keys, deferred * row, "{what}: deferred bytes");
        assert_eq!(mem.tail_keys, host_keys * row, "{what}: host key bytes");
        assert_eq!(cache.nbytes(), mem.allocated(), "{what}: nbytes");
        assert!(mem.allocated() >= mem.total(), "{what}: allocated < total");
    }
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    check(&cache, 0, 0, "empty");
    assert_eq!(cache.nbytes(), 0);

    cache.append_deferred(&key(0)).unwrap();
    check(&cache, 1, 0, "one deferred key");
    assert_eq!(cache.memory().total(), row);
    assert_eq!(
        cache.nbytes(),
        row,
        "a fresh cache holds only the deferred key"
    );
    for p in 1..4 {
        cache.append_deferred(&key(p)).unwrap();
    }
    check(&cache, 4, 0, "four deferred keys");
    cache.flush().unwrap();
    check(&cache, 0, 4, "after flush");

    for p in 4..w - 1 {
        cache.append_deferred(&key(p)).unwrap();
    }
    check(&cache, w - 5, 4, "before the block closes");
    cache.append_deferred(&key(w - 1)).unwrap();
    check(&cache, 0, w, "after the block closed");
    assert!(cache.memory().host_tables > 0);

    for p in w..w + 3 {
        cache.append_deferred(&key(p)).unwrap();
    }
    check(&cache, 3, w, "deferred in the open block");
    cache.trim(w + 1).unwrap();
    check(&cache, 0, w + 1, "after trim");

    for p in w + 1..w + 3 {
        cache.append_deferred(&key(p)).unwrap();
    }
    check(&cache, 2, w + 1, "deferred before snapshot");
    let restored = DlsaBucketCache::restore(cfg, &cache.snapshot()).unwrap();
    // A restore keeps only the open block's keys, on the host
    check(&restored, 0, 3, "restored");
    check(&cache, 2, w + 1, "snapshot keeps the keys deferred");

    cache.reset();
    check(&cache, 0, 0, "after reset");
    assert_eq!(cache.nbytes(), 0);
}
