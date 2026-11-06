// SPDX-License-Identifier: Apache-2.0
//
//! D-LSA serving integration on the tiny synthetic Qwen3.5 hybrid of `dlsa_tests` (`docs/dlsa.md`
//! §13): index state in snapshots (T1-T5), multi-row verification and rollback (S1-S4), and
//! speculative and constrained generation through a synthetic MTP head (S2, S5).
//!
//! These tests need Metal: run them under `./gpu-lock -- cargo test -p qw-runtime --features dlsa`.

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::{
    ConstraintCommit, ConstraintMask, LanguageModel, SamplingConfig, TokenConstraint,
};
use mlxcel_core::weights::WeightMap;

use super::dlsa_tests::{
    FULL_LAYERS, HIDDEN, VOCAB, assert_allclose, assert_bitwise, dlsa_lens, dlsa_model,
    dlsa_test_model, run, schedule, test_config, tokens,
};
use super::*;
use crate::dlsa::test_model::{
    TEST_HEAD_DIM, TEST_HEADS, TEST_INDEX, TEST_KV_HEADS, max_abs_diff, noise, noise_array,
};
use crate::dlsa::{DlsaConfig, DlsaIndexConfig, DlsaShortContext, to_host_f32};
use crate::portable_snapshot::{model_from_portable, portable_model_state};
use crate::qwen3_5_mtp::{Qwen35MtpDraftModel, Qwen35MtpGenerator};

fn seq() -> SequenceId {
    SequenceId::from_raw(1)
}

/// The index tables of one layer: `(len, centroids, members, sizes, rows)`.
type Tables = (usize, Vec<f32>, Vec<i32>, Vec<i32>, usize);

fn index_tables(model: &Qwen35Model) -> Vec<Tables> {
    let state = model.dlsa.as_ref().expect("D-LSA on").state.borrow();
    state
        .0
        .iter()
        .map(|c| {
            (
                c.len(),
                c.centroids().to_vec(),
                c.members().to_vec(),
                c.sizes().to_vec(),
                c.rows(),
            )
        })
        .collect()
}

/// Members and sizes exact, lengths equal, centroids within `tol` (`0.0`: bitwise).
#[track_caller]
fn assert_tables(actual: &[Tables], expected: &[Tables], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: layers");
    for (layer, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.0, b.0, "{what}: layer {layer} len");
        assert_eq!(a.2, b.2, "{what}: layer {layer} members");
        assert_eq!(a.3, b.3, "{what}: layer {layer} sizes");
        if tol == 0.0 {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.1), bits(&b.1), "{what}: layer {layer} centroids");
        } else {
            let diff = max_abs_diff(&a.1, &b.1);
            assert!(
                diff <= tol,
                "{what}: layer {layer} centroids differ by {diff:e}"
            );
        }
    }
}

/// KV offsets of the full-attention layers.
fn kv_offsets(model: &Qwen35Model) -> Vec<usize> {
    model.sequence_state.with_internal(|caches| {
        FULL_LAYERS
            .iter()
            .map(|&l| caches[l].offset() as usize)
            .collect()
    })
}

#[track_caller]
fn assert_lockstep(model: &Qwen35Model, what: &str) {
    if model.dlsa_enabled() {
        assert_eq!(dlsa_lens(model), kv_offsets(model), "{what}: index vs KV");
    }
}

fn sparse_config() -> DlsaConfig {
    test_config(TEST_INDEX, DlsaShortContext::Sparse)
}

fn covering_config() -> DlsaConfig {
    test_config(
        DlsaIndexConfig {
            top_k: 512,
            top_p: 128,
            ..TEST_INDEX
        },
        DlsaShortContext::Sparse,
    )
}

/// Chunks of at most 37 rows covering `n` tokens.
fn chunks(n: usize) -> Vec<usize> {
    let mut out = vec![37; n / 37];
    if !n.is_multiple_of(37) {
        out.push(n % 37);
    }
    out
}

fn argmax(row: &[f32]) -> i32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as i32
}

/// Top-1 minus top-2 logit of a row.
fn margin(row: &[f32]) -> f32 {
    let mut sorted = row.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[0] - sorted[1]
}

fn rows(logits: &[f32]) -> impl Iterator<Item = &[f32]> {
    logits.chunks(VOCAB as usize)
}

// --- 1a: snapshots -------------------------------------------------------------------------

/// T1: a snapshot at 70 (open block partly filled), 64 (open block empty) and 10 tokens (no
/// rows) restores into a fresh model and into the same model after a reset, and 20 decodes
/// then give bitwise the logits and tables of the uninterrupted run.
#[test]
fn snapshot_restore_continues_bitwise() {
    let toks = tokens(90, 11);
    for at in [70usize, 64, 10] {
        let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
        run(&model, &toks[..at], &chunks(at));
        let snapshot = model
            .capture_sequence_state(at, None, true)
            .expect("D-LSA snapshot");
        let closed = at / TEST_INDEX.window_size;
        assert_eq!(
            snapshot.tensor("layer.1.dlsa.centroids").is_some(),
            closed > 0,
            "{at}: centroids present iff rows"
        );
        assert_eq!(
            snapshot.tensor("layer.1.dlsa.open_keys").is_some(),
            !at.is_multiple_of(TEST_INDEX.window_size),
            "{at}: open keys present iff the open block holds keys"
        );
        let tail = &toks[at..at + 20];
        let want = run(&model, tail, &[1; 20]);
        let want_tables = index_tables(&model);

        let fresh = dlsa_model(KVCacheMode::Fp16, sparse_config());
        fresh
            .restore_sequence_state(seq(), &snapshot)
            .expect("restore into a fresh model");
        assert_eq!(dlsa_lens(&fresh), vec![at, at]);
        assert_bitwise(&run(&fresh, tail, &[1; 20]), &want, &format!("{at}: fresh"));
        assert_tables(&index_tables(&fresh), &want_tables, 0.0, "fresh");

        model.reset_runtime_state();
        model
            .restore_sequence_state(seq(), &snapshot)
            .expect("restore after reset");
        assert_bitwise(&run(&model, tail, &[1; 20]), &want, &format!("{at}: reset"));
        assert_tables(&index_tables(&model), &want_tables, 0.0, "reset");
    }
}

/// T2: the snapshot survives the portable (persisted) form and restores identically.
#[test]
fn snapshot_survives_portable_round_trip() {
    let toks = tokens(90, 12);
    for at in [70usize, 64, 10] {
        let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
        run(&model, &toks[..at], &chunks(at));
        let snapshot = model.capture_sequence_state(at, None, true).unwrap();
        let tail = &toks[at..at + 20];
        let want = run(&model, tail, &[1; 20]);
        let want_tables = index_tables(&model);
        let restored = model_from_portable(portable_model_state(&snapshot), false, false)
            .expect("portable D-LSA snapshot");
        let fresh = dlsa_model(KVCacheMode::Fp16, sparse_config());
        fresh
            .restore_sequence_state(seq(), &restored)
            .expect("restore the portable snapshot");
        assert_bitwise(
            &run(&fresh, tail, &[1; 20]),
            &want,
            &format!("{at}: portable"),
        );
        assert_tables(&index_tables(&fresh), &want_tables, 0.0, "portable");
    }
}

/// T3: restore at 70, then 30 rows + 10 decodes, equals a fresh prefill of the same tokens in
/// another chunk schedule: tables exact (centroids within projection rounding), logits close.
#[test]
fn snapshot_restore_equals_rebuild() {
    let toks = tokens(110, 13);
    let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
    run(&model, &toks[..70], &[37, 33]);
    let snapshot = model.capture_sequence_state(70, None, true).unwrap();
    let restored = dlsa_model(KVCacheMode::Fp16, sparse_config());
    restored.restore_sequence_state(seq(), &snapshot).unwrap();
    let got = run(&restored, &toks[70..], &schedule(&[30], 10));

    let rebuilt = dlsa_model(KVCacheMode::Fp16, sparse_config());
    let all = run(&rebuilt, &toks, &schedule(&[50, 20, 30], 10));
    let want = &all[70 * VOCAB as usize..];
    assert_allclose(&got, want, 1e-4, 1e-4, "restore vs rebuild");
    assert_tables(
        &index_tables(&restored),
        &index_tables(&rebuilt),
        1e-6,
        "restore vs rebuild",
    );
}

/// The tensor named `name` of a portable snapshot, as mutable bytes.
fn portable_tensor<'a>(
    portable: &'a mut crate::portable_snapshot::PortableModelState,
    name: &str,
) -> &'a mut Vec<u8> {
    &mut portable
        .tensors
        .iter_mut()
        .find(|t| t.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no tensor {name}"))
        .bytes
}

/// T4: dense <-> D-LSA snapshots, another index config, an out-of-range member and a
/// mismatching length all err without touching the live state; capture gives `None` while an
/// index is out of step with the KV cache.
#[test]
fn snapshot_invalidation() {
    let toks = tokens(70, 14);
    let dense = dlsa_test_model(KVCacheMode::Fp16);
    run(&dense, &toks, &chunks(70));
    let dense_snapshot = dense.capture_sequence_state(70, None, true).unwrap();
    let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
    run(&model, &toks, &chunks(70));
    let snapshot = model.capture_sequence_state(70, None, true).unwrap();

    let err = model
        .restore_sequence_state(seq(), &dense_snapshot)
        .expect_err("dense snapshot into a D-LSA model");
    assert!(err.contains("no D-LSA index state"), "{err}");
    let err = dense
        .restore_sequence_state(seq(), &snapshot)
        .expect_err("D-LSA snapshot into a dense model");
    assert!(err.contains("D-LSA is off"), "{err}");

    let other = dlsa_model(
        KVCacheMode::Fp16,
        test_config(
            DlsaIndexConfig {
                top_k: 40,
                ..TEST_INDEX
            },
            DlsaShortContext::Sparse,
        ),
    );
    let err = other
        .restore_sequence_state(seq(), &snapshot)
        .expect_err("another index config");
    assert!(err.contains("another index config"), "{err}");

    // A live model the failed restores below must leave untouched
    let live = dlsa_model(KVCacheMode::Fp16, sparse_config());
    run(&live, &toks[..33], &[33]);
    let before = index_tables(&live);
    let tamper = |name: &str, value: i32| {
        let mut portable = portable_model_state(&snapshot);
        portable_tensor(&mut portable, name)[..4].copy_from_slice(&value.to_ne_bytes());
        model_from_portable(portable, false, false).unwrap()
    };
    for (name, value, needle) in [
        ("layer.1.dlsa.members", 64, "out of range"),
        ("layer.3.dlsa.members", -2, "out of range"),
        ("layer.3.dlsa.sizes", 9, "out of range"),
        ("layer.1.dlsa.len", 69, "holds 69 keys"),
        ("meta.dlsa.index", 8, "another index config"),
    ] {
        let err = live
            .restore_sequence_state(seq(), &tamper(name, value))
            .expect_err(name);
        assert!(err.contains(needle), "{name}: {err}");
        assert_tables(&index_tables(&live), &before, 0.0, name);
        assert_eq!(kv_offsets(&live), vec![33, 33], "{name}: KV untouched");
    }

    // An index out of step with the KV cache cannot be captured
    model.dlsa.as_ref().unwrap().state.borrow_mut().0[1]
        .trim(69)
        .unwrap();
    assert!(model.capture_sequence_state(70, None, true).is_none());
}

/// T5: the D-LSA tensors cost exactly their table bytes on top of the dense snapshot.
#[test]
fn snapshot_dlsa_bytes() {
    let toks = tokens(70, 15);
    let dense = dlsa_test_model(KVCacheMode::Fp16);
    run(&dense, &toks, &chunks(70));
    let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
    run(&model, &toks, &chunks(70));
    let with = model.capture_sequence_state(70, None, true).unwrap();
    let without = dense.capture_sequence_state(70, None, true).unwrap();
    let cfg = TEST_INDEX;
    let rows = 70 / cfg.window_size * cfg.buckets_per_block();
    let open = 70 % cfg.window_size;
    let per_layer = 4 + 4 * rows * (cfg.index_dim + cfg.bucket_cap + 1) + 4 * open * cfg.index_dim;
    let expected = FULL_LAYERS.len() * per_layer + 4 * 8;
    assert_eq!(with.nbytes() - without.nbytes(), expected);
}

// --- 1b: verify and rollback ---------------------------------------------------------------

/// The logits rows `0..=accepted` of a verify (`[rows, V]` flattened).
fn verify_rows(logits: &MlxArray, accepted: usize) -> Vec<f32> {
    to_host_f32(logits)[..(accepted + 1) * VOCAB as usize].to_vec()
}

/// After `prefix`, verify each block and keep its first `accepted + 1` rows. Returns the kept
/// tokens and the logits of their rows. Checks lockstep after every round.
fn speculate(
    model: &Qwen35Model,
    prefix: &[i32],
    rounds: &[(Vec<i32>, usize)],
    dflash: bool,
) -> (Vec<i32>, Vec<f32>) {
    run(model, prefix, &chunks(prefix.len()));
    let (mut kept, mut logits) = (Vec::new(), Vec::new());
    for (round, (block, accepted)) in rounds.iter().enumerate() {
        let input = mlxcel_core::from_slice_i32(block, &[1, block.len() as i32]);
        let (out, gdn) = if dflash {
            let v = model.forward_dflash_verify(&input, &[1, 3]);
            (v.logits, v.gdn_states)
        } else {
            let v = model.forward_mtp_verify(&input);
            (v.logits, v.gdn_states)
        };
        logits.extend(verify_rows(&out, *accepted));
        model.rollback_mtp_verify(&gdn, *accepted, block.len(), round % 2 == 0);
        kept.extend_from_slice(&block[..=*accepted]);
        assert_lockstep(model, &format!("round {round}"));
        assert_eq!(kv_offsets(model)[0], prefix.len() + kept.len());
    }
    (kept, logits)
}

/// The non-speculative reference of [`speculate`]: the prefix, then the kept tokens one by one.
fn sequential(model: &Qwen35Model, prefix: &[i32], kept: &[i32]) -> Vec<f32> {
    run(model, prefix, &chunks(prefix.len()));
    run(model, kept, &vec![1; kept.len()])
}

/// Run `rounds` speculatively and sequentially, continue both with 20 decodes, and compare.
/// Members and sizes must agree exactly. Fp16: centroids within projection rounding (1e-6),
/// logits within 1e-4. Int8 KV: a multi-row read already differs from the sequential one
/// without D-LSA (measured max logit diffs at width 3: 6.8e-4 dense, 2.1e-3 sparse), and that
/// noise reaches the index keys of the second full-attention layer (centroids 2.2e-5), so
/// int8 gets 5e-3 on logits and 1e-4 on centroids.
/// The dense model's own speculative-vs-sequential difference is printed for comparison.
fn check_rollback(
    mode: KVCacheMode,
    config: DlsaConfig,
    prefix: &[i32],
    rounds: &[(Vec<i32>, usize)],
    dflash: bool,
    what: &str,
) -> Vec<Tables> {
    let spec = dlsa_model(mode, config.clone());
    let (kept, spec_logits) = speculate(&spec, prefix, rounds, dflash);
    let tables_after = index_tables(&spec);
    let reference = dlsa_model(mode, config);
    let ref_logits = sequential(&reference, prefix, &kept);
    let (rtol, atol, centroid_tol) = match mode {
        KVCacheMode::Int8 => (1e-3, 5e-3, 1e-4),
        _ => (1e-4, 1e-4, 1e-6),
    };
    let dense = dlsa_test_model(mode);
    let (_, d_spec) = speculate(&dense, prefix, rounds, dflash);
    let d_ref = sequential(&dlsa_test_model(mode), prefix, &kept);
    eprintln!(
        "{what}: speculative vs sequential max diff: D-LSA {:e}, dense {:e}",
        max_abs_diff(&spec_logits, &ref_logits),
        max_abs_diff(&d_spec, &d_ref),
    );
    assert_tables(&tables_after, &index_tables(&reference), centroid_tol, what);
    assert_allclose(&spec_logits, &ref_logits, rtol, atol, what);
    let tail = tokens(20, 99);
    let a = run(&spec, &tail, &[1; 20]);
    let b = run(&reference, &tail, &[1; 20]);
    let what = format!("{what}: continuation");
    assert_tables(
        &index_tables(&spec),
        &index_tables(&reference),
        centroid_tol,
        &what,
    );
    assert_allclose(&a, &b, rtol, atol, &what);
    tables_after
}

/// S1: a verify that closes the block at position 15 rolled back to 0 or 1 accepted rows
/// (reopening it), to partial counts, or fully accepted, equals the sequential run fed the
/// accepted tokens, for MTP-style 4-row and DFlash-style 8-row blocks.
#[test]
fn rollback_through_block_boundary() {
    let prefix = tokens(13, 21);
    let m = TEST_INDEX.buckets_per_block();
    for (width, dflash) in [(4usize, false), (8, true)] {
        let block = tokens(width, 22 + width as u64);
        for accepted in 0..width {
            let what = format!("width {width}, accepted {accepted}");
            let tables = check_rollback(
                KVCacheMode::Fp16,
                sparse_config(),
                &prefix,
                &[(block.clone(), accepted)],
                dflash,
                &what,
            );
            // Rows 13.. are kept: the block of 0..15 is closed only if position 15 is
            let closed = 13 + accepted + 1 >= 16;
            for t in &tables {
                assert_eq!(t.4, if closed { m } else { 0 }, "{what}: bucket rows");
            }
        }
    }
}

/// S1, random: 200 tokens of seeded accept counts at widths 3 and 8, fp16 and int8, through
/// the sparse range.
#[test]
fn rollback_random_rounds() {
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        for (width, dflash) in [(3usize, false), (8, true)] {
            let prefix = tokens(13, 31);
            let mut rounds = Vec::new();
            let mut total = 0;
            let mut i = 0u64;
            while total < 200 {
                let accepted = (((noise(i, 32) + 0.5) * width as f32) as usize).min(width - 1);
                rounds.push((tokens(width, 100 + i), accepted));
                total += accepted + 1;
                i += 1;
            }
            check_rollback(
                mode,
                test_config(TEST_INDEX, DlsaShortContext::DenseFallback),
                &prefix,
                &rounds,
                dflash,
                &format!("{mode} width {width}"),
            );
        }
    }
}

/// `dense_below` inside the sparse range of the tiny model: past the dense key set (positions
/// 50 on), so rows between there and the threshold would read partial lists if the policy were
/// per forward.
const THRESHOLD: usize = 96;

fn threshold_config() -> DlsaConfig {
    DlsaConfig {
        dense_below: THRESHOLD,
        ..test_config(TEST_INDEX, DlsaShortContext::DenseFallback)
    }
}

fn argmax_rows(logits: &[f32]) -> Vec<i32> {
    rows(logits).map(argmax).collect()
}

/// The `dense_below` policy is per row: a prefill chunk that crosses the threshold gives the
/// logits (within the multi-row tolerance) and greedy tokens of a single-token run, on every
/// backend, and the index of both is the same.
#[test]
fn dense_below_chunk_matches_single_tokens_across_the_threshold() {
    assert!(!crate::dlsa::reads_dense_key_set(&TEST_INDEX, 60));
    let toks = tokens(130, 51);
    for backend in super::dlsa_tests::BACKENDS {
        let config = DlsaConfig {
            backend,
            ..threshold_config()
        };
        // 60..110 crosses the threshold: 36 rows below it, 14 past it
        let block = dlsa_model(KVCacheMode::Fp16, config.clone());
        let got = run(&block, &toks, &[60, 50, 20]);
        let single = dlsa_model(KVCacheMode::Fp16, config);
        let want = run(&single, &toks, &schedule(&[60], 70));
        let what = format!("{backend:?}: chunk vs single tokens");
        assert_allclose(&got, &want, 1e-4, 1e-4, &what);
        assert_eq!(
            argmax_rows(&got),
            argmax_rows(&want),
            "{what}: greedy tokens"
        );
        assert_tables(&index_tables(&block), &index_tables(&single), 1e-6, &what);
        assert!(
            block.dlsa_stats().unwrap().sparse_forwards > 0,
            "{what}: no sparse read"
        );
    }
}

/// MTP-style 3-row and DFlash-style 8-row verifies with seeded rollbacks, through the
/// threshold: the speculative run equals the sequential one fed the accepted tokens (logits,
/// greedy tokens, tables, and a continuation), so a rollback across the threshold leaves the
/// state a single-token run would have.
#[test]
fn dense_below_verify_and_rollback_across_the_threshold() {
    for (width, dflash) in [(3usize, false), (8, true)] {
        let prefix = tokens(70, 61);
        let (mut rounds, mut at, mut crossed) = (Vec::new(), prefix.len(), 0);
        let mut i = 0u64;
        while at < THRESHOLD + 40 {
            let accepted = (((noise(i, 62) + 0.5) * width as f32) as usize).min(width - 1);
            crossed += usize::from(at < THRESHOLD && at + width > THRESHOLD);
            rounds.push((tokens(width, 200 + i), accepted));
            at += accepted + 1;
            i += 1;
        }
        assert!(
            crossed > 0,
            "width {width}: no verify crossed the threshold"
        );
        let what = format!("width {width}");
        check_rollback(
            KVCacheMode::Fp16,
            threshold_config(),
            &prefix,
            &rounds,
            dflash,
            &what,
        );
        let (kept, spec) = speculate(
            &dlsa_model(KVCacheMode::Fp16, threshold_config()),
            &prefix,
            &rounds,
            dflash,
        );
        let reference = sequential(
            &dlsa_model(KVCacheMode::Fp16, threshold_config()),
            &prefix,
            &kept,
        );
        assert_eq!(
            argmax_rows(&spec),
            argmax_rows(&reference),
            "{what}: greedy tokens"
        );
    }
}

/// A snapshot taken below the threshold restores and continues through it (a crossing chunk,
/// then decodes) bitwise like the uninterrupted run; one taken past it as well.
#[test]
fn dense_below_snapshot_restore_across_the_threshold() {
    let toks = tokens(140, 71);
    for at in [80usize, 110] {
        let model = dlsa_model(KVCacheMode::Fp16, threshold_config());
        run(&model, &toks[..at], &chunks(at));
        let snapshot = model
            .capture_sequence_state(at, None, true)
            .expect("snapshot");
        let tail = &toks[at..];
        let sched = schedule(&[30], tail.len() - 30);
        let want = run(&model, tail, &sched);
        let restored = dlsa_model(KVCacheMode::Fp16, threshold_config());
        restored
            .restore_sequence_state(seq(), &snapshot)
            .expect("restore");
        assert_bitwise(
            &run(&restored, tail, &sched),
            &want,
            &format!("restore at {at}"),
        );
        assert_tables(
            &index_tables(&restored),
            &index_tables(&model),
            0.0,
            "restore",
        );
    }
}

/// S3: DFlash2 verify, rollback and continuation at model level (the text part of
/// `dflash_multimodal_chunks_and_rollback_preserve_image_context`), with the verify closing a
/// block that the rollback reopens.
#[test]
fn dflash_verify_rollback_continuation() {
    let prompt = tokens(45, 41);
    let input = mlxcel_core::from_slice_i32(&prompt, &[1, 45]);
    let ids = [1usize, 3];
    let reference = dlsa_model(KVCacheMode::Fp16, sparse_config());
    reference
        .forward_dflash_prefill_segment_chunked(&input, &ids, usize::MAX, true, true, 16, None)
        .unwrap();
    let accepted = mlxcel_core::from_slice_i32(&[5, 19], &[1, 2]);
    let want_accepted = to_host_f32(&reference.forward_dflash_verify(&accepted, &ids).logits);
    let next = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
    let want_next = reference
        .forward_dflash_continuation(&next, &ids, usize::MAX)
        .unwrap();

    let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
    model
        .forward_dflash_prefill_segment_chunked(&input, &ids, usize::MAX, true, true, 16, None)
        .unwrap();
    let block = mlxcel_core::from_slice_i32(&[5, 19, 23, 11], &[1, 4]);
    let verify = model.forward_dflash_verify(&block, &ids);
    assert!(
        index_tables(&model).iter().all(|t| t.4 == 3 * 4),
        "the verify closed the third block"
    );
    let got = to_host_f32(&verify.logits);
    assert_allclose(
        &got[..2 * VOCAB as usize],
        &want_accepted,
        1e-4,
        1e-4,
        "accepted rows",
    );
    model.rollback_mtp_verify(&verify.gdn_states, 1, 4, true);
    assert_lockstep(&model, "after rollback");
    assert!(
        index_tables(&model).iter().all(|t| t.4 == 2 * 4),
        "the rollback reopened it"
    );
    let stats = model.dlsa_stats().unwrap();
    assert_eq!((stats.rollbacks, stats.reopened_blocks), (1, 1));
    let actual = model
        .forward_dflash_continuation(&next, &ids, usize::MAX)
        .unwrap();
    assert_allclose(
        &to_host_f32(&actual.first_logits),
        &to_host_f32(&want_next.first_logits),
        1e-4,
        1e-4,
        "continuation logits",
    );
    assert_allclose(
        &to_host_f32(&actual.hidden_concat),
        &to_host_f32(&want_next.hidden_concat),
        1e-4,
        1e-4,
        "continuation hidden",
    );
    assert_tables(
        &index_tables(&model),
        &index_tables(&reference),
        1e-6,
        "after continuation",
    );
}

/// S4: a snapshot captured after a partial rollback (an uncommitted, reopened block) restores
/// and continues bitwise like the live model (the DFlash grammar rebuild path).
#[test]
fn snapshot_after_partial_rollback() {
    let prompt = tokens(45, 51);
    let model = dlsa_model(KVCacheMode::Fp16, sparse_config());
    run(&model, &prompt, &chunks(45));
    let block = mlxcel_core::from_slice_i32(&[5, 19, 23, 11], &[1, 4]);
    let verify = model.forward_mtp_verify(&block);
    model.rollback_mtp_verify(&verify.gdn_states, 1, 4, false);
    assert_eq!(dlsa_lens(&model), vec![47, 47]);
    let snapshot = model
        .capture_sequence_state(47, None, false)
        .expect("transient snapshot after a rollback");
    let tail = tokens(10, 52);
    let want = run(&model, &tail, &[1; 10]);
    let restored = dlsa_model(KVCacheMode::Fp16, sparse_config());
    restored.restore_sequence_state(seq(), &snapshot).unwrap();
    assert_bitwise(&run(&restored, &tail, &[1; 10]), &want, "after rollback");
    assert_tables(
        &index_tables(&restored),
        &index_tables(&model),
        0.0,
        "tables",
    );
}

/// The dense sequential greedy tokens after `prompt`, and the smallest top-1 margin.
fn greedy_reference(model: &Qwen35Model, prompt: &[i32], n: usize) -> (Vec<i32>, f32) {
    model.reset_runtime_state();
    let logits = run(model, prompt, &chunks(prompt.len()));
    let mut row = rows(&logits).last().unwrap().to_vec();
    let (mut out, mut min_margin) = (Vec::new(), f32::INFINITY);
    while out.len() < n {
        min_margin = min_margin.min(margin(&row));
        let token = argmax(&row);
        out.push(token);
        row = run(model, &[token], &[1]);
    }
    (out, min_margin)
}

/// A greedy speculative loop at model level: verify `[current, proposals]`, accept while the
/// argmax equals the proposal, roll back the rest. Proposals come from `reference`, with one
/// of them corrupted every other round.
fn greedy_speculative(
    model: &Qwen35Model,
    prompt: &[i32],
    reference: &[i32],
    width: usize,
) -> Vec<i32> {
    model.reset_runtime_state();
    let logits = run(model, prompt, &chunks(prompt.len()));
    let mut out = vec![argmax(rows(&logits).last().unwrap())];
    let mut round = 0;
    while out.len() < reference.len() {
        let mut proposals: Vec<i32> = (0..width - 1)
            .map(|j| reference.get(out.len() + j).copied().unwrap_or(0))
            .collect();
        if round % 2 == 1 {
            let j = round / 2 % (width - 1);
            proposals[j] = (proposals[j] + 1) % VOCAB;
        }
        let mut block = vec![*out.last().unwrap()];
        block.extend(&proposals);
        let verify =
            model.forward_mtp_verify(&mlxcel_core::from_slice_i32(&block, &[1, width as i32]));
        let logits = to_host_f32(&verify.logits);
        let row_argmax: Vec<i32> = rows(&logits).map(argmax).collect();
        let accepted = proposals
            .iter()
            .zip(&row_argmax)
            .take_while(|(p, a)| p == a)
            .count();
        out.extend(&proposals[..accepted]);
        out.push(row_argmax[accepted]);
        model.rollback_mtp_verify(&verify.gdn_states, accepted, width, false);
        assert_lockstep(model, &format!("round {round}"));
        round += 1;
    }
    out.truncate(reference.len());
    out
}

/// S2: with a covering top-k, the greedy speculative tokens equal the dense model's under the
/// same loop and the sequential greedy run's.
#[test]
fn covering_greedy_speculation_matches_dense() {
    let prompt = tokens(300, 61);
    let dense = dlsa_test_model(KVCacheMode::Fp16);
    let (reference, min_margin) = greedy_reference(&dense, &prompt, 40);
    eprintln!("S2: smallest greedy top-1 margin {min_margin:e}");
    assert!(
        min_margin > 1e-3,
        "the fixture's greedy margin {min_margin:e} is too small to compare tokens"
    );
    for width in [3usize, 8] {
        assert_eq!(
            greedy_speculative(&dense, &prompt, &reference, width),
            reference,
            "dense speculative, width {width}"
        );
        let model = dlsa_model(KVCacheMode::Fp16, covering_config());
        assert_eq!(
            greedy_speculative(&model, &prompt, &reference, width),
            reference,
            "D-LSA speculative, width {width} (smallest margin {min_margin:e})"
        );
        assert!(model.dlsa_stats().unwrap().sparse_forwards > 0);
    }
}

// --- MTP generation through a synthetic head -----------------------------------------------

/// A bundled MTP head with the target's attention geometry and noise weights.
fn attach_mtp(model: &mut Qwen35Model) {
    let (hq, hkv, d) = (
        TEST_HEADS as i32,
        TEST_KV_HEADS as i32,
        TEST_HEAD_DIM as i32,
    );
    let mut weights = WeightMap::new();
    let ones = |shape: &[i32]| mlxcel_core::full_f32(shape, 1.0, mlxcel_core::dtype::FLOAT32);
    let mut seed = 700;
    let mut noise = |shape: &[i32], scale: f32| {
        seed += 1;
        noise_array(shape, seed, scale)
    };
    weights.insert("mtp.fc.weight".into(), noise(&[HIDDEN, 2 * HIDDEN], 0.2));
    for name in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
    ] {
        weights.insert(name.into(), ones(&[HIDDEN]));
    }
    for (projection, shape) in [
        ("q_proj", [2 * hq * d, HIDDEN]),
        ("k_proj", [hkv * d, HIDDEN]),
        ("v_proj", [hkv * d, HIDDEN]),
        ("o_proj", [HIDDEN, hq * d]),
    ] {
        weights.insert(
            format!("mtp.layers.0.self_attn.{projection}.weight"),
            noise(&shape, 0.25),
        );
    }
    for norm in ["q_norm", "k_norm"] {
        weights.insert(format!("mtp.layers.0.self_attn.{norm}.weight"), ones(&[d]));
    }
    for (projection, shape) in [
        ("gate_proj", [128, HIDDEN]),
        ("up_proj", [128, HIDDEN]),
        ("down_proj", [HIDDEN, 128]),
    ] {
        weights.insert(
            format!("mtp.layers.0.mlp.{projection}.weight"),
            noise(&shape, 0.2),
        );
    }
    model.mtp =
        Some(Qwen35MtpDraftModel::from_weights(&weights, &model.config).expect("synthetic MTP"));
}

fn mtp_generate(
    model: &Qwen35Model,
    prompt: &[i32],
    max_tokens: usize,
    block_size: usize,
    constraint: Option<&mut dyn TokenConstraint>,
) -> (Vec<i32>, usize, usize) {
    let generation = Qwen35MtpGenerator::new()
        .generate_streaming(
            model,
            prompt,
            max_tokens,
            &SamplingConfig::greedy(),
            block_size,
            None,
            &[],
            constraint,
            |_| true,
            false,
        )
        .expect("MTP generation");
    (
        generation.token_ids,
        generation.stats.proposed_draft_tokens,
        generation.stats.accepted_draft_tokens,
    )
}

/// S2 with the real generator: greedy MTP generation under a covering D-LSA equals the dense
/// model's, and both equal the sequential greedy run.
#[test]
fn covering_mtp_generation_matches_dense() {
    let prompt = tokens(300, 61);
    let mut dense = dlsa_test_model(KVCacheMode::Fp16);
    let (reference, min_margin) = greedy_reference(&dense, &prompt, 40);
    attach_mtp(&mut dense);
    let mut model = dlsa_model(KVCacheMode::Fp16, covering_config());
    attach_mtp(&mut model);
    for block_size in [2usize, 4] {
        let (want, proposed, accepted) = mtp_generate(&dense, &prompt, 40, block_size, None);
        eprintln!(
            "S2 MTP block {block_size}: {accepted} of {proposed} drafts accepted (margin {min_margin:e})"
        );
        assert!(
            accepted < proposed,
            "the synthetic head must be rejected sometimes"
        );
        assert_eq!(want, reference, "dense MTP vs sequential greedy");
        let (got, _, _) = mtp_generate(&model, &prompt, 40, block_size, None);
        assert_eq!(got, want, "D-LSA MTP, block {block_size}");
    }
}

/// A grammar stand-in: phase `k` allows the tokens `t` with `(t + k) % 3 != 0`, so drafts are
/// often disallowed and rejected; once, a committed token is replaced by a two-token splice
/// that backtracks it (a parser rewrite, which forces a rebuild from a snapshot).
struct PhaseConstraint {
    committed: Vec<i32>,
    saved: Option<Vec<i32>>,
    splice_at: usize,
    spliced: bool,
}

impl PhaseConstraint {
    fn new(splice_at: usize) -> Self {
        Self {
            committed: Vec::new(),
            saved: None,
            splice_at,
            spliced: false,
        }
    }
}

impl TokenConstraint for PhaseConstraint {
    fn begin_transaction(&mut self) -> Result<(), String> {
        if self.saved.is_some() {
            return Err("transaction already active".into());
        }
        self.saved = Some(self.committed.clone());
        Ok(())
    }

    fn commit_transaction(&mut self) -> Result<(), String> {
        self.saved
            .take()
            .map(|_| ())
            .ok_or_else(|| "no transaction".into())
    }

    fn rollback_transaction(&mut self) {
        if let Some(saved) = self.saved.take() {
            self.committed = saved;
        }
    }

    fn compute_mask(&mut self, _: &MlxArray, _: &[i32]) -> Result<ConstraintMask, String> {
        let k = self.committed.len() as i32;
        Ok(ConstraintMask::Allow(
            (0..VOCAB).filter(|t| (t + k) % 3 != 0).collect(),
        ))
    }

    fn commit_token(&mut self, token: i32) -> Result<ConstraintCommit, String> {
        if !self.spliced && self.committed.len() == self.splice_at {
            self.spliced = true;
            let commit = ConstraintCommit {
                backtrack: 1,
                tokens: vec![token, (token + 1) % VOCAB, (token + 2) % VOCAB],
                accept: false,
            };
            // The splice replaces the previous token as well
            commit.apply_to(&mut self.committed)?;
            return Ok(commit);
        }
        self.committed.push(token);
        Ok(ConstraintCommit::token(token))
    }
}

/// S5: constrained MTP generation with forced rejections and a backtracking splice: D-LSA
/// (covering) gives the dense model's tokens.
#[test]
fn constrained_mtp_generation_matches_dense() {
    let prompt = tokens(300, 71);
    let mut dense = dlsa_test_model(KVCacheMode::Fp16);
    attach_mtp(&mut dense);
    let mut model = dlsa_model(KVCacheMode::Fp16, covering_config());
    attach_mtp(&mut model);
    for block_size in [2usize, 4] {
        let mut constraint = PhaseConstraint::new(6);
        let (want, proposed, accepted) =
            mtp_generate(&dense, &prompt, 32, block_size, Some(&mut constraint));
        eprintln!("S5 block {block_size}: dense {want:?}, {accepted} of {proposed} drafts");
        assert!(constraint.spliced, "the splice ran");
        let mut constraint = PhaseConstraint::new(6);
        let (got, _, _) = mtp_generate(&model, &prompt, 32, block_size, Some(&mut constraint));
        assert_eq!(got, want, "constrained D-LSA MTP, block {block_size}");
        assert_eq!(constraint.committed, got, "the parser saw the output");
    }
}

/// Real-model checks of the serving integration (R4-R7; R8 is
/// `real_dlsa_constrained_prompt_bases_restore_after_splice_and_rollback` in `qwen3_5_dflash.rs`).
/// Run one at a time: `./gpu-lock -- cargo test --release -p qw-runtime --features dlsa --lib
/// real_ -- --ignored --test-threads=1 --nocapture`. The model, map and text conventions are
/// those of R1-R3 (`dlsa_tests::real`); DFlash2 also needs the drafter (`QW_DFLASH_DRAFT_MODEL_PATH`
/// or its default cache path).
mod real {
    use std::time::Duration;

    use super::super::dlsa_tests::real::{
        PREFILL_CHUNK, encode, host_logits, load_tokenizer, needle_prompt, prefill, real_inputs,
        repo_text,
    };
    use super::*;
    use crate::provider::Qwen35Provider;
    use crate::qwen3_5_dflash::Qwen35Dflash2Generator;

    const TOKENS: usize = 64;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Decoder {
        Mtp,
        Dflash2,
    }

    struct Generated {
        tokens: Vec<i32>,
        accepted: usize,
        proposed: usize,
        decode: Duration,
    }

    impl Generated {
        fn acceptance(&self) -> f64 {
            self.accepted as f64 / self.proposed.max(1) as f64
        }

        fn ms_per_token(&self) -> f64 {
            self.decode.as_secs_f64() * 1e3 / self.tokens.len().max(1) as f64
        }
    }

    /// The target (fp16 KV, with its MTP head) and the DFlash2 drafter.
    fn load() -> (
        Qwen35Model,
        Qwen35Dflash2Generator,
        DlsaConfig,
        std::path::PathBuf,
    ) {
        let (model_dir, config) = real_inputs();
        let draft_dir = crate::resolve_dflash2_draft_path(None).expect("the DFlash2 drafter");
        let model = Qwen35Model::load(&model_dir, KVCacheMode::Fp16).expect("load");
        assert!(model.has_mtp(), "the checkpoint has no MTP head");
        let dflash = Qwen35Dflash2Generator::new(&model, &draft_dir).expect("load the drafter");
        (model, dflash, config, model_dir)
    }

    fn generate(
        decoder: Decoder,
        model: &Qwen35Model,
        dflash: &mut Qwen35Dflash2Generator,
        prompt: &[i32],
        n: usize,
    ) -> Generated {
        let sampling = SamplingConfig::greedy();
        match decoder {
            Decoder::Mtp => {
                let g = Qwen35MtpGenerator::new()
                    .generate_streaming(
                        model,
                        prompt,
                        n,
                        &sampling,
                        3,
                        None,
                        &[],
                        None,
                        |_| true,
                        false,
                    )
                    .expect("MTP generation");
                Generated {
                    tokens: g.token_ids,
                    accepted: g.stats.accepted_draft_tokens,
                    proposed: g.stats.proposed_draft_tokens,
                    decode: g.stats.decode_time,
                }
            }
            Decoder::Dflash2 => {
                let g = dflash
                    .generate_streaming_with_prefill(
                        model,
                        prompt,
                        n,
                        &sampling,
                        None,
                        &[],
                        false,
                        None,
                        None,
                        |_, _| true,
                    )
                    .expect("DFlash2 generation");
                Generated {
                    tokens: g.token_ids,
                    accepted: g.stats.accepted_draft_tokens,
                    proposed: g.stats.proposed_draft_tokens,
                    decode: g.stats.decode_time,
                }
            }
        }
    }

    /// `None` when the runs agree, else the first differing step and the dense model's top-2
    /// margin there (D-LSA is left off).
    fn divergence(
        model: &mut Qwen35Model,
        prompt: &[i32],
        dense: &[i32],
        dlsa: &[i32],
    ) -> Option<(usize, f32)> {
        let at = dense
            .iter()
            .zip(dlsa)
            .position(|(a, b)| a != b)
            .or((dense.len() != dlsa.len()).then(|| dense.len().min(dlsa.len())))?;
        model.disable_dlsa();
        model.reset_runtime_state();
        let mut context = prompt.to_vec();
        context.extend_from_slice(&dense[..at]);
        let logits = host_logits(&prefill(model, &context));
        Some((at, margin(&logits)))
    }

    /// Dense, then D-LSA under `config`, for both decoders; the token sequences must agree.
    fn tokens_agree(what: &str, prompt: &[i32], config: DlsaConfig) {
        let (mut model, mut dflash, _, _) = load();
        let mut failures = Vec::new();
        for decoder in [Decoder::Mtp, Decoder::Dflash2] {
            model.disable_dlsa();
            let dense = generate(decoder, &model, &mut dflash, prompt, TOKENS);
            model.enable_dlsa(config.clone()).expect("enable D-LSA");
            let dlsa = generate(decoder, &model, &mut dflash, prompt, TOKENS);
            let stats = model.dlsa_stats().unwrap();
            eprintln!(
                "{what} {decoder:?}: acceptance dense {:.3} D-LSA {:.3}; decode dense {:.1} ms/token \
                 D-LSA {:.1} ms/token; sparse layer forwards {}, rollbacks {}, reopened blocks {}",
                dense.acceptance(),
                dlsa.acceptance(),
                dense.ms_per_token(),
                dlsa.ms_per_token(),
                stats.sparse_forwards,
                stats.rollbacks,
                stats.reopened_blocks
            );
            if let Some((at, margin)) = divergence(&mut model, prompt, &dense.tokens, &dlsa.tokens)
            {
                failures.push(format!(
                    "{decoder:?}: first divergence at token {at} (dense top-2 margin {margin:e}): \
                     dense {:?} D-LSA {:?}",
                    dense.tokens, dlsa.tokens
                ));
            }
        }
        assert!(failures.is_empty(), "{what}: {failures:#?}");
    }

    /// R4: about 4k tokens under the defaults (every forward on the dense fallback): MTP and
    /// DFlash2 tokens with D-LSA equal those without.
    #[test]
    #[ignore = "requires the real checkpoint, the DFlash2 drafter and the converted D-LSA map"]
    fn real_speculative_short_context_matches_dense() {
        let (model_dir, config) = real_inputs();
        let prompt = encode(&load_tokenizer(&model_dir), &repo_text(), 4000);
        tokens_agree("R4 4000 tokens", &prompt, config);
    }

    /// R5: about 8k tokens, sparse reads with a covering top-k (8192, top_p 1024): 64 greedy
    /// MTP and DFlash2 tokens with D-LSA equal those without.
    #[test]
    #[ignore = "requires the real checkpoint, the DFlash2 drafter and the converted D-LSA map"]
    fn real_speculative_covering_matches_dense() {
        let (model_dir, config) = real_inputs();
        let prompt = encode(&load_tokenizer(&model_dir), &repo_text(), 8000);
        let config = DlsaConfig {
            index: DlsaIndexConfig {
                top_k: 8192,
                top_p: 1024,
                ..config.index
            },
            short_context: DlsaShortContext::Sparse,
            ..config
        };
        tokens_agree("R5 8000 tokens covering", &prompt, config);
    }

    /// R6: the needle at 16k and 32k under the defaults, with MTP and DFlash2: D-LSA finds it
    /// when dense does; acceptance, decode speed and reopened blocks are reported.
    #[test]
    #[ignore = "requires the real checkpoint, the DFlash2 drafter and the converted D-LSA map"]
    fn real_speculative_needle_16k_32k() {
        let prompts = {
            let (model_dir, _) = real_inputs();
            let provider = Qwen35Provider::load(&model_dir, KVCacheMode::Fp16).expect("load");
            [16_384usize, 32_768].map(|target| needle_prompt(&provider, target))
        };
        let (mut model, mut dflash, config, model_dir) = load();
        let tokenizer = load_tokenizer(&model_dir);
        let text = |tokens: &[i32]| {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            tokenizer.decode(&ids, true).expect("decode")
        };
        for prompt in &prompts {
            for decoder in [Decoder::Mtp, Decoder::Dflash2] {
                model.disable_dlsa();
                let dense = generate(decoder, &model, &mut dflash, prompt, 48);
                model.enable_dlsa(config.clone()).expect("enable D-LSA");
                let dlsa = generate(decoder, &model, &mut dflash, prompt, 48);
                let stats = model.dlsa_stats().unwrap();
                let (dense_text, dlsa_text) = (text(&dense.tokens), text(&dlsa.tokens));
                eprintln!(
                    "R6 {} tokens {decoder:?}: acceptance dense {:.3} D-LSA {:.3}; decode dense {:.1} \
                     ms/token D-LSA {:.1} ms/token; rollbacks {}, reopened blocks {}; dense {:?} \
                     D-LSA {:?}",
                    prompt.len(),
                    dense.acceptance(),
                    dlsa.acceptance(),
                    dense.ms_per_token(),
                    dlsa.ms_per_token(),
                    stats.rollbacks,
                    stats.reopened_blocks,
                    dense_text,
                    dlsa_text
                );
                if dense_text.contains("7291") {
                    assert!(
                        dlsa_text.contains("7291"),
                        "{decoder:?} at {}: dense found the needle, D-LSA did not: {dlsa_text:?}",
                        prompt.len()
                    );
                } else {
                    eprintln!(
                        "R6 {decoder:?} at {}: dense did not find the needle either",
                        prompt.len()
                    );
                }
            }
        }
    }

    /// `steps` greedy tokens from `logits` on, and each step's logits.
    fn decode_greedy(
        model: &Qwen35Model,
        logits: Vec<f32>,
        steps: usize,
    ) -> (Vec<Vec<f32>>, Vec<i32>) {
        let mut all = vec![logits];
        let mut tokens = Vec::new();
        for step in 0..steps {
            let token = super::argmax(&all[step]);
            tokens.push(token);
            if step + 1 < steps {
                let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
                all.push(host_logits(&model.forward_last_logits(
                    &input,
                    &mut [],
                    None,
                    0,
                )));
            }
        }
        (all, tokens)
    }

    /// R7: a 16k prompt snapshotted at 12,288 tokens (chunk-aligned), restored (also through
    /// the portable form), the rest prefilled and 32 greedy tokens decoded: logits and tokens
    /// equal the uninterrupted run bitwise.
    #[test]
    #[ignore = "requires the real checkpoint and the converted D-LSA map"]
    fn real_snapshot_restore_is_bitwise() {
        const AT: usize = 12_288;
        const STEPS: usize = 32;
        assert!(AT.is_multiple_of(PREFILL_CHUNK));
        let (model_dir, config) = real_inputs();
        let prompt = encode(&load_tokenizer(&model_dir), &repo_text(), 16_384);
        let mut model = Qwen35Model::load(&model_dir, KVCacheMode::Fp16).expect("load");
        model.enable_dlsa(config).expect("enable D-LSA");

        model.reset_runtime_state();
        let last = host_logits(&prefill(&model, &prompt));
        let (want_logits, want_tokens) = decode_greedy(&model, last, STEPS);
        assert!(model.dlsa_stats().unwrap().sparse_forwards > 0);

        model.reset_runtime_state();
        prefill(&model, &prompt[..AT]);
        let snapshot = model
            .capture_sequence_state(AT, None, true)
            .expect("D-LSA snapshot");
        eprintln!(
            "R7 snapshot at {AT}: {:.1} MiB",
            snapshot.nbytes() as f64 / (1 << 20) as f64
        );
        let portable = model_from_portable(portable_model_state(&snapshot), false, false)
            .expect("portable D-LSA snapshot");
        for (what, snapshot) in [("in memory", &snapshot), ("portable", &portable)] {
            model.reset_runtime_state();
            model
                .restore_sequence_state(seq(), snapshot)
                .expect("restore the snapshot");
            let last = host_logits(&prefill(&model, &prompt[AT..]));
            let (logits, tokens) = decode_greedy(&model, last, STEPS);
            assert_eq!(tokens, want_tokens, "R7 {what}: tokens");
            for (step, (a, b)) in logits.iter().zip(&want_logits).enumerate() {
                assert_bitwise(a, b, &format!("R7 {what}: logits of step {step}"));
            }
        }
    }
}
