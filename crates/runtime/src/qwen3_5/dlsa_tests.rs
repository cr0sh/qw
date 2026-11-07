// SPDX-License-Identifier: Apache-2.0
//
//! D-LSA on a tiny synthetic Qwen3.5 hybrid (4 layers, full attention at 1 and 3, GQA 4/2,
//! head_dim 32): the dense fallback is bit-identical, a covering top-k equals dense, prefill /
//! decode / chunk schedules agree, and the refusals and lifecycle hold (`docs/dlsa.md` §12).
//!
//! These tests need Metal: run them under `./gpu-lock -- cargo test -p qw-runtime --features dlsa`.

use std::path::PathBuf;

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;

use super::*;
use crate::dlsa::test_model::{
    TEST_HEAD_DIM, TEST_HEADS, TEST_INDEX, TEST_KV_HEADS, allclose_excess, noise, noise_array,
    test_maps,
};
use crate::dlsa::{
    DlsaBackend, DlsaConfig, DlsaIndexConfig, DlsaMapDtype, DlsaShortContext, to_host_f32,
};

pub(super) const HIDDEN: i32 = 64;
pub(super) const VOCAB: i32 = 64;
const LAYERS: usize = 4;
pub(super) const FULL_LAYERS: [usize; 2] = [1, 3];

pub(super) fn dlsa_test_model(mode: KVCacheMode) -> Qwen35Model {
    let config: Qwen35Config = serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_5_text",
        "max_position_embeddings": 4096,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 128,
        "num_attention_heads": TEST_HEADS,
        "num_key_value_heads": TEST_KV_HEADS,
        "head_dim": TEST_HEAD_DIM,
        "full_attention_interval": 2,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "vocab_size": VOCAB,
        "tie_word_embeddings": true
    }))
    .expect("small hybrid config");
    let (hq, hkv, d) = (
        TEST_HEADS as i32,
        TEST_KV_HEADS as i32,
        TEST_HEAD_DIM as i32,
    );
    let mut weights = WeightMap::new();
    let mut seed = 0u64;
    let mut insert = |name: String, shape: &[i32], scale: f32| {
        seed += 1;
        weights.insert(name, noise_array(shape, seed, scale));
    };
    let ones = |shape: &[i32]| mlxcel_core::full_f32(shape, 1.0, mlxcel_core::dtype::FLOAT32);
    insert("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 2.0);
    let mut fixed: Vec<(String, UniquePtr<MlxArray>)> =
        vec![("model.norm.weight".into(), ones(&[HIDDEN]))];
    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            fixed.push((format!("{prefix}.{norm}.weight"), ones(&[HIDDEN])));
        }
        for (projection, shape) in [
            ("gate_proj", [128, HIDDEN]),
            ("up_proj", [128, HIDDEN]),
            ("down_proj", [HIDDEN, 128]),
        ] {
            insert(format!("{prefix}.mlp.{projection}.weight"), &shape, 0.2);
        }
        if FULL_LAYERS.contains(&layer) {
            for (projection, shape) in [
                ("q_proj", [2 * hq * d, HIDDEN]),
                ("k_proj", [hkv * d, HIDDEN]),
                ("v_proj", [hkv * d, HIDDEN]),
                ("o_proj", [HIDDEN, hq * d]),
            ] {
                insert(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    &shape,
                    0.25,
                );
            }
            for norm in ["q_norm", "k_norm"] {
                fixed.push((format!("{prefix}.self_attn.{norm}.weight"), ones(&[d])));
            }
        } else {
            for (projection, shape) in [
                ("in_proj_qkv", [96, HIDDEN]),
                ("in_proj_z", [32, HIDDEN]),
                ("in_proj_a", [1, HIDDEN]),
                ("in_proj_b", [1, HIDDEN]),
                ("out_proj", [HIDDEN, 32]),
            ] {
                insert(
                    format!("{prefix}.linear_attn.{projection}.weight"),
                    &shape,
                    0.2,
                );
            }
            insert(
                format!("{prefix}.linear_attn.conv1d.weight"),
                &[96, 4, 1],
                0.2,
            );
            fixed.push((format!("{prefix}.linear_attn.norm.weight"), ones(&[32])));
            fixed.push((
                format!("{prefix}.linear_attn.dt_bias"),
                mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32),
            ));
            fixed.push((
                format!("{prefix}.linear_attn.A_log"),
                mlxcel_core::full_f32(&[1], -1.0, mlxcel_core::dtype::FLOAT32),
            ));
        }
    }
    weights.extend(fixed);
    Qwen35Model::from_weights(&weights, &config, mode).expect("small hybrid weights")
}

/// Every backend: the task-3 oracle, the Metal kernels, and the parity mode.
pub(super) const BACKENDS: [DlsaBackend; 3] = [
    DlsaBackend::Mlx,
    DlsaBackend::Metal,
    DlsaBackend::MetalSelect,
];

/// The test config on the backend `QW_DLSA_BACKEND` names (default Metal), so the tests that do
/// not loop over [`BACKENDS`] (the serving tests among them) run under any backend.
pub(super) fn test_config(index: DlsaIndexConfig, short_context: DlsaShortContext) -> DlsaConfig {
    DlsaConfig {
        map_dtype: DlsaMapDtype::F32,
        index,
        short_context,
        prefill_mask_rows: 24,
        // The tiny model's contexts sit far below the serving threshold
        dense_below: 0,
        backend: DlsaBackend::from_env().expect("QW_DLSA_BACKEND"),
        ..DlsaConfig::new(PathBuf::new())
    }
}

/// `(rtol, atol)` for a backend's logits against task 3 or dense: `fp16` as given; the Metal
/// read under int8 KV 1e-3, as a K2 rounding difference (~1e-5) can move a later layer's key
/// across an int8 quantization step.
pub(super) fn metal_tolerance(
    mode: KVCacheMode,
    backend: DlsaBackend,
    fp16: (f32, f32),
) -> (f32, f32) {
    if backend == DlsaBackend::Metal && mode == KVCacheMode::Int8 {
        (1e-3, 1e-3)
    } else {
        fp16
    }
}

pub(super) fn backend_config(
    index: DlsaIndexConfig,
    short_context: DlsaShortContext,
    backend: DlsaBackend,
) -> DlsaConfig {
    DlsaConfig {
        backend,
        ..test_config(index, short_context)
    }
}

pub(super) fn dlsa_model(mode: KVCacheMode, config: DlsaConfig) -> Qwen35Model {
    let mut model = dlsa_test_model(mode);
    model
        .enable_dlsa_with_maps(config, test_maps(&FULL_LAYERS))
        .expect("enable D-LSA");
    model
}

pub(super) fn tokens(n: usize, seed: u64) -> Vec<i32> {
    (0..n as u64)
        .map(|i| ((noise(i, seed) + 0.5) * VOCAB as f32) as i32 % VOCAB)
        .collect()
}

/// Every row's logits of `forward` calls over `schedule`, concatenated `[sum(schedule), VOCAB]`.
pub(super) fn run(model: &Qwen35Model, tokens: &[i32], schedule: &[usize]) -> Vec<f32> {
    assert_eq!(schedule.iter().sum::<usize>(), tokens.len());
    let mut out = Vec::with_capacity(tokens.len() * VOCAB as usize);
    let mut at = 0;
    for &rows in schedule {
        let input = mlxcel_core::from_slice_i32(&tokens[at..at + rows], &[1, rows as i32]);
        let logits = model.forward(&input, &mut [], None);
        out.extend(to_host_f32(&logits));
        at += rows;
    }
    out
}

/// `prefill` tokens in chunks, then single-token decode steps.
pub(super) fn schedule(chunks: &[usize], decode: usize) -> Vec<usize> {
    chunks
        .iter()
        .copied()
        .chain(std::iter::repeat_n(1, decode))
        .collect()
}

pub(super) fn dlsa_lens(model: &Qwen35Model) -> Vec<usize> {
    model.dlsa.as_ref().expect("D-LSA on").cache_lens()
}

#[track_caller]
pub(super) fn assert_bitwise(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    assert!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "{what}: non-finite logits"
    );
    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differing, 0, "{what}: {differing} logits differ bitwise");
}

#[track_caller]
pub(super) fn assert_allclose(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, what: &str) {
    let excess = allclose_excess(actual, expected, rtol, atol);
    assert!(
        excess <= 0.0,
        "{what}: exceeds allclose(rtol {rtol:e}, atol {atol:e}) by {excess:e}"
    );
}

#[test]
fn short_context_is_bitwise_dense() {
    let toks = tokens(48, 1);
    let sched = schedule(&[40], 8);
    let dense = run(&dlsa_test_model(KVCacheMode::Fp16), &toks, &sched);
    for backend in BACKENDS {
        let model = dlsa_model(
            KVCacheMode::Fp16,
            backend_config(TEST_INDEX, DlsaShortContext::DenseFallback, backend),
        );
        let got = run(&model, &toks, &sched);
        assert_bitwise(&got, &dense, &format!("dense fallback ({backend:?})"));
        let stats = model.dlsa_stats().unwrap();
        assert_eq!(stats.sparse_forwards, 0);
        assert_eq!(stats.dense_key_set_forwards, 2 * sched.len() as u64);
        // The index is built all along, so later forwards find the blocks clustered
        assert_eq!(dlsa_lens(&model), vec![48, 48]);
        assert!(model.dlsa_memory_bytes() > 0);
    }
}

#[test]
fn covering_top_k_matches_dense() {
    let covering = DlsaIndexConfig {
        top_k: 512,
        top_p: 128,
        ..TEST_INDEX
    };
    let toks = tokens(310, 2);
    let sched = schedule(&[64, 37, 64, 37, 64, 34], 10);
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        let dense = run(&dlsa_test_model(mode), &toks, &sched);
        for backend in BACKENDS {
            let model = dlsa_model(
                mode,
                backend_config(covering, DlsaShortContext::Sparse, backend),
            );
            let got = run(&model, &toks, &sched);
            eprintln!(
                "covering top-k ({mode}, {backend:?}): max abs vs dense {:e}, max |dense| {:e}",
                crate::dlsa::test_model::max_abs_diff(&got, &dense),
                dense.iter().fold(0f32, |m, v| m.max(v.abs()))
            );
            // K2 rounds differently from SDPA (about 1e-5 here); under int8 KV a later
            // layer's quantization can turn that into one quantization step of a key
            let (rtol, atol) = metal_tolerance(mode, backend, (1e-4, 1e-5));
            assert_allclose(
                &got,
                &dense,
                rtol,
                atol,
                &format!("covering top-k ({mode}, {backend:?})"),
            );
            let stats = model.dlsa_stats().unwrap();
            assert_eq!(stats.dense_key_set_forwards, 0);
            assert_eq!(stats.sparse_forwards, 2 * sched.len() as u64);
            // Every row read its whole causal context
            let rows = toks.len() as u64;
            assert_eq!(stats.sparse_rows, 2 * rows);
            assert_eq!(stats.keys_read, 2 * rows * (rows + 1) / 2, "{backend:?}");
        }
    }
}

/// The schedules of `decode_prefill_consistency`.
fn consistency_schedules() -> [Vec<usize>; 3] {
    [
        vec![260],
        schedule(&[100], 160),
        schedule(&[37, 37, 37, 37, 37, 37, 37], 1),
    ]
}

#[test]
fn decode_prefill_consistency() {
    let toks = tokens(260, 3);
    for backend in BACKENDS {
        let config = backend_config(TEST_INDEX, DlsaShortContext::DenseFallback, backend);
        let mut reference: Option<(Vec<f32>, Qwen35Model)> = None;
        for sched in &consistency_schedules() {
            let model = dlsa_model(KVCacheMode::Fp16, config.clone());
            let logits = run(&model, &toks, sched);
            assert_eq!(dlsa_lens(&model), vec![260, 260]);
            let Some((ref_logits, ref_model)) = &reference else {
                let stats = model.dlsa_stats().unwrap();
                assert_eq!(stats.sparse_forwards, 2, "one sparse forward a layer");
                reference = Some((logits, model));
                continue;
            };
            let what = format!("{backend:?}: schedule of {} forwards", sched.len());
            assert_allclose(&logits, ref_logits, 1e-4, 1e-4, &what);
            let mut a = model.dlsa.as_ref().unwrap().state.borrow_mut();
            let mut b = ref_model.dlsa.as_ref().unwrap().state.borrow_mut();
            for (layer, (x, y)) in a.0.iter_mut().zip(b.0.iter_mut()).enumerate() {
                assert_eq!(x.members(), y.members(), "{what}: layer {layer} members");
                assert_eq!(x.sizes(), y.sizes(), "{what}: layer {layer} sizes");
                let diff = crate::dlsa::test_model::max_abs_diff(x.centroids(), y.centroids());
                assert!(
                    diff <= 1e-6,
                    "{what}: layer {layer} centroids differ by {diff:e}"
                );
                let diff = crate::dlsa::test_model::max_abs_diff(x.tail_keys(), y.tail_keys());
                assert!(
                    diff <= 1e-6,
                    "{what}: layer {layer} tail keys differ by {diff:e}"
                );
            }
        }
    }
}

/// Metal select + the task-3 MLX read gives the task-3 logits bit for bit, on every schedule
/// and both KV modes; the Metal read is close to them.
#[test]
fn metal_select_is_bitwise_task3_and_metal_read_is_close() {
    let toks = tokens(260, 3);
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        for sched in &consistency_schedules() {
            let logits = |backend| {
                let config = backend_config(TEST_INDEX, DlsaShortContext::DenseFallback, backend);
                let model = dlsa_model(mode, config);
                let logits = run(&model, &toks, sched);
                (logits, model.dlsa_stats().unwrap())
            };
            let (task3, task3_stats) = logits(DlsaBackend::Mlx);
            let what = format!("{mode}, {} forwards", sched.len());
            let (parity, parity_stats) = logits(DlsaBackend::MetalSelect);
            assert_bitwise(
                &parity,
                &task3,
                &format!("metal select + mlx read ({what})"),
            );
            assert_eq!(parity_stats.keys_read, task3_stats.keys_read, "{what}");
            let (metal, metal_stats) = logits(DlsaBackend::Metal);
            let (rtol, atol) = metal_tolerance(mode, DlsaBackend::Metal, (1e-4, 1e-4));
            assert_allclose(&metal, &task3, rtol, atol, &format!("metal ({what})"));
            assert_eq!(metal_stats.keys_read, task3_stats.keys_read, "{what}");
            assert_eq!(metal_stats.sparse_rows, task3_stats.sparse_rows, "{what}");
        }
    }
}

/// Greedy tokens of `steps` decode steps after a chunked prefill of `prompt`.
fn greedy(model: &Qwen35Model, prompt: &[i32], chunk: usize, steps: usize) -> Vec<i32> {
    let argmax = |logits: &[f32]| {
        let last = &logits[logits.len() - VOCAB as usize..];
        (0..VOCAB as usize).fold(0, |best, i| if last[i] > last[best] { i } else { best }) as i32
    };
    let mut last = Vec::new();
    for part in prompt.chunks(chunk) {
        let input = mlxcel_core::from_slice_i32(part, &[1, part.len() as i32]);
        last = to_host_f32(&model.forward(&input, &mut [], None));
    }
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let token = argmax(&last);
        out.push(token);
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        last = to_host_f32(&model.forward(&input, &mut [], None));
    }
    out
}

#[test]
fn metal_greedy_tokens_equal_task3() {
    let prompt = tokens(150, 7);
    let run_backend = |backend| {
        let config = backend_config(TEST_INDEX, DlsaShortContext::DenseFallback, backend);
        let model = dlsa_model(KVCacheMode::Fp16, config);
        let tokens = greedy(&model, &prompt, 48, 60);
        assert!(model.dlsa_stats().unwrap().sparse_forwards > 0);
        tokens
    };
    let task3 = run_backend(DlsaBackend::Mlx);
    assert_eq!(
        run_backend(DlsaBackend::Metal),
        task3,
        "metal greedy tokens"
    );
    assert_eq!(
        run_backend(DlsaBackend::MetalSelect),
        task3,
        "parity greedy tokens"
    );
}

#[test]
fn refusals_and_snapshot_guards() {
    let config = test_config(TEST_INDEX, DlsaShortContext::DenseFallback);
    for mode in [
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo3Asym,
        KVCacheMode::Turbo4Delegated,
    ] {
        let mut model = dlsa_test_model(mode);
        let err = model
            .enable_dlsa_with_maps(config.clone(), test_maps(&FULL_LAYERS))
            .expect_err("packed caches are refused")
            .to_string();
        assert!(
            err.contains(&mode.to_string()) && err.contains("fp16"),
            "{mode}: {err}"
        );
        assert!(!model.dlsa_enabled());
    }

    let mut model = dlsa_test_model(KVCacheMode::Int8);
    let err = model
        .enable_dlsa_with_maps(config.clone(), test_maps(&[1]))
        .expect_err("a map without layer 3")
        .to_string();
    assert!(err.contains("layer 3"), "{err}");
    let bad = DlsaConfig {
        prefill_mask_rows: 0,
        ..config.clone()
    };
    assert!(
        model
            .enable_dlsa_with_maps(bad, test_maps(&FULL_LAYERS))
            .is_err()
    );
    assert!(!model.dlsa_enabled());
    // Int8 is accepted
    let toks = tokens(20, 4);
    model
        .enable_dlsa_with_maps(config.clone(), test_maps(&FULL_LAYERS))
        .expect("int8 is supported");
    run(&model, &toks, &[20]);
    assert_eq!(dlsa_lens(&model), vec![20, 20]);

    // Off: snapshots work as before
    let mut model = dlsa_test_model(KVCacheMode::Fp16);
    run(&model, &toks, &[20]);
    assert!(model.supports_snapshot_reuse());
    let snapshot = model
        .capture_sequence_state(20, None, true)
        .expect("dense snapshot");
    let seq = SequenceId::from_raw(1);
    model
        .restore_sequence_state(seq, &snapshot)
        .expect("dense restore");

    // On: snapshots carry the index; a dense snapshot is refused, and so are image prefills
    model
        .enable_dlsa_with_maps(config, test_maps(&FULL_LAYERS))
        .expect("enable D-LSA");
    run(&model, &toks, &[20]);
    assert!(model.supports_snapshot_reuse());
    let sparse_snapshot = model
        .capture_sequence_state(20, None, true)
        .expect("D-LSA snapshot");
    let err = model
        .restore_sequence_state(seq, &snapshot)
        .expect_err("a dense snapshot under D-LSA");
    assert!(err.contains("D-LSA"), "{err}");
    model
        .restore_sequence_state(seq, &sparse_snapshot)
        .expect("D-LSA restore");
    assert_eq!(dlsa_lens(&model), vec![20, 20]);
    let input = mlxcel_core::from_slice_i32(&toks, &[1, 20]);
    let embeddings = model.embed_tokens.forward(&input);
    let positions = mlxcel_core::from_slice_i32(
        &(0..3).flat_map(|_| 0..20).collect::<Vec<i32>>(),
        &[3, 1, 20],
    );
    let err = model
        .forward_dflash_prefill_segment_chunked(
            &input,
            &[0, 1],
            4,
            true,
            true,
            8,
            Some((&embeddings, &positions, 0)),
        )
        .err()
        .expect("DFlash2 image prefill under D-LSA");
    assert!(err.contains("D-LSA"), "{err}");
    let err = model
        .forward_mtp_prefill_chunks(
            &input,
            Some(&embeddings),
            Some(&positions),
            Some(0),
            |_, _, _| {},
        )
        .err()
        .expect("MTP image prefill under D-LSA");
    assert!(err.contains("D-LSA"), "{err}");
    // Text prefills and verifies run, and keep the index in step
    model
        .forward_dflash_prefill_segment(&input, &[0, 1], 4, true, true)
        .expect("DFlash2 prefill under D-LSA");
    model
        .forward_mtp_text_suffix_chunks(&input, |_, _| {})
        .expect("MTP suffix prefill under D-LSA");
    assert_eq!(dlsa_lens(&model), vec![40, 40]);
    let block = mlxcel_core::from_slice_i32(&toks[..4], &[1, 4]);
    let verify = model.forward_mtp_verify(&block);
    assert_eq!(dlsa_lens(&model), vec![44, 44]);
    model.rollback_mtp_verify(&verify.gdn_states, 1, 4, true);
    assert_eq!(dlsa_lens(&model), vec![42, 42]);

    model.disable_dlsa();
    assert!(model.supports_snapshot_reuse());
    assert!(model.dlsa_stats().is_none());
}

#[test]
fn reset_restarts_the_index_and_reproduces_outputs() {
    let toks = tokens(130, 5);
    let sched = schedule(&[70, 50], 10);
    for backend in BACKENDS {
        let model = dlsa_model(
            KVCacheMode::Fp16,
            backend_config(TEST_INDEX, DlsaShortContext::DenseFallback, backend),
        );
        let first = run(&model, &toks, &sched);
        assert!(model.dlsa_stats().unwrap().sparse_forwards > 0);
        model.reset_runtime_state();
        assert_eq!(dlsa_lens(&model), vec![0, 0]);
        assert_eq!(model.dlsa_stats().unwrap(), Default::default());
        let second = run(&model, &toks, &sched);
        assert_bitwise(&second, &first, &format!("after reset ({backend:?})"));
    }
}

#[test]
fn disable_restores_bitwise_dense() {
    let toks = tokens(120, 6);
    let sched = schedule(&[60, 50], 10);
    let dense = run(&dlsa_test_model(KVCacheMode::Fp16), &toks, &sched);
    for backend in BACKENDS {
        let mut model = dlsa_model(
            KVCacheMode::Fp16,
            backend_config(TEST_INDEX, DlsaShortContext::DenseFallback, backend),
        );
        let sparse = run(&model, &toks, &sched);
        assert!(
            allclose_excess(&sparse, &dense, 1e-4, 1e-4) > 0.0,
            "the sparse run should differ from dense at this sparsity"
        );
        model.disable_dlsa();
        assert!(!model.dlsa_enabled());
        let got = run(&model, &toks, &sched);
        assert_bitwise(&got, &dense, &format!("after disable ({backend:?})"));
    }
}

/// Real-model checks with the released map (`docs/dlsa.md` §12). Run one at a time:
/// `./gpu-lock -- cargo test --release -p qw-runtime --features dlsa --lib real_ -- --ignored
/// --test-threads=1 --nocapture`. `QW_MODEL_PATH` / `QW_DLSA_MAP_PATH` override the default
/// cache paths; `QW_DLSA_MAP_KIND=pca` uses the PCA step-0 map instead of the distilled one.
pub(super) mod real {
    use std::path::Path;
    use std::time::{Duration, Instant};

    use tokenizers::Tokenizer;

    use super::*;
    use crate::dlsa::{DlsaMapKind, DlsaStats, resolve_dlsa_map_path};
    use crate::provider::{
        ChatMessage, ChatMessageContent, GenerationRequest, Qwen35GenerationMode, Qwen35Provider,
        SamplingOptions,
    };

    pub(in crate::qwen3_5) const PREFILL_CHUNK: usize = 1536;
    pub(in crate::qwen3_5) const NEEDLE: &str = "The vault code is 7291-ALPHA.";

    fn map_kind() -> DlsaMapKind {
        match std::env::var("QW_DLSA_MAP_KIND").as_deref() {
            Ok("pca") => DlsaMapKind::PcaStep0,
            _ => DlsaMapKind::Distilled,
        }
    }

    /// The checkpoint and the converted map, checked first so a missing file fails at once.
    pub(in crate::qwen3_5) fn real_inputs() -> (PathBuf, DlsaConfig) {
        let model = crate::resolve_model_path(None).expect(
            "QW_MODEL_PATH or the default model cache must hold the Qwen3.8-27B checkpoint",
        );
        let map = resolve_dlsa_map_path(map_kind(), None).expect(
            "the converted D-LSA map (run `qw download --dlsa-map`, or set QW_DLSA_MAP_PATH)",
        );
        eprintln!("model {} map {}", model.display(), map.display());
        // The sparse read from the dense-key-set boundary on, as tasks 3-5 measured it; the
        // served `dense_below` default would run these 16k-32k prompts mostly dense
        let config = DlsaConfig {
            dense_below: 0,
            ..DlsaConfig::new(map)
        };
        (model, config)
    }

    /// Deterministic long text: files of `dirs` (relative to the repository root), each sorted
    /// and walked recursively, keeping files with one of `extensions`.
    fn corpus(dirs: &[&str], extensions: &[&str]) -> String {
        fn walk(dir: &Path, extensions: &[&str], out: &mut Vec<PathBuf>) {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
                .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    walk(&path, extensions, out);
                } else if path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| extensions.contains(&e))
                    // D-LSA's own files change with the results they report
                    && !path.to_string_lossy().contains("dlsa")
                {
                    out.push(path);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut files = Vec::new();
        for dir in dirs {
            walk(&root.join(dir), extensions, &mut files);
        }
        let mut text = String::new();
        for file in files {
            text.push_str(&std::fs::read_to_string(&file).unwrap());
            text.push('\n');
        }
        // Keep chat-template control strings in the sources from parsing as special tokens
        text.replace("<|", "< |")
    }

    pub(in crate::qwen3_5) fn repo_text() -> String {
        corpus(&["docs", "crates/runtime/src"], &["md", "rs"])
    }

    fn core_text() -> String {
        corpus(&["crates/mlxcel-core/src"], &["rs"])
    }

    pub(in crate::qwen3_5) fn encode(tokenizer: &Tokenizer, text: &str, n: usize) -> Vec<i32> {
        let ids: Vec<i32> = tokenizer
            .encode(text, false)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|&t| t as i32)
            .collect();
        assert!(ids.len() >= n, "the corpus has only {} tokens", ids.len());
        ids[..n].to_vec()
    }

    pub(in crate::qwen3_5) fn load_tokenizer(model_dir: &Path) -> Tokenizer {
        Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer")
    }

    pub(in crate::qwen3_5) fn host_logits(logits: &MlxArray) -> Vec<f32> {
        to_host_f32(logits)
    }

    pub(in crate::qwen3_5) fn argmax(logits: &[f32]) -> i32 {
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "non-finite logit {v} at {i}");
            if v > logits[best] {
                best = i;
            }
        }
        best as i32
    }

    /// Prefill `tokens` in chunks through `forward_last_logits`; the last chunk's last logits.
    pub(in crate::qwen3_5) fn prefill(model: &Qwen35Model, tokens: &[i32]) -> UniquePtr<MlxArray> {
        let mut last = None;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let input = mlxcel_core::from_slice_i32(chunk, &[1, chunk.len() as i32]);
            let logits = model.forward_last_logits(&input, &mut [], None, chunk.len() - 1);
            mlxcel_core::eval(&logits);
            last = Some(logits);
        }
        model.after_prefill();
        last.expect("a non-empty prompt")
    }

    /// A fresh greedy run: the prefill's last logits and each decode step's, and the tokens.
    pub(in crate::qwen3_5) fn greedy(
        model: &Qwen35Model,
        prompt: &[i32],
        steps: usize,
    ) -> (Vec<Vec<f32>>, Vec<i32>, Duration, Duration) {
        model.reset_runtime_state();
        let started = Instant::now();
        let mut logits = vec![host_logits(&prefill(model, prompt))];
        let prefill_time = started.elapsed();
        let started = Instant::now();
        let mut tokens = Vec::with_capacity(steps);
        for step in 0..steps {
            let token = argmax(&logits[step]);
            tokens.push(token);
            if step + 1 < steps {
                let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
                logits.push(host_logits(&model.forward_last_logits(
                    &input,
                    &mut [],
                    None,
                    0,
                )));
            }
        }
        (logits, tokens, prefill_time, started.elapsed())
    }

    fn report_stats(what: &str, stats: Option<DlsaStats>, memory: usize) {
        let Some(s) = stats else { return };
        let per_row = if s.sparse_rows > 0 {
            s.keys_read as f64 / s.sparse_rows as f64
        } else {
            0.0
        };
        eprintln!(
            "{what}: dense-key-set layer forwards {}, sparse layer forwards {}, sparse rows {}, keys/row {per_row:.0}, index {:.2} s, select {:.2} s, index memory {:.1} MiB",
            s.dense_key_set_forwards,
            s.sparse_forwards,
            s.sparse_rows,
            s.index_time.as_secs_f64(),
            s.select_time.as_secs_f64(),
            memory as f64 / (1 << 20) as f64
        );
    }

    /// R1: at ~4k tokens every forward reads the dense key set, so D-LSA on is bit-identical.
    #[test]
    #[ignore = "requires the real checkpoint and the converted D-LSA map"]
    fn real_short_context_is_bitwise_dense() {
        let (model_dir, config) = real_inputs();
        let prompt = encode(&load_tokenizer(&model_dir), &repo_text(), 4000);
        let mut model = Qwen35Model::load(&model_dir, KVCacheMode::Fp16).expect("load");
        let (dense_logits, dense_tokens, dp, dd) = greedy(&model, &prompt, 16);
        model.enable_dlsa(config).expect("enable D-LSA");
        let (logits, tokens, sp, sd) = greedy(&model, &prompt, 16);
        eprintln!(
            "R1 4000 tokens: dense prefill {:.2} s decode {:.2} s; D-LSA prefill {:.2} s decode {:.2} s",
            dp.as_secs_f64(),
            dd.as_secs_f64(),
            sp.as_secs_f64(),
            sd.as_secs_f64()
        );
        report_stats("R1", model.dlsa_stats(), model.dlsa_memory_bytes());
        let stats = model.dlsa_stats().unwrap();
        assert_eq!(stats.sparse_forwards, 0);
        assert!(stats.dense_key_set_forwards > 0);
        assert_eq!(tokens, dense_tokens, "greedy tokens");
        for (step, (a, b)) in logits.iter().zip(&dense_logits).enumerate() {
            assert_bitwise(a, b, &format!("R1 logits of step {step}"));
        }
    }

    pub(in crate::qwen3_5) fn greedy_sampling(
        provider: &Qwen35Provider,
    ) -> mlxcel_core::generate::SamplingConfig {
        provider.baseline_sampling(
            false,
            SamplingOptions {
                temperature: Some(0.0),
                top_k: Some(1),
                top_p: Some(1.0),
                seed: Some(0),
                ..Default::default()
            },
        )
    }

    /// A chat prompt of about `target` tokens: repository text with the needle at 25% depth.
    pub(in crate::qwen3_5) fn needle_prompt(provider: &Qwen35Provider, target: usize) -> Vec<i32> {
        let tokenizer = provider.tokenizer();
        // This file and the docs name the code; the haystack must not
        let text = repo_text().replace("7291", "0000");
        assert!(
            !text.contains("7291"),
            "the haystack must not hold the needle"
        );
        let ids: Vec<u32> = encode(tokenizer, &text, target - 120)
            .into_iter()
            .map(|t| t as u32)
            .collect();
        let mut hay = tokenizer.decode(&ids, false).expect("decode");
        let mut at = hay.len() / 4;
        while !hay.is_char_boundary(at) || !hay[..at].ends_with('\n') {
            at += 1;
        }
        hay.insert_str(at, &format!("{NEEDLE}\n"));
        let content = format!(
            "Below is a long excerpt of a code repository.\n\n{hay}\n\nQuestion: according to the excerpt above, what is the vault code? Reply with just the code."
        );
        provider
            .tokenize_messages(
                &[ChatMessage {
                    role: "user".to_string(),
                    name: None,
                    content: Some(ChatMessageContent::Text(content)),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
                &[],
                None,
                false,
            )
            .expect("tokenize")
    }

    /// No token makes up more than half of a run of at least 8.
    fn degenerate(tokens: &[i32]) -> bool {
        if tokens.len() < 8 {
            return false;
        }
        let mut counts = std::collections::HashMap::new();
        for t in tokens {
            *counts.entry(t).or_insert(0usize) += 1;
        }
        counts.values().any(|&c| 2 * c > tokens.len())
    }

    /// R2: the needle at 16k and 32k, 48 greedy tokens through the provider, dense vs D-LSA.
    #[test]
    #[ignore = "requires the real checkpoint and the converted D-LSA map"]
    fn real_needle_16k_32k() {
        let (model_dir, config) = real_inputs();
        let mut provider = Qwen35Provider::load(&model_dir, KVCacheMode::Fp16).expect("load");
        let sampling = greedy_sampling(&provider);
        for target in [16_384usize, 32_768] {
            let prompt = needle_prompt(&provider, target);
            let mut texts = Vec::new();
            for dlsa in [false, true] {
                if dlsa {
                    provider.enable_dlsa(config.clone()).expect("enable D-LSA");
                } else {
                    provider.disable_dlsa();
                }
                let generation = provider
                    .generate_baseline_streaming(&prompt, 48, &sampling, None, None, &[], |_| true)
                    .expect("generate");
                let label = if dlsa { "D-LSA" } else { "dense" };
                eprintln!(
                    "R2 {} tokens {label}: prefill {:.2} s ({:.0} tok/s), decode {:.2} s for {} tokens ({:.2} tok/s): {:?}",
                    prompt.len(),
                    generation.prefill_time.as_secs_f64(),
                    prompt.len() as f64 / generation.prefill_time.as_secs_f64(),
                    generation.decode_time.as_secs_f64(),
                    generation.completion_tokens,
                    generation.completion_tokens as f64 / generation.decode_time.as_secs_f64(),
                    generation.text
                );
                if dlsa {
                    report_stats("R2", provider.dlsa_stats(), provider.dlsa_memory_bytes());
                    assert!(
                        !degenerate(&generation.token_ids),
                        "D-LSA output degenerates: {:?}",
                        generation.text
                    );
                }
                texts.push(generation.text);
            }
            if texts[0].contains("7291") {
                assert!(
                    texts[1].contains("7291"),
                    "dense found the needle at {target} but D-LSA did not: {:?}",
                    texts[1]
                );
            } else {
                eprintln!("R2 {target}: dense did not find the needle either");
            }
        }
        // An explicit MTP request runs while D-LSA is on (task 5 lifted the refusal)
        let request = GenerationRequest {
            prompt: "Hello".to_string(),
            max_tokens: 4,
            enable_thinking: false,
            reasoning_effort: None,
            temperature: Some(0.0),
            top_k: Some(1),
            top_p: Some(1.0),
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            frequency_penalty: None,
            seed: Some(0),
        };
        let (output, stats) = provider
            .generate_streaming_in_mode(&request, Qwen35GenerationMode::Mtp, |_| true)
            .expect("MTP under D-LSA");
        assert!(stats.is_some(), "the MTP decoder ran");
        assert!(!output.token_ids.is_empty());
    }

    /// Per-row log-softmax of the teacher-forced last `tail` tokens, in 128-row pieces
    /// (`[rows, V]` f32 on the device), after prefilling the rest in chunks.
    fn teacher_forced(
        model: &Qwen35Model,
        tokens: &[i32],
        tail: usize,
    ) -> (Vec<UniquePtr<MlxArray>>, Duration, Duration) {
        model.reset_runtime_state();
        let started = Instant::now();
        prefill(model, &tokens[..tokens.len() - tail]);
        let prefill_time = started.elapsed();
        let started = Instant::now();
        let mut pieces = Vec::new();
        for piece in tokens[tokens.len() - tail..].chunks(128) {
            let input = mlxcel_core::from_slice_i32(piece, &[1, piece.len() as i32]);
            let logits = model.forward(&input, &mut [], None);
            let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
            let shape = mlxcel_core::array_shape(&logits);
            let logits = mlxcel_core::reshape(&logits, &[shape[1], shape[2]]);
            let lp = mlxcel_core::log_softmax(&logits, -1);
            mlxcel_core::eval(&lp);
            pieces.push(lp);
        }
        (pieces, prefill_time, started.elapsed())
    }

    const DECODE: usize = 32;

    /// `steps` greedy tokens after the teacher-forced tail (the first from its last row), and the
    /// mean time of the `steps - 1` single-token decode forwards.
    fn decode_after(
        model: &Qwen35Model,
        pieces: &[UniquePtr<MlxArray>],
        steps: usize,
    ) -> (Vec<i32>, Duration) {
        let last = pieces.last().expect("a tail");
        let shape = mlxcel_core::array_shape(last);
        let row = mlxcel_core::slice(last, &[shape[0] - 1, 0], &[shape[0], shape[1]]);
        let mut token = argmax(&to_host_f32(&row));
        let mut tokens = vec![token];
        let started = Instant::now();
        for _ in 1..steps {
            let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            token = argmax(&host_logits(&model.forward_last_logits(
                &input,
                &mut [],
                None,
                0,
            )));
            tokens.push(token);
        }
        (tokens, started.elapsed() / (steps as u32 - 1))
    }

    struct Agreement {
        rows: usize,
        nll_dense: f64,
        nll_dlsa: f64,
        kl: f64,
        top1: usize,
    }

    /// NLL of the next tokens, mean KL(dense || dlsa) and top-1 agreement over the tail.
    fn agreement(
        dense: &[UniquePtr<MlxArray>],
        dlsa: &[UniquePtr<MlxArray>],
        targets: &[i32],
    ) -> Agreement {
        let mut a = Agreement {
            rows: 0,
            nll_dense: 0.0,
            nll_dlsa: 0.0,
            kl: 0.0,
            top1: 0,
        };
        let mut at = 0;
        for (d, s) in dense.iter().zip(dlsa) {
            let rows = mlxcel_core::array_shape(d)[0] as usize;
            // The tail's last row has no next token
            let scored = rows.min(targets.len() - at);
            let t = mlxcel_core::from_slice_i32(&targets[at..at + scored], &[scored as i32, 1]);
            let d =
                mlxcel_core::slice(d, &[0, 0], &[scored as i32, mlxcel_core::array_shape(d)[1]]);
            let s =
                mlxcel_core::slice(s, &[0, 0], &[scored as i32, mlxcel_core::array_shape(s)[1]]);
            let nd = to_host_f32(&mlxcel_core::take_along_axis(&d, &t, 1));
            let ns = to_host_f32(&mlxcel_core::take_along_axis(&s, &t, 1));
            let kl = mlxcel_core::sum_axis(
                &mlxcel_core::multiply(&mlxcel_core::exp(&d), &mlxcel_core::subtract(&d, &s)),
                -1,
                false,
            );
            let kl = to_host_f32(&kl);
            let ad = to_host_f32(&mlxcel_core::astype(
                &mlxcel_core::argmax(&d, -1, false),
                mlxcel_core::dtype::FLOAT32,
            ));
            let as_ = to_host_f32(&mlxcel_core::astype(
                &mlxcel_core::argmax(&s, -1, false),
                mlxcel_core::dtype::FLOAT32,
            ));
            for v in nd.iter().chain(&ns).chain(&kl) {
                assert!(v.is_finite(), "non-finite metric {v}");
            }
            a.rows += scored;
            a.nll_dense -= nd.iter().map(|&v| v as f64).sum::<f64>();
            a.nll_dlsa -= ns.iter().map(|&v| v as f64).sum::<f64>();
            a.kl += kl.iter().map(|&v| v as f64).sum::<f64>();
            a.top1 += ad.iter().zip(&as_).filter(|(x, y)| x == y).count();
            at += scored;
        }
        a
    }

    /// R3: teacher-forced logits of the last 512 tokens of long prompts, dense vs D-LSA.
    /// The thresholds (ppl ratio <= 1.10, top-1 >= 0.90) are provisional; task 6 decides.
    #[test]
    #[ignore = "requires the real checkpoint and the converted D-LSA map"]
    fn real_logits_vs_dense() {
        const TAIL: usize = 512;
        let (model_dir, config) = real_inputs();
        let tokenizer = load_tokenizer(&model_dir);
        let (repo, core) = (repo_text(), core_text());
        let prompts = [
            ("repo 16k", encode(&tokenizer, &repo, 16_384)),
            ("repo 32k", encode(&tokenizer, &repo, 32_768)),
            ("core 16k", encode(&tokenizer, &core, 16_384)),
        ];
        let mut model = Qwen35Model::load(&model_dir, KVCacheMode::Fp16).expect("load");
        let mut failures = Vec::new();
        for (name, tokens) in &prompts {
            model.disable_dlsa();
            let (dense, dp, dt) = teacher_forced(&model, tokens, TAIL);
            let (dense_tokens, dd) = decode_after(&model, &dense, DECODE);
            model.enable_dlsa(config.clone()).expect("enable D-LSA");
            let (dlsa, sp, st) = teacher_forced(&model, tokens, TAIL);
            let (dlsa_tokens, sd) = decode_after(&model, &dlsa, DECODE);
            let same = dense_tokens
                .iter()
                .zip(&dlsa_tokens)
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!(
                "R3 {name}: greedy decode dense {:.1} ms/token D-LSA {:.1} ms/token; first {same} of {DECODE} tokens agree",
                dd.as_secs_f64() * 1e3,
                sd.as_secs_f64() * 1e3
            );
            report_stats(
                &format!("R3 {name}"),
                model.dlsa_stats(),
                model.dlsa_memory_bytes(),
            );
            let targets = &tokens[tokens.len() - TAIL + 1..];
            let a = agreement(&dense, &dlsa, targets);
            let n = a.rows as f64;
            let (ppl_dense, ppl_dlsa) = ((a.nll_dense / n).exp(), (a.nll_dlsa / n).exp());
            let ratio = ppl_dlsa / ppl_dense;
            let top1 = a.top1 as f64 / n;
            eprintln!(
                "R3 {name} ({} rows): ppl dense {ppl_dense:.4} D-LSA {ppl_dlsa:.4} ratio {ratio:.4}; mean KL {:.5} nats; top-1 agreement {top1:.4}; prefill dense {:.1} s D-LSA {:.1} s; tail dense {:.1} s D-LSA {:.1} s",
                a.rows,
                a.kl / n,
                dp.as_secs_f64(),
                sp.as_secs_f64(),
                dt.as_secs_f64(),
                st.as_secs_f64()
            );
            if ratio > 1.10 || top1 < 0.90 {
                failures.push(format!("{name}: ppl ratio {ratio:.4}, top-1 {top1:.4}"));
            }
        }
        assert!(
            failures.is_empty(),
            "provisional thresholds missed: {failures:?}"
        );
    }
}
