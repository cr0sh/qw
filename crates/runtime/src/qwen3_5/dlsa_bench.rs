// SPDX-License-Identifier: Apache-2.0
//
//! Real-model D-LSA benchmark with a stage profile (`docs/dlsa.md` §14.3, §15): dense vs D-LSA
//! prefill and decode at long contexts, on the real checkpoint and the converted map, with
//! optional quality checks (teacher-forced tail, needles), a prefix-cache hit and the
//! speculative decoders.
//!
//! One context per run, under the GPU lock, after building outside it (as `docs/dlsa.md` §14.6):
//!
//! ```text
//! cargo test --release -p qw-runtime --features dlsa --lib --no-run   # prints the binary
//! MLXCEL_CACHE_LIMIT=<8589934592 | 4294967296 | 1073741824> \
//!   QW_DLSA_BENCH_CONTEXTS=<32768 | 131072 | 262144> \
//!   QW_DLSA_BENCH_MODES=dense,mlx,metal[,metal-select] QW_DLSA_BENCH_PROMPT=<token file> \
//!   QW_DLSA_BENCH_OUT=<unique>.json [QW_DLSA_BENCH_LOGITS=1] \
//!   ./gpu-lock -- target/release/deps/qw_runtime-<hash> \
//!   qwen3_5::dlsa_bench::real_dlsa_bench --exact --ignored --nocapture --test-threads=1
//! ```
//!
//! Environment:
//! - `QW_DLSA_BENCH_CONTEXTS` (default `32768`): contexts `T`; the prompt is
//!   `floor(T / 1536) * 1536` tokens (32,256 / 130,560 / 261,120), decode stays below `T`.
//! - `QW_DLSA_BENCH_MODES` (default `dense,mlx`): `dense` (D-LSA off), or a D-LSA backend:
//!   `mlx` (the task-3 path), `metal` (K1 + K2 kernels), `metal-select` (K1 + the task-3 read);
//!   a `-pca` suffix (`metal-pca`) loads the PCA step-0 map instead of the distilled one. The
//!   first mode is the reference the others are compared against (tail KL, tokens, answers).
//! - `QW_DLSA_BENCH_DECODE` (default 32): unfenced greedy decode steps.
//! - `QW_DLSA_BENCH_PROFILE_CHUNKS` (default 2) / `QW_DLSA_BENCH_PROFILE_DECODE` (default 8):
//!   the last prefill chunks, and extra decode steps after the unfenced ones, run fenced (every
//!   decoder layer and every D-LSA stage ends in an `eval`) for the stage breakdown. The
//!   throughput numbers come from the same pass: prefill tok/s includes the fenced chunks (they
//!   are reported separately), decode ms/token only counts unfenced steps.
//! - `QW_DLSA_BENCH_PROMPT`: a token file (i32 little-endian). Read if it exists, else written
//!   from the corpus, so later runs reuse the exact prompt whatever the worktree holds.
//! - `QW_DLSA_BENCH_OFFSET` (default 0): the prompt starts this many tokens into the file (one
//!   "document" per offset); the tail follows the prompt in the file.
//! - `QW_DLSA_BENCH_TAIL` (default 0): after the prefill, teacher-force the next `n` file tokens
//!   in 128-row forwards: NLL of each mode, and KL(reference ‖ mode) and top-1 agreement
//!   against the reference mode. Decode then starts after the tail.
//! - `QW_DLSA_BENCH_NEEDLES=1`: write five "secret code" needles into the prompt (at 10/30/50/
//!   70/90% depth, replacing tokens), snapshot the state after the prefill, and for each needle
//!   restore the snapshot, prefill a question and decode an answer greedily (exact match). The
//!   restore plus question prefill is a prefix-cache hit's time to first token.
//! - `QW_DLSA_BENCH_SPEC` (e.g. `mtp,dflash2`, default none) / `QW_DLSA_BENCH_SPEC_TOKENS`
//!   (default 64): after the rest, generate from a fresh state with the bundled MTP head
//!   (block 3) and the DFlash2 drafter, greedy with EOS suppressed: their prefill time, decode
//!   ms/token and acceptance.
//! - `QW_DLSA_BENCH_OUT`: the JSON report (must not exist); default under `$TMPDIR`.
//! - `QW_DLSA_BENCH_LOGITS=1`: also write the final-prefill and decode logits as raw f32 next
//!   to the report (the report always holds their sha256).
//! - `QW_DLSA_BENCH_KV=int8`: an Int8 KV cache instead of Fp16 (for memory pressure).
//! - `QW_DLSA_DENSE_BELOW` (default 12288, as served): rows below it read every causal key in
//!   the D-LSA modes; 0 reads sparse from the dense-key-set boundary on (tasks 3-5).
//! - `MLXCEL_CACHE_LIMIT` (bytes, default 8 GiB as the provider sets it): the MLX free-buffer
//!   cache. Without a limit the cache grows past the machine's memory at 256k and swaps.
//!   The report records the limit, the swap in use and `pmset -g therm` before and after each
//!   mode.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use super::{Qwen35AttentionVariant, Qwen35Model, prepare_attention_forward};
use crate::dlsa::profile::{self, DlsaProfile, DlsaStageTime};
use crate::dlsa::{
    DlsaBackend, DlsaConfig, DlsaMapKind, DlsaStats, resolve_dlsa_map_path, sha256_hex, to_host_f32,
};
use crate::qwen_vl_position::decode_rope_positions;
use crate::qwen3_5_dflash::Qwen35Dflash2Generator;
use crate::qwen3_5_mtp::Qwen35MtpGenerator;
use crate::qwen3_next::{DlsaLayerStep, Qwen3NextCache};

const PREFILL_CHUNK: usize = 1536;
const FULL_ATTENTION_LAYERS: usize = 16;
/// Rows per teacher-forced forward (as R3, `dlsa_tests::real::teacher_forced`).
const TAIL_PIECE: usize = 128;
/// `(vault, code)`: codes that do not occur in the corpus.
const NEEDLES: [(&str, &str); 5] = [
    ("amber", "4821-KESTREL"),
    ("cobalt", "7390-MARLIN"),
    ("crimson", "1567-OSPREY"),
    ("jade", "9043-HERON"),
    ("violet", "2678-FALCON"),
];
const NEEDLE_DEPTHS: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
const NEEDLE_ANSWER_TOKENS: usize = 12;
const MTP_BLOCK: usize = 3;

fn needle_text(vault: &str, code: &str) -> String {
    format!("\nThe secret code of the {vault} vault is {code}.\n")
}

fn question_text(vault: &str) -> String {
    format!(
        "\n\nQuestion: What is the secret code of the {vault} vault?\nAnswer: The secret code of the {vault} vault is"
    )
}

/// Bench knobs beyond the context and the mode (module docs).
struct BenchOptions {
    decode_steps: usize,
    profile_chunks: usize,
    profile_decode: usize,
    needles: bool,
    spec: Vec<String>,
    spec_tokens: usize,
}

/// What one context's later modes are compared against: the first mode's results.
struct Reference {
    mode: String,
    tail_logprobs: Vec<f32>,
    tokens: Vec<i32>,
    answers: Vec<String>,
}

/// The inputs of one context: the prompt (needles written in), the tail after it, and the
/// encoded needles and questions.
struct ContextInputs<'a> {
    context: usize,
    prompt: &'a [i32],
    tail: &'a [i32],
    needles: &'a [NeedleInput],
}

struct NeedleInput {
    vault: &'static str,
    code: &'static str,
    position: usize,
    question: Vec<i32>,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.trim()
                .parse()
                .unwrap_or_else(|e| panic!("{name}={v}: {e}"))
        })
        .unwrap_or(default)
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The forward of [`LanguageModel::forward_last_logits`], with every decoder layer's output
/// evaluated and timed (and the D-LSA stage fences on, set by the caller), recorded in the
/// process-wide [`DlsaProfile`]. Same ops in the same order; only the evals differ.
fn fenced_forward_last_logits(
    model: &Qwen35Model,
    input_ids: &MlxArray,
    last_pos: usize,
) -> UniquePtr<MlxArray> {
    let forward_started = Instant::now();
    let sequence_length = mlxcel_core::array_shape(input_ids)[1];
    let rope_delta = model.mrope_state.rope_delta();
    let (logits, offset) = model.sequence_state.with_internal(|caches| {
        let cache_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
        let position_ids =
            rope_delta.map(|delta| decode_rope_positions(cache_offset, sequence_length, delta));
        let position_ids = position_ids.as_deref();
        prepare_attention_forward(
            caches,
            model.kv_cache_mode,
            sequence_length,
            model.initial_prefill_complete.load(Ordering::Relaxed),
        );
        let mut hidden = model.embed_tokens.forward(input_ids);
        mlxcel_core::eval(&hidden);
        let dlsa = model.dlsa.as_ref();
        let mut state = dlsa.map(|d| d.state.borrow_mut());
        for (layer_idx, (layer, cache)) in model.layers.iter().zip(caches.iter_mut()).enumerate() {
            let started = Instant::now();
            let full_attention =
                matches!(layer.attention, Qwen35AttentionVariant::FullAttention(_));
            let ordinal = dlsa.and_then(|d| d.layer_ordinal.get(&layer_idx).copied());
            hidden = match (&layer.attention, cache, ordinal, state.as_mut(), dlsa) {
                (
                    Qwen35AttentionVariant::FullAttention(attention),
                    Qwen3NextCache::Attention(kv),
                    Some(ordinal),
                    Some(state),
                    Some(dlsa),
                ) => {
                    let (index_caches, stats) = &mut **state;
                    let step = DlsaLayerStep {
                        maps: dlsa
                            .maps
                            .layer(layer_idx)
                            .expect("D-LSA maps were checked against the layers"),
                        geometry: &dlsa.geometry,
                        config: &dlsa.config,
                        cache: &mut index_caches[ordinal],
                        stats,
                        commit: true,
                    };
                    // `Qwen35DecoderLayer::forward_dlsa`
                    let normed = layer.input_layernorm.forward(&hidden);
                    let residual = attention.forward_dlsa(&normed, kv, position_ids, step);
                    let h = mlxcel_core::add(&hidden, &residual);
                    let mlp = layer
                        .mlp
                        .forward(&layer.post_attention_layernorm.forward(&h));
                    mlxcel_core::add(&h, &mlp)
                }
                (_, cache, _, _, _) => layer.forward(&hidden, None, cache, position_ids),
            };
            mlxcel_core::eval(&hidden);
            profile::record_layer(full_attention, started.elapsed());
        }
        let shape = mlxcel_core::array_shape(&hidden);
        let position = i32::try_from(last_pos).unwrap_or(i32::MAX);
        let hidden = mlxcel_core::slice(
            &hidden,
            &[0, position, 0],
            &[shape[0], position + 1, shape[2]],
        );
        let logits = model.project_logits(&model.norm.forward(&hidden));
        mlxcel_core::eval(&logits);
        let offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
        (logits, offset)
    });
    model.mrope_state.set_position(offset);
    profile::record_forward(forward_started.elapsed());
    logits
}

/// One forward of `tokens` at the current position, fenced or not; returns the last row's
/// logits on the host and the wall time (forward, eval and readback).
fn step(model: &Qwen35Model, tokens: &[i32], fenced: bool) -> (Vec<f32>, Duration) {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let started = Instant::now();
    let logits = if fenced {
        profile::set_override(Some(true));
        let l = fenced_forward_last_logits(model, &input, tokens.len() - 1);
        profile::set_override(Some(false));
        l
    } else {
        let l = model.forward_last_logits(&input, &mut [], None, tokens.len() - 1);
        mlxcel_core::eval(&l);
        l
    };
    let host = to_host_f32(&logits);
    (host, started.elapsed())
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "non-finite logit {v} at {i}");
        if v > logits[best] {
            best = i;
        }
    }
    best as i32
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// A profile as JSON: per stage total ms, occurrences, ms per fenced forward and per layer,
/// and % of the fenced forwards' time.
fn profile_json(p: &DlsaProfile) -> Value {
    let forwards = p.forwards.count.max(1) as f64;
    let forward_ms = ms(p.forwards.time) / forwards;
    let entry = |t: DlsaStageTime, per_layer: f64| {
        let per_forward = ms(t.time) / forwards;
        json!({
            "total_ms": ms(t.time),
            "count": t.count,
            "ms_per_forward": per_forward,
            "ms_per_layer": per_forward / per_layer,
            "pct_of_forward": 100.0 * per_forward / forward_ms,
        })
    };
    let linear_layers = p.linear_layers.count as f64 / forwards;
    let full_layers = p.full_attention_layers.count as f64 / forwards;
    let stages: serde_json::Map<String, Value> = p
        .entries()
        .into_iter()
        .map(|(name, t)| (name.to_string(), entry(t, FULL_ATTENTION_LAYERS as f64)))
        .collect();
    let layers = ms(p.linear_layers.time + p.full_attention_layers.time) / forwards;
    json!({
        "fenced_forwards": p.forwards.count,
        "ms_per_forward": forward_ms,
        "linear_layers": entry(p.linear_layers, linear_layers.max(1.0)),
        "full_attention_layers": entry(p.full_attention_layers, full_layers.max(1.0)),
        "full_attention_share_of_layers_pct": 100.0 * ms(p.full_attention_layers.time) / forwards / layers,
        "embedding_and_head": {
            "ms_per_forward": forward_ms - layers,
        },
        "dlsa_layer_forwards": p.layer_forwards.count,
        "dlsa_stage_sum": entry(p.layer_forwards, FULL_ATTENTION_LAYERS as f64),
        "stages": stages,
    })
}

fn stats_json(s: Option<DlsaStats>, memory: usize) -> Value {
    let Some(s) = s else { return Value::Null };
    json!({
        "dense_key_set_forwards": s.dense_key_set_forwards,
        "sparse_forwards": s.sparse_forwards,
        "sparse_rows": s.sparse_rows,
        "keys_read": s.keys_read,
        "keys_per_row": if s.sparse_rows > 0 { s.keys_read as f64 / s.sparse_rows as f64 } else { 0.0 },
        "index_time_s": s.index_time.as_secs_f64(),
        "select_time_s": s.select_time.as_secs_f64(),
        "index_memory_bytes": memory,
    })
}

/// Deterministic long text (as `dlsa_tests::real::corpus`): files of `dirs` sorted and walked
/// recursively, keeping `extensions`, skipping D-LSA's own files.
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
    text.replace("<|", "< |")
}

/// `n` prompt tokens: from `QW_DLSA_BENCH_PROMPT` if it exists (written there otherwise), else
/// from the corpus, cycled if it is short (said in the report).
fn prompt_tokens(model_dir: &Path, n: usize) -> (Vec<i32>, Value) {
    let file = std::env::var_os("QW_DLSA_BENCH_PROMPT").map(PathBuf::from);
    if let Some(path) = file.as_ref().filter(|p| p.exists()) {
        let bytes = std::fs::read(path).expect("read the prompt file");
        let all: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        assert!(
            all.len() >= n,
            "{} holds {} tokens, {n} needed",
            path.display(),
            all.len()
        );
        let meta =
            json!({"source": path, "file_tokens": all.len(), "file_sha256": sha256_hex(&bytes)});
        return (all[..n].to_vec(), meta);
    }
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer");
    let text = corpus(&["docs", "crates"], &["md", "rs"]);
    let ids: Vec<i32> = tokenizer
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|&t| t as i32)
        .collect();
    let corpus_tokens = ids.len();
    let cycled = corpus_tokens < n;
    if cycled {
        eprintln!("the corpus has {corpus_tokens} tokens, fewer than {n}: cycling it");
    }
    let all: Vec<i32> = ids.iter().copied().cycle().take(n.max(262_144)).collect();
    let bytes: Vec<u8> = all.iter().flat_map(|t| t.to_le_bytes()).collect();
    if let Some(path) = &file {
        std::fs::write(path, &bytes).expect("write the prompt file");
        eprintln!("wrote {} prompt tokens to {}", all.len(), path.display());
    }
    let meta = json!({
        "source": "corpus docs+crates md/rs (dlsa files skipped)",
        "corpus_tokens": corpus_tokens,
        "cycled": cycled,
        "written_to": file,
        "file_tokens": all.len(),
        "file_sha256": sha256_hex(&bytes),
    });
    (all[..n].to_vec(), meta)
}

fn command_output(cmd: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(cmd)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn thermal() -> Option<String> {
    command_output("pmset", &["-g", "therm"]).map(|s| s.replace('\n', " / "))
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<i32> {
    tokenizer
        .encode(text, false)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|&t| t as i32)
        .collect()
}

fn decode_text(tokenizer: &tokenizers::Tokenizer, tokens: &[i32]) -> String {
    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    tokenizer.decode(&ids, false).expect("decode")
}

/// Writes the [`NEEDLES`] into `prompt` at [`NEEDLE_DEPTHS`] (replacing tokens, so the length
/// stays) and encodes their questions.
fn write_needles(tokenizer: &tokenizers::Tokenizer, prompt: &mut [i32]) -> Vec<NeedleInput> {
    NEEDLES
        .iter()
        .zip(NEEDLE_DEPTHS)
        .map(|(&(vault, code), depth)| {
            let ids = encode(tokenizer, &needle_text(vault, code));
            let position = (prompt.len() as f64 * depth) as usize;
            prompt[position..position + ids.len()].copy_from_slice(&ids);
            NeedleInput {
                vault,
                code,
                position,
                question: encode(tokenizer, &question_text(vault)),
            }
        })
        .collect()
}

fn argmax_row(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// Host log-softmax rows (`[rows, vocab]` row-major) of `tail` teacher-forced at the current
/// position in [`TAIL_PIECE`]-row forwards, and the wall time.
fn teacher_force(model: &Qwen35Model, tail: &[i32]) -> (Vec<f32>, Duration) {
    let started = Instant::now();
    let mut out = Vec::new();
    for piece in tail.chunks(TAIL_PIECE) {
        let input = mlxcel_core::from_slice_i32(piece, &[1, piece.len() as i32]);
        let logits = model.forward(&input, &mut [], None);
        let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        let shape = mlxcel_core::array_shape(&logits);
        let logits = mlxcel_core::reshape(&logits, &[shape[1], shape[2]]);
        let lp = mlxcel_core::log_softmax(&logits, -1);
        mlxcel_core::eval(&lp);
        out.extend(to_host_f32(&lp));
    }
    (out, started.elapsed())
}

/// Row `i` of the tail predicts `tail[i + 1]`: NLL, perplexity and next-token top-1 of this
/// mode, and against `reference` (the same rows of the reference mode): KL(reference ‖ this)
/// per row (mean, p90, max), top-1 agreement and the perplexity ratio.
fn tail_metrics(logprobs: &[f32], vocab: usize, tail: &[i32], reference: Option<&[f32]>) -> Value {
    let scored = tail.len() - 1;
    let row = |lp: &'_ [f32], r: usize| -> Vec<f32> { lp[r * vocab..(r + 1) * vocab].to_vec() };
    let nll_of = |lp: &[f32]| -> f64 {
        (0..scored)
            .map(|r| -f64::from(lp[r * vocab + tail[r + 1] as usize]))
            .sum::<f64>()
            / scored as f64
    };
    let nll = nll_of(logprobs);
    let next_top1 = (0..scored)
        .filter(|&r| argmax_row(&row(logprobs, r)) == tail[r + 1] as usize)
        .count();
    let mut out = json!({
        "scored_rows": scored,
        "nll": nll,
        "ppl": nll.exp(),
        "top1_is_next_token": next_top1 as f64 / scored as f64,
    });
    if let Some(reference) = reference {
        let mut kls = Vec::with_capacity(scored);
        let mut agree = 0;
        for r in 0..scored {
            let (a, b) = (row(reference, r), row(logprobs, r));
            let kl: f64 = a
                .iter()
                .zip(&b)
                .filter(|(a, _)| a.is_finite())
                .map(|(&a, &b)| f64::from(a).exp() * (f64::from(a) - f64::from(b)))
                .sum();
            assert!(kl.is_finite(), "non-finite KL at tail row {r}");
            kls.push(kl);
            if argmax_row(&a) == argmax_row(&b) {
                agree += 1;
            }
        }
        let reference_nll = nll_of(reference);
        out["vs_reference"] = json!({
            "mean_kl": kls.iter().sum::<f64>() / scored as f64,
            "p90_kl": quantile(&kls, 0.9),
            "max_kl": kls.iter().copied().fold(0.0, f64::max),
            "top1_agreement": agree as f64 / scored as f64,
            "reference_ppl": reference_nll.exp(),
            "ppl_ratio": (nll - reference_nll).exp(),
            "row_kl": kls,
        });
    }
    out
}

/// One mode at one context: a fresh sequence, chunked prefill, then (as enabled) a snapshot,
/// the teacher-forced tail, greedy decode, the needle questions and the speculative decoders.
#[allow(clippy::too_many_arguments)]
fn run_mode(
    model: &mut Qwen35Model,
    mode: &str,
    config: &DlsaConfig,
    inputs: &ContextInputs<'_>,
    opts: &BenchOptions,
    tokenizer: &tokenizers::Tokenizer,
    dflash: Option<&mut Qwen35Dflash2Generator>,
    reference: &mut Option<Reference>,
    logits_out: Option<&Path>,
) -> Value {
    let (prompt, context) = (inputs.prompt, inputs.context);
    let (backend, pca) = match mode.strip_suffix("-pca") {
        Some(backend) => (backend, true),
        None => (mode, false),
    };
    let dlsa_config = (mode != "dense").then(|| {
        let map_path = if pca {
            resolve_dlsa_map_path(DlsaMapKind::PcaStep0, None)
                .expect("the PCA step-0 map (run `qw download --dlsa-map --dlsa-pca`)")
        } else {
            config.map_path.clone()
        };
        DlsaConfig {
            backend: DlsaBackend::parse(backend)
                .unwrap_or_else(|e| panic!("mode {mode}: dense, or a backend: {e}")),
            map_path,
            ..config.clone()
        }
    });
    match &dlsa_config {
        None => model.disable_dlsa(),
        Some(c) => model.enable_dlsa(c.clone()).expect("enable D-LSA"),
    }
    let map_sha = dlsa_config
        .as_ref()
        .map(|c| sha256_hex(&std::fs::read(&c.map_path).expect("read the map")));
    model.reset_runtime_state();
    mlxcel_core::memory::clear_cache();
    let swap_before = command_output("sysctl", &["-n", "vm.swapusage"]);
    let thermal_before = thermal();
    mlxcel_core::memory::reset_peak_memory();
    profile::set_override(Some(false));
    profile::reset();

    let chunks: Vec<&[i32]> = prompt.chunks(PREFILL_CHUNK).collect();
    let fenced_from = chunks.len().saturating_sub(opts.profile_chunks);
    let mut chunk_ms = Vec::with_capacity(chunks.len());
    let mut last = Vec::new();
    let started = Instant::now();
    for (i, chunk) in chunks.iter().enumerate() {
        let (logits, t) = step(model, chunk, i >= fenced_from);
        chunk_ms.push(ms(t));
        if (i + 1) % 10 == 0 || i + 1 == chunks.len() {
            eprintln!(
                "{mode} T={context}: chunk {}/{} at {} took {:.0} ms",
                i + 1,
                chunks.len(),
                i * PREFILL_CHUNK,
                ms(t)
            );
        }
        last = logits;
    }
    let prefill = started.elapsed();
    model.after_prefill();
    let prefill_profile = profile::snapshot();
    let prefill_peak = mlxcel_core::memory::peak_memory();
    let index_bytes = model.dlsa_memory_bytes();
    profile::reset();
    let mut logits_dump = f32_bytes(&last);
    let final_prefill_sha = sha256_hex(&f32_bytes(&last));

    // The state at the prompt's end, for the needle questions (a prefix-cache hit)
    let snapshot = (!inputs.needles.is_empty()).then(|| {
        let started = Instant::now();
        let s = model
            .snapshot_sequence_state(SequenceId::from_raw(0), prompt.len(), None)
            .expect("a snapshot at the prompt's end");
        (s, started.elapsed())
    });

    let mut tail_report = Value::Null;
    if !inputs.tail.is_empty() {
        let (logprobs, t) = teacher_force(model, inputs.tail);
        let vocab = logprobs.len() / inputs.tail.len();
        let mut metrics = tail_metrics(
            &logprobs,
            vocab,
            inputs.tail,
            reference.as_ref().map(|r| r.tail_logprobs.as_slice()),
        );
        metrics["wall_s"] = json!(t.as_secs_f64());
        metrics["rows"] = json!(inputs.tail.len());
        eprintln!("{mode} T={context}: tail {metrics}");
        last = logprobs[logprobs.len() - vocab..].to_vec();
        if reference.is_none() {
            *reference = Some(Reference {
                mode: mode.to_string(),
                tail_logprobs: logprobs,
                tokens: Vec::new(),
                answers: Vec::new(),
            });
        }
        tail_report = metrics;
    }

    let mut logits_sha = Vec::new();
    let mut tokens = vec![argmax(&last)];
    let mut decode_ms = Vec::with_capacity(opts.decode_steps);
    let mut fenced_decode_ms = Vec::with_capacity(opts.profile_decode);
    for i in 0..opts.decode_steps + opts.profile_decode {
        let fenced = i >= opts.decode_steps;
        let (logits, t) = step(model, &[*tokens.last().unwrap()], fenced);
        if fenced {
            fenced_decode_ms.push(ms(t));
        } else {
            decode_ms.push(ms(t));
            logits_sha.push(sha256_hex(&f32_bytes(&logits)));
            if logits_out.is_some() {
                logits_dump.extend(f32_bytes(&logits));
            }
        }
        tokens.push(argmax(&logits));
    }
    let decode_profile = profile::snapshot();
    profile::set_override(None);
    if let Some(path) = logits_out {
        let path = path.with_extension(format!("{mode}.{context}.logits.f32"));
        std::fs::write(&path, &logits_dump).expect("write logits");
        eprintln!("wrote {}", path.display());
    }

    let mut needles = Vec::new();
    let mut answers = Vec::new();
    let mut snapshot_report = Value::Null;
    if let Some((snapshot, capture)) = &snapshot {
        for needle in inputs.needles {
            model.reset_runtime_state();
            let started = Instant::now();
            model
                .restore_sequence_state(SequenceId::from_raw(0), snapshot)
                .expect("restore the prompt snapshot");
            let restored = started.elapsed();
            let (logits, _) = step(model, &needle.question, false);
            let ttft = started.elapsed();
            let mut answer = vec![argmax(&logits)];
            while answer.len() < NEEDLE_ANSWER_TOKENS {
                let (logits, _) = step(model, &[*answer.last().unwrap()], false);
                answer.push(argmax(&logits));
            }
            let text = decode_text(tokenizer, &answer);
            let found = text.trim_start().starts_with(needle.code);
            eprintln!(
                "{mode} T={context}: needle {} at {}: {text:?} ({}), hit TTFT {:.0} ms",
                needle.vault,
                needle.position,
                if found { "found" } else { "missed" },
                ms(ttft)
            );
            let same = reference
                .as_ref()
                .filter(|r| r.mode != mode)
                .and_then(|r| r.answers.get(needles.len()))
                .map(|a| *a == text);
            needles.push(json!({
                "vault": needle.vault,
                "code": needle.code,
                "position": needle.position,
                "depth": needle.position as f64 / prompt.len() as f64,
                "answer": text,
                "exact_match": found,
                "same_answer_as_reference": same,
                "restore_ms": ms(restored),
                "question_tokens": needle.question.len(),
                "hit_ttft_ms": ms(ttft),
            }));
            answers.push(text);
        }
        snapshot_report = json!({
            "capture_ms": ms(*capture),
            "bytes": snapshot.nbytes(),
            "bytes_per_token": snapshot.nbytes() as f64 / prompt.len() as f64,
        });
    }
    drop(snapshot);

    let mut spec = Vec::new();
    let mut dflash = dflash;
    for decoder in &opts.spec {
        model.reset_runtime_state();
        mlxcel_core::memory::clear_cache();
        mlxcel_core::memory::reset_peak_memory();
        // Throughput runs to the token budget: EOS is suppressed (prompts end mid-file)
        let mut sampling = SamplingConfig::greedy();
        sampling.token_bias.suppress_tokens(&model.eos_token_ids());
        let (generated, accepted, proposed, prefill_time, decode_time, stop) =
            match decoder.as_str() {
                "mtp" => {
                    let g = Qwen35MtpGenerator::new()
                        .generate_streaming(
                            model,
                            prompt,
                            opts.spec_tokens,
                            &sampling,
                            MTP_BLOCK,
                            None,
                            &[],
                            None,
                            |_| true,
                            false,
                        )
                        .expect("MTP generation");
                    (
                        g.token_ids,
                        g.stats.accepted_draft_tokens,
                        g.stats.proposed_draft_tokens,
                        g.stats.prefill_time,
                        g.stats.decode_time,
                        g.stop_reason,
                    )
                }
                "dflash2" => {
                    let g = dflash
                        .as_deref_mut()
                        .expect("the DFlash2 drafter is loaded for QW_DLSA_BENCH_SPEC=dflash2")
                        .generate_streaming_with_prefill(
                            model,
                            prompt,
                            opts.spec_tokens,
                            &sampling,
                            None,
                            &[],
                            false,
                            None,
                            None,
                            |_, _| true,
                        )
                        .expect("DFlash2 generation");
                    (
                        g.token_ids,
                        g.stats.accepted_draft_tokens,
                        g.stats.proposed_draft_tokens,
                        g.stats.prefill_time,
                        g.stats.decode_time,
                        g.stop_reason,
                    )
                }
                other => panic!("QW_DLSA_BENCH_SPEC: {other} is not mtp or dflash2"),
            };
        let per_token = ms(decode_time) / generated.len().saturating_sub(1).max(1) as f64;
        let acceptance = accepted as f64 / proposed.max(1) as f64;
        eprintln!(
            "{mode} T={context}: {decoder} prefill {:.1} s, {} tokens, decode {per_token:.1} ms/token, acceptance {acceptance:.3}",
            prefill_time.as_secs_f64(),
            generated.len()
        );
        spec.push(json!({
            "decoder": decoder,
            "stop_reason": format!("{stop:?}"),
            "tokens": generated,
            "generated": generated.len(),
            "accepted": accepted,
            "proposed": proposed,
            "acceptance": acceptance,
            "prefill_s": prefill_time.as_secs_f64(),
            "prefill_tok_s": prompt.len() as f64 / prefill_time.as_secs_f64(),
            "decode_s": decode_time.as_secs_f64(),
            "decode_ms_per_token": per_token,
            "peak_memory_bytes": mlxcel_core::memory::peak_memory(),
        }));
    }
    let peak = mlxcel_core::memory::peak_memory();

    let same_tokens = reference
        .as_ref()
        .filter(|r| r.mode != mode && !r.tokens.is_empty())
        .map(|r| {
            r.tokens
                .iter()
                .zip(&tokens)
                .take_while(|(a, b)| a == b)
                .count()
        });
    if let Some(r) = reference.as_mut().filter(|r| r.mode == mode) {
        r.tokens = tokens.clone();
        r.answers = answers;
    } else if reference.is_none() {
        *reference = Some(Reference {
            mode: mode.to_string(),
            tail_logprobs: Vec::new(),
            tokens: tokens.clone(),
            answers,
        });
    }

    let n = prompt.len() as f64;
    let unfenced_ms: f64 = chunk_ms[..fenced_from].iter().sum();
    let decode_median = quantile(&decode_ms, 0.5);
    let report = json!({
        "mode": mode,
        "dlsa_config": dlsa_config.as_ref().map(|c| format!("{c:?}")),
        "map_sha256": map_sha,
        "context": context,
        "prompt_tokens": prompt.len(),
        "prefill": {
            "chunks": chunks.len(),
            "chunk": PREFILL_CHUNK,
            "wall_s": prefill.as_secs_f64(),
            "ttft_s": prefill.as_secs_f64(),
            "tok_s": n / prefill.as_secs_f64(),
            "fenced_chunks_from": fenced_from,
            "unfenced_chunks_tok_s": (fenced_from * PREFILL_CHUNK) as f64 / (unfenced_ms / 1e3),
            "chunk_ms": chunk_ms,
            "peak_memory_bytes": prefill_peak,
            "index_memory_bytes": index_bytes,
            "index_bytes_per_token": index_bytes as f64 / n,
        },
        "tail": tail_report,
        "decode": {
            "steps": opts.decode_steps,
            "median_ms": decode_median,
            "p90_ms": quantile(&decode_ms, 0.9),
            "mean_ms": decode_ms.iter().sum::<f64>() / decode_ms.len().max(1) as f64,
            "tok_s": 1e3 / decode_median,
            "step_ms": decode_ms,
            "fenced_step_ms": fenced_decode_ms,
            "tokens_equal_reference_prefix": same_tokens,
        },
        "tokens": tokens,
        "logits_sha256": {
            "final_prefill_row": final_prefill_sha,
            "decode_steps": logits_sha,
        },
        "snapshot": snapshot_report,
        "needles": needles,
        "spec": spec,
        "dlsa_stats": stats_json(model.dlsa_stats(), model.dlsa_memory_bytes()),
        "peak_memory_bytes": peak,
        "active_memory_bytes": mlxcel_core::memory::active_memory(),
        "cache_memory_bytes": mlxcel_core::memory::cache_memory(),
        "swap_before": swap_before,
        "swap_after": command_output("sysctl", &["-n", "vm.swapusage"]),
        "thermal_before": thermal_before,
        "thermal_after": thermal(),
        "profile": {
            "prefill": profile_json(&prefill_profile),
            "decode": profile_json(&decode_profile),
        },
    });
    eprintln!(
        "{mode} T={context}: prefill {:.1} s ({:.0} tok/s), decode median {decode_median:.1} ms (p90 {:.1}), prefill peak {:.1} GiB",
        prefill.as_secs_f64(),
        n / prefill.as_secs_f64(),
        quantile(&decode_ms, 0.9),
        prefill_peak as f64 / (1u64 << 30) as f64
    );
    // Free this mode's KV, index caches and cached buffers before the next mode starts
    model.reset_runtime_state();
    mlxcel_core::memory::clear_cache();
    report
}

/// The prompt length for context `T`: whole chunks, leaving room for the tail and decode.
fn prompt_len(context: usize) -> usize {
    (context - 1) / PREFILL_CHUNK * PREFILL_CHUNK
}

/// Dense vs D-LSA prefill and decode with a stage profile (module docs for the environment).
#[test]
#[ignore = "requires the real checkpoint, the converted D-LSA map and long GPU time"]
fn real_dlsa_bench() {
    let contexts: Vec<usize> = env_list("QW_DLSA_BENCH_CONTEXTS", "32768")
        .iter()
        .map(|s| s.parse().expect("QW_DLSA_BENCH_CONTEXTS"))
        .collect();
    let modes = env_list("QW_DLSA_BENCH_MODES", "dense,mlx");
    let opts = BenchOptions {
        decode_steps: env_usize("QW_DLSA_BENCH_DECODE", 32),
        profile_chunks: env_usize("QW_DLSA_BENCH_PROFILE_CHUNKS", 2),
        profile_decode: env_usize("QW_DLSA_BENCH_PROFILE_DECODE", 8),
        needles: std::env::var("QW_DLSA_BENCH_NEEDLES").is_ok_and(|v| v == "1"),
        spec: env_list("QW_DLSA_BENCH_SPEC", ""),
        spec_tokens: env_usize("QW_DLSA_BENCH_SPEC_TOKENS", 64),
    };
    let offset = env_usize("QW_DLSA_BENCH_OFFSET", 0);
    let tail_len = env_usize("QW_DLSA_BENCH_TAIL", 0);
    assert!(tail_len != 1, "QW_DLSA_BENCH_TAIL: 0, or at least 2 rows");
    let kv_mode = match std::env::var("QW_DLSA_BENCH_KV").as_deref() {
        Ok("int8") => KVCacheMode::Int8,
        _ => KVCacheMode::Fp16,
    };
    let model_dir = crate::resolve_model_path(None)
        .expect("QW_MODEL_PATH or the default model cache must hold the Qwen3.8-27B checkpoint");
    let map = resolve_dlsa_map_path(DlsaMapKind::Distilled, None)
        .expect("the converted D-LSA map (run `qw download --dlsa-map`, or set QW_DLSA_MAP_PATH)");
    let config = DlsaConfig {
        dense_below: env_usize(
            crate::dlsa::DLSA_DENSE_BELOW_ENV,
            crate::dlsa::DLSA_DENSE_BELOW_DEFAULT,
        ),
        ..DlsaConfig::new(map.clone())
    };
    let sha = command_output("git", &["rev-parse", "HEAD"]).unwrap_or_default();
    let dirty = command_output("git", &["status", "--porcelain", "--untracked-files=no"])
        .is_some_and(|s| !s.is_empty());
    let out = std::env::var_os("QW_DLSA_BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            std::env::temp_dir().join(format!(
                "dlsa_real_bench_{}_{ts}.json",
                &sha[..sha.len().min(10)]
            ))
        });
    assert!(
        !out.exists(),
        "{} exists; result files need unique names",
        out.display()
    );
    let logits_out = std::env::var("QW_DLSA_BENCH_LOGITS")
        .is_ok_and(|v| v == "1")
        .then_some(out.as_path());

    let max_prompt = contexts
        .iter()
        .map(|&t| prompt_len(t))
        .max()
        .expect("a context");
    let (all_tokens, prompt_meta) = prompt_tokens(&model_dir, offset + max_prompt + tail_len);
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer");
    let loaded = Instant::now();
    // As `provider::initialize_runtime` does for serving
    let cache_limit = std::env::var("MLXCEL_CACHE_LIMIT")
        .ok()
        .map(|v| {
            v.parse::<u64>()
                .expect("MLXCEL_CACHE_LIMIT is a byte count")
        })
        .unwrap_or(8 << 30);
    mlxcel_core::memory::set_cache_limit(cache_limit);
    let mut model = Qwen35Model::load(&model_dir, kv_mode).expect("load");
    let mut dflash = opts.spec.iter().any(|d| d == "dflash2").then(|| {
        let draft_dir = crate::resolve_dflash2_draft_path(None).expect("the DFlash2 drafter");
        Qwen35Dflash2Generator::new(&model, &draft_dir).expect("load the drafter")
    });
    if opts.spec.iter().any(|d| d == "mtp") {
        assert!(model.has_mtp(), "the checkpoint has no MTP head");
    }
    eprintln!(
        "loaded {} in {:.1} s",
        model_dir.display(),
        loaded.elapsed().as_secs_f64()
    );
    let config_sha = std::fs::read(model_dir.join("config.json"))
        .map(|b| sha256_hex(&b))
        .ok();
    let map_sha = sha256_hex(&std::fs::read(&map).expect("read the map"));

    let mut results = Vec::new();
    for &context in &contexts {
        let prompt_len = prompt_len(context);
        assert!(
            prompt_len + tail_len + opts.decode_steps + opts.profile_decode < context,
            "T={context}: prompt {prompt_len} + tail + decode does not fit"
        );
        let mut prompt = all_tokens[offset..offset + prompt_len].to_vec();
        let tail = &all_tokens[offset + prompt_len..offset + prompt_len + tail_len];
        let needles = if opts.needles {
            write_needles(&tokenizer, &mut prompt)
        } else {
            Vec::new()
        };
        let inputs = ContextInputs {
            context,
            prompt: &prompt,
            tail,
            needles: &needles,
        };
        let mut reference = None;
        let mut by_mode = Vec::new();
        for mode in &modes {
            by_mode.push(run_mode(
                &mut model,
                mode,
                &config,
                &inputs,
                &opts,
                &tokenizer,
                dflash.as_mut(),
                &mut reference,
                logits_out,
            ));
        }
        results.push(json!({
            "context": context,
            "prompt_sha256": sha256_hex(&prompt.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()),
            "reference_mode": reference.as_ref().map(|r| r.mode.clone()),
            "modes": by_mode,
        }));
    }
    model.disable_dlsa();

    let report = json!({
        "bench": "real_dlsa_bench",
        "git_commit": sha,
        "worktree_dirty": dirty,
        "machine": command_output("sysctl", &["-n", "machdep.cpu.brand_string", "hw.memsize"])
            .map(|s| s.replace('\n', " / ")),
        "model": model_dir,
        "model_config_sha256": config_sha,
        "kv_cache_mode": format!("{kv_mode:?}"),
        "mlx_cache_limit_bytes": cache_limit,
        "map": map,
        "map_sha256": map_sha,
        "dlsa_base_config": format!("{config:?}"),
        "prompt": prompt_meta,
        "prompt_offset": offset,
        "tail_rows": tail_len,
        "needles": opts.needles,
        "spec": opts.spec,
        "spec_tokens": opts.spec_tokens,
        "decode_steps": opts.decode_steps,
        "profile_chunks": opts.profile_chunks,
        "profile_decode_steps": opts.profile_decode,
        "results": results,
    });
    let text = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(&out, &text).expect("write the report");
    eprintln!("wrote {}", out.display());
}
