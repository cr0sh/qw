// SPDX-License-Identifier: Apache-2.0
//
//! Real-model D-LSA benchmark with a stage profile (`docs/dlsa.md` §14.3): dense vs D-LSA
//! prefill and decode at long contexts, on the real checkpoint and the converted map.
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
//!   `mlx` (the task-3 path), `metal` (K1 + K2 kernels), `metal-select` (K1 + the task-3 read).
//! - `QW_DLSA_BENCH_DECODE` (default 32): unfenced greedy decode steps.
//! - `QW_DLSA_BENCH_PROFILE_CHUNKS` (default 2) / `QW_DLSA_BENCH_PROFILE_DECODE` (default 8):
//!   the last prefill chunks, and extra decode steps after the unfenced ones, run fenced (every
//!   decoder layer and every D-LSA stage ends in an `eval`) for the stage breakdown. The
//!   throughput numbers come from the same pass: prefill tok/s includes the fenced chunks (they
//!   are reported separately), decode ms/token only counts unfenced steps.
//! - `QW_DLSA_BENCH_PROMPT`: a token file (i32 little-endian). Read if it exists, else written
//!   from the corpus, so later runs reuse the exact prompt whatever the worktree holds.
//! - `QW_DLSA_BENCH_OUT`: the JSON report (must not exist); default under `$TMPDIR`.
//! - `QW_DLSA_BENCH_LOGITS=1`: also write the final-prefill and decode logits as raw f32 next
//!   to the report (the report always holds their sha256).
//! - `QW_DLSA_BENCH_KV=int8`: an Int8 KV cache instead of Fp16 (for memory pressure).
//! - `MLXCEL_CACHE_LIMIT` (bytes, default 8 GiB as the provider sets it): the MLX free-buffer
//!   cache. Without a limit the cache grows past the machine's memory at 256k and swaps.
//!   The report records the limit and the swap in use before and after each mode.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use super::{Qwen35AttentionVariant, Qwen35Model, prepare_attention_forward};
use crate::dlsa::profile::{self, DlsaProfile, DlsaStageTime};
use crate::dlsa::{
    DlsaBackend, DlsaConfig, DlsaMapKind, DlsaStats, resolve_dlsa_map_path, sha256_hex, to_host_f32,
};
use crate::qwen_vl_position::decode_rope_positions;
use crate::qwen3_next::{DlsaLayerStep, Qwen3NextCache};

const PREFILL_CHUNK: usize = 1536;
const FULL_ATTENTION_LAYERS: usize = 16;

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

/// One mode at one context: a fresh sequence, chunked prefill, then greedy decode.
#[allow(clippy::too_many_arguments)]
fn run_mode(
    model: &mut Qwen35Model,
    mode: &str,
    config: &DlsaConfig,
    prompt: &[i32],
    context: usize,
    decode_steps: usize,
    profile_chunks: usize,
    profile_decode: usize,
    logits_out: Option<&Path>,
) -> Value {
    let dlsa_config = (mode != "dense").then(|| DlsaConfig {
        backend: DlsaBackend::parse(mode)
            .unwrap_or_else(|e| panic!("mode {mode}: dense, or a backend: {e}")),
        ..config.clone()
    });
    match &dlsa_config {
        None => model.disable_dlsa(),
        Some(c) => model.enable_dlsa(c.clone()).expect("enable D-LSA"),
    }
    model.reset_runtime_state();
    mlxcel_core::memory::clear_cache();
    let swap_before = command_output("sysctl", &["-n", "vm.swapusage"]);
    mlxcel_core::memory::reset_peak_memory();
    profile::set_override(Some(false));
    profile::reset();

    let chunks: Vec<&[i32]> = prompt.chunks(PREFILL_CHUNK).collect();
    let fenced_from = chunks.len().saturating_sub(profile_chunks);
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
    profile::reset();

    let mut logits_dump = f32_bytes(&last);
    let mut logits_sha = vec![sha256_hex(&f32_bytes(&last))];
    let mut tokens = vec![argmax(&last)];
    let mut decode_ms = Vec::with_capacity(decode_steps);
    let mut fenced_decode_ms = Vec::with_capacity(profile_decode);
    for i in 0..decode_steps + profile_decode {
        let fenced = i >= decode_steps;
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
    let peak = mlxcel_core::memory::peak_memory();
    if let Some(path) = logits_out {
        let path = path.with_extension(format!("{mode}.{context}.logits.f32"));
        std::fs::write(&path, &logits_dump).expect("write logits");
        eprintln!("wrote {}", path.display());
    }

    let n = prompt.len() as f64;
    let unfenced_ms: f64 = chunk_ms[..fenced_from].iter().sum();
    let decode_median = quantile(&decode_ms, 0.5);
    let report = json!({
        "mode": mode,
        "dlsa_config": dlsa_config.as_ref().map(|c| format!("{c:?}")),
        "context": context,
        "prompt_tokens": prompt.len(),
        "prefill": {
            "chunks": chunks.len(),
            "chunk": PREFILL_CHUNK,
            "wall_s": prefill.as_secs_f64(),
            "tok_s": n / prefill.as_secs_f64(),
            "fenced_chunks_from": fenced_from,
            "unfenced_chunks_tok_s": (fenced_from * PREFILL_CHUNK) as f64 / (unfenced_ms / 1e3),
            "chunk_ms": chunk_ms,
            "peak_memory_bytes": prefill_peak,
        },
        "decode": {
            "steps": decode_steps,
            "median_ms": decode_median,
            "p90_ms": quantile(&decode_ms, 0.9),
            "mean_ms": decode_ms.iter().sum::<f64>() / decode_ms.len().max(1) as f64,
            "tok_s": 1e3 / decode_median,
            "step_ms": decode_ms,
            "fenced_step_ms": fenced_decode_ms,
        },
        "tokens": tokens,
        "logits_sha256": {
            "final_prefill_row": logits_sha[0],
            "decode_steps": &logits_sha[1..],
        },
        "dlsa_stats": stats_json(model.dlsa_stats(), model.dlsa_memory_bytes()),
        "peak_memory_bytes": peak,
        "active_memory_bytes": mlxcel_core::memory::active_memory(),
        "cache_memory_bytes": mlxcel_core::memory::cache_memory(),
        "swap_before": swap_before,
        "swap_after": command_output("sysctl", &["-n", "vm.swapusage"]),
        "profile": {
            "prefill": profile_json(&prefill_profile),
            "decode": profile_json(&decode_profile),
        },
    });
    eprintln!(
        "{mode} T={context}: prefill {:.1} s ({:.0} tok/s), decode median {decode_median:.1} ms (p90 {:.1}), peak {:.1} GiB",
        prefill.as_secs_f64(),
        n / prefill.as_secs_f64(),
        quantile(&decode_ms, 0.9),
        peak as f64 / (1u64 << 30) as f64
    );
    // Free this mode's KV, index caches and cached buffers before the next mode starts
    model.reset_runtime_state();
    mlxcel_core::memory::clear_cache();
    report
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
    let decode_steps = env_usize("QW_DLSA_BENCH_DECODE", 32);
    let profile_chunks = env_usize("QW_DLSA_BENCH_PROFILE_CHUNKS", 2);
    let profile_decode = env_usize("QW_DLSA_BENCH_PROFILE_DECODE", 8);
    let kv_mode = match std::env::var("QW_DLSA_BENCH_KV").as_deref() {
        Ok("int8") => KVCacheMode::Int8,
        _ => KVCacheMode::Fp16,
    };
    let model_dir = crate::resolve_model_path(None)
        .expect("QW_MODEL_PATH or the default model cache must hold the Qwen3.8-27B checkpoint");
    let map = resolve_dlsa_map_path(DlsaMapKind::Distilled, None)
        .expect("the converted D-LSA map (run `qw download --dlsa-map`, or set QW_DLSA_MAP_PATH)");
    let config = DlsaConfig::new(map.clone());
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
        .map(|&t| t / PREFILL_CHUNK * PREFILL_CHUNK)
        .max()
        .expect("a context");
    let (all_tokens, prompt_meta) = prompt_tokens(&model_dir, max_prompt);
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
        let prompt_len = context / PREFILL_CHUNK * PREFILL_CHUNK;
        assert!(
            prompt_len + decode_steps + profile_decode < context,
            "T={context}: prompt {prompt_len} + decode does not fit"
        );
        let prompt = &all_tokens[..prompt_len];
        let mut by_mode = Vec::new();
        for mode in &modes {
            by_mode.push(run_mode(
                &mut model,
                mode,
                &config,
                prompt,
                context,
                decode_steps,
                profile_chunks,
                profile_decode,
                logits_out,
            ));
        }
        results.push(json!({"context": context, "modes": by_mode}));
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
        "decode_steps": decode_steps,
        "profile_chunks": profile_chunks,
        "profile_decode_steps": profile_decode,
        "results": results,
    });
    let text = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(&out, &text).expect("write the report");
    eprintln!("wrote {}", out.display());
}
