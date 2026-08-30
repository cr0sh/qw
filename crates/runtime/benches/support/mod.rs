use std::hint::black_box;

use qw_runtime::provider::Qwen4GenerationMode;
use qw_runtime::{
    ChatMessage, ChatMessageContent, GenerationRequest, KVCacheMode, PromptSnapshot, Qwen4Provider,
};

pub const DECODE_MAX_TOKENS: usize = 32;
pub const MTP_BLOCK_SIZE: usize = 3;
pub const LONG_CONTEXT_64K_MIN_TOKENS: usize = 64_000;
pub fn long_context_tokens() -> usize {
    std::env::var("QW_BENCH_LONG_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(LONG_CONTEXT_64K_MIN_TOKENS)
}

pub struct BenchmarkMemoryReport {
    enabled: bool,
}

impl BenchmarkMemoryReport {
    /// Resolve the opt-in once and reset the MLX peak immediately after model
    /// load, before fixture construction contributes to the measured peak.
    pub fn after_model_load() -> Self {
        let enabled = std::env::var("QW_BENCH_MEMORY").as_deref() == Ok("1");
        if enabled {
            mlxcel_core::memory::reset_peak_memory();
        }
        Self { enabled }
    }

    /// Emit the first long-context record after the prefix snapshot is
    /// extracted, before any cached restore.
    pub fn emit_fresh_snapshot(
        &self,
        context_tokens: usize,
        prefix_tokens: usize,
        snapshot: &PromptSnapshot,
    ) {
        self.emit(
            "fresh_snapshot",
            context_tokens,
            prefix_tokens,
            snapshot,
            None,
            None,
        );
    }

    /// Emit the second long-context record after fixture warm restores.
    ///
    /// Report emission is benchmark setup only, never a timed Criterion
    /// closure.
    pub fn emit_post_restore_fixture(
        &self,
        context_tokens: usize,
        prefix_tokens: usize,
        snapshot: &PromptSnapshot,
        baseline_token_ids: &[i32],
        mtp_token_ids: &[i32],
    ) {
        self.emit(
            "post_restore_fixture",
            context_tokens,
            prefix_tokens,
            snapshot,
            Some(baseline_token_ids),
            Some(mtp_token_ids),
        );
    }

    fn emit(
        &self,
        phase: &'static str,
        context_tokens: usize,
        prefix_tokens: usize,
        snapshot: &PromptSnapshot,
        baseline_token_ids: Option<&[i32]>,
        mtp_token_ids: Option<&[i32]>,
    ) {
        if !self.enabled {
            return;
        }
        let mlx = mlxcel_core::memory::snapshot();
        let process = mlxcel_core::memory::process_snapshot().map(|process| {
            serde_json::json!({
                "resident_bytes": process.resident_bytes,
                "wired_bytes": process.wired_bytes,
                "physical_footprint_bytes": process.physical_footprint_bytes,
                "lifetime_peak_physical_footprint_bytes":
                    process.lifetime_peak_physical_footprint_bytes,
            })
        });
        let recommended_working_set = mlxcel_core::get_wired_limit() as u64;
        let configured_wired_limit = (((u128::from(
            mlxcel_core::hardware::system_memory_bytes(),
        ) * 85)
            / 100) as u64)
            .min(recommended_working_set);
        let record = serde_json::json!({
            "schema": "qw_bench_memory",
            "phase": phase,
            "pid": std::process::id(),
            "context_tokens": context_tokens,
            "prefix_tokens": prefix_tokens,
            "prompt_snapshot_logical_bytes": PromptSnapshot::nbytes(snapshot),
            "baseline_token_ids": baseline_token_ids,
            "mtp_token_ids": mtp_token_ids,
            "mlx": {
                "active_bytes": mlx.active_bytes,
                "peak_bytes": mlx.peak_bytes,
                "cache_bytes": mlx.cache_bytes,
                "limit_bytes": mlx.limit_bytes,
                "recommended_max_working_set_bytes": recommended_working_set,
                "configured_wired_limit_bytes": configured_wired_limit,
            },
            "process": process,
        });
        eprintln!("QW_BENCH_MEMORY={record}");
    }
}

pub const FRESH_PREFILL_REPETITIONS: usize = 24;

const PROMPT: &str = concat!(
    "You are the on-call support operations analyst. ",
    "Summarize the incident, list affected systems, and state the next action. ",
    "Treat timestamps as UTC and do not invent facts.\n",
    "Incident: checkout requests intermittently failed after a catalog import; ",
    "one retry succeeded and one payment capture still fails.\n"
);

pub struct DecodeFixture {
    pub request: GenerationRequest,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
}
pub struct PrefillFixture {
    pub request: GenerationRequest,
    pub prompt_tokens: usize,
    pub first_token_id: i32,
}

pub struct LongConversationFixture {
    pub prompt_ids: Vec<i32>,
    pub prefix_tokens: usize,
    pub snapshot: PromptSnapshot,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
}

pub fn request(max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        max_tokens,
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        seed: Some(0),
    }
}

pub fn load_provider() -> Qwen4Provider {
    let model_dir = qw_runtime::resolve_model_path(None)
        .unwrap_or_else(|error| panic!("failed to resolve benchmark model path: {error:#}"));
    Qwen4Provider::load(&model_dir, KVCacheMode::Fp8)
        .unwrap_or_else(|error| panic!("failed to load {}: {error:#}", model_dir.display()))
}

pub fn prepare_decode_fixture(provider: &mut Qwen4Provider) -> DecodeFixture {
    let request = request(DECODE_MAX_TOKENS);
    let (baseline, _) = provider
        .generate_streaming_in_mode(&request, Qwen4GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm baseline decode");
    let (mtp, stats) = provider
        .generate_streaming_in_mode(&request, Qwen4GenerationMode::Mtp, |delta| {
            black_box(delta);
            true
        })
        .expect("warm MTP decode");
    // Baseline T=1 and batched MTP T=N are deterministic independently but
    // need not have identical greedy token alignment. Each benchmark iteration
    // below is checked against its own warmed token sequence.
    assert!(
        stats.is_some_and(|stats| stats.proposed_draft_tokens > 0),
        "MTP warmup must propose draft tokens"
    );
    DecodeFixture {
        request,
        baseline_token_ids: baseline.token_ids,
        mtp_token_ids: mtp.token_ids,
    }
}
pub fn prepare_prefill_fixture(provider: &mut Qwen4Provider) -> PrefillFixture {
    let request = GenerationRequest {
        prompt: PROMPT.repeat(FRESH_PREFILL_REPETITIONS),
        max_tokens: 1,
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        seed: Some(0),
    };
    let (generation, stats) = provider
        .benchmark_streaming_in_mode(&request, Qwen4GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm fresh prefill");
    assert!(stats.is_none());
    PrefillFixture {
        request,
        prompt_tokens: generation.prompt_tokens,
        first_token_id: generation.token_ids[0],
    }
}

fn long_prompt_ids(provider: &Qwen4Provider, min_tokens: usize) -> Vec<i32> {
    let mut prompt = String::new();
    for index in 0..20_000 {
        prompt.push_str(&format!("Record {index}: {PROMPT}\n"));
        if index % 128 != 127 {
            continue;
        }
        let ids = provider
            .tokenize_messages(
                &[ChatMessage {
                    role: "user".to_owned(),
                    name: None,
                    content: Some(ChatMessageContent::Text(prompt.clone())),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
                &[],
                None,
                true,
            )
            .expect("tokenize long benchmark prompt");
        if ids.len() >= min_tokens {
            let suffix_len = 16.min(min_tokens);
            let mut truncated = ids[..min_tokens - suffix_len].to_vec();
            truncated.extend_from_slice(&ids[ids.len() - suffix_len..]);
            return truncated;
        }
    }
    panic!("long benchmark prompt did not reach {min_tokens} tokens")
}

pub fn prepare_long_conversation_fixture(
    provider: &mut Qwen4Provider,
    memory_report: &BenchmarkMemoryReport,
    min_prefix_tokens: usize,
) -> LongConversationFixture {
    let prompt_ids = long_prompt_ids(provider, min_prefix_tokens + 1_536);
    let prefix_tokens = min_prefix_tokens;
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let mtp = provider
        .generate_mtp_streaming(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            MTP_BLOCK_SIZE,
            None,
            &[prefix_tokens],
            None,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("prepare long-context MTP snapshot");
    let snapshot = mtp
        .prompt_snapshots
        .into_iter()
        .find(|snapshot| snapshot.token_len() == prefix_tokens)
        .expect("MTP checkpoint at requested 64k prefix");
    memory_report.emit_fresh_snapshot(prompt_ids.len(), prefix_tokens, &snapshot);
    let target_snapshot = match &snapshot {
        PromptSnapshot::Baseline(snapshot) => snapshot,
        PromptSnapshot::Mtp(snapshot) => snapshot.target_snapshot(),
    };
    let auxiliary_name = target_snapshot
        .tensor_names()
        .find(|name| name.ends_with(".auxiliary_keys"))
        .expect("64k target snapshot must include bounded QSA keys");
    let auxiliary_keys = mlxcel_core::copy(
        target_snapshot
            .tensor(auxiliary_name)
            .expect("materialize 64k QSA keys"),
    );
    let layer_prefix = auxiliary_name
        .strip_suffix("auxiliary_keys")
        .expect("QSA snapshot tensor suffix");
    let scalar = |suffix: &str| {
        let value = target_snapshot
            .tensor(&format!("{layer_prefix}{suffix}"))
            .unwrap_or_else(|| panic!("64k QSA snapshot must include {suffix}"));
        mlxcel_core::item_i32(&mlxcel_core::reshape(value, &[]))
    };
    let tail_start = scalar("auxiliary_tail_start");
    let tail_end = scalar("auxiliary_tail_end");
    let horizon = scalar("auxiliary_rollback_horizon");
    let ratio = scalar("auxiliary_block_size");
    let raw_rows = mlxcel_core::array_shape(&auxiliary_keys)[1];
    assert_eq!(tail_end as usize, prefix_tokens);
    assert_eq!(horizon as usize, MTP_BLOCK_SIZE);
    assert_eq!(
        tail_start,
        tail_end.saturating_sub(horizon).max(0) / ratio * ratio
    );
    assert_eq!(raw_rows, tail_end - tail_start);
    assert!(raw_rows <= horizon + ratio - 1);
    let block_keys = target_snapshot
        .paged_tensor(&format!("{layer_prefix}auxiliary_block_keys"))
        .and_then(|tensor| tensor.materialize())
        .expect("64k QSA snapshot must retain logical block summaries");
    assert_eq!(mlxcel_core::array_shape(&block_keys)[2], tail_end / ratio);
    let (baseline, stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &snapshot,
            Qwen4GenerationMode::Baseline,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm cached 64k baseline");
    assert!(stats.is_none());
    let (baseline_repeat, stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &snapshot,
            Qwen4GenerationMode::Baseline,
            |_| true,
        )
        .expect("repeat warm cached long-context baseline");
    assert!(stats.is_none());
    assert_eq!(
        baseline_repeat.token_ids, baseline.token_ids,
        "repeated baseline snapshot restore changed greedy output"
    );
    let (mtp_warm, stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &snapshot,
            Qwen4GenerationMode::Mtp,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm cached 64k MTP");
    assert!(stats.is_some());
    // The cached MTP benchmark validates repeatability against this warmed MTP
    // sequence; baseline repeatability is checked separately above.
    LongConversationFixture {
        prompt_ids,
        prefix_tokens,
        snapshot,
        baseline_token_ids: baseline.token_ids,
        mtp_token_ids: mtp_warm.token_ids,
    }
}
