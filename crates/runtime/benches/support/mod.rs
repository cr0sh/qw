use std::hint::black_box;
use std::path::PathBuf;

use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{
    ChatMessage, ChatMessageContent, GenerationRequest, KVCacheMode, Qwen35Provider,
};

pub const DECODE_MAX_TOKENS: usize = 128;
pub const MTP_BLOCK_SIZE: usize = 3;
/// Environment variable pointing at the DFlash2 drafter checkpoint
/// directory (optional; defaults to the model cache path below).
pub const DRAFT_MODEL_ENV: &str = "QW_BENCH_DRAFT_MODEL";
/// Default DFlash2 drafter identifier resolved through the model cache.
pub const DEFAULT_DRAFT_MODEL_IDENTIFIER: &str = "incoai/Qwen3.8-27B-DFlash2";
pub const PROMPT: &str = concat!(
    "You are the on-call support operations analyst for Acme Commerce. ",
    "Review this incident and return only one compact JSON object with keys ",
    "`severity`, `summary`, `affected_order_ids`, `next_action`, and ",
    "`needs_escalation`. `severity` must be `low`, `medium`, or `high`; ",
    "`affected_order_ids` must contain only active orders; do not include names, ",
    "email addresses, payment details, or unverified root causes. Treat all ",
    "timestamps as UTC. Escalate when payment capture failures are still occurring.\n\n",
    "Incident INC-4821: At 09:14, after catalog import job 771 completed, ",
    "checkout returned `price_mismatch` for order A-1042 (active, $129.00) and ",
    "order A-1047 (active, $89.50). Order A-1038 was canceled before the import ",
    "and must not be included. At 09:21, a retry for A-1042 succeeded; at 09:26, ",
    "a new payment capture for A-1047 failed with the same error. The importer ",
    "reported no validation errors. Customer notes mention a cardholder's email ",
    "address, which must not be repeated. The next action should be a specific ",
    "operational step, not a diagnosis."
);

pub struct DecodeFixture {
    pub request: GenerationRequest,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
    pub mtp_decode_tokens: usize,
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

pub fn load_provider() -> Qwen35Provider {
    // Unified with `qw generate` / `qw serve`: QW_MODEL_PATH override, else
    // the model cache path for the resolver's default identifier.
    let model_dir = qw_runtime::resolve_model_path(None)
        .unwrap_or_else(|error| panic!("failed to resolve benchmark model path: {error:#}"));
    Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
        .unwrap_or_else(|error| panic!("failed to load {}: {error:#}", model_dir.display()))
}

#[allow(dead_code)]
pub fn prompt_token_ids(provider: &Qwen35Provider) -> Vec<i32> {
    provider
        .tokenize_messages(
            &[ChatMessage {
                role: "user".to_owned(),
                name: None,
                content: Some(ChatMessageContent::Text(PROMPT.to_owned())),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            &[],
            None,
            true,
        )
        .expect("tokenize single-user benchmark prompt")
}

/// Resolve the DFlash2 drafter directory: `QW_BENCH_DRAFT_MODEL` when set,
/// else the model cache path for the default identifier (mirrors the model
/// resolution `qw generate` / `qw serve` use).
#[allow(dead_code)]
pub fn draft_model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(DRAFT_MODEL_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME must be set to resolve the default drafter cache path");
    qw_runtime::model_cache_path(&home, DEFAULT_DRAFT_MODEL_IDENTIFIER)
        .expect("default DFlash2 drafter identifier is valid")
}

pub fn prepare_decode_fixture(provider: &mut Qwen35Provider) -> DecodeFixture {
    let request = request(DECODE_MAX_TOKENS);
    let (baseline_output, _) = provider
        .generate_streaming_in_mode(&request, Qwen35GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm up baseline single-user decode");
    let baseline_token_ids = baseline_output.token_ids;
    assert!(!baseline_token_ids.is_empty());

    let (mtp_output, mtp_stats) = provider
        .generate_streaming_in_mode(&request, Qwen35GenerationMode::Mtp, |delta| {
            black_box(delta);
            true
        })
        .unwrap_or_else(|error| panic!("warm up MTP k={MTP_BLOCK_SIZE}: {error:#}"));
    let mtp_stats = mtp_stats.expect("explicit MTP mode must return MTP statistics");
    assert!(
        !mtp_output.token_ids.is_empty(),
        "the deterministic MTP prompt must produce at least one completion token"
    );
    assert!(
        mtp_stats.proposed_draft_tokens > 0,
        "MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
    );
    eprintln!(
        "MTP_PROFILE tokens={} accepted={} proposed={} acceptance={:.2}% forwards={} draft_ms={:.3} verify_ms={:.3} walk_ms={:.3} reconcile_ms={:.3} materializations={} snapshots={}",
        mtp_output.token_ids.len(),
        mtp_stats.accepted_draft_tokens,
        mtp_stats.proposed_draft_tokens,
        mtp_stats.acceptance_percentage(),
        mtp_stats.target_forward_calls,
        mtp_stats.draft_time.as_secs_f64() * 1_000.0,
        mtp_stats.target_verify_time.as_secs_f64() * 1_000.0,
        mtp_stats.walk_time.as_secs_f64() * 1_000.0,
        mtp_stats.reconcile_time.as_secs_f64() * 1_000.0,
        mtp_stats.full_state_materializations,
        mtp_stats.cache_snapshot_count,
    );

    let mtp_decode_tokens = mtp_output.token_ids.len();
    DecodeFixture {
        request,
        baseline_token_ids,
        mtp_token_ids: mtp_output.token_ids,
        mtp_decode_tokens,
    }
}
