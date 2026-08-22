use std::hint::black_box;
use std::path::PathBuf;

use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{
    ChatMessage, ChatMessageContent, GenerationRequest, KVCacheMode, PromptSnapshot, Qwen35Provider,
};

pub const DECODE_MAX_TOKENS: usize = 128;
pub const MTP_BLOCK_SIZE: usize = 3;
pub const PREFILL_MIN_TOKENS: usize = 4_096;
pub const PREFILL_MAX_TOKENS: usize = 6_000;
pub const LONG_CONTEXT_MIN_TOKENS: usize = 10_000;
pub const LONG_CONTEXT_64K_MIN_TOKENS: usize = 64_000;
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

pub struct LongConversationFixture {
    pub context_label: &'static str,
    pub prompt_ids: Vec<i32>,
    pub prefix_tokens: usize,
    pub new_prompt_tokens: usize,
    pub mtp_snapshot: PromptSnapshot,
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

pub fn prompt_token_ids(provider: &Qwen35Provider) -> Vec<i32> {
    let mut prompt = String::new();
    for record_index in 1..=32 {
        prompt.push_str(&format!(
            "Operational record {record_index:02}\n{PROMPT}\n\n"
        ));
        let prompt_ids = provider
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
            .expect("tokenize single-user prefill benchmark prompt");
        if prompt_ids.len() >= PREFILL_MIN_TOKENS {
            assert!(
                prompt_ids.len() <= PREFILL_MAX_TOKENS,
                "single-user prefill prompt has {} tokens, expected at most {PREFILL_MAX_TOKENS}",
                prompt_ids.len()
            );
            return prompt_ids;
        }
    }
    panic!("single-user prefill prompt did not reach {PREFILL_MIN_TOKENS} tokens");
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

fn text_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        name: None,
        content: Some(ChatMessageContent::Text(content)),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn long_conversation_token_ids(
    provider: &Qwen35Provider,
    context_label: &str,
    min_prefix_tokens: usize,
) -> (Vec<i32>, usize) {
    let supported_context_tokens = provider.supported_context_tokens();
    assert!(
        min_prefix_tokens + DECODE_MAX_TOKENS < supported_context_tokens,
        "{context_label} minimum prefix of {min_prefix_tokens} tokens leaves no room in the model's {supported_context_tokens}-token context"
    );

    let mut messages = Vec::new();
    let mut turn = 1;
    let history_ids = loop {
        messages.push(text_message(
            "user",
            format!("Conversation incident record {turn:03}\n{PROMPT}"),
        ));
        messages.push(text_message(
            "assistant",
            concat!(
                r#"{"severity":"high","summary":"Payment capture failures remain active.","#,
                r#""affected_order_ids":["A-1042","A-1047"],"#,
                r#""next_action":"Escalate INC-4821 and pause catalog imports.","#,
                r#""needs_escalation":true}"#
            )
            .to_owned(),
        ));
        let history_ids = provider
            .tokenize_history(&messages, &[], None, true)
            .expect("tokenize long-conversation benchmark history");
        if history_ids.len() >= min_prefix_tokens {
            break history_ids;
        }
        assert!(
            history_ids.len() + DECODE_MAX_TOKENS < supported_context_tokens,
            "{context_label} long-conversation history cannot reach {min_prefix_tokens} tokens within the model's {supported_context_tokens}-token context"
        );
        turn += 1;
    };
    let prefix_tokens = history_ids.len();
    assert!(
        prefix_tokens >= min_prefix_tokens,
        "{context_label} long-conversation prefix has {prefix_tokens} tokens, expected at least {min_prefix_tokens}"
    );

    messages.push(text_message("user", PROMPT.to_owned()));
    let prompt_ids = provider
        .tokenize_messages(&messages, &[], None, true)
        .expect("tokenize long-conversation benchmark continuation");
    assert!(
        prompt_ids.starts_with(&history_ids),
        "{context_label} long-conversation history must be an exact prefix of the continuation prompt"
    );
    assert!(
        prompt_ids.len() > prefix_tokens,
        "{context_label} long-conversation continuation must add prompt tokens"
    );
    assert!(
        prompt_ids.len() + DECODE_MAX_TOKENS <= supported_context_tokens,
        "{context_label} prompt and decode budget require {} tokens, exceeding the model's {supported_context_tokens}-token context",
        prompt_ids.len() + DECODE_MAX_TOKENS
    );
    (prompt_ids, prefix_tokens)
}

pub fn prepare_long_conversation_fixture(
    provider: &mut Qwen35Provider,
    context_label: &'static str,
    min_prefix_tokens: usize,
) -> LongConversationFixture {
    let (prompt_ids, prefix_tokens) =
        long_conversation_token_ids(provider, context_label, min_prefix_tokens);
    let history_ids = &prompt_ids[..prefix_tokens];
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));

    let mtp_prefix = provider
        .generate_mtp_streaming(
            history_ids,
            1,
            &sampling,
            MTP_BLOCK_SIZE,
            None,
            &[prefix_tokens],
            None,
            |_| true,
        )
        .expect("prefill long-conversation MTP prefix");
    let mtp_snapshot = mtp_prefix
        .prompt_snapshots
        .into_iter()
        .next()
        .expect("capture long-conversation MTP prefix snapshot");
    assert!(
        matches!(&mtp_snapshot, PromptSnapshot::Mtp(_)),
        "MTP prefix generation returned the wrong snapshot family"
    );
    assert_eq!(mtp_snapshot.token_len(), prefix_tokens);

    let (baseline_output, baseline_stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &mtp_snapshot,
            Qwen35GenerationMode::Baseline,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm up long-conversation baseline decode");
    assert_eq!(baseline_output.cached_tokens, prefix_tokens);
    assert!(
        baseline_stats.is_none(),
        "long-conversation baseline mode returned MTP statistics"
    );
    assert!(!baseline_output.token_ids.is_empty());

    let (mtp_output, mtp_stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &mtp_snapshot,
            Qwen35GenerationMode::Mtp,
            |delta| {
                black_box(delta);
                true
            },
        )
        .unwrap_or_else(|error| {
            panic!("warm up long-conversation MTP k={MTP_BLOCK_SIZE}: {error:#}")
        });
    assert_eq!(mtp_output.cached_tokens, prefix_tokens);
    let mtp_stats = mtp_stats.expect("explicit MTP mode must return MTP statistics");
    assert!(
        !mtp_output.token_ids.is_empty(),
        "the deterministic long-conversation prompt must produce completion tokens"
    );
    assert!(
        mtp_stats.proposed_draft_tokens > 0,
        "long-conversation MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
    );
    eprintln!(
        "MTP_LONG_CONTEXT_PROFILE context={} tokens={} prefix_tokens={} accepted={} proposed={} acceptance={:.2}% forwards={} draft_ms={:.3} verify_ms={:.3} walk_ms={:.3} reconcile_ms={:.3} materializations={} snapshots={}",
        context_label,
        mtp_output.token_ids.len(),
        prefix_tokens,
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

    let new_prompt_tokens = prompt_ids.len() - prefix_tokens;
    let mtp_decode_tokens = mtp_output.token_ids.len();
    LongConversationFixture {
        context_label,
        prompt_ids,
        prefix_tokens,
        new_prompt_tokens,
        mtp_snapshot,
        baseline_token_ids: baseline_output.token_ids,
        mtp_token_ids: mtp_output.token_ids,
        mtp_decode_tokens,
    }
}
