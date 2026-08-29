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
    let target_snapshot = match &snapshot {
        PromptSnapshot::Baseline(snapshot) => snapshot,
        PromptSnapshot::Mtp(snapshot) => snapshot.target_snapshot(),
    };
    let auxiliary_name = target_snapshot
        .paged_tensor_names()
        .find(|name| name.ends_with(".auxiliary_keys"))
        .expect("64k target snapshot must include QSA keys");
    let auxiliary_keys = target_snapshot
        .paged_tensor(auxiliary_name)
        .and_then(|tensor| tensor.materialize())
        .expect("materialize 64k QSA keys");
    assert_eq!(
        mlxcel_core::array_shape(&auxiliary_keys)[1] as usize,
        prefix_tokens,
        "64k QSA snapshot must retain the complete prefix"
    );
    let (_baseline, stats) = provider
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
    let (_mtp_warm, stats) = provider
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
    LongConversationFixture {
        prompt_ids,
        prefix_tokens,
        snapshot,
    }
}
