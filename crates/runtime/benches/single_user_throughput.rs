use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{ChatMessage, ChatMessageContent, GenerationRequest, Qwen35Provider};

struct GenerationElements {
    prefill: u64,
    decode: u64,
}

fn generation_elements(
    batch_size: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> GenerationElements {
    GenerationElements {
        prefill: (batch_size * prompt_tokens) as u64,
        // Prefill produces the logits for the first completion token. Only the
        // remaining completion tokens require autoregressive decode steps.
        decode: (batch_size * completion_tokens.saturating_sub(1)) as u64,
    }
}

const MODEL_ENV: &str = "QW_BENCH_MODEL";
const PROMPT: &str = concat!(
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
const DECODE_MAX_TOKENS: usize = 32;
const MTP_BLOCK_SIZE: usize = 3;

fn request(max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        max_tokens,
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        seed: Some(0),
    }
}

fn single_user_throughput(criterion: &mut Criterion) {
    // Checkpoints are intentionally not vendored. QW_BENCH_MODEL points at the
    // deterministic local Qwen fixture used by the runtime and CLI.
    let model_dir = PathBuf::from(
        std::env::var_os(MODEL_ENV)
            .unwrap_or_else(|| panic!("{MODEL_ENV} must point to a local Qwen checkpoint")),
    );
    let mut provider = Qwen35Provider::load(&model_dir)
        .unwrap_or_else(|error| panic!("failed to load {}: {error:#}", model_dir.display()));

    let prompt_tokens = provider
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
        .len();
    let prefill_request = request(1);
    let (prefill_output, _) = provider
        .generate_streaming_in_mode(&prefill_request, Qwen35GenerationMode::Automatic, |delta| {
            black_box(delta);
            true
        })
        .expect("warm up single-user prefill");
    assert!(!prefill_output.token_ids.is_empty());
    let prefill_elements =
        generation_elements(1, prompt_tokens, prefill_output.token_ids.len()).prefill;

    {
        let mut group = criterion.benchmark_group("single_user_prefill");
        group.throughput(Throughput::Elements(prefill_elements));
        group.bench_function("qwen", |bencher| {
            bencher.iter(|| {
                let output = provider
                    .generate_streaming_in_mode(
                        &prefill_request,
                        Qwen35GenerationMode::Automatic,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .expect("benchmark single-user prefill")
                    .0;
                assert_eq!(output.token_ids, prefill_output.token_ids);
                black_box(output);
            });
        });
        group.finish();
    }

    let decode_request = request(DECODE_MAX_TOKENS);
    let (baseline_output, _) = provider
        .generate_streaming_in_mode(&decode_request, Qwen35GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm up baseline single-user decode");
    let baseline_token_ids = baseline_output.token_ids;
    let baseline_tokens =
        generation_elements(1, prompt_tokens, baseline_token_ids.len()).decode as usize;
    assert!(
        baseline_tokens > 0,
        "the deterministic prompt must produce at least one autoregressive decode token"
    );

    let (mtp_output, mtp_stats) = provider
        .generate_streaming_in_mode(&decode_request, Qwen35GenerationMode::Mtp, |delta| {
            black_box(delta);
            true
        })
        .unwrap_or_else(|error| panic!("warm up MTP k={MTP_BLOCK_SIZE}: {error:#}"));
    assert_eq!(
        &mtp_output.token_ids, &baseline_token_ids,
        "baseline and bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs diverged"
    );
    let mtp_decode_tokens =
        generation_elements(1, prompt_tokens, mtp_output.token_ids.len()).decode as usize;
    assert!(
        mtp_decode_tokens > 0,
        "the deterministic MTP prompt must produce at least one autoregressive decode token"
    );
    assert!(
        mtp_stats
            .expect("explicit MTP mode must return MTP statistics")
            .proposed_draft_tokens
            > 0,
        "MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
    );

    {
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(baseline_tokens as u64));
        group.bench_function("baseline", |bencher| {
            bencher.iter(|| {
                let output = provider
                    .generate_streaming_in_mode(
                        &decode_request,
                        Qwen35GenerationMode::Baseline,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .expect("benchmark baseline single-user decode")
                    .0;
                assert_eq!(
                    &output.token_ids, &baseline_token_ids,
                    "deterministic baseline token IDs changed"
                );
                black_box(output);
            });
        });
        group.throughput(Throughput::Elements(mtp_decode_tokens as u64));
        group.bench_function("mtp_k3", |bencher| {
            bencher.iter(|| {
                let output = provider
                    .generate_streaming_in_mode(
                        &decode_request,
                        Qwen35GenerationMode::Mtp,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .unwrap_or_else(|error| panic!("benchmark MTP k={MTP_BLOCK_SIZE}: {error:#}"))
                    .0;
                assert_eq!(
                    &output.token_ids, &baseline_token_ids,
                    "baseline and bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs diverged"
                );
                black_box(output);
            });
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = single_user_throughput
}
criterion_main!(benches);
