use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{GenerationRequest, Qwen35Provider};

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

fn measured_duration(milliseconds: f64) -> Duration {
    Duration::from_secs_f64(milliseconds / 1_000.0)
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

    let prefill_request = request(1);
    let (_, prefill_probe) = provider
        .generate_with_stats(&prefill_request)
        .expect("warm up single-user prefill");
    assert!(prefill_probe.prompt_tokens > 0);

    {
        let prompt_tokens = prefill_probe.prompt_tokens;
        let mut group = criterion.benchmark_group("single_user_prefill");
        group.throughput(Throughput::Elements(prompt_tokens as u64));
        group.bench_function("qwen", |bencher| {
            // iter_custom reports only the canonical generator's prefill
            // interval; model loading, tokenization, cache reset, and decode
            // all execute outside the duration returned to Criterion.
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let (output, stats) = provider
                        .generate_with_stats(&prefill_request)
                        .expect("benchmark single-user prefill");
                    assert_eq!(stats.prompt_tokens, prompt_tokens);
                    elapsed += measured_duration(stats.prefill_time_ms);
                    black_box(output);
                }
                elapsed
            });
        });
        group.finish();
    }

    let decode_request = request(DECODE_MAX_TOKENS);
    let (baseline_output, baseline_probe) = provider
        .generate_with_stats_in_mode(&decode_request, Qwen35GenerationMode::Baseline)
        .expect("warm up baseline single-user decode");
    let baseline_token_ids = baseline_output.token_ids;
    let baseline_tokens = baseline_probe.generated_tokens.saturating_sub(1);
    assert!(
        baseline_tokens > 0,
        "the deterministic prompt must produce at least one autoregressive decode token"
    );
    println!(
        "DECODE_BENCH_SUMMARY mode=baseline k=1 decode_tokens={baseline_tokens} decode_milliseconds={:.6} decode_tokens_per_second={:.6}",
        baseline_probe.decode_time_ms,
        baseline_probe.decode_tok_per_sec,
    );

    let (mtp_output, mtp_probe, mtp_stats) = provider
        .generate_with_mtp_stats(&decode_request, MTP_BLOCK_SIZE)
        .unwrap_or_else(|error| panic!("warm up MTP k={MTP_BLOCK_SIZE}: {error:#}"));
    assert_eq!(
        &mtp_output.token_ids, &baseline_token_ids,
        "baseline and bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs diverged"
    );
    let mtp_decode_tokens = mtp_probe.generated_tokens.saturating_sub(1);
    assert!(
        mtp_decode_tokens > 0,
        "the deterministic MTP prompt must produce at least one autoregressive decode token"
    );
    assert!(
        mtp_stats.proposed_draft_tokens > 0,
        "MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
    );
    println!(
        "MTP_BENCH_SUMMARY k={MTP_BLOCK_SIZE} accepted_draft_tokens={} proposed_draft_tokens={} acceptance_percentage={:.6} decode_tokens={mtp_decode_tokens} decode_milliseconds={:.6} decode_tokens_per_second={:.6}",
        mtp_stats.accepted_draft_tokens,
        mtp_stats.proposed_draft_tokens,
        mtp_stats.acceptance_percentage(),
        mtp_probe.decode_time_ms,
        mtp_probe.decode_tok_per_sec,
    );

    {
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(baseline_tokens as u64));
        group.bench_function("baseline", |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let (output, stats) = provider
                        .generate_with_stats_in_mode(
                            &decode_request,
                            Qwen35GenerationMode::Baseline,
                        )
                        .expect("benchmark baseline single-user decode");
                    assert_eq!(
                        stats.generated_tokens.saturating_sub(1),
                        baseline_tokens,
                        "deterministic baseline decode length changed"
                    );
                    assert_eq!(
                        &output.token_ids, &baseline_token_ids,
                        "deterministic baseline token IDs changed"
                    );
                    elapsed += measured_duration(stats.decode_time_ms);
                    black_box(output);
                }
                elapsed
            });
        });
        group.throughput(Throughput::Elements(mtp_decode_tokens as u64));
        group.bench_function("mtp_k3", |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let (output, stats, _) = provider
                        .generate_with_mtp_stats(&decode_request, MTP_BLOCK_SIZE)
                        .unwrap_or_else(|error| {
                            panic!("benchmark MTP k={MTP_BLOCK_SIZE}: {error:#}")
                        });
                    assert_eq!(
                        stats.generated_tokens.saturating_sub(1),
                        mtp_decode_tokens,
                        "deterministic MTP k={MTP_BLOCK_SIZE} decode length changed"
                    );
                    assert_eq!(
                        &output.token_ids, &baseline_token_ids,
                        "baseline and bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs diverged"
                    );
                    elapsed += measured_duration(stats.decode_time_ms);
                    black_box(output);
                }
                elapsed
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
