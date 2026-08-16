use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{GenerationRequest, Qwen35Provider};

const MODEL_ENV: &str = "QW_BENCH_MODEL";
const PROMPT: &str =
    "Continue counting upward from one, writing each integer on its own line without stopping.";
const DECODE_MAX_TOKENS: usize = 32;

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
    let (mtp_output, mtp_probe) = provider
        .generate_with_stats_in_mode(&decode_request, Qwen35GenerationMode::Mtp)
        .expect("warm up MTP single-user decode");
    assert_eq!(
        baseline_output.token_ids, mtp_output.token_ids,
        "baseline and bundled-MTP greedy token IDs diverged"
    );
    let baseline_tokens = baseline_probe.generated_tokens.saturating_sub(1);
    let mtp_tokens = mtp_probe.generated_tokens.saturating_sub(1);
    assert!(
        baseline_tokens > 0 && mtp_tokens > 0,
        "the deterministic prompt must produce at least one autoregressive decode token"
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
                    elapsed += measured_duration(stats.decode_time_ms);
                    black_box(output);
                }
                elapsed
            });
        });
        group.throughput(Throughput::Elements(mtp_tokens as u64));
        group.bench_function("mtp", |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let (output, stats) = provider
                        .generate_with_stats_in_mode(
                            &decode_request,
                            Qwen35GenerationMode::Mtp,
                        )
                        .expect("benchmark MTP single-user decode");
                    assert_eq!(
                        stats.generated_tokens.saturating_sub(1),
                        mtp_tokens,
                        "deterministic MTP decode length changed"
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
