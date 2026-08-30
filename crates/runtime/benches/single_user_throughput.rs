mod support;

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qw_runtime::provider::Qwen4GenerationMode;
use support::{
    BenchmarkMemoryReport, DECODE_MAX_TOKENS, load_provider, long_context_tokens,
    prepare_decode_fixture, prepare_long_conversation_fixture, prepare_prefill_fixture,
};
fn case_enabled(name: &str) -> bool {
    std::env::var("QW_BENCH_CASE").map_or(true, |selected| selected == name)
}

fn benchmark_fresh_decode(criterion: &mut Criterion) {
    if !case_enabled("fresh") && !case_enabled("fresh_decode") {
        return;
    }
    let mut provider = load_provider();
    let fixture = prepare_decode_fixture(&mut provider);
    let mut group = criterion.benchmark_group("single_user_decode");
    group.sample_size(10);
    group.throughput(Throughput::Elements(DECODE_MAX_TOKENS as u64));

    for (name, mode, expected) in [
        (
            "fresh_baseline",
            Qwen4GenerationMode::Baseline,
            fixture.baseline_token_ids.as_slice(),
        ),
        (
            "fresh_mtp",
            Qwen4GenerationMode::Mtp,
            fixture.mtp_token_ids.as_slice(),
        ),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iterations {
                    let (output, stats) = provider
                        .benchmark_streaming_in_mode(&fixture.request, mode, |delta| {
                            black_box(delta);
                            true
                        })
                        .unwrap_or_else(|error| panic!("benchmark {name}: {error:#}"));
                    assert_eq!(output.token_ids, expected, "deterministic output changed");
                    assert_eq!(stats.is_some(), mode == Qwen4GenerationMode::Mtp);
                    decode_time += output.decode_time;
                    black_box(output);
                }
                decode_time
            });
        });
    }
    group.finish();
}
fn benchmark_fresh_prefill(criterion: &mut Criterion) {
    if !case_enabled("fresh") && !case_enabled("fresh_prefill") {
        return;
    }
    let mut provider = load_provider();
    let fixture = prepare_prefill_fixture(&mut provider);
    let mut group = criterion.benchmark_group("single_user_prefill");
    group.sample_size(10);
    group.throughput(Throughput::Elements(fixture.prompt_tokens as u64));
    group.bench_function("fresh", |bencher| {
        bencher.iter_custom(|iterations| {
            let mut prefill_time = Duration::ZERO;
            for _ in 0..iterations {
                let (output, stats) = provider
                    .benchmark_streaming_in_mode(
                        &fixture.request,
                        Qwen4GenerationMode::Baseline,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .unwrap_or_else(|error| panic!("benchmark fresh prefill: {error:#}"));
                assert_eq!(output.prompt_tokens, fixture.prompt_tokens);
                assert_eq!(output.token_ids, [fixture.first_token_id]);
                assert!(stats.is_none());
                prefill_time += output.prefill_time;
                black_box(output);
            }
            prefill_time
        });
    });
    group.finish();
}

fn benchmark_64k_cached_decode(criterion: &mut Criterion) {
    if !case_enabled("64k") {
        return;
    }
    let mut provider = load_provider();
    let memory_report = BenchmarkMemoryReport::after_model_load();
    let fixture = prepare_long_conversation_fixture(&mut provider, long_context_tokens());
    memory_report.emit_post_fixture(
        fixture.prompt_ids.len(),
        fixture.prefix_tokens,
        Some(&fixture.snapshot),
    );
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let mut group = criterion.benchmark_group("single_user_decode");
    group.sample_size(10);
    group.throughput(Throughput::Elements(DECODE_MAX_TOKENS as u64));

    for (name, mode, expected) in [
        (
            "cached_64k_baseline",
            Qwen4GenerationMode::Baseline,
            fixture.baseline_token_ids.as_slice(),
        ),
        (
            "cached_64k_mtp",
            Qwen4GenerationMode::Mtp,
            fixture.mtp_token_ids.as_slice(),
        ),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iterations {
                    let (output, stats) = provider
                        .benchmark_cached_streaming_in_mode(
                            &fixture.prompt_ids,
                            DECODE_MAX_TOKENS,
                            &sampling,
                            &fixture.snapshot,
                            mode,
                            |delta| {
                                black_box(delta);
                                true
                            },
                        )
                        .unwrap_or_else(|error| panic!("benchmark {name}: {error:#}"));
                    assert_eq!(output.cached_tokens, fixture.prefix_tokens);
                    assert_eq!(output.token_ids.len(), DECODE_MAX_TOKENS);
                    assert_eq!(output.token_ids, expected, "deterministic output changed");
                    assert_eq!(stats.is_some(), mode == Qwen4GenerationMode::Mtp);
                    decode_time += output.decode_time;
                    black_box(output);
                }
                decode_time
            });
        });
    }
    group.finish();
    let uncached_tokens = fixture.prompt_ids.len() - fixture.prefix_tokens;
    let mut prefill_group = criterion.benchmark_group("single_user_prefill");
    prefill_group.sample_size(10);
    prefill_group.throughput(Throughput::Elements(uncached_tokens as u64));
    prefill_group.bench_function("cached_64k", |bencher| {
        bencher.iter_custom(|iterations| {
            let mut prefill_time = Duration::ZERO;
            for _ in 0..iterations {
                let (output, stats) = provider
                    .benchmark_cached_streaming_in_mode(
                        &fixture.prompt_ids,
                        1,
                        &sampling,
                        &fixture.snapshot,
                        Qwen4GenerationMode::Baseline,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .unwrap_or_else(|error| panic!("benchmark cached 64k prefill: {error:#}"));
                assert_eq!(output.cached_tokens, fixture.prefix_tokens);
                assert_eq!(output.prompt_tokens, fixture.prompt_ids.len());
                assert_eq!(output.token_ids, fixture.baseline_token_ids[..1]);
                assert!(stats.is_none());
                prefill_time += output.prefill_time;
                black_box(output);
            }
            prefill_time
        });
    });
    prefill_group.finish();
}

criterion_group!(
    benches,
    benchmark_fresh_decode,
    benchmark_fresh_prefill,
    benchmark_64k_cached_decode
);
criterion_main!(benches);
