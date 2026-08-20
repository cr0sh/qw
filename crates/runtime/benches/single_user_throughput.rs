mod support;

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qw_runtime::provider::Qwen35GenerationMode;
use support::{MTP_BLOCK_SIZE, prepare_decode_fixture, prompt_tokens};

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
        // True decode begins after the first token sampled from prefill logits.
        decode: (batch_size * completion_tokens.saturating_sub(1)) as u64,
    }
}

fn token_edit_distance(left: &[i32], right: &[i32]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_token) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_token) in right.iter().enumerate() {
            current[right_index + 1] = if left_token == right_token {
                previous[right_index]
            } else {
                1 + previous[right_index]
                    .min(current[right_index])
                    .min(previous[right_index + 1])
            };
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

fn single_user_throughput(criterion: &mut Criterion) {
    let mut provider = support::load_provider();
    let prompt_tokens = prompt_tokens(&provider);

    let prefill_request = support::request(1);
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

    let decode_fixture = prepare_decode_fixture(&mut provider);
    let decode_request = decode_fixture.request;
    let baseline_token_ids = decode_fixture.baseline_token_ids;
    let mtp_token_ids = decode_fixture.mtp_token_ids;
    let mtp_decode_tokens = generation_elements(1, prompt_tokens, decode_fixture.mtp_decode_tokens)
        .decode as usize;

    {
        let baseline_decode_tokens = baseline_token_ids.len().saturating_sub(1);
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(baseline_decode_tokens as u64));
        group.bench_function("baseline", |bencher| {
            bencher.iter_custom(|iters| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iters {
                    let (output, measured_decode, stats) = provider
                        .benchmark_streaming_in_mode(
                            &decode_request,
                            Qwen35GenerationMode::Baseline,
                            |delta| {
                                black_box(delta);
                                true
                            },
                        )
                        .expect("benchmark controlled non-MTP baseline");
                    assert_eq!(&output.token_ids, &baseline_token_ids);
                    assert!(stats.is_none(), "baseline mode returned MTP statistics");
                    decode_time += measured_decode;
                    black_box(output);
                }
                decode_time
            });
        });
        group.finish();
    }

    {
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(mtp_decode_tokens as u64));
        // Report tokens per second against the MTP decode phase only. The
        // generation call still executes prompt preparation and prefill.
        group.bench_function("mtp_k3", |bencher| {
            bencher.iter_custom(|iters| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iters {
                    let (output, stats) = provider
                        .generate_streaming_in_mode(
                            &decode_request,
                            Qwen35GenerationMode::Mtp,
                            |delta| {
                                black_box(delta);
                                true
                            },
                        )
                        .unwrap_or_else(|error| {
                            panic!("benchmark MTP k={MTP_BLOCK_SIZE}: {error:#}")
                        });
                    assert_eq!(
                        &output.token_ids, &mtp_token_ids,
                        "deterministic bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs changed"
                    );
                    decode_time += stats
                        .expect("explicit MTP mode must return MTP statistics")
                        .decode_time;
                    black_box(output);
                }
                decode_time
            });
        });
        group.finish();
    }

    // DFlash2 block-diffusion drafter, same deterministic decode fixture.
    // Keep token-level edit distance within 5% of the baseline so model-
    // invasive performance work cannot silently introduce a major regression.
    {
        let draft_dir = support::draft_model_dir();
        let (dflash2_output, dflash2_stats) = provider
            .generate_dflash2_streaming(&decode_request, &draft_dir, |delta| {
                black_box(delta);
                true
            })
            .unwrap_or_else(|error| panic!("warm up DFlash2 single-user decode: {error:#}"));
        let dflash2_token_edit_distance =
            token_edit_distance(&dflash2_output.token_ids, &baseline_token_ids);
        assert!(
            dflash2_token_edit_distance * 20 <= baseline_token_ids.len(),
            "DFlash2 token edit distance {dflash2_token_edit_distance} exceeds 5% of the baseline"
        );
        assert!(
            dflash2_stats.proposed_draft_tokens > 0,
            "DFlash2 must propose draft tokens"
        );
        eprintln!(
            "DFLASH2_PROFILE tokens={} token_edit_distance={} accepted={} proposed={} acceptance={:.2}% forwards={} draft_ms={:.3} verify_ms={:.3} walk_ms={:.3} reconcile_ms={:.3}",
            dflash2_output.token_ids.len(),
            dflash2_token_edit_distance,
            dflash2_stats.accepted_draft_tokens,
            dflash2_stats.proposed_draft_tokens,
            dflash2_stats.acceptance_percentage(),
            dflash2_stats.target_forward_calls,
            dflash2_stats.draft_time.as_secs_f64() * 1_000.0,
            dflash2_stats.target_verify_time.as_secs_f64() * 1_000.0,
            dflash2_stats.walk_time.as_secs_f64() * 1_000.0,
            dflash2_stats.reconcile_time.as_secs_f64() * 1_000.0,
        );
        let dflash2_decode_tokens =
            generation_elements(1, prompt_tokens, dflash2_output.token_ids.len()).decode as usize;

        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(dflash2_decode_tokens as u64));
        group.bench_function("dflash2", |bencher| {
            bencher.iter_custom(|iters| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iters {
                    let (output, stats) = provider
                        .generate_dflash2_streaming(&decode_request, &draft_dir, |delta| {
                            black_box(delta);
                            true
                        })
                        .unwrap_or_else(|error| {
                            panic!("benchmark DFlash2 single-user decode: {error:#}")
                        });
                    assert_eq!(
                        output.token_ids, dflash2_output.token_ids,
                        "deterministic DFlash2 greedy token IDs changed"
                    );
                    decode_time += stats.decode_time;
                    black_box(output);
                }
                decode_time
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
