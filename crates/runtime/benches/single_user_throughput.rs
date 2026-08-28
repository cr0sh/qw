mod support;

use std::env;
use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qw_runtime::Qwen35Provider;
use qw_runtime::provider::Qwen35GenerationMode;
#[cfg(feature = "specprefill")]
use qw_runtime::{PrefillMode, SpecPrefillConfig};
#[cfg(feature = "dflash2")]
use qw_runtime::Dflash2PrefixReuse;
use support::{
    DECODE_MAX_TOKENS, LONG_CONTEXT_64K_MIN_TOKENS, LONG_CONTEXT_MIN_TOKENS,
    LongConversationFixture, MTP_BLOCK_SIZE, prepare_decode_fixture,
    prepare_long_conversation_fixture, prompt_token_ids,
};

const LONG_CONTEXT_ONLY_ENV: &str = "QW_BENCH_LONG_CONTEXT_ONLY";
const FRESH_PREFILL_ONLY_ENV: &str = "QW_BENCH_FRESH_PREFILL_ONLY";

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

fn token_fingerprint(tokens: &[i32]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    tokens.iter().fold(FNV_OFFSET_BASIS, |hash, token| {
        token.to_le_bytes().into_iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
    })
}

fn benchmark_long_conversation(
    criterion: &mut Criterion,
    provider: &mut Qwen35Provider,
    fixture: &LongConversationFixture,
) {
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let prefill_name = format!("long_{}_qwen", fixture.context_label);
    let baseline_name = format!("long_{}_baseline", fixture.context_label);
    let mtp_name = format!("long_{}_mtp_k3", fixture.context_label);

    let mtp_token_fingerprint = token_fingerprint(&fixture.mtp_token_ids);
    let mtp_token_edit_distance =
        token_edit_distance(&fixture.mtp_token_ids, &fixture.baseline_token_ids);
    assert!(
        mtp_token_edit_distance * 20 <= fixture.baseline_token_ids.len(),
        "{} MTP token edit distance {mtp_token_edit_distance} exceeds 5% of the baseline",
        fixture.context_label
    );
    eprintln!(
        "MTP_LONG_CONTEXT_CORRECTNESS context={} token_edit_distance={mtp_token_edit_distance} token_fingerprint={mtp_token_fingerprint}",
        fixture.context_label
    );

    {
        let mut group = criterion.benchmark_group("single_user_prefill");
        group.throughput(Throughput::Elements(fixture.new_prompt_tokens as u64));
        group.bench_function(prefill_name, |bencher| {
            bencher.iter(|| {
                let (output, stats) = provider
                    .benchmark_cached_streaming_in_mode(
                        &fixture.prompt_ids,
                        1,
                        &sampling,
                        &fixture.mtp_snapshot,
                        Qwen35GenerationMode::Baseline,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .expect("benchmark long-conversation single-user prefill");
                assert_eq!(output.cached_tokens, fixture.prefix_tokens);
                assert_eq!(
                    output.token_ids.as_slice(),
                    &fixture.baseline_token_ids[..1]
                );
                assert!(
                    stats.is_none(),
                    "long-conversation baseline mode returned MTP statistics"
                );
                black_box(output);
            });
        });
        group.finish();
    }

    {
        let mut group = criterion.benchmark_group("single_user_prefill");
        group.throughput(Throughput::Elements(fixture.new_prompt_tokens as u64));
        group.bench_function(&mtp_name, |bencher| {
            bencher.iter_custom(|iters| {
                let mut prefill_time = Duration::ZERO;
                for _ in 0..iters {
                    let (output, stats) = provider
                        .benchmark_cached_streaming_in_mode(
                            &fixture.prompt_ids,
                            DECODE_MAX_TOKENS,
                            &sampling,
                            &fixture.mtp_snapshot,
                            Qwen35GenerationMode::Mtp,
                            |delta| {
                                black_box(delta);
                                true
                            },
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "benchmark long-conversation MTP k={MTP_BLOCK_SIZE} prefill: {error:#}"
                            )
                        });
                    assert_eq!(output.cached_tokens, fixture.prefix_tokens);
                    assert_eq!(
                        &output.token_ids, &fixture.mtp_token_ids,
                        "deterministic long-conversation bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs changed"
                    );
                    assert_eq!(
                        token_fingerprint(&output.token_ids),
                        mtp_token_fingerprint,
                        "deterministic long-conversation bundled-MTP k={MTP_BLOCK_SIZE} greedy token fingerprint changed"
                    );
                    let stats = stats.expect("explicit MTP mode must return MTP statistics");
                    assert!(
                        stats.proposed_draft_tokens > 0,
                        "long-conversation MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
                    );
                    prefill_time += stats.prefill_time;
                    black_box(output);
                }
                prefill_time
            });
        });
        group.finish();
    }

    {
        let baseline_decode_tokens = fixture.baseline_token_ids.len().saturating_sub(1);
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(baseline_decode_tokens as u64));
        group.bench_function(baseline_name, |bencher| {
            bencher.iter_custom(|iters| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iters {
                    let (output, stats) = provider
                        .benchmark_cached_streaming_in_mode(
                            &fixture.prompt_ids,
                            DECODE_MAX_TOKENS,
                            &sampling,
                            &fixture.mtp_snapshot,
                            Qwen35GenerationMode::Baseline,
                            |delta| {
                                black_box(delta);
                                true
                            },
                        )
                        .expect("benchmark long-conversation controlled baseline");
                    assert_eq!(output.cached_tokens, fixture.prefix_tokens);
                    assert_eq!(&output.token_ids, &fixture.baseline_token_ids);
                    assert!(
                        stats.is_none(),
                        "long-conversation baseline mode returned MTP statistics"
                    );
                    decode_time += output.decode_time;
                    black_box(output);
                }
                decode_time
            });
        });
        group.finish();
    }

    {
        let mtp_decode_tokens =
            generation_elements(1, fixture.new_prompt_tokens, fixture.mtp_decode_tokens).decode;
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(mtp_decode_tokens));
        group.bench_function(mtp_name, |bencher| {
            bencher.iter_custom(|iters| {
                let mut decode_time = Duration::ZERO;
                for _ in 0..iters {
                    let (output, stats) = provider
                        .benchmark_cached_streaming_in_mode(
                            &fixture.prompt_ids,
                            DECODE_MAX_TOKENS,
                            &sampling,
                            &fixture.mtp_snapshot,
                            Qwen35GenerationMode::Mtp,
                            |delta| {
                                black_box(delta);
                                true
                            },
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "benchmark long-conversation MTP k={MTP_BLOCK_SIZE}: {error:#}"
                            )
                        });
                    assert_eq!(output.cached_tokens, fixture.prefix_tokens);
                    assert_eq!(
                        &output.token_ids, &fixture.mtp_token_ids,
                        "deterministic long-conversation bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs changed"
                    );
                    assert_eq!(
                        token_fingerprint(&output.token_ids),
                        mtp_token_fingerprint,
                        "deterministic long-conversation bundled-MTP k={MTP_BLOCK_SIZE} greedy token fingerprint changed"
                    );
                    let stats = stats.expect("explicit MTP mode must return MTP statistics");
                    assert!(
                        stats.proposed_draft_tokens > 0,
                        "long-conversation MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
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
#[cfg(feature = "dflash2")]
fn benchmark_long_dflash2(
    criterion: &mut Criterion,
    provider: &mut Qwen35Provider,
    fixture: &LongConversationFixture,
) {
    assert_eq!(fixture.context_label, "64k");
    assert!(fixture.prefix_tokens >= LONG_CONTEXT_64K_MIN_TOKENS);
    let qw_runtime::PromptSnapshot::Dflash2(snapshot) = &fixture.dflash2_snapshot else {
        panic!("long-conversation DFlash2 fixture has the wrong snapshot family");
    };
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let expected_fingerprint = token_fingerprint(&fixture.dflash2_token_ids);
    let edit_distance =
        token_edit_distance(&fixture.dflash2_token_ids, &fixture.baseline_token_ids);
    assert!(
        edit_distance * 20 <= fixture.baseline_token_ids.len(),
        "64k DFlash2 token edit distance {edit_distance} exceeds 5% of the baseline"
    );
    assert!(
        fixture.dflash2_proposed_draft_tokens > 0,
        "64k DFlash2 warmup must propose draft tokens"
    );
    let decode_tokens = fixture.dflash2_decode_tokens.saturating_sub(1);
    let draft_dir = support::draft_model_dir();
    let mut group = criterion.benchmark_group("single_user_decode");
    group.throughput(Throughput::Elements(decode_tokens as u64));
    group.bench_function("long_64k_dflash2", |bencher| {
        bencher.iter_custom(|iters| {
            let mut decode_time = Duration::ZERO;
            for _ in 0..iters {
                let (output, stats, cached_tokens) = provider
                    .generate_dflash2_cached_streaming(
                        &fixture.prompt_ids,
                        DECODE_MAX_TOKENS,
                        &sampling,
                        &draft_dir,
                        Some(Dflash2PrefixReuse {
                            snapshot,
                            cached_tokens: fixture.prefix_tokens,
                        }),
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .unwrap_or_else(|error| {
                        panic!("benchmark long-conversation DFlash2: {error:#}")
                    });
                assert_eq!(cached_tokens, fixture.prefix_tokens);
                assert_eq!(
                    output.token_ids, fixture.dflash2_token_ids,
                    "deterministic long-conversation DFlash2 greedy token IDs changed"
                );
                assert_eq!(
                    token_fingerprint(&output.token_ids),
                    expected_fingerprint,
                    "deterministic long-conversation DFlash2 token fingerprint changed"
                );
                assert!(
                    stats.proposed_draft_tokens > 0,
                    "long-conversation DFlash2 must propose draft tokens"
                );
                assert!(
                    token_edit_distance(&output.token_ids, &fixture.baseline_token_ids) * 20
                        <= fixture.baseline_token_ids.len(),
                    "long-conversation DFlash2 token edit distance exceeds 5% of the baseline"
                );
                decode_time += stats.decode_time;
                black_box(output);
            }
            decode_time
        });
    });
    group.finish();
}

const FRESH_MTP_BENCHMARK: &str = "single_user_decode/fresh_mtp_k3";

fn is_exclusive_fresh_mtp_selection() -> bool {
    env::args().skip(1).any(|arg| arg == FRESH_MTP_BENCHMARK)
}

fn benchmark_fresh_mtp(
    criterion: &mut Criterion,
    provider: &mut Qwen35Provider,
    decode_request: &qw_runtime::GenerationRequest,
    mtp_token_ids: &[i32],
    mtp_decode_tokens: usize,
) {
    let mut group = criterion.benchmark_group("single_user_decode");
    group.throughput(Throughput::Elements(mtp_decode_tokens as u64));
    // Report tokens per second against the MTP decode phase only. The
    // generation call still executes prompt preparation and prefill.
    group.bench_function("fresh_mtp_k3", |bencher| {
        bencher.iter_custom(|iters| {
            let mut decode_time = Duration::ZERO;
            for _ in 0..iters {
                let (output, stats) = provider
                    .generate_streaming_in_mode(
                        decode_request,
                        Qwen35GenerationMode::Mtp,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .unwrap_or_else(|error| panic!("benchmark MTP k={MTP_BLOCK_SIZE}: {error:#}"));
                assert_eq!(
                    &output.token_ids, mtp_token_ids,
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

#[cfg(feature = "dflash2")]
const LONG_DFLASH2_BENCHMARK: &str = "single_user_decode/long_64k_dflash2";

#[cfg(feature = "dflash2")]
fn is_exclusive_long_dflash2_selection() -> bool {
    env::args()
        .skip(1)
        .any(|arg| arg == LONG_DFLASH2_BENCHMARK)
}

#[cfg(feature = "dflash2")]
const FRESH_DFLASH2_BENCHMARK: &str = "single_user_decode/fresh_dflash2";

#[cfg(feature = "dflash2")]
fn is_exclusive_fresh_dflash2_selection() -> bool {
    env::args()
        .skip(1)
        .any(|arg| arg == FRESH_DFLASH2_BENCHMARK)
}

#[cfg(feature = "dflash2")]
fn benchmark_fresh_dflash2(
    criterion: &mut Criterion,
    provider: &mut Qwen35Provider,
    decode_request: &qw_runtime::GenerationRequest,
    baseline_token_ids: &[i32],
) {
    let draft_dir = support::draft_model_dir();
    let (dflash2_output, dflash2_stats) = provider
        .generate_dflash2_streaming(decode_request, &draft_dir, |delta| {
            black_box(delta);
            true
        })
        .unwrap_or_else(|error| panic!("warm up DFlash2 single-user decode: {error:#}"));
    let dflash2_token_edit_distance =
        token_edit_distance(&dflash2_output.token_ids, baseline_token_ids);
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
    let dflash2_decode_tokens = dflash2_output.token_ids.len().saturating_sub(1);

    let mut group = criterion.benchmark_group("single_user_decode");
    group.throughput(Throughput::Elements(dflash2_decode_tokens as u64));
    group.bench_function("fresh_dflash2", |bencher| {
        bencher.iter_custom(|iters| {
            let mut decode_time = Duration::ZERO;
            for _ in 0..iters {
                let (output, stats) = provider
                    .generate_dflash2_streaming(decode_request, &draft_dir, |delta| {
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

fn single_user_throughput(criterion: &mut Criterion) {
    let long_context_only = match env::var(LONG_CONTEXT_ONLY_ENV) {
        Ok(value) if value == "64k" => true,
        Ok(value) => panic!("{LONG_CONTEXT_ONLY_ENV} must be `64k` when set, got `{value}`"),
        Err(env::VarError::NotPresent) => false,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("{LONG_CONTEXT_ONLY_ENV} must contain valid Unicode")
        }
    };
    let fresh_prefill_only = match env::var(FRESH_PREFILL_ONLY_ENV) {
        Ok(value) if value == "1" => true,
        Ok(value) => panic!("{FRESH_PREFILL_ONLY_ENV} must be `1` when set, got `{value}`"),
        Err(env::VarError::NotPresent) => false,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("{FRESH_PREFILL_ONLY_ENV} must contain valid Unicode")
        }
    };
    assert!(
        !(long_context_only && fresh_prefill_only),
        "{LONG_CONTEXT_ONLY_ENV} and {FRESH_PREFILL_ONLY_ENV} cannot both be set"
    );
    let mut provider = support::load_provider();
    if is_exclusive_fresh_mtp_selection() {
        let decode_fixture = prepare_decode_fixture(&mut provider);
        benchmark_fresh_mtp(
            criterion,
            &mut provider,
            &decode_fixture.request,
            &decode_fixture.mtp_token_ids,
            decode_fixture.mtp_decode_tokens,
        );
        return;
    }

    #[cfg(feature = "dflash2")]
    if is_exclusive_fresh_dflash2_selection() {
        let decode_fixture = prepare_decode_fixture(&mut provider);
        benchmark_fresh_dflash2(
            criterion,
            &mut provider,
            &decode_fixture.request,
            &decode_fixture.baseline_token_ids,
        );
        return;
    }
    #[cfg(feature = "dflash2")]
    if is_exclusive_long_dflash2_selection() {
        let long_64k =
            prepare_long_conversation_fixture(&mut provider, "64k", LONG_CONTEXT_64K_MIN_TOKENS);
        benchmark_long_dflash2(criterion, &mut provider, &long_64k);
        return;
    }

    if long_context_only {
        let long_64k =
            prepare_long_conversation_fixture(&mut provider, "64k", LONG_CONTEXT_64K_MIN_TOKENS);
        assert!(long_64k.prefix_tokens >= LONG_CONTEXT_64K_MIN_TOKENS);
        benchmark_long_conversation(criterion, &mut provider, &long_64k);
        #[cfg(feature = "dflash2")]
        benchmark_long_dflash2(criterion, &mut provider, &long_64k);
        return;
    }
    let prefill_prompt_ids = prompt_token_ids(&provider);
    let prompt_tokens = prefill_prompt_ids.len();
    let prefill_elements = generation_elements(1, prompt_tokens, 1).prefill;
    let prefill_sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let dense_warmup = provider
        .generate_baseline_streaming(
            &prefill_prompt_ids,
            1,
            &prefill_sampling,
            None,
            None,
            &[],
            #[cfg(feature = "specprefill")]
            PrefillMode::Dense,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm up dense single-user prefill");
    assert!(!dense_warmup.token_ids.is_empty());
    #[cfg(feature = "specprefill")]
    assert!(
        dense_warmup.specprefill_stats.is_none(),
        "dense prefill warmup returned SpecPrefill statistics"
    );
    let dense_token_ids = dense_warmup.token_ids;
    #[cfg(feature = "specprefill")]
    let specprefill_config = SpecPrefillConfig {
        min_tokens: 1,
        keep_rate: 0.25,
        protected_prefix_tokens: 0,
        ..Default::default()
    };
    #[cfg(feature = "specprefill")]
    let specprefill_warmup = provider
        .generate_baseline_streaming(
            &prefill_prompt_ids,
            1,
            &prefill_sampling,
            None,
            None,
            &[],
            PrefillMode::SpecPrefill(specprefill_config),
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm up single-user SpecPrefill");
    #[cfg(feature = "specprefill")]
    let specprefill_token_ids = specprefill_warmup.token_ids.clone();
    #[cfg(feature = "specprefill")]
    assert_eq!(
        specprefill_token_ids, dense_token_ids,
        "SpecPrefill changed the deterministic greedy output token"
    );
    #[cfg(feature = "specprefill")]
    let specprefill_stats = specprefill_warmup
        .specprefill_stats
        .as_ref()
        .expect("SpecPrefill warmup must activate sparse admission");
    #[cfg(feature = "specprefill")]
    assert!(specprefill_stats.eligible_target_tokens > specprefill_config.min_tokens);
    #[cfg(feature = "specprefill")]
    assert!(specprefill_stats.selected_target_tokens > 0);
    #[cfg(feature = "specprefill")]
    assert!(specprefill_stats.selected_target_tokens < specprefill_stats.eligible_target_tokens);
    #[cfg(feature = "specprefill")]
    assert!(
        specprefill_stats.selected_target_tokens * 2 <= specprefill_stats.eligible_target_tokens,
        "SpecPrefill selected {} of {} eligible target tokens",
        specprefill_stats.selected_target_tokens,
        specprefill_stats.eligible_target_tokens
    );
    #[cfg(feature = "specprefill")]
    black_box(specprefill_warmup);

    {
        let mut group = criterion.benchmark_group("single_user_prefill");
        group.throughput(Throughput::Elements(prefill_elements));
        group.bench_function("fresh_qwen", |bencher| {
            bencher.iter(|| {
                let output = provider
                    .generate_baseline_streaming(
                        &prefill_prompt_ids,
                        1,
                        &prefill_sampling,
                        None,
                        None,
                        &[],
                        #[cfg(feature = "specprefill")]
                        PrefillMode::Dense,
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .expect("benchmark dense single-user prefill");
                assert_eq!(output.token_ids, dense_token_ids);
                #[cfg(feature = "specprefill")]
                assert!(
                    output.specprefill_stats.is_none(),
                    "dense prefill benchmark returned SpecPrefill statistics"
                );
                black_box(output);
            });
        });
        #[cfg(feature = "specprefill")]
        group.bench_function("specprefill", |bencher| {
            bencher.iter(|| {
                let output = provider
                    .generate_baseline_streaming(
                        &prefill_prompt_ids,
                        1,
                        &prefill_sampling,
                        None,
                        None,
                        &[],
                        PrefillMode::SpecPrefill(specprefill_config),
                        |delta| {
                            black_box(delta);
                            true
                        },
                    )
                    .expect("benchmark single-user SpecPrefill");
                assert_eq!(output.token_ids, specprefill_token_ids);
                let stats = output
                    .specprefill_stats
                    .as_ref()
                    .expect("SpecPrefill benchmark must activate sparse admission");
                assert!(stats.eligible_target_tokens > specprefill_config.min_tokens);
                assert!(stats.selected_target_tokens > 0);
                assert!(stats.selected_target_tokens < stats.eligible_target_tokens);
                assert!(
                    stats.selected_target_tokens * 2 <= stats.eligible_target_tokens,
                    "SpecPrefill selected {} of {} eligible target tokens",
                    stats.selected_target_tokens,
                    stats.eligible_target_tokens
                );
                black_box(output);
            });
        });
        group.finish();
    }
    if fresh_prefill_only {
        return;
    }

    let long_10k = prepare_long_conversation_fixture(&mut provider, "10k", LONG_CONTEXT_MIN_TOKENS);
    assert!(long_10k.prefix_tokens >= LONG_CONTEXT_MIN_TOKENS);
    let long_64k =
        prepare_long_conversation_fixture(&mut provider, "64k", LONG_CONTEXT_64K_MIN_TOKENS);
    assert!(long_64k.prefix_tokens >= LONG_CONTEXT_64K_MIN_TOKENS);

    let decode_fixture = prepare_decode_fixture(&mut provider);
    let decode_request = decode_fixture.request;
    let baseline_token_ids = decode_fixture.baseline_token_ids;
    let mtp_token_ids = decode_fixture.mtp_token_ids;
    let mtp_token_edit_distance = token_edit_distance(&mtp_token_ids, &baseline_token_ids);
    assert!(
        mtp_token_edit_distance * 20 <= baseline_token_ids.len(),
        "MTP token edit distance {mtp_token_edit_distance} exceeds 5% of the baseline"
    );
    eprintln!("MTP_CORRECTNESS token_edit_distance={mtp_token_edit_distance}");
    let mtp_decode_tokens =
        generation_elements(1, prompt_tokens, decode_fixture.mtp_decode_tokens).decode as usize;

    {
        let baseline_decode_tokens = baseline_token_ids.len().saturating_sub(1);
        let mut group = criterion.benchmark_group("single_user_decode");
        group.throughput(Throughput::Elements(baseline_decode_tokens as u64));
        group.bench_function("fresh_baseline", |bencher| {
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

    benchmark_fresh_mtp(
        criterion,
        &mut provider,
        &decode_request,
        &mtp_token_ids,
        mtp_decode_tokens,
    );

    benchmark_long_conversation(criterion, &mut provider, &long_10k);
    benchmark_long_conversation(criterion, &mut provider, &long_64k);
    #[cfg(feature = "dflash2")]
    benchmark_long_dflash2(criterion, &mut provider, &long_64k);

    #[cfg(feature = "dflash2")]
    benchmark_fresh_dflash2(
        criterion,
        &mut provider,
        &decode_request,
        &baseline_token_ids,
    );
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
