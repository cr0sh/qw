mod support;

use std::env;
use std::hint::black_box;
#[cfg(feature = "dflash2")]
use std::path::Path;
use std::time::Duration;

use mlxcel_core::generate::SamplingConfig;
#[cfg(feature = "dflash2")]
use qw_runtime::Dflash2PrefixReuse;
#[cfg(feature = "specprefill")]
use qw_runtime::PrefillMode;
#[cfg(feature = "dflash2")]
use qw_runtime::PromptSnapshot;
use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{GenerationRequest, Qwen35Provider};
use support::{
    DECODE_MAX_TOKENS, LONG_CONTEXT_64K_MIN_TOKENS, LONG_CONTEXT_MIN_TOKENS,
    LongConversationFixture, prepare_decode_fixture, prepare_long_conversation_fixture,
    prompt_token_ids,
};

const TIMED_REPETITIONS: u64 = 3;

const FRESH_PREFILL: &str = "single_user_prefill/fresh_qwen";
const LONG_10K_PREFILL: &str = "single_user_prefill/long_10k_qwen";
const LONG_64K_PREFILL: &str = "single_user_prefill/long_64k_qwen";
#[cfg(feature = "dflash2")]
const FRESH_SPECULATIVE: &str = "single_user_decode/fresh_dflash2";
#[cfg(feature = "dflash2")]
const LONG_10K_SPECULATIVE: &str = "single_user_decode/long_10k_dflash2";
#[cfg(feature = "dflash2")]
const LONG_64K_SPECULATIVE: &str = "single_user_decode/long_64k_dflash2";
#[cfg(not(feature = "dflash2"))]
const FRESH_SPECULATIVE: &str = "single_user_decode/fresh_mtp_k3";
#[cfg(not(feature = "dflash2"))]
const LONG_10K_SPECULATIVE: &str = "single_user_decode/long_10k_mtp_k3";
#[cfg(not(feature = "dflash2"))]
const LONG_64K_SPECULATIVE: &str = "single_user_decode/long_64k_mtp_k3";

const BENCHMARK_LABELS: [&str; 6] = [
    FRESH_PREFILL,
    LONG_10K_PREFILL,
    LONG_64K_PREFILL,
    FRESH_SPECULATIVE,
    LONG_10K_SPECULATIVE,
    LONG_64K_SPECULATIVE,
];

struct Selection(Option<String>);

impl Selection {
    fn from_args() -> Self {
        let selected = env::args()
            .skip(1)
            .find(|argument| !argument.starts_with('-'));
        if let Some(selected) = &selected {
            assert!(
                BENCHMARK_LABELS.contains(&selected.as_str()),
                "unknown benchmark `{selected}`; expected one of {}",
                BENCHMARK_LABELS.join(", ")
            );
        }
        Self(selected)
    }

    fn includes(&self, label: &str) -> bool {
        self.0.as_deref().is_none_or(|selected| selected == label)
    }

    fn includes_either(&self, left: &str, right: &str) -> bool {
        self.includes(left) || self.includes(right)
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

fn measure(
    label: &str,
    context_tokens: usize,
    tokens_per_repetition: usize,
    mut sample: impl FnMut() -> Duration,
) {
    assert!(
        tokens_per_repetition > 0,
        "{label} must process at least one token"
    );
    let mut elapsed = Duration::ZERO;
    let mut elapsed_square_seconds = 0.0;
    for _ in 0..TIMED_REPETITIONS {
        let sample_elapsed = sample();
        assert!(
            !sample_elapsed.is_zero(),
            "{label} returned a zero-duration timed phase"
        );
        elapsed += sample_elapsed;
        elapsed_square_seconds += sample_elapsed.as_secs_f64().powi(2);
    }

    let repetitions = TIMED_REPETITIONS as f64;
    let mean_seconds = elapsed.as_secs_f64() / repetitions;
    let variance_seconds = (elapsed_square_seconds / repetitions - mean_seconds.powi(2)).max(0.0);
    let stddev_seconds = variance_seconds.sqrt();
    let processed_tokens = (tokens_per_repetition as u64)
        .checked_mul(TIMED_REPETITIONS)
        .expect("benchmark token count overflow");
    let tokens_per_second = processed_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "BENCHMARK_RESULT label={label} context_tokens={context_tokens} \
         tokens_per_repetition={tokens_per_repetition} processed_tokens={processed_tokens} \
         repetitions={TIMED_REPETITIONS} elapsed_seconds={:.6} mean_seconds={mean_seconds:.6} \
         stddev_seconds={stddev_seconds:.6} tokens_per_second={tokens_per_second:.3}",
        elapsed.as_secs_f64(),
    );
}

fn fresh_prefill_sample(
    provider: &mut Qwen35Provider,
    prompt_ids: &[i32],
    sampling: &SamplingConfig,
    expected_token_ids: &[i32],
) -> Duration {
    let output = provider
        .generate_baseline_streaming(
            prompt_ids,
            1,
            sampling,
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
        .expect("benchmark fresh target prefill");
    assert_eq!(
        output.token_ids, expected_token_ids,
        "deterministic fresh target prefill output changed"
    );
    let elapsed = output.prefill_time;
    black_box(output);
    elapsed
}

fn long_prefill_sample(
    provider: &mut Qwen35Provider,
    fixture: &LongConversationFixture,
    sampling: &SamplingConfig,
) -> Duration {
    let (output, stats) = provider
        .benchmark_cached_streaming_in_mode(
            &fixture.prompt_ids,
            1,
            sampling,
            &fixture.mtp_snapshot,
            Qwen35GenerationMode::Baseline,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("benchmark cached target prefill");
    assert_eq!(output.cached_tokens, fixture.prefix_tokens);
    assert_eq!(
        output.token_ids.as_slice(),
        &fixture.baseline_token_ids[..1],
        "deterministic cached target prefill output changed"
    );
    assert!(
        stats.is_none(),
        "target prefill unexpectedly returned speculative statistics"
    );
    let elapsed = output.prefill_time;
    black_box(output);
    elapsed
}

#[cfg(feature = "dflash2")]
fn fresh_dflash2_sample(
    provider: &mut Qwen35Provider,
    request: &GenerationRequest,
    draft_dir: &Path,
    expected_token_ids: &[i32],
) -> Duration {
    let (output, stats) = provider
        .generate_dflash2_streaming(request, draft_dir, |delta| {
            black_box(delta);
            true
        })
        .expect("benchmark fresh DFlash2 decode");
    assert_eq!(
        output.token_ids, expected_token_ids,
        "deterministic fresh DFlash2 output changed"
    );
    assert!(
        stats.proposed_draft_tokens > 0,
        "fresh DFlash2 must propose draft tokens"
    );
    let elapsed = stats.decode_time;
    black_box(output);
    elapsed
}

#[cfg(feature = "dflash2")]
fn long_dflash2_sample(
    provider: &mut Qwen35Provider,
    fixture: &LongConversationFixture,
    sampling: &SamplingConfig,
    draft_dir: &Path,
) -> Duration {
    let PromptSnapshot::Dflash2(snapshot) = &fixture.dflash2_snapshot else {
        panic!("long-context DFlash2 fixture has the wrong snapshot family");
    };
    let (output, stats, cached_tokens) = provider
        .generate_dflash2_cached_streaming(
            &fixture.prompt_ids,
            DECODE_MAX_TOKENS,
            sampling,
            draft_dir,
            Some(Dflash2PrefixReuse {
                snapshot,
                cached_tokens: fixture.prefix_tokens,
            }),
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("benchmark cached DFlash2 decode");
    assert_eq!(cached_tokens, fixture.prefix_tokens);
    assert_eq!(
        output.token_ids, fixture.dflash2_token_ids,
        "deterministic cached DFlash2 output changed"
    );
    assert!(
        stats.proposed_draft_tokens > 0,
        "cached DFlash2 must propose draft tokens"
    );
    let elapsed = stats.decode_time;
    black_box(output);
    elapsed
}

#[cfg(not(feature = "dflash2"))]
fn fresh_mtp_sample(
    provider: &mut Qwen35Provider,
    request: &GenerationRequest,
    expected_token_ids: &[i32],
) -> Duration {
    let (output, stats) = provider
        .generate_streaming_in_mode(request, Qwen35GenerationMode::Mtp, |delta| {
            black_box(delta);
            true
        })
        .expect("benchmark fresh bundled-MTP decode");
    assert_eq!(
        output.token_ids, expected_token_ids,
        "deterministic fresh bundled-MTP output changed"
    );
    let stats = stats.expect("explicit MTP mode must return MTP statistics");
    assert!(
        stats.proposed_draft_tokens > 0,
        "fresh bundled-MTP must propose draft tokens"
    );
    let elapsed = stats.decode_time;
    black_box(output);
    elapsed
}

#[cfg(not(feature = "dflash2"))]
fn long_mtp_sample(
    provider: &mut Qwen35Provider,
    fixture: &LongConversationFixture,
    sampling: &SamplingConfig,
) -> Duration {
    let (output, stats) = provider
        .benchmark_cached_streaming_in_mode(
            &fixture.prompt_ids,
            DECODE_MAX_TOKENS,
            sampling,
            &fixture.mtp_snapshot,
            Qwen35GenerationMode::Mtp,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("benchmark cached bundled-MTP decode");
    assert_eq!(output.cached_tokens, fixture.prefix_tokens);
    assert_eq!(
        output.token_ids, fixture.mtp_token_ids,
        "deterministic cached bundled-MTP output changed"
    );
    let stats = stats.expect("explicit MTP mode must return MTP statistics");
    assert!(
        stats.proposed_draft_tokens > 0,
        "cached bundled-MTP must propose draft tokens"
    );
    let elapsed = stats.decode_time;
    black_box(output);
    elapsed
}

fn main() {
    let selection = Selection::from_args();
    #[cfg(feature = "dflash2")]
    let speculative_engine = "dflash2";
    #[cfg(not(feature = "dflash2"))]
    let speculative_engine = "bundled_mtp_k3";
    println!(
        "BENCHMARK_CONFIG speculative_engine={speculative_engine} warmup_repetitions=1 \
         timed_repetitions={TIMED_REPETITIONS}"
    );

    let mut provider = support::load_provider();
    let fresh_prefill_prompt_ids = selection
        .includes(FRESH_PREFILL)
        .then(|| prompt_token_ids(&provider));
    let long_10k = selection
        .includes_either(LONG_10K_PREFILL, LONG_10K_SPECULATIVE)
        .then(|| prepare_long_conversation_fixture(&mut provider, "10k", LONG_CONTEXT_MIN_TOKENS));
    let long_64k = selection
        .includes_either(LONG_64K_PREFILL, LONG_64K_SPECULATIVE)
        .then(|| {
            prepare_long_conversation_fixture(&mut provider, "64k", LONG_CONTEXT_64K_MIN_TOKENS)
        });
    let fresh_decode = selection
        .includes(FRESH_SPECULATIVE)
        .then(|| prepare_decode_fixture(&mut provider));

    if let Some(prompt_ids) = fresh_prefill_prompt_ids {
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        let warmup = provider
            .generate_baseline_streaming(
                &prompt_ids,
                1,
                &sampling,
                None,
                None,
                &[],
                #[cfg(feature = "specprefill")]
                PrefillMode::Dense,
                |_| true,
            )
            .expect("warm up fresh target prefill");
        let expected_token_ids = warmup.token_ids.clone();
        black_box(warmup);
        measure(FRESH_PREFILL, 0, prompt_ids.len(), || {
            fresh_prefill_sample(&mut provider, &prompt_ids, &sampling, &expected_token_ids)
        });
    }

    if selection.includes(LONG_10K_PREFILL) {
        let fixture = long_10k.as_ref().expect("10k fixture was prepared");
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        black_box(long_prefill_sample(&mut provider, fixture, &sampling));
        measure(
            LONG_10K_PREFILL,
            fixture.prefix_tokens,
            fixture.new_prompt_tokens,
            || long_prefill_sample(&mut provider, fixture, &sampling),
        );
    }

    if selection.includes(LONG_64K_PREFILL) {
        let fixture = long_64k.as_ref().expect("64k fixture was prepared");
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        black_box(long_prefill_sample(&mut provider, fixture, &sampling));
        measure(
            LONG_64K_PREFILL,
            fixture.prefix_tokens,
            fixture.new_prompt_tokens,
            || long_prefill_sample(&mut provider, fixture, &sampling),
        );
    }

    #[cfg(feature = "dflash2")]
    {
        let draft_dir = support::draft_model_dir();
        if selection.includes(FRESH_SPECULATIVE) {
            let fixture = fresh_decode.as_ref().expect("fresh fixture was prepared");
            let (warmup, stats) = provider
                .generate_dflash2_streaming(&fixture.request, &draft_dir, |_| true)
                .expect("warm up fresh DFlash2 decode");
            let edit_distance = token_edit_distance(&warmup.token_ids, &fixture.baseline_token_ids);
            assert!(
                edit_distance * 20 <= fixture.baseline_token_ids.len(),
                "fresh DFlash2 token edit distance {edit_distance} exceeds 5% of the baseline"
            );
            assert!(
                stats.proposed_draft_tokens > 0,
                "fresh DFlash2 warmup must propose draft tokens"
            );
            let expected_token_ids = warmup.token_ids.clone();
            let decode_tokens = expected_token_ids.len().saturating_sub(1);
            black_box(warmup);
            measure(FRESH_SPECULATIVE, 0, decode_tokens, || {
                fresh_dflash2_sample(
                    &mut provider,
                    &fixture.request,
                    &draft_dir,
                    &expected_token_ids,
                )
            });
        }

        if selection.includes(LONG_10K_SPECULATIVE) {
            let fixture = long_10k.as_ref().expect("10k fixture was prepared");
            let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
            black_box(long_dflash2_sample(
                &mut provider,
                fixture,
                &sampling,
                &draft_dir,
            ));
            measure(
                LONG_10K_SPECULATIVE,
                fixture.prefix_tokens,
                fixture.dflash2_decode_tokens.saturating_sub(1),
                || long_dflash2_sample(&mut provider, fixture, &sampling, &draft_dir),
            );
        }

        if selection.includes(LONG_64K_SPECULATIVE) {
            let fixture = long_64k.as_ref().expect("64k fixture was prepared");
            let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
            black_box(long_dflash2_sample(
                &mut provider,
                fixture,
                &sampling,
                &draft_dir,
            ));
            measure(
                LONG_64K_SPECULATIVE,
                fixture.prefix_tokens,
                fixture.dflash2_decode_tokens.saturating_sub(1),
                || long_dflash2_sample(&mut provider, fixture, &sampling, &draft_dir),
            );
        }
    }

    #[cfg(not(feature = "dflash2"))]
    {
        if selection.includes(FRESH_SPECULATIVE) {
            let fixture = fresh_decode.as_ref().expect("fresh fixture was prepared");
            let edit_distance =
                token_edit_distance(&fixture.mtp_token_ids, &fixture.baseline_token_ids);
            assert!(
                edit_distance * 20 <= fixture.baseline_token_ids.len(),
                "fresh bundled-MTP token edit distance {edit_distance} exceeds 5% of the baseline"
            );
            black_box(fresh_mtp_sample(
                &mut provider,
                &fixture.request,
                &fixture.mtp_token_ids,
            ));
            measure(
                FRESH_SPECULATIVE,
                0,
                fixture.mtp_decode_tokens.saturating_sub(1),
                || fresh_mtp_sample(&mut provider, &fixture.request, &fixture.mtp_token_ids),
            );
        }

        if selection.includes(LONG_10K_SPECULATIVE) {
            let fixture = long_10k.as_ref().expect("10k fixture was prepared");
            let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
            black_box(long_mtp_sample(&mut provider, fixture, &sampling));
            measure(
                LONG_10K_SPECULATIVE,
                fixture.prefix_tokens,
                fixture.mtp_decode_tokens.saturating_sub(1),
                || long_mtp_sample(&mut provider, fixture, &sampling),
            );
        }

        if selection.includes(LONG_64K_SPECULATIVE) {
            let fixture = long_64k.as_ref().expect("64k fixture was prepared");
            let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
            black_box(long_mtp_sample(&mut provider, fixture, &sampling));
            measure(
                LONG_64K_SPECULATIVE,
                fixture.prefix_tokens,
                fixture.mtp_decode_tokens.saturating_sub(1),
                || long_mtp_sample(&mut provider, fixture, &sampling),
            );
        }
    }
}
