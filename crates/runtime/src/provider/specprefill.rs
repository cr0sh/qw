use super::*;
use crate::specprefill::{
    SpecPrefillConfig, admitted_cached_tokens, score_tokens, select_target_indices,
};
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::loop_detection::detect_repetition_loop;
use mlxcel_core::sampling::{
    SamplerState, sample_token_optimized, sample_token_optimized_with_state,
};
use std::time::Instant;

pub(super) struct SparsePlan {
    dense_end: usize,
    config: SpecPrefillConfig,
}

struct SparseGeneration {
    token_ids: Vec<i32>,
    stop_reason: GenerationStopReason,
    cached_tokens: usize,
    prefill_time: Duration,
    decode_time: Duration,
    stats: SpecPrefillStats,
}

fn generate_specprefill_tokens<F: FnMut(i32) -> bool>(
    provider: &Qwen35Provider,
    prompt_ids: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    prefix_reuse: Option<PrefixReuse<'_>>,
    plan: SparsePlan,
    on_token: F,
) -> Result<SparseGeneration> {
    let model = &provider.model;
    let draft = provider
        .specprefill_draft
        .as_ref()
        .expect("validated SpecPrefill draft");
    let dense_end = plan.dense_end;
    let config = plan.config;
    let eligible = &prompt_ids[dense_end..];

    let scoring_start = Instant::now();
    let importance = score_tokens(draft, eligible)?;
    let draft_scoring_time = scoring_start.elapsed();
    let selected = select_target_indices(&importance, dense_end, prompt_ids.len(), config);

    let target_start = Instant::now();
    let (logits, cached_tokens) = model
        .specprefill_sparse_prefill(prompt_ids, prefix_reuse, dense_end, &selected)
        .map_err(anyhow::Error::msg)
        .context("sparse target prefill failed")?;
    mlxcel_core::eval(&logits);
    let target_prefill_time = target_start.elapsed();

    let (generated, stop_reason, decode_time) =
        decode_after_sparse_prefill(model, prompt_ids, logits, max_tokens, sampling, on_token);
    let stats = SpecPrefillStats {
        draft_tokens: eligible.len(),
        eligible_target_tokens: eligible.len(),
        selected_target_tokens: selected.len(),
        protected_target_tokens: dense_end.saturating_sub(cached_tokens),
        cached_target_tokens: cached_tokens,
        draft_scoring_time,
        target_prefill_time,
    };
    Ok(SparseGeneration {
        token_ids: generated,
        stop_reason,
        cached_tokens,
        prefill_time: target_prefill_time,
        decode_time,
        stats,
    })
}

fn decode_after_sparse_prefill<F: FnMut(i32) -> bool>(
    model: &Qwen35Model,
    prompt_ids: &[i32],
    mut logits: UniquePtr<MlxArray>,
    max_tokens: usize,
    sampling: &SamplingConfig,
    mut on_token: F,
) -> (Vec<i32>, GenerationStopReason, Duration) {
    let mut effective_sampling = sampling.clone();
    effective_sampling
        .prompt_token_count
        .get_or_insert(prompt_ids.len());
    effective_sampling
        .token_bias
        .suppress_tokens(&model.output_suppressed_token_ids());
    seed_rng_if_needed(&effective_sampling);
    let eos_tokens =
        merged_eos_token_ids(model.eos_token_ids(), &effective_sampling.stop_token_ids);
    let needs_history = effective_sampling.needs_token_history();
    let mut token_history = initial_token_history(prompt_ids, needs_history);
    let mut sampler_state: Option<SamplerState> = None;
    let mut generated = Vec::with_capacity(max_tokens);
    let mut stop_reason = GenerationStopReason::MaxTokens;
    let decode_start = Instant::now();

    while generated.len() < max_tokens {
        let (token, _) = if needs_history {
            sample_token_optimized_with_state(
                &logits,
                &effective_sampling,
                &token_history,
                &mut sampler_state,
            )
        } else {
            sample_token_optimized(&logits, &effective_sampling, &token_history)
        };
        mlxcel_core::eval(&token);
        let token_id = mlxcel_core::item_i32(&token);
        if eos_tokens.contains(&token_id) {
            stop_reason = GenerationStopReason::Eos;
            break;
        }
        generated.push(token_id);
        if needs_history {
            token_history.push(token_id);
        }
        if !on_token(token_id) {
            stop_reason = GenerationStopReason::CallbackCancelled;
            break;
        }
        if detect_repetition_loop(&generated, &effective_sampling.loop_detection) {
            stop_reason = GenerationStopReason::RepetitionLoop;
            break;
        }
        if generated.len() == max_tokens {
            break;
        }
        logits = model.specprefill_decode(token_id, prompt_ids.len() + generated.len() - 1);
    }
    let decode_time = decode_start.elapsed();
    (generated, stop_reason, decode_time)
}

impl Qwen35Provider {
    pub fn load_with_specprefill_draft(
        model_dir: impl AsRef<Path>,
        draft_model_dir: impl AsRef<Path>,
        kv_cache_mode: KVCacheMode,
    ) -> Result<Self> {
        let mut provider = Self::load_target_only(model_dir.as_ref(), kv_cache_mode)?;
        let draft_model_dir = draft_model_dir.as_ref();
        let draft_tokenizer_path = draft_model_dir.join("tokenizer.json");
        ensure!(
            draft_tokenizer_path.is_file(),
            "missing SpecPrefill draft tokenizer {}",
            draft_tokenizer_path.display()
        );
        let draft_tokenizer = Tokenizer::from_file(&draft_tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to load SpecPrefill draft tokenizer {}",
                    draft_tokenizer_path.display()
                )
            })?;
        ensure!(
            provider.tokenizer.get_vocab(true) == draft_tokenizer.get_vocab(true),
            "target and SpecPrefill draft tokenizer vocabularies are incompatible"
        );
        provider.specprefill_draft = Some(Qwen35Model::load_specprefill_draft(draft_model_dir)?);
        Ok(provider)
    }

    #[cfg(feature = "specprefill")]
    pub(super) fn load_with_required_draft(
        model_dir: impl AsRef<Path>,
        kv_cache_mode: KVCacheMode,
    ) -> Result<Self> {
        let draft_model_dir = crate::resolve_specprefill_draft_path(None)?;
        Self::load_with_specprefill_draft(model_dir, &draft_model_dir, kv_cache_mode).with_context(
            || {
                format!(
                    "failed to load required SpecPrefill draft at {}; rerun `qw download {}`",
                    draft_model_dir.display(),
                    crate::DEFAULT_MODEL_IDENTIFIER
                )
            },
        )
    }

    pub(super) fn plan_specprefill(
        &self,
        prompt_ids: &[i32],
        max_tokens: usize,
        constrained: bool,
        prefix_reuse: Option<&PrefixReuse<'_>>,
        config: SpecPrefillConfig,
    ) -> Result<Option<SparsePlan>> {
        config.validate(prompt_ids.len())?;
        ensure!(
            !constrained,
            "SpecPrefill does not support token constraints"
        );
        ensure!(
            !prompt_ids.is_empty(),
            "prompt token sequence must not be empty"
        );
        ensure!(max_tokens > 0, "max_tokens must be greater than zero");
        self.specprefill_draft
            .as_ref()
            .context("SpecPrefill capability is unavailable because no draft model is loaded")?;
        let dense_end =
            config.dense_prefix_end(admitted_cached_tokens(prefix_reuse, prompt_ids.len()));
        Ok(config
            .should_activate(prompt_ids.len() - dense_end)
            .then_some(SparsePlan { dense_end, config }))
    }

    pub(super) fn generate_specprefill<F: FnMut(&str) -> bool>(
        &self,
        plan: SparsePlan,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        prefix_reuse: Option<PrefixReuse<'_>>,
        mut on_delta: F,
    ) -> Result<BaselineGeneration> {
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut callback_active = true;
        let sparse = generate_specprefill_tokens(
            self,
            prompt_ids,
            max_tokens,
            sampling,
            prefix_reuse,
            plan,
            |token_id| match decoder.push(token_id) {
                Ok(delta) => {
                    if delta.is_empty() {
                        true
                    } else {
                        callback_active = on_delta(&delta);
                        callback_active
                    }
                }
                Err(error) => {
                    decode_error = Some(error);
                    false
                }
            },
        )?;
        if let Some(error) = decode_error {
            return Err(error);
        }
        let final_delta = decoder.finish()?;
        if callback_active && !final_delta.is_empty() {
            let _ = on_delta(&final_delta);
        }
        let completion_tokens = sparse.token_ids.len();
        debug!(
            route = "specprefill",
            draft_scoring_ms = sparse.stats.draft_scoring_time.as_secs_f64() * 1_000.0,
            target_prefill_ms = sparse.stats.target_prefill_time.as_secs_f64() * 1_000.0,
            selected_tokens = sparse.stats.selected_target_tokens,
            eligible_tokens = sparse.stats.eligible_target_tokens,
            "completed sparse prefill generation"
        );
        log_generation_metrics(
            "specprefill",
            prompt_ids.len(),
            completion_tokens,
            sparse.cached_tokens,
            sparse.prefill_time,
            sparse.decode_time,
        );
        Ok(BaselineGeneration {
            text: decoder.emitted,
            token_ids: sparse.token_ids,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
            cached_tokens: sparse.cached_tokens,
            finish_outcome: sparse.stop_reason,
            prompt_snapshots: Vec::new(),
            final_snapshot: None,
            prefill_time: sparse.prefill_time,
            decode_time: sparse.decode_time,
            specprefill_stats: Some(sparse.stats),
        })
    }
}
