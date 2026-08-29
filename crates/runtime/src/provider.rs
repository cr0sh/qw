use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::{
    ControlledGeneration, CxxGenerator, GenerationStopReason, LanguageModel, ModelStateSnapshot,
    PrefixReuse, SamplingConfig, TokenConstraint,
};
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tracing::debug;

use crate::chat_template::ChatTemplateProcessor;
pub use crate::chat_template::{
    ChatContentPart, ChatContentRef, ChatCustomToolCall, ChatFile, ChatImageUrl, ChatInputAudio,
    ChatMessage, ChatMessageContent, ChatPromptCacheBreakpoint, ChatTool, ChatToolCall,
    ChatToolCallFunction, ChatToolFunction,
};
use crate::qwen4::Qwen4Model;
pub use crate::qwen4_mtp::{MtpGenerationStats, MtpPrefixReuse, MtpPromptSnapshot};
use crate::qwen4_mtp::{Qwen4MtpDraftModel, Qwen4MtpGenerator};

const DEFAULT_MTP_BLOCK_SIZE: usize = 3;

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

fn log_generation_metrics(
    route: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill_time: Duration,
    decode_time: Duration,
) {
    let prefill_tokens = prompt_tokens.saturating_sub(cached_tokens);
    debug!(
        phase = "generation.prefill",
        route,
        prompt_tokens,
        cached_tokens,
        prefill_tokens,
        elapsed_ms = prefill_time.as_secs_f64() * 1_000.0,
        tokens_per_second = tokens_per_second(prefill_tokens, prefill_time),
    );
    debug!(
        phase = "generation.decode",
        route,
        completion_tokens,
        elapsed_ms = decode_time.as_secs_f64() * 1_000.0,
        tokens_per_second = tokens_per_second(completion_tokens, decode_time),
    );
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<i32>,
}

pub enum PromptSnapshot {
    Baseline(ModelStateSnapshot),
    Mtp(MtpPromptSnapshot),
}

impl PromptSnapshot {
    pub fn token_len(&self) -> usize {
        match self {
            Self::Baseline(snapshot) => snapshot.token_len(),
            Self::Mtp(snapshot) => snapshot.token_len(),
        }
    }
}

pub struct BaselineGeneration {
    pub text: String,
    pub token_ids: Vec<i32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub finish_outcome: GenerationStopReason,
    pub prompt_snapshots: Vec<PromptSnapshot>,
    pub final_snapshot: Option<PromptSnapshot>,
    /// Wall time spent processing uncached prompt tokens.
    pub prefill_time: Duration,
    /// Wall time spent sampling and forwarding generated tokens.
    pub decode_time: Duration,
}
type QwenDecodeStream<'a> = tokenizers::DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

struct IncrementalTextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    stream: QwenDecodeStream<'a>,
    token_ids: Vec<u32>,
    emitted: String,
}

impl<'a> IncrementalTextDecoder<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            stream: tokenizer.decode_stream(false),
            tokenizer,
            token_ids: Vec::new(),
            emitted: String::new(),
        }
    }

    fn push(&mut self, token_id: i32) -> Result<String> {
        let token_id = u32::try_from(token_id).context("generated a negative token identifier")?;
        self.token_ids.push(token_id);
        let delta = self
            .stream
            .step(token_id)
            .map_err(anyhow::Error::msg)
            .context("failed to incrementally decode generated token")?
            .unwrap_or_default();
        self.emitted.push_str(&delta);
        Ok(delta)
    }

    fn finish(&mut self) -> Result<String> {
        let decoded = self
            .tokenizer
            .decode(&self.token_ids, false)
            .map_err(anyhow::Error::msg)
            .context("failed to decode generated tokens")?;
        self.advance(decoded, true)
    }

    fn advance(&mut self, decoded: String, final_chunk: bool) -> Result<String> {
        advance_decoded_text(&mut self.emitted, &decoded, final_chunk)
    }
}

fn advance_decoded_text(emitted: &mut String, decoded: &str, final_chunk: bool) -> Result<String> {
    ensure!(
        decoded.starts_with(emitted.as_str()),
        "incremental tokenizer decoding changed text already emitted"
    );
    let remaining = &decoded[emitted.len()..];
    if final_chunk {
        ensure!(
            !remaining.contains('\u{fffd}'),
            "generated token sequence ended with incomplete UTF-8"
        );
    }
    let stable_len = remaining.find('\u{fffd}').unwrap_or(remaining.len());
    let delta = remaining[..stable_len].to_string();
    emitted.push_str(&delta);
    Ok(delta)
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen4GenerationMode {
    Automatic,
    Baseline,
    Mtp,
}

pub struct Qwen4Provider {
    model: Qwen4Model,
    tokenizer: Tokenizer,
    chat_template: ChatTemplateProcessor,
    defaults: GenerationDefaults,
    generator: CxxGenerator,
    mtp_generator: Option<Qwen4MtpGenerator>,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(i32),
    Many(Vec<i32>),
}

struct GenerationDefaults {
    stop_token_ids: Vec<i32>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
}

impl Qwen4Provider {
    pub fn load(model_dir: impl AsRef<Path>, kv_cache_mode: KVCacheMode) -> Result<Self> {
        Self::load_target_only(model_dir.as_ref(), kv_cache_mode)
    }

    fn load_target_only(model_dir: &Path, kv_cache_mode: KVCacheMode) -> Result<Self> {
        initialize_runtime()?;
        ensure!(
            model_dir.is_dir(),
            "model directory does not exist or is not a directory: {}",
            model_dir.display()
        );

        let tokenizer_path = model_dir.join("tokenizer.json");
        ensure!(
            tokenizer_path.is_file(),
            "missing tokenizer {}",
            tokenizer_path.display()
        );
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
        let chat_template = ChatTemplateProcessor::from_model_path(model_dir)?;
        let defaults = load_generation_defaults(model_dir)?;
        let mut model = Qwen4Model::load(model_dir, kv_cache_mode)?;
        let mtp_model_dir = crate::resolve_mtp_model_path()?;
        ensure!(
            mtp_model_dir.is_dir(),
            "MTP companion model directory does not exist: {}",
            mtp_model_dir.display()
        );
        let mtp = Qwen4MtpDraftModel::load(&mtp_model_dir, &model.config)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to load Qwen4 MTP companion from {}",
                    mtp_model_dir.display()
                )
            })?;
        model.attach_mtp(mtp);
        let generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode);
        let mtp_generator = model.has_mtp().then(Qwen4MtpGenerator::new);

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            defaults,
            generator,
            mtp_generator,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn logits_vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    #[doc(hidden)]
    pub fn supported_context_tokens(&self) -> usize {
        self.model.config.max_position_embeddings
    }

    pub fn eos_token_id(&self) -> u32 {
        self.defaults.stop_token_ids[0] as u32
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp_generator.is_some()
    }

    #[tracing::instrument(
        name = "runtime.render_messages",
        skip_all,
        fields(
            message_count = messages.len(),
            tool_count = tools.len(),
            enable_thinking,
        ),
        err
    )]
    pub fn render_messages(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        reasoning_effort: Option<&str>,
        enable_thinking: bool,
    ) -> Result<String> {
        ensure!(
            messages
                .iter()
                .all(|message| message.image_urls().next().is_none()),
            "Qwen4 supports text input only"
        );
        self.chat_template
            .render_messages(messages, tools, reasoning_effort, enable_thinking, true)
    }

    #[tracing::instrument(
        name = "runtime.tokenize_messages",
        skip_all,
        fields(
            message_count = messages.len(),
            tool_count = tools.len(),
            enable_thinking,
        ),
        err
    )]
    pub fn tokenize_messages(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        reasoning_effort: Option<&str>,
        enable_thinking: bool,
    ) -> Result<Vec<i32>> {
        let rendered = self.render_messages(messages, tools, reasoning_effort, enable_thinking)?;
        let encoded = self
            .tokenizer
            .encode(rendered, true)
            .map_err(anyhow::Error::msg)
            .context("failed to tokenize rendered messages")?;
        let prompt_ids: Vec<i32> = encoded
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect();
        ensure!(
            !prompt_ids.is_empty(),
            "rendered messages tokenized to an empty sequence"
        );
        debug!(
            phase = "tokenization.complete",
            prompt_tokens = prompt_ids.len(),
        );
        Ok(prompt_ids)
    }

    /// Tokenize the same message list without the assistant generation marker.
    ///
    /// The result is a stable history checkpoint only when it is an exact
    /// prefix of [`Self::tokenize_messages`]' ordinary prompt.
    pub fn tokenize_history(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        reasoning_effort: Option<&str>,
        enable_thinking: bool,
    ) -> Result<Vec<i32>> {
        let rendered = self.chat_template.render_messages(
            messages,
            tools,
            reasoning_effort,
            enable_thinking,
            false,
        )?;
        let encoded = self
            .tokenizer
            .encode(rendered, true)
            .map_err(anyhow::Error::msg)
            .context("failed to tokenize rendered message history")?;
        let token_ids = encoded
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>();
        ensure!(
            !token_ids.is_empty(),
            "rendered message history tokenized to an empty sequence"
        );
        Ok(token_ids)
    }

    pub fn supports_qwen4_tool_calls(&self) -> bool {
        self.chat_template.supports_qwen4_tool_calls()
    }

    pub fn baseline_sampling(
        &self,
        temperature: Option<f32>,
        top_p: Option<f32>,
        seed: Option<u64>,
    ) -> SamplingConfig {
        SamplingConfig {
            temperature: temperature.unwrap_or(self.defaults.temperature),
            top_k: self.defaults.top_k,
            top_p: top_p.unwrap_or(self.defaults.top_p),
            seed,
            stop_token_ids: self.defaults.stop_token_ids.clone(),
            ..SamplingConfig::default()
        }
    }

    #[tracing::instrument(
        name = "runtime.generate_baseline",
        skip_all,
        fields(
            prompt_tokens = prompt_ids.len(),
            max_tokens,
            cached_tokens = prefix_reuse.as_ref().map_or(0, |reuse| reuse.cached_tokens),
            constrained = constraint.is_some(),
        ),
        err
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_baseline_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        prefix_reuse: Option<PrefixReuse<'_>>,
        constraint: Option<&mut dyn TokenConstraint>,
        checkpoint_token_lengths: &[usize],
        mut on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.model.clear_prepared_mrope();
        let buffer_output = constraint.is_some();
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut callback_active = true;
        let controlled: ControlledGeneration = self
            .generator
            .generate_streaming_controlled(
                &self.model,
                prompt_ids,
                prefix_reuse,
                max_tokens,
                sampling,
                constraint,
                checkpoint_token_lengths,
                |token_id| {
                    if buffer_output {
                        callback_active = on_delta("");
                        return callback_active;
                    }
                    match decoder.push(token_id) {
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
                    }
                },
            )
            .map_err(anyhow::Error::msg)
            .context("baseline generation failed")?;
        if let Some(error) = decode_error {
            return Err(error);
        }
        if buffer_output {
            for &token_id in &controlled.token_ids {
                let _ = decoder.push(token_id)?;
            }
            let _ = decoder.finish()?;
            if callback_active && !decoder.emitted.is_empty() {
                let _ = on_delta(&decoder.emitted);
            }
        } else {
            let final_delta = decoder.finish()?;
            if callback_active && !final_delta.is_empty() {
                let _ = on_delta(&final_delta);
            }
        }
        let text = decoder.emitted;
        let completion_tokens = controlled.token_ids.len();
        let prefill_time = controlled.prefill_time;
        let decode_time = controlled.decode_time;
        log_generation_metrics(
            "baseline",
            prompt_ids.len(),
            completion_tokens,
            controlled.cached_tokens,
            prefill_time,
            decode_time,
        );
        debug!(
            phase = "model.complete",
            prompt_tokens = prompt_ids.len(),
            completion_tokens,
            cached_tokens = controlled.cached_tokens,
            stop_reason = ?controlled.stop_reason,
            decode_seconds = decode_time.as_secs_f64(),
        );
        Ok(BaselineGeneration {
            text,
            token_ids: controlled.token_ids,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
            cached_tokens: controlled.cached_tokens,
            finish_outcome: controlled.stop_reason,
            prompt_snapshots: controlled
                .prompt_snapshots
                .into_iter()
                .map(PromptSnapshot::Baseline)
                .collect(),
            final_snapshot: controlled.final_snapshot.map(PromptSnapshot::Baseline),
            prefill_time,
            decode_time,
        })
    }

    pub fn generate_mtp_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        constraint: Option<&mut dyn TokenConstraint>,
        on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.generate_mtp_streaming_for_prompt(
            prompt_ids,
            max_tokens,
            sampling,
            block_size,
            prefix_reuse,
            checkpoint_token_lengths,
            constraint,
            on_delta,
        )
        .map(|(generation, _)| generation)
    }

    #[tracing::instrument(
        name = "runtime.generate_mtp",
        skip_all,
        fields(max_tokens, block_size),
        err
    )]

    fn generate_mtp_streaming_for_prompt<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        constraint: Option<&mut dyn TokenConstraint>,
        mut on_delta: F,
    ) -> Result<(BaselineGeneration, MtpGenerationStats)> {
        ensure!(block_size >= 2, "MTP block size must be at least 2");
        ensure!(
            self.mtp_generator.is_some(),
            "the loaded checkpoint does not contain a bundled Qwen4 MTP head"
        );
        let buffer_output = constraint.is_some();
        let prompt_tokens = prompt_ids.len();
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut callback_active = true;
        let generator = self
            .mtp_generator
            .as_mut()
            .expect("MTP capability was validated");
        let generated = generator
            .generate_streaming(
                &self.model,
                prompt_ids,
                max_tokens,
                sampling,
                block_size,
                prefix_reuse,
                checkpoint_token_lengths,
                constraint,
                |token_id| {
                    if buffer_output {
                        callback_active = on_delta("");
                        return callback_active;
                    }
                    match decoder.push(token_id) {
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
                    }
                },
            )
            .map_err(anyhow::Error::msg)
            .context("MTP generation failed")?;
        if let Some(error) = decode_error {
            return Err(error);
        }
        if buffer_output {
            for &token_id in &generated.token_ids {
                let _ = decoder.push(token_id)?;
            }
            let _ = decoder.finish()?;
            if callback_active && !decoder.emitted.is_empty() {
                let _ = on_delta(&decoder.emitted);
            }
        } else {
            let final_delta = decoder.finish()?;
            if callback_active && !final_delta.is_empty() {
                let _ = on_delta(&final_delta);
            }
        }
        let completion_tokens = generated.token_ids.len();
        let prefill_time = generated.stats.prefill_time;
        let decode_time = generated.stats.decode_time;
        log_generation_metrics(
            "mtp",
            prompt_tokens,
            completion_tokens,
            generated.cached_tokens,
            prefill_time,
            decode_time,
        );
        debug!(
            phase = "model.complete",
            prompt_tokens,
            completion_tokens,
            cached_tokens = generated.cached_tokens,
            stop_reason = ?generated.stop_reason,
            mtp_proposed_draft_tokens = generated.stats.proposed_draft_tokens,
            mtp_accepted_draft_tokens = generated.stats.accepted_draft_tokens,
            mtp_acceptance_percentage = generated.stats.acceptance_percentage(),
            mtp_decode_seconds = decode_time.as_secs_f64(),
            mtp_cache_clear_count = generated.stats.cache_clear_count,
            mtp_draft_seconds = generated.stats.draft_time.as_secs_f64(),
            mtp_target_verify_seconds = generated.stats.target_verify_time.as_secs_f64(),
            mtp_walk_seconds = generated.stats.walk_time.as_secs_f64(),
            mtp_reconcile_seconds = generated.stats.reconcile_time.as_secs_f64(),
            mtp_target_forward_calls = generated.stats.target_forward_calls,
            mtp_speculative_rounds = generated.stats.speculative_rounds,
            mtp_full_state_materializations = generated.stats.full_state_materializations,
            mtp_cache_snapshot_count = generated.stats.cache_snapshot_count,
            mtp_cache_clear_seconds = generated.stats.cache_clear_time.as_secs_f64(),
        );
        Ok((
            BaselineGeneration {
                text: decoder.emitted,
                token_ids: generated.token_ids,
                prompt_tokens,
                completion_tokens,
                cached_tokens: generated.cached_tokens,
                finish_outcome: generated.stop_reason,
                prompt_snapshots: generated
                    .prompt_snapshots
                    .into_iter()
                    .map(PromptSnapshot::Mtp)
                    .collect(),
                final_snapshot: generated.final_snapshot.map(PromptSnapshot::Mtp),
                prefill_time,
                decode_time,
            },
            generated.stats,
        ))
    }

    pub fn generate_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        on_delta: F,
    ) -> Result<GenerationOutput> {
        self.generate_streaming_in_mode(request, Qwen4GenerationMode::Automatic, on_delta)
            .map(|(output, _)| output)
    }

    #[tracing::instrument(
        name = "runtime.generate_streaming",
        skip_all,
        fields(max_tokens = request.max_tokens, mode = ?mode),
        err
    )]
    #[doc(hidden)]
    pub fn generate_streaming_in_mode<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        mode: Qwen4GenerationMode,
        on_delta: F,
    ) -> Result<(GenerationOutput, Option<MtpGenerationStats>)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let use_mtp = self.resolve_generation_mode(mode)?
            && !(mode == Qwen4GenerationMode::Automatic && request.max_tokens == 1);
        if !use_mtp {
            let generation = self.generate_baseline_streaming(
                &prompt_ids,
                request.max_tokens,
                &sampling,
                None,
                None,
                &[],
                on_delta,
            )?;
            return Ok((
                GenerationOutput {
                    text: generation.text,
                    token_ids: generation.token_ids,
                },
                None,
            ));
        }

        let (generation, stats) = self.generate_mtp_streaming_for_prompt(
            &prompt_ids,
            request.max_tokens,
            &sampling,
            DEFAULT_MTP_BLOCK_SIZE,
            None,
            &[],
            None,
            on_delta,
        )?;
        Ok((
            GenerationOutput {
                text: generation.text,
                token_ids: generation.token_ids,
            },
            Some(stats),
        ))
    }

    /// Controlled decode benchmark route. Production callers continue through
    /// [`Self::generate_streaming`], which selects bundled MTP when present.
    #[doc(hidden)]
    pub fn benchmark_streaming_in_mode<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        mode: Qwen4GenerationMode,
        on_delta: F,
    ) -> Result<(BaselineGeneration, Option<MtpGenerationStats>)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        if !self.resolve_generation_mode(mode)? {
            let generation = self.generate_baseline_streaming(
                &prompt_ids,
                request.max_tokens,
                &sampling,
                None,
                None,
                &[],
                on_delta,
            )?;
            return Ok((generation, None));
        }
        let (generation, stats) = self.generate_mtp_streaming_for_prompt(
            &prompt_ids,
            request.max_tokens,
            &sampling,
            DEFAULT_MTP_BLOCK_SIZE,
            None,
            &[],
            None,
            on_delta,
        )?;
        Ok((generation, Some(stats)))
    }

    /// Controlled cached-context benchmark route. The snapshot is reusable:
    /// each call restores the same production prefix state before processing
    /// the uncached prompt suffix.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_cached_streaming_in_mode<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        snapshot: &PromptSnapshot,
        mode: Qwen4GenerationMode,
        on_delta: F,
    ) -> Result<(BaselineGeneration, Option<MtpGenerationStats>)> {
        match (mode, snapshot) {
            (Qwen4GenerationMode::Baseline, snapshot) => {
                let snapshot = match snapshot {
                    PromptSnapshot::Baseline(snapshot) => snapshot,
                    PromptSnapshot::Mtp(snapshot) => snapshot.target_snapshot(),
                };
                let generation = self.generate_baseline_streaming(
                    prompt_ids,
                    max_tokens,
                    sampling,
                    Some(PrefixReuse {
                        snapshot,
                        cached_tokens: snapshot.token_len(),
                    }),
                    None,
                    &[],
                    on_delta,
                )?;
                Ok((generation, None))
            }
            (Qwen4GenerationMode::Mtp, PromptSnapshot::Mtp(snapshot)) => {
                let (generation, stats) = self.generate_mtp_streaming_for_prompt(
                    prompt_ids,
                    max_tokens,
                    sampling,
                    DEFAULT_MTP_BLOCK_SIZE,
                    Some(MtpPrefixReuse {
                        snapshot,
                        cached_tokens: snapshot.token_len(),
                        continuation_token: None,
                    }),
                    &[],
                    None,
                    on_delta,
                )?;
                Ok((generation, Some(stats)))
            }
            (Qwen4GenerationMode::Automatic, _) => {
                anyhow::bail!("cached benchmark mode must be explicit")
            }
            (Qwen4GenerationMode::Mtp, PromptSnapshot::Baseline(_)) => {
                anyhow::bail!("cached benchmark mode does not match the snapshot family")
            }
        }
    }

    fn resolve_generation_mode(&self, mode: Qwen4GenerationMode) -> Result<bool> {
        match mode {
            Qwen4GenerationMode::Automatic => Ok(self.mtp_generator.is_some()),
            Qwen4GenerationMode::Baseline => Ok(false),
            Qwen4GenerationMode::Mtp => {
                ensure!(
                    self.mtp_generator.is_some(),
                    "the loaded checkpoint does not contain a bundled Qwen4 MTP head"
                );
                Ok(true)
            }
        }
    }
    #[tracing::instrument(
        name = "runtime.prepare_generation",
        skip_all,
        fields(max_tokens = request.max_tokens),
        err
    )]

    fn prepare_generation(
        &self,
        request: &GenerationRequest,
    ) -> Result<(Vec<i32>, SamplingConfig)> {
        ensure!(!request.prompt.is_empty(), "prompt must not be empty");
        ensure!(
            request.max_tokens > 0,
            "max_tokens must be greater than zero"
        );

        let rendered = self.chat_template.render_user(&request.prompt)?;
        let encoded = self
            .tokenizer
            .encode(rendered, true)
            .map_err(anyhow::Error::msg)
            .context("failed to tokenize rendered prompt")?;
        let prompt_ids: Vec<i32> = encoded
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect();
        ensure!(
            !prompt_ids.is_empty(),
            "rendered prompt tokenized to an empty sequence"
        );

        let sampling = SamplingConfig {
            temperature: request.temperature.unwrap_or(self.defaults.temperature),
            top_k: request.top_k.unwrap_or(self.defaults.top_k),
            top_p: request.top_p.unwrap_or(self.defaults.top_p),
            seed: request.seed,
            stop_token_ids: self.defaults.stop_token_ids.clone(),
            ..SamplingConfig::default()
        };
        Ok((prompt_ids, sampling))
    }
}

fn metal_wired_limit(system_memory: u64, max_recommended_working_set: u64) -> Result<u64> {
    ensure!(
        system_memory > 0,
        "failed to determine physical system memory for the Metal wired-memory limit"
    );
    ensure!(
        max_recommended_working_set > 0,
        "Metal did not report a maximum recommended working-set size"
    );
    let eighty_five_percent = ((u128::from(system_memory) * 85) / 100) as u64;
    Ok(eighty_five_percent.min(max_recommended_working_set))
}

fn initialize_runtime() -> Result<()> {
    static INITIALIZED: LazyLock<std::result::Result<(), String>> = LazyLock::new(|| {
        // MLX's 256 MiB default command-buffer cap lets this model accumulate
        // enough decode work to delay submission. Fifteen MiB measured best on
        // the supported Apple-Silicon path. Respect an explicit operator override.
        if std::env::var_os("MLX_MAX_MB_PER_BUFFER").is_none() {
            // SAFETY: this one-time initializer runs before the first MLX
            // backend query or model worker is created.
            unsafe {
                std::env::set_var("MLX_MAX_MB_PER_BUFFER", "15");
            }
        }
        if !mlxcel_core::metal_is_available() {
            return Err("the MLX Metal backend is unavailable on this host".to_string());
        }
        mlxcel_core::set_default_device(true);
        let system_memory = mlxcel_core::hardware::system_memory_bytes();
        // Upstream `set_wired_limit` rejects values above Metal's recommended
        // maximum, so enforce both ceilings: 85% of physical unified memory
        // and `recommendedMaxWorkingSetSize`.
        let max_recommended_working_set = mlxcel_core::get_wired_limit() as u64;
        let wired_limit = metal_wired_limit(system_memory, max_recommended_working_set)
            .map_err(|error| error.to_string())?;
        mlxcel_core::set_wired_limit(wired_limit as usize);
        tracing::info!(
            wired_limit,
            system_memory,
            max_recommended_working_set,
            "configured Metal wired-memory ceiling"
        );
        const MLX_MEMORY_LIMIT: u64 = 45 * 1024 * 1024 * 1024;
        const MLX_CACHE_LIMIT: u64 = 512 * 1024 * 1024;
        mlxcel_core::memory::set_memory_limit(MLX_MEMORY_LIMIT);
        mlxcel_core::memory::set_cache_limit(MLX_CACHE_LIMIT);
        Ok(())
    });
    (*INITIALIZED).clone().map_err(anyhow::Error::msg)
}

fn load_generation_defaults(model_dir: &Path) -> Result<GenerationDefaults> {
    let path: PathBuf = model_dir.join("generation_config.json");
    let config = if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str::<GenerationConfig>(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?
    } else {
        GenerationConfig {
            eos_token_id: None,
            temperature: None,
            top_k: None,
            top_p: None,
        }
    };
    let mut stop_token_ids = match config.eos_token_id {
        Some(TokenIds::One(token)) => vec![token],
        Some(TokenIds::Many(tokens)) => tokens,
        None => vec![248046, 248044],
    };
    stop_token_ids.sort_unstable();
    stop_token_ids.dedup();
    ensure!(
        !stop_token_ids.is_empty(),
        "{} contains an empty eos_token_id list",
        path.display()
    );

    Ok(GenerationDefaults {
        stop_token_ids,
        temperature: config.temperature.unwrap_or(1.0),
        top_k: config.top_k.unwrap_or(20),
        top_p: config.top_p.unwrap_or(0.95),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    #[test]
    fn generation_metric_throughput_uses_elapsed_seconds() {
        assert_eq!(tokens_per_second(25, Duration::from_millis(500)), 50.0);
        assert_eq!(tokens_per_second(25, Duration::ZERO), 0.0);
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "qw-provider-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn metal_wired_limit_is_eighty_five_percent_or_device_max() {
        let gib = 1024_u64.pow(3);
        assert_eq!(
            metal_wired_limit(128 * gib, 120 * gib).unwrap(),
            108 * gib + 4 * gib / 5
        );
        assert_eq!(metal_wired_limit(128 * gib, 96 * gib).unwrap(), 96 * gib);
        assert!(metal_wired_limit(0, 96 * gib).is_err());
        assert!(metal_wired_limit(128 * gib, 0).is_err());
    }

    #[test]
    fn generation_defaults_match_checkpoint_contract() {
        let fixture = TestDir::new("generation-defaults");
        let fallback = load_generation_defaults(&fixture.0).expect("fallback defaults");
        assert_eq!(fallback.stop_token_ids, vec![248044, 248046]);
        assert_eq!(fallback.temperature, 1.0);
        assert_eq!(fallback.top_k, 20);
        assert_eq!(fallback.top_p, 0.95);

        std::fs::write(
            fixture.0.join("generation_config.json"),
            br#"{"eos_token_id":[9,7],"temperature":0.7,"top_k":11,"top_p":0.8}"#,
        )
        .expect("write generation config");
        let loaded = load_generation_defaults(&fixture.0).expect("checkpoint defaults");
        assert_eq!(loaded.stop_token_ids, vec![7, 9]);
        assert_eq!(loaded.temperature, 0.7);
        assert_eq!(loaded.top_k, 11);
        assert_eq!(loaded.top_p, 0.8);
    }

    #[test]
    fn provider_rejects_absent_chat_template_with_paths() {
        let fixture = TestDir::new("missing-chat-template");
        std::fs::write(
            fixture.0.join("tokenizer.json"),
            br#"{
                "version":"1.0",
                "truncation":null,
                "padding":null,
                "added_tokens":[],
                "normalizer":null,
                "pre_tokenizer":null,
                "post_processor":null,
                "decoder":null,
                "model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}
            }"#,
        )
        .expect("write tokenizer");
        std::fs::write(fixture.0.join("tokenizer_config.json"), b"{}")
            .expect("write tokenizer config");

        let error = match Qwen4Provider::load_target_only(&fixture.0, KVCacheMode::Fp8) {
            Ok(_) => panic!("provider load must reject an absent chat template"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("missing chat template"), "{error}");
        assert!(error.contains("chat_template.jinja"), "{error}");
        assert!(error.contains("tokenizer_config.json"), "{error}");
    }

    #[test]
    fn incremental_decoder_withholds_split_utf8_replacement_text() {
        let mut emitted = String::new();
        let deltas = [
            advance_decoded_text(&mut emitted, "\u{fffd}", false)
                .expect("first byte-fallback token"),
            advance_decoded_text(&mut emitted, "\u{fffd}", false)
                .expect("second byte-fallback token"),
            advance_decoded_text(&mut emitted, "你", false).expect("completed UTF-8 sequence"),
        ];
        assert_eq!(deltas.concat(), "你");
        assert_eq!(emitted, "你");
        assert!(!emitted.contains('\u{fffd}'));
    }

    #[test]
    fn incremental_decoder_deltas_equal_final_ordinary_bpe_decode() {
        let mut emitted = String::new();
        let mut deltas = Vec::new();
        for decoded in ["Hello", "Hello, ", "Hello, world"] {
            deltas.push(
                advance_decoded_text(&mut emitted, decoded, false).expect("monotonic BPE decode"),
            );
        }
        let final_delta =
            advance_decoded_text(&mut emitted, "Hello, world!", true).expect("final decode");
        deltas.push(final_delta);
        assert_eq!(deltas.concat(), "Hello, world!");
        assert_eq!(emitted, "Hello, world!");
    }
    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_baseline_and_mtp_greedy_outputs_match() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider =
            Qwen4Provider::load(&model_dir, KVCacheMode::Fp8).expect("load real Qwen checkpoint");
        let request = GenerationRequest {
            prompt: "Continue counting upward from one, writing each integer on its own line without stopping."
                .to_string(),
            max_tokens: 32,
            temperature: Some(0.0),
            top_k: Some(1),
            top_p: Some(1.0),
            seed: Some(0),
        };
        let mut baseline_deltas = String::new();
        let (baseline, _) = provider
            .generate_streaming_in_mode(&request, Qwen4GenerationMode::Baseline, |delta| {
                baseline_deltas.push_str(delta);
                true
            })
            .expect("baseline greedy generation");
        let mut mtp_deltas = String::new();
        let (mtp, mtp_stats) = provider
            .generate_streaming_in_mode(&request, Qwen4GenerationMode::Mtp, |delta| {
                mtp_deltas.push_str(delta);
                true
            })
            .expect("MTP greedy generation");
        assert_eq!(baseline.token_ids, mtp.token_ids, "{mtp_stats:?}");
        assert_eq!(baseline.text, mtp.text);
        assert_eq!(baseline_deltas, baseline.text);
        assert_eq!(mtp_deltas, mtp.text);
    }

    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_mtp_prefix_reuse_matches_cold_and_reduces_ttft() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider = Qwen4Provider::load(&model_dir, KVCacheMode::Fp8)
            .expect("load real bundled-MTP checkpoint");
        let base = provider
            .tokenizer
            .encode(
                "A deterministic cache benchmark paragraph. ".repeat(128),
                true,
            )
            .expect("encode benchmark prefix");
        let suffix = provider
            .tokenizer
            .encode(
                "Continue this exact history with one additional user turn.",
                false,
            )
            .expect("encode benchmark suffix");
        let base_ids = base
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>();
        let mut full_ids = base_ids.clone();
        full_ids.extend(suffix.get_ids().iter().map(|&token| token as i32));
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));

        let mut cold_ttft = None;
        let cold_started = Instant::now();
        let cold = provider
            .generate_mtp_streaming(
                &full_ids,
                32,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[],
                None,
                |_| {
                    cold_ttft.get_or_insert_with(|| cold_started.elapsed());
                    true
                },
            )
            .expect("cold MTP generation");

        let prefix = provider
            .generate_mtp_streaming(
                &base_ids,
                1,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[base_ids.len()],
                None,
                |_| true,
            )
            .expect("capture MTP prompt snapshot");
        let PromptSnapshot::Mtp(snapshot) = prefix
            .prompt_snapshots
            .into_iter()
            .next()
            .expect("complete MTP prompt snapshot")
        else {
            panic!("MTP generation must return an MTP snapshot");
        };
        let mut warm_ttft = None;
        let warm_started = Instant::now();
        let warm = provider
            .generate_mtp_streaming(
                &full_ids,
                32,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                Some(MtpPrefixReuse {
                    snapshot: &snapshot,
                    cached_tokens: base_ids.len(),
                    continuation_token: None,
                }),
                &[full_ids.len()],
                None,
                |_| {
                    warm_ttft.get_or_insert_with(|| warm_started.elapsed());
                    true
                },
            )
            .expect("warm MTP generation");

        let baseline_cold = provider
            .generate_baseline_streaming(&full_ids, 32, &sampling, None, None, &[], |_| true)
            .expect("cold baseline generation");
        let mtp_snapshot = PromptSnapshot::Mtp(snapshot);
        let (baseline_warm, baseline_stats) = provider
            .benchmark_cached_streaming_in_mode(
                &full_ids,
                32,
                &sampling,
                &mtp_snapshot,
                Qwen4GenerationMode::Baseline,
                |_| true,
            )
            .expect("baseline generation from MTP target snapshot");
        assert_eq!(baseline_warm.cached_tokens, base_ids.len());
        assert_eq!(baseline_warm.token_ids, baseline_cold.token_ids);
        assert_eq!(baseline_warm.text, baseline_cold.text);
        assert!(
            baseline_stats.is_none(),
            "explicit baseline mode returned MTP statistics"
        );

        assert_eq!(warm.cached_tokens, base_ids.len());
        assert_eq!(warm.token_ids, cold.token_ids);
        assert_eq!(warm.text, cold.text);
        let cold_ttft = cold_ttft.expect("cold generation emitted a token");
        let warm_ttft = warm_ttft.expect("warm generation emitted a token");
        eprintln!(
            "MTP prefix benchmark: cached_tokens={}, cold_ttft_ms={:.2}, warm_ttft_ms={:.2}, \
             cold_decode_ms={:.2}, warm_decode_ms={:.2}",
            warm.cached_tokens,
            cold_ttft.as_secs_f64() * 1_000.0,
            warm_ttft.as_secs_f64() * 1_000.0,
            cold.decode_time.as_secs_f64() * 1_000.0,
            warm.decode_time.as_secs_f64() * 1_000.0,
        );
        assert!(
            warm_ttft < cold_ttft,
            "warm suffix prefill must reduce TTFT: cold={cold_ttft:?}, warm={warm_ttft:?}"
        );
        assert_eq!(
            warm.final_snapshot
                .as_ref()
                .expect("warm generation must capture a final snapshot")
                .token_len(),
            cold.final_snapshot
                .as_ref()
                .expect("cold generation must capture a final snapshot")
                .token_len(),
            "prefix reuse must preserve the final snapshot boundary",
        );
    }

    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_mtp_prefix_reuse_covers_reasoning_and_plain_history() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider = Qwen4Provider::load(&model_dir, KVCacheMode::Fp8)
            .expect("load real bundled-MTP checkpoint");
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        let user = |content: &str| ChatMessage {
            role: "user".to_string(),
            name: None,
            content: Some(ChatMessageContent::Text(content.to_string())),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        };
        let history_for = |reasoning: Option<&str>| {
            vec![
                user("Explain why prefix caching is token based."),
                ChatMessage {
                    role: "assistant".to_string(),
                    name: None,
                    content: Some(ChatMessageContent::Text(
                        "The cache compares the rendered token prefix.".to_string(),
                    )),
                    reasoning_content: reasoning.map(str::to_string),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
            ]
        };

        for reasoning in [None, Some("I should answer precisely and briefly.")] {
            let history = history_for(reasoning);
            let history_ids = provider
                .tokenize_history(&history, &[], None, true)
                .expect("tokenize history");
            let mut next_turn = history.clone();
            next_turn.push(user("Now compare this with a plain history."));
            let next_ids = provider
                .tokenize_messages(&next_turn, &[], None, true)
                .expect("tokenize next turn");
            assert!(
                history_ids.len() < next_ids.len(),
                "next turn must extend the history checkpoint"
            );

            let checkpoint = provider
                .generate_mtp_streaming(
                    &history_ids,
                    1,
                    &sampling,
                    DEFAULT_MTP_BLOCK_SIZE,
                    None,
                    &[history_ids.len()],
                    None,
                    |_| true,
                )
                .expect("capture reasoning-aware history snapshot");
            let PromptSnapshot::Mtp(snapshot) = checkpoint
                .prompt_snapshots
                .into_iter()
                .next()
                .expect("history checkpoint snapshot")
            else {
                panic!("expected an MTP history checkpoint");
            };

            let reused = provider
                .generate_mtp_streaming(
                    &next_ids,
                    1,
                    &sampling,
                    DEFAULT_MTP_BLOCK_SIZE,
                    Some(MtpPrefixReuse {
                        snapshot: &snapshot,
                        cached_tokens: history_ids.len(),
                        continuation_token: None,
                    }),
                    &[next_ids.len()],
                    None,
                    |_| true,
                )
                .expect("reuse reasoning-aware history snapshot");
            assert_eq!(
                reused.cached_tokens,
                history_ids.len(),
                "exactly replayed reasoning/plain history must be reusable"
            );
            eprintln!(
                "MTP reasoning-prefix benchmark: reasoning={}, history_tokens={}, cached_tokens={}",
                reasoning.is_some(),
                history_ids.len(),
                reused.cached_tokens,
            );
        }

        let with_reasoning = provider
            .tokenize_history(&history_for(Some("private trace")), &[], None, true)
            .expect("tokenize history with reasoning");
        let without_reasoning = provider
            .tokenize_history(&history_for(None), &[], None, true)
            .expect("tokenize history without reasoning");
        let common_prefix = with_reasoning
            .iter()
            .zip(&without_reasoning)
            .take_while(|(left, right)| left == right)
            .count();
        eprintln!(
            "MTP reasoning-prefix boundary: common_tokens={}, with_reasoning={}, without_reasoning={}",
            common_prefix,
            with_reasoning.len(),
            without_reasoning.len(),
        );
        assert!(common_prefix > 0);
        assert!(
            common_prefix < with_reasoning.len().min(without_reasoning.len()),
            "omitting reasoning must create a cache boundary at the rendered divergence"
        );
    }

    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_mtp_max_output_has_bounded_terminal_tail() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider = Qwen4Provider::load(&model_dir, KVCacheMode::Fp8)
            .expect("load real bundled-MTP checkpoint");
        let prompt = provider
            .tokenizer
            .encode(
                "Produce a very long technical essay about distributed systems. \
                 Continue until the output limit and do not stop early.",
                true,
            )
            .expect("encode long-output prompt");
        let prompt_ids = prompt
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>();
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        let mut callback_count = 0;
        let mut last_callback = None;
        let generated = provider
            .generate_mtp_streaming(
                &prompt_ids,
                1024,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[],
                None,
                |_| {
                    callback_count += 1;
                    last_callback = Some(Instant::now());
                    true
                },
            )
            .expect("long MTP generation");
        let tail = last_callback
            .expect("long generation emitted a token")
            .elapsed();
        assert_eq!(callback_count, 1024);
        assert_eq!(generated.finish_outcome, GenerationStopReason::MaxTokens);
        eprintln!(
            "MTP terminal-tail benchmark: completion_tokens={}, callbacks={}, tail_ms={:.2}",
            generated.completion_tokens,
            callback_count,
            tail.as_secs_f64() * 1_000.0,
        );
        assert!(
            tail < Duration::from_secs(10),
            "terminal snapshot must not replay the output: tail={tail:?}"
        );
    }

    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_mtp_eos_has_bounded_terminal_tail_and_resumable_snapshot() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider =
            Qwen4Provider::load(&model_dir, KVCacheMode::Fp8).expect("load real MTP checkpoint");
        let prompt = provider
            .tokenizer
            .encode(
                "Continue this deterministic numbered sequence with one number per line: \
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
                true,
            )
            .expect("encode deterministic MTP prompt");
        let prompt_ids = prompt
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>();
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        let mut generator = provider
            .mtp_generator
            .take()
            .expect("real checkpoint must contain an MTP generator");
        let control = generator
            .generate_streaming(
                &provider.model,
                &prompt_ids,
                96,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[],
                None,
                |_| true,
            )
            .expect("generate deterministic greedy control sequence");
        let candidate = control
            .token_ids
            .iter()
            .enumerate()
            .skip(4)
            .find_map(|(index, &token)| {
                (!control.token_ids[..index].contains(&token)).then_some((index, token))
            })
            .or_else(|| {
                control
                    .token_ids
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find_map(|(index, &token)| {
                        (!control.token_ids[..index].contains(&token)).then_some((index, token))
                    })
            })
            .expect("control sequence must contain a stop-token candidate after its first token");
        let (candidate_index, candidate_token) = candidate;
        assert!(candidate_index > 0);

        let mut stop_sampling = sampling.clone();
        stop_sampling.stop_token_ids = vec![candidate_token];
        let mut callback_tokens = Vec::new();
        let mut last_callback = None;
        let stopped = generator
            .generate_streaming(
                &provider.model,
                &prompt_ids,
                96,
                &stop_sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[],
                None,
                |token| {
                    callback_tokens.push(token);
                    last_callback = Some(Instant::now());
                    true
                },
            )
            .expect("generate until selected stop token");
        let terminal_tail = last_callback
            .expect("stop-token generation must emit a visible token")
            .elapsed();

        assert_eq!(stopped.stop_reason, GenerationStopReason::Eos);
        assert_eq!(stopped.token_ids, control.token_ids[..candidate_index]);
        assert!(!callback_tokens.contains(&candidate_token));
        assert!(!stopped.token_ids.contains(&candidate_token));
        assert!(
            terminal_tail < Duration::from_secs(10),
            "terminal snapshot must not replay the prompt: tail={terminal_tail:?}"
        );
        let snapshot = stopped
            .final_snapshot
            .expect("stop-token generation must return a final MTP snapshot");
        assert_eq!(
            snapshot.token_len(),
            prompt_ids.len() + stopped.token_ids.len() - 1
        );

        let mut completed_prompt = prompt_ids.clone();
        completed_prompt.extend_from_slice(&stopped.token_ids);
        let resume_sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
        let cold = generator
            .generate_streaming(
                &provider.model,
                &completed_prompt,
                8,
                &resume_sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[],
                None,
                |_| true,
            )
            .expect("cold greedy continuation");
        let warm = generator
            .generate_streaming(
                &provider.model,
                &completed_prompt,
                8,
                &resume_sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                Some(MtpPrefixReuse {
                    snapshot: &snapshot,
                    cached_tokens: snapshot.token_len(),
                    continuation_token: stopped.token_ids.last().copied(),
                }),
                &[],
                None,
                |_| true,
            )
            .expect("snapshot-resumed greedy continuation");
        assert_eq!(
            warm.token_ids, cold.token_ids,
            "terminal MTP snapshot must preserve the next greedy tokens"
        );
    }

    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_cancelled_mtp_snapshot_portable_resume_matches_uninterrupted_greedy() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider = Qwen4Provider::load(&model_dir, KVCacheMode::Fp8)
            .expect("load real bundled-MTP checkpoint");
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                name: None,
                content: Some(ChatMessageContent::Text(
                    "What is the weather in Paris? Use the weather tool.".to_string(),
                )),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                name: None,
                content: None,
                reasoning_content: None,
                tool_calls: vec![ChatToolCall::Function {
                    id: "call_prior".to_string(),
                    function: ChatToolCallFunction {
                        name: "weather".to_string(),
                        arguments: serde_json::json!({"city": "Paris"}),
                    },
                }],
                tool_call_id: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                name: None,
                content: Some(ChatMessageContent::Text("18 C and clear".to_string())),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: Some("call_prior".to_string()),
            },
            ChatMessage {
                role: "user".to_string(),
                name: None,
                content: Some(ChatMessageContent::Text(
                    "Using the weather result, provide a detailed two-paragraph travel \
                     recommendation for Paris. Do not call another tool."
                        .to_string(),
                )),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        ];
        let prompt_ids = provider
            .tokenize_messages(&messages, &[], None, false)
            .expect("tokenize deterministic resume conversation");
        let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(7));
        let mut control = provider
            .generate_mtp_streaming(
                &prompt_ids,
                128,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[prompt_ids.len()],
                None,
                |_| true,
            )
            .expect("uninterrupted MTP generation");
        let prompt_snapshot = control
            .prompt_snapshots
            .pop()
            .expect("control donates full-prompt checkpoint");
        let prompt_portable = prompt_snapshot
            .to_portable()
            .expect("encode prompt checkpoint");
        let PromptSnapshot::Mtp(prompt_snapshot) =
            PromptSnapshot::from_portable(prompt_portable).expect("restore prompt checkpoint")
        else {
            panic!("portable prompt snapshot changed family");
        };

        let mut cold_callbacks = 0;
        let mut cold_interrupted = provider
            .generate_mtp_streaming(
                &prompt_ids,
                128,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                None,
                &[],
                None,
                |_| {
                    cold_callbacks += 1;
                    cold_callbacks < 20
                },
            )
            .expect("cold cancelled MTP generation");
        let cold_accepted = cold_interrupted.token_ids.len();
        let cold_snapshot = cold_interrupted
            .final_snapshot
            .take()
            .expect("cold cancellation donates snapshot")
            .to_portable()
            .expect("encode cold cancellation snapshot");

        let warm = provider
            .generate_mtp_streaming(
                &prompt_ids,
                128,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                Some(MtpPrefixReuse {
                    snapshot: &prompt_snapshot,
                    cached_tokens: prompt_ids.len(),
                    continuation_token: None,
                }),
                &[],
                None,
                |_| true,
            )
            .expect("uninterrupted full-prompt restore");
        assert_eq!(
            warm.token_ids, control.token_ids,
            "full-prompt restore diverged before cancellation"
        );

        let mut callbacks = 0;
        let mut interrupted = provider
            .generate_mtp_streaming(
                &prompt_ids,
                128,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                Some(MtpPrefixReuse {
                    snapshot: &prompt_snapshot,
                    cached_tokens: prompt_ids.len(),
                    continuation_token: None,
                }),
                &[],
                None,
                |_| {
                    callbacks += 1;
                    callbacks < 20
                },
            )
            .expect("cancelled MTP generation");
        assert_eq!(
            interrupted.finish_outcome,
            GenerationStopReason::CallbackCancelled
        );
        let accepted = interrupted.token_ids.len();
        assert!(accepted >= 20 && accepted < control.token_ids.len());
        assert_eq!(
            interrupted.token_ids,
            control.token_ids[..accepted],
            "interrupted generation diverged before snapshot donation"
        );
        let snapshot = interrupted
            .final_snapshot
            .take()
            .expect("cancelled generation donates an MTP snapshot");
        assert_eq!(snapshot.token_len(), prompt_ids.len() + accepted - 1);
        let portable = snapshot.to_portable().expect("encode cancelled snapshot");
        assert_eq!(accepted, cold_accepted, "cancellation boundary changed");
        assert_eq!(
            interrupted.token_ids, cold_interrupted.token_ids,
            "cold and restored-prefix cancellations emitted different tokens"
        );
        let (
            crate::PortablePromptSnapshot::Mtp {
                target: cold_target,
                draft: cold_draft,
                draft_offset: cold_draft_offset,
                last_hidden: cold_last_hidden,
                continuation_logits: cold_continuation_logits,
            },
            crate::PortablePromptSnapshot::Mtp {
                target,
                draft,
                draft_offset,
                last_hidden,
                continuation_logits,
            },
        ) = (&cold_snapshot, &portable)
        else {
            panic!("cancellation snapshots must both be MTP");
        };
        assert_eq!(
            target.token_len,
            prompt_ids.len() + accepted - 1,
            "target snapshot must exclude the unforwarded response bonus"
        );
        assert_eq!(
            *draft_offset,
            i32::try_from(target.token_len - 1).expect("snapshot length fits i32"),
            "drafter snapshot must match the shifted target boundary"
        );
        assert!(cold_target == target, "cancelled target states differ");
        assert_eq!(cold_draft_offset, draft_offset, "drafter offsets differ");
        assert!(cold_draft == draft, "drafter states differ");
        assert!(cold_last_hidden == last_hidden, "last hidden states differ");
        assert!(
            cold_continuation_logits == continuation_logits,
            "continuation logits differ"
        );
        let PromptSnapshot::Mtp(live_snapshot) = snapshot else {
            panic!("cancelled snapshot changed family");
        };

        let mut resume_prompt = prompt_ids.clone();
        resume_prompt.extend_from_slice(&interrupted.token_ids);
        let direct_resumed = provider
            .generate_mtp_streaming(
                &resume_prompt,
                128 - accepted,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                Some(MtpPrefixReuse {
                    snapshot: &live_snapshot,
                    cached_tokens: live_snapshot.token_len(),
                    continuation_token: interrupted.token_ids.last().copied(),
                }),
                &[],
                None,
                |_| true,
            )
            .expect("resume live MTP snapshot");
        let mut direct_combined = interrupted.token_ids.clone();
        direct_combined.extend_from_slice(&direct_resumed.token_ids);
        assert_eq!(
            direct_combined, control.token_ids,
            "live cancelled snapshot diverged before portable conversion"
        );
        let PromptSnapshot::Mtp(restored) =
            PromptSnapshot::from_portable(portable).expect("restore cancelled snapshot")
        else {
            panic!("portable MTP snapshot changed family");
        };

        let resumed = provider
            .generate_mtp_streaming(
                &resume_prompt,
                128 - accepted,
                &sampling,
                DEFAULT_MTP_BLOCK_SIZE,
                Some(MtpPrefixReuse {
                    snapshot: &restored,
                    cached_tokens: restored.token_len(),
                    continuation_token: interrupted.token_ids.last().copied(),
                }),
                &[],
                None,
                |_| true,
            )
            .expect("resume portable MTP snapshot");
        let mut combined = interrupted.token_ids;
        combined.extend_from_slice(&resumed.token_ids);
        assert_eq!(combined, control.token_ids);
    }
}
