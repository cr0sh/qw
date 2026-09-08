use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;
#[cfg(any(feature = "specprefill", test))]
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::{
    ControlledGeneration, CxxGenerator, GenerationStopReason, LanguageModel, ModelStateSnapshot,
    PrefixReuse, SamplingConfig, TokenConstraint,
};
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::loop_detection::detect_repetition_loop;
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::sampling::{
    SamplerState, sample_token_optimized, sample_token_optimized_with_state,
};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::chat_template::ChatTemplateProcessor;
pub use crate::chat_template::{
    ChatContentPart, ChatContentRef, ChatCustomToolCall, ChatFile, ChatImageUrl, ChatInputAudio,
    ChatMessage, ChatMessageContent, ChatPromptCacheBreakpoint, ChatTool, ChatToolCall,
    ChatToolCallFunction, ChatToolFunction,
};
use crate::qwen_vl::insert_qwen_vl_image_tokens;
use crate::qwen_vl_merge::merge_llava;
use crate::qwen_vl_position::compute_rope_index;
use crate::qwen_vl_processor::{PreparedImage, QwenVLProcessor};
use crate::qwen3_5::Qwen35Model;
#[cfg(any(feature = "dflash2", test))]
pub use crate::qwen3_5_dflash::{
    Dflash2GenerationStats, Dflash2PrefixReuse, Dflash2PromptSnapshot,
};
use crate::qwen3_5_mtp::Qwen35MtpGenerator;
pub use crate::qwen3_5_mtp::{MtpGenerationStats, MtpPrefixReuse, MtpPromptSnapshot};
#[cfg(any(feature = "specprefill", test))]
use crate::specprefill::{
    PrefillMode, SpecPrefillConfig, SpecPrefillStats, dense_prefix_end, score_tokens,
    select_target_indices, should_activate,
};

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
#[cfg(any(feature = "specprefill", test))]
struct SparseGeneration {
    token_ids: Vec<i32>,
    stop_reason: GenerationStopReason,
    cached_tokens: usize,
    prefill_time: Duration,
    decode_time: Duration,
    stats: SpecPrefillStats,
}

#[cfg(any(feature = "specprefill", test))]
#[allow(clippy::too_many_arguments)]
fn generate_specprefill_tokens<F: FnMut(i32) -> bool>(
    model: &Qwen35Model,
    draft: &Qwen35Model,
    prompt_ids: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    prefix_reuse: Option<PrefixReuse<'_>>,
    config: SpecPrefillConfig,
    mut on_token: F,
) -> Result<SparseGeneration> {
    let requested_cached_tokens = prefix_reuse.as_ref().map_or(0, |reuse| reuse.cached_tokens);
    let structurally_reusable = prefix_reuse.as_ref().is_some_and(|reuse| {
        reuse.cached_tokens > 0
            && reuse.cached_tokens <= prompt_ids.len()
            && reuse.snapshot.token_len() == reuse.cached_tokens
            && (reuse.cached_tokens < prompt_ids.len()
                || reuse.snapshot.continuation_logits().is_some())
    });
    let admitted_cached_tokens = structurally_reusable
        .then_some(requested_cached_tokens)
        .unwrap_or(0);
    let dense_end = dense_prefix_end(admitted_cached_tokens, config);
    let eligible = &prompt_ids[dense_end..];

    let scoring_start = Instant::now();
    let importance = score_tokens(draft, eligible)?;
    let draft_scoring_time = scoring_start.elapsed();
    let selected = select_target_indices(&importance, dense_end, prompt_ids.len(), config);

    let target_start = Instant::now();
    let (mut logits, cached_tokens) = model
        .specprefill_sparse_prefill(prompt_ids, prefix_reuse, dense_end, &selected)
        .map_err(anyhow::Error::msg)
        .context("sparse target prefill failed")?;
    mlxcel_core::eval(&logits);
    let target_prefill_time = target_start.elapsed();

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

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingOptions {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
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
    #[cfg(any(feature = "dflash2", test))]
    Dflash2(Dflash2PromptSnapshot),
}

impl PromptSnapshot {
    pub fn token_len(&self) -> usize {
        match self {
            Self::Baseline(snapshot) => snapshot.token_len(),
            Self::Mtp(snapshot) => snapshot.token_len(),
            #[cfg(any(feature = "dflash2", test))]
            Self::Dflash2(snapshot) => snapshot.token_len(),
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
    #[cfg(any(feature = "specprefill", test))]
    pub specprefill_stats: Option<SpecPrefillStats>,
}

pub struct PreparedMultimodalPrefill {
    pub prompt_ids: Vec<i32>,
    input_embeddings: UniquePtr<MlxArray>,
    position_ids: UniquePtr<MlxArray>,
    rope_delta: i32,
}

enum MtpPrompt<'a> {
    Text { prompt_ids: &'a [i32] },
    Multimodal(PreparedMultimodalPrefill),
}

struct IncrementalTextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    token_ids: Vec<u32>,
    emitted: String,
}

impl<'a> IncrementalTextDecoder<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            emitted: String::new(),
        }
    }

    fn push(&mut self, token_id: i32) -> Result<String> {
        self.token_ids
            .push(u32::try_from(token_id).context("generated a negative token identifier")?);
        let decoded = self
            .tokenizer
            .decode(&self.token_ids, false)
            .map_err(anyhow::Error::msg)
            .context("failed to incrementally decode generated tokens")?;
        self.advance(decoded, false)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35GenerationMode {
    Automatic,
    Baseline,
    Mtp,
    Dflash2,
}

/// Resolve one request to an explicit decoder. `Automatic` prefers available,
/// compatible DFlash2, then bundled MTP, then baseline. Explicit modes fail
/// when the requested decoder is unavailable or incompatible.
pub fn select_qwen35_decoder(
    mode: Qwen35GenerationMode,
    mtp_available: bool,
    dflash2_available: bool,
    dflash2_compatible: bool,
) -> std::result::Result<Qwen35GenerationMode, String> {
    match mode {
        Qwen35GenerationMode::Automatic => {
            if dflash2_available && dflash2_compatible {
                Ok(Qwen35GenerationMode::Dflash2)
            } else if mtp_available {
                Ok(Qwen35GenerationMode::Mtp)
            } else {
                Ok(Qwen35GenerationMode::Baseline)
            }
        }
        Qwen35GenerationMode::Baseline => Ok(Qwen35GenerationMode::Baseline),
        Qwen35GenerationMode::Mtp if mtp_available => Ok(Qwen35GenerationMode::Mtp),
        Qwen35GenerationMode::Mtp => {
            Err("the loaded checkpoint does not contain a bundled Qwen 3.5 MTP head".to_string())
        }
        Qwen35GenerationMode::Dflash2 if !dflash2_available => {
            Err("the DFlash2 draft checkpoint is unavailable".to_string())
        }
        Qwen35GenerationMode::Dflash2 if !dflash2_compatible => {
            Err("DFlash2 supports only unconstrained text generation".to_string())
        }
        Qwen35GenerationMode::Dflash2 => Ok(Qwen35GenerationMode::Dflash2),
    }
}

pub struct Qwen35Provider {
    model: Qwen35Model,
    tokenizer: Tokenizer,
    #[cfg(any(feature = "specprefill", test))]
    specprefill_draft: Option<Qwen35Model>,
    chat_template: ChatTemplateProcessor,
    defaults: GenerationDefaults,
    generator: CxxGenerator,
    mtp_generator: Option<Qwen35MtpGenerator>,
    #[cfg(any(feature = "dflash2", test))]
    dflash2_generator: Option<crate::qwen3_5_dflash::Qwen35Dflash2Generator>,
    vision_processor: Option<QwenVLProcessor>,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(i32),
    Many(Vec<i32>),
}

struct GenerationDefaults {
    stop_token_ids: Vec<i32>,
}

impl GenerationDefaults {
    fn sampling(&self, enable_thinking: bool, options: SamplingOptions) -> SamplingConfig {
        // Qwen/Qwen3.8-27B README, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0:
        // https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md
        // Quantized generation_config.json is not the source of this mode-dependent policy.
        let (temperature, top_p, top_k, presence_penalty) = if enable_thinking {
            (1.0, 0.95, 20, 0.0)
        } else {
            (0.7, 0.8, 20, 1.5)
        };
        SamplingConfig {
            temperature: options.temperature.unwrap_or(temperature),
            top_k: options.top_k.unwrap_or(top_k),
            top_p: options.top_p.unwrap_or(top_p),
            min_p: options.min_p.unwrap_or(0.0),
            presence_penalty: options.presence_penalty.unwrap_or(presence_penalty),
            repetition_penalty: options.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: options.frequency_penalty.unwrap_or(0.0),
            seed: options.seed,
            stop_token_ids: self.stop_token_ids.clone(),
            ..SamplingConfig::default()
        }
    }
}

impl Qwen35Provider {
    pub fn load(model_dir: impl AsRef<Path>, kv_cache_mode: KVCacheMode) -> Result<Self> {
        #[cfg(feature = "specprefill")]
        {
            let draft_model_dir = crate::resolve_specprefill_draft_path(None)?;
            return Self::load_with_specprefill_draft(model_dir, &draft_model_dir, kv_cache_mode)
                .with_context(|| {
                    format!(
                        "failed to load required SpecPrefill draft at {}; rerun `qw download {}`",
                        draft_model_dir.display(),
                        crate::DEFAULT_MODEL_IDENTIFIER
                    )
                });
        }
        #[cfg(not(feature = "specprefill"))]
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
        let model = Qwen35Model::load(model_dir, kv_cache_mode)?;
        if model.has_vision() {
            ensure!(
                chat_template.supports_image_content(),
                "unsupported Qwen3.5-VL chat template: expected image/vision marker behavior"
            );
        }
        let vision_processor = model
            .vision_config()
            .map(|vision| load_vision_processor(model_dir, vision))
            .transpose()?;
        let generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode);
        let mtp_generator = model.has_mtp().then(Qwen35MtpGenerator::new);

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            defaults,
            generator,
            #[cfg(any(feature = "specprefill", test))]
            specprefill_draft: None,
            mtp_generator,
            #[cfg(any(feature = "dflash2", test))]
            dflash2_generator: None,
            vision_processor,
        })
    }

    #[cfg(any(feature = "specprefill", test))]
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

    pub fn supports_image_inputs(&self) -> bool {
        self.vision_processor.is_some() && self.model.has_vision()
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

    #[tracing::instrument(
        name = "runtime.prepare_image",
        skip(self, rgb),
        fields(width, height, input_bytes = rgb.len()),
        err
    )]
    pub fn prepare_image(&self, width: u32, height: u32, rgb: Vec<u8>) -> Result<PreparedImage> {
        self.vision_processor
            .as_ref()
            .context("model does not support image inputs")?
            .prepare_rgb_bytes(width, height, rgb)
    }

    #[tracing::instrument(
        name = "runtime.prepare_multimodal_prefill",
        skip_all,
        fields(
            message_count = messages.len(),
            tool_count = tools.len(),
            image_count = images.len(),
            enable_thinking,
        ),
        err
    )]
    pub fn prepare_multimodal_prefill(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        reasoning_effort: Option<&str>,
        enable_thinking: bool,
        images: &[PreparedImage],
    ) -> Result<PreparedMultimodalPrefill> {
        ensure!(
            !images.is_empty(),
            "image prefill requires at least one image"
        );
        ensure!(
            self.supports_image_inputs(),
            "model does not support image inputs"
        );
        let declared_images = messages.iter().flat_map(ChatMessage::image_urls).count();
        ensure!(
            declared_images == images.len(),
            "prepared image count does not match rendered image count"
        );
        let vision_config = self
            .model
            .vision_config()
            .context("model does not support image inputs")?;
        let (image_token_id, video_token_id, vision_start_token_id) = self
            .model
            .multimodal_token_ids()
            .context("model does not support image inputs")?;
        let grids = images
            .iter()
            .map(|image| image.grid_thw)
            .collect::<Vec<_>>();
        let mut prompt_ids =
            self.tokenize_messages(messages, tools, reasoning_effort, enable_thinking)?;
        let expansion = insert_qwen_vl_image_tokens(
            &mut prompt_ids,
            &grids,
            vision_config.spatial_merge_size,
            vision_start_token_id,
            image_token_id,
        )?;
        let input_ids = mlxcel_core::from_slice_i32(&prompt_ids, &[1, prompt_ids.len() as i32]);
        let text_embeddings = self
            .model
            .embed_tokens(&input_ids)
            .context("Qwen3.5 input embeddings are unavailable")?;
        let mut pixel_values = images[0].to_mlx();
        for image in &images[1..] {
            pixel_values = mlxcel_core::concatenate(&pixel_values, &image.to_mlx(), 0);
        }
        let pixel_values =
            mlxcel_core::astype(&pixel_values, mlxcel_core::array_dtype(&text_embeddings));
        let vision_features = self.model.encode_vision(&pixel_values, &grids)?;
        ensure!(
            mlxcel_core::array_shape(&vision_features)[0] as usize == expansion.total_image_tokens,
            "vision encoder output count does not match expanded image token count"
        );
        let input_embeddings = merge_llava(
            image_token_id,
            &vision_features,
            &text_embeddings,
            &input_ids,
        )?;
        let positions = compute_rope_index(
            &prompt_ids,
            &grids,
            vision_config.spatial_merge_size,
            image_token_id,
            video_token_id,
        )?;
        let position_ids = positions.to_mlx();
        debug!(
            phase = "multimodal_prefill.complete",
            prompt_tokens = prompt_ids.len(),
            image_tokens = expansion.total_image_tokens,
        );
        Ok(PreparedMultimodalPrefill {
            prompt_ids,
            input_embeddings,
            position_ids,
            rope_delta: positions.rope_delta,
        })
    }

    pub fn supports_qwen35_tool_calls(&self) -> bool {
        self.chat_template.supports_qwen35_tool_calls()
    }

    pub fn baseline_sampling(
        &self,
        enable_thinking: bool,
        options: SamplingOptions,
    ) -> SamplingConfig {
        self.defaults.sampling(enable_thinking, options)
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
        #[cfg(any(feature = "specprefill", test))] prefill_mode: PrefillMode,
        mut on_delta: F,
    ) -> Result<BaselineGeneration> {
        #[cfg(any(feature = "specprefill", test))]
        if let PrefillMode::SpecPrefill(config) = prefill_mode {
            config.validate(prompt_ids.len())?;
            ensure!(
                constraint.is_none(),
                "SpecPrefill does not support token constraints"
            );
            ensure!(
                !prompt_ids.is_empty(),
                "prompt token sequence must not be empty"
            );
            ensure!(max_tokens > 0, "max_tokens must be greater than zero");
            let draft = self.specprefill_draft.as_ref().context(
                "SpecPrefill capability is unavailable because no draft model is loaded",
            )?;
            let requested_cached = prefix_reuse.as_ref().map_or(0, |reuse| reuse.cached_tokens);
            let reusable = prefix_reuse.as_ref().is_some_and(|reuse| {
                reuse.cached_tokens > 0
                    && reuse.cached_tokens <= prompt_ids.len()
                    && reuse.snapshot.token_len() == reuse.cached_tokens
                    && (reuse.cached_tokens < prompt_ids.len()
                        || reuse.snapshot.continuation_logits().is_some())
            });
            let dense_end =
                dense_prefix_end(reusable.then_some(requested_cached).unwrap_or(0), config);
            if should_activate(prompt_ids.len() - dense_end, config) {
                let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
                let mut decode_error = None;
                let mut callback_active = true;
                let sparse = generate_specprefill_tokens(
                    &self.model,
                    draft,
                    prompt_ids,
                    max_tokens,
                    sampling,
                    prefix_reuse,
                    config,
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
                return Ok(BaselineGeneration {
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
                    #[cfg(any(feature = "specprefill", test))]
                    specprefill_stats: Some(sparse.stats),
                });
            }
        }
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
            #[cfg(any(feature = "specprefill", test))]
            specprefill_stats: None,
        })
    }

    #[tracing::instrument(
        name = "runtime.generate_multimodal",
        skip_all,
        fields(
            prompt_tokens = prefill.prompt_ids.len(),
            max_tokens,
            constrained = constraint.is_some(),
        ),
        err
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_multimodal_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prefill: PreparedMultimodalPrefill,
        max_tokens: usize,
        sampling: &SamplingConfig,
        constraint: Option<&mut dyn TokenConstraint>,
        mut on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.model
            .prepare_mrope(&prefill.position_ids, prefill.rope_delta);
        let buffer_output = constraint.is_some();
        let mut callback_active = true;
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let controlled = self
            .generator
            .generate_streaming_controlled_with_embeddings(
                &self.model,
                &prefill.prompt_ids,
                Some(&prefill.input_embeddings),
                None,
                None,
                max_tokens,
                sampling,
                constraint,
                |token_id| {
                    if buffer_output {
                        callback_active = on_delta("");
                        return callback_active;
                    }
                    match decoder.push(token_id) {
                        Ok(delta) => {
                            callback_active = on_delta(&delta);
                            callback_active
                        }
                        Err(error) => {
                            decode_error = Some(error);
                            false
                        }
                    }
                },
            )
            .map_err(anyhow::Error::msg)
            .context("multimodal generation failed")?;
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
        let completion_tokens = controlled.token_ids.len();
        let prefill_time = controlled.prefill_time;
        let decode_time = controlled.decode_time;
        log_generation_metrics(
            "multimodal",
            prefill.prompt_ids.len(),
            completion_tokens,
            0,
            prefill_time,
            decode_time,
        );
        debug!(
            phase = "model.complete",
            prompt_tokens = prefill.prompt_ids.len(),
            completion_tokens,
            cached_tokens = 0,
            stop_reason = ?controlled.stop_reason,
        );
        Ok(BaselineGeneration {
            text: decoder.emitted,
            token_ids: controlled.token_ids,
            prompt_tokens: prefill.prompt_ids.len(),
            completion_tokens,
            cached_tokens: 0,
            finish_outcome: controlled.stop_reason,
            prompt_snapshots: Vec::new(),
            final_snapshot: None,
            prefill_time,
            decode_time,
            #[cfg(any(feature = "specprefill", test))]
            specprefill_stats: None,
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
        capture_final_snapshot: bool,
    ) -> Result<BaselineGeneration> {
        self.generate_mtp_streaming_for_prompt(
            MtpPrompt::Text { prompt_ids },
            max_tokens,
            sampling,
            block_size,
            prefix_reuse,
            checkpoint_token_lengths,
            constraint,
            on_delta,
            capture_final_snapshot,
        )
        .map(|(generation, _)| generation)
    }

    pub fn generate_mtp_multimodal_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prefill: PreparedMultimodalPrefill,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        constraint: Option<&mut dyn TokenConstraint>,
        on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.generate_mtp_streaming_for_prompt(
            MtpPrompt::Multimodal(prefill),
            max_tokens,
            sampling,
            block_size,
            None,
            &[],
            constraint,
            on_delta,
            false,
        )
        .map(|(generation, _)| generation)
    }

    #[cfg(any(feature = "dflash2", test))]
    pub fn prepare_dflash2_prefix(
        &mut self,
        prompt_ids: &[i32],
        draft_dir: &Path,
    ) -> Result<Dflash2PromptSnapshot> {
        if self.dflash2_generator.is_none() {
            self.dflash2_generator = Some(
                crate::qwen3_5_dflash::Qwen35Dflash2Generator::new(&self.model, draft_dir)
                    .map_err(|error| anyhow::anyhow!("failed to load DFlash2 drafter: {error}"))?,
            );
        }
        self.dflash2_generator
            .as_mut()
            .expect("DFlash2 generator was initialized")
            .capture_prompt_snapshot(&self.model, prompt_ids)
            .map_err(anyhow::Error::msg)
            .context("failed to capture DFlash2 prefix")
    }

    #[cfg(any(feature = "dflash2", test))]
    /// Generate with the DFlash2 block-diffusion drafter loaded from
    /// `draft_dir`, decoding deltas through the provider tokenizer.
    ///
    /// The generator is constructed lazily on first use and kept for
    /// subsequent calls, supporting greedy and stochastic sampling.
    #[tracing::instrument(name = "runtime.generate_dflash2", skip_all, fields(max_tokens), err)]
    pub fn generate_dflash2_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        draft_dir: &Path,
        on_delta: F,
    ) -> Result<(GenerationOutput, Dflash2GenerationStats)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let (output, stats, _) = self.generate_dflash2_cached_streaming(
            &prompt_ids,
            request.max_tokens,
            &sampling,
            draft_dir,
            None,
            on_delta,
        )?;
        Ok((output, stats))
    }

    #[cfg(any(feature = "dflash2", test))]
    pub fn generate_dflash2_cached_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        draft_dir: &Path,
        prefix_reuse: Option<Dflash2PrefixReuse<'_>>,
        on_delta: F,
    ) -> Result<(GenerationOutput, Dflash2GenerationStats, usize)> {
        self.generate_dflash2_cached_generation(
            prompt_ids,
            max_tokens,
            sampling,
            draft_dir,
            prefix_reuse,
            &[],
            false,
            on_delta,
        )
        .map(|(generation, stats)| {
            (
                GenerationOutput {
                    text: generation.text,
                    token_ids: generation.token_ids,
                },
                stats,
                generation.cached_tokens,
            )
        })
    }

    #[cfg(any(feature = "dflash2", test))]
    pub fn generate_dflash2_baseline_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        draft_dir: &Path,
        prefix_reuse: Option<Dflash2PrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        capture_final_snapshot: bool,
        on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.generate_dflash2_cached_generation(
            prompt_ids,
            max_tokens,
            sampling,
            draft_dir,
            prefix_reuse,
            checkpoint_token_lengths,
            capture_final_snapshot,
            on_delta,
        )
        .map(|(generation, _)| generation)
    }

    #[cfg(any(feature = "dflash2", test))]
    fn generate_dflash2_cached_generation<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        draft_dir: &Path,
        prefix_reuse: Option<Dflash2PrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        capture_final_snapshot: bool,
        mut on_delta: F,
    ) -> Result<(BaselineGeneration, Dflash2GenerationStats)> {
        if self.dflash2_generator.is_none() {
            self.dflash2_generator = Some(
                crate::qwen3_5_dflash::Qwen35Dflash2Generator::new(&self.model, draft_dir)
                    .map_err(|error| anyhow::anyhow!("failed to load DFlash2 drafter: {error}"))?,
            );
        }
        let generator = self
            .dflash2_generator
            .as_mut()
            .expect("DFlash2 generator was initialized");
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut callback_active = true;
        let mut decode_error = None;
        let generation = generator
            .generate_streaming(
                &self.model,
                prompt_ids,
                max_tokens,
                sampling,
                prefix_reuse,
                checkpoint_token_lengths,
                capture_final_snapshot,
                |token_id| match decoder.push(token_id) {
                    Ok(delta) => {
                        callback_active = on_delta(&delta);
                        callback_active
                    }
                    Err(error) => {
                        decode_error = Some(error);
                        false
                    }
                },
            )
            .map_err(anyhow::Error::msg)
            .context("DFlash2 generation failed")?;
        if let Some(error) = decode_error {
            return Err(error);
        }
        let final_delta = decoder.finish()?;
        if callback_active && !final_delta.is_empty() {
            let _ = on_delta(&final_delta);
        }
        let completion_tokens = generation.token_ids.len();
        let stats = generation.stats;
        debug!(phase = "dflash2.completed", ?stats);
        let stop_reason = generation.stop_reason;
        log_generation_metrics(
            "dflash2",
            prompt_ids.len(),
            completion_tokens,
            generation.cached_tokens,
            stats.prefill_time,
            stats.decode_time,
        );
        Ok((
            BaselineGeneration {
                text: decoder.emitted,
                token_ids: generation.token_ids,
                prompt_tokens: prompt_ids.len(),
                completion_tokens,
                cached_tokens: generation.cached_tokens,
                finish_outcome: stop_reason,
                prompt_snapshots: generation
                    .prompt_snapshots
                    .into_iter()
                    .map(PromptSnapshot::Dflash2)
                    .collect(),
                final_snapshot: generation.final_snapshot.map(PromptSnapshot::Dflash2),
                prefill_time: stats.prefill_time,
                decode_time: stats.decode_time,
                #[cfg(any(feature = "specprefill", test))]
                specprefill_stats: None,
            },
            stats,
        ))
    }

    #[tracing::instrument(
        name = "runtime.generate_mtp",
        skip_all,
        fields(max_tokens, block_size),
        err
    )]

    fn generate_mtp_streaming_for_prompt<F: FnMut(&str) -> bool>(
        &mut self,
        prompt: MtpPrompt<'_>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        constraint: Option<&mut dyn TokenConstraint>,
        mut on_delta: F,
        capture_final_snapshot: bool,
    ) -> Result<(BaselineGeneration, MtpGenerationStats)> {
        ensure!(block_size >= 2, "MTP block size must be at least 2");
        ensure!(
            self.mtp_generator.is_some(),
            "the loaded checkpoint does not contain a bundled Qwen 3.5 MTP head"
        );
        let buffer_output = constraint.is_some();
        let prompt_tokens = match &prompt {
            MtpPrompt::Text { prompt_ids } => prompt_ids.len(),
            MtpPrompt::Multimodal(prefill) => prefill.prompt_ids.len(),
        };
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut callback_active = true;
        let generator = self
            .mtp_generator
            .as_mut()
            .expect("MTP capability was validated");
        let generated = match prompt {
            MtpPrompt::Text { prompt_ids } => generator.generate_streaming(
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
                capture_final_snapshot,
            ),
            MtpPrompt::Multimodal(prefill) => generator.generate_streaming_with_embeddings(
                &self.model,
                &prefill.prompt_ids,
                &prefill.input_embeddings,
                &prefill.position_ids,
                prefill.rope_delta,
                max_tokens,
                sampling,
                block_size,
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
                capture_final_snapshot,
            ),
        }
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
            mtp_draft_materializations = generated.stats.draft_materializations,
            mtp_adaptive_stop_rounds = generated.stats.adaptive_stop_rounds,
            mtp_adaptive_skipped_draft_tokens =
                generated.stats.adaptive_skipped_draft_tokens,
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
                #[cfg(any(feature = "specprefill", test))]
                specprefill_stats: None,
            },
            generated.stats,
        ))
    }

    pub fn generate_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        on_delta: F,
    ) -> Result<GenerationOutput> {
        #[cfg(feature = "dflash2")]
        let draft_dir = crate::resolve_dflash2_draft_path(None).ok();
        #[cfg(not(feature = "dflash2"))]
        let draft_dir: Option<std::path::PathBuf> = None;
        self.generate_streaming_with_decoder(
            request,
            Qwen35GenerationMode::Automatic,
            draft_dir.as_deref(),
            on_delta,
        )
    }

    pub fn generate_streaming_with_decoder<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        mode: Qwen35GenerationMode,
        dflash2_draft_dir: Option<&Path>,
        on_delta: F,
    ) -> Result<GenerationOutput> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let dflash2_available =
            cfg!(feature = "dflash2") && dflash2_draft_dir.is_some_and(Path::is_dir);
        let decoder =
            select_qwen35_decoder(mode, self.mtp_generator.is_some(), dflash2_available, true)
                .map_err(anyhow::Error::msg)?;
        debug!(
            phase = "decoder.selected",
            decoder = ?decoder,
            configured_decoder = ?mode,
            prompt_tokens = prompt_ids.len(),
            dflash2_available,
            enable_thinking = request.enable_thinking,
            reasoning_effort = ?request.reasoning_effort,
            temperature = sampling.temperature,
            top_k = sampling.top_k,
            top_p = sampling.top_p,
            min_p = sampling.min_p,
            presence_penalty = sampling.presence_penalty,
            repetition_penalty = sampling.repetition_penalty,
            frequency_penalty = sampling.frequency_penalty,
        );
        let generation = match decoder {
            Qwen35GenerationMode::Baseline => self.generate_baseline_streaming(
                &prompt_ids,
                request.max_tokens,
                &sampling,
                None,
                None,
                &[],
                #[cfg(any(feature = "specprefill", test))]
                PrefillMode::Dense,
                on_delta,
            )?,
            Qwen35GenerationMode::Mtp => {
                self.generate_mtp_streaming_for_prompt(
                    MtpPrompt::Text {
                        prompt_ids: &prompt_ids,
                    },
                    request.max_tokens,
                    &sampling,
                    DEFAULT_MTP_BLOCK_SIZE,
                    None,
                    &[],
                    None,
                    on_delta,
                    false,
                )?
                .0
            }
            Qwen35GenerationMode::Dflash2 => {
                #[cfg(any(feature = "dflash2", test))]
                {
                    self.generate_dflash2_baseline_streaming(
                        &prompt_ids,
                        request.max_tokens,
                        &sampling,
                        dflash2_draft_dir.expect("selected DFlash2 has an available checkpoint"),
                        None,
                        &[],
                        false,
                        on_delta,
                    )?
                }
                #[cfg(not(any(feature = "dflash2", test)))]
                {
                    unreachable!("DFlash2 cannot be selected without its build feature")
                }
            }
            Qwen35GenerationMode::Automatic => {
                unreachable!("automatic decoder selection always returns an explicit decoder")
            }
        };
        Ok(GenerationOutput {
            text: generation.text,
            token_ids: generation.token_ids,
        })
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
        mode: Qwen35GenerationMode,
        on_delta: F,
    ) -> Result<(GenerationOutput, Option<MtpGenerationStats>)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let use_mtp = self.resolve_generation_mode(mode)?
            && !(mode == Qwen35GenerationMode::Automatic && request.max_tokens == 1);
        if !use_mtp {
            let generation = self.generate_baseline_streaming(
                &prompt_ids,
                request.max_tokens,
                &sampling,
                None,
                None,
                &[],
                #[cfg(any(feature = "specprefill", test))]
                PrefillMode::Dense,
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
            MtpPrompt::Text {
                prompt_ids: &prompt_ids,
            },
            request.max_tokens,
            &sampling,
            DEFAULT_MTP_BLOCK_SIZE,
            None,
            &[],
            None,
            on_delta,
            false,
        )?;
        Ok((
            GenerationOutput {
                text: generation.text,
                token_ids: generation.token_ids,
            },
            Some(stats),
        ))
    }
    /// Controlled decode benchmark route. Production CLI/server callers use
    /// [`Self::generate_streaming_with_decoder`] and the request-level router.
    #[doc(hidden)]
    pub fn benchmark_streaming_in_mode<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        mode: Qwen35GenerationMode,
        on_delta: F,
    ) -> Result<(GenerationOutput, Duration, Option<MtpGenerationStats>)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        if !self.resolve_generation_mode(mode)? {
            let generation = self.generate_baseline_streaming(
                &prompt_ids,
                request.max_tokens,
                &sampling,
                None,
                None,
                &[],
                #[cfg(any(feature = "specprefill", test))]
                PrefillMode::Dense,
                on_delta,
            )?;
            return Ok((
                GenerationOutput {
                    text: generation.text,
                    token_ids: generation.token_ids,
                },
                generation.decode_time,
                None,
            ));
        }
        let (generation, stats) = self.generate_mtp_streaming_for_prompt(
            MtpPrompt::Text {
                prompt_ids: &prompt_ids,
            },
            request.max_tokens,
            &sampling,
            DEFAULT_MTP_BLOCK_SIZE,
            None,
            &[],
            None,
            on_delta,
            false,
        )?;
        Ok((
            GenerationOutput {
                text: generation.text,
                token_ids: generation.token_ids,
            },
            generation.decode_time,
            Some(stats),
        ))
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
        mode: Qwen35GenerationMode,
        on_delta: F,
    ) -> Result<(BaselineGeneration, Option<MtpGenerationStats>)> {
        match (mode, snapshot) {
            (Qwen35GenerationMode::Baseline, snapshot) => {
                let snapshot = match snapshot {
                    PromptSnapshot::Baseline(snapshot) => snapshot,
                    PromptSnapshot::Mtp(snapshot) => snapshot.target_snapshot(),
                    #[cfg(any(feature = "dflash2", test))]
                    PromptSnapshot::Dflash2(snapshot) => snapshot.target_snapshot(),
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
                    #[cfg(any(feature = "specprefill", test))]
                    PrefillMode::Dense,
                    on_delta,
                )?;
                Ok((generation, None))
            }
            (Qwen35GenerationMode::Mtp, PromptSnapshot::Mtp(snapshot)) => {
                let (generation, stats) = self.generate_mtp_streaming_for_prompt(
                    MtpPrompt::Text { prompt_ids },
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
                    false,
                )?;
                Ok((generation, Some(stats)))
            }
            (Qwen35GenerationMode::Automatic, _) => {
                anyhow::bail!("cached benchmark mode must be explicit")
            }
            (Qwen35GenerationMode::Dflash2, _) => {
                anyhow::bail!("cached DFlash2 benchmarks use the dedicated DFlash2 entrypoint")
            }
            (Qwen35GenerationMode::Mtp, PromptSnapshot::Baseline(_)) => {
                anyhow::bail!("cached benchmark mode does not match the snapshot family")
            }
            #[cfg(any(feature = "dflash2", test))]
            (Qwen35GenerationMode::Mtp, PromptSnapshot::Dflash2(_)) => {
                anyhow::bail!("cached benchmark mode does not match the snapshot family")
            }
        }
    }

    fn resolve_generation_mode(&self, mode: Qwen35GenerationMode) -> Result<bool> {
        match mode {
            Qwen35GenerationMode::Automatic => Ok(self.mtp_generator.is_some()),
            Qwen35GenerationMode::Baseline => Ok(false),
            Qwen35GenerationMode::Mtp => {
                ensure!(
                    self.mtp_generator.is_some(),
                    "the loaded checkpoint does not contain a bundled Qwen 3.5 MTP head"
                );
                Ok(true)
            }
            Qwen35GenerationMode::Dflash2 => {
                anyhow::bail!("DFlash2 generation requires a draft checkpoint path")
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

        let rendered = self.chat_template.render_messages(
            &[ChatMessage {
                role: "user".to_string(),
                name: None,
                content: Some(ChatMessageContent::Text(request.prompt.clone())),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            &[],
            request.reasoning_effort.as_deref(),
            request.enable_thinking,
            true,
        )?;
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

        let mut sampling = self.baseline_sampling(
            request.enable_thinking,
            SamplingOptions {
                temperature: request.temperature,
                top_k: request.top_k,
                top_p: request.top_p,
                min_p: request.min_p,
                presence_penalty: request.presence_penalty,
                repetition_penalty: request.repetition_penalty,
                frequency_penalty: request.frequency_penalty,
                seed: request.seed,
            },
        );
        sampling.prompt_token_count = Some(prompt_ids.len());
        Ok((prompt_ids, sampling))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WiredLimitPolicy {
    system_memory_bytes: u64,
    metal_recommended_bytes: u64,
    wired_limit_bytes: u64,
}

fn configure_metal_wired_limit_with(
    physical_memory_bytes: impl FnOnce() -> Option<u64>,
    metal_recommended_bytes: impl FnOnce() -> u64,
    set_wired_limit: impl FnOnce(u64) -> std::result::Result<(), String>,
) -> std::result::Result<WiredLimitPolicy, String> {
    let system_memory_bytes = physical_memory_bytes()
        .ok_or_else(|| "failed to detect physical system memory via hw.memsize".to_string())?;
    if system_memory_bytes == 0 {
        return Err("physical system memory detection returned zero bytes".to_string());
    }

    let metal_recommended_bytes = metal_recommended_bytes();
    if metal_recommended_bytes == 0 {
        return Err(
            "Metal max_recommended_working_set_size detection returned zero bytes".to_string(),
        );
    }

    let wired_limit_bytes =
        mlxcel_core::memory::recommended_wired_limit(system_memory_bytes, metal_recommended_bytes);
    if wired_limit_bytes == 0 {
        return Err("wired-memory policy computed a zero-byte limit".to_string());
    }
    set_wired_limit(wired_limit_bytes).map_err(|error| {
        format!(
            "failed to configure MLX Metal wired-memory limit to {wired_limit_bytes} bytes: {error}"
        )
    })?;

    Ok(WiredLimitPolicy {
        system_memory_bytes,
        metal_recommended_bytes,
        wired_limit_bytes,
    })
}

const METAL_CACHE_LIMIT_ENV: &str = "MLXCEL_CACHE_LIMIT";
const DEFAULT_METAL_CACHE_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

// This allowance covers reusable free MLX buffers, not live tensors or prefix
// cache entries. Parse before changing allocator policy so bad overrides fail
// initialization rather than silently selecting a different memory budget.
fn parse_metal_cache_limit(raw: Option<&std::ffi::OsStr>) -> std::result::Result<u64, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_METAL_CACHE_LIMIT_BYTES);
    };
    let value = raw.to_str().ok_or_else(|| {
        format!("{METAL_CACHE_LIMIT_ENV} must contain a Unicode unsigned byte count")
    })?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "{METAL_CACHE_LIMIT_ENV} must be an unsigned decimal byte count (0 disables free-buffer caching), got {value:?}"
        ));
    }
    value.parse::<usize>().map(|bytes| bytes as u64).map_err(|error| {
        format!("{METAL_CACHE_LIMIT_ENV} byte count {value:?} exceeds the allocator's supported range: {error}")
    })
}

fn initialize_runtime() -> Result<()> {
    static INITIALIZED: LazyLock<std::result::Result<(), String>> = LazyLock::new(|| {
        if !mlxcel_core::metal_is_available() {
            return Err("the MLX Metal backend is unavailable on this host".to_string());
        }
        let cache_limit_bytes =
            parse_metal_cache_limit(std::env::var_os(METAL_CACHE_LIMIT_ENV).as_deref())?;
        let policy = configure_metal_wired_limit_with(
            mlxcel_core::hardware::physical_memory_bytes,
            mlxcel_core::memory::metal_recommended_working_set_size,
            |bytes| mlxcel_core::memory::set_wired_limit(bytes).map(|_| ()),
        )?;
        info!(
            system_memory_bytes = policy.system_memory_bytes,
            metal_recommended_bytes = policy.metal_recommended_bytes,
            wired_limit_bytes = policy.wired_limit_bytes,
            "configured MLX Metal wired-memory limit"
        );
        mlxcel_core::set_default_device(true);
        mlxcel_core::memory::set_cache_limit(cache_limit_bytes);
        info!(
            cache_limit_bytes,
            "configured MLX Metal free-buffer cache allowance"
        );
        Ok(())
    });
    (*INITIALIZED).clone().map_err(anyhow::Error::msg)
}

fn load_vision_processor(
    model_dir: &Path,
    config: &crate::qwen3_vl_vision::Qwen3VLVisionConfig,
) -> Result<QwenVLProcessor> {
    let factor = config.patch_size * config.spatial_merge_size;
    let default_min = 4 * factor * factor;
    let default_max = 16_384 * factor * factor;
    let path = model_dir.join("preprocessor_config.json");
    let value = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<serde_json::Value>(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            serde_json::Value::Object(Default::default())
        }
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let min_pixels = value
        .get("min_pixels")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default_min);
    let max_pixels = value
        .get("max_pixels")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default_max);
    QwenVLProcessor::new(
        config.patch_size,
        config.temporal_patch_size,
        config.spatial_merge_size,
        min_pixels,
        max_pixels,
    )
}

fn load_generation_defaults(model_dir: &Path) -> Result<GenerationDefaults> {
    let path: PathBuf = model_dir.join("generation_config.json");
    let config = if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str::<GenerationConfig>(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?
    } else {
        GenerationConfig { eos_token_id: None }
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

    Ok(GenerationDefaults { stop_token_ids })
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

    #[test]
    fn decoder_preference_and_explicit_unavailability() {
        use Qwen35GenerationMode::*;
        assert_eq!(
            select_qwen35_decoder(Automatic, true, true, true),
            Ok(Dflash2)
        );
        assert_eq!(select_qwen35_decoder(Automatic, true, false, true), Ok(Mtp));
        assert_eq!(select_qwen35_decoder(Automatic, true, true, false), Ok(Mtp));
        assert_eq!(
            select_qwen35_decoder(Automatic, false, true, false),
            Ok(Baseline)
        );
        assert_eq!(
            select_qwen35_decoder(Automatic, false, false, true),
            Ok(Baseline)
        );
        assert_eq!(
            select_qwen35_decoder(Baseline, true, true, true),
            Ok(Baseline)
        );
        assert_eq!(select_qwen35_decoder(Mtp, true, true, true), Ok(Mtp));
        assert_eq!(
            select_qwen35_decoder(Dflash2, false, true, true),
            Ok(Dflash2)
        );
        assert!(select_qwen35_decoder(Mtp, false, true, true).is_err());
        assert!(select_qwen35_decoder(Dflash2, true, false, true).is_err());
        assert!(select_qwen35_decoder(Dflash2, true, true, false).is_err());
    }

    #[test]
    fn metal_cache_limit_resolves_default_and_explicit_byte_counts() {
        use std::ffi::OsStr;

        assert_eq!(
            parse_metal_cache_limit(None).unwrap(),
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(parse_metal_cache_limit(Some(OsStr::new("0"))).unwrap(), 0);
        assert_eq!(
            parse_metal_cache_limit(Some(OsStr::new("1048576"))).unwrap(),
            1048576
        );
        assert_eq!(
            parse_metal_cache_limit(Some(OsStr::new(&usize::MAX.to_string()))).unwrap(),
            usize::MAX as u64
        );
    }

    #[test]
    fn metal_cache_limit_rejects_invalid_explicit_values_instead_of_defaulting() {
        use std::ffi::OsStr;

        for value in [
            "",
            "-1",
            "+1",
            " 1",
            "1 ",
            "1.5",
            "2GiB",
            "18446744073709551616",
        ] {
            assert!(
                parse_metal_cache_limit(Some(OsStr::new(value))).is_err(),
                "accepted invalid cache limit {value:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn metal_cache_limit_rejects_non_unicode_instead_of_treating_it_as_unset() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        assert!(parse_metal_cache_limit(Some(OsStr::from_bytes(b"\xff"))).is_err());
    }

    #[test]
    fn wired_limit_initialization_applies_computed_policy() {
        use std::cell::Cell;

        const GIB: u64 = 1024 * 1024 * 1024;
        let applied = Cell::new(None);
        let policy = configure_metal_wired_limit_with(
            || Some(64 * GIB),
            || 40 * GIB,
            |bytes| {
                applied.set(Some(bytes));
                Ok(())
            },
        )
        .expect("valid wired-memory policy");

        assert_eq!(
            policy,
            WiredLimitPolicy {
                system_memory_bytes: 64 * GIB,
                metal_recommended_bytes: 40 * GIB,
                wired_limit_bytes: 44 * GIB,
            }
        );
        assert_eq!(applied.get(), Some(44 * GIB));
    }

    #[test]
    fn wired_limit_initialization_fails_closed_without_mutating_metal() {
        use std::cell::Cell;

        let setter_called = Cell::new(false);
        let error = configure_metal_wired_limit_with(
            || None,
            || 1,
            |_| {
                setter_called.set(true);
                Ok(())
            },
        )
        .expect_err("missing physical memory must fail");
        assert!(error.contains("hw.memsize"), "{error}");
        assert!(!setter_called.get());

        let error = configure_metal_wired_limit_with(
            || Some(0),
            || 1,
            |_| {
                setter_called.set(true);
                Ok(())
            },
        )
        .expect_err("zero physical memory must fail");
        assert!(error.contains("zero bytes"), "{error}");
        assert!(!setter_called.get());

        let error = configure_metal_wired_limit_with(|| Some(1), || 0, |_| Ok(()))
            .expect_err("zero Metal recommendation must fail");
        assert!(
            error.contains("max_recommended_working_set_size"),
            "{error}"
        );

        let error = configure_metal_wired_limit_with(
            || Some(1024),
            || 1024,
            |_| Err("backend rejected limit".to_string()),
        )
        .expect_err("setter failure must abort initialization");
        assert!(error.contains("failed to configure"), "{error}");
        assert!(error.contains("backend rejected limit"), "{error}");
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
    fn qwen38_policy_is_location_independent_and_respects_request_overrides() {
        let fixture = TestDir::new("sampling-policy");
        std::fs::write(
            fixture.0.join("generation_config.json"),
            br#"{"eos_token_id":[9,7,9],"temperature":0.2,"top_k":11,"top_p":0.4}"#,
        )
        .expect("write quantization defaults");
        let defaults = load_generation_defaults(&fixture.0).expect("load policy");
        for thinking in [true, false] {
            let automatic = defaults.sampling(thinking, SamplingOptions::default());
            assert_eq!(automatic.temperature, if thinking { 1.0 } else { 0.7 });
            assert_eq!(automatic.top_p, if thinking { 0.95 } else { 0.8 });
            assert_eq!(automatic.top_k, 20);
            assert_eq!(automatic.min_p, 0.0);
            assert_eq!(automatic.presence_penalty, if thinking { 0.0 } else { 1.5 });
            assert_eq!(automatic.repetition_penalty, 1.0);
            assert_eq!(automatic.frequency_penalty, 0.0);
            assert_eq!(automatic.stop_token_ids, vec![7, 9]);
            let partial = defaults.sampling(
                thinking,
                SamplingOptions {
                    temperature: Some(0.0),
                    presence_penalty: Some(0.0),
                    ..Default::default()
                },
            );
            assert_eq!(partial.temperature, 0.0);
            assert_eq!(partial.presence_penalty, 0.0);
            assert_eq!(partial.top_p, automatic.top_p);
            assert_eq!(partial.top_k, automatic.top_k);
            let explicit = defaults.sampling(
                thinking,
                SamplingOptions {
                    temperature: Some(0.6),
                    top_k: Some(0),
                    top_p: Some(1.0),
                    min_p: Some(0.2),
                    presence_penalty: Some(0.4),
                    repetition_penalty: Some(1.2),
                    frequency_penalty: Some(0.3),
                    seed: Some(7),
                },
            );
            assert_eq!(explicit.temperature, 0.6);
            assert_eq!(explicit.top_k, 0);
            assert_eq!(explicit.top_p, 1.0);
            assert_eq!(explicit.min_p, 0.2);
            assert_eq!(explicit.presence_penalty, 0.4);
            assert_eq!(explicit.repetition_penalty, 1.2);
            assert_eq!(explicit.frequency_penalty, 0.3);
            assert_eq!(explicit.seed, Some(7));
        }
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

        let error = match Qwen35Provider::load_target_only(&fixture.0, KVCacheMode::Fp16) {
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
            Qwen35Provider::load(&model_dir, KVCacheMode::Fp16).expect("load real Qwen checkpoint");
        let request = GenerationRequest {
            prompt: "Continue counting upward from one, writing each integer on its own line without stopping."
                .to_string(),
            max_tokens: 32,
            enable_thinking: true,
            reasoning_effort: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            frequency_penalty: None,
            temperature: Some(0.0),
            top_k: Some(1),
            top_p: Some(1.0),
            seed: Some(0),
        };
        let mut baseline_deltas = String::new();
        let (baseline, _) = provider
            .generate_streaming_in_mode(&request, Qwen35GenerationMode::Baseline, |delta| {
                baseline_deltas.push_str(delta);
                true
            })
            .expect("baseline greedy generation");
        let mut mtp_deltas = String::new();
        let (mtp, _) = provider
            .generate_streaming_in_mode(&request, Qwen35GenerationMode::Mtp, |delta| {
                mtp_deltas.push_str(delta);
                true
            })
            .expect("MTP greedy generation");
        assert_eq!(baseline.token_ids, mtp.token_ids);
        assert_eq!(baseline.text, mtp.text);
        assert_eq!(baseline_deltas, baseline.text);
        assert_eq!(mtp_deltas, mtp.text);
    }

    #[test]
    #[ignore = "requires real target and SpecPrefill draft checkpoints at their configured or default cache paths"]
    fn real_model_dense_specprefill_dense_has_no_position_state_leakage() {
        let model_dir = crate::resolve_model_path(None).expect("resolve target checkpoint");
        let draft_dir = crate::resolve_specprefill_draft_path(None)
            .expect("resolve SpecPrefill draft checkpoint");
        let mut provider =
            Qwen35Provider::load_with_specprefill_draft(model_dir, draft_dir, KVCacheMode::Fp16)
                .expect("load target and SpecPrefill draft");
        let prompt = provider
            .tokenizer
            .encode(
                "Operational record: all systems nominal; preserve this context. ".repeat(1024),
                true,
            )
            .expect("encode long prompt")
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let sampling = SamplingConfig::greedy();
        let dense_before = provider
            .generate_baseline_streaming(
                &prompt,
                8,
                &sampling,
                None,
                None,
                &[],
                PrefillMode::Dense,
                |_| true,
            )
            .expect("first dense generation");
        let sparse = provider
            .generate_baseline_streaming(
                &prompt,
                8,
                &sampling,
                None,
                None,
                &[],
                PrefillMode::SpecPrefill(SpecPrefillConfig {
                    min_tokens: 512,
                    keep_rate: 0.30,
                    protected_prefix_tokens: 0,
                    ..Default::default()
                }),
                |_| true,
            )
            .expect("sparse generation");
        assert!(sparse.specprefill_stats.is_some());
        let dense_after = provider
            .generate_baseline_streaming(
                &prompt,
                8,
                &sampling,
                None,
                None,
                &[],
                PrefillMode::Dense,
                |_| true,
            )
            .expect("second dense generation");
        assert_eq!(dense_before.token_ids, dense_after.token_ids);
    }
    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_mtp_prefix_reuse_matches_cold_and_reduces_ttft() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider = Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
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
        let sampling = provider.baseline_sampling(
            true,
            SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(0),
                ..Default::default()
            },
        );

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
                true,
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
                true,
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
                true,
            )
            .expect("warm MTP generation");

        let baseline_cold = provider
            .generate_baseline_streaming(
                &full_ids,
                32,
                &sampling,
                None,
                None,
                &[],
                PrefillMode::Dense,
                |_| true,
            )
            .expect("cold baseline generation");
        let mtp_snapshot = PromptSnapshot::Mtp(snapshot);
        let (baseline_warm, baseline_stats) = provider
            .benchmark_cached_streaming_in_mode(
                &full_ids,
                32,
                &sampling,
                &mtp_snapshot,
                Qwen35GenerationMode::Baseline,
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
        let mut provider = Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
            .expect("load real bundled-MTP checkpoint");
        let sampling = provider.baseline_sampling(
            true,
            SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(0),
                ..Default::default()
            },
        );
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
                    true,
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
                    true,
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
        let mut provider = Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
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
        let sampling = provider.baseline_sampling(
            true,
            SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(0),
                ..Default::default()
            },
        );
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
                false,
            )
            .expect("long MTP generation");
        let tail = last_callback
            .expect("long generation emitted a token")
            .elapsed();
        assert_eq!(callback_count, 1024);
        assert_eq!(generated.finish_outcome, GenerationStopReason::MaxTokens);
        assert!(generated.final_snapshot.is_none());
        eprintln!(
            "MTP terminal-tail benchmark: completion_tokens={}, callbacks={}, tail_ms={:.2}",
            generated.completion_tokens,
            callback_count,
            tail.as_secs_f64() * 1_000.0,
        );
        assert!(
            tail < Duration::from_secs(10),
            "uncached generation must not perform terminal snapshot work: tail={tail:?}"
        );
    }

    #[test]
    #[ignore = "requires the real bundled-MTP checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn real_model_mtp_eos_has_bounded_terminal_tail_and_resumable_snapshot() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let mut provider = Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
            .expect("load real MTP checkpoint");
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
        let sampling = provider.baseline_sampling(
            true,
            SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(0),
                ..Default::default()
            },
        );
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
                true,
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
                &[prompt_ids.len()],
                None,
                |token| {
                    callback_tokens.push(token);
                    last_callback = Some(Instant::now());
                    true
                },
                true,
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
            terminal_tail < Duration::from_secs(1),
            "terminal snapshot must reuse the prompt checkpoint: tail={terminal_tail:?}"
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
        let resume_sampling = provider.baseline_sampling(
            true,
            SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(0),
                ..Default::default()
            },
        );
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
                true,
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
                true,
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
        let mut provider = Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
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
        let sampling = provider.baseline_sampling(
            true,
            SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(7),
                ..Default::default()
            },
        );
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
                true,
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
                true,
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
                true,
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
                true,
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
                true,
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
                true,
            )
            .expect("resume portable MTP snapshot");
        let mut combined = interrupted.token_ids;
        combined.extend_from_slice(&resumed.token_ids);
        assert_eq!(combined, control.token_ids);
    }
}
