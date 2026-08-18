use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::{
    ControlledGeneration, CxxGenerator, GenerationStopReason, LanguageModel, ModelStateSnapshot,
    PrefixReuse, SamplingConfig, TokenConstraint,
};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::info;

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
use crate::qwen3_5_mtp::Qwen35MtpGenerator;
pub use crate::qwen3_5_mtp::{MtpGenerationStats, MtpPrefixReuse, MtpPromptSnapshot};

const DEFAULT_MTP_BLOCK_SIZE: usize = 3;

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
    /// Wall time strictly after the first sampled token.
    #[doc(hidden)]
    pub decode_time: Duration,
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

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35GenerationMode {
    Automatic,
    Baseline,
    Mtp,
}

pub struct Qwen35Provider {
    model: Qwen35Model,
    tokenizer: Tokenizer,
    chat_template: ChatTemplateProcessor,
    defaults: GenerationDefaults,
    generator: CxxGenerator,
    mtp_generator: Option<Qwen35MtpGenerator>,
    vision_processor: Option<QwenVLProcessor>,
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

impl Qwen35Provider {
    #[tracing::instrument(name = "runtime.model_load", skip(model_dir), err)]
    pub fn load(model_dir: impl AsRef<Path>, kv_cache_mode: KVCacheMode) -> Result<Self> {
        initialize_runtime()?;
        let model_dir = model_dir.as_ref();
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
            mtp_generator,
            vision_processor,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn logits_vocab_size(&self) -> usize {
        self.model.vocab_size()
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
            .render_messages(messages, tools, reasoning_effort, enable_thinking)
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
        info!(
            phase = "tokenization.complete",
            prompt_tokens = prompt_ids.len(),
        );
        Ok(prompt_ids)
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
        info!(
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
        capture_prompt_snapshot: bool,
        mut on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.model.clear_prepared_mrope();
        let buffer_output = constraint.is_some();
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut callback_active = true;
        let mut decode_start = None;
        let controlled: ControlledGeneration = self
            .generator
            .generate_streaming_controlled(
                &self.model,
                prompt_ids,
                prefix_reuse,
                max_tokens,
                sampling,
                constraint,
                capture_prompt_snapshot,
                |token_id| {
                    decode_start.get_or_insert_with(Instant::now);
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
        let decode_time = decode_start.map_or(Duration::ZERO, |start| start.elapsed());
        info!(
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
            decode_time,
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
        let mut decode_start = None;
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
                false,
                |token_id| {
                    decode_start.get_or_insert_with(Instant::now);
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
        info!(
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
            decode_time: decode_start.map_or(Duration::ZERO, |start| start.elapsed()),
        })
    }

    pub fn generate_mtp_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        capture_prompt_snapshot: bool,
        constraint: Option<&mut dyn TokenConstraint>,
        on_delta: F,
    ) -> Result<BaselineGeneration> {
        self.generate_mtp_streaming_for_prompt(
            MtpPrompt::Text { prompt_ids },
            max_tokens,
            sampling,
            block_size,
            prefix_reuse,
            capture_prompt_snapshot,
            constraint,
            on_delta,
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
            false,
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
        prompt: MtpPrompt<'_>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        capture_prompt_snapshot: bool,
        constraint: Option<&mut dyn TokenConstraint>,
        mut on_delta: F,
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
                capture_prompt_snapshot,
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
        info!(
            phase = "model.complete",
            prompt_tokens,
            completion_tokens,
            cached_tokens = generated.cached_tokens,
            stop_reason = ?generated.stop_reason,
            mtp_proposed_draft_tokens = generated.stats.proposed_draft_tokens,
            mtp_accepted_draft_tokens = generated.stats.accepted_draft_tokens,
            mtp_acceptance_percentage = generated.stats.acceptance_percentage(),
            mtp_decode_seconds = generated.stats.decode_time.as_secs_f64(),
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
                    .prompt_snapshot
                    .map(PromptSnapshot::Mtp)
                    .into_iter()
                    .collect(),
                decode_time: generated.stats.decode_time,
            },
            generated.stats,
        ))
    }

    pub fn generate_streaming<F: FnMut(&str) -> bool>(
        &mut self,
        request: &GenerationRequest,
        on_delta: F,
    ) -> Result<GenerationOutput> {
        self.generate_streaming_in_mode(request, Qwen35GenerationMode::Automatic, on_delta)
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
        mode: Qwen35GenerationMode,
        on_delta: F,
    ) -> Result<(GenerationOutput, Option<MtpGenerationStats>)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let use_mtp = self.resolve_generation_mode(mode)?;
        if !use_mtp {
            let generation = self.generate_baseline_streaming(
                &prompt_ids,
                request.max_tokens,
                &sampling,
                None,
                None,
                false,
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
            false,
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
                false,
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
            false,
            None,
            on_delta,
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

fn initialize_runtime() -> Result<()> {
    static INITIALIZED: LazyLock<std::result::Result<(), String>> = LazyLock::new(|| {
        if !mlxcel_core::metal_is_available() {
            return Err("the MLX Metal backend is unavailable on this host".to_string());
        }
        mlxcel_core::set_default_device(true);
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

        let error = match Qwen35Provider::load(&fixture.0, KVCacheMode::Fp16) {
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
    #[ignore = "requires QW_BENCH_MODEL pointing at a real bundled-MTP checkpoint"]
    fn real_model_baseline_and_mtp_greedy_outputs_match() {
        let model_dir = std::env::var_os("QW_BENCH_MODEL")
            .map(PathBuf::from)
            .expect("QW_BENCH_MODEL must point at a real checkpoint");
        let mut provider =
            Qwen35Provider::load(&model_dir, KVCacheMode::Fp16).expect("load real Qwen checkpoint");
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
    #[ignore = "requires QW_BENCH_MODEL pointing at a real bundled-MTP checkpoint"]
    fn real_model_mtp_prefix_reuse_matches_cold_and_reduces_ttft() {
        let model_dir = std::env::var_os("QW_BENCH_MODEL")
            .map(PathBuf::from)
            .expect("QW_BENCH_MODEL must point at a real checkpoint");
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
                false,
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
                true,
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
                }),
                true,
                None,
                |_| {
                    warm_ttft.get_or_insert_with(|| warm_started.elapsed());
                    true
                },
            )
            .expect("warm MTP generation");

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
    }
}
