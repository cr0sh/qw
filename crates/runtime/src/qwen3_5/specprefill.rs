// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

const SPECPREFILL_TARGET_CHUNK_TOKENS: usize = 512;

impl Qwen35DecoderLayer {
    fn forward_with_query_capture(
        &self,
        x: &MlxArray,
        cache: &mut Qwen3NextCache,
    ) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
        let normed = self.input_layernorm.forward(x);
        let (residual, query) = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attention), Qwen3NextCache::Linear(cache)) => {
                (attention.forward(&normed, None, Some(cache)), None)
            }
            (
                Qwen35AttentionVariant::FullAttention(attention),
                Qwen3NextCache::Attention(cache),
            ) => {
                let (output, query) =
                    attention.forward_with_query_capture(&normed, cache, None, None);
                (output, Some(query))
            }
            _ => unreachable!("Qwen3.5 layer/cache topology must match"),
        };
        let hidden = mlxcel_core::add(x, &residual);
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden));
        (mlxcel_core::add(&hidden, &mlp), query)
    }
}

impl Qwen35Model {
    pub(crate) fn load_specprefill_draft(model_dir: &Path) -> Result<Self> {
        use crate::model_resolver::DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER;

        ensure!(
            model_dir.is_dir(),
            "SpecPrefill draft model directory does not exist or is not a directory: {}",
            model_dir.display()
        );
        // `parse_config` has already required the checkpoint root
        // `model_type == "qwen3_5"`; the returned config is the nested text
        // architecture, whose pinned identifier is `qwen3_5_text`.
        let mut config = Self::parse_config(model_dir)?;
        let quantization = config.quantization.as_ref();
        ensure!(
            config.model_type == "qwen3_5_text"
                && config.num_hidden_layers == 24
                && config.full_attention_interval == 4
                && config.hidden_size == 1_024
                && config.num_attention_heads == 8
                && config.num_key_value_heads == 2
                && config.vocab_size == 248_320
                && config.max_position_embeddings == 262_144
                && quantization
                    .is_some_and(|q| { q.mode == "affine" && q.bits == 8 && q.group_size == 64 }),
            "checkpoint {} does not match pinned SpecPrefill draft architecture {}",
            model_dir.display(),
            DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER
        );
        // The pinned checkpoint is a VLM distribution, but SpecPrefill uses
        // only its text scorer. Make the draft-only sanitized contract
        // explicitly language-only; ordinary target loading retains the
        // complete vision and bundled-MTP validation above.
        config.vision_config = None;
        config.image_token_id = None;
        config.video_token_id = None;
        config.vision_start_token_id = None;
        config.mtp_num_hidden_layers = None;
        config.mtp_use_dedicated_embeddings = None;
        Self::validate_shard_index(model_dir)?;
        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, |name| {
            name.starts_with("language_model.")
                || name.starts_with("model.language_model.")
                || name.starts_with("lm_head.")
        })
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to load SpecPrefill draft checkpoint shards from {}",
                model_dir.display()
            )
        })?;
        ensure!(
            !weights.is_empty(),
            "checkpoint {} contains no Qwen3.5 language-model tensors",
            model_dir.display()
        );
        let weights = sanitize_language_model_weights(weights, &config, model_dir)?;
        Self::from_weights(&weights.target, &config, KVCacheMode::Fp16)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to construct SpecPrefill draft model {} from {}",
                    DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER,
                    model_dir.display()
                )
            })
    }

    pub(crate) fn specprefill_draft_prefill(
        &self,
        prompt_ids: &[i32],
    ) -> Result<UniquePtr<MlxArray>> {
        ensure!(
            !prompt_ids.is_empty(),
            "SpecPrefill draft prompt must not be empty"
        );
        self.reset_runtime_state();
        let configured = mlxcel_core::generate::prefill_chunk_len();
        let chunk_len =
            mlxcel_core::generate::effective_prefill_chunk(configured, true, prompt_ids.len())
                .unwrap_or(prompt_ids.len());
        let mut final_logits = None;
        for (chunk_index, chunk) in prompt_ids.chunks(chunk_len).enumerate() {
            let input = mlxcel_core::from_slice_i32(chunk, &[1, chunk.len() as i32]);
            let hidden = self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&input, None, caches, None)
            });
            if (chunk_index + 1) * chunk_len >= prompt_ids.len() {
                let shape = mlxcel_core::array_shape(&hidden);
                let last = shape[1] - 1;
                let last_hidden =
                    mlxcel_core::slice(&hidden, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
                final_logits = Some(self.project_logits(&self.norm.forward(&last_hidden)));
            }
        }
        Ok(final_logits.expect("non-empty draft prompt produces logits"))
    }

    pub(crate) fn specprefill_draft_lookahead(
        &self,
        token_id: i32,
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        let input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        self.sequence_state.with_internal(|caches| {
            let mut hidden = self.embed_tokens.forward(&input);
            let mut queries =
                Vec::with_capacity(self.layers.len() / self.config.full_attention_interval);
            for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                let (next, query) = layer.forward_with_query_capture(&hidden, cache);
                hidden = next;
                if let Some(query) = query {
                    queries.push(query);
                }
            }
            (self.project_logits(&self.norm.forward(&hidden)), queries)
        })
    }

    pub(crate) fn specprefill_draft_prompt_keys(
        &self,
        prompt_len: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        let prompt_len = i32::try_from(prompt_len).unwrap_or(i32::MAX);
        self.sequence_state.with_internal(|caches| {
            caches
                .iter()
                .filter_map(|cache| match cache {
                    Qwen3NextCache::Attention(cache) => {
                        let keys = cache.keys.as_deref()?;
                        let shape = mlxcel_core::array_shape(keys);
                        Some(mlxcel_core::slice(
                            keys,
                            &[0, 0, 0, 0],
                            &[shape[0], shape[1], prompt_len, shape[3]],
                        ))
                    }
                    Qwen3NextCache::Linear(_) => None,
                })
                .collect()
        })
    }

    pub(crate) fn specprefill_sparse_prefill(
        &self,
        prompt_ids: &[i32],
        prefix_reuse: Option<mlxcel_core::generate::PrefixReuse<'_>>,
        dense_prefix_end: usize,
        selected_indices: &[usize],
    ) -> std::result::Result<(UniquePtr<MlxArray>, usize), String> {
        #[cfg(feature = "dlsa")]
        self.dlsa_refusal("SpecPrefill")?;
        self.reset_runtime_state();
        let cached_tokens =
            crate::specprefill::admitted_cached_tokens(prefix_reuse.as_ref(), prompt_ids.len());
        if let Some(reuse) = prefix_reuse
            && cached_tokens > 0
        {
            self.restore_sequence_state(SequenceId::from_raw(0), reuse.snapshot)?;
        }

        if cached_tokens < dense_prefix_end {
            let dense = &prompt_ids[cached_tokens..dense_prefix_end];
            let input = mlxcel_core::from_slice_i32(dense, &[1, dense.len() as i32]);
            self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&input, None, caches, None)
            });
        }

        if selected_indices.is_empty() {
            return Err("SpecPrefill selected no target tokens".to_string());
        }
        let mut final_hidden = None;
        for indices in selected_indices.chunks(SPECPREFILL_TARGET_CHUNK_TOKENS) {
            let sparse_ids = indices
                .iter()
                .map(|&index| prompt_ids[index])
                .collect::<Vec<_>>();
            let one_axis_positions = indices
                .iter()
                .map(|&index| i32::try_from(index).unwrap_or(i32::MAX))
                .collect::<Vec<_>>();
            let positions = one_axis_positions
                .iter()
                .chain(&one_axis_positions)
                .chain(&one_axis_positions)
                .copied()
                .collect::<Vec<_>>();
            let input = mlxcel_core::from_slice_i32(&sparse_ids, &[1, sparse_ids.len() as i32]);
            let position_ids =
                mlxcel_core::from_slice_i32(&positions, &[3, 1, sparse_ids.len() as i32]);
            final_hidden = Some(self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&input, None, caches, Some(&position_ids))
            }));
        }
        let hidden = final_hidden.expect("non-empty SpecPrefill selection produces hidden states");
        let shape = mlxcel_core::array_shape(&hidden);
        let last = shape[1] - 1;
        let last_hidden =
            mlxcel_core::slice(&hidden, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
        self.finish_initial_prefill();
        Ok((
            self.project_logits(&self.norm.forward(&last_hidden)),
            cached_tokens,
        ))
    }

    pub(crate) fn specprefill_decode(
        &self,
        token_id: i32,
        logical_position: usize,
    ) -> UniquePtr<MlxArray> {
        let input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        let position = i32::try_from(logical_position).unwrap_or(i32::MAX);
        let position_ids = mlxcel_core::from_slice_i32(&[position, position, position], &[3, 1, 1]);
        let hidden = self.sequence_state.with_internal(|caches| {
            self.forward_backbone_with_inputs(&input, None, caches, Some(&position_ids))
        });
        self.project_logits(&self.norm.forward(&hidden))
    }
}
