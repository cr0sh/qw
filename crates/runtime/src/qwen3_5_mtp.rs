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

//! Built-in Qwen 3.5 multi-token-prediction drafter and greedy round loop.
//!
//! Architecture and reconciliation follow the Apache-2.0 `mlx-vlm` Qwen 3.5
//! MTP drafter; target verification and rollback follow the checked-in
//! Apache-2.0 `mlxcel` Qwen 3.5 implementation.

use std::cell::RefCell;
use std::time::Instant;

use mlxcel_core::generate::{GenerationStats, LanguageModel, SamplingConfig};
use mlxcel_core::generation_policy::merged_eos_token_ids;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::speculative::mtp::speculative_walk;
use mlxcel_core::utils::create_causal_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::qwen3_5::{Qwen35Config, Qwen35DecoderLayer, Qwen35Model};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpGenerationStats {
    pub accepted_draft_tokens: usize,
    pub proposed_draft_tokens: usize,
}

impl MtpGenerationStats {
    pub fn acceptance_percentage(self) -> f64 {
        if self.proposed_draft_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_draft_tokens as f64 * 100.0
        }
    }
}

struct Qwen35MtpDraftState {
    cache: KVCache,
    seed_token: Option<i32>,
    seed_hidden: Option<UniquePtr<MlxArray>>,
    next_position: i32,
    round_appended: usize,
}

impl Qwen35MtpDraftState {
    fn new() -> Self {
        Self {
            cache: KVCache::new(),
            seed_token: None,
            seed_hidden: None,
            next_position: 0,
            round_appended: 0,
        }
    }
}

/// The single bundled Qwen 3.5 MTP layer. Embeddings and the output projection
/// remain owned by and shared with the target model.
pub(crate) struct Qwen35MtpDraftModel {
    pre_fc_norm_embedding: RMSNorm,
    pre_fc_norm_hidden: RMSNorm,
    fc: UnifiedLinear,
    layer: Qwen35DecoderLayer,
    norm: RMSNorm,
    state: RefCell<Qwen35MtpDraftState>,
}

impl Qwen35MtpDraftModel {
    pub(crate) fn from_weights(weights: &WeightMap, config: &Qwen35Config) -> Result<Self, String> {
        let embedding_norm = weights
            .get("mtp.pre_fc_norm_embedding.weight")
            .map(|weight| mlxcel_core::copy(weight))
            .ok_or_else(|| {
                "missing required tensor mtp.pre_fc_norm_embedding.weight".to_string()
            })?;
        let hidden_norm = weights
            .get("mtp.pre_fc_norm_hidden.weight")
            .map(|weight| mlxcel_core::copy(weight))
            .ok_or_else(|| "missing required tensor mtp.pre_fc_norm_hidden.weight".to_string())?;
        let norm = weights
            .get("mtp.norm.weight")
            .map(|weight| mlxcel_core::copy(weight))
            .ok_or_else(|| "missing required tensor mtp.norm.weight".to_string())?;
        let (fc_group_size, fc_bits) = config.quant_params("mtp.fc");
        let fc = UnifiedLinear::from_weights(weights, "mtp.fc", fc_group_size, fc_bits)?;
        let layer = Qwen35DecoderLayer::from_weights_at_prefix(
            weights,
            config,
            &config.to_qwen3next_config(),
            "mtp.layers.0",
            false,
        )?;

        Ok(Self {
            pre_fc_norm_embedding: RMSNorm::new(embedding_norm, config.rms_norm_eps),
            pre_fc_norm_hidden: RMSNorm::new(hidden_norm, config.rms_norm_eps),
            fc,
            layer,
            norm: RMSNorm::new(norm, config.rms_norm_eps),
            state: RefCell::new(Qwen35MtpDraftState::new()),
        })
    }

    pub(crate) fn reset(&self) {
        *self.state.borrow_mut() = Qwen35MtpDraftState::new();
    }

    fn forward_tokens(
        &self,
        target: &Qwen35Model,
        tokens: &MlxArray,
        target_hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) -> UniquePtr<MlxArray> {
        let token_embedding = target.embed_tokens.forward(tokens);
        let embedding = self.pre_fc_norm_embedding.forward(&token_embedding);
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden);
        let concatenated = mlxcel_core::concatenate(&embedding, &hidden, -1);
        let mut output = self.fc.forward(&concatenated);
        let steps = mlxcel_core::array_shape(&output)[1];
        let mask = if steps > 1 {
            Some(create_causal_mask(steps, state.cache.offset))
        } else {
            None
        };
        output = self
            .layer
            .forward_full_attention(&output, mask.as_deref(), &mut state.cache);
        state.next_position += steps;
        self.norm.forward(&output)
    }

    fn set_seed_from_hidden(
        &self,
        target: &Qwen35Model,
        hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) {
        let logits = target.project_logits(hidden);
        let token = mlxcel_core::argmax_last_axis(&logits);
        mlxcel_core::eval(&token);
        state.seed_token = Some(mlxcel_core::item_i32(&token));
        state.seed_hidden = Some(mlxcel_core::copy(hidden));
    }

    pub(crate) fn prefill_from_target_hidden(
        &self,
        target: &Qwen35Model,
        input_ids: &MlxArray,
        target_hidden: &MlxArray,
        bonus_token: i32,
    ) {
        let shape = mlxcel_core::array_shape(input_ids);
        let prompt_len = shape[1];
        if prompt_len == 0 {
            return;
        }
        let bonus = mlxcel_core::from_slice_i32(&[bonus_token], &[1, 1]);
        let shifted = if prompt_len == 1 {
            bonus
        } else {
            let tail = mlxcel_core::slice(input_ids, &[0, 1], &[shape[0], prompt_len]);
            mlxcel_core::concatenate(&tail, &bonus, 1)
        };
        let hidden_shape = mlxcel_core::array_shape(target_hidden);
        let hidden = mlxcel_core::slice(
            target_hidden,
            &[0, 0, 0],
            &[hidden_shape[0], prompt_len, hidden_shape[2]],
        );
        let mut state = self.state.borrow_mut();
        state.next_position = 0;
        let output = self.forward_tokens(target, &shifted, &hidden, &mut state);
        let output_shape = mlxcel_core::array_shape(&output);
        let last = output_shape[1] - 1;
        let last_hidden = mlxcel_core::slice(
            &output,
            &[0, last, 0],
            &[output_shape[0], last + 1, output_shape[2]],
        );
        self.set_seed_from_hidden(target, &last_hidden, &mut state);
    }

    pub(crate) fn draft_block(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        block_size: usize,
    ) -> Vec<i32> {
        let mut state = self.state.borrow_mut();
        state.round_appended = 0;
        let mut tokens = Vec::with_capacity(block_size.saturating_sub(1));
        let (mut token, mut hidden) = if let (Some(seed_token), Some(seed_hidden)) =
            (state.seed_token.take(), state.seed_hidden.take())
        {
            tokens.push(seed_token);
            (seed_token, seed_hidden)
        } else {
            (last_bonus, mlxcel_core::copy(target_hidden))
        };

        while tokens.len() < block_size.saturating_sub(1) {
            let token_array = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            hidden = self.forward_tokens(target, &token_array, &hidden, &mut state);
            state.round_appended += 1;
            let logits = target.project_logits(&hidden);
            let next = mlxcel_core::argmax_last_axis(&logits);
            mlxcel_core::eval(&next);
            token = mlxcel_core::item_i32(&next);
            tokens.push(token);
        }
        tokens
    }

    pub(crate) fn accept_verified_tokens(
        &self,
        target: &Qwen35Model,
        verify_hidden: &MlxArray,
        draft_tokens: &[i32],
        accepted: usize,
        new_tokens: &[i32],
    ) {
        let mut state = self.state.borrow_mut();
        let round_appended = state.round_appended;
        let keep_appended = trim_draft_cache(&mut state.cache, round_appended, accepted);
        state.next_position = state.cache.offset;

        let mut tokens = draft_tokens[keep_appended..accepted.min(draft_tokens.len())].to_vec();
        if let Some(&last) = new_tokens.last() {
            tokens.push(last);
        }
        if !tokens.is_empty() {
            let token_array = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
            let hidden_shape = mlxcel_core::array_shape(verify_hidden);
            let end = accepted.saturating_add(1).min(hidden_shape[1] as usize) as i32;
            let hidden = mlxcel_core::slice(
                verify_hidden,
                &[0, keep_appended as i32, 0],
                &[hidden_shape[0], end, hidden_shape[2]],
            );
            let output = self.forward_tokens(target, &token_array, &hidden, &mut state);
            let output_shape = mlxcel_core::array_shape(&output);
            let last = output_shape[1] - 1;
            let last_hidden = mlxcel_core::slice(
                &output,
                &[0, last, 0],
                &[output_shape[0], last + 1, output_shape[2]],
            );
            self.set_seed_from_hidden(target, &last_hidden, &mut state);
        }
        state.round_appended = 0;
    }
}

fn trim_draft_cache(cache: &mut KVCache, round_appended: usize, accepted: usize) -> usize {
    let keep_appended = accepted.min(round_appended);
    let trim = round_appended - keep_appended;
    if trim > 0 {
        cache.trim(i32::try_from(trim).unwrap_or(i32::MAX));
    }
    keep_appended
}

fn visible_tokens(tokens: &[i32], eos_tokens: &[i32]) -> (Vec<i32>, bool) {
    match tokens.iter().position(|token| eos_tokens.contains(token)) {
        Some(stop) => (tokens[..stop].to_vec(), true),
        None => (tokens.to_vec(), false),
    }
}

pub(crate) struct Qwen35MtpGenerator;

impl Qwen35MtpGenerator {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn generate(
        &mut self,
        model: &Qwen35Model,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
    ) -> (Vec<i32>, GenerationStats, MtpGenerationStats) {
        assert!(!prompt_tokens.is_empty(), "MTP prompt must not be empty");
        assert_eq!(sampling.temperature, 0.0, "Qwen MTP is greedy-only");
        assert!(block_size >= 2, "MTP block size must be at least 2");
        model.reset_runtime_state();
        let drafter = model
            .mtp()
            .expect("Qwen35MtpGenerator requires a bundled MTP head");
        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);

        let prefill_start = Instant::now();
        let prompt = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        let prefill = model.forward_mtp_prefill(&prompt);
        let first_token = prefill.first_token;
        mlxcel_core::eval(&prefill.hidden);

        let mut generated = Vec::with_capacity(max_tokens);
        let mut mtp_stats = MtpGenerationStats::default();
        let first_is_eos = eos_tokens.contains(&first_token);
        if max_tokens > 0 && !first_is_eos {
            generated.push(first_token);
        }
        if max_tokens > 1 && !first_is_eos {
            drafter.prefill_from_target_hidden(model, &prompt, &prefill.hidden, first_token);
        }
        let prefill_time = prefill_start.elapsed();

        let decode_start = Instant::now();
        if generated.len() < max_tokens && !first_is_eos {
            let prefill_shape = mlxcel_core::array_shape(&prefill.hidden);
            let last = prefill_shape[1] - 1;
            let mut next_hidden = mlxcel_core::slice(
                &prefill.hidden,
                &[0, last, 0],
                &[prefill_shape[0], last + 1, prefill_shape[2]],
            );
            let mut bonus = first_token;

            while generated.len() < max_tokens {
                let remaining = max_tokens - generated.len();
                let round_block_size = block_size.min(remaining + 1);
                if round_block_size <= 1 {
                    break;
                }
                let draft_tokens =
                    drafter.draft_block(model, bonus, &next_hidden, round_block_size);
                let mut verify_tokens = Vec::with_capacity(round_block_size);
                verify_tokens.push(bonus);
                verify_tokens.extend_from_slice(&draft_tokens);
                let verify_input = mlxcel_core::from_slice_i32(
                    &verify_tokens,
                    &[1, i32::try_from(verify_tokens.len()).unwrap_or(i32::MAX)],
                );
                let verify = model.forward_mtp_verify(&verify_input);
                let walk = speculative_walk(&draft_tokens, &verify.target_tokens, remaining);
                mtp_stats.accepted_draft_tokens += walk.accepted;
                mtp_stats.proposed_draft_tokens += draft_tokens.len();
                let (emitted, hit_eos) = visible_tokens(&walk.new_tokens, &eos_tokens);
                generated.extend_from_slice(&emitted);

                if hit_eos || generated.len() >= max_tokens {
                    break;
                }

                drafter.accept_verified_tokens(
                    model,
                    &verify.hidden,
                    &draft_tokens,
                    walk.accepted,
                    &walk.new_tokens,
                );
                if walk.accepted < draft_tokens.len() {
                    model.rollback_mtp_verify(&verify.gdn_states, walk.accepted, round_block_size);
                }
                let hidden_shape = mlxcel_core::array_shape(&verify.hidden);
                let accepted = i32::try_from(walk.accepted).unwrap_or(i32::MAX);
                next_hidden = mlxcel_core::slice(
                    &verify.hidden,
                    &[0, accepted, 0],
                    &[hidden_shape[0], accepted + 1, hidden_shape[2]],
                );
                bonus = *walk
                    .new_tokens
                    .last()
                    .expect("speculative walk emits at least one token");
            }
        }
        let decode_time = decode_start.elapsed();

        let prefill_ms = prefill_time.as_secs_f64() * 1_000.0;
        let decode_ms = decode_time.as_secs_f64() * 1_000.0;
        let decode_tokens = generated.len().saturating_sub(1);
        let stats = GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated.len(),
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_tokens.len() as f64 / (prefill_ms / 1_000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                decode_tokens as f64 / (decode_ms / 1_000.0)
            } else {
                0.0
            },
        };
        (generated, stats, mtp_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gated_delta::GatedDeltaCache;
    use crate::qwen3_5::rollback_plan;
    use crate::qwen3_next::Qwen3NextCache;
    use crate::qwen_mrope_state::MRopeState;

    #[test]
    fn acceptance_percentage_handles_zero_proposals_and_partial_acceptance() {
        let no_proposals = MtpGenerationStats::default();
        assert_eq!(no_proposals.acceptance_percentage(), 0.0);

        let partial = MtpGenerationStats {
            accepted_draft_tokens: 1,
            proposed_draft_tokens: 4,
        };
        assert_eq!(partial.acceptance_percentage(), 25.0);
    }

    #[test]
    fn speculative_walk_covers_acceptance_rejection_and_limits() {
        let full = speculative_walk(&[1, 2], &[1, 2, 3], 3);
        assert_eq!(full.accepted, 2);
        assert_eq!(full.new_tokens, vec![1, 2, 3]);

        let first_rejection = speculative_walk(&[1, 2], &[9, 2, 3], 3);
        assert_eq!(first_rejection.accepted, 0);
        assert_eq!(first_rejection.new_tokens, vec![9]);

        let partial = speculative_walk(&[1, 2, 3], &[1, 8, 3, 4], 4);
        assert_eq!(partial.accepted, 1);
        assert_eq!(partial.new_tokens, vec![1, 8]);

        let limited = speculative_walk(&[1, 2, 3], &[1, 2, 3, 4], 2);
        assert_eq!(limited.accepted, 3);
        assert_eq!(limited.new_tokens, vec![1, 2]);

        let (before_eos, stopped) = visible_tokens(&[1, 248044, 2], &[248044]);
        assert_eq!(before_eos, vec![1]);
        assert!(stopped);
    }

    #[test]
    fn rejection_reconciles_every_cache_to_the_accepted_prefix() {
        let prefix_before_verify = 10;
        let block_size = 6;
        let accepted = 2;
        let verify_offset = prefix_before_verify + block_size as i32;
        let plan = rollback_plan(verify_offset, accepted, block_size);
        assert_eq!(plan.accepted_block_len, 3);
        assert_eq!(plan.final_offset, prefix_before_verify + 3);

        let mut attention = KVCache::new();
        attention.offset = verify_offset;
        attention.trim(plan.trim);
        let mut linear = GatedDeltaCache::new();
        linear.offset = plan.final_offset;
        let caches = [
            Qwen3NextCache::Attention(Box::new(attention)),
            Qwen3NextCache::Linear(linear),
        ];
        assert_eq!(caches[0].offset(), plan.final_offset);
        assert_eq!(caches[1].offset(), plan.final_offset);

        let mut draft = KVCache::new();
        draft.offset = prefix_before_verify + 4;
        let kept = trim_draft_cache(&mut draft, 4, accepted);
        assert_eq!(kept, accepted);
        assert_eq!(draft.offset, prefix_before_verify + accepted as i32);
        draft.offset += 1;
        assert_eq!(draft.offset, plan.final_offset);

        let mrope = MRopeState::new();
        mrope.set_position(plan.final_offset);
        assert_eq!(mrope.position(), plan.final_offset);
    }
}
