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
use std::time::{Duration, Instant};

use mlxcel_core::generate::{
    ConstraintCommit, ConstraintMask, GenerationStopReason, LanguageModel, SamplingConfig,
    TokenConstraint, mask_logits_to_allowed,
};
use mlxcel_core::generation_policy::{merged_eos_token_ids, seed_rng_if_needed};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::sampling::{
    effective_token_distribution, sample_token_optimized, sample_token_with_distribution,
};
use mlxcel_core::speculative::mtp::speculative_walk;
use mlxcel_core::speculative::mtp::walk::WalkResult;
use mlxcel_core::speculative::stochastic_accept::{
    DraftVerdict, sampler_is_greedy, verify_draft_token,
};
use mlxcel_core::utils::create_causal_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::qwen_vl_position::decode_rope_positions;
use crate::qwen3_5::{Qwen35Config, Qwen35DecoderLayer, Qwen35Model};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpGenerationStats {
    pub accepted_draft_tokens: usize,
    pub proposed_draft_tokens: usize,
    /// Wall-clock time spent in the post-prefill MTP decode loop.
    pub decode_time: Duration,
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

pub(crate) struct MtpGeneration {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) stats: MtpGenerationStats,
    pub(crate) stop_reason: GenerationStopReason,
}

struct MtpProposal {
    token: i32,
    proposal_probs: UniquePtr<MlxArray>,
}

struct Qwen35MtpDraftState {
    cache: KVCache,
    seed_logits: Option<UniquePtr<MlxArray>>,
    seed_hidden: Option<UniquePtr<MlxArray>>,
    rope_delta: Option<i32>,
    round_appended: usize,
}

impl Qwen35MtpDraftState {
    fn new() -> Self {
        Self {
            cache: KVCache::new(),
            seed_logits: None,
            seed_hidden: None,
            rope_delta: None,
            round_appended: 0,
        }
    }
}

fn shift_prompt_embeddings(
    input_embeddings: &MlxArray,
    bonus_embedding: &MlxArray,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(input_embeddings);
    if shape[1] == 1 {
        mlxcel_core::copy(bonus_embedding)
    } else {
        let tail = mlxcel_core::slice(
            input_embeddings,
            &[0, 1, 0],
            &[shape[0], shape[1], shape[2]],
        );
        mlxcel_core::concatenate(&tail, bonus_embedding, 1)
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

    fn forward_embeddings(
        &self,
        token_embeddings: &MlxArray,
        target_hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let embedding = self.pre_fc_norm_embedding.forward(token_embeddings);
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden);
        let concatenated = mlxcel_core::concatenate(&embedding, &hidden, -1);
        let mut output = self.fc.forward(&concatenated);
        let steps = mlxcel_core::array_shape(&output)[1];
        let cache_offset = state.cache.offset;
        let mask = (steps > 1).then(|| create_causal_mask(steps, cache_offset));
        let decode_positions = if position_ids.is_none() {
            state
                .rope_delta
                .map(|delta| decode_rope_positions(cache_offset, steps, delta))
        } else {
            None
        };
        output = self.layer.forward_full_attention(
            &output,
            mask.as_deref(),
            &mut state.cache,
            position_ids.or(decode_positions.as_deref()),
        );
        self.norm.forward(&output)
    }

    fn forward_tokens(
        &self,
        target: &Qwen35Model,
        tokens: &MlxArray,
        target_hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) -> UniquePtr<MlxArray> {
        let token_embeddings = target.embed_tokens.forward(tokens);
        self.forward_embeddings(&token_embeddings, target_hidden, state, None)
    }

    fn set_seed_from_hidden(
        &self,
        target: &Qwen35Model,
        hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) {
        state.seed_logits = Some(target.project_logits(hidden));
        state.seed_hidden = Some(mlxcel_core::copy(hidden));
    }

    pub(crate) fn prefill_from_target_hidden(
        &self,
        target: &Qwen35Model,
        input_ids: &MlxArray,
        target_hidden: &MlxArray,
        bonus_token: i32,
    ) {
        self.prefill_from_target_hidden_with_inputs(
            target,
            input_ids,
            None,
            target_hidden,
            bonus_token,
            None,
            None,
        );
    }

    pub(crate) fn prefill_from_target_hidden_with_embeddings(
        &self,
        target: &Qwen35Model,
        input_ids: &MlxArray,
        input_embeddings: &MlxArray,
        target_hidden: &MlxArray,
        bonus_token: i32,
        position_ids: &MlxArray,
        rope_delta: i32,
    ) {
        self.prefill_from_target_hidden_with_inputs(
            target,
            input_ids,
            Some(input_embeddings),
            target_hidden,
            bonus_token,
            Some(position_ids),
            Some(rope_delta),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_from_target_hidden_with_inputs(
        &self,
        target: &Qwen35Model,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        target_hidden: &MlxArray,
        bonus_token: i32,
        position_ids: Option<&MlxArray>,
        rope_delta: Option<i32>,
    ) {
        let shape = mlxcel_core::array_shape(input_ids);
        let prompt_len = shape[1];
        if prompt_len == 0 {
            return;
        }
        let bonus = mlxcel_core::from_slice_i32(&[bonus_token], &[1, 1]);
        let bonus_embedding = target.embed_tokens.forward(&bonus);
        let shifted_embeddings = if let Some(input_embeddings) = input_embeddings {
            shift_prompt_embeddings(input_embeddings, &bonus_embedding)
        } else if prompt_len == 1 {
            bonus_embedding
        } else {
            let tail = mlxcel_core::slice(input_ids, &[0, 1], &[shape[0], prompt_len]);
            let shifted = mlxcel_core::concatenate(&tail, &bonus, 1);
            target.embed_tokens.forward(&shifted)
        };
        let hidden_shape = mlxcel_core::array_shape(target_hidden);
        let hidden = mlxcel_core::slice(
            target_hidden,
            &[0, 0, 0],
            &[hidden_shape[0], prompt_len, hidden_shape[2]],
        );
        let mut state = self.state.borrow_mut();
        state.rope_delta = rope_delta;
        let output = self.forward_embeddings(
            &shifted_embeddings,
            &hidden,
            &mut state,
            position_ids,
        );
        let output_shape = mlxcel_core::array_shape(&output);
        let last = output_shape[1] - 1;
        let last_hidden = mlxcel_core::slice(
            &output,
            &[0, last, 0],
            &[output_shape[0], last + 1, output_shape[2]],
        );
        self.set_seed_from_hidden(target, &last_hidden, &mut state);
    }

    fn draft_seed(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match (state.seed_logits.take(), state.seed_hidden.take()) {
            (Some(logits), Some(hidden)) => (logits, hidden),
            _ => {
                let bonus = mlxcel_core::from_slice_i32(&[last_bonus], &[1, 1]);
                let hidden = self.forward_tokens(target, &bonus, target_hidden, state);
                state.round_appended += 1;
                let logits = target.project_logits(&hidden);
                (logits, hidden)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draft_block_greedy(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        proposal_count: usize,
        sampling: &SamplingConfig,
        committed_history: &[i32],
        eos_tokens: &[i32],
    ) -> Vec<i32> {
        let mut state = self.state.borrow_mut();
        state.round_appended = 0;
        let mut tokens = Vec::with_capacity(proposal_count);
        let mut history = committed_history.to_vec();
        let (mut logits, mut hidden) =
            self.draft_seed(target, last_bonus, target_hidden, &mut state);

        while tokens.len() < proposal_count {
            let (token_array, _) = sample_token_optimized(&logits, sampling, &history);
            mlxcel_core::eval(&token_array);
            let token = mlxcel_core::item_i32(&token_array);
            tokens.push(token);
            if eos_tokens.contains(&token) || tokens.len() == proposal_count {
                break;
            }
            history.push(token);
            let token_array = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            hidden = self.forward_tokens(target, &token_array, &hidden, &mut state);
            state.round_appended += 1;
            logits = target.project_logits(&hidden);
        }
        tokens
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_block_stochastic(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        proposal_count: usize,
        sampling: &SamplingConfig,
        committed_history: &[i32],
        eos_tokens: &[i32],
    ) -> Vec<MtpProposal> {
        let mut state = self.state.borrow_mut();
        state.round_appended = 0;
        let mut proposals = Vec::with_capacity(proposal_count);
        let mut history = committed_history.to_vec();
        let (mut logits, mut hidden) =
            self.draft_seed(target, last_bonus, target_hidden, &mut state);

        while proposals.len() < proposal_count {
            let (token_array, proposal_probs) =
                sample_token_with_distribution(&logits, sampling, &history);
            mlxcel_core::eval(&token_array);
            let token = mlxcel_core::item_i32(&token_array);
            proposals.push(MtpProposal {
                token,
                proposal_probs,
            });
            if eos_tokens.contains(&token) || proposals.len() == proposal_count {
                break;
            }
            history.push(token);
            let token_array = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            hidden = self.forward_tokens(target, &token_array, &hidden, &mut state);
            state.round_appended += 1;
            logits = target.project_logits(&hidden);
        }
        proposals
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_block_greedy_constrained(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        proposal_count: usize,
        sampling: &SamplingConfig,
        prompt_tokens: &[i32],
        committed_output: &[i32],
        eos_tokens: &[i32],
        constraint: &mut dyn TokenConstraint,
    ) -> Result<Vec<i32>, String> {
        let mut state = self.state.borrow_mut();
        state.round_appended = 0;
        let mut tokens = Vec::with_capacity(proposal_count);
        let mut output = committed_output.to_vec();
        let mut history = Vec::with_capacity(prompt_tokens.len() + output.len() + proposal_count);
        rebuild_history(prompt_tokens, &output, &mut history);
        let (mut logits, mut hidden) =
            self.draft_seed(target, last_bonus, target_hidden, &mut state);

        while tokens.len() < proposal_count {
            let step = constraint_step(&logits, constraint, &history)?;
            let Some(logits_for_sample) = step.logits() else {
                if let ConstraintStepLogits::Splice(commit) = step {
                    commit.apply_to(&mut output)?;
                }
                break;
            };
            let (token_array, _) =
                sample_token_optimized(logits_for_sample, sampling, &history);
            mlxcel_core::eval(&token_array);
            let token = mlxcel_core::item_i32(&token_array);
            tokens.push(token);
            if eos_tokens.contains(&token) {
                break;
            }
            let commit = constraint.commit_token(token)?;
            let changed = !commit.is_token(token);
            commit.apply_to(&mut output)?;
            rebuild_history(prompt_tokens, &output, &mut history);
            if changed || commit.accept || tokens.len() == proposal_count {
                break;
            }
            let token_array = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            hidden = self.forward_tokens(target, &token_array, &hidden, &mut state);
            state.round_appended += 1;
            logits = target.project_logits(&hidden);
        }
        Ok(tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_block_stochastic_constrained(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        proposal_count: usize,
        sampling: &SamplingConfig,
        prompt_tokens: &[i32],
        committed_output: &[i32],
        eos_tokens: &[i32],
        constraint: &mut dyn TokenConstraint,
    ) -> Result<Vec<MtpProposal>, String> {
        let mut state = self.state.borrow_mut();
        state.round_appended = 0;
        let mut proposals = Vec::with_capacity(proposal_count);
        let mut output = committed_output.to_vec();
        let mut history = Vec::with_capacity(prompt_tokens.len() + output.len() + proposal_count);
        rebuild_history(prompt_tokens, &output, &mut history);
        let (mut logits, mut hidden) =
            self.draft_seed(target, last_bonus, target_hidden, &mut state);

        while proposals.len() < proposal_count {
            let step = constraint_step(&logits, constraint, &history)?;
            let Some(logits_for_sample) = step.logits() else {
                if let ConstraintStepLogits::Splice(commit) = step {
                    commit.apply_to(&mut output)?;
                }
                break;
            };
            let (token_array, proposal_probs) =
                sample_token_with_distribution(logits_for_sample, sampling, &history);
            mlxcel_core::eval(&token_array);
            let token = mlxcel_core::item_i32(&token_array);
            proposals.push(MtpProposal {
                token,
                proposal_probs,
            });
            if eos_tokens.contains(&token) {
                break;
            }
            let commit = constraint.commit_token(token)?;
            let changed = !commit.is_token(token);
            commit.apply_to(&mut output)?;
            rebuild_history(prompt_tokens, &output, &mut history);
            if changed || commit.accept || proposals.len() == proposal_count {
                break;
            }
            let token_array = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            hidden = self.forward_tokens(target, &token_array, &hidden, &mut state);
            state.round_appended += 1;
            logits = target.project_logits(&hidden);
        }
        Ok(proposals)
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

fn round_proposal_count(block_size: usize, remaining: usize) -> usize {
    block_size.saturating_sub(1).min(remaining)
}

fn logits_at(logits: &MlxArray, position: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(logits);
    mlxcel_core::slice(
        logits,
        &[0, position as i32, 0],
        &[shape[0], position as i32 + 1, shape[2]],
    )
}

enum ConstraintStepLogits {
    Masked(UniquePtr<MlxArray>),
    Splice(ConstraintCommit),
    Accept,
}

impl ConstraintStepLogits {
    fn logits(&self) -> Option<&MlxArray> {
        match self {
            Self::Masked(logits) => logits.as_ref(),
            Self::Splice(_) | Self::Accept => None,
        }
    }
}

fn constraint_step(
    logits: &MlxArray,
    constraint: &mut dyn TokenConstraint,
    history: &[i32],
) -> Result<ConstraintStepLogits, String> {
    match constraint.compute_mask(logits, history)? {
        ConstraintMask::Allow(allowed) => {
            Ok(ConstraintStepLogits::Masked(mask_logits_to_allowed(
                logits, &allowed,
            )?))
        }
        ConstraintMask::Splice(commit) => Ok(ConstraintStepLogits::Splice(commit)),
        ConstraintMask::Accept => Ok(ConstraintStepLogits::Accept),
    }
}

fn rebuild_history(prompt_tokens: &[i32], output: &[i32], history: &mut Vec<i32>) {
    history.clear();
    history.extend_from_slice(prompt_tokens);
    history.extend_from_slice(output);
}

fn rollback_constraint_transaction<T>(
    constraint: &mut dyn TokenConstraint,
    operation: impl FnOnce(&mut dyn TokenConstraint) -> Result<T, String>,
) -> Result<T, String> {
    constraint.begin_transaction()?;
    let result = operation(constraint);
    constraint.rollback_transaction();
    result
}

fn commit_constraint_transaction<T>(
    constraint: &mut dyn TokenConstraint,
    operation: impl FnOnce(&mut dyn TokenConstraint) -> Result<T, String>,
) -> Result<T, String> {
    constraint.begin_transaction()?;
    match operation(constraint) {
        Ok(value) => {
            constraint.commit_transaction()?;
            Ok(value)
        }
        Err(error) => {
            constraint.rollback_transaction();
            Err(error)
        }
    }
}

fn greedy_walk(
    draft_tokens: &[i32],
    verify_logits: &MlxArray,
    sampling: &SamplingConfig,
    committed_history: &[i32],
    max_new_tokens: usize,
) -> WalkResult {
    let mut history = committed_history.to_vec();
    let mut target_tokens = Vec::with_capacity(draft_tokens.len() + 1);
    for position in 0..=draft_tokens.len() {
        let logits = logits_at(verify_logits, position);
        let (token, _) = sample_token_optimized(&logits, sampling, &history);
        mlxcel_core::eval(&token);
        target_tokens.push(mlxcel_core::item_i32(&token));
        if position < draft_tokens.len() {
            history.push(draft_tokens[position]);
        }
    }
    speculative_walk(draft_tokens, &target_tokens, max_new_tokens)
}

fn stochastic_walk(
    proposals: &[MtpProposal],
    verify_logits: &MlxArray,
    sampling: &SamplingConfig,
    committed_history: &[i32],
    eos_tokens: &[i32],
    max_new_tokens: usize,
) -> WalkResult {
    let mut history = committed_history.to_vec();
    let mut accepted = 0;
    let mut new_tokens = Vec::with_capacity((proposals.len() + 1).min(max_new_tokens));

    for (position, proposal) in proposals.iter().enumerate() {
        let target_probs = effective_token_distribution(
            &logits_at(verify_logits, position),
            sampling,
            &history,
        );
        match verify_draft_token(&target_probs, &proposal.proposal_probs, proposal.token) {
            DraftVerdict::Accept => {
                accepted += 1;
                new_tokens.push(proposal.token);
                if eos_tokens.contains(&proposal.token) || new_tokens.len() == max_new_tokens {
                    return WalkResult {
                        accepted,

                        new_tokens,
                    };
                }
                history.push(proposal.token);
            }
            DraftVerdict::Reject { replacement } => {
                new_tokens.push(replacement);
                return WalkResult {
                    accepted,
                    new_tokens,
                };
            }
        }
    }

    if new_tokens.len() < max_new_tokens {
        let bonus_logits = logits_at(verify_logits, proposals.len());
        let (bonus, _) = sample_token_optimized(&bonus_logits, sampling, &history);
        mlxcel_core::eval(&bonus);
        new_tokens.push(mlxcel_core::item_i32(&bonus));
    }
    WalkResult {
        accepted,
        new_tokens,
    }
}

struct ConstrainedWalk {
    accepted: usize,
    new_tokens: Vec<i32>,
    output: Vec<i32>,
    rebuild: bool,
    stop_reason: Option<GenerationStopReason>,
}
#[allow(clippy::too_many_arguments)]
fn constrained_initial_step(
    logits: &MlxArray,
    sampling: &SamplingConfig,
    prompt_tokens: &[i32],
    committed_output: &[i32],
    eos_tokens: &[i32],
    max_tokens: usize,
    constraint: &mut dyn TokenConstraint,
) -> Result<ConstrainedWalk, String> {
    let mut output = committed_output.to_vec();
    let mut history = Vec::with_capacity(prompt_tokens.len() + output.len() + 1);
    rebuild_history(prompt_tokens, &output, &mut history);
    let step = constraint_step(logits, constraint, &history)?;
    let logits_for_sample = match &step {
        ConstraintStepLogits::Masked(logits) => logits
            .as_ref()
            .expect("masked constraint logits must not be null"),
        ConstraintStepLogits::Splice(commit) => {
            let stop_reason = apply_walk_splice(
                commit.clone(),
                prompt_tokens,
                &mut output,
                &mut history,
                max_tokens,
            )?;
            return Ok(ConstrainedWalk {
                accepted: 0,
                new_tokens: Vec::new(),
                output,
                rebuild: true,
                stop_reason,
            });
        }
        ConstraintStepLogits::Accept => {
            return Ok(ConstrainedWalk {
                accepted: 0,
                new_tokens: Vec::new(),
                output,
                rebuild: false,
                stop_reason: Some(GenerationStopReason::ConstraintAccepted),
            });
        }
    };
    let (token, _) = sample_token_optimized(logits_for_sample, sampling, &history);
    mlxcel_core::eval(&token);
    let token = mlxcel_core::item_i32(&token);
    if eos_tokens.contains(&token) {
        return Ok(ConstrainedWalk {
            accepted: 0,
            new_tokens: vec![token],
            output,
            rebuild: false,
            stop_reason: Some(GenerationStopReason::Eos),
        });
    }
    let commit = constraint.commit_token(token)?;
    let (rebuild, stop_reason) = apply_walk_commit(
        token,
        commit,
        prompt_tokens,
        &mut output,
        &mut history,
        max_tokens,
    )?;
    Ok(ConstrainedWalk {
        accepted: 0,
        new_tokens: vec![token],
        output,
        rebuild,
        stop_reason,
    })
}

fn apply_walk_commit(
    sampled: i32,
    commit: ConstraintCommit,
    prompt_tokens: &[i32],
    output: &mut Vec<i32>,
    history: &mut Vec<i32>,
    max_tokens: usize,
) -> Result<(bool, Option<GenerationStopReason>), String> {
    let rebuild = !commit.is_token(sampled);
    commit.apply_to(output)?;
    if output.len() > max_tokens {
        output.truncate(max_tokens);
    }
    rebuild_history(prompt_tokens, output, history);
    let stop_reason = if commit.accept {
        Some(GenerationStopReason::ConstraintAccepted)
    } else if output.len() == max_tokens {
        Some(GenerationStopReason::MaxTokens)
    } else {
        None
    };
    Ok((rebuild, stop_reason))
}

fn apply_walk_splice(
    commit: ConstraintCommit,
    prompt_tokens: &[i32],
    output: &mut Vec<i32>,
    history: &mut Vec<i32>,
    max_tokens: usize,
) -> Result<Option<GenerationStopReason>, String> {
    commit.apply_to(output)?;
    if output.len() > max_tokens {
        output.truncate(max_tokens);
    }
    rebuild_history(prompt_tokens, output, history);
    Ok(if commit.accept {
        Some(GenerationStopReason::ConstraintAccepted)
    } else if output.len() == max_tokens {
        Some(GenerationStopReason::MaxTokens)
    } else {
        None
    })
}

#[allow(clippy::too_many_arguments)]
fn constrained_greedy_walk(
    draft_tokens: &[i32],
    verify_logits: &MlxArray,
    sampling: &SamplingConfig,
    prompt_tokens: &[i32],
    committed_output: &[i32],
    eos_tokens: &[i32],
    max_tokens: usize,
    constraint: &mut dyn TokenConstraint,
) -> Result<ConstrainedWalk, String> {
    let mut output = committed_output.to_vec();
    let mut history = Vec::with_capacity(prompt_tokens.len() + output.len() + draft_tokens.len() + 1);
    rebuild_history(prompt_tokens, &output, &mut history);
    let mut accepted = 0;
    let mut new_tokens = Vec::with_capacity(draft_tokens.len() + 1);

    for position in 0..=draft_tokens.len() {
        let logits = logits_at(verify_logits, position);
        let step = constraint_step(&logits, constraint, &history)?;
        let logits_for_sample = match &step {
            ConstraintStepLogits::Masked(logits) => logits
                .as_ref()
                .expect("masked constraint logits must not be null"),
            ConstraintStepLogits::Splice(commit) => {
                let stop_reason = apply_walk_splice(
                    commit.clone(),
                    prompt_tokens,
                    &mut output,
                    &mut history,
                    max_tokens,
                )?;
                return Ok(ConstrainedWalk {
                    accepted,
                    new_tokens,
                    output,
                    rebuild: true,
                    stop_reason,
                });
            }
            ConstraintStepLogits::Accept => {
                return Ok(ConstrainedWalk {
                    accepted,
                    new_tokens,
                    output,
                    rebuild: false,
                    stop_reason: Some(GenerationStopReason::ConstraintAccepted),
                });
            }
        };
        let (token_array, _) = sample_token_optimized(logits_for_sample, sampling, &history);
        mlxcel_core::eval(&token_array);
        let target_token = mlxcel_core::item_i32(&token_array);
        new_tokens.push(target_token);
        if eos_tokens.contains(&target_token) {
            return Ok(ConstrainedWalk {
                accepted,
                new_tokens,
                output,
                rebuild: false,
                stop_reason: Some(GenerationStopReason::Eos),
            });
        }

        let matches_proposal =
            position < draft_tokens.len() && target_token == draft_tokens[position];
        if matches_proposal {
            accepted += 1;
        }
        let commit = constraint.commit_token(target_token)?;
        let (rebuild, stop_reason) = apply_walk_commit(
            target_token,
            commit,
            prompt_tokens,
            &mut output,
            &mut history,
            max_tokens,
        )?;
        if !matches_proposal || rebuild || stop_reason.is_some() || position == draft_tokens.len() {
            return Ok(ConstrainedWalk {
                accepted,
                new_tokens,
                output,
                rebuild,
                stop_reason,
            });
        }
    }
    unreachable!("greedy constrained walk includes one target bonus position")
}

#[allow(clippy::too_many_arguments)]
fn constrained_stochastic_walk(
    proposals: &[MtpProposal],
    verify_logits: &MlxArray,
    sampling: &SamplingConfig,
    prompt_tokens: &[i32],
    committed_output: &[i32],
    eos_tokens: &[i32],
    max_tokens: usize,
    constraint: &mut dyn TokenConstraint,
) -> Result<ConstrainedWalk, String> {
    let mut output = committed_output.to_vec();
    let mut history = Vec::with_capacity(prompt_tokens.len() + output.len() + proposals.len() + 1);
    rebuild_history(prompt_tokens, &output, &mut history);
    let mut accepted = 0;
    let mut new_tokens = Vec::with_capacity(proposals.len() + 1);

    for (position, proposal) in proposals.iter().enumerate() {
        let logits = logits_at(verify_logits, position);
        let step = constraint_step(&logits, constraint, &history)?;
        let logits_for_sample = match &step {
            ConstraintStepLogits::Masked(logits) => logits
                .as_ref()
                .expect("masked constraint logits must not be null"),
            ConstraintStepLogits::Splice(commit) => {
                let stop_reason = apply_walk_splice(
                    commit.clone(),
                    prompt_tokens,
                    &mut output,
                    &mut history,
                    max_tokens,
                )?;
                return Ok(ConstrainedWalk {
                    accepted,
                    new_tokens,
                    output,
                    rebuild: true,
                    stop_reason,
                });
            }
            ConstraintStepLogits::Accept => {
                return Ok(ConstrainedWalk {
                    accepted,
                    new_tokens,
                    output,
                    rebuild: false,
                    stop_reason: Some(GenerationStopReason::ConstraintAccepted),
                });
            }
        };
        let target_probs = effective_token_distribution(logits_for_sample, sampling, &history);
        let target_token = match verify_draft_token(
            &target_probs,
            &proposal.proposal_probs,
            proposal.token,
        ) {
            DraftVerdict::Accept => {
                accepted += 1;
                proposal.token
            }
            DraftVerdict::Reject { replacement } => replacement,
        };
        new_tokens.push(target_token);
        if eos_tokens.contains(&target_token) {
            return Ok(ConstrainedWalk {
                accepted,
                new_tokens,
                output,
                rebuild: false,
                stop_reason: Some(GenerationStopReason::Eos),
            });
        }
        let commit = constraint.commit_token(target_token)?;
        let (rebuild, stop_reason) = apply_walk_commit(
            target_token,
            commit,
            prompt_tokens,
            &mut output,
            &mut history,
            max_tokens,
        )?;
        if target_token != proposal.token || rebuild || stop_reason.is_some() {
            return Ok(ConstrainedWalk {
                accepted,
                new_tokens,
                output,
                rebuild,
                stop_reason,
            });
        }
    }

    let bonus_logits = logits_at(verify_logits, proposals.len());
    let step = constraint_step(&bonus_logits, constraint, &history)?;
    let logits_for_sample = match &step {
        ConstraintStepLogits::Masked(logits) => logits
            .as_ref()
            .expect("masked constraint logits must not be null"),
        ConstraintStepLogits::Splice(commit) => {
            let stop_reason = apply_walk_splice(
                commit.clone(),
                prompt_tokens,
                &mut output,
                &mut history,
                max_tokens,
            )?;
            return Ok(ConstrainedWalk {
                accepted,
                new_tokens,
                output,
                rebuild: true,
                stop_reason,
            });
        }
        ConstraintStepLogits::Accept => {
            return Ok(ConstrainedWalk {
                accepted,
                new_tokens,
                output,
                rebuild: false,
                stop_reason: Some(GenerationStopReason::ConstraintAccepted),
            });
        }
    };
    let (bonus, _) = sample_token_optimized(logits_for_sample, sampling, &history);
    mlxcel_core::eval(&bonus);
    let bonus = mlxcel_core::item_i32(&bonus);
    new_tokens.push(bonus);
    if eos_tokens.contains(&bonus) {
        return Ok(ConstrainedWalk {
            accepted,
            new_tokens,
            output,
            rebuild: false,
            stop_reason: Some(GenerationStopReason::Eos),
        });
    }
    let commit = constraint.commit_token(bonus)?;
    let (rebuild, stop_reason) = apply_walk_commit(
        bonus,
        commit,
        prompt_tokens,
        &mut output,
        &mut history,
        max_tokens,
    )?;
    Ok(ConstrainedWalk {
        accepted,
        new_tokens,
        output,
        rebuild,
        stop_reason,
    })
}

fn emit_walk_tokens<F: FnMut(i32) -> bool>(
    tokens: &[i32],
    eos_tokens: &[i32],
    max_tokens: usize,
    generated: &mut Vec<i32>,
    committed_history: &mut Vec<i32>,
    mut on_token: F,
) -> Option<GenerationStopReason> {
    for &token in tokens {
        if eos_tokens.contains(&token) {
            return Some(GenerationStopReason::Eos);
        }
        generated.push(token);
        committed_history.push(token);
        if !on_token(token) {
            return Some(GenerationStopReason::CallbackCancelled);
        }
        if generated.len() == max_tokens {
            return Some(GenerationStopReason::MaxTokens);
        }
    }
    None
}

#[derive(Clone, Copy)]
enum MtpPrefill<'a> {
    Text {
        prompt: &'a MlxArray,
    },
    Multimodal {
        prompt: &'a MlxArray,
        input_embeddings: &'a MlxArray,
        position_ids: &'a MlxArray,
        rope_delta: i32,
    },
}

struct ActiveMtpState {
    next_hidden: UniquePtr<MlxArray>,
    bonus: i32,
}

fn prefill_for_input(
    model: &Qwen35Model,
    prefill_input: MtpPrefill<'_>,
) -> Result<crate::qwen3_5::Qwen35MtpPrefill, String> {
    match prefill_input {
        MtpPrefill::Text { prompt } => Ok(model.forward_mtp_prefill(prompt)),
        MtpPrefill::Multimodal {
            prompt,
            input_embeddings,
            position_ids,
            rope_delta,
        } => model.forward_mtp_prefill_with_embeddings(
            prompt,
            input_embeddings,
            position_ids,
            rope_delta,
        ),
    }
}

fn seed_drafter_from_prefill(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prefill_input: MtpPrefill<'_>,
    hidden: &MlxArray,
    first_token: i32,
) {
    match prefill_input {
        MtpPrefill::Text { prompt } => {
            drafter.prefill_from_target_hidden(model, prompt, hidden, first_token);
        }
        MtpPrefill::Multimodal {
            prompt,
            input_embeddings,
            position_ids,
            rope_delta,
        } => drafter.prefill_from_target_hidden_with_embeddings(
            model,
            prompt,
            input_embeddings,
            hidden,
            first_token,
            position_ids,
            rope_delta,
        ),
    }
}

fn rebuild_mtp_state(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prefill_input: MtpPrefill<'_>,
    output: &[i32],
) -> Result<ActiveMtpState, String> {
    let first_token = *output
        .first()
        .ok_or_else(|| "cannot rebuild MTP state for an empty output".to_string())?;
    let prefill = prefill_for_input(model, prefill_input)?;
    mlxcel_core::eval(&prefill.hidden);
    seed_drafter_from_prefill(model, drafter, prefill_input, &prefill.hidden, first_token);
    let prefill_shape = mlxcel_core::array_shape(&prefill.hidden);
    let last = prefill_shape[1] - 1;
    let mut next_hidden = mlxcel_core::slice(
        &prefill.hidden,
        &[0, last, 0],
        &[prefill_shape[0], last + 1, prefill_shape[2]],
    );

    if output.len() > 1 {
        let cached_output = &output[..output.len() - 1];
        let verify_input = mlxcel_core::from_slice_i32(
            cached_output,
            &[1, i32::try_from(cached_output.len()).unwrap_or(i32::MAX)],
        );
        let verify = model.forward_mtp_verify(&verify_input);
        let draft_tokens = &output[1..output.len() - 1];
        drafter.accept_verified_tokens(
            model,
            &verify.hidden,
            draft_tokens,
            draft_tokens.len(),
            output,
        );
        let hidden_shape = mlxcel_core::array_shape(&verify.hidden);
        let accepted = i32::try_from(draft_tokens.len()).unwrap_or(i32::MAX);
        next_hidden = mlxcel_core::slice(
            &verify.hidden,
            &[0, accepted, 0],
            &[hidden_shape[0], accepted + 1, hidden_shape[2]],
        );
    }

    Ok(ActiveMtpState {
        next_hidden,
        bonus: *output.last().expect("non-empty output"),
    })
}

pub(crate) struct Qwen35MtpGenerator;

impl Qwen35MtpGenerator {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn generate_streaming<F: FnMut(i32) -> bool>(
        &mut self,
        model: &Qwen35Model,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        constraint: Option<&mut dyn TokenConstraint>,
        on_token: F,
    ) -> Result<MtpGeneration, String> {
        let prompt = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        self.generate_streaming_for_prefill(
            model,
            prompt_tokens,
            MtpPrefill::Text { prompt: &prompt },
            max_tokens,
            sampling,
            block_size,
            constraint,
            on_token,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_streaming_with_embeddings<F: FnMut(i32) -> bool>(
        &mut self,
        model: &Qwen35Model,
        prompt_tokens: &[i32],
        input_embeddings: &MlxArray,
        position_ids: &MlxArray,
        rope_delta: i32,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        constraint: Option<&mut dyn TokenConstraint>,
        on_token: F,
    ) -> Result<MtpGeneration, String> {
        let prompt = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        self.generate_streaming_for_prefill(
            model,
            prompt_tokens,
            MtpPrefill::Multimodal {
                prompt: &prompt,
                input_embeddings,
                position_ids,
                rope_delta,
            },
            max_tokens,
            sampling,
            block_size,
            constraint,
            on_token,
        )
    }

    fn generate_streaming_for_prefill<F: FnMut(i32) -> bool>(
        &mut self,
        model: &Qwen35Model,
        prompt_tokens: &[i32],
        prefill_input: MtpPrefill<'_>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        constraint: Option<&mut dyn TokenConstraint>,
        mut on_token: F,
    ) -> Result<MtpGeneration, String> {
        assert!(!prompt_tokens.is_empty(), "MTP prompt must not be empty");
        assert!(block_size >= 2, "MTP block size must be at least 2");
        if let Some(constraint) = constraint {
            return self.generate_streaming_constrained_for_prefill(
                model,
                prompt_tokens,
                prefill_input,
                max_tokens,
                sampling,
                block_size,
                constraint,
                on_token,
            );
        }
        let drafter = model
            .mtp()
            .expect("Qwen35MtpGenerator requires a bundled MTP head");
        let mut sampling = sampling.clone();
        sampling
            .token_bias
            .suppress_tokens(&model.output_suppressed_token_ids());
        seed_rng_if_needed(&sampling);
        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);
        if max_tokens == 0 {
            return Ok(MtpGeneration {
                token_ids: Vec::new(),
                stats: MtpGenerationStats::default(),
                stop_reason: GenerationStopReason::MaxTokens,
            });
        }

        let prefill = match prefill_input {
            MtpPrefill::Text { prompt } => model.forward_mtp_prefill(prompt),
            MtpPrefill::Multimodal {
                prompt,
                input_embeddings,
                position_ids,
                rope_delta,
            } => model
                .forward_mtp_prefill_with_embeddings(
                    prompt,
                    input_embeddings,
                    position_ids,
                    rope_delta,
                )
                .expect("MTP multimodal prefill requires prepared MRoPE state"),
        };
        let (first_token, _) =
            sample_token_optimized(&prefill.first_logits, &sampling, prompt_tokens);
        mlxcel_core::eval(&first_token);
        mlxcel_core::eval(&prefill.hidden);
        let first_token = mlxcel_core::item_i32(&first_token);

        let mut generated = Vec::with_capacity(max_tokens);
        let mut history = prompt_tokens.to_vec();
        let mut mtp_stats = MtpGenerationStats::default();
        let mut stop_reason = GenerationStopReason::MaxTokens;
        if eos_tokens.contains(&first_token) {
            stop_reason = GenerationStopReason::Eos;
        } else {
            generated.push(first_token);
            history.push(first_token);
            if !on_token(first_token) {
                stop_reason = GenerationStopReason::CallbackCancelled;
            }
        }
        let decode_start = Instant::now();

        if generated.len() < max_tokens && stop_reason == GenerationStopReason::MaxTokens {
            match prefill_input {
                MtpPrefill::Text { prompt } => {
                    drafter.prefill_from_target_hidden(model, prompt, &prefill.hidden, first_token);
                }
                MtpPrefill::Multimodal {
                    prompt,
                    input_embeddings,
                    position_ids,
                    rope_delta,
                } => drafter.prefill_from_target_hidden_with_embeddings(
                    model,
                    prompt,
                    input_embeddings,
                    &prefill.hidden,
                    first_token,
                    position_ids,
                    rope_delta,
                ),
            }

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
                let proposal_count = round_proposal_count(block_size, remaining);
                if proposal_count == 0 {
                    break;
                }
                let greedy = sampler_is_greedy(&sampling);
                let (draft_tokens, proposal_probs) = if greedy {
                    (

                        drafter.draft_block_greedy(
                            model,
                            bonus,
                            &next_hidden,
                            proposal_count,
                            &sampling,
                            &history,
                            &eos_tokens,
                        ),
                        None,
                    )
                } else {
                    let proposals = drafter.draft_block_stochastic(
                        model,
                        bonus,
                        &next_hidden,
                        proposal_count,
                        &sampling,
                        &history,
                        &eos_tokens,
                    );
                    let tokens = proposals.iter().map(|proposal| proposal.token).collect();
                    (tokens, Some(proposals))
                };
                if draft_tokens.is_empty() {
                    break;
                }

                let mut verify_tokens = Vec::with_capacity(draft_tokens.len() + 1);
                verify_tokens.push(bonus);
                verify_tokens.extend_from_slice(&draft_tokens);
                let verify_input = mlxcel_core::from_slice_i32(
                    &verify_tokens,
                    &[1, i32::try_from(verify_tokens.len()).unwrap_or(i32::MAX)],
                );
                let verify = model.forward_mtp_verify(&verify_input);
                let walk = if let Some(proposals) = proposal_probs.as_deref() {
                    stochastic_walk(
                        proposals,
                        &verify.logits,
                        &sampling,
                        &history,
                        &eos_tokens,
                        remaining,
                    )
                } else {
                    greedy_walk(
                        &draft_tokens,
                        &verify.logits,
                        &sampling,
                        &history,
                        remaining,
                    )
                };
                mtp_stats.accepted_draft_tokens += walk.accepted;
                mtp_stats.proposed_draft_tokens += draft_tokens.len();

                if let Some(reason) = emit_walk_tokens(
                    &walk.new_tokens,
                    &eos_tokens,
                    max_tokens,
                    &mut generated,
                    &mut history,
                    &mut on_token,
                ) {
                    stop_reason = reason;
                    break;
                }

                if walk.accepted < draft_tokens.len() {
                    model.rollback_mtp_verify(
                        &verify.gdn_states,
                        walk.accepted,
                        verify_tokens.len(),
                    );
                }
                drafter.accept_verified_tokens(
                    model,
                    &verify.hidden,
                    &draft_tokens,
                    walk.accepted,
                    &walk.new_tokens,
                );
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
        mtp_stats.decode_time = decode_start.elapsed();
        Ok(MtpGeneration {
            token_ids: generated,
            stats: mtp_stats,
            stop_reason,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn generate_streaming_constrained_for_prefill<F: FnMut(i32) -> bool>(
        &mut self,
        model: &Qwen35Model,
        prompt_tokens: &[i32],
        prefill_input: MtpPrefill<'_>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        constraint: &mut dyn TokenConstraint,
        mut on_token: F,
    ) -> Result<MtpGeneration, String> {
        let drafter = model
            .mtp()
            .expect("Qwen35MtpGenerator requires a bundled MTP head");
        let mut sampling = sampling.clone();
        sampling
            .token_bias
            .suppress_tokens(&model.output_suppressed_token_ids());
        seed_rng_if_needed(&sampling);
        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);
        if max_tokens == 0 {
            return Ok(MtpGeneration {
                token_ids: Vec::new(),
                stats: MtpGenerationStats::default(),
                stop_reason: GenerationStopReason::MaxTokens,
            });
        }

        let mut generated = Vec::with_capacity(max_tokens);
        let mut stats = MtpGenerationStats::default();
        let mut stop_reason = GenerationStopReason::MaxTokens;
        let mut state;

        loop {
            let prefill = prefill_for_input(model, prefill_input)?;
            mlxcel_core::eval(&prefill.hidden);
            let initial = commit_constraint_transaction(constraint, |active| {
                constrained_initial_step(
                    &prefill.first_logits,
                    &sampling,
                    prompt_tokens,
                    &generated,
                    &eos_tokens,
                    max_tokens,
                    active,
                )
            })?;
            generated = initial.output;

            let mut callback_cancelled = false;
            for &token in &initial.new_tokens {
                if eos_tokens.contains(&token) {
                    break;
                }
                if !on_token(token) {
                    callback_cancelled = true;
                    break;
                }
            }
            if initial.new_tokens.is_empty()
                && initial.rebuild
                && generated.last().is_some_and(|&token| !on_token(token))
            {
                callback_cancelled = true;
            }
            if callback_cancelled {
                return Ok(MtpGeneration {
                    token_ids: generated,
                    stats,
                    stop_reason: GenerationStopReason::CallbackCancelled,
                });
            }
            if let Some(reason) = initial.stop_reason {
                if initial.rebuild && !generated.is_empty() {
                    let _ =
                        rebuild_mtp_state(model, drafter, prefill_input, &generated)?;
                }
                return Ok(MtpGeneration {
                    token_ids: generated,
                    stats,
                    stop_reason: reason,
                });
            }
            if generated.is_empty() {
                continue;
            }

            if !initial.rebuild
                && initial.new_tokens.len() == 1
                && generated.as_slice() == initial.new_tokens.as_slice()
            {
                let first_token = generated[0];
                seed_drafter_from_prefill(
                    model,
                    drafter,
                    prefill_input,
                    &prefill.hidden,
                    first_token,
                );
                let shape = mlxcel_core::array_shape(&prefill.hidden);
                let last = shape[1] - 1;
                state = ActiveMtpState {
                    next_hidden: mlxcel_core::slice(
                        &prefill.hidden,
                        &[0, last, 0],
                        &[shape[0], last + 1, shape[2]],
                    ),
                    bonus: first_token,
                };
            } else {
                state = rebuild_mtp_state(model, drafter, prefill_input, &generated)?;
            }
            break;
        }

        while generated.len() < max_tokens && stop_reason == GenerationStopReason::MaxTokens {
            let remaining = max_tokens - generated.len();
            let proposal_count = round_proposal_count(block_size, remaining);
            if proposal_count == 0 {
                break;
            }
            let greedy = sampler_is_greedy(&sampling);
            let (draft_tokens, proposal_probs) = rollback_constraint_transaction(
                constraint,
                |active| {
                    if greedy {
                        drafter
                            .draft_block_greedy_constrained(
                                model,
                                state.bonus,
                                &state.next_hidden,
                                proposal_count,
                                &sampling,
                                prompt_tokens,
                                &generated,
                                &eos_tokens,
                                active,
                            )
                            .map(|tokens| (tokens, None))
                    } else {
                        let proposals = drafter.draft_block_stochastic_constrained(
                            model,
                            state.bonus,
                            &state.next_hidden,
                            proposal_count,
                            &sampling,
                            prompt_tokens,
                            &generated,
                            &eos_tokens,
                            active,
                        )?;
                        let tokens = proposals.iter().map(|proposal| proposal.token).collect();
                        Ok((tokens, Some(proposals)))
                    }
                },
            )?;

            let mut verify_tokens = Vec::with_capacity(draft_tokens.len() + 1);
            verify_tokens.push(state.bonus);
            verify_tokens.extend_from_slice(&draft_tokens);
            let verify_input = mlxcel_core::from_slice_i32(
                &verify_tokens,
                &[1, i32::try_from(verify_tokens.len()).unwrap_or(i32::MAX)],
            );
            let verify = model.forward_mtp_verify(&verify_input);
            let walk = commit_constraint_transaction(constraint, |active| {
                if let Some(proposals) = proposal_probs.as_deref() {
                    constrained_stochastic_walk(
                        proposals,
                        &verify.logits,
                        &sampling,
                        prompt_tokens,
                        &generated,
                        &eos_tokens,
                        max_tokens,
                        active,
                    )
                } else {
                    constrained_greedy_walk(
                        &draft_tokens,
                        &verify.logits,
                        &sampling,
                        prompt_tokens,
                        &generated,
                        &eos_tokens,
                        max_tokens,
                        active,
                    )
                }
            })?;
            stats.accepted_draft_tokens += walk.accepted;
            stats.proposed_draft_tokens += draft_tokens.len();
            generated = walk.output;

            let mut callback_cancelled = false;
            for &token in &walk.new_tokens {
                if eos_tokens.contains(&token) {
                    break;
                }
                if !on_token(token) {
                    callback_cancelled = true;
                    break;
                }
            }
            if walk.new_tokens.is_empty()
                && walk.rebuild
                && generated.last().is_some_and(|&token| !on_token(token))
            {
                callback_cancelled = true;
            }

            if walk.rebuild {
                if generated.is_empty() {
                    let _ = prefill_for_input(model, prefill_input)?;
                } else {
                    state = rebuild_mtp_state(model, drafter, prefill_input, &generated)?;
                }
            } else if walk.stop_reason != Some(GenerationStopReason::Eos)
                && !walk.new_tokens.is_empty()
            {
                if walk.accepted < draft_tokens.len() {
                    model.rollback_mtp_verify(
                        &verify.gdn_states,
                        walk.accepted,
                        verify_tokens.len(),
                    );
                }
                drafter.accept_verified_tokens(
                    model,
                    &verify.hidden,
                    &draft_tokens,
                    walk.accepted,
                    &walk.new_tokens,
                );
                let hidden_shape = mlxcel_core::array_shape(&verify.hidden);
                let accepted = i32::try_from(walk.accepted).unwrap_or(i32::MAX);
                state.next_hidden = mlxcel_core::slice(
                    &verify.hidden,
                    &[0, accepted, 0],
                    &[hidden_shape[0], accepted + 1, hidden_shape[2]],
                );
                state.bonus = *walk
                    .new_tokens
                    .last()
                    .expect("constrained walk emitted at least one token");
            }

            if callback_cancelled {
                stop_reason = GenerationStopReason::CallbackCancelled;
                break;
            }
            if let Some(reason) = walk.stop_reason {
                stop_reason = reason;
                break;
            }
            if generated.is_empty() {
                let prefill = prefill_for_input(model, prefill_input)?;
                let initial = commit_constraint_transaction(constraint, |active| {
                    constrained_initial_step(
                        &prefill.first_logits,
                        &sampling,
                        prompt_tokens,
                        &generated,
                        &eos_tokens,
                        max_tokens,
                        active,
                    )
                })?;
                generated = initial.output;
                if let Some(reason) = initial.stop_reason {
                    stop_reason = reason;
                    break;
                }
                if generated.is_empty() {
                    continue;
                }
                state = rebuild_mtp_state(model, drafter, prefill_input, &generated)?;
            }
        }

        Ok(MtpGeneration {
            token_ids: generated,
            stats,
            stop_reason,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::gated_delta::GatedDeltaCache;
    use crate::qwen_mrope_state::MRopeState;
    use crate::qwen3_5::rollback_plan;
    use crate::qwen3_next::Qwen3NextCache;

    fn logits_rows(rows: &[&[f32]]) -> UniquePtr<MlxArray> {
        let vocab = rows.first().expect("logits row").len();
        assert!(rows.iter().all(|row| row.len() == vocab));
        let values = rows.iter().flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
        mlxcel_core::from_slice_f32(&values, &[1, rows.len() as i32, vocab as i32])
    }

    fn probs(values: &[f32]) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_f32(values, &[1, values.len() as i32])
    }

    fn stochastic_config(seed: u64) -> SamplingConfig {
        SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(seed),
            ..SamplingConfig::default()
        }
    }

    fn chi_square(counts: &[usize], expected: &[f64]) -> f64 {
        let total = counts.iter().sum::<usize>() as f64;
        counts
            .iter()
            .zip(expected)
            .map(|(&observed, &probability)| {
                let expected = total * probability;
                (observed as f64 - expected).powi(2) / expected
            })
            .sum()
    }

    fn fake_stochastic_stream(seed: u64, len: usize) -> Vec<i32> {
        let sampling = stochastic_config(seed);
        seed_rng_if_needed(&sampling);
        let q_logits = logits_rows(&[&[0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln()]]);
        let p_logits = logits_rows(&[&[0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln()]]);
        let mut history = vec![11, 12];
        let mut output = Vec::with_capacity(len);
        for _ in 0..len {
            let (token, proposal_probs) =
                sample_token_with_distribution(&q_logits, &sampling, &history);
            mlxcel_core::eval(&token);
            let proposal = MtpProposal {
                token: mlxcel_core::item_i32(&token),
                proposal_probs,
            };
            let walk = stochastic_walk(&[proposal], &p_logits, &sampling, &history, &[], 1);
            let token = walk.new_tokens[0];
            output.push(token);
            history.push(token);
        }
        output
    }

    struct RecordingConstraint {
        committed: Vec<i32>,
        transaction: Option<Vec<i32>>,
        masks: Vec<Vec<i32>>,
        splice_on: Option<(i32, ConstraintCommit)>,
    }

    impl RecordingConstraint {
        fn active(&mut self) -> &mut Vec<i32> {
            self.transaction.as_mut().unwrap_or(&mut self.committed)
        }
    }

    impl TokenConstraint for RecordingConstraint {
        fn begin_transaction(&mut self) -> Result<(), String> {
            if self.transaction.is_some() {
                return Err("recording transaction already active".to_string());
            }
            self.transaction = Some(self.committed.clone());
            Ok(())
        }

        fn commit_transaction(&mut self) -> Result<(), String> {
            self.committed = self
                .transaction
                .take()
                .ok_or_else(|| "recording transaction not active".to_string())?;
            Ok(())
        }

        fn rollback_transaction(&mut self) {
            self.transaction = None;
        }

        fn compute_mask(
            &mut self,
            _logits: &MlxArray,
            _token_history: &[i32],
        ) -> Result<ConstraintMask, String> {
            let phase = self.active().len();
            Ok(self
                .masks
                .get(phase)
                .cloned()
                .map(ConstraintMask::Allow)
                .unwrap_or(ConstraintMask::Accept))
        }

        fn commit_token(&mut self, token_id: i32) -> Result<ConstraintCommit, String> {
            self.active().push(token_id);
            Ok(self
                .splice_on
                .as_ref()
                .filter(|(token, _)| *token == token_id)
                .map(|(_, commit)| commit.clone())
                .unwrap_or_else(|| ConstraintCommit::token(token_id)))
        }
    }

    fn recording_constraint(phases: usize, vocab: i32) -> RecordingConstraint {
        RecordingConstraint {
            committed: Vec::new(),
            transaction: None,
            masks: vec![(0..vocab).collect(); phases],
            splice_on: None,
        }
    }


    #[test]
    fn acceptance_percentage_handles_zero_and_partial() {
        assert_eq!(MtpGenerationStats::default().acceptance_percentage(), 0.0);
        assert_eq!(
            MtpGenerationStats {
                accepted_draft_tokens: 1,
                proposed_draft_tokens: 4,
                decode_time: Duration::ZERO,
            }
            .acceptance_percentage(),
            25.0
        );
    }

    #[test]
    fn constrained_greedy_walk_commits_full_partial_and_bonus_paths() {
        let sampling = SamplingConfig::greedy();
        let full_logits = logits_rows(&[
            &[0.0, 10.0, 0.0, 0.0],
            &[0.0, 0.0, 10.0, 0.0],
            &[0.0, 0.0, 0.0, 10.0],
        ]);
        let mut full_constraint = recording_constraint(3, 4);
        let full = commit_constraint_transaction(&mut full_constraint, |active| {
            constrained_greedy_walk(
                &[1, 2],
                &full_logits,
                &sampling,
                &[9],
                &[],
                &[],
                8,
                active,
            )
        })
        .expect("full constrained walk");
        assert_eq!(full.accepted, 2);
        assert_eq!(full.output, vec![1, 2, 3]);
        assert_eq!(full_constraint.committed, vec![1, 2, 3]);

        let partial_logits = logits_rows(&[
            &[0.0, 10.0, 0.0, 0.0],
            &[0.0, 0.0, 10.0, 0.0],
            &[0.0, 0.0, 0.0, 10.0],
        ]);
        let mut partial_constraint = recording_constraint(3, 4);
        let partial = commit_constraint_transaction(&mut partial_constraint, |active| {
            constrained_greedy_walk(
                &[1, 0],
                &partial_logits,
                &sampling,
                &[9],
                &[],
                &[],
                8,
                active,
            )
        })
        .expect("partial constrained walk");
        assert_eq!(partial.accepted, 1);
        assert_eq!(partial.output, vec![1, 2]);
        assert_eq!(partial_constraint.committed, vec![1, 2]);
    }

    #[test]
    fn constrained_stochastic_rejection_commits_residual_not_proposal() {
        let sampling = stochastic_config(19);
        seed_rng_if_needed(&sampling);
        let proposal = MtpProposal {
            token: 0,
            proposal_probs: probs(&[1.0, 0.0]),
        };
        let target_logits = logits_rows(&[&[f32::NEG_INFINITY, 0.0], &[0.0, 0.0]]);
        let mut constraint = recording_constraint(2, 2);
        let walk = commit_constraint_transaction(&mut constraint, |active| {
            constrained_stochastic_walk(
                &[proposal],
                &target_logits,
                &sampling,
                &[9],
                &[],
                &[],
                4,
                active,
            )
        })
        .expect("residual constrained walk");
        assert_eq!(walk.accepted, 0);
        assert_eq!(walk.output, vec![1]);
        assert_eq!(constraint.committed, vec![1]);
    }

    #[test]
    fn constrained_backtrack_fast_forward_marks_cache_for_rebuild() {
        let sampling = SamplingConfig::greedy();
        let logits = logits_rows(&[&[0.0, 10.0, 0.0]]);
        let mut constraint = recording_constraint(1, 3);
        constraint.splice_on = Some((
            1,
            ConstraintCommit {
                backtrack: 1,
                tokens: vec![2, 0],
                accept: true,
            },
        ));
        let walk = commit_constraint_transaction(&mut constraint, |active| {
            constrained_initial_step(
                &logits,
                &sampling,
                &[9],
                &[1],
                &[],
                8,
                active,
            )
        })
        .expect("spliced initial walk");
        assert_eq!(walk.output, vec![2, 0]);
        assert!(walk.rebuild);
        assert_eq!(
            walk.stop_reason,
            Some(GenerationStopReason::ConstraintAccepted)
        );
    }

    #[test]
    fn constrained_eos_is_not_committed_to_parser_or_output() {
        let sampling = SamplingConfig::greedy();
        let logits = logits_rows(&[&[0.0, 0.0, 10.0]]);
        let mut constraint = recording_constraint(1, 3);
        let walk = commit_constraint_transaction(&mut constraint, |active| {
            constrained_initial_step(
                &logits,
                &sampling,
                &[9],
                &[],
                &[2],
                8,
                active,
            )
        })
        .expect("EOS constrained walk");
        assert!(walk.output.is_empty());
        assert!(constraint.committed.is_empty());
        assert_eq!(walk.stop_reason, Some(GenerationStopReason::Eos));
    }

    #[test]
    fn k_two_k_three_and_budget_clamping_use_verify_block_semantics() {
        assert_eq!(round_proposal_count(2, 8), 1);
        assert_eq!(round_proposal_count(3, 8), 2);
        assert_eq!(round_proposal_count(3, 1), 1);
        assert_eq!(speculative_walk(&[1], &[1, 2], 2).new_tokens, [1, 2]);
        assert_eq!(
            speculative_walk(&[1, 2], &[1, 2, 3], 3).new_tokens,
            [1, 2, 3]
        );
        let clamped = speculative_walk(&[1, 2], &[1, 2, 3], 1);
        assert_eq!(clamped.accepted, 2);
        assert_eq!(clamped.new_tokens, [1]);
    }

    #[test]
    fn stochastic_walk_covers_zero_partial_and_full_acceptance() {
        let sampling = stochastic_config(7);
        let reject = stochastic_walk(
            &[MtpProposal {
                token: 0,
                proposal_probs: probs(&[1.0, 0.0]),
            }],
            &logits_rows(&[&[f32::NEG_INFINITY, 0.0]]),
            &sampling,
            &[],
            &[],
            1,
        );
        assert_eq!((reject.accepted, reject.new_tokens), (0, vec![1]));

        let partial = stochastic_walk(
            &[
                MtpProposal {
                    token: 1,
                    proposal_probs: probs(&[0.0, 1.0]),
                },
                MtpProposal {
                    token: 0,
                    proposal_probs: probs(&[1.0, 0.0]),
                },
            ],
            &logits_rows(&[
                &[f32::NEG_INFINITY, 0.0],
                &[f32::NEG_INFINITY, 0.0],
                &[0.0, f32::NEG_INFINITY],
            ]),
            &sampling,
            &[],
            &[],
            3,
        );
        assert_eq!((partial.accepted, partial.new_tokens), (1, vec![1, 1]));

        let full = stochastic_walk(
            &[MtpProposal {
                token: 1,
                proposal_probs: probs(&[0.0, 1.0]),
            }],
            &logits_rows(&[
                &[f32::NEG_INFINITY, 0.0],
                &[0.0, f32::NEG_INFINITY],
            ]),
            &sampling,
            &[],
            &[],
            2,
        );
        assert_eq!((full.accepted, full.new_tokens), (1, vec![1, 0]));
    }

    #[test]
    fn eos_cancellation_and_rejected_suffixes_only_commit_emitted_tokens() {
        let mut generated = Vec::new();
        let mut history = vec![10, 11];
        let reason = emit_walk_tokens(
            &[7, 99, 8],
            &[99],
            8,
            &mut generated,
            &mut history,
            |_| true,
        );
        assert_eq!(reason, Some(GenerationStopReason::Eos));
        assert_eq!(generated, [7]);
        assert_eq!(history, [10, 11, 7]);

        let rejected_proposals = [20, 21, 22];
        let mut generated = Vec::new();
        let mut history = vec![10, 11];
        let reason =
            emit_walk_tokens(&[42], &[], 8, &mut generated, &mut history, |_| false);
        assert_eq!(reason, Some(GenerationStopReason::CallbackCancelled));
        assert_eq!(history, [10, 11, 42]);
        assert!(
            rejected_proposals
                .iter()
                .all(|proposal| !history.contains(proposal))
        );
    }

    #[test]
    fn seeded_stochastic_streams_repeat_and_different_seeds_diverge() {
        let first = fake_stochastic_stream(1234, 24);
        assert_eq!(first, fake_stochastic_stream(1234, 24));
        assert_ne!(first, fake_stochastic_stream(1235, 24));
    }

    #[test]
    fn modified_rejection_matches_p_and_unconditional_resampling_does_not() {
        const SAMPLES: usize = 6_000;
        let expected = [0.7f64, 0.2, 0.1];
        let q_logits = logits_rows(&[&[0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln()]]);
        let p_logits = logits_rows(&[&[0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln()]]);
        let sampling = stochastic_config(901);
        seed_rng_if_needed(&sampling);
        let mut correct = [0usize; 3];
        for _ in 0..SAMPLES {
            let (token, proposal_probs) =
                sample_token_with_distribution(&q_logits, &sampling, &[]);
            mlxcel_core::eval(&token);
            let walk = stochastic_walk(
                &[MtpProposal {
                    token: mlxcel_core::item_i32(&token),
                    proposal_probs,
                }],
                &p_logits,
                &sampling,
                &[],
                &[],
                1,
            );
            correct[walk.new_tokens[0] as usize] += 1;
        }
        assert!(
            chi_square(&correct, &expected) < 30.0,
            "modified rejection frequencies: {correct:?}"
        );

        let sampling = stochastic_config(902);
        seed_rng_if_needed(&sampling);
        let mut mutant = [0usize; 3];
        for _ in 0..SAMPLES {
            let (token, proposal_probs) =
                sample_token_with_distribution(&q_logits, &sampling, &[]);
            mlxcel_core::eval(&token);
            let token = mlxcel_core::item_i32(&token);
            let target_probs = effective_token_distribution(&p_logits, &sampling, &[]);
            let emitted = match verify_draft_token(&target_probs, &proposal_probs, token) {
                DraftVerdict::Accept => token,
                DraftVerdict::Reject { .. } => {
                    let (unconditional, _) =
                        sample_token_optimized(&p_logits, &sampling, &[]);
                    mlxcel_core::eval(&unconditional);
                    mlxcel_core::item_i32(&unconditional)
                }
            };
            mutant[emitted as usize] += 1;
        }
        assert!(
            chi_square(&mutant, &expected) > 60.0,
            "unconditional mutant frequencies: {mutant:?}"
        );
    }

    #[test]
    fn sampling_filters_apply_to_proposals_and_replacements() {
        let mut sampling = stochastic_config(17);
        sampling.token_bias.suppress_tokens(&[2]);
        let logits = logits_rows(&[&[0.0, 0.0, 100.0]]);
        seed_rng_if_needed(&sampling);
        for _ in 0..256 {
            let (token, proposal_probs) =
                sample_token_with_distribution(&logits, &sampling, &[]);
            mlxcel_core::eval(&token);
            let walk = stochastic_walk(
                &[MtpProposal {
                    token: mlxcel_core::item_i32(&token),
                    proposal_probs,
                }],
                &logits,
                &sampling,
                &[],
                &[],
                1,
            );
            assert_ne!(walk.new_tokens[0], 2);
        }
    }

    #[test]
    fn rejection_reconciles_target_draft_and_mrope_at_zero_partial_full() {
        let prefix = 10;
        let block_size = 5;
        for accepted in [0, 2, 4] {
            let plan = rollback_plan(prefix + block_size as i32, accepted, block_size);
            assert_eq!(plan.final_offset, prefix + accepted as i32 + 1);
            let mut attention = KVCache::new();
            attention.offset = prefix + block_size as i32;
            attention.trim(plan.trim);
            let mut linear = GatedDeltaCache::new();
            linear.offset = plan.final_offset;
            let caches = [
                Qwen3NextCache::Attention(Box::new(attention)),
                Qwen3NextCache::Linear(linear),
            ];
            assert_eq!(caches[0].offset(), plan.final_offset);
            assert_eq!(caches[1].offset(), plan.final_offset);

            let round_appended = block_size - 2;
            let mut draft = KVCache::new();
            draft.offset = prefix + round_appended as i32;
            let kept = trim_draft_cache(&mut draft, round_appended, accepted);
            draft.offset += (accepted - kept + 1) as i32;
            assert_eq!(draft.offset, plan.final_offset);

            let mrope = MRopeState::new();
            mrope.restore(plan.final_offset, None, Some(-3));
            assert_eq!(mrope.position(), plan.final_offset);
            assert_eq!(mrope.rope_delta(), Some(-3));
        }
    }

    fn array_f32(array: &MlxArray) -> Vec<f32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 bytes")))
            .collect()
    }

    #[test]
    fn multimodal_shift_preserves_one_and_multiple_image_embeddings() {
        let one_image = mlxcel_core::from_slice_f32(
            &[1.0, 2.0, 10.0, 11.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 4, 2],
        );
        let bonus = mlxcel_core::from_slice_f32(&[90.0, 91.0], &[1, 1, 2]);
        assert_eq!(
            array_f32(&shift_prompt_embeddings(&one_image, &bonus)),
            [10.0, 11.0, 3.0, 4.0, 5.0, 6.0, 90.0, 91.0]
        );

        let images =
            mlxcel_core::from_slice_f32(&[1.0, 20.0, 2.0, 30.0, 3.0, 4.0], &[1, 6, 1]);
        let bonus = mlxcel_core::from_slice_f32(&[99.0], &[1, 1, 1]);
        assert_eq!(
            array_f32(&shift_prompt_embeddings(&images, &bonus)),
            [20.0, 2.0, 30.0, 3.0, 4.0, 99.0]
        );
    }

    #[test]
    fn signed_rope_delta_is_retained_for_draft_decode_positions() {
        let mut state = Qwen35MtpDraftState::new();
        state.cache.offset = 8;
        state.rope_delta = Some(-3);
        let positions =
            decode_rope_positions(state.cache.offset, 2, state.rope_delta.expect("delta"));
        mlxcel_core::eval(&positions);
        let values = mlxcel_core::array_to_raw_bytes(&positions)
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
            .collect::<Vec<_>>();
        assert_eq!(values, [5, 6, 5, 6, 5, 6]);
    }
}
