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

use crate::portable_snapshot::{
    PortableArray, PortablePromptSnapshot, array_from_portable, array_to_portable,
    portable_model_state,
};
use mlxcel_core::generate::{
    ConstraintCommit, ConstraintMask, GenerationStopReason, LanguageModel, ModelStateSnapshot,
    SamplingConfig, TokenConstraint, mask_logits_to_allowed,
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
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use tracing::info;

use crate::qwen_vl_position::decode_rope_positions;
use crate::qwen3_5::{Qwen35Config, Qwen35DecoderLayer, Qwen35Model};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpGenerationStats {
    pub accepted_draft_tokens: usize,
    pub proposed_draft_tokens: usize,
    /// Wall-clock time spent in the post-prefill MTP decode loop.
    pub decode_time: Duration,
    pub cache_clear_count: usize,
    pub cache_clear_time: Duration,
    pub draft_time: Duration,
    pub target_verify_time: Duration,
    pub walk_time: Duration,
    pub reconcile_time: Duration,
    pub target_forward_calls: usize,
    pub speculative_rounds: usize,
    pub full_state_materializations: usize,
    pub cache_snapshot_count: usize,
}

impl MtpGenerationStats {
    pub fn acceptance_percentage(self) -> f64 {
        if self.proposed_draft_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_draft_tokens as f64 * 100.0
        }
    }

    fn record_round(&mut self, accepted: usize, proposed: usize) {
        self.accepted_draft_tokens += accepted;
        self.proposed_draft_tokens += proposed;
    }

    fn record_cache_clear(&mut self, elapsed: Duration) {
        self.cache_clear_count += 1;
        self.cache_clear_time += elapsed;
    }
}

pub(crate) struct MtpGeneration {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) stats: MtpGenerationStats,
    pub(crate) stop_reason: GenerationStopReason,
    pub(crate) prompt_snapshots: Vec<MtpPromptSnapshot>,
    pub(crate) final_snapshot: Option<MtpPromptSnapshot>,
    pub(crate) cached_tokens: usize,
}

/// Detached target and drafter state at an exact text prompt boundary.
///
/// The target cache represents `token_len` tokens. The drafter cache represents
/// the shifted prompt through `token_len - 1`; it deliberately contains no
/// sampled completion seed or speculative-round state.
pub struct MtpPromptSnapshot {
    target: ModelStateSnapshot,
    draft_keys: Option<UniquePtr<MlxArray>>,
    draft_values: Option<UniquePtr<MlxArray>>,
    draft_offset: i32,
    last_hidden: UniquePtr<MlxArray>,
    continuation_logits: UniquePtr<MlxArray>,
}

impl MtpPromptSnapshot {
    pub fn token_len(&self) -> usize {
        self.target.token_len()
    }

    pub fn nbytes(&self) -> usize {
        self.target.nbytes()
            + self
                .draft_keys
                .as_deref()
                .map(mlxcel_core::array_nbytes)
                .unwrap_or(0)
            + self
                .draft_values
                .as_deref()
                .map(mlxcel_core::array_nbytes)
                .unwrap_or(0)
            + mlxcel_core::array_nbytes(&self.last_hidden)
            + mlxcel_core::array_nbytes(&self.continuation_logits)
    }

    pub(crate) fn to_portable(&self) -> PortablePromptSnapshot {
        PortablePromptSnapshot::Mtp {
            target: portable_model_state(&self.target),
            draft_keys: self
                .draft_keys
                .as_deref()
                .map(|array| array_to_portable(None, array)),
            draft_values: self
                .draft_values
                .as_deref()
                .map(|array| array_to_portable(None, array)),
            draft_offset: self.draft_offset,
            last_hidden: array_to_portable(None, &self.last_hidden),
            continuation_logits: array_to_portable(None, &self.continuation_logits),
        }
    }

    pub(crate) fn from_portable_parts(
        target: ModelStateSnapshot,
        draft_keys: Option<PortableArray>,
        draft_values: Option<PortableArray>,
        draft_offset: i32,
        last_hidden: PortableArray,
        continuation_logits: PortableArray,
    ) -> Result<Self, String> {
        let expected_offset = i32::try_from(
            target
                .token_len()
                .checked_sub(1)
                .ok_or_else(|| "MTP portable target token length must be nonzero".to_string())?,
        )
        .map_err(|_| "MTP portable target token length exceeds i32".to_string())?;
        if draft_offset != expected_offset {
            return Err("MTP portable target and drafter offsets do not match".to_string());
        }
        if draft_keys.is_some() != draft_values.is_some()
            || (expected_offset > 0 && draft_keys.is_none())
        {
            return Err("MTP portable drafter key/value layout is incomplete".to_string());
        }
        if let (Some(keys), Some(values)) = (&draft_keys, &draft_values)
            && (keys.shape.len() != 4 || keys.shape != values.shape)
        {
            return Err("MTP portable drafter key/value shapes do not match".to_string());
        }
        if last_hidden.shape.len() != 3 || last_hidden.shape[0] != 1 || last_hidden.shape[1] != 1 {
            return Err("MTP portable last-hidden layout must be [1, 1, hidden]".to_string());
        }
        if continuation_logits.shape.len() != 3
            || continuation_logits.shape[0] != 1
            || continuation_logits.shape[1] != 1
        {
            return Err(
                "MTP portable continuation-logits layout must be [1, 1, vocab]".to_string(),
            );
        }
        Ok(Self {
            target,
            draft_keys: draft_keys
                .map(|array| array_from_portable(array, None))
                .transpose()?,
            draft_values: draft_values
                .map(|array| array_from_portable(array, None))
                .transpose()?,
            draft_offset,
            last_hidden: array_from_portable(last_hidden, None)?,
            continuation_logits: array_from_portable(continuation_logits, None)?,
        })
    }
}

/// Exact-prefix MTP state supplied by the server cache.
#[derive(Clone, Copy)]
pub struct MtpPrefixReuse<'a> {
    pub snapshot: &'a MtpPromptSnapshot,
    pub cached_tokens: usize,
    /// Accepted response suffix already present in the resume prompt but not in
    /// the target snapshot. It becomes the next speculative-round bonus
    /// without being emitted or singly prefetched.
    pub continuation_token: Option<i32>,
}

struct MtpProposal {
    token: i32,
    proposal_probs: UniquePtr<MlxArray>,
}

struct Qwen35MtpDraftState {
    cache: KVCache,
    seed_hidden: Option<UniquePtr<MlxArray>>,
    rope_delta: Option<i32>,
    round_appended: usize,
}

impl Qwen35MtpDraftState {
    fn new() -> Self {
        Self {
            cache: KVCache::new(),
            seed_hidden: None,
            rope_delta: None,
            round_appended: 0,
        }
    }
}

fn shifted_embedding_range(
    input_embeddings: &MlxArray,
    start: i32,
    end: i32,
    bonus_embedding: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(input_embeddings);
    let stop = if bonus_embedding.is_some() {
        shape[1]
    } else {
        end + 1
    };
    let tail = (start + 1 < stop).then(|| {
        mlxcel_core::slice(
            input_embeddings,
            &[0, start + 1, 0],
            &[shape[0], stop, shape[2]],
        )
    });
    match (tail, bonus_embedding) {
        (Some(tail), Some(bonus)) => mlxcel_core::concatenate(&tail, bonus, 1),
        (Some(tail), None) => tail,
        (None, Some(bonus)) => mlxcel_core::copy(bonus),
        (None, None) => panic!("shifted embedding range must not be empty"),
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
        let decode_positions = if position_ids.is_none() {
            state
                .rope_delta
                .map(|delta| decode_rope_positions(cache_offset, steps, delta))
        } else {
            None
        };
        output = self.layer.forward_full_attention(
            &output,
            None,
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
        _target: &Qwen35Model,
        hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) {
        state.seed_hidden = Some(materialize_detached(mlxcel_core::copy(hidden)));
    }

    fn prefill_target_chunk(
        &self,
        target: &Qwen35Model,
        shifted_embeddings: &MlxArray,
        target_hidden: &MlxArray,
        position_ids: Option<&MlxArray>,
        rope_delta: Option<i32>,
        seed: bool,
    ) {
        let mut state = self.state.borrow_mut();
        state.rope_delta = rope_delta;
        let output =
            self.forward_embeddings(shifted_embeddings, target_hidden, &mut state, position_ids);
        if seed {
            let output_shape = mlxcel_core::array_shape(&output);
            let last = output_shape[1] - 1;
            let last_hidden = mlxcel_core::slice(
                &output,
                &[0, last, 0],
                &[output_shape[0], last + 1, output_shape[2]],
            );
            self.set_seed_from_hidden(target, &last_hidden, &mut state);
        }
    }

    fn draft_seed_hidden(
        &self,
        target: &Qwen35Model,
        last_bonus: i32,
        target_hidden: &MlxArray,
        state: &mut Qwen35MtpDraftState,
    ) -> UniquePtr<MlxArray> {
        state.seed_hidden.take().unwrap_or_else(|| {
            let bonus = mlxcel_core::from_slice_i32(&[last_bonus], &[1, 1]);
            let hidden = self.forward_tokens(target, &bonus, target_hidden, state);
            state.round_appended += 1;
            hidden
        })
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
        let mut hidden = self.draft_seed_hidden(target, last_bonus, target_hidden, &mut state);
        let mut logits = target.project_draft_logits(&hidden);
        let compact = target.has_compact_draft_head();

        while tokens.len() < proposal_count {
            let token_array = if compact && sampler_is_greedy(sampling) {
                mlxcel_core::argmax_last_axis(&logits)
            } else {
                sample_token_optimized(&logits, sampling, &history).0
            };
            mlxcel_core::eval(&token_array);
            let sampled = mlxcel_core::item_i32(&token_array);
            let token = if compact {
                Qwen35Model::map_draft_token(sampled)
            } else {
                sampled
            };
            tokens.push(token);
            if eos_tokens.contains(&token) || tokens.len() == proposal_count {
                break;
            }
            history.push(token);
            let token_array = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            hidden = self.forward_tokens(target, &token_array, &hidden, &mut state);
            state.round_appended += 1;
            logits = target.project_draft_logits(&hidden);
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
        let mut hidden = self.draft_seed_hidden(target, last_bonus, target_hidden, &mut state);
        let mut logits = target.project_logits(&hidden);

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
        let mut hidden = self.draft_seed_hidden(target, last_bonus, target_hidden, &mut state);
        let mut logits = target.project_logits(&hidden);

        while tokens.len() < proposal_count {
            let step = constraint_step(&logits, constraint, &history)?;
            let Some(logits_for_sample) = step.logits() else {
                if let ConstraintStepLogits::Splice(commit) = step {
                    commit.apply_to(&mut output)?;
                }
                break;
            };
            let (token_array, _) = sample_token_optimized(logits_for_sample, sampling, &history);
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
        let mut hidden = self.draft_seed_hidden(target, last_bonus, target_hidden, &mut state);
        let mut logits = target.project_logits(&hidden);

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
            let hidden = materialize_detached(mlxcel_core::slice(
                verify_hidden,
                &[0, keep_appended as i32, 0],
                &[hidden_shape[0], end, hidden_shape[2]],
            ));
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
        state.cache.materialize_state();
        state.round_appended = 0;
    }

    fn materialize_state(&self) {
        let mut state = self.state.borrow_mut();
        state.cache.materialize_state();
        if let Some(hidden) = state.seed_hidden.take() {
            state.seed_hidden = Some(materialize_detached(hidden));
        }
    }

    fn capture_prompt_snapshot(
        &self,
        target: ModelStateSnapshot,
        token_len: usize,
        last_hidden: &MlxArray,
        continuation_logits: &MlxArray,
    ) -> Option<MtpPromptSnapshot> {
        let expected_offset = i32::try_from(token_len.checked_sub(1)?).ok()?;
        self.materialize_state();
        let mut state = self.state.borrow_mut();
        if state.cache.offset < expected_offset {
            return None;
        }
        if state.cache.offset > expected_offset {
            let excess = state.cache.offset - expected_offset;
            state.cache.trim(excess);
        }
        state.round_appended = 0;
        state.seed_hidden = None;
        state.rope_delta = None;
        if state.cache.offset != expected_offset {
            return None;
        }
        let detached = |array: &MlxArray| materialize_detached(mlxcel_core::copy(array));
        Some(MtpPromptSnapshot {
            target,
            draft_keys: state.cache.keys.as_deref().map(detached),
            draft_values: state.cache.values.as_deref().map(detached),
            draft_offset: state.cache.offset,
            last_hidden: detached(last_hidden),
            continuation_logits: detached(continuation_logits),
        })
    }

    fn restore_prompt_snapshot(
        &self,
        snapshot: &MtpPromptSnapshot,
        token_len: usize,
    ) -> Result<(), String> {
        let expected_offset = i32::try_from(
            token_len
                .checked_sub(1)
                .ok_or_else(|| "MTP snapshots require a non-empty prompt".to_string())?,
        )
        .map_err(|_| "MTP snapshot token length exceeds i32".to_string())?;
        if snapshot.target.token_len() != token_len || snapshot.draft_offset != expected_offset {
            return Err(
                "MTP snapshot target/drafter offsets do not match the cached prefix".to_string(),
            );
        }
        if snapshot.draft_keys.is_some() != snapshot.draft_values.is_some()
            || (expected_offset > 0 && snapshot.draft_keys.is_none())
        {
            return Err("MTP snapshot drafter KV layout is incomplete".to_string());
        }
        let mut cache = KVCache::new();
        cache.keys = snapshot.draft_keys.as_deref().map(mlxcel_core::copy);
        cache.values = snapshot.draft_values.as_deref().map(mlxcel_core::copy);
        cache.offset = expected_offset;
        *self.state.borrow_mut() = Qwen35MtpDraftState {
            cache,
            seed_hidden: None,
            rope_delta: None,
            round_appended: 0,
        };
        Ok(())
    }
}

fn materialize_borrowed(array: &MlxArray) {
    mlxcel_core::eval(array);
    unsafe { mlxcel_core::detach_all(&[array as *const MlxArray]) };
}

fn materialize_detached(array: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::eval(&array);
    let ptr = array
        .as_ref()
        .expect("materialized MLX array must be non-null") as *const MlxArray;
    unsafe { mlxcel_core::detach_all(&[ptr]) };
    array
}

fn mtp_round_reaches_cache_clear(previous: usize, emitted: usize, interval: usize) -> bool {
    mlxcel_core::memory::should_clear_cache_crossing(previous, emitted, interval)
}
/// Number of target tokens to retain when a speculative round did not emit
/// the bonus token, such as the final max-token round.
fn target_cache_accepted_count(accepted: usize, emitted: usize) -> usize {
    accepted.saturating_sub(usize::from(emitted <= accepted))
}

const MTP_STATE_MATERIALIZE_INTERVAL: usize = 128;

// Variable MTP verify shapes accumulate reusable Metal buffers much faster than
// ordinary one-token decode. Keep a bounded cache, but retain those buffers
// until the bound is reached so the common shapes can be reused.
const MTP_CACHE_WATERMARK_BYTES: u64 = 3 * 256 * 1024 * 1024;

fn mtp_cache_should_clear(
    previous: usize,
    emitted: usize,
    configured_interval: usize,
    cache_bytes: u64,
    cache_watermark_bytes: u64,
) -> bool {
    cache_bytes >= cache_watermark_bytes
        || mtp_round_reaches_cache_clear(previous, emitted, configured_interval)
}

fn clear_mtp_cache_if_needed(previous: usize, emitted: usize) -> Option<Duration> {
    let memory = mlxcel_core::memory::snapshot();
    if !mtp_cache_should_clear(
        previous,
        emitted,
        mlxcel_core::memory::cache_clear_interval(),
        memory.cache_bytes,
        MTP_CACHE_WATERMARK_BYTES,
    ) {
        return None;
    }
    if emitted <= 64 {
        info!(
            phase = "mtp.decode.cache_clear.before",
            tokens = emitted,
            active_bytes = memory.active_bytes,
            cache_bytes = memory.cache_bytes,
            peak_bytes = memory.peak_bytes,
            limit_bytes = memory.limit_bytes,
        );
    }
    let started = Instant::now();
    mlxcel_core::clear_memory_cache();
    let elapsed = started.elapsed();
    if emitted <= 64 {
        log_mtp_memory("mtp.decode.cache_clear.after", emitted);
    }
    Some(elapsed)
}

fn log_mtp_memory(phase: &'static str, tokens: usize) {
    let memory = mlxcel_core::memory::snapshot();
    info!(
        phase,
        tokens,
        active_bytes = memory.active_bytes,
        cache_bytes = memory.cache_bytes,
        peak_bytes = memory.peak_bytes,
        limit_bytes = memory.limit_bytes,
    );
}

fn finish_mtp_request(model: &Qwen35Model) {
    if let Some(drafter) = model.mtp() {
        drafter.materialize_state();
    }
    model.materialize_mtp_cache_state();
    mlxcel_core::clear_memory_cache();
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
        ConstraintMask::Allow(allowed) => Ok(ConstraintStepLogits::Masked(mask_logits_to_allowed(
            logits, &allowed,
        )?)),
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

pub(crate) fn greedy_walk(
    draft_tokens: &[i32],
    verify_logits: &MlxArray,
    sampling: &SamplingConfig,
    committed_history: &[i32],
    max_new_tokens: usize,
) -> WalkResult {
    let history_independent = sampling.repetition_penalty == 1.0
        && sampling.dry_multiplier == 0.0
        && sampling.frequency_penalty == 0.0
        && sampling.presence_penalty == 0.0
        && sampling.xtc_probability == 0.0;
    let mut target_tokens = Vec::with_capacity(draft_tokens.len() + 1);
    if history_independent {
        let biased_logits =
            mlxcel_core::sampling::apply_token_bias(verify_logits, &sampling.token_bias);
        let targets = mlxcel_core::argmax_last_axis(&biased_logits);
        mlxcel_core::eval(&targets);
        let shape = mlxcel_core::array_shape(&targets);
        for position in 0..=draft_tokens.len() {
            let token = mlxcel_core::slice(
                &targets,
                &[0, position as i32],
                &[shape[0], position as i32 + 1],
            );
            target_tokens.push(mlxcel_core::item_i32(&token));
        }
    } else {
        let mut history = committed_history.to_vec();
        for position in 0..=draft_tokens.len() {
            let logits = logits_at(verify_logits, position);
            let (token, _) = sample_token_optimized(&logits, sampling, &history);
            mlxcel_core::eval(&token);
            target_tokens.push(mlxcel_core::item_i32(&token));
            if position < draft_tokens.len() {
                history.push(draft_tokens[position]);
            }
        }
    }
    speculative_walk(draft_tokens, &target_tokens, max_new_tokens)
}

/// Greedy verification with proposals retained on-device until the target
/// posterior has been evaluated.
pub(crate) fn greedy_walk_device_proposals(
    draft_tokens: &MlxArray,
    verify_logits: &MlxArray,
    sampling: &SamplingConfig,
    committed_history: &[i32],
    max_new_tokens: usize,
) -> (WalkResult, Vec<i32>) {
    let materialize_ids = |array: &MlxArray| {
        mlxcel_core::eval(array);
        mlxcel_core::array_evaluated_bytes(array)
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 token bytes")))
            .collect::<Vec<_>>()
    };
    let history_independent = sampling.repetition_penalty == 1.0
        && sampling.dry_multiplier == 0.0
        && sampling.frequency_penalty == 0.0
        && sampling.presence_penalty == 0.0
        && sampling.xtc_probability == 0.0;
    if !history_independent {
        let draft_tokens = materialize_ids(draft_tokens);
        let walk = greedy_walk(
            &draft_tokens,
            verify_logits,
            sampling,
            committed_history,
            max_new_tokens,
        );
        return (walk, draft_tokens);
    }

    let biased_logits =
        mlxcel_core::sampling::apply_token_bias(verify_logits, &sampling.token_bias);
    let targets = mlxcel_core::argmax_last_axis(&biased_logits);
    mlxcel_core::async_eval(&targets);
    let draft_tokens = materialize_ids(draft_tokens);
    let target_tokens = materialize_ids(&targets);
    (
        speculative_walk(&draft_tokens, &target_tokens, max_new_tokens),
        draft_tokens,
    )
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
        let target_probs =
            effective_token_distribution(&logits_at(verify_logits, position), sampling, &history);
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
    let mut history =
        Vec::with_capacity(prompt_tokens.len() + output.len() + draft_tokens.len() + 1);
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
        let target_token =
            match verify_draft_token(&target_probs, &proposal.proposal_probs, proposal.token) {
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

pub(crate) fn emit_walk_tokens<F: FnMut(i32) -> bool>(
    tokens: &[i32],
    eos_tokens: &[i32],
    max_tokens: usize,
    generated: &mut Vec<i32>,
    committed_history: &mut Vec<i32>,
    mut on_token: F,
) -> Option<GenerationStopReason> {
    let mut callback_cancelled = false;
    for &token in tokens {
        if eos_tokens.contains(&token) {
            return Some(if callback_cancelled {
                GenerationStopReason::CallbackCancelled
            } else {
                GenerationStopReason::Eos
            });
        }
        generated.push(token);
        committed_history.push(token);
        if !on_token(token) {
            callback_cancelled = true;
        }
        if generated.len() == max_tokens {
            return Some(if callback_cancelled {
                GenerationStopReason::CallbackCancelled
            } else {
                GenerationStopReason::MaxTokens
            });
        }
    }
    callback_cancelled.then_some(GenerationStopReason::CallbackCancelled)
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

fn prompt_for_prefill(prefill: MtpPrefill<'_>) -> &MlxArray {
    match prefill {
        MtpPrefill::Text { prompt } | MtpPrefill::Multimodal { prompt, .. } => prompt,
    }
}

fn shifted_embeddings_for_range(
    model: &Qwen35Model,
    prefill: MtpPrefill<'_>,
    start: i32,
    end: i32,
    bonus: Option<i32>,
) -> UniquePtr<MlxArray> {
    let prompt = prompt_for_prefill(prefill);
    let prompt_shape = mlxcel_core::array_shape(prompt);
    let prompt_len = prompt_shape[1];
    match prefill {
        MtpPrefill::Text { .. } => {
            let shifted_ids = if let Some(bonus) = bonus {
                let bonus = mlxcel_core::from_slice_i32(&[bonus], &[1, 1]);
                if start + 1 < prompt_len {
                    let tail =
                        mlxcel_core::slice(prompt, &[0, start + 1], &[prompt_shape[0], prompt_len]);
                    mlxcel_core::concatenate(&tail, &bonus, 1)
                } else {
                    bonus
                }
            } else {
                mlxcel_core::slice(prompt, &[0, start + 1], &[prompt_shape[0], end + 1])
            };
            model.embed_tokens.forward(&shifted_ids)
        }
        MtpPrefill::Multimodal {
            input_embeddings, ..
        } => {
            let bonus_embedding = bonus.map(|bonus| {
                let bonus = mlxcel_core::from_slice_i32(&[bonus], &[1, 1]);
                model.embed_tokens.forward(&bonus)
            });
            shifted_embedding_range(input_embeddings, start, end, bonus_embedding.as_deref())
        }
    }
}

fn position_ids_for_range(
    prefill: MtpPrefill<'_>,
    start: i32,
    end: i32,
) -> Option<UniquePtr<MlxArray>> {
    let MtpPrefill::Multimodal { position_ids, .. } = prefill else {
        return None;
    };
    let shape = mlxcel_core::array_shape(position_ids);
    Some(mlxcel_core::slice(
        position_ids,
        &[0, 0, start],
        &[shape[0], shape[1], end],
    ))
}

fn prefill_for_input(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prefill_input: MtpPrefill<'_>,
) -> Result<crate::qwen3_5::Qwen35MtpPrefill, String> {
    drafter.reset();
    let prompt = prompt_for_prefill(prefill_input);
    let (embeddings, positions, rope_delta) = match prefill_input {
        MtpPrefill::Text { .. } => (None, None, None),
        MtpPrefill::Multimodal {
            input_embeddings,
            position_ids,
            rope_delta,
            ..
        } => (Some(input_embeddings), Some(position_ids), Some(rope_delta)),
    };
    let prefill = model.forward_mtp_prefill_chunks(
        prompt,
        embeddings,
        positions,
        rope_delta,
        |start, end, hidden| {
            let shifted = shifted_embeddings_for_range(model, prefill_input, start, end, None);
            let chunk_positions = position_ids_for_range(prefill_input, start, end);
            drafter.prefill_target_chunk(
                model,
                &shifted,
                hidden,
                chunk_positions.as_deref(),
                rope_delta,
                false,
            );
            materialize_borrowed(hidden);
            drafter.materialize_state();
            model.materialize_mtp_cache_state();
            mlxcel_core::clear_memory_cache();
            log_mtp_memory("mtp.prefill.chunk_complete", end as usize);
        },
    )?;
    let prompt_len = mlxcel_core::array_shape(prompt)[1];
    let final_shape = mlxcel_core::array_shape(&prefill.hidden);
    let final_len = final_shape[1];
    if final_len > 1 {
        let start = prompt_len - final_len;
        let shifted =
            shifted_embeddings_for_range(model, prefill_input, start, prompt_len - 1, None);
        let hidden = mlxcel_core::slice(
            &prefill.hidden,
            &[0, 0, 0],
            &[final_shape[0], final_len - 1, final_shape[2]],
        );
        let positions = position_ids_for_range(prefill_input, start, prompt_len - 1);
        drafter.prefill_target_chunk(
            model,
            &shifted,
            &hidden,
            positions.as_deref(),
            rope_delta,
            false,
        );
    }
    drafter.materialize_state();
    model.materialize_mtp_cache_state();
    Ok(prefill)
}

fn prefill_text_with_reuse(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prompt_tokens: &[i32],
    reuse: Option<MtpPrefixReuse<'_>>,
) -> Result<(crate::qwen3_5::Qwen35MtpPrefill, usize), String> {
    let Some(reuse) = reuse else {
        tracing::info!(
            phase = "prefill.started",
            prompt_tokens = prompt_tokens.len(),
            requested_cached_tokens = 0,
            prefix_cached_tokens = 0,
            prefill_start = 0,
            prefill_tokens = prompt_tokens.len(),
        );
        let prompt = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        return prefill_for_input(model, drafter, MtpPrefill::Text { prompt: &prompt })
            .map(|prefill| (prefill, 0));
    };
    if reuse.cached_tokens == 0 || reuse.cached_tokens > prompt_tokens.len() {
        return Err("MTP cached token count is outside the prompt".to_string());
    }
    model.restore_sequence_state(
        mlxcel_core::cache::SequenceId::from_raw(0),
        &reuse.snapshot.target,
    )?;
    drafter.restore_prompt_snapshot(reuse.snapshot, reuse.cached_tokens)?;
    if let Some(continuation_token) = reuse.continuation_token {
        if reuse.cached_tokens + 1 != prompt_tokens.len()
            || prompt_tokens[reuse.cached_tokens] != continuation_token
        {
            return Err("MTP response continuation token does not match the prompt".to_string());
        }
        tracing::info!(
            phase = "prefill.started",
            prompt_tokens = prompt_tokens.len(),
            requested_cached_tokens = reuse.cached_tokens,
            prefix_cached_tokens = reuse.cached_tokens,
            prefill_start = reuse.cached_tokens,
            prefill_tokens = 0,
        );
        return Ok((
            crate::qwen3_5::Qwen35MtpPrefill {
                hidden: mlxcel_core::copy(&reuse.snapshot.last_hidden),
                first_logits: mlxcel_core::copy(&reuse.snapshot.continuation_logits),
            },
            reuse.cached_tokens,
        ));
    }
    if reuse.cached_tokens == prompt_tokens.len() {
        tracing::info!(
            phase = "prefill.started",
            prompt_tokens = prompt_tokens.len(),
            requested_cached_tokens = reuse.cached_tokens,
            prefix_cached_tokens = reuse.cached_tokens,
            prefill_start = reuse.cached_tokens,
            prefill_tokens = 0,
        );
        return Ok((
            crate::qwen3_5::Qwen35MtpPrefill {
                hidden: mlxcel_core::copy(&reuse.snapshot.last_hidden),
                first_logits: mlxcel_core::copy(&reuse.snapshot.continuation_logits),
            },
            reuse.cached_tokens,
        ));
    }

    let suffix = &prompt_tokens[reuse.cached_tokens..];
    tracing::info!(
        phase = "prefill.started",
        prompt_tokens = prompt_tokens.len(),
        requested_cached_tokens = reuse.cached_tokens,
        prefix_cached_tokens = reuse.cached_tokens,
        prefill_start = reuse.cached_tokens,
        prefill_tokens = suffix.len(),
    );
    let suffix_ids = mlxcel_core::from_slice_i32(
        suffix,
        &[1, i32::try_from(suffix.len()).unwrap_or(i32::MAX)],
    );
    let mut previous_hidden = mlxcel_core::copy(&reuse.snapshot.last_hidden);
    let prefill = model.forward_mtp_text_suffix_chunks(&suffix_ids, |ids, hidden| {
        let shape = mlxcel_core::array_shape(hidden);
        let target_hidden = if shape[1] == 1 {
            mlxcel_core::copy(&previous_hidden)
        } else {
            let prefix_hidden =
                mlxcel_core::slice(hidden, &[0, 0, 0], &[shape[0], shape[1] - 1, shape[2]]);
            mlxcel_core::concatenate(&previous_hidden, &prefix_hidden, 1)
        };
        let embeddings = model.embed_tokens.forward(ids);
        drafter.prefill_target_chunk(model, &embeddings, &target_hidden, None, None, false);
        previous_hidden = materialize_detached(mlxcel_core::slice(
            hidden,
            &[0, shape[1] - 1, 0],
            &[shape[0], shape[1], shape[2]],
        ));
        drafter.materialize_state();
        model.materialize_mtp_cache_state();
    })?;
    Ok((prefill, reuse.cached_tokens))
}

fn capture_mtp_prompt_snapshot(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    token_len: usize,
    prefill: &crate::qwen3_5::Qwen35MtpPrefill,
) -> Option<MtpPromptSnapshot> {
    let target =
        model.snapshot_sequence_state(mlxcel_core::cache::SequenceId::from_raw(0), token_len)?;
    let shape = mlxcel_core::array_shape(&prefill.hidden);
    let last = shape[1] - 1;

    let last_hidden = mlxcel_core::slice(
        &prefill.hidden,
        &[0, last, 0],
        &[shape[0], last + 1, shape[2]],
    );
    drafter.capture_prompt_snapshot(target, token_len, &last_hidden, &prefill.first_logits)
}
fn prefill_text_with_checkpoints(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prompt_tokens: &[i32],
    reuse: Option<MtpPrefixReuse<'_>>,
    checkpoint_token_lengths: &[usize],
) -> Result<
    (
        crate::qwen3_5::Qwen35MtpPrefill,
        usize,
        Vec<MtpPromptSnapshot>,
    ),
    String,
> {
    let cached_tokens = reuse.map_or(0, |value| value.cached_tokens);
    let mut source_reuse = reuse;
    let mut latest_checkpoint = None;
    let mut snapshots = Vec::with_capacity(checkpoint_token_lengths.len());
    let mut final_prefill = None;
    let boundaries = checkpoint_token_lengths
        .iter()
        .copied()
        .filter(|&length| length > cached_tokens)
        .chain(std::iter::once(prompt_tokens.len()));

    for token_len in boundaries {
        if final_prefill.is_some() {
            break;
        }
        let active_reuse = latest_checkpoint
            .as_ref()
            .map(|snapshot| MtpPrefixReuse {
                snapshot,
                cached_tokens: snapshot.token_len(),
                continuation_token: None,
            })
            .or_else(|| source_reuse.take());
        let (prefill, _) =
            prefill_text_with_reuse(model, drafter, &prompt_tokens[..token_len], active_reuse)?;
        if let Some(previous) = latest_checkpoint.take() {
            snapshots.push(previous);
        }
        let requested = checkpoint_token_lengths.binary_search(&token_len).is_ok();
        if requested {
            let snapshot = capture_mtp_prompt_snapshot(model, drafter, token_len, &prefill)
                .ok_or_else(|| format!("failed to capture MTP checkpoint at {token_len} tokens"))?;
            if token_len == prompt_tokens.len() {
                snapshots.push(snapshot);
            } else {
                latest_checkpoint = Some(snapshot);
            }
        }
        if token_len == prompt_tokens.len() {
            final_prefill = Some(prefill);
        }
    }

    Ok((
        final_prefill.expect("full prompt boundary is always included"),
        cached_tokens,
        snapshots,
    ))
}

fn finish_drafter_prefill(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prefill_input: MtpPrefill<'_>,
    prefill: crate::qwen3_5::Qwen35MtpPrefill,
    first_token: i32,
) -> UniquePtr<MlxArray> {
    let prompt_len = mlxcel_core::array_shape(prompt_for_prefill(prefill_input))[1];
    let final_shape = mlxcel_core::array_shape(&prefill.hidden);
    let last = final_shape[1] - 1;
    let last_hidden = materialize_detached(mlxcel_core::slice(
        &prefill.hidden,
        &[0, last, 0],
        &[final_shape[0], last + 1, final_shape[2]],
    ));
    let bonus = mlxcel_core::from_slice_i32(&[first_token], &[1, 1]);
    let bonus_embedding = model.embed_tokens.forward(&bonus);
    let positions = position_ids_for_range(prefill_input, prompt_len - 1, prompt_len);
    let rope_delta = match prefill_input {
        MtpPrefill::Text { .. } => None,
        MtpPrefill::Multimodal { rope_delta, .. } => Some(rope_delta),
    };
    drafter.prefill_target_chunk(
        model,
        &bonus_embedding,
        &last_hidden,
        positions.as_deref(),
        rope_delta,
        true,
    );
    drafter.materialize_state();
    model.materialize_mtp_cache_state();
    drop(prefill);
    mlxcel_core::clear_memory_cache();
    log_mtp_memory("mtp.prefill.final_complete", prompt_len as usize);
    last_hidden
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
    let prefill = prefill_for_input(model, drafter, prefill_input)?;
    let mut next_hidden =
        finish_drafter_prefill(model, drafter, prefill_input, prefill, first_token);

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
        next_hidden = materialize_detached(mlxcel_core::slice(
            &verify.hidden,
            &[0, accepted, 0],
            &[hidden_shape[0], accepted + 1, hidden_shape[2]],
        ));
        model.materialize_mtp_cache_state();
    }

    Ok(ActiveMtpState {
        next_hidden,
        bonus: *output.last().expect("non-empty output"),
    })
}

fn capture_mtp_snapshot_from_verify(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prompt_tokens: usize,
    generated_tokens: usize,
    verify_hidden: &MlxArray,
    verify_logits: &MlxArray,
    accepted: usize,
) -> Result<MtpPromptSnapshot, String> {
    let token_len = prompt_tokens
        .checked_add(generated_tokens)
        .and_then(|length| length.checked_sub(1))
        .ok_or_else(|| "MTP final snapshot requires generated tokens".to_string())?;
    let aligned = i32::try_from(accepted).unwrap_or(i32::MAX);
    let hidden_shape = mlxcel_core::array_shape(verify_hidden);
    let last_hidden = mlxcel_core::slice(
        verify_hidden,
        &[0, aligned, 0],
        &[hidden_shape[0], aligned + 1, hidden_shape[2]],
    );
    let logits_shape = mlxcel_core::array_shape(verify_logits);
    let continuation_logits = mlxcel_core::slice(
        verify_logits,
        &[0, aligned, 0],
        &[logits_shape[0], aligned + 1, logits_shape[2]],
    );
    model.materialize_mtp_cache_state();
    let target = model
        .snapshot_sequence_state(mlxcel_core::cache::SequenceId::from_raw(0), token_len)
        .ok_or_else(|| "failed to capture aligned MTP target state".to_string())?;
    drafter
        .capture_prompt_snapshot(target, token_len, &last_hidden, &continuation_logits)
        .ok_or_else(|| "failed to capture aligned MTP snapshot".to_string())
}

fn capture_mtp_final_snapshot(
    model: &Qwen35Model,
    drafter: &Qwen35MtpDraftModel,
    prompt_tokens: &[i32],
    prefill_input: MtpPrefill<'_>,
    prefix_reuse: Option<MtpPrefixReuse<'_>>,
    generated: &[i32],
) -> Result<Option<MtpPromptSnapshot>, String> {
    let Some(&first_token) = generated.first() else {
        return Ok(None);
    };
    let snapshot_start = Instant::now();
    tracing::info!(
        phase = "mtp.final_snapshot.started",
        prompt_tokens = prompt_tokens.len(),
        generated_tokens = generated.len(),
        prefix_cached_tokens = prefix_reuse.map_or(0, |reuse| reuse.cached_tokens),
    );
    // Preserve prefix reuse for final-state capture; rebuilding the whole prompt
    // here delays the terminal response after streamed output has finished.
    let prefill = match prefix_reuse {
        Some(reuse) => prefill_text_with_reuse(model, drafter, prompt_tokens, Some(reuse))?.0,
        None => prefill_for_input(model, drafter, prefill_input)?,
    };
    if generated.len() == 1 {
        let snapshot = capture_mtp_prompt_snapshot(model, drafter, prompt_tokens.len(), &prefill)
            .map(Some)
            .ok_or_else(|| "failed to capture aligned MTP final snapshot".to_string())?;
        tracing::info!(
            phase = "mtp.final_snapshot.complete",
            token_len = snapshot
                .as_ref()
                .expect("MTP final snapshot must be present")
                .token_len(),
            elapsed_seconds = snapshot_start.elapsed().as_secs_f64(),
        );
        return Ok(snapshot);
    }

    let _ = finish_drafter_prefill(model, drafter, prefill_input, prefill, first_token);
    let cached_output = &generated[..generated.len() - 1];
    let verify_input = mlxcel_core::from_slice_i32(
        cached_output,
        &[1, i32::try_from(cached_output.len()).unwrap_or(i32::MAX)],
    );
    let verify = model.forward_mtp_verify(&verify_input);
    mlxcel_core::eval(&verify.hidden);
    mlxcel_core::eval(&verify.logits);
    let draft_tokens = &generated[1..generated.len() - 1];
    drafter.accept_verified_tokens(
        model,
        &verify.hidden,
        draft_tokens,
        draft_tokens.len(),
        generated,
    );
    model.materialize_mtp_cache_state();

    let hidden_shape = mlxcel_core::array_shape(&verify.hidden);
    let aligned = i32::try_from(draft_tokens.len()).unwrap_or(i32::MAX);
    let last_hidden = mlxcel_core::slice(
        &verify.hidden,
        &[0, aligned, 0],
        &[hidden_shape[0], aligned + 1, hidden_shape[2]],
    );
    let logits_shape = mlxcel_core::array_shape(&verify.logits);
    let last = logits_shape[1] - 1;
    let continuation_logits = mlxcel_core::slice(
        &verify.logits,
        &[0, last, 0],
        &[logits_shape[0], last + 1, logits_shape[2]],
    );
    let token_len = prompt_tokens.len() + generated.len() - 1;
    let target = model
        .snapshot_sequence_state(mlxcel_core::cache::SequenceId::from_raw(0), token_len)
        .ok_or_else(|| "failed to capture aligned MTP target state".to_string())?;
    let snapshot = drafter
        .capture_prompt_snapshot(target, token_len, &last_hidden, &continuation_logits)
        .map(Some)
        .ok_or_else(|| "failed to capture aligned MTP final snapshot".to_string())?;
    tracing::info!(
        phase = "mtp.final_snapshot.complete",
        token_len = snapshot
            .as_ref()
            .expect("MTP final snapshot must be present")
            .token_len(),
        elapsed_seconds = snapshot_start.elapsed().as_secs_f64(),
    );
    Ok(snapshot)
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
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        constraint: Option<&mut dyn TokenConstraint>,
        on_token: F,
    ) -> Result<MtpGeneration, String> {
        let prompt = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        let result = self.generate_streaming_for_prefill(
            model,
            prompt_tokens,
            MtpPrefill::Text { prompt: &prompt },
            max_tokens,
            sampling,
            block_size,
            prefix_reuse,
            checkpoint_token_lengths,
            constraint,
            on_token,
        );
        finish_mtp_request(model);
        result
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
        let result = self.generate_streaming_for_prefill(
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
            None,
            &[],
            constraint,
            on_token,
        );
        finish_mtp_request(model);
        result
    }

    fn generate_streaming_for_prefill<F: FnMut(i32) -> bool>(
        &mut self,
        model: &Qwen35Model,
        prompt_tokens: &[i32],
        prefill_input: MtpPrefill<'_>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        block_size: usize,
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        constraint: Option<&mut dyn TokenConstraint>,
        mut on_token: F,
    ) -> Result<MtpGeneration, String> {
        assert!(!prompt_tokens.is_empty(), "MTP prompt must not be empty");
        assert!(block_size >= 2, "MTP block size must be at least 2");
        if checkpoint_token_lengths
            .iter()
            .any(|&length| length == 0 || length > prompt_tokens.len())
            || checkpoint_token_lengths
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "MTP checkpoint token lengths must be sorted, unique, and within the prompt"
                    .to_string(),
            );
        }
        if let Some(constraint) = constraint {
            return self.generate_streaming_constrained_for_prefill(
                model,
                prompt_tokens,
                prefill_input,
                max_tokens,
                sampling,
                block_size,
                prefix_reuse,
                checkpoint_token_lengths,
                constraint,
                on_token,
            );
        }
        let drafter = model
            .mtp()
            .expect("Qwen35MtpGenerator requires a bundled MTP head");
        let final_prefix_reuse = prefix_reuse;
        let continuation_token = final_prefix_reuse.and_then(|reuse| reuse.continuation_token);
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
                prompt_snapshots: Vec::new(),
                final_snapshot: None,
                cached_tokens: 0,
            });
        }

        let (prefill, cached_tokens, prompt_snapshots) = match prefill_input {
            MtpPrefill::Text { .. } => prefill_text_with_checkpoints(
                model,
                drafter,
                prompt_tokens,
                prefix_reuse,
                checkpoint_token_lengths,
            ),
            MtpPrefill::Multimodal { .. } => prefill_for_input(model, drafter, prefill_input)
                .map(|prefill| (prefill, 0, Vec::new())),
        }
        .expect("MTP prefill requires valid synchronized chunks");
        let first_token = if let Some(token) = continuation_token {
            token
        } else {
            let (token, _) =
                sample_token_optimized(&prefill.first_logits, &sampling, prompt_tokens);
            mlxcel_core::eval(&token);
            mlxcel_core::item_i32(&token)
        };
        mlxcel_core::eval(&prefill.hidden);
        log_mtp_memory("mtp.first_token.handoff", prompt_tokens.len());

        let mut generated = Vec::with_capacity(max_tokens);
        let mut history = prompt_tokens.to_vec();
        let mut mtp_stats = MtpGenerationStats::default();
        let mut stop_reason = GenerationStopReason::MaxTokens;
        if continuation_token.is_none() {
            if eos_tokens.contains(&first_token) {
                stop_reason = GenerationStopReason::Eos;
            } else {
                generated.push(first_token);
                history.push(first_token);
                if !on_token(first_token) {
                    stop_reason = GenerationStopReason::CallbackCancelled;
                }
            }
        }
        let decode_start = Instant::now();

        if generated.len() < max_tokens && stop_reason == GenerationStopReason::MaxTokens {
            let mut next_hidden =
                finish_drafter_prefill(model, drafter, prefill_input, prefill, first_token);
            let mut bonus = first_token;

            while generated.len() < max_tokens {
                let emitted_before = generated.len();
                let remaining = max_tokens - generated.len();
                let proposal_count = round_proposal_count(block_size, remaining);
                if proposal_count == 0 {
                    break;
                }
                let greedy = sampler_is_greedy(&sampling);
                let phase_start = Instant::now();
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
                mtp_stats.draft_time += phase_start.elapsed();
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
                let phase_start = Instant::now();
                let verify = model.forward_mtp_verify(&verify_input);
                mlxcel_core::eval(&verify.logits);
                mtp_stats.target_verify_time += phase_start.elapsed();
                mtp_stats.target_forward_calls += 1;
                mtp_stats.speculative_rounds += 1;
                let phase_start = Instant::now();
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
                mtp_stats.walk_time += phase_start.elapsed();
                let phase_start = Instant::now();
                mtp_stats.record_round(walk.accepted, draft_tokens.len());

                let round_stop_reason = emit_walk_tokens(
                    &walk.new_tokens,
                    &eos_tokens,
                    max_tokens,
                    &mut generated,
                    &mut history,
                    &mut on_token,
                );
                if let Some(reason) = round_stop_reason {
                    stop_reason = reason;
                }

                let rollback_accepted =
                    target_cache_accepted_count(walk.accepted, walk.new_tokens.len());
                if rollback_accepted < draft_tokens.len() {
                    model.rollback_mtp_verify(
                        &verify.gdn_states,
                        rollback_accepted,
                        verify_tokens.len(),
                        false,
                    );
                }
                if mtp_round_reaches_cache_clear(
                    emitted_before,
                    generated.len(),
                    MTP_STATE_MATERIALIZE_INTERVAL,
                ) {
                    model.materialize_mtp_cache_state();
                    mtp_stats.full_state_materializations += 1;
                }
                drafter.accept_verified_tokens(
                    model,
                    &verify.hidden,
                    &draft_tokens,
                    walk.accepted,
                    &walk.new_tokens,
                );
                let can_capture_round_snapshot = round_stop_reason.is_some()
                    && generated.len() - emitted_before == walk.new_tokens.len();
                if can_capture_round_snapshot {
                    let final_snapshot = if matches!(prefill_input, MtpPrefill::Text { .. }) {
                        Some(capture_mtp_snapshot_from_verify(
                            model,
                            drafter,
                            prompt_tokens.len(),
                            generated.len(),
                            &verify.hidden,
                            &verify.logits,
                            walk.accepted,
                        )?)
                    } else {
                        None
                    };
                    mtp_stats.reconcile_time += phase_start.elapsed();
                    mtp_stats.decode_time = decode_start.elapsed();
                    return Ok(MtpGeneration {
                        token_ids: generated,
                        stats: mtp_stats,
                        stop_reason: round_stop_reason
                            .expect("round snapshot requires a stop reason"),
                        prompt_snapshots,
                        final_snapshot,
                        cached_tokens,
                    });
                }
                if round_stop_reason.is_none() {
                    let hidden_shape = mlxcel_core::array_shape(&verify.hidden);
                    let accepted = i32::try_from(walk.accepted).unwrap_or(i32::MAX);
                    next_hidden = materialize_detached(mlxcel_core::slice(
                        &verify.hidden,
                        &[0, accepted, 0],
                        &[hidden_shape[0], accepted + 1, hidden_shape[2]],
                    ));
                    bonus = *walk
                        .new_tokens
                        .last()
                        .expect("speculative walk emits at least one token");
                    mtp_stats.full_state_materializations += 1;
                    mtp_stats.cache_snapshot_count += 1;
                }
                if let Some(elapsed) = clear_mtp_cache_if_needed(emitted_before, generated.len()) {
                    mtp_stats.record_cache_clear(elapsed);
                }
                mtp_stats.reconcile_time += phase_start.elapsed();
                if round_stop_reason.is_some() {
                    break;
                }
            }
        }
        mtp_stats.decode_time = decode_start.elapsed();
        let final_snapshot = if matches!(prefill_input, MtpPrefill::Text { .. }) {
            capture_mtp_final_snapshot(
                model,
                drafter,
                prompt_tokens,
                prefill_input,
                final_prefix_reuse,
                &generated,
            )?
        } else {
            None
        };
        Ok(MtpGeneration {
            token_ids: generated,
            stats: mtp_stats,
            stop_reason,
            prompt_snapshots,
            final_snapshot,
            cached_tokens,
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
        prefix_reuse: Option<MtpPrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
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
                prompt_snapshots: Vec::new(),
                final_snapshot: None,
                cached_tokens: 0,
            });
        }

        let mut generated = Vec::with_capacity(max_tokens);
        let mut stats = MtpGenerationStats::default();
        let mut stop_reason = GenerationStopReason::MaxTokens;
        let final_prefix_reuse = prefix_reuse;
        let mut prefix_reuse = prefix_reuse;
        let mut prompt_snapshots = Vec::new();
        let mut cached_tokens = 0;
        let mut state;

        loop {
            let (prefill, reused, checkpoints) = if generated.is_empty() {
                match prefill_input {
                    MtpPrefill::Text { .. } => prefill_text_with_checkpoints(
                        model,
                        drafter,
                        prompt_tokens,
                        prefix_reuse.take(),
                        checkpoint_token_lengths,
                    ),
                    MtpPrefill::Multimodal { .. } => {
                        prefill_for_input(model, drafter, prefill_input)
                            .map(|prefill| (prefill, 0, Vec::new()))
                    }
                }
            } else {
                prefill_for_input(model, drafter, prefill_input)
                    .map(|prefill| (prefill, 0, Vec::new()))
            }?;
            cached_tokens = cached_tokens.max(reused);
            if prompt_snapshots.is_empty() {
                prompt_snapshots = checkpoints;
            }
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
                let final_snapshot = if matches!(prefill_input, MtpPrefill::Text { .. }) {
                    capture_mtp_final_snapshot(
                        model,
                        drafter,
                        prompt_tokens,
                        prefill_input,
                        final_prefix_reuse,
                        &generated,
                    )?
                } else {
                    None
                };
                return Ok(MtpGeneration {
                    token_ids: generated,
                    stats,
                    prompt_snapshots,
                    final_snapshot,
                    cached_tokens,
                    stop_reason: GenerationStopReason::CallbackCancelled,
                });
            }
            if let Some(reason) = initial.stop_reason {
                if initial.rebuild && !generated.is_empty() {
                    let _ = rebuild_mtp_state(model, drafter, prefill_input, &generated)?;
                }
                let final_snapshot = if matches!(prefill_input, MtpPrefill::Text { .. }) {
                    capture_mtp_final_snapshot(
                        model,
                        drafter,
                        prompt_tokens,
                        prefill_input,
                        final_prefix_reuse,
                        &generated,
                    )?
                } else {
                    None
                };
                return Ok(MtpGeneration {
                    token_ids: generated,
                    stats,
                    prompt_snapshots,
                    final_snapshot,
                    cached_tokens,
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
                state = ActiveMtpState {
                    next_hidden: finish_drafter_prefill(
                        model,
                        drafter,
                        prefill_input,
                        prefill,
                        first_token,
                    ),
                    bonus: first_token,
                };
            } else {
                state = rebuild_mtp_state(model, drafter, prefill_input, &generated)?;
            }
            break;
        }

        drafter.materialize_state();
        model.materialize_mtp_cache_state();
        while generated.len() < max_tokens && stop_reason == GenerationStopReason::MaxTokens {
            let emitted_before = generated.len();
            let remaining = max_tokens - generated.len();
            let proposal_count = round_proposal_count(block_size, remaining);
            if proposal_count == 0 {
                break;
            }
            let greedy = sampler_is_greedy(&sampling);
            let (draft_tokens, proposal_probs) =
                rollback_constraint_transaction(constraint, |active| {
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
                })?;

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
            stats.record_round(walk.accepted, draft_tokens.len());
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
                    let _ = prefill_for_input(model, drafter, prefill_input)?;
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
                        true,
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
                state.next_hidden = materialize_detached(mlxcel_core::slice(
                    &verify.hidden,
                    &[0, accepted, 0],
                    &[hidden_shape[0], accepted + 1, hidden_shape[2]],
                ));
                state.bonus = *walk
                    .new_tokens
                    .last()
                    .expect("constrained walk emitted at least one token");
            }
            drafter.materialize_state();
            model.materialize_mtp_cache_state();
            if let Some(elapsed) = clear_mtp_cache_if_needed(emitted_before, generated.len()) {
                stats.record_cache_clear(elapsed);
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
                let prefill = prefill_for_input(model, drafter, prefill_input)?;
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

        let final_snapshot = if matches!(prefill_input, MtpPrefill::Text { .. }) {
            capture_mtp_final_snapshot(
                model,
                drafter,
                prompt_tokens,
                prefill_input,
                final_prefix_reuse,
                &generated,
            )?
        } else {
            None
        };
        Ok(MtpGeneration {
            token_ids: generated,
            stats,
            stop_reason,
            prompt_snapshots,
            final_snapshot,
            cached_tokens,
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

    #[test]
    fn mtp_prompt_snapshot_arrays_remain_owned_after_sources_drop() {
        let keys = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 1, 2]);
        let values = mlxcel_core::from_slice_f32(&[3.0, 4.0], &[1, 1, 1, 2]);
        let hidden = mlxcel_core::from_slice_f32(&[5.0, 6.0], &[1, 1, 2]);
        let logits = mlxcel_core::from_slice_f32(&[7.0, 8.0], &[1, 1, 2]);
        let snapshot = MtpPromptSnapshot {
            target: ModelStateSnapshot::new("test", 2),
            draft_keys: Some(materialize_detached(mlxcel_core::copy(&keys))),
            draft_values: Some(materialize_detached(mlxcel_core::copy(&values))),
            draft_offset: 1,
            last_hidden: materialize_detached(mlxcel_core::copy(&hidden)),
            continuation_logits: materialize_detached(mlxcel_core::copy(&logits)),
        };
        drop((keys, values, hidden, logits));
        mlxcel_core::clear_memory_cache();

        let expected_keys = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 1, 2]);
        let expected_hidden = mlxcel_core::from_slice_f32(&[5.0, 6.0], &[1, 1, 2]);
        let keys_equal = mlxcel_core::allclose(
            snapshot.draft_keys.as_deref().expect("draft keys"),
            &expected_keys,
            0.0,
            0.0,
        );
        let hidden_equal = mlxcel_core::allclose(&snapshot.last_hidden, &expected_hidden, 0.0, 0.0);
        mlxcel_core::eval(&keys_equal);
        mlxcel_core::eval(&hidden_equal);
        assert!(mlxcel_core::item_bool(&keys_equal));
        assert!(mlxcel_core::item_bool(&hidden_equal));
    }

    #[test]
    fn mtp_portable_snapshot_round_trips_and_rejects_incomplete_pairs() {
        let keys = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 1, 2]);
        let values = mlxcel_core::astype(&keys, mlxcel_core::dtype::FLOAT16);
        let hidden = mlxcel_core::from_slice_f32(&[3.0, 4.0], &[1, 1, 2]);
        let logits = mlxcel_core::from_slice_f32(&[5.0, 6.0], &[1, 1, 2]);
        let mut target = ModelStateSnapshot::new("mtp-portable-test", 2);
        target.push_tensor("target.state", &hidden);
        let snapshot = crate::PromptSnapshot::Mtp(MtpPromptSnapshot {
            target,
            draft_keys: Some(materialize_detached(mlxcel_core::copy(&values))),
            draft_values: Some(materialize_detached(mlxcel_core::copy(&values))),
            draft_offset: 1,
            last_hidden: materialize_detached(mlxcel_core::copy(&hidden)),
            continuation_logits: materialize_detached(mlxcel_core::copy(&logits)),
        });
        let expected = snapshot.to_portable().expect("encode MTP snapshot");
        let restored = crate::PromptSnapshot::from_portable(expected.clone())
            .expect("restore MTP snapshot")
            .to_portable()
            .expect("re-encode MTP snapshot");
        assert_eq!(restored, expected);

        let crate::PortablePromptSnapshot::Mtp {
            target,
            draft_keys,
            draft_offset,
            last_hidden,
            continuation_logits,
            ..
        } = expected
        else {
            unreachable!();
        };
        let incomplete = crate::PortablePromptSnapshot::Mtp {
            target,
            draft_keys,
            draft_values: None,
            draft_offset,
            last_hidden,
            continuation_logits,
        };
        assert!(crate::PromptSnapshot::from_portable(incomplete).is_err());
    }

    fn logits_rows(rows: &[&[f32]]) -> UniquePtr<MlxArray> {
        let vocab = rows.first().expect("logits row").len();
        assert!(rows.iter().all(|row| row.len() == vocab));
        let values = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
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
                ..MtpGenerationStats::default()
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
            constrained_greedy_walk(&[1, 2], &full_logits, &sampling, &[9], &[], &[], 8, active)
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
            constrained_initial_step(&logits, &sampling, &[9], &[1], &[], 8, active)
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
            constrained_initial_step(&logits, &sampling, &[9], &[], &[2], 8, active)
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
    fn final_budget_round_rolls_back_unemitted_bonus() {
        assert_eq!(target_cache_accepted_count(2, 3), 2);
        assert_eq!(target_cache_accepted_count(2, 2), 1);
        assert_eq!(target_cache_accepted_count(0, 1), 0);
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
            &logits_rows(&[&[f32::NEG_INFINITY, 0.0], &[0.0, f32::NEG_INFINITY]]),
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
        let reason = emit_walk_tokens(&[7, 99, 8], &[99], 8, &mut generated, &mut history, |_| {
            true
        });
        assert_eq!(reason, Some(GenerationStopReason::Eos));
        assert_eq!(generated, [7]);
        assert_eq!(history, [10, 11, 7]);

        let rejected_proposals = [20, 21, 22];
        let mut generated = Vec::new();
        let mut history = vec![10, 11];
        let reason = emit_walk_tokens(&[42], &[], 8, &mut generated, &mut history, |_| false);
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
            let (token, proposal_probs) = sample_token_with_distribution(&q_logits, &sampling, &[]);
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
            let (token, proposal_probs) = sample_token_with_distribution(&q_logits, &sampling, &[]);
            mlxcel_core::eval(&token);
            let token = mlxcel_core::item_i32(&token);
            let target_probs = effective_token_distribution(&p_logits, &sampling, &[]);
            let emitted = match verify_draft_token(&target_probs, &proposal_probs, token) {
                DraftVerdict::Accept => token,
                DraftVerdict::Reject { .. } => {
                    let (unconditional, _) = sample_token_optimized(&p_logits, &sampling, &[]);
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
            let (token, proposal_probs) = sample_token_with_distribution(&logits, &sampling, &[]);
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

    #[test]
    fn round_cache_clear_reaches_boundaries_crossed_by_accepted_blocks() {
        assert!(!mtp_round_reaches_cache_clear(0, 3, 4));
        assert!(mtp_round_reaches_cache_clear(3, 6, 4));
        assert!(mtp_round_reaches_cache_clear(7, 12, 4));
        assert!(!mtp_round_reaches_cache_clear(4, 7, 4));
        assert!(!mtp_round_reaches_cache_clear(0, usize::MAX, 0));
    }

    const fn tensor_bytes(elements: u64, bits_per_element: u64) -> u64 {
        elements.saturating_mul(bits_per_element).div_ceil(8)
    }

    const fn shaped_tensor_bytes(shape: &[u64], bits_per_element: u64) -> u64 {
        let mut elements = 1u64;
        let mut index = 0;
        while index < shape.len() {
            elements = elements.saturating_mul(shape[index]);
            index += 1;
        }
        tensor_bytes(elements, bits_per_element)
    }

    #[test]
    fn theoretical_tensor_bytes_cover_packed_model_cache_and_transient_geometry() {
        let packed_model = shaped_tensor_bytes(&[4, 64, 32], 4);
        let kv_cache = 2 * shaped_tensor_bytes(&[4, 2, 128, 16], 4);
        let verify_transient =
            shaped_tensor_bytes(&[1, 3, 64], 16) + shaped_tensor_bytes(&[1, 3, 96], 16);

        assert_eq!(packed_model, 4_096);
        assert_eq!(kv_cache, 16_384);
        assert_eq!(verify_transient, 960);
        assert_eq!(packed_model + kv_cache + verify_transient, 21_440);
    }

    #[test]
    fn watermark_bounds_variable_shape_cache_before_256_token_cadence() {
        let persistent = shaped_tensor_bytes(&[8, 64, 64], 4);
        let bounded_transient = shaped_tensor_bytes(&[1, 3, 64], 16);
        let cache_per_round = shaped_tensor_bytes(&[2, 3, 64], 16);
        let cache_watermark = 4 * cache_per_round;
        let total_budget = persistent + bounded_transient + cache_watermark;

        let cache_at_256_cadence = 86 * cache_per_round;
        assert!(
            persistent + bounded_transient + cache_at_256_cadence > total_budget,
            "a cadence-only 256-token policy must exceed this geometry's budget"
        );

        let mut cached = 0;
        let mut maximum_used = persistent + bounded_transient;
        let mut previous = 0;
        for round in 1..=86 {
            cached += cache_per_round;
            let emitted = round * 3;
            maximum_used = maximum_used.max(persistent + bounded_transient + cached);
            if mtp_cache_should_clear(previous, emitted, 256, cached, cache_watermark) {
                cached = 0;
            }
            previous = emitted;
        }
        assert_eq!(
            maximum_used,
            persistent + bounded_transient + cache_watermark
        );
        assert!(maximum_used <= total_budget);
    }

    #[test]
    fn mtp_round_stats_record_proposals_and_accepted_tokens() {
        let mut stats = MtpGenerationStats::default();
        stats.record_round(2, 2);
        stats.record_round(1, 2);
        assert_eq!(stats.proposed_draft_tokens, 4);
        assert_eq!(stats.accepted_draft_tokens, 3);
        assert_eq!(stats.acceptance_percentage(), 75.0);
        stats.record_cache_clear(Duration::from_millis(3));
        assert_eq!(stats.cache_clear_count, 1);
        assert_eq!(stats.cache_clear_time, Duration::from_millis(3));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_variable_shape_cycles_obey_a_computed_cache_watermark() {
        mlxcel_core::clear_memory_cache();
        let baseline = mlxcel_core::memory::snapshot().active_bytes;
        let transient_bytes = shaped_tensor_bytes(&[1, 16, 64], 32);
        let cache_watermark = 4 * transient_bytes;
        let mut clears = 0;

        for round in 1..=32 {
            let width = 16 + (round % 4) * 16;
            let temporary = mlxcel_core::zeros(&[1, width as i32, 64], mlxcel_core::dtype::FLOAT32);
            mlxcel_core::eval(&temporary);
            drop(temporary);
            let memory = mlxcel_core::memory::snapshot();
            if mtp_cache_should_clear(
                (round - 1) * 3,
                round * 3,
                256,
                memory.cache_bytes,
                cache_watermark,
            ) {
                mlxcel_core::clear_memory_cache();
                clears += 1;
            }
        }

        let final_memory = mlxcel_core::memory::snapshot();
        assert!(
            clears > 0,
            "variable Metal shapes must exercise the watermark"
        );
        assert!(
            final_memory.used_bytes() <= baseline + transient_bytes + cache_watermark,
            "final allocator bytes {} exceeded computed bound {}",
            final_memory.used_bytes(),
            baseline + transient_bytes + cache_watermark,
        );
        mlxcel_core::clear_memory_cache();
    }

    #[test]
    fn materialized_hidden_and_trimmed_draft_cache_preserve_visible_state() {
        let hidden_source = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let hidden =
            materialize_detached(mlxcel_core::slice(&hidden_source, &[0, 1, 0], &[1, 2, 2]));
        drop(hidden_source);
        mlxcel_core::clear_memory_cache();
        assert_eq!(array_f32(&hidden), [3.0, 4.0]);

        let mut cache = KVCache::new();
        cache.update(
            mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]),
            mlxcel_core::from_slice_f32(&[11.0, 12.0, 13.0], &[1, 1, 3, 1]),
        );
        cache.update(
            mlxcel_core::from_slice_f32(&[4.0], &[1, 1, 1, 1]),
            mlxcel_core::from_slice_f32(&[14.0], &[1, 1, 1, 1]),
        );
        assert_eq!(cache.trim(1), 1);
        cache.materialize_state();
        let (keys, values) = cache.update_and_fetch(
            mlxcel_core::from_slice_f32(&[5.0], &[1, 1, 1, 1]),
            mlxcel_core::from_slice_f32(&[15.0], &[1, 1, 1, 1]),
        );
        assert_eq!(cache.offset, 4);
        assert_eq!(array_f32(&keys), [1.0, 2.0, 3.0, 5.0]);
        assert_eq!(array_f32(&values), [11.0, 12.0, 13.0, 15.0]);
    }
    fn array_f32(array: &MlxArray) -> Vec<f32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 bytes")))
            .collect()
    }

    #[test]
    fn multimodal_shift_preserves_chunk_alignment_and_bonus_tail() {
        let embeddings = mlxcel_core::from_slice_f32(&[1.0, 20.0, 2.0, 30.0, 3.0, 4.0], &[1, 6, 1]);
        let bonus = mlxcel_core::from_slice_f32(&[99.0], &[1, 1, 1]);
        let first = shifted_embedding_range(&embeddings, 0, 2, None);
        let second = shifted_embedding_range(&embeddings, 2, 4, None);
        let final_chunk = shifted_embedding_range(&embeddings, 4, 6, Some(&bonus));
        assert_eq!(array_f32(&first), [20.0, 2.0]);
        assert_eq!(array_f32(&second), [30.0, 3.0]);
        assert_eq!(array_f32(&final_chunk), [4.0, 99.0]);

        let synchronized = mlxcel_core::concatenate(
            &mlxcel_core::concatenate(&first, &second, 1),
            &final_chunk,
            1,
        );
        assert_eq!(array_f32(&synchronized), [20.0, 2.0, 30.0, 3.0, 4.0, 99.0]);
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
