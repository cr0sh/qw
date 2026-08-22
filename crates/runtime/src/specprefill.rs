// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 oMLX contributors
//
// Adapted from https://github.com/jundot/omlx/tree/0436bdf2792edf74e8f077beea20782bd0f4a298/omlx/specprefill
// and omlx/patches/specprefill.py.

use std::cmp::Ordering;
use std::time::Duration;

use anyhow::{Result, ensure};
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::generation_policy::seed_rng_if_needed;
use mlxcel_core::sampling::sample_token_optimized;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::qwen3_5::Qwen35Model;

pub const SPECPREFILL_DRAFT_MODEL_IDENTIFIER: &str =
    "mlx-community/Qwen3.5-0.8B-MLX-8bit";

const LOOKAHEAD_TOKENS: usize = 8;
const POOL_WIDTH: i32 = 13;
const CHUNK_TOKENS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecPrefillConfig {
    pub min_tokens: usize,
    pub keep_rate: f32,
    pub protected_prefix_tokens: usize,
    pub keep_first_tokens: usize,
    pub keep_last_tokens: usize,
}

impl Default for SpecPrefillConfig {
    fn default() -> Self {
        Self {
            min_tokens: 8192,
            keep_rate: 0.25,
            protected_prefix_tokens: 0,
            keep_first_tokens: 256,
            keep_last_tokens: 256,
        }
    }
}

impl SpecPrefillConfig {
    pub(crate) fn validate(self, prompt_len: usize) -> Result<()> {
        ensure!(self.min_tokens > 0, "SpecPrefill min_tokens must be greater than zero");
        ensure!(
            self.keep_rate.is_finite() && self.keep_rate > 0.0 && self.keep_rate <= 1.0,
            "SpecPrefill keep_rate must be in (0, 1]"
        );
        ensure!(
            self.protected_prefix_tokens <= prompt_len,
            "SpecPrefill protected_prefix_tokens exceeds prompt length"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefillMode {
    Dense,
    SpecPrefill(SpecPrefillConfig),
}

#[derive(Debug, Clone)]
pub struct SpecPrefillStats {
    pub draft_tokens: usize,
    pub eligible_target_tokens: usize,
    pub selected_target_tokens: usize,
    pub protected_target_tokens: usize,
    pub cached_target_tokens: usize,
    pub draft_scoring_time: Duration,
    pub target_prefill_time: Duration,
}

pub(crate) fn dense_prefix_end(cached_tokens: usize, config: SpecPrefillConfig) -> usize {
    cached_tokens.max(config.protected_prefix_tokens)
}

pub(crate) fn should_activate(eligible_tokens: usize, config: SpecPrefillConfig) -> bool {
    eligible_tokens > config.min_tokens
}

pub(crate) fn score_tokens(draft: &Qwen35Model, prompt_ids: &[i32]) -> Result<Vec<f32>> {
    let mut logits = draft.specprefill_draft_prefill(prompt_ids)?;
    let sampling = SamplingConfig {
        temperature: 0.6,
        top_p: 0.95,
        seed: Some(0),
        ..SamplingConfig::default()
    };
    seed_rng_if_needed(&sampling);
    let mut captures: Vec<Vec<UniquePtr<MlxArray>>> = Vec::new();
    let mut history = prompt_ids.to_vec();
    let mut token = sample_token_optimized(&logits, &sampling, &history).0;
    mlxcel_core::eval(&token);

    for _ in 0..LOOKAHEAD_TOKENS {
        let token_id = mlxcel_core::item_i32(&token);
        history.push(token_id);
        let (next_logits, queries) = draft.specprefill_draft_lookahead(token_id);
        if captures.is_empty() {
            ensure!(
                !queries.is_empty(),
                "SpecPrefill draft captured no full-attention queries"
            );
            captures = queries
                .into_iter()
                .map(|query| {
                    let mut layer = Vec::with_capacity(LOOKAHEAD_TOKENS);
                    layer.push(query);
                    layer
                })
                .collect();
        } else {
            ensure!(
                queries.len() == captures.len(),
                "SpecPrefill draft attention capture topology changed"
            );
            for (layer, query) in captures.iter_mut().zip(queries) {
                layer.push(query);
            }
        }
        logits = next_logits;
        token = sample_token_optimized(&logits, &sampling, &history).0;
        mlxcel_core::eval(&token);
    }

    let keys = draft.specprefill_draft_prompt_keys(prompt_ids.len());
    compute_importance(&captures, &keys, prompt_ids.len())
}

fn compute_importance(
    captures: &[Vec<UniquePtr<MlxArray>>],
    prompt_keys: &[UniquePtr<MlxArray>],
    prompt_len: usize,
) -> Result<Vec<f32>> {
    let mut all_scores: Option<UniquePtr<MlxArray>> = None;
    for (queries, keys) in captures.iter().zip(prompt_keys) {
        ensure!(!queries.is_empty(), "SpecPrefill captured no draft queries");
        let query_refs = queries
            .iter()
            .map(|query| query.as_ref().expect("captured query is non-null"))
            .collect::<Vec<_>>();
        let mut stacked = mlxcel_core::copy(query_refs[0]);
        for query in &query_refs[1..] {
            stacked = mlxcel_core::concatenate(&stacked, query, 2);
        }
        let key_shape = mlxcel_core::array_shape(keys);
        let query_shape = mlxcel_core::array_shape(&stacked);
        ensure!(
            key_shape[2] >= prompt_len as i32,
            "SpecPrefill draft cache does not span the scored prompt"
        );
        ensure!(
            query_shape[1] % key_shape[1] == 0,
            "SpecPrefill draft query/KV head ratio is not integral"
        );
        let query_groups = query_shape[1] / key_shape[1];
        let grouped_queries = mlxcel_core::reshape(
            &stacked,
            &[
                query_shape[0],
                key_shape[1],
                query_groups,
                query_shape[2],
                query_shape[3],
            ],
        );
        let transposed_keys = mlxcel_core::transpose_axes(keys, &[0, 1, 3, 2]);
        let grouped_keys = mlxcel_core::expand_dims(&transposed_keys, 2);
        let scores = mlxcel_core::matmul(&grouped_queries, &grouped_keys);
        let scale = mlxcel_core::full_f32(&[1], 1.0 / (key_shape[3] as f32).sqrt(), mlxcel_core::dtype::FLOAT32);
        let scores = mlxcel_core::multiply(&mlxcel_core::astype(&scores, mlxcel_core::dtype::FLOAT32), &scale);
        let weights = mlxcel_core::softmax_precise(&scores, -1);
        let weights = mlxcel_core::reshape(&weights, &[-1, LOOKAHEAD_TOKENS as i32, prompt_len as i32]);
        all_scores = Some(match all_scores {
            Some(existing) => mlxcel_core::concatenate(&existing, &weights, 0),
            None => weights,
        });
    }
    let combined = all_scores.expect("pinned draft contains full-attention layers");
    let pooled = average_pool_1d(&combined, POOL_WIDTH);
    let max_over_layers_and_heads = mlxcel_core::max_axis(&pooled, 0, false);
    let importance = mlxcel_core::mean_axis(&max_over_layers_and_heads, 0, false);
    let importance = mlxcel_core::astype(&importance, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&importance);
    let bytes = mlxcel_core::array_to_raw_bytes(&importance);
    ensure!(
        bytes.len() == prompt_len * std::mem::size_of::<f32>(),
        "SpecPrefill importance vector has an unexpected byte length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32")))
        .collect())
}

fn average_pool_1d(values: &MlxArray, width: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(values);
    let pad = width / 2;
    let padded = mlxcel_core::pad(values, &[0, 0, 0, 0, pad, pad], 0.0);
    let prefix = mlxcel_core::cumsum(&padded, -1, false, true);
    let zero = mlxcel_core::zeros(&[shape[0], shape[1], 1], mlxcel_core::array_dtype(values));
    let prefix = mlxcel_core::concatenate(&zero, &prefix, -1);
    let prefix_shape = mlxcel_core::array_shape(&prefix);
    let right = mlxcel_core::slice(
        &prefix,
        &[0, 0, width],
        &[prefix_shape[0], prefix_shape[1], prefix_shape[2]],
    );
    let left = mlxcel_core::slice(
        &prefix,
        &[0, 0, 0],
        &[prefix_shape[0], prefix_shape[1], prefix_shape[2] - width],
    );
    let divisor = mlxcel_core::full_f32(&[1], width as f32, mlxcel_core::array_dtype(values));
    mlxcel_core::divide(&mlxcel_core::subtract(&right, &left), &divisor)
}

pub(crate) fn select_target_indices(
    importance: &[f32],
    eligible_start: usize,
    prompt_len: usize,
    config: SpecPrefillConfig,
) -> Vec<usize> {
    debug_assert_eq!(importance.len(), prompt_len - eligible_start);
    if config.keep_rate >= 1.0 {
        return (eligible_start..prompt_len).collect();
    }
    let chunk_count = importance.len().div_ceil(CHUNK_TOKENS);
    let keep_chunks = ((chunk_count as f32) * config.keep_rate).ceil() as usize;
    let mut ranked = (0..chunk_count)
        .map(|chunk| {
            let start = chunk * CHUNK_TOKENS;
            let end = (start + CHUNK_TOKENS).min(importance.len());
            let mean = importance[start..end].iter().sum::<f32>() / (end - start) as f32;
            (chunk, mean)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_chunk, left), (right_chunk, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_chunk.cmp(right_chunk))
    });
    let mut selected = vec![false; importance.len()];
    for &(chunk, _) in ranked.iter().take(keep_chunks) {
        let start = chunk * CHUNK_TOKENS;
        let end = (start + CHUNK_TOKENS).min(importance.len());
        selected[start..end].fill(true);
    }
    let prefix_end = importance.len().min(config.keep_first_tokens);
    selected[..prefix_end].fill(true);
    let suffix_start = importance.len().saturating_sub(config.keep_last_tokens);
    selected[suffix_start..].fill(true);
    selected
        .into_iter()
        .enumerate()
        .filter_map(|(index, keep)| keep.then_some(eligible_start + index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specprefill_config_validation() {
        assert_eq!(SpecPrefillConfig::default().keep_rate, 0.25);
        assert_eq!(SpecPrefillConfig::default().keep_first_tokens, 256);
        assert_eq!(SpecPrefillConfig::default().keep_last_tokens, 256);
        assert!(SpecPrefillConfig { min_tokens: 0, ..Default::default() }.validate(10).is_err());
        assert!(SpecPrefillConfig { keep_rate: 0.0, ..Default::default() }.validate(10).is_err());
        assert!(SpecPrefillConfig { keep_rate: 1.01, ..Default::default() }.validate(10).is_err());
        assert!(SpecPrefillConfig { protected_prefix_tokens: 11, ..Default::default() }.validate(10).is_err());
        assert!(SpecPrefillConfig {
            min_tokens: 1,
            keep_rate: 1.0,
            protected_prefix_tokens: 10,
            keep_first_tokens: 0,
            keep_last_tokens: 0,
        }
        .validate(10)
        .is_ok());
    }

    #[test]
    fn threshold_equality_stays_dense() {
        let config = SpecPrefillConfig { min_tokens: 8, ..Default::default() };
        assert!(!should_activate(8, config));
        assert!(should_activate(9, config));
    }

    #[test]
    fn cached_and_protected_prefix_take_the_larger_boundary() {
        let config = SpecPrefillConfig { protected_prefix_tokens: 40, ..Default::default() };
        assert_eq!(dense_prefix_end(20, config), 40);
        assert_eq!(dense_prefix_end(60, config), 60);
    }

    fn selection_config(keep_rate: f32) -> SpecPrefillConfig {
        SpecPrefillConfig {
            keep_rate,
            ..Default::default()
        }
    }

    #[test]
    fn mandatory_disjoint_ends_are_selected_exactly() {
        let importance = vec![0.0; 1024];
        let selected =
            select_target_indices(&importance, 7, 1031, selection_config(0.01));
        let expected = (7..263).chain(775..1031).collect::<Vec<_>>();
        assert_eq!(selected, expected);
    }

    #[test]
    fn overlapping_mandatory_ends_select_every_token_once() {
        let selected =
            select_target_indices(&vec![0.0; 400], 10, 410, selection_config(0.01));
        assert_eq!(selected, (10..410).collect::<Vec<_>>());
    }

    #[test]
    fn partial_ranked_chunk_joins_mandatory_ends() {
        let mut importance = vec![0.0; 525];
        importance[256..288].fill(1.0);
        let selected =
            select_target_indices(&importance, 5, 530, selection_config(0.01));
        assert_eq!(selected, (5..530).collect::<Vec<_>>());
    }

    #[test]
    fn mandatory_ends_can_exceed_budget_and_are_sorted_unique() {
        let mut importance = vec![0.0; 1024];
        importance[512..544].fill(1.0);
        let selected =
            select_target_indices(&importance, 7, 1031, selection_config(0.01));
        assert_eq!(selected.len(), 2 * 256 + CHUNK_TOKENS);
        assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(selected.first(), Some(&7));
        assert_eq!(selected.last(), Some(&1030));
        assert!(selected.contains(&(7 + 512)));
    }

    #[test]
    fn zero_edge_counts_disable_mandatory_selection() {
        let mut importance = vec![0.0; 1024];
        importance[512..544].fill(1.0);
        let config = SpecPrefillConfig {
            keep_first_tokens: 0,
            keep_last_tokens: 0,
            ..selection_config(0.01)
        };
        assert_eq!(
            select_target_indices(&importance, 7, 1031, config),
            (519..551).collect::<Vec<_>>()
        );
    }

    #[test]
    fn complete_keep_rate_selects_every_eligible_token() {
        let selected =
            select_target_indices(&vec![0.0; 65], 35, 100, selection_config(1.0));
        assert_eq!(selected, (35..100).collect::<Vec<_>>());
    }
}
