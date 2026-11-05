// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 oMLX contributors
//
// Adapted from https://github.com/jundot/omlx/tree/0436bdf2792edf74e8f077beea20782bd0f4a298/omlx/specprefill
// and omlx/patches/specprefill.py.

use std::cmp::Ordering;

use super::SpecPrefillConfig;

/// Target-token chunk granularity used for ranking.
const CHUNK_TOKENS: usize = 32;

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

    fn selection_config(keep_rate: f32) -> SpecPrefillConfig {
        SpecPrefillConfig {
            keep_rate,
            ..Default::default()
        }
    }

    #[test]
    fn mandatory_disjoint_ends_are_selected_exactly() {
        let importance = vec![0.0; 1024];
        let selected = select_target_indices(&importance, 7, 1031, selection_config(0.01));
        let expected = (7..263).chain(775..1031).collect::<Vec<_>>();
        assert_eq!(selected, expected);
    }

    #[test]
    fn overlapping_mandatory_ends_select_every_token_once() {
        let selected = select_target_indices(&vec![0.0; 400], 10, 410, selection_config(0.01));
        assert_eq!(selected, (10..410).collect::<Vec<_>>());
    }

    #[test]
    fn partial_ranked_chunk_joins_mandatory_ends() {
        let mut importance = vec![0.0; 525];
        importance[256..288].fill(1.0);
        let selected = select_target_indices(&importance, 5, 530, selection_config(0.01));
        assert_eq!(selected, (5..530).collect::<Vec<_>>());
    }

    #[test]
    fn mandatory_ends_can_exceed_budget_and_are_sorted_unique() {
        let mut importance = vec![0.0; 1024];
        importance[512..544].fill(1.0);
        let selected = select_target_indices(&importance, 7, 1031, selection_config(0.01));
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
        let selected = select_target_indices(&vec![0.0; 65], 35, 100, selection_config(1.0));
        assert_eq!(selected, (35..100).collect::<Vec<_>>());
    }
}
