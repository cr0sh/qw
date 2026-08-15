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

//! Adaptive MTP block-size controller.
//!
//! Ports upstream `mlx_vlm.speculative.mtp._effective_mtp_block_size`.
//! Gemma 4 assistants are configured for a 4-token verify block, but users
//! may request a larger `--draft-block-size`. The reference treats that
//! larger value as a ceiling: stay at the configured depth until the recent
//! acceptance history shows the configured prefix is usually fully accepted,
//! then expand to the requested ceiling.

/// Minimum number of completed MTP rounds before expanding above the
/// drafter's configured block size.
const MIN_HISTORY_FOR_EXPANSION: usize = 8;
/// Number of most-recent MTP rounds considered by the expansion gate.
const RECENT_HISTORY: usize = 32;
/// Required hit rate for accepting the entire configured draft prefix.
const CONFIGURED_PREFIX_HIT_RATE: f64 = 0.65;

/// Choose the next MTP verify block size.
///
/// `requested_block_total` is the user-facing ceiling. `configured_block_total`
/// comes from the drafter checkpoint. `accept_lens` stores accepted draft-token
/// counts per round (batched callers record the per-round row average, matching
/// upstream), and `remaining_budget` includes the prefix bonus position.
///
/// Used by: Gemma 4 MTP B=1 and B>1 round loops.
pub(crate) fn effective_mtp_block_size(
    requested_block_total: usize,
    configured_block_total: usize,
    accept_lens: &[f64],
    remaining_budget: usize,
) -> usize {
    let block_total = requested_block_total.min(remaining_budget);
    let configured_block_total = configured_block_total.min(block_total);
    if block_total <= configured_block_total || configured_block_total <= 1 {
        return block_total;
    }

    if accept_lens.len() < MIN_HISTORY_FOR_EXPANSION {
        return configured_block_total;
    }

    let recent_start = accept_lens.len().saturating_sub(RECENT_HISTORY);
    let recent = &accept_lens[recent_start..];
    let configured_draft_count = (configured_block_total - 1) as f64;
    let configured_prefix_hits = recent
        .iter()
        .filter(|&&accepted| accepted >= configured_draft_count)
        .count();
    let configured_prefix_hit_rate = configured_prefix_hits as f64 / recent.len() as f64;
    if configured_prefix_hit_rate < CONFIGURED_PREFIX_HIT_RATE {
        configured_block_total
    } else {
        block_total
    }
}

#[cfg(test)]
mod tests {
    use super::effective_mtp_block_size;

    #[test]
    fn stays_at_requested_when_not_above_configured() {
        assert_eq!(effective_mtp_block_size(4, 4, &[], 16), 4);
        assert_eq!(effective_mtp_block_size(3, 4, &[], 16), 3);
    }

    #[test]
    fn caps_by_remaining_budget() {
        assert_eq!(effective_mtp_block_size(8, 4, &[3.0; 16], 3), 3);
    }

    #[test]
    fn warms_up_at_configured_depth_before_history_threshold() {
        assert_eq!(effective_mtp_block_size(8, 4, &[3.0; 7], 16), 4);
    }

    #[test]
    fn stays_configured_when_recent_prefix_hit_rate_is_low() {
        let mut history = vec![3.0; 16];
        history.extend([0.0; 16]);
        assert_eq!(effective_mtp_block_size(8, 4, &history, 16), 4);
    }

    #[test]
    fn expands_to_requested_when_recent_prefix_hit_rate_is_high() {
        let mut history = vec![0.0; 8];
        history.extend([3.0; 24]);
        assert_eq!(effective_mtp_block_size(8, 4, &history, 16), 8);
    }
}
