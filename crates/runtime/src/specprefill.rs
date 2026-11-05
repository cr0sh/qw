// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 oMLX contributors
//
// Adapted from https://github.com/jundot/omlx/tree/0436bdf2792edf74e8f077beea20782bd0f4a298/omlx/specprefill
// and omlx/patches/specprefill.py.

use std::time::Duration;

use anyhow::{Result, ensure};

#[cfg(any(feature = "specprefill", test))]
mod scoring;
#[cfg(any(feature = "specprefill", test))]
mod selection;
#[cfg(any(feature = "specprefill", test))]
pub(crate) use scoring::score_tokens;
#[cfg(any(feature = "specprefill", test))]
pub(crate) use selection::select_target_indices;

pub const DEFAULT_KEEP_RATE: f32 = 0.25;
pub const DEFAULT_KEEP_FIRST_TOKENS: usize = 256;
pub const DEFAULT_KEEP_LAST_TOKENS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecPrefillConfig {
    /// Activate only when the eligible suffix exceeds this many tokens.
    pub min_tokens: usize,
    /// Fraction of score-ranked eligible chunks to retain.
    pub keep_rate: f32,
    /// Dense history prefix; scoring begins after this or the admitted cache, whichever is larger.
    pub protected_prefix_tokens: usize,
    /// Tokens force-kept at the start of the eligible suffix.
    pub keep_first_tokens: usize,
    /// Tokens force-kept at the end of the eligible suffix.
    pub keep_last_tokens: usize,
}

impl Default for SpecPrefillConfig {
    fn default() -> Self {
        Self {
            min_tokens: 8192,
            keep_rate: DEFAULT_KEEP_RATE,
            protected_prefix_tokens: 0,
            keep_first_tokens: DEFAULT_KEEP_FIRST_TOKENS,
            keep_last_tokens: DEFAULT_KEEP_LAST_TOKENS,
        }
    }
}

impl SpecPrefillConfig {
    pub fn validate(self, prompt_len: usize) -> Result<()> {
        ensure!(
            self.min_tokens > 0,
            "SpecPrefill min_tokens must be greater than zero"
        );
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

    /// End of the dense prefix protected by cache reuse or request policy.
    pub fn dense_prefix_end(self, cached_tokens: usize) -> usize {
        cached_tokens.max(self.protected_prefix_tokens)
    }

    /// Activation is strictly above the eligible-suffix threshold.
    pub fn should_activate(self, eligible_tokens: usize) -> bool {
        eligible_tokens > self.min_tokens
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

#[cfg(any(feature = "specprefill", test))]
pub(crate) fn admitted_cached_tokens(
    prefix_reuse: Option<&mlxcel_core::generate::PrefixReuse<'_>>,
    prompt_len: usize,
) -> usize {
    prefix_reuse
        .filter(|reuse| {
            reuse.cached_tokens > 0
                && reuse.cached_tokens <= prompt_len
                && reuse.snapshot.token_len() == reuse.cached_tokens
                && (reuse.cached_tokens < prompt_len
                    || reuse.snapshot.continuation_logits().is_some())
        })
        .map_or(0, |reuse| reuse.cached_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_admission_preserves_structural_checks() {
        use mlxcel_core::generate::{ModelStateSnapshot, PrefixReuse};

        assert_eq!(admitted_cached_tokens(None, 10), 0);
        for (snapshot_len, cached_tokens, expected) in
            [(0, 0, 0), (4, 4, 4), (5, 4, 0), (11, 11, 0), (10, 10, 0)]
        {
            let snapshot = ModelStateSnapshot::new("test", snapshot_len);
            let reuse = PrefixReuse {
                snapshot: &snapshot,
                cached_tokens,
            };
            assert_eq!(admitted_cached_tokens(Some(&reuse), 10), expected);
        }
    }

    #[test]
    fn full_prefix_admission_requires_continuation_logits() {
        use mlxcel_core::generate::{ModelStateSnapshot, PrefixReuse};

        let mut snapshot = ModelStateSnapshot::new("test", 10);
        let logits = mlxcel_core::from_slice_f32(&[0.0, 1.0], &[1, 1, 2]);
        snapshot.set_continuation_logits(&logits);
        let reuse = PrefixReuse {
            snapshot: &snapshot,
            cached_tokens: 10,
        };
        assert_eq!(admitted_cached_tokens(Some(&reuse), 10), 10);
    }

    #[test]
    fn specprefill_config_validation() {
        assert_eq!(SpecPrefillConfig::default().keep_rate, 0.25);
        assert_eq!(SpecPrefillConfig::default().keep_first_tokens, 256);
        assert_eq!(SpecPrefillConfig::default().keep_last_tokens, 256);
        assert!(
            SpecPrefillConfig {
                min_tokens: 0,
                ..Default::default()
            }
            .validate(10)
            .is_err()
        );
        assert!(
            SpecPrefillConfig {
                keep_rate: 0.0,
                ..Default::default()
            }
            .validate(10)
            .is_err()
        );
        assert!(
            SpecPrefillConfig {
                keep_rate: 1.01,
                ..Default::default()
            }
            .validate(10)
            .is_err()
        );
        assert!(
            SpecPrefillConfig {
                protected_prefix_tokens: 11,
                ..Default::default()
            }
            .validate(10)
            .is_err()
        );
        assert!(
            SpecPrefillConfig {
                min_tokens: 1,
                keep_rate: 1.0,
                protected_prefix_tokens: 10,
                keep_first_tokens: 0,
                keep_last_tokens: 0,
            }
            .validate(10)
            .is_ok()
        );
    }

    #[test]
    fn threshold_equality_stays_dense() {
        let config = SpecPrefillConfig {
            min_tokens: 8,
            ..Default::default()
        };
        assert!(!config.should_activate(8));
        assert!(config.should_activate(9));
    }

    #[test]
    fn cached_and_protected_prefix_take_the_larger_boundary() {
        let config = SpecPrefillConfig {
            protected_prefix_tokens: 40,
            ..Default::default()
        };
        assert_eq!(config.dense_prefix_end(20), 40);
        assert_eq!(config.dense_prefix_end(60), 60);
    }
}
