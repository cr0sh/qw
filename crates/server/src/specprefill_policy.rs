use anyhow::{Result, ensure};
use clap_derive::Args;
use qw_runtime::{
    ChatMessage, DEFAULT_KEEP_FIRST_TOKENS, DEFAULT_KEEP_LAST_TOKENS, DEFAULT_KEEP_RATE,
    PrefillMode, SpecPrefillConfig,
};
use tracing::{debug, warn};

pub(crate) const ENABLED: bool = true;

#[derive(Debug, Args)]
pub(crate) struct SpecPrefillArgs {
    /// Current-turn token threshold above which SpecPrefill activates.
    #[arg(long, default_value_t = 8_000)]
    specprefill_min_turn_tokens: usize,
    /// Fraction of score-ranked prompt chunks retained by SpecPrefill.
    #[arg(long, default_value_t = DEFAULT_KEEP_RATE)]
    specprefill_keep_rate: f32,
    /// Tokens force-kept at the start of the SpecPrefill-eligible suffix.
    #[arg(long, default_value_t = DEFAULT_KEEP_FIRST_TOKENS)]
    specprefill_keep_first_tokens: usize,
    /// Tokens force-kept at the end of the SpecPrefill-eligible suffix.
    #[arg(long, default_value_t = DEFAULT_KEEP_LAST_TOKENS)]
    specprefill_keep_last_tokens: usize,
}

impl SpecPrefillArgs {
    pub(crate) fn policy(&self) -> SpecPrefillPolicyConfig {
        SpecPrefillPolicyConfig {
            min_turn_tokens: self.specprefill_min_turn_tokens,
            keep_rate: self.specprefill_keep_rate,
            keep_first_tokens: self.specprefill_keep_first_tokens,
            keep_last_tokens: self.specprefill_keep_last_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecPrefillPolicyConfig {
    pub min_turn_tokens: usize,
    pub keep_rate: f32,
    pub keep_first_tokens: usize,
    pub keep_last_tokens: usize,
}

impl Default for SpecPrefillPolicyConfig {
    fn default() -> Self {
        Self {
            // Server thresholds count the current turn, not the runtime's whole eligible suffix (8192).
            min_turn_tokens: 8_000,
            keep_rate: DEFAULT_KEEP_RATE,
            keep_first_tokens: DEFAULT_KEEP_FIRST_TOKENS,
            keep_last_tokens: DEFAULT_KEEP_LAST_TOKENS,
        }
    }
}

impl SpecPrefillPolicyConfig {
    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            self.min_turn_tokens > 0,
            "--specprefill-min-turn-tokens must be greater than zero"
        );
        ensure!(
            self.keep_rate.is_finite() && self.keep_rate > 0.0 && self.keep_rate <= 1.0,
            "--specprefill-keep-rate must be in (0, 1]"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CurrentTurnPolicy {
    current_turn_tokens: Option<usize>,
    pub(crate) prefill_mode: PrefillMode,
}

fn messages_before_model_input_turn(messages: &[ChatMessage]) -> Option<&[ChatMessage]> {
    let boundary = if messages
        .last()
        .is_some_and(|message| message.role == "tool")
    {
        messages
            .iter()
            .rposition(|message| message.role != "tool")
            .map_or(0, |index| index + 1)
    } else {
        messages
            .iter()
            .rposition(|message| message.role == "user")?
    };
    let preceding = &messages[..boundary];
    preceding
        .iter()
        .any(|message| message.role == "assistant")
        .then_some(preceding)
}

fn current_turn_policy(
    prompt_ids: &[i32],
    preceding_ids: Option<&[i32]>,
    specprefill_allowed: bool,
    config: SpecPrefillPolicyConfig,
) -> CurrentTurnPolicy {
    let Some(preceding_ids) = preceding_ids.filter(|ids| prompt_ids.starts_with(ids)) else {
        return CurrentTurnPolicy {
            current_turn_tokens: None,
            prefill_mode: PrefillMode::Dense,
        };
    };
    let current_turn_tokens = prompt_ids.len() - preceding_ids.len();
    let prefill_mode = if specprefill_allowed && current_turn_tokens > config.min_turn_tokens {
        PrefillMode::SpecPrefill(SpecPrefillConfig {
            min_tokens: config.min_turn_tokens,
            keep_rate: config.keep_rate,
            protected_prefix_tokens: preceding_ids.len(),
            keep_first_tokens: config.keep_first_tokens,
            keep_last_tokens: config.keep_last_tokens,
        })
    } else {
        PrefillMode::Dense
    };
    CurrentTurnPolicy {
        current_turn_tokens: Some(current_turn_tokens),
        prefill_mode,
    }
}

impl CurrentTurnPolicy {
    pub(crate) fn active(&self) -> bool {
        matches!(self.prefill_mode, PrefillMode::SpecPrefill(_))
    }
}

pub(crate) fn resolve_for_request(
    provider: &qw_runtime::Qwen35Provider,
    request: super::SpecPrefillRequest<'_>,
    config: SpecPrefillPolicyConfig,
) -> CurrentTurnPolicy {
    let preceding_ids = if request.has_images {
        None
    } else if let Some(messages) = messages_before_model_input_turn(request.messages) {
        match provider.tokenize_history(
            messages,
            request.tools,
            request.reasoning_effort,
            request.enable_thinking,
        ) {
            Ok(ids) => Some(ids),
            Err(error) => {
                warn!(phase = "specprefill.policy", error = %error);
                None
            }
        }
    } else {
        None
    };
    let policy = current_turn_policy(
        request.prompt_ids,
        preceding_ids.as_deref(),
        request.allowed,
        config,
    );
    debug!(
        phase = "specprefill.policy",
        current_turn_tokens = policy.current_turn_tokens.unwrap_or(0),
        current_turn_boundary_valid = policy.current_turn_tokens.is_some(),
        prefill_mode = ?policy.prefill_mode,
        specprefill_active = policy.active(),
    );
    policy
}

#[cfg(test)]
mod tests {
    use super::*;
    fn message(role: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            name: None,
            content: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    #[test]
    fn initial_system_developer_and_user_input_has_no_later_turn_boundary() {
        for messages in [
            vec![message("user")],
            vec![message("system"), message("user")],
            vec![message("system"), message("developer"), message("user")],
        ] {
            assert!(messages_before_model_input_turn(&messages).is_none());
        }
        assert_eq!(
            current_turn_policy(&[1, 2, 3], None, true, Default::default()).prefill_mode,
            PrefillMode::Dense
        );
    }

    #[test]
    fn second_user_input_is_a_later_model_input_turn() {
        let messages = [
            message("system"),
            message("user"),
            message("assistant"),
            message("user"),
        ];
        let preceding =
            messages_before_model_input_turn(&messages).expect("assistant precedes second user");
        assert_eq!(preceding.len(), 3);
    }

    #[test]
    fn trailing_tool_results_form_one_later_model_input_turn() {
        let single = [
            message("system"),
            message("user"),
            message("assistant"),
            message("tool"),
        ];
        assert_eq!(
            messages_before_model_input_turn(&single)
                .expect("tool result follows model output")
                .len(),
            3
        );

        let parallel = [
            message("system"),
            message("user"),
            message("assistant"),
            message("tool"),
            message("tool"),
        ];
        assert_eq!(
            messages_before_model_input_turn(&parallel)
                .expect("parallel tool results follow model output")
                .len(),
            3
        );
    }

    #[test]
    fn assistant_absent_keeps_user_and_tool_input_dense() {
        assert!(
            messages_before_model_input_turn(&[
                message("system"),
                message("developer"),
                message("user"),
            ])
            .is_none()
        );
        assert!(
            messages_before_model_input_turn(&[
                message("system"),
                message("user"),
                message("tool"),
            ])
            .is_none()
        );
    }

    #[test]
    fn tool_turn_threshold_is_strict_and_protects_exact_history() {
        let config = SpecPrefillPolicyConfig {
            min_turn_tokens: 512,
            ..Default::default()
        };
        let preceding_ids = vec![11; 37];
        for current_turn_tokens in [512, 513, 580, 1_948] {
            let mut prompt_ids = preceding_ids.clone();
            prompt_ids.resize(preceding_ids.len() + current_turn_tokens, 20);
            let policy = current_turn_policy(&prompt_ids, Some(&preceding_ids), true, config);
            assert_eq!(policy.current_turn_tokens, Some(current_turn_tokens));
            if current_turn_tokens == 512 {
                assert_eq!(policy.prefill_mode, PrefillMode::Dense);
            } else {
                assert_eq!(
                    policy.prefill_mode,
                    PrefillMode::SpecPrefill(SpecPrefillConfig {
                        min_tokens: 512,
                        keep_rate: 0.25,
                        protected_prefix_tokens: preceding_ids.len(),
                        keep_first_tokens: 256,
                        keep_last_tokens: 256,
                    })
                );
            }
        }
    }

    #[test]
    fn invalid_boundary_or_ineligible_route_stays_dense() {
        let config = SpecPrefillPolicyConfig {
            min_turn_tokens: 1,
            ..Default::default()
        };
        for preceding in [vec![9, 9], vec![1, 2, 3, 4]] {
            let policy = current_turn_policy(&[1, 2, 3], Some(&preceding), true, config);
            assert_eq!(policy.current_turn_tokens, None);
            assert_eq!(policy.prefill_mode, PrefillMode::Dense);
        }
        let disallowed = current_turn_policy(&[1, 2, 3], Some(&[1]), false, config);
        assert_eq!(disallowed.current_turn_tokens, Some(2));
        assert_eq!(disallowed.prefill_mode, PrefillMode::Dense);
    }

    #[test]
    fn configured_policy_values_flow_to_runtime_config() {
        let config = SpecPrefillPolicyConfig {
            min_turn_tokens: 2,
            keep_rate: 0.4,
            keep_first_tokens: 0,
            keep_last_tokens: 64,
        };
        let policy = current_turn_policy(&[1, 2, 3, 4], Some(&[1]), true, config);
        assert_eq!(
            policy.prefill_mode,
            PrefillMode::SpecPrefill(SpecPrefillConfig {
                min_tokens: 2,
                keep_rate: 0.4,
                protected_prefix_tokens: 1,
                keep_first_tokens: 0,
                keep_last_tokens: 64,
            })
        );
    }

    #[test]
    fn specprefill_policy_validation_accepts_zero_edges_and_rejects_invalid_admission() {
        assert!(
            SpecPrefillPolicyConfig {
                keep_first_tokens: 0,
                keep_last_tokens: 0,
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        assert!(
            SpecPrefillPolicyConfig {
                min_turn_tokens: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        for keep_rate in [0.0, 1.01, f32::NAN] {
            assert!(
                SpecPrefillPolicyConfig {
                    keep_rate,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }
}
