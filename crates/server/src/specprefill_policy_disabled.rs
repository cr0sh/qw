use anyhow::Result;
use clap_derive::Args;
use qw_runtime::PrefillMode;

pub(crate) const ENABLED: bool = false;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpecPrefillPolicyConfig {}

impl SpecPrefillPolicyConfig {
    pub(crate) fn validate(self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Args)]
pub(crate) struct SpecPrefillArgs {}

impl SpecPrefillArgs {
    pub(crate) fn policy(&self) -> SpecPrefillPolicyConfig {
        SpecPrefillPolicyConfig::default()
    }
}

pub(crate) struct CurrentTurnPolicy {
    pub(crate) prefill_mode: PrefillMode,
}

impl CurrentTurnPolicy {
    pub(crate) fn active(&self) -> bool {
        false
    }
}

pub(crate) fn resolve_for_request(
    _provider: &qw_runtime::Qwen35Provider,
    _request: super::SpecPrefillRequest<'_>,
    _config: SpecPrefillPolicyConfig,
) -> CurrentTurnPolicy {
    CurrentTurnPolicy {
        prefill_mode: PrefillMode::Dense,
    }
}
