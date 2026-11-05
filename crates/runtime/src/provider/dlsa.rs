// SPDX-License-Identifier: Apache-2.0
//
//! The provider's D-LSA switch and the request paths it refuses (`docs/dlsa.md` §12). D-LSA is
//! a property of the loaded model; while it is on, only the baseline decoder runs, without
//! prefix reuse, snapshots, SpecPrefill or image prompts (D-LSA task 5 lifts these).

use anyhow::{Result, bail};
use mlxcel_core::generate::PrefixReuse;
use tracing::info;

use super::{Qwen35GenerationMode, Qwen35Provider};
use crate::dlsa::{DlsaConfig, DlsaStats};
use crate::specprefill::PrefillMode;

/// The decoder a request gets under D-LSA: `Automatic` resolves to baseline, explicit MTP and
/// DFlash2 are refused (their multi-row verification and rollback are not index-aware yet).
pub(crate) fn dlsa_decoder(
    mode: Qwen35GenerationMode,
) -> std::result::Result<Qwen35GenerationMode, String> {
    match mode {
        Qwen35GenerationMode::Automatic | Qwen35GenerationMode::Baseline => {
            Ok(Qwen35GenerationMode::Baseline)
        }
        Qwen35GenerationMode::Mtp | Qwen35GenerationMode::Dflash2 => Err(format!(
            "the {} decoder does not support D-LSA yet (D-LSA task 5); use --decoder baseline or disable D-LSA",
            if mode == Qwen35GenerationMode::Mtp {
                "MTP"
            } else {
                "DFlash2"
            }
        )),
    }
}

fn refusal(what: &str) -> String {
    format!("D-LSA (task 3) does not support {what} yet; disable D-LSA for this request")
}

impl Qwen35Provider {
    /// Turn D-LSA on for the loaded model (which must use an fp16 or int8 KV cache, as
    /// `qw generate` loads it). Every later generation runs the baseline decoder with sparse
    /// attention in the full-attention layers.
    pub fn enable_dlsa(&mut self, config: DlsaConfig) -> Result<()> {
        self.model.enable_dlsa(config)
    }

    /// Back to dense attention.
    pub fn disable_dlsa(&mut self) {
        self.model.disable_dlsa();
    }

    pub fn dlsa_enabled(&self) -> bool {
        self.model.dlsa_enabled()
    }

    /// Counters since D-LSA was enabled or the last generation started; `None` when off.
    pub fn dlsa_stats(&self) -> Option<DlsaStats> {
        self.model.dlsa_stats()
    }

    /// Bytes the D-LSA index of the current sequence holds.
    pub fn dlsa_memory_bytes(&self) -> usize {
        self.model.dlsa_memory_bytes()
    }

    /// `Err` naming `what` while D-LSA is on.
    pub(super) fn refuse_under_dlsa(&self, what: &str) -> Result<()> {
        if self.model.dlsa_enabled() {
            bail!(refusal(what));
        }
        Ok(())
    }

    /// The baseline request options D-LSA cannot serve yet: a restored prefix and SpecPrefill.
    pub(super) fn refuse_baseline_options_under_dlsa(
        &self,
        prefix_reuse: Option<&PrefixReuse<'_>>,
        prefill_mode: &PrefillMode,
    ) -> Result<()> {
        if prefix_reuse.is_some_and(|reuse| reuse.cached_tokens > 0) {
            self.refuse_under_dlsa("prefix reuse (snapshots carry no index state)")?;
        }
        if matches!(prefill_mode, PrefillMode::SpecPrefill(_)) {
            self.refuse_under_dlsa("SpecPrefill")?;
        }
        Ok(())
    }

    /// The decoder of a request under D-LSA ([`dlsa_decoder`]); `mode` unchanged when off.
    pub(super) fn dlsa_generation_mode(
        &self,
        mode: Qwen35GenerationMode,
    ) -> Result<Qwen35GenerationMode> {
        if !self.model.dlsa_enabled() {
            return Ok(mode);
        }
        let decoder = dlsa_decoder(mode).map_err(anyhow::Error::msg)?;
        if mode == Qwen35GenerationMode::Automatic {
            info!(
                phase = "decoder.dlsa",
                "D-LSA is on: the automatic decoder resolves to baseline"
            );
        }
        Ok(decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_resolves_to_baseline_and_speculative_decoders_are_refused() {
        assert_eq!(
            dlsa_decoder(Qwen35GenerationMode::Automatic),
            Ok(Qwen35GenerationMode::Baseline)
        );
        assert_eq!(
            dlsa_decoder(Qwen35GenerationMode::Baseline),
            Ok(Qwen35GenerationMode::Baseline)
        );
        for (mode, name) in [
            (Qwen35GenerationMode::Mtp, "MTP"),
            (Qwen35GenerationMode::Dflash2, "DFlash2"),
        ] {
            let err = dlsa_decoder(mode).unwrap_err();
            assert!(
                err.contains(name) && err.contains("--decoder baseline"),
                "{err}"
            );
        }
    }

    #[test]
    fn refusals_name_what_is_refused() {
        let message = refusal("SpecPrefill");
        assert!(
            message.contains("D-LSA") && message.contains("SpecPrefill"),
            "{message}"
        );
    }
}
