// SPDX-License-Identifier: Apache-2.0
//
//! The provider's D-LSA switch and the request paths it refuses (`docs/dlsa.md` §12, §13). D-LSA
//! is a property of the loaded model; while it is on, every decoder (baseline, MTP, DFlash2),
//! prefix reuse and snapshots work, and SpecPrefill and image prompts are refused.

use anyhow::{Result, bail};

use super::Qwen35Provider;
use crate::dlsa::{DlsaConfig, DlsaStats};
use crate::specprefill::PrefillMode;

fn refusal(what: &str) -> String {
    format!("D-LSA does not support {what}; disable D-LSA (start without --dlsa) to use it")
}

impl Qwen35Provider {
    /// Turn D-LSA on for the loaded model (which must use an fp16 or int8 KV cache). Every
    /// later generation runs sparse attention in the full-attention layers, and image inputs
    /// are refused ([`Qwen35Provider::supports_image_inputs`] turns false).
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

    /// SpecPrefill keeps a subset of the prompt in the KV cache, so its slots are not positions,
    /// which the index requires (`docs/dlsa.md` §7.1.4).
    pub(super) fn refuse_prefill_mode_under_dlsa(&self, prefill_mode: &PrefillMode) -> Result<()> {
        if matches!(prefill_mode, PrefillMode::SpecPrefill(_)) {
            self.refuse_under_dlsa("SpecPrefill (its KV slots are not positions)")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_name_what_is_refused() {
        let message = refusal("SpecPrefill");
        assert!(
            message.contains("D-LSA") && message.contains("SpecPrefill"),
            "{message}"
        );
        assert!(!message.contains("task"), "{message}");
    }
}
