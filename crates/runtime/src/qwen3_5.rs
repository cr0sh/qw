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

//! Dense Qwen3.5 hybrid text model.
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py

use crate::gated_delta::{
    GatedDeltaCache, RMSNormGated, gated_delta_update, scaled_fast_rms_norm_no_weight,
};
use crate::model_owned::ModelOwnedSequenceState;
use crate::qwen_mrope_state::MRopeState;
use crate::qwen_vl_position::decode_rope_positions;
use crate::qwen3_5_mtp::Qwen35MtpDraftModel;
use crate::qwen3_next::{Mlp, Quantization, Qwen3NextAttention, Qwen3NextCache, Qwen3NextConfig};
use crate::qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};
use anyhow::{Context, Result, ensure};
use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::{
    KVCache, QuantizedWeight, RMSNorm, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::silu;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

const MTP_DRAFT_PREFIX: i32 = 65_536;
const MTP_DRAFT_PADDED: i32 = 65_568;
const DFLASH_VERIFY_PREFIX: i32 = 131_072;
const DFLASH_VERIFY_PADDED: i32 = 131_104;
const DRAFT_CONTROL_START: i32 = 248_044;
const DRAFT_CONTROL_END: i32 = 248_070;
#[cfg(any(feature = "specprefill", test))]
const SPECPREFILL_TARGET_CHUNK_TOKENS: usize = 512;

fn compact_rows(array: &MlxArray, prefix_len: i32, padded_len: i32) -> UniquePtr<MlxArray> {
    let columns = mlxcel_core::array_shape(array)[1];
    let prefix = mlxcel_core::slice(array, &[0, 0], &[prefix_len, columns]);
    let controls = mlxcel_core::slice(
        array,
        &[DRAFT_CONTROL_START, 0],
        &[DRAFT_CONTROL_END, columns],
    );
    let real = prefix_len + DRAFT_CONTROL_END - DRAFT_CONTROL_START;
    let padding = mlxcel_core::slice(array, &[0, 0], &[padded_len - real, columns]);
    let compact = concatenate(&prefix, &controls, 0);
    concatenate(&compact, &padding, 0)
}

fn compact_head(
    head: &UnifiedLinear,
    vocab_size: usize,
    prefix_len: i32,
    padded_len: i32,
) -> Option<UnifiedLinear> {
    let UnifiedLinear::Quantized { weight, bias: None } = head else {
        return None;
    };
    (vocab_size == 248_320).then(|| UnifiedLinear::Quantized {
        weight: QuantizedWeight {
            weight: compact_rows(&weight.weight, prefix_len, padded_len),
            scales: compact_rows(&weight.scales, prefix_len, padded_len),
            biases: weight
                .biases
                .as_ref()
                .map(|x| compact_rows(x, prefix_len, padded_len)),
            group_size: weight.group_size,
            bits: weight.bits,
            mode: weight.mode.clone(),
            global_scale: weight.global_scale.as_ref().map(|x| mlxcel_core::copy(x)),
        },
        bias: None,
    })
}

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,

    // Linear attention parameters
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,

    // Rope parameters (dict format)
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,

    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default, alias = "quantization_config")]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: Option<bool>,
    #[serde(default)]
    pub vision_config: Option<Qwen3VLVisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub video_token_id: Option<i32>,
    #[serde(default)]
    pub vision_start_token_id: Option<i32>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_linear_num_value_heads() -> usize {
    64
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_key_head_dim() -> usize {
    192
}
fn default_linear_value_head_dim() -> usize {
    128
}
fn default_linear_conv_kernel_dim() -> usize {
    4
}

impl Qwen35Config {
    pub fn quant_params(&self, prefix: &str) -> (i32, i32) {
        let Some(quantization) = &self.quantization else {
            return (64, 4);
        };
        let language_model_prefix = format!("language_model.{prefix}");
        quantization
            .overrides
            .get(prefix)
            .or_else(|| quantization.overrides.get(&language_model_prefix))
            .map(|value| (value.group_size, value.bits))
            .unwrap_or((quantization.group_size, quantization.bits))
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(100000.0)
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.25)
    }

    pub(crate) fn mrope_section(&self) -> Vec<i32> {
        self.rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get("mrope_section"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_i64)
                    .map(|value| value as i32)
                    .collect::<Vec<_>>()
            })
            .filter(|sections| sections.len() == 3)
            .unwrap_or_else(|| vec![11, 11, 10])
    }

    fn head_dim_resolved(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Convert to Qwen3NextConfig for reusing shared components
    pub fn to_qwen3next_config(&self) -> Qwen3NextConfig {
        Qwen3NextConfig {
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim_resolved(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta(),
            partial_rotary_factor: self.partial_rotary_factor(),
            quantization: self.quantization.clone(),
            mrope_section: self.mrope_section(),
        }
    }

    fn validate_mtp_metadata(&self, config_path: &Path) -> Result<()> {
        match (
            self.mtp_num_hidden_layers,
            self.mtp_use_dedicated_embeddings,
        ) {
            (None, None) => Ok(()),
            (Some(1), Some(false)) => Ok(()),
            (Some(layers), Some(false)) => anyhow::bail!(
                "text_config.mtp_num_hidden_layers in {} must be exactly 1, got {layers}",
                config_path.display()
            ),
            (_, Some(true)) => anyhow::bail!(
                "text_config.mtp_use_dedicated_embeddings in {} must be false",
                config_path.display()
            ),
            _ => anyhow::bail!(
                "incomplete MTP metadata in {}; text_config.mtp_num_hidden_layers and \
                 text_config.mtp_use_dedicated_embeddings must be declared together",
                config_path.display()
            ),
        }
    }

    pub(crate) fn has_mtp_metadata(&self) -> bool {
        self.mtp_num_hidden_layers == Some(1) && self.mtp_use_dedicated_embeddings == Some(false)
    }
}

pub(crate) struct GdnRollbackSnapshot {
    layer_idx: usize,
    q: UniquePtr<MlxArray>,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
    a: UniquePtr<MlxArray>,
    b: UniquePtr<MlxArray>,
    init_state: Option<UniquePtr<MlxArray>>,
    conv_input: UniquePtr<MlxArray>,
}

pub(crate) struct Qwen35MtpPrefill {
    pub(crate) hidden: UniquePtr<MlxArray>,
    pub(crate) first_logits: UniquePtr<MlxArray>,
}

pub(crate) struct Qwen35MtpVerifyOutput {
    pub(crate) hidden: UniquePtr<MlxArray>,
    pub(crate) logits: UniquePtr<MlxArray>,
    pub(crate) gdn_states: Vec<GdnRollbackSnapshot>,
}

#[cfg(any(feature = "dflash2", test))]
/// DFlash2 verify output: the post-layer hidden state at each
/// `target_layer_ids[i]` (copied before the final norm), the final logits,
/// and the GDN rollback snapshots. Mirrors the SGLang DFLASH verify forward,
/// which captures the target's hidden states at the draft's configured layer
/// ids for the next draft round's context.
pub(crate) struct Qwen35DflashVerifyOutput {
    pub(crate) hidden_by_layer: Vec<UniquePtr<MlxArray>>,
    pub(crate) logits: UniquePtr<MlxArray>,
    pub(crate) gdn_states: Vec<GdnRollbackSnapshot>,
}

#[cfg(any(feature = "dflash2", test))]
/// DFlash2 prefill output: `hidden_concat` is the per-layer captured hidden
/// states concatenated along the hidden axis `[1, P', K * hidden]`; the
/// drafter consumes it as its context buffer. `first_logits` is the last
/// position's logits (the first sampled token). `hidden_offset` is the number
/// of leading rows dropped once the captured rows exceeded the drafter's
/// sliding-window limit (0 when nothing was dropped); the drafter cache
/// offsets are aligned to it.
pub(crate) struct Qwen35DflashPrefill {
    pub(crate) hidden_concat: UniquePtr<MlxArray>,
    pub(crate) first_logits: UniquePtr<MlxArray>,
    pub(crate) hidden_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen35RollbackPlan {
    pub(crate) accepted_block_len: i32,
    pub(crate) trim: i32,
    pub(crate) final_offset: i32,
}

pub(crate) fn rollback_plan(
    verify_offset: i32,
    accepted: usize,
    block_size: usize,
) -> Qwen35RollbackPlan {
    let accepted_block_len = i32::try_from(accepted.saturating_add(1)).unwrap_or(i32::MAX);
    let block_size = i32::try_from(block_size).unwrap_or(i32::MAX);
    let trim = (block_size - accepted_block_len).max(0);
    Qwen35RollbackPlan {
        accepted_block_len,
        trim,
        final_offset: verify_offset - trim,
    }
}

enum Qwen35GatedAuxProjections {
    Separate {
        z: UnifiedLinear,
        b: UnifiedLinear,
        a: UnifiedLinear,
    },
    Fused(UnifiedLinear),
}

fn fuse_gated_aux_projections(
    weights: &WeightMap,
    prefixes: [&str; 3],
    z: UnifiedLinear,
    b: UnifiedLinear,
    a: UnifiedLinear,
) -> Qwen35GatedAuxProjections {
    let can_drop_linear_biases = prefixes
        .iter()
        .all(|prefix| !weights.contains_key(&format!("{prefix}.bias")));
    let fused = (|| {
        let (z_weight, b_weight, a_weight) = (
            z.quantized_weight()?,
            b.quantized_weight()?,
            a.quantized_weight()?,
        );
        if !can_drop_linear_biases
            || z_weight.group_size != b_weight.group_size
            || z_weight.group_size != a_weight.group_size
            || z_weight.bits != b_weight.bits
            || z_weight.bits != a_weight.bits
            || z_weight.mode != b_weight.mode
            || z_weight.mode != a_weight.mode
            || z_weight.global_scale.is_some()
            || b_weight.global_scale.is_some()
            || a_weight.global_scale.is_some()
        {
            return None;
        }
        let biases = [
            z_weight.biases.as_deref()?,
            b_weight.biases.as_deref()?,
            a_weight.biases.as_deref()?,
        ];
        let weight = concatenate(
            &concatenate(&z_weight.weight, &b_weight.weight, 0),
            &a_weight.weight,
            0,
        );
        let scales = concatenate(
            &concatenate(&z_weight.scales, &b_weight.scales, 0),
            &a_weight.scales,
            0,
        );
        let biases = concatenate(&concatenate(biases[0], biases[1], 0), biases[2], 0);
        Some(UnifiedLinear::new(
            QuantizedWeight::new(weight, scales, biases, z_weight.group_size, z_weight.bits),
            None,
        ))
    })();

    fused.map_or(
        Qwen35GatedAuxProjections::Separate { z, b, a },
        Qwen35GatedAuxProjections::Fused,
    )
}

// GatedDeltaNet - Qwen3.5 variant with separately stored projections.
/// Fuses compatible z, beta, and decay projections at load time.
#[allow(dead_code)]
pub(crate) struct Qwen35GatedDeltaNet {
    hidden_size: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,

    conv1d_weight: UniquePtr<MlxArray>,
    in_proj_qkv: UnifiedLinear,
    aux_projections: Qwen35GatedAuxProjections,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    norm: RMSNormGated,
    out_proj: UnifiedLinear,
}

impl Qwen35GatedDeltaNet {
    pub(crate) fn forward(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        let out = self.forward_hidden_internal(inputs, mask, cache, None);
        self.out_proj.forward(&out)
    }

    fn forward_with_capture(
        &self,
        layer_idx: usize,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut GatedDeltaCache>,
        snapshots: &mut Vec<GdnRollbackSnapshot>,
    ) -> UniquePtr<MlxArray> {
        let out = self.forward_hidden_internal(inputs, mask, cache, Some((layer_idx, snapshots)));
        self.out_proj.forward(&out)
    }

    fn forward_hidden_internal(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
        snapshot: Option<(usize, &mut Vec<GdnRollbackSnapshot>)>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        let effective_mask = mask;

        let qkv = self.in_proj_qkv.forward(inputs);
        let (z, b_proj, a) = match &self.aux_projections {
            Qwen35GatedAuxProjections::Separate { z, b, a } => {
                (z.forward(inputs), b.forward(inputs), a.forward(inputs))
            }
            Qwen35GatedAuxProjections::Fused(projection) => {
                let projected = projection.forward(inputs);
                let z_end = self.value_dim as i32;
                let b_end = z_end + self.num_v_heads as i32;
                let a_end = b_end + self.num_v_heads as i32;
                (
                    mlxcel_core::slice(&projected, &[0, 0, 0], &[b, s, z_end]),
                    mlxcel_core::slice(&projected, &[0, 0, z_end], &[b, s, b_end]),
                    mlxcel_core::slice(&projected, &[0, 0, b_end], &[b, s, a_end]),
                )
            }
        };
        let z = mlxcel_core::reshape(&z, &[b, s, self.num_v_heads as i32, self.head_v_dim as i32]);

        // Get conv state from cache
        let input_dtype = mlxcel_core::array_dtype(&qkv);
        let conv_state = if let Some(ref c) = cache {
            c.conv_state
                .as_ref()
                .and_then(|s| {
                    let s = s.as_ref().unwrap();
                    let state_shape = mlxcel_core::array_shape(s);
                    // Guard: reinitialize if batch dimension doesn't match (continuous batching)
                    if state_shape[0] != b {
                        None
                    } else {
                        Some(mlxcel_core::copy(s))
                    }
                })
                .unwrap_or_else(|| {
                    mlxcel_core::zeros(
                        &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
                        input_dtype,
                    )
                })
        } else {
            mlxcel_core::zeros(
                &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
                input_dtype,
            )
        };

        // Guard: discard mask if batch dimension doesn't match (continuous batching).
        // Uses guarded_mask consistently for both the conv masking and gated_delta_update.
        let guarded_mask = effective_mask.filter(|m| {
            let mask_shape = mlxcel_core::array_shape(m);
            mask_shape[0] == b
        });

        // Apply mask if present (mask qkv before conv)
        let qkv = if let Some(m) = guarded_mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, input_dtype);
            mlxcel_core::where_cond(&m_exp, &qkv, &zero)
        } else {
            qkv
        };

        // Concatenate with conv state
        let conv_input = concatenate(&conv_state, &qkv, 1);

        // Update cache with new conv state.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full conv_input allocation, causing a
        // memory leak proportional to the sequence length.
        if let Some(c) = cache.as_deref_mut() {
            let n_keep = (self.conv_kernel_size - 1) as i32;
            let conv_shape = mlxcel_core::array_shape(&conv_input);
            let conv_len = conv_shape[1];
            let tail = mlxcel_core::slice(
                &conv_input,
                &[0, conv_len - n_keep, 0],
                &[b, conv_len, self.conv_dim as i32],
            );
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        // Apply conv1d with SiLU activation
        let conv_out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = silu(&conv_out);

        // Split conv output into q, k, v
        // Note: MLX slice with stop=-1 means dim_size-1 (excludes last), not "to end"
        // Use actual conv_out seq length for correct slicing
        let conv_out_shape = mlxcel_core::array_shape(&conv_out);
        let conv_seq = conv_out_shape[1];
        let q_out = mlxcel_core::slice(&conv_out, &[0, 0, 0], &[b, conv_seq, self.key_dim as i32]);
        let k_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, self.key_dim as i32],
            &[b, conv_seq, (2 * self.key_dim) as i32],
        );
        let v_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, (2 * self.key_dim) as i32],
            &[b, conv_seq, self.conv_dim as i32],
        );

        // Reshape to heads
        let q = mlxcel_core::reshape(
            &q_out,
            &[b, s, self.num_k_heads as i32, self.head_k_dim as i32],
        );
        let k = mlxcel_core::reshape(
            &k_out,
            &[b, s, self.num_k_heads as i32, self.head_k_dim as i32],
        );
        let v = mlxcel_core::reshape(
            &v_out,
            &[b, s, self.num_v_heads as i32, self.head_v_dim as i32],
        );

        // Get recurrent state from cache
        // Guard: discard cached state if batch dimension doesn't match (continuous batching)
        let state = cache.as_ref().and_then(|c| {
            c.state_cache.as_ref().and_then(|s| {
                let s = s.as_ref().unwrap();
                let state_shape = mlxcel_core::array_shape(s);
                if state_shape[0] != b {
                    None
                } else {
                    Some(mlxcel_core::copy(s))
                }
            })
        });

        // Apply RMS norm with scaling (same as Qwen3Next). Reference mlx-lm
        // keeps this on mx.fast.rms_norm rather than expanding it into
        // primitive ops.
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        if let Some((layer_idx, snapshots)) = snapshot {
            snapshots.push(GdnRollbackSnapshot {
                layer_idx,
                q: mlxcel_core::share(&q),
                k: mlxcel_core::share(&k),
                v: mlxcel_core::share(&v),
                a: mlxcel_core::share(&a),
                b: mlxcel_core::share(&b_proj),
                init_state: state.as_ref().map(|value| mlxcel_core::share(value)),
                conv_input: mlxcel_core::share(&conv_input),
            });
        }

        // Run gated delta update (use guarded_mask which is None if batch dims mismatch)
        let (out, new_state) = gated_delta_update(
            (&q, &k, &v),
            (&a, &b_proj, &self.a_log, &self.dt_bias),
            state.as_deref(),
            guarded_mask,
        );

        // Update cache state
        if let Some(c) = cache {
            c.state_cache = Some(new_state);
            c.advance(s);
        }

        // Apply norm with gating
        let out = self.norm.forward(&out, Some(&z));
        mlxcel_core::reshape(&out, &[b, s, -1])
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv_prefix = format!("{}.in_proj_qkv", prefix);
        let z_prefix = format!("{}.in_proj_z", prefix);
        let b_prefix = format!("{}.in_proj_b", prefix);
        let a_prefix = format!("{}.in_proj_a", prefix);
        let out_prefix = format!("{}.out_proj", prefix);
        let (qkv_group_size, qkv_bits) = config.quant_params(&qkv_prefix);
        let (z_group_size, z_bits) = config.quant_params(&z_prefix);
        let (b_group_size, b_bits) = config.quant_params(&b_prefix);
        let (a_group_size, a_bits) = config.quant_params(&a_prefix);
        let (out_group_size, out_bits) = config.quant_params(&out_prefix);

        let conv1d_weight = weights
            .get(&format!("{}.conv1d.weight", prefix))
            .map(|w| {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    mlxcel_core::swap_axes(w, -1, -2)
                } else {
                    mlxcel_core::copy(w)
                }
            })
            .ok_or_else(|| format!("Missing conv1d weight: {}", prefix))?;

        // Qwen3.5 uses separate projections instead of combined projections.
        let in_proj_qkv =
            UnifiedLinear::from_weights(weights, &qkv_prefix, qkv_group_size, qkv_bits)?;
        let in_proj_z = UnifiedLinear::from_weights(weights, &z_prefix, z_group_size, z_bits)?;
        let in_proj_b = UnifiedLinear::from_weights(weights, &b_prefix, b_group_size, b_bits)?;
        let in_proj_a = UnifiedLinear::from_weights(weights, &a_prefix, a_group_size, a_bits)?;
        let aux_projections = fuse_gated_aux_projections(
            weights,
            [&z_prefix, &b_prefix, &a_prefix],
            in_proj_z,
            in_proj_b,
            in_proj_a,
        );

        let dt_bias = weights
            .get(&format!("{}.dt_bias", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing dt_bias: {}", prefix))?;

        let a_log = weights
            .get(&format!("{}.A_log", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing A_log: {}", prefix))?;

        let norm_weight = weights
            .get(&format!("{}.norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing norm weight: {}", prefix))?;

        let out_proj = UnifiedLinear::from_weights(weights, &out_prefix, out_group_size, out_bits)?;

        Ok(Self {
            hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d_weight,
            in_proj_qkv,
            aux_projections,
            dt_bias,
            a_log,
            norm: RMSNormGated::new(norm_weight, config.rms_norm_eps),
            out_proj,
        })
    }
}

// Decoder Layer.
/// Attention variant for Qwen3.5
pub(crate) enum Qwen35AttentionVariant {
    FullAttention(Qwen3NextAttention),
    Linear(Qwen35GatedDeltaNet),
}

pub(crate) struct Qwen35DecoderLayer {
    pub(crate) is_linear: bool,
    pub(crate) attention: Qwen35AttentionVariant,
    pub(crate) mlp: Mlp,
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) post_attention_layernorm: RMSNorm,
}

impl Qwen35DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let residual = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attention), Qwen3NextCache::Linear(cache)) => {
                attention.forward(&normed, mask, Some(cache))
            }
            (Qwen35AttentionVariant::Linear(attention), _) => {
                attention.forward(&normed, mask, None)
            }
            (
                Qwen35AttentionVariant::FullAttention(attention),
                Qwen3NextCache::Attention(cache),
            ) => attention.forward_with_position_ids(&normed, cache, mask, position_ids),
            (Qwen35AttentionVariant::FullAttention(attention), _) => {
                let mut temporary = KVCache::new();
                attention.forward_with_position_ids(&normed, &mut temporary, mask, position_ids)
            }
        };
        let hidden = mlxcel_core::add(x, &residual);
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden));
        mlxcel_core::add(&hidden, &mlp)
    }

    #[cfg(any(feature = "specprefill", test))]
    fn forward_with_query_capture(
        &self,
        x: &MlxArray,
        cache: &mut Qwen3NextCache,
    ) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
        let normed = self.input_layernorm.forward(x);
        let (residual, query) = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attention), Qwen3NextCache::Linear(cache)) => {
                (attention.forward(&normed, None, Some(cache)), None)
            }
            (Qwen35AttentionVariant::FullAttention(attention), Qwen3NextCache::Attention(cache)) => {
                let (output, query) =
                    attention.forward_with_query_capture(&normed, cache, None, None);
                (output, Some(query))
            }
            _ => unreachable!("Qwen3.5 layer/cache topology must match"),
        };
        let hidden = mlxcel_core::add(x, &residual);
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden));
        (mlxcel_core::add(&hidden, &mlp), query)
    }

    fn forward_with_capture(
        &self,
        layer_idx: usize,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
        position_ids: Option<&MlxArray>,
        snapshots: &mut Vec<GdnRollbackSnapshot>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let residual = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attn), Qwen3NextCache::Linear(cache)) => {
                attn.forward_with_capture(layer_idx, &normed, mask, Some(cache), snapshots)
            }
            (Qwen35AttentionVariant::Linear(attn), _) => {
                attn.forward_with_capture(layer_idx, &normed, mask, None, snapshots)
            }
            (Qwen35AttentionVariant::FullAttention(attn), Qwen3NextCache::Attention(cache)) => {
                attn.forward_verify(&normed, cache, mask, position_ids)
            }
            (Qwen35AttentionVariant::FullAttention(attn), _) => {
                let mut temporary = KVCache::new();
                attn.forward_verify(&normed, &mut temporary, mask, position_ids)
            }
        };
        let hidden = mlxcel_core::add(x, &residual);
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden));
        mlxcel_core::add(&hidden, &mlp)
    }

    pub(crate) fn forward_full_attention(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let residual = match &self.attention {
            Qwen35AttentionVariant::FullAttention(attention) => {
                attention.forward_with_position_ids(&normed, cache, mask, position_ids)
            }
            Qwen35AttentionVariant::Linear(_) => {
                unreachable!("the bundled MTP layer must use full attention")
            }
        };
        let hidden = mlxcel_core::add(x, &residual);
        let mlp = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden));
        mlxcel_core::add(&hidden, &mlp)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        qn_config: &Qwen3NextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        Self::from_weights_at_prefix(
            weights,
            config,
            qn_config,
            &format!("model.layers.{layer_idx}"),
            config.is_linear_layer(layer_idx),
        )
    }

    pub(crate) fn from_weights_at_prefix(
        weights: &WeightMap,
        config: &Qwen35Config,
        qn_config: &Qwen3NextConfig,
        prefix: &str,
        is_linear: bool,
    ) -> Result<Self, String> {
        let attention = if is_linear {
            Qwen35AttentionVariant::Linear(Qwen35GatedDeltaNet::from_weights(
                weights,
                config,
                &format!("{}.linear_attn", prefix),
            )?)
        } else {
            Qwen35AttentionVariant::FullAttention(Qwen3NextAttention::from_weights(
                weights,
                qn_config,
                &format!("{}.self_attn", prefix),
            )?)
        };

        let mlp = Mlp::from_weights(weights, qn_config, &format!("{}.mlp", prefix))?;

        let input_norm_weight = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing input_layernorm: {}", prefix))?;

        let post_norm_weight = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing post_attention_layernorm: {}", prefix))?;

        Ok(Self {
            is_linear,
            attention,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_weight, config.rms_norm_eps),
        })
    }
}

// Qwen3.5 Model.
// The 27B Qwen3.5 target uses 4 GiB of FP16 K/V at this boundary. Keeping
// 64k native avoids the much slower compressed-attention path; longer contexts
// still transition to bounded Turbo4 storage.
const MTP_FP16_TARGET_MAX_TOKENS: i32 = 65_536;

fn mtp_target_cache_mode(has_mtp: bool, requested: KVCacheMode) -> KVCacheMode {
    if has_mtp && requested == KVCacheMode::Turbo4 {
        KVCacheMode::Fp16
    } else {
        requested
    }
}

pub struct Qwen35Model {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<Qwen35DecoderLayer>,
    pub(crate) norm: RMSNorm,
    pub(crate) lm_head: Option<UnifiedLinear>,
    compact_draft_head: Option<UnifiedLinear>,
    compact_dflash_verify_head: Option<UnifiedLinear>,
    pub(crate) config: Qwen35Config,
    mtp: Option<Qwen35MtpDraftModel>,
    kv_cache_mode: KVCacheMode,
    bounded_mtp_fp16: bool,
    vision: Option<Qwen3VLVisionEncoder>,
    /// Model-owned heterogeneous cache state used by one synchronous sequence.
    sequence_state: ModelOwnedSequenceState<Qwen3NextCache>,
    /// MRoPE position state retained for the Qwen3.5 text path.
    mrope_state: MRopeState,
}

impl Qwen35Model {
    fn forward_backbone_with_inputs(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen3NextCache],
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut hidden = input_embeddings
            .map(mlxcel_core::copy)
            .unwrap_or_else(|| self.embed_tokens.forward(input_ids));

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(&hidden, None, cache, position_ids);
        }
        hidden
    }

    pub(crate) fn project_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(hidden)
        } else {
            self.embed_tokens.as_linear(hidden)
        }
    }
    pub(crate) fn project_mtp_continuation_logits(
        &self,
        hidden_row: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        self.project_logits(&self.norm.forward(hidden_row))
    }

    pub(crate) fn project_draft_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.project_compact_logits(hidden, &self.compact_draft_head, MTP_DRAFT_PREFIX)
    }

    pub(crate) fn project_dflash_verify_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.project_compact_logits(
            hidden,
            &self.compact_dflash_verify_head,
            DFLASH_VERIFY_PREFIX,
        )
    }

    fn project_compact_logits(
        &self,
        hidden: &MlxArray,
        head: &Option<UnifiedLinear>,
        prefix_len: i32,
    ) -> UniquePtr<MlxArray> {
        head.as_ref().map_or_else(
            || self.project_logits(hidden),
            |head| {
                let padded = head.forward(hidden);
                let shape = mlxcel_core::array_shape(&padded);
                mlxcel_core::slice(
                    &padded,
                    &[0, 0, 0],
                    &[
                        shape[0],
                        shape[1],
                        prefix_len + DRAFT_CONTROL_END - DRAFT_CONTROL_START,
                    ],
                )
            },
        )
    }

    pub(crate) fn has_compact_draft_head(&self) -> bool {
        self.compact_draft_head.is_some()
    }

    #[cfg(any(feature = "dflash2", test))]
    pub(crate) fn has_compact_dflash_verify_head(&self) -> bool {
        self.compact_dflash_verify_head.is_some()
    }

    pub(crate) fn map_draft_token(token: i32) -> i32 {
        if token < MTP_DRAFT_PREFIX {
            token
        } else {
            token + DRAFT_CONTROL_START - MTP_DRAFT_PREFIX
        }
    }

    pub(crate) fn map_dflash_verify_token(token: i32) -> i32 {
        if token < DFLASH_VERIFY_PREFIX {
            token
        } else {
            token + DRAFT_CONTROL_START - DFLASH_VERIFY_PREFIX
        }
    }

    fn make_internal_caches(&self) -> Vec<Qwen3NextCache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                if layer.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    let mut cache = KVCache::new_with_mode(self.kv_cache_mode);
                    if self.bounded_mtp_fp16 && index >= 48 {
                        cache.enable_fp16_v_quantization_on_write();
                    }
                    Qwen3NextCache::Attention(Box::new(cache))
                }
            })
            .collect()
    }

    pub(crate) fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub(crate) fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub(crate) fn vision_config(&self) -> Option<&Qwen3VLVisionConfig> {
        self.config.vision_config.as_ref()
    }

    pub(crate) fn multimodal_token_ids(&self) -> Option<(i32, i32, i32)> {
        Some((
            self.config.image_token_id?,
            self.config.video_token_id?,
            self.config.vision_start_token_id?,
        ))
    }

    pub(crate) fn encode_vision(
        &self,
        pixel_values: &MlxArray,
        grids: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>> {
        let vision = self
            .vision
            .as_ref()
            .context("model does not support image inputs")?;
        let output = vision.forward_with_grid(pixel_values, grids);
        ensure!(
            output.deepstack_features.is_empty(),
            "Qwen3.5 DeepStack vision features are not supported"
        );
        Ok(output.hidden_states)
    }

    pub(crate) fn prepare_mrope(&self, position_ids: &MlxArray, rope_delta: i32) {
        self.mrope_state.prepare(position_ids, rope_delta);
    }

    pub(crate) fn clear_prepared_mrope(&self) {
        self.mrope_state.clear_prepared();
    }

    pub(crate) fn mtp(&self) -> Option<&Qwen35MtpDraftModel> {
        self.mtp.as_ref()
    }
    fn enforce_mtp_cache_bound(&self, projected_tokens: i32) {
        if !self.bounded_mtp_fp16 || projected_tokens <= MTP_FP16_TARGET_MAX_TOKENS {
            return;
        }
        self.sequence_state.with_internal(|caches| {
            for cache in caches {
                if let Qwen3NextCache::Attention(cache) = cache {
                    if cache.offset == 0 {
                        cache.mode = KVCacheMode::Turbo4;
                    } else {
                        cache.demote_fp16_to_turbo4();
                    }
                }
            }
        });
    }

    pub(crate) fn forward_mtp_prefill_chunks<F>(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
        rope_delta: Option<i32>,
        mut consume_chunk: F,
    ) -> std::result::Result<Qwen35MtpPrefill, String>
    where
        F: FnMut(i32, i32, &MlxArray),
    {
        self.reset_runtime_state();
        if let (Some(position_ids), Some(rope_delta)) = (position_ids, rope_delta) {
            self.mrope_state.prepare(position_ids, rope_delta);
            self.mrope_state.activate_prepared()?;
        }

        let shape = mlxcel_core::array_shape(input_ids);
        let prompt_len = shape[1];
        let configured = mlxcel_core::generate::prefill_chunk_len();
        let chunk_len =
            mlxcel_core::generate::effective_prefill_chunk(configured, true, prompt_len as usize)
                .unwrap_or(prompt_len as usize) as i32;
        let mut final_chunk = None;
        let mut final_logits = None;
        self.enforce_mtp_cache_bound(prompt_len);
        let mut start = 0;
        while start < prompt_len {
            let end = (start + chunk_len).min(prompt_len);
            let ids = mlxcel_core::slice(input_ids, &[0, start], &[shape[0], end]);
            let embeddings = input_embeddings.map(|embeddings| {
                let embedding_shape = mlxcel_core::array_shape(embeddings);
                mlxcel_core::slice(
                    embeddings,
                    &[0, start, 0],
                    &[embedding_shape[0], end, embedding_shape[2]],
                )
            });
            let positions = position_ids.map(|positions| {
                let position_shape = mlxcel_core::array_shape(positions);
                mlxcel_core::slice(
                    positions,
                    &[0, 0, start],
                    &[position_shape[0], position_shape[1], end],
                )
            });
            let hidden = self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(
                    &ids,
                    embeddings.as_deref(),
                    caches,
                    positions.as_deref(),
                )
            });
            if end < prompt_len {
                consume_chunk(start, end, &hidden);
            } else {
                let hidden_shape = mlxcel_core::array_shape(&hidden);
                let last = hidden_shape[1] - 1;
                let last_hidden = mlxcel_core::slice(
                    &hidden,
                    &[0, last, 0],
                    &[hidden_shape[0], last + 1, hidden_shape[2]],
                );
                final_logits = Some(self.project_logits(&self.norm.forward(&last_hidden)));
                final_chunk = Some(hidden);
            }
            start = end;
        }

        let offset = self
            .sequence_state
            .with_internal(|caches| caches.first().map(Qwen3NextCache::offset).unwrap_or(0));
        self.mrope_state.set_position(offset);
        if position_ids.is_some() {
            self.mrope_state.finish_prefill();
        }
        Ok(Qwen35MtpPrefill {
            hidden: final_chunk.expect("MTP prefill requires a non-empty prompt"),
            first_logits: final_logits.expect("MTP prefill requires a non-empty prompt"),
        })
    }

    pub(crate) fn forward_mtp_text_suffix_chunks<F>(
        &self,
        input_ids: &MlxArray,
        mut consume_chunk: F,
    ) -> std::result::Result<Qwen35MtpPrefill, String>
    where
        F: FnMut(&MlxArray, &MlxArray),
    {
        let shape = mlxcel_core::array_shape(input_ids);
        let suffix_len = shape[1];
        if suffix_len == 0 {
            return Err("MTP suffix prefill requires at least one token".to_string());
        }
        let cached_len = self
            .sequence_state
            .with_internal(|caches| caches.first().map(Qwen3NextCache::offset).unwrap_or(0));
        self.enforce_mtp_cache_bound(cached_len + suffix_len);
        let configured = mlxcel_core::generate::prefill_chunk_len();
        let chunk_len =
            mlxcel_core::generate::effective_prefill_chunk(configured, true, suffix_len as usize)
                .unwrap_or(suffix_len as usize) as i32;
        let mut final_hidden = None;
        let mut start = 0;
        while start < suffix_len {
            let end = (start + chunk_len).min(suffix_len);
            let ids = mlxcel_core::slice(input_ids, &[0, start], &[shape[0], end]);
            let hidden = self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&ids, None, caches, None)
            });
            consume_chunk(&ids, &hidden);
            final_hidden = Some(hidden);
            start = end;
        }
        let hidden = final_hidden.expect("non-empty suffix produces target hidden state");
        let hidden_shape = mlxcel_core::array_shape(&hidden);
        let last = hidden_shape[1] - 1;
        let last_hidden = mlxcel_core::slice(
            &hidden,
            &[0, last, 0],
            &[hidden_shape[0], last + 1, hidden_shape[2]],
        );
        let first_logits = self.project_logits(&self.norm.forward(&last_hidden));
        let offset = self
            .sequence_state
            .with_internal(|caches| caches.first().map(Qwen3NextCache::offset).unwrap_or(0));
        self.mrope_state.set_position(offset);
        Ok(Qwen35MtpPrefill {
            hidden,
            first_logits,
        })
    }

    pub(crate) fn forward_mtp_verify(&self, input_ids: &MlxArray) -> Qwen35MtpVerifyOutput {
        self.forward_mtp_verify_with_compact(input_ids, false)
    }

    pub(crate) fn forward_mtp_verify_with_compact(
        &self,
        input_ids: &MlxArray,
        compact_logits: bool,
    ) -> Qwen35MtpVerifyOutput {
        let input_len = mlxcel_core::array_shape(input_ids)[1];
        let projected = self.sequence_state.with_internal(|caches| {
            caches.first().map(Qwen3NextCache::offset).unwrap_or(0) + input_len
        });
        self.enforce_mtp_cache_bound(projected);
        let rope_delta = self.mrope_state.rope_delta();
        let (output, offset) = self.sequence_state.with_internal(|caches| {
            let mut hidden = self.embed_tokens.forward(input_ids);
            let shape = mlxcel_core::array_shape(&hidden);
            let seq_len = shape[1];
            let cache_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            let position_ids =
                rope_delta.map(|delta| decode_rope_positions(cache_offset, seq_len, delta));
            let mut gdn_states = Vec::new();
            for (layer_idx, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate()
            {
                hidden = layer.forward_with_capture(
                    layer_idx,
                    &hidden,
                    None,
                    cache,
                    position_ids.as_deref(),
                    &mut gdn_states,
                );
            }
            let normalized = self.norm.forward(&hidden);
            let logits = if compact_logits {
                self.project_dflash_verify_logits(&normalized)
            } else {
                self.project_logits(&normalized)
            };
            let offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            (
                Qwen35MtpVerifyOutput {
                    hidden,
                    logits,
                    gdn_states,
                },
                offset,
            )
        });
        self.mrope_state.set_position(offset);
        output
    }

    #[cfg(any(feature = "dflash2", test))]
    /// DFlash2 target verify: run the backbone over `input_ids` exactly like
    /// `forward_mtp_verify` but additionally retain the post-layer hidden
    /// handle at each `target_layer_ids[i]`. `target_layer_ids` is a small
    /// sorted slice (5 ids for the Qwen3.8-27B-DFlash2 checkpoint); a linear
    /// scan per layer is fine. The captured hiddens feed the next draft
    /// round's context buffer (SGLang `DFLASH` captures the same features).
    pub(crate) fn forward_dflash_verify(
        &self,
        input_ids: &MlxArray,
        target_layer_ids: &[usize],
        compact_logits: bool,
    ) -> Qwen35DflashVerifyOutput {
        let input_len = mlxcel_core::array_shape(input_ids)[1];
        let projected = self.sequence_state.with_internal(|caches| {
            caches.first().map(Qwen3NextCache::offset).unwrap_or(0) + input_len
        });
        self.enforce_mtp_cache_bound(projected);
        let rope_delta = self.mrope_state.rope_delta();
        let (output, offset) = self.sequence_state.with_internal(|caches| {
            let mut hidden = self.embed_tokens.forward(input_ids);
            let shape = mlxcel_core::array_shape(&hidden);
            let seq_len = shape[1];
            let cache_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            let position_ids =
                rope_delta.map(|delta| decode_rope_positions(cache_offset, seq_len, delta));
            let mut hidden_by_layer = Vec::with_capacity(target_layer_ids.len());
            let mut gdn_states = Vec::new();
            for (layer_idx, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate()
            {
                hidden = layer.forward_with_capture(
                    layer_idx,
                    &hidden,
                    None,
                    cache,
                    position_ids.as_deref(),
                    &mut gdn_states,
                );
                if target_layer_ids.contains(&layer_idx) {
                    hidden_by_layer.push(mlxcel_core::share(&hidden));
                }
            }
            let normalized = self.norm.forward(&hidden);
            let logits = if compact_logits {
                self.project_dflash_verify_logits(&normalized)
            } else {
                self.project_logits(&normalized)
            };
            let offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            (
                Qwen35DflashVerifyOutput {
                    hidden_by_layer,
                    logits,
                    gdn_states,
                },
                offset,
            )
        });
        self.mrope_state.set_position(offset);
        output
    }

    #[cfg(any(feature = "dflash2", test))]
    /// DFlash2 prefill: chunked backbone forward that also captures the
    /// post-layer hidden state at each `target_layer_ids[i]`, keeping only the
    /// last `hidden_limit` rows per layer (dropping leading rows once over the
    /// limit, as the drafter's sliding window bounds the context the draft
    /// layers can attend to). After all chunks, `hidden_concat` is the
    /// per-layer hiddens concatenated along `-1`; `hidden_offset` is the number
    /// of dropped leading rows.
    pub(crate) fn forward_dflash_prefill(
        &self,
        input_ids: &MlxArray,
        target_layer_ids: &[usize],
        hidden_limit: usize,
    ) -> std::result::Result<Qwen35DflashPrefill, String> {
        self.reset_runtime_state();
        let shape = mlxcel_core::array_shape(input_ids);
        let prompt_len = shape[1];
        if prompt_len == 0 {
            return Err("DFlash2 prefill requires at least one token".to_owned());
        }
        let configured = mlxcel_core::generate::prefill_chunk_len();
        let chunk_len =
            mlxcel_core::generate::effective_prefill_chunk(configured, true, prompt_len as usize)
                .unwrap_or(prompt_len as usize) as i32;
        self.enforce_mtp_cache_bound(prompt_len);
        // One captured hidden buffer per target layer, kept under `hidden_limit`
        // rows.
        let mut layer_hiddens: Vec<Option<UniquePtr<MlxArray>>> =
            (0..target_layer_ids.len()).map(|_| None).collect();
        let mut hidden_offset = 0usize;
        let rope_delta = self.mrope_state.rope_delta();
        let mut first_logits = None;
        let mut start = 0;
        while start < prompt_len {
            let end = (start + chunk_len).min(prompt_len);
            let ids = mlxcel_core::slice(input_ids, &[0, start], &[shape[0], end]);
            let final_hidden = self.sequence_state.with_internal(|caches| {
                let cache_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
                let seq_len = end - start;
                let position_ids =
                    rope_delta.map(|delta| decode_rope_positions(cache_offset, seq_len, delta));
                let mut h = self.embed_tokens.forward(&ids);
                for (layer_idx, (layer, cache)) in
                    self.layers.iter().zip(caches.iter_mut()).enumerate()
                {
                    h = layer.forward(&h, None, cache, position_ids.as_deref());
                    if let Some(capture_idx) = target_layer_ids.iter().position(|&t| t == layer_idx)
                    {
                        // Accumulate this layer's hidden rows (all positions of
                        // this chunk row-wise).
                        let h_rows = mlxcel_core::copy(&h);
                        let current = layer_hiddens[capture_idx].take();
                        let mut combined = match current {
                            Some(prior) => mlxcel_core::concatenate(&prior, &h_rows, 1),
                            None => h_rows,
                        };
                        // Keep at most `hidden_limit` rows; drop leading ones.
                        let combined_shape = mlxcel_core::array_shape(&combined);
                        let rows = combined_shape[1];
                        if rows as usize > hidden_limit {
                            let drop = rows as usize - hidden_limit;
                            hidden_offset += drop;
                            combined = mlxcel_core::slice(
                                &combined,
                                &[0, drop as i32, 0],
                                &[combined_shape[0], rows, combined_shape[2]],
                            );
                        }
                        layer_hiddens[capture_idx] = Some(combined);
                    }
                }
                h
            });
            if end == prompt_len {
                // Last chunk: final-row logits (the first sampled token).
                let hidden_shape = mlxcel_core::array_shape(&final_hidden);
                let last = hidden_shape[1] - 1;
                let last_hidden = mlxcel_core::slice(
                    &final_hidden,
                    &[0, last, 0],
                    &[hidden_shape[0], last + 1, hidden_shape[2]],
                );
                first_logits = Some(self.project_logits(&self.norm.forward(&last_hidden)));
            }
            start = end;
        }

        // Concatenate the per-layer captures along the hidden axis.
        let mut hidden_concat: Option<UniquePtr<MlxArray>> = None;
        for slot in layer_hiddens {
            let h = slot.expect("DFlash2 prefill captured every target layer");
            hidden_concat = Some(match hidden_concat {
                Some(acc) => mlxcel_core::concatenate(&acc, &h, -1),
                None => h,
            });
        }
        let hidden_concat =
            hidden_concat.expect("DFlash2 prefill captures at least one target layer");

        let offset = self
            .sequence_state
            .with_internal(|caches| caches.first().map(Qwen3NextCache::offset).unwrap_or(0));
        self.mrope_state.set_position(offset);
        Ok(Qwen35DflashPrefill {
            hidden_concat,
            first_logits: first_logits.expect("non-empty prefill produces first_logits"),
            hidden_offset,
        })
    }

    pub(crate) fn rollback_mtp_verify(
        &self,
        gdn_states: &[GdnRollbackSnapshot],
        accepted: usize,
        block_size: usize,
        materialize: bool,
    ) -> Qwen35RollbackPlan {
        let plan = self.sequence_state.with_internal(|caches| {
            let verify_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            let plan = rollback_plan(verify_offset, accepted, block_size);
            for cache in caches.iter_mut() {
                if let Qwen3NextCache::Attention(cache) = cache
                    && plan.trim > 0
                {
                    cache.trim(plan.trim);
                }
            }

            for snapshot in gdn_states {
                let Some(Qwen3NextCache::Linear(cache)) = caches.get_mut(snapshot.layer_idx) else {
                    continue;
                };
                let Qwen35AttentionVariant::Linear(layer) =
                    &self.layers[snapshot.layer_idx].attention
                else {
                    continue;
                };
                let replay_len = plan.accepted_block_len;
                let q_shape = mlxcel_core::array_shape(&snapshot.q);
                let k_shape = mlxcel_core::array_shape(&snapshot.k);
                let v_shape = mlxcel_core::array_shape(&snapshot.v);
                let a_shape = mlxcel_core::array_shape(&snapshot.a);
                let b_shape = mlxcel_core::array_shape(&snapshot.b);
                let q = mlxcel_core::slice(
                    &snapshot.q,
                    &[0, 0, 0, 0],
                    &[q_shape[0], replay_len, q_shape[2], q_shape[3]],
                );
                let k = mlxcel_core::slice(
                    &snapshot.k,
                    &[0, 0, 0, 0],
                    &[k_shape[0], replay_len, k_shape[2], k_shape[3]],
                );
                let v = mlxcel_core::slice(
                    &snapshot.v,
                    &[0, 0, 0, 0],
                    &[v_shape[0], replay_len, v_shape[2], v_shape[3]],
                );
                let a = mlxcel_core::slice(
                    &snapshot.a,
                    &[0, 0, 0],
                    &[a_shape[0], replay_len, a_shape[2]],
                );
                let b = mlxcel_core::slice(
                    &snapshot.b,
                    &[0, 0, 0],
                    &[b_shape[0], replay_len, b_shape[2]],
                );
                let (_, replayed_state) = gated_delta_update(
                    (&q, &k, &v),
                    (&a, &b, &layer.a_log, &layer.dt_bias),
                    snapshot.init_state.as_deref(),
                    None,
                );
                cache.state_cache = Some(replayed_state);

                let conv_shape = mlxcel_core::array_shape(&snapshot.conv_input);
                let start = plan.accepted_block_len;
                let end = start + layer.conv_kernel_size as i32 - 1;
                let conv_state = mlxcel_core::slice(
                    &snapshot.conv_input,
                    &[0, start, 0],
                    &[conv_shape[0], end, conv_shape[2]],
                );
                cache.conv_state = Some(mlxcel_core::contiguous(&conv_state, false));
                cache.offset = plan.final_offset;
            }
            if materialize {
                for cache in caches.iter_mut() {
                    cache.materialize_state();
                }
            }
            plan
        });
        self.mrope_state.set_position(plan.final_offset);
        plan
    }

    /// Sever persistent target state from the completed MTP round.
    pub(crate) fn materialize_mtp_cache_state(&self) {
        self.sequence_state.with_internal(|caches| {
            for cache in caches.iter_mut() {
                cache.materialize_state();
            }
        });
    }

    fn parse_config(model_dir: &Path) -> Result<Qwen35Config> {
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let root: Value = serde_json::from_str(&config_text)
            .with_context(|| format!("failed to parse {}", config_path.display()))?;
        let root_object = root
            .as_object()
            .with_context(|| format!("{} must contain a JSON object", config_path.display()))?;
        let model_type = root_object
            .get("model_type")
            .and_then(Value::as_str)
            .with_context(|| format!("{} is missing string model_type", config_path.display()))?;
        ensure!(
            model_type == "qwen3_5",
            "unsupported architecture {model_type:?} in {}; expected \"qwen3_5\"",
            config_path.display()
        );

        let mut text_config = root_object
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| root.clone());
        let text_object = text_config.as_object_mut().with_context(|| {
            format!(
                "text_config in {} must contain a JSON object",
                config_path.display()
            )
        })?;

        for object in [Some(root_object), Some(&*text_object)] {
            let object = object.expect("config object");
            if let Some(value) = object.get("num_experts") {
                let experts = value.as_i64().with_context(|| {
                    format!(
                        "num_experts in {} must be an integer",
                        config_path.display()
                    )
                })?;
                ensure!(
                    experts == 0,
                    "MoE checkpoints are not supported: num_experts={experts} in {}",
                    config_path.display()
                );
            }
            if let Some(kind) = object.get("model_type").and_then(Value::as_str) {
                ensure!(
                    !kind.ends_with("_moe"),
                    "MoE architecture {kind:?} is not supported in {}",
                    config_path.display()
                );
            }
        }

        if let Some(quantization) = root_object
            .get("quantization")
            .or_else(|| root_object.get("quantization_config"))
        {
            validate_quantization(quantization, &config_path)?;
            text_object.insert("quantization".to_string(), quantization.clone());
        } else if let Some(quantization) = text_object
            .get("quantization")
            .or_else(|| text_object.get("quantization_config"))
        {
            validate_quantization(quantization, &config_path)?;
        }

        for key in [
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "rope_parameters",
        ] {
            if let Some(value) = root_object.get(key) {
                text_object.insert(key.to_string(), value.clone());
            }
        }

        let mut config: Qwen35Config = serde_json::from_value(text_config).with_context(|| {
            format!(
                "failed to parse dense text config in {}",
                config_path.display()
            )
        })?;
        ensure!(
            config.model_type == "qwen3_5" || config.model_type == "qwen3_5_text",
            "unsupported dense text architecture {:?} in {}",
            config.model_type,
            config_path.display()
        );
        ensure!(
            config.hidden_size > 0
                && config.intermediate_size > 0
                && config.vocab_size > 0
                && config.num_attention_heads > 0
                && config.num_key_value_heads > 0,
            "dense text dimensions must be greater than zero in {}",
            config_path.display()
        );
        ensure!(
            config.full_attention_interval > 0,
            "full_attention_interval must be greater than zero in {}",
            config_path.display()
        );
        ensure!(
            config.num_hidden_layers > 0,
            "num_hidden_layers must be greater than zero in {}",
            config_path.display()
        );
        config.validate_mtp_metadata(&config_path)?;
        let root_quantization = config.quantization.clone();
        if let Some(vision) = config.vision_config.as_mut()
            && let Some(quantization) = vision
                .quantization_config
                .as_ref()
                .or(root_quantization.as_ref())
        {
            ensure!(
                quantization.mode == "affine",
                "unsupported vision quantization mode {:?} in {}",
                quantization.mode,
                config_path.display()
            );
            vision.quant_group_size = quantization.group_size;
            vision.quant_bits = quantization.bits;
        }
        if let Some(vision) = config.vision_config.as_ref() {
            ensure!(
                vision.deepstack_visual_indexes.is_empty(),
                "Qwen3.5-VL checkpoints with DeepStack visual indexes are not supported"
            );
            ensure!(
                vision.out_hidden_size == config.hidden_size,
                "vision output width must match the text hidden size in {}",
                config_path.display()
            );
            ensure!(
                config.image_token_id.is_some()
                    && config.video_token_id.is_some()
                    && config.vision_start_token_id.is_some(),
                "Qwen3.5-VL config in {} is missing multimodal token identifiers",
                config_path.display()
            );
            if let Some(raw_sections) = config
                .rope_parameters
                .as_ref()
                .and_then(|parameters| parameters.get("mrope_section"))
            {
                ensure!(
                    raw_sections.as_array().is_some_and(|values| {
                        values.len() == 3 && values.iter().all(|value| value.as_i64().is_some())
                    }),
                    "rope_parameters.mrope_section must contain three integers in {}",
                    config_path.display()
                );
            }
            let sections = config.mrope_section();
            let rotary_half =
                ((config.head_dim_resolved() as f32 * config.partial_rotary_factor()) as i32) / 2;
            ensure!(
                sections.iter().all(|&section| section > 0)
                    && sections.iter().sum::<i32>() == rotary_half,
                "rope_parameters.mrope_section is incompatible with the rotary dimension in {}",
                config_path.display()
            );
        }
        Ok(config)
    }

    fn validate_shard_index(model_dir: &Path) -> Result<()> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: Value = serde_json::from_str(&index_text)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .with_context(|| format!("{} is missing object weight_map", index_path.display()))?;
        ensure!(
            !weight_map.is_empty(),
            "{} contains an empty weight_map",
            index_path.display()
        );

        let mut shards = BTreeSet::new();
        for (tensor, shard) in weight_map {
            let shard = shard.as_str().with_context(|| {
                format!(
                    "shard for tensor {tensor:?} in {} must be a string",
                    index_path.display()
                )
            })?;
            let shard_path = Path::new(shard);
            ensure!(
                shard_path.components().count() == 1,
                "invalid shard path {shard:?} in {}",
                index_path.display()
            );
            shards.insert(shard);
        }
        for shard in shards {
            let path = model_dir.join(shard);
            ensure!(
                path.is_file(),
                "missing checkpoint shard {}",
                path.display()
            );
        }
        Ok(())
    }

    pub fn load(model_dir: &Path, kv_cache_mode: KVCacheMode) -> Result<Self> {
        ensure!(
            model_dir.is_dir(),
            "model directory does not exist or is not a directory: {}",
            model_dir.display()
        );
        let config = Self::parse_config(model_dir)?;
        Self::validate_shard_index(model_dir)?;

        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, |name| {
            name.starts_with("language_model.")
                || name.starts_with("model.language_model.")
                || name.starts_with("model.visual.")
                || name.starts_with("visual.")
                || name.starts_with("vision_tower.")
                || name.starts_with("lm_head.")
        })
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to load checkpoint shards from {}",
                model_dir.display()
            )
        })?;
        ensure!(
            !weights.is_empty(),
            "checkpoint {} contains no Qwen3.5 model tensors",
            model_dir.display()
        );
        let weights = sanitize_language_model_weights(weights, &config, model_dir)?;
        let target_cache_mode = mtp_target_cache_mode(weights.mtp.is_some(), kv_cache_mode);
        tracing::info!(
            requested_cache_mode = ?kv_cache_mode,
            effective_target_cache_mode = ?target_cache_mode,
            mtp_fp16_target_cap_tokens = MTP_FP16_TARGET_MAX_TOKENS,
            "selected Qwen3.5 target cache policy"
        );
        let mut model = Self::from_weights(&weights.target, &config, target_cache_mode)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to construct dense model from checkpoint {}",
                    model_dir.display()
                )
            })?;
        model.bounded_mtp_fp16 = weights.mtp.is_some() && kv_cache_mode == KVCacheMode::Turbo4;
        model
            .sequence_state
            .replace_internal(model.make_internal_caches());
        if let Some(mtp_weights) = weights.mtp.as_ref() {
            model.mtp = Some(
                Qwen35MtpDraftModel::from_weights(mtp_weights, &config)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| {
                        format!(
                            "failed to construct bundled MTP head from checkpoint {}",
                            model_dir.display()
                        )
                    })?,
            );
        }
        if let Some(mut vision_config) = config.vision_config.clone() {
            if vision_config.quantization_config.is_none() {
                let (group_size, bits) = config.quant_params("model.visual");
                vision_config.quant_group_size = group_size;
                vision_config.quant_bits = bits;
            }
            model.vision = Some(
                Qwen3VLVisionEncoder::from_weights(&weights.vision, &vision_config, "vision_tower")
                    .map_err(anyhow::Error::msg)
                    .with_context(|| {
                        format!(
                            "failed to construct Qwen3.5 vision encoder from checkpoint {}",
                            model_dir.display()
                        )
                    })?,
            );
        }
        Ok(model)
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn load_specprefill_draft(model_dir: &Path) -> Result<Self> {
        use crate::specprefill::SPECPREFILL_DRAFT_MODEL_IDENTIFIER;

        ensure!(
            model_dir.is_dir(),
            "SpecPrefill draft model directory does not exist or is not a directory: {}",
            model_dir.display()
        );
        // `parse_config` has already required the checkpoint root
        // `model_type == "qwen3_5"`; the returned config is the nested text
        // architecture, whose pinned identifier is `qwen3_5_text`.
        let mut config = Self::parse_config(model_dir)?;
        let quantization = config.quantization.as_ref();
        ensure!(
            config.model_type == "qwen3_5_text"
                && config.num_hidden_layers == 24
                && config.full_attention_interval == 4
                && config.hidden_size == 1_024
                && config.num_attention_heads == 8
                && config.num_key_value_heads == 2
                && config.vocab_size == 248_320
                && config.max_position_embeddings == 262_144
                && quantization.is_some_and(|q| {
                    q.mode == "affine" && q.bits == 8 && q.group_size == 64
                }),
            "checkpoint {} does not match pinned SpecPrefill draft architecture {}",
            model_dir.display(),
            SPECPREFILL_DRAFT_MODEL_IDENTIFIER
        );
        // The pinned checkpoint is a VLM distribution, but SpecPrefill uses
        // only its text scorer. Make the draft-only sanitized contract
        // explicitly language-only; ordinary target loading retains the
        // complete vision and bundled-MTP validation above.
        config.vision_config = None;
        config.image_token_id = None;
        config.video_token_id = None;
        config.vision_start_token_id = None;
        config.mtp_num_hidden_layers = None;
        config.mtp_use_dedicated_embeddings = None;
        Self::validate_shard_index(model_dir)?;
        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, |name| {
            name.starts_with("language_model.")
                || name.starts_with("model.language_model.")
                || name.starts_with("lm_head.")
        })
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to load SpecPrefill draft checkpoint shards from {}",
                model_dir.display()
            )
        })?;
        ensure!(
            !weights.is_empty(),
            "checkpoint {} contains no Qwen3.5 language-model tensors",
            model_dir.display()
        );
        let weights = sanitize_language_model_weights(weights, &config, model_dir)?;
        Self::from_weights(&weights.target, &config, KVCacheMode::Fp16)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to construct SpecPrefill draft model {} from {}",
                    SPECPREFILL_DRAFT_MODEL_IDENTIFIER,
                    model_dir.display()
                )
            })
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn specprefill_draft_prefill(&self, prompt_ids: &[i32]) -> Result<UniquePtr<MlxArray>> {
        ensure!(!prompt_ids.is_empty(), "SpecPrefill draft prompt must not be empty");
        self.reset_runtime_state();
        let configured = mlxcel_core::generate::prefill_chunk_len();
        let chunk_len = mlxcel_core::generate::effective_prefill_chunk(
            configured,
            true,
            prompt_ids.len(),
        )
        .unwrap_or(prompt_ids.len());
        let mut final_logits = None;
        for (chunk_index, chunk) in prompt_ids.chunks(chunk_len).enumerate() {
            let input = mlxcel_core::from_slice_i32(chunk, &[1, chunk.len() as i32]);
            let hidden = self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&input, None, caches, None)
            });
            if (chunk_index + 1) * chunk_len >= prompt_ids.len() {
                let shape = mlxcel_core::array_shape(&hidden);
                let last = shape[1] - 1;
                let last_hidden = mlxcel_core::slice(
                    &hidden,
                    &[0, last, 0],
                    &[shape[0], last + 1, shape[2]],
                );
                final_logits = Some(self.project_logits(&self.norm.forward(&last_hidden)));
            }
        }
        Ok(final_logits.expect("non-empty draft prompt produces logits"))
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn specprefill_draft_lookahead(
        &self,
        token_id: i32,
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        let input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        self.sequence_state.with_internal(|caches| {
            let mut hidden = self.embed_tokens.forward(&input);
            let mut queries = Vec::with_capacity(self.layers.len() / self.config.full_attention_interval);
            for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                let (next, query) = layer.forward_with_query_capture(&hidden, cache);
                hidden = next;
                if let Some(query) = query {
                    queries.push(query);
                }
            }
            (self.project_logits(&self.norm.forward(&hidden)), queries)
        })
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn specprefill_draft_prompt_keys(
        &self,
        prompt_len: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        let prompt_len = i32::try_from(prompt_len).unwrap_or(i32::MAX);
        self.sequence_state.with_internal(|caches| {
            caches
                .iter()
                .filter_map(|cache| match cache {
                    Qwen3NextCache::Attention(cache) => {
                        let keys = cache.keys.as_deref()?;
                        let shape = mlxcel_core::array_shape(keys);
                        Some(mlxcel_core::slice(
                            keys,
                            &[0, 0, 0, 0],
                            &[shape[0], shape[1], prompt_len, shape[3]],
                        ))
                    }
                    Qwen3NextCache::Linear(_) => None,
                })
                .collect()
        })
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn specprefill_sparse_prefill(
        &self,
        prompt_ids: &[i32],
        prefix_reuse: Option<mlxcel_core::generate::PrefixReuse<'_>>,
        dense_prefix_end: usize,
        selected_indices: &[usize],
    ) -> std::result::Result<(UniquePtr<MlxArray>, usize), String> {
        self.reset_runtime_state();
        let mut cached_tokens = 0;
        if let Some(reuse) = prefix_reuse
            && reuse.cached_tokens > 0
            && reuse.cached_tokens <= prompt_ids.len()
            && reuse.snapshot.token_len() == reuse.cached_tokens
            && (reuse.cached_tokens < prompt_ids.len()
                || reuse.snapshot.continuation_logits().is_some())
        {
            self.restore_sequence_state(SequenceId::from_raw(0), reuse.snapshot)?;
            cached_tokens = reuse.cached_tokens;
        }

        if cached_tokens < dense_prefix_end {
            let dense = &prompt_ids[cached_tokens..dense_prefix_end];
            let input = mlxcel_core::from_slice_i32(dense, &[1, dense.len() as i32]);
            self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&input, None, caches, None)
            });
        }

        if selected_indices.is_empty() {
            return Err("SpecPrefill selected no target tokens".to_string());
        }
        let mut final_hidden = None;
        for indices in selected_indices.chunks(SPECPREFILL_TARGET_CHUNK_TOKENS) {
            let sparse_ids = indices
                .iter()
                .map(|&index| prompt_ids[index])
                .collect::<Vec<_>>();
            let one_axis_positions = indices
                .iter()
                .map(|&index| i32::try_from(index).unwrap_or(i32::MAX))
                .collect::<Vec<_>>();
            let positions = one_axis_positions
                .iter()
                .chain(&one_axis_positions)
                .chain(&one_axis_positions)
                .copied()
                .collect::<Vec<_>>();
            let input =
                mlxcel_core::from_slice_i32(&sparse_ids, &[1, sparse_ids.len() as i32]);
            let position_ids =
                mlxcel_core::from_slice_i32(&positions, &[3, 1, sparse_ids.len() as i32]);
            final_hidden = Some(self.sequence_state.with_internal(|caches| {
                self.forward_backbone_with_inputs(&input, None, caches, Some(&position_ids))
            }));
        }
        let hidden = final_hidden.expect("non-empty SpecPrefill selection produces hidden states");
        let shape = mlxcel_core::array_shape(&hidden);
        let last = shape[1] - 1;
        let last_hidden = mlxcel_core::slice(
            &hidden,
            &[0, last, 0],
            &[shape[0], last + 1, shape[2]],
        );
        Ok((
            self.project_logits(&self.norm.forward(&last_hidden)),
            cached_tokens,
        ))
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn specprefill_decode(
        &self,
        token_id: i32,
        logical_position: usize,
    ) -> UniquePtr<MlxArray> {
        let input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        let position = i32::try_from(logical_position).unwrap_or(i32::MAX);
        let position_ids =
            mlxcel_core::from_slice_i32(&[position, position, position], &[3, 1, 1]);
        let hidden = self.sequence_state.with_internal(|caches| {
            self.forward_backbone_with_inputs(&input, None, caches, Some(&position_ids))
        });
        self.project_logits(&self.norm.forward(&hidden))
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        kv_cache_mode: KVCacheMode,
    ) -> std::result::Result<Self, String> {
        let qn_config = config.to_qwen3next_config();
        let (embed_group_size, embed_bits) = config.quant_params("model.embed_tokens");
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "model.embed_tokens",
            embed_group_size,
            embed_bits,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(Qwen35DecoderLayer::from_weights(
                weights, config, &qn_config, layer_idx,
            )?);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|weight| mlxcel_core::copy(weight))
            .ok_or_else(|| "missing required tensor model.norm.weight".to_string())?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let (group_size, bits) = config.quant_params("lm_head");
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };
        let compact_draft_head = lm_head.as_ref().and_then(|head| {
            compact_head(head, config.vocab_size, MTP_DRAFT_PREFIX, MTP_DRAFT_PADDED)
        });
        let compact_dflash_verify_head = lm_head.as_ref().and_then(|head| {
            compact_head(
                head,
                config.vocab_size,
                DFLASH_VERIFY_PREFIX,
                DFLASH_VERIFY_PADDED,
            )
        });
        let internal_caches = layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(Box::new(KVCache::new_with_mode(kv_cache_mode)))
                }
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            compact_draft_head,
            compact_dflash_verify_head,
            config: config.clone(),
            kv_cache_mode,
            mtp: None,
            bounded_mtp_fp16: false,
            vision: None,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            mrope_state: MRopeState::new(),
        })
    }
}

fn validate_quantization(value: &Value, config_path: &Path) -> Result<()> {
    let object = value.as_object().with_context(|| {
        format!(
            "quantization in {} must contain a JSON object",
            config_path.display()
        )
    })?;

    validate_quantization_entry(object, "quantization", config_path)?;
    for (name, value) in object {
        if matches!(name.as_str(), "group_size" | "bits" | "mode") {
            continue;
        }
        let entry = value.as_object().with_context(|| {
            format!(
                "quantization override {name:?} in {} must contain an object",
                config_path.display()
            )
        })?;
        validate_quantization_entry(entry, name, config_path)?;
    }
    Ok(())
}

fn validate_quantization_entry(
    object: &serde_json::Map<String, Value>,
    name: &str,
    config_path: &Path,
) -> Result<()> {
    let group_size = object
        .get("group_size")
        .and_then(Value::as_i64)
        .with_context(|| {
            format!(
                "{name}.group_size in {} must be an integer",
                config_path.display()
            )
        })?;
    let bits = object
        .get("bits")
        .and_then(Value::as_i64)
        .with_context(|| {
            format!(
                "{name}.bits in {} must be an integer",
                config_path.display()
            )
        })?;
    let mode = object
        .get("mode")
        .and_then(Value::as_str)
        .with_context(|| format!("{name}.mode in {} must be a string", config_path.display()))?;
    ensure!(
        group_size > 0 && group_size <= i32::MAX as i64,
        "{name}.group_size in {} must be positive",
        config_path.display()
    );
    ensure!(
        [2, 3, 4, 5, 6, 8].contains(&bits),
        "{name}.bits in {} is unsupported: {bits}",
        config_path.display()
    );
    ensure!(
        mode == "affine",
        "{name}.mode in {} is unsupported: {mode:?}",
        config_path.display()
    );
    Ok(())
}

// Weight Sanitization.

/// Whether a `conv1d.weight` tensor is still in the raw torch layout.
///
/// The gated-delta conv is depthwise (`groups == channels`), so a raw torch
/// weight is `[out, 1, kW]` and the converted MLX weight is `[out, kW, 1]`. A
/// tensor of rank < 3 carries no layout information, so it reads as converted:
/// the norm-shift gate in [`sanitize_weights`] keys on this predicate, and a
/// degenerate tensor must not be able to trigger a shift that transposes nothing.
fn is_raw_conv1d_layout(shape: &[i32]) -> bool {
    shape.len() >= 3 && shape[shape.len() - 1] != 1
}

pub(crate) fn sanitize_weights(mut weights: WeightMap, config: &Qwen35Config) -> WeightMap {
    let raw_conv1d = weights.iter().any(|(name, value)| {
        name.contains("conv1d.weight") && is_raw_conv1d_layout(&mlxcel_core::array_shape(value))
    });

    weights.retain(|name, _| !name.starts_with("mtp.") && !name.contains(".mtp."));
    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }

    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        if key.contains("conv1d.weight") {
            let value = weights
                .get(&key)
                .expect("key collected from the same weight map");
            if is_raw_conv1d_layout(&mlxcel_core::array_shape(value)) {
                weights.insert(key.clone(), mlxcel_core::swap_axes(value, -1, -2));
            }
        }
        if raw_conv1d && norm_suffixes.iter().any(|suffix| key.ends_with(suffix)) {
            let value = weights
                .get(&key)
                .expect("key collected from the same weight map");
            if mlxcel_core::array_shape(value).len() == 1 {
                let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(value));
                weights.insert(key, mlxcel_core::add(value, &one));
            }
        }
    }
    weights
}

struct SanitizedLanguageWeights {
    target: WeightMap,
    vision: WeightMap,
    mtp: Option<WeightMap>,
}

fn sanitize_language_model_weights(
    weights: WeightMap,
    config: &Qwen35Config,
    checkpoint_path: &Path,
) -> Result<SanitizedLanguageWeights> {
    let mut target = WeightMap::new();
    let mut vision = WeightMap::new();
    let mut mtp = WeightMap::new();
    for (name, value) in weights {
        let normalized = if let Some(rest) = name.strip_prefix("model.language_model.") {
            format!("model.{rest}")
        } else if let Some(rest) = name.strip_prefix("language_model.") {
            rest.to_string()
        } else {
            name
        };
        if let Some(rest) = normalized
            .strip_prefix("model.visual.")
            .or_else(|| normalized.strip_prefix("visual."))
            .or_else(|| normalized.strip_prefix("vision_tower."))
        {
            vision.insert(format!("vision_tower.{rest}"), value);
        } else if let Some(rest) = normalized.strip_prefix("model.mtp.") {
            mtp.insert(format!("mtp.{rest}"), value);
        } else if normalized.starts_with("mtp.") {
            mtp.insert(normalized, value);
        } else {
            target.insert(normalized, value);
        }
    }
    ensure!(
        config.vision_config.is_some() || vision.is_empty(),
        "checkpoint {} contains vision tensors but config.json has no vision_config",
        checkpoint_path.display()
    );
    ensure!(
        config.vision_config.is_none() || !vision.is_empty(),
        "checkpoint {} declares vision_config but contains no vision tensors",
        checkpoint_path.display()
    );

    let declared = config.has_mtp_metadata();
    ensure!(
        declared || mtp.is_empty(),
        "checkpoint {} contains language_model.mtp.* tensors but text_config does not declare \
         mtp_num_hidden_layers and mtp_use_dedicated_embeddings",
        checkpoint_path.display()
    );
    ensure!(
        !declared || !mtp.is_empty(),
        "checkpoint {} declares a bundled MTP head but contains no language_model.mtp.* tensors",
        checkpoint_path.display()
    );

    let raw_layout = target.iter().any(|(name, value)| {
        name.contains("conv1d.weight") && is_raw_conv1d_layout(&mlxcel_core::array_shape(value))
    });
    let mtp = if declared {
        validate_mtp_weights(&mtp, checkpoint_path)?;
        Some(sanitize_mtp_weights(mtp, raw_layout))
    } else {
        None
    };
    Ok(SanitizedLanguageWeights {
        target: sanitize_weights(target, config),
        vision,
        mtp,
    })
}

fn validate_mtp_weights(weights: &WeightMap, checkpoint_path: &Path) -> Result<()> {
    const REQUIRED: &[&str] = &[
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.norm.weight",
    ];
    for name in REQUIRED {
        ensure!(
            weights.contains_key(*name),
            "checkpoint {} is missing required tensor language_model.{name}",
            checkpoint_path.display()
        );
    }
    if let Some(name) = weights
        .keys()
        .find(|name| name.starts_with("mtp.embed_tokens."))
    {
        anyhow::bail!(
            "checkpoint {} contains dedicated MTP embedding tensor language_model.{name} \
             while text_config.mtp_use_dedicated_embeddings is false",
            checkpoint_path.display()
        );
    }
    if let Some(name) = weights
        .keys()
        .find(|name| name.starts_with("mtp.layers.") && !name.starts_with("mtp.layers.0."))
    {
        anyhow::bail!(
            "checkpoint {} contains unsupported extra MTP layer tensor language_model.{name}",
            checkpoint_path.display()
        );
    }
    Ok(())
}

fn sanitize_mtp_weights(mut weights: WeightMap, raw_layout: bool) -> WeightMap {
    if !raw_layout {
        return weights;
    }
    const NORM_SUFFIXES: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ];
    let names: Vec<String> = weights.keys().cloned().collect();
    for name in names {
        if !NORM_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
            continue;
        }
        let value = weights
            .get(&name)
            .expect("name collected from the same MTP weight map");
        if mlxcel_core::array_shape(value).len() == 1 {
            let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(value));
            weights.insert(name, mlxcel_core::add(value, &one));
        }
    }
    weights
}

const QWEN35_SNAPSHOT_FAMILY: &str = "qwen3.5-target-v1";

fn snapshot_i32(snapshot: &ModelStateSnapshot, name: &str) -> std::result::Result<i32, String> {
    let value = snapshot
        .tensor(name)
        .ok_or_else(|| format!("Qwen3.5 snapshot is missing {name}"))?;
    if mlxcel_core::array_size(value) != 1 {
        return Err(format!("Qwen3.5 snapshot field {name} must be scalar"));
    }
    Ok(mlxcel_core::item_i32(&mlxcel_core::reshape(value, &[])))
}

fn push_snapshot_i32(snapshot: &mut ModelStateSnapshot, name: &str, value: i32) {
    let array = mlxcel_core::from_slice_i32(&[value], &[1]);
    snapshot.push_tensor(name, &array);
}

fn snapshot_cache_mode(value: i32) -> std::result::Result<KVCacheMode, String> {
    match value {
        0 => Ok(KVCacheMode::Fp16),
        1 => Ok(KVCacheMode::Turbo4),
        _ => Err("Qwen3.5 snapshot contains an unsupported adaptive cache mode".to_string()),
    }
}

fn validate_snapshot_tensor_names(
    snapshot: &ModelStateSnapshot,
    layers: &[Qwen35DecoderLayer],
    kv_cache_mode: KVCacheMode,
    bounded_mtp_fp16: bool,
) -> std::result::Result<(), String> {
    let mut expected = BTreeSet::from([
        "meta.layer_count".to_string(),
        "mrope.position".to_string(),
        "mrope.rope_delta".to_string(),
    ]);
    if bounded_mtp_fp16 {
        expected.insert("meta.adaptive_mtp_cache".to_string());
    }
    if snapshot.tensor("mrope.position_ids").is_some() {
        expected.insert("mrope.position_ids".to_string());
    }
    for (index, layer) in layers.iter().enumerate() {
        expected.insert(format!("layer.{index}.kind"));
        expected.insert(format!("layer.{index}.offset"));
        if layer.is_linear {
            expected.insert(format!("layer.{index}.conv_state"));
            expected.insert(format!("layer.{index}.state_cache"));
            continue;
        }
        let mode = if bounded_mtp_fp16 {
            let name = format!("layer.{index}.mode");
            expected.insert(name.clone());
            snapshot_cache_mode(snapshot_i32(snapshot, &name)?)?
        } else {
            kv_cache_mode
        };
        if mode == KVCacheMode::Turbo4 {
            for suffix in ["k_packed", "k_norms", "v_packed", "v_norms", "v_rescale"] {
                expected.insert(format!("layer.{index}.{suffix}"));
            }
        } else {
            expected.insert(format!("layer.{index}.keys"));
            expected.insert(format!("layer.{index}.values"));
        }
    }
    let actual: BTreeSet<String> = snapshot.tensor_names().map(str::to_owned).collect();
    if actual.len() != snapshot.tensor_count() || actual != expected {
        return Err("Qwen3.5 snapshot tensor layout does not match the loaded model".to_string());
    }
    Ok(())
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen35Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let sequence_length = mlxcel_core::array_shape(input)[1];
        let rope_delta = self.mrope_state.rope_delta();
        let (logits, offset) = self.sequence_state.with_internal(|caches| {
            let cache_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            let position_ids =
                rope_delta.map(|delta| decode_rope_positions(cache_offset, sequence_length, delta));
            let hidden =
                self.forward_backbone_with_inputs(input, None, caches, position_ids.as_deref());
            let logits = self.project_logits(&self.norm.forward(&hidden));
            let offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            (logits, offset)
        });
        self.mrope_state.set_position(offset);
        logits
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let sequence_length = mlxcel_core::array_shape(input_ids)[1];
        let rope_delta = self.mrope_state.rope_delta();
        let (logits, offset) = self.sequence_state.with_internal(|caches| {
            let cache_offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            let position_ids =
                rope_delta.map(|delta| decode_rope_positions(cache_offset, sequence_length, delta));
            let hidden =
                self.forward_backbone_with_inputs(input_ids, None, caches, position_ids.as_deref());
            let shape = mlxcel_core::array_shape(&hidden);
            let position = i32::try_from(last_pos).unwrap_or(i32::MAX);
            assert!(
                position < shape[1],
                "last logits position is outside the input sequence"
            );
            let hidden = mlxcel_core::slice(
                &hidden,
                &[0, position, 0],
                &[shape[0], position + 1, shape[2]],
            );
            let logits = self.project_logits(&self.norm.forward(&hidden));
            let offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            (logits, offset)
        });
        self.mrope_state.set_position(offset);
        logits
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (logits, offset) = self.sequence_state.with_internal(|caches| {
            let hidden = self.mrope_state.with_position_ids(|position_ids| {
                self.forward_backbone_with_inputs(input_ids, input_embeddings, caches, position_ids)
            });
            let logits = self.project_logits(&self.norm.forward(&hidden));
            let offset = caches.first().map(Qwen3NextCache::offset).unwrap_or(0);
            (logits, offset)
        });
        self.mrope_state.set_position(offset);
        logits
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        [self.config.image_token_id, self.config.video_token_id]
            .into_iter()
            .flatten()
            .collect()
    }

    fn prepare_embedding_prefill(&self) -> std::result::Result<(), String> {
        self.mrope_state.activate_prepared()
    }

    fn after_prefill(&self) {
        self.mrope_state.finish_prefill();
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn reset_runtime_state(&self) {
        self.sequence_state
            .replace_internal(self.make_internal_caches());
        self.mrope_state.clear();
        if let Some(mtp) = &self.mtp {
            mtp.reset();
        }
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        _seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        let token_len_i32 = i32::try_from(token_len).ok()?;
        let mut snapshot = ModelStateSnapshot::new(QWEN35_SNAPSHOT_FAMILY, token_len);
        push_snapshot_i32(
            &mut snapshot,
            "meta.layer_count",
            i32::try_from(self.layers.len()).ok()?,
        );
        if self.bounded_mtp_fp16 {
            push_snapshot_i32(&mut snapshot, "meta.adaptive_mtp_cache", 1);
        }
        push_snapshot_i32(&mut snapshot, "mrope.position", self.mrope_state.position());
        push_snapshot_i32(
            &mut snapshot,
            "mrope.rope_delta",
            self.mrope_state.rope_delta().unwrap_or(i32::MIN),
        );
        self.mrope_state.with_position_ids(|position_ids| {
            if let Some(position_ids) = position_ids {
                snapshot.push_tensor("mrope.position_ids", position_ids);
            }
        });

        let complete = self.sequence_state.with_internal(|caches| {
            if caches.len() != self.layers.len() {
                return false;
            }
            for (index, (layer, cache)) in self.layers.iter().zip(caches.iter()).enumerate() {
                if cache.offset() != token_len_i32 {
                    return false;
                }
                push_snapshot_i32(
                    &mut snapshot,
                    &format!("layer.{index}.kind"),
                    if layer.is_linear { 1 } else { 0 },
                );
                push_snapshot_i32(
                    &mut snapshot,
                    &format!("layer.{index}.offset"),
                    cache.offset(),
                );
                match cache {
                    Qwen3NextCache::Attention(cache) => {
                        if self.bounded_mtp_fp16 {
                            let mode = match cache.mode {
                                KVCacheMode::Fp16 => 0,
                                KVCacheMode::Turbo4 => 1,
                                _ => return false,
                            };
                            push_snapshot_i32(&mut snapshot, &format!("layer.{index}.mode"), mode);
                        }
                        if cache.mode == KVCacheMode::Turbo4 {
                            let Some(tensors) = cache.turbo4_snapshot_tensors() else {
                                return false;
                            };
                            snapshot
                                .push_tensor(format!("layer.{index}.k_packed"), tensors.k_packed);
                            snapshot.push_tensor(format!("layer.{index}.k_norms"), tensors.k_norms);
                            snapshot
                                .push_tensor(format!("layer.{index}.v_packed"), tensors.v_packed);
                            snapshot.push_tensor(format!("layer.{index}.v_norms"), tensors.v_norms);
                            snapshot
                                .push_tensor(format!("layer.{index}.v_rescale"), tensors.v_rescale);
                        } else {
                            let (Some(keys), Some(values)) =
                                (cache.keys.as_deref(), cache.values.as_deref())
                            else {
                                return false;
                            };
                            snapshot.push_tensor(format!("layer.{index}.keys"), keys);
                            snapshot.push_tensor(format!("layer.{index}.values"), values);
                        }
                    }
                    Qwen3NextCache::Linear(cache) => {
                        let (Some(conv_state), Some(state_cache)) =
                            (cache.conv_state.as_deref(), cache.state_cache.as_deref())
                        else {
                            return false;
                        };
                        snapshot.push_tensor(format!("layer.{index}.conv_state"), conv_state);
                        snapshot.push_tensor(format!("layer.{index}.state_cache"), state_cache);
                    }
                }
            }
            true
        });
        complete.then_some(snapshot)
    }

    fn restore_sequence_state(
        &self,
        _seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> std::result::Result<(), String> {
        if snapshot.family() != QWEN35_SNAPSHOT_FAMILY {
            return Err(format!(
                "Qwen3.5 snapshot family mismatch: expected {QWEN35_SNAPSHOT_FAMILY}, got {}",
                snapshot.family()
            ));
        }
        let token_len = i32::try_from(snapshot.token_len())
            .map_err(|_| "Qwen3.5 snapshot token length exceeds i32".to_string())?;
        if snapshot_i32(snapshot, "meta.layer_count")?
            != i32::try_from(self.layers.len()).unwrap_or(i32::MAX)
        {
            return Err("Qwen3.5 snapshot layer count does not match the loaded model".to_string());
        }
        if self.bounded_mtp_fp16 && snapshot_i32(snapshot, "meta.adaptive_mtp_cache")? != 1 {
            return Err("Qwen3.5 snapshot adaptive cache policy mismatch".to_string());
        }
        validate_snapshot_tensor_names(
            snapshot,
            &self.layers,
            self.kv_cache_mode,
            self.bounded_mtp_fp16,
        )?;

        let mut restored = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let expected_kind = if layer.is_linear { 1 } else { 0 };
            if snapshot_i32(snapshot, &format!("layer.{index}.kind"))? != expected_kind {
                return Err(format!(
                    "Qwen3.5 snapshot layer {index} cache variant mismatch"
                ));
            }
            if snapshot_i32(snapshot, &format!("layer.{index}.offset"))? != token_len {
                return Err(format!("Qwen3.5 snapshot layer {index} offset mismatch"));
            }
            if layer.is_linear {
                let conv_state = snapshot
                    .tensor(&format!("layer.{index}.conv_state"))
                    .ok_or_else(|| {
                        format!("Qwen3.5 snapshot is missing layer {index} conv state")
                    })?;
                let state_cache = snapshot
                    .tensor(&format!("layer.{index}.state_cache"))
                    .ok_or_else(|| {
                        format!("Qwen3.5 snapshot is missing layer {index} recurrent state")
                    })?;
                if mlxcel_core::array_shape(conv_state).len() != 3
                    || mlxcel_core::array_shape(state_cache).len() != 4
                {
                    return Err(format!(
                        "Qwen3.5 snapshot layer {index} linear cache layout mismatch"
                    ));
                }
                restored.push(Qwen3NextCache::Linear(GatedDeltaCache {
                    conv_state: Some(mlxcel_core::copy(conv_state)),
                    state_cache: Some(mlxcel_core::copy(state_cache)),
                    offset: token_len,
                }));
            } else if (if self.bounded_mtp_fp16 {
                snapshot_cache_mode(snapshot_i32(snapshot, &format!("layer.{index}.mode"))?)?
            } else {
                self.kv_cache_mode
            }) == KVCacheMode::Turbo4
            {
                let tensor = |suffix: &str| {
                    snapshot
                        .tensor(&format!("layer.{index}.{suffix}"))
                        .map(mlxcel_core::copy)
                        .ok_or_else(|| {
                            format!("Qwen3.5 snapshot is missing layer {index} {suffix}")
                        })
                };
                let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4);
                cache.restore_turbo4_snapshot(
                    token_len,
                    tensor("k_packed")?,
                    tensor("k_norms")?,
                    tensor("v_packed")?,
                    tensor("v_norms")?,
                    tensor("v_rescale")?,
                )?;
                restored.push(Qwen3NextCache::Attention(Box::new(cache)));
            } else {
                let keys = snapshot
                    .tensor(&format!("layer.{index}.keys"))
                    .ok_or_else(|| format!("Qwen3.5 snapshot is missing layer {index} keys"))?;
                let values = snapshot
                    .tensor(&format!("layer.{index}.values"))
                    .ok_or_else(|| format!("Qwen3.5 snapshot is missing layer {index} values"))?;
                let key_shape = mlxcel_core::array_shape(keys);
                let value_shape = mlxcel_core::array_shape(values);
                if key_shape.len() != 4
                    || key_shape != value_shape
                    || key_shape[0] != 1
                    || key_shape[2] < token_len
                {
                    return Err(format!(
                        "Qwen3.5 snapshot layer {index} attention cache layout mismatch"
                    ));
                }
                let mode = if self.bounded_mtp_fp16 {
                    snapshot_cache_mode(snapshot_i32(snapshot, &format!("layer.{index}.mode"))?)?
                } else {
                    self.kv_cache_mode
                };
                let mut cache = KVCache::new_with_mode(mode);
                cache.keys = Some(mlxcel_core::copy(keys));
                cache.values = Some(mlxcel_core::copy(values));
                cache.offset = token_len;
                if self.bounded_mtp_fp16 && index >= 48 {
                    cache.enable_fp16_v_quantization_on_write();
                }
                restored.push(Qwen3NextCache::Attention(Box::new(cache)));
            }
        }

        let position = snapshot_i32(snapshot, "mrope.position")?;
        if position != token_len {
            return Err("Qwen3.5 snapshot MRoPE position does not match token length".to_string());
        }
        let rope_delta = match snapshot_i32(snapshot, "mrope.rope_delta")? {
            i32::MIN => None,
            value => Some(value),
        };
        self.sequence_state.replace_internal(restored);
        self.mrope_state
            .restore(position, snapshot.tensor("mrope.position_ids"), rope_delta);
        Ok(())
    }

    fn snapshot_truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        snapshot.family() == QWEN35_SNAPSHOT_FAMILY && target_len == snapshot.token_len()
    }

    fn restore_sequence_state_truncated(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> std::result::Result<(), String> {
        if !self.snapshot_truncatable_to(snapshot, target_len) {
            return Err(
                "Qwen3.5 recurrent snapshots cannot be truncated to an earlier token".to_string(),
            );
        }
        self.restore_sequence_state(seq_id, snapshot)
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        vec![248046, 248044]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn dense_config(quantization: Option<Value>) -> Qwen35Config {
        let mut value = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 16,
            "num_hidden_layers": 8,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "full_attention_interval": 4,
            "vocab_size": 100
        });
        if let Some(quantization) = quantization {
            value
                .as_object_mut()
                .expect("test config object")
                .insert("quantization".to_string(), quantization);
        }
        serde_json::from_value(value).expect("minimal dense config")
    }

    fn mtp_config() -> Qwen35Config {
        let mut config = dense_config(None);
        config.mtp_num_hidden_layers = Some(1);
        config.mtp_use_dedicated_embeddings = Some(false);
        config
    }

    #[test]
    fn bundled_mtp_uses_bounded_native_target_cache() {
        assert_eq!(
            mtp_target_cache_mode(true, KVCacheMode::Turbo4),
            KVCacheMode::Fp16
        );
        assert_eq!(
            mtp_target_cache_mode(false, KVCacheMode::Turbo4),
            KVCacheMode::Turbo4
        );
        assert_eq!(
            mtp_target_cache_mode(true, KVCacheMode::Int8),
            KVCacheMode::Int8
        );
    }

    #[test]
    fn qwen35_mtp_fp16_cap_is_four_gibibytes() {
        let bytes = 16_u64 * 4 * MTP_FP16_TARGET_MAX_TOKENS as u64 * 256 * 2 * 2;
        assert_eq!(bytes, 4_u64 << 30);
    }

    #[test]
    fn adaptive_snapshot_mode_round_trips_fp16_and_turbo4() {
        for (encoded, expected) in [(0, KVCacheMode::Fp16), (1, KVCacheMode::Turbo4)] {
            let mut snapshot = ModelStateSnapshot::new("test", 32_768);
            push_snapshot_i32(&mut snapshot, "layer.0.mode", encoded);
            let restored =
                snapshot_cache_mode(snapshot_i32(&snapshot, "layer.0.mode").expect("mode field"))
                    .expect("supported adaptive cache mode");
            assert_eq!(restored, expected);
        }
        assert!(snapshot_cache_mode(2).is_err());
    }

    fn insert_required_mtp_weights(weights: &mut WeightMap) {
        for name in [
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.norm.weight",
        ] {
            weights.insert(
                format!("language_model.{name}"),
                mlxcel_core::from_slice_f32(&[0.0; 4], &[4]),
            );
        }
    }

    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "qw-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn outer_qwen35_text_config_and_mixed_quantization_are_accepted() {
        let fixture = TestDir::new("outer-config");
        std::fs::write(
            fixture.0.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 16,
                    "num_hidden_layers": 8,
                    "intermediate_size": 32,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "full_attention_interval": 4,
                    "vocab_size": 100
                },
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }
                }
            }))
            .expect("serialize config"),
        )
        .expect("write config");

        let config = Qwen35Model::parse_config(&fixture.0).expect("parse dense outer config");
        assert_eq!(
            config.quant_params("model.layers.0.linear_attn.in_proj_qkv"),
            (64, 5)
        );
        assert_eq!(
            config.quant_params("model.layers.1.self_attn.q_proj"),
            (64, 4)
        );
    }

    #[test]
    fn unsupported_architecture_and_moe_are_rejected() {
        for (name, config, expected) in [
            (
                "unsupported",
                serde_json::json!({"model_type": "llama"}),
                "unsupported architecture",
            ),
            (
                "moe",
                serde_json::json!({
                    "model_type": "qwen3_5",
                    "num_experts": 8,
                    "text_config": {"model_type": "qwen3_5_text"}
                }),
                "MoE checkpoints are not supported",
            ),
        ] {
            let fixture = TestDir::new(name);
            std::fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).expect("serialize config"),
            )
            .expect("write config");
            let error = Qwen35Model::parse_config(&fixture.0)
                .expect_err("invalid architecture must fail")
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn missing_checkpoint_shard_is_path_specific() {
        let fixture = TestDir::new("missing-shard");
        std::fs::write(
            fixture.0.join("model.safetensors.index.json"),
            br#"{"weight_map":{"language_model.model.embed_tokens.weight":"missing.safetensors"}}"#,
        )
        .expect("write index");

        let error = Qwen35Model::validate_shard_index(&fixture.0)
            .expect_err("missing shard must fail")
            .to_string();
        assert!(error.contains("missing.safetensors"), "{error}");
        assert!(error.contains("missing checkpoint shard"), "{error}");
    }

    #[test]
    fn layer_selection_uses_every_fourth_layer_for_full_attention() {
        let config = dense_config(None);
        let full_attention_layers: Vec<_> = (0..config.num_hidden_layers)
            .filter(|&layer| !config.is_linear_layer(layer))
            .collect();
        assert_eq!(full_attention_layers, vec![3, 7]);
    }

    #[test]
    fn mtp_metadata_accepts_absent_and_valid_and_rejects_incompatible_forms() {
        let path = Path::new("/checkpoint/config.json");
        dense_config(None)
            .validate_mtp_metadata(path)
            .expect("absent MTP metadata is valid");
        mtp_config()
            .validate_mtp_metadata(path)
            .expect("one shared-embedding MTP layer is valid");

        for (layers, dedicated, expected) in [
            (Some(1), None, "incomplete MTP metadata"),
            (Some(2), Some(false), "must be exactly 1"),
            (Some(1), Some(true), "must be false"),
        ] {
            let mut config = dense_config(None);
            config.mtp_num_hidden_layers = layers;
            config.mtp_use_dedicated_embeddings = dedicated;
            let error = config
                .validate_mtp_metadata(path)
                .expect_err("incompatible MTP metadata must fail")
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn declared_mtp_requires_the_complete_exact_tensor_set() {
        let config = mtp_config();
        let error =
            sanitize_language_model_weights(WeightMap::new(), &config, Path::new("/checkpoint"))
                .err()
                .expect("declared MTP without tensors must fail")
                .to_string();
        assert!(error.contains("/checkpoint"), "{error}");
        assert!(
            error.contains("contains no language_model.mtp.* tensors"),
            "{error}"
        );

        let mut incomplete = WeightMap::new();
        incomplete.insert(
            "language_model.mtp.fc.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 4], &[4]),
        );
        let error = sanitize_language_model_weights(incomplete, &config, Path::new("/checkpoint"))
            .err()
            .expect("incomplete MTP tensors must fail")
            .to_string();
        assert!(
            error.contains("language_model.mtp.pre_fc_norm_embedding.weight"),
            "{error}"
        );
    }

    #[test]
    fn sanitizer_splits_mtp_and_converts_its_norms_only_for_raw_layout() {
        let config = mtp_config();
        let mut weights = WeightMap::new();
        weights.insert(
            "language_model.model.layers.0.linear_attn.conv1d.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 12], &[4, 1, 3]),
        );

        let mut undeclared = WeightMap::new();
        undeclared.insert(
            "language_model.mtp.fc.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 4], &[4]),
        );
        let error = sanitize_language_model_weights(
            undeclared,
            &dense_config(None),
            Path::new("/undeclared"),
        )
        .err()
        .expect("MTP tensors without metadata must fail")
        .to_string();
        assert!(error.contains("/undeclared"), "{error}");
        assert!(error.contains("does not declare"), "{error}");
        weights.insert(
            "language_model.model.layers.0.linear_attn.in_proj_qkv.scales".to_string(),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
        insert_required_mtp_weights(&mut weights);

        let sanitized = sanitize_language_model_weights(weights, &config, Path::new("/checkpoint"))
            .expect("valid bundled MTP weights");
        let mtp = sanitized.mtp.expect("retained MTP partition");
        assert!(mtp.contains_key("mtp.fc.weight"));
        let expected = mlxcel_core::from_slice_f32(&[1.0; 4], &[4]);
        let shifted = mlxcel_core::allclose(
            mtp.get("mtp.pre_fc_norm_hidden.weight")
                .expect("MTP hidden norm"),
            &expected,
            0.0,
            0.0,
        );
        mlxcel_core::eval(&shifted);
        assert!(mlxcel_core::item_bool(&shifted));

        let mut converted = WeightMap::new();
        converted.insert(
            "mtp.norm.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 4], &[4]),
        );
        let converted = sanitize_mtp_weights(converted, false);
        let expected_zero = mlxcel_core::from_slice_f32(&[0.0; 4], &[4]);
        let unchanged = mlxcel_core::allclose(
            converted.get("mtp.norm.weight").expect("MTP final norm"),
            &expected_zero,
            0.0,
            0.0,
        );
        mlxcel_core::eval(&unchanged);
        assert!(mlxcel_core::item_bool(&unchanged));
    }

    #[test]
    fn raw_conv1d_is_transposed_and_norms_are_shifted_once() {
        let config = dense_config(None);
        let mut weights = WeightMap::new();
        weights.insert(
            "model.layers.0.linear_attn.conv1d.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 12], &[4, 1, 3]),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 4], &[4]),
        );

        let sanitized = sanitize_weights(weights, &config);
        assert_eq!(
            mlxcel_core::array_shape(
                sanitized
                    .get("model.layers.0.linear_attn.conv1d.weight")
                    .expect("conv weight")
            ),
            vec![4, 3, 1]
        );
        let expected = mlxcel_core::from_slice_f32(&[1.0; 4], &[4]);
        let close = mlxcel_core::allclose(
            sanitized
                .get("model.layers.0.input_layernorm.weight")
                .expect("norm weight"),
            &expected,
            0.0,
            0.0,
        );
        mlxcel_core::eval(&close);
        assert!(mlxcel_core::item_bool(&close));

        let sanitized_again = sanitize_weights(sanitized, &config);
        let close = mlxcel_core::allclose(
            sanitized_again
                .get("model.layers.0.input_layernorm.weight")
                .expect("norm weight"),
            &expected,
            0.0,
            0.0,
        );
        mlxcel_core::eval(&close);
        assert!(mlxcel_core::item_bool(&close));
    }

    #[test]
    fn sanitizer_partitions_visual_weights_without_dropping_text_weights() {
        let mut config = dense_config(None);
        config.vision_config = Some(
            serde_json::from_value(serde_json::json!({
                "hidden_size": 8,
                "out_hidden_size": 16,
                "deepstack_visual_indexes": []
            }))
            .expect("vision config"),
        );
        let mut weights = WeightMap::new();
        weights.insert(
            "model.visual.patch_embed.proj.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
        weights.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            mlxcel_core::from_slice_f32(&[2.0], &[1]),
        );
        let sanitized = sanitize_language_model_weights(weights, &config, Path::new("/checkpoint"))
            .expect("partition weights");
        assert!(
            sanitized
                .vision
                .contains_key("vision_tower.patch_embed.proj.weight")
        );
        assert!(sanitized.target.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    #[ignore = "requires the real dense Qwen3.5 checkpoint at QW_MODEL_PATH or the default model cache path"]
    fn restored_mixed_target_snapshot_matches_uninterrupted_next_token_and_text() {
        let model_dir = crate::resolve_model_path(None)
            .expect("QW_MODEL_PATH or the default model cache path must hold a real checkpoint");
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .expect("load tokenizer");
        let model = Qwen35Model::load(&model_dir, KVCacheMode::Fp16).expect("load Qwen3.5 model");
        let prompt = tokenizer
            .encode("Snapshot restore invariant", true)
            .expect("encode prompt");
        let prompt_ids: Vec<i32> = prompt.get_ids().iter().map(|&id| id as i32).collect();
        assert!(!prompt_ids.is_empty());

        model.reset_runtime_state();
        let prompt_array = mlxcel_core::from_slice_i32(&prompt_ids, &[1, prompt_ids.len() as i32]);
        let prompt_logits =
            model.forward_last_logits(&prompt_array, &mut [], None, prompt_ids.len() - 1);
        let first = mlxcel_core::argmax_last_axis(&prompt_logits);
        mlxcel_core::eval(&first);
        let first_id = mlxcel_core::item_i32(&first);
        let snapshot = model
            .snapshot_sequence_state(SequenceId::from_raw(7), prompt_ids.len())
            .expect("capture complete mixed-state snapshot");

        let first_array = mlxcel_core::from_slice_i32(&[first_id], &[1, 1]);
        let uninterrupted_logits = model.forward_last_logits(&first_array, &mut [], None, 0);
        let uninterrupted = mlxcel_core::argmax_last_axis(&uninterrupted_logits);
        mlxcel_core::eval(&uninterrupted);
        let uninterrupted_id = mlxcel_core::item_i32(&uninterrupted);

        model.reset_runtime_state();
        model
            .restore_sequence_state(SequenceId::from_raw(9), &snapshot)
            .expect("restore complete mixed-state snapshot");
        let restored_logits = model.forward_last_logits(&first_array, &mut [], None, 0);
        let restored = mlxcel_core::argmax_last_axis(&restored_logits);
        mlxcel_core::eval(&restored);
        let restored_id = mlxcel_core::item_i32(&restored);

        assert_eq!(restored_id, uninterrupted_id);
        let uninterrupted_text = tokenizer
            .decode(&[uninterrupted_id as u32], false)
            .expect("decode uninterrupted token");
        let restored_text = tokenizer
            .decode(&[restored_id as u32], false)
            .expect("decode restored token");
        assert_eq!(restored_text, uninterrupted_text);
        assert!(model.layers.iter().any(|layer| layer.is_linear));
        assert!(model.layers.iter().any(|layer| !layer.is_linear));
    }
}
