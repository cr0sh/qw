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

//! Dense Qwen4 hybrid text model.
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen4.py

use crate::gated_delta::{
    GatedDeltaCache, RMSNormGated, gated_delta_update, scaled_fast_rms_norm_no_weight,
};
use crate::model_owned::ModelOwnedSequenceState;
use crate::ngram_offload::{NGramTable, NGramTableSpec};
use crate::qwen_position::decode_rope_positions;
use crate::qwen_rope_state::RopeState;
use crate::qwen4_attention::{
    Mlp, Quantization, Qwen4Attention, Qwen4AttentionConfig, Qwen4LayerCache, Qwen4PrefillPolicy,
};
use crate::qwen4_mtp::Qwen4MtpDraftModel;
use anyhow::{Context, Result, ensure};
use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::{KVCache, MoESwitch, QuantizedWeight, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::silu;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use serde_json::Value;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

const MTP_DRAFT_PREFIX: i32 = 65_536;
const MTP_DRAFT_PADDED: i32 = 65_568;
const MTP_VERIFY_PREFIX: i32 = 80_896;
const MTP_VERIFY_PADDED: i32 = 80_928;
const DRAFT_CONTROL_START: i32 = 248_044;
const DRAFT_CONTROL_END: i32 = 248_070;

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
pub struct Qwen4Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
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
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_hc_count")]
    pub hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    pub hc_lowrank: usize,
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: usize,
    #[serde(default = "default_ple_conv_kernel_size")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default = "default_ngram_vocab_size_base")]
    pub ngram_vocab_size_base: usize,
    #[serde(default = "default_ngram_divisor")]
    pub make_ngram_vocab_size_divisible_by: usize,
    #[serde(default = "default_split_ngram_parts")]
    pub split_ngram_parts: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub eos_token_id: Option<i32>,
    #[serde(default = "default_output_gate_type")]
    pub output_gate_type: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default, alias = "quantization_config")]
    pub quantization: Option<Quantization>,
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
fn default_hc_count() -> usize {
    1
}
fn default_hc_lowrank() -> usize {
    320
}
fn default_ple_conv_kernel_size() -> usize {
    4
}
fn default_ngram_size() -> usize {
    3
}
fn default_heads_per_ngram() -> usize {
    8
}
fn default_ngram_vocab_size_base() -> usize {
    20_000_000
}
fn default_ngram_divisor() -> usize {
    128
}
fn default_split_ngram_parts() -> usize {
    128
}
fn default_seed() -> u64 {
    1234
}
fn default_output_gate_type() -> String {
    "silu".to_owned()
}

impl Qwen4Config {
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
        self.layer_types.get(layer_idx).map_or_else(
            || !(layer_idx + 1).is_multiple_of(self.full_attention_interval),
            |kind| kind == "linear_attention",
        )
    }

    /// Convert to Qwen4AttentionConfig for reusing shared components
    pub fn to_qwen4_attention_config(&self) -> Qwen4AttentionConfig {
        Qwen4AttentionConfig {
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim_resolved(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta(),
            partial_rotary_factor: self.partial_rotary_factor(),
            quantization: self.quantization.clone(),
            mrope_section: self.mrope_section(),
            indexer_n_heads: 4,
            indexer_kv_heads: 1,
            indexer_head_dim: 128,
            indexer_budget: 2_048,
            indexer_compress_ratio: 4,
        }
    }
}

pub(crate) struct GdnRollbackSnapshot {
    layer_idx: usize,
    batch: i32,
    q: UniquePtr<MlxArray>,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
    a: UniquePtr<MlxArray>,
    b: UniquePtr<MlxArray>,
    init_state: Option<UniquePtr<MlxArray>>,
    conv_input: UniquePtr<MlxArray>,
    ple_inputs: Option<UniquePtr<MlxArray>>,
    ple_input_ids: Option<UniquePtr<MlxArray>>,
    ple_conv_state: Option<UniquePtr<MlxArray>>,
    ple_token_history: Option<Vec<i32>>,
}

pub(crate) struct Qwen4MtpPrefill {
    pub(crate) hidden: UniquePtr<MlxArray>,
    pub(crate) first_logits: UniquePtr<MlxArray>,
}

pub(crate) struct Qwen4MtpVerifyOutput {
    pub(crate) hidden: UniquePtr<MlxArray>,
    pub(crate) logits: UniquePtr<MlxArray>,
    pub(crate) gdn_states: Vec<GdnRollbackSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen4RollbackPlan {
    pub(crate) accepted_block_len: i32,
    pub(crate) trim: i32,
    pub(crate) final_offset: i32,
}

fn rollback_plan_for_retained_inputs(
    verify_offset: i32,
    retained_inputs: usize,
    block_size: usize,
) -> Qwen4RollbackPlan {
    let accepted_block_len = i32::try_from(retained_inputs).unwrap_or(i32::MAX);
    let block_size = i32::try_from(block_size).unwrap_or(i32::MAX);
    let trim = (block_size - accepted_block_len).max(0);
    Qwen4RollbackPlan {
        accepted_block_len,
        trim,
        final_offset: verify_offset - trim,
    }
}

#[cfg(test)]
pub(crate) fn rollback_plan(
    verify_offset: i32,
    accepted: usize,
    block_size: usize,
) -> Qwen4RollbackPlan {
    rollback_plan_for_retained_inputs(verify_offset, accepted.saturating_add(1), block_size)
}

enum Qwen4GatedAuxProjections {
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
) -> Qwen4GatedAuxProjections {
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
        Qwen4GatedAuxProjections::Separate { z, b, a },
        Qwen4GatedAuxProjections::Fused,
    )
}

// GatedDeltaNet - Qwen4 variant with separately stored projections.
/// Fuses compatible z, beta, and decay projections at load time.
#[allow(dead_code)]
pub(crate) struct Qwen4GatedDeltaNet {
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
    aux_projections: Qwen4GatedAuxProjections,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    norm: RMSNormGated,
    out_proj: UnifiedLinear,
}

const QWEN4_PREFILL_MICROCHUNK_TOKENS: i32 = 128;

impl Qwen4GatedDeltaNet {
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
        let sequence = shape[1];
        if sequence <= QWEN4_PREFILL_MICROCHUNK_TOKENS || snapshot.is_some() {
            return self.forward_hidden_chunk(inputs, mask, cache, snapshot, true);
        }

        let mut outputs = Vec::with_capacity(
            ((sequence + QWEN4_PREFILL_MICROCHUNK_TOKENS - 1)
                / QWEN4_PREFILL_MICROCHUNK_TOKENS) as usize,
        );
        let mut start = 0;
        while start < sequence {
            let stop = (start + QWEN4_PREFILL_MICROCHUNK_TOKENS).min(sequence);
            let input_chunk =
                mlxcel_core::slice(inputs, &[0, start, 0], &[shape[0], stop, shape[2]]);
            let mask_chunk =
                mask.map(|value| mlxcel_core::slice(value, &[0, start], &[shape[0], stop]));
            outputs.push(self.forward_hidden_chunk(
                &input_chunk,
                mask_chunk.as_deref(),
                cache.as_deref_mut(),
                None,
                false,
            ));
            start = stop;
        }
        if let Some(cache) = cache {
            cache.advance(sequence);
        }
        mlxcel_core::concatenate_owned(&outputs, 1)
    }

    fn forward_hidden_chunk(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
        snapshot: Option<(usize, &mut Vec<GdnRollbackSnapshot>)>,
        advance_cache: bool,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        let effective_mask = mask;

        let qkv = self.in_proj_qkv.forward(inputs);
        let (z, b_proj, a) = match &self.aux_projections {
            Qwen4GatedAuxProjections::Separate { z, b, a } => {
                (z.forward(inputs), b.forward(inputs), a.forward(inputs))
            }
            Qwen4GatedAuxProjections::Fused(projection) => {
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
                        Some(mlxcel_core::share(s))
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
                    Some(mlxcel_core::share(s))
                }
            })
        });

        // Apply RMS norm with scaling (same as Qwen4). Reference mlx-lm
        // keeps this on mx.fast.rms_norm rather than expanding it into
        // primitive ops.
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        if let Some((layer_idx, snapshots)) = snapshot {
            snapshots.push(GdnRollbackSnapshot {
                layer_idx,
                batch: b,
                q: mlxcel_core::share(&q),
                k: mlxcel_core::share(&k),
                v: mlxcel_core::share(&v),
                a: mlxcel_core::share(&a),
                b: mlxcel_core::share(&b_proj),
                init_state: state.as_ref().map(|value| mlxcel_core::share(value)),
                conv_input: mlxcel_core::share(&conv_input),
                ple_inputs: None,
                ple_input_ids: None,
                ple_conv_state: None,
                ple_token_history: None,
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
            if advance_cache {
                c.advance(s);
            }
        }

        // Apply norm with gating
        let out = self.norm.forward(&out, Some(&z));
        mlxcel_core::reshape(&out, &[b, s, -1])
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
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

        // Qwen4 uses separate projections instead of combined projections.
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

        let norm = if config.output_gate_type == "sigmoid" {
            RMSNormGated::new_sigmoid(norm_weight, config.rms_norm_eps)
        } else {
            RMSNormGated::new(norm_weight, config.rms_norm_eps)
        };
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
            norm,
            out_proj,
        })
    }
}

pub(crate) struct Qwen4RmsNorm {
    adjusted_weight: UniquePtr<MlxArray>,
    eps: f32,
    group_size: Option<usize>,
}

impl Qwen4RmsNorm {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        name: &str,
        eps: f32,
        group_size: Option<usize>,
    ) -> Result<Self, String> {
        let weight = weights
            .get(name)
            .ok_or_else(|| format!("missing required tensor {name}"))?;
        let shape = mlxcel_core::array_shape(weight);
        if let Some(group_size) = group_size {
            let width = *shape
                .last()
                .ok_or_else(|| format!("invalid scalar Qwen4 RMSNorm tensor {name}"))?
                as usize;
            if !width.is_multiple_of(group_size) {
                return Err(format!(
                    "Qwen4 RMSNorm tensor {name} width {width} is not divisible by {group_size}"
                ));
            }
        }
        let ones = mlxcel_core::ones(&shape, mlxcel_core::array_dtype(weight));
        Ok(Self {
            adjusted_weight: mlxcel_core::add(weight, &ones),
            eps,
            group_size,
        })
    }

    pub(crate) fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let Some(group_size) = self.group_size else {
            return mlxcel_core::fast_rms_norm(input, &self.adjusted_weight, self.eps);
        };
        mlxcel_core::compiled_group_rms_norm(
            input,
            &self.adjusted_weight,
            group_size as i32,
            self.eps,
        )
    }
}

pub(crate) struct Qwen4HyperConnection {
    hc_count: usize,
    norm: Qwen4RmsNorm,
    input_mix_weight_down: UnifiedLinear,
    input_mix_weight_up: UnifiedLinear,
    block_inject_weight: Option<UnifiedLinear>,
}

impl Qwen4HyperConnection {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
        prefix: &str,
        with_injection: bool,
    ) -> Result<Self, String> {
        let norm = Qwen4RmsNorm::from_weights(
            weights,
            &format!("{prefix}.hc_norm.weight"),
            config.rms_norm_eps,
            Some(config.hidden_size),
        )?;
        let down_prefix = format!("{prefix}.input_mix_weight_down");
        let up_prefix = format!("{prefix}.input_mix_weight_up");
        let (down_group, down_bits) = config.quant_params(&down_prefix);
        let (up_group, up_bits) = config.quant_params(&up_prefix);
        let input_mix_weight_down =
            UnifiedLinear::from_weights(weights, &down_prefix, down_group, down_bits)?;
        let input_mix_weight_up =
            UnifiedLinear::from_weights(weights, &up_prefix, up_group, up_bits)?;
        let block_inject_weight = if with_injection {
            let block_prefix = format!("{prefix}.block_inject_weight");
            let (group, bits) = config.quant_params(&block_prefix);
            Some(UnifiedLinear::from_weights(
                weights,
                &block_prefix,
                group,
                bits,
            )?)
        } else {
            None
        };
        Ok(Self {
            hc_count: config.hc_count,
            norm,
            input_mix_weight_down,
            input_mix_weight_up,
            block_inject_weight,
        })
    }

    fn mix_from_normed(
        &self,
        normed: &MlxArray,
        _hyper_input_shape: &[i32],
    ) -> UniquePtr<MlxArray> {
        let down = self.input_mix_weight_down.forward(normed);
        let down = mlxcel_core::compiled_scaled_silu(&down, 1.0 / self.hc_count as f32);
        let mix_logits = self.input_mix_weight_up.forward(&down);
        mlxcel_core::compiled_hyper_mix(normed, &mix_logits, self.hc_count as i32)
    }

    fn mix(&self, hyper_input: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.norm.forward(hyper_input);
        self.mix_from_normed(&normed, &mlxcel_core::array_shape(hyper_input))
    }

    fn forward(
        &self,
        hyper_input: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let normed = self.norm.forward(hyper_input);
        let mixed = self.mix_from_normed(&normed, &mlxcel_core::array_shape(hyper_input));
        let inject = self
            .block_inject_weight
            .as_ref()
            .expect("decoder hyper-connection carries injection weights")
            .forward(&normed);
        (mixed, mlxcel_core::share(hyper_input), inject)
    }

    fn inject(
        &self,
        branch: &MlxArray,
        hyper_input: &MlxArray,
        weights: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::compiled_hyper_inject(branch, hyper_input, weights, self.hc_count as i32)
    }
}

pub(crate) struct Qwen4FinalMixer(Qwen4HyperConnection);

impl Qwen4FinalMixer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self(Qwen4HyperConnection::from_weights(
            weights, config, prefix, false,
        )?))
    }

    pub(crate) fn forward(&self, hyper_input: &MlxArray) -> UniquePtr<MlxArray> {
        self.0.mix(hyper_input)
    }
}

enum Qwen4ExpertSwitch {
    Quantized(MoESwitch),
    Regular {
        gate_proj: UniquePtr<MlxArray>,
        up_proj: UniquePtr<MlxArray>,
        down_proj: UniquePtr<MlxArray>,
    },
}

impl Qwen4ExpertSwitch {
    fn forward(&self, input: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        if let Self::Quantized(switch) = self {
            return switch.forward(input, indices);
        }
        let Self::Regular {
            gate_proj,
            up_proj,
            down_proj,
        } = self
        else {
            unreachable!()
        };
        let input = mlxcel_core::expand_dims(input, -2);
        let input = mlxcel_core::expand_dims(&input, -3);
        let project = |input: &MlxArray, weight: &MlxArray| unsafe {
            mlxcel_core::gather_mm(input, weight, std::ptr::null(), indices as *const _, true)
        };
        let gate = project(&input, gate_proj);
        let up = project(&input, up_proj);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        mlxcel_core::squeeze_axis(&project(&activated, down_proj), -2)
    }
}

pub(crate) struct Qwen4SparseMoe {
    gate: UnifiedLinear,
    switch_mlp: Qwen4ExpertSwitch,
    shared_expert: Mlp,
    shared_expert_gate: UnifiedLinear,
    num_experts: usize,
    top_k: usize,
}

fn canonicalize_expert_selection(
    indices: &MlxArray,
    scores: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let order = mlxcel_core::argsort(indices, -1);
    (
        mlxcel_core::take_along_axis(indices, &order, -1),
        mlxcel_core::take_along_axis(scores, &order, -1),
    )
}

impl Qwen4SparseMoe {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
        qn_config: &Qwen4AttentionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gate_prefix = format!("{prefix}.gate");
        let (gate_group, gate_bits) = config.quant_params(&gate_prefix);
        let gate = UnifiedLinear::from_weights(weights, &gate_prefix, gate_group, gate_bits)?;
        let switch_prefix = format!("{prefix}.switch_mlp");
        let switch_mlp = if weights.contains_key(&format!("{switch_prefix}.gate_proj.scales")) {
            let load_expert = |name: &str| {
                let prefix = format!("{switch_prefix}.{name}");
                let (group_size, bits) = config.quant_params(&prefix);
                match UnifiedLinear::from_weights(weights, &prefix, group_size, bits)? {
                    UnifiedLinear::Quantized { weight, .. } => Ok(weight),
                    UnifiedLinear::Regular(_) => Err(format!(
                        "Qwen4 expert {prefix} has incomplete quantization tensors"
                    )),
                }
            };
            Qwen4ExpertSwitch::Quantized(MoESwitch::new(
                load_expert("gate_proj")?,
                load_expert("up_proj")?,
                load_expert("down_proj")?,
                config.num_experts as i32,
            ))
        } else {
            let load_expert = |name: &str| {
                let tensor_name = format!("{switch_prefix}.{name}.weight");
                weights
                    .get(&tensor_name)
                    .map(|weight| mlxcel_core::swap_axes(weight, -1, -2))
                    .ok_or_else(|| format!("missing required tensor {tensor_name}"))
            };
            Qwen4ExpertSwitch::Regular {
                gate_proj: load_expert("gate_proj")?,
                up_proj: load_expert("up_proj")?,
                down_proj: load_expert("down_proj")?,
            }
        };
        let shared_expert =
            Mlp::from_weights(weights, qn_config, &format!("{prefix}.shared_expert"))?;
        let shared_gate_prefix = format!("{prefix}.shared_expert_gate");
        let (shared_group, shared_bits) = config.quant_params(&shared_gate_prefix);
        let shared_expert_gate =
            UnifiedLinear::from_weights(weights, &shared_gate_prefix, shared_group, shared_bits)?;
        Ok(Self {
            gate,
            switch_mlp,
            shared_expert,
            shared_expert_gate,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
        })
    }

    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let gates = mlxcel_core::softmax_precise(&self.gate.forward(input), -1);
        let partitioned = mlxcel_core::argpartition(&gates, -(self.top_k as i32), -1);
        let shape = mlxcel_core::array_shape(&partitioned);
        let indices = mlxcel_core::slice(
            &partitioned,
            &[0, 0, (self.num_experts - self.top_k) as i32],
            &[shape[0], shape[1], self.num_experts as i32],
        );
        let scores = mlxcel_core::take_along_axis(&gates, &indices, -1);
        let score_sum = mlxcel_core::sum_axis(&scores, -1, true);
        let scores = mlxcel_core::divide(&scores, &score_sum);
        // Canonicalize every row, not only one-token decode. Batched MTP
        // verification must gather and reduce experts in the same order as N
        // sequential one-token forwards.
        let (indices, scores) = canonicalize_expert_selection(&indices, &scores);
        let experts = self.switch_mlp.forward(input, &indices);
        let routed = mlxcel_core::sum_axis(
            &mlxcel_core::multiply(&experts, &mlxcel_core::expand_dims(&scores, -1)),
            -2,
            false,
        );
        let shared = self.shared_expert.forward(input);
        let shared_gate = mlxcel_core::sigmoid(&self.shared_expert_gate.forward(input));
        mlxcel_core::add(&routed, &mlxcel_core::multiply(&shared, &shared_gate))
    }
}

struct Qwen4Ple {
    table: Arc<NGramTable>,
    key_proj: UnifiedLinear,
    value_proj: UnifiedLinear,
    norm_key: Qwen4RmsNorm,
    norm_query: Qwen4RmsNorm,
    norm_conv: Qwen4RmsNorm,
    conv1d_weight: UniquePtr<MlxArray>,
    hidden_size: usize,
    hc_count: usize,
    ngram_size: usize,
    heads_per_ngram: usize,
    head_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    multipliers: Vec<u64>,
    eos_token_id: i32,
    conv_state_len: usize,
}

fn append_ple_conv_state(
    state: &MlxArray,
    normed: &MlxArray,
    state_len: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let input = mlxcel_core::concatenate(state, normed, 1);
    let input_shape = mlxcel_core::array_shape(&input);
    let input_len = input_shape[1];
    let tail = mlxcel_core::contiguous(
        &mlxcel_core::slice(
            &input,
            &[0, input_len - state_len, 0],
            &[input_shape[0], input_len, input_shape[2]],
        ),
        false,
    );
    (input, tail)
}

#[allow(clippy::too_many_arguments)]
fn ple_convolution(
    gated: &MlxArray,
    normed: &MlxArray,
    state: &MlxArray,
    weight: &MlxArray,
    state_len: i32,
    dilation: i32,
    channels: i32,
    chunk_tokens: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(gated);
    let (batch, sequence) = (shape[0], shape[1]);
    if sequence <= chunk_tokens {
        let (conv_input, next_state) = append_ple_conv_state(state, normed, state_len);
        let conv = mlxcel_core::conv1d(
            &conv_input,
            weight,
            1,
            0,
            dilation,
            channels,
        );
        return (mlxcel_core::add(gated, &silu(&conv)), next_state);
    }

    let mut current_state = mlxcel_core::share(state);
    let mut outputs =
        Vec::with_capacity(((sequence + chunk_tokens - 1) / chunk_tokens) as usize);
    let mut start = 0;
    while start < sequence {
        let stop = (start + chunk_tokens).min(sequence);
        let gated_chunk =
            mlxcel_core::slice(gated, &[0, start, 0], &[batch, stop, channels]);
        let normed_chunk =
            mlxcel_core::slice(normed, &[0, start, 0], &[batch, stop, channels]);
        let (conv_input, next_state) =
            append_ple_conv_state(&current_state, &normed_chunk, state_len);
        let conv = mlxcel_core::conv1d(
            &conv_input,
            weight,
            1,
            0,
            dilation,
            channels,
        );
        outputs.push(mlxcel_core::add(&gated_chunk, &silu(&conv)));
        current_state = next_state;
        start = stop;
    }
    (
        mlxcel_core::concatenate_owned(&outputs, 1),
        current_state,
    )
}

fn ple_shifted_tokens(
    tokens: &[i32],
    history: &[i32],
    batch: usize,
    sequence: usize,
    ngram_size: usize,
    eos_token_id: i32,
) -> (Vec<i32>, Vec<i32>) {
    let context_len = ngram_size - 1;
    let mut shifted_tokens = Vec::with_capacity(batch * sequence * ngram_size);
    let mut next_history = Vec::with_capacity(batch * context_len);
    for row in 0..batch {
        let mut row_history = history[row * context_len..(row + 1) * context_len].to_vec();
        for &token in &tokens[row * sequence..(row + 1) * sequence] {
            shifted_tokens.push(token);
            for distance in 1..ngram_size {
                let candidate = row_history[context_len - distance];
                let crosses_eos = row_history[context_len - distance..]
                    .iter()
                    .skip(1)
                    .any(|&value| value == eos_token_id);
                shifted_tokens.push(if crosses_eos {
                    eos_token_id
                } else {
                    candidate
                });
            }
            row_history.rotate_left(1);
            row_history[context_len - 1] = token;
        }
        next_history.extend_from_slice(&row_history);
    }
    (shifted_tokens, next_history)
}

impl Qwen4Ple {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
        prefix: &str,
        table: Arc<NGramTable>,
    ) -> Result<Self, String> {
        let linear = |suffix: &str| {
            let name = format!("{prefix}.{suffix}");
            let (group, bits) = config.quant_params(&name);
            UnifiedLinear::from_weights(weights, &name, group, bits)
        };
        let conv_name = format!("{prefix}.conv1d.weight");
        let conv = weights
            .get(&conv_name)
            .ok_or_else(|| format!("missing required tensor {conv_name}"))?;
        let conv_shape = mlxcel_core::array_shape(conv);
        let conv1d_weight = if conv_shape.last().copied() != Some(1) {
            mlxcel_core::swap_axes(conv, -1, -2)
        } else {
            mlxcel_core::copy(conv)
        };
        let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
        let mut head_sizes = Vec::with_capacity(ngram_heads);
        let mut head_offsets = Vec::with_capacity(ngram_heads);
        let mut offset = 0u64;
        for head in 0..ngram_heads {
            let size = nth_prime_after(config.ngram_vocab_size_base as u64 - 1, head + 1);
            head_sizes.push(size);
            head_offsets.push(offset);
            offset += size;
        }
        let max_long = i64::MAX as u64;
        let half_bound = (max_long / config.vocab_size as u64 / 2).max(1);
        let multipliers = (0..config.ngram_size)
            .map(|index| {
                let value = config
                    .seed
                    .wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
                2 * (splitmix64(value) % half_bound) + 1
            })
            .collect();
        Ok(Self {
            table,
            key_proj: linear("key_proj")?,
            value_proj: linear("value_proj")?,
            norm_key: Qwen4RmsNorm::from_weights(
                weights,
                &format!("{prefix}.norm_key.weight"),
                config.rms_norm_eps,
                Some(config.hidden_size),
            )?,
            norm_query: Qwen4RmsNorm::from_weights(
                weights,
                &format!("{prefix}.norm_query.weight"),
                config.rms_norm_eps,
                Some(config.hidden_size),
            )?,
            norm_conv: Qwen4RmsNorm::from_weights(
                weights,
                &format!("{prefix}.norm_conv.weight"),
                config.rms_norm_eps,
                Some(config.hidden_size),
            )?,
            conv1d_weight,
            hidden_size: config.hidden_size,
            hc_count: config.hc_count,
            ngram_size: config.ngram_size,
            heads_per_ngram: config.heads_per_ngram,
            head_sizes,
            head_offsets,
            multipliers,
            eos_token_id: config.eos_token_id.unwrap_or(248_044),
            conv_state_len: (config.ple_conv_kernel_size - 1) * config.ngram_size,
        })
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        input_ids: &MlxArray,
        cache: &mut GatedDeltaCache,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        let sequence = input_shape[1];
        if sequence <= QWEN4_PREFILL_MICROCHUNK_TOKENS {
            return self.forward_chunk(hidden_states, input_ids, cache);
        }

        let hidden_shape = mlxcel_core::array_shape(hidden_states);
        let mut outputs = Vec::with_capacity(
            ((sequence + QWEN4_PREFILL_MICROCHUNK_TOKENS - 1)
                / QWEN4_PREFILL_MICROCHUNK_TOKENS) as usize,
        );
        let mut start = 0;
        while start < sequence {
            let stop = (start + QWEN4_PREFILL_MICROCHUNK_TOKENS).min(sequence);
            let hidden_chunk = mlxcel_core::slice(
                hidden_states,
                &[0, start, 0],
                &[hidden_shape[0], stop, hidden_shape[2]],
            );
            let id_chunk =
                mlxcel_core::slice(input_ids, &[0, start], &[input_shape[0], stop]);
            outputs.push(self.forward_chunk(&hidden_chunk, &id_chunk, cache)?);
            start = stop;
        }
        Ok(mlxcel_core::concatenate_owned(&outputs, 1))
    }

    fn forward_chunk(
        &self,
        hidden_states: &MlxArray,
        input_ids: &MlxArray,
        cache: &mut GatedDeltaCache,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        let batch = input_shape[0] as usize;
        let sequence = input_shape[1] as usize;
        let bytes = mlxcel_core::array_to_raw_bytes(input_ids);
        let tokens = bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("four-byte token")))
            .collect::<Vec<_>>();
        if tokens.len() != batch * sequence {
            return Err("Qwen4 PLE input token shape does not match host data".to_owned());
        }
        let context_len = self.ngram_size - 1;
        if cache.ple_token_history.len() != batch * context_len {
            cache.ple_token_history = vec![self.eos_token_id; batch * context_len];
        }
        let (shifted_tokens, next_history) = ple_shifted_tokens(
            &tokens,
            &cache.ple_token_history,
            batch,
            sequence,
            self.ngram_size,
            self.eos_token_id,
        );
        let mut indices = Vec::with_capacity(batch * sequence * self.head_sizes.len());
        for shifted in shifted_tokens.chunks_exact(self.ngram_size) {
            for ngram in 2..=self.ngram_size {
                let mut mixed = (shifted[0] as u64).wrapping_mul(self.multipliers[0]);
                for position in 1..ngram {
                    mixed ^=
                        (shifted[position] as u64).wrapping_mul(self.multipliers[position]);
                }
                let first_head = (ngram - 2) * self.heads_per_ngram;
                for head in first_head..first_head + self.heads_per_ngram {
                    indices.push(self.head_offsets[head] + mixed % self.head_sizes[head]);
                }
            }
        }
        cache.ple_token_history = next_history;
        let indices = indices
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let embeddings = self
            .table
            .gather_bf16(&indices)
            .map_err(|error| error.to_string())?;
        let embeddings = mlxcel_core::from_bytes_f16(
            &embeddings,
            &[
                batch as i32,
                sequence as i32,
                self.head_sizes.len() as i32,
                self.table.embedding_dim() as i32,
            ],
            true,
        );
        let embeddings = mlxcel_core::reshape(
            &embeddings,
            &[
                batch as i32,
                sequence as i32,
                (self.head_sizes.len() * self.table.embedding_dim()) as i32,
            ],
        );

        let keys = self.norm_key.forward(&self.key_proj.forward(&embeddings));
        let values = self.value_proj.forward(&embeddings);
        let queries = self.norm_query.forward(hidden_states);
        let keys = mlxcel_core::reshape(
            &keys,
            &[
                batch as i32,
                sequence as i32,
                self.hc_count as i32,
                self.hidden_size as i32,
            ],
        );
        let queries = mlxcel_core::reshape(
            &queries,
            &[
                batch as i32,
                sequence as i32,
                self.hc_count as i32,
                self.hidden_size as i32,
            ],
        );
        let gate = mlxcel_core::sum_axis(&mlxcel_core::multiply(&keys, &queries), -1, true);
        let gate = mlxcel_core::multiply_scalar(&gate, 1.0 / (self.hidden_size as f32).sqrt());
        let magnitude = mlxcel_core::abs(&gate);
        let epsilon = mlxcel_core::full_f32(&[1], 1e-6, mlxcel_core::array_dtype(&gate));
        let gate = mlxcel_core::multiply(
            &mlxcel_core::sign(&gate),
            &mlxcel_core::sqrt(&mlxcel_core::maximum(&magnitude, &epsilon)),
        );
        let gated = mlxcel_core::multiply(
            &mlxcel_core::sigmoid(&gate),
            &mlxcel_core::expand_dims(&values, -2),
        );
        let gated = mlxcel_core::reshape(
            &gated,
            &[
                batch as i32,
                sequence as i32,
                (self.hc_count * self.hidden_size) as i32,
            ],
        );
        let normed = self.norm_conv.forward(&gated);
        let state = cache
            .ple_conv_state
            .as_ref()
            .filter(|state| mlxcel_core::array_shape(state)[0] == batch as i32)
            .map(|state| mlxcel_core::copy(state))
            .unwrap_or_else(|| {
                mlxcel_core::zeros(
                    &[
                        batch as i32,
                        self.conv_state_len as i32,
                        (self.hc_count * self.hidden_size) as i32,
                    ],
                    mlxcel_core::array_dtype(hidden_states),
                )
            });
        let channels = (self.hc_count * self.hidden_size) as i32;
        let (output, next_state) = ple_convolution(
            &gated,
            &normed,
            &state,
            &self.conv1d_weight,
            self.conv_state_len as i32,
            self.ngram_size as i32,
            channels,
            QWEN4_PREFILL_MICROCHUNK_TOKENS,
        );
        cache.ple_conv_state = Some(next_state);
        Ok(output)
    }
}

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

// Decoder layer.
pub(crate) enum Qwen4AttentionVariant {
    FullAttention(Qwen4Attention),
    Linear(Qwen4GatedDeltaNet),
}

pub(crate) struct Qwen4DecoderLayer {
    pub(crate) is_linear: bool,
    pub(crate) attention: Qwen4AttentionVariant,
    pub(crate) mlp: Qwen4SparseMoe,
    ple: Option<Qwen4Ple>,
    attn_hyper_connection: Qwen4HyperConnection,
    mlp_hyper_connection: Qwen4HyperConnection,
}

impl Qwen4DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen4LayerCache,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let hidden =
            if let (Some(ple), Qwen4LayerCache::Linear(linear_cache)) = (&self.ple, &mut *cache) {
                mlxcel_core::add(
                    x,
                    &ple.forward(x, input_ids, linear_cache)
                        .expect("validated Qwen4 PLE lookup must succeed"),
                )
            } else {
                mlxcel_core::share(x)
            };
        let (mixed, hyper_input, injection_weights) = self.attn_hyper_connection.forward(&hidden);
        let branch = match (&self.attention, &mut *cache) {
            (Qwen4AttentionVariant::Linear(attention), Qwen4LayerCache::Linear(cache)) => {
                attention.forward(&mixed, mask, Some(cache))
            }
            (Qwen4AttentionVariant::Linear(attention), _) => attention.forward(&mixed, mask, None),
            (
                Qwen4AttentionVariant::FullAttention(attention),
                Qwen4LayerCache::Attention(cache),
            ) => attention.forward_with_position_ids(&mixed, cache, mask, position_ids),
            (Qwen4AttentionVariant::FullAttention(attention), _) => {
                let mut temporary = KVCache::new();
                attention.forward_with_position_ids(&mixed, &mut temporary, mask, position_ids)
            }
        };
        let hidden = self
            .attn_hyper_connection
            .inject(&branch, &hyper_input, &injection_weights);
        let (mixed, hyper_input, injection_weights) = self.mlp_hyper_connection.forward(&hidden);
        let branch = self.mlp.forward(&mixed);
        self.mlp_hyper_connection
            .inject(&branch, &hyper_input, &injection_weights)
    }

    fn forward_with_capture(
        &self,
        layer_idx: usize,
        x: &MlxArray,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen4LayerCache,
        position_ids: Option<&MlxArray>,
        snapshots: &mut Vec<GdnRollbackSnapshot>,
    ) -> UniquePtr<MlxArray> {
        let ple_rollback = match (&self.ple, &*cache) {
            (Some(_), Qwen4LayerCache::Linear(linear_cache)) => Some((
                mlxcel_core::share(x),
                mlxcel_core::share(input_ids),
                linear_cache
                    .ple_conv_state
                    .as_ref()
                    .map(|state| mlxcel_core::share(state)),
                linear_cache.ple_token_history.clone(),
            )),
            _ => None,
        };
        let hidden =
            if let (Some(ple), Qwen4LayerCache::Linear(linear_cache)) = (&self.ple, &mut *cache) {
                mlxcel_core::add(
                    x,
                    &ple.forward(x, input_ids, linear_cache)
                        .expect("validated Qwen4 PLE lookup must succeed"),
                )
            } else {
                mlxcel_core::share(x)
            };
        let (mixed, hyper_input, injection_weights) = self.attn_hyper_connection.forward(&hidden);
        let branch = match (&self.attention, &mut *cache) {
            (Qwen4AttentionVariant::Linear(attn), Qwen4LayerCache::Linear(cache)) => {
                attn.forward_with_capture(layer_idx, &mixed, mask, Some(cache), snapshots)
            }
            (Qwen4AttentionVariant::Linear(attn), _) => {
                attn.forward_with_capture(layer_idx, &mixed, mask, None, snapshots)
            }
            (Qwen4AttentionVariant::FullAttention(attn), Qwen4LayerCache::Attention(cache)) => {
                attn.forward_verify(&mixed, cache, mask, position_ids)
            }
            (Qwen4AttentionVariant::FullAttention(attn), _) => {
                let mut temporary = KVCache::new();
                attn.forward_verify(&mixed, &mut temporary, mask, position_ids)
            }
        };
        if let Some((inputs, input_ids, conv_state, token_history)) = ple_rollback {
            let snapshot = snapshots
                .last_mut()
                .filter(|snapshot| snapshot.layer_idx == layer_idx)
                .expect("PLE must be attached to a captured linear-attention layer");
            snapshot.ple_inputs = Some(inputs);
            snapshot.ple_input_ids = Some(input_ids);
            snapshot.ple_conv_state = conv_state;
            snapshot.ple_token_history = Some(token_history);
        }
        let hidden = self
            .attn_hyper_connection
            .inject(&branch, &hyper_input, &injection_weights);
        let (mixed, hyper_input, injection_weights) = self.mlp_hyper_connection.forward(&hidden);
        let branch = self.mlp.forward(&mixed);
        self.mlp_hyper_connection
            .inject(&branch, &hyper_input, &injection_weights)
    }

    pub(crate) fn forward_full_attention(
        &self,
        x: &MlxArray,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (mixed, hyper_input, injection_weights) = self.attn_hyper_connection.forward(x);
        let branch = match &self.attention {
            Qwen4AttentionVariant::FullAttention(attention) => {
                attention.forward_with_position_ids(&mixed, cache, mask, position_ids)
            }
            Qwen4AttentionVariant::Linear(_) => {
                unreachable!("the MTP layer must use full attention")
            }
        };
        let hidden = self
            .attn_hyper_connection
            .inject(&branch, &hyper_input, &injection_weights);
        let (mixed, hyper_input, injection_weights) = self.mlp_hyper_connection.forward(&hidden);
        let branch = self.mlp.forward(&mixed);
        let _ = input_ids;
        self.mlp_hyper_connection
            .inject(&branch, &hyper_input, &injection_weights)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
        qn_config: &Qwen4AttentionConfig,
        layer_idx: usize,
        ngram_table: Option<Arc<NGramTable>>,
        prefill_policy: Arc<Qwen4PrefillPolicy>,
    ) -> Result<Self, String> {
        Self::from_weights_at_prefix(
            weights,
            config,
            qn_config,
            &format!("model.layers.{layer_idx}"),
            config.is_linear_layer(layer_idx),
            ngram_table,
            config.ple_layer_ids.contains(&(layer_idx + 1)),
            prefill_policy,
        )
    }

    pub(crate) fn from_weights_at_prefix(
        weights: &WeightMap,
        config: &Qwen4Config,
        qn_config: &Qwen4AttentionConfig,
        prefix: &str,
        is_linear: bool,
        ngram_table: Option<Arc<NGramTable>>,
        has_ple: bool,
        prefill_policy: Arc<Qwen4PrefillPolicy>,
    ) -> Result<Self, String> {
        let attention = if is_linear {
            Qwen4AttentionVariant::Linear(Qwen4GatedDeltaNet::from_weights(
                weights,
                config,
                &format!("{prefix}.linear_attn"),
            )?)
        } else {
            Qwen4AttentionVariant::FullAttention(Qwen4Attention::from_weights_centered(
                weights,
                qn_config,
                &format!("{prefix}.self_attn"),
                prefill_policy,
            )?)
        };
        let ple = if has_ple {
            Some(Qwen4Ple::from_weights(
                weights,
                config,
                &format!("{prefix}.ple"),
                ngram_table.ok_or_else(|| "Qwen4 PLE requires an SSD n-gram table".to_owned())?,
            )?)
        } else {
            None
        };
        Ok(Self {
            is_linear,
            attention,
            mlp: Qwen4SparseMoe::from_weights(
                weights,
                config,
                qn_config,
                &format!("{prefix}.mlp"),
            )?,
            ple,
            attn_hyper_connection: Qwen4HyperConnection::from_weights(
                weights,
                config,
                &format!("{prefix}.attn_hyper_connection"),
                true,
            )?,
            mlp_hyper_connection: Qwen4HyperConnection::from_weights(
                weights,
                config,
                &format!("{prefix}.mlp_hyper_connection"),
                true,
            )?,
        })
    }
}

// Qwen4 Model.

fn new_initial_attention_cache() -> Qwen4LayerCache {
    Qwen4LayerCache::Attention(Box::new(KVCache::new_with_mode(KVCacheMode::Fp8)))
}

pub struct Qwen4Model {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<Qwen4DecoderLayer>,
    pub(crate) norm: Qwen4FinalMixer,
    pub(crate) lm_head: Option<UnifiedLinear>,
    compact_draft_head: Option<UnifiedLinear>,
    compact_mtp_verify_head: Option<UnifiedLinear>,
    pub(crate) config: Qwen4Config,
    mtp: Option<Qwen4MtpDraftModel>,
    /// Model-owned heterogeneous cache state used by one synchronous sequence.
    sequence_state: ModelOwnedSequenceState<Qwen4LayerCache>,
    /// Rotary-position state retained for the active text sequence.
    rope_state: RopeState,
    prefill_policy: Arc<Qwen4PrefillPolicy>,
    /// Rollback depth that every QSA layer must preserve in its bounded raw tail.
    qsa_rollback_horizon: Cell<i32>,
}

impl Qwen4Model {
    pub(crate) fn approximate_prefill_used(&self) -> bool {
        self.prefill_policy.approximate_used()
    }

    pub(crate) fn set_reference_prefill(&self, enabled: bool) {
        self.prefill_policy.set_force_reference(enabled);
    }

    fn forward_backbone_with_inputs(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen4LayerCache],
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let embedded = input_embeddings
            .map(mlxcel_core::copy)
            .unwrap_or_else(|| self.embed_tokens.forward(input_ids));
        let mut hidden = mlxcel_core::tile(&embedded, &[1, 1, self.config.hc_count as i32]);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(&hidden, input_ids, None, cache, position_ids);
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

    pub(crate) fn project_mtp_verify_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.project_compact_logits(hidden, &self.compact_mtp_verify_head, MTP_VERIFY_PREFIX)
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

    pub(crate) fn map_draft_token(token: i32) -> i32 {
        if token < MTP_DRAFT_PREFIX {
            token
        } else {
            token + DRAFT_CONTROL_START - MTP_DRAFT_PREFIX
        }
    }

    pub(crate) fn map_mtp_verify_token(token: i32) -> i32 {
        if token < MTP_VERIFY_PREFIX {
            token
        } else {
            token + DRAFT_CONTROL_START - MTP_VERIFY_PREFIX
        }
    }

    fn make_internal_caches(&self) -> Vec<Qwen4LayerCache> {
        let horizon = self.qsa_rollback_horizon.get();
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Qwen4LayerCache::Linear(GatedDeltaCache::new())
                } else {
                    let mut cache = new_initial_attention_cache();
                    if let Qwen4LayerCache::Attention(cache) = &mut cache {
                        cache
                            .set_auxiliary_rollback_horizon(horizon)
                            .expect("fresh QSA cache accepts configured rollback horizon");
                    }
                    cache
                }
            })
            .collect()
    }

    pub(crate) fn set_qsa_rollback_horizon(&self, horizon: i32) -> std::result::Result<(), String> {
        self.sequence_state
            .with_internal(|caches| -> std::result::Result<(), String> {
                for cache in caches {
                    if let Qwen4LayerCache::Attention(cache) = cache {
                        cache.set_auxiliary_rollback_horizon(horizon)?;
                    }
                }
                Ok(())
            })?;
        self.qsa_rollback_horizon.set(horizon);
        Ok(())
    }

    pub(crate) fn reserve_prefill_capacity(&self, total_tokens: i32) {
        self.sequence_state.with_internal(|caches| {
            for cache in caches {
                if let Qwen4LayerCache::Attention(cache) = cache {
                    cache.reserve_prefill_capacity(total_tokens);
                }
            }
        });
    }

    pub(crate) fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub(crate) fn clear_prepared_mrope(&self) {
        self.rope_state.clear_prepared();
    }

    pub(crate) fn attach_mtp(&mut self, mtp: Qwen4MtpDraftModel) {
        self.mtp = Some(mtp);
    }

    pub(crate) fn mtp(&self) -> Option<&Qwen4MtpDraftModel> {
        self.mtp.as_ref()
    }

    pub(crate) fn forward_mtp_prefill_chunks<F>(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
        rope_delta: Option<i32>,
        mut consume_chunk: F,
    ) -> std::result::Result<Qwen4MtpPrefill, String>
    where
        F: FnMut(i32, i32, &MlxArray),
    {
        self.reset_runtime_state();
        let shape = mlxcel_core::array_shape(input_ids);
        let prompt_len = shape[1];
        if prompt_len == 0 {
            return Err("MTP prefill requires at least one token".to_string());
        }
        self.reserve_prefill_capacity(prompt_len);
        if let (Some(position_ids), Some(rope_delta)) = (position_ids, rope_delta) {
            self.rope_state.prepare(position_ids, rope_delta);
            self.rope_state.activate_prepared()?;
        }
        let configured = mlxcel_core::generate::prefill_chunk_len();
        let chunk_len =
            mlxcel_core::generate::effective_prefill_chunk(configured, true, prompt_len as usize)
                .unwrap_or(prompt_len as usize) as i32;
        let mut final_chunk = None;
        let mut final_logits = None;

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
            .with_internal(|caches| caches.first().map(Qwen4LayerCache::offset).unwrap_or(0));
        self.rope_state.set_position(offset);
        if position_ids.is_some() {
            self.rope_state.finish_prefill();
        }
        Ok(Qwen4MtpPrefill {
            hidden: final_chunk.expect("MTP prefill requires a non-empty prompt"),
            first_logits: final_logits.expect("MTP prefill requires a non-empty prompt"),
        })
    }

    pub(crate) fn forward_mtp_text_suffix_chunks<F>(
        &self,
        input_ids: &MlxArray,
        mut consume_chunk: F,
    ) -> std::result::Result<Qwen4MtpPrefill, String>
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
            .with_internal(|caches| caches.first().map(Qwen4LayerCache::offset).unwrap_or(0));
        self.reserve_prefill_capacity(cached_len.saturating_add(suffix_len));
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
            .with_internal(|caches| caches.first().map(Qwen4LayerCache::offset).unwrap_or(0));
        self.rope_state.set_position(offset);
        Ok(Qwen4MtpPrefill {
            hidden,
            first_logits,
        })
    }

    pub(crate) fn forward_mtp_verify(&self, input_ids: &MlxArray) -> Qwen4MtpVerifyOutput {
        self.forward_mtp_verify_with_compact(input_ids, false)
    }

    pub(crate) fn forward_mtp_verify_with_compact(
        &self,
        input_ids: &MlxArray,
        compact_logits: bool,
    ) -> Qwen4MtpVerifyOutput {
        let rope_delta = self.rope_state.rope_delta();
        let (output, offset) = self.sequence_state.with_internal(|caches| {
            let embedded = self.embed_tokens.forward(input_ids);
            let mut hidden = mlxcel_core::tile(&embedded, &[1, 1, self.config.hc_count as i32]);
            let shape = mlxcel_core::array_shape(&hidden);
            let seq_len = shape[1];
            let cache_offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            let position_ids =
                rope_delta.map(|delta| decode_rope_positions(cache_offset, seq_len, delta));
            let mut gdn_states = Vec::with_capacity(self.layers.len());
            for (layer_idx, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate()
            {
                hidden = layer.forward_with_capture(
                    layer_idx,
                    &hidden,
                    input_ids,
                    None,
                    cache,
                    position_ids.as_deref(),
                    &mut gdn_states,
                );
            }
            let normalized = self.norm.forward(&hidden);
            let verifier_head = if compact_logits {
                self.compact_mtp_verify_head.as_ref()
            } else {
                self.lm_head.as_ref()
            };
            let promote_verifier_logits = verifier_head
                .and_then(UnifiedLinear::as_quantized_weight)
                .is_some_and(|weight| {
                    weight.mode == "affine"
                        && mlxcel_core::array_dtype(&weight.scales) == mlxcel_core::dtype::BFLOAT16
                });
            // For affine QMM, FP16 rows plus BF16 scales promote the verifier
            // logits to FP32. Cast only after RMSNorm so earlier residual
            // intermediates retain BF16's wider exponent range.
            let normalized = if promote_verifier_logits {
                mlxcel_core::astype(&normalized, mlxcel_core::dtype::FLOAT16)
            } else {
                normalized
            };
            let logits = if compact_logits {
                self.project_mtp_verify_logits(&normalized)
            } else {
                self.project_logits(&normalized)
            };
            let offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            (
                Qwen4MtpVerifyOutput {
                    hidden,
                    logits,
                    gdn_states,
                },
                offset,
            )
        });
        self.rope_state.set_position(offset);
        output
    }

    fn rollback_mtp_verify_to_retained_inputs(
        &self,
        gdn_states: &[GdnRollbackSnapshot],
        retained_inputs: usize,
        block_size: usize,
        materialize: bool,
    ) -> Qwen4RollbackPlan {
        let plan = self.sequence_state.with_internal(|caches| {
            let verify_offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            let plan =
                rollback_plan_for_retained_inputs(verify_offset, retained_inputs, block_size);
            for cache in caches.iter_mut() {
                if let Qwen4LayerCache::Attention(cache) = cache
                    && plan.trim > 0
                {
                    cache.trim(plan.trim);
                }
            }
            for snapshot in gdn_states {
                let Some(Qwen4LayerCache::Linear(cache)) = caches.get_mut(snapshot.layer_idx)
                else {
                    continue;
                };
                let Qwen4AttentionVariant::Linear(layer) =
                    &self.layers[snapshot.layer_idx].attention
                else {
                    continue;
                };
                let replay_len = plan.accepted_block_len;
                cache.state_cache = if replay_len == 0 {
                    snapshot
                        .init_state
                        .as_ref()
                        .map(|state| mlxcel_core::share(state))
                } else {
                    let batch = snapshot.batch;
                    let q = mlxcel_core::slice(
                        &snapshot.q,
                        &[0, 0, 0, 0],
                        &[
                            batch,
                            replay_len,
                            layer.num_k_heads as i32,
                            layer.head_k_dim as i32,
                        ],
                    );
                    let k = mlxcel_core::slice(
                        &snapshot.k,
                        &[0, 0, 0, 0],
                        &[
                            batch,
                            replay_len,
                            layer.num_k_heads as i32,
                            layer.head_k_dim as i32,
                        ],
                    );
                    let v = mlxcel_core::slice(
                        &snapshot.v,
                        &[0, 0, 0, 0],
                        &[
                            batch,
                            replay_len,
                            layer.num_v_heads as i32,
                            layer.head_v_dim as i32,
                        ],
                    );
                    let a = mlxcel_core::slice(
                        &snapshot.a,
                        &[0, 0, 0],
                        &[batch, replay_len, layer.num_v_heads as i32],
                    );
                    let b = mlxcel_core::slice(
                        &snapshot.b,
                        &[0, 0, 0],
                        &[batch, replay_len, layer.num_v_heads as i32],
                    );
                    let (_, replayed_state) = gated_delta_update(
                        (&q, &k, &v),
                        (&a, &b, &layer.a_log, &layer.dt_bias),
                        snapshot.init_state.as_deref(),
                        None,
                    );
                    Some(replayed_state)
                };
                let batch = snapshot.batch;
                let start = replay_len;
                let end = start + layer.conv_kernel_size as i32 - 1;
                let conv_state = mlxcel_core::slice(
                    &snapshot.conv_input,
                    &[0, start, 0],
                    &[batch, end, layer.conv_dim as i32],
                );
                cache.conv_state = Some(mlxcel_core::contiguous(&conv_state, false));
                cache.offset = plan.final_offset;
                if let Some(token_history) = &snapshot.ple_token_history {
                    cache.ple_conv_state = snapshot
                        .ple_conv_state
                        .as_ref()
                        .map(|state| mlxcel_core::share(state));
                    cache.ple_token_history = token_history.clone();
                    if replay_len > 0 {
                        let inputs = snapshot
                            .ple_inputs
                            .as_ref()
                            .expect("PLE rollback must retain layer inputs");
                        let input_ids = snapshot
                            .ple_input_ids
                            .as_ref()
                            .expect("PLE rollback must retain token IDs");
                        let hidden_size = mlxcel_core::array_shape(inputs)[2];
                        let inputs = mlxcel_core::slice(
                            inputs,
                            &[0, 0, 0],
                            &[batch, replay_len, hidden_size],
                        );
                        let replay_ids =
                            mlxcel_core::slice(input_ids, &[0, 0], &[batch, replay_len]);
                        let ple = self.layers[snapshot.layer_idx]
                            .ple
                            .as_ref()
                            .expect("PLE rollback snapshot must retain its layer");
                        let output = ple
                            .forward(&inputs, &replay_ids, cache)
                            .expect("captured PLE replay must remain valid");
                        mlxcel_core::eval(&output);
                    }
                }
            }
            if materialize {
                for cache in caches.iter_mut() {
                    cache.materialize_state();
                }
            }
            plan
        });
        self.rope_state.set_position(plan.final_offset);
        plan
    }

    pub(crate) fn rollback_mtp_verify(
        &self,
        gdn_states: &[GdnRollbackSnapshot],
        accepted: usize,
        block_size: usize,
        materialize: bool,
    ) -> Qwen4RollbackPlan {
        self.rollback_mtp_verify_to_retained_inputs(
            gdn_states,
            accepted.saturating_add(1),
            block_size,
            materialize,
        )
    }

    pub(crate) fn rollback_mtp_verify_to_prefix(
        &self,
        gdn_states: &[GdnRollbackSnapshot],
        block_size: usize,
        materialize: bool,
    ) -> Qwen4RollbackPlan {
        self.rollback_mtp_verify_to_retained_inputs(gdn_states, 0, block_size, materialize)
    }

    /// Sever persistent target state from the completed MTP round.
    pub(crate) fn materialize_mtp_cache_state(&self) {
        self.sequence_state.with_internal(|caches| {
            for cache in caches.iter_mut() {
                cache.materialize_state();
            }
        });
    }

    fn parse_config(model_dir: &Path) -> Result<Qwen4Config> {
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
            model_type == "qwen4_exp",
            "unsupported architecture {model_type:?} in {}; expected \"qwen4_exp\"",
            config_path.display()
        );
        let mut text_config = root_object
            .get("text_config")
            .cloned()
            .with_context(|| format!("{} is missing text_config", config_path.display()))?;
        let text_object = text_config.as_object_mut().with_context(|| {
            format!(
                "text_config in {} must contain a JSON object",
                config_path.display()
            )
        })?;
        let quantization = root_object
            .get("quantization")
            .or_else(|| root_object.get("quantization_config"))
            .with_context(|| {
                format!("{} is missing quantization metadata", config_path.display())
            })?;
        validate_quantization(quantization, &config_path)?;
        text_object.insert("quantization".to_owned(), quantization.clone());
        let config: Qwen4Config = serde_json::from_value(text_config).with_context(|| {
            format!(
                "failed to parse Qwen4 text config in {}",
                config_path.display()
            )
        })?;
        ensure!(
            config.model_type == "qwen4_exp_text",
            "unsupported text architecture {:?} in {}",
            config.model_type,
            config_path.display()
        );
        ensure!(
            config.hidden_size == 2_560
                && config.num_hidden_layers == 48
                && config.num_attention_heads == 24
                && config.num_key_value_heads == 2
                && config.head_dim_resolved() == 256
                && config.linear_num_key_heads == 16
                && config.linear_num_value_heads == 48
                && config.linear_key_head_dim == 128
                && config.linear_value_head_dim == 128
                && config.vocab_size == 248_320,
            "checkpoint {} does not match the fixed Qwen3.8 Flash Next text backbone",
            config_path.display()
        );
        ensure!(
            config.num_experts == 288
                && config.num_experts_per_tok == 10
                && config.moe_intermediate_size == 640
                && config.shared_expert_intermediate_size == 640,
            "checkpoint {} does not match the REAP-288 expert layout",
            config_path.display()
        );
        ensure!(
            config.hc_count == 4
                && config.hc_lowrank == 320
                && config.layer_types.len() == config.num_hidden_layers
                && config.ple_layer_ids == [2]
                && config.ple_embed_dim == 2_560
                && config.ngram_size == 3
                && config.heads_per_ngram == 8
                && config.make_ngram_vocab_size_divisible_by == 128
                && config.split_ngram_parts == 128,
            "checkpoint {} does not match the fixed Qwen4 hyper-connection/PLE layout",
            config_path.display()
        );
        ensure!(
            config.output_gate_type == "sigmoid" && config.eos_token_id.is_some(),
            "checkpoint {} has incompatible Qwen4 gating or EOS metadata",
            config_path.display()
        );
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
        ensure!(
            kv_cache_mode == KVCacheMode::Fp8,
            "Qwen3.8 Flash Next supports only the FP8 KV cache"
        );
        let config = Self::parse_config(model_dir)?;
        Self::validate_shard_index(model_dir)?;
        let ngram_prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards";
        let (ngram_group_size, ngram_bits) =
            config.quant_params("model.layers.1.ple.ple_embedding.ngram_embedding.shards.0");
        ensure!(
            ngram_group_size == 32 && ngram_bits == 4,
            "Qwen4 SSD n-gram table requires affine 4-bit/group-32 rows"
        );
        let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
        let table = Arc::new(NGramTable::prepare(
            model_dir,
            &NGramTableSpec {
                tensor_prefix: ngram_prefix.to_owned(),
                shard_count: config.split_ngram_parts,
                embedding_dim: config.ple_embed_dim / ngram_heads,
                group_size: ngram_group_size as usize,
            },
        )?);

        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, |name| {
            (name.starts_with("language_model.")
                || name.starts_with("model.language_model.")
                || name.starts_with("lm_head."))
                && !name.contains(".ngram_embedding.shards.")
        })
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to load Qwen4 checkpoint shards from {}",
                model_dir.display()
            )
        })?;
        ensure!(
            !weights.is_empty(),
            "checkpoint {} contains no Qwen4 language-model tensors",
            model_dir.display()
        );
        let weights = sanitize_language_model_weights(weights, &config, model_dir)?;
        tracing::info!(
            requested_cache_mode = ?kv_cache_mode,
            ngram_rows = table.rows(),
            "selected Qwen4 target cache and SSD n-gram policy"
        );
        Self::from_weights(&weights.target, &config, kv_cache_mode, Some(table))
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to construct Qwen4 model from checkpoint {}",
                    model_dir.display()
                )
            })
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen4Config,
        kv_cache_mode: KVCacheMode,
        ngram_table: Option<Arc<NGramTable>>,
    ) -> std::result::Result<Self, String> {
        if kv_cache_mode != KVCacheMode::Fp8 {
            return Err("Qwen3.8 Flash Next supports only the FP8 KV cache".to_string());
        }
        let qn_config = config.to_qwen4_attention_config();
        let prefill_policy = Arc::new(Qwen4PrefillPolicy::default());
        let (embed_group_size, embed_bits) = config.quant_params("model.embed_tokens");
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "model.embed_tokens",
            embed_group_size,
            embed_bits,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(Qwen4DecoderLayer::from_weights(
                weights,
                config,
                &qn_config,
                layer_idx,
                ngram_table.clone(),
                prefill_policy.clone(),
            )?);
        }
        let norm = Qwen4FinalMixer::from_weights(weights, config, "model.hyper_connection_mixer")?;
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
        let compact_mtp_verify_head = lm_head.as_ref().and_then(|head| {
            compact_head(
                head,
                config.vocab_size,
                MTP_VERIFY_PREFIX,
                MTP_VERIFY_PADDED,
            )
        });
        let internal_caches = layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Qwen4LayerCache::Linear(GatedDeltaCache::new())
                } else {
                    new_initial_attention_cache()
                }
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            compact_draft_head,
            compact_mtp_verify_head,
            config: config.clone(),
            mtp: None,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            rope_state: RopeState::new(),
            prefill_policy,
            qsa_rollback_horizon: Cell::new(0),
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

pub(crate) fn sanitize_weights(mut weights: WeightMap, config: &Qwen4Config) -> WeightMap {
    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }
    let keys = weights.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        if key.contains("conv1d.weight") {
            let value = weights
                .get(&key)
                .expect("key collected from the same weight map");
            if is_raw_conv1d_layout(&mlxcel_core::array_shape(value)) {
                weights.insert(key, mlxcel_core::swap_axes(value, -1, -2));
            }
        }
    }
    weights
}

struct SanitizedLanguageWeights {
    target: WeightMap,
}

fn sanitize_language_model_weights(
    weights: WeightMap,
    config: &Qwen4Config,
    _checkpoint_path: &Path,
) -> Result<SanitizedLanguageWeights> {
    let mut target = WeightMap::new();
    for (name, value) in weights {
        let normalized = if let Some(rest) = name.strip_prefix("model.language_model.") {
            format!("model.{rest}")
        } else if let Some(rest) = name.strip_prefix("language_model.") {
            rest.to_owned()
        } else {
            name
        };
        target.insert(normalized, value);
    }
    Ok(SanitizedLanguageWeights {
        target: sanitize_weights(target, config),
    })
}

const QWEN4_SNAPSHOT_FAMILY: &str = "qwen4-target";

fn snapshot_i32(snapshot: &ModelStateSnapshot, name: &str) -> std::result::Result<i32, String> {
    let value = snapshot
        .tensor(name)
        .ok_or_else(|| format!("Qwen4 snapshot is missing {name}"))?;
    if mlxcel_core::array_size(value) != 1 {
        return Err(format!("Qwen4 snapshot field {name} must be scalar"));
    }
    Ok(mlxcel_core::item_i32(&mlxcel_core::reshape(value, &[])))
}

fn push_snapshot_i32(snapshot: &mut ModelStateSnapshot, name: &str, value: i32) {
    let array = mlxcel_core::from_slice_i32(&[value], &[1]);
    snapshot.push_tensor(name, &array);
}

fn validate_snapshot_tensor_names(
    snapshot: &ModelStateSnapshot,
    layers: &[Qwen4DecoderLayer],
) -> std::result::Result<(), String> {
    let mut expected = BTreeSet::from([
        "meta.layer_count".to_string(),
        "mrope.position".to_string(),
        "mrope.rope_delta".to_string(),
    ]);
    if snapshot.tensor("mrope.position_ids").is_some() {
        expected.insert("mrope.position_ids".to_string());
    }
    let actual_dense: BTreeSet<String> = snapshot.tensor_names().map(str::to_owned).collect();
    let actual_paged: BTreeSet<String> = snapshot.paged_tensor_names().map(str::to_owned).collect();
    for (index, layer) in layers.iter().enumerate() {
        expected.insert(format!("layer.{index}.kind"));
        expected.insert(format!("layer.{index}.offset"));
        if !layer.is_linear {
            expected.insert(format!("layer.{index}.auxiliary_keys"));
            expected.insert(format!("layer.{index}.auxiliary_tail_start"));
            expected.insert(format!("layer.{index}.auxiliary_tail_end"));
            expected.insert(format!("layer.{index}.auxiliary_rollback_horizon"));
            expected.insert(format!("layer.{index}.auxiliary_block_size"));
            let block_name = format!("layer.{index}.auxiliary_block_keys");
            if actual_dense.contains(&block_name) || actual_paged.contains(&block_name) {
                expected.insert(block_name);
            }
        }
        if layer.is_linear {
            expected.insert(format!("layer.{index}.conv_state"));
            expected.insert(format!("layer.{index}.state_cache"));
            if layer.ple.is_some() {
                expected.insert(format!("layer.{index}.ple_conv_state"));
                expected.insert(format!("layer.{index}.ple_token_history"));
            }
        } else {
            expected.insert(format!("layer.{index}.keys"));
            expected.insert(format!("layer.{index}.values"));
        }
    }
    if actual_dense.len() + actual_paged.len()
        != snapshot.tensor_count() + snapshot.paged_tensor_names().count()
        || actual_dense
            .union(&actual_paged)
            .any(|name| !expected.contains(name))
    {
        return Err("Qwen4 snapshot tensor layout does not match the loaded model".to_string());
    }
    for (index, layer) in layers.iter().enumerate() {
        if !layer.is_linear
            && (!actual_dense.contains(&format!("layer.{index}.auxiliary_keys"))
                || ["keys", "values"].iter().any(|suffix| {
                    !actual_paged.contains(&format!("layer.{index}.{suffix}"))
                        && !actual_dense.contains(&format!("layer.{index}.{suffix}"))
                }))
        {
            return Err("Qwen4 snapshot is missing attention state".to_string());
        }
    }
    Ok(())
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen4Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let sequence_length = mlxcel_core::array_shape(input)[1];
        let rope_delta = self.rope_state.rope_delta();
        let (logits, offset) = self.sequence_state.with_internal(|caches| {
            let cache_offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            let position_ids =
                rope_delta.map(|delta| decode_rope_positions(cache_offset, sequence_length, delta));
            let hidden =
                self.forward_backbone_with_inputs(input, None, caches, position_ids.as_deref());
            let logits = self.project_logits(&self.norm.forward(&hidden));
            let offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            (logits, offset)
        });
        self.rope_state.set_position(offset);
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
        let rope_delta = self.rope_state.rope_delta();
        let (logits, offset) = self.sequence_state.with_internal(|caches| {
            let cache_offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
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
            let offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            (logits, offset)
        });
        self.rope_state.set_position(offset);
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
            let hidden = self.rope_state.with_position_ids(|position_ids| {
                self.forward_backbone_with_inputs(input_ids, input_embeddings, caches, position_ids)
            });
            let logits = self.project_logits(&self.norm.forward(&hidden));
            let offset = caches.first().map(Qwen4LayerCache::offset).unwrap_or(0);
            (logits, offset)
        });
        self.rope_state.set_position(offset);
        logits
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn prepare_embedding_prefill(&self) -> std::result::Result<(), String> {
        self.rope_state.activate_prepared()
    }

    fn after_prefill(&self) {
        self.rope_state.finish_prefill();
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn reserve_prefill_capacity(&self, _caches: &mut [KVCache], total_tokens: usize) {
        self.set_qsa_rollback_horizon(0)
            .expect("baseline prefill must configure a zero QSA rewind horizon");
        self.reserve_prefill_capacity(i32::try_from(total_tokens).unwrap_or(i32::MAX));
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
        self.rope_state.clear();
        self.prefill_policy.reset();
        if let Some(mtp) = &self.mtp {
            mtp.reset();
        }
    }

    fn approximate_prefill_used(&self) -> bool {
        self.approximate_prefill_used()
    }

    fn set_reference_prefill(&self, enabled: bool) {
        self.set_reference_prefill(enabled);
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        _seq_id: SequenceId,
        token_len: usize,
        previous: Option<&ModelStateSnapshot>,
    ) -> Option<ModelStateSnapshot> {
        let token_len_i32 = i32::try_from(token_len).ok()?;
        let mut snapshot = ModelStateSnapshot::new(QWEN4_SNAPSHOT_FAMILY, token_len);
        push_snapshot_i32(
            &mut snapshot,
            "meta.layer_count",
            i32::try_from(self.layers.len()).ok()?,
        );
        push_snapshot_i32(&mut snapshot, "mrope.position", self.rope_state.position());
        push_snapshot_i32(
            &mut snapshot,
            "mrope.rope_delta",
            self.rope_state.rope_delta().unwrap_or(i32::MIN),
        );
        self.rope_state.with_position_ids(|position_ids| {
            if let Some(position_ids) = position_ids {
                snapshot.push_tensor("mrope.position_ids", position_ids);
            }
        });

        let complete = self.sequence_state.with_internal(|caches| {
            if caches.len() != self.layers.len() {
                return false;
            }
            for (index, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
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
                    Qwen4LayerCache::Attention(cache) => {
                        let (tail_start, tail_end) = cache.auxiliary_raw_tail_range();
                        push_snapshot_i32(
                            &mut snapshot,
                            &format!("layer.{index}.auxiliary_tail_start"),
                            tail_start,
                        );
                        push_snapshot_i32(
                            &mut snapshot,
                            &format!("layer.{index}.auxiliary_tail_end"),
                            tail_end,
                        );
                        push_snapshot_i32(
                            &mut snapshot,
                            &format!("layer.{index}.auxiliary_rollback_horizon"),
                            cache.auxiliary_rollback_horizon(),
                        );
                        push_snapshot_i32(
                            &mut snapshot,
                            &format!("layer.{index}.auxiliary_block_size"),
                            cache.auxiliary_block_size().unwrap_or(0),
                        );
                        let Some(auxiliary_keys) = cache.auxiliary_keys.as_deref() else {
                            return false;
                        };
                        // The bounded raw tail slides in absolute coordinates;
                        // it is not append-only, so relative snapshot pages from
                        // an earlier tail are never reusable.
                        snapshot
                            .push_tensor(format!("layer.{index}.auxiliary_keys"), auxiliary_keys);
                        if let Some(auxiliary_block_keys) = cache.auxiliary_block_keys_view()
                            && snapshot
                                .push_paged_tensor(
                                    previous.filter(|snapshot| snapshot.token_len() < token_len),
                                    format!("layer.{index}.auxiliary_block_keys"),
                                    &auxiliary_block_keys,
                                    2,
                                )
                                .is_err()
                        {
                            return false;
                        }
                        if cache.mode != KVCacheMode::Fp8 {
                            return false;
                        }
                        let Some(tensors) = cache.fp8_snapshot_tensors() else {
                            return false;
                        };
                        if snapshot
                            .push_paged_tensor(
                                previous,
                                format!("layer.{index}.keys"),
                                &tensors.keys,
                                2,
                            )
                            .is_err()
                            || snapshot
                                .push_paged_tensor(
                                    previous,
                                    format!("layer.{index}.values"),
                                    &tensors.values,
                                    2,
                                )
                                .is_err()
                        {
                            return false;
                        }
                    }
                    Qwen4LayerCache::Linear(cache) => {
                        let (Some(conv_state), Some(state_cache)) =
                            (cache.conv_state.as_deref(), cache.state_cache.as_deref())
                        else {
                            return false;
                        };
                        snapshot.push_tensor(format!("layer.{index}.conv_state"), conv_state);
                        snapshot.push_tensor(format!("layer.{index}.state_cache"), state_cache);
                        if layer.ple.is_some() {
                            let Some(ple_conv_state) = cache.ple_conv_state.as_deref() else {
                                return false;
                            };
                            if cache.ple_token_history.is_empty() {
                                return false;
                            }
                            snapshot.push_tensor(
                                format!("layer.{index}.ple_conv_state"),
                                ple_conv_state,
                            );
                            let history = mlxcel_core::from_slice_i32(
                                &cache.ple_token_history,
                                &[cache.ple_token_history.len() as i32],
                            );
                            snapshot
                                .push_tensor(format!("layer.{index}.ple_token_history"), &history);
                        }
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
        if snapshot.family() != QWEN4_SNAPSHOT_FAMILY {
            return Err(format!(
                "Qwen4 snapshot family mismatch: expected {QWEN4_SNAPSHOT_FAMILY}, got {}",
                snapshot.family()
            ));
        }
        let token_len = i32::try_from(snapshot.token_len())
            .map_err(|_| "Qwen4 snapshot token length exceeds i32".to_string())?;
        if snapshot_i32(snapshot, "meta.layer_count")?
            != i32::try_from(self.layers.len()).unwrap_or(i32::MAX)
        {
            return Err("Qwen4 snapshot layer count does not match the loaded model".to_string());
        }
        validate_snapshot_tensor_names(snapshot, &self.layers)?;

        let mut restored = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let expected_kind = if layer.is_linear { 1 } else { 0 };
            if snapshot_i32(snapshot, &format!("layer.{index}.kind"))? != expected_kind {
                return Err(format!(
                    "Qwen4 snapshot layer {index} cache variant mismatch"
                ));
            }
            if snapshot_i32(snapshot, &format!("layer.{index}.offset"))? != token_len {
                return Err(format!("Qwen4 snapshot layer {index} offset mismatch"));
            }
            if layer.is_linear {
                let conv_state = snapshot
                    .tensor(&format!("layer.{index}.conv_state"))
                    .ok_or_else(|| format!("Qwen4 snapshot is missing layer {index} conv state"))?;
                let state_cache = snapshot
                    .tensor(&format!("layer.{index}.state_cache"))
                    .ok_or_else(|| {
                        format!("Qwen4 snapshot is missing layer {index} recurrent state")
                    })?;
                if mlxcel_core::array_shape(conv_state).len() != 3
                    || mlxcel_core::array_shape(state_cache).len() != 4
                {
                    return Err(format!(
                        "Qwen4 snapshot layer {index} linear cache layout mismatch"
                    ));
                }
                let (ple_conv_state, ple_token_history) = if layer.ple.is_some() {
                    let conv = snapshot
                        .tensor(&format!("layer.{index}.ple_conv_state"))
                        .ok_or_else(|| {
                            format!("Qwen4 snapshot is missing layer {index} PLE convolution state")
                        })?;
                    let history = snapshot
                        .tensor(&format!("layer.{index}.ple_token_history"))
                        .ok_or_else(|| {
                            format!("Qwen4 snapshot is missing layer {index} PLE token history")
                        })?;
                    if mlxcel_core::array_shape(conv).len() != 3
                        || mlxcel_core::array_shape(history).len() != 1
                    {
                        return Err(format!(
                            "Qwen4 snapshot layer {index} PLE cache layout mismatch"
                        ));
                    }
                    let bytes = mlxcel_core::array_to_raw_bytes(history);
                    let history = bytes
                        .chunks_exact(4)
                        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("four-byte token")))
                        .collect();
                    (Some(mlxcel_core::copy(conv)), history)
                } else {
                    (None, Vec::new())
                };
                restored.push(Qwen4LayerCache::Linear(GatedDeltaCache {
                    conv_state: Some(mlxcel_core::copy(conv_state)),
                    state_cache: Some(mlxcel_core::copy(state_cache)),
                    ple_conv_state,
                    ple_token_history,
                    offset: token_len,
                }));
            } else {
                let keys = snapshot
                    .paged_tensor(&format!("layer.{index}.keys"))
                    .and_then(|p| p.materialize())
                    .or_else(|| {
                        snapshot
                            .tensor(&format!("layer.{index}.keys"))
                            .map(mlxcel_core::copy)
                    })
                    .ok_or_else(|| format!("Qwen4 snapshot is missing layer {index} keys"))?;
                let values = snapshot
                    .paged_tensor(&format!("layer.{index}.values"))
                    .and_then(|p| p.materialize())
                    .or_else(|| {
                        snapshot
                            .tensor(&format!("layer.{index}.values"))
                            .map(mlxcel_core::copy)
                    })
                    .ok_or_else(|| format!("Qwen4 snapshot is missing layer {index} values"))?;
                let key_shape = mlxcel_core::array_shape(keys.as_ref().expect("materialized keys"));
                let value_shape =
                    mlxcel_core::array_shape(values.as_ref().expect("materialized values"));
                if key_shape.len() != 4
                    || key_shape != value_shape
                    || key_shape[0] != 1
                    || key_shape[2] < token_len
                {
                    return Err(format!(
                        "Qwen4 snapshot layer {index} attention cache layout mismatch"
                    ));
                }
                let mut cache = KVCache::new_with_mode(KVCacheMode::Fp8);
                cache.restore_fp8_snapshot(token_len, keys, values)?;
                let auxiliary_keys = snapshot
                    .tensor(&format!("layer.{index}.auxiliary_keys"))
                    .map(mlxcel_core::copy)
                    .ok_or_else(|| format!("Qwen4 snapshot is missing layer {index} QSA keys"))?;
                let tail_start =
                    snapshot_i32(snapshot, &format!("layer.{index}.auxiliary_tail_start"))?;
                let tail_end =
                    snapshot_i32(snapshot, &format!("layer.{index}.auxiliary_tail_end"))?;
                let horizon = snapshot_i32(
                    snapshot,
                    &format!("layer.{index}.auxiliary_rollback_horizon"),
                )?;
                let block_size =
                    snapshot_i32(snapshot, &format!("layer.{index}.auxiliary_block_size"))?;
                let auxiliary_block_keys = snapshot
                    .paged_tensor(&format!("layer.{index}.auxiliary_block_keys"))
                    .and_then(|paged| paged.materialize())
                    .or_else(|| {
                        snapshot
                            .tensor(&format!("layer.{index}.auxiliary_block_keys"))
                            .map(mlxcel_core::copy)
                    });
                cache.restore_auxiliary_block_keys(block_size, auxiliary_block_keys);
                cache.restore_auxiliary_keys(tail_start, tail_end, horizon, auxiliary_keys)?;
                cache.set_auxiliary_rollback_horizon(self.qsa_rollback_horizon.get())?;
                restored.push(Qwen4LayerCache::Attention(Box::new(cache)));
            }
        }

        let position = snapshot_i32(snapshot, "mrope.position")?;
        if position != token_len {
            return Err("Qwen4 snapshot MRoPE position does not match token length".to_string());
        }
        let rope_delta = match snapshot_i32(snapshot, "mrope.rope_delta")? {
            i32::MIN => None,
            value => Some(value),
        };
        self.sequence_state.replace_internal(restored);
        self.rope_state
            .restore(position, snapshot.tensor("mrope.position_ids"), rope_delta);
        Ok(())
    }

    fn snapshot_truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        snapshot.family() == QWEN4_SNAPSHOT_FAMILY && target_len == snapshot.token_len()
    }

    fn restore_sequence_state_truncated(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> std::result::Result<(), String> {
        if !self.snapshot_truncatable_to(snapshot, target_len) {
            return Err(
                "Qwen4 recurrent snapshots cannot be truncated to an earlier token".to_string(),
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

    fn dense_config(quantization: Option<Value>) -> Qwen4Config {
        let mut value = serde_json::json!({
            "model_type": "qwen4_exp_text",
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
    fn mixed_quantization_overrides_are_selected() {
        let mut config = dense_config(None);
        config.quantization = Some(Quantization {
            group_size: 64,
            bits: 4,
            mode: "affine".to_owned(),
            overrides: std::collections::HashMap::from([(
                "model.layers.0.linear_attn.in_proj_qkv".to_owned(),
                crate::qwen4_attention::TensorQuantization {
                    group_size: 64,
                    bits: 5,
                },
            )]),
        });
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
    fn unsupported_architecture_is_rejected() {
        let fixture = TestDir::new("unsupported");
        std::fs::write(
            fixture.0.join("config.json"),
            serde_json::to_vec(&serde_json::json!({"model_type": "llama"}))
                .expect("serialize config"),
        )
        .expect("write config");
        let error = Qwen4Model::parse_config(&fixture.0)
            .expect_err("invalid architecture must fail")
            .to_string();
        assert!(error.contains("unsupported architecture"), "{error}");
    }

    #[test]
    fn missing_checkpoint_shard_is_path_specific() {
        let fixture = TestDir::new("missing-shard");
        std::fs::write(
            fixture.0.join("model.safetensors.index.json"),
            br#"{"weight_map":{"language_model.model.embed_tokens.weight":"missing.safetensors"}}"#,
        )
        .expect("write index");

        let error = Qwen4Model::validate_shard_index(&fixture.0)
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
    fn raw_conv1d_is_transposed_without_mutating_norms() {
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
        let expected = mlxcel_core::from_slice_f32(&[0.0; 4], &[4]);
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
    fn batched_expert_selection_matches_sequential_row_order() {
        let indices = mlxcel_core::from_slice_i32(&[7, 2, 5, 3, 4, 1], &[1, 2, 3]);
        let scores = mlxcel_core::from_slice_f32(&[0.7, 0.2, 0.5, 0.3, 0.4, 0.1], &[1, 2, 3]);
        let (batch_indices, batch_scores) = canonicalize_expert_selection(&indices, &scores);

        let mut sequential_indices = Vec::new();
        let mut sequential_scores = Vec::new();
        for row in 0..2 {
            let row_indices = mlxcel_core::slice(&indices, &[0, row, 0], &[1, row + 1, 3]);
            let row_scores = mlxcel_core::slice(&scores, &[0, row, 0], &[1, row + 1, 3]);
            let (row_indices, row_scores) =
                canonicalize_expert_selection(&row_indices, &row_scores);
            sequential_indices.push(row_indices);
            sequential_scores.push(row_scores);
        }
        let sequential_indices = mlxcel_core::concatenate_owned(&sequential_indices, 1);
        let sequential_scores = mlxcel_core::concatenate_owned(&sequential_scores, 1);

        let expected_indices = mlxcel_core::from_slice_i32(&[2, 5, 7, 1, 3, 4], &[1, 2, 3]);
        let expected_scores =
            mlxcel_core::from_slice_f32(&[0.2, 0.5, 0.7, 0.1, 0.3, 0.4], &[1, 2, 3]);
        for equal in [
            mlxcel_core::allclose(&batch_indices, &sequential_indices, 0.0, 0.0),
            mlxcel_core::allclose(&batch_scores, &sequential_scores, 0.0, 0.0),
            mlxcel_core::allclose(&batch_indices, &expected_indices, 0.0, 0.0),
            mlxcel_core::allclose(&batch_scores, &expected_scores, 0.0, 0.0),
        ] {
            mlxcel_core::eval(&equal);
            assert!(mlxcel_core::item_bool(&equal));
        }
    }

    #[test]
    fn batched_ple_tail_matches_sequential_dispatches_at_materialization_boundary() {
        const STATE_LEN: i32 = 6;
        const SEQUENCE: i32 = 4;
        let initial =
            mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, STATE_LEN, 1]);
        let rows = mlxcel_core::from_slice_f32(&[7.0, 8.0, 9.0, 10.0], &[1, SEQUENCE, 1]);

        let (_, batched_tail) = append_ple_conv_state(&initial, &rows, STATE_LEN);
        let mut batched_cache = Qwen4LayerCache::Linear(GatedDeltaCache {
            ple_conv_state: Some(batched_tail),
            ..GatedDeltaCache::new()
        });
        batched_cache.materialize_state();
        mlxcel_core::clear_memory_cache();

        let mut sequential = initial;
        for row in 0..SEQUENCE {
            let next = mlxcel_core::slice(&rows, &[0, row, 0], &[1, row + 1, 1]);
            let (_, tail) = append_ple_conv_state(&sequential, &next, STATE_LEN);
            sequential = tail;
        }
        let Qwen4LayerCache::Linear(cache) = batched_cache else {
            unreachable!("test constructed a linear cache")
        };
        let batched = cache
            .ple_conv_state
            .as_deref()
            .expect("materialized PLE tail");
        let equal = mlxcel_core::allclose(batched, &sequential, 0.0, 0.0);
        mlxcel_core::eval(&equal);
        assert!(mlxcel_core::item_bool(&equal));
    }

    fn assert_arrays_close(left: &MlxArray, right: &MlxArray, tolerance: f32) {
        assert_eq!(
            mlxcel_core::array_shape(left),
            mlxcel_core::array_shape(right)
        );
        let close = mlxcel_core::allclose(left, right, tolerance, tolerance);
        mlxcel_core::eval(&close);
        assert!(mlxcel_core::item_bool(&close));
    }

    fn test_regular_linear(out_dim: i32, in_dim: i32, salt: i32) -> UnifiedLinear {
        let values = (0..out_dim * in_dim)
            .map(|index| 0.01 * (((index * 7 + salt) % 19) as f32 - 9.0))
            .collect::<Vec<_>>();
        let mut weights = WeightMap::new();
        weights.insert(
            "projection.weight".to_owned(),
            mlxcel_core::from_slice_f32(&values, &[out_dim, in_dim]),
        );
        UnifiedLinear::from_weights(&weights, "projection", 64, 4)
            .expect("regular test projection")
    }

    #[test]
    fn gated_delta_prefill_microchunks_match_monolithic_output_and_cache() {
        const SEQUENCE: i32 = 131;
        const HIDDEN: i32 = 2;
        const KEY_DIM: i32 = 2;
        const VALUE_DIM: i32 = 2;
        const CONV_DIM: i32 = 6;

        let conv_values = (0..CONV_DIM * 3)
            .map(|index| 0.015 * (((index * 5) % 17) as f32 - 8.0))
            .collect::<Vec<_>>();
        let layer = Qwen4GatedDeltaNet {
            hidden_size: HIDDEN as usize,
            num_v_heads: 1,
            num_k_heads: 1,
            head_k_dim: KEY_DIM as usize,
            head_v_dim: VALUE_DIM as usize,
            key_dim: KEY_DIM as usize,
            value_dim: VALUE_DIM as usize,
            conv_kernel_size: 3,
            conv_dim: CONV_DIM as usize,
            conv1d_weight: mlxcel_core::from_slice_f32(
                &conv_values,
                &[CONV_DIM, 3, 1],
            ),
            in_proj_qkv: test_regular_linear(CONV_DIM, HIDDEN, 1),
            aux_projections: Qwen4GatedAuxProjections::Separate {
                z: test_regular_linear(VALUE_DIM, HIDDEN, 3),
                b: test_regular_linear(1, HIDDEN, 5),
                a: test_regular_linear(1, HIDDEN, 7),
            },
            dt_bias: mlxcel_core::from_slice_f32(&[0.1], &[1]),
            a_log: mlxcel_core::from_slice_f32(&[-0.2], &[1]),
            norm: RMSNormGated::new(
                mlxcel_core::from_slice_f32(&[1.0; VALUE_DIM as usize], &[VALUE_DIM]),
                1e-6,
            ),
            out_proj: test_regular_linear(HIDDEN, VALUE_DIM, 11),
        };
        let input_values = (0..SEQUENCE * HIDDEN)
            .map(|index| 0.02 * ((index % 29) as f32 - 14.0))
            .collect::<Vec<_>>();
        let inputs =
            mlxcel_core::from_slice_f32(&input_values, &[1, SEQUENCE, HIDDEN]);

        let mut reference_cache = GatedDeltaCache::new();
        let reference =
            layer.forward_hidden_chunk(&inputs, None, Some(&mut reference_cache), None, true);
        let mut chunked_cache = GatedDeltaCache::new();
        let chunked =
            layer.forward_hidden_internal(&inputs, None, Some(&mut chunked_cache), None);

        assert_arrays_close(&chunked, &reference, 1e-5);
        assert_arrays_close(
            chunked_cache.conv_state.as_deref().expect("chunked conv tail"),
            reference_cache
                .conv_state
                .as_deref()
                .expect("reference conv tail"),
            0.0,
        );
        assert_arrays_close(
            chunked_cache
                .state_cache
                .as_deref()
                .expect("chunked recurrent state"),
            reference_cache
                .state_cache
                .as_deref()
                .expect("reference recurrent state"),
            1e-5,
        );
        assert_eq!(chunked_cache.offset, SEQUENCE);
        assert_eq!(reference_cache.offset, SEQUENCE);
    }

    #[test]
    fn ple_convolution_microchunks_preserve_output_and_uneven_tail() {
        const SEQUENCE: i32 = 7;
        const CHANNELS: i32 = 2;
        const STATE_LEN: i32 = 4;
        let state = mlxcel_core::from_slice_f32(
            &[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            &[1, STATE_LEN, CHANNELS],
        );
        let gated_values = (0..SEQUENCE * CHANNELS)
            .map(|index| 0.03 * (index as f32 - 5.0))
            .collect::<Vec<_>>();
        let normed_values = (0..SEQUENCE * CHANNELS)
            .map(|index| 0.02 * (((index * 3) % 11) as f32 - 5.0))
            .collect::<Vec<_>>();
        let gated =
            mlxcel_core::from_slice_f32(&gated_values, &[1, SEQUENCE, CHANNELS]);
        let normed =
            mlxcel_core::from_slice_f32(&normed_values, &[1, SEQUENCE, CHANNELS]);
        let weight = mlxcel_core::from_slice_f32(
            &[0.2, -0.1, 0.05, -0.15, 0.25, 0.1],
            &[CHANNELS, 3, 1],
        );

        let (reference, reference_tail) = ple_convolution(
            &gated,
            &normed,
            &state,
            &weight,
            STATE_LEN,
            2,
            CHANNELS,
            SEQUENCE,
        );
        let (chunked, chunked_tail) =
            ple_convolution(&gated, &normed, &state, &weight, STATE_LEN, 2, CHANNELS, 3);
        assert_arrays_close(&chunked, &reference, 0.0);
        assert_arrays_close(&chunked_tail, &reference_tail, 0.0);
        assert_eq!(
            mlxcel_core::array_shape(&chunked_tail).as_slice(),
            &[1, STATE_LEN, CHANNELS]
        );
    }

    #[test]
    fn ple_token_history_is_identical_across_uneven_microchunks() {
        const BATCH: usize = 2;
        const SEQUENCE: usize = 7;
        const NGRAM: usize = 4;
        const EOS: i32 = -1;
        let tokens = vec![
            1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
        ];
        let initial_history = vec![EOS, EOS, EOS, 91, 92, 93];
        let (reference_shifted, reference_history) = ple_shifted_tokens(
            &tokens,
            &initial_history,
            BATCH,
            SEQUENCE,
            NGRAM,
            EOS,
        );

        let mut history = initial_history;
        let mut shifted_by_row = vec![Vec::new(); BATCH];
        for (start, stop) in [(0, 3), (3, 6), (6, 7)] {
            let mut chunk_tokens = Vec::with_capacity(BATCH * (stop - start));
            for row in 0..BATCH {
                chunk_tokens.extend_from_slice(
                    &tokens[row * SEQUENCE + start..row * SEQUENCE + stop],
                );
            }
            let (shifted, next_history) = ple_shifted_tokens(
                &chunk_tokens,
                &history,
                BATCH,
                stop - start,
                NGRAM,
                EOS,
            );
            for (row, rows) in shifted.chunks_exact((stop - start) * NGRAM).enumerate() {
                shifted_by_row[row].extend_from_slice(rows);
            }
            history = next_history;
        }
        let chunked_shifted = shifted_by_row.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(chunked_shifted, reference_shifted);
        assert_eq!(history, reference_history);
    }

    #[test]
    fn rollback_plan_can_restore_the_pre_verify_prefix() {
        let plan = rollback_plan_for_retained_inputs(64_004, 0, 4);
        assert_eq!(
            plan,
            Qwen4RollbackPlan {
                accepted_block_len: 0,
                trim: 4,
                final_offset: 64_000,
            }
        );
    }
}
