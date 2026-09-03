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

//! Shared dense Qwen3.5 attention, MLP, quantization, and cache primitives.

use crate::gated_delta::GatedDeltaCache;
use crate::qwen_mrope::{InterleavedMRoPE, apply_multimodal_rotary_pos_emb};
use crate::qwen3_5_weights::{
    LayerTensor, ModelRole, Qwen35Linear, Qwen35QkvProjection, Qwen35WeightSource, TensorSlot,
};
use crate::qwen38_plan::{QWEN38_MLP_FUSION_PLAN, Qwen38PlanRole};
use mlxcel_core::cache::KVCacheMode;
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::concatenate;
use mlxcel_core::layers::{KVCache, RMSNorm};
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::layers::{QuantizedWeight, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorQuantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
    #[serde(flatten)]
    pub overrides: HashMap<String, TensorQuantization>,
}

#[derive(Debug, Clone)]
pub struct Qwen3NextConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub quantization: Option<Quantization>,
    pub mrope_section: Vec<i32>,
}

impl Qwen3NextConfig {
    pub fn quant_params(&self, prefix: &str) -> (i32, i32) {
        let Some(quantization) = &self.quantization else {
            return (64, 4);
        };
        debug_assert_eq!(quantization.mode, "affine");
        let language_model_prefix = format!("language_model.{prefix}");
        quantization
            .overrides
            .get(prefix)
            .or_else(|| quantization.overrides.get(&language_model_prefix))
            .map(|value| (value.group_size, value.bits))
            .unwrap_or((quantization.group_size, quantization.bits))
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f32 * self.partial_rotary_factor) as i32
    }
}

// Cache Types.
/// Mixed cache type for Qwen3Next layers
pub enum Qwen3NextCache {
    Attention(Box<KVCache>),
    Linear(GatedDeltaCache),
}

impl Qwen3NextCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::Attention(kv) => kv.offset,
            Self::Linear(gd) => gd.offset,
        }
    }

    /// Materialize persistent model state at an MTP round boundary.
    pub(crate) fn materialize_state(&mut self) {
        match self {
            Self::Attention(cache) => cache.materialize_state(),
            Self::Linear(cache) => {
                let arrays: Vec<*const MlxArray> =
                    [cache.conv_state.as_deref(), cache.state_cache.as_deref()]
                        .into_iter()
                        .flatten()
                        .map(|array| {
                            mlxcel_core::eval(array);
                            array as *const MlxArray
                        })
                        .collect();
                if !arrays.is_empty() {
                    unsafe { mlxcel_core::detach_all(&arrays) };
                }
            }
        }
    }
}

const QWEN38_HIGH_M_FP16_ATTENTION_ENV: &str = "QWR_EXPERIMENT_LONG_ATTN_F16";
const QWEN38_HIGH_M_FP16_ATTENTION_MIN_ROWS: i32 = 128;

fn qwen38_high_m_fp16_attention_requested() -> bool {
    std::env::var(QWEN38_HIGH_M_FP16_ATTENTION_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

fn use_qwen38_high_m_fp16_attention(
    pinned_qwen38_target: bool,
    enabled: bool,
    batch_size: i32,
    sequence_length: i32,
) -> bool {
    pinned_qwen38_target
        && enabled
        && batch_size
            .checked_mul(sequence_length)
            .is_some_and(|rows| rows >= QWEN38_HIGH_M_FP16_ATTENTION_MIN_ROWS)
}

#[cfg(test)]
pub(crate) struct F32Diagnostics {
    pub(crate) max_abs: f32,
    pub(crate) rmse: f64,
    pub(crate) cosine: f64,
    pub(crate) kl: f64,
    pub(crate) top1_equal: bool,
    pub(crate) top10_overlap: usize,
}

#[cfg(test)]
pub(crate) fn diagnose_f32(reference: &[f32], candidate: &[f32]) -> F32Diagnostics {
    assert_eq!(reference.len(), candidate.len());
    assert!(reference.len() >= 10);
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut candidate_norm = 0.0f64;
    for (&reference_value, &candidate_value) in reference.iter().zip(candidate) {
        assert!(reference_value.is_finite() && candidate_value.is_finite());
        let error = (reference_value - candidate_value).abs();
        max_abs = max_abs.max(error);
        squared_error += f64::from(error) * f64::from(error);
        dot += f64::from(reference_value) * f64::from(candidate_value);
        reference_norm += f64::from(reference_value) * f64::from(reference_value);
        candidate_norm += f64::from(candidate_value) * f64::from(candidate_value);
    }
    let reference_max = reference
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let reference_exp = reference
        .iter()
        .map(|&value| f64::from(value - reference_max).exp())
        .collect::<Vec<_>>();
    let reference_sum = reference_exp.iter().sum::<f64>();
    let reference_log_sum = f64::from(reference_max) + reference_sum.ln();
    let candidate_log_sum = f64::from(candidate_max)
        + candidate
            .iter()
            .map(|&value| f64::from(value - candidate_max).exp())
            .sum::<f64>()
            .ln();
    let kl = reference
        .iter()
        .zip(candidate)
        .zip(&reference_exp)
        .map(|((&reference_value, &candidate_value), &reference_value_exp)| {
            let probability = reference_value_exp / reference_sum;
            probability
                * ((f64::from(reference_value) - reference_log_sum)
                    - (f64::from(candidate_value) - candidate_log_sum))
        })
        .sum::<f64>();
    let mut reference_order = (0..reference.len()).collect::<Vec<_>>();
    reference_order
        .sort_unstable_by(|&left, &right| reference[right].total_cmp(&reference[left]));
    let mut candidate_order = (0..candidate.len()).collect::<Vec<_>>();
    candidate_order
        .sort_unstable_by(|&left, &right| candidate[right].total_cmp(&candidate[left]));
    F32Diagnostics {
        max_abs,
        rmse: (squared_error / reference.len() as f64).sqrt(),
        cosine: dot / (reference_norm.sqrt() * candidate_norm.sqrt()),
        kl,
        top1_equal: reference_order[0] == candidate_order[0],
        top10_overlap: reference_order[..10]
            .iter()
            .filter(|index| candidate_order[..10].contains(index))
            .count(),
    }
}

// Attention with Gated Output.
pub(crate) struct Qwen3NextAttention {
    qkv_proj: Qwen35QkvProjection,
    o_proj: Qwen35Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_dims: i32,
    rope_base: f32,
    mrope: InterleavedMRoPE,
    pinned_qwen38_target: bool,
    high_m_fp16_attention: bool,
}


impl Qwen3NextAttention {
    pub(crate) fn forward_with_position_ids(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (output, _) = self.forward_impl(x, cache, mask, position_ids, false);
        self.project_output(&output)
    }


    #[cfg(any(feature = "specprefill", test))]
    /// Draft-lookahead entry point used by SpecPrefill. The captured tensor is
    /// the normalized, post-RoPE query in `[B, H, L, D]` layout.
    pub(crate) fn forward_with_query_capture(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let (output, queries) = self.forward_impl(x, cache, mask, position_ids, true);
        (
            self.project_output(&output),
            queries.expect("query capture was requested"),
        )
    }

    /// Verify-only entry point. Multi-token blocks use bottom-right causal
    /// SDPA so every query attends through its own sequential-decode prefix.
    pub(crate) fn forward_verify(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (output, _) = self.forward_impl(x, cache, mask, position_ids, false);
        self.project_output(&output)
    }

    fn project_output(&self, output: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(output);
        if shape.len() == 3 && shape[1] == 4 && self.o_proj.needs_m4_exact_split() {
            let first = mlxcel_core::slice(output, &[0, 0, 0], &[shape[0], 3, shape[2]]);
            let last = mlxcel_core::slice(output, &[0, 3, 0], &[shape[0], 4, shape[2]]);
            return mlxcel_core::concatenate(
                &self.o_proj.forward(&first),
                &self.o_proj.forward(&last),
                1,
            );
        }
        self.o_proj.forward(output)
    }

    fn forward_impl(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
        capture_query: bool,
    ) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Q includes the learned gate plane, followed by K and V.
        let (q_proj_output, keys, values) = self.qkv_proj.forward(x);
        let q_proj_reshaped = mlxcel_core::reshape(&q_proj_output, &[b, l, self.num_heads, -1]);

        // Split into queries and gate
        let queries = mlxcel_core::slice(
            &q_proj_reshaped,
            &[0, 0, 0, 0],
            &[b, l, self.num_heads, self.head_dim],
        );
        // Note: MLX slice stop=-1 means dim_size-1 (excludes last), not "to end"
        let q_last_dim = mlxcel_core::array_shape(&q_proj_reshaped)[3];
        let gate = mlxcel_core::slice(
            &q_proj_reshaped,
            &[0, 0, 0, self.head_dim],
            &[b, l, self.num_heads, q_last_dim],
        );
        let gate = mlxcel_core::reshape(&gate, &[b, l, -1]);

        // Reshape and apply Q/K norms
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.num_heads, self.head_dim]);
        let keys = mlxcel_core::reshape(&keys, &[b, l, self.num_kv_heads, self.head_dim]);
        let values = mlxcel_core::reshape(&values, &[b, l, self.num_kv_heads, self.head_dim]);

        let queries = self.q_norm.forward(&queries);
        let keys = self.k_norm.forward(&keys);

        // Transpose to [B, H, L, D]
        let mut queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let mut keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let mut values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let offset = cache.offset;
        if let Some(position_ids) = position_ids {
            let (cosine, sine) = self.mrope.forward(position_ids);
            let dtype = mlxcel_core::array_dtype(&queries);
            let cosine = mlxcel_core::astype(&cosine, dtype);
            let sine = mlxcel_core::astype(&sine, dtype);
            let query_rotary = mlxcel_core::slice(
                &queries,
                &[0, 0, 0, 0],
                &[b, self.num_heads, l, self.rope_dims],
            );
            let query_pass = mlxcel_core::slice(
                &queries,
                &[0, 0, 0, self.rope_dims],
                &[b, self.num_heads, l, self.head_dim],
            );
            let key_rotary = mlxcel_core::slice(
                &keys,
                &[0, 0, 0, 0],
                &[b, self.num_kv_heads, l, self.rope_dims],
            );
            let key_pass = mlxcel_core::slice(
                &keys,
                &[0, 0, 0, self.rope_dims],
                &[b, self.num_kv_heads, l, self.head_dim],
            );
            let (query_rotary, key_rotary) =
                apply_multimodal_rotary_pos_emb(&query_rotary, &key_rotary, &cosine, &sine);
            queries = mlxcel_core::concatenate(&query_rotary, &query_pass, -1);
            keys = mlxcel_core::concatenate(&key_rotary, &key_pass, -1);
        } else {
            queries = mlxcel_core::fast_rope(
                &queries,
                self.rope_dims,
                false,
                self.rope_base,
                1.0,
                offset,
            );
            keys =
                mlxcel_core::fast_rope(&keys, self.rope_dims, false, self.rope_base, 1.0, offset);
        }

        let captured_query = capture_query.then(|| mlxcel_core::share(&queries));
        let use_fp16_attention = use_qwen38_high_m_fp16_attention(
            self.pinned_qwen38_target,
            self.high_m_fp16_attention,
            b,
            l,
        );
        if use_fp16_attention {
            queries = mlxcel_core::astype(&queries, mlxcel_core::dtype::FLOAT16);
            keys = mlxcel_core::astype(&keys, mlxcel_core::dtype::FLOAT16);
            values = mlxcel_core::astype(&values, mlxcel_core::dtype::FLOAT16);
        }

        // Symmetric Turbo4 MTP verification retains bottom-right causal
        // metadata while routing each row through the packed M1 reduction.
        // Corresponding rows therefore preserve sequential-decode arithmetic.
        let attn_out = if cache.mode == KVCacheMode::Turbo4 {
            if l > 1 && mask.is_none() {
                cache.update_and_turbo4_causal_attention(&queries, keys, values, self.scale)
            } else {
                cache.update_and_turbo4_attention(&queries, keys, values, self.scale, mask)
            }
        } else if l == 4 && mask.is_none() {
            // The native M4 SDPA crosses into a different reduction kernel.
            // Preserve corresponding-token M1 arithmetic with the exact M3
            // path plus one decode row, updating the cache once per segment.
            let first_queries = mlxcel_core::slice(
                &queries,
                &[0, 0, 0, 0],
                &[b, self.num_heads, 3, self.head_dim],
            );
            let last_query = mlxcel_core::slice(
                &queries,
                &[0, 0, 3, 0],
                &[b, self.num_heads, 4, self.head_dim],
            );
            let first_keys = mlxcel_core::slice(
                &keys,
                &[0, 0, 0, 0],
                &[b, self.num_kv_heads, 3, self.head_dim],
            );
            let first_keys = mlxcel_core::contiguous(&first_keys, false);
            let last_key = mlxcel_core::slice(
                &keys,
                &[0, 0, 3, 0],
                &[b, self.num_kv_heads, 4, self.head_dim],
            );
            let last_key = mlxcel_core::contiguous(&last_key, false);
            let first_values = mlxcel_core::slice(
                &values,
                &[0, 0, 0, 0],
                &[b, self.num_kv_heads, 3, self.head_dim],
            );
            let first_values = mlxcel_core::contiguous(&first_values, false);
            let last_value = mlxcel_core::slice(
                &values,
                &[0, 0, 3, 0],
                &[b, self.num_kv_heads, 4, self.head_dim],
            );
            let last_value = mlxcel_core::contiguous(&last_value, false);

            let (first_cache_k, first_cache_v) =
                cache.update_and_fetch(first_keys, first_values);
            let first = mlxcel_core::causal_attention(
                &first_queries,
                &first_cache_k,
                &first_cache_v,
                self.scale,
                0.0,
                0,
            );
            let (cache_k, cache_v) = cache.update_and_fetch(last_key, last_value);
            let last = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &last_query,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
            mlxcel_core::concatenate(&first, &last, 2)
        } else {
            let (cache_k, cache_v) = cache.update_and_fetch(keys, values);
            if l > 1 && mask.is_none() {
                mlxcel_core::causal_attention(&queries, &cache_k, &cache_v, self.scale, 0.0, 0)
            } else {
                let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
                unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &queries, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                    )
                }
            }
        };

        // Transpose back and reshape
        let output = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        // Keep the output projection and residual stream at their existing
        // FP32 boundary; only the high-M attention core changes precision.
        let output = if use_fp16_attention {
            mlxcel_core::astype(&output, mlxcel_core::dtype::FLOAT32)
        } else {
            output
        };

        // Apply sigmoid gating to output
        let gate_sigmoid = mlxcel_core::sigmoid(&gate);
        let gated = mlxcel_core::multiply(&output, &gate_sigmoid);
        (gated, captured_query)
    }

    pub(crate) fn from_weights(
        weights: &dyn Qwen35WeightSource,
        config: &Qwen3NextConfig,
        role: ModelRole,
        layer: usize,
    ) -> Result<Self, String> {
        let prefix = match role {
            ModelRole::Target => format!("model.layers.{layer}.self_attn"),
            ModelRole::Mtp => format!("mtp.layers.{layer}.self_attn"),
        };
        let (q_group_size, q_bits) = config.quant_params(&format!("{prefix}.q_proj"));
        let (o_group_size, o_bits) = config.quant_params(&format!("{prefix}.o_proj"));
        let slot = |tensor| TensorSlot::Layer {
            role,
            layer,
            tensor,
        };
        let qkv_proj = weights.qkv(
            role,
            layer,
            q_group_size,
            q_bits,
            (config.num_attention_heads * 2) as i32,
            config.num_key_value_heads as i32,
            config.head_dim as i32,
        )?;
        let o_proj = weights.linear(slot(LayerTensor::AttentionOutput), o_group_size, o_bits)?;
        let q_norm_weight = weights.tensor(slot(LayerTensor::AttentionQueryNorm))?;
        let k_norm_weight = weights.tensor(slot(LayerTensor::AttentionKeyNorm))?;
        let head_dim = config.head_dim as i32;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, config.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, config.rms_norm_eps),
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: config.rope_dims(),
            rope_base: config.rope_theta,
            mrope: InterleavedMRoPE::new(
                config.rope_dims() as usize,
                config.rope_theta,
                config.mrope_section.clone(),
            ),
            pinned_qwen38_target: weights.is_pinned_qwen38_gguf() && role == ModelRole::Target,
            high_m_fp16_attention: qwen38_high_m_fp16_attention_requested(),
        })
    }

    #[cfg(test)]
    pub(crate) fn set_high_m_fp16_attention_for_test(&mut self, enabled: bool) {
        self.high_m_fp16_attention = enabled;
    }
}

// Dense MLP.
enum MlpInputProjections {
    Separate {
        gate: Qwen35Linear,
        up: Qwen35Linear,
    },
    #[cfg(any(feature = "specprefill", test))]
    Fused {
        projection: Qwen35Linear,
        intermediate_size: i32,
    },
}

#[cfg(any(feature = "specprefill", test))]
fn fuse_mlp_input_projections(
    weights: Option<&WeightMap>,
    gate_prefix: &str,
    up_prefix: &str,
    gate: Qwen35Linear,
    up: Qwen35Linear,
) -> MlpInputProjections {
    let can_drop_linear_biases = weights.is_some_and(|weights| {
        !weights.contains_key(&format!("{gate_prefix}.bias"))
            && !weights.contains_key(&format!("{up_prefix}.bias"))
    });
    let fused = (|| {
        let (gate_weight, up_weight) = (
            gate.legacy_ref()?.quantized_weight()?,
            up.legacy_ref()?.quantized_weight()?,
        );
        if !can_drop_linear_biases
            || gate_weight.group_size != up_weight.group_size
            || gate_weight.bits != up_weight.bits
            || gate_weight.mode != up_weight.mode
            || gate_weight.global_scale.is_some()
            || up_weight.global_scale.is_some()
        {
            return None;
        }
        let weight = concatenate(&gate_weight.weight, &up_weight.weight, 0);
        let scales = concatenate(&gate_weight.scales, &up_weight.scales, 0);
        let biases = concatenate(
            gate_weight.biases.as_deref()?,
            up_weight.biases.as_deref()?,
            0,
        );
        Some(MlpInputProjections::Fused {
            projection: Qwen35Linear::legacy(UnifiedLinear::new(
                QuantizedWeight::new(
                    weight,
                    scales,
                    biases,
                    gate_weight.group_size,
                    gate_weight.bits,
                ),
                None,
            )),
            intermediate_size: mlxcel_core::array_shape(&gate_weight.weight)[0],
        })
    })();

    fused.unwrap_or(MlpInputProjections::Separate { gate, up })
}

#[cfg(not(any(feature = "specprefill", test)))]
fn fuse_mlp_input_projections(
    _weights: Option<&WeightMap>,
    _gate_prefix: &str,
    _up_prefix: &str,
    gate: Qwen35Linear,
    up: Qwen35Linear,
) -> MlpInputProjections {
    MlpInputProjections::Separate { gate, up }
}

/// Dense MLP layer
pub(crate) struct Mlp {
    execution: MlpExecution,
}

enum MlpExecution {
    Separate {
        input_projections: MlpInputProjections,
        down_proj: Qwen35Linear,
    },
    PinnedAffine(mlxcel_core::Qwen38AffineMlpFusion),
}

impl Mlp {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.execution {
            MlpExecution::Separate {
                input_projections,
                down_proj,
            } => {
                let (gate, up) = match input_projections {
                    MlpInputProjections::Separate { gate, up } => (gate.forward(x), up.forward(x)),
                    #[cfg(any(feature = "specprefill", test))]
                    MlpInputProjections::Fused {
                        projection,
                        intermediate_size,
                    } => {
                        let projected = projection.forward(x);
                        let shape = mlxcel_core::array_shape(&projected);
                        (
                            mlxcel_core::slice(
                                &projected,
                                &[0, 0, 0],
                                &[shape[0], shape[1], *intermediate_size],
                            ),
                            mlxcel_core::slice(
                                &projected,
                                &[0, 0, *intermediate_size],
                                &[shape[0], shape[1], *intermediate_size * 2],
                            ),
                        )
                    }
                };
                let gated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
                down_proj.forward(&gated)
            }
            MlpExecution::PinnedAffine(fusion) => fusion
                .forward(x)
                .expect("validated pinned Qwen3.8 MLP fusion must succeed"),
        }
    }

    #[cfg(test)]
    pub(crate) fn fusion_stats(&self, input_rows: usize) -> Option<mlxcel_core::Qwen38FusionStats> {
        match &self.execution {
            MlpExecution::PinnedAffine(fusion) => fusion.dispatch_stats(input_rows).ok(),
            MlpExecution::Separate { .. } => None,
        }
    }

    pub(crate) fn from_weights(
        weights: &dyn Qwen35WeightSource,
        config: &Qwen3NextConfig,
        role: ModelRole,
        layer: usize,
    ) -> Result<Self, String> {
        let prefix = match role {
            ModelRole::Target => format!("model.layers.{layer}.mlp"),
            ModelRole::Mtp => format!("mtp.layers.{layer}.mlp"),
        };
        let gate_prefix = format!("{prefix}.gate_proj");
        let up_prefix = format!("{prefix}.up_proj");
        let down_prefix = format!("{prefix}.down_proj");
        let (gate_group_size, gate_bits) = config.quant_params(&gate_prefix);
        let (up_group_size, up_bits) = config.quant_params(&up_prefix);
        let (down_group_size, down_bits) = config.quant_params(&down_prefix);
        let slot = |tensor| TensorSlot::Layer {
            role,
            layer,
            tensor,
        };
        let gate = weights.linear(slot(LayerTensor::MlpGate), gate_group_size, gate_bits)?;
        let up = weights.linear(slot(LayerTensor::MlpUp), up_group_size, up_bits)?;
        let down = weights.linear(slot(LayerTensor::MlpDown), down_group_size, down_bits)?;
        let descriptor = QWEN38_MLP_FUSION_PLAN.iter().find(|descriptor| {
            descriptor.layer
                == match role {
                    ModelRole::Target => layer,
                    ModelRole::Mtp => 64,
                }
                && descriptor.role
                    == match role {
                        ModelRole::Target => Qwen38PlanRole::Target,
                        ModelRole::Mtp => Qwen38PlanRole::Mtp,
                    }
        });
        if weights.qwen38_fusion_enabled()
            && role == ModelRole::Mtp
            && descriptor.is_some_and(|descriptor| descriptor.all_affine())
            && gate.is_m2_affine()
            && up.is_m2_affine()
            && down.is_m2_affine()
        {
            let gate = gate.into_m2_affine().expect("checked M2 affine gate");
            let up = up.into_m2_affine().expect("checked M2 affine up");
            let down = down.into_m2_affine().expect("checked M2 affine down");
            return mlxcel_core::Qwen38AffineMlpFusion::new_m2(gate, up, down)
                .map(|fusion| Self {
                    execution: MlpExecution::PinnedAffine(fusion),
                })
                .map_err(|error| error.to_string());
        }
        if weights.qwen38_fusion_enabled()
            && descriptor.is_some_and(|descriptor| descriptor.all_affine())
            && gate.is_affine()
            && up.is_affine()
            && down.is_affine()
        {
            let (gate, gate_m234) = gate.into_affine().expect("checked affine gate");
            let (up, up_m234) = up.into_affine().expect("checked affine up");
            let (down, down_m234) = down.into_affine().expect("checked affine down");
            return mlxcel_core::Qwen38AffineMlpFusion::new(
                gate,
                up,
                down,
                [gate_m234, up_m234, down_m234],
            )
            .map(|fusion| Self {
                execution: MlpExecution::PinnedAffine(fusion),
            })
            .map_err(|error| error.to_string());
        }

        Ok(Self {
            execution: MlpExecution::Separate {
                input_projections: fuse_mlp_input_projections(
                    weights.legacy_weights(),
                    &gate_prefix,
                    &up_prefix,
                    gate,
                    up,
                ),
                down_proj: down,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_fp16_attention_gate_is_pinned_and_high_m_only() {
        for rows in [1, 3, 4, 127] {
            assert!(!use_qwen38_high_m_fp16_attention(
                true, true, 1, rows
            ));
        }
        assert!(use_qwen38_high_m_fp16_attention(
            true, true, 1, 128
        ));
        assert!(use_qwen38_high_m_fp16_attention(
            true, true, 1, 288
        ));
        assert!(!use_qwen38_high_m_fp16_attention(
            false, true, 1, 288
        ));
        assert!(!use_qwen38_high_m_fp16_attention(
            true, false, 1, 288
        ));
    }

    #[test]
    #[ignore = "runs exact-shape Metal SDPA for the Qwen3.8 attention envelope"]
    fn experimental_qwen38_fp16_sdpa_component_quality() {
        const QUERY_HEADS: i32 = 24;
        const KV_HEADS: i32 = 4;
        const QUERY_ROWS: i32 = 128;
        const KEY_ROWS: i32 = 384;
        const HEAD_DIM: i32 = 256;
        let values = |len: usize, stride: usize, modulus: usize| {
            (0..len)
                .map(|index| {
                    ((index.wrapping_mul(stride) % modulus) as f32
                        - (modulus / 2) as f32)
                        / (modulus / 2) as f32
                })
                .collect::<Vec<_>>()
        };
        let query_values = values(
            (QUERY_HEADS * QUERY_ROWS * HEAD_DIM) as usize,
            7_919,
            2_003,
        );
        let key_values = values(
            (KV_HEADS * KEY_ROWS * HEAD_DIM) as usize,
            104_729,
            2_011,
        );
        let value_values = values(
            (KV_HEADS * KEY_ROWS * HEAD_DIM) as usize,
            1_879,
            2_021,
        );
        let query = mlxcel_core::from_slice_f32(
            &query_values,
            &[1, QUERY_HEADS, QUERY_ROWS, HEAD_DIM],
        );
        let keys = mlxcel_core::from_slice_f32(
            &key_values,
            &[1, KV_HEADS, KEY_ROWS, HEAD_DIM],
        );
        let values = mlxcel_core::from_slice_f32(
            &value_values,
            &[1, KV_HEADS, KEY_ROWS, HEAD_DIM],
        );
        let keys = mlxcel_core::astype(&keys, mlxcel_core::dtype::FLOAT16);
        let values = mlxcel_core::astype(&values, mlxcel_core::dtype::FLOAT16);
        let reference = mlxcel_core::causal_attention(
            &query,
            &keys,
            &values,
            1.0 / (HEAD_DIM as f32).sqrt(),
            0.0,
            0,
        );
        let query_f16 = mlxcel_core::astype(&query, mlxcel_core::dtype::FLOAT16);
        let candidate = mlxcel_core::causal_attention(
            &query_f16,
            &keys,
            &values,
            1.0 / (HEAD_DIM as f32).sqrt(),
            0.0,
            0,
        );
        assert_eq!(
            mlxcel_core::array_dtype(&reference),
            mlxcel_core::dtype::FLOAT32
        );
        assert_eq!(
            mlxcel_core::array_dtype(&candidate),
            mlxcel_core::dtype::FLOAT16
        );
        let candidate = mlxcel_core::astype(&candidate, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&reference);
        mlxcel_core::eval(&candidate);
        let to_f32 = |array: &MlxArray| {
            mlxcel_core::array_to_raw_bytes(array)
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let reference = to_f32(&reference);
        let candidate = to_f32(&candidate);
        let result = diagnose_f32(&reference, &candidate);
        eprintln!(
            "QWEN38_FP16_SDPA_COMPONENT M={QUERY_ROWS} K={KEY_ROWS} finite=true max_abs={:.9e} rmse={:.9e} cosine={:.12} kl={:.9e} top1_equal={} top10_overlap={}/10",
            result.max_abs,
            result.rmse,
            result.cosine,
            result.kl,
            result.top1_equal,
            result.top10_overlap,
        );
    }

    #[test]
    #[ignore = "runs exact Turbo4 cache writes on Metal"]
    fn qwen38_fp16_attention_preserves_turbo4_cache_bytes() {
        const HEADS: i32 = 4;
        const ROWS: i32 = 128;
        const HEAD_DIM: i32 = 256;
        let len = (HEADS * ROWS * HEAD_DIM) as usize;
        let keys = (0..len)
            .map(|index| ((index * 7_919 % 2_003) as f32 - 1_001.0) / 1_001.0)
            .collect::<Vec<_>>();
        let values = (0..len)
            .map(|index| ((index * 104_729 % 2_011) as f32 - 1_005.0) / 1_005.0)
            .collect::<Vec<_>>();
        let mut reference = KVCache::new_with_mode(KVCacheMode::Turbo4);
        reference.update(
            mlxcel_core::from_slice_f32(&keys, &[1, HEADS, ROWS, HEAD_DIM]),
            mlxcel_core::from_slice_f32(&values, &[1, HEADS, ROWS, HEAD_DIM]),
        );
        let candidate_keys =
            mlxcel_core::from_slice_f32(&keys, &[1, HEADS, ROWS, HEAD_DIM]);
        let candidate_values =
            mlxcel_core::from_slice_f32(&values, &[1, HEADS, ROWS, HEAD_DIM]);
        let mut candidate = KVCache::new_with_mode(KVCacheMode::Turbo4);
        candidate.update(
            mlxcel_core::astype(&candidate_keys, mlxcel_core::dtype::FLOAT16),
            mlxcel_core::astype(&candidate_values, mlxcel_core::dtype::FLOAT16),
        );
        assert_eq!(reference.seq_len(), ROWS);
        assert_eq!(candidate.seq_len(), ROWS);
        let reference = reference
            .turbo4_snapshot_tensors()
            .expect("reference Turbo4 sidecars");
        let candidate = candidate
            .turbo4_snapshot_tensors()
            .expect("candidate Turbo4 sidecars");
        for (name, reference, candidate) in [
            ("k_packed", reference.k_packed, candidate.k_packed),
            ("k_rescale", reference.k_rescale, candidate.k_rescale),
            ("v_packed", reference.v_packed, candidate.v_packed),
            ("v_norms", reference.v_norms, candidate.v_norms),
            ("v_rescale", reference.v_rescale, candidate.v_rescale),
        ] {
            mlxcel_core::eval(reference);
            mlxcel_core::eval(candidate);
            assert_eq!(
                mlxcel_core::array_shape(reference),
                mlxcel_core::array_shape(candidate),
                "{name} shape"
            );
            assert_eq!(
                mlxcel_core::array_dtype(reference),
                mlxcel_core::array_dtype(candidate),
                "{name} dtype"
            );
            assert_eq!(
                mlxcel_core::array_to_raw_bytes(reference),
                mlxcel_core::array_to_raw_bytes(candidate),
                "{name} bytes"
            );
        }
    }

    fn insert_f32(weights: &mut WeightMap, name: &str, shape: &[i32], value: f32) {
        let len = shape.iter().map(|&dim| dim as usize).product();
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&vec![value; len], shape),
        );
    }

    fn unequal_width_attention() -> Qwen3NextAttention {
        const HIDDEN_SIZE: i32 = 3;
        const NUM_HEADS: i32 = 2;
        const NUM_KV_HEADS: i32 = 1;
        const HEAD_DIM: i32 = 2;
        const ATTENTION_WIDTH: i32 = NUM_HEADS * HEAD_DIM;

        let mut weights = WeightMap::new();
        insert_f32(
            &mut weights,
            "model.layers.0.self_attn.q_proj.weight",
            &[2 * ATTENTION_WIDTH, HIDDEN_SIZE],
            0.0,
        );
        insert_f32(
            &mut weights,
            "model.layers.0.self_attn.k_proj.weight",
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            0.0,
        );
        insert_f32(
            &mut weights,
            "model.layers.0.self_attn.v_proj.weight",
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            0.0,
        );
        insert_f32(
            &mut weights,
            "model.layers.0.self_attn.o_proj.weight",
            &[HIDDEN_SIZE, ATTENTION_WIDTH],
            0.0,
        );
        insert_f32(
            &mut weights,
            "model.layers.0.self_attn.q_norm.weight",
            &[HEAD_DIM],
            1.0,
        );
        insert_f32(
            &mut weights,
            "model.layers.0.self_attn.k_norm.weight",
            &[HEAD_DIM],
            1.0,
        );

        Qwen3NextAttention::from_weights(
            &weights,
            &Qwen3NextConfig {
                num_attention_heads: NUM_HEADS as usize,
                num_key_value_heads: NUM_KV_HEADS as usize,
                head_dim: HEAD_DIM as usize,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                quantization: None,
                mrope_section: vec![1, 1, 1],
            },
            ModelRole::Target,
            0,
        )
        .expect("synthetic attention weights")
    }

    #[test]
    fn attention_projects_head_width_back_to_hidden_size() {
        let attention = unequal_width_attention();
        let input = mlxcel_core::from_slice_f32(&[0.0; 6], &[1, 2, 3]);

        let mut ordinary_cache = KVCache::new();
        let ordinary = attention.forward_with_position_ids(&input, &mut ordinary_cache, None, None);
        assert_eq!(mlxcel_core::array_shape(&ordinary), vec![1, 2, 3]);

        let mut verify_cache = KVCache::new();
        let verify = attention.forward_verify(&input, &mut verify_cache, None, None);
        assert_eq!(mlxcel_core::array_shape(&verify), vec![1, 2, 3]);
    }
    #[test]
    fn m4_m3_plus_m1_attention_matches_sequential_outputs_and_fp16_cache() {
        const QUERY_HEADS: i32 = 24;
        const KV_HEADS: i32 = 4;
        const HEAD_DIM: i32 = 256;
        const PREFIX: i32 = 17;
        const ROWS: i32 = 4;
        const SCALE: f32 = 0.0625;

        let tensor = |shape: &[i32], seed: usize| {
            let len = shape.iter().map(|dimension| *dimension as usize).product::<usize>();
            let values = (0..len)
                .map(|index| {
                    let centered = ((index * 17 + seed * 29) % 257) as i32 - 128;
                    centered as f32 * 0.001901
                })
                .collect::<Vec<_>>();
            mlxcel_core::from_slice_f32(&values, shape)
        };
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        let max_ulp = |left: &[u8], right: &[u8]| {
            assert_eq!(left.len(), right.len());
            left.chunks_exact(4)
                .zip(right.chunks_exact(4))
                .map(|(left, right)| {
                    let left = ordered(f32::from_le_bytes(left.try_into().unwrap()));
                    let right = ordered(f32::from_le_bytes(right.try_into().unwrap()));
                    left.abs_diff(right)
                })
                .max()
                .unwrap_or(0)
        };
        let initialize = || {
            let mut cache = KVCache::new_with_mode(KVCacheMode::Fp16);
            cache.update(
                tensor(&[1, KV_HEADS, PREFIX, HEAD_DIM], 1),
                tensor(&[1, KV_HEADS, PREFIX, HEAD_DIM], 2),
            );
            cache
        };
        let queries = tensor(&[1, ROWS, QUERY_HEADS, HEAD_DIM], 3);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = tensor(&[1, ROWS, KV_HEADS, HEAD_DIM], 4);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = tensor(&[1, ROWS, KV_HEADS, HEAD_DIM], 5);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let mut sequential_cache = initialize();
        let mut sequential_rows = Vec::with_capacity(ROWS as usize);
        let mut sequential_final = None;
        for row in 0..ROWS {
            let query = mlxcel_core::slice(
                &queries,
                &[0, 0, row, 0],
                &[1, QUERY_HEADS, row + 1, HEAD_DIM],
            );
            let key = mlxcel_core::slice(
                &keys,
                &[0, 0, row, 0],
                &[1, KV_HEADS, row + 1, HEAD_DIM],
            );
            let value = mlxcel_core::slice(
                &values,
                &[0, 0, row, 0],
                &[1, KV_HEADS, row + 1, HEAD_DIM],
            );
            let (cache_k, cache_v) = sequential_cache.update_and_fetch(key, value);
            let output = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &query,
                    &cache_k,
                    &cache_v,
                    SCALE,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
            let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
            mlxcel_core::eval(&output);
            sequential_rows.push(mlxcel_core::array_to_raw_bytes(&output));
            sequential_final = Some((cache_k, cache_v));
        }
        let (sequential_k, sequential_v) = sequential_final.unwrap();

        let mut split_cache = initialize();
        let first_queries =
            mlxcel_core::slice(&queries, &[0, 0, 0, 0], &[1, QUERY_HEADS, 3, HEAD_DIM]);
        let last_query =
            mlxcel_core::slice(&queries, &[0, 0, 3, 0], &[1, QUERY_HEADS, 4, HEAD_DIM]);
        let first_keys =
            mlxcel_core::slice(&keys, &[0, 0, 0, 0], &[1, KV_HEADS, 3, HEAD_DIM]);
        let first_keys = mlxcel_core::contiguous(&first_keys, false);
        let last_key = mlxcel_core::slice(&keys, &[0, 0, 3, 0], &[1, KV_HEADS, 4, HEAD_DIM]);
        let last_key = mlxcel_core::contiguous(&last_key, false);
        let first_values =
            mlxcel_core::slice(&values, &[0, 0, 0, 0], &[1, KV_HEADS, 3, HEAD_DIM]);
        let first_values = mlxcel_core::contiguous(&first_values, false);
        let last_value =
            mlxcel_core::slice(&values, &[0, 0, 3, 0], &[1, KV_HEADS, 4, HEAD_DIM]);
        let last_value = mlxcel_core::contiguous(&last_value, false);
        let (first_cache_k, first_cache_v) =
            split_cache.update_and_fetch(first_keys, first_values);
        let first = mlxcel_core::causal_attention(
            &first_queries,
            &first_cache_k,
            &first_cache_v,
            SCALE,
            0.0,
            0,
        );
        let (split_k, split_v) = split_cache.update_and_fetch(last_key, last_value);
        let last = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &last_query,
                &split_k,
                &split_v,
                SCALE,
                std::ptr::null(),
                0.0,
                0,
            )
        };
        let split = mlxcel_core::concatenate(&first, &last, 2);
        let split = mlxcel_core::transpose_axes(&split, &[0, 2, 1, 3]);
        for value in [&split, &sequential_k, &sequential_v, &split_k, &split_v] {
            mlxcel_core::eval(value);
        }
        let split = mlxcel_core::array_to_raw_bytes(&split);
        let row_bytes = split.len() / ROWS as usize;
        let mut output_max_ulp = 0;
        for row in 0..ROWS as usize {
            let row_ulp = max_ulp(
                &sequential_rows[row],
                &split[row * row_bytes..(row + 1) * row_bytes],
            );
            output_max_ulp = output_max_ulp.max(row_ulp);
        }
        let sequential_k = mlxcel_core::array_to_raw_bytes(&sequential_k);
        let sequential_v = mlxcel_core::array_to_raw_bytes(&sequential_v);
        let split_k = mlxcel_core::array_to_raw_bytes(&split_k);
        let split_v = mlxcel_core::array_to_raw_bytes(&split_v);
        assert!(output_max_ulp <= 1);
        assert_eq!(sequential_cache.offset, PREFIX + ROWS);
        assert_eq!(split_cache.offset, PREFIX + ROWS);
        assert_eq!(sequential_k, split_k);
        assert_eq!(sequential_v, split_v);
    }

}
