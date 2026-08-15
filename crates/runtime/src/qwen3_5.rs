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
use crate::qwen3_next::{
    MLP, Quantization, Qwen3NextAttention, Qwen3NextCache, Qwen3NextConfig,
};
use anyhow::{Context, Result, ensure};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

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
        }
    }
}

// GatedDeltaNet - Qwen3.5 variant with separate projections.
/// GatedDeltaNet for Qwen3.5 with separate in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
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
    in_proj_z: UnifiedLinear,
    in_proj_b: UnifiedLinear,
    in_proj_a: UnifiedLinear,
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
        let out = self.forward_hidden_internal(inputs, mask, cache);
        self.out_proj.forward(&out)
    }

    fn forward_hidden_internal(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        let effective_mask = mask;

        // Separate projections (different from Qwen3Next's combined projections)
        let qkv = self.in_proj_qkv.forward(inputs);
        let z = self.in_proj_z.forward(inputs);
        let z = mlxcel_core::reshape(&z, &[b, s, self.num_v_heads as i32, self.head_v_dim as i32]);
        let b_proj = self.in_proj_b.forward(inputs);
        let a = self.in_proj_a.forward(inputs);

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

        // Run gated delta update (use guarded_mask which is None if batch dims mismatch)
        let (out, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
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
        let in_proj_z =
            UnifiedLinear::from_weights(weights, &z_prefix, z_group_size, z_bits)?;
        let in_proj_b =
            UnifiedLinear::from_weights(weights, &b_prefix, b_group_size, b_bits)?;
        let in_proj_a =
            UnifiedLinear::from_weights(weights, &a_prefix, a_group_size, a_bits)?;

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

        let out_proj =
            UnifiedLinear::from_weights(weights, &out_prefix, out_group_size, out_bits)?;

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
            in_proj_z,
            in_proj_b,
            in_proj_a,
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
    pub(crate) mlp: MLP,
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) post_attention_layernorm: RMSNorm,
}

impl Qwen35DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let r = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attn), Qwen3NextCache::Linear(c)) => {
                attn.forward(&normed, mask, Some(c))
            }
            (Qwen35AttentionVariant::Linear(attn), _) => attn.forward(&normed, mask, None),
            (Qwen35AttentionVariant::FullAttention(attn), Qwen3NextCache::Attention(c)) => {
                attn.forward(&normed, c, mask)
            }
            (Qwen35AttentionVariant::FullAttention(attn), _) => {
                let mut temp_cache = KVCache::new();
                attn.forward(&normed, &mut temp_cache, mask)
            }
        };

        let h = mlxcel_core::add(x, &r);

        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &mlp_out)
    }


    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        qn_config: &Qwen3NextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let is_linear = config.is_linear_layer(layer_idx);

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

        let mlp = MLP::from_weights(weights, qn_config, &format!("{}.mlp", prefix))?;

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
pub struct Qwen35Model {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<Qwen35DecoderLayer>,
    pub(crate) norm: RMSNorm,
    pub(crate) lm_head: Option<UnifiedLinear>,
    pub(crate) config: Qwen35Config,
    /// Model-owned heterogeneous cache state used by one synchronous sequence.
    sequence_state: ModelOwnedSequenceState<Qwen3NextCache>,
    /// MRoPE position state retained for the Qwen3.5 text path.
    mrope_state: MRopeState,
}

impl Qwen35Model {
    fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
    ) -> UniquePtr<MlxArray> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        let seq_len = mlxcel_core::array_shape(&hidden)[1];
        let attention_layer = self.config.full_attention_interval.saturating_sub(1);
        let attention_mask = if seq_len > 1 {
            let offset = caches
                .get(attention_layer)
                .map(Qwen3NextCache::offset)
                .unwrap_or(0);
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let mask = if layer.is_linear {
                None
            } else {
                attention_mask.as_deref()
            };
            hidden = layer.forward(&hidden, mask, cache);
        }
        self.norm.forward(&hidden)
    }

    fn project_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(hidden)
        } else {
            self.embed_tokens.as_linear(hidden)
        }
    }

    fn forward_internal(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
    ) -> UniquePtr<MlxArray> {
        let hidden = self.forward_hidden(input_ids, caches);
        self.project_logits(&hidden)
    }

    fn forward_last_internal(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.forward_hidden(input_ids, caches);
        let shape = mlxcel_core::array_shape(&hidden);
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let position = i32::try_from(last_pos).unwrap_or(i32::MAX);
        assert!(position < seq_len, "last logits position is outside the input sequence");
        let last_hidden = mlxcel_core::slice(
            &hidden,
            &[0, position, 0],
            &[batch, position + 1, hidden_size],
        );
        self.project_logits(&last_hidden)
    }

    fn make_internal_caches(&self) -> Vec<Qwen3NextCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect()
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
            format!("text_config in {} must contain a JSON object", config_path.display())
        })?;

        for object in [Some(root_object), Some(&*text_object)] {
            let object = object.expect("config object");
            if let Some(value) = object.get("num_experts") {
                let experts = value.as_i64().with_context(|| {
                    format!("num_experts in {} must be an integer", config_path.display())
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

        if let Some(quantization) = root_object.get("quantization") {
            validate_quantization(quantization, &config_path)?;
            text_object.insert("quantization".to_string(), quantization.clone());
        } else if let Some(quantization) = text_object.get("quantization") {
            validate_quantization(quantization, &config_path)?;
        }

        let config: Qwen35Config = serde_json::from_value(text_config)
            .with_context(|| format!("failed to parse dense text config in {}", config_path.display()))?;
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
                format!("shard for tensor {tensor:?} in {} must be a string", index_path.display())
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
            ensure!(path.is_file(), "missing checkpoint shard {}", path.display());
        }
        Ok(())
    }

    pub fn load(model_dir: &Path) -> Result<Self> {
        ensure!(
            model_dir.is_dir(),
            "model directory does not exist or is not a directory: {}",
            model_dir.display()
        );
        let config = Self::parse_config(model_dir)?;
        Self::validate_shard_index(model_dir)?;

        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, |name| {
            name.starts_with("language_model.")
        })
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to load checkpoint shards from {}", model_dir.display()))?;
        ensure!(
            !weights.is_empty(),
            "checkpoint {} contains no language_model.* tensors",
            model_dir.display()
        );
        let weights = sanitize_language_model_weights(weights, &config);
        Self::from_weights(&weights, &config)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to construct dense model from checkpoint {}",
                    model_dir.display()
                )
            })
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
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
                weights,
                config,
                &qn_config,
                layer_idx,
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
        let internal_caches = layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            config: config.clone(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            mrope_state: MRopeState::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn config(&self) -> &Qwen35Config {
        &self.config
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
            format!("{name}.bits in {} must be an integer", config_path.display())
        })?;
    let mode = object
        .get("mode")
        .and_then(Value::as_str)
        .with_context(|| {
            format!("{name}.mode in {} must be a string", config_path.display())
        })?;
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
        name.contains("conv1d.weight")
            && is_raw_conv1d_layout(&mlxcel_core::array_shape(value))
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

fn sanitize_language_model_weights(
    weights: WeightMap,
    config: &Qwen35Config,
) -> WeightMap {
    let mut language_weights = WeightMap::new();
    for (name, value) in weights {
        let Some(name) = name.strip_prefix("language_model.") else {
            continue;
        };
        if name.starts_with("visual.") || name.starts_with("vision_tower.") || name.starts_with("mtp.") {
            continue;
        }
        language_weights.insert(name.to_string(), value);
    }
    sanitize_weights(language_weights, config)
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen35Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_internal(|caches| self.forward_internal(input, caches))
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_internal(|caches| self.forward_last_internal(input_ids, caches, last_pos))
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

    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "qwr-{name}-{}-{}",
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
        assert_eq!(config.quant_params("model.layers.1.self_attn.q_proj"), (64, 4));
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
    fn sanitization_preserves_mixed_quantization_and_excludes_vision_and_mtp() {
        let config = dense_config(Some(serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                "group_size": 64,
                "bits": 5,
                "mode": "affine"
            }
        })));
        let mut weights = WeightMap::new();
        weights.insert(
            "language_model.model.layers.0.linear_attn.in_proj_qkv.scales".to_string(),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
        weights.insert(
            "language_model.visual.patch_embed.weight".to_string(),
            mlxcel_core::from_slice_f32(&[2.0], &[1]),
        );
        weights.insert(
            "language_model.mtp.layers.0.weight".to_string(),
            mlxcel_core::from_slice_f32(&[3.0], &[1]),
        );

        let sanitized = sanitize_language_model_weights(weights, &config);
        assert!(
            sanitized.contains_key("model.layers.0.linear_attn.in_proj_qkv.scales")
        );
        assert!(!sanitized.keys().any(|name| name.contains("visual")));
        assert!(!sanitized.keys().any(|name| name.contains("mtp")));
        assert_eq!(
            config.quant_params("model.layers.0.linear_attn.in_proj_qkv"),
            (64, 5)
        );
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
}
