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
use mlxcel_core::layers::{FusedQKVLinear, KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::utils::silu;
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
    Attention(KVCache),
    Linear(GatedDeltaCache),
}

impl Qwen3NextCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::Attention(kv) => kv.offset,
            Self::Linear(gd) => gd.offset,
        }
    }
}


// Attention with Gated Output.
pub(crate) struct Qwen3NextAttention {
    qkv_proj: FusedQKVLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_dims: i32,
    rope_base: f32,
}

impl Qwen3NextAttention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let output = self.forward_impl(x, cache, mask, false);
        self.o_proj.forward(&output)
    }

    /// Verify-only path from the checked-in mlxcel commit 4038da96. Each
    /// query attends through its own sequential-decode prefix.
    pub(crate) fn forward_verify(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let output = self.forward_impl(x, cache, mask, true);
        self.o_proj.forward(&output)
    }



    fn forward_impl(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        target_verify: bool,
    ) -> UniquePtr<MlxArray> {
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
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Dense text checkpoints use standard RoPE with the cache offset.
        queries = mlxcel_core::fast_rope(
            &queries,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            offset,
        );
        keys = mlxcel_core::fast_rope(
            &keys,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            offset,
        );

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(keys, values);

        let attn_out = if target_verify && l > 1 {
            self.attend_per_position(&queries, &cache_k, &cache_v)
        } else if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&queries, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        // Transpose back and reshape
        let output = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);

        // Apply sigmoid gating to output
        let gate_sigmoid = mlxcel_core::sigmoid(&gate);
        mlxcel_core::multiply(&output, &gate_sigmoid)
    }

    fn attend_per_position(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(queries);
        let k_shape = mlxcel_core::array_shape(keys);
        let v_shape = mlxcel_core::array_shape(values);
        let batch = q_shape[0];
        let query_heads = q_shape[1];
        let query_len = q_shape[2];
        let head_dim = q_shape[3];
        let kv_heads = k_shape[1];
        let prefix_len = k_shape[2] - query_len;

        let mut output: Option<UniquePtr<MlxArray>> = None;
        for position in 0..query_len {
            let query = mlxcel_core::slice(
                queries,
                &[0, 0, position, 0],
                &[batch, query_heads, position + 1, head_dim],
            );
            let kv_len = prefix_len + position + 1;
            let key = mlxcel_core::slice(
                keys,
                &[0, 0, 0, 0],
                &[batch, kv_heads, kv_len, head_dim],
            );
            let value = mlxcel_core::slice(
                values,
                &[0, 0, 0, 0],
                &[batch, kv_heads, kv_len, v_shape[3]],
            );
            let attended = mlxcel_core::layers::attention(
                &query,
                &key,
                &value,
                self.scale,
                None,
                0.0,
                0,
            );
            output = Some(match output {
                None => attended,
                Some(previous) => mlxcel_core::concatenate(&previous, &attended, 2),
            });
        }
        output.expect("verify attention requires at least one query position")
    }


    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let q_prefix = format!("{}.q_proj", prefix);
        let o_prefix = format!("{}.o_proj", prefix);
        let (q_group_size, q_bits) = config.quant_params(&q_prefix);
        let (o_group_size, o_bits) = config.quant_params(&o_prefix);

        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            prefix,
            q_group_size,
            q_bits,
            (config.num_attention_heads * 2) as i32,
            config.num_key_value_heads as i32,
            config.head_dim as i32,
        )?;
        let o_proj = UnifiedLinear::from_weights(weights, &o_prefix, o_group_size, o_bits)?;

        let q_norm_weight = weights
            .get(&format!("{}.q_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing q_norm weight: {}", prefix))?;
        let k_norm_weight = weights
            .get(&format!("{}.k_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing k_norm weight: {}", prefix))?;

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
        })
    }
}

// Dense MLP.
/// Dense MLP layer
pub(crate) struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gated = self.forward_hidden(x);
        self.down_proj.forward(&gated)
    }

    pub(crate) fn forward_hidden(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        mlxcel_core::multiply(&gate, &up)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gate_prefix = format!("{}.gate_proj", prefix);
        let up_prefix = format!("{}.up_proj", prefix);
        let down_prefix = format!("{}.down_proj", prefix);
        let (gate_group_size, gate_bits) = config.quant_params(&gate_prefix);
        let (up_group_size, up_bits) = config.quant_params(&up_prefix);
        let (down_group_size, down_bits) = config.quant_params(&down_prefix);

        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &gate_prefix,
                gate_group_size,
                gate_bits,
            )?,
            up_proj: UnifiedLinear::from_weights(weights, &up_prefix, up_group_size, up_bits)?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &down_prefix,
                down_group_size,
                down_bits,
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            "self_attn.q_proj.weight",
            &[2 * ATTENTION_WIDTH, HIDDEN_SIZE],
            0.0,
        );
        insert_f32(
            &mut weights,
            "self_attn.k_proj.weight",
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            0.0,
        );
        insert_f32(
            &mut weights,
            "self_attn.v_proj.weight",
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            0.0,
        );
        insert_f32(
            &mut weights,
            "self_attn.o_proj.weight",
            &[HIDDEN_SIZE, ATTENTION_WIDTH],
            0.0,
        );
        insert_f32(
            &mut weights,
            "self_attn.q_norm.weight",
            &[HEAD_DIM],
            1.0,
        );
        insert_f32(
            &mut weights,
            "self_attn.k_norm.weight",
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
            },
            "self_attn",
        )
        .expect("synthetic attention weights")
    }

    #[test]
    fn attention_projects_head_width_back_to_hidden_size() {
        let attention = unequal_width_attention();
        let input = mlxcel_core::from_slice_f32(&[0.0; 6], &[1, 2, 3]);

        let mut ordinary_cache = KVCache::new();
        let ordinary = attention.forward(&input, &mut ordinary_cache, None);
        assert_eq!(mlxcel_core::array_shape(&ordinary), vec![1, 2, 3]);

        let mut verify_cache = KVCache::new();
        let verify = attention.forward_verify(&input, &mut verify_cache, None);
        assert_eq!(mlxcel_core::array_shape(&verify), vec![1, 2, 3]);
    }
}
