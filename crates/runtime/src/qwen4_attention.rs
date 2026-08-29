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

//! Shared dense Qwen4 attention, MLP, quantization, and cache primitives.

use crate::gated_delta::GatedDeltaCache;
use crate::qwen_rope::{InterleavedMRoPE, apply_rotary_pos_emb};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, QuantizedWeight, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
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
pub struct Qwen4AttentionConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub quantization: Option<Quantization>,
    pub mrope_section: Vec<i32>,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
}

impl Qwen4AttentionConfig {
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
/// Mixed cache type for Qwen4 layers
pub enum Qwen4LayerCache {
    Attention(Box<KVCache>),
    Linear(GatedDeltaCache),
}

impl Qwen4LayerCache {
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

struct Qwen4QsaIndexer {
    projection: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    n_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    compress_ratio: i32,
    block_topk: i32,
    rope_dims: i32,
    mrope: InterleavedMRoPE,
}
enum Qwen4QsaPlan {
    Mask(UniquePtr<MlxArray>),
    DecodeIndices(UniquePtr<MlxArray>),
    VerifyIndices(Vec<UniquePtr<MlxArray>>),
    PrefillIndices {
        indices: UniquePtr<MlxArray>,
        valid: UniquePtr<MlxArray>,
    },
}

impl Qwen4QsaIndexer {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        attention_prefix: &str,
    ) -> Result<Self, String> {
        let prefix = format!("{attention_prefix}.indexer");
        let projection_prefix = format!("{prefix}.index_qk_proj");
        let (group, bits) = config.quant_params(&projection_prefix);
        let projection = UnifiedLinear::from_weights(weights, &projection_prefix, group, bits)?;
        let centered_norm = |name: &str| -> Result<RMSNorm, String> {
            let weight = weights
                .get(name)
                .ok_or_else(|| format!("missing required tensor {name}"))?;
            let ones = mlxcel_core::ones(
                &mlxcel_core::array_shape(weight),
                mlxcel_core::array_dtype(weight),
            );
            Ok(RMSNorm::new(
                mlxcel_core::add(weight, &ones),
                config.rms_norm_eps,
            ))
        };
        Ok(Self {
            projection,
            q_norm: centered_norm(&format!("{prefix}.q_layernorm.weight"))?,
            k_norm: centered_norm(&format!("{prefix}.k_layernorm.weight"))?,
            n_heads: config.indexer_n_heads as i32,
            kv_heads: config.indexer_kv_heads as i32,
            head_dim: config.indexer_head_dim as i32,
            compress_ratio: config.indexer_compress_ratio as i32,
            block_topk: (config.indexer_budget / config.indexer_compress_ratio) as i32,
            rope_dims: (config.indexer_head_dim as f32 * config.partial_rotary_factor) as i32,
            mrope: InterleavedMRoPE::new(
                (config.indexer_head_dim as f32 * config.partial_rotary_factor) as usize,
                config.rope_theta,
                config.mrope_section.clone(),
            ),
        })
    }

    fn apply_rope(&self, input: &MlxArray, positions: &[i32], batch: i32) -> UniquePtr<MlxArray> {
        let mut ids = Vec::with_capacity(3 * batch as usize * positions.len());
        for _axis in 0..3 {
            for _row in 0..batch {
                ids.extend_from_slice(positions);
            }
        }
        let position_ids = mlxcel_core::from_slice_i32(&ids, &[3, batch, positions.len() as i32]);
        let (cosine, sine) = self.mrope.forward(&position_ids);
        let dtype = mlxcel_core::array_dtype(input);
        let cosine = mlxcel_core::astype(&cosine, dtype);
        let sine = mlxcel_core::astype(&sine, dtype);
        let shape = mlxcel_core::array_shape(input);
        let rotary = mlxcel_core::slice(
            input,
            &[0, 0, 0, 0],
            &[shape[0], shape[1], shape[2], self.rope_dims],
        );
        let pass = mlxcel_core::slice(
            input,
            &[0, 0, 0, self.rope_dims],
            &[shape[0], shape[1], shape[2], self.head_dim],
        );
        let (rotary, _) = apply_rotary_pos_emb(&rotary, &rotary, &cosine, &sine);
        mlxcel_core::concatenate(&rotary, &pass, -1)
    }

    fn plan(&self, input: &MlxArray, cache: &mut KVCache) -> Option<Qwen4QsaPlan> {
        let input_shape = mlxcel_core::array_shape(input);
        let batch = input_shape[0];
        let sequence = input_shape[1];
        let past_len = cache.offset;
        let projected = self.projection.forward(input);
        let projected = mlxcel_core::reshape(
            &projected,
            &[batch, sequence, self.n_heads + self.kv_heads, self.head_dim],
        );
        let query = mlxcel_core::slice(
            &projected,
            &[0, 0, 0, 0],
            &[batch, sequence, self.n_heads, self.head_dim],
        );
        let raw_keys = mlxcel_core::slice(
            &projected,
            &[0, 0, self.n_heads, 0],
            &[batch, sequence, self.n_heads + self.kv_heads, self.head_dim],
        );
        let raw_keys = mlxcel_core::reshape(&raw_keys, &[batch, sequence, self.head_dim]);
        cache.auxiliary_keys = Some(match cache.auxiliary_keys.take() {
            Some(previous) => mlxcel_core::concatenate(&previous, &raw_keys, 1),
            None => raw_keys,
        });
        let raw_keys = cache
            .auxiliary_keys
            .as_deref()
            .expect("QSA raw keys were just installed");
        let key_len = mlxcel_core::array_shape(raw_keys)[1];
        if key_len != past_len + sequence {
            cache.auxiliary_block_keys = None;
            return None;
        }
        let complete_blocks = key_len / self.compress_ratio;
        if complete_blocks <= self.block_topk {
            return None;
        }

        let query = self.q_norm.forward(&query);
        let query = mlxcel_core::transpose_axes(&query, &[0, 2, 1, 3]);
        let query_positions = (past_len..past_len + sequence).collect::<Vec<_>>();
        let query = self.apply_rope(&query, &query_positions, batch);

        let complete_key_len = complete_blocks * self.compress_ratio;
        let cached_blocks = match cache.auxiliary_block_keys.as_deref() {
            Some(keys) => {
                let shape = mlxcel_core::array_shape(keys);
                if shape[0] != batch || shape[1] != 1 || shape[3] != self.head_dim {
                    cache.auxiliary_block_keys = None;
                    0
                } else if shape[2] > complete_blocks {
                    cache.auxiliary_block_keys = Some(mlxcel_core::slice(
                        keys,
                        &[0, 0, 0, 0],
                        &[batch, 1, complete_blocks, self.head_dim],
                    ));
                    complete_blocks
                } else {
                    shape[2]
                }
            }
            None => 0,
        };
        if cached_blocks < complete_blocks {
            let new_block_count = complete_blocks - cached_blocks;
            let pooled = mlxcel_core::slice(
                raw_keys,
                &[0, cached_blocks * self.compress_ratio, 0],
                &[batch, complete_key_len, self.head_dim],
            );
            let pooled = mlxcel_core::reshape(
                &pooled,
                &[batch, new_block_count, self.compress_ratio, self.head_dim],
            );
            let pooled = mlxcel_core::mean_axis(
                &mlxcel_core::astype(&pooled, mlxcel_core::dtype::FLOAT32),
                2,
                false,
            );
            let pooled = mlxcel_core::astype(&pooled, mlxcel_core::array_dtype(raw_keys));
            let pooled = self.k_norm.forward(&pooled);
            let pooled = mlxcel_core::expand_dims(&pooled, 1);
            let block_positions = (cached_blocks..complete_blocks)
                .map(|block| block * self.compress_ratio)
                .collect::<Vec<_>>();
            let pooled = self.apply_rope(&pooled, &block_positions, batch);
            cache.auxiliary_block_keys = Some(match cache.auxiliary_block_keys.take() {
                Some(previous) => mlxcel_core::concatenate(&previous, &pooled, 2),
                None => pooled,
            });
        }
        let pooled = cache
            .auxiliary_block_keys
            .as_deref()
            .expect("QSA block keys were just installed");

        let query_f32 = mlxcel_core::astype(&query, mlxcel_core::dtype::FLOAT32);
        let pooled_f32 = mlxcel_core::astype(&pooled, mlxcel_core::dtype::FLOAT32);
        let pooled_t = mlxcel_core::transpose_axes(&pooled_f32, &[0, 1, 3, 2]);
        let scores = mlxcel_core::matmul(&query_f32, &pooled_t);
        let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
        let scores = mlxcel_core::sum_axis(&mlxcel_core::maximum(&scores, &zero), 1, false);
        let scores = mlxcel_core::multiply_scalar(&scores, 1.0 / (self.head_dim as f32).sqrt());

        let query_ends = mlxcel_core::arange_i32(past_len + 1, past_len + sequence + 1, 1);
        let query_ends = mlxcel_core::reshape(&query_ends, &[1, sequence, 1]);
        let ratio = mlxcel_core::from_slice_i32(&[self.compress_ratio], &[1]);
        let complete_counts = mlxcel_core::floor_divide(&query_ends, &ratio);
        let block_ids = mlxcel_core::arange_i32(0, complete_blocks, 1);
        let block_ids = mlxcel_core::reshape(&block_ids, &[1, 1, complete_blocks]);
        let valid_blocks = mlxcel_core::less(&block_ids, &complete_counts);
        // Quantize the relevance score before selection. Long-context QSA
        // matmuls can vary by a few low bits across Metal schedules; without
        // a stable key, those differences change the top-k boundary and then
        // cascade through subsequent recurrent layers. Integer ranking keeps
        // the score bin and recent-first block-ID tie break exact.
        let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
        let quantized = mlxcel_core::astype(
            &mlxcel_core::floor(&mlxcel_core::add(
                &mlxcel_core::multiply_scalar(&scores, 256.0),
                &half,
            )),
            mlxcel_core::dtype::INT32,
        );
        let rank_stride = mlxcel_core::from_slice_i32(&[complete_blocks + 1], &[1]);
        let ranked = mlxcel_core::add(&mlxcel_core::multiply(&quantized, &rank_stride), &block_ids);
        let invalid_rank = mlxcel_core::from_slice_i32(&[i32::MIN], &[1]);
        let ranked = mlxcel_core::where_cond(&valid_blocks, &ranked, &invalid_rank);
        let selected = mlxcel_core::argpartition(&ranked, -self.block_topk, -1);
        let selected = mlxcel_core::slice(
            &selected,
            &[0, 0, complete_blocks - self.block_topk],
            &[batch, sequence, complete_blocks],
        );
        // `argpartition` leaves the selected suffix unordered. Sparse
        // attention must reduce tokens in chronological order so repeated
        // snapshot restores use the same BF16 accumulation path.
        let selected = mlxcel_core::sort(&selected, -1);
        let selected = mlxcel_core::expand_dims(&selected, -1);
        let selected = mlxcel_core::multiply(&selected, &ratio);
        let offsets = mlxcel_core::arange_i32(0, self.compress_ratio, 1);
        let offsets = mlxcel_core::reshape(&offsets, &[1, 1, 1, self.compress_ratio]);
        let selected = mlxcel_core::add(&selected, &offsets);
        let selected = mlxcel_core::reshape(
            &selected,
            &[batch, sequence, self.block_topk * self.compress_ratio],
        );
        if batch == 1 && sequence == 1 {
            let selected = if complete_key_len < key_len {
                let tail = mlxcel_core::arange_i32(complete_key_len, key_len, 1);
                let tail = mlxcel_core::reshape(&tail, &[1, 1, key_len - complete_key_len]);
                mlxcel_core::concatenate(&selected, &tail, -1)
            } else {
                selected
            };
            return Some(Qwen4QsaPlan::DecodeIndices(mlxcel_core::reshape(
                &selected,
                &[-1],
            )));
        }
        if batch == 1
            && sequence > 1
            && sequence < 64
            && past_len / self.compress_ratio > self.block_topk
            && matches!(
                cache.mode,
                KVCacheMode::Fp16 | KVCacheMode::Fp8 | KVCacheMode::Int8
            )
        {
            let mut rows = Vec::with_capacity(sequence as usize);
            for row in 0..sequence {
                let row_selected = mlxcel_core::slice(
                    &selected,
                    &[0, row, 0],
                    &[1, row + 1, self.block_topk * self.compress_ratio],
                );
                let row_selected = mlxcel_core::reshape(&row_selected, &[-1]);
                let query_end = past_len + row + 1;
                let complete_end = (query_end / self.compress_ratio) * self.compress_ratio;
                let row_selected = if complete_end < query_end {
                    let tail = mlxcel_core::arange_i32(complete_end, query_end, 1);
                    mlxcel_core::concatenate(&row_selected, &tail, 0)
                } else {
                    row_selected
                };
                rows.push(row_selected);
            }
            return Some(Qwen4QsaPlan::VerifyIndices(rows));
        }
        if batch == 1
            && sequence >= 64
            && past_len / self.compress_ratio > self.block_topk
            && matches!(
                cache.mode,
                KVCacheMode::Fp16 | KVCacheMode::Fp8 | KVCacheMode::Int8
            )
        {
            let tail_starts = mlxcel_core::multiply(&complete_counts, &ratio);
            let offsets = mlxcel_core::arange_i32(0, self.compress_ratio, 1);
            let offsets = mlxcel_core::reshape(&offsets, &[1, 1, self.compress_ratio]);
            let tail_indices = mlxcel_core::add(&tail_starts, &offsets);
            let tail_valid = mlxcel_core::less(&tail_indices, &query_ends);
            let last_visible =
                mlxcel_core::subtract(&query_ends, &mlxcel_core::from_slice_i32(&[1], &[1]));
            let tail_indices = mlxcel_core::where_cond(&tail_valid, &tail_indices, &last_visible);
            let indices = mlxcel_core::concatenate(&selected, &tail_indices, -1);
            let selected_valid = mlxcel_core::ones(
                &mlxcel_core::array_shape(&selected),
                mlxcel_core::dtype::BOOL,
            );
            let valid = mlxcel_core::concatenate(&selected_valid, &tail_valid, -1);
            return Some(Qwen4QsaPlan::PrefillIndices { indices, valid });
        }

        let selected_shape = mlxcel_core::array_shape(&selected);
        let selected_values = mlxcel_core::ones(&selected_shape, mlxcel_core::dtype::BOOL);
        let selected_mask =
            mlxcel_core::zeros(&[batch, sequence, key_len + 1], mlxcel_core::dtype::BOOL);
        let selected_mask =
            mlxcel_core::put_along_axis(&selected_mask, &selected, &selected_values, -1);
        let selected_mask =
            mlxcel_core::slice(&selected_mask, &[0, 0, 0], &[batch, sequence, key_len]);

        let tokens = mlxcel_core::arange_i32(0, key_len, 1);
        let tokens = mlxcel_core::reshape(&tokens, &[1, 1, key_len]);
        let tail_starts = mlxcel_core::multiply(&complete_counts, &ratio);
        let tail = mlxcel_core::logical_and(
            &mlxcel_core::greater_equal(&tokens, &tail_starts),
            &mlxcel_core::less(&tokens, &query_ends),
        );
        let causal = mlxcel_core::less(&tokens, &query_ends);
        let topk = mlxcel_core::from_slice_i32(&[self.block_topk], &[1]);
        let use_sparse = mlxcel_core::greater(&complete_counts, &topk);
        let sparse = mlxcel_core::logical_or(&selected_mask, &tail);
        Some(Qwen4QsaPlan::Mask(mlxcel_core::expand_dims(
            &mlxcel_core::where_cond(&use_sparse, &sparse, &causal),
            1,
        )))
    }
}

// Attention with Gated Output.
pub(crate) struct Qwen4Attention {
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
    mrope: InterleavedMRoPE,
    indexer: Option<Qwen4QsaIndexer>,
}

impl Qwen4Attention {
    pub(crate) fn forward_with_position_ids(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (output, _) = self.forward_impl(x, cache, mask, position_ids, false);
        self.o_proj.forward(&output)
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
        self.o_proj.forward(&output)
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
        let qsa_plan = self
            .indexer
            .as_ref()
            .and_then(|indexer| indexer.plan(x, cache));
        let qsa_mask = match &qsa_plan {
            Some(Qwen4QsaPlan::Mask(mask)) => Some(mask.as_ref().expect("QSA mask is non-null")),
            _ => None,
        };
        let qsa_indices = match (mask.is_none(), &qsa_plan) {
            (true, Some(Qwen4QsaPlan::DecodeIndices(indices))) => {
                Some(indices.as_ref().expect("QSA indices are non-null"))
            }
            _ => None,
        };
        let qsa_verify = match &qsa_plan {
            Some(Qwen4QsaPlan::VerifyIndices(rows)) => Some(rows.as_slice()),
            _ => None,
        };
        let qsa_prefill = match (mask.is_none(), &qsa_plan) {
            (true, Some(Qwen4QsaPlan::PrefillIndices { indices, valid })) => Some((
                indices.as_ref().expect("QSA prefill indices are non-null"),
                valid.as_ref().expect("QSA prefill validity is non-null"),
            )),
            _ => None,
        };
        let mask = qsa_mask.or(mask);

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
                apply_rotary_pos_emb(&query_rotary, &key_rotary, &cosine, &sine);
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

        let attn_out = if let Some(rows) = qsa_verify {
            cache.update(keys, values);
            let mut outputs = Vec::with_capacity(rows.len());
            for (row, indices) in rows.iter().enumerate() {
                let query = mlxcel_core::slice(
                    &queries,
                    &[0, 0, row as i32, 0],
                    &[b, self.num_heads, row as i32 + 1, self.head_dim],
                );
                let (cache_k, cache_v) = cache.fetch_selected(indices);
                outputs.push(unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &query,
                        &cache_k,
                        &cache_v,
                        self.scale,
                        std::ptr::null(),
                        0.0,
                        0,
                    )
                });
            }
            mlxcel_core::concatenate_owned(&outputs, 2)
        } else if let Some((indices, valid)) = qsa_prefill {
            let (cache_k, cache_v) = cache.update_and_fetch(keys, values);
            mlxcel_core::qsa_sparse_prefill_attention(
                &queries,
                &cache_k,
                &cache_v,
                indices,
                valid,
                self.scale,
            )
        } else if let Some(indices) = qsa_indices {
            let (cache_k, cache_v) = cache.update_and_fetch_selected(keys, values, indices);
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
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

        // Apply sigmoid gating to output
        let gate_sigmoid = mlxcel_core::sigmoid(&gate);
        let gated = mlxcel_core::multiply(&output, &gate_sigmoid);
        (gated, captured_query)
    }

    #[cfg(test)]
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Self::from_weights_with_norm_mode(weights, config, prefix, false)
    }

    pub(crate) fn from_weights_centered(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Self::from_weights_with_norm_mode(weights, config, prefix, true)
    }

    fn from_weights_with_norm_mode(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        prefix: &str,
        centered_norm: bool,
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

        let norm_weight = |name: &str| {
            weights
                .get(name)
                .map(|weight| {
                    if centered_norm {
                        let ones = mlxcel_core::ones(
                            &mlxcel_core::array_shape(weight),
                            mlxcel_core::array_dtype(weight),
                        );
                        mlxcel_core::add(weight, &ones)
                    } else {
                        mlxcel_core::copy(weight)
                    }
                })
                .ok_or_else(|| format!("Missing norm weight: {name}"))
        };
        let q_norm_weight = norm_weight(&format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = norm_weight(&format!("{}.k_norm.weight", prefix))?;

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
            indexer: if centered_norm && config.indexer_n_heads > 0 {
                Some(Qwen4QsaIndexer::from_weights(weights, config, prefix)?)
            } else {
                None
            },
        })
    }
}

// Dense MLP.
enum MlpInputProjections {
    Separate {
        gate: UnifiedLinear,
        up: UnifiedLinear,
    },
    Fused {
        projection: UnifiedLinear,
        intermediate_size: i32,
    },
}

fn fuse_mlp_input_projections(
    weights: &WeightMap,
    gate_prefix: &str,
    up_prefix: &str,
    gate: UnifiedLinear,
    up: UnifiedLinear,
) -> MlpInputProjections {
    let can_drop_linear_biases = !weights.contains_key(&format!("{gate_prefix}.bias"))
        && !weights.contains_key(&format!("{up_prefix}.bias"));
    let fused = (|| {
        let (gate_weight, up_weight) = (gate.quantized_weight()?, up.quantized_weight()?);
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
            projection: UnifiedLinear::new(
                QuantizedWeight::new(
                    weight,
                    scales,
                    biases,
                    gate_weight.group_size,
                    gate_weight.bits,
                ),
                None,
            ),
            intermediate_size: mlxcel_core::array_shape(&gate_weight.weight)[0],
        })
    })();

    fused.unwrap_or(MlpInputProjections::Separate { gate, up })
}

/// Dense MLP layer
pub(crate) struct Mlp {
    input_projections: MlpInputProjections,
    down_proj: UnifiedLinear,
}

impl Mlp {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gated = self.forward_hidden(x);
        self.down_proj.forward(&gated)
    }

    pub(crate) fn forward_hidden(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (gate, up) = match &self.input_projections {
            MlpInputProjections::Separate { gate, up } => (gate.forward(x), up.forward(x)),
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
        mlxcel_core::compiled_swiglu_activation(&gate, &up)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gate_prefix = format!("{}.gate_proj", prefix);
        let up_prefix = format!("{}.up_proj", prefix);
        let down_prefix = format!("{}.down_proj", prefix);
        let (gate_group_size, gate_bits) = config.quant_params(&gate_prefix);
        let (up_group_size, up_bits) = config.quant_params(&up_prefix);
        let (down_group_size, down_bits) = config.quant_params(&down_prefix);
        let gate = UnifiedLinear::from_weights(weights, &gate_prefix, gate_group_size, gate_bits)?;
        let up = UnifiedLinear::from_weights(weights, &up_prefix, up_group_size, up_bits)?;

        Ok(Self {
            input_projections: fuse_mlp_input_projections(
                weights,
                &gate_prefix,
                &up_prefix,
                gate,
                up,
            ),
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

    fn unequal_width_attention() -> Qwen4Attention {
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
        insert_f32(&mut weights, "self_attn.q_norm.weight", &[HEAD_DIM], 1.0);
        insert_f32(&mut weights, "self_attn.k_norm.weight", &[HEAD_DIM], 1.0);

        Qwen4Attention::from_weights(
            &weights,
            &Qwen4AttentionConfig {
                num_attention_heads: NUM_HEADS as usize,
                num_key_value_heads: NUM_KV_HEADS as usize,
                head_dim: HEAD_DIM as usize,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                quantization: None,
                mrope_section: vec![1, 1, 1],
                indexer_n_heads: 0,
                indexer_kv_heads: 0,
                indexer_head_dim: 0,
                indexer_budget: 0,
                indexer_compress_ratio: 1,
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
        let ordinary = attention.forward_with_position_ids(&input, &mut ordinary_cache, None, None);
        assert_eq!(mlxcel_core::array_shape(&ordinary), vec![1, 2, 3]);

        let mut verify_cache = KVCache::new();
        let verify = attention.forward_verify(&input, &mut verify_cache, None, None);
        assert_eq!(mlxcel_core::array_shape(&verify), vec![1, 2, 3]);
    }

    #[test]
    fn sparse_prefill_streams_selected_values_without_gathering() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        let queries = mlxcel_core::zeros(&[1, 2, 2, 4], mlxcel_core::dtype::FLOAT32);
        let keys = mlxcel_core::zeros(&[1, 1, 5, 4], mlxcel_core::dtype::FLOAT32);
        let values = mlxcel_core::from_slice_f32(
            &[
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 4.0,
            ],
            &[1, 1, 5, 4],
        );
        let indices = mlxcel_core::from_slice_i32(&[0, 2, 4, 1, 3, 4], &[1, 2, 3]);
        let valid = mlxcel_core::ones(&[1, 2, 3], mlxcel_core::dtype::BOOL);
        let actual = mlxcel_core::qsa_sparse_prefill_attention(
            &queries, &keys, &values, &indices, &valid, 0.5,
        );
        let expected = mlxcel_core::from_slice_f32(
            &[
                2.0,
                2.0,
                2.0,
                2.0,
                8.0 / 3.0,
                8.0 / 3.0,
                8.0 / 3.0,
                8.0 / 3.0,
                2.0,
                2.0,
                2.0,
                2.0,
                8.0 / 3.0,
                8.0 / 3.0,
                8.0 / 3.0,
                8.0 / 3.0,
            ],
            &[1, 2, 2, 4],
        );
        let close = mlxcel_core::allclose(&actual, &expected, 1e-5, 1e-5);
        mlxcel_core::eval(&close);
        assert!(mlxcel_core::item_bool(&close));
    }
}
