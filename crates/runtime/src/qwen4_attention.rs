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
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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
                let arrays: Vec<*const MlxArray> = [
                    cache.conv_state.as_deref(),
                    cache.state_cache.as_deref(),
                    cache.ple_conv_state.as_deref(),
                ]
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

#[derive(Default)]
pub(crate) struct Qwen4PrefillPolicy {
    force_reference: AtomicBool,
    approximate_used: AtomicBool,
}

impl Qwen4PrefillPolicy {
    pub(crate) fn set_force_reference(&self, enabled: bool) {
        self.force_reference.store(enabled, Ordering::Relaxed);
        if enabled {
            self.approximate_used.store(false, Ordering::Relaxed);
        }
    }

    pub(crate) fn approximate_used(&self) -> bool {
        self.approximate_used.load(Ordering::Relaxed)
    }

    pub(crate) fn reset(&self) {
        self.force_reference.store(false, Ordering::Relaxed);
        self.approximate_used.store(false, Ordering::Relaxed);
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
    prefill_policy: Arc<Qwen4PrefillPolicy>,
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

const QSA_PREFILL_QUERY_TILE: i32 = 128;
const QSA_PREFILL_BLOCK_TILE: i32 = 2048;

fn qsa_ranked_tile(
    query: &MlxArray,
    pooled: &MlxArray,
    complete_counts: Option<&MlxArray>,
    block_start: i32,
    block_end: i32,
    complete_blocks: i32,
    head_dim: i32,
) -> UniquePtr<MlxArray> {
    let pooled_t = mlxcel_core::transpose_axes(pooled, &[0, 1, 3, 2]);
    let scores = mlxcel_core::matmul(query, &pooled_t);
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    let scores = mlxcel_core::sum_axis(&mlxcel_core::maximum(&scores, &zero), 1, false);
    let scores = mlxcel_core::multiply_scalar(&scores, 1.0 / (head_dim as f32).sqrt());
    // Quantization stabilizes the score bin; the global block ID makes every
    // rank unique and preserves the recent-first tie break across tiles.
    let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
    let quantized = mlxcel_core::astype(
        &mlxcel_core::floor(&mlxcel_core::add(
            &mlxcel_core::multiply_scalar(&scores, 256.0),
            &half,
        )),
        mlxcel_core::dtype::INT32,
    );
    let rank_stride = mlxcel_core::from_slice_i32(&[complete_blocks + 1], &[1]);
    let block_ids = mlxcel_core::arange_i32(block_start, block_end, 1);
    let block_ids = mlxcel_core::reshape(&block_ids, &[1, 1, block_end - block_start]);
    let ranked = mlxcel_core::add(&mlxcel_core::multiply(&quantized, &rank_stride), &block_ids);
    match complete_counts {
        Some(complete_counts) => {
            let valid_blocks = mlxcel_core::less(&block_ids, complete_counts);
            let invalid_rank = mlxcel_core::from_slice_i32(&[i32::MIN], &[1]);
            mlxcel_core::where_cond(&valid_blocks, &ranked, &invalid_rank)
        }
        None => ranked,
    }
}

fn qsa_topk_block_ids(
    ranked: &MlxArray,
    block_topk: i32,
    complete_blocks: i32,
) -> UniquePtr<MlxArray> {
    let selected = mlxcel_core::topk(ranked, block_topk, -1);
    let selected = mlxcel_core::negative(&mlxcel_core::sort(&mlxcel_core::negative(&selected), -1));
    let rank_stride = mlxcel_core::from_slice_i32(&[complete_blocks + 1], &[1]);
    mlxcel_core::remainder(&selected, &rank_stride)
}

fn qsa_tiled_topk_block_ids(
    query: &MlxArray,
    pooled: &MlxArray,
    complete_counts: &MlxArray,
    block_topk: i32,
    query_tile_size: i32,
    block_tile_size: i32,
) -> UniquePtr<MlxArray> {
    let query_shape = mlxcel_core::array_shape(query);
    let batch = query_shape[0];
    let n_heads = query_shape[1];
    let sequence = query_shape[2];
    let head_dim = query_shape[3];
    let complete_blocks = mlxcel_core::array_shape(pooled)[2];
    let mut query_tiles =
        Vec::with_capacity(((sequence + query_tile_size - 1) / query_tile_size) as usize);
    for query_start in (0..sequence).step_by(query_tile_size as usize) {
        let query_end = (query_start + query_tile_size).min(sequence);
        let query_tile = mlxcel_core::slice(
            query,
            &[0, 0, query_start, 0],
            &[batch, n_heads, query_end, head_dim],
        );
        let complete_counts =
            mlxcel_core::slice(complete_counts, &[0, query_start, 0], &[1, query_end, 1]);
        let mut running: Option<UniquePtr<MlxArray>> = None;
        for block_start in (0..complete_blocks).step_by(block_tile_size as usize) {
            let block_end = (block_start + block_tile_size).min(complete_blocks);
            let pooled_tile = mlxcel_core::slice(
                pooled,
                &[0, 0, block_start, 0],
                &[batch, 1, block_end, head_dim],
            );
            let ranked = qsa_ranked_tile(
                &query_tile,
                &pooled_tile,
                Some(&complete_counts),
                block_start,
                block_end,
                complete_blocks,
                head_dim,
            );
            let tile_topk = block_topk.min(block_end - block_start);
            let candidates = mlxcel_core::topk(&ranked, tile_topk, -1);
            running = Some(match running {
                Some(previous) => {
                    let merged = mlxcel_core::concatenate(&previous, &candidates, -1);
                    let merged_width = mlxcel_core::array_shape(&merged)[2];
                    mlxcel_core::topk(&merged, block_topk.min(merged_width), -1)
                }
                None => candidates,
            });
        }
        let ranked = running.expect("QSA tiling requires at least one complete block");
        query_tiles.push(qsa_topk_block_ids(&ranked, block_topk, complete_blocks));
    }
    let mut query_tiles = query_tiles.into_iter();
    let mut selected = query_tiles
        .next()
        .expect("QSA tiling requires at least one query");
    for tile in query_tiles {
        selected = mlxcel_core::concatenate(&selected, &tile, 1);
    }
    selected
}

impl Qwen4QsaIndexer {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        attention_prefix: &str,
        prefill_policy: Arc<Qwen4PrefillPolicy>,
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
            prefill_policy,
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
        cache.set_auxiliary_block_size(self.compress_ratio);
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
        let raw_dtype = mlxcel_core::array_dtype(&raw_keys);
        cache
            .append_auxiliary_keys(past_len, raw_keys)
            .unwrap_or_else(|error| panic!("QSA raw append invariant failed: {error}"));
        let key_len = past_len
            .checked_add(sequence)
            .expect("QSA absolute key length overflowed i32");
        let complete_blocks = key_len / self.compress_ratio;

        let query = self.q_norm.forward(&query);
        let query = mlxcel_core::transpose_axes(&query, &[0, 2, 1, 3]);
        let query_positions = (past_len..past_len + sequence).collect::<Vec<_>>();
        let query = self.apply_rope(&query, &query_positions, batch);

        let complete_key_len = complete_blocks * self.compress_ratio;
        let cached_blocks = if let Some(keys) = cache.auxiliary_block_keys.as_deref() {
            let shape = mlxcel_core::array_shape(keys);
            assert!(
                shape[0] == batch
                    && shape[1] == 1
                    && shape[3] == self.head_dim
                    && cache.auxiliary_block_len() <= shape[2],
                "QSA summary geometry is incompatible with the active indexer: {shape:?}"
            );
            cache.truncate_auxiliary_blocks(complete_blocks);
            cache.auxiliary_block_len()
        } else {
            cache.truncate_auxiliary_blocks(0);
            0
        };
        if cached_blocks < complete_blocks {
            let new_block_count = complete_blocks - cached_blocks;
            let pooled = cache
                .auxiliary_keys_absolute(
                    cached_blocks * self.compress_ratio,
                    complete_key_len,
                )
                .unwrap_or_else(|error| {
                    panic!("QSA discarded raw rows required for summary repair: {error}")
                });
            let pooled = mlxcel_core::reshape(
                &pooled,
                &[batch, new_block_count, self.compress_ratio, self.head_dim],
            );
            let pooled = mlxcel_core::mean_axis(
                &mlxcel_core::astype(&pooled, mlxcel_core::dtype::FLOAT32),
                2,
                false,
            );
            let pooled = mlxcel_core::astype(&pooled, raw_dtype);
            let pooled = self.k_norm.forward(&pooled);
            let pooled = mlxcel_core::expand_dims(&pooled, 1);
            let block_positions = (cached_blocks..complete_blocks)
                .map(|block| block * self.compress_ratio)
                .collect::<Vec<_>>();
            let pooled = self.apply_rope(&pooled, &block_positions, batch);
            cache.append_auxiliary_blocks(&pooled);
        }
        cache
            .prune_auxiliary_keys()
            .unwrap_or_else(|error| panic!("QSA raw-tail pruning invariant failed: {error}"));
        if complete_blocks <= self.block_topk {
            return None;
        }
        let pooled = cache
            .auxiliary_block_keys_view()
            .expect("QSA logical block keys were just installed");

        let query_f32 = mlxcel_core::astype(&query, mlxcel_core::dtype::FLOAT32);
        let pooled_f32 = mlxcel_core::astype(&pooled, mlxcel_core::dtype::FLOAT32);
        let query_ends = mlxcel_core::arange_i32(past_len + 1, past_len + sequence + 1, 1);
        let query_ends = mlxcel_core::reshape(&query_ends, &[1, sequence, 1]);
        let ratio = mlxcel_core::from_slice_i32(&[self.compress_ratio], &[1]);
        let complete_counts = mlxcel_core::floor_divide(&query_ends, &ratio);

        // Every tile uses the same globally unique integer rank, so retaining
        // each tile's top-k and repeatedly taking the top-k of their union is
        // exactly the global top-k. Query tiling only separates independent rows.
        let all_rows_have_topk = past_len / self.compress_ratio > self.block_topk;
        let selected = if all_rows_have_topk
            && sequence > QSA_PREFILL_QUERY_TILE
            && complete_blocks > QSA_PREFILL_BLOCK_TILE
        {
            qsa_tiled_topk_block_ids(
                &query_f32,
                &pooled_f32,
                &complete_counts,
                self.block_topk,
                QSA_PREFILL_QUERY_TILE,
                QSA_PREFILL_BLOCK_TILE,
            )
        } else {
            let ranked = qsa_ranked_tile(
                &query_f32,
                &pooled_f32,
                (sequence != 1).then_some(&*complete_counts),
                0,
                complete_blocks,
                complete_blocks,
                self.head_dim,
            );
            if all_rows_have_topk {
                qsa_topk_block_ids(&ranked, self.block_topk, complete_blocks)
            } else {
                // Early prefill rows can contain the i32::MIN invalid sentinel,
                // whose remainder is not an index. Retain index-producing
                // argpartition for those rows.
                let selected = mlxcel_core::argpartition(&ranked, -self.block_topk, -1);
                mlxcel_core::slice(
                    &selected,
                    &[0, 0, complete_blocks - self.block_topk],
                    &[batch, sequence, complete_blocks],
                )
            }
        };
        // Attention consumes selected tokens in block order; ranking helpers
        // above keep descending global rank until this existing presentation step.
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
            && !self.prefill_policy.force_reference.load(Ordering::Relaxed)
            && past_len / self.compress_ratio > self.block_topk
            && matches!(
                cache.mode,
                KVCacheMode::Fp16 | KVCacheMode::Fp8 | KVCacheMode::Int8
            )
        {
            self.prefill_policy
                .approximate_used
                .store(true, Ordering::Relaxed);
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
            if cache.mode == KVCacheMode::Fp8 && mlxcel_core::metal_is_available() {
                let raw = cache.update_and_fetch_raw_fp8(keys, values);
                debug_assert_eq!(mlxcel_core::array_shape(&raw.keys)[2], raw.live_len);
                debug_assert_eq!(mlxcel_core::array_shape(&raw.values)[2], raw.live_len);
                mlxcel_core::qsa_sparse_prefill_attention_raw_fp8(
                    &queries,
                    &raw.keys,
                    &raw.values,
                    indices,
                    valid,
                    self.scale,
                )
            } else {
                // Keep the decoded path for FP16/INT8 and for FP8 whenever
                // raw-E4M3 Metal execution is unavailable.
                let (cache_k, cache_v) = cache.update_and_fetch(keys, values);
                mlxcel_core::qsa_sparse_prefill_attention(
                    &queries, &cache_k, &cache_v, indices, valid, self.scale,
                )
            }
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
        Self::from_weights_with_norm_mode(
            weights,
            config,
            prefix,
            false,
            Arc::new(Qwen4PrefillPolicy::default()),
        )
    }

    pub(crate) fn from_weights_centered(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        prefix: &str,
        prefill_policy: Arc<Qwen4PrefillPolicy>,
    ) -> Result<Self, String> {
        Self::from_weights_with_norm_mode(weights, config, prefix, true, prefill_policy)
    }

    fn from_weights_with_norm_mode(
        weights: &WeightMap,
        config: &Qwen4AttentionConfig,
        prefix: &str,
        centered_norm: bool,
        prefill_policy: Arc<Qwen4PrefillPolicy>,
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
                Some(Qwen4QsaIndexer::from_weights(
                    weights,
                    config,
                    prefix,
                    prefill_policy,
                )?)
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

    fn assert_arrays_equal(actual: &MlxArray, expected: &MlxArray) {
        let equal = mlxcel_core::array_equal(actual, expected, false);
        mlxcel_core::eval(&equal);
        assert!(mlxcel_core::item_bool(&equal));
    }

    fn exercise_tiled_qsa_ranking(materialize_inputs: bool) {
        const QUERIES: i32 = 7;
        const BLOCKS: i32 = 11;
        const HEADS: i32 = 2;
        const HEAD_DIM: i32 = 2;
        const TOPK: i32 = 3;

        let mut query_values = Vec::new();
        for head in 0..HEADS {
            for row in 0..QUERIES {
                if row == 0 {
                    query_values.extend_from_slice(&[0.0, 0.0]);
                } else {
                    query_values.extend_from_slice(&[
                        (row - 3) as f32,
                        if head == 0 {
                            (2 - row) as f32
                        } else {
                            (row - 5) as f32
                        },
                    ]);
                }
            }
        }
        let query = mlxcel_core::from_slice_f32(&query_values, &[1, HEADS, QUERIES, HEAD_DIM]);
        let pooled_values = (0..BLOCKS)
            .flat_map(|block| [(block - 5) as f32, ((block * 3) % 7 - 3) as f32])
            .collect::<Vec<_>>();
        let pooled = mlxcel_core::from_slice_f32(&pooled_values, &[1, 1, BLOCKS, HEAD_DIM]);
        let complete_counts =
            mlxcel_core::from_slice_i32(&[3, 4, 4, 7, 8, 10, 11], &[1, QUERIES, 1]);
        if materialize_inputs {
            mlxcel_core::eval(&query);
            mlxcel_core::eval(&pooled);
            mlxcel_core::eval(&complete_counts);
        }

        let ranked = qsa_ranked_tile(
            &query,
            &pooled,
            Some(&complete_counts),
            0,
            BLOCKS,
            BLOCKS,
            HEAD_DIM,
        );
        let monolithic = qsa_topk_block_ids(&ranked, TOPK, BLOCKS);
        if materialize_inputs {
            mlxcel_core::eval(&monolithic);
        }
        let tiled = qsa_tiled_topk_block_ids(&query, &pooled, &complete_counts, TOPK, 3, 4);
        assert_arrays_equal(&tiled, &monolithic);

        let first_row = mlxcel_core::slice(&tiled, &[0, 0, 0], &[1, 1, TOPK]);
        let expected_tie_order = mlxcel_core::from_slice_i32(&[2, 1, 0], &[1, 1, TOPK]);
        assert_arrays_equal(&first_row, &expected_tie_order);
    }

    #[test]
    fn qsa_tiled_ranking_matches_monolithic_across_causal_tails_lazy_and_eager() {
        exercise_tiled_qsa_ranking(false);
        exercise_tiled_qsa_ranking(true);
    }

    #[test]
    fn qsa_tiled_ranking_preserves_relu_and_global_tie_order() {
        let query = mlxcel_core::from_slice_f32(&[1.0, -1.0, -1.0, -1.0], &[1, 2, 2, 1]);
        let pooled = mlxcel_core::from_slice_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 1, 5, 1]);
        let complete_counts = mlxcel_core::from_slice_i32(&[5, 5], &[1, 2, 1]);
        let ranked = qsa_ranked_tile(&query, &pooled, Some(&complete_counts), 0, 5, 5, 1);
        let monolithic = qsa_topk_block_ids(&ranked, 3, 5);
        let tiled = qsa_tiled_topk_block_ids(&query, &pooled, &complete_counts, 3, 1, 2);
        let expected = mlxcel_core::from_slice_i32(&[4, 0, 3, 0, 1, 4], &[1, 2, 3]);
        assert_arrays_equal(&monolithic, &expected);
        assert_arrays_equal(&tiled, &expected);
    }
    fn qsa_rollback_indexer() -> Qwen4QsaIndexer {
        const HEAD_DIM: i32 = 2;
        let mut weights = WeightMap::new();
        weights.insert(
            "self_attn.indexer.index_qk_proj.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[4, HEAD_DIM]),
        );
        insert_f32(
            &mut weights,
            "self_attn.indexer.q_layernorm.weight",
            &[HEAD_DIM],
            0.0,
        );
        insert_f32(
            &mut weights,
            "self_attn.indexer.k_layernorm.weight",
            &[HEAD_DIM],
            0.0,
        );
        Qwen4QsaIndexer::from_weights(
            &weights,
            &Qwen4AttentionConfig {
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: HEAD_DIM as usize,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                quantization: None,
                mrope_section: vec![1, 0, 0],
                indexer_n_heads: 1,
                indexer_kv_heads: 1,
                indexer_head_dim: HEAD_DIM as usize,
                indexer_budget: 4,
                indexer_compress_ratio: 4,
            },
            "self_attn",
            Arc::new(Qwen4PrefillPolicy::default()),
        )
        .expect("synthetic QSA indexer")
    }

    fn qsa_token_rows(tokens: &[i32]) -> UniquePtr<MlxArray> {
        let rows = tokens
            .iter()
            .flat_map(|&token| [token as f32, ((token * 7) % 19 - 9) as f32])
            .collect::<Vec<_>>();
        mlxcel_core::from_slice_f32(&rows, &[1, tokens.len() as i32, 2])
    }

    fn append_qsa_rows(indexer: &Qwen4QsaIndexer, cache: &mut KVCache, tokens: &[i32]) {
        let _ = indexer.plan(&qsa_token_rows(tokens), cache);
        cache.offset += tokens.len() as i32;
    }

    fn assert_qsa_blocks_equal(actual: &KVCache, expected: &KVCache) {
        let actual = actual
            .auxiliary_block_keys_view()
            .expect("actual QSA summaries");
        let expected = expected
            .auxiliary_block_keys_view()
            .expect("expected QSA summaries");
        let equal = mlxcel_core::allclose(&actual, &expected, 0.0, 0.0);
        mlxcel_core::eval(&equal);
        assert!(mlxcel_core::item_bool(&equal));
    }

    fn exercise_qsa_rollback_reappend(materialize_before_rollback: bool) {
        let indexer = qsa_rollback_indexer();
        let mut cache = KVCache::new();
        cache
            .set_auxiliary_rollback_horizon(3)
            .expect("fresh QSA cache accepts rollback horizon");
        append_qsa_rows(&indexer, &mut cache, &[1, 2, 3, 4, 5, 6, 7, 8]);
        append_qsa_rows(&indexer, &mut cache, &[9, 10, 11, 12]);
        if materialize_before_rollback {
            cache.materialize_state();
        }

        assert_eq!(cache.trim(3), 3);
        if !materialize_before_rollback {
            cache.materialize_state();
        }
        assert_eq!(cache.auxiliary_block_len(), 2);

        append_qsa_rows(&indexer, &mut cache, &[90, 91, 92]);
        let first_rebuild = cache
            .auxiliary_block_keys_view()
            .expect("first rebuilt summaries");
        mlxcel_core::eval(&first_rebuild);
        let mut first_reference = KVCache::new();
        append_qsa_rows(&indexer, &mut first_reference, &[1, 2, 3, 4, 5, 6, 7, 8]);
        append_qsa_rows(&indexer, &mut first_reference, &[9, 90, 91, 92]);
        assert_qsa_blocks_equal(&cache, &first_reference);

        assert_eq!(cache.trim(3), 3);
        append_qsa_rows(&indexer, &mut cache, &[190, 191, 192]);
        let mut second_reference = KVCache::new();
        append_qsa_rows(&indexer, &mut second_reference, &[1, 2, 3, 4, 5, 6, 7, 8]);
        append_qsa_rows(&indexer, &mut second_reference, &[9, 190, 191, 192]);
        assert_qsa_blocks_equal(&cache, &second_reference);

        let second_rebuild = cache
            .auxiliary_block_keys_view()
            .expect("second rebuilt summaries");
        let changed = mlxcel_core::allclose(&first_rebuild, &second_rebuild, 0.0, 0.0);
        mlxcel_core::eval(&changed);
        assert!(
            !mlxcel_core::item_bool(&changed),
            "replacement token content must rebuild the rolled-back block",
        );
    }

    #[test]
    fn qsa_rollback_rebuilds_replaced_blocks_before_and_after_materialization() {
        exercise_qsa_rollback_reappend(true);
        exercise_qsa_rollback_reappend(false);
    }

    fn exercise_qsa_acceptance_alignments(materialize: bool) {
        let indexer = qsa_rollback_indexer();
        for horizon in [2_i32, 3, 5, 9] {
            for alignment in 0..4_i32 {
                let base_len = 12 + alignment;
                let base = (0..base_len).collect::<Vec<_>>();
                for accepted in 0..=horizon {
                    let mut cache = KVCache::new();
                    cache
                        .set_auxiliary_rollback_horizon(horizon)
                        .expect("fresh QSA cache accepts rollback horizon");
                    append_qsa_rows(&indexer, &mut cache, &base);
                    let verify = (0..horizon)
                        .map(|row| 1_000 + alignment * 100 + row)
                        .collect::<Vec<_>>();
                    append_qsa_rows(&indexer, &mut cache, &verify);
                    if materialize {
                        cache.materialize_state();
                    }
                    let rejected = horizon - accepted;
                    assert_eq!(cache.trim(rejected), rejected);
                    let replacements = (0..rejected)
                        .map(|row| 2_000 + accepted * 100 + row)
                        .collect::<Vec<_>>();
                    append_qsa_rows(&indexer, &mut cache, &replacements);

                    let mut expected_tokens = base.clone();
                    expected_tokens.extend_from_slice(&verify[..accepted as usize]);
                    expected_tokens.extend_from_slice(&replacements);
                    let mut expected = KVCache::new();
                    append_qsa_rows(&indexer, &mut expected, &expected_tokens);
                    assert_qsa_blocks_equal(&cache, &expected);

                    let (start, end) = cache.auxiliary_raw_tail_range();
                    assert_eq!(end, cache.offset);
                    assert!(end - start <= horizon + 3);
                    assert_eq!(start % 4, 0);
                }
            }
        }
    }

    #[test]
    fn qsa_tail_is_exact_for_all_ratio_alignments_mtp_depths_and_acceptance_counts() {
        exercise_qsa_acceptance_alignments(false);
        exercise_qsa_acceptance_alignments(true);
    }

    #[test]
    #[should_panic(expected = "exceeds configured rollback horizon")]
    fn qsa_deep_trim_is_rejected_before_state_changes() {
        let indexer = qsa_rollback_indexer();
        let mut cache = KVCache::new();
        cache
            .set_auxiliary_rollback_horizon(3)
            .expect("fresh QSA cache accepts rollback horizon");
        append_qsa_rows(&indexer, &mut cache, &(0..20).collect::<Vec<_>>());
        let _ = cache.trim(4);
    }

    #[test]
    fn qsa_snapshot_tail_restore_reuses_cached_suffix_exactly() {
        let indexer = qsa_rollback_indexer();
        let mut uninterrupted = KVCache::new();
        uninterrupted
            .set_auxiliary_rollback_horizon(5)
            .expect("fresh QSA cache accepts rollback horizon");
        append_qsa_rows(
            &indexer,
            &mut uninterrupted,
            &(0..19).collect::<Vec<_>>(),
        );
        let (start, end) = uninterrupted.auxiliary_raw_tail_range();
        let raw = uninterrupted
            .auxiliary_keys
            .as_deref()
            .map(mlxcel_core::copy)
            .expect("snapshot raw tail");
        let blocks = uninterrupted.auxiliary_block_keys_view();

        let mut restored = KVCache::new();
        restored.offset = end;
        restored.restore_auxiliary_block_keys(4, blocks);
        restored
            .restore_auxiliary_keys(start, end, 5, raw)
            .expect("restore bounded QSA raw tail");

        let suffix = [91, 92, 93, 94, 95];
        let uninterrupted_plan = indexer.plan(&qsa_token_rows(&suffix), &mut uninterrupted);
        let restored_plan = indexer.plan(&qsa_token_rows(&suffix), &mut restored);
        assert_qsa_plans_equal(uninterrupted_plan, restored_plan);
        uninterrupted.offset += suffix.len() as i32;
        restored.offset += suffix.len() as i32;
        assert_qsa_blocks_equal(&restored, &uninterrupted);
        assert_eq!(
            restored.auxiliary_raw_tail_range(),
            uninterrupted.auxiliary_raw_tail_range()
        );
    }

    #[test]
    fn qsa_64k_append_keeps_raw_storage_bounded_and_all_logical_summaries() {
        let indexer = qsa_rollback_indexer();
        let mut cache = KVCache::new();
        cache
            .set_auxiliary_rollback_horizon(9)
            .expect("fresh QSA cache accepts rollback horizon");
        append_qsa_rows(
            &indexer,
            &mut cache,
            &(0..65_536).collect::<Vec<_>>(),
        );
        let raw_rows = mlxcel_core::array_shape(
            cache
                .auxiliary_keys
                .as_deref()
                .expect("bounded QSA raw tail"),
        )[1];
        assert!(raw_rows <= 9 + 4 - 1);
        assert_eq!(cache.auxiliary_block_len(), 65_536 / 4);
    }

    fn assert_qsa_plans_equal(actual: Option<Qwen4QsaPlan>, expected: Option<Qwen4QsaPlan>) {
        let arrays_equal = |actual: &MlxArray, expected: &MlxArray| {
            assert_arrays_equal(actual, expected);
        };
        match (actual, expected) {
            (None, None) => {}
            (Some(Qwen4QsaPlan::Mask(actual)), Some(Qwen4QsaPlan::Mask(expected)))
            | (
                Some(Qwen4QsaPlan::DecodeIndices(actual)),
                Some(Qwen4QsaPlan::DecodeIndices(expected)),
            ) => arrays_equal(&actual, &expected),
            (
                Some(Qwen4QsaPlan::VerifyIndices(actual)),
                Some(Qwen4QsaPlan::VerifyIndices(expected)),
            ) => {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    arrays_equal(actual, expected);
                }
            }
            (
                Some(Qwen4QsaPlan::PrefillIndices {
                    indices: actual_indices,
                    valid: actual_valid,
                }),
                Some(Qwen4QsaPlan::PrefillIndices {
                    indices: expected_indices,
                    valid: expected_valid,
                }),
            ) => {
                arrays_equal(&actual_indices, &expected_indices);
                arrays_equal(&actual_valid, &expected_valid);
            }
            _ => panic!("reserved and unreserved QSA selected different plan variants"),
        }
    }

    fn exercise_reserved_qsa_matches_unreserved(materialize_between_chunks: bool) {
        let indexer = qsa_rollback_indexer();
        let mut reserved = KVCache::new();
        reserved.reserve_prefill_capacity(20);
        let mut unreserved = KVCache::new();

        for tokens in [&[1, 2, 3, 4, 5, 6, 7, 8][..], &[9, 10, 11, 12][..]] {
            let reserved_plan = indexer.plan(&qsa_token_rows(tokens), &mut reserved);
            let unreserved_plan = indexer.plan(&qsa_token_rows(tokens), &mut unreserved);
            assert_qsa_plans_equal(reserved_plan, unreserved_plan);
            reserved.offset += tokens.len() as i32;
            unreserved.offset += tokens.len() as i32;
            if materialize_between_chunks {
                reserved.materialize_state();
                unreserved.materialize_state();
            }
        }

        assert_eq!(reserved.auxiliary_block_len(), 3);
        assert_eq!(reserved.auxiliary_block_capacity(), 5);
        assert_eq!(unreserved.auxiliary_block_len(), 3);
        assert_eq!(unreserved.auxiliary_block_capacity(), 3);
        assert_qsa_blocks_equal(&reserved, &unreserved);
    }

    #[test]
    fn reserved_qsa_matches_unreserved_plans_lazy_and_eager() {
        exercise_reserved_qsa_matches_unreserved(false);
        exercise_reserved_qsa_matches_unreserved(true);
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

    #[test]
    fn raw_fp8_sparse_prefill_decoder_matches_mlx_for_every_byte() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        let bytes: Vec<u8> = (0..=u8::MAX).collect();
        let raw_values =
            mlxcel_core::from_bytes(&bytes, &[1, 1, 1, 256], mlxcel_core::dtype::UINT8);
        let raw_keys = mlxcel_core::zeros(&[1, 1, 1, 256], mlxcel_core::dtype::UINT8);
        let queries = mlxcel_core::zeros(&[1, 1, 1, 256], mlxcel_core::dtype::BFLOAT16);
        let indices = mlxcel_core::from_slice_i32(&[0], &[1, 1, 1]);
        let valid = mlxcel_core::ones(&[1, 1, 1], mlxcel_core::dtype::BOOL);
        let actual = mlxcel_core::qsa_sparse_prefill_attention_raw_fp8(
            &queries,
            &raw_keys,
            &raw_values,
            &indices,
            &valid,
            1.0,
        );
        let expected = mlxcel_core::from_fp8(&raw_values);
        let read_f32 = |array: &MlxArray| {
            let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
            mlxcel_core::eval(&array);
            mlxcel_core::array_to_raw_bytes(&array)
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
                .collect::<Vec<_>>()
        };
        assert_eq!(read_f32(&actual), read_f32(&expected));
    }

    #[test]
    fn raw_fp8_sparse_prefill_matches_decoded_for_sparse_multihead_rows() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        for eager in [false, true] {
            let query_values: Vec<f32> = (0..32)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.125)
                .collect();
            let key_values: Vec<f32> = (0..40)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.25)
                .collect();
            let mut value_values: Vec<f32> = (0..40)
                .map(|index| ((index % 17) as f32 - 8.0) * -0.375)
                .collect();
            // Token four is invalid padding in both query rows. Extreme signed
            // values make accidental UINT8-magnitude reads or invalid-row
            // accumulation dominate the result.
            value_values[16..20].copy_from_slice(&[448.0, -448.0, 448.0, -448.0]);

            let queries = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&query_values, &[1, 4, 2, 4]),
                mlxcel_core::dtype::BFLOAT16,
            );
            let compact_keys =
                mlxcel_core::to_fp8(&mlxcel_core::from_slice_f32(&key_values, &[1, 2, 5, 4]));
            let compact_values =
                mlxcel_core::to_fp8(&mlxcel_core::from_slice_f32(&value_values, &[1, 2, 5, 4]));
            // Match a step-reserved cache: the live five-token view retains an
            // eight-token physical head stride and must not be flattened.
            let padding = mlxcel_core::zeros(&[1, 2, 3, 4], mlxcel_core::dtype::UINT8);
            let padded_keys = mlxcel_core::concatenate(&compact_keys, &padding, 2);
            let padded_values = mlxcel_core::concatenate(&compact_values, &padding, 2);
            let raw_keys = mlxcel_core::slice(&padded_keys, &[0, 0, 0, 0], &[1, 2, 5, 4]);
            let raw_values = mlxcel_core::slice(&padded_values, &[0, 0, 0, 0], &[1, 2, 5, 4]);
            if eager {
                mlxcel_core::eval(&raw_keys);
                mlxcel_core::eval(&raw_values);
            }
            let decoded_keys = mlxcel_core::from_fp8(&raw_keys);
            let decoded_values = mlxcel_core::from_fp8(&raw_values);
            let indices = mlxcel_core::from_slice_i32(&[3, 0, 2, 4, 4, 2, 3, 1], &[1, 2, 4]);
            let valid = mlxcel_core::astype(
                &mlxcel_core::from_slice_i32(&[1, 1, 1, 0, 0, 1, 1, 1], &[1, 2, 4]),
                mlxcel_core::dtype::BOOL,
            );
            let actual = mlxcel_core::qsa_sparse_prefill_attention_raw_fp8(
                &queries,
                &raw_keys,
                &raw_values,
                &indices,
                &valid,
                0.5,
            );
            let expected = mlxcel_core::qsa_sparse_prefill_attention(
                &queries,
                &decoded_keys,
                &decoded_values,
                &indices,
                &valid,
                0.5,
            );
            let close = mlxcel_core::allclose(&actual, &expected, 1e-5, 1e-5);
            mlxcel_core::eval(&close);
            assert!(
                mlxcel_core::item_bool(&close),
                "raw FP8 sparse attention mismatch (eager={eager})"
            );
        }
    }
}
