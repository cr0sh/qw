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

//! DFlash2 block-diffusion drafter and distribution-preserving round loop.
//!
//! Rust port of the SGLang DFlash2 implementation
//! (`python/sglang/srt/models/dflash.py` at commit cf3813f4 — the
//! `DFlash2DraftModel` / `CandidateSelector` / `DFlashGroupedConv` /
//! `DFlashAttention` classes), which is the production reference the
//! `z-lab/Qwen3.8-27B-DFlash2` checkpoint is served with.
//!
//! What DFlash2 adds over the DFlash1 block-diffusion drafter:
//!
//! - **Candidate selector.** The drafter keeps the top-`selector_top_k`
//!   candidate tokens at every position (scored through the *target's* LM
//!   head) and traces one coherent path through them with a low-rank
//!   bilinear scorer over predecessor/successor codebooks. This raises the
//!   acceptance length vs. committing to one greedy token per position.
//! - **Two-tap dynamic convolution.** Every decoder layer wraps its
//!   attention and MLP sub-layers with a grouped dynamic depthwise conv
//!   (`prepare` convolves the input, `finish` convolves the output with a
//!   kernel projected from the input). The conv is causal *within the draft
//!   block* — position `p` sees positions `< p` of the same block — which
//!   keeps the draft from decaying toward the end of the block.
//!
//! The drafter checkpoint ships no `embed_tokens`/`lm_head`: tokens are
//! embedded with the target's embedding table and candidates are scored
//! with the target's LM head (SGLang `compute_candidates`). Stochastic
//! verification uses modified rejection sampling against the full target
//! vocabulary; compact draft support never restricts target output support.
//!
//! Apple Silicon precision rules apply (see
//! `docs/apple-silicon-precision.md`): the drafter runs in the checkpoint's
//! native bf16 — an f16 conversion overflows the drafter's gate*up
//! activations (values beyond f16's 65504 max poison the residual stream
//! with inf → NaN). Attention masks are the one exception — additive 0/-inf
//! f32 sentinels, matching every other mask builder in the repo.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use mlxcel_core::cache::{RotatingKVCacheSnapshotState, SequenceId};
use mlxcel_core::generate::{GenerationStopReason, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::{
    KVCache, QuantizedWeight, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr, concatenate, multiply_scalar};
use serde_json::Value;

use crate::portable_snapshot::{
    PortableArray, PortablePromptSnapshot, array_from_portable, array_to_portable,
    portable_model_state,
};
use crate::qwen3_5::Qwen35Model;
const DRAFT_QUANT_GROUP_SIZE: i32 = 128;
const DRAFT_QUANT_BITS: i32 = 4;

fn quantized_draft_linear(weights: &WeightMap, prefix: &str) -> Result<UnifiedLinear, String> {
    let weight_name = format!("{prefix}.weight");
    let dense = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let dense_shape = mlxcel_core::array_shape(dense);
    if dense_shape.last().copied().unwrap_or_default() % DRAFT_QUANT_GROUP_SIZE != 0 {
        return UnifiedLinear::from_weights(
            weights,
            prefix,
            DRAFT_QUANT_GROUP_SIZE,
            DRAFT_QUANT_BITS,
        );
    }
    let quantized = mlxcel_core::quantize_weights(dense, DRAFT_QUANT_GROUP_SIZE, DRAFT_QUANT_BITS);
    let weight = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    if !mlxcel_core::quantized_weights_has_biases(&quantized) {
        return Err(format!(
            "Affine quantization produced no biases for {prefix}"
        ));
    }
    let biases = mlxcel_core::quantized_weights_biases(&quantized);
    mlxcel_core::eval(&weight);
    mlxcel_core::eval(&scales);
    mlxcel_core::eval(&biases);
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|value| mlxcel_core::copy(value));
    Ok(UnifiedLinear::new(
        QuantizedWeight::new(
            weight,
            scales,
            biases,
            DRAFT_QUANT_GROUP_SIZE,
            DRAFT_QUANT_BITS,
        ),
        bias,
    ))
}

// ---------------------------------------------------------------------------
// # 1. DFlash2 config
// ---------------------------------------------------------------------------

/// DFlash2 drafter configuration.
///
/// Mirrors SGLang `parse_dflash_draft_config` / `DFlashDraftConfig`
/// (`python/sglang/srt/speculative/dflash_utils.py`): `dflash_config`
/// sub-object fields take precedence over the top-level / `text_config`
/// fields, matching the published `z-lab/Qwen3.8-27B-DFlash2` checkpoint
/// (`dflash_config` carries `block_size`, `conv_*`, `selector_*`,
/// `mask_token_id`, `target_layer_ids`).
#[derive(Debug, Clone, PartialEq, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DFlash2Config {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default = "default_mask_token_id")]
    pub mask_token_id: i32,
    #[serde(default)]
    pub target_layer_ids: Vec<usize>,
    #[serde(default = "default_num_target_layers")]
    pub num_target_layers: usize,

    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// `None` means "per layer type" (sliding layers causal, full-attention
    /// layers non-causal); the Qwen3.8-27B-DFlash2 checkpoint sets it to
    /// `false` explicitly, making every layer non-causal.
    #[serde(default)]
    pub is_causal: Option<bool>,

    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_conv_group_size")]
    pub conv_group_size: usize,
    #[serde(default = "default_selector_rank")]
    pub selector_rank: usize,
    #[serde(default = "default_selector_top_k")]
    pub selector_top_k: usize,

    #[serde(default = "default_output_multiplier")]
    pub output_multiplier: f32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
}

fn default_hidden_size() -> usize {
    5120
}
fn default_num_hidden_layers() -> usize {
    5
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_intermediate_size() -> usize {
    17_408
}
fn default_vocab_size() -> usize {
    248_320
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_max_position_embeddings() -> usize {
    262_144
}
fn default_block_size() -> usize {
    8
}
fn default_mask_token_id() -> i32 {
    248_070
}
fn default_num_target_layers() -> usize {
    64
}
fn default_conv_kernel_size() -> usize {
    2
}
fn default_conv_group_size() -> usize {
    16
}
fn default_selector_rank() -> usize {
    256
}
fn default_selector_top_k() -> usize {
    16
}
fn default_output_multiplier() -> f32 {
    1.0
}

impl DFlash2Config {
    /// Parse the drafter checkpoint's `config.json` (already parsed to a
    /// [`Value`]). Mirrors SGLang `parse_dflash_draft_config`: `dflash_config`
    /// sub-object first, then top-level / `text_config` fallbacks, and the
    /// SGLang validations (`sliding_window` required for sliding layers,
    /// `conv_group_size` divides `hidden_size`, `target_layer_ids` in
    /// range).
    pub fn from_json(config: &Value) -> Result<Self, String> {
        let root = config.as_object().ok_or("config.json must be an object")?;
        let text = root
            .get("text_config")
            .and_then(Value::as_object)
            .unwrap_or(root);
        let dflash = root
            .get("dflash_config")
            .and_then(Value::as_object)
            .ok_or("config.json must carry a dflash_config object")?;

        let get = |field: &str| {
            dflash
                .get(field)
                .or_else(|| text.get(field))
                .or_else(|| root.get(field))
        };
        let get_int = |field: &str, default: usize| -> Result<usize, String> {
            match get(field) {
                Some(value) => value
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("dflash config field {field:?} must be an integer")),
                None => Ok(default),
            }
        };
        let get_f32 = |field: &str, default: f32| -> Result<f32, String> {
            match get(field) {
                Some(value) => value
                    .as_f64()
                    .map(|n| n as f32)
                    .ok_or_else(|| format!("dflash config field {field:?} must be a number")),
                None => Ok(default),
            }
        };

        let hidden_size = get_int("hidden_size", default_hidden_size())?;
        let num_hidden_layers = get_int("num_hidden_layers", default_num_hidden_layers())?;
        let num_target_layers = get_int("num_target_layers", default_num_target_layers())?;
        let conv_group_size = get_int("conv_group_size", default_conv_group_size())?;
        let conv_kernel_size = get_int("conv_kernel_size", default_conv_kernel_size())?;
        let block_size = get_int("block_size", default_block_size())?;
        let mask_token_id = dflash
            .get("mask_token_id")
            .and_then(Value::as_i64)
            .map(|n| n as i32)
            .ok_or("dflash_config.mask_token_id is required")?;

        let target_layer_ids = match get("target_layer_ids") {
            Some(Value::Array(ids)) => {
                let mut resolved = Vec::with_capacity(ids.len());
                for id in ids {
                    let id = id
                        .as_u64()
                        .ok_or("dflash_config.target_layer_ids must be integers")?
                        as usize;
                    if id >= num_target_layers {
                        return Err(format!(
                            "target_layer_ids contains out-of-range layer id {id} \
                             (num_target_layers = {num_target_layers})"
                        ));
                    }
                    resolved.push(id);
                }
                if resolved.is_empty() {
                    return Err("target_layer_ids must be non-empty".to_owned());
                }
                resolved
            }
            _ => return Err("dflash_config.target_layer_ids is required".to_owned()),
        };

        let layer_types: Vec<String> = match get("layer_types") {
            Some(Value::Array(types)) => types
                .iter()
                .map(|t| {
                    t.as_str()
                        .map(str::to_owned)
                        .ok_or("layer_types must be strings")
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("config layer_types is required for the DFlash2 drafter".to_owned()),
        };
        if layer_types.len() != num_hidden_layers {
            return Err(format!(
                "layer_types must have one entry per draft layer: \
                 got {} entries for num_hidden_layers = {num_hidden_layers}",
                layer_types.len()
            ));
        }
        let sliding_window = match get("sliding_window") {
            Some(value) => {
                Some(value.as_u64().ok_or("sliding_window must be an integer")? as usize)
            }
            None => None,
        };
        if layer_types.iter().any(|t| t == "sliding_attention") && sliding_window.is_none() {
            return Err("sliding_attention layers require config.sliding_window".to_owned());
        }
        for layer_type in &layer_types {
            if layer_type != "full_attention" && layer_type != "sliding_attention" {
                return Err(format!(
                    "unsupported DFlash2 layer type {layer_type:?} \
                     (expected \"full_attention\" or \"sliding_attention\")"
                ));
            }
        }
        if conv_group_size == 0 || hidden_size % conv_group_size != 0 {
            return Err(format!(
                "conv_group_size={conv_group_size} must divide hidden_size={hidden_size}"
            ));
        }
        let is_causal = match get("is_causal") {
            Some(value) => Some(value.as_bool().ok_or("is_causal must be a boolean")?),
            None => None,
        };
        let selector_rank = get_int("selector_rank", default_selector_rank())?;
        let selector_top_k = get_int("selector_top_k", default_selector_top_k())?;
        if selector_rank == 0 {
            return Err("selector_rank must be greater than zero".to_owned());
        }
        if selector_top_k == 0 {
            return Err("selector_top_k must be greater than zero".to_owned());
        }

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads: get_int("num_attention_heads", default_num_attention_heads())?,
            num_key_value_heads: get_int("num_key_value_heads", default_num_key_value_heads())?,
            head_dim: get_int("head_dim", default_head_dim())?,
            intermediate_size: get_int("intermediate_size", default_intermediate_size())?,
            vocab_size: get_int("vocab_size", default_vocab_size())?,
            rms_norm_eps: get_f32("rms_norm_eps", default_rms_norm_eps())?,
            rope_theta: get_f32("rope_theta", default_rope_theta())?,
            max_position_embeddings: get_int(
                "max_position_embeddings",
                default_max_position_embeddings(),
            )?,
            block_size,
            mask_token_id,
            target_layer_ids,
            num_target_layers,
            layer_types,
            sliding_window,
            is_causal,
            conv_kernel_size,
            conv_group_size,
            selector_rank,
            selector_top_k,
            output_multiplier: get_f32("output_multiplier", default_output_multiplier())?,
            final_logit_softcapping: get("final_logit_softcapping")
                .and_then(Value::as_f64)
                .map(|n| n as f32),
        })
    }
}

// ---------------------------------------------------------------------------
// # 2. Grouped dynamic (two-tap) causal convolution
// ---------------------------------------------------------------------------

/// Grouped dynamic depthwise convolution across one DFlash block, ported from
/// SGLang `DFlashGroupedConv` + `_grouped_conv` (dflash.py).
///
/// Each sub-layer is wrapped: `prepare` convolves its input and returns the
/// kernel projected from that input; `finish` convolves the output with the
/// same kernel. The convolution is causal *within the draft block*: with the
/// SGLang position convention `pos & (block_size - 1)`, a position `p` sees
/// the values at `p - 1 .. p - (taps - 1)` of the same block (positions with
/// `p < tap` contribute nothing). The B=1 round loop drafts one block per
/// forward, so the positions are exactly the local row indices `0..L`.
pub struct DFlash2GroupedConv {
    /// `[2, taps, hidden]` — `[side, tap, channel]`, the layout the training
    /// export stores (`base_kernel`).
    base_kernel: UniquePtr<MlxArray>,
    /// `[2 * taps * groups, hidden]`.
    kernel_projection: UnifiedLinear,
    block_size: i32,
    taps: i32,
    group_size: i32,
    num_groups: i32,
}

impl DFlash2GroupedConv {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DFlash2Config,
    ) -> Result<Self, String> {
        let base_kernel = weights
            .get(&format!("{prefix}.base_kernel"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.base_kernel"))?;
        let kernel_projection =
            quantized_draft_linear(weights, &format!("{prefix}.kernel_projection"))?;
        let taps = config.conv_kernel_size as i32;
        Ok(Self {
            base_kernel,
            kernel_projection,
            block_size: config.block_size as i32,
            taps,
            group_size: config.conv_group_size as i32,
            num_groups: (config.hidden_size / config.conv_group_size) as i32,
        })
    }

    /// `DFlashGroupedConv.prepare`: convolve `hidden` with the input-side
    /// kernel and return the output-side kernel coefficients.
    pub fn prepare(&self, hidden: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let projected = self.kernel_projection.forward(hidden);
        let shape = mlxcel_core::array_shape(&projected);
        let coefficients = mlxcel_core::reshape(
            &projected,
            &[shape[0], shape[1], 2, self.taps, self.num_groups],
        );
        let side0 = mlxcel_core::slice(
            &coefficients,
            &[0, 0, 0, 0, 0],
            &[shape[0], shape[1], 1, self.taps, self.num_groups],
        );
        let side1 = mlxcel_core::slice(
            &coefficients,
            &[0, 0, 1, 0, 0],
            &[shape[0], shape[1], 2, self.taps, self.num_groups],
        );
        let side0 = mlxcel_core::reshape(&side0, &[shape[0], shape[1], self.taps, self.num_groups]);
        let side1 = mlxcel_core::reshape(&side1, &[shape[0], shape[1], self.taps, self.num_groups]);
        let base_shape = mlxcel_core::array_shape(&self.base_kernel);
        let base0 = mlxcel_core::slice(
            &self.base_kernel,
            &[0, 0, 0],
            &[1, self.taps, base_shape[2]],
        );
        (self.convolve(hidden, &side0, &base0), side1)
    }

    /// `DFlashGroupedConv.finish`: convolve `hidden` with the kernel returned
    /// by [`Self::prepare`].
    pub fn finish(&self, hidden: &MlxArray, dynamic: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let base1 = mlxcel_core::slice(&self.base_kernel, &[1, 0, 0], &[2, self.taps, shape[2]]);
        self.convolve(hidden, dynamic, &base1)
    }

    /// SGLang `_grouped_conv`: `coefficients = base + delta`, output is the
    /// sum over taps of `coefficients[tap] * shift(blocks, tap)` with the
    /// within-block position mask `(pos & (block_size - 1)) >= tap`.
    fn convolve(
        &self,
        hidden: &MlxArray,
        dynamic: &MlxArray,
        base_side: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let hidden_shape = mlxcel_core::array_shape(hidden);
        let batch = hidden_shape[0];
        let length = hidden_shape[1];
        let gs = self.group_size;
        let groups = self.num_groups;
        let blocks = mlxcel_core::reshape(hidden, &[batch, length, groups, gs]);
        let hidden_dtype = mlxcel_core::array_dtype(hidden);
        let base = if mlxcel_core::array_dtype(base_side) == hidden_dtype {
            mlxcel_core::copy(base_side)
        } else {
            mlxcel_core::astype(base_side, hidden_dtype)
        };
        // base [taps, hidden] -> [1, taps, groups, gs]
        let base = mlxcel_core::reshape(&base, &[1, self.taps, groups, gs]);

        // Position mask: with the SGLang position convention
        // `pos & (block_size - 1)`, a shifted tap contributes only where
        // `(pos & (block_size - 1)) >= tap`. The B=1 round loop drafts one
        // block per forward (`L <= block_size`), where the zero-padding below
        // already zeroes `pos < tap`; the mask is exact for multi-block
        // forwards too.
        let need_mask = length > self.block_size;
        let mut output = mlxcel_core::zeros(&[batch, length, groups, gs], hidden_dtype);

        for tap in 0..self.taps {
            let values = if tap == 0 {
                mlxcel_core::share(&blocks)
            } else {
                // SGLang `F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))`:
                // prepend `tap` zero rows to blocks[0 : L - tap], so row `j`
                // sees row `j - tap`.
                let tail =
                    mlxcel_core::slice(&blocks, &[0, 0, 0, 0], &[batch, length - tap, groups, gs]);
                let zero_block = mlxcel_core::zeros(&[batch, tap, groups, gs], hidden_dtype);
                mlxcel_core::concatenate(&zero_block, &tail, 1)
            };
            let mut values = values;
            if need_mask && tap > 0 {
                let rows: Vec<f32> = (0..length)
                    .map(|pos| {
                        if (pos & (self.block_size - 1)) >= tap {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let mask = mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(&rows, &[1, length, 1, 1]),
                    hidden_dtype,
                );
                values = mlxcel_core::multiply(&values, &mask);
            }
            // delta [B, L, taps, groups] -> [B, L, groups, 1]; the trailing
            // singleton broadcasts the per-group kernel over the group_size
            // channels (SGLang `delta.unsqueeze(-1)` against
            // `base.view(1, taps, groups, group_size)`).
            let delta_tap =
                mlxcel_core::slice(dynamic, &[0, 0, tap, 0], &[batch, length, tap + 1, groups]);
            let delta_tap = mlxcel_core::reshape(&delta_tap, &[batch, length, groups, 1]);
            let delta_tap = mlxcel_core::astype(&delta_tap, hidden_dtype);
            // base_tap [1, 1, groups, gs]
            let base_tap = mlxcel_core::slice(&base, &[0, tap, 0, 0], &[1, tap + 1, groups, gs]);
            let base_tap = mlxcel_core::reshape(&base_tap, &[1, 1, groups, gs]);
            // coefficients = base_tap + delta_tap, broadcast [B, L, groups, gs]
            let coefficients = mlxcel_core::add(&base_tap, &delta_tap);
            let contribution = mlxcel_core::multiply(&coefficients, &values);
            output = mlxcel_core::add(&output, &contribution);
        }
        mlxcel_core::reshape(&output, &hidden_shape)
    }
}

// ---------------------------------------------------------------------------
// # 3. Sliding-window split-projection attention
// ---------------------------------------------------------------------------

/// Drafter-side KV cache: a rotating sliding-window cache for
/// `sliding_attention` layers, a plain cache otherwise. Only context K/V is
/// ever written; proposal K/V is concatenated post-hoc and never cached.
pub enum DFlash2KVCache {
    Full(KVCache),
    Sliding(RotatingKVCache),
}

impl DFlash2KVCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::Full(cache) => cache.offset,
            Self::Sliding(cache) => cache.offset,
        }
    }

    pub fn update_and_fetch(
        &mut self,
        new_k: UniquePtr<MlxArray>,
        new_v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Self::Full(cache) => cache.update_and_fetch(new_k, new_v),
            Self::Sliding(cache) => cache.update_and_fetch(new_k, new_v),
        }
    }

    /// Rewind to a committed prefix. The round loop only ever writes
    /// committed context rows into the cache, so this is normally a no-op;
    /// it guards the sliding-trim edge where the in-attention window drop
    /// advanced the offset past the committed count.
    pub fn trim_to_committed(&mut self, committed: i32) {
        let trim = self.offset().saturating_sub(committed);
        if trim <= 0 {
            return;
        }
        match self {
            Self::Full(cache) => {
                cache.trim(trim);
            }
            Self::Sliding(cache) => {
                cache.trim(trim);
            }
        }
    }
}

struct Dflash2SlidingCacheSnapshot {
    state: RotatingKVCacheSnapshotState,
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
}

impl Dflash2SlidingCacheSnapshot {
    fn capture(caches: &[DFlash2KVCache]) -> Option<Vec<Self>> {
        caches
            .iter()
            .map(|cache| {
                let DFlash2KVCache::Sliding(cache) = cache else {
                    return None;
                };
                Some(Self {
                    state: cache.snapshot_state(),
                    keys: mlxcel_core::share(cache.keys.as_ref()?),
                    values: mlxcel_core::share(cache.values.as_ref()?),
                })
            })
            .collect()
    }

    fn restore_all(snapshots: &[Self]) -> Result<Vec<DFlash2KVCache>, String> {
        snapshots
            .iter()
            .map(|snapshot| {
                let mut cache = RotatingKVCache::new(snapshot.state.max_size);
                cache.restore_fp16_snapshot_state(
                    snapshot.state,
                    Some(mlxcel_core::share(&snapshot.keys)),
                    Some(mlxcel_core::share(&snapshot.values)),
                )?;
                Ok(DFlash2KVCache::Sliding(cache))
            })
            .collect()
    }

    fn materialize_and_detach_all(snapshots: &[Self]) {
        let arrays = snapshots
            .iter()
            .flat_map(|snapshot| [&*snapshot.keys, &*snapshot.values])
            .map(|array| array as *const MlxArray)
            .collect::<Vec<_>>();
        unsafe {
            mlxcel_core::eval_all(&arrays);
            mlxcel_core::detach_all(&arrays);
        }
    }
}

// Steady-state decode already projects only `accepted + 1` newly committed
// target rows and retains their FC-to-K/V result in the live caches. This one
// bounded memo covers the remaining expensive case: rebuilding a restored
// prompt snapshot at the start of every repeated cached generation.
struct Dflash2ProjectedPrefix {
    snapshot_id: u64,
    committed_suffix: Vec<i32>,
    caches: Vec<Dflash2SlidingCacheSnapshot>,
}

impl Dflash2ProjectedPrefix {
    fn restore(&self) -> Result<Vec<DFlash2KVCache>, String> {
        Dflash2SlidingCacheSnapshot::restore_all(&self.caches)
    }

    fn materialize_and_detach(&self) {
        Dflash2SlidingCacheSnapshot::materialize_and_detach_all(&self.caches);
    }
}

/// DFlash2 split-projection attention, ported from SGLang `DFlashAttention`
/// (dflash.py). The attention over the draft block is non-causal (the
/// block-diffusion forward denoises all block positions at once); sliding
/// layers additionally bound the attended context through the rotating
/// cache, and `config.is_causal` may force causal attention.
pub struct DFlash2Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_base: f32,
    is_sliding: bool,
    sliding_window: i32,
    is_causal: bool,
}

impl DFlash2Attention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DFlash2Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let q_proj = quantized_draft_linear(weights, &format!("{prefix}.q_proj"))?;
        let k_proj = quantized_draft_linear(weights, &format!("{prefix}.k_proj"))?;
        let v_proj = quantized_draft_linear(weights, &format!("{prefix}.v_proj"))?;
        let o_proj = quantized_draft_linear(weights, &format!("{prefix}.o_proj"))?;
        let q_norm_w = weights
            .get(&format!("{prefix}.q_norm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.q_norm.weight"))?;
        let k_norm_w = weights
            .get(&format!("{prefix}.k_norm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.k_norm.weight"))?;

        let is_sliding = config
            .layer_types
            .get(layer_idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(false);
        let sliding_window = config.sliding_window.unwrap_or(0) as i32;
        // SGLang `_get_dflash_attention_type`: an explicit `is_causal` wins;
        // otherwise sliding layers default to causal (DECODER) and
        // full-attention layers to non-causal (ENCODER_ONLY).
        let is_causal = config.is_causal.unwrap_or(is_sliding);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_w, config.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_w, config.rms_norm_eps),
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            scale: (config.head_dim as f32).powf(-0.5),
            rope_base: config.rope_theta,
            is_sliding,
            sliding_window,
            is_causal,
        })
    }

    /// Forward with split projections.
    ///
    /// - `x` — proposal sequence `[B, L, hidden_size]`.
    /// - `x_ctx` — newly committed context `[B, S, hidden_size]`, or `None`
    ///   when an exact projected prefix has already populated `cache`.
    /// - `cache` — this layer's K/V cache; updated only with context-side K/V.
    ///
    /// Returns `[B, L, hidden_size]`.
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: Option<&MlxArray>,
        cache: &mut DFlash2KVCache,
    ) -> UniquePtr<MlxArray> {
        let x_shape = mlxcel_core::array_shape(x);
        let b = x_shape[0];
        let l = x_shape[1];

        // Proposal projections are never cached. The projected-prefix path
        // below restores only committed context K/V, so rejected proposals
        // cannot leak into a later request or round.
        let queries = self.q_proj.forward(x);
        let prop_keys = self.k_proj.forward(x);
        let prop_values = self.v_proj.forward(x);
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.n_heads, self.head_dim]);
        let prop_keys = mlxcel_core::reshape(&prop_keys, &[b, l, self.n_kv_heads, self.head_dim]);
        let prop_values =
            mlxcel_core::reshape(&prop_values, &[b, l, self.n_kv_heads, self.head_dim]);
        let queries = self.q_norm.forward(&queries);
        let prop_keys = self.k_norm.forward(&prop_keys);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let prop_keys = mlxcel_core::transpose_axes(&prop_keys, &[0, 2, 1, 3]);
        let prop_values = mlxcel_core::transpose_axes(&prop_values, &[0, 2, 1, 3]);

        let (keys, values, s, proposal_offset) = if let Some(x_ctx) = x_ctx {
            let ctx_shape = mlxcel_core::array_shape(x_ctx);
            let mut s = ctx_shape[1];
            let mut x_ctx = mlxcel_core::copy(x_ctx);
            if self.is_sliding && s > self.sliding_window - 1 {
                let skip = s - (self.sliding_window - 1);
                x_ctx = mlxcel_core::slice(&x_ctx, &[0, skip, 0], &[b, s, ctx_shape[2]]);
                s = self.sliding_window - 1;
                if let DFlash2KVCache::Sliding(cache) = cache {
                    cache.offset += skip;
                }
            }

            let past_offset = cache.offset();
            let ctx_keys = self.k_proj.forward(&x_ctx);
            let ctx_values = self.v_proj.forward(&x_ctx);
            let ctx_keys = mlxcel_core::reshape(&ctx_keys, &[b, s, self.n_kv_heads, self.head_dim]);
            let ctx_values =
                mlxcel_core::reshape(&ctx_values, &[b, s, self.n_kv_heads, self.head_dim]);
            let ctx_keys = self.k_norm.forward(&ctx_keys);
            let ctx_keys = mlxcel_core::transpose_axes(&ctx_keys, &[0, 2, 1, 3]);
            let ctx_values = mlxcel_core::transpose_axes(&ctx_values, &[0, 2, 1, 3]);
            let ctx_keys = mlxcel_core::fast_rope(
                &ctx_keys,
                self.head_dim,
                false,
                self.rope_base,
                1.0,
                past_offset,
            );
            let (keys, values) = cache.update_and_fetch(ctx_keys, ctx_values);
            (keys, values, s, past_offset + s)
        } else {
            // A no-context call is valid only immediately after restoring the
            // reusable projected prefix. Its sliding snapshots are captured in
            // chronological order before any ring wrap.
            let DFlash2KVCache::Sliding(cache) = cache else {
                unreachable!("only sliding DFlash2 caches have projected-prefix snapshots");
            };
            let keys = mlxcel_core::share(
                cache
                    .keys
                    .as_ref()
                    .expect("projected DFlash2 prefix must contain keys"),
            );
            let values = mlxcel_core::share(
                cache
                    .values
                    .as_ref()
                    .expect("projected DFlash2 prefix must contain values"),
            );
            let s = mlxcel_core::array_shape(&keys)[2];
            (keys, values, s, cache.offset)
        };

        let queries = mlxcel_core::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            proposal_offset,
        );
        let prop_keys = mlxcel_core::fast_rope(
            &prop_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            proposal_offset,
        );
        let keys_combined = mlxcel_core::concatenate(&keys, &prop_keys, 2);
        let values_combined = mlxcel_core::concatenate(&values, &prop_values, 2);

        let mask = self.build_mask(l, s, &keys_combined);
        let mask_ptr = mask
            .as_ref()
            .map(|m| &**m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let attn = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &queries,
                &keys_combined,
                &values_combined,
                self.scale,
                mask_ptr,
            )
        };

        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&attn)
    }

    /// Additive `[1, 1, L, total]` f32 mask (0.0 = attend, -inf = block),
    /// or `None` when no key needs masking. `keys_combined` is
    /// `[B, H, total, D]`; `S` is the context length after trimming.
    fn build_mask(&self, l: i32, s: i32, keys_combined: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        let total = mlxcel_core::array_shape(keys_combined)[2];
        let need_mask = self.is_causal || (self.is_sliding && s + l > self.sliding_window);
        if !need_mask {
            return None;
        }
        let mut mask = vec![0.0_f32; (l * total) as usize];
        for q in 0..l {
            for key in 0..total {
                let visible = if self.is_sliding {
                    let in_context = key < s;
                    let within_window = (s + q - key) < self.sliding_window;
                    let in_block = key >= s;
                    let block_ok = !self.is_causal || (key <= q + s);
                    (in_context && within_window) || (in_block && block_ok)
                } else {
                    key <= q + s
                };
                if !visible {
                    mask[(q * total + key) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        Some(mlxcel_core::from_slice_f32(&mask, &[1, 1, l, total]))
    }
}

// ---------------------------------------------------------------------------
// # 4. MLP + decoder layer
// ---------------------------------------------------------------------------

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))` (SGLang `DFlashMLP`).
pub struct DFlash2Mlp {
    gate: UnifiedLinear,
    up: UnifiedLinear,
    down: UnifiedLinear,
}

impl DFlash2Mlp {
    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let gate = quantized_draft_linear(weights, &format!("{prefix}.gate_proj"))?;
        let up = quantized_draft_linear(weights, &format!("{prefix}.up_proj"))?;
        let down = quantized_draft_linear(weights, &format!("{prefix}.down_proj"))?;
        Ok(Self { gate, up, down })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = mlxcel_core::silu(&self.gate.forward(x));
        let up = self.up.forward(x);
        let product = mlxcel_core::multiply(&gate, &up);
        self.down.forward(&product)
    }
}

/// One DFlash2 decoder layer: pre-norm attention and post-norm MLP, both
/// optionally wrapped by a grouped dynamic conv, ported from SGLang
/// `DFlashDecoderLayer`.
pub struct DFlash2DecoderLayer {
    self_attn: DFlash2Attention,
    mlp: DFlash2Mlp,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    attention_conv: Option<DFlash2GroupedConv>,
    mlp_conv: Option<DFlash2GroupedConv>,
}

impl DFlash2DecoderLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DFlash2Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let self_attn = DFlash2Attention::from_weights(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
        )?;
        let mlp = DFlash2Mlp::from_weights(weights, &format!("{prefix}.mlp"))?;
        let input_norm_w = weights
            .get(&format!("{prefix}.input_layernorm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.input_layernorm.weight"))?;
        let post_norm_w = weights
            .get(&format!("{prefix}.post_attention_layernorm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.post_attention_layernorm.weight"))?;
        let (attention_conv, mlp_conv) = if config.conv_kernel_size > 0 {
            (
                Some(DFlash2GroupedConv::from_weights(
                    weights,
                    &format!("{prefix}.attention_conv"),
                    config,
                )?),
                Some(DFlash2GroupedConv::from_weights(
                    weights,
                    &format!("{prefix}.mlp_conv"),
                    config,
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_w, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_w, config.rms_norm_eps),
            attention_conv,
            mlp_conv,
        })
    }

    /// SGLang `DFlashDecoderLayer.forward`: conv-wrapped attention and MLP
    /// sub-layers on residual branches.
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: Option<&MlxArray>,
        cache: &mut DFlash2KVCache,
    ) -> UniquePtr<MlxArray> {
        let residual = x;
        let h = self.input_layernorm.forward(x);
        let attn_out = if let Some(conv) = &self.attention_conv {
            let (convolved, kernel) = conv.prepare(&h);
            let attn = self.self_attn.forward(&convolved, x_ctx, cache);
            conv.finish(&attn, &kernel)
        } else {
            self.self_attn.forward(&h, x_ctx, cache)
        };
        let h = mlxcel_core::add(residual, &attn_out);

        let m = self.post_attention_layernorm.forward(&h);
        let mlp_out = if let Some(conv) = &self.mlp_conv {
            let (convolved, kernel) = conv.prepare(&m);
            let mlp = self.mlp.forward(&convolved);
            conv.finish(&mlp, &kernel)
        } else {
            self.mlp.forward(&m)
        };
        mlxcel_core::add(&h, &mlp_out)
    }
}

// ---------------------------------------------------------------------------
// # 5. Candidate selector
// ---------------------------------------------------------------------------

/// Output of a conditional selector walk.
pub struct SelectorOutput {
    /// Draft token ids `[1, L]` int32, one per position after the anchor.
    pub path: UniquePtr<MlxArray>,
    /// Actual compact proposal distributions, absent on the greedy path.
    proposal_probs: Vec<UniquePtr<MlxArray>>,
    /// Target-token IDs aligned with each compact distribution.
    candidates: UniquePtr<MlxArray>,
}

/// DFlash2 candidate selector, ported from SGLang `CandidateSelector`
/// (dflash.py): scores the K×K transitions between adjacent proposal slots
/// and walks one coherent path through them.
///
/// `score[b, e, p, c] = unary[b, e, c] + <A[pred[b, e, p]] * proj(h[b, e]), B[c]>`
/// where `pred` is the candidate at slot `e - 1` (the verified anchor for
/// slot 0), `A`/`B` are the predecessor/successor codebooks and `proj` the
/// hidden projection. The greedy walk follows the argmax chain:
/// `path[0] = argmax_c score[0][*][c]`; `path[e] = argmax_c score[e][path[e-1]][c]`
/// (SGLang `sample_path` with `greedy_mask` set — the sequential form is
/// exactly the chain the lattice maps define).
///
/// Stochastic walks retain the exact conditional proposal distribution after
/// temperature and filters. Target history penalties are applied at verification,
/// not to compact candidate indices (which are not target token IDs).
pub struct CandidateSelector {
    pub top_k: usize,
    predecessor_codebook: UniquePtr<MlxArray>, // [vocab, rank]
    successor_codebook: UniquePtr<MlxArray>,   // [vocab, rank]
    hidden_projection: UnifiedLinear,          // hidden -> rank
}

impl CandidateSelector {
    pub fn from_weights(weights: &WeightMap, config: &DFlash2Config) -> Result<Self, String> {
        let predecessor_codebook = weights
            .get("candidate_selector.predecessor_codebook")
            .map(|w| mlxcel_core::copy(w))
            .ok_or("Weight not found: candidate_selector.predecessor_codebook")?;
        let successor_codebook = weights
            .get("candidate_selector.successor_codebook")
            .map(|w| mlxcel_core::copy(w))
            .ok_or("Weight not found: candidate_selector.successor_codebook")?;
        let hidden_projection =
            quantized_draft_linear(weights, "candidate_selector.hidden_projection")?;
        Ok(Self {
            top_k: config.selector_top_k,
            predecessor_codebook,
            successor_codebook,
            hidden_projection,
        })
    }

    /// Conditional path walk over the masked block positions.
    ///
    /// - `hidden` `[1, L, H]` — the drafter's post-`norm` output for the
    ///   proposal positions (SGLang `pred_hidden`, position 0 = anchor
    ///   excluded).
    /// - `candidates` `[1, L, K]` int32 — top-K candidate ids per position
    ///   (from the target LM head).
    /// - `unary` `[1, L, K]` — the transformed top-K logits.
    /// - `anchor_ids` `[1]` int32 — the token immediately before the block.
    ///
    /// Returns the chosen draft token path `[1, L]` int32. `edge_scale`
    /// calibrates the learned DFlash2/LiLiCorr-style correlation against the
    /// unary target-head score; `1.0` is the checkpoint's exact default.
    pub fn select(
        &self,
        hidden: &MlxArray,
        candidates: &MlxArray,
        unary: &MlxArray,
        anchor_ids: &MlxArray,
        edge_scale: f32,
        sampling: &mlxcel_core::generate::SamplingConfig,
    ) -> Result<SelectorOutput, String> {
        let logits_shape = mlxcel_core::array_shape(unary);
        let batch = logits_shape[0];
        let npos = logits_shape[1];
        let k = logits_shape[2];
        let hidden_proj = self.hidden_projection.forward(hidden);
        let rank = mlxcel_core::array_shape(&hidden_proj)[2];

        // Edges follow the authoritative einsum "blpr,blcr->blpc".
        // Each row conditions on the predecessor actually selected.
        let anchor = mlxcel_core::reshape(anchor_ids, &[batch]);
        let mut predecessor = anchor; // [B] int32
        let mut path_rows = Vec::with_capacity(npos as usize);
        let stochastic = !mlxcel_core::speculative::stochastic_accept::sampler_is_greedy(sampling);
        let proposal_sampling = mlxcel_core::generate::SamplingConfig {
            temperature: sampling.temperature,
            // A cutoff covering the entire compact domain is no filter.
            // Passing the target-vocabulary cutoff directly can exceed K.
            top_k: if sampling.top_k > 0 && sampling.top_k < k {
                sampling.top_k
            } else {
                0
            },
            top_p: sampling.top_p,
            min_p: sampling.min_p,
            ..mlxcel_core::generate::SamplingConfig::default()
        };
        let mut proposal_probs = Vec::with_capacity(if stochastic { npos as usize } else { 0 });
        for position in 0..npos {
            let pred_emb = mlxcel_core::embedding(&self.predecessor_codebook, &predecessor); // [B, rank]
            let candidate_slice =
                mlxcel_core::slice(candidates, &[0, position, 0], &[batch, position + 1, k]);
            let succ_emb = mlxcel_core::embedding(&self.successor_codebook, &candidate_slice); // [B, K, rank]
            let hidden_row = mlxcel_core::slice(
                &hidden_proj,
                &[0, position, 0],
                &[batch, position + 1, rank],
            ); // [B, 1, rank]
            let hidden_row = mlxcel_core::reshape(&hidden_row, &[batch, rank]); // [B, rank]
            let pred_emb = mlxcel_core::expand_dims(&pred_emb, 1); // [B, 1, rank]
            let hidden_row = mlxcel_core::expand_dims(&hidden_row, 1); // [B, 1, rank]
            let edges = mlxcel_core::sum_axis(
                &mlxcel_core::multiply(
                    &mlxcel_core::multiply(&pred_emb, &hidden_row),
                    &succ_emb, // [B, K, rank]
                ),
                -1,
                false,
            ); // [B, K]
            let unary_row = mlxcel_core::slice(unary, &[0, position, 0], &[batch, position + 1, k]);
            let scores = if edge_scale == 1.0 {
                mlxcel_core::add(&unary_row, &edges)
            } else {
                mlxcel_core::add(&unary_row, &multiply_scalar(&edges, edge_scale))
            }; // [B, K]
            let scores = mlxcel_core::reshape(&scores, &[batch, k]);
            let selected = if stochastic {
                let (token, probs) = mlxcel_core::sampling::sample_token_with_distribution(
                    &scores,
                    &proposal_sampling,
                    &[],
                );
                proposal_probs.push(probs);
                token
            } else {
                mlxcel_core::argmax(&scores, -1, false)
            };
            let candidate_row = mlxcel_core::reshape(&candidate_slice, &[batch, k]);
            let selected_emb = mlxcel_core::reshape(&selected, &[batch, 1]);
            let selected_id = mlxcel_core::take_along_axis(&candidate_row, &selected_emb, -1); // [B, 1]
            predecessor = mlxcel_core::reshape(&selected_id, &[batch]); // [B]
            path_rows.push(mlxcel_core::share(&predecessor));
        }
        let path_ptrs = path_rows.iter().map(|row| row.as_ptr()).collect::<Vec<_>>();
        let path = mlxcel_core::stack(&path_ptrs, 1);
        Ok(SelectorOutput {
            path,
            proposal_probs,
            candidates: mlxcel_core::share(candidates),
        })
    }
}

/// Select an exact top-k set from either the full target vocabulary or the
/// compact DFlash candidate domain. Compact indices are remapped only after
/// gathering their logits, preserving candidate/value alignment on device.
fn top_k_candidate_logits(
    logits: &MlxArray,
    top_k: usize,
    compact_domain: bool,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
    let shape = mlxcel_core::array_shape(logits);
    if shape.len() != 3 {
        return Err(format!(
            "DFlash2 candidate logits must have layout [batch, positions, vocab], got {shape:?}"
        ));
    }
    let vocab = shape[2];
    let k = i32::try_from(top_k).map_err(|_| "selector_top_k too large".to_owned())?;
    if k <= 0 || k >= vocab {
        return Err(format!(
            "selector_top_k={k} must be in 1..{vocab} for the projected token domain"
        ));
    }
    let partition = mlxcel_core::argpartition(logits, -k, -1);
    let compact_ids =
        mlxcel_core::slice(&partition, &[0, 0, vocab - k], &[shape[0], shape[1], vocab]);
    let compact_ids = mlxcel_core::contiguous(&compact_ids, false);
    let values = mlxcel_core::take_along_axis(logits, &compact_ids, -1);
    let target_ids = if compact_domain {
        Qwen35Model::map_dflash_candidate_tokens(&compact_ids)
    } else {
        compact_ids
    };
    Ok((target_ids, values))
}

// ---------------------------------------------------------------------------
// # 6. DFlash2 draft model
// ---------------------------------------------------------------------------

/// Assembled DFlash2 drafter, ported from SGLang `DFlash2DraftModel`
/// (dflash.py). No embedding or LM head of its own: the round loop binds the
/// target's embedding table and scores candidates through the target head
/// (SGLang `compute_candidates`).
pub struct DFlash2DraftModel {
    pub config: DFlash2Config,
    fc: UnifiedLinear, // [len(target_layer_ids) * hidden, hidden]
    hidden_norm: RMSNorm,
    layers: Vec<DFlash2DecoderLayer>,
    norm: RMSNorm,
    embed_tokens: UnifiedEmbedding, // bound from the target at load
    candidate_selector: CandidateSelector,
    output_multiplier: f32,
    final_logit_softcapping: Option<f32>,
}

impl DFlash2DraftModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: DFlash2Config,
        embed_tokens: UnifiedEmbedding,
    ) -> Result<Self, String> {
        let fc = quantized_draft_linear(weights, "fc")?;
        let hidden_norm_w = weights
            .get("hidden_norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or("Weight not found: hidden_norm.weight")?;
        let hidden_norm = RMSNorm::new(hidden_norm_w, config.rms_norm_eps);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DFlash2DecoderLayer::from_weights(
                weights,
                &format!("layers.{i}"),
                &config,
                i,
            )?);
        }
        let norm_w = weights
            .get("norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or("Weight not found: norm.weight")?;
        let norm = RMSNorm::new(norm_w, config.rms_norm_eps);
        let candidate_selector = CandidateSelector::from_weights(weights, &config)?;
        let output_multiplier = config.output_multiplier;
        let final_logit_softcapping = config.final_logit_softcapping;
        Ok(Self {
            config,
            fc,
            hidden_norm,
            layers,
            norm,
            embed_tokens,
            candidate_selector,
            output_multiplier,
            final_logit_softcapping,
        })
    }

    pub fn make_cache(&self) -> Vec<DFlash2KVCache> {
        self.config
            .layer_types
            .iter()
            .map(|layer_type| {
                if layer_type == "sliding_attention" {
                    let window = self
                        .config
                        .sliding_window
                        .expect("sliding_attention layers require sliding_window");
                    DFlash2KVCache::Sliding(RotatingKVCache::new((window - 1) as i32))
                } else {
                    DFlash2KVCache::Full(KVCache::new())
                }
            })
            .collect()
    }

    /// Backbone forward over the masked draft block, returning the final
    /// normalized hidden states `[1, bs, H]` (SGLang `DFlashDraftModel.forward`
    /// + `norm`).
    /// Project newly committed target-layer rows once, then reuse the result
    /// across every draft layer. Passing `None` is reserved for an exact
    /// projected-prefix restore whose per-layer K/V is already in `caches`.
    pub fn hidden_states(
        &self,
        inputs: &MlxArray,
        target_hidden: Option<&MlxArray>,
        caches: &mut [DFlash2KVCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(inputs);
        let projected_context = target_hidden.map(|target_hidden| {
            let fc_out = self.fc.forward(target_hidden);
            self.hidden_norm.forward(&fc_out)
        });
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, projected_context.as_deref(), cache);
        }
        self.norm.forward(&h)
    }

    /// SGLang `_transform_unary_logits`: apply `output_multiplier` and the
    /// optional `final_logit_softcapping` (tanh) to candidate scores.
    fn transform_unary_logits(&self, logits: &MlxArray) -> UniquePtr<MlxArray> {
        let logits = if self.output_multiplier != 1.0 {
            multiply_scalar(logits, self.output_multiplier)
        } else {
            mlxcel_core::copy(logits)
        };
        match self.final_logit_softcapping {
            Some(softcap) if softcap > 0.0 => {
                let scaled = mlxcel_core::divide_scalar(&logits, softcap);
                let tanh = mlxcel_core::tanh(&scaled);
                multiply_scalar(&tanh, softcap)
            }
            _ => logits,
        }
    }

    /// Top-K draft candidates per position via the target's compact candidate head
    /// when available (SGLang `compute_candidates`):
    /// `hidden [1, L, H]` → `candidates [1, L, K]` target-token ids + `unary
    /// [1, L, K]` transformed logits. The compact projection contains 80,922
    /// unique rows instead of all 248,320 target rows.
    pub fn compute_candidates(
        &self,
        hidden: &MlxArray,
        target: &Qwen35Model,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let compact_domain = target.has_compact_dflash_candidate_head();
        let logits = target.project_dflash_candidate_logits(hidden);
        let (candidates, unary) =
            top_k_candidate_logits(&logits, self.candidate_selector.top_k, compact_domain)?;
        Ok((candidates, self.transform_unary_logits(&unary)))
    }

    /// One masked-forward draft round: hidden → target-head top-K candidates
    /// → conditional selector path. Returns `[1, bs-1]` draft tokens.
    pub fn propose(
        &self,
        inputs: &MlxArray,
        target_hidden: Option<&MlxArray>,
        caches: &mut [DFlash2KVCache],
        target: &Qwen35Model,
        selector_edge_scale: f32,
        sampling: &mlxcel_core::generate::SamplingConfig,
    ) -> Result<SelectorOutput, String> {
        let hidden = self.hidden_states(inputs, target_hidden, caches);
        let hidden_shape = mlxcel_core::array_shape(&hidden);
        // Position 0 is the anchor; the draft positions are 1..bs
        // (SGLang `_SelectorDraftSampler`: `hs = hidden[:, 1:, :]`).
        let pred_hidden = mlxcel_core::slice(
            &hidden,
            &[0, 1, 0],
            &[hidden_shape[0], hidden_shape[1], hidden_shape[2]],
        );
        let (candidates, unary) = self.compute_candidates(&pred_hidden, target)?;
        let anchor = mlxcel_core::slice(inputs, &[0, 0], &[1, 1]);
        self.candidate_selector.select(
            &pred_hidden,
            &candidates,
            &unary,
            &anchor,
            selector_edge_scale,
            sampling,
        )
    }
}

// ---------------------------------------------------------------------------
// # 7. Weight loading
// ---------------------------------------------------------------------------

/// Load the drafter weights and config from a checkpoint directory, keeping
/// the checkpoint's native bf16 dtype (matching SGLang, which runs the
/// DFlash2 drafter in bf16). An f16 conversion would overflow the drafter's
/// activations: the MLP gate×up intermediate reaches values beyond f16's
/// 65504 max, which poisons the residual stream with inf → NaN. bf16's
/// 3.4e38 range is ample. Quantization auxiliaries (`.scales` / `.biases`
/// suffixes) are skipped for robustness; the DFlash2 checkpoint is entirely
/// non-quantized BF16.
pub fn load_draft_weights(dir: &std::path::Path) -> Result<(WeightMap, DFlash2Config), String> {
    let config_path = dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
    let config_value: Value = serde_json::from_str(&config_text)
        .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?;
    let config = DFlash2Config::from_json(&config_value)?;
    let weights = load_weights_from_dir(dir)?;
    Ok((weights, config))
}

// The current MLX quantized matmul path has two useful DFlash2 verify shapes.
// Keeping the controller on this closed set avoids accumulating one compiled
// graph per tail length while still retaining the measured long-context win of
// width five. Tail rounds intentionally use a full static width and let exact
// verification truncate the committed output to the caller's remaining budget.
const DFLASH2_STATIC_VERIFY_WIDTHS: [usize; 2] = [4, 5];
const DFLASH2_LONG_CONTEXT_TOKENS: usize = 64_000;
const DFLASH2_SELECTOR_EDGE_SCALES: [f32; 3] = [0.75, 1.0, 1.25];
const DFLASH2_DEFAULT_SELECTOR_ARM: usize = 1;
const DFLASH2_CONFIDENCE_Z: f64 = 1.644_853_626_951_472_2;
const DFLASH2_WIDTH_MIN_SAMPLES: u64 = 4;
const DFLASH2_WIDTH_PROBE_INTERVAL: u64 = 16;
const DFLASH2_SELECTOR_BASELINE_SAMPLES: u64 = 8;
const DFLASH2_SELECTOR_MIN_SAMPLES: u64 = 4;
const DFLASH2_SELECTOR_PROBE_INTERVAL: u64 = 32;
const DFLASH2_VERIFY_WIDTH_ENV: &str = "QW_DFLASH2_VERIFY_WIDTH";
const DFLASH2_SELECTOR_SCALE_ENV: &str = "QW_DFLASH2_SELECTOR_EDGE_SCALE";

#[derive(Debug, Clone, Copy, Default)]
struct RunningMoments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningMoments {
    fn observe(&mut self, sample: f64) {
        if !sample.is_finite() || sample <= 0.0 {
            return;
        }
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (sample - self.mean);
    }

    fn confidence_bounds(self) -> Option<(f64, f64)> {
        if self.count < DFLASH2_WIDTH_MIN_SAMPLES {
            return None;
        }
        let variance = if self.count > 1 {
            (self.m2 / (self.count - 1) as f64).max(0.0)
        } else {
            0.0
        };
        let radius = DFLASH2_CONFIDENCE_Z * (variance / self.count as f64).sqrt();
        Some((
            (self.mean - radius).max(f64::MIN_POSITIVE),
            self.mean + radius,
        ))
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PrefixSurvival {
    trials: [u64; 4],
    survived: [u64; 4],
}

impl PrefixSurvival {
    fn observe(&mut self, accepted: usize, proposed: usize) {
        for depth in 0..proposed.min(self.trials.len()) {
            self.trials[depth] += 1;
            if accepted > depth {
                self.survived[depth] += 1;
            }
        }
    }

    /// Wilson interval for `P(accepted_prefix >= depth + 1)`.
    fn confidence_bounds(self, depth: usize) -> Option<(f64, f64)> {
        let n = *self.trials.get(depth)?;
        if n < DFLASH2_SELECTOR_MIN_SAMPLES {
            return None;
        }
        let p = self.survived[depth] as f64 / n as f64;
        let z2 = DFLASH2_CONFIDENCE_Z * DFLASH2_CONFIDENCE_Z;
        let denominator = 1.0 + z2 / n as f64;
        let center = (p + z2 / (2.0 * n as f64)) / denominator;
        let margin = DFLASH2_CONFIDENCE_Z
            * ((p * (1.0 - p) + z2 / (4.0 * n as f64)) / n as f64).sqrt()
            / denominator;
        Some(((center - margin).max(0.0), (center + margin).min(1.0)))
    }

    /// Expected accepted-prefix length, not a global path likelihood.
    ///
    /// `E[A] = sum_d P(A >= d)`: a later position contributes only when every
    /// earlier proposal survived target verification, matching speculative
    /// decoding's actual utility.
    fn utility_bounds(self, proposed: usize) -> Option<(f64, f64)> {
        let mut lower = 0.0;
        let mut upper = 0.0;
        for depth in 0..proposed.min(self.trials.len()) {
            let (depth_lower, depth_upper) = self.confidence_bounds(depth)?;
            lower += depth_lower;
            upper += depth_upper;
        }
        Some((lower, upper))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dflash2ContextClass {
    Short,
    Long,
}

impl Dflash2ContextClass {
    fn for_tokens(tokens: usize) -> Self {
        if tokens >= DFLASH2_LONG_CONTEXT_TOKENS {
            Self::Long
        } else {
            Self::Short
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Short => 0,
            Self::Long => 1,
        }
    }

    fn default_width_index(self) -> usize {
        match self {
            Self::Short => 0,
            Self::Long => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Dflash2ContextCalibration {
    prefix: PrefixSurvival,
    width_latency: [RunningMoments; 2],
    width_rounds: [u64; 2],
    selector_prefix: [PrefixSurvival; 3],
    selector_rounds: [u64; 3],
    rounds: u64,
}

impl Dflash2ContextCalibration {
    fn throughput_bounds(&self) -> Option<[(f64, f64); 2]> {
        let (base_lower, base_upper) = self.prefix.utility_bounds(3)?;
        let (marginal_lower, marginal_upper) = self.prefix.confidence_bounds(3)?;
        let (narrow_latency_lower, narrow_latency_upper) =
            self.width_latency[0].confidence_bounds()?;
        let (wide_latency_lower, wide_latency_upper) = self.width_latency[1].confidence_bounds()?;

        // Every verified round emits the target bonus. Width five adds exactly
        // one possible accepted proposal, whose marginal yield is survival to
        // depth four. Comparing these intervals is equivalent to comparing
        // `marginal_yield / marginal_latency`, but remains stable when the
        // measured latency delta is close to zero.
        let base_lower = 1.0 + base_lower;
        let base_upper = 1.0 + base_upper;
        Some([
            (
                base_lower / narrow_latency_upper,
                base_upper / narrow_latency_lower,
            ),
            (
                (base_lower + marginal_lower) / wide_latency_upper,
                (base_upper + marginal_upper) / wide_latency_lower,
            ),
        ])
    }

    fn choose_width_index(&self, default: usize) -> usize {
        if self.width_rounds[default] < DFLASH2_WIDTH_MIN_SAMPLES {
            return default;
        }
        let challenger = 1 - default;
        if self.width_rounds[challenger] < DFLASH2_WIDTH_MIN_SAMPLES {
            return challenger;
        }
        if self.rounds > 0 && self.rounds.is_multiple_of(DFLASH2_WIDTH_PROBE_INTERVAL) {
            return challenger;
        }
        let Some(bounds) = self.throughput_bounds() else {
            return default;
        };
        if bounds[challenger].0 > bounds[default].1 {
            challenger
        } else {
            default
        }
    }

    fn choose_selector_arm(&self, proposed: usize) -> usize {
        if self.selector_rounds[DFLASH2_DEFAULT_SELECTOR_ARM] < DFLASH2_SELECTOR_BASELINE_SAMPLES {
            return DFLASH2_DEFAULT_SELECTOR_ARM;
        }

        let challenger = [0, 2]
            .into_iter()
            .min_by_key(|&arm| self.selector_rounds[arm])
            .expect("selector has two calibration arms");
        if self.selector_rounds[challenger] < DFLASH2_SELECTOR_MIN_SAMPLES {
            return challenger;
        }
        if self.rounds > 0 && self.rounds.is_multiple_of(DFLASH2_SELECTOR_PROBE_INTERVAL) {
            return challenger;
        }

        let Some((baseline_lower, baseline_upper)) =
            self.selector_prefix[DFLASH2_DEFAULT_SELECTOR_ARM].utility_bounds(proposed)
        else {
            return DFLASH2_DEFAULT_SELECTOR_ARM;
        };
        let mut selected = DFLASH2_DEFAULT_SELECTOR_ARM;
        let mut selected_lower = baseline_lower;
        for arm in [0, 2] {
            let Some((lower, _upper)) = self.selector_prefix[arm].utility_bounds(proposed) else {
                continue;
            };
            // Only replace the checkpoint-compatible scale when its accepted
            // prefix utility is confidently beaten. Ties and overlap retain the
            // deterministic DFlash2 default.
            if lower > baseline_upper && lower > selected_lower {
                selected = arm;
                selected_lower = lower;
            }
        }
        selected
    }

    fn observe(
        &mut self,
        width_index: usize,
        selector_arm: usize,
        accepted: usize,
        proposed: usize,
        compute_latency: Duration,
    ) {
        self.prefix.observe(accepted, proposed);
        self.width_latency[width_index].observe(compute_latency.as_secs_f64());
        self.width_rounds[width_index] += 1;
        self.selector_prefix[selector_arm].observe(accepted, proposed);
        self.selector_rounds[selector_arm] += 1;
        self.rounds += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Dflash2RoundPolicy {
    width_index: usize,
    width: usize,
    selector_arm: usize,
    selector_edge_scale: f32,
}

#[derive(Debug, Clone, Default)]
struct Dflash2Calibration {
    contexts: [Dflash2ContextCalibration; 2],
    forced_width_index: Option<usize>,
    forced_selector_arm: Option<usize>,
}

impl Dflash2Calibration {
    fn from_environment() -> Result<Self, String> {
        let forced_width_index = std::env::var(DFLASH2_VERIFY_WIDTH_ENV)
            .ok()
            .map(|value| match value.trim() {
                "4" => Ok(0),
                "5" => Ok(1),
                other => Err(format!(
                    "{DFLASH2_VERIFY_WIDTH_ENV} must be one of 4 or 5, got {other:?}"
                )),
            })
            .transpose()?;
        let forced_selector_arm = std::env::var(DFLASH2_SELECTOR_SCALE_ENV)
            .ok()
            .map(|value| match value.trim() {
                "0.75" => Ok(0),
                "1" | "1.0" | "1.00" => Ok(1),
                "1.25" => Ok(2),
                other => Err(format!(
                    "{DFLASH2_SELECTOR_SCALE_ENV} must be one of 0.75, 1.0, or 1.25, \
                     got {other:?}"
                )),
            })
            .transpose()?;
        Ok(Self {
            contexts: Default::default(),
            forced_width_index,
            forced_selector_arm,
        })
    }

    fn policy(&self, context_tokens: usize) -> Dflash2RoundPolicy {
        let class = Dflash2ContextClass::for_tokens(context_tokens);
        let context = &self.contexts[class.index()];
        let width_index = self
            .forced_width_index
            .unwrap_or_else(|| context.choose_width_index(class.default_width_index()));
        let width = DFLASH2_STATIC_VERIFY_WIDTHS[width_index];
        let selector_arm = self
            .forced_selector_arm
            .unwrap_or_else(|| context.choose_selector_arm(width - 1));
        Dflash2RoundPolicy {
            width_index,
            width,
            selector_arm,
            selector_edge_scale: DFLASH2_SELECTOR_EDGE_SCALES[selector_arm],
        }
    }

    fn observe(
        &mut self,
        context_tokens: usize,
        policy: Dflash2RoundPolicy,
        accepted: usize,
        proposed: usize,
        compute_latency: Duration,
    ) {
        self.contexts[Dflash2ContextClass::for_tokens(context_tokens).index()].observe(
            policy.width_index,
            policy.selector_arm,
            accepted,
            proposed,
            compute_latency,
        );
    }
}

// ---------------------------------------------------------------------------
// # 8. Distribution-preserving round loop + generator
// ---------------------------------------------------------------------------

/// Detached target state and target-layer hidden context at an exact prompt
/// boundary. The hidden rows seed the drafter's bounded sliding context after
/// the target snapshot is restored.
static NEXT_DFLASH2_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

fn next_dflash2_snapshot_id() -> u64 {
    NEXT_DFLASH2_SNAPSHOT_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct Dflash2PromptSnapshot {
    target: ModelStateSnapshot,
    hidden_concat: UniquePtr<MlxArray>,
    hidden_offset: usize,
    continuation_logits: UniquePtr<MlxArray>,
    id: u64,
}

impl Dflash2PromptSnapshot {
    fn capture(
        target: &Qwen35Model,
        token_len: usize,
        hidden_concat: &MlxArray,
        continuation_logits: &MlxArray,
        previous: Option<&Self>,
    ) -> Result<Self, String> {
        let hidden_rows = mlxcel_core::array_shape(hidden_concat)[1] as usize;
        let hidden_offset = token_len
            .checked_sub(hidden_rows)
            .ok_or_else(|| "DFlash2 hidden context exceeds its target boundary".to_string())?;
        target.materialize_mtp_cache_state();
        let target = target
            .snapshot_sequence_state(
                SequenceId::from_raw(0),
                token_len,
                previous.map(|snapshot| &snapshot.target),
            )
            .ok_or_else(|| {
                format!("failed to capture DFlash2 target state at {token_len} tokens")
            })?;
        Ok(Self {
            id: next_dflash2_snapshot_id(),
            target,
            hidden_concat: materialize_detached(mlxcel_core::contiguous(hidden_concat, false)),
            hidden_offset,
            continuation_logits: materialize_detached(mlxcel_core::contiguous(
                continuation_logits,
                false,
            )),
        })
    }

    pub fn token_len(&self) -> usize {
        self.target.token_len()
    }

    pub fn target_snapshot(&self) -> &ModelStateSnapshot {
        &self.target
    }

    pub fn nbytes(&self) -> usize {
        self.target.nbytes()
            + mlxcel_core::array_nbytes(&self.hidden_concat)
            + mlxcel_core::array_nbytes(&self.continuation_logits)
    }
    pub fn storage_summary(&self) -> mlxcel_core::generate::SnapshotStorageSummary {
        let mut summary = self.target.storage_summary();
        summary.local_bytes += mlxcel_core::array_nbytes(&self.hidden_concat)
            + mlxcel_core::array_nbytes(&self.continuation_logits);
        summary
    }

    pub(crate) fn to_portable(&self) -> PortablePromptSnapshot {
        PortablePromptSnapshot::Dflash2 {
            target: portable_model_state(&self.target),
            hidden_concat: array_to_portable(None, &self.hidden_concat),
            hidden_offset: self.hidden_offset,
            continuation_logits: array_to_portable(None, &self.continuation_logits),
        }
    }

    pub(crate) fn from_portable_parts(
        target: ModelStateSnapshot,
        hidden_concat: PortableArray,
        hidden_offset: usize,
        continuation_logits: PortableArray,
    ) -> Result<Self, String> {
        let hidden_rows = hidden_concat
            .shape
            .get(1)
            .copied()
            .filter(|&rows| rows > 0)
            .ok_or_else(|| {
                "DFlash2 portable hidden context must have layout [1, rows, hidden]".to_string()
            })? as usize;
        if hidden_concat.shape.len() != 3
            || hidden_concat.shape[0] != 1
            || hidden_offset.checked_add(hidden_rows) != Some(target.token_len())
        {
            return Err(
                "DFlash2 portable hidden context does not match the target boundary".to_string(),
            );
        }
        if continuation_logits.shape.len() != 3
            || continuation_logits.shape[0] != 1
            || continuation_logits.shape[1] != 1
        {
            return Err(
                "DFlash2 portable continuation-logits layout must be [1, 1, vocab]".to_string(),
            );
        }
        validate_dflash2_portable_float(&hidden_concat)?;
        validate_dflash2_portable_float(&continuation_logits)?;
        Ok(Self {
            id: next_dflash2_snapshot_id(),
            target,
            hidden_concat: array_from_portable(hidden_concat, None)?,
            hidden_offset,
            continuation_logits: array_from_portable(continuation_logits, None)?,
        })
    }
}

fn validate_dflash2_portable_float(array: &PortableArray) -> Result<(), String> {
    let finite = match array.dtype {
        mlxcel_core::dtype::FLOAT32 => array.bytes.chunks_exact(4).all(|bytes| {
            f32::from_ne_bytes(bytes.try_into().expect("four-byte float")).is_finite()
        }),
        mlxcel_core::dtype::FLOAT16 | mlxcel_core::dtype::BFLOAT16 => {
            let exponent_mask = if array.dtype == mlxcel_core::dtype::FLOAT16 {
                0x7c00
            } else {
                0x7f80
            };
            array.bytes.chunks_exact(2).all(|bytes| {
                u16::from_ne_bytes(bytes.try_into().expect("two-byte float")) & exponent_mask
                    != exponent_mask
            })
        }
        _ => {
            return Err(
                "DFlash2 hidden context and logits require floating-point arrays".to_string(),
            );
        }
    };
    if !finite {
        return Err("DFlash2 hidden context and logits must be finite".to_string());
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct Dflash2PrefixReuse<'a> {
    pub snapshot: &'a Dflash2PromptSnapshot,
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dflash2GenerationStats {
    pub accepted_draft_tokens: usize,
    pub proposed_draft_tokens: usize,
    /// Wall-clock time spent processing the uncached prompt.
    pub prefill_time: Duration,
    /// Wall-clock time spent in the post-prefill DFlash2 decode loop.
    pub decode_time: Duration,
    pub draft_time: Duration,
    pub target_verify_time: Duration,
    pub walk_time: Duration,
    pub reconcile_time: Duration,
    pub target_forward_calls: usize,
    pub speculative_rounds: usize,
    /// Per-width calibration outcomes. Index 0 is verify width 4; index 1 is
    /// verify width 5. These observable counters can be used by offline
    /// benchmarks before pinning `QW_DFLASH2_VERIFY_WIDTH`.
    pub verify_width_rounds: [usize; 2],
    pub verify_width_accepted_draft_tokens: [usize; 2],
    pub verify_width_proposed_draft_tokens: [usize; 2],
    /// Per-selector calibration outcomes for edge scales `[0.75, 1.0, 1.25]`.
    /// Offline runs can pin one with `QW_DFLASH2_SELECTOR_EDGE_SCALE`.
    pub selector_scale_rounds: [usize; 3],
    pub selector_scale_accepted_draft_tokens: [usize; 3],
    pub selector_scale_proposed_draft_tokens: [usize; 3],
    /// Number of exact prompt-boundary projected K/V snapshots reused.
    pub projected_context_cache_hits: usize,
}

impl Dflash2GenerationStats {
    pub fn acceptance_percentage(self) -> f64 {
        if self.proposed_draft_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_draft_tokens as f64 * 100.0
        }
    }

    fn record_round(&mut self, policy: Dflash2RoundPolicy, accepted: usize, proposed: usize) {
        self.accepted_draft_tokens += accepted;
        self.proposed_draft_tokens += proposed;
        self.verify_width_rounds[policy.width_index] += 1;
        self.verify_width_accepted_draft_tokens[policy.width_index] += accepted;
        self.verify_width_proposed_draft_tokens[policy.width_index] += proposed;
        self.selector_scale_rounds[policy.selector_arm] += 1;
        self.selector_scale_accepted_draft_tokens[policy.selector_arm] += accepted;
        self.selector_scale_proposed_draft_tokens[policy.selector_arm] += proposed;
    }
}

pub(crate) struct Dflash2Generation {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) stats: Dflash2GenerationStats,
    pub(crate) cached_tokens: usize,
    pub(crate) prompt_snapshots: Vec<Dflash2PromptSnapshot>,
    pub(crate) final_snapshot: Option<Dflash2PromptSnapshot>,
    pub(crate) stop_reason: GenerationStopReason,
}

fn emit_initial_dflash2_token<F: FnMut(i32) -> bool>(
    bonus: i32,
    eos_tokens: &[i32],
    max_tokens: usize,
    generated: &mut Vec<i32>,
    history: &mut Vec<i32>,
    on_token: &mut F,
) -> (GenerationStopReason, bool) {
    if max_tokens == 0 {
        return (GenerationStopReason::MaxTokens, false);
    }
    if eos_tokens.contains(&bonus) {
        return (GenerationStopReason::Eos, false);
    }
    generated.push(bonus);
    history.push(bonus);
    if !on_token(bonus) {
        (GenerationStopReason::CallbackCancelled, false)
    } else {
        (
            GenerationStopReason::MaxTokens,
            generated.len() < max_tokens,
        )
    }
}

fn emit_dflash2_walk_tokens<F: FnMut(i32) -> bool>(
    tokens: &[i32],
    eos_tokens: &[i32],
    max_tokens: usize,
    generated: &mut Vec<i32>,
    history: &mut Vec<i32>,
    on_token: &mut F,
) -> Option<GenerationStopReason> {
    for &token in tokens {
        let (reason, keep_going) =
            emit_initial_dflash2_token(token, eos_tokens, max_tokens, generated, history, on_token);
        if !keep_going {
            return Some(reason);
        }
    }
    None
}

/// Verify row zero is the already-emitted anchor. Only accepted proposals
/// actually emitted by the callback join it; a correction remains unforwarded.
fn committed_dflash2_rows(accepted: usize, emitted: usize) -> usize {
    1 + accepted.min(emitted)
}

/// Verify only the reachable prefix. The full-vocabulary target distribution
/// includes every target penalty/filter; q is zero outside its compact support.
/// Temporarily append accepted tokens to the existing history rather than
/// copying a long prompt each round. Emission owns the permanent history update.
fn stochastic_dflash2_walk(
    proposal: &SelectorOutput,
    verify_logits: &MlxArray,
    sampling: &mlxcel_core::generate::SamplingConfig,
    history: &mut Vec<i32>,
    eos_tokens: &[i32],
    max_new_tokens: usize,
    sampler_state: &mut Option<mlxcel_core::sampling::SamplerState>,
) -> (mlxcel_core::speculative::mtp::walk::WalkResult, Vec<i32>) {
    use mlxcel_core::sampling::effective_token_distribution_with_state;
    use mlxcel_core::speculative::mtp::walk::WalkResult;
    use mlxcel_core::speculative::stochastic_accept::{
        AcceptanceRule, DraftVerdict, note_rule, verify_draft_token,
    };

    note_rule(AcceptanceRule::Stochastic);
    mlxcel_core::eval(&proposal.path);
    let draft_tokens = mlxcel_core::array_evaluated_bytes(&proposal.path)
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 token bytes")))
        .collect::<Vec<_>>();
    let shape = mlxcel_core::array_shape(verify_logits);
    let candidate_shape = mlxcel_core::array_shape(&proposal.candidates);
    let initial_history_len = history.len();
    let mut walk = WalkResult {
        accepted: 0,
        new_tokens: Vec::with_capacity((draft_tokens.len() + 1).min(max_new_tokens)),
    };
    for position in 0..=draft_tokens.len() {
        if walk.new_tokens.len() == max_new_tokens {
            break;
        }
        let row = position as i32;
        let logits = mlxcel_core::slice(verify_logits, &[0, row, 0], &[1, row + 1, shape[2]]);
        let p = effective_token_distribution_with_state(&logits, sampling, history, sampler_state);
        if position == draft_tokens.len() {
            // Sample the already-transformed p, not the logits again: a
            // randomized preprocessing filter must be evaluated exactly once.
            let token = mlxcel_core::fused_sample(&mlxcel_core::log(&p), 1.0, 0, 1.0, 0.0);
            mlxcel_core::eval(&token);
            walk.new_tokens.push(mlxcel_core::item_i32(&token));
            break;
        }
        let ids = mlxcel_core::slice(
            &proposal.candidates,
            &[0, row, 0],
            &[1, row + 1, candidate_shape[2]],
        );
        let ids = mlxcel_core::reshape(&ids, &[1, candidate_shape[2]]);
        let q = mlxcel_core::put_along_axis(
            &mlxcel_core::zeros(&[1, shape[2]], mlxcel_core::dtype::FLOAT32),
            &ids,
            &proposal.proposal_probs[position],
            -1,
        );
        match verify_draft_token(&p, &q, draft_tokens[position]) {
            DraftVerdict::Accept => {
                let token = draft_tokens[position];
                walk.accepted += 1;
                walk.new_tokens.push(token);
                if eos_tokens.contains(&token) {
                    break;
                }
                history.push(token);
            }
            DraftVerdict::Reject { replacement } => {
                walk.new_tokens.push(replacement);
                break;
            }
        }
    }
    history.truncate(initial_history_len);
    (walk, draft_tokens)
}

/// DFlash2 generation driver: prefill → speculative draft/verify rounds.
///
/// B=1 exact target sampling: greedy comparison for deterministic samplers,
/// otherwise maximal-coupling acceptance with full-vocabulary correction and
/// bonus distributions.
pub struct Qwen35Dflash2Generator {
    model: DFlash2DraftModel,
    caches: Vec<DFlash2KVCache>,
    calibration: Dflash2Calibration,
    target_layer_ids: Vec<usize>,
    hidden_limit: usize,
    projected_prefix: Option<Dflash2ProjectedPrefix>,
}

impl Qwen35Dflash2Generator {
    /// Load the drafter from `draft_dir` and bind its embedding to the
    /// target model's (the checkpoint ships no `embed_tokens.weight`).
    pub fn new(target: &Qwen35Model, draft_dir: &std::path::Path) -> Result<Self, String> {
        let (weights, config) = load_draft_weights(draft_dir)?;
        let embed = target.embed_tokens.clone_shared();
        let model = DFlash2DraftModel::from_weights(&weights, config.clone(), embed)?;
        let caches = model.make_cache();
        let sliding = config.layer_types.iter().all(|t| t == "sliding_attention");
        let hidden_limit = if sliding {
            config.sliding_window.unwrap_or(0).saturating_sub(1)
        } else {
            usize::MAX
        };
        let calibration = Dflash2Calibration::from_environment()?;
        Ok(Self {
            model,
            projected_prefix: None,
            caches,
            calibration,
            target_layer_ids: config.target_layer_ids.clone(),
            hidden_limit,
        })
    }
    /// Capture an extendable prefix without finalizing initial prefill.
    /// Generation finalizes it at the eventual full prompt boundary.
    pub fn capture_prompt_snapshot(
        &mut self,
        target: &Qwen35Model,
        prompt_tokens: &[i32],
    ) -> Result<Dflash2PromptSnapshot, String> {
        if prompt_tokens.is_empty() {
            return Err("DFlash2 snapshots require a non-empty prompt".to_string());
        }
        self.caches = self.model.make_cache();
        let prompt_array = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        let prefill = target.forward_dflash_prefill_segment(
            &prompt_array,
            &self.target_layer_ids,
            self.hidden_limit,
            true,
            false,
        )?;
        Dflash2PromptSnapshot::capture(
            target,
            prompt_tokens.len(),
            &prefill.hidden_concat,
            &prefill.first_logits,
            None,
        )
    }

    /// Advance the live target monotonically through requested boundaries.
    /// Recurrent target state cannot be sliced backwards after prefill.
    fn prefill_with_checkpoints(
        &self,
        target: &Qwen35Model,
        prompt_tokens: &[i32],
        reuse: Option<Dflash2PrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        retain_hidden: bool,
    ) -> Result<
        (
            Option<UniquePtr<MlxArray>>,
            UniquePtr<MlxArray>,
            Vec<Dflash2PromptSnapshot>,
        ),
        String,
    > {
        let cached_tokens = reuse.map_or(0, |reuse| reuse.cached_tokens);
        let mut boundaries = checkpoint_token_lengths
            .iter()
            .copied()
            .filter(|&length| length > cached_tokens && length < prompt_tokens.len())
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        let capture_prompt = cached_tokens < prompt_tokens.len()
            && checkpoint_token_lengths.contains(&prompt_tokens.len());
        let mut snapshots = Vec::with_capacity(boundaries.len() + usize::from(capture_prompt));
        let need_hidden = retain_hidden || !boundaries.is_empty() || capture_prompt;
        let mut hidden = None;
        let mut logits = None;
        if let Some(reuse) = reuse {
            target.restore_sequence_state(SequenceId::from_raw(0), &reuse.snapshot.target)?;
            if need_hidden {
                hidden = Some(mlxcel_core::share(&reuse.snapshot.hidden_concat));
            }
            logits = Some(mlxcel_core::share(&reuse.snapshot.continuation_logits));
        }
        let mut start = cached_tokens;
        for token_len in boundaries
            .into_iter()
            .chain(std::iter::once(prompt_tokens.len()))
        {
            if token_len == start {
                // A restored standalone prefix may still hold initial FP16
                // state. Finalization is idempotent for already-finished
                // prompt and terminal snapshots.
                target.finish_initial_prefill();
                continue;
            }
            let input = mlxcel_core::from_slice_i32(
                &prompt_tokens[start..token_len],
                &[1, (token_len - start) as i32],
            );
            let prefill = target.forward_dflash_prefill_segment(
                &input,
                &self.target_layer_ids,
                self.hidden_limit,
                start == 0,
                token_len == prompt_tokens.len(),
            )?;
            if need_hidden {
                hidden = Some(match hidden.take() {
                    Some(prefix) => {
                        merge_hidden_context(&prefix, &prefill.hidden_concat, self.hidden_limit)
                    }
                    None => prefill.hidden_concat,
                });
            }
            logits = Some(prefill.first_logits);
            if token_len < prompt_tokens.len() || capture_prompt {
                let snapshot = Dflash2PromptSnapshot::capture(
                    target,
                    token_len,
                    hidden
                        .as_deref()
                        .expect("checkpoints retain hidden context"),
                    logits
                        .as_deref()
                        .expect("prefill produces continuation logits"),
                    snapshots
                        .last()
                        .or_else(|| reuse.map(|reuse| reuse.snapshot)),
                )?;
                // Continue from detached immutable boundary rows rather than
                // retaining the preceding prefill construction graph.
                hidden = Some(mlxcel_core::share(&snapshot.hidden_concat));
                logits = Some(mlxcel_core::share(&snapshot.continuation_logits));
                snapshots.push(snapshot);
            }
            start = token_len;
        }
        Ok((
            hidden,
            logits.ok_or_else(|| "DFlash2 prefill requires at least one token".to_string())?,
            snapshots,
        ))
    }

    /// Generate with distribution-preserving DFlash2 draft verification.
    pub fn generate_streaming<F: FnMut(i32) -> bool>(
        &mut self,
        target: &Qwen35Model,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &mlxcel_core::generate::SamplingConfig,
        prefix_reuse: Option<Dflash2PrefixReuse<'_>>,
        checkpoint_token_lengths: &[usize],
        capture_final_snapshot: bool,
        mut on_token: F,
    ) -> Result<Dflash2Generation, String> {
        let mut resolved_sampling = sampling.clone();
        resolved_sampling
            .prompt_token_count
            .get_or_insert(prompt_tokens.len());
        resolved_sampling
            .token_bias
            .suppress_tokens(&target.output_suppressed_token_ids());
        let sampling = &resolved_sampling;
        mlxcel_core::generation_policy::seed_rng_if_needed(sampling);
        let eos_tokens = mlxcel_core::generation_policy::merged_eos_token_ids(
            target.eos_token_ids(),
            &sampling.stop_token_ids,
        );
        if max_tokens == 0 && checkpoint_token_lengths.is_empty() && !capture_final_snapshot {
            return Ok(Dflash2Generation {
                token_ids: Vec::new(),
                stats: Dflash2GenerationStats::default(),
                cached_tokens: 0,
                prompt_snapshots: Vec::new(),
                final_snapshot: None,
                stop_reason: GenerationStopReason::MaxTokens,
            });
        }
        if prompt_tokens.is_empty() || prompt_tokens.len() > i32::MAX as usize {
            return Err("DFlash2 prompt length must be in 1..=i32::MAX".to_string());
        }

        // Restore an exact target prefix when supplied, then process only the
        // uncached prompt suffix. Repeated reuse can also restore projected
        // drafter K/V when both the snapshot identity and committed suffix
        // match exactly; a changed suffix takes the raw-hidden rebuild path.
        let prefill_start = Instant::now();
        let reusable = prefix_reuse.filter(|reuse| {
            reuse.cached_tokens == reuse.snapshot.token_len()
                && reuse.cached_tokens <= prompt_tokens.len()
        });
        let cached_tokens = reusable.map_or(0, |reuse| reuse.cached_tokens);
        if let Some(reuse) = reusable {
            let hidden_shape = mlxcel_core::array_shape(&reuse.snapshot.hidden_concat);
            let logits_shape = mlxcel_core::array_shape(&reuse.snapshot.continuation_logits);
            if hidden_shape[2] as usize
                != self.target_layer_ids.len() * self.model.config.hidden_size
                || logits_shape[2] as usize != target.vocab_size()
                || hidden_shape[1] as usize != reuse.cached_tokens.min(self.hidden_limit)
            {
                return Err("DFlash2 prefix snapshot does not match the loaded model".to_string());
            }
        }
        // Splitting a suffix at new recurrent checkpoints changes its prefill
        // execution. Do not reuse or publish projections under an unsplit key.
        let segmented_prefill = checkpoint_token_lengths
            .iter()
            .any(|&length| length > cached_tokens && length < prompt_tokens.len());
        let projected_caches = reusable
            .filter(|_| !segmented_prefill)
            .and_then(|reuse| {
                let committed_suffix = &prompt_tokens[reuse.cached_tokens..];
                self.projected_prefix.as_ref().filter(|projected| {
                    projected.snapshot_id == reuse.snapshot.id
                        && projected.committed_suffix == committed_suffix
                })
            })
            .map(Dflash2ProjectedPrefix::restore)
            .transpose()?;
        let projected_cache_hit = projected_caches.is_some();
        self.caches = projected_caches.unwrap_or_else(|| self.model.make_cache());

        let (mut hidden_concat, first_logits, prompt_snapshots) = self.prefill_with_checkpoints(
            target,
            prompt_tokens,
            reusable,
            checkpoint_token_lengths,
            !projected_cache_hit || capture_final_snapshot,
        )?;
        let mut snapshot_hidden = if capture_final_snapshot {
            Some(materialize_detached(mlxcel_core::share(
                hidden_concat
                    .as_deref()
                    .expect("final capture retains hidden context"),
            )))
        } else {
            None
        };
        if !projected_cache_hit {
            let hidden_rows = mlxcel_core::array_shape(
                hidden_concat
                    .as_deref()
                    .expect("uncached projections require hidden context"),
            )[1] as usize;
            let hidden_offset = prompt_tokens.len() - hidden_rows;
            for cache in self.caches.iter_mut() {
                match cache {
                    DFlash2KVCache::Full(c) => c.offset = hidden_offset as i32,
                    DFlash2KVCache::Sliding(c) => c.offset = hidden_offset as i32,
                }
            }
        }
        if projected_cache_hit {
            // Memoized K/V already contains the entire restored prompt suffix.
            // Raw rows retained for snapshots must not be projected twice.
            hidden_concat = None;
        }
        let mut bonus = if max_tokens == 0 {
            0
        } else {
            let (token, _) = mlxcel_core::sampling::sample_token_optimized(
                &first_logits,
                sampling,
                prompt_tokens,
            );
            mlxcel_core::eval(&token);
            mlxcel_core::item_i32(&token)
        };
        let mut committed_tokens = prompt_tokens.len();
        let mut terminal_logits = capture_final_snapshot.then_some(first_logits);
        let prefill_time = prefill_start.elapsed();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut history = prompt_tokens.to_vec();
        let mut stats = Dflash2GenerationStats {
            prefill_time,
            projected_context_cache_hits: usize::from(projected_cache_hit),
            ..Dflash2GenerationStats::default()
        };
        let (mut stop_reason, continue_decoding) = emit_initial_dflash2_token(
            bonus,
            &eos_tokens,
            max_tokens,
            &mut generated,
            &mut history,
            &mut on_token,
        );
        let decode_start = Instant::now();
        let mut draft_block =
            vec![self.model.config.mask_token_id; DFLASH2_STATIC_VERIFY_WIDTHS[1]];
        let mut sampler_state = None;
        let mut projected_prefix_to_capture = reusable
            .filter(|_| {
                !projected_cache_hit && !segmented_prefill && self.hidden_limit != usize::MAX
            })
            .map(|reuse| {
                (
                    reuse.snapshot.id,
                    prompt_tokens[reuse.cached_tokens..].to_vec(),
                )
            });

        while continue_decoding && generated.len() < max_tokens {
            let remaining = max_tokens - generated.len();
            let round_context_tokens = prompt_tokens.len() + generated.len();
            let policy = self.calibration.policy(round_context_tokens);
            let bs = policy.width;
            let round_compute_start = Instant::now();
            let phase_start = Instant::now();

            // Reuse one maximum-width host buffer. The only staged values are
            // the anchor and a static width from `DFLASH2_STATIC_VERIFY_WIDTHS`;
            // a short final budget never introduces a new compiled MLX shape.
            draft_block[0] = bonus;
            let inputs = mlxcel_core::from_slice_i32(&draft_block[..bs], &[1, bs as i32]);
            let out = self.model.propose(
                &inputs,
                hidden_concat.as_deref(),
                &mut self.caches,
                target,
                policy.selector_edge_scale,
                sampling,
            )?;
            let pending_projected_prefix =
                projected_prefix_to_capture
                    .take()
                    .and_then(|(snapshot_id, committed_suffix)| {
                        Dflash2SlidingCacheSnapshot::capture(&self.caches).map(|caches| {
                            Dflash2ProjectedPrefix {
                                snapshot_id,
                                caches,
                                committed_suffix,
                            }
                        })
                    });
            mlxcel_core::async_eval(&out.path);
            stats.draft_time += phase_start.elapsed();

            // Verify the block against the target in a single batched forward.
            // Keep the proposal on-device. This mirrors dflash-mlx's fast
            // path: target verification consumes the draft array directly,
            // so draft and verify can be submitted without an intervening
            // device-to-host synchronization.
            let bonus_input = mlxcel_core::slice(&inputs, &[0, 0], &[1, 1]);
            let verify_input = mlxcel_core::concatenate(&bonus_input, &out.path, 1);
            let phase_start = Instant::now();
            let verify = target.forward_dflash_verify(&verify_input, &self.target_layer_ids);
            stats.target_forward_calls += 1;
            stats.speculative_rounds += 1;

            let (walk, draft_tokens) = if out.proposal_probs.is_empty() {
                crate::qwen3_5_mtp::greedy_walk_device_proposals(
                    &out.path,
                    &verify.logits,
                    sampling,
                    &history,
                    remaining,
                )
            } else {
                stochastic_dflash2_walk(
                    &out,
                    &verify.logits,
                    sampling,
                    &mut history,
                    &eos_tokens,
                    remaining,
                    &mut sampler_state,
                )
            };
            if let Some(projected_prefix) = pending_projected_prefix {
                // The target walk above has synchronized every draft ancestor.
                // Detaching here retains only the bounded committed K/V, not
                // the large FC/projection construction graph.
                projected_prefix.materialize_and_detach();
                self.projected_prefix = Some(projected_prefix);
            }
            stats.target_verify_time += phase_start.elapsed();
            let compute_latency = round_compute_start.elapsed();
            stats.record_round(policy, walk.accepted, draft_tokens.len());
            self.calibration.observe(
                round_context_tokens,
                policy,
                walk.accepted,
                draft_tokens.len(),
                compute_latency,
            );

            // Emit the accepted prefix (and possibly a corrected token).
            let phase_start = Instant::now();
            let generated_before_round = generated.len();
            let round_stop_reason = emit_dflash2_walk_tokens(
                &walk.new_tokens,
                &eos_tokens,
                max_tokens,
                &mut generated,
                &mut history,
                &mut on_token,
            );
            let emitted = generated.len() - generated_before_round;
            let committed_rows = committed_dflash2_rows(walk.accepted, emitted);
            if committed_rows < bs {
                target.rollback_mtp_verify(&verify.gdn_states, committed_rows - 1, bs, false);
            }
            committed_tokens += committed_rows;

            // The drafter consumes only new committed rows. A separate bounded
            // window is maintained solely when terminal snapshot capture is on.
            let committed_hidden = concatenate_hiddens(&verify.hidden_by_layer, committed_rows);
            if let Some(window) = snapshot_hidden.take() {
                snapshot_hidden = Some(materialize_detached(merge_hidden_context(
                    &window,
                    &committed_hidden,
                    self.hidden_limit,
                )));
            }
            hidden_concat = Some(committed_hidden);
            if round_stop_reason.is_some()
                && capture_final_snapshot
                && committed_tokens == history.len()
            {
                let row = (committed_rows - 1) as i32;
                let shape = mlxcel_core::array_shape(&verify.logits);
                terminal_logits = Some(mlxcel_core::slice(
                    &verify.logits,
                    &[0, row, 0],
                    &[shape[0], row + 1, shape[2]],
                ));
            }
            bonus = *walk
                .new_tokens
                .last()
                .expect("speculative walk emits at least one token");

            // Drafter cache trim to the committed prefix (a no-op while the
            // round wrote only committed context rows).
            for cache in self.caches.iter_mut() {
                cache.trim_to_committed(
                    i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)
                        + i32::try_from(generated.len()).unwrap_or(i32::MAX),
                );
            }
            stats.reconcile_time += phase_start.elapsed();
            if let Some(reason) = round_stop_reason {
                stop_reason = reason;
                break;
            }
        }
        let final_snapshot = if capture_final_snapshot {
            if committed_tokens < history.len() {
                // At most one emitted correction/bonus has not entered target
                // state. Forward that token alone, never replay the prompt.
                debug_assert_eq!(committed_tokens + 1, history.len());
                let input = mlxcel_core::from_slice_i32(&history[committed_tokens..], &[1, 1]);
                let continuation = target.forward_dflash_continuation(
                    &input,
                    &self.target_layer_ids,
                    self.hidden_limit,
                )?;
                stats.target_forward_calls += 1;
                snapshot_hidden = Some(merge_hidden_context(
                    snapshot_hidden
                        .as_deref()
                        .expect("capture retains hidden window"),
                    &continuation.hidden_concat,
                    self.hidden_limit,
                ));
                terminal_logits = Some(continuation.first_logits);
            }
            Some(Dflash2PromptSnapshot::capture(
                target,
                history.len(),
                snapshot_hidden
                    .as_deref()
                    .expect("capture retains hidden window"),
                terminal_logits
                    .as_deref()
                    .expect("capture retains continuation logits"),
                prompt_snapshots
                    .last()
                    .or_else(|| reusable.map(|reuse| reuse.snapshot)),
            )?)
        } else {
            None
        };
        stats.decode_time = decode_start.elapsed();
        Ok(Dflash2Generation {
            token_ids: generated,
            stats,
            cached_tokens,
            prompt_snapshots,
            final_snapshot,
            stop_reason,
        })
    }
}
fn materialize_detached(array: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::eval(&array);
    let ptr = array
        .as_ref()
        .expect("materialized MLX array must be non-null") as *const MlxArray;
    unsafe { mlxcel_core::detach_all(&[ptr]) };
    array
}

fn merge_hidden_context(
    prefix: &MlxArray,
    suffix: &MlxArray,
    hidden_limit: usize,
) -> UniquePtr<MlxArray> {
    let suffix_shape = mlxcel_core::array_shape(suffix);
    if suffix_shape[1] as usize >= hidden_limit {
        let start = suffix_shape[1] - hidden_limit as i32;
        return mlxcel_core::slice(
            suffix,
            &[0, start, 0],
            &[suffix_shape[0], suffix_shape[1], suffix_shape[2]],
        );
    }
    let combined = mlxcel_core::concatenate(prefix, suffix, 1);
    let shape = mlxcel_core::array_shape(&combined);
    if shape[1] as usize <= hidden_limit {
        return combined;
    }
    let start = shape[1] - i32::try_from(hidden_limit).unwrap_or(i32::MAX);
    mlxcel_core::slice(&combined, &[0, start, 0], &[shape[0], shape[1], shape[2]])
}

/// Concatenate a `[1, L, H]` per-target-layer hidden list along `-1`.
fn concatenate_hiddens(hiddens: &[UniquePtr<MlxArray>], prefix_len: usize) -> UniquePtr<MlxArray> {
    debug_assert!(
        !hiddens.is_empty(),
        "DFlash2 verify must capture hidden states"
    );
    let prefix = |hidden: &MlxArray| {
        let shape = mlxcel_core::array_shape(hidden);
        mlxcel_core::slice(hidden, &[0, 0, 0], &[shape[0], prefix_len as i32, shape[2]])
    };
    let mut acc = prefix(hiddens[0].as_ref().expect("captured hidden"));
    for hidden in &hiddens[1..] {
        let hidden = prefix(hidden.as_ref().expect("captured hidden"));
        acc = concatenate(&acc, &hidden, -1);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5::{
        DFLASH_COMPACT_PREFIX, DFLASH_COMPACT_TOKEN_COUNT, DFLASH_CONTROL_END, DFLASH_CONTROL_START,
    };
    use mlxcel_core::generate::SamplingConfig;
    use std::collections::BTreeMap;

    fn raw_i32(array: &MlxArray) -> Vec<i32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_evaluated_bytes(array)
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
            .collect()
    }

    fn raw_f32(array: &MlxArray) -> Vec<f32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_evaluated_bytes(array)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 bytes")))
            .collect()
    }

    fn candidate_map(ids: &MlxArray, logits: &MlxArray) -> BTreeMap<i32, i32> {
        raw_i32(ids)
            .into_iter()
            .zip(raw_f32(logits))
            .map(|(token, score)| (token, score as i32))
            .collect()
    }

    fn verify_logits(target_tokens: &[i32], vocab: usize) -> UniquePtr<MlxArray> {
        let mut logits = vec![-100.0_f32; target_tokens.len() * vocab];
        for (position, &token) in target_tokens.iter().enumerate() {
            logits[position * vocab + token as usize] = 100.0;
        }
        mlxcel_core::from_slice_f32(&logits, &[1, target_tokens.len() as i32, vocab as i32])
    }

    #[test]
    fn stochastic_walk_preserves_history_penalties_and_outside_support_correction() {
        let proposal = SelectorOutput {
            path: mlxcel_core::from_slice_i32(&[1, 1], &[1, 2]),
            candidates: mlxcel_core::from_slice_i32(&[1, 1], &[1, 2, 1]),
            proposal_probs: vec![
                mlxcel_core::from_slice_f32(&[1.0], &[1, 1]),
                mlxcel_core::from_slice_f32(&[1.0], &[1, 1]),
            ],
        };
        let logits = mlxcel_core::from_slice_f32(
            &[
                -100.0, 2.0, -100.0, 0.0, -100.0, 2.0, -100.0, 0.0, -100.0, 2.0, -100.0, 0.0,
            ],
            &[1, 3, 4],
        );
        let sampling = SamplingConfig {
            temperature: 1.0,
            top_p: 0.01,
            presence_penalty: 3.0,
            prompt_token_count: Some(1),
            ..Default::default()
        };
        let mut history = vec![1];
        let (walk, _) = stochastic_dflash2_walk(
            &proposal,
            &logits,
            &sampling,
            &mut history,
            &[],
            3,
            &mut None,
        );
        assert_eq!(walk.accepted, 1);
        assert_eq!(walk.new_tokens, [1, 3]);
        assert_eq!(history, [1]);
    }

    #[test]
    fn stochastic_walk_samples_full_vocabulary_bonus_and_stops_at_budget_or_eos() {
        let proposal = SelectorOutput {
            path: mlxcel_core::from_slice_i32(&[1], &[1, 1]),
            candidates: mlxcel_core::from_slice_i32(&[1], &[1, 1, 1]),
            proposal_probs: vec![mlxcel_core::from_slice_f32(&[1.0], &[1, 1])],
        };
        let logits = verify_logits(&[1, 3], 4);
        let sampling = SamplingConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let mut history = vec![0];
        let (walk, _) = stochastic_dflash2_walk(
            &proposal,
            &logits,
            &sampling,
            &mut history,
            &[],
            2,
            &mut None,
        );
        assert_eq!(walk.accepted, 1);
        assert_eq!(walk.new_tokens, [1, 3]);
        for (eos, budget) in [(&[1][..], 2), (&[][..], 1)] {
            let (walk, _) = stochastic_dflash2_walk(
                &proposal,
                &logits,
                &sampling,
                &mut history,
                eos,
                budget,
                &mut None,
            );
            assert_eq!(walk.accepted, 1);
            assert_eq!(walk.new_tokens, [1]);
            assert_eq!(history, [0]);
        }
    }

    #[test]
    fn stochastic_selector_retains_distribution_conditioned_on_sampled_predecessor() {
        let mut weights = WeightMap::new();
        weights.insert(
            "candidate_selector.predecessor_codebook".to_owned(),
            mlxcel_core::from_slice_f32(&[0.0, 2.0, -2.0, 0.0], &[4, 1]),
        );
        weights.insert(
            "candidate_selector.successor_codebook".to_owned(),
            mlxcel_core::from_slice_f32(&[0.0, 0.0, 1.0, -1.0], &[4, 1]),
        );
        weights.insert(
            "candidate_selector.hidden_projection.weight".to_owned(),
            mlxcel_core::from_slice_f32(&[1.0], &[1, 1]),
        );
        let selector = CandidateSelector::from_weights(
            &weights,
            &DFlash2Config {
                hidden_size: 1,
                selector_rank: 1,
                selector_top_k: 2,
                ..Default::default()
            },
        )
        .expect("tiny selector");
        let hidden = mlxcel_core::from_slice_f32(&[1.0, 1.0], &[1, 2, 1]);
        let candidates = mlxcel_core::from_slice_i32(&[1, 2, 2, 3], &[1, 2, 2]);
        let unary = mlxcel_core::from_slice_f32(&[0.0; 4], &[1, 2, 2]);
        let anchor = mlxcel_core::from_slice_i32(&[0], &[1]);
        let sampling = SamplingConfig {
            temperature: 1.0,
            top_k: 20,
            ..Default::default()
        };
        for seed in 0..16 {
            mlxcel_core::random_seed(seed);
            let out = selector
                .select(&hidden, &candidates, &unary, &anchor, 1.0, &sampling)
                .expect("stochastic selector");
            let path = raw_i32(&out.path);
            assert!(matches!(path[0], 1 | 2));
            assert!(matches!(path[1], 2 | 3));
            let q0 = raw_f32(&out.proposal_probs[0]);
            assert_eq!(q0, [0.5, 0.5]);
            let q1 = raw_f32(&out.proposal_probs[1]);
            let expected = 1.0
                / (1.0
                    + if path[0] == 1 {
                        (-4.0_f32).exp()
                    } else {
                        4.0_f32.exp()
                    });
            assert!((q1[0] - expected).abs() < 1e-6);
            assert!((q1[1] - (1.0 - expected)).abs() < 1e-6);
        }
    }

    fn fixture_config_json() -> Value {
        serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 5120,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000,
            "max_position_embeddings": 262144,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "sliding_window": 2048,
            "is_causal": false,
            "num_target_layers": 64,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248070,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61]
            }
        })
    }

    #[test]
    fn config_parses_dflash_config_sub_object() {
        let config = DFlash2Config::from_json(&fixture_config_json()).expect("parse fixture");
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.block_size, 8);
        assert_eq!(config.conv_kernel_size, 2);
        assert_eq!(config.conv_group_size, 16);
        assert_eq!(config.selector_rank, 256);
        assert_eq!(config.selector_top_k, 16);
        assert_eq!(config.mask_token_id, 248_070);
        assert_eq!(config.target_layer_ids, vec![5, 19, 33, 47, 61]);
        assert_eq!(config.layer_types.len(), 5);
        assert_eq!(config.sliding_window, Some(2048));
        assert_eq!(config.is_causal, Some(false));
        // SGLang default output multiplier.
        assert_eq!(config.output_multiplier, 1.0);
        assert!(config.final_logit_softcapping.is_none());
    }

    #[test]
    fn config_validates_layer_count_and_ids() {
        let mut json = fixture_config_json();
        json["layer_types"] = serde_json::json!(["sliding_attention"]);
        assert!(
            DFlash2Config::from_json(&json).is_err(),
            "layer_types count must match num_hidden_layers"
        );

        let mut json = fixture_config_json();
        json["dflash_config"]["target_layer_ids"] = serde_json::json!([64]);
        assert!(
            DFlash2Config::from_json(&json).is_err(),
            "target_layer_ids must stay below num_target_layers"
        );

        let mut json = fixture_config_json();
        json["dflash_config"] = serde_json::json!({"block_size": 8});
        assert!(
            DFlash2Config::from_json(&json).is_err(),
            "missing mask_token_id / target_layer_ids must fail"
        );
    }

    #[test]
    fn config_rejects_empty_selector_domains() {
        for field in ["selector_rank", "selector_top_k"] {
            let mut json = fixture_config_json();
            json["dflash_config"][field] = serde_json::json!(0);
            let error = DFlash2Config::from_json(&json).expect_err("zero selector dimension");
            assert!(error.contains(field), "{error}");
        }
    }

    #[test]
    fn config_requires_sliding_window_for_sliding_layers() {
        let mut json = fixture_config_json();
        json["sliding_window"] = Value::Null;
        assert!(
            DFlash2Config::from_json(&json).is_err(),
            "sliding_attention layers require sliding_window"
        );
    }

    #[test]
    fn compact_candidate_top_k_matches_full_head_when_top_k_is_representable() {
        let compact_len = DFLASH_COMPACT_TOKEN_COUNT as usize;
        assert_eq!(compact_len, 80_922);
        let peaks = [
            (0_i32, 40.0_f32),
            (DFLASH_COMPACT_PREFIX - 1, 50.0),
            (DFLASH_COMPACT_PREFIX, 60.0),
            (DFLASH_COMPACT_TOKEN_COUNT - 1, 70.0),
        ];

        let mut compact_scores = vec![-1_000.0_f32; compact_len];
        for &(id, score) in &peaks {
            compact_scores[id as usize] = score;
        }
        let compact_logits =
            mlxcel_core::from_slice_f32(&compact_scores, &[1, 1, compact_len as i32]);
        let (compact_ids, compact_values) =
            top_k_candidate_logits(&compact_logits, peaks.len(), true).expect("compact top-k");
        let compact = candidate_map(&compact_ids, &compact_values);

        let mut full_scores = vec![-2_000.0_f32; 248_320];
        for compact_id in 0..DFLASH_COMPACT_TOKEN_COUNT {
            let target_id = Qwen35Model::map_dflash_candidate_token(compact_id);
            full_scores[target_id as usize] = compact_scores[compact_id as usize];
        }
        let full_logits = mlxcel_core::from_slice_f32(&full_scores, &[1, 1, 248_320]);
        let (full_ids, full_values) =
            top_k_candidate_logits(&full_logits, peaks.len(), false).expect("full top-k");
        let full = candidate_map(&full_ids, &full_values);

        assert_eq!(compact, full);
        assert_eq!(
            compact,
            BTreeMap::from([
                (0, 40),
                (DFLASH_COMPACT_PREFIX - 1, 50),
                (DFLASH_CONTROL_START, 60),
                (DFLASH_CONTROL_END - 1, 70),
            ])
        );
        assert!(
            !compact.contains_key(&248_070),
            "input-only mask token must not enter the candidate lattice"
        );
    }

    #[test]
    fn candidate_top_k_rejects_zero_and_domain_size_boundaries() {
        let logits = mlxcel_core::zeros(&[1, 1, 4], mlxcel_core::dtype::FLOAT32);
        assert!(top_k_candidate_logits(&logits, 0, false).is_err());
        assert!(top_k_candidate_logits(&logits, 4, false).is_err());
        assert!(top_k_candidate_logits(&logits, 5, false).is_err());
        assert!(top_k_candidate_logits(&logits, 1, false).is_ok());
    }

    #[test]
    fn projected_sliding_cache_restore_preserves_eviction_order() {
        let initial_values = (0..31).map(|value| value as f32).collect::<Vec<_>>();
        let initial_keys =
            mlxcel_core::from_slice_f32(&initial_values, &[1, 1, initial_values.len() as i32, 1]);
        let initial_vals =
            mlxcel_core::from_slice_f32(&initial_values, &[1, 1, initial_values.len() as i32, 1]);
        let mut original = DFlash2KVCache::Sliding(RotatingKVCache::new(32));
        let DFlash2KVCache::Sliding(cache) = &mut original else {
            unreachable!()
        };
        cache.offset = 69;
        original.update_and_fetch(initial_keys, initial_vals);

        let snapshots = Dflash2SlidingCacheSnapshot::capture(std::slice::from_ref(&original))
            .expect("snapshot");
        Dflash2SlidingCacheSnapshot::materialize_and_detach_all(&snapshots);
        let mut restored =
            Dflash2SlidingCacheSnapshot::restore_all(&snapshots).expect("restore snapshot");

        let appended = [31.0_f32, 32.0, 33.0];
        let original_out = original.update_and_fetch(
            mlxcel_core::from_slice_f32(&appended, &[1, 1, 3, 1]),
            mlxcel_core::from_slice_f32(&appended, &[1, 1, 3, 1]),
        );
        let restored_out = restored[0].update_and_fetch(
            mlxcel_core::from_slice_f32(&appended, &[1, 1, 3, 1]),
            mlxcel_core::from_slice_f32(&appended, &[1, 1, 3, 1]),
        );

        assert_eq!(raw_f32(&original_out.0), raw_f32(&restored_out.0));
        assert_eq!(raw_f32(&original_out.1), raw_f32(&restored_out.1));
        let (DFlash2KVCache::Sliding(original), DFlash2KVCache::Sliding(restored)) =
            (&original, &restored[0])
        else {
            unreachable!()
        };
        assert_eq!(original.snapshot_state(), restored.snapshot_state());
    }

    #[test]
    fn adaptive_width_uses_static_defaults_until_calibrated() {
        let calibration = Dflash2Calibration::default();
        assert_eq!(calibration.policy(63_999).width, 4);
        assert_eq!(calibration.policy(64_000).width, 5);
        assert_eq!(
            calibration.policy(usize::MAX).selector_edge_scale,
            1.0,
            "untrained runtime state must preserve checkpoint scoring"
        );
        assert!(
            calibration
                .contexts
                .iter()
                .flat_map(|context| context.width_rounds)
                .all(|rounds| rounds == 0)
        );
    }

    #[test]
    fn adaptive_width_selects_prefix_survival_over_marginal_latency() {
        let mut faster_wide = Dflash2ContextCalibration::default();
        for _ in 0..201 {
            faster_wide.observe(0, 1, 3, 3, Duration::from_micros(100));
            faster_wide.observe(1, 1, 4, 4, Duration::from_micros(105));
        }
        assert_eq!(
            faster_wide.choose_width_index(0),
            1,
            "one extra surviving prefix token pays for 5us marginal latency"
        );

        let mut expensive_wide = Dflash2ContextCalibration::default();
        for _ in 0..201 {
            expensive_wide.observe(0, 1, 3, 3, Duration::from_micros(100));
            expensive_wide.observe(1, 1, 3, 4, Duration::from_micros(150));
        }
        assert_eq!(
            expensive_wide.choose_width_index(1),
            0,
            "no depth-four survival cannot pay 50us marginal latency"
        );
    }

    #[test]
    fn selector_calibration_optimizes_accepted_prefix_utility() {
        let mut context = Dflash2ContextCalibration::default();
        for _ in 0..200 {
            context.selector_prefix[0].observe(4, 4);
            context.selector_rounds[0] += 1;
            context.selector_prefix[1].observe(1, 4);
            context.selector_rounds[1] += 1;
            context.selector_prefix[2].observe(0, 4);
            context.selector_rounds[2] += 1;
        }
        assert_eq!(context.choose_selector_arm(4), 0);
        assert_eq!(DFLASH2_SELECTOR_EDGE_SCALES[0], 0.75);
    }

    #[test]
    fn selector_edge_scale_changes_the_local_correlation_decision() {
        let mut weights = WeightMap::new();
        weights.insert(
            "candidate_selector.predecessor_codebook".to_owned(),
            mlxcel_core::from_slice_f32(&[1.0; 7], &[7, 1]),
        );
        weights.insert(
            "candidate_selector.successor_codebook".to_owned(),
            mlxcel_core::from_slice_f32(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], &[7, 1]),
        );
        weights.insert(
            "candidate_selector.hidden_projection.weight".to_owned(),
            mlxcel_core::from_slice_f32(&[1.0], &[1, 1]),
        );
        let selector = CandidateSelector::from_weights(
            &weights,
            &DFlash2Config {
                hidden_size: 1,
                selector_rank: 1,
                selector_top_k: 2,
                ..Default::default()
            },
        )
        .expect("tiny selector");
        let hidden = mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1]);
        let candidates = mlxcel_core::from_slice_i32(&[1, 2], &[1, 1, 2]);
        let unary = mlxcel_core::from_slice_f32(&[0.9, 0.0], &[1, 1, 2]);
        let anchor = mlxcel_core::from_slice_i32(&[0], &[1]);

        let unary_favored = selector
            .select(
                &hidden,
                &candidates,
                &unary,
                &anchor,
                0.75,
                &mlxcel_core::generate::SamplingConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .expect("low correlation scale");
        let correlation_favored = selector
            .select(
                &hidden,
                &candidates,
                &unary,
                &anchor,
                1.25,
                &mlxcel_core::generate::SamplingConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .expect("high correlation scale");
        assert_eq!(raw_i32(&unary_favored.path), [1]);
        assert_eq!(raw_i32(&correlation_favored.path), [2]);
    }

    #[test]
    fn target_verification_rejects_drafts_for_winners_outside_candidate_domain() {
        // Both are ordinary tokenizer rows omitted by the compact candidate
        // head: "_verification" and "_human". They must still win verification.
        let proposals = [813, 3208];
        let target_tokens = [813, 81_336, 83_268];
        let logits = verify_logits(&target_tokens, 248_320);
        let sampling = SamplingConfig::greedy();
        let mtp = crate::qwen3_5_mtp::greedy_walk(&proposals, &logits, &sampling, &[], 3);
        assert_eq!(mtp.accepted, 1);
        assert_eq!(mtp.new_tokens, [813, 81_336]);
        let proposal_array = mlxcel_core::from_slice_i32(&proposals, &[1, 2]);
        let (dflash, _) = crate::qwen3_5_mtp::greedy_walk_device_proposals(
            &proposal_array,
            &logits,
            &sampling,
            &[],
            3,
        );
        assert_eq!(dflash.accepted, 1);
        assert_eq!(dflash.new_tokens, mtp.new_tokens);
        let accepted = mlxcel_core::from_slice_i32(&target_tokens[..2], &[1, 2]);
        let (bonus, _) =
            crate::qwen3_5_mtp::greedy_walk_device_proposals(&accepted, &logits, &sampling, &[], 3);
        assert_eq!(bonus.accepted, 2);
        assert_eq!(bonus.new_tokens, target_tokens);
    }

    #[test]
    fn static_width_and_selector_choices_preserve_exact_target_output() {
        let sampling = SamplingConfig::greedy();
        for &width in &DFLASH2_STATIC_VERIFY_WIDTHS {
            let proposals = (1..width as i32).collect::<Vec<_>>();
            for accepted in 0..proposals.len() {
                let mut target_tokens = proposals.clone();
                target_tokens[accepted] = 6;
                target_tokens.push(0);
                let proposal_array =
                    mlxcel_core::from_slice_i32(&proposals, &[1, proposals.len() as i32]);
                let logits = verify_logits(&target_tokens, 7);
                let (walk, materialized) = crate::qwen3_5_mtp::greedy_walk_device_proposals(
                    &proposal_array,
                    &logits,
                    &sampling,
                    &[],
                    width,
                );
                assert_eq!(materialized, proposals);
                assert_eq!(walk.accepted, accepted);
                let mut expected = proposals[..accepted].to_vec();
                expected.push(6);
                assert_eq!(walk.new_tokens, expected);
            }

            let mut target_tokens = proposals.clone();
            target_tokens.push(0);
            let proposal_array =
                mlxcel_core::from_slice_i32(&proposals, &[1, proposals.len() as i32]);
            let logits = verify_logits(&target_tokens, 7);
            let (walk, _) = crate::qwen3_5_mtp::greedy_walk_device_proposals(
                &proposal_array,
                &logits,
                &sampling,
                &[],
                width,
            );
            assert_eq!(walk.accepted, proposals.len());
            assert_eq!(walk.new_tokens, target_tokens);

            let (tail, _) = crate::qwen3_5_mtp::greedy_walk_device_proposals(
                &proposal_array,
                &logits,
                &sampling,
                &[],
                1,
            );
            assert_eq!(tail.new_tokens, target_tokens[..1]);
        }
    }

    #[test]
    fn initial_callback_cancellation_stops_before_a_verify_round() {
        let mut generated = Vec::new();
        let mut history = vec![3, 4];
        let mut callbacks = 0;
        let (reason, continue_decoding) =
            emit_initial_dflash2_token(5, &[], 10, &mut generated, &mut history, &mut |_| {
                callbacks += 1;
                false
            });
        assert_eq!(reason, GenerationStopReason::CallbackCancelled);
        assert!(!continue_decoding);
        assert_eq!(generated, [5]);
        assert_eq!(history, [3, 4, 5]);
        assert_eq!(callbacks, 1);
    }

    #[test]
    fn stopped_walk_commits_only_emitted_verified_rows() {
        // The target verified anchor 10 and proposals 11,12,13. Only 11,12
        // matched; 99 is the correction. Cancellation must not emit 12 or 99
        // after the consumer declined 11, nor retain their target rows.
        let verified = [10, 11, 12, 13];
        let walk = [11, 12, 99];
        for (eos, budget, cancel_after, expected, reason) in [
            (
                Vec::new(),
                20,
                1,
                vec![10, 11],
                GenerationStopReason::CallbackCancelled,
            ),
            (
                vec![12],
                20,
                usize::MAX,
                vec![10, 11],
                GenerationStopReason::Eos,
            ),
            (
                vec![11],
                20,
                usize::MAX,
                vec![10],
                GenerationStopReason::Eos,
            ),
            (
                Vec::new(),
                3,
                usize::MAX,
                vec![10, 11, 12],
                GenerationStopReason::MaxTokens,
            ),
            (
                Vec::new(),
                4,
                usize::MAX,
                vec![10, 11, 12, 99],
                GenerationStopReason::MaxTokens,
            ),
        ] {
            let mut generated = vec![10];
            let mut history = vec![7, 10];
            let mut callbacks = 0;
            let stopped = emit_dflash2_walk_tokens(
                &walk,
                &eos,
                budget,
                &mut generated,
                &mut history,
                &mut |_| {
                    callbacks += 1;
                    callbacks < cancel_after
                },
            );
            assert_eq!(stopped, Some(reason));
            assert_eq!(generated, expected);
            assert_eq!(callbacks, expected.len() - 1);
            assert_eq!(&history[1..], expected);
            let committed = committed_dflash2_rows(2, generated.len() - 1);
            assert_eq!(&verified[..committed], &expected[..committed]);
            assert_eq!(
                &expected[committed..],
                if expected.last() == Some(&99) {
                    &[99][..]
                } else {
                    &[]
                },
                "only an emitted correction may remain unforwarded"
            );
        }
    }

    #[test]
    fn committed_hidden_window_evicts_old_rows_and_excludes_rejections() {
        let mut window = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 4, 1]);
        let verified = vec![mlxcel_core::from_slice_f32(
            &[4.0, 5.0, 90.0, 91.0],
            &[1, 4, 1],
        )];
        let committed = concatenate_hiddens(&verified, committed_dflash2_rows(3, 1));
        window = materialize_detached(merge_hidden_context(&window, &committed, 4));
        assert_eq!(raw_f32(&window), [2.0, 3.0, 4.0, 5.0]);
        let suffix = mlxcel_core::from_slice_f32(&[6.0, 7.0, 8.0, 9.0, 10.0], &[1, 5, 1]);
        window = materialize_detached(merge_hidden_context(&window, &suffix, 4));
        assert_eq!(raw_f32(&window), [7.0, 8.0, 9.0, 10.0]);

        let snapshot = Dflash2PromptSnapshot::from_portable_parts(
            ModelStateSnapshot::new("qwen3.5", 11),
            array_to_portable(None, &window),
            7,
            array_to_portable(None, &verify_logits(&[2], 3)),
        )
        .expect("restore evicted window at its absolute target boundary");
        assert_eq!(snapshot.hidden_offset, 7);
        assert_eq!(raw_f32(&snapshot.hidden_concat), [7.0, 8.0, 9.0, 10.0]);
        let PortablePromptSnapshot::Dflash2 {
            hidden_offset,
            hidden_concat,
            ..
        } = snapshot.to_portable()
        else {
            panic!("DFlash2 snapshot changed family")
        };
        assert_eq!(hidden_offset + hidden_concat.shape[1] as usize, 11);
    }

    #[test]
    fn portable_dflash_snapshot_rejects_misaligned_nonfinite_and_nonfloat_context() {
        let hidden = PortableArray {
            name: None,
            shape: vec![1, 2, 1],
            dtype: mlxcel_core::dtype::FLOAT32,
            bytes: [1.0_f32, 2.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
        };
        let logits = PortableArray {
            shape: vec![1, 1, 2],
            ..hidden.clone()
        };
        let restore = |hidden, offset, logits| {
            Dflash2PromptSnapshot::from_portable_parts(
                ModelStateSnapshot::new("qwen3.5", 8),
                hidden,
                offset,
                logits,
            )
        };
        assert!(restore(hidden.clone(), 5, logits.clone()).is_err());
        assert!(restore(hidden.clone(), usize::MAX, logits.clone()).is_err());
        let mut wrong_layout = logits.clone();
        wrong_layout.shape = vec![1, 2, 1];
        assert!(restore(hidden.clone(), 6, wrong_layout).is_err());
        let mut empty_hidden = hidden.clone();
        empty_hidden.shape[2] = 0;
        assert!(restore(empty_hidden, 6, logits.clone()).is_err());

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid_hidden = hidden.clone();
            invalid_hidden.bytes[..4].copy_from_slice(&invalid.to_ne_bytes());
            assert!(restore(invalid_hidden, 6, logits.clone()).is_err());
            let mut invalid_logits = logits.clone();
            invalid_logits.bytes[..4].copy_from_slice(&invalid.to_ne_bytes());
            assert!(restore(hidden.clone(), 6, invalid_logits).is_err());
        }
        for (dtype, finite, nonfinite) in [
            (mlxcel_core::dtype::FLOAT16, 0x3c00_u16, 0x7c00_u16),
            (mlxcel_core::dtype::BFLOAT16, 0x3f80_u16, 0x7fc0_u16),
        ] {
            let mut half = hidden.clone();
            half.dtype = dtype;
            half.bytes = [finite, finite]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect();
            assert!(restore(half.clone(), 6, logits.clone()).is_ok());
            half.bytes[..2].copy_from_slice(&nonfinite.to_ne_bytes());
            assert!(restore(half, 6, logits.clone()).is_err());
        }
        let mut integer = hidden;
        integer.dtype = mlxcel_core::dtype::INT32;
        assert!(restore(integer, 6, logits).is_err());
    }

    #[test]
    #[ignore = "requires real QW_MODEL_PATH and QW_DFLASH_DRAFT_MODEL_PATH checkpoints (or their default cache paths)"]
    fn real_model_dflash2_checkpoints_and_cancelled_portable_resume() {
        use crate::{ChatMessage, ChatMessageContent, KVCacheMode, PromptSnapshot, Qwen35Provider};

        fn restored(snapshot: PromptSnapshot) -> Dflash2PromptSnapshot {
            let portable = snapshot.to_portable().expect("encode DFlash2 snapshot");
            let PromptSnapshot::Dflash2(snapshot) =
                PromptSnapshot::from_portable(portable).expect("restore portable DFlash2 snapshot")
            else {
                panic!("portable DFlash2 snapshot changed family")
            };
            snapshot
        }

        let model_dir = crate::resolve_model_path(None).expect("resolve real target checkpoint");
        let draft_dir =
            crate::resolve_dflash2_draft_path(None).expect("resolve real draft checkpoint");
        let config_json = std::fs::read(draft_dir.join("config.json")).expect("read draft config");
        let config = DFlash2Config::from_json(
            &serde_json::from_slice(&config_json).expect("parse draft config"),
        )
        .expect("validate draft config");
        let hidden_limit = config.sliding_window.expect("real sliding-window drafter") - 1;
        // FP16 isolates exact state restoration from Turbo4's sensitivity to
        // speculative regrouping after interruption. Default Turbo4 cache
        // storage and end-to-end reuse are exercised separately.
        let mut provider =
            Qwen35Provider::load(&model_dir, KVCacheMode::Fp16).expect("load real DFlash2 target");
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                name: None,
                content: Some(ChatMessageContent::Text(
                    "Accounts require verification before changing contact information. "
                        .repeat(hidden_limit / 4 + 32),
                )),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".to_string(),
                name: None,
                content: Some(ChatMessageContent::Text(
                    "Explain in three detailed paragraphs how a customer should report a \
                     lost debit card, protect their account, and obtain a replacement."
                        .to_string(),
                )),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        ];
        let prompt = provider
            .tokenize_messages(&messages, &[], None, false)
            .expect("tokenize long snapshot fixture");
        assert!(
            prompt.len() > hidden_limit + 32,
            "fixture must exercise hidden-window eviction"
        );
        let sampling = provider.baseline_sampling(
            true,
            crate::SamplingOptions {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(7),
                ..Default::default()
            },
        );
        let control = provider
            .generate_dflash2_baseline_streaming(
                &prompt,
                48,
                &sampling,
                &draft_dir,
                None,
                &[],
                false,
                |_| true,
            )
            .expect("uninterrupted uncached generation");
        assert_eq!(
            control.token_ids.len(),
            48,
            "fixture must reach the requested token budget"
        );
        assert!(control.prompt_snapshots.is_empty() && control.final_snapshot.is_none());

        // Capture alone must not change the live target's numerical state.
        // This control keeps precisely the cold prefill execution schedule.
        let prompt_checkpointed = provider
            .generate_dflash2_baseline_streaming(
                &prompt,
                48,
                &sampling,
                &draft_dir,
                None,
                &[prompt.len()],
                true,
                |_| true,
            )
            .expect("capture full prompt and terminal without structural segmentation");
        assert_eq!(prompt_checkpointed.token_ids, control.token_ids);
        assert_eq!(
            prompt_checkpointed
                .prompt_snapshots
                .iter()
                .map(PromptSnapshot::token_len)
                .collect::<Vec<_>>(),
            [prompt.len()]
        );
        assert_eq!(
            prompt_checkpointed
                .final_snapshot
                .as_ref()
                .expect("terminal capture control")
                .token_len(),
            prompt.len() + control.token_ids.len()
        );
        drop(prompt_checkpointed);

        let boundary = prompt.len() - 17;
        let mut checkpointed = provider
            .generate_dflash2_baseline_streaming(
                &prompt,
                48,
                &sampling,
                &draft_dir,
                None,
                &[prompt.len(), boundary, 0, boundary, prompt.len() + 1],
                true,
                |_| true,
            )
            .expect("capture exact structural, prompt, and terminal boundaries");
        assert_eq!(checkpointed.token_ids, control.token_ids);
        assert_eq!(checkpointed.cached_tokens, 0);
        assert_eq!(
            checkpointed
                .prompt_snapshots
                .iter()
                .map(PromptSnapshot::token_len)
                .collect::<Vec<_>>(),
            [boundary, prompt.len()]
        );
        let terminal = checkpointed
            .final_snapshot
            .take()
            .expect("terminal snapshot");
        assert_eq!(
            terminal.token_len(),
            prompt.len() + checkpointed.token_ids.len()
        );
        let logical_bytes = checkpointed
            .prompt_snapshots
            .iter()
            .chain(std::iter::once(&terminal))
            .map(|snapshot| match snapshot {
                PromptSnapshot::Dflash2(snapshot) => snapshot.nbytes(),
                _ => panic!("DFlash2 generation donated a foreign snapshot"),
            })
            .sum::<usize>();
        eprintln!(
            "DFlash2 retained snapshots=3 logical_bytes={logical_bytes} hidden_limit={hidden_limit}"
        );
        for snapshot in checkpointed
            .prompt_snapshots
            .iter()
            .chain(std::iter::once(&terminal))
        {
            let PromptSnapshot::Dflash2(snapshot) = snapshot else {
                unreachable!()
            };
            assert_eq!(
                mlxcel_core::array_shape(&snapshot.hidden_concat)[1] as usize,
                hidden_limit
            );
            assert_eq!(snapshot.hidden_offset, snapshot.token_len() - hidden_limit);
        }
        drop(terminal);
        let mut checkpoints = checkpointed.prompt_snapshots.into_iter();
        let structural = restored(checkpoints.next().expect("structural checkpoint"));
        let full_prompt = restored(checkpoints.next().expect("prompt checkpoint"));
        for repetition in 0..2 {
            let (warm, stats, cached_tokens) = provider
                .generate_dflash2_cached_streaming(
                    &prompt,
                    48,
                    &sampling,
                    &draft_dir,
                    Some(Dflash2PrefixReuse {
                        snapshot: &structural,
                        cached_tokens: boundary,
                    }),
                    |_| true,
                )
                .expect("continue cached prompt suffix");
            assert_eq!(warm.token_ids, checkpointed.token_ids);
            assert_eq!(cached_tokens, boundary);
            assert_eq!(stats.projected_context_cache_hits, repetition);
        }
        drop(structural);

        // Initial cancellation, cancellation within a verify block, and a
        // max-token tail each donate exactly the emitted prefix, not a bonus-
        // shifted state. Later iterations also reuse memoized prompt K/V while
        // maintaining a raw hidden window for terminal capture.
        for (stop_after, cancel) in [(1, true), (2, true), (7, true), (5, false)] {
            let mut callbacks = 0;
            let mut interrupted = provider
                .generate_dflash2_baseline_streaming(
                    &prompt,
                    if cancel { 48 } else { stop_after },
                    &sampling,
                    &draft_dir,
                    Some(Dflash2PrefixReuse {
                        snapshot: &full_prompt,
                        cached_tokens: prompt.len(),
                    }),
                    &[],
                    true,
                    |_| {
                        callbacks += 1;
                        !cancel || callbacks < stop_after
                    },
                )
                .expect("stop DFlash2 at an emitted boundary");
            assert_eq!(interrupted.token_ids, checkpointed.token_ids[..stop_after]);
            assert_eq!(callbacks, stop_after);
            assert_eq!(interrupted.cached_tokens, prompt.len());
            assert_eq!(
                interrupted.finish_outcome,
                if cancel {
                    GenerationStopReason::CallbackCancelled
                } else {
                    GenerationStopReason::MaxTokens
                }
            );
            let snapshot = restored(interrupted.final_snapshot.take().expect("stopped snapshot"));
            let mut resume_prompt = prompt.clone();
            resume_prompt.extend_from_slice(&interrupted.token_ids);
            assert_eq!(snapshot.token_len(), resume_prompt.len());
            assert_eq!(snapshot.hidden_offset, resume_prompt.len() - hidden_limit);
            let resumed = provider
                .generate_dflash2_baseline_streaming(
                    &resume_prompt,
                    48 - stop_after,
                    &sampling,
                    &draft_dir,
                    Some(Dflash2PrefixReuse {
                        cached_tokens: snapshot.token_len(),
                        snapshot: &snapshot,
                    }),
                    &[],
                    false,
                    |_| true,
                )
                .expect("resume portable cancelled/terminal snapshot");
            assert_eq!(resumed.cached_tokens, resume_prompt.len());
            let mut combined = interrupted.token_ids;
            combined.extend_from_slice(&resumed.token_ids);
            assert_eq!(
                combined, checkpointed.token_ids,
                "resumed tokens differ at stop={stop_after}"
            );
        }

        // Turn one of the real verified tokens into a stop token, then remove
        // that policy on resume. The un-emitted EOS row must not enter state.
        let eos_index = checkpointed
            .token_ids
            .iter()
            .position(|&token| token != checkpointed.token_ids[0])
            .expect("fixture contains a noninitial stop token");
        for eos_index in [0, eos_index] {
            let mut eos_sampling = sampling.clone();
            eos_sampling.stop_token_ids = vec![checkpointed.token_ids[eos_index]];
            let mut stopped = provider
                .generate_dflash2_baseline_streaming(
                    &prompt,
                    48,
                    &eos_sampling,
                    &draft_dir,
                    Some(Dflash2PrefixReuse {
                        snapshot: &full_prompt,
                        cached_tokens: prompt.len(),
                    }),
                    &[],
                    true,
                    |_| true,
                )
                .expect("stop before emitting EOS");
            assert_eq!(stopped.finish_outcome, GenerationStopReason::Eos);
            assert_eq!(stopped.token_ids, checkpointed.token_ids[..eos_index]);
            let snapshot = restored(stopped.final_snapshot.take().expect("EOS snapshot"));
            let mut resume_prompt = prompt.clone();
            resume_prompt.extend_from_slice(&stopped.token_ids);
            assert_eq!(snapshot.token_len(), resume_prompt.len());
            let resumed = provider
                .generate_dflash2_baseline_streaming(
                    &resume_prompt,
                    48 - eos_index,
                    &sampling,
                    &draft_dir,
                    Some(Dflash2PrefixReuse {
                        cached_tokens: snapshot.token_len(),
                        snapshot: &snapshot,
                    }),
                    &[],
                    false,
                    |_| true,
                )
                .expect("resume before the un-emitted EOS token");
            assert_eq!(resumed.token_ids, checkpointed.token_ids[eos_index..]);
        }
    }

    /// Synthetic conv check: with `base_kernel[side][tap=0]` = 1 and
    /// `kernel_projection` = 0, the conv is the identity (delta contributes
    /// nothing, tap 0 base = 1). With a tap-1 base of 1, output at position
    /// `p` is `x[p] + x[p-1]` (zero at p = 0) — the within-block causal
    /// two-tap semantics.
    #[test]
    fn grouped_conv_identity_and_two_tap() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        let hidden: usize = 4;
        let group_size: usize = 2;
        let taps: usize = 2;
        let groups = hidden / group_size;
        // base_kernel [2, taps, hidden]; side 0 (prepare) tap 0 and tap 1
        // both identity over all channels, side 1 (finish) all zeros.
        let mut base = vec![0.0_f32; 2 * taps * hidden];
        base[..taps * hidden].fill(1.0);
        weights.insert(
            "conv.base_kernel".to_owned(),
            mlxcel_core::from_slice_f32(&base, &[2, taps as i32, hidden as i32]),
        );
        weights.insert(
            "conv.kernel_projection.weight".to_owned(),
            mlxcel_core::zeros(
                &[(2 * taps * groups) as i32, hidden as i32],
                mlxcel_core::dtype::FLOAT32,
            ),
        );
        let config = DFlash2Config {
            hidden_size: hidden,
            conv_kernel_size: taps,
            conv_group_size: group_size,
            block_size: 4,
            ..Default::default()
        };
        let conv = DFlash2GroupedConv::from_weights(&weights, "conv", &config).expect("conv");
        // x = [[1, 2, 3, 4], [5, 6, 7, 8]] rows over hidden channels.
        let x = mlxcel_core::from_slice_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 2, hidden as i32],
        );
        // Projected delta is zero; side-0 base: tap0 = 1, tap1 = 1.
        let (out, kernel) = conv.prepare(&x);
        mlxcel_core::eval(&out);
        let out_shape = mlxcel_core::array_shape(&out);
        let mut got = Vec::new();
        for i in 0..(out_shape[1] * out_shape[2]) {
            let pos = mlxcel_core::slice(
                &out,
                &[0, (i / out_shape[2]) as i32, (i % out_shape[2]) as i32],
                &[
                    1,
                    (i / out_shape[2]) as i32 + 1,
                    (i % out_shape[2]) as i32 + 1,
                ],
            );
            got.push(mlxcel_core::item_f32(&pos));
        }
        // Row 0: out[j][c] = x[j][c] + x[j-1][c] (x[-1] = 0): [1, 2, 3, 4].
        // Row 1: [5+1, 6+2, 7+3, 8+4] = [6, 8, 10, 12].
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        // finish with a zero kernel is a no-op.
        let finished = conv.finish(&out, &kernel);
        mlxcel_core::eval(&finished);
        assert_eq!(mlxcel_core::array_shape(&finished), out_shape);
    }
}
