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

//! DFlash2 block-diffusion drafter and greedy round loop.
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
//! with the target's LM head (SGLang `compute_candidates`). The round loop
//! below is the B=1 greedy path only (lossless speculative decoding: greedy
//! output matches the target exactly); the stochastic rejection-sampling
//! path is intentionally not ported yet.
//!
//! Apple Silicon precision rules apply (see
//! `docs/apple-silicon-precision.md`): the drafter runs in the checkpoint's
//! native bf16 — an f16 conversion overflows the drafter's gate*up
//! activations (values beyond f16's 65504 max poison the residual stream
//! with inf → NaN). Attention masks are the one exception — additive 0/-inf
//! f32 sentinels, matching every other mask builder in the repo.

use std::time::{Duration, Instant};

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, Linear, RMSNorm, RotatingKVCache, UnifiedEmbedding};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr, concatenate, multiply_scalar};
use serde_json::Value;

use crate::qwen3_5::Qwen35Model;

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
            _ => {
                return Err(
                    "config layer_types is required for the DFlash2 drafter".to_owned()
                )
            }
        };
        if layer_types.len() != num_hidden_layers {
            return Err(format!(
                "layer_types must have one entry per draft layer: \
                 got {} entries for num_hidden_layers = {num_hidden_layers}",
                layer_types.len()
            ));
        }
        let sliding_window = match get("sliding_window") {
            Some(value) => Some(
                value
                    .as_u64()
                    .ok_or("sliding_window must be an integer")? as usize,
            ),
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
            Some(value) => Some(
                value
                    .as_bool()
                    .ok_or("is_causal must be a boolean")?,
            ),
            None => None,
        };

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
            selector_rank: get_int("selector_rank", default_selector_rank())?,
            selector_top_k: get_int("selector_top_k", default_selector_top_k())?,
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
    kernel_projection: Linear,
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
            Linear::from_weights(weights, &format!("{prefix}.kernel_projection"))?;
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
        let side0 = mlxcel_core::reshape(
            &side0,
            &[shape[0], shape[1], self.taps, self.num_groups],
        );
        let side1 = mlxcel_core::reshape(
            &side1,
            &[shape[0], shape[1], self.taps, self.num_groups],
        );
        let base_shape = mlxcel_core::array_shape(&self.base_kernel);
        let base0 = mlxcel_core::slice(
            &self.base_kernel,
            &[0, 0, 0],
            &[1, self.taps, base_shape[2]],
        );
        (
            self.convolve(hidden, &side0, &base0),
            side1,
        )
    }

    /// `DFlashGroupedConv.finish`: convolve `hidden` with the kernel returned
    /// by [`Self::prepare`].
    pub fn finish(&self, hidden: &MlxArray, dynamic: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let base1 = mlxcel_core::slice(
            &self.base_kernel,
            &[1, 0, 0],
            &[2, self.taps, shape[2]],
        );
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
                let tail = mlxcel_core::slice(
                    &blocks,
                    &[0, 0, 0, 0],
                    &[batch, length - tap, groups, gs],
                );
                let zero_block =
                    mlxcel_core::zeros(&[batch, tap, groups, gs], hidden_dtype);
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
            let delta_tap = mlxcel_core::slice(
                dynamic,
                &[0, 0, tap, 0],
                &[batch, length, tap + 1, groups],
            );
            let delta_tap = mlxcel_core::reshape(
                &delta_tap,
                &[batch, length, groups, 1],
            );
            let delta_tap = mlxcel_core::astype(&delta_tap, hidden_dtype);
            // base_tap [1, 1, groups, gs]
            let base_tap = mlxcel_core::slice(
                &base,
                &[0, tap, 0, 0],
                &[1, tap + 1, groups, gs],
            );
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

/// DFlash2 split-projection attention, ported from SGLang `DFlashAttention`
/// (dflash.py). The attention over the draft block is non-causal (the
/// block-diffusion forward denoises all block positions at once); sliding
/// layers additionally bound the attended context through the rotating
/// cache, and `config.is_causal` may force causal attention.
pub struct DFlash2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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
        let q_proj = Linear::from_weights(weights, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::from_weights(weights, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::from_weights(weights, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::from_weights(weights, &format!("{prefix}.o_proj"))?;
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
    /// - `x_ctx` — context buffer `[B, S, hidden_size]`.
    /// - `cache` — this layer's K/V cache; updated in place with the
    ///   context-side K/V only.
    ///
    /// Returns `[B, L, hidden_size]`.
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: &MlxArray,
        cache: &mut DFlash2KVCache,
    ) -> UniquePtr<MlxArray> {
        let x_shape = mlxcel_core::array_shape(x);
        let ctx_shape = mlxcel_core::array_shape(x_ctx);
        let b = x_shape[0];
        let l = x_shape[1];
        let mut s = ctx_shape[1];

        // Sliding trim: keep at most `sliding_window - 1` context rows
        // (SGLang `get_dflash_attention_sliding_window_size` converts the HF
        // window, which includes the current token, to a window_left).
        let mut x_ctx = mlxcel_core::copy(x_ctx);
        if self.is_sliding && s > self.sliding_window - 1 {
            let skip = s - (self.sliding_window - 1);
            x_ctx = mlxcel_core::slice(&x_ctx, &[0, skip, 0], &[b, s, ctx_shape[2]]);
            s = self.sliding_window - 1;
            match cache {
                DFlash2KVCache::Sliding(cache) => cache.offset += skip,
                DFlash2KVCache::Full(_) => {}
            }
        }

        // Project context and proposal separately (SGLang's fused QKV is the
        // same linear map; the checkpoint stores q/k/v independently).
        let queries = self.q_proj.forward(x);
        let ctx_keys = self.k_proj.forward(&x_ctx);
        let ctx_values = self.v_proj.forward(&x_ctx);
        let prop_keys = self.k_proj.forward(x);
        let prop_values = self.v_proj.forward(x);

        // Reshape to [B, seq, n_*, head_dim] for the per-head norms.
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.n_heads, self.head_dim]);
        let ctx_keys = mlxcel_core::reshape(&ctx_keys, &[b, s, self.n_kv_heads, self.head_dim]);
        let ctx_values = mlxcel_core::reshape(&ctx_values, &[b, s, self.n_kv_heads, self.head_dim]);
        let prop_keys = mlxcel_core::reshape(&prop_keys, &[b, l, self.n_kv_heads, self.head_dim]);
        let prop_values =
            mlxcel_core::reshape(&prop_values, &[b, l, self.n_kv_heads, self.head_dim]);

        let queries = self.q_norm.forward(&queries);
        let ctx_keys = self.k_norm.forward(&ctx_keys);
        let prop_keys = self.k_norm.forward(&prop_keys);

        // Transpose to [B, n_heads, seq, head_dim].
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let ctx_keys = mlxcel_core::transpose_axes(&ctx_keys, &[0, 2, 1, 3]);
        let ctx_values = mlxcel_core::transpose_axes(&ctx_values, &[0, 2, 1, 3]);
        let prop_keys = mlxcel_core::transpose_axes(&prop_keys, &[0, 2, 1, 3]);
        let prop_values = mlxcel_core::transpose_axes(&prop_values, &[0, 2, 1, 3]);

        // RoPE offsets (absolute positions): the context rows sit at
        // [past_offset, past_offset + S); the proposal block right after.
        let past_offset = cache.offset();
        let after_ctx_offset = past_offset + s;
        let queries = mlxcel_core::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            after_ctx_offset,
        );
        let ctx_keys = mlxcel_core::fast_rope(
            &ctx_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            past_offset,
        );
        let prop_keys = mlxcel_core::fast_rope(
            &prop_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            after_ctx_offset,
        );

        // Cache write: ONLY context K/V. The proposal K/V is concatenated
        // post-hoc and never enters the cache. This is the load-bearing
        // invariant of the DFlash drafter forward — the next round's offset
        // and RoPE positions depend on it (pin it so a future edit cannot
        // silently regress to writing proposal K/V into the cache).
        let (keys, values) = cache.update_and_fetch(ctx_keys, ctx_values);
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
    fn build_mask(
        &self,
        l: i32,
        s: i32,
        keys_combined: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
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
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl DFlash2Mlp {
    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let gate = Linear::from_weights(weights, &format!("{prefix}.gate_proj"))?;
        let up = Linear::from_weights(weights, &format!("{prefix}.up_proj"))?;
        let down = Linear::from_weights(weights, &format!("{prefix}.down_proj"))?;
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
        x_ctx: &MlxArray,
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

/// Output of a greedy selector walk.
pub struct SelectorOutput {
    /// Draft token ids `[1, L]` int32, one per position after the anchor.
    pub path: UniquePtr<MlxArray>,
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
/// Greedy-only: the stochastic rejection-sampling branch is intentionally
/// not implemented; callers must pass `temperature <= 0`.
pub struct CandidateSelector {
    pub top_k: usize,
    predecessor_codebook: UniquePtr<MlxArray>, // [vocab, rank]
    successor_codebook: UniquePtr<MlxArray>,   // [vocab, rank]
    hidden_projection: Linear,                 // hidden -> rank
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
            Linear::from_weights(weights, "candidate_selector.hidden_projection")?;
        Ok(Self {
            top_k: config.selector_top_k,
            predecessor_codebook,
            successor_codebook,
            hidden_projection,
        })
    }

    /// Greedy path walk over the masked block positions.
    ///
    /// - `hidden` `[1, L, H]` — the drafter's post-`norm` output for the
    ///   proposal positions (SGLang `pred_hidden`, position 0 = anchor
    ///   excluded).
    /// - `candidates` `[1, L, K]` int32 — top-K candidate ids per position
    ///   (from the target LM head).
    /// - `unary` `[1, L, K]` — the transformed top-K logits.
    /// - `anchor_ids` `[1]` int32 — the token immediately before the block.
    ///
    /// Returns the chosen draft token path `[1, L]` int32.
    pub fn select(
        &self,
        hidden: &MlxArray,
        candidates: &MlxArray,
        unary: &MlxArray,
        anchor_ids: &MlxArray,
    ) -> Result<SelectorOutput, String> {
        let logits_shape = mlxcel_core::array_shape(unary);
        let batch = logits_shape[0];
        let npos = logits_shape[1];
        let k = logits_shape[2];
        let hidden_proj = self.hidden_projection.forward(hidden);
        let rank = mlxcel_core::array_shape(&hidden_proj)[2];

        // Sequential greedy walk (equivalent to following the SGLang lattice
        // maps `maps[e][path[e]]`). Edges follow the authoritative einsum
        // `"blpr,blcr->blpc"`: pred*proj(h) dotted with the successor rows.
        let anchor = mlxcel_core::reshape(anchor_ids, &[batch]);
        let mut predecessor = anchor; // [B] int32
        let mut path_rows = Vec::with_capacity(npos as usize);
        for position in 0..npos {
            let pred_emb = mlxcel_core::embedding(&self.predecessor_codebook, &predecessor); // [B, rank]
            let candidate_slice = mlxcel_core::slice(
                candidates,
                &[0, position, 0],
                &[batch, position + 1, k],
            );
            let succ_emb =
                mlxcel_core::embedding(&self.successor_codebook, &candidate_slice); // [B, K, rank]
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
            let unary_row = mlxcel_core::slice(
                unary,
                &[0, position, 0],
                &[batch, position + 1, k],
            );
            let scores = mlxcel_core::add(&unary_row, &edges); // [B, K]
            let selected = mlxcel_core::argmax(&scores, -1, false); // [B]
            let candidate_row = mlxcel_core::slice(
                candidates,
                &[0, position, 0],
                &[batch, position + 1, k],
            );
            let selected_emb = mlxcel_core::expand_dims(&selected, -1); // [B, 1]
            let selected_id = mlxcel_core::take_along_axis(&candidate_row, &selected_emb, -1); // [B, 1]
            predecessor = mlxcel_core::reshape(&selected_id, &[batch]); // [B]
            path_rows.push(mlxcel_core::share(&predecessor));
        }
        let path_ptrs = path_rows.iter().map(|row| row.as_ptr()).collect::<Vec<_>>();
        let path = mlxcel_core::stack(&path_ptrs, 1);
        Ok(SelectorOutput { path })
    }
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
    fc: Linear, // [len(target_layer_ids) * hidden, hidden]
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
        let fc = Linear::from_weights(weights, "fc")?;
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
    pub fn hidden_states(
        &self,
        inputs: &MlxArray,
        target_hidden: &MlxArray,
        caches: &mut [DFlash2KVCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(inputs);
        let fc_out = self.fc.forward(target_hidden);
        let h_ctx = self.hidden_norm.forward(&fc_out);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, &h_ctx, cache);
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

    /// Top-K candidates per position via the target's LM head (SGLang
    /// `compute_candidates`): `hidden [1, L, H]` → `candidates [1, L, K]`
    /// int32 ids + `unary [1, L, K]` transformed logits.
    pub fn compute_candidates(
        &self,
        hidden: &MlxArray,
        target: &Qwen35Model,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let logits = target.project_logits(hidden);
        let logits_shape = mlxcel_core::array_shape(&logits);
        let k = i32::try_from(self.candidate_selector.top_k)
            .map_err(|_| "selector_top_k too large".to_owned())?;
        let vocab = logits_shape[2];
        if k >= vocab {
            return Err(format!(
                "selector_top_k={k} must be smaller than the target vocab size {vocab}"
            ));
        }
        let args = mlxcel_core::argpartition(&logits, -k, -1);
        let candidates = mlxcel_core::slice(
            &args,
            &[0, 0, vocab - k],
            &[logits_shape[0], logits_shape[1], vocab],
        );
        let candidates = mlxcel_core::contiguous(&candidates, false);
        let unary = self.transform_unary_logits(&mlxcel_core::take_along_axis(
            &logits,
            &candidates,
            -1,
        ));
        Ok((candidates, unary))
    }

    /// One masked-forward draft round: hidden → target-head top-K candidates
    /// → greedy selector path. Returns `[1, bs-1]` draft tokens.
    pub fn propose(
        &self,
        inputs: &MlxArray,
        target_hidden: &MlxArray,
        caches: &mut [DFlash2KVCache],
        target: &Qwen35Model,
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
        self.candidate_selector
            .select(&pred_hidden, &candidates, &unary, &anchor)
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

// ---------------------------------------------------------------------------
// # 8. Greedy round loop + generator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dflash2GenerationStats {
    pub accepted_draft_tokens: usize,
    pub proposed_draft_tokens: usize,
    /// Wall-clock time spent in the post-prefill DFlash2 decode loop.
    pub decode_time: Duration,
    pub draft_time: Duration,
    pub target_verify_time: Duration,
    pub walk_time: Duration,
    pub reconcile_time: Duration,
    pub target_forward_calls: usize,
    pub speculative_rounds: usize,
}

impl Dflash2GenerationStats {
    pub fn acceptance_percentage(self) -> f64 {
        if self.proposed_draft_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_draft_tokens as f64 * 100.0
        }
    }

    fn record_round(&mut self, accepted: usize, proposed: usize) {
        self.accepted_draft_tokens += accepted;
        self.proposed_draft_tokens += proposed;
    }
}

pub(crate) struct Dflash2Generation {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) stats: Dflash2GenerationStats,
}

/// DFlash2 generation driver: prefill → speculative draft/verify rounds.
///
/// B=1 greedy only: `sampling` must be temperature-0 / top-k-1 (the
/// stochastic DFlash2 rejection-sampling path is not ported). Lossless by
/// construction: the greedy walk only accepts draft tokens that match the
/// target's argmax.
pub struct Qwen35Dflash2Generator {
    model: DFlash2DraftModel,
    caches: Vec<DFlash2KVCache>,
    block_size: usize,
    target_layer_ids: Vec<usize>,
    hidden_limit: usize,
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
        Ok(Self {
            model,
            caches,
            block_size: config.block_size,
            target_layer_ids: config.target_layer_ids.clone(),
            hidden_limit,
        })
    }

    /// Generate greedily with DFlash2 draft verification.
    pub fn generate_streaming<F: FnMut(i32) -> bool>(
        &mut self,
        target: &Qwen35Model,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &mlxcel_core::generate::SamplingConfig,
        mut on_token: F,
    ) -> Result<Dflash2Generation, String> {
        mlxcel_core::generation_policy::seed_rng_if_needed(sampling);
        let eos_tokens = mlxcel_core::generation_policy::merged_eos_token_ids(
            target.eos_token_ids(),
            &sampling.stop_token_ids,
        );
        if max_tokens == 0 {
            return Ok(Dflash2Generation {
                token_ids: Vec::new(),
                stats: Dflash2GenerationStats::default(),
            });
        }

        // Prefill: capture the target-layer hidden states the drafter attends
        // to (trimmed to the sliding window) and sample the first token.
        let prompt_array = mlxcel_core::from_slice_i32(
            prompt_tokens,
            &[1, i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX)],
        );
        let prefill = target.forward_dflash_prefill(
            &prompt_array,
            &self.target_layer_ids,
            self.hidden_limit,
        )?;
        let mut hidden_concat = prefill.hidden_concat;
        // The drafter caches start at the same logical position as the
        // captured target hidden (which may have dropped leading window rows
        // during prefill).
        for cache in self.caches.iter_mut() {
            match cache {
                DFlash2KVCache::Full(c) => c.offset = prefill.hidden_offset as i32,
                DFlash2KVCache::Sliding(c) => c.offset = prefill.hidden_offset as i32,
            }
        }
        let mut bonus = {
            let (token, _) = mlxcel_core::sampling::sample_token_optimized(
                &prefill.first_logits,
                sampling,
                prompt_tokens,
            );
            mlxcel_core::eval(&token);
            mlxcel_core::item_i32(&token)
        };
        let mut generated = Vec::with_capacity(max_tokens);
        let mut history = prompt_tokens.to_vec();
        let mut stats = Dflash2GenerationStats::default();
        if eos_tokens.contains(&bonus) {
            // No tokens emitted; the caller observes the empty token list.
        } else {
            generated.push(bonus);
            history.push(bonus);
            let _ = on_token(bonus);
        }
        let decode_start = Instant::now();

        while generated.len() < max_tokens {
            let remaining = max_tokens - generated.len();
            let bs = self.block_size.min(remaining + 1);
            if bs <= 1 {
                break;
            }
            let proposal_count = bs - 1;
            let phase_start = Instant::now();

            // Propose a draft block over mask tokens: [bonus, mask, ..., mask].
            let mut block = Vec::with_capacity(bs);
            block.push(bonus);
            block.extend(std::iter::repeat_n(
                self.model.config.mask_token_id,
                proposal_count,
            ));
            let inputs = mlxcel_core::from_slice_i32(&block, &[1, bs as i32]);
            let out = self
                .model
                .propose(&inputs, &hidden_concat, &mut self.caches, target)?;
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
            mlxcel_core::eval(&verify.logits);
            stats.target_verify_time += phase_start.elapsed();
            stats.target_forward_calls += 1;
            let draft_tokens = materialize_i32(&out.path);
            stats.speculative_rounds += 1;

            // Greedy walk over the verified block.
            let phase_start = Instant::now();
            let walk = crate::qwen3_5_mtp::greedy_walk(
                &draft_tokens,
                &verify.logits,
                sampling,
                &history,
                remaining,
            );
            stats.walk_time += phase_start.elapsed();
            stats.record_round(walk.accepted, draft_tokens.len());

            // Emit the accepted prefix (and possibly a corrected token).
            let phase_start = Instant::now();
            let round_stop_reason = crate::qwen3_5_mtp::emit_walk_tokens(
                &walk.new_tokens,
                &eos_tokens,
                max_tokens,
                &mut generated,
                &mut history,
                &mut on_token,
            );
            if walk.accepted < draft_tokens.len() {
                target.rollback_mtp_verify(&verify.gdn_states, walk.accepted, bs, false);
            }

            // Next context: the target-layer hidden states of the accepted
            // prefix (the verify captured hiddens for the whole block).
            let verify_retained = concatenate_hiddens(&verify.hidden_by_layer);
            let retained_shape = mlxcel_core::array_shape(&verify_retained);
            let accepted_plus_one = i32::try_from(walk.accepted + 1).unwrap_or(i32::MAX);
            hidden_concat = mlxcel_core::slice(
                &verify_retained,
                &[0, 0, 0],
                &[retained_shape[0], accepted_plus_one, retained_shape[2]],
            );
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
            if round_stop_reason.is_some() {
                break;
            }
        }
        stats.decode_time = decode_start.elapsed();
        Ok(Dflash2Generation {
            token_ids: generated,
            stats,
        })
    }
}

/// Materialize a `[1, L]` int32 array into a host `Vec<i32>`.
fn materialize_i32(array: &MlxArray) -> Vec<i32> {
    mlxcel_core::eval(array);
    let shape = mlxcel_core::array_shape(array);
    let total = shape[0] * shape[1];
    let mut out = Vec::with_capacity(total as usize);
    for i in 0..total {
        let pos = mlxcel_core::slice(
            array,
            &[(i / shape[1]) as i32, (i % shape[1]) as i32],
            &[(i / shape[1]) as i32 + 1, (i % shape[1]) as i32 + 1],
        );
        out.push(mlxcel_core::item_i32(&pos));
    }
    out
}

/// Concatenate a `[1, L, H]` per-target-layer hidden list along `-1`.
fn concatenate_hiddens(hiddens: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let n = hiddens.len();
    debug_assert!(n > 0, "DFlash2 verify must capture hidden states");
    // Concatenate along the last (hidden) axis, mirroring the SGLang target
    // feature capture (`extract_context_feature`).
    let mut acc = mlxcel_core::share(hiddens[0].as_ref().expect("captured hidden"));
    for hid in &hiddens[1..] {
        acc = concatenate(&acc, hid.as_ref().expect("captured hidden"), -1);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn config_requires_sliding_window_for_sliding_layers() {
        let mut json = fixture_config_json();
        json["sliding_window"] = Value::Null;
        assert!(
            DFlash2Config::from_json(&json).is_err(),
            "sliding_attention layers require sliding_window"
        );
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
                &[1, (i / out_shape[2]) as i32 + 1, (i % out_shape[2]) as i32 + 1],
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
