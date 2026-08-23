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

use std::sync::LazyLock;

use cxx::UniquePtr;

use crate::{MlxArray, dtype, ffi};

use super::quant::{TurboQuantParams, turbo4_k_rotate, turbo4_v_inverse_rotate};

pub const TURBO4_FUSED_ATTENTION_ENV_VAR: &str = "MLXCEL_TURBO4_FUSED_ATTENTION";
fn parse_fused_attention_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

static TURBO4_FUSED_ATTENTION_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    let value = std::env::var(TURBO4_FUSED_ATTENTION_ENV_VAR).ok();
    parse_fused_attention_enabled(value.as_deref())
});

pub fn turbo4_fused_attention_enabled() -> bool {
    cfg!(target_os = "macos") && ffi::metal_is_available() && *TURBO4_FUSED_ATTENTION_ENABLED
}

fn supported_inputs(
    q: &MlxArray,
    k_packed: &MlxArray,
    k_rescale: &MlxArray,
    v_packed: &MlxArray,
    v_rescale: &MlxArray,
    params: &TurboQuantParams,
    causal: bool,
) -> bool {
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(k_packed);
    let kr_shape = ffi::array_shape(k_rescale);
    let v_shape = ffi::array_shape(v_packed);
    let vr_shape = ffi::array_shape(v_rescale);
    if q_shape.len() != 4
        || k_shape.len() != 4
        || kr_shape.len() != 4
        || v_shape.len() != 4
        || vr_shape.len() != 4
    {
        return false;
    }

    let q_dtype = ffi::array_dtype(q);
    if !matches!(q_dtype, dtype::FLOAT16 | dtype::BFLOAT16 | dtype::FLOAT32)
        || ffi::array_dtype(k_packed) != dtype::UINT8
        || ffi::array_dtype(v_packed) != dtype::UINT8
        || ffi::array_dtype(k_rescale) != dtype::FLOAT16
        || ffi::array_dtype(v_rescale) != dtype::FLOAT16
    {
        return false;
    }

    let [batch, hq, tq, dim] = [q_shape[0], q_shape[1], q_shape[2], q_shape[3]];
    let [k_batch, hkv, tk, packed_dim] = [k_shape[0], k_shape[1], k_shape[2], k_shape[3]];
    batch > 0
        && hq > 0
        && hkv > 0
        && (2..=4).contains(&tq)
        && tk > 2048
        && hq % hkv == 0
        && hq / hkv <= 32
        && dim >= 32
        && dim <= 256
        && dim % 32 == 0
        && (dim & (dim - 1)) == 0
        && dim as u32 == params.head_dim
        && k_batch == batch
        && packed_dim == dim / 2
        && v_shape == k_shape
        && kr_shape == [batch, hkv, tk, 1]
        && vr_shape == [batch, hkv, tk, 1]
        && causal
        && params.codebook.centroids.len() == 16
}

#[allow(clippy::too_many_arguments)]
pub fn attention_turbo4_fused(
    q: &MlxArray,
    k_packed: &MlxArray,
    k_rescale: &MlxArray,
    v_packed: &MlxArray,
    v_rescale: &MlxArray,
    params: &TurboQuantParams,
    scale: f32,
    causal: bool,
) -> Option<UniquePtr<MlxArray>> {
    if !turbo4_fused_attention_enabled()
        || !supported_inputs(q, k_packed, k_rescale, v_packed, v_rescale, params, causal)
    {
        return None;
    }

    let q_rot = turbo4_k_rotate(q, params);
    let codebook = ffi::from_slice_f32(
        params.codebook.centroids.as_ref(),
        &[params.codebook.centroids.len() as i32],
    );
    let rotated = ffi::turbo4_attention(
        &q_rot, k_packed, k_rescale, v_packed, v_rescale, &codebook, scale, causal,
    );
    Some(turbo4_v_inverse_rotate(&rotated, params))
}

#[cfg(test)]
mod tests {
    use super::parse_fused_attention_enabled;

    #[test]
    fn turbo4_mtp_verify_gate_is_default_on_and_accepts_false_literals() {
        assert!(parse_fused_attention_enabled(None));
        for value in ["1", "true", "on", "yes", "", "invalid"] {
            assert!(parse_fused_attention_enabled(Some(value)));
        }
        for value in ["0", "false", "off", "no", " FALSE "] {
            assert!(!parse_fused_attention_enabled(Some(value)));
        }
    }
}
