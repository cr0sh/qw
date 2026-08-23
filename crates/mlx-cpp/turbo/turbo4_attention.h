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

#pragma once

#include <mlx/array.h>

namespace mlxcel::turbo {

// Exact symmetric-Turbo4 attention over packed K/V. Q is already in the
// rotated K basis; the FP32 result remains in the rotated V basis.
mlx::core::array turbo4_attention(
    const mlx::core::array& q_rot,       // [B, Hq, Tq, D] f32
    const mlx::core::array& k_packed,    // [B, Hkv, Tk, D/2] u8
    const mlx::core::array& k_rescale,   // [B, Hkv, Tk, 1] f16
    const mlx::core::array& v_packed,    // [B, Hkv, Tk, D/2] u8
    const mlx::core::array& v_rescale,   // [B, Hkv, Tk, 1] f16
    const mlx::core::array& codebook,    // [16] f32
    float scale,
    bool causal);

} // namespace mlxcel::turbo
