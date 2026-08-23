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

#include "turbo4_attention.h"

#include <mlx/backend/metal/metal.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

constexpr int SIMD_WIDTH = 32;
constexpr int MIN_MTP_VERIFY_TOKENS = 2048;

// Two-pass block decomposition adapted from oMLX TurboQuant's fused MTP verify
// pass at jundot/omlx@309e8c51e7f46b6da355485018477c25c6f77b23
// (Apache-2.0). The staged multi-head reuse below follows the threadgroup
// tiling approach used by antirez/ds4's causal group-8 attention kernel at
// antirez/ds4@84cc882352757baf628a1776badf7cc54d584e28 (MIT).
// One threadgroup owns (batch, KV head, block). Its SIMDgroups map the GQA
// repeats, cooperatively unpack short K/V stages once, then reuse the decoded
// rows across all query heads and verify rows.
constexpr const char* TURBO4_MTP_VERIFY_PARTIAL_SOURCE = R"(
    constexpr uint StageRows = 16;

    uint lane = thread_index_in_simdgroup;
    uint tid = thread_index_in_threadgroup;
    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint repeat = thread_position_in_threadgroup.y;

    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;
    uint d0 = lane * dpt;
    uint hq_count = (uint)q_rot_shape[1];
    uint tk = (uint)k_packed_shape[2];
    uint packed_width = (uint)k_packed_shape[3];
    uint hkv_count = (uint)k_packed_shape[1];
    uint q_head = kv_head * (uint)RepeatCount + repeat;
    uint row_base = ((batch * hq_count + q_head) * (uint)QRows);

    float q[QRows][DimsPerThread];
    for (uint r = 0; r < (uint)QRows; r++) {
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            q[r][j] = d < dim
                ? q_rot[(row_base + r) * dim + d]
                : 0.0f;
        }
    }

    float out[QRows][DimsPerThread];
    float max_score[QRows];
    float sum_score[QRows];
    for (uint r = 0; r < (uint)QRows; r++) {
        max_score[r] = -INFINITY;
        sum_score[r] = 0.0f;
        for (uint j = 0; j < dpt; j++) {
            out[r][j] = 0.0f;
        }
    }

    threadgroup half staged_k[StageRows * Dim];
    threadgroup half staged_v[StageRows * Dim];

    float scale_log2e = scale[0] * 1.4426950408889634f;
    uint bh = batch * hkv_count + kv_head;
    uint block_tokens = tk > block
        ? ((tk - 1u - block) / (uint)Blocks + 1u)
        : 0u;
    uint threads = 32u * (uint)RepeatCount;
    for (uint base = 0; base < block_tokens; base += StageRows) {
        uint stage_rows = min(StageRows, block_tokens - base);
        uint stage_bytes = stage_rows * packed_width;
        for (uint off = tid; off < stage_bytes; off += threads) {
            uint rr = off / packed_width;
            uint packed_col = off - rr * packed_width;
            uint t = block + (base + rr) * (uint)Blocks;
            uint packed_base = (bh * tk + t) * packed_width;
            uint sidecar = bh * tk + t;
            float k_scale = (float)k_rescale[sidecar];
            float v_scale = (float)v_rescale[sidecar];
            uint k_byte = (uint)k_packed[packed_base + packed_col];
            uint v_byte = (uint)v_packed[packed_base + packed_col];
            uint d = packed_col * 2u;
            half2 k_pair = half2(
                codebook[k_byte & 0x0fu] * k_scale,
                codebook[(k_byte >> 4u) & 0x0fu] * k_scale);
            half2 v_pair = half2(
                codebook[v_byte & 0x0fu] * v_scale,
                codebook[(v_byte >> 4u) & 0x0fu] * v_scale);
            *((threadgroup half2 *)(staged_k + rr * dim + d)) = k_pair;
            *((threadgroup half2 *)(staged_v + rr * dim + d)) = v_pair;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0; rr < stage_rows; rr++) {
            uint t = block + (base + rr) * (uint)Blocks;
            threadgroup const half *k_row = staged_k + rr * dim;
            threadgroup const half *v_row = staged_v + rr * dim;
            float k[DimsPerThread];
            float v[DimsPerThread];
            for (uint j = 0; j < dpt; j++) {
                uint d = d0 + j;
                k[j] = (float)k_row[d];
                v[j] = (float)v_row[d];
            }

            // Bottom-right causal visibility: row r is positioned at
            // tk-QRows+r.
            int first_visible_row = (int)t - (int)tk + QRows;
            for (uint r = 0; r < (uint)QRows; r++) {
                if ((int)r < first_visible_row) {
                    continue;
                }
                float dot = 0.0f;
                for (uint j = 0; j < dpt; j++) {
                    dot += q[r][j] * k[j];
                }
                float score = simd_sum(dot) * scale_log2e;
                float next_max = fmax(max_score[r], score);
                float correction = exp2(max_score[r] - next_max);
                float probability = exp2(score - next_max);
                sum_score[r] = sum_score[r] * correction + probability;
                for (uint j = 0; j < dpt; j++) {
                    out[r][j] = out[r][j] * correction + probability * v[j];
                }
                max_score[r] = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < (uint)QRows; r++) {
        uint partial_row = (row_base + r) * (uint)Blocks + block;
        if (lane == 0u) {
            partial_sums[partial_row] = sum_score[r];
            partial_maxs[partial_row] = max_score[r];
        }
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            if (d < dim) {
                partial_acc[partial_row * dim + d] = out[r][j];
            }
        }
    }
)";

// Stable oMLX-style second pass. Thirty-two SIMDgroups reduce block-local
// states in parallel, then transpose their value accumulators through
// threadgroup memory for the final SIMD reductions.
constexpr const char* TURBO4_MTP_VERIFY_MERGE_SOURCE = R"(
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int ElementsPerThread = Dim / BD;

    float value[ElementsPerThread] = {};
    threadgroup float outputs[BN * BD];

    uint row = threadgroup_position_in_grid.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    const device float* sums = partial_sums + row * (uint)Blocks;
    const device float* maxs = partial_maxs + row * (uint)Blocks;
    float max_score = -INFINITY;
    float sum_score = 0.0f;

    for (uint block = simd_group; block < (uint)Blocks; block += (uint)BN) {
        float block_max = maxs[block];
        float block_sum = sums[block];
        float next_max = fmax(max_score, block_max);
        float correction = exp2(max_score - next_max);
        float block_correction = exp2(block_max - next_max);
        sum_score = sum_score * correction + block_sum * block_correction;
        for (uint i = 0; i < (uint)ElementsPerThread; i++) {
            value[i] = value[i] * correction
                + partial_acc[((row * (uint)Blocks + block) * (uint)Dim)
                              + lane * (uint)ElementsPerThread + i]
                    * block_correction;
        }
        max_score = next_max;
    }

    threadgroup float simd_maxs[BN];
    threadgroup float simd_sums[BN];
    if (lane == 0u) {
        simd_maxs[simd_group] = max_score;
        simd_sums[simd_group] = sum_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float group_max = simd_max(simd_maxs[lane]);
    float group_correction = exp2(simd_maxs[lane] - group_max);
    float total_sum = simd_sum(simd_sums[lane] * group_correction);
    float my_correction = exp2(max_score - group_max);

    for (uint i = 0; i < (uint)ElementsPerThread; i++) {
        outputs[lane * (uint)BD + simd_group] = value[i] * my_correction;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value[i] = simd_sum(outputs[simd_group * (uint)BD + lane]);
        value[i] = total_sum > 0.0f ? value[i] / total_sum : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        for (uint i = 0; i < (uint)ElementsPerThread; i++) {
            out[row * (uint)Dim + simd_group * (uint)ElementsPerThread + i] = value[i];
        }
    }
)";

struct PartialKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_mtp_verify_partial",
                {"q_rot", "k_packed", "k_rescale", "v_packed", "v_rescale", "codebook", "scale"},
                {"partial_acc", "partial_sums", "partial_maxs"},
                std::string(TURBO4_MTP_VERIFY_PARTIAL_SOURCE));
        });
        return *kernel;
    }
};

struct MergeKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_mtp_verify_merge",
                {"partial_acc", "partial_sums", "partial_maxs"},
                {"out"},
                std::string(TURBO4_MTP_VERIFY_MERGE_SOURCE));
        });
        return *kernel;
    }
};

PartialKernelHolder& partial_kernel_holder() {
    static PartialKernelHolder holder;
    return holder;
}

MergeKernelHolder& merge_kernel_holder() {
    static MergeKernelHolder holder;
    return holder;
}

int mtp_verify_blocks(int tokens) {
    if (tokens <= 8192) {
        return 64;
    }
    if (tokens <= 32768) {
        return 128;
    }
    if (tokens <= 65536) {
        return 160;
    }
    return 512;
}

void validate_inputs(
    const mlx::core::array& q_rot,
    const mlx::core::array& k_packed,
    const mlx::core::array& k_rescale,
    const mlx::core::array& v_packed,
    const mlx::core::array& v_rescale,
    const mlx::core::array& codebook,
    bool causal) {
    if (!mlx::core::metal::is_available()) {
        throw std::invalid_argument("turbo4_attention requires the Metal backend");
    }
    if (q_rot.ndim() != 4 || k_packed.ndim() != 4 || k_rescale.ndim() != 4
        || v_packed.ndim() != 4 || v_rescale.ndim() != 4
        || codebook.ndim() != 1) {
        throw std::invalid_argument("turbo4_attention input ranks are invalid");
    }
    if (q_rot.dtype() != mlx::core::float32
        || k_packed.dtype() != mlx::core::uint8
        || v_packed.dtype() != mlx::core::uint8
        || k_rescale.dtype() != mlx::core::float16
        || v_rescale.dtype() != mlx::core::float16
        || codebook.dtype() != mlx::core::float32) {
        throw std::invalid_argument("turbo4_attention input dtypes are invalid");
    }

    const auto& q_shape = q_rot.shape();
    const auto& k_shape = k_packed.shape();
    const auto& kr_shape = k_rescale.shape();
    const auto& v_shape = v_packed.shape();
    const auto& vr_shape = v_rescale.shape();
    const int batch = q_shape[0];
    const int hq = q_shape[1];
    const int tq = q_shape[2];
    const int dim = q_shape[3];
    const int hkv = k_shape[1];
    const int tk = k_shape[2];
    if (batch <= 0 || hq <= 0 || hkv <= 0 || tk <= MIN_MTP_VERIFY_TOKENS) {
        throw std::invalid_argument("turbo4_attention requires a non-empty long target cache");
    }
    if (!causal || tq < 2 || tq > 4 || tk < tq) {
        throw std::invalid_argument("turbo4_attention only supports causal MTP verify rows 2..=4");
    }
    if (hq % hkv != 0 || hq / hkv > 32) {
        throw std::invalid_argument("turbo4_attention has unsupported GQA repeat geometry");
    }
    if (dim < 32 || dim > 256 || dim % SIMD_WIDTH != 0
        || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument(
            "turbo4_attention head dimension must be a 32-multiple power-of-two <= 256");
    }
    if (k_shape[0] != batch || k_shape[3] != dim / 2
        || v_shape[0] != batch || v_shape[1] != hkv || v_shape[2] != tk
        || v_shape[3] != dim / 2
        || kr_shape[0] != batch || kr_shape[1] != hkv || kr_shape[2] != tk
        || kr_shape[3] != 1
        || vr_shape[0] != batch || vr_shape[1] != hkv || vr_shape[2] != tk
        || vr_shape[3] != 1) {
        throw std::invalid_argument("turbo4_attention packed and sidecar shapes disagree");
    }
    if (codebook.shape()[0] != 16) {
        throw std::invalid_argument("turbo4_attention codebook must have shape [16]");
    }
}

} // namespace

mlx::core::array turbo4_attention(
    const mlx::core::array& q_rot,
    const mlx::core::array& k_packed,
    const mlx::core::array& k_rescale,
    const mlx::core::array& v_packed,
    const mlx::core::array& v_rescale,
    const mlx::core::array& codebook,
    float scale,
    bool causal) {
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    validate_inputs(
        q_rot, k_packed, k_rescale, v_packed, v_rescale, codebook, causal);

    const auto& q_shape = q_rot.shape();
    const auto& k_shape = k_packed.shape();
    const int batch = q_shape[0];
    const int hq = q_shape[1];
    const int tq = q_shape[2];
    const int dim = q_shape[3];
    const int hkv = k_shape[1];
    const int tk = k_shape[2];
    const int repeats = hq / hkv;
    const int dims_per_thread = dim / SIMD_WIDTH;
    const int blocks = mtp_verify_blocks(tk);
    const int rows = batch * hq * tq;

    auto scale_array = mlx::core::full(Shape{1}, scale, mlx::core::float32);
    std::vector<std::pair<std::string, TemplateArg>> partial_template_args = {
        {"Dim", dim},
        {"DimsPerThread", dims_per_thread},
        {"RepeatCount", repeats},
        {"QRows", tq},
        {"Blocks", blocks},
        {"QType", q_rot.dtype()},
        {"PackedType", k_packed.dtype()},
        {"RescaleType", k_rescale.dtype()},
        {"CodebookType", codebook.dtype()},
    };
    auto partials = partial_kernel_holder().get()(
        {q_rot, k_packed, k_rescale, v_packed, v_rescale, codebook, scale_array},
        {Shape{rows * blocks, dim}, Shape{rows * blocks}, Shape{rows * blocks}},
        {mlx::core::float16, mlx::core::float32, mlx::core::float32},
        std::make_tuple(hkv * SIMD_WIDTH, batch * repeats, blocks),
        std::make_tuple(SIMD_WIDTH, repeats, 1),
        partial_template_args,
        std::nullopt,
        false,
        {});

    std::vector<std::pair<std::string, TemplateArg>> merge_template_args = {
        {"Dim", dim},
        {"Blocks", blocks},
        {"AccumulatorType", partials[0].dtype()},
        {"StatsType", partials[1].dtype()},
    };
    auto merged = merge_kernel_holder().get()(
        {partials[0], partials[1], partials[2]},
        {Shape{batch, hq, tq, dim}},
        {mlx::core::float32},
        std::make_tuple(rows * 1024, 1, 1),
        std::make_tuple(1024, 1, 1),
        merge_template_args,
        std::nullopt,
        false,
        {});
    return merged[0];
}

} // namespace mlxcel::turbo
