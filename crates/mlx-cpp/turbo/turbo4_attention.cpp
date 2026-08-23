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

#include <algorithm>
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
constexpr int THREADGROUP_BUDGET_BYTES = 28 * 1024;
constexpr int MAX_NUM_WARPS = 8;
constexpr int MAX_QUERY_ROWS = 2;
constexpr int MAX_ACCUMULATOR_FLOATS = 32;
constexpr int TURBO4_ATTENTION_CHUNK_SIZE = 2048;

// One threadgroup owns one (batch, query tile, key chunk, KV head, GQA head
// group). Lanes partition D; SIMD groups stripe the key chunk. Packed K/V are
// decoded once per token and reused across every query row/head owned by the
// threadgroup.
constexpr const char* TURBO4_ATTENTION_PARTIAL_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint yblk = threadgroup_position_in_grid.y;
    uint zblk = threadgroup_position_in_grid.z;

    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;
    uint d0 = lane * dpt;
    uint hq_count = (uint)q_rot_shape[1];
    uint tq = (uint)q_rot_shape[2];
    uint hkv_count = (uint)k_packed_shape[1];
    uint tk = (uint)k_packed_shape[2];
    uint packed_width = (uint)k_packed_shape[3];

    uint kv_head = yblk / (uint)QGroups;
    uint q_group = yblk - kv_head * (uint)QGroups;
    uint q_head_base = kv_head * (uint)NRep + q_group * (uint)QHeads;

    uint chunk = zblk % (uint)NumChunks;
    uint query_batch_tile = zblk / (uint)NumChunks;
    uint q_tile = query_batch_tile % (uint)QueryTiles;
    uint batch = query_batch_tile / (uint)QueryTiles;
    uint q_index_base = q_tile * (uint)QueryRows;
    uint key_begin = chunk * (uint)ChunkSize;
    uint key_end = min(key_begin + (uint)ChunkSize, tk);

    float q_reg[QHeads * QueryRows * DimsPerThread];
    for (uint g = 0; g < (uint)QHeads; g++) {
        uint h = q_head_base + g;
        for (uint r = 0; r < (uint)QueryRows; r++) {
            uint qi = q_index_base + r;
            uint item = g * (uint)QueryRows + r;
            for (uint j = 0; j < dpt; j++) {
                uint d = d0 + j;
                q_reg[item * dpt + j] =
                    (h < hq_count && qi < tq && d < dim)
                    ? q_rot[((batch * hq_count + h) * tq + qi) * dim + d]
                    : 0.0f;
            }
        }
    }

    float m[QHeads * QueryRows];
    float l[QHeads * QueryRows];
    float acc[QHeads * QueryRows * DimsPerThread];
    for (uint item = 0; item < (uint)(QHeads * QueryRows); item++) {
        m[item] = -INFINITY;
        l[item] = 0.0f;
        for (uint j = 0; j < dpt; j++) {
            acc[item * dpt + j] = 0.0f;
        }
    }

    float scale_log2e = scale[0] * 1.4426950408889634f;
    for (uint t = key_begin + sg; t < key_end; t += (uint)NumWarps) {
        uint packed_base = ((batch * hkv_count + kv_head) * tk + t) * packed_width;
        uint sidecar_index = (batch * hkv_count + kv_head) * tk + t;
        float k_scale = (float)k_rescale[sidecar_index];
        float v_scale = (float)v_rescale[sidecar_index];
        float k_reg[DimsPerThread];
        float v_reg[DimsPerThread];
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            if (d < dim) {
                uint k_byte = (uint)k_packed[packed_base + (d >> 1)];
                uint v_byte = (uint)v_packed[packed_base + (d >> 1)];
                uint shift = (d & 1u) * 4u;
                uint k_index = (k_byte >> shift) & 0x0fu;
                uint v_index = (v_byte >> shift) & 0x0fu;
                k_reg[j] = codebook[k_index] * k_scale;
                v_reg[j] = codebook[v_index] * v_scale;
            } else {
                k_reg[j] = 0.0f;
                v_reg[j] = 0.0f;
            }
        }

        for (uint g = 0; g < (uint)QHeads; g++) {
            uint h = q_head_base + g;
            for (uint r = 0; r < (uint)QueryRows; r++) {
                uint qi = q_index_base + r;
                uint item = g * (uint)QueryRows + r;
                if (h >= hq_count || qi >= tq) {
                    continue;
                }
                if ((uint)Causal != 0u && t > tk - tq + qi) {
                    continue;
                }
                float partial = 0.0f;
                for (uint j = 0; j < dpt; j++) {
                    partial += q_reg[item * dpt + j] * k_reg[j];
                }
                float score = simd_sum(partial) * scale_log2e;
                float m_new = fmax(m[item], score);
                float corr = exp2(m[item] - m_new);
                float probability = exp2(score - m_new);
                l[item] = l[item] * corr + probability;
                for (uint j = 0; j < dpt; j++) {
                    acc[item * dpt + j] =
                        acc[item * dpt + j] * corr + probability * v_reg[j];
                }
                m[item] = m_new;
            }
        }
    }

    threadgroup float tg_m[NumWarps * QHeads * QueryRows];
    threadgroup float tg_l[NumWarps * QHeads * QueryRows];
    threadgroup float tg_acc[NumWarps * QHeads * QueryRows * Dim];
    for (uint item = 0; item < (uint)(QHeads * QueryRows); item++) {
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            if (d < dim) {
                tg_acc[(sg * (uint)(QHeads * QueryRows) + item) * dim + d] =
                    acc[item * dpt + j];
            }
        }
        if (lane == 0u) {
            uint state = sg * (uint)(QHeads * QueryRows) + item;
            tg_m[state] = m[item];
            tg_l[state] = l[item];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        for (uint g = 0; g < (uint)QHeads; g++) {
            uint h = q_head_base + g;
            for (uint r = 0; r < (uint)QueryRows; r++) {
                uint qi = q_index_base + r;
                uint item = g * (uint)QueryRows + r;
                if (h >= hq_count || qi >= tq) {
                    continue;
                }
                float merged_m = -INFINITY;
                for (uint s = 0; s < (uint)NumWarps; s++) {
                    merged_m = fmax(
                        merged_m,
                        tg_m[s * (uint)(QHeads * QueryRows) + item]);
                }
                float merged_l = 0.0f;
                if (merged_m > -INFINITY) {
                    for (uint s = 0; s < (uint)NumWarps; s++) {
                        uint state = s * (uint)(QHeads * QueryRows) + item;
                        merged_l += tg_l[state] * exp2(tg_m[state] - merged_m);
                    }
                }
                uint partial_row =
                    (((batch * (uint)QueryTiles + q_tile) * (uint)NumChunks + chunk)
                     * hq_count + h) * (uint)QueryRows + r;
                for (uint j = 0; j < dpt; j++) {
                    uint d = d0 + j;
                    if (d < dim) {
                        float merged_acc = 0.0f;
                        if (merged_l > 0.0f) {
                            for (uint s = 0; s < (uint)NumWarps; s++) {
                                uint state = s * (uint)(QHeads * QueryRows) + item;
                                merged_acc += tg_acc[state * dim + d]
                                    * exp2(tg_m[state] - merged_m);
                            }
                        }
                        partial_v[partial_row * dim + d] =
                            merged_l > 0.0f ? merged_acc / merged_l : 0.0f;
                    }
                }
                if (lane == 0u) {
                    partial_lse[partial_row] = merged_l > 0.0f
                        ? merged_m + log2(merged_l)
                        : -INFINITY;
                }
            }
        }
    }
)";

// One thread owns one output dimension and combines all key-chunk states for
// that (batch, query row, query head) with the same base-2 flash rescaling.
constexpr const char* TURBO4_ATTENTION_MERGE_SOURCE = R"(
    uint d = thread_position_in_threadgroup.x;
    uint h = threadgroup_position_in_grid.y;
    uint output_row = threadgroup_position_in_grid.z;

    uint dim = (uint)Dim;
    uint heads = (uint)Heads;
    uint tq = (uint)Tq;
    if (d >= dim || h >= heads) {
        return;
    }
    uint batch = output_row / tq;
    uint qi = output_row - batch * tq;
    uint q_tile = qi / (uint)QueryRows;
    uint q_row = qi - q_tile * (uint)QueryRows;

    float merged_m = -INFINITY;
    float merged_l = 0.0f;
    float merged_acc = 0.0f;
    for (uint chunk = 0; chunk < (uint)NumChunks; chunk++) {
        uint partial_row =
            (((batch * (uint)QueryTiles + q_tile) * (uint)NumChunks + chunk)
             * heads + h) * (uint)QueryRows + q_row;
        float state_lse = partial_lse[partial_row];
        if (!(state_lse > -INFINITY)) {
            continue;
        }
        float m_new = fmax(merged_m, state_lse);
        float corr = exp2(merged_m - m_new);
        float weight = exp2(state_lse - m_new);
        merged_l = merged_l * corr + weight;
        merged_acc = merged_acc * corr
            + weight * partial_v[partial_row * dim + d];
        merged_m = m_new;
    }
    out[((batch * heads + h) * tq + qi) * dim + d] =
        merged_l > 0.0f ? merged_acc / merged_l : 0.0f;
)";

const std::vector<std::string>& partial_input_names() {
    static const std::vector<std::string> names = {
        "q_rot",
        "k_packed",
        "k_rescale",
        "v_packed",
        "v_rescale",
        "codebook",
        "scale",
    };
    return names;
}

struct PartialKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_attention_partial",
                partial_input_names(),
                {"partial_v", "partial_lse"},
                std::string(TURBO4_ATTENTION_PARTIAL_SOURCE));
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
                "mlxcel_turbo4_attention_merge",
                {"partial_v", "partial_lse"},
                {"out"},
                std::string(TURBO4_ATTENTION_MERGE_SOURCE));
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

int query_heads_per_group(int dim, int n_rep, int query_rows) {
    const int dims_per_thread = (dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
    int cap = MAX_ACCUMULATOR_FLOATS / (query_rows * dims_per_thread);
    cap = std::min(cap, n_rep);
    for (int candidate = cap; candidate >= 1; --candidate) {
        if (n_rep % candidate == 0) {
            return candidate;
        }
    }
    return 1;
}

int num_warps_for(int dim, int query_heads, int query_rows) {
    const int bytes_per_warp = query_heads * query_rows * (dim + 2) * 4;
    int budget = THREADGROUP_BUDGET_BYTES / bytes_per_warp;
    budget = std::min(budget, MAX_NUM_WARPS);
    int warps = 1;
    while (warps * 2 <= budget) {
        warps *= 2;
    }
    return warps;
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
    if (batch <= 0 || hq <= 0 || hkv <= 0 || tq <= 0 || tk <= 0) {
        throw std::invalid_argument("turbo4_attention inputs must be non-empty");
    }
    if (hq % hkv != 0) {
        throw std::invalid_argument("turbo4_attention requires Hq divisible by Hkv");
    }
    if (dim <= 0 || dim > 256 || (dim & 1) != 0 || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument(
            "turbo4_attention head dimension must be even, power-of-two, and <= 256");
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
    if (causal && tk < tq) {
        throw std::invalid_argument("turbo4_attention causal calls require Tk >= Tq");
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
    using mlx::core::Dtype;
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
    const int n_rep = hq / hkv;
    const int query_rows = std::min(tq, MAX_QUERY_ROWS);
    const int query_tiles = (tq + query_rows - 1) / query_rows;
    const int num_chunks =
        (tk + TURBO4_ATTENTION_CHUNK_SIZE - 1) / TURBO4_ATTENTION_CHUNK_SIZE;
    const int dims_per_thread = (dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
    const int query_heads = query_heads_per_group(dim, n_rep, query_rows);
    const int query_groups = n_rep / query_heads;
    const int num_warps = num_warps_for(dim, query_heads, query_rows);

    auto scale_array = mlx::core::full(Shape{1}, scale, mlx::core::float32);
    std::vector<mlx::core::array> partial_inputs = {
        q_rot,
        k_packed,
        k_rescale,
        v_packed,
        v_rescale,
        codebook,
        scale_array,
    };
    std::vector<std::pair<std::string, TemplateArg>> partial_template_args = {
        {"Dim", dim},
        {"NRep", n_rep},
        {"QHeads", query_heads},
        {"QGroups", query_groups},
        {"QueryRows", query_rows},
        {"QueryTiles", query_tiles},
        {"NumChunks", num_chunks},
        {"ChunkSize", TURBO4_ATTENTION_CHUNK_SIZE},
        {"DimsPerThread", dims_per_thread},
        {"NumWarps", num_warps},
        {"Causal", causal ? 1 : 0},
        {"QType", q_rot.dtype()},
        {"PackedType", k_packed.dtype()},
        {"RescaleType", k_rescale.dtype()},
        {"CodebookType", codebook.dtype()},
    };
    std::vector<Shape> partial_shapes = {
        Shape{batch, query_tiles, num_chunks, hq, query_rows, dim},
        Shape{batch, query_tiles, num_chunks, hq, query_rows},
    };
    std::vector<Dtype> partial_dtypes = {
        mlx::core::float32,
        mlx::core::float32,
    };
    auto partials = partial_kernel_holder().get()(
        partial_inputs,
        partial_shapes,
        partial_dtypes,
        std::make_tuple(
            SIMD_WIDTH,
            num_warps * hkv * query_groups,
            batch * query_tiles * num_chunks),
        std::make_tuple(SIMD_WIDTH, num_warps, 1),
        partial_template_args,
        std::nullopt,
        false,
        {});

    std::vector<std::pair<std::string, TemplateArg>> merge_template_args = {
        {"Dim", dim},
        {"Heads", hq},
        {"Tq", tq},
        {"QueryRows", query_rows},
        {"QueryTiles", query_tiles},
        {"NumChunks", num_chunks},
        {"PartialType", partials[0].dtype()},
        {"LseType", partials[1].dtype()},
    };
    auto merged = merge_kernel_holder().get()(
        {partials[0], partials[1]},
        {Shape{batch, hq, tq, dim}},
        {mlx::core::float32},
        std::make_tuple(dim, hq, batch * tq),
        std::make_tuple(dim, 1, 1),
        merge_template_args,
        std::nullopt,
        false,
        {});
    return merged[0];
}

} // namespace mlxcel::turbo
