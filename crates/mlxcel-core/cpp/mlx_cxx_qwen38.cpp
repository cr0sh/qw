// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");

// Fixed-artifact Metal kernels for the pinned Qwen3.8 target.

#include "mlx_cxx_internal.h"

#include <limits>

namespace mlx_cxx {
namespace {

static const char* QWEN38_AFFINE_M234_METAL_HEADER = R"(
template <int bits, int wsize = 8>
inline constexpr short qwen38_pack_factor() {
    return bits == 5 ? 8 : wsize / bits;
}

template <int bits, int wsize = 8>
inline constexpr short qwen38_bytes_per_pack() {
    return bits == 5 ? 5 : wsize / 8;
}

template <int values_per_thread, int bits>
inline float qwen38_load_vector(
    const device float* x,
    thread float* x_thread
) {
    float sum = 0.0f;
    if constexpr (bits == 4) {
        for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if constexpr (bits == 5) {
        for (int i = 0; i < values_per_thread; i += 8) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 32.0f;
            x_thread[i + 2] = x[i + 2] / 4.0f;
            x_thread[i + 3] = x[i + 3] / 128.0f;
            x_thread[i + 4] = x[i + 4] / 16.0f;
            x_thread[i + 5] = x[i + 5] / 2.0f;
            x_thread[i + 6] = x[i + 6] / 64.0f;
            x_thread[i + 7] = x[i + 7] / 8.0f;
        }
    } else {
        for (int i = 0; i < values_per_thread; ++i) {
            sum += x[i];
            x_thread[i] = x[i];
        }
    }
    return sum;
}

template <int values_per_thread, int bits>
inline float qwen38_qdot(
    const thread uchar* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum
) {
    float accum = 0.0f;
    if constexpr (bits == 4) {
        for (int i = 0; i < values_per_thread / 4; ++i) {
            ushort packed = (ushort)w[2 * i]
                | ((ushort)w[2 * i + 1] << 8u);
            accum +=
                (x_thread[4 * i] * (packed & 0x000f)
                 + x_thread[4 * i + 1] * (packed & 0x00f0)
                 + x_thread[4 * i + 2] * (packed & 0x0f00)
                 + x_thread[4 * i + 3] * (packed & 0xf000));
        }
    } else if constexpr (bits == 5) {
        for (int i = 0; i < values_per_thread / 8; ++i) {
            const thread float* xv = x_thread + 8 * i;
            const thread uchar* wp = w + 5 * i;
            accum += (wp[0] & 0x1f) * xv[0];
            accum += (wp[0] & 0xe0) * xv[1];
            accum += (wp[1] & 0x03) * (xv[1] * 256.0f);
            accum += (wp[1] & 0x7c) * xv[2];
            accum += (wp[1] & 0x80) * xv[3];
            accum += (wp[2] & 0x0f) * (xv[3] * 256.0f);
            accum += (wp[2] & 0xf0) * xv[4];
            accum += (wp[3] & 0x01) * (xv[4] * 256.0f);
            accum += (wp[3] & 0x3e) * xv[5];
            accum += (wp[3] & 0xc0) * xv[6];
            accum += (wp[4] & 0x07) * (xv[6] * 256.0f);
            accum += (wp[4] & 0xf8) * xv[7];
        }
    } else {
        for (int i = 0; i < values_per_thread; ++i) {
            accum += x_thread[i] * w[i];
        }
    }
    return scale * accum + sum * bias;
}

template <int rows>
inline void qwen38_mixed_q5_update4(
    const device half* x,
    int row_stride,
    int column,
    half4 weights,
    thread float* accum
) {
    for (int row = 0; row < rows; ++row) {
        half4 activation = *reinterpret_cast<const device half4*>(
            x + row * row_stride + column);
        half4 product = activation * weights;
        accum[row] += static_cast<float>(product[0]);
        accum[row] += static_cast<float>(product[1]);
        accum[row] += static_cast<float>(product[2]);
        accum[row] += static_cast<float>(product[3]);
    }
}

template <int rows>
inline void qwen38_mixed_q5_pack(
    const device half* x,
    int row_stride,
    const device uchar* packed,
    half scale,
    half bias,
    thread float* accum
) {
    uchar b0 = packed[0];
    uchar b1 = packed[1];
    uchar b2 = packed[2];
    uchar b3 = packed[3];
    uchar b4 = packed[4];

    ushort c0 = b0 & 0x1fu;
    ushort c1 = (b0 >> 5u) | ((b1 & 0x03u) << 3u);
    ushort c2 = (b1 >> 2u) & 0x1fu;
    ushort c3 = (b1 >> 7u) | ((b2 & 0x0fu) << 1u);
    qwen38_mixed_q5_update4<rows>(
        x,
        row_stride,
        0,
        half4(
            scale * static_cast<half>(c0) + bias,
            scale * static_cast<half>(c1) + bias,
            scale * static_cast<half>(c2) + bias,
            scale * static_cast<half>(c3) + bias),
        accum);

    c0 = (b2 >> 4u) | ((b3 & 0x01u) << 4u);
    c1 = (b3 >> 1u) & 0x1fu;
    c2 = (b3 >> 6u) | ((b4 & 0x07u) << 2u);
    c3 = b4 >> 3u;
    qwen38_mixed_q5_update4<rows>(
        x,
        row_stride,
        4,
        half4(
            scale * static_cast<half>(c0) + bias,
            scale * static_cast<half>(c1) + bias,
            scale * static_cast<half>(c2) + bias,
            scale * static_cast<half>(c3) + bias),
        accum);
}

template <int rows>
inline void qwen38_mixed_q5_update4_cached(
    const thread half4* x,
    int row_stride,
    int vector,
    half4 weights,
    thread float* accum
) {
    for (int row = 0; row < rows; ++row) {
        half4 product = x[row * row_stride + vector] * weights;
        accum[row] += static_cast<float>(product[0]);
        accum[row] += static_cast<float>(product[1]);
        accum[row] += static_cast<float>(product[2]);
        accum[row] += static_cast<float>(product[3]);
    }
}

template <int rows>
inline void qwen38_mixed_q5_pack_cached(
    const thread half4* x,
    int row_stride,
    int vector,
    const device uchar* packed,
    half scale,
    half bias,
    thread float* accum
) {
    uchar b0 = packed[0];
    uchar b1 = packed[1];
    uchar b2 = packed[2];
    uchar b3 = packed[3];
    uchar b4 = packed[4];

    ushort c0 = b0 & 0x1fu;
    ushort c1 = (b0 >> 5u) | ((b1 & 0x03u) << 3u);
    ushort c2 = (b1 >> 2u) & 0x1fu;
    ushort c3 = (b1 >> 7u) | ((b2 & 0x0fu) << 1u);
    qwen38_mixed_q5_update4_cached<rows>(
        x,
        row_stride,
        vector,
        half4(
            scale * static_cast<half>(c0) + bias,
            scale * static_cast<half>(c1) + bias,
            scale * static_cast<half>(c2) + bias,
            scale * static_cast<half>(c3) + bias),
        accum);

    c0 = (b2 >> 4u) | ((b3 & 0x01u) << 4u);
    c1 = (b3 >> 1u) & 0x1fu;
    c2 = (b3 >> 6u) | ((b4 & 0x07u) << 2u);
    c3 = b4 >> 3u;
    qwen38_mixed_q5_update4_cached<rows>(
        x,
        row_stride,
        vector + 1,
        half4(
            scale * static_cast<half>(c0) + bias,
            scale * static_cast<half>(c1) + bias,
            scale * static_cast<half>(c2) + bias,
            scale * static_cast<half>(c3) + bias),
        accum);
}

inline float qwen38_affine_sigmoid(float x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
)";

static const char* QWEN38_AFFINE_M234_METAL_SOURCE = R"(
    constexpr int packs_per_thread = 2;
    constexpr int pack_factor = qwen38_pack_factor<Bits, 32>();
    constexpr int bytes_per_pack = qwen38_bytes_per_pack<Bits, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = 32 / values_per_thread;
    constexpr int cached_weight_bytes = packs_per_thread * bytes_per_pack;

    uint simdgroup = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int output_base = (int)threadgroup_position_in_grid.y * 8
        + (int)simdgroup * 4;
    int width = (int)x_shape[1];
    int output_rows = (int)weight_shape[0];
    int packed_row_bytes = width * Bits / 8;
    int groups_per_row = width / 32;

    float x_values[MRows][values_per_thread];
    float results[MRows][4];
    for (int input_row = 0; input_row < MRows; ++input_row) {
        for (int output = 0; output < 4; ++output) {
            results[input_row][output] = 0.0f;
        }
    }

    for (int k = 0; k < width; k += block_size) {
        float sums[MRows];
        for (int input_row = 0; input_row < MRows; ++input_row) {
            const device float* xv = x + input_row * width + k
                + (int)lane * values_per_thread;
            sums[input_row] =
                qwen38_load_vector<values_per_thread, Bits>(
                    xv, x_values[input_row]);
        }

        for (int output = 0; output < 4; ++output) {
            int output_row = output_base + output;
            if (output_row >= output_rows) {
                continue;
            }
            int byte_offset = output_row * packed_row_bytes + k * Bits / 8
                + (int)lane * cached_weight_bytes;
            const device uchar* source =
                reinterpret_cast<const device uchar*>(weight) + byte_offset;
            uchar cached[cached_weight_bytes];
            for (int byte = 0; byte < cached_weight_bytes; ++byte) {
                cached[byte] = source[byte];
            }
            int group_offset = output_row * groups_per_row + k / 32
                + (int)lane / scale_step_per_thread;
            float scale = scales[group_offset];
            float bias = biases[group_offset];
            for (int input_row = 0; input_row < MRows; ++input_row) {
                results[input_row][output] +=
                    qwen38_qdot<values_per_thread, Bits>(
                        cached,
                        x_values[input_row],
                        scale,
                        bias,
                        sums[input_row]);
            }
        }
    }

    for (int input_row = 0; input_row < MRows; ++input_row) {
        for (int output = 0; output < 4; ++output) {
            float total = simd_sum(results[input_row][output]);
            int output_row = output_base + output;
            if (lane == 0u && output_row < output_rows) {
                out[input_row * output_rows + output_row] = total;
            }
        }
    }
)";

static const char* QWEN38_AFFINE_MIXED_Q5_METAL_SOURCE = R"(
    constexpr int packs_per_thread = 2;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int outputs_per_simd = MRows == 3 ? 1 : 4;
    constexpr int outputs_per_threadgroup = outputs_per_simd * 2;

    uint simdgroup = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int output_base = (int)threadgroup_position_in_grid.y * outputs_per_threadgroup
        + (int)simdgroup * outputs_per_simd;
    int width = (int)x_shape[1];
    int output_rows = (int)weight_shape[0];
    int packed_row_bytes = width * 5 / 8;
    int groups_per_row = width / 32;

    float results[MRows][outputs_per_simd] = {};
    for (int k = 0; k < width; k += block_size) {
        for (int output = 0; output < outputs_per_simd; ++output) {
            int output_row = output_base + output;
            if (output_row >= output_rows) {
                continue;
            }
            int byte_offset = output_row * packed_row_bytes + k * 5 / 8
                + (int)lane * 10;
            const device uchar* source =
                reinterpret_cast<const device uchar*>(weight) + byte_offset;
            int group_offset = output_row * groups_per_row + k / 32
                + (int)lane / 2;
            half scale = scales[group_offset];
            half bias = biases[group_offset];
            float accum[MRows] = {};
            const device half* activation = x + k
                + (int)lane * values_per_thread;
            qwen38_mixed_q5_pack<MRows>(
                activation, width, source, scale, bias, accum);
            qwen38_mixed_q5_pack<MRows>(
                activation + 8, width, source + 5, scale, bias, accum);
            for (int input_row = 0; input_row < MRows; ++input_row) {
                results[input_row][output] += accum[input_row];
            }
        }
    }

    for (int input_row = 0; input_row < MRows; ++input_row) {
        for (int output = 0; output < outputs_per_simd; ++output) {
            float total = simd_sum(results[input_row][output]);
            int output_row = output_base + output;
            if (lane == 0u && output_row < output_rows) {
                out[input_row * output_rows + output_row] = total;
            }
        }
    }
)";

static const char* QWEN38_AFFINE_MLP_GATE_UP_METAL_SOURCE = R"(
    constexpr int packs_per_thread = 2;
    constexpr int pack_factor = qwen38_pack_factor<Bits, 32>();
    constexpr int bytes_per_pack = qwen38_bytes_per_pack<Bits, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = 32 / values_per_thread;
    constexpr int cached_weight_bytes = packs_per_thread * bytes_per_pack;

    uint simdgroup = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int output_base = (int)threadgroup_position_in_grid.y * 8
        + (int)simdgroup * 4;
    int width = (int)x_shape[1];
    int output_rows = (int)gate_weight_shape[0];
    int packed_row_bytes = width * Bits / 8;
    int groups_per_row = width / 32;

    float x_values[MRows][values_per_thread];
    float gate_results[MRows][4];
    float up_results[MRows][4];
    for (int input_row = 0; input_row < MRows; ++input_row) {
        for (int output = 0; output < 4; ++output) {
            gate_results[input_row][output] = 0.0f;
            up_results[input_row][output] = 0.0f;
        }
    }

    for (int k = 0; k < width; k += block_size) {
        float sums[MRows];
        for (int input_row = 0; input_row < MRows; ++input_row) {
            const device float* xv = x + input_row * width + k
                + (int)lane * values_per_thread;
            sums[input_row] =
                qwen38_load_vector<values_per_thread, Bits>(
                    xv, x_values[input_row]);
        }

        for (int output = 0; output < 4; ++output) {
            int output_row = output_base + output;
            if (output_row >= output_rows) {
                continue;
            }
            int byte_offset = output_row * packed_row_bytes + k * Bits / 8
                + (int)lane * cached_weight_bytes;
            int group_offset = output_row * groups_per_row + k / 32
                + (int)lane / scale_step_per_thread;

            const device uchar* gate_source =
                reinterpret_cast<const device uchar*>(gate_weight) + byte_offset;
            uchar gate_cached[cached_weight_bytes];
            for (int byte = 0; byte < cached_weight_bytes; ++byte) {
                gate_cached[byte] = gate_source[byte];
            }
            float gate_scale = gate_scales[group_offset];
            float gate_bias = gate_biases[group_offset];
            for (int input_row = 0; input_row < MRows; ++input_row) {
                gate_results[input_row][output] +=
                    qwen38_qdot<values_per_thread, Bits>(
                        gate_cached,
                        x_values[input_row],
                        gate_scale,
                        gate_bias,
                        sums[input_row]);
            }

            const device uchar* up_source =
                reinterpret_cast<const device uchar*>(up_weight) + byte_offset;
            uchar up_cached[cached_weight_bytes];
            for (int byte = 0; byte < cached_weight_bytes; ++byte) {
                up_cached[byte] = up_source[byte];
            }
            float up_scale = up_scales[group_offset];
            float up_bias = up_biases[group_offset];
            for (int input_row = 0; input_row < MRows; ++input_row) {
                up_results[input_row][output] +=
                    qwen38_qdot<values_per_thread, Bits>(
                        up_cached,
                        x_values[input_row],
                        up_scale,
                        up_bias,
                        sums[input_row]);
            }
        }
    }

    for (int input_row = 0; input_row < MRows; ++input_row) {
        for (int output = 0; output < 4; ++output) {
            float gate = simd_sum(gate_results[input_row][output]);
            float up = simd_sum(up_results[input_row][output]);
            int output_row = output_base + output;
            if (lane == 0u && output_row < output_rows) {
                out[input_row * output_rows + output_row] =
                    (gate * qwen38_affine_sigmoid(gate)) * up;
            }
        }
    }
)";

static const char* QWEN38_AFFINE_MLP_GATE_UP_MIXED_Q5_METAL_SOURCE = R"(
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int outputs_per_simd = MRows == 3 ? 1 : 4;
    constexpr int outputs_per_threadgroup = outputs_per_simd * 2;
    constexpr int activation_vectors = values_per_thread / 4;

    uint simdgroup = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int output_base = (int)threadgroup_position_in_grid.y * outputs_per_threadgroup
        + (int)simdgroup * outputs_per_simd;
    int width = (int)x_shape[1];
    int output_rows = (int)gate_weight_shape[0];
    int packed_row_bytes = width * 5 / 8;
    int groups_per_row = width / 32;

    float gate_results[MRows][outputs_per_simd] = {};
    float up_results[MRows][outputs_per_simd] = {};
    for (int k = 0; k < width; k += block_size) {
        half4 activation[MRows][activation_vectors];
        for (int input_row = 0; input_row < MRows; ++input_row) {
            const device half* source = x + input_row * width + k
                + (int)lane * values_per_thread;
            for (int vector = 0; vector < activation_vectors; ++vector) {
                activation[input_row][vector] =
                    *reinterpret_cast<const device half4*>(source + vector * 4);
            }
        }

        for (int output = 0; output < outputs_per_simd; ++output) {
            int output_row = output_base + output;
            if (output_row >= output_rows) {
                continue;
            }
            int byte_offset = output_row * packed_row_bytes + k * 5 / 8
                + (int)lane * 10;
            int group_offset = output_row * groups_per_row + k / 32
                + (int)lane / 2;

            const device uchar* gate_source =
                reinterpret_cast<const device uchar*>(gate_weight) + byte_offset;
            half gate_scale = gate_scales[group_offset];
            half gate_bias = gate_biases[group_offset];
            float gate_accum[MRows] = {};
            qwen38_mixed_q5_pack_cached<MRows>(
                &activation[0][0],
                activation_vectors,
                0,
                gate_source,
                gate_scale,
                gate_bias,
                gate_accum);
            qwen38_mixed_q5_pack_cached<MRows>(
                &activation[0][0],
                activation_vectors,
                2,
                gate_source + 5,
                gate_scale,
                gate_bias,
                gate_accum);

            const device uchar* up_source =
                reinterpret_cast<const device uchar*>(up_weight) + byte_offset;
            half up_scale = up_scales[group_offset];
            half up_bias = up_biases[group_offset];
            float up_accum[MRows] = {};
            qwen38_mixed_q5_pack_cached<MRows>(
                &activation[0][0],
                activation_vectors,
                0,
                up_source,
                up_scale,
                up_bias,
                up_accum);
            qwen38_mixed_q5_pack_cached<MRows>(
                &activation[0][0],
                activation_vectors,
                2,
                up_source + 5,
                up_scale,
                up_bias,
                up_accum);

            for (int input_row = 0; input_row < MRows; ++input_row) {
                gate_results[input_row][output] += gate_accum[input_row];
                up_results[input_row][output] += up_accum[input_row];
            }
        }
    }

    for (int input_row = 0; input_row < MRows; ++input_row) {
        for (int output = 0; output < outputs_per_simd; ++output) {
            float gate = simd_sum(gate_results[input_row][output]);
            float up = simd_sum(up_results[input_row][output]);
            int output_row = output_base + output;
            if (lane == 0u && output_row < output_rows) {
                out[input_row * output_rows + output_row] =
                    (gate * qwen38_affine_sigmoid(gate)) * up;
            }
        }
    }
)";

// The packed prework topology is adapted from:
// - Yukon, Qwen35.swift:
//   https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/commit/e3b4531d947cbcab06a2d25929900c022dd9ad1b
//   MIT; Copyright (c) 2026 Layr Labs Inc.; co-author mega-dmitriy
//   <225149244+mega-dmitriy@users.noreply.github.com>.
// - oMLX, qwen35_gdn_prework.py:
//   https://github.com/jundot/omlx/commit/293d697c2d5a773225891636af75a2b8dd2b8d3f
//   Apache-2.0; Copyright 2026 Jun Kim (jundot).
//
// This fixed-artifact FP32 variant deliberately stops before the recurrent
// coefficient/beta update. It also keeps the reference chain's materialization
// boundaries: depthwise conv -> SiLU -> independent Q/K RMSNorm -> scale.
static const char* QWEN38_GDN_PREWORK_METAL_SOURCE = R"(
    constexpr uint NKeep = 3;
    constexpr uint ConvDim = 10240;
    constexpr uint Hk = 16;
    constexpr uint Hv = 48;
    constexpr uint Dk = 128;
    constexpr uint Dv = 128;

    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint logical_head = threadgroup_position_in_grid.z;
    const bool is_q = logical_head < Hk;
    const bool is_k = logical_head >= Hk && logical_head < 2 * Hk;
    const uint head = is_q
        ? logical_head
        : (is_k ? logical_head - Hk : logical_head - 2 * Hk);
    const uint channel_base = is_q
        ? head * Dk
        : (is_k ? Hk * Dk + head * Dk : 2 * Hk * Dk + head * Dv);

    // The cache tail is the last three rows of [old_state | qkv]. The launch
    // keeps at least three grid rows even for M1, so every tail row is written
    // without a concatenate/slice/contiguous graph.
    if (row < NKeep) {
        const uint input_row = MRows + row;
        for (uint i = 0; i < 4; ++i) {
            const uint channel = channel_base + lane * 4 + i;
            const float value = input_row < NKeep
                ? conv_state[
                    ulong(input_row) * ulong(conv_state_strides[1])
                    + ulong(channel) * ulong(conv_state_strides[2])]
                : qkv[
                    ulong(input_row - NKeep) * ulong(qkv_strides[1])
                    + ulong(channel) * ulong(qkv_strides[2])];
            conv_tail[(row * ConvDim) + channel] = value;
        }
    }
    if (row >= MRows) {
        return;
    }

    float activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            const uint input_row = row + tap;
            const float value = input_row < NKeep
                ? conv_state[
                    ulong(input_row) * ulong(conv_state_strides[1])
                    + ulong(channel) * ulong(conv_state_strides[2])]
                : qkv[
                    ulong(input_row - NKeep) * ulong(qkv_strides[1])
                    + ulong(channel) * ulong(qkv_strides[2])];
            const float weight = conv_weight[
                ulong(channel) * ulong(conv_weight_strides[0])
                + ulong(tap) * ulong(conv_weight_strides[1])];
            acc += value * weight;
        }

        // These volatile assignments preserve the two FP32 producer
        // boundaries that were physical buffers in the reference graph.
        volatile float conv_value = acc;
        const float sigmoid_base =
            1.0f / (1.0f + metal::exp(metal::abs(conv_value)));
        const float sigmoid = conv_value < 0.0f
            ? sigmoid_base
            : 1.0f - sigmoid_base;
        volatile float activated_value = conv_value * sigmoid;
        activated[i] = activated_value;
        sumsq += activated[i] * activated[i];
    }

    if (is_q || is_k) {
        sumsq = simd_sum(sumsq);
        const float inv_rms = metal::precise::rsqrt(sumsq / Dk + eps);
        const float scale = is_q ? q_scale : k_scale;
        const uint output_base = (row * Hk + head) * Dk + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            // fast::rms_norm and the following scalar multiply are separate
            // reference kernels. Keep their intermediate FP32 rounding here.
            volatile float normalized = activated[i] * inv_rms;
            const float value = normalized * scale;
            if (is_q) {
                q_out[output_base + i] = value;
            } else {
                k_out[output_base + i] = value;
            }
        }
    } else {
        const uint output_base = (row * Hv + head) * Dv + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[output_base + i] = activated[i];
        }
    }
)";

struct Qwen38KernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> affine_m234;
    std::optional<mlx::core::fast::CustomKernelFunction> affine_mixed_q5;
    std::optional<mlx::core::fast::CustomKernelFunction> affine_mlp_gate_up;
    std::optional<mlx::core::fast::CustomKernelFunction> affine_mlp_gate_up_mixed_q5;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_prework;
    std::once_flag initialize_once;

    void initialize() {
        std::call_once(initialize_once, [this] {
            affine_m234 = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_m234_v1",
                {"x", "weight", "scales", "biases"},
                {"out"},
                QWEN38_AFFINE_M234_METAL_SOURCE,
                QWEN38_AFFINE_M234_METAL_HEADER,
                false);
            affine_mixed_q5 = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mixed_q5_v2",
                {"x", "weight", "scales", "biases"},
                {"out"},
                QWEN38_AFFINE_MIXED_Q5_METAL_SOURCE,
                QWEN38_AFFINE_M234_METAL_HEADER,
                false);
            affine_mlp_gate_up = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mlp_gate_up_v2",
                {
                    "x",
                    "gate_weight",
                    "gate_scales",
                    "gate_biases",
                    "up_weight",
                    "up_scales",
                    "up_biases",
                },
                {"out"},
                QWEN38_AFFINE_MLP_GATE_UP_METAL_SOURCE,
                QWEN38_AFFINE_M234_METAL_HEADER,
                false);
            affine_mlp_gate_up_mixed_q5 = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mlp_gate_up_mixed_q5_v1",
                {
                    "x",
                    "gate_weight",
                    "gate_scales",
                    "gate_biases",
                    "up_weight",
                    "up_scales",
                    "up_biases",
                },
                {"out"},
                QWEN38_AFFINE_MLP_GATE_UP_MIXED_Q5_METAL_SOURCE,
                QWEN38_AFFINE_M234_METAL_HEADER,
                false);
            gdn_prework = mlx::core::fast::metal_kernel(
                "qw_qwen38_gdn_prework_f32_v1",
                {"qkv", "conv_state", "conv_weight", "q_scale", "k_scale", "eps"},
                {"q_out", "k_out", "v_out", "conv_tail"},
                QWEN38_GDN_PREWORK_METAL_SOURCE,
                "",
                false);
        });
    }
};

Qwen38KernelHolder& qwen38_kernels() {
    static Qwen38KernelHolder holder;
    holder.initialize();
    return holder;
}

bool qwen38_affine_shape(int32_t in_features, int32_t out_features) {
    return (in_features == 5120 && out_features == 48)
        || (in_features == 5120 && out_features == 17408)
        || (in_features == 17408 && out_features == 5120)
        || (in_features == 5120 && out_features == 10240)
        || (in_features == 5120 && out_features == 6144)
        || (in_features == 6144 && out_features == 5120)
        || (in_features == 5120 && out_features == 12288)
        || (in_features == 10240 && out_features == 5120);
}


void validate_affine_planes(
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int32_t bits,
    int32_t in_features,
    int32_t out_features,
    bool use_mixed_q5
) {
    using namespace mlx::core;
    const int64_t packed_width =
        static_cast<int64_t>(in_features) * bits / 32;
    const int64_t groups = in_features / 32;
    const bool f16_sidecars =
        scales.dtype() == float16 && biases.dtype() == float16;
    const bool valid_sidecars =
        (scales.dtype() == float32 && biases.dtype() == float32)
        || (bits == 5 && f16_sidecars);
    if ((bits != 4 && bits != 5 && bits != 8)
            || in_features <= 0 || in_features % 512 != 0
            || out_features <= 0 || out_features % 8 != 0
            || packed_width > std::numeric_limits<int32_t>::max()
            || weight.dtype() != uint32
            || weight.shape() != Shape{out_features, static_cast<int32_t>(packed_width)}
            || !valid_sidecars
            || (use_mixed_q5 && !f16_sidecars)
            || scales.shape() != Shape{out_features, static_cast<int32_t>(groups)}
            || biases.shape() != Shape{out_features, static_cast<int32_t>(groups)}) {
        throw std::invalid_argument("pinned affine M2/M3/M4 planes are invalid");
    }
}

} // namespace

std::unique_ptr<MlxArray> qwen38_affine_m234_matmul(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    int32_t bits,
    int32_t in_features,
    int32_t out_features,
    int32_t input_rows,
    bool require_pinned_shape
) {
#ifndef __APPLE__
    throw std::invalid_argument("pinned affine execution requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("pinned affine execution requires Metal");
    }
    const bool mixed_q5 = bits == 5
        && scales.inner.dtype() == float16
        && biases.inner.dtype() == float16
        && (input_rows == 1 || input_rows == 3 || input_rows == 4)
        && qwen38_affine_shape(in_features, out_features);
    if ((require_pinned_shape && !qwen38_affine_shape(in_features, out_features))
            || (!mixed_q5 && input_rows != 2 && input_rows != 3 && input_rows != 4)
            || x.inner.dtype() != float32
            || x.inner.size() != static_cast<size_t>(input_rows)
                * static_cast<size_t>(in_features)
            || x.inner.shape().empty()
            || x.inner.shape().back() != in_features) {
        throw std::invalid_argument("pinned affine activation is invalid");
    }
    validate_affine_planes(
        weight.inner,
        scales.inner,
        biases.inner,
        bits,
        in_features,
        out_features,
        mixed_q5);

    auto input = contiguous(reshape(x.inner, {input_rows, in_features}));
    auto kernel_input = mixed_q5 ? astype(input, float16) : input;
    const auto args = mixed_q5
        ? std::vector<std::pair<std::string, fast::TemplateArg>>{
              {"MRows", input_rows}}
        : std::vector<std::pair<std::string, fast::TemplateArg>>{
              {"Bits", bits}, {"MRows", input_rows}};
    const int32_t output_rows_per_threadgroup =
        mixed_q5 && input_rows == 3 ? 2 : 8;
    auto results = mixed_q5
        ? (*qwen38_kernels().affine_mixed_q5)(
              {kernel_input, weight.inner, scales.inner, biases.inner},
              {Shape{input_rows, out_features}},
              {float32},
              std::make_tuple(
                  32,
                  ((out_features + output_rows_per_threadgroup - 1)
                   / output_rows_per_threadgroup) * 2,
                  1),
              std::make_tuple(32, 2, 1),
              args,
              std::nullopt,
              false,
              {})
        : (*qwen38_kernels().affine_m234)(
              {input, weight.inner, scales.inner, biases.inner},
              {Shape{input_rows, out_features}},
              {float32},
              std::make_tuple(32, ((out_features + 7) / 8) * 2, 1),
              std::make_tuple(32, 2, 1),
              args,
              std::nullopt,
              false,
              {});
    Shape output_shape(x.inner.shape().begin(), x.inner.shape().end() - 1);
    output_shape.push_back(out_features);
    return std::make_unique<MlxArray>(reshape(results[0], output_shape));
#endif
}

std::unique_ptr<MlxArray> qwen38_affine_mlp_gate_up(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& gate_scales,
    const MlxArray& gate_biases,
    const MlxArray& up_weight,
    const MlxArray& up_scales,
    const MlxArray& up_biases,
    int32_t bits,
    int32_t input_rows
) {
#ifndef __APPLE__
    throw std::invalid_argument("pinned affine MLP gate/up requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("pinned affine MLP gate/up requires Metal");
    }
    constexpr int32_t in_features = 5120;
    constexpr int32_t out_features = 17408;
    const bool mixed_q5 = bits == 5
        && gate_scales.inner.dtype() == float16
        && gate_biases.inner.dtype() == float16;
    if ((input_rows != 1 && input_rows != 3 && input_rows != 4)
            || x.inner.dtype() != float32
            || x.inner.size() != static_cast<size_t>(input_rows * in_features)
            || x.inner.shape().empty()
            || x.inner.shape().back() != in_features
            || gate_scales.inner.dtype() != up_scales.inner.dtype()
            || gate_biases.inner.dtype() != up_biases.inner.dtype()) {
        throw std::invalid_argument("pinned affine MLP gate/up activation is invalid");
    }
    validate_affine_planes(
        gate_weight.inner,
        gate_scales.inner,
        gate_biases.inner,
        bits,
        in_features,
        out_features,
        mixed_q5);
    validate_affine_planes(
        up_weight.inner,
        up_scales.inner,
        up_biases.inner,
        bits,
        in_features,
        out_features,
        mixed_q5);

    // The custom kernels flat-index every input. Contiguous elides the data
    // copy for canonical runtime arrays and materializes public strided views.
    auto input = reshape(contiguous(x.inner), {input_rows, in_features});
    auto kernel_input =
        mixed_q5 ? contiguous(astype(input, float16)) : input;
    auto gate_weight_input = contiguous(gate_weight.inner);
    auto gate_scales_input = contiguous(gate_scales.inner);
    auto gate_biases_input = contiguous(gate_biases.inner);
    auto up_weight_input = contiguous(up_weight.inner);
    auto up_scales_input = contiguous(up_scales.inner);
    auto up_biases_input = contiguous(up_biases.inner);
    const auto args = mixed_q5
        ? std::vector<std::pair<std::string, fast::TemplateArg>>{
              {"MRows", input_rows}}
        : std::vector<std::pair<std::string, fast::TemplateArg>>{
              {"Bits", bits}, {"MRows", input_rows}};
    const int32_t output_rows_per_threadgroup =
        mixed_q5 && input_rows == 3 ? 2 : 8;
    const std::vector<array> inputs = {
        kernel_input,
        gate_weight_input,
        gate_scales_input,
        gate_biases_input,
        up_weight_input,
        up_scales_input,
        up_biases_input,
    };
    auto results = mixed_q5
        ? (*qwen38_kernels().affine_mlp_gate_up_mixed_q5)(
              inputs,
              {Shape{input_rows, out_features}},
              {float32},
              std::make_tuple(
                  32,
                  ((out_features + output_rows_per_threadgroup - 1)
                   / output_rows_per_threadgroup) * 2,
                  1),
              std::make_tuple(32, 2, 1),
              args,
              std::nullopt,
              false,
              {})
        : (*qwen38_kernels().affine_mlp_gate_up)(
              inputs,
              {Shape{input_rows, out_features}},
              {float32},
              std::make_tuple(32, ((out_features + 7) / 8) * 2, 1),
              std::make_tuple(32, 2, 1),
              args,
              std::nullopt,
              false,
              {});
    Shape output_shape(x.inner.shape().begin(), x.inner.shape().end() - 1);
    output_shape.push_back(out_features);
    return std::make_unique<MlxArray>(reshape(results[0], output_shape));
#endif
}

std::unique_ptr<Qwen38GdnPreworkOutputs> qwen38_gdn_prework(
    const MlxArray& qkv,
    const MlxArray& conv_state,
    const MlxArray& conv_weight,
    float q_scale,
    float k_scale,
    float eps
) {
#ifndef __APPLE__
    throw std::invalid_argument("pinned Qwen3.8 GDN prework requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("pinned Qwen3.8 GDN prework requires Metal");
    }
    if (qkv.inner.dtype() != float32
            || (conv_state.inner.dtype() != float32
                && conv_state.inner.dtype() != float16)
            || conv_weight.inner.dtype() != float32
            || qkv.inner.ndim() != 3
            || qkv.inner.shape(0) != 1
            || (qkv.inner.shape(1) != 1
                && qkv.inner.shape(1) != 3
                && qkv.inner.shape(1) != 4)
            || qkv.inner.shape(2) != 10240
            || conv_state.inner.shape() != Shape{1, 3, 10240}
            || conv_weight.inner.shape() != Shape{10240, 4, 1}) {
        throw std::invalid_argument("pinned Qwen3.8 GDN prework inputs are invalid");
    }

    const int rows = qkv.inner.shape(1);
    // A lazy view does not expose trustworthy stride flags until evaluation,
    // while the custom kernel receives unsigned stride metadata. Contiguous
    // returns its input unchanged for production row-major arrays and inserts
    // a safe materialization for arbitrary public-FFI views.
    const auto qkv_input = contiguous(qkv.inner);
    const auto conv_state_input = contiguous(conv_state.inner);
    const auto conv_weight_input = contiguous(conv_weight.inner);
    const auto q_scale_array = array(q_scale);
    const auto k_scale_array = array(k_scale);
    const auto eps_array = array(eps);
    // q_scale/k_scale/eps are zero-dimensional constant references in MLX's
    // custom-kernel ABI, so the Metal source uses the bare names consistently.
    auto results = (*qwen38_kernels().gdn_prework)(
        {
            qkv_input,
            conv_state_input,
            conv_weight_input,
            q_scale_array,
            k_scale_array,
            eps_array,
        },
        {
            Shape{1, rows, 16, 128},
            Shape{1, rows, 16, 128},
            Shape{1, rows, 48, 128},
            Shape{1, 3, 10240},
        },
        {float32, float32, float32, float32},
        std::make_tuple(32, std::max(rows, 3), 80),
        std::make_tuple(32, 1, 1),
        {{"MRows", rows}},
        std::nullopt,
        false,
        {});
    auto outputs = std::make_unique<Qwen38GdnPreworkOutputs>();
    outputs->q = std::make_unique<MlxArray>(std::move(results[0]));
    outputs->k = std::make_unique<MlxArray>(std::move(results[1]));
    outputs->v = std::make_unique<MlxArray>(std::move(results[2]));
    outputs->conv_tail = std::make_unique<MlxArray>(std::move(results[3]));
    return outputs;
#endif
}

std::unique_ptr<MlxArray> qwen38_gdn_prework_take_q(Qwen38GdnPreworkOutputs& outputs) {
    return std::move(outputs.q);
}

std::unique_ptr<MlxArray> qwen38_gdn_prework_take_k(Qwen38GdnPreworkOutputs& outputs) {
    return std::move(outputs.k);
}

std::unique_ptr<MlxArray> qwen38_gdn_prework_take_v(Qwen38GdnPreworkOutputs& outputs) {
    return std::move(outputs.v);
}

std::unique_ptr<MlxArray> qwen38_gdn_prework_take_conv_tail(
    Qwen38GdnPreworkOutputs& outputs
) {
    return std::move(outputs.conv_tail);
}

} // namespace mlx_cxx
