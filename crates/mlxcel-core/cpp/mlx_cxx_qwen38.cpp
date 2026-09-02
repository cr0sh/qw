// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");

// Fixed-artifact Metal kernels for the pinned Qwen3.8 target.

#include "mlx_cxx_internal.h"

#include <limits>

namespace mlx_cxx {
namespace {

static const char* QWEN38_AFFINE_M23_METAL_HEADER = R"(
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
)";

static const char* QWEN38_AFFINE_M23_METAL_SOURCE = R"(
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

struct Qwen38KernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> affine_m23;
    std::once_flag initialize_once;

    void initialize() {
        std::call_once(initialize_once, [this] {
            affine_m23 = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_m23_v1",
                {"x", "weight", "scales", "biases"},
                {"out"},
                QWEN38_AFFINE_M23_METAL_SOURCE,
                QWEN38_AFFINE_M23_METAL_HEADER,
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
    return (in_features == 5120 && out_features == 17408)
        || (in_features == 17408 && out_features == 5120)
        || (in_features == 5120 && out_features == 10240)
        || (in_features == 5120 && out_features == 6144)
        || (in_features == 6144 && out_features == 5120)
        || (in_features == 5120 && out_features == 12288);
}

void validate_affine_planes(
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int32_t bits,
    int32_t in_features,
    int32_t out_features
) {
    using namespace mlx::core;
    const int64_t packed_width =
        static_cast<int64_t>(in_features) * bits / 32;
    const int64_t groups = in_features / 32;
    if ((bits != 4 && bits != 5 && bits != 8)
            || in_features <= 0 || in_features % 512 != 0
            || out_features <= 0 || out_features % 8 != 0
            || packed_width > std::numeric_limits<int32_t>::max()
            || weight.dtype() != uint32
            || weight.shape() != Shape{out_features, static_cast<int32_t>(packed_width)}
            || scales.dtype() != float32
            || scales.shape() != Shape{out_features, static_cast<int32_t>(groups)}
            || biases.dtype() != float32
            || biases.shape() != Shape{out_features, static_cast<int32_t>(groups)}) {
        throw std::invalid_argument("pinned affine M2/M3 planes are invalid");
    }
}

} // namespace

std::unique_ptr<MlxArray> qwen38_affine_m23_matmul(
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
    throw std::invalid_argument("pinned affine M2/M3 execution requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("pinned affine M2/M3 execution requires Metal");
    }
    if ((require_pinned_shape && !qwen38_affine_shape(in_features, out_features))
            || (input_rows != 2 && input_rows != 3)
            || x.inner.dtype() != float32
            || x.inner.size() != static_cast<size_t>(input_rows)
                * static_cast<size_t>(in_features)
            || x.inner.shape().empty()
            || x.inner.shape().back() != in_features) {
        throw std::invalid_argument("pinned affine M2/M3 activation is invalid");
    }
    validate_affine_planes(
        weight.inner,
        scales.inner,
        biases.inner,
        bits,
        in_features,
        out_features);

    auto input = contiguous(reshape(x.inner, {input_rows, in_features}));
    const auto args = std::vector<std::pair<std::string, fast::TemplateArg>>{
        {"Bits", bits}, {"MRows", input_rows}};
    auto results = (*qwen38_kernels().affine_m23)(
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

} // namespace mlx_cxx
