// Direct Metal execution of original GGUF block bytes for the fixed Qwen3.8
// runtime. No dense weight tensor is created by this translation unit.

#include "mlx_cxx_internal.h"

#include <climits>
#include <limits>

namespace mlx_cxx {
namespace {

struct GgmlBlockInfo {
    int32_t elements;
    int32_t bytes;
};

GgmlBlockInfo ggml_block_info(int32_t qtype) {
    switch (qtype) {
        case 0:  return {1, 4};    // F32
        case 8:  return {32, 34};  // Q8_0
        case 11: return {256, 110}; // Q3_K
        case 12: return {256, 144}; // Q4_K
        case 13: return {256, 176}; // Q5_K
        case 14: return {256, 210}; // Q6_K
        case 20: return {32, 18};   // IQ4_NL
        case 21: return {256, 110}; // IQ3_S
        case 23: return {256, 136}; // IQ4_XS
        default:
            throw std::invalid_argument("unsupported packed GGML qtype");
    }
}

void validate_iq3_table(const mlx::core::array& table, int32_t qtype) {
    const size_t expected = qtype == 21 ? 512u : 1u;
    if (table.dtype() != mlx::core::uint32 || table.size() != expected) {
        throw std::invalid_argument("packed GGML IQ3 table shape or dtype mismatch");
    }
}

// MLX fast::metal_kernel supplies the named buffers and template arguments to
// this body. One simdgroup owns either one output row (decode), a tile of up to
// four activation rows (prefill/verify), or one selected embedding row.
static const char* GGML_PACKED_METAL_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;
    uint output_group = thread_position_in_grid.y;

    if (embedding_mode) {
        uint selected = output_group;
        uint row = (uint)indices[selected];
        bool valid_row = row < (uint)out_features;
        uint row_base = valid_row ? row * (uint)row_bytes : 0u;

        for (uint column = lane; column < (uint)in_features; column += 32u) {
            float weight = 0.0f;
            if (valid_row) {
                if (qtype == 0) {
                    uint base = row_base + column * 4u;
                    uint bits = (uint)packed[base]
                        | ((uint)packed[base + 1u] << 8u)
                        | ((uint)packed[base + 2u] << 16u)
                        | ((uint)packed[base + 3u] << 24u);
                    weight = as_type<float>(bits);
                } else if (qtype == 8) {
                    uint base = row_base + (column >> 5u) * 34u;
                    uint local = column & 31u;
                    ushort bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
                    weight = (float)as_type<half>(bits) * (float)(char)packed[base + 2u + local];
                } else if (qtype == 11) {
                    uint local = column & 255u;
                    uint base = row_base + (column >> 8u) * 110u;
                    uint group = local >> 4u;
                    uint scale_low = group < 8u
                        ? ((uint)packed[base + 96u + group] & 15u)
                        : ((uint)packed[base + 88u + group] >> 4u);
                    uint scale_high = ((uint)packed[base + 104u + (group & 3u)]
                        >> (2u * (group >> 2u))) & 3u;
                    int scale = (int)(scale_low | (scale_high << 4u)) - 32;
                    uint q = (uint)packed[base + 32u + ((local >> 7u) * 32u) + (local & 31u)];
                    uint low = (q >> (2u * ((local >> 5u) & 3u))) & 3u;
                    uint high = (uint)packed[base + (local & 31u)] & (1u << (local >> 5u));
                    int quant = (int)low - (high != 0u ? 0 : 4);
                    ushort bits = (ushort)packed[base + 108u] | ((ushort)packed[base + 109u] << 8u);
                    weight = (float)as_type<half>(bits) * (float)scale * (float)quant;
                } else if (qtype == 12 || qtype == 13) {
                    uint local = column & 255u;
                    uint group = local >> 5u;
                    uint group_column = local & 31u;
                    uint block_bytes = qtype == 12 ? 144u : 176u;
                    uint base = row_base + (column >> 8u) * block_bytes;
                    ushort d_bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
                    ushort min_bits = (ushort)packed[base + 2u] | ((ushort)packed[base + 3u] << 8u);
                    uint scale;
                    uint minimum;
                    if (group < 4u) {
                        scale = (uint)packed[base + 4u + group] & 63u;
                        minimum = (uint)packed[base + 8u + group] & 63u;
                    } else {
                        scale = ((uint)packed[base + 8u + group] & 15u)
                            | (((uint)packed[base + group] >> 6u) << 4u);
                        minimum = ((uint)packed[base + 8u + group] >> 4u)
                            | (((uint)packed[base + 4u + group] >> 6u) << 4u);
                    }
                    uint quant_offset = qtype == 12 ? 16u : 48u;
                    uchar q = packed[base + quant_offset + (group >> 1u) * 32u + group_column];
                    uint quant = (group & 1u) == 0u ? ((uint)q & 15u) : ((uint)q >> 4u);
                    if (qtype == 13) {
                        quant += (packed[base + 16u + group_column] & (uchar)(1u << group)) != 0 ? 16u : 0u;
                    }
                    weight = (float)as_type<half>(d_bits) * (float)scale * (float)quant
                        - (float)as_type<half>(min_bits) * (float)minimum;
                } else if (qtype == 14) {
                    uint local = column & 255u;
                    uint half_index = local >> 7u;
                    uint half_local = local & 127u;
                    uint quadrant = half_local >> 5u;
                    uint group_column = half_local & 31u;
                    uint base = row_base + (column >> 8u) * 210u;
                    uchar ql = packed[base + half_index * 64u + group_column + (quadrant & 1u) * 32u];
                    uchar qh = packed[base + 128u + half_index * 32u + group_column];
                    uint low = quadrant < 2u ? ((uint)ql & 15u) : ((uint)ql >> 4u);
                    int quant = (int)(low | ((((uint)qh >> (quadrant * 2u)) & 3u) << 4u)) - 32;
                    int scale = (int)(char)packed[base + 192u + half_index * 8u
                        + group_column / 16u + quadrant * 2u];
                    ushort bits = (ushort)packed[base + 208u] | ((ushort)packed[base + 209u] << 8u);
                    weight = (float)as_type<half>(bits) * (float)scale * (float)quant;
                } else if (qtype == 20) {
                    uint local = column & 31u;
                    uint base = row_base + (column >> 5u) * 18u;
                    uchar q = packed[base + 2u + (local & 15u)];
                    uint index = local < 16u ? ((uint)q & 15u) : ((uint)q >> 4u);
                    constexpr char values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                        1, 13, 25, 38, 53, 69, 89, 113};
                    ushort bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
                    weight = (float)as_type<half>(bits) * (float)values[index];
                } else if (qtype == 21) {
                    uint local = column & 255u;
                    uint block = row_base + (column >> 8u) * 110u;
                    uint group = local >> 5u;
                    uint subblock = (local & 31u) >> 3u;
                    uint lane8 = local & 7u;
                    uint high = ((uint)packed[block + 66u + group]
                        >> (2u * subblock + (lane8 >= 4u ? 1u : 0u))) & 1u;
                    uint grid_index = (uint)packed[block + 2u + group * 8u + subblock * 2u
                        + (lane8 >= 4u ? 1u : 0u)] | (high << 8u);
                    uint grid = iq3_grid[grid_index];
                    uint magnitude = (grid >> ((lane8 & 3u) * 8u)) & 255u;
                    uint sign = ((uint)packed[block + 74u + group * 4u + subblock] >> lane8) & 1u;
                    uint scales = (uint)packed[block + 106u + group / 2u];
                    uint scale = (group & 1u) == 0u ? (scales & 15u) : (scales >> 4u);
                    ushort bits = (ushort)packed[block] | ((ushort)packed[block + 1u] << 8u);
                    weight = (float)as_type<half>(bits) * (float)(1u + 2u * scale)
                        * (float)magnitude * (sign != 0u ? -1.0f : 1.0f);
                } else if (qtype == 23) {
                    uint local = column & 255u;
                    uint block = row_base + (column >> 8u) * 136u;
                    uint group = local >> 5u;
                    uint group_column = local & 31u;
                    uint scales_high = (uint)packed[block + 2u] | ((uint)packed[block + 3u] << 8u);
                    uint scales_low = (uint)packed[block + 4u + group / 2u];
                    uint scale_bits = ((scales_low >> (4u * (group & 1u))) & 15u)
                        | (((scales_high >> (2u * group)) & 3u) << 4u);
                    int scale = (int)scale_bits - 32;
                    uchar q = packed[block + 8u + group * 16u + (group_column & 15u)];
                    uint index = group_column < 16u ? ((uint)q & 15u) : ((uint)q >> 4u);
                    constexpr char values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                        1, 13, 25, 38, 53, 69, 89, 113};
                    ushort bits = (ushort)packed[block] | ((ushort)packed[block + 1u] << 8u);
                    weight = (float)as_type<half>(bits) * (float)scale * (float)values[index];
                }
            }
            out[selected * (uint)in_features + column] = weight;
        }
        return;
    }

    uint tile = output_group / (uint)out_features;
    uint projection_row = output_group - tile * (uint)out_features;
    uint first_input_row = tile * (uint)rows_per_group;
    uint row_base = projection_row * (uint)row_bytes;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint column = lane; column < (uint)in_features; column += 32u) {
        float weight = 0.0f;
        if (qtype == 0) {
            uint base = row_base + column * 4u;
            uint bits = (uint)packed[base]
                | ((uint)packed[base + 1u] << 8u)
                | ((uint)packed[base + 2u] << 16u)
                | ((uint)packed[base + 3u] << 24u);
            weight = as_type<float>(bits);
        } else if (qtype == 8) {
            uint base = row_base + (column >> 5u) * 34u;
            uint local = column & 31u;
            ushort bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
            weight = (float)as_type<half>(bits) * (float)(char)packed[base + 2u + local];
        } else if (qtype == 11) {
            uint local = column & 255u;
            uint base = row_base + (column >> 8u) * 110u;
            uint group = local >> 4u;
            uint scale_low = group < 8u
                ? ((uint)packed[base + 96u + group] & 15u)
                : ((uint)packed[base + 88u + group] >> 4u);
            uint scale_high = ((uint)packed[base + 104u + (group & 3u)]
                >> (2u * (group >> 2u))) & 3u;
            int scale = (int)(scale_low | (scale_high << 4u)) - 32;
            uint q = (uint)packed[base + 32u + ((local >> 7u) * 32u) + (local & 31u)];
            uint low = (q >> (2u * ((local >> 5u) & 3u))) & 3u;
            uint high = (uint)packed[base + (local & 31u)] & (1u << (local >> 5u));
            int quant = (int)low - (high != 0u ? 0 : 4);
            ushort bits = (ushort)packed[base + 108u] | ((ushort)packed[base + 109u] << 8u);
            weight = (float)as_type<half>(bits) * (float)scale * (float)quant;
        } else if (qtype == 12 || qtype == 13) {
            uint local = column & 255u;
            uint group = local >> 5u;
            uint group_column = local & 31u;
            uint block_bytes = qtype == 12 ? 144u : 176u;
            uint base = row_base + (column >> 8u) * block_bytes;
            ushort d_bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
            ushort min_bits = (ushort)packed[base + 2u] | ((ushort)packed[base + 3u] << 8u);
            uint scale;
            uint minimum;
            if (group < 4u) {
                scale = (uint)packed[base + 4u + group] & 63u;
                minimum = (uint)packed[base + 8u + group] & 63u;
            } else {
                scale = ((uint)packed[base + 8u + group] & 15u)
                    | (((uint)packed[base + group] >> 6u) << 4u);
                minimum = ((uint)packed[base + 8u + group] >> 4u)
                    | (((uint)packed[base + 4u + group] >> 6u) << 4u);
            }
            uint quant_offset = qtype == 12 ? 16u : 48u;
            uchar q = packed[base + quant_offset + (group >> 1u) * 32u + group_column];
            uint quant = (group & 1u) == 0u ? ((uint)q & 15u) : ((uint)q >> 4u);
            if (qtype == 13) {
                quant += (packed[base + 16u + group_column] & (uchar)(1u << group)) != 0 ? 16u : 0u;
            }
            weight = (float)as_type<half>(d_bits) * (float)scale * (float)quant
                - (float)as_type<half>(min_bits) * (float)minimum;
        } else if (qtype == 14) {
            uint local = column & 255u;
            uint half_index = local >> 7u;
            uint half_local = local & 127u;
            uint quadrant = half_local >> 5u;
            uint group_column = half_local & 31u;
            uint base = row_base + (column >> 8u) * 210u;
            uchar ql = packed[base + half_index * 64u + group_column + (quadrant & 1u) * 32u];
            uchar qh = packed[base + 128u + half_index * 32u + group_column];
            uint low = quadrant < 2u ? ((uint)ql & 15u) : ((uint)ql >> 4u);
            int quant = (int)(low | ((((uint)qh >> (quadrant * 2u)) & 3u) << 4u)) - 32;
            int scale = (int)(char)packed[base + 192u + half_index * 8u
                + group_column / 16u + quadrant * 2u];
            ushort bits = (ushort)packed[base + 208u] | ((ushort)packed[base + 209u] << 8u);
            weight = (float)as_type<half>(bits) * (float)scale * (float)quant;
        } else if (qtype == 20) {
            uint local = column & 31u;
            uint base = row_base + (column >> 5u) * 18u;
            uchar q = packed[base + 2u + (local & 15u)];
            uint index = local < 16u ? ((uint)q & 15u) : ((uint)q >> 4u);
            constexpr char values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                1, 13, 25, 38, 53, 69, 89, 113};
            ushort bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
            weight = (float)as_type<half>(bits) * (float)values[index];
        } else if (qtype == 21) {
            uint local = column & 255u;
            uint block = row_base + (column >> 8u) * 110u;
            uint group = local >> 5u;
            uint subblock = (local & 31u) >> 3u;
            uint lane8 = local & 7u;
            uint high = ((uint)packed[block + 66u + group]
                >> (2u * subblock + (lane8 >= 4u ? 1u : 0u))) & 1u;
            uint grid_index = (uint)packed[block + 2u + group * 8u + subblock * 2u
                + (lane8 >= 4u ? 1u : 0u)] | (high << 8u);
            uint grid = iq3_grid[grid_index];
            uint magnitude = (grid >> ((lane8 & 3u) * 8u)) & 255u;
            uint sign = ((uint)packed[block + 74u + group * 4u + subblock] >> lane8) & 1u;
            uint scales = (uint)packed[block + 106u + group / 2u];
            uint scale = (group & 1u) == 0u ? (scales & 15u) : (scales >> 4u);
            ushort bits = (ushort)packed[block] | ((ushort)packed[block + 1u] << 8u);
            weight = (float)as_type<half>(bits) * (float)(1u + 2u * scale)
                * (float)magnitude * (sign != 0u ? -1.0f : 1.0f);
        } else if (qtype == 23) {
            uint local = column & 255u;
            uint block = row_base + (column >> 8u) * 136u;
            uint group = local >> 5u;
            uint group_column = local & 31u;
            uint scales_high = (uint)packed[block + 2u] | ((uint)packed[block + 3u] << 8u);
            uint scales_low = (uint)packed[block + 4u + group / 2u];
            uint scale_bits = ((scales_low >> (4u * (group & 1u))) & 15u)
                | (((scales_high >> (2u * group)) & 3u) << 4u);
            int scale = (int)scale_bits - 32;
            uchar q = packed[block + 8u + group * 16u + (group_column & 15u)];
            uint index = group_column < 16u ? ((uint)q & 15u) : ((uint)q >> 4u);
            constexpr char values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                1, 13, 25, 38, 53, 69, 89, 113};
            ushort bits = (ushort)packed[block] | ((ushort)packed[block + 1u] << 8u);
            weight = (float)as_type<half>(bits) * (float)scale * (float)values[index];
        }

        for (uint row_offset = 0u; row_offset < (uint)rows_per_group; ++row_offset) {
            uint input_row = first_input_row + row_offset;
            if (input_row < (uint)input_rows) {
                sums[row_offset] += x[input_row * (uint)in_features + column] * weight;
            }
        }
    }

    for (uint row_offset = 0u; row_offset < (uint)rows_per_group; ++row_offset) {
        float total = simd_sum(sums[row_offset]);
        uint input_row = first_input_row + row_offset;
        if (lane == 0u && input_row < (uint)input_rows) {
            out[input_row * (uint)out_features + projection_row] = total;
        }
    }
)";

struct GgmlKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> decode;
    std::optional<mlx::core::fast::CustomKernelFunction> prefill;
    std::optional<mlx::core::fast::CustomKernelFunction> embedding;
    std::once_flag initialize_once;

    void initialize() {
        std::call_once(initialize_once, [this] {
            const std::vector<std::string> inputs = {"x", "packed", "indices", "iq3_grid"};
            decode = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_decode_v1", inputs, {"out"}, GGML_PACKED_METAL_SOURCE);
            prefill = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_prefill4_v1", inputs, {"out"}, GGML_PACKED_METAL_SOURCE);
            embedding = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_embedding_v1", inputs, {"out"}, GGML_PACKED_METAL_SOURCE);
        });
    }
};

GgmlKernelHolder& ggml_kernels() {
    static GgmlKernelHolder holder;
    holder.initialize();
    return holder;
}

size_t checked_packed_size(int32_t in_features, int32_t out_features, GgmlBlockInfo info) {
    if (in_features <= 0 || out_features <= 0 || in_features % info.elements != 0) {
        throw std::invalid_argument("packed GGML matrix shape is invalid");
    }
    const size_t blocks = static_cast<size_t>(in_features / info.elements);
    if (blocks > std::numeric_limits<size_t>::max() / static_cast<size_t>(info.bytes)) {
        throw std::invalid_argument("packed GGML row byte count overflow");
    }
    const size_t row_bytes = blocks * static_cast<size_t>(info.bytes);
    if (row_bytes > static_cast<size_t>(INT32_MAX)
        || static_cast<size_t>(out_features) > std::numeric_limits<size_t>::max() / row_bytes) {
        throw std::invalid_argument("packed GGML matrix byte count overflow");
    }
    return row_bytes * static_cast<size_t>(out_features);
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_arguments(
    int32_t qtype,
    int32_t in_features,
    int32_t out_features,
    int32_t row_bytes,
    int32_t input_rows,
    int32_t rows_per_group,
    bool embedding_mode
) {
    return {
        {"qtype", qtype},
        {"in_features", in_features},
        {"out_features", out_features},
        {"row_bytes", row_bytes},
        {"input_rows", input_rows},
        {"rows_per_group", rows_per_group},
        {"embedding_mode", embedding_mode ? 1 : 0},
    };
}

} // namespace

std::unique_ptr<MlxArray> ggml_packed_matmul(
    const MlxArray& x,
    const MlxArray& packed,
    const MlxArray& iq3_grid,
    int32_t qtype,
    int32_t in_features,
    int32_t out_features,
    int32_t input_rows
) {
#ifndef __APPLE__
    throw std::invalid_argument("packed GGML execution requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("packed GGML execution requires Metal");
    }
    const GgmlBlockInfo info = ggml_block_info(qtype);
    const size_t expected = checked_packed_size(in_features, out_features, info);
    if (packed.inner.dtype() != uint8 || packed.inner.size() != expected) {
        throw std::invalid_argument("packed GGML buffer shape, dtype, or bounds mismatch");
    }
    validate_iq3_table(iq3_grid.inner, qtype);
    const auto& shape = x.inner.shape();
    if (shape.empty() || shape.back() != in_features || input_rows <= 0
        || x.inner.size() != static_cast<size_t>(input_rows) * static_cast<size_t>(in_features)) {
        throw std::invalid_argument("packed GGML activation shape mismatch");
    }
    const int32_t rows_per_group = input_rows == 1 ? 1 : std::min(input_rows, 4);
    const int64_t tiles = (static_cast<int64_t>(input_rows) + rows_per_group - 1) / rows_per_group;
    const int64_t output_groups = tiles * out_features;
    if (output_groups <= 0 || output_groups > INT32_MAX) {
        throw std::invalid_argument("packed GGML output grid overflow");
    }
    const int32_t row_bytes = in_features / info.elements * info.bytes;
    auto input = contiguous(reshape(astype(x.inner, float32), {input_rows, in_features}));
    auto& holder = ggml_kernels();
    auto& kernel = input_rows == 1 ? *holder.decode : *holder.prefill;
    const auto args = template_arguments(
        qtype, in_features, out_features, row_bytes, input_rows, rows_per_group, false);
    auto results = kernel(
        {input, packed.inner, iq3_grid.inner, iq3_grid.inner},
        {Shape{input_rows, out_features}},
        {float32},
        std::make_tuple(32, static_cast<int32_t>(output_groups), 1),
        std::make_tuple(32, 1, 1),
        args,
        std::nullopt,
        false,
        {});
    Shape output_shape(shape.begin(), shape.end() - 1);
    output_shape.push_back(out_features);
    return std::make_unique<MlxArray>(reshape(results[0], output_shape));
#endif
}

std::unique_ptr<MlxArray> ggml_packed_embedding(
    const MlxArray& indices,
    const MlxArray& packed,
    const MlxArray& iq3_grid,
    int32_t qtype,
    int32_t embedding_dim,
    int32_t vocab_size
) {
#ifndef __APPLE__
    throw std::invalid_argument("packed GGML execution requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("packed GGML execution requires Metal");
    }
    const GgmlBlockInfo info = ggml_block_info(qtype);
    const size_t expected = checked_packed_size(embedding_dim, vocab_size, info);
    if (packed.inner.dtype() != uint8 || packed.inner.size() != expected) {
        throw std::invalid_argument("packed GGML embedding buffer shape, dtype, or bounds mismatch");
    }
    validate_iq3_table(iq3_grid.inner, qtype);
    if (indices.inner.size() == 0 || indices.inner.size() > static_cast<size_t>(INT32_MAX)) {
        throw std::invalid_argument("packed GGML embedding selection grid is invalid");
    }
    const int32_t selected_rows = static_cast<int32_t>(indices.inner.size());
    const int32_t row_bytes = embedding_dim / info.elements * info.bytes;
    auto index_data = contiguous(astype(indices.inner, uint32));
    auto& kernel = *ggml_kernels().embedding;
    const auto args = template_arguments(
        qtype, embedding_dim, vocab_size, row_bytes, selected_rows, 1, true);
    auto results = kernel(
        {packed.inner, packed.inner, index_data, iq3_grid.inner},
        {Shape{selected_rows, embedding_dim}},
        {float32},
        std::make_tuple(32, selected_rows, 1),
        std::make_tuple(32, 1, 1),
        args,
        std::nullopt,
        false,
        {});
    Shape output_shape(indices.inner.shape().begin(), indices.inner.shape().end());
    output_shape.push_back(embedding_dim);
    return std::make_unique<MlxArray>(reshape(results[0], output_shape));
#endif
}

} // namespace mlx_cxx
