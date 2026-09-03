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
    const size_t expected = qtype == 21 ? 512u : 8u;
    if (table.dtype() != mlx::core::uint32 || table.size() != expected) {
        throw std::invalid_argument("packed GGML IQ3 table shape or dtype mismatch");
    }
}

static const char* GGML_DECODE_METAL_HEADER = R"(
inline uint ggml_block_elements(int qtype) {
    return qtype == 0 ? 1u : ((qtype == 8 || qtype == 20) ? 32u : 256u);
}

inline uint ggml_block_bytes(int qtype) {
    if (qtype == 0) return 4u;
    if (qtype == 8) return 34u;
    if (qtype == 11 || qtype == 21) return 110u;
    if (qtype == 12) return 144u;
    if (qtype == 13) return 176u;
    if (qtype == 14) return 210u;
    if (qtype == 20) return 18u;
    return 136u;
}

inline uint ggml_source_row(
    const constant uint* row_ranges,
    uint range_count,
    uint output_row
) {
    for (uint range = 0u; range < range_count; ++range) {
        uint count = row_ranges[range * 2u + 1u];
        if (output_row < count) {
            return row_ranges[range * 2u] + output_row;
        }
        output_row -= count;
    }
    return 0xffffffffu;
}

inline float ggml_decode_weight(
    const device uchar* packed,
    uint row_base,
    uint column,
    int qtype,
    const device uint* iq3_grid
) {
    if (qtype == 0) {
        uint base = row_base + column * 4u;
        uint bits = (uint)packed[base]
            | ((uint)packed[base + 1u] << 8u)
            | ((uint)packed[base + 2u] << 16u)
            | ((uint)packed[base + 3u] << 24u);
        return as_type<float>(bits);
    }
    if (qtype == 8) {
        uint base = row_base + (column >> 5u) * 34u;
        uint local = column & 31u;
        ushort bits = (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
        return (float)as_type<half>(bits) * (float)(char)packed[base + 2u + local];
    }
    if (qtype == 11) {
        uint local = column & 255u;
        uint base = row_base + (column >> 8u) * 110u;
        uint group = local >> 4u;
        uint scale_low = group < 8u
            ? ((uint)packed[base + 96u + group] & 15u)
            : ((uint)packed[base + 88u + group] >> 4u);
        uint scale_high = ((uint)packed[base + 104u + (group & 3u)]
            >> (2u * (group >> 2u))) & 3u;
        int scale = (int)(scale_low | (scale_high << 4u)) - 32;
        uint q = (uint)packed[
            base + 32u + ((local >> 7u) * 32u) + (local & 31u)];
        uint low = (q >> (2u * ((local >> 5u) & 3u))) & 3u;
        uint high = (uint)packed[base + (local & 31u)] & (1u << (local >> 5u));
        int quant = (int)low - (high != 0u ? 0 : 4);
        ushort bits =
            (ushort)packed[base + 108u] | ((ushort)packed[base + 109u] << 8u);
        return (float)as_type<half>(bits) * (float)scale * (float)quant;
    }
    if (qtype == 12 || qtype == 13) {
        uint local = column & 255u;
        uint group = local >> 5u;
        uint group_column = local & 31u;
        uint block_bytes = qtype == 12 ? 144u : 176u;
        uint base = row_base + (column >> 8u) * block_bytes;
        ushort d_bits =
            (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
        ushort min_bits =
            (ushort)packed[base + 2u] | ((ushort)packed[base + 3u] << 8u);
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
        uchar q = packed[
            base + quant_offset + (group >> 1u) * 32u + group_column];
        uint quant =
            (group & 1u) == 0u ? ((uint)q & 15u) : ((uint)q >> 4u);
        if (qtype == 13) {
            quant += (packed[base + 16u + group_column]
                          & (uchar)(1u << group)) != 0
                ? 16u
                : 0u;
        }
        return (float)as_type<half>(d_bits) * (float)scale * (float)quant
            - (float)as_type<half>(min_bits) * (float)minimum;
    }
    if (qtype == 14) {
        uint local = column & 255u;
        uint half_index = local >> 7u;
        uint half_local = local & 127u;
        uint quadrant = half_local >> 5u;
        uint group_column = half_local & 31u;
        uint base = row_base + (column >> 8u) * 210u;
        uchar ql = packed[
            base + half_index * 64u + group_column + (quadrant & 1u) * 32u];
        uchar qh =
            packed[base + 128u + half_index * 32u + group_column];
        uint low =
            quadrant < 2u ? ((uint)ql & 15u) : ((uint)ql >> 4u);
        int quant =
            (int)(low
                | ((((uint)qh >> (quadrant * 2u)) & 3u) << 4u))
            - 32;
        int scale = (int)(char)packed[
            base + 192u + half_index * 8u
            + group_column / 16u + quadrant * 2u];
        ushort bits =
            (ushort)packed[base + 208u] | ((ushort)packed[base + 209u] << 8u);
        return (float)as_type<half>(bits) * (float)scale * (float)quant;
    }
    if (qtype == 20) {
        uint local = column & 31u;
        uint base = row_base + (column >> 5u) * 18u;
        uchar q = packed[base + 2u + (local & 15u)];
        uint index =
            local < 16u ? ((uint)q & 15u) : ((uint)q >> 4u);
        constexpr char values[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};
        ushort bits =
            (ushort)packed[base] | ((ushort)packed[base + 1u] << 8u);
        return (float)as_type<half>(bits) * (float)values[index];
    }
    if (qtype == 21) {
        uint local = column & 255u;
        uint block = row_base + (column >> 8u) * 110u;
        uint group = local >> 5u;
        uint subblock = (local & 31u) >> 3u;
        uint lane8 = local & 7u;
        uint high = ((uint)packed[block + 66u + group]
            >> (2u * subblock + (lane8 >= 4u ? 1u : 0u))) & 1u;
        uint grid_index =
            (uint)packed[
                block + 2u + group * 8u + subblock * 2u
                + (lane8 >= 4u ? 1u : 0u)]
            | (high << 8u);
        uint grid = iq3_grid[grid_index];
        uint magnitude =
            (grid >> ((lane8 & 3u) * 8u)) & 255u;
        uint sign = ((uint)packed[
            block + 74u + group * 4u + subblock] >> lane8) & 1u;
        uint scales = (uint)packed[block + 106u + group / 2u];
        uint scale =
            (group & 1u) == 0u ? (scales & 15u) : (scales >> 4u);
        ushort bits =
            (ushort)packed[block] | ((ushort)packed[block + 1u] << 8u);
        return (float)as_type<half>(bits)
            * (float)(1u + 2u * scale)
            * (float)magnitude
            * (sign != 0u ? -1.0f : 1.0f);
    }

    uint local = column & 255u;
    uint block = row_base + (column >> 8u) * 136u;
    uint group = local >> 5u;
    uint group_column = local & 31u;
    uint scales_high =
        (uint)packed[block + 2u] | ((uint)packed[block + 3u] << 8u);
    uint scales_low = (uint)packed[block + 4u + group / 2u];
    uint scale_bits =
        ((scales_low >> (4u * (group & 1u))) & 15u)
        | (((scales_high >> (2u * group)) & 3u) << 4u);
    int scale = (int)scale_bits - 32;
    uchar q = packed[
        block + 8u + group * 16u + (group_column & 15u)];
    uint index =
        group_column < 16u ? ((uint)q & 15u) : ((uint)q >> 4u);
    constexpr char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113};
    ushort bits =
        (ushort)packed[block] | ((ushort)packed[block + 1u] << 8u);
    return (float)as_type<half>(bits)
        * (float)scale
        * (float)values[index];
}

inline bool ggml_argmax_better(
    float score,
    uint id,
    float best_score,
    uint best_id) {
    if (id == 0xffffffffu) {
        return false;
    }
    if (best_id == 0xffffffffu) {
        return true;
    }
    bool score_nan = metal::isnan(score);
    bool best_nan = metal::isnan(best_score);
    if (score_nan != best_nan) {
        return best_nan;
    }
    if (score_nan) {
        return id < best_id;
    }
    return score > best_score || (score == best_score && id < best_id);
}
)";

static const char* GGML_QMV_METAL_SOURCE = R"(
    uint lane = thread_index_in_simdgroup;
    uint output_row = thread_position_in_grid.y;
    uint source_row =
        ggml_source_row(row_ranges, (uint)row_ranges_shape[0], output_row);
    if (source_row >= (uint)packed_shape[0]) {
        if (lane == 0u) out[output_row] = 0.0f;
        return;
    }
    uint row_base = source_row * (uint)RowBytes;
    float sum = 0.0f;
    for (uint column = lane; column < (uint)Width; column += 32u) {
        sum += x[column]
            * ggml_decode_weight(packed, row_base, column, QType, iq3_grid);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[output_row] = sum;
)";

static const char* GGML_VERIFY_QMM_METAL_SOURCE = R"(
    uint lane = thread_index_in_simdgroup;
    uint output_row = thread_position_in_grid.y;
    uint range_count = (uint)row_ranges_shape[0];
    uint selected_rows = 0u;
    for (uint range = 0u; range < range_count; ++range) {
        selected_rows += row_ranges[range * 2u + 1u];
    }
    uint source_row =
        ggml_source_row(row_ranges, range_count, output_row);
    if (source_row >= (uint)packed_shape[0]) {
        for (uint row = 0u; row < (uint)x_shape[0]; ++row) {
            if (lane == 0u) out[row * selected_rows + output_row] = 0.0f;
        }
        return;
    }
    uint row_base = source_row * (uint)packed_shape[1];
    uint input_rows = (uint)x_shape[0];
    uint width = (uint)x_shape[1];
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < width; column += 32u) {
        float weight =
            ggml_decode_weight(packed, row_base, column, QType, iq3_grid);
        for (uint row = 0u; row < input_rows; ++row) {
            sums[row] += x[row * width + column] * weight;
        }
    }
    for (uint row = 0u; row < input_rows; ++row) {
        float total = simd_sum(sums[row]);
        if (lane == 0u) out[row * selected_rows + output_row] = total;
    }
)";

static const char* QWEN38_Q6_HEAD_VERIFY_R8_SOURCE = R"(
    constexpr uint RowsPerThreadgroup = 8u;
    uint lane = thread_index_in_simdgroup;
    uint output_row = threadgroup_position_in_grid.y * RowsPerThreadgroup
        + simdgroup_index_in_threadgroup;
    uint range_count = (uint)row_ranges_shape[0];
    uint selected_rows = 0u;
    for (uint range = 0u; range < range_count; ++range) {
        selected_rows += row_ranges[range * 2u + 1u];
    }
    if (output_row >= selected_rows) {
        return;
    }
    uint source_row =
        ggml_source_row(row_ranges, range_count, output_row);
    if (source_row >= (uint)packed_shape[0]) {
        return;
    }
    uint row_base = source_row * (uint)packed_shape[1];
    uint input_rows = (uint)x_shape[0];
    uint width = (uint)x_shape[1];
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < width; column += 32u) {
        float weight =
            ggml_decode_weight(packed, row_base, column, QType, iq3_grid);
        for (uint row = 0u; row < input_rows; ++row) {
            sums[row] += x[row * width + column] * weight;
        }
    }
    for (uint row = 0u; row < input_rows; ++row) {
        float total = simd_sum(sums[row]);
        if (lane == 0u) {
            out[row * selected_rows + output_row] = total;
        }
    }
)";

static const char* Q6_ARGMAX_PARTIAL_SOURCE = R"(
    constexpr uint RowsPerThreadgroup = 8u;
    constexpr uint MaxInputRows = 4u;
    uint lane = thread_index_in_simdgroup;
    uint candidate = threadgroup_position_in_grid.y * RowsPerThreadgroup
        + simdgroup_index_in_threadgroup;
    uint input_rows = (uint)x_shape[0];
    uint width = (uint)x_shape[1];
    uint output_rows = (uint)packed_shape[0];
    bool valid_candidate = candidate < output_rows;
    uint row_base = valid_candidate
        ? candidate * (uint)packed_shape[1]
        : 0u;
    float sums[MaxInputRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (valid_candidate) {
        for (uint column = lane; column < width; column += 32u) {
            float weight =
                ggml_decode_weight(packed, row_base, column, QType, iq3_grid);
            for (uint row = 0u; row < input_rows; ++row) {
                sums[row] += x[row * width + column] * weight;
            }
        }
    }

    for (uint row = 0u; row < input_rows; ++row) {
        sums[row] = simd_sum(sums[row]);
    }
    threadgroup float candidate_scores[RowsPerThreadgroup * MaxInputRows];
    threadgroup uint candidate_ids[RowsPerThreadgroup * MaxInputRows];
    if (lane == 0u) {
        uint slot = simdgroup_index_in_threadgroup;
        for (uint row = 0u; row < input_rows; ++row) {
            uint index = row * RowsPerThreadgroup + slot;
            candidate_scores[index] = valid_candidate ? sums[row] : -INFINITY;
            candidate_ids[index] = valid_candidate ? candidate : 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = thread_index_in_threadgroup;
    if (row < input_rows) {
        float best_score = -INFINITY;
        uint best_id = 0xffffffffu;
        for (uint slot = 0u; slot < RowsPerThreadgroup; ++slot) {
            uint index = row * RowsPerThreadgroup + slot;
            float score = candidate_scores[index];
            uint id = candidate_ids[index];
            if (ggml_argmax_better(score, id, best_score, best_id)) {
                best_score = score;
                best_id = id;
            }
        }
        uint group = threadgroup_position_in_grid.y;
        uint groups = (output_rows + RowsPerThreadgroup - 1u) / RowsPerThreadgroup;
        partial_scores[row * groups + group] = best_score;
        partial_ids[row * groups + group] = best_id;
    }
)";

static const char* Q6_ARGMAX_MERGE_SOURCE = R"(
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup;
    uint groups = (uint)partial_scores_shape[1];
    float best_score = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint group = tid; group < groups; group += 256u) {
        uint index = row * groups + group;
        float score = partial_scores[index];
        uint id = partial_ids[index];
        if (ggml_argmax_better(score, id, best_score, best_id)) {
            best_score = score;
            best_id = id;
        }
    }

    threadgroup float scores[256];
    threadgroup uint ids[256];
    scores[tid] = best_score;
    ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float score = scores[tid + stride];
            uint id = ids[tid + stride];
            if (ggml_argmax_better(
                    score, id, scores[tid], ids[tid])) {
                scores[tid] = score;
                ids[tid] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        max_scores[row] = scores[0];
        max_ids[row] = ids[0];
    }
)";

static const char* GGML_QMM_METAL_SOURCE = R"(
    constexpr uint BM = (uint)BlockM;
    constexpr uint BN = 8u;
    constexpr uint BK = 256u;

    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup = simdgroup_index_in_threadgroup;
    uint output_row = threadgroup_position_in_grid.y * BN + simdgroup;
    uint input_base = threadgroup_position_in_grid.z * BM;
    uint range_count = (uint)row_ranges_shape[0];
    uint selected_rows = 0u;
    for (uint range = 0u; range < range_count; ++range) {
        selected_rows += row_ranges[range * 2u + 1u];
    }
    uint source_row =
        ggml_source_row(row_ranges, range_count, output_row);
    bool valid_output =
        output_row < selected_rows && source_row < (uint)packed_shape[0];
    uint row_base = valid_output
        ? source_row * (uint)packed_shape[1]
        : 0u;
    uint input_rows = (uint)x_shape[0];
    uint width = (uint)x_shape[1];
    threadgroup float staged_x[BM * BK];
    float sums[BM];
    for (uint row = 0u; row < BM; ++row) sums[row] = 0.0f;

    for (uint column_base = 0u; column_base < width; column_base += BK) {
        for (uint index = tid; index < BM * BK; index += 256u) {
            uint row = index / BK;
            uint column = index - row * BK;
            uint input_row = input_base + row;
            uint input_column = column_base + column;
            staged_x[index] =
                input_row < input_rows && input_column < width
                ? x[input_row * width + input_column]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid_output) {
            for (uint column = lane; column < BK; column += 32u) {
                uint input_column = column_base + column;
                if (input_column < width) {
                    float weight = ggml_decode_weight(
                        packed, row_base, input_column, QType, iq3_grid);
                    for (uint row = 0u; row < BM; ++row) {
                        sums[row] += staged_x[row * BK + column] * weight;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row = 0u; row < BM; ++row) {
        float total = simd_sum(sums[row]);
        uint input_row = input_base + row;
        if (lane == 0u && valid_output && input_row < input_rows) {
            out[input_row * selected_rows + output_row] = total;
        }
    }
)";

static const char* GGML_EMBEDDING_METAL_SOURCE = R"(
    uint lane = thread_index_in_simdgroup;
    uint selected = thread_position_in_grid.y;
    uint source_row = (uint)indices[selected];
    uint block_elements = ggml_block_elements(QType);
    uint block_bytes = ggml_block_bytes(QType);
    uint width = (uint)packed_shape[1] / block_bytes * block_elements;
    bool valid_row = source_row < (uint)packed_shape[0];
    uint row_base =
        valid_row ? source_row * (uint)packed_shape[1] : 0u;
    for (uint column = lane; column < width; column += 32u) {
        out[selected * width + column] = valid_row
            ? ggml_decode_weight(
                packed, row_base, column, QType, iq3_grid)
            : 0.0f;
    }
)";

struct GgmlKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> decode;
    std::optional<mlx::core::fast::CustomKernelFunction> verify;
    std::optional<mlx::core::fast::CustomKernelFunction> qmm;
    std::optional<mlx::core::fast::CustomKernelFunction> qwen38_q6_head_verify_r8;
    std::optional<mlx::core::fast::CustomKernelFunction> q6_argmax_partial;
    std::optional<mlx::core::fast::CustomKernelFunction> q6_argmax_merge;
    std::optional<mlx::core::fast::CustomKernelFunction> embedding;
    std::once_flag initialize_once;

    void initialize() {
        std::call_once(initialize_once, [this] {
            const std::vector<std::string> linear_inputs = {
                "x", "packed", "row_ranges", "iq3_grid"};
            decode = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_qmv_v3",
                linear_inputs,
                {"out"},
                GGML_QMV_METAL_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
            verify = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_verify_qmm_v2",
                linear_inputs,
                {"out"},
                GGML_VERIFY_QMM_METAL_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
            qmm = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_qmm_mx8_v3",
                linear_inputs,
                {"out"},
                GGML_QMM_METAL_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
            qwen38_q6_head_verify_r8 = mlx::core::fast::metal_kernel(
                "qw_qwen38_q6_head_verify_r8_v1",
                linear_inputs,
                {"out"},
                QWEN38_Q6_HEAD_VERIFY_R8_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
            q6_argmax_partial = mlx::core::fast::metal_kernel(
                "qw_ggml_q6_argmax_partial_v2",
                {"x", "packed", "iq3_grid"},
                {"partial_scores", "partial_ids"},
                Q6_ARGMAX_PARTIAL_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
            q6_argmax_merge = mlx::core::fast::metal_kernel(
                "qw_ggml_q6_argmax_merge_v2",
                {"partial_scores", "partial_ids"},
                {"max_scores", "max_ids"},
                Q6_ARGMAX_MERGE_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
            embedding = mlx::core::fast::metal_kernel(
                "qw_ggml_packed_embedding_v2",
                {"packed", "indices", "iq3_grid"},
                {"out"},
                GGML_EMBEDDING_METAL_SOURCE,
                GGML_DECODE_METAL_HEADER,
                false);
        });
    }
};

GgmlKernelHolder& ggml_kernels() {
    static GgmlKernelHolder holder;
    holder.initialize();
    return holder;
}

size_t checked_packed_size(
    int32_t in_features,
    int32_t out_features,
    GgmlBlockInfo info
) {
    if (in_features <= 0 || out_features <= 0
        || in_features % info.elements != 0) {
        throw std::invalid_argument("packed GGML matrix shape is invalid");
    }
    const size_t blocks =
        static_cast<size_t>(in_features / info.elements);
    if (blocks > std::numeric_limits<size_t>::max()
            / static_cast<size_t>(info.bytes)) {
        throw std::invalid_argument("packed GGML row byte count overflow");
    }
    const size_t row_bytes =
        blocks * static_cast<size_t>(info.bytes);
    if (row_bytes > static_cast<size_t>(INT32_MAX)
        || static_cast<size_t>(out_features)
            > std::numeric_limits<size_t>::max() / row_bytes) {
        throw std::invalid_argument(
            "packed GGML matrix byte count overflow");
    }
    return row_bytes * static_cast<size_t>(out_features);
}

void validate_row_ranges(
    const mlx::core::array& row_ranges,
    int32_t selected_rows
) {
    const auto& shape = row_ranges.shape();
    if (row_ranges.dtype() != mlx::core::uint32
        || shape.size() != 2 || shape[0] <= 0 || shape[0] > 3
        || shape[1] != 2 || selected_rows <= 0) {
        throw std::invalid_argument(
            "packed GGML row ranges are invalid");
    }
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
qtype_template_argument(int32_t qtype) {
    return {{"QType", qtype}};
}

} // namespace

std::unique_ptr<MlxArray> ggml_packed_matmul(
    const MlxArray& x,
    const MlxArray& packed,
    const MlxArray& row_ranges,
    const MlxArray& iq3_grid,
    int32_t qtype,
    int32_t in_features,
    int32_t out_features,
    int32_t selected_rows,
    int32_t input_rows,
    bool qwen38_q6_head_verify_r8
) {
#ifndef __APPLE__
    throw std::invalid_argument("packed GGML execution requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("packed GGML execution requires Metal");
    }
    const GgmlBlockInfo info = ggml_block_info(qtype);
    const size_t expected =
        checked_packed_size(in_features, out_features, info);
    const int32_t row_bytes =
        in_features / info.elements * info.bytes;
    if (packed.inner.dtype() != uint8
        || packed.inner.size() != expected
        || packed.inner.shape() != Shape{out_features, row_bytes}) {
        throw std::invalid_argument(
            "packed GGML buffer shape, dtype, or bounds mismatch");
    }
    validate_iq3_table(iq3_grid.inner, qtype);
    validate_row_ranges(row_ranges.inner, selected_rows);
    const auto& shape = x.inner.shape();
    if (shape.empty() || shape.back() != in_features || input_rows <= 0
        || x.inner.size()
            != static_cast<size_t>(input_rows)
                * static_cast<size_t>(in_features)) {
        throw std::invalid_argument(
            "packed GGML activation shape mismatch");
    }
    const auto args = qtype_template_argument(qtype);
    auto input =
        contiguous(reshape(astype(x.inner, float32), {input_rows, in_features}));
    auto& holder = ggml_kernels();
    std::vector<array> results;
    const bool qwen38_q6_head_shape =
        qtype == 14 && in_features == 5120 && out_features == 248320
        && selected_rows == 80922;
    if (qwen38_q6_head_verify_r8 && !qwen38_q6_head_shape) {
        throw std::invalid_argument(
            "pinned Qwen3.8 Q6 output-head descriptor is invalid");
    }
    if (input_rows == 1) {
        auto decode_args = args;
        decode_args.push_back({"Width", in_features});
        decode_args.push_back({"RowBytes", row_bytes});
        results = (*holder.decode)(
            {input, packed.inner, row_ranges.inner, iq3_grid.inner},
            {Shape{1, selected_rows}},
            {float32},
            std::make_tuple(32, selected_rows, 1),
            std::make_tuple(32, 1, 1),
            decode_args,
            std::nullopt,
            false,
            {});
    } else if (qwen38_q6_head_verify_r8
            && (input_rows == 3 || input_rows == 4)) {
        results = (*holder.qwen38_q6_head_verify_r8)(
            {input, packed.inner, row_ranges.inner, iq3_grid.inner},
            {Shape{input_rows, selected_rows}},
            {float32},
            std::make_tuple(
                256,
                (selected_rows + 7) / 8,
                1),
            std::make_tuple(256, 1, 1),
            args,
            std::nullopt,
            false,
            {});
    } else if (input_rows <= 4) {
        results = (*holder.verify)(
            {input, packed.inner, row_ranges.inner, iq3_grid.inner},
            {Shape{input_rows, selected_rows}},
            {float32},
            std::make_tuple(32, selected_rows, 1),
            std::make_tuple(32, 1, 1),
            args,
            std::nullopt,
            false,
            {});
    } else {
        const int32_t block_m = input_rows <= 512 ? 16 : 32;
        const int32_t n_tiles = (selected_rows + 7) / 8;
        const int32_t m_tiles = (input_rows + block_m - 1) / block_m;
        auto qmm_args = args;
        qmm_args.push_back({"BlockM", block_m});
        results = (*holder.qmm)(
            {input, packed.inner, row_ranges.inner, iq3_grid.inner},
            {Shape{input_rows, selected_rows}},
            {float32},
            std::make_tuple(256, n_tiles, m_tiles),
            std::make_tuple(256, 1, 1),
            qmm_args,
            std::nullopt,
            false,
            {});
    }
    Shape output_shape(shape.begin(), shape.end() - 1);
    output_shape.push_back(selected_rows);
    return std::make_unique<MlxArray>(
        reshape(results[0], output_shape));
#endif
}

std::unique_ptr<Qwen38Q6HeadArgmaxOutputs> ggml_q6_head_argmax(
    const MlxArray& x,
    const MlxArray& packed,
    const MlxArray& iq3_grid,
    int32_t in_features,
    int32_t out_features,
    int32_t input_rows
) {
#ifndef __APPLE__
    throw std::invalid_argument("packed GGML Q6 argmax requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("packed GGML Q6 argmax requires Metal");
    }
    constexpr int32_t qtype = 14;
    if (input_rows != 3 && input_rows != 4) {
        throw std::invalid_argument("packed GGML Q6 argmax requires M3 or M4");
    }
    const GgmlBlockInfo info = ggml_block_info(qtype);
    const size_t expected =
        checked_packed_size(in_features, out_features, info);
    const int32_t row_bytes =
        in_features / info.elements * info.bytes;
    if (packed.inner.dtype() != uint8
        || packed.inner.size() != expected
        || packed.inner.shape() != Shape{out_features, row_bytes}) {
        throw std::invalid_argument(
            "packed GGML Q6 argmax buffer shape, dtype, or bounds mismatch");
    }
    validate_iq3_table(iq3_grid.inner, qtype);
    const auto& shape = x.inner.shape();
    if (shape.empty() || shape.back() != in_features
        || x.inner.size()
            != static_cast<size_t>(input_rows)
                * static_cast<size_t>(in_features)) {
        throw std::invalid_argument(
            "packed GGML Q6 argmax activation shape mismatch");
    }

    // The custom kernel flat-indexes every input. Contiguous elides the data
    // copy for canonical runtime arrays and materializes public strided views.
    auto input =
        reshape(contiguous(astype(x.inner, float32)), {input_rows, in_features});
    auto packed_input = contiguous(packed.inner);
    auto table_input = contiguous(iq3_grid.inner);
    auto& holder = ggml_kernels();
    const int32_t groups = (out_features + 7) / 8;
    const auto args = qtype_template_argument(qtype);
    auto partials = (*holder.q6_argmax_partial)(
        {input, packed_input, table_input},
        {Shape{input_rows, groups}, Shape{input_rows, groups}},
        {float32, uint32},
        std::make_tuple(256, groups, 1),
        std::make_tuple(256, 1, 1),
        args,
        std::nullopt,
        false,
        {});
    auto merged = (*holder.q6_argmax_merge)(
        {partials[0], partials[1]},
        {Shape{input_rows}, Shape{input_rows}},
        {float32, uint32},
        std::make_tuple(256, input_rows, 1),
        std::make_tuple(256, 1, 1),
        {},
        std::nullopt,
        false,
        {});
    return std::make_unique<Qwen38Q6HeadArgmaxOutputs>(
        Qwen38Q6HeadArgmaxOutputs{
            std::make_unique<MlxArray>(std::move(merged[0])),
            std::make_unique<MlxArray>(std::move(merged[1]))});
#endif
}

std::unique_ptr<MlxArray> qwen38_q6_head_argmax_take_scores(
    Qwen38Q6HeadArgmaxOutputs& outputs
) {
    return std::move(outputs.scores);
}

std::unique_ptr<MlxArray> qwen38_q6_head_argmax_take_ids(
    Qwen38Q6HeadArgmaxOutputs& outputs
) {
    return std::move(outputs.ids);
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
    const size_t expected =
        checked_packed_size(embedding_dim, vocab_size, info);
    const int32_t row_bytes =
        embedding_dim / info.elements * info.bytes;
    if (packed.inner.dtype() != uint8
        || packed.inner.size() != expected
        || packed.inner.shape() != Shape{vocab_size, row_bytes}) {
        throw std::invalid_argument(
            "packed GGML embedding buffer shape, dtype, or bounds mismatch");
    }
    validate_iq3_table(iq3_grid.inner, qtype);
    if (indices.inner.size() == 0
        || indices.inner.size() > static_cast<size_t>(INT32_MAX)) {
        throw std::invalid_argument(
            "packed GGML embedding selection grid is invalid");
    }
    const int32_t selected_rows =
        static_cast<int32_t>(indices.inner.size());
    auto index_data = contiguous(astype(indices.inner, uint32));
    auto results = (*ggml_kernels().embedding)(
        {packed.inner, index_data, iq3_grid.inner},
        {Shape{selected_rows, embedding_dim}},
        {float32},
        std::make_tuple(32, selected_rows, 1),
        std::make_tuple(32, 1, 1),
        qtype_template_argument(qtype),
        std::nullopt,
        false,
        {});
    Shape output_shape(
        indices.inner.shape().begin(), indices.inner.shape().end());
    output_shape.push_back(embedding_dim);
    return std::make_unique<MlxArray>(
        reshape(results[0], output_shape));
#endif
}

} // namespace mlx_cxx
