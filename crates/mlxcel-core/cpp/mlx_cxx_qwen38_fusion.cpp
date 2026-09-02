// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");

// Fixed-artifact affine fusion for the pinned Qwen3.8 target.

#include "mlx_cxx_internal.h"
#include "qwen38_quantized_metal.h"

#include <limits>

namespace mlx_cxx {
namespace {

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::fast::TemplateArg;


static const char* QWEN38_QMM_FUSION_HEADER = R"(
template <int Bits, bool AlignedN>
METAL_FUNC void qwen38_qmm_accumulate(
    const device uint32_t* w,
    const device float* scales,
    const device float* biases,
    const device float* x,
    int K,
    int N,
    int M,
    int K_eff,
    uint3 tid,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup float* Xs,
    threadgroup float* Ws,
    thread mlx::steel::BlockMMA<
        float, float, 32, 32, 32, 2, 2, false, true, 36, 36>& mma_op
) {
    constexpr int pack_factor = get_pack_factor<Bits, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<Bits>();
    using loader_x_t = mlx::steel::BlockLoader<float, 32, 32, 36, 1, 128>;
    using loader_w_t = QuantizedBlockLoader<
        float, 32, 32, 36, 1, 128, 32, Bits>;

    const int K_w = K * bytes_per_pack / pack_factor;
    const int K_g = K / 32;
    const int y_row = tid.y * 32;
    const int y_col = tid.x * 32;
    auto wl = reinterpret_cast<const device uint8_t*>(w);
    x += y_row * static_cast<int64_t>(K);
    wl += y_col * K_w;
    scales += y_col * K_g;
    biases += y_col * K_g;

    const short num_els = min(32, M - y_row);
    const short num_outs = min(32, N - y_col);
    loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
    loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

    if (num_els < 32) {
        if (!AlignedN && num_outs < 32) {
            for (int k = 0; k < K_eff; k += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                loader_x.load_safe(short2(32, num_els));
                loader_w.load_safe(short2(32, num_outs));
                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma_op.mma(Xs, Ws);
                loader_x.next();
                loader_w.next();
            }
        } else {
            for (int k = 0; k < K_eff; k += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                loader_x.load_safe(short2(32, num_els));
                loader_w.load_unsafe();
                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma_op.mma(Xs, Ws);
                loader_x.next();
                loader_w.next();
            }
        }
    } else if (!AlignedN && num_outs < 32) {
        for (int k = 0; k < K_eff; k += 32) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_x.load_unsafe();
            loader_w.load_safe(short2(32, num_outs));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(Xs, Ws);
            loader_x.next();
            loader_w.next();
        }
    } else {
        for (int k = 0; k < K_eff; k += 32) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_x.load_unsafe();
            loader_w.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(Xs, Ws);
            loader_x.next();
            loader_w.next();
        }
    }
}

template <int Bits, bool AlignedN>
METAL_FUNC void qwen38_qmm_accumulate_paired_rows(
    const device uint32_t* w,
    const device float* scales,
    const device float* biases,
    const device float* x,
    int K,
    int N,
    int M,
    int K_eff,
    uint3 tid,
    ushort row_team,
    ushort local_gid,
    ushort simd_lid,
    threadgroup float* Xs,
    threadgroup float* Ws,
    thread mlx::steel::BlockMMA<
        float, float, 32, 32, 32, 2, 2, false, true, 36, 36>& mma_op
) {
    constexpr int pack_factor = get_pack_factor<Bits, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<Bits>();
    using loader_x_t = mlx::steel::BlockLoader<float, 32, 32, 36, 1, 128>;
    using loader_w_t = QuantizedBlockLoader<
        float, 32, 32, 36, 1, 128, 32, Bits>;

    const int K_w = K * bytes_per_pack / pack_factor;
    const int K_g = K / 32;
    const int y_row = tid.y * 32;
    const int y_col = tid.x * 32;
    const short num_els = y_row < M ? min(32, M - y_row) : 0;
    const short num_outs = min(32, N - y_col);
    auto wl = reinterpret_cast<const device uint8_t*>(w);
    x += (num_els > 0 ? y_row : 0) * static_cast<int64_t>(K);
    wl += y_col * K_w;
    scales += y_col * K_g;
    biases += y_col * K_g;

    loader_x_t loader_x(x, K, Xs, local_gid, simd_lid);
    loader_w_t loader_w(wl, scales, biases, K, Ws, local_gid, simd_lid);
    for (int k = 0; k < K_eff; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (num_els == 32) {
            loader_x.load_unsafe();
        } else if (num_els > 0) {
            loader_x.load_safe(short2(32, num_els));
        }
        if (row_team == 0) {
            if (!AlignedN && num_outs < 32) {
                loader_w.load_safe(short2(32, num_outs));
            } else {
                loader_w.load_unsafe();
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (num_els > 0) {
            mma_op.mma(Xs, Ws);
            loader_x.next();
        }
        if (row_team == 0) {
            loader_w.next();
        }
    }
}

template <typename Mma>
METAL_FUNC void qwen38_store_tile(
    thread Mma& mma,
    threadgroup float* tile
) {
    mma.Ctile.template store<float, 2, 2, 32, 1>(
        tile + mma.sm * 32 + mma.sn);
}

METAL_FUNC float qwen38_sigmoid(float x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}

METAL_FUNC void qwen38_reduce_small(
    const device float* in,
    device float* out,
    int split,
    int stride,
    int block,
    ushort tx,
    ushort ty,
    threadgroup float* scratch
) {
    const int ysize = min(8, split);
    float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int column = block * 128 + tx * 4;
    if (ty < ysize && column < stride) {
        for (int row = ty; row < split; row += ysize) {
            const device float* values = in + row * stride + column;
            for (int i = 0; i < 4; ++i) {
                float value = column + i < stride ? values[i] : 0.0f;
                totals[i] = value + totals[i];
            }
        }
        for (int i = 0; i < 4; ++i) {
            scratch[ty * 128 + tx * 4 + i] = totals[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty == 0 && column < stride) {
        for (int i = 0; i < 4; ++i) {
            totals[i] = scratch[tx * 4 + i];
        }
        for (int row = 1; row < ysize; ++row) {
            for (int i = 0; i < 4; ++i) {
                totals[i] = scratch[row * 128 + tx * 4 + i] + totals[i];
            }
        }
        for (int i = 0; i < 4 && column + i < stride; ++i) {
            out[column + i] = totals[i];
        }
    }
}

METAL_FUNC void qwen38_reduce_looped(
    const device float* in,
    device float* out,
    int split,
    int stride,
    int block,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup float* scratch
) {
    constexpr int n_reads = 4;
    const int lid = simd_gid * 32 + simd_lid;
    const int offset_x = (lid % 8) * n_reads;
    const int offset_y = lid / 8;
    const int column = block * 32 + offset_x;
    float totals[n_reads] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int row = offset_y; row < split; row += 32) {
        const device float* values = in + row * stride + column;
        for (int i = 0; i < n_reads; ++i) {
            float value = column + i < stride ? values[i] : 0.0f;
            totals[i] = value + totals[i];
        }
    }
    for (int i = 0; i < n_reads; ++i) {
        scratch[offset_y * 32 + offset_x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int out_x = simd_gid * n_reads;
    for (int i = 0; i < n_reads; ++i) {
        totals[i] = simd_sum(scratch[simd_lid * 32 + out_x + i]);
    }
    if (simd_lid == 0) {
        const int out_column = block * 32 + out_x;
        for (int i = 0; i < n_reads && out_column + i < stride; ++i) {
            out[out_column + i] = totals[i];
        }
    }
}
)";

static const char* QWEN38_MLP_GATE_UP_SOURCE = R"(
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int K = 5120;
    constexpr int N = 17408;
    threadgroup float Xs[BM * 36];
    threadgroup float Ws[BN * 36];
    threadgroup float up_values[BM * BN];
    uint3 tid(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, 0);

    mlx::steel::BlockMMA<
        float, float, BM, BN, 32, 2, 2, false, true, 36, 36> gate_mma(
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    qwen38_qmm_accumulate<GateBits, true>(
        gate_w, gate_s, gate_b, x, K, N, MRows, K, tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
        Xs, Ws, gate_mma);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int y_row = threadgroup_position_in_grid.y * BM;
    const int y_col = threadgroup_position_in_grid.x * BN;
    const short num_rows = min(BM, MRows - y_row);
    device float* gate_out = out + y_row * N + y_col;
    if (num_rows < BM) {
        gate_mma.store_result_safe(gate_out, N, short2(BN, num_rows));
    } else {
        gate_mma.store_result(gate_out, N);
    }
    threadgroup_barrier(mem_flags::mem_device);

    mlx::steel::BlockMMA<
        float, float, BM, BN, 32, 2, 2, false, true, 36, 36> up_mma(
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    qwen38_qmm_accumulate<UpBits, true>(
        up_w, up_s, up_b, x, K, N, MRows, K, tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
        Xs, Ws, up_mma);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qwen38_store_tile(up_mma, up_values);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int linear_lid = thread_index_in_threadgroup;
    for (int index = linear_lid; index < BM * BN; index += 128) {
        const int row = index / BN;
        const int col = index % BN;
        if (y_row + row < MRows) {
            const int output_index = (y_row + row) * N + y_col + col;
            float gate = out[output_index];
            float up = up_values[index];
            out[output_index] = (gate * qwen38_sigmoid(gate)) * up;
        }
    }
)";

static const char* QWEN38_MLP_DOWN_SOURCE = R"(
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int K = 17408;
    constexpr int N = 5120;
    threadgroup float Xs[SplitK * BM * 36];
    threadgroup float Ws[SplitK * BN * 36];
    threadgroup float tiles[SplitK * BM * BN];
    const ushort team = simdgroup_index_in_threadgroup / 4;
    const ushort local_gid = simdgroup_index_in_threadgroup % 4;
    const int partition = K / SplitK;
    const int k_start = team * partition;
    constexpr int pack_factor = get_pack_factor<Bits, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<Bits>();
    const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(weight);
    wb += k_start * bytes_per_pack / pack_factor;
    mlx::steel::BlockMMA<
        float, float, BM, BN, 32, 2, 2, false, true, 36, 36> mma(
            local_gid, thread_index_in_simdgroup);
    uint3 qtid(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, team);
    qwen38_qmm_accumulate<Bits, true>(
        reinterpret_cast<const device uint32_t*>(wb),
        scales + k_start / 32,
        biases + k_start / 32,
        x + k_start,
        K, N, MRows, partition, qtid,
        local_gid, thread_index_in_simdgroup,
        Xs + team * BM * 36, Ws + team * BN * 36, mma);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (SplitK == 1) {
        const int y_row = threadgroup_position_in_grid.y * BM;
        const int y_col = threadgroup_position_in_grid.x * BN;
        const short num_rows = min(BM, MRows - y_row);
        const short num_cols = min(BN, N - y_col);
        device float* dst = out + y_row * N + y_col;
        if (num_rows < BM || num_cols < BN) {
            mma.store_result_safe(dst, N, short2(num_cols, num_rows));
        } else {
            mma.store_result(dst, N);
        }
    } else {
        qwen38_store_tile(mma, tiles + team * BM * BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int row_base = threadgroup_position_in_grid.y * BM;
        const int col_base = threadgroup_position_in_grid.x * BN;
        const int linear_lid = thread_index_in_threadgroup;
        for (int index = linear_lid; index < BM * BN; index += 256) {
            const int row = index / BN;
            const int col = index % BN;
            if (row_base + row < MRows && col_base + col < N) {
                out[(row_base + row) * N + col_base + col] =
                    tiles[index] + tiles[BM * BN + index];
            }
        }
    }
)";

static const char* QWEN38_MLP_GATE_UP_PAIRED_SOURCE = R"(
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int K = 5120;
    constexpr int N = 17408;
    threadgroup float Xs[2 * BM * 36];
    threadgroup float Ws[BN * 36];
    threadgroup float up_values[2 * BM * BN];
    const ushort row_team = simdgroup_index_in_threadgroup / 4;
    const ushort local_gid = simdgroup_index_in_threadgroup % 4;
    const int mtile = threadgroup_position_in_grid.y * 2 + row_team;
    const int y_row = mtile * BM;
    const int y_col = threadgroup_position_in_grid.x * BN;
    const bool valid = y_row < MRows;
    uint3 tid(threadgroup_position_in_grid.x, mtile, 0);

    mlx::steel::BlockMMA<
        float, float, BM, BN, 32, 2, 2, false, true, 36, 36> gate_mma(
            local_gid, thread_index_in_simdgroup);
    qwen38_qmm_accumulate_paired_rows<GateBits, true>(
        gate_w, gate_s, gate_b, x, K, N, MRows, K, tid,
        row_team, local_gid, thread_index_in_simdgroup,
        Xs + row_team * BM * 36, Ws, gate_mma);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        const short num_rows = min(BM, MRows - y_row);
        device float* gate_out = out + y_row * N + y_col;
        if (num_rows < BM) {
            gate_mma.store_result_safe(gate_out, N, short2(BN, num_rows));
        } else {
            gate_mma.store_result(gate_out, N);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    mlx::steel::BlockMMA<
        float, float, BM, BN, 32, 2, 2, false, true, 36, 36> up_mma(
            local_gid, thread_index_in_simdgroup);
    qwen38_qmm_accumulate_paired_rows<UpBits, true>(
        up_w, up_s, up_b, x, K, N, MRows, K, tid,
        row_team, local_gid, thread_index_in_simdgroup,
        Xs + row_team * BM * 36, Ws, up_mma);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qwen38_store_tile(up_mma, up_values + row_team * BM * BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int linear_lid = thread_index_in_threadgroup;
    for (int index = linear_lid; index < 2 * BM * BN; index += 256) {
        const int output_team = index / (BM * BN);
        const int local_index = index % (BM * BN);
        const int row = local_index / BN;
        const int col = local_index % BN;
        const int output_row = (threadgroup_position_in_grid.y * 2 + output_team) * BM + row;
        if (output_row < MRows) {
            const int output_index = output_row * N + y_col + col;
            float gate = out[output_index];
            float up = up_values[index];
            out[output_index] = (gate * qwen38_sigmoid(gate)) * up;
        }
    }
)";

static const char* QWEN38_MLP_DOWN_PAIRED_SOURCE = R"(
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int K = 17408;
    constexpr int N = 5120;
    threadgroup float Xs[2 * BM * 36];
    threadgroup float Ws[BN * 36];
    const ushort row_team = simdgroup_index_in_threadgroup / 4;
    const ushort local_gid = simdgroup_index_in_threadgroup % 4;
    const int mtile = threadgroup_position_in_grid.y * 2 + row_team;
    const int y_row = mtile * BM;
    const int y_col = threadgroup_position_in_grid.x * BN;
    const bool valid = y_row < MRows;
    uint3 qtid(threadgroup_position_in_grid.x, mtile, 0);
    mlx::steel::BlockMMA<
        float, float, BM, BN, 32, 2, 2, false, true, 36, 36> mma(
            local_gid, thread_index_in_simdgroup);
    qwen38_qmm_accumulate_paired_rows<Bits, true>(
        weight, scales, biases, x, K, N, MRows, K, qtid,
        row_team, local_gid, thread_index_in_simdgroup,
        Xs + row_team * BM * 36, Ws, mma);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        const short num_rows = min(BM, MRows - y_row);
        device float* dst = out + y_row * N + y_col;
        if (num_rows < BM) {
            mma.store_result_safe(dst, N, short2(BN, num_rows));
        } else {
            mma.store_result(dst, N);
        }
    }
)";

static const char* QWEN38_GDN_OLD_QZ_SOURCE = R"(
    constexpr int K = 5120;
    constexpr int Nq = 10240;
    constexpr int Nz = 6144;
    constexpr int Tq = Nq / 32;
    constexpr int Tz = Nz / 32;
    constexpr int Wq = Tq;
    threadgroup float Xs[32 * 36];
    threadgroup float Ws[32 * 36];
    const int work = threadgroup_position_in_grid.x;
    const int mtile = threadgroup_position_in_grid.y;
    const bool is_qkv = work < Wq;
    const int local = is_qkv ? work : work - Wq;
    const int split = is_qkv ? 1 : SplitZ;
    const int n = is_qkv ? Nq : Nz;
    const int tiles = is_qkv ? Tq : Tz;
    const int part = local / tiles;
    const int tile = local % tiles;
    const int partition = K / split;
    const int k_start = part * partition;
    const device uint32_t* w = is_qkv ? qkv_w : z_w;
    const device float* s = is_qkv ? qkv_s : z_s;
    const device float* b = is_qkv ? qkv_b : z_b;
    device float* y = is_qkv ? qkv_part : z_part;
    uint3 qtid(tile, mtile, part);
    mlx::steel::BlockMMA<
        float, float, 32, 32, 32, 2, 2, false, true, 36, 36> mma(
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
#define QWEN38_GDN_QZ_ACCUM(BITS) \
    constexpr int pf = get_pack_factor<BITS, 8>(); \
    constexpr int bp = get_bytes_per_pack<BITS>(); \
    const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w); \
    wb += k_start * bp / pf; \
    qwen38_qmm_accumulate<BITS, true>( \
        reinterpret_cast<const device uint32_t*>(wb), \
        s + k_start / 32, b + k_start / 32, x + k_start, \
        K, n, MRows, partition, qtid, \
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup, \
        Xs, Ws, mma)
    if (is_qkv) {
        QWEN38_GDN_QZ_ACCUM(QkvBits);
    } else {
        QWEN38_GDN_QZ_ACCUM(ZBits);
    }
#undef QWEN38_GDN_QZ_ACCUM
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int y_row = mtile * 32;
    const int y_col = tile * 32;
    const short num_rows = min(32, MRows - y_row);
    device float* dst = y + part * MRows * n + y_row * n + y_col;
    if (num_rows < 32) {
        mma.store_result_safe(dst, n, short2(32, num_rows));
    } else {
        mma.store_result(dst, n);
    }
)";

static const char* QWEN38_GDN_OLD_BA_SOURCE = R"(
    constexpr int K = 5120;
    constexpr int N = 48;
    constexpr int Tiles = 2;
    constexpr int Wb = Tiles * SplitB;
    threadgroup float Xs[32 * 36];
    threadgroup float Ws[32 * 36];
    const int work = threadgroup_position_in_grid.x;
    const int mtile = threadgroup_position_in_grid.y;
    const bool is_beta = work < Wb;
    const int local = is_beta ? work : work - Wb;
    const int split = is_beta ? SplitB : SplitA;
    const int part = local / Tiles;
    const int tile = local % Tiles;
    const int partition = K / split;
    const int k_start = part * partition;
    const device uint32_t* w = is_beta ? beta_w : alpha_w;
    const device float* s = is_beta ? beta_s : alpha_s;
    const device float* b = is_beta ? beta_b : alpha_b;
    device float* y = is_beta ? beta_part : alpha_part;
    uint3 qtid(tile, mtile, part);
    mlx::steel::BlockMMA<
        float, float, 32, 32, 32, 2, 2, false, true, 36, 36> mma(
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
#define QWEN38_GDN_BA_ACCUM(BITS) \
    constexpr int pf = get_pack_factor<BITS, 8>(); \
    constexpr int bp = get_bytes_per_pack<BITS>(); \
    const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w); \
    wb += k_start * bp / pf; \
    qwen38_qmm_accumulate<BITS, false>( \
        reinterpret_cast<const device uint32_t*>(wb), \
        s + k_start / 32, b + k_start / 32, x + k_start, \
        K, N, MRows, partition, qtid, \
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup, \
        Xs, Ws, mma)
    if (is_beta) {
        QWEN38_GDN_BA_ACCUM(BetaBits);
    } else {
        QWEN38_GDN_BA_ACCUM(AlphaBits);
    }
#undef QWEN38_GDN_BA_ACCUM
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int y_row = mtile * 32;
    const int y_col = tile * 32;
    const short num_rows = min(32, MRows - y_row);
    const short num_cols = min(32, N - y_col);
    device float* dst = y + part * MRows * N + y_row * N + y_col;
    if (num_rows < 32 || num_cols < 32) {
        mma.store_result_safe(dst, N, short2(num_cols, num_rows));
    } else {
        mma.store_result(dst, N);
    }
)";

static const char* QWEN38_GDN_PAIRED_QZ_SOURCE = R"(
    constexpr int K = 5120;
    constexpr int Nq = 10240;
    constexpr int Nz = 6144;
    constexpr int Tq = Nq / 32;
    constexpr int Tz = Nz / 32;
    constexpr int Wq = Tq;
    threadgroup float Xs[2 * 32 * 36];
    threadgroup float Ws[32 * 36];
    const ushort row_team = simdgroup_index_in_threadgroup / 4;
    const ushort local_gid = simdgroup_index_in_threadgroup % 4;
    const int work = threadgroup_position_in_grid.x;
    const int mtile = threadgroup_position_in_grid.y * 2 + row_team;
    const bool is_qkv = work < Wq;
    const int local = is_qkv ? work : work - Wq;
    const int split = is_qkv ? 1 : SplitZ;
    const int n = is_qkv ? Nq : Nz;
    const int tiles = is_qkv ? Tq : Tz;
    const int part = local / tiles;
    const int tile = local % tiles;
    const int partition = K / split;
    const int k_start = part * partition;
    const device uint32_t* w = is_qkv ? qkv_w : z_w;
    const device float* s = is_qkv ? qkv_s : z_s;
    const device float* b = is_qkv ? qkv_b : z_b;
    device float* y = is_qkv ? qkv_part : z_part;
    uint3 qtid(tile, mtile, part);
    mlx::steel::BlockMMA<
        float, float, 32, 32, 32, 2, 2, false, true, 36, 36> mma(
            local_gid, thread_index_in_simdgroup);
#define QWEN38_GDN_PAIRED_QZ_ACCUM(BITS) \
    constexpr int pf = get_pack_factor<BITS, 8>(); \
    constexpr int bp = get_bytes_per_pack<BITS>(); \
    const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w); \
    wb += k_start * bp / pf; \
    qwen38_qmm_accumulate_paired_rows<BITS, true>( \
        reinterpret_cast<const device uint32_t*>(wb), \
        s + k_start / 32, b + k_start / 32, x + k_start, \
        K, n, MRows, partition, qtid, row_team, local_gid, \
        thread_index_in_simdgroup, Xs + row_team * 32 * 36, Ws, mma)
    if (is_qkv) {
        QWEN38_GDN_PAIRED_QZ_ACCUM(QkvBits);
    } else {
        QWEN38_GDN_PAIRED_QZ_ACCUM(ZBits);
    }
#undef QWEN38_GDN_PAIRED_QZ_ACCUM
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int y_row = mtile * 32;
    if (y_row < MRows) {
        const int y_col = tile * 32;
        const short num_rows = min(32, MRows - y_row);
        device float* dst = y + part * MRows * n + y_row * n + y_col;
        if (num_rows < 32) {
            mma.store_result_safe(dst, n, short2(32, num_rows));
        } else {
            mma.store_result(dst, n);
        }
    }
)";

static const char* QWEN38_GDN_PAIRED_BA_SOURCE = R"(
    constexpr int K = 5120;
    constexpr int N = 48;
    constexpr int Tiles = 2;
    constexpr int Wb = Tiles * SplitB;
    threadgroup float Xs[2 * 32 * 36];
    threadgroup float Ws[32 * 36];
    const ushort row_team = simdgroup_index_in_threadgroup / 4;
    const ushort local_gid = simdgroup_index_in_threadgroup % 4;
    const int work = threadgroup_position_in_grid.x;
    const int mtile = threadgroup_position_in_grid.y * 2 + row_team;
    const bool is_beta = work < Wb;
    const int local = is_beta ? work : work - Wb;
    const int split = is_beta ? SplitB : SplitA;
    const int part = local / Tiles;
    const int tile = local % Tiles;
    const int partition = K / split;
    const int k_start = part * partition;
    const device uint32_t* w = is_beta ? beta_w : alpha_w;
    const device float* s = is_beta ? beta_s : alpha_s;
    const device float* b = is_beta ? beta_b : alpha_b;
    device float* y = is_beta ? beta_part : alpha_part;
    uint3 qtid(tile, mtile, part);
    mlx::steel::BlockMMA<
        float, float, 32, 32, 32, 2, 2, false, true, 36, 36> mma(
            local_gid, thread_index_in_simdgroup);
#define QWEN38_GDN_PAIRED_BA_ACCUM(BITS) \
    constexpr int pf = get_pack_factor<BITS, 8>(); \
    constexpr int bp = get_bytes_per_pack<BITS>(); \
    const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w); \
    wb += k_start * bp / pf; \
    qwen38_qmm_accumulate_paired_rows<BITS, false>( \
        reinterpret_cast<const device uint32_t*>(wb), \
        s + k_start / 32, b + k_start / 32, x + k_start, \
        K, N, MRows, partition, qtid, row_team, local_gid, \
        thread_index_in_simdgroup, Xs + row_team * 32 * 36, Ws, mma)
    if (is_beta) {
        QWEN38_GDN_PAIRED_BA_ACCUM(BetaBits);
    } else {
        QWEN38_GDN_PAIRED_BA_ACCUM(AlphaBits);
    }
#undef QWEN38_GDN_PAIRED_BA_ACCUM
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int y_row = mtile * 32;
    if (y_row < MRows) {
        const int y_col = tile * 32;
        const short num_rows = min(32, MRows - y_row);
        const short num_cols = min(32, N - y_col);
        device float* dst = y + part * MRows * N + y_row * N + y_col;
        if (num_rows < 32 || num_cols < 32) {
            mma.store_result_safe(dst, N, short2(num_cols, num_rows));
        } else {
            mma.store_result(dst, N);
        }
    }
)";

static const char* QWEN38_GDN_REDUCE_ZBA_SOURCE = R"(
    constexpr int Nz = 6144;
    constexpr int Ns = 48;
    constexpr int ZStride = MRows * Nz;
    constexpr int SStride = MRows * Ns;
    constexpr int ZBlocks = (ZStride + 127) / 128;
    constexpr int BBlockWidth = SplitB < 32 ? 128 : 32;
    constexpr int BBlocks = (SStride + BBlockWidth - 1) / BBlockWidth;
    threadgroup float scratch[1024];
    int work = threadgroup_position_in_grid.x;
    if (work < ZBlocks) {
        qwen38_reduce_small(
            z_part, z_out, SplitZ, ZStride, work,
            thread_position_in_threadgroup.x,
            thread_position_in_threadgroup.y, scratch);
    } else if (work < ZBlocks + BBlocks) {
        int block = work - ZBlocks;
        if constexpr (SplitB < 32) {
            qwen38_reduce_small(
                beta_part, beta_out, SplitB, SStride, block,
                thread_position_in_threadgroup.x,
                thread_position_in_threadgroup.y, scratch);
        } else {
            qwen38_reduce_looped(
                beta_part, beta_out, SplitB, SStride, block,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, scratch);
        }
    } else {
        int block = work - ZBlocks - BBlocks;
        if constexpr (SplitA < 32) {
            qwen38_reduce_small(
                alpha_part, alpha_out, SplitA, SStride, block,
                thread_position_in_threadgroup.x,
                thread_position_in_threadgroup.y, scratch);
        } else {
            qwen38_reduce_looped(
                alpha_part, alpha_out, SplitA, SStride, block,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, scratch);
        }
    }
)";

static const char* QWEN38_GDN_REDUCE_BA_SOURCE = R"(
    constexpr int Ns = 48;
    constexpr int Stride = MRows * Ns;
    constexpr int BlockWidth = SplitB < 32 ? 128 : 32;
    constexpr int Blocks = (Stride + BlockWidth - 1) / BlockWidth;
    threadgroup float scratch[1024];
    int work = threadgroup_position_in_grid.x;
    if (work < Blocks) {
        if constexpr (SplitB < 32) {
            qwen38_reduce_small(
                beta_part, beta_out, SplitB, Stride, work,
                thread_position_in_threadgroup.x,
                thread_position_in_threadgroup.y, scratch);
        } else {
            qwen38_reduce_looped(
                beta_part, beta_out, SplitB, Stride, work,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, scratch);
        }
    } else {
        int block = work - Blocks;
        if constexpr (SplitA < 32) {
            qwen38_reduce_small(
                alpha_part, alpha_out, SplitA, Stride, block,
                thread_position_in_threadgroup.x,
                thread_position_in_threadgroup.y, scratch);
        } else {
            qwen38_reduce_looped(
                alpha_part, alpha_out, SplitA, Stride, block,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, scratch);
        }
    }
)";

struct Qwen38FusionKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> mlp_gate_up;
    std::optional<mlx::core::fast::CustomKernelFunction> mlp_down;
    std::optional<mlx::core::fast::CustomKernelFunction> mlp_gate_up_paired;
    std::optional<mlx::core::fast::CustomKernelFunction> mlp_down_paired;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_qz;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_ba;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_qz_paired;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_ba_paired;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_reduce_zba;
    std::optional<mlx::core::fast::CustomKernelFunction> gdn_reduce_ba;
    std::once_flag initialize_once;

    void initialize() {
        std::call_once(initialize_once, [this] {
            std::string header = QWEN38_QUANTIZED_METAL;
            header += QWEN38_QMM_FUSION_HEADER;
            mlp_gate_up = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mlp_gu_qmm_v2",
                {"x", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"},
                {"out"}, QWEN38_MLP_GATE_UP_SOURCE, header, false);
            mlp_down = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mlp_down_v1",
                {"x", "weight", "scales", "biases"},
                {"out"}, QWEN38_MLP_DOWN_SOURCE, header, false);
            mlp_gate_up_paired = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mlp_gu_pair_rows_v1",
                {"x", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"},
                {"out"}, QWEN38_MLP_GATE_UP_PAIRED_SOURCE, header, false);
            mlp_down_paired = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_mlp_down_pair_rows_v1",
                {"x", "weight", "scales", "biases"},
                {"out"}, QWEN38_MLP_DOWN_PAIRED_SOURCE, header, false);
            gdn_qz = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_gdn_qz_v2",
                {"x", "qkv_w", "qkv_s", "qkv_b", "z_w", "z_s", "z_b"},
                {"qkv_part", "z_part"}, QWEN38_GDN_OLD_QZ_SOURCE, header, false);
            gdn_ba = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_gdn_ba_v2",
                {"x", "beta_w", "beta_s", "beta_b", "alpha_w", "alpha_s", "alpha_b"},
                {"beta_part", "alpha_part"}, QWEN38_GDN_OLD_BA_SOURCE, header, false);
            gdn_qz_paired = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_gdn_qz_pair_rows_v1",
                {"x", "qkv_w", "qkv_s", "qkv_b", "z_w", "z_s", "z_b"},
                {"qkv_part", "z_part"}, QWEN38_GDN_PAIRED_QZ_SOURCE, header, false);
            gdn_ba_paired = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_gdn_ba_pair_rows_v1",
                {"x", "beta_w", "beta_s", "beta_b", "alpha_w", "alpha_s", "alpha_b"},
                {"beta_part", "alpha_part"}, QWEN38_GDN_PAIRED_BA_SOURCE, header, false);
            gdn_reduce_zba = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_gdn_reduce_zba_v2",
                {"z_part", "beta_part", "alpha_part"},
                {"z_out", "beta_out", "alpha_out"},
                QWEN38_GDN_REDUCE_ZBA_SOURCE, header, false);
            gdn_reduce_ba = mlx::core::fast::metal_kernel(
                "qw_qwen38_affine_gdn_reduce_ba_v2",
                {"beta_part", "alpha_part"},
                {"beta_out", "alpha_out"},
                QWEN38_GDN_REDUCE_BA_SOURCE, header, false);
        });
    }
};

Qwen38FusionKernelHolder& qwen38_fusion_kernels() {
    static Qwen38FusionKernelHolder holder;
    holder.initialize();
    return holder;
}

int qwen38_input_rows(const array& x, int expected_k) {
    if (x.dtype() != mlx::core::float32 || x.shape().empty()
            || x.shape().back() != expected_k || !x.flags().row_contiguous) {
        throw std::invalid_argument("pinned Qwen3.8 fusion input is invalid");
    }
    size_t rows = x.size() / static_cast<size_t>(expected_k);
    if (rows < 4 || rows > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("pinned Qwen3.8 fusion row count is invalid");
    }
    return static_cast<int>(rows);
}

void validate_plane(
    const array& weight,
    const array& scales,
    const array& biases,
    int bits,
    int k,
    int n) {
    int packed = k * bits / 32;
    int groups = k / 32;
    if ((bits != 4 && bits != 5 && bits != 8)
            || weight.dtype() != mlx::core::uint32
            || weight.shape() != Shape{n, packed}
            || scales.dtype() != mlx::core::float32
            || scales.shape() != Shape{n, groups}
            || biases.dtype() != mlx::core::float32
            || biases.shape() != Shape{n, groups}) {
        throw std::invalid_argument("pinned Qwen3.8 affine fusion plane is invalid");
    }
}

int qwen38_split_k(int m, int k, int n) {
    int m_tiles = (m + 31) / 32;
    int n_tiles = (n + 31) / 32;
    int split = std::max(1, 512 / (m_tiles * n_tiles));
    split = std::min(split, k / 32);
    while (split > 1 && k % (split * 32) != 0) {
        --split;
    }
    return split;
}

bool qwen38_pair_rows(int m) {
    return m == 288 || m == 2048;
}

std::vector<std::pair<std::string, TemplateArg>> args(
    std::initializer_list<std::pair<std::string, int>> values) {
    std::vector<std::pair<std::string, TemplateArg>> result;
    result.reserve(values.size());
    for (const auto& [name, value] : values) {
        result.emplace_back(name, value);
    }
    return result;
}

} // namespace

std::unique_ptr<MlxArray> qwen38_affine_mlp_fused(
    const MlxArray& x,
    const MlxArray& gate_w,
    const MlxArray& gate_s,
    const MlxArray& gate_b,
    int32_t gate_bits,
    const MlxArray& up_w,
    const MlxArray& up_s,
    const MlxArray& up_b,
    int32_t up_bits,
    const MlxArray& down_w,
    const MlxArray& down_s,
    const MlxArray& down_b,
    int32_t down_bits,
    bool paired_rows) {
#ifndef __APPLE__
    throw std::invalid_argument("pinned Qwen3.8 affine fusion requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("pinned Qwen3.8 affine fusion requires Metal");
    }
    int m = qwen38_input_rows(x.inner, 5120);
    validate_plane(gate_w.inner, gate_s.inner, gate_b.inner, gate_bits, 5120, 17408);
    validate_plane(up_w.inner, up_s.inner, up_b.inner, up_bits, 5120, 17408);
    array input = reshape(x.inner, {m, 5120});
    if (paired_rows && !qwen38_pair_rows(m)) {
        throw std::invalid_argument("paired Qwen3.8 MLP row count is invalid");
    }
    auto& kernels = qwen38_fusion_kernels();
    auto& gate_up_kernel = paired_rows ? kernels.mlp_gate_up_paired : kernels.mlp_gate_up;
    int row_simdgroups = paired_rows ? 8 : 4;
    int row_groups = paired_rows ? (m + 63) / 64 : (m + 31) / 32;
    auto gated = (*gate_up_kernel)(
        {input, gate_w.inner, gate_s.inner, gate_b.inner,
         up_w.inner, up_s.inner, up_b.inner},
        {Shape{m, 17408}}, {float32},
        std::make_tuple(((17408 + 31) / 32) * 32, row_groups * row_simdgroups, 1),
        std::make_tuple(32, row_simdgroups, 1),
        args({{"GateBits", gate_bits}, {"UpBits", up_bits}, {"MRows", m}}),
        std::nullopt, false, {});
    validate_plane(down_w.inner, down_s.inner, down_b.inner, down_bits, 17408, 5120);
    int split = qwen38_split_k(m, 17408, 5120);
    if (split != 1 && split != 2) {
        throw std::invalid_argument("pinned Qwen3.8 MLP down split changed");
    }
    if (paired_rows && split != 1) {
        throw std::invalid_argument("pinned Qwen3.8 paired MLP down split changed");
    }
    int simdgroups = paired_rows ? 8 : 4 * split;
    auto& down_kernel = paired_rows ? kernels.mlp_down_paired : kernels.mlp_down;
    auto output = (*down_kernel)(
        {gated[0], down_w.inner, down_s.inner, down_b.inner},
        {Shape{m, 5120}}, {float32},
        std::make_tuple(((5120 + 31) / 32) * 32,
                        row_groups * simdgroups, 1),
        std::make_tuple(32, simdgroups, 1),
        args({{"Bits", down_bits}, {"SplitK", split}, {"MRows", m}}),
        std::nullopt, false, {});
    Shape shape(x.inner.shape().begin(), x.inner.shape().end() - 1);
    shape.push_back(5120);
    return std::make_unique<MlxArray>(reshape(output[0], shape));
#endif
}

std::unique_ptr<Qwen38GdnIngressOutputs> qwen38_affine_gdn_ingress_fused(
    const MlxArray& x,
    const MlxArray& qkv_w,
    const MlxArray& qkv_s,
    const MlxArray& qkv_b,
    int32_t qkv_bits,
    const MlxArray& z_w,
    const MlxArray& z_s,
    const MlxArray& z_b,
    int32_t z_bits,
    const MlxArray& beta_w,
    const MlxArray& beta_s,
    const MlxArray& beta_b,
    int32_t beta_bits,
    const MlxArray& alpha_w,
    const MlxArray& alpha_s,
    const MlxArray& alpha_b,
    int32_t alpha_bits,
    bool paired_rows) {
#ifndef __APPLE__
    throw std::invalid_argument("pinned Qwen3.8 affine fusion requires Metal");
#else
    using namespace mlx::core;
    if (!metal::is_available()) {
        throw std::invalid_argument("pinned Qwen3.8 affine fusion requires Metal");
    }
    int m = qwen38_input_rows(x.inner, 5120);
    validate_plane(qkv_w.inner, qkv_s.inner, qkv_b.inner, qkv_bits, 5120, 10240);
    validate_plane(z_w.inner, z_s.inner, z_b.inner, z_bits, 5120, 6144);
    validate_plane(beta_w.inner, beta_s.inner, beta_b.inner, beta_bits, 5120, 48);
    validate_plane(alpha_w.inner, alpha_s.inner, alpha_b.inner, alpha_bits, 5120, 48);
    int sq = qwen38_split_k(m, 5120, 10240);
    int sz = qwen38_split_k(m, 5120, 6144);
    int sb = qwen38_split_k(m, 5120, 48);
    int sa = sb;
    if (sq != 1 || (sz != 1 && sz != 2)
            || (sb != 1 && sb != 4 && sb != 5 && sb != 8 && sb != 10
                && sb != 16 && sb != 20 && sb != 32 && sb != 40
                && sb != 80 && sb != 160)) {
        throw std::invalid_argument("pinned Qwen3.8 GDN split changed");
    }
    if (paired_rows && !qwen38_pair_rows(m)) {
        throw std::invalid_argument("paired Qwen3.8 GDN row count is invalid");
    }
    auto& kernels = qwen38_fusion_kernels();
    int row_simdgroups = paired_rows ? 8 : 4;
    int row_groups = paired_rows ? (m + 63) / 64 : (m + 31) / 32;
    array input = reshape(x.inner, {m, 5120});
    std::vector<array> final_outputs;
    int qz_work = 10240 / 32 + (6144 / 32) * sz;
    auto& qz_kernel = paired_rows ? kernels.gdn_qz_paired : kernels.gdn_qz;
    auto qz = (*qz_kernel)(
        {input, qkv_w.inner, qkv_s.inner, qkv_b.inner, z_w.inner, z_s.inner,
         z_b.inner},
        {Shape{m, 10240}, Shape{sz, m, 6144}}, {float32, float32},
        std::make_tuple(qz_work * 32, row_groups * row_simdgroups, 1),
        std::make_tuple(32, row_simdgroups, 1),
        args({{"QkvBits", qkv_bits},
              {"ZBits", z_bits},
              {"SplitZ", sz},
              {"MRows", m}}),
        std::nullopt, false, {});
    int ba_work = 2 * sb + 2 * sa;
    auto& ba_kernel = paired_rows ? kernels.gdn_ba_paired : kernels.gdn_ba;
    auto ba = (*ba_kernel)(
        {input, beta_w.inner, beta_s.inner, beta_b.inner, alpha_w.inner,
         alpha_s.inner, alpha_b.inner},
        {Shape{sb, m, 48}, Shape{sa, m, 48}}, {float32, float32},
        std::make_tuple(ba_work * 32, row_groups * row_simdgroups, 1),
        std::make_tuple(32, row_simdgroups, 1),
        args({{"BetaBits", beta_bits},
              {"AlphaBits", alpha_bits},
              {"SplitB", sb},
              {"SplitA", sa},
              {"MRows", m}}),
        std::nullopt, false, {});
    if (sz == 1 && sb == 1) {
      final_outputs = {qz[0], qz[1], ba[0], ba[1]};
    } else if (sz == 1) {
      int block_width = sb < 32 ? 128 : 32;
      int blocks = (m * 48 + block_width - 1) / block_width;
      int reducer_rows = m == 2048 && sb == 4 ? 4 : 8;
      auto reduced = (*kernels.gdn_reduce_ba)(
          {ba[0], ba[1]}, {Shape{m, 48}, Shape{m, 48}}, {float32, float32},
          std::make_tuple(blocks * 2 * 32, reducer_rows, 1),
          std::make_tuple(32, reducer_rows, 1),
          args({{"SplitB", sb}, {"SplitA", sa}, {"MRows", m}}), std::nullopt,
          false, {});
      final_outputs = {qz[0], qz[1], reduced[0], reduced[1]};
    } else {
      int z_blocks = (m * 6144 + 127) / 128;
      int block_width = sb < 32 ? 128 : 32;
      int b_blocks = (m * 48 + block_width - 1) / block_width;
      int reducer_rows = m == 2048 && sb == 4 ? 4 : 8;
      auto reduced = (*kernels.gdn_reduce_zba)(
          {qz[1], ba[0], ba[1]}, {Shape{m, 6144}, Shape{m, 48}, Shape{m, 48}},
          {float32, float32, float32},
          std::make_tuple((z_blocks + 2 * b_blocks) * 32, reducer_rows, 1),
          std::make_tuple(32, reducer_rows, 1),
          args({{"SplitZ", sz}, {"SplitB", sb}, {"SplitA", sa}, {"MRows", m}}),
          std::nullopt, false, {});
      final_outputs = {qz[0], reduced[0], reduced[1], reduced[2]};
    }
    array qkv = final_outputs[0];
    array z = final_outputs[1];
    array beta = final_outputs[2];
    array alpha = final_outputs[3];
    Shape prefix(x.inner.shape().begin(), x.inner.shape().end() - 1);
    auto with_last = [&prefix](int n) {
        Shape shape = prefix;
        shape.push_back(n);
        return shape;
    };
    auto outputs = std::make_unique<Qwen38GdnIngressOutputs>();
    outputs->qkv = std::make_unique<MlxArray>(reshape(qkv, with_last(10240)));
    outputs->z = std::make_unique<MlxArray>(reshape(z, with_last(6144)));
    outputs->beta = std::make_unique<MlxArray>(reshape(beta, with_last(48)));
    outputs->alpha = std::make_unique<MlxArray>(reshape(alpha, with_last(48)));
    return outputs;
#endif
}

std::unique_ptr<MlxArray> qwen38_gdn_take_qkv(Qwen38GdnIngressOutputs& outputs) {
    return std::move(outputs.qkv);
}
std::unique_ptr<MlxArray> qwen38_gdn_take_z(Qwen38GdnIngressOutputs& outputs) {
    return std::move(outputs.z);
}
std::unique_ptr<MlxArray> qwen38_gdn_take_beta(Qwen38GdnIngressOutputs& outputs) {
    return std::move(outputs.beta);
}
std::unique_ptr<MlxArray> qwen38_gdn_take_alpha(Qwen38GdnIngressOutputs& outputs) {
    return std::move(outputs.alpha);
}

} // namespace mlx_cxx
