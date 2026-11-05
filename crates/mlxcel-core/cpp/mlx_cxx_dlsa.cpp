// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0
//
// D-LSA K1: the fused top-p bucket select + expansion + window + sinks
// (docs/dlsa.md §14.1). One threadgroup per query row turns the row's centroid
// scores into its read list, bit-identical to the host reference
// (crates/runtime/src/dlsa/select.rs `select_from_scores`):
//
// 1. Rank key. Bucket b with score s gets the unique 64-bit key
//    `(ord(s) << 32) | (0x7FFFFFFF - b)`: `ord` is the order-preserving map of
//    the f32 bits, after `-0.0 -> +0.0` (the host compares with `partial_cmp`,
//    so ±0 tie and break on the lower bucket). A larger key is a better bucket,
//    and since the keys are unique the selection has no ties to break.
// 2. Threshold. With more visible buckets than `top_p`, an MSD radix select
//    over 8-bit digits (at most 8 passes, a 256-bin threadgroup histogram each,
//    the row's scores re-read from device) finds the top_p-th largest key τ;
//    it stops as soon as the bin holding τ holds exactly the keys still needed.
// 3. Compact the keys >= τ (exactly min(visible, top_p) of them) into
//    threadgroup memory and bitonic-sort them descending: the picked buckets,
//    best first.
// 4. Expansion in rank order (the host's walk): count each picked bucket's
//    eligible members (`sink <= m < win_start`), exclusive-scan the counts in
//    rank order, and write bucket i's eligible members (ascending) at
//    `offset_i + j` while below top_k. Then the window [win_start, p], the
//    sinks [0, min(sink, win_start, p + 1)), and -1 padding.
//
// NaN: the host refuses a NaN score. Here a NaN ranks below every number
// (ord 0) and sets bit 0 of the row's `status`; callers read `status` only at
// their existing sync points. Bit 1 of `status`: the row's visible buckets
// exceed the scores' columns, and only the columns present were ranked.

#include "mlx_cxx_internal.h"

namespace mlx_cxx {

namespace {

// Most buckets a row may pick: the keys buffer of the threadgroup. Matches
// qw-runtime's TOP_P_LIMIT (upstream's top-k engine limit).
constexpr int32_t DLSA_TOP_P_MAX = 2048;

static const char* DLSA_SELECT_HEADER = R"(
// Order-preserving map of f32 bits: a larger float gets a larger uint. -0.0
// maps as +0.0, and a NaN maps to 0, below -inf (0x007FFFFF).
static inline uint dlsa_ord(float f) {
    uint u = as_type<uint>(f);
    uint mag = u & 0x7FFFFFFFu;
    if (mag > 0x7F800000u) {
        return 0u;
    }
    if (mag == 0u) {
        u = 0u;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The unique rank key of bucket b: score descending, then bucket ascending.
static inline ulong dlsa_key(float f, uint b) {
    return (ulong(dlsa_ord(f)) << 32) | ulong(0x7FFFFFFFu - b);
}
)";

static const char* DLSA_SELECT_SOURCE = R"(
    constexpr uint KEYS = 2048;
    constexpr uint NSG = TG / 32;
    constexpr uint ITEMS = KEYS / TG;

    threadgroup atomic_uint hist[256];
    threadgroup ulong keys[KEYS];
    threadgroup uint offs[KEYS];
    threadgroup uint sg_tot[32];
    threadgroup uint pick[4];
    threadgroup atomic_uint counter;
    threadgroup atomic_uint nan_seen;

    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;

    const int p = positions[row];
    const uint S = (uint)stride;
    const uint tp = (uint)top_p;
    const uint cap = (uint)bucket_cap;
    const uint width = (uint)(top_k + window + sink);

    uint vis = (uint)(p / window) * (uint)buckets_per_block;
    vis = metal::min(vis, (uint)total_rows);
    uint status_bits = 0u;
    if (vis > S) {
        vis = S;
        status_bits |= 2u;
    }
    const uint need = metal::min(vis, tp);
    // `auto`: MLX passes inputs below 8 elements in the constant address space
    auto srow = scores + (ulong)row * (ulong)S;

    // ── 2. MSD radix select of the need-th largest key ──────────────────────
    ulong prefix = 0;
    if (vis > tp) {
        ulong pmask = 0;
        uint remaining = need;
        for (int shift = 56; shift >= 0; shift -= 8) {
            for (uint i = tid; i < 256; i += TG) {
                atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint base = 0; base < vis; base += TG) {
                uint i = base + tid;
                bool valid = false;
                uint digit = 0u;
                if (i < vis) {
                    ulong key = dlsa_key(srow[i], i);
                    valid = (key & pmask) == prefix;
                    digit = (uint)(key >> (uint)shift) & 0xFFu;
                }
                // The leading digits of nearby scores mostly agree: a
                // simdgroup whose valid lanes share one digit adds once.
                uint dmin = simd_min(valid ? digit : 256u);
                bool uniform = simd_all(!valid || digit == dmin);
                uint cnt = simd_sum(valid ? 1u : 0u);
                if (uniform) {
                    if (lane == 0 && cnt > 0) {
                        atomic_fetch_add_explicit(&hist[dmin], cnt, memory_order_relaxed);
                    }
                } else if (valid) {
                    atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                // Lane l holds bins 255 - 8l down to 248 - 8l (best first)
                uint c[8];
                uint s = 0;
                for (uint k = 0; k < 8; ++k) {
                    c[k] = atomic_load_explicit(&hist[255 - 8 * lane - k], memory_order_relaxed);
                    s += c[k];
                }
                uint above = simd_prefix_exclusive_sum(s);
                if (above < remaining && remaining <= above + s) {
                    uint acc = above;
                    for (uint k = 0; k < 8; ++k) {
                        if (acc + c[k] >= remaining) {
                            pick[0] = 255 - 8 * lane - k;
                            pick[1] = acc;
                            pick[2] = c[k];
                            break;
                        }
                        acc += c[k];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint dsel = pick[0];
            uint above = pick[1];
            uint in_bin = pick[2];
            remaining -= above;
            prefix |= ((ulong)dsel) << (uint)shift;
            pmask |= ((ulong)0xFF) << (uint)shift;
            // Every key of this bin is needed: τ is the bin's lowest key
            if (in_bin == remaining) {
                break;
            }
        }
    }
    const ulong tau = prefix;

    // ── 3. Compact the keys >= τ, then sort them descending ────────────────
    if (tid == 0) {
        atomic_store_explicit(&counter, 0u, memory_order_relaxed);
        atomic_store_explicit(&nan_seen, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < vis; base += TG) {
        uint i = base + tid;
        bool take = false;
        ulong key = 0;
        if (i < vis) {
            float f = srow[i];
            if ((as_type<uint>(f) & 0x7FFFFFFFu) > 0x7F800000u) {
                atomic_store_explicit(&nan_seen, 1u, memory_order_relaxed);
            }
            key = dlsa_key(f, i);
            take = key >= tau;
        }
        uint before = simd_prefix_exclusive_sum(take ? 1u : 0u);
        uint total = simd_sum(take ? 1u : 0u);
        uint slot0 = 0;
        if (lane == 0 && total > 0) {
            slot0 = atomic_fetch_add_explicit(&counter, total, memory_order_relaxed);
        }
        slot0 = simd_broadcast_first(slot0);
        if (take && slot0 + before < KEYS) {
            keys[slot0 + before] = key;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n2 = 1;
    while (n2 < need) {
        n2 <<= 1;
    }
    for (uint i = need + tid; i < n2; i += TG) {
        keys[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= n2; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint t = tid; t < n2 / 2; t += TG) {
                uint i = 2 * j * (t / j) + (t % j);
                uint l = i + j;
                ulong a = keys[i];
                ulong b = keys[l];
                bool desc = (i & k) == 0;
                if ((a < b) == desc) {
                    keys[i] = b;
                    keys[l] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── 4. Expansion in rank order ──────────────────────────────────────────
    const int win_start = metal::max(0, p + 1 - window);
    const int lo = sink;
    const int hi = win_start;
    device int* prow = picked + (ulong)row * (ulong)tp;
    for (uint i = tid; i < KEYS; i += TG) {
        uint c = 0;
        if (i < need) {
            uint b = 0x7FFFFFFFu - (uint)keys[i];
            prow[i] = (int)b;
            auto mem = members + (ulong)b * (ulong)cap;
            for (uint j = 0; j < cap; ++j) {
                int m = mem[j];
                c += (m >= lo && m < hi) ? 1u : 0u;
            }
        } else if (i < tp) {
            prow[i] = -1;
        }
        offs[i] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Exclusive scan of offs[0..KEYS): ITEMS consecutive entries a thread
    uint vals[ITEMS];
    uint mine = 0;
    for (uint k = 0; k < ITEMS; ++k) {
        vals[k] = offs[tid * ITEMS + k];
        mine += vals[k];
    }
    uint lane_before = simd_prefix_exclusive_sum(mine);
    if (lane == 31) {
        sg_tot[sg] = lane_before + mine;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint v = lane < NSG ? sg_tot[lane] : 0u;
        uint ex = simd_prefix_exclusive_sum(v);
        uint all = simd_sum(v);
        if (lane < NSG) {
            sg_tot[lane] = ex;
        }
        if (lane == 0) {
            pick[3] = all;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint run = sg_tot[sg] + lane_before;
    for (uint k = 0; k < ITEMS; ++k) {
        offs[tid * ITEMS + k] = run;
        run += vals[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tk = (uint)top_k;
    device int* lrow = lists + (ulong)row * (ulong)width;
    for (uint i = tid; i < need; i += TG) {
        uint off = offs[i];
        if (off >= tk) {
            continue;
        }
        uint b = 0x7FFFFFFFu - (uint)keys[i];
        auto mem = members + (ulong)b * (ulong)cap;
        for (uint j = 0; j < cap && off < tk; ++j) {
            int m = mem[j];
            if (m >= lo && m < hi) {
                lrow[off] = m;
                off += 1;
            }
        }
    }
    const uint expanded = metal::min(pick[3], tk);
    const uint wn = (uint)(p - win_start + 1);
    const uint sn = (uint)metal::min(metal::min(sink, win_start), p + 1);
    const uint len = expanded + wn + sn;
    for (uint idx = expanded + tid; idx < width; idx += TG) {
        int value;
        if (idx < expanded + wn) {
            value = win_start + (int)(idx - expanded);
        } else if (idx < len) {
            value = (int)(idx - expanded - wn);
        } else {
            value = -1;
        }
        lrow[idx] = value;
    }
    if (tid == 0) {
        lengths[row] = (int)len;
        status[row] = (int)(status_bits | atomic_load_explicit(&nan_seen, memory_order_relaxed));
    }
)";

struct DlsaSelectKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    mlx::core::fast::CustomKernelFunction& get() {
        if (!kernel) {
            kernel = mlx::core::fast::metal_kernel(
                "dlsa_select",
                {"scores", "members", "positions", "total_rows", "stride", "window",
                 "sink", "top_k", "top_p", "bucket_cap", "buckets_per_block"},
                {"lists", "lengths", "picked", "status"},
                DLSA_SELECT_SOURCE,
                DLSA_SELECT_HEADER);
        }
        return *kernel;
    }
};

DlsaSelectKernelHolder& get_dlsa_select_kernel() {
    static DlsaSelectKernelHolder holder;
    return holder;
}

void dlsa_require(bool ok, const char* what) {
    if (!ok) {
        throw std::invalid_argument(std::string("[dlsa_select] ") + what);
    }
}

}  // namespace

void dlsa_select(
    const MlxArray& scores,
    const MlxArray& members,
    const MlxArray& positions,
    int32_t total_rows,
    int32_t window,
    int32_t sink,
    int32_t top_k,
    int32_t top_p,
    int32_t bucket_cap,
    int32_t buckets_per_block,
    int32_t threads,
    std::unique_ptr<MlxArray>& lists_out,
    std::unique_ptr<MlxArray>& lengths_out,
    std::unique_ptr<MlxArray>& picked_out,
    std::unique_ptr<MlxArray>& status_out
) {
    using namespace mlx::core;
    const array& sc = scores.inner;
    const array& mem = members.inner;
    const array& pos = positions.inner;
    dlsa_require(sc.ndim() == 2 && sc.dtype() == float32, "scores must be [R, S] float32");
    dlsa_require(mem.ndim() == 2 && mem.dtype() == int32, "members must be [N, cap] int32");
    dlsa_require(pos.ndim() == 1 && pos.dtype() == int32, "positions must be [R] int32");
    dlsa_require(pos.shape(0) == sc.shape(0), "positions and scores disagree on R");
    dlsa_require(mem.shape(1) == bucket_cap, "members' row width must be bucket_cap");
    dlsa_require(window >= 1 && top_k >= 1 && sink >= 0 && bucket_cap >= 1
                     && buckets_per_block >= 1,
                 "window, top_k, bucket_cap and buckets_per_block must be >= 1, sink >= 0");
    dlsa_require(top_p >= 1 && top_p <= DLSA_TOP_P_MAX, "top_p must be in [1, 2048]");
    dlsa_require(total_rows >= 0 && total_rows <= mem.shape(0),
                 "total_rows must be in [0, members rows]");
    const int64_t width64 = int64_t(top_k) + window + sink;
    dlsa_require(width64 <= INT32_MAX, "list width overflows i32");
    const int32_t width = static_cast<int32_t>(width64);
    int32_t tg = threads == 0 ? 1024 : threads;
    dlsa_require(tg >= 64 && tg <= 1024 && (tg & (tg - 1)) == 0,
                 "threads must be 0 or a power of two in [64, 1024]");

    const int32_t rows = sc.shape(0);
    const int32_t stride = sc.shape(1);
    if (rows == 0) {
        lists_out = std::make_unique<MlxArray>(zeros({0, width}, int32));
        lengths_out = std::make_unique<MlxArray>(zeros({0}, int32));
        picked_out = std::make_unique<MlxArray>(zeros({0, top_p}, int32));
        status_out = std::make_unique<MlxArray>(zeros({0}, int32));
        return;
    }
    // A kernel buffer may not be empty: an empty cache has nothing to read,
    // and the kernel clamps its visible buckets to `stride` = 0 and
    // `total_rows` = 0, so a one-element stand-in is never read.
    array sc_in = sc.size() == 0 ? zeros({1}, float32) : sc;
    array mem_in = mem.size() == 0 ? full({1}, -1, int32) : mem;
    auto results = get_dlsa_select_kernel().get()(
        {sc_in, mem_in, pos, array(total_rows), array(stride), array(window), array(sink),
         array(top_k), array(top_p), array(bucket_cap), array(buckets_per_block)},
        {{rows, width}, {rows}, {rows, top_p}, {rows}},
        {int32, int32, int32, int32},
        std::make_tuple(rows * tg, 1, 1),
        std::make_tuple(tg, 1, 1),
        {{"TG", tg}},
        std::nullopt,
        false,
        {});
    lists_out = std::make_unique<MlxArray>(std::move(results[0]));
    lengths_out = std::make_unique<MlxArray>(std::move(results[1]));
    picked_out = std::make_unique<MlxArray>(std::move(results[2]));
    status_out = std::make_unique<MlxArray>(std::move(results[3]));
}

}  // namespace mlx_cxx
