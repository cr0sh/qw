// SPDX-License-Identifier: Apache-2.0
//
// D-LSA fused sparse gather-attention (K2, docs/dlsa.md §14.1).
//
// Every query row r of q [1,Hq,L,D] attends over the keys lists[r, :lengths[r]]
// of k/v [1,Hkv,T,D] only: no mask, no causality check (the lists are causal by
// construction). Modeled on MLX's sdpa_vector / sdpa_vector_2pass
// (mlx/backend/metal/kernels/sdpa_vector.h at the pinned commit), with GQA
// folded in: one threadgroup per (row, KV head) owns all G = Hq/Hkv query heads
// of the group, so each gathered K/V row is fetched once for G heads (its HG
// head-group simdgroups read it close together, from L1).
//
// Layout: NSG simdgroups of 32 lanes. The G heads split into HG groups of HS
// heads (6 = 3 + 3 at Qwen geometry); simdgroup s serves head group s % HG over
// key slice s / HG. A lane holds D/32 consecutive dims of its HS queries (in
// q's dtype) and HS x D/32 f32 accumulators. A slice walks 32-entry chunks of
// the list (lane j holds entry j, loaded a chunk ahead); per TK = 2 keys it
// issues the K and V row loads, then HS x TK simd_sum dots and an online
// softmax in the exp2 domain (one rescale per head per step, uniform across
// the simdgroup). A cross-simdgroup merge through threadgroup memory finishes.
//
// Split-K (decode / verify, few rows): PARTS contiguous partitions of each
// list run as separate threadgroups writing f32 partials (unnormalized
// accumulator, running max, sum), and a merge kernel combines them.
//
// K/V are read through their strides (ensure_row_contiguous = false): the
// fetched cache is a slice of a larger buffer and must not be copied.
// The reduction order is fixed by the shapes and launch knobs, so results are
// deterministic for fixed inputs and independent of the list order up to f32
// rounding.

#include "mlx_cxx_internal.h"

#include "mlx/fast.h"

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlx_cxx {

namespace {

constexpr int DLSA_MAX_SIMDGROUPS = 16;
constexpr int DLSA_MAX_PARTS = 128;

// Helpers shared by both kernels (the `header` of metal_kernel).
const char* DLSA_ATTN_HEADER = R"(
// Finite "no key yet" running max (as sdpa_vector's finite_min), so the
// online-softmax rescale exp2(m - m_new) of the first key needs no branch.
// The merges skip an empty simdgroup or partition (m still the sentinel)
// explicitly rather than trusting fast::exp2 at -3.4e38.
constant float DLSA_NEG = -3.402823466e+38f;
// N consecutive elements at p (aligned to 4 elements when Vec), copied.
template <int N, bool Vec, typename T>
METAL_FUNC void dlsa_copy(const device T* p, thread T* out) {
  if (Vec && N % 4 == 0) {
    const device vec<T, 4>* p4 = (const device vec<T, 4>*)p;
    #pragma unroll
    for (int c = 0; c < N / 4; c++) {
      vec<T, 4> x = p4[c];
      out[4 * c + 0] = x[0];
      out[4 * c + 1] = x[1];
      out[4 * c + 2] = x[2];
      out[4 * c + 3] = x[3];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      out[i] = p[i];
    }
  }
}
)";

// Shared prologue + main loop + in-threadgroup merge. Leaves, per head g:
// M[g] (max, log2 domain), Lsum[g] (sum of exp2), and the merged unnormalized
// accumulator delivered dim by dim to the epilogue through `DLSA_EMIT(g, d, acc)`.
const char* DLSA_ATTN_BODY = R"(
  constexpr int DPT = D / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tid = sg * 32 + lane;
  const uint kvh = threadgroup_position_in_grid.x;
  const uint row = threadgroup_position_in_grid.y;
  const uint part = threadgroup_position_in_grid.z;
  const int L = q_shape[2];
  const int Hkv = k_shape[1];
  const int Tk = k_shape[2];
  const int Hq = Hkv * G;
  const int W = lists_shape[1];
  const long kst = k_strides[2];
  const long vst = v_strides[2];
  const float qscale = scale * 1.4426950408889634f;

  // Head split: the G heads form HG = G / HS groups of HS heads; simdgroup sg
  // serves head group hg = sg % HG over the key slice ks = sg / HG of KS. Both
  // head groups read the same K/V rows at about the same time (L1 hits); the
  // smaller per-lane state raises occupancy.
  constexpr int HG = G / HS;
  constexpr int KS = NSG / HG;
  const uint hg = sg % HG;
  const uint ks = sg / HG;
  // q kept in its own dtype (exact) and unscaled; the scale applies to the
  // reduced score. K/V rows stay in their dtype until the FMA.
  QT qv[HS][DPT];
  float o[HS][DPT];
  float m[HS];
  float l[HS];
  #pragma unroll
  for (int g = 0; g < HS; g++) {
    const uint head = kvh * G + hg * HS + g;
    const device QT* qp = q + ((long)head * L + row) * D + lane * DPT;
    dlsa_copy<DPT, true>(qp, qv[g]);
    #pragma unroll
    for (int i = 0; i < DPT; i++) {
      o[g][i] = 0.0f;
    }
    m[g] = DLSA_NEG;
    l[g] = 0.0f;
  }

  const int len = clamp(lengths[row], 0, W);
  // Partitions of whole 32-entry chunks
  const int per = ((len + PARTS - 1) / PARTS + 31) / 32 * 32;
  const int begin = (int)part * per;
  const int end = min(len, begin + per);
  // `auto`: metal_kernel passes inputs of fewer than 8 elements as `constant`
  const auto lrow = lists + (long)row * W;
  const device KT* kb = k + (long)kvh * k_strides[1] + lane * DPT;
  const device KT* vb = v + (long)kvh * v_strides[1] + lane * DPT;

  // Simdgroup slice ks walks 32-entry chunks ks, ks + KS, ... of [begin, end).
  // Lane j holds entry j of the current chunk (one coalesced load, issued a
  // chunk ahead), so a key's row address never waits on its index load.
  const int nchunks = max(0, end - begin + 31) / 32;
  int c = (int)ks;
  int nxt = -1;
  if (c < nchunks && begin + c * 32 + (int)lane < end) {
    nxt = lrow[begin + c * 32 + lane];
  }
  for (; c < nchunks; c += KS) {
   const int cur = nxt;
   const int cn = c + KS;
   nxt = (cn < nchunks && begin + cn * 32 + (int)lane < end) ? lrow[begin + cn * 32 + lane] : -1;
   const int cnt = min(32, end - (begin + c * 32));
   for (int u = 0; u < cnt; u += TK) {
    int idx[TK];
    bool ok[TK];
    #pragma unroll
    for (int t = 0; t < TK; t++) {
      // Entries past the chunk's end hold -1
      const int p = simd_shuffle(cur, (ushort)(u + t));
      ok[t] = p >= 0 && p < Tk;
      idx[t] = ok[t] ? p : 0;
    }
    KT kf[TK][DPT];
    KT vf[TK][DPT];
    #pragma unroll
    for (int t = 0; t < TK; t++) {
      dlsa_copy<DPT, VEC>(kb + (long)idx[t] * kst, kf[t]);
    }
    #pragma unroll
    for (int t = 0; t < TK; t++) {
      dlsa_copy<DPT, VEC>(vb + (long)idx[t] * vst, vf[t]);
    }
    float s[TK][HS];
    // simd_sum per (key, head): a reduce-scatter butterfly over all TK*G
    // partials measured slower on M4 Max (70 vs 54 ms per 1536-row chunk).
    #pragma unroll
    for (int t = 0; t < TK; t++) {
      #pragma unroll
      for (int g = 0; g < HS; g++) {
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPT; i++) {
          acc += float(qv[g][i]) * float(kf[t][i]);
        }
        s[t][g] = simd_sum(acc) * qscale;
      }
    }
    #pragma unroll
    for (int g = 0; g < HS; g++) {
      float mx = -INFINITY;
      #pragma unroll
      for (int t = 0; t < TK; t++) {
        mx = ok[t] ? max(mx, s[t][g]) : mx;
      }
      if (mx > m[g]) {
        const float f = fast::exp2(m[g] - mx);
        l[g] *= f;
        #pragma unroll
        for (int i = 0; i < DPT; i++) {
          o[g][i] *= f;
        }
        m[g] = mx;
      }
      #pragma unroll
      for (int t = 0; t < TK; t++) {
        const float pr = ok[t] ? fast::exp2(s[t][g] - m[g]) : 0.0f;
        l[g] += pr;
        #pragma unroll
        for (int i = 0; i < DPT; i++) {
          o[g][i] += pr * float(vf[t][i]);
        }
      }
    }
   }
  }

  // Cross-simdgroup merge: head hg*HS + g sums the KS simdgroups hg + HG*kk.
  threadgroup float tg_m[NSG * HS];
  threadgroup float tg_l[NSG * HS];
  threadgroup float tg_o[NSG * D];
  if (lane == 0) {
    #pragma unroll
    for (int g = 0; g < HS; g++) {
      tg_m[sg * HS + g] = m[g];
      tg_l[sg * HS + g] = l[g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // M, Lsum of every head of the group (indexed by head within the group)
  float M[G];
  float Lsum[G];
  #pragma unroll
  for (int gg = 0; gg < G; gg++) {
    const int h0 = gg / HS;
    const int gl = gg % HS;
    float mg = DLSA_NEG;
    for (int kk = 0; kk < KS; kk++) {
      mg = max(mg, tg_m[(h0 + HG * kk) * HS + gl]);
    }
    float ls = 0.0f;
    for (int kk = 0; kk < KS; kk++) {
      const int s2 = h0 + HG * kk;
      const float ms = tg_m[s2 * HS + gl];
      ls += ms == DLSA_NEG ? 0.0f : tg_l[s2 * HS + gl] * fast::exp2(ms - mg);
    }
    M[gg] = mg;
    Lsum[gg] = ls;
  }
  #pragma unroll
  for (int gl = 0; gl < HS; gl++) {
    // An empty slice (m still the sentinel) contributes nothing; o is 0 there
    const float fac = m[gl] == DLSA_NEG ? 0.0f : fast::exp2(m[gl] - M[hg * HS + gl]);
    #pragma unroll
    for (int i = 0; i < DPT; i++) {
      tg_o[sg * D + lane * DPT + i] = o[gl][i] * fac;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < (uint)D; d += NSG * 32) {
      #pragma unroll
      for (int h0 = 0; h0 < HG; h0++) {
        const int g = h0 * HS + gl;
        const uint head = kvh * G + g;
        float acc = 0.0f;
        for (int kk = 0; kk < KS; kk++) {
          acc += tg_o[(h0 + HG * kk) * D + d];
        }
        DLSA_EMIT
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
)";

// A simdgroup-matrix variant (8-key tiles: S^T = K.Q^T, P = softmax, O += P.V,
// heads padded to 8) measured 117 ms vs 54 ms for this layout on M4 Max: with
// G <= 8 nothing is reused across fragments, so the MMAs only add fragment
// traffic. Not kept.

// One pass: normalize and write the output row.
const char* DLSA_EMIT_DIRECT = R"(
      out[((long)head * L + row) * D + d] =
          static_cast<QT>(Lsum[g] > 0.0f ? acc / Lsum[g] : 0.0f);
)";

// Split-K: write the partition's f32 partial.
const char* DLSA_EMIT_PARTIAL = R"(
      const long prow = ((long)row * Hq + head) * PARTS + part;
      part_o[prow * D + d] = acc;
      if (d == 0) {
        part_m[prow] = M[g];
        part_l[prow] = Lsum[g];
      }
)";

// Merge the PARTS partials of one (head, row): one simdgroup, D/32 dims a lane.
const char* DLSA_ATTN_MERGE = R"(
  constexpr int DPT = D / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.x;
  const uint row = threadgroup_position_in_grid.y;
  const int Hq = threadgroups_per_grid.x;
  const int L = threadgroups_per_grid.y;
  const long base = ((long)row * Hq + head) * PARTS;
  float mg = DLSA_NEG;
  for (int p = 0; p < PARTS; p++) {
    mg = max(mg, part_m[base + p]);
  }
  float acc[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; i++) {
    acc[i] = 0.0f;
  }
  float ls = 0.0f;
  for (int p = 0; p < PARTS; p++) {
    const float mp = part_m[base + p];
    const float f = mp == DLSA_NEG ? 0.0f : fast::exp2(mp - mg);
    ls += part_l[base + p] * f;
    const device float* po = part_o + (base + p) * D + lane * DPT;
    #pragma unroll
    for (int i = 0; i < DPT; i++) {
      acc[i] += po[i] * f;
    }
  }
  device OT* op = out + ((long)head * L + row) * D + lane * DPT;
  #pragma unroll
  for (int i = 0; i < DPT; i++) {
    op[i] = static_cast<OT>(ls > 0.0f ? acc[i] / ls : 0.0f);
  }
)";

std::string with_emit(const char* body_src, const char* emit) {
    std::string body(body_src);
    const std::string marker = "DLSA_EMIT";
    body.replace(body.find(marker), marker.size(), emit);
    return body;
}

using mlx::core::fast::CustomKernelFunction;

struct DlsaAttnKernels {
    std::optional<CustomKernelFunction> direct;
    std::optional<CustomKernelFunction> partial;
    std::optional<CustomKernelFunction> merge;
    std::once_flag init;

    void ensure() {
        std::call_once(init, [this] {
            // ensure_row_contiguous = false: K/V are strided views of the cache.
            direct = mlx::core::fast::metal_kernel(
                "mlxcel_dlsa_sparse_attn",
                {"q", "k", "v", "lists", "lengths", "scale"},
                {"out"},
                with_emit(DLSA_ATTN_BODY, DLSA_EMIT_DIRECT),
                DLSA_ATTN_HEADER,
                /*ensure_row_contiguous=*/false);
            partial = mlx::core::fast::metal_kernel(
                "mlxcel_dlsa_sparse_attn_partial",
                {"q", "k", "v", "lists", "lengths", "scale"},
                {"part_o", "part_m", "part_l"},
                with_emit(DLSA_ATTN_BODY, DLSA_EMIT_PARTIAL),
                DLSA_ATTN_HEADER,
                /*ensure_row_contiguous=*/false);
            merge = mlx::core::fast::metal_kernel(
                "mlxcel_dlsa_sparse_attn_merge",
                {"part_o", "part_m", "part_l"},
                {"out"},
                DLSA_ATTN_MERGE,
                DLSA_ATTN_HEADER,
                /*ensure_row_contiguous=*/true);
        });
    }
};

DlsaAttnKernels& kernels() {
    static DlsaAttnKernels k;
    return k;
}

bool float_dtype(Dtype t) {
    return t == float16 || t == bfloat16 || t == float32;
}

// Launch policy, measured on M4 Max at the Qwen geometry (Hq 24, Hkv 4,
// D 256, fp16, 4612-key lists, T = 32k..256k):
// - One pass once the rows alone fill the GPU: L * Hkv >= 160 threadgroups
//   (a 1536-row prefill chunk is 6144). 8 simdgroups: 38.7 ms at 32k vs 44.6
//   (2), 39.6 (4), 39.0 (16).
// - Otherwise split-K: about 512 threadgroups, at least 64 keys a partition,
//   4 simdgroups. Decode L = 1 at 32k: one pass 0.50 ms, 16 parts 0.20, 64
//   parts 0.16-0.20 (the auto choice, 72 parts, 0.19-0.20); L = 8: one pass
//   0.55, 16 parts 0.41 (auto).
int auto_parts(int units, int width, int nsg) {
    constexpr int kTargetTgs = 512;
    constexpr int kOnePassUnits = 160;
    constexpr int kMinKeysPerPart = 64;
    if (units >= kOnePassUnits) {
        return 1;
    }
    int parts = (kTargetTgs + units - 1) / units;
    const int by_keys = std::max(1, width / std::max(kMinKeysPerPart, nsg * 4));
    parts = std::min(parts, by_keys);
    return std::max(1, std::min(parts, DLSA_MAX_PARTS));
}

int auto_simdgroups(int parts) {
    return parts > 1 ? 4 : 8;
}

// Heads per simdgroup: groups of more than 4 heads split into equal groups of
// at most 4 (6 -> 3 + 3, 8 -> 4 + 4), which keeps the per-lane state small
// enough for full occupancy (G = 6 at D = 256: 54 ms -> 38.7 ms per 1536-row
// prefill chunk at 32k; 2 + 2 + 2 was 43.8 ms).
int heads_per_simdgroup(int group) {
    if (group <= 4) {
        return group;
    }
    for (int hs = 4; hs >= 2; hs--) {
        if (group % hs == 0) {
            return hs;
        }
    }
    return group;
}
array run(
    const array& q_in,
    const array& k_in,
    const array& v_in,
    const array& lists_in,
    const array& lengths_in,
    float scale,
    int parts,
    int nsg) {
    if (!mlx::core::metal::is_available()) {
        throw std::invalid_argument("dlsa_sparse_attention requires the Metal backend");
    }
    if (q_in.ndim() != 4 || k_in.ndim() != 4 || v_in.ndim() != 4) {
        throw std::invalid_argument("dlsa_sparse_attention: q/k/v must be rank 4");
    }
    const int hq = q_in.shape(1);
    const int rows = q_in.shape(2);
    const int dim = q_in.shape(3);
    const int hkv = k_in.shape(1);
    const int tk = k_in.shape(2);
    if (q_in.shape(0) != 1 || k_in.shape(0) != 1 || v_in.shape(0) != 1) {
        throw std::invalid_argument("dlsa_sparse_attention: batch must be 1");
    }
    if (k_in.shape() != v_in.shape() || k_in.shape(3) != dim) {
        throw std::invalid_argument("dlsa_sparse_attention: k/v shapes disagree with q");
    }
    if (hkv <= 0 || hq % hkv != 0 || tk <= 0 || rows <= 0) {
        throw std::invalid_argument("dlsa_sparse_attention: bad head or length geometry");
    }
    const int group = hq / hkv;
    if (group > 8) {
        throw std::invalid_argument("dlsa_sparse_attention: at most 8 query heads per KV head");
    }
    if (dim != 32 && dim != 64 && dim != 128 && dim != 256) {
        throw std::invalid_argument("dlsa_sparse_attention: head dim must be 32, 64, 128 or 256");
    }
    if (!float_dtype(q_in.dtype()) || !float_dtype(k_in.dtype())
        || k_in.dtype() != v_in.dtype()) {
        throw std::invalid_argument("dlsa_sparse_attention: q/k/v must be f16/bf16/f32, k and v alike");
    }
    if (lists_in.ndim() != 2 || lists_in.shape(0) != rows || lists_in.dtype() != int32
        || lengths_in.ndim() != 1 || lengths_in.shape(0) != rows
        || lengths_in.dtype() != int32) {
        throw std::invalid_argument("dlsa_sparse_attention: lists [L,W] i32 and lengths [L] i32 expected");
    }
    const int width = lists_in.shape(1);
    if (width <= 0) {
        throw std::invalid_argument("dlsa_sparse_attention: empty list width");
    }

    // q, lists and lengths are small: make them row-contiguous (no-op when
    // they already are). K/V stay strided; only a non-unit last-dim stride,
    // which no cache view has, forces a copy.
    array q = contiguous(q_in);
    array lists = contiguous(lists_in);
    array lengths = contiguous(lengths_in);
    array k = k_in.strides(3) == 1 ? k_in : contiguous(k_in);
    array v = v_in.strides(3) == 1 ? v_in : contiguous(v_in);

    // Vector loads need every row start 4-element aligned.
    const int64_t esize = static_cast<int64_t>(size_of(k.dtype()));
    auto aligned = [&](const array& a) {
        return a.strides(1) % 4 == 0 && a.strides(2) % 4 == 0
            && (a.offset() / esize) % 4 == 0 && a.offset() % esize == 0;
    };
    const bool vec = aligned(k) && aligned(v);

    if (nsg <= 0) {
        nsg = auto_simdgroups(parts <= 0 ? auto_parts(rows * hkv, width, 4) : parts);
    }
    const int hs = heads_per_simdgroup(group);
    const int head_groups = group / hs;
    nsg = std::max(1, std::min(nsg, DLSA_MAX_SIMDGROUPS));
    // Every head group gets the same number of key slices
    nsg = std::max(head_groups, nsg / head_groups * head_groups);
    if (parts <= 0) {
        parts = auto_parts(rows * hkv, width, nsg);
    }
    parts = std::max(1, std::min(parts, DLSA_MAX_PARTS));
    // Keys a simdgroup loads before reducing; must divide the 32-entry chunk.
    // 2 measured best (1: 58 ms, 4: 73 ms per prefill chunk, more registers).
    constexpr int tk_unroll = 2;

    auto scale_arr = array(scale, float32);
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> targs = {
        {"QT", q.dtype()},
        {"KT", k.dtype()},
        {"D", dim},
        {"G", group},
        {"NSG", nsg},
        {"TK", tk_unroll},
        {"HS", hs},
        {"PARTS", parts},
        {"VEC", vec},
    };
    auto& ks = kernels();
    ks.ensure();
    const auto grid = std::make_tuple(hkv * nsg * 32, rows, parts);
    const auto tg = std::make_tuple(nsg * 32, 1, 1);
    if (parts == 1) {
        auto outs = (*ks.direct)(
            {q, k, v, lists, lengths, scale_arr},
            {Shape{1, hq, rows, dim}},
            {q.dtype()},
            grid,
            tg,
            targs,
            std::nullopt,
            false,
            {});
        return outs[0];
    }
    const int prow = rows * hq * parts;
    auto partials = (*ks.partial)(
        {q, k, v, lists, lengths, scale_arr},
        {Shape{prow, dim}, Shape{prow}, Shape{prow}},
        {float32, float32, float32},
        grid,
        tg,
        targs,
        std::nullopt,
        false,
        {});
    auto merged = (*ks.merge)(
        {partials[0], partials[1], partials[2]},
        {Shape{1, hq, rows, dim}},
        {q.dtype()},
        std::make_tuple(hq * 32, rows, 1),
        std::make_tuple(32, 1, 1),
        {{"OT", q.dtype()}, {"D", dim}, {"PARTS", parts}},
        std::nullopt,
        false,
        {});
    return merged[0];
}

} // namespace

std::unique_ptr<MlxArray> dlsa_sparse_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    const MlxArray& lists,
    const MlxArray& lengths,
    float scale) {
    return std::make_unique<MlxArray>(
        run(q.inner, k.inner, v.inner, lists.inner, lengths.inner, scale, 0, 0));
}

std::unique_ptr<MlxArray> dlsa_sparse_attention_with(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    const MlxArray& lists,
    const MlxArray& lengths,
    float scale,
    int32_t parts,
    int32_t simdgroups) {
    return std::make_unique<MlxArray>(run(
        q.inner, k.inner, v.inner, lists.inner, lengths.inner, scale, parts, simdgroups));
}

} // namespace mlx_cxx
