// SPDX-License-Identifier: Apache-2.0
//
// D-LSA fused sparse gather-attention (K2, docs/dlsa.md §14.1): every query row
// attends over its own read list of cached keys only, GQA folded so each
// gathered K/V row is loaded once per KV head group. JIT Metal kernels
// (mlx::core::fast::metal_kernel) in mlx_cxx_dlsa_attention.cpp.

#pragma once

#include <cstdint>
#include <memory>

namespace mlx_cxx {

struct MlxArray;

// q [1,Hq,L,D] (f16/bf16/f32), k/v [1,Hkv,T,D] (same dtype as each other, may
// be strided views of a larger cache buffer: never copied), lists [L,W] i32
// (any order, no duplicates, -1 pad past lengths[r]), lengths [L] i32.
// Returns [1,Hq,L,D] in q's dtype. Head h reads KV head h / (Hq/Hkv).
std::unique_ptr<MlxArray> dlsa_sparse_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    const MlxArray& lists,
    const MlxArray& lengths,
    float scale);

// The same with explicit launch knobs: parts 0 = auto, 1 = one pass,
// >1 = split-K partitions of each list (plus a merge kernel); simdgroups
// 0 = auto, else simdgroups per threadgroup (1..16).
std::unique_ptr<MlxArray> dlsa_sparse_attention_with(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    const MlxArray& lists,
    const MlxArray& lengths,
    float scale,
    int32_t parts,
    int32_t simdgroups);

} // namespace mlx_cxx
