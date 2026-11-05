// Copyright 2025 mlx-lm-rs authors
// Native safetensors loading and Metal 4 / turbo / paged fused-attention
// launchers for the mlx_cxx bridge. Split out of mlx_cxx_bridge.cpp.

#include "mlx_cxx_internal.h"
#include "sparse_v_sdpa.h"          // fused Sparse-V SDPA kernel.
#include "turbo4_delegated_sdpa.h"  // fused Turbo4Delegated SDPA kernel.
#include "turbo4_attention.h"       // direct symmetric-Turbo4 attention.
#include "paged_attention.h"        // fused paged-attention decode kernel (#123).
#include "paged_attention_v2.h"     // paged decode v2 + merge kernels (#898).
#include "fused_norm.h"             // fused residual-add + RMSNorm kernel (#905).
#include "fused_rope_append.h"      // fused q/k RoPE + KV-append layout (#905).

namespace mlx_cxx {

// ── Native safetensors loading ────────────────────────────────────────

// ── Metal 4 fused attention kernel scaffolding ────────────────────────────────
//
// BACKGROUND: Metal 4 (available on M5 and later) introduces TensorOps —
// first-class tensor primitives built into the Metal Shading Language (MSL).
// Key capabilities relevant to attention fusion:
//
//   - MTLTensor: Native GPU tensor resource type with explicit layout control.
//     Allows keeping intermediate Q, K, V, scores, and context vectors
//     on-chip in registers between shader stages.
//
//   - Tensor matmul / reduction: MSL built-in operators for fusing QKV
//     projection, RoPE, score computation, softmax, and output projection
//     into a SINGLE GPU dispatch, eliminating all intermediate memory writes.
//
//   - MTL4MachineLearningCommandEncoder: Encodes full ML subgraphs onto the
//     GPU command timeline, enabling the Neural Accelerator to execute an
//     entire attention layer without returning to the CPU scheduler.
//
// CURRENT STATUS:
//   - For standard SDPA, this bridge delegates to upstream MLX
//     `fast::scaled_dot_product_attention()`, which can dispatch to the
//     backend Metal/NAX fused kernels when eligible.
//   - For softcap attention (Gemma family), this bridge currently keeps the
//     existing compiled softcap fallback.
//   - window_size metadata is currently represented by explicit masks from
//     Rust model code (no dedicated on-chip band-mask kernel here yet).
//
// WHEN TO IMPLEMENT THE FULL KERNEL:
//   Requirements:
//     - macOS 26.2 or later (first macOS release supporting Metal 4)
//     - Xcode with Metal 4 SDK (WWDC25 release cycle)
//     - M5 hardware for compilation and testing
//   Reference material:
//     - WWDC25 "Metal 4 TensorOps" session
//     - WWDC25 "Accelerate ML inference with Metal 4" session
//     - https://github.com/liuliu/example_matmul_metal4  (open-source matmul example)
//     - mlx/backend/metal/steel_attention.metal  (MLX baseline for reference)
//
std::unique_ptr<MlxArray> fused_metal4_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask,
    float softcap,
    int32_t window_size,
    bool use_metal4
) {
    // window_size is currently represented by an explicit mask from Rust.
    (void)window_size;

    std::optional<mlx::core::array> mask_opt = std::nullopt;
    std::string mask_mode = "";
    if (mask) {
        auto m = mask->inner;
        // Float masks are additive masks and should match q dtype.
        // Bool/int masks must preserve original semantics.
        if (mlx::core::issubdtype(m.dtype(), mlx::core::floating) &&
            m.dtype() != q.inner.dtype()) {
            m = mlx::core::astype(m, q.inner.dtype());
        }
        mask_opt = m;
        mask_mode = "array";
    }

    // Keep existing softcap semantics (including GQA) on the Metal4 bridge path
    // until upstream MLX provides a native fused softcap SDPA variant.
    if (softcap > 0.0f) {
        auto q_heads = q.inner.shape(1);
        auto kv_heads = k.inner.shape(1);
        std::unique_ptr<MlxArray> mask_holder = nullptr;
        if (mask_opt.has_value()) {
            mask_holder = std::make_unique<MlxArray>(mask_opt.value());
        }
        const MlxArray* mask_ptr = mask_holder ? mask_holder.get() : nullptr;

        if (q_heads > kv_heads && q_heads % kv_heads == 0) {
            int32_t n_rep = static_cast<int32_t>(q_heads / kv_heads);
            return compiled_softcap_sdpa_gqa(q, k, v, scale, softcap, n_rep, mask_ptr);
        }
        return compiled_softcap_sdpa(q, k, v, scale, softcap, mask_ptr);
    }

    // Standard path: delegate to upstream MLX fast SDPA so backend kernel
    // selection (including M5 NAX full/vector kernels) remains active.
    (void)use_metal4;
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, /*window_size=*/-1, mask_mode, mask_opt
    ));
}

std::unique_ptr<MlxArray> turbo4_attention(
    const MlxArray& q_rot,
    const MlxArray& k_packed,
    const MlxArray& k_rescale,
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    float scale,
    bool causal) {
    auto out = mlxcel::turbo::turbo4_attention(
        q_rot.inner,
        k_packed.inner,
        k_rescale.inner,
        v_packed.inner,
        v_rescale.inner,
        codebook.inner,
        scale,
        causal);
    return std::make_unique<MlxArray>(std::move(out));
}

// Fused Sparse-V SDPA Metal kernel launcher. Implementation in
// `src/lib/mlx-cpp/turbo/sparse_v_sdpa.cpp`; we forward the call here so the
// new symbol shows up in the cxx-bridge ABI without bloating the bridge .cpp.
std::unique_ptr<MlxArray> turbo_sparse_v_weighted_sum(
    const MlxArray& attn_weights,
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    float threshold) {
    auto out = mlxcel::turbo::sparse_v_weighted_sum(
        attn_weights.inner,
        v_packed.inner,
        v_rescale.inner,
        codebook.inner,
        dim,
        n_rep,
        threshold);
    return std::make_unique<MlxArray>(std::move(out));
}

// Fused Turbo4Delegated cold-V weighted-sum kernel launcher.
// Implementation in `src/lib/mlx-cpp/turbo/turbo4_delegated_sdpa.cpp`; we
// forward the call here so the new symbol shows up in the cxx-bridge ABI
// without bloating the bridge .cpp. The unrotated cold weighted sum returned
// here is paired with a host-side hot V matmul; the dequantised cold V never
// materialises in global memory (the whole point).
std::unique_ptr<MlxArray> turbo4_delegated_cold_weighted_sum(
    const MlxArray& attn_weights_cold,
    const MlxArray& v_packed_cold,
    const MlxArray& v_rescale_cold,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    float threshold) {
    auto out = mlxcel::turbo::turbo4_delegated_cold_weighted_sum(
        attn_weights_cold.inner,
        v_packed_cold.inner,
        v_rescale_cold.inner,
        codebook.inner,
        dim,
        n_rep,
        threshold);
    return std::make_unique<MlxArray>(std::move(out));
}

std::unique_ptr<MlxArray> turbo4_delegated_bulk_dequant_rotated(
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    int32_t dim) {
    auto out = mlxcel::turbo::turbo4_delegated_bulk_dequant_rotated(
        v_packed.inner,
        v_rescale.inner,
        codebook.inner,
        dim);
    return std::make_unique<MlxArray>(std::move(out));
}

// Steel-attention-envelope fused Turbo4Delegated SDPA bridge.
// Wraps the launcher in `src/lib/mlx-cpp/turbo/turbo4_delegated_sdpa.cpp` and
// repackages the std::vector<mlx::core::array> return into the `cxx`-friendly
// `Turbo4DelegatedSteelOutputs` struct (cxx does not directly model multiple
// return values, but it does model unique_ptr<struct>).
std::unique_ptr<Turbo4DelegatedSteelOutputs> turbo4_delegated_steel_sdpa(
    const MlxArray& scores,
    const MlxArray& cold_packed,
    const MlxArray& cold_rescale,
    const MlxArray& hot_v,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    int32_t cold_offset,
    int32_t hot_offset,
    float threshold) {
    auto outs = mlxcel::turbo::turbo4_delegated_steel_sdpa(
        scores.inner,
        cold_packed.inner,
        cold_rescale.inner,
        hot_v.inner,
        codebook.inner,
        dim,
        n_rep,
        cold_offset,
        hot_offset,
        threshold);
    auto wrapper = std::make_unique<Turbo4DelegatedSteelOutputs>();
    wrapper->out_cold_pre = std::make_unique<MlxArray>(std::move(outs[0]));
    wrapper->out_hot      = std::make_unique<MlxArray>(std::move(outs[1]));
    return wrapper;
}

// Fused paged-attention decode kernel launcher (epic #116 Phase 6, #123).
// Implementation in `src/lib/mlx-cpp/turbo/paged_attention.cpp`; forwarded here
// so the new symbol shows up in the cxx-bridge ABI. Reads scattered KV blocks
// out of the global pool via the block table with no separate gather copy.
std::unique_ptr<MlxArray> paged_attention_decode(
    const MlxArray& q,
    const MlxArray& k_pool,
    const MlxArray& v_pool,
    const MlxArray& rows,
    const MlxArray& row_offsets,
    const MlxArray& logical_starts,
    const MlxArray& visible_lens,
    float scale,
    int32_t num_splits_override) {
    auto out = mlxcel::turbo::paged_attention_decode(
        q.inner,
        k_pool.inner,
        v_pool.inner,
        rows.inner,
        row_offsets.inner,
        logical_starts.inner,
        visible_lens.inner,
        scale,
        static_cast<int>(num_splits_override));
    return std::make_unique<MlxArray>(std::move(out));
}

int32_t paged_attention_num_splits_cap(int32_t dim) {
    return static_cast<int32_t>(
        mlxcel::turbo::paged_attention_num_splits_cap(static_cast<int>(dim)));
}

// Paged-attention decode v2 (issue #898). Implementation in
// `src/lib/mlx-cpp/turbo/paged_attention_v2.cpp` (partial kernel) and
// `paged_attention_v2_merge.cpp` (merge kernel); forwarded here so the symbols
// show up in the cxx-bridge ABI. Two outputs each, so both use out-params.
void paged_attention_decode_v2_partial(
    const MlxArray& q,
    const MlxArray& k_pool,
    const MlxArray& v_pool,
    const MlxArray& indices,
    const MlxArray& indptr,
    const MlxArray& last_page_len,
    const MlxArray& first_page_offset,
    const MlxArray& request_indices,
    const MlxArray& kv_tile_indices,
    const MlxArray& params,
    float scale,
    std::unique_ptr<MlxArray>& partial_v_out,
    std::unique_ptr<MlxArray>& lse_out) {
    auto outs = mlxcel::turbo::paged_attention_decode_v2_partial(
        q.inner,
        k_pool.inner,
        v_pool.inner,
        indices.inner,
        indptr.inner,
        last_page_len.inner,
        first_page_offset.inner,
        request_indices.inner,
        kv_tile_indices.inner,
        params.inner,
        scale);
    partial_v_out = std::make_unique<MlxArray>(std::move(outs[0]));
    lse_out = std::make_unique<MlxArray>(std::move(outs[1]));
}

void paged_attention_merge_states(
    const MlxArray& v_in,
    const MlxArray& lse_in,
    const MlxArray& o_indptr,
    std::unique_ptr<MlxArray>& v_out,
    std::unique_ptr<MlxArray>& lse_out) {
    auto outs = mlxcel::turbo::paged_attention_merge_states(
        v_in.inner, lse_in.inner, o_indptr.inner);
    v_out = std::make_unique<MlxArray>(std::move(outs[0]));
    lse_out = std::make_unique<MlxArray>(std::move(outs[1]));
}

int32_t paged_attention_v2_q_heads_per_cta(int32_t dim, int32_t n_rep) {
    return static_cast<int32_t>(mlxcel::turbo::paged_attention_v2_q_heads_per_cta(
        static_cast<int>(dim), static_cast<int>(n_rep)));
}

int32_t paged_attention_v2_num_warps(int32_t dim, int32_t q_heads_per_cta) {
    return static_cast<int32_t>(mlxcel::turbo::paged_attention_v2_num_warps(
        static_cast<int>(dim), static_cast<int>(q_heads_per_cta)));
}

// Fused residual-add + RMSNorm kernel launcher (issue #905). Implementation in
// `src/lib/mlx-cpp/turbo/fused_norm.cpp`; forwarded here so the symbol shows up
// in the cxx-bridge ABI. Two outputs because MLX arrays are immutable from the
// graph's perspective, so the in-place residual update has to be expressed as a
// value rather than a mutation.
void fused_add_rms_norm(
    const MlxArray& x,
    const MlxArray& residual,
    const MlxArray& weight,
    float eps,
    float weight_bias,
    std::unique_ptr<MlxArray>& normed_out,
    std::unique_ptr<MlxArray>& new_residual_out) {
    auto outs = mlxcel::turbo::fused_add_rms_norm(
        x.inner, residual.inner, weight.inner, eps, weight_bias);
    normed_out = std::make_unique<MlxArray>(std::move(outs[0]));
    new_residual_out = std::make_unique<MlxArray>(std::move(outs[1]));
}

bool fused_add_rms_norm_available() {
    return mlxcel::turbo::fused_add_rms_norm_available();
}

// Fused q/k RoPE + KV-append-layout kernel launcher (issue #905).
// Implementation in `src/lib/mlx-cpp/turbo/fused_rope_append.cpp`.
void fused_rope_qk_append(
    const MlxArray& qkv,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    float rope_scale,
    bool traditional,
    int32_t positions_base,
    int32_t dest_layout,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out) {
    auto outs = mlxcel::turbo::fused_rope_qk_append(
        qkv.inner,
        static_cast<int>(num_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        static_cast<int>(rope_dims),
        rope_base,
        rope_scale,
        traditional,
        static_cast<int>(positions_base),
        static_cast<int>(dest_layout));
    q_out = std::make_unique<MlxArray>(std::move(outs[0]));
    k_out = std::make_unique<MlxArray>(std::move(outs[1]));
    v_out = std::make_unique<MlxArray>(std::move(outs[2]));
}

bool fused_rope_qk_append_available() {
    return mlxcel::turbo::fused_rope_qk_append_available();
}

std::unique_ptr<MlxLoadedWeights> mlx_load_safetensors(rust::Str path) {
    std::string path_str(path.data(), path.size());
    auto [weights_map, metadata] = mlx::core::load_safetensors(path_str);

    auto result = std::make_unique<MlxLoadedWeights>();
    result->names.reserve(weights_map.size());
    result->arrays.reserve(weights_map.size());

    for (auto& [name, arr] : weights_map) {
        result->names.push_back(std::move(name));
        result->arrays.push_back(std::make_unique<MlxArray>(std::move(arr)));
    }

    return result;
}

size_t loaded_weights_len(const MlxLoadedWeights& w) {
    return w.names.size();
}

rust::String loaded_weights_name(const MlxLoadedWeights& w, size_t index) {
    return rust::String(w.names.at(index));
}

std::unique_ptr<MlxArray> loaded_weights_take(MlxLoadedWeights& w, size_t index) {
    return std::move(w.arrays.at(index));
}

// steel SDPA output takers. The cxx bridge does not directly
// model destructuring a struct returned over the FFI boundary; we expose two
// simple `move-out` accessors that the Rust caller invokes once each. After
// both calls the struct's `unique_ptr` slots are empty (a second call would
// return a null `unique_ptr`, which is harmless because the Rust side drops
// the struct after the second take).
std::unique_ptr<MlxArray> steel_outputs_take_cold(Turbo4DelegatedSteelOutputs& o) {
    return std::move(o.out_cold_pre);
}

std::unique_ptr<MlxArray> steel_outputs_take_hot(Turbo4DelegatedSteelOutputs& o) {
    return std::move(o.out_hot);
}

// ── Test-only compiled-vs-eager probe ─────────────────────────────────────────
//
// Runs one fixed graph on (a, b) eagerly (mode 0), compiled (1) or compiled
// shapeless (2). The Metal compiled-kernel overlay used to cast comparison
// operands to bool, so a compiled `greater_equal` returned the wrong mask; no
// production code compiles a comparison any more, so this is the only way to
// pin the overlay fix (upstream mlxcel d702f4e5, #2325). The overlay was retired
// with the MLX pin at 7ca40033 (ml-explore/mlx#4351 `cast_to` for fused
// AsType); graphs 3 and 4 mix input dtypes inside one fused kernel to pin that
// upstream emission promotes them correctly without it.
// Graph 0: where(a < b, a, b).
// Graph 1: min-p filter, where(softmax(a) >= 0.3 * max(softmax(a)), a, lowest).
// Graph 2: where(isnan(a), 7, a)  (isnan lowers to not_equal(a, a)).
// Graph 3: a * float32(b) + bfloat16(0.5)  (half x f32 array, + bf16 scalar).
// Graph 4: (a * int32(b) - a) * float32(1.5)  (half x int32 array, f32 scalar).
std::unique_ptr<MlxArray> compile_dtype_probe_for_test(
    int32_t graph, const MlxArray& a, const MlxArray& b, int32_t mode) {
    namespace mx = mlx::core;
    auto fn = [graph](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        const auto& x = in[0];
        const auto& y = in[1];
        switch (graph) {
            case 0:
                return {mx::where(mx::less(x, y), x, y)};
            case 1: {
                auto probs = mx::softmax(x, -1);
                auto thr = mx::multiply(mx::max(probs, -1, true), mx::array(0.3f));
                return {mx::where(
                    mx::greater_equal(probs, thr),
                    x,
                    mx::array(std::numeric_limits<float>::lowest(), x.dtype()))};
            }
            case 2:
                return {mx::where(mx::isnan(x), mx::array(7.0f, x.dtype()), x)};
            case 3:
                return {mx::add(
                    mx::multiply(x, mx::astype(y, mx::float32)),
                    mx::array(0.5f, mx::bfloat16))};
            case 4:
                return {mx::multiply(
                    mx::subtract(mx::multiply(x, mx::astype(y, mx::int32)), x),
                    mx::array(1.5f, mx::float32))};
            default:
                throw std::invalid_argument("compile_dtype_probe_for_test: unknown graph");
        }
    };
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> run = fn;
    if (mode == 1) {
        run = mx::compile(fn, /*shapeless=*/false);
    } else if (mode == 2) {
        run = mx::compile(fn, /*shapeless=*/true);
    } else if (mode != 0) {
        throw std::invalid_argument("compile_dtype_probe_for_test: mode must be 0, 1 or 2");
    }
    auto out = run({a.inner, b.inner});
    mx::eval(out[0]);
    return std::make_unique<MlxArray>(std::move(out[0]));
}

}  // namespace mlx_cxx
