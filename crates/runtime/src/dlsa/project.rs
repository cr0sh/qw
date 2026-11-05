// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/core/index_space.py (position_free, regroup, project, rownorm).
//
//! Index key / index query projection in MLX ops (`docs/dlsa.md` §4.1).
//!
//! Inputs are a full-attention layer's heads after q_norm/k_norm and before RoPE (dims
//! `rotary_dim..head_dim` are the same after RoPE), laid out `[1, L, H, D]` as in
//! `Qwen3NextAttention::forward_impl`, or `[L, H, D]`. Outputs are `[L, ...]` f32.

use anyhow::{Result, bail, ensure};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::DlsaGeometry;

/// Norm floor of `normalize` (`index_space.rownorm`, `const.Math.NORM_EPS`).
pub const NORM_EPS: f32 = 1e-12;

/// `[L, heads, head_dim]` view of `x` (`[1, L, H, D]` or `[L, H, D]`), and L.
fn rows_heads(x: &MlxArray, heads: usize, head_dim: usize, what: &str) -> Result<i32> {
    let shape = mlxcel_core::array_shape(x);
    let ok = match shape.len() {
        3 => true,
        4 => shape[0] == 1,
        _ => false,
    };
    ensure!(
        ok,
        "{what} must be [1, L, {heads}, {head_dim}] or [L, {heads}, {head_dim}], got {shape:?}"
    );
    let n = shape.len();
    ensure!(
        shape[n - 2] as usize == heads && shape[n - 1] as usize == head_dim,
        "{what} must have {heads} heads of {head_dim} dims, got {shape:?}"
    );
    Ok(shape[n - 3])
}

/// Each KV head's position-free dims side by side, head-major: `[L, kv_heads * free_dim]` f32.
pub fn concat_keys(keys: &MlxArray, geom: &DlsaGeometry) -> Result<UniquePtr<MlxArray>> {
    geom.validate()?;
    let l = rows_heads(keys, geom.kv_heads, geom.head_dim, "keys")?;
    let (h, d, r) = (
        geom.kv_heads as i32,
        geom.head_dim as i32,
        geom.rotary_dim as i32,
    );
    let k = mlxcel_core::reshape(keys, &[l, h, d]);
    let free = mlxcel_core::slice(&k, &[0, 0, r], &[l, h, d]);
    let free = mlxcel_core::astype(&free, dtype::FLOAT32);
    Ok(mlxcel_core::reshape(&free, &[l, h * (d - r)]))
}

/// Each KV group's query heads summed over their position-free dims (in f32), groups
/// concatenated: `[L, kv_heads * free_dim]` f32, not normalized.
pub fn group_queries(queries: &MlxArray, geom: &DlsaGeometry) -> Result<UniquePtr<MlxArray>> {
    geom.validate()?;
    let l = rows_heads(queries, geom.query_heads, geom.head_dim, "queries")?;
    let (hq, hkv, g, d, r) = (
        geom.query_heads as i32,
        geom.kv_heads as i32,
        geom.group() as i32,
        geom.head_dim as i32,
        geom.rotary_dim as i32,
    );
    let q = mlxcel_core::reshape(queries, &[l, hq, d]);
    let free = mlxcel_core::slice(&q, &[0, 0, r], &[l, hq, d]);
    // The served path sums the group in fp32 (`docs/dlsa.md` §4.1)
    let free = mlxcel_core::astype(&free, dtype::FLOAT32);
    let grouped = mlxcel_core::reshape(&free, &[l, hkv, g, d - r]);
    let summed = mlxcel_core::sum_axis(&grouped, 2, false);
    Ok(mlxcel_core::reshape(&summed, &[l, hkv * (d - r)]))
}

/// `v / max(||v||_2, 1e-12)` over the last axis of a 2-D array, in f32.
pub fn normalize_rows(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    let squares = mlxcel_core::square(&x);
    let norms = mlxcel_core::sqrt(&mlxcel_core::sum_axis(&squares, -1, true));
    let floor = mlxcel_core::full_f32(&[1, 1], NORM_EPS, dtype::FLOAT32);
    mlxcel_core::divide(&x, &mlxcel_core::maximum(&norms, &floor))
}

/// `normalize(x @ W)`: `x` is cast to the map's dtype (f32 for parity runs, bf16 as upstream
/// serves), the product is taken in that dtype and normalized in f32.
fn project(x: &MlxArray, map: &MlxArray, input_dim: usize) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(map);
    ensure!(
        shape.len() == 2 && shape[0] as usize == input_dim,
        "an index map must be [{input_dim}, index_dim], got {shape:?}"
    );
    let map_dtype = mlxcel_core::array_dtype(map);
    if map_dtype != dtype::FLOAT32 && map_dtype != dtype::BFLOAT16 {
        bail!("an index map must be f32 or bf16 (`DlsaMapDtype`), got dtype {map_dtype}");
    }
    let x = mlxcel_core::astype(x, map_dtype);
    let product = mlxcel_core::matmul(&x, map);
    Ok(normalize_rows(&product))
}

/// Index keys `normalize(concat_keys(k) @ W_K_idx)`: `[L, index_dim]` f32 unit rows.
pub fn index_keys(
    keys: &MlxArray,
    w_k_idx: &MlxArray,
    geom: &DlsaGeometry,
) -> Result<UniquePtr<MlxArray>> {
    let concat = concat_keys(keys, geom)?;
    project(&concat, w_k_idx, geom.input_dim())
}

/// Index queries `normalize(group_queries(q) @ W_Q_idx)`: `[L, index_dim]` f32 unit rows.
pub fn index_queries(
    queries: &MlxArray,
    w_q_idx: &MlxArray,
    geom: &DlsaGeometry,
) -> Result<UniquePtr<MlxArray>> {
    let grouped = group_queries(queries, geom)?;
    project(&grouped, w_q_idx, geom.input_dim())
}

/// Evaluate an array and copy it to the host as f32 (any float dtype is upcast).
pub fn to_host_f32(x: &MlxArray) -> Vec<f32> {
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    mlxcel_core::eval(&x);
    mlxcel_core::array_to_raw_bytes(&x)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}
