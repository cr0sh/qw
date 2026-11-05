// SPDX-License-Identifier: Apache-2.0
//
// D-LSA (Distilled Latent Sparse Attention) index maps for Qwen3.8-27B.
// Map format and values: https://github.com/Prefrontal-AI/latent-sparse-attention
// at 5ebea732d639576d2a7579b0de40e6252f9001f5 (Apache-2.0); see docs/dlsa.md.
//
//! Map conversion, pinned assets, resolution and loading. See `docs/dlsa.md` §2 and §9.

mod convert;
mod loader;
mod resolve;
mod torch_pt;

pub use convert::{METADATA_KEY, convert_pt_to_safetensors, install_dlsa_map, sha256_hex};
pub use loader::{DlsaLayerMaps, DlsaLayout, DlsaMapDtype, DlsaMaps};
pub use resolve::{
    DLSA_MAP_ENV, DlsaAsset, DlsaMapKind, dlsa_map_cache_path, dlsa_source_cache_path,
    resolve_dlsa_map_path, verify_converted_map,
};
pub use torch_pt::{TorchTensorF32, read_f32_tensor_dict};

/// Upstream commit the spec, the map format and the pinned assets refer to.
pub const UPSTREAM_COMMIT: &str = "5ebea732d639576d2a7579b0de40e6252f9001f5";
/// Width of the index space `d` (the "d2048" in the asset name is the distillation depth, not this).
pub const INDEX_DIM: usize = 64;

#[cfg(test)]
mod tests;
