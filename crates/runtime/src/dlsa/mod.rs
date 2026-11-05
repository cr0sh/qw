// SPDX-License-Identifier: Apache-2.0
//
// D-LSA (Distilled Latent Sparse Attention) index maps for Qwen3.8-27B.
// Map format and values: https://github.com/Prefrontal-AI/latent-sparse-attention
// at 5ebea732d639576d2a7579b0de40e6252f9001f5 (Apache-2.0); see docs/dlsa.md.
//
//! Map conversion, pinned assets, resolution and loading (`docs/dlsa.md` §2, §10), and the
//! flat-design indexer core: projection, pooling, Ward, the bucket cache and selection
//! (`docs/dlsa.md` §4, §11).

mod buckets;
mod cache;
mod config;
mod convert;
mod loader;
mod project;
mod resolve;
mod select;
mod torch_pt;
mod ward;

pub use buckets::{BlockBuckets, CENTROID_NORM_FLOOR, cluster_block};
pub use cache::{DlsaBucketCache, DlsaCacheMemory, DlsaCacheStats, DlsaIndexSnapshot};
pub use config::{DlsaGeometry, DlsaIndexConfig, TOP_P_LIMIT};
pub use project::{
    NORM_EPS, concat_keys, group_queries, index_keys, index_queries, normalize_rows, to_host_f32,
};
pub use select::{DlsaSelection, expand_buckets, score_buckets_host, top_p_buckets};
pub use ward::{WARD_MAX_ROUNDS, WardOutcome, pool_keys, ward_labels, ward_labels_with_rounds};

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
mod fixtures;
#[cfg(test)]
mod indexer_tests;
#[cfg(test)]
mod tests;
