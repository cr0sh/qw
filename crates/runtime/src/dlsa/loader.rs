// SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::{MlxArray, UniquePtr};
use safetensors::{Dtype, SafeTensors};

use super::INDEX_DIM;
use super::convert::METADATA_KEY;
use crate::qwen3_5::Qwen35Config;

/// What a checkpoint must look like for a map to apply: which layers are full attention and the
/// width of the position-free key concatenation the maps read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlsaLayout {
    /// 0-based decoder indices of the full-attention layers, ascending.
    pub full_attention_layers: Vec<usize>,
    /// `kv_heads * (head_dim - rotary_dim)`: 768 for Qwen3.8-27B.
    pub input_dim: usize,
}

impl DlsaLayout {
    pub fn from_qwen35_config(config: &Qwen35Config) -> Result<Self> {
        let n = config.to_qwen3next_config();
        let rotary_dim = (n.head_dim as f32 * n.partial_rotary_factor) as usize;
        ensure!(
            rotary_dim < n.head_dim,
            "head_dim {} leaves no position-free dims (rotary_dim {rotary_dim})",
            n.head_dim
        );
        let full_attention_layers: Vec<usize> = (0..config.num_hidden_layers)
            .filter(|&i| !config.is_linear_layer(i))
            .collect();
        ensure!(
            !full_attention_layers.is_empty(),
            "the checkpoint has no full-attention layers"
        );
        Ok(Self {
            full_attention_layers,
            input_dim: n.num_key_value_heads * (n.head_dim - rotary_dim),
        })
    }
}

/// Dtype the maps are held in on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlsaMapDtype {
    /// What upstream serves (`docs/dlsa.md` §4.1): bf16 maps, bf16 products.
    Bf16,
    /// The stored precision; the reference path (fixtures, parity checks).
    F32,
}

/// The index projections of one full-attention layer, each `[input_dim, 64]`, used as `x @ W`.
pub struct DlsaLayerMaps {
    pub w_q_idx: UniquePtr<MlxArray>,
    pub w_k_idx: UniquePtr<MlxArray>,
}

/// All per-layer maps of one converted file, keyed by 0-based decoder layer index.
pub struct DlsaMaps {
    pub layers: BTreeMap<usize, DlsaLayerMaps>,
    pub input_dim: usize,
    pub dtype: DlsaMapDtype,
    /// File metadata (source sha256, upstream commit, ...).
    pub metadata: BTreeMap<String, String>,
}

impl DlsaMaps {
    pub fn layer(&self, layer_idx: usize) -> Option<&DlsaLayerMaps> {
        self.layers.get(&layer_idx)
    }

    /// Load a converted map file and validate it against `layout`.
    pub fn load(path: &Path, layout: &DlsaLayout, dtype: DlsaMapDtype) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("cannot read D-LSA map {}", path.display()))?;
        Self::from_bytes(&bytes, layout, dtype)
            .with_context(|| format!("invalid D-LSA map {}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8], layout: &DlsaLayout, dtype: DlsaMapDtype) -> Result<Self> {
        let (_, header) = SafeTensors::read_metadata(bytes)?;
        let info: serde_json::Value = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get(METADATA_KEY))
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null);
        ensure!(
            info["format"] == "qw-dlsa-map",
            "not a QW D-LSA map file (missing `{METADATA_KEY}` metadata); convert the upstream .pt with `qw download --dlsa-map`"
        );
        ensure!(
            info["index_dim"] == INDEX_DIM,
            "map index_dim {} != {INDEX_DIM}",
            info["index_dim"]
        );
        let metadata: BTreeMap<String, String> = info
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            v.as_str().map_or_else(|| v.to_string(), str::to_owned),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let st = SafeTensors::deserialize(bytes)?;
        let mut names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
        names.sort();
        let mut expected: Vec<String> = layout
            .full_attention_layers
            .iter()
            .flat_map(|l| [format!("layers.{l}.w_k_idx"), format!("layers.{l}.w_q_idx")])
            .collect();
        expected.sort();
        if names != expected {
            let missing: Vec<_> = expected.iter().filter(|n| !names.contains(n)).collect();
            let extra: Vec<_> = names.iter().filter(|n| !expected.contains(n)).collect();
            bail!(
                "map layers do not match the checkpoint's full-attention layers {:?}: missing {missing:?}, unexpected {extra:?}",
                layout.full_attention_layers
            );
        }
        let load = |name: String| -> Result<UniquePtr<MlxArray>> {
            let t = st.tensor(&name)?;
            ensure!(
                t.dtype() == Dtype::F32,
                "`{name}` is {:?}, expected F32",
                t.dtype()
            );
            ensure!(
                t.shape() == [layout.input_dim, INDEX_DIM],
                "`{name}` has shape {:?}, expected [{}, {INDEX_DIM}] (kv_heads * position-free dims, index_dim)",
                t.shape(),
                layout.input_dim
            );
            let shape = [layout.input_dim as i32, INDEX_DIM as i32];
            let f32_arr = mlxcel_core::from_bytes(t.data(), &shape, mlxcel_core::dtype::FLOAT32);
            Ok(match dtype {
                DlsaMapDtype::F32 => f32_arr,
                DlsaMapDtype::Bf16 => mlxcel_core::astype(&f32_arr, mlxcel_core::dtype::BFLOAT16),
            })
        };
        let mut layers = BTreeMap::new();
        for &l in &layout.full_attention_layers {
            layers.insert(
                l,
                DlsaLayerMaps {
                    w_q_idx: load(format!("layers.{l}.w_q_idx"))?,
                    w_k_idx: load(format!("layers.{l}.w_k_idx"))?,
                },
            );
        }
        Ok(Self {
            layers,
            input_dim: layout.input_dim,
            dtype,
            metadata,
        })
    }
}
