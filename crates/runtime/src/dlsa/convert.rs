// SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};

use super::torch_pt::read_f32_tensor_dict;
use super::{INDEX_DIM, UPSTREAM_COMMIT};

/// Safetensors metadata key holding the JSON info object.
pub const METADATA_KEY: &str = "qw_dlsa_map";

/// Lower-case hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Parse a released `<layer>.W_Q_idx` / `<layer>.W_K_idx` key.
fn parse_key(key: &str) -> Result<(usize, bool)> {
    let (layer, which) = key
        .split_once('.')
        .with_context(|| format!("unexpected map key `{key}`"))?;
    let layer: usize = layer
        .parse()
        .with_context(|| format!("unexpected map key `{key}`: layer is not an integer"))?;
    match which {
        "W_Q_idx" => Ok((layer, true)),
        "W_K_idx" => Ok((layer, false)),
        _ => anyhow::bail!("unexpected map key `{key}`"),
    }
}

/// Convert the bytes of a released torch `.pt` map into the bytes of QW's safetensors map file.
///
/// Output: tensors `layers.<i>.w_q_idx` / `layers.<i>.w_k_idx`, F32, `[input_dim, 64]` row-major,
/// plus metadata (`format`, `source_asset`, `source_sha256`, `upstream_commit`, `index_dim`,
/// `input_dim`, `layers`). The output is deterministic.
pub fn convert_pt_to_safetensors(pt_bytes: &[u8], source_asset: &str) -> Result<Vec<u8>> {
    let tensors = read_f32_tensor_dict(pt_bytes)?;
    ensure!(!tensors.is_empty(), "the map file holds no tensors");
    let mut by_layer: BTreeMap<usize, [Option<Vec<u8>>; 2]> = BTreeMap::new();
    let mut input_dim = None;
    for (key, t) in tensors {
        let (layer, is_q) = parse_key(&key)?;
        ensure!(
            t.shape.len() == 2 && t.shape[1] == INDEX_DIM,
            "`{key}` has shape {:?}, expected [input_dim, {INDEX_DIM}]",
            t.shape
        );
        let dim = *input_dim.get_or_insert(t.shape[0]);
        ensure!(
            t.shape[0] == dim,
            "`{key}` input dim {} differs from {dim}",
            t.shape[0]
        );
        let slot = &mut by_layer.entry(layer).or_default()[usize::from(!is_q)];
        ensure!(slot.is_none(), "duplicate map key `{key}`");
        *slot = Some(t.bytes);
    }
    let input_dim = input_dim.unwrap_or(0);
    let mut names = Vec::new();
    let mut data = Vec::new();
    for (layer, [q, k]) in by_layer.iter() {
        let (Some(q), Some(k)) = (q, k) else {
            anyhow::bail!("layer {layer} lacks one of W_Q_idx / W_K_idx");
        };
        names.push(format!("layers.{layer}.w_q_idx"));
        data.push(q.as_slice());
        names.push(format!("layers.{layer}.w_k_idx"));
        data.push(k.as_slice());
    }
    let views = names
        .iter()
        .zip(&data)
        .map(|(n, d)| {
            Ok((
                n.as_str(),
                TensorView::new(Dtype::F32, vec![input_dim, INDEX_DIM], d)?,
            ))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    // One metadata key holding a JSON object: safetensors writes its metadata map in hash order,
    // so several keys would make the output (and its pinned sha256) nondeterministic.
    let info = serde_json::json!({
        "format": "qw-dlsa-map",
        "source_asset": source_asset,
        "source_sha256": sha256_hex(pt_bytes),
        "upstream_commit": UPSTREAM_COMMIT,
        "index_dim": INDEX_DIM,
        "input_dim": input_dim,
        "layers": by_layer.keys().collect::<Vec<_>>(),
    });
    let metadata = std::collections::HashMap::from([(METADATA_KEY.to_owned(), info.to_string())]);
    Ok(safetensors::serialize(views, Some(metadata))?)
}

/// Verify the upstream `.pt` bytes of `asset`, convert them, verify the pinned sha256 of the
/// result and write it atomically to `destination`.
pub fn install_dlsa_map(
    asset: &super::DlsaAsset,
    pt_bytes: &[u8],
    destination: &std::path::Path,
) -> Result<()> {
    asset.verify_source(pt_bytes)?;
    let converted = convert_pt_to_safetensors(pt_bytes, asset.source_name)?;
    let actual = sha256_hex(&converted);
    ensure!(
        actual == asset.converted_sha256,
        "{}: converted map sha256 {actual} differs from the pinned {}",
        asset.source_name,
        asset.converted_sha256
    );
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut partial = destination.as_os_str().to_owned();
    partial.push(".qw-part");
    let partial = std::path::PathBuf::from(partial);
    std::fs::write(&partial, &converted)
        .with_context(|| format!("cannot write {}", partial.display()))?;
    std::fs::rename(&partial, destination)?;
    Ok(())
}
