// SPDX-License-Identifier: Apache-2.0
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::convert::sha256_hex;

/// Environment override for the converted map path.
pub const DLSA_MAP_ENV: &str = "QW_DLSA_MAP_PATH";

const RELEASE_BASE_URL: &str =
    "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1";

/// Which released Qwen3.8-27B map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlsaMapKind {
    /// The distilled map the authors serve.
    Distilled,
    /// The training-free PCA step-0 map (a no-distillation baseline).
    PcaStep0,
}

/// A pinned upstream asset and the converted file QW derives from it.
#[derive(Debug, Clone, Copy)]
pub struct DlsaAsset {
    pub kind: DlsaMapKind,
    /// File name of the upstream `.pt` asset in release `d-lsa-weights-v1`.
    pub source_name: &'static str,
    pub source_size: u64,
    pub source_sha256: &'static str,
    /// File name of the converted safetensors file in the cache.
    pub converted_name: &'static str,
    /// sha256 of the converted file (the conversion is deterministic).
    pub converted_sha256: &'static str,
}

impl DlsaMapKind {
    pub const fn asset(self) -> DlsaAsset {
        match self {
            Self::Distilled => DlsaAsset {
                kind: self,
                source_name: "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt",
                source_size: 6_302_911,
                source_sha256: "24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b",
                converted_name: "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors",
                converted_sha256: "00200004e8ab6cadb2738dbc466230b542fecc592e138be6c31f29d16e0462e1",
            },
            Self::PcaStep0 => DlsaAsset {
                kind: self,
                source_name: "qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.pt",
                source_size: 6_303_063,
                source_sha256: "baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65",
                converted_name: "qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.safetensors",
                converted_sha256: "03dbc21117cb96c6c18f13bc4eacb6c44eb5733e9787d5c5b127ab365362f9fb",
            },
        }
    }
}

impl DlsaAsset {
    pub fn source_url(&self) -> String {
        format!("{RELEASE_BASE_URL}/{}", self.source_name)
    }

    /// Check the downloaded upstream `.pt` bytes against the pinned size and sha256.
    pub fn verify_source(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() as u64 != self.source_size {
            bail!(
                "{}: size {} differs from the pinned {} bytes; refusing to convert (re-download it)",
                self.source_name,
                bytes.len(),
                self.source_size
            );
        }
        let actual = sha256_hex(bytes);
        if actual != self.source_sha256 {
            bail!(
                "{}: sha256 mismatch (got {actual}, pinned {}); the download is corrupt or upstream changed it",
                self.source_name,
                self.source_sha256
            );
        }
        Ok(())
    }
}

/// `~/.cache/qw/dlsa`, next to `~/.cache/qw/models`.
fn cache_dir(home: &Path) -> PathBuf {
    home.join(".cache/qw/dlsa")
}

/// Cache path of the converted map for `kind`.
pub fn dlsa_map_cache_path(home: &Path, kind: DlsaMapKind) -> PathBuf {
    cache_dir(home).join(kind.asset().converted_name)
}

/// Upstream `.pt` cache location.
pub fn dlsa_source_cache_path(home: &Path, kind: DlsaMapKind) -> PathBuf {
    cache_dir(home).join(kind.asset().source_name)
}

/// Check a converted map file against the pinned sha256 of `kind`.
pub fn verify_converted_map(path: &Path, kind: DlsaMapKind) -> Result<()> {
    let asset = kind.asset();
    let bytes =
        std::fs::read(path).with_context(|| format!("cannot read D-LSA map {}", path.display()))?;
    let actual = sha256_hex(&bytes);
    if actual != asset.converted_sha256 {
        bail!(
            "D-LSA map {} has sha256 {actual}, expected {}; delete it and re-run `qw download --dlsa-map`",
            path.display(),
            asset.converted_sha256
        );
    }
    Ok(())
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

pub(super) fn resolve_dir(
    kind: DlsaMapKind,
    cli_override: Option<&Path>,
    env_override: Option<&Path>,
    home: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = cli_override.or(env_override) {
        // A user-supplied file is trusted as is; the loader still validates its structure.
        if !path.is_file() {
            bail!("D-LSA map {} does not exist", path.display());
        }
        return Ok(path.to_path_buf());
    }
    let Some(home) = home else {
        bail!("HOME is not set");
    };
    let path = dlsa_map_cache_path(home, kind);
    if !path.is_file() {
        bail!(
            "D-LSA map {} not found; run `qw download --dlsa-map{}` or set {DLSA_MAP_ENV}",
            path.display(),
            if kind == DlsaMapKind::PcaStep0 {
                " --dlsa-pca"
            } else {
                ""
            }
        );
    }
    // The pinned cache copy must match its checksum. The env/CLI overrides skip this.
    verify_converted_map(&path, kind)?;
    Ok(path)
}

/// Resolve the converted map: CLI override, then `QW_DLSA_MAP_PATH`, then the verified cache file.
pub fn resolve_dlsa_map_path(kind: DlsaMapKind, cli_override: Option<&Path>) -> Result<PathBuf> {
    let env_override = non_empty_env_path(DLSA_MAP_ENV);
    let home = non_empty_env_path("HOME");
    resolve_dir(kind, cli_override, env_override.as_deref(), home.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("qw-dlsa-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn cache_path_convention() {
        let p = dlsa_map_cache_path(Path::new("/h"), DlsaMapKind::Distilled);
        assert_eq!(
            p,
            Path::new(
                "/h/.cache/qw/dlsa/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"
            )
        );
        assert!(
            DlsaMapKind::Distilled
                .asset()
                .source_url()
                .ends_with("d2048.pt")
        );
    }

    #[test]
    fn override_precedence_and_missing_errors() {
        let dir = tmp("res");
        let a = dir.join("a.safetensors");
        let b = dir.join("b.safetensors");
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"y").unwrap();
        let got = resolve_dir(DlsaMapKind::Distilled, Some(&a), Some(&b), None).unwrap();
        assert_eq!(got, a);
        let got = resolve_dir(DlsaMapKind::Distilled, None, Some(&b), None).unwrap();
        assert_eq!(got, b);
        let err = resolve_dir(DlsaMapKind::Distilled, Some(&dir.join("no")), None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not exist"), "{err}");
        let err = resolve_dir(DlsaMapKind::Distilled, None, None, Some(&dir))
            .unwrap_err()
            .to_string();
        assert!(err.contains("qw download --dlsa-map"), "{err}");
    }

    #[test]
    fn cached_file_checksum_mismatch_is_reported() {
        let dir = tmp("sum");
        let path = dlsa_map_cache_path(&dir, DlsaMapKind::Distilled);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"corrupt").unwrap();
        let err = resolve_dir(DlsaMapKind::Distilled, None, None, Some(&dir))
            .unwrap_err()
            .to_string();
        assert!(err.contains("sha256"), "{err}");
    }

    #[test]
    fn source_verification() {
        let asset = DlsaMapKind::PcaStep0.asset();
        let err = asset.verify_source(b"nope").unwrap_err().to_string();
        assert!(err.contains("size"), "{err}");
    }
}
