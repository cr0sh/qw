use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnedArtifactRole {
    Target,
    Mtp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedArtifact {
    pub role: PinnedArtifactRole,
    pub source_path: &'static str,
    pub relative_path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const PINNED_REPOSITORY: &str = "unsloth/Qwen3.8-27B-GGUF";
pub const PINNED_REVISION: &str = "4ca720788d1e01f1bff70c033e0d0028fd02e502";
pub const PINNED_CACHE_RELATIVE_DIR: &str = ".cache/qw/models/unsloth/Qwen3.8-27B-GGUF";

pub const PINNED_TARGET: PinnedArtifact = PinnedArtifact {
    role: PinnedArtifactRole::Target,
    source_path: "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    relative_path: "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    size: 17_559_178_144,
    sha256: "3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e",
};

pub const PINNED_MTP: PinnedArtifact = PinnedArtifact {
    role: PinnedArtifactRole::Mtp,
    source_path: "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
    relative_path: "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
    size: 1_369_590_656,
    sha256: "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e",
};

pub const PINNED_ARTIFACTS: [PinnedArtifact; 2] = [PINNED_TARGET, PINNED_MTP];

pub fn pinned_model_dir(home: &Path) -> PathBuf {
    home.join(PINNED_CACHE_RELATIVE_DIR)
}

pub fn resolve_pinned_model_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .context("HOME is not set; cannot resolve the pinned model cache")?;
    Ok(pinned_model_dir(&home))
}

pub fn verify_target_file(path: &Path) -> Result<()> {
    verify_artifact_file(path, PINNED_TARGET)
}

pub fn verify_mtp_file(path: &Path) -> Result<()> {
    verify_artifact_file(path, PINNED_MTP)
}

pub fn verify_artifact_file(path: &Path, artifact: PinnedArtifact) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat pinned artifact {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "pinned artifact is not a regular non-symlink file: {}",
        path.display()
    );
    ensure!(
        metadata.len() == artifact.size,
        "pinned artifact {} has {} bytes; expected {}",
        path.display(),
        metadata.len(),
        artifact.size
    );
    let digest = crate::sha256::sha256_file(path)
        .with_context(|| format!("failed to hash pinned artifact {}", path.display()))?;
    ensure!(
        digest == artifact.sha256,
        "pinned artifact {} SHA-256 {digest} does not match {}",
        path.display(),
        artifact.sha256
    );
    Ok(())
}

pub(crate) fn verify_pair_directory(root: &Path) -> Result<(PathBuf, PathBuf)> {
    let metadata = std::fs::symlink_metadata(root)
        .with_context(|| format!("failed to stat pinned model directory {}", root.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "pinned model path is not a non-symlink directory: {}",
        root.display()
    );
    let target = root.join(PINNED_TARGET.relative_path);
    let mtp = root.join(PINNED_MTP.relative_path);
    verify_target_file(&target)?;
    verify_mtp_file(&mtp)?;
    Ok((target, mtp))
}

pub fn io_verify_artifact_file(path: &Path, artifact: PinnedArtifact) -> Result<(), Error> {
    verify_artifact_file(path, artifact).map_err(|error| Error::new(ErrorKind::InvalidData, error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_resolver_has_no_cli_or_environment_precedence() {
        assert_eq!(
            pinned_model_dir(Path::new("/home/test")),
            Path::new("/home/test/.cache/qw/models/unsloth/Qwen3.8-27B-GGUF")
        );
        assert_eq!(PINNED_ARTIFACTS, [PINNED_TARGET, PINNED_MTP]);
    }

    #[test]
    fn artifact_roles_and_hashes_are_closed() {
        assert_eq!(PINNED_TARGET.role, PinnedArtifactRole::Target);
        assert_eq!(PINNED_MTP.role, PinnedArtifactRole::Mtp);
        assert_eq!(PINNED_TARGET.sha256.len(), 64);
        assert_eq!(PINNED_MTP.sha256.len(), 64);
    }

    #[test]
    fn payload_mutation_is_rejected_by_sha_verification() {
        let path = std::env::temp_dir().join(format!(
            "qw-pinned-sha-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"abc").expect("write fixture");
        let artifact = PinnedArtifact {
            role: PinnedArtifactRole::Target,
            source_path: "fixture.gguf",
            relative_path: "fixture.gguf",
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        verify_artifact_file(&path, artifact).expect("original payload");
        std::fs::write(&path, b"abd").expect("mutate payload");
        let error = verify_artifact_file(&path, artifact)
            .expect_err("same-size payload mutation must fail")
            .to_string();
        assert!(error.contains("SHA-256"), "{error}");
        std::fs::remove_file(path).expect("remove fixture");
    }
}
