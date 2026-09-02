#[cfg(any(feature = "specprefill", test))]
use std::path::{Path, PathBuf};

#[cfg(any(feature = "specprefill", test))]
use anyhow::{Result, bail};

#[cfg(any(feature = "specprefill", test))]
pub(crate) const SPECPREFILL_DRAFT_CACHE_RELATIVE_DIR: &str =
    ".cache/qw/models/mlx-community/Qwen3.5-0.8B-MLX-8bit";

#[cfg(any(feature = "specprefill", test))]
fn resolve_specprefill_draft_dir(
    cli_override: Option<&Path>,
    env_override: Option<&Path>,
    home: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = cli_override {
        return Ok(path.to_path_buf());
    }
    if let Some(path) = env_override {
        return Ok(path.to_path_buf());
    }
    let Some(home) = home else {
        bail!("HOME is not set");
    };
    Ok(home.join(SPECPREFILL_DRAFT_CACHE_RELATIVE_DIR))
}

#[cfg(any(feature = "specprefill", test))]
pub(crate) fn resolve_specprefill_draft_path(cli_override: Option<&Path>) -> Result<PathBuf> {
    let env_override = std::env::var_os("QW_SPECPREFILL_DRAFT_MODEL_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    resolve_specprefill_draft_dir(cli_override, env_override.as_deref(), home.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_resolver_does_not_affect_pinned_target() {
        assert_eq!(
            resolve_specprefill_draft_dir(None, None, Some(Path::new("/home/user")))
                .expect("draft path"),
            Path::new("/home/user/.cache/qw/models/mlx-community/Qwen3.5-0.8B-MLX-8bit")
        );
    }
}
