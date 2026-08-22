use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// Fixed default identifier for checkpoints that `qw generate`, `qw serve`,
/// and the runtime benchmarks use when neither `--model` nor `QW_MODEL_PATH`
/// is given. Users override the *path* to a checkpoint, never this identifier.
pub const DEFAULT_MODEL_IDENTIFIER: &str = "Jundot/Qwen3.8-27B-oQ4e-fp16-mtp";
#[cfg(any(feature = "specprefill", test))]
pub const DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER: &str =
    "mlx-community/Qwen3.5-0.8B-MLX-8bit";

/// Split a Hugging Face model identifier into the namespace and model
/// components that form the cache subdirectory. Rejects identifiers that could
/// escape the cache directory or encode a Windows-style separator.
pub fn validate_identifier(identifier: &str) -> Result<Vec<&str>> {
    let components: Vec<_> = identifier.split('/').collect();
    if components.is_empty()
        || components.len() > 2
        || components
            .iter()
            .any(|component| component.is_empty() || *component == "." || *component == "..")
        || identifier.starts_with('/')
        || identifier.contains('\\')
    {
        bail!("invalid Hugging Face model identifier `{identifier}`");
    }
    Ok(components)
}

/// Compute the on-disk cache path for a Hugging Face model identifier under
/// `~/.cache/qw/models/<namespace>/<model>`.
pub fn model_cache_path(home: &Path, identifier: &str) -> Result<PathBuf> {
    let mut destination = home.join(".cache/qw/models");
    for component in validate_identifier(identifier)? {
        destination.push(component);
    }
    Ok(destination)
}

/// Resolve the checkpoint directory to load, honoring (in order) the CLI
/// override, the `QW_MODEL_PATH` environment override, then the default cache
/// path for `DEFAULT_MODEL_IDENTIFIER`. Pure: reads no environment itself.
pub fn resolve_model_dir(
    cli_override: Option<&Path>,
    env_override: Option<&Path>,
    home: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(c) = cli_override {
        return Ok(c.to_path_buf());
    }
    if let Some(e) = env_override {
        return Ok(e.to_path_buf());
    }
    let Some(h) = home else {
        bail!("HOME is not set");
    };
    model_cache_path(h, DEFAULT_MODEL_IDENTIFIER)
}

/// Convenience wrapper around `resolve_model_dir` that reads `QW_MODEL_PATH`
/// and `HOME` from the process environment.
pub fn resolve_model_path(cli_override: Option<&Path>) -> Result<PathBuf> {
    let env_override = std::env::var_os("QW_MODEL_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    resolve_model_dir(cli_override, env_override.as_deref(), home.as_deref())
}

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
    model_cache_path(home, DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER)
}

#[cfg(any(feature = "specprefill", test))]
pub fn resolve_specprefill_draft_path(cli_override: Option<&Path>) -> Result<PathBuf> {
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
    use super::{
        DEFAULT_MODEL_IDENTIFIER, DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER, model_cache_path,
        resolve_model_dir, resolve_specprefill_draft_dir,
    };
    use std::path::Path;

    #[test]
    fn model_cache_path_preserves_namespace() {
        assert_eq!(
            model_cache_path(Path::new("/home/user"), "Qwen/Qwen3.5-0.8B")
                .expect("valid identifier"),
            Path::new("/home/user/.cache/qw/models/Qwen/Qwen3.5-0.8B")
        );
    }

    #[test]
    fn model_cache_path_rejects_unsafe_identifiers() {
        for identifier in [
            "",
            "/Qwen/model",
            ".",
            "..",
            "Qwen/.",
            "Qwen/..",
            "Qwen//model",
            "Qwen/model/extra",
            r"Qwen\model",
        ] {
            assert!(
                model_cache_path(Path::new("/home/user"), identifier).is_err(),
                "{identifier:?} should be rejected"
            );
        }
    }

    #[test]
    fn resolve_model_dir_precedence() {
        assert_eq!(
            resolve_model_dir(Some(Path::new("/cli")), Some(Path::new("/env")), Some(Path::new("/home")))
                .expect("cli override"),
            Path::new("/cli")
        );
        assert_eq!(
            resolve_model_dir(None, Some(Path::new("/env")), Some(Path::new("/home")))
                .expect("env override"),
            Path::new("/env")
        );
        assert_eq!(
            resolve_model_dir(None, None, Some(Path::new("/home")))
                .expect("default cache"),
            Path::new("/home/.cache/qw/models/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp")
        );
    }

    #[test]
    fn resolve_model_dir_requires_home_only_when_unoverridden() {
        assert!(resolve_model_dir(None, None, None).is_err(), "HOME required without overrides");
        assert_eq!(
            resolve_model_dir(Some(Path::new("/cli")), None, None)
                .expect("cli override needs no HOME"),
            Path::new("/cli")
        );
    }

    #[test]
    fn default_identifier_cache_path_matches() {
        assert_eq!(
            model_cache_path(Path::new("/home/user"), DEFAULT_MODEL_IDENTIFIER)
                .expect("default identifier is valid"),
            Path::new("/home/user/.cache/qw/models/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp")
        );
    }

    #[test]
    fn specprefill_draft_default_path_and_precedence() {
        assert_eq!(
            resolve_specprefill_draft_dir(
                Some(Path::new("/cli-draft")),
                Some(Path::new("/env-draft")),
                Some(Path::new("/home")),
            )
            .expect("explicit override"),
            Path::new("/cli-draft")
        );
        assert_eq!(
            resolve_specprefill_draft_dir(
                None,
                Some(Path::new("/env-draft")),
                Some(Path::new("/home")),
            )
            .expect("environment override"),
            Path::new("/env-draft")
        );
        assert_eq!(
            resolve_specprefill_draft_dir(None, None, Some(Path::new("/home")))
                .expect("default draft cache"),
            model_cache_path(
                Path::new("/home"),
                DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER,
            )
            .expect("valid pinned identifier")
        );
    }
}