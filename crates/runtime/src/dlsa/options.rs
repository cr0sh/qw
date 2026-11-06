// SPDX-License-Identifier: Apache-2.0
//
//! Serving options: the D-LSA settings `qw serve` and `qw generate` take from flags and the
//! `QW_DLSA_*` environment, resolved to a [`DlsaConfig`], and the identity of a resolved config
//! that keys the prefix cache (`docs/dlsa.md` §13).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tracing::warn;

use super::config::DlsaIndexConfig;
use super::convert::sha256_hex;
use super::resolve::{DLSA_MAP_ENV, DlsaMapKind, resolve_dir};
use super::runtime::{DLSA_DENSE_BELOW_DEFAULT, DlsaBackend, DlsaConfig};

/// `1`/`true` enables D-LSA, `0`/`false` leaves it off.
pub const DLSA_ENV: &str = "QW_DLSA";
/// `distilled` (default) or `pca`.
pub const DLSA_MAP_KIND_ENV: &str = "QW_DLSA_MAP_KIND";
pub const DLSA_SINK_ENV: &str = "QW_DLSA_SINK";
pub const DLSA_WINDOW_ENV: &str = "QW_DLSA_WINDOW";
pub const DLSA_TOP_K_ENV: &str = "QW_DLSA_TOP_K";
pub const DLSA_TOP_P_ENV: &str = "QW_DLSA_TOP_P";
/// Query rows below this many keys read every causal key (`DlsaConfig::dense_below`); 0 disables it.
pub const DLSA_DENSE_BELOW_ENV: &str = "QW_DLSA_DENSE_BELOW";
/// `mlx`, `metal` (default) or `metal-select`: where the select and the read run.
pub const DLSA_BACKEND_ENV: &str = "QW_DLSA_BACKEND";

/// The window the released maps were distilled at; it is also the cluster block size.
pub const DISTILLED_WINDOW: usize = 512;

/// `distilled` or `pca` (the names `--dlsa-map-kind` and `QW_DLSA_MAP_KIND` take).
pub fn parse_dlsa_map_kind(value: &str) -> Option<DlsaMapKind> {
    match value {
        "distilled" => Some(DlsaMapKind::Distilled),
        "pca" => Some(DlsaMapKind::PcaStep0),
        _ => None,
    }
}

/// What the user asked for on the command line; `None` fields fall back to the environment,
/// then to [`DlsaIndexConfig::QWEN_DEFAULTS`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DlsaOptions {
    /// `--dlsa`. Off here still lets `QW_DLSA=1` enable it.
    pub enabled: bool,
    /// `--dlsa-map`.
    pub map_path: Option<PathBuf>,
    /// `--dlsa-map-kind`: which cached map to use when no path is given.
    pub map_kind: Option<DlsaMapKind>,
    /// `--dlsa-sink`.
    pub sink: Option<usize>,
    /// `--dlsa-window`.
    pub window: Option<usize>,
    /// `--dlsa-top-k`.
    pub top_k: Option<usize>,
    /// `--dlsa-top-p`.
    pub top_p: Option<usize>,
    /// `--dlsa-dense-below`.
    pub dense_below: Option<usize>,
    /// `--dlsa-backend`.
    pub backend: Option<DlsaBackend>,
}

/// A resolved D-LSA setting: the config to enable and the map kind it was resolved for.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedDlsa {
    pub config: DlsaConfig,
    pub kind: DlsaMapKind,
}

/// Where a knob's value came from, for error messages.
fn knob(
    cli: Option<usize>,
    flag: &'static str,
    env: &impl Fn(&str) -> Option<String>,
    var: &'static str,
    default: usize,
) -> Result<(usize, String)> {
    if let Some(value) = cli {
        return Ok((value, flag.to_string()));
    }
    if let Some(raw) = non_empty(env, var) {
        let value = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("{var} must be a non-negative integer, got {raw:?}"))?;
        return Ok((value, var.to_string()));
    }
    Ok((default, "default".to_string()))
}

fn non_empty(env: &impl Fn(&str) -> Option<String>, var: &str) -> Option<String> {
    env(var).filter(|value| !value.is_empty())
}

fn parse_switch(raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{DLSA_ENV} must be 1/true or 0/false, got {raw:?}"),
    }
}

impl DlsaOptions {
    /// The flags given on the command line, by name, for "requires --dlsa" errors.
    fn given_flags(&self) -> Vec<&'static str> {
        [
            (self.map_path.is_some(), "--dlsa-map"),
            (self.map_kind.is_some(), "--dlsa-map-kind"),
            (self.sink.is_some(), "--dlsa-sink"),
            (self.window.is_some(), "--dlsa-window"),
            (self.top_k.is_some(), "--dlsa-top-k"),
            (self.top_p.is_some(), "--dlsa-top-p"),
            (self.dense_below.is_some(), "--dlsa-dense-below"),
            (self.backend.is_some(), "--dlsa-backend"),
        ]
        .into_iter()
        .filter_map(|(given, flag)| given.then_some(flag))
        .collect()
    }

    /// The config to enable, or `None` when D-LSA stays off. Each setting comes from its flag,
    /// then its `QW_DLSA_*` variable (read through `env`, so tests need no process env), then
    /// the Qwen defaults. Errors name the flag or variable the bad value came from.
    pub fn resolve(&self, env: impl Fn(&str) -> Option<String>) -> Result<Option<ResolvedDlsa>> {
        let enabled = self.enabled
            || non_empty(&env, DLSA_ENV)
                .map(|raw| parse_switch(&raw))
                .transpose()?
                .unwrap_or(false);
        if !enabled {
            let given = self.given_flags();
            if !given.is_empty() {
                bail!("{} requires --dlsa (or {DLSA_ENV}=1)", given.join(", "));
            }
            return Ok(None);
        }
        let kind = match self.map_kind {
            Some(kind) => kind,
            None => match non_empty(&env, DLSA_MAP_KIND_ENV) {
                Some(raw) => parse_dlsa_map_kind(raw.trim()).with_context(|| {
                    format!("{DLSA_MAP_KIND_ENV} must be distilled or pca, got {raw:?}")
                })?,
                None => DlsaMapKind::Distilled,
            },
        };
        let defaults = DlsaIndexConfig::QWEN_DEFAULTS;
        let (sink, sink_from) = knob(
            self.sink,
            "--dlsa-sink",
            &env,
            DLSA_SINK_ENV,
            defaults.sink_size,
        )?;
        let (window, window_from) = knob(
            self.window,
            "--dlsa-window",
            &env,
            DLSA_WINDOW_ENV,
            defaults.window_size,
        )?;
        let (top_k, top_k_from) = knob(
            self.top_k,
            "--dlsa-top-k",
            &env,
            DLSA_TOP_K_ENV,
            defaults.top_k,
        )?;
        let (top_p, top_p_from) = knob(
            self.top_p,
            "--dlsa-top-p",
            &env,
            DLSA_TOP_P_ENV,
            defaults.top_p,
        )?;
        let (dense_below, _) = knob(
            self.dense_below,
            "--dlsa-dense-below",
            &env,
            DLSA_DENSE_BELOW_ENV,
            DLSA_DENSE_BELOW_DEFAULT,
        )?;
        let backend = match self.backend {
            Some(backend) => backend,
            None => match non_empty(&env, DLSA_BACKEND_ENV) {
                Some(raw) => DlsaBackend::parse(&raw)
                    .with_context(|| format!("{DLSA_BACKEND_ENV} is invalid"))?,
                None => DlsaBackend::default(),
            },
        };
        let mut config = DlsaConfig::new(PathBuf::new());
        config.backend = backend;
        config.dense_below = dense_below;
        config.index = DlsaIndexConfig {
            window_size: window,
            sink_size: sink,
            top_k,
            top_p,
            ..defaults
        };
        config.validate().with_context(|| {
            format!(
                "invalid D-LSA settings: sink {sink} ({sink_from}), window {window} ({window_from}), top_k {top_k} ({top_k_from}), top_p {top_p} ({top_p_from})"
            )
        })?;
        if window != DISTILLED_WINDOW {
            warn!(
                phase = "dlsa.options",
                window,
                source = %window_from,
                "the D-LSA maps were distilled at window {DISTILLED_WINDOW}, which is also the cluster block size; other windows are untested"
            );
        }
        let env_map = non_empty(&env, DLSA_MAP_ENV).map(PathBuf::from);
        let home = non_empty(&env, "HOME").map(PathBuf::from);
        config.map_path = resolve_dir(
            kind,
            self.map_path.as_deref(),
            env_map.as_deref(),
            home.as_deref(),
        )?;
        Ok(Some(ResolvedDlsa { config, kind }))
    }
}

/// What a prefix-cache entry built under D-LSA depends on: the map file's bytes, the map dtype,
/// the index config, the short-context policy and the read (the Metal read rounds differently
/// from the MLX one, so their KV never mix; Metal select + MLX read is bitwise the MLX path
/// and shares its entries).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlsaIdentity {
    pub map_sha256: String,
    bytes: Vec<u8>,
}

impl DlsaIdentity {
    /// The bytes appended to the prefix-cache namespace parts.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Hash `config.map_path` and encode the settings that shape the index state.
pub fn dlsa_identity(config: &DlsaConfig) -> Result<DlsaIdentity> {
    let map_sha256 = map_sha256(&config.map_path)?;
    let index = &config.index;
    let bytes = format!(
        "qw-dlsa;map_sha256={map_sha256};map_dtype={:?};short_context={:?};dense_below={};read={};window={};sink={};top_k={};top_p={};index_dim={};pool={};bucket={};cap={}",
        config.map_dtype,
        config.short_context,
        config.dense_below,
        match config.backend {
            DlsaBackend::Mlx | DlsaBackend::MetalSelect => "mlx",
            DlsaBackend::Metal => "metal",
        },
        index.window_size,
        index.sink_size,
        index.top_k,
        index.top_p,
        index.index_dim,
        index.pool_size,
        index.bucket_size,
        index.bucket_cap,
    )
    .into_bytes();
    Ok(DlsaIdentity { map_sha256, bytes })
}

fn map_sha256(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("cannot read D-LSA map {}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::dlsa::{DlsaMapDtype, DlsaShortContext, dlsa_map_cache_path};

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("qw-dlsa-opt-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    fn map_file(dir: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn off_unless_enabled_and_knobs_need_the_switch() {
        assert_eq!(DlsaOptions::default().resolve(env(&[])).unwrap(), None);
        // Env knobs alone do not enable it
        assert_eq!(
            DlsaOptions::default()
                .resolve(env(&[(DLSA_TOP_K_ENV, "8"), (DLSA_ENV, "0")]))
                .unwrap(),
            None
        );
        let err = DlsaOptions {
            window: Some(256),
            map_path: Some("/m".into()),
            ..Default::default()
        }
        .resolve(env(&[]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--dlsa-map, --dlsa-window requires --dlsa"),
            "{err}"
        );
        let err = DlsaOptions::default()
            .resolve(env(&[(DLSA_ENV, "maybe")]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("QW_DLSA must be"), "{err}");
    }

    #[test]
    fn precedence_is_cli_then_env_then_defaults() {
        let dir = tmp("prec");
        let cli_map = map_file(&dir, "cli.safetensors", b"cli");
        let env_map = map_file(&dir, "env.safetensors", b"env");
        let lookup = env(&[
            (DLSA_ENV, "true"),
            (DLSA_MAP_ENV, env_map.to_str().unwrap()),
            (DLSA_SINK_ENV, "8"),
            (DLSA_TOP_K_ENV, "2048"),
            (DLSA_TOP_P_ENV, "512"),
        ]);
        let resolved = DlsaOptions {
            top_k: Some(1024),
            ..Default::default()
        }
        .resolve(&lookup)
        .unwrap()
        .expect("QW_DLSA enables");
        assert_eq!(resolved.config.map_path, env_map);
        assert_eq!(resolved.kind, DlsaMapKind::Distilled);
        let index = resolved.config.index;
        assert_eq!(index.sink_size, 8);
        assert_eq!(index.top_k, 1024);
        assert_eq!(index.top_p, 512);
        assert_eq!(index.window_size, 512);
        assert_eq!(resolved.config.map_dtype, DlsaMapDtype::Bf16);
        assert_eq!(
            resolved.config.short_context,
            DlsaShortContext::DenseFallback
        );

        let resolved = DlsaOptions {
            enabled: true,
            map_path: Some(cli_map.clone()),
            sink: Some(2),
            ..Default::default()
        }
        .resolve(&lookup)
        .unwrap()
        .unwrap();
        assert_eq!(resolved.config.map_path, cli_map);
        assert_eq!(resolved.config.index.sink_size, 2);

        let defaults = DlsaOptions {
            enabled: true,
            map_path: Some(cli_map),
            ..Default::default()
        }
        .resolve(env(&[]))
        .unwrap()
        .unwrap();
        assert_eq!(defaults.config.index, DlsaIndexConfig::QWEN_DEFAULTS);
    }

    #[test]
    fn map_kind_selects_the_cached_map() {
        let home = tmp("kind");
        let lookup = env(&[
            (DLSA_ENV, "1"),
            (DLSA_MAP_KIND_ENV, "pca"),
            ("HOME", home.to_str().unwrap()),
        ]);
        let err = DlsaOptions::default()
            .resolve(&lookup)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--dlsa-pca"), "{err}");
        assert!(
            err.contains(
                &dlsa_map_cache_path(&home, DlsaMapKind::PcaStep0)
                    .display()
                    .to_string()
            ),
            "{err}"
        );
        // The flag wins over the variable
        let err = DlsaOptions {
            map_kind: Some(DlsaMapKind::Distilled),
            ..Default::default()
        }
        .resolve(&lookup)
        .unwrap_err()
        .to_string();
        assert!(!err.contains("--dlsa-pca"), "{err}");
        let err = DlsaOptions::default()
            .resolve(env(&[(DLSA_ENV, "1"), (DLSA_MAP_KIND_ENV, "big")]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("QW_DLSA_MAP_KIND"), "{err}");
        assert_eq!(parse_dlsa_map_kind("pca"), Some(DlsaMapKind::PcaStep0));
        assert_eq!(parse_dlsa_map_kind("PCA"), None);
    }

    #[test]
    fn invalid_values_name_their_flag_or_variable() {
        let dir = tmp("bad");
        let map = map_file(&dir, "m.safetensors", b"m");
        let on = DlsaOptions {
            enabled: true,
            map_path: Some(map),
            ..Default::default()
        };
        for (var, value) in [
            (DLSA_SINK_ENV, "-1"),
            (DLSA_WINDOW_ENV, "wide"),
            (DLSA_TOP_K_ENV, "1.5"),
            (DLSA_TOP_P_ENV, "x"),
        ] {
            let err = on.resolve(env(&[(var, value)])).unwrap_err().to_string();
            assert!(err.contains(var), "{var}: {err}");
        }
        let err = DlsaOptions {
            top_k: Some(0),
            ..on.clone()
        }
        .resolve(env(&[]))
        .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("top_k 0 (--dlsa-top-k)"), "{message}");
        assert!(message.contains("top_k must be at least 1"), "{message}");
        let err = on.resolve(env(&[(DLSA_TOP_P_ENV, "4096")])).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("(QW_DLSA_TOP_P)"), "{message}");
        let err = DlsaOptions {
            window: Some(500),
            ..on.clone()
        }
        .resolve(env(&[]))
        .unwrap_err();
        assert!(format!("{err:#}").contains("(--dlsa-window)"), "{err:#}");
        // A smaller window that keeps the block arithmetic valid is accepted (with a warning)
        let resolved = DlsaOptions {
            window: Some(256),
            ..on
        }
        .resolve(env(&[]))
        .unwrap()
        .unwrap();
        assert_eq!(resolved.config.index.window_size, 256);
    }

    #[test]
    fn identity_tracks_map_bytes_and_settings() {
        let dir = tmp("ident");
        let a = map_file(&dir, "a.safetensors", b"map a");
        let b = map_file(&dir, "b.safetensors", b"map b");
        let a_copy = map_file(&dir, "a2.safetensors", b"map a");
        let base = dlsa_identity(&DlsaConfig::new(a.clone())).unwrap();
        assert_eq!(base.map_sha256, sha256_hex(b"map a"));
        assert_eq!(base, dlsa_identity(&DlsaConfig::new(a_copy)).unwrap());
        assert_ne!(base, dlsa_identity(&DlsaConfig::new(b)).unwrap());
        let mut config = DlsaConfig::new(a.clone());
        config.index.top_k = 2048;
        assert_ne!(base, dlsa_identity(&config).unwrap());
        let mut config = DlsaConfig::new(a.clone());
        config.map_dtype = DlsaMapDtype::F32;
        assert_ne!(base, dlsa_identity(&config).unwrap());
        let mut config = DlsaConfig::new(a.clone());
        config.short_context = DlsaShortContext::Sparse;
        assert_ne!(base, dlsa_identity(&config).unwrap());
        let mut config = DlsaConfig::new(a.clone());
        config.dense_below = 0;
        assert_ne!(base, dlsa_identity(&config).unwrap());
        // The default (Metal) read never shares KV with the MLX read; the parity mode does
        assert_eq!(DlsaConfig::new(a.clone()).backend, DlsaBackend::Metal);
        let on = |backend| {
            let mut config = DlsaConfig::new(a.clone());
            config.backend = backend;
            dlsa_identity(&config).unwrap()
        };
        assert_ne!(base, on(DlsaBackend::Mlx));
        assert_eq!(on(DlsaBackend::Mlx), on(DlsaBackend::MetalSelect));
        let err = dlsa_identity(&DlsaConfig::new(dir.join("missing")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot read D-LSA map"), "{err}");
    }

    #[test]
    fn backend_comes_from_the_flag_then_the_variable() {
        let dir = tmp("backend");
        let map = map_file(&dir, "m.safetensors", b"m");
        let on = DlsaOptions {
            enabled: true,
            map_path: Some(map),
            ..Default::default()
        };
        let backend = |options: &DlsaOptions, pairs: &[(&str, &str)]| {
            options.resolve(env(pairs)).unwrap().unwrap().config.backend
        };
        assert_eq!(backend(&on, &[]), DlsaBackend::Metal);
        assert_eq!(
            backend(&on, &[(DLSA_BACKEND_ENV, "metal-select")]),
            DlsaBackend::MetalSelect
        );
        let mlx = DlsaOptions {
            backend: Some(DlsaBackend::Mlx),
            ..on.clone()
        };
        assert_eq!(
            backend(&mlx, &[(DLSA_BACKEND_ENV, "metal")]),
            DlsaBackend::Mlx
        );
        let err = on.resolve(env(&[(DLSA_BACKEND_ENV, "cuda")])).unwrap_err();
        assert!(format!("{err:#}").contains(DLSA_BACKEND_ENV), "{err:#}");
        let err = DlsaOptions {
            backend: Some(DlsaBackend::Mlx),
            ..Default::default()
        }
        .resolve(env(&[]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("--dlsa-backend requires --dlsa"), "{err}");
    }

    #[test]
    fn dense_below_comes_from_the_flag_then_the_variable() {
        let dir = tmp("dense-below");
        let map = map_file(&dir, "m.safetensors", b"m");
        let on = DlsaOptions {
            enabled: true,
            map_path: Some(map),
            ..Default::default()
        };
        let dense_below = |options: &DlsaOptions, pairs: &[(&str, &str)]| {
            options
                .resolve(env(pairs))
                .unwrap()
                .unwrap()
                .config
                .dense_below
        };
        assert_eq!(dense_below(&on, &[]), DLSA_DENSE_BELOW_DEFAULT);
        assert_eq!(dense_below(&on, &[(DLSA_DENSE_BELOW_ENV, "0")]), 0);
        let flag = DlsaOptions {
            dense_below: Some(32_768),
            ..on.clone()
        };
        assert_eq!(dense_below(&flag, &[(DLSA_DENSE_BELOW_ENV, "0")]), 32_768);
        let err = on
            .resolve(env(&[(DLSA_DENSE_BELOW_ENV, "16k")]))
            .unwrap_err()
            .to_string();
        assert!(err.contains(DLSA_DENSE_BELOW_ENV), "{err}");
        let err = DlsaOptions {
            dense_below: Some(0),
            ..Default::default()
        }
        .resolve(env(&[]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("--dlsa-dense-below requires --dlsa"), "{err}");
    }
}
