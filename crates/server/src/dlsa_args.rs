//! D-LSA flags (`docs/dlsa.md` §13), shared by `qw serve` and `qw generate`.

use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap_derive::Args;
use qw_runtime::{DlsaBackend, DlsaMapKind, DlsaOptions, ResolvedDlsa, parse_dlsa_map_kind};

fn parse_map_kind(value: &str) -> Result<DlsaMapKind, String> {
    parse_dlsa_map_kind(value)
        .ok_or_else(|| format!("invalid D-LSA map kind '{value}': expected distilled or pca"))
}

fn parse_backend(value: &str) -> Result<DlsaBackend, String> {
    DlsaBackend::parse(value).map_err(|e| e.to_string())
}

#[derive(Debug, Clone, Args)]
pub struct DlsaArgs {
    /// Enable D-LSA sparse attention in the full-attention layers (also QW_DLSA=1). The server needs --no-kv-quantization.
    #[arg(long)]
    pub dlsa: bool,

    /// Converted D-LSA map; defaults to QW_DLSA_MAP_PATH or the `qw download --dlsa-map` cache.
    #[arg(long)]
    pub dlsa_map: Option<PathBuf>,

    /// Which cached map to use: distilled (default) or pca; also QW_DLSA_MAP_KIND.
    #[arg(long, value_name = "distilled|pca", value_parser = parse_map_kind)]
    pub dlsa_map_kind: Option<DlsaMapKind>,

    /// First keys every query reads [default: 4; env QW_DLSA_SINK].
    #[arg(long)]
    pub dlsa_sink: Option<usize>,

    /// Recent keys every query reads, also the cluster block size [default: 512, the window the maps were distilled at; env QW_DLSA_WINDOW].
    #[arg(long)]
    pub dlsa_window: Option<usize>,

    /// Long-range keys a query reads [default: 4096; env QW_DLSA_TOP_K].
    #[arg(long)]
    pub dlsa_top_k: Option<usize>,

    /// Buckets a query expands [default: 1024; env QW_DLSA_TOP_P].
    #[arg(long)]
    pub dlsa_top_p: Option<usize>,

    /// Query rows at positions below this many keys read every causal key (dense); sparse reads start there, where
    /// they beat dense [default: 12288; 0: only the dense-key-set rule; env QW_DLSA_DENSE_BELOW].
    #[arg(long, value_name = "TOKENS")]
    pub dlsa_dense_below: Option<usize>,

    /// Where the select and the read run: metal (the Metal kernels, default), mlx (the MLX
    /// reference path) or metal-select (Metal select + MLX read, bitwise mlx); env QW_DLSA_BACKEND.
    #[arg(long, value_name = "metal|mlx|metal-select", value_parser = parse_backend)]
    pub dlsa_backend: Option<DlsaBackend>,
}

impl DlsaArgs {
    pub fn options(&self) -> DlsaOptions {
        DlsaOptions {
            enabled: self.dlsa,
            map_path: self.dlsa_map.clone(),
            map_kind: self.dlsa_map_kind,
            sink: self.dlsa_sink,
            window: self.dlsa_window,
            top_k: self.dlsa_top_k,
            top_p: self.dlsa_top_p,
            dense_below: self.dlsa_dense_below,
            backend: self.dlsa_backend,
        }
    }

    /// The flags, then the `QW_DLSA_*` variables `env` returns, then the defaults.
    pub fn resolve(&self, env: impl Fn(&str) -> Option<String>) -> Result<Option<ResolvedDlsa>> {
        self.options().resolve(env)
    }
}

/// The process environment, as [`DlsaArgs::resolve`] reads it.
pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// D-LSA runs on an fp16 KV cache only; the server's default TurboQuant cache is refused.
pub(crate) fn require_unquantized_kv(dlsa_enabled: bool, no_kv_quantization: bool) -> Result<()> {
    ensure!(
        !dlsa_enabled || no_kv_quantization,
        "D-LSA (--dlsa or QW_DLSA) requires --no-kv-quantization: it does not support the default TurboQuant KV cache"
    );
    Ok(())
}
