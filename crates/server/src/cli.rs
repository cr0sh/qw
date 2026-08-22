use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap_derive::{Args as DeriveArgs, ValueEnum};
use qw_prefix_cache::CacheConfig;
use qw_runtime::{KVCacheMode, resolve_model_path};
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::{Engine, router};
#[cfg(feature = "specprefill")]
use crate::SpecPrefillPolicyConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Debug, DeriveArgs)]
pub struct ServerArgs {
    /// Checkpoint directory or HF identifier; defaults to the qw model cache unless QW_MODEL_PATH is set.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Only accept requests for this exact model ID.
    #[arg(long)]
    model_id: Option<String>,

    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8000")]
    bind: String,

    /// Byte capacity of the in-memory prefix snapshot tier.
    #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024_u64)]
    prefix_cache_memory_bytes: u64,

    /// Directory for persistent prefix snapshots; defaults to `~/.cache/qw/checkpoint`.
    #[arg(long)]
    prefix_cache_directory: Option<PathBuf>,

    /// Byte capacity of the filesystem prefix snapshot tier.
    #[arg(long, default_value_t = 16 * 1024 * 1024 * 1024_u64)]
    prefix_cache_filesystem_bytes: u64,

    /// MTP verify input block size (bonus token plus proposals).
    #[arg(long = "mtp-k", default_value_t = 3)]
    mtp_k: usize,

    #[cfg(feature = "specprefill")]
    /// Current-turn token threshold above which SpecPrefill activates.
    #[arg(long, default_value_t = 8_000)]
    specprefill_min_turn_tokens: usize,
    #[cfg(feature = "specprefill")]
    /// Fraction of score-ranked prompt chunks retained by SpecPrefill.
    #[arg(long, default_value_t = 0.25)]
    specprefill_keep_rate: f32,
    #[cfg(feature = "specprefill")]
    /// Tokens force-kept at the start of the SpecPrefill-eligible suffix.
    #[arg(long, default_value_t = 256)]
    specprefill_keep_first_tokens: usize,
    #[cfg(feature = "specprefill")]
    /// Tokens force-kept at the end of the SpecPrefill-eligible suffix.
    #[arg(long, default_value_t = 256)]
    specprefill_keep_last_tokens: usize,
    /// Disable the default 4-bit TurboQuant KV cache.
    #[arg(long)]
    no_kv_quantization: bool,

    /// Tracing output format.
    #[arg(long, value_enum, default_value = "human")]
    output_format: OutputFormat,
}

impl ServerArgs {
    fn kv_cache_mode(&self) -> KVCacheMode {
        if self.no_kv_quantization {
            KVCacheMode::Fp16
        } else {
            KVCacheMode::Turbo4
        }
    }

    #[cfg(feature = "specprefill")]
    fn specprefill_policy(&self) -> SpecPrefillPolicyConfig {
        SpecPrefillPolicyConfig {
            min_turn_tokens: self.specprefill_min_turn_tokens,
            keep_rate: self.specprefill_keep_rate,
            keep_first_tokens: self.specprefill_keep_first_tokens,
            keep_last_tokens: self.specprefill_keep_last_tokens,
        }
    }
}

fn validate_cli(cli: &ServerArgs) -> Result<()> {
    ensure!(
        cli.prefix_cache_memory_bytes > 0,
        "--prefix-cache-memory-bytes must be greater than zero"
    );
    ensure!(
        cli.prefix_cache_filesystem_bytes > 0,
        "--prefix-cache-filesystem-bytes must be greater than zero"
    );
    ensure!(cli.mtp_k >= 2, "--mtp-k must be at least 2");
    #[cfg(feature = "specprefill")]
    cli.specprefill_policy().validate()?;
    Ok(())
}

fn default_prefix_cache_directory() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .context("HOME is not set; cannot resolve the default prefix cache directory")?;
    Ok(PathBuf::from(home).join(".cache/qw/checkpoint"))
}

fn resolve_prefix_cache_directory(directory: Option<PathBuf>) -> Result<PathBuf> {
    directory
        .map(Ok)
        .unwrap_or_else(default_prefix_cache_directory)
}

pub async fn serve(cli: ServerArgs) -> Result<()> {
    validate_cli(&cli)?;
    match cli.output_format {
        OutputFormat::Human => tracing_subscriber::fmt::init(),
        OutputFormat::Json => tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).json().init(),
    }
    let bind: SocketAddr = cli
        .bind
        .parse()
        .with_context(|| format!("invalid --bind address {:?}", cli.bind))?;
    info!(phase = "server.starting", bind = %bind);

    let kv_cache_mode = cli.kv_cache_mode();
    #[cfg(feature = "specprefill")]
    let specprefill_policy = cli.specprefill_policy();
    let model = resolve_model_path(cli.model.as_deref())?;
    let prefix_cache_directory = resolve_prefix_cache_directory(cli.prefix_cache_directory)?;
    let engine = Engine::start_qwen(
        model,
        cli.model_id,
        CacheConfig {
            memory_bytes: cli.prefix_cache_memory_bytes,
            directory: Some(prefix_cache_directory),
            filesystem_bytes: cli.prefix_cache_filesystem_bytes,
        },
        cli.mtp_k,
        kv_cache_mode,
        #[cfg(feature = "specprefill")]
        specprefill_policy,
    )?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;
    info!(phase = "server.listening", bind = %bind);
    axum::serve(listener, router(engine))
        .await
        .context("HTTP server failed")
}

#[cfg(test)]
mod tests {
    use super::{OutputFormat, ServerArgs, resolve_prefix_cache_directory, validate_cli};
    use clap::{CommandFactory as _, Parser as _};
    use clap_derive::Parser;
    use qw_runtime::KVCacheMode;

    #[derive(Debug, Parser)]
    #[command(name = "qw-server", about = "OpenAI-compatible dense Qwen3.5 server")]
    struct TestCli {
        #[command(flatten)]
        args: ServerArgs,
    }

    #[test]
    fn cli_model_is_optional() {
        let cli = TestCli::try_parse_from(["qw-server"]).expect("CLI");
        assert_eq!(cli.args.model, None);
    }

    #[test]
    fn cli_exposes_optional_model_id() {
        let unrestricted =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(unrestricted.args.model_id, None);

        let configured = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--model-id",
            "served-model",
        ])
        .expect("CLI");
        assert_eq!(configured.args.model_id.as_deref(), Some("served-model"));

        let help = TestCli::command().render_long_help().to_string();
        assert!(help.contains("--model-id <MODEL_ID>"), "{help}");
    }

    #[test]
    fn cli_parses_output_format_and_documents_values() {
        let default =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(default.args.output_format, OutputFormat::Human);

        for (value, expected) in [("human", OutputFormat::Human), ("json", OutputFormat::Json)] {
            let cli = TestCli::try_parse_from([
                "qw-server",
                "--model",
                "/tmp/checkpoint",
                "--output-format",
                value,
            ])
            .expect("CLI");
            assert_eq!(cli.args.output_format, expected);
        }

        let help = TestCli::command().render_long_help().to_string();
        assert!(help.contains("--output-format <OUTPUT_FORMAT>"), "{help}");
        assert!(help.contains("human"), "{help}");
        assert!(help.contains("json"), "{help}");
    }

    #[test]
    fn turbo4_kv_quantization_is_default_with_explicit_opt_out() {
        let default =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(default.args.kv_cache_mode(), KVCacheMode::Turbo4);

        let unquantized = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--no-kv-quantization",
        ])
        .expect("CLI");
        assert_eq!(unquantized.args.kv_cache_mode(), KVCacheMode::Fp16);

        let help = TestCli::command().render_long_help().to_string();
        assert!(help.contains("--no-kv-quantization"), "{help}");
        assert!(help.contains("default 4-bit TurboQuant KV cache"), "{help}");
    }

    #[test]
    fn cli_configures_memory_and_filesystem_cache_tiers() {
        let defaults =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(
            defaults.args.prefix_cache_memory_bytes,
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(defaults.args.prefix_cache_directory, None);
        assert_eq!(
            defaults.args.prefix_cache_filesystem_bytes,
            16 * 1024 * 1024 * 1024
        );
        let home = std::env::var_os("HOME").expect("HOME is set in the test environment");
        assert_eq!(
            resolve_prefix_cache_directory(None).expect("default cache directory"),
            std::path::PathBuf::from(home).join(".cache/qw/checkpoint")
        );
        validate_cli(&defaults.args).expect("default cache configuration");

        let configured = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--prefix-cache-memory-bytes",
            "4096",
            "--prefix-cache-directory",
            "/tmp/prefixes",
            "--prefix-cache-filesystem-bytes",
            "8192",
        ])
        .expect("CLI");
        assert_eq!(configured.args.prefix_cache_memory_bytes, 4096);
        assert_eq!(
            configured.args.prefix_cache_directory.as_deref(),
            Some(std::path::Path::new("/tmp/prefixes"))
        );
        assert_eq!(configured.args.prefix_cache_filesystem_bytes, 8192);
        validate_cli(&configured.args).expect("configured cache tiers");

        let invalid_memory = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--prefix-cache-memory-bytes",
            "0",
        ])
        .expect("CLI");
        assert_eq!(
            validate_cli(&invalid_memory.args).unwrap_err().to_string(),
            "--prefix-cache-memory-bytes must be greater than zero"
        );

        let invalid_filesystem = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--prefix-cache-filesystem-bytes",
            "0",
        ])
        .expect("CLI");
        assert_eq!(
            validate_cli(&invalid_filesystem.args)
                .unwrap_err()
                .to_string(),
            "--prefix-cache-filesystem-bytes must be greater than zero"
        );

        let help = TestCli::command().render_long_help().to_string();
        assert!(help.contains("--prefix-cache-memory-bytes"), "{help}");
        assert!(help.contains("--prefix-cache-directory"), "{help}");
        assert!(help.contains("--prefix-cache-filesystem-bytes"), "{help}");
        assert!(!help.contains("--prefix-cache-max-tokens"), "{help}");
    }

    #[cfg(not(feature = "specprefill"))]
    #[test]
    fn cli_omits_specprefill_controls_without_feature() {
        let help = TestCli::command().render_long_help().to_string();
        assert!(!help.contains("--specprefill-"), "{help}");
    }

    #[cfg(feature = "specprefill")]
    #[test]
    fn cli_configures_specprefill_policy() {
        let defaults =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        let policy = defaults.args.specprefill_policy();
        assert_eq!(policy.min_turn_tokens, 8_000);
        assert_eq!(policy.keep_rate, 0.25);
        assert_eq!(policy.keep_first_tokens, 256);
        assert_eq!(policy.keep_last_tokens, 256);
        validate_cli(&defaults.args).expect("default SpecPrefill policy");

        let custom = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--specprefill-min-turn-tokens",
            "12000",
            "--specprefill-keep-rate",
            "0.4",
            "--specprefill-keep-first-tokens",
            "0",
            "--specprefill-keep-last-tokens",
            "64",
        ])
        .expect("custom SpecPrefill CLI");
        let policy = custom.args.specprefill_policy();
        assert_eq!(policy.min_turn_tokens, 12_000);
        assert_eq!(policy.keep_rate, 0.4);
        assert_eq!(policy.keep_first_tokens, 0);
        assert_eq!(policy.keep_last_tokens, 64);
        validate_cli(&custom.args).expect("custom SpecPrefill policy");

        let invalid_threshold = TestCli::try_parse_from([
            "qw-server",
            "--specprefill-min-turn-tokens",
            "0",
        ])
        .expect("threshold reaches startup validation");
        assert!(validate_cli(&invalid_threshold.args).is_err());

        for rate in ["0", "1.1", "NaN"] {
            let invalid_rate = TestCli::try_parse_from([
                "qw-server",
                "--specprefill-keep-rate",
                rate,
            ])
            .expect("keep rate reaches startup validation");
            assert!(validate_cli(&invalid_rate.args).is_err(), "rate={rate}");
        }

        let help = TestCli::command().render_long_help().to_string();
        for option in [
            "--specprefill-min-turn-tokens",
            "--specprefill-keep-rate",
            "--specprefill-keep-first-tokens",
            "--specprefill-keep-last-tokens",
        ] {
            assert!(help.contains(option), "{help}");
        }
    }

    #[test]
    fn cli_parses_and_validates_mtp_k() {
        let default =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(default.args.mtp_k, 3);
        validate_cli(&default.args).expect("default MTP K");

        for args in [
            vec!["qw-server", "--model", "/tmp/checkpoint", "--mtp-k=5"],
            vec!["qw-server", "--model", "/tmp/checkpoint", "--mtp-k", "5"],
        ] {
            let cli = TestCli::try_parse_from(args).expect("CLI");
            assert_eq!(cli.args.mtp_k, 5);
            validate_cli(&cli.args).expect("valid MTP K");
        }

        for invalid in [0, 1] {
            let cli = TestCli::try_parse_from([
                "qw-server",
                "--model",
                "/tmp/checkpoint",
                "--mtp-k",
                &invalid.to_string(),
            ])
            .expect("CLI parsing reaches startup validation");
            assert_eq!(
                validate_cli(&cli.args)
                    .expect_err("invalid MTP K")
                    .to_string(),
                "--mtp-k must be at least 2"
            );
        }
    }
}
