use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser as _;
use clap_derive::Parser;
use qw_runtime::KVCacheMode;
use qw_server::{Engine, router};
use tracing::info;

#[derive(Debug, Parser)]
#[command(name = "qw-server", about = "OpenAI-compatible dense Qwen3.5 server")]
struct Cli {
    /// Local Qwen3.5 checkpoint directory.
    #[arg(long)]
    model: PathBuf,

    /// Only accept requests for this exact model ID.
    #[arg(long)]
    model_id: Option<String>,

    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8000")]
    bind: String,

    /// Total prompt-token capacity of the shared prefix cache.
    #[arg(long, default_value_t = 32_768)]
    prefix_cache_max_tokens: usize,

    /// MTP verify input block size (bonus token plus proposals).
    #[arg(long = "mtp-k", default_value_t = 3)]
    mtp_k: usize,
    /// Disable the default 4-bit TurboQuant KV cache.
    #[arg(long)]
    no_kv_quantization: bool,
}
impl Cli {
    fn kv_cache_mode(&self) -> KVCacheMode {
        if self.no_kv_quantization {
            KVCacheMode::Fp16
        } else {
            KVCacheMode::Turbo4
        }
    }
}


fn validate_cli(cli: &Cli) -> Result<()> {
    ensure!(
        cli.prefix_cache_max_tokens > 0,
        "--prefix-cache-max-tokens must be greater than zero"
    );
    ensure!(cli.mtp_k >= 2, "--mtp-k must be at least 2");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_cli(&cli)?;
    tracing_subscriber::fmt::init();
    let bind: SocketAddr = cli
        .bind
        .parse()
        .with_context(|| format!("invalid --bind address {:?}", cli.bind))?;
    info!(phase = "server.starting", bind = %bind);

    let kv_cache_mode = cli.kv_cache_mode();
    let engine = Engine::start_qwen(
        cli.model,
        cli.model_id,
        cli.prefix_cache_max_tokens,
        cli.mtp_k,
        kv_cache_mode,
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
    use super::{Cli, validate_cli};
    use qw_runtime::KVCacheMode;
    use clap::{CommandFactory as _, Parser as _};

    #[test]
    fn cli_exposes_optional_model_id() {
        let unrestricted =
            Cli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(unrestricted.model_id, None);

        let configured = Cli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--model-id",
            "served-model",
        ])
        .expect("CLI");
        assert_eq!(configured.model_id.as_deref(), Some("served-model"));

        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--model-id <MODEL_ID>"), "{help}");
    }
    #[test]
    fn turbo4_kv_quantization_is_default_with_explicit_opt_out() {
        let default =
            Cli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(default.kv_cache_mode(), KVCacheMode::Turbo4);

        let unquantized = Cli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--no-kv-quantization",
        ])
        .expect("CLI");
        assert_eq!(unquantized.kv_cache_mode(), KVCacheMode::Fp16);

        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--no-kv-quantization"), "{help}");
        assert!(help.contains("default 4-bit TurboQuant KV cache"), "{help}");
    }


    #[test]
    fn cli_parses_and_validates_mtp_k() {
        let default =
            Cli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(default.mtp_k, 3);
        validate_cli(&default).expect("default MTP K");

        for args in [
            vec!["qw-server", "--model", "/tmp/checkpoint", "--mtp-k=5"],
            vec!["qw-server", "--model", "/tmp/checkpoint", "--mtp-k", "5"],
        ] {
            let cli = Cli::try_parse_from(args).expect("CLI");
            assert_eq!(cli.mtp_k, 5);
            validate_cli(&cli).expect("valid MTP K");
        }

        for invalid in [0, 1] {
            let cli = Cli::try_parse_from([
                "qw-server",
                "--model",
                "/tmp/checkpoint",
                "--mtp-k",
                &invalid.to_string(),
            ])
            .expect("CLI parsing reaches startup validation");
            assert_eq!(
                validate_cli(&cli).expect_err("invalid MTP K").to_string(),
                "--mtp-k must be at least 2"
            );
        }
    }
}
