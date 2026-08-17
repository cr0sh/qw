use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser as _;
use clap_derive::Parser;
use qw_server::{Engine, router};

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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.prefix_cache_max_tokens > 0,
        "--prefix-cache-max-tokens must be greater than zero"
    );
    let bind: SocketAddr = cli
        .bind
        .parse()
        .with_context(|| format!("invalid --bind address {:?}", cli.bind))?;

    let engine = Engine::start_qwen(cli.model, cli.model_id, cli.prefix_cache_max_tokens)?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;
    axum::serve(listener, router(engine))
        .await
        .context("HTTP server failed")
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory as _, Parser as _};

    use super::Cli;

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
}
