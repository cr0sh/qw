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

    let engine = Engine::start_qwen(cli.model, cli.prefix_cache_max_tokens)?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;
    axum::serve(listener, router(engine))
        .await
        .context("HTTP server failed")
}
