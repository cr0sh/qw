use clap::Parser;
use clap_derive::Parser as DeriveParser;
use qw_server::{ServerArgs, serve};

#[derive(Debug, DeriveParser)]
#[command(name = "qw-server", about = "OpenAI-compatible Qwen3.8 Flash Next server")]
struct Cli {
    #[command(flatten)]
    args: ServerArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    serve(Cli::parse().args).await
}
