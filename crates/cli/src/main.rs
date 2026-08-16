use std::path::PathBuf;

use clap::Parser as _;
use clap_derive::{Args, Parser, Subcommand};
use qw_runtime::{GenerationRequest, Qwen35Provider};

#[derive(Debug, Parser)]
#[command(
    name = "qw",
    about = "Local dense Qwen3.5 inference",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate one response from a local checkpoint.
    Generate(GenerateArgs),
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Local Qwen3.5 checkpoint directory.
    #[arg(long)]
    model: PathBuf,

    /// User prompt text.
    #[arg(long)]
    prompt: String,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Sampling temperature; the checkpoint default is used when omitted.
    #[arg(long)]
    temperature: Option<f32>,

    /// Top-k sampling cutoff; the checkpoint default is used when omitted.
    #[arg(long)]
    top_k: Option<i32>,

    /// Top-p sampling cutoff; the checkpoint default is used when omitted.
    #[arg(long)]
    top_p: Option<f32>,

    /// Sampling seed.
    #[arg(long)]
    seed: Option<u64>,
}

impl GenerateArgs {
    fn request(&self) -> GenerationRequest {
        GenerationRequest {
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Generate(args) => {
            eprintln!("Loading model from {}", args.model.display());
            let mut provider = Qwen35Provider::load(&args.model)?;
            let output = provider.generate(&args.request())?;
            print!("{}", output.text);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run(Cli::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_requires_model_and_prompt() {
        assert!(Cli::try_parse_from(["qw", "generate"]).is_err());
        assert!(
            Cli::try_parse_from(["qw", "generate", "--model", "/tmp/model"]).is_err()
        );
        assert!(
            Cli::try_parse_from(["qw", "generate", "--prompt", "hello"]).is_err()
        );
    }

    #[test]
    fn omitted_sampling_flags_use_checkpoint_defaults() {
        let cli = Cli::try_parse_from([
            "qw",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
        ])
        .expect("parse generate command");
        let Command::Generate(args) = cli.command;
        let request = args.request();
        assert_eq!(request.max_tokens, 128);
        assert_eq!(request.temperature, None);
        assert_eq!(request.top_k, None);
        assert_eq!(request.top_p, None);
        assert_eq!(request.seed, None);
    }

    #[test]
    fn all_generation_flags_map_to_the_request() {
        let cli = Cli::try_parse_from([
            "qw",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--max-tokens",
            "9",
            "--temperature",
            "0.7",
            "--top-k",
            "11",
            "--top-p",
            "0.8",
            "--seed",
            "42",
        ])
        .expect("parse generate command");
        let Command::Generate(args) = cli.command;
        let request = args.request();
        assert_eq!(request.max_tokens, 9);
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_k, Some(11));
        assert_eq!(request.top_p, Some(0.8));
        assert_eq!(request.seed, Some(42));
    }
}
