use std::ffi::OsString;
use std::io::{Error, ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use clap::Parser as _;
use clap_derive::{Args, Parser, Subcommand};
use qw_runtime::{GenerationRequest, KVCacheMode, Qwen35Provider};
use qw_server::serve;

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
    /// Download a Hugging Face model snapshot.
    Download(DownloadArgs),
    /// Generate one response from a local checkpoint.
    Generate(GenerateArgs),
    /// Run the OpenAI-compatible HTTP server.
    Serve(qw_server::ServerArgs),
}

#[derive(Debug, Args)]
struct DownloadArgs {
    /// Hugging Face model identifier, such as Qwen/Qwen3.5-0.8B.
    identifier: String,
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

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

fn validate_identifier(identifier: &str) -> Result<Vec<&str>, Error> {
    let components: Vec<_> = identifier.split('/').collect();
    if components.is_empty()
        || components.len() > 2
        || components
            .iter()
            .any(|component| component.is_empty() || *component == "." || *component == "..")
        || identifier.starts_with('/')
        || identifier.contains('\\')
    {
        return Err(invalid_input(format!(
            "invalid Hugging Face model identifier `{identifier}`"
        )));
    }
    Ok(components)
}

fn model_cache_path(home: &Path, identifier: &str) -> Result<PathBuf, Error> {
    let mut destination = home.join(".cache/qw/models");
    for component in validate_identifier(identifier)? {
        destination.push(component);
    }
    Ok(destination)
}

fn sibling_path(destination: &Path, filename: &str) -> Result<PathBuf, Error> {
    if filename.is_empty()
        || filename.starts_with('/')
        || filename.contains('\\')
        || filename
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(invalid_input(format!(
            "model API returned unsafe filename `{filename}`"
        )));
    }
    Ok(destination.join(filename))
}

fn encode_url_path(path: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
    }
    encoded
}

fn add_hf_token(command: &mut ProcessCommand) {
    if let Some(token) = std::env::var_os("HF_TOKEN") {
        let mut header = OsString::from("Authorization: Bearer ");
        header.push(token);
        command.arg("--header").arg(header);
    }
}

fn download_model(identifier: &str) -> Result<(), Box<dyn std::error::Error>> {
    validate_identifier(identifier)?;
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "HOME is not set"))?;
    let destination = model_cache_path(Path::new(&home), identifier)?;
    std::fs::create_dir_all(&destination)?;

    eprintln!("Fetching file list for {identifier}");
    let api_url = format!(
        "https://huggingface.co/api/models/{}",
        encode_url_path(identifier)
    );
    let mut api_command = ProcessCommand::new("curl");
    api_command.args(["--fail", "--silent", "--show-error", "--location"]);
    add_hf_token(&mut api_command);
    let output = api_command.arg(&api_url).output().map_err(|error| {
        Error::new(
            error.kind(),
            format!("failed to run curl for Hugging Face model API: {error}"),
        )
    })?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(Error::other(format!(
            "Hugging Face model API request failed ({}): {}",
            output.status,
            detail.trim()
        ))
        .into());
    }

    let response: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let siblings = response
        .get("siblings")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::other("Hugging Face model API response did not contain `siblings`"))?;

    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::other("Hugging Face model API returned a sibling without `rfilename`")
            })?;
        let file_path = sibling_path(&destination, filename)?;
        if file_path.is_file() {
            eprintln!("Already downloaded {filename}");
            continue;
        }
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut partial_name = file_path
            .file_name()
            .expect("validated sibling path has a filename")
            .to_os_string();
        partial_name.push(".qw-part");
        let partial_path = file_path.with_file_name(partial_name);
        let file_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            encode_url_path(identifier),
            encode_url_path(filename)
        );
        eprintln!("Downloading {filename}");
        let mut command = ProcessCommand::new("curl");
        command.args(["--fail", "--location", "--show-error"]);
        if partial_path
            .metadata()
            .map(|metadata| metadata.len() > 0)
            .unwrap_or(false)
        {
            command.args(["--continue-at", "-"]);
        }
        add_hf_token(&mut command);
        let status = command
            .arg("--output")
            .arg(&partial_path)
            .arg(&file_url)
            .status()
            .map_err(|error| {
                Error::new(
                    error.kind(),
                    format!("failed to run curl while downloading `{filename}`: {error}"),
                )
            })?;
        if !status.success() {
            return Err(Error::other(format!(
                "curl failed while downloading `{filename}` ({status})"
            ))
            .into());
        }
        std::fs::rename(&partial_path, &file_path)?;
    }

    eprintln!("Downloaded {identifier} to {}", destination.display());
    Ok(())
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Download(args) => {
            download_model(&args.identifier)?;
        }
        Command::Generate(args) => {
            eprintln!("Loading model from {}", args.model.display());
            let mut provider = Qwen35Provider::load(&args.model, KVCacheMode::Fp16)?;
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            let mut io_error = None;
            let generation = provider.generate_streaming(&args.request(), |delta| {
                if delta.is_empty() {
                    return true;
                }
                if let Err(error) = stdout
                    .write_all(delta.as_bytes())
                    .and_then(|()| stdout.flush())
                {
                    io_error = Some(error);
                    false
                } else {
                    true
                }
            });
            if let Some(error) = io_error {
                return Err(Box::new(error));
            }
            generation?;
        }
        Command::Serve(args) => {
            serve(args).await?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run(Cli::parse()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory as _;
    #[test]
    fn download_accepts_positional_identifier() {
        let cli = Cli::try_parse_from(["qw", "download", "Qwen/Qwen3.5-0.8B"])
            .expect("parse download command");
        let Command::Download(args) = cli.command else {
            panic!("expected download command");
        };
        assert_eq!(args.identifier, "Qwen/Qwen3.5-0.8B");
    }

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
    fn sibling_paths_cannot_escape_destination() {
        let destination = Path::new("/home/user/.cache/qw/models/Qwen/model");
        assert_eq!(
            sibling_path(destination, "weights/model.safetensors").expect("safe sibling"),
            destination.join("weights/model.safetensors")
        );
        for filename in ["", "/etc/passwd", "../token", "weights/../../token", r"..\token"] {
            assert!(
                sibling_path(destination, filename).is_err(),
                "{filename:?} should be rejected"
            );
        }
    }

    #[test]
    fn generate_requires_model_and_prompt() {
        assert!(Cli::try_parse_from(["qw", "generate"]).is_err());
        assert!(Cli::try_parse_from(["qw", "generate", "--model", "/tmp/model"]).is_err());
        assert!(Cli::try_parse_from(["qw", "generate", "--prompt", "hello"]).is_err());
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
        let Command::Generate(args) = cli.command else {
            panic!("expected generate command");
        };
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
        let Command::Generate(args) = cli.command else {
            panic!("expected generate command");
        };
        let request = args.request();
        assert_eq!(request.max_tokens, 9);
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_k, Some(11));
        assert_eq!(request.top_p, Some(0.8));
        assert_eq!(request.seed, Some(42));
    }

    #[test]
    fn serve_accepts_server_options() {
        let cli = Cli::try_parse_from([
            "qw",
            "serve",
            "--model",
            "/tmp/model",
            "--model-id",
            "served-model",
            "--bind",
            "127.0.0.1:9000",
            "--prefix-cache-memory-bytes",
            "4096",
            "--prefix-cache-directory",
            "/tmp/prefixes",
            "--prefix-cache-filesystem-bytes",
            "8192",
            "--mtp-k",
            "5",
            "--no-kv-quantization",
            "--output-format",
            "json",
        ])
        .expect("parse serve command");
        assert!(matches!(cli.command, Command::Serve(_)));
    }

    #[test]
    fn help_documents_serve_command_and_server_options() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("serve"), "{help}");

        let serve_help = Cli::try_parse_from(["qw", "serve", "--help"])
            .expect_err("serve help exits through clap");
        let serve_help = serve_help.to_string();
        assert!(serve_help.contains("--model"), "{serve_help}");
        assert!(serve_help.contains("--bind"), "{serve_help}");
        assert!(serve_help.contains("--model-id"), "{serve_help}");
        assert!(
            serve_help.contains("--prefix-cache-memory-bytes"),
            "{serve_help}"
        );
        assert!(
            serve_help.contains("--prefix-cache-filesystem-bytes"),
            "{serve_help}"
        );
        assert!(serve_help.contains("--mtp-k"), "{serve_help}");
        assert!(serve_help.contains("--no-kv-quantization"), "{serve_help}");
        assert!(serve_help.contains("--output-format"), "{serve_help}");
    }
}
