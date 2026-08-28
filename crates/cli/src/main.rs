use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{Error, ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::thread;

use clap::Parser as _;
use clap_derive::{Args, Parser, Subcommand};
use qw_runtime::{
    DEFAULT_MODEL_IDENTIFIER, GenerationRequest, KVCacheMode, Qwen4Provider, model_cache_path,
    resolve_model_path, validate_identifier,
};
use qw_server::serve;

#[derive(Debug, Parser)]
#[command(
    name = "qw",
    about = "Local Qwen3.8 Flash Next REAP inference",
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
    /// Show local cache usage and model resolver status.
    Stats,
    /// Generate one response from a local checkpoint.
    Generate(GenerateArgs),
    /// Run the OpenAI-compatible HTTP server.
    Serve(qw_server::ServerArgs),
}

#[derive(Debug, Args)]
struct DownloadArgs {
    /// Hugging Face model identifier; defaults to the fixed Qwen3.8 Flash Next REAP model.
    identifier: Option<String>,
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Checkpoint directory or HF identifier; defaults to the qw model cache unless QW_MODEL_PATH is set.
    #[arg(long)]
    model: Option<PathBuf>,

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
fn cache_usage(path: &Path) -> Result<(u64, u64), Error> {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok((0, 0)),
        Err(error) => return Err(error),
    };
    let mut files = 0_u64;
    let mut bytes = 0_u64;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            let (child_files, child_bytes) = cache_usage(&entry.path())?;
            files += child_files;
            bytes += child_bytes;
        } else if file_type.is_file() {
            files += 1;
            bytes += entry.metadata()?.len();
        }
    }
    Ok((files, bytes))
}

fn human_readable_bytes(bytes: u64) -> String {
    const UNITS: [&str; 7] = ["B", "K", "M", "G", "T", "P", "E"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1_000.0 && unit < UNITS.len() - 1 {
        value /= 1_000.0;
        unit += 1;
    }
    format!("{value:.2}{}", UNITS[unit])
}

fn stats_report() -> Result<String, Box<dyn std::error::Error>> {
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "HOME is not set"))?;
    let cache_root = home.join(".cache/qw");
    let (cache_files, cache_bytes) = cache_usage(&cache_root)?;
    let (model_cache_files, model_cache_bytes) = cache_usage(&cache_root.join("models"))?;
    let (checkpoint_cache_files, checkpoint_cache_bytes) =
        cache_usage(&cache_root.join("checkpoint"))?;
    let model_path = resolve_model_path(None)?;
    let model_override = std::env::var_os("QW_MODEL_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let selected_model = model_override.as_deref().map_or_else(
        || DEFAULT_MODEL_IDENTIFIER.to_owned(),
        |path| path.display().to_string(),
    );
    let model_exists = model_path.try_exists()?;

    let mut report = String::new();
    writeln!(report, "Cache root: {}", cache_root.display())?;
    writeln!(
        report,
        "Cache usage: {cache_files} files, {cache_bytes} bytes ({})",
        human_readable_bytes(cache_bytes)
    )?;
    writeln!(
        report,
        "Model cache usage: {model_cache_files} files, {model_cache_bytes} bytes ({})",
        human_readable_bytes(model_cache_bytes)
    )?;
    writeln!(
        report,
        "Checkpoint cache usage: {checkpoint_cache_files} files, {checkpoint_cache_bytes} bytes ({})",
        human_readable_bytes(checkpoint_cache_bytes)
    )?;
    writeln!(report, "Selected model: {selected_model}")?;
    writeln!(report, "Model path: {}", model_path.display())?;
    writeln!(
        report,
        "Model exists locally: {}",
        if model_exists { "yes" } else { "no" }
    )?;
    Ok(report)
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

const MAX_CONCURRENT_DOWNLOADS: usize = 4;

struct DownloadJob {
    filename: String,
    file_path: PathBuf,
    partial_path: PathBuf,
}

fn run_bounded<T, F>(items: &[T], limit: usize, operation: &F) -> Result<(), Error>
where
    T: Sync,
    F: Fn(&T) -> Result<(), Error> + Sync,
{
    assert!(limit > 0, "concurrency limit must be positive");
    for batch in items.chunks(limit) {
        thread::scope(|scope| {
            let handles: Vec<_> = batch
                .iter()
                .map(|item| scope.spawn(move || operation(item)))
                .collect();
            let mut first_error = None;
            for handle in handles {
                let result = handle
                    .join()
                    .unwrap_or_else(|_| Err(Error::other("download worker panicked")));
                if first_error.is_none() {
                    first_error = result.err();
                }
            }
            first_error.map_or(Ok(()), Err)
        })?;
    }
    Ok(())
}

fn download_sibling(identifier: &str, job: &DownloadJob) -> Result<(), Error> {
    let file_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        encode_url_path(identifier),
        encode_url_path(&job.filename)
    );
    eprintln!("Downloading {}", job.filename);
    let mut command = ProcessCommand::new("curl");
    command.args(["--fail", "--location", "--show-error"]);
    if job
        .partial_path
        .metadata()
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
    {
        command.args(["--continue-at", "-"]);
    }
    add_hf_token(&mut command);
    let status = command
        .arg("--output")
        .arg(&job.partial_path)
        .arg(&file_url)
        .status()
        .map_err(|error| {
            Error::new(
                error.kind(),
                format!(
                    "failed to run curl while downloading `{}`: {error}",
                    job.filename
                ),
            )
        })?;
    if !status.success() {
        return Err(Error::other(format!(
            "curl failed while downloading `{}` ({status})",
            job.filename
        )));
    }
    std::fs::rename(&job.partial_path, &job.file_path)
}

fn download_snapshot(identifier: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        .ok_or_else(|| {
            Error::other("Hugging Face model API response did not contain `siblings`")
        })?;

    let mut jobs = Vec::with_capacity(siblings.len());
    let mut seen = HashSet::with_capacity(siblings.len());
    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::other("Hugging Face model API returned a sibling without `rfilename`")
            })?;
        let file_path = sibling_path(&destination, filename)?;
        if !seen.insert(file_path.clone()) {
            continue;
        }
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
        jobs.push(DownloadJob {
            filename: filename.to_owned(),
            file_path,
            partial_path,
        });
    }

    run_bounded(&jobs, MAX_CONCURRENT_DOWNLOADS, &|job| {
        download_sibling(identifier, job)
    })?;

    eprintln!("Downloaded {identifier} to {}", destination.display());
    Ok(())
}

fn resolve_download_identifier(identifier: Option<&str>) -> &str {
    identifier.unwrap_or(DEFAULT_MODEL_IDENTIFIER)
}

fn download_model(identifier: &str) -> Result<(), Box<dyn std::error::Error>> {
    validate_identifier(identifier)?;
    download_snapshot(identifier)
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Download(args) => {
            download_model(resolve_download_identifier(args.identifier.as_deref()))?;
        }
        Command::Stats => {
            print!("{}", stats_report()?);
        }
        Command::Generate(args) => {
            let model = qw_runtime::resolve_model_path(args.model.as_deref())?;
            eprintln!("Loading model from {}", model.display());
            let mut provider = Qwen4Provider::load(&model, KVCacheMode::Turbo8)?;
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn bounded_runner_visits_each_item_without_exceeding_limit() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let completed = AtomicUsize::new(0);
        let items = [0; 12];

        run_bounded(&items, 4, &|_| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            active.fetch_sub(1, Ordering::SeqCst);
            completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .expect("bounded work succeeds");

        assert_eq!(completed.load(Ordering::SeqCst), items.len());
        assert!(peak.load(Ordering::SeqCst) <= 4);
        assert!(peak.load(Ordering::SeqCst) > 1);
    }

    #[test]
    fn bounded_runner_propagates_worker_failure() {
        let completed = AtomicUsize::new(0);
        let error = run_bounded(&[0, 1, 2], 2, &|item| {
            completed.fetch_add(1, Ordering::SeqCst);
            if *item == 0 {
                Err(Error::other("expected failure"))
            } else {
                Ok(())
            }
        })
        .expect_err("worker failure should be returned");

        assert_eq!(error.to_string(), "expected failure");
        assert_eq!(completed.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn download_defaults_to_resolver_model() {
        let cli = Cli::try_parse_from(["qw", "download"]).expect("parse download command");
        let Command::Download(args) = cli.command else {
            panic!("expected download command");
        };
        assert_eq!(args.identifier, None);
        assert_eq!(
            resolve_download_identifier(args.identifier.as_deref()),
            DEFAULT_MODEL_IDENTIFIER
        );
    }

    #[test]
    fn download_accepts_explicit_identifier_override() {
        let cli = Cli::try_parse_from(["qw", "download", "Qwen/Qwen4-0.8B"])
            .expect("parse download command");
        let Command::Download(args) = cli.command else {
            panic!("expected download command");
        };
        assert_eq!(args.identifier.as_deref(), Some("Qwen/Qwen4-0.8B"));
        assert_eq!(
            resolve_download_identifier(args.identifier.as_deref()),
            "Qwen/Qwen4-0.8B"
        );
    }
    #[test]
    fn sibling_paths_cannot_escape_destination() {
        let destination = Path::new("/home/user/.cache/qw/models/Qwen/model");
        assert_eq!(
            sibling_path(destination, "weights/model.safetensors").expect("safe sibling"),
            destination.join("weights/model.safetensors")
        );
        for filename in [
            "",
            "/etc/passwd",
            "../token",
            "weights/../../token",
            r"..\token",
        ] {
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
        let cli = Cli::try_parse_from(["qw", "generate", "--prompt", "hello"])
            .expect("model is optional");
        assert!(matches!(cli.command, Command::Generate(args) if args.model.is_none()));
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
    fn help_documents_stats_and_serve_commands_and_server_options() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("stats"), "{help}");
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
