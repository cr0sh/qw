use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{Error, ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::thread;

use clap::Parser as _;
use clap_derive::{Args, Parser, Subcommand};
use qw_runtime::{
    GenerationRequest, KVCacheMode, PINNED_ARTIFACTS, PINNED_REPOSITORY, PINNED_REVISION,
    PinnedArtifact, Qwen35Provider, io_verify_artifact_file, pinned_model_dir,
    resolve_pinned_model_dir,
};
use qw_server::serve;

#[derive(Debug, Parser)]
#[command(
    name = "qw",
    about = "Local Qwen3.8 27B GGUF inference",
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
struct DownloadArgs {}

#[derive(Debug, Args)]
struct GenerateArgs {
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
    let model_path = resolve_pinned_model_dir()?;
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
    writeln!(
        report,
        "Pinned model: {PINNED_REPOSITORY}@{PINNED_REVISION}"
    )?;
    writeln!(report, "Model path: {}", model_path.display())?;
    writeln!(
        report,
        "Model exists locally: {}",
        if model_exists { "yes" } else { "no" }
    )?;
    Ok(report)
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

const MAX_CONCURRENT_DOWNLOADS: usize = 2;

type SelectedFile = PinnedArtifact;

struct DownloadJob {
    selected: SelectedFile,
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

fn download_selected_file(job: &DownloadJob) -> Result<(), Error> {
    let file_url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        encode_url_path(PINNED_REPOSITORY),
        PINNED_REVISION,
        encode_url_path(job.selected.source_path),
    );
    eprintln!("Downloading {}", job.selected.relative_path);
    let mut command = ProcessCommand::new("curl");
    command.args(["--fail", "--location", "--show-error"]);
    let resume = match std::fs::symlink_metadata(&job.partial_path) {
        Ok(metadata) => {
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                return Err(Error::other(format!(
                    "partial download path is not a regular file: {}",
                    job.partial_path.display()
                )));
            }
            if metadata.len() > job.selected.size {
                return Err(Error::other(format!(
                    "partial download {} exceeds pinned size {}",
                    job.partial_path.display(),
                    job.selected.size
                )));
            }
            metadata.len() > 0
        }
        Err(error) if error.kind() == ErrorKind::NotFound => false,
        Err(error) => return Err(error),
    };
    if resume {
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
                    job.selected.relative_path
                ),
            )
        })?;
    if !status.success() {
        return Err(Error::other(format!(
            "curl failed while downloading `{}` ({status})",
            job.selected.relative_path
        )));
    }
    if let Err(error) = verify_selected_file(&job.partial_path, job.selected) {
        let _ = std::fs::remove_file(&job.partial_path);
        return Err(error);
    }
    std::fs::rename(&job.partial_path, &job.file_path)
}

fn verify_selected_file(path: &Path, selected: SelectedFile) -> Result<(), Error> {
    io_verify_artifact_file(path, selected)
}

fn download_selected_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "HOME is not set"))?;
    let destination = pinned_model_dir(Path::new(&home));
    std::fs::create_dir_all(&destination)?;
    let mut jobs = Vec::new();
    for selected in PINNED_ARTIFACTS {
        let file_path = destination.join(selected.relative_path);
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if file_path.exists() {
            verify_selected_file(&file_path, selected)?;
            eprintln!("Verified {}", selected.relative_path);
            continue;
        }
        let mut partial_name = file_path
            .file_name()
            .expect("fixed selected path has a filename")
            .to_os_string();
        partial_name.push(".qw-part");
        let partial_path = file_path.with_file_name(partial_name);
        jobs.push(DownloadJob {
            selected,
            file_path,
            partial_path,
        });
    }
    run_bounded(&jobs, MAX_CONCURRENT_DOWNLOADS, &download_selected_file)?;
    for selected in PINNED_ARTIFACTS {
        verify_selected_file(&destination.join(selected.relative_path), selected)?;
    }
    write_selected_provenance(&destination)?;
    eprintln!(
        "Downloaded {}@{} to {}",
        PINNED_REPOSITORY,
        PINNED_REVISION,
        destination.display()
    );
    Ok(())
}

fn write_selected_provenance(destination: &Path) -> Result<(), Error> {
    let files = PINNED_ARTIFACTS
        .iter()
        .map(|file| {
            serde_json::json!({
                "repository": PINNED_REPOSITORY,
                "revision": PINNED_REVISION,
                "source": file.source_path,
                "relative": file.relative_path,
                "size": file.size,
                "sha256": file.sha256,
            })
        })
        .collect::<Vec<_>>();
    let provenance = serde_json::json!({
        "schema": "qwr.pinned-gguf-pair",
        "repository": PINNED_REPOSITORY,
        "revision": PINNED_REVISION,
        "files": files,
    });
    let bytes = serde_json::to_vec_pretty(&provenance).map_err(Error::other)?;
    let partial = destination.join(format!(".provenance-{}.qw-part", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(partial, destination.join("provenance.json"))
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Download(_) => {
            download_selected_snapshot()?;
        }
        Command::Stats => {
            print!("{}", stats_report()?);
        }
        Command::Generate(args) => {
            let model = resolve_pinned_model_dir()?;
            eprintln!("Loading pinned model from {}", model.display());
            let mut provider = Qwen35Provider::load(KVCacheMode::Fp16)?;
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
    fn download_is_fixed_to_the_pinned_gguf_pair() {
        let cli = Cli::try_parse_from(["qw", "download"]).expect("parse download command");
        assert!(matches!(cli.command, Command::Download(_)));
        assert!(
            Cli::try_parse_from(["qw", "download", "Qwen/Qwen3.5-0.8B"]).is_err(),
            "the pinned checkpoint contract has no identifier override"
        );
        assert_eq!(PINNED_ARTIFACTS.len(), 2);
        assert_eq!(PINNED_ARTIFACTS[0].role, PinnedArtifactRole::Target);
        assert_eq!(PINNED_ARTIFACTS[1].role, PinnedArtifactRole::Mtp);
    }

    #[test]
    fn generate_requires_prompt_and_rejects_model_overrides() {
        assert!(Cli::try_parse_from(["qw", "generate"]).is_err());
        assert!(Cli::try_parse_from(["qw", "generate", "--model", "/tmp/model"]).is_err());
        let cli = Cli::try_parse_from(["qw", "generate", "--prompt", "hello"])
            .expect("parse fixed-model generation");
        assert!(matches!(cli.command, Command::Generate(_)));
    }

    #[test]
    fn omitted_sampling_flags_use_checkpoint_defaults() {
        let cli = Cli::try_parse_from(["qw", "generate", "--prompt", "hello"])
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

    #[cfg(feature = "specprefill")]
    #[test]
    fn serve_accepts_server_options() {
        let cli = Cli::try_parse_from([
            "qw",
            "serve",
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
            "--specprefill-min-turn-tokens",
            "12000",
            "--specprefill-keep-rate",
            "0.4",
            "--specprefill-keep-first-tokens",
            "0",
            "--specprefill-keep-last-tokens",
            "64",
            "--no-kv-quantization",
            "--output-format",
            "json",
        ])
        .expect("parse serve command");
        assert!(matches!(cli.command, Command::Serve(_)));
    }
    #[test]
    fn help_documents_stats_and_serve_commands_and_server_options() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("stats"), "{help}");
        assert!(help.contains("serve"), "{help}");

        let serve_help = Cli::try_parse_from(["qw", "serve", "--help"])
            .expect_err("serve help exits through clap");
        let serve_help = serve_help.to_string();
        assert!(!serve_help.contains("--model <"), "{serve_help}");
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
        #[cfg(feature = "specprefill")]
        assert!(
            serve_help.contains("--specprefill-min-turn-tokens"),
            "{serve_help}"
        );
        #[cfg(feature = "specprefill")]
        assert!(
            serve_help.contains("--specprefill-keep-rate"),
            "{serve_help}"
        );
        #[cfg(feature = "specprefill")]
        assert!(
            serve_help.contains("--specprefill-keep-first-tokens"),
            "{serve_help}"
        );
        #[cfg(feature = "specprefill")]
        assert!(
            serve_help.contains("--specprefill-keep-last-tokens"),
            "{serve_help}"
        );
        #[cfg(not(feature = "specprefill"))]
        assert!(!serve_help.contains("--specprefill-"), "{serve_help}");
        assert!(serve_help.contains("--no-kv-quantization"), "{serve_help}");
        assert!(serve_help.contains("--output-format"), "{serve_help}");
    }
}
