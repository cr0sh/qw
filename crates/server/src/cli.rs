use std::net::SocketAddr;
use std::path::PathBuf;

#[cfg(feature = "specprefill")]
use crate::SpecPrefillPolicyConfig;
use crate::{Engine, router};
use anyhow::{Context, Result, ensure};
use clap_derive::{Args as DeriveArgs, ValueEnum};
use qw_prefix_cache::CacheConfig;
#[cfg(feature = "dflash2")]
use qw_runtime::resolve_dflash2_draft_path;
use qw_runtime::{
    DEFAULT_DECODER_CROSSOVER_TOKENS, KVCacheMode, Qwen35GenerationMode, resolve_model_path,
};
use tracing::info;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::Layer as _;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, fmt, registry};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Decoder {
    #[value(name = "auto")]
    Automatic,
    Baseline,
    Mtp,
    #[value(name = "dflash")]
    Dflash2,
}

impl Decoder {
    pub fn generation_mode(self) -> Qwen35GenerationMode {
        match self {
            Self::Automatic => Qwen35GenerationMode::Automatic,
            Self::Baseline => Qwen35GenerationMode::Baseline,
            Self::Mtp => Qwen35GenerationMode::Mtp,
            Self::Dflash2 => Qwen35GenerationMode::Dflash2,
        }
    }
}

fn parse_si_bytes(value: &str) -> Result<u64, String> {
    let value_without_b = value.strip_suffix(['B', 'b']).unwrap_or(value);
    let (number, multiplier) = match value_without_b.as_bytes().last().copied() {
        Some(b'K' | b'k') => (&value_without_b[..value_without_b.len() - 1], 1_000_u64),
        Some(b'M' | b'm') => (&value_without_b[..value_without_b.len() - 1], 1_000_000),
        Some(b'G' | b'g') => (&value_without_b[..value_without_b.len() - 1], 1_000_000_000),
        Some(b'T' | b't') => (
            &value_without_b[..value_without_b.len() - 1],
            1_000_000_000_000,
        ),
        Some(b'P' | b'p') => (
            &value_without_b[..value_without_b.len() - 1],
            1_000_000_000_000_000,
        ),
        Some(b'E' | b'e') => (
            &value_without_b[..value_without_b.len() - 1],
            1_000_000_000_000_000_000,
        ),
        _ => (value_without_b, 1),
    };
    let invalid = || {
        format!(
            "invalid byte size '{value}': expected a non-negative integer or decimal with an optional SI suffix K, M, G, T, P, or E (optional B)"
        )
    };
    let (whole, fraction) = number
        .split_once('.')
        .map_or((number, None), |(whole, fraction)| (whole, Some(fraction)));
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|fraction| {
            fraction.is_empty()
                || fraction.len() > 2
                || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return Err(invalid());
    }
    let whole = whole.parse::<u64>().map_err(|_| invalid())?;
    let mut bytes = whole
        .checked_mul(multiplier)
        .ok_or_else(|| format!("byte size '{value}' overflows u64"))?;
    if let Some(fraction) = fraction {
        if multiplier == 1 {
            return Err(format!(
                "byte size '{value}' is not a whole number of bytes"
            ));
        }
        let denominator = 10_u64.pow(fraction.len() as u32);
        let numerator = fraction.parse::<u64>().map_err(|_| invalid())?;
        let scaled = numerator
            .checked_mul(multiplier)
            .ok_or_else(|| format!("byte size '{value}' overflows u64"))?;
        if scaled % denominator != 0 {
            return Err(format!(
                "byte size '{value}' is not a whole number of bytes"
            ));
        }
        bytes = bytes
            .checked_add(scaled / denominator)
            .ok_or_else(|| format!("byte size '{value}' overflows u64"))?;
    }
    Ok(bytes)
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

    /// Disable prefix snapshot lookup and persistence.
    #[arg(long)]
    disable_prefix_cache: bool,

    /// Byte capacity of the in-memory prefix snapshot tier (decimal SI suffixes K-E accepted).
    #[arg(
        long,
        default_value_t = 2 * 1024 * 1024 * 1024_u64,
        value_parser = parse_si_bytes
    )]
    prefix_cache_memory_bytes: u64,

    /// Directory for persistent prefix snapshots; defaults to `~/.cache/qw/checkpoint`.
    #[arg(long)]
    prefix_cache_directory: Option<PathBuf>,

    /// Byte capacity of the filesystem prefix snapshot tier (decimal SI suffixes K-E accepted).
    #[arg(
        long,
        default_value_t = 16 * 1024 * 1024 * 1024_u64,
        value_parser = parse_si_bytes
    )]
    prefix_cache_filesystem_bytes: u64,

    /// MTP verify input block size (bonus token plus proposals).
    #[arg(long = "mtp-k", default_value_t = 3)]
    mtp_k: usize,

    /// Decoder policy. Explicit values ignore the context crossover.
    #[arg(long, value_enum, default_value = "auto")]
    decoder: Decoder,

    /// Prompt-token boundary where automatic routing changes from MTP to DFlash.
    #[arg(long, default_value_t = DEFAULT_DECODER_CROSSOVER_TOKENS)]
    decoder_crossover_tokens: usize,

    /// DFlash2 draft checkpoint directory; defaults to QW_DFLASH_DRAFT_MODEL_PATH or the model cache.
    #[arg(long)]
    dflash_draft_model: Option<PathBuf>,

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

    /// Disable the daily trace log file under `~/.cache/qw/log/`.
    #[arg(long)]
    no_file_logging: bool,

    /// Persistent log filter directives; overrides QW_LOG and defaults to `debug,qw_server=trace,qw_runtime=trace,qw_prefix_cache=trace,mlxcel_core=trace,qw_cli=trace,tokenizers=info`.
    #[arg(long)]
    persistent_log_filter: Option<String>,

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
    ensure!(
        cli.decoder_crossover_tokens > 0,
        "--decoder-crossover-tokens must be greater than zero"
    );
    #[cfg(not(feature = "dflash2"))]
    ensure!(
        cli.decoder != Decoder::Dflash2,
        "--decoder dflash requires the dflash2 build feature"
    );
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

const DEFAULT_PERSISTENT_LOG_FILTER: &str =
    "debug,qw_server=trace,qw_runtime=trace,qw_prefix_cache=trace,qw_cli=trace,tokenizers=info";

fn resolve_persistent_log_filter(
    cli_filter: Option<&str>,
    env_filter: Option<&str>,
) -> Result<EnvFilter> {
    let (source, directives) = if let Some(filter) = cli_filter {
        ("--persistent-log-filter", filter)
    } else if let Some(filter) = env_filter {
        ("QW_LOG", filter)
    } else {
        ("default", DEFAULT_PERSISTENT_LOG_FILTER)
    };
    EnvFilter::try_new(directives).with_context(|| {
        format!("invalid persistent log filter directives from {source}: {directives:?}")
    })
}

fn rust_log_filter() -> EnvFilter {
    EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
}

fn persistent_log_format() -> fmt::format::Format<fmt::format::Json> {
    fmt::format().json()
}

fn init_tracing(cli: &ServerArgs) -> Result<Option<WorkerGuard>> {
    let file_guard = if cli.no_file_logging {
        None
    } else {
        let home = std::env::var_os("HOME")
            .filter(|home| !home.is_empty())
            .context("HOME is not set; cannot resolve the trace log directory")?;
        let log_directory = PathBuf::from(home).join(".cache/qw/log");
        std::fs::create_dir_all(&log_directory)
            .with_context(|| format!("failed to create {}", log_directory.display()))?;
        let appender = tracing_appender::rolling::daily(log_directory, "qw.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        let env_filter = std::env::var("QW_LOG").ok();
        let persistent_filter = resolve_persistent_log_filter(
            cli.persistent_log_filter.as_deref(),
            env_filter.as_deref(),
        )?;
        match cli.output_format {
            OutputFormat::Human => registry()
                .with(fmt::layer().with_filter(rust_log_filter()))
                .with(
                    fmt::layer()
                        .fmt_fields(fmt::format::JsonFields::new())
                        .event_format(persistent_log_format())
                        .with_ansi(false)
                        .with_writer(non_blocking.clone())
                        .with_filter(persistent_filter),
                )
                .init(),
            OutputFormat::Json => registry()
                .with(fmt::layer().json().with_filter(rust_log_filter()))
                .with(
                    fmt::layer()
                        .fmt_fields(fmt::format::JsonFields::new())
                        .event_format(persistent_log_format())
                        .with_ansi(false)
                        .with_writer(non_blocking)
                        .with_filter(persistent_filter),
                )
                .init(),
        }
        Some(guard)
    };
    if cli.no_file_logging {
        match cli.output_format {
            OutputFormat::Human => fmt::Subscriber::builder()
                .with_env_filter(rust_log_filter())
                .init(),
            OutputFormat::Json => fmt::Subscriber::builder()
                .with_env_filter(rust_log_filter())
                .json()
                .init(),
        }
    }
    Ok(file_guard)
}

fn resolve_prefix_cache_directory(directory: Option<PathBuf>) -> Result<PathBuf> {
    directory
        .map(Ok)
        .unwrap_or_else(default_prefix_cache_directory)
}
pub async fn serve(cli: ServerArgs) -> Result<()> {
    validate_cli(&cli)?;
    let _file_guard = init_tracing(&cli)?;
    let bind: SocketAddr = cli
        .bind
        .parse()
        .with_context(|| format!("invalid --bind address {:?}", cli.bind))?;
    info!(
        phase = "server.starting",
        bind = %bind,
        prefix_cache_enabled = !cli.disable_prefix_cache,
    );

    let kv_cache_mode = cli.kv_cache_mode();
    #[cfg(feature = "specprefill")]
    let specprefill_policy = cli.specprefill_policy();
    let model = resolve_model_path(cli.model.as_deref())?;
    #[cfg(feature = "dflash2")]
    let dflash_draft_model = resolve_dflash2_draft_path(cli.dflash_draft_model.as_deref())?;
    #[cfg(not(feature = "dflash2"))]
    let dflash_draft_model = cli.dflash_draft_model.unwrap_or_default();
    let prefix_cache_directory = resolve_prefix_cache_directory(cli.prefix_cache_directory)?;
    let engine = Engine::start_qwen(
        model,
        cli.model_id,
        CacheConfig {
            memory_bytes: cli.prefix_cache_memory_bytes,
            directory: Some(prefix_cache_directory),
            filesystem_bytes: cli.prefix_cache_filesystem_bytes,
        },
        !cli.disable_prefix_cache,
        cli.mtp_k,
        crate::DecoderConfig {
            mode: cli.decoder.generation_mode(),
            crossover_tokens: cli.decoder_crossover_tokens,
            dflash2_draft_model: dflash_draft_model,
        },
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
    use super::{
        Decoder, OutputFormat, ServerArgs, parse_si_bytes, persistent_log_format,
        resolve_persistent_log_filter, resolve_prefix_cache_directory, validate_cli,
    };
    use clap::{CommandFactory as _, Parser as _};
    use clap_derive::Parser;
    use qw_runtime::{DEFAULT_DECODER_CROSSOVER_TOKENS, KVCacheMode};
    use std::fs::OpenOptions;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt as _};
    #[derive(Debug, Parser)]
    #[command(name = "qw-server", about = "OpenAI-compatible dense Qwen3.5 server")]
    struct TestCli {
        #[command(flatten)]
        args: ServerArgs,
    }

    #[test]
    fn persistent_log_format_emits_json_lines() {
        let path = std::env::temp_dir().join(format!(
            "qw-persistent-log-json-{}-{}.log",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let writer_path = path.clone();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .fmt_fields(tracing_subscriber::fmt::format::JsonFields::new())
                .event_format(persistent_log_format())
                .with_writer(move || {
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&writer_path)
                        .unwrap()
                }),
        );

        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "server.request",
                endpoint = "Chat",
                model = tracing::field::Empty,
                stream = tracing::field::Empty,
            );
            let _entered = span.enter();
            span.record("model", "Qwen3.8-27B");
            span.record("stream", true);
            tracing::info!(request_id = 42, "persisted event");
        });

        let output = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(!output.contains("field_error"), "{output}");
        let parsed: serde_json::Value = serde_json::from_str(output.trim_end()).unwrap();
        assert_eq!(parsed["spans"][0]["name"], "server.request");
        assert_eq!(parsed["spans"][0]["model"], "Qwen3.8-27B");
        assert_eq!(parsed["spans"][0]["stream"], true);
        assert_eq!(parsed["fields"]["message"], "persisted event");
        assert_eq!(parsed["fields"]["request_id"], 42);
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
    fn cli_prefix_cache_opt_out_defaults_to_enabled() {
        let default = TestCli::try_parse_from(["qw-server"]).expect("CLI");
        assert!(!default.args.disable_prefix_cache);

        let disabled =
            TestCli::try_parse_from(["qw-server", "--disable-prefix-cache"]).expect("CLI");
        assert!(disabled.args.disable_prefix_cache);

        let help = TestCli::command().render_long_help().to_string();
        assert!(help.contains("--disable-prefix-cache"), "{help}");
    }

    fn capture_persistent_log_filter_events(filter: EnvFilter) -> String {
        let path = std::env::temp_dir().join(format!(
            "qw-persistent-log-filter-{}-{}.log",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let writer_path = path.clone();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(move || {
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&writer_path)
                        .unwrap()
                })
                .with_filter(filter),
        );

        tracing::subscriber::with_default(subscriber, || {
            tracing::trace!(target: "qw_server::server", "qw_server trace");
            tracing::debug!(target: "qw_server::server", "qw_server debug");
            tracing::info!(target: "qw_server::server", "qw_server info");
            tracing::trace!(target: "qw_runtime::runtime", "qw_runtime trace");
            tracing::trace!(target: "qw_prefix_cache::cache", "qw_prefix_cache trace");
            tracing::trace!(target: "mlxcel_core::core", "mlxcel_core trace");
            tracing::trace!(target: "qw_cli::cli", "qw_cli trace");
            tracing::debug!(target: "third_party::worker", "third_party debug");
            tracing::trace!(target: "third_party::worker", "third_party trace");
            tracing::info!(target: "tokenizers::model", "tokenizer info");
            tracing::debug!(target: "tokenizers::model", "tokenizer debug");
            tracing::trace!(target: "tokenizers::model", "tokenizer trace");
        });

        let output = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        output
    }

    #[test]
    fn persistent_log_filter_defaults_and_preserves_precedence() {
        let default_output = capture_persistent_log_filter_events(
            resolve_persistent_log_filter(None, None).unwrap(),
        );
        for message in [
            "qw_server trace",
            "qw_runtime trace",
            "qw_prefix_cache trace",
            "mlxcel_core trace",
            "qw_cli trace",
            "third_party debug",
            "tokenizer info",
        ] {
            assert!(default_output.contains(message), "{default_output}");
        }
        for message in ["third_party trace", "tokenizer debug", "tokenizer trace"] {
            assert!(!default_output.contains(message), "{default_output}");
        }

        let cli = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--persistent-log-filter",
            "qw_server=info",
        ])
        .expect("CLI");
        let cli_output = capture_persistent_log_filter_events(
            resolve_persistent_log_filter(
                cli.args.persistent_log_filter.as_deref(),
                Some("qw_server=trace"),
            )
            .unwrap(),
        );
        assert!(cli_output.contains("qw_server info"), "{cli_output}");
        assert!(!cli_output.contains("qw_server debug"), "{cli_output}");
        assert!(!cli_output.contains("qw_server trace"), "{cli_output}");

        let env_output = capture_persistent_log_filter_events(
            resolve_persistent_log_filter(None, Some("qw_server=debug")).unwrap(),
        );
        assert!(env_output.contains("qw_server debug"), "{env_output}");
        assert!(!env_output.contains("qw_server trace"), "{env_output}");

        let help = TestCli::command().render_long_help().to_string();
        assert!(
            help.contains("--persistent-log-filter <PERSISTENT_LOG_FILTER>"),
            "{help}"
        );
        assert!(help.contains("overrides QW_LOG"), "{help}");
        assert!(help.contains("defaults to"), "{help}");
        assert!(help.contains("qw_server=trace"), "{help}");
        assert!(help.contains("tokenizers=info"), "{help}");
    }

    #[test]
    fn persistent_log_filter_reports_invalid_directives() {
        let error = resolve_persistent_log_filter(Some("target=not-a-level"), None)
            .expect_err("invalid filter");
        let message = error.to_string();
        assert!(
            message.contains("invalid persistent log filter directives"),
            "{message}"
        );
        assert!(message.contains("--persistent-log-filter"), "{message}");
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

    #[test]
    fn cli_parses_decimal_si_cache_byte_capacities() {
        let parsed = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--prefix-cache-memory-bytes",
            "1.25KB",
            "--prefix-cache-filesystem-bytes",
            "2.5g",
        ])
        .expect("SI byte capacities");
        assert_eq!(parsed.args.prefix_cache_memory_bytes, 1_250);
        assert_eq!(parsed.args.prefix_cache_filesystem_bytes, 2_500_000_000);

        assert_eq!(parse_si_bytes("18446744073709551615").unwrap(), u64::MAX);
        assert_eq!(parse_si_bytes("18E").unwrap(), 18_000_000_000_000_000_000);
        assert_eq!(parse_si_bytes("1b").unwrap(), 1);
    }

    #[test]
    fn cli_rejects_invalid_or_overflowing_cache_byte_capacities() {
        for value in [
            "1.5",
            ".5K",
            "1.001K",
            "0.0001K",
            "1KiB",
            "1e3",
            "19E",
            "18446744073709551616",
        ] {
            let error = TestCli::try_parse_from([
                "qw-server",
                "--model",
                "/tmp/checkpoint",
                "--prefix-cache-memory-bytes",
                value,
            ])
            .expect_err("invalid byte capacity");
            assert!(
                error.to_string().contains("invalid value"),
                "{value}: {error}"
            );
        }

        let negative = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--prefix-cache-memory-bytes=-1",
        ])
        .expect_err("negative byte capacity");
        assert!(
            negative.to_string().contains("invalid byte size '-1'"),
            "{negative}"
        );
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

        let invalid_threshold =
            TestCli::try_parse_from(["qw-server", "--specprefill-min-turn-tokens", "0"])
                .expect("threshold reaches startup validation");
        assert!(validate_cli(&invalid_threshold.args).is_err());

        for rate in ["0", "1.1", "NaN"] {
            let invalid_rate =
                TestCli::try_parse_from(["qw-server", "--specprefill-keep-rate", rate])
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
    fn cli_exposes_stable_decoder_policy_defaults_and_overrides() {
        let default =
            TestCli::try_parse_from(["qw-server", "--model", "/tmp/checkpoint"]).expect("CLI");
        assert_eq!(default.args.decoder, Decoder::Automatic);
        assert_eq!(
            default.args.decoder_crossover_tokens,
            DEFAULT_DECODER_CROSSOVER_TOKENS
        );
        assert_eq!(default.args.dflash_draft_model, None);

        let explicit = TestCli::try_parse_from([
            "qw-server",
            "--model",
            "/tmp/checkpoint",
            "--decoder",
            "dflash",
            "--decoder-crossover-tokens",
            "8192",
            "--dflash-draft-model",
            "/tmp/dflash",
        ])
        .expect("explicit decoder CLI");
        assert_eq!(explicit.args.decoder, Decoder::Dflash2);
        assert_eq!(explicit.args.decoder_crossover_tokens, 8192);
        assert_eq!(
            explicit.args.dflash_draft_model.as_deref(),
            Some(std::path::Path::new("/tmp/dflash"))
        );

        let invalid = TestCli::try_parse_from(["qw-server", "--decoder-crossover-tokens", "0"])
            .expect("crossover reaches startup validation");
        assert_eq!(
            validate_cli(&invalid.args)
                .expect_err("zero crossover")
                .to_string(),
            "--decoder-crossover-tokens must be greater than zero"
        );

        let help = TestCli::command().render_long_help().to_string();
        for option in [
            "--decoder",
            "--decoder-crossover-tokens",
            "--dflash-draft-model",
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
