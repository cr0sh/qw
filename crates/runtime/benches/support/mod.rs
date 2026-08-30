use std::fs::{self, OpenOptions};
use std::hint::black_box;
use std::io::Write as _;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use qw_prefix_cache::{
    AdaptivePrefixCache, CacheConfig, CacheNamespaces, Manifest as PrefixCacheManifest,
    SnapshotRoute, namespace_hash,
};
use qw_runtime::provider::Qwen4GenerationMode;
use qw_runtime::{
    ChatMessage, ChatMessageContent, GenerationRequest, KVCacheMode, PromptSnapshot, Qwen4Provider,
};
use serde::{Deserialize, Serialize};

pub const DECODE_MAX_TOKENS: usize = 32;
pub const MTP_BLOCK_SIZE: usize = 3;
pub const LONG_CONTEXT_64K_MIN_TOKENS: usize = 64_000;
pub fn long_context_tokens() -> usize {
    std::env::var("QW_BENCH_LONG_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(LONG_CONTEXT_64K_MIN_TOKENS)
}

pub struct BenchmarkMemoryReport {
    enabled: bool,
}

impl BenchmarkMemoryReport {
    /// Resolve the opt-in once and reset the MLX peak immediately after model
    /// load, before fixture construction contributes to the measured peak.
    pub fn after_model_load() -> Self {
        let enabled = std::env::var("QW_BENCH_MEMORY").as_deref() == Ok("1");
        if enabled {
            mlxcel_core::memory::reset_peak_memory();
        }
        Self { enabled }
    }

    /// Emit the first long-context record after the prefix snapshot is
    /// extracted, before any cached restore.
    pub fn emit_fresh_snapshot(
        &self,
        context_tokens: usize,
        prefix_tokens: usize,
        snapshot: &PromptSnapshot,
    ) {
        self.emit(
            "fresh_snapshot",
            context_tokens,
            prefix_tokens,
            snapshot,
            None,
            None,
        );
    }

    /// Emit the second long-context record after fixture warm restores.
    ///
    /// Report emission is benchmark setup only, never a timed Criterion
    /// closure.
    pub fn emit_post_restore_fixture(
        &self,
        context_tokens: usize,
        prefix_tokens: usize,
        snapshot: &PromptSnapshot,
        baseline_token_ids: &[i32],
        mtp_token_ids: &[i32],
    ) {
        self.emit(
            "post_restore_fixture",
            context_tokens,
            prefix_tokens,
            snapshot,
            Some(baseline_token_ids),
            Some(mtp_token_ids),
        );
    }

    fn emit(
        &self,
        phase: &'static str,
        context_tokens: usize,
        prefix_tokens: usize,
        snapshot: &PromptSnapshot,
        baseline_token_ids: Option<&[i32]>,
        mtp_token_ids: Option<&[i32]>,
    ) {
        if !self.enabled {
            return;
        }
        let mlx = mlxcel_core::memory::snapshot();
        let process = mlxcel_core::memory::process_snapshot().map(|process| {
            serde_json::json!({
                "resident_bytes": process.resident_bytes,
                "wired_bytes": process.wired_bytes,
                "physical_footprint_bytes": process.physical_footprint_bytes,
                "lifetime_peak_physical_footprint_bytes":
                    process.lifetime_peak_physical_footprint_bytes,
            })
        });
        let recommended_working_set = mlxcel_core::get_wired_limit() as u64;
        let configured_wired_limit = (((u128::from(
            mlxcel_core::hardware::system_memory_bytes(),
        ) * 85)
            / 100) as u64)
            .min(recommended_working_set);
        let record = serde_json::json!({
            "schema": "qw_bench_memory",
            "phase": phase,
            "pid": std::process::id(),
            "context_tokens": context_tokens,
            "prefix_tokens": prefix_tokens,
            "prompt_snapshot_logical_bytes": PromptSnapshot::nbytes(snapshot),
            "baseline_token_ids": baseline_token_ids,
            "mtp_token_ids": mtp_token_ids,
            "mlx": {
                "active_bytes": mlx.active_bytes,
                "peak_bytes": mlx.peak_bytes,
                "cache_bytes": mlx.cache_bytes,
                "limit_bytes": mlx.limit_bytes,
                "recommended_max_working_set_bytes": recommended_working_set,
                "configured_wired_limit_bytes": configured_wired_limit,
            },
            "process": process,
        });
        eprintln!("QW_BENCH_MEMORY={record}");
    }
}

pub const FRESH_PREFILL_REPETITIONS: usize = 24;

const PROMPT: &str = concat!(
    "You are the on-call support operations analyst. ",
    "Summarize the incident, list affected systems, and state the next action. ",
    "Treat timestamps as UTC and do not invent facts.\n",
    "Incident: checkout requests intermittently failed after a catalog import; ",
    "one retry succeeded and one payment capture still fails.\n"
);

const LONG_CONTEXT_SUFFIX_TOKENS: usize = 1_536;
const LONG_PROMPT_TAIL_TOKENS: usize = 16;
const LONG_PROMPT_MAX_RECORDS: usize = 20_000;
const LONG_PROMPT_TOKENIZE_INTERVAL: usize = 128;
const PREFIX_CACHE_MEMORY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const PREFIX_CACHE_FILESYSTEM_BYTES: u64 = 20 * 1024 * 1024 * 1024;
static BENCHMARK_STARTED: OnceLock<Instant> = OnceLock::new();
static MODEL_IDENTITY: LazyLock<Result<BenchmarkModelIdentity, String>> = LazyLock::new(|| {
    let target = qw_runtime::resolve_model_path(None)
        .map_err(|error| format!("resolve target model: {error:#}"))?;
    let draft = qw_runtime::resolve_mtp_model_path()
        .map_err(|error| format!("resolve draft model: {error:#}"))?;
    Ok(BenchmarkModelIdentity {
        target: model_directory_identity(&target)?,
        draft: model_directory_identity(&draft)?,
    })
});

struct BenchmarkModelIdentity {
    target: String,
    draft: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodeFixtureManifest {
    namespace: String,
    baseline_route: SnapshotRoute,
    mtp_route: SnapshotRoute,
    baseline_token_ids: Vec<i32>,
    mtp_token_ids: Vec<i32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefillFixtureManifest {
    namespace: String,
    route: SnapshotRoute,
    prompt_tokens: usize,
    first_token_id: i32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LongConversationFixtureManifest {
    namespace: String,
    route: SnapshotRoute,
    prompt_ids: Vec<i32>,
    prefix_tokens: usize,
    baseline_token_ids: Vec<i32>,
    mtp_token_ids: Vec<i32>,
}

pub struct DecodeFixture {
    pub request: GenerationRequest,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
}
pub struct PrefillFixture {
    pub request: GenerationRequest,
    pub prompt_tokens: usize,
    pub first_token_id: i32,
}

pub struct LongConversationFixture {
    pub prompt_ids: Vec<i32>,
    pub prefix_tokens: usize,
    pub prefix_cache: AdaptivePrefixCache,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
}

pub fn request(max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        max_tokens,
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        seed: Some(0),
    }
}

fn emit_setup_phase(phase: &str, elapsed: Duration, source: &str) {
    let total = BENCHMARK_STARTED.get_or_init(Instant::now).elapsed();
    eprintln!(
        "QW_BENCH_SETUP phase={phase} elapsed_ms={} total_ms={} source={source}",
        elapsed.as_millis(),
        total.as_millis(),
    );
}

fn append_identity_field(identity: &mut Vec<u8>, field: &[u8]) {
    identity.extend_from_slice(&(field.len() as u64).to_le_bytes());
    identity.extend_from_slice(field);
}

fn model_directory_identity(model_dir: &Path) -> Result<String, String> {
    let canonical_dir = fs::canonicalize(model_dir)
        .map_err(|error| format!("canonicalize {}: {error}", model_dir.display()))?;
    let mut identity = Vec::new();
    append_identity_field(
        &mut identity,
        canonical_dir.to_string_lossy().as_bytes(),
    );
    let mut entries = fs::read_dir(model_dir)
        .map_err(|error| format!("read {}: {error}", model_dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read {}: {error}", model_dir.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::metadata(&path)
            .map_err(|error| format!("stat {}: {error}", path.display()))?;
        if !metadata.is_file() {
            continue;
        }
        append_identity_field(&mut identity, entry.file_name().to_string_lossy().as_bytes());
        let canonical_path = fs::canonicalize(&path)
            .map_err(|error| format!("canonicalize {}: {error}", path.display()))?;
        append_identity_field(
            &mut identity,
            canonical_path.to_string_lossy().as_bytes(),
        );
        append_identity_field(&mut identity, &metadata.len().to_le_bytes());
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .unwrap_or_default();
        append_identity_field(&mut identity, &modified.as_secs().to_le_bytes());
        append_identity_field(&mut identity, &modified.subsec_nanos().to_le_bytes());
        #[cfg(unix)]
        {
            append_identity_field(&mut identity, &metadata.dev().to_le_bytes());
            append_identity_field(&mut identity, &metadata.ino().to_le_bytes());
            append_identity_field(&mut identity, &metadata.ctime().to_le_bytes());
            append_identity_field(&mut identity, &metadata.ctime_nsec().to_le_bytes());
        }
        if path.extension().and_then(|extension| extension.to_str()).is_some_and(
            |extension| matches!(extension, "json" | "jinja" | "txt"),
        ) {
            let bytes =
                fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
            append_identity_field(&mut identity, &bytes);
        }
    }
    Ok(namespace_hash(&[&identity]))
}

fn benchmark_model_identity() -> &'static BenchmarkModelIdentity {
    MODEL_IDENTITY
        .as_ref()
        .unwrap_or_else(|error| panic!("fingerprint benchmark models: {error}"))
}

fn fixture_namespace(kind: &[u8], request_parts: &[&[u8]]) -> String {
    let models = benchmark_model_identity();
    let mut parts = vec![
        kind,
        b"kv_cache_mode=fp8",
        models.target.as_bytes(),
        models.draft.as_bytes(),
    ];
    parts.extend_from_slice(request_parts);
    namespace_hash(&parts)
}

fn cache_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("HOME").unwrap_or_else(|| panic!("HOME is required for benchmark cache")),
    )
    .join(".cache/qw/benchmarks/single_user_throughput")
}

fn fixture_manifest_path(kind: &str, namespace: &str) -> PathBuf {
    cache_root()
        .join("fixtures")
        .join(format!("{kind}-{namespace}.json"))
}

fn read_fixture_manifest<T>(path: &Path) -> Option<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn publish_fixture_manifest<T>(path: &Path, manifest: &T) -> Result<(), String>
where
    T: Serialize,
{
    let parent = path
        .parent()
        .ok_or_else(|| format!("fixture manifest has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("create {}: {error}", parent.display()))?;
    let bytes = serde_json::to_vec(manifest).map_err(|error| error.to_string())?;
    let temp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .map_err(|error| format!("create {}: {error}", temp.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("write {}: {error}", temp.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync {}: {error}", temp.display()))?;
    drop(file);
    fs::rename(&temp, path).map_err(|error| {
        let _ = fs::remove_file(&temp);
        format!("publish {}: {error}", path.display())
    })
}

fn token_ids_valid(token_ids: &[i32], expected_len: usize, vocab_size: usize) -> bool {
    token_ids.len() == expected_len
        && token_ids
            .iter()
            .all(|token| *token >= 0 && (*token as usize) < vocab_size)
}

fn long_manifest_valid(
    manifest: &LongConversationFixtureManifest,
    namespace: &str,
    prefix_tokens: usize,
    vocab_size: usize,
) -> bool {
    let Some(prompt_tokens) = prefix_tokens.checked_add(LONG_CONTEXT_SUFFIX_TOKENS) else {
        return false;
    };
    manifest.namespace == namespace
        && manifest.route == SnapshotRoute::Mtp
        && manifest.prefix_tokens == prefix_tokens
        && token_ids_valid(&manifest.prompt_ids, prompt_tokens, vocab_size)
        && token_ids_valid(
            &manifest.baseline_token_ids,
            DECODE_MAX_TOKENS,
            vocab_size,
        )
        && token_ids_valid(&manifest.mtp_token_ids, DECODE_MAX_TOKENS, vocab_size)
}

fn prefix_cache_namespaces(namespace: &str) -> (CacheNamespaces, String) {
    let baseline = namespace_hash(&[
        namespace.as_bytes(),
        SnapshotRoute::Baseline.as_str().as_bytes(),
    ]);
    let mtp = namespace_hash(&[
        namespace.as_bytes(),
        SnapshotRoute::Mtp.as_str().as_bytes(),
    ]);
    (
        CacheNamespaces {
            baseline,
            mtp: mtp.clone(),
        },
        mtp,
    )
}

fn open_prefix_cache(namespaces: CacheNamespaces, prefix_root: &Path) -> AdaptivePrefixCache {
    AdaptivePrefixCache::new(
        namespaces,
        CacheConfig {
            memory_bytes: PREFIX_CACHE_MEMORY_BYTES,
            directory: Some(prefix_root.to_owned()),
            filesystem_bytes: PREFIX_CACHE_FILESYSTEM_BYTES,
        },
    )
    .unwrap_or_else(|error| panic!("open persistent benchmark prefix cache: {error}"))
}

fn persistent_snapshot_complete(
    prefix_root: &Path,
    namespace: &str,
    prompt_ids: &[i32],
    prefix_tokens: usize,
) -> bool {
    let Some(prefix_ids) = prompt_ids.get(..prefix_tokens) else {
        return false;
    };
    let Ok(entries) = fs::read_dir(prefix_root.join("entries").join(namespace)) else {
        return false;
    };
    for entry in entries.flatten() {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(manifest) = fs::read(entry.path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PrefixCacheManifest>(&bytes).ok())
        else {
            continue;
        };
        if manifest.namespace != namespace
            || manifest.route != SnapshotRoute::Mtp
            || manifest.token_len != prefix_tokens
            || manifest.token_ids != prefix_ids
        {
            continue;
        }
        if manifest.blob_sha256.is_empty() {
            continue;
        }
        let stored_bytes = manifest
            .blob_sha256
            .iter()
            .try_fold(0u64, |total, digest| {
                let metadata = fs::metadata(prefix_root.join("blobs").join(digest)).ok()?;
                metadata
                    .is_file()
                    .then(|| total.checked_add(metadata.len()))
                    .flatten()
            });
        if stored_bytes == Some(manifest.total_bytes) {
            return true;
        }
    }
    false
}

fn exact_mtp_snapshot<'a>(
    cache: &'a mut AdaptivePrefixCache,
    prompt_ids: &[i32],
    prefix_tokens: usize,
) -> Option<&'a PromptSnapshot> {
    let hit = cache.lookup(prompt_ids, SnapshotRoute::Mtp)?;
    (hit.token_count == prefix_tokens
        && hit.snapshot.token_len() == prefix_tokens
        && SnapshotRoute::Mtp.matches(hit.snapshot))
    .then_some(hit.snapshot)
}


pub fn load_provider() -> Qwen4Provider {
    let _ = BENCHMARK_STARTED.set(Instant::now());
    let model_dir = qw_runtime::resolve_model_path(None)
        .unwrap_or_else(|error| panic!("failed to resolve benchmark model path: {error:#}"));
    Qwen4Provider::load(&model_dir, KVCacheMode::Fp8)
        .unwrap_or_else(|error| panic!("failed to load {}: {error:#}", model_dir.display()))
}

pub fn prepare_decode_fixture(provider: &mut Qwen4Provider) -> DecodeFixture {
    let setup_started = Instant::now();
    let request = request(DECODE_MAX_TOKENS);
    let max_tokens = (DECODE_MAX_TOKENS as u64).to_le_bytes();
    let mtp_block_size = (MTP_BLOCK_SIZE as u64).to_le_bytes();
    let namespace = fixture_namespace(
        b"fresh_decode",
        &[
            PROMPT.as_bytes(),
            b"temperature=0;top_k=1;top_p=1;seed=0",
            b"routes=baseline,mtp",
            &max_tokens,
            &mtp_block_size,
        ],
    );
    let manifest_path = fixture_manifest_path("fresh-decode", &namespace);
    let load_started = Instant::now();
    if let Some(manifest) = read_fixture_manifest::<DecodeFixtureManifest>(&manifest_path)
        && manifest.namespace == namespace
        && manifest.baseline_route == SnapshotRoute::Baseline
        && manifest.mtp_route == SnapshotRoute::Mtp
        && token_ids_valid(
            &manifest.baseline_token_ids,
            DECODE_MAX_TOKENS,
            provider.logits_vocab_size(),
        )
        && token_ids_valid(
            &manifest.mtp_token_ids,
            DECODE_MAX_TOKENS,
            provider.logits_vocab_size(),
        )
    {
        emit_setup_phase("fresh_decode_load", load_started.elapsed(), "persistent");
        emit_setup_phase("fresh_decode_ready", setup_started.elapsed(), "persistent");
        return DecodeFixture {
            request,
            baseline_token_ids: manifest.baseline_token_ids,
            mtp_token_ids: manifest.mtp_token_ids,
        };
    }
    emit_setup_phase("fresh_decode_load", load_started.elapsed(), "cold");

    let build_started = Instant::now();
    let (baseline, _) = provider
        .generate_streaming_in_mode(&request, Qwen4GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm baseline decode");
    let (mtp, stats) = provider
        .generate_streaming_in_mode(&request, Qwen4GenerationMode::Mtp, |delta| {
            black_box(delta);
            true
        })
        .expect("warm MTP decode");
    // Baseline T=1 and batched MTP T=N are deterministic independently but
    // need not have identical greedy token alignment. Each benchmark iteration
    // below is checked against its own warmed token sequence.
    assert!(
        stats.is_some_and(|stats| stats.proposed_draft_tokens > 0),
        "MTP warmup must propose draft tokens"
    );
    assert!(token_ids_valid(
        &baseline.token_ids,
        DECODE_MAX_TOKENS,
        provider.logits_vocab_size(),
    ));
    assert!(token_ids_valid(
        &mtp.token_ids,
        DECODE_MAX_TOKENS,
        provider.logits_vocab_size(),
    ));
    emit_setup_phase("fresh_decode_build", build_started.elapsed(), "cold");

    let persist_started = Instant::now();
    publish_fixture_manifest(
        &manifest_path,
        &DecodeFixtureManifest {
            namespace,
            baseline_route: SnapshotRoute::Baseline,
            mtp_route: SnapshotRoute::Mtp,
            baseline_token_ids: baseline.token_ids.clone(),
            mtp_token_ids: mtp.token_ids.clone(),
        },
    )
    .unwrap_or_else(|error| panic!("persist fresh decode fixture: {error}"));
    emit_setup_phase("fresh_decode_persist", persist_started.elapsed(), "cold");
    emit_setup_phase("fresh_decode_ready", setup_started.elapsed(), "cold");
    DecodeFixture {
        request,
        baseline_token_ids: baseline.token_ids,
        mtp_token_ids: mtp.token_ids,
    }
}
pub fn prepare_prefill_fixture(provider: &mut Qwen4Provider) -> PrefillFixture {
    let setup_started = Instant::now();
    let request = GenerationRequest {
        prompt: PROMPT.repeat(FRESH_PREFILL_REPETITIONS),
        max_tokens: 1,
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        seed: Some(0),
    };
    let repetitions = (FRESH_PREFILL_REPETITIONS as u64).to_le_bytes();
    let namespace = fixture_namespace(
        b"fresh_prefill",
        &[
            PROMPT.as_bytes(),
            b"max_tokens=1;temperature=0;top_k=1;top_p=1;seed=0",
            b"route=baseline",
            &repetitions,
        ],
    );
    let manifest_path = fixture_manifest_path("fresh-prefill", &namespace);
    let load_started = Instant::now();
    if let Some(manifest) = read_fixture_manifest::<PrefillFixtureManifest>(&manifest_path)
        && manifest.namespace == namespace
        && manifest.route == SnapshotRoute::Baseline
        && manifest.prompt_tokens > 0
        && token_ids_valid(
            &[manifest.first_token_id],
            1,
            provider.logits_vocab_size(),
        )
    {
        emit_setup_phase("fresh_prefill_load", load_started.elapsed(), "persistent");
        emit_setup_phase("fresh_prefill_ready", setup_started.elapsed(), "persistent");
        return PrefillFixture {
            request,
            prompt_tokens: manifest.prompt_tokens,
            first_token_id: manifest.first_token_id,
        };
    }
    emit_setup_phase("fresh_prefill_load", load_started.elapsed(), "cold");

    let build_started = Instant::now();
    let (generation, stats) = provider
        .benchmark_streaming_in_mode(&request, Qwen4GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm fresh prefill");
    assert!(stats.is_none());
    assert!(generation.prompt_tokens > 0);
    assert!(token_ids_valid(
        &generation.token_ids,
        1,
        provider.logits_vocab_size(),
    ));
    emit_setup_phase("fresh_prefill_build", build_started.elapsed(), "cold");

    let persist_started = Instant::now();
    publish_fixture_manifest(
        &manifest_path,
        &PrefillFixtureManifest {
            namespace,
            route: SnapshotRoute::Baseline,
            prompt_tokens: generation.prompt_tokens,
            first_token_id: generation.token_ids[0],
        },
    )
    .unwrap_or_else(|error| panic!("persist fresh prefill fixture: {error}"));
    emit_setup_phase("fresh_prefill_persist", persist_started.elapsed(), "cold");
    emit_setup_phase("fresh_prefill_ready", setup_started.elapsed(), "cold");
    PrefillFixture {
        request,
        prompt_tokens: generation.prompt_tokens,
        first_token_id: generation.token_ids[0],
    }
}

fn long_prompt_ids(provider: &Qwen4Provider, min_tokens: usize) -> Vec<i32> {
    let mut prompt = String::new();
    for index in 0..LONG_PROMPT_MAX_RECORDS {
        prompt.push_str(&format!("Record {index}: {PROMPT}\n"));
        if index % LONG_PROMPT_TOKENIZE_INTERVAL != LONG_PROMPT_TOKENIZE_INTERVAL - 1 {
            continue;
        }
        let ids = provider
            .tokenize_messages(
                &[ChatMessage {
                    role: "user".to_owned(),
                    name: None,
                    content: Some(ChatMessageContent::Text(prompt.clone())),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
                &[],
                None,
                true,
            )
            .expect("tokenize long benchmark prompt");
        if ids.len() >= min_tokens {
            let suffix_len = LONG_PROMPT_TAIL_TOKENS.min(min_tokens);
            let mut truncated = ids[..min_tokens - suffix_len].to_vec();
            truncated.extend_from_slice(&ids[ids.len() - suffix_len..]);
            return truncated;
        }
    }
    panic!("long benchmark prompt did not reach {min_tokens} tokens")
}

pub fn prepare_long_conversation_fixture(
    provider: &mut Qwen4Provider,
    memory_report: &BenchmarkMemoryReport,
    min_prefix_tokens: usize,
) -> LongConversationFixture {
    let setup_started = Instant::now();
    let requested_tokens = (min_prefix_tokens as u64).to_le_bytes();
    let suffix_tokens = (LONG_CONTEXT_SUFFIX_TOKENS as u64).to_le_bytes();
    let tail_tokens = (LONG_PROMPT_TAIL_TOKENS as u64).to_le_bytes();
    let max_records = (LONG_PROMPT_MAX_RECORDS as u64).to_le_bytes();
    let tokenize_interval = (LONG_PROMPT_TOKENIZE_INTERVAL as u64).to_le_bytes();
    let decode_tokens = (DECODE_MAX_TOKENS as u64).to_le_bytes();
    let mtp_block_size = (MTP_BLOCK_SIZE as u64).to_le_bytes();
    let namespace = fixture_namespace(
        b"long_conversation",
        &[
            PROMPT.as_bytes(),
            b"prompt_format=Record {index}: {PROMPT}\\n;truncate=head_plus_template_tail",
            b"role=user;tools=[];reasoning_effort=null;enable_thinking=true",
            b"temperature=0;top_p=1;seed=0",
            b"snapshot_route=mtp;benchmark_routes=baseline,mtp",
            &requested_tokens,
            &suffix_tokens,
            &tail_tokens,
            &max_records,
            &tokenize_interval,
            &decode_tokens,
            &mtp_block_size,
        ],
    );
    let root = cache_root();
    let prefix_root = root.join("prefix");
    let manifest_path = fixture_manifest_path("long-conversation", &namespace);
    let (cache_namespaces, mtp_namespace) = prefix_cache_namespaces(&namespace);
    let load_started = Instant::now();
    let mut prefix_cache = open_prefix_cache(cache_namespaces.clone(), &prefix_root);
    if let Some(manifest) =
        read_fixture_manifest::<LongConversationFixtureManifest>(&manifest_path)
        && long_manifest_valid(
            &manifest,
            &namespace,
            min_prefix_tokens,
            provider.logits_vocab_size(),
        )
        && exact_mtp_snapshot(
            &mut prefix_cache,
            &manifest.prompt_ids,
            manifest.prefix_tokens,
        )
        .is_some()
    {
        emit_setup_phase("long_conversation_load", load_started.elapsed(), "persistent");
        {
            let snapshot = exact_mtp_snapshot(
                &mut prefix_cache,
                &manifest.prompt_ids,
                manifest.prefix_tokens,
            )
            .expect("validated persistent MTP snapshot");
            memory_report.emit_post_restore_fixture(
                manifest.prompt_ids.len(),
                manifest.prefix_tokens,
                snapshot,
                &manifest.baseline_token_ids,
                &manifest.mtp_token_ids,
            );
        }
        emit_setup_phase(
            "long_conversation_ready",
            setup_started.elapsed(),
            "persistent",
        );
        return LongConversationFixture {
            prompt_ids: manifest.prompt_ids,
            prefix_tokens: manifest.prefix_tokens,
            prefix_cache,
            baseline_token_ids: manifest.baseline_token_ids,
            mtp_token_ids: manifest.mtp_token_ids,
        };
    }
    emit_setup_phase("long_conversation_load", load_started.elapsed(), "cold");

    let build_started = Instant::now();
    let prompt_tokens = min_prefix_tokens
        .checked_add(LONG_CONTEXT_SUFFIX_TOKENS)
        .expect("long benchmark prompt length");
    let prompt_ids = long_prompt_ids(provider, prompt_tokens);
    let prefix_tokens = min_prefix_tokens;
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let mtp = provider
        .generate_mtp_streaming(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            MTP_BLOCK_SIZE,
            None,
            &[prefix_tokens],
            None,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("prepare long-context MTP snapshot");
    let snapshot = mtp
        .prompt_snapshots
        .into_iter()
        .find(|snapshot| snapshot.token_len() == prefix_tokens)
        .expect("MTP checkpoint at requested 64k prefix");
    memory_report.emit_fresh_snapshot(prompt_ids.len(), prefix_tokens, &snapshot);
    let target_snapshot = match &snapshot {
        PromptSnapshot::Baseline(snapshot) => snapshot,
        PromptSnapshot::Mtp(snapshot) => snapshot.target_snapshot(),
    };
    let auxiliary_name = target_snapshot
        .tensor_names()
        .find(|name| name.ends_with(".auxiliary_keys"))
        .expect("64k target snapshot must include bounded QSA keys");
    let auxiliary_keys = mlxcel_core::copy(
        target_snapshot
            .tensor(auxiliary_name)
            .expect("materialize 64k QSA keys"),
    );
    let layer_prefix = auxiliary_name
        .strip_suffix("auxiliary_keys")
        .expect("QSA snapshot tensor suffix");
    let scalar = |suffix: &str| {
        let value = target_snapshot
            .tensor(&format!("{layer_prefix}{suffix}"))
            .unwrap_or_else(|| panic!("64k QSA snapshot must include {suffix}"));
        mlxcel_core::item_i32(&mlxcel_core::reshape(value, &[]))
    };
    let tail_start = scalar("auxiliary_tail_start");
    let tail_end = scalar("auxiliary_tail_end");
    let horizon = scalar("auxiliary_rollback_horizon");
    let ratio = scalar("auxiliary_block_size");
    let raw_rows = mlxcel_core::array_shape(&auxiliary_keys)[1];
    assert_eq!(tail_end as usize, prefix_tokens);
    assert_eq!(horizon as usize, MTP_BLOCK_SIZE);
    assert_eq!(
        tail_start,
        tail_end.saturating_sub(horizon).max(0) / ratio * ratio
    );
    assert_eq!(raw_rows, tail_end - tail_start);
    assert!(raw_rows <= horizon + ratio - 1);
    let block_keys = target_snapshot
        .paged_tensor(&format!("{layer_prefix}auxiliary_block_keys"))
        .and_then(|tensor| tensor.materialize())
        .expect("64k QSA snapshot must retain logical block summaries");
    assert_eq!(mlxcel_core::array_shape(&block_keys)[2], tail_end / ratio);
    drop(auxiliary_keys);
    drop(block_keys);
    emit_setup_phase("long_conversation_build", build_started.elapsed(), "cold");

    let persist_started = Instant::now();
    prefix_cache.insert(&prompt_ids, vec![snapshot], SnapshotRoute::Mtp);
    prefix_cache.flush_persistence();
    assert!(
        persistent_snapshot_complete(
            &prefix_root,
            &mtp_namespace,
            &prompt_ids,
            prefix_tokens,
        ),
        "persistent MTP snapshot write did not complete"
    );
    drop(prefix_cache);
    let mut prefix_cache = open_prefix_cache(cache_namespaces, &prefix_root);
    emit_setup_phase(
        "long_conversation_persist",
        persist_started.elapsed(),
        "cold",
    );

    // Warm benchmark iterations restore the filesystem-hydrated snapshot, so
    // establish both goldens only after taking that same lifecycle transition.
    let golden_started = Instant::now();
    let (baseline_token_ids, mtp_token_ids) = {
        let snapshot = exact_mtp_snapshot(&mut prefix_cache, &prompt_ids, prefix_tokens)
            .expect("filesystem-hydrated MTP snapshot");
        let (baseline, stats) = provider
            .benchmark_cached_streaming_in_mode(
                &prompt_ids,
                DECODE_MAX_TOKENS,
                &sampling,
                snapshot,
                Qwen4GenerationMode::Baseline,
                |delta| {
                    black_box(delta);
                    true
                },
            )
            .expect("warm hydrated cached 64k baseline");
        assert!(stats.is_none());
        let (baseline_repeat, stats) = provider
            .benchmark_cached_streaming_in_mode(
                &prompt_ids,
                DECODE_MAX_TOKENS,
                &sampling,
                snapshot,
                Qwen4GenerationMode::Baseline,
                |_| true,
            )
            .expect("repeat warm hydrated cached long-context baseline");
        assert!(stats.is_none());
        assert_eq!(
            baseline_repeat.token_ids, baseline.token_ids,
            "repeated baseline snapshot restore changed greedy output"
        );
        let (mtp, stats) = provider
            .benchmark_cached_streaming_in_mode(
                &prompt_ids,
                DECODE_MAX_TOKENS,
                &sampling,
                snapshot,
                Qwen4GenerationMode::Mtp,
                |delta| {
                    black_box(delta);
                    true
                },
            )
            .expect("warm hydrated cached 64k MTP");
        assert!(stats.is_some());
        let (mtp_repeat, stats) = provider
            .benchmark_cached_streaming_in_mode(
                &prompt_ids,
                DECODE_MAX_TOKENS,
                &sampling,
                snapshot,
                Qwen4GenerationMode::Mtp,
                |_| true,
            )
            .expect("repeat warm hydrated cached 64k MTP");
        assert!(stats.is_some());
        assert_eq!(
            mtp_repeat.token_ids, mtp.token_ids,
            "repeated MTP snapshot restore changed greedy output"
        );
        assert!(token_ids_valid(
            &baseline.token_ids,
            DECODE_MAX_TOKENS,
            provider.logits_vocab_size(),
        ));
        assert!(token_ids_valid(
            &mtp.token_ids,
            DECODE_MAX_TOKENS,
            provider.logits_vocab_size(),
        ));
        (baseline.token_ids, mtp.token_ids)
    };
    let manifest = LongConversationFixtureManifest {
        namespace,
        route: SnapshotRoute::Mtp,
        prompt_ids,
        prefix_tokens,
        baseline_token_ids,
        mtp_token_ids,
    };
    publish_fixture_manifest(&manifest_path, &manifest)
        .unwrap_or_else(|error| panic!("publish long-conversation fixture: {error}"));
    emit_setup_phase(
        "long_conversation_golden",
        golden_started.elapsed(),
        "cold",
    );
    {
        let snapshot = exact_mtp_snapshot(
            &mut prefix_cache,
            &manifest.prompt_ids,
            manifest.prefix_tokens,
        )
        .expect("cache-owned freshly persisted MTP snapshot");
        memory_report.emit_post_restore_fixture(
            manifest.prompt_ids.len(),
            manifest.prefix_tokens,
            snapshot,
            &manifest.baseline_token_ids,
            &manifest.mtp_token_ids,
        );
    }
    emit_setup_phase("long_conversation_ready", setup_started.elapsed(), "cold");
    LongConversationFixture {
        prompt_ids: manifest.prompt_ids,
        prefix_tokens: manifest.prefix_tokens,
        prefix_cache,
        baseline_token_ids: manifest.baseline_token_ids,
        mtp_token_ids: manifest.mtp_token_ids,
    }
}
