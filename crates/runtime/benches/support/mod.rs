use std::fs::{self, File};
use std::hint::black_box;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use qw_runtime::provider::Qwen35GenerationMode;
use qw_runtime::{
    ChatMessage, ChatMessageContent, GenerationRequest, KVCacheMode, PortableArray,
    PortableModelState, PortablePage, PortablePagedTensor, PortablePromptSnapshot, PromptSnapshot,
    Qwen35Provider,
};

pub const DECODE_MAX_TOKENS: usize = 128;
pub const MTP_BLOCK_SIZE: usize = 3;
pub const PREFILL_MIN_TOKENS: usize = 4_096;
pub const PREFILL_MAX_TOKENS: usize = 6_000;
pub const LONG_CONTEXT_MIN_TOKENS: usize = 10_000;
pub const LONG_CONTEXT_64K_MIN_TOKENS: usize = 64_000;
/// Environment variable pointing at the DFlash2 drafter checkpoint
/// directory (optional; defaults to the model cache path below).
pub const DRAFT_MODEL_ENV: &str = "QW_BENCH_DRAFT_MODEL";
/// Default DFlash2 drafter identifier resolved through the model cache.
pub const DEFAULT_DRAFT_MODEL_IDENTIFIER: &str = "incoai/Qwen3.8-27B-DFlash2";
pub const PROMPT: &str = concat!(
    "You are the on-call support operations analyst for Acme Commerce. ",
    "Review this incident and return only one compact JSON object with keys ",
    "`severity`, `summary`, `affected_order_ids`, `next_action`, and ",
    "`needs_escalation`. `severity` must be `low`, `medium`, or `high`; ",
    "`affected_order_ids` must contain only active orders; do not include names, ",
    "email addresses, payment details, or unverified root causes. Treat all ",
    "timestamps as UTC. Escalate when payment capture failures are still occurring.\n\n",
    "Incident INC-4821: At 09:14, after catalog import job 771 completed, ",
    "checkout returned `price_mismatch` for order A-1042 (active, $129.00) and ",
    "order A-1047 (active, $89.50). Order A-1038 was canceled before the import ",
    "and must not be included. At 09:21, a retry for A-1042 succeeded; at 09:26, ",
    "a new payment capture for A-1047 failed with the same error. The importer ",
    "reported no validation errors. Customer notes mention a cardholder's email ",
    "address, which must not be repeated. The next action should be a specific ",
    "operational step, not a diagnosis."
);

pub struct DecodeFixture {
    pub request: GenerationRequest,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
    pub mtp_decode_tokens: usize,
}

pub struct LongConversationFixture {
    pub context_label: &'static str,
    pub prompt_ids: Vec<i32>,
    pub prefix_tokens: usize,
    pub new_prompt_tokens: usize,
    pub mtp_snapshot: PromptSnapshot,
    pub baseline_token_ids: Vec<i32>,
    pub mtp_token_ids: Vec<i32>,
    pub mtp_decode_tokens: usize,
    pub mtp_accepted_draft_tokens: usize,
    pub mtp_proposed_draft_tokens: usize,
    pub mtp_target_forward_calls: usize,
}

const LONG_CONTEXT_CACHE_VERSION: u32 = 2;
const MAX_CACHE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_CACHE_STRING_BYTES: usize = 1024 * 1024;
const MAX_CACHE_TOKEN_IDS: usize = 1024 * 1024;
const MAX_CACHE_TENSORS: usize = 4096;
const MAX_CACHE_PAGES: usize = 1_048_576;
const MAX_CACHE_RANK: usize = 16;

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x100000001b3;
    for &byte in bytes {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
}

fn long_context_cache_identity(context_label: &str, min_prefix_tokens: usize) -> io::Result<u64> {
    let model_dir = qw_runtime::resolve_model_path(None)
        .map_err(|error| io::Error::other(error.to_string()))?
        .canonicalize()?;
    let mut hash = 0xcbf29ce484222325_u64;
    for bytes in [
        b"long-context-fixture-v1".as_slice(),
        context_label.as_bytes(),
        &min_prefix_tokens.to_le_bytes(),
        &DECODE_MAX_TOKENS.to_le_bytes(),
        &MTP_BLOCK_SIZE.to_le_bytes(),
        std::env::consts::OS.as_bytes(),
        std::env::consts::ARCH.as_bytes(),
        b"alternating-user-assistant-history-v1".as_slice(),
        b"bounded-mtp-fp16-65536-v1".as_slice(),
        PROMPT.as_bytes(),
        include_bytes!("../../src/portable_snapshot.rs").as_slice(),
        include_bytes!("../../src/qwen3_5_mtp.rs").as_slice(),
    ] {
        hash_bytes(&mut hash, bytes);
    }
    hash_bytes(&mut hash, model_dir.as_os_str().as_encoded_bytes());

    let mut model_files = fs::read_dir(&model_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    model_files.sort();
    for path in model_files {
        if !path.is_file() {
            continue;
        }
        let extension = path.extension().and_then(|value| value.to_str());
        if !matches!(extension, Some("json" | "safetensors")) {
            continue;
        }
        hash_bytes(&mut hash, path.as_os_str().as_encoded_bytes());
        if extension == Some("json") {
            hash_bytes(&mut hash, &fs::read(&path)?);
            continue;
        }
        let metadata = path.metadata()?;
        hash_bytes(&mut hash, &metadata.len().to_le_bytes());
        let modified = metadata
            .modified()?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        hash_bytes(&mut hash, &modified.to_le_bytes());
    }
    Ok(hash)
}

fn long_context_cache_path(context_label: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/qw-bench-fixtures")
        .join(format!("long_{context_label}_mtp_k{MTP_BLOCK_SIZE}.bin"))
}

fn write_u8(writer: &mut impl Write, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_usize(writer: &mut impl Write, value: usize) -> io::Result<()> {
    write_u64(
        writer,
        u64::try_from(value)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "usize exceeds u64"))?,
    )
}

fn write_bytes(writer: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    write_usize(writer, bytes.len())?;
    writer.write_all(bytes)
}

fn write_string(writer: &mut impl Write, value: &str) -> io::Result<()> {
    write_bytes(writer, value.as_bytes())
}

fn write_i32s(writer: &mut impl Write, values: &[i32]) -> io::Result<()> {
    write_usize(writer, values.len())?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_portable_array(writer: &mut impl Write, array: &PortableArray) -> io::Result<()> {
    match &array.name {
        Some(name) => {
            write_u8(writer, 1)?;
            write_string(writer, name)?;
        }
        None => write_u8(writer, 0)?,
    }
    write_i32s(writer, &array.shape)?;
    writer.write_all(&array.dtype.to_le_bytes())?;
    write_bytes(writer, &array.bytes)
}

fn write_optional_array(writer: &mut impl Write, array: Option<&PortableArray>) -> io::Result<()> {
    match array {
        Some(array) => {
            write_u8(writer, 1)?;
            write_portable_array(writer, array)
        }
        None => write_u8(writer, 0),
    }
}

fn write_portable_model(writer: &mut impl Write, model: &PortableModelState) -> io::Result<()> {
    write_string(writer, &model.family)?;
    write_usize(writer, model.token_len)?;
    write_usize(writer, model.tensors.len())?;
    for tensor in &model.tensors {
        write_portable_array(writer, tensor)?;
    }
    write_usize(writer, model.paged_tensors.len())?;
    for tensor in &model.paged_tensors {
        write_string(writer, &tensor.name)?;
        write_usize(writer, tensor.token_axis)?;
        write_usize(writer, tensor.token_len)?;
        write_usize(writer, tensor.pages.len())?;
        for page in &tensor.pages {
            write_usize(writer, page.token_start)?;
            write_usize(writer, page.token_end)?;
            write_i32s(writer, &page.shape)?;
            writer.write_all(&page.dtype.to_le_bytes())?;
            write_bytes(writer, &page.bytes)?;
        }
    }
    write_optional_array(writer, model.continuation_logits.as_ref())
}

fn write_portable_snapshot(
    writer: &mut impl Write,
    snapshot: &PortablePromptSnapshot,
) -> io::Result<()> {
    match snapshot {
        PortablePromptSnapshot::Baseline(model) => {
            write_u8(writer, 0)?;
            write_portable_model(writer, model)
        }
        PortablePromptSnapshot::Mtp {
            target,
            draft,
            draft_offset,
            last_hidden,
            continuation_logits,
        } => {
            write_u8(writer, 1)?;
            write_portable_model(writer, target)?;
            write_portable_model(writer, draft)?;
            writer.write_all(&draft_offset.to_le_bytes())?;
            write_portable_array(writer, last_hidden)?;
            write_portable_array(writer, continuation_logits)
        }
    }
}


struct FixtureReader {
    inner: BufReader<File>,
    remaining: u64,
}

impl FixtureReader {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len > MAX_CACHE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "long-context fixture cache is unreasonably large",
            ));
        }
        Ok(Self {
            inner: BufReader::new(file),
            remaining: len,
        })
    }

    fn read_exact<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        if self.remaining < N as u64 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated long-context fixture cache",
            ));
        }
        let mut bytes = [0; N];
        self.inner.read_exact(&mut bytes)?;
        self.remaining -= N as u64;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_exact::<1>()?[0])
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn read_usize(&mut self, maximum: usize) -> io::Result<usize> {
        let value = usize::try_from(self.read_u64()?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "length exceeds usize"))?;
        if value > maximum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "long-context fixture cache length exceeds its bound",
            ));
        }
        Ok(value)
    }

    fn read_bytes(&mut self) -> io::Result<Vec<u8>> {
        let maximum = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        let len = self.read_usize(maximum)?;
        if len as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated long-context fixture cache payload",
            ));
        }
        let mut bytes = vec![0; len];
        self.inner.read_exact(&mut bytes)?;
        self.remaining -= len as u64;
        Ok(bytes)
    }

    fn read_string(&mut self) -> io::Result<String> {
        let bytes = self.read_bounded_bytes(MAX_CACHE_STRING_BYTES)?;
        String::from_utf8(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "cache string is not UTF-8"))
    }

    fn read_bounded_bytes(&mut self, maximum: usize) -> io::Result<Vec<u8>> {
        let len = self.read_usize(maximum)?;
        if len as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated long-context fixture cache payload",
            ));
        }
        let mut bytes = vec![0; len];
        self.inner.read_exact(&mut bytes)?;
        self.remaining -= len as u64;
        Ok(bytes)
    }

    fn read_i32s(&mut self, maximum: usize) -> io::Result<Vec<i32>> {
        let len = self.read_usize(maximum)?;
        let byte_len = len.checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "i32 vector length overflow")
        })?;
        if byte_len as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated long-context fixture cache i32 vector",
            ));
        }
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(i32::from_le_bytes(self.read_exact()?));
        }
        Ok(values)
    }

    fn read_optional_array(&mut self) -> io::Result<Option<PortableArray>> {
        match self.read_u8()? {
            0 => Ok(None),
            1 => self.read_portable_array().map(Some),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid optional-array tag",
            )),
        }
    }

    fn read_portable_array(&mut self) -> io::Result<PortableArray> {
        let name = match self.read_u8()? {
            0 => None,
            1 => Some(self.read_string()?),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid optional-string tag",
                ));
            }
        };
        let shape = self.read_i32s(MAX_CACHE_RANK)?;
        let dtype = i32::from_le_bytes(self.read_exact()?);
        let bytes = self.read_bytes()?;
        Ok(PortableArray {
            name,
            shape,
            dtype,
            bytes,
        })
    }

    fn read_portable_model(&mut self) -> io::Result<PortableModelState> {
        let family = self.read_string()?;
        let token_len = self.read_usize(MAX_CACHE_TOKEN_IDS)?;
        let tensor_count = self.read_usize(MAX_CACHE_TENSORS)?;
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            tensors.push(self.read_portable_array()?);
        }
        let paged_tensor_count = self.read_usize(MAX_CACHE_TENSORS)?;
        let mut paged_tensors = Vec::with_capacity(paged_tensor_count);
        for _ in 0..paged_tensor_count {
            let name = self.read_string()?;
            let token_axis = self.read_usize(MAX_CACHE_RANK)?;
            let token_len = self.read_usize(MAX_CACHE_TOKEN_IDS)?;
            let page_count = self.read_usize(MAX_CACHE_PAGES)?;
            let mut pages = Vec::with_capacity(page_count);
            for _ in 0..page_count {
                let token_start = self.read_usize(MAX_CACHE_TOKEN_IDS)?;
                let token_end = self.read_usize(MAX_CACHE_TOKEN_IDS)?;
                let shape = self.read_i32s(MAX_CACHE_RANK)?;
                let dtype = i32::from_le_bytes(self.read_exact()?);
                let bytes = self.read_bytes()?.into();
                pages.push(PortablePage {
                    token_start,
                    token_end,
                    shape,
                    dtype,
                    bytes,
                });
            }
            paged_tensors.push(PortablePagedTensor {
                name,
                token_axis,
                token_len,
                pages,
            });
        }
        let continuation_logits = self.read_optional_array()?;
        Ok(PortableModelState {
            family,
            token_len,
            tensors,
            paged_tensors,
            continuation_logits,
        })
    }

    fn read_portable_snapshot(&mut self) -> io::Result<PortablePromptSnapshot> {
        match self.read_u8()? {
            0 => self
                .read_portable_model()
                .map(PortablePromptSnapshot::Baseline),
            1 => {
                let target = self.read_portable_model()?;
                let draft = self.read_portable_model()?;
                let draft_offset = i32::from_le_bytes(self.read_exact()?);
                let last_hidden = self.read_portable_array()?;
                let continuation_logits = self.read_portable_array()?;
                Ok(PortablePromptSnapshot::Mtp {
                    target,
                    draft,
                    draft_offset,
                    last_hidden,
                    continuation_logits,
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid portable-snapshot tag",
            )),
        }
    }
}

fn write_long_context_cache(
    path: &Path,
    identity: u64,
    min_prefix_tokens: usize,
    fixture: &LongConversationFixture,
) -> io::Result<()> {
    let portable = fixture
        .mtp_snapshot
        .to_portable()
        .map_err(io::Error::other)?;
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("fixture cache path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let result = (|| {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        writer.write_all(LONG_CONTEXT_CACHE_MAGIC)?;
        write_u32(&mut writer, LONG_CONTEXT_CACHE_VERSION)?;
        write_u64(&mut writer, identity)?;
        write_string(&mut writer, fixture.context_label)?;
        write_usize(&mut writer, min_prefix_tokens)?;
        write_i32s(&mut writer, &fixture.prompt_ids)?;
        write_usize(&mut writer, fixture.prefix_tokens)?;
        write_usize(&mut writer, fixture.new_prompt_tokens)?;
        write_i32s(&mut writer, &fixture.baseline_token_ids)?;
        write_i32s(&mut writer, &fixture.mtp_token_ids)?;
        write_usize(&mut writer, fixture.mtp_decode_tokens)?;
        write_usize(&mut writer, fixture.mtp_accepted_draft_tokens)?;
        write_usize(&mut writer, fixture.mtp_proposed_draft_tokens)?;
        write_usize(&mut writer, fixture.mtp_target_forward_calls)?;
        write_portable_snapshot(&mut writer, &portable)?;
        writer.flush()?;
        drop(writer);
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn read_long_context_cache(
    path: &Path,
    expected_identity: u64,
    context_label: &'static str,
    min_prefix_tokens: usize,
) -> io::Result<Option<LongConversationFixture>> {
    let mut reader = FixtureReader::open(path)?;
    if reader.read_exact::<8>()? != *LONG_CONTEXT_CACHE_MAGIC
        || reader.read_u32()? != LONG_CONTEXT_CACHE_VERSION
        || reader.read_u64()? != expected_identity
    {
        return Ok(None);
    }
    if reader.read_string()? != context_label
        || reader.read_usize(MAX_CACHE_TOKEN_IDS)? != min_prefix_tokens
    {
        return Ok(None);
    }
    let prompt_ids = reader.read_i32s(MAX_CACHE_TOKEN_IDS)?;
    let prefix_tokens = reader.read_usize(MAX_CACHE_TOKEN_IDS)?;
    let new_prompt_tokens = reader.read_usize(MAX_CACHE_TOKEN_IDS)?;
    let baseline_token_ids = reader.read_i32s(DECODE_MAX_TOKENS)?;
    let mtp_token_ids = reader.read_i32s(DECODE_MAX_TOKENS)?;
    let mtp_decode_tokens = reader.read_usize(DECODE_MAX_TOKENS)?;
    let mtp_accepted_draft_tokens = reader.read_usize(MAX_CACHE_TOKEN_IDS)?;
    let mtp_proposed_draft_tokens = reader.read_usize(MAX_CACHE_TOKEN_IDS)?;
    let mtp_target_forward_calls = reader.read_usize(MAX_CACHE_TOKEN_IDS)?;
    let portable = reader.read_portable_snapshot()?;
    if reader.remaining != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "long-context fixture cache has trailing bytes",
        ));
    }
    if prefix_tokens < min_prefix_tokens
        || prompt_ids.len() <= prefix_tokens
        || prompt_ids.len() - prefix_tokens != new_prompt_tokens
        || baseline_token_ids.is_empty()
        || mtp_token_ids.is_empty()
        || mtp_token_ids.len() != mtp_decode_tokens
        || mtp_proposed_draft_tokens == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "long-context fixture cache metadata is inconsistent",
        ));
    }
    let mtp_snapshot = PromptSnapshot::from_portable(portable).map_err(io::Error::other)?;
    if !matches!(&mtp_snapshot, PromptSnapshot::Mtp(_)) || mtp_snapshot.token_len() != prefix_tokens
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "long-context fixture cache snapshot is inconsistent",
        ));
    }
    Ok(Some(LongConversationFixture {
        context_label,
        prompt_ids,
        prefix_tokens,
        new_prompt_tokens,
        mtp_snapshot,
        baseline_token_ids,
        mtp_token_ids,
        mtp_decode_tokens,
        mtp_accepted_draft_tokens,
        mtp_proposed_draft_tokens,
        mtp_target_forward_calls,
    }))
}

fn warm_cached_long_context_fixture(
    provider: &mut Qwen35Provider,
    fixture: &LongConversationFixture,
) {
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));
    let (baseline_output, baseline_stats) = provider
        .benchmark_cached_streaming_in_mode(
            &fixture.prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &fixture.mtp_snapshot,
            Qwen35GenerationMode::Baseline,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm restored long-conversation baseline fixture");
    assert_eq!(baseline_output.cached_tokens, fixture.prefix_tokens);
    assert_eq!(baseline_output.token_ids, fixture.baseline_token_ids);
    assert!(baseline_stats.is_none());

    let (mtp_output, mtp_stats) = provider
        .benchmark_cached_streaming_in_mode(
            &fixture.prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &fixture.mtp_snapshot,
            Qwen35GenerationMode::Mtp,
            |delta| {
                black_box(delta);
                true
            },
        )
        .unwrap_or_else(|error| {
            panic!("warm restored long-conversation MTP k={MTP_BLOCK_SIZE}: {error:#}")
        });
    assert_eq!(mtp_output.cached_tokens, fixture.prefix_tokens);
    assert_eq!(mtp_output.token_ids, fixture.mtp_token_ids);
    let mtp_stats = mtp_stats.expect("explicit MTP mode must return MTP statistics");
    assert_eq!(
        mtp_stats.accepted_draft_tokens,
        fixture.mtp_accepted_draft_tokens
    );
    assert_eq!(
        mtp_stats.proposed_draft_tokens,
        fixture.mtp_proposed_draft_tokens
    );
    assert_eq!(
        mtp_stats.target_forward_calls,
        fixture.mtp_target_forward_calls
    );
    eprintln!(
        "MTP_LONG_CONTEXT_PROFILE context={} tokens={} prefix_tokens={} accepted={} proposed={} acceptance={:.2}% forwards={} draft_ms={:.3} verify_ms={:.3} walk_ms={:.3} reconcile_ms={:.3} materializations={} snapshots={} fixture_cache=hit",
        fixture.context_label,
        mtp_output.token_ids.len(),
        fixture.prefix_tokens,
        mtp_stats.accepted_draft_tokens,
        mtp_stats.proposed_draft_tokens,
        mtp_stats.acceptance_percentage(),
        mtp_stats.target_forward_calls,
        mtp_stats.draft_time.as_secs_f64() * 1_000.0,
        mtp_stats.target_verify_time.as_secs_f64() * 1_000.0,
        mtp_stats.walk_time.as_secs_f64() * 1_000.0,
        mtp_stats.reconcile_time.as_secs_f64() * 1_000.0,
        mtp_stats.full_state_materializations,
        mtp_stats.cache_snapshot_count,
    );
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

pub fn load_provider() -> Qwen35Provider {
    // Unified with `qw generate` / `qw serve`: QW_MODEL_PATH override, else
    // the model cache path for the resolver's default identifier.
    let model_dir = qw_runtime::resolve_model_path(None)
        .unwrap_or_else(|error| panic!("failed to resolve benchmark model path: {error:#}"));
    Qwen35Provider::load(&model_dir, KVCacheMode::Turbo4)
        .unwrap_or_else(|error| panic!("failed to load {}: {error:#}", model_dir.display()))
}

pub fn prompt_token_ids(provider: &Qwen35Provider) -> Vec<i32> {
    let mut prompt = String::new();
    for record_index in 1..=32 {
        prompt.push_str(&format!(
            "Operational record {record_index:02}\n{PROMPT}\n\n"
        ));
        let prompt_ids = provider
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
            .expect("tokenize single-user prefill benchmark prompt");
        if prompt_ids.len() >= PREFILL_MIN_TOKENS {
            assert!(
                prompt_ids.len() <= PREFILL_MAX_TOKENS,
                "single-user prefill prompt has {} tokens, expected at most {PREFILL_MAX_TOKENS}",
                prompt_ids.len()
            );
            return prompt_ids;
        }
    }
    panic!("single-user prefill prompt did not reach {PREFILL_MIN_TOKENS} tokens");
}

/// Resolve the DFlash2 drafter directory: `QW_BENCH_DRAFT_MODEL` when set,
/// else the model cache path for the default identifier (mirrors the model
/// resolution `qw generate` / `qw serve` use).
#[allow(dead_code)]
pub fn draft_model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(DRAFT_MODEL_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME must be set to resolve the default drafter cache path");
    qw_runtime::model_cache_path(&home, DEFAULT_DRAFT_MODEL_IDENTIFIER)
        .expect("default DFlash2 drafter identifier is valid")
}

pub fn prepare_decode_fixture(provider: &mut Qwen35Provider) -> DecodeFixture {
    let request = request(DECODE_MAX_TOKENS);
    let (baseline_output, _) = provider
        .generate_streaming_in_mode(&request, Qwen35GenerationMode::Baseline, |delta| {
            black_box(delta);
            true
        })
        .expect("warm up baseline single-user decode");
    let baseline_token_ids = baseline_output.token_ids;
    assert!(!baseline_token_ids.is_empty());

    let (mtp_output, mtp_stats) = provider
        .generate_streaming_in_mode(&request, Qwen35GenerationMode::Mtp, |delta| {
            black_box(delta);
            true
        })
        .unwrap_or_else(|error| panic!("warm up MTP k={MTP_BLOCK_SIZE}: {error:#}"));
    let mtp_stats = mtp_stats.expect("explicit MTP mode must return MTP statistics");
    assert!(
        !mtp_output.token_ids.is_empty(),
        "the deterministic MTP prompt must produce at least one completion token"
    );
    assert!(
        mtp_stats.proposed_draft_tokens > 0,
        "MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
    );
    eprintln!(
        "MTP_PROFILE tokens={} accepted={} proposed={} acceptance={:.2}% forwards={} draft_ms={:.3} verify_ms={:.3} walk_ms={:.3} reconcile_ms={:.3} materializations={} snapshots={}",
        mtp_output.token_ids.len(),
        mtp_stats.accepted_draft_tokens,
        mtp_stats.proposed_draft_tokens,
        mtp_stats.acceptance_percentage(),
        mtp_stats.target_forward_calls,
        mtp_stats.draft_time.as_secs_f64() * 1_000.0,
        mtp_stats.target_verify_time.as_secs_f64() * 1_000.0,
        mtp_stats.walk_time.as_secs_f64() * 1_000.0,
        mtp_stats.reconcile_time.as_secs_f64() * 1_000.0,
        mtp_stats.full_state_materializations,
        mtp_stats.cache_snapshot_count,
    );

    let mtp_decode_tokens = mtp_output.token_ids.len();
    DecodeFixture {
        request,
        baseline_token_ids,
        mtp_token_ids: mtp_output.token_ids,
        mtp_decode_tokens,
    }
}

fn text_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        name: None,
        content: Some(ChatMessageContent::Text(content)),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn long_conversation_token_ids(
    provider: &Qwen35Provider,
    context_label: &str,
    min_prefix_tokens: usize,
) -> (Vec<i32>, usize) {
    let supported_context_tokens = provider.supported_context_tokens();
    assert!(
        min_prefix_tokens + DECODE_MAX_TOKENS < supported_context_tokens,
        "{context_label} minimum prefix of {min_prefix_tokens} tokens leaves no room in the model's {supported_context_tokens}-token context"
    );

    let mut messages = Vec::new();
    let mut turn = 1;
    let history_ids = loop {
        messages.push(text_message(
            "user",
            format!("Conversation incident record {turn:03}\n{PROMPT}"),
        ));
        messages.push(text_message(
            "assistant",
            concat!(
                r#"{"severity":"high","summary":"Payment capture failures remain active.","#,
                r#""affected_order_ids":["A-1042","A-1047"],"#,
                r#""next_action":"Escalate INC-4821 and pause catalog imports.","#,
                r#""needs_escalation":true}"#
            )
            .to_owned(),
        ));
        let history_ids = provider
            .tokenize_history(&messages, &[], None, true)
            .expect("tokenize long-conversation benchmark history");
        if history_ids.len() >= min_prefix_tokens {
            break history_ids;
        }
        assert!(
            history_ids.len() + DECODE_MAX_TOKENS < supported_context_tokens,
            "{context_label} long-conversation history cannot reach {min_prefix_tokens} tokens within the model's {supported_context_tokens}-token context"
        );
        turn += 1;
    };
    let prefix_tokens = history_ids.len();
    assert!(
        prefix_tokens >= min_prefix_tokens,
        "{context_label} long-conversation prefix has {prefix_tokens} tokens, expected at least {min_prefix_tokens}"
    );

    messages.push(text_message("user", PROMPT.to_owned()));
    let prompt_ids = provider
        .tokenize_messages(&messages, &[], None, true)
        .expect("tokenize long-conversation benchmark continuation");
    assert!(
        prompt_ids.starts_with(&history_ids),
        "{context_label} long-conversation history must be an exact prefix of the continuation prompt"
    );
    assert!(
        prompt_ids.len() > prefix_tokens,
        "{context_label} long-conversation continuation must add prompt tokens"
    );
    assert!(
        prompt_ids.len() + DECODE_MAX_TOKENS <= supported_context_tokens,
        "{context_label} prompt and decode budget require {} tokens, exceeding the model's {supported_context_tokens}-token context",
        prompt_ids.len() + DECODE_MAX_TOKENS
    );
    (prompt_ids, prefix_tokens)
}

fn prepare_long_conversation_fixture_uncached(
    provider: &mut Qwen35Provider,
    context_label: &'static str,
    min_prefix_tokens: usize,
) -> LongConversationFixture {
    let (prompt_ids, prefix_tokens) =
        long_conversation_token_ids(provider, context_label, min_prefix_tokens);
    let history_ids = &prompt_ids[..prefix_tokens];
    let sampling = provider.baseline_sampling(Some(0.0), Some(1.0), Some(0));

    let mtp_prefix = provider
        .generate_mtp_streaming(
            history_ids,
            1,
            &sampling,
            MTP_BLOCK_SIZE,
            None,
            &[prefix_tokens],
            None,
            |_| true,
        )
        .expect("prefill long-conversation MTP prefix");
    let mtp_snapshot = mtp_prefix
        .prompt_snapshots
        .into_iter()
        .next()
        .expect("capture long-conversation MTP prefix snapshot");
    assert!(
        matches!(&mtp_snapshot, PromptSnapshot::Mtp(_)),
        "MTP prefix generation returned the wrong snapshot family"
    );
    assert_eq!(mtp_snapshot.token_len(), prefix_tokens);

    let (baseline_output, baseline_stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &mtp_snapshot,
            Qwen35GenerationMode::Baseline,
            |delta| {
                black_box(delta);
                true
            },
        )
        .expect("warm up long-conversation baseline decode");
    assert_eq!(baseline_output.cached_tokens, prefix_tokens);
    assert!(
        baseline_stats.is_none(),
        "long-conversation baseline mode returned MTP statistics"
    );
    assert!(!baseline_output.token_ids.is_empty());

    let (mtp_output, mtp_stats) = provider
        .benchmark_cached_streaming_in_mode(
            &prompt_ids,
            DECODE_MAX_TOKENS,
            &sampling,
            &mtp_snapshot,
            Qwen35GenerationMode::Mtp,
            |delta| {
                black_box(delta);
                true
            },
        )
        .unwrap_or_else(|error| {
            panic!("warm up long-conversation MTP k={MTP_BLOCK_SIZE}: {error:#}")
        });
    assert_eq!(mtp_output.cached_tokens, prefix_tokens);
    let mtp_stats = mtp_stats.expect("explicit MTP mode must return MTP statistics");
    assert!(
        !mtp_output.token_ids.is_empty(),
        "the deterministic long-conversation prompt must produce completion tokens"
    );
    assert!(
        mtp_stats.proposed_draft_tokens > 0,
        "long-conversation MTP k={MTP_BLOCK_SIZE} must propose draft tokens"
    );
    eprintln!(
        "MTP_LONG_CONTEXT_PROFILE context={} tokens={} prefix_tokens={} accepted={} proposed={} acceptance={:.2}% forwards={} draft_ms={:.3} verify_ms={:.3} walk_ms={:.3} reconcile_ms={:.3} materializations={} snapshots={}",
        context_label,
        mtp_output.token_ids.len(),
        prefix_tokens,
        mtp_stats.accepted_draft_tokens,
        mtp_stats.proposed_draft_tokens,
        mtp_stats.acceptance_percentage(),
        mtp_stats.target_forward_calls,
        mtp_stats.draft_time.as_secs_f64() * 1_000.0,
        mtp_stats.target_verify_time.as_secs_f64() * 1_000.0,
        mtp_stats.walk_time.as_secs_f64() * 1_000.0,
        mtp_stats.reconcile_time.as_secs_f64() * 1_000.0,
        mtp_stats.full_state_materializations,
        mtp_stats.cache_snapshot_count,
    );

    let new_prompt_tokens = prompt_ids.len() - prefix_tokens;
    let mtp_decode_tokens = mtp_output.token_ids.len();
    let mtp_accepted_draft_tokens = mtp_stats.accepted_draft_tokens;
    let mtp_proposed_draft_tokens = mtp_stats.proposed_draft_tokens;
    let mtp_target_forward_calls = mtp_stats.target_forward_calls;
    LongConversationFixture {
        context_label,
        prompt_ids,
        prefix_tokens,
        new_prompt_tokens,
        mtp_snapshot,
        baseline_token_ids: baseline_output.token_ids,
        mtp_token_ids: mtp_output.token_ids,
        mtp_decode_tokens,
        mtp_accepted_draft_tokens,
        mtp_proposed_draft_tokens,
        mtp_target_forward_calls,
    }
}

pub fn prepare_long_conversation_fixture(
    provider: &mut Qwen35Provider,
    context_label: &'static str,
    min_prefix_tokens: usize,
) -> LongConversationFixture {
    let cache_path = long_context_cache_path(context_label);
    let identity = match long_context_cache_identity(context_label, min_prefix_tokens) {
        Ok(identity) => Some(identity),
        Err(error) => {
            eprintln!(
                "LONG_CONTEXT_FIXTURE_CACHE disabled path={} error={error}",
                cache_path.display()
            );
            None
        }
    };
    if let Some(identity) = identity {
        match read_long_context_cache(&cache_path, identity, context_label, min_prefix_tokens) {
            Ok(Some(fixture)) => {
                eprintln!(
                    "LONG_CONTEXT_FIXTURE_CACHE hit path={}",
                    cache_path.display()
                );
                warm_cached_long_context_fixture(provider, &fixture);
                return fixture;
            }
            Ok(None) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                eprintln!(
                    "LONG_CONTEXT_FIXTURE_CACHE rebuild path={} error={error}",
                    cache_path.display()
                );
            }
        }
    }

    let fixture =
        prepare_long_conversation_fixture_uncached(provider, context_label, min_prefix_tokens);
    if let Some(identity) = identity {
        match write_long_context_cache(&cache_path, identity, min_prefix_tokens, &fixture) {
            Ok(()) => eprintln!(
                "LONG_CONTEXT_FIXTURE_CACHE stored path={}",
                cache_path.display()
            ),
            Err(error) => eprintln!(
                "LONG_CONTEXT_FIXTURE_CACHE write_failed path={} error={error}",
                cache_path.display()
            ),
        }
    }
    fixture
}
