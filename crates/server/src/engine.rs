use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc as std_mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use mlxcel_core::generate::{GenerationStopReason, PrefixReuse};
use qw_prefix_cache::{
    AdaptivePrefixCache, CacheConfig, CacheNamespaces, ResponseResumeMetadata, ResumeLookupError,
    SnapshotRoute as CacheSnapshotRoute, namespace_hash,
};
#[cfg(test)]
use qw_runtime::ChatContentRef;
#[cfg(test)]
use qw_runtime::ChatMessage;
use qw_runtime::{KVCacheMode, MtpPrefixReuse, PromptSnapshot, Qwen4Provider};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{Span, debug, error, info, info_span, trace, warn};

use crate::grammar::GrammarFactory;
#[cfg(test)]
use crate::media::DecodedImage;

use crate::protocol::{
    CompletionRequest, Endpoint, OutputFormat, ReasoningEffort, ToolChoice, request_fingerprint,
};
use crate::tool_calls::{
    ToolCallStreamDelta, ToolCallStreamParser, parse_assistant_output,
};

const JOB_QUEUE_CAPACITY: usize = 8;
const EVENT_QUEUE_CAPACITY: usize = 32;
const JOB_BATCH_CAPACITY: usize = 4;
const CACHE_MAINTENANCE_CAPACITY: usize = JOB_QUEUE_CAPACITY;
const CACHE_MAINTENANCE_GRACE: Duration = Duration::from_millis(5);
struct CacheMaintenance {
    kind: &'static str,
    enqueued_at: Instant,
    work: CacheMaintenanceWork,
}

type CacheMaintenanceWork = Box<dyn FnOnce(&mut AdaptivePrefixCache)>;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq)]
struct GenerationMetrics {
    prefill_tps: f64,
    decode_tps: f64,
    total_tokens: usize,
    prefilled_tokens: usize,
    prefix_reused_tokens: usize,
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

fn generation_metrics(
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill_time: Duration,
    decode_time: Duration,
) -> GenerationMetrics {
    let prefilled_tokens = prompt_tokens.saturating_sub(cached_tokens);
    GenerationMetrics {
        prefill_tps: tokens_per_second(prefilled_tokens, prefill_time),
        decode_tps: tokens_per_second(completion_tokens, decode_time),
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        prefilled_tokens,
        prefix_reused_tokens: cached_tokens,
    }
}

pub(super) fn log_generation_metrics(
    chat_id: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill_time: Duration,
    decode_time: Duration,
) {
    let metrics = generation_metrics(
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        prefill_time,
        decode_time,
    );
    info!(
        event = "prefill.complete",
        chat_id,
        tps = metrics.prefill_tps,
        total_tokens = prompt_tokens,
        prefilled_tokens = metrics.prefilled_tokens,
        prefix_reused_tokens = metrics.prefix_reused_tokens,
    );
    trace!(
        event = "ttft.prefill",
        chat_id,
        prompt_tokens,
        cached_tokens,
        prefilled_tokens = metrics.prefilled_tokens,
        prefill_duration_ms = prefill_time.as_secs_f64() * 1_000.0,
        cache_reused = cached_tokens > 0,
    );
    info!(
        event = "decode.complete",
        chat_id,
        tps = metrics.decode_tps,
        total_tokens = metrics.total_tokens,
        decoded_tokens = completion_tokens,
        prefix_reused_tokens = metrics.prefix_reused_tokens,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenGenerationRoute {
    BaselineText,
    MtpText,
}

fn qwen_generation_route(has_mtp: bool, constraint_active: bool) -> QwenGenerationRoute {
    if has_mtp && !constraint_active {
        QwenGenerationRoute::MtpText
    } else {
        QwenGenerationRoute::BaselineText
    }
}
fn generation_temperature(requested: Option<f32>, constraint_active: bool) -> Option<f32> {
    requested.or_else(|| constraint_active.then_some(0.0))
}


fn cache_snapshot_route(route: QwenGenerationRoute) -> Option<CacheSnapshotRoute> {
    match route {
        QwenGenerationRoute::BaselineText => Some(CacheSnapshotRoute::Baseline),
        QwenGenerationRoute::MtpText => Some(CacheSnapshotRoute::Mtp),
    }
}

fn cache_lookup_route(route: QwenGenerationRoute) -> Option<CacheSnapshotRoute> {
    cache_snapshot_route(route)
}

fn validate_mtp_k(mtp_k: usize) -> Result<()> {
    ensure!(mtp_k >= 2, "--mtp-k must be at least 2");
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Admission {
    pub response_id: String,
    pub message_id: String,
    pub created: u64,
}

#[derive(Debug, Clone)]
pub struct GeneratedToolCall {
    pub id: String,
    pub item_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

#[derive(Debug, Clone)]
pub struct CompletionRecord {
    pub admission: Admission,
    pub endpoint: Endpoint,
    pub model: String,
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<GeneratedToolCall>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub finish_reason: FinishReason,
    pub stream_include_usage: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    InvalidRequest,
    Server,
    ResumeMismatch,
    ResumeNotFound,
    ResumeUnsupported,
}

#[derive(Debug, Clone)]
pub struct WorkerFailure {
    pub kind: FailureKind,
    pub message: String,
    pub param: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerDelta {
    Reasoning(String),
    Content(String),
    ToolCallStart { index: usize, name: String },
    ToolCallArguments { index: usize, fragment: String },
}

#[derive(Debug)]
pub enum WorkerEvent {
    Started(Admission),
    Delta(WorkerDelta),
    Complete { record: CompletionRecord },
    Failed(WorkerFailure),
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceState {
    Reasoning,
    AfterClose,
    Content,
}

#[derive(Debug)]
struct ReasoningTraceParser {
    state: TraceState,
    pending: String,
    checking_opener: bool,
    strip_opening_line_break: bool,
}

impl Default for ReasoningTraceParser {
    fn default() -> Self {
        Self {
            state: TraceState::Reasoning,
            pending: String::new(),
            checking_opener: true,
            strip_opening_line_break: false,
        }
    }
}
impl ReasoningTraceParser {
    fn new(enable_thinking: bool) -> Self {
        if enable_thinking {
            Self::default()
        } else {
            Self {
                state: TraceState::Content,
                pending: String::new(),
                checking_opener: false,
                strip_opening_line_break: false,
            }
        }
    }

    fn feed(&mut self, fragment: &str) -> Vec<WorkerDelta> {
        if fragment.is_empty() {
            return Vec::new();
        }
        if self.state == TraceState::Content {
            return vec![WorkerDelta::Content(fragment.to_string())];
        }
        if self.state == TraceState::AfterClose {
            return self.feed_after_close(fragment);
        }

        self.pending.push_str(fragment);
        if self.checking_opener {
            if self.pending.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(&self.pending) {
                return Vec::new();
            }
            if self.pending.starts_with(THINK_OPEN) {
                self.pending.drain(..THINK_OPEN.len());
                self.strip_opening_line_break = true;
            }
            self.checking_opener = false;
        }
        if self.strip_opening_line_break {
            let content = self.pending.trim_start_matches(['\r', '\n']);
            if content.len() != self.pending.len() {
                self.pending = content.to_string();
            }
            if self.pending.is_empty() {
                return Vec::new();
            }
            self.strip_opening_line_break = false;
        }

        if let Some(marker_start) = self.pending.find(THINK_CLOSE) {
            let reasoning = self.pending[..marker_start].to_string();
            let content_start = marker_start + THINK_CLOSE.len();
            let content = self.pending[content_start..].to_string();
            self.pending.clear();
            self.state = TraceState::AfterClose;
            let mut deltas = Vec::with_capacity(2);
            if !reasoning.is_empty() {
                deltas.push(WorkerDelta::Reasoning(reasoning));
            }
            deltas.extend(self.feed_after_close(&content));
            return deltas;
        }

        let retained = longest_marker_prefix_suffix(&self.pending);
        let released_len = self.pending.len() - retained;
        if released_len == 0 {
            return Vec::new();
        }
        let reasoning = self.pending.drain(..released_len).collect();
        vec![WorkerDelta::Reasoning(reasoning)]
    }

    fn finish(&mut self) -> Vec<WorkerDelta> {
        if self.state != TraceState::Reasoning || self.pending.is_empty() {
            self.pending.clear();
            return Vec::new();
        }
        vec![WorkerDelta::Reasoning(std::mem::take(&mut self.pending))]
    }

    fn feed_after_close(&mut self, fragment: &str) -> Vec<WorkerDelta> {
        let content = fragment.trim_start_matches(['\r', '\n']);
        if content.is_empty() {
            return Vec::new();
        }
        self.state = TraceState::Content;
        vec![WorkerDelta::Content(content.to_string())]
    }
}

fn longest_marker_prefix_suffix(text: &str) -> usize {
    (1..THINK_CLOSE.len())
        .rev()
        .find(|&length| text.ends_with(&THINK_CLOSE[..length]))
        .unwrap_or(0)
}

fn split_reasoning_trace(text: &str) -> (String, String) {
    let mut parser = ReasoningTraceParser::default();
    let mut reasoning = String::new();
    let mut content = String::new();
    for delta in parser.feed(text).into_iter().chain(parser.finish()) {
        match delta {
            WorkerDelta::Reasoning(fragment) => reasoning.push_str(&fragment),
            WorkerDelta::Content(fragment) => content.push_str(&fragment),
            WorkerDelta::ToolCallStart { .. } | WorkerDelta::ToolCallArguments { .. } => {
                unreachable!("reasoning parser emits only reasoning and content")
            }
        }
    }
    (reasoning, content)
}

fn convert_tool_delta(delta: ToolCallStreamDelta) -> WorkerDelta {
    match delta {
        ToolCallStreamDelta::Content(content) => WorkerDelta::Content(content),
        ToolCallStreamDelta::Start { index, name } => {
            WorkerDelta::ToolCallStart { index, name }
        }
        ToolCallStreamDelta::Arguments { index, fragment } => {
            WorkerDelta::ToolCallArguments { index, fragment }
        }
    }
}

fn stream_worker_delta(
    delta: WorkerDelta,
    tool_enabled: bool,
    tool_parser: &mut ToolCallStreamParser,
) -> Result<Vec<WorkerDelta>, String> {
    match delta {
        WorkerDelta::Reasoning(_) => Ok(vec![delta]),
        WorkerDelta::Content(content) if tool_enabled => tool_parser
            .feed(&content)
            .map(|deltas| deltas.into_iter().map(convert_tool_delta).collect())
            .map_err(|error| error.to_string()),
        WorkerDelta::Content(content) if content.is_empty() => Ok(Vec::new()),
        WorkerDelta::Content(_) => Ok(vec![delta]),
        WorkerDelta::ToolCallStart { .. } | WorkerDelta::ToolCallArguments { .. } => {
            Err("reasoning parser emitted a tool delta".to_string())
        }
    }
}

struct StreamOutputTracker {
    trace_parser: ReasoningTraceParser,
    tool_parser: ToolCallStreamParser,
    tool_enabled: bool,
    emitted_reasoning_text: String,
    emitted_content_text: String,
}

impl StreamOutputTracker {
    fn new(enable_thinking: bool, tool_enabled: bool) -> Self {
        Self {
            trace_parser: ReasoningTraceParser::new(enable_thinking),
            tool_parser: ToolCallStreamParser::default(),
            tool_enabled,
            emitted_reasoning_text: String::new(),
            emitted_content_text: String::new(),
        }
    }

    fn feed(&mut self, fragment: &str) -> Result<Vec<WorkerDelta>, String> {
        let mut deltas = Vec::new();
        for delta in self.trace_parser.feed(fragment) {
            deltas.extend(stream_worker_delta(
                delta,
                self.tool_enabled,
                &mut self.tool_parser,
            )?);
        }
        Ok(deltas)
    }

    fn finish(&mut self) -> Result<Vec<WorkerDelta>, String> {
        let mut deltas = Vec::new();
        for delta in self.trace_parser.finish() {
            deltas.extend(stream_worker_delta(
                delta,
                self.tool_enabled,
                &mut self.tool_parser,
            )?);
        }
        if self.tool_enabled {
            deltas.extend(
                self.tool_parser
                    .finish()
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .map(convert_tool_delta),
            );
        }
        Ok(deltas)
    }

    fn record_sent(&mut self, delta: &WorkerDelta) {
        match delta {
            WorkerDelta::Reasoning(fragment) => self.emitted_reasoning_text.push_str(fragment),
            WorkerDelta::Content(fragment) => self.emitted_content_text.push_str(fragment),
            WorkerDelta::ToolCallStart { .. } | WorkerDelta::ToolCallArguments { .. } => {}
        }
    }

    fn prime_and_replay(
        &mut self,
        raw_text: &str,
        emitted_reasoning_text: &str,
        emitted_content_text: &str,
    ) -> Result<Vec<WorkerDelta>, String> {
        let deltas = self.feed(raw_text)?;
        let mut reasoning_offset = 0;
        let mut content_offset = 0;
        let mut replay = Vec::new();
        for delta in deltas {
            let (reasoning, fragment, emitted, offset) = match delta {
                WorkerDelta::Reasoning(fragment) => (
                    true,
                    fragment,
                    emitted_reasoning_text,
                    &mut reasoning_offset,
                ),
                WorkerDelta::Content(fragment) => {
                    (false, fragment, emitted_content_text, &mut content_offset)
                }
                WorkerDelta::ToolCallStart { .. } | WorkerDelta::ToolCallArguments { .. } => {
                    return Err(
                        "response continuation cannot replay constrained tool output".to_string()
                    );
                }
            };
            let remaining = &emitted[*offset..];
            let skipped = remaining.len().min(fragment.len());
            if !remaining.is_char_boundary(skipped)
                || !fragment.is_char_boundary(skipped)
                || fragment[..skipped] != remaining[..skipped]
            {
                return Err("stored emitted output is not a prefix of accepted output".to_string());
            }
            *offset += skipped;
            if skipped < fragment.len() {
                let fragment = fragment[skipped..].to_string();
                replay.push(if reasoning {
                    WorkerDelta::Reasoning(fragment)
                } else {
                    WorkerDelta::Content(fragment)
                });
            }
        }
        if reasoning_offset != emitted_reasoning_text.len()
            || content_offset != emitted_content_text.len()
        {
            return Err("stored emitted output exceeds accepted output".to_string());
        }
        self.emitted_reasoning_text.push_str(emitted_reasoning_text);
        self.emitted_content_text.push_str(emitted_content_text);
        Ok(replay)
    }
}

fn send_delta(job: &Job, delta: WorkerDelta) -> bool {
    if job.cancelled.load(Ordering::Acquire) {
        debug!(phase = "generation.cancelled");
        return false;
    }
    if job.events.blocking_send(WorkerEvent::Delta(delta)).is_err() {
        job.cancelled.store(true, Ordering::Release);
        warn!(phase = "response.receiver_closed");
        false
    } else {
        true
    }
}

struct Job {
    request: CompletionRequest,
    admission: Admission,
    events: mpsc::Sender<WorkerEvent>,
    cancelled: Arc<AtomicBool>,
    span: Span,
}

pub struct Submission {
    pub admission: Admission,
    pub events: mpsc::Receiver<WorkerEvent>,
    pub cancelled: Arc<AtomicBool>,
    pub span: Span,
}

#[derive(Clone)]
pub struct Engine {
    jobs: mpsc::Sender<Job>,
    configured_model_id: Option<Arc<str>>,
    supports_image_inputs: bool,
}

impl Engine {
    pub fn start_qwen(
        model_path: PathBuf,
        model_id: Option<String>,
        cache_config: CacheConfig,
        mtp_k: usize,
        kv_cache_mode: KVCacheMode,
    ) -> Result<Self> {
        cache_config.validate().map_err(anyhow::Error::msg)?;
        validate_mtp_k(mtp_k)?;
        let (jobs_tx, jobs_rx) = mpsc::channel(JOB_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        thread::Builder::new()
            .name("qw-generation".to_string())
            .spawn(move || {
                match QwenWorker::load(&model_path, cache_config, mtp_k, kv_cache_mode) {
                    Ok(mut worker) => {
                        let _ = ready_tx.send(Ok(false));
                        worker.run(jobs_rx);
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })
            .context("failed to spawn generation thread")?;
        let supports_image_inputs = ready_rx
            .recv()
            .context("generation thread exited during startup")??;
        Ok(Self {
            jobs: jobs_tx,
            configured_model_id: model_id.map(Arc::from),
            supports_image_inputs,
        })
    }

    pub fn configured_model_id(&self) -> Option<&str> {
        self.configured_model_id.as_deref()
    }

    pub fn supports_image_inputs(&self) -> bool {
        self.supports_image_inputs
    }
    pub fn submit(&self, request: CompletionRequest) -> Result<Submission, SubmitError> {
        let admission = new_admission(request.endpoint);
        let span = info_span!(
            "generation",
            response_id = %admission.response_id,
            endpoint = ?request.endpoint,
            model = %request.model,
            stream = request.stream,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            image_count = request.image_params.len(),
            max_tokens = request.max_tokens,
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let (events_tx, events_rx) = mpsc::channel(EVENT_QUEUE_CAPACITY);
        let job = Job {
            request,
            admission: admission.clone(),
            events: events_tx,
            cancelled: cancelled.clone(),
            span: span.clone(),
        };
        self.jobs.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => SubmitError::Full,
            mpsc::error::TrySendError::Closed(_) => SubmitError::Closed,
        })?;
        debug!(parent: &span, phase = "dispatch.enqueued");
        Ok(Submission {
            admission,
            events: events_rx,
            cancelled,
            span,
        })
    }

    #[cfg(test)]
    pub fn start_fake(model_id: Option<&str>, queue_capacity: usize) -> Self {
        let (jobs_tx, mut jobs_rx) = mpsc::channel::<Job>(queue_capacity);
        thread::spawn(move || {
            fn message_text(message: &ChatMessage) -> String {
                let mut text = String::new();
                message.visit_content(|part| {
                    if let ChatContentRef::Text(part) = part {
                        text.push_str(part);
                    }
                });
                text
            }

            fn fake_prompt_observation(
                messages: &[ChatMessage],
                images: &[DecodedImage],
            ) -> String {
                let mut prompt = String::new();
                let mut image_index = 0;
                for (message_index, message) in messages.iter().enumerate() {
                    if message_index != 0 {
                        prompt.push('|');
                    }
                    message.visit_content(|part| match part {
                        ChatContentRef::Text(text) => prompt.push_str(text),
                        ChatContentRef::Image(_) => {
                            let image = images
                                .get(image_index)
                                .expect("decoded image order must match normalized content");
                            prompt.push_str("[image:");
                            prompt.push_str(image.format.as_str());
                            prompt.push(']');
                            image_index += 1;
                        }
                    });
                }
                assert_eq!(
                    image_index,
                    images.len(),
                    "decoded image order must match normalized content"
                );
                prompt
            }

            let grammar = GrammarFactory::single_byte().expect("single-byte grammar factory");
            #[derive(Clone)]
            struct FakeResume {
                admission: Admission,
                fingerprint: String,
                raw_text: String,
                completion_tokens: usize,
            }

            let mut response_resumes = std::collections::HashMap::<String, FakeResume>::new();
            let mut cached_prompt: Option<String> = None;
            while let Some(mut job) = jobs_rx.blocking_recv() {
                let span = job.span.clone();
                let _entered = span.enter();
                debug!(phase = "worker.accepted");
                let effective_tools = if job.request.tool_choice == ToolChoice::None {
                    &[][..]
                } else {
                    job.request.tools.as_slice()
                };
                let tool_grammar = matches!(job.request.output_format, OutputFormat::Text)
                    && !effective_tools.is_empty();
                let grammar_active = match grammar.compile(
                    &job.request.output_format,
                    effective_tools,
                    job.request.parallel_tool_calls,
                ) {
                    Ok(constraint) => constraint.is_some(),
                    Err(error) => {
                        send_failure(
                            &job,
                            FailureKind::InvalidRequest,
                            if tool_grammar {
                                format!("invalid tool schema: {error}")
                            } else {
                                format!("invalid structured output schema: {error}")
                            },
                            Some(
                                if tool_grammar {
                                    "tools"
                                } else {
                                    output_format_param(job.request.endpoint)
                                }
                                .to_string(),
                            ),
                        );
                        continue;
                    }
                };
                let fingerprint = request_fingerprint(&job.request);
                let mut resumed = None;
                if let Some(response_id) = &job.request.resume_response_id {
                    if !job.request.image_params.is_empty() || grammar_active {
                        send_failure(
                            &job,
                            FailureKind::ResumeUnsupported,
                            "response continuation supports only unconstrained text generation"
                                .to_string(),
                            Some("resume_response_id".to_string()),
                        );
                        continue;
                    }
                    let Some(checkpoint) = response_resumes.get(response_id).cloned() else {
                        send_failure(
                            &job,
                            FailureKind::ResumeNotFound,
                            "response continuation checkpoint was not found".to_string(),
                            Some("resume_response_id".to_string()),
                        );
                        continue;
                    };
                    if checkpoint.fingerprint != fingerprint {
                        send_failure(
                            &job,
                            FailureKind::ResumeMismatch,
                            "resume request does not match the original request".to_string(),
                            Some("resume_response_id".to_string()),
                        );
                        continue;
                    }
                    response_resumes.remove(response_id);
                    job.admission = checkpoint.admission.clone();
                    resumed = Some(checkpoint);
                }
                debug!(phase = "generation.started");
                if job
                    .events
                    .blocking_send(WorkerEvent::Started(job.admission.clone()))
                    .is_err()
                {
                    job.cancelled.store(true, Ordering::Release);
                    continue;
                }
                if message_text(&job.request.messages[0]) == "hold" {
                    while !job.cancelled.load(Ordering::Acquire) {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    debug!(phase = "generation.cancelled");
                    continue;
                }
                if message_text(&job.request.messages[0]) == "fail-after-start" {
                    send_failure(
                        &job,
                        FailureKind::Server,
                        "generation failed".to_string(),
                        None,
                    );
                    continue;
                }
                let has_images = !job.request.decoded_images.is_empty();
                let prompt =
                    fake_prompt_observation(&job.request.messages, &job.request.decoded_images);
                let cached_tokens = if has_images {
                    0
                } else {
                    cached_prompt.as_ref().map_or(0, |cached| {
                        cached
                            .bytes()
                            .zip(prompt.bytes())
                            .take_while(|(cached, requested)| cached == requested)
                            .count()
                    })
                };
                let tool_results = job
                    .request
                    .messages
                    .iter()
                    .filter(|message| message.role == "tool")
                    .filter_map(ChatMessage::text_content)
                    .collect::<Vec<_>>();
                let latest_user = job
                    .request
                    .messages
                    .iter()
                    .rev()
                    .find(|message| message.role == "user")
                    .map(message_text);
                if latest_user.as_deref() == Some("call-tool-parallel-violation")
                    && job.request.tool_choice == ToolChoice::Auto
                    && !job.request.parallel_tool_calls
                    && !job.request.tools.is_empty()
                {
                    send_failure(
                        &job,
                        FailureKind::Server,
                        "model generated parallel tool calls when parallel_tool_calls was false"
                            .to_string(),
                        None,
                    );
                    continue;
                }
                let fake_tool_turn = tool_results.is_empty()
                    && matches!(
                        latest_user.as_deref(),
                        Some("call-tool" | "call-tool-with-preamble" | "call-tool-with-reasoning")
                    )
                    && job.request.tool_choice == ToolChoice::Auto
                    && !job.request.tools.is_empty();
                let (content, tool_calls, finish_reason) = if fake_tool_turn {
                    let count = if job.request.parallel_tool_calls {
                        2
                    } else {
                        1
                    };
                    let tool_calls = (0..count)
                        .map(|index| {
                            let tool = &job.request.tools[index % job.request.tools.len()];
                            let arguments = if index == 0 {
                                r#"{"city":"Paris"}"#
                            } else {
                                r#"{"zone":"UTC"}"#
                            };
                            generated_tool_call(
                                &job.admission,
                                index,
                                tool.function.name.clone(),
                                arguments.to_string(),
                            )
                        })
                        .collect();
                    (
                        if latest_user.as_deref() == Some("call-tool-with-preamble") {
                            "I will use tools.".to_string()
                        } else {
                            String::new()
                        },
                        tool_calls,
                        FinishReason::ToolCalls,
                    )
                } else {
                    let text = if !tool_results.is_empty() {
                        format!("tool-results:{}", tool_results.join("|"))
                    } else {
                        match &job.request.output_format {
                            OutputFormat::Text => format!("echo:{prompt}"),
                            OutputFormat::JsonObject => "{\"answer\":1}".to_string(),
                            OutputFormat::JsonSchema { .. } => "{\"answer\":1}".to_string(),
                        }
                    };
                    let finish_reason = if job.request.max_tokens == 1 {
                        FinishReason::Length
                    } else {
                        FinishReason::Stop
                    };
                    (text, Vec::new(), finish_reason)
                };
                let reasoning = matches!(
                    latest_user.as_deref(),
                    Some("reasoning" | "call-tool-with-reasoning")
                )
                .then_some("fake reasoning")
                .unwrap_or_default();
                let generated_text = format!("{reasoning}{THINK_CLOSE}\n\n{content}");
                let interrupt =
                    resumed.is_none() && latest_user.as_deref() == Some("resume-interrupt");
                let emitted_text = if interrupt {
                    format!("{THINK_CLOSE}\n\necho:resume-")
                } else if let Some(checkpoint) = &resumed {
                    generated_text
                        .strip_prefix(&checkpoint.raw_text)
                        .unwrap_or(&generated_text)
                        .to_string()
                } else {
                    generated_text.clone()
                };
                let mut trace_parser = ReasoningTraceParser::default();
                if let Some(checkpoint) = &resumed {
                    let _ = trace_parser.feed(&checkpoint.raw_text);
                }
                for fragment in emitted_text.as_bytes().chunks(4) {
                    let fragment = String::from_utf8(fragment.to_vec()).expect("ASCII fake output");
                    for delta in trace_parser.feed(&fragment) {
                        if job.cancelled.load(Ordering::Acquire)
                            || job.events.blocking_send(WorkerEvent::Delta(delta)).is_err()
                        {
                            job.cancelled.store(true, Ordering::Release);
                            break;
                        }
                    }
                    if job.cancelled.load(Ordering::Acquire) {
                        break;
                    }
                }
                for delta in trace_parser.finish() {
                    if job.events.blocking_send(WorkerEvent::Delta(delta)).is_err() {
                        job.cancelled.store(true, Ordering::Release);
                        break;
                    }
                }
                if !interrupt {
                    for (index, call) in tool_calls.iter().enumerate() {
                        if !send_delta(
                            &job,
                            WorkerDelta::ToolCallStart {
                                index,
                                name: call.name.clone(),
                            },
                        ) || !send_delta(
                            &job,
                            WorkerDelta::ToolCallArguments {
                                index,
                                fragment: call.arguments.clone(),
                            },
                        ) {
                            break;
                        }
                    }
                }
                if job.cancelled.load(Ordering::Acquire) {
                    debug!(phase = "generation.cancelled");
                    continue;
                }
                if interrupt {
                    response_resumes.insert(
                        job.admission.response_id.clone(),
                        FakeResume {
                            admission: job.admission,
                            fingerprint,
                            raw_text: emitted_text,
                            completion_tokens: "echo:resume-".len(),
                        },
                    );
                    debug!(phase = "generation.cancelled");
                    continue;
                }
                cached_prompt = (!has_images && prompt.len() <= 16).then_some(prompt.clone());
                let completion_tokens = if let Some(checkpoint) = &resumed {
                    checkpoint.completion_tokens
                        + content.len().saturating_sub(checkpoint.completion_tokens)
                } else if job.request.max_tokens == 1 {
                    1
                } else {
                    reasoning.len()
                        + content.len()
                        + tool_calls
                            .iter()
                            .map(|call| call.arguments.len())
                            .sum::<usize>()
                };
                let (reasoning_content, content) = split_reasoning_trace(&generated_text);
                let record = CompletionRecord {
                    admission: job.admission,
                    endpoint: job.request.endpoint,
                    model: job.request.model.clone(),
                    content,
                    reasoning_content,
                    tool_calls,
                    prompt_tokens: prompt.len(),
                    completion_tokens,
                    cached_tokens: if resumed.is_some() {
                        cached_tokens.max(prompt.len())
                    } else {
                        cached_tokens
                    },
                    finish_reason,
                    stream_include_usage: job.request.stream_include_usage,
                };
                debug!(
                    phase = "generation.complete",
                    prompt_tokens = record.prompt_tokens,
                    completion_tokens = record.completion_tokens,
                    cached_tokens = record.cached_tokens,
                    finish_reason = ?record.finish_reason,
                    generated_tool_count = record.tool_calls.len(),
                );
                publish_completion(&job.events, record);
            }
        });
        Self {
            jobs: jobs_tx,
            configured_model_id: model_id.map(Arc::from),
            supports_image_inputs: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    Full,
    Closed,
}

struct QwenWorker {
    provider: Qwen4Provider,
    grammar: GrammarFactory,
    prefix_cache: AdaptivePrefixCache,
    mtp_k: usize,
}

enum WorkerNextAction<T> {
    Job(T),
    Maintenance,
    Wait,
    Stop,
}

fn next_worker_action<T>(
    job: Result<T, mpsc::error::TryRecvError>,
    has_maintenance: bool,
) -> WorkerNextAction<T> {
    match job {
        Ok(job) => WorkerNextAction::Job(job),
        Err(mpsc::error::TryRecvError::Empty) if has_maintenance => WorkerNextAction::Maintenance,
        Err(mpsc::error::TryRecvError::Empty) => WorkerNextAction::Wait,
        Err(mpsc::error::TryRecvError::Disconnected) => WorkerNextAction::Stop,
    }
}

fn collect_job_batch<T>(first: T, jobs: &mut mpsc::Receiver<T>) -> Vec<T> {
    let mut batch = Vec::with_capacity(JOB_BATCH_CAPACITY);
    batch.push(first);
    while batch.len() < JOB_BATCH_CAPACITY {
        match jobs.try_recv() {
            Ok(job) => batch.push(job),
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                break;
            }
        }
    }
    batch
}

fn enqueue_cache_maintenance(maintenance: &mut VecDeque<CacheMaintenance>, work: CacheMaintenance) {
    if maintenance.len() == CACHE_MAINTENANCE_CAPACITY {
        warn!(
            event = "cache.maintenance_dropped",
            kind = work.kind,
            queue_depth = maintenance.len(),
            queue_capacity = CACHE_MAINTENANCE_CAPACITY,
        );
        return;
    }
    maintenance.push_back(work);
}

impl QwenWorker {
    fn load(
        model_path: &Path,
        cache_config: CacheConfig,
        mtp_k: usize,
        kv_cache_mode: KVCacheMode,
    ) -> Result<Self> {
        validate_mtp_k(mtp_k)?;
        let provider = Qwen4Provider::load(model_path, kv_cache_mode)?;
        ensure!(
            provider.supports_qwen4_tool_calls(),
            "unsupported Qwen4 chat template: expected <tool_call>, <function=, and <parameter= literals"
        );
        let tokenizer_json_path = model_path.join("tokenizer.json");
        let tokenizer_json: Value = serde_json::from_slice(
            &std::fs::read(&tokenizer_json_path)
                .with_context(|| format!("failed to read {}", tokenizer_json_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", tokenizer_json_path.display()))?;
        let tokenizer_vocab_size = provider.tokenizer().get_vocab_size(true);
        let grammar = GrammarFactory::from_tokenizer_json(
            &tokenizer_json,
            tokenizer_vocab_size,
            provider.logits_vocab_size(),
            provider.eos_token_ids(),
        )?;
        let config_bytes = std::fs::read(model_path.join("config.json"))
            .context("failed to read model config for prefix cache namespace")?;
        let tokenizer_bytes = std::fs::read(model_path.join("tokenizer.json"))
            .context("failed to read tokenizer for prefix cache namespace")?;
        let tokenizer_config_bytes = std::fs::read(model_path.join("tokenizer_config.json"))
            .context("failed to read tokenizer config for prefix cache namespace")?;
        let tokenizer_config: Value = serde_json::from_slice(&tokenizer_config_bytes)
            .context("failed to parse tokenizer config for prefix cache namespace")?;
        let standalone_template = std::fs::read_to_string(model_path.join("chat_template.jinja"))
            .ok()
            .filter(|template| !template.trim().is_empty())
            .map(String::into_bytes);
        let selected_template = standalone_template.or_else(|| {
            tokenizer_config
                .get("chat_template")
                .and_then(Value::as_str)
                .map(|template| template.as_bytes().to_vec())
        });
        let selected_template =
            selected_template.context("loaded provider has no selected chat template")?;
        let cache_mode = format!("{kv_cache_mode:?}");
        let namespace_parts = |route: &'static [u8]| {
            namespace_hash(&[
                &config_bytes,
                &tokenizer_bytes,
                &selected_template,
                cache_mode.as_bytes(),
                route,
            ])
        };
        let namespaces = CacheNamespaces {
            baseline: namespace_parts(b"baseline"),
            mtp: namespace_parts(b"mtp"),
        };
        let prefix_cache =
            AdaptivePrefixCache::new(namespaces, cache_config).map_err(anyhow::Error::msg)?;
        Ok(Self {
            provider,
            grammar,
            prefix_cache,
            mtp_k,
        })
    }
    fn run(&mut self, mut jobs: mpsc::Receiver<Job>) {
        let mut maintenance = VecDeque::<CacheMaintenance>::new();
        loop {
            match next_worker_action(jobs.try_recv(), !maintenance.is_empty()) {
                WorkerNextAction::Job(first) => {
                    self.process_job_batch(first, &mut jobs, &mut maintenance);
                }
                WorkerNextAction::Maintenance => {
                    thread::sleep(CACHE_MAINTENANCE_GRACE);
                    match jobs.try_recv() {
                        Ok(first) => self.process_job_batch(first, &mut jobs, &mut maintenance),
                        Err(mpsc::error::TryRecvError::Empty) => {
                            self.run_cache_maintenance(&mut maintenance);
                        }
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            self.run_cache_maintenance(&mut maintenance);
                            break;
                        }
                    }
                }
                WorkerNextAction::Wait => match jobs.blocking_recv() {
                    Some(first) => self.process_job_batch(first, &mut jobs, &mut maintenance),
                    None => break,
                },
                WorkerNextAction::Stop => break,
            }
        }
    }

    fn process_job_batch(
        &mut self,
        first: Job,
        jobs: &mut mpsc::Receiver<Job>,
        maintenance: &mut VecDeque<CacheMaintenance>,
    ) {
        let batch = collect_job_batch(first, jobs);
        if batch.len() > 1 {
            info!(
                event = "worker.batch",
                batch_size = batch.len(),
                batch_capacity = JOB_BATCH_CAPACITY,
                job_queue_capacity = JOB_QUEUE_CAPACITY,
            );
        }
        for job in batch {
            self.process(job, maintenance);
        }
    }

    fn run_cache_maintenance(&mut self, maintenance: &mut VecDeque<CacheMaintenance>) {
        let batch_size = maintenance.len();
        let oldest_delay = maintenance
            .front()
            .map_or(Duration::ZERO, |queued| queued.enqueued_at.elapsed());
        let started = Instant::now();
        while let Some(queued) = maintenance.pop_front() {
            let kind = queued.kind;
            let queue_delay = queued.enqueued_at.elapsed();
            let work_started = Instant::now();
            (queued.work)(&mut self.prefix_cache);
            trace!(
                event = "cache.maintenance",
                kind,
                queue_delay_ms = queue_delay.as_secs_f64() * 1_000.0,
                execution_duration_ms = work_started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
        if batch_size > 1 {
            info!(
                event = "cache.maintenance_batch",
                batch_size,
                batch_capacity = CACHE_MAINTENANCE_CAPACITY,
                remaining_depth = maintenance.len(),
                oldest_queue_delay_ms = oldest_delay.as_secs_f64() * 1_000.0,
                execution_duration_ms = started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
    }

    fn process(&mut self, mut job: Job, maintenance: &mut VecDeque<CacheMaintenance>) {
        let span = job.span.clone();
        let _entered = span.enter();
        debug!(phase = "worker.accepted");
        let effective_tools = if job.request.tool_choice == ToolChoice::None {
            &[][..]
        } else {
            job.request.tools.as_slice()
        };
        let tool_grammar = matches!(job.request.output_format, OutputFormat::Text)
            && !effective_tools.is_empty();
        let mut constraint = match self.grammar.compile(
            &job.request.output_format,
            effective_tools,
            job.request.parallel_tool_calls,
        ) {
            Ok(constraint) => constraint,
            Err(error) => {
                send_failure(
                    &job,
                    FailureKind::InvalidRequest,
                    if tool_grammar {
                        format!("invalid tool schema: {error}")
                    } else {
                        format!("invalid structured output schema: {error}")
                    },
                    Some(
                        if tool_grammar {
                            "tools"
                        } else {
                            output_format_param(job.request.endpoint)
                        }
                        .to_string(),
                    ),
                );
                return;
            }
        };
        let enable_thinking = job.request.enable_thinking && constraint.is_none();
        let has_images = !job.request.decoded_images.is_empty();
        if job.request.resume_response_id.is_some() && (has_images || constraint.is_some()) {
            send_failure(
                &job,
                FailureKind::ResumeUnsupported,
                "response continuation supports only unconstrained text generation".to_string(),
                Some("resume_response_id".to_string()),
            );
            return;
        }
        if has_images {
            send_failure(
                &job,
                FailureKind::InvalidRequest,
                "model does not support image inputs".to_string(),
                job.request.image_params.first().cloned(),
            );
            return;
        }
        let tool_enabled = !effective_tools.is_empty();
        let reasoning_effort = job.request.reasoning_effort.map(ReasoningEffort::as_str);
        let prompt_ids = match self.provider.tokenize_messages(
            &job.request.messages,
            effective_tools,
            reasoning_effort,
            enable_thinking,
        ) {
            Ok(tokens) => tokens,
            Err(error) => {
                send_failure(
                    &job,
                    FailureKind::InvalidRequest,
                    format!("failed to render messages: {error}"),
                    Some("messages".to_string()),
                );
                return;
            }
        };
        let mut checkpoint_token_lengths = Vec::new();
        if !has_images {
            match self.provider.tokenize_history(
                &job.request.messages,
                effective_tools,
                reasoning_effort,
                enable_thinking,
            ) {
                Ok(history_ids)
                    if history_ids.len() < prompt_ids.len()
                        && prompt_ids.starts_with(&history_ids) =>
                {
                    checkpoint_token_lengths.push(history_ids.len());
                }
                Ok(_) => {}
                Err(error) => {
                    send_failure(
                        &job,
                        FailureKind::InvalidRequest,
                        format!("failed to render message history: {error}"),
                        Some("messages".to_string()),
                    );
                    return;
                }
            }
            checkpoint_token_lengths.push(prompt_ids.len());
        }
        debug!(
            phase = "prompt.prepared",
            prompt_tokens = prompt_ids.len(),
            image_count = 0,
            tool_count = effective_tools.len(),
        );

        let sampling = self.provider.baseline_sampling(
            generation_temperature(job.request.temperature, constraint.is_some()),
            job.request.top_p,
            job.request.seed,
        );
        let mtp_available =
            self.provider.has_mtp() && std::env::var_os("QW_BENCH_DISABLE_MTP").is_none();
        let route = qwen_generation_route(mtp_available, constraint.is_some());
        let cache_route = cache_snapshot_route(route);
        let lookup_cache_route = cache_lookup_route(route);
        if job.request.resume_response_id.is_some() && cache_route.is_none() {
            send_failure(
                &job,
                FailureKind::ResumeUnsupported,
                "response continuation supports only unconstrained text generation".to_string(),
                Some("resume_response_id".to_string()),
            );
            return;
        }
        let cache_lookup_started = Instant::now();
        let fingerprint = request_fingerprint(&job.request);
        let resume_entry = if let Some(response_id) = &job.request.resume_response_id {
            match self.prefix_cache.take_resume(
                response_id,
                &fingerprint,
                cache_route.expect("text resume route has a cache namespace"),
            ) {
                Ok(entry) => Some(entry),
                Err(ResumeLookupError::Mismatch) => {
                    send_failure(
                        &job,
                        FailureKind::ResumeMismatch,
                        "resume request does not match the original request".to_string(),
                        Some("resume_response_id".to_string()),
                    );
                    return;
                }
                Err(ResumeLookupError::NotFound) => {
                    send_failure(
                        &job,
                        FailureKind::ResumeNotFound,
                        "response continuation checkpoint was not found".to_string(),
                        Some("resume_response_id".to_string()),
                    );
                    return;
                }
            }
        } else {
            None
        };
        if let Some(resume) = &resume_entry {
            if !resume.token_ids.starts_with(&prompt_ids)
                || resume.metadata.prompt_token_count != prompt_ids.len()
                || resume.metadata.generated_token_ids.len() >= resume.metadata.original_max_tokens
                || !matches!(
                    (route, &resume.snapshot),
                    (
                        QwenGenerationRoute::BaselineText,
                        PromptSnapshot::Baseline(_)
                    ) | (QwenGenerationRoute::MtpText, PromptSnapshot::Mtp(_))
                )
            {
                send_failure(
                    &job,
                    FailureKind::ResumeNotFound,
                    "response continuation checkpoint was not found".to_string(),
                    Some("resume_response_id".to_string()),
                );
                return;
            }
            job.admission = Admission {
                response_id: resume.metadata.response_id.clone(),
                message_id: resume.metadata.message_id.clone(),
                created: resume.metadata.created_unix_seconds,
            };
        }
        let mut generation_prompt_ids = prompt_ids.clone();
        let mut max_tokens = job.request.max_tokens;
        if let Some(resume) = &resume_entry {
            generation_prompt_ids.extend_from_slice(&resume.metadata.generated_token_ids);
            max_tokens = resume
                .metadata
                .original_max_tokens
                .saturating_sub(resume.metadata.generated_token_ids.len());
            checkpoint_token_lengths.clear();
        }
        if job
            .events
            .blocking_send(WorkerEvent::Started(job.admission.clone()))
            .is_err()
        {
            job.cancelled.store(true, Ordering::Release);
            return;
        }
        if job.cancelled.load(Ordering::Acquire) {
            return;
        }
        let (resume_snapshot, prior_metadata) = match resume_entry {
            Some(resume) => (Some(resume.snapshot), Some(resume.metadata)),
            None => (None, None),
        };
        let mtp_k = self.mtp_k;
        let (provider, cache) = (&mut self.provider, &mut self.prefix_cache);
        if resume_snapshot.is_none()
            && let Some(lookup_cache_route) = lookup_cache_route
        {
            checkpoint_token_lengths = cache.checkpoint_lengths(
                &generation_prompt_ids,
                &checkpoint_token_lengths,
                lookup_cache_route,
            );
        }
        let hit = if resume_snapshot.is_none() {
            lookup_cache_route.and_then(|route| cache.lookup(&generation_prompt_ids, route))
        } else {
            None
        };
        let (prefix_reuse, mtp_prefix_reuse) = match (route, hit) {
            (QwenGenerationRoute::BaselineText, Some(hit)) => match hit.snapshot {
                PromptSnapshot::Baseline(snapshot) => (
                    Some(PrefixReuse {
                        snapshot,
                        cached_tokens: hit.token_count,
                    }),
                    None,
                ),
                PromptSnapshot::Mtp(snapshot) => (
                    Some(PrefixReuse {
                        snapshot: snapshot.target_snapshot(),
                        cached_tokens: hit.token_count,
                    }),
                    None,
                ),
            },
            (QwenGenerationRoute::MtpText, Some(hit)) => match hit.snapshot {
                PromptSnapshot::Mtp(snapshot) => (
                    None,
                    Some(MtpPrefixReuse {
                        snapshot,
                        cached_tokens: hit.token_count,
                        continuation_token: None,
                    }),
                ),
                PromptSnapshot::Baseline(_) => (None, None),
            },
            _ => (None, None),
        };
        let cache_source = if resume_snapshot.is_some() {
            "response_resume"
        } else if lookup_cache_route.is_none() {
            "disabled"
        } else if prefix_reuse.is_some() || mtp_prefix_reuse.is_some() {
            "prefix_hit"
        } else {
            "prefix_miss"
        };
        trace!(
            event = "prompt.tokenized",
            response_id = %job.admission.response_id,
            prompt_token_ids = ?generation_prompt_ids,
            prompt_tokens = generation_prompt_ids.len(),
            effective_tools = ?effective_tools,
            tool_choice = ?job.request.tool_choice,
            reasoning_effort = ?reasoning_effort,
            enable_thinking,
            max_tokens,
            temperature = ?job.request.temperature,
            top_p = ?job.request.top_p,
            seed = ?job.request.seed,
            route = ?route,
        );
        let reused_tokens = resume_snapshot.as_ref().map_or_else(
            || {
                prefix_reuse.as_ref().map_or_else(
                    || mtp_prefix_reuse.as_ref().map_or(0, |reuse| reuse.cached_tokens),
                    |reuse| reuse.cached_tokens,
                )
            },
            PromptSnapshot::token_len,
        );
        trace!(
            event = "cache.decision",
            response_id = %job.admission.response_id,
            route = ?route,
            source = cache_source,
            reused_tokens,
            prompt_tokens = generation_prompt_ids.len(),
            lookup_duration_ms = cache_lookup_started.elapsed().as_secs_f64() * 1_000.0,
        );
        debug!(
            phase = "model_generation.started",
            route = ?route,
            mtp_k,
            prefix_cached_tokens = reused_tokens,
        );
        let mut output = StreamOutputTracker::new(enable_thinking, tool_enabled);
        if let Some(metadata) = &prior_metadata {
            let replay = match output.prime_and_replay(
                &metadata.raw_text,
                &metadata.emitted_reasoning_text,
                &metadata.emitted_content_text,
            ) {
                Ok(replay) => replay,
                Err(error) => {
                    warn!(phase = "cache.resume", error = %error);
                    send_failure(
                        &job,
                        FailureKind::ResumeNotFound,
                        "response continuation checkpoint was not found".to_string(),
                        Some("resume_response_id".to_string()),
                    );
                    return;
                }
            };
            for delta in replay {
                if !send_delta(&job, delta.clone()) {
                    return;
                }
                output.record_sent(&delta);
            }
        }
        let mut stream_error = None;
        let mut emit_delta = |fragment: &str| {
            if job.cancelled.load(Ordering::Acquire) {
                return false;
            }
            let deltas = match output.feed(fragment) {
                Ok(deltas) => deltas,
                Err(error) => {
                    stream_error = Some(error);
                    return false;
                }
            };
            for delta in deltas {
                if !send_delta(&job, delta.clone()) {
                    return false;
                }
                output.record_sent(&delta);
            }
            true
        };
        let generated = match (route, resume_snapshot) {
            (QwenGenerationRoute::MtpText, Some(PromptSnapshot::Mtp(snapshot))) => provider
                .generate_mtp_streaming_owned_response(
                    &generation_prompt_ids,
                    max_tokens,
                    &sampling,
                    mtp_k,
                    snapshot,
                    prior_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.generated_token_ids.last().copied()),
                    &checkpoint_token_lengths,
                    constraint
                        .as_mut()
                        .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                    &mut emit_delta,
                ),
            (
                QwenGenerationRoute::BaselineText,
                Some(PromptSnapshot::Baseline(snapshot)),
            ) => provider.generate_baseline_streaming_owned_response(
                &generation_prompt_ids,
                max_tokens,
                &sampling,
                snapshot,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                &checkpoint_token_lengths,
                &mut emit_delta,
            ),
            (QwenGenerationRoute::MtpText, None) => provider.generate_mtp_streaming(
                &generation_prompt_ids,
                max_tokens,
                &sampling,
                mtp_k,
                mtp_prefix_reuse,
                &checkpoint_token_lengths,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                &mut emit_delta,
            ),
            (QwenGenerationRoute::BaselineText, None) => provider.generate_baseline_streaming(
                &generation_prompt_ids,
                max_tokens,
                &sampling,
                prefix_reuse,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                &checkpoint_token_lengths,
                &mut emit_delta,
            ),
            _ => unreachable!("response resume route was validated before generation"),
        };
        drop(emit_delta);
        if let Some(error) = stream_error {
            send_failure(
                &job,
                FailureKind::Server,
                format!("generated tool-call stream was invalid: {error}"),
                None,
            );
            return;
        }
        let mut generated = match generated {
            Ok(generated) => generated,
            Err(generation_error) => {
                error!(
                    phase = "model_generation.error",
                    error = %generation_error,
                );
                send_failure(
                    &job,
                    FailureKind::Server,
                    "generation failed".to_string(),
                    None,
                );
                return;
            }
        };
        log_generation_metrics(
            &job.admission.response_id,
            generated.prompt_tokens,
            generated.completion_tokens,
            generated.cached_tokens,
            generated.prefill_time,
            generated.decode_time,
        );
        debug!(
            phase = "model_generation.complete",
            prompt_tokens = generated.prompt_tokens,
            completion_tokens = generated.completion_tokens,
            cached_tokens = generated.cached_tokens,
            stop_reason = ?generated.finish_outcome,
        );
        let mut combined_token_ids = prior_metadata
            .as_ref()
            .map(|metadata| metadata.generated_token_ids.clone())
            .unwrap_or_default();
        combined_token_ids.extend_from_slice(&generated.token_ids);
        let mut combined_raw_text = prior_metadata
            .as_ref()
            .map(|metadata| metadata.raw_text.clone())
            .unwrap_or_default();
        combined_raw_text.push_str(&generated.text);
        if generated.finish_outcome != GenerationStopReason::CallbackCancelled {
            let deltas = match output.finish() {
                Ok(deltas) => deltas,
                Err(error) => {
                    send_failure(
                        &job,
                        FailureKind::Server,
                        format!("generated tool-call stream was invalid: {error}"),
                        None,
                    );
                    return;
                }
            };
            for delta in deltas {
                if !send_delta(&job, delta.clone()) {
                    break;
                }
                output.record_sent(&delta);
            }
        }
        let cancelled = job.cancelled.load(Ordering::Acquire)
            || generated.finish_outcome == GenerationStopReason::CallbackCancelled;
        if cancelled {
            if let Some(cache_route) = cache_route {
                let prompt_snapshots = std::mem::take(&mut generated.prompt_snapshots);
                let final_work = generated.final_snapshot.take().and_then(|final_snapshot| {
                    let mut completed_tokens =
                        Vec::with_capacity(generation_prompt_ids.len() + generated.token_ids.len());
                    completed_tokens.extend_from_slice(&generation_prompt_ids);
                    completed_tokens.extend_from_slice(&generated.token_ids);
                    completed_tokens.truncate(final_snapshot.token_len());
                    (completed_tokens.len() == final_snapshot.token_len())
                        .then_some((completed_tokens, final_snapshot))
                });
                let metadata = final_work.as_ref().map(|_| ResponseResumeMetadata {
                    response_id: job.admission.response_id.clone(),
                    message_id: job.admission.message_id.clone(),
                    created_unix_seconds: job.admission.created,
                    prompt_token_count: prior_metadata
                        .as_ref()
                        .map_or(prompt_ids.len(), |metadata| metadata.prompt_token_count),
                    request_fingerprint: fingerprint,
                    generated_token_ids: combined_token_ids.clone(),
                    raw_text: combined_raw_text.clone(),
                    emitted_reasoning_text: output.emitted_reasoning_text.clone(),
                    emitted_content_text: output.emitted_content_text.clone(),
                    original_max_tokens: prior_metadata
                        .as_ref()
                        .map_or(job.request.max_tokens, |metadata| {
                            metadata.original_max_tokens
                        }),
                });
                enqueue_cache_maintenance(
                    maintenance,
                    CacheMaintenance {
                        kind: "cancelled",
                        enqueued_at: Instant::now(),
                        work: Box::new(move |cache| {
                            if !prompt_snapshots.is_empty() {
                                cache.insert(&generation_prompt_ids, prompt_snapshots, cache_route);
                            }
                            if let (Some((completed_tokens, final_snapshot)), Some(metadata)) =
                                (final_work, metadata)
                            {
                                cache.insert_resume(
                                    &completed_tokens,
                                    final_snapshot,
                                    cache_route,
                                    metadata,
                                );
                            }
                        }),
                    },
                );
            }
            debug!(phase = "generation.cancelled");
            return;
        }
        generated.text = combined_raw_text;
        generated.token_ids = combined_token_ids;
        generated.completion_tokens = generated.token_ids.len();
        if let Some(metadata) = &prior_metadata {
            generated.prompt_tokens = metadata.prompt_token_count;
        }
        let core_finish_reason = match generated.finish_outcome {
            GenerationStopReason::MaxTokens => FinishReason::Length,
            GenerationStopReason::Eos
            | GenerationStopReason::ConstraintAccepted
            | GenerationStopReason::RepetitionLoop => FinishReason::Stop,
            GenerationStopReason::CallbackCancelled => unreachable!("handled above"),
        };
        let (reasoning_content, visible_content) = if enable_thinking {
            split_reasoning_trace(&generated.text)
        } else {
            (String::new(), generated.text.clone())
        };
        let (content, tool_calls, finish_reason) = if tool_enabled {
            let declared_names = effective_tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>();
            let parsed = match parse_assistant_output(
                &visible_content,
                &declared_names,
                generated.completion_tokens,
                job.request.max_tokens,
            ) {
                Ok(parsed) => parsed,
                Err(error) => {
                    send_failure(
                        &job,
                        FailureKind::Server,
                        format!("generated tool-call output was invalid: {error}"),
                        None,
                    );
                    return;
                }
            };
            if !job.request.parallel_tool_calls && parsed.tool_calls.len() > 1 {
                send_failure(
                    &job,
                    FailureKind::Server,
                    "model generated parallel tool calls when parallel_tool_calls was false"
                        .to_string(),
                    None,
                );
                return;
            }
            let has_calls = !parsed.tool_calls.is_empty();
            let tool_calls = parsed
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| {
                    generated_tool_call(&job.admission, index, call.name, call.arguments)
                })
                .collect();
            (
                parsed.content,
                tool_calls,
                if has_calls {
                    FinishReason::ToolCalls
                } else {
                    core_finish_reason
                },
            )
        } else {
            (visible_content, Vec::new(), core_finish_reason)
        };
        if !tool_enabled
            && !matches!(job.request.output_format, OutputFormat::Text)
            && finish_reason == FinishReason::Stop
            && serde_json::from_str::<Value>(&content).is_err()
        {
            send_failure(
                &job,
                FailureKind::Server,
                "structured generation did not produce one JSON value".to_string(),
                None,
            );
            return;
        }

        let record = CompletionRecord {
            admission: job.admission,
            endpoint: job.request.endpoint,
            model: job.request.model,
            content,
            reasoning_content,
            tool_calls,
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.completion_tokens,
            cached_tokens: generated.cached_tokens,
            finish_reason,
            stream_include_usage: job.request.stream_include_usage,
        };
        debug!(
            phase = "generation.complete",
            prompt_tokens = record.prompt_tokens,
            completion_tokens = record.completion_tokens,
            cached_tokens = record.cached_tokens,
            finish_reason = ?record.finish_reason,
            generated_tool_count = record.tool_calls.len(),
        );
        publish_completion(&job.events, record);
        if let Some(cache_route) = cache_route {
            let prompt_snapshots = std::mem::take(&mut generated.prompt_snapshots);
            let final_work = generated.final_snapshot.take().and_then(|final_snapshot| {
                let mut completed_tokens =
                    Vec::with_capacity(generation_prompt_ids.len() + generated.token_ids.len());
                completed_tokens.extend_from_slice(&generation_prompt_ids);
                completed_tokens.extend_from_slice(&generated.token_ids);
                completed_tokens.truncate(final_snapshot.token_len());
                (completed_tokens.len() == final_snapshot.token_len())
                    .then_some((completed_tokens, final_snapshot))
            });
            enqueue_cache_maintenance(
                maintenance,
                CacheMaintenance {
                    kind: "completed",
                    enqueued_at: Instant::now(),
                    work: Box::new(move |cache| {
                        if !prompt_snapshots.is_empty() {
                            cache.insert(&generation_prompt_ids, prompt_snapshots, cache_route);
                        }
                        if let Some((completed_tokens, final_snapshot)) = final_work {
                            cache.insert(&completed_tokens, vec![final_snapshot], cache_route);
                        }
                    }),
                },
            );
        }
    }
}

pub(super) fn generated_tool_call(
    admission: &Admission,
    index: usize,
    name: String,
    arguments: String,
) -> GeneratedToolCall {
    let suffix = admission
        .response_id
        .strip_prefix("chatcmpl-")
        .or_else(|| admission.response_id.strip_prefix("resp_"))
        .expect("admission response ID has a known prefix");
    GeneratedToolCall {
        id: format!("call_{suffix}_{index}"),
        item_id: format!("fc_{suffix}_{index}"),
        name,
        arguments,
    }
}
fn publish_completion(events: &mpsc::Sender<WorkerEvent>, record: CompletionRecord) {
    let _ = events.blocking_send(WorkerEvent::Complete { record });
}

fn send_failure(job: &Job, kind: FailureKind, message: String, param: Option<String>) {
    error!(
        phase = "generation.failed",
        failure_kind = ?kind,
        parameter = param.as_deref(),
        error = %message,
    );
    let _ = job.events.blocking_send(WorkerEvent::Failed(WorkerFailure {
        kind,
        message,
        param,
    }));
}

fn output_format_param(endpoint: Endpoint) -> &'static str {
    match endpoint {
        Endpoint::Chat => "response_format",
        Endpoint::Responses => "text.format",
    }
}

fn new_admission(endpoint: Endpoint) -> Admission {
    let ordinal = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let suffix = format!("{nanos:x}{ordinal:x}");
    let response_id = match endpoint {
        Endpoint::Chat => format!("chatcmpl-{suffix}"),
        Endpoint::Responses => format!("resp_{suffix}"),
    };
    Admission {
        response_id,
        message_id: format!("msg_{suffix}"),
        created: (nanos / 1_000_000_000) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_metrics_account_for_cache_totals_and_zero_durations() {
        let metrics = generation_metrics(100, 40, 25, Duration::from_secs(3), Duration::ZERO);

        assert_eq!(
            metrics,
            GenerationMetrics {
                prefill_tps: 25.0,
                decode_tps: 0.0,
                total_tokens: 140,
                prefilled_tokens: 75,
                prefix_reused_tokens: 25,
            }
        );
        assert_eq!(
            generation_metrics(usize::MAX, 1, usize::MAX, Duration::ZERO, Duration::ZERO,)
                .total_tokens,
            usize::MAX,
        );
    }

    fn parse_fragments(fragments: &[&str]) -> (String, String) {
        let mut parser = ReasoningTraceParser::default();
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut append = |delta| match delta {
            WorkerDelta::Reasoning(fragment) => reasoning.push_str(&fragment),
            WorkerDelta::Content(fragment) => content.push_str(&fragment),
            WorkerDelta::ToolCallStart { .. } | WorkerDelta::ToolCallArguments { .. } => {
                unreachable!("reasoning parser emitted a tool delta")
            }
        };
        for fragment in fragments {
            for delta in parser.feed(fragment) {
                append(delta);
            }
        }
        for delta in parser.finish() {
            append(delta);
        }
        (reasoning, content)
    }

    #[test]
    fn splits_normal_qwen_reasoning_trace() {
        assert_eq!(
            split_reasoning_trace("private trace</think>\nfinal answer"),
            ("private trace".to_string(), "final answer".to_string())
        );
    }

    #[test]
    fn removes_echoed_think_opener() {
        assert_eq!(
            parse_fragments(&["<thi", "nk>", "\nprivate trace", "</think>", "\nanswer"]),
            ("private trace".to_string(), "answer".to_string())
        );
    }

    #[test]
    fn closing_marker_split_at_every_byte_boundary_never_leaks() {
        for boundary in 0..=THINK_CLOSE.len() {
            let (reasoning, content) = parse_fragments(&[
                "private trace",
                &THINK_CLOSE[..boundary],
                &THINK_CLOSE[boundary..],
                "\n\nfinal answer",
            ]);
            assert_eq!(reasoning, "private trace", "boundary {boundary}");
            assert_eq!(content, "final answer", "boundary {boundary}");
            assert!(!reasoning.contains("<think"), "boundary {boundary}");
            assert!(!content.contains("</think"), "boundary {boundary}");
        }
    }

    #[test]
    fn removes_only_line_breaks_between_trace_and_visible_content() {
        assert_eq!(
            split_reasoning_trace("trace</think>\r\n\nanswer"),
            ("trace".to_string(), "answer".to_string())
        );
        assert_eq!(
            split_reasoning_trace("trace</think>  answer"),
            ("trace".to_string(), "  answer".to_string())
        );
    }

    #[test]
    fn disabled_thinking_keeps_all_output_visible() {
        let mut parser = ReasoningTraceParser::new(false);
        assert_eq!(
            parser.feed("The capital of France is Paris."),
            vec![WorkerDelta::Content(
                "The capital of France is Paris.".to_string()
            )]
        );
        assert!(parser.finish().is_empty());
    }

    #[test]
    fn resume_replays_callback_failed_fragment_without_duplicating_delivered_text() {
        let mut interrupted = StreamOutputTracker::new(false, false);
        let delivered = interrupted.feed("it is the").expect("stream text");
        assert_eq!(
            delivered,
            vec![WorkerDelta::Content("it is the".to_string())]
        );
        interrupted.record_sent(&delivered[0]);

        let failed = interrupted.feed(" perfect").expect("stream suffix");
        assert_eq!(failed, vec![WorkerDelta::Content(" perfect".to_string())]);

        let mut resumed = StreamOutputTracker::new(false, false);
        let replay = resumed
            .prime_and_replay(
                "it is the perfect",
                &interrupted.emitted_reasoning_text,
                &interrupted.emitted_content_text,
            )
            .expect("stored delivered text is a prefix");
        assert_eq!(replay, vec![WorkerDelta::Content(" perfect".to_string())]);
        resumed.record_sent(&replay[0]);
        let suffix = resumed.feed(" time").expect("stream continuation");
        resumed.record_sent(&suffix[0]);

        assert_eq!(resumed.emitted_content_text, "it is the perfect time");
    }

    #[test]
    fn resume_replays_only_unsent_delta_when_one_fragment_crosses_trace_boundary() {
        let raw_text = "private trace</think>\nfinal answer";
        let mut interrupted = StreamOutputTracker::new(true, false);
        let deltas = interrupted.feed(raw_text).expect("stream trace");
        assert_eq!(
            deltas,
            vec![
                WorkerDelta::Reasoning("private trace".to_string()),
                WorkerDelta::Content("final answer".to_string()),
            ]
        );
        interrupted.record_sent(&deltas[0]);

        let mut resumed = StreamOutputTracker::new(true, false);
        assert_eq!(
            resumed
                .prime_and_replay(
                    raw_text,
                    &interrupted.emitted_reasoning_text,
                    &interrupted.emitted_content_text,
                )
                .expect("partially delivered fragment replays"),
            vec![WorkerDelta::Content("final answer".to_string())]
        );
    }

    #[test]
    fn unfinished_generation_remains_reasoning() {
        assert_eq!(
            parse_fragments(&["unfinished trace", "</thi"]),
            ("unfinished trace</thi".to_string(), String::new())
        );
    }

    #[test]
    fn closing_marker_before_tool_call_exposes_only_tool_xml() {
        let tool_xml =
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>";
        let (reasoning, content) =
            split_reasoning_trace(&format!("use the weather tool</think>\n{tool_xml}"));
        assert_eq!(reasoning, "use the weather tool");
        assert_eq!(content, tool_xml);
        let parsed = parse_assistant_output(&content, &["weather"], 10, 128)
            .expect("visible tool call parses");
        assert!(parsed.content.is_empty());
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "weather");
    }

    #[test]
    fn active_constraints_always_route_to_baseline_generation() {
        assert_eq!(
            qwen_generation_route(false, false),
            QwenGenerationRoute::BaselineText
        );
        assert_eq!(
            qwen_generation_route(false, true),
            QwenGenerationRoute::BaselineText
        );
        assert_eq!(
            qwen_generation_route(true, true),
            QwenGenerationRoute::BaselineText
        );
        assert_eq!(
            qwen_generation_route(true, false),
            QwenGenerationRoute::MtpText
        );
    }
    #[test]
    fn constrained_generation_defaults_to_greedy_without_overriding_requests() {
        assert_eq!(generation_temperature(None, true), Some(0.0));
        assert_eq!(generation_temperature(None, false), None);
        assert_eq!(generation_temperature(Some(0.4), true), Some(0.4));
    }


    #[test]
    fn text_routes_are_prefix_cache_eligible() {
        for route in [
            QwenGenerationRoute::BaselineText,
            QwenGenerationRoute::MtpText,
        ] {
            assert!(cache_snapshot_route(route).is_some());
            assert!(cache_lookup_route(route).is_some());
        }
    }

    #[test]
    fn mtp_k_validation_rejects_library_callers_below_two() {
        assert!(validate_mtp_k(2).is_ok());
        assert_eq!(
            validate_mtp_k(1).expect_err("invalid K").to_string(),
            "--mtp-k must be at least 2"
        );
        assert_eq!(
            validate_mtp_k(0).expect_err("invalid K").to_string(),
            "--mtp-k must be at least 2"
        );
    }

    #[test]
    fn queued_job_takes_priority_over_cache_maintenance() {
        let (jobs, mut receiver) = mpsc::channel(1);
        jobs.try_send(7).expect("job queue accepts test job");

        let action = next_worker_action(receiver.try_recv(), true);

        assert!(matches!(action, WorkerNextAction::Job(7)));
    }

    #[test]
    fn job_batch_drains_only_the_configured_capacity() {
        let (jobs, mut receiver) = mpsc::channel(JOB_QUEUE_CAPACITY);
        for value in 1..=JOB_BATCH_CAPACITY {
            jobs.try_send(value).expect("test queue has capacity");
        }

        let first = receiver.try_recv().expect("first queued job");
        let batch = collect_job_batch(first, &mut receiver);

        assert_eq!(batch, (1..=JOB_BATCH_CAPACITY).collect::<Vec<_>>());
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn job_batch_leaves_excess_work_queued_for_backpressure() {
        let (jobs, mut receiver) = mpsc::channel(JOB_QUEUE_CAPACITY);
        for value in 1..=JOB_BATCH_CAPACITY + 1 {
            jobs.try_send(value).expect("test queue has capacity");
        }

        let first = receiver.try_recv().expect("first queued job");
        let batch = collect_job_batch(first, &mut receiver);

        assert_eq!(batch.len(), JOB_BATCH_CAPACITY);
        assert_eq!(
            receiver.try_recv().expect("excess job remains queued"),
            JOB_BATCH_CAPACITY + 1
        );
    }

    #[test]
    fn cache_maintenance_queue_drops_work_at_its_hard_bound() {
        let mut maintenance = VecDeque::new();
        for _ in 0..CACHE_MAINTENANCE_CAPACITY {
            enqueue_cache_maintenance(
                &mut maintenance,
                CacheMaintenance {
                    kind: "accepted",
                    enqueued_at: Instant::now(),
                    work: Box::new(|_| {}),
                },
            );
        }
        enqueue_cache_maintenance(
            &mut maintenance,
            CacheMaintenance {
                kind: "dropped",
                enqueued_at: Instant::now(),
                work: Box::new(|_| {}),
            },
        );

        assert_eq!(maintenance.len(), CACHE_MAINTENANCE_CAPACITY);
        assert!(maintenance.iter().all(|work| work.kind == "accepted"));
    }

    #[test]
    fn cache_maintenance_runs_when_job_queue_is_empty() {
        let (_jobs, mut receiver) = mpsc::channel::<u8>(1);

        let action = next_worker_action(receiver.try_recv(), true);

        assert!(matches!(action, WorkerNextAction::Maintenance));
    }
}
