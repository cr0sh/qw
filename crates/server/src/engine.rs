use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc as std_mpsc};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use mlxcel_core::generate::{GenerationStopReason, PrefixReuse};
#[cfg(test)]
use qw_runtime::{ChatContentRef, ChatMessage};
use qw_runtime::{KVCacheMode, MtpPrefixReuse, PromptSnapshot, Qwen35Provider};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{Span, error, info, info_span, warn};

use crate::grammar::GrammarFactory;
#[cfg(test)]
use crate::media::DecodedImage;
use crate::prefix_cache::PrefixCache;
use crate::protocol::{CompletionRequest, Endpoint, OutputFormat, ReasoningEffort, ToolChoice};
use crate::tool_calls::{ToolCallGate, parse_assistant_output};

const JOB_QUEUE_CAPACITY: usize = 8;
const EVENT_QUEUE_CAPACITY: usize = 32;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenGenerationRoute {
    BaselineText,
    BaselineMultimodal,
    MtpText,
    MtpMultimodal,
}

fn qwen_generation_route(
    has_mtp: bool,
    has_images: bool,
    _has_constraint: bool,
) -> QwenGenerationRoute {
    match (has_mtp, has_images) {
        (true, false) => QwenGenerationRoute::MtpText,
        (true, true) => QwenGenerationRoute::MtpMultimodal,
        (false, false) => QwenGenerationRoute::BaselineText,
        (false, true) => QwenGenerationRoute::BaselineMultimodal,
    }
}

fn route_uses_prefix_cache(route: QwenGenerationRoute) -> bool {
    matches!(
        route,
        QwenGenerationRoute::BaselineText | QwenGenerationRoute::MtpText
    )
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
}

#[derive(Debug)]
pub enum WorkerEvent {
    Started,
    Delta(WorkerDelta),
    Complete(CompletionRecord),
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
        }
    }
    (reasoning, content)
}

fn gate_worker_delta(
    delta: WorkerDelta,
    tool_enabled: bool,
    gate: &mut ToolCallGate,
) -> Option<WorkerDelta> {
    match delta {
        WorkerDelta::Reasoning(_) => Some(delta),
        WorkerDelta::Content(content) if tool_enabled => {
            gate.feed(&content).map(WorkerDelta::Content)
        }
        WorkerDelta::Content(content) if content.is_empty() => None,
        WorkerDelta::Content(_) => Some(delta),
    }
}

fn send_delta(job: &Job, delta: WorkerDelta) -> bool {
    if job.cancelled.load(Ordering::Acquire) {
        info!(phase = "generation.cancelled");
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
        prefix_cache_max_tokens: usize,
        mtp_k: usize,
        kv_cache_mode: KVCacheMode,
    ) -> Result<Self> {
        ensure!(
            prefix_cache_max_tokens > 0,
            "prefix cache capacity must be nonzero"
        );
        validate_mtp_k(mtp_k)?;
        let (jobs_tx, jobs_rx) = mpsc::channel(JOB_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        thread::Builder::new()
            .name("qw-generation".to_string())
            .spawn(move || {
                match QwenWorker::load(&model_path, prefix_cache_max_tokens, mtp_k, kv_cache_mode) {
                    Ok(mut worker) => {
                        let supports_image_inputs = worker.provider.supports_image_inputs();
                        let _ = ready_tx.send(Ok(supports_image_inputs));
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
        info!(parent: &span, phase = "dispatch.enqueued");
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
            let mut cached_prompt: Option<String> = None;
            while let Some(job) = jobs_rx.blocking_recv() {
                let span = job.span.clone();
                let _entered = span.enter();
                info!(phase = "worker.accepted");
                if let Err(error) = grammar.compile(&job.request.output_format) {
                    send_failure(
                        &job,
                        FailureKind::InvalidRequest,
                        format!("invalid structured output schema: {error}"),
                        Some(output_format_param(job.request.endpoint).to_string()),
                    );
                    continue;
                }
                info!(phase = "generation.started");
                if job.events.blocking_send(WorkerEvent::Started).is_err() {
                    job.cancelled.store(true, Ordering::Release);
                    continue;
                }
                if message_text(&job.request.messages[0]) == "hold" {
                    while !job.cancelled.load(Ordering::Acquire) {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    info!(phase = "generation.cancelled");
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
                    cached_prompt
                        .as_ref()
                        .filter(|cached| prompt.starts_with(cached.as_str()))
                        .map_or(0, |cached| cached.len())
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
                let mut trace_parser = ReasoningTraceParser::default();
                for fragment in generated_text.as_bytes().chunks(4) {
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
                if job.cancelled.load(Ordering::Acquire) {
                    info!(phase = "generation.cancelled");
                    continue;
                }
                cached_prompt = (!has_images && prompt.len() <= 16).then_some(prompt.clone());
                let completion_tokens = if job.request.max_tokens == 1 {
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
                    cached_tokens,
                    finish_reason,
                    stream_include_usage: job.request.stream_include_usage,
                };
                info!(
                    phase = "generation.complete",
                    prompt_tokens = record.prompt_tokens,
                    completion_tokens = record.completion_tokens,
                    cached_tokens = record.cached_tokens,
                    finish_reason = ?record.finish_reason,
                    generated_tool_count = record.tool_calls.len(),
                );
                let _ = job.events.blocking_send(WorkerEvent::Complete(record));
            }
        });
        Self {
            jobs: jobs_tx,
            configured_model_id: model_id.map(Arc::from),
            supports_image_inputs: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    Full,
    Closed,
}

struct QwenWorker {
    provider: Qwen35Provider,
    grammar: GrammarFactory,
    prefix_cache: PrefixCache,
    mtp_k: usize,
}

impl QwenWorker {
    fn load(
        model_path: &Path,
        prefix_cache_max_tokens: usize,
        mtp_k: usize,
        kv_cache_mode: KVCacheMode,
    ) -> Result<Self> {
        validate_mtp_k(mtp_k)?;
        let provider = Qwen35Provider::load(model_path, kv_cache_mode)?;
        ensure!(
            provider.supports_qwen35_tool_calls(),
            "unsupported Qwen3.5 chat template: expected <tool_call>, <function=, and <parameter= literals"
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
            provider.eos_token_id(),
        )?;
        Ok(Self {
            provider,
            grammar,
            prefix_cache: PrefixCache::new(prefix_cache_max_tokens),
            mtp_k,
        })
    }

    fn run(&mut self, mut jobs: mpsc::Receiver<Job>) {
        while let Some(job) = jobs.blocking_recv() {
            self.process(job);
        }
    }

    fn process(&mut self, mut job: Job) {
        let span = job.span.clone();
        let _entered = span.enter();
        info!(phase = "worker.accepted");
        let mut constraint = match self.grammar.compile(&job.request.output_format) {
            Ok(constraint) => constraint,
            Err(error) => {
                send_failure(
                    &job,
                    FailureKind::InvalidRequest,
                    format!("invalid structured output schema: {error}"),
                    Some(output_format_param(job.request.endpoint).to_string()),
                );
                return;
            }
        };
        let enable_thinking = job.request.enable_thinking && constraint.is_none();
        let has_images = !job.request.decoded_images.is_empty();
        if has_images && !self.provider.supports_image_inputs() {
            send_failure(
                &job,
                FailureKind::InvalidRequest,
                "model does not support image inputs".to_string(),
                job.request.image_params.first().cloned(),
            );
            return;
        }
        let mut prepared_images = Vec::with_capacity(job.request.decoded_images.len());
        for image in std::mem::take(&mut job.request.decoded_images) {
            match self
                .provider
                .prepare_image(image.width, image.height, image.rgb)
            {
                Ok(image) => prepared_images.push(image),
                Err(error) => {
                    send_failure(
                        &job,
                        FailureKind::InvalidRequest,
                        format!("failed to preprocess image: {error}"),
                        job.request.image_params.get(prepared_images.len()).cloned(),
                    );
                    return;
                }
            }
        }
        let effective_tools = if job.request.tool_choice == ToolChoice::None {
            &[][..]
        } else {
            job.request.tools.as_slice()
        };
        let tool_enabled = !effective_tools.is_empty();
        let reasoning_effort = job.request.reasoning_effort.map(ReasoningEffort::as_str);
        let multimodal_prefill = if has_images {
            match self.provider.prepare_multimodal_prefill(
                &job.request.messages,
                effective_tools,
                reasoning_effort,
                enable_thinking,
                &prepared_images,
            ) {
                Ok(prefill) => Some(prefill),
                Err(error) => {
                    send_failure(
                        &job,
                        FailureKind::InvalidRequest,
                        format!("failed to prepare image prompt: {error}"),
                        job.request.image_params.first().cloned(),
                    );
                    return;
                }
            }
        } else {
            None
        };
        let prompt_ids = if let Some(prefill) = &multimodal_prefill {
            prefill.prompt_ids.clone()
        } else {
            match self.provider.tokenize_messages(
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
            }
        };
        info!(
            phase = "prompt.prepared",
            prompt_tokens = prompt_ids.len(),
            image_count = prepared_images.len(),
            tool_count = effective_tools.len(),
        );
        if job.events.blocking_send(WorkerEvent::Started).is_err() {
            job.cancelled.store(true, Ordering::Release);
            return;
        }
        if job.cancelled.load(Ordering::Acquire) {
            return;
        }

        let sampling = self.provider.baseline_sampling(
            job.request.temperature,
            job.request.top_p,
            job.request.seed,
        );
        let mtp_available =
            self.provider.has_mtp() && std::env::var_os("QW_BENCH_DISABLE_MTP").is_none();
        let route = qwen_generation_route(mtp_available, has_images, constraint.is_some());
        let mtp_k = self.mtp_k;
        let (provider, cache) = (&mut self.provider, &mut self.prefix_cache);
        let hit = if route_uses_prefix_cache(route) {
            cache.lookup(&prompt_ids)
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
                PromptSnapshot::Mtp(_) => (None, None),
            },
            (QwenGenerationRoute::MtpText, Some(hit)) => match hit.snapshot {
                PromptSnapshot::Mtp(snapshot) => (
                    None,
                    Some(MtpPrefixReuse {
                        snapshot,
                        cached_tokens: hit.token_count,
                    }),
                ),
                PromptSnapshot::Baseline(_) => (None, None),
            },
            _ => (None, None),
        };
        info!(
            phase = "model_generation.started",
            route = ?route,
            mtp_k,
            prefix_cached_tokens = prefix_reuse
                .as_ref()
                .map_or_else(
                    || mtp_prefix_reuse.as_ref().map_or(0, |reuse| reuse.cached_tokens),
                    |reuse| reuse.cached_tokens,
                ),
        );
        let mut trace_parser = ReasoningTraceParser::new(enable_thinking);
        let mut gate = ToolCallGate::default();
        let mut emit_delta = |fragment: &str| {
            if job.cancelled.load(Ordering::Acquire) {
                return false;
            }
            for delta in trace_parser.feed(fragment) {
                let Some(delta) = gate_worker_delta(delta, tool_enabled, &mut gate) else {
                    continue;
                };
                if !send_delta(&job, delta) {
                    return false;
                }
            }
            true
        };
        let generated = match route {
            QwenGenerationRoute::MtpMultimodal => provider.generate_mtp_multimodal_streaming(
                multimodal_prefill.expect("multimodal route requires prepared embeddings"),
                job.request.max_tokens,
                &sampling,
                mtp_k,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                &mut emit_delta,
            ),
            QwenGenerationRoute::MtpText => provider.generate_mtp_streaming(
                &prompt_ids,
                job.request.max_tokens,
                &sampling,
                mtp_k,
                mtp_prefix_reuse,
                true,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                &mut emit_delta,
            ),
            QwenGenerationRoute::BaselineMultimodal => provider.generate_multimodal_streaming(
                multimodal_prefill.expect("multimodal route requires prepared embeddings"),
                job.request.max_tokens,
                &sampling,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                &mut emit_delta,
            ),
            QwenGenerationRoute::BaselineText => provider.generate_baseline_streaming(
                &prompt_ids,
                job.request.max_tokens,
                &sampling,
                prefix_reuse,
                constraint
                    .as_mut()
                    .map(|value| value as &mut dyn mlxcel_core::generate::TokenConstraint),
                true,
                &mut emit_delta,
            ),
        };
        let generated = match generated {
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
        info!(
            phase = "model_generation.complete",
            prompt_tokens = generated.prompt_tokens,
            completion_tokens = generated.completion_tokens,
            cached_tokens = generated.cached_tokens,
            stop_reason = ?generated.finish_outcome,
        );
        for delta in trace_parser.finish() {
            let Some(delta) = gate_worker_delta(delta, tool_enabled, &mut gate) else {
                continue;
            };
            if !send_delta(&job, delta) {
                return;
            }
        }
        if tool_enabled
            && let Some(content) = gate.flush()
            && !send_delta(&job, WorkerDelta::Content(content))
        {
            return;
        }
        if job.cancelled.load(Ordering::Acquire) {
            return;
        }
        let core_finish_reason = match generated.finish_outcome {
            GenerationStopReason::MaxTokens => FinishReason::Length,
            GenerationStopReason::Eos
            | GenerationStopReason::ConstraintAccepted
            | GenerationStopReason::RepetitionLoop => FinishReason::Stop,
            GenerationStopReason::CallbackCancelled => {
                info!(phase = "generation.cancelled");
                return;
            }
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
        if route_uses_prefix_cache(route)
            && let Some(snapshot) = generated.prompt_snapshot
        {
            cache.insert(prompt_ids, snapshot);
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
        info!(
            phase = "generation.complete",
            prompt_tokens = record.prompt_tokens,
            completion_tokens = record.completion_tokens,
            cached_tokens = record.cached_tokens,
            finish_reason = ?record.finish_reason,
            generated_tool_count = record.tool_calls.len(),
        );
        let _ = job.events.blocking_send(WorkerEvent::Complete(record));
    }
}

fn generated_tool_call(
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

    fn parse_fragments(fragments: &[&str]) -> (String, String) {
        let mut parser = ReasoningTraceParser::default();
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut append = |delta| match delta {
            WorkerDelta::Reasoning(fragment) => reasoning.push_str(&fragment),
            WorkerDelta::Content(fragment) => content.push_str(&fragment),
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
    fn mtp_routing_matrix_includes_constrained_capable_requests() {
        for has_mtp in [false, true] {
            for has_images in [false, true] {
                for constrained in [false, true] {
                    for temperature in [0.0f32, 0.7] {
                        let route = qwen_generation_route(has_mtp, has_images, constrained);
                        let expected = match (has_mtp, has_images) {
                            (true, false) => QwenGenerationRoute::MtpText,
                            (true, true) => QwenGenerationRoute::MtpMultimodal,
                            (false, false) => QwenGenerationRoute::BaselineText,
                            (false, true) => QwenGenerationRoute::BaselineMultimodal,
                        };
                        assert_eq!(
                            route, expected,
                            "has_mtp={has_mtp} has_images={has_images} \
                             constrained={constrained} temperature={temperature}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn text_routes_are_prefix_cache_eligible() {
        for route in [
            QwenGenerationRoute::BaselineText,
            QwenGenerationRoute::BaselineMultimodal,
            QwenGenerationRoute::MtpText,
            QwenGenerationRoute::MtpMultimodal,
        ] {
            assert_eq!(
                route_uses_prefix_cache(route),
                matches!(
                    route,
                    QwenGenerationRoute::BaselineText | QwenGenerationRoute::MtpText
                )
            );
        }
        assert_ne!(
            qwen_generation_route(true, true, false),
            QwenGenerationRoute::BaselineText
        );
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
}
