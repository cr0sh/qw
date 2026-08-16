use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc as std_mpsc};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use mlxcel_core::generate::{GenerationStopReason, PrefixReuse};
use qw_runtime::Qwen35Provider;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::grammar::GrammarFactory;
use crate::prefix_cache::PrefixCache;
use crate::protocol::{CompletionRequest, Endpoint, OutputFormat, ToolChoice};
use crate::tool_calls::{ToolCallGate, parse_assistant_output};

const JOB_QUEUE_CAPACITY: usize = 8;
const EVENT_QUEUE_CAPACITY: usize = 32;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

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

#[derive(Debug)]
pub enum WorkerEvent {
    Started,
    Delta(String),
    Complete(CompletionRecord),
    Failed(WorkerFailure),
}

struct Job {
    request: CompletionRequest,
    admission: Admission,
    events: mpsc::Sender<WorkerEvent>,
    cancelled: Arc<AtomicBool>,
}

pub struct Submission {
    pub admission: Admission,
    pub events: mpsc::Receiver<WorkerEvent>,
    pub cancelled: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct Engine {
    jobs: mpsc::Sender<Job>,
    model_id: Arc<str>,
}

impl Engine {
    pub fn start_qwen(
        model_path: PathBuf,
        prefix_cache_max_tokens: usize,
    ) -> Result<Self> {
        ensure!(prefix_cache_max_tokens > 0, "prefix cache capacity must be nonzero");
        let model_id = checkpoint_model_id(&model_path)?;
        let (jobs_tx, jobs_rx) = mpsc::channel(JOB_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        thread::Builder::new()
            .name("qw-generation".to_string())
            .spawn(move || {
                let initialized = QwenWorker::load(&model_path, prefix_cache_max_tokens);
                match initialized {
                    Ok(mut worker) => {
                        let _ = ready_tx.send(Ok(()));
                        worker.run(jobs_rx);
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })
            .context("failed to spawn generation thread")?;
        ready_rx
            .recv()
            .context("generation thread exited during startup")??;
        Ok(Self {
            jobs: jobs_tx,
            model_id: model_id.into(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn submit(&self, request: CompletionRequest) -> Result<Submission, SubmitError> {
        let admission = new_admission(request.endpoint);
        let cancelled = Arc::new(AtomicBool::new(false));
        let (events_tx, events_rx) = mpsc::channel(EVENT_QUEUE_CAPACITY);
        let job = Job {
            request,
            admission: admission.clone(),
            events: events_tx,
            cancelled: cancelled.clone(),
        };
        self.jobs.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => SubmitError::Full,
            mpsc::error::TrySendError::Closed(_) => SubmitError::Closed,
        })?;
        Ok(Submission {
            admission,
            events: events_rx,
            cancelled,
        })
    }

    #[cfg(test)]
    pub fn start_fake(model_id: &str, queue_capacity: usize) -> Self {
        let (jobs_tx, mut jobs_rx) = mpsc::channel::<Job>(queue_capacity);
        let model_id_owned = model_id.to_string();
        thread::spawn(move || {
            let grammar = GrammarFactory::single_byte().expect("single-byte grammar factory");
            let mut cached_prompt: Option<String> = None;
            while let Some(job) = jobs_rx.blocking_recv() {
                if let Err(error) = grammar.compile(&job.request.output_format) {
                    send_failure(
                        &job,
                        FailureKind::InvalidRequest,
                        format!("invalid structured output schema: {error}"),
                        Some(output_format_param(job.request.endpoint).to_string()),
                    );
                    continue;
                }
                if job.events.blocking_send(WorkerEvent::Started).is_err() {
                    job.cancelled.store(true, Ordering::Release);
                    continue;
                }
                if job.request.messages[0].content.as_deref() == Some("hold") {
                    while !job.cancelled.load(Ordering::Acquire) {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    continue;
                }
                if job.request.messages[0].content.as_deref() == Some("fail-after-start") {
                    send_failure(
                        &job,
                        FailureKind::Server,
                        "generation failed".to_string(),
                        None,
                    );
                    continue;
                }
                let prompt = job
                    .request
                    .messages
                    .iter()
                    .map(|message| message.content.as_deref().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("|");
                let cached_tokens = cached_prompt
                    .as_ref()
                    .filter(|cached| prompt.starts_with(cached.as_str()))
                    .map_or(0, |cached| cached.len());
                let tool_results = job
                    .request
                    .messages
                    .iter()
                    .filter(|message| message.role == "tool")
                    .filter_map(|message| message.content.as_deref())
                    .collect::<Vec<_>>();
                let latest_user = job
                    .request
                    .messages
                    .iter()
                    .rev()
                    .find(|message| message.role == "user")
                    .and_then(|message| message.content.as_deref());
                if latest_user == Some("call-tool-parallel-violation")
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
                        latest_user,
                        Some("call-tool" | "call-tool-with-preamble")
                    )
                    && job.request.tool_choice == ToolChoice::Auto
                    && !job.request.tools.is_empty();
                let (content, tool_calls, finish_reason) = if fake_tool_turn {
                    let count = if job.request.parallel_tool_calls { 2 } else { 1 };
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
                        (latest_user == Some("call-tool-with-preamble"))
                            .then(|| "I will use tools.".to_string())
                            .unwrap_or_default(),
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
                for delta in content.as_bytes().chunks(4) {
                    if job.cancelled.load(Ordering::Acquire)
                        || job
                            .events
                            .blocking_send(WorkerEvent::Delta(
                                String::from_utf8(delta.to_vec()).expect("ASCII fake output"),
                            ))
                            .is_err()
                    {
                        job.cancelled.store(true, Ordering::Release);
                        break;
                    }
                }
                if job.cancelled.load(Ordering::Acquire) {
                    continue;
                }
                cached_prompt = (prompt.len() <= 16).then_some(prompt.clone());
                let completion_tokens = if job.request.max_tokens == 1 {
                    1
                } else {
                    content.len()
                        + tool_calls
                            .iter()
                            .map(|call| call.arguments.len())
                            .sum::<usize>()
                };
                let record = CompletionRecord {
                    admission: job.admission,
                    endpoint: job.request.endpoint,
                    model: model_id_owned.clone(),
                    content,
                    tool_calls,
                    prompt_tokens: prompt.len(),
                    completion_tokens,
                    cached_tokens,
                    finish_reason,
                    stream_include_usage: job.request.stream_include_usage,
                };
                let _ = job.events.blocking_send(WorkerEvent::Complete(record));
            }
        });
        Self {
            jobs: jobs_tx,
            model_id: model_id.into(),
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
}

impl QwenWorker {
    fn load(model_path: &Path, prefix_cache_max_tokens: usize) -> Result<Self> {
        let provider = Qwen35Provider::load(model_path)?;
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
        })
    }

    fn run(&mut self, mut jobs: mpsc::Receiver<Job>) {
        while let Some(job) = jobs.blocking_recv() {
            self.process(job);
        }
    }

    fn process(&mut self, job: Job) {
        let effective_tools = if job.request.tool_choice == ToolChoice::None {
            &[][..]
        } else {
            job.request.tools.as_slice()
        };
        let tool_enabled = !effective_tools.is_empty();
        let prompt_ids = match self
            .provider
            .tokenize_messages(&job.request.messages, effective_tools)
        {
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
        let (provider, cache) = (&mut self.provider, &mut self.prefix_cache);
        let hit = cache.lookup(&prompt_ids);
        let prefix_reuse = hit.map(|hit| PrefixReuse {
            snapshot: hit.snapshot,
            cached_tokens: hit.token_count,
        });
        let mut gate = ToolCallGate::default();
        let generated = provider.generate_baseline_streaming(
            &prompt_ids,
            job.request.max_tokens,
            &sampling,
            prefix_reuse,
            constraint
                .as_mut()
                .map(|constraint| constraint as &mut dyn mlxcel_core::generate::TokenConstraint),
            true,
            |delta| {
                if job.cancelled.load(Ordering::Acquire) {
                    return false;
                }
                let released = if tool_enabled {
                    gate.feed(delta)
                } else if delta.is_empty() {
                    None
                } else {
                    Some(delta.to_string())
                };
                let Some(released) = released else {
                    return true;
                };
                if job
                    .events
                    .blocking_send(WorkerEvent::Delta(released))
                    .is_err()
                {
                    job.cancelled.store(true, Ordering::Release);
                    false
                } else {
                    true
                }
            },
        );
        let generated = match generated {
            Ok(generated) => generated,
            Err(_) => {
                send_failure(
                    &job,
                    FailureKind::Server,
                    "generation failed".to_string(),
                    None,
                );
                return;
            }
        };
        if job.cancelled.load(Ordering::Acquire) {
            return;
        }
        let core_finish_reason = match generated.finish_outcome {
            GenerationStopReason::MaxTokens => FinishReason::Length,
            GenerationStopReason::Eos
            | GenerationStopReason::ConstraintAccepted
            | GenerationStopReason::RepetitionLoop => FinishReason::Stop,
            GenerationStopReason::CallbackCancelled => return,
        };
        let (content, tool_calls, finish_reason) = if tool_enabled {
            let declared_names = effective_tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>();
            let parsed = match parse_assistant_output(
                &generated.text,
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
            if let Some(released) = gate.flush()
                && job
                    .events
                    .blocking_send(WorkerEvent::Delta(released))
                    .is_err()
            {
                job.cancelled.store(true, Ordering::Release);
                return;
            }
            let has_calls = !parsed.tool_calls.is_empty();
            let tool_calls = parsed
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| {
                    generated_tool_call(
                        &job.admission,
                        index,
                        call.name,
                        call.arguments,
                    )
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
            (
                generated.text.clone(),
                Vec::new(),
                core_finish_reason,
            )
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
        if let Some(snapshot) = generated.prompt_snapshot {
            cache.insert(prompt_ids, snapshot);
        }
        let record = CompletionRecord {
            admission: job.admission,
            endpoint: job.request.endpoint,
            model: job.request.model,
            content,
            tool_calls,
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.completion_tokens,
            cached_tokens: generated.cached_tokens,
            finish_reason,
            stream_include_usage: job.request.stream_include_usage,
        };
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
    let _ = job.events.blocking_send(WorkerEvent::Failed(WorkerFailure {
        kind,
        message,
        param,
    }));
}

fn checkpoint_model_id(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .context("model checkpoint path must have a UTF-8 directory basename")
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
