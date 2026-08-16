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
use crate::protocol::{CompletionRequest, Endpoint, OutputFormat};

const JOB_QUEUE_CAPACITY: usize = 8;
const EVENT_QUEUE_CAPACITY: usize = 32;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct Admission {
    pub response_id: String,
    pub message_id: String,
    pub created: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone)]
pub struct CompletionRecord {
    pub admission: Admission,
    pub endpoint: Endpoint,
    pub model: String,
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub finish_reason: FinishReason,
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
                if job.request.messages[0].content == "hold" {
                    while !job.cancelled.load(Ordering::Acquire) {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    continue;
                }
                if job.request.messages[0].content == "fail-after-start" {
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
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>()
                    .join("|");
                let cached_tokens = cached_prompt
                    .as_ref()
                    .filter(|cached| prompt.starts_with(cached.as_str()))
                    .map_or(0, |cached| cached.len());
                let text = match &job.request.output_format {
                    OutputFormat::Text => format!("echo:{prompt}"),
                    OutputFormat::JsonObject => "{\"answer\":1}".to_string(),
                    OutputFormat::JsonSchema { .. } => "{\"answer\":1}".to_string(),
                };
                for delta in text.as_bytes().chunks(4) {
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
                let completion_tokens = if job.request.max_tokens == 1 { 1 } else { text.len() };
                let finish_reason = if job.request.max_tokens == 1 {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                };
                let record = CompletionRecord {
                    admission: job.admission,
                    endpoint: job.request.endpoint,
                    model: model_id_owned.clone(),
                    text,
                    prompt_tokens: prompt.len(),
                    completion_tokens,
                    cached_tokens,
                    finish_reason,
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
        let prompt_ids = match self.provider.tokenize_messages(&job.request.messages) {
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
                if delta.is_empty() {
                    return true;
                }
                if job
                    .events
                    .blocking_send(WorkerEvent::Delta(delta.to_string()))
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
        let finish_reason = match generated.finish_outcome {
            GenerationStopReason::MaxTokens => FinishReason::Length,
            GenerationStopReason::Eos
            | GenerationStopReason::ConstraintAccepted
            | GenerationStopReason::RepetitionLoop => FinishReason::Stop,
            GenerationStopReason::CallbackCancelled => return,
        };
        if !matches!(job.request.output_format, OutputFormat::Text)
            && finish_reason == FinishReason::Stop
            && serde_json::from_str::<Value>(&generated.text).is_err()
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
            text: generated.text,
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.completion_tokens,
            cached_tokens: generated.cached_tokens,
            finish_reason,
        };
        let _ = job.events.blocking_send(WorkerEvent::Complete(record));
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
