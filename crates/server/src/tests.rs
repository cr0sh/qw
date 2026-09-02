use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use futures_util::StreamExt;
use qw_prefix_cache::CacheConfig;
use qw_runtime::KVCacheMode;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;

use super::*;

const MODEL: &str = "test-model";

async fn post(app: Router, path: &str, body: Value) -> (StatusCode, axum::http::HeaderMap, String) {
    let response = app
        .oneshot(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router response");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    (
        status,
        headers,
        String::from_utf8(bytes.to_vec()).expect("UTF-8 response"),
    )
}

fn chat_request(prompt: &str) -> Value {
    json!({"model": MODEL, "messages": [{"role": "user", "content": prompt}]})
}

fn responses_request(prompt: &str) -> Value {
    json!({"model": MODEL, "input": prompt})
}

fn chat_tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up weather",
                "parameters": {"type":"object"},
                "strict": true
            }
        },
        {
            "type": "function",
            "function": {
                "name": "time",
                "parameters": {"type":"object"}
            }
        }
    ])
}

fn responses_tools() -> Value {
    json!([
        {
            "type": "function",
            "name": "weather",
            "description": "Look up weather",
            "parameters": {"type":"object"},
            "strict": true
        },
        {
            "type": "function",
            "name": "time",
            "parameters": {"type":"object"}
        }
    ])
}

fn chat_tool_request(prompt: &str) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role":"user","content":prompt}],
        "tools": chat_tools()
    })
}
fn representative_documented_chat_request() -> Value {
    serde_json::from_str(
        r#"{
            "model": "test-model",
            "messages": [
                {
                    "role": "developer",
                    "name": "policy",
                    "content": [{
                        "type": "text",
                        "text": "follow policy",
                        "prompt_cache_breakpoint": {"mode": "explicit"}
                    }]
                },
                {"role": "system", "name": "context", "content": "system context"},
                {
                    "role": "user",
                    "name": "alice",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "UklGRg==", "format": "wav"},
                            "prompt_cache_breakpoint": {"mode": "explicit"}
                        },
                        {
                            "type": "file",
                            "file": {"file_data": "ZmlsZQ==", "filename": "note.txt"}
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "name": "agent",
                    "content": null,
                    "audio": {"id": "audio_1"},
                    "refusal": null,
                    "tool_calls": [{
                        "id": "custom_1",
                        "type": "custom",
                        "custom": {"name": "shell", "input": "status"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "custom_1",
                    "content": [{"type": "text", "text": "ready"}]
                },
                {"role": "function", "name": "legacy", "content": null},
                {"role": "user", "name": "alice", "content": "finish"}
            ],
            "audio": {"format": "wav", "voice": {"id": "voice_1"}},
            "body": {"provider": "opencode"},
            "frequency_penalty": 0.1,
            "function_call": {"name": "legacy"},
            "functions": [{"name": "legacy", "description": null}],
            "logit_bias": {"42": -1},
            "logprobs": true,
            "max_completion_tokens": 32,
            "max_tokens": 16,
            "metadata": {"request_kind": "compatibility"},
            "name": "opencode",
            "modalities": ["text", "audio"],
            "moderation": {"model": "omni-moderation-latest"},
            "n": 2,
            "parallel_tool_calls": false,
            "prediction": {
                "type": "content",
                "content": [{"type": "text", "text": "predicted"}]
            },
            "presence_penalty": 0.2,
            "prompt_cache_key": "cache-key",
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "prompt_cache_retention": "in_memory",
            "reasoning_effort": "high",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "an answer",
                    "strict": false
                }
            },
            "safety_identifier": "safe-user",
            "seed": 7,
            "service_tier": "flex",
            "stop": ["done"],
            "store": false,
            "stream_options": {"include_obfuscation": false, "include_usage": true},
            "temperature": 0.5,
            "tool_choice": {"type": "function", "function": {"name": "legacy"}},
            "tools": [{
                "type": "custom",
                "custom": {
                    "name": "shell",
                    "description": "run a command",
                    "format": {
                        "type": "grammar",
                        "grammar": {"definition": "start: /.+/", "syntax": "lark"}
                    }
                }
            }],
            "top_logprobs": 5,
            "top_p": 0.9,
            "user": "legacy-user",
            "verbosity": "high",
            "web_search_options": {
                "search_context_size": "low",
                "user_location": {
                    "type": "approximate",
                    "approximate": {"country": "US", "timezone": "America/Los_Angeles"}
                }
            }
        }"#,
    )
    .expect("representative request JSON")
}
mod trace_capture {
    use std::cell::RefCell;
    use std::fmt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, Once};

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber, level_filters::LevelFilter};

    static INSTALL: Once = Once::new();
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

    thread_local! {
        static STACK: RefCell<Vec<Id>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn install() {
        INSTALL.call_once(|| {
            tracing::subscriber::set_global_default(CaptureSubscriber)
                .expect("server tests install only one tracing subscriber");
        });
    }

    pub(super) fn clear() {
        LINES.lock().expect("trace lines lock").clear();
    }

    pub(super) fn snapshot() -> Vec<String> {
        LINES.lock().expect("trace lines lock").clone()
    }

    struct CaptureSubscriber;

    impl Subscriber for CaptureSubscriber {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            (metadata.target().starts_with("qw_server")
                || metadata.target().starts_with("qw_runtime"))
                && *metadata.level() <= Level::INFO
        }

        fn max_level_hint(&self) -> Option<LevelFilter> {
            Some(LevelFilter::INFO)
        }

        fn new_span(&self, attributes: &Attributes<'_>) -> Id {
            let id = Id::from_u64(NEXT_ID.fetch_add(1, Ordering::Relaxed));
            let contextual_parent = STACK.with(|stack| stack.borrow().last().cloned());
            let parent = attributes.parent().cloned().or(contextual_parent);
            let mut line = format!(
                "span {} id={} parent={:?}",
                attributes.metadata().name(),
                id.into_u64(),
                parent.as_ref().map(Id::into_u64),
            );
            attributes.record(&mut LineVisitor(&mut line));
            LINES.lock().expect("trace lines lock").push(line);
            id
        }

        fn record(&self, span: &Id, values: &Record<'_>) {
            let mut line = format!("record id={}", span.into_u64());
            values.record(&mut LineVisitor(&mut line));
            LINES.lock().expect("trace lines lock").push(line);
        }

        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

        fn event(&self, event: &Event<'_>) {
            let mut line = format!("event {}", event.metadata().target());
            event.record(&mut LineVisitor(&mut line));
            LINES.lock().expect("trace lines lock").push(line);
        }

        fn enter(&self, span: &Id) {
            STACK.with(|stack| stack.borrow_mut().push(span.clone()));
        }

        fn exit(&self, span: &Id) {
            STACK.with(|stack| {
                let popped = stack.borrow_mut().pop();
                assert_eq!(
                    popped.as_ref(),
                    Some(span),
                    "tracing span stack is balanced"
                );
            });
        }

        fn clone_span(&self, id: &Id) -> Id {
            id.clone()
        }
    }

    struct LineVisitor<'a>(&'a mut String);

    impl Visit for LineVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            use fmt::Write as _;
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }
}

fn responses_tool_request(prompt: &str) -> Value {
    json!({
        "model": MODEL,
        "input": [{"role":"user","content":prompt}],
        "tools": responses_tools()
    })
}

#[derive(Debug)]
struct SseFrame {
    event: Option<String>,
    data: Value,
}

fn parse_sse(body: &str) -> (Vec<SseFrame>, bool) {
    let mut frames = Vec::new();
    let mut done = false;
    for block in body.split("\n\n").filter(|block| !block.is_empty()) {
        let event = block
            .lines()
            .find_map(|line| line.strip_prefix("event: "))
            .map(str::to_string);
        let Some(data) = block.lines().find_map(|line| line.strip_prefix("data: ")) else {
            continue;
        };
        if data == "[DONE]" {
            done = true;
        } else {
            frames.push(SseFrame {
                event,
                data: serde_json::from_str(data).expect("SSE JSON"),
            });
        }
    }
    (frames, done)
}

#[derive(Debug)]
struct ResponsesSseMeasurement {
    completed_response: Value,
    assistant_text: String,
    ttft: Duration,
    terminal_tail: Duration,
}

async fn read_responses_sse(app: Router, body: Value) -> Result<ResponsesSseMeasurement, String> {
    let started = Instant::now();
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .map_err(|error| format!("request: {error}"))?,
        )
        .await
        .map_err(|error| format!("router response: {error}"))?;
    if response.status() != StatusCode::OK {
        return Err(format!("unexpected HTTP status {}", response.status()));
    }

    let mut stream = response.into_body().into_data_stream();
    let mut buffer = Vec::new();
    let mut assistant_text = String::new();
    let mut first_delta_at = None;
    let mut last_delta_at = None;
    let mut completed_response = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("SSE body stream: {error}"))?;
        buffer.extend_from_slice(&chunk);
        while let Some(separator) = buffer.windows(2).position(|window| window == b"\n\n") {
            let block = buffer.drain(..separator).collect::<Vec<_>>();
            buffer.drain(..2);
            let block = String::from_utf8(block).map_err(|error| format!("SSE UTF-8: {error}"))?;
            let event = block
                .lines()
                .find_map(|line| line.strip_prefix("event: "))
                .ok_or_else(|| "SSE event is missing its event name".to_string())?;
            let data = block
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .ok_or_else(|| format!("SSE event {event} is missing data"))?;
            if data == "[DONE]" {
                continue;
            }
            let value: Value = serde_json::from_str(data)
                .map_err(|error| format!("SSE event {event} JSON: {error}"))?;
            match event {
                "response.output_text.delta" => {
                    let delta = value["delta"]
                        .as_str()
                        .ok_or_else(|| "output delta is not a string".to_string())?;
                    let now = Instant::now();
                    first_delta_at.get_or_insert(now);
                    last_delta_at = Some(now);
                    assistant_text.push_str(delta);
                }
                "response.completed" => {
                    completed_response = Some(
                        value
                            .get("response")
                            .cloned()
                            .ok_or_else(|| "response.completed lacks response".to_string())?,
                    );
                }
                _ => {}
            }
        }
    }
    if !buffer.is_empty() {
        return Err("SSE body ended with an incomplete event".to_string());
    }
    let eof = Instant::now();
    let first_delta_at = first_delta_at.ok_or_else(|| {
        "Responses SSE ended without a response.output_text.delta event".to_string()
    })?;
    let last_delta_at = last_delta_at.expect("first and last delta timestamps are paired");
    let completed_response = completed_response
        .ok_or_else(|| "Responses SSE ended without response.completed".to_string())?;
    Ok(ResponsesSseMeasurement {
        completed_response,
        assistant_text,
        ttft: first_delta_at.duration_since(started),
        terminal_tail: eof.duration_since(last_delta_at),
    })
}

#[tokio::test]
#[ignore = "requires the complete pinned Qwen3.8 27B GGUF pair"]
async fn real_responses_sse_latency_stays_bounded_across_cold_fork_and_continuation() {
    let engine = Engine::start_qwen(
        Some(MODEL.to_string()),
        CacheConfig {
            directory: None,
            ..CacheConfig::default()
        },
        true,
        3,
        KVCacheMode::Turbo4,
        #[cfg(feature = "specprefill")]
        SpecPrefillPolicyConfig::default(),
    )
    .expect("start real Qwen engine");
    let app = router(engine);
    let shared_prefix = || {
        vec![
            json!({
                "role": "system",
                "content": "You are a deterministic benchmark assistant. Follow the user's visible-answer instruction exactly."
            }),
            json!({
                "role": "user",
                "content": "A deterministic cache benchmark paragraph. ".repeat(128)
            }),
            json!({
                "role": "assistant",
                "content": "The shared benchmark prefix is acknowledged."
            }),
        ]
    };
    let request = |input: Vec<Value>| {
        json!({
            "model": MODEL,
            "input": input,
            "stream": true,
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 128
        })
    };
    let mut cold_input = shared_prefix();
    cold_input.push(json!({
        "role": "user",
        "content": "Output exactly the single plain word COLD and then stop. Do not explain, reason, or output any other text."
    }));
    let cold = tokio::time::timeout(
        Duration::from_secs(30),
        read_responses_sse(app.clone(), request(cold_input)),
    )
    .await
    .expect("cold Responses SSE exceeded the 30-second hang guard")
    .expect("cold Responses SSE");
    let mut fork_input = shared_prefix();

    fork_input.push(json!({
        "role": "user",
        "content": "Output exactly the single plain word FORK and then stop. Do not explain, reason, or output any other text."
    }));
    let fork = tokio::time::timeout(
        Duration::from_secs(30),
        read_responses_sse(app.clone(), request(fork_input)),
    )
    .await
    .expect("fork Responses SSE exceeded the 30-second hang guard")
    .expect("fork Responses SSE");

    let mut continuation_input = shared_prefix();
    continuation_input.push(json!({
        "role": "assistant",
        "content": cold.assistant_text.clone()
    }));
    continuation_input.push(json!({
        "role": "user",
        "content": "Output exactly the single plain word CONTINUATION and then stop. Do not explain, reason, or output any other text."
    }));
    let continuation = tokio::time::timeout(
        Duration::from_secs(30),
        read_responses_sse(app, request(continuation_input)),
    )
    .await
    .expect("continuation Responses SSE exceeded the 30-second hang guard")
    .expect("continuation Responses SSE");

    let cached_tokens = |measurement: &ResponsesSseMeasurement| {
        measurement.completed_response["usage"]["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .expect("Responses completion cached token count")
    };
    let cold_cached = cached_tokens(&cold);
    let fork_cached = cached_tokens(&fork);
    let continuation_cached = cached_tokens(&continuation);
    let diagnostics = format!(
        "cold(ttft={:?}, tail={:?}, cached={cold_cached}), \
         fork(ttft={:?}, tail={:?}, cached={fork_cached}), \
         continuation(ttft={:?}, tail={:?}, cached={continuation_cached})",
        cold.ttft,
        cold.terminal_tail,
        fork.ttft,
        fork.terminal_tail,
        continuation.ttft,
        continuation.terminal_tail,
    );
    assert!(cold.terminal_tail < Duration::from_secs(1), "{diagnostics}");
    assert!(fork.terminal_tail < Duration::from_secs(1), "{diagnostics}");
    assert!(
        continuation.terminal_tail < Duration::from_secs(1),
        "{diagnostics}"
    );
    assert!(fork_cached > 0, "{diagnostics}");
    assert!(continuation_cached > 0, "{diagnostics}");
    assert!(fork.ttft < cold.ttft, "{diagnostics}");
    assert!(continuation.ttft < cold.ttft, "{diagnostics}");
}

fn responses_replay_call(item: &Value) -> Value {
    json!({
        "type": "function_call",
        "id": item["id"],
        "call_id": item["call_id"],
        "name": item["name"],
        "arguments": item["arguments"]
    })
}

#[tokio::test]
async fn buffered_chat_completion_has_openai_shape_and_usage() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(app, "/v1/chat/completions", chat_request("hello")).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).expect("JSON response");
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(value["choices"][0]["message"]["role"], "assistant");
    assert_eq!(value["choices"][0]["message"]["content"], "echo:hello");
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
    assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn streamed_chat_emits_role_content_terminal_usage_and_done() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = chat_request("hello");
    request["stream"] = Value::Bool(true);
    let (status, headers, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .expect("content type")
            .starts_with("text/event-stream")
    );
    assert!(body.contains("\"role\":\"assistant\""), "{body}");
    assert!(body.contains("\"content\":"), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
    assert!(body.contains("\"usage\":"), "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
}

fn terminal_chat_record() -> CompletionRecord {
    CompletionRecord {
        admission: Admission {
            response_id: "chatcmpl-terminal-eof".to_string(),
            message_id: "msg-terminal-eof".to_string(),
            created: 0,
        },
        endpoint: Endpoint::Chat,
        model: MODEL.to_string(),
        content: "done".to_string(),
        reasoning_content: String::new(),
        tool_calls: Vec::new(),
        prompt_tokens: 3,
        completion_tokens: 1,
        cached_tokens: 0,
        finish_reason: FinishReason::Stop,
        stream_include_usage: false,
    }
}

#[tokio::test]
async fn completion_acknowledgment_waits_for_terminal_events_to_drain() {
    let (events_tx, events_rx) = mpsc::channel(1);
    let (acknowledged, mut acknowledged_rx) = oneshot::channel();
    let record = terminal_chat_record();
    let admission = record.admission.clone();
    events_tx
        .send(WorkerEvent::Complete {
            record,
            acknowledged: Some(acknowledged),
        })
        .await
        .expect("send completion");
    let mut state = SseState::new(
        Endpoint::Chat,
        admission,
        MODEL.to_string(),
        events_rx,
        Arc::new(AtomicBool::new(false)),
        tracing::info_span!("stream_terminal_eof_test"),
    );

    assert!(state.next_event().await.is_some(), "initial event");
    assert!(state.next_event().await.is_some(), "terminal event");
    assert!(
        matches!(
            acknowledged_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ),
        "cache work must not begin before all terminal events are consumed"
    );
    assert!(state.next_event().await.is_some(), "done event");
    assert!(
        matches!(
            acknowledged_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ),
        "cache work must not begin before the stream reaches EOF"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), state.next_event())
            .await
            .expect("terminal EOF should not wait for channel closure")
            .is_none()
    );
    assert_eq!(
        acknowledged_rx.try_recv(),
        Ok(()),
        "cache work begins at the EOF boundary"
    );
}

#[tokio::test]
async fn dropping_stream_releases_pending_completion_acknowledgment() {
    let (events_tx, events_rx) = mpsc::channel(1);
    let (acknowledged, acknowledged_rx) = oneshot::channel();
    let record = terminal_chat_record();
    let admission = record.admission.clone();
    events_tx
        .send(WorkerEvent::Complete {
            record,
            acknowledged: Some(acknowledged),
        })
        .await
        .expect("send completion");
    let mut state = SseState::new(
        Endpoint::Chat,
        admission,
        MODEL.to_string(),
        events_rx,
        Arc::new(AtomicBool::new(false)),
        tracing::info_span!("stream_drop_acknowledgment_test"),
    );

    assert!(state.next_event().await.is_some(), "initial event");
    assert!(state.next_event().await.is_some(), "terminal event");
    drop(state);

    assert!(
        tokio::time::timeout(Duration::from_millis(100), acknowledged_rx)
            .await
            .expect("dropping the stream should release the worker")
            .is_err(),
        "disconnect must drop rather than send the pending acknowledgment"
    );
}

#[tokio::test]
async fn interrupted_response_resumes_once_without_replaying_deltas() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut interrupted = chat_request("resume-interrupt");
    interrupted["stream"] = json!(true);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", interrupted.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, done) = parse_sse(&body);
    assert!(!done);
    let response_id = frames
        .iter()
        .find_map(|frame| frame.data["id"].as_str())
        .expect("partial stream response ID")
        .to_string();
    let partial = frames
        .iter()
        .filter_map(|frame| frame.data["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(partial, "echo:resume-");

    let mut mismatch = interrupted.clone();
    mismatch["stream"] = json!(false);
    mismatch["temperature"] = json!(0.5);
    mismatch["resume_response_id"] = json!(response_id);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", mismatch).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).expect("mismatch JSON")["error"]["code"],
        "resume_mismatch"
    );

    let mut unsupported = interrupted.clone();
    unsupported["stream"] = json!(false);
    unsupported["response_format"] = json!({"type": "json_object"});
    unsupported["resume_response_id"] = json!(response_id);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", unsupported).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).expect("unsupported JSON")["error"]["code"],
        "resume_unsupported"
    );

    let mut resumed = interrupted;
    resumed["stream"] = json!(false);
    resumed["resume_response_id"] = json!(response_id);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", resumed.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("resumed response JSON");
    assert_eq!(response["id"], response_id);
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "echo:resume-interrupt"
    );
    assert!(
        response["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .expect("cached token count")
            > 0
    );

    let (status, _, body) = post(app, "/v1/chat/completions", resumed).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).expect("not found JSON")["error"]["code"],
        "resume_not_found"
    );
}

#[tokio::test]
async fn responses_resume_preserves_response_and_message_ids() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut interrupted = responses_request("resume-interrupt");
    interrupted["stream"] = json!(true);
    let (status, _, body) = post(app.clone(), "/v1/responses", interrupted.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, done) = parse_sse(&body);
    assert!(!done);
    let response_id = frames
        .iter()
        .find_map(|frame| frame.data["response"]["id"].as_str())
        .expect("partial Responses response ID")
        .to_string();
    let message_id = frames
        .iter()
        .find_map(|frame| frame.data["item"]["id"].as_str())
        .expect("partial Responses message ID")
        .to_string();

    let mut resumed = interrupted;
    resumed["stream"] = json!(false);
    resumed["resume_response_id"] = json!(response_id);
    let (status, _, body) = post(app, "/v1/responses", resumed).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("resumed Responses JSON");
    assert_eq!(response["id"], response_id);
    assert_eq!(response["output"][0]["id"], message_id);
    assert_eq!(
        response["output"][0]["content"][0]["text"],
        "echo:resume-interrupt"
    );
}

#[tokio::test]
async fn buffered_chat_separates_reasoning_from_visible_content() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(app, "/v1/chat/completions", chat_request("reasoning")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("chat JSON");
    let message = &value["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "fake reasoning");
    assert_eq!(message["content"], "echo:reasoning");
    assert!(!body.contains("<think"));
    assert!(!body.contains("</think>"));
}

#[tokio::test]
async fn streamed_chat_uses_reasoning_and_content_deltas_without_markers() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = chat_request("reasoning");
    request["stream"] = json!(true);
    let (status, _, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, done) = parse_sse(&body);
    assert!(done);
    let reasoning = frames
        .iter()
        .filter_map(|frame| frame.data["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect::<String>();
    let content = frames
        .iter()
        .filter_map(|frame| frame.data["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(reasoning, "fake reasoning");
    assert_eq!(content, "echo:reasoning");
    assert!(!body.contains("<think"));
    assert!(!body.contains("</think>"));
}

#[tokio::test]
async fn responses_suppresses_reasoning_buffered_and_streamed() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) =
        post(app.clone(), "/v1/responses", responses_request("reasoning")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("responses JSON");
    assert_eq!(value["output"][0]["content"][0]["text"], "echo:reasoning");
    assert!(!body.contains("fake reasoning"));
    assert!(!body.contains("reasoning_content"));
    assert!(!body.contains("<think"));

    let mut request = responses_request("reasoning");
    request["stream"] = json!(true);
    let (status, _, body) = post(app, "/v1/responses", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, _) = parse_sse(&body);
    let content = frames
        .iter()
        .filter(|frame| frame.event.as_deref() == Some("response.output_text.delta"))
        .filter_map(|frame| frame.data["delta"].as_str())
        .collect::<String>();
    assert_eq!(content, "echo:reasoning");
    assert!(!body.contains("fake reasoning"));
    assert!(!body.contains("reasoning_content"));
    assert!(!body.contains("<think"));
}
#[tokio::test]
async fn buffered_responses_completion_has_output_and_cached_usage() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(app, "/v1/responses", responses_request("hello")).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).expect("JSON response");
    assert_eq!(value["object"], "response");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["output"][0]["content"][0]["text"], "echo:hello");
    assert_eq!(value["usage"]["input_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn streamed_responses_events_have_exact_order_and_monotonic_sequences() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = responses_request("hello");
    request["stream"] = Value::Bool(true);
    let (status, _, body) = post(app, "/v1/responses", request).await;
    assert_eq!(status, StatusCode::OK);
    let event_names: Vec<&str> = body
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(
        event_names,
        [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let sequences: Vec<u64> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|value| value["sequence_number"].as_u64())
        .collect();
    assert_eq!(sequences, (0..sequences.len() as u64).collect::<Vec<_>>());
    for frame in body
        .split("\n\n")
        .filter(|frame| frame.starts_with("event: "))
    {
        let mut lines = frame.lines();
        let name = lines
            .next()
            .and_then(|line| line.strip_prefix("event: "))
            .expect("event name");
        let data = lines
            .find_map(|line| line.strip_prefix("data: "))
            .expect("event data");
        let value: Value = serde_json::from_str(data).expect("event JSON");
        assert_eq!(value["type"], name);
    }
}

#[tokio::test]
async fn unconfigured_model_id_routes_arbitrary_models_and_preserves_response_identity() {
    let app = router(Engine::start_fake(None, 8));

    let mut buffered_request = chat_request("first");
    buffered_request["model"] = json!("arbitrary-one");
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", buffered_request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("buffered response JSON");
    assert_eq!(response["model"], "arbitrary-one");

    let mut streamed_request = chat_request("second");
    streamed_request["model"] = json!("org/arbitrary-two");
    streamed_request["stream"] = json!(true);
    let (status, _, body) = post(app, "/v1/chat/completions", streamed_request).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (frames, done) = parse_sse(&body);
    assert!(done);
    let response_models = frames
        .iter()
        .filter_map(|frame| frame.data["model"].as_str())
        .collect::<Vec<_>>();
    assert!(!response_models.is_empty());
    assert!(
        response_models
            .iter()
            .all(|model| *model == "org/arbitrary-two")
    );
}
#[test]
fn strict_protocols_accept_top_level_resume_response_id() {
    let mut chat = chat_request("hello");
    chat["resume_response_id"] = json!("chatcmpl-original");
    assert_eq!(
        protocol::parse_chat(chat)
            .expect("chat resume field")
            .resume_response_id
            .as_deref(),
        Some("chatcmpl-original")
    );

    let mut responses = responses_request("hello");
    responses["resume_response_id"] = json!("resp_original");
    assert_eq!(
        protocol::parse_responses(responses)
            .expect("responses resume field")
            .resume_response_id
            .as_deref(),
        Some("resp_original")
    );
}

#[tokio::test]
async fn configured_model_id_accepts_exact_match_and_rejects_mismatch() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(
        app.clone(),
        "/v1/chat/completions",
        chat_request("accepted"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("success JSON");
    assert_eq!(response["model"], MODEL);

    let mut request = chat_request("rejected");
    request["model"] = json!("other");
    let (status, _, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let value: Value = serde_json::from_str(&body).expect("error JSON");
    assert_eq!(
        value,
        json!({
            "error": {
                "message": "model \"other\" was not found",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        })
    );
}

#[tokio::test]
async fn malformed_fields_are_rejected() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    for request in [
        json!({"model": MODEL, "messages": []}),
        json!({"model": MODEL, "messages": [{"role":"user","content":[]}]}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "max_tokens": 0}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "unknown_control": true}),
    ] {
        let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    for request in [
        json!({"model": MODEL, "input": []}),
        json!({"model": MODEL, "input": [{"role":"user","content":[]}]}),
        json!({"model": MODEL, "input": "x", "store": false}),
        json!({"model": MODEL, "input": "x", "conversation": "c"}),
        json!({"model": MODEL, "input": "x", "include": []}),
    ] {
        let (status, _, body) = post(app.clone(), "/v1/responses", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}
#[test]
fn chat_protocol_accepts_message_names_and_documented_nested_variants() {
    let parsed = protocol::parse_chat(representative_documented_chat_request())
        .expect("representative documented request");
    assert_eq!(parsed.messages[0].name.as_deref(), Some("policy"));
    assert_eq!(parsed.messages[1].name.as_deref(), Some("context"));
    assert_eq!(parsed.messages[2].name.as_deref(), Some("alice"));
    assert_eq!(parsed.messages[3].name.as_deref(), Some("agent"));
    assert_eq!(parsed.messages[5].name.as_deref(), Some("legacy"));
    assert_eq!(
        parsed.reasoning_effort,
        Some(protocol::ReasoningEffort::High)
    );
    assert_eq!(parsed.max_tokens, 32);
    assert_eq!(parsed.tools.len(), 1);

    let messages = serde_json::to_value(&parsed.messages).expect("serialize normalized messages");
    assert_eq!(messages[0]["name"], "policy");
    assert_eq!(messages[2]["content"][1]["type"], "input_audio");
    assert_eq!(messages[2]["content"][2]["type"], "file");
    assert_eq!(messages[3]["tool_calls"][0]["type"], "custom");
    assert_eq!(messages[4]["content"][0]["type"], "text");
}

#[tokio::test]
async fn documented_optional_chat_inputs_reach_the_handler() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(
        app,
        "/v1/chat/completions",
        representative_documented_chat_request(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("chat response JSON");
    assert_eq!(response["object"], "chat.completion");
}
#[tokio::test]
async fn structured_tracing_covers_request_stream_error_and_cancellation_without_bodies() {
    trace_capture::install();
    trace_capture::clear();
    engine::log_generation_metrics(
        "chatcmpl-metric-test",
        10,
        3,
        4,
        Duration::ZERO,
        Duration::ZERO,
    );

    const SECRET_PROMPT: &str = "trace-secret-prompt-7b5c";
    const SECRET_ARGUMENTS: &str = "trace-secret-tool-arguments-29af";
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let request = json!({
        "model": MODEL,
        "tools": chat_tools(),
        "messages": [
            {"role": "user", "content": SECRET_PROMPT},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "trace_call",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": format!(r#"{{"value":"{SECRET_ARGUMENTS}"}}"#)
                    }
                }]
            },
            {"role": "tool", "tool_call_id": "trace_call", "content": "ready"},
            {"role": "user", "content": "finish"}
        ]
    });
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let mut streaming = chat_request("stream trace");
    streaming["stream"] = json!(true);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", streaming).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let mut failing = chat_request("fail-after-start");
    failing["stream"] = json!(true);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", failing).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("\"type\":\"server_error\""), "{body}");

    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "hold"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("stream response");
    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
    tokio::time::sleep(Duration::from_millis(30)).await;

    let traces = trace_capture::snapshot().join("\n");
    assert!(traces.contains("span server.request"), "{traces}");
    assert!(
        traces
            .lines()
            .any(|line| line.contains("span generation") && line.contains("parent=Some")),
        "{traces}"
    );
    assert!(traces.contains("response.streaming_failed"), "{traces}");
    for phase in [
        "request.validation_complete",
        "dispatch.enqueued",
        "model_generation.complete",
        "generation.complete",
        "response.buffered_complete",
        "response.streaming_admitted",
        "response.streaming_complete",
    ] {
        assert!(!traces.contains(phase), "unexpected INFO {phase}: {traces}");
    }
    let metric_lines = traces
        .lines()
        .filter(|line| {
            line.contains("event=\"prefill.complete\"")
                || line.contains("event=\"decode.complete\"")
        })
        .collect::<Vec<_>>();
    assert_eq!(metric_lines.len(), 2, "{traces}");
    assert!(
        metric_lines[0].contains("event=\"prefill.complete\""),
        "{traces}"
    );
    assert!(
        metric_lines[0].contains("chat_id=\"chatcmpl-metric-test\""),
        "{traces}"
    );
    assert!(metric_lines[0].contains("tps=0.0"), "{traces}");
    assert!(metric_lines[0].contains("total_tokens=10"), "{traces}");
    assert!(metric_lines[0].contains("prefilled_tokens=6"), "{traces}");
    assert!(
        metric_lines[0].contains("prefix_reused_tokens=4"),
        "{traces}"
    );
    assert!(
        metric_lines[1].contains("event=\"decode.complete\""),
        "{traces}"
    );
    assert!(
        metric_lines[1].contains("chat_id=\"chatcmpl-metric-test\""),
        "{traces}"
    );
    assert!(metric_lines[1].contains("tps=0.0"), "{traces}");
    assert!(metric_lines[1].contains("total_tokens=13"), "{traces}");
    assert!(metric_lines[1].contains("decoded_tokens=3"), "{traces}");
    assert!(
        metric_lines[1].contains("prefix_reused_tokens=4"),
        "{traces}"
    );
    assert!(!traces.contains(SECRET_PROMPT), "{traces}");
    assert!(!traces.contains(SECRET_ARGUMENTS), "{traces}");
    assert!(
        !traces.contains("response.stream_worker_closed"),
        "{traces}"
    );
}

#[test]
fn chat_protocol_accepts_documented_reasoning_efforts() {
    for (value, expected) in [
        ("none", protocol::ReasoningEffort::None),
        ("minimal", protocol::ReasoningEffort::Minimal),
        ("low", protocol::ReasoningEffort::Low),
        ("medium", protocol::ReasoningEffort::Medium),
        ("high", protocol::ReasoningEffort::High),
        ("xhigh", protocol::ReasoningEffort::XHigh),
        ("max", protocol::ReasoningEffort::Max),
    ] {
        let mut request = chat_request("hello");
        request["reasoning_effort"] = json!(value);
        let parsed = protocol::parse_chat(request).expect("documented reasoning effort");
        assert_eq!(parsed.reasoning_effort, Some(expected));
        assert_eq!(
            parsed.reasoning_effort.map(|effort| effort.as_str()),
            Some(value)
        );
    }
    for value in [None, Some(Value::Null)] {
        let mut request = chat_request("hello");
        if let Some(value) = value {
            request["reasoning_effort"] = value;
        }
        assert_eq!(
            protocol::parse_chat(request)
                .expect("omitted or null effort")
                .reasoning_effort,
            None
        );
    }
}

#[test]
fn chat_protocol_accepts_nested_reasoning_effort_with_top_level_precedence() {
    let mut request = chat_request("hello");
    request["chat_template_kwargs"] = json!({
        "enable_thinking": true,
        "preserve_thinking": true,
        "reasoning_effort": "medium"
    });
    let parsed = protocol::parse_chat(request).expect("nested reasoning effort");
    assert_eq!(
        parsed.reasoning_effort,
        Some(protocol::ReasoningEffort::Medium)
    );
    assert!(parsed.enable_thinking);

    let mut request = chat_request("hello");
    request["reasoning_effort"] = json!("high");
    request["chat_template_kwargs"] = json!({"reasoning_effort": "medium"});
    let parsed = protocol::parse_chat(request).expect("top-level reasoning effort");
    assert_eq!(
        parsed.reasoning_effort,
        Some(protocol::ReasoningEffort::High)
    );

    let mut request = chat_request("hello");
    request["chat_template_kwargs"] = json!({"reasoning_effort": null});
    let parsed = protocol::parse_chat(request).expect("null nested reasoning effort");
    assert_eq!(parsed.reasoning_effort, None);
}

#[test]
fn chat_protocol_accepts_qwen_thinking_extensions() {
    let mut request = chat_request("hello");
    request["preserve_thinking"] = json!(true);
    request["chat_template_kwargs"] = json!({"preserve_thinking": true, "enable_thinking": false});
    let parsed = protocol::parse_chat(request).expect("Qwen thinking extensions");
    assert!(!parsed.enable_thinking);
}

#[test]
fn chat_protocol_accepts_opencode_thinking_options() {
    let mut request = chat_request("hello");
    request["thinking"] = json!({"type": "enabled", "budgetTokens": 15999});
    request["mcp_timeout"] = json!(60);
    let parsed = protocol::parse_chat(request).expect("opencode thinking options");
    assert!(parsed.enable_thinking);

    let mut request = chat_request("hello");
    request["thinking"] = json!({"type": "disabled"});
    let parsed = protocol::parse_chat(request).expect("disabled opencode thinking");
    assert!(!parsed.enable_thinking);
}

#[test]
fn chat_protocol_rejects_invalid_reasoning_efforts() {
    for value in [json!("unknown"), json!(""), json!(1)] {
        let mut request = chat_request("hello");
        request["reasoning_effort"] = value;
        let error = protocol::parse_chat(request).expect_err("invalid reasoning effort");
        assert_eq!(error.param.as_deref(), Some("reasoning_effort"));
    }
}

#[test]
fn chat_protocol_rejects_invalid_nested_reasoning_efforts_at_exact_param() {
    for (value, message) in [
        (
            json!("unknown"),
            "chat_template_kwargs.reasoning_effort is not a documented value",
        ),
        (
            json!(1),
            "chat_template_kwargs.reasoning_effort must be a string or null",
        ),
    ] {
        let mut request = chat_request("hello");
        request["chat_template_kwargs"] = json!({"reasoning_effort": value});
        let error = protocol::parse_chat(request).expect_err("invalid nested reasoning effort");
        assert_eq!(
            error.param.as_deref(),
            Some("chat_template_kwargs.reasoning_effort")
        );
        assert_eq!(error.message, message);
    }
}

#[tokio::test]
async fn documented_reasoning_efforts_reach_the_handler() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    for effort in ["none", "minimal", "high", "max"] {
        let mut request = chat_request("hello");
        request["reasoning_effort"] = json!(effort);
        let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
}

#[tokio::test]
async fn nested_reasoning_effort_reaches_the_handler() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = chat_request("hello");
    request["chat_template_kwargs"] = json!({
        "enable_thinking": true,
        "preserve_thinking": true,
        "reasoning_effort": "medium"
    });
    let (status, _, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
#[test]
fn chat_protocol_replays_assistant_reasoning_content() {
    let parsed = protocol::parse_chat(json!({
        "model": MODEL,
        "messages": [
            {"role":"user","content":"question"},
            {
                "role":"assistant",
                "reasoning_content":"private trace",
                "content":"answer"
            },
            {"role":"user","content":"follow-up"}
        ]
    }))
    .expect("assistant reasoning replay");
    assert_eq!(
        parsed.messages[1].reasoning_content.as_deref(),
        Some("private trace")
    );
    assert_eq!(parsed.messages[1].text_content(), Some("answer"));

    let parsed = protocol::parse_chat(json!({
        "model": MODEL,
        "messages": [
            {"role":"user","content":"question"},
            {"role":"assistant","reasoning_content":null,"content":"answer"}
        ]
    }))
    .expect("nullable reasoning content");
    assert_eq!(parsed.messages[1].reasoning_content, None);
}
#[tokio::test]
async fn json_object_and_strict_schema_are_generated_under_constraints() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut object = chat_request("json");
    object["response_format"] = json!({"type":"json_object"});
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", object).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("response JSON");
    let generated = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("generated text");
    assert!(serde_json::from_str::<Value>(generated).is_ok());

    let mut schema = responses_request("schema");
    schema["text"] = json!({"format": {
        "type":"json_schema",
        "name":"answer",
        "strict":true,
        "schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}
    }});
    let (status, _, body) = post(app, "/v1/responses", schema).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn every_strict_schema_rejection_class_returns_400() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let schemas = [
        json!({"$ref":"https://example.com/schema.json"}),
        json!({"oneOf":[{"type":"string"},{"type":"string"}]}),
        json!({"type":"object","unknownKeyword":true}),
        json!({"type":"string","format":"credit-card"}),
        json!({"type":"string","pattern":"a(?=b)"}),
        json!({"type":"object","additionalProperties":{"type":"string"}}),
        json!({"type":"object","patternProperties":{".*":{"type":"string"}}}),
    ];
    for schema in schemas {
        let mut request = chat_request("schema");
        request["response_format"] = json!({
            "type":"json_schema",
            "json_schema":{"name":"bad","strict":true,"schema":schema}
        });
        let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }

    let mut non_strict = responses_request("schema");
    non_strict["text"] =
        json!({"format":{"type":"json_schema","name":"bad","strict":false,"schema":{}}});
    let (status, _, _) = post(app, "/v1/responses", non_strict).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn max_token_fields_map_to_length_and_incomplete() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut chat = chat_request("short");
    chat["max_tokens"] = json!(1);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", chat).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).expect("chat JSON");
    assert_eq!(value["choices"][0]["finish_reason"], "length");

    let mut responses = responses_request("short");
    responses["max_output_tokens"] = json!(1);
    let (status, _, body) = post(app, "/v1/responses", responses).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).expect("responses JSON");
    assert_eq!(value["status"], "incomplete");
    assert_eq!(value["incomplete_details"]["reason"], "max_output_tokens");
}

#[tokio::test]
async fn repeated_prefix_reports_cached_tokens_and_matches_cold_output() {
    let warm_app = router(Engine::start_fake(Some(MODEL), 8));
    let _ = post(
        warm_app.clone(),
        "/v1/chat/completions",
        chat_request("abc"),
    )
    .await;
    let (status, _, warm_body) =
        post(warm_app, "/v1/chat/completions", chat_request("abcdef")).await;
    assert_eq!(status, StatusCode::OK);
    let warm: Value = serde_json::from_str(&warm_body).expect("warm JSON");
    assert!(
        warm["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .expect("cached tokens")
            > 0
    );

    let cold_app = router(Engine::start_fake(Some(MODEL), 8));
    let (_, _, cold_body) = post(cold_app, "/v1/chat/completions", chat_request("abcdef")).await;
    let cold: Value = serde_json::from_str(&cold_body).expect("cold JSON");
    assert_eq!(
        warm["choices"][0]["message"]["content"],
        cold["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn divergent_multi_turn_chat_reuses_exact_common_prefix_and_matches_cold_output() {
    let warm_app = router(Engine::start_fake(Some(MODEL), 8));
    let warmed = json!({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"}
        ]
    });
    let divergent = json!({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "x"}
        ]
    });
    let _ = post(warm_app.clone(), "/v1/chat/completions", warmed).await;
    let (status, _, warm_body) = post(warm_app, "/v1/chat/completions", divergent.clone()).await;
    assert_eq!(status, StatusCode::OK, "{warm_body}");
    let warm: Value = serde_json::from_str(&warm_body).expect("warm JSON");
    assert_eq!(
        warm["usage"]["prompt_tokens_details"]["cached_tokens"], 6,
        "a|b|c| is the exact common prompt prefix"
    );

    let cold_app = router(Engine::start_fake(Some(MODEL), 8));
    let (_, _, cold_body) = post(cold_app, "/v1/chat/completions", divergent).await;
    let cold: Value = serde_json::from_str(&cold_body).expect("cold JSON");
    assert_eq!(
        warm["choices"][0]["message"]["content"],
        cold["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn five_sequential_turns_keep_reusing_the_latest_complete_prefix() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut messages = vec![json!({"role": "user", "content": "a"})];
    let expected_cached = [0_u64, 1, 5, 9, 13];

    for (turn, expected) in expected_cached.into_iter().enumerate() {
        let request = json!({
            "model": MODEL,
            "messages": messages,
        });
        let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::OK, "turn {}: {body}", turn + 1);
        let value: Value = serde_json::from_str(&body).expect("chat JSON");
        assert_eq!(
            value["usage"]["prompt_tokens_details"]["cached_tokens"],
            expected,
            "turn {} must restore the full previous prompt state",
            turn + 1,
        );

        let prompt = messages
            .iter()
            .map(|message| message["content"].as_str().expect("text content"))
            .collect::<Vec<_>>()
            .join("|");
        assert_eq!(
            value["choices"][0]["message"]["content"],
            format!("echo:{prompt}"),
            "cache reuse must preserve the cold-generation result on turn {}",
            turn + 1,
        );

        let next = b'b' + u8::try_from(turn * 2).expect("small turn");
        messages.push(json!({
            "role": "assistant",
            "content": char::from(next).to_string(),
        }));
        messages.push(json!({
            "role": "user",
            "content": char::from(next + 1).to_string(),
        }));
    }
}

#[tokio::test]
async fn over_budget_prefix_is_not_cached() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let long = "x".repeat(20);
    let _ = post(app.clone(), "/v1/responses", responses_request(&long)).await;
    let (status, _, body) = post(app, "/v1/responses", responses_request(&(long + "suffix"))).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).expect("response JSON");
    assert_eq!(value["usage"]["input_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn full_generation_queue_returns_503() {
    let engine = Engine::start_fake(Some(MODEL), 1);
    let mut held = engine
        .submit(protocol::parse_chat(chat_request("hold")).expect("held request"))
        .expect("submit held job");
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), held.events.recv()).await,
        Ok(Some(WorkerEvent::Started(_)))
    ));
    let queued = engine
        .submit(protocol::parse_chat(chat_request("queued")).expect("queued request"))
        .expect("submit queued job");
    let full = engine.submit(protocol::parse_chat(chat_request("third")).expect("third request"));
    assert!(matches!(full, Err(SubmitError::Full)));
    held.cancelled.store(true, Ordering::Release);
    drop(held);
    drop(queued);
}

#[tokio::test]
async fn dropping_stream_cancels_generation_and_releases_worker() {
    let engine = Engine::start_fake(Some(MODEL), 1);
    let app = router(engine.clone());
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model":MODEL,"messages":[{"role":"user","content":"hold"}],"stream":true}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("stream response");
    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (status, _, body) = post(app, "/v1/chat/completions", chat_request("after")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn post_header_failures_use_endpoint_terminal_frames() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut chat = chat_request("fail-after-start");
    chat["stream"] = Value::Bool(true);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", chat).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"type\":\"server_error\""), "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");

    let mut responses = responses_request("fail-after-start");
    responses["stream"] = Value::Bool(true);
    let (status, _, body) = post(app, "/v1/responses", responses).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("event: response.failed"), "{body}");
    assert!(body.contains("\"type\":\"response.failed\""), "{body}");
}

#[tokio::test]
async fn buffered_chat_completes_a_two_turn_tool_loop() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(
        app.clone(),
        "/v1/chat/completions",
        chat_tool_request("call-tool"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: Value = serde_json::from_str(&body).expect("first response");
    let message = &first["choices"][0]["message"];
    assert_eq!(message["content"], Value::Null);
    assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
    let calls = message["tool_calls"].as_array().expect("tool calls");
    assert_eq!(calls.len(), 2);
    for (index, call) in calls.iter().enumerate() {
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["type"], "function");
        assert_eq!(
            call["function"]["name"],
            if index == 0 { "weather" } else { "time" }
        );
        assert!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap())
                .unwrap()
                .is_object()
        );
    }

    let mut replay = chat_tool_request("unused");
    replay["messages"] = json!([
        {"role":"user","content":"call-tool"},
        message,
        {"role":"tool","tool_call_id":calls[0]["id"],"content":"sunny"},
        {"role":"tool","tool_call_id":calls[1]["id"],"content":"noon"}
    ]);
    let (status, _, body) = post(app, "/v1/chat/completions", replay).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let second: Value = serde_json::from_str(&body).expect("second response");
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
    let content = second["choices"][0]["message"]["content"]
        .as_str()
        .expect("final content");
    assert!(
        content.contains("sunny") && content.contains("noon"),
        "{content}"
    );
}

#[tokio::test]
async fn buffered_tool_call_preserves_reasoning_without_visible_tool_xml() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(
        app,
        "/v1/chat/completions",
        chat_tool_request("call-tool-with-reasoning"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("tool response");
    let message = &value["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "fake reasoning");
    assert_eq!(message["content"], Value::Null);
    assert_eq!(
        message["tool_calls"].as_array().expect("tool calls").len(),
        2
    );
    assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    assert!(!body.contains("<think"));
    assert!(!body.contains("<tool_call>"));
}
#[tokio::test]
async fn streamed_chat_emits_stable_indexed_calls_separate_usage_and_replays() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = chat_tool_request("call-tool");
    request["stream"] = json!(true);
    request["stream_options"] = json!({"include_usage":true});
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!body.contains("<tool_call>"));
    let (frames, done) = parse_sse(&body);
    assert!(done);
    assert_eq!(frames[0].data["choices"][0]["delta"]["role"], "assistant");
    assert!(frames.iter().all(|frame| frame.data["model"] == MODEL));
    let call_headers = frames
        .iter()
        .filter(|frame| {
            frame.data["choices"][0]["delta"]["tool_calls"][0]["id"]
                .as_str()
                .is_some()
        })
        .collect::<Vec<_>>();
    assert_eq!(call_headers.len(), 2);
    let argument_frames = frames
        .iter()
        .filter(|frame| {
            frame.data["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .is_some_and(|arguments| !arguments.is_empty())
        })
        .collect::<Vec<_>>();
    assert_eq!(argument_frames.len(), 2);
    for index in 0..2 {
        assert_eq!(
            call_headers[index].data["choices"][0]["delta"]["tool_calls"][0]["index"],
            index
        );
        assert_eq!(
            argument_frames[index].data["choices"][0]["delta"]["tool_calls"][0]["index"],
            index
        );
    }
    let terminal = frames
        .iter()
        .find(|frame| frame.data["choices"][0]["finish_reason"] == "tool_calls")
        .expect("terminal tool reason");
    assert!(terminal.data.get("usage").is_none());
    let usage = frames.last().expect("usage frame");
    assert_eq!(usage.data["choices"], json!([]));
    assert!(usage.data["usage"]["completion_tokens"].as_u64().is_some());

    let calls = call_headers
        .iter()
        .zip(argument_frames.iter())
        .map(|(header, arguments)| {
            json!({
                "id": header.data["choices"][0]["delta"]["tool_calls"][0]["id"],
                "type": "function",
                "function": {
                    "name": header.data["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
                    "arguments": arguments.data["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                }
            })
        })
        .collect::<Vec<_>>();
    let mut replay = chat_tool_request("unused");
    replay["stream"] = json!(true);
    replay["messages"] = json!([
        {"role":"user","content":"call-tool"},
        {"role":"assistant","content":null,"tool_calls":calls},
        {"role":"tool","tool_call_id":calls[0]["id"],"content":"sunny"},
        {"role":"tool","tool_call_id":calls[1]["id"],"content":"noon"}
    ]);
    let (status, _, body) = post(app, "/v1/chat/completions", replay).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, done) = parse_sse(&body);
    assert!(done);
    let content = frames
        .iter()
        .filter_map(|frame| frame.data["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert!(
        content.contains("sunny") && content.contains("noon"),
        "{content}"
    );
}

#[tokio::test]
async fn buffered_responses_completes_a_two_turn_tool_loop() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(
        app.clone(),
        "/v1/responses",
        responses_tool_request("call-tool"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: Value = serde_json::from_str(&body).expect("first response");
    assert_eq!(first["status"], "completed");
    let calls = first["output"].as_array().expect("outputs");
    assert_eq!(calls.len(), 2);
    for call in calls {
        assert_eq!(call["type"], "function_call");
        assert_eq!(call["status"], "completed");
        assert!(call["id"].as_str().unwrap().starts_with("fc_"));
        assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
    }

    let mut replay = responses_tool_request("unused");
    replay["input"] = json!([
        {"role":"user","content":"call-tool"},
        responses_replay_call(&calls[0]),
        responses_replay_call(&calls[1]),
        {"type":"function_call_output","call_id":calls[0]["call_id"],"output":"sunny"},
        {"type":"function_call_output","call_id":calls[1]["call_id"],"output":"noon"}
    ]);
    let (status, _, body) = post(app, "/v1/responses", replay).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let second: Value = serde_json::from_str(&body).expect("second response");
    let content = second["output"][0]["content"][0]["text"]
        .as_str()
        .expect("final content");
    assert!(
        content.contains("sunny") && content.contains("noon"),
        "{content}"
    );
}

#[tokio::test]
async fn streamed_responses_has_lazy_exact_function_lifecycle_and_replays() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = responses_tool_request("call-tool");
    request["stream"] = json!(true);
    let (status, _, body) = post(app.clone(), "/v1/responses", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!body.contains("<tool_call>"));
    let (frames, done) = parse_sse(&body);
    assert!(!done);
    let names = frames
        .iter()
        .map(|frame| frame.event.as_deref().expect("named Responses event"))
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert!(
        names
            .iter()
            .all(|name| !name.contains("content_part") && !name.contains("output_text"))
    );
    for (sequence, frame) in frames.iter().enumerate() {
        assert_eq!(frame.data["sequence_number"], sequence);
        assert_eq!(frame.data["type"], frame.event.as_deref().unwrap());
    }
    let added = frames
        .iter()
        .filter(|frame| frame.event.as_deref() == Some("response.output_item.added"))
        .collect::<Vec<_>>();
    assert_eq!(added[0].data["output_index"], 0);
    assert_eq!(added[1].data["output_index"], 1);
    let completed = &frames.last().unwrap().data["response"];
    assert_eq!(completed["output"].as_array().unwrap().len(), 2);
    for (index, item) in completed["output"].as_array().unwrap().iter().enumerate() {
        assert_eq!(item["id"], added[index].data["item"]["id"]);
        assert_eq!(item["call_id"], added[index].data["item"]["call_id"]);
    }

    let calls = completed["output"].as_array().unwrap();
    let mut replay = responses_tool_request("unused");
    replay["stream"] = json!(true);
    replay["input"] = json!([
        {"role":"user","content":"call-tool"},
        responses_replay_call(&calls[0]),
        responses_replay_call(&calls[1]),
        {"type":"function_call_output","call_id":calls[0]["call_id"],"output":"sunny"},
        {"type":"function_call_output","call_id":calls[1]["call_id"],"output":"noon"}
    ]);
    let (status, _, body) = post(app, "/v1/responses", replay).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, _) = parse_sse(&body);
    let text = frames
        .iter()
        .filter(|frame| frame.event.as_deref() == Some("response.output_text.delta"))
        .filter_map(|frame| frame.data["delta"].as_str())
        .collect::<String>();
    assert!(text.contains("sunny") && text.contains("noon"), "{text}");
}

#[tokio::test]
async fn streamed_responses_closes_preamble_before_function_items() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut request = responses_tool_request("call-tool-with-preamble");
    request["stream"] = json!(true);
    let (status, _, body) = post(app, "/v1/responses", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, _) = parse_sse(&body);
    let first_function = frames
        .iter()
        .position(|frame| {
            frame.event.as_deref() == Some("response.output_item.added")
                && frame.data["item"]["type"] == "function_call"
        })
        .expect("function added");
    let message_done = frames
        .iter()
        .position(|frame| {
            frame.event.as_deref() == Some("response.output_item.done")
                && frame.data["item"]["type"] == "message"
        })
        .expect("message done");
    assert!(message_done < first_function);
    assert_eq!(frames[first_function].data["output_index"], 1);
    let completed = &frames.last().unwrap().data["response"];
    assert_eq!(completed["output"][0]["type"], "message");
    assert_eq!(completed["output"][1]["type"], "function_call");
    assert_eq!(completed["output"][2]["type"], "function_call");
}

#[tokio::test]
async fn tool_choice_parallel_policy_and_tool_free_shapes_are_preserved() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let mut none = chat_tool_request("call-tool");
    none["tool_choice"] = json!("none");
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", none).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
    assert_eq!(value["choices"][0]["message"]["content"], "echo:call-tool");
    assert!(value["choices"][0]["message"].get("tool_calls").is_none());

    let mut serial = chat_tool_request("call-tool");
    serial["parallel_tool_calls"] = json!(false);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", serial).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    let mut violation = chat_tool_request("call-tool-parallel-violation");
    violation["parallel_tool_calls"] = json!(false);
    let (status, _, body) = post(app, "/v1/chat/completions", violation).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");

    let ordinary = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(ordinary, "/v1/chat/completions", chat_request("shape")).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        value["choices"][0]["message"],
        json!({"role":"assistant","reasoning_content":"","content":"echo:shape"})
    );
}

#[tokio::test]
async fn chat_tool_validation_reports_exact_paths() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let valid_tool = chat_tools()[0].clone();
    let cases = [
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"code_interpreter","function":{"name":"x","parameters":{}}}]}),
            "tools[0].type",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"","parameters":{}}}]}),
            "tools[0].function.name",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}],"tools":[valid_tool.clone(),valid_tool.clone()]}),
            "tools[1].function.name",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x","parameters":[]}}]}),
            "tools[0].function.parameters",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x","parameters":{},"strict":"yes"}}]}),
            "tools[0].function.strict",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x","parameters":{},"extra":1}}]}),
            "tools[0].function.extra",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":null}]}),
            "messages[0].content",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":null}]}),
            "messages[1].content",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":"x","extra":1}]}),
            "messages[0].extra",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"weather","arguments":"bad"}}]},{"role":"tool","tool_call_id":"c","content":"x"}]}),
            "messages[1].tool_calls[0].function.arguments",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"weather","arguments":"[]"}}]},{"role":"tool","tool_call_id":"c","content":"x"}]}),
            "messages[1].tool_calls[0].function.arguments",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"missing","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"x"}]}),
            "messages[1].tool_calls[0].function.name",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"tool","tool_call_id":"orphan","content":"x"}]}),
            "messages[1].tool_call_id",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"weather","arguments":"{}"}}]}]}),
            "messages[1].tool_calls[0].id",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"weather","arguments":"{}"}},{"id":"c","type":"function","function":{"name":"time","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"x"}]}),
            "messages[1].tool_calls[1].id",
        ),
        (
            json!({"model":MODEL,"tools":chat_tools(),"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"weather","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"x"},{"role":"tool","tool_call_id":"c","content":"again"}]}),
            "messages[3].tool_call_id",
        ),
    ];
    for (request, expected_param) in cases {
        let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{expected_param}: {body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["param"], expected_param, "{body}");
    }
}

#[tokio::test]
async fn responses_tool_validation_reports_exact_paths() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let calls = json!([
        {"type":"function_call","id":"fc_1","call_id":"c","name":"weather","arguments":"{}"}
    ]);
    let cases = [
        (
            json!({"model":MODEL,"input":"x","tools":[{"type":"code","name":"x","parameters":{}}]}),
            "tools[0].type",
        ),
        (
            json!({"model":MODEL,"input":"x","tools":[{"type":"function","name":"x","parameters":{},"extra":1}]}),
            "tools[0].extra",
        ),
        (
            json!({"model":MODEL,"input":"x","tools":responses_tools(),"text":{"format":{"type":"json_object"}}}),
            "text.format",
        ),
        (
            json!({"model":MODEL,"input":[{"role":"user","content":null}]}),
            "input[0].content",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},{"type":"function_call","id":"fc_1","call_id":"","name":"weather","arguments":"{}"}]}),
            "input[1].call_id",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},{"type":"function_call","id":"fc_1","call_id":"c","name":"weather","arguments":"[]"},{"type":"function_call_output","call_id":"c","output":"x"}]}),
            "input[1].arguments",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},{"type":"function_call","id":"fc_1","call_id":"c","name":"missing","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"x"}]}),
            "input[1].name",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},{"type":"function_call_output","call_id":"orphan","output":"x"}]}),
            "input[1].call_id",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},calls[0].clone()]}),
            "input[1].call_id",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},{"type":"function_call","id":"fc_1","call_id":"c","name":"weather","arguments":"{}","extra":1},{"type":"function_call_output","call_id":"c","output":"x"}]}),
            "input[1].extra",
        ),
        (
            json!({"model":MODEL,"tools":responses_tools(),"input":[{"role":"user","content":"x"},{"type":"unknown"}]}),
            "input[1].type",
        ),
    ];
    for (request, expected_param) in cases {
        let (status, _, body) = post(app.clone(), "/v1/responses", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{expected_param}: {body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["param"], expected_param, "{body}");
    }
}

fn tiny_png_data_uri() -> &'static str {
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
}

#[tokio::test]
async fn chat_and_responses_preserve_mixed_image_order_buffered_and_streamed() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let chat = json!({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type":"text","text":"before"},
                {"type":"image_url","image_url":{"url":tiny_png_data_uri()}},
                {"type":"text","text":"after"}
            ]
        }]
    });
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", chat.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("chat response");
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "echo:before[image:png]after"
    );

    let image_only = json!({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type":"image_url","image_url":{"url":tiny_png_data_uri()}}
            ]
        }]
    });
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", image_only).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("image-only response");
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "echo:[image:png]"
    );

    let responses = json!({
        "model": MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type":"input_text","text":"before"},
                {"type":"input_image","image_url":tiny_png_data_uri()},
                {"type":"input_text","text":"after"}
            ]
        }]
    });
    let (status, _, body) = post(app.clone(), "/v1/responses", responses.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("responses response");
    assert_eq!(
        value["output"][0]["content"][0]["text"],
        "echo:before[image:png]after"
    );

    let mut chat_stream = chat;
    chat_stream["stream"] = json!(true);
    chat_stream["stream_options"] = json!({"include_usage":true});
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", chat_stream).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, done) = parse_sse(&body);
    assert!(done);
    let content = frames
        .iter()
        .filter_map(|frame| frame.data["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(content, "echo:before[image:png]after", "{body}");
    assert!(frames.iter().any(|frame| !frame.data["usage"].is_null()));

    let mut responses_stream = responses;
    responses_stream["stream"] = json!(true);
    let (status, _, body) = post(app, "/v1/responses", responses_stream).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (frames, done) = parse_sse(&body);
    assert!(!done);
    let text = frames
        .iter()
        .filter(|frame| frame.event.as_deref() == Some("response.output_text.delta"))
        .filter_map(|frame| frame.data["delta"].as_str())
        .collect::<String>();
    assert_eq!(text, "echo:before[image:png]after", "{body}");
    assert!(
        frames.windows(2).all(|pair| {
            pair[1].data["sequence_number"].as_u64()
                == pair[0].data["sequence_number"]
                    .as_u64()
                    .map(|value| value + 1)
        }),
        "{body}"
    );
}

#[tokio::test]
async fn image_validation_reports_exact_openai_parameter_paths() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let (status, _, body) = post(
        app.clone(),
        "/v1/chat/completions",
        json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":tiny_png_data_uri(),"detail":"high"}}]}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chat_cases = [
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}),
            "messages[0].content[0].image_url.url",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":tiny_png_data_uri()}}]},{"role":"user","content":"x"}]}),
            "messages[0].content[0].image_url",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":tiny_png_data_uri(),"extra":1}}]}]}),
            "messages[0].content[0].image_url.extra",
        ),
        (
            json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%%%%"}}]}]}),
            "messages[0].content[0].image_url.url",
        ),
    ];
    for (request, expected_param) in chat_cases {
        let (status, _, body) = post(app.clone(), "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let error: Value = serde_json::from_str(&body).expect("error response");
        assert_eq!(error["error"]["param"], expected_param, "{body}");
    }

    let response_cases = [
        (
            json!({"model":MODEL,"input":[{"role":"user","content":[{"type":"input_image","image_url":"file:///tmp/a.png"}]}]}),
            "input[0].content[0].image_url",
        ),
        (
            json!({"model":MODEL,"input":[{"role":"user","content":[{"type":"input_image","file_id":"file_1"}]}]}),
            "input[0].content[0].file_id",
        ),
        (
            json!({"model":MODEL,"input":[{"role":"assistant","content":[{"type":"input_image","image_url":tiny_png_data_uri()}]}]}),
            "input[0].content[0].image_url",
        ),
    ];
    for (request, expected_param) in response_cases {
        let (status, _, body) = post(app.clone(), "/v1/responses", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let error: Value = serde_json::from_str(&body).expect("error response");
        assert_eq!(error["error"]["param"], expected_param, "{body}");
    }

    let parts = (0..=protocol::MAX_IMAGES_PER_REQUEST)
        .map(|_| json!({"type":"image_url","image_url":{"url":tiny_png_data_uri()}}))
        .collect::<Vec<_>>();
    let request = json!({"model":MODEL,"messages":[{"role":"user","content":parts}]});
    let (status, _, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let error: Value = serde_json::from_str(&body).expect("error response");
    assert_eq!(
        error["error"]["param"],
        format!(
            "messages[0].content[{}].image_url.url",
            protocol::MAX_IMAGES_PER_REQUEST
        )
    );
}

#[tokio::test]
async fn image_requests_never_use_or_populate_the_text_prefix_cache() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let image_request = json!({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type":"text","text":"same"},
                {"type":"image_url","image_url":{"url":tiny_png_data_uri()}}
            ]
        }]
    });
    for _ in 0..2 {
        let (status, _, body) =
            post(app.clone(), "/v1/chat/completions", image_request.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).expect("image response");
        assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    }
    let _ = post(app.clone(), "/v1/chat/completions", chat_request("same")).await;
    let (status, _, body) = post(app, "/v1/chat/completions", chat_request("same-more")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("text response");
    assert!(
        value["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0)
    );
}

#[tokio::test]
async fn image_messages_survive_chat_and_responses_tool_replay() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let chat_user = json!({
        "role":"user",
        "content":[
            {"type":"text","text":"call-tool"},
            {"type":"image_url","image_url":{"url":tiny_png_data_uri()}}
        ]
    });
    let mut first_chat = chat_tool_request("unused");
    first_chat["messages"] = json!([chat_user.clone()]);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", first_chat).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: Value = serde_json::from_str(&body).expect("first chat response");
    let message = first["choices"][0]["message"].clone();
    let calls = message["tool_calls"].as_array().expect("chat tool calls");
    let mut replay = chat_tool_request("unused");
    replay["messages"] = json!([
        chat_user,
        message,
        {"role":"tool","tool_call_id":calls[0]["id"],"content":"sunny"},
        {"role":"tool","tool_call_id":calls[1]["id"],"content":"noon"}
    ]);
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", replay).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("sunny") && body.contains("noon"), "{body}");

    let responses_user = json!({
        "role":"user",
        "content":[
            {"type":"input_text","text":"call-tool"},
            {"type":"input_image","image_url":tiny_png_data_uri()}
        ]
    });
    let mut first_responses = responses_tool_request("unused");
    first_responses["input"] = json!([responses_user.clone()]);
    let (status, _, body) = post(app.clone(), "/v1/responses", first_responses).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: Value = serde_json::from_str(&body).expect("first Responses response");
    let calls = first["output"].as_array().expect("Responses tool calls");
    let mut replay = responses_tool_request("unused");
    replay["input"] = json!([
        responses_user,
        responses_replay_call(&calls[0]),
        responses_replay_call(&calls[1]),
        {"type":"function_call_output","call_id":calls[0]["call_id"],"output":"sunny"},
        {"type":"function_call_output","call_id":calls[1]["call_id"],"output":"noon"}
    ]);
    let (status, _, body) = post(app, "/v1/responses", replay).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("sunny") && body.contains("noon"), "{body}");
}

#[tokio::test]
async fn image_requests_keep_structured_output_and_tool_choice_none_contracts() {
    let app = router(Engine::start_fake(Some(MODEL), 8));
    let chat = json!({
        "model": MODEL,
        "messages": [{"role":"user","content":[
            {"type":"image_url","image_url":{"url":tiny_png_data_uri()}}
        ]}],
        "response_format": {"type":"json_object"}
    });
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", chat).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("structured chat response");
    assert_eq!(value["choices"][0]["message"]["content"], "{\"answer\":1}");
    assert_eq!(value["choices"][0]["finish_reason"], "stop");

    let responses = json!({
        "model": MODEL,
        "input": [{"role":"user","content":[
            {"type":"input_image","image_url":tiny_png_data_uri()}
        ]}],
        "text": {"format":{"type":"json_schema","name":"answer","strict":true,
            "schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}
    });
    let (status, _, body) = post(app.clone(), "/v1/responses", responses).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("structured Responses response");
    assert_eq!(value["output"][0]["content"][0]["text"], "{\"answer\":1}");

    let chat_with_tools = json!({
        "model": MODEL,
        "messages": [{"role":"user","content":[
            {"type":"text","text":"call-tool"},
            {"type":"image_url","image_url":{"url":tiny_png_data_uri()}}
        ]}],
        "tools": chat_tools(),
        "tool_choice": "none"
    });
    let (status, _, body) = post(app.clone(), "/v1/chat/completions", chat_with_tools).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("tool-free chat response");
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "echo:call-tool[image:png]"
    );
    assert!(value["choices"][0]["message"].get("tool_calls").is_none());

    let responses_with_tools = json!({
        "model": MODEL,
        "input": [{"role":"user","content":[
            {"type":"input_text","text":"call-tool"},
            {"type":"input_image","image_url":tiny_png_data_uri()}
        ]}],
        "tools": responses_tools(),
        "tool_choice": "none"
    });
    let (status, _, body) = post(app, "/v1/responses", responses_with_tools).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("tool-free Responses response");
    let output = value["output"].as_array().expect("Responses output");
    assert_eq!(output[0]["content"][0]["text"], "echo:call-tool[image:png]");
    assert!(output.iter().all(|item| item["type"] != "function_call"));
}

#[tokio::test]
async fn cancelling_an_image_request_leaves_the_next_text_request_clean() {
    let engine = Engine::start_fake(Some(MODEL), 8);
    let mut image_request = protocol::parse_chat(json!({
        "model": MODEL,
        "messages": [{"role":"user","content":[
            {"type":"text","text":"hold"},
            {"type":"image_url","image_url":{"url":tiny_png_data_uri()}}
        ]}]
    }))
    .expect("parse image request");
    media::decode_request_images(&mut image_request).expect("decode image request");
    let mut image_submission = engine.submit(image_request).expect("submit image request");
    assert!(matches!(
        image_submission.events.recv().await,
        Some(WorkerEvent::Started(_))
    ));
    image_submission
        .cancelled
        .store(true, std::sync::atomic::Ordering::Release);

    let text_request = protocol::parse_chat(chat_request("clean")).expect("parse text request");
    let mut text_submission = engine.submit(text_request).expect("submit text request");
    let record = loop {
        match text_submission.events.recv().await {
            Some(WorkerEvent::Complete { record, .. }) => break record,
            Some(WorkerEvent::Started(_) | WorkerEvent::Delta(_)) => {}
            Some(WorkerEvent::Failed(failure)) => panic!("text request failed: {failure:?}"),
            None => panic!("text request event channel closed"),
        }
    };
    assert_eq!(record.content, "echo:clean");
    assert_eq!(record.cached_tokens, 0);
}
