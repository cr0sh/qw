mod engine;
mod grammar;
mod prefix_cache;
mod media;
pub mod protocol;
mod tool_calls;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::Json;
use axum::extract::{State, rejection::JsonRejection};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use futures_util::stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;

pub use engine::{Engine, SubmitError};
use engine::{
    Admission, CompletionRecord, FailureKind, FinishReason, GeneratedToolCall, WorkerDelta,
    WorkerEvent, WorkerFailure,
};
use protocol::{Endpoint, RequestError};

#[derive(Clone)]
struct AppState {
    engine: Engine,
}

pub fn router(engine: Engine) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .with_state(AppState { engine })
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    handle(state, payload, Endpoint::Chat).await
}

async fn responses(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    handle(state, payload, Endpoint::Responses).await
}

async fn handle(
    state: AppState,
    payload: Result<Json<Value>, JsonRejection>,
    endpoint: Endpoint,
) -> Response {
    let value = match payload {
        Ok(Json(value)) => value,
        Err(error) => {
            return ApiError::invalid(format!("invalid JSON request body: {error}"), None)
                .into_response();
        }
    };
    let request = match endpoint {
        Endpoint::Chat => protocol::parse_chat(value),
        Endpoint::Responses => protocol::parse_responses(value),
    };
    let mut request = match request {
        Ok(request) => request,
        Err(error) => return ApiError::from_request(error).into_response(),
    };
    if !request.image_params.is_empty() && !state.engine.supports_image_inputs() {
        return ApiError::invalid(
            "model does not support image inputs",
            request.image_params.first().cloned(),
        )
        .into_response();
    }
    if request.model != state.engine.model_id() {
        return ApiError::model_not_found(&request.model).into_response();
    }
    if let Err(error) = media::decode_request_images(&mut request) {
        return ApiError::from_request(error).into_response();
    }
    let stream_requested = request.stream;
    let model_id = state.engine.model_id().to_string();
    let submission = match state.engine.submit(request) {
        Ok(submission) => submission,
        Err(SubmitError::Full) => return ApiError::queue_full().into_response(),
        Err(SubmitError::Closed) => return ApiError::server("generation worker is unavailable").into_response(),
    };
    if stream_requested {
        streaming_response(endpoint, model_id, submission).await
    } else {
        buffered_response(submission).await
    }
}

async fn buffered_response(mut submission: engine::Submission) -> Response {
    let mut guard = CancelGuard {
        cancelled: submission.cancelled.clone(),
        armed: true,
    };
    while let Some(event) = submission.events.recv().await {
        match event {
            WorkerEvent::Started | WorkerEvent::Delta(_) => {}
            WorkerEvent::Complete(record) => {
                guard.armed = false;
                return Json(buffered_json(&record)).into_response();
            }
            WorkerEvent::Failed(failure) => {
                guard.armed = false;
                return ApiError::from_worker(failure).into_response();
            }
        }
    }
    ApiError::server("generation worker closed without a result").into_response()
}

async fn streaming_response(
    endpoint: Endpoint,
    model: String,
    mut submission: engine::Submission,
) -> Response {
    let mut admission_guard = CancelGuard {
        cancelled: submission.cancelled.clone(),
        armed: true,
    };
    let first = submission.events.recv().await;
    match first {
        Some(WorkerEvent::Started) => admission_guard.armed = false,
        Some(WorkerEvent::Failed(failure)) => {
            admission_guard.armed = false;
            return ApiError::from_worker(failure).into_response();
        }
        Some(WorkerEvent::Complete(record)) => {
            admission_guard.armed = false;
            return Json(buffered_json(&record)).into_response();
        }
        Some(WorkerEvent::Delta(_)) => {
            return ApiError::server("generation worker emitted output before admission").into_response();
        }
        None => return ApiError::server("generation worker closed during admission").into_response(),
    }

    let state = SseState::new(
        endpoint,
        submission.admission,
        model,
        submission.events,
        submission.cancelled,
    );
    let events = stream::unfold(state, |mut state| async move {
        let event = state.next_event().await?;
        Some((Ok::<Event, Infallible>(event), state))
    });
    Sse::new(events).into_response()
}

struct CancelGuard {
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

struct SseState {
    endpoint: Endpoint,
    admission: Admission,
    model: String,
    receiver: mpsc::Receiver<WorkerEvent>,
    pending: VecDeque<Event>,
    sequence: u64,
    response_message_open: bool,
    guard: CancelGuard,
}

impl SseState {
    fn new(
        endpoint: Endpoint,
        admission: Admission,
        model: String,
        receiver: mpsc::Receiver<WorkerEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        let mut state = Self {
            endpoint,
            model,
            admission,
            receiver,
            pending: VecDeque::new(),
            sequence: 0,
            response_message_open: false,
            guard: CancelGuard {
                cancelled,
                armed: true,
            },
        };
        state.enqueue_initial();
        state
    }

    async fn next_event(&mut self) -> Option<Event> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            match self.receiver.recv().await {
                Some(WorkerEvent::Started) => continue,
                Some(WorkerEvent::Delta(delta)) => self.enqueue_delta(delta),
                Some(WorkerEvent::Complete(record)) => {
                    self.enqueue_complete(record);
                    self.guard.armed = false;
                }
                Some(WorkerEvent::Failed(failure)) => {
                    self.enqueue_failure(failure);
                    self.guard.armed = false;
                }
                None => {
                    self.guard.cancelled.store(true, Ordering::Release);
                    return None;
                }
            }
        }
    }

    fn enqueue_initial(&mut self) {
        match self.endpoint {
            Endpoint::Chat => {
                let chunk = json!({
                    "id": self.admission.response_id,
                    "object": "chat.completion.chunk",
                    "created": self.admission.created,
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
                });
                self.pending.push_back(data_event(chunk));
            }
            Endpoint::Responses => {
                let created = json!({
                    "type": "response.created",
                    "sequence_number": self.next_sequence(),
                    "response": {
                        "id": self.admission.response_id,
                        "object": "response",
                        "created_at": self.admission.created,
                        "model": self.model,
                        "status": "in_progress",
                        "output": []
                    }
                });
                self.pending
                    .push_back(named_event("response.created", created));
            }
        }
    }

    fn enqueue_delta(&mut self, delta: WorkerDelta) {
        match delta {
            WorkerDelta::Reasoning(reasoning) => {
                if reasoning.is_empty() || self.endpoint == Endpoint::Responses {
                    return;
                }
                self.pending.push_back(data_event(json!({
                    "id": self.admission.response_id,
                    "object": "chat.completion.chunk",
                    "created": self.admission.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"reasoning_content": reasoning},
                        "finish_reason": null
                    }]
                })));
            }
            WorkerDelta::Content(content) => {
                if content.is_empty() {
                    return;
                }
                match self.endpoint {
                    Endpoint::Chat => {
                        self.pending.push_back(data_event(json!({
                            "id": self.admission.response_id,
                            "object": "chat.completion.chunk",
                            "created": self.admission.created,
                            "model": self.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": null
                            }]
                        })));
                    }
                    Endpoint::Responses => {
                        self.ensure_response_message_open();
                        let event = json!({
                            "type": "response.output_text.delta",
                            "sequence_number": self.next_sequence(),
                            "item_id": self.admission.message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": content
                        });
                        self.pending
                            .push_back(named_event("response.output_text.delta", event));
                    }
                }
            }
        }
    }

    fn ensure_response_message_open(&mut self) {
        if self.response_message_open {
            return;
        }
        let added = json!({
            "type": "response.output_item.added",
            "sequence_number": self.next_sequence(),
            "output_index": 0,
            "item": {
                "id": self.admission.message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        });
        self.pending
            .push_back(named_event("response.output_item.added", added));
        let part = json!({
            "type": "response.content_part.added",
            "sequence_number": self.next_sequence(),
            "item_id": self.admission.message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        });
        self.pending
            .push_back(named_event("response.content_part.added", part));
        self.response_message_open = true;
    }

    fn enqueue_complete(&mut self, record: CompletionRecord) {
        match self.endpoint {
            Endpoint::Chat => self.enqueue_chat_complete(&record),
            Endpoint::Responses => self.enqueue_responses_complete(&record),
        }
    }

    fn enqueue_chat_complete(&mut self, record: &CompletionRecord) {
        for (index, call) in record.tool_calls.iter().enumerate() {
            self.pending.push_back(data_event(json!({
                "id": record.admission.response_id,
                "object": "chat.completion.chunk",
                "created": record.admission.created,
                "model": record.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": index,
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": ""}
                        }]
                    },
                    "finish_reason": null
                }]
            })));
            self.pending.push_back(data_event(json!({
                "id": record.admission.response_id,
                "object": "chat.completion.chunk",
                "created": record.admission.created,
                "model": record.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": index,
                            "function": {"arguments": call.arguments}
                        }]
                    },
                    "finish_reason": null
                }]
            })));
        }
        let mut terminal = json!({
            "id": record.admission.response_id,
            "object": "chat.completion.chunk",
            "created": record.admission.created,
            "model": record.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason_string(record.finish_reason)
            }]
        });
        if !record.stream_include_usage {
            terminal
                .as_object_mut()
                .expect("terminal chunk is an object")
                .insert("usage".to_string(), chat_usage(record));
        }
        self.pending.push_back(data_event(terminal));
        if record.stream_include_usage {
            self.pending.push_back(data_event(json!({
                "id": record.admission.response_id,
                "object": "chat.completion.chunk",
                "created": record.admission.created,
                "model": record.model,
                "choices": [],
                "usage": chat_usage(record)
            })));
        }
        self.pending.push_back(Event::default().data("[DONE]"));
    }

    fn enqueue_responses_complete(&mut self, record: &CompletionRecord) {
        if !record.content.is_empty() || record.tool_calls.is_empty() {
            self.ensure_response_message_open();
        }
        if self.response_message_open {
            let text_done = json!({
                "type": "response.output_text.done",
                "sequence_number": self.next_sequence(),
                "item_id": record.admission.message_id,
                "output_index": 0,
                "content_index": 0,
                "text": record.content
            });
            self.pending
                .push_back(named_event("response.output_text.done", text_done));
            let part_done = json!({
                "type": "response.content_part.done",
                "sequence_number": self.next_sequence(),
                "item_id": record.admission.message_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": record.content, "annotations": []}
            });
            self.pending
                .push_back(named_event("response.content_part.done", part_done));
            let item_done = json!({
                "type": "response.output_item.done",
                "sequence_number": self.next_sequence(),
                "output_index": 0,
                "item": response_message_item(record)
            });
            self.pending
                .push_back(named_event("response.output_item.done", item_done));
        }
        let first_call_index = usize::from(self.response_message_open);
        for (index, call) in record.tool_calls.iter().enumerate() {
            let output_index = first_call_index + index;
            let added = json!({
                "type": "response.output_item.added",
                "sequence_number": self.next_sequence(),
                "output_index": output_index,
                "item": response_function_item(call, false)
            });
            self.pending
                .push_back(named_event("response.output_item.added", added));
            let arguments_delta = json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": self.next_sequence(),
                "item_id": call.item_id,
                "output_index": output_index,
                "delta": call.arguments
            });
            self.pending.push_back(named_event(
                "response.function_call_arguments.delta",
                arguments_delta,
            ));
            let arguments_done = json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": self.next_sequence(),
                "item_id": call.item_id,
                "output_index": output_index,
                "arguments": call.arguments
            });
            self.pending.push_back(named_event(
                "response.function_call_arguments.done",
                arguments_done,
            ));
            let item_done = json!({
                "type": "response.output_item.done",
                "sequence_number": self.next_sequence(),
                "output_index": output_index,
                "item": response_function_item(call, true)
            });
            self.pending
                .push_back(named_event("response.output_item.done", item_done));
        }
        let completed = json!({
            "type": "response.completed",
            "sequence_number": self.next_sequence(),
            "response": responses_json(record)
        });
        self.pending
            .push_back(named_event("response.completed", completed));
    }

    fn enqueue_failure(&mut self, failure: WorkerFailure) {
        let error = error_json(
            &failure.message,
            match failure.kind {
                FailureKind::InvalidRequest => "invalid_request_error",
                FailureKind::Server => "server_error",
            },
            failure.param.as_deref(),
            None,
        );
        match self.endpoint {
            Endpoint::Chat => {
                self.pending.push_back(data_event(error));
                self.pending.push_back(Event::default().data("[DONE]"));
            }
            Endpoint::Responses => {
                let failed = json!({
                    "type": "response.failed",
                    "sequence_number": self.next_sequence(),
                    "response": {
                        "id": self.admission.response_id,
                        "object": "response",
                        "model": self.model,
                        "status": "failed",
                        "error": error["error"].clone()
                    }
                });
                self.pending
                    .push_back(named_event("response.failed", failed));
            }
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let current = self.sequence;
        self.sequence += 1;
        current
    }
}

fn buffered_json(record: &CompletionRecord) -> Value {
    match record.endpoint {
        Endpoint::Chat => json!({
            "id": record.admission.response_id,
            "object": "chat.completion",
            "created": record.admission.created,
            "model": record.model,
            "choices": [{
                "index": 0,
                "message": chat_message(record),
                "finish_reason": finish_reason_string(record.finish_reason)
            }],
            "usage": chat_usage(record)
        }),
        Endpoint::Responses => responses_json(record),
    }
}

fn chat_message(record: &CompletionRecord) -> Value {
    let mut message = serde_json::Map::new();
    message.insert("role".to_string(), json!("assistant"));
    message.insert(
        "reasoning_content".to_string(),
        Value::String(record.reasoning_content.clone()),
    );
    message.insert(
        "content".to_string(),
        if record.content.is_empty() && !record.tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(record.content.clone())
        },
    );
    if !record.tool_calls.is_empty() {
        message.insert(
            "tool_calls".to_string(),
            Value::Array(
                record
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }
    Value::Object(message)
}

fn responses_json(record: &CompletionRecord) -> Value {
    let incomplete = match record.finish_reason {
        FinishReason::Stop | FinishReason::ToolCalls => Value::Null,
        FinishReason::Length => json!({"reason": "max_output_tokens"}),
    };
    json!({
        "id": record.admission.response_id,
        "object": "response",
        "created_at": record.admission.created,
        "model": record.model,
        "status": if record.finish_reason == FinishReason::Length { "incomplete" } else { "completed" },
        "incomplete_details": incomplete,
        "output": response_output_items(record),
        "usage": {
            "input_tokens": record.prompt_tokens,
            "input_tokens_details": {"cached_tokens": record.cached_tokens},
            "output_tokens": record.completion_tokens,
            "total_tokens": record.prompt_tokens + record.completion_tokens
        }
    })
}

fn response_output_items(record: &CompletionRecord) -> Vec<Value> {
    let mut output = Vec::with_capacity(record.tool_calls.len() + 1);
    if !record.content.is_empty() || record.tool_calls.is_empty() {
        output.push(response_message_item(record));
    }
    output.extend(
        record
            .tool_calls
            .iter()
            .map(|call| response_function_item(call, true)),
    );
    output
}

fn response_message_item(record: &CompletionRecord) -> Value {
    json!({
        "id": record.admission.message_id,
        "type": "message",
        "status": if record.finish_reason == FinishReason::Length { "incomplete" } else { "completed" },
        "role": "assistant",
        "content": [{"type": "output_text", "text": record.content, "annotations": []}]
    })
}

fn response_function_item(call: &GeneratedToolCall, completed: bool) -> Value {
    json!({
        "id": call.item_id,
        "type": "function_call",
        "status": if completed { "completed" } else { "in_progress" },
        "call_id": call.id,
        "name": call.name,
        "arguments": if completed { call.arguments.as_str() } else { "" }
    })
}

fn chat_usage(record: &CompletionRecord) -> Value {
    json!({
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.prompt_tokens + record.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": record.cached_tokens}
    })
}

fn finish_reason_string(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

fn data_event(value: Value) -> Event {
    Event::default().data(value.to_string())
}

fn named_event(name: &'static str, value: Value) -> Event {
    Event::default().event(name).data(value.to_string())
}

struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    param: Option<String>,
    code: Option<&'static str>,
}

impl ApiError {
    fn invalid(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            param,
            code: None,
        }
    }

    fn from_request(error: RequestError) -> Self {
        Self::invalid(error.message, error.param)
    }

    fn from_worker(error: WorkerFailure) -> Self {
        match error.kind {
            FailureKind::InvalidRequest => Self::invalid(error.message, error.param),
            FailureKind::Server => Self::server(error.message),
        }
    }

    fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("model {model:?} was not found"),
            error_type: "invalid_request_error",
            param: Some("model".to_string()),
            code: Some("model_not_found"),
        }
    }

    fn queue_full() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "generation queue is full".to_string(),
            error_type: "server_error",
            param: None,
            code: Some("queue_full"),
        }
    }

    fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            error_type: "server_error",
            param: None,
            code: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(error_json(
                &self.message,
                self.error_type,
                self.param.as_deref(),
                self.code,
            )),
        )
            .into_response()
    }
}

fn error_json(
    message: &str,
    error_type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Value {
    json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code
        }
    })
}

#[cfg(test)]
mod tests;
