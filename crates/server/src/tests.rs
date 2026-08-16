use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
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
    (status, headers, String::from_utf8(bytes.to_vec()).expect("UTF-8 response"))
}

fn chat_request(prompt: &str) -> Value {
    json!({"model": MODEL, "messages": [{"role": "user", "content": prompt}]})
}

fn responses_request(prompt: &str) -> Value {
    json!({"model": MODEL, "input": prompt})
}

#[tokio::test]
async fn buffered_chat_completion_has_openai_shape_and_usage() {
    let app = router(Engine::start_fake(MODEL, 8));
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
    let app = router(Engine::start_fake(MODEL, 8));
    let mut request = chat_request("hello");
    request["stream"] = Value::Bool(true);
    let (status, headers, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers[header::CONTENT_TYPE]
        .to_str()
        .expect("content type")
        .starts_with("text/event-stream"));
    assert!(body.contains("\"role\":\"assistant\""), "{body}");
    assert!(body.contains("\"content\":"), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
    assert!(body.contains("\"usage\":"), "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
}

#[tokio::test]
async fn buffered_responses_completion_has_output_and_cached_usage() {
    let app = router(Engine::start_fake(MODEL, 8));
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
    let app = router(Engine::start_fake(MODEL, 8));
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
    for frame in body.split("\n\n").filter(|frame| frame.starts_with("event: ")) {
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
async fn model_mismatch_returns_openai_404() {
    let app = router(Engine::start_fake(MODEL, 8));
    let mut request = chat_request("hello");
    request["model"] = Value::String("other".to_string());
    let (status, _, body) = post(app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let value: Value = serde_json::from_str(&body).expect("error JSON");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["param"], "model");
    assert_eq!(value["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn malformed_and_unsupported_fields_are_rejected() {
    let app = router(Engine::start_fake(MODEL, 8));
    for request in [
        json!({"model": MODEL, "messages": []}),
        json!({"model": MODEL, "messages": [{"role":"user","content":[]}]}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "tools": []}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "n": 2}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "logprobs": true}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "stop": ["x"]}),
        json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "max_tokens": 1, "max_completion_tokens": 1}),
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
        json!({"model": MODEL, "input": "x", "tools": []}),
    ] {
        let (status, _, body) = post(app.clone(), "/v1/responses", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}

#[tokio::test]
async fn json_object_and_strict_schema_are_generated_under_constraints() {
    let app = router(Engine::start_fake(MODEL, 8));
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
    let app = router(Engine::start_fake(MODEL, 8));
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
    non_strict["text"] = json!({"format":{"type":"json_schema","name":"bad","strict":false,"schema":{}}});
    let (status, _, _) = post(app, "/v1/responses", non_strict).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn max_token_fields_map_to_length_and_incomplete() {
    let app = router(Engine::start_fake(MODEL, 8));
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
    let warm_app = router(Engine::start_fake(MODEL, 8));
    let _ = post(warm_app.clone(), "/v1/chat/completions", chat_request("abc")).await;
    let (status, _, warm_body) = post(
        warm_app,
        "/v1/chat/completions",
        chat_request("abcdef"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let warm: Value = serde_json::from_str(&warm_body).expect("warm JSON");
    assert!(warm["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("cached tokens") > 0);

    let cold_app = router(Engine::start_fake(MODEL, 8));
    let (_, _, cold_body) = post(cold_app, "/v1/chat/completions", chat_request("abcdef")).await;
    let cold: Value = serde_json::from_str(&cold_body).expect("cold JSON");
    assert_eq!(
        warm["choices"][0]["message"]["content"],
        cold["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn over_budget_prefix_is_not_cached() {
    let app = router(Engine::start_fake(MODEL, 8));
    let long = "x".repeat(20);
    let _ = post(app.clone(), "/v1/responses", responses_request(&long)).await;
    let (status, _, body) = post(
        app,
        "/v1/responses",
        responses_request(&(long + "suffix")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).expect("response JSON");
    assert_eq!(value["usage"]["input_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn full_generation_queue_returns_503() {
    let engine = Engine::start_fake(MODEL, 1);
    let mut held = engine
        .submit(protocol::parse_chat(chat_request("hold")).expect("held request"))
        .expect("submit held job");
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), held.events.recv()).await,
        Ok(Some(WorkerEvent::Started))
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
    let engine = Engine::start_fake(MODEL, 1);
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
    let app = router(Engine::start_fake(MODEL, 8));
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
