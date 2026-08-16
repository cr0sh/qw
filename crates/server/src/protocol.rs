use serde::Deserialize;
use serde_json::{Map, Value};

use qw_runtime::ChatMessage;

pub const DEFAULT_MAX_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Chat,
    Responses,
}

#[derive(Debug, Clone)]
pub enum OutputFormat {
    Text,
    JsonObject,
    JsonSchema { name: String, schema: Value },
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub endpoint: Endpoint,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone)]
pub struct RequestError {
    pub message: String,
    pub param: Option<String>,
}

impl RequestError {
    fn new(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            message: message.into(),
            param,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatWire {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(default)]
    stream: bool,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    response_format: Option<ChatFormat>,
    n: Option<usize>,
    logprobs: Option<Value>,
    stop: Option<Value>,
    tools: Option<Value>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ChatFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: ChatJsonSchema },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatJsonSchema {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ResponsesInput {
    String(String),
    Messages(Vec<WireMessage>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesWire {
    model: String,
    input: ResponsesInput,
    #[serde(default)]
    stream: bool,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    text: Option<ResponsesText>,
    store: Option<Value>,
    conversation: Option<Value>,
    tools: Option<Value>,
    include: Option<Value>,
    parallel_tool_calls: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesText {
    format: Option<ResponsesFormat>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ResponsesFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        strict: bool,
        schema: Value,
    },
}

pub fn parse_chat(value: Value) -> Result<CompletionRequest, RequestError> {
    let object = require_object(&value)?;
    reject_present(object, "tools", "tools are not supported")?;
    reject_present(object, "logprobs", "logprobs are not supported")?;
    let wire: ChatWire = serde_json::from_value(value).map_err(|error| {
        RequestError::new(format!("invalid Chat Completions request: {error}"), None)
    })?;
    if wire.messages.is_empty() {
        return Err(RequestError::new(
            "messages must contain at least one message",
            Some("messages".to_string()),
        ));
    }
    if wire.max_completion_tokens.is_some() && wire.max_tokens.is_some() {
        return Err(RequestError::new(
            "max_completion_tokens and max_tokens cannot both be provided",
            Some("max_completion_tokens".to_string()),
        ));
    }
    if wire.n.is_some_and(|n| n != 1) {
        return Err(RequestError::new(
            "only n=1 is supported",
            Some("n".to_string()),
        ));
    }
    if let Some(stop) = wire.stop.as_ref()
        && stop_is_nonempty(stop)
    {
        return Err(RequestError::new(
            "stop sequences are not supported",
            Some("stop".to_string()),
        ));
    }
    let max_tokens = wire
        .max_completion_tokens
        .or(wire.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    validate_sampling(max_tokens, wire.temperature, wire.top_p)?;
    let output_format = match wire.response_format.unwrap_or(ChatFormat::Text) {
        ChatFormat::Text => OutputFormat::Text,
        ChatFormat::JsonObject => OutputFormat::JsonObject,
        ChatFormat::JsonSchema { json_schema } => {
            if !json_schema.strict {
                return Err(RequestError::new(
                    "json_schema.strict must be true",
                    Some("response_format.json_schema.strict".to_string()),
                ));
            }
            require_schema_object(&json_schema.schema, "response_format.json_schema.schema")?;
            OutputFormat::JsonSchema {
                name: json_schema.name,
                schema: json_schema.schema,
            }
        }
    };
    Ok(CompletionRequest {
        endpoint: Endpoint::Chat,
        model: wire.model,
        messages: convert_messages(wire.messages)?,
        stream: wire.stream,
        max_tokens,
        temperature: wire.temperature,
        top_p: wire.top_p,
        seed: wire.seed,
        output_format,
    })
}

pub fn parse_responses(value: Value) -> Result<CompletionRequest, RequestError> {
    let object = require_object(&value)?;
    for (field, message) in [
        ("store", "stored responses are not supported"),
        ("conversation", "conversation IDs are not supported"),
        ("tools", "tools are not supported"),
        ("include", "include fields are not supported"),
        ("parallel_tool_calls", "parallel tool calls are not supported"),
    ] {
        reject_present(object, field, message)?;
    }
    let wire: ResponsesWire = serde_json::from_value(value)
        .map_err(|error| RequestError::new(format!("invalid Responses request: {error}"), None))?;
    let max_tokens = wire.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    validate_sampling(max_tokens, wire.temperature, wire.top_p)?;
    let messages = match wire.input {
        ResponsesInput::String(content) => vec![ChatMessage {
            role: "user".to_string(),
            content: Some(content),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        ResponsesInput::Messages(messages) => convert_messages(messages)?,
    };
    if messages.is_empty() {
        return Err(RequestError::new(
            "input messages must not be empty",
            Some("input".to_string()),
        ));
    }
    let output_format = match wire.text.and_then(|text| text.format).unwrap_or(ResponsesFormat::Text) {
        ResponsesFormat::Text => OutputFormat::Text,
        ResponsesFormat::JsonObject => OutputFormat::JsonObject,
        ResponsesFormat::JsonSchema { name, strict, schema } => {
            if !strict {
                return Err(RequestError::new(
                    "text.format.strict must be true",
                    Some("text.format.strict".to_string()),
                ));
            }
            require_schema_object(&schema, "text.format.schema")?;
            OutputFormat::JsonSchema { name, schema }
        }
    };
    Ok(CompletionRequest {
        endpoint: Endpoint::Responses,
        model: wire.model,
        messages,
        stream: wire.stream,
        max_tokens,
        temperature: wire.temperature,
        top_p: wire.top_p,
        seed: None,
        output_format,
    })
}

fn require_object(value: &Value) -> Result<&Map<String, Value>, RequestError> {
    value.as_object().ok_or_else(|| {
        RequestError::new("request body must be a JSON object", None)
    })
}

fn reject_present(
    object: &Map<String, Value>,
    field: &str,
    message: &str,
) -> Result<(), RequestError> {
    if object.contains_key(field) {
        Err(RequestError::new(message, Some(field.to_string())))
    } else {
        Ok(())
    }
}

fn stop_is_nonempty(stop: &Value) -> bool {
    match stop {
        Value::Null => false,
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        _ => true,
    }
}

fn validate_sampling(
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), RequestError> {
    if max_tokens == 0 {
        return Err(RequestError::new(
            "maximum output tokens must be greater than zero",
            Some("max_tokens".to_string()),
        ));
    }
    if temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
        return Err(RequestError::new(
            "temperature must be finite and between 0 and 2",
            Some("temperature".to_string()),
        ));
    }
    if top_p.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0) {
        return Err(RequestError::new(
            "top_p must be finite and greater than 0 and at most 1",
            Some("top_p".to_string()),
        ));
    }
    Ok(())
}

fn convert_messages(messages: Vec<WireMessage>) -> Result<Vec<ChatMessage>, RequestError> {
    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            if !matches!(message.role.as_str(), "system" | "user" | "assistant" | "tool") {
                return Err(RequestError::new(
                    format!("unsupported message role {:?}", message.role),
                    Some(format!("messages.{index}.role")),
                ));
            }
            Ok(ChatMessage {
                role: message.role,
                content: Some(message.content),
                tool_calls: Vec::new(),
                tool_call_id: None,
            })
        })
        .collect()
}

fn require_schema_object(schema: &Value, param: &str) -> Result<(), RequestError> {
    if schema.is_object() {
        Ok(())
    } else {
        Err(RequestError::new(
            "JSON Schema must be an object",
            Some(param.to_string()),
        ))
    }
}
