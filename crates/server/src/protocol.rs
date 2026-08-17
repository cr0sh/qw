use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use serde_json::{Map, Value};

use qw_runtime::{
    ChatContentPart, ChatImageUrl, ChatMessage, ChatMessageContent, ChatTool, ChatToolCall,
    ChatToolCallFunction, ChatToolFunction,
};

use crate::media::DecodedImage;

pub const DEFAULT_MAX_TOKENS: usize = 128;
pub const MAX_IMAGES_PER_REQUEST: usize = 16;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    XHigh,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::XHigh => "xhigh",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub endpoint: Endpoint,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ChatTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub enable_thinking: bool,
    pub stream: bool,
    pub stream_include_usage: bool,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub output_format: OutputFormat,
    pub image_params: Vec<String>,
    pub decoded_images: Vec<DecodedImage>,
}

#[derive(Debug, Clone)]
pub struct RequestError {
    pub message: String,
    pub param: Option<String>,
}

impl RequestError {
    pub(crate) fn new(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            message: message.into(),
            param,
        }
    }

    pub(crate) fn at(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self::new(message, Some(param.into()))
    }
}

#[derive(Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
enum ChatWireMessage {
    System {
        content: Value,
    },
    User {
        content: Value,
    },
    Assistant {
        #[serde(default)]
        content: Value,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<Value>>,
    },
    Tool {
        content: Value,
        tool_call_id: Value,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatWire {
    model: String,
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<ChatStreamOptions>,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
    reasoning_effort: Option<String>,
    thinking: Option<ChatThinking>,
    #[serde(rename = "preserve_thinking")]
    _preserve_thinking: Option<bool>,
    enable_thinking: Option<bool>,
    #[serde(rename = "chat_template_kwargs")]
    _chat_template_kwargs: Option<Map<String, Value>>,
    #[serde(rename = "mcp_timeout")]
    _mcp_timeout: Option<Value>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    response_format: Option<ChatFormat>,
    n: Option<usize>,
    stop: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum ChatThinking {
    Enabled {
        #[serde(rename = "budgetTokens")]
        _budget_tokens: Option<usize>,
    },
    Disabled,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatStreamOptions {
    #[serde(default)]
    include_usage: bool,
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
    Items(Vec<Value>),
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
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesMessageWire {
    role: String,
    content: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesFunctionCallWire {
    #[serde(rename = "type")]
    item_type: String,
    id: Value,
    call_id: Value,
    name: Value,
    arguments: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesFunctionOutputWire {
    #[serde(rename = "type")]
    item_type: String,
    call_id: Value,
    output: Value,
}

pub fn parse_chat(value: Value) -> Result<CompletionRequest, RequestError> {
    let object = require_object(&value)?;
    reject_present(object, "logprobs", "logprobs are not supported")?;
    if let Some(reasoning_effort) = object.get("reasoning_effort")
        && !reasoning_effort.is_string()
    {
        return Err(RequestError::at(
            "reasoning_effort must be a string",
            "reasoning_effort",
        ));
    }
    let wire: ChatWire = serde_json::from_value(value).map_err(|error| {
        RequestError::new(format!("invalid Chat Completions request: {error}"), None)
    })?;
    if wire.messages.is_empty() {
        return Err(RequestError::at(
            "messages must contain at least one message",
            "messages",
        ));
    }
    if wire.max_completion_tokens.is_some() && wire.max_tokens.is_some() {
        return Err(RequestError::at(
            "max_completion_tokens and max_tokens cannot both be provided",
            "max_completion_tokens",
        ));
    }
    if wire.n.is_some_and(|n| n != 1) {
        return Err(RequestError::at("only n=1 is supported", "n"));
    }
    if let Some(stop) = wire.stop.as_ref()
        && stop_is_nonempty(stop)
    {
        return Err(RequestError::at("stop sequences are not supported", "stop"));
    }
    let max_tokens = wire
        .max_completion_tokens
        .or(wire.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    validate_sampling(max_tokens, wire.temperature, wire.top_p)?;
    let reasoning_effort = parse_reasoning_effort(wire.reasoning_effort.as_deref())?;
    let template_enable_thinking = wire
        ._chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                RequestError::at(
                    "chat_template_kwargs.enable_thinking must be a boolean",
                    "chat_template_kwargs.enable_thinking",
                )
            })
        })
        .transpose()?;
    let opencode_enable_thinking = wire
        .thinking
        .map(|thinking| matches!(thinking, ChatThinking::Enabled { .. }));
    let enable_thinking = wire
        .enable_thinking
        .or(template_enable_thinking)
        .or(opencode_enable_thinking)
        .unwrap_or(true);
    let output_format = parse_chat_format(wire.response_format)?;
    let tools = parse_tools(wire.tools.as_deref().unwrap_or_default(), ToolDialect::Chat)?;
    if !tools.is_empty() && !matches!(output_format, OutputFormat::Text) {
        return Err(RequestError::at(
            "tools cannot be combined with structured response formats",
            "response_format",
        ));
    }
    let tool_choice = parse_tool_choice(wire.tool_choice.as_ref())?;
    let mut image_params = Vec::new();
    let messages = parse_chat_messages(wire.messages, &tools, &mut image_params)?;
    Ok(CompletionRequest {
        endpoint: Endpoint::Chat,
        model: wire.model,
        messages,
        tools,
        tool_choice,
        reasoning_effort,
        enable_thinking,
        parallel_tool_calls: wire.parallel_tool_calls.unwrap_or(true),
        stream: wire.stream,
        stream_include_usage: wire
            .stream_options
            .is_some_and(|options| options.include_usage),
        max_tokens,
        temperature: wire.temperature,
        top_p: wire.top_p,
        seed: wire.seed,
        output_format,
        image_params,
        decoded_images: Vec::new(),
    })
}

pub fn parse_responses(value: Value) -> Result<CompletionRequest, RequestError> {
    let object = require_object(&value)?;
    for (field, message) in [
        ("store", "stored responses are not supported"),
        ("conversation", "conversation IDs are not supported"),
        ("include", "include fields are not supported"),
    ] {
        reject_present(object, field, message)?;
    }
    let wire: ResponsesWire = serde_json::from_value(value)
        .map_err(|error| RequestError::new(format!("invalid Responses request: {error}"), None))?;
    let max_tokens = wire.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    validate_sampling(max_tokens, wire.temperature, wire.top_p)?;
    let output_format = parse_responses_format(wire.text.and_then(|text| text.format))?;
    let tools = parse_tools(
        wire.tools.as_deref().unwrap_or_default(),
        ToolDialect::Responses,
    )?;
    if !tools.is_empty() && !matches!(output_format, OutputFormat::Text) {
        return Err(RequestError::at(
            "tools cannot be combined with structured response formats",
            "text.format",
        ));
    }
    let tool_choice = parse_tool_choice(wire.tool_choice.as_ref())?;
    let mut image_params = Vec::new();
    let messages = match wire.input {
        ResponsesInput::String(content) => {
            if content.is_empty() {
                return Err(RequestError::at("input must not be empty", "input"));
            }
            vec![ChatMessage {
                role: "user".to_string(),
                content: Some(ChatMessageContent::Text(content)),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }]
        }
        ResponsesInput::Items(items) => parse_responses_items(items, &tools, &mut image_params)?,
    };
    Ok(CompletionRequest {
        endpoint: Endpoint::Responses,
        model: wire.model,
        messages,
        tools,
        tool_choice,
        reasoning_effort: None,
        enable_thinking: true,
        parallel_tool_calls: wire.parallel_tool_calls.unwrap_or(true),
        stream: wire.stream,
        stream_include_usage: false,
        max_tokens,
        temperature: wire.temperature,
        top_p: wire.top_p,
        seed: None,
        output_format,
        image_params,
        decoded_images: Vec::new(),
    })
}

fn parse_chat_format(format: Option<ChatFormat>) -> Result<OutputFormat, RequestError> {
    match format.unwrap_or(ChatFormat::Text) {
        ChatFormat::Text => Ok(OutputFormat::Text),
        ChatFormat::JsonObject => Ok(OutputFormat::JsonObject),
        ChatFormat::JsonSchema { json_schema } => {
            if !json_schema.strict {
                return Err(RequestError::at(
                    "json_schema.strict must be true",
                    "response_format.json_schema.strict",
                ));
            }
            require_schema_object(&json_schema.schema, "response_format.json_schema.schema")?;
            Ok(OutputFormat::JsonSchema {
                name: json_schema.name,
                schema: json_schema.schema,
            })
        }
    }
}

fn parse_responses_format(format: Option<ResponsesFormat>) -> Result<OutputFormat, RequestError> {
    match format.unwrap_or(ResponsesFormat::Text) {
        ResponsesFormat::Text => Ok(OutputFormat::Text),
        ResponsesFormat::JsonObject => Ok(OutputFormat::JsonObject),
        ResponsesFormat::JsonSchema {
            name,
            strict,
            schema,
        } => {
            if !strict {
                return Err(RequestError::at(
                    "text.format.strict must be true",
                    "text.format.strict",
                ));
            }
            require_schema_object(&schema, "text.format.schema")?;
            Ok(OutputFormat::JsonSchema { name, schema })
        }
    }
}

#[derive(Clone, Copy)]
enum ToolDialect {
    Chat,
    Responses,
}

fn parse_tools(values: &[Value], dialect: ToolDialect) -> Result<Vec<ChatTool>, RequestError> {
    let mut tools = Vec::with_capacity(values.len());
    let mut names = HashSet::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let base = format!("tools[{index}]");
        let object = value
            .as_object()
            .ok_or_else(|| RequestError::at("tool definition must be an object", &base))?;
        let (function, function_path) = match dialect {
            ToolDialect::Chat => {
                reject_unknown_fields(object, &["type", "function"], &base)?;
                require_function_type(object.get("type"), &format!("{base}.type"))?;
                let path = format!("{base}.function");
                let function = object
                    .get("function")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        RequestError::at("tool function must be an object", path.clone())
                    })?;
                reject_unknown_fields(
                    function,
                    &["name", "description", "parameters", "strict"],
                    &path,
                )?;
                (function, path)
            }
            ToolDialect::Responses => {
                reject_unknown_fields(
                    object,
                    &["type", "name", "description", "parameters", "strict"],
                    &base,
                )?;
                require_function_type(object.get("type"), &format!("{base}.type"))?;
                (object, base.clone())
            }
        };
        let name_path = format!("{function_path}.name");
        let name = require_nonempty_string(function.get("name"), &name_path)?.to_string();
        if !names.insert(name.clone()) {
            return Err(RequestError::at("tool names must be unique", name_path));
        }
        let description = match function.get("description") {
            None => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(RequestError::at(
                    "tool description must be a string",
                    format!("{function_path}.description"),
                ));
            }
        };
        let parameters_path = format!("{function_path}.parameters");
        let parameters = function
            .get("parameters")
            .filter(|value| value.is_object())
            .cloned()
            .ok_or_else(|| {
                RequestError::at("tool parameters must be an object", parameters_path)
            })?;
        let strict = match function.get("strict") {
            None => None,
            Some(Value::Bool(value)) => Some(*value),
            Some(_) => {
                return Err(RequestError::at(
                    "tool strict must be a boolean",
                    format!("{function_path}.strict"),
                ));
            }
        };
        tools.push(ChatTool {
            tool_type: "function".to_string(),
            function: ChatToolFunction {
                name,
                description,
                parameters,
                strict,
            },
        });
    }
    Ok(tools)
}

fn require_function_type(value: Option<&Value>, param: &str) -> Result<(), RequestError> {
    if value.and_then(Value::as_str) == Some("function") {
        Ok(())
    } else {
        Err(RequestError::at("only function tools are supported", param))
    }
}

fn parse_tool_choice(value: Option<&Value>) -> Result<ToolChoice, RequestError> {
    match value {
        None => Ok(ToolChoice::Auto),
        Some(Value::String(value)) if value == "auto" => Ok(ToolChoice::Auto),
        Some(Value::String(value)) if value == "none" => Ok(ToolChoice::None),
        _ => Err(RequestError::at(
            "tool_choice must be \"auto\" or \"none\"",
            "tool_choice",
        )),
    }
}

fn parse_reasoning_effort(value: Option<&str>) -> Result<Option<ReasoningEffort>, RequestError> {
    match value {
        None => Ok(None),
        Some("low") => Ok(Some(ReasoningEffort::Low)),
        Some("medium") => Ok(Some(ReasoningEffort::Medium)),
        Some("xhigh") => Ok(Some(ReasoningEffort::XHigh)),
        Some(_) => Err(RequestError::at(
            "reasoning_effort must be \"low\", \"medium\", or \"xhigh\"",
            "reasoning_effort",
        )),
    }
}

fn parse_chat_messages(
    values: Vec<Value>,
    tools: &[ChatTool],
    image_params: &mut Vec<String>,
) -> Result<Vec<ChatMessage>, RequestError> {
    let declared = declared_names(tools);
    let mut history = HistoryState::default();
    let mut messages = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let base = format!("messages[{index}]");
        let role = value
            .as_object()
            .and_then(|object| object.get("role"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RequestError::at("message role must be a string", format!("{base}.role"))
            })?;
        if role == "system" && index != 0 {
            return Err(RequestError::at(
                "system messages are only allowed at index zero",
                format!("{base}.role"),
            ));
        }
        let message_object = value
            .as_object()
            .expect("role lookup already required an object");
        let allowed_fields = match role {
            "system" | "user" => &["role", "content"][..],
            "assistant" => &["role", "content", "reasoning_content", "tool_calls"][..],
            "tool" => &["role", "content", "tool_call_id"][..],
            _ => {
                return Err(RequestError::at(
                    "unsupported message role",
                    format!("{base}.role"),
                ));
            }
        };
        reject_unknown_fields(message_object, allowed_fields, &base)?;
        if let Some(reasoning_content) = message_object.get("reasoning_content")
            && !reasoning_content.is_string()
        {
            return Err(RequestError::at(
                "assistant reasoning_content must be a string",
                format!("{base}.reasoning_content"),
            ));
        }
        if role != "user" {
            reject_chat_role_images(message_object.get("content"), &format!("{base}.content"))?;
        }
        let tool_calls_field_present = message_object.contains_key("tool_calls");
        let wire: ChatWireMessage = serde_json::from_value(value).map_err(|error| {
            RequestError::at(format!("invalid chat message: {error}"), base.clone())
        })?;
        let message = match wire {
            ChatWireMessage::System { content } => ordinary_message("system", content, &base)?,
            ChatWireMessage::User { content } => ChatMessage {
                reasoning_content: None,
                role: "user".to_string(),
                content: Some(parse_chat_user_content(
                    &content,
                    &format!("{base}.content"),
                    image_params,
                )?),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatWireMessage::Assistant {
                content,
                tool_calls,
                reasoning_content,
            } => {
                let calls_present = tool_calls_field_present;
                let calls = parse_chat_tool_calls(
                    tool_calls.unwrap_or_default(),
                    index,
                    &declared,
                    &mut history,
                )?;
                if calls_present && calls.is_empty() {
                    return Err(RequestError::at(
                        "assistant tool_calls must not be empty",
                        format!("{base}.tool_calls"),
                    ));
                }
                let content = match content {
                    Value::Null => None,
                    Value::String(value) => Some(ChatMessageContent::Text(value)),
                    _ => {
                        return Err(RequestError::at(
                            "assistant content must be a string or null",
                            format!("{base}.content"),
                        ));
                    }
                };
                if content
                    .as_ref()
                    .and_then(|content| match content {
                        ChatMessageContent::Text(text) => Some(text.as_str()),
                        ChatMessageContent::Parts(_) => None,
                    })
                    .unwrap_or_default()
                    .is_empty()
                    && calls.is_empty()
                {
                    return Err(RequestError::at(
                        "assistant content may be empty only when tool_calls are present",
                        format!("{base}.content"),
                    ));
                }
                ChatMessage {
                    reasoning_content,
                    role: "assistant".to_string(),
                    content,
                    tool_calls: calls,
                    tool_call_id: None,
                }
            }
            ChatWireMessage::Tool {
                content,
                tool_call_id,
            } => {
                let content = require_nonempty_string_value(&content, &format!("{base}.content"))?;
                let call_id =
                    require_nonempty_string_value(&tool_call_id, &format!("{base}.tool_call_id"))?;
                history.resolve(&call_id, format!("{base}.tool_call_id"))?;
                ChatMessage {
                    reasoning_content: None,
                    role: "tool".to_string(),
                    content: Some(ChatMessageContent::Text(content)),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(call_id),
                }
            }
        };
        messages.push(message);
    }
    history.finish()?;
    Ok(messages)
}

fn parse_chat_user_content(
    content: &Value,
    base: &str,
    image_params: &mut Vec<String>,
) -> Result<ChatMessageContent, RequestError> {
    match content {
        Value::String(text) if !text.is_empty() => Ok(ChatMessageContent::Text(text.clone())),
        Value::String(_) => Err(RequestError::at("user content must not be empty", base)),
        Value::Array(parts) => {
            if parts.is_empty() {
                return Err(RequestError::at(
                    "user content parts must not be empty",
                    base,
                ));
            }
            let mut normalized = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                let part_base = format!("{base}[{index}]");
                let object = part.as_object().ok_or_else(|| {
                    RequestError::at("content part must be an object", &part_base)
                })?;
                let part_type =
                    require_nonempty_string(object.get("type"), &format!("{part_base}.type"))?;
                match part_type {
                    "text" => {
                        reject_unknown_fields(object, &["type", "text"], &part_base)?;
                        let text = require_nonempty_string(
                            object.get("text"),
                            &format!("{part_base}.text"),
                        )?;
                        normalized.push(ChatContentPart::Text {
                            text: text.to_string(),
                        });
                    }
                    "image_url" => {
                        reject_unknown_fields(object, &["type", "image_url"], &part_base)?;
                        let image_base = format!("{part_base}.image_url");
                        let image = object
                            .get("image_url")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                RequestError::at("image_url must be an object", &image_base)
                            })?;
                        reject_unknown_fields(image, &["url", "detail"], &image_base)?;
                        let url_path = format!("{image_base}.url");
                        let url = require_nonempty_string(image.get("url"), &url_path)?;
                        validate_data_image_uri(url, &url_path)?;
                        let detail_path = format!("{image_base}.detail");
                        let detail = match image.get("detail") {
                            None => "auto",
                            Some(Value::String(detail)) if detail == "auto" => "auto",
                            _ => {
                                return Err(RequestError::at(
                                    "image detail must be \"auto\"",
                                    detail_path,
                                ));
                            }
                        };
                        record_image_param(image_params, &url_path)?;
                        normalized.push(ChatContentPart::ImageUrl {
                            image_url: ChatImageUrl {
                                url: url.to_string(),
                                detail: detail.to_string(),
                            },
                        });
                    }
                    _ => {
                        return Err(RequestError::at(
                            "unsupported content part type",
                            format!("{part_base}.type"),
                        ));
                    }
                }
            }
            Ok(ChatMessageContent::Parts(normalized))
        }
        _ => Err(RequestError::at(
            "user content must be a string or an array of content parts",
            base,
        )),
    }
}

fn parse_chat_tool_calls(
    values: Vec<Value>,
    message_index: usize,
    declared: &HashSet<&str>,
    history: &mut HistoryState,
) -> Result<Vec<ChatToolCall>, RequestError> {
    let mut calls = Vec::with_capacity(values.len());
    for (call_index, value) in values.into_iter().enumerate() {
        let base = format!("messages[{message_index}].tool_calls[{call_index}]");
        let object = value
            .as_object()
            .ok_or_else(|| RequestError::at("tool call must be an object", &base))?;
        reject_unknown_fields(object, &["id", "type", "function"], &base)?;
        let id = require_nonempty_string(object.get("id"), &format!("{base}.id"))?.to_string();
        require_function_type(object.get("type"), &format!("{base}.type"))?;
        let function_path = format!("{base}.function");
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                RequestError::at("tool call function must be an object", &function_path)
            })?;
        reject_unknown_fields(function, &["name", "arguments"], &function_path)?;
        let name_path = format!("{function_path}.name");
        let name = require_nonempty_string(function.get("name"), &name_path)?.to_string();
        require_declared(&name, declared, &name_path)?;
        let arguments_path = format!("{function_path}.arguments");
        let arguments_text = require_nonempty_string(function.get("arguments"), &arguments_path)?;
        let arguments: Value = serde_json::from_str(arguments_text).map_err(|_| {
            RequestError::at("tool call arguments must be valid JSON", &arguments_path)
        })?;
        if !arguments.is_object() {
            return Err(RequestError::at(
                "tool call arguments must decode to an object",
                arguments_path,
            ));
        }
        history.add(&id, format!("{base}.id"))?;
        calls.push(ChatToolCall {
            id,
            tool_type: "function".to_string(),
            function: ChatToolCallFunction { name, arguments },
        });
    }
    Ok(calls)
}

fn parse_responses_items(
    values: Vec<Value>,
    tools: &[ChatTool],
    image_params: &mut Vec<String>,
) -> Result<Vec<ChatMessage>, RequestError> {
    if values.is_empty() {
        return Err(RequestError::at(
            "input messages must not be empty",
            "input",
        ));
    }
    let declared = declared_names(tools);
    let mut history = HistoryState::default();
    let mut messages = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let base = format!("input[{index}]");
        let object = value
            .as_object()
            .ok_or_else(|| RequestError::at("input item must be an object", &base))?;
        let item_type = object.get("type").and_then(Value::as_str);
        let allowed_fields = match item_type {
            Some("function_call") => &["type", "id", "call_id", "name", "arguments"][..],
            Some("function_call_output") => &["type", "call_id", "output"][..],
            Some(_) => &["type"][..],
            None => &["role", "content"][..],
        };
        reject_unknown_fields(object, allowed_fields, &base)?;
        match item_type {
            Some("function_call") => {
                let wire: ResponsesFunctionCallWire =
                    serde_json::from_value(Value::Object(object.clone())).map_err(|error| {
                        RequestError::at(format!("invalid function call item: {error}"), &base)
                    })?;
                debug_assert_eq!(wire.item_type, "function_call");
                let _item_id = require_nonempty_string_value(&wire.id, &format!("{base}.id"))?;
                let call_id =
                    require_nonempty_string_value(&wire.call_id, &format!("{base}.call_id"))?;
                let name = require_nonempty_string_value(&wire.name, &format!("{base}.name"))?;
                require_declared(&name, &declared, &format!("{base}.name"))?;
                let arguments_text =
                    require_nonempty_string_value(&wire.arguments, &format!("{base}.arguments"))?;
                let arguments: Value = serde_json::from_str(&arguments_text).map_err(|_| {
                    RequestError::at(
                        "function call arguments must be valid JSON",
                        format!("{base}.arguments"),
                    )
                })?;
                if !arguments.is_object() {
                    return Err(RequestError::at(
                        "function call arguments must decode to an object",
                        format!("{base}.arguments"),
                    ));
                }
                history.add(&call_id, format!("{base}.call_id"))?;
                messages.push(ChatMessage {
                    reasoning_content: None,
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: vec![ChatToolCall {
                        id: call_id.clone(),
                        tool_type: "function".to_string(),
                        function: ChatToolCallFunction { name, arguments },
                    }],
                    tool_call_id: None,
                });
            }
            Some("function_call_output") => {
                let wire: ResponsesFunctionOutputWire =
                    serde_json::from_value(Value::Object(object.clone())).map_err(|error| {
                        RequestError::at(format!("invalid function output item: {error}"), &base)
                    })?;
                debug_assert_eq!(wire.item_type, "function_call_output");
                let call_id =
                    require_nonempty_string_value(&wire.call_id, &format!("{base}.call_id"))?;
                let output = require_string_value(&wire.output, &format!("{base}.output"))?;
                history.resolve(&call_id, format!("{base}.call_id"))?;
                messages.push(ChatMessage {
                    reasoning_content: None,
                    role: "tool".to_string(),
                    content: Some(ChatMessageContent::Text(output)),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(call_id),
                });
            }
            Some(_) => {
                return Err(RequestError::at(
                    "unsupported input item type",
                    format!("{base}.type"),
                ));
            }
            None => {
                let wire: ResponsesMessageWire =
                    serde_json::from_value(Value::Object(object.clone())).map_err(|error| {
                        RequestError::at(format!("invalid input message: {error}"), &base)
                    })?;
                if wire.role == "system" && index != 0 {
                    return Err(RequestError::at(
                        "system messages are only allowed at index zero",
                        format!("{base}.role"),
                    ));
                }
                if !matches!(wire.role.as_str(), "system" | "user" | "assistant") {
                    return Err(RequestError::at(
                        "unsupported input message role",
                        format!("{base}.role"),
                    ));
                }
                let content = if wire.role == "user" {
                    parse_responses_user_content(
                        &wire.content,
                        &format!("{base}.content"),
                        image_params,
                    )?
                } else {
                    if let Value::Array(parts) = &wire.content
                        && let Some((part_index, _)) = parts.iter().enumerate().find(|(_, part)| {
                            part.get("type").and_then(Value::as_str) == Some("input_image")
                        })
                    {
                        return Err(RequestError::at(
                            "image inputs are only allowed on user messages",
                            format!("{base}.content[{part_index}].image_url"),
                        ));
                    }
                    let text =
                        require_nonempty_string_value(&wire.content, &format!("{base}.content"))?;
                    ChatMessageContent::Text(text)
                };
                messages.push(ChatMessage {
                    reasoning_content: None,
                    role: wire.role,
                    content: Some(content),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
            }
        }
    }
    history.finish()?;
    Ok(messages)
}

fn parse_responses_user_content(
    content: &Value,
    base: &str,
    image_params: &mut Vec<String>,
) -> Result<ChatMessageContent, RequestError> {
    match content {
        Value::String(text) if !text.is_empty() => Ok(ChatMessageContent::Text(text.clone())),
        Value::String(_) => Err(RequestError::at("user content must not be empty", base)),
        Value::Array(parts) => {
            if parts.is_empty() {
                return Err(RequestError::at(
                    "user content parts must not be empty",
                    base,
                ));
            }
            let mut normalized = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                let part_base = format!("{base}[{index}]");
                let object = part.as_object().ok_or_else(|| {
                    RequestError::at("content part must be an object", &part_base)
                })?;
                let part_type =
                    require_nonempty_string(object.get("type"), &format!("{part_base}.type"))?;
                match part_type {
                    "input_text" => {
                        reject_unknown_fields(object, &["type", "text"], &part_base)?;
                        let text = require_nonempty_string(
                            object.get("text"),
                            &format!("{part_base}.text"),
                        )?;
                        normalized.push(ChatContentPart::Text {
                            text: text.to_string(),
                        });
                    }
                    "input_image" => {
                        reject_unknown_fields(
                            object,
                            &["type", "image_url", "detail"],
                            &part_base,
                        )?;
                        let url_path = format!("{part_base}.image_url");
                        let url = require_nonempty_string(object.get("image_url"), &url_path)?;
                        validate_data_image_uri(url, &url_path)?;
                        let detail_path = format!("{part_base}.detail");
                        let detail = match object.get("detail") {
                            None => "auto",
                            Some(Value::String(detail)) if detail == "auto" => "auto",
                            _ => {
                                return Err(RequestError::at(
                                    "image detail must be \"auto\"",
                                    detail_path,
                                ));
                            }
                        };
                        record_image_param(image_params, &url_path)?;
                        normalized.push(ChatContentPart::ImageUrl {
                            image_url: ChatImageUrl {
                                url: url.to_string(),
                                detail: detail.to_string(),
                            },
                        });
                    }
                    _ => {
                        return Err(RequestError::at(
                            "unsupported content part type",
                            format!("{part_base}.type"),
                        ));
                    }
                }
            }
            Ok(ChatMessageContent::Parts(normalized))
        }
        _ => Err(RequestError::at(
            "user content must be a string or an array of input parts",
            base,
        )),
    }
}

fn validate_data_image_uri(url: &str, param: &str) -> Result<(), RequestError> {
    let payload = [
        "data:image/png;base64,",
        "data:image/jpeg;base64,",
        "data:image/webp;base64,",
    ]
    .into_iter()
    .find_map(|prefix| url.strip_prefix(prefix))
    .ok_or_else(|| {
        RequestError::at(
            "image_url must be a base64 data URI for PNG, JPEG, or WebP",
            param,
        )
    })?;
    if payload.is_empty() {
        return Err(RequestError::at(
            "image data URI payload must not be empty",
            param,
        ));
    }
    Ok(())
}

fn record_image_param(image_params: &mut Vec<String>, param: &str) -> Result<(), RequestError> {
    if image_params.len() == MAX_IMAGES_PER_REQUEST {
        return Err(RequestError::at(
            format!("requests may contain at most {MAX_IMAGES_PER_REQUEST} images"),
            param,
        ));
    }
    image_params.push(param.to_string());
    Ok(())
}

fn reject_chat_role_images(content: Option<&Value>, base: &str) -> Result<(), RequestError> {
    let Some(Value::Array(parts)) = content else {
        return Ok(());
    };
    if let Some((index, _)) = parts
        .iter()
        .enumerate()
        .find(|(_, part)| part.get("type").and_then(Value::as_str) == Some("image_url"))
    {
        return Err(RequestError::at(
            "image inputs are only allowed on user messages",
            format!("{base}[{index}].image_url.url"),
        ));
    }
    Ok(())
}

fn ordinary_message(role: &str, content: Value, base: &str) -> Result<ChatMessage, RequestError> {
    Ok(ChatMessage {
        reasoning_content: None,
        role: role.to_string(),
        content: Some(ChatMessageContent::Text(require_nonempty_string_value(
            &content,
            &format!("{base}.content"),
        )?)),
        tool_calls: Vec::new(),
        tool_call_id: None,
    })
}

#[derive(Default)]
struct HistoryState {
    seen: HashSet<String>,
    unresolved: HashMap<String, String>,
}

impl HistoryState {
    fn add(&mut self, id: &str, param: String) -> Result<(), RequestError> {
        if !self.seen.insert(id.to_string()) {
            return Err(RequestError::at("tool call IDs must be unique", param));
        }
        self.unresolved.insert(id.to_string(), param);
        Ok(())
    }

    fn resolve(&mut self, id: &str, param: String) -> Result<(), RequestError> {
        if self.unresolved.remove(id).is_some() {
            return Ok(());
        }
        if self.seen.contains(id) {
            Err(RequestError::at("duplicate tool result", param))
        } else {
            Err(RequestError::at("orphan tool result", param))
        }
    }

    fn finish(self) -> Result<(), RequestError> {
        if let Some((_, param)) = self.unresolved.into_iter().next() {
            Err(RequestError::at("tool call is missing a result", param))
        } else {
            Ok(())
        }
    }
}

fn declared_names(tools: &[ChatTool]) -> HashSet<&str> {
    tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect()
}

fn require_declared(name: &str, declared: &HashSet<&str>, param: &str) -> Result<(), RequestError> {
    if declared.contains(name) {
        Ok(())
    } else {
        Err(RequestError::at(
            "replayed tool call names must reference a declared function",
            param,
        ))
    }
}

fn require_nonempty_string<'a>(
    value: Option<&'a Value>,
    param: &str,
) -> Result<&'a str, RequestError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| RequestError::at("value must be a non-empty string", param))
}

fn require_string_value(value: &Value, param: &str) -> Result<String, RequestError> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| RequestError::at("content must be a string", param))
}

fn require_nonempty_string_value(value: &Value, param: &str) -> Result<String, RequestError> {
    require_nonempty_string(Some(value), param).map(str::to_string)
}

fn reject_unknown_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    base: &str,
) -> Result<(), RequestError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        Err(RequestError::at("unknown field", format!("{base}.{field}")))
    } else {
        Ok(())
    }
}

fn require_object(value: &Value) -> Result<&Map<String, Value>, RequestError> {
    value
        .as_object()
        .ok_or_else(|| RequestError::new("request body must be a JSON object", None))
}

fn reject_present(
    object: &Map<String, Value>,
    field: &str,
    message: &str,
) -> Result<(), RequestError> {
    if object.contains_key(field) {
        Err(RequestError::at(message, field))
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
        return Err(RequestError::at(
            "maximum output tokens must be greater than zero",
            "max_tokens",
        ));
    }
    if temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
        return Err(RequestError::at(
            "temperature must be finite and between 0 and 2",
            "temperature",
        ));
    }
    if top_p.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0) {
        return Err(RequestError::at(
            "top_p must be finite and greater than 0 and at most 1",
            "top_p",
        ));
    }
    Ok(())
}

fn require_schema_object(schema: &Value, param: &str) -> Result<(), RequestError> {
    if schema.is_object() {
        Ok(())
    } else {
        Err(RequestError::at("JSON Schema must be an object", param))
    }
}
