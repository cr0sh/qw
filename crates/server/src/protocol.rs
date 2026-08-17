use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use serde_json::{Map, Value};

use qw_runtime::{
    ChatContentPart, ChatCustomToolCall, ChatFile, ChatImageUrl, ChatInputAudio, ChatMessage,
    ChatMessageContent, ChatPromptCacheBreakpoint, ChatTool, ChatToolCall, ChatToolCallFunction,
    ChatToolFunction,
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
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
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
    Developer {
        content: Value,
        name: Option<String>,
    },
    System {
        content: Value,
        name: Option<String>,
    },
    User {
        content: Value,
        name: Option<String>,
    },
    Assistant {
        #[serde(default)]
        content: Value,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<Value>>,
        #[serde(rename = "audio")]
        _audio: Option<Value>,
        function_call: Option<Value>,
        name: Option<String>,
        #[serde(rename = "refusal")]
        _refusal: Option<String>,
    },
    Tool {
        content: Value,
        tool_call_id: Value,
    },
    Function {
        #[serde(default)]
        content: Value,
        name: Value,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatWire {
    model: String,
    messages: Vec<Value>,
    stream: Option<bool>,
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
    #[serde(rename = "n")]
    _n: Option<usize>,
    #[serde(rename = "stop")]
    _stop: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    functions: Option<Vec<Value>>,
    function_call: Option<Value>,
    #[serde(rename = "audio")]
    _audio: Option<Value>,
    #[serde(rename = "frequency_penalty")]
    _frequency_penalty: Option<f32>,
    #[serde(rename = "logit_bias")]
    _logit_bias: Option<Map<String, Value>>,
    #[serde(rename = "logprobs")]
    _logprobs: Option<bool>,
    #[serde(rename = "metadata")]
    _metadata: Option<Map<String, Value>>,
    #[serde(rename = "name")]
    _name: Option<String>,
    #[serde(rename = "modalities")]
    _modalities: Option<Vec<String>>,
    #[serde(rename = "moderation")]
    _moderation: Option<Value>,
    #[serde(rename = "prediction")]
    _prediction: Option<Value>,
    #[serde(rename = "presence_penalty")]
    _presence_penalty: Option<f32>,
    #[serde(rename = "prompt_cache_key")]
    _prompt_cache_key: Option<String>,
    #[serde(rename = "prompt_cache_options")]
    _prompt_cache_options: Option<Value>,
    #[serde(rename = "prompt_cache_retention")]
    _prompt_cache_retention: Option<String>,
    #[serde(rename = "safety_identifier")]
    _safety_identifier: Option<String>,
    #[serde(rename = "service_tier")]
    _service_tier: Option<String>,
    #[serde(rename = "store")]
    _store: Option<bool>,
    #[serde(rename = "top_logprobs")]
    _top_logprobs: Option<usize>,
    #[serde(rename = "user")]
    _user: Option<String>,
    #[serde(rename = "verbosity")]
    _verbosity: Option<String>,
    #[serde(rename = "web_search_options")]
    _web_search_options: Option<Value>,
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
    #[serde(rename = "include_obfuscation")]
    _include_obfuscation: Option<bool>,
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
    #[serde(rename = "description")]
    _description: Option<String>,
    #[serde(rename = "strict")]
    _strict: Option<bool>,
    schema: Option<Value>,
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
    if let Some(reasoning_effort) = object.get("reasoning_effort")
        && !reasoning_effort.is_null()
        && !reasoning_effort.is_string()
    {
        return Err(RequestError::at(
            "reasoning_effort must be a string or null",
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
    let mut tool_values = wire.tools.unwrap_or_default();
    tool_values.extend(
        wire.functions
            .unwrap_or_default()
            .into_iter()
            .map(|function| json_object_tool(function)),
    );
    let tools = parse_tools(&tool_values, ToolDialect::Chat)?;
    let tool_choice = parse_tool_choice(wire.tool_choice.as_ref().or(wire.function_call.as_ref()))?;
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
        stream: wire.stream.unwrap_or(false),
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
                name: None,
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
            let schema = json_schema
                .schema
                .unwrap_or_else(|| Value::Object(Map::new()));
            require_schema_object(&schema, "response_format.json_schema.schema")?;
            Ok(OutputFormat::JsonSchema {
                name: json_schema.name,
                schema,
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
fn json_object_tool(function: Value) -> Value {
    let mut tool = Map::new();
    tool.insert("type".to_string(), Value::String("function".to_string()));
    tool.insert("function".to_string(), function);
    Value::Object(tool)
}
fn parse_custom_tool(object: &Map<String, Value>, base: &str) -> Result<(), RequestError> {
    reject_unknown_fields(object, &["type", "custom"], base)?;
    let custom_path = format!("{base}.custom");
    let custom = object
        .get("custom")
        .and_then(Value::as_object)
        .ok_or_else(|| RequestError::at("custom tool must be an object", &custom_path))?;
    reject_unknown_fields(custom, &["name", "description", "format"], &custom_path)?;
    require_nonempty_string(custom.get("name"), &format!("{custom_path}.name"))?;
    if let Some(description) = custom.get("description")
        && !description.is_null()
        && !description.is_string()
    {
        return Err(RequestError::at(
            "custom tool description must be a string",
            format!("{custom_path}.description"),
        ));
    }
    let Some(format) = custom.get("format") else {
        return Ok(());
    };
    let format_path = format!("{custom_path}.format");
    let format = format
        .as_object()
        .ok_or_else(|| RequestError::at("custom tool format must be an object", &format_path))?;
    match format.get("type").and_then(Value::as_str) {
        Some("text") => reject_unknown_fields(format, &["type"], &format_path),
        Some("grammar") => {
            reject_unknown_fields(format, &["type", "grammar"], &format_path)?;
            let grammar_path = format!("{format_path}.grammar");
            let grammar = format
                .get("grammar")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    RequestError::at("custom tool grammar must be an object", &grammar_path)
                })?;
            reject_unknown_fields(grammar, &["definition", "syntax"], &grammar_path)?;
            require_nonempty_string(
                grammar.get("definition"),
                &format!("{grammar_path}.definition"),
            )?;
            match grammar.get("syntax").and_then(Value::as_str) {
                Some("lark" | "regex") => Ok(()),
                _ => Err(RequestError::at(
                    "custom tool grammar syntax must be \"lark\" or \"regex\"",
                    format!("{grammar_path}.syntax"),
                )),
            }
        }
        _ => Err(RequestError::at(
            "custom tool format type must be \"text\" or \"grammar\"",
            format!("{format_path}.type"),
        )),
    }
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
                let tool_type =
                    require_nonempty_string(object.get("type"), &format!("{base}.type"))?;
                if tool_type == "custom" {
                    parse_custom_tool(object, &base)?;
                    continue;
                }
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
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(RequestError::at(
                    "tool description must be a string",
                    format!("{function_path}.description"),
                ));
            }
        };
        let parameters_path = format!("{function_path}.parameters");
        let parameters = match function.get("parameters") {
            None | Some(Value::Null) => Value::Object(Map::new()),
            Some(value) if value.is_object() => value.clone(),
            Some(_) => {
                return Err(RequestError::at(
                    "tool parameters must be an object",
                    parameters_path,
                ));
            }
        };
        let strict = match function.get("strict") {
            None | Some(Value::Null) => None,
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
        Some(Value::String(value)) if value == "auto" || value == "required" => {
            Ok(ToolChoice::Auto)
        }
        Some(Value::String(value)) if value == "none" => Ok(ToolChoice::None),
        Some(Value::Object(object))
            if object
                .get("name")
                .or_else(|| object.get("function").and_then(|value| value.get("name")))
                .or_else(|| object.get("custom").and_then(|value| value.get("name")))
                .and_then(Value::as_str)
                .is_some_and(|name| !name.is_empty()) =>
        {
            Ok(ToolChoice::Auto)
        }
        Some(Value::Object(object))
            if object.get("type").and_then(Value::as_str) == Some("allowed_tools")
                && object.get("allowed_tools").is_some_and(Value::is_object) =>
        {
            Ok(ToolChoice::Auto)
        }
        _ => Err(RequestError::at(
            "tool_choice is not a documented tool selection",
            "tool_choice",
        )),
    }
}

fn parse_reasoning_effort(value: Option<&str>) -> Result<Option<ReasoningEffort>, RequestError> {
    match value {
        None => Ok(None),
        Some("none") => Ok(Some(ReasoningEffort::None)),
        Some("minimal") => Ok(Some(ReasoningEffort::Minimal)),
        Some("low") => Ok(Some(ReasoningEffort::Low)),
        Some("medium") => Ok(Some(ReasoningEffort::Medium)),
        Some("high") => Ok(Some(ReasoningEffort::High)),
        Some("xhigh") => Ok(Some(ReasoningEffort::XHigh)),
        Some("max") => Ok(Some(ReasoningEffort::Max)),
        Some(_) => Err(RequestError::at(
            "reasoning_effort is not a documented value",
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
        let message_object = value
            .as_object()
            .ok_or_else(|| RequestError::at("message must be an object", &base))?;
        let role = message_object
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RequestError::at("message role must be a string", format!("{base}.role"))
            })?;
        let allowed_fields = match role {
            "developer" | "system" | "user" => &["role", "content", "name"][..],
            "assistant" => &[
                "role",
                "audio",
                "content",
                "function_call",
                "name",
                "reasoning_content",
                "refusal",
                "tool_calls",
            ][..],
            "tool" => &["role", "content", "tool_call_id"][..],
            "function" => &["role", "content", "name"][..],
            _ => {
                return Err(RequestError::at(
                    "unsupported message role",
                    format!("{base}.role"),
                ));
            }
        };
        reject_unknown_fields(message_object, allowed_fields, &base)?;
        if let Some(reasoning_content) = message_object.get("reasoning_content")
            && !reasoning_content.is_null()
            && !reasoning_content.is_string()
        {
            return Err(RequestError::at(
                "assistant reasoning_content must be a string",
                format!("{base}.reasoning_content"),
            ));
        }
        let tool_calls_field_present = message_object.contains_key("tool_calls");
        let wire: ChatWireMessage = serde_json::from_value(value).map_err(|error| {
            RequestError::at(format!("invalid chat message: {error}"), base.clone())
        })?;
        let message = match wire {
            ChatWireMessage::Developer { content, name } => {
                ordinary_message("developer", content, name, &base)?
            }
            ChatWireMessage::System { content, name } => {
                ordinary_message("system", content, name, &base)?
            }
            ChatWireMessage::User { content, name } => ChatMessage {
                role: "user".to_string(),
                name,
                content: Some(parse_chat_user_content(
                    &content,
                    &format!("{base}.content"),
                    image_params,
                )?),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            ChatWireMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
                _audio: _,
                function_call,
                name,
                _refusal: refusal,
            } => {
                let mut calls = parse_chat_tool_calls(
                    tool_calls.unwrap_or_default(),
                    index,
                    &declared,
                    &mut history,
                )?;
                if tool_calls_field_present && calls.is_empty() {
                    return Err(RequestError::at(
                        "assistant tool_calls must not be empty",
                        format!("{base}.tool_calls"),
                    ));
                }
                if let Some(function_call) = function_call.filter(|value| !value.is_null()) {
                    calls.push(parse_legacy_function_call(
                        &function_call,
                        index,
                        &declared,
                    )?);
                }
                let content =
                    parse_assistant_content(content, refusal, &format!("{base}.content"))?;
                let content_empty = match content.as_ref() {
                    None => true,
                    Some(ChatMessageContent::Text(text)) => text.is_empty(),
                    Some(ChatMessageContent::Parts(parts)) => parts.is_empty(),
                };
                if content_empty && calls.is_empty() {
                    return Err(RequestError::at(
                        "assistant content may be empty only when tool_calls are present",
                        format!("{base}.content"),
                    ));
                }
                ChatMessage {
                    role: "assistant".to_string(),
                    name,
                    content,
                    reasoning_content,
                    tool_calls: calls,
                    tool_call_id: None,
                }
            }
            ChatWireMessage::Tool {
                content,
                tool_call_id,
            } => {
                let call_id =
                    require_nonempty_string_value(&tool_call_id, &format!("{base}.tool_call_id"))?;
                history.resolve(&call_id, format!("{base}.tool_call_id"))?;
                ChatMessage {
                    role: "tool".to_string(),
                    name: None,
                    content: Some(parse_chat_text_content(
                        &content,
                        &format!("{base}.content"),
                    )?),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: Some(call_id),
                }
            }
            ChatWireMessage::Function { content, name } => ChatMessage {
                role: "function".to_string(),
                name: Some(require_nonempty_string_value(
                    &name,
                    &format!("{base}.name"),
                )?),
                content: match content {
                    Value::Null => None,
                    Value::String(content) => Some(ChatMessageContent::Text(content)),
                    _ => {
                        return Err(RequestError::at(
                            "function content must be a string or null",
                            format!("{base}.content"),
                        ));
                    }
                },
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        };
        messages.push(message);
    }
    history.finish()?;
    Ok(messages)
}

fn parse_prompt_cache_breakpoint(
    object: &Map<String, Value>,
    base: &str,
) -> Result<Option<ChatPromptCacheBreakpoint>, RequestError> {
    let Some(value) = object.get("prompt_cache_breakpoint") else {
        return Ok(None);
    };
    let path = format!("{base}.prompt_cache_breakpoint");
    let breakpoint = value
        .as_object()
        .ok_or_else(|| RequestError::at("prompt cache breakpoint must be an object", &path))?;
    reject_unknown_fields(breakpoint, &["mode"], &path)?;
    if breakpoint.get("mode").and_then(Value::as_str) != Some("explicit") {
        return Err(RequestError::at(
            "prompt cache breakpoint mode must be \"explicit\"",
            format!("{path}.mode"),
        ));
    }
    Ok(Some(ChatPromptCacheBreakpoint {
        mode: "explicit".to_string(),
    }))
}

fn parse_chat_text_content(
    content: &Value,
    base: &str,
) -> Result<ChatMessageContent, RequestError> {
    match content {
        Value::String(text) if !text.is_empty() => Ok(ChatMessageContent::Text(text.clone())),
        Value::String(_) => Err(RequestError::at("message content must not be empty", base)),
        Value::Array(parts) if !parts.is_empty() => {
            let mut normalized = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                let part_base = format!("{base}[{index}]");
                let object = part.as_object().ok_or_else(|| {
                    RequestError::at("content part must be an object", &part_base)
                })?;
                reject_unknown_fields(
                    object,
                    &["type", "text", "prompt_cache_breakpoint"],
                    &part_base,
                )?;
                if object.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(RequestError::at(
                        "only text content parts are allowed for this role",
                        format!("{part_base}.type"),
                    ));
                }
                let text =
                    require_nonempty_string(object.get("text"), &format!("{part_base}.text"))?;
                normalized.push(ChatContentPart::Text {
                    text: text.to_string(),
                    prompt_cache_breakpoint: parse_prompt_cache_breakpoint(object, &part_base)?,
                });
            }
            Ok(ChatMessageContent::Parts(normalized))
        }
        Value::Array(_) => Err(RequestError::at("content parts must not be empty", base)),
        _ => Err(RequestError::at(
            "message content must be a string or an array of text parts",
            base,
        )),
    }
}

fn parse_assistant_content(
    content: Value,
    refusal: Option<String>,
    base: &str,
) -> Result<Option<ChatMessageContent>, RequestError> {
    let mut content = match content {
        Value::Null => None,
        Value::String(value) => Some(ChatMessageContent::Text(value)),
        Value::Array(parts) if !parts.is_empty() => {
            let mut normalized = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                let part_base = format!("{base}[{index}]");
                let object = part.as_object().ok_or_else(|| {
                    RequestError::at("assistant content part must be an object", &part_base)
                })?;
                match object.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        reject_unknown_fields(
                            object,
                            &["type", "text", "prompt_cache_breakpoint"],
                            &part_base,
                        )?;
                        normalized.push(ChatContentPart::Text {
                            text: require_nonempty_string(
                                object.get("text"),
                                &format!("{part_base}.text"),
                            )?
                            .to_string(),
                            prompt_cache_breakpoint: parse_prompt_cache_breakpoint(
                                object, &part_base,
                            )?,
                        });
                    }
                    Some("refusal") => {
                        reject_unknown_fields(object, &["type", "refusal"], &part_base)?;
                        normalized.push(ChatContentPart::Refusal {
                            refusal: require_nonempty_string(
                                object.get("refusal"),
                                &format!("{part_base}.refusal"),
                            )?
                            .to_string(),
                        });
                    }
                    _ => {
                        return Err(RequestError::at(
                            "unsupported assistant content part type",
                            format!("{part_base}.type"),
                        ));
                    }
                }
            }
            Some(ChatMessageContent::Parts(normalized))
        }
        Value::Array(_) => {
            return Err(RequestError::at(
                "assistant content parts must not be empty",
                base,
            ));
        }
        _ => {
            return Err(RequestError::at(
                "assistant content must be a string, array, or null",
                base,
            ));
        }
    };
    if let Some(refusal) = refusal {
        let refusal = ChatContentPart::Refusal { refusal };
        content = Some(match content {
            None => ChatMessageContent::Parts(vec![refusal]),
            Some(ChatMessageContent::Text(text)) => ChatMessageContent::Parts(vec![
                ChatContentPart::Text {
                    text,
                    prompt_cache_breakpoint: None,
                },
                refusal,
            ]),
            Some(ChatMessageContent::Parts(mut parts)) => {
                parts.push(refusal);
                ChatMessageContent::Parts(parts)
            }
        });
    }
    Ok(content)
}

fn parse_chat_user_content(
    content: &Value,
    base: &str,
    image_params: &mut Vec<String>,
) -> Result<ChatMessageContent, RequestError> {
    match content {
        Value::String(text) if !text.is_empty() => Ok(ChatMessageContent::Text(text.clone())),
        Value::String(_) => Err(RequestError::at("user content must not be empty", base)),
        Value::Array(parts) if !parts.is_empty() => {
            let mut normalized = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                let part_base = format!("{base}[{index}]");
                let object = part.as_object().ok_or_else(|| {
                    RequestError::at("content part must be an object", &part_base)
                })?;
                let part_type =
                    require_nonempty_string(object.get("type"), &format!("{part_base}.type"))?;
                let prompt_cache_breakpoint = parse_prompt_cache_breakpoint(object, &part_base)?;
                match part_type {
                    "text" => {
                        reject_unknown_fields(
                            object,
                            &["type", "text", "prompt_cache_breakpoint"],
                            &part_base,
                        )?;
                        let text = require_nonempty_string(
                            object.get("text"),
                            &format!("{part_base}.text"),
                        )?;
                        normalized.push(ChatContentPart::Text {
                            text: text.to_string(),
                            prompt_cache_breakpoint,
                        });
                    }
                    "image_url" => {
                        reject_unknown_fields(
                            object,
                            &["type", "image_url", "prompt_cache_breakpoint"],
                            &part_base,
                        )?;
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
                        let detail = match image.get("detail") {
                            None => "auto",
                            Some(Value::String(detail))
                                if matches!(detail.as_str(), "auto" | "low" | "high") =>
                            {
                                detail
                            }
                            _ => {
                                return Err(RequestError::at(
                                    "image detail must be \"auto\", \"low\", or \"high\"",
                                    format!("{image_base}.detail"),
                                ));
                            }
                        };
                        record_image_param(image_params, &url_path)?;
                        normalized.push(ChatContentPart::ImageUrl {
                            image_url: ChatImageUrl {
                                url: url.to_string(),
                                detail: detail.to_string(),
                            },
                            prompt_cache_breakpoint,
                        });
                    }
                    "input_audio" => {
                        reject_unknown_fields(
                            object,
                            &["type", "input_audio", "prompt_cache_breakpoint"],
                            &part_base,
                        )?;
                        let audio_path = format!("{part_base}.input_audio");
                        let audio = object
                            .get("input_audio")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                RequestError::at("input_audio must be an object", &audio_path)
                            })?;
                        reject_unknown_fields(audio, &["data", "format"], &audio_path)?;
                        let data = require_nonempty_string(
                            audio.get("data"),
                            &format!("{audio_path}.data"),
                        )?;
                        let format = match audio.get("format").and_then(Value::as_str) {
                            Some(format @ ("wav" | "mp3")) => format,
                            _ => {
                                return Err(RequestError::at(
                                    "input audio format must be \"wav\" or \"mp3\"",
                                    format!("{audio_path}.format"),
                                ));
                            }
                        };
                        normalized.push(ChatContentPart::InputAudio {
                            input_audio: ChatInputAudio {
                                data: data.to_string(),
                                format: format.to_string(),
                            },
                            prompt_cache_breakpoint,
                        });
                    }
                    "file" => {
                        reject_unknown_fields(
                            object,
                            &["type", "file", "prompt_cache_breakpoint"],
                            &part_base,
                        )?;
                        let file_path = format!("{part_base}.file");
                        let file =
                            object
                                .get("file")
                                .and_then(Value::as_object)
                                .ok_or_else(|| {
                                    RequestError::at("file must be an object", &file_path)
                                })?;
                        reject_unknown_fields(
                            file,
                            &["file_data", "file_id", "filename"],
                            &file_path,
                        )?;
                        let string_field = |field: &str| -> Result<Option<String>, RequestError> {
                            match file.get(field) {
                                None | Some(Value::Null) => Ok(None),
                                Some(Value::String(value)) => Ok(Some(value.clone())),
                                Some(_) => Err(RequestError::at(
                                    format!("{field} must be a string"),
                                    format!("{file_path}.{field}"),
                                )),
                            }
                        };
                        let file_data = string_field("file_data")?;
                        let file_id = string_field("file_id")?;
                        normalized.push(ChatContentPart::File {
                            file: ChatFile {
                                file_data,
                                file_id,
                                filename: string_field("filename")?,
                            },
                            prompt_cache_breakpoint,
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
        Value::Array(_) => Err(RequestError::at(
            "user content parts must not be empty",
            base,
        )),
        _ => Err(RequestError::at(
            "user content must be a string or an array of content parts",
            base,
        )),
    }
}

fn parse_legacy_function_call(
    value: &Value,
    message_index: usize,
    declared: &HashSet<&str>,
) -> Result<ChatToolCall, RequestError> {
    let base = format!("messages[{message_index}].function_call");
    let function = value
        .as_object()
        .ok_or_else(|| RequestError::at("function_call must be an object", &base))?;
    reject_unknown_fields(function, &["name", "arguments"], &base)?;
    let name = require_nonempty_string(function.get("name"), &format!("{base}.name"))?.to_string();
    if !declared.is_empty() {
        require_declared(&name, declared, &format!("{base}.name"))?;
    }
    let arguments_path = format!("{base}.arguments");
    let arguments_text = require_nonempty_string(function.get("arguments"), &arguments_path)?;
    let arguments: Value = serde_json::from_str(arguments_text).map_err(|_| {
        RequestError::at(
            "function call arguments must be valid JSON",
            &arguments_path,
        )
    })?;
    if !arguments.is_object() {
        return Err(RequestError::at(
            "function call arguments must decode to an object",
            arguments_path,
        ));
    }
    Ok(ChatToolCall::Function {
        id: format!("legacy_call_{message_index}"),
        function: ChatToolCallFunction { name, arguments },
    })
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
        let id = require_nonempty_string(object.get("id"), &format!("{base}.id"))?.to_string();
        let call = match object.get("type").and_then(Value::as_str) {
            Some("function") => {
                reject_unknown_fields(object, &["id", "type", "function"], &base)?;
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
                let arguments_text =
                    require_nonempty_string(function.get("arguments"), &arguments_path)?;
                let arguments: Value = serde_json::from_str(arguments_text).map_err(|_| {
                    RequestError::at("tool call arguments must be valid JSON", &arguments_path)
                })?;
                if !arguments.is_object() {
                    return Err(RequestError::at(
                        "tool call arguments must decode to an object",
                        arguments_path,
                    ));
                }
                ChatToolCall::Function {
                    id: id.clone(),
                    function: ChatToolCallFunction { name, arguments },
                }
            }
            Some("custom") => {
                reject_unknown_fields(object, &["id", "type", "custom"], &base)?;
                let custom_path = format!("{base}.custom");
                let custom = object
                    .get("custom")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        RequestError::at("custom tool call must be an object", &custom_path)
                    })?;
                reject_unknown_fields(custom, &["name", "input"], &custom_path)?;
                ChatToolCall::Custom {
                    id: id.clone(),
                    custom: ChatCustomToolCall {
                        name: require_nonempty_string(
                            custom.get("name"),
                            &format!("{custom_path}.name"),
                        )?
                        .to_string(),
                        input: require_string_value(
                            custom.get("input").ok_or_else(|| {
                                RequestError::at(
                                    "custom tool input is required",
                                    format!("{custom_path}.input"),
                                )
                            })?,
                            &format!("{custom_path}.input"),
                        )?,
                    },
                }
            }
            _ => {
                return Err(RequestError::at(
                    "tool call type must be \"function\" or \"custom\"",
                    format!("{base}.type"),
                ));
            }
        };
        history.add(&id, format!("{base}.id"))?;
        calls.push(call);
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
                    role: "assistant".to_string(),
                    name: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: vec![ChatToolCall::Function {
                        id: call_id.clone(),
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
                    role: "tool".to_string(),
                    name: None,
                    content: Some(ChatMessageContent::Text(output)),
                    reasoning_content: None,
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
                    role: wire.role,
                    name: None,
                    content: Some(content),
                    reasoning_content: None,
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
                            prompt_cache_breakpoint: None,
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
                            Some(Value::String(detail))
                                if matches!(detail.as_str(), "auto" | "low" | "high") =>
                            {
                                detail
                            }
                            _ => {
                                return Err(RequestError::at(
                                    "image detail must be \"auto\", \"low\", or \"high\"",
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
                            prompt_cache_breakpoint: None,
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

fn ordinary_message(
    role: &str,
    content: Value,
    name: Option<String>,
    base: &str,
) -> Result<ChatMessage, RequestError> {
    Ok(ChatMessage {
        role: role.to_string(),
        name,
        content: Some(parse_chat_text_content(
            &content,
            &format!("{base}.content"),
        )?),
        reasoning_content: None,
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
