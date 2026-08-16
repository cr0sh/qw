// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::Path;
use anyhow::{Context, Result};
use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, context};
use serde::Serialize;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text { text: String },
    ImageUrl { image_url: ChatImageUrl },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatImageUrl {
    pub url: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatContentRef<'a> {
    Text(&'a str),
    Image(&'a ChatImageUrl),
}

impl ChatMessage {
    pub fn text_content(&self) -> Option<&str> {
        match self.content.as_ref()? {
            ChatMessageContent::Text(text) => Some(text),
            ChatMessageContent::Parts(_) => None,
        }
    }

    pub fn image_urls(&self) -> impl Iterator<Item = &ChatImageUrl> {
        self.content
            .as_ref()
            .and_then(|content| match content {
                ChatMessageContent::Text(_) => None,
                ChatMessageContent::Parts(parts) => Some(parts.as_slice()),
            })
            .into_iter()
            .flatten()
            .filter_map(|part| match part {
                ChatContentPart::Text { .. } => None,
                ChatContentPart::ImageUrl { image_url } => Some(image_url),
            })
    }

    pub fn visit_content(&self, mut visitor: impl FnMut(ChatContentRef<'_>)) {
        match self.content.as_ref() {
            None => {}
            Some(ChatMessageContent::Text(text)) => visitor(ChatContentRef::Text(text)),
            Some(ChatMessageContent::Parts(parts)) => {
                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => {
                            visitor(ChatContentRef::Text(text));
                        }
                        ChatContentPart::ImageUrl { image_url } => {
                            visitor(ChatContentRef::Image(image_url));
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatToolFunction,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatToolCallFunction,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatToolCallFunction {
    pub name: String,
    pub arguments: JsonValue,
}

pub(crate) struct ChatTemplateProcessor {
    template: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplateProcessor {
    pub(crate) fn from_model_path(model_dir: &Path) -> Result<Self> {
        let tokenizer_config_path = model_dir.join("tokenizer_config.json");
        let tokenizer_config_text = std::fs::read_to_string(&tokenizer_config_path)
            .with_context(|| format!("failed to read {}", tokenizer_config_path.display()))?;
        let tokenizer_config: JsonValue = serde_json::from_str(&tokenizer_config_text)
            .with_context(|| format!("failed to parse {}", tokenizer_config_path.display()))?;

        let standalone_path = model_dir.join("chat_template.jinja");
        let standalone = match std::fs::read_to_string(&standalone_path) {
            Ok(value) if !value.trim().is_empty() => Some(value),
            Ok(_) => None,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to read {}", standalone_path.display()));
            }
        };
        let configured = tokenizer_config
            .get("chat_template")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned);
        let template = standalone.or(configured).with_context(|| {
            format!(
                "missing chat template: expected non-empty {} or chat_template in {}",
                standalone_path.display(),
                tokenizer_config_path.display()
            )
        })?;

        let mut environment = Environment::new();
        configure_environment(&mut environment);
        environment
            .add_template("chat", &template)
            .with_context(|| format!("failed to parse {}", standalone_path.display()))?;

        Ok(Self {
            template,
            bos_token: extract_token(&tokenizer_config, "bos_token"),
            eos_token: extract_token(&tokenizer_config, "eos_token"),
        })
    }

    pub(crate) fn render_user(&self, prompt: &str) -> Result<String> {
        self.render_messages(
            &[ChatMessage {
                role: "user".to_string(),
                content: Some(ChatMessageContent::Text(prompt.to_string())),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            &[],
        )
    }

    pub(crate) fn render_messages(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
    ) -> Result<String> {
        anyhow::ensure!(!messages.is_empty(), "messages must not be empty");
        for (index, message) in messages.iter().enumerate() {
            match message.role.as_str() {
                "system" => {
                    anyhow::ensure!(
                        matches!(&message.content, Some(ChatMessageContent::Text(text)) if !text.is_empty())
                            && message.tool_calls.is_empty()
                            && message.tool_call_id.is_none(),
                        "message {index} has fields incompatible with role \"system\""
                    );
                }
                "user" => {
                    let valid_content = match message.content.as_ref() {
                        Some(ChatMessageContent::Text(text)) => !text.is_empty(),
                        Some(ChatMessageContent::Parts(parts)) => {
                            !parts.is_empty()
                                && parts.iter().all(|part| match part {
                                    ChatContentPart::Text { text } => !text.is_empty(),
                                    ChatContentPart::ImageUrl { image_url } => {
                                        !image_url.url.is_empty() && image_url.detail == "auto"
                                    }
                                })
                        }
                        None => false,
                    };
                    anyhow::ensure!(
                        valid_content
                            && message.tool_calls.is_empty()
                            && message.tool_call_id.is_none(),
                        "message {index} has fields incompatible with role \"user\""
                    );
                }
                "assistant" => {
                    anyhow::ensure!(
                        (matches!(&message.content, Some(ChatMessageContent::Text(text)) if !text.is_empty())
                            || !message.tool_calls.is_empty())
                            && !matches!(&message.content, Some(ChatMessageContent::Parts(_)))
                            && message.tool_call_id.is_none(),
                        "message {index} has fields incompatible with role \"assistant\""
                    );
                    for (call_index, call) in message.tool_calls.iter().enumerate() {
                        anyhow::ensure!(
                            call.tool_type == "function"
                                && call.function.arguments.is_object(),
                            "message {index} tool call {call_index} is invalid"
                        );
                    }
                }
                "tool" => {
                    anyhow::ensure!(
                        matches!(&message.content, Some(ChatMessageContent::Text(text)) if !text.is_empty())
                            && message.tool_calls.is_empty()
                            && message.tool_call_id.is_some(),
                        "message {index} has fields incompatible with role \"tool\""
                    );
                }
                _ => {
                    anyhow::bail!(
                        "message {index} has unsupported role {:?}",
                        message.role
                    );
                }
            }
        }
        let mut environment = Environment::new();
        configure_environment(&mut environment);
        environment
            .add_template("chat", &self.template)
            .context("failed to parse chat template")?;
        let template = environment.get_template("chat")?;
        template
            .render(context! {
                messages => Value::from_serialize(messages),
                tools => Value::from_serialize(tools),
                bos_token => self.bos_token.as_str(),
                eos_token => self.eos_token.as_str(),
                add_generation_prompt => true,
                enable_thinking => true,
            })
            .context("failed to render chat messages")
    }

    pub(crate) fn supports_qwen35_tool_calls(&self) -> bool {
        self.template.contains("<tool_call>")
            && self.template.contains("<function=")
            && self.template.contains("<parameter=")
    }

    pub(crate) fn supports_image_content(&self) -> bool {
        self.template.contains("image_url")
            || (self.template.contains("content")
                && (self.template.contains("\"image\"")
                    || self.template.contains("'image'"))
                && (self.template.contains("vision_start")
                    || self.template.contains("image_pad")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TOOL_TEMPLATE: &str = r#"
{%- if tools %}<tools>{{ tools|tojson }}</tools>{% endif -%}
{%- for message in messages -%}
{%- if message.role == "assistant" and message.tool_calls -%}
{{- message.content or "" -}}
{%- for call in message.tool_calls -%}
<tool_call><function={{ call.function.name }}>
{%- for key, value in call.function.arguments|items -%}
<parameter={{ key }}>{{ value|tojson }}</parameter>
{%- endfor -%}
</function></tool_call>
{%- endfor -%}
{%- elif message.role == "tool" -%}
<tool_response>{{ message.content }}</tool_response>
{%- else -%}
{{ message.role }}:{{ message.content }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}assistant:{% endif -%}
"#;

    fn processor() -> ChatTemplateProcessor {
        ChatTemplateProcessor {
            template: TOOL_TEMPLATE.to_string(),
            bos_token: String::new(),
            eos_token: String::new(),
        }
    }

    fn user(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Some(ChatMessageContent::Text(content.to_string())),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    fn tool() -> ChatTool {
        ChatTool {
            tool_type: "function".to_string(),
            function: ChatToolFunction {
                name: "weather".to_string(),
                description: Some("Look up weather".to_string()),
                parameters: json!({"type":"object"}),
                strict: Some(true),
            },
        }
    }

    #[test]
    fn renders_tool_declarations_and_assistant_calls() {
        let messages = [
            user("weather?"),
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: vec![ChatToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: ChatToolCallFunction {
                        name: "weather".to_string(),
                        arguments: json!({"city":"Paris","days":2}),
                    },
                }],
                tool_call_id: None,
            },
        ];
        let rendered = processor()
            .render_messages(&messages, &[tool()])
            .expect("render tools");
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains(r#""name":"weather""#));
        assert!(rendered.contains("<tool_call><function=weather>"));
        assert!(rendered.contains("<parameter=city>\"Paris\"</parameter>"));
        assert!(rendered.contains("<parameter=days>2</parameter>"));
    }

    #[test]
    fn renders_consecutive_tool_results() {
        let messages = [
            user("weather?"),
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: vec![ChatToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: ChatToolCallFunction {
                        name: "weather".to_string(),
                        arguments: json!({}),
                    },
                }],
                tool_call_id: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: Some(ChatMessageContent::Text("sunny".to_string())),
                tool_calls: Vec::new(),
                tool_call_id: Some("call_1".to_string()),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: Some(ChatMessageContent::Text("warm".to_string())),
                tool_calls: Vec::new(),
                tool_call_id: Some("call_2".to_string()),
            },
        ];
        let rendered = processor()
            .render_messages(&messages, &[tool()])
            .expect("render tool results");
        assert!(rendered.contains(
            "<tool_response>sunny</tool_response><tool_response>warm</tool_response>"
        ));
    }

    #[test]
    fn empty_tools_and_render_user_keep_ordinary_chat() {
        let processor = processor();
        let rendered = processor
            .render_messages(&[user("hello")], &[])
            .expect("render without tools");
        assert_eq!(rendered, "user:helloassistant:");
        assert!(!rendered.contains("<tools>"));
        assert_eq!(
            processor.render_user("hello").expect("render user"),
            rendered
        );
    }
    #[test]
    fn ordered_openai_image_parts_are_serialized_losslessly() {
        let processor = ChatTemplateProcessor {
            template: "{{ messages|tojson }}".to_string(),
            bos_token: String::new(),
            eos_token: String::new(),
        };
        let message = ChatMessage {
            role: "user".to_string(),
            content: Some(ChatMessageContent::Parts(vec![
                ChatContentPart::Text {
                    text: "before".to_string(),
                },
                ChatContentPart::ImageUrl {
                    image_url: ChatImageUrl {
                        url: "data:image/png;base64,AA==".to_string(),
                        detail: "auto".to_string(),
                    },
                },
                ChatContentPart::Text {
                    text: "after".to_string(),
                },
            ])),
            tool_calls: Vec::new(),
            tool_call_id: None,
        };
        let rendered = processor
            .render_messages(&[message], &[])
            .expect("render image parts");
        let before = rendered.find("\"before\"").expect("leading text");
        let image = rendered.find("\"image_url\"").expect("image part");
        let after = rendered.find("\"after\"").expect("trailing text");
        assert!(before < image && image < after);
        assert!(rendered.contains("\"detail\":\"auto\""));
    }

}

fn extract_token(config: &JsonValue, name: &str) -> String {
    let Some(value) = config.get(name) else {
        return String::new();
    };
    if let Some(value) = value.as_str() {
        return value.to_owned();
    }
    value
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn configure_environment(environment: &mut Environment<'_>) {
    environment.set_keep_trailing_newline(true);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_fuel(Some(50_000_000));
    environment.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    environment.set_unknown_method_callback(|_state, value, method, args| {
        if value.kind() != ValueKind::String {
            return Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("unknown method {method}"),
            ));
        }
        let string = value.as_str().unwrap_or_default();
        let argument = || args.first().and_then(Value::as_str).unwrap_or_default();
        match method {
            "startswith" => Ok(Value::from(string.starts_with(argument()))),
            "endswith" => Ok(Value::from(string.ends_with(argument()))),
            "strip" => Ok(Value::from(string.trim().to_owned())),
            "lstrip" => Ok(Value::from(string.trim_start().to_owned())),
            "rstrip" => Ok(Value::from(string.trim_end().to_owned())),
            "split" => {
                let separator = argument();
                if separator.is_empty() {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "chat template split requires a separator",
                    ));
                }
                Ok(Value::from(
                    string
                        .split(separator)
                        .map(str::to_owned)
                        .map(Value::from)
                        .collect::<Vec<_>>(),
                ))
            }
            _ => Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("unknown string method {method}"),
            )),
        }
    });
}
