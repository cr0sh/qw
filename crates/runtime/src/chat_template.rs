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

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
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
        let mut environment = Environment::new();
        configure_environment(&mut environment);
        environment
            .add_template("chat", &self.template)
            .context("failed to parse chat template")?;
        let template = environment.get_template("chat")?;
        let messages = [ChatMessage {
            role: "user",
            content: prompt,
        }];
        template
            .render(context! {
                messages => Value::from_serialize(&messages),
                tools => Value::from_serialize(Vec::<JsonValue>::new()),
                bos_token => self.bos_token.as_str(),
                eos_token => self.eos_token.as_str(),
                add_generation_prompt => true,
                enable_thinking => true,
            })
            .context("failed to render one-user chat template")
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
