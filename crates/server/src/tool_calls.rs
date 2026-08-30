use std::collections::HashSet;

use anyhow::{Result, ensure};
use serde_json::{Map, Number, Value};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";
const MAX_CALLS: usize = 128;
const MAX_PARAMETERS: usize = 256;
const MAX_NAME_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedAssistantOutput {
    pub content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

pub fn parse_assistant_output(
    text: &str,
    declared_names: &[&str],
    generated_tokens: usize,
    generated_token_limit: usize,
) -> Result<ParsedAssistantOutput> {
    ensure!(
        generated_tokens <= generated_token_limit,
        "generated output exceeded the request token limit"
    );
    let Some(first_call) = text.find(TOOL_CALL_OPEN) else {
        ensure!(
            !text.contains("<tool_call") && !text.contains("<tool_"),
            "malformed tool-call marker"
        );
        return Ok(ParsedAssistantOutput {
            content: text.to_string(),
            tool_calls: Vec::new(),
        });
    };
    let declared: HashSet<&str> = declared_names.iter().copied().collect();
    let content = text[..first_call].to_string();
    let mut calls = Vec::new();
    let mut remaining = &text[first_call..];
    loop {
        remaining = trim_leading_whitespace(remaining);
        if remaining.is_empty() {
            break;
        }
        ensure!(
            remaining.starts_with(TOOL_CALL_OPEN),
            "unexpected text between tool calls"
        );
        ensure!(calls.len() < MAX_CALLS, "too many tool calls");
        remaining = &remaining[TOOL_CALL_OPEN.len()..];
        remaining = trim_leading_whitespace(remaining);
        ensure!(
            remaining.starts_with(FUNCTION_OPEN),
            "tool call is missing a function opener"
        );
        let (name, after_name) = parse_name(&remaining[FUNCTION_OPEN.len()..], "function")?;
        ensure!(
            declared.contains(name.as_str()),
            "undeclared function {name:?}"
        );
        remaining = after_name;

        let mut arguments = Map::new();
        loop {
            remaining = trim_leading_whitespace(remaining);
            if remaining.starts_with(FUNCTION_CLOSE) {
                remaining = &remaining[FUNCTION_CLOSE.len()..];
                break;
            }
            ensure!(
                remaining.starts_with(PARAMETER_OPEN),
                "malformed or unclosed function body"
            );
            ensure!(
                arguments.len() < MAX_PARAMETERS,
                "too many function parameters"
            );
            let (parameter, after_parameter_name) =
                parse_name(&remaining[PARAMETER_OPEN.len()..], "parameter")?;
            ensure!(
                !arguments.contains_key(&parameter),
                "duplicate parameter name {parameter:?}"
            );
            let end = after_parameter_name
                .find(PARAMETER_CLOSE)
                .ok_or_else(|| anyhow::anyhow!("unclosed parameter {parameter:?}"))?;
            let value = trim_parameter_value(&after_parameter_name[..end]);
            arguments.insert(parameter, coerce_parameter(value));
            remaining = &after_parameter_name[end + PARAMETER_CLOSE.len()..];
        }
        remaining = trim_leading_whitespace(remaining);
        ensure!(
            remaining.starts_with(TOOL_CALL_CLOSE),
            "tool call is missing its closing marker"
        );
        remaining = &remaining[TOOL_CALL_CLOSE.len()..];
        let arguments = Value::Object(arguments);
        ensure!(arguments.is_object(), "tool arguments must be an object");
        calls.push(ParsedToolCall {
            name,
            arguments: serde_json::to_string(&arguments)?,
        });
    }
    ensure!(!calls.is_empty(), "tool marker contained no calls");
    Ok(ParsedAssistantOutput {
        content,
        tool_calls: calls,
    })
}

fn parse_name<'a>(input: &'a str, kind: &str) -> Result<(String, &'a str)> {
    let end = input
        .find('>')
        .ok_or_else(|| anyhow::anyhow!("unclosed {kind} name"))?;
    let raw = input[..end].trim();
    let name = if raw.len() >= 2
        && ((raw.starts_with('"') && raw.ends_with('"'))
            || (raw.starts_with('\'') && raw.ends_with('\'')))
    {
        &raw[1..raw.len() - 1]
    } else {
        raw
    };
    ensure!(!name.is_empty(), "empty {kind} name");
    ensure!(name.len() <= MAX_NAME_BYTES, "{kind} name is too long");
    ensure!(
        !name.bytes().any(|byte| byte == b'<' || byte == b'>'),
        "invalid {kind} name"
    );
    Ok((name.to_string(), &input[end + 1..]))
}

fn trim_parameter_value(mut value: &str) -> &str {
    if let Some(stripped) = value.strip_prefix('\n') {
        value = stripped;
    }
    if let Some(stripped) = value.strip_suffix('\n') {
        value = stripped;
    }
    value.trim()
}

fn coerce_parameter(value: &str) -> Value {
    if let Ok(parsed) = serde_json::from_str(value) {
        return parsed;
    }
    let lower = value.to_lowercase();
    if matches!(lower.as_str(), "null" | "none" | "nil") {
        return Value::Null;
    }
    if let Ok(integer) = value.parse::<i64>() {
        return Value::Number(Number::from(integer));
    }
    if let Ok(float) = value.parse::<f64>()
        && let Some(number) = Number::from_f64(float)
    {
        return Value::Number(number);
    }
    if matches!(lower.as_str(), "true" | "1" | "yes" | "on") {
        return Value::Bool(true);
    }
    if matches!(lower.as_str(), "false" | "0" | "no" | "off") {
        return Value::Bool(false);
    }
    Value::String(value.to_string())
}

fn trim_leading_whitespace(value: &str) -> &str {
    value.trim_start_matches(char::is_whitespace)
}

#[derive(Debug, Default)]
pub struct ToolCallGate {
    retained: String,
    entered: bool,
}

impl ToolCallGate {
    pub fn feed(&mut self, fragment: &str) -> Option<String> {
        if self.entered || fragment.is_empty() {
            return None;
        }
        self.retained.push_str(fragment);
        if let Some(marker) = self.retained.find(TOOL_CALL_OPEN) {
            let released = self.retained[..marker].to_string();
            self.retained.clear();
            self.entered = true;
            return (!released.is_empty()).then_some(released);
        }
        let suffix = longest_marker_prefix_suffix(&self.retained);
        let released_end = self.retained.len() - suffix;
        if released_end == 0 {
            return None;
        }
        let suffix_value = self.retained[released_end..].to_string();
        let released = self.retained[..released_end].to_string();
        self.retained = suffix_value;
        Some(released)
    }

    pub fn flush(&mut self) -> Option<String> {
        if self.entered || self.retained.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.retained))
        }
    }

    #[cfg(test)]
    pub fn entered(&self) -> bool {
        self.entered
    }

    #[cfg(test)]
    pub fn has_partial_marker(&self) -> bool {
        !self.entered && !self.retained.is_empty()
    }
}

fn longest_marker_prefix_suffix(value: &str) -> usize {
    (1..TOOL_CALL_OPEN.len())
        .rev()
        .find(|&length| {
            value
                .as_bytes()
                .ends_with(&TOOL_CALL_OPEN.as_bytes()[..length])
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const WEATHER: &[&str] = &["weather", "search", "empty"];

    fn parse(text: &str) -> Result<ParsedAssistantOutput> {
        parse_assistant_output(text, WEATHER, 8, 8)
    }

    #[test]
    fn marker_split_at_every_byte_boundary_never_leaks_xml() {
        let xml =
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>";
        for boundary in 0..=TOOL_CALL_OPEN.len() {
            let mut gate = ToolCallGate::default();
            let mut released = String::new();
            let first = format!("Before{}", &TOOL_CALL_OPEN[..boundary]);
            let second = format!(
                "{}{}",
                &TOOL_CALL_OPEN[boundary..],
                &xml[TOOL_CALL_OPEN.len()..]
            );
            for fragment in [&first, &second] {
                if let Some(value) = gate.feed(fragment) {
                    released.push_str(&value);
                }
            }
            if let Some(value) = gate.flush() {
                released.push_str(&value);
            }
            assert_eq!(released, "Before", "split {boundary}");
            assert!(gate.entered());
        }
    }

    #[test]
    fn gate_releases_ordinary_text_and_lone_angle_bracket() {
        let mut gate = ToolCallGate::default();
        assert_eq!(gate.feed("one <"), Some("one ".to_string()));
        assert_eq!(gate.feed(" two"), Some("< two".to_string()));
        assert_eq!(gate.feed(" three"), Some(" three".to_string()));
        assert_eq!(gate.flush(), None);
        assert!(!gate.entered());
    }

    #[test]
    fn gate_flushes_an_uncompleted_marker_prefix() {
        let mut gate = ToolCallGate::default();
        assert_eq!(gate.feed("text<tool_"), Some("text".to_string()));
        assert!(gate.has_partial_marker());
        assert_eq!(gate.flush(), Some("<tool_".to_string()));
    }

    #[test]
    fn parses_prose_multiple_calls_and_zero_parameters() {
        let output = parse(
            "Before<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>\n<tool_call><function=empty></function></tool_call>",
        )
        .expect("parse calls");
        assert_eq!(output.content, "Before");
        assert_eq!(output.tool_calls.len(), 2);
        assert_eq!(output.tool_calls[0].name, "weather");
        assert_eq!(output.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(output.tool_calls[1].arguments, "{}");
    }

    #[test]
    fn coerces_nested_json_arrays_scalars_and_all_caps() {
        let output = parse(
            "<tool_call><function=search>\
             <parameter=nil>NULL</parameter>\
             <parameter=int>42</parameter>\
             <parameter=float>2.5</parameter>\
             <parameter=bool>TRUE</parameter>\
             <parameter=obj>{\"nested\":[1,{\"x\":false}]}</parameter>\
             <parameter=array>[1,\"two\"]</parameter>\
             <parameter=quoted>\"hello\"</parameter>\
             <parameter=text>hello</parameter>\
             </function></tool_call>",
        )
        .expect("parse coercions");
        let arguments: Value = serde_json::from_str(&output.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["nil"], Value::Null);
        assert_eq!(arguments["int"], 42);
        assert_eq!(arguments["float"], 2.5);
        assert_eq!(arguments["bool"], true);
        assert_eq!(arguments["obj"], json!({"nested":[1,{"x":false}]}));
        assert_eq!(arguments["array"], json!([1, "two"]));
        assert_eq!(arguments["text"], "hello");
        assert_eq!(arguments["quoted"], "hello");
    }

    #[test]
    fn ordinary_text_is_unchanged() {
        let text = "ordinary < text";
        assert_eq!(
            parse(text).expect("ordinary text"),
            ParsedAssistantOutput {
                content: text.to_string(),
                tool_calls: Vec::new()
            }
        );
    }

    #[test]
    fn rejects_undeclared_and_all_bounds() {
        assert!(parse("<tool_call><function=other></function></tool_call>").is_err());
        let long_name = "x".repeat(MAX_NAME_BYTES + 1);
        assert!(
            parse(&format!(
                "<tool_call><function={long_name}></function></tool_call>"
            ))
            .is_err()
        );
        let calls = "<tool_call><function=empty></function></tool_call>".repeat(MAX_CALLS + 1);
        assert!(parse(&calls).is_err());
        let params = (0..=MAX_PARAMETERS)
            .map(|index| format!("<parameter=p{index}>x</parameter>"))
            .collect::<String>();
        assert!(
            parse(&format!(
                "<tool_call><function=search>{params}</function></tool_call>"
            ))
            .is_err()
        );
        assert!(parse_assistant_output("text", WEATHER, 9, 8).is_err());
    }

    #[test]
    fn rejects_duplicate_and_empty_parameter_names() {
        assert!(parse("<tool_call><function=search><parameter=x>1</parameter><parameter=x>2</parameter></function></tool_call>").is_err());
        assert!(
            parse("<tool_call><function=search><parameter=>1</parameter></function></tool_call>")
                .is_err()
        );
    }

    #[test]
    fn rejects_every_malformed_or_truncated_delimiter_class() {
        for malformed in [
            "<tool_call",
            "<tool_call>",
            "<tool_call><function=weather",
            "<tool_call><function=>",
            "<tool_call><function=weather>",
            "<tool_call><function=weather><parameter=city",
            "<tool_call><function=weather><parameter=city>",
            "<tool_call><function=weather><parameter=city>x</function></tool_call>",
            "<tool_call><function=weather></function>",
            "<tool_call><function=weather></function></tool_call>trailing",
            "<tool_call><function=weather></function><tool_call>",
            "<tool_call></tool_call>",
        ] {
            assert!(parse(malformed).is_err(), "accepted {malformed:?}");
        }
    }
}
