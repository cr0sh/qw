use anyhow::{Result, ensure};
use qw_runtime::ChatTool;
use serde_json::{Map, Value};

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
    declared_tools: &[ChatTool],
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
        let tool = declared_tools
            .iter()
            .find(|tool| tool.function.name == name)
            .ok_or_else(|| anyhow::anyhow!("undeclared function {name:?}"))?;
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
            let schema = tool
                .function
                .parameters
                .get("properties")
                .and_then(|properties| properties.get(&parameter));
            let value = decode_parameter(value, schema);
            arguments.insert(parameter, value);
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
    value
}

fn decode_parameter(value: &str, schema: Option<&Value>) -> Value {
    // Qwen emits strings verbatim and JSON-encodes nonstrings. Without an
    // unambiguous type excluding strings, interpreting the text loses information.
    // This is a wire decoder, not a JSON Schema validator or reference resolver.
    let types = schema.and_then(|schema| schema.get("type"));
    let allows = |kind: &str| match types {
        Some(Value::String(declared)) => declared == kind,
        Some(Value::Array(declared)) => declared.iter().any(|item| item.as_str() == Some(kind)),
        _ => false,
    };
    let known_nonstring = match types {
        Some(Value::String(kind)) => {
            matches!(
                kind.as_str(),
                "object" | "array" | "integer" | "number" | "boolean" | "null"
            )
        }
        Some(Value::Array(kinds)) => {
            !kinds.is_empty()
                && kinds.iter().all(|kind| {
                    matches!(
                        kind.as_str(),
                        Some("object" | "array" | "integer" | "number" | "boolean" | "null")
                    )
                })
        }
        _ => false,
    };
    if known_nonstring {
        let trimmed = value.trim();
        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
            let matches = match &parsed {
                Value::Null => allows("null"),
                Value::Bool(_) => allows("boolean"),
                Value::Number(number) => {
                    allows("number") || (allows("integer") && (number.is_i64() || number.is_u64()))
                }
                Value::Array(_) => allows("array"),
                Value::Object(_) => allows("object"),
                Value::String(_) => false,
            };
            if matches {
                return parsed;
            }
        }
        if allows("null")
            && ["null", "none", "nil"]
                .iter()
                .any(|word| trimmed.eq_ignore_ascii_case(word))
        {
            return Value::Null;
        }
        if allows("boolean") {
            if ["true", "1", "yes", "on"]
                .iter()
                .any(|word| trimmed.eq_ignore_ascii_case(word))
            {
                return Value::Bool(true);
            }
            if ["false", "0", "no", "off"]
                .iter()
                .any(|word| trimmed.eq_ignore_ascii_case(word))
            {
                return Value::Bool(false);
            }
        }
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

    fn tools() -> Vec<ChatTool> {
        ["weather", "search", "empty"]
            .into_iter()
            .map(|name| ChatTool {
                tool_type: "function".to_string(),
                function: qw_runtime::ChatToolFunction {
                    name: name.to_string(),
                    description: None,
                    parameters: json!({"type":"object", "properties": {
                        "city":{"type":"string"}, "nil":{"type":"null"},
                        "int":{"type":"integer"}, "float":{"type":"number"},
                        "bool":{"type":"boolean"}, "obj":{"type":"object"},
                        "array":{"type":"array"}, "text":{"type":"string"}
                    }}),
                    strict: None,
                },
            })
            .collect()
    }

    fn parse(text: &str) -> Result<ParsedAssistantOutput> {
        parse_assistant_output(text, &tools(), 8, 8)
    }

    fn parse_value(text: &str, schema: Value) -> Value {
        let mut tools = tools();
        tools[0].function.parameters = json!({
            "type": "object", "properties": {"value": schema}
        });
        let output = parse_assistant_output(
            &format!("<tool_call><function=weather><parameter=value>{text}</parameter></function></tool_call>"),
            &tools,
            8,
            8,
        ).expect("parse parameter");
        let arguments: Value = serde_json::from_str(&output.tool_calls[0].arguments).unwrap();
        arguments["value"].clone()
    }

    #[test]
    fn declared_strings_preserve_json_scalars_quotes_and_empty_text() {
        for text in [
            r#"{"account_id":"abc","amount":42}"#,
            r#"[1,"two"]"#,
            "true",
            "42",
            "2.5",
            "null",
            "NONE",
            r#""quoted\ntext""#,
            "",
            "  significant spaces  ",
        ] {
            assert_eq!(parse_value(text, json!({"type":"string"})), json!(text));
        }
        assert_eq!(
            parse_value("\n  first\nsecond  \n", json!({"type":"string"})),
            json!("  first\nsecond  ")
        );
    }

    #[test]
    fn ambiguous_or_missing_types_preserve_text_instead_of_guessing() {
        for schema in [
            Value::Null,
            json!({}),
            json!({"type":["string","object"]}),
            json!({"anyOf":[{"type":"string"},{"type":"object"}]}),
            json!({"oneOf":[{"type":"integer"},{"type":"boolean"}]}),
            json!({"$ref":"#/$defs/payload"}),
        ] {
            assert_eq!(parse_value("42", schema.clone()), json!("42"));
            assert_eq!(parse_value(r#"{"x":1}"#, schema), json!(r#"{"x":1}"#));
        }
    }

    #[test]
    fn nonstring_types_decode_only_matching_values() {
        assert_eq!(parse_value("1", json!({"type":"boolean"})), json!(true));
        assert_eq!(parse_value("OFF", json!({"type":"boolean"})), json!(false));
        assert_eq!(parse_value("nil", json!({"type":"null"})), Value::Null);
        assert_eq!(
            parse_value("18446744073709551615", json!({"type":"integer"})),
            json!(u64::MAX)
        );
        assert_eq!(
            parse_value(" \n[1,true]\n ", json!({"type":["array","null"]})),
            json!([1, true])
        );
        assert_eq!(
            parse_value("null", json!({"type":["array","null"]})),
            Value::Null
        );
        assert_eq!(parse_value("2.5", json!({"type":"integer"})), json!("2.5"));
        assert_eq!(
            parse_value(r#"{"x":1}"#, json!({"type":"array"})),
            json!(r#"{"x":1}"#)
        );
        assert_eq!(
            parse_value("{invalid}", json!({"type":"object"})),
            json!("{invalid}")
        );
    }

    #[test]
    fn parameter_schema_is_selected_by_function_name() {
        let mut tools = tools();
        tools[0].function.parameters = json!({"properties":{"value":{"type":"string"}}});
        tools[1].function.parameters = json!({"properties":{"value":{"type":"object"}}});
        let output = parse_assistant_output(
            "<tool_call><function=weather><parameter=value>{\"x\":1}</parameter></function></tool_call>\
             <tool_call><function=search><parameter=value>{\"x\":1}</parameter></function></tool_call>",
            &tools, 8, 8,
        ).expect("parse differently typed calls");
        let first: Value = serde_json::from_str(&output.tool_calls[0].arguments).unwrap();
        let second: Value = serde_json::from_str(&output.tool_calls[1].arguments).unwrap();
        assert_eq!(first["value"], json!(r#"{"x":1}"#));
        assert_eq!(second["value"], json!({"x":1}));
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
        assert!(parse_assistant_output("text", &tools(), 9, 8).is_err());
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
