use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use serde_json::{Number, Value};

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

        let mut parameter_names = HashSet::new();
        let mut arguments = String::from("{");
        let mut first_parameter = true;
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
                parameter_names.len() < MAX_PARAMETERS,
                "too many function parameters"
            );
            let (parameter, after_parameter_name) =
                parse_name(&remaining[PARAMETER_OPEN.len()..], "parameter")?;
            ensure!(
                parameter_names.insert(parameter.clone()),
                "duplicate parameter name {parameter:?}"
            );
            let (end, value) = parse_parameter_value(after_parameter_name)
                .ok_or_else(|| anyhow::anyhow!("unclosed parameter {parameter:?}"))?;
            if !first_parameter {
                arguments.push(',');
            }
            first_parameter = false;
            arguments.push_str(
                &serde_json::to_string(&parameter)
                    .context("failed to serialize generated parameter name")?,
            );
            arguments.push(':');
            arguments.push_str(&value);
            remaining = &after_parameter_name[end + PARAMETER_CLOSE.len()..];
        }
        arguments.push('}');
        remaining = trim_leading_whitespace(remaining);
        ensure!(
            remaining.starts_with(TOOL_CALL_CLOSE),
            "tool call is missing its closing marker"
        );
        remaining = &remaining[TOOL_CALL_CLOSE.len()..];
        ensure!(
            serde_json::from_str::<Value>(&arguments)?.is_object(),
            "tool arguments must be an object"
        );
        calls.push(ParsedToolCall { name, arguments });
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
    Ok((normalize_name(&input[..end], kind)?, &input[end + 1..]))
}

fn normalize_name(raw: &str, kind: &str) -> Result<String> {
    let raw = raw.trim();
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
    Ok(name.to_string())
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

fn parse_parameter_value(input: &str) -> Option<(usize, String)> {
    let mut first_close = None;
    for (end, _) in input.match_indices(PARAMETER_CLOSE) {
        first_close.get_or_insert(end);
        let value = trim_parameter_value(&input[..end]);
        if serde_json::from_str::<Value>(value).is_ok() {
            return Some((end, value.to_string()));
        }
    }
    let end = first_close?;
    let value = coerce_parameter(trim_parameter_value(&input[..end]));
    Some((end, serde_json::to_string(&value).ok()?))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallStreamDelta {
    Content(String),
    Start { index: usize, name: String },
    Arguments { index: usize, fragment: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamPhase {
    Content,
    FunctionOpen,
    FunctionBody,
    ParameterOpen,
    ParameterValue,
    ParameterClose,
    ToolCallClose,
    BetweenCalls,
}

#[derive(Debug)]
pub struct ToolCallStreamParser {
    phase: StreamPhase,
    pending: String,
    call_index: usize,
    parameter_names: HashSet<String>,
    first_parameter: bool,
    value: JsonValueScanner,
}

impl Default for ToolCallStreamParser {
    fn default() -> Self {
        Self {
            phase: StreamPhase::Content,
            pending: String::new(),
            call_index: 0,
            parameter_names: HashSet::new(),
            first_parameter: true,
            value: JsonValueScanner::default(),
        }
    }
}

impl ToolCallStreamParser {
    pub fn feed(&mut self, fragment: &str) -> Result<Vec<ToolCallStreamDelta>> {
        self.pending.push_str(fragment);
        let mut deltas = Vec::new();
        loop {
            match self.phase {
                StreamPhase::Content => {
                    if let Some(marker) = self.pending.find(TOOL_CALL_OPEN) {
                        let released = self.pending[..marker].to_string();
                        self.pending.drain(..marker + TOOL_CALL_OPEN.len());
                        self.phase = StreamPhase::FunctionOpen;
                        if !released.is_empty() {
                            deltas.push(ToolCallStreamDelta::Content(released));
                        }
                        continue;
                    }
                    let suffix = longest_marker_prefix_suffix(&self.pending, TOOL_CALL_OPEN);
                    let released_end = self.pending.len() - suffix;
                    if released_end == 0 {
                        break;
                    }
                    let released = self.pending.drain(..released_end).collect();
                    deltas.push(ToolCallStreamDelta::Content(released));
                    break;
                }
                StreamPhase::FunctionOpen => {
                    self.discard_leading_whitespace();
                    if self.pending.is_empty() || FUNCTION_OPEN.starts_with(&self.pending) {
                        break;
                    }
                    ensure!(
                        self.pending.starts_with(FUNCTION_OPEN),
                        "tool call is missing a function opener"
                    );
                    let Some(end) = self.pending[FUNCTION_OPEN.len()..].find('>') else {
                        ensure!(
                            self.pending.len() <= FUNCTION_OPEN.len() + MAX_NAME_BYTES,
                            "function name is too long"
                        );
                        break;
                    };
                    let name_end = FUNCTION_OPEN.len() + end;
                    let name = normalize_name(&self.pending[FUNCTION_OPEN.len()..name_end], "function")?;
                    ensure!(self.call_index < MAX_CALLS, "too many tool calls");
                    self.pending.drain(..=name_end);
                    self.parameter_names.clear();
                    self.first_parameter = true;
                    self.phase = StreamPhase::FunctionBody;
                    deltas.push(ToolCallStreamDelta::Start {
                        index: self.call_index,
                        name,
                    });
                    deltas.push(ToolCallStreamDelta::Arguments {
                        index: self.call_index,
                        fragment: "{".to_string(),
                    });
                }
                StreamPhase::FunctionBody => {
                    self.discard_leading_whitespace();
                    if self.pending.starts_with(FUNCTION_CLOSE) {
                        self.pending.drain(..FUNCTION_CLOSE.len());
                        self.phase = StreamPhase::ToolCallClose;
                        deltas.push(ToolCallStreamDelta::Arguments {
                            index: self.call_index,
                            fragment: "}".to_string(),
                        });
                        continue;
                    }
                    if self.pending.starts_with(PARAMETER_OPEN) {
                        self.phase = StreamPhase::ParameterOpen;
                        continue;
                    }
                    if self.pending.is_empty()
                        || FUNCTION_CLOSE.starts_with(&self.pending)
                        || PARAMETER_OPEN.starts_with(&self.pending)
                    {
                        break;
                    }
                    anyhow::bail!("malformed or unclosed function body");
                }
                StreamPhase::ParameterOpen => {
                    let Some(end) = self.pending[PARAMETER_OPEN.len()..].find('>') else {
                        ensure!(
                            self.pending.len() <= PARAMETER_OPEN.len() + MAX_NAME_BYTES,
                            "parameter name is too long"
                        );
                        break;
                    };
                    let name_end = PARAMETER_OPEN.len() + end;
                    let name =
                        normalize_name(&self.pending[PARAMETER_OPEN.len()..name_end], "parameter")?;
                    ensure!(
                        self.parameter_names.len() < MAX_PARAMETERS,
                        "too many function parameters"
                    );
                    ensure!(
                        self.parameter_names.insert(name.clone()),
                        "duplicate parameter name {name:?}"
                    );
                    self.pending.drain(..=name_end);
                    let mut property = String::new();
                    if !self.first_parameter {
                        property.push(',');
                    }
                    self.first_parameter = false;
                    property.push_str(
                        &serde_json::to_string(&name)
                            .context("failed to serialize generated parameter name")?,
                    );
                    property.push(':');
                    self.value = JsonValueScanner::default();
                    self.phase = StreamPhase::ParameterValue;
                    deltas.push(ToolCallStreamDelta::Arguments {
                        index: self.call_index,
                        fragment: property,
                    });
                }
                StreamPhase::ParameterValue => {
                    let scanned = self.value.consume(&self.pending)?;
                    if scanned.consumed > 0 {
                        self.pending.drain(..scanned.consumed);
                    }
                    if !scanned.fragment.is_empty() {
                        deltas.push(ToolCallStreamDelta::Arguments {
                            index: self.call_index,
                            fragment: scanned.fragment,
                        });
                    }
                    if scanned.complete {
                        self.phase = StreamPhase::ParameterClose;
                        continue;
                    }
                    break;
                }
                StreamPhase::ParameterClose => {
                    if self.pending.starts_with(PARAMETER_CLOSE) {
                        self.value.validate()?;
                        self.pending.drain(..PARAMETER_CLOSE.len());
                        self.phase = StreamPhase::FunctionBody;
                        continue;
                    }
                    if self.pending.is_empty() || PARAMETER_CLOSE.starts_with(&self.pending) {
                        break;
                    }
                    anyhow::bail!("parameter JSON value is not followed by its closing marker");
                }
                StreamPhase::ToolCallClose => {
                    self.discard_leading_whitespace();
                    if self.pending.starts_with(TOOL_CALL_CLOSE) {
                        self.pending.drain(..TOOL_CALL_CLOSE.len());
                        self.call_index += 1;
                        self.phase = StreamPhase::BetweenCalls;
                        continue;
                    }
                    if self.pending.is_empty() || TOOL_CALL_CLOSE.starts_with(&self.pending) {
                        break;
                    }
                    anyhow::bail!("tool call is missing its closing marker");
                }
                StreamPhase::BetweenCalls => {
                    self.discard_leading_whitespace();
                    if self.pending.starts_with(TOOL_CALL_OPEN) {
                        self.pending.drain(..TOOL_CALL_OPEN.len());
                        self.phase = StreamPhase::FunctionOpen;
                        continue;
                    }
                    if self.pending.is_empty() || TOOL_CALL_OPEN.starts_with(&self.pending) {
                        break;
                    }
                    anyhow::bail!("unexpected text between tool calls");
                }
            }
        }
        Ok(deltas)
    }

    pub fn finish(&mut self) -> Result<Vec<ToolCallStreamDelta>> {
        let mut deltas = self.feed("")?;
        match self.phase {
            StreamPhase::Content => {
                if !self.pending.is_empty() {
                    deltas.push(ToolCallStreamDelta::Content(std::mem::take(
                        &mut self.pending,
                    )));
                }
            }
            StreamPhase::BetweenCalls if self.pending.trim().is_empty() => {
                self.pending.clear();
            }
            _ => anyhow::bail!("generated tool-call output ended before the current XML token"),
        }
        Ok(deltas)
    }

    fn discard_leading_whitespace(&mut self) {
        let retained = self.pending.trim_start_matches(char::is_whitespace).len();
        let discard = self.pending.len() - retained;
        if discard != 0 {
            self.pending.drain(..discard);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonRootKind {
    Container,
    String,
    Primitive,
}

#[derive(Debug, Default)]
struct JsonValueScanner {
    raw: String,
    root: Option<JsonRootKind>,
    stack: Vec<char>,
    in_string: bool,
    escaped: bool,
    complete: bool,
}

struct JsonScan {
    consumed: usize,
    fragment: String,
    complete: bool,
}

impl JsonValueScanner {
    fn consume(&mut self, input: &str) -> Result<JsonScan> {
        let mut consumed = 0;
        let mut fragment = String::new();
        for (offset, character) in input.char_indices() {
            let end = offset + character.len_utf8();
            if self.root.is_none() {
                if character.is_whitespace() {
                    consumed = end;
                    continue;
                }
                match character {
                    '{' => {
                        self.root = Some(JsonRootKind::Container);
                        self.stack.push('}');
                    }
                    '[' => {
                        self.root = Some(JsonRootKind::Container);
                        self.stack.push(']');
                    }
                    '"' => {
                        self.root = Some(JsonRootKind::String);
                        self.in_string = true;
                    }
                    '<' => anyhow::bail!("parameter is missing a JSON value"),
                    _ => self.root = Some(JsonRootKind::Primitive),
                }
                self.push(character, &mut fragment);
                consumed = end;
                continue;
            }
            if self.complete {
                if character.is_whitespace() {
                    consumed = end;
                    continue;
                }
                break;
            }

            match self.root.expect("JSON root was initialized") {
                JsonRootKind::Primitive => {
                    if character == '<' {
                        self.complete = true;
                        break;
                    }
                    if character.is_whitespace() {
                        self.complete = true;
                        consumed = end;
                        continue;
                    }
                    self.push(character, &mut fragment);
                    consumed = end;
                }
                JsonRootKind::String => {
                    self.push(character, &mut fragment);
                    consumed = end;
                    if self.escaped {
                        self.escaped = false;
                    } else if character == '\\' {
                        self.escaped = true;
                    } else if character == '"' {
                        self.in_string = false;
                        self.complete = true;
                    }
                }
                JsonRootKind::Container => {
                    self.push(character, &mut fragment);
                    consumed = end;
                    if self.in_string {
                        if self.escaped {
                            self.escaped = false;
                        } else if character == '\\' {
                            self.escaped = true;
                        } else if character == '"' {
                            self.in_string = false;
                        }
                        continue;
                    }
                    match character {
                        '"' => self.in_string = true,
                        '{' => self.stack.push('}'),
                        '[' => self.stack.push(']'),
                        '}' | ']' => {
                            ensure!(
                                self.stack.pop() == Some(character),
                                "mismatched JSON container delimiter"
                            );
                            if self.stack.is_empty() {
                                self.complete = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(JsonScan {
            consumed,
            fragment,
            complete: self.complete,
        })
    }

    fn push(&mut self, character: char, fragment: &mut String) {
        self.raw.push(character);
        fragment.push(character);
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.complete, "parameter JSON value is incomplete");
        serde_json::from_str::<Value>(&self.raw)
            .context("generated parameter was not one complete JSON value")?;
        Ok(())
    }
}

fn longest_marker_prefix_suffix(value: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&length| value.as_bytes().ends_with(&marker.as_bytes()[..length]))
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

    fn collect_stream(
        fragments: &[&str],
    ) -> Result<(String, Vec<(usize, String, String)>)> {
        let mut parser = ToolCallStreamParser::default();
        let mut content = String::new();
        let mut calls = Vec::<(usize, String, String)>::new();
        let mut deltas = Vec::new();
        for fragment in fragments {
            deltas.extend(parser.feed(fragment)?);
        }
        deltas.extend(parser.finish()?);
        for delta in deltas {
            match delta {
                ToolCallStreamDelta::Content(fragment) => content.push_str(&fragment),
                ToolCallStreamDelta::Start { index, name } => {
                    assert_eq!(index, calls.len());
                    calls.push((index, name, String::new()));
                }
                ToolCallStreamDelta::Arguments { index, fragment } => {
                    assert_eq!(calls[index].0, index);
                    calls[index].2.push_str(&fragment);
                }
            }
        }
        Ok((content, calls))
    }

    #[test]
    fn every_xml_byte_boundary_streams_typed_calls_without_leaking_markup() {
        let generated = concat!(
            "Before<tool_call><function=weather>",
            "<parameter=city>\"Paris\"</parameter>",
            "<parameter=meta>{\"literal\":\"</parameter>\",\"days\":2}</parameter>",
            "</function></tool_call>",
            "<tool_call><function=empty></function></tool_call>"
        );
        for boundary in 0..=generated.len() {
            let (content, calls) =
                collect_stream(&[&generated[..boundary], &generated[boundary..]])
                    .unwrap_or_else(|error| panic!("split {boundary}: {error}"));
            assert_eq!(content, "Before", "split {boundary}");
            assert_eq!(
                calls,
                vec![
                    (
                        0,
                        "weather".to_string(),
                        "{\"city\":\"Paris\",\"meta\":{\"literal\":\"</parameter>\",\"days\":2}}"
                            .to_string()
                    ),
                    (1, "empty".to_string(), "{}".to_string())
                ],
                "split {boundary}"
            );
        }
    }

    #[test]
    fn function_opener_immediately_exposes_typed_start_and_json_object() {
        let mut parser = ToolCallStreamParser::default();
        assert_eq!(
            parser.feed("<tool_call><function=weather>").unwrap(),
            vec![
                ToolCallStreamDelta::Start {
                    index: 0,
                    name: "weather".to_string()
                },
                ToolCallStreamDelta::Arguments {
                    index: 0,
                    fragment: "{".to_string()
                }
            ]
        );
    }

    #[test]
    fn plain_content_and_ambiguous_marker_prefixes_are_preserved() {
        let (content, calls) =
            collect_stream(&["one <", " two", " three<tool_"]).expect("plain stream");
        assert_eq!(content, "one < two three<tool_");
        assert!(calls.is_empty());
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
    fn parses_a_json_string_containing_the_parameter_close_marker() {
        let output = parse(
            "<tool_call><function=weather>\
             <parameter=city>\"/>.</parameter>=\"</parameter>\
             </function></tool_call>",
        )
        .expect("parse embedded marker");
        let arguments: Value = serde_json::from_str(&output.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["city"], "/>.</parameter>=");
    }

    #[test]
    fn ordinary_text_is_unchanged() {
        let text = "ordinary < text and ambiguous <tool_call plus <tool_";
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
