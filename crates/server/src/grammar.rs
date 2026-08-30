use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use llguidance::{Constraint, ParserFactory, token_bytes_from_tokenizer_json};
use mlxcel_core::generate::{ConstraintCommit, ConstraintMask, TokenConstraint};
use qw_runtime::ChatTool;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;
use toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

use crate::protocol::OutputFormat;

struct LocalTokenizerEnv {
    trie: TokTrie,
    tokenizer: Tokenizer,
}

impl TokenizerEnv for LocalTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        let Ok(text) = std::str::from_utf8(bytes) else {
            return self.trie.greedy_tokenize(bytes);
        };
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_else(|_| self.trie.greedy_tokenize(bytes))
    }

    fn tokenize_is_canonical(&self) -> bool {
        true
    }
}

const TOOL_CALL_OPEN: &[u8] = b"<tool_call>";
const MAX_TOOL_CALL_NAME_BYTES: usize = 256;
const MAX_TOOL_PARAMETERS: usize = 256;

pub struct GrammarFactory {
    parser: ParserFactory,
    token_bytes: Arc<Vec<Vec<u8>>>,
    eos_tokens: Arc<[TokenId]>,
}

impl GrammarFactory {
    pub fn from_tokenizer_json(
        tokenizer_json: &Value,
        tokenizer_vocab_size: usize,
        logits_vocab_size: usize,
        eos_tokens: &[i32],
    ) -> Result<Self> {
        ensure!(
            tokenizer_vocab_size <= logits_vocab_size,
            "tokenizer vocabulary exceeds model logits vocabulary"
        );
        let mut token_bytes = token_bytes_from_tokenizer_json(tokenizer_json)
            .context("failed to derive token bytes from tokenizer.json")?;
        ensure!(
            token_bytes.len() == tokenizer_vocab_size,
            "tokenizer IDs are not contiguous: tokenizer reports {tokenizer_vocab_size}, token bytes contain {} entries",
            token_bytes.len()
        );
        token_bytes.resize_with(logits_vocab_size, Vec::new);
        let mut eos_tokens = eos_tokens
            .iter()
            .copied()
            .map(|token| {
                u32::try_from(token).context("EOS token is negative")
            })
            .collect::<Result<Vec<_>>>()?;
        eos_tokens.sort_unstable();
        eos_tokens.dedup();
        ensure!(!eos_tokens.is_empty(), "EOS token list is empty");
        ensure!(
            eos_tokens
                .iter()
                .all(|&token| token < logits_vocab_size as u32),
            "EOS token is outside the model logits vocabulary"
        );
        let info = TokRxInfo {
            vocab_size: logits_vocab_size as u32,
            tok_eos: eos_tokens[0],
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let tokenizer = Tokenizer::from_bytes(
            serde_json::to_vec(tokenizer_json).context("failed to serialize tokenizer.json")?,
        )
        .map_err(anyhow::Error::msg)
        .context("failed to initialize canonical structured-output tokenizer")?;
        let eos_tokens: Arc<[TokenId]> = eos_tokens.into();
        let token_bytes = Arc::new(token_bytes);
        let env: TokEnv = Arc::new(LocalTokenizerEnv {
            trie: TokTrie::from(&info, token_bytes.as_ref()).with_eos_tokens(&eos_tokens),
            tokenizer,
        });
        let parser = constraint_parser_factory(&env)
            .context("failed to initialize structured-output parser")?;
        Ok(Self {
            parser,
            token_bytes,
            eos_tokens,
        })
    }

    #[cfg(test)]
    pub fn single_byte() -> Result<Self> {
        let env = toktrie::ApproximateTokEnv::single_byte_env();
        let token_bytes = Arc::new(
            (0..env.tok_trie().vocab_size())
                .map(|token| env.tok_trie().token(token as u32).to_vec())
                .collect(),
        );
        let eos_tokens = Arc::from(env.tok_trie().eos_tokens());
        let parser = constraint_parser_factory(&env)?;
        Ok(Self {
            parser,
            token_bytes,
            eos_tokens,
        })
    }

    pub fn compile(
        &self,
        format: &OutputFormat,
        tools: &[ChatTool],
        parallel_tool_calls: bool,
    ) -> Result<Option<GuidanceConstraint>> {
        let schema = match format {
            OutputFormat::Text if tools.is_empty() => return Ok(None),
            OutputFormat::Text => {
                let grammar = tool_call_lark(tools, parallel_tool_calls)?;
                let parser = self
                    .parser
                    .create_parser(TopLevelGrammar {
                        grammars: vec![GrammarWithLexer {
                            name: None,
                            json_schema: None,
                            lark_grammar: Some(grammar),
                        }],
                        max_tokens: None,
                    })
                    .context("failed to compile tool-call grammar")?;
                return Ok(Some(GuidanceConstraint::tool_call(
                    Constraint::new(parser),
                    Arc::clone(&self.token_bytes),
                    Arc::clone(&self.eos_tokens),
                    !parallel_tool_calls,
                )));
            }
            OutputFormat::JsonObject => Value::Object(Map::new()),
            OutputFormat::JsonSchema { schema, .. } => validate_and_normalize_schema(schema)?,
        };
        let grammar = TopLevelGrammar {
            grammars: vec![GrammarWithLexer {
                name: None,
                json_schema: Some(schema),
                lark_grammar: None,
            }],
            max_tokens: None,
        };
        let parser = self
            .parser
            .create_parser(grammar)
            .context("failed to compile structured-output grammar")?;
        Ok(Some(GuidanceConstraint::grammar(
            Constraint::new(parser),
            Arc::clone(&self.eos_tokens),
        )))
    }
}

pub struct GuidanceConstraint {
    inner: GuidanceState,
    transaction: Option<GuidanceState>,
    eos_tokens: Arc<[TokenId]>,
}

enum GuidanceState {
    Grammar(Constraint),
    ToolCall(ToolCallState),
}

impl GuidanceState {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Grammar(inner) => Self::Grammar(inner.deep_clone()),
            Self::ToolCall(inner) => Self::ToolCall(inner.deep_clone()),
        }
    }
}

struct ToolCallState {
    inner: Constraint,
    token_bytes: Arc<Vec<Vec<u8>>>,
    marker_prefix: usize,
    active: bool,
    accept_on_accepting: bool,
}

impl ToolCallState {
    fn deep_clone(&self) -> Self {
        Self {
            inner: self.inner.deep_clone(),
            token_bytes: Arc::clone(&self.token_bytes),
            marker_prefix: self.marker_prefix,
            active: self.active,
            accept_on_accepting: self.accept_on_accepting,
        }
    }
}

impl GuidanceConstraint {
    fn grammar(inner: Constraint, eos_tokens: Arc<[TokenId]>) -> Self {
        Self {
            inner: GuidanceState::Grammar(inner),
            transaction: None,
            eos_tokens,
        }
    }

    fn tool_call(
        inner: Constraint,
        token_bytes: Arc<Vec<Vec<u8>>>,
        eos_tokens: Arc<[TokenId]>,
        accept_on_accepting: bool,
    ) -> Self {
        Self {
            inner: GuidanceState::ToolCall(ToolCallState {
                inner,
                token_bytes,
                marker_prefix: 0,
                active: false,
                accept_on_accepting,
            }),
            transaction: None,
            eos_tokens,
        }
    }

    fn active(&mut self) -> &mut GuidanceState {
        self.transaction.as_mut().unwrap_or(&mut self.inner)
    }
}

fn tool_call_lark(tools: &[ChatTool], parallel_tool_calls: bool) -> Result<String> {
    ensure!(!tools.is_empty(), "tool-call grammar requires at least one tool");
    let mut grammar = String::from("start: function \"</tool_call>\"");
    if parallel_tool_calls {
        grammar.push_str(
            " tool_call*\ntool_call: \"<tool_call>\" function \"</tool_call>\"",
        );
    }
    grammar.push_str("\nfunction: ");
    for index in 0..tools.len() {
        if index != 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("function_{index}"));
    }
    grammar.push('\n');

    const ROOT_KEYWORDS: &[&str] = &[
        "$schema",
        "$id",
        "$defs",
        "definitions",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "properties",
        "required",
        "additionalProperties",
    ];
    for (function_index, tool) in tools.iter().enumerate() {
        validate_tag_name(&tool.function.name, "function")?;
        let schema = validate_and_normalize_schema(&tool.function.parameters)
            .with_context(|| format!("invalid parameters for tool {:?}", tool.function.name))?;
        let object = schema
            .as_object()
            .expect("normalized tool parameters must be an object");
        for key in object.keys() {
            ensure!(
                ROOT_KEYWORDS.contains(&key.as_str()),
                "tool {:?} uses root JSON Schema keyword {key:?}, which cannot be represented as ordered parameters",
                tool.function.name
            );
        }
        if let Some(schema_type) = object.get("type") {
            ensure!(
                schema_type.as_str() == Some("object"),
                "tool {:?} parameters must have JSON Schema type \"object\"",
                tool.function.name
            );
        }
        let empty_properties = Map::new();
        let properties = match object.get("properties") {
            None => &empty_properties,
            Some(Value::Object(properties)) => properties,
            Some(_) => bail!(
                "tool {:?} JSON Schema properties must be an object",
                tool.function.name
            ),
        };
        ensure!(
            properties.len() <= MAX_TOOL_PARAMETERS,
            "tool {:?} declares too many parameters",
            tool.function.name
        );
        let mut required = BTreeSet::new();
        if let Some(required_values) = object.get("required") {
            let required_values = required_values.as_array().with_context(|| {
                format!(
                    "tool {:?} JSON Schema required must be an array",
                    tool.function.name
                )
            })?;
            for value in required_values {
                let name = value.as_str().with_context(|| {
                    format!(
                        "tool {:?} JSON Schema required entries must be strings",
                        tool.function.name
                    )
                })?;
                ensure!(
                    required.insert(name.to_string()),
                    "tool {:?} lists required parameter {name:?} more than once",
                    tool.function.name
                );
                ensure!(
                    properties.contains_key(name),
                    "tool {:?} requires undeclared parameter {name:?}",
                    tool.function.name
                );
            }
        }

        grammar.push_str(&format!(
            "function_{function_index}: {}",
            lark_literal(&format!("<function={}>", tool.function.name))?
        ));
        for (parameter_index, (name, schema)) in properties.iter().enumerate() {
            validate_tag_name(name, "parameter")?;
            validate_nested_references(schema, &tool.function.name, name)?;
            grammar.push(' ');
            grammar.push_str(&format!("parameter_{function_index}_{parameter_index}"));
            if !required.contains(name.as_str()) {
                grammar.push('?');
            }
        }
        grammar.push(' ');
        grammar.push_str(&lark_literal("</function>")?);
        grammar.push('\n');

        for (parameter_index, (name, schema)) in properties.iter().enumerate() {
            let nested_schema = nested_property_schema(object, schema);
            grammar.push_str(&format!(
                "parameter_{function_index}_{parameter_index}: {} value_{function_index}_{parameter_index} {}\n",
                lark_literal(&format!("<parameter={name}>"))?,
                lark_literal("</parameter>")?,
            ));
            grammar.push_str(&format!(
                "value_{function_index}_{parameter_index}: %json {}\n",
                serde_json::to_string(&nested_schema)
                    .context("failed to serialize tool parameter schema")?
            ));
        }
    }
    Ok(grammar)
}

fn lark_literal(value: &str) -> Result<String> {
    serde_json::to_string(value).context("failed to escape tool-call grammar literal")
}

fn validate_tag_name(name: &str, kind: &str) -> Result<()> {
    ensure!(!name.is_empty(), "{kind} name must not be empty");
    ensure!(
        name.len() <= MAX_TOOL_CALL_NAME_BYTES,
        "{kind} name is too long"
    );
    ensure!(
        name.trim() == name,
        "{kind} name must not have leading or trailing whitespace"
    );
    ensure!(
        !name.bytes().any(|byte| matches!(byte, b'<' | b'>')),
        "{kind} name contains an XML delimiter"
    );
    ensure!(
        !(name.len() >= 2
            && ((name.starts_with('"') && name.ends_with('"'))
                || (name.starts_with('\'') && name.ends_with('\'')))),
        "{kind} name cannot be represented without changing its value"
    );
    Ok(())
}

fn validate_nested_references(schema: &Value, tool: &str, parameter: &str) -> Result<()> {
    match schema {
        Value::Array(values) => {
            for value in values {
                validate_nested_references(value, tool, parameter)?;
            }
        }
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                ensure!(
                    reference == "#"
                        || reference.starts_with("#/$defs/")
                        || reference.starts_with("#/definitions/"),
                    "tool {tool:?} parameter {parameter:?} uses reference {reference:?}, which cannot be represented independently"
                );
            }
            for value in object.values() {
                validate_nested_references(value, tool, parameter)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn nested_property_schema(root: &Map<String, Value>, property: &Value) -> Value {
    let Value::Object(property) = property else {
        return property.clone();
    };
    let mut nested = property.clone();
    for key in ["$defs", "definitions"] {
        if let Some(definitions) = root.get(key) {
            nested.insert(key.to_string(), definitions.clone());
        }
    }
    Value::Object(nested)
}

fn constraint_parser_factory(env: &TokEnv) -> Result<ParserFactory> {
    ParserFactory::new(
        env,
        InferenceCapabilities {
            ff_tokens: true,
            conditional_ff_tokens: false,
            backtrack: true,
            fork: false,
        },
        &llguidance::earley::SlicedBiasComputer::general_slices(),
    )
}

fn guidance_commit(
    result: llguidance::CommitResult,
    accepting: bool,
) -> std::result::Result<ConstraintCommit, String> {
    let tokens = result
        .ff_tokens
        .into_iter()
        .map(|token| {
            i32::try_from(token)
                .map_err(|_| "structured-output grammar produced a token outside i32".to_string())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(ConstraintCommit {
        backtrack: result.backtrack as usize,
        tokens,
        accept: result.stop || accepting,
    })
}

fn guidance_mask(
    active: &mut Constraint,
    eos_tokens: &[TokenId],
    accept_on_accepting: bool,
) -> std::result::Result<ConstraintMask, String> {
    let parser_accepting = active.parser.is_accepting();
    let step = active.compute_mask().map_err(|error| error.to_string())?;
    if step.is_stop() {
        return Ok(ConstraintMask::Accept);
    }
    if let Some(mask) = step.sample_mask.as_ref() {
        let mut allowed = Vec::new();
        mask.iter_set_entries(|token| {
            let token = token as TokenId;
            if parser_accepting || eos_tokens.binary_search(&token).is_err() {
                allowed.push(token as i32);
            }
        });
        if allowed.is_empty() {
            return Err("structured-output grammar produced an empty token mask".to_string());
        }
        return Ok(ConstraintMask::Allow(allowed));
    }
    let result = active
        .commit_token(None)
        .map_err(|error| error.to_string())?;
    let accepting = accept_on_accepting && active.parser.is_accepting();
    guidance_commit(result, accepting).map(ConstraintMask::Splice)
}

fn guidance_token(
    active: &mut Constraint,
    token: TokenId,
    accept_on_accepting: bool,
) -> std::result::Result<ConstraintCommit, String> {
    let result = active
        .commit_token(Some(token))
        .map_err(|error| error.to_string())?;
    let accepting = accept_on_accepting && active.parser.is_accepting();
    guidance_commit(result, accepting)
}

impl ToolCallState {
    fn commit_passthrough(
        &mut self,
        token: TokenId,
    ) -> std::result::Result<ConstraintCommit, String> {
        let token_index = token as usize;
        let token_bytes = self.token_bytes.get(token_index).ok_or_else(|| {
            "tool-call constraint received a token outside the vocabulary".to_string()
        })?;
        let token_bytes = token_bytes
            .strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER])
            .unwrap_or(token_bytes);
        for (index, &byte) in token_bytes.iter().enumerate() {
            if byte == TOOL_CALL_OPEN[self.marker_prefix] {
                self.marker_prefix += 1;
                if self.marker_prefix == TOOL_CALL_OPEN.len() {
                    self.marker_prefix = 0;
                    if index + 1 != token_bytes.len() {
                        return Err(
                            "tool-call marker completed before the end of a token".to_string(),
                        );
                    }
                    self.inner.start_without_prompt();
                    self.active = true;
                    return Ok(ConstraintCommit::token(token as i32));
                }
            } else {
                self.marker_prefix = usize::from(byte == TOOL_CALL_OPEN[0]);
            }
        }
        Ok(ConstraintCommit::token(token as i32))
    }
}

impl TokenConstraint for GuidanceConstraint {
    fn begin_transaction(&mut self) -> std::result::Result<(), String> {
        if self.transaction.is_some() {
            return Err("structured-output constraint transaction is already active".to_string());
        }
        self.transaction = Some(self.inner.deep_clone());
        Ok(())
    }

    fn commit_transaction(&mut self) -> std::result::Result<(), String> {
        self.inner = self
            .transaction
            .take()
            .ok_or_else(|| "structured-output constraint transaction is not active".to_string())?;
        Ok(())
    }

    fn rollback_transaction(&mut self) {
        self.transaction = None;
    }

    fn compute_mask(
        &mut self,
        _logits: &mlxcel_core::MlxArray,
        _token_history: &[i32],
    ) -> std::result::Result<ConstraintMask, String> {
        let eos_tokens = self.eos_tokens.as_ref();
        let active = self.transaction.as_mut().unwrap_or(&mut self.inner);
        match active {
            GuidanceState::Grammar(active) => guidance_mask(active, eos_tokens, true),
            GuidanceState::ToolCall(active) if !active.active => {
                Ok(ConstraintMask::PassThrough)
            }
            GuidanceState::ToolCall(active) => {
                let accept_on_accepting = active.accept_on_accepting;
                guidance_mask(&mut active.inner, eos_tokens, accept_on_accepting)
            }
        }
    }

    fn commit_token(&mut self, token_id: i32) -> std::result::Result<ConstraintCommit, String> {
        let token = u32::try_from(token_id)
            .map_err(|_| "structured-output grammar received a negative token".to_string())?;
        match self.active() {
            GuidanceState::Grammar(active) => guidance_token(active, token, true),
            GuidanceState::ToolCall(active) if !active.active => {
                active.commit_passthrough(token)
            }
            GuidanceState::ToolCall(active) => {
                let accept_on_accepting = active.accept_on_accepting;
                guidance_token(&mut active.inner, token, accept_on_accepting)
            }
        }
    }
}

pub fn validate_and_normalize_schema(schema: &Value) -> Result<Value> {
    ensure!(schema.is_object(), "JSON Schema must be an object");
    let mut normalized = schema.clone();
    validate_schema_node(&mut normalized, false)?;
    Ok(normalized)
}

fn validate_schema_node(value: &mut Value, property_map: bool) -> Result<()> {
    let Some(object) = value.as_object_mut() else {
        if value.is_boolean() {
            return Ok(());
        }
        bail!("JSON Schema nodes must be objects or booleans");
    };
    if property_map {
        for child in object.values_mut() {
            validate_schema_node(child, false)?;
        }
        return Ok(());
    }

    const KEYWORDS: &[&str] = &[
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "enum",
        "const",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "format",
        "contentEncoding",
        "contentMediaType",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxContains",
        "minContains",
        "items",
        "prefixItems",
        "contains",
        "maxProperties",
        "minProperties",
        "required",
        "properties",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "dependentRequired",
        "dependentSchemas",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    ];
    let known: BTreeSet<&str> = KEYWORDS.iter().copied().collect();
    for key in object.keys() {
        ensure!(
            known.contains(key.as_str()),
            "unknown JSON Schema keyword {key:?}"
        );
    }

    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        ensure!(
            reference == "#" || reference.starts_with("#/"),
            "external or remote $ref is not supported"
        );
    }
    if let Some(format) = object.get("format").and_then(Value::as_str) {
        ensure!(
            matches!(
                format,
                "date-time"
                    | "time"
                    | "date"
                    | "duration"
                    | "email"
                    | "hostname"
                    | "ipv4"
                    | "ipv6"
                    | "uuid"
                    | "uri"
            ),
            "unsupported JSON Schema format {format:?}"
        );
    }
    if let Some(pattern) = object.get("pattern").and_then(Value::as_str) {
        reject_regex_lookaround(pattern)?;
    }
    if object
        .get("additionalProperties")
        .is_some_and(|value| value.is_object())
    {
        bail!(
            "schema-valued additionalProperties requires property-name uniqueness that cannot be enforced"
        );
    }
    if object
        .get("patternProperties")
        .and_then(Value::as_object)
        .is_some_and(|patterns| !patterns.is_empty())
    {
        bail!("patternProperties requires property-name uniqueness that cannot be enforced");
    }
    if object.get("uniqueItems") == Some(&Value::Bool(true)) {
        bail!("uniqueItems cannot be enforced during generation");
    }

    if let Some(one_of) = object.remove("oneOf") {
        let branches = one_of.as_array().context("oneOf must contain an array")?;
        ensure!(!branches.is_empty(), "oneOf must not be empty");
        let mut types = BTreeSet::new();
        for branch in branches {
            let branch_type = branch
                .as_object()
                .and_then(|branch| branch.get("type"))
                .and_then(Value::as_str)
                .context("oneOf branches must have distinct string type values")?;
            ensure!(
                types.insert(branch_type.to_string()),
                "oneOf branches are not disjoint"
            );
        }
        object.insert("anyOf".to_string(), one_of);
    }

    for key in ["properties", "$defs", "definitions", "dependentSchemas"] {
        if let Some(child) = object.get_mut(key) {
            validate_schema_node(child, true)?;
        }
    }
    if let Some(patterns) = object.get("patternProperties").and_then(Value::as_object) {
        for pattern in patterns.keys() {
            reject_regex_lookaround(pattern)?;
        }
    }
    for key in [
        "items",
        "contains",
        "additionalProperties",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    ] {
        if let Some(child) = object.get_mut(key) {
            validate_schema_node(child, false)?;
        }
    }
    for key in ["prefixItems", "allOf", "anyOf"] {
        if let Some(children) = object.get_mut(key).and_then(Value::as_array_mut) {
            for child in children {
                validate_schema_node(child, false)?;
            }
        }
    }
    Ok(())
}

fn reject_regex_lookaround(pattern: &str) -> Result<()> {
    ensure!(
        !["(?=", "(?!", "(?<=", "(?<!"]
            .iter()
            .any(|needle| pattern.contains(needle)),
        "regex lookarounds are not supported"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qw_runtime::ChatToolFunction;
    use serde_json::json;

    fn mask_signature(mask: ConstraintMask) -> (u8, Vec<i32>) {
        match mask {
            ConstraintMask::PassThrough => (0, Vec::new()),
            ConstraintMask::Allow(tokens) => (1, tokens),
            ConstraintMask::Splice(commit) => {
                let mut values = vec![commit.backtrack as i32, i32::from(commit.accept)];
                values.extend(commit.tokens);
                (2, values)
            }
            ConstraintMask::Accept => (3, Vec::new()),
        }
    }

    fn tool(name: &str, parameters: Value) -> ChatTool {
        ChatTool {
            tool_type: "function".to_string(),
            function: ChatToolFunction {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
            },
        }
    }

    fn logits() -> impl std::ops::Deref<Target = mlxcel_core::MlxArray> {
        mlxcel_core::from_slice_f32(&[0.0; 262], &[1, 1, 262])
    }
    struct CanonicalTestTokenizerEnv {
        trie: TokTrie,
    }

    impl TokenizerEnv for CanonicalTestTokenizerEnv {
        fn tok_trie(&self) -> &TokTrie {
            &self.trie
        }

        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
            self.trie.greedy_tokenize(bytes)
        }

        fn tokenize_is_canonical(&self) -> bool {
            true
        }
    }
    const PRIMARY_EOS: TokenId = 248_044;
    const SECONDARY_EOS: TokenId = 248_046;
    const TOOL_CALL_TOKEN: TokenId = 248_058;
    const QWEN_VOCAB_SIZE: usize = 248_064;

    fn qwen_token_factory() -> Result<GrammarFactory> {
        let base = toktrie::ApproximateTokEnv::single_byte();
        let mut token_bytes = (0..base.tok_trie().vocab_size())
            .map(|token| base.tok_trie().token(token as TokenId).to_vec())
            .collect::<Vec<_>>();
        token_bytes.resize_with(QWEN_VOCAB_SIZE, Vec::new);
        token_bytes[TOOL_CALL_TOKEN as usize] = TOOL_CALL_OPEN.to_vec();
        let info = TokRxInfo::new(QWEN_VOCAB_SIZE as u32, PRIMARY_EOS);
        let trie =
            TokTrie::from(&info, &token_bytes).with_eos_tokens(&[PRIMARY_EOS, SECONDARY_EOS]);
        let env: TokEnv = Arc::new(CanonicalTestTokenizerEnv { trie });
        let parser = constraint_parser_factory(&env)?;
        Ok(GrammarFactory {
            parser,
            token_bytes: Arc::new(token_bytes),
            eos_tokens: Arc::from([PRIMARY_EOS, SECONDARY_EOS]),
        })
    }


    fn output_bytes(output: &[i32]) -> Vec<u8> {
        output
            .iter()
            .map(|&token| u8::try_from(token).expect("single-byte token"))
            .collect()
    }

    fn drive_to(
        constraint: &mut GuidanceConstraint,
        logits: &mlxcel_core::MlxArray,
        output: &mut Vec<i32>,
        target: &[u8],
    ) -> bool {
        let mut accepted = false;
        for _ in 0..1024 {
            let bytes = output_bytes(output);
            assert!(target.starts_with(&bytes), "grammar emitted {bytes:?}");
            if bytes.len() == target.len() {
                return accepted;
            }
            match constraint
                .compute_mask(logits, output)
                .expect("constraint mask")
            {
                ConstraintMask::PassThrough => {
                    let token = target[bytes.len()] as i32;
                    let commit = constraint
                        .commit_token(token)
                        .expect("commit pass-through token");
                    accepted = commit.accept;
                    commit.apply_to(output).expect("apply pass-through token");
                }
                ConstraintMask::Allow(allowed) => {
                    let token = target[bytes.len()] as i32;
                    assert!(
                        allowed.contains(&token),
                        "target byte {:?} is not allowed",
                        target[bytes.len()]
                    );
                    let commit = constraint.commit_token(token).expect("commit allowed token");
                    accepted = commit.accept;
                    commit.apply_to(output).expect("apply allowed token");
                }
                ConstraintMask::Splice(commit) => {
                    accepted = commit.accept;
                    commit.apply_to(output).expect("apply fast-forward");
                }
                ConstraintMask::Accept => return true,
            }
            if accepted {
                assert_eq!(output_bytes(output).as_slice(), target);
                return true;
            }
        }
        panic!("constraint did not reach target");
    }

    #[test]
    fn guidance_transaction_rollback_restores_parser_state() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let mut constraint = factory
            .compile(
                &OutputFormat::JsonSchema {
                    name: "one".to_string(),
                    schema: json!({"type":"integer","const":1}),
                },
                &[],
                false,
            )
            .expect("compile grammar")
            .expect("constraint");
        let logits = logits();

        constraint.begin_transaction().expect("begin transaction");
        let speculative = mask_signature(
            constraint
                .compute_mask(&logits, &[])
                .expect("speculative mask"),
        );
        constraint.rollback_transaction();
        let committed = mask_signature(
            constraint
                .compute_mask(&logits, &[])
                .expect("committed mask"),
        );
        assert_eq!(speculative, committed);
    }

    #[test]
    fn guidance_fast_forward_produces_schema_valid_json() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let mut constraint = factory
            .compile(
                &OutputFormat::JsonSchema {
                    name: "one".to_string(),
                    schema: json!({"type":"integer","const":1}),
                },
                &[],
                false,
            )
            .expect("compile grammar")
            .expect("constraint");
        let logits = logits();
        let mut output = Vec::new();

        for _ in 0..16 {
            match constraint
                .compute_mask(&logits, &output)
                .expect("constraint mask")
            {
                ConstraintMask::PassThrough => panic!("structured grammar passed through"),
                ConstraintMask::Allow(allowed) => {
                    let token = if allowed.contains(&(b'1' as i32)) {
                        b'1' as i32
                    } else {
                        *allowed.first().expect("non-empty allowed set")
                    };
                    constraint
                        .commit_token(token)
                        .expect("commit allowed token")
                        .apply_to(&mut output)
                        .expect("apply token commit");
                }
                ConstraintMask::Splice(commit) => {
                    let accept = commit.accept;
                    commit.apply_to(&mut output).expect("apply fast-forward");
                    if accept {
                        break;
                    }
                }
                ConstraintMask::Accept => break,
            }
        }

        let value: Value =
            serde_json::from_slice(&output_bytes(&output)).expect("valid generated JSON");
        assert_eq!(value, json!(1));
    }

    #[test]
    fn tool_constraint_passes_through_until_complete_marker() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [tool("empty", json!({"type":"object","properties":{}}))];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, false)
            .expect("compile tool grammar")
            .expect("constraint");
        let logits = logits();
        let mut output = Vec::new();

        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            b"ordinary<tool_",
        ));
        assert!(matches!(
            constraint.compute_mask(&logits, &output),
            Ok(ConstraintMask::PassThrough)
        ));
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            b"ordinary<tool_call>",
        ));
        assert!(!matches!(
            constraint.compute_mask(&logits, &output),
            Ok(ConstraintMask::PassThrough)
        ));
    }

    #[test]
    fn special_tool_marker_token_activates_and_constrains_read_body_transactionally() {
        const TOOL_CALL_TOKEN_ID: usize = 248_058;

        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [tool(
            "read",
            json!({
                "type":"object",
                "properties":{"path":{"type":"string"}},
                "required":["path"],
                "additionalProperties":false
            }),
        )];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, false)
            .expect("compile tool grammar")
            .expect("constraint");
        let GuidanceState::ToolCall(state) = &mut constraint.inner else {
            panic!("tool-call constraint");
        };
        let mut token_bytes = state.token_bytes.as_ref().clone();
        token_bytes.resize_with(TOOL_CALL_TOKEN_ID + 1, Vec::new);
        token_bytes[TOOL_CALL_TOKEN_ID] = [
            &[TokTrie::SPECIAL_TOKEN_MARKER][..],
            TOOL_CALL_OPEN,
        ]
        .concat();
        state.token_bytes = Arc::new(token_bytes);
        let logits = logits();

        constraint.begin_transaction().expect("begin transaction");
        constraint
            .commit_token(TOOL_CALL_TOKEN_ID as i32)
            .expect("commit speculative special marker");
        assert!(!matches!(
            constraint.compute_mask(&logits, &[]),
            Ok(ConstraintMask::PassThrough)
        ));
        constraint.rollback_transaction();
        assert!(matches!(
            constraint.compute_mask(&logits, &[]),
            Ok(ConstraintMask::PassThrough)
        ));

        constraint.begin_transaction().expect("begin transaction");
        constraint
            .commit_token(TOOL_CALL_TOKEN_ID as i32)
            .expect("commit accepted special marker");
        constraint
            .commit_transaction()
            .expect("commit marker transaction");
        let target = b"<function=read><parameter=path>\"Cargo.toml\"</parameter></function></tool_call>";
        let mut output = Vec::new();
        assert!(drive_to(
            &mut constraint,
            &logits,
            &mut output,
            target
        ));
        assert_eq!(output_bytes(&output).as_slice(), target);
    }

    #[test]
    fn tool_constraint_rejects_a_token_with_bytes_after_the_marker() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [tool("empty", json!({"type":"object","properties":{}}))];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, false)
            .expect("compile tool grammar")
            .expect("constraint");
        let GuidanceState::ToolCall(state) = &mut constraint.inner else {
            panic!("tool-call constraint");
        };
        state.token_bytes = Arc::new(vec![b"<tool_call>x".to_vec()]);

        assert_eq!(
            constraint.commit_token(0),
            Err("tool-call marker completed before the end of a token".to_string())
        );
    }

    #[test]
    fn tool_constraint_emits_declared_schema_values_in_property_order() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [tool(
            "weather",
            json!({
                "type":"object",
                "properties":{
                    "city":{"type":"string"},
                    "days":{"type":"integer","minimum":1},
                    "units":{"type":"string","enum":["c","f"]}
                },
                "required":["city","days"],
                "additionalProperties":false
            }),
        )];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, false)
            .expect("compile tool grammar")
            .expect("constraint");
        let logits = logits();
        let target = b"<tool_call><function=weather><parameter=city>\"Paris\"</parameter><parameter=days>2</parameter><parameter=units>\"c\"</parameter></function></tool_call>";
        let mut output = Vec::new();

        assert!(drive_to(
            &mut constraint,
            &logits,
            &mut output,
            target
        ));
        let parsed = crate::tool_calls::parse_assistant_output(
            std::str::from_utf8(target).expect("ASCII tool call"),
            &["weather"],
            output.len(),
            output.len(),
        )
        .expect("parse constrained tool call");
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.tool_calls[0].arguments)
                .expect("tool arguments"),
            json!({"city":"Paris","days":2,"units":"c"})
        );
    }

    #[test]
    fn non_parallel_tool_constraint_accepts_at_the_close_tag() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [tool("empty", json!({"type":"object","properties":{}}))];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, false)
            .expect("compile tool grammar")
            .expect("constraint");
        let logits = logits();
        let target = b"<tool_call><function=empty></function></tool_call>";
        let mut output = Vec::new();

        assert!(drive_to(
            &mut constraint,
            &logits,
            &mut output,
            target
        ));
        assert_eq!(output_bytes(&output).as_slice(), target);
    }

    #[test]
    fn parallel_tool_constraint_allows_a_second_call() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [
            tool("first", json!({"type":"object","properties":{}})),
            tool("second", json!({"type":"object","properties":{}})),
        ];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, true)
            .expect("compile tool grammar")
            .expect("constraint");
        let logits = logits();
        let target = b"<tool_call><function=first></function></tool_call><tool_call><function=second></function></tool_call>";
        let mut output = Vec::new();

        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            target
        ));
        assert_eq!(output_bytes(&output).as_slice(), target);
        assert!(!matches!(
            constraint.compute_mask(&logits, &output),
            Ok(ConstraintMask::PassThrough)
        ));
    }

    #[test]
    fn qwen_eos_tokens_are_masked_until_parallel_tool_call_is_complete() {
        let factory = qwen_token_factory().expect("Qwen-token grammar");
        let tools = [tool(
            "read",
            json!({
                "type":"object",
                "properties":{"path":{"type":"string"}},
                "required":["path"],
                "additionalProperties":false
            }),
        )];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, true)
            .expect("compile tool grammar")
            .expect("constraint");
        let logits = logits();
        constraint
            .commit_token(TOOL_CALL_TOKEN as i32)
            .expect("activate real tool-call token");

        let mut output = Vec::new();
        let string_prefix = b"<function=read><parameter=path>\"";
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            string_prefix
        ));
        let ConstraintMask::Allow(string_mask) = constraint
            .compute_mask(&logits, &output)
            .expect("path string mask")
        else {
            panic!("path string must produce a sample mask");
        };
        assert!(!string_mask.contains(&(PRIMARY_EOS as i32)));
        assert!(!string_mask.contains(&(SECONDARY_EOS as i32)));

        let body_prefix = b"<function=read><parameter=path>\"Cargo";
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            body_prefix
        ));
        let ConstraintMask::Allow(body_mask) = constraint
            .compute_mask(&logits, &output)
            .expect("path body mask")
        else {
            panic!("path body must produce a sample mask");
        };
        assert!(!body_mask.contains(&(PRIMARY_EOS as i32)));
        assert!(!body_mask.contains(&(SECONDARY_EOS as i32)));
        constraint.begin_transaction().expect("begin transaction");
        assert!(constraint.commit_token(PRIMARY_EOS as i32).is_err());
        constraint.rollback_transaction();
        constraint.begin_transaction().expect("begin transaction");
        assert!(constraint.commit_token(SECONDARY_EOS as i32).is_err());
        constraint.rollback_transaction();

        let checkpoint = output.clone();
        let complete =
            b"<function=read><parameter=path>\"Cargo.toml\"</parameter></function></tool_call>";
        constraint.begin_transaction().expect("begin transaction");
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            complete
        ));
        let ConstraintMask::Allow(accepting_mask) = constraint
            .compute_mask(&logits, &output)
            .expect("accepting parallel mask")
        else {
            panic!("accepting parallel grammar must produce a sample mask");
        };
        assert!(accepting_mask.contains(&(PRIMARY_EOS as i32)));
        assert!(accepting_mask.contains(&(SECONDARY_EOS as i32)));
        assert!(accepting_mask.contains(&(TOOL_CALL_TOKEN as i32)));
        constraint
            .commit_token(SECONDARY_EOS as i32)
            .expect("commit secondary EOS");
        assert!(matches!(
            constraint.compute_mask(&logits, &output),
            Ok(ConstraintMask::Accept)
        ));

        constraint.rollback_transaction();
        output = checkpoint.clone();
        let ConstraintMask::Allow(rolled_back_mask) = constraint
            .compute_mask(&logits, &output)
            .expect("rolled-back path body mask")
        else {
            panic!("rolled-back path body must produce a sample mask");
        };
        assert!(!rolled_back_mask.contains(&(PRIMARY_EOS as i32)));
        assert!(!rolled_back_mask.contains(&(SECONDARY_EOS as i32)));

        constraint.begin_transaction().expect("begin transaction");
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            complete
        ));
        let ConstraintMask::Allow(primary_accepting_mask) = constraint
            .compute_mask(&logits, &output)
            .expect("accepting parallel mask")
        else {
            panic!("accepting parallel grammar must produce a sample mask");
        };
        assert!(primary_accepting_mask.contains(&(PRIMARY_EOS as i32)));
        assert!(primary_accepting_mask.contains(&(SECONDARY_EOS as i32)));
        constraint
            .commit_token(PRIMARY_EOS as i32)
            .expect("commit primary EOS");
        constraint
            .commit_transaction()
            .expect("commit primary EOS transaction");
        assert!(matches!(
            constraint.compute_mask(&logits, &output),
            Ok(ConstraintMask::Accept)
        ));
    }

    #[test]
    fn tool_constraint_transaction_restores_pretrigger_and_active_state() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let tools = [tool("empty", json!({"type":"object","properties":{}}))];
        let mut constraint = factory
            .compile(&OutputFormat::Text, &tools, false)
            .expect("compile tool grammar")
            .expect("constraint");
        let logits = logits();
        let mut output = Vec::new();
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            b"<tool_",
        ));

        constraint.begin_transaction().expect("begin transaction");
        let mut speculative_output = output.clone();
        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut speculative_output,
            b"<tool_call>",
        ));
        let speculative_active = mask_signature(
            constraint
                .compute_mask(&logits, &speculative_output)
                .expect("active speculative mask"),
        );
        assert_ne!(speculative_active.0, 0);
        constraint.rollback_transaction();
        assert!(matches!(
            constraint.compute_mask(&logits, &output),
            Ok(ConstraintMask::PassThrough)
        ));

        assert!(!drive_to(
            &mut constraint,
            &logits,
            &mut output,
            b"<tool_call>",
        ));
        constraint.begin_transaction().expect("begin active transaction");
        let speculative = mask_signature(
            constraint
                .compute_mask(&logits, &output)
                .expect("speculative active mask"),
        );
        constraint.rollback_transaction();
        let committed = mask_signature(
            constraint
                .compute_mask(&logits, &output)
                .expect("committed active mask"),
        );
        assert_eq!(speculative, committed);
    }

    #[test]
    fn tool_constraint_rejects_unrepresentable_names_and_parameter_schemas() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        assert!(
            factory
                .compile(
                    &OutputFormat::Text,
                    &[tool("<bad", json!({"type":"object"}))],
                    false,
                )
                .is_err()
        );
        assert!(
            factory
                .compile(
                    &OutputFormat::Text,
                    &[tool(
                        "bad_schema",
                        json!({
                            "type":"object",
                            "properties":{},
                            "required":["missing"]
                        }),
                    )],
                    false,
                )
                .is_err()
        );
    }
}
