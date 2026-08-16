use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use llguidance::{Constraint, ParserFactory, token_bytes_from_tokenizer_json};
use mlxcel_core::generate::{ConstraintCommit, ConstraintMask, TokenConstraint};
use serde_json::{Map, Value};
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

use crate::protocol::OutputFormat;

struct LocalTokenizerEnv {
    trie: TokTrie,
}

impl TokenizerEnv for LocalTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(bytes)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

pub struct GrammarFactory {
    parser: ParserFactory,
}

impl GrammarFactory {
    pub fn from_tokenizer_json(
        tokenizer_json: &Value,
        tokenizer_vocab_size: usize,
        logits_vocab_size: usize,
        eos_token: u32,
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
        ensure!(
            eos_token < logits_vocab_size as u32,
            "EOS token is outside the model logits vocabulary"
        );
        let info = TokRxInfo {
            vocab_size: logits_vocab_size as u32,
            tok_eos: eos_token,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let env: TokEnv = Arc::new(LocalTokenizerEnv {
            trie: TokTrie::from(&info, &token_bytes),
        });
        let parser = ParserFactory::new_simple(&env)
            .context("failed to initialize structured-output parser")?;
        Ok(Self { parser })
    }

    #[cfg(test)]
    pub fn single_byte() -> Result<Self> {
        let env = toktrie::ApproximateTokEnv::single_byte_env();
        let parser = ParserFactory::new_simple(&env)?;
        Ok(Self { parser })
    }

    pub fn compile(&self, format: &OutputFormat) -> Result<Option<GuidanceConstraint>> {
        let schema = match format {
            OutputFormat::Text => return Ok(None),
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
        Ok(Some(GuidanceConstraint {
            inner: Constraint::new(parser),
        }))
    }
}

pub struct GuidanceConstraint {
    inner: Constraint,
}

impl TokenConstraint for GuidanceConstraint {
    fn compute_mask(
        &mut self,
        _logits: &mlxcel_core::MlxArray,
        _token_history: &[i32],
    ) -> std::result::Result<ConstraintMask, String> {
        let step = self.inner.compute_mask().map_err(|error| error.to_string())?;
        if step.is_stop() {
            return Ok(ConstraintMask::Accept);
        }
        let mask = step
            .sample_mask
            .as_ref()
            .ok_or_else(|| "structured-output grammar requested unsupported fast-forward tokens".to_string())?;
        let mut allowed = Vec::new();
        mask.iter_set_entries(|token| allowed.push(token as i32));
        if allowed.is_empty() {
            return Err("structured-output grammar produced an empty token mask".to_string());
        }
        Ok(ConstraintMask::Allow(allowed))
    }

    fn commit_token(
        &mut self,
        token_id: i32,
    ) -> std::result::Result<ConstraintCommit, String> {
        let token = u32::try_from(token_id)
            .map_err(|_| "structured-output grammar received a negative token".to_string())?;
        let result = self
            .inner
            .commit_token(Some(token))
            .map_err(|error| error.to_string())?;
        if result.backtrack != 0 || !result.ff_tokens.is_empty() {
            return Err(
                "structured-output grammar requested unsupported rollback or fast-forward"
                    .to_string(),
            );
        }
        Ok(if result.stop {
            ConstraintCommit::Accept
        } else {
            ConstraintCommit::Continue
        })
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
        "$schema", "$id", "$ref", "$defs", "definitions", "title", "description",
        "default", "examples", "deprecated", "readOnly", "writeOnly", "type", "enum",
        "const", "multipleOf", "maximum", "exclusiveMaximum", "minimum",
        "exclusiveMinimum", "maxLength", "minLength", "pattern", "format", "contentEncoding",
        "contentMediaType", "maxItems", "minItems", "uniqueItems", "maxContains",
        "minContains", "items", "prefixItems", "contains", "maxProperties", "minProperties",
        "required", "properties", "patternProperties", "additionalProperties", "propertyNames",
        "dependentRequired", "dependentSchemas", "allOf", "anyOf", "oneOf", "not", "if",
        "then", "else", "unevaluatedItems", "unevaluatedProperties",
    ];
    let known: BTreeSet<&str> = KEYWORDS.iter().copied().collect();
    for key in object.keys() {
        ensure!(known.contains(key.as_str()), "unknown JSON Schema keyword {key:?}");
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
                "date-time" | "time" | "date" | "duration" | "email" | "hostname"
                    | "ipv4" | "ipv6" | "uuid" | "uri"
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
        bail!("schema-valued additionalProperties requires property-name uniqueness that cannot be enforced");
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
        let branches = one_of
            .as_array()
            .context("oneOf must contain an array")?;
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
        "items", "contains", "additionalProperties", "propertyNames", "not", "if", "then",
        "else", "unevaluatedItems", "unevaluatedProperties",
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
