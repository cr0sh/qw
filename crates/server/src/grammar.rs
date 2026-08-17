use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use llguidance::{Constraint, ParserFactory, token_bytes_from_tokenizer_json};
use mlxcel_core::generate::{ConstraintCommit, ConstraintMask, TokenConstraint};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;
use toktrie::{
    InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
};

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
        let tokenizer = Tokenizer::from_bytes(
            serde_json::to_vec(tokenizer_json)
                .context("failed to serialize tokenizer.json")?,
        )
        .map_err(anyhow::Error::msg)
        .context("failed to initialize canonical structured-output tokenizer")?;
        let env: TokEnv = Arc::new(LocalTokenizerEnv {
            trie: TokTrie::from(&info, &token_bytes),
            tokenizer,
        });
        let parser = constraint_parser_factory(&env)
            .context("failed to initialize structured-output parser")?;
        Ok(Self { parser })
    }

    #[cfg(test)]
    pub fn single_byte() -> Result<Self> {
        let env = toktrie::ApproximateTokEnv::single_byte_env();
        let parser = constraint_parser_factory(&env)?;
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
            transaction: None,
        }))
    }
}

pub struct GuidanceConstraint {
    inner: Constraint,
    transaction: Option<Constraint>,
}

impl GuidanceConstraint {
    fn active(&mut self) -> &mut Constraint {
        self.transaction.as_mut().unwrap_or(&mut self.inner)
    }
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
        let active = self.active();
        let step = active.compute_mask().map_err(|error| error.to_string())?;
        if step.is_stop() {
            return Ok(ConstraintMask::Accept);
        }
        if let Some(mask) = step.sample_mask.as_ref() {
            let mut allowed = Vec::new();
            mask.iter_set_entries(|token| allowed.push(token as i32));
            if allowed.is_empty() {
                return Err("structured-output grammar produced an empty token mask".to_string());
            }
            return Ok(ConstraintMask::Allow(allowed));
        }
        let result = active
            .commit_token(None)
            .map_err(|error| error.to_string())?;
        let accepting = active.parser.is_accepting();
        guidance_commit(result, accepting).map(ConstraintMask::Splice)
    }

    fn commit_token(
        &mut self,
        token_id: i32,
    ) -> std::result::Result<ConstraintCommit, String> {
        let token = u32::try_from(token_id)
            .map_err(|_| "structured-output grammar received a negative token".to_string())?;
        let active = self.active();
        let result = active
            .commit_token(Some(token))
            .map_err(|error| error.to_string())?;
        let accepting = active.parser.is_accepting();
        guidance_commit(result, accepting)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mask_signature(mask: ConstraintMask) -> (u8, Vec<i32>) {
        match mask {
            ConstraintMask::Allow(tokens) => (0, tokens),
            ConstraintMask::Splice(commit) => {
                let mut values = vec![commit.backtrack as i32, i32::from(commit.accept)];
                values.extend(commit.tokens);
                (1, values)
            }
            ConstraintMask::Accept => (2, Vec::new()),
        }
    }

    #[test]
    fn guidance_transaction_rollback_restores_parser_state() {
        let factory = GrammarFactory::single_byte().expect("single-byte grammar");
        let mut constraint = factory
            .compile(&OutputFormat::JsonSchema {
                name: "one".to_string(),
                schema: json!({"type":"integer","const":1}),
            })
            .expect("compile grammar")
            .expect("constraint");
        let logits = mlxcel_core::from_slice_f32(&[0.0; 262], &[1, 1, 262]);

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
            .compile(&OutputFormat::JsonSchema {
                name: "one".to_string(),
                schema: json!({"type":"integer","const":1}),
            })
            .expect("compile grammar")
            .expect("constraint");
        let logits = mlxcel_core::from_slice_f32(&[0.0; 262], &[1, 1, 262]);
        let mut output = Vec::new();

        for _ in 0..16 {
            match constraint
                .compute_mask(&logits, &output)
                .expect("constraint mask")
            {
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

        let bytes = output
            .into_iter()
            .map(|token| u8::try_from(token).expect("single-byte token"))
            .collect::<Vec<_>>();
        let value: Value = serde_json::from_slice(&bytes).expect("valid JSON");
        assert_eq!(value, json!(1));
    }
}
