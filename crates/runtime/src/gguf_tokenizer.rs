use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::gguf::{GgufFile, MetadataValue};

const QWEN35_SPLIT_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub(crate) struct GgufTextAssets {
    pub tokenizer: Tokenizer,
    pub chat_template: String,
    pub bos_token: String,
    pub eos_token: String,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
}

impl GgufTextAssets {
    pub(crate) fn load(target: &GgufFile) -> Result<Self> {
        let metadata = target.metadata();
        let value = |key: &str| {
            metadata
                .get(key)
                .with_context(|| format!("selected GGUF is missing tokenizer metadata {key}"))
        };
        ensure!(
            value("tokenizer.ggml.model")?.as_str() == Some("gpt2")
                && value("tokenizer.ggml.pre")?.as_str() == Some("qwen35"),
            "selected GGUF tokenizer model/pre-tokenizer is unsupported"
        );
        let tokens = string_array(value("tokenizer.ggml.tokens")?, "tokenizer.ggml.tokens")?;
        let types = integer_array(
            value("tokenizer.ggml.token_type")?,
            "tokenizer.ggml.token_type",
        )?;
        let merges = string_array(value("tokenizer.ggml.merges")?, "tokenizer.ggml.merges")?;
        ensure!(
            tokens.len() == 248_320 && types.len() == tokens.len() && merges.len() == 247_587,
            "selected GGUF tokenizer inventory has unexpected lengths"
        );

        let mut vocab = Map::new();
        let mut added_tokens = Vec::new();
        for (id, (token, token_type)) in tokens.iter().zip(&types).enumerate() {
            match *token_type {
                1 => {
                    ensure!(
                        id < 248_044,
                        "normal token appears inside the padded vocabulary tail"
                    );
                    ensure!(
                        vocab.insert(token.clone(), Value::from(id)).is_none(),
                        "duplicate GGUF tokenizer vocabulary token at id {id}"
                    );
                }
                3 | 4 => {
                    ensure!(
                        (248_044..=248_076).contains(&id),
                        "special token id {id} is outside the pinned range"
                    );
                    added_tokens.push(json!({
                        "id": id,
                        "content": token,
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true,
                    }));
                }
                5 => ensure!(
                    id >= 248_077 && token == &format!("[PAD{id}]"),
                    "invalid padded tokenizer entry at id {id}"
                ),
                other => anyhow::bail!("unsupported GGUF tokenizer token type {other} at id {id}"),
            }
        }
        ensure!(
            vocab.len() == 248_044 && added_tokens.len() == 33,
            "tokenizer split mismatch"
        );
        for merge in &merges {
            ensure!(
                merge.split_once(' ').is_some(),
                "GGUF tokenizer merge is malformed"
            );
        }

        let tokenizer_json = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": { "Regex": QWEN35_SPLIT_REGEX },
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": false,
                        "use_regex": false
                    }
                ]
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges
            }
        });
        let tokenizer_bytes = serde_json::to_vec(&tokenizer_json)
            .context("failed to serialize safe GGUF tokenizer data")?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(anyhow::Error::msg)
            .context("failed to construct tokenizer from safe GGUF metadata")?;

        let bos_id = metadata_integer(metadata, "tokenizer.ggml.bos_token_id")?;
        let eos_id = metadata_integer(metadata, "tokenizer.ggml.eos_token_id")?;
        ensure!(
            bos_id == 248_044
                && eos_id == 248_046
                && metadata_integer(metadata, "tokenizer.ggml.padding_token_id")? == 248_055,
            "selected GGUF tokenizer special-token policy changed"
        );
        Ok(Self {
            tokenizer,
            chat_template: value("tokenizer.chat_template")?
                .as_str()
                .context("selected GGUF chat template is not a string")?
                .to_owned(),
            bos_token: tokens[bos_id].clone(),
            eos_token: tokens[eos_id].clone(),
            top_k: metadata_integer(metadata, "general.sampling.top_k")?,
            top_p: value("general.sampling.top_p")?
                .as_f64()
                .context("general.sampling.top_p is not numeric")? as f32,
            temperature: value("general.sampling.temp")?
                .as_f64()
                .context("general.sampling.temp is not numeric")? as f32,
        })
    }
}

fn string_array(value: &MetadataValue, key: &str) -> Result<Vec<String>> {
    value
        .as_array()
        .with_context(|| format!("{key} is not an array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("{key} contains a non-string item"))
        })
        .collect()
}

fn integer_array(value: &MetadataValue, key: &str) -> Result<Vec<usize>> {
    value
        .as_array()
        .with_context(|| format!("{key} is not an array"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .with_context(|| format!("{key} contains a non-integer item"))
        })
        .collect()
}

fn metadata_integer(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<usize> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .with_context(|| format!("selected GGUF metadata {key} is missing or non-integer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 27B GGUF pair"]
    fn real_selected_metadata_builds_qwen35_tokenizer_and_chat_assets() {
        let pair = crate::gguf::PinnedGgufPair::open().expect("open pinned GGUF pair");
        let assets = GgufTextAssets::load(&pair.target).expect("load GGUF text assets");
        let encoded = assets
            .tokenizer
            .encode("<|im_start|>user\nHello<|im_end|>\n", false)
            .expect("encode chat turn");
        assert!(!encoded.get_ids().is_empty());
        assert_eq!(assets.bos_token, "<|endoftext|>");
        assert_eq!(assets.eos_token, "<|im_end|>");
        assert_eq!(assets.top_k, 20);
        assert!((assets.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(assets.temperature, 1.0);
        assert!(assets.chat_template.contains("<|im_start|>"));
    }
}
