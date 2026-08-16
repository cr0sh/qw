use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, ensure};
use mlxcel_core::generate::{CxxGenerator, GenerationStats, LanguageModel, SamplingConfig};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::chat_template::ChatTemplateProcessor;
use crate::qwen3_5::Qwen35Model;
use crate::qwen3_5_mtp::Qwen35MtpGenerator;

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<i32>,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35GenerationMode {
    Automatic,
    Baseline,
    Mtp,
}

pub struct Qwen35Provider {
    model: Qwen35Model,
    tokenizer: Tokenizer,
    chat_template: ChatTemplateProcessor,
    defaults: GenerationDefaults,
    generator: CxxGenerator,
    mtp_generator: Option<Qwen35MtpGenerator>,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(i32),
    Many(Vec<i32>),
}

struct GenerationDefaults {
    stop_token_ids: Vec<i32>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
}

impl Qwen35Provider {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        initialize_runtime()?;
        let model_dir = model_dir.as_ref();
        ensure!(
            model_dir.is_dir(),
            "model directory does not exist or is not a directory: {}",
            model_dir.display()
        );

        let tokenizer_path = model_dir.join("tokenizer.json");
        ensure!(
            tokenizer_path.is_file(),
            "missing tokenizer {}",
            tokenizer_path.display()
        );
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
        let chat_template = ChatTemplateProcessor::from_model_path(model_dir)?;
        let defaults = load_generation_defaults(model_dir)?;
        let model = Qwen35Model::load(model_dir)?;
        let generator = CxxGenerator::new(model.num_layers());
        let mtp_generator = model.has_mtp().then(Qwen35MtpGenerator::new);

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            defaults,
            generator,
            mtp_generator,
        })
    }

    pub fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationOutput> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let use_mtp = self.resolve_generation_mode(Qwen35GenerationMode::Automatic, &sampling)?;
        let token_ids = if use_mtp {
            self.mtp_generator
                .as_mut()
                .expect("MTP mode requires an initialized generator")
                .generate(&self.model, &prompt_ids, request.max_tokens, &sampling)
                .0
        } else {
            self.generator
                .generate(&self.model, &prompt_ids, request.max_tokens, &sampling)
        };
        self.output_from_token_ids(token_ids)
    }

    /// Generate one response and return phase timings from the canonical
    /// prefill/decode loop.
    pub fn generate_with_stats(
        &mut self,
        request: &GenerationRequest,
    ) -> Result<(GenerationOutput, GenerationStats)> {
        self.generate_with_stats_in_mode(request, Qwen35GenerationMode::Automatic)
    }

    #[doc(hidden)]
    pub fn generate_with_stats_in_mode(
        &mut self,
        request: &GenerationRequest,
        mode: Qwen35GenerationMode,
    ) -> Result<(GenerationOutput, GenerationStats)> {
        let (prompt_ids, sampling) = self.prepare_generation(request)?;
        let use_mtp = self.resolve_generation_mode(mode, &sampling)?;
        let (token_ids, stats) = if use_mtp {
            self.mtp_generator
                .as_mut()
                .expect("MTP mode requires an initialized generator")
                .generate(&self.model, &prompt_ids, request.max_tokens, &sampling)
        } else {
            self.generator.generate_with_stats(
                &self.model,
                &prompt_ids,
                request.max_tokens,
                &sampling,
            )
        };
        Ok((self.output_from_token_ids(token_ids)?, stats))
    }

    fn resolve_generation_mode(
        &self,
        mode: Qwen35GenerationMode,
        sampling: &SamplingConfig,
    ) -> Result<bool> {
        match mode {
            Qwen35GenerationMode::Automatic => {
                Ok(self.mtp_generator.is_some() && sampling.temperature == 0.0)
            }
            Qwen35GenerationMode::Baseline => Ok(false),
            Qwen35GenerationMode::Mtp => {
                ensure!(
                    self.mtp_generator.is_some(),
                    "the loaded checkpoint does not contain a bundled Qwen 3.5 MTP head"
                );
                ensure!(
                    sampling.temperature == 0.0,
                    "Qwen 3.5 MTP decoding is available only for greedy requests"
                );
                Ok(true)
            }
        }
    }

    fn prepare_generation(
        &self,
        request: &GenerationRequest,
    ) -> Result<(Vec<i32>, SamplingConfig)> {
        ensure!(!request.prompt.is_empty(), "prompt must not be empty");
        ensure!(
            request.max_tokens > 0,
            "max_tokens must be greater than zero"
        );

        let rendered = self.chat_template.render_user(&request.prompt)?;
        let encoded = self
            .tokenizer
            .encode(rendered, true)
            .map_err(anyhow::Error::msg)
            .context("failed to tokenize rendered prompt")?;
        let prompt_ids: Vec<i32> = encoded
            .get_ids()
            .iter()
            .map(|&token| token as i32)
            .collect();
        ensure!(
            !prompt_ids.is_empty(),
            "rendered prompt tokenized to an empty sequence"
        );

        let sampling = SamplingConfig {
            temperature: request.temperature.unwrap_or(self.defaults.temperature),
            top_k: request.top_k.unwrap_or(self.defaults.top_k),
            top_p: request.top_p.unwrap_or(self.defaults.top_p),
            seed: request.seed,
            stop_token_ids: self.defaults.stop_token_ids.clone(),
            ..SamplingConfig::default()
        };
        Ok((prompt_ids, sampling))
    }

    fn output_from_token_ids(&self, mut token_ids: Vec<i32>) -> Result<GenerationOutput> {
        if token_ids
            .last()
            .is_some_and(|token| self.defaults.stop_token_ids.contains(token))
        {
            token_ids.pop();
        }
        let decoded_ids: Vec<u32> = token_ids.iter().map(|&token| token as u32).collect();
        let text = self
            .tokenizer
            .decode(&decoded_ids, false)
            .map_err(anyhow::Error::msg)
            .context("failed to decode generated tokens")?;

        Ok(GenerationOutput { text, token_ids })
    }
}

fn initialize_runtime() -> Result<()> {
    static INITIALIZED: LazyLock<std::result::Result<(), String>> = LazyLock::new(|| {
        if !mlxcel_core::metal_is_available() {
            return Err("the MLX Metal backend is unavailable on this host".to_string());
        }
        mlxcel_core::set_default_device(true);
        Ok(())
    });
    (*INITIALIZED).clone().map_err(anyhow::Error::msg)
}

fn load_generation_defaults(model_dir: &Path) -> Result<GenerationDefaults> {
    let path: PathBuf = model_dir.join("generation_config.json");
    let config = if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str::<GenerationConfig>(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?
    } else {
        GenerationConfig {
            eos_token_id: None,
            temperature: None,
            top_k: None,
            top_p: None,
        }
    };
    let mut stop_token_ids = match config.eos_token_id {
        Some(TokenIds::One(token)) => vec![token],
        Some(TokenIds::Many(tokens)) => tokens,
        None => vec![248046, 248044],
    };
    stop_token_ids.sort_unstable();
    stop_token_ids.dedup();
    ensure!(
        !stop_token_ids.is_empty(),
        "{} contains an empty eos_token_id list",
        path.display()
    );

    Ok(GenerationDefaults {
        stop_token_ids,
        temperature: config.temperature.unwrap_or(1.0),
        top_k: config.top_k.unwrap_or(20),
        top_p: config.top_p.unwrap_or(0.95),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "qw-provider-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn generation_defaults_match_checkpoint_contract() {
        let fixture = TestDir::new("generation-defaults");
        let fallback = load_generation_defaults(&fixture.0).expect("fallback defaults");
        assert_eq!(fallback.stop_token_ids, vec![248044, 248046]);
        assert_eq!(fallback.temperature, 1.0);
        assert_eq!(fallback.top_k, 20);
        assert_eq!(fallback.top_p, 0.95);

        std::fs::write(
            fixture.0.join("generation_config.json"),
            br#"{"eos_token_id":[9,7],"temperature":0.7,"top_k":11,"top_p":0.8}"#,
        )
        .expect("write generation config");
        let loaded = load_generation_defaults(&fixture.0).expect("checkpoint defaults");
        assert_eq!(loaded.stop_token_ids, vec![7, 9]);
        assert_eq!(loaded.temperature, 0.7);
        assert_eq!(loaded.top_k, 11);
        assert_eq!(loaded.top_p, 0.8);
    }

    #[test]
    fn provider_rejects_absent_chat_template_with_paths() {
        let fixture = TestDir::new("missing-chat-template");
        std::fs::write(
            fixture.0.join("tokenizer.json"),
            br#"{
                "version":"1.0",
                "truncation":null,
                "padding":null,
                "added_tokens":[],
                "normalizer":null,
                "pre_tokenizer":null,
                "post_processor":null,
                "decoder":null,
                "model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}
            }"#,
        )
        .expect("write tokenizer");
        std::fs::write(fixture.0.join("tokenizer_config.json"), b"{}")
            .expect("write tokenizer config");

        let error = match Qwen35Provider::load(&fixture.0) {
            Ok(_) => panic!("provider load must reject an absent chat template"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("missing chat template"), "{error}");
        assert!(error.contains("chat_template.jinja"), "{error}");
        assert!(error.contains("tokenizer_config.json"), "{error}");
    }

    #[test]
    #[ignore = "requires QW_BENCH_MODEL pointing at a real bundled-MTP checkpoint"]
    fn real_model_baseline_and_mtp_greedy_outputs_match() {
        let model_dir = std::env::var_os("QW_BENCH_MODEL")
            .map(PathBuf::from)
            .expect("QW_BENCH_MODEL must point at a real checkpoint");
        let mut provider = Qwen35Provider::load(&model_dir).expect("load real Qwen checkpoint");
        let request = GenerationRequest {
            prompt: "Continue counting upward from one, writing each integer on its own line without stopping."
                .to_string(),
            max_tokens: 32,
            temperature: Some(0.0),
            top_k: Some(1),
            top_p: Some(1.0),
            seed: Some(0),
        };
        let (baseline, _) = provider
            .generate_with_stats_in_mode(&request, Qwen35GenerationMode::Baseline)
            .expect("baseline greedy generation");
        let (mtp, _) = provider
            .generate_with_stats_in_mode(&request, Qwen35GenerationMode::Mtp)
            .expect("MTP greedy generation");
        assert_eq!(baseline.token_ids, mtp.token_ids);
        assert_eq!(baseline.text, mtp.text);
    }
}
