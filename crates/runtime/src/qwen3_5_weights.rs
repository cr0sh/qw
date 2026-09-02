use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use memmap2::{Advice, Mmap, MmapOptions, UncheckedAdvice};
use mlxcel_core::layers::{FusedQKVLinear, Linear, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{
    GgmlAffineEmbedding, GgmlAffineMatrix, GgmlAffineRows, GgmlAffineTranscodeStats,
    GgmlQuantizedEmbedding, GgmlQuantizedMatrix, GgmlQuantizedRows, MlxArray, UniquePtr, dtype,
};

use crate::gguf::{GgufModelPair, GgufShardSet, GgufTensorInfo, MetadataValue};
use crate::qwen3_5::Qwen35Config;

pub(crate) enum Qwen35Linear {
    Legacy(UnifiedLinear),
    Affine(GgmlAffineMatrix),
    AffineRows(GgmlAffineRows),
    Gguf(GgmlQuantizedMatrix),
    GgufRows(GgmlQuantizedRows),
}

impl Qwen35Linear {
    pub(crate) fn legacy(linear: UnifiedLinear) -> Self {
        Self::Legacy(linear)
    }

    pub(crate) fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Legacy(linear) => linear.forward(input),
            Self::Affine(linear) => linear
                .forward(input)
                .expect("validated GGML affine matrix execution must succeed"),
            Self::AffineRows(linear) => linear
                .forward(input)
                .expect("validated GGML affine row projection must succeed"),
            Self::Gguf(linear) => linear
                .forward(input)
                .expect("validated GGML matrix execution must succeed"),
            Self::GgufRows(linear) => linear
                .forward(input)
                .expect("validated GGML row projection must succeed"),
        }
    }

    pub(crate) fn legacy_ref(&self) -> Option<&UnifiedLinear> {
        match self {
            Self::Legacy(linear) => Some(linear),
            Self::Affine(_) | Self::AffineRows(_) | Self::Gguf(_) | Self::GgufRows(_) => None,
        }
    }

    pub(crate) fn select_gguf_rows(
        &self,
        ranges: &[std::ops::Range<usize>],
    ) -> Option<Self> {
        match self {
            Self::Affine(linear) => linear.select_rows(ranges).ok().map(Self::AffineRows),
            Self::Gguf(linear) => linear.select_rows(ranges).ok().map(Self::GgufRows),
            Self::Legacy(_) | Self::AffineRows(_) | Self::GgufRows(_) => None,
        }
    }
}

pub(crate) enum Qwen35Embedding {
    Legacy(UnifiedEmbedding),
    Affine(GgmlAffineEmbedding),
    Gguf(GgmlQuantizedEmbedding),
}

impl Qwen35Embedding {
    pub(crate) fn clone_shared(&self) -> Self {
        match self {
            Self::Legacy(embedding) => Self::Legacy(embedding.clone_shared()),
            Self::Affine(embedding) => Self::Affine(embedding.clone_shared()),
            Self::Gguf(embedding) => Self::Gguf(embedding.clone_shared()),
        }
    }

    pub(crate) fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Legacy(embedding) => embedding.forward(indices),
            Self::Affine(embedding) => {
                let converted = (mlxcel_core::array_dtype(indices) == dtype::INT64)
                    .then(|| mlxcel_core::astype(indices, dtype::INT32));
                embedding
                    .forward(converted.as_deref().unwrap_or(indices))
                    .expect("validated GGML affine embedding execution must succeed")
            }
            Self::Gguf(embedding) => {
                let converted = (mlxcel_core::array_dtype(indices) == dtype::INT64)
                    .then(|| mlxcel_core::astype(indices, dtype::INT32));
                embedding
                    .forward(converted.as_deref().unwrap_or(indices))
                    .expect("validated GGML embedding execution must succeed")
            }
        }
    }

    pub(crate) fn as_linear(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Legacy(embedding) => embedding.as_linear(input),
            Self::Affine(embedding) => embedding
                .as_linear(input)
                .expect("validated GGML affine embedding projection must succeed"),
            Self::Gguf(embedding) => embedding
                .as_linear(input)
                .expect("validated GGML embedding projection must succeed"),
        }
    }
}

pub(crate) enum Qwen35QkvProjection {
    Legacy(FusedQKVLinear),
    Separate {
        query: Qwen35Linear,
        key: Qwen35Linear,
        value: Qwen35Linear,
    },
}

impl Qwen35QkvProjection {
    pub(crate) fn forward(
        &self,
        input: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        match self {
            Self::Legacy(projection) => projection.forward(input),
            Self::Separate { query, key, value } => (
                query.forward(input),
                key.forward(input),
                value.forward(input),
            ),
        }
    }
}

pub(crate) trait Qwen35WeightSource {
    fn linear(
        &self,
        name: &str,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Linear, String>;

    fn tensor(&self, name: &str) -> std::result::Result<UniquePtr<MlxArray>, String>;

    fn embedding(
        &self,
        name: &str,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Embedding, String>;

    fn qkv(
        &self,
        prefix: &str,
        group_size: i32,
        bits: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> std::result::Result<Qwen35QkvProjection, String>;

    fn legacy_weights(&self) -> Option<&WeightMap> {
        None
    }

    fn gguf_ssm_a_is_coefficient(&self) -> bool {
        false
    }
}

impl Qwen35WeightSource for WeightMap {
    fn linear(
        &self,
        name: &str,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Linear, String> {
        UnifiedLinear::from_weights(self, name, group_size, bits).map(Qwen35Linear::Legacy)
    }

    fn tensor(&self, name: &str) -> std::result::Result<UniquePtr<MlxArray>, String> {
        self.get(name)
            .map(|tensor| mlxcel_core::copy(tensor))
            .ok_or_else(|| format!("missing required tensor {name}"))
    }

    fn embedding(
        &self,
        name: &str,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Embedding, String> {
        UnifiedEmbedding::from_weights(self, name, group_size, bits).map(Qwen35Embedding::Legacy)
    }

    fn qkv(
        &self,
        prefix: &str,
        group_size: i32,
        bits: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> std::result::Result<Qwen35QkvProjection, String> {
        FusedQKVLinear::from_weights_separate(
            self,
            prefix,
            group_size,
            bits,
            query_heads,
            kv_heads,
            head_dim,
        )
        .map(Qwen35QkvProjection::Legacy)
    }

    fn legacy_weights(&self) -> Option<&WeightMap> {
        Some(self)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GgufAffineLoadStats {
    pub tensors: usize,
    pub source_bytes: usize,
    pub resident_bytes: usize,
    pub peak_active_bytes: usize,
    pub elapsed: Duration,
}

#[derive(Clone, Copy)]
enum Artifact {
    Target,
    Mtp,
}

pub(crate) struct GgufWeightSource {
    pair: GgufModelPair,
    target_maps: Vec<Mmap>,
    mtp_maps: Vec<Mmap>,
    used: RefCell<BTreeSet<(u8, String)>>,
    affine_stats: RefCell<GgufAffineLoadStats>,
}

impl GgufWeightSource {
    pub(crate) fn open(root: &Path) -> Result<Self> {
        let pair = GgufModelPair::open_selected(root)?;
        let target_maps = map_shards(&pair.target)?;
        let mtp_maps = map_shards(&pair.mtp)?;
        Ok(Self {
            pair,
            target_maps,
            mtp_maps,
            used: RefCell::new(BTreeSet::new()),
            affine_stats: RefCell::new(GgufAffineLoadStats::default()),
        })
    }

    pub(crate) fn affine_stats(&self) -> GgufAffineLoadStats {
        *self.affine_stats.borrow()
    }

    fn record_affine(&self, transcode: GgmlAffineTranscodeStats) {
        let mut stats = self.affine_stats.borrow_mut();
        stats.tensors += 1;
        stats.source_bytes += transcode.source_bytes;
        stats.resident_bytes += transcode.resident_bytes;
        stats.peak_active_bytes = stats.peak_active_bytes.max(transcode.peak_active_bytes);
        stats.elapsed += transcode.elapsed;
    }

    pub(crate) fn target(&self) -> &GgufShardSet {
        &self.pair.target
    }

    pub(crate) fn config(&self) -> Result<Qwen35Config> {
        config_from_metadata(self.pair.target.metadata())
    }

    pub(crate) fn finish(&self) -> Result<()> {
        let used = self.used.borrow();
        let mut missing = Vec::new();
        for (artifact, set) in [
            (Artifact::Target, &self.pair.target),
            (Artifact::Mtp, &self.pair.mtp),
        ] {
            for shard in set.shards() {
                for tensor in shard.tensors() {
                    let intentionally_shared_or_embedded = match artifact {
                        Artifact::Target => tensor.name.starts_with("blk.64."),
                        Artifact::Mtp => matches!(
                            tensor.name.as_str(),
                            "output.weight" | "output_norm.weight" | "token_embd.weight"
                        ),
                    };
                    if intentionally_shared_or_embedded {
                        continue;
                    }
                    let key = (artifact_tag(artifact), tensor.name.clone());
                    if !used.contains(&key) {
                        missing.push(format!(
                            "{}:{}",
                            if matches!(artifact, Artifact::Target) {
                                "target"
                            } else {
                                "mtp"
                            },
                            tensor.name
                        ));
                    }
                }
            }
        }
        ensure!(
            missing.is_empty(),
            "unmapped selected GGUF tensors: {}",
            missing.into_iter().take(16).collect::<Vec<_>>().join(", ")
        );
        Ok(())
    }

    fn lookup(&self, canonical: &str) -> Result<(&GgufTensorInfo, &[u8])> {
        let (artifact, actual) = map_canonical_name(canonical)?;
        let (set, maps) = match artifact {
            Artifact::Target => (&self.pair.target, &self.target_maps),
            Artifact::Mtp => (&self.pair.mtp, &self.mtp_maps),
        };
        let location = set.tensor(&actual).with_context(|| {
            format!("selected GGUF is missing mapped tensor {actual} for {canonical}")
        })?;
        ensure!(
            self.used
                .borrow_mut()
                .insert((artifact_tag(artifact), actual.clone())),
            "selected GGUF tensor {actual} was consumed more than once"
        );
        let tensor = location.tensor;
        let start =
            usize::try_from(tensor.absolute_offset).context("tensor offset exceeds usize")?;
        let len = usize::try_from(tensor.byte_len).context("tensor byte length exceeds usize")?;
        let end = start.checked_add(len).context("tensor slice overflow")?;
        let bytes = maps[location.shard]
            .get(start..end)
            .with_context(|| format!("mapped tensor {actual} exceeds its GGUF file"))?;
        Ok((tensor, bytes))
    }

    fn discard_tensor_pages(&self, canonical: &str) {
        let Ok((artifact, actual)) = map_canonical_name(canonical) else {
            return;
        };
        let (set, maps) = match artifact {
            Artifact::Target => (&self.pair.target, &self.target_maps),
            Artifact::Mtp => (&self.pair.mtp, &self.mtp_maps),
        };
        let Some(location) = set.tensor(&actual) else {
            return;
        };
        let Ok(start) = usize::try_from(location.tensor.absolute_offset) else {
            return;
        };
        let Ok(len) = usize::try_from(location.tensor.byte_len) else {
            return;
        };
        let page_start = start / 4096 * 4096;
        let page_end = start
            .saturating_add(len)
            .div_ceil(4096)
            .saturating_mul(4096)
            .min(maps[location.shard].len());
        // SAFETY: packed/native constructors have been evaluated before this
        // call and no borrowed GGUF slice survives. The mapping is immutable.
        let _ = unsafe {
            maps[location.shard].unchecked_advise_range(
                UncheckedAdvice::DontNeed,
                page_start,
                page_end - page_start,
            )
        };
    }

    fn discard_tensor_byte_range(&self, canonical: &str, relative: std::ops::Range<usize>) {
        let Ok((artifact, actual)) = map_canonical_name(canonical) else {
            return;
        };
        let (set, maps) = match artifact {
            Artifact::Target => (&self.pair.target, &self.target_maps),
            Artifact::Mtp => (&self.pair.mtp, &self.mtp_maps),
        };
        let Some(location) = set.tensor(&actual) else {
            return;
        };
        let Ok(tensor_start) = usize::try_from(location.tensor.absolute_offset) else {
            return;
        };
        let Ok(tensor_len) = usize::try_from(location.tensor.byte_len) else {
            return;
        };
        if relative.start > relative.end || relative.end > tensor_len {
            return;
        }
        let absolute_start = tensor_start.saturating_add(relative.start);
        let absolute_end = tensor_start.saturating_add(relative.end);
        let page_start = absolute_start.div_ceil(4096).saturating_mul(4096);
        let page_end = absolute_end / 4096 * 4096;
        if page_start >= page_end || page_end > maps[location.shard].len() {
            return;
        }
        // SAFETY: the callback fires only after consuming this immutable range.
        // A future access faults discarded pages back in.
        let _ = unsafe {
            maps[location.shard].unchecked_advise_range(
                UncheckedAdvice::DontNeed,
                page_start,
                page_end - page_start,
            )
        };
    }

    fn load_native(&self, canonical: &str) -> Result<UniquePtr<MlxArray>> {
        let (tensor, bytes) = self.lookup(canonical)?;
        ensure!(
            tensor.tensor_type.id() == 0,
            "native tensor {canonical} must use F32, got {}",
            tensor.tensor_type.name()
        );
        let shape = tensor
            .dimensions
            .iter()
            .rev()
            .map(|dimension| i32::try_from(*dimension).context("tensor dimension exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        let array = mlxcel_core::from_bytes(bytes, &shape, dtype::FLOAT32);
        mlxcel_core::eval(array.as_ref().unwrap());
        self.discard_tensor_pages(canonical);
        Ok(array)
    }

    fn load_linear(&self, canonical: &str) -> Result<Qwen35Linear> {
        let (tensor, bytes) = self.lookup(canonical)?;
        ensure!(
            (1..=2).contains(&tensor.dimensions.len()),
            "linear tensor {canonical} has rank {}",
            tensor.dimensions.len()
        );
        let input = usize::try_from(tensor.dimensions[0]).context("linear input exceeds usize")?;
        let output = tensor
            .dimensions
            .get(1)
            .copied()
            .map(usize::try_from)
            .transpose()
            .context("linear output exceeds usize")?
            .unwrap_or(1);
        let result = if tensor.tensor_type.id() == 0 {
            let array =
                mlxcel_core::from_bytes(bytes, &[output as i32, input as i32], dtype::FLOAT32);
            mlxcel_core::eval(array.as_ref().unwrap());
            Qwen35Linear::Legacy(UnifiedLinear::Regular(Linear::new(array, None)))
        } else if GgmlAffineMatrix::is_representable(tensor.tensor_type.id()) {
            let affine = GgmlAffineMatrix::from_ggml_bytes_with_progress(
                bytes,
                tensor.tensor_type.id(),
                input,
                output,
                |range| self.discard_tensor_byte_range(canonical, range),
            )?;
            self.record_affine(affine.transcode_stats());
            Qwen35Linear::Affine(affine)
        } else {
            Qwen35Linear::Gguf(GgmlQuantizedMatrix::from_bytes(
                bytes,
                tensor.tensor_type.id(),
                input,
                output,
            )?)
        };
        self.discard_tensor_pages(canonical);
        Ok(result)
    }

    fn load_embedding(&self, canonical: &str) -> Result<Qwen35Embedding> {
        let (tensor, bytes) = self.lookup(canonical)?;
        ensure!(
            tensor.dimensions.len() == 2,
            "embedding tensor {canonical} must have rank two"
        );
        let embedding_dim = usize::try_from(tensor.dimensions[0])?;
        let vocab_size = usize::try_from(tensor.dimensions[1])?;
        let result = if GgmlAffineMatrix::is_representable(tensor.tensor_type.id()) {
            let affine = GgmlAffineEmbedding::from_ggml_bytes_with_progress(
                bytes,
                tensor.tensor_type.id(),
                embedding_dim,
                vocab_size,
                |range| self.discard_tensor_byte_range(canonical, range),
            )?;
            self.record_affine(affine.transcode_stats());
            Qwen35Embedding::Affine(affine)
        } else {
            Qwen35Embedding::Gguf(GgmlQuantizedEmbedding::from_bytes(
                bytes,
                tensor.tensor_type.id(),
                embedding_dim,
                vocab_size,
            )?)
        };
        self.discard_tensor_pages(canonical);
        Ok(result)
    }
}

impl Qwen35WeightSource for GgufWeightSource {
    fn linear(
        &self,
        name: &str,
        _group_size: i32,
        _bits: i32,
    ) -> std::result::Result<Qwen35Linear, String> {
        self.load_linear(&format!("{name}.weight"))
            .map_err(|error| error.to_string())
    }

    fn tensor(&self, name: &str) -> std::result::Result<UniquePtr<MlxArray>, String> {
        self.load_native(name).map_err(|error| error.to_string())
    }

    fn embedding(
        &self,
        name: &str,
        _group_size: i32,
        _bits: i32,
    ) -> std::result::Result<Qwen35Embedding, String> {
        self.load_embedding(&format!("{name}.weight"))
            .map_err(|error| error.to_string())
    }

    fn qkv(
        &self,
        prefix: &str,
        group_size: i32,
        bits: i32,
        _query_heads: i32,
        _kv_heads: i32,
        _head_dim: i32,
    ) -> std::result::Result<Qwen35QkvProjection, String> {
        Ok(Qwen35QkvProjection::Separate {
            query: self.linear(&format!("{prefix}.q_proj"), group_size, bits)?,
            key: self.linear(&format!("{prefix}.k_proj"), group_size, bits)?,
            value: self.linear(&format!("{prefix}.v_proj"), group_size, bits)?,
        })
    }

    fn gguf_ssm_a_is_coefficient(&self) -> bool {
        true
    }
}

fn map_shards(set: &GgufShardSet) -> Result<Vec<Mmap>> {
    set.shards()
        .iter()
        .map(|shard| {
            let file = File::open(shard.path())
                .with_context(|| format!("failed to open GGUF {}", shard.path().display()))?;
            let map = unsafe { MmapOptions::new().map(&file) }
                .with_context(|| format!("failed to mmap GGUF {}", shard.path().display()))?;
            map.advise(Advice::Sequential).with_context(|| {
                format!("failed to mark GGUF {} sequential", shard.path().display())
            })?;
            Ok(map)
        })
        .collect()
}

fn artifact_tag(artifact: Artifact) -> u8 {
    match artifact {
        Artifact::Target => 0,
        Artifact::Mtp => 1,
    }
}

fn map_canonical_name(canonical: &str) -> Result<(Artifact, String)> {
    let global = match canonical {
        "model.embed_tokens.weight" => Some((Artifact::Target, "token_embd.weight")),
        "lm_head.weight" => Some((Artifact::Target, "output.weight")),
        "model.norm.weight" => Some((Artifact::Target, "output_norm.weight")),
        "mtp.pre_fc_norm_embedding.weight" => Some((Artifact::Mtp, "blk.64.nextn.enorm.weight")),
        "mtp.pre_fc_norm_hidden.weight" => Some((Artifact::Mtp, "blk.64.nextn.hnorm.weight")),
        "mtp.fc.weight" => Some((Artifact::Mtp, "blk.64.nextn.eh_proj.weight")),
        "mtp.norm.weight" => Some((Artifact::Mtp, "blk.64.nextn.shared_head_norm.weight")),
        _ => None,
    };
    if let Some((artifact, actual)) = global {
        return Ok((artifact, actual.to_owned()));
    }

    if let Some(rest) = canonical.strip_prefix("model.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .with_context(|| format!("invalid target layer tensor {canonical}"))?;
        let layer = layer
            .parse::<usize>()
            .context("invalid target layer index")?;
        ensure!(layer < 64, "target layer index is outside 0..64");
        return Ok((
            Artifact::Target,
            format!("blk.{layer}.{}", map_layer_suffix(suffix)?),
        ));
    }
    if let Some(suffix) = canonical.strip_prefix("mtp.layers.0.") {
        return Ok((
            Artifact::Mtp,
            format!("blk.64.{}", map_layer_suffix(suffix)?),
        ));
    }
    anyhow::bail!("no selected GGUF mapping for canonical tensor {canonical}")
}

fn map_layer_suffix(suffix: &str) -> Result<&'static str> {
    Ok(match suffix {
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.dt_bias" => "ssm_dt.bias",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        _ => anyhow::bail!("no selected GGUF layer mapping for {suffix}"),
    })
}

fn config_from_metadata(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
) -> Result<Qwen35Config> {
    let integer = |key: &str| -> Result<usize> {
        metadata
            .get(key)
            .and_then(MetadataValue::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .with_context(|| format!("selected GGUF metadata {key} is missing or invalid"))
    };
    let number = |key: &str| -> Result<f32> {
        metadata
            .get(key)
            .and_then(MetadataValue::as_f64)
            .map(|value| value as f32)
            .with_context(|| format!("selected GGUF metadata {key} is missing or invalid"))
    };
    let blocks = integer("qwen35.block_count")?;
    let nextn = integer("qwen35.nextn_predict_layers")?;
    ensure!(blocks > nextn, "selected GGUF has no target decoder layers");
    let sections = metadata
        .get("qwen35.rope.dimension_sections")
        .and_then(MetadataValue::as_array)
        .context("selected GGUF is missing qwen35.rope.dimension_sections")?;
    let sections = sections
        .iter()
        .take(3)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| i32::try_from(value).ok())
                .context("selected GGUF has invalid RoPE dimension sections")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        sections.len() == 3,
        "selected GGUF must contain three text RoPE sections"
    );
    let head_dim = integer("qwen35.attention.key_length")?;
    let rope_dim = integer("qwen35.rope.dimension_count")?;
    Ok(Qwen35Config {
        model_type: "qwen3_5_text".to_owned(),
        hidden_size: integer("qwen35.embedding_length")?,
        num_hidden_layers: blocks - nextn,
        intermediate_size: integer("qwen35.feed_forward_length")?,
        num_attention_heads: integer("qwen35.attention.head_count")?,
        num_key_value_heads: integer("qwen35.attention.head_count_kv")?,
        head_dim: Some(head_dim),
        linear_num_value_heads: integer("qwen35.ssm.inner_size")?
            / integer("qwen35.ssm.state_size")?,
        linear_num_key_heads: integer("qwen35.ssm.group_count")?,
        linear_key_head_dim: integer("qwen35.ssm.state_size")?,
        linear_value_head_dim: integer("qwen35.ssm.state_size")?,
        linear_conv_kernel_dim: integer("qwen35.ssm.conv_kernel")?,
        rope_parameters: Some(serde_json::json!({
            "rope_theta": number("qwen35.rope.freq_base")?,
            "partial_rotary_factor": rope_dim as f32 / head_dim as f32,
            "mrope_section": sections,
        })),
        full_attention_interval: integer("qwen35.full_attention_interval")?,
        rms_norm_eps: number("qwen35.attention.layer_norm_rms_epsilon")?,
        tie_word_embeddings: false,
        vocab_size: 248_320,
        max_position_embeddings: integer("qwen35.context_length")?,
        quantization: None,
        mtp_num_hidden_layers: Some(nextn),
        mtp_use_dedicated_embeddings: Some(false),
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_mapping_covers_target_and_separate_mtp_head() {
        assert_eq!(
            map_canonical_name("model.layers.63.self_attn.o_proj.weight")
                .unwrap()
                .1,
            "blk.63.attn_output.weight"
        );
        assert_eq!(
            map_canonical_name("model.layers.62.linear_attn.in_proj_qkv.weight")
                .unwrap()
                .1,
            "blk.62.attn_qkv.weight"
        );
        assert_eq!(
            map_canonical_name("mtp.layers.0.mlp.down_proj.weight")
                .unwrap()
                .1,
            "blk.64.ffn_down.weight"
        );
        assert_eq!(
            map_canonical_name("mtp.fc.weight").unwrap().1,
            "blk.64.nextn.eh_proj.weight"
        );
        assert!(map_canonical_name("model.layers.64.mlp.down_proj.weight").is_err());
    }

    #[test]
    fn packed_embedding_accepts_internal_int64_token_ids() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        let mut bytes = vec![0_u8; 68];
        for (row, quant) in bytes.chunks_exact_mut(34).zip([1_u8, 2]) {
            row[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            row[2..].fill(quant);
        }
        let embedding =
            Qwen35Embedding::Gguf(GgmlQuantizedEmbedding::from_bytes(&bytes, 8, 32, 2).unwrap());
        let indices = mlxcel_core::from_slice_i64(&[1], &[1]);
        let output = embedding.forward(&indices);
        mlxcel_core::eval(&output);
        assert_eq!(mlxcel_core::array_shape(&output), [1, 32]);
    }
}
