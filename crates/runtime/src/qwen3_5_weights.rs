use std::cell::RefCell;
use std::fs::File;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use memmap2::{Advice, Mmap, MmapOptions, UncheckedAdvice};
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::layers::{FusedQKVLinear, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{
    GgmlAffineEmbedding, GgmlAffineMatrix, GgmlAffineRows, GgmlAffineTranscodeStats, GgmlQType,
    GgmlQuantizedEmbedding, GgmlQuantizedMatrix, GgmlQuantizedRows, MlxArray, Qwen38Q6DualMatrix,
    Qwen38Q6Shape, Qwen38Q6TranscodeStats, UniquePtr, dtype,
};

use crate::gguf::{GgufFile, GgufTensorInfo, PinnedGgufPair};
use crate::qwen3_5::Qwen35Config;

pub(crate) enum Qwen35Linear {
    #[cfg(any(feature = "specprefill", test))]
    Legacy(UnifiedLinear),
    Affine(GgmlAffineMatrix),
    AffineRows(GgmlAffineRows),
    Q6Dual(Qwen38Q6DualMatrix),
    Gguf(GgmlQuantizedMatrix),
    GgufRows(GgmlQuantizedRows),
}

impl Qwen35Linear {
    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn legacy(linear: UnifiedLinear) -> Self {
        Self::Legacy(linear)
    }

    pub(crate) fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            #[cfg(any(feature = "specprefill", test))]
            Self::Legacy(linear) => linear.forward(input),
            Self::Affine(linear) => linear
                .forward(input)
                .expect("validated GGML affine matrix execution must succeed"),
            Self::AffineRows(linear) => linear
                .forward(input)
                .expect("validated GGML affine row projection must succeed"),
            Self::Q6Dual(linear) => linear
                .forward(input)
                .expect("validated pinned Q6_K dual execution must succeed"),
            Self::Gguf(linear) => linear
                .forward(input)
                .expect("validated GGML matrix execution must succeed"),
            Self::GgufRows(linear) => linear
                .forward(input)
                .expect("validated GGML row projection must succeed"),
        }
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn legacy_ref(&self) -> Option<&UnifiedLinear> {
        match self {
            Self::Legacy(linear) => Some(linear),
            Self::Affine(_)
            | Self::AffineRows(_)
            | Self::Q6Dual(_)
            | Self::Gguf(_)
            | Self::GgufRows(_) => None,
        }
    }

    pub(crate) fn select_gguf_rows(&self, ranges: &[std::ops::Range<usize>]) -> Option<Self> {
        match self {
            Self::Affine(linear) => linear.select_rows(ranges).ok().map(Self::AffineRows),
            Self::Gguf(linear) => linear.select_rows(ranges).ok().map(Self::GgufRows),
            #[cfg(any(feature = "specprefill", test))]
            Self::Legacy(_) => None,
            Self::AffineRows(_) | Self::Q6Dual(_) | Self::GgufRows(_) => None,
        }
    }
}

pub(crate) enum Qwen35Embedding {
    #[cfg(any(feature = "specprefill", test))]
    Legacy(UnifiedEmbedding),
    Affine(GgmlAffineEmbedding),
    Gguf(GgmlQuantizedEmbedding),
}

impl Qwen35Embedding {
    #[cfg(any(feature = "dflash2", test))]
    pub(crate) fn clone_shared(&self) -> Self {
        match self {
            #[cfg(any(feature = "specprefill", test))]
            Self::Legacy(embedding) => Self::Legacy(embedding.clone_shared()),
            Self::Affine(embedding) => Self::Affine(embedding.clone_shared()),
            Self::Gguf(embedding) => Self::Gguf(embedding.clone_shared()),
        }
    }

    pub(crate) fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            #[cfg(any(feature = "specprefill", test))]
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
            #[cfg(any(feature = "specprefill", test))]
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
    #[cfg(any(feature = "specprefill", test))]
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
            #[cfg(any(feature = "specprefill", test))]
            Self::Legacy(projection) => projection.forward(input),
            Self::Separate { query, key, value } => (
                query.forward(input),
                key.forward(input),
                value.forward(input),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelRole {
    Target,
    Mtp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerTensor {
    InputNorm,
    PostAttentionNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    AttentionQueryNorm,
    AttentionKeyNorm,
    MlpGate,
    MlpUp,
    MlpDown,
    LinearQkv,
    LinearGate,
    LinearBeta,
    LinearAlpha,
    LinearConv,
    LinearDtBias,
    LinearA,
    LinearNorm,
    LinearOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TensorSlot {
    TokenEmbedding,
    Output,
    OutputNorm,
    Layer {
        role: ModelRole,
        layer: usize,
        tensor: LayerTensor,
    },
    MtpEmbeddingNorm,
    MtpHiddenNorm,
    MtpProjection,
    MtpHeadNorm,
}

impl TensorSlot {
    fn canonical_name(self) -> String {
        match self {
            Self::TokenEmbedding => "model.embed_tokens.weight".to_owned(),
            Self::Output => "lm_head.weight".to_owned(),
            Self::OutputNorm => "model.norm.weight".to_owned(),
            Self::MtpEmbeddingNorm => "mtp.pre_fc_norm_embedding.weight".to_owned(),
            Self::MtpHiddenNorm => "mtp.pre_fc_norm_hidden.weight".to_owned(),
            Self::MtpProjection => "mtp.fc.weight".to_owned(),
            Self::MtpHeadNorm => "mtp.norm.weight".to_owned(),
            Self::Layer {
                role,
                layer,
                tensor,
            } => {
                let prefix = match role {
                    ModelRole::Target => format!("model.layers.{layer}"),
                    ModelRole::Mtp => format!("mtp.layers.{layer}"),
                };
                let suffix = match tensor {
                    LayerTensor::InputNorm => "input_layernorm.weight",
                    LayerTensor::PostAttentionNorm => "post_attention_layernorm.weight",
                    LayerTensor::AttentionQuery => "self_attn.q_proj.weight",
                    LayerTensor::AttentionKey => "self_attn.k_proj.weight",
                    LayerTensor::AttentionValue => "self_attn.v_proj.weight",
                    LayerTensor::AttentionOutput => "self_attn.o_proj.weight",
                    LayerTensor::AttentionQueryNorm => "self_attn.q_norm.weight",
                    LayerTensor::AttentionKeyNorm => "self_attn.k_norm.weight",
                    LayerTensor::MlpGate => "mlp.gate_proj.weight",
                    LayerTensor::MlpUp => "mlp.up_proj.weight",
                    LayerTensor::MlpDown => "mlp.down_proj.weight",
                    LayerTensor::LinearQkv => "linear_attn.in_proj_qkv.weight",
                    LayerTensor::LinearGate => "linear_attn.in_proj_z.weight",
                    LayerTensor::LinearBeta => "linear_attn.in_proj_b.weight",
                    LayerTensor::LinearAlpha => "linear_attn.in_proj_a.weight",
                    LayerTensor::LinearConv => "linear_attn.conv1d.weight",
                    LayerTensor::LinearDtBias => "linear_attn.dt_bias",
                    LayerTensor::LinearA => "linear_attn.A_log",
                    LayerTensor::LinearNorm => "linear_attn.norm.weight",
                    LayerTensor::LinearOutput => "linear_attn.out_proj.weight",
                };
                format!("{prefix}.{suffix}")
            }
        }
    }
}

pub(crate) trait Qwen35WeightSource {
    fn linear(
        &self,
        slot: TensorSlot,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Linear, String>;

    fn tensor(&self, slot: TensorSlot) -> std::result::Result<UniquePtr<MlxArray>, String>;

    fn embedding(
        &self,
        slot: TensorSlot,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Embedding, String>;

    fn qkv(
        &self,
        role: ModelRole,
        layer: usize,
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

#[cfg(any(feature = "specprefill", test))]
impl Qwen35WeightSource for WeightMap {
    fn linear(
        &self,
        slot: TensorSlot,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Linear, String> {
        let name = slot.canonical_name();
        let prefix = name.strip_suffix(".weight").unwrap_or(&name);
        UnifiedLinear::from_weights(self, prefix, group_size, bits).map(Qwen35Linear::Legacy)
    }

    fn tensor(&self, slot: TensorSlot) -> std::result::Result<UniquePtr<MlxArray>, String> {
        let name = slot.canonical_name();
        self.get(&name)
            .map(|tensor| mlxcel_core::copy(tensor))
            .ok_or_else(|| format!("missing required tensor {name}"))
    }

    fn embedding(
        &self,
        slot: TensorSlot,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Qwen35Embedding, String> {
        let name = slot.canonical_name();
        let prefix = name.strip_suffix(".weight").unwrap_or(&name);
        UnifiedEmbedding::from_weights(self, prefix, group_size, bits).map(Qwen35Embedding::Legacy)
    }

    fn qkv(
        &self,
        role: ModelRole,
        layer: usize,
        group_size: i32,
        bits: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> std::result::Result<Qwen35QkvProjection, String> {
        let prefix = match role {
            ModelRole::Target => format!("model.layers.{layer}.self_attn"),
            ModelRole::Mtp => format!("mtp.layers.{layer}.self_attn"),
        };
        FusedQKVLinear::from_weights_separate(
            self,
            &prefix,
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

fn pinned_affine_qtype(type_id: u32) -> bool {
    matches!(type_id, 8 | 11 | 12 | 13 | 20 | 21 | 23)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PinnedSlot {
    Target(usize),
    Mtp(usize),
}

fn pinned_q6_shape(slot: PinnedSlot, tensor: &GgufTensorInfo) -> Result<Option<Qwen38Q6Shape>> {
    if tensor.tensor_type.id() != 14 {
        return Ok(None);
    }
    if slot == PinnedSlot::Target(0) {
        ensure!(
            tensor.dimensions == [5120, 248_320],
            "pinned direct Q6 output shape changed"
        );
        return Ok(None);
    }
    let shape = match tensor.dimensions.as_slice() {
        [5120, 1024] => Qwen38Q6Shape::K5120N1024,
        [5120, 6144] => Qwen38Q6Shape::K5120N6144,
        [5120, 10240] => Qwen38Q6Shape::K5120N10240,
        [5120, 12288] => Qwen38Q6Shape::K5120N12288,
        [5120, 17408] => Qwen38Q6Shape::K5120N17408,
        [6144, 5120] => Qwen38Q6Shape::K6144N5120,
        [10240, 5120] => Qwen38Q6Shape::K10240N5120,
        [17408, 5120] => Qwen38Q6Shape::K17408N5120,
        dimensions => anyhow::bail!("unexpected pinned Q6_K dimensions {dimensions:?}"),
    };
    Ok(Some(shape))
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GgufAffineLoadStats {
    pub tensors: usize,
    pub source_bytes: usize,
    pub resident_bytes: usize,
    pub peak_active_bytes: usize,
    pub elapsed: Duration,
    pub q6_tensors: usize,
    pub q6_source_bytes: usize,
    pub q6_dense_bytes: usize,
    pub q6_peak_active_bytes: usize,
    pub q6_elapsed: Duration,
}

pub(crate) struct GgufWeightSource {
    pair: PinnedGgufPair,
    target_map: Mmap,
    mtp_map: Mmap,
    target_used: RefCell<[bool; 866]>,
    mtp_used: RefCell<[bool; 18]>,
    affine_stats: RefCell<GgufAffineLoadStats>,
}

impl GgufWeightSource {
    pub(crate) fn open() -> Result<Self> {
        let pair = PinnedGgufPair::open()?;
        // PinnedGgufPair verifies both payload hashes and exact plans first.
        let target_map = map_file(&pair.target)?;
        let mtp_map = map_file(&pair.mtp)?;
        Ok(Self {
            pair,
            target_map,
            mtp_map,
            target_used: RefCell::new([false; 866]),
            mtp_used: RefCell::new([false; 18]),
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

    fn record_q6(&self, transcode: Qwen38Q6TranscodeStats) {
        let mut stats = self.affine_stats.borrow_mut();
        stats.q6_tensors += 1;
        stats.q6_source_bytes += transcode.source_bytes;
        stats.q6_dense_bytes += transcode.dense_bytes;
        stats.q6_peak_active_bytes = stats.q6_peak_active_bytes.max(transcode.peak_active_bytes);
        stats.q6_elapsed += transcode.elapsed;
    }

    pub(crate) fn target(&self) -> &GgufFile {
        &self.pair.target
    }

    pub(crate) fn config(&self) -> Result<Qwen35Config> {
        Ok(Qwen35Config::pinned())
    }

    pub(crate) fn finish(&self) -> Result<()> {
        let target = self.target_used.borrow();
        let mtp = self.mtp_used.borrow();
        let mut missing = Vec::new();
        for slot in 0..target.len() {
            if !target[slot] && !crate::qwen38_plan::TARGET_NONRESIDENT_SLOTS.contains(&slot) {
                missing.push(format!("target:{slot}"));
            }
        }
        for slot in 0..mtp.len() {
            if !mtp[slot] && !crate::qwen38_plan::MTP_NONRESIDENT_SLOTS.contains(&slot) {
                missing.push(format!("mtp:{slot}"));
            }
        }
        ensure!(
            missing.is_empty(),
            "unmapped pinned GGUF plan slots: {}",
            missing.into_iter().take(16).collect::<Vec<_>>().join(", ")
        );
        Ok(())
    }

    fn slot_parts(&self, slot: PinnedSlot) -> (&GgufTensorInfo, &Mmap) {
        match slot {
            PinnedSlot::Target(index) => (&self.pair.target.tensors()[index], &self.target_map),
            PinnedSlot::Mtp(index) => (&self.pair.mtp.tensors()[index], &self.mtp_map),
        }
    }

    fn lookup(&self, slot: PinnedSlot) -> Result<(&GgufTensorInfo, &[u8])> {
        let first_use = match slot {
            PinnedSlot::Target(index) => {
                let mut used = self.target_used.borrow_mut();
                let first = !used[index];
                used[index] = true;
                first
            }
            PinnedSlot::Mtp(index) => {
                let mut used = self.mtp_used.borrow_mut();
                let first = !used[index];
                used[index] = true;
                first
            }
        };
        ensure!(
            first_use,
            "pinned GGUF plan slot {slot:?} was consumed more than once"
        );
        let (tensor, map) = self.slot_parts(slot);
        let start =
            usize::try_from(tensor.absolute_offset).context("tensor offset exceeds usize")?;
        let len = usize::try_from(tensor.byte_len).context("tensor byte length exceeds usize")?;
        let end = start.checked_add(len).context("tensor slice overflow")?;
        let bytes = map
            .get(start..end)
            .with_context(|| format!("pinned tensor slot {slot:?} exceeds its GGUF file"))?;
        Ok((tensor, bytes))
    }

    fn discard_tensor_pages(&self, slot: PinnedSlot) {
        let (tensor, map) = self.slot_parts(slot);
        let Ok(start) = usize::try_from(tensor.absolute_offset) else {
            return;
        };
        let Ok(len) = usize::try_from(tensor.byte_len) else {
            return;
        };
        let page_start = start / 4096 * 4096;
        let page_end = start
            .saturating_add(len)
            .div_ceil(4096)
            .saturating_mul(4096)
            .min(map.len());
        // SAFETY: constructors have evaluated before this call and no borrowed
        // GGUF slice survives. The mapping is immutable.
        let _ = unsafe {
            map.unchecked_advise_range(UncheckedAdvice::DontNeed, page_start, page_end - page_start)
        };
    }

    fn discard_tensor_byte_range(&self, slot: PinnedSlot, relative: std::ops::Range<usize>) {
        let (tensor, map) = self.slot_parts(slot);
        let Ok(tensor_start) = usize::try_from(tensor.absolute_offset) else {
            return;
        };
        let Ok(tensor_len) = usize::try_from(tensor.byte_len) else {
            return;
        };
        if relative.start > relative.end || relative.end > tensor_len {
            return;
        }
        let absolute_start = tensor_start.saturating_add(relative.start);
        let absolute_end = tensor_start.saturating_add(relative.end);
        let page_start = absolute_start.div_ceil(4096).saturating_mul(4096);
        let page_end = absolute_end / 4096 * 4096;
        if page_start >= page_end || page_end > map.len() {
            return;
        }
        // SAFETY: the callback fires only after consuming this immutable range.
        let _ = unsafe {
            map.unchecked_advise_range(UncheckedAdvice::DontNeed, page_start, page_end - page_start)
        };
    }

    fn load_native(&self, slot: PinnedSlot, label: &str) -> Result<UniquePtr<MlxArray>> {
        let (tensor, bytes) = self.lookup(slot)?;
        ensure!(
            tensor.tensor_type.id() == 0,
            "native tensor {label} must use pinned F32"
        );
        let shape = tensor
            .dimensions
            .iter()
            .rev()
            .map(|dimension| i32::try_from(*dimension).context("tensor dimension exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        let array = mlxcel_core::from_bytes(bytes, &shape, dtype::FLOAT32);
        mlxcel_core::eval(array.as_ref().unwrap());
        self.discard_tensor_pages(slot);
        Ok(array)
    }

    fn load_linear(&self, slot: PinnedSlot, label: &str) -> Result<Qwen35Linear> {
        let (tensor, bytes) = self.lookup(slot)?;
        ensure!(
            tensor.dimensions.len() == 2 && tensor.tensor_type.id() != 0,
            "linear tensor {label} escaped the exact packed plan"
        );
        let input = usize::try_from(tensor.dimensions[0]).context("linear input exceeds usize")?;
        let output =
            usize::try_from(tensor.dimensions[1]).context("linear output exceeds usize")?;
        let q6_shape = pinned_q6_shape(slot, tensor)?;
        let qtype = GgmlQType::try_from(tensor.tensor_type.id())
            .context("pinned GGML qtype escaped plan validation")?;
        let result = if let Some(shape) = q6_shape {
            let dual =
                Qwen38Q6DualMatrix::from_pinned_bytes_with_progress(bytes, shape, |range| {
                    self.discard_tensor_byte_range(slot, range)
                })?;
            self.record_q6(dual.transcode_stats());
            Qwen35Linear::Q6Dual(dual)
        } else if pinned_affine_qtype(tensor.tensor_type.id()) {
            let affine = GgmlAffineMatrix::from_ggml_bytes_with_progress(
                bytes,
                qtype,
                input,
                output,
                |range| self.discard_tensor_byte_range(slot, range),
            )?;
            self.record_affine(affine.transcode_stats());
            Qwen35Linear::Affine(affine)
        } else {
            ensure!(
                slot == PinnedSlot::Target(0) && tensor.tensor_type.id() == 14,
                "linear tensor {label} has no pinned resident representation"
            );
            Qwen35Linear::Gguf(GgmlQuantizedMatrix::from_bytes(
                bytes, qtype, input, output,
            )?)
        };
        self.discard_tensor_pages(slot);
        Ok(result)
    }

    fn load_embedding(&self, slot: PinnedSlot, label: &str) -> Result<Qwen35Embedding> {
        let (tensor, bytes) = self.lookup(slot)?;
        ensure!(
            tensor.dimensions.len() == 2,
            "embedding tensor {label} must have rank two"
        );
        let embedding_dim = usize::try_from(tensor.dimensions[0])?;
        let vocab_size = usize::try_from(tensor.dimensions[1])?;
        let qtype = GgmlQType::try_from(tensor.tensor_type.id())
            .context("pinned embedding qtype escaped plan validation")?;
        let result = if pinned_affine_qtype(tensor.tensor_type.id()) {
            let affine = GgmlAffineEmbedding::from_ggml_bytes_with_progress(
                bytes,
                qtype,
                embedding_dim,
                vocab_size,
                |range| self.discard_tensor_byte_range(slot, range),
            )?;
            self.record_affine(affine.transcode_stats());
            Qwen35Embedding::Affine(affine)
        } else {
            Qwen35Embedding::Gguf(GgmlQuantizedEmbedding::from_bytes(
                bytes,
                qtype,
                embedding_dim,
                vocab_size,
            )?)
        };
        self.discard_tensor_pages(slot);
        Ok(result)
    }
}

impl Qwen35WeightSource for GgufWeightSource {
    fn linear(
        &self,
        slot: TensorSlot,
        _group_size: i32,
        _bits: i32,
    ) -> std::result::Result<Qwen35Linear, String> {
        let label = slot.canonical_name();
        self.load_linear(pinned_slot(slot)?, &label)
            .map_err(|error| error.to_string())
    }

    fn tensor(&self, slot: TensorSlot) -> std::result::Result<UniquePtr<MlxArray>, String> {
        let label = slot.canonical_name();
        self.load_native(pinned_slot(slot)?, &label)
            .map_err(|error| error.to_string())
    }

    fn embedding(
        &self,
        slot: TensorSlot,
        _group_size: i32,
        _bits: i32,
    ) -> std::result::Result<Qwen35Embedding, String> {
        let label = slot.canonical_name();
        self.load_embedding(pinned_slot(slot)?, &label)
            .map_err(|error| error.to_string())
    }

    fn qkv(
        &self,
        role: ModelRole,
        layer: usize,
        group_size: i32,
        bits: i32,
        _query_heads: i32,
        _kv_heads: i32,
        _head_dim: i32,
    ) -> std::result::Result<Qwen35QkvProjection, String> {
        let slot = |tensor| TensorSlot::Layer {
            role,
            layer,
            tensor,
        };
        Ok(Qwen35QkvProjection::Separate {
            query: self.linear(slot(LayerTensor::AttentionQuery), group_size, bits)?,
            key: self.linear(slot(LayerTensor::AttentionKey), group_size, bits)?,
            value: self.linear(slot(LayerTensor::AttentionValue), group_size, bits)?,
        })
    }

    fn gguf_ssm_a_is_coefficient(&self) -> bool {
        true
    }
}

fn map_file(file: &GgufFile) -> Result<Mmap> {
    let handle = File::open(file.path())
        .with_context(|| format!("failed to open GGUF {}", file.path().display()))?;
    let map = unsafe { MmapOptions::new().map(&handle) }
        .with_context(|| format!("failed to mmap GGUF {}", file.path().display()))?;
    map.advise(Advice::Sequential)
        .with_context(|| format!("failed to mark GGUF {} sequential", file.path().display()))?;
    Ok(map)
}

fn pinned_slot(slot: TensorSlot) -> std::result::Result<PinnedSlot, String> {
    match slot {
        TensorSlot::TokenEmbedding => Ok(PinnedSlot::Target(2)),
        TensorSlot::Output => Ok(PinnedSlot::Target(0)),
        TensorSlot::OutputNorm => Ok(PinnedSlot::Target(1)),
        TensorSlot::MtpEmbeddingNorm => Ok(PinnedSlot::Mtp(14)),
        TensorSlot::MtpHiddenNorm => Ok(PinnedSlot::Mtp(15)),
        TensorSlot::MtpProjection => Ok(PinnedSlot::Mtp(13)),
        TensorSlot::MtpHeadNorm => Ok(PinnedSlot::Mtp(16)),
        TensorSlot::Layer {
            role: ModelRole::Mtp,
            layer: 0,
            tensor: LayerTensor::PostAttentionNorm,
        } => Ok(PinnedSlot::Mtp(17)),
        TensorSlot::Layer {
            role,
            layer,
            tensor,
        } => {
            let (base, full) = match role {
                ModelRole::Target if layer < 64 => (
                    3 + layer * 14 - (layer / 4) * 3,
                    (layer + 1).is_multiple_of(4),
                ),
                ModelRole::Mtp if layer == 0 => (3, true),
                _ => {
                    return Err(format!(
                        "layer slot is outside the pinned topology: {slot:?}"
                    ));
                }
            };
            let offset = layer_slot_offset(full, tensor)?;
            Ok(match role {
                ModelRole::Target => PinnedSlot::Target(base + offset),
                ModelRole::Mtp => PinnedSlot::Mtp(base + offset),
            })
        }
    }
}

fn layer_slot_offset(full: bool, tensor: LayerTensor) -> std::result::Result<usize, String> {
    let offset = if full {
        match tensor {
            LayerTensor::AttentionKey => 0,
            LayerTensor::AttentionKeyNorm => 1,
            LayerTensor::InputNorm => 2,
            LayerTensor::AttentionOutput => 3,
            LayerTensor::AttentionQuery => 4,
            LayerTensor::AttentionQueryNorm => 5,
            LayerTensor::AttentionValue => 6,
            LayerTensor::MlpDown => 7,
            LayerTensor::MlpGate => 8,
            LayerTensor::MlpUp => 9,
            LayerTensor::PostAttentionNorm => 10,
            _ => return Err(format!("{tensor:?} is not a full-attention tensor")),
        }
    } else {
        match tensor {
            LayerTensor::LinearGate => 0,
            LayerTensor::InputNorm => 1,
            LayerTensor::LinearQkv => 2,
            LayerTensor::MlpDown => 3,
            LayerTensor::MlpGate => 4,
            LayerTensor::MlpUp => 5,
            LayerTensor::PostAttentionNorm => 6,
            LayerTensor::LinearA => 7,
            LayerTensor::LinearAlpha => 8,
            LayerTensor::LinearBeta => 9,
            LayerTensor::LinearConv => 10,
            LayerTensor::LinearDtBias => 11,
            LayerTensor::LinearNorm => 12,
            LayerTensor::LinearOutput => 13,
            _ => return Err(format!("{tensor:?} is not a linear-attention tensor")),
        }
    };
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_typed_mapping_covers_target_and_mtp_topology() {
        assert_eq!(
            pinned_slot(TensorSlot::Layer {
                role: ModelRole::Target,
                layer: 63,
                tensor: LayerTensor::AttentionOutput,
            }),
            Ok(PinnedSlot::Target(843))
        );
        assert_eq!(
            pinned_slot(TensorSlot::Layer {
                role: ModelRole::Target,
                layer: 62,
                tensor: LayerTensor::LinearQkv,
            }),
            Ok(PinnedSlot::Target(828))
        );
        assert_eq!(
            pinned_slot(TensorSlot::Layer {
                role: ModelRole::Mtp,
                layer: 0,
                tensor: LayerTensor::MlpDown,
            }),
            Ok(PinnedSlot::Mtp(10))
        );
        assert_eq!(
            pinned_slot(TensorSlot::Layer {
                role: ModelRole::Mtp,
                layer: 0,
                tensor: LayerTensor::PostAttentionNorm,
            }),
            Ok(PinnedSlot::Mtp(17))
        );
        assert_eq!(
            pinned_slot(TensorSlot::MtpProjection),
            Ok(PinnedSlot::Mtp(13))
        );
        assert!(
            pinned_slot(TensorSlot::Layer {
                role: ModelRole::Target,
                layer: 64,
                tensor: LayerTensor::MlpDown,
            })
            .is_err()
        );
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
        let embedding = Qwen35Embedding::Gguf(
            GgmlQuantizedEmbedding::from_bytes(&bytes, GgmlQType::Q8_0, 32, 2).unwrap(),
        );
        let indices = mlxcel_core::from_slice_i64(&[1], &[1]);
        let output = embedding.forward(&indices);
        mlxcel_core::eval(&output);
        assert_eq!(mlxcel_core::array_shape(&output), [1, 32]);
    }
}
