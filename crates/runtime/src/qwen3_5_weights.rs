use std::cell::RefCell;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use memmap2::{Advice, Mmap, MmapOptions, UncheckedAdvice};
#[cfg(any(feature = "specprefill", test))]
use mlxcel_core::layers::{FusedQKVLinear, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{
    GgmlAffineEmbedding, GgmlAffineMatrix, GgmlAffineRows, GgmlAffineTranscodeStats, GgmlQType,
    GgmlQuantizedEmbedding, GgmlQuantizedMatrix, GgmlQuantizedRows, MlxArray, Qwen38MixedQkvBundle,
    Qwen38Q6DualMatrix, Qwen38Q6HeadArgmax, Qwen38Q6Shape, Qwen38Q6TranscodeStats, Qwen38QkvMatrix,
    UniquePtr, dtype,
};

use crate::gguf::{GgufFile, GgufTensorInfo, PinnedGgufPair};
use crate::qwen3_5::Qwen35Config;

pub(crate) const QWEN38_MLP_CARRIER_ENV_VAR: &str = "MLXCEL_QWEN38_MLP_CARRIER";

fn parse_qwen38_mlp_carrier_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

fn qwen38_mlp_carrier_enabled_from_env() -> bool {
    let value = std::env::var(QWEN38_MLP_CARRIER_ENV_VAR).ok();
    parse_qwen38_mlp_carrier_enabled(value.as_deref())
}

pub(crate) enum Qwen35Linear {
    #[cfg(any(feature = "specprefill", test))]
    Legacy(UnifiedLinear),
    Affine(GgmlAffineMatrix),
    PinnedM234Affine(GgmlAffineMatrix),
    PinnedM2Affine(GgmlAffineMatrix),
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
            Self::PinnedM234Affine(linear) => linear.forward_qwen38_m234(input)
                .expect("validated pinned Qwen3.8 M2/M3/M4 affine execution must succeed"),
            Self::PinnedM2Affine(linear) => linear
                .forward_qwen38_m2(input)
                .expect("validated pinned Qwen3.8 M2 affine execution must succeed"),
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

    pub(crate) fn supports_high_m_sigmoid_gate_f16(&self) -> bool {
        matches!(self, Self::Affine(_) | Self::PinnedM234Affine(_))
    }

    pub(crate) fn forward_high_m_sigmoid_gate_f16(
        &self,
        gate: &MlxArray,
        value: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        match self {
            Self::Affine(linear) | Self::PinnedM234Affine(linear) => Some(
                linear
                    .forward_sigmoid_gated_f16(gate, value)
                    .expect("validated high-M FP16 gated affine execution must succeed"),
            ),
            #[cfg(any(feature = "specprefill", test))]
            Self::Legacy(_) => None,
            Self::PinnedM2Affine(_)
            | Self::AffineRows(_)
            | Self::Q6Dual(_)
            | Self::Gguf(_)
            | Self::GgufRows(_) => None,
        }
    }

    pub(crate) fn is_affine(&self) -> bool {
        matches!(self, Self::Affine(_) | Self::PinnedM234Affine(_))
    }

    pub(crate) fn supports_qwen38_mlp_carrier_with(&self, other: &Self) -> bool {
        let (Self::Affine(left) | Self::PinnedM234Affine(left)) = self else {
            return false;
        };
        let (Self::Affine(right) | Self::PinnedM234Affine(right)) = other else {
            return false;
        };
        left.supports_qwen38_mlp_carrier_with(right)
    }

    pub(crate) fn needs_m4_exact_split(&self) -> bool {
        matches!(self, Self::Q6Dual(_) | Self::Gguf(_))
    }

    pub(crate) fn supports_qwen38_q6_head_argmax(&self) -> bool {
        matches!(self, Self::Gguf(linear) if linear.is_qwen38_q6_head())
    }

    pub(crate) fn qwen38_q6_head_argmax(&self, input: &MlxArray) -> Option<Qwen38Q6HeadArgmax> {
        match self {
            Self::Gguf(linear) if linear.is_qwen38_q6_head() => {
                linear.compact_argmax_m34(input).ok()
            }
            _ => None,
        }
    }


    pub(crate) fn is_m2_affine(&self) -> bool {
        matches!(self, Self::PinnedM2Affine(_))
    }

    pub(crate) fn into_affine(self) -> Option<(GgmlAffineMatrix, bool)> {
        match self {
            Self::Affine(matrix) => Some((matrix, false)),
            Self::PinnedM234Affine(matrix) => Some((matrix, true)),
            _ => None,
        }
    }

    pub(crate) fn into_m2_affine(self) -> Option<GgmlAffineMatrix> {
        match self {
            Self::PinnedM2Affine(matrix) => Some(matrix),
            _ => None,
        }
    }

    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn legacy_ref(&self) -> Option<&UnifiedLinear> {
        match self {
            Self::Legacy(linear) => Some(linear),
            Self::Affine(_)
            | Self::PinnedM234Affine(_)
            | Self::PinnedM2Affine(_)
            | Self::AffineRows(_)
            | Self::Q6Dual(_)
            | Self::Gguf(_)
            | Self::GgufRows(_) => None,
        }
    }

    pub(crate) fn select_gguf_rows(&self, ranges: &[std::ops::Range<usize>]) -> Option<Self> {
        match self {
            Self::Affine(linear) => linear.select_rows(ranges).ok().map(Self::AffineRows),
            Self::PinnedM234Affine(linear) => linear.select_rows(ranges).ok().map(Self::AffineRows),
            Self::PinnedM2Affine(linear) => linear.select_rows(ranges).ok().map(Self::AffineRows),
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
    Bundled(Qwen38MixedQkvBundle),
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
            Self::Bundled(projection) => {
                let output = projection
                    .forward(input)
                    .expect("validated pinned full-attention QKV bundle must succeed");
                (output.query, output.key, output.value)
            }
        }
    }

    pub(crate) fn supports_high_m_f16(&self) -> bool {
        matches!(self, Self::Bundled(_))
    }

    pub(crate) fn forward_high_m_f16(
        &self,
        input: &MlxArray,
    ) -> Option<(
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    )> {
        let Self::Bundled(projection) = self else {
            return None;
        };
        let output = projection
            .forward_f16(input)
            .expect("validated pinned high-M FP16 QKV execution must succeed");
        Some((output.query, output.key, output.value))
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

    fn qwen38_fusion_enabled(&self) -> bool {
        false
    }

    fn qwen38_mlp_carrier_enabled(&self) -> bool {
        false
    }

    fn is_pinned_qwen38_gguf(&self) -> bool {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PinnedM234AffineSignature {
    target_slot: usize,
    layer: usize,
    tensor: LayerTensor,
    qtype: u32,
    dimensions: [u64; 2],
}

fn pinned_m234_affine_signature(slot: TensorSlot) -> Option<PinnedM234AffineSignature> {
    let TensorSlot::Layer {
        role: ModelRole::Target,
        layer,
        tensor,
    } = slot
    else {
        return None;
    };
    if layer >= 64 {
        return None;
    }
    let full_attention = (layer + 1).is_multiple_of(4);
    let dimensions = match (full_attention, tensor) {
        (_, LayerTensor::MlpGate | LayerTensor::MlpUp) => [5120, 17_408],
        (_, LayerTensor::MlpDown) => [17_408, 5120],
        (false, LayerTensor::LinearQkv) => [5120, 10_240],
        (false, LayerTensor::LinearGate) => [5120, 6144],
        (false, LayerTensor::LinearAlpha | LayerTensor::LinearBeta) => [5120, 48],
        (false, LayerTensor::LinearOutput) => [6144, 5120],
        (true, LayerTensor::AttentionQuery) => [5120, 12_288],
        (true, LayerTensor::AttentionOutput) => [6144, 5120],
        _ => return None,
    };
    let PinnedSlot::Target(target_slot) = pinned_slot(slot).ok()? else {
        return None;
    };
    let descriptor = &crate::qwen38_plan::TARGET_TENSOR_PLAN[target_slot];
    if descriptor.dimensions != dimensions || !pinned_affine_qtype(descriptor.qtype) {
        return None;
    }
    Some(PinnedM234AffineSignature {
        target_slot,
        layer,
        tensor,
        qtype: descriptor.qtype,
        dimensions,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PinnedQkvBundleSignature {
    key_slot: usize,
    query_slot: usize,
    value_slot: usize,
    query_qtype: u32,
    key_qtype: u32,
    value_qtype: u32,
}

const PINNED_QKV_QTYPES: [(u32, u32, u32); 16] = [
    (13, 12, 13),
    (12, 13, 13),
    (13, 14, 8),
    (12, 14, 8),
    (12, 14, 14),
    (12, 13, 8),
    (23, 14, 8),
    (13, 8, 8),
    (13, 14, 8),
    (12, 14, 14),
    (13, 14, 14),
    (23, 14, 13),
    (13, 14, 8),
    (12, 14, 8),
    (13, 14, 8),
    (13, 14, 8),
];

fn pinned_qkv_bundle_signature(role: ModelRole, layer: usize) -> Option<PinnedQkvBundleSignature> {
    if role != ModelRole::Target || layer >= 64 || !(layer + 1).is_multiple_of(4) {
        return None;
    }
    let index = layer / 4;
    let (query_qtype, key_qtype, value_qtype) = PINNED_QKV_QTYPES[index];
    let base = 3 + layer * 14 - (layer / 4) * 3;
    let signature = PinnedQkvBundleSignature {
        key_slot: base,
        query_slot: base + 4,
        value_slot: base + 6,
        query_qtype,
        key_qtype,
        value_qtype,
    };
    let key = crate::qwen38_plan::TARGET_TENSOR_PLAN.get(signature.key_slot)?;
    let query = crate::qwen38_plan::TARGET_TENSOR_PLAN.get(signature.query_slot)?;
    let value = crate::qwen38_plan::TARGET_TENSOR_PLAN.get(signature.value_slot)?;
    if key.qtype != key_qtype
        || key.dimensions != [5120, 1024]
        || query.qtype != query_qtype
        || query.dimensions != [5120, 12_288]
        || value.qtype != value_qtype
        || value.dimensions != [5120, 1024]
    {
        return None;
    }
    Some(signature)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PinnedM2AffineSignature {
    mtp_slot: usize,
    qtype: u32,
    dimensions: [u64; 2],
}

fn pinned_m2_affine_signature(slot: TensorSlot) -> Option<PinnedM2AffineSignature> {
    let (mtp_slot, dimensions) = match slot {
        TensorSlot::Layer {
            role: ModelRole::Mtp,
            layer: 0,
            tensor: LayerTensor::MlpDown,
        } => (10, [17_408, 5120]),
        TensorSlot::Layer {
            role: ModelRole::Mtp,
            layer: 0,
            tensor: LayerTensor::MlpGate,
        } => (11, [5120, 17_408]),
        TensorSlot::Layer {
            role: ModelRole::Mtp,
            layer: 0,
            tensor: LayerTensor::MlpUp,
        } => (12, [5120, 17_408]),
        TensorSlot::MtpProjection => (13, [10_240, 5120]),
        _ => return None,
    };
    Some(PinnedM2AffineSignature {
        mtp_slot,
        qtype: 12,
        dimensions,
    })
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
    enable_m234: bool,
    enable_fusion: bool,
    enable_mlp_carrier: bool,
    mixed_q5_sidecars: bool,
}

impl GgufWeightSource {
    pub(crate) fn open() -> Result<Self> {
        Self::open_with_features(true, true, true, qwen38_mlp_carrier_enabled_from_env())
    }


    #[cfg(test)]
    pub(crate) fn open_without_fusion() -> Result<Self> {
        Self::open_with_features(true, false, true, false)
    }

    #[cfg(test)]
    pub(crate) fn open_without_mlp_carrier() -> Result<Self> {
        Self::open_with_features(true, true, true, false)
    }

    #[cfg(test)]
    pub(crate) fn open_without_mixed_q5_sidecars() -> Result<Self> {
        Self::open_with_features(true, true, false, qwen38_mlp_carrier_enabled_from_env())
    }

    fn open_with_features(
        enable_m234: bool,
        enable_fusion: bool,
        mixed_q5_sidecars: bool,
        enable_mlp_carrier: bool,
    ) -> Result<Self> {
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
            enable_m234,
            enable_fusion,
            mixed_q5_sidecars,
            enable_mlp_carrier,
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

    fn load_linear(
        &self,
        typed_slot: TensorSlot,
        slot: PinnedSlot,
        label: &str,
    ) -> Result<Qwen35Linear> {
        let m234_signature = pinned_m234_affine_signature(typed_slot);
        let m2_signature = pinned_m2_affine_signature(typed_slot);
        if let Some(signature) = m234_signature {
            ensure!(
                slot == PinnedSlot::Target(signature.target_slot),
                "typed pinned M2/M3/M4 slot changed"
            );
        }
        if let Some(signature) = m2_signature {
            ensure!(
                slot == PinnedSlot::Mtp(signature.mtp_slot),
                "typed pinned M2 slot changed"
            );
        }
        let (tensor, bytes) = self.lookup(slot)?;
        ensure!(
            tensor.dimensions.len() == 2 && tensor.tensor_type.id() != 0,
            "linear tensor {label} escaped the exact packed plan"
        );
        let input = usize::try_from(tensor.dimensions[0]).context("linear input exceeds usize")?;
        let output =
            usize::try_from(tensor.dimensions[1]).context("linear output exceeds usize")?;
        if let Some(signature) = m234_signature {
            ensure!(
                tensor.tensor_type.id() == signature.qtype
                    && tensor.dimensions == signature.dimensions,
                "typed pinned M2/M3/M4 descriptor changed"
            );
        }
        if let Some(signature) = m2_signature {
            ensure!(
                tensor.tensor_type.id() == signature.qtype
                    && tensor.dimensions == signature.dimensions,
                "typed pinned M2 descriptor changed"
            );
        }
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
            let f16_sidecars = self.mixed_q5_sidecars
                && m234_signature
                    .is_some_and(|signature| matches!(signature.qtype, 13 | 21));
            let affine = if f16_sidecars {
                GgmlAffineMatrix::from_ggml_bytes_with_progress_f16_sidecars(
                    bytes,
                    qtype,
                    input,
                    output,
                    |range| self.discard_tensor_byte_range(slot, range),
                )
            } else {
                GgmlAffineMatrix::from_ggml_bytes_with_progress(
                    bytes,
                    qtype,
                    input,
                    output,
                    |range| self.discard_tensor_byte_range(slot, range),
                )
            }?;
            self.record_affine(affine.transcode_stats());
            if self.enable_m234 && m234_signature.is_some() {
                Qwen35Linear::PinnedM234Affine(affine)
            } else if self.enable_m234 && m2_signature.is_some() {
                Qwen35Linear::PinnedM2Affine(affine)
            } else {
                Qwen35Linear::Affine(affine)
            }
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

    fn load_qkv_affine(
        &self,
        typed_slot: TensorSlot,
        expected_slot: usize,
        expected_qtype: u32,
        expected_output: u64,
    ) -> Result<GgmlAffineMatrix> {
        let slot = pinned_slot(typed_slot).map_err(anyhow::Error::msg)?;
        ensure!(
            slot == PinnedSlot::Target(expected_slot),
            "pinned QKV affine slot changed"
        );
        let (tensor, bytes) = self.lookup(slot)?;
        ensure!(
            tensor.dimensions == [5120, expected_output]
                && tensor.tensor_type.id() == expected_qtype
                && matches!(expected_qtype, 8 | 12 | 13 | 23),
            "pinned QKV affine descriptor changed"
        );
        let qtype = GgmlQType::try_from(expected_qtype)
            .context("pinned QKV affine qtype escaped allowlist")?;
        let f16_sidecars = self.mixed_q5_sidecars
            && expected_qtype == 13
            && expected_output == 12_288;
        let matrix = if f16_sidecars {
            GgmlAffineMatrix::from_ggml_bytes_with_progress_f16_sidecars(
                bytes,
                qtype,
                5120,
                usize::try_from(expected_output)?,
                |range| self.discard_tensor_byte_range(slot, range),
            )
        } else {
            GgmlAffineMatrix::from_ggml_bytes_with_progress(
                bytes,
                qtype,
                5120,
                usize::try_from(expected_output)?,
                |range| self.discard_tensor_byte_range(slot, range),
            )
        }?;
        self.record_affine(matrix.transcode_stats());
        self.discard_tensor_pages(slot);
        Ok(matrix)
    }

    fn load_qkv_kv(
        &self,
        typed_slot: TensorSlot,
        expected_slot: usize,
        expected_qtype: u32,
    ) -> Result<Qwen38QkvMatrix> {
        let slot = pinned_slot(typed_slot).map_err(anyhow::Error::msg)?;
        ensure!(
            slot == PinnedSlot::Target(expected_slot),
            "pinned QKV KV slot changed"
        );
        let (tensor, bytes) = self.lookup(slot)?;
        ensure!(
            tensor.dimensions == [5120, 1024]
                && tensor.tensor_type.id() == expected_qtype
                && matches!(expected_qtype, 8 | 12 | 13 | 14 | 23),
            "pinned QKV KV descriptor changed"
        );
        let qtype =
            GgmlQType::try_from(expected_qtype).context("pinned QKV KV qtype escaped allowlist")?;
        let matrix = if qtype == GgmlQType::Q6K {
            let dual = Qwen38Q6DualMatrix::from_pinned_bytes_with_progress(
                bytes,
                Qwen38Q6Shape::K5120N1024,
                |range| self.discard_tensor_byte_range(slot, range),
            )?;
            self.record_q6(dual.transcode_stats());
            Qwen38QkvMatrix::Q6(dual)
        } else {
            let affine = GgmlAffineMatrix::from_ggml_bytes_with_progress(
                bytes,
                qtype,
                5120,
                1024,
                |range| self.discard_tensor_byte_range(slot, range),
            )?;
            self.record_affine(affine.transcode_stats());
            Qwen38QkvMatrix::Affine(affine)
        };
        self.discard_tensor_pages(slot);
        Ok(matrix)
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
        self.load_linear(slot, pinned_slot(slot)?, &label)
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
        if self.enable_fusion
            && cfg!(target_os = "macos")
            && let Some(signature) = pinned_qkv_bundle_signature(role, layer)
        {
            let query = self
                .load_qkv_affine(
                    slot(LayerTensor::AttentionQuery),
                    signature.query_slot,
                    signature.query_qtype,
                    12_288,
                )
                .map_err(|error| error.to_string())?;
            let key = self
                .load_qkv_kv(
                    slot(LayerTensor::AttentionKey),
                    signature.key_slot,
                    signature.key_qtype,
                )
                .map_err(|error| error.to_string())?;
            let value = self
                .load_qkv_kv(
                    slot(LayerTensor::AttentionValue),
                    signature.value_slot,
                    signature.value_qtype,
                )
                .map_err(|error| error.to_string())?;
            return Qwen38MixedQkvBundle::new(query, key, value)
                .map(Qwen35QkvProjection::Bundled)
                .map_err(|error| error.to_string());
        }
        Ok(Qwen35QkvProjection::Separate {
            query: self.linear(slot(LayerTensor::AttentionQuery), group_size, bits)?,
            key: self.linear(slot(LayerTensor::AttentionKey), group_size, bits)?,
            value: self.linear(slot(LayerTensor::AttentionValue), group_size, bits)?,
        })
    }

    fn gguf_ssm_a_is_coefficient(&self) -> bool {
        true
    }

    fn qwen38_fusion_enabled(&self) -> bool {
        self.enable_fusion && cfg!(target_os = "macos")
    }

    fn qwen38_mlp_carrier_enabled(&self) -> bool {
        self.enable_fusion && self.enable_mlp_carrier && cfg!(target_os = "macos")
    }

    fn is_pinned_qwen38_gguf(&self) -> bool {
        true
    }
}

fn map_file(file: &GgufFile) -> Result<Mmap> {
    let map = unsafe { MmapOptions::new().map(file.handle()) }
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
    use std::collections::BTreeMap;

    #[test]
    fn qwen38_mlp_carrier_kill_switch_defaults_on() {
        assert!(parse_qwen38_mlp_carrier_enabled(None));
        assert!(parse_qwen38_mlp_carrier_enabled(Some("1")));
        assert!(parse_qwen38_mlp_carrier_enabled(Some("yes")));
        for disabled in ["0", "false", "OFF", " no "] {
            assert!(!parse_qwen38_mlp_carrier_enabled(Some(disabled)));
        }
    }

    fn max_ulp_bytes(left: &[u8], right: &[u8]) -> u32 {
        assert_eq!(left.len(), right.len());
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        left.chunks_exact(4)
            .zip(right.chunks_exact(4))
            .map(|(left, right)| {
                let left = ordered(f32::from_le_bytes(left.try_into().unwrap()));
                let right = ordered(f32::from_le_bytes(right.try_into().unwrap()));
                left.abs_diff(right)
            })
            .max()
            .unwrap_or(0)
    }


    fn max_ulp(left: &MlxArray, right: &MlxArray) -> u32 {
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        mlxcel_core::array_to_raw_bytes(left)
            .chunks_exact(4)
            .zip(mlxcel_core::array_to_raw_bytes(right).chunks_exact(4))
            .map(|(left, right)| {
                let left = ordered(f32::from_le_bytes(left.try_into().unwrap()));
                let right = ordered(f32::from_le_bytes(right.try_into().unwrap()));
                left.abs_diff(right)
            })
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn pinned_qkv_bundle_gate_is_exactly_the_16_full_attention_descriptors() {
        let expected_layers = [
            3usize, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
        ];
        for layer in 0..=64 {
            let signature = pinned_qkv_bundle_signature(ModelRole::Target, layer);
            if let Some(index) = expected_layers
                .iter()
                .position(|expected| *expected == layer)
            {
                let signature = signature.expect("pinned full-attention layer must be bundled");
                let base = 3 + layer * 14 - (layer / 4) * 3;
                assert_eq!(
                    (
                        signature.key_slot,
                        signature.query_slot,
                        signature.value_slot,
                    ),
                    (base, base + 4, base + 6),
                );
                assert_eq!(
                    (
                        signature.query_qtype,
                        signature.key_qtype,
                        signature.value_qtype,
                    ),
                    PINNED_QKV_QTYPES[index],
                );
            } else {
                assert!(
                    signature.is_none(),
                    "unexpected QKV bundle at layer {layer}"
                );
            }
            assert!(pinned_qkv_bundle_signature(ModelRole::Mtp, layer).is_none());
        }
    }

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
    fn pinned_m2_allowlist_is_exactly_four_mtp_descriptors() {
        const TENSORS: [LayerTensor; 20] = [
            LayerTensor::InputNorm,
            LayerTensor::PostAttentionNorm,
            LayerTensor::AttentionQuery,
            LayerTensor::AttentionKey,
            LayerTensor::AttentionValue,
            LayerTensor::AttentionOutput,
            LayerTensor::AttentionQueryNorm,
            LayerTensor::AttentionKeyNorm,
            LayerTensor::MlpGate,
            LayerTensor::MlpUp,
            LayerTensor::MlpDown,
            LayerTensor::LinearQkv,
            LayerTensor::LinearGate,
            LayerTensor::LinearBeta,
            LayerTensor::LinearAlpha,
            LayerTensor::LinearConv,
            LayerTensor::LinearDtBias,
            LayerTensor::LinearA,
            LayerTensor::LinearNorm,
            LayerTensor::LinearOutput,
        ];

        let mut slots = Vec::new();
        for tensor in TENSORS {
            let mtp = TensorSlot::Layer {
                role: ModelRole::Mtp,
                layer: 0,
                tensor,
            };
            if let Some(signature) = pinned_m2_affine_signature(mtp) {
                assert_eq!(pinned_slot(mtp), Ok(PinnedSlot::Mtp(signature.mtp_slot)));
                let descriptor = &crate::qwen38_plan::MTP_TENSOR_PLAN[signature.mtp_slot];
                assert_eq!(descriptor.qtype, signature.qtype);
                assert_eq!(descriptor.dimensions, signature.dimensions);
                slots.push(signature.mtp_slot);
            }
            assert!(
                pinned_m2_affine_signature(TensorSlot::Layer {
                    role: ModelRole::Target,
                    layer: 0,
                    tensor,
                })
                .is_none()
            );
            assert!(
                pinned_m2_affine_signature(TensorSlot::Layer {
                    role: ModelRole::Mtp,
                    layer: 1,
                    tensor,
                })
                .is_none()
            );
        }

        let projection =
            pinned_m2_affine_signature(TensorSlot::MtpProjection).expect("MTP projection");
        assert_eq!(projection.mtp_slot, 13);
        assert_eq!(projection.qtype, 12);
        assert_eq!(projection.dimensions, [10_240, 5120]);
        slots.push(projection.mtp_slot);
        slots.sort_unstable();
        assert_eq!(slots, [10, 11, 12, 13]);
        assert_eq!(
            slots
                .iter()
                .map(|&slot| crate::qwen38_plan::MTP_TENSOR_PLAN[slot].name)
                .collect::<Vec<_>>(),
            [
                "blk.64.ffn_down.weight",
                "blk.64.ffn_gate.weight",
                "blk.64.ffn_up.weight",
                "blk.64.nextn.eh_proj.weight",
            ]
        );

        for root in [
            TensorSlot::TokenEmbedding,
            TensorSlot::Output,
            TensorSlot::OutputNorm,
            TensorSlot::MtpEmbeddingNorm,
            TensorSlot::MtpHiddenNorm,
            TensorSlot::MtpHeadNorm,
        ] {
            assert!(pinned_m2_affine_signature(root).is_none());
        }
        for (slot, tensor) in [
            (3, LayerTensor::AttentionKey),
            (6, LayerTensor::AttentionOutput),
            (7, LayerTensor::AttentionQuery),
            (9, LayerTensor::AttentionValue),
        ] {
            let typed = TensorSlot::Layer {
                role: ModelRole::Mtp,
                layer: 0,
                tensor,
            };
            assert!(pinned_m2_affine_signature(typed).is_none());
            assert_eq!(pinned_slot(typed), Ok(PinnedSlot::Mtp(slot)));
            assert_eq!(crate::qwen38_plan::MTP_TENSOR_PLAN[slot].qtype, 14);
        }
    }

    #[test]
    fn pinned_m234_allowlist_is_exactly_the_430_target_descriptors() {
        const TENSORS: [LayerTensor; 10] = [
            LayerTensor::AttentionQuery,
            LayerTensor::MlpGate,
            LayerTensor::MlpUp,
            LayerTensor::MlpDown,
            LayerTensor::LinearQkv,
            LayerTensor::LinearGate,
            LayerTensor::LinearAlpha,
            LayerTensor::LinearBeta,
            LayerTensor::LinearOutput,
            LayerTensor::AttentionOutput,
        ];
        const EXPECTED_LAYERS: [usize; 64] = [
            8, 7, 7, 4, 8, 8, 6, 5, 8, 8, 8, 5, 8, 8, 8, 5, 8, 8, 8, 5, 8, 7, 7, 4, 7, 7, 8, 4, 7,
            8, 8, 4, 8, 8, 8, 4, 8, 8, 8, 5, 8, 8, 8, 4, 8, 8, 8, 5, 8, 8, 7, 4, 7, 8, 8, 5, 6, 6,
            5, 2, 7, 8, 7, 1,
        ];

        let mut signatures = Vec::new();
        for layer in 0..64 {
            for tensor in TENSORS {
                let slot = TensorSlot::Layer {
                    role: ModelRole::Target,
                    layer,
                    tensor,
                };
                if let Some(signature) = pinned_m234_affine_signature(slot) {
                    assert_eq!(
                        pinned_slot(slot),
                        Ok(PinnedSlot::Target(signature.target_slot))
                    );
                    let descriptor = &crate::qwen38_plan::TARGET_TENSOR_PLAN[signature.target_slot];
                    assert_eq!(descriptor.qtype, signature.qtype);
                    assert_eq!(descriptor.dimensions, signature.dimensions);
                    signatures.push(signature);
                }
                assert!(
                    pinned_m234_affine_signature(TensorSlot::Layer {
                        role: ModelRole::Mtp,
                        layer: 0,
                        tensor,
                    })
                    .is_none()
                );
            }
        }

        assert_eq!(signatures.len(), 430);
        let mut layer_counts = [0usize; 64];
        let mut tensor_counts = [0usize; 10];
        let mut qtype_counts = [0usize; 7];
        let mut tensor_qtypes = [[0usize; 7]; 10];
        let mut shape_counts = [0usize; 7];
        let mut occupied_slots = [false; 866];
        let mut slots = Vec::with_capacity(signatures.len());
        for signature in signatures {
            layer_counts[signature.layer] += 1;
            let tensor_index = match signature.tensor {
                LayerTensor::AttentionQuery => 0,
                LayerTensor::MlpGate => 1,
                LayerTensor::MlpUp => 2,
                LayerTensor::MlpDown => 3,
                LayerTensor::LinearQkv => 4,
                LayerTensor::LinearGate => 5,
                LayerTensor::LinearAlpha => 6,
                LayerTensor::LinearBeta => 7,
                LayerTensor::LinearOutput => 8,
                LayerTensor::AttentionOutput => 9,
                _ => unreachable!(),
            };
            let qtype_index = match signature.qtype {
                8 => 0,
                11 => 1,
                12 => 2,
                13 => 3,
                20 => 4,
                21 => 5,
                23 => 6,
                _ => unreachable!(),
            };
            tensor_counts[tensor_index] += 1;
            qtype_counts[qtype_index] += 1;
            tensor_qtypes[tensor_index][qtype_index] += 1;
            shape_counts[match signature.dimensions {
                [5120, 17_408] => 0,
                [17_408, 5120] => 1,
                [5120, 10_240] => 2,
                [5120, 6144] => 3,
                [5120, 48] => 4,
                [6144, 5120] => 5,
                [5120, 12_288] => 6,
                _ => unreachable!(),
            }] += 1;
            assert!(!occupied_slots[signature.target_slot]);
            occupied_slots[signature.target_slot] = true;
            slots.push(signature.target_slot);
        }
        assert_eq!(layer_counts, EXPECTED_LAYERS);
        assert_eq!(tensor_counts, [16, 60, 62, 59, 47, 47, 48, 48, 35, 8]);
        assert_eq!(qtype_counts, [97, 3, 67, 186, 6, 1, 70]);
        assert_eq!(
            tensor_qtypes,
            [
                [0, 0, 6, 8, 0, 0, 2],
                [0, 1, 13, 19, 2, 0, 25],
                [0, 2, 6, 32, 1, 0, 21],
                [0, 0, 5, 35, 2, 1, 16],
                [0, 0, 23, 21, 1, 0, 2],
                [0, 0, 13, 31, 0, 0, 3],
                [48, 0, 0, 0, 0, 0, 0],
                [48, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 33, 0, 0, 0],
                [0, 0, 0, 7, 0, 0, 1],
            ]
        );
        assert_eq!(shape_counts, [122, 59, 47, 47, 96, 43, 16]);

        slots.sort_unstable();
        let slot_fingerprint = slots.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, slot| {
            (*slot as u64)
                .to_le_bytes()
                .iter()
                .fold(hash, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
                })
        });
        assert_eq!(slots.first(), Some(&3));
        assert_eq!(slots.last(), Some(&844));
        assert_eq!(slots.iter().sum::<usize>(), 178_202);
        assert_eq!(slot_fingerprint, 0x5eae_6e1c_c1d2_f58c);
    }

    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 target and MTP GGUF pair"]
    fn real_mtp_m2_all_affine_rows_and_q6_sharing() {
        if !mlxcel_core::metal_is_available() {
            return;
        }

        let layer_slot = |tensor| TensorSlot::Layer {
            role: ModelRole::Mtp,
            layer: 0,
            tensor,
        };
        let weights = GgufWeightSource::open().expect("open pinned GGUF pair");
        let affine = [
            (10usize, layer_slot(LayerTensor::MlpDown)),
            (11, layer_slot(LayerTensor::MlpGate)),
            (12, layer_slot(LayerTensor::MlpUp)),
            (13, TensorSlot::MtpProjection),
        ];
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        for (plan_slot, typed_slot) in affine {
            let descriptor = &crate::qwen38_plan::MTP_TENSOR_PLAN[plan_slot];
            let signature =
                pinned_m2_affine_signature(typed_slot).expect("allowlisted MTP M2 signature");
            assert_eq!(signature.mtp_slot, plan_slot);
            assert_eq!(descriptor.qtype, 12);
            assert_eq!(descriptor.dimensions, signature.dimensions);
            let linear = weights
                .load_linear(typed_slot, PinnedSlot::Mtp(plan_slot), descriptor.name)
                .expect("load MTP affine");
            let Qwen35Linear::PinnedM2Affine(matrix) = &linear else {
                panic!("MTP slot {plan_slot} did not receive the pinned M2 path");
            };

            let width = matrix.in_features();
            for input_rows in [1usize, 2, 3, 4] {
                let values = (0..input_rows * width)
                    .map(|index| {
                        let row = index / width;
                        let column = index % width;
                        (column as i32 % 43 - 21) as f32 * 0.001953125 + row as f32 * 0.00048828125
                    })
                    .collect::<Vec<_>>();
                let input =
                    mlxcel_core::from_slice_f32(&values, &[1, input_rows as i32, width as i32]);
                let split_stats = matrix.dispatch_stats(input_rows).unwrap();
                let selected_stats = matrix.qwen38_m2_dispatch_stats(input_rows).unwrap();
                if input_rows == 2 {
                    assert_eq!(
                        selected_stats.path,
                        mlxcel_core::GgmlKernelPath::Qwen38AffineM234
                    );
                    assert_eq!(
                        split_stats.packed_bytes_read,
                        selected_stats.packed_bytes_read * 2
                    );
                    assert_eq!(
                        split_stats.activation_bytes_read,
                        selected_stats.activation_bytes_read
                    );
                    assert_eq!(
                        split_stats.output_bytes_written,
                        selected_stats.output_bytes_written
                    );
                } else {
                    assert_eq!(
                        selected_stats, split_stats,
                        "MTP slot {plan_slot} changed M={input_rows} dispatch"
                    );
                }

                let split = matrix.forward(input.as_ref().unwrap()).unwrap();
                let selected = linear.forward(input.as_ref().unwrap());
                mlxcel_core::eval(split.as_ref().unwrap());
                mlxcel_core::eval(selected.as_ref().unwrap());
                let split = mlxcel_core::array_to_raw_bytes(split.as_ref().unwrap());
                let selected = mlxcel_core::array_to_raw_bytes(selected.as_ref().unwrap());
                let max_ulp = split
                    .chunks_exact(4)
                    .zip(selected.chunks_exact(4))
                    .map(|(left, right)| {
                        let left = f32::from_le_bytes(left.try_into().unwrap());
                        let right = f32::from_le_bytes(right.try_into().unwrap());
                        ordered(left).abs_diff(ordered(right))
                    })
                    .max()
                    .unwrap_or(0);
                assert!(
                    split == selected || max_ulp <= 1,
                    "MTP slot {plan_slot} M={input_rows} differs by {max_ulp} ULP"
                );
            }
        }

        for (plan_slot, tensor) in [
            (3usize, LayerTensor::AttentionKey),
            (6, LayerTensor::AttentionOutput),
            (7, LayerTensor::AttentionQuery),
            (9, LayerTensor::AttentionValue),
        ] {
            let typed_slot = layer_slot(tensor);
            let descriptor = &crate::qwen38_plan::MTP_TENSOR_PLAN[plan_slot];
            assert_eq!(descriptor.qtype, 14);
            let linear = weights
                .load_linear(typed_slot, PinnedSlot::Mtp(plan_slot), descriptor.name)
                .expect("load MTP Q6");
            let Qwen35Linear::Q6Dual(matrix) = &linear else {
                panic!("MTP slot {plan_slot} no longer uses Q6Dual");
            };
            let stats = matrix.dispatch_stats(2).unwrap();
            assert_eq!(stats.path, mlxcel_core::GgmlKernelPath::VerifyM2To4);
            assert_eq!(
                stats.packed_bytes_read,
                usize::try_from(descriptor.byte_len).unwrap()
            );

            let width = usize::try_from(descriptor.dimensions[0]).unwrap();
            let values = (0..2 * width)
                .map(|index| (index as i32 % 29 - 14) as f32 * 0.001953125)
                .collect::<Vec<_>>();
            let input = mlxcel_core::from_slice_f32(&values, &[1, 2, width as i32]);
            let output = linear.forward(input.as_ref().unwrap());
            mlxcel_core::eval(output.as_ref().unwrap());
            assert_eq!(
                mlxcel_core::array_shape(output.as_ref().unwrap()),
                [1, 2, i32::try_from(descriptor.dimensions[1]).unwrap()]
            );
        }
    }


    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 target GGUF and exclusive Metal access"]
    fn real_pinned_m234_all_shape_qtype_pairs_are_exact() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        const TENSORS: [LayerTensor; 9] = [
            LayerTensor::AttentionQuery,
            LayerTensor::MlpGate,
            LayerTensor::MlpUp,
            LayerTensor::MlpDown,
            LayerTensor::LinearQkv,
            LayerTensor::LinearGate,
            LayerTensor::LinearAlpha,
            LayerTensor::LinearBeta,
            LayerTensor::LinearOutput,
        ];
        let mut representatives = BTreeMap::new();
        for layer in 0..64 {
            for tensor in TENSORS {
                let slot = TensorSlot::Layer {
                    role: ModelRole::Target,
                    layer,
                    tensor,
                };
                if let Some(signature) = pinned_m234_affine_signature(slot) {
                    representatives
                        .entry((signature.qtype, signature.dimensions))
                        .or_insert((slot, signature));
                }
            }
        }
        assert_eq!(
            representatives.len(),
            24,
            "pinned affine shape/qtype coverage changed"
        );

        let weights =
            GgufWeightSource::open_without_mixed_q5_sidecars().expect("open pinned GGUF pair");
        let mut aggregate_max_ulp = 0;
        for ((qtype, dimensions), (slot, signature)) in representatives {
            let linear = weights
                .load_linear(
                    slot,
                    PinnedSlot::Target(signature.target_slot),
                    "M234 shape/qtype representative",
                )
                .expect("load M234 representative");
            let Qwen35Linear::PinnedM234Affine(matrix) = linear else {
                panic!("representative did not receive the pinned M234 path");
            };
            assert!(
                !matrix.has_f16_sidecars(),
                "generic M234 gate requires FP32 sidecars"
            );
            let width = matrix.in_features();
            let output_width = matrix.out_features();
            let values = (0..4 * width)
                .map(|index| {
                    let row = index / width;
                    let column = index % width;
                    (column as i32 % 43 - 21) as f32 * 0.001953125
                        + row as f32 * 0.00048828125
                })
                .collect::<Vec<_>>();
            let input = mlxcel_core::from_slice_f32(&values, &[1, 4, width as i32]);
            let one_pass = matrix
                .forward_qwen38_m234(input.as_ref().unwrap())
                .expect("one-pass M4 representative");
            mlxcel_core::eval(one_pass.as_ref().unwrap());
            let one_pass = mlxcel_core::array_to_raw_bytes(one_pass.as_ref().unwrap());
            let selected_stats = matrix.qwen38_m234_dispatch_stats(4).unwrap();
            let sequential_stats = matrix.dispatch_stats(1).unwrap();
            assert_eq!(
                selected_stats.path,
                mlxcel_core::GgmlKernelPath::Qwen38AffineM234
            );
            assert_eq!(
                sequential_stats.packed_bytes_read,
                selected_stats.packed_bytes_read
            );

            let mut pair_max_ulp = 0;
            for row in 0..4 {
                let input_m1 = mlxcel_core::from_slice_f32(
                    &values[row * width..(row + 1) * width],
                    &[1, 1, width as i32],
                );
                let sequential = matrix.forward(input_m1.as_ref().unwrap()).expect("M1 row");
                mlxcel_core::eval(sequential.as_ref().unwrap());
                let sequential = mlxcel_core::array_to_raw_bytes(sequential.as_ref().unwrap());
                let row_bytes = output_width * 4;
                let range = row * row_bytes..(row + 1) * row_bytes;
                pair_max_ulp = pair_max_ulp.max(max_ulp_bytes(&sequential, &one_pass[range]));
            }
            assert!(
                pair_max_ulp <= 1,
                "slot {} qtype {} shape {:?} M4 differs by {} ULP",
                signature.target_slot,
                qtype,
                dimensions,
                pair_max_ulp
            );
            aggregate_max_ulp = aggregate_max_ulp.max(pair_max_ulp);
            eprintln!(
                "QWEN38_AFFINE_M234_PAIR qtype={} shape={}x{} max_ulp={}",
                qtype, dimensions[0], dimensions[1], pair_max_ulp
            );
            drop(matrix);
            mlxcel_core::memory::clear_cache();
        }
        eprintln!(
            "QWEN38_AFFINE_M234_COVERAGE pairs=24 max_m1_vs_m4_ulp={}",
            aggregate_max_ulp
        );
    }

    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 target GGUF and exclusive Metal access"]
    fn real_pinned_mlp_carrier_all_production_pairs_are_within_one_ulp() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        let mut representatives = BTreeMap::new();
        for descriptor in crate::qwen38_plan::QWEN38_MLP_FUSION_PLAN {
            if descriptor.role == crate::qwen38_plan::Qwen38PlanRole::Target
                && descriptor.carrier_compatible()
            {
                representatives
                    .entry((descriptor.qtypes[0], descriptor.qtypes[1]))
                    .or_insert(descriptor);
            }
        }
        assert_eq!(
            representatives.keys().copied().collect::<Vec<_>>(),
            vec![(11, 23), (12, 12), (13, 13), (20, 23), (23, 11), (23, 23)],
        );

        let weights = GgufWeightSource::open().expect("open pinned GGUF pair");
        let mut aggregate_max_ulp = 0u32;
        for ((gate_qtype, up_qtype), descriptor) in representatives {
            let gate = weights
                .load_linear(
                    TensorSlot::Layer {
                        role: ModelRole::Target,
                        layer: descriptor.layer,
                        tensor: LayerTensor::MlpGate,
                    },
                    PinnedSlot::Target(descriptor.slots[0]),
                    "MLP carrier gate representative",
                )
                .expect("load MLP carrier gate");
            let up = weights
                .load_linear(
                    TensorSlot::Layer {
                        role: ModelRole::Target,
                        layer: descriptor.layer,
                        tensor: LayerTensor::MlpUp,
                    },
                    PinnedSlot::Target(descriptor.slots[1]),
                    "MLP carrier up representative",
                )
                .expect("load MLP carrier up");
            let Qwen35Linear::PinnedM234Affine(gate) = gate else {
                panic!("MLP carrier gate representative is not pinned affine");
            };
            let Qwen35Linear::PinnedM234Affine(up) = up else {
                panic!("MLP carrier up representative is not pinned affine");
            };
            assert!(gate.supports_qwen38_mlp_carrier_with(&up));
            let baseline_gate = gate.clone_shared();
            let baseline_up = up.clone_shared();
            let fusion =
                mlxcel_core::Qwen38AffineMlpInputFusion::new(gate, up, [true, true], true)
                    .expect("construct MLP carrier");

            let mut pair_max_ulp = 0u32;
            for rows in [1usize, 3, 4] {
                let values = (0..rows * 5120)
                    .map(|index| {
                        let row = index / 5120;
                        let column = index % 5120;
                        (column as i32 % 47 - 23) as f32 * 0.001953125
                            + row as f32 * 0.000244140625
                    })
                    .collect::<Vec<_>>();
                let input =
                    mlxcel_core::from_slice_f32(&values, &[1, rows as i32, 5120]);
                let gate = baseline_gate
                    .forward_qwen38_m234(input.as_ref().unwrap())
                    .expect("baseline gate");
                let up = baseline_up
                    .forward_qwen38_m234(input.as_ref().unwrap())
                    .expect("baseline up");
                let expected = mlxcel_core::compiled_swiglu_activation(
                    gate.as_ref().unwrap(),
                    up.as_ref().unwrap(),
                );
                let actual = fusion
                    .forward(input.as_ref().unwrap())
                    .expect("MLP carrier output");
                mlxcel_core::eval(expected.as_ref().unwrap());
                mlxcel_core::eval(actual.as_ref().unwrap());
                let max_ulp = max_ulp(expected.as_ref().unwrap(), actual.as_ref().unwrap());
                assert!(
                    max_ulp <= 1,
                    "layer {} qtypes {gate_qtype}/{up_qtype} M{rows} differs by {max_ulp} ULP",
                    descriptor.layer,
                );
                pair_max_ulp = pair_max_ulp.max(max_ulp);
                let stats = fusion.dispatch_stats(rows).unwrap();
                assert_eq!(stats.physical_dispatches, 1);
                assert_eq!(
                    stats.intermediate_bytes_avoided,
                    rows * 17_408 * 8,
                );
            }
            aggregate_max_ulp = aggregate_max_ulp.max(pair_max_ulp);
            eprintln!(
                "QWEN38_MLP_CARRIER_PAIR layer={} qtypes={}/{} max_ulp={}",
                descriptor.layer, gate_qtype, up_qtype, pair_max_ulp,
            );
            drop(fusion);
            drop(baseline_gate);
            drop(baseline_up);
            mlxcel_core::memory::clear_cache();
        }
        eprintln!(
            "QWEN38_MLP_CARRIER_COVERAGE pairs=6 rows=1,3,4 max_ulp={aggregate_max_ulp}"
        );
    }

    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 target GGUF and exclusive Metal access"]
    fn real_pinned_mixed_q5_all_descriptors_are_exact_and_compact() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        const TENSORS: [LayerTensor; 8] = [
            LayerTensor::AttentionQuery,
            LayerTensor::MlpGate,
            LayerTensor::MlpUp,
            LayerTensor::MlpDown,
            LayerTensor::LinearQkv,
            LayerTensor::LinearGate,
            LayerTensor::LinearOutput,
            LayerTensor::AttentionOutput,
        ];
        let mut signatures = Vec::new();
        for layer in 0..64 {
            for tensor in TENSORS {
                let slot = TensorSlot::Layer {
                    role: ModelRole::Target,
                    layer,
                    tensor,
                };
                if let Some(signature) = pinned_m234_affine_signature(slot)
                    && matches!(signature.qtype, 13 | 21)
                {
                    signatures.push((slot, signature));
                }
            }
        }
        assert_eq!(signatures.len(), 187);
        assert_eq!(
            signatures
                .iter()
                .filter(|(_, signature)| signature.qtype == 13)
                .count(),
            186
        );
        assert_eq!(
            signatures
                .iter()
                .filter(|(_, signature)| signature.qtype == 21)
                .count(),
            1
        );

        let weights = GgufWeightSource::open().expect("open mixed pinned GGUF pair");
        let mut total_old_resident = 0usize;
        let mut total_mixed_resident = 0usize;
        let mut max_m1_vs_m3_ulp = 0;
        let mut max_m1_vs_m4_ulp = 0;
        for (slot, signature) in signatures {
            let linear = weights
                .load_linear(
                    slot,
                    PinnedSlot::Target(signature.target_slot),
                    "FP16-sidecar Q5 descriptor",
                )
                .expect("load pinned Q5 descriptor");
            let Qwen35Linear::PinnedM234Affine(matrix) = linear else {
                panic!("slot {} did not receive pinned affine storage", signature.target_slot);
            };
            let width = matrix.in_features();
            let output_width = matrix.out_features();
            let source_slot = PinnedSlot::Target(signature.target_slot);
            let (tensor, map) = weights.slot_parts(source_slot);
            let start = usize::try_from(tensor.absolute_offset).expect("tensor offset");
            let end = start
                .checked_add(usize::try_from(tensor.byte_len).expect("tensor byte length"))
                .expect("tensor end");
            let old_matrix = GgmlAffineMatrix::from_ggml_bytes(
                &map[start..end],
                GgmlQType::try_from(signature.qtype).expect("Q5/IQ3S qtype"),
                width,
                output_width,
            )
            .expect("construct FP32-sidecar reference from verified source");
            let resident_before = matrix.transcode_stats();
            let old_resident_before = old_matrix.transcode_stats();
            let weight_count = width
                .checked_mul(output_width)
                .expect("pinned Q5 weight count");
            assert!(matrix.has_f16_sidecars());
            assert!(!old_matrix.has_f16_sidecars());
            assert_eq!(resident_before.resident_bytes, weight_count * 3 / 4);
            assert_eq!(old_resident_before.resident_bytes, weight_count);
            total_old_resident += old_resident_before.resident_bytes;
            total_mixed_resident += resident_before.resident_bytes;
            let values = (0..4 * width)
                .map(|index| {
                    let row = index / width;
                    let column = index % width;
                    let centered = ((column * 17 + row * 13) % 257) as i32 - 128;
                    centered as f32 * 0.001901 + row as f32 * 0.000173
                })
                .collect::<Vec<_>>();
            let input_m3 = mlxcel_core::from_slice_f32(&values[..3 * width], &[1, 3, width as i32]);
            let input_m4 = mlxcel_core::from_slice_f32(&values, &[1, 4, width as i32]);
            let mixed_m3 = matrix.forward_qwen38_m234(input_m3.as_ref().unwrap())
                .expect("mixed M3");
            let mixed_m3_repeat = matrix.forward_qwen38_m234(input_m3.as_ref().unwrap())
                .expect("mixed M3 repeat");
            let mixed_m4 = matrix.forward_qwen38_m234(input_m4.as_ref().unwrap())
                .expect("mixed M4");
            for output in [&mixed_m3, &mixed_m3_repeat, &mixed_m4] {
                mlxcel_core::eval(output.as_ref().unwrap());
            }
            assert_eq!(
                mlxcel_core::array_shape(mixed_m3.as_ref().unwrap()),
                [1, 3, output_width as i32]
            );
            assert_eq!(
                mlxcel_core::array_shape(mixed_m4.as_ref().unwrap()),
                [1, 4, output_width as i32]
            );
            let mixed_m3 = mlxcel_core::array_to_raw_bytes(mixed_m3.as_ref().unwrap());
            let mixed_m3_repeat =
                mlxcel_core::array_to_raw_bytes(mixed_m3_repeat.as_ref().unwrap());
            let mixed_m4 = mlxcel_core::array_to_raw_bytes(mixed_m4.as_ref().unwrap());
            assert_eq!(
                mixed_m3, mixed_m3_repeat,
                "slot {} mixed M3 is not repeat deterministic",
                signature.target_slot
            );

            for row in 0..4 {
                let input_m1 = mlxcel_core::from_slice_f32(
                    &values[row * width..(row + 1) * width],
                    &[1, 1, width as i32],
                );
                let mixed_m1 = matrix.forward_qwen38_m234(input_m1.as_ref().unwrap())
                    .expect("mixed M1");
                mlxcel_core::eval(mixed_m1.as_ref().unwrap());
                let mixed_m1 = mlxcel_core::array_to_raw_bytes(mixed_m1.as_ref().unwrap());
                let row_bytes = output_width * 4;
                let m4_range = row * row_bytes..(row + 1) * row_bytes;
                let m4_ulp = max_ulp_bytes(&mixed_m1, &mixed_m4[m4_range]);
                max_m1_vs_m4_ulp = max_m1_vs_m4_ulp.max(m4_ulp);
                assert!(
                    m4_ulp <= 1,
                    "slot {} qtype {} mixed M1 row {row} vs M4 differs by {m4_ulp} ULP",
                    signature.target_slot,
                    signature.qtype,
                );
                if row < 3 {
                    let m3_range = row * row_bytes..(row + 1) * row_bytes;
                    let m3_ulp = max_ulp_bytes(&mixed_m1, &mixed_m3[m3_range]);
                    max_m1_vs_m3_ulp = max_m1_vs_m3_ulp.max(m3_ulp);
                    assert!(
                        m3_ulp <= 1,
                        "slot {} qtype {} mixed M1 row {row} vs M3 differs by {m3_ulp} ULP",
                        signature.target_slot,
                        signature.qtype,
                    );
                }
            }

            assert_eq!(
                matrix.transcode_stats(),
                resident_before,
                "slot {} changed resident affine storage during dispatch",
                signature.target_slot,
            );
            assert_eq!(
                old_matrix.transcode_stats(),
                old_resident_before,
                "slot {} changed old resident affine storage during dispatch",
                signature.target_slot,
            );
            drop((matrix, old_matrix));
            mlxcel_core::memory::clear_cache();
        }

        println!(
            "QWEN38_MIXED_Q5_PARITY descriptors=187 q5=186 iq3s=1 m3_repeat_exact=true max_m1_vs_m3_ulp={} max_m1_vs_m4_ulp={}",
            max_m1_vs_m3_ulp,
            max_m1_vs_m4_ulp,
        );
        println!(
            "QWEN38_MIXED_Q5_RESIDENCY descriptors=187 old_bytes={} mixed_bytes={} saved_bytes={} bytes_per_weight_old=1 bytes_per_weight_mixed=0.75 reduction_percent={:.4}",
            total_old_resident,
            total_mixed_resident,
            total_old_resident - total_mixed_resident,
            100.0 * (total_old_resident - total_mixed_resident) as f64
                / total_old_resident as f64,
        );
    }

    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 target and MTP GGUF pair"]
    fn real_pinned_qkv_bundle_is_bit_exact_for_all_layers_and_rows() {
        if !mlxcel_core::metal_is_available() {
            return;
        }
        let baseline_weights =
            GgufWeightSource::open_without_fusion().expect("open baseline pinned GGUF pair");
        let bundled_weights = GgufWeightSource::open().expect("open bundled pinned GGUF pair");
        for layer in [
            3usize, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
        ] {
            let baseline = baseline_weights
                .qkv(ModelRole::Target, layer, 32, 4, 96, 8, 128)
                .expect("load baseline full-attention QKV");
            assert!(matches!(&baseline, Qwen35QkvProjection::Separate { .. }));
            let bundled = bundled_weights
                .qkv(ModelRole::Target, layer, 32, 4, 96, 8, 128)
                .expect("load bundled full-attention QKV");
            assert!(matches!(&bundled, Qwen35QkvProjection::Bundled(_)));

            for input_rows in [1usize, 3, 4] {
                let values = (0..input_rows * 5120)
                    .map(|index| {
                        let row = index / 5120;
                        let column = index % 5120;
                        (column as i32 % 43 - 21) as f32 * 0.001953125 + row as f32 * 0.00048828125
                    })
                    .collect::<Vec<_>>();
                let input = mlxcel_core::from_slice_f32(&values, &[1, input_rows as i32, 5120]);
                let mut baseline_q: Option<UniquePtr<MlxArray>> = None;
                let mut baseline_k: Option<UniquePtr<MlxArray>> = None;
                let mut baseline_v: Option<UniquePtr<MlxArray>> = None;
                for row in 0..input_rows {
                    let row_input = mlxcel_core::slice(
                        input.as_ref().unwrap(),
                        &[0, row as i32, 0],
                        &[1, row as i32 + 1, 5120],
                    );
                    let (row_q, row_k, row_v) = baseline.forward(&row_input);
                    baseline_q = Some(match baseline_q {
                        Some(previous) => mlxcel_core::concatenate(&previous, &row_q, 1),
                        None => row_q,
                    });
                    baseline_k = Some(match baseline_k {
                        Some(previous) => mlxcel_core::concatenate(&previous, &row_k, 1),
                        None => row_k,
                    });
                    baseline_v = Some(match baseline_v {
                        Some(previous) => mlxcel_core::concatenate(&previous, &row_v, 1),
                        None => row_v,
                    });
                }
                let baseline_q = baseline_q.unwrap();
                let baseline_k = baseline_k.unwrap();
                let baseline_v = baseline_v.unwrap();
                let (bundled_q, bundled_k, bundled_v) = bundled.forward(input.as_ref().unwrap());
                for (name, baseline, bundled, output_rows) in [
                    (
                        "Q",
                        baseline_q.as_ref().unwrap(),
                        bundled_q.as_ref().unwrap(),
                        12_288,
                    ),
                    (
                        "K",
                        baseline_k.as_ref().unwrap(),
                        bundled_k.as_ref().unwrap(),
                        1024,
                    ),
                    (
                        "V",
                        baseline_v.as_ref().unwrap(),
                        bundled_v.as_ref().unwrap(),
                        1024,
                    ),
                ] {
                    mlxcel_core::eval(baseline);
                    mlxcel_core::eval(bundled);
                    let expected_shape = [1, input_rows as i32, output_rows];
                    assert_eq!(mlxcel_core::array_shape(baseline), expected_shape);
                    assert_eq!(mlxcel_core::array_shape(bundled), expected_shape);
                    assert_eq!(
                        max_ulp(baseline, bundled),
                        0,
                        "layer {layer} {name} M={input_rows} changed",
                    );
                }
            }
        }
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
