use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, symlink_metadata};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Clone, Copy)]
pub(crate) struct GgufLimits {
    pub max_file_bytes: u64,
    pub max_header_bytes: u64,
    pub max_metadata_bytes: u64,
    pub max_metadata_count: u64,
    pub max_metadata_values: u64,
    pub max_array_elements: u64,
    pub max_string_bytes: u64,
    pub max_tensor_count: u64,
    pub max_alignment: u32,
    pub max_array_depth: usize,
}

impl Default for GgufLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 1 << 40,
            max_header_bytes: 512 << 20,
            max_metadata_bytes: 256 << 20,
            max_metadata_count: 65_536,
            max_metadata_values: 4_000_000,
            max_array_elements: 2_000_000,
            max_string_bytes: 16 << 20,
            max_tensor_count: 1_000_000,
            max_alignment: 1 << 20,
            max_array_depth: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum MetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataType {
    fn parse(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => bail!("unknown GGUF metadata type {raw}"),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: MetadataType,
        values: Vec<MetadataValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(value) => Some((*value).into()),
            Self::Uint16(value) => Some((*value).into()),
            Self::Uint32(value) => Some((*value).into()),
            Self::Uint64(value) => Some(*value),
            Self::Int8(value) => (*value).try_into().ok(),
            Self::Int16(value) => (*value).try_into().ok(),
            Self::Int32(value) => (*value).try_into().ok(),
            Self::Int64(value) => (*value).try_into().ok(),
            _ => None,
        }
    }

    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(value) => Some((*value).into()),
            Self::Float64(value) => Some(*value),
            _ => self.as_u64().map(|value| value as f64),
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array { values, .. } => Some(values),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct GgmlType(u32);

impl GgmlType {
    pub(crate) fn id(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug)]
pub(crate) struct GgufFile {
    path: PathBuf,
    file_len: u64,
    data_offset: u64,
    alignment: u32,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<GgufTensorInfo>,
}

impl GgufFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        Self::open_with_limits(path, GgufLimits::default())
    }

    pub(crate) fn open_with_limits(path: &Path, limits: GgufLimits) -> Result<Self> {
        validate_local_file(path)?;
        let file_len = symlink_metadata(path)
            .with_context(|| format!("failed to stat GGUF {}", path.display()))?
            .len();
        ensure!(
            file_len <= limits.max_file_bytes,
            "GGUF file exceeds byte limit"
        );
        ensure!(file_len >= 24, "GGUF file is shorter than its fixed header");

        let file =
            File::open(path).with_context(|| format!("failed to open GGUF {}", path.display()))?;
        let mut reader = BoundedReader::new(file, file_len, limits.max_header_bytes);
        ensure!(reader.bytes::<4>()? == GGUF_MAGIC, "invalid GGUF magic");
        let version = reader.u32()?;
        ensure!(
            version == GGUF_VERSION,
            "unsupported GGUF version {version}; expected v3"
        );
        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        ensure!(
            tensor_count <= limits.max_tensor_count,
            "GGUF tensor count {tensor_count} exceeds limit {}",
            limits.max_tensor_count
        );
        ensure!(
            metadata_count <= limits.max_metadata_count,
            "GGUF metadata count {metadata_count} exceeds limit {}",
            limits.max_metadata_count
        );

        let metadata_start = reader.position();
        let mut value_count = 0_u64;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            ensure!(
                reader.position() - metadata_start <= limits.max_metadata_bytes,
                "GGUF metadata exceeds byte limit"
            );
            let key = reader.string(65_535, "metadata key")?;
            validate_metadata_key(&key)?;
            let value_type = MetadataType::parse(reader.u32()?)?;
            let value = read_metadata_value(&mut reader, value_type, 0, &limits, &mut value_count)?;
            ensure!(
                metadata.insert(key.clone(), value).is_none(),
                "duplicate GGUF metadata key {key:?}"
            );
        }
        ensure!(
            reader.position() - metadata_start <= limits.max_metadata_bytes,
            "GGUF metadata exceeds byte limit"
        );

        let alignment = metadata
            .get("general.alignment")
            .map(|value| {
                value
                    .as_u64()
                    .context("general.alignment must be a non-negative integer")
                    .and_then(|value| {
                        u32::try_from(value).context("general.alignment does not fit u32")
                    })
            })
            .transpose()?
            .unwrap_or(DEFAULT_ALIGNMENT);
        ensure!(
            alignment >= 8 && alignment.is_multiple_of(8) && alignment <= limits.max_alignment,
            "invalid GGUF alignment {alignment}"
        );

        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(tensor_count as usize)
            .context("failed to reserve GGUF tensor inventory")?;
        let mut tensor_names = BTreeSet::new();
        for _ in 0..tensor_count {
            let name = reader.string(64, "tensor name")?;
            validate_tensor_name(&name)?;
            ensure!(
                tensor_names.insert(name.clone()),
                "duplicate GGUF tensor name {name:?}"
            );
            let rank = reader.u32()?;
            ensure!(
                (1..=4).contains(&rank),
                "invalid rank {rank} for tensor {name}"
            );
            let mut dimensions = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                let dimension = reader.u64()?;
                ensure!(dimension > 0, "tensor {name} has a zero dimension");
                dimensions.push(dimension);
            }
            let raw_type = reader.u32()?;
            let layout = type_layout(raw_type)
                .with_context(|| format!("tensor {name} has unsupported GGML type {raw_type}"))?;
            ensure!(
                dimensions[0].is_multiple_of(layout.block_elements),
                "tensor {name} row width {} is not divisible by {}-element {} blocks",
                dimensions[0],
                layout.block_elements,
                layout.name
            );
            let relative_offset = reader.u64()?;
            ensure!(
                relative_offset.is_multiple_of(alignment.into()),
                "tensor {name} offset {relative_offset} is not aligned to {alignment}"
            );
            let byte_len = tensor_byte_len(&dimensions, layout)
                .with_context(|| format!("tensor {name} byte length overflow"))?;
            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                tensor_type: GgmlType(raw_type),
                relative_offset,
                absolute_offset: 0,
                byte_len,
            });
        }

        let header_end = reader.position();
        let data_offset =
            align_up(header_end, alignment.into()).context("GGUF data offset overflow")?;
        ensure!(
            data_offset <= file_len,
            "GGUF tensor data starts past end of file"
        );
        let padding_len =
            usize::try_from(data_offset - header_end).context("GGUF padding too large")?;
        if padding_len > 0 {
            let padding = reader.vec(padding_len)?;
            ensure!(
                padding.iter().all(|byte| *byte == 0),
                "GGUF header padding is not zero-filled"
            );
        }

        let mut ranges = Vec::with_capacity(tensors.len());
        for tensor in &mut tensors {
            tensor.absolute_offset = data_offset
                .checked_add(tensor.relative_offset)
                .context("GGUF tensor absolute offset overflow")?;
            let end = tensor
                .absolute_offset
                .checked_add(tensor.byte_len)
                .context("GGUF tensor end offset overflow")?;
            ensure!(
                end <= file_len,
                "tensor {} range {}..{} exceeds file length {file_len}",
                tensor.name,
                tensor.absolute_offset,
                end
            );
            ranges.push((tensor.absolute_offset, end, tensor.name.as_str()));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            ensure!(
                pair[1].0 >= pair[0].1,
                "GGUF tensors {:?} and {:?} overlap",
                pair[0].2,
                pair[1].2
            );
        }

        Ok(Self {
            path: path.to_owned(),
            file_len,
            data_offset,
            alignment,
            metadata,
            tensors,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    pub(crate) fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }
}

#[derive(Debug)]
pub(crate) struct PinnedGgufPair {
    pub target: GgufFile,
    pub mtp: GgufFile,
}

impl PinnedGgufPair {
    pub(crate) fn open() -> Result<Self> {
        let root = crate::pinned_model::resolve_pinned_model_dir()?;
        Self::open_in(&root)
    }

    fn open_in(root: &Path) -> Result<Self> {
        // Payload integrity is established before parsing, mmap, MLX, or FFI.
        let (target_path, mtp_path) = crate::pinned_model::verify_pair_directory(root)?;
        let target = GgufFile::open(&target_path).context("failed to parse pinned target GGUF")?;
        let mtp = GgufFile::open(&mtp_path).context("failed to parse pinned MTP GGUF")?;
        validate_plan(
            &target,
            crate::pinned_model::PINNED_TARGET.size,
            crate::qwen38_plan::TARGET_DATA_OFFSET,
            &crate::qwen38_plan::TARGET_TENSOR_PLAN,
            "target",
        )?;
        validate_plan(
            &mtp,
            crate::pinned_model::PINNED_MTP.size,
            crate::qwen38_plan::MTP_DATA_OFFSET,
            &crate::qwen38_plan::MTP_TENSOR_PLAN,
            "MTP",
        )?;
        validate_pinned_metadata(&target, &mtp)?;
        Ok(Self { target, mtp })
    }
}

fn validate_plan(
    file: &GgufFile,
    expected_file_len: u64,
    expected_data_offset: u64,
    plan: &[crate::qwen38_plan::PinnedTensorDescriptor],
    role: &str,
) -> Result<()> {
    ensure!(
        file.file_len == expected_file_len,
        "pinned {role} file size does not match its manifest"
    );
    ensure!(
        file.alignment == 32 && file.data_offset == expected_data_offset,
        "pinned {role} GGUF header geometry does not match the exact plan"
    );
    ensure!(
        file.tensors.len() == plan.len(),
        "pinned {role} tensor count {} does not match exact plan {}",
        file.tensors.len(),
        plan.len()
    );
    for (slot, (actual, expected)) in file.tensors.iter().zip(plan).enumerate() {
        ensure!(
            actual.name == expected.name,
            "pinned {role} tensor slot {slot} name {:?} does not match {:?}",
            actual.name,
            expected.name
        );
        ensure!(
            actual.dimensions == expected.dimensions,
            "pinned {role} tensor slot {slot} dimensions do not match the exact plan"
        );
        ensure!(
            actual.tensor_type.id() == expected.qtype
                && crate::qwen38_plan::PINNED_QTYPES.contains(&expected.qtype),
            "pinned {role} tensor slot {slot} qtype does not match the closed plan"
        );
        ensure!(
            actual.relative_offset == expected.relative_offset
                && actual.byte_len == expected.byte_len
                && actual.absolute_offset == expected.data_offset,
            "pinned {role} tensor slot {slot} byte range does not match the exact plan"
        );
    }
    Ok(())
}

fn validate_pinned_metadata(target: &GgufFile, mtp: &GgufFile) -> Result<()> {
    const INTEGERS: &[(&str, u64)] = &[
        ("general.sampling.top_k", 20),
        ("general.base_model.count", 1),
        ("qwen35.block_count", 65),
        ("qwen35.context_length", 262_144),
        ("qwen35.embedding_length", 5_120),
        ("qwen35.feed_forward_length", 17_408),
        ("qwen35.attention.head_count", 24),
        ("qwen35.attention.head_count_kv", 4),
        ("qwen35.attention.key_length", 256),
        ("qwen35.attention.value_length", 256),
        ("qwen35.nextn_predict_layers", 1),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.ssm.state_size", 128),
        ("qwen35.ssm.group_count", 16),
        ("qwen35.ssm.time_step_rank", 48),
        ("qwen35.ssm.inner_size", 6_144),
        ("qwen35.full_attention_interval", 4),
        ("qwen35.rope.dimension_count", 64),
        ("tokenizer.ggml.eos_token_id", 248_046),
        ("tokenizer.ggml.padding_token_id", 248_055),
        ("tokenizer.ggml.bos_token_id", 248_044),
        ("general.quantization_version", 2),
    ];
    for (role, file, file_type, template_len) in
        [("target", target, 15, 9_993), ("MTP", mtp, 14, 8_945)]
    {
        let metadata = &file.metadata;
        ensure!(
            metadata.len() == 50,
            "pinned {role} metadata count does not match the exact contract"
        );
        for (key, expected) in [
            ("general.architecture", "qwen35"),
            ("general.type", "model"),
            ("general.name", "Qwen3.8-27B"),
            ("general.basename", "Qwen3.8-27B"),
            ("general.size_label", "27B"),
            ("general.license", "apache-2.0"),
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "qwen35"),
        ] {
            ensure!(
                metadata.get(key).and_then(MetadataValue::as_str) == Some(expected),
                "pinned {role} metadata {key} does not match"
            );
        }
        for (key, expected) in INTEGERS {
            ensure!(
                metadata.get(*key).and_then(MetadataValue::as_u64) == Some(*expected),
                "pinned {role} metadata {key} does not match"
            );
        }
        ensure!(
            metadata
                .get("general.file_type")
                .and_then(MetadataValue::as_u64)
                == Some(file_type),
            "pinned {role} general.file_type does not match"
        );
        for (key, expected_len) in [
            ("tokenizer.ggml.tokens", 248_320),
            ("tokenizer.ggml.token_type", 248_320),
            ("tokenizer.ggml.merges", 247_587),
        ] {
            ensure!(
                metadata
                    .get(key)
                    .and_then(MetadataValue::as_array)
                    .is_some_and(|values| values.len() == expected_len),
                "pinned {role} metadata {key} length does not match"
            );
        }
        ensure!(
            metadata
                .get("tokenizer.chat_template")
                .and_then(MetadataValue::as_str)
                .is_some_and(|template| template.len() == template_len),
            "pinned {role} chat template length does not match"
        );
    }
    let sections = target
        .metadata
        .get("qwen35.rope.dimension_sections")
        .and_then(MetadataValue::as_array)
        .context("pinned target is missing RoPE dimension sections")?;
    ensure!(
        sections
            .iter()
            .map(MetadataValue::as_u64)
            .collect::<Option<Vec<_>>>()
            == Some(vec![11, 11, 10, 0]),
        "pinned target RoPE dimension sections do not match"
    );
    Ok(())
}

fn read_metadata_value(
    reader: &mut BoundedReader,
    value_type: MetadataType,
    depth: usize,
    limits: &GgufLimits,
    value_count: &mut u64,
) -> Result<MetadataValue> {
    *value_count = value_count
        .checked_add(1)
        .context("GGUF metadata value count overflow")?;
    ensure!(
        *value_count <= limits.max_metadata_values,
        "GGUF metadata value count exceeds limit"
    );
    Ok(match value_type {
        MetadataType::Uint8 => MetadataValue::Uint8(reader.u8()?),
        MetadataType::Int8 => MetadataValue::Int8(reader.u8()? as i8),
        MetadataType::Uint16 => MetadataValue::Uint16(reader.u16()?),
        MetadataType::Int16 => MetadataValue::Int16(reader.u16()? as i16),
        MetadataType::Uint32 => MetadataValue::Uint32(reader.u32()?),
        MetadataType::Int32 => MetadataValue::Int32(reader.u32()? as i32),
        MetadataType::Float32 => MetadataValue::Float32(f32::from_bits(reader.u32()?)),
        MetadataType::Bool => {
            let value = reader.u8()?;
            ensure!(value <= 1, "invalid GGUF boolean value {value}");
            MetadataValue::Bool(value == 1)
        }
        MetadataType::String => {
            MetadataValue::String(reader.string(limits.max_string_bytes, "metadata string")?)
        }
        MetadataType::Array => {
            ensure!(
                depth < limits.max_array_depth,
                "GGUF metadata array nesting is too deep"
            );
            let element_type = MetadataType::parse(reader.u32()?)?;
            let len = reader.u64()?;
            ensure!(
                len <= limits.max_array_elements,
                "GGUF metadata array length {len} exceeds limit {}",
                limits.max_array_elements
            );
            let mut values = Vec::new();
            values
                .try_reserve_exact(len as usize)
                .context("failed to reserve GGUF metadata array")?;
            for _ in 0..len {
                values.push(read_metadata_value(
                    reader,
                    element_type,
                    depth + 1,
                    limits,
                    value_count,
                )?);
            }
            MetadataValue::Array {
                element_type,
                values,
            }
        }
        MetadataType::Uint64 => MetadataValue::Uint64(reader.u64()?),
        MetadataType::Int64 => MetadataValue::Int64(reader.u64()? as i64),
        MetadataType::Float64 => MetadataValue::Float64(f64::from_bits(reader.u64()?)),
    })
}

fn validate_local_file(path: &Path) -> Result<()> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("GGUF path must end in a UTF-8 filename")?;
    validate_gguf_filename(filename)?;
    let metadata = symlink_metadata(path)
        .with_context(|| format!("failed to stat GGUF {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "GGUF path must be a regular file"
    );
    ensure!(
        !metadata.file_type().is_symlink(),
        "GGUF path must not be a symlink"
    );
    Ok(())
}

fn validate_gguf_filename(filename: &str) -> Result<()> {
    ensure!(
        !filename.is_empty() && filename.len() <= 255,
        "invalid GGUF filename length"
    );
    ensure!(filename.is_ascii(), "GGUF filename must be ASCII");
    ensure!(
        filename.ends_with(".gguf"),
        "GGUF filename must end in .gguf"
    );
    ensure!(
        Path::new(filename).components().count() == 1
            && filename != "."
            && filename != ".."
            && !filename.contains(['/', '\\']),
        "GGUF filename must not contain a path"
    );
    ensure!(
        filename
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')),
        "GGUF filename contains unsafe characters"
    );
    Ok(())
}

fn validate_metadata_key(key: &str) -> Result<()> {
    ensure!(!key.is_empty(), "GGUF metadata key is empty");
    ensure!(key.is_ascii(), "GGUF metadata key must be ASCII");
    for segment in key.split('.') {
        ensure!(
            !segment.is_empty(),
            "GGUF metadata key has an empty segment"
        );
        ensure!(
            segment
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
            "GGUF metadata key {key:?} is not lower_snake_case"
        );
    }
    Ok(())
}

fn validate_tensor_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty(), "GGUF tensor name is empty");
    ensure!(
        name.chars().all(|character| !character.is_control()),
        "GGUF tensor name contains a control character"
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct TypeLayout {
    name: &'static str,
    block_elements: u64,
    block_bytes: u64,
}

fn type_layout(raw: u32) -> Option<TypeLayout> {
    let (name, block_elements, block_bytes) = match raw {
        0 => ("F32", 1, 4),
        8 => ("Q8_0", 32, 34),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        23 => ("IQ4_XS", 256, 136),
        _ => return None,
    };
    Some(TypeLayout {
        name,
        block_elements,
        block_bytes,
    })
}

fn tensor_byte_len(dimensions: &[u64], layout: TypeLayout) -> Option<u64> {
    let row_bytes = (dimensions[0] / layout.block_elements).checked_mul(layout.block_bytes)?;
    dimensions[1..]
        .iter()
        .try_fold(row_bytes, |bytes, dimension| bytes.checked_mul(*dimension))
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|value| value / alignment * alignment)
}

struct BoundedReader {
    file: File,
    position: u64,
    file_len: u64,
    max_position: u64,
}

impl BoundedReader {
    fn new(file: File, file_len: u64, max_position: u64) -> Self {
        Self {
            file,
            position: 0,
            file_len,
            max_position,
        }
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0_u8; N];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn vec(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .context("failed to reserve GGUF byte field")?;
        bytes.resize(len, 0);
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .context("GGUF read offset overflow")?;
        ensure!(end <= self.file_len, "unexpected end of GGUF file");
        ensure!(end <= self.max_position, "GGUF header exceeds byte limit");
        self.file
            .read_exact(bytes)
            .context("failed to read GGUF data")?;
        self.position = end;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self, max_len: u64, what: &str) -> Result<String> {
        let len = self.u64()?;
        ensure!(
            len <= max_len,
            "GGUF {what} length {len} exceeds limit {max_len}"
        );
        let len =
            usize::try_from(len).with_context(|| format!("GGUF {what} length is too large"))?;
        String::from_utf8(self.vec(len)?).with_context(|| format!("GGUF {what} is not UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone)]
    struct TestTensor<'a> {
        name: &'a str,
        dimensions: &'a [u64],
        tensor_type: u32,
        offset: u64,
        data: Vec<u8>,
    }

    fn temp_dir(label: &str) -> PathBuf {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("qw-gguf-{label}-{}-{id}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }


    fn build_gguf(metadata: &[(&str, MetadataValue)], tensors: &[TestTensor<'_>]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            string(&mut bytes, key);
            match value {
                MetadataValue::Uint32(value) => {
                    bytes.extend_from_slice(&(MetadataType::Uint32 as u32).to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                MetadataValue::Uint16(value) => {
                    bytes.extend_from_slice(&(MetadataType::Uint16 as u32).to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                MetadataValue::Int32(value) => {
                    bytes.extend_from_slice(&(MetadataType::Int32 as u32).to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                MetadataValue::String(value) => {
                    bytes.extend_from_slice(&(MetadataType::String as u32).to_le_bytes());
                    string(&mut bytes, value);
                }
                other => panic!("unsupported test metadata {other:?}"),
            }
        }
        for tensor in tensors {
            string(&mut bytes, tensor.name);
            bytes.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
            for dimension in tensor.dimensions {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&tensor.tensor_type.to_le_bytes());
            bytes.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let alignment = metadata
            .iter()
            .find_map(|(key, value)| {
                (*key == "general.alignment").then(|| value.as_u64().unwrap() as usize)
            })
            .unwrap_or(DEFAULT_ALIGNMENT as usize);
        bytes.resize(bytes.len().div_ceil(alignment) * alignment, 0);
        for tensor in tensors {
            let start = bytes.len() + tensor.offset as usize;
            if bytes.len() < start {
                bytes.resize(start, 0);
            }
            bytes.extend_from_slice(&tensor.data);
        }
        bytes
    }

    fn write_valid_file(path: &Path, name: &str) {
        let tensor = TestTensor {
            name,
            dimensions: &[32],
            tensor_type: 8,
            offset: 0,
            data: vec![0; 34],
        };
        fs::write(
            path,
            build_gguf(
                &[
                    ("general.alignment", MetadataValue::Uint32(32)),
                    (
                        "general.architecture",
                        MetadataValue::String("qwen3next".into()),
                    ),
                ],
                &[tensor],
            ),
        )
        .unwrap();
    }

    #[test]
    fn parses_v3_inventory_and_exact_tensor_ranges() {
        let directory = temp_dir("valid");
        let path = directory.join("model.gguf");
        write_valid_file(&path, "blk.0.attn_q.weight");
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(file.alignment, 32);
        assert_eq!(file.tensors.len(), 1);
        let tensor = &file.tensors[0];
        assert_eq!(tensor.name, "blk.0.attn_q.weight");
        assert_eq!(tensor.dimensions, [32]);
        assert_eq!(tensor.tensor_type.id(), 8);
        assert_eq!(tensor.byte_len, 34);
        assert_eq!(tensor.absolute_offset, file.data_offset);
        assert_eq!(file.file_len, file.data_offset + 34);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_magic_version_boolean_and_nested_array_violations() {
        let directory = temp_dir("metadata-malformed");
        let path = directory.join("bad.gguf");

        let mut bytes = build_gguf(&[], &[]);
        bytes[0] = b'X';
        fs::write(&path, &bytes).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("magic")
        );

        bytes[0..4].copy_from_slice(&GGUF_MAGIC);
        bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("version")
        );

        let mut invalid_bool = Vec::new();
        invalid_bool.extend_from_slice(&GGUF_MAGIC);
        invalid_bool.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        invalid_bool.extend_from_slice(&0_u64.to_le_bytes());
        invalid_bool.extend_from_slice(&1_u64.to_le_bytes());
        string(&mut invalid_bool, "general.flag");
        invalid_bool.extend_from_slice(&(MetadataType::Bool as u32).to_le_bytes());
        invalid_bool.push(2);
        fs::write(&path, invalid_bool).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("boolean")
        );

        let mut nested = Vec::new();
        nested.extend_from_slice(&GGUF_MAGIC);
        nested.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        nested.extend_from_slice(&0_u64.to_le_bytes());
        nested.extend_from_slice(&1_u64.to_le_bytes());
        string(&mut nested, "general.nested");
        nested.extend_from_slice(&(MetadataType::Array as u32).to_le_bytes());
        for _ in 0..=GgufLimits::default().max_array_depth {
            nested.extend_from_slice(&(MetadataType::Array as u32).to_le_bytes());
            nested.extend_from_slice(&1_u64.to_le_bytes());
        }
        nested.extend_from_slice(&(MetadataType::Uint8 as u32).to_le_bytes());
        nested.extend_from_slice(&1_u64.to_le_bytes());
        nested.push(0);
        fs::write(&path, nested).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("nesting")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn enforces_declared_count_and_length_limits_before_allocating() {
        let directory = temp_dir("bounds");
        let path = directory.join("bad.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(GgufLimits::default().max_tensor_count + 1).to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("tensor count")
        );

        bytes[8..16].copy_from_slice(&0_u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&66_000_u64.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("key length")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_rank_dimension_alignment_overflow_overlap_and_file_escape() {
        let directory = temp_dir("tensor-malformed");
        let path = directory.join("bad.gguf");
        let base_metadata = [("general.alignment", MetadataValue::Uint32(32))];

        let zero = TestTensor {
            name: "zero",
            dimensions: &[0],
            tensor_type: 0,
            offset: 0,
            data: vec![],
        };
        fs::write(&path, build_gguf(&base_metadata, &[zero])).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("zero dimension")
        );

        let bad_block = TestTensor {
            name: "bad_block",
            dimensions: &[31],
            tensor_type: 8,
            offset: 0,
            data: vec![0; 34],
        };
        fs::write(&path, build_gguf(&base_metadata, &[bad_block])).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("not divisible")
        );

        let unaligned = TestTensor {
            name: "unaligned",
            dimensions: &[1],
            tensor_type: 0,
            offset: 1,
            data: vec![0; 4],
        };
        fs::write(&path, build_gguf(&base_metadata, &[unaligned])).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("not aligned")
        );

        let huge = TestTensor {
            name: "huge",
            dimensions: &[u64::MAX, 2],
            tensor_type: 0,
            offset: 0,
            data: vec![],
        };
        fs::write(&path, build_gguf(&base_metadata, &[huge])).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );

        let first = TestTensor {
            name: "first",
            dimensions: &[16],
            tensor_type: 0,
            offset: 0,
            data: vec![0; 64],
        };
        let second = TestTensor {
            name: "second",
            dimensions: &[16],
            tensor_type: 0,
            offset: 32,
            data: vec![0; 64],
        };
        fs::write(&path, build_gguf(&base_metadata, &[first, second])).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );

        let escaped = TestTensor {
            name: "escaped",
            dimensions: &[16],
            tensor_type: 0,
            offset: 0,
            data: vec![0; 4],
        };
        fs::write(&path, build_gguf(&base_metadata, &[escaped])).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("exceeds file")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn exact_plan_rejects_each_descriptor_mismatch_class() {
        let directory = temp_dir("plan-mismatch");
        let path = directory.join("model.gguf");
        write_valid_file(&path, "tensor.weight");
        let file = GgufFile::open(&path).expect("parse fixture");
        let actual = &file.tensors[0];
        let exact = crate::qwen38_plan::PinnedTensorDescriptor {
            name: "tensor.weight",
            dimensions: &[32],
            qtype: 8,
            relative_offset: actual.relative_offset,
            byte_len: actual.byte_len,
            data_offset: actual.absolute_offset,
        };
        validate_plan(&file, file.file_len, file.data_offset, &[exact], "fixture")
            .expect("exact descriptor");
        for (descriptor, message) in [
            (
                crate::qwen38_plan::PinnedTensorDescriptor {
                    name: "other.weight",
                    ..exact
                },
                "name",
            ),
            (
                crate::qwen38_plan::PinnedTensorDescriptor {
                    dimensions: &[64],
                    ..exact
                },
                "dimensions",
            ),
            (
                crate::qwen38_plan::PinnedTensorDescriptor { qtype: 12, ..exact },
                "qtype",
            ),
            (
                crate::qwen38_plan::PinnedTensorDescriptor {
                    byte_len: exact.byte_len + 1,
                    ..exact
                },
                "byte range",
            ),
        ] {
            let error = validate_plan(
                &file,
                file.file_len,
                file.data_offset,
                &[descriptor],
                "fixture",
            )
            .expect_err("mismatch must fail")
            .to_string();
            assert!(error.contains(message), "{error}");
        }
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires the complete pinned Qwen3.8 27B GGUF pair"]
    fn real_pinned_pair_matches_exact_inventory_and_mtp_geometry() {
        let pair = PinnedGgufPair::open().expect("open pinned GGUF pair");
        assert_eq!(pair.target.tensors.len(), 866);
        assert_eq!(pair.mtp.tensors.len(), 18);
        assert_eq!(pair.target.tensors[2].dimensions, [5_120, 248_320]);
        assert_eq!(pair.target.tensors[0].dimensions, [5_120, 248_320]);
        assert_eq!(pair.mtp.tensors[13].dimensions, [10_240, 5_120]);
        assert_eq!(pair.mtp.tensors[7].dimensions, [5_120, 12_288]);
    }
}
