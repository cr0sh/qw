// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");

//! Validated, owned execution of original GGUF block bytes on Metal.
//!
//! Construction is the load boundary: qtype, block geometry, dimensions,
//! overflow, payload length, and lookup-table storage are checked before a
//! custom kernel can be launched. Forward calls never decode weights on the
//! host and never materialize a dense weight matrix.

use cxx::UniquePtr;
use thiserror::Error;

use crate::{MlxArray, dtype};

/// GGML qtypes present in the fixed Qwen3.8 target and MTP artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GgmlQType {
    F32 = 0,
    Q8_0 = 8,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq4Xs = 23,
}

impl GgmlQType {
    pub const TARGET_TYPES: [Self; 9] = [
        Self::F32,
        Self::Q8_0,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq4Xs,
    ];

    pub const fn id(self) -> u32 {
        self as u32
    }

    pub const fn block_elements(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::Q8_0 | Self::Iq4Nl => 32,
            Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Iq3S | Self::Iq4Xs => 256,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Q8_0 => 34,
            Self::Q3K | Self::Iq3S => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq4Nl => 18,
            Self::Iq4Xs => 136,
        }
    }
}

impl TryFrom<u32> for GgmlQType {
    type Error = GgmlQuantError;

    fn try_from(id: u32) -> Result<Self, Self::Error> {
        match id {
            0 => Ok(Self::F32),
            8 => Ok(Self::Q8_0),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            23 => Ok(Self::Iq4Xs),
            _ => Err(GgmlQuantError::UnsupportedQType(id)),
        }
    }
}

#[derive(Debug, Error)]
pub enum GgmlQuantError {
    #[error("unsupported packed GGML qtype {0}")]
    UnsupportedQType(u32),
    #[error("packed GGML dimensions must be positive")]
    EmptyShape,
    #[error("packed GGML input width {width} is not aligned to qtype {qtype:?} block size {block_elements}")]
    UnalignedWidth {
        qtype: GgmlQType,
        width: usize,
        block_elements: usize,
    },
    #[error("packed GGML dimension or byte-count overflow")]
    Overflow,
    #[error("packed GGML payload has {actual} bytes; expected {expected}")]
    ByteLength { expected: usize, actual: usize },
    #[error("packed GGML input shape is incompatible with matrix width {expected_width}")]
    InputShape { expected_width: usize },
    #[error("packed GGML input dtype {0} is not a supported floating dtype")]
    InputDType(i32),
    #[error("packed GGML embedding indices dtype {0} is not INT32 or UINT32")]
    IndexDType(i32),
    #[error("packed GGML embedding indices must not be empty")]
    EmptyIndices,
    #[error("packed GGML row range {start}..{end} is outside 0..{rows}")]
    InvalidRowRange {
        start: usize,
        end: usize,
        rows: usize,
    },
    #[error("packed GGML row selection must contain one to three non-empty ranges")]
    InvalidRowSelection,
    #[error("packed GGML IQ3 lookup table failed integrity validation")]
    InvalidTable,
    #[error("packed GGML Metal launch failed: {0}")]
    Backend(String),
}

/// Shape-selected kernel and its logical memory traffic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgmlKernelPath {
    DecodeM1,
    VerifyM2To4,
    TiledQmm16x8,
    TiledQmm32x8,
    Embedding,
}

/// Logical bytes touched by one packed dispatch. Hardware cache hits are not
/// subtracted. Workspace excludes the returned output allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgmlDispatchStats {
    pub path: GgmlKernelPath,
    pub packed_bytes_read: usize,
    pub activation_bytes_read: usize,
    pub output_bytes_written: usize,
    pub floating_point_operations: usize,
    pub threadgroup_bytes: usize,
    pub workspace_bytes: usize,
}

/// One owned `[out_features, in_features]` matrix in original GGUF bytes.
pub struct GgmlQuantizedMatrix {
    packed: UniquePtr<MlxArray>,
    iq3_grid: UniquePtr<MlxArray>,
    row_ranges: UniquePtr<MlxArray>,
    qtype: GgmlQType,
    in_features: i32,
    out_features: i32,
    row_bytes: usize,
}

impl GgmlQuantizedMatrix {
    pub fn from_bytes(
        bytes: &[u8],
        qtype_id: u32,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, GgmlQuantError> {
        // Unsupported types fail before any MLX/FFI array construction.
        let qtype = GgmlQType::try_from(qtype_id)?;
        if in_features == 0 || out_features == 0 {
            return Err(GgmlQuantError::EmptyShape);
        }
        if !in_features.is_multiple_of(qtype.block_elements()) {
            return Err(GgmlQuantError::UnalignedWidth {
                qtype,
                width: in_features,
                block_elements: qtype.block_elements(),
            });
        }
        let in_features_i32 = i32::try_from(in_features).map_err(|_| GgmlQuantError::Overflow)?;
        let out_features_i32 = i32::try_from(out_features).map_err(|_| GgmlQuantError::Overflow)?;
        let row_bytes = in_features
            .checked_div(qtype.block_elements())
            .and_then(|blocks| blocks.checked_mul(qtype.block_bytes()))
            .ok_or(GgmlQuantError::Overflow)?;
        let expected = row_bytes
            .checked_mul(out_features)
            .ok_or(GgmlQuantError::Overflow)?;
        if bytes.len() != expected {
            return Err(GgmlQuantError::ByteLength {
                expected,
                actual: bytes.len(),
            });
        }
        let row_bytes_i32 = i32::try_from(row_bytes).map_err(|_| GgmlQuantError::Overflow)?;
        let table_values: &[u32] = if qtype == GgmlQType::Iq3S {
            validate_iq3_grid()?;
            &IQ3S_GRID
        } else {
            &[0; 8]
        };
        let packed = crate::from_bytes(
            bytes,
            &[out_features_i32, row_bytes_i32],
            dtype::UINT8,
        );
        let iq3_grid = crate::from_slice_u32(table_values, &[table_values.len() as i32]);
        let (row_ranges, selected_rows) =
            make_row_ranges(&[0..out_features], out_features)?;
        debug_assert_eq!(selected_rows, out_features_i32);
        validate_device_buffers(
            &packed,
            &iq3_grid,
            qtype,
            expected,
            out_features_i32,
            row_bytes_i32,
        )?;
        crate::eval(packed.as_ref().ok_or(GgmlQuantError::InvalidTable)?);
        crate::eval(iq3_grid.as_ref().ok_or(GgmlQuantError::InvalidTable)?);

        Ok(Self {
            packed,
            iq3_grid,
            row_ranges,
            qtype,
            in_features: in_features_i32,
            out_features: out_features_i32,
            row_bytes,
        })
    }

    pub const fn qtype(&self) -> GgmlQType {
        self.qtype
    }

    pub const fn in_features(&self) -> usize {
        self.in_features as usize
    }

    pub const fn out_features(&self) -> usize {
        self.out_features as usize
    }

    pub const fn packed_bytes(&self) -> usize {
        self.row_bytes * self.out_features as usize
    }

    /// Produce an independent handle over the same resident packed bytes and
    /// immutable lookup table. `copy` adds lazy MLX array aliases; it does not
    /// duplicate either payload.
    pub fn clone_shared(&self) -> Self {
        Self {
            packed: crate::copy(
                self.packed
                    .as_ref()
                    .expect("validated packed GGML buffer is always present"),
            ),
            iq3_grid: crate::copy(
                self.iq3_grid
                    .as_ref()
                    .expect("validated GGML lookup table is always present"),
            ),
            row_ranges: crate::copy(
                self.row_ranges
                    .as_ref()
                    .expect("validated GGML row ranges are always present"),
            ),
            qtype: self.qtype,
            in_features: self.in_features,
            out_features: self.out_features,
            row_bytes: self.row_bytes,
        }
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlQuantError> {
        launch_matmul(
            input,
            self.packed.as_ref().ok_or(GgmlQuantError::InvalidTable)?,
            self.row_ranges
                .as_ref()
                .ok_or(GgmlQuantError::InvalidTable)?,
            self.iq3_grid.as_ref().ok_or(GgmlQuantError::InvalidTable)?,
            self.qtype,
            self.in_features,
            self.out_features,
            self.out_features,
        )
    }

    /// Build a zero-copy projection view over up to three ordered packed row
    /// ranges. The view owns only MLX aliases and a tiny immutable range table.
    pub fn select_rows(
        &self,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<GgmlQuantizedRows, GgmlQuantError> {
        let (row_ranges, selected_rows) =
            make_row_ranges(ranges, self.out_features())?;
        Ok(GgmlQuantizedRows {
            packed: crate::copy(
                self.packed
                    .as_ref()
                    .expect("validated packed GGML buffer is always present"),
            ),
            iq3_grid: crate::copy(
                self.iq3_grid
                    .as_ref()
                    .expect("validated GGML lookup table is always present"),
            ),
            row_ranges,
            qtype: self.qtype,
            in_features: self.in_features,
            out_features: self.out_features,
            selected_rows,
            row_bytes: self.row_bytes,
        })
    }

    pub fn dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlQuantError> {
        packed_dispatch_stats(
            input_rows,
            self.in_features(),
            self.out_features(),
            self.row_bytes,
        )
    }
}

/// Zero-copy ordered row selection over one resident packed matrix.
pub struct GgmlQuantizedRows {
    packed: UniquePtr<MlxArray>,
    iq3_grid: UniquePtr<MlxArray>,
    row_ranges: UniquePtr<MlxArray>,
    qtype: GgmlQType,
    in_features: i32,
    out_features: i32,
    selected_rows: i32,
    row_bytes: usize,
}

impl GgmlQuantizedRows {
    pub const fn qtype(&self) -> GgmlQType {
        self.qtype
    }

    pub const fn in_features(&self) -> usize {
        self.in_features as usize
    }

    pub const fn selected_rows(&self) -> usize {
        self.selected_rows as usize
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlQuantError> {
        launch_matmul(
            input,
            self.packed.as_ref().ok_or(GgmlQuantError::InvalidTable)?,
            self.row_ranges
                .as_ref()
                .ok_or(GgmlQuantError::InvalidTable)?,
            self.iq3_grid.as_ref().ok_or(GgmlQuantError::InvalidTable)?,
            self.qtype,
            self.in_features,
            self.out_features,
            self.selected_rows,
        )
    }

    pub fn dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlQuantError> {
        packed_dispatch_stats(
            input_rows,
            self.in_features(),
            self.selected_rows(),
            self.row_bytes,
        )
    }
}

/// Embedding facade over the same owned packed matrix and arithmetic.
pub struct GgmlQuantizedEmbedding {
    matrix: GgmlQuantizedMatrix,
}

impl GgmlQuantizedEmbedding {
    pub fn from_bytes(
        bytes: &[u8],
        qtype_id: u32,
        embedding_dim: usize,
        vocab_size: usize,
    ) -> Result<Self, GgmlQuantError> {
        Ok(Self {
            matrix: GgmlQuantizedMatrix::from_bytes(
                bytes,
                qtype_id,
                embedding_dim,
                vocab_size,
            )?,
        })
    }

    pub const fn qtype(&self) -> GgmlQType {
        self.matrix.qtype()
    }

    pub const fn embedding_dim(&self) -> usize {
        self.matrix.in_features()
    }

    pub const fn vocab_size(&self) -> usize {
        self.matrix.out_features()
    }

    /// Produce an independent embedding handle sharing the resident packed
    /// matrix and lookup table.
    pub fn clone_shared(&self) -> Self {
        Self {
            matrix: self.matrix.clone_shared(),
        }
    }

    pub fn forward(&self, indices: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlQuantError> {
        if crate::array_size(indices) == 0 {
            return Err(GgmlQuantError::EmptyIndices);
        }
        let index_dtype = crate::array_dtype(indices);
        if !matches!(index_dtype, dtype::INT32 | dtype::UINT32) {
            return Err(GgmlQuantError::IndexDType(index_dtype));
        }
        crate::ggml_packed_embedding(
            indices,
            self.matrix
                .packed
                .as_ref()
                .ok_or(GgmlQuantError::InvalidTable)?,
            self.matrix
                .iq3_grid
                .as_ref()
                .ok_or(GgmlQuantError::InvalidTable)?,
            self.qtype().id() as i32,
            self.matrix.in_features,
            self.matrix.out_features,
        )
        .map_err(|error| GgmlQuantError::Backend(error.what().to_owned()))
    }

    pub fn as_linear(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlQuantError> {
        self.matrix.forward(input)
    }

    /// Build a zero-copy linear projection over ordered vocabulary row ranges.
    pub fn select_linear_rows(
        &self,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<GgmlQuantizedRows, GgmlQuantError> {
        self.matrix.select_rows(ranges)
    }

    pub fn dispatch_stats(&self, selected_rows: usize) -> Result<GgmlDispatchStats, GgmlQuantError> {
        if selected_rows == 0 {
            return Err(GgmlQuantError::EmptyIndices);
        }
        Ok(GgmlDispatchStats {
            path: GgmlKernelPath::Embedding,
            packed_bytes_read: self
                .matrix
                .row_bytes
                .checked_mul(selected_rows)
                .ok_or(GgmlQuantError::Overflow)?,
            activation_bytes_read: 0,
            output_bytes_written: selected_rows
                .checked_mul(self.embedding_dim())
                .and_then(|n| n.checked_mul(4))
                .ok_or(GgmlQuantError::Overflow)?,
            floating_point_operations: 0,
            threadgroup_bytes: 0,
            workspace_bytes: 0,
        })
    }
}

fn launch_matmul(
    input: &MlxArray,
    packed: &MlxArray,
    row_ranges: &MlxArray,
    iq3_grid: &MlxArray,
    qtype: GgmlQType,
    in_features: i32,
    out_features: i32,
    selected_rows: i32,
) -> Result<UniquePtr<MlxArray>, GgmlQuantError> {
    let shape = crate::array_shape(input);
    if shape.is_empty() || shape.last().copied() != Some(in_features) {
        return Err(GgmlQuantError::InputShape {
            expected_width: in_features as usize,
        });
    }
    let input_rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, dimension| {
            let dimension =
                usize::try_from(*dimension).map_err(|_| GgmlQuantError::InputShape {
                    expected_width: in_features as usize,
                })?;
            rows.checked_mul(dimension).ok_or(GgmlQuantError::Overflow)
        })?;
    if input_rows == 0 {
        return Err(GgmlQuantError::InputShape {
            expected_width: in_features as usize,
        });
    }
    let input_rows = i32::try_from(input_rows).map_err(|_| GgmlQuantError::Overflow)?;
    let input_dtype = crate::array_dtype(input);
    if !matches!(input_dtype, dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16) {
        return Err(GgmlQuantError::InputDType(input_dtype));
    }
    crate::ggml_packed_matmul(
        input,
        packed,
        row_ranges,
        iq3_grid,
        qtype.id() as i32,
        in_features,
        out_features,
        selected_rows,
        input_rows,
    )
    .map_err(|error| GgmlQuantError::Backend(error.what().to_owned()))
}

fn make_row_ranges(
    ranges: &[std::ops::Range<usize>],
    total_rows: usize,
) -> Result<(UniquePtr<MlxArray>, i32), GgmlQuantError> {
    if ranges.is_empty() || ranges.len() > 3 {
        return Err(GgmlQuantError::InvalidRowSelection);
    }
    let mut selected_rows = 0usize;
    let mut flattened = Vec::with_capacity(ranges.len() * 2);
    for range in ranges {
        if range.start >= range.end || range.end > total_rows {
            return Err(GgmlQuantError::InvalidRowRange {
                start: range.start,
                end: range.end,
                rows: total_rows,
            });
        }
        let count = range.end - range.start;
        selected_rows = selected_rows
            .checked_add(count)
            .ok_or(GgmlQuantError::Overflow)?;
        flattened.push(u32::try_from(range.start).map_err(|_| GgmlQuantError::Overflow)?);
        flattened.push(u32::try_from(count).map_err(|_| GgmlQuantError::Overflow)?);
    }
    let selected_rows = i32::try_from(selected_rows).map_err(|_| GgmlQuantError::Overflow)?;
    let range_count = i32::try_from(ranges.len()).map_err(|_| GgmlQuantError::Overflow)?;
    let array = crate::from_slice_u32(&flattened, &[range_count, 2]);
    let reference = array.as_ref().ok_or(GgmlQuantError::InvalidRowSelection)?;
    if crate::array_dtype(reference) != dtype::UINT32
        || crate::array_shape(reference) != [range_count, 2]
        || crate::array_nbytes(reference) != flattened.len() * 4
    {
        return Err(GgmlQuantError::InvalidRowSelection);
    }
    crate::eval(reference);
    Ok((array, selected_rows))
}

fn packed_dispatch_stats(
    input_rows: usize,
    in_features: usize,
    selected_rows: usize,
    row_bytes: usize,
) -> Result<GgmlDispatchStats, GgmlQuantError> {
    if input_rows == 0 || selected_rows == 0 {
        return Err(GgmlQuantError::EmptyShape);
    }
    let (path, m_tiles, n_tiles, threadgroup_bytes) = if input_rows == 1 {
        (GgmlKernelPath::DecodeM1, 1usize, selected_rows, 0usize)
    } else if input_rows <= 4 {
        (
            GgmlKernelPath::VerifyM2To4,
            1usize,
            selected_rows,
            0usize,
        )
    } else if input_rows <= 512 {
        (
            GgmlKernelPath::TiledQmm16x8,
            input_rows.div_ceil(16),
            selected_rows.div_ceil(8),
            16 * 256 * 4,
        )
    } else {
        (
            GgmlKernelPath::TiledQmm32x8,
            input_rows.div_ceil(32),
            selected_rows.div_ceil(8),
            32 * 256 * 4,
        )
    };
    let packed_bytes_read = row_bytes
        .checked_mul(selected_rows)
        .and_then(|bytes| bytes.checked_mul(m_tiles))
        .ok_or(GgmlQuantError::Overflow)?;
    let activation_bytes_read = input_rows
        .checked_mul(in_features)
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|bytes| bytes.checked_mul(n_tiles))
        .ok_or(GgmlQuantError::Overflow)?;
    let output_bytes_written = input_rows
        .checked_mul(selected_rows)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or(GgmlQuantError::Overflow)?;
    let floating_point_operations = input_rows
        .checked_mul(selected_rows)
        .and_then(|elements| elements.checked_mul(in_features))
        .and_then(|fmas| fmas.checked_mul(2))
        .ok_or(GgmlQuantError::Overflow)?;
    Ok(GgmlDispatchStats {
        path,
        packed_bytes_read,
        activation_bytes_read,
        output_bytes_written,
        floating_point_operations,
        threadgroup_bytes,
        workspace_bytes: 0,
    })
}

fn validate_device_buffers(
    packed: &UniquePtr<MlxArray>,
    table: &UniquePtr<MlxArray>,
    qtype: GgmlQType,
    expected_bytes: usize,
    out_features: i32,
    row_bytes: i32,
) -> Result<(), GgmlQuantError> {
    let packed = packed.as_ref().ok_or(GgmlQuantError::InvalidTable)?;
    if crate::array_dtype(packed) != dtype::UINT8
        || crate::array_nbytes(packed) != expected_bytes
        || crate::array_shape(packed) != [out_features, row_bytes]
    {
        return Err(GgmlQuantError::ByteLength {
            expected: expected_bytes,
            actual: crate::array_nbytes(packed),
        });
    }
    let table = table.as_ref().ok_or(GgmlQuantError::InvalidTable)?;
    let expected_table_len = if qtype == GgmlQType::Iq3S { 512 } else { 8 };
    if crate::array_dtype(table) != dtype::UINT32
        || crate::array_shape(table) != [expected_table_len]
        || crate::array_nbytes(table) != expected_table_len as usize * 4
    {
        return Err(GgmlQuantError::InvalidTable);
    }
    Ok(())
}

fn validate_iq3_grid() -> Result<(), GgmlQuantError> {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in IQ3S_GRID {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    if hash != 0xfa37_020c_25b4_4829 {
        return Err(GgmlQuantError::InvalidTable);
    }
    Ok(())
}

// Canonical ggml IQ3_S codebook (ggml-common.h `iq3s_grid`).
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

#[cfg(test)]
#[path = "ggml_tests.rs"]
mod tests;
