// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");

//! Lossless load-time GGML-to-MLX affine repacking.
//!
//! Codes, effective scales, and biases are produced directly from GGUF blocks.
//! No dense weight plane exists at any point. Q6_K retains the direct packed
//! kernel because its two independent 16-element scales cannot in general be
//! represented by MLX's smallest supported affine group (32 elements).

use std::ops::Range;
use std::time::{Duration, Instant};

use cxx::UniquePtr;
use thiserror::Error;

use crate::ggml::{GgmlDispatchStats, GgmlKernelPath, GgmlQType, GgmlQuantError};
use crate::{MlxArray, dtype};

const GROUP_SIZE: usize = 32;
const RELEASE_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const IQ4_VALUES: [i16; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Debug, Error)]
pub enum GgmlAffineError {
    #[error(transparent)]
    Quant(#[from] GgmlQuantError),
    #[error("GGML qtype {0:?} has no exact MLX affine representation")]
    NotRepresentable(GgmlQType),
    #[error("GGML affine repack dimensions or allocation size overflow")]
    Overflow,
    #[error("GGML affine repack payload has {actual} bytes; expected {expected}")]
    ByteLength { expected: usize, actual: usize },
    #[error("GGML affine coefficient range {minimum}..={maximum} exceeds UINT8")]
    CoefficientRange { minimum: i16, maximum: i16 },
    #[error("GGML affine repack produced an invalid MLX plane")]
    InvalidPlane,
    #[error("GGML affine row selection must contain one to three valid ranges")]
    InvalidRowSelection,
    #[error("GGML affine input shape or dtype is invalid")]
    InvalidInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgmlAffineTranscodeStats {
    pub source_bytes: usize,
    pub resident_bytes: usize,
    pub peak_active_bytes: usize,
    pub peak_transient_bytes: usize,
    pub source_release_chunk_bytes: usize,
    pub elapsed: Duration,
}

struct AffinePlanes {
    weight: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
    rows: i32,
    packed_width: i32,
    groups: i32,
}

impl AffinePlanes {
    fn clone_shared(&self) -> Self {
        Self {
            weight: crate::copy(self.weight.as_ref().expect("affine weight plane")),
            scales: crate::copy(self.scales.as_ref().expect("affine scale plane")),
            biases: crate::copy(self.biases.as_ref().expect("affine bias plane")),
            rows: self.rows,
            packed_width: self.packed_width,
            groups: self.groups,
        }
    }

    fn slice_rows(&self, range: Range<usize>) -> Result<Self, GgmlAffineError> {
        let start = i32::try_from(range.start).map_err(|_| GgmlAffineError::Overflow)?;
        let end = i32::try_from(range.end).map_err(|_| GgmlAffineError::Overflow)?;
        Ok(Self {
            weight: crate::slice(
                self.weight.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                &[start, 0],
                &[end, self.packed_width],
            ),
            scales: crate::slice(
                self.scales.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                &[start, 0],
                &[end, self.groups],
            ),
            biases: crate::slice(
                self.biases.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                &[start, 0],
                &[end, self.groups],
            ),
            rows: end - start,
            packed_width: self.packed_width,
            groups: self.groups,
        })
    }
}

pub struct GgmlAffineMatrix {
    planes: AffinePlanes,
    qtype: GgmlQType,
    in_features: i32,
    out_features: i32,
    bits: i32,
    resident_row_bytes: usize,
    stats: GgmlAffineTranscodeStats,
}

impl GgmlAffineMatrix {
    pub fn is_representable(qtype_id: u32) -> bool {
        GgmlQType::try_from(qtype_id)
            .is_ok_and(|qtype| affine_bits(qtype).is_some())
    }

    pub fn from_ggml_bytes(
        bytes: &[u8],
        qtype_id: u32,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, GgmlAffineError> {
        Self::from_ggml_bytes_with_progress(
            bytes,
            qtype_id,
            in_features,
            out_features,
            |_| {},
        )
    }

    pub fn from_ggml_bytes_with_progress(
        bytes: &[u8],
        qtype_id: u32,
        in_features: usize,
        out_features: usize,
        mut release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, GgmlAffineError> {
        let started = Instant::now();
        let qtype = GgmlQType::try_from(qtype_id)?;
        let bits = affine_bits(qtype).ok_or(GgmlAffineError::NotRepresentable(qtype))?;
        if in_features == 0
            || out_features == 0
            || !in_features.is_multiple_of(qtype.block_elements())
            || !in_features.is_multiple_of(GROUP_SIZE)
        {
            return Err(GgmlAffineError::Overflow);
        }
        let source_row_bytes = in_features
            .checked_div(qtype.block_elements())
            .and_then(|blocks| blocks.checked_mul(qtype.block_bytes()))
            .ok_or(GgmlAffineError::Overflow)?;
        let expected = source_row_bytes
            .checked_mul(out_features)
            .ok_or(GgmlAffineError::Overflow)?;
        if bytes.len() != expected {
            return Err(GgmlAffineError::ByteLength {
                expected,
                actual: bytes.len(),
            });
        }
        let packed_row_bytes = in_features
            .checked_mul(bits)
            .and_then(|value| value.checked_div(8))
            .ok_or(GgmlAffineError::Overflow)?;
        let groups_per_row = in_features / GROUP_SIZE;
        let weight_bytes = packed_row_bytes
            .checked_mul(out_features)
            .ok_or(GgmlAffineError::Overflow)?;
        let group_values = groups_per_row
            .checked_mul(out_features)
            .ok_or(GgmlAffineError::Overflow)?;
        let sidecar_bytes = group_values
            .checked_mul(4)
            .ok_or(GgmlAffineError::Overflow)?;
        let resident_bytes = weight_bytes
            .checked_add(sidecar_bytes.checked_mul(2).ok_or(GgmlAffineError::Overflow)?)
            .ok_or(GgmlAffineError::Overflow)?;

        let mut weight = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        weight
            .try_reserve_exact(weight_bytes)
            .map_err(|_| GgmlAffineError::Overflow)?;
        scales
            .try_reserve_exact(group_values)
            .map_err(|_| GgmlAffineError::Overflow)?;
        biases
            .try_reserve_exact(group_values)
            .map_err(|_| GgmlAffineError::Overflow)?;
        let mut released = 0usize;
        for (row_index, row) in bytes.chunks_exact(source_row_bytes).enumerate() {
            transcode_row(qtype, row, &mut weight, &mut scales, &mut biases)?;
            let consumed = (row_index + 1) * source_row_bytes;
            if consumed - released >= RELEASE_CHUNK_BYTES {
                release_source(released..consumed);
                released = consumed;
            }
        }
        if released < bytes.len() {
            release_source(released..bytes.len());
        }
        if weight.len() != weight_bytes
            || scales.len() != group_values
            || biases.len() != group_values
        {
            return Err(GgmlAffineError::InvalidPlane);
        }

        let rows = i32::try_from(out_features).map_err(|_| GgmlAffineError::Overflow)?;
        let packed_width = i32::try_from(packed_row_bytes / 4)
            .map_err(|_| GgmlAffineError::Overflow)?;
        let groups = i32::try_from(groups_per_row).map_err(|_| GgmlAffineError::Overflow)?;
        let weight_array = crate::from_bytes(&weight, &[rows, packed_width], dtype::UINT32);
        validate_plane(&weight_array, dtype::UINT32, &[rows, packed_width], weight_bytes)?;
        crate::eval(weight_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
        drop(weight);
        let scale_array = crate::from_slice_f32(&scales, &[rows, groups]);
        validate_plane(&scale_array, dtype::FLOAT32, &[rows, groups], sidecar_bytes)?;
        crate::eval(scale_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
        drop(scales);
        let bias_array = crate::from_slice_f32(&biases, &[rows, groups]);
        validate_plane(&bias_array, dtype::FLOAT32, &[rows, groups], sidecar_bytes)?;
        crate::eval(bias_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
        drop(biases);

        let peak_active_bytes = resident_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(expected.min(RELEASE_CHUNK_BYTES)))
            .ok_or(GgmlAffineError::Overflow)?;
        let stats = GgmlAffineTranscodeStats {
            source_bytes: expected,
            resident_bytes,
            peak_active_bytes,
            peak_transient_bytes: peak_active_bytes - resident_bytes,
            source_release_chunk_bytes: RELEASE_CHUNK_BYTES,
            elapsed: started.elapsed(),
        };
        Ok(Self {
            planes: AffinePlanes {
                weight: weight_array,
                scales: scale_array,
                biases: bias_array,
                rows,
                packed_width,
                groups,
            },
            qtype,
            in_features: i32::try_from(in_features)
                .map_err(|_| GgmlAffineError::Overflow)?,
            out_features: rows,
            bits: bits as i32,
            resident_row_bytes: packed_row_bytes + groups_per_row * 8,
            stats,
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

    pub const fn bits(&self) -> i32 {
        self.bits
    }

    pub const fn transcode_stats(&self) -> GgmlAffineTranscodeStats {
        self.stats
    }

    pub fn clone_shared(&self) -> Self {
        Self {
            planes: self.planes.clone_shared(),
            qtype: self.qtype,
            in_features: self.in_features,
            out_features: self.out_features,
            bits: self.bits,
            resident_row_bytes: self.resident_row_bytes,
            stats: self.stats,
        }
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        Ok(affine_matmul(input, &self.planes, self.bits))
    }

    pub fn select_rows(
        &self,
        ranges: &[Range<usize>],
    ) -> Result<GgmlAffineRows, GgmlAffineError> {
        if ranges.is_empty() || ranges.len() > 3 {
            return Err(GgmlAffineError::InvalidRowSelection);
        }
        let mut planes = Vec::with_capacity(ranges.len());
        let mut selected_rows = 0usize;
        for range in ranges {
            if range.start >= range.end || range.end > self.out_features() {
                return Err(GgmlAffineError::InvalidRowSelection);
            }
            selected_rows = selected_rows
                .checked_add(range.end - range.start)
                .ok_or(GgmlAffineError::Overflow)?;
            planes.push(self.planes.slice_rows(range.clone())?);
        }
        Ok(GgmlAffineRows {
            planes,
            qtype: self.qtype,
            in_features: self.in_features,
            bits: self.bits,
            selected_rows: i32::try_from(selected_rows)
                .map_err(|_| GgmlAffineError::Overflow)?,
            resident_row_bytes: self.resident_row_bytes,
        })
    }

    pub fn dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlAffineError> {
        affine_dispatch_stats(
            input_rows,
            self.in_features(),
            self.out_features(),
            self.resident_row_bytes,
        )
    }
}

pub struct GgmlAffineRows {
    planes: Vec<AffinePlanes>,
    qtype: GgmlQType,
    in_features: i32,
    bits: i32,
    selected_rows: i32,
    resident_row_bytes: usize,
}

impl GgmlAffineRows {
    pub const fn qtype(&self) -> GgmlQType {
        self.qtype
    }

    pub const fn selected_rows(&self) -> usize {
        self.selected_rows as usize
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        let mut outputs = self
            .planes
            .iter()
            .map(|planes| affine_matmul(input, planes, self.bits));
        let mut output = outputs.next().ok_or(GgmlAffineError::InvalidRowSelection)?;
        for next in outputs {
            output = crate::concatenate(
                output.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                next.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                -1,
            );
        }
        Ok(output)
    }

    pub fn dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlAffineError> {
        affine_dispatch_stats(
            input_rows,
            self.in_features as usize,
            self.selected_rows(),
            self.resident_row_bytes,
        )
    }
}

pub struct GgmlAffineEmbedding {
    matrix: GgmlAffineMatrix,
}

impl GgmlAffineEmbedding {
    pub fn from_ggml_bytes(
        bytes: &[u8],
        qtype_id: u32,
        embedding_dim: usize,
        vocab_size: usize,
    ) -> Result<Self, GgmlAffineError> {
        Ok(Self {
            matrix: GgmlAffineMatrix::from_ggml_bytes(
                bytes,
                qtype_id,
                embedding_dim,
                vocab_size,
            )?,
        })
    }

    pub fn from_ggml_bytes_with_progress(
        bytes: &[u8],
        qtype_id: u32,
        embedding_dim: usize,
        vocab_size: usize,
        release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, GgmlAffineError> {
        Ok(Self {
            matrix: GgmlAffineMatrix::from_ggml_bytes_with_progress(
                bytes,
                qtype_id,
                embedding_dim,
                vocab_size,
                release_source,
            )?,
        })
    }

    pub fn clone_shared(&self) -> Self {
        Self {
            matrix: self.matrix.clone_shared(),
        }
    }

    pub const fn embedding_dim(&self) -> usize {
        self.matrix.in_features()
    }

    pub const fn vocab_size(&self) -> usize {
        self.matrix.out_features()
    }

    pub const fn transcode_stats(&self) -> GgmlAffineTranscodeStats {
        self.matrix.transcode_stats()
    }

    pub fn forward(&self, indices: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        let dtype_code = crate::array_dtype(indices);
        if crate::array_size(indices) == 0
            || !matches!(dtype_code, dtype::INT32 | dtype::UINT32)
        {
            return Err(GgmlAffineError::InvalidInput);
        }
        Ok(unsafe {
            crate::quantized_embedding(
                self.matrix.planes.weight.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                self.matrix.planes.scales.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                self.matrix.planes.biases.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                indices,
                GROUP_SIZE as i32,
                self.matrix.bits,
                "affine",
            )
        })
    }

    pub fn as_linear(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        self.matrix.forward(input)
    }

    pub fn select_linear_rows(
        &self,
        ranges: &[Range<usize>],
    ) -> Result<GgmlAffineRows, GgmlAffineError> {
        self.matrix.select_rows(ranges)
    }
}

fn affine_bits(qtype: GgmlQType) -> Option<usize> {
    match qtype {
        GgmlQType::Q8_0 | GgmlQType::Q3K | GgmlQType::Iq4Nl | GgmlQType::Iq4Xs => Some(8),
        GgmlQType::Q4K => Some(4),
        GgmlQType::Q5K | GgmlQType::Iq3S => Some(5),
        GgmlQType::F32 | GgmlQType::Q6K => None,
    }
}

fn transcode_row(
    qtype: GgmlQType,
    row: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) -> Result<(), GgmlAffineError> {
    for block in row.chunks_exact(qtype.block_bytes()) {
        match qtype {
            GgmlQType::Q8_0 => transcode_q8_0(block, weight, scales, biases),
            GgmlQType::Q3K => transcode_q3_k(block, weight, scales, biases)?,
            GgmlQType::Q4K => transcode_q4_k(block, weight, scales, biases),
            GgmlQType::Q5K => transcode_q5_k(block, weight, scales, biases),
            GgmlQType::Iq4Nl => transcode_iq4_nl(block, weight, scales, biases),
            GgmlQType::Iq3S => transcode_iq3_s(block, weight, scales, biases),
            GgmlQType::Iq4Xs => transcode_iq4_xs(block, weight, scales, biases),
            GgmlQType::F32 | GgmlQType::Q6K => {
                return Err(GgmlAffineError::NotRepresentable(qtype));
            }
        }
    }
    Ok(())
}

fn transcode_q8_0(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) {
    let scale = f16_to_f32(u16_at(block, 0));
    let mut codes = [0u8; 32];
    for (code, &value) in codes.iter_mut().zip(&block[2..]) {
        *code = ((value as i8) as i16 + 128) as u8;
    }
    pack_codes(&codes, 8, weight);
    scales.push(scale);
    biases.push(-128.0 * scale);
}

fn transcode_q4_k(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) {
    let d = f16_to_f32(u16_at(block, 0));
    let dmin = f16_to_f32(u16_at(block, 2));
    for group in 0..8 {
        let (scale, minimum) = scale_min_k4(group, &block[4..16]);
        let mut codes = [0u8; 32];
        for column in 0..32 {
            let packed = block[16 + (group / 2) * 32 + column];
            codes[column] = if group.is_multiple_of(2) {
                packed & 15
            } else {
                packed >> 4
            };
        }
        pack_codes(&codes, 4, weight);
        scales.push(d * f32::from(scale));
        biases.push(-(dmin * f32::from(minimum)));
    }
}

fn transcode_q5_k(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) {
    let d = f16_to_f32(u16_at(block, 0));
    let dmin = f16_to_f32(u16_at(block, 2));
    for group in 0..8 {
        let (scale, minimum) = scale_min_k4(group, &block[4..16]);
        let mut codes = [0u8; 32];
        for column in 0..32 {
            let packed = block[48 + (group / 2) * 32 + column];
            let low = if group.is_multiple_of(2) {
                packed & 15
            } else {
                packed >> 4
            };
            codes[column] =
                low + if block[16 + column] & (1 << group) != 0 { 16 } else { 0 };
        }
        pack_codes(&codes, 5, weight);
        scales.push(d * f32::from(scale));
        biases.push(-(dmin * f32::from(minimum)));
    }
}

fn transcode_q3_k(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) -> Result<(), GgmlAffineError> {
    let d = f16_to_f32(u16_at(block, 108));
    for group32 in 0..8 {
        let mut coefficients = [0i16; 32];
        for column32 in 0..32 {
            let column = group32 * 32 + column32;
            let group16 = column / 16;
            let scale_low = if group16 < 8 {
                block[96 + group16] & 15
            } else {
                block[88 + group16] >> 4
            };
            let scale_high =
                (block[104 + group16 % 4] >> (2 * (group16 / 4))) & 3;
            let scale = i16::from(scale_low | (scale_high << 4)) - 32;
            let packed = block[32 + (column / 128) * 32 + column % 32];
            let low = (packed >> (2 * ((column / 32) % 4))) & 3;
            let high = block[column % 32] & (1 << (column / 32));
            let quant = i16::from(low) - if high == 0 { 4 } else { 0 };
            coefficients[column32] = scale * quant;
        }
        let minimum = *coefficients.iter().min().unwrap();
        let maximum = *coefficients.iter().max().unwrap();
        if maximum - minimum > 255 {
            return Err(GgmlAffineError::CoefficientRange { minimum, maximum });
        }
        let codes = coefficients.map(|value| (value - minimum) as u8);
        pack_codes(&codes, 8, weight);
        scales.push(d);
        biases.push(d * f32::from(minimum));
    }
    Ok(())
}

fn transcode_iq4_nl(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) {
    let scale = f16_to_f32(u16_at(block, 0));
    let mut codes = [0u8; 32];
    for column in 0..16 {
        codes[column] = (IQ4_VALUES[(block[2 + column] & 15) as usize] + 127) as u8;
        codes[column + 16] = (IQ4_VALUES[(block[2 + column] >> 4) as usize] + 127) as u8;
    }
    pack_codes(&codes, 8, weight);
    scales.push(scale);
    biases.push(-127.0 * scale);
}

fn transcode_iq3_s(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) {
    let d = f16_to_f32(u16_at(block, 0));
    for group in 0..8 {
        let packed_scale = block[106 + group / 2];
        let subscale = if group.is_multiple_of(2) {
            packed_scale & 15
        } else {
            packed_scale >> 4
        };
        let scale = d * f32::from(1 + 2 * u16::from(subscale));
        let mut codes = [0u8; 32];
        for column in 0..32 {
            let subblock = column / 8;
            let lane = column % 8;
            let high = (block[66 + group]
                >> (2 * subblock + usize::from(lane >= 4)))
                & 1;
            let grid_index = usize::from(
                block[2 + group * 8 + subblock * 2 + usize::from(lane >= 4)],
            ) | (usize::from(high) << 8);
            let grid = crate::ggml::IQ3S_GRID[grid_index];
            let magnitude = ((grid >> ((lane & 3) * 8)) & 255) as i16;
            let signed = if block[74 + group * 4 + subblock] & (1 << lane) != 0 {
                -magnitude
            } else {
                magnitude
            };
            codes[column] = (signed + 15) as u8;
        }
        pack_codes(&codes, 5, weight);
        scales.push(scale);
        biases.push(-15.0 * scale);
    }
}

fn transcode_iq4_xs(
    block: &[u8],
    weight: &mut Vec<u8>,
    scales: &mut Vec<f32>,
    biases: &mut Vec<f32>,
) {
    let d = f16_to_f32(u16_at(block, 0));
    let scales_high = u16_at(block, 2);
    for group in 0..8 {
        let scales_low = block[4 + group / 2];
        let low = (scales_low >> (4 * (group & 1))) & 15;
        let high = ((scales_high >> (2 * group)) & 3) as u8;
        let scale = d * f32::from(i16::from(low | (high << 4)) - 32);
        let mut codes = [0u8; 32];
        for column in 0..32 {
            let packed = block[8 + group * 16 + column % 16];
            let index = if column < 16 { packed & 15 } else { packed >> 4 };
            codes[column] = (IQ4_VALUES[index as usize] + 127) as u8;
        }
        pack_codes(&codes, 8, weight);
        scales.push(scale);
        biases.push(-127.0 * scale);
    }
}

fn pack_codes(codes: &[u8], bits: usize, output: &mut Vec<u8>) {
    let values_per_pack = if matches!(bits, 3 | 5) {
        8
    } else if bits == 6 {
        4
    } else {
        8 / bits
    };
    for values in codes.chunks_exact(values_per_pack) {
        let mut packed = 0u64;
        for (index, &value) in values.iter().enumerate() {
            debug_assert!(u16::from(value) < (1u16 << bits));
            packed |= u64::from(value) << (index * bits);
        }
        output.extend_from_slice(&packed.to_le_bytes()[..values_per_pack * bits / 8]);
    }
}

fn affine_matmul(input: &MlxArray, planes: &AffinePlanes, bits: i32) -> UniquePtr<MlxArray> {
    let shape = crate::array_shape(input);
    let input_rows = shape[..shape.len() - 1]
        .iter()
        .fold(1i32, |rows, dimension| rows * *dimension);
    if !(2..=3).contains(&input_rows) {
        return affine_matmul_raw(input, planes, bits);
    }
    let width = *shape.last().expect("validated affine input rank");
    let flat = crate::reshape(input, &[input_rows, width]);
    let mut outputs = (0..input_rows).map(|row| {
        let input_row = crate::slice(flat.as_ref().unwrap(), &[row, 0], &[row + 1, width]);
        affine_matmul_raw(input_row.as_ref().unwrap(), planes, bits)
    });
    let mut output = outputs.next().expect("positive affine input rows");
    for next in outputs {
        output = crate::concatenate(output.as_ref().unwrap(), next.as_ref().unwrap(), 0);
    }
    let mut output_shape = shape;
    *output_shape.last_mut().unwrap() = planes.rows;
    crate::reshape(output.as_ref().unwrap(), &output_shape)
}

fn affine_matmul_raw(
    input: &MlxArray,
    planes: &AffinePlanes,
    bits: i32,
) -> UniquePtr<MlxArray> {
    unsafe {
        crate::quantized_matmul(
            input,
            planes.weight.as_ref().expect("validated affine weight"),
            planes.scales.as_ref().expect("validated affine scales"),
            planes.biases.as_ref().expect("validated affine biases"),
            true,
            GROUP_SIZE as i32,
            bits,
            "affine",
        )
    }
}

fn validate_input(input: &MlxArray, in_features: i32) -> Result<(), GgmlAffineError> {
    let shape = crate::array_shape(input);
    let dtype_code = crate::array_dtype(input);
    if shape.is_empty()
        || shape.last().copied() != Some(in_features)
        || !matches!(dtype_code, dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16)
    {
        return Err(GgmlAffineError::InvalidInput);
    }
    Ok(())
}

fn validate_plane(
    plane: &UniquePtr<MlxArray>,
    expected_dtype: i32,
    expected_shape: &[i32],
    expected_bytes: usize,
) -> Result<(), GgmlAffineError> {
    let plane = plane.as_ref().ok_or(GgmlAffineError::InvalidPlane)?;
    if crate::array_dtype(plane) != expected_dtype
        || crate::array_shape(plane) != expected_shape
        || crate::array_nbytes(plane) != expected_bytes
    {
        return Err(GgmlAffineError::InvalidPlane);
    }
    Ok(())
}

fn affine_dispatch_stats(
    input_rows: usize,
    in_features: usize,
    output_rows: usize,
    resident_row_bytes: usize,
) -> Result<GgmlDispatchStats, GgmlAffineError> {
    if input_rows == 0 || output_rows == 0 {
        return Err(GgmlAffineError::InvalidInput);
    }
    let qmv_passes = if (2..=3).contains(&input_rows) {
        input_rows
    } else {
        1
    };
    Ok(GgmlDispatchStats {
        path: if input_rows < 4 {
            GgmlKernelPath::AffineQmv
        } else {
            GgmlKernelPath::AffineQmm
        },
        packed_bytes_read: resident_row_bytes
            .checked_mul(output_rows)
            .and_then(|bytes| bytes.checked_mul(qmv_passes))
            .ok_or(GgmlAffineError::Overflow)?,
        activation_bytes_read: input_rows
            .checked_mul(in_features)
            .and_then(|values| values.checked_mul(4))
            .ok_or(GgmlAffineError::Overflow)?,
        output_bytes_written: input_rows
            .checked_mul(output_rows)
            .and_then(|values| values.checked_mul(4))
            .ok_or(GgmlAffineError::Overflow)?,
        floating_point_operations: input_rows
            .checked_mul(output_rows)
            .and_then(|values| values.checked_mul(in_features))
            .and_then(|fmas| fmas.checked_mul(2))
            .ok_or(GgmlAffineError::Overflow)?,
        threadgroup_bytes: 0,
        workspace_bytes: 0,
    })
}

fn scale_min_k4(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 31;
    let mut mantissa = (bits & 1023) as u32;
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut unbiased = -14i32;
            while mantissa & 1024 == 0 {
                mantissa <<= 1;
                unbiased -= 1;
            }
            mantissa &= 1023;
            sign | (((unbiased + 127) as u32) << 23) | (mantissa << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | (((u32::from(exponent) + 112) & 255) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_half(block: &mut [u8], offset: usize, bits: u16) {
        block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
    }

    fn unpack_codes(packed: &[u8], bits: usize, count: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(count);
        let values_per_pack = if matches!(bits, 3 | 5) {
            8
        } else if bits == 6 {
            4
        } else {
            8 / bits
        };
        let bytes_per_pack = values_per_pack * bits / 8;
        for bytes in packed.chunks_exact(bytes_per_pack) {
            let mut value = 0u64;
            for (index, &byte) in bytes.iter().enumerate() {
                value |= u64::from(byte) << (index * 8);
            }
            for index in 0..values_per_pack {
                output.push(((value >> (index * bits)) & ((1 << bits) - 1)) as u8);
            }
        }
        output.truncate(count);
        output
    }

    fn transcode_block(qtype: GgmlQType, block: &[u8]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut weight = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        transcode_row(qtype, block, &mut weight, &mut scales, &mut biases).unwrap();
        (weight, scales, biases)
    }

    #[test]
    fn frozen_k_quant_blocks_preserve_codes_scales_mins_and_high_bits() {
        let mut q4 = vec![0u8; 144];
        put_half(&mut q4, 0, 0x3c00);
        put_half(&mut q4, 2, 0x3800);
        q4[4] = 2;
        q4[8] = 3;
        q4[16] = 4;
        let (weight, scales, biases) = transcode_block(GgmlQType::Q4K, &q4);
        assert_eq!(unpack_codes(&weight, 4, 256)[0], 4);
        assert_eq!(scales[0], 2.0);
        assert_eq!(biases[0], -1.5);

        let mut q5 = vec![0u8; 176];
        put_half(&mut q5, 0, 0x3c00);
        q5[4] = 2;
        q5[16] = 1;
        q5[48] = 4;
        let (weight, scales, biases) = transcode_block(GgmlQType::Q5K, &q5);
        assert_eq!(unpack_codes(&weight, 5, 256)[0], 20);
        assert_eq!(scales[0], 2.0);
        assert_eq!(biases[0], 0.0);

        let mut q3 = vec![0u8; 110];
        put_half(&mut q3, 108, 0x3c00);
        q3[96] = 1;
        q3[104] = 2;
        q3[32] = 2;
        let (weight, scales, biases) = transcode_block(GgmlQType::Q3K, &q3);
        let codes = unpack_codes(&weight, 8, 256);
        assert_eq!(scales[0], 1.0);
        assert_eq!(scales[0] * f32::from(codes[0]) + biases[0], -2.0);
        assert_eq!(scales[0] * f32::from(codes[1]) + biases[0], -4.0);
    }

    #[test]
    fn frozen_q8_and_iq_blocks_preserve_signs_tables_and_subscales() {
        let mut q8 = vec![0u8; 34];
        put_half(&mut q8, 0, 0x3c00);
        q8[2] = (-128i8) as u8;
        q8[33] = 127;
        let (weight, scales, biases) = transcode_block(GgmlQType::Q8_0, &q8);
        let codes = unpack_codes(&weight, 8, 32);
        assert_eq!((codes[0], codes[31], scales[0], biases[0]), (0, 255, 1.0, -128.0));

        let mut iq4 = vec![0u8; 18];
        put_half(&mut iq4, 0, 0x3c00);
        iq4[2] = 0xf0;
        let (weight, scales, biases) = transcode_block(GgmlQType::Iq4Nl, &iq4);
        let codes = unpack_codes(&weight, 8, 32);
        assert_eq!((codes[0], codes[16], scales[0], biases[0]), (0, 240, 1.0, -127.0));

        let mut iq3 = vec![0u8; 110];
        put_half(&mut iq3, 0, 0x3c00);
        iq3[74] = 1;
        let (weight, scales, biases) = transcode_block(GgmlQType::Iq3S, &iq3);
        let codes = unpack_codes(&weight, 5, 256);
        assert_eq!((codes[0], codes[1], scales[0], biases[0]), (14, 16, 1.0, -15.0));

        let mut iq4xs = vec![0u8; 136];
        put_half(&mut iq4xs, 0, 0x3c00);
        iq4xs[2] = 2;
        iq4xs[4] = 1;
        iq4xs[8] = 0xf0;
        let (weight, scales, biases) = transcode_block(GgmlQType::Iq4Xs, &iq4xs);
        let codes = unpack_codes(&weight, 8, 256);
        assert_eq!((codes[0], codes[16], scales[0], biases[0]), (0, 240, 1.0, -127.0));
    }

    #[test]
    fn q6_k_two_scale_subgroups_are_not_generally_affine_group32() {
        let minimum = -32i32 * 127;
        let maximum = -32i32 * -128;
        assert!(maximum - minimum > 255);
        assert!(!GgmlAffineMatrix::is_representable(GgmlQType::Q6K.id()));
        assert!(GgmlAffineMatrix::is_representable(GgmlQType::Q5K.id()));
    }

    #[test]
    fn affine_load_boundary_rejects_fallback_lengths_and_overflow() {
        assert!(matches!(
            GgmlAffineMatrix::from_ggml_bytes(&[], GgmlQType::Q6K.id(), 256, 1),
            Err(GgmlAffineError::NotRepresentable(GgmlQType::Q6K))
        ));
        assert!(matches!(
            GgmlAffineMatrix::from_ggml_bytes(&[0; 175], GgmlQType::Q5K.id(), 256, 1),
            Err(GgmlAffineError::ByteLength {
                expected: 176,
                actual: 175
            })
        ));
        assert!(matches!(
            GgmlAffineMatrix::from_ggml_bytes(&[], GgmlQType::Q5K.id(), usize::MAX, 2),
            Err(GgmlAffineError::Overflow)
        ));
    }
}
