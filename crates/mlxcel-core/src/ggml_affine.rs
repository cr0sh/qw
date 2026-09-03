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
use crate::qwen38_q6::f32_to_f16_bits;
use crate::{MlxArray, Qwen38Q6DualMatrix, dtype};

const GROUP_SIZE: usize = 32;
const F16_PREFILL_MIN_ROWS: i32 = 128;
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
    #[error("pinned Qwen3.8 affine M2/M3/M4 Metal launch failed: {0}")]
    Backend(String),
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
    // F32-primary matrices retain evaluated F16 mirrors for high-M BlockMMA.
    // Q5/IQ3S matrices store F16 coefficients as their sole representation.
    scales_f16: Option<UniquePtr<MlxArray>>,
    biases_f16: Option<UniquePtr<MlxArray>>,
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
            scales_f16: self.scales_f16.as_ref().map(|plane| {
                crate::copy(plane.as_ref().expect("affine FP16 scale plane"))
            }),
            biases_f16: self.biases_f16.as_ref().map(|plane| {
                crate::copy(plane.as_ref().expect("affine FP16 bias plane"))
            }),
            rows: self.rows,
            packed_width: self.packed_width,
            groups: self.groups,
        }
    }

    fn slice_rows(&self, range: Range<usize>) -> Result<Self, GgmlAffineError> {
        let start = i32::try_from(range.start).map_err(|_| GgmlAffineError::Overflow)?;
        let end = i32::try_from(range.end).map_err(|_| GgmlAffineError::Overflow)?;
        let slice_optional = |plane: &Option<UniquePtr<MlxArray>>| {
            plane.as_ref().map(|plane| {
                crate::slice(
                    plane.as_ref().expect("validated affine FP16 plane"),
                    &[start, 0],
                    &[end, self.groups],
                )
            })
        };
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
            scales_f16: slice_optional(&self.scales_f16),
            biases_f16: slice_optional(&self.biases_f16),
            rows: end - start,
            packed_width: self.packed_width,
            groups: self.groups,
        })
    }

    fn primary_is_f16(&self) -> bool {
        crate::array_dtype(self.scales.as_ref().expect("validated affine scale plane"))
            == dtype::FLOAT16
    }

    fn scales_f16(&self) -> Result<&MlxArray, GgmlAffineError> {
        if self.primary_is_f16() {
            return self.scales.as_ref().ok_or(GgmlAffineError::InvalidPlane);
        }
        self.scales_f16
            .as_ref()
            .and_then(UniquePtr::as_ref)
            .ok_or(GgmlAffineError::InvalidPlane)
    }

    fn biases_f16(&self) -> Result<&MlxArray, GgmlAffineError> {
        if self.primary_is_f16() {
            return self.biases.as_ref().ok_or(GgmlAffineError::InvalidPlane);
        }
        self.biases_f16
            .as_ref()
            .and_then(UniquePtr::as_ref)
            .ok_or(GgmlAffineError::InvalidPlane)
    }
}

pub struct GgmlAffineMatrix {
    planes: AffinePlanes,
    in_features: i32,
    out_features: i32,
    bits: i32,
    resident_row_bytes: usize,
    stats: GgmlAffineTranscodeStats,
}

impl GgmlAffineMatrix {
    pub fn from_ggml_bytes(
        bytes: &[u8],
        qtype: GgmlQType,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, GgmlAffineError> {
        Self::from_ggml_bytes_with_progress(bytes, qtype, in_features, out_features, |_| {})
    }

    pub fn from_ggml_bytes_with_progress(
        bytes: &[u8],
        qtype: GgmlQType,
        in_features: usize,
        out_features: usize,
        release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, GgmlAffineError> {
        Self::from_ggml_bytes_with_progress_storage(
            bytes,
            qtype,
            in_features,
            out_features,
            false,
            release_source,
        )
    }

    #[doc(hidden)]
    pub fn from_ggml_bytes_with_progress_f16_sidecars(
        bytes: &[u8],
        qtype: GgmlQType,
        in_features: usize,
        out_features: usize,
        release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, GgmlAffineError> {
        Self::from_ggml_bytes_with_progress_storage(
            bytes,
            qtype,
            in_features,
            out_features,
            true,
            release_source,
        )
    }

    fn from_ggml_bytes_with_progress_storage(
        bytes: &[u8],
        qtype: GgmlQType,
        in_features: usize,
        out_features: usize,
        f16_sidecars: bool,
        mut release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, GgmlAffineError> {
        let started = Instant::now();
        let bits = affine_bits(qtype).ok_or(GgmlAffineError::NotRepresentable(qtype))?;
        if f16_sidecars
            && (!matches!(qtype, GgmlQType::Q5K | GgmlQType::Iq3S)
                || !qwen38_m234_shape(
                    i32::try_from(in_features).map_err(|_| GgmlAffineError::Overflow)?,
                    i32::try_from(out_features).map_err(|_| GgmlAffineError::Overflow)?,
                ))
        {
            return Err(GgmlAffineError::InvalidInput);
        }
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
        let primary_sidecar_item_bytes = if f16_sidecars { 2 } else { 4 };
        let primary_sidecar_bytes = group_values
            .checked_mul(primary_sidecar_item_bytes)
            .ok_or(GgmlAffineError::Overflow)?;
        let f16_sidecar_bytes = group_values
            .checked_mul(2)
            .ok_or(GgmlAffineError::Overflow)?;
        let mirror_sidecar_bytes = if f16_sidecars {
            0
        } else {
            f16_sidecar_bytes
                .checked_mul(2)
                .ok_or(GgmlAffineError::Overflow)?
        };
        let resident_bytes = weight_bytes
            .checked_add(
                primary_sidecar_bytes
                    .checked_mul(2)
                    .ok_or(GgmlAffineError::Overflow)?,
            )
            .and_then(|bytes| bytes.checked_add(mirror_sidecar_bytes))
            .ok_or(GgmlAffineError::Overflow)?;

        let mut weight = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        let mut scales_f16_bytes = Vec::new();
        let mut biases_f16_bytes = Vec::new();
        let mut row_scales = Vec::new();
        let mut row_biases = Vec::new();
        weight
            .try_reserve_exact(weight_bytes)
            .map_err(|_| GgmlAffineError::Overflow)?;
        if f16_sidecars {
            scales_f16_bytes
                .try_reserve_exact(f16_sidecar_bytes)
                .map_err(|_| GgmlAffineError::Overflow)?;
            biases_f16_bytes
                .try_reserve_exact(f16_sidecar_bytes)
                .map_err(|_| GgmlAffineError::Overflow)?;
            row_scales
                .try_reserve_exact(groups_per_row)
                .map_err(|_| GgmlAffineError::Overflow)?;
            row_biases
                .try_reserve_exact(groups_per_row)
                .map_err(|_| GgmlAffineError::Overflow)?;
        } else {
            scales
                .try_reserve_exact(group_values)
                .map_err(|_| GgmlAffineError::Overflow)?;
            biases
                .try_reserve_exact(group_values)
                .map_err(|_| GgmlAffineError::Overflow)?;
        }
        let mut released = 0usize;
        for (row_index, row) in bytes.chunks_exact(source_row_bytes).enumerate() {
            if f16_sidecars {
                transcode_row(
                    qtype,
                    row,
                    &mut weight,
                    &mut row_scales,
                    &mut row_biases,
                )?;
                if row_scales.len() != groups_per_row || row_biases.len() != groups_per_row {
                    return Err(GgmlAffineError::InvalidPlane);
                }
                extend_f16_bytes(&row_scales, &mut scales_f16_bytes);
                extend_f16_bytes(&row_biases, &mut biases_f16_bytes);
                row_scales.clear();
                row_biases.clear();
            } else {
                transcode_row(qtype, row, &mut weight, &mut scales, &mut biases)?;
            }
            let consumed = (row_index + 1) * source_row_bytes;
            if consumed - released >= RELEASE_CHUNK_BYTES {
                release_source(released..consumed);
                released = consumed;
            }
        }
        if released < bytes.len() {
            release_source(released..bytes.len());
        }
        let valid_sidecars = if f16_sidecars {
            scales_f16_bytes.len() == f16_sidecar_bytes
                && biases_f16_bytes.len() == f16_sidecar_bytes
        } else {
            scales.len() == group_values && biases.len() == group_values
        };
        if weight.len() != weight_bytes || !valid_sidecars {
            return Err(GgmlAffineError::InvalidPlane);
        }

        let rows = i32::try_from(out_features).map_err(|_| GgmlAffineError::Overflow)?;
        let packed_width =
            i32::try_from(packed_row_bytes / 4).map_err(|_| GgmlAffineError::Overflow)?;
        let groups = i32::try_from(groups_per_row).map_err(|_| GgmlAffineError::Overflow)?;
        let weight_array = crate::from_bytes(&weight, &[rows, packed_width], dtype::UINT32);
        validate_plane(
            &weight_array,
            dtype::UINT32,
            &[rows, packed_width],
            weight_bytes,
        )?;
        crate::eval(weight_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
        drop(weight);

        let (scale_array, bias_array, scale_array_f16, bias_array_f16) = if f16_sidecars {
            let scale_array =
                crate::from_bytes(&scales_f16_bytes, &[rows, groups], dtype::FLOAT16);
            let bias_array =
                crate::from_bytes(&biases_f16_bytes, &[rows, groups], dtype::FLOAT16);
            validate_plane(
                &scale_array,
                dtype::FLOAT16,
                &[rows, groups],
                f16_sidecar_bytes,
            )?;
            validate_plane(
                &bias_array,
                dtype::FLOAT16,
                &[rows, groups],
                f16_sidecar_bytes,
            )?;
            crate::eval(scale_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
            crate::eval(bias_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
            (scale_array, bias_array, None, None)
        } else {
            let scale_array = crate::from_slice_f32(&scales, &[rows, groups]);
            let bias_array = crate::from_slice_f32(&biases, &[rows, groups]);
            validate_plane(
                &scale_array,
                dtype::FLOAT32,
                &[rows, groups],
                primary_sidecar_bytes,
            )?;
            validate_plane(
                &bias_array,
                dtype::FLOAT32,
                &[rows, groups],
                primary_sidecar_bytes,
            )?;
            crate::eval(scale_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
            crate::eval(bias_array.as_ref().ok_or(GgmlAffineError::InvalidPlane)?);
            let scale_array_f16 = crate::astype(
                scale_array
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
                dtype::FLOAT16,
            );
            let bias_array_f16 = crate::astype(
                bias_array
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
                dtype::FLOAT16,
            );
            validate_plane(
                &scale_array_f16,
                dtype::FLOAT16,
                &[rows, groups],
                f16_sidecar_bytes,
            )?;
            validate_plane(
                &bias_array_f16,
                dtype::FLOAT16,
                &[rows, groups],
                f16_sidecar_bytes,
            )?;
            crate::eval(
                scale_array_f16
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
            );
            crate::eval(
                bias_array_f16
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
            );
            (
                scale_array,
                bias_array,
                Some(scale_array_f16),
                Some(bias_array_f16),
            )
        };
        drop((scales, biases, scales_f16_bytes, biases_f16_bytes));

        let host_sidecar_bytes = if f16_sidecars {
            primary_sidecar_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(groups_per_row * 8))
                .ok_or(GgmlAffineError::Overflow)?
        } else {
            group_values
                .checked_mul(8)
                .ok_or(GgmlAffineError::Overflow)?
        };
        let peak_active_bytes = resident_bytes
            .checked_add(weight_bytes)
            .and_then(|bytes| bytes.checked_add(host_sidecar_bytes))
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
                scales_f16: scale_array_f16,
                biases_f16: bias_array_f16,
                rows,
                packed_width,
                groups,
            },
            in_features: i32::try_from(in_features).map_err(|_| GgmlAffineError::Overflow)?,
            out_features: rows,
            bits: bits as i32,
            resident_row_bytes: packed_row_bytes
                + groups_per_row * (primary_sidecar_item_bytes * 2 + if f16_sidecars { 0 } else { 4 }),
            stats,
        })
    }

    pub const fn in_features(&self) -> usize {
        self.in_features as usize
    }

    pub const fn out_features(&self) -> usize {
        self.out_features as usize
    }

    pub const fn transcode_stats(&self) -> GgmlAffineTranscodeStats {
        self.stats
    }

    #[doc(hidden)]
    pub fn has_f16_sidecars(&self) -> bool {
        crate::array_dtype(
            self.planes
                .scales
                .as_ref()
                .expect("validated affine scale plane"),
        ) == dtype::FLOAT16
    }

    pub fn clone_shared(&self) -> Self {
        Self {
            planes: self.planes.clone_shared(),
            in_features: self.in_features,
            out_features: self.out_features,
            bits: self.bits,
            resident_row_bytes: self.resident_row_bytes,
            stats: self.stats,
        }
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        let input_rows = affine_input_rows(input)?;
        if input_rows >= F16_PREFILL_MIN_ROWS || self.planes.primary_is_f16() {
            return affine_matmul_f16_compute(input, &self.planes, self.bits);
        }
        Ok(affine_matmul(input, &self.planes, self.bits))
    }

    /// Runs exact pinned Qwen3.8 target shapes through a one-pass small-row
    /// kernel. FP32-sidecar affine matrices use the exact M2/M3/M4 kernel;
    /// Q5/IQ3S M1/M3/M4 uses its F16-only mixed-compute kernel.
    pub fn forward_qwen38_m234(
        &self,
        input: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        let contiguous_input = (!crate::array_is_row_contiguous(input))
            .then(|| crate::contiguous(input, false));
        let input = contiguous_input.as_deref().unwrap_or(input);
        let input_rows = affine_input_rows(input)?;
        let pinned = qwen38_m234_shape(self.in_features, self.out_features);
        if self.bits == 5
            && self.has_f16_sidecars()
            && matches!(input_rows, 1 | 3 | 4)
            && pinned
        {
            return qwen38_affine_m234_matmul(input, self, input_rows, true);
        }
        if !(2..=4).contains(&input_rows)
            || crate::array_dtype(input) != dtype::FLOAT32
            || !pinned
        {
            return self.forward(input);
        }
        qwen38_affine_m234_matmul(input, self, input_rows, true)
    }

    /// Runs the exact pinned Qwen3.8 MTP shapes through one affine weight pass
    /// only for M2. Every other shape, dtype, or row count keeps the ordinary
    /// affine dispatcher.
    pub fn forward_qwen38_m2(
        &self,
        input: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        let input_rows = affine_input_rows(input)?;
        if input_rows != 2
            || crate::array_dtype(input) != dtype::FLOAT32
            || !qwen38_m2_shape(self.in_features, self.out_features)
        {
            return self.forward(input);
        }
        qwen38_affine_m234_matmul(input, self, input_rows, true)
    }

    #[cfg(test)]
    pub(crate) fn forward_m234_test_only(
        &self,
        input: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        let input_rows = affine_input_rows(input)?;
        if !(2..=4).contains(&input_rows)
            || crate::array_dtype(input) != dtype::FLOAT32
        {
            return self.forward(input);
        }
        qwen38_affine_m234_matmul(input, self, input_rows, false)
    }

    pub fn select_rows(&self, ranges: &[Range<usize>]) -> Result<GgmlAffineRows, GgmlAffineError> {
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
            in_features: self.in_features,
            bits: self.bits,
            selected_rows: i32::try_from(selected_rows).map_err(|_| GgmlAffineError::Overflow)?,
            resident_row_bytes: self.resident_row_bytes,
        })
    }

    pub fn dispatch_stats(&self, input_rows: usize) -> Result<GgmlDispatchStats, GgmlAffineError> {
        affine_dispatch_stats(
            input_rows,
            self.in_features(),
            self.out_features(),
            self.resident_row_bytes,
        )
    }

    pub fn qwen38_m234_dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlAffineError> {
        let pinned = qwen38_m234_shape(self.in_features, self.out_features);
        let one_pass = pinned
            && ((self.bits == 5
                && self.has_f16_sidecars()
                && matches!(input_rows, 1 | 3 | 4))
                || (2..=4).contains(&input_rows));
        if one_pass {
            affine_dispatch_stats_with(
                input_rows,
                self.in_features(),
                self.out_features(),
                self.resident_row_bytes,
                1,
                GgmlKernelPath::Qwen38AffineM234,
            )
        } else {
            self.dispatch_stats(input_rows)
        }
    }

    pub fn qwen38_m2_dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlAffineError> {
        if input_rows == 2 && qwen38_m2_shape(self.in_features, self.out_features) {
            affine_dispatch_stats_with(
                input_rows,
                self.in_features(),
                self.out_features(),
                self.resident_row_bytes,
                1,
                GgmlKernelPath::Qwen38AffineM234,
            )
        } else {
            self.dispatch_stats(input_rows)
        }
    }

    #[cfg(test)]
    pub(crate) fn m234_dispatch_stats_test_only(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, GgmlAffineError> {
        affine_dispatch_stats_with(
            input_rows,
            self.in_features(),
            self.out_features(),
            self.resident_row_bytes,
            1,
            GgmlKernelPath::Qwen38AffineM234,
        )
    }
}


pub enum Qwen38QkvMatrix {
    Affine(GgmlAffineMatrix),
    Q6(Qwen38Q6DualMatrix),
}

pub struct Qwen38MixedQkvOutput {
    pub query: UniquePtr<MlxArray>,
    pub key: UniquePtr<MlxArray>,
    pub value: UniquePtr<MlxArray>,
}

pub struct Qwen38MixedQkvBundle {
    query: GgmlAffineMatrix,
    key: Qwen38QkvMatrix,
    value: Qwen38QkvMatrix,
}

impl Qwen38MixedQkvBundle {
    pub fn new(
        query: GgmlAffineMatrix,
        key: Qwen38QkvMatrix,
        value: Qwen38QkvMatrix,
    ) -> Result<Self, GgmlAffineError> {
        if (query.in_features, query.out_features) != (5120, 12_288)
            || !matches!(query.bits, 4 | 5 | 8)
            || !Self::valid_kv(&key)
            || !Self::valid_kv(&value)
        {
            return Err(GgmlAffineError::InvalidPlane);
        }
        Ok(Self { query, key, value })
    }

    pub const fn uses_bundled_dispatch(input_rows: i32) -> bool {
        matches!(input_rows, 3 | 4)
    }

    fn valid_kv(matrix: &Qwen38QkvMatrix) -> bool {
        match matrix {
            Qwen38QkvMatrix::Affine(matrix) => {
                (matrix.in_features, matrix.out_features) == (5120, 1024)
                    && matches!(matrix.bits, 4 | 5 | 8)
            }
            Qwen38QkvMatrix::Q6(matrix) => {
                let packed = matrix.packed_ref();
                packed.qtype() == GgmlQType::Q6K
                    && (packed.in_features(), packed.out_features()) == (5120, 1024)
            }
        }
    }

    fn projection_parts<'a>(
        &'a self,
        matrix: &'a Qwen38QkvMatrix,
    ) -> Result<(&'a MlxArray, &'a MlxArray, &'a MlxArray, &'a MlxArray, i32), GgmlAffineError>
    {
        let fallback_weight = self
            .query
            .planes
            .weight
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?;
        let fallback_scale = self
            .query
            .planes
            .scales
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?;
        match matrix {
            Qwen38QkvMatrix::Affine(matrix) => {
                let weight = matrix
                    .planes
                    .weight
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?;
                Ok((
                    weight,
                    matrix
                        .planes
                        .scales
                        .as_ref()
                        .ok_or(GgmlAffineError::InvalidPlane)?,
                    matrix
                        .planes
                        .biases
                        .as_ref()
                        .ok_or(GgmlAffineError::InvalidPlane)?,
                    weight,
                    matrix.bits,
                ))
            }
            Qwen38QkvMatrix::Q6(matrix) => Ok((
                fallback_weight,
                fallback_scale,
                fallback_scale,
                matrix.packed_ref().packed_ref()?,
                14,
            )),
        }
    }

    pub fn forward(&self, input: &MlxArray) -> Result<Qwen38MixedQkvOutput, GgmlAffineError> {
        validate_input(input, 5120)?;
        let input_rows = affine_input_rows(input)?;
        if crate::array_dtype(input) != dtype::FLOAT32 {
            return Err(GgmlAffineError::InvalidInput);
        }
        if input_rows == 4 {
            // The bundled M4 K/V path does not preserve corresponding M1
            // arithmetic. Reuse its exact M3 path plus one decode row.
            let shape = crate::array_shape(input);
            let first_input =
                crate::slice(input, &[0, 0, 0], &[shape[0], 3, shape[2]]);
            let last_input =
                crate::slice(input, &[0, 3, 0], &[shape[0], 4, shape[2]]);
            let first = self.forward(&first_input)?;
            let last = self.forward(&last_input)?;
            return Ok(Qwen38MixedQkvOutput {
                query: crate::concatenate(&first.query, &last.query, 1),
                key: crate::concatenate(&first.key, &last.key, 1),
                value: crate::concatenate(&first.value, &last.value, 1),
            });
        }
        if !Self::uses_bundled_dispatch(input_rows) {
            let forward = |matrix: &Qwen38QkvMatrix| match matrix {
                Qwen38QkvMatrix::Affine(matrix) => matrix.forward(input),
                Qwen38QkvMatrix::Q6(matrix) => matrix
                    .forward(input)
                    .map_err(|error| GgmlAffineError::Backend(error.to_string())),
            };
            return Ok(Qwen38MixedQkvOutput {
                query: self.query.forward_qwen38_m234(input)?,
                key: forward(&self.key)?,
                value: forward(&self.value)?,
            });
        }

        let query_weight = self
            .query
            .planes
            .weight
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?;
        let query_scales = self
            .query
            .planes
            .scales
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?;
        let query_biases = self
            .query
            .planes
            .biases
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?;
        let (key_weight, key_scales, key_biases, key_packed, key_code) =
            self.projection_parts(&self.key)?;
        let (value_weight, value_scales, value_biases, value_packed, value_code) =
            self.projection_parts(&self.value)?;
        let mut outputs = crate::qwen38_mixed_qkv_bundle(
            input,
            query_weight,
            query_scales,
            query_biases,
            self.query.bits,
            key_weight,
            key_scales,
            key_biases,
            key_packed,
            key_code,
            value_weight,
            value_scales,
            value_biases,
            value_packed,
            value_code,
            input_rows,
        )
        .map_err(|error| GgmlAffineError::Backend(error.what().to_owned()))?;
        let output = outputs.pin_mut();
        let query = crate::qwen38_ggml_qkv_take_query(output);
        let output = outputs.pin_mut();
        let key = crate::qwen38_ggml_qkv_take_key(output);
        let output = outputs.pin_mut();
        let value = crate::qwen38_ggml_qkv_take_value(output);
        Ok(Qwen38MixedQkvOutput { query, key, value })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FusionStats {
    pub physical_dispatches: usize,
    pub matrix_bytes_read: usize,
    pub intermediate_bytes_avoided: usize,
    pub workspace_bytes: usize,
    pub hidden_copy_bytes: usize,
}

pub struct Qwen38AffineMlpFusion {
    gate: GgmlAffineMatrix,
    up: GgmlAffineMatrix,
    down: GgmlAffineMatrix,
    gate_m234: bool,
    up_m234: bool,
    down_m234: bool,
    m2_only: bool,
}

impl Qwen38AffineMlpFusion {
    pub fn new(
        gate: GgmlAffineMatrix,
        up: GgmlAffineMatrix,
        down: GgmlAffineMatrix,
        m234: [bool; 3],
    ) -> Result<Self, GgmlAffineError> {
        Self::with_small_row_paths(gate, up, down, m234, false)
    }

    pub fn new_m2(
        gate: GgmlAffineMatrix,
        up: GgmlAffineMatrix,
        down: GgmlAffineMatrix,
    ) -> Result<Self, GgmlAffineError> {
        Self::with_small_row_paths(gate, up, down, [false; 3], true)
    }

    fn with_small_row_paths(
        gate: GgmlAffineMatrix,
        up: GgmlAffineMatrix,
        down: GgmlAffineMatrix,
        m234: [bool; 3],
        m2_only: bool,
    ) -> Result<Self, GgmlAffineError> {
        if (gate.in_features(), gate.out_features()) != (5120, 17_408)
            || (up.in_features(), up.out_features()) != (5120, 17_408)
            || (down.in_features(), down.out_features()) != (17_408, 5120)
        {
            return Err(GgmlAffineError::InvalidPlane);
        }
        Ok(Self {
            gate,
            up,
            down,
            gate_m234: m234[0],
            up_m234: m234[1],
            down_m234: m234[2],
            m2_only,
        })
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, 5120)?;
        let rows = affine_input_rows(input)?;
        // Share the normalized-input cast and keep the MLP interior in F16;
        // the native quantized matmuls still accumulate in F32.
        if rows >= F16_PREFILL_MIN_ROWS {
            let input_f16 = crate::astype(input, dtype::FLOAT16);
            let input_f16 = input_f16
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?;
            let gate = affine_matmul_f16_output(input_f16, &self.gate.planes, self.gate.bits)?;
            let up = affine_matmul_f16_output(input_f16, &self.up.planes, self.up.bits)?;
            let activated = crate::compiled_swiglu_activation(
                gate.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                up.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
            );
            let activated = activated
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?;
            if crate::array_dtype(activated) != dtype::FLOAT16 {
                return Err(GgmlAffineError::InvalidPlane);
            }
            let output_f16 =
                affine_matmul_f16_output(activated, &self.down.planes, self.down.bits)?;
            return Ok(crate::astype(
                output_f16
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
                dtype::FLOAT32,
            ));
        }
        // Only the pinned M=33 BlockMMA shape beats ordinary MLX.
        if rows != 33
            || crate::array_dtype(input) != dtype::FLOAT32
            || !crate::ffi::array_is_row_contiguous(input)
            || self.gate.has_f16_sidecars()
            || self.up.has_f16_sidecars()
            || self.down.has_f16_sidecars()
        {
            let forward = |matrix: &GgmlAffineMatrix, input: &MlxArray, m234| {
                if rows == 2 && self.m2_only {
                    matrix.forward_qwen38_m2(input)
                } else if m234 {
                    matrix.forward_qwen38_m234(input)
                } else {
                    matrix.forward(input)
                }
            };
            let gate = forward(&self.gate, input, self.gate_m234)?;
            let up = forward(&self.up, input, self.up_m234)?;
            let activated = crate::compiled_swiglu_activation(
                gate.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                up.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
            );
            return forward(
                &self.down,
                activated.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                self.down_m234,
            );
        }
        crate::qwen38_affine_mlp_fused(
            input,
            self.gate
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.gate
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.gate
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.gate.bits,
            self.up
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.up
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.up
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.up.bits,
            self.down
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.down
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.down
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.down.bits,
        )
        .map_err(|error| GgmlAffineError::Backend(error.to_string()))
    }

    pub fn dispatch_stats(&self, input_rows: usize) -> Result<Qwen38FusionStats, GgmlAffineError> {
        if input_rows == 0 {
            return Err(GgmlAffineError::InvalidInput);
        }
        let matrix_bytes_read =
            [&self.gate, &self.up, &self.down]
                .into_iter()
                .try_fold(0usize, |total, matrix| {
                    matrix
                        .resident_row_bytes
                        .checked_mul(matrix.out_features())
                        .and_then(|bytes| total.checked_add(bytes))
                        .ok_or(GgmlAffineError::Overflow)
                })?;
        if input_rows >= F16_PREFILL_MIN_ROWS as usize {
            let intermediate_bytes_avoided = input_rows
                .checked_mul(2 * 5120 + 12 * 17_408)
                .ok_or(GgmlAffineError::Overflow)?;
            return Ok(Qwen38FusionStats {
                physical_dispatches: 6,
                matrix_bytes_read,
                intermediate_bytes_avoided,
                workspace_bytes: 0,
                hidden_copy_bytes: 0,
            });
        }
        if input_rows != 33 {
            return Ok(Qwen38FusionStats {
                physical_dispatches: 4,
                matrix_bytes_read,
                intermediate_bytes_avoided: 0,
                workspace_bytes: 0,
                hidden_copy_bytes: 0,
            });
        }
        let split = qwen38_split_k(input_rows, 17_408, 5120);
        let down_workspace = if split > 1 {
            split
                .checked_mul(input_rows)
                .and_then(|n| n.checked_mul(5120))
                .and_then(|n| n.checked_mul(4))
                .ok_or(GgmlAffineError::Overflow)?
        } else {
            0
        };
        let gate_up_bytes = input_rows
            .checked_mul(17_408)
            .and_then(|n| n.checked_mul(8))
            .ok_or(GgmlAffineError::Overflow)?;
        Ok(Qwen38FusionStats {
            physical_dispatches: 2,
            matrix_bytes_read,
            intermediate_bytes_avoided: gate_up_bytes
                .checked_add(down_workspace)
                .ok_or(GgmlAffineError::Overflow)?,
            workspace_bytes: 0,
            hidden_copy_bytes: 0,
        })
    }
}

pub struct Qwen38GdnIngressOutput {
    pub qkv: UniquePtr<MlxArray>,
    pub z: UniquePtr<MlxArray>,
    pub beta: UniquePtr<MlxArray>,
    pub alpha: UniquePtr<MlxArray>,
}

pub struct Qwen38AffineGdnIngressFusion {
    qkv: GgmlAffineMatrix,
    z: GgmlAffineMatrix,
    beta: GgmlAffineMatrix,
    alpha: GgmlAffineMatrix,
    m234: [bool; 4],
}

impl Qwen38AffineGdnIngressFusion {
    pub fn new(
        qkv: GgmlAffineMatrix,
        z: GgmlAffineMatrix,
        beta: GgmlAffineMatrix,
        alpha: GgmlAffineMatrix,
        m234: [bool; 4],
    ) -> Result<Self, GgmlAffineError> {
        if (qkv.in_features(), qkv.out_features()) != (5120, 10_240)
            || (z.in_features(), z.out_features()) != (5120, 6144)
            || (beta.in_features(), beta.out_features()) != (5120, 48)
            || (alpha.in_features(), alpha.out_features()) != (5120, 48)
        {
            return Err(GgmlAffineError::InvalidPlane);
        }
        Ok(Self {
            qkv,
            z,
            beta,
            alpha,
            m234,
        })
    }

    pub fn forward(&self, input: &MlxArray) -> Result<Qwen38GdnIngressOutput, GgmlAffineError> {
        validate_input(input, 5120)?;
        let rows = affine_input_rows(input)?;
        // Only the pinned M=33 and M=128 BlockMMA shapes beat ordinary MLX.
        if !matches!(rows, 33 | 128)
            || crate::array_dtype(input) != dtype::FLOAT32
            || !crate::ffi::array_is_row_contiguous(input)
            || self.qkv.has_f16_sidecars()
            || self.z.has_f16_sidecars()
            || self.beta.has_f16_sidecars()
            || self.alpha.has_f16_sidecars()
        {
            let forward = |matrix: &GgmlAffineMatrix, m234| {
                if m234 {
                    matrix.forward_qwen38_m234(input)
                } else {
                    matrix.forward(input)
                }
            };
            return Ok(Qwen38GdnIngressOutput {
                qkv: forward(&self.qkv, self.m234[0])?,
                z: forward(&self.z, self.m234[1])?,
                beta: forward(&self.beta, self.m234[2])?,
                alpha: forward(&self.alpha, self.m234[3])?,
            });
        }
        let mut outputs = crate::qwen38_affine_gdn_ingress_fused(
            input,
            self.qkv
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.qkv
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.qkv
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.qkv.bits,
            self.z
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.z
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.z
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.z.bits,
            self.beta
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.beta
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.beta
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.beta.bits,
            self.alpha
                .planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.alpha
                .planes
                .scales
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.alpha
                .planes
                .biases
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            self.alpha.bits,
        )
        .map_err(|error| GgmlAffineError::Backend(error.to_string()))?;
        let output = outputs.pin_mut();
        let qkv = crate::qwen38_gdn_take_qkv(output);
        let output = outputs.pin_mut();
        let z = crate::qwen38_gdn_take_z(output);
        let output = outputs.pin_mut();
        let beta = crate::qwen38_gdn_take_beta(output);
        let output = outputs.pin_mut();
        let alpha = crate::qwen38_gdn_take_alpha(output);
        Ok(Qwen38GdnIngressOutput {
            qkv,
            z,
            beta,
            alpha,
        })
    }

    pub fn dispatch_stats(&self, input_rows: usize) -> Result<Qwen38FusionStats, GgmlAffineError> {
        if input_rows == 0 {
            return Err(GgmlAffineError::InvalidInput);
        }
        let matrix_bytes_read = [&self.qkv, &self.z, &self.beta, &self.alpha]
            .into_iter()
            .try_fold(0usize, |total, matrix| {
                matrix
                    .resident_row_bytes
                    .checked_mul(matrix.out_features())
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or(GgmlAffineError::Overflow)
            })?;
        if !matches!(input_rows, 33 | 128) {
            return Ok(Qwen38FusionStats {
                physical_dispatches: 4,
                matrix_bytes_read,
                intermediate_bytes_avoided: 0,
                workspace_bytes: 0,
                hidden_copy_bytes: 0,
            });
        }
        let splits = [
            qwen38_split_k(input_rows, 5120, 10_240),
            qwen38_split_k(input_rows, 5120, 6144),
            qwen38_split_k(input_rows, 5120, 48),
            qwen38_split_k(input_rows, 5120, 48),
        ];
        let widths = [10_240usize, 6144, 48, 48];
        let workspace_bytes = splits
            .into_iter()
            .zip(widths)
            .filter(|(split, _)| *split > 1)
            .try_fold(0usize, |total, (split, width)| {
                split
                    .checked_mul(input_rows)
                    .and_then(|n| n.checked_mul(width))
                    .and_then(|n| n.checked_mul(4))
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or(GgmlAffineError::Overflow)
            })?;
        Ok(Qwen38FusionStats {
            physical_dispatches: if workspace_bytes == 0 { 2 } else { 3 },
            matrix_bytes_read,
            intermediate_bytes_avoided: 0,
            workspace_bytes,
            hidden_copy_bytes: 0,
        })
    }
}

pub struct GgmlAffineRows {
    planes: Vec<AffinePlanes>,
    in_features: i32,
    bits: i32,
    selected_rows: i32,
    resident_row_bytes: usize,
}

impl GgmlAffineRows {
    pub const fn selected_rows(&self) -> usize {
        self.selected_rows as usize
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
        validate_input(input, self.in_features)?;
        let input_rows = affine_input_rows(input)?;
        let use_f16 = input_rows >= F16_PREFILL_MIN_ROWS
            || self
                .planes
                .first()
                .is_some_and(AffinePlanes::primary_is_f16);
        let mut outputs = self.planes.iter().map(|planes| {
            if use_f16 {
                affine_matmul_f16_compute(input, planes, self.bits)
            } else {
                Ok(affine_matmul(input, planes, self.bits))
            }
        });
        let mut output = outputs
            .next()
            .ok_or(GgmlAffineError::InvalidRowSelection)??;
        for next in outputs {
            let next = next?;
            output = crate::concatenate(
                output.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                next.as_ref().ok_or(GgmlAffineError::InvalidPlane)?,
                -1,
            );
        }
        Ok(output)
    }

    pub fn dispatch_stats(&self, input_rows: usize) -> Result<GgmlDispatchStats, GgmlAffineError> {
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
        qtype: GgmlQType,
        embedding_dim: usize,
        vocab_size: usize,
    ) -> Result<Self, GgmlAffineError> {
        Ok(Self {
            matrix: GgmlAffineMatrix::from_ggml_bytes(bytes, qtype, embedding_dim, vocab_size)?,
        })
    }

    pub fn from_ggml_bytes_with_progress(
        bytes: &[u8],
        qtype: GgmlQType,
        embedding_dim: usize,
        vocab_size: usize,
        release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, GgmlAffineError> {
        Ok(Self {
            matrix: GgmlAffineMatrix::from_ggml_bytes_with_progress(
                bytes,
                qtype,
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
        if crate::array_size(indices) == 0 || !matches!(dtype_code, dtype::INT32 | dtype::UINT32) {
            return Err(GgmlAffineError::InvalidInput);
        }
        Ok(unsafe {
            crate::quantized_embedding(
                self.matrix
                    .planes
                    .weight
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
                self.matrix
                    .planes
                    .scales
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
                self.matrix
                    .planes
                    .biases
                    .as_ref()
                    .ok_or(GgmlAffineError::InvalidPlane)?,
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
            codes[column] = low
                + if block[16 + column] & (1 << group) != 0 {
                    16
                } else {
                    0
                };
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
            let scale_high = (block[104 + group16 % 4] >> (2 * (group16 / 4))) & 3;
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
            let high = (block[66 + group] >> (2 * subblock + usize::from(lane >= 4))) & 1;
            let grid_index =
                usize::from(block[2 + group * 8 + subblock * 2 + usize::from(lane >= 4)])
                    | (usize::from(high) << 8);
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
            let index = if column < 16 {
                packed & 15
            } else {
                packed >> 4
            };
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

fn extend_f16_bytes(values: &[f32], output: &mut Vec<u8>) {
    for &value in values {
        output.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
    }
}

fn affine_input_rows(input: &MlxArray) -> Result<i32, GgmlAffineError> {
    crate::array_shape(input)[..crate::array_ndim(input) - 1]
        .iter()
        .try_fold(1i32, |rows, dimension| {
            rows.checked_mul(*dimension)
                .ok_or(GgmlAffineError::Overflow)
        })
}

fn qwen38_m234_shape(in_features: i32, out_features: i32) -> bool {
    matches!(
        (in_features, out_features),
        (5120, 48)
            | (5120, 17408)
            | (17408, 5120)
            | (5120, 10240)
            | (5120, 6144)
            | (6144, 5120)
            | (5120, 12288)
    )
}

fn qwen38_m2_shape(in_features: i32, out_features: i32) -> bool {
    qwen38_m234_shape(in_features, out_features) || (in_features, out_features) == (10_240, 5120)
}

fn qwen38_split_k(input_rows: usize, in_features: usize, out_features: usize) -> usize {
    let m_tiles = input_rows.div_ceil(32);
    let n_tiles = out_features.div_ceil(32);
    let mut split = (512 / (m_tiles * n_tiles)).max(1).min(in_features / 32);
    while split > 1 && !in_features.is_multiple_of(split * 32) {
        split -= 1;
    }
    split
}

fn qwen38_affine_m234_matmul(
    input: &MlxArray,
    matrix: &GgmlAffineMatrix,
    input_rows: i32,
    require_pinned_shape: bool,
) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
    crate::ffi::qwen38_affine_m234_matmul(
        input,
        matrix
            .planes
            .weight
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?,
        matrix
            .planes
            .scales
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?,
        matrix
            .planes
            .biases
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?,
        matrix.bits,
        matrix.in_features,
        matrix.out_features,
        input_rows,
        require_pinned_shape,
    )
    .map_err(|error| GgmlAffineError::Backend(error.to_string()))
}
fn affine_matmul(
    input: &MlxArray,
    planes: &AffinePlanes,
    bits: i32,
) -> UniquePtr<MlxArray> {
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

fn affine_matmul_f16_compute(
    input: &MlxArray,
    planes: &AffinePlanes,
    bits: i32,
) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
    // F16 operands select MLX's half BlockMMA, whose default accumulator is
    // `float`; only the stored matrix result is F16 before this F32 boundary.
    let input_f16 = crate::astype(input, dtype::FLOAT16);
    let output_f16 = affine_matmul_f16_output(
        input_f16
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?,
        planes,
        bits,
    )?;
    Ok(crate::astype(
        output_f16
            .as_ref()
            .ok_or(GgmlAffineError::InvalidPlane)?,
        dtype::FLOAT32,
    ))
}

fn affine_matmul_f16_output(
    input_f16: &MlxArray,
    planes: &AffinePlanes,
    bits: i32,
) -> Result<UniquePtr<MlxArray>, GgmlAffineError> {
    Ok(unsafe {
        crate::quantized_matmul(
            input_f16,
            planes
                .weight
                .as_ref()
                .ok_or(GgmlAffineError::InvalidPlane)?,
            planes.scales_f16()?,
            planes.biases_f16()?,
            true,
            GROUP_SIZE as i32,
            bits,
            "affine",
        )
    })
}

fn validate_input(input: &MlxArray, in_features: i32) -> Result<(), GgmlAffineError> {
    let shape = crate::array_shape(input);
    let dtype_code = crate::array_dtype(input);
    if shape.is_empty()
        || shape.last().copied() != Some(in_features)
        || !matches!(
            dtype_code,
            dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16
        )
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
    let qmv_passes = if (2..=3).contains(&input_rows) {
        input_rows
    } else {
        1
    };
    let path = if input_rows < 4 {
        GgmlKernelPath::AffineQmv
    } else {
        GgmlKernelPath::AffineQmm
    };
    affine_dispatch_stats_with(
        input_rows,
        in_features,
        output_rows,
        resident_row_bytes,
        qmv_passes,
        path,
    )
}

fn affine_dispatch_stats_with(
    input_rows: usize,
    in_features: usize,
    output_rows: usize,
    resident_row_bytes: usize,
    weight_passes: usize,
    path: GgmlKernelPath,
) -> Result<GgmlDispatchStats, GgmlAffineError> {
    if input_rows == 0 || output_rows == 0 {
        return Err(GgmlAffineError::InvalidInput);
    }
    Ok(GgmlDispatchStats {
        path,
        packed_bytes_read: resident_row_bytes
            .checked_mul(output_rows)
            .and_then(|bytes| bytes.checked_mul(weight_passes))
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

    #[test]
    fn mixed_qkv_dispatch_is_exclusive_to_m3_and_m4() {
        for input_rows in 0..=8 {
            assert_eq!(
                Qwen38MixedQkvBundle::uses_bundled_dispatch(input_rows),
                matches!(input_rows, 3 | 4),
            );
        }
    }

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
        assert_eq!(
            (codes[0], codes[31], scales[0], biases[0]),
            (0, 255, 1.0, -128.0)
        );

        let mut iq4 = vec![0u8; 18];
        put_half(&mut iq4, 0, 0x3c00);
        iq4[2] = 0xf0;
        let (weight, scales, biases) = transcode_block(GgmlQType::Iq4Nl, &iq4);
        let codes = unpack_codes(&weight, 8, 32);
        assert_eq!(
            (codes[0], codes[16], scales[0], biases[0]),
            (0, 240, 1.0, -127.0)
        );

        let mut iq3 = vec![0u8; 110];
        put_half(&mut iq3, 0, 0x3c00);
        iq3[74] = 1;
        let (weight, scales, biases) = transcode_block(GgmlQType::Iq3S, &iq3);
        let codes = unpack_codes(&weight, 5, 256);
        assert_eq!(
            (codes[0], codes[1], scales[0], biases[0]),
            (14, 16, 1.0, -15.0)
        );

        let mut iq4xs = vec![0u8; 136];
        put_half(&mut iq4xs, 0, 0x3c00);
        iq4xs[2] = 2;
        iq4xs[4] = 1;
        iq4xs[8] = 0xf0;
        let (weight, scales, biases) = transcode_block(GgmlQType::Iq4Xs, &iq4xs);
        let codes = unpack_codes(&weight, 8, 256);
        assert_eq!(
            (codes[0], codes[16], scales[0], biases[0]),
            (0, 240, 1.0, -127.0)
        );
    }

    #[test]
    fn q6_k_two_scale_subgroups_are_not_generally_affine_group32() {
        let minimum = -32i32 * 127;
        let maximum = -32i32 * -128;
        assert!(maximum - minimum > 255);
        assert!(affine_bits(GgmlQType::Q6K).is_none());
        assert_eq!(affine_bits(GgmlQType::Q5K), Some(5));
    }

    #[test]
    fn affine_load_boundary_rejects_fallback_lengths_and_overflow() {
        assert!(matches!(
            GgmlAffineMatrix::from_ggml_bytes(&[], GgmlQType::Q6K, 256, 1),
            Err(GgmlAffineError::NotRepresentable(GgmlQType::Q6K))
        ));
        assert!(matches!(
            GgmlAffineMatrix::from_ggml_bytes(&[0; 175], GgmlQType::Q5K, 256, 1),
            Err(GgmlAffineError::ByteLength {
                expected: 176,
                actual: 175
            })
        ));
        assert!(matches!(
            GgmlAffineMatrix::from_ggml_bytes(&[], GgmlQType::Q5K, usize::MAX, 2),
            Err(GgmlAffineError::Overflow)
        ));
    }
}
