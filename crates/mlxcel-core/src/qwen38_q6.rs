// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");

//! Dual packed/F16 execution for the pinned Qwen3.8 Q6_K non-head inventory.

use std::ops::Range;
use std::time::{Duration, Instant};

use cxx::UniquePtr;
use thiserror::Error;

use crate::{GgmlDispatchStats, GgmlKernelPath, GgmlQType, GgmlQuantizedMatrix, MlxArray, dtype};

const Q6_BLOCK_ELEMENTS: usize = 256;
const Q6_BLOCK_BYTES: usize = 210;
const RELEASE_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// Exact non-head Q6_K shapes in the pinned target/MTP pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38Q6Shape {
    K5120N1024,
    K5120N6144,
    K5120N10240,
    K5120N12288,
    K5120N17408,
    K6144N5120,
    K10240N5120,
    K17408N5120,
}

impl Qwen38Q6Shape {
    pub const fn dimensions(self) -> (usize, usize) {
        match self {
            Self::K5120N1024 => (5120, 1024),
            Self::K5120N6144 => (5120, 6144),
            Self::K5120N10240 => (5120, 10240),
            Self::K5120N12288 => (5120, 12288),
            Self::K5120N17408 => (5120, 17408),
            Self::K6144N5120 => (6144, 5120),
            Self::K10240N5120 => (10240, 5120),
            Self::K17408N5120 => (17408, 5120),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38Q6TranscodeStats {
    pub source_bytes: usize,
    pub dense_bytes: usize,
    pub resident_bytes: usize,
    pub peak_active_bytes: usize,
    pub peak_transient_bytes: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Error)]
pub enum Qwen38Q6Error {
    #[error("pinned Q6_K payload has {actual} bytes; expected {expected}")]
    ByteLength { expected: usize, actual: usize },
    #[error("pinned Q6_K dimensions or allocation size overflow")]
    Overflow,
    #[error("pinned Q6_K input shape or dtype is invalid")]
    InvalidInput,
    #[error("pinned Q6_K MLX dense plane is invalid")]
    InvalidDensePlane,
    #[error("pinned Q6_K packed fallback failed: {0}")]
    Packed(String),
}

pub struct Qwen38Q6DualMatrix {
    packed: GgmlQuantizedMatrix,
    dense_f16: UniquePtr<MlxArray>,
    shape: Qwen38Q6Shape,
    stats: Qwen38Q6TranscodeStats,
}

impl Qwen38Q6DualMatrix {
    pub fn from_pinned_bytes_with_progress(
        bytes: &[u8],
        shape: Qwen38Q6Shape,
        mut release_source: impl FnMut(Range<usize>),
    ) -> Result<Self, Qwen38Q6Error> {
        let started = Instant::now();
        let (in_features, out_features) = shape.dimensions();
        let source_row_bytes = in_features / Q6_BLOCK_ELEMENTS * Q6_BLOCK_BYTES;
        let source_bytes = source_row_bytes
            .checked_mul(out_features)
            .ok_or(Qwen38Q6Error::Overflow)?;
        if bytes.len() != source_bytes {
            return Err(Qwen38Q6Error::ByteLength {
                expected: source_bytes,
                actual: bytes.len(),
            });
        }
        let packed = GgmlQuantizedMatrix::from_bytes(
            bytes,
            GgmlQType::Q6K,
            in_features,
            out_features,
        )
        .map_err(|error| Qwen38Q6Error::Packed(error.to_string()))?;
        let dense_bytes = in_features
            .checked_mul(out_features)
            .and_then(|values| values.checked_mul(2))
            .ok_or(Qwen38Q6Error::Overflow)?;
        let mut dense = Vec::new();
        dense
            .try_reserve_exact(dense_bytes)
            .map_err(|_| Qwen38Q6Error::Overflow)?;
        let mut released = 0usize;
        for (row_index, row) in bytes.chunks_exact(source_row_bytes).enumerate() {
            transcode_q6_row_to_f16(row, &mut dense);
            let consumed = (row_index + 1) * source_row_bytes;
            if consumed - released >= RELEASE_CHUNK_BYTES {
                release_source(released..consumed);
                released = consumed;
            }
        }
        if released < bytes.len() {
            release_source(released..bytes.len());
        }
        if dense.len() != dense_bytes {
            return Err(Qwen38Q6Error::InvalidDensePlane);
        }
        let dense_f16 = crate::from_bytes_f16(
            &dense,
            &[out_features as i32, in_features as i32],
            false,
        );
        let dense_ref = dense_f16
            .as_ref()
            .ok_or(Qwen38Q6Error::InvalidDensePlane)?;
        if crate::array_dtype(dense_ref) != dtype::FLOAT16
            || crate::array_shape(dense_ref) != [out_features as i32, in_features as i32]
            || crate::array_nbytes(dense_ref) != dense_bytes
        {
            return Err(Qwen38Q6Error::InvalidDensePlane);
        }
        crate::eval(dense_ref);
        drop(dense);

        let resident_bytes = source_bytes
            .checked_add(dense_bytes)
            .ok_or(Qwen38Q6Error::Overflow)?;
        let peak_active_bytes = source_bytes
            .checked_add(dense_bytes.checked_mul(2).ok_or(Qwen38Q6Error::Overflow)?)
            .and_then(|value| value.checked_add(source_bytes.min(RELEASE_CHUNK_BYTES)))
            .ok_or(Qwen38Q6Error::Overflow)?;
        Ok(Self {
            packed,
            dense_f16,
            shape,
            stats: Qwen38Q6TranscodeStats {
                source_bytes,
                dense_bytes,
                resident_bytes,
                peak_active_bytes,
                peak_transient_bytes: peak_active_bytes - resident_bytes,
                elapsed: started.elapsed(),
            },
        })
    }

    pub const fn transcode_stats(&self) -> Qwen38Q6TranscodeStats {
        self.stats
    }

    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, Qwen38Q6Error> {
        let input_rows = validate_input(input, self.shape)?;
        if input_rows < 5 {
            return self
                .packed
                .forward(input)
                .map_err(|error| Qwen38Q6Error::Packed(error.to_string()));
        }
        let transposed = crate::transpose(
            self.dense_f16
                .as_ref()
                .ok_or(Qwen38Q6Error::InvalidDensePlane)?,
        );
        Ok(crate::matmul(
            input,
            transposed
                .as_ref()
                .ok_or(Qwen38Q6Error::InvalidDensePlane)?,
        ))
    }

    pub fn dispatch_stats(
        &self,
        input_rows: usize,
    ) -> Result<GgmlDispatchStats, Qwen38Q6Error> {
        let (in_features, out_features) = self.shape.dimensions();
        if input_rows == 0 {
            return Err(Qwen38Q6Error::InvalidInput);
        }
        if input_rows < 5 {
            return self
                .packed
                .dispatch_stats(input_rows)
                .map_err(|error| Qwen38Q6Error::Packed(error.to_string()));
        }
        Ok(GgmlDispatchStats {
            path: GgmlKernelPath::Qwen38Q6DenseF16,
            packed_bytes_read: self.stats.dense_bytes,
            activation_bytes_read: input_rows
                .checked_mul(in_features)
                .and_then(|values| values.checked_mul(4))
                .ok_or(Qwen38Q6Error::Overflow)?,
            output_bytes_written: input_rows
                .checked_mul(out_features)
                .and_then(|values| values.checked_mul(4))
                .ok_or(Qwen38Q6Error::Overflow)?,
            floating_point_operations: input_rows
                .checked_mul(in_features)
                .and_then(|values| values.checked_mul(out_features))
                .and_then(|fmas| fmas.checked_mul(2))
                .ok_or(Qwen38Q6Error::Overflow)?,
            threadgroup_bytes: 0,
            workspace_bytes: 0,
        })
    }
}

fn validate_input(input: &MlxArray, shape: Qwen38Q6Shape) -> Result<usize, Qwen38Q6Error> {
    let (in_features, _) = shape.dimensions();
    let dimensions = crate::array_shape(input);
    if dimensions.is_empty()
        || dimensions.last().copied() != Some(in_features as i32)
        || !matches!(
            crate::array_dtype(input),
            dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16
        )
    {
        return Err(Qwen38Q6Error::InvalidInput);
    }
    dimensions[..dimensions.len() - 1]
        .iter()
        .try_fold(1usize, |rows, dimension| {
            let dimension =
                usize::try_from(*dimension).map_err(|_| Qwen38Q6Error::InvalidInput)?;
            rows.checked_mul(dimension).ok_or(Qwen38Q6Error::Overflow)
        })
}

fn transcode_q6_row_to_f16(row: &[u8], output: &mut Vec<u8>) {
    for block in row.chunks_exact(Q6_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for column in 0..Q6_BLOCK_ELEMENTS {
            let half = column / 128;
            let within_half = column % 128;
            let quadrant = within_half / 32;
            let lane = within_half % 32;
            let low_packed = block[half * 64 + lane + (quadrant & 1) * 32];
            let low = if quadrant < 2 {
                low_packed & 15
            } else {
                low_packed >> 4
            };
            let high = (block[128 + half * 32 + lane] >> (2 * quadrant)) & 3;
            let quant = i32::from(low | (high << 4)) - 32;
            let scale = block[192 + half * 8 + lane / 16 + quadrant * 2] as i8;
            let value = d * f32::from(scale) * quant as f32;
            output.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return sign | if magnitude == 0x7f80_0000 { 0x7c00 } else { 0x7e00 };
    }
    let exponent = ((magnitude >> 23) as i32) - 127;
    let mantissa = magnitude & 0x7f_ffff;
    if exponent > 15 {
        return sign | 0x7c00;
    }
    if exponent >= -14 {
        let mut half_exp = (exponent + 15) as u16;
        let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
        if rounded & 0x80_0000 != 0 {
            half_exp += 1;
            if half_exp >= 31 {
                return sign | 0x7c00;
            }
        }
        return sign | (half_exp << 10) | ((rounded >> 13) as u16 & 0x03ff);
    }
    if exponent < -24 {
        return sign;
    }
    let significand = mantissa | 0x80_0000;
    let shift = (-exponent - 14 + 13) as u32;
    let rounded =
        significand + ((1u32 << (shift - 1)) - 1) + ((significand >> shift) & 1);
    sign | (rounded >> shift) as u16
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
    fn pinned_shapes_and_load_bounds_are_frozen() {
        assert_eq!(Qwen38Q6Shape::K6144N5120.dimensions(), (6144, 5120));
        assert_eq!(Qwen38Q6Shape::K5120N17408.dimensions(), (5120, 17408));
        assert!(matches!(
            Qwen38Q6DualMatrix::from_pinned_bytes_with_progress(
                &[],
                Qwen38Q6Shape::K5120N1024,
                |_| {}
            ),
            Err(Qwen38Q6Error::ByteLength {
                expected: 4_300_800,
                actual: 0
            })
        ));
    }

    #[test]
    fn host_f16_rounding_is_ties_to_even() {
        for value in [0.0, -0.0, 1.0, -2.0, 0.000_061_035_156, 65_504.0] {
            assert_eq!(f16_to_f32(f32_to_f16_bits(value)), value);
        }
        let halfway = f32::from_bits(1.0f32.to_bits() + (1 << 12));
        assert_eq!(f32_to_f16_bits(halfway), 0x3c00);
    }
}
