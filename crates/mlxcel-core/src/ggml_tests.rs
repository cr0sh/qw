use super::*;

const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mut mantissa = (bits & 0x03ff) as u32;
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut unbiased = -14_i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                unbiased -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((unbiased + 127) as u32) << 23) | (mantissa << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | (((exponent as u32 + 112) & 0xff) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn scale_min_k4(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

// Direct translations of ggml's canonical scalar row dequantizers. They are
// test-only so production cannot accidentally materialize a dense model.
fn decode_block(qtype: GgmlQType, block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), qtype.block_bytes());
    let mut output = vec![0.0; qtype.block_elements()];
    match qtype {
        GgmlQType::F32 => output[0] = f32::from_le_bytes(block.try_into().unwrap()),
        GgmlQType::Q8_0 => {
            let d = half_to_f32(u16_at(block, 0));
            for (value, quant) in output.iter_mut().zip(&block[2..]) {
                *value = d * (*quant as i8) as f32;
            }
        }
        GgmlQType::Q3K => {
            let d = half_to_f32(u16_at(block, 108));
            for (column, value) in output.iter_mut().enumerate() {
                let group = column / 16;
                let scale_low = if group < 8 {
                    block[96 + group] & 0x0f
                } else {
                    block[88 + group] >> 4
                };
                let scale_high = (block[104 + group % 4] >> (2 * (group / 4))) & 3;
                let scale = (scale_low | (scale_high << 4)) as i32 - 32;
                let q = block[32 + (column / 128) * 32 + column % 32];
                let low = (q >> (2 * ((column / 32) % 4))) & 3;
                let high = block[column % 32] & (1 << (column / 32));
                let quant = i32::from(low) - if high == 0 { 4 } else { 0 };
                *value = d * scale as f32 * quant as f32;
            }
        }
        GgmlQType::Q4K | GgmlQType::Q5K => {
            let d = half_to_f32(u16_at(block, 0));
            let dmin = half_to_f32(u16_at(block, 2));
            let scales = &block[4..16];
            let low_offset = if qtype == GgmlQType::Q4K { 16 } else { 48 };
            for (column, value) in output.iter_mut().enumerate() {
                let group = column / 32;
                let group_column = column % 32;
                let (scale, minimum) = scale_min_k4(group, scales);
                let packed = block[low_offset + (group / 2) * 32 + group_column];
                let mut quant = if group.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                if qtype == GgmlQType::Q5K && block[16 + group_column] & (1 << group) != 0 {
                    quant += 16;
                }
                *value = d * scale as f32 * quant as f32 - dmin * minimum as f32;
            }
        }
        GgmlQType::Q6K => {
            let d = half_to_f32(u16_at(block, 208));
            for (column, value) in output.iter_mut().enumerate() {
                let half = column / 128;
                let within_half = column % 128;
                let quadrant = within_half / 32;
                let lane = within_half % 32;
                let low_packed = block[half * 64 + lane + (quadrant & 1) * 32];
                let low = if quadrant < 2 {
                    low_packed & 0x0f
                } else {
                    low_packed >> 4
                };
                let high = (block[128 + half * 32 + lane] >> (2 * quadrant)) & 3;
                let quant = i32::from(low | (high << 4)) - 32;
                let scale = block[192 + half * 8 + lane / 16 + quadrant * 2] as i8;
                *value = d * scale as f32 * quant as f32;
            }
        }
        GgmlQType::Iq4Nl => {
            let d = half_to_f32(u16_at(block, 0));
            for column in 0..16 {
                output[column] = d * IQ4_VALUES[(block[2 + column] & 0x0f) as usize] as f32;
                output[column + 16] = d * IQ4_VALUES[(block[2 + column] >> 4) as usize] as f32;
            }
        }
        GgmlQType::Iq3S => {
            let d = half_to_f32(u16_at(block, 0));
            for (column, value) in output.iter_mut().enumerate() {
                let group = column / 32;
                let subblock = (column % 32) / 8;
                let lane = column % 8;
                let high = (block[66 + group] >> (2 * subblock + usize::from(lane >= 4))) & 1;
                let grid_index =
                    usize::from(block[2 + group * 8 + subblock * 2 + usize::from(lane >= 4)])
                        | (usize::from(high) << 8);
                let grid = IQ3S_GRID[grid_index];
                let magnitude = ((grid >> ((lane & 3) * 8)) & 0xff) as f32;
                let sign = if block[74 + group * 4 + subblock] & (1 << lane) != 0 {
                    -1.0
                } else {
                    1.0
                };
                let scales = block[106 + group / 2];
                let scale = if group.is_multiple_of(2) {
                    scales & 0x0f
                } else {
                    scales >> 4
                };
                *value = d * (1 + 2 * u32::from(scale)) as f32 * magnitude * sign;
            }
        }
        GgmlQType::Iq4Xs => {
            let d = half_to_f32(u16_at(block, 0));
            let scales_high = u16_at(block, 2);
            for (column, value) in output.iter_mut().enumerate() {
                let group = column / 32;
                let group_column = column % 32;
                let scales_low = block[4 + group / 2];
                let low = (scales_low >> (4 * (group & 1))) & 0x0f;
                let high = ((scales_high >> (2 * group)) & 3) as u8;
                let scale = i32::from(low | (high << 4)) - 32;
                let packed = block[8 + group * 16 + group_column % 16];
                let index = if group_column < 16 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                *value = d * scale as f32 * IQ4_VALUES[index as usize] as f32;
            }
        }
    }
    output
}

fn decode_row(qtype: GgmlQType, bytes: &[u8], elements: usize) -> Vec<f32> {
    assert!(elements.is_multiple_of(qtype.block_elements()));
    assert_eq!(
        bytes.len(),
        elements / qtype.block_elements() * qtype.block_bytes()
    );
    bytes
        .chunks_exact(qtype.block_bytes())
        .flat_map(|block| decode_block(qtype, block))
        .collect()
}

fn put_half(block: &mut [u8], offset: usize, bits: u16) {
    block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

fn fixture_block(qtype: GgmlQType, seed: u32) -> Vec<u8> {
    if qtype == GgmlQType::F32 {
        return ((seed as i32 % 31 - 15) as f32 * 0.03125)
            .to_le_bytes()
            .to_vec();
    }
    let mut state = seed.wrapping_add(0x9e37_79b9);
    let mut block = vec![0u8; qtype.block_bytes()];
    for byte in &mut block {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    match qtype {
        GgmlQType::Q8_0 | GgmlQType::Iq4Nl | GgmlQType::Iq3S | GgmlQType::Iq4Xs => {
            put_half(&mut block, 0, 0x2800)
        }
        GgmlQType::Q3K => put_half(&mut block, 108, 0x2800),
        GgmlQType::Q4K | GgmlQType::Q5K => {
            put_half(&mut block, 0, 0x2800);
            put_half(&mut block, 2, 0x2000);
        }
        GgmlQType::Q6K => put_half(&mut block, 208, 0x2800),
        GgmlQType::F32 => unreachable!(),
    }
    block
}

fn fixture_matrix(qtype: GgmlQType, width: usize, rows: usize) -> Vec<u8> {
    let blocks_per_row = width / qtype.block_elements();
    let mut bytes = Vec::with_capacity(blocks_per_row * rows * qtype.block_bytes());
    for row in 0..rows {
        for block in 0..blocks_per_row {
            bytes.extend(fixture_block(qtype, (row * 19 + block * 7 + 1) as u32));
        }
    }
    bytes
}

fn raw_f32(array: &MlxArray) -> Vec<f32> {
    crate::eval(array);
    crate::array_to_raw_bytes(array)
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    ordered(left).abs_diff(ordered(right))
}

fn median_forward(mut launch: impl FnMut() -> UniquePtr<MlxArray>) -> std::time::Duration {
    let warm = launch();
    crate::eval(warm.as_ref().unwrap());
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let started = std::time::Instant::now();
        let output = launch();
        crate::eval(output.as_ref().unwrap());
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    samples[2]
}

#[test]
fn fixed_qtype_table_matches_parsed_target_contract() {
    let actual = GgmlQType::TARGET_TYPES
        .map(|qtype| (qtype.id(), qtype.block_elements(), qtype.block_bytes()));
    assert_eq!(
        actual,
        [
            (0, 1, 4),
            (8, 32, 34),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (20, 32, 18),
            (21, 256, 110),
            (23, 256, 136)
        ]
    );
    assert!(GgmlQType::try_from(2).is_err());
    assert!(GgmlQType::try_from(17).is_err());
}

#[test]
fn load_boundary_rejects_invalid_qtypes_shapes_lengths_and_overflow() {
    assert!(matches!(
        GgmlQuantizedMatrix::from_bytes(&[], GgmlQType::Q3K, 255, 1),
        Err(GgmlQuantError::UnalignedWidth { .. })
    ));
    assert!(matches!(
        GgmlQuantizedMatrix::from_bytes(&[0; 109], GgmlQType::Q3K, 256, 1),
        Err(GgmlQuantError::ByteLength {
            expected: 110,
            actual: 109
        })
    ));
    assert!(matches!(
        GgmlQuantizedMatrix::from_bytes(&[], GgmlQType::F32, usize::MAX, 2),
        Err(GgmlQuantError::Overflow)
    ));
}

#[test]
fn reference_layouts_pin_nibbles_high_bits_signs_and_table_indices() {
    let mut q8 = vec![0; 34];
    put_half(&mut q8, 0, 0x3c00);
    q8[2] = (-128i8) as u8;
    q8[33] = 127;
    let decoded = decode_block(GgmlQType::Q8_0, &q8);
    assert_eq!((decoded[0], decoded[31]), (-128.0, 127.0));

    let mut q3 = vec![0; 110];
    put_half(&mut q3, 108, 0x3c00);
    q3[96] = 0x11;
    q3[104] = 0x22;
    q3[32] = 2;
    q3[64] = 3;
    let decoded = decode_block(GgmlQType::Q3K, &q3);
    assert_eq!(decoded[0], -2.0);
    assert_eq!(decoded[128], -1.0);
    q3[0] = 0x11;
    let decoded = decode_block(GgmlQType::Q3K, &q3);
    assert_eq!(decoded[0], 2.0);
    assert_eq!(decoded[128], 3.0);

    let mut q4 = vec![0; 144];
    put_half(&mut q4, 0, 0x3c00);
    q4[4] = 1;
    q4[5] = 2;
    q4[16] = 3 | (4 << 4);
    let decoded = decode_block(GgmlQType::Q4K, &q4);
    assert_eq!((decoded[0], decoded[32]), (3.0, 8.0));

    let mut q5 = vec![0; 176];
    put_half(&mut q5, 0, 0x3c00);
    q5[4] = 1;
    q5[5] = 2;
    q5[16] = 0b11;
    q5[48] = 3 | (4 << 4);
    let decoded = decode_block(GgmlQType::Q5K, &q5);
    assert_eq!((decoded[0], decoded[32]), (19.0, 40.0));

    let mut q6 = vec![0; 210];
    put_half(&mut q6, 208, 0x3c00);
    q6[0] = 0x21;
    q6[128] = 0b11_10_01_00;
    q6[192] = 2;
    q6[196] = (-3i8) as u8;
    let decoded = decode_block(GgmlQType::Q6K, &q6);
    assert_eq!(decoded[0], -62.0);
    assert_eq!(decoded[64], -6.0);

    let mut iq4 = vec![0; 18];
    put_half(&mut iq4, 0, 0x3c00);
    iq4[2] = 0xf0;
    let decoded = decode_block(GgmlQType::Iq4Nl, &iq4);
    assert_eq!((decoded[0], decoded[16]), (-127.0, 113.0));

    let mut iq3 = vec![0; 110];
    put_half(&mut iq3, 0, 0x3c00);
    iq3[66] = 1;
    iq3[74] = 1;
    iq3[106] = 2;
    let decoded = decode_block(GgmlQType::Iq3S, &iq3);
    let grid = IQ3S_GRID[256];
    assert_eq!(decoded[0], -5.0 * (grid & 0xff) as f32);
    assert_eq!(decoded[1], 5.0 * ((grid >> 8) & 0xff) as f32);
    assert_eq!(decoded[4], 5.0);

    let mut iq4xs = vec![0; 136];
    put_half(&mut iq4xs, 0, 0x3c00);
    iq4xs[2] = 2;
    iq4xs[4] = 1;
    iq4xs[8] = 0xf0;
    let decoded = decode_block(GgmlQType::Iq4Xs, &iq4xs);
    assert_eq!((decoded[0], decoded[16]), (-127.0, 113.0));
}

#[test]
fn dispatch_stats_record_zero_workspace_and_shape_selected_traffic() {
    let qtype = GgmlQType::Q4K;
    let width = 512;
    let rows = 3;
    let packed = fixture_matrix(qtype, width, rows);
    let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, rows).unwrap();
    let decode = matrix.dispatch_stats(1).unwrap();
    assert_eq!(decode.path, GgmlKernelPath::DecodeM1);
    assert_eq!(decode.packed_bytes_read, packed.len());
    assert_eq!(decode.workspace_bytes, 0);
    let prefill = matrix.dispatch_stats(33).unwrap();
    assert_eq!(prefill.path, GgmlKernelPath::TiledQmm16x8);
    assert_eq!(prefill.packed_bytes_read, packed.len() * 3);
    assert_eq!(prefill.activation_bytes_read, 33 * width * 4);
    assert_eq!(prefill.threadgroup_bytes, 16 * 256 * 4);
    assert_eq!(prefill.workspace_bytes, 0);
    let embedding = GgmlQuantizedEmbedding::from_bytes(&packed, qtype, width, rows).unwrap();
    let lookup = embedding.dispatch_stats(2).unwrap();
    assert_eq!(lookup.path, GgmlKernelPath::Embedding);
    assert_eq!(lookup.packed_bytes_read, packed.len() / rows * 2);
    assert_eq!(lookup.workspace_bytes, 0);
    let cloned = embedding.clone_shared();
    assert_eq!(cloned.embedding_dim(), embedding.embedding_dim());
    assert_eq!(cloned.vocab_size(), embedding.vocab_size());
    assert_eq!(cloned.dispatch_stats(2).unwrap(), lookup);
}

#[test]
fn packed_row_ranges_preserve_order_bounds_and_arithmetic() {
    let qtype = GgmlQType::Q4K;
    let width = 512usize;
    let output_rows = 16usize;
    let packed = fixture_matrix(qtype, width, output_rows);
    let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, output_rows).unwrap();
    assert!(matches!(
        matrix.select_rows(&[]),
        Err(GgmlQuantError::InvalidRowSelection)
    ));
    assert!(matches!(
        matrix.select_rows(&[2..2]),
        Err(GgmlQuantError::InvalidRowRange { .. })
    ));
    assert!(matches!(
        matrix.select_rows(&[15..17]),
        Err(GgmlQuantError::InvalidRowRange { .. })
    ));
    assert!(matches!(
        matrix.select_rows(&[0..1, 1..2, 2..3, 3..4]),
        Err(GgmlQuantError::InvalidRowSelection)
    ));
    let rows = matrix.select_rows(&[2..5, 10..14]).unwrap();
    assert_eq!(rows.selected_rows(), 7);
    let stats = rows.dispatch_stats(3).unwrap();
    assert_eq!(stats.path, GgmlKernelPath::VerifyM2To4);
    assert_eq!(stats.packed_bytes_read, matrix.row_bytes * 7);
    assert_eq!(stats.floating_point_operations, 2 * 3 * 7 * width);
    assert_eq!(stats.workspace_bytes, 0);
    if !crate::metal_is_available() {
        return;
    }
    let input_values: Vec<f32> = (0..3 * width)
        .map(|index| (index as i32 % 19 - 9) as f32 * 0.0078125)
        .collect();
    let input = crate::from_slice_f32(&input_values, &[3, width as i32]);
    let full = raw_f32(
        matrix
            .forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    let selected = raw_f32(
        rows.forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    let mapping = [2usize, 3, 4, 10, 11, 12, 13];
    for input_row in 0..3 {
        for (selected_row, full_row) in mapping.into_iter().enumerate() {
            assert!(
                ulp_distance(
                    selected[input_row * mapping.len() + selected_row],
                    full[input_row * output_rows + full_row],
                ) <= 1
            );
        }
    }
}
#[test]
fn qwen38_q6_verify_head_gate_selects_only_exact_m3_m4() {
    let verify_ranges = [0..80_896, 248_044..248_070];
    assert!(is_qwen38_q6_verify_head_selection(
        GgmlQType::Q6K,
        5_120,
        248_320,
        &verify_ranges,
    ));
    assert!(!is_qwen38_q6_verify_head_selection(
        GgmlQType::Q6K,
        5_120,
        248_320,
        &[0..65_536, 248_044..248_070],
    ));
    assert!(!is_qwen38_q6_verify_head_selection(
        GgmlQType::Q5K,
        5_120,
        248_320,
        &verify_ranges,
    ));
    assert!(!is_qwen38_q6_verify_head_selection(
        GgmlQType::Q6K,
        5_120,
        248_320,
        &[0..80_896, 248_045..248_071],
    ));
    assert!(!is_qwen38_q6_verify_head_selection(
        GgmlQType::Q6K,
        5_120,
        248_319,
        &verify_ranges,
    ));

    assert_eq!(
        packed_dispatch_stats(1, 5_120, 80_922, 4_200, true)
            .unwrap()
            .path,
        GgmlKernelPath::DecodeM1,
    );
    assert_eq!(
        packed_dispatch_stats(2, 5_120, 80_922, 4_200, true)
            .unwrap()
            .path,
        GgmlKernelPath::VerifyM2To4,
    );
    for rows in [3, 4] {
        let stats = packed_dispatch_stats(rows, 5_120, 80_922, 4_200, true).unwrap();
        assert_eq!(stats.path, GgmlKernelPath::Qwen38Q6HeadVerifyR8);
        assert_eq!(stats.packed_bytes_read, 339_872_400);
        assert_eq!(stats.workspace_bytes, 0);
    }
    assert_eq!(
        packed_dispatch_stats(3, 5_120, 80_922, 4_200, false)
            .unwrap()
            .path,
        GgmlKernelPath::VerifyM2To4,
    );
}

#[test]
fn metal_matrix_and_embedding_match_reference_for_every_target_qtype() {
    if !crate::metal_is_available() {
        return;
    }
    for qtype in GgmlQType::TARGET_TYPES {
        let width = if qtype == GgmlQType::F32 {
            33
        } else {
            qtype.block_elements() * 2
        };
        let output_rows = 3;
        let packed = fixture_matrix(qtype, width, output_rows);
        let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, output_rows).unwrap();
        let weights: Vec<Vec<f32>> = (0..output_rows)
            .map(|row| {
                let rb = matrix.row_bytes;
                decode_row(qtype, &packed[row * rb..(row + 1) * rb], width)
            })
            .collect();
        let activation: Vec<f32> = (0..width)
            .map(|column| (column as i32 % 23 - 11) as f32 * 0.015625)
            .collect();
        let input = crate::from_slice_f32(&activation, &[1, width as i32]);
        let decode = matrix.forward(input.as_ref().unwrap()).unwrap();
        let decode_values = raw_f32(decode.as_ref().unwrap());
        for row in 0..output_rows {
            let expected = activation
                .iter()
                .zip(&weights[row])
                .map(|(input, weight)| input * weight)
                .sum::<f32>();
            let actual = decode_values[row];
            let tolerance = 0.002 + expected.abs() * 0.00002;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{qtype:?} decode row {row}: actual {actual}, expected {expected}"
            );
        }
        let mut verify_input = Vec::with_capacity(width * 3);
        for row in 0..3 {
            verify_input.extend(
                activation
                    .iter()
                    .map(|value| value + row as f32 * 0.0078125),
            );
        }
        let input = crate::from_slice_f32(&verify_input, &[3, width as i32]);
        let verify = matrix.forward(input.as_ref().unwrap()).unwrap();
        let verify_values = raw_f32(verify.as_ref().unwrap());
        for output in 0..output_rows {
            assert!(
                ulp_distance(decode_values[output], verify_values[output]) <= 1,
                "{qtype:?} M=1/M=3 mismatch: {} vs {}",
                decode_values[output],
                verify_values[output]
            );
        }
        for input_row in 0..3 {
            for output in 0..output_rows {
                let expected = verify_input[input_row * width..(input_row + 1) * width]
                    .iter()
                    .zip(&weights[output])
                    .map(|(input, weight)| input * weight)
                    .sum::<f32>();
                let actual = verify_values[input_row * output_rows + output];
                let tolerance = 0.002 + expected.abs() * 0.00002;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{qtype:?} verify input {input_row} output {output}: actual {actual}, expected {expected}"
                );
            }
        }
        let embedding =
            GgmlQuantizedEmbedding::from_bytes(&packed, qtype, width, output_rows).unwrap();
        let indices = crate::from_slice_i32(&[2, 0, 1], &[3]);
        let selected = embedding.forward(indices.as_ref().unwrap()).unwrap();
        let selected = raw_f32(selected.as_ref().unwrap());
        for (selected_row, source_row) in [2usize, 0, 1].into_iter().enumerate() {
            for column in 0..width {
                let actual = selected[selected_row * width + column];
                let expected = weights[source_row][column];
                assert!(
                    ulp_distance(actual, expected) <= 1,
                    "{qtype:?} embedding row {source_row} column {column}: actual {actual}, expected {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires verified local target and MTP GGUF artifacts"]
fn actual_gguf_blocks_match_reference_on_metal() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    if !crate::metal_is_available() {
        return;
    }
    let structure_path = std::env::var("QWR_GGUF_STRUCTURE_PATH").expect("QWR_GGUF_STRUCTURE_PATH");
    let structure: serde_json::Value =
        serde_json::from_reader(File::open(structure_path).unwrap()).unwrap();
    let mtp_types = [
        GgmlQType::F32,
        GgmlQType::Q3K,
        GgmlQType::Q4K,
        GgmlQType::Q6K,
    ];
    for (model_key, path_variable, qtypes) in [
        (
            "target",
            "QWR_TARGET_GGUF_SMOKE_PATH",
            GgmlQType::TARGET_TYPES.as_slice(),
        ),
        ("mtp", "QWR_MTP_GGUF_SMOKE_PATH", mtp_types.as_slice()),
    ] {
        let mut file = File::open(std::env::var(path_variable).expect(path_variable)).unwrap();
        let model = &structure[model_key];
        let data_start = model["data_start"].as_u64().unwrap();
        let tensors = model["tensor_infos_enriched"].as_array().unwrap();
        for &qtype in qtypes {
            let tensor = tensors
                .iter()
                .find(|tensor| {
                    tensor["type"].as_u64() == Some(u64::from(qtype.id()))
                        && tensor["shape"]
                            .as_array()
                            .and_then(|shape| shape.first())
                            .and_then(serde_json::Value::as_u64)
                            .is_some_and(|width| {
                                (width as usize).is_multiple_of(qtype.block_elements())
                            })
                })
                .unwrap_or_else(|| panic!("{model_key} has no aligned {qtype:?} tensor"));
            let width = tensor["shape"][0].as_u64().unwrap() as usize;
            let row_bytes = width / qtype.block_elements() * qtype.block_bytes();
            assert!(tensor["payload_bytes"].as_u64().unwrap() as usize >= row_bytes);
            let offset = tensor["offset"].as_u64().unwrap();
            let mut packed = vec![0u8; row_bytes];
            file.seek(SeekFrom::Start(data_start + offset)).unwrap();
            file.read_exact(&mut packed).unwrap();

            let weights = decode_row(qtype, &packed, width);
            let activation: Vec<f32> = (0..width)
                .map(|column| (column as i32 % 29 - 14) as f32 * 0.00390625)
                .collect();
            let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, 1).unwrap();
            let input = crate::from_slice_f32(&activation, &[1, width as i32]);
            let cold = matrix.forward(input.as_ref().unwrap()).unwrap();
            let _ = raw_f32(cold.as_ref().unwrap());
            let started = Instant::now();
            let warm = matrix.forward(input.as_ref().unwrap()).unwrap();
            let decode_actual = raw_f32(warm.as_ref().unwrap())[0];
            let elapsed = started.elapsed();
            let expected = activation
                .iter()
                .zip(&weights)
                .map(|(input, weight)| input * weight)
                .sum::<f32>();
            let tolerance = 0.003 + expected.abs() * 0.00005;
            assert!(
                (decode_actual - expected).abs() <= tolerance,
                "{model_key} {qtype:?} tensor {}: actual {decode_actual}, expected {expected}",
                tensor["name"].as_str().unwrap()
            );

            let mut row_counts = vec![3usize, 245, 288];
            if model_key == "target" && qtype == GgmlQType::Q5K {
                row_counts.push(2048);
            }
            for rows in row_counts {
                let mut values = Vec::with_capacity(rows * width);
                for row in 0..rows {
                    values.extend(
                        activation
                            .iter()
                            .map(|value| value + (row % 7) as f32 * 0.0009765625),
                    );
                }
                let input = crate::from_slice_f32(&values, &[rows as i32, width as i32]);
                let output = matrix.forward(input.as_ref().unwrap()).unwrap();
                let actual = raw_f32(output.as_ref().unwrap());
                if rows == 3 {
                    assert!(ulp_distance(decode_actual, actual[0]) <= 1);
                }
                for row in 0..rows {
                    let expected = values[row * width..(row + 1) * width]
                        .iter()
                        .zip(&weights)
                        .map(|(input, weight)| input * weight)
                        .sum::<f32>();
                    let tolerance = 0.003 + expected.abs() * 0.00005;
                    assert!(
                        (actual[row] - expected).abs() <= tolerance,
                        "{model_key} {qtype:?} M={rows} row={row}: {} vs {expected}",
                        actual[row]
                    );
                }
            }

            let embedding = GgmlQuantizedEmbedding::from_bytes(&packed, qtype, width, 1).unwrap();
            let index = crate::from_slice_i32(&[0], &[1]);
            let selected = embedding.forward(index.as_ref().unwrap()).unwrap();
            let selected = raw_f32(selected.as_ref().unwrap());
            for (column, (&actual, &expected)) in selected.iter().zip(&weights).enumerate() {
                assert!(
                    ulp_distance(actual, expected) <= 1,
                    "{model_key} {qtype:?} actual embedding column {column}: {actual} vs {expected}"
                );
            }
            eprintln!(
                "actual GGUF {model_key} {qtype:?} width={width} warm_m1_row={elapsed:?} workspace=0B"
            );
        }
    }
}

#[test]
#[ignore = "requires verified local target GGUF artifact"]
fn actual_q5k_row_ranges_match_full_projection() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    if !crate::metal_is_available() {
        return;
    }
    let structure: serde_json::Value = serde_json::from_reader(
        File::open(std::env::var("QWR_GGUF_STRUCTURE_PATH").unwrap()).unwrap(),
    )
    .unwrap();
    let model = &structure["target"];
    let tensor = model["tensor_infos_enriched"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| {
            tensor["type"].as_u64() == Some(13)
                && tensor["shape"][0].as_u64() == Some(5120)
                && tensor["shape"][1].as_u64().is_some_and(|rows| rows >= 64)
        })
        .unwrap();
    let qtype = GgmlQType::Q5K;
    let width = 5120usize;
    let output_rows = 64usize;
    let row_bytes = width / qtype.block_elements() * qtype.block_bytes();
    let mut packed = vec![0u8; row_bytes * output_rows];
    let mut file = File::open(std::env::var("QWR_TARGET_GGUF_SMOKE_PATH").unwrap()).unwrap();
    file.seek(SeekFrom::Start(
        model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap(),
    ))
    .unwrap();
    file.read_exact(&mut packed).unwrap();
    let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, output_rows).unwrap();
    let selected = matrix.select_rows(&[0..7, 40..47, 60..64]).unwrap();
    let input_values: Vec<f32> = (0..3 * width)
        .map(|index| (index as i32 % 23 - 11) as f32 * 0.00390625)
        .collect();
    let input = crate::from_slice_f32(&input_values, &[3, width as i32]);
    let full = raw_f32(
        matrix
            .forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    let compact = raw_f32(
        selected
            .forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    let mapping = [
        0usize, 1, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 60, 61, 62, 63,
    ];
    for input_row in 0..3 {
        for (compact_row, full_row) in mapping.into_iter().enumerate() {
            assert!(
                ulp_distance(
                    compact[input_row * mapping.len() + compact_row],
                    full[input_row * output_rows + full_row],
                ) <= 1
            );
        }
    }
}

#[test]
#[ignore = "requires verified local target GGUF artifact"]
fn actual_q5k_qmm_microbench() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::{Duration, Instant};

    if !crate::metal_is_available() {
        return;
    }
    let structure_path = std::env::var("QWR_GGUF_STRUCTURE_PATH").expect("QWR_GGUF_STRUCTURE_PATH");
    let structure: serde_json::Value =
        serde_json::from_reader(File::open(structure_path).unwrap()).unwrap();
    let model = &structure["target"];
    let tensor = model["tensor_infos_enriched"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| {
            tensor["type"].as_u64() == Some(13)
                && tensor["shape"][0].as_u64() == Some(5120)
                && tensor["shape"][1].as_u64().is_some_and(|rows| rows >= 1024)
        })
        .expect("representative 5120-wide Q5_K matrix");
    let qtype = GgmlQType::Q5K;
    let width = 5120usize;
    let output_rows = 1024usize;
    let row_bytes = width / qtype.block_elements() * qtype.block_bytes();
    let mut packed = vec![0u8; row_bytes * output_rows];
    let mut file = File::open(std::env::var("QWR_TARGET_GGUF_SMOKE_PATH").unwrap()).unwrap();
    let offset = model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.read_exact(&mut packed).unwrap();
    let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, output_rows).unwrap();

    for rows in [1usize, 3, 245, 288, 2048] {
        let input_values: Vec<f32> = (0..rows * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&input_values, &[rows as i32, width as i32]);
        crate::eval(input.as_ref().unwrap());
        let warm = matrix.forward(input.as_ref().unwrap()).unwrap();
        crate::eval(warm.as_ref().unwrap());
        let mut samples = Vec::with_capacity(3);
        for _ in 0..3 {
            let started = Instant::now();
            let output = matrix.forward(input.as_ref().unwrap()).unwrap();
            crate::eval(output.as_ref().unwrap());
            samples.push(started.elapsed());
        }
        samples.sort_unstable();
        let median = samples[1];
        let seconds = median.as_secs_f64();
        let stats = matrix.dispatch_stats(rows).unwrap();
        let gbytes = (stats.packed_bytes_read + stats.activation_bytes_read) as f64 / 1e9;
        let flops = stats.floating_point_operations as f64;
        eprintln!(
            "actual Q5_K qmm M={rows} N={output_rows} K={width} median={median:?} throughput={:.3} TFLOP/s logical={:.3} GB/s workspace={}B",
            flops / seconds / 1e12,
            gbytes / seconds,
            stats.workspace_bytes
        );
        assert!(median < Duration::from_secs(30));
    }
}

#[test]
#[ignore = "requires the exact pinned target GGUF and 1.1 GB packed allocation"]
fn actual_q6_lm_head_selected_logits_and_ids_are_exact() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    fn compact_reference(full: &MlxArray, prefix_rows: i32) -> UniquePtr<MlxArray> {
        let shape = crate::array_shape(full);
        let axis = shape.len() - 1;
        let start = vec![0; shape.len()];
        let mut end = shape.clone();
        end[axis] = prefix_rows;
        let prefix = crate::slice(full, &start, &end);
        let mut control_start = vec![0; shape.len()];
        let mut control_end = shape;
        control_start[axis] = 248_044;
        control_end[axis] = 248_070;
        let controls = crate::slice(full, &control_start, &control_end);
        crate::concatenate(prefix.as_ref().unwrap(), controls.as_ref().unwrap(), -1)
    }

    fn evaluated_bytes(array: &MlxArray) -> Vec<u8> {
        crate::eval(array);
        crate::array_to_raw_bytes(array)
    }

    fn assert_case(
        label: &str,
        matrix: &GgmlQuantizedMatrix,
        selected: &GgmlQuantizedRows,
        input: &MlxArray,
        prefix_rows: i32,
        rows: usize,
        expected_selected_path: GgmlKernelPath,
    ) {
        assert_eq!(
            matrix.dispatch_stats(rows).unwrap().path,
            if rows == 1 {
                GgmlKernelPath::DecodeM1
            } else {
                GgmlKernelPath::VerifyM2To4
            },
        );
        assert_eq!(
            selected.dispatch_stats(rows).unwrap().path,
            expected_selected_path,
        );
        let full = matrix.forward(input).unwrap();
        let expected = compact_reference(full.as_ref().unwrap(), prefix_rows);
        let actual = selected.forward(input).unwrap();
        let expected_bytes = evaluated_bytes(expected.as_ref().unwrap());
        let actual_bytes = evaluated_bytes(actual.as_ref().unwrap());
        let max_ulp = expected_bytes
            .chunks_exact(4)
            .zip(actual_bytes.chunks_exact(4))
            .map(|(left, right)| {
                ulp_distance(
                    f32::from_le_bytes(left.try_into().unwrap()),
                    f32::from_le_bytes(right.try_into().unwrap()),
                )
            })
            .max()
            .unwrap_or(0);
        assert!(
            expected_bytes == actual_bytes,
            "{label} selected logits differ (max ULP {max_ulp})",
        );
        assert_eq!(max_ulp, 0, "{label} selected logits are not exact");

        let expected_ids = crate::argmax_last_axis(expected.as_ref().unwrap());
        let actual_ids = crate::argmax_last_axis(actual.as_ref().unwrap());
        assert_eq!(
            evaluated_bytes(expected_ids.as_ref().unwrap()),
            evaluated_bytes(actual_ids.as_ref().unwrap()),
            "{label} selected argmax IDs differ",
        );
    }

    if !crate::metal_is_available() {
        return;
    }
    let qtype = GgmlQType::Q6K;
    let width = 5_120usize;
    let output_rows = 248_320usize;
    let row_bytes = width / qtype.block_elements() * qtype.block_bytes();
    assert_eq!(row_bytes, 4_200);
    let path = std::env::var_os("QWR_TARGET_GGUF_SMOKE_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".cache/qw/models/unsloth/Qwen3.8-27B-GGUF")
                .join("Qwen3.8-27B-UD-Q4_K_XL.gguf")
        });
    let mut packed = vec![0u8; row_bytes * output_rows];
    let mut file = File::open(path).unwrap();
    file.seek(SeekFrom::Start(10_996_640)).unwrap();
    file.read_exact(&mut packed).unwrap();
    let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype, width, output_rows).unwrap();
    drop(packed);

    let draft = matrix.select_rows(&[0..65_536, 248_044..248_070]).unwrap();
    let draft_input = crate::from_slice_f32(
        &(0..width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect::<Vec<_>>(),
        &[1, 1, width as i32],
    );
    assert_case(
        "draft M1",
        &matrix,
        &draft,
        draft_input.as_ref().unwrap(),
        65_536,
        1,
        GgmlKernelPath::DecodeM1,
    );
    drop(draft);

    let verify = matrix.select_rows(&[0..80_896, 248_044..248_070]).unwrap();
    for rows in [3usize, 4] {
        let input = crate::from_slice_f32(
            &(0..rows * width)
                .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
                .collect::<Vec<_>>(),
            &[1, rows as i32, width as i32],
        );
        assert_case(
            &format!("verify M{rows}"),
            &matrix,
            &verify,
            input.as_ref().unwrap(),
            80_896,
            rows,
            GgmlKernelPath::Qwen38Q6HeadVerifyR8,
        );
    }
}

#[test]
fn qwen38_affine_m23_reads_each_bits_plane_once_with_qmv_parity() {
    if !crate::metal_is_available() {
        return;
    }
    let width = 512usize;
    let output_rows = 8usize;
    for qtype in [GgmlQType::Q4K, GgmlQType::Q5K, GgmlQType::Q8_0] {
        let source = fixture_matrix(qtype, width, output_rows);
        let matrix =
            crate::GgmlAffineMatrix::from_ggml_bytes(&source, qtype, width, output_rows).unwrap();
        for input_rows in [2usize, 3] {
            let values = (0..input_rows * width)
                .map(|index| {
                    let row = index / width;
                    let column = index % width;
                    (column as i32 % 37 - 18) as f32 * 0.00390625 + row as f32 * 0.0009765625
                })
                .collect::<Vec<_>>();
            let input = crate::from_slice_f32(&values, &[1, input_rows as i32, width as i32]);
            let split = raw_f32(
                matrix
                    .forward(input.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            let one_pass = raw_f32(
                matrix
                    .forward_m23_test_only(input.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            let max_ulp = split
                .iter()
                .zip(&one_pass)
                .map(|(&left, &right)| ulp_distance(left, right))
                .max()
                .unwrap_or(0);
            assert!(
                max_ulp <= 1,
                "{qtype:?} M={input_rows} one-pass affine differs by {max_ulp} ULP"
            );

            let split_stats = matrix.dispatch_stats(input_rows).unwrap();
            let one_pass_stats = matrix.m23_dispatch_stats_test_only(input_rows).unwrap();
            assert_eq!(one_pass_stats.path, GgmlKernelPath::Qwen38AffineM23);
            assert_eq!(
                split_stats.packed_bytes_read,
                one_pass_stats.packed_bytes_read * input_rows
            );
            assert_eq!(
                split_stats.activation_bytes_read,
                one_pass_stats.activation_bytes_read
            );
            assert_eq!(
                split_stats.output_bytes_written,
                one_pass_stats.output_bytes_written
            );
        }
    }
}

#[test]
#[ignore = "requires verified local target GGUF artifact"]
fn prototype_actual_q5_k_affine_repack() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    if !crate::metal_is_available() {
        return;
    }
    let structure: serde_json::Value = serde_json::from_reader(
        File::open(std::env::var("QWR_GGUF_STRUCTURE_PATH").unwrap()).unwrap(),
    )
    .unwrap();
    let model = &structure["target"];
    let tensor = model["tensor_infos_enriched"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| {
            tensor["type"].as_u64() == Some(13)
                && tensor["shape"][0].as_u64() == Some(5120)
                && tensor["shape"][1].as_u64().is_some_and(|rows| rows >= 1024)
        })
        .unwrap();
    let width = 5120usize;
    let rows = 1024usize;
    let row_bytes = width / 256 * 176;
    let mut source = vec![0u8; rows * row_bytes];
    let mut file = File::open(std::env::var("QWR_TARGET_GGUF_SMOKE_PATH").unwrap()).unwrap();
    file.seek(SeekFrom::Start(
        model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap(),
    ))
    .unwrap();
    file.read_exact(&mut source).unwrap();
    let direct = GgmlQuantizedMatrix::from_bytes(&source, GgmlQType::Q5K, width, rows).unwrap();
    let affine =
        crate::GgmlAffineMatrix::from_ggml_bytes(&source, GgmlQType::Q5K, width, rows).unwrap();
    let transcode = affine.transcode_stats();
    eprintln!(
        "Q5_K affine prototype transcode={:?} source={} resident={} peak_active={}",
        transcode.elapsed,
        transcode.source_bytes,
        transcode.resident_bytes,
        transcode.peak_active_bytes,
    );

    for m in [1usize, 3] {
        let values: Vec<f32> = (0..m * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&values, &[m as i32, width as i32]);
        let custom = raw_f32(
            direct
                .forward(input.as_ref().unwrap())
                .unwrap()
                .as_ref()
                .unwrap(),
        );
        let upstream = raw_f32(
            affine
                .forward(input.as_ref().unwrap())
                .unwrap()
                .as_ref()
                .unwrap(),
        );
        let max_ulp = custom
            .iter()
            .zip(&upstream)
            .map(|(&left, &right)| ulp_distance(left, right))
            .max()
            .unwrap();
        let max_abs = custom
            .iter()
            .zip(&upstream)
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Q5_K affine parity M={m} max_ulp={max_ulp} max_abs={max_abs}");
    }

    for m in [1usize, 3, 288, 2048] {
        let values: Vec<f32> = (0..m * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&values, &[m as i32, width as i32]);
        crate::eval(input.as_ref().unwrap());
        let custom = median_forward(|| direct.forward(input.as_ref().unwrap()).unwrap());
        let upstream = median_forward(|| affine.forward(input.as_ref().unwrap()).unwrap());
        let stats = affine.dispatch_stats(m).unwrap();
        let flops = stats.floating_point_operations as f64;
        let gbytes = (stats.packed_bytes_read + stats.activation_bytes_read) as f64 / 1e9;
        eprintln!(
            "Q5_K affine prototype M={m} custom={custom:?} {:.3}TF upstream={upstream:?} {:.3}TF {:.3}GB/s speedup={:.3}x",
            flops / custom.as_secs_f64() / 1e12,
            flops / upstream.as_secs_f64() / 1e12,
            gbytes / upstream.as_secs_f64(),
            custom.as_secs_f64() / upstream.as_secs_f64(),
        );
    }
}

#[test]
#[ignore = "requires verified local target and MTP GGUF artifacts"]
fn actual_affine_repack_matches_reference_for_supported_qtypes() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    if !crate::metal_is_available() {
        return;
    }
    let structure: serde_json::Value = serde_json::from_reader(
        File::open(std::env::var("QWR_GGUF_STRUCTURE_PATH").unwrap()).unwrap(),
    )
    .unwrap();
    let supported = [
        GgmlQType::Q8_0,
        GgmlQType::Q3K,
        GgmlQType::Q4K,
        GgmlQType::Q5K,
        GgmlQType::Iq4Nl,
        GgmlQType::Iq3S,
        GgmlQType::Iq4Xs,
    ];
    for (model_key, path_variable) in [
        ("target", "QWR_TARGET_GGUF_SMOKE_PATH"),
        ("mtp", "QWR_MTP_GGUF_SMOKE_PATH"),
    ] {
        let model = &structure[model_key];
        let mut file = File::open(std::env::var(path_variable).unwrap()).unwrap();
        for &qtype in &supported {
            let Some(tensor) = model["tensor_infos_enriched"]
                .as_array()
                .unwrap()
                .iter()
                .find(|tensor| {
                    tensor["type"].as_u64() == Some(u64::from(qtype.id()))
                        && tensor["shape"]
                            .as_array()
                            .is_some_and(|shape| shape.len() == 2)
                        && tensor["shape"][1].as_u64().is_some_and(|rows| rows >= 3)
                })
            else {
                continue;
            };
            let width = tensor["shape"][0].as_u64().unwrap() as usize;
            let rows = 3usize;
            let row_bytes = width / qtype.block_elements() * qtype.block_bytes();
            let mut source = vec![0u8; row_bytes * rows];
            file.seek(SeekFrom::Start(
                model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap(),
            ))
            .unwrap();
            file.read_exact(&mut source).unwrap();
            let direct = GgmlQuantizedMatrix::from_bytes(&source, qtype, width, rows).unwrap();
            let mut released = Vec::new();
            let affine = crate::GgmlAffineMatrix::from_ggml_bytes_with_progress(
                &source,
                qtype,
                width,
                rows,
                |range| released.push(range),
            )
            .unwrap();
            assert_eq!(released, [0..source.len()]);
            let weights: Vec<Vec<f32>> = (0..rows)
                .map(|row| {
                    decode_row(
                        qtype,
                        &source[row * row_bytes..(row + 1) * row_bytes],
                        width,
                    )
                })
                .collect();
            let activation: Vec<f32> = (0..width)
                .map(|column| (column as i32 % 29 - 14) as f32 * 0.00390625)
                .collect();
            let input_m1 = crate::from_slice_f32(&activation, &[1, width as i32]);
            let direct_m1 = raw_f32(
                direct
                    .forward(input_m1.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            let affine_m1 = raw_f32(
                affine
                    .forward(input_m1.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            for output in 0..rows {
                let expected = activation
                    .iter()
                    .zip(&weights[output])
                    .map(|(input, weight)| input * weight)
                    .sum::<f32>();
                let tolerance = 0.003 + expected.abs() * 0.00005;
                assert!((direct_m1[output] - expected).abs() <= tolerance);
                assert!((affine_m1[output] - expected).abs() <= tolerance);
            }
            let mut verify_values = Vec::with_capacity(3 * width);
            for row in 0..3 {
                verify_values.extend(
                    activation
                        .iter()
                        .map(|value| value + row as f32 * 0.0009765625),
                );
            }
            let input_m3 = crate::from_slice_f32(&verify_values, &[3, width as i32]);
            let affine_m3 = raw_f32(
                affine
                    .forward(input_m3.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            let corresponding_ulp = affine_m1
                .iter()
                .zip(&affine_m3[..rows])
                .map(|(&left, &right)| ulp_distance(left, right))
                .max()
                .unwrap();
            eprintln!(
                "actual affine {model_key} {qtype:?} width={width} direct_affine_max_ulp={} corresponding_ulp={corresponding_ulp} resident={} source={}",
                direct_m1
                    .iter()
                    .zip(&affine_m1)
                    .map(|(&left, &right)| ulp_distance(left, right))
                    .max()
                    .unwrap(),
                affine.transcode_stats().resident_bytes,
                source.len(),
            );
            assert!(corresponding_ulp <= 1);

            let selected = affine.select_rows(&[2..3, 0..1]).unwrap();
            let selected_values = raw_f32(
                selected
                    .forward(input_m3.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            for input_row in 0..3 {
                assert!(
                    ulp_distance(selected_values[input_row * 2], affine_m3[input_row * 3 + 2]) <= 1
                );
                assert!(
                    ulp_distance(selected_values[input_row * 2 + 1], affine_m3[input_row * 3],)
                        <= 1
                );
            }
            let embedding =
                crate::GgmlAffineEmbedding::from_ggml_bytes(&source, qtype, width, rows).unwrap();
            let indices = crate::from_slice_i32(&[2, 0, 1], &[3]);
            let embedded = raw_f32(
                embedding
                    .forward(indices.as_ref().unwrap())
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            for (selected_row, source_row) in [2usize, 0, 1].into_iter().enumerate() {
                for column in 0..width {
                    assert!(
                        ulp_distance(
                            embedded[selected_row * width + column],
                            weights[source_row][column],
                        ) <= 1
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires verified local target GGUF artifact"]
fn actual_q6_k_contains_non_affine_group32() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let structure: serde_json::Value = serde_json::from_reader(
        File::open(std::env::var("QWR_GGUF_STRUCTURE_PATH").unwrap()).unwrap(),
    )
    .unwrap();
    let model = &structure["target"];
    let tensor = model["tensor_infos_enriched"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| {
            tensor["type"].as_u64() == Some(14) && tensor["shape"][0].as_u64() == Some(5120)
        })
        .unwrap();
    let row_bytes = 5120 / 256 * 210;
    let mut source = vec![0u8; row_bytes * 8];
    let mut file = File::open(std::env::var("QWR_TARGET_GGUF_SMOKE_PATH").unwrap()).unwrap();
    file.seek(SeekFrom::Start(
        model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap(),
    ))
    .unwrap();
    file.read_exact(&mut source).unwrap();
    let mut widest = 0i32;
    for block in source.chunks_exact(210) {
        for group32 in 0..8 {
            let mut minimum = i32::MAX;
            let mut maximum = i32::MIN;
            for group_column in 0..32 {
                let column = group32 * 32 + group_column;
                let half = column / 128;
                let quadrant = (column % 128) / 32;
                let lane = column % 32;
                let low_packed = block[half * 64 + lane + (quadrant & 1) * 32];
                let low = if quadrant < 2 {
                    low_packed & 15
                } else {
                    low_packed >> 4
                };
                let high = (block[128 + half * 32 + lane] >> (2 * quadrant)) & 3;
                let quant = i32::from(low | (high << 4)) - 32;
                let scale = block[192 + half * 8 + lane / 16 + quadrant * 2] as i8 as i32;
                let coefficient = scale * quant;
                minimum = minimum.min(coefficient);
                maximum = maximum.max(coefficient);
            }
            widest = widest.max(maximum - minimum);
        }
    }
    eprintln!("actual Q6_K widest group32 integer span={widest}");
    assert!(
        widest > 255,
        "actual Q6_K unexpectedly fits an UINT8 affine group"
    );
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return sign
            | if magnitude == 0x7f80_0000 {
                0x7c00
            } else {
                0x7e00
            };
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
    let rounded = significand + ((1u32 << (shift - 1)) - 1) + ((significand >> shift) & 1);
    sign | (rounded >> shift) as u16
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn dense_q6_bytes(source: &[u8], width: usize, rows: usize, bf16: bool) -> Vec<u8> {
    let source_row_bytes = width / 256 * 210;
    assert_eq!(source.len(), source_row_bytes * rows);
    let mut output = Vec::with_capacity(rows * width * 2);
    for row in source.chunks_exact(source_row_bytes) {
        for block in row.chunks_exact(210) {
            for value in decode_block(GgmlQType::Q6K, block) {
                let bits = if bf16 {
                    f32_to_bf16_bits(value)
                } else {
                    f32_to_f16_bits(value)
                };
                output.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    output
}

#[test]
#[ignore = "requires verified local target GGUF artifact"]
fn prototype_fixed_q6_ssm_out_dense_prefill() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    if !crate::metal_is_available() {
        return;
    }
    let structure: serde_json::Value = serde_json::from_reader(
        File::open(std::env::var("QWR_GGUF_STRUCTURE_PATH").unwrap()).unwrap(),
    )
    .unwrap();
    let model = &structure["target"];
    let tensor = model["tensor_infos_enriched"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| tensor["name"].as_str() == Some("blk.1.ssm_out.weight"))
        .unwrap();
    assert_eq!(tensor["type"].as_u64(), Some(14));
    assert_eq!(tensor["shape"], serde_json::json!([6144, 5120]));
    let width = 6144usize;
    let rows = 1024usize;
    let source_row_bytes = width / 256 * 210;
    let mut source = vec![0u8; source_row_bytes * rows];
    let mut file = File::open(std::env::var("QWR_TARGET_GGUF_SMOKE_PATH").unwrap()).unwrap();
    file.seek(SeekFrom::Start(
        model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap(),
    ))
    .unwrap();
    file.read_exact(&mut source).unwrap();
    let direct = GgmlQuantizedMatrix::from_bytes(&source, GgmlQType::Q6K, width, rows).unwrap();

    let started = Instant::now();
    let f16_bytes = dense_q6_bytes(&source, width, rows, false);
    let f16_time = started.elapsed();
    let started = Instant::now();
    let bf16_bytes = dense_q6_bytes(&source, width, rows, true);
    let bf16_time = started.elapsed();
    let f16 = crate::from_bytes_f16(&f16_bytes, &[rows as i32, width as i32], false);
    let bf16 = crate::from_bytes_f16(&bf16_bytes, &[rows as i32, width as i32], true);
    crate::eval(f16.as_ref().unwrap());
    crate::eval(bf16.as_ref().unwrap());
    eprintln!(
        "fixed Q6 dense transcode F16={f16_time:?} BF16={bf16_time:?} source={} dense={}",
        source.len(),
        f16_bytes.len(),
    );

    let original = decode_row(GgmlQType::Q6K, &source[..source_row_bytes], width);
    let f16_row: Vec<f32> = f16_bytes[..width * 2]
        .chunks_exact(2)
        .map(|bytes| half_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    let bf16_row: Vec<f32> = bf16_bytes[..width * 2]
        .chunks_exact(2)
        .map(|bytes| f32::from_bits((u16::from_le_bytes(bytes.try_into().unwrap()) as u32) << 16))
        .collect();
    for (name, dense) in [("F16", &f16_row), ("BF16", &bf16_row)] {
        let max_abs = original
            .iter()
            .zip(dense)
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        let rms = (original
            .iter()
            .zip(dense)
            .map(|(&left, &right)| (left - right).powi(2))
            .sum::<f32>()
            / width as f32)
            .sqrt();
        eprintln!("fixed Q6 dense {name} weight max_abs={max_abs} rms={rms}");
    }

    for m in [1usize, 3] {
        let values: Vec<f32> = (0..m * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&values, &[m as i32, width as i32]);
        let direct_values = raw_f32(
            direct
                .forward(input.as_ref().unwrap())
                .unwrap()
                .as_ref()
                .unwrap(),
        );
        for (name, dense) in [("F16", &f16), ("BF16", &bf16)] {
            let transposed = crate::transpose(dense.as_ref().unwrap());
            let output = crate::matmul(input.as_ref().unwrap(), transposed.as_ref().unwrap());
            let output = raw_f32(output.as_ref().unwrap());
            let max_abs = direct_values
                .iter()
                .zip(&output)
                .map(|(&left, &right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            eprintln!("fixed Q6 dense {name} output M={m} max_abs={max_abs}");
        }
    }

    for m in [1usize, 3, 245, 288, 2048] {
        let values: Vec<f32> = (0..m * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&values, &[m as i32, width as i32]);
        crate::eval(input.as_ref().unwrap());
        let custom = median_forward(|| direct.forward(input.as_ref().unwrap()).unwrap());
        let flops = 2.0 * m as f64 * rows as f64 * width as f64;
        for (name, dense) in [("F16", &f16), ("BF16", &bf16)] {
            let dense_time = median_forward(|| {
                let transposed = crate::transpose(dense.as_ref().unwrap());
                crate::matmul(input.as_ref().unwrap(), transposed.as_ref().unwrap())
            });
            eprintln!(
                "fixed Q6 dense {name} M={m} custom={custom:?} {:.3}TF dense={dense_time:?} {:.3}TF speedup={:.3}x",
                flops / custom.as_secs_f64() / 1e12,
                flops / dense_time.as_secs_f64() / 1e12,
                custom.as_secs_f64() / dense_time.as_secs_f64(),
            );
        }
    }
}

#[test]
#[ignore = "requires verified local target GGUF artifact"]
fn actual_fixed_q6_dual_dispatch_is_bounded_and_deterministic() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    if !crate::metal_is_available() {
        return;
    }
    let structure: serde_json::Value = serde_json::from_reader(
        File::open(std::env::var("QWR_GGUF_STRUCTURE_PATH").unwrap()).unwrap(),
    )
    .unwrap();
    let model = &structure["target"];
    let tensor = model["tensor_infos_enriched"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| tensor["name"].as_str() == Some("blk.1.ssm_out.weight"))
        .unwrap();
    let width = 6144usize;
    let rows = 5120usize;
    let source_row_bytes = width / 256 * 210;
    let mut source = vec![0u8; source_row_bytes * rows];
    let mut file = File::open(std::env::var("QWR_TARGET_GGUF_SMOKE_PATH").unwrap()).unwrap();
    file.seek(SeekFrom::Start(
        model["data_start"].as_u64().unwrap() + tensor["offset"].as_u64().unwrap(),
    ))
    .unwrap();
    file.read_exact(&mut source).unwrap();
    let direct = GgmlQuantizedMatrix::from_bytes(&source, GgmlQType::Q6K, width, rows).unwrap();
    let mut released = Vec::new();
    let dual = crate::Qwen38Q6DualMatrix::from_pinned_bytes_with_progress(
        &source,
        crate::Qwen38Q6Shape::K6144N5120,
        |range| released.push(range),
    )
    .unwrap();
    assert_eq!(released.first().unwrap().start, 0);
    assert_eq!(released.last().unwrap().end, source.len());
    assert!(released.windows(2).all(|pair| pair[0].end == pair[1].start));
    let stats = dual.transcode_stats();
    assert_eq!(stats.source_bytes, source.len());
    assert_eq!(stats.dense_bytes, rows * width * 2);
    assert_eq!(stats.resident_bytes, stats.source_bytes + stats.dense_bytes);
    assert!(stats.peak_active_bytes < 500 * 1024 * 1024);

    for m in [1usize, 3] {
        let values: Vec<f32> = (0..m * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&values, &[m as i32, width as i32]);
        let expected = raw_f32(
            direct
                .forward(input.as_ref().unwrap())
                .unwrap()
                .as_ref()
                .unwrap(),
        );
        let actual = raw_f32(
            dual.forward(input.as_ref().unwrap())
                .unwrap()
                .as_ref()
                .unwrap(),
        );
        assert_eq!(actual, expected);
    }

    let values: Vec<f32> = (0..245 * width)
        .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
        .collect();
    let input = crate::from_slice_f32(&values, &[245, width as i32]);
    let expected = raw_f32(
        direct
            .forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    let first = raw_f32(
        dual.forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    let second = raw_f32(
        dual.forward(input.as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap(),
    );
    assert_eq!(first, second);
    let max_abs = expected
        .iter()
        .zip(&first)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    eprintln!("fixed Q6 dual full-shape M=245 max_abs={max_abs}");
    assert!(max_abs <= 0.0005);

    for m in [245usize, 288, 2048] {
        let values: Vec<f32> = (0..m * width)
            .map(|index| (index as i32 % 31 - 15) as f32 * 0.00390625)
            .collect();
        let input = crate::from_slice_f32(&values, &[m as i32, width as i32]);
        crate::eval(input.as_ref().unwrap());
        let direct_time = median_forward(|| direct.forward(input.as_ref().unwrap()).unwrap());
        let dual_time = median_forward(|| dual.forward(input.as_ref().unwrap()).unwrap());
        let flops = dual.dispatch_stats(m).unwrap().floating_point_operations as f64;
        eprintln!(
            "fixed Q6 dual M={m} direct={direct_time:?} {:.3}TF F16={dual_time:?} {:.3}TF speedup={:.3}x",
            flops / direct_time.as_secs_f64() / 1e12,
            flops / dual_time.as_secs_f64() / 1e12,
            direct_time.as_secs_f64() / dual_time.as_secs_f64(),
        );
    }
}
