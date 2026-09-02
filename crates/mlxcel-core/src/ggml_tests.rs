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
                let scale_low = if group < 8 { block[96 + group] & 0x0f } else { block[88 + group] >> 4 };
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
                let mut quant = if group.is_multiple_of(2) { packed & 0x0f } else { packed >> 4 };
                if qtype == GgmlQType::Q5K && block[16 + group_column] & (1 << group) != 0 { quant += 16; }
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
                let low = if quadrant < 2 { low_packed & 0x0f } else { low_packed >> 4 };
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
                let grid_index = usize::from(block[2 + group * 8 + subblock * 2 + usize::from(lane >= 4)]) | (usize::from(high) << 8);
                let grid = IQ3S_GRID[grid_index];
                let magnitude = ((grid >> ((lane & 3) * 8)) & 0xff) as f32;
                let sign = if block[74 + group * 4 + subblock] & (1 << lane) != 0 { -1.0 } else { 1.0 };
                let scales = block[106 + group / 2];
                let scale = if group.is_multiple_of(2) { scales & 0x0f } else { scales >> 4 };
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
                let index = if group_column < 16 { packed & 0x0f } else { packed >> 4 };
                *value = d * scale as f32 * IQ4_VALUES[index as usize] as f32;
            }
        }
    }
    output
}

fn decode_row(qtype: GgmlQType, bytes: &[u8], elements: usize) -> Vec<f32> {
    assert!(elements.is_multiple_of(qtype.block_elements()));
    assert_eq!(bytes.len(), elements / qtype.block_elements() * qtype.block_bytes());
    bytes.chunks_exact(qtype.block_bytes()).flat_map(|block| decode_block(qtype, block)).collect()
}

fn put_half(block: &mut [u8], offset: usize, bits: u16) {
    block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

fn fixture_block(qtype: GgmlQType, seed: u32) -> Vec<u8> {
    if qtype == GgmlQType::F32 {
        return ((seed as i32 % 31 - 15) as f32 * 0.03125).to_le_bytes().to_vec();
    }
    let mut state = seed.wrapping_add(0x9e37_79b9);
    let mut block = vec![0u8; qtype.block_bytes()];
    for byte in &mut block {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    match qtype {
        GgmlQType::Q8_0 | GgmlQType::Iq4Nl | GgmlQType::Iq3S | GgmlQType::Iq4Xs => put_half(&mut block, 0, 0x2800),
        GgmlQType::Q3K => put_half(&mut block, 108, 0x2800),
        GgmlQType::Q4K | GgmlQType::Q5K => { put_half(&mut block, 0, 0x2800); put_half(&mut block, 2, 0x2000); }
        GgmlQType::Q6K => put_half(&mut block, 208, 0x2800),
        GgmlQType::F32 => unreachable!(),
    }
    block
}

fn fixture_matrix(qtype: GgmlQType, width: usize, rows: usize) -> Vec<u8> {
    let blocks_per_row = width / qtype.block_elements();
    let mut bytes = Vec::with_capacity(blocks_per_row * rows * qtype.block_bytes());
    for row in 0..rows {
        for block in 0..blocks_per_row { bytes.extend(fixture_block(qtype, (row * 19 + block * 7 + 1) as u32)); }
    }
    bytes
}

fn raw_f32(array: &MlxArray) -> Vec<f32> {
    crate::eval(array);
    crate::array_to_raw_bytes(array).chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect()
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    let ordered = |value: f32| { let bits = value.to_bits() as i32; if bits < 0 { i32::MIN - bits } else { bits } };
    ordered(left).abs_diff(ordered(right))
}

#[test]
fn fixed_qtype_table_matches_parsed_target_contract() {
    let actual = GgmlQType::TARGET_TYPES.map(|qtype| (qtype.id(), qtype.block_elements(), qtype.block_bytes()));
    assert_eq!(actual, [(0, 1, 4), (8, 32, 34), (11, 256, 110), (12, 256, 144), (13, 256, 176), (14, 256, 210), (20, 32, 18), (21, 256, 110), (23, 256, 136)]);
    assert!(GgmlQType::try_from(2).is_err());
    assert!(GgmlQType::try_from(17).is_err());
}

#[test]
fn load_boundary_rejects_invalid_qtypes_shapes_lengths_and_overflow() {
    assert!(matches!(GgmlQuantizedMatrix::from_bytes(&[], 2, 256, 1), Err(GgmlQuantError::UnsupportedQType(2))));
    assert!(matches!(GgmlQuantizedMatrix::from_bytes(&[], 11, 255, 1), Err(GgmlQuantError::UnalignedWidth { .. })));
    assert!(matches!(GgmlQuantizedMatrix::from_bytes(&[0; 109], 11, 256, 1), Err(GgmlQuantError::ByteLength { expected: 110, actual: 109 })));
    assert!(matches!(GgmlQuantizedMatrix::from_bytes(&[], 0, usize::MAX, 2), Err(GgmlQuantError::Overflow)));
}

#[test]
fn reference_layouts_pin_nibbles_high_bits_signs_and_table_indices() {
    let mut q8 = vec![0; 34]; put_half(&mut q8, 0, 0x3c00); q8[2] = (-128i8) as u8; q8[33] = 127;
    let decoded = decode_block(GgmlQType::Q8_0, &q8); assert_eq!((decoded[0], decoded[31]), (-128.0, 127.0));

    let mut q3 = vec![0; 110]; put_half(&mut q3, 108, 0x3c00); q3[96] = 0x11; q3[104] = 0x22; q3[32] = 2; q3[64] = 3;
    let decoded = decode_block(GgmlQType::Q3K, &q3); assert_eq!(decoded[0], -2.0); assert_eq!(decoded[128], -1.0);
    q3[0] = 0x11; let decoded = decode_block(GgmlQType::Q3K, &q3); assert_eq!(decoded[0], 2.0); assert_eq!(decoded[128], 3.0);

    let mut q4 = vec![0; 144]; put_half(&mut q4, 0, 0x3c00); q4[4] = 1; q4[5] = 2; q4[16] = 3 | (4 << 4);
    let decoded = decode_block(GgmlQType::Q4K, &q4); assert_eq!((decoded[0], decoded[32]), (3.0, 8.0));

    let mut q5 = vec![0; 176]; put_half(&mut q5, 0, 0x3c00); q5[4] = 1; q5[5] = 2; q5[16] = 0b11; q5[48] = 3 | (4 << 4);
    let decoded = decode_block(GgmlQType::Q5K, &q5); assert_eq!((decoded[0], decoded[32]), (19.0, 40.0));

    let mut q6 = vec![0; 210]; put_half(&mut q6, 208, 0x3c00); q6[0] = 0x21; q6[128] = 0b11_10_01_00; q6[192] = 2; q6[196] = (-3i8) as u8;
    let decoded = decode_block(GgmlQType::Q6K, &q6); assert_eq!(decoded[0], -62.0); assert_eq!(decoded[64], -6.0);

    let mut iq4 = vec![0; 18]; put_half(&mut iq4, 0, 0x3c00); iq4[2] = 0xf0;
    let decoded = decode_block(GgmlQType::Iq4Nl, &iq4); assert_eq!((decoded[0], decoded[16]), (-127.0, 113.0));

    let mut iq3 = vec![0; 110]; put_half(&mut iq3, 0, 0x3c00); iq3[66] = 1; iq3[74] = 1; iq3[106] = 2;
    let decoded = decode_block(GgmlQType::Iq3S, &iq3); let grid = IQ3S_GRID[256];
    assert_eq!(decoded[0], -5.0 * (grid & 0xff) as f32); assert_eq!(decoded[1], 5.0 * ((grid >> 8) & 0xff) as f32); assert_eq!(decoded[4], 5.0);

    let mut iq4xs = vec![0; 136]; put_half(&mut iq4xs, 0, 0x3c00); iq4xs[2] = 2; iq4xs[4] = 1; iq4xs[8] = 0xf0;
    let decoded = decode_block(GgmlQType::Iq4Xs, &iq4xs); assert_eq!((decoded[0], decoded[16]), (-127.0, 113.0));
}

#[test]
fn dispatch_stats_record_zero_workspace_and_shape_selected_traffic() {
    let qtype = GgmlQType::Q4K; let width = 512; let rows = 3; let packed = fixture_matrix(qtype, width, rows);
    let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype.id(), width, rows).unwrap();
    let decode = matrix.dispatch_stats(1).unwrap(); assert_eq!(decode.path, GgmlKernelPath::DecodeM1); assert_eq!(decode.packed_bytes_read, packed.len()); assert_eq!(decode.workspace_bytes, 0);
    let prefill = matrix.dispatch_stats(5).unwrap(); assert_eq!(prefill.path, GgmlKernelPath::PrefillRows4); assert_eq!(prefill.packed_bytes_read, packed.len() * 2); assert_eq!(prefill.workspace_bytes, 0);
    let embedding = GgmlQuantizedEmbedding::from_bytes(&packed, qtype.id(), width, rows).unwrap();
    let lookup = embedding.dispatch_stats(2).unwrap(); assert_eq!(lookup.path, GgmlKernelPath::Embedding); assert_eq!(lookup.packed_bytes_read, packed.len() / rows * 2); assert_eq!(lookup.workspace_bytes, 0);
}

#[test]
fn metal_matrix_and_embedding_match_reference_for_every_target_qtype() {
    if !crate::metal_is_available() { return; }
    for qtype in GgmlQType::TARGET_TYPES {
        let width = if qtype == GgmlQType::F32 { 33 } else { qtype.block_elements() * 2 };
        let output_rows = 3; let packed = fixture_matrix(qtype, width, output_rows);
        let matrix = GgmlQuantizedMatrix::from_bytes(&packed, qtype.id(), width, output_rows).unwrap();
        let weights: Vec<Vec<f32>> = (0..output_rows).map(|row| { let rb = matrix.row_bytes; decode_row(qtype, &packed[row * rb..(row + 1) * rb], width) }).collect();
        let activation: Vec<f32> = (0..width).map(|column| (column as i32 % 23 - 11) as f32 * 0.015625).collect();
        let input = crate::from_slice_f32(&activation, &[1, width as i32]);
        let decode = matrix.forward(input.as_ref().unwrap()).unwrap(); let decode_values = raw_f32(decode.as_ref().unwrap());
        for row in 0..output_rows {
            let expected = activation.iter().zip(&weights[row]).map(|(input, weight)| input * weight).sum::<f32>(); let actual = decode_values[row];
            let tolerance = 0.002 + expected.abs() * 0.00002;
            assert!((actual - expected).abs() <= tolerance, "{qtype:?} decode row {row}: actual {actual}, expected {expected}");
        }
        let mut prefill_input = Vec::with_capacity(width * 5);
        for row in 0..5 { prefill_input.extend(activation.iter().map(|value| value + row as f32 * 0.0078125)); }
        let input = crate::from_slice_f32(&prefill_input, &[5, width as i32]);
        let prefill = matrix.forward(input.as_ref().unwrap()).unwrap(); let prefill_values = raw_f32(prefill.as_ref().unwrap());
        for output in 0..output_rows { assert!(ulp_distance(decode_values[output], prefill_values[output]) <= 1, "{qtype:?} M=1/prefill mismatch: {} vs {}", decode_values[output], prefill_values[output]); }
        for input_row in 0..5 { for output in 0..output_rows {
            let expected = prefill_input[input_row * width..(input_row + 1) * width].iter().zip(&weights[output]).map(|(input, weight)| input * weight).sum::<f32>();
            let actual = prefill_values[input_row * output_rows + output]; let tolerance = 0.002 + expected.abs() * 0.00002;
            assert!((actual - expected).abs() <= tolerance, "{qtype:?} prefill input {input_row} output {output}: actual {actual}, expected {expected}");
        }}
        let embedding = GgmlQuantizedEmbedding::from_bytes(&packed, qtype.id(), width, output_rows).unwrap();
        let indices = crate::from_slice_i32(&[2, 0, 1], &[3]); let selected = embedding.forward(indices.as_ref().unwrap()).unwrap(); let selected = raw_f32(selected.as_ref().unwrap());
        for (selected_row, source_row) in [2usize, 0, 1].into_iter().enumerate() { for column in 0..width {
            let actual = selected[selected_row * width + column]; let expected = weights[source_row][column];
            assert!(ulp_distance(actual, expected) <= 1, "{qtype:?} embedding row {source_row} column {column}: actual {actual}, expected {expected}");
        }}
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
    let structure_path =
        std::env::var("QWR_GGUF_STRUCTURE_PATH").expect("QWR_GGUF_STRUCTURE_PATH");
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
            let input = crate::from_slice_f32(&activation, &[1, width as i32]);
            let matrix =
                GgmlQuantizedMatrix::from_bytes(&packed, qtype.id(), width, 1).unwrap();
            // First call compiles the specialization. Time a synchronized warm call.
            let cold = matrix.forward(input.as_ref().unwrap()).unwrap();
            let _ = raw_f32(cold.as_ref().unwrap());
            let started = Instant::now();
            let warm = matrix.forward(input.as_ref().unwrap()).unwrap();
            let actual = raw_f32(warm.as_ref().unwrap())[0];
            let elapsed = started.elapsed();
            let expected = activation
                .iter()
                .zip(&weights)
                .map(|(input, weight)| input * weight)
                .sum::<f32>();
            let tolerance = 0.003 + expected.abs() * 0.00005;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{model_key} {qtype:?} tensor {}: actual {actual}, expected {expected}",
                tensor["name"].as_str().unwrap()
            );

            let embedding =
                GgmlQuantizedEmbedding::from_bytes(&packed, qtype.id(), width, 1).unwrap();
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
