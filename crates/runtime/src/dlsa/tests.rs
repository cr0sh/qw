// SPDX-License-Identifier: Apache-2.0
//! Tests with a tiny synthetic torch archive (zip of stored entries + protocol-2 pickle).

use super::*;

fn push_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn pickle_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'X');
    push_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

/// Build what `torch.save({name: float32 [rows, cols] tensor})` writes.
fn build_pt(tensors: &[(&str, usize, usize, Vec<f32>)]) -> Vec<u8> {
    let mut p = vec![0x80, 2, b'}'];
    p.extend_from_slice(b"q\x00(");
    for (i, (name, rows, cols, _)) in tensors.iter().enumerate() {
        pickle_str(&mut p, name);
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n(");
        p.push(b'(');
        pickle_str(&mut p, "storage");
        p.extend_from_slice(b"ctorch\nFloatStorage\n");
        pickle_str(&mut p, &i.to_string());
        pickle_str(&mut p, "cpu");
        p.push(b'M');
        push_u16(&mut p, (rows * cols) as u16);
        p.extend_from_slice(b"tQ"); // TUPLE, BINPERSID
        p.extend_from_slice(&[b'K', 0, b'K', *rows as u8, b'K', *cols as u8, 0x86]);
        p.extend_from_slice(&[b'K', *cols as u8, b'K', 1, 0x86, 0x89]);
        p.extend_from_slice(b"ccollections\nOrderedDict\n)Rt"); // OrderedDict(), TUPLE
        p.push(b'R');
    }
    p.extend_from_slice(b"u.");
    let mut files: Vec<(String, Vec<u8>)> = vec![
        ("arc/data.pkl".into(), p),
        ("arc/byteorder".into(), b"little".to_vec()),
    ];
    for (i, (_, _, _, vals)) in tensors.iter().enumerate() {
        let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        files.push((format!("arc/data/{i}"), bytes));
    }
    zip_stored(&files)
}

fn zip_stored(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut zip = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let offset = zip.len() as u32;
        push_u32(&mut zip, 0x0403_4b50);
        push_u16(&mut zip, 20);
        push_u16(&mut zip, 0);
        push_u16(&mut zip, 0);
        push_u32(&mut zip, 0);
        push_u32(&mut zip, 0);
        push_u32(&mut zip, data.len() as u32);
        push_u32(&mut zip, data.len() as u32);
        push_u16(&mut zip, name.len() as u16);
        push_u16(&mut zip, 0);
        zip.extend_from_slice(name.as_bytes());
        zip.extend_from_slice(data);

        push_u32(&mut central, 0x0201_4b50);
        push_u16(&mut central, 20);
        push_u16(&mut central, 20);
        push_u16(&mut central, 0);
        push_u16(&mut central, 0);
        push_u32(&mut central, 0);
        push_u32(&mut central, 0);
        push_u32(&mut central, data.len() as u32);
        push_u32(&mut central, data.len() as u32);
        push_u16(&mut central, name.len() as u16);
        push_u16(&mut central, 0);
        push_u16(&mut central, 0);
        push_u32(&mut central, 0);
        push_u32(&mut central, 0);
        push_u32(&mut central, offset);
        central.extend_from_slice(name.as_bytes());
    }
    let cd_offset = zip.len() as u32;
    zip.extend_from_slice(&central);
    push_u32(&mut zip, 0x0605_4b50);
    push_u16(&mut zip, 0);
    push_u16(&mut zip, 0);
    push_u16(&mut zip, files.len() as u16);
    push_u16(&mut zip, files.len() as u16);
    push_u32(&mut zip, central.len() as u32);
    push_u32(&mut zip, cd_offset);
    push_u16(&mut zip, 0);
    zip
}

fn ramp(rows: usize, base: f32) -> Vec<f32> {
    (0..rows * INDEX_DIM)
        .map(|i| base + i as f32 * 0.25)
        .collect()
}

/// Tiny map: layers 3 and 7, input_dim 8.
fn tiny_pt() -> Vec<u8> {
    build_pt(&[
        ("3.W_Q_idx", 8, INDEX_DIM, ramp(8, 0.0)),
        ("3.W_K_idx", 8, INDEX_DIM, ramp(8, 1.0)),
        ("7.W_Q_idx", 8, INDEX_DIM, ramp(8, 2.0)),
        ("7.W_K_idx", 8, INDEX_DIM, ramp(8, 3.0)),
    ])
}

fn layout(layers: &[usize], input_dim: usize) -> DlsaLayout {
    DlsaLayout {
        full_attention_layers: layers.to_vec(),
        input_dim,
    }
}

fn f32_values(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr32 = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr32);
    mlxcel_core::array_to_raw_bytes(&arr32)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn reads_synthetic_torch_archive() {
    let t = read_f32_tensor_dict(&tiny_pt()).unwrap();
    assert_eq!(t.len(), 4);
    assert_eq!(t[0].0, "3.W_Q_idx");
    assert_eq!(t[0].1.shape, vec![8, INDEX_DIM]);
    assert_eq!(&t[1].1.bytes[..4], &1.0f32.to_le_bytes());
}

#[test]
fn rejects_non_torch_input() {
    assert!(read_f32_tensor_dict(b"hello").is_err());
    let mut bad = tiny_pt();
    let n = bad.len();
    bad.truncate(n - 5);
    assert!(read_f32_tensor_dict(&bad).is_err());
}

#[test]
fn conversion_is_deterministic_and_carries_metadata() {
    let pt = tiny_pt();
    let a = convert_pt_to_safetensors(&pt, "tiny.pt").unwrap();
    let b = convert_pt_to_safetensors(&pt, "tiny.pt").unwrap();
    assert_eq!(sha256_hex(&a), sha256_hex(&b));
    let (_, header) = safetensors::SafeTensors::read_metadata(&a).unwrap();
    let md: serde_json::Value =
        serde_json::from_str(&header.metadata().as_ref().unwrap()["qw_dlsa_map"]).unwrap();
    assert_eq!(md["source_sha256"], sha256_hex(&pt));
    assert_eq!(md["layers"], serde_json::json!([3, 7]));
    assert_eq!(md["input_dim"], 8);
    assert_eq!(md["upstream_commit"], UPSTREAM_COMMIT);
}

#[test]
fn loader_roundtrip_f32_and_bf16() {
    let file = convert_pt_to_safetensors(&tiny_pt(), "tiny.pt").unwrap();
    let l = layout(&[3, 7], 8);
    let maps = DlsaMaps::from_bytes(&file, &l, DlsaMapDtype::F32).unwrap();
    let k7 = f32_values(&maps.layer(7).unwrap().w_k_idx);
    assert_eq!(k7, ramp(8, 3.0));
    assert_eq!(
        mlxcel_core::array_shape(&maps.layer(3).unwrap().w_q_idx),
        vec![8, INDEX_DIM as i32]
    );
    let bf = DlsaMaps::from_bytes(&file, &l, DlsaMapDtype::Bf16).unwrap();
    let w = &bf.layer(3).unwrap().w_q_idx;
    assert_eq!(mlxcel_core::array_dtype(w), mlxcel_core::dtype::BFLOAT16);
    // bf16 keeps 8 significant bits: relative error below 2^-8.
    for (got, want) in f32_values(w).into_iter().zip(ramp(8, 0.0)) {
        assert!((got - want).abs() <= want.abs() / 128.0, "{got} vs {want}");
    }
}

#[test]
fn loader_rejects_layout_mismatches() {
    let file = convert_pt_to_safetensors(&tiny_pt(), "tiny.pt").unwrap();
    let err = |l: DlsaLayout| {
        DlsaMaps::from_bytes(&file, &l, DlsaMapDtype::F32)
            .err()
            .expect("must fail")
            .to_string()
    };
    assert!(err(layout(&[3, 7, 11], 8)).contains("missing"));
    assert!(err(layout(&[3], 8)).contains("unexpected"));
    assert!(err(layout(&[3, 7], 16)).contains("shape"));
}

#[test]
fn loader_rejects_foreign_safetensors() {
    let t =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &[0; 4]).unwrap();
    let bytes = safetensors::serialize([("x", t)], None).unwrap();
    let err = DlsaMaps::from_bytes(&bytes, &layout(&[3], 8), DlsaMapDtype::F32)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("not a QW D-LSA map"), "{err}");
}

#[test]
fn conversion_rejects_wrong_index_dim() {
    let pt = build_pt(&[
        ("3.W_Q_idx", 4, 8, vec![0.0; 32]),
        ("3.W_K_idx", 4, 8, vec![0.0; 32]),
    ]);
    assert!(convert_pt_to_safetensors(&pt, "x.pt").is_err());
}

#[test]
fn layout_from_qwen35_config() {
    let config: crate::qwen3_5::Qwen35Config = serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_5_text", "hidden_size": 5120, "vocab_size": 1000, "num_hidden_layers": 64,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        "full_attention_interval": 4,
        "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25}
    }))
    .unwrap();
    let l = DlsaLayout::from_qwen35_config(&config).unwrap();
    assert_eq!(l.input_dim, 768);
    assert_eq!(
        l.full_attention_layers,
        (0..16).map(|i| 4 * i + 3).collect::<Vec<_>>()
    );
}

/// Real released files. Needs the pinned assets: `qw download --dlsa-map --dlsa-pca`, or set
/// `QW_DLSA_SOURCE_DIR` to a directory holding the upstream `.pt` files.
#[test]
#[ignore = "needs the downloaded upstream assets"]
fn real_maps_convert_to_pinned_checksums_and_load() {
    let dir = std::env::var_os("QW_DLSA_SOURCE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/qw/dlsa")
        });
    let l = layout(&(0..16).map(|i| 4 * i + 3).collect::<Vec<_>>(), 768);
    for kind in [DlsaMapKind::Distilled, DlsaMapKind::PcaStep0] {
        let asset = kind.asset();
        let pt = std::fs::read(dir.join(asset.source_name)).unwrap();
        let out =
            std::env::temp_dir().join(format!("qw-dlsa-real-{}.safetensors", std::process::id()));
        install_dlsa_map(&asset, &pt, &out).unwrap();
        verify_converted_map(&out, kind).unwrap();
        let maps = DlsaMaps::load(&out, &l, DlsaMapDtype::Bf16).unwrap();
        assert_eq!(maps.layers.len(), 16);
        let q = f32_values(&maps.layer(63).unwrap().w_q_idx);
        assert_eq!(q.len(), 768 * INDEX_DIM);
        assert!(q.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_file(out);
    }
}

fn archive_with_pickle(pickle: Vec<u8>) -> Vec<u8> {
    zip_stored(&[
        ("arc/data.pkl".into(), pickle),
        ("arc/data/0".into(), vec![0; 24]),
    ])
}

/// Pickle of `{"a": _rebuild_tensor_v2(storage "0", 0, <shape>, <stride>, False, OrderedDict())}`.
fn one_tensor_pickle(shape: &[u8], stride: &[u8]) -> Vec<u8> {
    let mut p = vec![0x80, 2, b'}', b'('];
    pickle_str(&mut p, "a");
    p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
    pickle_str(&mut p, "storage");
    p.extend_from_slice(b"ctorch\nFloatStorage\n");
    pickle_str(&mut p, "0");
    pickle_str(&mut p, "cpu");
    p.extend_from_slice(&[b'K', 6, b't', b'Q', b'K', 0]);
    p.extend_from_slice(shape);
    p.extend_from_slice(stride);
    p.extend_from_slice(b"\x89ccollections\nOrderedDict\n)Rt");
    p.extend_from_slice(b"Ru.");
    p
}

fn int4(v: i32) -> [u8; 5] {
    let b = v.to_le_bytes();
    [b'J', b[0], b[1], b[2], b[3]]
}

#[test]
fn shape_product_overflow_is_an_error() {
    let mut shape = Vec::new();
    for _ in 0..3 {
        shape.extend_from_slice(&int4(i32::MAX));
    }
    shape.push(0x87);
    let mut stride = Vec::new();
    for _ in 0..3 {
        stride.extend_from_slice(&int4(1));
    }
    stride.push(0x87);
    let err = read_f32_tensor_dict(&archive_with_pickle(one_tensor_pickle(&shape, &stride)))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("absurd") || err.contains("contiguous"),
        "{err}"
    );
    // sanity: a valid one-tensor pickle of this shape works
    let ok = one_tensor_pickle(&[b'K', 2, b'K', 3, 0x86], &[b'K', 3, b'K', 1, 0x86]);
    assert!(read_f32_tensor_dict(&archive_with_pickle(ok)).is_ok());
}

#[test]
fn memo_amplification_is_rejected() {
    // t = (); repeat: t = (t, t) with memo get/put: unbounded growth if tuples were deep-cloned.
    let mut p = vec![0x80, 2, b')', b'q', 0];
    for _ in 0..64 {
        p.extend_from_slice(&[b'h', 0, b'h', 0, 0x86, b'q', 0]);
    }
    p.push(b'.');
    let err = read_f32_tensor_dict(&archive_with_pickle(p))
        .unwrap_err()
        .to_string();
    assert!(err.contains("nested too deeply"), "{err}");
}

#[test]
fn memo_reference_to_dict_is_rejected() {
    let p = vec![0x80, 2, b'}', b'q', 0, b'h', 0, b'.'];
    assert!(read_f32_tensor_dict(&archive_with_pickle(p)).is_err());
}

#[test]
fn nested_dict_values_are_rejected_at_insertion() {
    // {"k": {}} built with SETITEM.
    let p = b"\x80\x02}X\x01\x00\x00\x00k}s.".to_vec();
    let err = read_f32_tensor_dict(&archive_with_pickle(p))
        .unwrap_err()
        .to_string();
    assert!(err.contains("strings to tensors"), "{err}");
}
