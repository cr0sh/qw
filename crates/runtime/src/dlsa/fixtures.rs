// SPDX-License-Identifier: Apache-2.0
//! Test-only reader of the golden fixtures in `tests/fixtures/dlsa/` (see its README.md).

use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use super::config::{DlsaGeometry, DlsaIndexConfig};
use super::convert::sha256_hex;

pub const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/dlsa/");

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, mut f) => {
            // Subnormal: normalize
            let mut e: i32 = 127 - 15 + 1;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// One fixture case: its safetensors bytes (sha256-checked against `manifest.json`) and its
/// manifest entry.
pub struct Fixture {
    pub name: String,
    bytes: Vec<u8>,
    pub manifest: Value,
}

impl Fixture {
    pub fn load(name: &str) -> Self {
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(format!("{FIXTURE_DIR}manifest.json")).expect("read manifest.json"),
        )
        .expect("parse manifest.json");
        let entry = manifest["cases"][name].clone();
        assert!(entry.is_object(), "no fixture case {name}");
        let file = entry["file"].as_str().unwrap();
        let bytes = std::fs::read(format!("{FIXTURE_DIR}{file}")).expect("read fixture");
        assert_eq!(
            sha256_hex(&bytes),
            entry["sha256"].as_str().unwrap(),
            "fixture {file} does not match manifest.json (stale or edited)"
        );
        Self {
            name: name.to_string(),
            bytes,
            manifest: entry,
        }
    }

    fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.bytes).expect("parse fixture")
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors().tensor(name).is_ok()
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.tensors()
            .tensor(name)
            .unwrap_or_else(|_| panic!("{}: no tensor {name}", self.name))
            .shape()
            .to_vec()
    }

    /// A float tensor upcast to f32 (F32, F16 or BF16).
    pub fn f32(&self, name: &str) -> Vec<f32> {
        let st = self.tensors();
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("{}: no tensor {name}", self.name));
        let data = t.data();
        match t.dtype() {
            Dtype::F32 => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::F16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            Dtype::BF16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
            other => panic!("{}: {name} is {other:?}, not a float", self.name),
        }
    }

    /// The raw bytes and dtype of a tensor (for MLX upload).
    pub fn raw(&self, name: &str) -> (Vec<u8>, Dtype) {
        let st = self.tensors();
        let t = st.tensor(name).unwrap();
        (t.data().to_vec(), t.dtype())
    }

    pub fn i32(&self, name: &str) -> Vec<i32> {
        let st = self.tensors();
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("{}: no tensor {name}", self.name));
        assert_eq!(t.dtype(), Dtype::I32, "{}: {name}", self.name);
        t.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }

    fn config_usize(&self, key: &str) -> usize {
        self.manifest["config"][key]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: config has no {key}", self.name)) as usize
    }

    pub fn cfg(&self) -> DlsaIndexConfig {
        DlsaIndexConfig {
            window_size: self.config_usize("window_size"),
            sink_size: self.config_usize("sink_size"),
            top_k: self.config_usize("top_k"),
            top_p: self.config_usize("top_p"),
            index_dim: self.config_usize("index_dim"),
            pool_size: self.config_usize("pool_size"),
            bucket_size: self.config_usize("bucket_size"),
            bucket_cap: self.config_usize("bucket_cap"),
        }
    }

    pub fn prefill_len(&self) -> Option<usize> {
        self.manifest["config"]["prefill_len"]
            .as_u64()
            .map(|v| v as usize)
    }

    pub fn key_count(&self) -> usize {
        self.config_usize("key_count")
    }

    pub fn geometry(&self) -> DlsaGeometry {
        let s = &self.manifest["shapes"];
        let get = |k: &str| s[k].as_u64().unwrap() as usize;
        DlsaGeometry {
            query_heads: get("query_heads"),
            kv_heads: get("kv_heads"),
            head_dim: get("head_dim"),
            rotary_dim: get("rotary_dim"),
        }
    }
}

#[test]
fn f16_decoding_matches_known_values() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x7bff), 65504.0);
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert!(f16_to_f32(0x7c00).is_infinite());
}
