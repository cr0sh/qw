use std::collections::HashSet;

use mlxcel_core::generate::{ModelStateSnapshot, SnapshotPage};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::provider::PromptSnapshot;
use crate::qwen3_5_mtp::MtpPromptSnapshot;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePage {
    pub token_start: usize,
    pub token_end: usize,
    pub shape: Vec<i32>,
    pub dtype: i32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableArray {
    pub name: Option<String>,
    pub shape: Vec<i32>,
    pub dtype: i32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePagedTensor {
    pub name: String,
    pub token_axis: usize,
    pub token_len: usize,
    pub pages: Vec<PortablePage>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableModelState {
    pub family: String,
    pub token_len: usize,
    pub tensors: Vec<PortableArray>,
    pub paged_tensors: Vec<PortablePagedTensor>,
    pub continuation_logits: Option<PortableArray>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortablePromptSnapshot {
    Baseline(PortableModelState),
    Mtp {
        target: PortableModelState,
        draft_keys: Option<PortableArray>,
        draft_values: Option<PortableArray>,
        draft_offset: i32,
        last_hidden: PortableArray,
        continuation_logits: PortableArray,
    },
}

fn supported_dtype(dtype: i32) -> bool {
    matches!(
        dtype,
        mlxcel_core::dtype::UINT8
            | mlxcel_core::dtype::INT8
            | mlxcel_core::dtype::UINT32
            | mlxcel_core::dtype::UINT64
            | mlxcel_core::dtype::INT32
            | mlxcel_core::dtype::INT64
            | mlxcel_core::dtype::FLOAT16
            | mlxcel_core::dtype::FLOAT32
            | mlxcel_core::dtype::BFLOAT16
    )
}

pub(crate) fn array_to_portable(name: Option<String>, array: &MlxArray) -> PortableArray {
    PortableArray {
        name,
        shape: mlxcel_core::array_shape(array),
        dtype: mlxcel_core::array_dtype(array),
        bytes: mlxcel_core::array_to_raw_bytes(array),
    }
}

pub(crate) fn array_from_portable(
    array: PortableArray,
    expected_name: Option<&str>,
) -> Result<UniquePtr<MlxArray>, String> {
    match (expected_name, array.name.as_deref()) {
        (Some(expected), Some(actual)) if expected == actual => {}
        (None, None) => {}
        (Some(expected), actual) => {
            return Err(format!(
                "portable array name mismatch: expected {expected:?}, got {actual:?}"
            ));
        }
        (None, Some(actual)) => {
            return Err(format!("portable array unexpectedly named {actual:?}"));
        }
    }
    if array.shape.is_empty() || array.shape.iter().any(|&dimension| dimension <= 0) {
        return Err("portable array shape must contain only positive dimensions".to_string());
    }
    if !supported_dtype(array.dtype) {
        return Err(format!("unsupported portable array dtype {}", array.dtype));
    }
    let elements = array.shape.iter().try_fold(1usize, |count, &dimension| {
        count.checked_mul(dimension as usize)
    });
    let expected_bytes = elements
        .and_then(|count| count.checked_mul(mlxcel_core::dtype::size_bytes(array.dtype)?))
        .ok_or_else(|| "portable array byte length overflow".to_string())?;
    if array.bytes.len() != expected_bytes {
        return Err(format!(
            "portable array byte length mismatch: expected {expected_bytes}, got {}",
            array.bytes.len()
        ));
    }
    let restored = mlxcel_core::from_bytes(&array.bytes, &array.shape, array.dtype);
    mlxcel_core::eval(&restored);
    Ok(restored)
}

fn model_to_portable(snapshot: &ModelStateSnapshot) -> PortableModelState {
    let tensors = snapshot
        .tensor_names()
        .map(|name| {
            array_to_portable(
                Some(name.to_string()),
                snapshot
                    .tensor(name)
                    .expect("snapshot tensor name must resolve to its array"),
            )
        })
        .collect();
    PortableModelState {
        family: snapshot.family().to_string(),
        token_len: snapshot.token_len(),
        tensors,
        paged_tensors: snapshot
            .paged_tensor_names()
            .filter_map(|name| snapshot.paged_tensor(name).map(|tensor| PortablePagedTensor {
                name: name.to_string(),
                token_axis: tensor.token_axis(),
                token_len: tensor.token_len(),
                pages: tensor.pages().iter().map(|page| PortablePage {
                    token_start: page.token_range().start,
                    token_end: page.token_range().end,
                    shape: page.shape().to_vec(),
                    dtype: page.dtype(),
                    bytes: page.portable_bytes().to_vec(),
                }).collect(),
            }))
            .collect(),
        continuation_logits: snapshot.continuation_logits().map(|array| array_to_portable(None, array)),
    }
}

fn model_from_portable(
    portable: PortableModelState,
    require_continuation_logits: bool,
) -> Result<ModelStateSnapshot, String> {
    if portable.family.is_empty() {
        return Err("portable model-state family must not be empty".to_string());
    }
    if portable.token_len == 0 {
        return Err("portable model-state token length must be nonzero".to_string());
    }
    if require_continuation_logits && portable.continuation_logits.is_none() {
        return Err("portable baseline snapshot is missing continuation logits".to_string());
    }
    let mut names = HashSet::with_capacity(portable.tensors.len() + portable.paged_tensors.len());
    let mut restored_tensors = Vec::with_capacity(portable.tensors.len());
    for tensor in portable.tensors {
        let name = tensor.name.clone().ok_or_else(|| "portable model tensor is missing its name".to_string())?;
        if name.is_empty() || !names.insert(name.clone()) { return Err(format!("duplicate or empty portable model tensor name {name:?}")); }
        restored_tensors.push((name.clone(), array_from_portable(tensor, Some(&name))?));
    }
    let mut restored_pages = Vec::with_capacity(portable.paged_tensors.len());
    for tensor in portable.paged_tensors {
        if tensor.name.is_empty() || !names.insert(tensor.name.clone()) { return Err("duplicate portable paged tensor name".into()); }
        let pages = tensor.pages.into_iter().map(|page| {
            SnapshotPage::from_portable(page.token_start, page.token_end, page.shape, page.dtype, &page.bytes)
        }).collect::<Result<Vec<_>, _>>()?;
        restored_pages.push((tensor.name, tensor.token_axis, pages));
    }
    let continuation_logits = portable.continuation_logits.map(|array| array_from_portable(array, None)).transpose()?;
    let mut snapshot = ModelStateSnapshot::new(portable.family, portable.token_len);
    for (name, array) in restored_tensors { snapshot.push_tensor(name, &array); }
    for (name, axis, pages) in restored_pages { snapshot.push_paged_pages(name, axis, pages)?; }
    if let Some(logits) = continuation_logits { snapshot.set_continuation_logits(&logits); }
    Ok(snapshot)
}

impl PromptSnapshot {
    pub fn nbytes(&self) -> usize {
        match self {
            Self::Baseline(snapshot) => snapshot.nbytes(),
            Self::Mtp(snapshot) => snapshot.nbytes(),
        }
    }

    pub fn to_portable(&self) -> Result<PortablePromptSnapshot, String> {
        Ok(match self {
            Self::Baseline(snapshot) => {
                PortablePromptSnapshot::Baseline(model_to_portable(snapshot))
            }
            Self::Mtp(snapshot) => snapshot.to_portable(),
        })
    }

    pub fn from_portable(portable: PortablePromptSnapshot) -> Result<Self, String> {
        match portable {
            PortablePromptSnapshot::Baseline(snapshot) => {
                model_from_portable(snapshot, true).map(Self::Baseline)
            }
            PortablePromptSnapshot::Mtp {
                target,
                draft_keys,
                draft_values,
                draft_offset,
                last_hidden,
                continuation_logits,
            } => MtpPromptSnapshot::from_portable_parts(
                model_from_portable(target, false)?,
                draft_keys,
                draft_values,
                draft_offset,
                last_hidden,
                continuation_logits,
            )
            .map(Self::Mtp),
        }
    }
}

pub(crate) fn portable_model_state(snapshot: &ModelStateSnapshot) -> PortableModelState {
    model_to_portable(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline(dtype: i32) -> PromptSnapshot {
        let source = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let array = if dtype == mlxcel_core::dtype::FLOAT32 {
            source
        } else {
            mlxcel_core::astype(&source, dtype)
        };
        let mut snapshot = ModelStateSnapshot::new("portable-test", 4);
        snapshot.push_tensor("state", &array);
        snapshot.set_continuation_logits(&array);
        PromptSnapshot::Baseline(snapshot)
    }

    #[test]
    fn baseline_arrays_round_trip_exactly_for_supported_storage_types() {
        for dtype in [
            mlxcel_core::dtype::FLOAT16,
            mlxcel_core::dtype::FLOAT32,
            mlxcel_core::dtype::UINT8,
        ] {
            let snapshot = baseline(dtype);
            let expected = snapshot.to_portable().expect("encode baseline");
            let restored = PromptSnapshot::from_portable(expected.clone())
                .expect("restore baseline")
                .to_portable()
                .expect("re-encode baseline");
            assert_eq!(restored, expected);
        }
    }

    #[test]
    fn portable_arrays_reject_corrupt_lengths_shapes_dtypes_and_names() {
        let PortablePromptSnapshot::Baseline(mut portable) = baseline(mlxcel_core::dtype::FLOAT32)
            .to_portable()
            .expect("encode baseline")
        else {
            unreachable!();
        };
        portable.tensors[0].bytes.pop();
        assert!(PromptSnapshot::from_portable(PortablePromptSnapshot::Baseline(portable)).is_err());

        let PortablePromptSnapshot::Baseline(mut portable) = baseline(mlxcel_core::dtype::FLOAT32)
            .to_portable()
            .expect("encode baseline")
        else {
            unreachable!();
        };
        portable.tensors[0].shape.clear();
        assert!(PromptSnapshot::from_portable(PortablePromptSnapshot::Baseline(portable)).is_err());

        let PortablePromptSnapshot::Baseline(mut portable) = baseline(mlxcel_core::dtype::FLOAT32)
            .to_portable()
            .expect("encode baseline")
        else {
            unreachable!();
        };
        portable.tensors[0].dtype = i32::MAX;
        assert!(PromptSnapshot::from_portable(PortablePromptSnapshot::Baseline(portable)).is_err());

        let PortablePromptSnapshot::Baseline(mut portable) = baseline(mlxcel_core::dtype::FLOAT32)
            .to_portable()
            .expect("encode baseline")
        else {
            unreachable!();
        };
        portable.tensors.push(portable.tensors[0].clone());
        assert!(PromptSnapshot::from_portable(PortablePromptSnapshot::Baseline(portable)).is_err());
    }
}
