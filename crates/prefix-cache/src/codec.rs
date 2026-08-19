use std::collections::HashSet;

use qw_runtime::{PortableArray, PortableModelState, PortablePromptSnapshot, PromptSnapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{EntryKey, SnapshotRoute};

pub const MAX_BLOB_BYTES: usize = 20 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetentionMetadata {
    pub observations: u64,
    pub reuse_count: u64,
    pub last_access_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResponseResumeMetadata {
    pub response_id: String,
    pub request_fingerprint: String,
    pub generated_token_ids: Vec<i32>,
    pub raw_text: String,
    pub original_max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub namespace: String,
    pub route: SnapshotRoute,
    pub token_ids: Vec<i32>,
    pub token_len: usize,
    pub family: String,
    #[serde(deserialize_with = "required_option")]
    pub draft_offset: Option<i32>,
    pub arrays: Vec<ArrayDescriptor>,
    pub retention: RetentionMetadata,
    pub expires_at_unix_ms: u64,
    #[serde(deserialize_with = "required_option")]
    pub response_resume: Option<ResponseResumeMetadata>,
    pub blob_sha256: Vec<String>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrayDescriptor {
    pub role: ArrayRole,
    #[serde(deserialize_with = "required_option")]
    pub name: Option<String>,
    pub shape: Vec<i32>,
    pub dtype: i32,
    pub offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArrayRole {
    ModelTensor,
    ModelContinuation,
    DraftKeys,
    DraftValues,
    LastHidden,
    MtpContinuation,
}

#[derive(Debug, Clone)]
pub struct EncodedEntry {
    pub key: EntryKey,
    pub manifest: Vec<u8>,
    pub blobs: Vec<Vec<u8>>,
}

pub struct DecodedEntry {
    pub manifest: Manifest,
    pub snapshot: PromptSnapshot,
}

pub fn entry_key(namespace: &str, route: SnapshotRoute, tokens: &[i32]) -> EntryKey {
    let mut hash = Sha256::new();
    hash.update(namespace.as_bytes());
    hash.update([route as u8]);
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    EntryKey(format!("{namespace}/{}", hex(&hash.finalize())))
}

pub fn namespace_hash(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hex(&hash.finalize())
}


pub fn encode_portable(
    namespace: &str,
    route: SnapshotRoute,
    token_ids: &[i32],
    portable: PortablePromptSnapshot,
    retention: RetentionMetadata,
    expires_at_unix_ms: u64,
    response_resume: Option<ResponseResumeMetadata>,
) -> Result<EncodedEntry, String> {
    let (portable_route, portable_token_len) = match &portable {
        PortablePromptSnapshot::Baseline(model) => (SnapshotRoute::Baseline, model.token_len),
        PortablePromptSnapshot::Mtp { target, .. } => (SnapshotRoute::Mtp, target.token_len),
    };
    if token_ids.is_empty() || portable_token_len != token_ids.len() || portable_route != route {
        return Err("portable snapshot route and token length must match the cache entry".to_string());
    }
    let (family, draft_offset, arrays) = flatten(portable)?;
    let mut descriptors = Vec::with_capacity(arrays.len());
    let mut payload = Vec::new();
    for (role, array) in arrays {
        let offset = payload.len() as u64;
        let byte_len = array.bytes.len() as u64;
        payload.extend_from_slice(&array.bytes);
        descriptors.push(ArrayDescriptor {
            role,
            name: array.name,
            shape: array.shape,
            dtype: array.dtype,
            offset,
            byte_len,
        });
    }
    let blobs = payload.chunks(MAX_BLOB_BYTES).map(<[u8]>::to_vec).collect::<Vec<_>>();
    let blob_sha256 = blobs.iter().map(|blob| hex(&Sha256::digest(blob))).collect();
    let total_bytes = payload.len() as u64;
    let manifest = Manifest {
        namespace: namespace.to_string(),
        route,
        token_ids: token_ids.to_vec(),
        token_len: token_ids.len(),
        family,
        draft_offset,
        arrays: descriptors,
        retention,
        expires_at_unix_ms,
        response_resume,
        blob_sha256,
        total_bytes,
    };
    let manifest = serde_json::to_vec(&manifest).map_err(|error| error.to_string())?;
    Ok(EncodedEntry {
        key: entry_key(namespace, route, token_ids),
        manifest,
        blobs,
    })
}

pub fn decode(expected_namespace: &str, manifest: &[u8], blobs: Vec<Vec<u8>>) -> Result<DecodedEntry, String> {
    let manifest: Manifest = serde_json::from_slice(manifest).map_err(|error| error.to_string())?;
    validate_manifest(expected_namespace, &manifest)?;
    if blobs.len() != manifest.blob_sha256.len() {
        return Err("cache blob count does not match manifest".to_string());
    }
    let mut payload = Vec::with_capacity(manifest.total_bytes as usize);
    for (index, (blob, expected)) in blobs.iter().zip(&manifest.blob_sha256).enumerate() {
        if blob.len() > MAX_BLOB_BYTES {
            return Err(format!("cache blob {index} exceeds the size limit"));
        }
        if hex(&Sha256::digest(blob)) != *expected {
            return Err(format!("cache blob {index} digest mismatch"));
        }
        payload.extend_from_slice(blob);
    }
    if payload.len() as u64 != manifest.total_bytes {
        return Err("cache payload byte count does not match manifest".to_string());
    }
    let arrays = manifest.arrays.iter().map(|descriptor| {
        let start = usize::try_from(descriptor.offset).map_err(|_| "array offset overflow")?;
        let len = usize::try_from(descriptor.byte_len).map_err(|_| "array length overflow")?;
        let end = start.checked_add(len).ok_or("array range overflow")?;
        let bytes = payload.get(start..end).ok_or("array range is outside payload")?.to_vec();
        Ok(PortableArray {
            name: descriptor.name.clone(),
            shape: descriptor.shape.clone(),
            dtype: descriptor.dtype,
            bytes,
        })
    }).collect::<Result<Vec<_>, &str>>().map_err(str::to_string)?;
    let portable = inflate(&manifest, arrays)?;
    let snapshot = PromptSnapshot::from_portable(portable)?;
    if snapshot.token_len() != manifest.token_len || !manifest.route.matches(&snapshot) {
        return Err("restored snapshot does not match manifest".to_string());
    }
    Ok(DecodedEntry { manifest, snapshot })
}

pub fn parse_manifest(expected_namespace: &str, bytes: &[u8]) -> Result<Manifest, String> {
    let manifest: Manifest = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    validate_manifest(expected_namespace, &manifest)?;
    Ok(manifest)
}

fn validate_manifest(expected_namespace: &str, manifest: &Manifest) -> Result<(), String> {
    if manifest.namespace != expected_namespace {
        return Err("cache namespace mismatch".to_string());
    }
    if manifest.token_ids.is_empty() || manifest.token_ids.len() != manifest.token_len {
        return Err("cache token length does not match token IDs".to_string());
    }
    if manifest.family.is_empty() || manifest.arrays.is_empty() {
        return Err("cache manifest is missing model state".to_string());
    }
    if manifest.blob_sha256.iter().any(|digest| digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit())) {
        return Err("cache manifest contains an invalid blob digest".to_string());
    }
    let mut end = 0u64;
    for descriptor in &manifest.arrays {
        if descriptor.shape.is_empty() || descriptor.shape.iter().any(|dimension| *dimension <= 0) {
            return Err("cache array has an invalid shape".to_string());
        }
        if descriptor.offset != end || descriptor.byte_len == 0 {
            return Err("cache array descriptors are not contiguous".to_string());
        }
        end = end.checked_add(descriptor.byte_len).ok_or("cache array range overflow")?;
    }
    if end != manifest.total_bytes {
        return Err("cache array bytes do not match total bytes".to_string());
    }
    Ok(())
}

fn flatten(portable: PortablePromptSnapshot) -> Result<(String, Option<i32>, Vec<(ArrayRole, PortableArray)>), String> {
    match portable {
        PortablePromptSnapshot::Baseline(model) => {
            let family = model.family.clone();
            let mut arrays = model.tensors.into_iter().map(|array| (ArrayRole::ModelTensor, array)).collect::<Vec<_>>();
            arrays.push((ArrayRole::ModelContinuation, model.continuation_logits.ok_or("baseline snapshot is missing continuation logits")?));
            Ok((family, None, arrays))
        }
        PortablePromptSnapshot::Mtp { target, draft_keys, draft_values, draft_offset, last_hidden, continuation_logits } => {
            if draft_keys.is_some() != draft_values.is_some() {
                return Err("MTP draft key/value pairing is invalid".to_string());
            }
            let family = target.family.clone();
            let mut arrays = target.tensors.into_iter().map(|array| (ArrayRole::ModelTensor, array)).collect::<Vec<_>>();
            if let Some(array) = target.continuation_logits { arrays.push((ArrayRole::ModelContinuation, array)); }
            if let Some(array) = draft_keys { arrays.push((ArrayRole::DraftKeys, array)); }
            if let Some(array) = draft_values { arrays.push((ArrayRole::DraftValues, array)); }
            arrays.push((ArrayRole::LastHidden, last_hidden));
            arrays.push((ArrayRole::MtpContinuation, continuation_logits));
            Ok((family, Some(draft_offset), arrays))
        }
    }
}

fn inflate(manifest: &Manifest, arrays: Vec<PortableArray>) -> Result<PortablePromptSnapshot, String> {
    let mut roles = manifest.arrays.iter().map(|descriptor| descriptor.role).zip(arrays);
    let mut tensors = Vec::new();
    let mut model_continuation = None;
    let mut draft_keys = None;
    let mut draft_values = None;
    let mut last_hidden = None;
    let mut mtp_continuation = None;
    for (role, array) in roles.by_ref() {
        match role {
            ArrayRole::ModelTensor => tensors.push(array),
            ArrayRole::ModelContinuation => {
                if model_continuation.replace(array).is_some() {
                    return Err("cache manifest contains duplicate array roles".to_string());
                }
            }
            ArrayRole::DraftKeys => {
                if draft_keys.replace(array).is_some() {
                    return Err("cache manifest contains duplicate array roles".to_string());
                }
            }
            ArrayRole::DraftValues => {
                if draft_values.replace(array).is_some() {
                    return Err("cache manifest contains duplicate array roles".to_string());
                }
            }
            ArrayRole::LastHidden => {
                if last_hidden.replace(array).is_some() {
                    return Err("cache manifest contains duplicate array roles".to_string());
                }
            }
            ArrayRole::MtpContinuation => {
                if mtp_continuation.replace(array).is_some() {
                    return Err("cache manifest contains duplicate array roles".to_string());
                }
            }
        }
    }
    let mut names = HashSet::new();
    if tensors.iter().any(|array| array.name.as_ref().is_none_or(|name| name.is_empty() || !names.insert(name.clone()))) {
        return Err("cache model tensor names must be present and unique".to_string());
    }
    let draft_offset = match manifest.route {
        SnapshotRoute::Baseline if manifest.draft_offset.is_none() => None,
        SnapshotRoute::Mtp => Some(
            manifest
                .draft_offset
                .ok_or("MTP manifest is missing draft offset")?,
        ),
        SnapshotRoute::Baseline => {
            return Err("baseline manifest contains an MTP draft offset".to_string());
        }
    };
    let family = manifest.family.clone();
    let target = PortableModelState { family, token_len: manifest.token_len, tensors, continuation_logits: model_continuation };
    match manifest.route {
        SnapshotRoute::Baseline => {
            if draft_keys.is_some() || draft_values.is_some() || last_hidden.is_some() || mtp_continuation.is_some() {
                return Err("baseline manifest contains MTP arrays".to_string());
            }
            Ok(PortablePromptSnapshot::Baseline(target))
        }
        SnapshotRoute::Mtp => {
            if draft_keys.is_some() != draft_values.is_some() {
                return Err("MTP draft key/value pairing is invalid".to_string());
            }
            Ok(PortablePromptSnapshot::Mtp {
                target,
                draft_keys,
                draft_values,
                draft_offset: draft_offset.expect("MTP route has parsed offset"),
                last_hidden: last_hidden.ok_or("MTP manifest is missing last_hidden")?,
                continuation_logits: mtp_continuation.ok_or("MTP manifest is missing continuation logits")?,
            })
        }
    }
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
