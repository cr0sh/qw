use crate::{EntryKey, SnapshotRoute};
use qw_runtime::{
    PortableArray, PortableModelState, PortablePage, PortablePagedTensor, PortablePromptSnapshot,
    PromptSnapshot,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
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
    pub message_id: String,
    pub created_unix_seconds: u64,
    pub prompt_token_count: usize,
    pub request_fingerprint: String,
    pub generated_token_ids: Vec<i32>,
    pub raw_text: String,
    pub emitted_reasoning_text: String,
    pub emitted_content_text: String,
    pub original_max_tokens: usize,
    /// An unused RNG branch reserved when this response was interrupted.
    pub continuation_seed: u64,
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
    pub draft_family: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub draft_offset: Option<i32>,
    #[cfg(feature = "dflash2")]
    #[serde(deserialize_with = "required_option")]
    pub hidden_offset: Option<usize>,
    pub arrays: Vec<ArrayDescriptor>,
    pub paged_tensors: Vec<PagedTensorDescriptor>,
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
    pub blob_sha256: String,
    pub byte_len: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagedTensorDescriptor {
    pub role: ArrayRole,
    pub name: String,
    pub token_axis: usize,
    pub token_len: usize,
    pub pages: Vec<PageDescriptor>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageDescriptor {
    pub token_start: usize,
    pub token_end: usize,
    pub shape: Vec<i32>,
    pub dtype: i32,
    pub blob_sha256: String,
    pub byte_len: u64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ArrayRole {
    ModelTensor,
    ModelContinuation,
    DraftKeys,
    DraftValues,
    LastHidden,
    MtpContinuation,
    TargetTensor,
    TargetContinuation,
    DraftTensor,
    DraftContinuation,
    #[cfg(feature = "dflash2")]
    DflashHidden,
    #[cfg(feature = "dflash2")]
    DflashContinuation,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentBlob {
    pub sha256: String,
    pub bytes: Arc<[u8]>,
}
#[derive(Debug, Clone)]
pub struct EncodedEntry {
    pub key: EntryKey,
    pub manifest: Vec<u8>,
    pub blobs: Vec<ContentBlob>,
}
pub struct DecodedEntry {
    pub manifest: Manifest,
    pub snapshot: PromptSnapshot,
}
pub struct PortableDecodedEntry {
    pub manifest: Manifest,
    pub snapshot: PortablePromptSnapshot,
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn digest(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}
pub fn entry_key(ns: &str, r: SnapshotRoute, t: &[i32]) -> EntryKey {
    let mut h = Sha256::new();
    h.update(ns.as_bytes());
    h.update([r as u8]);
    for x in t {
        h.update(x.to_le_bytes())
    }
    EntryKey(format!("{ns}/{}", hex(&h.finalize())))
}
pub fn namespace_hash(p: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for x in p {
        h.update((x.len() as u64).to_le_bytes());
        h.update(x)
    }
    hex(&h.finalize())
}
pub fn encode_portable(
    ns: &str,
    r: SnapshotRoute,
    t: &[i32],
    p: PortablePromptSnapshot,
    ret: RetentionMetadata,
    exp: u64,
    res: Option<ResponseResumeMetadata>,
    reserve_manifest: impl FnOnce(u64) -> Result<(), String>,
) -> Result<EncodedEntry, String> {
    let len = match &p {
        PortablePromptSnapshot::Baseline(m) => m.token_len,
        PortablePromptSnapshot::Mtp { target, .. } => target.token_len,
        #[cfg(feature = "dflash2")]
        PortablePromptSnapshot::Dflash2 { target, .. } => target.token_len,
    };
    let route_matches = match (&p, r) {
        (PortablePromptSnapshot::Baseline(_), SnapshotRoute::Baseline)
        | (PortablePromptSnapshot::Mtp { .. }, SnapshotRoute::Mtp) => true,
        #[cfg(feature = "dflash2")]
        (PortablePromptSnapshot::Dflash2 { .. }, SnapshotRoute::Dflash2) => true,
        _ => false,
    };
    if t.is_empty() || len != t.len() || !route_matches {
        return Err("portable snapshot route and token length must match the cache entry".into());
    }
    validate_resume_metadata(t, res.as_ref())?;
    #[cfg(feature = "dflash2")]
    let hidden_offset = match &p {
        PortablePromptSnapshot::Dflash2 { hidden_offset, .. } => Some(*hidden_offset),
        _ => None,
    };
    let (f, df, off, dense, paged) = flatten(p)?;
    let mut blobs = Vec::new();
    let mut seen = HashSet::new();
    enum BlobBytes {
        Dense(Vec<u8>),
        Shared(Arc<[u8]>),
    }
    impl BlobBytes {
        fn as_slice(&self) -> &[u8] {
            match self {
                Self::Dense(bytes) => bytes,
                Self::Shared(bytes) => bytes,
            }
        }

        fn into_arc(self) -> Arc<[u8]> {
            match self {
                Self::Dense(bytes) => Arc::from(bytes),
                Self::Shared(bytes) => bytes,
            }
        }
    }
    let mut add = |b: BlobBytes| {
        let s = digest(b.as_slice());
        if seen.insert(s.clone()) {
            blobs.push(ContentBlob {
                sha256: s.clone(),
                bytes: b.into_arc(),
            })
        }
        s
    };
    let arrays = dense
        .into_iter()
        .map(|(role, a)| {
            let n = a.bytes.len() as u64;
            ArrayDescriptor {
                role,
                name: a.name,
                shape: a.shape,
                dtype: a.dtype,
                blob_sha256: add(BlobBytes::Dense(a.bytes)),
                byte_len: n,
            }
        })
        .collect();
    let paged_tensors = paged
        .into_iter()
        .map(|(role, x)| PagedTensorDescriptor {
            role,
            name: x.name,
            token_axis: x.token_axis,
            token_len: x.token_len,
            pages: x
                .pages
                .into_iter()
                .map(|p| {
                    let n = p.bytes.len() as u64;
                    PageDescriptor {
                        token_start: p.token_start,
                        token_end: p.token_end,
                        shape: p.shape,
                        dtype: p.dtype,
                        blob_sha256: add(BlobBytes::Shared(p.bytes)),
                        byte_len: n,
                    }
                })
                .collect(),
        })
        .collect();
    let total = blobs.iter().map(|b| b.bytes.len() as u64).sum();
    let m = Manifest {
        namespace: ns.into(),
        route: r,
        token_ids: t.to_vec(),
        token_len: len,
        family: f,
        draft_family: df,
        draft_offset: off,
        #[cfg(feature = "dflash2")]
        hidden_offset,
        arrays,
        paged_tensors,
        retention: ret,
        expires_at_unix_ms: exp,
        response_resume: res,
        blob_sha256: blobs.iter().map(|b| b.sha256.clone()).collect(),
        total_bytes: total,
    };
    validate_manifest(ns, &m)?;
    // Count without allocating the serialized buffer, then reserve its exact
    // bytes before allocation. Both passes stay on the publication worker.
    struct ByteCount(u64);
    impl std::io::Write for ByteCount {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| std::io::Error::other("manifest byte count overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = ByteCount(0);
    serde_json::to_writer(&mut count, &m).map_err(|e| e.to_string())?;
    reserve_manifest(count.0)?;
    let size = usize::try_from(count.0).map_err(|e| e.to_string())?;
    let mut manifest = Vec::with_capacity(size);
    serde_json::to_writer(&mut manifest, &m).map_err(|e| e.to_string())?;
    Ok(EncodedEntry {
        key: entry_key(ns, r, t),
        manifest,
        blobs,
    })
}
pub fn decode(ns: &str, b: &[u8], blobs: Vec<ContentBlob>) -> Result<DecodedEntry, String> {
    let decoded = decode_portable(ns, b, blobs)?;
    restore(decoded)
}

pub fn restore(decoded: PortableDecodedEntry) -> Result<DecodedEntry, String> {
    let s = PromptSnapshot::from_portable(decoded.snapshot)?;
    if s.token_len() != decoded.manifest.token_len || !decoded.manifest.route.matches(&s) {
        return Err("restored snapshot does not match manifest".into());
    }
    Ok(DecodedEntry {
        manifest: decoded.manifest,
        snapshot: s,
    })
}

pub fn decode_portable(
    ns: &str,
    b: &[u8],
    blobs: Vec<ContentBlob>,
) -> Result<PortableDecodedEntry, String> {
    let m: Manifest = serde_json::from_slice(b).map_err(|e| e.to_string())?;
    validate_manifest(ns, &m)?;
    let map: HashMap<_, _> = blobs.into_iter().map(|x| (x.sha256, x.bytes)).collect();
    if map.len() != m.blob_sha256.len() {
        return Err("cache blob set does not match manifest".into());
    }
    for d in &m.blob_sha256 {
        let b = map.get(d).ok_or("cache blob missing")?;
        if digest(b) != *d {
            return Err("cache blob digest mismatch".into());
        }
    }
    let get = |d: &str, n: u64| {
        map.get(d)
            .filter(|b| b.len() as u64 == n)
            .map(|b| b.to_vec())
            .ok_or_else(|| "cache descriptor blob is missing or wrong length".to_string())
    };
    let dense = m
        .arrays
        .iter()
        .map(|x| {
            Ok((
                x.role,
                PortableArray {
                    name: x.name.clone(),
                    shape: x.shape.clone(),
                    dtype: x.dtype,
                    bytes: get(&x.blob_sha256, x.byte_len)?,
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let paged = m
        .paged_tensors
        .iter()
        .map(|x| {
            Ok((
                x.role,
                PortablePagedTensor {
                    name: x.name.clone(),
                    token_axis: x.token_axis,
                    token_len: x.token_len,
                    pages: x
                        .pages
                        .iter()
                        .map(|p| {
                            Ok(PortablePage {
                                token_start: p.token_start,
                                token_end: p.token_end,
                                shape: p.shape.clone(),
                                dtype: p.dtype,
                                bytes: map
                                    .get(&p.blob_sha256)
                                    .filter(|b| b.len() as u64 == p.byte_len)
                                    .cloned()
                                    .ok_or("cache descriptor blob is missing or wrong length")?,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let p = inflate(&m, dense, paged)?;
    Ok(PortableDecodedEntry {
        manifest: m,
        snapshot: p,
    })
}
pub fn parse_manifest(ns: &str, b: &[u8]) -> Result<Manifest, String> {
    let m: Manifest = serde_json::from_slice(b).map_err(|e| e.to_string())?;
    validate_manifest(ns, &m)?;
    Ok(m)
}
fn validate_manifest(ns: &str, m: &Manifest) -> Result<(), String> {
    if m.namespace != ns || m.token_ids.is_empty() || m.token_ids.len() != m.token_len {
        return Err("cache manifest namespace or token length is invalid".into());
    }
    validate_resume_metadata(&m.token_ids, m.response_resume.as_ref())?;
    let family_metadata_valid = match (&m.route, &m.draft_offset, &m.draft_family) {
        (SnapshotRoute::Baseline, None, None) => true,
        (SnapshotRoute::Mtp, Some(_), Some(family)) => !family.is_empty(),
        #[cfg(feature = "dflash2")]
        (SnapshotRoute::Dflash2, None, None) => true,
        _ => false,
    };
    if m.family.is_empty()
        || !family_metadata_valid
        || (m.arrays.is_empty() && m.paged_tensors.is_empty())
    {
        return Err("cache manifest is missing model state".into());
    }
    #[cfg(feature = "dflash2")]
    if m.route == SnapshotRoute::Dflash2 {
        validate_dflash_manifest(m)?;
    } else if m.hidden_offset.is_some()
        || m.arrays.iter().any(|a| {
            matches!(
                a.role,
                ArrayRole::DflashHidden | ArrayRole::DflashContinuation
            )
        })
        || m.paged_tensors.iter().any(|a| {
            matches!(
                a.role,
                ArrayRole::DflashHidden | ArrayRole::DflashContinuation
            )
        })
    {
        return Err("non-DFlash2 manifest contains DFlash2 state".into());
    }
    let mut total = 0u64;
    let mut sizes = HashMap::new();
    for a in &m.arrays {
        if a.shape.is_empty()
            || a.shape.iter().any(|x| *x <= 0)
            || a.byte_len == 0
            || a.blob_sha256.len() != 64
        {
            return Err("cache array descriptor is invalid".into());
        }
        if let Some(old) = sizes.insert(a.blob_sha256.clone(), a.byte_len) {
            if old != a.byte_len {
                return Err("cache blob descriptor sizes disagree".into());
            }
        }
    }
    for t in &m.paged_tensors {
        let mut end = 0;
        for p in &t.pages {
            if p.token_start != end
                || p.token_end <= p.token_start
                || p.token_end > t.token_len
                || p.byte_len == 0
                || p.blob_sha256.len() != 64
            {
                return Err("cache page descriptor is invalid".into());
            }
            end = p.token_end;
            if let Some(old) = sizes.insert(p.blob_sha256.clone(), p.byte_len) {
                if old != p.byte_len {
                    return Err("cache blob descriptor sizes disagree".into());
                }
            }
        }
        if t.name.is_empty() || t.token_len == 0 || end != t.token_len {
            return Err("cache pages do not cover token length".into());
        }
    }
    for n in sizes.values() {
        total = total
            .checked_add(*n)
            .ok_or("cache payload byte count overflow")?;
    }
    if total != m.total_bytes {
        return Err("cache payload byte count does not match manifest".into());
    }
    Ok(())
}

#[cfg(feature = "dflash2")]
fn validate_dflash_manifest(m: &Manifest) -> Result<(), String> {
    use mlxcel_core::dtype::{BFLOAT16, FLOAT16, FLOAT32};

    let offset = m
        .hidden_offset
        .ok_or("DFlash2 manifest is missing hidden offset")?;
    let mut names = HashSet::new();
    let mut singleton_roles = HashSet::new();
    for array in &m.arrays {
        validate_dflash_array(&array.shape, array.dtype, array.byte_len)?;
        match array.role {
            ArrayRole::TargetTensor => {
                let name = array
                    .name
                    .as_deref()
                    .ok_or("DFlash2 target tensor is unnamed")?;
                if name.is_empty() || !names.insert(name) {
                    return Err("DFlash2 target tensor names must be unique and nonempty".into());
                }
            }
            ArrayRole::TargetContinuation
            | ArrayRole::DflashHidden
            | ArrayRole::DflashContinuation => {
                if array.name.is_some() || !singleton_roles.insert(array.role) {
                    return Err("DFlash2 auxiliary tensor is named or duplicated".into());
                }
                if array.shape.len() != 3
                    || array.shape[0] != 1
                    || !matches!(array.dtype, FLOAT16 | FLOAT32 | BFLOAT16)
                {
                    return Err(
                        "DFlash2 auxiliary tensor must be floating-point [1, rows, width]".into(),
                    );
                }
                if array.role == ArrayRole::DflashHidden {
                    if offset.checked_add(array.shape[1] as usize) != Some(m.token_len) {
                        return Err(
                            "DFlash2 hidden context does not match the target boundary".into()
                        );
                    }
                } else if array.shape[1] != 1 {
                    return Err("DFlash2 continuation logits must have one row".into());
                }
            }
            _ => return Err("DFlash2 manifest contains a tensor from another route".into()),
        }
    }
    if !singleton_roles.contains(&ArrayRole::DflashHidden)
        || !singleton_roles.contains(&ArrayRole::DflashContinuation)
    {
        return Err("DFlash2 manifest is missing hidden context or continuation logits".into());
    }
    for tensor in &m.paged_tensors {
        if tensor.role != ArrayRole::TargetTensor
            || tensor.name.is_empty()
            || !names.insert(tensor.name.as_str())
            || tensor.token_len != m.token_len
        {
            return Err("DFlash2 target page role, name, or boundary is invalid".into());
        }
        let first = tensor
            .pages
            .first()
            .ok_or("DFlash2 target pages are empty")?;
        for page in &tensor.pages {
            validate_dflash_array(&page.shape, page.dtype, page.byte_len)?;
            if page.shape.len() != first.shape.len()
                || page.dtype != first.dtype
                || page.shape.get(tensor.token_axis).map(|&n| n as usize)
                    != page.token_end.checked_sub(page.token_start)
                || page
                    .shape
                    .iter()
                    .zip(&first.shape)
                    .enumerate()
                    .any(|(axis, (a, b))| axis != tensor.token_axis && a != b)
            {
                return Err("DFlash2 target page layout is inconsistent".into());
            }
        }
    }
    if names.is_empty() {
        return Err("DFlash2 manifest is missing target model state".into());
    }
    let referenced = m
        .arrays
        .iter()
        .map(|a| a.blob_sha256.as_str())
        .chain(
            m.paged_tensors
                .iter()
                .flat_map(|t| t.pages.iter().map(|p| p.blob_sha256.as_str())),
        )
        .collect::<HashSet<_>>();
    if m.blob_sha256.len() != referenced.len()
        || m.blob_sha256
            .iter()
            .any(|digest| !referenced.contains(digest.as_str()))
        || m.blob_sha256.iter().collect::<HashSet<_>>().len() != referenced.len()
    {
        return Err("DFlash2 blob set does not match tensor descriptors".into());
    }
    Ok(())
}

#[cfg(feature = "dflash2")]
fn validate_dflash_array(shape: &[i32], dtype: i32, byte_len: u64) -> Result<(), String> {
    use mlxcel_core::dtype::*;

    if shape.is_empty()
        || shape.iter().any(|&n| n <= 0)
        || !matches!(
            dtype,
            UINT8 | INT8 | UINT32 | UINT64 | INT32 | INT64 | FLOAT16 | FLOAT32 | BFLOAT16
        )
    {
        return Err("DFlash2 tensor shape or dtype is invalid".into());
    }
    let bytes = shape
        .iter()
        .try_fold(size_bytes(dtype).unwrap() as u64, |bytes, &n| {
            bytes.checked_mul(n as u64)
        })
        .ok_or("DFlash2 tensor byte length overflow")?;
    if bytes != byte_len {
        return Err("DFlash2 tensor byte length does not match shape and dtype".into());
    }
    Ok(())
}

pub(crate) fn validate_resume_metadata(
    t: &[i32],
    m: Option<&ResponseResumeMetadata>,
) -> Result<(), String> {
    let Some(m) = m else { return Ok(()) };
    if m.response_id.is_empty()
        || m.message_id.is_empty()
        || m.request_fingerprint.is_empty()
        || m.prompt_token_count == 0
        || m.prompt_token_count > t.len()
        || m.generated_token_ids.is_empty()
        || m.generated_token_ids.len() >= m.original_max_tokens
    {
        return Err("cache response resume metadata is invalid".into());
    }
    let x = &t[m.prompt_token_count..];
    if x.len() > m.generated_token_ids.len() || x != &m.generated_token_ids[..x.len()] {
        return Err("cache response resume tokens do not match the snapshot".into());
    }
    Ok(())
}
fn flatten(
    p: PortablePromptSnapshot,
) -> Result<
    (
        String,
        Option<String>,
        Option<i32>,
        Vec<(ArrayRole, PortableArray)>,
        Vec<(ArrayRole, PortablePagedTensor)>,
    ),
    String,
> {
    fn m(
        mut x: PortableModelState,
        r: ArrayRole,
        cr: ArrayRole,
    ) -> (
        String,
        Vec<(ArrayRole, PortableArray)>,
        Vec<(ArrayRole, PortablePagedTensor)>,
    ) {
        let f = x.family;
        let mut d = x.tensors.into_iter().map(|a| (r, a)).collect::<Vec<_>>();
        if let Some(a) = x.continuation_logits.take() {
            d.push((cr, a));
        }
        (f, d, x.paged_tensors.into_iter().map(|a| (r, a)).collect())
    }
    match p {
        PortablePromptSnapshot::Baseline(x) => {
            let (f, d, p) = m(x, ArrayRole::ModelTensor, ArrayRole::ModelContinuation);
            Ok((f, None, None, d, p))
        }
        PortablePromptSnapshot::Mtp {
            target,
            draft,
            draft_offset,
            last_hidden,
            continuation_logits,
        } => {
            let (f, mut d, mut p) = m(
                target,
                ArrayRole::TargetTensor,
                ArrayRole::TargetContinuation,
            );
            let (df, mut dd, mut pp) =
                m(draft, ArrayRole::DraftTensor, ArrayRole::DraftContinuation);
            d.append(&mut dd);
            p.append(&mut pp);
            d.push((ArrayRole::LastHidden, last_hidden));
            d.push((ArrayRole::MtpContinuation, continuation_logits));
            Ok((f, Some(df), Some(draft_offset), d, p))
        }
        #[cfg(feature = "dflash2")]
        PortablePromptSnapshot::Dflash2 {
            target,
            hidden_concat,
            continuation_logits,
            ..
        } => {
            let (f, mut d, p) = m(
                target,
                ArrayRole::TargetTensor,
                ArrayRole::TargetContinuation,
            );
            d.push((ArrayRole::DflashHidden, hidden_concat));
            d.push((ArrayRole::DflashContinuation, continuation_logits));
            Ok((f, None, None, d, p))
        }
    }
}
fn inflate(
    m: &Manifest,
    d: Vec<(ArrayRole, PortableArray)>,
    p: Vec<(ArrayRole, PortablePagedTensor)>,
) -> Result<PortablePromptSnapshot, String> {
    let mut t = PortableModelState {
        family: m.family.clone(),
        token_len: m.token_len,
        tensors: Vec::new(),
        paged_tensors: Vec::new(),
        continuation_logits: None,
    };
    let mut dr = PortableModelState {
        family: String::new(),
        token_len: 0,
        tensors: Vec::new(),
        paged_tensors: Vec::new(),
        continuation_logits: None,
    };
    let (mut h, mut c) = (None, None);
    for (r, a) in d {
        match r {
            ArrayRole::ModelTensor | ArrayRole::TargetTensor => t.tensors.push(a),
            ArrayRole::ModelContinuation | ArrayRole::TargetContinuation => {
                t.continuation_logits = Some(a)
            }
            ArrayRole::DraftTensor => dr.tensors.push(a),
            ArrayRole::LastHidden => h = Some(a),
            ArrayRole::MtpContinuation => c = Some(a),
            #[cfg(feature = "dflash2")]
            ArrayRole::DflashHidden => h = Some(a),
            #[cfg(feature = "dflash2")]
            ArrayRole::DflashContinuation => c = Some(a),
            _ => {}
        }
    }
    for (r, a) in p {
        match r {
            ArrayRole::ModelTensor | ArrayRole::TargetTensor => t.paged_tensors.push(a),
            ArrayRole::DraftTensor => dr.paged_tensors.push(a),
            _ => {}
        }
    }
    #[cfg(feature = "dflash2")]
    if m.route == SnapshotRoute::Dflash2 {
        return Ok(PortablePromptSnapshot::Dflash2 {
            target: t,
            hidden_concat: h.ok_or("DFlash2 manifest is missing hidden context")?,
            hidden_offset: m
                .hidden_offset
                .ok_or("DFlash2 manifest is missing hidden offset")?,
            continuation_logits: c.ok_or("DFlash2 manifest is missing continuation logits")?,
        });
    }
    if m.route == SnapshotRoute::Baseline {
        Ok(PortablePromptSnapshot::Baseline(t))
    } else {
        let draft_offset = m
            .draft_offset
            .ok_or("MTP manifest is missing draft offset")?;
        dr.family = m
            .draft_family
            .clone()
            .ok_or("MTP manifest is missing draft family")?;
        dr.token_len =
            usize::try_from(draft_offset).map_err(|_| "MTP manifest draft offset is invalid")?;
        Ok(PortablePromptSnapshot::Mtp {
            target: t,
            draft: dr,
            draft_offset,
            last_hidden: h.ok_or("MTP manifest is missing last_hidden")?,
            continuation_logits: c.ok_or("MTP manifest is missing continuation logits")?,
        })
    }
}
fn required_option<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(d)
}
