use crate::{EntryKey, codec::ContentBlob};
use std::{
    fs::{self, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
#[derive(Debug, Clone)]
pub struct ScannedEntry {
    pub key: EntryKey,
    pub manifest: Vec<u8>,
}
#[derive(Debug)]
pub struct StoredEntry {
    pub key: EntryKey,
    pub manifest: Vec<u8>,
    pub blobs: Vec<ContentBlob>,
}
pub trait PersistentSnapshotStore: Send {
    fn scan(&mut self, namespace: &str, now: u64) -> Result<Vec<ScannedEntry>, String>;
    fn load(&mut self, key: &EntryKey) -> Result<Option<StoredEntry>, String>;
    /// Advisory reads must enforce the byte limit before allocating payload buffers.
    /// Stores without a bounded reader opt out rather than allocating speculatively.
    fn load_bounded(&mut self, _key: &EntryKey, _limit: u64) -> Result<Option<StoredEntry>, String> {
        Ok(None)
    }
    fn put(&mut self, e: StoredEntry, expires: u64) -> Result<(), String>;
    fn refresh(&mut self, key: &EntryKey, expires: u64) -> Result<(), String>;
    fn remove(&mut self, key: &EntryKey) -> Result<(), String>;
}
pub struct FilesystemSnapshotStore {
    root: PathBuf,
    #[cfg(all(test, target_vendor = "apple"))]
    pub(super) before_blob_barrier: Option<Box<dyn FnOnce() -> Result<(), String> + Send>>,
}
impl FilesystemSnapshotStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, String> {
        let r = root.into();
        fs::create_dir_all(r.join("entries")).map_err(|e| e.to_string())?;
        fs::create_dir_all(r.join("blobs")).map_err(|e| e.to_string())?;
        Ok(Self {
            root: r,
            #[cfg(all(test, target_vendor = "apple"))]
            before_blob_barrier: None,
        })
    }
    fn path(&self, k: &EntryKey) -> Result<PathBuf, String> {
        let mut p = k.0.split('/');
        let n = p.next().ok_or("invalid cache key")?;
        let d = p.next().ok_or("invalid cache key")?;
        if p.next().is_some() || !safe(n) || !safe(d) {
            return Err("cache key contains invalid path components".into());
        }
        Ok(self.root.join("entries").join(n).join(format!("{d}.json")))
    }
    fn blob(&self, d: &str) -> PathBuf {
        self.root.join("blobs").join(d)
    }
    fn cleanup_orphans(&self) -> Result<(), String> {
        let mut refs = std::collections::HashSet::new();
        if let Ok(names) = fs::read_dir(self.root.join("entries")) {
            for ns in names.flatten() {
                if let Ok(entries) = fs::read_dir(ns.path()) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
                            continue;
                        };
                        if name.starts_with(".tmp-") || name.contains(".json.tmp-") {
                            let _ = fs::remove_file(path);
                            continue;
                        }
                        if let Ok(bytes) = fs::read(&path) {
                            if let Ok(m) = serde_json::from_slice::<crate::Manifest>(&bytes) {
                                refs.extend(m.blob_sha256);
                            }
                        }
                    }
                }
            }
        }
        if let Ok(blobs) = fs::read_dir(self.root.join("blobs")) {
            for b in blobs.flatten() {
                let path = b.path();
                let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
                    continue;
                };
                if name.starts_with(".tmp-") || name.contains(".tmp-") {
                    let _ = fs::remove_file(path);
                } else if !refs.contains(name) {
                    let _ = fs::remove_file(path);
                }
            }
        }
        Ok(())
    }
}
impl PersistentSnapshotStore for FilesystemSnapshotStore {
    fn scan(&mut self, ns: &str, _: u64) -> Result<Vec<ScannedEntry>, String> {
        if !safe(ns) {
            return Err("cache namespace is not path-safe".into());
        }
        let mut out = Vec::new();
        let dir = self.root.join("entries").join(ns);
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|x| x.to_str()).map(str::to_owned)
                else {
                    continue;
                };
                if let Ok(manifest) = fs::read(path) {
                    out.push(ScannedEntry {
                        key: EntryKey(format!("{ns}/{stem}")),
                        manifest,
                    });
                }
            }
        }
        self.cleanup_orphans()?;
        Ok(out)
    }
    fn load(&mut self, k: &EntryKey) -> Result<Option<StoredEntry>, String> {
        let p = self.path(k)?;
        let Ok(manifest) = fs::read(&p) else {
            return Ok(None);
        };
        let m: crate::Manifest = serde_json::from_slice(&manifest).map_err(|e| e.to_string())?;
        let mut blobs = Vec::new();
        for d in m.blob_sha256 {
            let b = fs::read(self.blob(&d))
                .map_err(|e| format!("failed to read cache blob {d}: {e}"))?;
            blobs.push(ContentBlob {
                sha256: d,
                bytes: b.into(),
            });
        }
        Ok(Some(StoredEntry {
            key: k.clone(),
            manifest,
            blobs,
        }))
    }
    fn load_bounded(&mut self, key: &EntryKey, limit: u64) -> Result<Option<StoredEntry>, String> {
        let path = self.path(key)?;
        if !path.exists() {
            return Ok(None);
        }
        let manifest = read_bounded(&path, limit.min(1024 * 1024))?;
        let ns = key.0.split('/').next().ok_or("invalid entry key")?;
        let parsed = crate::codec::parse_manifest(ns, &manifest)?;
        let mut remaining = limit.saturating_sub(manifest.len() as u64);
        let mut blobs = Vec::new();
        for digest in parsed.blob_sha256 {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid blob digest".into());
            }
            let bytes = read_bounded(&self.blob(&digest), remaining)?;
            remaining = remaining.checked_sub(bytes.len() as u64).ok_or("prefetch byte limit")?;
            blobs.push(ContentBlob { sha256: digest, bytes: bytes.into() });
        }
        Ok(Some(StoredEntry { key: key.clone(), manifest, blobs }))
    }
    fn put(&mut self, e: StoredEntry, _: u64) -> Result<(), String> {
        let p = self.path(&e.key)?;
        fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
        #[cfg(target_vendor = "apple")]
        let mut blob_barrier = None;
        for b in &e.blobs {
            if b.sha256.len() != 64 || !b.sha256.bytes().all(|x| x.is_ascii_hexdigit()) {
                return Err("invalid blob digest".into());
            }
            let q = self.blob(&b.sha256);
            if !q.exists() {
                #[cfg(target_vendor = "apple")]
                {
                    let blob = write_host_synced_blob(&q, &b.bytes)?;
                    if blob_barrier.is_none() {
                        blob_barrier = Some(blob);
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                write_synced(&q, &b.bytes)?;
            }
            #[cfg(target_vendor = "apple")]
            if blob_barrier.is_none() {
                // Existing blobs can be orphans from a failed earlier Put.
                blob_barrier = Some(std::fs::File::open(&q).map_err(|e| e.to_string())?);
            }
        }
        #[cfg(target_vendor = "apple")]
        if let Some(blob) = blob_barrier {
            #[cfg(test)]
            if let Some(before_barrier) = self.before_blob_barrier.take() {
                before_barrier()?;
            }
            // Flush drive buffers after all blob fsyncs and before manifest publication.
            blob.sync_all().map_err(|e| e.to_string())?;
        }
        let tmp = p.with_extension(format!(
            "tmp-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        write_synced(&tmp, &e.manifest)?;
        fs::rename(tmp, p).map_err(|e| e.to_string())?;
        Ok(())
    }
    fn refresh(&mut self, k: &EntryKey, expires: u64) -> Result<(), String> {
        let p = self.path(k)?;
        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(&p).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        v["expires_at_unix_ms"] = expires.into();
        let tmp = p.with_extension("tmp");
        write_synced(&tmp, &serde_json::to_vec(&v).map_err(|e| e.to_string())?)?;
        fs::rename(tmp, p).map_err(|e| e.to_string())
    }
    fn remove(&mut self, k: &EntryKey) -> Result<(), String> {
        match fs::remove_file(self.path(k)?) {
            Ok(()) => self.cleanup_orphans(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    if size > limit {
        return Err("prefetch byte limit".into());
    }
    let size = usize::try_from(size).map_err(|e| e.to_string())?;
    let mut bytes = vec![0; size];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let mut extra = [0];
    if file.read(&mut extra).map_err(|e| e.to_string())? != 0 {
        return Err("cache file changed during bounded read".into());
    }
    Ok(bytes)
}
fn write_synced(p: &Path, b: &[u8]) -> Result<(), String> {
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(p)
        .map_err(|e| e.to_string())?;
    f.write_all(b).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}
#[cfg(target_vendor = "apple")]
fn write_host_synced_blob(p: &Path, b: &[u8]) -> Result<std::fs::File, String> {
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        fn fsync(fd: std::ffi::c_int) -> std::ffi::c_int;
    }

    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(p)
        .map_err(|e| e.to_string())?;
    f.write_all(b).map_err(|e| e.to_string())?;
    // Apple's sync_data also requests F_FULLFSYNC, so use ordinary fsync here.
    loop {
        // SAFETY: f owns an open descriptor and fsync does not retain it.
        if unsafe { fsync(f.as_raw_fd()) } == 0 {
            return Ok(f);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error.to_string());
        }
    }
}
fn safe(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
