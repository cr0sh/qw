use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::EntryKey;

#[derive(Debug, Clone)]
pub struct ScannedEntry {
    pub key: EntryKey,
    pub manifest: Vec<u8>,
}

#[derive(Debug)]
pub struct StoredEntry {
    pub key: EntryKey,
    pub manifest: Vec<u8>,
    pub blobs: Vec<Vec<u8>>,
}

pub trait PersistentSnapshotStore: Send {
    fn scan(&mut self, namespace: &str, now_unix_ms: u64) -> Result<Vec<ScannedEntry>, String>;
    fn load(&mut self, key: &EntryKey) -> Result<Option<StoredEntry>, String>;
    fn put(&mut self, entry: StoredEntry, expires_at_unix_ms: u64) -> Result<(), String>;
    fn refresh(&mut self, key: &EntryKey, expires_at_unix_ms: u64) -> Result<(), String>;
    fn remove(&mut self, key: &EntryKey) -> Result<(), String>;
}

pub struct FilesystemSnapshotStore {
    root: PathBuf,
}

impl FilesystemSnapshotStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|error| {
            format!(
                "failed to create cache directory {}: {error}",
                root.display()
            )
        })?;
        Ok(Self { root })
    }

    fn entry_path(&self, key: &EntryKey) -> Result<PathBuf, String> {
        let mut parts = key.0.split('/');
        let namespace = parts.next().ok_or("cache key is missing namespace")?;
        let digest = parts.next().ok_or("cache key is missing digest")?;
        if parts.next().is_some() || !safe_component(namespace) || !safe_component(digest) {
            return Err("cache key contains invalid path components".to_string());
        }
        Ok(self.root.join(namespace).join(digest))
    }

    fn delete_path(path: &Path) -> Result<(), String> {
        match fs::remove_dir_all(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!(
                "failed to remove cache entry {}: {error}",
                path.display()
            )),
        }
    }
}

impl PersistentSnapshotStore for FilesystemSnapshotStore {
    fn scan(&mut self, namespace: &str, _now_unix_ms: u64) -> Result<Vec<ScannedEntry>, String> {
        if !safe_component(namespace) {
            return Err("cache namespace is not path-safe".to_string());
        }
        let directory = self.root.join(namespace);
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(format!(
                    "failed to scan cache directory {}: {error}",
                    directory.display()
                ));
            }
        };
        let mut scanned = Vec::new();
        for item in entries {
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    tracing::warn!(phase = "cache.persistence_error", error = %error);
                    continue;
                }
            };
            let path = item.path();
            let Some(digest) = item.file_name().to_str().map(str::to_owned) else {
                let _ = Self::delete_path(&path);
                continue;
            };
            if !item.file_type().is_ok_and(|kind| kind.is_dir())
                || !safe_component(&digest)
                || digest.starts_with(".tmp-")
            {
                let _ = Self::delete_path(&path);
                continue;
            }
            match fs::read(path.join("manifest.json")) {
                Ok(manifest) => scanned.push(ScannedEntry {
                    key: EntryKey(format!("{namespace}/{digest}")),
                    manifest,
                }),
                Err(error) => {
                    tracing::warn!(phase = "cache.persistence_error", error = %error, entry = %digest);
                    let _ = Self::delete_path(&path);
                }
            }
        }
        Ok(scanned)
    }

    fn load(&mut self, key: &EntryKey) -> Result<Option<StoredEntry>, String> {
        let path = self.entry_path(key)?;
        let manifest = match fs::read(path.join("manifest.json")) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to read cache manifest: {error}")),
        };
        let mut blobs = Vec::new();
        for index in 0usize.. {
            let blob_path = path.join(format!("{index:05}.blob"));
            match fs::read(&blob_path) {
                Ok(blob) => blobs.push(blob),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => {
                    return Err(format!(
                        "failed to read cache blob {}: {error}",
                        blob_path.display()
                    ));
                }
            }
        }
        Ok(Some(StoredEntry {
            key: key.clone(),
            manifest,
            blobs,
        }))
    }

    fn put(&mut self, entry: StoredEntry, _expires_at_unix_ms: u64) -> Result<(), String> {
        let destination = self.entry_path(&entry.key)?;
        let parent = destination.parent().ok_or("cache entry has no parent")?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create cache namespace: {error}"))?;
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary = parent.join(format!(".tmp-{}-{unique}", std::process::id()));
        fs::create_dir(&temporary)
            .map_err(|error| format!("failed to create temporary cache entry: {error}"))?;
        let write_result = (|| {
            for (index, blob) in entry.blobs.iter().enumerate() {
                write_synced(&temporary.join(format!("{index:05}.blob")), blob)?;
            }
            write_synced(&temporary.join("manifest.json"), &entry.manifest)?;
            File::open(&temporary)
                .and_then(|file| file.sync_all())
                .map_err(|error| format!("failed to sync temporary cache entry: {error}"))?;
            if destination.exists() {
                Self::delete_path(&destination)?;
            }
            fs::rename(&temporary, &destination)
                .map_err(|error| format!("failed to publish cache entry: {error}"))?;
            File::open(parent)
                .and_then(|file| file.sync_all())
                .map_err(|error| format!("failed to sync cache namespace: {error}"))?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = Self::delete_path(&temporary);
        }
        write_result
    }

    fn refresh(&mut self, key: &EntryKey, expires_at_unix_ms: u64) -> Result<(), String> {
        let path = self.entry_path(key)?;
        let manifest_path = path.join("manifest.json");
        let bytes = fs::read(&manifest_path)
            .map_err(|error| format!("failed to read cache manifest for refresh: {error}"))?;
        let mut value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse cache manifest for refresh: {error}"))?;
        let object = value
            .as_object_mut()
            .ok_or("cache manifest is not an object")?;
        let expiry = object
            .get_mut("expires_at_unix_ms")
            .ok_or("cache manifest is missing expiry")?;
        *expiry = serde_json::Value::from(expires_at_unix_ms);
        let refreshed = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
        let temporary = path.join("manifest.json.tmp");
        write_synced(&temporary, &refreshed)?;
        fs::rename(&temporary, &manifest_path)
            .map_err(|error| format!("failed to publish refreshed cache manifest: {error}"))?;
        File::open(&path)
            .and_then(|file| file.sync_all())
            .map_err(|error| format!("failed to sync refreshed cache entry: {error}"))
    }

    fn remove(&mut self, key: &EntryKey) -> Result<(), String> {
        Self::delete_path(&self.entry_path(key)?)
    }
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", path.display()))
}

fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}
