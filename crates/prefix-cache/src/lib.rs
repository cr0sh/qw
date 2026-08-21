mod codec;
mod store;
#[cfg(test)]
mod tests;
mod trie;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use qw_runtime::{PortablePromptSnapshot, PromptSnapshot};
use serde::{Deserialize, Serialize};

pub use codec::{Manifest, ResponseResumeMetadata, namespace_hash};
use codec::{
    RetentionMetadata, decode, encode_portable, entry_key, parse_manifest, validate_resume_metadata,
};
pub use store::{FilesystemSnapshotStore, PersistentSnapshotStore, ScannedEntry, StoredEntry};
use trie::{RadixTrie, Terminal};

const INITIAL_TTL_MS: u64 = 2 * 60 * 60 * 1000;
const MAX_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1000;
const IO_QUEUE_CAPACITY: usize = 64;
const MAX_CHECKPOINT_INTERVAL: usize = 256;
pub trait Clock: Send {
    fn now_unix_ms(&self) -> u64;
}

struct SystemClock;

impl Clock for SystemClock {
    fn now_unix_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntryKey(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SnapshotRoute {
    Baseline,
    Mtp,
}

impl SnapshotRoute {
    pub fn matches(self, snapshot: &PromptSnapshot) -> bool {
        matches!(
            (self, snapshot),
            (Self::Baseline, PromptSnapshot::Baseline(_)) | (Self::Mtp, PromptSnapshot::Mtp(_))
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Mtp => "mtp",
        }
    }
}
#[derive(Debug, Clone)]
pub struct CacheNamespaces {
    pub baseline: String,
    pub mtp: String,
}

impl CacheNamespaces {
    fn get(&self, route: SnapshotRoute) -> &str {
        match route {
            SnapshotRoute::Baseline => &self.baseline,
            SnapshotRoute::Mtp => &self.mtp,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (SnapshotRoute, &str)> {
        [
            (SnapshotRoute::Baseline, self.baseline.as_str()),
            (SnapshotRoute::Mtp, self.mtp.as_str()),
        ]
        .into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub memory_bytes: u64,
    pub directory: Option<PathBuf>,
    pub filesystem_bytes: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_bytes: 2 * 1024 * 1024 * 1024,
            directory: None,
            filesystem_bytes: 20 * 1024 * 1024 * 1024,
        }
    }
}

impl CacheConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.memory_bytes == 0 {
            return Err("prefix cache memory capacity must be nonzero".to_string());
        }
        if self.directory.is_some() && self.filesystem_bytes == 0 {
            return Err("prefix cache filesystem capacity must be nonzero".to_string());
        }
        Ok(())
    }
}

pub struct PrefixMatch<'a> {
    pub token_count: usize,
    pub snapshot: &'a PromptSnapshot,
}

pub struct ResumeEntry {
    pub token_ids: Vec<i32>,
    pub snapshot: PromptSnapshot,
    pub metadata: ResponseResumeMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeLookupError {
    NotFound,
    Mismatch,
}

pub struct AdaptivePrefixCache {
    namespaces: CacheNamespaces,
    trie: RadixTrie,
    memory_cap: u64,
    memory_bytes: u64,
    filesystem_cap: Option<u64>,
    filesystem_bytes: u64,
    io: Option<CacheIo>,
    clock: Box<dyn Clock>,
    resumes: HashMap<String, (SnapshotRoute, Vec<i32>)>,
}

struct CacheIo {
    tx: SyncSender<IoCommand>,
    refreshes: Arc<Mutex<HashMap<EntryKey, u64>>>,
    refresh_enqueued: Arc<AtomicBool>,
}

enum IoCommand {
    Load {
        key: EntryKey,
        reply: mpsc::Sender<Result<Option<StoredEntry>, String>>,
    },
    Put {
        namespace: String,
        route: SnapshotRoute,
        token_ids: Vec<i32>,
        portable: PortablePromptSnapshot,
        retention: RetentionMetadata,
        expires_at_unix_ms: u64,
        response_resume: Option<ResponseResumeMetadata>,
    },
    Remove(EntryKey),
    FlushRefresh,
    Flush(mpsc::Sender<()>),
}

impl AdaptivePrefixCache {
    pub fn new(namespaces: CacheNamespaces, config: CacheConfig) -> Result<Self, String> {
        config.validate()?;
        let store = config
            .directory
            .as_ref()
            .map(FilesystemSnapshotStore::new)
            .transpose()?
            .map(|store| Box::new(store) as Box<dyn PersistentSnapshotStore>);
        Self::with_optional_store(namespaces, config, store, Box::new(SystemClock))
    }

    pub fn with_store(
        namespaces: CacheNamespaces,
        config: CacheConfig,
        store: Box<dyn PersistentSnapshotStore>,
    ) -> Result<Self, String> {
        config.validate()?;
        Self::with_optional_store(namespaces, config, Some(store), Box::new(SystemClock))
    }

    pub fn with_store_and_clock(
        namespaces: CacheNamespaces,
        config: CacheConfig,
        store: Box<dyn PersistentSnapshotStore>,
        clock: Box<dyn Clock>,
    ) -> Result<Self, String> {
        config.validate()?;
        Self::with_optional_store(namespaces, config, Some(store), clock)
    }

    fn with_optional_store(
        namespaces: CacheNamespaces,
        config: CacheConfig,
        mut store: Option<Box<dyn PersistentSnapshotStore>>,
        clock: Box<dyn Clock>,
    ) -> Result<Self, String> {
        if namespaces.iter().any(|(_, namespace)| {
            namespace.is_empty() || !namespace.bytes().all(|byte| byte.is_ascii_hexdigit())
        }) {
            return Err("prefix cache namespaces must be hexadecimal digests".to_string());
        }
        let now = clock.now_unix_ms();
        let mut trie = RadixTrie::new();
        let mut resumes = HashMap::new();
        let mut filesystem_bytes = 0u64;
        if let Some(store) = store.as_mut() {
            for (expected_route, namespace) in namespaces.iter() {
                let scanned = store.scan(namespace, now)?;
                for scanned_entry in scanned {
                    match parse_manifest(namespace, &scanned_entry.manifest) {
                        Ok(manifest)
                            if manifest.expires_at_unix_ms > now
                                && manifest.route == expected_route =>
                        {
                            let expected_key =
                                entry_key(namespace, manifest.route, &manifest.token_ids);
                            if expected_key != scanned_entry.key {
                                tracing::warn!(
                                    phase = "cache.persistence_error",
                                    error = "entry digest mismatch"
                                );
                                let _ = store.remove(&scanned_entry.key);
                                continue;
                            }
                            filesystem_bytes =
                                filesystem_bytes.saturating_add(manifest.total_bytes);
                            trie.ensure(
                                &manifest.token_ids,
                                manifest.route,
                                Terminal {
                                    route: manifest.route,
                                    observations: manifest.retention.observations,
                                    reuse_count: manifest.retention.reuse_count,
                                    last_access_unix_ms: manifest.retention.last_access_unix_ms,
                                    expires_at_unix_ms: manifest.expires_at_unix_ms,
                                    serialized_bytes: manifest.total_bytes,
                                    snapshot: None,
                                    persistent_key: Some(scanned_entry.key),
                                    response_resume: manifest.response_resume.clone(),
                                },
                            );
                            if let Some(resume) = manifest.response_resume {
                                resumes.insert(
                                    resume.response_id,
                                    (manifest.route, manifest.token_ids),
                                );
                            }
                        }
                        Ok(_) => {
                            let _ = store.remove(&scanned_entry.key);
                        }
                        Err(error) => {
                            tracing::warn!(phase = "cache.persistence_error", error = %error);
                            let _ = store.remove(&scanned_entry.key);
                        }
                    }
                }
            }
        }
        let io = store.map(spawn_io_thread);
        let mut cache = Self {
            namespaces,
            trie,
            memory_cap: config.memory_bytes,
            memory_bytes: 0,
            filesystem_cap: config
                .directory
                .as_ref()
                .map(|_| config.filesystem_bytes)
                .or_else(|| io.as_ref().map(|_| config.filesystem_bytes)),
            filesystem_bytes,
            io,
            clock,
            resumes,
        };
        cache.evict_persistent(now);
        tracing::info!(
            phase = "cache.insert",
            tier = "memory",
            capacity_bytes = cache.memory_cap
        );
        if let Some(capacity) = cache.filesystem_cap {
            tracing::info!(
                phase = "cache.insert",
                tier = "filesystem",
                capacity_bytes = capacity
            );
        }
        Ok(cache)
    }

    pub fn checkpoint_lengths(
        &mut self,
        tokens: &[i32],
        required: &[usize],
        route: SnapshotRoute,
    ) -> Vec<usize> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let now = self.clock.now_unix_ms();
        self.expire(now);
        self.trie
            .observe(tokens, route, now, now.saturating_add(INITIAL_TTL_MS));
        let mut lengths = required
            .iter()
            .copied()
            .filter(|length| *length > 0 && *length <= tokens.len())
            .collect::<Vec<_>>();
        lengths.extend(
            (MAX_CHECKPOINT_INTERVAL..=tokens.len()).step_by(MAX_CHECKPOINT_INTERVAL),
        );
        lengths.push(tokens.len());
        let structural =
            self.trie
                .path(tokens, route)
                .into_iter()
                .rev()
                .find_map(|(node, length)| {
                    let terminal = self.trie.terminal(node, route)?;
                    (length < tokens.len()
                        && terminal.observations >= 2
                        && terminal.snapshot.is_none()
                        && terminal.persistent_key.is_none())
                    .then_some(length)
                });
        if let Some(common) = structural {
            let boundary = required
                .iter()
                .copied()
                .filter(|length| *length <= common)
                .max()
                .unwrap_or((common / MAX_CHECKPOINT_INTERVAL) * MAX_CHECKPOINT_INTERVAL);
            if boundary > 0 {
                lengths.push(boundary);
            }
        }
        lengths.sort_unstable();
        lengths.dedup();
        lengths
    }

    pub fn lookup(&mut self, prompt: &[i32], route: SnapshotRoute) -> Option<PrefixMatch<'_>> {
        let now = self.clock.now_unix_ms();
        self.expire(now);
        let candidates = self.trie.path(prompt, route);
        let mut hit = None;
        for (node, length) in candidates.into_iter().rev() {
            let has_snapshot = self
                .trie
                .terminal(node, route)
                .is_some_and(|terminal| terminal.snapshot.is_some());
            if has_snapshot {
                hit = Some((node, length));
                break;
            }
            let key = self
                .trie
                .terminal(node, route)
                .and_then(|terminal| terminal.persistent_key.clone());
            if let Some(key) = key {
                if self.load_persistent(node, route, &key, now) {
                    hit = Some((node, length));
                    break;
                }
            }
        }
        let Some((node, token_count)) = hit else {
            tracing::debug!(phase = "cache.lookup", hit = false, route = route.as_str());
            return None;
        };
        let (key, refresh_expiry) = {
            let terminal = self
                .trie
                .terminal_mut(node, route)
                .expect("lookup terminal exists");
            terminal.reuse_count = terminal.reuse_count.saturating_add(1);
            terminal.last_access_unix_ms = now;
            let ttl = reuse_ttl_ms(terminal.reuse_count);
            let remaining = terminal.expires_at_unix_ms.saturating_sub(now);
            if remaining < ttl / 2 {
                terminal.expires_at_unix_ms = now.saturating_add(ttl);
                (
                    terminal.persistent_key.clone(),
                    Some(terminal.expires_at_unix_ms),
                )
            } else {
                (None, None)
            }
        };
        if let (Some(key), Some(expiry)) = (key, refresh_expiry) {
            self.queue_refresh(key, expiry);
        }
        tracing::debug!(
            phase = "cache.lookup",
            hit = true,
            route = route.as_str(),
            cached_tokens = token_count
        );
        let snapshot = self.trie.terminal(node, route)?.snapshot.as_ref()?;
        Some(PrefixMatch {
            token_count,
            snapshot,
        })
    }

    pub fn take_resume(
        &mut self,
        response_id: &str,
        request_fingerprint: &str,
        expected_route: SnapshotRoute,
    ) -> Result<ResumeEntry, ResumeLookupError> {
        let now = self.clock.now_unix_ms();
        self.expire(now);
        let (route, token_ids) = self
            .resumes
            .get(response_id)
            .cloned()
            .ok_or(ResumeLookupError::NotFound)?;
        if route != expected_route {
            return Err(ResumeLookupError::NotFound);
        }
        let Some((node, length)) = self.trie.path(&token_ids, route).into_iter().last() else {
            self.resumes.remove(response_id);
            return Err(ResumeLookupError::NotFound);
        };
        if length != token_ids.len() {
            self.resumes.remove(response_id);
            return Err(ResumeLookupError::NotFound);
        }
        let metadata = self
            .trie
            .terminal(node, route)
            .and_then(|terminal| terminal.response_resume.clone())
            .ok_or(ResumeLookupError::NotFound)?;
        if metadata.request_fingerprint != request_fingerprint {
            return Err(ResumeLookupError::Mismatch);
        }
        if self
            .trie
            .terminal(node, route)
            .is_some_and(|terminal| terminal.snapshot.is_none())
        {
            let key = self
                .trie
                .terminal(node, route)
                .and_then(|terminal| terminal.persistent_key.clone())
                .ok_or(ResumeLookupError::NotFound)?;
            if !self.load_persistent(node, route, &key, now) {
                self.resumes.remove(response_id);
                return Err(ResumeLookupError::NotFound);
            }
        }
        let terminal = self
            .trie
            .remove_terminal(node, route)
            .ok_or(ResumeLookupError::NotFound)?;
        self.resumes.remove(response_id);
        self.memory_bytes = self.memory_bytes.saturating_sub(
            terminal
                .snapshot
                .as_ref()
                .map_or(0, |snapshot| snapshot.nbytes() as u64),
        );
        self.filesystem_bytes = self
            .filesystem_bytes
            .saturating_sub(terminal.serialized_bytes);
        if let Some(key) = terminal.persistent_key {
            self.try_io(IoCommand::Remove(key));
        }
        let snapshot = terminal.snapshot.ok_or(ResumeLookupError::NotFound)?;
        tracing::debug!(phase = "cache.resume", response_id, route = route.as_str());
        Ok(ResumeEntry {
            token_ids,
            snapshot,
            metadata,
        })
    }

    pub fn insert(&mut self, tokens: &[i32], snapshots: Vec<PromptSnapshot>, route: SnapshotRoute) {
        self.insert_snapshots(tokens, snapshots, route, None);
    }

    pub fn insert_resume(
        &mut self,
        tokens: &[i32],
        snapshot: PromptSnapshot,
        route: SnapshotRoute,
        metadata: ResponseResumeMetadata,
    ) {
        if let Err(error) = validate_resume_metadata(tokens, Some(&metadata)) {
            tracing::warn!(phase = "cache.persistence_error", error = %error);
            return;
        }
        self.insert_snapshots(tokens, vec![snapshot], route, Some(metadata));
    }

    fn insert_snapshots(
        &mut self,
        tokens: &[i32],
        snapshots: Vec<PromptSnapshot>,
        route: SnapshotRoute,
        response_resume: Option<ResponseResumeMetadata>,
    ) {
        let now = self.clock.now_unix_ms();
        self.expire(now);
        for snapshot in snapshots {
            let token_len = snapshot.token_len();
            if token_len == 0 || token_len > tokens.len() || !route.matches(&snapshot) {
                continue;
            }
            let prefix = &tokens[..token_len];
            let resume = (token_len == tokens.len())
                .then(|| response_resume.clone())
                .flatten();
            let node = self.trie.ensure_node(prefix);
            let (previous, previous_persistent_bytes, previous_resume) = self
                .trie
                .terminal_mut(node, route)
                .map(|terminal| {
                    (
                        terminal.snapshot.take(),
                        terminal
                            .persistent_key
                            .is_some()
                            .then_some(terminal.serialized_bytes),
                        terminal.response_resume.take(),
                    )
                })
                .unwrap_or((None, None, None));
            if let Some(previous) = previous {
                self.memory_bytes = self.memory_bytes.saturating_sub(previous.nbytes() as u64);
            }
            if let Some(previous_bytes) = previous_persistent_bytes {
                self.filesystem_bytes = self.filesystem_bytes.saturating_sub(previous_bytes);
            }
            if let Some(previous_resume) = previous_resume {
                self.resumes.remove(&previous_resume.response_id);
            }
            let bytes = snapshot.nbytes() as u64;
            let expiry = now.saturating_add(INITIAL_TTL_MS);
            let namespace = self.namespaces.get(route);
            let key = entry_key(namespace, route, prefix);
            let (observations, reuse_count, last_access) =
                if let Some(terminal) = self.trie.terminal_mut(node, route) {
                    terminal.snapshot = Some(snapshot);
                    terminal.expires_at_unix_ms = expiry;
                    terminal.serialized_bytes = bytes;
                    terminal.persistent_key = self.io.as_ref().map(|_| key.clone());
                    terminal.response_resume = resume.clone();
                    (
                        terminal.observations.max(1),
                        terminal.reuse_count,
                        terminal.last_access_unix_ms,
                    )
                } else {
                    self.trie.ensure(
                        prefix,
                        route,
                        Terminal {
                            route,
                            observations: 1,
                            reuse_count: 0,
                            last_access_unix_ms: now,
                            expires_at_unix_ms: expiry,
                            serialized_bytes: bytes,
                            snapshot: Some(snapshot),
                            persistent_key: self.io.as_ref().map(|_| key.clone()),
                            response_resume: resume.clone(),
                        },
                    );
                    (1, 0, now)
                };
            self.memory_bytes = self.memory_bytes.saturating_add(bytes);
            if let Some(resume) = resume.clone() {
                self.resumes
                    .insert(resume.response_id.clone(), (route, prefix.to_vec()));
            }
            // MLX array handles are thread-bound and !Send, so materialize only after the
            // hot snapshot is installed, then hand portable bytes to the I/O thread.
            let portable = self.io.as_ref().and_then(|_| {
                self.trie
                    .terminal(node, route)
                    .and_then(|terminal| terminal.snapshot.as_ref())
                    .and_then(|snapshot| {
                        snapshot
                            .to_portable()
                            .map_err(|error| {
                                tracing::warn!(
                                    phase = "cache.persistence_error",
                                    error = %error
                                )
                            })
                            .ok()
                    })
            });
            if let Some(portable) = portable {
                self.filesystem_bytes = self.filesystem_bytes.saturating_add(bytes);
                self.try_io(IoCommand::Put {
                    namespace: self.namespaces.get(route).to_string(),
                    route,
                    token_ids: prefix.to_vec(),
                    portable,
                    retention: RetentionMetadata {
                        observations,
                        reuse_count,
                        last_access_unix_ms: last_access,
                    },
                    expires_at_unix_ms: expiry,
                    response_resume: resume,
                });
            }
            tracing::debug!(
                phase = "cache.insert",
                route = route.as_str(),
                token_count = token_len,
                snapshot_bytes = bytes
            );
        }
        self.evict_memory();
        self.evict_persistent(now);
    }

    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub fn flush_persistence(&self) {
        let Some(io) = &self.io else {
            return;
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if io.tx.send(IoCommand::Flush(reply_tx)).is_ok() {
            let _ = reply_rx.recv();
        }
    }

    fn load_persistent(
        &mut self,
        node: usize,
        route: SnapshotRoute,
        key: &EntryKey,
        now: u64,
    ) -> bool {
        let Some(io) = &self.io else {
            return false;
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if io
            .tx
            .send(IoCommand::Load {
                key: key.clone(),
                reply: reply_tx,
            })
            .is_err()
        {
            return false;
        }
        let loaded = reply_rx.recv().ok().and_then(Result::ok).flatten();
        let Some(loaded) = loaded else {
            if let Some(terminal) = self.trie.terminal_mut(node, route) {
                terminal.persistent_key = None;
            }
            return false;
        };
        let namespace = self.namespaces.get(route);
        match decode(namespace, &loaded.manifest, loaded.blobs) {
            Ok(decoded)
                if decoded.manifest.expires_at_unix_ms > now
                    && decoded.manifest.route == route
                    && entry_key(
                        namespace,
                        decoded.manifest.route,
                        &decoded.manifest.token_ids,
                    ) == *key =>
            {
                let bytes = decoded.snapshot.nbytes() as u64;
                let terminal = self
                    .trie
                    .terminal_mut(node, route)
                    .expect("persistent terminal exists");
                terminal.snapshot = Some(decoded.snapshot);
                terminal.response_resume = decoded.manifest.response_resume;
                self.memory_bytes = self.memory_bytes.saturating_add(bytes);
                self.evict_memory();
                tracing::debug!(
                    phase = "cache.promote",
                    from = "filesystem",
                    to = "memory",
                    route = route.as_str()
                );
                self.trie
                    .terminal(node, route)
                    .is_some_and(|terminal| terminal.snapshot.is_some())
            }
            Ok(_) | Err(_) => {
                tracing::warn!(
                    phase = "cache.persistence_error",
                    error = "stale or corrupt cache entry"
                );
                if let Some(terminal) = self.trie.terminal_mut(node, route) {
                    terminal.persistent_key = None;
                }
                self.try_io(IoCommand::Remove(key.clone()));
                false
            }
        }
    }

    fn expire(&mut self, now: u64) {
        for (node, route) in self.trie.terminal_ids() {
            if self
                .trie
                .terminal(node, route)
                .is_some_and(|terminal| terminal.expires_at_unix_ms <= now)
            {
                if let Some(terminal) = self.trie.remove_terminal(node, route) {
                    if let Some(resume) = &terminal.response_resume {
                        self.resumes.remove(&resume.response_id);
                    }
                    if let Some(snapshot) = terminal.snapshot {
                        self.memory_bytes =
                            self.memory_bytes.saturating_sub(snapshot.nbytes() as u64);
                    }
                    self.filesystem_bytes = self
                        .filesystem_bytes
                        .saturating_sub(terminal.serialized_bytes);
                    if let Some(key) = terminal.persistent_key {
                        self.try_io(IoCommand::Remove(key));
                    }
                    tracing::debug!(phase = "cache.expire", route = route.as_str());
                }
            }
        }
    }

    fn evict_memory(&mut self) {
        while self.memory_bytes > self.memory_cap {
            let victim = self
                .trie
                .terminal_ids()
                .into_iter()
                .filter(|(node, route)| {
                    self.trie
                        .terminal(*node, *route)
                        .is_some_and(|terminal| terminal.snapshot.is_some())
                })
                .min_by_key(|(node, route)| {
                    let t = self.trie.terminal(*node, *route).unwrap();
                    (t.reuse_count, t.last_access_unix_ms)
                });
            let Some((node, route)) = victim else {
                break;
            };
            let terminal = self.trie.terminal_mut(node, route).unwrap();
            let snapshot = terminal.snapshot.take().unwrap();
            self.memory_bytes = self.memory_bytes.saturating_sub(snapshot.nbytes() as u64);
            tracing::debug!(
                phase = "cache.evict",
                tier = "memory",
                route = route.as_str()
            );
        }
    }

    fn evict_persistent(&mut self, _now: u64) {
        let Some(cap) = self.filesystem_cap else {
            return;
        };
        while self.filesystem_bytes > cap {
            let victim = self
                .trie
                .terminal_ids()
                .into_iter()
                .filter(|(node, route)| {
                    self.trie
                        .terminal(*node, *route)
                        .is_some_and(|terminal| terminal.persistent_key.is_some())
                })
                .min_by_key(|(node, route)| {
                    let t = self.trie.terminal(*node, *route).unwrap();
                    (t.reuse_count, t.last_access_unix_ms)
                });
            let Some((node, route)) = victim else {
                break;
            };
            let terminal = self.trie.terminal_mut(node, route).unwrap();
            let key = terminal.persistent_key.take().unwrap();
            self.filesystem_bytes = self
                .filesystem_bytes
                .saturating_sub(terminal.serialized_bytes);
            self.try_io(IoCommand::Remove(key));
            tracing::debug!(
                phase = "cache.evict",
                tier = "filesystem",
                route = route.as_str()
            );
        }
    }

    fn try_io(&self, command: IoCommand) {
        let Some(io) = &self.io else {
            return;
        };
        match io.tx.try_send(command) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => tracing::warn!(
                phase = "cache.persistence_error",
                error = "cache I/O queue is full"
            ),
            Err(TrySendError::Disconnected(_)) => tracing::warn!(
                phase = "cache.persistence_error",
                error = "cache I/O thread stopped"
            ),
        }
    }

    fn queue_refresh(&self, key: EntryKey, expires_at_unix_ms: u64) {
        let Some(io) = &self.io else {
            return;
        };
        io.refreshes
            .lock()
            .expect("refresh map lock")
            .insert(key, expires_at_unix_ms);
        if !io.refresh_enqueued.swap(true, Ordering::AcqRel)
            && io.tx.try_send(IoCommand::FlushRefresh).is_err()
        {
            io.refresh_enqueued.store(false, Ordering::Release);
            tracing::warn!(
                phase = "cache.persistence_error",
                error = "cache I/O queue is full"
            );
        }
    }
}

fn spawn_io_thread(mut store: Box<dyn PersistentSnapshotStore>) -> CacheIo {
    let (tx, rx) = mpsc::sync_channel(IO_QUEUE_CAPACITY);
    let refreshes = Arc::new(Mutex::new(HashMap::new()));
    let refresh_enqueued = Arc::new(AtomicBool::new(false));
    let thread_refreshes = Arc::clone(&refreshes);
    let thread_refresh_enqueued = Arc::clone(&refresh_enqueued);
    thread::Builder::new()
        .name("qw-prefix-cache-io".to_string())
        .spawn(move || io_loop(&mut *store, rx, thread_refreshes, thread_refresh_enqueued))
        .expect("failed to spawn prefix cache I/O thread");
    CacheIo {
        tx,
        refreshes,
        refresh_enqueued,
    }
}

fn io_loop(
    store: &mut dyn PersistentSnapshotStore,
    rx: Receiver<IoCommand>,
    refreshes: Arc<Mutex<HashMap<EntryKey, u64>>>,
    refresh_enqueued: Arc<AtomicBool>,
) {
    while let Ok(command) = rx.recv() {
        let result = match command {
            IoCommand::Load { key, reply } => {
                let _ = reply.send(store.load(&key));
                continue;
            }
            IoCommand::Put {
                namespace,
                route,
                token_ids,
                portable,
                retention,
                expires_at_unix_ms,
                response_resume,
            } => encode_portable(
                &namespace,
                route,
                &token_ids,
                portable,
                retention,
                expires_at_unix_ms,
                response_resume,
            )
            .and_then(|encoded| {
                store.put(
                    StoredEntry {
                        key: encoded.key,
                        manifest: encoded.manifest,
                        blobs: encoded.blobs,
                    },
                    expires_at_unix_ms,
                )
            }),
            IoCommand::Remove(key) => store.remove(&key),
            IoCommand::FlushRefresh => {
                loop {
                    let pending = {
                        let mut pending = refreshes.lock().expect("refresh map lock");
                        std::mem::take(&mut *pending)
                    };
                    for (key, expiry) in pending {
                        if let Err(error) = store.refresh(&key, expiry) {
                            tracing::warn!(phase = "cache.persistence_error", error = %error);
                        }
                    }
                    refresh_enqueued.store(false, Ordering::Release);
                    if refreshes.lock().expect("refresh map lock").is_empty()
                        || refresh_enqueued.swap(true, Ordering::AcqRel)
                    {
                        break;
                    }
                }
                continue;
            }
            IoCommand::Flush(reply) => {
                let _ = reply.send(());
                continue;
            }
        };
        if let Err(error) = result {
            tracing::warn!(phase = "cache.persistence_error", error = %error);
        }
    }
}

fn reuse_ttl_ms(reuse_count: u64) -> u64 {
    let power = u64::BITS - (reuse_count.saturating_add(1)).leading_zeros() - 1;
    INITIAL_TTL_MS
        .saturating_mul(1u64.checked_shl(power).unwrap_or(u64::MAX))
        .min(MAX_TTL_MS)
}
