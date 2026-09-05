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
    #[cfg(feature = "dflash2")]
    Dflash2,
}

impl SnapshotRoute {
    pub fn matches(self, snapshot: &PromptSnapshot) -> bool {
        match self {
            Self::Baseline => matches!(snapshot, PromptSnapshot::Baseline(_)),
            Self::Mtp => matches!(snapshot, PromptSnapshot::Mtp(_)),
            #[cfg(feature = "dflash2")]
            Self::Dflash2 => matches!(snapshot, PromptSnapshot::Dflash2(_)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Mtp => "mtp",
            #[cfg(feature = "dflash2")]
            Self::Dflash2 => "dflash2",
        }
    }
}
#[derive(Debug, Clone)]
pub struct CacheNamespaces {
    pub baseline: String,
    pub mtp: String,
    #[cfg(feature = "dflash2")]
    pub dflash2: String,
}

impl CacheNamespaces {
    fn get(&self, route: SnapshotRoute) -> &str {
        match route {
            SnapshotRoute::Baseline => &self.baseline,
            SnapshotRoute::Mtp => &self.mtp,
            #[cfg(feature = "dflash2")]
            SnapshotRoute::Dflash2 => &self.dflash2,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (SnapshotRoute, &str)> {
        [
            (SnapshotRoute::Baseline, self.baseline.as_str()),
            (SnapshotRoute::Mtp, self.mtp.as_str()),
            #[cfg(feature = "dflash2")]
            (SnapshotRoute::Dflash2, self.dflash2.as_str()),
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

/// Persistent resume records and writes already in flight may temporarily exceed the
/// evictable filesystem capacity, but together they may use at most one extra capacity.
fn filesystem_hard_cap(capacity: u64) -> u64 {
    capacity.saturating_add(capacity)
}

pub struct AdaptivePrefixCache {
    namespaces: CacheNamespaces,
    trie: RadixTrie,
    memory_cap: u64,
    memory_bytes: u64,
    memory_pages: HashMap<u64, (usize, u64)>,
    filesystem_cap: Option<u64>,
    filesystem_bytes: u64,
    pending_filesystem_bytes: u64,
    filesystem_blobs: HashMap<String, (usize, u64)>,
    io: Option<CacheIo>,
    clock: Box<dyn Clock>,
    resumes: HashMap<String, (SnapshotRoute, Vec<i32>)>,
}

struct CacheIo {
    tx: SyncSender<IoCommand>,
    completions: Receiver<PutCompletion>,
    refreshes: Arc<Mutex<HashMap<EntryKey, u64>>>,
    refresh_enqueued: Arc<AtomicBool>,
}

struct PutCompletion {
    route: SnapshotRoute,
    token_ids: Vec<i32>,
    key: EntryKey,
    reserved_bytes: u64,
    result: Result<(Vec<(String, u64)>, u64), String>,
}
enum IoCommand {
    Load {
        key: EntryKey,
        reply: mpsc::Sender<Result<Option<StoredEntry>, String>>,
    },
    Put {
        key: EntryKey,
        namespace: String,
        route: SnapshotRoute,
        reserved_bytes: u64,
        token_ids: Vec<i32>,
        portable: PortablePromptSnapshot,
        retention: RetentionMetadata,
        expires_at_unix_ms: u64,
        response_resume: Option<ResponseResumeMetadata>,
    },
    Remove(EntryKey),
    RemoveSync {
        key: EntryKey,
        reply: mpsc::Sender<Result<(), String>>,
    },
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
        let mut filesystem_blobs: HashMap<String, (usize, u64)> = HashMap::new();
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
                            for (digest, size) in manifest_blob_refs(&manifest) {
                                let entry = filesystem_blobs.entry(digest).or_insert((0, size));
                                entry.0 += 1;
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
                                    page_refs: Vec::new(),
                                    local_bytes: 0,
                                    blob_refs: manifest_blob_refs(&manifest),
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
            memory_pages: HashMap::new(),
            filesystem_cap: config
                .directory
                .as_ref()
                .map(|_| config.filesystem_bytes)
                .or_else(|| io.as_ref().map(|_| config.filesystem_bytes)),
            filesystem_bytes: filesystem_blobs.values().map(|(_, bytes)| *bytes).sum(),
            pending_filesystem_bytes: 0,
            filesystem_blobs,
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
    fn drain_put_completions(&mut self) {
        let Some(io) = &self.io else {
            return;
        };
        while let Ok(completion) = io.completions.try_recv() {
            self.pending_filesystem_bytes = self
                .pending_filesystem_bytes
                .saturating_sub(completion.reserved_bytes);
            let Some((node, _)) = self
                .trie
                .path(&completion.token_ids, completion.route)
                .into_iter()
                .last()
            else {
                continue;
            };
            if self
                .trie
                .terminal(node, completion.route)
                .is_none_or(|terminal| terminal.persistent_key.as_ref() != Some(&completion.key))
            {
                continue;
            }
            let terminal = self.trie.terminal_mut(node, completion.route).unwrap();
            match completion.result {
                Ok((refs, bytes)) => {
                    terminal.blob_refs = refs;
                    terminal.serialized_bytes = bytes;
                }
                Err(error) => {
                    terminal.persistent_key = None;
                    terminal.blob_refs.clear();
                    terminal.serialized_bytes = 0;
                    tracing::warn!(phase = "cache.persistence_error", error = %error);
                }
            }
        }
        self.rebuild_accounting();
        self.evict_persistent(self.clock.now_unix_ms());
    }

    pub fn checkpoint_lengths(
        &mut self,
        tokens: &[i32],
        required: &[usize],
        route: SnapshotRoute,
    ) -> Vec<usize> {
        self.drain_put_completions();
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
            lengths.push(common);
        }
        lengths.sort_unstable();
        lengths.dedup();
        lengths
    }

    pub fn lookup(&mut self, prompt: &[i32], route: SnapshotRoute) -> Option<PrefixMatch<'_>> {
        self.drain_put_completions();
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
        self.drain_put_completions();
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
            self.queue_remove(key);
        }
        let snapshot = terminal.snapshot.ok_or(ResumeLookupError::NotFound)?;
        self.rebuild_accounting();
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
        self.drain_put_completions();
        let now = self.clock.now_unix_ms();
        let filesystem_before = self.filesystem_bytes;
        let mut inserted_logical_bytes = 0u64;
        let mut inserted_unique_page_bytes = 0u64;
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
            let (previous, previous_persistent_key, previous_resume) = self
                .trie
                .terminal_mut(node, route)
                .map(|terminal| {
                    let previous_persistent_key = terminal.persistent_key.take();
                    terminal.blob_refs.clear();
                    terminal.serialized_bytes = 0;
                    (
                        terminal.snapshot.take(),
                        previous_persistent_key,
                        terminal.response_resume.take(),
                    )
                })
                .unwrap_or((None, None, None));
            if let Some(previous) = previous {
                self.memory_bytes = self.memory_bytes.saturating_sub(previous.nbytes() as u64);
            }
            if let Some(previous_key) = previous_persistent_key {
                self.queue_remove(previous_key);
            }
            if let Some(previous_resume) = previous_resume {
                self.resumes.remove(&previous_resume.response_id);
            }
            self.rebuild_accounting();
            let bytes = snapshot.nbytes() as u64;
            let summary = snapshot.storage_summary();
            inserted_logical_bytes = inserted_logical_bytes.saturating_add(bytes);
            inserted_unique_page_bytes = inserted_unique_page_bytes.saturating_add(
                summary
                    .pages
                    .iter()
                    .map(|(_, page_bytes)| *page_bytes as u64)
                    .sum::<u64>(),
            );
            let expiry = now.saturating_add(INITIAL_TTL_MS);
            let namespace = self.namespaces.get(route);
            let key = entry_key(namespace, route, prefix);
            let persist = self.io.is_some() && self.can_reserve_persistence(bytes);
            let (observations, reuse_count, last_access) =
                if let Some(terminal) = self.trie.terminal_mut(node, route) {
                    terminal.snapshot = Some(snapshot);
                    terminal.page_refs = summary.pages.clone();
                    terminal.local_bytes = summary.local_bytes;
                    terminal.expires_at_unix_ms = expiry;
                    terminal.serialized_bytes = 0;
                    terminal.persistent_key = persist.then(|| key.clone());
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
                            serialized_bytes: 0,
                            page_refs: summary.pages.clone(),
                            local_bytes: summary.local_bytes,
                            blob_refs: Vec::new(),
                            snapshot: Some(snapshot),
                            persistent_key: persist.then(|| key.clone()),
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
            let portable = persist
                .then(|| {
                    self.trie
                        .terminal(node, route)
                        .and_then(|terminal| terminal.snapshot.as_ref())
                        .and_then(|snapshot| snapshot.to_portable().ok())
                })
                .flatten();
            if let Some(portable) = portable {
                let queued = self.try_io(IoCommand::Put {
                    key: key.clone(),
                    namespace: self.namespaces.get(route).to_string(),
                    route,
                    token_ids: prefix.to_vec(),
                    portable,
                    reserved_bytes: bytes,
                    retention: RetentionMetadata {
                        observations,
                        reuse_count,
                        last_access_unix_ms: last_access,
                    },
                    expires_at_unix_ms: expiry,
                    response_resume: resume,
                });
                if queued {
                    self.pending_filesystem_bytes =
                        self.pending_filesystem_bytes.saturating_add(bytes);
                } else if let Some(terminal) = self.trie.terminal_mut(node, route) {
                    terminal.persistent_key = None;
                    terminal.blob_refs.clear();
                    terminal.serialized_bytes = 0;
                }
            } else if let Some(terminal) = self.trie.terminal_mut(node, route) {
                terminal.persistent_key = None;
                terminal.blob_refs.clear();
                terminal.serialized_bytes = 0;
            }
            tracing::debug!(
                phase = "cache.insert",
                route = route.as_str(),
                token_count = token_len,
                logical_snapshot_bytes = bytes,
                unique_page_bytes = summary
                    .pages
                    .iter()
                    .map(|(_, page_bytes)| *page_bytes as u64)
                    .sum::<u64>(),
            );
        }
        self.rebuild_accounting();
        let mut blobs = HashMap::<String, (usize, u64)>::new();
        for (node, route) in self.trie.terminal_ids() {
            if let Some(t) = self.trie.terminal(node, route) {
                for (digest, bytes) in &t.blob_refs {
                    let entry = blobs.entry(digest.clone()).or_insert((0, *bytes));
                    entry.0 += 1;
                }
            }
        }
        self.filesystem_blobs = blobs;
        self.filesystem_bytes = self
            .filesystem_blobs
            .values()
            .map(|(_, bytes)| *bytes)
            .sum();
        self.evict_memory();
        self.evict_persistent(now);
        tracing::debug!(
            phase = "cache.insert.summary",
            route = route.as_str(),
            logical_snapshot_bytes = inserted_logical_bytes,
            unique_page_bytes = inserted_unique_page_bytes,
            filesystem_unique_delta_bytes =
                self.filesystem_bytes as i128 - filesystem_before as i128,
        );
    }

    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub fn flush_persistence(&mut self) {
        let Some(io) = &self.io else {
            return;
        };
        let tx = io.tx.clone();
        let (reply_tx, reply_rx) = mpsc::channel();
        if tx.send(IoCommand::Flush(reply_tx)).is_ok() {
            let _ = reply_rx.recv();
            self.drain_put_completions();
            let (reply_tx, reply_rx) = mpsc::channel();
            if tx.send(IoCommand::Flush(reply_tx)).is_ok() {
                let _ = reply_rx.recv();
                self.drain_put_completions();
                let (reply_tx, reply_rx) = mpsc::channel();
                if tx.send(IoCommand::Flush(reply_tx)).is_ok() {
                    let _ = reply_rx.recv();
                }
            }
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
        let loaded = match reply_rx.recv() {
            Ok(Ok(Some(loaded))) => loaded,
            Ok(Ok(None)) => {
                if let Some(terminal) = self.trie.terminal_mut(node, route) {
                    terminal.persistent_key = None;
                    terminal.blob_refs.clear();
                    terminal.serialized_bytes = 0;
                }
                self.rebuild_accounting();
                return false;
            }
            Ok(Err(error)) => {
                tracing::warn!(phase = "cache.persistence_error", error = %error);
                if let Some(terminal) = self.trie.terminal_mut(node, route) {
                    terminal.persistent_key = None;
                    terminal.blob_refs.clear();
                    terminal.serialized_bytes = 0;
                }
                self.rebuild_accounting();
                self.remove_persistent_sync(key);
                return false;
            }
            Err(_) => return false,
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
                let summary = decoded.snapshot.storage_summary();
                terminal.snapshot = Some(decoded.snapshot);
                terminal.page_refs = summary.pages;
                terminal.local_bytes = summary.local_bytes;
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
                    terminal.blob_refs.clear();
                    terminal.serialized_bytes = 0;
                }
                self.rebuild_accounting();
                self.remove_persistent_sync(key);
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
                        self.queue_remove(key);
                    }
                    tracing::debug!(phase = "cache.expire", route = route.as_str());
                }
            }
        }
        self.rebuild_accounting();
    }

    fn evict_memory(&mut self) {
        while self.memory_bytes > self.memory_cap {
            let victim = self
                .trie
                .terminal_ids()
                .into_iter()
                .filter(|(node, route)| {
                    self.trie.terminal(*node, *route).is_some_and(|terminal| {
                        terminal.snapshot.is_some() && terminal.response_resume.is_none()
                    })
                })
                .min_by_key(|(node, route)| {
                    let t = self.trie.terminal(*node, *route).unwrap();
                    (t.reuse_count, t.last_access_unix_ms)
                });
            let Some((node, route)) = victim else {
                break;
            };
            let before = self.memory_bytes;
            let terminal = self.trie.terminal_mut(node, route).unwrap();
            terminal.snapshot.take().unwrap();
            terminal.page_refs.clear();
            terminal.local_bytes = 0;
            self.rebuild_accounting();
            let reclaimed_bytes = before.saturating_sub(self.memory_bytes);
            tracing::debug!(
                phase = "cache.evict",
                tier = "memory",
                route = route.as_str(),
                reclaimed_bytes,
            );
        }
        self.rebuild_accounting();
    }

    fn evict_persistent(&mut self, _now: u64) {
        let Some(cap) = self.filesystem_cap else {
            return;
        };
        self.evict_persistent_to(cap, false);
        self.evict_persistent_to(filesystem_hard_cap(cap), true);
        self.rebuild_accounting();
    }

    fn evict_persistent_to(&mut self, limit: u64, include_resumes: bool) {
        while self.filesystem_bytes > limit {
            let victim = self
                .trie
                .terminal_ids()
                .into_iter()
                .filter(|(node, route)| {
                    self.trie.terminal(*node, *route).is_some_and(|terminal| {
                        terminal.persistent_key.is_some()
                            && (include_resumes || terminal.response_resume.is_none())
                    })
                })
                .min_by_key(|(node, route)| {
                    let t = self.trie.terminal(*node, *route).unwrap();
                    (t.reuse_count, t.last_access_unix_ms)
                });
            let Some((node, route)) = victim else {
                break;
            };
            let before = self.filesystem_bytes;
            let terminal = self.trie.terminal_mut(node, route).unwrap();
            let key = terminal.persistent_key.take().unwrap();
            terminal.blob_refs.clear();
            terminal.serialized_bytes = 0;
            self.rebuild_accounting();
            let reclaimed_bytes = before.saturating_sub(self.filesystem_bytes);
            self.queue_remove(key);
            tracing::debug!(
                phase = "cache.evict",
                tier = "filesystem",
                route = route.as_str(),
                reclaimed_bytes,
            );
        }
    }

    fn can_reserve_persistence(&self, bytes: u64) -> bool {
        self.filesystem_cap.is_some_and(|cap| {
            self.filesystem_bytes
                .saturating_add(self.pending_filesystem_bytes)
                .saturating_add(bytes)
                <= filesystem_hard_cap(cap)
        })
    }

    fn try_io(&self, command: IoCommand) -> bool {
        let Some(io) = &self.io else {
            return false;
        };
        match io.tx.try_send(command) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    phase = "cache.persistence_error",
                    error = "cache I/O queue is full"
                );
                false
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::warn!(
                    phase = "cache.persistence_error",
                    error = "cache I/O thread stopped"
                );
                false
            }
        }
    }
    fn queue_remove(&self, key: EntryKey) {
        let Some(io) = &self.io else {
            return;
        };
        if io.tx.send(IoCommand::Remove(key)).is_err() {
            tracing::warn!(
                phase = "cache.persistence_error",
                error = "cache I/O thread stopped before queuing cache removal"
            );
        }
    }
    fn remove_persistent_sync(&self, key: &EntryKey) {
        let Some(io) = &self.io else {
            return;
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if io
            .tx
            .send(IoCommand::RemoveSync {
                key: key.clone(),
                reply: reply_tx,
            })
            .is_err()
        {
            tracing::warn!(
                phase = "cache.persistence_error",
                error = "cache I/O thread stopped before removing corrupt entry"
            );
            return;
        }
        match reply_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(phase = "cache.persistence_error", error = %error);
            }
            Err(_) => {
                tracing::warn!(
                    phase = "cache.persistence_error",
                    error = "cache I/O thread stopped before reporting corrupt entry removal"
                );
            }
        }
    }

    fn rebuild_accounting(&mut self) {
        let mut pages = HashMap::<u64, (usize, u64)>::new();
        for (node, route) in self.trie.terminal_ids() {
            let Some(t) = self.trie.terminal(node, route) else {
                continue;
            };
            for &(id, bytes) in &t.page_refs {
                let entry = pages.entry(id).or_insert((0, bytes as u64));
                entry.0 += 1;
            }
        }
        self.memory_pages = pages;
        self.memory_bytes = self
            .memory_pages
            .values()
            .map(|(_, bytes)| *bytes)
            .sum::<u64>()
            + self
                .trie
                .terminal_ids()
                .into_iter()
                .filter_map(|(n, r)| self.trie.terminal(n, r).map(|t| t.local_bytes as u64))
                .sum::<u64>();
        let mut blobs = HashMap::<String, (usize, u64)>::new();
        for (node, route) in self.trie.terminal_ids() {
            if let Some(t) = self.trie.terminal(node, route) {
                for (digest, bytes) in &t.blob_refs {
                    let entry = blobs.entry(digest.clone()).or_insert((0, *bytes));
                    entry.0 += 1;
                }
            }
        }
        self.filesystem_blobs = blobs;
        self.filesystem_bytes = self
            .filesystem_blobs
            .values()
            .map(|(_, bytes)| *bytes)
            .sum();
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

fn manifest_blob_refs(manifest: &Manifest) -> Vec<(String, u64)> {
    let mut refs = HashMap::<String, u64>::new();
    for array in &manifest.arrays {
        refs.entry(array.blob_sha256.clone())
            .or_insert(array.byte_len);
    }
    for tensor in &manifest.paged_tensors {
        for page in &tensor.pages {
            refs.entry(page.blob_sha256.clone())
                .or_insert(page.byte_len);
        }
    }
    refs.into_iter().collect()
}

fn spawn_io_thread(mut store: Box<dyn PersistentSnapshotStore>) -> CacheIo {
    let (tx, rx) = mpsc::sync_channel(IO_QUEUE_CAPACITY);
    let (completion_tx, completion_rx) = mpsc::channel();
    let refreshes = Arc::new(Mutex::new(HashMap::new()));
    let refresh_enqueued = Arc::new(AtomicBool::new(false));
    let thread_refreshes = Arc::clone(&refreshes);
    let thread_refresh_enqueued = Arc::clone(&refresh_enqueued);
    thread::Builder::new()
        .name("qw-prefix-cache-io".to_string())
        .spawn(move || {
            io_loop(
                &mut *store,
                rx,
                completion_tx,
                thread_refreshes,
                thread_refresh_enqueued,
            )
        })
        .expect("failed to spawn prefix cache I/O thread");
    CacheIo {
        tx,
        completions: completion_rx,
        refreshes,
        refresh_enqueued,
    }
}

fn io_loop(
    store: &mut dyn PersistentSnapshotStore,
    rx: Receiver<IoCommand>,
    completion_tx: mpsc::Sender<PutCompletion>,
    refreshes: Arc<Mutex<HashMap<EntryKey, u64>>>,
    refresh_enqueued: Arc<AtomicBool>,
) {
    while let Ok(command) = rx.recv() {
        let result = match command {
            IoCommand::Load { key, reply } => {
                let _ = reply.send(store.load(&key));
                continue;
            }
            IoCommand::RemoveSync { key, reply } => {
                let _ = reply.send(store.remove(&key));
                continue;
            }
            IoCommand::Put {
                key,
                namespace,
                route,
                reserved_bytes,
                token_ids,
                portable,
                retention,
                expires_at_unix_ms,
                response_resume,
            } => {
                let completion_tokens = token_ids.clone();
                let result = encode_portable(
                    &namespace,
                    route,
                    &token_ids,
                    portable,
                    retention,
                    expires_at_unix_ms,
                    response_resume,
                )
                .and_then(|encoded| {
                    let refs = encoded
                        .blobs
                        .iter()
                        .map(|blob| (blob.sha256.clone(), blob.bytes.len() as u64))
                        .collect();
                    let bytes = encoded
                        .blobs
                        .iter()
                        .map(|blob| blob.bytes.len() as u64)
                        .sum();
                    store.put(
                        StoredEntry {
                            key: encoded.key,
                            manifest: encoded.manifest,
                            blobs: encoded.blobs,
                        },
                        expires_at_unix_ms,
                    )?;
                    Ok((refs, bytes))
                });
                let _ = completion_tx.send(PutCompletion {
                    route,
                    token_ids: completion_tokens,
                    key,
                    reserved_bytes,
                    result,
                });
                continue;
            }
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
