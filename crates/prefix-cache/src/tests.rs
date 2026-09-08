use super::*;
use crate::codec::ContentBlob;
use mlxcel_core::generate::ModelStateSnapshot;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct InfoCounter(
    Arc<AtomicUsize>,
    Option<Arc<Mutex<Vec<HashMap<String, String>>>>>,
);

#[derive(Default)]
struct RecordedFields(HashMap<String, String>);

impl tracing::field::Visit for RecordedFields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl tracing::Subscriber for InfoCounter {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        self.1.is_some() || *metadata.level() <= tracing::Level::INFO
    }

    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(if self.1.is_some() {
            tracing::level_filters::LevelFilter::DEBUG
        } else {
            tracing::level_filters::LevelFilter::INFO
        })
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::INFO {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(events) = &self.1 {
            let mut fields = RecordedFields::default();
            event.record(&mut fields);
            events.lock().expect("events lock").push(fields.0);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}

    fn clone_span(&self, span: &tracing::span::Id) -> tracing::span::Id {
        span.clone()
    }
}

const NAMESPACE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MTP_NAMESPACE: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
#[cfg(feature = "dflash2")]
const DFLASH_NAMESPACE: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

fn namespaces() -> CacheNamespaces {
    CacheNamespaces {
        baseline: NAMESPACE.to_string(),
        mtp: MTP_NAMESPACE.to_string(),
        #[cfg(feature = "dflash2")]
        dflash2: DFLASH_NAMESPACE.to_string(),
    }
}

fn snapshot(token_len: usize, values: &[f32]) -> PromptSnapshot {
    let array = mlxcel_core::from_slice_f32(values, &[values.len() as i32]);
    let mut snapshot = ModelStateSnapshot::new("test", token_len);
    snapshot.push_tensor("state", &array);
    snapshot.set_continuation_logits(&array);
    PromptSnapshot::Baseline(snapshot)
}
fn paged_snapshot_chain() -> Vec<PromptSnapshot> {
    let values = (0..768 * 2).map(|i| i as f32).collect::<Vec<_>>();
    let array = mlxcel_core::from_slice_f32(&values, &[1, 768, 2]);
    let mut first = ModelStateSnapshot::new("test", 256);
    first
        .push_paged_tensor(None, "kv", &array, 1)
        .expect("first page");
    let mut second = ModelStateSnapshot::new("test", 512);
    second
        .push_paged_tensor(Some(&first), "kv", &array, 1)
        .expect("second pages");
    let mut third = ModelStateSnapshot::new("test", 768);
    third
        .push_paged_tensor(Some(&second), "kv", &array, 1)
        .expect("third pages");
    vec![
        PromptSnapshot::Baseline(first),
        PromptSnapshot::Baseline(second),
        PromptSnapshot::Baseline(third),
    ]
}

fn resume_metadata(response_id: &str, fingerprint: &str) -> ResponseResumeMetadata {
    ResponseResumeMetadata {
        response_id: response_id.to_string(),
        message_id: "msg_original".to_string(),
        created_unix_seconds: 123,
        prompt_token_count: 2,
        request_fingerprint: fingerprint.to_string(),
        generated_token_ids: vec![3, 4],
        raw_text: "partial".to_string(),
        emitted_reasoning_text: String::new(),
        emitted_content_text: "partial".to_string(),
        original_max_tokens: 8,
        continuation_seed: 0x6a09_e667_bb67_ae85,
    }
}

fn memory_config(memory_bytes: u64) -> CacheConfig {
    CacheConfig {
        memory_bytes,
        directory: None,
        filesystem_bytes: 1024 * 1024,
    }
}

#[derive(Clone)]
struct ManualClock(Arc<AtomicU64>);
impl Clock for ManualClock {
    fn now_unix_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}
impl ManualClock {
    fn new(now: u64) -> Self {
        Self(Arc::new(AtomicU64::new(now)))
    }
    fn set(&self, now: u64) {
        self.0.store(now, Ordering::SeqCst);
    }
}

struct EmptyStore;
impl PersistentSnapshotStore for EmptyStore {
    fn scan(&mut self, _namespace: &str, _now_unix_ms: u64) -> Result<Vec<ScannedEntry>, String> {
        Ok(Vec::new())
    }
    fn load(&mut self, _key: &EntryKey) -> Result<Option<StoredEntry>, String> {
        Ok(None)
    }
    fn put(&mut self, _entry: StoredEntry, _expires_at_unix_ms: u64) -> Result<(), String> {
        Ok(())
    }
    fn refresh(&mut self, _key: &EntryKey, _expires_at_unix_ms: u64) -> Result<(), String> {
        Ok(())
    }
    fn remove(&mut self, _key: &EntryKey) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn persistent_store_contract_requires_no_filesystem_types() {
    fn accepts_backend<T: PersistentSnapshotStore>() {}
    accepts_backend::<EmptyStore>();
}

#[derive(Default)]
struct RecordingState {
    entries: HashMap<EntryKey, (Vec<u8>, Vec<ContentBlob>)>,
    refreshes: Vec<(EntryKey, u64)>,
    fail_put: bool,
    demand_loads: usize,
    prefetch_loads: usize,
    fail_load: bool,
    read_gate: Option<Arc<(Mutex<BlockingState>, Condvar)>>,
}

struct RecordingStore(Arc<Mutex<RecordingState>>);

impl PersistentSnapshotStore for RecordingStore {
    fn scan(&mut self, namespace: &str, _now_unix_ms: u64) -> Result<Vec<ScannedEntry>, String> {
        let state = self.0.lock().expect("recording store lock");
        Ok(state
            .entries
            .iter()
            .filter(|(key, _)| key.0.starts_with(namespace))
            .map(|(key, (manifest, _))| ScannedEntry {
                key: key.clone(),
                manifest: manifest.clone(),
            })
            .collect())
    }

    fn load(&mut self, key: &EntryKey) -> Result<Option<StoredEntry>, String> {
        let mut state = self.0.lock().expect("recording store lock");
        state.demand_loads += 1;
        if state.fail_load {
            return Err("injected load failure".into());
        }
        Ok(state.entries.get(key).map(|(manifest, blobs)| StoredEntry {
            key: key.clone(),
            manifest: manifest.clone(),
            blobs: blobs.clone(),
        }))
    }
    fn manifest_bytes(&mut self, key: &EntryKey) -> Result<Option<u64>, String> {
        Ok(self
            .0
            .lock()
            .expect("store")
            .entries
            .get(key)
            .map(|(manifest, _)| manifest.len() as u64))
    }

    fn load_bounded(&mut self, key: &EntryKey, limit: u64) -> Result<Option<StoredEntry>, String> {
        let mut state = self.0.lock().unwrap();
        state.prefetch_loads += 1;
        if state.fail_load {
            return Err("injected load failure".into());
        }
        let Some((manifest, blobs)) = state.entries.get(key) else {
            return Ok(None);
        };
        let bytes = manifest.len() as u64 + blobs.iter().map(|b| b.bytes.len() as u64).sum::<u64>();
        if bytes > limit {
            return Err("prefetch byte limit".into());
        }
        let entry = StoredEntry {
            key: key.clone(),
            manifest: manifest.clone(),
            blobs: blobs.clone(),
        };
        let gate = state.read_gate.clone();
        drop(state);
        if let Some(gate) = gate {
            let (lock, wake) = &*gate;
            let mut state = lock.lock().expect("read gate");
            state.entered = true;
            wake.notify_all();
            while !state.released {
                state = wake.wait(state).expect("read gate");
            }
        }
        Ok(Some(entry))
    }

    fn put(&mut self, entry: StoredEntry, _expires_at_unix_ms: u64) -> Result<(), String> {
        let mut state = self.0.lock().expect("recording store lock");
        if state.fail_put {
            return Err("injected put failure".to_string());
        }
        state
            .entries
            .insert(entry.key, (entry.manifest, entry.blobs));
        Ok(())
    }

    fn refresh(&mut self, key: &EntryKey, expires_at_unix_ms: u64) -> Result<(), String> {
        self.0
            .lock()
            .expect("recording store lock")
            .refreshes
            .push((key.clone(), expires_at_unix_ms));
        Ok(())
    }

    fn remove(&mut self, key: &EntryKey) -> Result<(), String> {
        self.0
            .lock()
            .expect("recording store lock")
            .entries
            .remove(key);
        Ok(())
    }
}
#[derive(Default)]
struct BlockingState {
    entered: bool,
    released: bool,
}

struct BlockingStore(Arc<(Mutex<BlockingState>, Condvar)>);

impl PersistentSnapshotStore for BlockingStore {
    fn scan(&mut self, _namespace: &str, _now_unix_ms: u64) -> Result<Vec<ScannedEntry>, String> {
        Ok(Vec::new())
    }

    fn load(&mut self, _key: &EntryKey) -> Result<Option<StoredEntry>, String> {
        Ok(None)
    }

    fn put(&mut self, _entry: StoredEntry, _expires_at_unix_ms: u64) -> Result<(), String> {
        let (state, wake) = &*self.0;
        let mut state = state.lock().expect("blocking store lock");
        state.entered = true;
        wake.notify_all();
        while !state.released {
            state = wake.wait(state).expect("blocking store wait");
        }
        Ok(())
    }

    fn refresh(&mut self, _key: &EntryKey, _expires_at_unix_ms: u64) -> Result<(), String> {
        Ok(())
    }

    fn remove(&mut self, _key: &EntryKey) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn divergent_long_prompt_keeps_only_full_checkpoint() {
    let clock = ManualClock::new(1_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1_000_000),
        Box::new(EmptyStore),
        Box::new(clock),
    )
    .expect("cache");
    assert!(cache.lookup(&[], SnapshotRoute::Baseline).is_none());
    let prompt = (0..1_024).collect::<Vec<i32>>();
    cache.insert(
        &prompt,
        vec![snapshot(prompt.len(), &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );

    let mut divergent_prompt = prompt.clone();
    divergent_prompt[900] = 10_000;
    assert!(
        cache
            .lookup(&divergent_prompt, SnapshotRoute::Baseline)
            .is_none()
    );
}

#[test]
fn ttl_progression_expiry_and_byte_eviction_are_adaptive() {
    assert_eq!(
        (0..8).map(reuse_ttl_ms).collect::<Vec<_>>(),
        vec![
            2 * 60 * 60 * 1000,
            4 * 60 * 60 * 1000,
            4 * 60 * 60 * 1000,
            8 * 60 * 60 * 1000,
            8 * 60 * 60 * 1000,
            8 * 60 * 60 * 1000,
            8 * 60 * 60 * 1000,
            16 * 60 * 60 * 1000,
        ]
    );
    let clock = ManualClock::new(10_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1),
        Box::new(EmptyStore),
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );
    assert_eq!(
        cache.memory_bytes(),
        0,
        "snapshot-byte budget evicts oversized state"
    );

    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1_000_000),
        Box::new(EmptyStore),
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(&[1, 2], vec![snapshot(2, &[1.0])], SnapshotRoute::Baseline);
    clock.set(10_000 + INITIAL_TTL_MS + 1);
    assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Baseline).is_none());
}

#[test]
fn half_life_refreshes_are_coalesced_per_entry() {
    let start = 20_000;
    let clock = ManualClock::new(start);
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1_000_000),
        Box::new(RecordingStore(Arc::clone(&state))),
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(&[1], vec![snapshot(1, &[1.0])], SnapshotRoute::Baseline);
    cache.flush_persistence();

    clock.set(start + 1);
    assert!(cache.lookup(&[1, 2], SnapshotRoute::Baseline).is_some());
    assert!(cache.lookup(&[1, 3], SnapshotRoute::Baseline).is_some());
    cache.flush_persistence();

    {
        let mut state = state.lock().expect("recording store lock");
        assert_eq!(state.refreshes.len(), 1);
        assert_eq!(state.refreshes[0].1, start + 1 + 4 * 60 * 60 * 1000,);
        state.refreshes.clear();
    }

    let key = entry_key(NAMESPACE, SnapshotRoute::Baseline, &[1]);
    let io = cache.io.as_ref().expect("persistent I/O");
    io.refresh_enqueued.store(true, Ordering::Release);
    cache.queue_refresh(key.clone(), 100);
    cache.queue_refresh(key.clone(), 200);
    io.refresh_enqueued.store(false, Ordering::Release);
    cache.queue_refresh(key, 300);
    cache.flush_persistence();
    assert_eq!(
        state.lock().expect("recording store lock").refreshes,
        vec![(entry_key(NAMESPACE, SnapshotRoute::Baseline, &[1]), 300)],
    );
}

#[test]
fn filesystem_byte_cap_evicts_persistent_entries_by_snapshot_bytes() {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let config = CacheConfig {
        memory_bytes: 1_000_000,
        directory: None,
        filesystem_bytes: 1,
    };
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        config,
        Box::new(RecordingStore(Arc::clone(&state))),
    )
    .expect("cache");
    cache.insert(
        &[1],
        vec![snapshot(1, &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    assert_eq!(cache.filesystem_bytes, 0);
    assert!(
        cache.filesystem_blobs.is_empty(),
        "evicted terminal has no blob accounting"
    );
    assert!(
        state
            .lock()
            .expect("recording store lock")
            .entries
            .is_empty(),
        "write-through is followed by persistent eviction under the byte cap",
    );
}

#[test]
fn in_flight_persistence_is_reserved_against_the_grace_ceiling() {
    let blocking = Arc::new((Mutex::new(BlockingState::default()), Condvar::new()));
    let first = snapshot(1, &[1.0, 2.0]);
    let snapshot_bytes = first.nbytes() as u64;
    let config = CacheConfig {
        memory_bytes: 1_000_000,
        directory: None,
        filesystem_bytes: snapshot_bytes / 2,
    };
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        config,
        Box::new(BlockingStore(Arc::clone(&blocking))),
    )
    .expect("cache");
    assert_eq!(
        filesystem_hard_cap(cache.filesystem_cap.unwrap()),
        snapshot_bytes,
        "the first write lands exactly on the grace boundary"
    );
    cache.insert(&[1], vec![first], SnapshotRoute::Baseline);

    let (state_lock, wake) = &*blocking;
    let state = state_lock.lock().expect("blocking store lock");
    let (state, timeout) = wake
        .wait_timeout_while(state, std::time::Duration::from_secs(5), |state| {
            !state.entered
        })
        .expect("blocking store wait");
    assert!(!timeout.timed_out() && state.entered, "write did not start");
    drop(state);

    cache.insert(
        &[2],
        vec![snapshot(1, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    assert_eq!(cache.pending_filesystem_bytes, snapshot_bytes);
    let second_node = cache
        .trie
        .path(&[2], SnapshotRoute::Baseline)
        .into_iter()
        .last()
        .unwrap()
        .0;
    assert!(
        cache
            .trie
            .terminal(second_node, SnapshotRoute::Baseline)
            .unwrap()
            .persistent_key
            .is_none(),
        "a queued write beyond the hard grace ceiling must not be admitted"
    );

    let mut state = state_lock.lock().expect("blocking store lock");
    state.released = true;
    wake.notify_all();
    drop(state);
    cache.flush_persistence();
    assert_eq!(cache.pending_filesystem_bytes, 0);
    assert!(cache.filesystem_bytes <= filesystem_hard_cap(cache.filesystem_cap.unwrap()));
}

#[test]
fn failed_persistence_clears_metadata_but_keeps_memory_snapshot() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(InfoCounter(
        Arc::new(AtomicUsize::new(0)),
        Some(Arc::clone(&events)),
    ));
    let state = Arc::new(Mutex::new(RecordingState {
        fail_put: true,
        ..RecordingState::default()
    }));
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        memory_config(1_000_000),
        Box::new(RecordingStore(Arc::clone(&state))),
    )
    .expect("cache");
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    assert!(
        cache.lookup(&[1, 2, 3], SnapshotRoute::Baseline).is_some(),
        "failed persistence must not evict hot memory",
    );
    let node = cache
        .trie
        .path(&[1, 2], SnapshotRoute::Baseline)
        .into_iter()
        .last()
        .map(|(node, _)| node)
        .expect("terminal path");
    let terminal = cache
        .trie
        .terminal(node, SnapshotRoute::Baseline)
        .expect("terminal");
    assert!(terminal.persistent_key.is_none());
    assert!(terminal.blob_refs.is_empty());
    assert_eq!(terminal.serialized_bytes, 0);
    assert_eq!(cache.filesystem_bytes, 0);
    let expected_id = entry_key(NAMESPACE, SnapshotRoute::Baseline, &[1, 2]).0;
    let events = events.lock().expect("events lock");
    for phase in ["cache.insert", "cache.persistence_error", "cache.lookup"] {
        assert!(
            events.iter().any(|event| {
                event.get("phase").is_some_and(|value| value == phase)
                    && event.get("entry_id") == Some(&expected_id)
            }),
            "{phase} must retain the same entry identity after persistence fails"
        );
    }
}

#[test]
fn persistent_accounting_uses_unique_payload_bytes() {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        memory_config(1_000_000),
        Box::new(RecordingStore(Arc::clone(&state))),
    )
    .expect("cache");
    let tokens = [9, 10];
    cache.insert(
        &tokens,
        vec![snapshot(tokens.len(), &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();

    let key = entry_key(NAMESPACE, SnapshotRoute::Baseline, &tokens);
    let state = state.lock().expect("recording store lock");
    let (manifest_bytes, _) = state.entries.get(&key).expect("stored entry");
    let manifest: Manifest = serde_json::from_slice(manifest_bytes).expect("manifest");
    assert_eq!(cache.filesystem_bytes, manifest.total_bytes);
    assert_ne!(manifest.total_bytes, manifest_bytes.len() as u64);
}

#[test]
fn filesystem_hit_survives_hot_eviction_preference() {
    let directory = TempDirectory::new();
    let state = snapshot(2, &[1.0, 2.0]);
    let expected = state.to_portable().unwrap();
    let capacity = state.nbytes() as u64;
    let mut cache = AdaptivePrefixCache::new(
        namespaces(),
        CacheConfig {
            memory_bytes: capacity,
            directory: Some(directory.path.clone()),
            filesystem_bytes: 1_000_000,
        },
    )
    .expect("cache");
    cache.insert(&[1, 2], vec![state], SnapshotRoute::Baseline);
    cache.insert(
        &[3, 4],
        vec![snapshot(2, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    assert!(cache.lookup(&[3, 4], SnapshotRoute::Baseline).is_some());
    let hit = cache
        .lookup(&[1, 2, 5], SnapshotRoute::Baseline)
        .expect("disk hit remains usable under hot-tier promotion pressure");
    assert_eq!(hit.token_count, 2);
    assert_eq!(hit.snapshot().to_portable().unwrap(), expected);
    drop(hit);
    assert_eq!(cache.memory_bytes(), capacity);
}

#[test]
fn fresh_boundary_displaces_a_popular_old_prefix_in_memory() {
    let clock = ManualClock::new(1_000);
    let old = snapshot(1, &[1.0, 2.0]);
    let mut cache = AdaptivePrefixCache::with_optional_store(
        namespaces(),
        memory_config(old.nbytes() as u64),
        None,
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(&[1], vec![old], SnapshotRoute::Baseline);
    for _ in 0..8 {
        assert_eq!(
            cache
                .lookup(&[1], SnapshotRoute::Baseline)
                .unwrap()
                .token_count,
            1
        );
    }
    clock.set(2_000);
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    assert_eq!(
        cache
            .lookup(&[1, 2, 3], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2,
        "a fresh boundary must survive until its first use despite the old prefix's popularity"
    );
    assert!(cache.lookup(&[1], SnapshotRoute::Baseline).is_none());
}

#[test]
fn longest_hit_does_not_refresh_unused_ancestors_and_upserts_are_recent() {
    let clock = ManualClock::new(1_000);
    let old = snapshot(1, &[1.0, 2.0]);
    let mut cache = AdaptivePrefixCache::with_optional_store(
        namespaces(),
        memory_config(2 * old.nbytes() as u64),
        None,
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(&[1], vec![old], SnapshotRoute::Baseline);
    for _ in 0..8 {
        drop(
            cache
                .lookup(&[1], SnapshotRoute::Baseline)
                .expect("old hit"),
        );
    }
    clock.set(2_000);
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    clock.set(3_000);
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    clock.set(4_000);
    cache.insert(
        &[9],
        vec![snapshot(1, &[5.0, 6.0])],
        SnapshotRoute::Baseline,
    );
    assert!(cache.lookup(&[1], SnapshotRoute::Baseline).is_none());

    // Materializing an existing terminal is activity even without another hit.
    clock.set(5_000);
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[7.0, 8.0])],
        SnapshotRoute::Baseline,
    );
    clock.set(6_000);
    cache.insert(
        &[8],
        vec![snapshot(1, &[9.0, 10.0])],
        SnapshotRoute::Baseline,
    );
    assert!(cache.lookup(&[9], SnapshotRoute::Baseline).is_none());
    let hit = cache
        .lookup(&[1, 2], SnapshotRoute::Baseline)
        .expect("rematerialized boundary");
    assert_eq!(
        hit.snapshot().to_portable().unwrap(),
        snapshot(2, &[7.0, 8.0]).to_portable().unwrap()
    );
}

#[test]
fn filesystem_pressure_keeps_the_recent_boundary_and_reports_the_old_victim() {
    let clock = ManualClock::new(1_000);
    let events = Arc::new(Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(InfoCounter(
        Arc::new(AtomicUsize::new(0)),
        Some(Arc::clone(&events)),
    ));
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1),
        Box::new(RecordingStore(Arc::clone(&state))),
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(
        &[1],
        vec![snapshot(1, &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    cache.filesystem_cap = Some(cache.filesystem_bytes * 2);
    for _ in 0..8 {
        drop(
            cache
                .lookup(&[1], SnapshotRoute::Baseline)
                .expect("old disk hit"),
        );
    }
    clock.set(2_000);
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    clock.set(3_000);
    cache.insert(
        &[9],
        vec![snapshot(1, &[5.0, 6.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    assert_eq!(
        cache
            .lookup(&[1, 2, 3], SnapshotRoute::Baseline)
            .expect("fresh disk boundary")
            .token_count,
        2
    );
    assert!(cache.lookup(&[1], SnapshotRoute::Baseline).is_none());
    let old_id = entry_key(NAMESPACE, SnapshotRoute::Baseline, &[1]).0;
    let boundary_id = entry_key(NAMESPACE, SnapshotRoute::Baseline, &[1, 2]).0;
    let events = events.lock().expect("events lock");
    assert!(events.iter().any(|event| {
        event
            .get("phase")
            .is_some_and(|value| value == "cache.evict")
            && event.get("tier").is_some_and(|value| value == "filesystem")
            && event.get("entry_id") == Some(&old_id)
    }));
    assert!(events.iter().any(|event| {
        event
            .get("phase")
            .is_some_and(|value| value == "cache.restore")
            && event.get("entry_id") == Some(&boundary_id)
    }));
}

#[test]
fn disk_hit_becomes_recent_before_hot_promotion_pressure() {
    let clock = ManualClock::new(1_000);
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let old = snapshot(1, &[1.0, 2.0]);
    let expected = old.to_portable().unwrap();
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(old.nbytes() as u64),
        Box::new(RecordingStore(Arc::clone(&state))),
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert(&[1], vec![old], SnapshotRoute::Baseline);
    clock.set(2_000);
    cache.insert(
        &[2],
        vec![snapshot(1, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    for _ in 0..8 {
        drop(
            cache
                .lookup(&[2], SnapshotRoute::Baseline)
                .expect("popular hot hit"),
        );
    }
    clock.set(3_000);
    let hit = cache
        .lookup(&[1], SnapshotRoute::Baseline)
        .expect("restored hit");
    assert_eq!(hit.snapshot().to_portable().unwrap(), expected);
    drop(hit);
    state.lock().expect("store lock").entries.remove(&entry_key(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &[1],
    ));
    assert_eq!(
        cache
            .lookup(&[1], SnapshotRoute::Baseline)
            .expect("restored state stays hot without its disk copy")
            .snapshot()
            .to_portable()
            .unwrap(),
        expected
    );
}

#[test]
fn memory_only_entry_identity_survives_eviction_and_expiry() {
    let clock = ManualClock::new(1_000);
    let events = Arc::new(Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(InfoCounter(
        Arc::new(AtomicUsize::new(0)),
        Some(Arc::clone(&events)),
    ));
    let old = snapshot(1, &[1.0, 2.0]);
    let mut cache = AdaptivePrefixCache::with_optional_store(
        namespaces(),
        memory_config(old.nbytes() as u64),
        None,
        Box::new(clock.clone()),
    )
    .expect("cache");
    assert!(cache.lookup(&[42], SnapshotRoute::Baseline).is_none());
    cache.insert(&[1], vec![old], SnapshotRoute::Baseline);
    drop(
        cache
            .lookup(&[1], SnapshotRoute::Baseline)
            .expect("memory hit"),
    );
    clock.set(2_000);
    cache.insert(
        &[9],
        vec![snapshot(1, &[3.0, 4.0])],
        SnapshotRoute::Baseline,
    );
    clock.set(MAX_TTL_MS + 3_000);
    assert!(cache.lookup(&[9], SnapshotRoute::Baseline).is_none());
    let id = entry_key(NAMESPACE, SnapshotRoute::Baseline, &[1]).0;
    let events = events.lock().expect("events lock");
    for phase in [
        "cache.insert",
        "cache.lookup",
        "cache.evict",
        "cache.expire",
    ] {
        assert!(
            events.iter().any(|event| {
                event.get("phase").is_some_and(|value| value == phase)
                    && event.get("entry_id") == Some(&id)
            }),
            "missing {phase} for memory-only identity"
        );
    }
    for event in events.iter().filter(|event| {
        event.contains_key("capacity_bytes")
            || event.get("hit").is_some_and(|value| value == "false")
    }) {
        assert!(
            !event.contains_key("entry_id"),
            "setup and misses have no single entry identity"
        );
    }
}

#[test]
fn oversized_snapshot_preserves_a_hot_prefix_that_fits_by_unique_pages() {
    let clock = ManualClock::new(1_000);
    let capacity = 512 * std::mem::size_of::<f32>() as u64;
    let mut cache = AdaptivePrefixCache::with_optional_store(
        namespaces(),
        memory_config(capacity),
        None,
        Box::new(clock.clone()),
    )
    .expect("cache");
    let array = mlxcel_core::from_slice_f32(&[1.0; 512], &[1, 256, 2]);
    let mut state = ModelStateSnapshot::new("test", 256);
    state.push_paged_tensor(None, "key", &array, 1).unwrap();
    let shared_pages = state.paged_tensor("key").unwrap().pages().to_vec();
    state.push_paged_pages("value", 1, shared_pages).unwrap();
    let state = PromptSnapshot::Baseline(state);
    let expected = state.to_portable().unwrap();
    let query = (0..257).collect::<Vec<i32>>();
    cache.insert(&query[..256], vec![state], SnapshotRoute::Baseline);

    clock.set(2_000);
    cache.insert(
        &[999],
        vec![snapshot(1, &[3.0; 512])],
        SnapshotRoute::Baseline,
    );

    let hit = cache
        .lookup(&query, SnapshotRoute::Baseline)
        .expect("an oversized insertion must not evict a prefix that fits");
    assert_eq!(hit.token_count, 256);
    assert_eq!(hit.snapshot().to_portable().unwrap(), expected);
    drop(hit);
    assert_eq!(cache.memory_bytes(), capacity);
}

#[test]
fn oversized_filesystem_prefix_remains_usable_without_hot_residency() {
    let directory = TempDirectory::new();
    let config = CacheConfig {
        memory_bytes: 1,
        directory: Some(directory.path.clone()),
        filesystem_bytes: 1_000_000,
    };
    let tokens = [4, 5, 6];
    let state = snapshot(tokens.len(), &[1.0, 2.0]);
    let expected = state.to_portable().expect("portable snapshot");
    {
        let mut cache = AdaptivePrefixCache::new(namespaces(), config.clone()).expect("cache");
        cache.insert(&tokens, vec![state], SnapshotRoute::Baseline);
        assert_eq!(cache.memory_bytes(), 0);
        // No flush or idle grace: Load must follow the pending Put on the I/O queue.
        let hit = cache
            .lookup(&[4, 5, 6, 7], SnapshotRoute::Baseline)
            .expect("oversized disk prefix is usable immediately");
        assert_eq!(hit.token_count, tokens.len());
        assert_eq!(hit.snapshot().to_portable().unwrap(), expected);
        drop(hit);
        assert_eq!(cache.memory_bytes(), 0);
        assert!(
            cache
                .lookup(&[4, 5, 9, 7], SnapshotRoute::Baseline)
                .is_none()
        );
        cache.flush_persistence();
    }
    let mut restarted = AdaptivePrefixCache::new(namespaces(), config).expect("restart");
    let hit = restarted
        .lookup(&[4, 5, 6, 8], SnapshotRoute::Baseline)
        .expect("oversized prefix survives restart");
    assert_eq!(hit.token_count, tokens.len());
    assert_eq!(hit.snapshot().to_portable().unwrap(), expected);
    drop(hit);
    assert_eq!(restarted.memory_bytes(), 0);
}

#[test]
fn filesystem_restart_promotes_valid_entry_and_deletes_corrupt_payload() {
    let directory = TempDirectory::new();
    let config = CacheConfig {
        memory_bytes: 1_000_000,
        directory: Some(directory.path.clone()),
        filesystem_bytes: 1_000_000,
    };
    let tokens = vec![4, 5, 6];
    let key = entry_key(NAMESPACE, SnapshotRoute::Baseline, &tokens);
    {
        let mut cache = AdaptivePrefixCache::new(namespaces(), config.clone()).expect("cache");
        assert!(cache.lookup(&tokens, SnapshotRoute::Baseline).is_none());
        cache.insert(
            &tokens,
            vec![snapshot(tokens.len(), &[1.0, 2.0])],
            SnapshotRoute::Baseline,
        );
        cache.flush_persistence();
    }
    {
        let mut restarted =
            AdaptivePrefixCache::new(namespaces(), config.clone()).expect("restart");
        let hit = restarted
            .lookup(&[4, 5, 6, 7], SnapshotRoute::Baseline)
            .expect("filesystem hit");
        assert_eq!(hit.token_count, 3);
        restarted.flush_persistence();
    }
    let entry_path = directory
        .path
        .join("entries")
        .join(NAMESPACE)
        .join(format!("{}.json", key.0.split('/').nth(1).unwrap()));
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&entry_path).expect("entry manifest"))
            .expect("manifest");
    let digest = manifest["blob_sha256"][0].as_str().expect("blob digest");
    std::fs::write(directory.path.join("blobs").join(digest), b"corrupt").expect("corrupt blob");
    {
        let mut restarted =
            AdaptivePrefixCache::new(namespaces(), config).expect("restart corrupt");
        assert!(restarted.lookup(&tokens, SnapshotRoute::Baseline).is_none());
    }
    assert!(!entry_path.exists(), "corrupt entry is deletion-as-miss");
}

#[cfg(feature = "dflash2")]
fn dflash_portable(token_len: usize, hidden_offset: usize) -> PortablePromptSnapshot {
    use qw_runtime::{PortableArray, PortableModelState, PortablePage, PortablePagedTensor};

    let floats = |shape: Vec<i32>, start: usize| {
        let count = shape.iter().map(|&n| n as usize).product::<usize>();
        PortableArray {
            name: None,
            shape,
            dtype: mlxcel_core::dtype::FLOAT32,
            bytes: (start..start + count)
                .flat_map(|n| (n as f32).to_le_bytes())
                .collect(),
        }
    };
    let logits = floats(vec![1, 1, 2], 99);
    let pages = [(0, 1), (1, token_len)]
        .into_iter()
        .filter(|(start, end)| start < end)
        .map(|(start, end)| {
            let array = floats(vec![1, 1, (end - start) as i32, 2], 1 + start * 2);
            PortablePage {
                token_start: start,
                token_end: end,
                shape: array.shape,
                dtype: array.dtype,
                bytes: Arc::from(array.bytes),
            }
        })
        .collect();
    PortablePromptSnapshot::Dflash2 {
        target: PortableModelState {
            family: "qwen3.5-target-v1".to_string(),
            token_len,
            tensors: [
                ("meta.layer_count", 1),
                ("layer.0.offset", token_len as i32),
            ]
            .into_iter()
            .map(|(name, value)| PortableArray {
                name: Some(name.into()),
                shape: vec![1],
                dtype: mlxcel_core::dtype::INT32,
                bytes: value.to_le_bytes().to_vec(),
            })
            .collect(),
            paged_tensors: vec![PortablePagedTensor {
                name: "layer.0.keys".into(),
                token_axis: 2,
                token_len,
                pages,
            }],
            continuation_logits: Some(logits.clone()),
        },
        hidden_concat: floats(vec![1, (token_len - hidden_offset) as i32, 2], 10),
        hidden_offset,
        continuation_logits: logits,
    }
}

#[cfg(feature = "dflash2")]
fn encode_dflash(portable: PortablePromptSnapshot) -> Result<codec::EncodedEntry, String> {
    codec::encode_portable(
        DFLASH_NAMESPACE,
        SnapshotRoute::Dflash2,
        &[1, 2, 3],
        portable,
        RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 0,
        },
        INITIAL_TTL_MS,
        None,
        |_| Ok(()),
    )
}

#[cfg(feature = "dflash2")]
#[test]
fn dflash2_roundtrip_preserves_target_pages_hidden_window_and_logits() {
    for hidden_offset in [0, 1] {
        let portable = dflash_portable(3, hidden_offset);
        let encoded = encode_dflash(portable.clone()).expect("encode DFlash2");
        let unique_bytes = 8 + 24 + (3 - hidden_offset) as u64 * 8 + 8;
        assert_eq!(
            encoded
                .blobs
                .iter()
                .map(|b| b.bytes.len() as u64)
                .sum::<u64>(),
            unique_bytes
        );
        let decoded = codec::decode(DFLASH_NAMESPACE, &encoded.manifest, encoded.blobs)
            .expect("decode DFlash2");
        assert_eq!(decoded.manifest.total_bytes, unique_bytes);
        assert_eq!(decoded.snapshot.nbytes() as u64, unique_bytes + 8);
        assert_eq!(
            decoded.snapshot.to_portable().expect("portable DFlash2"),
            portable
        );
    }
}

#[cfg(feature = "dflash2")]
#[test]
fn dflash2_rejects_invalid_boundaries_shapes_and_tensor_roles() {
    let encoded = encode_dflash(dflash_portable(3, 1)).expect("encode DFlash2");
    let original: Manifest = serde_json::from_slice(&encoded.manifest).expect("manifest");
    let mut malformed = Vec::new();
    let mut changed = original.clone();
    changed.hidden_offset = Some(0);
    malformed.push(changed);
    let mut changed = original.clone();
    changed.hidden_offset = Some(usize::MAX);
    malformed.push(changed);
    let mut changed = original.clone();
    changed.hidden_offset = None;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.draft_family = Some("qwen3.5-mtp-draft".into());
    malformed.push(changed);
    let mut changed = original.clone();
    changed.token_len = 2;
    changed.token_ids.pop();
    malformed.push(changed);
    let mut changed = original.clone();
    changed
        .arrays
        .retain(|a| a.role != codec::ArrayRole::DflashContinuation);
    malformed.push(changed);
    let mut changed = original.clone();
    let hidden = changed
        .arrays
        .iter()
        .find(|a| a.role == codec::ArrayRole::DflashHidden)
        .unwrap()
        .clone();
    changed.arrays.push(hidden);
    malformed.push(changed);
    let mut changed = original.clone();
    changed.arrays[0].role = codec::ArrayRole::ModelTensor;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.arrays[0].name = Some("layer.0.keys".into());
    malformed.push(changed);
    let mut changed = original.clone();
    changed
        .arrays
        .iter_mut()
        .find(|a| a.role == codec::ArrayRole::DflashHidden)
        .unwrap()
        .shape = vec![1, 1, 4];
    malformed.push(changed);
    let mut changed = original.clone();
    changed
        .arrays
        .iter_mut()
        .find(|a| a.role == codec::ArrayRole::DflashContinuation)
        .unwrap()
        .shape = vec![1, 2, 1];
    malformed.push(changed);
    let mut changed = original.clone();
    changed
        .arrays
        .iter_mut()
        .find(|a| a.role == codec::ArrayRole::DflashContinuation)
        .unwrap()
        .dtype = mlxcel_core::dtype::INT32;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.arrays[0].byte_len += 1;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.paged_tensors[0].token_len = 2;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.paged_tensors[0].token_axis = 7;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.paged_tensors[0].pages[1].token_start = 0;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.paged_tensors[0].pages[1].shape = vec![1, 2, 2, 1];
    malformed.push(changed);
    let mut changed = original.clone();
    changed.paged_tensors[0].role = codec::ArrayRole::DflashHidden;
    malformed.push(changed);
    let mut changed = original.clone();
    changed.blob_sha256.push(changed.blob_sha256[0].clone());
    malformed.push(changed);
    let mut changed = original.clone();
    changed.total_bytes += 1;
    malformed.push(changed);
    for (index, manifest) in malformed.into_iter().enumerate() {
        let bytes = serde_json::to_vec(&manifest).unwrap();
        assert!(
            codec::parse_manifest(DFLASH_NAMESPACE, &bytes).is_err(),
            "invalid case {index}"
        );
        assert!(
            codec::decode(DFLASH_NAMESPACE, &bytes, encoded.blobs.clone()).is_err(),
            "invalid case {index}"
        );
    }
    let PortablePromptSnapshot::Dflash2 {
        target,
        hidden_concat,
        continuation_logits,
        ..
    } = dflash_portable(3, 1)
    else {
        unreachable!()
    };
    assert!(
        encode_dflash(PortablePromptSnapshot::Dflash2 {
            target,
            hidden_concat,
            hidden_offset: 0,
            continuation_logits,
        })
        .is_err()
    );
    assert!(encode_dflash(dflash_portable(2, 0)).is_err());
    for route in [SnapshotRoute::Baseline, SnapshotRoute::Mtp] {
        let mut changed = original.clone();
        changed.route = route;
        changed.hidden_offset = None;
        if route == SnapshotRoute::Mtp {
            changed.draft_family = Some("qwen3.5-mtp-draft".into());
            changed.draft_offset = Some(2);
        }
        assert!(
            codec::decode(
                DFLASH_NAMESPACE,
                &serde_json::to_vec(&changed).unwrap(),
                encoded.blobs.clone()
            )
            .is_err()
        );
        assert_ne!(entry_key(DFLASH_NAMESPACE, route, &[1, 2, 3]), encoded.key);
    }
}

#[cfg(feature = "dflash2")]
#[test]
fn dflash2_filesystem_restart_isolates_routes_and_deletes_corrupt_payload() {
    let directory = TempDirectory::new();
    let config = CacheConfig {
        memory_bytes: 1_000_000,
        directory: Some(directory.path.clone()),
        filesystem_bytes: 1_000_000,
    };
    let portable = dflash_portable(3, 1);
    let key = entry_key(DFLASH_NAMESPACE, SnapshotRoute::Dflash2, &[1, 2, 3]);
    {
        let mut cache = AdaptivePrefixCache::new(namespaces(), config.clone()).expect("cache");
        cache.insert(
            &[1, 2, 3],
            vec![PromptSnapshot::from_portable(portable.clone()).unwrap()],
            SnapshotRoute::Dflash2,
        );
        assert_eq!(cache.memory_bytes(), 64);
        assert_eq!(
            cache
                .lookup(&[1, 2, 3], SnapshotRoute::Dflash2)
                .unwrap()
                .snapshot()
                .to_portable()
                .unwrap(),
            portable
        );
        assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Baseline).is_none());
        assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Mtp).is_none());
        cache.flush_persistence();
        assert_eq!(cache.filesystem_bytes, 56);
    }
    {
        let mut restarted =
            AdaptivePrefixCache::new(namespaces(), config.clone()).expect("restart");
        let hit = restarted
            .lookup(&[1, 2, 3, 4], SnapshotRoute::Dflash2)
            .expect("DFlash2 disk hit");
        assert_eq!(hit.token_count, 3);
        assert_eq!(hit.snapshot().to_portable().unwrap(), portable);
        assert!(
            restarted
                .lookup(&[1, 2, 3], SnapshotRoute::Baseline)
                .is_none()
        );
        assert!(restarted.lookup(&[1, 2, 3], SnapshotRoute::Mtp).is_none());
        restarted.flush_persistence();
    }
    let entry_path = directory
        .path
        .join("entries")
        .join(format!("{}.json", key.0));
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&entry_path).unwrap()).unwrap();
    let hidden = manifest
        .arrays
        .iter()
        .find(|a| a.role == codec::ArrayRole::DflashHidden)
        .unwrap();
    std::fs::write(
        directory.path.join("blobs").join(&hidden.blob_sha256),
        b"corrupt",
    )
    .unwrap();
    let mut restarted = AdaptivePrefixCache::new(namespaces(), config).expect("restart corrupt");
    assert!(
        restarted
            .lookup(&[1, 2, 3], SnapshotRoute::Dflash2)
            .is_none()
    );
    assert!(!entry_path.exists());
}

#[cfg(feature = "dflash2")]
#[test]
fn dflash2_resume_survives_restart_is_route_safe_and_one_shot() {
    let directory = TempDirectory::new();
    let config = CacheConfig {
        memory_bytes: 1,
        directory: Some(directory.path.clone()),
        filesystem_bytes: 1_000_000,
    };
    let portable = dflash_portable(3, 1);
    let metadata = resume_metadata("resp_dflash", "fingerprint");
    {
        let mut cache = AdaptivePrefixCache::new(namespaces(), config.clone()).unwrap();
        cache.insert_resume(
            &[1, 2, 3],
            PromptSnapshot::from_portable(portable.clone()).unwrap(),
            SnapshotRoute::Dflash2,
            metadata.clone(),
        );
        cache.flush_persistence();
        assert_eq!(
            cache.memory_bytes(),
            64,
            "active resume survives memory pressure"
        );
    }
    {
        let mut restarted = AdaptivePrefixCache::new(namespaces(), config.clone()).unwrap();
        assert!(matches!(
            restarted.take_resume("resp_dflash", "fingerprint", SnapshotRoute::Mtp),
            Err(ResumeLookupError::NotFound)
        ));
        assert!(matches!(
            restarted.take_resume("resp_dflash", "other", SnapshotRoute::Dflash2),
            Err(ResumeLookupError::Mismatch)
        ));
        let resumed = restarted
            .take_resume("resp_dflash", "fingerprint", SnapshotRoute::Dflash2)
            .expect("DFlash2 resume");
        assert_eq!(resumed.token_ids, [1, 2, 3]);
        assert_eq!(resumed.metadata, metadata);
        assert_eq!(resumed.snapshot.to_portable().unwrap(), portable);
        assert_eq!(restarted.memory_bytes(), 0);
        restarted.flush_persistence();
    }
    let mut restarted = AdaptivePrefixCache::new(namespaces(), config).unwrap();
    assert!(matches!(
        restarted.take_resume("resp_dflash", "fingerprint", SnapshotRoute::Dflash2),
        Err(ResumeLookupError::NotFound)
    ));
}

#[cfg(feature = "dflash2")]
#[test]
fn dflash2_hidden_and_logits_count_toward_eviction_and_expiry() {
    let clock = ManualClock::new(10_000);
    let portable = dflash_portable(3, 1);
    let mut cache = AdaptivePrefixCache::with_optional_store(
        namespaces(),
        memory_config(63),
        None,
        Box::new(clock.clone()),
    )
    .unwrap();
    cache.insert(
        &[1, 2, 3],
        vec![PromptSnapshot::from_portable(portable.clone()).unwrap()],
        SnapshotRoute::Dflash2,
    );
    assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Dflash2).is_none());
    assert_eq!(cache.memory_bytes(), 0);
    let directory = TempDirectory::new();
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        CacheConfig {
            filesystem_bytes: 55,
            directory: Some(directory.path.clone()),
            ..memory_config(64)
        },
        Box::new(FilesystemSnapshotStore::new(&directory.path).unwrap()),
        Box::new(clock.clone()),
    )
    .unwrap();
    cache.insert(
        &[1, 2, 3],
        vec![PromptSnapshot::from_portable(portable).unwrap()],
        SnapshotRoute::Dflash2,
    );
    cache.flush_persistence();
    let key = entry_key(DFLASH_NAMESPACE, SnapshotRoute::Dflash2, &[1, 2, 3]);
    assert!(
        !directory
            .path
            .join("entries")
            .join(format!("{}.json", key.0))
            .exists()
    );
    assert_eq!(cache.filesystem_bytes, 0);
    assert_eq!(cache.memory_bytes(), 64);
    assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Dflash2).is_some());
    clock.set(10_000 + INITIAL_TTL_MS + 1);
    assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Dflash2).is_none());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn strict_manifest_rejects_unknown_fields_and_namespace_mismatch() {
    let encoded = codec::encode_portable(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &[1],
        snapshot(1, &[1.0])
            .to_portable()
            .expect("portable snapshot"),
        RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 0,
        },
        INITIAL_TTL_MS,
        None,
        |_| Ok(()),
    )
    .expect("encode");
    let mut value: serde_json::Value =
        serde_json::from_slice(&encoded.manifest).expect("manifest JSON");
    value
        .as_object_mut()
        .unwrap()
        .insert("format_version".to_string(), 1.into());
    let changed = serde_json::to_vec(&value).unwrap();
    assert!(codec::decode(NAMESPACE, &changed, encoded.blobs.clone()).is_err());
    assert!(
        codec::decode(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            &encoded.manifest,
            encoded.blobs.clone()
        )
        .is_err()
    );
    let mut missing: serde_json::Value =
        serde_json::from_slice(&encoded.manifest).expect("manifest JSON");
    missing.as_object_mut().unwrap().remove("response_resume");
    assert!(
        codec::decode(
            NAMESPACE,
            &serde_json::to_vec(&missing).unwrap(),
            encoded.blobs.clone(),
        )
        .is_err()
    );
    let mut missing_draft_family: serde_json::Value =
        serde_json::from_slice(&encoded.manifest).expect("manifest JSON");
    missing_draft_family
        .as_object_mut()
        .unwrap()
        .remove("draft_family");
    assert!(
        codec::parse_manifest(
            NAMESPACE,
            &serde_json::to_vec(&missing_draft_family).unwrap(),
        )
        .is_err()
    );
    let mut baseline_with_draft_family: serde_json::Value =
        serde_json::from_slice(&encoded.manifest).expect("manifest JSON");
    baseline_with_draft_family["draft_family"] = "qwen3.5-mtp-draft".into();
    assert!(
        codec::parse_manifest(
            NAMESPACE,
            &serde_json::to_vec(&baseline_with_draft_family).unwrap(),
        )
        .is_err()
    );
    let mut baseline_with_draft_offset: serde_json::Value =
        serde_json::from_slice(&encoded.manifest).expect("manifest JSON");
    baseline_with_draft_offset["draft_offset"] = 0.into();
    assert!(
        codec::parse_manifest(
            NAMESPACE,
            &serde_json::to_vec(&baseline_with_draft_offset).unwrap(),
        )
        .is_err()
    );
}

#[test]
fn mtp_manifest_round_trip_preserves_route_and_offset_validation() {
    let array = |bytes: Vec<u8>| qw_runtime::PortableArray {
        name: None,
        shape: vec![1, 1, 2],
        dtype: mlxcel_core::dtype::FLOAT32,
        bytes,
    };
    let model = || qw_runtime::PortableModelState {
        family: "qwen3.5-target-v1".to_string(),
        token_len: 1,
        tensors: Vec::new(),
        paged_tensors: Vec::new(),
        continuation_logits: None,
    };
    let draft = qw_runtime::PortableModelState {
        family: "qwen3.5-mtp-draft".to_string(),
        token_len: 0,
        tensors: Vec::new(),
        paged_tensors: Vec::new(),
        continuation_logits: None,
    };
    let portable = qw_runtime::PortablePromptSnapshot::Mtp {
        target: model(),
        draft,
        draft_offset: 0,
        last_hidden: array(vec![0; 8]),
        continuation_logits: array(vec![1; 8]),
    };
    let encoded = codec::encode_portable(
        MTP_NAMESPACE,
        SnapshotRoute::Mtp,
        &[7],
        portable,
        RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 0,
        },
        INITIAL_TTL_MS,
        None,
        |_| Ok(()),
    )
    .expect("encode MTP");
    let manifest: serde_json::Value =
        serde_json::from_slice(&encoded.manifest).expect("manifest JSON");
    assert_eq!(manifest["family"], "qwen3.5-target-v1");
    assert_eq!(manifest["draft_family"], "qwen3.5-mtp-draft");
    let decoded =
        codec::decode(MTP_NAMESPACE, &encoded.manifest, encoded.blobs.clone()).expect("decode MTP");
    assert!(matches!(decoded.snapshot, PromptSnapshot::Mtp(_)));
    let restored = decoded
        .snapshot
        .to_portable()
        .expect("restored portable MTP");
    let qw_runtime::PortablePromptSnapshot::Mtp { target, draft, .. } = restored else {
        panic!("restored baseline snapshot from MTP manifest");
    };
    assert_eq!(target.family, "qwen3.5-target-v1");
    assert_eq!(draft.family, "qwen3.5-mtp-draft");

    for draft_family in [serde_json::Value::Null, "".into()] {
        let mut malformed = manifest.clone();
        malformed["draft_family"] = draft_family;
        assert!(
            codec::parse_manifest(MTP_NAMESPACE, &serde_json::to_vec(&malformed).unwrap(),)
                .is_err()
        );
    }
    let mut missing_offset = manifest;
    missing_offset["draft_offset"] = serde_json::Value::Null;
    assert!(
        codec::parse_manifest(MTP_NAMESPACE, &serde_json::to_vec(&missing_offset).unwrap(),)
            .is_err()
    );
}

#[test]
fn filesystem_namespace_isolation_and_partial_recovery_are_misses() {
    let directory = TempDirectory::new();
    let config = CacheConfig {
        memory_bytes: 1_000_000,
        directory: Some(directory.path.clone()),
        filesystem_bytes: 1_000_000,
    };
    {
        let mut cache = AdaptivePrefixCache::new(namespaces(), config.clone()).expect("cache");
        cache.insert(&[9], vec![snapshot(1, &[9.0])], SnapshotRoute::Baseline);
        cache.flush_persistence();
    }
    let partial = directory
        .path
        .join("entries")
        .join(NAMESPACE)
        .join(".tmp-interrupted");
    std::fs::create_dir_all(partial.parent().unwrap()).expect("partial directory");
    std::fs::write(&partial, b"partial").expect("partial manifest");

    let other_namespace = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let mut isolated = AdaptivePrefixCache::new(
        CacheNamespaces {
            baseline: other_namespace.to_string(),
            mtp: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            #[cfg(feature = "dflash2")]
            dflash2: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string(),
        },
        config.clone(),
    )
    .expect("isolated cache");
    assert!(isolated.lookup(&[9], SnapshotRoute::Baseline).is_none());
    assert!(directory.path.join("entries").join(NAMESPACE).exists());

    let _recovered = AdaptivePrefixCache::new(
        namespaces(),
        CacheConfig {
            memory_bytes: 1_000_000,
            directory: Some(directory.path.clone()),
            filesystem_bytes: 1_000_000,
        },
    )
    .expect("recovered cache");
    assert!(
        !partial.exists(),
        "startup removes interrupted temporary entries"
    );
}

#[cfg(target_vendor = "apple")]
#[test]
fn failed_blob_barrier_defers_publication_and_orphan_retry_until_durable() {
    let encode = |token, value| {
        let portable = snapshot(1, &[value]).to_portable().unwrap();
        let encoded = codec::encode_portable(
            NAMESPACE,
            SnapshotRoute::Baseline,
            &[token],
            portable.clone(),
            RetentionMetadata {
                observations: 1,
                reuse_count: 0,
                last_access_unix_ms: 0,
            },
            INITIAL_TTL_MS,
            None,
            |_| Ok(()),
        )
        .unwrap();
        (
            StoredEntry {
                key: encoded.key,
                manifest: encoded.manifest,
                blobs: encoded.blobs,
            },
            portable,
        )
    };
    let load = |store: &mut FilesystemSnapshotStore, key: &EntryKey| {
        let loaded = store.load(key).unwrap().expect("committed entry");
        codec::decode(NAMESPACE, &loaded.manifest, loaded.blobs)
            .unwrap()
            .snapshot
            .to_portable()
            .unwrap()
    };
    let directory = TempDirectory::new();
    let mut store = FilesystemSnapshotStore::new(&directory.path).unwrap();
    let (old, old_snapshot) = encode(1, 1.0);
    let old_key = old.key.clone();
    store.put(old, INITIAL_TTL_MS).unwrap();
    let (new, new_snapshot) = encode(2, 2.0);
    let new_key = new.key.clone();
    let retry = StoredEntry {
        key: new.key.clone(),
        manifest: new.manifest.clone(),
        blobs: new.blobs.clone(),
    };
    let (entered, blocked) = mpsc::channel();
    let (release, proceed) = mpsc::channel();
    store.before_blob_barrier = Some(Box::new(move || {
        entered.send(()).unwrap();
        proceed.recv().unwrap();
        Err("injected pre-manifest barrier failure".into())
    }));
    let writer = std::thread::spawn(move || store.put(new, INITIAL_TTL_MS));
    blocked
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("writer reached blob barrier");
    let mut reader = FilesystemSnapshotStore::new(&directory.path).unwrap();
    assert_eq!(load(&mut reader, &old_key), old_snapshot);
    assert!(reader.load(&new_key).unwrap().is_none());
    release.send(()).unwrap();
    assert!(writer.join().unwrap().is_err());
    assert_eq!(load(&mut reader, &old_key), old_snapshot);
    assert!(reader.load(&new_key).unwrap().is_none());
    for blob in &retry.blobs {
        assert_eq!(
            std::fs::read(directory.path.join("blobs").join(&blob.sha256)).unwrap(),
            blob.bytes.as_ref(),
        );
    }

    let mut store = FilesystemSnapshotStore::new(&directory.path).unwrap();
    let (entered, blocked) = mpsc::channel();
    let (release, proceed) = mpsc::channel();
    store.before_blob_barrier = Some(Box::new(move || {
        entered.send(()).unwrap();
        proceed.recv().unwrap();
        Ok(())
    }));
    let writer = std::thread::spawn(move || store.put(retry, INITIAL_TTL_MS));
    blocked
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("existing orphan retry reached blob barrier");
    assert_eq!(load(&mut reader, &old_key), old_snapshot);
    assert!(reader.load(&new_key).unwrap().is_none());
    release.send(()).unwrap();
    writer.join().unwrap().unwrap();
    drop(reader);
    let mut reopened = FilesystemSnapshotStore::new(&directory.path).unwrap();
    assert_eq!(load(&mut reopened, &old_key), old_snapshot);
    assert_eq!(load(&mut reopened, &new_key), new_snapshot);
}

struct TempDirectory {
    path: PathBuf,
}
impl TempDirectory {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "qw-prefix-cache-test-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create temp directory");
        Self { path }
    }
}
impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn resume_records_are_mismatch_safe_one_shot_and_expire() {
    let clock = ManualClock::new(50_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1_000_000),
        Box::new(EmptyStore),
        Box::new(clock.clone()),
    )
    .expect("cache");
    cache.insert_resume(
        &[1, 2, 3],
        snapshot(3, &[1.0]),
        SnapshotRoute::Baseline,
        resume_metadata("chatcmpl-original", "fingerprint"),
    );
    assert!(matches!(
        cache.take_resume("chatcmpl-original", "different", SnapshotRoute::Baseline,),
        Err(ResumeLookupError::Mismatch)
    ));
    let resumed = cache
        .take_resume("chatcmpl-original", "fingerprint", SnapshotRoute::Baseline)
        .expect("resume checkpoint");
    assert_eq!(resumed.token_ids, vec![1, 2, 3]);
    assert_eq!(resumed.metadata.message_id, "msg_original");
    assert!(matches!(
        cache.take_resume("chatcmpl-original", "fingerprint", SnapshotRoute::Baseline,),
        Err(ResumeLookupError::NotFound)
    ));

    cache.insert_resume(
        &[5],
        snapshot(1, &[2.0]),
        SnapshotRoute::Baseline,
        resume_metadata("chatcmpl-expired", "fingerprint"),
    );
    clock.set(50_000 + INITIAL_TTL_MS + 1);
    assert!(matches!(
        cache.take_resume("chatcmpl-expired", "fingerprint", SnapshotRoute::Baseline,),
        Err(ResumeLookupError::NotFound)
    ));
}

#[test]
fn active_resume_is_exempt_from_pressure_until_consumed() {
    let clock = ManualClock::new(50_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1),
        Box::new(EmptyStore),
        Box::new(clock.clone()),
    )
    .expect("cache");
    let active = snapshot(3, &[1.0]);
    let active_bytes = active.nbytes() as u64;
    cache.insert_resume(
        &[1, 2, 3],
        active,
        SnapshotRoute::Baseline,
        resume_metadata("chatcmpl-pressure", "fingerprint"),
    );
    cache.insert(&[9, 9], vec![snapshot(2, &[2.0])], SnapshotRoute::Baseline);
    assert_eq!(cache.memory_bytes(), active_bytes);
    assert!(
        cache
            .take_resume("chatcmpl-pressure", "fingerprint", SnapshotRoute::Baseline)
            .is_ok()
    );
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn persistent_resumes_stop_at_the_finite_grace_ceiling_but_remain_hot() {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let snapshot_bytes = snapshot(3, &[1.0]).nbytes() as u64;
    let config = CacheConfig {
        memory_bytes: 1_000_000,
        directory: None,
        filesystem_bytes: snapshot_bytes,
    };
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        config,
        Box::new(RecordingStore(Arc::clone(&state))),
    )
    .expect("cache");

    for (tokens, value, response_id) in [
        ([1, 10, 3], 1.0, "resume-one"),
        ([2, 20, 3], 2.0, "resume-two"),
        ([4, 30, 3], 3.0, "resume-three"),
        ([5, 40, 3], 4.0, "resume-four"),
    ] {
        cache.insert_resume(
            &tokens,
            snapshot(3, &[value]),
            SnapshotRoute::Baseline,
            resume_metadata(response_id, "fingerprint"),
        );
        cache.flush_persistence();
    }

    let hard_cap = filesystem_hard_cap(cache.filesystem_cap.unwrap());
    assert_eq!(hard_cap, snapshot_bytes * 2);
    assert!(cache.filesystem_bytes <= hard_cap);
    assert_eq!(
        state.lock().expect("recording store lock").entries.len(),
        3,
        "the fourth pinned entry is beyond grace and must not reach storage"
    );
    for response_id in ["resume-one", "resume-two", "resume-three", "resume-four"] {
        assert!(
            cache
                .take_resume(response_id, "fingerprint", SnapshotRoute::Baseline)
                .is_ok(),
            "capacity pressure must not break the hot resume {response_id}"
        );
    }
}

#[test]
fn filesystem_grace_ceiling_saturates_without_overflow() {
    assert_eq!(filesystem_hard_cap(5), 10);
    assert_eq!(filesystem_hard_cap(u64::MAX), u64::MAX);
}
#[test]
fn resume_is_hot_before_persistent_write_completes() {
    let blocking = Arc::new((Mutex::new(BlockingState::default()), Condvar::new()));
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        memory_config(1_000_000),
        Box::new(BlockingStore(Arc::clone(&blocking))),
    )
    .expect("cache");
    cache.insert_resume(
        &[1, 2, 3],
        snapshot(3, &[1.0]),
        SnapshotRoute::Baseline,
        resume_metadata("chatcmpl-hot", "fingerprint"),
    );

    let (state_lock, wake) = &*blocking;
    let state = state_lock.lock().expect("blocking store lock");
    let (state, timeout) = wake
        .wait_timeout_while(state, std::time::Duration::from_secs(5), |state| {
            !state.entered
        })
        .expect("blocking store wait");
    let entered = state.entered;
    drop(state);
    let resumed = cache.take_resume("chatcmpl-hot", "fingerprint", SnapshotRoute::Baseline);
    let mut state = state_lock.lock().expect("blocking store lock");
    state.released = true;
    wake.notify_all();
    drop(state);
    cache.flush_persistence();

    assert!(
        !timeout.timed_out() && entered,
        "persistence write did not start"
    );
    assert!(resumed.is_ok(), "hot resume must not wait for persistence");
}

#[test]
fn resume_record_survives_persistent_restart_and_is_removed_on_take() {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    {
        let mut cache = AdaptivePrefixCache::with_store(
            namespaces(),
            memory_config(1_000_000),
            Box::new(RecordingStore(Arc::clone(&state))),
        )
        .expect("cache");
        cache.insert_resume(
            &[7, 8],
            snapshot(2, &[3.0]),
            SnapshotRoute::Baseline,
            resume_metadata("resp_original", "fingerprint"),
        );
        cache.flush_persistence();
    }
    {
        let mut restarted = AdaptivePrefixCache::with_store(
            namespaces(),
            memory_config(1_000_000),
            Box::new(RecordingStore(Arc::clone(&state))),
        )
        .expect("restarted cache");
        let resumed = restarted
            .take_resume("resp_original", "fingerprint", SnapshotRoute::Baseline)
            .expect("persistent resume");
        assert_eq!(resumed.metadata.raw_text, "partial");
        assert_eq!(resumed.metadata.emitted_reasoning_text, "");
        assert_eq!(resumed.metadata.emitted_content_text, "partial");
        restarted.flush_persistence();
    }
    assert!(
        state
            .lock()
            .expect("recording store lock")
            .entries
            .is_empty(),
        "taking a resume checkpoint removes persistent one-shot state"
    );
}

#[test]
fn cache_block_churn_emits_no_info_events() {
    let info_events = Arc::new(AtomicUsize::new(0));
    let subscriber = InfoCounter(Arc::clone(&info_events), None);

    tracing::subscriber::with_default(subscriber, || {
        let mut cache =
            AdaptivePrefixCache::new(namespaces(), memory_config(1)).expect("short cache");
        let before_short = info_events.load(Ordering::Relaxed);
        cache.insert(&[1], vec![snapshot(1, &[1.0])], SnapshotRoute::Baseline);
        let short_events = info_events.load(Ordering::Relaxed) - before_short;

        let mut cache =
            AdaptivePrefixCache::new(namespaces(), memory_config(1)).expect("long cache");
        let before_long = info_events.load(Ordering::Relaxed);
        cache.insert(
            &[1, 2, 3, 4],
            vec![
                snapshot(1, &[1.0]),
                snapshot(2, &[2.0]),
                snapshot(3, &[3.0]),
                snapshot(4, &[4.0]),
            ],
            SnapshotRoute::Baseline,
        );
        let long_events = info_events.load(Ordering::Relaxed) - before_long;

        assert_eq!((short_events, long_events), (0, 0));
    });
}

#[test]
fn adaptive_memory_accounts_shared_checkpoint_pages_once() {
    let mut cache =
        AdaptivePrefixCache::new(namespaces(), memory_config(1_000_000)).expect("cache");
    let snapshots = paged_snapshot_chain();
    cache.insert(
        &(0..768).map(|i| i as i32).collect::<Vec<_>>(),
        snapshots,
        SnapshotRoute::Baseline,
    );

    assert_eq!(
        cache.memory_bytes(),
        3 * 256 * 2 * std::mem::size_of::<f32>() as u64,
        "linear checkpoints charge unique page bytes, not cumulative logical sizes"
    );
}

fn lookahead_fixture() -> (AdaptivePrefixCache, Arc<Mutex<RecordingState>>, ManualClock) {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let clock = ManualClock::new(1_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1),
        Box::new(RecordingStore(Arc::clone(&state))),
        Box::new(clock.clone()),
    )
    .unwrap();
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[1., 2.])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    (cache, state, clock)
}

#[test]
fn prefetch_ready_is_read_only_and_unrelated_lookup_preserves_it() {
    let (mut cache, state, _) = lookahead_fixture();
    let node = cache
        .trie
        .path(&[1, 2], SnapshotRoute::Baseline)
        .last()
        .unwrap()
        .0;
    let before = cache
        .trie
        .terminal(node, SnapshotRoute::Baseline)
        .unwrap()
        .reuse_count;
    cache.prefetch(&[1, 2, 3], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert_eq!(
        cache
            .trie
            .terminal(node, SnapshotRoute::Baseline)
            .unwrap()
            .reuse_count,
        before
    );
    assert!(cache.lookup(&[9], SnapshotRoute::Baseline).is_none());
    let hit = cache.lookup(&[1, 2, 3], SnapshotRoute::Baseline).unwrap();
    assert_eq!(hit.token_count, 2);
    assert_eq!(hit.snapshot().token_len(), 2);
    assert_eq!(state.lock().unwrap().demand_loads, 0);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn prefetch_and_demand_restore_identical_payloads() {
    let (mut cache, state, _) = lookahead_fixture();
    let demand = cache.lookup(&[1, 2, 3], SnapshotRoute::Baseline).unwrap();
    let expected = codec::encode_portable(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &[1, 2],
        demand.snapshot().to_portable().unwrap(),
        codec::RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 1,
        },
        10_000,
        None,
        |_| Ok(()),
    )
    .unwrap();
    cache.prefetch(&[1, 2, 3], SnapshotRoute::Baseline);
    cache.flush_persistence();
    let hit = cache.lookup(&[1, 2, 3], SnapshotRoute::Baseline).unwrap();
    let actual = codec::encode_portable(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &[1, 2],
        hit.snapshot().to_portable().unwrap(),
        codec::RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 1,
        },
        10_000,
        None,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(actual.manifest, expected.manifest);
    assert_eq!(actual.blobs, expected.blobs);
    assert_eq!(state.lock().unwrap().demand_loads, 1);
}

#[test]
fn cancelled_and_replaced_lookahead_release_and_fall_back() {
    let (mut cache, state, _) = lookahead_fixture();
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert!(cache.staging.used.load(Ordering::Acquire) > 0);
    cache.clear_prefetch();
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    cache.prefetch(&[9], SnapshotRoute::Baseline);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    assert_eq!(state.lock().unwrap().demand_loads, 1);
}

#[test]
fn ready_prefetch_cannot_resurrect_expired_or_replaced_entry() {
    let (mut cache, state, clock) = lookahead_fixture();
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    clock.set(1_000 + INITIAL_TTL_MS);
    assert!(cache.lookup(&[1, 2], SnapshotRoute::Baseline).is_none());
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[7., 8., 9.])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    let hit = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    assert_eq!(hit.snapshot().nbytes(), snapshot(2, &[7., 8., 9.]).nbytes());
    assert_eq!(state.lock().unwrap().demand_loads, 1);
    cache.clear_prefetch();
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn failed_missing_and_corrupt_prefetch_never_poison_current_entry() {
    for mode in 0..5 {
        let (mut cache, state, _) = lookahead_fixture();
        let original = state.lock().unwrap().entries.clone();
        {
            let mut s = state.lock().unwrap();
            match mode {
                0 => s.fail_load = true,
                1 => s.entries.clear(),
                2 => s.entries.values_mut().next().unwrap().1[0].bytes = Arc::from([0u8]),
                _ => {
                    let (bytes, _) = s.entries.values_mut().next().unwrap();
                    let mut manifest: Manifest = serde_json::from_slice(bytes).unwrap();
                    if mode == 3 {
                        manifest.namespace = MTP_NAMESPACE.to_string();
                    } else {
                        manifest.token_ids = vec![8, 9];
                    }
                    *bytes = serde_json::to_vec(&manifest).unwrap();
                }
            }
        }
        cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
        cache.flush_persistence();
        assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
        {
            let mut s = state.lock().unwrap();
            s.fail_load = false;
            s.entries = original;
        }
        assert_eq!(
            cache
                .lookup(&[1, 2], SnapshotRoute::Baseline)
                .unwrap()
                .token_count,
            2
        );
        assert_eq!(state.lock().unwrap().demand_loads, 1);
    }
}

#[test]
fn prefetch_budget_saturation_does_not_block_demand_and_releases() {
    let (mut cache, state, _) = lookahead_fixture();
    let all = cache.staging.reserve(cache.staging.limit).unwrap();
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    assert!(cache.prefetch.is_none());
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    drop(all);
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    assert_eq!(state.lock().unwrap().demand_loads, 1);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn lookahead_revalidates_longest_prefix_and_route() {
    let (mut cache, state, _) = lookahead_fixture();
    cache.prefetch(&[1, 2, 3], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert!(cache.lookup(&[1, 2, 3], SnapshotRoute::Mtp).is_none());
    cache.insert(
        &[1, 2, 3],
        vec![snapshot(3, &[3., 4.])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    assert_eq!(
        cache
            .lookup(&[1, 2, 3, 4], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        3
    );
    assert_eq!(state.lock().unwrap().demand_loads, 1);
    // The shorter candidate remains usable, not consumed by the longer demand.
    assert_eq!(
        cache
            .lookup(&[1, 2, 9], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    assert_eq!(state.lock().unwrap().demand_loads, 1);
}

#[test]
fn owned_hot_match_survives_mutation_without_copying_snapshot() {
    let mut cache = AdaptivePrefixCache::new(namespaces(), memory_config(1_000)).unwrap();
    cache.insert(&[1, 2], vec![snapshot(2, &[1.])], SnapshotRoute::Baseline);
    let first = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    let second = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    assert!(std::ptr::eq(first.snapshot(), second.snapshot()));
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[1., 2.])],
        SnapshotRoute::Baseline,
    );
    assert_eq!(first.snapshot().nbytes(), snapshot(2, &[1.]).nbytes());
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .snapshot()
            .nbytes(),
        snapshot(2, &[1., 2.]).nbytes()
    );
}

#[test]
fn delayed_prefetch_is_nonblocking_and_cancelled_inflight_releases_budget() {
    let (mut cache, state, _) = lookahead_fixture();
    let gate = Arc::new((Mutex::new(BlockingState::default()), Condvar::new()));
    state.lock().expect("store").read_gate = Some(Arc::clone(&gate));
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    let (lock, wake) = &*gate;
    let guard = lock.lock().expect("gate");
    let (guard, timeout) = wake
        .wait_timeout_while(guard, std::time::Duration::from_secs(5), |s| !s.entered)
        .expect("gate");
    let entered = guard.entered && !timeout.timed_out();
    drop(guard);
    // Neither an unrelated miss nor cancellation waits for the blocked worker.
    let missed = cache.lookup(&[9], SnapshotRoute::Baseline).is_none();
    cache.clear_prefetch();
    let reserved = cache.staging.used.load(Ordering::Acquire);
    let mut guard = lock.lock().expect("gate");
    guard.released = true;
    wake.notify_all();
    drop(guard);
    cache.flush_persistence();
    assert!(entered);
    assert!(missed);
    assert!(
        reserved > 0,
        "inflight bytes must stay accounted until IO exits"
    );
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    assert_eq!(state.lock().expect("store").demand_loads, 0);
}

#[test]
fn publication_staging_releases_on_success_failure_and_saturation() {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        memory_config(1_000),
        Box::new(RecordingStore(Arc::clone(&state))),
    )
    .unwrap();
    let all = cache.staging.reserve(cache.staging.limit).unwrap();
    cache.insert(&[1], vec![snapshot(1, &[1.])], SnapshotRoute::Baseline);
    assert_eq!(
        cache
            .lookup(&[1], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        1
    );
    drop(all);
    cache.flush_persistence();
    assert!(state.lock().expect("store").entries.is_empty());
    state.lock().expect("store").fail_put = true;
    cache.insert(&[2], vec![snapshot(1, &[2.])], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
    assert_eq!(
        cache
            .lookup(&[2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        1
    );
    state.lock().expect("store").fail_put = false;
    cache.insert(&[2], vec![snapshot(1, &[3.])], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
    cache.memory_cap = 0;
    cache.evict_memory(None);
    assert_eq!(
        cache
            .lookup(&[2], SnapshotRoute::Baseline)
            .unwrap()
            .snapshot()
            .to_portable()
            .unwrap(),
        snapshot(1, &[3.]).to_portable().unwrap()
    );
}

#[test]
fn bounded_filesystem_prefetch_rejects_oversized_payloads() {
    let directory = TempDirectory::new();
    let mut store = FilesystemSnapshotStore::new(&directory.path).unwrap();
    let encoded = codec::encode_portable(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &[1],
        snapshot(1, &[1., 2.]).to_portable().unwrap(),
        RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 1,
        },
        u64::MAX,
        None,
        |_| Ok(()),
    )
    .unwrap();
    let key = encoded.key.clone();
    let size = encoded.manifest.len() as u64
        + encoded
            .blobs
            .iter()
            .map(|b| b.bytes.len() as u64)
            .sum::<u64>();
    store
        .put(
            StoredEntry {
                key: encoded.key,
                manifest: encoded.manifest,
                blobs: encoded.blobs,
            },
            u64::MAX,
        )
        .unwrap();
    assert!(store.load_bounded(&key, size - 1).is_err());
    let loaded = store.load_bounded(&key, size).unwrap().unwrap();
    assert_eq!(
        codec::decode(NAMESPACE, &loaded.manifest, loaded.blobs)
            .unwrap()
            .snapshot
            .to_portable()
            .unwrap(),
        snapshot(1, &[1., 2.]).to_portable().unwrap()
    );
}

#[test]
fn ready_prefetch_replacement_and_stale_manifest_fall_back_to_current_state() {
    let (mut cache, state, _) = lookahead_fixture();
    let original = state.lock().expect("store").entries.clone();
    {
        let mut state = state.lock().expect("store");
        let (bytes, _) = state.entries.values_mut().next().unwrap();
        let mut manifest: Manifest = serde_json::from_slice(bytes).unwrap();
        manifest.expires_at_unix_ms = 999;
        *bytes = serde_json::to_vec(&manifest).unwrap();
    }
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    state.lock().expect("store").entries = original;
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .token_count,
        2
    );
    assert_eq!(state.lock().expect("store").demand_loads, 1);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    cache.flush_persistence();
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[7., 8., 9.])],
        SnapshotRoute::Baseline,
    );
    cache.flush_persistence();
    assert_eq!(
        cache
            .lookup(&[1, 2], SnapshotRoute::Baseline)
            .unwrap()
            .snapshot()
            .to_portable()
            .unwrap(),
        snapshot(2, &[7., 8., 9.]).to_portable().unwrap()
    );
    cache.clear_prefetch();
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn demand_joins_delayed_prefetch_without_duplicate_store_read() {
    let (mut cache, state, _) = lookahead_fixture();
    let gate = Arc::new((Mutex::new(BlockingState::default()), Condvar::new()));
    state.lock().expect("store").read_gate = Some(Arc::clone(&gate));
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    let (joined, joining) = mpsc::channel();
    cache.prefetch.as_mut().unwrap().demand_join = Some(joined);
    let release = thread::spawn(move || {
        let joined = joining
            .recv_timeout(std::time::Duration::from_secs(5))
            .is_ok();
        let (lock, wake) = &*gate;
        let mut guard = lock.lock().expect("gate");
        guard.released = true;
        wake.notify_all();
        joined
    });
    let hit = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    assert!(release.join().unwrap());
    assert_eq!(
        hit.snapshot().to_portable().unwrap(),
        snapshot(2, &[1., 2.]).to_portable().unwrap()
    );
    assert_eq!(state.lock().expect("store").prefetch_loads, 1);
    assert_eq!(state.lock().expect("store").demand_loads, 0);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn bounded_hot_lookahead_survives_current_publication_eviction() {
    let (mut cache, state, clock) = lookahead_fixture();
    let capacity = snapshot(2, &[1., 2.]).nbytes() as u64;
    cache.memory_cap = capacity;
    let next = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    let expected = next.snapshot().to_portable().unwrap();
    drop(next);
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), capacity);
    clock.set(1_001);
    cache.insert(&[9], vec![snapshot(1, &[9., 8.])], SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert!(
        cache
            .trie
            .terminal(
                cache
                    .trie
                    .path(&[1, 2], SnapshotRoute::Baseline)
                    .last()
                    .unwrap()
                    .0,
                SnapshotRoute::Baseline
            )
            .unwrap()
            .snapshot
            .is_none()
    );
    let hit = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    assert_eq!(hit.snapshot().to_portable().unwrap(), expected);
    assert_eq!(state.lock().expect("store").demand_loads, 1);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn prefetch_after_unflushed_publication_observes_fifo_write_visibility() {
    let state = Arc::new(Mutex::new(RecordingState::default()));
    let mut cache = AdaptivePrefixCache::with_store(
        namespaces(),
        memory_config(1),
        Box::new(RecordingStore(Arc::clone(&state))),
    )
    .unwrap();
    cache.insert(
        &[1, 2],
        vec![snapshot(2, &[3., 4.])],
        SnapshotRoute::Baseline,
    );
    cache.prefetch(&[1, 2], SnapshotRoute::Baseline);
    let hit = cache.lookup(&[1, 2], SnapshotRoute::Baseline).unwrap();
    assert_eq!(
        hit.snapshot().to_portable().unwrap(),
        snapshot(2, &[3., 4.]).to_portable().unwrap()
    );
    cache.flush_persistence();
    assert_eq!(state.lock().expect("store").demand_loads, 0);
    assert_eq!(cache.pending_filesystem_bytes, 0);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}

#[test]
fn multi_megabyte_paged_manifest_uses_exact_combined_prefetch_budget() {
    use qw_runtime::{PortableModelState, PortablePage, PortablePagedTensor};
    let token_len = 29_184;
    let tokens = vec![1; token_len];
    let page_bytes: Arc<[u8]> = vec![0; 32 * 2 * 4].into();
    let portable = PortablePromptSnapshot::Baseline(PortableModelState {
        family: "test".into(),
        token_len,
        tensors: Vec::new(),
        paged_tensors: (0..16)
            .map(|layer| PortablePagedTensor {
                name: format!("layers.{layer}.attention.key"),
                token_axis: 1,
                token_len,
                pages: (0..token_len)
                    .step_by(32)
                    .map(|start| PortablePage {
                        token_start: start,
                        token_end: start + 32,
                        shape: vec![1, 32, 2],
                        dtype: mlxcel_core::dtype::FLOAT32,
                        bytes: Arc::clone(&page_bytes),
                    })
                    .collect(),
            })
            .collect(),
        continuation_logits: Some(qw_runtime::PortableArray {
            name: None,
            shape: vec![1, 2],
            dtype: mlxcel_core::dtype::FLOAT32,
            bytes: vec![0; 8],
        }),
    });
    let encoded = codec::encode_portable(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &tokens,
        portable.clone(),
        RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 1,
        },
        u64::MAX,
        None,
        |_| Ok(()),
    )
    .unwrap();
    let manifest_bytes = encoded.manifest.len() as u64;
    assert!(
        manifest_bytes > 2 * 1024 * 1024,
        "fixture represents long paged-cache manifests"
    );
    let full_bytes = manifest_bytes
        + encoded
            .blobs
            .iter()
            .map(|blob| blob.bytes.len() as u64)
            .sum::<u64>();
    let key = encoded.key.clone();
    let directory = TempDirectory::new();
    let mut store = FilesystemSnapshotStore::new(&directory.path).unwrap();
    store
        .put(
            StoredEntry {
                key: encoded.key,
                manifest: encoded.manifest,
                blobs: encoded.blobs,
            },
            u64::MAX,
        )
        .unwrap();
    assert_eq!(store.manifest_bytes(&key).unwrap(), Some(manifest_bytes));
    assert!(store.load_bounded(&key, manifest_bytes - 1).is_err());
    assert!(store.load_bounded(&key, full_bytes - 1).is_err());
    assert!(store.load_bounded(&key, full_bytes).unwrap().is_some());
    let mut cache =
        AdaptivePrefixCache::with_store(namespaces(), memory_config(1), Box::new(store)).unwrap();
    cache.prefetch(&tokens, SnapshotRoute::Baseline);
    cache.flush_persistence();
    assert_eq!(
        cache.staging.used.load(Ordering::Acquire),
        3 * full_bytes,
        "ready large-manifest prefetch retains its exact combined reservation"
    );
    let hit = cache.lookup(&tokens, SnapshotRoute::Baseline).unwrap();
    assert_eq!(hit.token_count, token_len);
    assert_eq!(hit.snapshot().to_portable().unwrap(), portable);
    assert_eq!(cache.staging.used.load(Ordering::Acquire), 0);
}
