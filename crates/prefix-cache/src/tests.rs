use super::*;
use crate::codec::ContentBlob;
use mlxcel_core::generate::ModelStateSnapshot;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct InfoCounter(Arc<AtomicUsize>);

impl tracing::Subscriber for InfoCounter {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::INFO
    }

    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::INFO)
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
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}

    fn clone_span(&self, span: &tracing::span::Id) -> tracing::span::Id {
        span.clone()
    }
}

const NAMESPACE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MTP_NAMESPACE: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

fn namespaces() -> CacheNamespaces {
    CacheNamespaces {
        baseline: NAMESPACE.to_string(),
        mtp: MTP_NAMESPACE.to_string(),
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
        let state = self.0.lock().expect("recording store lock");
        Ok(state.entries.get(key).map(|(manifest, blobs)| StoredEntry {
            key: key.clone(),
            manifest: manifest.clone(),
            blobs: blobs.clone(),
        }))
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
fn radix_divergence_promotes_second_observation_and_selects_longest_snapshot() {
    let clock = ManualClock::new(1_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1_000_000),
        Box::new(EmptyStore),
        Box::new(clock),
    )
    .expect("cache");
    let first = (0..400).collect::<Vec<i32>>();
    let mut second = first[..300].to_vec();
    second.extend(1_000..1_100);
    let mut third = first[..300].to_vec();
    third.extend(2_000..2_100);
    assert_eq!(
        cache.checkpoint_lengths(&first, &[400], SnapshotRoute::Baseline),
        vec![400]
    );
    assert_eq!(
        cache.checkpoint_lengths(&second, &[400], SnapshotRoute::Baseline),
        vec![300, 400]
    );
    cache.insert(
        &second,
        vec![snapshot(256, &[1.0, 2.0])],
        SnapshotRoute::Baseline,
    );
    let hit = cache
        .lookup(&third, SnapshotRoute::Baseline)
        .expect("promoted shared prefix");
    assert_eq!(hit.token_count, 256);
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
    let prompt = (0..1_024).collect::<Vec<i32>>();
    let checkpoint_lengths =
        cache.checkpoint_lengths(&prompt, &[prompt.len()], SnapshotRoute::Baseline);
    assert_eq!(checkpoint_lengths, vec![1_024]);
    cache.insert(
        &prompt,
        checkpoint_lengths
            .into_iter()
            .map(|token_len| snapshot(token_len, &[1.0, 2.0]))
            .collect(),
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
fn first_observation_of_long_prompt_is_sparse() {
    let clock = ManualClock::new(1_000);
    let mut cache = AdaptivePrefixCache::with_store_and_clock(
        namespaces(),
        memory_config(1_000_000),
        Box::new(EmptyStore),
        Box::new(clock),
    )
    .expect("cache");
    let prompt = (0..34_000).collect::<Vec<i32>>();
    assert_eq!(
        cache.checkpoint_lengths(&prompt, &[17_000], SnapshotRoute::Baseline),
        vec![17_000, 34_000]
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
    assert!(
        cache.memory_pages.is_empty(),
        "evicted terminal has no page accounting"
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
        cache.checkpoint_lengths(&tokens, &[tokens.len()], SnapshotRoute::Baseline);
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
#[test]
fn dflash2_portable_snapshot_is_rejected() {
    let array = qw_runtime::PortableArray {
        name: None,
        shape: vec![1],
        dtype: mlxcel_core::dtype::FLOAT32,
        bytes: vec![0; std::mem::size_of::<f32>()],
    };
    let snapshot = PortablePromptSnapshot::Dflash2 {
        target: qw_runtime::PortableModelState {
            family: "test".to_string(),
            token_len: 1,
            tensors: Vec::new(),
            paged_tensors: Vec::new(),
            continuation_logits: None,
        },
        hidden_concat: array.clone(),
        hidden_offset: 0,
        continuation_logits: array,
    };
    let error = codec::encode_portable(
        NAMESPACE,
        SnapshotRoute::Baseline,
        &[1],
        snapshot,
        RetentionMetadata {
            observations: 1,
            reuse_count: 0,
            last_access_unix_ms: 0,
        },
        INITIAL_TTL_MS,
        None,
    )
    .expect_err("DFlash2 snapshots must not enter the prefix cache");
    assert_eq!(
        error,
        "portable snapshot route and token length must match the cache entry"
    );
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
    let subscriber = InfoCounter(Arc::clone(&info_events));

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
