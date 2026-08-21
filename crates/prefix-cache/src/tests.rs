use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use mlxcel_core::generate::ModelStateSnapshot;

use super::*;

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
    entries: HashMap<EntryKey, (Vec<u8>, Vec<Vec<u8>>)>,
    refreshes: Vec<(EntryKey, u64)>,
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
        self.0
            .lock()
            .expect("recording store lock")
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
        vec![256, 400]
    );
    assert_eq!(
        cache.checkpoint_lengths(&second, &[400], SnapshotRoute::Baseline),
        vec![256, 400],
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
fn divergent_long_prefix_reuses_latest_regular_checkpoint() {
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
    assert_eq!(checkpoint_lengths, vec![256, 512, 768, 1_024]);
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
    let hit = cache
        .lookup(&divergent_prompt, SnapshotRoute::Baseline)
        .expect("reusable checkpoint before divergence");
    assert_eq!(hit.token_count, 768);
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
        state
            .lock()
            .expect("recording store lock")
            .entries
            .is_empty(),
        "write-through is followed by persistent eviction under the byte cap",
    );
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
    }
    std::fs::write(directory.path.join(&key.0).join("00000.blob"), b"corrupt")
        .expect("corrupt blob");
    {
        let mut restarted =
            AdaptivePrefixCache::new(namespaces(), config).expect("restart corrupt");
        assert!(restarted.lookup(&tokens, SnapshotRoute::Baseline).is_none());
        restarted.flush_persistence();
    }
    assert!(
        !directory.path.join(&key.0).exists(),
        "corrupt entry is deletion-as-miss"
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
}

#[test]
fn mtp_manifest_round_trip_preserves_route_and_offset_validation() {
    let array = |bytes: Vec<u8>| qw_runtime::PortableArray {
        name: None,
        shape: vec![1, 1, 2],
        dtype: mlxcel_core::dtype::FLOAT32,
        bytes,
    };
    let portable = qw_runtime::PortablePromptSnapshot::Mtp {
        target: qw_runtime::PortableModelState {
            family: "test".to_string(),
            token_len: 1,
            tensors: Vec::new(),
            continuation_logits: None,
        },
        draft_keys: None,
        draft_values: None,
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
    let decoded =
        codec::decode(MTP_NAMESPACE, &encoded.manifest, encoded.blobs).expect("decode MTP");
    assert!(matches!(decoded.snapshot, PromptSnapshot::Mtp(_)));
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
    let partial = directory.path.join(NAMESPACE).join(".tmp-interrupted");
    std::fs::create_dir(&partial).expect("partial directory");
    std::fs::write(partial.join("00000.blob"), b"partial").expect("partial blob");

    let other_namespace = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let mut isolated = AdaptivePrefixCache::new(
        CacheNamespaces {
            baseline: other_namespace.to_string(),
            mtp: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
        },
        config,
    )
    .expect("isolated cache");
    assert!(isolated.lookup(&[9], SnapshotRoute::Baseline).is_none());
    assert!(directory.path.join(NAMESPACE).exists());

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
        cache.insert(
            &[1],
            vec![snapshot(1, &[1.0])],
            SnapshotRoute::Baseline,
        );
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
