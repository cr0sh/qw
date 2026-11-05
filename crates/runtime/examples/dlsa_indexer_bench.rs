// SPDX-License-Identifier: Apache-2.0
//! Unoptimized D-LSA indexer core baseline (`docs/dlsa.md` §11.7, §14.3): one Qwen3.8-27B
//! full-attention layer (Hq 24, Hkv 4, D 256, rot 64, served defaults), synthetic topical bf16 q/k.
//!
//! Measures, at each context T: chunked prefill (projection, block closes with Ward, selection of
//! every row), decode steps at context T for 1/3/8 rows (append + select, rolled back after each
//! step as a verify would be), and the cache's memory. Then the attention read at Qwen geometry
//! (synthetic fp16 rotated q `[1, 24, L, 256]`, K/V `[1, 4, T, 256]`): one prefill chunk ending at
//! depths T/4, T/2, 3T/4 and T, and one decode row at T, dense vs each mode's sparse read on the
//! real lists of the cache. Every timed stage ends in an `eval`; medians over `--reps`
//! (prefill) and `--attn-decode-steps` (decode). `--modes` picks the reads: `dense` (MLX SDPA
//! over all keys), `mlx` (task 3: select with readback and host pick, masked SDPA for prefill,
//! take + SDPA for decode) and `metal` (K1 device select, K2 sparse read). Run
//! under the GPU lock, after building outside it:
//!
//! ```text
//! cargo build --release -p qw-runtime --features dlsa --example dlsa_indexer_bench
//! ./gpu-lock -- target/release/examples/dlsa_indexer_bench --contexts 32768,131072,262144 \
//!   --out "$AGENT_SCRATCH/dlsa_bench_<sha>_<ts>.json"
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use qw_runtime::profile::{self, DlsaProfile, DlsaStage};
use qw_runtime::{
    DlsaBucketCache, DlsaGeometry, DlsaIndexConfig, DlsaLayout, DlsaMapDtype, DlsaMapKind,
    DlsaMaps, DlsaSelection, gather_attention, index_keys, index_queries, masked_attention,
    resolve_dlsa_map_path, sha256_hex, sparse_read, to_host_f32,
};
use serde_json::{Value, json};

const GEOM: DlsaGeometry = DlsaGeometry::QWEN38_27B;
const CFG: DlsaIndexConfig = DlsaIndexConfig::QWEN_DEFAULTS;
const LAYER: usize = 3;
const FULL_ATTENTION_LAYERS: usize = 16;
const TOPICS: usize = 48;
/// Bytes per bucket row as upstream serves it (bf16 centroid + i32 members + i32 size).
const SERVED_ROW_BYTES: f64 = 196.0;

/// Rows one task-3 prefill mask covers (`DlsaConfig::prefill_mask_rows`).
const MASK_ROWS: usize = 256;
const MODES: [&str; 3] = ["dense", "mlx", "metal"];

struct Args {
    contexts: Vec<usize>,
    chunk: usize,
    decode_steps: usize,
    map: Option<PathBuf>,
    map_dtype: DlsaMapDtype,
    out: Option<PathBuf>,
    seed: u64,
    modes: Vec<String>,
    reps: usize,
    attn_decode_steps: usize,
    attention: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        contexts: vec![32768, 131072, 262144],
        chunk: 1536,
        decode_steps: 20,
        map: None,
        map_dtype: DlsaMapDtype::Bf16,
        out: None,
        seed: 1,
        modes: vec!["dense".into(), "mlx".into()],
        reps: 5,
        attn_decode_steps: 25,
        attention: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--contexts" => {
                args.contexts = val()?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?
            }
            "--chunk" => args.chunk = val()?.parse()?,
            "--decode-steps" => args.decode_steps = val()?.parse()?,
            "--map" => args.map = Some(PathBuf::from(val()?)),
            "--map-dtype" => {
                args.map_dtype = match val()?.as_str() {
                    "bf16" => DlsaMapDtype::Bf16,
                    "f32" => DlsaMapDtype::F32,
                    other => bail!("--map-dtype is bf16 or f32, got {other}"),
                }
            }
            "--out" => args.out = Some(PathBuf::from(val()?)),
            "--seed" => args.seed = val()?.parse()?,
            "--modes" => args.modes = val()?.split(',').map(|s| s.trim().to_string()).collect(),
            "--reps" => args.reps = val()?.parse()?,
            "--attn-decode-steps" => args.attn_decode_steps = val()?.parse()?,
            "--no-attention" => args.attention = false,
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(
        args.chunk > 0 && args.decode_steps > 0 && args.reps > 0 && args.attn_decode_steps > 0,
        "chunk, decode steps and repetitions must be positive"
    );
    for mode in &args.modes {
        ensure!(
            MODES.contains(&mode.as_str()),
            "--modes takes {MODES:?}, got {mode}"
        );
    }
    Ok(args)
}

/// splitmix64 with a cheap approximate normal (Irwin-Hall of 4 uniforms).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn normal(&mut self) -> f32 {
        let x = self.next();
        let s: u32 = (0..4).map(|i| ((x >> (16 * i)) & 0xffff) as u32).sum();
        (s as f32 / 65536.0 - 2.0) * 3f32.sqrt()
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Topical synthetic heads (as `gen_fixtures.py` `synthetic_heads`): keys in runs of a topic,
/// each query drawn towards the topic of one earlier key outside its window.
struct Synth {
    centers: Vec<f32>, // [TOPICS, Hkv, D]
    labels: Vec<u16>,
    rng: Rng,
}

impl Synth {
    fn new(seed: u64, tokens: usize) -> Self {
        let mut rng = Rng(seed);
        let centers = (0..TOPICS * GEOM.kv_heads * GEOM.head_dim)
            .map(|_| rng.normal())
            .collect();
        let mut labels = Vec::with_capacity(tokens);
        while labels.len() < tokens {
            let topic = rng.below(TOPICS) as u16;
            let run = 1 + rng.below(40);
            labels.extend(std::iter::repeat_n(topic, run));
        }
        labels.truncate(tokens);
        Self {
            centers,
            labels,
            rng,
        }
    }

    /// bf16 q `[L, Hq, D]` and k `[L, Hkv, D]` for positions `start..start + L`.
    fn heads(&mut self, start: usize, len: usize) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let (hq, hkv, dh, g) = (GEOM.query_heads, GEOM.kv_heads, GEOM.head_dim, GEOM.group());
        let mut k = Vec::with_capacity(len * hkv * dh * 2);
        let mut q = Vec::with_capacity(len * hq * dh * 2);
        for p in start..start + len {
            let c = self.labels[p] as usize * hkv * dh;
            for i in 0..hkv * dh {
                let v = 0.7 * self.centers[c + i] + 0.7 * self.rng.normal();
                k.extend_from_slice(&bf16_bits(v).to_le_bytes());
            }
            let source = if p > CFG.window_size {
                self.rng.below(p - CFG.window_size)
            } else {
                p
            };
            let t = self.labels[source] as usize * hkv * dh;
            for h in 0..hq {
                let kv = h / g;
                for i in 0..dh {
                    let v = self.centers[t + kv * dh + i] + 0.8 * self.rng.normal();
                    q.extend_from_slice(&bf16_bits(v).to_le_bytes());
                }
            }
        }
        let l = len as i32;
        (
            mlxcel_core::from_bytes_f16(&q, &[l, hq as i32, dh as i32], true),
            mlxcel_core::from_bytes_f16(&k, &[l, hkv as i32, dh as i32], true),
        )
    }
}

/// Layer-3 maps: the converted distilled map if present, else the real layer-3 map stored in
/// the `qwen_layer` fixture, else a seeded random map. Returns (w_q, w_k, description).
fn load_maps(args: &Args) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>, Value)> {
    let cast = |a: UniquePtr<MlxArray>| match args.map_dtype {
        DlsaMapDtype::F32 => a,
        DlsaMapDtype::Bf16 => mlxcel_core::astype(&a, dtype::BFLOAT16),
    };
    if let Ok(path) = resolve_dlsa_map_path(DlsaMapKind::Distilled, args.map.as_deref()) {
        let layout = DlsaLayout {
            full_attention_layers: (0..64).filter(|i| (i + 1) % 4 == 0).collect(),
            input_dim: GEOM.input_dim(),
        };
        let mut maps = DlsaMaps::load(&path, &layout, args.map_dtype)?;
        let layer = maps.layers.remove(&LAYER).context("layer 3")?;
        let sha = sha256_hex(&std::fs::read(&path)?);
        return Ok((
            layer.w_q_idx,
            layer.w_k_idx,
            json!({"source": "converted distilled map", "path": path, "sha256": sha, "layer": LAYER}),
        ));
    }
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/dlsa/qwen_layer.safetensors"
    );
    if let Ok(bytes) = std::fs::read(fixture) {
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let get = |n: &str| -> Result<UniquePtr<MlxArray>> {
            let t = st.tensor(n)?;
            ensure!(t.dtype() == safetensors::Dtype::F32);
            Ok(mlxcel_core::from_bytes(
                t.data(),
                &[GEOM.input_dim() as i32, CFG.index_dim as i32],
                dtype::FLOAT32,
            ))
        };
        return Ok((
            cast(get("real.query_map")?),
            cast(get("real.key_map")?),
            json!({"source": "real layer-3 distilled map from tests/fixtures/dlsa/qwen_layer.safetensors (real.*)",
                   "fixture_sha256": sha256_hex(&bytes), "layer": LAYER}),
        ));
    }
    let mut rng = Rng(args.seed ^ 0xabcdef);
    let scale = 1.0 / (GEOM.input_dim() as f32).sqrt();
    let mut random = || -> Vec<f32> {
        (0..GEOM.input_dim() * CFG.index_dim)
            .map(|_| rng.normal() * scale)
            .collect()
    };
    let shape = [GEOM.input_dim() as i32, CFG.index_dim as i32];
    let (q, k) = (random(), random());
    Ok((
        cast(mlxcel_core::from_slice_f32(&q, &shape)),
        cast(mlxcel_core::from_slice_f32(&k, &shape)),
        json!({"source": "seeded random [768, 64]", "seed": args.seed}),
    ))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Score-stage split (device centroid sync, the f32 matmul, the scores readback) from the
/// profile laps inside `score_mlx`, in seconds.
fn score_split(p: &DlsaProfile) -> Value {
    let s = |stage| p.stage(stage).time.as_secs_f64();
    json!({
        "centroid_sync_s": s(DlsaStage::CentroidSync),
        "score_matmul_s": s(DlsaStage::ScoreMatmul),
        "scores_readback_s": s(DlsaStage::ScoresReadback),
    })
}

fn run_context(args: &Args, t: usize, wq: &MlxArray, wk: &MlxArray) -> Result<Value> {
    ensure!(
        t.is_multiple_of(CFG.window_size),
        "contexts must be multiples of {}",
        CFG.window_size
    );
    let decode_rows = [1usize, 3, 8];
    let mut synth = Synth::new(args.seed.wrapping_add(t as u64), t + 8 * args.decode_steps);
    let mut cache = DlsaBucketCache::new(CFG)?;
    let zero = Duration::ZERO;
    let (mut generation, mut project, mut readback, mut append, mut score, mut pick) =
        (zero, zero, zero, zero, zero, zero);
    profile::reset();
    let mut peak_tail = 0usize;
    let mut peak_tail_allocated = 0usize;
    let mut all_keys: Vec<f32> = Vec::with_capacity(t * CFG.index_dim);
    let mut checksum = 0i64;
    let wall = Instant::now();
    let mut at = 0;
    while at < t {
        let step = args.chunk.min(t - at);
        let s = Instant::now();
        let (q, k) = synth.heads(at, step);
        mlxcel_core::eval(&q);
        mlxcel_core::eval(&k);
        generation += s.elapsed();
        let s = Instant::now();
        let ik = index_keys(&k, wk, &GEOM)?;
        let iq = index_queries(&q, wq, &GEOM)?;
        mlxcel_core::eval(&ik);
        mlxcel_core::eval(&iq);
        project += s.elapsed();
        let s = Instant::now();
        let ik_host = to_host_f32(&ik);
        readback += s.elapsed();
        let s = Instant::now();
        cache.append(&ik_host)?;
        append += s.elapsed();
        peak_tail = peak_tail.max(cache.memory().tail_keys);
        peak_tail_allocated = peak_tail_allocated.max(cache.memory().tail_keys_allocated);
        all_keys.extend_from_slice(&ik_host);
        let rows: Vec<usize> = (at..at + step).collect();
        let s = Instant::now();
        profile::begin(at, &[]);
        let (scores, stride) = cache.score_mlx(&iq, &rows)?;
        profile::end();
        score += s.elapsed();
        let s = Instant::now();
        let sel = cache.select_from_scores(&scores, stride, &rows)?;
        pick += s.elapsed();
        checksum += sel.lengths.iter().map(|&l| l as i64).sum::<i64>();
        cache.commit();
        at += step;
    }
    let prefill_wall = wall.elapsed();
    let indexer = project + readback + append + score + pick;
    let prefill_score_split = score_split(&profile::snapshot());
    let mem = cache.memory();
    // The same keys built in one batch (the snapshot option (a) rebuild), after its commit
    let built = DlsaBucketCache::build(CFG, &all_keys)?;
    ensure!(
        built.centroids() == cache.centroids() && built.members() == cache.members(),
        "batch build differs from the chunked prefill"
    );
    let built_mem = built.memory();
    drop(built);
    drop(all_keys);
    let stats = cache.stats();
    eprintln!(
        "T={t}: prefill indexer {:.1} s ({:.0} tok/s/layer), score {:.1} s, pick+expand {:.1} s, project {:.1} s, append {:.2} s",
        indexer.as_secs_f64(),
        t as f64 / indexer.as_secs_f64(),
        score.as_secs_f64(),
        pick.as_secs_f64(),
        project.as_secs_f64(),
        append.as_secs_f64()
    );

    let mut decode = serde_json::Map::new();
    for &r in &decode_rows {
        let (mut total, mut proj, mut app, mut sc, mut pk) =
            (vec![], vec![], vec![], vec![], vec![]);
        profile::reset();
        for _ in 0..args.decode_steps {
            let (q, k) = synth.heads(t, r);
            mlxcel_core::eval(&q);
            mlxcel_core::eval(&k);
            let s0 = Instant::now();
            let ik = index_keys(&k, wk, &GEOM)?;
            let ik = to_host_f32(&ik);
            let iq = index_queries(&q, wq, &GEOM)?;
            mlxcel_core::eval(&iq);
            let s1 = Instant::now();
            cache.append(&ik)?;
            let s2 = Instant::now();
            let rows: Vec<usize> = (t..t + r).collect();
            profile::begin(t, &[]);
            let (scores, stride) = cache.score_mlx(&iq, &rows)?;
            profile::end();
            let s3 = Instant::now();
            let sel = cache.select_from_scores(&scores, stride, &rows)?;
            let s4 = Instant::now();
            checksum += sel.lengths.iter().map(|&l| l as i64).sum::<i64>();
            // Back to context T, as a rejected verify would roll back
            cache.trim(t)?;
            proj.push(ms(s1 - s0));
            app.push(ms(s2 - s1));
            sc.push(ms(s3 - s2));
            pk.push(ms(s4 - s3));
            total.push(ms(s4 - s0));
        }
        let entry = json!({
            "median_ms": median(total.clone()),
            "min_ms": total.iter().cloned().fold(f64::INFINITY, f64::min),
            "project_median_ms": median(proj),
            "append_median_ms": median(app),
            "score_median_ms": median(sc),
            "pick_expand_median_ms": median(pk),
            "steps": args.decode_steps,
            "score_split_total": score_split(&profile::snapshot()),
        });
        eprintln!("T={t}: decode rows={r}: {entry}");
        decode.insert(format!("rows_{r}"), entry);
    }

    let attention = if args.attention {
        attention_stages(args, t, &mut cache, &mut synth, wq)?
    } else {
        Value::Null
    };

    let per_tok = |b: usize| b as f64 / t as f64;
    Ok(json!({
        "context": t,
        "chunk": args.chunk,
        "prefill": {
            "wall_s": prefill_wall.as_secs_f64(),
            "synthetic_generation_s": generation.as_secs_f64(),
            "indexer_s": indexer.as_secs_f64(),
            "indexer_tokens_per_s_per_layer": t as f64 / indexer.as_secs_f64(),
            "indexer_tokens_per_s_16_layers": t as f64 / indexer.as_secs_f64() / FULL_ATTENTION_LAYERS as f64,
            "project_s": project.as_secs_f64(),
            "key_readback_s": readback.as_secs_f64(),
            "append_s": append.as_secs_f64(),
            "score_s": score.as_secs_f64(),
            "score_split": prefill_score_split,
            "pick_expand_s": pick.as_secs_f64(),
            "selection_rows": t,
            "sampled": false,
        },
        "ward": {
            "blocks": stats.blocks_clustered,
            "mean_rounds": stats.ward_rounds as f64 / stats.blocks_clustered.max(1) as f64,
            "mean_ms_per_block": ms(stats.cluster_time) / stats.blocks_clustered.max(1) as f64,
        },
        "decode": decode,
        "memory": {
            "rows": cache.rows(),
            "host_tables_bytes": mem.host_tables,
            "host_tables_allocated_bytes": mem.host_tables_allocated,
            "device_centroids_bytes": mem.device_centroids,
            "tail_keys_bytes": mem.tail_keys,
            "tail_keys_allocated_bytes": mem.tail_keys_allocated,
            "peak_tail_keys_bytes": peak_tail,
            "peak_tail_keys_allocated_bytes": peak_tail_allocated,
            "allocated_bytes": mem.allocated(),
            "allocated_bytes_per_token_per_layer": per_tok(mem.allocated()),
            "batch_build_host_tables_allocated_bytes": built_mem.host_tables_allocated,
            "batch_build_tail_keys_allocated_bytes": built_mem.tail_keys_allocated,
            "host_tables_bytes_per_token_per_layer": per_tok(mem.host_tables),
            "device_centroids_bytes_per_token_per_layer": per_tok(mem.device_centroids),
            "persistent_bytes_per_token_per_layer": per_tok(mem.host_tables + mem.device_centroids),
            "persistent_bytes_per_token_16_layers": per_tok(mem.host_tables + mem.device_centroids) * FULL_ATTENTION_LAYERS as f64,
            "served_bf16_bytes_per_token_per_layer": cache.rows() as f64 * SERVED_ROW_BYTES / t as f64,
        },
        "checksum_list_lengths": checksum,
        "attention": attention,
    }))
}

fn normal_f16(shape: &[i32], key: &MlxArray) -> UniquePtr<MlxArray> {
    // SAFETY: `key` outlives the call.
    let a = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT16, key as *const MlxArray) };
    mlxcel_core::eval(&a);
    a
}

/// Median ms of `reps` runs of `f` (each evaluated), after one warm-up run.
fn time_median(reps: usize, mut f: impl FnMut() -> UniquePtr<MlxArray>) -> f64 {
    mlxcel_core::eval(&f());
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        let out = f();
        mlxcel_core::eval(&out);
        v.push(ms(s.elapsed()));
    }
    median(v)
}

/// Median ms of `reps` runs of a host-synchronous `f`, after one warm-up run.
fn time_median_host(reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        f()?;
        v.push(ms(s.elapsed()));
    }
    Ok(median(v))
}

/// K1 select (score matmul + device select, no readback) and the K2 read on its lazy lists,
/// checked against the task-3 host lists.
#[allow(clippy::too_many_arguments)]
fn metal_stages(
    reps: usize,
    cache: &mut DlsaBucketCache,
    iq: &MlxArray,
    positions: &[usize],
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    entry: &mut serde_json::Map<String, Value>,
) -> Result<()> {
    let select = time_median_host(reps, || {
        let sel = cache.select_device(iq, positions)?;
        mlxcel_core::eval(&sel.lists);
        mlxcel_core::eval(&sel.lengths);
        Ok(())
    })?;
    entry.insert("metal_select_ms".into(), json!(select));
    let sel = cache.select_device(iq, positions)?;
    let host = sel.to_host()?;
    let oracle = cache.select(iq, positions)?;
    entry.insert(
        "metal_lists_equal_mlx".into(),
        json!(host.lists == oracle.lists && host.lengths == oracle.lengths),
    );
    let read = time_median(reps, || sparse_read(q, k, v, &sel, scale));
    entry.insert("metal_read_ms".into(), json!(read));
    Ok(())
}

/// One fenced run of `f`, its time by profile stage (ms).
fn fenced_split(f: impl FnOnce() -> UniquePtr<MlxArray>) -> Value {
    profile::reset();
    profile::begin(0, &[]);
    let out = f();
    profile::lap(DlsaStage::Read, &[&out]);
    profile::end();
    let p = profile::snapshot();
    let stages: serde_json::Map<String, Value> = p
        .entries()
        .into_iter()
        .map(|(name, t)| (format!("{name}_ms"), json!(ms(t.time))))
        .collect();
    Value::Object(stages)
}

fn mean_len(sel: &DlsaSelection) -> f64 {
    sel.lengths.iter().map(|&l| l as f64).sum::<f64>() / sel.rows.max(1) as f64
}

/// The attention read at Qwen geometry over a cache of depth `t`: prefill chunks ending at
/// T/4, T/2, 3T/4, T and one decode row at T, dense vs the modes' sparse reads.
fn attention_stages(
    args: &Args,
    t: usize,
    cache: &mut DlsaBucketCache,
    synth: &mut Synth,
    wq: &MlxArray,
) -> Result<Value> {
    let (hq, hkv, d) = (
        GEOM.query_heads as i32,
        GEOM.kv_heads as i32,
        GEOM.head_dim as i32,
    );
    let scale = 1.0 / (GEOM.head_dim as f32).sqrt();
    let key = mlxcel_core::random_key(args.seed ^ t as u64);
    let keys = normal_f16(&[1, hkv, t as i32, d], &key);
    let key = mlxcel_core::random_key(args.seed ^ t as u64 ^ 0x5555);
    let values = normal_f16(&[1, hkv, t as i32, d], &key);
    let has = |m: &str| args.modes.iter().any(|x| x == m);

    let mut prefill = Vec::new();
    for quarter in 1..=4 {
        let depth = t * quarter / 4;
        let rows = args.chunk.min(depth);
        let start = depth - rows;
        let key = mlxcel_core::random_key(args.seed ^ depth as u64 ^ 0xaaaa);
        let q = normal_f16(&[1, hq, rows as i32, d], &key);
        let k = mlxcel_core::slice(&keys, &[0, 0, 0, 0], &[1, hkv, depth as i32, d]);
        let v = mlxcel_core::slice(&values, &[0, 0, 0, 0], &[1, hkv, depth as i32, d]);
        mlxcel_core::eval(&k);
        mlxcel_core::eval(&v);
        let mut entry = serde_json::Map::new();
        entry.insert("depth".into(), json!(depth));
        entry.insert("rows".into(), json!(rows));
        if has("dense") {
            let dense = time_median(args.reps, || {
                mlxcel_core::causal_attention(&q, &k, &v, scale, 0.0, 0)
            });
            entry.insert("dense_ms".into(), json!(dense));
        }
        // Real lists of the cache for rows start..depth, from fresh topical queries
        let (qh, _) = synth.heads(start, rows);
        let iq = index_queries(&qh, wq, &GEOM)?;
        mlxcel_core::eval(&iq);
        let positions: Vec<usize> = (start..depth).collect();
        if has("mlx") {
            // Task 3: score matmul, scores readback, host pick + expansion
            let select = time_median_host(args.reps, || {
                cache.select(&iq, &positions)?;
                Ok(())
            })?;
            entry.insert("mlx_select_ms".into(), json!(select));
            let sel = cache.select(&iq, &positions)?;
            let masked = time_median(args.reps, || {
                masked_attention(&q, &k, &v, &sel, start, MASK_ROWS, scale)
            });
            entry.insert("mlx_masked_ms".into(), json!(masked));
            entry.insert(
                "mlx_masked_fenced".into(),
                fenced_split(|| masked_attention(&q, &k, &v, &sel, start, MASK_ROWS, scale)),
            );
            entry.insert("mean_list_len".into(), json!(mean_len(&sel)));
        }
        if has("metal") {
            metal_stages(
                args.reps, cache, &iq, &positions, &q, &k, &v, scale, &mut entry,
            )?;
        }
        eprintln!("T={t}: prefill attention {}", Value::Object(entry.clone()));
        prefill.push(Value::Object(entry));
    }

    let key = mlxcel_core::random_key(args.seed ^ t as u64 ^ 0x1234);
    let q1 = normal_f16(&[1, hq, 1, d], &key);
    let mut decode = serde_json::Map::new();
    decode.insert("depth".into(), json!(t));
    decode.insert("steps".into(), json!(args.attn_decode_steps));
    if has("dense") {
        let dense = time_median(args.attn_decode_steps, || {
            // SAFETY: no mask.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q1,
                    &keys,
                    &values,
                    scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        });
        decode.insert("dense_ms".into(), json!(dense));
    }
    let (qh, _) = synth.heads(t - 1, 1);
    let iq = index_queries(&qh, wq, &GEOM)?;
    mlxcel_core::eval(&iq);
    if has("mlx") {
        let select = time_median_host(args.attn_decode_steps, || {
            cache.select(&iq, &[t - 1])?;
            Ok(())
        })?;
        decode.insert("mlx_select_ms".into(), json!(select));
        let sel = cache.select(&iq, &[t - 1])?;
        let list = sel.list(0).to_vec();
        let gather = time_median(args.attn_decode_steps, || {
            gather_attention(&q1, &keys, &values, &list, scale)
        });
        decode.insert("mlx_gather_ms".into(), json!(gather));
        decode.insert(
            "mlx_gather_fenced".into(),
            fenced_split(|| gather_attention(&q1, &keys, &values, &list, scale)),
        );
        decode.insert("list_len".into(), json!(list.len()));
    }
    if has("metal") {
        metal_stages(
            args.attn_decode_steps,
            cache,
            &iq,
            &[t - 1],
            &q1,
            &keys,
            &values,
            scale,
            &mut decode,
        )?;
    }
    eprintln!("T={t}: decode attention {}", Value::Object(decode.clone()));
    Ok(json!({
        "qkv_dtype": "f16",
        "mask_rows": MASK_ROWS,
        "prefill_chunks": prefill,
        "decode_row": decode,
    }))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if let Some(out) = &args.out {
        ensure!(
            !out.exists(),
            "{} exists; result files must have unique names",
            out.display()
        );
    }
    // The score/read stage splits come from profile laps, which run only between begin/end
    profile::set_override(Some(true));
    let (wq, wk, map) = load_maps(&args)?;
    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .is_some_and(|o| !o.stdout.is_empty());
    let machine = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string", "hw.memsize"])
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .replace('\n', " / ")
        });
    eprintln!("map: {map}");
    let mut results = Vec::new();
    for &t in &args.contexts {
        results.push(run_context(&args, t, &wq, &wk)?);
    }
    let report = json!({
        "bench": "dlsa_indexer_bench",
        "git_commit": git,
        "worktree_dirty": dirty,
        "machine": machine,
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "config": format!("{CFG:?}"),
        "geometry": format!("{GEOM:?}"),
        "map": map,
        "map_dtype": format!("{:?}", args.map_dtype),
        "qk_dtype": "bf16",
        "modes": args.modes,
        "reps": args.reps,
        "results": results,
    });
    let text = serde_json::to_string_pretty(&report)?;
    println!("{text}");
    if let Some(out) = &args.out {
        ensure!(
            !out.exists(),
            "{} exists; result files must have unique names",
            out.display()
        );
        std::fs::write(out, &text)?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}
