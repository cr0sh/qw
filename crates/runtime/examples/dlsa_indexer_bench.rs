// SPDX-License-Identifier: Apache-2.0
//! Unoptimized D-LSA indexer core baseline (`docs/dlsa.md` §11.7): one Qwen3.8-27B full-attention
//! layer (Hq 24, Hkv 4, D 256, rot 64, served defaults), synthetic topical bf16 q/k.
//!
//! Measures, at each context T: chunked prefill (projection, block closes with Ward, selection of
//! every row), decode steps at context T for 1/3/8 rows (append + select, rolled back after each
//! step as a verify would be), and the cache's memory. Run under the GPU lock:
//!
//! ```text
//! ./gpu-lock -- cargo run --release -p qw-runtime --features dlsa --example dlsa_indexer_bench -- \
//!   --contexts 32768,131072 --out "$AGENT_SCRATCH/dlsa_indexer_bench_<sha>_<ts>.json"
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use qw_runtime::{
    DlsaBucketCache, DlsaGeometry, DlsaIndexConfig, DlsaLayout, DlsaMapDtype, DlsaMapKind,
    DlsaMaps, index_keys, index_queries, resolve_dlsa_map_path, sha256_hex, to_host_f32,
};
use serde_json::{Value, json};

const GEOM: DlsaGeometry = DlsaGeometry::QWEN38_27B;
const CFG: DlsaIndexConfig = DlsaIndexConfig::QWEN_DEFAULTS;
const LAYER: usize = 3;
const FULL_ATTENTION_LAYERS: usize = 16;
const TOPICS: usize = 48;
/// Bytes per bucket row as upstream serves it (bf16 centroid + i32 members + i32 size).
const SERVED_ROW_BYTES: f64 = 196.0;

struct Args {
    contexts: Vec<usize>,
    chunk: usize,
    decode_steps: usize,
    map: Option<PathBuf>,
    map_dtype: DlsaMapDtype,
    out: Option<PathBuf>,
    seed: u64,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        contexts: vec![32768, 131072],
        chunk: 1536,
        decode_steps: 20,
        map: None,
        map_dtype: DlsaMapDtype::Bf16,
        out: None,
        seed: 1,
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
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(
        args.chunk > 0 && args.decode_steps > 0,
        "chunk and decode steps must be positive"
    );
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
    let (mut generation, mut project, mut append, mut score, mut pick) =
        (zero, zero, zero, zero, zero);
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
        let ik_host = to_host_f32(&ik);
        mlxcel_core::eval(&iq);
        project += s.elapsed();
        let s = Instant::now();
        cache.append(&ik_host)?;
        append += s.elapsed();
        peak_tail = peak_tail.max(cache.memory().tail_keys);
        peak_tail_allocated = peak_tail_allocated.max(cache.memory().tail_keys_allocated);
        all_keys.extend_from_slice(&ik_host);
        let rows: Vec<usize> = (at..at + step).collect();
        let s = Instant::now();
        let (scores, stride) = cache.score_mlx(&iq, &rows)?;
        score += s.elapsed();
        let s = Instant::now();
        let sel = cache.select_from_scores(&scores, stride, &rows)?;
        pick += s.elapsed();
        checksum += sel.lengths.iter().map(|&l| l as i64).sum::<i64>();
        cache.commit();
        at += step;
    }
    let prefill_wall = wall.elapsed();
    let indexer = project + append + score + pick;
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
            let (scores, stride) = cache.score_mlx(&iq, &rows)?;
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
        });
        eprintln!("T={t}: decode rows={r}: {entry}");
        decode.insert(format!("rows_{r}"), entry);
    }

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
            "append_s": append.as_secs_f64(),
            "score_s": score.as_secs_f64(),
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
    }))
}

fn main() -> Result<()> {
    let args = parse_args()?;
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
