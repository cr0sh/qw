// SPDX-License-Identifier: Apache-2.0
//! Indexer core tests: golden fixtures stage by stage and end to end, properties,
//! incremental-vs-batch equivalence, rollback and snapshots. Tolerances: `docs/dlsa.md` §11.6.
//!
//! Tests that touch MLX need Metal: run them under `./gpu-lock -- cargo test ...`.

use mlxcel_core::{MlxArray, UniquePtr};

use super::fixtures::Fixture;
use super::*;

// ------------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------------

/// Cases with read lists, and the tensor prefix of each map.
const LIST_CASES: &[(&str, &str)] = &[
    ("tiny", ""),
    ("tiny_ties", ""),
    ("short_dense", ""),
    ("qwen_layer", "random."),
    ("qwen_layer", "real."),
];

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[track_caller]
fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    let diff = max_abs_diff(a, b);
    if std::env::var_os("DLSA_TEST_VERBOSE").is_some() {
        eprintln!("{what}: max abs diff {diff:e} (tolerance {tol:e})");
    }
    assert!(diff <= tol, "{what}: max abs diff {diff:e} > {tol:e}");
}

fn mlx_f32(data: &[f32], shape: &[usize]) -> UniquePtr<MlxArray> {
    let shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    mlxcel_core::from_slice_f32(data, &shape)
}

/// A fixture's F16 heads tensor as an f16 MLX array (as the model holds q/k).
fn mlx_heads(f: &Fixture, name: &str) -> UniquePtr<MlxArray> {
    let (bytes, dtype) = f.raw(name);
    assert_eq!(dtype, safetensors::Dtype::F16);
    let shape: Vec<i32> = f.shape(name).iter().map(|&s| s as i32).collect();
    mlxcel_core::from_bytes_f16(&bytes, &shape, false)
}

fn rows_slice(x: &MlxArray, start: usize, end: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[0] = start as i32;
    stops[0] = end as i32;
    mlxcel_core::slice(x, &starts, &stops)
}

fn row<T>(data: &[T], width: usize, r: usize) -> &[T] {
    &data[r * width..(r + 1) * width]
}

/// The fixture's list row `p` (valid part).
fn fixture_list(selected: &[i32], lengths: &[i32], width: usize, p: usize) -> Vec<i32> {
    row(selected, width, p)[..lengths[p] as usize].to_vec()
}

/// Row `r` of `sel` padded to its full width must equal the fixture's padded row.
#[track_caller]
fn assert_row_eq(sel: &DlsaSelection, r: usize, want: &[i32], want_len: i32, what: &str) {
    assert_eq!(sel.lengths[r], want_len, "{what}: length");
    assert_eq!(row(&sel.lists, sel.width, r), want, "{what}: list");
}

// ------------------------------------------------------------------------------------------------
// Fixture stages
// ------------------------------------------------------------------------------------------------

#[test]
fn fixture_configs_validate() {
    for name in [
        "tiny",
        "tiny_ties",
        "short_dense",
        "qwen_layer",
        "qwen_index_defaults",
    ] {
        Fixture::load(name).cfg().validate().unwrap();
    }
    assert_eq!(
        Fixture::load("qwen_index_defaults").cfg(),
        DlsaIndexConfig::QWEN_DEFAULTS
    );
    assert_eq!(
        Fixture::load("qwen_layer").geometry(),
        DlsaGeometry::QWEN38_27B
    );
    let q = DlsaIndexConfig::QWEN_DEFAULTS;
    assert_eq!(
        (q.buckets_per_block(), q.pools_per_block(), q.list_width()),
        (64, 128, 4612)
    );
    let bad = |f: fn(&mut DlsaIndexConfig)| {
        let mut c = q;
        f(&mut c);
        c.validate().is_err()
    };
    assert!(bad(|c| c.bucket_cap = 15));
    assert!(bad(|c| c.pool_size = 3));
    assert!(bad(|c| c.bucket_size = 24));
    assert!(bad(|c| c.top_p = 2049));
    assert!(bad(|c| c.top_k = 1024 * 16 + 1));
    assert!(bad(|c| c.window_size = 0));
}

#[test]
fn fixture_concat_and_group_queries() {
    for (name, prefix) in LIST_CASES {
        let f = Fixture::load(name);
        let geom = f.geometry();
        if f.has("concat_keys") {
            let got = to_host_f32(&concat_keys(&mlx_heads(&f, "k"), &geom).unwrap());
            assert_close(
                &got,
                &f.f32("concat_keys"),
                1e-5,
                &format!("{name} concat_keys"),
            );
        }
        let rows: Vec<usize> = f
            .i32("attention_rows")
            .iter()
            .map(|&r| r as usize)
            .collect();
        let grouped = if f.has("q") {
            let q = mlx_heads(&f, "q");
            let all = to_host_f32(&group_queries(&q, &geom).unwrap());
            let w = geom.input_dim();
            rows.iter()
                .flat_map(|&r| row(&all, w, r).to_vec())
                .collect::<Vec<_>>()
        } else {
            to_host_f32(&group_queries(&mlx_heads(&f, "q_rows"), &geom).unwrap())
        };
        assert_close(
            &grouped,
            &f.f32(&format!("{prefix}grouped_queries")),
            1e-5,
            &format!("{name} {prefix}grouped_queries"),
        );
    }
}

#[test]
fn fixture_index_projection_f32_maps() {
    for (name, prefix) in LIST_CASES {
        let f = Fixture::load(name);
        let geom = f.geometry();
        let d = f.cfg().index_dim;
        let qmap = mlx_f32(
            &f.f32(&format!("{prefix}query_map")),
            &[geom.input_dim(), d],
        );
        let kmap = mlx_f32(&f.f32(&format!("{prefix}key_map")), &[geom.input_dim(), d]);
        let keys = to_host_f32(&index_keys(&mlx_heads(&f, "k"), &kmap, &geom).unwrap());
        assert_close(
            &keys,
            &f.f32(&format!("{prefix}index_keys")),
            1e-5,
            &format!("{name} {prefix}index_keys"),
        );
        let want_q = f.f32(&format!("{prefix}index_queries"));
        if f.has("q") {
            let qs = to_host_f32(&index_queries(&mlx_heads(&f, "q"), &qmap, &geom).unwrap());
            assert_close(&qs, &want_q, 1e-5, &format!("{name} index_queries"));
        } else {
            let qs = to_host_f32(&index_queries(&mlx_heads(&f, "q_rows"), &qmap, &geom).unwrap());
            let want: Vec<f32> = f
                .i32("attention_rows")
                .iter()
                .flat_map(|&r| row(&want_q, d, r as usize).to_vec())
                .collect();
            assert_close(&qs, &want, 1e-5, &format!("{name} {prefix}index_queries"));
        }
        // The [1, L, H, D] layout of `forward_impl` gives the same rows
        let k = mlx_heads(&f, "k");
        let shape = mlxcel_core::array_shape(&k);
        let k4 = mlxcel_core::reshape(&k, &[1, shape[0], shape[1], shape[2]]);
        let keys4 = to_host_f32(&index_keys(&k4, &kmap, &geom).unwrap());
        assert_eq!(keys4, keys, "{name}: [1, L, H, D] input");
    }
}

#[test]
fn bf16_maps_give_unit_rows_close_to_f32() {
    let f = Fixture::load("qwen_layer");
    let geom = f.geometry();
    let kmap = mlx_f32(&f.f32("real.key_map"), &[geom.input_dim(), 64]);
    let kmap16 = mlxcel_core::astype(&kmap, mlxcel_core::dtype::BFLOAT16);
    let k = mlx_heads(&f, "k");
    let keys16 = to_host_f32(&index_keys(&k, &kmap16, &geom).unwrap());
    let keys32 = to_host_f32(&index_keys(&k, &kmap, &geom).unwrap());
    for r in keys16.as_chunks::<64>().0 {
        let n: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "bf16-map index key norm {n}");
    }
    // bf16 products: a bf16-level difference, nowhere near a different vector
    assert_close(&keys16, &keys32, 3e-2, "bf16 vs f32 maps");
    // An f16 map is refused (DlsaMapDtype is F32 or Bf16)
    let kmap_f16 = mlxcel_core::astype(&kmap, mlxcel_core::dtype::FLOAT16);
    assert!(index_keys(&k, &kmap_f16, &geom).is_err());
}

#[test]
fn fixture_pool_keys() {
    for (name, prefix) in LIST_CASES {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        let keys = f.f32(&format!("{prefix}index_keys"));
        let blocks = f.key_count() / cfg.window_size;
        let got = pool_keys(
            &keys[..blocks * cfg.window_size * cfg.index_dim],
            cfg.index_dim,
            cfg.pool_size,
        );
        assert_close(
            &got,
            &f.f32(&format!("{prefix}pool_keys")),
            1e-6,
            &format!("{name} {prefix}pool_keys"),
        );
    }
}

#[test]
fn fixture_ward_labels_from_fixture_pools() {
    for (name, prefix) in LIST_CASES {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        let pools = f.f32(&format!("{prefix}pool_keys"));
        let want = f.i32(&format!("{prefix}ward_labels"));
        let per = cfg.pools_per_block();
        let sizes = vec![cfg.pool_size as f32; per];
        for (b, block) in pools.chunks_exact(per * cfg.index_dim).enumerate() {
            let got = ward_labels(
                block,
                &sizes,
                cfg.index_dim,
                cfg.buckets_per_block(),
                cfg.bucket_cap,
            )
            .unwrap();
            let got: Vec<i32> = got.labels.iter().map(|&l| l as i32).collect();
            assert_eq!(got, row(&want, per, b), "{name} {prefix} block {b}");
        }
    }
}

#[test]
fn fixture_ward_tie_policy_follows_the_kernel() {
    let f = Fixture::load("ward_tie_policy");
    let got = ward_labels(&f.f32("pool_keys"), &f.f32("pool_sizes"), 2, 3, 4).unwrap();
    let got: Vec<i32> = got.labels.iter().map(|&l| l as i32).collect();
    assert_eq!(got, f.i32("ward_labels"));
    assert_eq!(got, [0, 1, 1, 2]);
    assert_ne!(got, f.i32("reference_ward_labels"));
}

#[test]
fn fixture_ward_labels_from_own_pools() {
    let mut cases: Vec<(&str, &str)> = LIST_CASES.to_vec();
    cases.push(("qwen_index_defaults", ""));
    for (name, prefix) in cases {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        // qwen_index_defaults stores BF16 index keys; upstream clusters them upcast to f32
        let keys = f.f32(&format!("{prefix}index_keys"));
        let want = f.i32(&format!("{prefix}ward_labels"));
        let per = cfg.pools_per_block();
        let blocks = f.key_count() / cfg.window_size;
        assert_eq!(want.len(), blocks * per);
        let sizes = vec![cfg.pool_size as f32; per];
        for b in 0..blocks {
            let w = cfg.window_size * cfg.index_dim;
            let pools = pool_keys(&keys[b * w..(b + 1) * w], cfg.index_dim, cfg.pool_size);
            let got = ward_labels(
                &pools,
                &sizes,
                cfg.index_dim,
                cfg.buckets_per_block(),
                cfg.bucket_cap,
            )
            .unwrap();
            let got: Vec<i32> = got.labels.iter().map(|&l| l as i32).collect();
            assert_eq!(got, row(&want, per, b), "{name} {prefix} block {b}");
        }
    }
}

/// Every case with buckets: (fixture, prefix, positions of its query rows).
fn bucket_cases() -> Vec<(Fixture, &'static str, Vec<usize>)> {
    let mut out: Vec<_> = LIST_CASES
        .iter()
        .map(|(n, p)| {
            let f = Fixture::load(n);
            let t = f.key_count();
            (f, *p, (0..t).collect())
        })
        .collect();
    let f = Fixture::load("qwen_index_defaults");
    let positions = f.i32("positions").iter().map(|&p| p as usize).collect();
    out.push((f, "", positions));
    out
}

/// Prefill in chunks of `chunk`, committing after each, then one key a step.
fn incremental_cache(
    cfg: DlsaIndexConfig,
    keys: &[f32],
    prefill: usize,
    chunk: usize,
) -> DlsaBucketCache {
    let d = cfg.index_dim;
    let t = keys.len() / d;
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    let mut at = 0;
    while at < t {
        let step = if at < prefill {
            chunk.min(prefill - at)
        } else {
            1
        };
        cache.append(&keys[at * d..(at + step) * d]).unwrap();
        cache.commit();
        at += step;
    }
    cache
}

#[test]
fn fixture_bucket_tables_incremental() {
    for (f, prefix, _) in bucket_cases() {
        let cfg = f.cfg();
        let keys = f.f32(&format!("{prefix}index_keys"));
        let prefill = f.prefill_len().unwrap_or(f.key_count() / 2);
        let cache = incremental_cache(cfg, &keys, prefill, 7);
        let what = format!("{} {prefix}", f.name);
        assert_close(
            cache.centroids(),
            &f.f32(&format!("{prefix}bucket_centroids")),
            1e-6,
            &format!("{what} centroids"),
        );
        assert_eq!(
            cache.members(),
            f.i32(&format!("{prefix}bucket_members")),
            "{what} members"
        );
        assert_eq!(
            cache.sizes(),
            f.i32(&format!("{prefix}bucket_sizes")),
            "{what} sizes"
        );
        let built = DlsaBucketCache::build(cfg, &keys).unwrap();
        assert_eq!(
            built.centroids(),
            cache.centroids(),
            "{what}: batch == incremental"
        );
        assert_eq!(built.members(), cache.members());
    }
}

#[test]
fn fixture_visible_buckets() {
    for (f, prefix, positions) in bucket_cases() {
        let cfg = f.cfg();
        let cache = DlsaBucketCache::build(cfg, &f.f32(&format!("{prefix}index_keys"))).unwrap();
        let got: Vec<i32> = positions
            .iter()
            .map(|&p| cache.visible_buckets(p) as i32)
            .collect();
        assert_eq!(
            got,
            f.i32(&format!("{prefix}visible_buckets")),
            "{}",
            f.name
        );
        let got: Vec<i32> = positions
            .iter()
            .map(|&p| cfg.visible_buckets(p) as i32)
            .collect();
        assert_eq!(
            got,
            f.i32(&format!("{prefix}visible_buckets")),
            "{}",
            f.name
        );
    }
}

/// A cache holding exactly the fixture's final bucket tables (via restore).
fn fixture_tables_cache(f: &Fixture, prefix: &str) -> DlsaBucketCache {
    let cfg = f.cfg();
    let keys = f.f32(&format!("{prefix}index_keys"));
    let t = f.key_count();
    let open = t / cfg.window_size * cfg.window_size;
    let snap = DlsaIndexSnapshot {
        cfg,
        len: t,
        centroids: f.f32(&format!("{prefix}bucket_centroids")),
        members: f.i32(&format!("{prefix}bucket_members")),
        sizes: f.i32(&format!("{prefix}bucket_sizes")),
        open_keys: keys[open * cfg.index_dim..].to_vec(),
    };
    DlsaBucketCache::restore(cfg, &snap).unwrap()
}

#[test]
fn fixture_bucket_scores_mlx_and_host() {
    for (f, prefix, positions) in bucket_cases() {
        let cfg = f.cfg();
        let d = cfg.index_dim;
        let mut cache = fixture_tables_cache(&f, prefix);
        let queries = f.f32(&format!("{prefix}index_queries"));
        let want = f.f32(&format!("{prefix}bucket_scores"));
        let rows = cache.rows();
        let (scores, stride) = cache
            .score_mlx(&mlx_f32(&queries, &[positions.len(), d]), &positions)
            .unwrap();
        let host = score_buckets_host(&queries, cache.centroids(), d, stride);
        for (r, _) in positions.iter().enumerate() {
            let w = &want[r * rows..r * rows + stride];
            assert_close(
                row(&scores, stride, r),
                w,
                1e-6,
                &format!("{} mlx scores", f.name),
            );
            assert_close(
                row(&host, stride, r),
                w,
                1e-6,
                &format!("{} host scores", f.name),
            );
        }
    }
}

#[test]
fn fixture_picked_buckets_from_fixture_scores() {
    for (f, prefix, _) in bucket_cases() {
        let cfg = f.cfg();
        let scores = f.f32(&format!("{prefix}bucket_scores"));
        let visible = f.i32(&format!("{prefix}visible_buckets"));
        let want = f.i32(&format!("{prefix}picked_buckets"));
        let rows = scores.len() / visible.len();
        for (r, &v) in visible.iter().enumerate() {
            let got = top_p_buckets(&row(&scores, rows, r)[..v as usize], cfg.top_p).unwrap();
            let mut padded = vec![-1i32; cfg.top_p];
            for (s, &b) in padded.iter_mut().zip(&got) {
                *s = b as i32;
            }
            assert_eq!(
                padded,
                row(&want, cfg.top_p, r),
                "{} {prefix} row {r}",
                f.name
            );
        }
    }
}

#[test]
fn fixture_expansion_from_fixture_picks() {
    for (f, prefix, positions) in bucket_cases() {
        let cfg = f.cfg();
        let picked = f.i32(&format!("{prefix}picked_buckets"));
        let members = f.i32(&format!("{prefix}bucket_members"));
        let selected = f.i32(&format!("{prefix}selected"));
        let lengths = f.i32(&format!("{prefix}selected_lengths"));
        let width = cfg.list_width();
        for (r, &p) in positions.iter().enumerate() {
            let chosen: Vec<u32> = row(&picked, cfg.top_p, r)
                .iter()
                .filter(|&&b| b >= 0)
                .map(|&b| b as u32)
                .collect();
            let mut out = Vec::new();
            expand_buckets(&chosen, &members, p, &cfg, &mut out);
            assert_eq!(
                out,
                fixture_list(&selected, &lengths, width, r),
                "{} {prefix} row {r} (p {p})",
                f.name
            );
            assert!(
                row(&selected, width, r)[out.len()..]
                    .iter()
                    .all(|&x| x == -1)
            );
        }
    }
}

/// Lists from fixture index keys/queries through the cache, as a forward would run: prefill
/// chunks (append, select the chunk's rows, commit), then one key a step. Both the MLX and
/// the host paths must give the fixture's lists exactly, in order.
#[test]
fn fixture_end_to_end_from_index_vectors() {
    for (f, prefix, positions) in bucket_cases() {
        let cfg = f.cfg();
        let d = cfg.index_dim;
        let keys = f.f32(&format!("{prefix}index_keys"));
        let queries = f.f32(&format!("{prefix}index_queries"));
        let selected = f.i32(&format!("{prefix}selected"));
        let lengths = f.i32(&format!("{prefix}selected_lengths"));
        let picked = f.i32(&format!("{prefix}picked_buckets"));
        let width = cfg.list_width();
        let what = format!("{} {prefix}", f.name);
        let mut cache = DlsaBucketCache::new(cfg).unwrap();
        if f.name == "qwen_index_defaults" {
            // Sparse positions: grow the cache to each row's key, then read that row
            for (r, &p) in positions.iter().enumerate() {
                let at = cache.len();
                cache.append(&keys[at * d..(p + 1) * d]).unwrap();
                cache.commit();
                let q = row(&queries, d, r);
                let mlx = cache.select(&mlx_f32(q, &[1, d]), &[p]).unwrap();
                let host = cache.select_host(q, &[p]).unwrap();
                assert_eq!(mlx, host, "{what} p {p}: mlx == host");
                assert_row_eq(
                    &mlx,
                    0,
                    row(&selected, width, r),
                    lengths[r],
                    &format!("{what} p {p}"),
                );
                assert_eq!(mlx.picked, row(&picked, cfg.top_p, r), "{what} p {p} picks");
            }
            continue;
        }
        let t = f.key_count();
        let prefill = f.prefill_len().unwrap();
        let mut at = 0;
        let chunk = 2 * cfg.window_size + 3;
        while at < t {
            let step = if at < prefill {
                chunk.min(prefill - at)
            } else {
                1
            };
            cache.append(&keys[at * d..(at + step) * d]).unwrap();
            let rows: Vec<usize> = (at..at + step).collect();
            let q = &queries[at * d..(at + step) * d];
            let mlx = cache.select(&mlx_f32(q, &[step, d]), &rows).unwrap();
            let host = cache.select_host(q, &rows).unwrap();
            assert_eq!(mlx, host, "{what} rows {at}..{}: mlx == host", at + step);
            for (i, &p) in rows.iter().enumerate() {
                assert_row_eq(
                    &mlx,
                    i,
                    row(&selected, width, p),
                    lengths[p],
                    &format!("{what} p {p}"),
                );
                assert_eq!(
                    row(&mlx.picked, cfg.top_p, i),
                    row(&picked, cfg.top_p, p),
                    "{what} p {p} picks"
                );
            }
            cache.commit();
            at += step;
        }
    }
}

/// Lists from raw q/k through `project.rs` (MLX, F32 maps), the cache and MLX selection. The
/// smallest top-p gap in the fixtures is 3.3e-6 (`real`), the projection error ~1e-7, so any
/// flip is a bug, not rounding.
#[test]
fn fixture_end_to_end_from_raw_heads() {
    for (name, prefix) in LIST_CASES {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        let geom = f.geometry();
        let d = cfg.index_dim;
        let qmap = mlx_f32(
            &f.f32(&format!("{prefix}query_map")),
            &[geom.input_dim(), d],
        );
        let kmap = mlx_f32(&f.f32(&format!("{prefix}key_map")), &[geom.input_dim(), d]);
        let selected = f.i32(&format!("{prefix}selected"));
        let lengths = f.i32(&format!("{prefix}selected_lengths"));
        let width = cfg.list_width();
        let k = mlx_heads(&f, "k");
        let what = format!("{name} {prefix} raw");
        let mut cache = DlsaBucketCache::new(cfg).unwrap();
        if f.has("q") {
            let q = mlx_heads(&f, "q");
            let t = f.key_count();
            let prefill = f.prefill_len().unwrap();
            let mut at = 0;
            while at < t {
                let step = if at < prefill { 5.min(prefill - at) } else { 1 };
                let keys = index_keys(&rows_slice(&k, at, at + step), &kmap, &geom).unwrap();
                cache.append_mlx(&keys).unwrap();
                let qs = index_queries(&rows_slice(&q, at, at + step), &qmap, &geom).unwrap();
                let rows: Vec<usize> = (at..at + step).collect();
                let sel = cache.select(&qs, &rows).unwrap();
                for (i, &p) in rows.iter().enumerate() {
                    assert_row_eq(
                        &sel,
                        i,
                        row(&selected, width, p),
                        lengths[p],
                        &format!("{what} p {p}"),
                    );
                }
                cache.commit();
                at += step;
            }
        } else {
            // qwen_layer stores only the attention rows' queries
            let q_rows = mlx_heads(&f, "q_rows");
            let qs = index_queries(&q_rows, &qmap, &geom).unwrap();
            for (i, &p) in f.i32("attention_rows").iter().enumerate() {
                let p = p as usize;
                let at = cache.len();
                let keys = index_keys(&rows_slice(&k, at, p + 1), &kmap, &geom).unwrap();
                cache.append_mlx(&keys).unwrap();
                cache.commit();
                let sel = cache.select(&rows_slice(&qs, i, i + 1), &[p]).unwrap();
                assert_row_eq(
                    &sel,
                    0,
                    row(&selected, width, p),
                    lengths[p],
                    &format!("{what} p {p}"),
                );
            }
        }
    }
}

#[test]
fn fixture_short_dense_reads_every_key() {
    let f = Fixture::load("short_dense");
    let cfg = f.cfg();
    let keys = f.f32("index_keys");
    let queries = f.f32("index_queries");
    let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
    let positions: Vec<usize> = (0..f.key_count()).collect();
    let sel = cache.select_host(&queries, &positions).unwrap();
    let want: Vec<i32> = (1..=42).collect();
    assert_eq!(sel.lengths, want);
    for &p in &positions {
        let mut ids = sel.list(p).to_vec();
        ids.sort_unstable();
        assert_eq!(ids, (0..=p as i32).collect::<Vec<_>>(), "p {p}");
    }
}

/// The list semantics carried to attention: a CPU fp64 softmax over each `tiny` row's list
/// (RoPE is the identity in the fixtures) matches the reference's `attention_out`.
#[test]
fn fixture_tiny_attention_over_lists() {
    for name in ["tiny", "tiny_ties", "short_dense"] {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        let geom = f.geometry();
        let (hq, hkv, dh) = (geom.query_heads, geom.kv_heads, geom.head_dim);
        let (q, k, v) = (f.f32("q"), f.f32("k"), f.f32("v"));
        let cache = DlsaBucketCache::build(cfg, &f.f32("index_keys")).unwrap();
        let positions: Vec<usize> = (0..f.key_count()).collect();
        let sel = cache
            .select_host(&f.f32("index_queries"), &positions)
            .unwrap();
        let want = f.f32("attention_out");
        let scale = (dh as f64).powf(-0.5);
        let mut got = Vec::with_capacity(want.len());
        for &p in &positions {
            for h in 0..hq {
                let g = h / (hq / hkv);
                let qv = &q[(p * hq + h) * dh..(p * hq + h + 1) * dh];
                let logits: Vec<f64> = sel
                    .list(p)
                    .iter()
                    .map(|&j| {
                        let kv = &k[(j as usize * hkv + g) * dh..(j as usize * hkv + g + 1) * dh];
                        qv.iter()
                            .zip(kv)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for t in 0..dh {
                    let o: f64 = sel
                        .list(p)
                        .iter()
                        .zip(&w)
                        .map(|(&j, wi)| wi * v[(j as usize * hkv + g) * dh + t] as f64)
                        .sum();
                    got.push((o / z) as f32);
                }
            }
        }
        assert_close(&got, &want, 1e-5, &format!("{name} attention_out"));
    }
}

// ------------------------------------------------------------------------------------------------
// Synthetic data and properties (host paths: no GPU)
// ------------------------------------------------------------------------------------------------

/// splitmix64.
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
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(NORM_EPS);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

/// `n` unit vectors of `d` dims in topical runs; `quantized` draws components from {-1, 0, 1}
/// so that equal vectors, equal pools, equal costs and equal scores are common.
fn synthetic_vectors(rng: &mut Rng, n: usize, d: usize, quantized: bool) -> Vec<f32> {
    let topics: Vec<Vec<f32>> = (0..6)
        .map(|_| (0..d).map(|_| rng.normal()).collect())
        .collect();
    let mut out = Vec::with_capacity(n * d);
    let mut topic = 0;
    for i in 0..n {
        if i == 0 || rng.below(4) == 0 {
            topic = rng.below(topics.len());
        }
        let v: Vec<f32> = if quantized {
            (0..d).map(|_| rng.below(3) as f32 - 1.0).collect()
        } else {
            topics[topic]
                .iter()
                .map(|c| 0.8 * c + 0.6 * rng.normal())
                .collect()
        };
        let v = if v.iter().all(|&x| x == 0.0) {
            let mut v = v;
            v[0] = 1.0;
            v
        } else {
            v
        };
        out.extend(normalized(v));
    }
    out
}

fn random_config(rng: &mut Rng) -> DlsaIndexConfig {
    let pool_size = [1, 2, 4][rng.below(3)];
    let bucket_size = pool_size * rng.range(1, 3);
    let window_size = bucket_size * rng.range(1, 6);
    let bucket_cap = pool_size * (2 * bucket_size / pool_size + rng.range(0, 3));
    let top_p = rng.range(1, 8);
    let cfg = DlsaIndexConfig {
        window_size,
        sink_size: rng.range(0, 4),
        top_k: rng.range(1, top_p * bucket_cap),
        top_p,
        index_dim: rng.range(2, 8),
        pool_size,
        bucket_size,
        bucket_cap,
    };
    cfg.validate().unwrap();
    cfg
}

/// The reference's read of one row, written independently: a full stable sort of the visible
/// scores descending (equal scores keep bucket order), then the flatten-filter-truncate of
/// `expand_buckets`.
fn naive_list(cache: &DlsaBucketCache, query: &[f32], p: usize) -> Vec<i32> {
    let cfg = cache.config();
    let d = cfg.index_dim;
    let v = cache.visible_buckets(p);
    let scores = score_buckets_host(query, cache.centroids(), d, v);
    let mut order: Vec<usize> = (0..v).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    order.truncate(cfg.top_p);
    let ws = (p + 1).saturating_sub(cfg.window_size);
    let mut list: Vec<i32> = order
        .iter()
        .flat_map(|&b| row(cache.members(), cfg.bucket_cap, b).to_vec())
        .filter(|&m| m >= cfg.sink_size as i32 && (m as i64) < ws as i64)
        .take(cfg.top_k)
        .collect();
    list.extend(ws as i32..=p as i32);
    list.extend(0..cfg.sink_size.min(ws).min(p + 1) as i32);
    list
}

/// Every invariant of one selected row (`docs/dlsa.md` §4.5, §4.7).
#[track_caller]
fn check_row(cache: &DlsaBucketCache, sel: &DlsaSelection, r: usize, p: usize) {
    let cfg = cache.config();
    let (w, cap) = (cfg.window_size, cfg.bucket_cap);
    let full = row(&sel.lists, sel.width, r);
    let len = sel.lengths[r] as usize;
    assert!(len <= sel.width);
    assert!(full[len..].iter().all(|&x| x == -1), "padding after length");
    let list = &full[..len];
    assert!(
        list.iter().all(|&x| x >= 0 && x as usize <= p),
        "an id outside [0, p] at p {p}"
    );
    let mut sorted = list.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), len, "duplicate id at p {p}");
    let ws = (p + 1).saturating_sub(w);
    let sinks = cfg.sink_size.min(ws).min(p + 1);
    let window_len = p + 1 - ws;
    let exp_len = len - window_len - sinks;
    let (expansion, rest) = list.split_at(exp_len);
    assert_eq!(
        rest[..window_len],
        (ws as i32..=p as i32).collect::<Vec<_>>()[..],
        "window"
    );
    assert_eq!(
        rest[window_len..],
        (0..sinks as i32).collect::<Vec<_>>()[..],
        "sinks"
    );
    let closed_end = (p / w * w) as i32;
    assert!(
        expansion
            .iter()
            .all(|&m| m >= cfg.sink_size as i32 && (m as usize) < ws && m < closed_end),
        "expansion outside closed blocks before the window at p {p}"
    );
    assert!(exp_len <= cfg.top_k);
    let picked: Vec<usize> = row(&sel.picked, sel.top_p, r)
        .iter()
        .take_while(|&&b| b >= 0)
        .map(|&b| b as usize)
        .collect();
    let v = cache.visible_buckets(p);
    assert_eq!(picked.len(), v.min(cfg.top_p));
    assert!(picked.iter().all(|&b| b < v));
    let offered: usize = picked
        .iter()
        .flat_map(|&b| row(cache.members(), cap, b))
        .filter(|&&m| m >= cfg.sink_size as i32 && (m as usize) < ws)
        .count();
    assert_eq!(
        exp_len,
        offered.min(cfg.top_k),
        "expansion fills top_k when it can"
    );
    // Short context: every causal key (`docs/dlsa.md` §4.7)
    if p < cfg.list_width() && v <= cfg.top_p {
        assert_eq!(
            sorted,
            (0..=p as i32).collect::<Vec<_>>(),
            "short context reads all keys"
        );
    }
}

#[test]
fn property_lists_and_naive_oracle() {
    for seed in 0..60u64 {
        let mut rng = Rng(seed);
        let cfg = random_config(&mut rng);
        let n = rng.range(1, 6 * cfg.window_size + 5);
        let quantized = seed % 3 == 0;
        let keys = synthetic_vectors(&mut rng, n, cfg.index_dim, quantized);
        let queries = synthetic_vectors(&mut rng, n, cfg.index_dim, quantized);
        let mut cache = DlsaBucketCache::new(cfg).unwrap();
        let mut at = 0;
        while at < n {
            let step = rng.range(1, 2 * cfg.window_size + 3).min(n - at);
            cache
                .append(&keys[at * cfg.index_dim..(at + step) * cfg.index_dim])
                .unwrap();
            let rows: Vec<usize> = (at..at + step).collect();
            let sel = cache
                .select_host(
                    &queries[at * cfg.index_dim..(at + step) * cfg.index_dim],
                    &rows,
                )
                .unwrap();
            for (i, &p) in rows.iter().enumerate() {
                check_row(&cache, &sel, i, p);
                assert_eq!(
                    sel.list(i),
                    naive_list(&cache, row(&queries, cfg.index_dim, p), p),
                    "seed {seed} {cfg:?} p {p}"
                );
            }
            cache.commit();
            at += step;
        }
    }
}

#[test]
fn property_qwen_defaults_lists() {
    let cfg = DlsaIndexConfig::QWEN_DEFAULTS;
    let mut rng = Rng(77);
    let n = 6000;
    let keys = synthetic_vectors(&mut rng, n, 64, false);
    let queries = synthetic_vectors(&mut rng, n, 64, false);
    let cache = DlsaBucketCache::build(cfg, &keys).unwrap();
    let positions: Vec<usize> = (0..n)
        .step_by(37)
        .chain([4611, 4612, 5119, 5120, n - 1])
        .collect();
    let q: Vec<f32> = positions
        .iter()
        .flat_map(|&p| row(&queries, 64, p).to_vec())
        .collect();
    let sel = cache.select_host(&q, &positions).unwrap();
    for (r, &p) in positions.iter().enumerate() {
        check_row(&cache, &sel, r, p);
    }
    // p <= 4611 reads every causal key (4612 = sink + top_k + window)
    let r = positions.iter().position(|&p| p == 4611).unwrap();
    assert_eq!(sel.lengths[r], 4612);
}

#[test]
fn property_ward_partitions() {
    for seed in 0..80u64 {
        let mut rng = Rng(1000 + seed);
        let cfg = random_config(&mut rng);
        let keys = synthetic_vectors(&mut rng, cfg.window_size, cfg.index_dim, seed % 2 == 0);
        let base = cfg.window_size * rng.below(5);
        let b = cluster_block(&keys, base, &cfg).unwrap();
        let m = cfg.buckets_per_block();
        assert_eq!(b.sizes.len(), m);
        assert_eq!(b.sizes.iter().sum::<i32>() as usize, cfg.window_size);
        let mut seen = vec![false; cfg.window_size];
        let mut first_pool = Vec::new();
        for bucket in 0..m {
            let size = b.sizes[bucket] as usize;
            assert!(size >= 1 && size <= cfg.bucket_cap, "bucket size {size}");
            assert_eq!(size % cfg.pool_size, 0, "a pool was split");
            let mem = row(&b.members, cfg.bucket_cap, bucket);
            assert!(mem[size..].iter().all(|&x| x == -1));
            assert!(
                mem[..size].windows(2).all(|w| w[0] < w[1]),
                "members ascending"
            );
            for chunk in mem[..size].chunks(cfg.pool_size) {
                let p0 = chunk[0] as usize - base;
                assert_eq!(p0 % cfg.pool_size, 0);
                for (i, &x) in chunk.iter().enumerate() {
                    assert_eq!(x as usize - base, p0 + i, "pool kept whole");
                }
            }
            for &x in &mem[..size] {
                assert!(!seen[x as usize - base]);
                seen[x as usize - base] = true;
            }
            first_pool.push(mem[0]);
            let c = row(&b.centroids, cfg.index_dim, bucket);
            let n: f32 = c.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5);
        }
        assert!(seen.iter().all(|&s| s));
        assert!(
            first_pool.windows(2).all(|w| w[0] < w[1]),
            "numbered by first pool"
        );
        // Deterministic: same keys, bit-identical tables
        assert_eq!(cluster_block(&keys, base, &cfg).unwrap(), b);
    }
}

#[test]
fn ward_errors_when_the_cap_stops_it_short() {
    // Four pools of size 2, cap 3: nothing can merge, 2 buckets are unreachable
    let pools = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    assert!(ward_labels(&pools, &[2.0; 4], 2, 2, 3).is_err());
    // Enough buckets already: identity labels, no rounds
    let w = ward_labels(&pools, &[2.0; 4], 2, 4, 3).unwrap();
    assert_eq!((w.labels, w.rounds), (vec![0, 1, 2, 3], 0));
}

/// Equal bucket tables (bitwise) of two caches.
#[track_caller]
fn assert_same_tables(a: &DlsaBucketCache, b: &DlsaBucketCache, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(a.centroids()),
        bits(b.centroids()),
        "{what}: centroids"
    );
    assert_eq!(a.members(), b.members(), "{what}: members");
    assert_eq!(a.sizes(), b.sizes(), "{what}: sizes");
}

#[test]
fn incremental_equals_batch() {
    for seed in 0..30u64 {
        let mut rng = Rng(5000 + seed);
        let cfg = random_config(&mut rng);
        let n = rng.range(1, 5 * cfg.window_size + 7);
        let d = cfg.index_dim;
        let keys = synthetic_vectors(&mut rng, n, d, seed % 2 == 1);
        let queries = synthetic_vectors(&mut rng, n, d, seed % 2 == 1);
        let batch = DlsaBucketCache::build(cfg, &keys).unwrap();
        let all: Vec<usize> = (0..n).collect();
        let batch_sel = batch.select_host(&queries, &all).unwrap();
        // Token by token, each row read as it arrives
        let mut tok = DlsaBucketCache::new(cfg).unwrap();
        // Random chunks crossing 0, 1 or 2 block boundaries, without commits
        let mut chunked = DlsaBucketCache::new(cfg).unwrap();
        for p in 0..n {
            tok.append(row(&keys, d, p)).unwrap();
            tok.commit();
            let sel = tok.select_host(row(&queries, d, p), &[p]).unwrap();
            assert_eq!(sel.list(0), batch_sel.list(p), "seed {seed} token p {p}");
        }
        let mut at = 0;
        while at < n {
            let step = rng.range(1, 2 * cfg.window_size + 3).min(n - at);
            chunked.append(&keys[at * d..(at + step) * d]).unwrap();
            let rows: Vec<usize> = (at..at + step).collect();
            let sel = chunked
                .select_host(&queries[at * d..(at + step) * d], &rows)
                .unwrap();
            for (i, &p) in rows.iter().enumerate() {
                assert_eq!(sel.list(i), batch_sel.list(p), "seed {seed} chunk p {p}");
            }
            at += step;
        }
        assert_same_tables(&tok, &batch, "token vs batch");
        assert_same_tables(&chunked, &batch, "chunks vs batch");
    }
}

/// Projection batched per chunk vs per token from raw fixture heads: bucket tables agree to
/// rounding ("a projection rounds by the size of its batch", `docs/dlsa.md` §7.4) and the
/// lists are identical.
#[test]
fn projection_batch_size_does_not_change_lists() {
    for (name, prefix) in [("tiny", ""), ("tiny_ties", ""), ("qwen_layer", "real.")] {
        let f = Fixture::load(name);
        let cfg = f.cfg();
        let geom = f.geometry();
        let d = cfg.index_dim;
        let kmap = mlx_f32(&f.f32(&format!("{prefix}key_map")), &[geom.input_dim(), d]);
        let k = mlx_heads(&f, "k");
        let t = f.key_count();
        let whole =
            DlsaBucketCache::build(cfg, &to_host_f32(&index_keys(&k, &kmap, &geom).unwrap()))
                .unwrap();
        let mut per_token = DlsaBucketCache::new(cfg).unwrap();
        for p in 0..t {
            per_token
                .append_mlx(&index_keys(&rows_slice(&k, p, p + 1), &kmap, &geom).unwrap())
                .unwrap();
            per_token.commit();
        }
        assert_close(
            per_token.centroids(),
            whole.centroids(),
            1e-6,
            &format!("{name} centroids"),
        );
        assert_eq!(per_token.members(), whole.members(), "{name} members");
        let queries = f.f32(&format!("{prefix}index_queries"));
        let positions: Vec<usize> = (0..queries.len() / d).collect();
        let a = whole.select_host(&queries, &positions).unwrap();
        let b = per_token.select_host(&queries, &positions).unwrap();
        assert_eq!(a.lists, b.lists, "{name}: lists");
    }
}

#[test]
fn multi_row_verify_crossing_a_block_boundary() {
    let cfg = DlsaIndexConfig {
        window_size: 16,
        sink_size: 2,
        top_k: 12,
        top_p: 4,
        index_dim: 8,
        pool_size: 2,
        bucket_size: 4,
        bucket_cap: 8,
    };
    let d = cfg.index_dim;
    let mut rng = Rng(9);
    let n = 64;
    let keys = synthetic_vectors(&mut rng, n, d, false);
    let queries = synthetic_vectors(&mut rng, n, d, false);
    let batch = DlsaBucketCache::build(cfg, &keys).unwrap();
    // Decode to 16*2 + 13, then one 8-row verify at 45..53 closes block 2 (key 47) mid-forward
    let start = 45;
    let mut cache = DlsaBucketCache::build(cfg, &keys[..start * d]).unwrap();
    assert_eq!(cache.rows(), 8);
    cache.append(&keys[start * d..(start + 8) * d]).unwrap();
    assert_eq!(cache.rows(), 12, "the block closed inside the forward");
    let rows: Vec<usize> = (start..start + 8).collect();
    let sel = cache
        .select_host(&queries[start * d..(start + 8) * d], &rows)
        .unwrap();
    for (i, &p) in rows.iter().enumerate() {
        let want = if p < 48 { 8 } else { 12 };
        assert_eq!(sel.visible[i], want, "row {p} sees block 2 only from 48 on");
        let one = batch.select_host(row(&queries, d, p), &[p]).unwrap();
        assert_eq!(sel.list(i), one.list(0), "verify row {p} == batch");
        check_row(&cache, &sel, i, p);
    }
}

#[test]
fn rollback_reopens_a_block_and_matches_a_fresh_build() {
    let cfg = DlsaIndexConfig {
        window_size: 16,
        sink_size: 2,
        top_k: 12,
        top_p: 4,
        index_dim: 8,
        pool_size: 2,
        bucket_size: 4,
        bucket_cap: 8,
    };
    let d = cfg.index_dim;
    let mut rng = Rng(11);
    let accepted = synthetic_vectors(&mut rng, 80, d, false);
    let rejected = synthetic_vectors(&mut rng, 8, d, false);
    let queries = synthetic_vectors(&mut rng, 80, d, false);
    // Committed through 45 (block 2 open at 32..48), a verify appends 45..53 with draft keys
    let mut cache = DlsaBucketCache::build(cfg, &accepted[..45 * d]).unwrap();
    let mem_before = cache.memory();
    cache.append(&accepted[45 * d..47 * d]).unwrap(); // rows 45, 46 accepted
    cache.append(&rejected[..6 * d]).unwrap(); // rows 47..53 rejected; block 2 closed on 47
    assert_eq!(cache.rows(), 12);
    assert!(cache.memory().tail_keys > mem_before.tail_keys);
    // Accept 45, 46: roll back to 47, which reopens block 2
    cache.trim(47).unwrap();
    assert_eq!((cache.len(), cache.rows()), (47, 8));
    cache.commit();
    assert_same_tables(
        &cache,
        &DlsaBucketCache::build(cfg, &accepted[..47 * d]).unwrap(),
        "after trim",
    );
    // Continue with the accepted keys: the block closes again from the right keys
    for p in 47..80 {
        cache.append(row(&accepted, d, p)).unwrap();
        let sel = cache.select_host(row(&queries, d, p), &[p]).unwrap();
        let fresh = DlsaBucketCache::build(cfg, &accepted[..(p + 1) * d]).unwrap();
        assert_eq!(
            sel,
            fresh.select_host(row(&queries, d, p), &[p]).unwrap(),
            "p {p}"
        );
        cache.commit();
    }
    assert_same_tables(
        &cache,
        &DlsaBucketCache::build(cfg, &accepted).unwrap(),
        "end",
    );
}

#[test]
fn rollback_limits_and_commit_frees_keys() {
    let cfg = DlsaIndexConfig {
        window_size: 16,
        sink_size: 2,
        top_k: 12,
        top_p: 4,
        index_dim: 8,
        pool_size: 2,
        bucket_size: 4,
        bucket_cap: 8,
    };
    let d = cfg.index_dim;
    let mut rng = Rng(13);
    let keys = synthetic_vectors(&mut rng, 100, d, false);
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    cache.append(&keys[..40 * d]).unwrap();
    // Nothing committed yet: every key is retained, a rollback to 0 is possible
    assert_eq!(cache.tail_start(), 0);
    assert_eq!(cache.tail_keys().len(), 40 * d);
    cache.commit();
    // Commit at 40 frees blocks 0 and 1 (keys 0..32); the open block 32..40 stays
    assert_eq!((cache.tail_start(), cache.tail_keys().len()), (32, 8 * d));
    assert_eq!(cache.memory().tail_keys, 8 * d * 4);
    assert!(cache.trim(31).is_err(), "below tail_start");
    assert!(cache.trim(41).is_err(), "past len");
    cache.append(&keys[40 * d..70 * d]).unwrap();
    assert_eq!(cache.rows(), 16);
    // Roll back across two closed blocks to 33, still >= tail_start
    cache.trim(33).unwrap();
    assert_eq!((cache.len(), cache.rows(), cache.committed()), (33, 8, 33));
    let mut fresh = DlsaBucketCache::build(cfg, &keys[..33 * d]).unwrap();
    assert_same_tables(&cache, &fresh, "trim(33)");
    cache.append(&keys[33 * d..100 * d]).unwrap();
    fresh.append(&keys[33 * d..100 * d]).unwrap();
    assert_same_tables(&cache, &fresh, "regrown");
    // An empty cache: no buckets, short rows read every key
    cache.reset();
    assert_eq!((cache.len(), cache.rows(), cache.nbytes()), (0, 0, 0));
    cache.append(row(&keys, d, 0)).unwrap();
    let sel = cache.select_host(row(&keys, d, 0), &[0]).unwrap();
    assert_eq!(sel.list(0), [0]);
    assert!(
        cache.select_host(row(&keys, d, 0), &[1]).is_err(),
        "future position"
    );
}

/// `memory()`/`nbytes()` account for what is allocated, and commit/reset release it: a batch
/// build must not keep its whole key buffer after the commit.
#[test]
fn allocation_accounting_after_build_commit_and_reset() {
    let cfg = DlsaIndexConfig {
        window_size: 16,
        sink_size: 2,
        top_k: 12,
        top_p: 4,
        index_dim: 8,
        pool_size: 2,
        bucket_size: 4,
        bucket_cap: 8,
    };
    let d = cfg.index_dim;
    let bound = 2 * cfg.window_size * d * 4; // bytes of retained-key capacity kept across commits
    let mut rng = Rng(23);
    let n = 4000 + 5;
    let keys = synthetic_vectors(&mut rng, n, d, false);
    // Before the commit the whole buffer is held and reported
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    cache.append(&keys).unwrap();
    assert!(cache.memory().tail_keys_allocated >= n * d * 4);
    assert_eq!(cache.nbytes(), cache.memory().allocated());
    cache.commit();
    let mem = cache.memory();
    assert_eq!(mem.tail_keys, 5 * d * 4, "only the open block is retained");
    assert!(mem.tail_keys_allocated >= mem.tail_keys);
    assert!(
        mem.tail_keys_allocated <= bound,
        "tail capacity {} > {bound}",
        mem.tail_keys_allocated
    );
    // One append closing every block reserves the tables exactly
    assert_eq!(mem.host_tables_allocated, mem.host_tables);
    let built = DlsaBucketCache::build(cfg, &keys).unwrap();
    assert_eq!(built.memory(), mem);
    assert_eq!(built.nbytes(), mem.host_tables + mem.tail_keys_allocated);
    // Decode: one key a step stays within the bound, tables within geometric growth
    for p in 0..300 {
        cache.append(row(&keys, d, p)).unwrap();
        cache.commit();
        let m = cache.memory();
        assert!(m.tail_keys_allocated <= bound);
        assert!(m.host_tables_allocated <= 2 * m.host_tables);
    }
    // A rollback keeps capacity for the regrowth; reset releases everything
    cache.append(&keys[..40 * d]).unwrap();
    cache.trim(cache.len() - 30).unwrap();
    assert!(cache.memory().allocated() >= cache.memory().total());
    cache.reset();
    assert_eq!(cache.memory(), DlsaCacheMemory::default());
    assert_eq!(cache.nbytes(), 0);
}
#[test]
fn snapshot_restore_continues_identically() {
    let cfg = DlsaIndexConfig {
        window_size: 32,
        sink_size: 4,
        top_k: 40,
        top_p: 8,
        index_dim: 16,
        pool_size: 4,
        bucket_size: 8,
        bucket_cap: 16,
    };
    let d = cfg.index_dim;
    let mut rng = Rng(17);
    let n = 300;
    let keys = synthetic_vectors(&mut rng, n, d, false);
    let queries = synthetic_vectors(&mut rng, n, d, false);
    let mut straight = DlsaBucketCache::build(cfg, &keys[..150 * d]).unwrap();
    // An uncommitted append before the snapshot is captured as committed
    let mut source = DlsaBucketCache::build(cfg, &keys[..140 * d]).unwrap();
    source.append(&keys[140 * d..150 * d]).unwrap();
    let snap = source.snapshot();
    assert_eq!(snap.open_keys.len(), (150 - 128) * d);
    assert_eq!(snap.centroids.len(), 4 * 4 * d);
    let mut restored = DlsaBucketCache::restore(cfg, &snap).unwrap();
    assert_eq!(restored.snapshot(), snap);
    assert_eq!(restored.tail_start(), 128);
    for p in 150..n {
        straight.append(row(&keys, d, p)).unwrap();
        restored.append(row(&keys, d, p)).unwrap();
        let a = straight.select_host(row(&queries, d, p), &[p]).unwrap();
        let b = restored.select_host(row(&queries, d, p), &[p]).unwrap();
        assert_eq!(a, b, "p {p}");
        straight.commit();
        restored.commit();
    }
    assert_same_tables(&straight, &restored, "restored");
    // A config mismatch or inconsistent tables are refused
    let mut other = cfg;
    other.top_k = 32;
    assert!(DlsaBucketCache::restore(other, &snap).is_err());
    let mut bad = snap.clone();
    bad.open_keys.pop();
    assert!(DlsaBucketCache::restore(cfg, &bad).is_err());
}

/// The MLX scoring path stays in sync with the host tables across growth, trim and restore.
#[test]
fn mlx_select_tracks_cache_changes() {
    let cfg = DlsaIndexConfig {
        window_size: 16,
        sink_size: 2,
        top_k: 12,
        top_p: 4,
        index_dim: 8,
        pool_size: 2,
        bucket_size: 4,
        bucket_cap: 8,
    };
    let d = cfg.index_dim;
    let mut rng = Rng(19);
    let keys = synthetic_vectors(&mut rng, 120, d, false);
    let other = synthetic_vectors(&mut rng, 120, d, false);
    let queries = synthetic_vectors(&mut rng, 120, d, false);
    let mut cache = DlsaBucketCache::new(cfg).unwrap();
    let check = |cache: &mut DlsaBucketCache, p: usize| {
        let q = row(&queries, d, p);
        let mlx = cache.select(&mlx_f32(q, &[1, d]), &[p]).unwrap();
        assert_eq!(mlx, cache.select_host(q, &[p]).unwrap(), "p {p}");
    };
    for p in 0..60 {
        cache.append(row(&keys, d, p)).unwrap();
        check(&mut cache, p);
    }
    assert!(cache.memory().device_centroids > 0);
    // Roll back past a block and close it again with other keys under the same row numbers
    cache.trim(40).unwrap();
    cache.append(&other[40 * d..60 * d]).unwrap();
    check(&mut cache, 59);
    let snap = cache.snapshot();
    let mut restored = DlsaBucketCache::restore(cfg, &snap).unwrap();
    check(&mut restored, 59);
}
