# D-LSA golden fixtures (flat design)

Golden outputs of the upstream D-LSA torch reference, for testing QW's port of the flat design used for
Qwen3.8-27B (spec: [`docs/dlsa.md`](../../../docs/dlsa.md)).

- Upstream: <https://github.com/Prefrontal-AI/latent-sparse-attention> at commit
  `5ebea732d639576d2a7579b0de40e6252f9001f5` (Apache-2.0). The values are derived from its reference code.
- Real map: release `d-lsa-weights-v1`, asset `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`,
  sha256 `24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b`. Only layer `3` is copied in, as
  `real.query_map` / `real.key_map` in `qwen_layer.safetensors`.
- Generator: [`gen_fixtures.py`](gen_fixtures.py). This is an **offline dev tool and not part of the build**. Its
  header says how to set up the scratch environment (pinned clone, `uv venv --python 3.12`, `torch==2.13.0 pyyaml`;
  NumPy is not needed, since the writer copies tensor memory with `ctypes`) and how to regenerate. The output is
  deterministic: two runs give byte-identical files. Verified in a fresh venv without NumPy. CPU only, about 2 s.
- Format: one `.safetensors` file per case, which `qw-runtime`'s existing `safetensors` dependency can read. The
  `__metadata__` of each file carries `case`, `upstream_commit` and `config` (JSON).
  [`manifest.json`](manifest.json) lists every case with its config, shapes, sha256, byte size, the dtype and shape of
  every tensor, and diagnostics.
- Total size: about 5.3 MB (`qwen_layer` 3.0 MB, `qwen_index_defaults` 1.9 MB, the rest 0.35 MB).

## Conventions (all cases)

- Positions are 0-based token indices within one sequence. Query row `i` sits at position `i`, except in
  `qwen_index_defaults`, which carries `positions`. Key ids in lists are positions.
- `q`, `k`, `v` are the layer's heads **after q_norm/k_norm and before RoPE**, in the layout
  `[T, heads, head_dim]`. They are synthetic. Each value is fp16-representable and stored as F16. The reference
  computed with exactly these values upcast to fp32, so the stored F16 is lossless.
- **RoPE is the identity in these fixtures.** The attention outputs attend with `q`/`k` as given. The index never
  reads the rotated prefix: the generator asserts that rewriting dims `[0, rotary_dim)` leaves the index bit-identical.
  So a Rust test either uses identity RoPE for the attention check, or compares only index tensors when it applies real RoPE.
- GQA: query head `h` reads KV head `h / (Hq/Hkv)`. Group `g` is query heads `g*G .. g*G+G-1`. Concatenation is head-major.
- Attention scale is `head_dim^-0.5`. Softmax and accumulation are in fp32.
- `-1` pads every index table (`picked_buckets`, `bucket_members`, `selected`).
- Bucket rows are numbered block after block. Closed cluster block `b` owns rows `[b*m, (b+1)*m)`, where
  `m = window_size / bucket_size`. Within a block, buckets are numbered in the order of their first pool.
- Prefill means the rows `< prefill_len`, read against the cache built from keys `[0, prefill_len)`. Decode means each
  row `p >= prefill_len`, read against a cache rebuilt from keys `[0, p]` (one new key a step), as in incremental
  decode. The generator asserts both equal a single read against the final cache, so a list depends on the position
  alone, never on chunking.

## Tensors

`<m>.` prefixes the per-map tensors of `qwen_layer` (`random.` or `real.`). The other cases have one map and no prefix.

| tensor | dtype, shape | meaning | upstream function that produced it |
|---|---|---|---|
| `q` / `q_rows` | F16 `[T, Hq, dh]` / `[R, Hq, dh]` | query heads (post-norm, pre-RoPE). `qwen_layer` stores only the `attention_rows` | synthetic |
| `k`, `v` | F16 `[T, Hkv, dh]` | key heads (post-norm, pre-RoPE) and value heads | synthetic |
| `attention_rows` | I32 `[R]` | positions whose attention output is stored (all rows in the tiny cases) | - |
| `query_map`, `key_map` | F16 (random, fp16-exact) or F32 (real) `[Hkv*(dh-rot), d]` | `W_Q_idx`, `W_K_idx`, laid out `[input, index_dim]` as in the `.pt` | - |
| `grouped_queries` | F32 `[R, Hkv*(dh-rot)]` | each KV group's query heads summed over their position-free dims, groups concatenated, not normalized | `index_space.regroup_partial` over `position_free` |
| `concat_keys` | F32 `[T, Hkv*(dh-rot)]` | each KV head's position-free dims, concatenated (tiny cases only) | same |
| `index_keys` | F32 `[T, d]` (BF16 in `qwen_index_defaults`) | `normalize(concat_keys @ key_map)` | `index_space.regroup` + `project` |
| `index_queries` | F32 `[T, d]` (`[R, d]` in `qwen_index_defaults`) | `normalize(grouped_queries @ query_map)` | same |
| `pool_keys` | F32 `[blocks, window/pool, d]` | sum of each run of `pool_size` index keys of every closed cluster block | `ward.pool_keys` |
| `ward_labels` | I32 `[blocks, window/pool]` | bucket of each pool within its block, `0..m-1` | `ward.ward_merge`, cross-checked against the kernel tie policy (`ward_labels_kernel_policy`) |
| `bucket_centroids` | F32 `[rows, d]` | normalized key sum of each bucket | `reference.build_buckets` |
| `bucket_members` | I32 `[rows, bucket_cap]` | absolute positions of each bucket's keys, ascending, `-1` padded | same |
| `bucket_sizes` | I32 `[rows]` | keys in each bucket | same |
| `visible_buckets` | I32 `[T]` / `[R]` | buckets the row may score: `(p / window) * m` | `reference._read_row` line 262 |
| `bucket_scores` | F32 `[T, rows]` / `[R, rows]` | `index_query . centroid` for **every** bucket of the final cache. Only the first `visible_buckets` columns are in play | `reference.read` line 228 |
| `picked_buckets` | I32 `[T, top_p]` | visible buckets, best score first, ties to the lower bucket, at most `top_p`, `-1` padded | `reference._read_row` line 264 |
| `selected` | I32 `[T, top_k+window+sink]` | the read list: expansion (picked buckets' members outside sink and window, in pick order then position order, cut at `top_k`), then the window ascending, then the sinks ascending, `-1` padded | `reference.read` |
| `selected_lengths` | I32 `[T]` | valid entries of each list | same |
| `attention_out` | F32 `[R, Hq, dh]` | causal attention of the row over its list (window and sinks included), fp32 | `attention.topk_attention` |
| `dense_out` | F32 `[R, Hq, dh]` | plain causal dense attention, an independent oracle (tiny cases only) | generator's `dense_attention` |
| `positions` | I32 `[R]` | query positions (`qwen_index_defaults` only) | - |
| `pool_sizes`, `reference_ward_labels` | F32 `[1, 4]`, I32 `[1, 4]` | `ward_tie_policy` only: the pool sizes, and what this torch's CPU reference returned (implementation-defined) | `ward.ward_merge` |

## Cases

| case | geometry | budget | what it exercises |
|---|---|---|---|
| `tiny` | Hq 6, Hkv 2, dh 16, rot 4, d 8 | window 16, sink 2, top_k 12, top_p 4, pool 2, bucket 4, cap 8; T 84, prefill 78 | every stage at toy size. Decode crosses the block boundary at 80. 52 rows are truncated by top-p and cut at top-k |
| `tiny_ties` | same | same | a text repeating every 8 tokens: equal pools, equal merge costs, exact score ties (`min_top_p_boundary_gap` 0). Pins bucket-score ties (lower bucket wins) and Ward partner ties (lowest index). It has **no** oversubscribed equal-cost mutual pairs, so it cannot tell the reference's and the kernel's final-round tie policies apart |
| `ward_tie_policy` | 4 pools of 1 key, d 2 | 3 buckets, cap 4 | pools e1, e2, e2, e1: equal-cost mutual pairs (0,3) and (1,2) with one merge left. `ward_labels` = `[0,1,1,2]`, the served Triton kernel's policy (rank pairs by packed `(cost, partner index)`, so the lower absorbed index wins), which QW follows. `reference_ward_labels` = `[0,1,2,0]` from the CPU reference's unstable argsort, which is implementation-defined |
| `short_dense` | same as `tiny` | window 16, sink 2, top_k 24, top_p 8; T 42 = sink+top_k+window | below the budget every list is exactly the causal key set; `attention_out` matches `dense_out` to `allclose(1e-5)`, not bitwise (different summation order) |
| `qwen_layer` | **Qwen3.8-27B layer shapes**: Hq 24, Hkv 4, dh 256, rot 64, key input 768, d 64 | window 64, sink 4, top_k 64, top_p 16, pool 4, bucket 8, cap 16; T 260, prefill 252 | real head and index geometry with a scaled-down budget (to keep K/V small). Two maps on the same K/Q/V: `random.*` (random fp16 map) and `real.*` (layer 3 of the distilled map). Decode crosses the block boundary at 256 |
| `qwen_index_defaults` | key input 768 → d 64, real layer-3 map | the **served defaults**: window 512, sink 4, top_k 4096, top_p 1024, pool 4, bucket 8, cap 16; 9,216 keys | index only (no K/V): BF16 index keys, as the serving cache clusters them. Ward at its real size (128 pools → 64 buckets per block), lists of 4,612. Rows ≤ 4611 read every key; rows ≥ 8704 see 1,088 buckets and are cut by top-p |

## Robustness notes for tests

- `manifest.json` `diagnostics.min_top_p_boundary_gap` is the smallest score gap between the last picked bucket and the
  first unpicked one. For `real.*` in `qwen_layer` it is 3.3e-6, so a port must score in fp32 with an error well below
  that, or it may flip a pick. Test each stage from the fixture's own inputs of that stage (for example, picks from the
  stored `bucket_scores`, expansion from the stored `picked_buckets`), so errors do not compound.
- Ward tie policy: the generator re-runs every block's Ward with the served kernel's tie policy
  (`ward_labels_kernel_policy`: lowest-index partner, final-round pairs ranked by `(cost, partner index)`) and asserts it
  equals the reference's `ward_labels`. That holds only because no fixture except `ward_tie_policy` contains an
  oversubscribed equal-cost tie. It is not a general equivalence: the reference's choice there is implementation-defined,
  and the kernel's Gram is tf32x3.
- `diagnostics.serving_numerics_drift_real_map` records how many lists change under the served numerics (bf16 map,
  keys, centroids and query) against this fp32 reference: 2 of 260 rows, min Jaccard 0.94.
- `sparse_vs_dense_relative_error_*` measures synthetic data with flat attention. It is not an accuracy estimate.
