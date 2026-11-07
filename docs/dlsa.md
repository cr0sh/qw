# D-LSA (flat design) for Qwen3.8-27B: spec for the QW port

Status: roadmap tasks 0-7 done: this spec, golden fixtures in
[`tests/fixtures/dlsa/`](../tests/fixtures/dlsa/README.md), the map loader (§10), the indexer core (§11), sparse
attention in Qwen3.5's 16 full-attention layers behind `enable_dlsa` (§12), serving integration (§13: snapshots and
the prefix cache, MTP/DFlash2 verify and rollback, grammar, `qw serve --dlsa`), Metal kernels (§14), the throughput
evaluation and the `dense_below` threshold (§15), and RULER accuracy (§16). On a RULER subsample (10 rows per task)
at 32k, no task degrades under D-LSA: identical scores on 12 of 13 tasks, cwe +0.010 (2 of 130 rows change score, one
each way), mean 0.968 for both (upstream's bf16 D-LSA: 0.968). 64k and longer are not measured, and D-LSA stays
opt-in (§16.5).

With the default Metal backend and `dense_below` 12,288, D-LSA beats dense on the real model from about 20k tokens
(fp16 KV, M4 Max; §15.2, §15.3, §15.7):

| context | prefill tok/s, D-LSA vs dense | decode ms/token, D-LSA vs dense |
|---|---|---|
| 32k | 237.3 vs 225.6 (1.05×) | 38.9 vs 41.1 (1.06×) |
| 128k | 229.2 vs 138.5 (1.66×) | 39.5 vs 55.2 (1.40×) |
| 256k | 223.2 vs 83.9 (2.66×) | 39.9 vs 86.6 (2.17×) |

Below that, `dense_below` keeps the dense read: prefill at 8k–16k is within 0.5% of dense (the index build) and
decode equals dense (§15.7). The 32k prefill row is §15.7's run with the threshold; the 128k and 256k rows are §15.2's,
measured before it. §14.3–§14.4 hold task 4's earlier measurements, taken before the threshold existed (historical).

D-LSA (Distilled Latent Sparse Attention, arXiv 2610.42147) leaves the attention kernel unchanged. It only decides which
cached keys each query of a full-attention layer reads: the first `sink_size` keys, the last `window_size` keys, and up
to `top_k` long-range keys picked through a small learned index. This document specifies the **flat design**
(`top_level: 0`), the one served for Qwen3.8-27B, closely enough to implement it. It also maps the design onto QW.

**Source.** <https://github.com/Prefrontal-AI/latent-sparse-attention>, Apache-2.0, pinned at commit
**`5ebea732d639576d2a7579b0de40e6252f9001f5`** (2026-11-04). Citations `path:line` refer to that commit:
`src/lsa/...` for upstream code and `configs/...` for upstream configs. QW citations refer to this repo at `782b61b`.
Code ported from upstream must carry `SPDX-License-Identifier: Apache-2.0` and an "Adapted from …@5ebea73" header.

> **Correction to the task brief: the index dim is 64, not 2048.** In the map's name, "d2048" is the distillation's
> *target depth* (`run.objective.target_depth: 2048`, `configs/training/distillation/qwen38_27b/long_16k_1m_heavy_key_and_query_d2048.yaml:52-54`).
> The same config sets `index_dim: 64` (`:26`), and the released tensors are `[768, 64]` (§2). Every size below follows from d = 64.

---

## 1. Model facts (verified)

| fact | value | where verified |
|---|---|---|
| layers | 64: 48 Gated DeltaNet + **16 full attention**, every 4th | QW `qwen3_5.rs:254-256` (`!(i+1) % 4`), checkpoint `config.json` (`full_attention_interval: 4`), upstream `sglang/models/qwen3_5/index_space.py:122-146` |
| full-attention layer ids (0-based decoder index) | **3, 7, 11, …, 63** | same, and the map's keys (§2) |
| heads | 24 query heads, 4 KV heads, head_dim 256, GQA group G = 6 | checkpoint config, QW `qwen3_next.rs:200-202` |
| rotation | first **64** dims of each head (`partial_rotary_factor` 0.25), rotate-half pairs (i, i+32). Dims 64..255 are never rotated | QW `qwen3_next.rs:212-253`; upstream `assets/models/qwen38_27b.yaml:21-22` (`rotary_end: prefix`, `rotary_dim: 64`), `core/index_space.py:9-13` |
| QK norm | per-head RMSNorm over all 256 dims, **before** RoPE | QW `qwen3_next.rs:204-205`; upstream adapter `evaluation/adapter/qwen3_5.py:120-121` |
| output gate | second half of each head's 512-wide q_proj slice; `out * sigmoid(gate)` after attention | QW `qwen3_next.rs:182-197, 281-287`; upstream `qwen3_5.py:115-118`. The index never reads the gate |
| GQA mapping | query head h reads KV head h / 6 (heads 6g..6g+5 form group g) | QW `qwen3_next.rs` (MLX SDPA convention); upstream `core/attention.py:101`, `core/index_space.py:176-182` |
| native context / rope | 262,144; theta 1e7; mrope sections [11,11,10], which reduce to 1-D RoPE for text. QW parses no YaRN | checkpoint config; QW `qwen3_5.rs:215-247` (integration notes §2) |
| upstream eval rope | YaRN factor 40 over 262,144, static (applied at every length) | `assets/models/qwen38_27b.yaml:5-30`, `configs/evaluation/accuracy/qwen38_27b/_base.yaml:5-15` |

## 2. The map file (`.pt`)

Release `d-lsa-weights-v1`, inspected with `torch.load(weights_only=True)` on torch 2.13.0:

| asset | bytes | sha256 | verified |
|---|---|---|---|
| `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt` (served, distilled) | 6,302,911 | `24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b` | size and sha256 match the brief and the release's `SHA256SUMS` |
| `qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.pt` (training-free PCA baseline) | 6,303,063 | `baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65` | same |
| `.json` sidecars | - | - | **not in the release** (HTTP 404). Release assets: 3 GLM maps, these 2, and `SHA256SUMS`. The server reads only the `.pt` (`sglang/index/weights.py:120-162`) |

- **Container:** a torch zip archive (`<stem>/data.pkl` + `data/<n>` storages), holding a plain `dict` of 32 tensors. Reading
  it needs a pickle reader, so QW should convert it offline (§9, task 1).
- **Keys:** `"<layer>.W_Q_idx"` and `"<layer>.W_K_idx"` for layer ∈ {3, 7, …, 63}. These are **0-based decoder indices**,
  the same as QW's `layer_idx`. There are no other keys: no means, scales, biases or norms.
- **Tensors:** **float32**, shape **`[768, 64]`** = `[input_dim, index_dim]`, row-major and contiguous (stride (64, 1)). Use them
  as `x @ W`, not transposed. Row r covers input dim r of the head-major concatenation: rows 0..191 = KV head 0's dims
  64..255, rows 192..383 = KV head 1's, and so on. `weights.py:99-162` checks `[input_dim, index_dim]`.
- **Distilled map:** W_Q ≠ W_K. Column norms range from about 0.9 to 1.8 (per-element std about 0.042).
- **PCA map:** **W_Q == W_K exactly** in every layer. Its columns are orthonormal to within 4e-4 (the top 64 principal
  directions of the 768-dim key input, `configs/evaluation/accuracy/qwen38_27b/lsa_flat_pca_tp1.yaml:4-12`).
- **Size:** 16 layers × 2 maps × 768 × 64 × 4 B = 6.29 MB, so the file is essentially raw fp32.

## 3. Defaults actually served for Qwen

Every Qwen `lsa_flat_*` accuracy config leaves `model.indexer` unset (`lsa_flat_tp1.yaml:12-28` sets only
`attention: lsa`, `spare_contexts` and server flags; `_base.yaml:34-35` names the map). The values are therefore the dataclass
defaults `IndexerConfig` (`src/lsa/sglang/settings/design.py:172-209`). The README confirms that "every key of the block
defaults to the paper's flat design" (README.md:133), and the fixture generator asserts the values:

| knob | value | cite |
|---|---|---|
| `window_size` n_w (also the cluster block size) | **512** | design.py:173, notation.md:162 |
| `sink_size` | **4** | design.py:175 |
| `top_k` (long-range keys) | **4096** | design.py:177 |
| `top_p` (buckets picked) | **1024** | design.py:179 |
| `index_dim` d | **64** | design.py:181 |
| `pool_size` | **4** | design.py:183 |
| `bucket_size` (mean keys a bucket) | **8**, so **64 buckets a block** | design.py:185 |
| `bucket_cap` | **16** | design.py:187 |
| `top_level` | 0 (flat) | design.py:189 |
| `bucket_shape` | none (centroid score) | design.py:199 |
| `input` / `query_heads` | `key_and_query` / `all` | design.py:204-207, 242-243 |
| `early_budget` | **none** for Qwen (it is a GLM option) | design.py:209, notation.md:71 |
| list width | `top_k + window + sink` = **4612** | core/reference.py:225 |
| `PAGE_SIZE` (bucket-cache paging) | 8192 keys (1024 bucket rows) | design.py:30 |

Validation the port must keep (design.py:257-275): `bucket_cap ≥ 2·bucket_size`; `pool_size` divides `bucket_size`,
`bucket_cap` and `window_size`; `bucket_size` divides `window_size`; `top_p ≤ 2048`; `top_p·bucket_cap ≥ top_k`.
Variant `lsa_flat_k2048_tp1.yaml:11-14` uses `top_k 2048`, `top_p 512`. Distillation used `top_k 2048` (distill config
`:25`), but the server reads at 4096.

## 4. Algorithm (flat design), exactly

Notation: T keys in the context. A query at position p (0-based). `free(x)` = dims `64..255` of a head.
`normalize(v) = v / max(‖v‖₂, 1e-12)` (`core/index_space.py:315-334`).

### 4.1 Index key and index query (per full-attention layer ℓ)

The inputs are the layer's own q and k **after q_norm/k_norm and before RoPE**. On dims 64..255 these values equal the
post-RoPE ones, because RoPE does not touch them. The gate and v are not used.

```
k768_t = concat_{h=0..3} free(k_t[h])                       # [768], head-major (index_space.py:185)
q768_t = concat_{g=0..3} Σ_{j=0..5} free(q_t[6g+j])         # [768], group sums (index_space.py:176-182)
index_key_t   = normalize(k768_t @ W_K_idx[ℓ])              # [64] unit
index_query_t = normalize(q768_t @ W_Q_idx[ℓ])              # [64] unit
```

- Reference path (distillation, fixtures): `position_free` (`core/index_space.py:62-107`), then `regroup` (`:110-131`,
  which also unit-normalizes the 768-vector), then `project` (`:405-419`). The first normalization is a positive scalar,
  so it cancels in the second.
- Served path: `sglang/models/qwen3_5/index_space.py:260-289` sums the group in fp32 and casts keys to bf16.
  `KeyAndQueryProjection` (`sglang/index/layer_index.py:295-355`) applies **bf16 maps** with bf16 products:
  - key: `normalize((k768.bf16 @ W_K.bf16).f32).bf16`
  - query: `normalize((q768_f32.bf16 @ W_Q.bf16).f32)`
- Why one list serves all heads: Σ_h q_h·k_{h/6} = q768·k768 on the position-free dims (`index_space.py:115-118`).
  The selection is therefore **one list per query, shared by all 24 heads** (`core/attention.py:15-62` builds one mask per query).
- **The query's norm does not affect the selection.** A positive scale of `index_query` multiplies every bucket score
  equally, so the ranking stays the same. The kernels accept "index queries of any norm" (`core/kernels/scan_select.py:36`).
  The key normalization does matter, because centroids sum unit keys (§4.3).

### 4.2 Cluster blocks and pools

- Keys are cut into **cluster blocks of `window_size` = 512 consecutive positions**: block b = `[512b, 512b+512)`
  (notation.md:158-163). A block **closes** when its last key (position 512b+511) is in the cache. It is then clustered
  **once and never again**. In the flat design nothing ever re-clusters or merges across blocks. A trailing partial block
  belongs to no bucket (`core/reference.py:97`). Its keys are always inside the window of any query that can see them, as
  shown below.
- **Pool:** 4 consecutive index keys, **summed and not normalized**: `pool_key = Σ_{4 keys} index_key`
  (`core/ward.py:15-26`). That gives 128 pools a block, each of size 4. A pool is never split across buckets.

### 4.3 Ward clustering of one closed block (`core/ward.py:29-125`)

The 128 pools of a block are merged into exactly **m = 512/8 = 64 buckets**, each of at most `bucket_cap` = 16 keys
(4 pools). Since 128 → 64, every bucket holds 1 to 4 pools, and the mean bucket holds 2 pools (8 keys).

State: each live bucket b has a key sum s_b (starting at its pool key), a size |b| (starting at 4), and an alive flag.
Repeat for up to 256 rounds (`max_rounds`):

1. `merges_left = alive_count − 64`. Stop when it reaches 0 (`ward.py:61-63`).
2. Gram matrix G = S Sᵀ over the key sums (fp32). Alignment a_b = √G_bb. **Merge cost**
   `Δ(b,b') = a_b + a_b' − √(G_bb + G_b'b' + 2G_bb')` = ‖s_b‖ + ‖s_b'‖ − ‖s_b + s_b'‖, the "alignment the union loses"
   (`ward.py:64-68`). This is a spherical Ward cost (a norm loss), not the classic squared-distance Ward.
3. Allowed pairs: both alive, b ≠ b', and |b| + |b'| ≤ 16 (`:69-70`). Each b takes its cheapest allowed partner π(b).
   On equal cost, `min` keeps the **lowest index** (`:71`).
4. A pair merges when it is **mutual** (π(π(b)) = b) and b < π(b) (`:74`). If there are more mutual pairs than
   `merges_left`, only the `merges_left` cheapest merge (`:76-77`, rank by cost).
5. Each merge folds the higher index into the lower: s_b += s_π(b), |b| += |π(b)|, and π(b) dies (`:80-85`).
6. Labels: a pool follows its parents to its surviving bucket. Buckets are numbered 0..63 **in the order of their first
   (lowest) pool** (`:86-90`).

- `ward_merge` raises if a block stops short of 64. That cannot happen with cap ≥ 2·bucket_size (`:96-101`).
- There is no seed and no k-means pass: the merges are the partition.
- **Ties: the reference and the served kernel differ in one place, and QW follows the kernel.**
  - Partner choice. Both take the lowest index among equal-cost candidates: the reference through `min` (`ward.py:71`),
    the Triton kernel through the min of packed `(cost, column)` keys (`core/kernels/triton/ward.py:154-178`).
  - Final-round oversubscription. When more mutual pairs exist than `merges_left`, the reference ranks them with an
    **unstable** `argsort` of the cost (`ward.py:76`), so its choice among equal-cost pairs is implementation-defined.
    The kernel ranks each pair by its lower bucket's packed `(cost, partner)` key (`triton/ward.py:182-204`): an
    equal-cost tie goes to the pair whose **higher (absorbed) bucket has the lower index**.
  - Example (fixture `ward_tie_policy`, from the review): pools e1, e2, e2, e1, one key each, target 3 buckets, cap 4.
    Mutual pairs (0,3) and (1,2) both cost 0 and one merge is left. The kernel merges (1,2), giving labels
    `[0, 1, 1, 2]`. The CPU reference keeps (0,3).
  - **QW policy:** rank mutual pairs by `(cost, partner index)` ascending, as the kernel does. The generator implements
    that policy (`ward_labels_kernel_policy`) and asserts it gives the reference's labels for every other fixture
    case. Those cases (`tiny_ties` included) contain no oversubscribed equal-cost tie, so they cannot tell the policies
    apart. Only `ward_tie_policy` does.
- Upstream's serving path and kernels use the same rounds (`core/kernels/triton/ward.py:1-8`). The kernel's Gram is
  tf32x3 rather than fp32 (§5), so near-tie costs can also order differently between the two.

Per bucket (`core/reference.py:131-143`, served `sglang/index/clustering.py:80-130`):

- `bucket_centroid = normalize(Σ_{keys in bucket} index_key)`. This equals the normalized sum of its pool keys.
- `bucket_members` = the absolute positions of its keys, **ascending**, padded with −1 to 16, int32.
- `bucket_sizes` = number of members.
- Block b's buckets occupy rows `[64b, 64b+64)`.

### 4.4 Bucket cache and its growth

The flat cache is append-only, one per (layer, sequence) (`sglang/index/bucket_cache.py:97-106`):

| tensor | shape | dtype (served) | bytes per bucket |
|---|---|---|---|
| `bucket_centroids` | [rows, 64] | **bf16** (notation.md:243: "Bucket centroids are bf16 only") | 128 |
| `bucket_members` | [rows, 16] | int32 | 64 |
| `bucket_sizes` | [rows] | int32 | 4 |

- Each bucket row costs **196 B** and covers 8 tokens on average, which is **24.5 B/token/layer** and **392 B/token over the 16
  layers**. That comes to 12.3 MiB at 32k, 49 MiB at 128k and 98 MiB at 262k: about 2.4% of the Turbo4 KV cache and 0.6% of
  the FP16 one.
- Upstream pages the rows by `PAGE_SIZE` = 8192 keys (1024 rows) per page through a per-request page table (`:97-106`, `:577-578`).
  This matters only for multi-request pooling, and QW is single-sequence.
- **Per-token growth:** appending a key is free until it completes a block. On the key at 512b+511, the block's 512 index keys
  are pooled and Ward-clustered, and 64 rows are appended (`bucket_cache.py:507-523, 559-601`).
- Upstream re-projects those 512 index keys from the bf16 KV pool's keys at close time (`index_space.py:8-11`,
  `layer_index.py:494-507`), so it stores no per-token index state.
- **Index keys are needed until their block closes and is committed.** The flat design never scores an individual key:
  scoring uses centroids, and expansion uses member positions.
  - Without speculative decoding, a port needs the open block plus one forward's worth of keys a layer:
    (511 + chunk) × 64 × 2 B. For QW's 1536-row chunk that is (511 + 1536) × 128 B ≈ 256 KiB a layer, 4 MiB for
    16 layers. Decode alone needs 64 KiB a layer, 1 MiB.
  - With speculative verification (§7.3), a block that closes inside a verify forward may be rolled back, which
    reopens it, so **its index keys must be retained until the verify commits or rolls back**. In Turbo4 mode they
    cannot be recomputed exactly from the cache. Retain the last closed block's 512 keys beside the open block and the
    verify rows: ≤ (512 + 511 + rows) × 128 B a layer, about 2 MiB for 16 layers at 8 rows.
  - The tree design and prefix fingerprints read more, which is irrelevant here.
- **Visible buckets** for a query at p: `visible(p) = ⌊p / 512⌋ · 64`, the buckets of blocks closed *before the query's own
  block* (`bucket_cache.py:384-394`, `core/reference.py:262`). Decode uses the device copy of the cache's closed-key count
  (`bucket_cache.py:396-402`). Because the cache is updated before every forward (`selector.py:162-193`), it holds the same value.
- **Coverage invariant:** `⌊p/512⌋·512 ≥ p − 511 = window_start`. So every key outside the window lies in a closed block,
  and every key is reachable, either through the window or through a bucket.

### 4.5 Scoring, top-p and expansion (one query at position p)

Reference `core/reference.py:167-289`:

```
s_b       = index_query · bucket_centroid_b          for b < visible(p)        # fp32 dot (reference.py:228)
picked    = the top_p buckets by s_b, descending; equal s_b → lower b first      (stable argsort, :263-264)
win_start = max(0, p + 1 − 512)
expansion = [ m for b in picked (best first) for m in members(b) (ascending)
              if sink_size ≤ m < win_start ][: top_k]                           (:284-286)
window    = [win_start, …, p]                                                   (:287)
sinks     = [0, …, min(sink_size, win_start, p+1) − 1]                          (:288)
list      = expansion ++ window ++ sinks      (≤ 4612 ids, padded with −1; length returned)
```

- **Members in the window or the sinks are skipped and do not count toward top_k**, because they are read anyway.
- **Only the last bucket reached is cut short**, keeping its first members in position order.
- The walk stops at whichever comes first: `top_p` buckets or `top_k` keys. `top_p · cap ≥ top_k` is enforced, but with
  a mean of 8 keys a bucket, 1024 buckets usually hold about 8192 members. In practice `top_k` binds once the context is
  past about 4.6k.
- Duplicates cannot occur: buckets partition the closed keys, and the window and sinks are disjoint from the expansion.
- **Tie-breaking (exact score tie):** the lower bucket index wins. Kernels make this explicit with a packed int64 key
  `(ordered_bits(score) << 32) | (0x7FFFFFFF − bucket)` (`core/kernels/triton/keys.py:1-35`).
- Do not rely on MLX `argsort`/`argpartition` breaking ties stably. Pack the same key, or sort by (−score, bucket).
- **Threshold vs. rank order:** the served kernels run *expansion by threshold* (`core/kernels/triton/expansion.py:1-11, 396-430`).
  It finds the packed key at which the best-first walk stops, takes every bucket strictly above it whole and the stop bucket's
  first members up to the room left. It yields **the same set of positions** as the rank-order walk, but in listing order
  rather than score order. List order does not matter to attention, since it is a set. It only changes the fp32 summation
  order inside the attention kernel.
- **Early budget** (`budget.py:60-74`): if set, rows with p < `position` use a smaller `top_k`/`top_p`. Qwen does not set it.

### 4.6 Attention over the list

`core/attention.py:15-146`:

- The read mask is selection ∪ (q − k < window) ∪ (k < sink), then intersected with causality.
- Then the model's own softmax(q·kᵀ·scale)·v runs over the masked keys with the **rotated** q/k and unchanged scale,
  scores in fp32.
- At a budget ≥ T the mask is the dense causal mask, so the key set and the mathematics are dense attention
  (`attention.py:4-5`). Floating-point outputs match dense only to within kernel rounding (§4.7).
- The served read is FlashAttention's paged kernel at page size 1, with each query its own length-1 sequence and its list
  as its page table (`sglang/models/qwen3_5/backend.py:426-460`, `sglang/index/read.py:1-43`).
- The output gate, o_proj and everything else are unchanged.

### 4.7 Short contexts: the dense key set

A query reads every causal key iff `p + 1 ≤ sink + top_k + window = 4612` (all non-window, non-sink keys fit in top_k)
and `visible(p) ≤ top_p` (at p ≤ 4611, visible ≤ 9·64 = 576 ≤ 1024).

- **Key-set equivalence (exact):** every context up to 4,612 tokens selects exactly the dense causal key set
  (fixture `short_dense` asserts `selected == 0..p` for every row; `qwen_index_defaults` rows p ≤ 4611; upstream tests
  `tests/unit/core/test_reference.py:196, 230`). The read is therefore mathematically dense attention.
- **Numerical equivalence (within rounding, not bitwise):** the same keys do not imply the same kernel or reduction
  order. The list holds expansion, then window, then sinks, and a gathered or paged read sums in a different order
  than the dense kernel. The fixtures assert `allclose(attention_out, dense_out, 1e-5)`, not equality.
- **Bit identity with dense below the budget needs a dense fallback.** If QW wants it, route a full-attention forward
  whose rows all satisfy the condition above through the unchanged dense path. That is legal because the key set is
  identical. This is separate from the requirement that the flag-off path stay bit-identical, which holds trivially.
- The paper's RULER 4k score (0.974, README.md:182) is therefore essentially a dense run. Sparsity starts at the
  4,613th token.
- Upstream has no separate dense fallback for Qwen: every forward goes through the sparse read (`backend.py:783-821`),
  and a short context happens to select everything.

### 4.8 Prefill vs. decode

- **A query's list is a function of its position and the keys before it, and nothing else.** It does not depend on how
  the prefill is chunked or whether the row is prefill or decode. The fixture generator asserts that
  incremental-decode reads, prefill reads and one read over the final cache are identical.
- Prefill (`sglang/index/selector.py:195-261`): append the chunk's keys, close (cluster) every block the chunk completes,
  then read each row with `visible(p)`. Rows early in the chunk see fewer buckets.
- Decode: `maintain` closes a block when its last key is already in the pool. Then the step scores all `visible(p)`
  buckets (`selector.py:603-660`).
- A block that completes *inside* a forward is visible to the rows after it in the same forward. This matters for QW's
  1536-row prefill chunks and for multi-row MTP/DFlash verification (§7.3).
- Upstream refuses speculative decoding under LSA (`backend.py:26-28, 808-812`): "speculative decoding and a cache other
  than bf16 are refused".

### 4.9 RoPE / YaRN

The index reads only dims 64..255 of pre-RoPE heads. **It is independent of the RoPE settings** (theta, YaRN factor,
mrope sections). The generator asserts that an arbitrary rewrite of the rotated prefix leaves the index bit-identical.

QW's RoPE choice affects only the attention over the selected keys, as it does for dense. The one indirect coupling: the
map was **distilled against the teacher attention under YaRN factor 40** (`qwen38_27b.yaml:5-10`). Static YaRN rescales
the rotated part and adds an attention temperature, so the attention distribution the map imitates is not exactly the one
QW computes at native RoPE. This is a small accuracy risk, measured in task 6.

## 5. Reference vs. kernels numerics

| step | reference (`core/reference.py`, fixtures) | served Triton / CuTeDSL | exact? |
|---|---|---|---|
| index key | fp32 maps, fp32 products | bf16 maps × bf16 key → fp32 normalize → **stored bf16** (`layer_index.py:346-355`) | approximate (bf16) |
| index query | fp32 | fp32 group sum → **bf16** → × bf16 map → fp32 normalize (`:331-344`); cast to **bf16** again for scoring (`selector.py:403, 629`) | approximate |
| pooling / Ward | fp32 on the given keys (fp32 or bf16) | pools in fp32 from **bf16 index keys** (`clustering.py:101`); Gram in **tf32x3** by default (`core/kernels/ward.py:53`, `triton/ward.py:18-19`) | tf32x3 ≈ fp32; the merge order can flip on near-tie costs |
| centroids | fp32 | bf16 (`clustering.py:117-130`) | approximate |
| bucket scores | fp32 dot | bf16 q × bf16 centroid, fp32 accumulate. Decode: `scan_paged` (`triton/paged.py:1-4`) or CuTeDSL `DecodeSelect`; short prefill: one GEMM `out_dtype=f32` (`selector.py:409`) | sum order differs |
| top-p | stable argsort | exact top-k on packed (score, id) keys (`triton/top_k.py:1-22`). From 32,768 visible buckets (≥ 262,144 context) the fused scan-select samples a threshold, lists buckets within `MARGIN = 1e-4`·‖q‖ below it and **re-scores them per query in fixed-order fp32**, falling back to dense scoring on overflow or underflow (`triton/scan_select.py:1-25, 47, 61`) | exact given its scores; no approximate top-k |
| expansion | rank-order walk | threshold walk (`triton/expansion.py:396-430`) or CuTeDSL, "the same read either way" (`core/kernels/dispatch.py:1-33`) | same set, different order |
| attention | fp32 softmax over a mask | FlashAttention paged, bf16 | the usual kernel rounding |

- **Measured drift (fixture `qwen_layer`, real layer-3 map):** the served numerics change **2 of 260** lists against the fp32
  reference (min Jaccard 0.94, mean 0.9995). No Ward bucket changed (`manifest.json`
  `diagnostics.serving_numerics_drift_real_map`).
- The smallest top-p boundary gap there is 3.3e-6. A port should score in fp32 for parity tests and must treat a
  near-tie flip as a numerics difference, not a bug.
- Nothing in the selection path is an approximate-by-design top-k. All approximation is bf16 rounding (and tf32x3 in Ward).
- The CuTeDSL kernels are enabled only on H200/B300 (`dispatch.py:46-60`). They are irrelevant to QW, which has no CUDA.
- The reference itself has one edge case: `build_buckets` raises on a cache with no closed block (`ward.py:115` reduces an
  empty tensor). Serving never calls it that way. A port should simply return no buckets.

## 6. Costs

Sizes: d = 64, 16 layers, Hq·dh = 6144, 768-dim index input.

| item | per token per layer | 16 layers | at 32k / 128k / 262k |
|---|---|---|---|
| index key/query projection | 768×64 MAC each (~0.2 MFLOP), plus a 24→4 group sum | ~3.1 MFLOP | negligible next to the 27B linears (~54 GFLOP/token) |
| Ward at block close | 128-pool Gram (128²×64, ~2 MFLOP) per round, 3-4 rounds per 512-key block (measured on the `qwen_index_defaults` blocks) | ~0.3 MFLOP amortized | negligible |
| bucket cache (persistent) | 24.5 B | 392 B | 12 / 49 / 98 MiB |
| buffered index keys (open block, chunk in flight, last closed block until commit) | ≤ (512 + 511 + chunk) × 128 B per layer | 1 MiB decode only; ~2 MiB with 8-row verify; ~4-6 MiB during a 1536-row prefill chunk | constant |
| decode: centroid scan | reads T/8 × 128 B | 256 B × T | 8 / 32 / 64 MiB per step |
| decode: gathered KV (4612 keys) | FP16 4 KiB/key, Turbo4 ~1 KiB/key | | 288 MiB (FP16), ~74 MiB (Turbo4) per step, constant |
| prefill: scoring | each query × T/8 centroids × 64 | ≈ 256·T FLOP/query | ≈ 0.13% of dense attention FLOPs |

Dense attention for comparison: KV read per decode step = 64 KiB·T (FP16) or 16.4 KiB·T (Turbo4), which is 2 / 8 / 16 GiB
(FP16) and 0.5 / 2 / 4.1 GiB (Turbo4) at 32k / 128k / 262k. Causal prefill attention ≈ 196,608·T FLOP/token over the 16 layers,
against about 5.4·10¹⁰ FLOP/token of linears. Its FLOP share is therefore T/(T+275k): 11% at 32k, 19% at 64k, 32% at 128k,
49% at 262k.

### Feasibility verdict

**Feasible, cheap in memory, and only worthwhile at 128k-262k.**

- **Upper bounds on this Mac** (M4 Max, about 16 GB of 4-bit weights read per decode token). These assume ideal sparse kernels:

  | context | decode, FP16 KV | decode, Turbo4 KV (QW default) | prefill (FLOPs) |
  |---|---|---|---|
  | 32k | ≤ 1.11× | ≤ 1.03× | ≤ 1.08× |
  | 64k | ≤ 1.24× | ≤ 1.06× | ≤ 1.20× |
  | 128k | ≤ 1.50× | ≤ 1.13× | ≤ 1.43× |
  | 262k | ≤ 2.0× | ≤ 1.26× | ≤ 1.9× |

  These are byte ratios for decode, (weights + dense KV) / (weights + gathered 4,612-key KV + centroid scan), and the
  ideal FLOP ratio for prefill. GDN state and activations are ignored.

  - QW's D=256 multi-row attention materializes scores (integration notes §4), so dense prefill attention is probably
    *less* efficient than its FLOP share suggests. The real prefill share should be measured first, and it raises the
    prefill ceiling.
  - Gathered (non-contiguous) KV reads and a per-query list on Metal will eat into these bounds.
- **Below ~64k it is close to pointless.** At most a few percent remain after overheads. Within 4,612 tokens it selects
  the dense key set (equal to dense within rounding, §4.7).
- The headline 72.4× / 42.5× (README.md:21) is **one layer at 10M tokens on an H200**. QW's 262k native context never
  reaches that regime.
- Accuracy: the paper reports RULER 0.974 at 4k (the dense key set) and 0.968 at 32k (README.md:182, bf16, H200).
  QW must re-measure on the 4-bit oQ4e checkpoint at native RoPE.

## 7. QW integration

This section draws on the researcher's integration notes, checked against the code at the points cited.

### 7.1 Hook points

1. **Where q/k are available pre-RoPE.**
   - `Qwen3NextAttention::forward_impl` (`crates/runtime/src/qwen3_next.rs:168-288`) is the single path for prefill
     chunks, decode, MTP verify and DFlash2 verify.
   - After `q_norm`/`k_norm` (`:204-205`), `queries [B,L,24,256]` and `keys [B,L,4,256]` are exactly the upstream inputs.
   - Even after RoPE (`:212-253`, `fast_rope(.., 64, traditional=false, ..)` or the mrope branch on `[..., :64]`), dims
     64..255 are untouched. So the index can be computed either before or after the rotation.
   - Head-major concat with group g = heads 6g..6g+5 matches upstream (`index_space.py:176-185`).
2. **Index keys must come from the exact K at write time, never from the cache.**
   - QW's default cache is **Turbo4**: a WHT over all 256 dims, then 4-bit (`cache.rs:1438-1520`, `turbo/quant.rs`). The 192
     position-free dims are mixed with the rotated ones and only recoverable lossily.
   - So, unlike upstream, which re-projects from the bf16 pool, QW computes `index_key_t` inside `forward_impl` from the
     post-norm K (`:205`) for every new token. It keeps index keys in a small per-layer buffer until their block closes
     and, under speculative verification, until the closing forward commits (§4.4, §7.3):
     ≤ (512 + 511 + rows in flight) × 64 × (f32 or bf16).
   - In Fp16 mode the pool *could* serve as the source, but the buffer is cheaper and mode-independent.
3. **Index state next to `KVCache`.** One per full-attention layer: the bucket cache (§4.4), the open-block key buffer,
   and the retained keys of a block closed by an uncommitted forward. Operations: append, close block (Ward), commit,
   trim (rollback, which can reopen a block), reset, nbytes, snapshot/restore.
   - Turbo4 demotion (`demote_fp16_to_turbo4`, `cache.rs:3591-3605`) does not touch it, because the index never reads the KV.
4. **Selection.** Per query row: score `visible(p)` centroids, pack (score, bucket) keys, top-p, expand, add window and sinks
   (§4.5). Positions are **logical token positions**.
   - Cache slots must equal positions. (SpecPrefill, a former QW sparse-prefill mode that kept keys at slots ≠ positions
     and so could not be combined with D-LSA, has since been removed from QW.)
5. **Restricted attention.**
   - Fp16 decode: `take` the list rows of K/V `[1,4,T,256]` → `[1,4,4612,256]`, then plain SDPA (no mask, since all rows
     are causal), or reuse `paged_v2::run_sparse_decode` (`mlxcel-core/src/paged_v2/sparse.rs`). That kernel needs q_len 1,
     a uniform count per head (the list ×4) and has an untested D=256 path.
   - Turbo4 decode: gather the packed rows (`k_packed`, `k_rescale`, `v_packed`, `v_rescale`), then the fused
     `turbo4_attention` (S > 2048) or `attention_turbo4_dequant_sdpa`.
   - Prefill: parity can use an additive mask `[L, T]` over the dense path. That is exact but no faster, because D=256
     materializes scores anyway. A speedup needs a new Metal sparse-prefill kernel (task 4).
   - The current chunk's K must be in the cache before the read. `update_and_fetch` already does that in Fp16. The Turbo4
     fused update+attend calls need splitting.
6. **Feature gating**:
   - A cargo feature `dlsa` in `qw-runtime`, forwarded by the server/cli crates.
   - Code under `#[cfg(any(feature = "dlsa", test))]`, a server flag stub when the feature is off, and an off-by-default
     runtime switch.
   - With the feature off, or the switch off, the dense path is untouched (bit-identical).

### 7.2 What changes from the integration notes

- The notes assumed a 2048-dim index key, costed at 64 KiB/token. **With d = 64 a stored index key is 128 B/token/layer =
  2 KiB/token** (3% of FP16 KV, 12.5% of Turbo4). The flat design buffers it only for the open block, the chunk in
  flight and, under speculative verification, the last closed block until commit (1-6 MiB in total, §6).
  Persistent per-token state is the bucket cache, at 392 B/token.
- **The proposed factored score `(W_K q̂)·k768/‖k768 W_K‖` does not apply.** The upstream score is a plain dot product of
  normalized vectors, but it is taken against **bucket centroids**, not individual keys: `s_b = q̂·c_b` with
  `c_b = normalize(Σ_{j∈b} normalize(k768_j W_K))` (`reference.py:134-136, 228`). No per-key score exists in the flat design.
  The per-key normalization inside the sum is nonlinear across keys, so a centroid cannot be rebuilt from cached K plus a
  norm without re-projecting each member. Nothing needs it anyway, because centroids are computed once at block close from
  the buffered index keys.
- Recall-vs-reference metrics for Turbo4 remain useful for the *attention* read. The selection is identical in both cache
  modes, because it is computed from the exact K at write time.

### 7.3 Multi-row verification (MTP 3 rows, DFlash2 8 rows), chunked prefill, rollback

- Each row reads at its own position (§4.8). If a block completes inside the verify block, the later rows must see it
  clustered **including the draft tokens' keys**. Upstream semantics require that.
  - Treating the closure as deferred is *not* equivalent: up to (rows − 1) keys of that block fall out of every row's
    window and become unreachable.
  - **Rollback can reopen a closed block.** Example: the open block ends at 512b+511, a verify forward appends rows
    512b+509 .. 512b+516 and closes the block at 512b+511, and only rows up to 512b+510 are accepted. The rollback must:
    - drop the block's 64 bucket rows,
    - restore the accepted keys [512b, 512b+510] as the open buffer, and
    - drop the rejected keys, all in lockstep with the KV trim (`qwen3_5.rs:1641-1657`).
  - Restoring needs the **index keys of the closed block, retained until the verify commits or rolls back**. In Turbo4
    mode they cannot be recomputed exactly from the cache, and in Fp16 mode recomputing them is possible but wasteful.
    Keep the last closed block's 512 keys (or a pre-verify copy of the open buffer) until commit. The cost is bounded:
    ≤ (512 + 511 + rows) × 128 B a layer, about 2 MiB for 16 layers at 8 verify rows (§4.4).
  - Re-clustering happens when the block closes again with the accepted keys plus new ones. That is cheap and exact,
    because Ward is deterministic given the keys.
  - Committing frees the retained keys.
- The MTP draft layer has its own FP16 cache (`qwen3_5_mtp.rs:259-280`). There is no map for it, so it stays dense.
- Upstream never ran D-LSA with speculative decoding. Its interaction with acceptance rates is unmeasured.

### 7.4 Snapshots / prefix cache

- `push_attention_snapshot` (`qwen3_5.rs:2538-2573`) pages tensors on the token axis in 256-token pages. Restore validates
  an exact name set (`:2603-2648`).
- The bucket cache's row axis is per block (64 rows per 512 tokens), and a half-closed block has no rows. So it does not page
  on a 256-token boundary.
- Two exact options:
  - (a) Store per-token index keys `layer.{i}.dlsa_index_keys [1,1,T,64]` bf16, token-paged like K (2 KiB/token over
    16 layers). Rebuild buckets on restore: Ward is deterministic given the keys, so the rebuild is bit-reproducible.
  - (b) Store the bucket tables densely (`push_tensor`) plus the open-block keys.
- (a) is simpler and reuses pages. The index-key projection must then be bit-stable across chunk sizes, or the stored keys
  must be authoritative. Upstream notes that "a projection rounds by the size of its batch" (`layer_index.py:589-590`).
- Old snapshots without the names fail validation and act as a cache miss. No versioning is needed (AGENTS.md).

## 8. Risks

1. **Little gain below ~128k on a Mac, and less with the default Turbo4 cache** (§6). The main risk is building it for
   nothing. Measure first.
2. **Sparse prefill needs a new Metal kernel.** Lists are per query (shared across heads), length 4612, with heavy overlap
   between neighbouring rows (window and shared buckets). Gather-per-row is infeasible at 1536-row chunks
   (6.4 GB at S = 2K, notes §4).
3. **4-bit checkpoint:** the maps were distilled on bf16 q/k. The oQ4e projections shift q/k, which can lower recall.
   Compare the distilled and PCA maps.
4. **Distilled under YaRN 40, served at native RoPE** (§4.9). Low to moderate risk.
5. **Speculative decoding** (MTP/DFlash): exact semantics need in-forward block closure, keys retained until commit, and
   a rollback that can reopen a block (§7.3). Upstream never validated it.
6. **Numerics:** near-tie flips (3.3e-6 gaps observed) make bit-exact parity with the reference fragile in fp16/bf16. Tests
   must be stage-wise from fixture inputs, and end-to-end comparisons need overlap metrics.
7. **MLX sort stability and tie rules** (§4.5) must be enforced explicitly.
8. **Map container:** the `.pt` is a torch pickle and there are no sidecars. A conversion step is required, and it must be
   pinned by sha256.
9. **Snapshot complexity** (§7.4).
10. **Accuracy evidence is upstream's.** RULER 0.968 at 32k is on H200 bf16 with YaRN 40. Nothing is verified on QW yet.
    (Since measured on QW at 4k and 32k: §16.)

## 9. Roadmap refinement

0. **Spec + fixtures** (this task). Done: `docs/dlsa.md`, `tests/fixtures/dlsa/`.
1. **Map conversion / loader.**
   - An offline script (same style as `gen_fixtures.py`) converts each `.pt` to safetensors: tensors
     `layers.<i>.w_q_idx` / `w_k_idx`, `[768,64]` F32, with metadata (source sha256, upstream commit, index_dim, layers).
   - Verify both source sha256 values (§2), plus the output sha256.
   - Rust loader in `qw-runtime` behind feature `dlsa`: validate the shapes, that layers == the full-attention layers,
     and the dims, mirroring upstream `weights.py:120-162`.
   - Decide where the converted file lives (user cache path plus a sha check; do not commit 6 MB per map).
   - Also convert the PCA map for task 6.
2. **Indexer core** (pure MLX ops, no Metal, no model). **Done, see §11** (Ward, top-p and expansion run on the host):
   - index key/query
   - pooling
   - Ward rounds (an MLX port of `ward.py:29-90` with the kernel's tie policy, §4.3; the `ward_tie_policy` fixture pins it)
   - bucket cache (append, close, commit, trim with block reopening from retained keys, reset)
   - visible counts
   - packed-key top-p
   - expansion with sink/window
   - Unit tests against every fixture stage (`tiny`, `tiny_ties`, `ward_tie_policy`, `short_dense`, `qwen_layer`,
     `qwen_index_defaults`),
     stage-wise.
3. **Sparse attention integration**, under the `dlsa` feature and a runtime flag, default off:
   - The `forward_impl` hook (§7.1), index state beside `KVCache`, and the index-key buffer from exact K.
   - Short-context policy: either accept numerical (not bitwise) dense equivalence below 4,612 tokens, or route those
     forwards through the unchanged dense path for bit identity (§4.7).
   - Fp16 decode via gather + SDPA, Turbo4 decode via packed gather. Prefill via mask (parity only).
   - Refuse MTP/DFlash when `dlsa` is on, as upstream does.
   - `qwen_layer` attention parity tests, an `#[ignore]` real-checkpoint parity test (Fp16), and a dense bit-identity test
     with the flag off.
   - **Add an early go/no-go microbenchmark**: decode gather+SDPA vs dense at 128k/262k in Fp16 and Turbo4, before task 4.
4. **Metal kernels:**
   - a sparse-prefill attention kernel over per-query lists (the main speedup)
   - a fused centroid scan + packed top-p
   - gathered decode attention (or extend `paged_v2` sparse to D=256)
   - optionally Ward on the GPU
   - Benchmark under `./gpu-lock`.
5. **Serving integration:**
   - snapshot/prefix-cache state (§7.4, option a)
   - exact MTP/DFlash support (§7.3: in-forward closure, keys retained until commit, rollback that reopens a block) or a
     documented exclusion
   - tests for a verify block that crosses a 512-key boundary and is partly rejected: after rollback and re-decode, the
     bucket cache and lists must equal a non-speculative run, in both Fp16 and Turbo4
   - CLI/server flags
   - README/DEVELOPMENT docs
6. **Evaluation & decision** on oQ4e at native RoPE:
   - RULER subset at 32k/128k/262k against dense, for the distilled and PCA maps
   - Fp16 vs Turbo4
   - prefill and decode throughput
   - Turn on by default only above a measured context threshold (expected ≥ 128k), and only if accuracy is within a
     preset tolerance of dense. Otherwise keep it as an opt-in or drop it.

## Appendix: verification of the brief's "known facts"

| claim | verdict |
|---|---|
| 16 full-attention layers, every 4th | ✓ (3, 7, …, 63, 0-based) |
| 24 q / 4 kv heads, head_dim 256, 64 rotated dims | ✓ |
| **d = 2048** | ✗ **d = 64**. "d2048" is the target depth |
| one shared index list per query for all heads | ✓ (`attention.py:15-62`) |
| RULER 0.974 @ 4k, 0.968 @ 32k | ✓ as reported (README.md:182; the paper's columns say 0.975 / 0.966). At 4k every row reads the dense key set |
| speedups large only at very long context | ✓ (72.4× / 42.5× is one layer at 10M, README.md:21) |
| `.json` sidecars | ✗ not published |
| index taken pre-RoPE, post-QK-norm, last 192 dims, gate unused | ✓ (`qwen3_5.py:115-123`, `index_space.py:8-11`) |

## 10. Map files, download and loader (roadmap task 1; feature `dlsa`)

- **Decision (container).** The released `.pt` is a torch zip with a pickle. `qw-runtime` reads it with a minimal built-in
  reader (`crates/runtime/src/dlsa/torch_pt.rs`: stored zip entries plus a whitelist pickle VM that accepts only
  `_rebuild_tensor_v2` over `FloatStorage` and `OrderedDict`; it never executes anything). No new dependency beyond `sha2`
  (already in the lockfile), optional under `dlsa`. So `qw download` needs neither Python nor torch.
- **Decision (converted format).** safetensors, since QW already uses it for weights and `safetensors` is a dependency.
  Tensors `layers.<i>.w_q_idx` / `layers.<i>.w_k_idx`, **F32** `[768, 64]` (lossless; the loader casts to bf16, which is what
  upstream serves, or keeps F32 for reference runs: `DlsaMapDtype`). One metadata key `qw_dlsa_map` holds a JSON object
  (`format`, `source_asset`, `source_sha256`, `upstream_commit`, `index_dim`, `input_dim`, `layers`); a single key keeps the
  output byte-deterministic (safetensors writes multi-key metadata in hash order).
- **Hosting.** Nothing is committed or mirrored. `qw download --dlsa-map [--dlsa-pca]` (with no model identifier it skips
  the model) downloads the upstream release asset with `curl` (URL `https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/<asset>`),
  checks its size and sha256, converts it in Rust, checks the converted sha256 and writes it atomically to
  `~/.cache/qw/dlsa/<asset>.safetensors` (the `.pt` stays beside it). Pins live in `dlsa/resolve.rs`:

  | map | source sha256 (`.pt`) | converted sha256 (`.safetensors`, 6,294,584 B) |
  |---|---|---|
  | distilled (served) | `24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b` | `00200004e8ab6cadb2738dbc466230b542fecc592e138be6c31f29d16e0462e1` |
  | PCA step-0 | `baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65` | `03dbc21117cb96c6c18f13bc4eacb6c44eb5733e9787d5c5b127ab365362f9fb` |

- **Resolution** (`resolve_dlsa_map_path(kind, cli_override)`): CLI path, then `QW_DLSA_MAP_PATH`, then the cache file. Only the
  cache file is checked against its pinned sha256; an explicit override is trusted but structurally validated by the loader.
  Errors name the missing path and the command to run, or the expected and actual checksum.
- **Loader** (`DlsaMaps::load(path, &DlsaLayout, DlsaMapDtype)`): `DlsaLayout::from_qwen35_config` derives the full-attention
  layer ids and `input_dim = kv_heads * (head_dim - rotary_dim)` from the checkpoint config (3, 7, ..., 63 and 768 for
  Qwen3.8-27B). It errors if the file's layers differ from that set (missing or unexpected), a tensor is not F32
  `[input_dim, 64]`, or the file is not a QW map.
- **Independent check.** `tests/fixtures/dlsa/verify_map_conversion.py` (stdlib only, no torch) compares every converted tensor
  byte-for-byte with the `.pt`. Both real maps pass.
- **Tests.** `cargo test -p qw-runtime --features dlsa dlsa` (synthetic archive, layout/shape/checksum errors). The
  `#[ignore]` test `real_maps_convert_to_pinned_checksums_and_load` converts the real files (from `~/.cache/qw/dlsa` or
  `QW_DLSA_SOURCE_DIR`), checks the pinned checksums and loads all 16 layers.
- **Rope note for later tasks.** The loader is independent of rope scaling: the index reads dims 64..255, which RoPE (YaRN
  included) never touches (§1), so QW's lack of YaRN does not change what the maps read.

## 11. Indexer core (roadmap task 2; feature `dlsa`)

`crates/runtime/src/dlsa/`, re-exported from `qw_runtime`. No model hook yet (task 3), no Metal kernels (task 4).
Projection and centroid scoring are MLX ops (f32 matmul). Ward, top-p and expansion run on the host in f32 with explicit
comparators, so the tie rules are exact and readable.

### 11.1 Modules

| file | contents | ported from (`@5ebea73`) |
|---|---|---|
| `config.rs` | `DlsaIndexConfig` (+ `QWEN_DEFAULTS`, `validate()` = `design.py:257-275`, `buckets_per_block`, `pools_per_block`, `list_width`, `visible_buckets`), `DlsaGeometry` (+ `QWEN38_27B`) | `sglang/settings/design.py` |
| `project.rs` | `concat_keys`, `group_queries`, `index_keys`, `index_queries`, `normalize_rows`, `to_host_f32` | `core/index_space.py` |
| `ward.rs` | `pool_keys`, `ward_labels` (kernel tie policy), `WardOutcome` | `core/ward.py`, `core/kernels/triton/ward.py:154-204` |
| `buckets.rs` | `cluster_block` -> `BlockBuckets` (centroids, members, sizes of one closed block) | `core/reference.py` `build_buckets` |
| `cache.rs` | `DlsaBucketCache` (append, commit, trim, reset, memory, snapshot/restore, build), `DlsaIndexSnapshot` | `core/reference.py`, `sglang/index/bucket_cache.py` |
| `select.rs` | `top_p_buckets`, `expand_buckets`, `score_buckets_host`, `DlsaSelection`, `DlsaBucketCache::{select, select_host, score_mlx, select_from_scores}` | `core/reference.py` `read`, `_read_row`, `expand_buckets` |
| `fixtures.rs`, `indexer_tests.rs` | test-only fixture reader (sha256-checked against `manifest.json`) and the tests of §11.6 | - |

### 11.2 API for the integration (tasks 3 and 5)

```rust
// One per (full-attention layer, sequence).
let mut cache = DlsaBucketCache::new(DlsaIndexConfig::QWEN_DEFAULTS)?;
// In forward_impl, after q_norm/k_norm, before or after RoPE (dims 64..255 are untouched):
let ik = index_keys(&keys, &maps.w_k_idx, &DlsaGeometry::QWEN38_27B)?;    // [L, 64] f32 unit rows
let iq = index_queries(&queries, &maps.w_q_idx, &DlsaGeometry::QWEN38_27B)?; // [L, 64] f32
cache.append_mlx(&ik)?;                  // all L keys first: closes every block they complete
let sel = cache.select(&iq, &positions)?; // one list per row, shared by all heads
let lists = sel.lists_mlx();             // [L, 4612] i32, -1 padded; sel.lengths: valid entries
cache.commit();                          // or cache.trim(accepted_len)? on a rejected verify
```

- Inputs are `[1, L, H, D]` (the `forward_impl` layout) or `[L, H, D]`, any float dtype. Keys are cast to the map's dtype
  for the product: F32 maps reproduce the reference (parity), BF16 maps the served numerics (§4.1). F16 maps are refused.
- `select` requires every position `< len()`: the forward's own keys must be appended first. Causality is then
  structural (a row only scores the `visible(p)` buckets of blocks before its own; `positions` past `len` are an error).
- `DlsaSelection { rows, width, top_p, lists, lengths, picked, visible }`: `lists` is row-major `[rows, width]` in the
  reference's order (expansion, window, sinks); `picked` the picked bucket rows best first (`-1` padded); `list(r)` the
  valid part of row `r`.
- `select_host` (sequential f32 dots) is the oracle of the MLX path; on every fixture both give identical lists.
  `score_mlx` + `select_from_scores` split scoring from the pick for benchmarks. The host pick fans out over threads once
  `rows x visible >= 65536`.
- `memory()` -> `DlsaCacheMemory { host_tables, host_tables_allocated, tail_keys, tail_keys_allocated, device_centroids }`:
  logical sizes (`len`) and host allocations (`capacity`); `total()` is logical, `allocated()` (= `nbytes()`) what the
  cache actually holds. `stats()` counts blocks, Ward rounds and clustering time.

### 11.3 Tie policies as implemented

- **Ward partner:** the cheapest allowed partner, the **lowest index** on an exact cost tie (strict `<` over ascending j).
  The Gram is computed once per unordered pair and mirrored, so `cost(i, j) == cost(j, i)` bitwise.
- **Ward final round:** mutual pairs are ranked by `(cost, partner index)` ascending and only the first `merges_left`
  merge: the served kernel's policy (§4.3). `ward_tie_policy` gives `[0, 1, 1, 2]`, not the reference's
  implementation-defined `[0, 1, 2, 0]`.
- **Top-p:** `(score descending, bucket ascending)` with `partial_cmp`, so `-0.0 == +0.0` as in torch (packed ordered bits
  would not treat them as equal; QW never packs). NaN scores are an error. `select_nth_unstable_by` plus a sort of the
  prefix: the comparator is a total order, so unstable sorting is exact.
- **Expansion:** the rank-order walk of the reference (not the kernels' threshold walk: same set, different order).
- **Centroid:** the bucket's keys summed in position order (as `index_add_`), divided by `max(norm, 1e-30)` as
  `build_buckets` does (not the `1e-12` of `normalize`; irrelevant for non-zero sums). Pools and centroids of BF16 keys
  are summed from the keys upcast to f32, as upstream does.

### 11.4 Incremental semantics, commit and rollback

- `append` clusters a block as soon as its last key arrives, also mid-append, so a row later in the same forward sees it
  (§4.8, §7.3). The cache always holds `floor(len / 512)` closed blocks; the reference's `build_buckets` over the same
  keys gives bit-identical tables, whatever the chunking (tested).
- The cache retains the index keys of `[tail_start, len)`, where `tail_start = 512 * floor(committed / 512)`.
  `commit()` moves the commit point to `len` and frees the keys of blocks closed before it, and shrinks the key buffer's
  capacity to at most two blocks (`2 x 512 x 64 x 4 B` = 256 KiB a layer), so a prefill chunk or a batch build does not
  stay allocated. Without speculation call it after every forward: the retained keys are then the open block only
  (`< 512 x 256 B` a layer).
- `trim(n)` rolls back to the first `n` keys for any `tail_start <= n <= len`: it drops the bucket rows of every block
  that ends past `n` (reopening it), truncates the keys and the commit point, and drops the device copy of the centroids
  if rows were removed. A reopened block closes again from the retained keys plus the new ones, which is exact because
  Ward is deterministic given the keys. `n < tail_start` is an error. So a verify forward appends its rows (possibly
  closing a block), selects, then the caller either `commit()`s or `trim(accepted)`s and `commit()`s.
- `reset()` empties the cache and releases every allocation. An empty cache has no buckets; a query then reads its
  window and sinks (the reference raises in `build_buckets` there, §5).

### 11.5 Snapshot layout (`DlsaIndexSnapshot`, for task 5)

| field | type, shape | meaning |
|---|---|---|
| `cfg` | `DlsaIndexConfig` | must equal the restoring cache's config |
| `len` | usize | keys appended (positions `0..len`) |
| `centroids` | f32 `[rows, 64]`, `rows = 64 * floor(len / 512)` | bucket centroids, block after block |
| `members` | i32 `[rows, 16]` | absolute member positions ascending, `-1` padded |
| `sizes` | i32 `[rows]` | members per bucket |
| `open_keys` | f32 `[len - 512 * floor(len / 512), 64]` | index keys of the open block |

`restore(cfg, &snap)` checks the config, every length and the member range, and gives a committed cache (it cannot roll
back past its open block; an uncommitted state is captured as if committed). No version field (AGENTS.md). Task 5 may
instead store per-token index keys and rebuild with `DlsaBucketCache::build` (§7.4 option a): both are exact, because
Ward is deterministic given the keys. Option (a) then needs stable stored keys: the projection's rounding depends on its
batch size (§7.4), which changes centroids by `<= 1.5e-7` on the fixtures and no list (test
`projection_batch_size_does_not_change_lists`), but is not bitwise.

### 11.6 Tests and tolerances

`./gpu-lock -- cargo test -p qw-runtime --features dlsa dlsa::` (MLX tests need Metal). Every stage is tested from the
fixture's own inputs of that stage, then end to end. Set `DLSA_TEST_VERBOSE=1` to print every measured difference.

| stage | fixtures | tolerance | measured max |
|---|---|---|---|
| `concat_keys`, `group_queries` from `k` / `q` (`q_rows`) | tiny, tiny_ties, short_dense, qwen_layer | 1e-5 | 0 |
| `index_keys`, `index_queries` (MLX, F32 maps) | the same, both qwen_layer maps | 1e-5 | 4.9e-7 |
| `pool_keys` from fixture `index_keys` | all list cases | 1e-6 | 0 |
| Ward labels from fixture `pool_keys`, and from own pools (incl. `qwen_index_defaults`, 18 blocks of 128 pools from BF16 keys); `ward_tie_policy` | all | exact | exact |
| centroids / members / sizes via the cache (chunks of 7, then decode) | all list cases + qwen_index_defaults | 1e-6 / exact / exact | 1.8e-7 |
| `visible_buckets` | all | exact | exact |
| bucket scores (MLX and host) vs fixture `bucket_scores` | all | 1e-6 | 1.2e-7 |
| picks from fixture `bucket_scores` (tiny_ties: exact score ties) | all | exact | exact |
| expansion from fixture picks + members, in order | all | exact | exact |
| lists end to end from fixture index vectors (MLX == host, chunks of `2W+3`, then decode; qwen_index_defaults row by row) | all | exact, in order | exact |
| lists end to end from raw q/k through `project.rs` and MLX selection | tiny, tiny_ties, short_dense, qwen_layer x2 | exact, in order | exact |
| attention over the lists (fp64 CPU softmax, identity RoPE) vs `attention_out` | tiny, tiny_ties, short_dense | 1e-5 | 6.0e-7 |

Why these tolerances: f32 sums of 8-64 terms in a different order differ by a few ulp of values of order 1 (`<= 5e-7`
measured), so 1e-6 / 1e-5 leave margin without hiding a wrong formula (which errs by `>= 1e-3`). Index tables and lists
are exact: the smallest top-p gap in the fixtures is 3.3e-6 (`real`), an order above the scoring error, and no Ward cost
is that close; a flip would be a bug, not rounding. The bf16-map sanity test allows 3e-2 (bf16 products, measured 1.8e-3).

Property and equivalence tests (host paths, seeded splitmix64, 60-80 random valid configs each, a third of them with
`{-1, 0, 1}`-quantized vectors full of exact ties): every list against an independently written oracle (full stable sort,
flatten-filter-truncate) and its invariants (length and padding, no duplicates, ids in `[0, p]`, window and sinks
present and ordered, expansion only from closed blocks below the window, expansion `== min(top_k, offered)`, short
context reads every causal key); Ward partitions (exactly `m` buckets, cap, pools whole, numbered by first pool,
deterministic); token-by-token, random-chunk (crossing 0-2 block boundaries) and batch schedules give bit-identical
tables and identical lists; an 8-row verify crossing a block boundary (rows from 48 on see the new block); rollback
that reopens a block equals a fresh build and stays equal when decoding on; trim limits and commit freeing; allocation
accounting (after a batch build + commit the key buffer is at most two blocks, the tables exactly sized; decode stays
bounded; reset releases everything and reports 0); snapshot ->
restore -> continue equals an uninterrupted run; the MLX device copy follows growth, trim and restore; Qwen defaults at
6000 keys.

### 11.7 Unoptimized baseline (one layer, for task 4)

`crates/runtime/examples/dlsa_indexer_bench.rs`: one full-attention layer at Qwen geometry and the served defaults,
synthetic topical bf16 q/k (as `gen_fixtures.py` makes them), the real layer-3 distilled map (from the `qwen_layer`
fixture when no converted map is installed) in bf16. Prefill: 1536-row chunks, each projected, appended (closing blocks
with Ward), and **every row** selected (MLX scoring, host pick), then committed. Decode: 25 steps of 1 / 3 / 8 rows at
context T (project, append, select, then `trim(T)` as a rejected verify would), median.

```text
./gpu-lock -- cargo run --release -p qw-runtime --features dlsa --example dlsa_indexer_bench -- \
  --contexts 32768,131072 --decode-steps 25 --out "$AGENT_SCRATCH/dlsa_indexer_bench_<sha>_<ts>.json"
```

Apple M4 Max (16 cores, 64 GB), base `3f3c6d1` + this change, release build, 2026-11-05:

| measurement (one layer) | T = 32,768 | T = 131,072 |
|---|---|---|
| prefill indexer, tokens/s (x16 layers) | 111,400 (6,960) | 57,400 (3,590) |
| of which: projection / append+Ward / MLX scoring / host pick+expansion (s) | 0.04 / 0.02 / 0.04 / 0.19 | 0.15 / 0.08 / 0.79 / 1.26 |
| Ward per closed block: mean rounds, ms | 3.13, 0.28 | 3.18, 0.28 |
| decode step, 1 row, median ms (project / score / pick) | 0.66 (0.42 / 0.17 / 0.08) | 1.35 (0.80 / 0.21 / 0.15) |
| decode step, 3 rows (MTP), median ms | 0.72 | 0.95 |
| decode step, 8 rows (DFlash), median ms | 1.06 | 0.86 |
| host tables in use (f32 centroids + i32 members + sizes), B/token/layer | 40.5 | 40.5 |
| host tables allocated after the chunked prefill (geometric growth), B/token/layer | 60.75 | 60.75 |
| device centroid copy (f32), B/token/layer | 32.0 | 32.0 |
| persistent in use, B/token/layer (x16 layers) | 72.5 (1,160) | 72.5 (1,160) |
| allocated after the prefill (tables + device + key buffer), B/token/layer (x16 layers) | 100.75 (1,612) | 94.75 (1,516) |
| served bf16 layout for comparison (§4.4), B/token/layer | 24.5 | 24.5 |
| retained index keys in use: peak during a chunk / after commit | 384 KiB / 0 | 384 KiB / 0 |
| key buffer allocated: peak during a chunk / after commit | 512 KiB / 256 KiB | 512 KiB / 256 KiB |
| batch `build` of the same keys, after its commit: tables allocated / key buffer allocated | 40.5 B/token / 256 KiB | 40.5 B/token / 256 KiB |

Decode medians vary by about 0.3 ms between runs (the 1-row 128k step's projection took 0.42 ms in an earlier run); the
prefill and memory figures are stable. The batch build gives the same tables as the chunked prefill (checked by the bench).

Readings for task 4:
- The host pick and expansion dominate prefill (55% at 128k) and grow with T x visible buckets; MLX scoring includes
  a host readback of `[1536, T/8]` f32 scores a chunk (96 MiB at 128k). A fused scan + top-p on the GPU removes both.
- Decode costs about 1 ms a layer, half of it the projection's MLX dispatch and readback, so ~16 ms a token over the 16
  layers, which would add ~60% to QW's ~27 ms decode step at 64k. Batching the 16 layers' projections, keeping the index
  keys on the device and a GPU top-p are needed before this is worth serving.
- Memory is 3x the served layout in use (4x allocated, since the tables grow geometrically) because centroids are f32 and
  held twice (host and device). Keeping one bf16 device copy, with tables reserved to size, gives the 24.5 B/token/layer of
  §4.4.

## 12. Model integration (roadmap task 3; feature `dlsa`)

### 12.1 API

```rust
let config = DlsaConfig::resolve(DlsaMapKind::Distilled, None)?; // CLI override, QW_DLSA_MAP_PATH, then the cache
provider.enable_dlsa(config)?;      // or Qwen35Model::enable_dlsa; resets the sequence
provider.dlsa_stats();              // Option<DlsaStats>, counters per full-attention *layer* forward
provider.dlsa_memory_bytes();       // bucket caches of the current sequence
provider.disable_dlsa();            // dense again, fresh sequence
```

`qw generate --dlsa [--dlsa-map <PATH>]` (built with `--features dlsa`) enables it on the fp16 model `qw generate` loads
and prints the stats to stderr. The serving flags, env and server integration are §13.

`DlsaConfig { map_path, map_dtype (Bf16, as served; F32 for parity), index (QWEN_DEFAULTS), short_context
(DenseFallback; Sparse only for tests), prefill_mask_rows (256) }`. `validate` checks the index config, `index_dim == 64`
and `prefill_mask_rows > 0`; loading checks the map against the checkpoint's full-attention layers and
`kv_heads * (head_dim - rotary_dim)`. D-LSA is a property of the loaded model and its one sequence, not of a request: a
per-request switch would leave the bucket cache built from a different prefix.

### 12.2 What a full-attention layer does (`qwen3_next/dlsa.rs`)

`Qwen3NextAttention::forward_impl` is split into `project_heads`, `rotate`, `dense_attention` and `finish` with the same
ops in the same order; the dense path calls exactly these. `forward_dlsa`, for `L` rows at offset `o`:

1. `project_heads` (q/k/v, gate, q_norm/k_norm). The index reads q/k here, **after the norms and before RoPE** (§4.1).
2. Asserts batch 1, `kv.live_len() == kv.offset` (slots are positions) and `index.len() == o`.
3. `index_keys` and `append_mlx`, always, also on the dense fallback, so later forwards find every block clustered.
4. Dense fallback when `short_context == DenseFallback` and the chunk's **last** row reads the dense key set:
   `p < top_k + window + sink && visible(p) <= top_p` (`reads_dense_key_set`; 4,612 keys under the defaults). The
   predicate is monotone in `p`, so every earlier row reads it too. Otherwise `index_queries` and `select` for `o..o+L`.
5. `commit` (no speculation in this path), then `rotate`.
6. Dense: the unchanged `dense_attention`. Sparse: `update_and_fetch` (fp16 or dequantized int8 `[1,4,T,256]`), then
   - decode (`L == 1`): the list sorted ascending, `take` along the key axis, SDPA without a mask (GQA by head grouping);
   - prefill: row blocks of `prefill_mask_rows`; each an additive mask `[1,1,rb,T]` in the q dtype (0 on listed keys,
     -inf elsewhere; `-1` pads replaced by the row's own position, which every list holds), built by `put_along_axis`
     of zeros into a -inf array as wide as the block's longest list, then SDPA over all cached keys. Additive masks
     match the `create_causal_mask` callers; the C++ bridge casts float masks to Q's dtype.
7. `finish` (gate) and `o_proj`.

Linear (GatedDeltaNet) layers, embeddings, norms and the LM head are untouched.

### 12.3 Support matrix and refusals

| feature | under D-LSA |
|---|---|
| fp16, int8 KV | supported |
| turbo4, fp16+turbo4, fp16+turbo3, turbo4-delegated | refused at `enable_dlsa` (names the mode, says to load fp16) |
| decoder `auto` | resolves to baseline (info log) |
| decoder `mtp`, `dflash` | refused ("use --decoder baseline") |
| prefix reuse (`cached_tokens > 0`), cached benchmarks | refused by the provider |
| image prompts | refused by the provider (nothing in the index blocks mrope; untested) |
| snapshots | `supports_snapshot_reuse() == false`, capture gives `None`, restore errs |
| grammar constraints | expected to work (with snapshot reuse off the generator keeps no snapshot); not tested yet |
| model-level MTP / DFlash2 prefill, verify, rollback | `Err` or panic with a message |

This is the matrix task 3 shipped. Task 5 lifted the speculative, snapshot and serving refusals; the current matrix is
§13.1.

### 12.4 Tests

`./gpu-lock -- cargo test -p qw-runtime --features dlsa --lib`, synthetic (4-layer hybrid, full attention at 1 and 3,
4 q / 2 kv heads, head_dim 32, f32 maps; index `window 16, sink 2, top_k 32, top_p 8, pool 2, bucket 4, cap 8`, so
forwards past position 47 are sparse):

| test | checks | tolerance |
|---|---|---|
| `dlsa::attend` (5) | gathered and masked reads vs an f64 host softmax; full lists vs dense causal; mask row blocks 1/7/256; pads; the 4,612 boundary | 1e-5 (1e-6 blocks) |
| `short_context_is_bitwise_dense` | 40-token prefill + 8 decodes == D-LSA off; index holds 48 keys | bitwise |
| `covering_top_k_matches_dense` | `top_k 512, top_p 128`, `Sparse`: 310 tokens in chunks of 64/37 + 10 decodes, fp16 and int8; every row reads its whole context | allclose 1e-4 / 1e-5 |
| `layer_sparse_read_matches_host_mask` | one layer, L = 120, 37, 1, 1, 37, 1, vs dense SDPA under a host mask from `select_host` of an independent cache; expansion non-empty, lists partial | 1e-5 |
| `dense_key_set_forward_is_bitwise_dense` | one layer under the fallback == `forward_with_position_ids` | bitwise |
| `decode_prefill_consistency` | 260 tokens as 1 forward, 100 + 160 decodes, chunks of 37: logits; members, sizes; centroids | 1e-4; exact; 1e-6 |
| `refusals_and_snapshot_guards`, `reset_...`, `disable_...` | §12.3; reset empties the index and reproduces outputs bitwise; disable is bitwise dense | |

The comparison helpers reject non-finite values on either side. Mutations seen to fail: dropping the own key from the
gathered read (5 tests), and testing the fallback on the chunk's first row instead of its last (T3, T4).

### 12.5 Real model

Ignored tests in `qwen3_5/dlsa_tests.rs`, `mod real`:

```text
./gpu-lock -- cargo test --release -p qw-runtime --features dlsa --lib real_ -- --ignored --test-threads=1 --nocapture
```

Model `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` (4-bit oQ4e, fp16 KV), distilled map in bf16 (`QW_DLSA_MAP_KIND=pca` for the
PCA step-0 map), QWEN_DEFAULTS. Text: the repository's `docs/*.md` then `crates/runtime/src/**/*.rs` ("repo"), or
`crates/mlxcel-core/src/**/*.rs` ("core"), sorted, D-LSA files excluded, truncated by tokens. Apple M4 Max (64 GB),
base `4341216` + this change, release build, 2026-11-05.

**R1 `real_short_context_is_bitwise_dense`**: 4,000 tokens (1536-token chunks) + 16 greedy steps. The final prefill
position's logits and each decode step's logits are bitwise equal to D-LSA off, and the tokens are identical; 288 layer forwards, all on the
fallback. Prefill 15.75 s both, 16 steps 0.72 s dense / 0.69 s D-LSA.

**R2 `real_needle_16k_32k`**: needle "The vault code is 7291-ALPHA." at 25% depth, then the question, chat template,
thinking off, 48 greedy tokens through `generate_baseline_streaming`.

| prompt | dense answer | D-LSA answer | prefill dense | prefill D-LSA |
|---|---|---|---|---|
| 16,322 tokens | `7291-ALPHA` | `7291-ALPHA` | 67.5 s (242 tok/s) | 69.3 s (236 tok/s) |
| 32,706 tokens | `7291-ALPHA` | `7291-ALPHA` | 144.3 s (227 tok/s) | 151.6 s (216 tok/s) |

An explicit MTP request then errs. The provider's `decode_time` here is not decode speed: when generation stops at EOS,
`CxxGenerator::generate_streaming_controlled` resets the model and re-runs prompt + generated tokens in one forward to
align the final state (`mlxcel-core/src/generate.rs`, the `aligned_token_len != final_token_len` branch), dense and
D-LSA alike (72 s / 75 s at 16k for 7 tokens). That replay is also the one forward the R2 stats saw (16,328 rows a
layer, 3,961 keys a row at 16k, 4,287 at 32k). Decode speed is measured in R3.

**R3 `real_logits_vs_dense`**: prefill all but the last 512 tokens (1536-token chunks), teacher-force the last 512 in
128-row forwards, then 32 greedy decode steps; NLL over the 511 scored rows.

| prompt | ppl dense | ppl D-LSA | ratio | mean KL(dense‖D-LSA), nats | top-1 agreement | greedy tokens equal |
|---|---|---|---|---|---|---|
| repo 16k | 2.6435 | 2.6477 | 1.0016 | 0.00091 | 0.9922 | 32 / 32 |
| repo 32k | 1.1763 | 1.1821 | 1.0049 | 0.00340 | 0.9922 | 32 / 32 |
| core 16k | 1.2320 | 1.2351 | 1.0025 | 0.00338 | 0.9941 | 32 / 32 |

The asserted thresholds are provisional: ppl ratio <= 1.10 and top-1 >= 0.90; all three pass with a wide margin. Task 6
sets the real acceptance criteria (RULER, more and longer prompts).

| prompt | prefill dense | prefill D-LSA | decode dense | decode D-LSA | index memory |
|---|---|---|---|---|---|
| repo 16k | 65.5 s | 67.2 s (+3%) | 39.7 ms/token | 50.1 ms/token (+26%) | 27.2 MiB |
| repo 32k | 152.3 s | 156.7 s (+3%) | 49.9 ms/token | 81.5 ms/token (+63%) | 50.4 MiB |
| core 16k | 66.9 s | 76.7 s (+15%) | 41.1 ms/token | 54.1 ms/token (+32%) | 27.2 MiB |

Single runs; the 128-row tail timings varied by 2x between runs (2.5-7.0 s), so read prefill differences under ~5% as
noise. Every prefill row past 4,612 read 4,612 keys.

### 12.6 Costs and what task 4 must fix

- **No speedup yet, by design.** The prefill read is a mask over all T keys, so SDPA still computes every score; decode
  gathers 4,612 keys but then pays the indexer. D-LSA on is slower than dense at 16-32k (prefill +3-15%, decode
  +26-63%). The dense path itself is unchanged (R1, and the bitwise synthetic tests).
- **Host syncs per layer.** `append_mlx` and `select` read back to the host in each of the 16 layers, so every forward
  splits into 16 evaluated segments. `DlsaStats::index_time` includes the evaluation of the layer's inputs that the
  first readback forces, so it is mostly model compute, not indexing; `select_time` (1.7-5 s per 16-32k prefill) is
  the indexer's own cost.
- Decode overhead grows with T (+10 ms at 16k, +32 ms at 32k): scoring all visible centroids, the host top-p and the
  per-layer sync. A GPU scan + top-p, batched projections and a gathered sparse prefill kernel are task 4.
- Masks are `rb x T` in the q dtype: 16 MiB a 256-row block at 32k, 128 MiB at 262k.
- The index holds about 1.6 KiB/token over the 16 layers (50 MiB at 32k), §11.7.

## 13. Serving integration (roadmap task 5; feature `dlsa`)

### 13.1 Support matrix

| feature | under D-LSA |
|---|---|
| fp16 KV | supported, including snapshots and the prefix cache (memory and filesystem tiers) |
| int8 KV | supported; no snapshots (Int8 has none without D-LSA either) |
| turbo4 and the other packed modes | refused at `enable_dlsa`; `qw serve --dlsa` without `--no-kv-quantization` is a startup error naming the flag |
| decoders baseline, MTP, DFlash2, `auto` | supported; `auto` selects as without D-LSA |
| grammar-constrained decoding (all decoders) | supported; snapshots and rollback are index-aware (S5, R8) |
| image prompts | refused: `supports_image_inputs()` is false, so the server answers 400 with a D-LSA-specific message; model-level multimodal entry points err |
| per-request switch | none: D-LSA is a property of the loaded model (§12.1) |

### 13.2 Snapshots

`capture_sequence_state` appends, after the KV and GatedDeltaNet state, dense tensors (`push_tensor`, never paged) for
each full-attention layer `i`, from `DlsaBucketCache::snapshot()`:

| name | dtype, shape | present |
|---|---|---|
| `layer.{i}.dlsa.len` | i32 `[1]` | always |
| `layer.{i}.dlsa.centroids` | f32 `[rows, index_dim]` | `rows > 0` |
| `layer.{i}.dlsa.members` | i32 `[rows, bucket_cap]` | `rows > 0` |
| `layer.{i}.dlsa.sizes` | i32 `[rows]` | `rows > 0` |
| `layer.{i}.dlsa.open_keys` | f32 `[open, index_dim]` | `open > 0` |
| `meta.dlsa.index` | i32 `[8]`: window, sink, top_k, top_p, pool, bucket, cap, index_dim | once |

with `rows = floor(len / W) * (W / bucket_size)` and `open = len mod W`. Why this shape:

- `array_from_portable` rejects zero-sized dims (`portable_snapshot.rs`), so empty tables are absent, not empty.
- The DFlash2 manifest requires every paged tensor's `token_len` to equal the snapshot's (`prefix-cache/src/codec.rs`),
  and bucket rows are per block, not per token, so the tables are dense.
- f32 keeps the restore bitwise.

The tables cost about 650 B/token over the 16 layers (40.5 B/token/layer for closed blocks, plus 256 B per open key per
layer), about 1% of the fp16 KV (64 KiB/token). Each snapshot copies them; there is no page sharing.

Capture returns `None` while any index length differs from the token length. `supports_snapshot_reuse()` is true under
D-LSA.

**Invalidation.** Restore checks the exact name set it derives from the model's D-LSA state and each layer's
`dlsa.len`, so a dense model rejects D-LSA snapshots and a D-LSA model rejects dense ones. Under D-LSA it then checks
`meta.dlsa.index` against the model's index config and every `len` against the snapshot length, and rebuilds every
layer with `DlsaBucketCache::restore` (which rejects out-of-range members) before replacing any state. The rebuilt
caches are installed only after the KV restore succeeded; the stats are kept. In the server a restore error fails the
request, so the prefix-cache namespace keeps D-LSA and dense entries apart (§13.5).

### 13.3 Verify and rollback

`forward_dlsa` commits the index at the **start** of every forward, accepting what the previous forward left after
any rollback, and again at the end only when `DlsaLayerStep::commit` is true. The loops outside
`forward_backbone_with_inputs` route their full-attention layers through one dispatch, `forward_layer_dlsa`: the verify
loops `forward_mtp_verify` and `forward_dflash_verify` with `commit = false`, the DFlash2 prefill/continuation loop
`forward_dflash_prefill_segment_chunked` with `commit = true`. Linear layers and the DFlash hidden capture are
unchanged, and with D-LSA off the loops run their unchanged dense code. So a rollback of the latest verify can always
reach `final_offset >= tail_start`, and the uncommitted keys stay bounded by one forward.

`rollback_mtp_verify` trims the KV caches by the plan, then trims every index to `final_offset`, reopening a cluster
block that rejected rows had closed, and asserts `index.len() == kv.offset` in every layer. `DlsaStats::rollbacks`
counts rollbacks that dropped index keys and `reopened_blocks` those that reopened a block.

Tolerances in the tests: members and sizes exact, centroids within 1e-6 (projection batch rounding, §11.5), logits
allclose 1e-4 (fp16). A multi-row verify reads through a mask and a decode through a gather, so their rounding differs;
with int8 KV the multi-row read already differs without D-LSA, and the int8 tests allow 5e-3 on logits and 1e-4 on
centroids.

### 13.4 Flags and environment

`qw serve` and `qw generate` (built with `--features dlsa`) share `qw_server::DlsaArgs`; without the feature the
flags do not exist. `DlsaOptions::resolve(env)` takes each value from its flag, then its variable, then
`QWEN_DEFAULTS`; the environment is read through an injected lookup.

| flag | variable | default |
|---|---|---|
| `--dlsa` | `QW_DLSA` (`1/true/yes/on`, `0/false/no/off`) | off |
| `--dlsa-map` | `QW_DLSA_MAP_PATH` | the verified cache file of the map kind |
| `--dlsa-map-kind distilled\|pca` | `QW_DLSA_MAP_KIND` | `distilled` |
| `--dlsa-sink` | `QW_DLSA_SINK` | 4 |
| `--dlsa-window` | `QW_DLSA_WINDOW` | 512 |
| `--dlsa-top-k` | `QW_DLSA_TOP_K` | 4096 |
| `--dlsa-top-p` | `QW_DLSA_TOP_P` | 1024 |
| `--dlsa-dense-below <TOKENS>` | `QW_DLSA_DENSE_BELOW` | 12288 (task 6, §15.7): query rows at positions below it read every causal key; 0 keeps only the dense-key-set rule |
| `--dlsa-backend metal\|mlx\|metal-select` | `QW_DLSA_BACKEND` | `metal` (task 4's kernels; `mlx` is the task-3 path, `metal-select` its bitwise twin) |

The map is held in bf16 with the dense fallback: a forward runs the unchanged dense attention (bitwise) when its
rows read the dense key set (up to 4,612 tokens under the defaults) or all sit below `--dlsa-dense-below`. A forward
that crosses the threshold reads its rows below it over every causal key and its other rows sparse (§15.7; the
index is built throughout). Errors name the flag or variable: an unparsable variable, a config `DlsaConfig::validate` refuses
(the message lists each knob with its source), a `--dlsa-*` flag without `--dlsa`/`QW_DLSA`, a missing or corrupt
map. A window other than 512 is accepted with a warning: the maps were distilled at 512, and the window is also the
cluster block size.

### 13.5 Server

- `qw serve --dlsa --no-kv-quantization`. The KV check runs on the flag (`validate_cli`) and again on the resolved
  setting, so `QW_DLSA=1` without `--no-kv-quantization` fails too.
- The worker loads the model, enables D-LSA, then builds the prefix-cache namespaces. The D-LSA identity (`dlsa_identity`:
  sha256 of the map file, map dtype, short-context policy, `dense_below`, the read (`metal`, or `mlx` for both MLX-read backends) and
  the full index config) is appended to the namespace parts
  **only when D-LSA is on**, so dense namespaces are byte-identical to a build without the feature; the DFlash2
  namespace inherits it.
- Startup logs `info!(phase = "server.dlsa", map, map_sha256, kind, map_dtype, sink, window, top_k, top_p,
  short_context, "D-LSA armed")`.
- Image requests get HTTP 400: "image inputs are not supported while D-LSA sparse attention is enabled on this server".
- Each request logs `debug!(phase = "dlsa.request")` with the `DlsaStats` counters (dense-key-set and sparse forwards,
  sparse rows, keys read, rollbacks, reopened blocks, index and select time) and `index_bytes`.

### 13.6 Tests

Synthetic, in `qwen3_5/dlsa_serving_tests.rs` (`./gpu-lock -- cargo test -p qw-runtime --features dlsa --lib`):

| test | checks |
|---|---|
| T1 `snapshot_restore_continues_bitwise` | capture at 70, 64 and 10 tokens; restore into a fresh model and after a reset; 20 decodes bitwise, tables bitwise |
| T2 `snapshot_survives_portable_round_trip` | the persisted (portable) form restores identically |
| T3 `snapshot_restore_equals_rebuild` | restore + continue == a fresh prefill in another chunk schedule |
| T4 `snapshot_invalidation` | dense <-> D-LSA, other index config, out-of-range member, wrong length all err; capture `None` out of step |
| T5 `snapshot_dlsa_bytes` | the D-LSA tensors cost exactly their table bytes |
| S1 `rollback_through_block_boundary`, `rollback_random_rounds` | 4- and 8-row verifies closing a block, rolled back to 0, 1, partial or all rows; 200 seeded tokens at widths 3 and 8, fp16 and int8; lockstep after every round |
| S2 `covering_greedy_speculation_matches_dense`, `covering_mtp_generation_matches_dense` | covering top-k: greedy speculation and a synthetic MTP head give the dense and the sequential tokens |
| S3 `dflash_verify_rollback_continuation` | DFlash2 verify, rollback and continuation at model level |
| S4 `snapshot_after_partial_rollback` | a snapshot after a partial rollback restores and continues bitwise |
| S5 `constrained_mtp_generation_matches_dense` | constrained MTP generation with forced rejections and a splice == dense |

Server and CLI (`cargo test -p qw-server --features dlsa`, `-p qw-cli --features dlsa`, and without the feature): flag
parsing and help, env precedence and error naming (`dlsa::options`), the KV refusal, namespaces (dense unchanged, D-LSA
distinct per config), and the image 400.

Real model (ignored; the model, map and text conventions of §12.5, plus the DFlash2 drafter):

```text
./gpu-lock -- cargo test --release -p qw-runtime --features dlsa --lib -- \
  qwen3_5::dlsa_serving_tests::real real_dlsa_constrained --ignored --test-threads=1 --nocapture
```

R4 `real_speculative_short_context_matches_dense` (~4k, defaults), R5 `real_speculative_covering_matches_dense` (~8k,
`Sparse`, top_k 8192, top_p 1024), R6 `real_speculative_needle_16k_32k`, R7 `real_snapshot_restore_is_bitwise` (16k,
snapshot at 12,288), R8 `real_dlsa_constrained_prompt_bases_restore_after_splice_and_rollback`
(`qwen3_5_dflash.rs`).

### 13.7 Measurements

Apple M4 Max (16 cores, 64 GB), base `89b4778` + this change, release build, 2026-11-05. Model
`Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` (fp16 KV), distilled map in bf16, drafter `incoai/Qwen3.8-27B-DFlash2`, greedy. All
eight real tests (R1-R8) pass. Decode is ms/token, dense vs D-LSA; acceptance likewise.

| test | decoder | tokens | acceptance | decode ms/token | rollbacks / reopened |
|---|---|---|---|---|---|
| R4, 4,000 tokens, defaults (all forwards on the dense fallback), 64 tokens | MTP | equal | 0.830 / 0.830 | 20.0 / 20.6 | 7 / 0 |
| | DFlash2 | equal | 0.647 / 0.688 | 26.9 / 26.4 | 13 / 0 |
| R5, 8,000 tokens, `Sparse`, top_k 8192, top_p 1024 (464 sparse layer forwards), 64 tokens | MTP | equal | 0.889 / 0.889 | 18.1 / 24.2 | 5 / 0 |
| | DFlash2 | equal | 0.857 / 0.855 | 20.0 / 24.1 | 5 / 0 |
| R6 needle, 16,322 tokens, defaults, 48 tokens | MTP | `7291-ALPHA` both | 1.000 / 1.000 | 24.7 / 26.5 | 1 / 0 |
| | DFlash2 | `7291-ALPHA` both | 0.556 / 0.455 | 44.9 / 52.6 | 3 / 0 |
| R6 needle, 32,706 tokens | MTP | `7291-ALPHA` both | 1.000 / 1.000 | 24.6 / 27.6 | 1 / 0 |
| | DFlash2 | `7291-ALPHA` both | 0.455 / 0.556 | 52.2 / 54.7 | 3 / 0 |

DFlash2's acceptance counters vary between runs with identical tokens (R4); R6's answers are 7 tokens, so its
acceptance differences are noise. R5 found no divergence, so no margin was printed. No rollback reopened a block in
these runs (the synthetic S1 tests cover reopening).

- **R7**: a 16,384-token prompt snapshotted at 12,288, restored (in memory and through the portable form), the rest
  prefilled, 32 greedy tokens: logits and tokens bitwise equal to the uninterrupted run. The snapshot is 850.4 MiB,
  almost all fp16 KV; the D-LSA tables add about 0.6 KiB/token over the 16 layers (~1%).
- **R8**: the DFlash2 constrained prompt bases restore after a splice and a rollback with D-LSA on; the final snapshot
  carries `meta.dlsa.index`.
- **Regressions (§12.5)**: R1 bitwise (prefill 15.76 s vs 15.75 s dense; index 7.8 MiB). R2 `7291-ALPHA` at 16k and
  32k (prefill 67.6 / 69.4 s and 144.4 / 148.4 s). R3:

| prompt | ppl ratio | mean KL, nats | top-1 agreement | greedy tokens equal | decode ms/token dense / D-LSA | index MiB |
|---|---|---|---|---|---|---|
| repo 16k | 1.0017 | 0.00090 | 0.9961 | 32 / 32 | 39.5 / 49.8 | 27.2 |
| repo 32k | 1.0042 | 0.00260 | 0.9980 | 20 / 32 | 47.7 / 88.2 | 50.4 |
| core 16k | 1.0025 | 0.00330 | 0.9941 | 32 / 32 | 40.2 / 53.8 | 27.2 |

**Server smoke** (`qw serve --dlsa --no-kv-quantization`, a 29,307-token prompt, 64 tokens):

- the `D-LSA armed` line shows sink 4, window 512, top_k 4096, top_p 1024, `DenseFallback`, `Bf16` and the pinned
  map sha256 (`00200004e8ab…62e1`);
- baseline, `mtp` and `dflash` each answer identically on a repeat, with `cached_tokens` 29,307; the request time
  drops from about 135 s to 7.7 / 5.9 / 6.1 s. After a restart the filesystem tier gives the same output, again with
  `cached_tokens` 29,307;
- `json_schema` returns valid JSON under all three decoders;
- an image request gets HTTP 400 "image inputs are not supported while D-LSA sparse attention is enabled on this
  server";
- without `--no-kv-quantization` the server exits 1 with "D-LSA (--dlsa or QW_DLSA) requires
  --no-kv-quantization…";

As §12.6 predicts, D-LSA costs decode time once reads are sparse: R5 +21-34%, the R6 needles +7-12% (MTP) and
+5-17% (DFlash2), R3 +26-85% (most at 32k). On the dense fallback (R4) it is within noise. Task 4's kernels and
GPU-side selection are meant to remove this cost.

## 14. Metal kernels (roadmap task 4; feature `dlsa`)

Task 4 moves selection and the sparse read onto the GPU. Code comments that cite "§13" for the kernels mean this
section.

### 14.1 Design

- **Scores stay the task-2 matmul.** `score_device` is the same MLX f32 matmul of index queries against the device
  centroids that task 3 reads back. So every backend ranks identical scores, and "same lists as task 3" holds by
  construction. The centroids live in a device mirror (`dlsa/device.rs`): preallocated arrays that grow
  geometrically and are filled with `slice_update` from the host tables. The host tables stay authoritative; a trim
  only lowers the synced row count.
- **K1 `dlsa_select`** (`mlxcel-core/cpp/mlx_cxx_dlsa.cpp`, `src/dlsa_kernels.rs`) runs one threadgroup per query
  row:
  - Each bucket gets a unique 64-bit rank key: `(ord(score) << 32) | (0x7FFFFFFF - bucket)`, with `-0.0` mapped to
    `+0.0`.
  - An MSD radix select over 8-bit digits (≤ 8 passes, a 256-bin threadgroup histogram each) finds the `top_p`-th
    key, so no sampled threshold or fp32 re-scoring is needed.
  - The keys at or above it are compacted and bitonic-sorted (the picks, best first).
  - Expansion walks the picks in rank order: an exclusive scan of each pick's eligible member count, written while
    below `top_k`, then the window, the sinks and `-1` padding.
  - Output is bit-identical to `select_from_scores` on the same scores. The one deviation: a NaN ranks last and sets
    `status` bit 0, instead of the host's error; `status` is read only at existing sync points.
- **K2 `sparse_attention`** (`cpp/mlx_cxx_dlsa_attention.cpp`, `src/dlsa_attention.rs`) is modeled on MLX's
  `sdpa_vector`:
  - One threadgroup per (row, KV head) owns the G = 6 query heads of its group, so each listed K/V row is fetched
    once for all six. The heads split 3 + 3 over simdgroups.
  - An online softmax runs in the exp2 domain, with f32 accumulators.
  - K/V are read through their strides: the cache's fetched view is never copied.
  - Prefill rows (L·Hkv ≥ 160 threadgroups) run one pass with 8 simdgroups. Decode and verify rows use split-K,
    about 512 threadgroups merged by a second kernel.
  - Results are deterministic for fixed shapes and launch knobs.
  - A simdgroup-matrix variant measured 117 ms vs 54 ms per 32k chunk and was dropped.
- **Integration** (`qwen3_next/dlsa.rs`, `dlsa/gpu.rs`, `dlsa/cache.rs`): `DlsaConfig.backend` picks the backend,
  set by `--dlsa-backend` / `QW_DLSA_BACKEND` (§13.4).
  - `Mlx` is task 3: scores read back, host pick, masked SDPA (prefill) or take + SDPA (decode).
  - `Metal` (the default) runs K1, then K2 on K1's device lists, with nothing read back between them.
  - `MetalSelect` runs K1, then reads the lists back for the task-3 read. It is a parity mode that is bitwise equal
    to `Mlx`.
  - **Deferred decode keys** (Metal only): index keys that close no block stay on the device (`append_deferred`).
    Anything host-side flushes them first with one readback: closing a block, `tail_keys`, `snapshot`, `trim`. So
    decode syncs about once per 512 tokens per layer instead of twice per token per layer. Their finiteness check
    moves to the flush.
  - Prefill chunks always close blocks, so Ward stays on the host (see G5).
- **Profile hooks** (`dlsa/profile.rs`): `QW_DLSA_PROFILE=1` (and optionally `QW_DLSA_PROFILE_FROM=<offset>`) ends
  every stage of a D-LSA layer forward in an `eval` and adds its wall time to a process-wide profile. With the
  variable unset, each hook is one atomic and thread-local check: no `eval`, timer or MLX op.

### 14.2 Tests

- **K1** (`mlxcel-core` `dlsa_kernels_tests`, runtime `dlsa/gpu_select_tests`): kernel == host on every fixture
  case run as forwards would, on the fixtures' reference scores, on 72 random flat configs, and on tie-heavy scores
  (`{-1,0,1}`, ±0, constants). Also covered: binding knobs and block edges, an empty cache, Qwen defaults at
  6k/32k/131k/262k, `top_p` at the 2048 limit, threadgroup-size invariance, NaN status, bad arguments.
- **K2** (`dlsa_attention` unit test, `dlsa/gpu_attend_tests`):
  - fp16 against task 3's masked and gathered reads, both against an f64 host softmax on sampled rows;
  - full-coverage lists equal dense causal attention;
  - short and padded lists;
  - a strided cache view bitwise equal to the same data made contiguous;
  - index arithmetic at T = 262,144.
- **Device state** (`dlsa/device_tests`): the mirror grows, trims and scores like a fresh upload; deferred keys equal
  eager appends over random schedules, flush when a block closes and check finiteness; memory accounting counts
  deferred keys.
- **Model** (`qwen3_5/dlsa_tests`): `metal_select_is_bitwise_task3_and_metal_read_is_close` (every schedule, fp16 and
  int8 KV) and `metal_greedy_tokens_equal_task3`. `qwen3_next/dlsa.rs`'s layer test also runs the sparse read against
  a host mask on all three backends and both KV modes.
- **Real model**:
  - R1 `real_short_context_is_bitwise_dense` passes with the default (Metal) backend: 288 dense-key-set layer
    forwards, 0 sparse, prefill 15.76 s both.
  - R9 is in §14.3.

### 14.3 Measurements

**Setup.**

- Machine: Apple M4 Max, 64 GB. Build: base `8d985ff` plus the uncommitted task-4 tree (`git diff | sha256`
  `c6ad9730…`; diff and untracked files `4ace8e64…`), release.
- Model `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp`, fp16 KV; distilled map (sha256 `00200004…62e1`) in bf16; Qwen defaults;
  greedy.
- Prompt: a fixed token file, 262,144 tokens of the repo corpus (sha256 `43583dad…78b1`), cut to
  `floor(T / 1536) · 1536` tokens: 32,256 / 130,560 / 261,120.
- Bench: `real_dlsa_bench` (`qwen3_5/dlsa_bench.rs`), one process per context, every mode in one process. Prefill
  goes through `forward_last_logits` in 1536-row chunks, then 32 unfenced greedy decode steps. The last 2 prefill
  chunks and 8 extra decode steps are fenced for the stage profile.

**Memory methodology.**

- The tests load the model without `provider::initialize_runtime`, so MLX's free-buffer cache had no limit. At 262k
  the process reached 57 GB next to a 24 GB VM and swapped 27 GB. Task 4's first ("before") 128k/256k runs were
  invalidated by this and are not reported.
- The bench now sets `MLXCEL_CACHE_LIMIT` (default 8 GiB, as serving does), clears the cache between modes, and
  records swap before and after each mode.
- Limits used: 8 GiB at 32k, 4 GiB at 128k, 1 GiB at 256k. The 256k dense run may lose a little to the small cache.
- No run reported here ended with more swap in use than it started with. In 128k and the 32k runs "used" stayed
  within 32 MB (128k: 19,161.50M → 19,129.50M over the three modes; 32k: unchanged at 19,161.50M and 19,460.62M). At
  256k it started at 29,438.50M, left over from other work, and fell to 19,572.62M during the dense mode, then
  19,476.62M and 19,460.62M after mlx and metal.

**Real model, fp16 KV.** Speedups are relative to dense: prefill wall time, decode median.

| context | mode | prefill s | prefill tok/s | ×dense | chunk s at T/4 · T/2 · 3T/4 · T | decode ms median / p90 | ×dense | peak GiB |
|---|---|---|---|---|---|---|---|---|
| 32k | dense | 143.2 | 225 | 1.00 | 6.3 · 6.7 · 7.1 · 7.6 | 41.2 / 41.3 | 1.00 | 19.5 |
| | mlx (task 3) | 147.4 | 219 | 0.97 | 6.5 · 6.9 · 7.4 · 7.7 | 49.2 / 50.0 | 0.84 | 20.1 |
| | **metal** | 137.5 | **235** | **1.04** | 6.5 · 6.6 · 6.6 · 6.6 | **39.0 / 39.3** | **1.06** | 18.8 |
| | metal-select | 144.9 | 223 | 0.99 | 6.4 · 6.8 · 7.3 · 7.5 | 49.8 / 50.2 | 0.83 | 20.1 |
| 128k | dense | 943.3 | 138 | 1.00 | 7.6 · 10.3 · 14.4 · 17.4 | 55.0 / 55.2 | 1.00 | 26.0 |
| | mlx | 847.5 | 154 | 1.11 | 7.9 · 9.9 · 11.8 · 13.7 | 50.1 / 50.6 | 1.10 | 27.6 |
| | **metal** | 569.0 | **229** | **1.66** | 6.6 · 6.8 · 6.8 · 6.9 | **39.4 / 39.6** | **1.40** | 25.1 |
| 256k | dense | 3096.2 | 84 | 1.00 | 11.0 · 17.4 · 24.1 · 34.1 | 86.4 / 86.7 | 1.00 | 34.4 |
| | mlx | 2903.3 | 90 | 1.07 | 10.1 · 14.2 · 25.6 · 33.3 | 51.5 / 51.8 | 1.68 | 34.3 |
| | **metal** | 1166.6 | **224** | **2.65** | 6.7 · 6.8 · 7.0 · 7.2 | **39.9 / 40.1** | **2.17** | 34.1 |

- 32k was run 3 times. Prefill tok/s dense 225–226, mlx 219–220, metal 235–236; decode median dense 41.0–41.2, mlx
  48.9–49.2, metal 38.9–39.0 ms.
- The first decode step after a long prefill is slow in every mode (160–270 ms at 32k, 390–410 ms at 128k, 650–800
  ms at 256k). The medians exclude nothing.
- "Peak" is MLX's active peak. The index adds 54 / 206 / 407 MiB at 32k / 128k / 256k. Under Metal this
  underreports by at most one open block of deferred f32 keys per layer, a counting bug fixed after these runs.

**Identity at 32k (R9).** Final-prefill logits plus 32 decode steps: 33 positions.

| comparison | greedy tokens | logits |
|---|---|---|
| `MetalSelect` vs `Mlx` (task 3) | 33 / 33 | **bitwise equal** at all 33 positions |
| `Metal` (K2 read) vs `Mlx` | 33 / 33 | not bitwise: max-abs 0.084 (final prefill row), 0.30 (max); mean KL 7.8e-6 nats (max 9.8e-5); top-1 agreement 1.0 |
| dense vs `Mlx`, for scale | 33 / 33 | max-abs 1.36 / 3.42; mean KL 1.5e-3 nats (max 0.027); top-1 1.0 |

In the 1a microbench below, K1's lists equal task 3's at every measured depth.

**Stage breakdown, real model.** Each value is ms per full-attention layer in a fenced 1536-row chunk at depth T
(prefill) or a fenced decode step, with % of the fenced forward.

| stage | 32k mlx | 32k metal | 128k mlx | 128k metal | 256k mlx | 256k metal |
|---|---|---|---|---|---|---|
| qkv projection | 18.4 (3.6%) | 18.9 (4.5%) | 18.1 (2.1%) | 17.9 (4.1%) | 19.3 (1.1%) | 18.2 (3.9%) |
| score matmul | 0.27 | 0.26 | 0.55 | 0.55 | 4.3 (0.24%) | 4.5 (0.96%) |
| scores readback | 0.88 | — | 4.0 (0.46%) | — | 16.3 (0.89%) | — |
| host pick / K1 select | 12.1 (2.4%) | 1.46 (0.34%) | 20.2 (2.3%) | 2.2 (0.51%) | 30.3 (1.7%) | 3.3 (0.70%) |
| Ward | 0.88 | 0.78 | 0.71 | 0.70 | 0.75 | 0.85 |
| KV fetch | 4.5 | 4.5 | 17.9 (2.0%) | 17.5 (4.0%) | 39.6 (2.2%) | 36.0 (7.7%) |
| mask build | 2.6 | — | 4.1 | — | 12.9 (0.70%) | — |
| read | 106.3 (21.0%) | **38.7 (9.1%)** | 451.1 (51.5%) | **39.8 (9.1%)** | 1338 (73.1%) | **39.3 (8.4%)** |
| finish | 8.2 | 8.6 | 8.1 | 8.2 | 8.7 | 10.6 |
| full-attn layers, % of forward | 43.6% | 32.7% | 67.4% | 34.8% | 84.2% | 38.7% |
| fenced forward, s | 8.10 | 6.79 | 14.02 | 6.99 | 29.29 | 7.52 |

Dense's full-attention layers take 41.0% / 74.4% / 87.0% of the fenced prefill forward at 32k / 128k / 256k. In
decode they take 28.7% / 43.1% / 60.6%.

In a decode step, the full-attention stages per layer are:

| context | backend | stages, ms per layer | sum |
|---|---|---|---|
| 32k | mlx | key readback 0.03, score 0.13, scores readback 0.02, host pick 0.08, take 0.44, read 0.18 | 2.67 |
| 32k | metal | key readback 0.00 (deferred), score 0.14, K1 0.18, K2 0.20 | 2.42 |
| 256k | mlx | — | 2.98 |
| 256k | metal | — | 2.57 |

Fences inflate decode (89–97 ms fenced vs 39–52 unfenced), so the unfenced medians are the decode result. The
mlx-to-metal gap there (10–12 ms per token) is mostly the per-layer host syncs that Metal removes.

**One layer, synthetic Qwen geometry** (`examples/dlsa_indexer_bench.rs`, fp16 q/K/V, real lists from the cache,
median of 5 runs; decode median of 25). Each cell is ms for one 1536-row chunk ending at the given depth, or one
decode row at T.

| T | depth | dense SDPA | task 3: select + masked SDPA | Metal: K1 + K2 |
|---|---|---|---|---|
| 32k | 8k / 16k / 24k / 32k | 25.9 / 51.9 / 77.5 / 106.1 | 6.8+27.4 / 9.3+53.6 / 10.6+77.6 / 11.7+105.6 | 1.0+38.3 / 1.4+38.4 / 1.4+38.5 / 1.5+38.6 |
| 128k | 32k / 64k / 96k / 128k | 106.0 / 221.8 / 375.3 / 561.6 | 11.5+106.3 / 16.4+214.2 / 20.6+339.3 / 24.3+454.0 | 1.5+38.6 / 1.9+39.0 / 2.2+39.8 / 2.5+41.8 |
| 256k | 64k / 128k / 192k / 256k | 222.7 / 561.3 / 812.1 / 1589.1 | 16.0+214.7 / 24.4+453.9 / 31.7+974.5 / 44.1+1075.4 | 1.9+38.9 / 2.5+41.6 / 3.1+44.5 / 3.8+44.9 |
| decode | at 32k / 128k / 256k | 0.44 / 1.36 / 3.41 | 0.21+0.35 / 0.22+0.36 / 0.26+0.36 | 0.18+0.20 / 0.21+0.20 / 0.22+0.20 |

The dense prefill baseline is QW's current dense path. With D = 256 and a continued prefill, MLX's SDPA materializes
the scores in query chunks (`layers.rs` `sdpa_materializes_scores`; task 3's masked read pays the same). So part of
the prefill gain is "flash-style sparse kernel vs materialized dense": a dense D = 256 flash kernel would narrow it.

**Short sparse depths are slower than dense.** The dense fallback ends at 4,612 tokens, so chunks at depths from
4.6k to about 12k keys run K1 + K2 and are slower than dense SDPA there. In 1a at 8k, one layer takes 1.0 + 38.3 ms
vs 25.9 ms dense, about +13 ms. The crossover is near 12k (dense grows about 3.2 ms per 1k keys); at 16k Metal
already wins (39.8 vs 51.9 ms).

- Estimate from those per-layer numbers: 16 layers × +13 ms ≈ +0.2 s on a 6.0–6.3 s chunk at 8k, about +3%.
  Shallower sparse chunks (4.6k–8k, where dense is cheaper still) lose up to about +6%.
- Inside the 32k run, the real model agrees: the chunk at 8k depth took 6.5 s with Metal vs 6.3 s dense.
- A whole short prompt (about 8k–16k) was **not measured end to end**.
- Decode below 32k was not measured either. At 32k Metal decode beats dense, 39.0 vs 41.2 ms. Dense decode per layer
  shrinks with T (1a: 0.44 ms at 32k), while Metal's index projections and K1 + K2 cost about 0.4 ms per layer at any
  depth. So Metal decode may be slower than dense somewhere below 32k. Task 3's 16k decode was +26% (§13.7, R3).

### 14.4 Verdict

| context | prefill | decode |
|---|---|---|
| 32k | **Beats dense by 4%** (235 vs 225 tok/s). The read is 21% of the dense forward and K2 cuts it only 2.7×; the 48 linear-attention layers (≈ 67% of the forward) are untouched. | **Beats dense by 6%** (39.0 vs 41.2 ms). The read is cheaper than dense and no host syncs remain; task 3 lost 19% here. |
| 128k | **Beats dense by 1.66×** (229 vs 138 tok/s). K2 stays at about 40 ms per chunk while dense grows to 17 s per chunk at T. | **Beats dense by 1.40×** (39.4 vs 55.0 ms). |
| 256k | **Beats dense by 2.65×** (224 vs 84 tok/s). Prefill throughput is flat in T (6.7 → 7.2 s per chunk). | **Beats dense by 2.17×** (39.9 vs 86.4 ms). Metal decode is flat in T (39.0 → 39.9 ms from 32k to 256k). |

What is left in the Metal path is mostly dense code:

- linear layers, 4.55 s per chunk;
- q/k/v projections, 18 ms per layer;
- the KV fetch, 36 ms per layer at 256k (cache growth: a dense cost);
- the K2 read itself, about 40 ms per layer.

Selection is under 1% of the forward.

Where D-LSA does not beat dense (Metal backend):

- Prefill chunks at sparse depths of about 4.6k–12k keys: an estimated +3–6% per chunk (above). Not measured end to
  end for short prompts.
- Decode at contexts below 32k: may be slower than dense, not measured.
- Prompts up to 4,612 tokens run the dense fallback and equal dense bitwise (R1).

### 14.5 Gates

The gates were decided from task 4's first profile (32k real model, and the 1a microbench at all three contexts).

- **G1 (K2 prefill): built.** The masked read was the largest stage: 21% of the forward at 32k, 51–73% at 128k–256k
  in the runs above.
- **G2 (K1): built.** Select plus readback took 12–47 ms per layer per chunk and grew with T. K1 takes 1.5–3.8 ms.
- **G3 (deferred keys): built.** Task 3 decode cost +8 ms per token at 32k with near-equal per-layer compute, which
  means sync stalls.
- **G4 (K2 for decode): built.** It beats take + SDPA, 0.20 vs 0.35 ms per layer.
- **G5, GPU Ward: pruned.** Ward is 0.7–0.9 ms per layer per prefill chunk, 0.04–0.18% of the forward (0.28 ms per
  block), far below the 3% threshold.
- **G6, fused centroid scan: pruned.** The score matmul is 0.05–0.96% of the forward (4.5 ms per layer at 256k).
  The scores readback, the only real cost of the scan, is gone with K1.

### 14.6 Reproducing

Build outside the GPU lock (`cargo test --release -p qw-runtime --features dlsa --lib --no-run`), then run one
context per lock hold:

```text
MLXCEL_CACHE_LIMIT=<8589934592 | 4294967296 | 1073741824> QW_DLSA_BENCH_CONTEXTS=<32768 | 131072 | 262144> \
  QW_DLSA_BENCH_MODES=dense,mlx,metal[,metal-select] QW_DLSA_BENCH_PROMPT=<token file> \
  QW_DLSA_BENCH_OUT=<unique>.json [QW_DLSA_BENCH_LOGITS=1] \
  ./gpu-lock -- target/release/deps/qw_runtime-<hash> qwen3_5::dlsa_bench::real_dlsa_bench --exact --ignored \
  --nocapture --test-threads=1
./gpu-lock -- target/release/examples/dlsa_indexer_bench --contexts <T> --modes dense,mlx,metal --out <unique>.json
```

If the prompt file is missing, the bench writes it from the corpus. `QW_DLSA_BENCH_LOGITS=1` writes the raw f32
logits used for R9 next to the report. Each report records git sha, model config, map sha256, cache limit, swap and
peak memory. These runs predate `dense_below` (§15.7); to reproduce them now, set `QW_DLSA_DENSE_BELOW=0`.

## 15. Evaluation (roadmap task 6)

This section measures the final implementation (task 4's Metal kernels, task 5's serving integration) against
dense attention on one machine. The focus is prefill throughput. RULER is not run here; it is task 7, in §16.
**Accuracy at scale was unverified until then.** §15.5 is a cheap sanity check only, and it already shows two
documents where D-LSA diverges from dense.

### 15.1 Setup

- **Machine.** Apple M4 Max (16 cores, 64 GB), 2026-11-06.
- **Build.** Base `56490f7` plus this task's bench extension, uncommitted (`git diff | sha256` `2a5f3a6e…`),
  release, `--features dlsa`.
  - At the time of the main runs `dense_below` did not exist, so every D-LSA mode read sparse from the
    dense-key-set boundary (4,612) on, as tasks 3–5 did. That equals today's `QW_DLSA_DENSE_BELOW=0`.
  - The §15.7 runs use the final tree with the 12,288 default; the §15.6 diagnostics use `QW_DLSA_DENSE_BELOW=0`.
- **Model and maps.**
  - Model `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` (4-bit oQ4e, `config.json` sha256 `7a063b70…`), fp16 KV.
  - Distilled map (sha256 `00200004… 62e1`) in bf16. PCA step-0 map (`03dbc211…f9fb`, fetched with
    `qw download --dlsa-map --dlsa-pca`) in bf16.
  - Qwen defaults: sink 4, window 512, top_k 4096, top_p 1024. Backend `metal`.
- **Prompts.**
  - Task 4's token file: 262,144 tokens of the repository corpus, not cycled, sha256 `43583dad…78b1`.
  - Each repetition is a different "document": the prompt starts at token offset 0, 65,536 or 131,072 of the file.
  - The prompt is `floor((T - 1) / 1536) · 1536` tokens (3,072 / 7,680 / 10,752 / 15,360 / 23,040 / 32,256 /
    64,512 / 130,560 / 261,120). Five needles are written into it (§15.5).
  - Prefill runs in 1536-row chunks through `forward_last_logits`.
- **Repetitions.**
  - 4k–64k: 3 documents (offsets 0, 65,536, 131,072).
  - 128k: 2 (offsets 0 and 131,072).
  - 256k: 1 (offset 0). Task 4's run (§14.3: 84 / 224 tok/s, same prompt, earlier tree) is a second sample.
  - Dense and D-LSA run in the same process, on the same prompt, back to back, so ratios are paired.
- **Memory.**
  - `MLXCEL_CACHE_LIMIT` 8 GiB up to 64k, 4 GiB at 128k, 1 GiB at 256k (as §14.3).
  - Swap "used" was 19.1–19.4 GB, left over from earlier work, and **never grew** during a mode: every run's
    after-value is equal to or below its before-value. No run swapped.
- **Thermal.** `pmset -g therm`, before and after every mode, reported no thermal or performance warning level
  and no CPU power limit.
- **Bench.** `real_dlsa_bench`, extended (`qwen3_5/dlsa_bench.rs`, module docs). Why the extension rather than
  `single_user_throughput`:
  - That bench has fixed fresh/10k/64k fixtures, a cached-prefix prefill, and no D-LSA switch.
  - The real bench already sweeps contexts, runs every mode in one process, holds the stage profile, and records
    the cache limit and swap.
  - What it adds, per mode:
    - a teacher-forced 256-row tail (§15.5);
    - five needles plus a prefix-cache hit (§15.4);
    - the MTP and DFlash2 generators (§15.4);
    - a `metal-pca` mode;
    - `pmset`.
- **Results.** The raw reports (minified) are in `eval/dlsa/task6/`, with `summarize.py`, which produced every
  table below.
- **GPU time.** About 8.5 h including §15.6–15.7, one context per `./gpu-lock` hold.

### 15.2 Prefill

Prefill tok/s is prompt tokens over the whole chunked prefill. That includes the final logits readback, so it is also
the time to first token (TTFT); the fenced profile chunks (the last 2) are included too. Each cell gives the median
and the range over documents; the ratio is per document, metal over dense.

| context | prompt | docs | dense tok/s | D-LSA metal tok/s | metal / dense | TTFT dense / metal | peak GiB dense / metal | index B/token |
|---|---|---|---|---|---|---|---|---|
| 4k | 3,072 | 3 | 244.1 (239–247) | 250.7 (250–252) | 1.031 (1.013–1.047) | 12.6 / 12.3 s | 17.4 / 17.2 | 2,013 |
| 8k | 7,680 | 3 | 249.8 (249–250) | 243.9 (244–244) | **0.977** (0.976–0.978) | 30.7 / 31.5 s | 17.7 / 17.7 | 2,266 |
| 12k | 10,752 | 3 | 247.1 (247–247) | 241.2 (241–241) | **0.976** (0.976–0.976) | 43.5 / 44.6 s | 17.9 / 17.7 | 2,106 |
| 16k | 15,360 | 3 | 242.7 (243–243) | 239.2 (239–239) | **0.986** (0.985–0.986) | 63.3 / 64.2 s | 18.5 / 17.8 | 1,993 |
| 24k | 23,040 | 3 | 234.7 (235–235) | 237.3 (237–237) | 1.010 (1.010–1.011) | 98.2 / 97.1 s | 18.7 / 18.3 | 1,783 |
| 32k | 32,256 | 3 | 225.4 (225–225) | 235.9 (236–236) | 1.047 (1.046–1.048) | 143.1 / 136.7 s | 19.5 / 18.8 | 1,768 |
| 64k | 64,512 | 3 | 198.4 (198–199) | 232.9 (233–233) | 1.174 (1.171–1.176) | 325.2 / 277.0 s | 21.5 / 20.8 | 1,703 |
| 128k | 130,560 | 2 | 138.5 (138–138) | 229.2 (229–229) | **1.656** (1.655–1.657) | 943.0 / 569.6 s | 26.0 / 25.3 | 1,650 |
| 256k | 261,120 | 1 (+§14.3) | 83.9 (§14.3: 84) | 223.2 (§14.3: 224) | **2.661** | 3113 / 1170 s | 34.4 / 33.7 | 1,634 |

Notes on the table:

- **Noise.** Spreads are under 0.5% except dense at 4k (239–247). There D-LSA is bitwise dense (every forward
  is on the dense key set), so the 4k "gain" is noise: dense runs first in each process and pays the warm-up.
- **Peak.** "Peak" is MLX's active peak during the prefill, model included. D-LSA's is slightly lower from 16k on,
  because it skips the dense SDPA's score materialization.
- **Index.** "Index B/token" is the bucket cache (host tables plus the device mirror) after the prefill, over the
  16 layers: about 1.6–2.3 KiB a token, 404 MiB at 256k.
- **Snapshots.** The D-LSA tables add exactly 648 B/token to a snapshot at every length (dense 66–91 KB/token,
  mostly fp16 KV), about 1%, as §13.2 predicted.
- **PCA map.** `metal-pca` prefill equals `metal` within 0.1% (32k: 236.0, 64k: 232.7, 128k: 229.2 tok/s). It
  runs the same kernels on the same shapes; only the lists differ.

**Crossover.**

- **End to end**, D-LSA (with sparse reads from 4,612 on) overtakes dense between 16k (0.986) and 24k (1.010),
  at about 20k tokens.
- **Per chunk** it happens earlier. The chunk times of the 32k runs, metal / dense, by chunk start:
  - 4.6k: 1.056; 6.1k: 1.044; 7.7k: 1.032; 9.2k: 1.021; 10.8k: 1.008;
  - 12.3k: 0.995; 13.8k: 0.978; 15.4k: 0.949; 24.6k: 0.883; 27.6k: 0.868.
  - So a 1536-row chunk wins once it starts at about 12k keys, as task 4 estimated from the 1a microbench
    (§14.3).
- Below that, every sparse chunk pays K2's fixed ~38 ms per layer, where dense SDPA is cheaper. The short
  prompts (8k–16k) lose 1.4–2.4% because they are mostly such chunks.

**Decomposition** (Metal, the 2 fenced chunks at the end of the prefill, ms per full-attention layer, first
document). The categories:

- indexing/clustering: index key and query projections, key readback, Ward, centroid sync;
- scoring/selection: score matmul, K1;
- attention read: K2;
- other attention: input, q/k/v projections, RoPE, KV fetch, gate and o_proj.

| context | indexing / clustering | scoring / selection | attention read (K2) | other attention | full-attn layer incl. MLP | dense full-attn layer | linear-attn layers, % of forward | fenced forward s, dense / metal |
|---|---|---|---|---|---|---|---|---|
| 8k | 2.1 | 1.3 | 38.3 | 27.9 | 132.1 | 109.3 | 68% | 6.32 / 6.65 |
| 16k | 2.3 | 1.5 | 38.3 | 29.0 | 133.8 | 134.9 | 68% | 6.71 / 6.69 |
| 32k | 2.8 | 1.7 | 38.6 | 31.3 | 137.6 | 202.1 | 67% | 7.80 / 6.77 |
| 64k | 2.4 | 2.0 | 39.3 | 36.5 | 143.5 | 320.1 | 66% | 9.69 / 6.85 |
| 128k | 2.3 | 2.8 | 40.5 | 45.3 | 153.7 | 827.6 | 65% | 17.81 / 7.03 |
| 256k | 2.9 | 7.7 | 39.4 | 68.0 | 183.1 | 1919.6 | 61% | 35.30 / 7.54 |

What the decomposition shows:

- **Selection is cheap.** Indexing plus selection is 3.4–10.6 ms per layer: 0.8–2.3% of the forward.
- **K2 is flat in T** at 38–40 ms per layer.
- **The growing part is the KV fetch.** Inside "other attention", the fetch rises from 1 ms (8k) to 36 ms (256k)
  per layer.
- **The linear layers dominate.** The 48 linear-attention (Gated DeltaNet) layers are 61–68% of every D-LSA
  forward, about 4.5 s of a 6.7–7.5 s chunk. They are untouched by D-LSA and bound throughput at about 230
  tok/s at any length.

### 15.3 Decode (baseline greedy)

32 unfenced greedy steps after the tail. Each cell is the median step, then the median over documents.

| context | dense ms/token | D-LSA ms/token | metal / dense |
|---|---|---|---|
| 4k | 36.9 | 37.1 | 1.00 |
| 8k | 37.8 | 39.0 | 1.03 |
| 12k | 38.2 | 38.9 | 1.02 |
| 16k | 38.9 | 39.1 | 1.00 |
| 24k | 39.9 | 38.8 | 0.97 |
| 32k | 41.1 | 38.9 | 0.95 |
| 64k | 45.9 | 39.2 | 0.85 |
| 128k | 55.2 | 39.5 | 0.72 |
| 256k | 86.6 | 39.9 | 0.46 |

- D-LSA decode is flat (37–40 ms). Dense grows with T.
- The decode crossover is about 16k. Below it D-LSA costs up to 3% (8k), which `dense_below` removes (§15.7).

### 15.4 Interactions

**Prefix-cache hit.**

- After the prefill, the bench snapshots the state (`snapshot_sequence_state`, as the server's prefix cache does).
- For each needle it then restores the snapshot, prefills a question of about 25 tokens, and reads the first
  logits. That is the TTFT of a second request sharing the whole prompt as a prefix.
- Medians over 5 needles × documents:

| context | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| hit TTFT dense, ms | 156 | 169 | 191 | 245 | 335 | 545 |
| hit TTFT D-LSA, ms | 157 | 170 | 173 | 183 | 202 | 241 |
| restore dense / D-LSA, ms | 3 / 5 | 3 / 5 | 3 / 6 | 6 / 6 | 4 / 8 | 4 / 10 |

- The restore, which rebuilds the D-LSA tables, adds 2–6 ms.
- The question's forward is where D-LSA wins: a short forward at depth T costs dense O(T), while D-LSA reads
  4,612 keys. A prefix hit at 128k answers 2.3× sooner under D-LSA.
- Task 5's server smoke (§13.7) covers the MTP and DFlash2 prefix-cache paths.

**SpecPrefill.** SpecPrefill, a sparse-prefill mode that kept KV slots ≠ positions and was therefore always refused
under D-LSA, has since been removed from QW. There is nothing to measure.

**Speculative decoders.**

- `Qwen35MtpGenerator` (block 3) and `Qwen35Dflash2Generator` from a fresh state, on the same prompts, document 0
  only.
- 64 greedy tokens with EOS suppressed: the prompts end mid-file, and an early EOS otherwise ended some runs after
  one token, dense as well as D-LSA.
- The prefill column is the generator's own prefill, a second, independent TTFT sample.

| context | decoder | prefill s, dense / D-LSA | decode ms/token, dense / D-LSA | acceptance, dense / D-LSA |
|---|---|---|---|---|
| 8k | MTP | 30.8 / 31.5 | 19.0 / 19.3 | 0.909 / 0.909 |
| 8k | DFlash2 | 30.6 / 31.3 | 24.9 / 24.6 | 0.662 / 0.705 |
| 16k | MTP | 63.6 / 64.5 | 27.1 / 26.4 | 0.630 / 0.686 |
| 16k | DFlash2 | 63.2 / 64.1 | 37.4 / 36.5 | 0.370 / 0.382 |
| 32k | MTP | 144.0 / 138.3 | 22.6 / 21.3 | 0.932 / 0.911 |
| 32k | DFlash2 | 142.8 / 136.6 | 28.3 / 25.5 | 0.726 / 0.754 |
| 64k | MTP | 328.6 / 283.7 | 28.5 / 23.1 | 0.932 / 0.955 |
| 64k | DFlash2 | 324.7 / 277.3 | 31.7 / 27.6 | 0.783 / 0.778 |
| 128k | MTP | 971.6 / 598.7 | 49.4 / 36.2 | 0.812 / 0.812 |

What the decoders show:

- **Acceptance is not hurt.** Across runs it moves by at most ±0.06, both ways. The drafts only see what the
  target accepted, and D-LSA's target logits stay close to dense (§15.5).
- **Speculative decode gains like baseline decode.** D-LSA saves 13–27% per token from 32k on (MTP at 128k:
  36.2 vs 49.4 ms) and is within noise below.
- **DFlash2 is never faster than MTP here.** On these code prompts MTP wins at every length; that is a property of
  the drafter, not of D-LSA.
- The DFlash2 runs at 128k were skipped for time.

**Multi-row forwards.** The teacher-forced tail (256 rows in two 128-row forwards, the shape of a long verify or
prefix continuation) took, dense / D-LSA: 1.18 / 1.24 s at 8k, 1.45 / 1.27 s at 32k, 2.38 / 1.27 s at 128k,
5.38 / 1.37 s at 256k.

### 15.5 Quality sanity (not RULER)

**Teacher-forced tail.**

- After the prefill, the next 256 file tokens are teacher-forced in 128-row forwards; the 255 rows with a known
  next token are scored.
- Per row: KL(dense ‖ D-LSA) over the full vocabulary, top-1 agreement with dense, and the perplexity ratio.
- The cells list each document (doc 0 / doc 1 / doc 2, by offset 0 / 65,536 / 131,072):

| context | ppl ratio | mean KL, nats | max KL | top-1 agreement |
|---|---|---|---|---|
| 4k | 1.000 / 1.000 / 1.000 (bitwise dense) | 0 | 0 | 1.000 |
| 8k | 1.002 / 0.998 / 1.002 | 0.0020 / 0.0010 / 0.0009 | 0.44 | 0.984–0.996 |
| 12k | 1.002 / 1.000 / 0.999 | 0.0005 / 0.0011 / 0.0023 | 0.16 | 0.988–1.000 |
| 16k | 0.997 / 0.995 / 1.003 | 0.0021 / 0.0014 / 0.0024 | 0.05 | 0.980–0.996 |
| 24k | 0.998 / 0.997 / 0.998 | 0.0016 / 0.0083 / 0.0046 | 0.66 | 0.961–0.992 |
| 32k | 1.004 / 1.007 / 0.996 | 0.0036 / 0.0054 / 0.0031 | 0.18 | 0.969–1.000 |
| 64k | 1.018 / **1.082** / 1.012 | 0.0086 / **0.508** / 0.0119 | **24.9** | 0.973 / **0.898** / 0.969 |
| 128k | 1.036 / — / **0.822** | 0.0178 / — / **0.465** | **21.2** | 0.961 / — / 0.933 |
| 256k | 1.002 | 0.0054 | 0.16 | 0.988 |
| PCA map, doc 0: 32k / 64k / 128k | 1.004 / 1.020 / 1.038 | 0.0047 / 0.0086 / 0.0214 | 0.30 | 0.976 / 0.984 / 0.961 |

**Needles.**

- Five lines "The secret code of the {amber, cobalt, crimson, jade, violet} vault is {4821-KESTREL, …}." are
  written at 10/30/50/70/90% depth. Each question ("Question: What is the secret code of the amber vault?
  Answer: The secret code of the amber vault is") is answered greedily from a restored snapshot (12 tokens).
- A needle counts when the answer starts with the code.
- Dense and D-LSA found 15/15 at every context from 4k to 32k. No needles at 256k: there is no snapshot there,
  for memory.
- At 64k doc 1, dense itself found 0/5 and D-LSA 1/5. Both switched into chat mode (`<|im_end|><|im_start|>…`,
  a raw-completion prompt over a corpus full of chat-template code). Docs 0 and 2 were 5/5 for both.
- At 128k doc 0 both found 5/5. At **128k doc 2, dense found 5/5 and D-LSA 2/5**: the amber (10% depth), jade
  (70%) and violet (90%) answers left for chat mode.
- The PCA map found 5/5 on doc 0 at 32k, 64k and 128k.
- Answers beyond the code (the 12-token text) equal dense in 15/15 runs at 4k–8k and in 9–14/15 at 12k–64k.

**Reading.**

- No gross breakage: no 4-bit interaction of the kind that would wreck every prompt.
  - Up to 32k every document stays within 1% perplexity, mean KL under 0.01 nats and top-1 at 0.96 or above.
  - 256k doc 0 is equally close.
- But **two of the five long documents (64k doc 1, 128k doc 2) diverge clearly**: mean KL about 0.5 nats, some
  rows above 20 nats, top-1 0.90–0.93. On one of them D-LSA misses needles that dense finds.
- Doc 0 also drifts slowly with length (ppl ratio 1.004 → 1.018 → 1.036 at 32k / 64k / 128k): top_k = 4096
  covers a shrinking share of the context.
- The PCA step-0 map is no worse than the distilled one on these documents. That is unexpected and needs RULER
  (task 7) to judge; §16 ran the distilled map only.

### 15.6 Divergence diagnostics

The two divergent documents were rerun with task 3's `mlx` backend added. That backend does a host pick and
MLX's masked SDPA, and it is the oracle the Metal kernels were checked against (§14.2). Same prompts, needles and
tail; `QW_DLSA_DENSE_BELOW=0`.

| document | mode | prefill tok/s | tail ppl | mean KL | max KL | top-1 | rows with KL > 1 | needles |
|---|---|---|---|---|---|---|---|---|
| 64k doc 1 | dense | 198.0 | 4.541 | — | — | — | — | 0/5 |
| | mlx | 193.7 | 4.774 | 0.508 | 26.1 | 0.898 | 17 of 255 | 1/5 |
| | metal | 232.8 | 4.913 | 0.508 | 24.9 | 0.898 | 16 of 255 | 1/5 |
| 128k doc 2 | dense | 138.7 | 3.091 | — | — | — | — | 5/5 |
| | mlx | 154.1 | 2.616 | 0.482 | 21.7 | 0.929 | 14 of 255 | 2/5 |
| | metal | 229.6 | 2.540 | 0.465 | 21.2 | 0.933 | 13 of 255 | 2/5 |

- **Not the Metal kernels.** `mlx` and `metal` diverge from dense on the same rows (row indices agree, KL within
  about 10%) and miss the same needles (amber, jade and violet at 128k).
  - The Metal runs reproduce the main runs' numbers exactly; the bench is deterministic.
  - So the cause is shared by both paths: the selection itself (index projections through the bf16 map, Ward
    buckets, centroid scores, top_p/top_k). That is either the algorithm's behavior on these documents with this
    4-bit checkpoint, or a defect in the shared index code. The golden fixtures (tasks 0–2) only check that code
    at fixture sizes, not at 64k–128k.
- **Where it hits.** The divergence is concentrated: 13–17 rows of 255 carry nearly all the KL.
  - The 64k ones are identifier continuations in code (`pool_k`, `pool_v`, `PagedDecodeOutcome` after `Paged`,
    `launched`): tokens a dense model copies from earlier occurrences.
  - The other rows agree closely, which suggests the copy source is missing from the selected list, not that
    reads are wrong in general.
- **Perplexity can move either way.** At 128k D-LSA's tail perplexity is *lower* than dense (2.54 vs 3.09), while
  top-1 agreement is only 0.93. Perplexity ratios alone would not have flagged this document.
- **Follow-up for task 7.** This is a candidate for task 7 to study, together with RULER's retrieval tasks. Also
  worth checking there: a run with the bf16 checkpoint, if one fits in memory, to separate the 4-bit interaction
  from the algorithm, and the upstream reference on one of these documents. (§16 ran RULER at 4k and 32k only;
  these documents remain open.)

### 15.7 Decision and the `dense_below` threshold

**Recommendation: keep D-LSA opt-in (`--dlsa`), and when it is on, read sparse only from 12,288 keys on
(implemented: `dense_below`, default 12,288).** Not default-on:

- **Accuracy at scale is unverified.** Two of five long documents diverge (§15.5–15.6), and RULER (task 7) has
  not run. (§16 has since run it at 4k and 32k: no task degrades at 32k, cwe +0.010; 64k and longer remain
  unmeasured.)
- **The KV setting conflicts.** D-LSA needs `--no-kv-quantization`, while the server's default KV cache is
  TurboQuant, which D-LSA refuses (§13.1). Default-on would silently change the KV mode, or fail.
- **Short prompts gain nothing.** Prefill gains only from about 20k tokens; most requests are shorter.

**The threshold.**

- `DlsaConfig::dense_below` (`--dlsa-dense-below <TOKENS>`, `QW_DLSA_DENSE_BELOW`, default 12,288, 0 = the old
  rule) is a **per-row** policy (`DlsaConfig::dense_rows`): a query row at a position below it reads every
  causal key, a row at or past it its sparse list. So a verify block or a chunk reads each row as single-token
  forwards would, and a rollback or a snapshot restore never depends on how the rows were batched. The index is
  built throughout, so later forwards read sparse lists as before.
- **Semantics.**
  - A forward with every row below the threshold (or on the dense key set) runs the unchanged dense attention:
    bitwise dense.
  - A forward that crosses it splits its rows. The rows below run `causal_attention`, SDPA over the fetched
    cache under an additive causal mask, which equals dense attention up to rounding, not bitwise. The rows past
    it select and read as before (K2, or the task-3 masked/gathered read).
  - Multi-row and single-row reads already round differently under D-LSA (§13.3). The tests hold block vs single
    tokens to the same 1e-4 tolerance, with greedy tokens equal.
- 12,288 is the measured per-chunk crossover (§15.2): the first chunk that starts at 12,288 is 0.5% faster
  sparse, and the one before, at 10,752, is 0.8% slower.
- It also covers the decode crossover (about 16k) to within 1 ms.
- It is part of the prefix-cache identity (`dense_below=` in `dlsa_identity`) and of the `D-LSA armed` log line.
- With the feature off nothing changes; dense namespaces are unaffected.
- **The measurements below are unaffected by the per-row split.** No measured forward crossed 12,288:
  - the prompts are whole 1536-row chunks, and 12,288 = 8 · 1536;
  - decode is single-row;
  - the teacher-forced tails sit wholly below or wholly above the threshold;
  - the speculative decoders were not run with the threshold.
  The split only adds an SDPA over at most one verify block or chunk per sequence.
- Tests (each fails with a per-forward decision, shown by reverting `dense_rows`, except the restore test, which
  checks consistency):
  - `qwen3_next::dlsa::tests::dense_below_runs_dense_until_the_threshold` (layer, all three backends): forwards
    below the threshold are bitwise dense. A crossing forward's rows below match dense within 1e-5 and its other
    rows read sparse; `sparse_rows` counts only those.
  - `qwen3_5::dlsa_serving_tests::dense_below_chunk_matches_single_tokens_across_the_threshold` (model, all
    backends): a chunk crossing the threshold vs single tokens, logits within 1e-4, greedy tokens and index
    tables equal.
  - `…::dense_below_verify_and_rollback_across_the_threshold`: 3-row (MTP) and 8-row (DFlash2) verifies with
    seeded rollbacks through the threshold vs the sequential run. Logits, greedy tokens, tables and a 20-token
    continuation all agree.
  - `…::dense_below_snapshot_restore_across_the_threshold`: snapshots below and past the threshold restore and
    continue through it bitwise like the uninterrupted run.
  - `dlsa::options::tests::dense_below_comes_from_the_flag_then_the_variable`, the identity test, and the
    `qw-server` and `qw-cli` flag parsing and help tests.
- The other synthetic tests and the real-model tests R1–R9 set `dense_below: 0`, so they keep exercising the
  sparse path at their (short) lengths.

Measured with the threshold (document 0, same prompts and needles):

| context | dense tok/s | D-LSA tok/s | ratio | decode dense / D-LSA, ms | tail mean KL | needles |
|---|---|---|---|---|---|---|
| 8k | 249.8 | 248.5 | 0.995 | 37.7 / 37.8 | 0 (bitwise) | 5/5 |
| 12k | 247.1 | 245.8 | 0.995 | 38.2 / 38.3 | 0 (bitwise) | 5/5 |
| 16k | 242.7 | 242.1 | 0.998 | 39.0 / 39.1 | 0.0021 | 5/5 |
| 24k | 234.9 | 239.1 | **1.018** | 40.0 / 39.0 | 0.0017 | 5/5 |
| 32k | 225.6 | 237.3 | **1.052** | 41.2 / 38.9 | 0.0036 | 5/5 |

Against the main runs:

- **Short prompts.** The 2.2–2.4% loss at 8k–12k shrinks to 0.5%, which is the index build (about 2.3 ms per
  layer and chunk) that later sparse forwards need.
- **Longer prompts.** 24k and 32k gain a little more (1.018 vs 1.010, 1.052 vs 1.047), since their first 8 chunks
  no longer pay K2.
- **Decode** at 8k–16k now equals dense.

**What would still pay off** (in order):

1. **The linear-attention layers.** They are 61–68% of a D-LSA prefill forward and cap prefill near 230 tok/s.
   This is dense code outside D-LSA; it is the only route to large further prefill gains.
2. **The KV fetch** (`update_and_fetch`). It grows to 36 ms per layer at 256k, now the largest growing D-LSA
   stage. K2 could read the cache in place without the fetched view's copy.
3. **K2 for short sparse depths.** About 38 ms per layer regardless of list length or depth. A kernel that is
   cheaper below about 12k keys would let `dense_below` drop, and push the crossover down. A dense D = 256
   flash kernel would also shrink the dense baseline's lead there (§14.3).
4. **Not worth it:** K1, Ward and the score matmul. Together they are under 2.5% of the forward at every length
   (G5/G6 stay pruned).

### 15.8 Reproducing

Build outside the GPU lock as in §14.6, then one context per lock hold:

```text
MLXCEL_CACHE_LIMIT=<8 GiB up to 64k | 4 GiB at 128k | 1 GiB at 256k> QW_DLSA_BENCH_CONTEXTS=<T> \
  QW_DLSA_BENCH_MODES=dense,metal[,metal-pca] QW_DLSA_BENCH_OFFSET=<0 | 65536 | 131072> \
  QW_DLSA_BENCH_TAIL=256 QW_DLSA_BENCH_NEEDLES=1 [QW_DLSA_BENCH_SPEC=mtp,dflash2] [QW_DLSA_DENSE_BELOW=0] \
  QW_DLSA_BENCH_PROMPT=<token file> QW_DLSA_BENCH_OUT=<unique>.json \
  ./gpu-lock -- target/release/deps/qw_runtime-<hash> qwen3_5::dlsa_bench::real_dlsa_bench --exact --ignored \
  --nocapture --test-threads=1
python3 eval/dlsa/task6/summarize.py <directory of reports>
```

Notes:

- No needles at 256k: the snapshot would add 16 GiB next to a 34 GiB peak.
- The report's `dlsa_stats` counts only the last needle's forwards (each restore starts a fresh sequence). The
  prefill's index bytes are recorded separately.

## 16. RULER accuracy (roadmap task 7)

This section runs upstream's RULER protocol (13 tasks, 4k and 32k) against `qw serve`, dense and with `--dlsa`, on a
subsample of the rows. The harness and the protocol are in
[`eval/dlsa/ruler/README.md`](../eval/dlsa/ruler/README.md); every number below comes from the result files in
[`eval/dlsa/ruler/results/task7/`](../eval/dlsa/ruler/results/task7/): `summary.md` and `summary.json`
(`summarize.py`), the per-row records `dense/*.jsonl` and `dlsa/*.jsonl` (with the reply text), and
`server-log-{dense,dlsa}.json` (`server_log_stats.py`, per request). The upstream reference is README.md:180 at
`5ebea73`: Qwen3.8-27B D-LSA (Flat), bf16 on one H200, 100 rows per task, 0.974 at 4k and 0.968 at 32k (the paper's
columns: 0.975 / 0.966).

### 16.1 Setup

- **Machine.** Apple M4 Max (64 GB), 2026-11-06.
- **Build.** `cccbf65` (this task's base; also the harness's `git describe` in each `meta.json`), release,
  `--features dlsa`.
- **Model.** `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` (4-bit oQ4e), fp16 KV (`--no-kv-quantization`), decoder `auto`. Auto
  chose DFlash2 (drafter `incoai/Qwen3.8-27B-DFlash2`) for every request of both servers: 546 of 546
  `decoder.selected` events, route `dflash2` in every `server-log-*.json` entry.
- **Servers.**
  - Dense: `qw serve --no-kv-quantization`.
  - D-LSA: `qw serve --dlsa --no-kv-quantization`, the served defaults (§13.4, §15.7). Its `D-LSA armed` line: the
    distilled map (sha256 `00200004…62e1`), bf16, backend `metal`, sink 4, window 512, top_k 4096, top_p 1024,
    short context `DenseFallback`, `dense_below` 12,288.
  - Both: `MLXCEL_CACHE_LIMIT` 8 GiB, prefix cache on as shipped. It reused nothing at 32k (`cached_tokens` 0 in every
    row) and 15 tokens per row on average in the dense 4k run (0 in the D-LSA 4k run).
- **Where D-LSA is active.** The 4k prompts (mean 3,716 tokens) are all below 4,612, so every forward reads the dense
  key set: the server log shows 0 sparse forwards in all 26 D-LSA 4k requests. The 32k prompts (mean 32,026 tokens)
  read sparse from position 12,288 on. The log's `sparse_forwards` counts full-attention layer forwards (16 per model
  forward, `DlsaStats` in `crates/runtime/src/dlsa/runtime.rs`), prefill chunks and DFlash2 verify forwards alike:
  128 dense-key-set forwards (8 model forwards) in every 32k request, and 192–18,208 sparse ones, from a
  1-token reply to a 4,096-token cwe reply with 1,124 verify rounds. The index is 54–75 MiB per request (mean 56).
- **Data.** Upstream's RULER sets for Qwen3.8-27B, regenerated with upstream's generators (`prepare_ruler.py`:
  NVIDIA/RULER @ `c3f5e3b4` with upstream's patch, seed 42, tokenizer `Qwen/Qwen3.8-27b`, tokenizers 0.23.2):
  13 tasks × {4096, 32768} × 100 rows. All 30 pinned prompt-set hashes of upstream's `hashes_qwen38_27b.json`
  match (`prepare_ruler.py --check-hashes`: 30/30).
- **Protocol.** Upstream's (eval/dlsa/ruler/README.md): one user message, `input` + `gen_prefix` (with a single
  space only where the seam has none), greedy (temperature 0, top_p 1), max_tokens 4096,
  `chat_template_kwargs.enable_thinking=false`, substring scoring (`string_match_all`; `string_match_part` for
  qa_squad and qa_hotpot), replies cut at max_tokens scored as they are, the length's score the mean of the 13 task
  scores.
- **Rows.** The first N rows of each file (an N-row set is a prefix of the 100-row set, so a larger run later reuses
  these rows):

| config | 4k rows per task | 32k rows per task | GPU time |
|---|---|---|---|
| dense | 20 (260 rows) | 10 (130 rows) | 6.8 h (4k 1.45 h, 32k 5.35 h) |
| D-LSA | 2 (26 rows, an identity check) | 10 (130 rows) | 5.3 h (4k 0.15 h, 32k 5.12 h) |

- **Why a subsample.** The full protocol is 100 rows per task. On this machine a 32k row takes about 145 s and rows
  share no prefix, so 100 rows cost about 7 h at 4k and 52 h at 32k per configuration, about 110 h for dense and
  D-LSA at both lengths. The user chose this scope (dense 20/10 rows, D-LSA 10 rows at 32k plus 2 at 4k) out of three
  offered; no PCA map and no 64k.
- **Statistical power.** At 10 rows a task score moves in steps of 0.1, and its binomial standard error is up to
  about 0.16. The worst case for the 13-task mean is about 0.04. On these rows, where only cwe, qa_squad and
  qa_hotpot vary, the mean's empirical standard error is about 0.014 at 32k for both servers. The paper publishes
  no dense RULER number as text (its Figure 11 shows dense and D-LSA within a point or two of each other at 4k–32k),
  and a gap of that size is below either error. So the run can detect gross degradation only, and the sensitive
  instrument is the row-paired comparison: the same rows under both servers (§16.3).

### 16.2 Scores

Task scores; n is the number of rows. "paired" means the rows both servers finished (at 32k all 10; at 4k the first
2), where dense's 4k paired mean is 0.954.

| task | 4k dense (20) | 4k D-LSA (2) | 32k dense (10) | 32k D-LSA (10) |
|---|---|---|---|---|
| niah_single_1/2/3, niah_multikey_1/2/3, niah_multiquery, niah_multivalue | 1.000 each | 1.000 each | 1.000 each | 1.000 each |
| vt | 1.000 | 1.000 | 1.000 | 1.000 |
| cwe | 0.945 | 0.900 | 0.980 | **0.990** |
| fwe | 1.000 | 1.000 | 1.000 | 1.000 |
| qa_squad | 0.900 | 1.000 | 0.800 | 0.800 |
| qa_hotpot | 0.650 | 0.500 | 0.800 | 0.800 |
| **mean** | **0.961** | 0.954 (= dense on the same rows) | **0.968** | **0.968** |
| upstream D-LSA bf16 (README.md:180) | 0.974 | | 0.968 | |
| difference to upstream | −0.013 | | −0.000 | +0.000 |

- **32k.** No task degrades under D-LSA: scores are identical on 12 of 13 tasks, and cwe is +0.010 (0.980 → 0.990,
  two rows changing, one each way, §16.3). The means are 0.968 for both (paired delta +0.001), equal to the upstream
  D-LSA mean to three digits. At this row count that agreement with upstream is a property of the sample, not a
  measured match.
- **4k.** Dense reads 0.961, 0.013 below upstream's 0.974. D-LSA plays no part in that gap: upstream's 4k run reads
  the dense key set too (§4.7), and on the paired check D-LSA was inactive here (0 sparse forwards, bitwise dense):
  its 2 rows per task score exactly as dense on the same rows (0.954 both) and every reply is byte-identical to dense
  (26 of 26). What the −0.013 is made of stays unresolved. It compares QW's first 20 rows per task with upstream's
  100-row aggregate, on the 4-bit oQ4e checkpoint and QW against bf16 on upstream's SGLang fork and an H200; the row
  sample alone could explain it, and there is no paired bf16 baseline. Separating quantization from the sample and
  the stack would take the same rows on the bf16 checkpoint, which was not run (27B bf16 weights alone are about
  54 GB on this 64 GB machine).

### 16.3 Row-level agreement (32k, 130 rows)

| task | rows whose score differs | rows whose reply text differs |
|---|---|---|
| niah_single_2 | 0 | 1 |
| niah_single_3 | 0 | 3 |
| niah_multikey_2 | 0 | 3 |
| niah_multikey_3 | 0 | 4 |
| niah_multiquery | 0 | 1 |
| vt | 0 | 2 |
| cwe | **2** (row 3: 1.0 → 0.9; row 5: 0.8 → 1.0) | 10 |
| niah_single_1, niah_multikey_1, niah_multivalue, fwe, qa_squad, qa_hotpot | 0 | 0 |
| **total** | **2 of 130**, one each way | **24 of 130** |

- **Scores.** Only two rows change score, both cwe, one down and one up. In row 3 both replies were cut at
  max_tokens. In row 5 dense's reply was cut and D-LSA's finished.
- **The 14 non-cwe text differences** are phrasing or formatting around the same correct answer:
  - "mentioned in the text" vs "mentioned in the provided text" (niah_single_2 row 7, niah_single_3 rows 1 and 7);
  - a colon and line break before the uuid (niah_single_3 row 6);
  - a bold vs a plain uuid (niah_multikey_3 rows 2, 3, 5, 9; D-LSA plain);
  - a bare number (dense) vs a full sentence (D-LSA) in niah_multikey_2 rows 2, 5, 9;
  - the full sentence (dense) vs the bare variable list (D-LSA) in vt rows 2 and 7;
  - a shorter list heading in niah_multiquery row 7.
- **cwe differs in all 10 rows.** The model answers cwe with a long word-counting transcript. Under D-LSA it takes a
  different route from the first sentence on: every pair parts at character 42 ("we must count" vs "we first
  count", row 5 "we count"). cwe prompts are a random word stream, where the selected key set plausibly changes the
  hidden states even when the answer does not; the scores agree in 8 of the 10 rows.
- **qa_squad and qa_hotpot**: identical replies in all 20 rows.
- **cwe runaway.** cwe replies were cut at max_tokens in 11 of 20 dense 4k rows, 1 of 2 D-LSA 4k rows, 8 of 10
  dense 32k rows and 8 of 10 D-LSA 32k rows (`summary.md`, "Errors and replies cut at max_tokens"); no other task
  was ever cut, and no row errored. They are scored as upstream does, by substring over the cut reply. Such a row
  decodes 4,096 tokens: a cut row takes 91 s on average at 4k (the 4k median row is 16.2 s, hence the mean of
  20.1 s) and 219–237 s at 32k (the median is 137–145 s).
- **Against §15.5–15.6.** Nothing like the two divergent documents shows up on RULER at 32k. Those were at 64k and
  128k, where top_k 4096 covers a smaller share of the context, and they remain unexplained here: 64k was not run.

### 16.4 Decoder and timing

Per request from `server-log-*.json`, joined to the rows by response id. Medians over requests; acceptance is
accepted over proposed draft tokens summed over the length's requests.

| | 4k dense | 4k D-LSA | 32k dense | 32k D-LSA |
|---|---|---|---|---|
| requests | 260 | 26 | 130 | 130 |
| prefill tok/s (median) | 253.8 | 252.6 | 226.5 | 238.3 (1.05×) |
| decode tok/s (median, DFlash2) | 41.4 | 38.3 | 31.6 | 35.0 |
| row latency s, mean / median | 20.1 / 16.2 | 21.0 / 16.3 | 148.2 / 144.7 | 141.7 / 137.5 |
| DFlash2 acceptance, all rows | 0.918 | 0.926 | 0.850 | 0.857 |
| acceptance, rows with ≤ 200 decoded tokens | 0.806 (240) | 0.793 (24) | 0.728 (120) | 0.737 (120) |

- **Prefill** at 32k gains 1.05×, as §15.2 measured on the bench (1.047).
- **Acceptance** is unchanged by D-LSA, as §15.4 found: within 0.013 at both lengths, with and without the cwe
  runaways (the 4k D-LSA figures are 26 rows). Together with §16.3 this shows that the multi-row verify under D-LSA
  does not change what is scored.
- **Cost of the full protocol** (`summarize.py --extrapolate 100`, each task's mean row time × 100): 7.3 h (dense)
  and 7.6 h (D-LSA) at 4k, 53.5 h and 51.2 h at 32k, per configuration.

### 16.5 Reading and decision

- **At 32k no RULER task degrades under D-LSA in this sample.** On the 4-bit checkpoint with the served defaults:
  identical scores on 12 of 13 tasks, cwe +0.010; paired delta +0.001; 2 of 130 rows change score (both cwe, one each
  way); 106 of 130 replies identical and the other non-cwe ones only reworded. The mean equals upstream's bf16 D-LSA
  mean.
- **At 4k dense is 0.013 below upstream**, with D-LSA inactive (bitwise dense). Whether that is the 4-bit checkpoint,
  the 20-row sample or the serving stack is not resolved (§16.2).
- **Limits.**
  - At 10 rows per task the run cannot resolve differences in the mean below about 0.03.
  - It says nothing about 64k–256k, where §15.5–15.6 saw two documents diverge.
  - The distilled map only; the PCA map was not run.
- **Decision: keep D-LSA opt-in.** Accuracy is no longer the reason against enabling it at contexts up to 32k. The
  other blockers of §15.7 are untouched: D-LSA needs `--no-kv-quantization` while the server defaults to TurboQuant,
  and prompts below about 20k gain nothing.
- **Before any default-on:** RULER at 64k and 128k, the two divergent documents of §15.6, and the full 100 rows at
  32k. A 100-row 32k run costs about 52 h per configuration; the harness resumes, so the rows here count toward it
  (raise `--rows` on the same `--out`).

### 16.6 Reproducing

The RULER data is not committed (167 MB for the 26 files); regenerate it with `prepare_ruler.py` and check it with
`--check-hashes` (eval/dlsa/ruler/README.md). One server at a time, through the GPU lock. The debug filter makes
the DFlash2 statistics and the per-request D-LSA counters appear in the stdout log, which `server_log_stats.py`
reads:

```text
RUST_LOG=info,qw_server::engine=debug,qw_runtime::provider=debug MLXCEL_CACHE_LIMIT=8589934592 \
  ./gpu-lock -- qw serve [--dlsa] --no-kv-quantization --bind 127.0.0.1:8011 --no-file-logging \
  --output-format json > serve-<dense|dlsa>.log 2>&1

# dense server
python3 eval/dlsa/ruler/run_ruler.py --base-url http://127.0.0.1:8011 --data-root DATA --out results/dense \
  --label dense --lengths 4096 --rows 20 --note "..."
python3 eval/dlsa/ruler/run_ruler.py --base-url http://127.0.0.1:8011 --data-root DATA --out results/dense \
  --label dense --lengths 32768 --rows 10 --note "..."
# D-LSA server
python3 eval/dlsa/ruler/run_ruler.py --base-url http://127.0.0.1:8011 --data-root DATA --out results/dlsa \
  --label dlsa --lengths 32768 --rows 10 --note "..."
python3 eval/dlsa/ruler/run_ruler.py --base-url http://127.0.0.1:8011 --data-root DATA --out results/dlsa \
  --label dlsa --lengths 4096 --rows 2 --note "4k identity check"

python3 eval/dlsa/ruler/summarize.py --runs dense=results/dense dlsa=results/dlsa --extrapolate 100 \
  --json results/summary.json > results/summary.md
python3 eval/dlsa/ruler/server_log_stats.py serve-dense.log --json results/server-log-dense.json
python3 eval/dlsa/ruler/server_log_stats.py serve-dlsa.log --json results/server-log-dlsa.json
```

Each run's `meta.json` records the invocations (rows, lengths, server URL, note, harness commit) and each data
file's sha256 and prompt-set hash. The split of the server-log numbers by length in §16.4 joins each log entry's
`id` to the rows' `response_id`.
