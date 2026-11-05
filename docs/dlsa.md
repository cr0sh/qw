# D-LSA (flat design) for Qwen3.8-27B: spec for the QW port

Status: research spec (roadmap task 0 of 6). There is no QW code yet. Golden fixtures are in
[`tests/fixtures/dlsa/`](../tests/fixtures/dlsa/README.md).

D-LSA (Distilled Latent Sparse Attention, arXiv 2610.42147) leaves the attention kernel unchanged. It only decides which
cached keys each query of a full-attention layer reads: the first `sink_size` keys, the last `window_size` keys, and up
to `top_k` long-range keys picked through a small learned index. This document specifies the **flat design**
(`top_level: 0`), the one served for Qwen3.8-27B, closely enough to implement it. It also maps the design onto QW.

**Source.** <https://github.com/Prefrontal-AI/latent-sparse-attention>, Apache-2.0, pinned at commit
**`5ebea732d639576d2a7579b0de40e6252f9001f5`** (2026-11-04). Citations `path:line` refer to that commit:
`src/lsa/...` for upstream code and `configs/...` for upstream configs. QW citations refer to this repo at `782b61b`.
Code ported from upstream must carry `SPDX-License-Identifier: Apache-2.0` and an "Adapted from …@5ebea73" header, as
the SpecPrefill files do.

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
     chunks, decode, MTP verify, DFlash2 verify and SpecPrefill.
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
   - SpecPrefill keeps keys at cache slots ≠ positions (`qwen3_5/specprefill.rs:219-240`). D-LSA and SpecPrefill should be
     mutually exclusive.
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
6. **Feature gating** (as SpecPrefill does):
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
9. **Snapshot complexity** (§7.4) and interaction with SpecPrefill (positions ≠ slots). Exclude SpecPrefill.
10. **Accuracy evidence is upstream's.** RULER 0.968 at 32k is on H200 bf16 with YaRN 40. Nothing is verified on QW yet.

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
2. **Indexer core** (pure MLX ops, no Metal, no model):
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
   - Refuse MTP/DFlash/SpecPrefill when `dlsa` is on, as upstream does.
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
