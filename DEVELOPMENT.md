# Development notes

These notes cover shared local resources used while developing and measuring
QW. Dated memory measurements below identify their source and configuration;
performance results belong in a record with the same provenance.

## Cache layout

QW stores its user cache below `~/.cache/qw`:

- model snapshots are under `~/.cache/qw/models`;
- persistent prefix snapshots are under `~/.cache/qw/checkpoint` by default;
- `qw stats` reports total, model, and checkpoint cache usage plus resolver
  status.

The default prefix-snapshot tier capacities are 2 GiB in memory and 16 GiB on
the filesystem. These are snapshot-tier budgets, not process-memory limits.
The server accepts `--prefix-cache-memory-bytes`,
`--prefix-cache-filesystem-bytes`, and `--prefix-cache-directory` when a
workload needs different limits or a separate location.

New prompt checkpoints capture the full incoming model prompt, including the
rendered generation prefix, rather than history-only or adaptive intermediate
prefixes. Exact output, terminal, and cancellation checkpoints remain available.
A durable response continuation is available for unconstrained text only. It
preserves the response identity, delivered prefix, original penalty boundary,
remaining token budget, and independently saved RNG state; an unrelated request
must not change its continuation stream. Image and constrained requests use
ordinary prefix snapshots, not serialized parser or response continuations.

Ordinary memory and filesystem eviction prefer the least recently used or
materialized entry; reuse frequency breaks recency ties and still determines
TTL growth. Merely matching a shorter ancestor does not refresh its materialized
recency. A successful disk hit becomes recent before hot-tier promotion pressure.
Oversized ordinary snapshots still cannot displace viable hot prefixes, and
active continuation protections and tier budgets are unchanged.

Per-entry cache lifecycle traces carry the existing deterministic `entry_id`
derived from namespace, route, token prefix, and ordered image identities. The
identity survives hot or filesystem removal and persistence failure, including
memory-only entries.
Use it to join insert, hit, restore/promotion, eviction, and expiry evidence;
capacity setup, genuine misses, and aggregate events do not identify one entry.

The worker prepares the immediate next eligible text request already collected
in its FIFO batch and starts portable cache prefetch before current generation.
Successful prompt preparation is reused at that request's turn. Late arrivals,
response continuations, structured/multimodal requests, and history-dependent
sparse-prefill configurations retain ordinary lookup behavior. This is not GPU
batching or a guarantee that every boundary has prefetched state.

Prefetch performs bounded reads and portable validation/decoding on the I/O
thread; MLX materialization/restoration remains on the generation thread.
Immutable pages share ownership instead of being deep-copied. An owner-thread
hot pin protects lookahead from intervening publication eviction. Demand joins
a matching unfinished read, revalidates the candidate, and falls back normally
on stale, missing, corrupt, or cancelled work. Publication stays visible before
response completion; FIFO persistence ordering is preserved.

Additional staging reservations are bounded to four hot-tier capacities with a
64 MiB floor: 8 GiB with the default 2 GiB hot tier. Publication reserves twice
the logical snapshot bytes plus exact serialized manifest bytes; settled disk
lookahead reserves three times combined blob and manifest bytes; a hot pin
reserves logical snapshot bytes. Pending writes discover and reserve manifest
size before reading. Budget exhaustion skips optional work rather than growing
an unbounded queue. These are conservative payload reservations, not measured
RSS or a total-process limit: metadata, active model/request state, and ordinary
demand restoration are separate. Reservations release when ownership ends,
including cancellation/failure; active cancelled I/O retains its reservation
until it actually finishes.

Use `cache.prefetch` lifecycle and `cache.staging.reserve/release` traces to
verify overlap and bounded ownership; a queued hint alone is not proof of a
completed prefetch or a latency improvement.

Snapshot budgets do not bound request latency. Persistence still uses serialized
I/O; queued writes and large FP16 snapshots can delay a subsequent filesystem
lookup. Darwin durability barriers are batched, but this does not remove FIFO
queueing or make checkpoint publication free. Report lookup, uncached prefill,
decode, and terminal/publication timing separately.

## Long-context memory breakdown

**Advised operating ceiling: 36 GiB of process physical footprint**, including
model/request state, caches, host allocations, and allocator/driver residency.
This is a total-process target, not a 36 GiB allowance for each component and
not an enforced MLX or prefix-cache limit. The measured long-context physical
peaks below still exceed it. Bounded retention is not evidence that the 36 GiB
target has been met; do not raise the advised ceiling to match observed peaks.

### Ownership and accounting

| Component | What it contains | Budget / measurement |
|---|---|---|
| Live MLX allocations | Target/draft weights, active and restored KV, retained GPU snapshot pages, recurrent state, logits, and temporary workspaces | MLX active bytes; peak bytes include transient allocations missed by settled samples |
| Reusable MLX buffers | Freed buffers retained for later GPU work | Default 8 GiB allowance; separate from active bytes, and not a hard residency cap |
| Hot prefix snapshots | Unique GPU pages, unique retained host page mirrors, and snapshot-local arrays | `--prefix-cache-memory-bytes`; overlaps the MLX and host counters rather than adding another independent allocation |
| Live host heap | Portable snapshot bytes, metadata/indexes, and other CPU allocations | Darwin `malloc_zone_statistics(NULL, ...)` reports live bytes across allocator zones; not exclusively cache payloads |
| I/O staging and lookahead | Pending publications, portable prefetch, and pinned snapshots | Reservations described above; conservative ownership accounting, not measured physical bytes |
| Retained allocator space and other residency | Unused heap capacity, driver allocations, mapped/resident pages, and other process overhead | Reserved heap bytes and Darwin physical footprint; neither is interchangeable with live allocation bytes |

Do not sum these rows or independently measured peaks. Hot-tier and staging
accounting overlap physical allocations; reserved heap bytes include live heap
bytes, and not all reserved space is physically resident. MLX and heap counters
are diagnostic views, not an exhaustive additive decomposition of physical
footprint. These measurements do not isolate weights from live KV/workspaces.

### Recorded characterization — 2026-09-10

Implementation: `d4468dc9df1bf44068d85bb730c62a67df07257c`.
Integrated into `main` by `9708be1afb813b7499e142d8f4897c470e5cc5b1`.
Baseline: `b28c5db0e500be5731e52ede55366f8ae351f6f6`.
Hardware: Apple M4 Max / Metal. Target:
`Jundot/Qwen3.8-27B-oQ4e-fp16-mtp`; draft:
`incoai/Qwen3.8-27B-DFlash2`; Turbo4 KV, greedy DFlash2 generation,
up to 32 generated tokens per request, and the 8 GiB MLX free-buffer allowance.
Each soak ran 64 requests with short and 1,537-token appended turns.
All memory figures below use GiB (`2^30` bytes).

| Workload | Hot-tier budget | Maximum prompt tokens | Maximum sampled MLX active | MLX peak | Physical peak |
|---|---:|---:|---:|---:|---:|
| One retained long context | 8 | 211,896 | 24.11 | 27.91 | 41.39 |
| Alternating long contexts with cache pressure | 6 | 204,139 | 23.21 | 30.11 | 49.19 |

Settled checkpoints from the mixed-context run (request counts are one-based):

| After request | MLX active | MLX reusable buffers | Live malloc bytes | Reserved malloc bytes | Physical footprint |
|---|---:|---:|---:|---:|---:|
| 8 | 21.87 | 8.08 | 6.29 | 8.91 | 35.96 |
| 56 | 22.06 | 8.00 | 6.95 | 11.24 | 41.74 |
| 64 | 22.07 | 8.00 | 3.23 | 11.15 | 39.59 |

The final request has no next-request prefetch, so its lower live heap value is
not a like-for-like retention comparison with earlier lookahead checkpoints.
The largest sampled live heap was 6.95 GiB, only 0.66 GiB above request 8.
After cache/provider teardown and MLX cache clearing, MLX active bytes returned
to 2 bytes and live malloc usage to 53.41 MiB, while malloc still reserved
11.09 GiB and process physical footprint remained 4.19 GiB. Reserved or resident
space must not be mislabeled as live snapshot ownership.

Both soaks completed their restart/teardown checks. All 64 mixed-context output
sequences matched the baseline when replaying identical prompt hashes and cache
boundaries. Different hit boundaries can change prefill chunking and subsequent
outputs; they are not equivalent numerical comparisons. These are finite
workload measurements, not an indefinite bound or a 36 GiB acceptance pass.

### Memory reduction efforts already made

- Established the default 8 GiB free-buffer allowance and explicit
  `MLXCEL_CACHE_LIMIT` override; reducing it can trade throughput for retention.
- Tiled oversized fallback attention score matrices and materialized/detached
  DFlash prefill chunk outputs to avoid cross-chunk lazy-graph retention.
- Compacted retained snapshot page backing allocations so small page views do
  not pin historical full-context buffers; materialized continuation logits
  before retaining them.
- Shared memoized portable page mirrors and charged unique host allocations
  independently of GPU page identities, including mirrors acquired after
  initial hot-tier admission.
- Cached immutable page-accounting summaries and refreshed only dynamic host
  accounting, avoiding redundant export graphs and repeated publication work.
  Publication remains synchronous; this is optimization, not offloading.
- Bounded optional publication/lookahead staging and preserved cancellation
  ownership until I/O actually finishes, as described in the cache section.

These changes do not truncate context or make the hot-tier budget a process
cap. Further work toward 36 GiB must measure physical footprint as well as MLX
and host ownership, and preserve output correctness and prefill/decode
throughput rather than merely shifting bytes between counters.

## Generation policy

Qwen request admission uses the checkpoint's declared
`max_position_embeddings` as the context limit. The fully rendered prompt
(including retained reasoning, tool definitions/results, and expanded image
tokens) plus the effective output-token budget must fit that limit. Equality
is allowed; arithmetic overflow or an excess is rejected, not truncated.
For the current 262,144-token checkpoint and a 32,768-token output budget,
the rendered prompt must contain at most 229,376 tokens.

Over-budget Chat and Responses requests fail with HTTP 400 and
`context_length_exceeded`, before ordinary prefix-cache restoration or a
streaming response starts. Prepared lookahead must not prefetch an inadmissible
request. Durable continuations retain the original total output budget rather
than obtaining a fresh allowance, and direct runtime generation enforces the
same bound. Speculative proposals/verification must also stay within the
context, even if excess proposals would not be emitted.

This is not history compaction, an automatic RoPE extension, or enforcement of
the 36 GiB physical-memory target. Native Tau3 can record an over-budget model
request as a reward-zero execution failure; do not silently truncate or
selectively retry it to improve the score.

Qwen3.8 defaults follow the upstream model's mode-dependent policy, not the
quantized checkpoint's generation configuration. Thinking uses temperature 1.0,
top-p 0.95, top-k 20, and presence penalty 0.0; non-thinking uses temperature
0.7, top-p 0.8, top-k 20, and presence penalty 1.5. Both use min-p 0.0,
repetition penalty 1.0, and frequency penalty 0.0 unless explicitly overridden.
Stochastic speculative decoding verifies against the full-vocabulary target
distribution after the configured sampling transforms, not a draft-candidate-only
renormalization. A seed is not a promise of identical text across decoders.

Qwen's Metal runtime sets an 8 GiB allowance for reusable free MLX buffers once
at initialization, before model weights are loaded. `MLXCEL_CACHE_LIMIT`
overrides this allowance with an unsigned decimal byte count; `0` disables
free-buffer caching. Invalid explicit values fail initialization rather than
silently selecting a fallback. The selected allowance is logged at startup.
This policy does not change the wired-residency or allocation limits, prefix
snapshot budgets, KV precision, or generation algorithms.

The allocator allowance is not a hard process-memory cap. Live weights, KV,
workspaces, driver allocations, and host snapshot mirrors are separate, and MLX
may transiently exceed its free-buffer allowance until a subsequent allocation
reclaims buffers. Smaller overrides can trade throughput for lower retention;
compare both prefill and decode before changing the default for a workload.

DFlash's existing selector calibrates only verification widths 4 and 5 (three
and four proposed draft tokens). Contexts below 64,000 tokens initially prefer
width 4; longer contexts initially prefer width 5. These are adaptive defaults,
not fixed-width benchmark claims. `QW_DFLASH2_VERIFY_WIDTH=4` or `=5` can pin a
controlled experiment; keep production selection unchanged without evidence.

Server decode throughput divides committed output tokens by the runtime's
post-prefill wall time, not request-to-terminal time or pure GPU compute.
MTP starts that clock after the first-token callback and stops before final
snapshot capture; DFlash includes final snapshot capture. Preserve this timing
scope difference when comparing routes, together with actual token counts,
stop reasons, cache sources, effective sampling, and separately labeled memory.
These phase clocks are distinct from content TTFT: measure the latter through
the first nonempty generated content delta, never an HTTP header or SSE role
event. The canonical benchmark excludes the first output token from its decode
numerator. Report schema compilation and cache state when comparing constrained
and unconstrained requests.

Attention geometries without a native fused SDPA kernel use query tiling when
their score matrix exceeds `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` (existing default:
1024 MiB; 0 disables it). This includes Metal multi-query attention with head
dimension 256. Each tile retains the same visible key prefix and causal offset;
Metal evaluates and detaches its output before constructing the next tile so
lazy graphs do not retain every tile's transient score buffers together.
The budget covers raw attention scores, not total process or GPU memory.

DFlash target prefill also materializes each configured chunk before advancing:
captured hidden rows and final logits are evaluated before recurrent and attention
cache state is materialized, then the retained output roots are detached. Hidden
captures are trimmed to the draft model's required window before copying. This
bounds cross-chunk lazy-graph retention, not live KV storage or total memory.

This workspace scheduling does not truncate model context, change the
DFlash/GDN prefill chunk boundaries, alter sampling or decoder selection, or
change caller token budgets. Weights, live/restored KV state, allocator history,
and driver residency still contribute memory pressure; a fresh-process replay
does not reproduce all conditions of a long-lived server.

The Qwen XML tool-call decoder uses each declared function's parameter schema.
Strings are emitted verbatim by the model template, so JSON-looking strings,
quoted text, and meaningful whitespace remain strings; only framing newlines
are removed. Explicit nonstring types decode matching JSON values, with existing
boolean/null aliases also accepted. Missing or unresolved types, composition-only
schemas, and unions admitting strings conservatively retain text; explicit
nonstring `type` arrays decode matching JSON values before aliases. This is
wire decoding, not full JSON Schema validation or reference resolution.
Evaluation episodes produced by schema-blind parameter coercion are not
comparable baselines: a declared JSON string could have reached a tool as an
object, changing both tool execution and the subsequent conversation.

## DFlash2 constrained decoding

With a DFlash2 draft available, explicit and automatic decoder selection support
JSON Schema, images, and their combination. Constraints apply to target
verification; stochastic rejection retains the original draft probabilities
rather than renormalizing proposals over the grammar's allowed tokens.

Parser transactions commit canonical accepted output only. Backtracks and
speculative splice rebuilds restore an immutable prompt base, including prefill
and RoPE state. An append-only fast-forward at a fully committed target boundary
continues from the live state and evaluates only its new suffix. An atomic splice
that exceeds the output budget is rolled back, not partially published.
Exact prompt checkpoints and full reused prefixes can
supply that base without an extra snapshot. Otherwise, a transient checkpoint
reuses existing attention pages or retains compact immutable tensors instead of
allocating new persistent pages. It is frozen before the first target mutation,
not before first-token sampling; its cost remains inside the unchanged phase
timers. Published cache snapshots remain paged. Constrained rounds use the same
adaptive verification-width calibration and asynchronous draft launch as
unconstrained rounds. Full hidden-context windows are maintained only when
terminal snapshot capture requires them.

Eligible constrained requests populate and reuse the same exact-prompt projected
drafter K/V memo as unconstrained requests; no unconstrained warmup is required.
The key binds the reusable snapshot identity and complete prompt suffix, and
excludes segmented or new-image prefills. Cold priming projects immutable prompt
hiddens before any generated rows enter the drafter cache. Replay restores that
projected base and supplies only generated rows while the prompt boundary remains
inside the retained window. Evicted boundaries and unbounded attention use the
raw-context rebuild path. Attention masks use the full visible context and its
logical ring order, never unused backing capacity.

Greedy selection can reuse an already-allowed unmasked winner only when the
sampling transforms preserve equivalence to canonical masked selection.
Verification rows can be sampled together. History-dependent transforms use the
hypothetical accepted draft prefix for each row; candidates after the first
rejection or splice are discarded.
Guidance can validate ordinary greedy candidates directly when no forced token,
token-healing prefix, or pending stop requires canonical mask computation.
Forbidden winners, unsupported validation, and unsafe sampling transforms retain
full masking and the canonical sampler.

Streaming publishes the parser's committed, never-retractable byte prefix, not
tentative speculative tokens. The provider buffers incomplete UTF-8 boundaries
and reconciles the final canonical token sequence with previously emitted bytes.
The grammar factory retains at most one compiled schema with its initial mask
and deep-clones that zero-output parser state for each request. Schema changes
replace the template; mutable parser state is never shared between requests.
A cold schema still pays compilation and initial-mask computation costs.

On the M4 Max with the Qwen3.8-27B oQ4e target and DFlash2 draft, the verified
structured-output gate allows TTFT regression of `max(5%, 5 ms)` and at most 5%
loss in prefill or decode throughput. TTFT starts before schema compilation and
ends at the first nonempty content delta, not a role/header event. The paired
matrix uses one warmup and three timed AB/BA repetitions, 128 output tokens,
greedy sampling, and the normal presence/frequency penalties of 1.5/0.5.
This is a text-schema provider probe with terminal snapshot capture disabled;
the combined image/cache/HTTP paths are verified separately.

| Prompt case | TTFT off → on (ms) | Prefill off → on (tokens/s) | Decode off → on (tokens/s) |
| --- | ---: | ---: | ---: |
| Fully cached, 52 tokens | 6.41 → 7.34 | No new prompt tokens | 37.70 → 37.83 |
| Fresh, 4,179 tokens | 16,447.08 → 16,445.76 | 254.09 → 254.11 | 40.14 → 39.17 |
| 10,580 tokens, 10,337 cached | 1,113.53 → 1,112.35 | 218.23 → 218.46 | 38.96 → 38.34 |
| 64,540 tokens, 64,297 cached | 1,631.06 → 1,635.47 | 148.99 → 148.59 | 22.27 → 21.79 |

All four cases pass that gate. The absolute TTFT allowance is material only for
the fully cached case (+0.93 ms); the other cases also pass a relative-only 5%
TTFT limit. All measured outputs were schema-valid prefixes and streamed bytes
matched canonical decoding. Fresh and 64k runs can differ in token sequence
because packed-cache verification regrouping changes floating-point evaluation;
these figures are representative measurements, not a guarantee for every schema
or output distribution. Cold schema compilation is not amortized away from
individual request timing, but the table reports warmed repetitions.

The separate standard unconstrained throughput benchmark remains within 5% of
the preserved pre-change baseline: prefill changes range from -0.05% to +0.22%,
and decode changes from +0.22% to +1.03% across fresh, 10k, and 64k contexts.
Final Chat Completions and Responses SSE checks preserve JSON and UTF-8 output.
Repeated red images reuse all 61 prompt tokens; changed blue pixels miss;
an appended blue-image turn reuses 108 tokens. Restarting with automatic decoder
selection restores the red image's 61-token prefix from persistent storage.

## DFlash2 image verification

The DFlash2 image path uses the existing vision processor and merged embeddings,
then captures selected target-layer hidden states during chunked multimodal
prefill. Each chunk slices the same image embeddings and three-axis positions.
Decode and speculative rollback retain the image RoPE delta; a fresh text
request clears it. The draft/verify/sampling loop is shared with text generation.
Image-aware prompt keys bind each image's ordered token span to a SHA-256 digest
of decoded RGB pixels and dimensions. A lookup never ends inside an image span.
Appending another image can reuse the earlier covered images; changed pixels
invalidate reuse across that image even when text tokens and dimensions match.
Both hot and persistent DFlash2 prefix snapshots preserve multimodal state.
Persistent manifests bind the actual token/image content to the entry identity;
inconsistent bindings are rejected rather than reused as text-only prefixes.
Image and constrained completions, including cancellation, can publish ordinary
prefix snapshots. Durable response continuation remains unconstrained text only.

The deterministic OCR PNGs in `tests/fixtures/images` are committed. Their
manifest includes exact transcriptions, photo provenance, hashes, and semantic
checks. The photo files are git-ignored; after reviewing their rights, fetch
the exact originals with:

```bash
python3 tests/fixtures/images/fetch_sources.py --accept-source-rights
```

Focused regressions (also included in the ordinary test suite):

```bash
./gpu-lock -- cargo test -p qw-runtime qwen3_5::tests::dflash_multimodal_chunks_and_rollback_preserve_image_context -- --exact
./gpu-lock -- cargo test -p qw-server tests::image_requests_larger_than_two_mib_reach_both_endpoints -- --exact
```

The first checks image-conditioned hidden states and logits against an independent
target forward, chunk boundaries, partial speculative rollback, and image-to-text
state reset. The second submits a valid PNG beyond Axum's former implicit 2 MiB
body limit through both HTTP endpoints; the server now explicitly caps JSON at
128 MiB.

For real-model verification, compare baseline and DFlash2 on all four fixtures
with greedy sampling and thinking disabled. Check OCR against the manifest
(receipt column padding may be normalized), and compare complete generated text
between decoders. Also exercise Chat/Responses streaming, a multi-image prompt
longer than the prefill chunk, cancellation followed by a new request, and text
prefix reuse across automatic-decoder image requests. DFlash2 logs must show
`Dflash2Multimodal` and actual proposed/accepted draft tokens, not a fallback.
Use the paired canonical text benchmarks below to check throughput regressions.

## Upstream-native banking evaluation

See [the evaluation reproduction runbook](eval/README.md) for the pinned
environment, credentials, model-server setup, dataset reuse, fresh runs, native
resume, and verification commands. That file is the canonical operational guide;
`eval/run_tau3_banking.py` is the configuration-only launcher.

The reference workflow uses unmodified EvalScope 1.11.0 and official Tau2 v1.0.0
(`17e07b1da2bbc0cadfddeea36412686e0604127b`). Repository-owned generation,
scoring, retry, and persistence hooks are removed. Earlier custom-adapter
campaigns remain diagnostics, not reference results.

## Shared worktrees

Each worktree must link Cargo's `target/` to the primary repository target:
`target -> ../../target` for the direct `worktrees/<task>` layout. Verify that
the link resolves to the primary `target/` before running Cargo, build, or
benchmark commands. Never replace a pre-existing real directory or unrelated
symlink.

Sharing `target/` is not race-free. Serialize concurrent Cargo commands that
write shared artifacts with a separate build lock. The GPU lock below does not
provide this build serialization.

For baseline/candidate comparisons, use distinct `CARGO_TARGET_DIR` directories
under the primary `target/` and retain both build logs. A shared output directory
can reuse a different worktree's executable even when Cargo commands are
serialized. Verify source identity in the benchmark output and record source,
model, environment, and fixture hashes; a successful exit alone is insufficient.

The canonical runtime benchmark has separate fresh, 10k, and 64k prefill/decode
cases, one warmup and three timed repetitions. Preserve its timing definitions
and output assertions. The default feature uses DFlash2; `--no-default-features`
uses bundled MTP. Both variants use the same fixture filenames with different
identities, so pair baseline/candidate runs within one variant before switching,
and preserve the fixture inputs and `BENCHMARK_RESULT` rows.

## DFlash2 optimization trials — 2026-09-12

Baseline runtime: `aad83fe7928e5f9eee065d8c71538f06f832ae54` (documentation
base `980fe9561b4176f136e800e88aac8378a1e9c253`). Accepted implementation:
`db04cb2cf094fe0be18600a21c724f6fbfc13033`. Hardware and checkpoint are the
README benchmark configuration. Each run used one warmup, three timed
repetitions, and the existing deterministic-output assertions.

Only the five-row, head-dimension-256 Turbo4 attention path now stages 32
tokens instead of 16. Four-row verification retains 16-token stages; per-head
arithmetic, block decomposition, quantization, and decoder selection are
unchanged. The regression checks bitwise grouped-versus-rowwise attention
parity for Qwen's 24-query/4-KV-head geometry at ragged long prefixes and a
65,536-token reduction-tier crossing.

| 64k trial | Decode tokens/s | Decision |
|---|---:|---|
| Baseline, adaptive ABBA endpoints | 36.446 / 36.271 | Reference |
| 32-token stages only at width 5, adaptive ABBA middle runs | 36.782 / 36.915 | Accepted; pooled improvement about 1.35% |
| Two heads per SIMDgroup instead of three | 33.694 | Rejected |
| Six heads per SIMDgroup instead of three | 28.268 | Rejected |
| Eight-token stages | 35.921 | Rejected |
| Partial fusion of compatible GDN auxiliary projections | 36.561 / 36.635 | No repeatable 64k gain; not integrated |

Using 32-token stages at both widths initially improved fixed-width-5 ABBA
throughput from 36.603/36.682 to 37.086/37.012 tokens/s, but regressed
fixed-width-4 from 36.552 to 36.190. This motivated the width-specific
specialization, not a change to the adaptive width policy. Disabling grouped
attention measured 24.867 tokens/s and is not an optimization.

The accepted full suite measured prefill 253.941/232.541/154.739 and decode
56.798/54.910/36.948 tokens/s for fresh/10k/64k. These modest gains do not
establish a practical route to the idealized bandwidth ceiling. Draft/verify
profile intervals are asynchronous wall-time accounting: verification drains
pending draft GPU work, so they are not isolated GPU phase measurements.

Local build logs, benchmark rows, fixture/model hashes, and source provenance
are retained under `target/dflash2-opt-20260912-151310/`. Rejected candidates
were not integrated; their trial branches and worktrees were removed.

## DFlash2 deep bottleneck investigation

The follow-up investigation did **not** reach the 60-token/s target. No runtime
candidate from this investigation was integrated. The earlier bandwidth-only
roofline must not be read as a practical throughput prediction.

All candidates forked documentation base `5a4cd8a`, retaining the accepted
`db04cb2` runtime. The same cached 64k fixture, default adaptive width policy,
one warmup, three timed repetitions, and exact deterministic-output assertions
were used. GPU commands were serialized through `gpu-lock`; Cargo builds used
isolated artifact directories and the shared build lock.

| 64k experiment | Decode tokens/s | Decision |
|---|---:|---|
| Accepted-runtime contemporaneous control | 36.698 | Reference |
| 512 attention partitions above 32k (`35708f3`) | 36.619 | No improvement |
| Blocking draft evaluation (`c0bcdfd`) | 35.687 | Rejected |
| Detached committed hidden buffer (`2f1756f`) | 36.519 | No improvement |
| Existing dequantized native SDPA control | 28.192 | Rejected |
| Packed SIMDgroup-matrix QK/PV (`c3eda55`) | 24.097 | Rejected |
| Dequantized QK/PV with GQA folded into 24/30 query rows (`b385df0`) | 21.575 | Rejected |

The two dequantized controls set `MLXCEL_TURBO4_FUSED_ATTENTION=0`; other rows
used default dispatch. All listed runs passed the existing output assertions.
This is fixture-level correctness evidence, not proof of arbitrary-input
numerical equivalence.

### Attribution and the required latency reduction

The accepted-runtime control generated 381 timed tokens in 10.382 seconds,
over 88 verification rounds. Its asynchronous verification interval accounted
for 9.351 seconds (90.1% of decode). Reaching 60 tokens/s requires completing
the same work in 6.350 seconds: a **38.8% total latency reduction**. Holding
other intervals and the round schedule fixed would require reducing the
verification interval from 106.27 to 60.45 ms/round, or **43.1%**. This interval
also drains pending draft GPU work, so it is not a pure target-GPU measurement.

A separate diagnostic drained inputs and evaluated outputs at every attention
and MLP sublayer for one five-row verification round. Summed wall times were:

| Sublayers | Synchronized time |
|---|---:|
| 48 GDN sublayers, including their projections | 31.410 ms |
| 16 full-attention sublayers, including their projections | 44.416 ms |
| 64 MLP sublayers | 48.869 ms |

These measurements include added synchronization overhead and are not GPU
counters or an additive decomposition of normal asynchronous decode latency.
Two Metal System Trace attempts failed to finalize after the target exited;
neither supplied usable hardware attribution.

### Quantized projection experiments

The target's actual affine sidecars and runtime activations are FP16, despite
its config advertising BF16. Real-weight probes therefore used FP16 inputs,
affine group size 64, and the checkpoint's 4/5-bit packed weights. The initial
BF16-input probe promoted outputs to FP32 and was discarded as unrepresentative.

MLX already shares weight reads across four/five verification rows through
its wide affine QMV. Padding to 16/32 rows to force matrix dispatch was slower.
Three additional kernels were implemented and exercised on the actual fused
MLP gate/up, MLP down, GDN QKV/output, and vocabulary projection weights:

- An 8-by-32 SIMDgroup matrix tile with in-kernel dequantization.
- Wide vector kernels using 16 and 32 K lanes rather than the native mapping.

None beat stock MLX across the tested four/five-row cases. For five-row MLP
down projection, stock measured 343/342 microseconds before/after the trials,
versus 1,232/478/655 microseconds respectively. For the vocabulary projection,
stock measured 2,663/2,701 microseconds versus 5,643/3,145/4,861. Probe argmaxes
matched, but changed accumulation orders were not bitwise equivalent. Slower
projection candidates were not routed into the full decoder.

The measurements establish substantial projection cost alongside attention;
they do not establish that 60 tokens/s is impossible, or that it is achievable
with another kernel. No model precision reduction, context truncation, or
timing-boundary change was used to manufacture a gain.

Logs, source/binary hashes, numerical probe rows, and the latency calculation
are retained in `target/dflash2-deep-20260912/results-and-provenance.json` and
its neighboring logs. Experimental branches and worktrees were removed without
integrating their sources. The later physical-counter measurement artifacts
remain under `target/dflash2-physical-20260912/`; diagnostic branches and
worktrees were also removed.

## GPU serialization

GPU tests, benchmarks, and smoke commands from concurrent worktrees must use
the repository wrapper:

```bash
./gpu-lock -- cargo test -p mlxcel-core
./gpu-lock -- cargo bench -p qw-runtime --bench single_user_throughput
```

`gpu-lock` takes a blocking exclusive advisory lock at
`~/.cache/qw/gpu_lock`, reports the current holder while waiting, then replaces
itself with the wrapped command so exit statuses and signals propagate
unchanged. Do not delete the lock file. This serializes Metal work; it does
not replace the separate build lock required for Cargo writes to shared
artifacts.

Keep benchmark output and result files uniquely named when worktrees share the
repository. Do not treat historical measurements as current quality or
throughput baselines; publish a result only after it has been reproduced
against the current implementation.
