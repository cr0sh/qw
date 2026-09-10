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
A durable continuation preserves the response identity, delivered prefix,
original penalty boundary, remaining token budget, and independently saved RNG
state; an unrelated request must not change its continuation stream.

Ordinary memory and filesystem eviction prefer the least recently used or
materialized entry; reuse frequency breaks recency ties and still determines
TTL growth. Merely matching a shorter ancestor does not refresh its materialized
recency. A successful disk hit becomes recent before hot-tier promotion pressure.
Oversized ordinary snapshots still cannot displace viable hot prefixes, and
active continuation protections and tier budgets are unchanged.

Per-entry cache lifecycle traces carry the existing deterministic `entry_id`
derived from namespace, route, and token prefix. The identity survives hot or
filesystem removal and persistence failure, including memory-only entries.
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
