# Development notes

These notes cover shared local resources used while developing and measuring
QW. They do not publish performance results; measurements belong in a record
that identifies the exact source revision and decoding configuration.

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

## Generation policy

Qwen3.8 defaults follow the upstream model's mode-dependent policy, not the
quantized checkpoint's generation configuration. Thinking uses temperature 1.0,
top-p 0.95, top-k 20, and presence penalty 0.0; non-thinking uses temperature
0.7, top-p 0.8, top-k 20, and presence penalty 1.5. Both use min-p 0.0,
repetition penalty 1.0, and frequency penalty 0.0 unless explicitly overridden.
Stochastic speculative decoding verifies against the full-vocabulary target
distribution after the configured sampling transforms, not a draft-candidate-only
renormalization. A seed is not a promise of identical text across decoders.

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
