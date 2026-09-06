# QW (QuasarWave)

QW (QuasarWave) is a highly opinionated LLM inference runtime targeting MLX-only,
single-user deployments.

## Showcase

[![asciicast demo implementing a QR Code generator webapp](https://asciinema.org/a/x86fzcfENyZzge26.svg)](https://asciinema.org/a/x86fzcfENyZzge26)

And the Pi agent with QW runtime built this qr-code app in 2m 40s:

![QR App screenshot](./static/qr-app.png)

## Project Goals

QW aims to be explicitly "focused", to achieve these goals below:

- Smooth user experience: It should "just work" without complex first-time
  configuration. Decode TPS and TTFT (time to first token) are the top priority
  metrics to optimize.
- Minimal: No fancy features, fixed model, fixed environment.
  - No fancy features: GUI, MCP integration, etc. are outside this project's
    scope. 4-bit TurboQuant cache quantization and request-level MTP/DFlash2
    decoder routing are enabled by default.
  - Fixed model: Qwen3.8 27B (dense model) only. No generalization across
    different model structures.
    - QW serves [Jundot/Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp) as the default model.
    - If a better model with similar requirements is released, this project
      may migrate to the new model, but it will never support more than one
      model at a time.
  - Fixed environment: MLX only. No generalization across CUDA, ROCm, ...

## Quickstart

Install prebuilt binary via shell script:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cr0sh/qw/releases/download/v0.1.0/qw-cli-installer.sh | sh
 ``````

Or, compile from the source. Prerequisites:

 1. macOS on Apple Silicon
 2. Rust 1.85 or newer
 3. CMake 3.16 or newer
 4. Xcode Command Line Tools (`xcode-select --install`)

```bash
cargo install --locked --git https://github.com/cr0sh/qw [--tag TAG] qw-cli
```

Download the target model and the DFlash2 draft checkpoint from Hugging Face
(the first model ID defaults to `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp`):

```bash
qw download
qw download incoai/Qwen3.8-27B-DFlash2
```

If the draft checkpoint is absent, automatic routing falls back to bundled
MTP, or baseline decoding when bundled MTP is unavailable.
Run the server:

```bash
qw serve
```

`--decoder auto` is the default for both `qw generate` and `qw serve`. It
prefers DFlash2 whenever the draft is available and the request is compatible,
regardless of prompt length. DFlash2 supports greedy, unconstrained text
generation; sampling, structured output, and image requests retain MTP or
baseline routing. `--decoder baseline|mtp|dflash` makes an explicit selection.
Override the draft checkpoint with `--dflash-draft-model` or
`QW_DFLASH_DRAFT_MODEL_PATH`.

Both speculative decoders verify proposals against the full target vocabulary.
Compact vocabulary heads are used only to propose draft tokens; they must not
exclude a target winner during acceptance, correction, or bonus-token selection.

DFlash2 participates in the server's memory and filesystem prefix cache, in a
separate decoder namespace. Snapshots retain target state at its actual resident
precision, the bounded drafter hidden window, and continuation logits. Structural
checkpoints do not prematurely quantize the live prefill cache. Interrupted
response snapshots end at the emitted-token boundary without replaying the prompt.
Cache maintenance publishes snapshots while the generation queue is idle, so
immediately queued requests can arrive before a preceding snapshot is available.
Turbo4 speculative regrouping can change floating-point results and greedy token
choices; byte-preserving snapshots do not promise token-identical continuations
across different execution groupings.

The server rejects malformed or undeclared generated tool calls and violations
of `parallel_tool_calls=false`. These output-contract failures return HTTP 422
with `error.type=model_output_error` and `error.code=invalid_model_output`, rather
than a retryable server error. Streaming responses use the same error type/code
in the endpoint's terminal error event after HTTP headers have been sent.
The error does not include raw generated argument bodies.

For τ³ evaluation, this strict rejection differs from Tau2's live environment,
which can return unknown-tool feedback to the agent. The banking runner aborts
on such protocol failures rather than skipping tasks or assigning reward zero;
an aborted run is not a complete benchmark score.
The full Tau2 result is the authoritative trajectory. Report-only user tool
actions retain the user role and exact calls in `tau2_user_tool_calls` metadata;
tool-only report entries use an empty content list, never invented user text.
This report representation is not sent to models. Opt-in capture writes the
completed task result before report conversion.

The runner persists `simulator_empty_response_retries=3`: only a
`user_simulator_response` from the user model that ends with a clean stop and
no visible text or tool calls can repeat the identical request, at most three
extra times. This is a deliberate deviation from stock Tau2's handling of
empty simulator responses. Each attempt and protocol-retry reason is captured;
no synthetic STOP, extra simulator turn, or repeated tool execution is added.
Visible answers, refusals, and tool actions are never resampled. Agent/judge
calls are not eligible. Empty truncated responses, completion errors, and retry
exhaustion abort rather than becoming task reward zero; nonempty truncated
responses retain stock Tau2 handling. Capture separates original
`tau_tools` from actual EvalScope-serialized `wire_tools` and `wire_messages`.

The simulator also receives a task-independent clarification that every
invocation, including after tool results, must produce visible text or a tool
call; a finished interaction uses the already-specified termination marker.
This clarifies Tau2's nonempty-step contract but is an explicit simulator-prompt
deviation, not an unchanged stock-prompt score. The exact text is persisted as
`simulator_step_protocol` in EvalScope configuration and appears in captured
wire input. Original Tau2 goals/history remain untouched; agent and judge
prompts do not receive it. Start a fresh campaign when adopting this policy;
do not mix records from the earlier prompt configuration.

Run the durable banking launcher from a dedicated worktree using a Python
environment with EvalScope 1.11.0 and `tau2[knowledge]` v1.0.0 installed:

```bash
python eval/run_tau3_banking.py                 # validate configuration only
python eval/run_tau3_banking.py --limit 1 --run # one-task integration diagnostic
python eval/run_tau3_banking.py --run           # all 97 tasks, one repeat
```

The agent is `qwen3.8-27b` at `http://127.0.0.1:8883/v1`; the simulator is
`openai-codex/gpt-5.6-luna`, defaulting to the authenticated gateway at
`http://127.0.0.1:18766/v1`. Keep QW and the auth services available; configure
`TAU_SIMULATOR_API_URL` and `TAU_SIMULATOR_TOKEN_FILE` if their defaults differ
(the token file defaults to `~/.omp/auth-gateway.token`). No server is started
by the launcher. Optional `TAU3_DATASET_ID=/path/to/dataset` reuses an existing
local dataset read-only; otherwise the configured dataset is downloaded into
the new run's cache. Each fresh invocation chooses a UUID result directory
under `eval/outputs/`. Opt in to request/response/result capture with
`--capture /unique/path.jsonl` (or `TAU3_CAPTURE_PATH`); existing capture paths
are rejected. Use a fresh QW process and prefix-cache directory for a fresh
scored campaign; do not resume an invalid run as a scored baseline.

To continue verified completed native records after a terminal interruption:

```bash
python eval/run_tau3_banking.py --run --resume /path/to/outer-UUID/YYYYMMDD_HHMMSS
```

Pass the exact **inner timestamp directory** containing `configs/`,
`predictions/`, and `reviews/`, not the outer UUID directory or a JSONL file.
Its existing `progress.json` must say `error` or `completed` with total count
97; missing, running (including stale-running), and diagnostic runs are
rejected. First observe the previous evaluator's exit. Never edit progress to
make an active or unverifiable run resumable. `--limit` cannot accompany resume.

Before native configuration is overwritten, the launcher checks EvalScope's
evaluation identity and every cached prediction/review against the current
97-task banking dataset and canonical Tau2 result. This includes task order,
model, completed reward metadata, finite rewards (including legitimate zero
and `max_steps` results), lossless reasoning/user-tool trajectories, and
matching reviews. Duplicate, orphan, corrupt, and infrastructure-error rows
fail closed; there is no forced reuse or `rerun_review` bypass.

Completed predictions are reused, missing reviews are computed, and only
uncached tasks rerun. **Partial trajectories are not restored**: a failed task
with no completed prediction starts again, so this is not a way to preserve
its agent prefix or justify blind stochastic retries. A zero-valid-row failed
campaign has no completed work to preserve. No error is converted to reward
zero or silently skipped.

Fresh and resumed writers hold the same nonblocking, persistent outer
`.writer.lock` throughout validation/evaluation; leave that file in place.
Competing updated launchers fail before writing native run artifacts. Legacy
launchers do not take this lock: it cannot protect against them, so observed
legacy evaluator exit remains a procedural prerequisite. Resume uses the
outer directory's `data-cache/`, just like the original fresh run.

Keep the original local dataset snapshot read-only and unchanged. Native
identity does not checksum local dataset contents: 97 rows plus matching
cached metadata cannot establish that uncached task contents are unchanged.
This resume contract assumes the known read-only snapshot, rather than
inventing a retroactive checksum. Offline native CLI verification used
explicitly nonscored fixtures: one complete cache hit, one pending review
completed, and the next uncached task aborted before inference; a competing
resume CLI also failed against a fresh writer's lock. This proves resume
mechanics, not completion or a score for the 97-task benchmark.

Obtain statistics about storage usage and status:

```bash
qw stats
```

For a more detailed manual, use `--help` — or just ask your LLM.

## GPU command serialization

GPU tests, benchmarks, and smoke commands can be serialized across worktrees and
agent sessions with the repository wrapper:

```bash
./gpu-lock -- cargo test -p mlxcel-core
./gpu-lock -- target/release/deps/single_user_throughput-HASH single_user_decode/long_10k_mtp_k3
```

The wrapper takes a blocking exclusive advisory lock on the persistent
`~/.cache/qw/gpu_lock` file, reports the current holder while waiting, and
replaces itself with the command so exit statuses and signals propagate
unchanged. Do not delete the lock file.

## Performance

Latest recorded local measurements for the default oQ4e checkpoint and Turbo4
cache, collected on September 2–4, 2026, on a Mac Studio with an Apple M4 Max
chip, 64 GB of memory, and a 40-core GPU. These are separate runs, not a single
benchmark of the current revision or the v0.1.0 release.

   |  | fresh | 10k | 64k |
   |---|------:|-----:|-----:|
   | **prefill** | 253.737 | 231.129 | 153.974 |
   | **decode (baseline, historical)** | 27.595 | 21.914 | 12.856 |
   | **decode (MTP)** | 57.710 | 53.782 | 36.903 |

All values are tokens/s. Fresh prefill processes 4,341 tokens; `10k`/`64k`
prefill processes a 288-token suffix after cached prefixes of 10,337/64,297
tokens, not the entire history. Decode measures 127 tokens after the first token.

Prefill comes from the September 4 six-case run at `1a36b3b`; MTP comes from
the grouped-attention runs merged as `f4887cb`, using the latest 10k repeat.
Baseline decode retains the September 2 Criterion point estimates: the newer
six-case harness no longer measures ordinary baseline decode. The newer rows
report total tokens divided by total phase time over three timed repetitions,
after one untimed warmup.

Latest greedy decoder measurements:

   | decoder | fresh | 10k | 64k |
   |---|------:|-----:|-----:|
   | **MTP** | 57.710 | 53.782 | 36.903 |
   | **DFlash2** | 57.510 | 56.477 | 40.050 |

All decoder values are tokens/s. DFlash2 comes from the September 4 combined
optimization runs merged through `7ada6d4`; 64k uses the latest confirmation
(40.050), not the earlier 40.268 result. Long-context timed DFlash2 repetitions
reuse the same live prompt snapshot and exact suffix, including projected-context
cache hits; these are warm-reuse measurements, not cold-request throughput.

These historical speculative-decoder timings predate the full-vocabulary target
verification correction. They used a restricted target head that could exclude
ordinary tokens, including parts of function names; do not treat them as current
correct-decoding throughput or quality baselines.

Automatic routing is capability-based, not calibrated to a prompt-length
crossover. These historical throughput measurements do not establish an
end-to-end latency win for every short request or cache state.

Reproduce the current prefill and DFlash2 cases with:

```bash
./gpu-lock -- cargo bench -p qw-runtime --bench single_user_throughput
```

For the prefill and bundled-MTP cases, add `--no-default-features`.

QW stores prefix caches under `~/.cache/qw/checkpoint`. Disk usage is capped at
16 GB by default; the hard ceiling is twice the configured limit.

The default build includes the DFlash2 router. Other experimental inference
features remain feature-gated; see `crates/runtime/Cargo.toml`.

## Project Policy

Any configuration other than the default is considered experimental and out
of scope for testing by the maintainer. The default is:

- Target checkpoint (`Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` on Hugging Face), its
  quantization method, and the DFlash2 draft checkpoint
  (`incoai/Qwen3.8-27B-DFlash2`)
- Automatic DFlash2 preference, with capability-based MTP/baseline fallback
- TurboQuant 4-bit KV cache with an FP16 resident fast path; snapshots preserve
  the resident representation rather than changing precision during capture
- The above configuration is tested on an M4 Max 40-core GPU with 64 GB of
  unified memory

## Disclosures & Acknowledgements

This repository is heavily AI-assisted, aka "vibe coding". Commit messages
include an `Assisted-by` footer indicating which coding agent/model was used.

This repository is inspired by antirez's [ds4](https://github.com/antirez/ds4)
inference engine, for its minimalism and simplicity.

This repository started from a stripped version of the core component
(`mlxcel-core`) of [mlxcel](https://github.com/lablup/mlxcel). I highly
appreciate [Lablup](https://lablup.com/)'s open-source contributions.

This repository adopts several optimization approaches from
[oMLX](https://github.com/Jundot/oMLX), [ds4](https://github.com/antirez/ds4),
and [qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge). Each derivative work is attributed in the relevant commit messages.

