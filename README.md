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

