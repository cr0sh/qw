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
  - No fancy features: GUI, MCP integration, and vision inputs are outside this
    project's current scope. E4M3 FP8 KV-cache storage and the checkpoint's
    native MTP head are enabled by default.
  - Fixed model: Qwen3.8 Flash Next REAP-288 only. No generalization across
    different model structures.
    - QW serves [sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) as the default model.
    - The 51B-parameter n-gram embedding table remains in a separate pageable
      SSD-backed mapping; it is never materialized as an MLX/Metal tensor.
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

Download the model from Hugging Face (an unset model ID selects `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`):

```bash
qw download
```


Run the server:

```bash
qw serve
```

Obtain statistics about storage usage and status:

```bash
qw stats
```

For a more detailed manual, use `--help` — or just ask your LLM.

## Performance

QW targets the following minimum throughput on an M4 Max with 64 GB of unified
memory. Benchmarks must stay below 50 GB total resident memory.

   |  | fresh | 64k |
   |---|------:|-----:|
   | **prefill** | 400 | 300 |
   | **decode (baseline)** | 20 | 15 |
   | **decode (MTP)** | 34 | 25.5 |

All values are tokens/s. The MTP rows encode the required minimum 1.7x decode
speedup over the corresponding baseline.

QW stores prefix caches under `~/.cache/qw/checkpoint`. Disk usage is capped at
16 GB by default; the hard ceiling is twice the configured limit.

The n-gram embedding table is served row-by-row from its pageable SSD mapping.
Only selected rows enter a small staging buffer; the mapping must remain
outside pinned Metal memory.

## Project Policy

Any configuration other than the default is considered experimental and out
of scope for testing by the maintainer. The default is:

- Model checkpoint (`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` on
  Hugging Face) and its 4-bit affine weight quantization
- The checkpoint's native MTP head
- KV cache is 8-bit quantized with TurboQuant
- N-gram embeddings remain SSD-backed and total resident memory stays below
  50 GB
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

