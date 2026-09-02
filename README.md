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
    scope. 4-bit TurboQuant cache quantization and MTP with depth `$k=3$` are
    enabled by default.
  - Fixed model: Qwen3.8 27B (dense model) only. No generalization across
    different model structures.
    - QW serves [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`, using `Qwen3.8-27B-UD-Q4_K_XL.gguf` with the matching `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` head.
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

Download and verify the fixed target and MTP GGUF artifacts:

```bash
qw download
```

The files are stored under
`~/.cache/qw/models/unsloth/Qwen3.8-27B-GGUF/`, preserving the `MTP/`
subdirectory. Downloads resume into `.qw-part` files and are accepted only
after the pinned byte length and SHA-256 digest match.


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

QW aims to be fast enough for daily use. Below is the benchmark table from
tag v0.1.0. You can reproduce it with `cargo bench`. The benchmark was run on
a Mac Studio with an Apple M4 Max chip, 64 GB of memory, and a 40-core GPU.

   |  | fresh | 10k | 64k |
   |---|------:|-----:|-----:|
   | **prefill** | 254.27 | 221.07 | 97.08 |
   | **decode (baseline)** | 27.20 | 21.65 | 12.80 |
   | **decode (MTP)** | 58.20 | 53.38 | 35.66 |

All values are tokens/s; `10k`/`64k` are prefilled prompt lengths in tokens.

QW stores prefix caches under `~/.cache/qw/checkpoint`. Disk usage is capped at
16 GB by default; the hard ceiling is twice the configured limit.

An attempt to achieve better TPS/TTFT numbers for hardcore use produced
unsatisfying results, so these options are feature-gated and disabled by
default. See `crates/runtime/Cargo.toml`.

## Project Policy

Any configuration other than the default is considered experimental and out
of scope for testing by the maintainer. The default is:

- Model checkpoint (`unsloth/Qwen3.8-27B-GGUF` at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`) with the fixed UD-Q4_K_XL target and matching MTP head
- MTP with depth $k=3$
- KV cache is 4-bit quantized with TurboQuant
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

