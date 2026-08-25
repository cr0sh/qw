# QW (QuasarWave)

QW is a highly opinionated LLM inference runtime targeting MLX-only,
single-user deployments.

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
    - QW serves [Jundot/Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp) as the default model.
  - Fixed environment: MLX only. No generalization across CUDA, ROCm, ...

## Quickstart

Prerequisites:

 1. macOS on Apple Silicon
 2. Rust 1.85 or newer
 3. CMake 3.16 or newer
 4. Xcode Command Line Tools (`xcode-select --install`)

Installation:

```bash
cargo install --locked --git https://github.com/cr0sh/qw [--tag TAG] qw-cli
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

QW aims to be fast enough for daily use. Below is the benchmark result from
tag v0.1.0. You can reproduce it with `cargo bench`. The benchmark was run on
a Mac Studio with an Apple M4 Max chip, 64 GB of memory, and a 40-core GPU.

QW stores prefix caches under `~/.cache/qw/checkpoint`. Disk usage is capped at
16 GB by default; the hard ceiling is twice the configured limit.

An attempt to achieve better TPS/TTFT numbers for hardcore use produced
unsatisfying results, so these options are feature-gated and disabled by
default. See `crates/runtime/Cargo.toml`.

## Project Policy

Any configuration other than the default is considered experimental and out
of scope for testing by the maintainer. The default is:

- Model checkpoint (`Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` on HuggingFace) and its
  quantization method
- MTP with depth $k=3$
- KV cache is 4-bit quantized with TurboQuant
- The above configuration is tested on an M4 Max 40-core GPU with 64 GB of
  unified memory

This project does not accept issues or pull requests. This is a precautionary measure
against possible occurrence of accounts generating PRs for AI-related repositories.

If you find a bug or need a feature, use your favorite LLM to fix/implement. This
project is heavily assisted by AI, so even if an issue had been opened, I would
have done the same.

If you want to contribute, fork this repository on your GitHub account and push
the changes, then the maintainer will merge the commits if they're considered legit.
If you want to opt out of this behavior, duplicate the repository without forking
or explicitly consent in the README.

## Disclosures & Acknowledgements

This repository is heavily AI-assisted, aka "vibe coding". Commit messages
include an `Assisted-by` footer indicating which coding agent/model was used.

This repository started from a stripped version of the core component
(`mlxcel-core`) of [mlxcel](https://github.com/lablup/mlxcel). I highly
appreciate Lablup's open-source contributions.

This repository adopts several optimization approaches from
[oMLX](https://github.com/Jundot/oMLX), [ds4](https://github.com/antirez/ds4),
and [qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge). Each derivative work is attributed in the relevant commit messages.

