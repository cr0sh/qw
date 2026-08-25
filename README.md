# QW (QuasarWave)

QW is a highly opinionated LLM inference runtime targeting MLX-only,
single-user deployments.

QW aims to be explicitly "focused", to achieve these goals below:

- Smooth user experience: It should "just work" without complex first time
  configuration. Decode TPS and TTFT is the first priority metric to
  optimize.
- Minimal: No fancy features, fixed model, fixed environment.
  - No fancy features: GUI, integrated MCP, etc. are not in scope of this
    project. 4-bit TurboQuant for caches and MTP with depth `$k=3$` is enabled
    by default.
  - Fixed model: Qwen3.8 27B (dense model) only. No generalization over
    different model structures.
    - QW serves [Jundot/Qwen3.8-27B-oQ4e-fp16-mtp] on HuggingFace as the default model.
  - Fixed environment: MLX only. No generalization accross CUDA, ROCm, ...

## Quickstart


Prerequisites:

 1. Rust
 2. CMake 3.16 or newer
 3. Xcode Command Line Tools (`xcode-select --install`)
 
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

For more detailed manual, use `--help` or just ask LLM.

## Performance

QW is aimed to be performant enough for daily use. Below is a benchmark result
of tag v0.1.0. You can reproduce with `cargo bench`. The benchmark had been run
on a Mac Studio, with Apple M4 Max chip with 64GB memory, 40-cores GPU option.

```

```

QW stores prefix caches under `~/.cache/qw/checkpoint`. Disk usage is capped at
16GB by default, but the hard ceiling at the twice of the limit set.

There was an attempt of bring more better TPS/TTFT numbers for hardcore use, but
both had unsatisfactory results so were feature gated and disabled by
default. See `crates/runtime/Cargo.toml`.

## Project Policy

Any configuration other than default is considered experimental and out
of scope for testing by the maintainer. The default is:

- Model checkpoint (`Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` on HuggingFace) and its
  quantization method
- MTP with depth $k=3$
- KV cache 4-bit quantized with TurboQuant
- Above configuration tested on M4 Max, 40-cores GPU, 64GB unified memory model

This project does not accept issues or pull requests. This is a precautionary measure
against possible occurrence of accounts generating PRs for AI-related repositories.

If you found a bug/need a feature, use your favorite LLM to fix/implement. This
project is heavily assisted by AI, so even if an issue had been opened, I would
have done the same.

If you want to contribute, fork this repository on your GitHub account and push
the changes, then the maintainer will merge the commit if it's considered legit.
If you want to opt out of this behavior, duplicate the repository without forking
or explicitly consent in README.

## Disclosures & Acknowledgements

This repository is heavily AI-assisted, aka "vibe coding". On commit
descriptions there should be an `Assisted-by` footer of which coding agent/model is used.

This repository is started from a stripped version of [mlxcel](https://github.com/lablup/mlxcel)'s
core component(`mlxcel-core`). I highly appreciate Lablup's open source software
work.

This repository adopts several optimization strategies from
[oMLX](https://github.com/Jundot/oMLX), [ds4](https://github.com/antirez/ds4),
and [qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge). These derivative works are attributed on commit descriptions on each.

