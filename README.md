# QW (QuasarWave)

QW is a focused, local LLM inference runtime for single-user Apple Silicon
systems. It serves the Qwen3.8 27B dense checkpoint through an
OpenAI-compatible HTTP API and includes a command-line client for downloading
models, generating responses, and inspecting the local cache.

QW is intentionally narrow: MLX is the supported backend, and the default
model is
[Jundot/Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp).
It is not a general multi-model or CUDA/ROCm runtime.

## Showcase

Asciicast demo implementing a QR Code generator webapp(no speedup):
![asciicast demo implementing a QR Code generator webapp](./static/qrcode-demo.gif)

The Pi agent with QW built this QR-code app in 2m 40s:

![QR App screenshot](./static/qr-app.png)

## Install

Prebuilt binaries can be installed with the release installer:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cr0sh/qw/releases/download/v0.1.0/qw-cli-installer.sh | sh
```

To compile from source, use macOS on Apple Silicon with the Rust toolchain,
CMake 3.16 or newer, and the Xcode Command Line Tools:

```bash
xcode-select --install
cargo install --locked --git https://github.com/cr0sh/qw qw-cli
```

To install a tagged release, add `--tag <tag>` to the `cargo install`
command.

## Quickstart

Download the default target checkpoint and the optional DFlash2 draft
checkpoint:

```bash
qw download
qw download incoai/Qwen3.8-27B-DFlash2
```

Start the local server:

```bash
qw serve
```

The server listens at `http://127.0.0.1:8000` by default. Its OpenAI-compatible
base URL is `http://127.0.0.1:8000/v1`; supported POST endpoints include
`/chat/completions` and `/responses`.

For example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

Generate one response without starting the HTTP server:

```bash
qw generate --prompt 'Explain prefix caching in one sentence.'
```

Inspect local model and cache state, or view all command-line options:

```bash
qw stats
qw --help
qw serve --help
qw generate --help
```

## Models and decoder options

The target model can be selected by checkpoint path with `--model` or
`QW_MODEL_PATH`; when neither is set, QW resolves the default model from its
cache. The optional DFlash2 draft checkpoint can be selected with
`--dflash-draft-model` or `QW_DFLASH_DRAFT_MODEL_PATH`.

Both `qw generate` and `qw serve` expose `--decoder auto|baseline|mtp|dflash`.
The `dflash` option requires a build with the `dflash2` feature and an
available draft checkpoint. Use `--help` for the options enabled by the
current build.

## Historical performance

Latest complete `cargo bench` decode suites after the Metal allocator-memory
update (tokens/s):

| Context | Bundled MTP (k=3) | DFlash2 |
|---|---:|---:|
| Fresh | 57.354 | 56.116 |
| 10,337-token cached prefix | 52.639 | 54.372 |
| 64,297-token cached prefix | 34.853 | 36.283 |

Measured at [commit `117fecb`](https://github.com/cr0sh/qw/commit/117fecbc3c0f1be69a44c7bfe274f5f81d7ab24c). Both used the default
[Jundot/Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp)
checkpoint with Turbo4 KV cache on a Mac Studio with an Apple M4 Max
40-core GPU and 64 GB of unified memory; decoding was greedy
(`temperature=0`, `top_p=1`, seed `0`) for 127 output tokens after the first
token, with one warmup and three timed repetitions.

## Development

Cache layout, GPU serialization, and benchmark commands are collected in
[`DEVELOPMENT.md`](DEVELOPMENT.md). When running GPU work from more than one
worktree, always invoke it through `./gpu-lock -- ...`; do not delete the
shared lock file.


## Evaluation

The τ³ banking evaluation launch and capture rules are documented in
[`eval/README.md`](eval/README.md).

## Acknowledgements

This repository is heavily AI-assisted. Commit messages include an
`Assisted-by` footer identifying the coding agent/model used.

QW is inspired by antirez's [ds4](https://github.com/antirez/ds4) inference
engine for its minimalism and simplicity.

The repository started from a stripped version of `mlxcel-core` from
[mlxcel](https://github.com/lablup/mlxcel). We appreciate
[Lablup](https://lablup.com/)'s open-source contributions.

QW adopts optimization approaches from [oMLX](https://github.com/Jundot/oMLX),
[ds4](https://github.com/antirez/ds4), and
[qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge).
Derivative work is attributed in the relevant commit messages.
