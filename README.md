# QW (QuasarWave)

QW is a focused, local LLM inference runtime for single-user Apple Silicon
systems. It serves the Qwen3.8 27B dense checkpoint through an
OpenAI-compatible HTTP API and includes a command-line client for downloading
models, generating responses, and inspecting the local cache.

QW is intentionally narrow: MLX is the supported backend, and the default
model is
[Jundot/Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp).
It is not a general multi-model or CUDA/ROCm runtime.

QW also supports:

- MTP with included heads in the default model checkpoint
- DFlash2 speculative decoding (the `dflash2` cargo feature, on by default;
  needs the separate draft checkpoint)
- Image vision
- Grammar constrained decoding
- D-LSA sparse attention for long prompts (experimental, opt-in: the `dlsa`
  cargo feature, off by default; see [D-LSA](#d-lsa-experimental))

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

The only release so far, `v0.1.0` (August 2026), predates most of this README:
it is built without DFlash2 (no `--decoder` option) and without D-LSA. Build
from source for the current version.

To compile from source, use macOS 14 or newer on Apple Silicon with the Rust
toolchain, CMake 3.25 or newer, and Xcode with its Metal toolchain (the Xcode
Command Line Tools alone have no Metal compiler). On Xcode 26 the Metal
toolchain is a separate component; `xcodebuild -showComponent MetalToolchain`
shows whether it is installed:

```bash
xcodebuild -downloadComponent MetalToolchain
git clone https://github.com/cr0sh/qw && cd qw
cargo install --locked --path crates/cli
```

Add `--features dlsa` to the `cargo install` line to build in D-LSA; to build a
tagged release, clone with `--branch <tag>`. (`cargo install --git` currently
fails without GitHub SSH access: it also fetches the `eval/tau2-bench`
submodule, whose URL is SSH.) The build fetches MLX from GitHub at the commit
pinned in [`crates/mlx-cpp/CMakeLists.txt`](crates/mlx-cpp/CMakeLists.txt)
(`7ca40033`, v0.32.4 + 35 commits), applies QW's one Metal overlay to it, and
compiles it with its Metal kernels; the first build takes a while and needs
network access.

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

## D-LSA (experimental)

D-LSA ([Distilled Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention))
is optional sparse attention for the full-attention layers of Qwen3.8-27B. The
attention kernel is unchanged; a small learned index picks which cached keys
each query reads: the first `sink` keys, the last `window` keys, and up to
`top_k` long-range keys. Forwards whose query rows all sit below 12,288 tokens
(`--dlsa-dense-below`; with `0`, up to 4,612 keys, where the list covers them
all) run the unchanged dense attention, bit for bit.
The design and the port are documented in [`docs/dlsa.md`](docs/dlsa.md).

It is off by default and absent from default builds. Install with the feature,
download the checksum-pinned index map once, and start the server on an fp16
KV cache:

```bash
cargo install --locked --path crates/cli --features dlsa   # in a clone, see Install
qw download --dlsa-map          # add --dlsa-pca for the training-free PCA map
qw serve --dlsa --no-kv-quantization
qw generate --dlsa --prompt "..."   # qw generate always uses an fp16 KV cache
```

| Flag | Environment | Default |
| --- | --- | --- |
| `--dlsa` | `QW_DLSA` (`1`/`true`/`yes`/`on` enables, `0`/`false`/`no`/`off` leaves it off) | off |
| `--dlsa-map <PATH>` | `QW_DLSA_MAP_PATH` | the `qw download --dlsa-map` cache under `~/.cache/qw/dlsa/` |
| `--dlsa-map-kind distilled\|pca` | `QW_DLSA_MAP_KIND` | `distilled` (which cached map to use when no path is given) |
| `--dlsa-sink <N>` | `QW_DLSA_SINK` | `4` |
| `--dlsa-window <N>` | `QW_DLSA_WINDOW` | `512` (also the cluster block size; the maps were distilled at 512, other values warn) |
| `--dlsa-top-k <N>` | `QW_DLSA_TOP_K` | `4096` |
| `--dlsa-top-p <N>` | `QW_DLSA_TOP_P` | `1024` (buckets a query expands, at most 2048) |
| `--dlsa-dense-below <TOKENS>` | `QW_DLSA_DENSE_BELOW` | `12288` (query rows below this position read every causal key; `0` keeps only the dense-key-set rule) |
| `--dlsa-backend metal\|mlx\|metal-select` | `QW_DLSA_BACKEND` | `metal` (the Metal kernels; `mlx` is the reference path, `metal-select` its bitwise twin with the Metal select) |

A flag wins over its variable, which wins over the default. The map is held in
bf16. D-LSA is a property of the loaded model; there is no per-request switch.

Startup errors, each naming what to fix: a `--dlsa-*` setting without
`--dlsa` (or `QW_DLSA=1`); an invalid value (the message names the flag or
variable it came from); a missing map ("run `qw download --dlsa-map` or set
QW_DLSA_MAP_PATH"); a cached map whose sha256 differs from the pinned one;
and, for the server, D-LSA without `--no-kv-quantization`.

| Feature | Under D-LSA |
| --- | --- |
| fp16 KV (`--no-kv-quantization`) | supported, with snapshots and the prefix cache |
| int8 KV (runtime API only) | supported, without snapshots (as without D-LSA) |
| Turbo4 KV (the server default) | refused: `qw serve --dlsa` without `--no-kv-quantization` fails at startup |
| decoders `baseline`, `mtp`, `dflash`, `auto` | supported; `auto` selects as usual |
| grammar-constrained output (`json_schema`, tools) | supported |
| image inputs | refused with HTTP 400 ("image inputs are not supported while D-LSA sparse attention is enabled on this server") |

**Prefix cache.** Snapshots carry the index state next to the KV cache, so the
memory and filesystem tiers work under D-LSA as they do without it (the index
adds about 1% to an fp16 snapshot). The cache namespace includes a D-LSA
identity (the map file's sha256, map dtype, index settings, short-context
policy, `--dlsa-dense-below` and the read backend), so snapshots from a dense
server, or from one with other D-LSA settings, are never restored; dense
namespaces are unchanged by the feature.

At startup the server logs one `D-LSA armed` line (`phase="server.dlsa"`) with
the map path, its sha256, kind, dtype, the index settings and `dense_below`;
each request logs `phase="dlsa.request"` at debug level with the index
counters and bytes.

With the Metal kernels, D-LSA is faster than dense attention from about 20k
tokens on (prefill 1.05× at 32k, 1.66× at 128k, 2.66× at 256k; decode flat at
~39 ms/token) and within 0.5% below that, where `--dlsa-dense-below` (default
12,288) keeps the dense read. Measurements are in
[`docs/dlsa.md`](docs/dlsa.md) §15.

Accuracy: on a RULER subsample (upstream's protocol, 13 tasks, 10 rows per task)
at 32k, no task degrades under D-LSA with the defaults: scores are identical on
12 of 13 tasks and cwe is +0.010 (two rows change, one each way), mean 0.968
for both, equal to upstream's bf16 D-LSA result. Longer contexts are not
measured yet, so it stays opt-in; see [`docs/dlsa.md`](docs/dlsa.md) §16.

## Performance

`cargo bench -p qw-runtime --bench single_user_throughput` (tokens/s), measured
on 2026-11-07 at commit `5519b07` on a Mac Studio with an Apple M4 Max (40-core
GPU, 64 GB of unified memory, macOS 27.2), with the default
[Jundot/Qwen3.8-27B-oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp)
checkpoint, Turbo4 KV cache and greedy sampling:

| Context                    | Target prefill | DFlash2 decode | Bundled MTP (k=3) decode |
| -------------------------- | -------------: | -------------: | -----------------------: |
| Fresh                      |        254.250 |         54.751 |                   58.720 |
| 10,337-token cached prefix |        232.508 |         52.924 |                   53.496 |
| 64,297-token cached prefix |        155.498 |         35.835 |                   34.549 |

Fresh prefill processes 4,341 prompt tokens; cached-prefix prefill processes
only the 288 newly appended tokens, not the cached prefix. Decode excludes
prefill and measures 127 output tokens after the first token. Each cell is one
warmup and three timed repetitions. The DFlash2 column is the default build;
the MTP column is a separate run with `--no-default-features` (its target
prefill was within 0.12% of the column shown). The raw logs, commands and
checkpoint hashes are in
[`eval/benchmarks/2026-11-07-5519b07/`](eval/benchmarks/2026-11-07-5519b07/).

## Development

Cache layout, GPU serialization, and benchmark commands are collected in
[`DEVELOPMENT.md`](DEVELOPMENT.md). When running GPU work from more than one
worktree, always invoke it through `./gpu-lock -- ...`; do not delete the
shared lock file.

## Evaluation

Qwen3.8-27B (medium reasoning) achieved **43.3% Pass@1** on τ³-Banking (**42/97
tasks**, one trial per task), compared with **47.4%** reported by Artificial
Analysis as an average over five runs. Our single-run result may be affected by
sampling variability. The evaluation took approximately 26 hours on a single
M4 Max machine, so we did not run additional trials.

The τ³ banking evaluation launch and capture rules are documented in
[`eval/README.md`](eval/README.md).

## Acknowledgements

This repository is heavily AI-assisted. Commit messages identify the coding
agent/model used, in an `Assisted-by` footer or, in later commits, `Agent` and
`Model` trailers.

QW is inspired by antirez's [ds4](https://github.com/antirez/ds4) inference
engine for its minimalism and simplicity.

The repository started from a stripped version of `mlxcel-core` from
[mlxcel](https://github.com/lablup/mlxcel). We appreciate
[Lablup](https://lablup.com/)'s open-source contributions. The vendored copy in
[`crates/mlxcel-core`](crates/mlxcel-core) has since diverged and is not kept in
sync with upstream: it was trimmed to a Metal-only binding layer for what QW
uses, carries QW-specific additions such as the D-LSA Metal kernels, and takes
selected upstream fixes by hand.

QW adopts optimization approaches from [oMLX](https://github.com/Jundot/oMLX),
[ds4](https://github.com/antirez/ds4), and
[qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge).
Derivative work is attributed in the relevant commit messages.
