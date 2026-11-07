# `single_user_throughput` at `5519b07` — 2026-11-07

Raw outputs behind the README's Performance table.

- Source: commit `5519b07947bab98b4b4f4b66081f26350d14f5cc` (no tracked source
  changes in the worktree while the benches ran).
- Machine: Mac Studio (`Mac16,9`), Apple M4 Max, 16-core CPU (12P + 4E),
  40-core GPU, 64 GB unified memory, macOS 27.2 (26B123); Xcode 26.6 with Metal
  toolchain 17F109; `rustc 1.99.1`; MLX pinned at `7ca40033`.
- Checkpoints: the resolver defaults `~/.cache/qw/models/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp`
  (target) and `~/.cache/qw/models/incoai/Qwen3.8-27B-DFlash2` (draft); file
  hashes in [`model-sha256.txt`](model-sha256.txt).
- Environment: no `QW_*` or `MLXCEL_*` variables set; Turbo4 KV cache (the
  bench's fixed mode); greedy sampling.
- Fixtures: the long-context fixtures under `target/qw-bench-fixtures/` did not
  exist (the target directory had been cleaned), so each run regenerated them
  for its own variant (`LONG_CONTEXT_FIXTURE_CACHE stored` lines).

| File | Command (from the repository root) | Run (UTC) |
| --- | --- | --- |
| [`dflash2-default.log`](dflash2-default.log) | `./gpu-lock -- cargo bench -p qw-runtime --bench single_user_throughput` | 08:09:17–08:24:36 |
| [`mtp-no-default-features.log`](mtp-no-default-features.log) | `./gpu-lock -- cargo bench -p qw-runtime --bench single_user_throughput --no-default-features` | 08:24:54–08:33:58 |

The two runs were serialized through `gpu-lock`; no Cargo build or other GPU
work ran during them. Each log starts with a `#` header (command, commit,
machine, start) and ends with the exit status and end time; the
`BENCHMARK_RESULT` rows are the bench's own output.

Results (tokens/s):

| Case | Default (DFlash2) | `--no-default-features` (bundled MTP, k=3) |
| --- | ---: | ---: |
| Prefill, fresh (4,341 tokens) | 254.250 | 254.268 |
| Prefill, 10,337 cached + 288 new | 232.508 | 232.632 |
| Prefill, 64,297 cached + 288 new | 155.498 | 155.671 |
| Decode, fresh (127 tokens) | 54.751 | 58.720 |
| Decode, 10,337 cached | 52.924 | 53.496 |
| Decode, 64,297 cached | 35.835 | 34.549 |
