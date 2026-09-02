#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE="$ROOT_DIR/.autoresearch.env"
RESULT_FILE="$ROOT_DIR/target/autoresearch-single-user-throughput.txt"

if [[ ! -f "$ENV_FILE" ]]; then
    printf 'missing benchmark environment: %s\n' "$ENV_FILE" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"
MODEL_DIR=${QW_BENCH_MODEL:?QW_BENCH_MODEL is not set in .autoresearch.env}
DRAFT_MODEL_DIR=${QW_BENCH_DRAFT_MODEL:?QW_BENCH_DRAFT_MODEL is not set in .autoresearch.env}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    printf 'missing target checkpoint: %s\n' "$MODEL_DIR" >&2
    exit 1
fi

if [[ ! -f "$DRAFT_MODEL_DIR/config.json" ]]; then
    printf 'missing DFlash2 checkpoint: %s\n' "$DRAFT_MODEL_DIR" >&2
    exit 1
fi

rm -f "$RESULT_FILE"

CARGO_TERM_COLOR=never \
QW_MODEL_PATH="$MODEL_DIR" \
QW_BENCH_DRAFT_MODEL="$DRAFT_MODEL_DIR" \
    cargo bench --offline -p qw-runtime --bench single_user_throughput \
    --features dflash2 -- single_user_decode/long_64k_dflash2 |
    tee "$RESULT_FILE"

python3 - "$RESULT_FILE" <<'PY'
import math
import pathlib
import sys

result_file = pathlib.Path(sys.argv[1])
benchmark_id = "single_user_decode/long_64k_dflash2"
row = None
for line in result_file.read_text(encoding="utf-8").splitlines():
    if not line.startswith("BENCHMARK_RESULT "):
        continue
    fields = dict(field.split("=", 1) for field in line.split()[1:])
    if fields.get("label") == benchmark_id:
        row = fields
        break

if row is None:
    raise SystemExit(f"missing benchmark result for {benchmark_id!r}")

decode_tokens = int(row["tokens_per_repetition"])
mean_seconds = float(row["mean_seconds"])
stddev_seconds = float(row["stddev_seconds"])
tokens_per_second = float(row["tokens_per_second"])
for name, value in (
    ("decode_tokens", decode_tokens),
    ("mean_seconds", mean_seconds),
    ("tokens_per_second", tokens_per_second),
):
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid {name}: {value!r}")
if not math.isfinite(stddev_seconds) or stddev_seconds < 0:
    raise SystemExit(f"invalid stddev_seconds: {stddev_seconds!r}")

print(f"METRIC dflash2_tokens_per_second={tokens_per_second:.6f}")
print(f"METRIC dflash2_decode_latency_ms={mean_seconds * 1_000.0:.6f}")
print(f"METRIC dflash2_decode_stddev_ms={stddev_seconds * 1_000.0:.6f}")
print(f"METRIC dflash2_decode_tokens={decode_tokens}")
PY
