#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE="$ROOT_DIR/.autoresearch.env"
RESULT_DIR="$ROOT_DIR/target/criterion/single_user_prefill/cached_64k/new"

if [[ ! -f "$ENV_FILE" ]]; then
    printf 'missing benchmark environment: %s\n' "$ENV_FILE" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"
MODEL_DIR=${QW_BENCH_MODEL:?QW_BENCH_MODEL is not set in .autoresearch.env}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    printf 'missing Qwen checkpoint: %s\n' "$MODEL_DIR" >&2
    exit 1
fi

rm -rf "$RESULT_DIR"

CARGO_TERM_COLOR=never \
QW_MODEL_PATH="$MODEL_DIR" \
QW_BENCH_CASE=64k \
QW_BENCH_LONG_TOKENS=64000 \
    cargo bench --offline -p qw-runtime --bench single_user_throughput \
    -- single_user_prefill/cached_64k

python3 - "$RESULT_DIR" <<'PY'
import json
import math
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
with (result_dir / "benchmark.json").open(encoding="utf-8") as file:
    benchmark = json.load(file)
with (result_dir / "estimates.json").open(encoding="utf-8") as file:
    estimates = json.load(file)

benchmark_id = "single_user_prefill/cached_64k"
if benchmark.get("full_id") != benchmark_id:
    raise SystemExit(
        f"expected benchmark {benchmark_id!r}, got {benchmark.get('full_id')!r}"
    )

throughput = benchmark.get("throughput")
if not isinstance(throughput, dict) or set(throughput) != {"Elements"}:
    raise SystemExit(f"expected element throughput metadata, got {throughput!r}")
tokens = throughput["Elements"]
median_ns = estimates["median"]["point_estimate"]
stddev_ns = estimates["std_dev"]["point_estimate"]
for name, value in (
    ("prefill_tokens", tokens),
    ("median_ns", median_ns),
    ("stddev_ns", stddev_ns),
):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid {name}: {value!r}")

prefill_tokens_per_second = tokens * 1_000_000_000.0 / median_ns
print(f"METRIC prefill_tokens_per_second={prefill_tokens_per_second:.6f}")
print(f"METRIC prefill_latency_ms={median_ns / 1_000_000.0:.6f}")
print(f"METRIC prefill_stddev_ms={stddev_ns / 1_000_000.0:.6f}")
print(f"METRIC prefill_tokens={tokens}")
PY
