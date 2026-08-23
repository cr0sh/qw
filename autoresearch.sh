#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE="$ROOT_DIR/.autoresearch.env"

if [[ ! -f "$ENV_FILE" ]]; then
    printf 'missing benchmark environment: %s\n' "$ENV_FILE" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"
MODEL_DIR=${QW_BENCH_MODEL:?QW_BENCH_MODEL is not set in .autoresearch.env}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    printf 'missing target checkpoint: %s\n' "$MODEL_DIR" >&2
    exit 1
fi

OUTPUT=$(mktemp)
trap 'rm -f "$OUTPUT"' EXIT

(
    cd "$ROOT_DIR"
    CARGO_TERM_COLOR=never \
    QW_MODEL_PATH="$MODEL_DIR" \
        cargo bench -p qw-runtime --bench single_user_throughput -- \
        single_user_decode/fresh_mtp_k3 --quick
) 2>&1 | tee "$OUTPUT"

python3 - "$OUTPUT" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
benchmark_id = "single_user_decode/fresh_mtp_k3"
result_pattern = (
    rf"(?ms)^{re.escape(benchmark_id)}\s*\n"
    r".*?time:\s*\[\s*([0-9.eE+-]+)\s+s\s+"
    r"([0-9.eE+-]+)\s+s\s+([0-9.eE+-]+)\s+s\s*\]\s*\n"
    r"\s*thrpt:\s*\[\s*([0-9.eE+-]+)\s+elem/s\s+"
    r"([0-9.eE+-]+)\s+elem/s\s+([0-9.eE+-]+)\s+elem/s\s*\]"
)
results = re.findall(result_pattern, text)
if len(results) != 1:
    raise SystemExit(f"expected one {benchmark_id} result, found {len(results)}")

_, median_time_s, _, _, median_throughput, _ = map(float, results[0])
if median_time_s <= 0.0 or median_throughput <= 0.0:
    raise SystemExit(
        f"invalid benchmark result: time={median_time_s}, throughput={median_throughput}"
    )

profiles = re.findall(
    r"^MTP_PROFILE tokens=(\d+) accepted=(\d+) proposed=(\d+) "
    r"acceptance=([0-9.]+)% forwards=(\d+)",
    text,
    flags=re.MULTILINE,
)
if len(profiles) != 1:
    raise SystemExit(f"expected one fresh MTP profile, found {len(profiles)}")
completion_tokens, accepted, proposed, acceptance_pct, target_forwards = profiles[0]
for name, value in (
    ("completion_tokens", completion_tokens),
    ("proposed_draft_tokens", proposed),
    ("target_forwards", target_forwards),
):
    if int(value) <= 0:
        raise SystemExit(f"invalid {name}: {value}")

correctness = re.findall(
    r"^MTP_CORRECTNESS token_edit_distance=(\d+)$",
    text,
    flags=re.MULTILINE,
)
if len(correctness) != 1:
    raise SystemExit(f"expected one fresh MTP correctness result, found {len(correctness)}")

print(f"METRIC fresh_mtp_k3_elem_s={median_throughput:.6f}")
print(f"METRIC fresh_mtp_k3_time_s={median_time_s:.6f}")
print(f"METRIC token_edit_distance={correctness[0]}")
print(f"METRIC completion_tokens={completion_tokens}")
print(f"METRIC accepted_draft_tokens={accepted}")
print(f"METRIC proposed_draft_tokens={proposed}")
print(f"METRIC acceptance_pct={float(acceptance_pct):.6f}")
print(f"METRIC target_forwards={target_forwards}")
PY
