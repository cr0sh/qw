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
    QW_BENCH_LONG_CONTEXT_ONLY=64k \
        cargo bench -p qw-runtime --bench single_user_throughput -- \
        single_user_decode/long_64k_mtp_k3 --quick
) 2>&1 | tee "$OUTPUT"

python3 - "$OUTPUT" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()

throughput = re.findall(
    r"thrpt:\s*\[\s*[0-9.eE+-]+\s+elem/s\s+([0-9.eE+-]+)\s+elem/s\s+[0-9.eE+-]+\s+elem/s\s*\]",
    text,
)
if len(throughput) != 1:
    raise SystemExit(
        f"expected one long_64k_mtp_k3 elem/s result, found {len(throughput)}"
    )
throughput_value = float(throughput[0])
if not throughput_value > 0.0:
    raise SystemExit(f"invalid long_64k_mtp_k3 throughput: {throughput_value}")

profiles = re.findall(
    r"MTP_LONG_CONTEXT_PROFILE context=64k .*?acceptance=([0-9.]+)% forwards=(\d+)",
    text,
)
if len(profiles) != 1:
    raise SystemExit(f"expected one 64k-context MTP profile, found {len(profiles)}")
acceptance_pct, target_forwards = profiles[0]

distances = re.findall(
    r"MTP_LONG_CONTEXT_CORRECTNESS context=64k token_edit_distance=(\d+)",
    text,
)
if len(distances) != 1:
    raise SystemExit(
        f"expected one 64k-context correctness result, found {len(distances)}"
    )

print(f"METRIC long_64k_mtp_k3_elem_s={throughput_value:.6f}")
print(f"METRIC token_edit_distance={distances[0]}")
print(f"METRIC acceptance_pct={float(acceptance_pct):.6f}")
print(f"METRIC target_forwards={target_forwards}")
PY
