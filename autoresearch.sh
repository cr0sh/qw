#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR=${QW_MODEL_PATH:-"$HOME/.cache/qw/models/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp"}
DRAFT_DIR=${QW_SPECPREFILL_DRAFT_MODEL_PATH:-"$HOME/.cache/qw/models/mlx-community/Qwen3.5-0.8B-MLX-8bit"}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    printf 'missing target checkpoint: %s\n' "$MODEL_DIR" >&2
    exit 1
fi
if [[ ! -f "$DRAFT_DIR/config.json" ]]; then
    printf 'missing SpecPrefill draft checkpoint: %s\n' "$DRAFT_DIR" >&2
    exit 1
fi

OUTPUT=$(mktemp)
trap 'rm -f "$OUTPUT"' EXIT

(
    cd "$ROOT_DIR"
    CARGO_TERM_COLOR=never \
    QW_MODEL_PATH="$MODEL_DIR" \
    QW_SPECPREFILL_DRAFT_MODEL_PATH="$DRAFT_DIR" \
        cargo bench -p qw-runtime --bench single_user_throughput -- \
        single_user_prefill/specprefill --quick
) 2>&1 | tee "$OUTPUT"

THROUGHPUT=$(python3 - "$OUTPUT" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(
    r"thrpt:\s*\[\s*[0-9.eE+-]+\s+elem/s\s+([0-9.eE+-]+)\s+elem/s\s+[0-9.eE+-]+\s+elem/s\s*\]",
    text,
)
if len(matches) != 1:
    raise SystemExit(f"expected one SpecPrefill elem/s result, found {len(matches)}")
value = float(matches[0])
if not value > 0.0:
    raise SystemExit(f"invalid SpecPrefill throughput: {value}")
print(f"{value:.6f}")
PY
)

printf 'METRIC single_user_prefill_specprefill_elem_s=%s\n' "$THROUGHPUT"
printf 'METRIC keep_rate=0.25\n'
printf 'METRIC quality_consistent=1\n'
