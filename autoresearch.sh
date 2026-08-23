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
        long_64k_mtp_k3 --quick
) 2>&1 | tee "$OUTPUT"

python3 - "$OUTPUT" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()

benchmark_results = re.findall(
    r"(?ms)^(single_user_(?:prefill|decode)/long_64k_mtp_k3)\s*\n"
    r".*?thrpt:\s*\[\s*[0-9.eE+-]+\s+elem/s\s+"
    r"([0-9.eE+-]+)\s+elem/s\s+[0-9.eE+-]+\s+elem/s\s*\]",
    text,
)
expected_ids = {
    "single_user_decode/long_64k_mtp_k3",
    "single_user_prefill/long_64k_mtp_k3",
}
results = {}
for benchmark_id, value_text in benchmark_results:
    if benchmark_id in results:
        raise SystemExit(f"duplicate benchmark result for {benchmark_id}")
    results[benchmark_id] = float(value_text)
if set(results) != expected_ids:
    missing = sorted(expected_ids - set(results))
    unexpected = sorted(set(results) - expected_ids)
    raise SystemExit(
        f"expected exact long_64k_mtp_k3 benchmark IDs; missing={missing}, unexpected={unexpected}"
    )
for benchmark_id, value in results.items():
    if not value > 0.0:
        raise SystemExit(f"invalid throughput for {benchmark_id}: {value}")

profiles = re.findall(
    r"MTP_LONG_CONTEXT_PROFILE context=64k tokens=(\d+) prefix_tokens=(\d+) "
    r"accepted=(\d+) proposed=(\d+) acceptance=([0-9.]+)% forwards=(\d+)",
    text,
)
if len(profiles) != 1:
    raise SystemExit(f"expected one 64k-context MTP profile, found {len(profiles)}")
completion_tokens, prefix_tokens, accepted, proposed, acceptance_pct, target_forwards = (
    profiles[0]
)
for name, value in (
    ("completion_tokens", completion_tokens),
    ("prefix_tokens", prefix_tokens),
    ("proposed_draft_tokens", proposed),
    ("target_forwards", target_forwards),
):
    if int(value) <= 0:
        raise SystemExit(f"invalid {name}: {value}")

correctness = re.findall(
    r"MTP_LONG_CONTEXT_CORRECTNESS context=64k token_edit_distance=(\d+) "
    r"token_fingerprint=(\d+)",
    text,
)
if len(correctness) != 1:
    raise SystemExit(
        f"expected one 64k-context correctness result, found {len(correctness)}"
    )
token_edit_distance, token_fingerprint = correctness[0]
if int(token_fingerprint) <= 0:
    raise SystemExit(f"invalid token fingerprint: {token_fingerprint}")

print(
    "METRIC long_64k_mtp_k3_prefill_elem_s="
    f"{results['single_user_prefill/long_64k_mtp_k3']:.6f}"
)
print(
    "METRIC long_64k_mtp_k3_elem_s="
    f"{results['single_user_decode/long_64k_mtp_k3']:.6f}"
)
print(f"METRIC token_edit_distance={token_edit_distance}")
print(f"METRIC token_fingerprint={token_fingerprint}")
print(f"METRIC completion_tokens={completion_tokens}")
print(f"METRIC prefix_tokens={prefix_tokens}")
print(f"METRIC accepted_draft_tokens={accepted}")
print(f"METRIC proposed_draft_tokens={proposed}")
print(f"METRIC acceptance_pct={float(acceptance_pct):.6f}")
print(f"METRIC target_forwards={target_forwards}")
PY
