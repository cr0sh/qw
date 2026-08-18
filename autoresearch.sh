#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .autoresearch.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .autoresearch.env
  set +a
fi

: "${QW_BENCH_MODEL:?QW_BENCH_MODEL must point to the local Qwen3.8-27B MTP checkpoint}"
[[ -d "$QW_BENCH_MODEL" ]] || {
  echo "QW_BENCH_MODEL is not a directory: $QW_BENCH_MODEL" >&2
  exit 1
}

output="$(mktemp)"
trap 'rm -f "$output"' EXIT

CARGO_TERM_COLOR=never cargo bench -p qw-runtime --bench single_user_throughput -- \
  'single_user_decode/mtp_k3' --noplot 2>&1 | tee "$output"

metric="$(awk '/^[[:space:]]*thrpt:/ && /elem\/s/ { print $4; exit }' "$output")"
[[ "$metric" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "Failed to parse single_user_decode/mtp_k3 median elem/s" >&2
  exit 1
}

printf 'METRIC single_user_decode_mtp_k3_elem_per_s=%s\n' "$metric"
