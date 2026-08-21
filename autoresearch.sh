#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

[[ -f .autoresearch.env ]] || {
  echo ".autoresearch.env is required" >&2
  exit 1
}

set -a
# shellcheck disable=SC1091
source .autoresearch.env
set +a

: "${QW_BENCH_MODEL:?QW_BENCH_MODEL must point to the local target checkpoint}"
[[ -d "$QW_BENCH_MODEL" ]] || {
  echo "QW_BENCH_MODEL is not a directory: $QW_BENCH_MODEL" >&2
  exit 1
}
export QW_MODEL_PATH="$QW_BENCH_MODEL"

draft_model="${QW_BENCH_DRAFT_MODEL:-$HOME/.cache/qw/models/incoai/Qwen3.8-27B-DFlash2}"
[[ -d "$draft_model" ]] || {
  echo "DFlash2 draft checkpoint is not a directory: $draft_model" >&2
  exit 1
}
export QW_BENCH_DRAFT_MODEL="$draft_model"

output="$(mktemp)"
trap 'rm -f "$output"' EXIT

CARGO_TERM_COLOR=never cargo bench -p qw-runtime --bench single_user_throughput -- \
  'single_user_decode/mtp_k3$' --noplot 2>&1 | tee "$output"

value="$(
  awk '
    /^single_user_decode\/mtp_k3$/ { benchmark = 1; next }
    benchmark && /^[[:space:]]*thrpt:/ && /elem\/s/ {
      print $4
      exit
    }
  ' "$output"
)"
edit_distance="$(
  awk '
    /^MTP_CORRECTNESS / {
      for (field = 1; field <= NF; field++) {
        if ($field ~ /^token_edit_distance=/) {
          split($field, parts, "=")
          print parts[2]
          exit
        }
      }
    }
  ' "$output"
)"
acceptance="$(
  awk '
    /^MTP_PROFILE / {
      for (field = 1; field <= NF; field++) {
        if ($field ~ /^acceptance=/) {
          split($field, parts, "=")
          sub(/%$/, "", parts[2])
          print parts[2]
          exit
        }
      }
    }
  ' "$output"
)"

[[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "Failed to parse single_user_decode/mtp_k3 throughput" >&2
  exit 1
}
[[ "$edit_distance" =~ ^[0-9]+$ ]] || {
  echo "Failed to parse MTP token edit distance" >&2
  exit 1
}
[[ "$acceptance" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "Failed to parse MTP acceptance percentage" >&2
  exit 1
}
printf 'METRIC mtp_k3_decode_tokens_per_second=%s\n' "$value"
printf 'METRIC mtp_k3_token_edit_distance=%s\n' "$edit_distance"
printf 'METRIC mtp_k3_acceptance_percentage=%s\n' "$acceptance"

