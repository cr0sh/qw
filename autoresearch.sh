#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .autoresearch.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .autoresearch.env
  set +a
fi

# Unified with the resolver: QW_MODEL_PATH override, else the default cache
# path for DEFAULT_MODEL_IDENTIFIER (model_resolver.rs).
model_dir="${QW_MODEL_PATH:-$HOME/.cache/qw/models/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp}"
[[ -d "$model_dir" ]] || {
  echo "model checkpoint not found at $model_dir; set QW_MODEL_PATH to point at a local Qwen3.8-27B MTP checkpoint" >&2
  exit 1
}

output="$(mktemp)"
trap 'rm -f "$output"' EXIT

CARGO_TERM_COLOR=never cargo bench -p qw-runtime --bench single_user_throughput -- \
  'single_user_(prefill/qwen|decode/mtp_k3)$' --noplot 2>&1 | tee "$output"

metrics="$(
  awk '
    /^single_user_prefill\/qwen$/ { benchmark = "single_user_prefill_qwen_elem_per_s"; next }
    /^single_user_decode\/mtp_k3$/ { benchmark = "single_user_decode_mtp_k3_tokens_per_s"; next }
    benchmark != "" && /^[[:space:]]*thrpt:/ && /elem\/s/ {
      print benchmark "=" $4
      benchmark = ""
    }
  ' "$output"
)"

for name in \
  single_user_prefill_qwen_elem_per_s \
  single_user_decode_mtp_k3_tokens_per_s
do
  value="$(printf '%s\n' "$metrics" | awk -F= -v name="$name" '$1 == name { print $2; exit }')"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "Failed to parse $name" >&2
    exit 1
  }
  printf 'METRIC %s=%s\n' "$name" "$value"
done

