#!/usr/bin/env bash
set -euo pipefail
if [[ -f .autoresearch.env ]]; then
    # Local checkpoint configuration; this file is intentionally ignored.
    source .autoresearch.env
fi


: "${QW_BENCH_MODEL:?set QW_BENCH_MODEL to the local Qwen3.8 27B checkpoint}"
export LC_ALL=C

output="$(mktemp)"
trap 'rm -f "$output"' EXIT

cargo bench -p qw-runtime --bench single_user_throughput -- --test | tee "$output" >&2

awk '
function value(prefix,    i, parts) {
    for (i = 1; i <= NF; i++) {
        split($i, parts, "=")
        if (parts[1] == prefix) return parts[2]
    }
    return ""
}
/^DECODE_BENCH_SUMMARY / || /^MTP_BENCH_SUMMARY / {
    tps = value("decode_tokens_per_second") + 0
    k = value("k") + 0
    if (!found || tps > best_tps) {
        found = 1
        best_tps = tps
        best_k = k
    }
}
END {
    if (!found || best_tps <= 0) exit 1
    printf "METRIC decode_tps=%.6f\n", best_tps
    printf "METRIC best_block_size=%d\n", best_k
}
' "$output"
