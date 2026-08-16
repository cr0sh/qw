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
/^MTP_BENCH_SUMMARY / && value("k") == "3" {
    found = 1
    tps = value("decode_tokens_per_second") + 0
    milliseconds = value("decode_milliseconds") + 0
    acceptance = value("acceptance_percentage") + 0
}
END {
    if (!found || tps <= 0 || milliseconds <= 0) exit 1
    printf "METRIC single_stream_decode_tps=%.6f\n", tps
    printf "METRIC decode_milliseconds=%.6f\n", milliseconds
    printf "METRIC acceptance_percentage=%.6f\n", acceptance
    printf "METRIC block_size=3\n"
}
' "$output"
