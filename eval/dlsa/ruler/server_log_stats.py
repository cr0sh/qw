#!/usr/bin/env python3
"""Per-request prefill, decode, speculative-decoder and D-LSA statistics from a QW JSON log.

usage: server_log_stats.py LOG [--json PATH]

LOG is `qw serve --output-format json` output (stdout) or the daily file `~/.cache/qw/log/qw.log.<date>`: one
tracing JSON object per line, `{"level", "fields": {...}, "target", "span": {...}}`; any other line is ignored.
Events are grouped by the request's `response_id` (the `generation` span, or `chat_id`), the id the chat reply
carries and run_ruler.py records. What is read, and the level it is logged at:
  - `event=prefill.complete` / `event=decode.complete` (qw_server::engine, INFO): tps and token counts.
  - `phase=generation.prefill` (qw_runtime::provider, DEBUG): the decoder route (baseline, mtp, dflash2).
  - `phase=model.complete` (qw_runtime::provider, DEBUG): on the MTP route, `mtp_proposed_draft_tokens`,
    `mtp_accepted_draft_tokens`, `mtp_acceptance_percentage`, `mtp_speculative_rounds`.
  - `phase=dflash2.completed` (qw_runtime::provider, DEBUG): `stats`, the Debug text of
    Dflash2GenerationStats, whose `accepted_draft_tokens` / `proposed_draft_tokens` are read.
  - `phase=dlsa.request` (qw_server::engine, DEBUG, dlsa builds): dense_key_set_forwards, sparse_forwards,
    sparse_rows, keys_read, rollbacks, reopened_blocks, index_ms, select_ms, index_bytes.
stdout logs at RUST_LOG (INFO by default), so the DEBUG events need
`RUST_LOG=info,qw_server::engine=debug,qw_runtime::provider=debug`; the daily file logs at
`--persistent-log-filter` / `QW_LOG`, by default `debug,...`, which already has them (unless `--no-file-logging`).
Python 3 standard library only.
"""
import argparse
import json
import re
import statistics
import sys

DLSA_FIELDS = (
    "dense_key_set_forwards",
    "sparse_forwards",
    "sparse_rows",
    "keys_read",
    "rollbacks",
    "reopened_blocks",
    "index_ms",
    "select_ms",
    "index_bytes",
)
DEBUG_NUMBER = re.compile(r"(\w+): ([0-9][0-9.]*)(?=[,\s}])")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log")
    parser.add_argument("--json", help="also write the per-request numbers here")
    return parser.parse_args(argv)


def events(lines):
    """Yield (request id or None, fields) for every JSON tracing line."""
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        fields = obj["fields"] if isinstance(obj.get("fields"), dict) else obj
        spans = [obj.get("span") or {}, *(obj.get("spans") or [])]
        rid = fields.get("chat_id") or next((s["response_id"] for s in spans if s.get("response_id")), None)
        yield rid, fields


def debug_numbers(text):
    """Return the unit-less number fields of a Rust Debug struct: `S { a: 1, b: 2.5ms, c: [1, 2] }` -> {a: 1}."""
    return {name: float(value) if "." in value else int(value) for name, value in DEBUG_NUMBER.findall(text)}


def requests(lines):
    """Return per-request statistics in the order the requests were first logged."""
    out, last = {}, None
    for rid, fields in events(lines):
        kind = fields.get("event") or fields.get("phase")
        if kind not in (
            "prefill.complete",
            "decode.complete",
            "generation.prefill",
            "model.complete",
            "dflash2.completed",
            "dlsa.request",
        ):
            continue
        # An event outside any request span belongs to the request logged last (one sequence at a time)
        rid = rid or last
        if rid is None:
            continue
        last = rid
        r = out.setdefault(rid, {"id": rid})
        if kind == "prefill.complete":
            r.update(
                prefill_tps=fields.get("tps"),
                prompt_tokens=fields.get("total_tokens"),
                prefilled_tokens=fields.get("prefilled_tokens"),
                prefix_reused_tokens=fields.get("prefix_reused_tokens"),
            )
        elif kind == "decode.complete":
            r.update(decode_tps=fields.get("tps"), decoded_tokens=fields.get("decoded_tokens"))
        elif kind == "generation.prefill":
            r["route"] = fields.get("route")
        elif kind == "model.complete" and "mtp_proposed_draft_tokens" in fields:
            r.update(
                decoder="mtp",
                proposed=fields["mtp_proposed_draft_tokens"],
                accepted=fields.get("mtp_accepted_draft_tokens"),
                rounds=fields.get("mtp_speculative_rounds"),
            )
        elif kind == "dflash2.completed":
            stats = debug_numbers(str(fields.get("stats", "")))
            r.update(
                decoder="dflash2",
                proposed=stats.get("proposed_draft_tokens"),
                accepted=stats.get("accepted_draft_tokens"),
                rounds=stats.get("speculative_rounds"),
            )
        elif kind == "dlsa.request":
            r["dlsa"] = {name: fields.get(name) for name in DLSA_FIELDS}
    for r in out.values():
        if r.get("proposed"):
            r["acceptance"] = r["accepted"] / r["proposed"]
    return list(out.values())


def fmt(value, digits=1):
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render(rows):
    out = ["| request | prompt | prefilled | reused | prefill tok/s | decoded | decode tok/s | decoder | "
           "accepted/proposed | acceptance | sparse rows | keys read | rollbacks | index ms | select ms |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r.get("dlsa") or {}
        spec = f"{r['accepted']}/{r['proposed']}" if r.get("proposed") is not None else "—"
        out.append(
            f"| {r['id'][-12:]} | {fmt(r.get('prompt_tokens'))} | {fmt(r.get('prefilled_tokens'))} | "
            f"{fmt(r.get('prefix_reused_tokens'))} | {fmt(r.get('prefill_tps'))} | {fmt(r.get('decoded_tokens'))} | "
            f"{fmt(r.get('decode_tps'))} | {r.get('decoder') or r.get('route') or '—'} | {spec} | "
            f"{fmt(r.get('acceptance'), 3)} | {fmt(d.get('sparse_rows'))} | {fmt(d.get('keys_read'))} | "
            f"{fmt(d.get('rollbacks'))} | {fmt(d.get('index_ms'))} | {fmt(d.get('select_ms'))} |"
        )
    out.append("")
    prefill = [r["prefill_tps"] for r in rows if r.get("prefill_tps") is not None]
    decode = [r["decode_tps"] for r in rows if r.get("decode_tps") is not None]
    out.append(f"requests: {len(rows)}")
    if prefill:
        out.append(f"prefill tok/s: median {statistics.median(prefill):.1f}, mean {statistics.mean(prefill):.1f}")
    if decode:
        out.append(f"decode tok/s: median {statistics.median(decode):.1f}, mean {statistics.mean(decode):.1f}")
    spec = [r for r in rows if r.get("proposed")]
    if spec:
        accepted, proposed = sum(r["accepted"] for r in spec), sum(r["proposed"] for r in spec)
        out.append(f"draft acceptance: {accepted}/{proposed} = {accepted / proposed:.3f} over {len(spec)} requests")
    dlsa = [r["dlsa"] for r in rows if r.get("dlsa")]
    if dlsa:
        totals = {k: sum(d[k] or 0 for d in dlsa) for k in DLSA_FIELDS if k != "index_bytes"}
        sums = ", ".join(f"{k}={fmt(v)}" for k, v in totals.items())
        out.append(f"dlsa.request over {len(dlsa)} requests (sums): {sums}")
    return out


def main(argv=None):
    args = parse_args(argv)
    with open(args.log, errors="replace") as source:
        rows = requests(source)
    print("\n".join(render(rows)))
    if args.json:
        with open(args.json, "w") as sink:
            json.dump(rows, sink, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
