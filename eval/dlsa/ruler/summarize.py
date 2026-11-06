#!/usr/bin/env python3
"""Summarize run_ruler.py result directories of several QW configurations into markdown tables.

usage: summarize.py --runs dense=DIR dlsa=DIR [dlsa-pca=DIR ...] [--extrapolate 100] [--compare dense dlsa]
                    [--json PATH]

A bare DIR takes its label from DIR/meta.json (else the directory name). The first run is the reference
(dense). Per length:
  (a) task x configuration scores: each configuration's own mean over its finished rows (n shown), then the
      means over the rows every configuration finished ("paired"), with deltas vs the reference, the 13-task
      mean, and the gap to the upstream README's D-LSA (Flat) bf16 result for Qwen3.8-27B;
  (b) timing per configuration and an estimate of the hours N rows per task would take;
  (c) rows that errored or stopped at max_tokens;
  (d) where two configurations' rows disagree (score, or the reply text), to attribute a degradation.
Scores follow upstream: a task is the mean over its rows without an error, a length the mean of its tasks.
Python 3 standard library only.
"""
import argparse
import datetime
import json
import os
import statistics
import sys

from ruler_score import TASKS

# Upstream README.md:178-183 (Qwen3.8-27B, D-LSA (Flat), bf16, 1 H200); the paper reports 0.975 / 0.966
PAPER = {4096: 0.974, 32768: 0.968}
PAPER_LABEL = "upstream D-LSA (Flat) bf16, README.md:180 (paper 0.975 / 0.966)"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=DIR")
    parser.add_argument("--extrapolate", type=int, default=100, help="rows per task to estimate hours for")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="labels for the agreement table")
    parser.add_argument("--json", help="also write the numbers here")
    return parser.parse_args(argv)


def load_run(spec):
    """Return (label, meta, {(length, task, index): final record}) for LABEL=DIR or DIR."""
    label, _, directory = spec.rpartition("=")
    meta = {}
    if os.path.exists(os.path.join(directory, "meta.json")):
        with open(os.path.join(directory, "meta.json")) as source:
            meta = json.load(source)
    label = label or meta.get("label") or os.path.basename(os.path.normpath(directory))
    records = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".jsonl"):
            continue
        # Line by line in binary: a record torn inside a multibyte character is skipped, not fatal
        with open(os.path.join(directory, name), "rb") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    return label, meta, final_records(records)


def final_records(records):
    """Return each row's record that counts: its last one without an error, else its last one."""
    out = {}
    for record in records:
        key = (record["length"], record["task"], record["index"])
        if key not in out or record.get("error") is None or out[key].get("error") is not None:
            out[key] = record
    return out


def finished(final, length, task):
    """Return {index: record} of a cell's rows that finished without an error."""
    return {k[2]: r for k, r in final.items() if k[0] == length and k[1] == task and r.get("error") is None}


def mean_or_none(values):
    values = list(values)
    return statistics.mean(values) if values else None


def score_table(runs, length):
    """Return the per-task own and paired scores of one length, and their task means."""
    tasks = [t for t in TASKS if any(finished(final, length, t) for _, _, final in runs)]
    rows = []
    for task in tasks:
        done = [finished(final, length, task) for _, _, final in runs]
        paired = sorted(set.intersection(*(set(d) for d in done)))
        rows.append(
            {
                "task": task,
                "own": [mean_or_none(r["score"] for r in d.values()) for d in done],
                "n": [len(d) for d in done],
                "paired_n": len(paired),
                "paired": [mean_or_none(d[i]["score"] for i in paired) if paired else None for d in done],
            }
        )
    columns = range(len(runs))
    own_mean = [mean_or_none(r["own"][c] for r in rows if r["own"][c] is not None) for c in columns]
    own_tasks = [sum(r["own"][c] is not None for r in rows) for c in columns]
    paired_rows = [r for r in rows if r["paired_n"]]
    paired_mean = [mean_or_none(r["paired"][c] for r in paired_rows) for c in columns]
    return {
        "length": length,
        "labels": [label for label, _, _ in runs],
        "tasks": rows,
        "own_mean": own_mean,
        "own_tasks": own_tasks,
        "paired_mean": paired_mean,
        "paired_tasks": len(paired_rows),
        "paper": PAPER.get(length),
    }


def timing_table(runs, length, extrapolate):
    """Return per configuration: rows, latency and token means, total time, and hours for N rows per task."""
    out = []
    for label, _, final in runs:
        cell = [r for k, r in final.items() if k[0] == length]
        ok = [r for r in cell if r.get("error") is None]
        if not cell:
            continue
        latency = [r["latency_s"] for r in ok]
        mean_latency = mean_or_none(latency)
        per_task = [mean_or_none(r["latency_s"] for r in ok if r["task"] == t) for t in TASKS]
        estimate = (
            extrapolate * sum(m if m is not None else mean_latency for m in per_task) / 3600
            if mean_latency is not None
            else None
        )
        starts = [parse_time(r["started_at"]) for r in cell]
        ends = [parse_time(r["finished_at"]) for r in cell]
        out.append(
            {
                "label": label,
                "length": length,
                "rows": len(ok),
                "errors": len(cell) - len(ok),
                "mean_latency_s": mean_latency,
                "median_latency_s": statistics.median(latency) if latency else None,
                "mean_prompt_tokens": mean_or_none(r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None),
                "mean_completion_tokens": mean_or_none(
                    r["completion_tokens"] for r in ok if r["completion_tokens"] is not None
                ),
                "mean_cached_tokens": mean_or_none(r["cached_tokens"] for r in ok if r["cached_tokens"] is not None),
                "latency_sum_h": sum(r["latency_s"] for r in cell) / 3600,
                "span_h": (max(ends) - min(starts)).total_seconds() / 3600,
                "estimate_h": estimate,
                "estimate_rows": extrapolate,
            }
        )
    return out


def parse_time(text):
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def flagged(runs):
    """Return rows that end in an error or were cut at max_tokens (those are still scored, as upstream does)."""
    out = []
    for label, _, final in runs:
        order = sorted(final.items(), key=lambda kv: (kv[0][0], TASKS.index(kv[0][1]), kv[0][2]))
        for (length, task, index), r in order:
            if r.get("error") is not None or r.get("finish_reason") == "length":
                out.append(
                    {
                        "label": label,
                        "length": length,
                        "task": task,
                        "index": index,
                        "issue": f"error: {r['error']}" if r.get("error") is not None else "finish_reason=length",
                        "completion_tokens": r.get("completion_tokens"),
                        "score": r.get("score"),
                    }
                )
    return out


def agreement(a, b, length):
    """Return per task where run b's paired rows differ from run a's: by score and by reply text."""
    out = []
    for task in TASKS:
        da, db = finished(a[2], length, task), finished(b[2], length, task)
        paired = sorted(set(da) & set(db))
        if not paired:
            continue
        lower = [i for i in paired if db[i]["score"] < da[i]["score"]]
        higher = [i for i in paired if db[i]["score"] > da[i]["score"]]
        text = [i for i in paired if db[i]["content"] != da[i]["content"]]
        out.append(
            {
                "task": task,
                "paired_n": len(paired),
                "score_differs": len(lower) + len(higher),
                "b_lower": lower,
                "b_higher": higher,
                "content_differs": len(text),
            }
        )
    return out


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def delta(value, reference):
    return "—" if value is None or reference is None else f"{value - reference:+.3f}"


def render_scores(table):
    labels, out = table["labels"], []
    P = out.append
    P(f"### RULER {table['length']}: task scores\n")
    head = ["task", *(f"{lab} (n)" for lab in labels), "paired n", *(f"{lab} paired" for lab in labels)]
    head += [f"Δ {lab} − {labels[0]}" for lab in labels[1:]]
    P("| " + " | ".join(head) + " |")
    P("|" + "---|" * len(head))
    for r in table["tasks"]:
        cells = [r["task"], *(f"{fmt(s)} ({n})" for s, n in zip(r["own"], r["n"])), str(r["paired_n"])]
        cells += [fmt(s) for s in r["paired"]]
        cells += [delta(s, r["paired"][0]) for s in r["paired"][1:]]
        P("| " + " | ".join(cells) + " |")
    cells = [f"**mean** (tasks: own {'/'.join(map(str, table['own_tasks']))}, paired {table['paired_tasks']})"]
    cells += [f"**{fmt(s)}**" for s in table["own_mean"]]
    cells += [""] + [f"**{fmt(s)}**" for s in table["paired_mean"]]
    cells += [f"**{delta(s, table['paired_mean'][0])}**" for s in table["paired_mean"][1:]]
    P("| " + " | ".join(cells) + " |")
    if table["paper"] is not None:
        P(f"\nvs {PAPER_LABEL}: {table['paper']:.3f} at {table['length']}")
        for i, lab in enumerate(labels):
            P(
                f"- {lab}: own mean {delta(table['own_mean'][i], table['paper'])}, "
                f"paired mean {delta(table['paired_mean'][i], table['paper'])}"
                + ("" if table["own_tasks"][i] == len(TASKS) else f" (only {table['own_tasks'][i]} of 13 tasks)")
            )
    return out


def render_timing(rows):
    out = ["### Timing\n"]
    P = out.append
    est = rows[0]["estimate_rows"] if rows else 0
    P(
        "| config | length | rows | errors | mean s | median s | mean prompt tok | mean completion tok | "
        f"mean cached tok | sum of latencies h | first start → last finish h | est. h for {est} rows × 13 tasks |"
    )
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        P(
            f"| {r['label']} | {r['length']} | {r['rows']} | {r['errors']} | {fmt(r['mean_latency_s'], 1)} | "
            f"{fmt(r['median_latency_s'], 1)} | {fmt(r['mean_prompt_tokens'], 0)} | "
            f"{fmt(r['mean_completion_tokens'], 1)} | {fmt(r['mean_cached_tokens'], 0)} | "
            f"{r['latency_sum_h']:.2f} | {r['span_h']:.2f} | {fmt(r['estimate_h'], 1)} |"
        )
    P("\nThe estimate is N × the sum over the 13 tasks of each task's mean latency (a task not run yet takes the")
    P("length's mean), one request at a time.")
    return out


def render_flagged(rows):
    out = ["### Errors and replies cut at max_tokens\n"]
    if not rows:
        return out + ["none"]
    out.append("| config | length | task | index | issue | completion tok | score |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        issue = r["issue"].replace("|", "\\|").replace("\n", " ")[:160]
        out.append(
            f"| {r['label']} | {r['length']} | {r['task']} | {r['index']} | {issue} | "
            f"{r['completion_tokens']} | {fmt(r['score'])} |"
        )
    return out


def render_agreement(a, b, length, rows):
    out = [f"### Agreement {length}: {b} vs {a} (rows both finished)\n"]
    out.append(f"| task | paired n | score differs | {b} lower (indices) | {b} higher (indices) | reply text differs |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['task']} | {r['paired_n']} | {r['score_differs']} | {len(r['b_lower'])} {r['b_lower'][:12]} | "
            f"{len(r['b_higher'])} {r['b_higher'][:12]} | {r['content_differs']} |"
        )
    return out


def summarize(runs, extrapolate, compare):
    """Return (markdown lines, the same numbers as a dict)."""
    lengths = sorted({k[0] for _, _, final in runs for k in final})
    numbers = {"scores": [], "timing": [], "flagged": flagged(runs), "agreement": []}
    out = []
    for length in lengths:
        table = score_table(runs, length)
        numbers["scores"].append(table)
        out += render_scores(table) + [""]
    for length in lengths:
        numbers["timing"] += timing_table(runs, length, extrapolate)
    out += render_timing(numbers["timing"]) + [""]
    out += render_flagged(numbers["flagged"]) + [""]
    if compare:
        by_label = {run[0]: run for run in runs}
        missing = [lab for lab in compare if lab not in by_label]
        if missing:
            raise SystemExit(f"--compare: no run labelled {missing}; labels are {list(by_label)}")
        a, b = by_label[compare[0]], by_label[compare[1]]
        for length in lengths:
            rows = agreement(a, b, length)
            numbers["agreement"].append({"a": a[0], "b": b[0], "length": length, "tasks": rows})
            out += render_agreement(a[0], b[0], length, rows) + [""]
    return out, numbers


def main(argv=None):
    args = parse_args(argv)
    runs = [load_run(spec) for spec in args.runs]
    labels = [label for label, _, _ in runs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"labels must differ: {labels} - name them with LABEL=DIR")
    compare = args.compare or (labels[:2] if len(runs) > 1 else None)
    out, numbers = summarize(runs, args.extrapolate, compare)
    print("\n".join(out).rstrip("\n"))  # exactly one newline at the end
    if args.json:
        with open(args.json, "w") as sink:
            json.dump(numbers, sink, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
