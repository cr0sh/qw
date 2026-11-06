#!/usr/bin/env python3
"""Run RULER rows against a QW server, the way upstream latent-sparse-attention sends and grades them.

usage: run_ruler.py --data-root DIR --out DIR [--label dense] [--lengths 4096 32768] [--rows 100] ...

One chat request per row: the row's prompt as the only user turn, greedy, `max_tokens` 4096, the think block
off through `chat_template_kwargs`, no stop strings (upstream client.py:106-113, sampling.py:41-56,
configs/evaluation/accuracy/qwen38_27b/_base.yaml:99-104). Every reply is scored inline with the official
metric (ruler_score.py) and appended to `<out>/<task>_<length>.jsonl` the moment it lands, so a stopped run
loses only the requests in flight. Started again with the same `--out`, a run skips every row that already has
a record without an error and sends the errored ones again; rows go length by length, cheapest length first,
then task and index, so a partial run is a well-defined prefix.

A 32k row is ~145 s of GPU on this machine and the full 2 x 13 x 100 set ~50 h: everything a row costs is kept
in its record, and `<out>/meta.json` holds what the run was (configuration label, server, data hashes).
Python 3 standard library only.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from ruler_score import TASKS, prompt_set_hash, prompt_text, score_row

HERE = os.path.dirname(os.path.abspath(__file__))
RETRIES = 3
# What must stay the same for records of one output directory to be one configuration's results
PROTOCOL_KEYS = ("label", "model", "max_tokens")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--data-root", required=True, help="directory holding ruler/<task>_<length>.jsonl")
    parser.add_argument("--out", required=True, help="results directory (resumed when it exists)")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS, metavar="TASK")
    parser.add_argument("--rows", type=int, default=100, help="first N rows of each file")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requests in flight (QW runs one sequence at a time here; the paper's client kept 8 in flight)",
    )
    parser.add_argument("--timeout", type=float, default=14400, help="seconds one request may take")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--label", default="", help="configuration name, e.g. dense, dlsa, dlsa-pca")
    parser.add_argument("--note", default="", help="free text kept in meta.json")
    parser.add_argument(
        "--order",
        choices=("cheap-first", "as-given"),
        default="cheap-first",
        help="cheap-first: lengths ascending; as-given: in --lengths order",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the first request body")
    return parser.parse_args(argv)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def data_path(data_root, task, length):
    return os.path.join(data_root, "ruler", f"{task}_{length}.jsonl")


def out_path(out, task, length):
    return os.path.join(out, f"{task}_{length}.jsonl")


def load_rows(data_root, task, length, rows):
    """Return the first `rows` rows of a cell; an N-row set is the first N of the seed-42 stream."""
    path = data_path(data_root, task, length)
    with open(path) as source:
        found = [json.loads(line) for line in source if line.strip()]
    if len(found) < rows:
        raise SystemExit(f"{path}: {len(found)} rows, {rows} asked for")
    return found[:rows]


def load_records(path):
    """Return every record of a results file in append order, skipping a line torn by a killed run.

    Lines are decoded one by one, so a record cut inside a multibyte character is skipped like any other torn
    line, and the complete records before it still count.
    """
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "rb") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"{path}:{number}: unreadable record skipped (rerun)", file=sys.stderr)
    return records


def resume_state(records):
    """Return ({index: attempts so far}, {indices with a record without an error})."""
    attempts, done = {}, set()
    for record in records:
        attempts[record["index"]] = attempts.get(record["index"], 0) + 1
        if record.get("error") is None:
            done.add(record["index"])
    return attempts, done


def request_body(row, model, max_tokens):
    """The chat request upstream sends for a row - exactly these keys."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text(row)}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def post(url, body, timeout):
    """POST one body; return (parsed reply or None, error or None, HTTP attempts made).

    Connection errors, HTTP 5xx and malformed replies are retried with backoff min(2**attempt, 30) s, as
    upstream's client does; a 4xx and a timeout are recorded at once - a retry of a timed-out row would hold
    the server for another whole timeout (client.py:146-152). urllib's timeout bounds each socket wait, which
    for this non-streaming endpoint is the wait for the whole reply.
    """
    data = json.dumps(body).encode()
    error = None
    for attempt in range(RETRIES):
        if attempt:
            time.sleep(min(2 ** (attempt - 1), 30))
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                reply = json.loads(response.read())
            reply["choices"][0]["message"]  # a reply without a message is malformed, and retried
            return reply, None, attempt + 1
        except urllib.error.HTTPError as exc:
            error = f"http {exc.code}: {exc.read()[:300].decode(errors='replace')}"
            if exc.code < 500:
                return None, error, attempt + 1
        except TimeoutError as exc:
            return None, f"timeout after {timeout} s: {exc}", attempt + 1
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return None, f"timeout after {timeout} s: {exc.reason}", attempt + 1
            error = f"URLError: {exc.reason}"
        except (http.client.HTTPException, ConnectionError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"malformed reply: {type(exc).__name__}: {exc}"
    return None, error, RETRIES


def run_row(url, task, length, row, args, attempt) -> dict[str, Any]:
    """Send one row and return its record."""
    started_at, start = utc_now(), time.perf_counter()
    reply, error, http_attempts = post(url, request_body(row, args.model, args.max_tokens), args.timeout)
    record: dict[str, Any] = {
        "index": row["index"],
        "task": task,
        "length": length,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "finish_reason": None,
        "content": None,
        "reasoning_content": None,
        "score": None,
        "latency_s": round(time.perf_counter() - start, 3),
        "started_at": started_at,
        "finished_at": utc_now(),
        "error": error,
        "attempt": attempt,
        "http_attempts": http_attempts,
        "response_id": None,
    }
    if reply is not None:
        choice = reply["choices"][0]
        message = choice.get("message") or {}
        usage = reply.get("usage") or {}
        content = message.get("content") or ""
        record.update(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            finish_reason=choice.get("finish_reason"),
            content=content,
            reasoning_content=message.get("reasoning_content") or "",
            # Only the answer is graded; a reply cut at max_tokens is scored as it stands
            score=score_row(task, content, row["outputs"]),
            response_id=reply.get("id"),
        )
    return record


def plan(args):
    """Return [(length, [(task, rows to send, total rows, {index: attempts})])] in run order."""
    lengths = sorted(set(args.lengths)) if args.order == "cheap-first" else list(dict.fromkeys(args.lengths))
    tasks = [task for task in TASKS if task in args.tasks]
    out = []
    for length in lengths:
        cells = []
        for task in tasks:
            rows = load_rows(args.data_root, task, length, args.rows)
            attempts, done = resume_state(load_records(out_path(args.out, task, length)))
            cells.append((task, [row for row in rows if row["index"] not in done], len(rows), attempts))
        out.append((length, cells))
    return out


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_hashes(args):
    """Return each cell's file sha256, its manifest's sha256 when the manifest exists, and its prompt-set hash."""
    out = {}
    for length in args.lengths:
        for task in args.tasks:
            path = data_path(args.data_root, task, length)
            rows = load_rows(args.data_root, task, length, args.rows)
            entry = {"sha256": sha256_file(path), "prompt_set_hash": prompt_set_hash(rows)}
            manifest = os.path.join(args.data_root, "manifest", f"ruler_{task}_{length}.jsonl.json")
            if os.path.exists(manifest):
                with open(manifest) as source:
                    entries = json.load(source)
                entry["manifest_sha256"] = entries.get(f"ruler/{task}_{length}.jsonl", {}).get("sha256")
                if entry["manifest_sha256"] != entry["sha256"]:
                    print(f"warning: {path} does not match its manifest sha256", file=sys.stderr)
            out[f"{task}_{length}"] = entry
    return out


def check_data(args, hashes):
    """Refuse to resume a cell's records unless its data file is the one every earlier invocation sent.

    Rows are matched to records by index alone, so a regenerated or edited file would silently keep the old
    scores; the file's sha256 (prompts and gold answers) is held to what meta.json recorded for that cell. A
    cell with no records yet (a new length or task) needs no history; raising --rows keeps the same file.
    """
    recorded = {}
    meta = os.path.join(args.out, "meta.json")
    if os.path.exists(meta):
        with open(meta) as source:
            for invocation in json.load(source).get("invocations", []):
                for cell, entry in invocation.get("data", {}).items():
                    recorded.setdefault(cell, set()).add(entry["sha256"])
    for length in args.lengths:
        for task in args.tasks:
            results = out_path(args.out, task, length)
            if not os.path.exists(results) or not os.path.getsize(results):
                continue
            cell, found = f"{task}_{length}", hashes[f"{task}_{length}"]["sha256"]
            known = recorded.get(cell, set())
            if known != {found}:
                raise SystemExit(
                    f"{results}: its records were sent from data with sha256 {sorted(known) or 'unrecorded'}, "
                    f"but {data_path(args.data_root, task, length)} now has {found} - "
                    "use another --out for other data"
                )


def git_describe():
    result = subprocess.run(
        ["git", "-C", HERE, "describe", "--always", "--dirty"], capture_output=True, text=True
    )
    return result.stdout.strip() or None


def server_models(base_url):
    """Return the server's /v1/models reply, or the reason there is none (QW serves no such route today)."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=10) as response:
            return json.loads(response.read())
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def write_meta(args, hashes):
    """Write meta.json, appending this invocation to an existing one after checking it is the same configuration."""
    path = os.path.join(args.out, "meta.json")
    invocation = {
        "started_at": utc_now(),
        "note": args.note,
        "base_url": args.base_url,
        "data_root": os.path.abspath(args.data_root),
        "rows": args.rows,
        "lengths": args.lengths,
        "tasks": args.tasks,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "harness": git_describe(),
        "server_models": server_models(args.base_url),
        "data": hashes,
    }
    meta: dict[str, Any] = {"label": args.label, "model": args.model, "max_tokens": args.max_tokens}
    if os.path.exists(path):
        with open(path) as source:
            prior = json.load(source)
        changed = [key for key in PROTOCOL_KEYS if prior.get(key) != meta[key]]
        if changed:
            raise SystemExit(
                f"{path}: {', '.join(f'{k}={prior.get(k)!r}' for k in changed)} differ from this run's - "
                "use another --out for another configuration"
            )
        meta = prior
    sampling = request_body({"input": "", "gen_prefix": ""}, args.model, args.max_tokens)
    del sampling["messages"]
    meta.setdefault("request", sampling)
    meta.setdefault("invocations", []).append(invocation)
    with open(path + ".tmp", "w") as sink:
        json.dump(meta, sink, indent=1)
    os.replace(path + ".tmp", path)


def truncated(text, keep=200):
    if len(text) <= 2 * keep:
        return text
    return f"{text[:keep]} ...[{len(text) - 2 * keep} chars]... {text[-keep:]}"


def dry_run(args, schedule):
    for length, cells in schedule:
        todo = sum(len(rows) for _, rows, _, _ in cells)
        print(f"length {length}: {todo} rows to send")
        for task, rows, total, attempts in cells:
            reruns = sum(1 for row in rows if row["index"] in attempts)
            print(f"  {task}: {len(rows)}/{total} to send ({reruns} errored before)")
    first = next((rows[0] for _, cells in schedule for _, rows, _, _ in cells if rows), None)
    if first is None:
        print("nothing to send")
        return
    body = request_body(first, args.model, args.max_tokens)
    body["messages"][0]["content"] = truncated(body["messages"][0]["content"])
    print(f"POST {args.base_url.rstrip('/')}/v1/chat/completions")
    print(json.dumps(body, indent=1, ensure_ascii=False))


def append_record(path, record, lock):
    """Append one record durably; the lock keeps concurrent rows' lines whole.

    A line torn by a killed run is closed first, so the new record starts a line of its own.
    """
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    with lock, open(path, "a+b") as sink:
        if sink.seek(0, os.SEEK_END):
            sink.seek(-1, os.SEEK_END)
            if sink.read(1) != b"\n":
                line = b"\n" + line
        sink.write(line)
        sink.flush()
        os.fsync(sink.fileno())


def run_length(url, length, cells, args, lock):
    """Send every pending row of one length; return how many ended with an error.

    Each worker appends its own record as its reply lands, so after a Ctrl-C the rows in flight still land
    (the pool waits for them; the queued ones are cancelled).
    """
    jobs = [
        (task, row, total, attempts.get(row["index"], 0) + 1)
        for task, rows, total, attempts in cells
        for row in rows
    ]
    position = {task: total - len(rows) for task, rows, total, _ in cells}
    latencies, errors = [], 0

    def one(job):
        task, row, _, attempt = job
        record = run_row(url, task, length, row, args, attempt)
        append_record(out_path(args.out, task, length), record, lock)
        return job, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        try:
            for finished, future in enumerate(concurrent.futures.as_completed(futures), 1):
                (task, _, total, _), record = future.result()
                position[task] += 1
                latencies.append(record["latency_s"])
                errors += record["error"] is not None
                eta = statistics.mean(latencies) * (len(jobs) - finished) / args.concurrency
                status = f"error={record['error'][:120]}" if record["error"] else (
                    f"score={record['score']:.3g} prompt={record['prompt_tokens']} comp={record['completion_tokens']}"
                    + (f" finish={record['finish_reason']}" if record["finish_reason"] != "stop" else "")
                )
                print(
                    f"[{length} {task} {position[task]}/{total}] {status} {record['latency_s']:.1f}s "
                    f"eta={datetime.timedelta(seconds=round(eta))} ({finished}/{len(jobs)} of {length})",
                    file=sys.stderr,
                    flush=True,
                )
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return errors


def main(argv=None):
    args = parse_args(argv)
    if args.concurrency < 1 or args.rows < 1:
        raise SystemExit("--concurrency and --rows must be at least 1")
    hashes = data_hashes(args)
    check_data(args, hashes)
    schedule = plan(args)
    if args.dry_run:
        dry_run(args, schedule)
        return 0
    os.makedirs(args.out, exist_ok=True)
    write_meta(args, hashes)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    lock, errors = threading.Lock(), 0
    try:
        for length, cells in schedule:
            errors += run_length(url, length, cells, args, lock)
    except KeyboardInterrupt:
        print("stopped; run again with the same --out to continue", file=sys.stderr)
        return 130
    if errors:
        print(f"{errors} rows ended with an error; run again with the same --out to retry them", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
