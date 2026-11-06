#!/usr/bin/env python3
"""Generate upstream's RULER files on macOS, and check them against upstream's pinned prompt-set hashes.

usage: prepare_ruler.py --lsa-clone LSA --root DATA [--lengths 4096 32768] [--tasks ...] [--samples 100]
       prepare_ruler.py --root DATA --check-hashes LSA/src/lsa/evaluation/accuracy/data/hashes_qwen38_27b.json

Generation drives upstream's own `lsa.evaluation.accuracy.data.ruler` (latent-sparse-attention @ 5ebea732),
imported from the clone, so it runs in the pinned venv the README describes, not with the standard library
alone. Upstream's `lsa data prepare ruler` does not run on macOS, for two reasons this driver works around
without editing upstream's files:
  - `os.sched_getaffinity` does not exist on macOS (ruler.py:270,381): it is supplied as every CPU;
  - `publish()` moves the verified prerequisites with GNU `mv --no-clobber --no-target-directory`
    (ruler.py:559), which BSD mv lacks: it is replaced by a rename that never overwrites.
The submitit job upstream sends the grid through is skipped: each cell is `ruler.prepare_cell`, as the job runs
it. Prerequisites (NVIDIA/RULER at the card's commit with upstream's patch, the haystack sources, NLTK punkt)
come from upstream's `ensure()`, which verifies every sha256 before anything is generated.

`--check-hashes` needs no upstream import: it recomputes upstream's prompt-set hash (tasks.py:534-568) of the
first 4 / 100 / all rows of each file and compares it with the pinned one, exiting 1 on any difference.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ruler_score import TASKS, prompt_set_hash  # noqa: E402

PIN = re.compile(r"^ruler_(?P<cell>.+):(?:first(?P<first>\d+)|all)$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lsa-clone", help="latent-sparse-attention checkout (its src/ is imported)")
    parser.add_argument("--root", required=True, help="corpus root: ruler/ and manifest/ are written here")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27b")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS, metavar="TASK")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="tokenizer threads per cell")
    parser.add_argument("--force", action="store_true", help="regenerate cells whose file exists")
    parser.add_argument("--check-hashes", metavar="HASHES_JSON", help="only check prompt-set hashes")
    return parser.parse_args(argv)


def check_hashes(root, pins_path):
    """Print one line per pinned (cell, prefix) whose file exists; return how many differ."""
    with open(pins_path) as source:
        pins = json.load(source)
    differ = checked = 0
    for key, expected in sorted(pins.items()):
        match = PIN.match(key)
        if not match:
            continue
        path = os.path.join(root, "ruler", f"{match['cell']}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as source:
            rows = [json.loads(line) for line in source if line.strip()]
        n = int(match["first"]) if match["first"] else len(rows)
        if len(rows) < n:
            print(f"{key}: {len(rows)} rows, fewer than pinned - skipped")
            continue
        found = prompt_set_hash(rows[:n])
        checked += 1
        differ += found != expected
        print(f"{key} {found} {expected} {'MATCH' if found == expected else 'DIFF'}")
    print(f"{checked - differ}/{checked} pinned prompt sets match")
    return differ


def portable_publish(source, destination):
    """Upstream's publish() without GNU mv: rename into place, keeping a concurrent installation that won."""
    if os.path.lexists(destination):
        return
    os.rename(source, destination)


def generate(args):
    if not args.lsa_clone:
        raise SystemExit("--lsa-clone is required to generate")
    # Upstream reads its data root and derived paths (RULER_NVIDIA, NLTK_DATA) when the module is imported
    root = os.path.abspath(args.root)
    os.environ["LSA_EVAL_DATA"] = root
    sys.path.insert(0, os.path.join(os.path.abspath(args.lsa_clone), "src"))
    if not hasattr(os, "sched_getaffinity"):
        os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))
    from lsa.evaluation.accuracy.data import ruler

    ruler.publish = portable_publish
    print(f"RULER_NVIDIA {ruler.RULER_NVIDIA}  NLTK_DATA {ruler.NLTK_DATA}", flush=True)
    ruler.ensure()
    for length in args.lengths:
        for task in [t for t in TASKS if t in args.tasks]:
            path = ruler.ruler_path(task, length, root)
            if os.path.exists(path) and not args.force:
                print(f"{path}: exists, kept (--force regenerates)", flush=True)
                continue
            ruler.prepare_cell(task, length, args.samples, root, args.tokenizer, threads=args.threads)


def main(argv=None):
    args = parse_args(argv)
    if args.check_hashes:
        return 1 if check_hashes(args.root, args.check_hashes) else 0
    generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
