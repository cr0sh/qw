"""RULER prompts and scoring exactly as upstream latent-sparse-attention builds and grades them (stdlib only).

Upstream is Prefrontal-AI/latent-sparse-attention @ 5ebea732. The prompt is lm-eval's: the row's `input`, then
its `gen_prefix`, joined by the task's target delimiter " " only where neither side already has whitespace at
the seam (lm-eval's `maybe_delimit`; tasks.py:217 sets the delimiter). A plain `input + " " + gen_prefix`
differs on the qa_squad / qa_hotpot rows whose question ends in a space, and then misses 4 of the 30 pinned
prompt-set hashes in `hashes_qwen38_27b.json`; this join matches all 30.

The score is upstream's `grading/ruler.py`: lm-eval's `postprocess_pred` on the reply, then
`string_match_part` (any reference found counts) for qa_squad and qa_hotpot and `string_match_all` (the
fraction of references found) for the other 11 tasks, both case-insensitive substring matches.
"""
import hashlib
import re

TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
    "qa_squad",
    "qa_hotpot",
)
# grading/ruler.py:7 - the QA tasks carry several gold answers and any one counts
ANY_REFERENCE = ("qa_squad", "qa_hotpot")
CONTROL = re.compile(r"[\x00-\x1f]")


def prompt_text(row):
    """Return the user message of a row: lm-eval's `maybe_delimit(input, gen_prefix, " ")`."""
    question, prefix = row["input"], row["gen_prefix"]
    if not question or not prefix or question[-1].isspace() or prefix[0].isspace():
        return question + prefix
    return question + " " + prefix


def prompt_set_hash(rows):
    """Return upstream's prompt-set hash (tasks.py:534-568): sha256 over the rows' 16-hex prompt hashes."""
    joined = "".join(hashlib.sha256(prompt_text(row).encode()).hexdigest()[:16] for row in rows)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def postprocess_pred(reply):
    """lm-eval's RULER `postprocess_pred` for one reply: strip, control characters to newlines, strip."""
    return CONTROL.sub("\n", reply.strip()).strip()


def string_match_all(pred, refs):
    """Official RULER: the fraction of references found in the answer (grading/ruler.py:9)."""
    return sum(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs) / len(refs)


def string_match_part(pred, refs):
    """Official RULER: 1.0 when any reference is found in the answer (grading/ruler.py:14)."""
    return max(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs)


def metric_for_task(task):
    """Return the match a task is graded by."""
    if task not in TASKS:
        raise KeyError(f"unknown RULER task {task!r}")
    return string_match_part if task in ANY_REFERENCE else string_match_all


def score_row(task, content, outputs):
    """Score one reply (`choices[0].message.content`) against a row's gold `outputs`."""
    return metric_for_task(task)(postprocess_pred(content), outputs)
