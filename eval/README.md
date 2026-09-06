# τ³ banking evaluation

`run_tau3_banking.py` runs the repository-owned adapter against the τ³
`banking_knowledge` benchmark. The launcher requires `--run` to execute
anything; without it, configuration is validated and no dataset, server, or
model request is touched.

## Prerequisites

Use a dedicated worktree and a Python 3.11-or-newer environment containing
EvalScope 1.11.0 and the τ² knowledge benchmark package. Start QW separately
before an integration run:

```bash
qw serve --bind 127.0.0.1:8883
```

The launcher does not start QW. It expects the agent endpoint at
`http://127.0.0.1:8883/v1` and uses model ID `qwen3.8-27b`.

Simulator and natural-language judge configuration is owned by
[`run_tau3_banking.py`](run_tau3_banking.py). Consult that source for the
current endpoint and credential setup rather than copying environment details
into this runbook.

## Commands

Validate configuration only, run a bounded diagnostic, or run the complete
97-task campaign:

```bash
python eval/run_tau3_banking.py
python eval/run_tau3_banking.py --limit 1 --run
python eval/run_tau3_banking.py --run
```

A fresh run uses one repeat, seed 42, and BM25 retrieval. Each invocation
creates a UUID-named result directory below `eval/outputs/`. The optional
`TAU3_DATASET_ID=/path/to/dataset` environment variable reuses an existing
local dataset read-only; otherwise the configured dataset is downloaded into
the run's cache.

Opt in to request, response, and result capture with either
`--capture /unique/path.jsonl` or `TAU3_CAPTURE_PATH`. Existing capture paths
are rejected. Capture is disabled by default.

## Output-contract policy

The server rejects malformed or undeclared generated tool calls and violations
of `parallel_tool_calls=false`. These output-contract failures return HTTP 422
with `error.type=model_output_error` and `error.code=invalid_model_output`,
rather than a retryable server error. Streaming responses use the same
error type and code in the endpoint's terminal error event after HTTP headers
have been sent. The error does not include raw generated argument bodies.

An output-contract failure aborts the banking task rather than being silently
skipped or converted to reward zero; an aborted run is not a complete
benchmark score. Treat the full Tau2 result as the authoritative trajectory.

The launcher is deliberately conservative about failures: completion errors and
retry exhaustion are surfaced to the caller. Review the captured request and
response artifacts for diagnosis rather than treating a failed task as a
successful result.

## Result layout

Fresh results are written below `eval/outputs/tau3-banking-<timestamp>-<uuid>/`.
Keep result artifacts and any read-only dataset snapshot intact when sharing a
diagnostic run.
