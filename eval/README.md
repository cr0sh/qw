# τ³ banking evaluation

[`run_tau3_banking.py`](run_tau3_banking.py) configures **unmodified EvalScope's
native τ³ adapter** through `TaskConfig` and `run_task`. It adds no generation,
message-conversion, retry, scoring, termination, capture, or checkpoint hooks.
EvalScope's own Tau2 generation bridge is part of the upstream implementation.

All commands below run from the repository/worktree root. They are reproduction
instructions, not commands to run alongside an already-active campaign.

## Prerequisites

- Python 3.12 or 3.13; the commands below select 3.12.
- `uv` and the locked dependencies in [`pyproject.toml`](pyproject.toml) and
  [`uv.lock`](uv.lock): EvalScope **1.11.1** and `tau2[knowledge]` installed from
  the local `eval/tau2-bench` submodule. The fork starts at upstream **v1.0.1**
  (`fc0055dc4e0a316c3f83133267fbd6faaa770992`), cherry-picks
  `07a202cc9ffb2d73911835dd58c55ea280855935`, then repairs reasoning conversion
  for native and EvalScope histories. The gitlink pins the complete fork.
- On macOS, PortAudio must be available to build the pinned PyAudio dependency.
  Upstream's text path eagerly imports voice dependencies; the lock includes
  those required dependencies.
- A tested QW binary and the Qwen3.8-27B model checkpoint.

Initialize the pinned fork using authenticated SSH:

```bash
git submodule sync -- eval/tau2-bench
git submodule update --init --checkout eval/tau2-bench
```

The fork is hosted at `git@github.com:cr0sh/tau2-bench.git`, with the reasoning
changes on `qwr/v1.0.1-pr389-reasoning`. The gitlink pins the tested commit;
initialization requires SSH access to GitHub.

Create the isolated reference environment without modifying an older environment:

```bash
(cd eval && UV_PROJECT_ENVIRONMENT=.venv-reference uv sync --locked --python 3.12 --no-editable)
```

Use this interpreter explicitly. EvalScope remains unmodified; only the pinned
Tau2 fork changes history conversion. After updating that fork, reinstall into
the same reference environment:

```bash
(cd eval && UV_PROJECT_ENVIRONMENT=.venv-reference uv sync --locked --python 3.12 --no-editable --reinstall-package tau2)
```

## Credentials

Store `DEEPSEEK_API_KEY` in the **primary worktree's** private `eval/.env.local`.
Create/edit that file privately and set its mode to `600`; it must not be a
symlink. A linked worktree's own `eval/.env.local` is not used.

Only an executing evaluator loads the key, exporting it within that process as
`OPENAI_API_KEY` and `EVALSCOPE_API_KEY`. The latter is EvalScope's native API-key
fallback. The target's configuration uses `api_key="EMPTY"`; the simulator uses
`api_key=None` and the environment fallback. Never put a real key in commands,
TaskConfig, committed files, or shared artifacts.

## Start the model server separately

The launcher does not start, build, stop, or reconfigure QW. For a new campaign,
use a fresh prefix-cache directory and record the tested binary's commit/hash.
See [shared worktree and GPU locking rules](../DEVELOPMENT.md#shared-worktrees)
before building or running GPU work.

The Metal runtime defaults to an 8 GiB reusable allocator-buffer allowance,
independent of the prefix-cache budgets and not a total-process memory cap.
Leave `MLXCEL_CACHE_LIMIT` unset to reproduce that default, or explicitly record
an override in raw decimal bytes (`0` disables free-buffer caching). Tighter
allowances may trade decode throughput for lower retention; see the
[runtime policy notes](../DEVELOPMENT.md#generation-policy).

```bash
export QW_BIN="$PWD/target/release/qw"  # or the tested frozen binary
export QW_MODEL="$HOME/.cache/qw/models/Jundot/Qwen3.8-27B-oQ4e-fp16-mtp"
export QW_PREFIX_CACHE="$PWD/outputs/tau3-prefix-$(date -u +%Y%m%dT%H%M%SZ)"

./gpu-lock -- env -u QW_DFLASH2_VERIFY_WIDTH "$QW_BIN" serve \
  --bind 127.0.0.1:8883 \
  --model-id qwen3.8-27b \
  --model "$QW_MODEL" \
  --decoder dflash \
  --prefix-cache-directory "$QW_PREFIX_CACHE"
```

Wait for the server to listen before launching evaluation in another terminal.
Do not start a second server on the running campaign's port or bypass `gpu-lock`.

## Dataset

By default, the native loader downloads `evalscope/tau3-bench-data` from
ModelScope into the invocation's data cache. It downloads the full dataset
repository, not only the selected banking files.

To avoid downloading it again, point the launcher at an intact, verified local
snapshot root containing `tau2/`:

```bash
export TAU3_DATASET_ID=/absolute/path/to/snapshots/master
```

Use the same frozen snapshot for a campaign and its resumes. Record its source
and file hashes when sharing results. Reusing dataset assets is distinct from
reusing predictions or a warmed model prefix cache.
If purging a previous run that contains the snapshot, first preserve the complete
dataset snapshot outside that run's output/cache tree and verify its hashes.
Set `TAU3_DATASET_ID` to the preserved root, not to its `tau2/` child. Do not copy
old predictions, reviews, task configurations, or model prefix-cache files.

## Run

Construct configuration without loading credentials, datasets, or invoking models:

```bash
eval/.venv-reference/bin/python -I -B eval/run_tau3_banking.py
```

Run one diagnostic episode, then launch a fresh full campaign with concurrency 2:

```bash
eval/.venv-reference/bin/python -I -B eval/run_tau3_banking.py --run --limit 1 --eval-batch-size 2
eval/.venv-reference/bin/python -I -B eval/run_tau3_banking.py --run --eval-batch-size 2
```

The full banking dataset contains **97 tasks**. The launcher uses one repeat,
seed 42, and BM25 retrieval. `--eval-batch-size` defaults to 3 and controls
EvalScope's native task worker pool—not GPU batching or token-level interleaving.
The explicit flag above selects two workers while preserving that general default.
In EvalScope 1.11.1, `TaskConfig` initialization also synchronizes
`generation_config.batch_size` to `eval_batch_size`; the native evaluator passes
`eval_batch_size` as the task pool's `max_workers`. For this fresh campaign both
values must be 2 in the saved configuration. Use the launcher flag rather than
loading an old task snapshot or manually overriding either value after construction.
This limits concurrent episodes to two, not a guarantee of two simultaneous model
requests: episodes alternate simulator, tool, agent, and judge work, and QW serves
queued requests at request boundaries.

Each fresh invocation creates a unique directory under
`eval/outputs/tau3-banking-native-<UTC timestamp>-<uuid>/`; EvalScope adds an inner
timestamp directory containing configurations, predictions, reviews, logs, and
reports. The launcher prints `work_dir`, and EvalScope logs the final output
directory. Do not reuse smoke or historical custom-adapter results for a fresh
reference campaign.

## Evaluation policy

| Role | Configuration |
| --- | --- |
| Agent under test | `qwen3.8-27b`, `http://127.0.0.1:8883/v1`, thinking enabled, medium reasoning effort, 32768 output tokens |
| User simulator and native NL-assertion judge | `deepseek-v4.1-flash`, `https://api.deepseek.com`, thinking disabled, temperature 0, 32768 output tokens |

The seven target sampling overrides are omitted so QW resolves its model policy.
Native retry, termination, scoring, and error handling are unchanged. The
upstream adapter converts caught task-execution exceptions to **reward-zero
results** rather than aborting the entire campaign. Include those outcomes in
native scores and report recorded errors; do not selectively retry model failures.
The upstream catch also includes some infrastructure failures—there is no local
scoring policy that separates them.

EvalScope's native bridge strips reasoning from participant-visible messages.
The Tau2 fork recovers the agent's private reasoning from its stored model output
and restores it to subsequent model requests. The launcher explicitly selects
`reasoning_history="reasoning_field"`; Qwen3.8's template defaults to preserving
that `reasoning_content`. Simulator reasoning is not forwarded to the agent.
This is unmodified EvalScope with a patched Tau2 dependency, not stock Tau2.
DeepSeek simulation/judging also differs from other leaderboard configurations.

Before starting a full campaign, require a successful native smoke and verify
that subsequent SDK requests contain the earlier reasoning traces unchanged.
Verify positive cache reuse in server `generation.complete` events; for an
unchanged full prefix, the next request's `cached_tokens` equals the previous
request's `prompt_tokens + completion_tokens`. Start the full campaign with a
different, empty prefix-cache directory and a fresh output directory.

Earlier repository-owned-adapter campaigns are diagnostic artifacts, not
upstream-reference results. The removed `--capture` flag and `TAU3_CAPTURE_PATH`
are not part of this workflow. Use native artifacts and server logs for diagnosis.

## Resume and supervision

After the previous evaluator has stopped, use its **inner timestamp directory**:

```bash
eval/.venv-reference/bin/python -I -B eval/run_tau3_banking.py \
  --run --eval-batch-size 2 \
  --resume /absolute/path/to/tau3-banking-native-UTC-UUID/INNER_TIMESTAMP
```

Keep the dataset setting and evaluation configuration unchanged. `--resume` maps
directly to native `use_cache`; upstream determines which records can be reused.
Never run multiple evaluators against one output directory.
The example resumes a concurrency-2 `deepseek-v4.1-flash` campaign only; the model
change requires a fresh campaign, not resuming results from a different simulator.

The launcher adds no writer lock, record salvage, stale-state repair, synchronized
writes, or power-loss durability guarantee. Native resume may reject an
interrupted/incompatible run; unfinished episodes may execute again. Do not
confuse this with retrying a completed reward-zero episode for a better answer.

Supervise long runs externally. The managed reference campaign uses persistent,
detached model/evaluator services so exiting the assistant harness does not stop
them. Process persistence does not guarantee checkpoint recovery after an OS or
storage failure.

## Launcher and reasoning checks

These checks do not call either model or modify the running campaign:

```bash
eval/.venv-reference/bin/python -B -m unittest eval.test_run_tau3_banking
eval/.venv-reference/bin/python -B -m unittest discover -s eval/tau2-bench/tests -p test_reasoning_history.py
```
