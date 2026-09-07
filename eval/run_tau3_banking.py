"""Run the τ³ banking_knowledge benchmark with the repo-owned adapter.

This launcher requires ``--run``. Without it, configuration is validated and no
dataset, server, or model request is touched.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the 97-task campaign")
    parser.add_argument("--capture", type=Path, help="opt-in JSONL request/response/error capture path")
    parser.add_argument("--limit", type=int, help="optional bounded diagnostic task count")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="EvalScope task concurrency (default: 1)")
    parser.add_argument("--resume", type=Path, help="recover validated native records under the campaign writer lock")
    return parser.parse_args()


def _load_simulator_credentials() -> None:
    """Load the primary worktree's private credential only in the evaluator process."""
    from dotenv import dotenv_values

    common_dir = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    credential_file = Path(common_dir).parent / "eval" / ".env.local"
    if not credential_file.is_file() or credential_file.is_symlink():
        raise SystemExit("primary eval/.env.local must provide the direct simulator credential")
    if credential_file.stat().st_mode & 0o077:
        raise SystemExit("primary eval/.env.local must be private (chmod 600)")
    key = dotenv_values(credential_file, interpolate=False).get("DEEPSEEK_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise SystemExit("primary eval/.env.local must define DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = key.strip()


def _validate_banking_dataset(datasets, expected_count: int):
    if set(datasets.keys()) != {"banking_knowledge"}:
        raise ValueError("evaluation requires only the banking_knowledge dataset")
    dataset = datasets["banking_knowledge"]
    if len(dataset) != expected_count:
        raise ValueError(f"expected {expected_count} banking tasks, loaded {len(dataset)}")
    if len({sample.metadata["id"] for sample in dataset}) != expected_count:
        raise ValueError("banking dataset has duplicate task identities")
    return dataset


@contextmanager
def _campaign_writer(lock_dir: Path):
    path = lock_dir / ".writer.lock"
    if path.is_symlink():
        raise ValueError("campaign writer lock must not be a symlink")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another banking evaluator owns this campaign") from error
        yield lock


def _require_campaign_writer(run_dir: Path, lock) -> None:
    path = run_dir.parent / ".writer.lock"
    if path.is_symlink():
        raise ValueError("campaign writer lock must not be a symlink")
    expected = path.stat()
    actual = os.fstat(lock.fileno())
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("resume requires the exact campaign writer lock")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another banking evaluator owns this campaign") from error


def _validate_resume(config, run_dir: Path, writer_lock) -> tuple[int, int]:
    _require_campaign_writer(run_dir, writer_lock)
    from evalscope.api.evaluator.cache import CacheManager
    from evalscope.api.registry import get_benchmark
    from evalscope.config import load_task_config_snapshot
    from evalscope.evaluation_versioning import (
        ResolvedBenchmarkSpec, build_evaluation_identity, validate_cached_evaluation_identity,
    )
    from evalscope.utils.io_utils import OutputsStructure
    if __package__:
        from .tau3_adapter import _recover_cached_records
    else:
        from tau3_adapter import _recover_cached_records

    if (run_dir / "INVALID.json").exists():
        raise ValueError("invalidated benchmark runs cannot be resumed or reused")
    if not (run_dir / "configs").is_dir() or (run_dir / "configs").is_symlink():
        raise ValueError("resume requires the exact inner timestamp run directory with its configuration snapshot")
    # Native OutputsStructure creates these lazily. Death before the first
    # review (or prediction) must not invalidate already durable records.
    for name in ("predictions", "reviews"):
        path = run_dir / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"native {name} path must be a directory when present")
    progress_path = run_dir / "progress.json"
    if not progress_path.is_file() or progress_path.is_symlink():
        raise ValueError("resume cannot verify a run without its native progress.json")
    progress = json.loads(progress_path.read_text())
    if progress.get("status") not in {"running", "error", "completed"} or progress.get("total_count") != 97:
        raise ValueError("resume requires a full 97-task native run")
    benchmark = get_benchmark("tau3_bench", config)
    meta = benchmark.benchmark_meta
    identity = build_evaluation_identity(
        {"tau3_bench": ResolvedBenchmarkSpec.from_meta(meta, config)},
        {"tau3_bench": meta.evaluation_version},
        config,
    )
    outputs = OutputsStructure(str(run_dir), is_make=False)
    validate_cached_evaluation_identity(
        load_task_config_snapshot(str(Path(outputs.configs_dir) / "task_config.yaml")),
        identity,
        False,
    )
    dataset = _validate_banking_dataset(benchmark.load_dataset(), 97)
    cache = CacheManager(outputs, config.model_id, "tau3_bench")
    return _recover_cached_records(cache, dataset, config.model_id, run_dir)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.eval_batch_size < 1:
        raise SystemExit("--eval-batch-size must be positive")
    if args.resume is not None and args.limit is not None:
        raise SystemExit("--resume is only available for full 97-task runs; do not combine it with --limit")

    endpoint = "http://127.0.0.1:8883/v1"
    model = "qwen3.8-27b"
    generation = {
        # Omit sampling controls so the runtime resolves its thinking policy.
        "max_tokens": 32768,
        "reasoning_effort": "medium",
        "reasoning_history": "reasoning_field",
        "timeout": 7200,
        # Keep the OpenAI client's transport retries, but avoid EvalScope's
        # duplicate outer loop (which retried typed model-output errors).
        "retries": 1,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }

    simulator_model = "deepseek-v4-pro"
    simulator_endpoint = "https://api.deepseek.com"

    setup_dir = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
    work_dir = args.resume.expanduser().resolve() if args.resume is not None else setup_dir / "outputs" / f"tau3-banking-{run_id}"
    cache_dir = (work_dir.parent if args.resume is not None else work_dir) / "data-cache"
    for name, path in {
        "EVALSCOPE_CACHE": cache_dir / "evalscope",
        "HF_HOME": cache_dir / "huggingface",
        "HF_DATASETS_CACHE": cache_dir / "huggingface" / "datasets",
        "MODELSCOPE_CACHE": cache_dir / "modelscope",
        "XDG_CACHE_HOME": cache_dir / "xdg",
    }.items():
        os.environ[name] = str(path)

    dataset_id = os.environ.get("TAU3_DATASET_ID", "evalscope/tau3-bench-data")
    if dataset_id.startswith(("http://", "https://")):
        raise SystemExit("TAU3_DATASET_ID must be a local path or modelscope dataset id, not a URL")
    capture_path = args.capture or (
        Path(os.environ["TAU3_CAPTURE_PATH"]) if os.environ.get("TAU3_CAPTURE_PATH") else None
    )
    if Path(dataset_id).is_dir():
        os.environ["TAU2_DATA_DIR"] = dataset_id

    if __package__:
        from .tau3_adapter import configure_capture, install
    else:
        from tau3_adapter import configure_capture, install

    install()
    from evalscope import TaskConfig, run_task

    config = TaskConfig(
        model=model,
        model_args={"max_retries": 5},
        api_url=endpoint,
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=["tau3_bench"],
        dataset_args={
            "tau3_bench": {
                "dataset_id": dataset_id,
                "subset_list": ["banking_knowledge"],
                "extra_params": {
                    "user_model": simulator_model,
                    "api_base": simulator_endpoint,
                    "api_key": None,
                    "generation_config": {
                        "temperature": 0.0,
                        "max_tokens": 32768,
                        "extra_body": {"thinking": {"type": "disabled"}},
                        "timeout": 7200,
                        "retries": 1,
                    },
                    "retrieval_config": "bm25",
                    "retrieval_config_kwargs": {},
                },
            }
        },
        dataset_dir=str(cache_dir / "evalscope"),
        eval_batch_size=args.eval_batch_size,
        limit=args.limit,
        repeats=1,
        seed=42,
        generation_config=generation,
        use_cache=str(work_dir) if args.resume is not None else None,
        work_dir=str(work_dir),
        no_timestamp=False,
        enable_progress_tracker=True,
        # Preserve EvalScope's exception propagation; failures must not silently
        # alter the benchmark denominator or become reward zeroes.
        ignore_errors=False,
    )

    print("dataset=tau3_bench subset=banking_knowledge tasks=97 repeats=1 retrieval=bm25", flush=True)
    print(f"eval_batch_size={args.eval_batch_size} scheduling=independent_tasks gpu_batching=false", flush=True)
    print("agent=qwen3.8-27b sampling=runtime_policy reasoning_effort=medium thinking=enabled max_tokens=32768", flush=True)
    print(f"simulator_and_nl_judge={simulator_model} endpoint={simulator_endpoint}", flush=True)
    print("simulator_thinking=disabled canonical_prompts=true empty_response_retries=0", flush=True)
    print(f"work_dir={work_dir} capture={capture_path or 'disabled'} dataset={dataset_id}", flush=True)
    if not args.run:
        print("Configuration validated only; no dataset loaded, server started, or model request made.", flush=True)
        return 0
    if args.resume is None:
        work_dir.mkdir(parents=True, exist_ok=False)
    elif not work_dir.is_dir():
        raise SystemExit("resume requires the existing inner timestamp run directory")
    # Fresh runs use the outer UUID directory; native use_cache points to its
    # inner timestamp directory. Both writers hold the same stable empty lock.
    lock_dir = work_dir.parent if args.resume is not None else work_dir
    with _campaign_writer(lock_dir) as lock:
        if args.resume is not None:
            predictions, reviews = _validate_resume(config, work_dir, lock)
            print(f"Verified durable native resume: {predictions}/97 predictions, {reviews}/97 reviews; {97 - predictions} unfinished episodes may run again.", flush=True)
        else:
            from evalscope.api.registry import get_benchmark

            dataset = _validate_banking_dataset(
                get_benchmark("tau3_bench", config).load_dataset(), args.limit or 97,
            )
            print(f"Verified fresh banking dataset: {len(dataset)} unique tasks.", flush=True)
        configure_capture(capture_path)
        _load_simulator_credentials()
        run_task(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
