"""Run the τ³ banking_knowledge benchmark with the repo-owned adapter.

This launcher requires ``--run``. Without it, configuration is validated and no
dataset, server, or model request is touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the 97-task campaign")
    parser.add_argument("--capture", type=Path, help="opt-in JSONL request/response/error capture path")
    parser.add_argument("--limit", type=int, help="optional bounded diagnostic task count")
    parser.add_argument("--resume", type=Path, help="reuse verified completed records from a terminal full 97-task run")
    return parser.parse_args()


def _validate_resume(config, run_dir: Path) -> tuple[int, int]:
    from evalscope.api.evaluator.cache import CacheManager
    from evalscope.api.registry import get_benchmark
    from evalscope.config import load_task_config_snapshot
    from evalscope.evaluation_versioning import (
        ResolvedBenchmarkSpec, build_evaluation_identity, validate_cached_evaluation_identity,
    )
    from evalscope.utils.io_utils import OutputsStructure
    from tau3_adapter import _validate_cached_records

    if not all((run_dir / name).is_dir() for name in ("configs", "predictions", "reviews")):
        raise ValueError("resume requires the exact inner timestamp run directory containing configs, predictions, and reviews")
    progress_path = run_dir / "progress.json"
    if not progress_path.is_file():
        raise ValueError("resume cannot verify a terminal run without its progress.json")
    progress = json.loads(progress_path.read_text())
    if progress.get("status") not in {"error", "completed"} or progress.get("total_count") != 97:
        raise ValueError("resume requires a terminal full 97-task run; active, stale-running, and diagnostic runs are rejected")
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
    datasets = benchmark.load_dataset()
    if set(datasets.keys()) != {"banking_knowledge"} or len(datasets["banking_knowledge"]) != 97:
        raise ValueError("resume requires the unchanged complete 97-task banking dataset")
    dataset = datasets["banking_knowledge"]
    if len({sample.metadata["id"] for sample in dataset}) != 97:
        raise ValueError("resume dataset has duplicate task identities")
    cache = CacheManager(outputs, config.model_id, "tau3_bench")
    return _validate_cached_records(cache, dataset, config.model_id)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.resume is not None and args.limit is not None:
        raise SystemExit("--resume is only available for full 97-task runs; do not combine it with --limit")

    endpoint = "http://127.0.0.1:8883/v1"
    model = "qwen3.8-27b"
    generation = {
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 32768,
        "reasoning_effort": "medium",
        "reasoning_history": "reasoning_field",
        "timeout": 7200,
        # Keep the OpenAI client's transport retries, but avoid EvalScope's
        # duplicate outer loop (which retried typed model-output errors).
        "retries": 1,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
    }

    simulator_model = "openai-codex/gpt-5.6-luna"
    simulator_endpoint = os.environ.get("TAU_SIMULATOR_API_URL", "http://127.0.0.1:18766/v1")
    if simulator_endpoint.rstrip("/") == endpoint.rstrip("/"):
        raise SystemExit("simulator endpoint must be distinct from the QW endpoint")
    token_file = Path(os.environ.get("TAU_SIMULATOR_TOKEN_FILE", "~/.omp/auth-gateway.token")).expanduser()
    simulator_key = os.environ.get("TAU_SIMULATOR_API_KEY") or token_file.read_text(encoding="utf-8").strip()
    if not simulator_key:
        raise SystemExit("simulator gateway credential must not be empty")
    os.environ["EVALSCOPE_API_KEY"] = simulator_key

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

    from tau3_adapter import SIMULATOR_EMPTY_RESPONSE_RETRIES, SIMULATOR_STEP_PROTOCOL, configure_capture, install

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
                    "simulator_empty_response_retries": SIMULATOR_EMPTY_RESPONSE_RETRIES,
                    "simulator_step_protocol": SIMULATOR_STEP_PROTOCOL,
                    "api_base": simulator_endpoint,
                    "api_key": None,
                    "generation_config": {
                        "temperature": 0.0,
                        "max_tokens": 32768,
                        "reasoning_effort": "low",
                        "reasoning_history": "reasoning_field",
                        "timeout": 7200,
                        "retries": 1,
                    },
                    "retrieval_config": "bm25",
                    "retrieval_config_kwargs": {},
                },
            }
        },
        dataset_dir=str(cache_dir / "evalscope"),
        eval_batch_size=1,
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
    print("agent=qwen3.8-27b temperature=0 reasoning_effort=medium max_tokens=32768", flush=True)
    print(f"simulator_and_nl_judge={simulator_model} endpoint={simulator_endpoint}", flush=True)
    print(f"simulator_empty_response_retries={SIMULATOR_EMPTY_RESPONSE_RETRIES} (user simulator clean-stop empty responses only)", flush=True)
    print("simulator_step_protocol=explicit nonempty invocation clarification (simulator-only prompt deviation)", flush=True)
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
    with (lock_dir / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another banking evaluator owns this campaign") from error
        if args.resume is not None:
            predictions, reviews = _validate_resume(config, work_dir)
            print(f"Verified native resume: {predictions}/97 completed predictions, {reviews}/97 completed reviews.", flush=True)
        configure_capture(capture_path)
        run_task(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
