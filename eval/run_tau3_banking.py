"""Configure upstream EvalScope's native τ³ banking_knowledge evaluation.

Without --run, construct configuration only: no credentials, dataset, or API calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the native evaluation")
    parser.add_argument("--limit", type=int, help="bounded diagnostic task count")
    parser.add_argument("--eval-batch-size", type=int, default=3, help="native task concurrency (default: 3)")
    parser.add_argument("--resume", type=Path, help="native use_cache run directory (no custom recovery)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    return args


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
    os.environ["OPENAI_API_KEY"] = key.strip()
    os.environ["EVALSCOPE_API_KEY"] = key.strip()


def main() -> int:
    args = parse_args()
    setup_dir = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
    work_dir = args.resume.expanduser().resolve() if args.resume is not None else setup_dir / "outputs" / f"tau3-banking-native-{run_id}"
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
    if Path(dataset_id).is_dir():
        os.environ["TAU2_DATA_DIR"] = dataset_id

    from evalscope import TaskConfig, run_task

    config = TaskConfig(
        model="qwen3.8-27b",
        api_url="http://127.0.0.1:8883/v1",
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=["tau3_bench"],
        dataset_args={
            "tau3_bench": {
                "dataset_id": dataset_id,
                "subset_list": ["banking_knowledge"],
                "extra_params": {
                    "user_model": "deepseek-v4-pro",
                    "api_base": "https://api.deepseek.com",
                    "api_key": None,
                    "generation_config": {
                        "temperature": 0.0,
                        "max_tokens": 32768,
                        "extra_body": {"thinking": {"type": "disabled"}},
                        "timeout": 7200,
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
        generation_config={
            "max_tokens": 32768,
            "reasoning_effort": "medium",
            "reasoning_history": "reasoning_field",
            "timeout": 7200,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        },
        use_cache=str(work_dir) if args.resume is not None else None,
        work_dir=str(work_dir),
    )

    print(f"evaluator=evalscope-{version('evalscope')} native tau3_bench subset=banking_knowledge repeats=1 retrieval=bm25", flush=True)
    print("tau2_source=local_submodule base=v1.0.1 (gitlink pins upstream cherry-pick and compatibility repair)", flush=True)
    print(f"eval_batch_size={args.eval_batch_size} limit={args.limit} native_resume={args.resume is not None}", flush=True)
    print("agent=qwen3.8-27b endpoint=http://127.0.0.1:8883/v1 reasoning_effort=medium thinking=enabled max_tokens=32768", flush=True)
    print("simulator_and_nl_judge=deepseek-v4-pro endpoint=https://api.deepseek.com thinking=disabled temperature=0", flush=True)
    print(f"work_dir={work_dir} dataset={dataset_id}", flush=True)
    if not args.run:
        print("Configuration only; no credentials loaded, dataset loaded, or model request made.", flush=True)
        return 0
    _load_simulator_credentials()
    run_task(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
