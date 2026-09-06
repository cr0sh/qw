"""Run the τ³ banking_knowledge benchmark with the repo-owned adapter.

This launcher requires ``--run``. Without it, configuration is validated and no
dataset, server, or model request is touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the 97-task campaign")
    parser.add_argument("--capture", type=Path, help="opt-in JSONL request/response/error capture path")
    parser.add_argument("--limit", type=int, help="optional bounded diagnostic task count")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

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
    work_dir = setup_dir / "outputs" / f"tau3-banking-{run_id}"
    cache_dir = work_dir / "data-cache"
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

    configure_capture(capture_path)
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
        use_cache=None,
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
    work_dir.mkdir(parents=True, exist_ok=False)
    run_task(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
