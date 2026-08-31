import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


PROJECT_DIR = Path(__file__).resolve().parent
MIN_FREE_BYTES = 15 * 1024**3
# The qwen27b server does not expose top_k on its Chat Completions protocol.
# The checkpoint generation_config.json supplies the official top_k=20 default.
SUITES = {
    "gpqa_diamond": {
        "generation_config": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32_768,
            "reasoning_effort": "xhigh",
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                }
            },
        },
        "repeats": 1,
    },
    "ifbench": {
        "generation_config": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32_768,
            "reasoning_effort": "xhigh",
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                }
            },
        },
        "repeats": 1,
    },
    "live_code_bench": {
        "generation_config": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32_768,
            "reasoning_effort": "xhigh",
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                }
            },
        },
        "repeats": 1,
    },
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def generation_config(dataset: str) -> dict[str, object]:
    return SUITES[dataset]["generation_config"].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded EvalScope benchmark against the local qw-server."
    )
    parser.add_argument("dataset", choices=SUITES)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs",
        help="root directory for timestamped EvalScope outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_root.expanduser().resolve() / args.dataset
    cache_dir = PROJECT_DIR / "cache"
    cache_environment = {
        "EVALSCOPE_CACHE": str(cache_dir / "evalscope"),
        "HF_DATASETS_CACHE": str(cache_dir / "huggingface" / "datasets"),
        "HF_HOME": str(cache_dir / "huggingface"),
        "MODELSCOPE_CACHE": str(cache_dir / "modelscope"),
        "NLTK_DATA": str(cache_dir / "nltk"),
        "UV_CACHE_DIR": str(cache_dir / "uv"),
        "XDG_CACHE_HOME": str(cache_dir / "xdg"),
    }

    config = generation_config(args.dataset)
    command = [
        "evalscope",
        "eval",
        "--model",
        "qwen3.8-27b",
        "--eval-type",
        "openai_api",
        "--api-url",
        "http://127.0.0.1:8883/v1",
        "--api-key",
        "EMPTY",
        "--datasets",
        args.dataset,
        "--generation-config",
        json.dumps(config, separators=(",", ":")),
        "--eval-batch-size",
        "1",
        "--timeout",
        "1800",
        "--repeats",
        str(SUITES[args.dataset]["repeats"]),
        "--seed",
        "42",
        "--work-dir",
        str(output_dir),
        "--enable-progress-tracker",
    ]
    if args.dataset == "live_code_bench":
        command.extend(
            [
                "--dataset-args",
                json.dumps(
                    {"live_code_bench": {"subset_list": ["release_v6"]}},
                    separators=(",", ":"),
                ),
            ]
        )
        sandbox = {
            "enabled": True,
            "engine": "docker",
            "pool_size": 4,
            "default_config": {
                "image": "python:3.11-slim",
                "platform": None,
                "network_enabled": False,
                "memory_limit": "512m",
                "tools_config": {
                    "shell_executor": {},
                    "python_executor": {},
                },
            },
        }
        command.extend(
            ["--sandbox", json.dumps(sandbox, separators=(",", ":"))]
        )
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    free_bytes = shutil.disk_usage(PROJECT_DIR).free
    print(
        f"Starting free space: {free_bytes} bytes "
        f"({free_bytes / 1024**3:.2f} GiB)"
    )
    print(f"Cache environment: {json.dumps(cache_environment, sort_keys=True)}")
    print(f"Command: {shlex.join(command)}", flush=True)
    if free_bytes < MIN_FREE_BYTES:
        print("Refusing to start: less than 15 GiB of free disk space.")
        return 1

    environment = os.environ.copy()
    environment.update(cache_environment)
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
