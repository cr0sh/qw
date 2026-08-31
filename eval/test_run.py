import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from eval import run


class GenerationConfigTests(unittest.TestCase):
    def command_for(self, dataset: str, *, limit: int | None = None) -> list[str]:
        args = SimpleNamespace(
            dataset=dataset,
            limit=limit,
            output_root=Path("/tmp/qwr-eval-test"),
        )

        with (
            mock.patch.object(run, "parse_args", return_value=args),
            mock.patch.object(
                run.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=run.MIN_FREE_BYTES),
            ),
            mock.patch.object(
                run.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as subprocess_run,
        ):
            self.assertEqual(run.main(), 0)

        return subprocess_run.call_args.args[0]

    def test_suites_emit_exact_evalscope_commands(self) -> None:
        expected_configs = {
            "gpqa_diamond": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 4_096,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            },
            "ifbench": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 4_096,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            },
            "live_code_bench": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 8_192,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            },
        }

        for dataset, config in expected_configs.items():
            with self.subTest(dataset=dataset):
                expected = [
                    "evalscope",
                    "eval",
                    "--model",
                    "qwen3.8-flash-next",
                    "--eval-type",
                    "openai_api",
                    "--api-url",
                    "http://127.0.0.1:8883/v1",
                    "--api-key",
                    "EMPTY",
                    "--datasets",
                    dataset,
                    "--generation-config",
                    json.dumps(config, separators=(",", ":")),
                    "--eval-batch-size",
                    "1",
                    "--timeout",
                    "1800",
                    "--repeats",
                    "1",
                    "--seed",
                    "42",
                    "--work-dir",
                    str(Path("/tmp/qwr-eval-test").resolve() / dataset),
                    "--enable-progress-tracker",
                ]
                if dataset == "live_code_bench":
                    expected.extend(
                        [
                            "--dataset-args",
                            json.dumps(
                                {
                                    "live_code_bench": {
                                        "subset_list": ["release_v6"]
                                    }
                                },
                                separators=(",", ":"),
                            ),
                            "--sandbox",
                            json.dumps(
                                {
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
                                },
                                separators=(",", ":"),
                            ),
                        ]
                    )

                self.assertEqual(self.command_for(dataset), expected)

    def test_limit_is_forwarded_to_evalscope(self) -> None:
        command = self.command_for("gpqa_diamond", limit=7)
        limit_index = command.index("--limit") + 1
        self.assertEqual(command[limit_index], "7")

if __name__ == "__main__":
    unittest.main()
