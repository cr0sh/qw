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

    def test_suites_emit_exact_generation_protocols(self) -> None:
        expected = {
            "gpqa_diamond": {
                "generation_config": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True,
                    "max_tokens": 32_768,
                },
                "repeats": "1",
            },
            "ifbench": {
                "generation_config": {
                    "temperature": 0,
                    "max_tokens": 32_768,
                },
                "repeats": "1",
            },
            "live_code_bench": {
                "generation_config": {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_tokens": 2_000,
                },
                "repeats": "10",
            },
        }

        for dataset, protocol in expected.items():
            with self.subTest(dataset=dataset):
                command = self.command_for(dataset)
                config_index = command.index("--generation-config") + 1
                config_json = command[config_index]
                self.assertEqual(
                    config_json,
                    json.dumps(
                        protocol["generation_config"], separators=(",", ":")
                    ),
                )
                config = json.loads(config_json)
                self.assertNotIn("seed", config)
                self.assertNotIn("extra_body", config)
                self.assertNotIn("enable_thinking", config_json)

                repeats_index = command.index("--repeats") + 1
                self.assertEqual(command[repeats_index], protocol["repeats"])
                seed_index = command.index("--seed") + 1
                self.assertEqual(command[seed_index], "42")
                datasets_index = command.index("--datasets") + 1
                self.assertEqual(command[datasets_index], dataset)
                api_url_index = command.index("--api-url") + 1
                self.assertEqual(command[api_url_index], "http://127.0.0.1:8883/v1")
                batch_size_index = command.index("--eval-batch-size") + 1
                self.assertEqual(command[batch_size_index], "1")
                self.assertEqual(command.count("--timeout"), 1)
                timeout_index = command.index("--timeout") + 1
                self.assertEqual(command[timeout_index], "1800")
                work_dir_index = command.index("--work-dir") + 1
                self.assertEqual(
                    command[work_dir_index],
                    str(Path("/tmp/qwr-eval-test").resolve() / dataset),
                )
                self.assertNotIn("--limit", command)

                if dataset == "live_code_bench":
                    dataset_args_index = command.index("--dataset-args") + 1
                    self.assertEqual(
                        command[dataset_args_index],
                        json.dumps(
                            {
                                "live_code_bench": {
                                    "subset_list": ["release_v6"]
                                }
                            },
                            separators=(",", ":"),
                        ),
                    )
                    sandbox_index = command.index("--sandbox") + 1
                    self.assertEqual(
                        command[sandbox_index],
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
                    )
                else:
                    self.assertNotIn("--dataset-args", command)
                    self.assertNotIn("--sandbox", command)

    def test_limit_is_forwarded_to_evalscope(self) -> None:
        command = self.command_for("gpqa_diamond", limit=7)
        limit_index = command.index("--limit") + 1
        self.assertEqual(command[limit_index], "7")

if __name__ == "__main__":
    unittest.main()
