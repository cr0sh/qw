import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from eval import run


class GenerationConfigTests(unittest.TestCase):
    def test_suites_emit_exact_non_thinking_generation_config(self) -> None:
        expected_max_tokens = {
            "gpqa_diamond": 4_096,
            "ifbench": 4_096,
            "live_code_bench": 8_192,
        }

        for dataset, max_tokens in expected_max_tokens.items():
            with self.subTest(dataset=dataset):
                expected = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_tokens": max_tokens,
                    "seed": 42,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                    },
                }
                args = argparse.Namespace(
                    dataset=dataset,
                    limit=None,
                    repeats=1,
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

                command = subprocess_run.call_args.args[0]
                config_index = command.index("--generation-config") + 1
                self.assertEqual(
                    command[config_index],
                    json.dumps(expected, separators=(",", ":")),
                )


if __name__ == "__main__":
    unittest.main()
