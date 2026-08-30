import json
import unittest

from eval import run


class GenerationConfigTests(unittest.TestCase):
    def test_thinking_suites_emit_only_supported_sampling_keys(self) -> None:
        expected_keys = {"temperature", "top_p", "top_k", "max_tokens", "seed"}
        expected_max_tokens = {
            "gpqa_diamond": 16_384,
            "ifbench": 8_192,
            "live_code_bench": 16_384,
        }

        for dataset, max_tokens in expected_max_tokens.items():
            with self.subTest(dataset=dataset):
                emitted = json.loads(
                    json.dumps(run.generation_config(dataset), separators=(",", ":"))
                )
                self.assertEqual(set(emitted), expected_keys)
                self.assertEqual(emitted["temperature"], 1.0)
                self.assertEqual(emitted["top_p"], 0.95)
                self.assertEqual(emitted["top_k"], 20)
                self.assertEqual(emitted["max_tokens"], max_tokens)
                self.assertEqual(emitted["seed"], 42)


if __name__ == "__main__":
    unittest.main()
