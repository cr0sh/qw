from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiturn_latency import (  # noqa: E402
    TERMINATION_BOUND_NS,
    ChatSseAccumulator,
    first_turn_request,
    intra_turn_request,
    request_digest,
    shared_context,
    turn_boundary_request,
    validate_cache_traces,
    validate_scenarios,
)


def stamp(monotonic_ns: int) -> dict[str, int]:
    return {"unix_ns": 1_000_000_000 + monotonic_ns, "monotonic_ns": monotonic_ns}


class ChatSseAccumulatorTests(unittest.TestCase):
    def test_reports_ttft_terminal_and_eof_as_separate_intervals(self) -> None:
        accumulator = ChatSseAccumulator(stamp(10))
        accumulator.response_headers = stamp(15)
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
            stamp(16),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{"reasoning_content":"R"},"finish_reason":null}]}',
            stamp(18),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{"content":"A"},"finish_reason":null}]}',
            stamp(20),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0}]},"finish_reason":null}]}',
            stamp(25),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{"content":"B"},"finish_reason":null}]}',
            stamp(30),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}]}',
            stamp(40),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102,"prompt_tokens_details":{"cached_tokens":60}}}',
            stamp(45),
        )
        accumulator.accept("[DONE]", stamp(50))

        result = accumulator.finish(stamp(60))

        self.assertEqual(result["assistant_text"], "AB")
        self.assertEqual(result["latency_ns"]["ttft"], 10)
        self.assertEqual(result["latency_ns"]["time_to_first_reasoning"], 8)
        self.assertEqual(result["latency_ns"]["time_to_first_tool"], 15)
        self.assertEqual(result["latency_ns"]["last_semantic_to_terminal"], 10)
        self.assertEqual(result["latency_ns"]["terminal_to_done"], 10)
        self.assertEqual(result["latency_ns"]["done_to_eof"], 10)
        self.assertEqual(result["latency_ns"]["terminal_to_eof"], 20)
        self.assertEqual(result["latency_ns"]["last_semantic_to_eof"], 30)
        self.assertEqual(result["cache_evidence"]["cached_tokens"], 60)
        self.assertTrue(result["cache_evidence"]["partial_prefix_hit"])

    def test_rejects_eof_without_done(self) -> None:
        accumulator = ChatSseAccumulator(stamp(10))
        accumulator.response_headers = stamp(15)
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{"content":"A"},"finish_reason":null}]}',
            stamp(20),
        )
        accumulator.accept(
            '{"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"prompt_tokens_details":{"cached_tokens":0}}}',
            stamp(30),
        )
        with self.assertRaisesRegex(ValueError, r"data: \[DONE\]"):
            accumulator.finish(stamp(40))


class ScenarioTests(unittest.TestCase):
    def test_requests_cover_true_multiturn_and_distinct_intra_turn_branches(self) -> None:
        shared = shared_context(2)
        first = first_turn_request(shared, 8)
        boundary = turn_boundary_request(shared, "SEED", 8)
        discovery = intra_turn_request(shared, "discovery", "DISCOVERY", 8)
        probe = intra_turn_request(shared, "probe", "PROBE", 8)

        self.assertEqual(
            [message["role"] for message in boundary["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(boundary["messages"][:2], first["messages"])
        self.assertNotEqual(request_digest(discovery), request_digest(probe))
        discovery_content = discovery["messages"][1]["content"]
        probe_content = probe["messages"][1]["content"]
        self.assertTrue(discovery_content.startswith(shared))
        self.assertTrue(probe_content.startswith(shared))
        self.assertNotEqual(discovery_content, probe_content)

    def test_validation_enforces_strict_termination_and_checkpoint_gain(self) -> None:
        def scenario(
            name: str,
            cached_tokens: int,
            prompt_tokens: int,
            request_sha256: str,
            termination_ns: int = 1,
        ) -> dict[str, object]:
            return {
                "name": name,
                "request_sha256": request_sha256,
                "cache_evidence": {
                    "cached_tokens": cached_tokens,
                    "prompt_tokens": prompt_tokens,
                    "partial_prefix_hit": 0 < cached_tokens < prompt_tokens,
                },
                "latency_ns": {"last_semantic_to_eof": termination_ns},
            }

        passing = [
            scenario("first_turn", 0, 100, "first"),
            scenario("turn_boundary", 80, 140, "boundary"),
            scenario("intra_turn_discovery", 0, 110, "discovery"),
            scenario("intra_turn_probe", 70, 110, "probe"),
        ]
        checks = validate_scenarios(passing)
        self.assertTrue(all(checks.values()))

        passing[-1]["latency_ns"]["last_semantic_to_eof"] = TERMINATION_BOUND_NS
        with self.assertRaisesRegex(AssertionError, "under_one_second"):
            validate_scenarios(passing)
    def test_cache_trace_validation_requires_lookup_and_decision_evidence(self) -> None:
        scenarios = []
        for name, hit, cached_tokens in [
            ("first_turn", False, 0),
            ("turn_boundary", True, 80),
            ("intra_turn_discovery", False, 0),
            ("intra_turn_probe", True, 70),
        ]:
            scenarios.append(
                {
                    "name": name,
                    "cache_evidence": {"cached_tokens": cached_tokens},
                    "server_cache_trace": [
                        {
                            "phase": "cache.lookup",
                            "hit": hit,
                            **({"cached_tokens": cached_tokens} if hit else {}),
                        },
                        {
                            "event": "cache.decision",
                            "source": "prefix_hit" if hit else "prefix_miss",
                            "reused_tokens": cached_tokens,
                        },
                    ],
                }
            )

        checks = validate_cache_traces(scenarios)
        self.assertTrue(all(checks.values()))
        scenarios[-1]["server_cache_trace"][0]["cached_tokens"] = 1
        with self.assertRaisesRegex(AssertionError, "cache trace checks"):
            validate_cache_traces(scenarios)



if __name__ == "__main__":
    unittest.main()
