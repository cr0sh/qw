import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiturn_latency import (  # noqa: E402
    TERMINATION_BOUND_NS,
    ChatSseAccumulator,
    attach_cache_traces,
    first_turn_request,
    intra_turn_request,
    next_user_turn_request,
    request_digest,
    server_trace_events,
    shared_context,
    tool_turn_request,
    turn_boundary_request,
    validate_cache_traces,
    validate_extended_scenarios,
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
        self.assertEqual(result["assistant_reasoning_text"], "R")
        self.assertEqual(result["latency_ns"]["ttft"], 10)
        self.assertEqual(result["latency_ns"]["semantic_ttft"], 8)
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
        boundary = turn_boundary_request(shared, "SEED", "reasoning", 8)
        discovery = intra_turn_request(shared, "discovery", "DISCOVERY", 8)
        probe = intra_turn_request(shared, "probe", "PROBE", 8)

        self.assertEqual(
            [message["role"] for message in boundary["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(boundary["messages"][:2], first["messages"])
        self.assertEqual(boundary["messages"][2]["reasoning_content"], "reasoning")
        self.assertNotEqual(request_digest(discovery), request_digest(probe))
        discovery_content = discovery["messages"][1]["content"]
        probe_content = probe["messages"][1]["content"]
        self.assertTrue(discovery_content.startswith(shared))
        self.assertTrue(probe_content.startswith(shared))
        self.assertNotEqual(discovery_content, probe_content)

    def test_sampling_defaults_are_omitted_and_tool_history_is_retained(self) -> None:
        shared = shared_context(2)
        omitted = first_turn_request(shared, 8)
        self.assertNotIn("temperature", omitted)
        self.assertNotIn("top_p", omitted)
        self.assertNotIn("top_k", omitted)
        self.assertNotIn("enable_thinking", omitted)

        explicit = first_turn_request(
            shared,
            8,
            {
                "enable_thinking": False,
                "reasoning_effort": "medium",
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
            },
        )
        self.assertEqual(explicit["temperature"], 0.7)
        self.assertEqual(explicit["top_k"], 20)
        self.assertEqual(explicit["presence_penalty"], 1.5)
        tool = tool_turn_request(shared, 8)
        followup = next_user_turn_request(shared, "ACK", "lookup reasoning", 8)
        self.assertEqual(
            [message["role"] for message in tool["messages"]],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(followup["messages"][:4], tool["messages"])
        self.assertEqual(followup["messages"][-1]["role"], "user")

    def test_validation_accepts_edited_input_misses_but_enforces_termination(self) -> None:
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
            scenario("intra_turn_probe", 0, 110, "probe"),
        ]
        checks = validate_scenarios(passing)
        self.assertTrue(all(checks.values()))

        passing[-1]["latency_ns"]["last_semantic_to_eof"] = TERMINATION_BOUND_NS
        with self.assertRaises(AssertionError):
            validate_scenarios(passing)

    def test_extended_scenarios_require_tool_history_reuse_and_publication(self) -> None:
        def scenario(
            name: str, message_count: int, cached_tokens: int
        ) -> dict[str, object]:
            return {
                "name": name,
                "message_count": message_count,
                "cache_evidence": {"cached_tokens": cached_tokens},
                "cache_publication": [{"event": "cache.published", "duration_ms": 1.0}],
                "filesystem_persistence": (
                    {"manifest_count": 1} if name == "next_user_turn" else None
                ),
            }

        checks = validate_extended_scenarios(
            [
                scenario("tool_turn", 4, 0),
                scenario("next_user_turn", 6, 10),
                scenario("filesystem_restore", 6, 10),
            ]
        )
        self.assertTrue(all(checks.values()))

    def test_cache_trace_validation_requires_lookup_and_decision_evidence(self) -> None:
        scenarios = []
        raw_events = []
        for name, hit, cached_tokens in [
            ("first_turn", False, 0),
            ("turn_boundary", True, 80),
            ("intra_turn_discovery", True, 70),
            ("intra_turn_probe", True, 70),
        ]:
            scenarios.append(
                {
                    "name": name,
                    "response_id": name,
                    "cache_evidence": {"cached_tokens": cached_tokens},
                }
            )
            raw_events.extend(
                [
                    {
                        "fields": {
                            "phase": "cache.lookup",
                            "hit": hit,
                            **({"cached_tokens": cached_tokens} if hit else {}),
                        },
                        "spans": [{"name": "generation", "response_id": name}],
                    },
                    {
                        "fields": {
                            "event": "cache.decision",
                            "source": "prefix_hit" if hit else "prefix_miss",
                            "reused_tokens": cached_tokens,
                            "response_id": name,
                        },
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.jsonl"
            log.write_text("\n".join(json.dumps(event) for event in raw_events))
            attach_cache_traces(scenarios, server_trace_events(log))

        checks = validate_cache_traces(scenarios)
        self.assertTrue(all(checks.values()))
        scenarios[-1]["server_cache_trace"][0]["cached_tokens"] = 1
        with self.assertRaises(AssertionError):
            validate_cache_traces(scenarios)



if __name__ == "__main__":
    unittest.main()
