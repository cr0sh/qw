#!/usr/bin/env python3
"""Measure multi-turn TTFT, prefix reuse, and streamed termination on `qw serve`."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_LOCK = REPO_ROOT / "gpu-lock"
MODEL_ID = "qwen3.8-27b"
TERMINATION_BOUND_NS = 1_000_000_000
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_SHARED_REPETITIONS = 3_000
DEFAULT_MIN_CONTEXT_TOKENS = 50_000
DEFAULT_MAX_CONTEXT_TOKENS = 73_000
TTFT_DEFINITION = (
    "monotonic time immediately before the HTTP client writes the POST request "
    "through receipt of the first non-empty choices[0].delta.content SSE frame"
)
SEMANTIC_TTFT_DEFINITION = (
    "monotonic time immediately before the HTTP client writes the POST request "
    "through the first non-empty content, reasoning, or tool-call SSE delta"
)
TERMINATION_DEFINITION = (
    "the last non-empty content, reasoning, or tool-call delta to the Chat "
    "finish_reason terminal frame; terminal frame to data: [DONE], data: [DONE] "
    "to HTTP response-body EOF, and terminal frame to EOF are reported separately"
)
REQUEST_POLICY_FIELDS = (
    "enable_thinking",
    "reasoning_effort",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "seed",
)


def timestamp() -> dict[str, int]:
    return {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def canonical_body(request: dict[str, Any]) -> bytes:
    return json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")


def request_digest(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_body(request)).hexdigest()


def user_message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def make_request(
    messages: list[dict[str, Any]],
    max_tokens: int,
    controls: dict[str, Any] | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    if controls:
        for field in REQUEST_POLICY_FIELDS:
            value = controls.get(field)
            if value is not None:
                request[field] = value
    if tools is not None:
        request["tools"] = tools
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return request


def shared_context(repetitions: int) -> str:
    return "\n".join(
        f"Shared context line {index:04d}: amber birch cobalt delta ember fjord granite harbor."
        for index in range(repetitions)
    )


def first_turn_request(
    shared: str, max_tokens: int, controls: dict[str, Any] | None = None
) -> dict[str, Any]:
    return make_request(
        [
            {
                "role": "system",
                "content": "Answer with only the requested uppercase word and no punctuation.",
            },
            user_message(
                f"{shared}\nDivergent suffix: seed. Reply with the single word SEED."
            ),
        ],
        max_tokens,
        controls,
    )


def turn_boundary_request(
    shared: str,
    first_assistant_text: str,
    first_reasoning_text: str,
    max_tokens: int,
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = first_turn_request(shared, max_tokens, controls)
    request["messages"].extend(
        [
            {
                "role": "assistant",
                "content": first_assistant_text,
                "reasoning_content": first_reasoning_text,
            },
            user_message(
                "This is the next conversational turn. Reply with the single word BOUNDARY."
            ),
        ]
    )
    return request


def intra_turn_request(
    shared: str,
    suffix: str,
    answer: str,
    max_tokens: int,
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_request(
        [
            {
                "role": "system",
                "content": "Answer with only the requested uppercase word and no punctuation.",
            },
            user_message(
                f"{shared}\nDivergent suffix: {suffix}. Reply with the single word {answer}."
            ),
        ],
        max_tokens,
        controls,
    )


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a stable acceptance-test value.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def tool_turn_request(
    shared: str, max_tokens: int, controls: dict[str, Any] | None = None
) -> dict[str, Any]:
    return make_request(
        [
            {
                "role": "system",
                "content": "Use the tool history, then answer with one uppercase word.",
            },
            user_message(
                f"{shared}\nAsk the lookup tool for the acceptance value, then acknowledge it."
            ),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_acceptance_lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key":"acceptance"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_acceptance_lookup",
                "content": "stable acceptance value",
            },
        ],
        max_tokens,
        controls,
        tools=tool_definitions(),
        tool_choice="none",
    )


def next_user_turn_request(
    shared: str,
    tool_assistant_text: str,
    tool_assistant_reasoning: str,
    max_tokens: int,
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = tool_turn_request(shared, max_tokens, controls)
    request["messages"].extend(
        [
            {
                "role": "assistant",
                "content": tool_assistant_text,
                "reasoning_content": tool_assistant_reasoning,
            },
            user_message(
                "Now answer the next user turn with the single word FOLLOWUP."
            ),
        ]
    )
    return request


class ChatSseAccumulator:
    SEMANTIC_KINDS = ("content", "reasoning", "tool")

    def __init__(self, request_started: dict[str, int]) -> None:
        self.request_started = request_started
        self.response_headers: dict[str, int] | None = None
        self.first_semantic: dict[str, dict[str, int] | None] = {
            kind: None for kind in self.SEMANTIC_KINDS
        }
        self.last_semantic: dict[str, dict[str, int] | None] = {
            kind: None for kind in self.SEMANTIC_KINDS
        }
        self.terminal_finish: dict[str, int] | None = None
        self.terminal_done: dict[str, int] | None = None
        self.response_id: str | None = None
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.assistant_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.sse_frames = 0

    def record_semantic(self, kind: str, received: dict[str, int]) -> None:
        if self.first_semantic[kind] is None:
            self.first_semantic[kind] = received
        self.last_semantic[kind] = received

    def accept(self, data: str, received: dict[str, int]) -> None:
        self.sse_frames += 1
        if data == "[DONE]":
            if self.terminal_done is not None:
                raise ValueError("stream emitted data: [DONE] more than once")
            self.terminal_done = received
            return

        value = json.loads(data)
        response_id = value.get("id")
        if isinstance(response_id, str):
            if self.response_id is not None and self.response_id != response_id:
                raise ValueError("stream response ID changed")
            self.response_id = response_id
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if isinstance(content, str) and content:
                self.record_semantic("content", received)
                self.assistant_parts.append(content)
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                self.record_semantic("reasoning", received)
                self.reasoning_parts.append(reasoning)
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                self.record_semantic("tool", received)
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self.finish_reason = finish_reason
                self.terminal_finish = received
        usage = value.get("usage")
        if isinstance(usage, dict):
            self.usage = usage

    def finish(self, eof: dict[str, int]) -> dict[str, Any]:
        if self.response_headers is None:
            raise ValueError("response headers timestamp was not recorded")
        observed = [
            stamp
            for stamp in self.last_semantic.values()
            if stamp is not None
        ]
        if not observed:
            raise ValueError("stream ended without a non-empty semantic assistant delta")
        if self.terminal_finish is None or self.finish_reason is None:
            raise ValueError("stream ended without a terminal finish_reason chunk")
        if self.terminal_done is None:
            raise ValueError("stream ended without data: [DONE]")
        if self.usage is None:
            raise ValueError("stream ended without usage")
        if self.response_id is None:
            raise ValueError("stream ended without a response ID")

        prompt_details = self.usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            raise ValueError("usage omitted prompt_tokens_details")
        cached_tokens = prompt_details.get("cached_tokens")
        prompt_tokens = self.usage.get("prompt_tokens")
        if not isinstance(cached_tokens, int) or not isinstance(prompt_tokens, int):
            raise ValueError("usage cached/prompt token counts are not integers")

        request_ns = self.request_started["monotonic_ns"]
        terminal_ns = self.terminal_finish["monotonic_ns"]
        done_ns = self.terminal_done["monotonic_ns"]
        eof_ns = eof["monotonic_ns"]
        first_semantic_ns = {
            kind: stamp["monotonic_ns"] if stamp is not None else None
            for kind, stamp in self.first_semantic.items()
        }
        last_semantic_ns = {
            kind: stamp["monotonic_ns"] if stamp is not None else None
            for kind, stamp in self.last_semantic.items()
        }
        first_any_ns = min(
            value for value in first_semantic_ns.values() if value is not None
        )
        last_any_ns = max(
            value for value in last_semantic_ns.values() if value is not None
        )
        if not request_ns <= first_any_ns <= last_any_ns <= terminal_ns <= done_ns <= eof_ns:
            raise ValueError("observed SSE timestamps are not monotonic")

        def since_request(kind: str) -> int | None:
            value = first_semantic_ns[kind]
            return None if value is None else value - request_ns

        def to_terminal(kind: str) -> int | None:
            value = last_semantic_ns[kind]
            return None if value is None else terminal_ns - value

        assistant_text = "".join(self.assistant_parts)
        reasoning_text = "".join(self.reasoning_parts)
        return {
            "response_id": self.response_id,
            "assistant_text": assistant_text,
            "assistant_reasoning_text": reasoning_text,
            "assistant_text_sha256": hashlib.sha256(
                assistant_text.encode("utf-8")
            ).hexdigest(),
            "assistant_reasoning_sha256": hashlib.sha256(
                reasoning_text.encode("utf-8")
            ).hexdigest(),
            "finish_reason": self.finish_reason,
            "sse_frames": self.sse_frames,
            "timestamps": {
                "request_started": self.request_started,
                "response_headers": self.response_headers,
                "first_content_delta": self.first_semantic["content"],
                "first_reasoning_delta": self.first_semantic["reasoning"],
                "first_tool_delta": self.first_semantic["tool"],
                "last_content_delta": self.last_semantic["content"],
                "last_reasoning_delta": self.last_semantic["reasoning"],
                "last_tool_delta": self.last_semantic["tool"],
                "last_semantic_delta": max(
                    observed, key=lambda stamp: stamp["monotonic_ns"]
                ),
                "terminal_finish_chunk": self.terminal_finish,
                "terminal_sse_done": self.terminal_done,
                "eof": eof,
            },
            "latency_ns": {
                "ttft": since_request("content"),
                "semantic_ttft": first_any_ns - request_ns,
                "time_to_first_reasoning": since_request("reasoning"),
                "time_to_first_tool": since_request("tool"),
                "last_content_to_terminal": to_terminal("content"),
                "last_reasoning_to_terminal": to_terminal("reasoning"),
                "last_tool_to_terminal": to_terminal("tool"),
                "last_semantic_to_terminal": terminal_ns - last_any_ns,
                "terminal_to_done": done_ns - terminal_ns,
                "done_to_eof": eof_ns - done_ns,
                "terminal_to_eof": eof_ns - terminal_ns,
                "last_semantic_to_eof": eof_ns - last_any_ns,
            },
            "usage": self.usage,
            "cache_evidence": {
                "cached_tokens": cached_tokens,
                "prompt_tokens": prompt_tokens,
                "partial_prefix_hit": 0 < cached_tokens < prompt_tokens,
            },
        }


def post_stream(
    host: str,
    port: int,
    request: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = canonical_body(request)
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    started = timestamp()
    accumulator = ChatSseAccumulator(started)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        accumulator.response_headers = timestamp()
        if response.status != 200:
            error_body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status}: {error_body}")

        event_lines: list[str] = []
        while True:
            line = response.readline()
            received = timestamp()
            if not line:
                if event_lines:
                    raise ValueError("HTTP body ended with an incomplete SSE frame")
                result = accumulator.finish(received)
                break
            text = line.decode("utf-8").rstrip("\r\n")
            if text:
                event_lines.append(text)
                continue
            data_lines = [
                entry.removeprefix("data:").lstrip()
                for entry in event_lines
                if entry.startswith("data:")
            ]
            event_lines.clear()
            if data_lines:
                accumulator.accept("\n".join(data_lines), received)
    finally:
        connection.close()

    result["request_sha256"] = hashlib.sha256(body).hexdigest()
    result["request_body_bytes"] = len(body)
    result["message_count"] = len(request["messages"])
    return result


def reserve_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def wait_until_ready(
    process: subprocess.Popen[bytes], host: str, port: int, timeout_seconds: float
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"qw serve exited before readiness with status {returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return timestamp()
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"qw serve did not listen within {timeout_seconds:.1f}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)

def persisted_manifest_state(cache_directory: str) -> dict[str, tuple[int, int]]:
    root = Path(cache_directory) / "entries"
    state: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return state
    for path in root.rglob("*.json"):
        try:
            metadata = path.stat()
        except OSError:
            continue
        state[os.fspath(path)] = (metadata.st_mtime_ns, metadata.st_size)
    return state


def wait_for_persisted_manifest(
    cache_directory: str,
    previous: dict[str, tuple[int, int]],
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = persisted_manifest_state(cache_directory)
        if current != previous:
            return {
                "manifest_count": len(current),
                "manifest_paths": sorted(current),
            }
        time.sleep(0.01)
    raise TimeoutError("cache snapshot was published but no persisted manifest appeared")



TRACE_FIELDS = {
    "event",
    "phase",
    "response_id",
    "entry_id",
    "tier",
    "from",
    "to",
    "token_count",
    "capacity_bytes",
    "snapshot_bytes",
    "hot_capacity_bytes",
    "logical_snapshot_bytes",
    "unique_page_bytes",
    "reclaimed_bytes",
    "error",
    "kind",
    "source",
    "hit",
    "route",
    "decoder",
    "configured_decoder",
    "dflash2_available",
    "dflash2_compatible",
    "enable_thinking",
    "reasoning_effort",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "prompt_tokens",
    "cached_tokens",
    "prefilled_tokens",
    "completion_tokens",
    "decoded_tokens",
    "tps",
    "prefill_duration_ms",
    "decode_duration_ms",
    "elapsed_ms",
    "duration_ms",
    "hot_bytes",
    "reused_tokens",
    "lookup_duration_ms",
    "accepted_draft_tokens",
    "proposed_draft_tokens",
    "acceptance_percentage",
    "projected_context_cache_hits",
    "verify_width_rounds",
    "verify_width_accepted_draft_tokens",
    "verify_width_proposed_draft_tokens",
    "selector_scale_rounds",
    "selector_scale_accepted_draft_tokens",
    "selector_scale_proposed_draft_tokens",
    "stats",
}
TRACE_MARKERS = {
    "prompt.tokenized",
    "decoder.selected",
    "cache.lookup",
    "cache.insert",
    "cache.insert.summary",
    "cache.restore",
    "cache.promote",
    "cache.expire",
    "cache.evict",
    "cache.persistence_error",
    "cache.resume",
    "cache.decision",
    "cache.published",
    "prefill.complete",
    "ttft.prefill",
    "decode.complete",
    "dflash2.completed",
}


def _trace_response_id(event: dict[str, Any], fields: dict[str, Any]) -> str | None:
    response_id = fields.get("response_id")
    if isinstance(response_id, str):
        return response_id
    span_candidates: list[dict[str, Any]] = []
    span = event.get("span")
    if isinstance(span, dict):
        span_candidates.append(span)
    spans = event.get("spans")
    if isinstance(spans, list):
        span_candidates.extend(
            candidate for candidate in reversed(spans) if isinstance(candidate, dict)
        )
    for candidate in span_candidates:
        response_id = candidate.get("response_id")
        if isinstance(response_id, str):
            return response_id
    return None


def server_trace_events(log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        fields = event.get("fields")
        if not isinstance(fields, dict):
            continue
        marker = fields.get("event") or fields.get("phase")
        if marker not in TRACE_MARKERS:
            continue
        evidence = {
            key: fields[key] for key in TRACE_FIELDS if key in fields
        }
        response_id = _trace_response_id(event, fields)
        if response_id is not None:
            evidence["response_id"] = response_id
        evidence["log_timestamp"] = event.get("timestamp")
        events.append(evidence)
    return events


def cache_trace_events(log_path: Path) -> list[dict[str, Any]]:
    return [
        event
        for event in server_trace_events(log_path)
        if event.get("event", "").startswith("cache.")
        or event.get("phase", "").startswith("cache.")
    ]


def attach_cache_traces(
    scenarios: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    for scenario in scenarios:
        response_id = scenario.get("response_id")
        scenario["server_cache_trace"] = [
            event for event in events if event.get("response_id") == response_id
        ]


def attach_server_traces(
    scenarios: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    for scenario in scenarios:
        response_id = scenario.get("response_id")
        traces = [
            event for event in events if event.get("response_id") == response_id
        ]
        scenario["server_trace"] = traces
        scenario["cache_publication"] = [
            event for event in traces if event.get("event") == "cache.published"
        ]
def validate_cache_traces(scenarios: list[dict[str, Any]]) -> dict[str, bool]:
    expected_hits = {
        "first_turn": False,
        "turn_boundary": True,
        "intra_turn_discovery": None,
        "intra_turn_probe": None,
        "tool_turn": False,
        "next_user_turn": True,
        "filesystem_restore": True,
    }
    checks: dict[str, bool] = {}
    for scenario in scenarios:
        name = scenario["name"]
        if name not in expected_hits:
            continue
        traces = scenario["server_cache_trace"]
        decisions = [event for event in traces if event.get("event") == "cache.decision"]
        lookups = [event for event in traces if event.get("phase") == "cache.lookup"]
        cached_tokens = scenario["cache_evidence"]["cached_tokens"]
        expected_hit = expected_hits[name]
        if expected_hit is None:
            expected_hit = cached_tokens > 0
        checks[f"{name}_cache_decision"] = (
            len(decisions) == 1
            and decisions[0].get("source")
            == ("prefix_hit" if expected_hit else "prefix_miss")
            and decisions[0].get("reused_tokens") == cached_tokens
        )
        checks[f"{name}_cache_lookup"] = (
            len(lookups) == 1
            and lookups[0].get("hit") is expected_hit
            and (
                not expected_hit
                or lookups[0].get("cached_tokens") == cached_tokens
            )
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"failed server cache trace checks {failed}")
    return checks


def validate_context_window(
    scenarios: list[dict[str, Any]], minimum: int, maximum: int
) -> dict[str, bool]:
    checks = {
        f"{scenario['name']}_prompt_in_context_window": minimum
        <= scenario["cache_evidence"]["prompt_tokens"]
        <= maximum
        for scenario in scenarios
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        values = {
            scenario["name"]: scenario["cache_evidence"]["prompt_tokens"]
            for scenario in scenarios
        }
        raise AssertionError(
            f"prompt token counts outside [{minimum}, {maximum}]: {values}"
        )
    return checks


def validate_extended_scenarios(
    scenarios: list[dict[str, Any]]
) -> dict[str, bool]:
    by_name = {scenario["name"]: scenario for scenario in scenarios}
    required = {"tool_turn", "next_user_turn", "filesystem_restore"}
    if not required <= by_name.keys():
        raise AssertionError(
            f"extended scenario set missing: {sorted(required - by_name.keys())}"
        )
    checks = {
        "tool_turn_has_tool_history": by_name["tool_turn"]["message_count"] >= 4,
        "next_user_retains_tool_history": by_name["next_user_turn"]["message_count"]
        > by_name["tool_turn"]["message_count"],
        "next_user_reuses_tool_prefix": by_name["next_user_turn"][
            "cache_evidence"
        ]["cached_tokens"]
        > 0,
        "filesystem_restore_reuses_prefix": by_name["filesystem_restore"][
            "cache_evidence"
        ]["cached_tokens"]
        > 0,
        "filesystem_manifest_persisted_before_restart": bool(
            by_name["next_user_turn"].get("filesystem_persistence")
        ),
        "publication_evidence_present": all(
            by_name[name].get("cache_publication") for name in required
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"failed extended scenario checks {failed}")
    return checks


def validate_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {scenario["name"]: scenario for scenario in scenarios}
    required = {"first_turn", "turn_boundary", "intra_turn_discovery", "intra_turn_probe"}
    if not required <= by_name.keys():
        raise AssertionError(f"scenario set missing: {sorted(required - by_name.keys())}")
    first = by_name["first_turn"]
    boundary = by_name["turn_boundary"]
    discovery = by_name["intra_turn_discovery"]
    probe = by_name["intra_turn_probe"]
    first_cached = first["cache_evidence"]["cached_tokens"]
    boundary_cached = boundary["cache_evidence"]["cached_tokens"]
    discovery_cached = discovery["cache_evidence"]["cached_tokens"]
    probe_cached = probe["cache_evidence"]["cached_tokens"]
    checks = {
        "isolated_first_turn_cache_miss": first_cached == 0,
        "turn_boundary_partial_prefix_hit": boundary["cache_evidence"][
            "partial_prefix_hit"
        ],
        "intra_turn_requests_are_distinct": discovery["request_sha256"]
        != probe["request_sha256"],
        "intra_turn_edit_without_completed_prefix_misses": discovery_cached == 0
        and probe_cached == 0,
        "all_streams_terminate_under_one_second": all(
            scenario["latency_ns"]["last_semantic_to_eof"] < TERMINATION_BOUND_NS
            for scenario in scenarios
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        values = {
            "first_cached": first_cached,
            "boundary_cached": boundary_cached,
            "discovery_cached": discovery_cached,
            "probe_cached": probe_cached,
        }
        raise AssertionError(f"failed checks {failed}; cache values {values}")
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "target" / "debug" / "qw",
        help="already-built qw binary",
    )
    parser.add_argument("--model", type=Path, help="explicit model checkpoint directory")
    parser.add_argument(
        "--dflash-draft-model", type=Path, help="explicit DFlash2 draft checkpoint directory"
    )
    parser.add_argument(
        "--decoder",
        choices=("auto", "baseline", "mtp", "dflash"),
        default="auto",
        help="decoder policy passed to qw serve",
    )
    parser.add_argument(
        "--mtp-k",
        type=int,
        default=3,
        help="MTP verify block size for --decoder mtp (default: 3)",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="send enable_thinking=true; omitted by default so server defaults apply",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="send enable_thinking=false",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--reasoning-effort", default="medium")
    for name, value_type in (
        ("temperature", float),
        ("top-p", float),
        ("top-k", int),
        ("min-p", float),
        ("presence-penalty", float),
        ("repetition-penalty", float),
        ("seed", int),
    ):
        parser.add_argument(
            f"--{name}",
            dest=name.replace("-", "_"),
            type=value_type,
            help="optional explicit sampling override; omitted fields use runtime defaults",
        )
    parser.add_argument("--port", type=int, help="isolated TCP port; defaults to an ephemeral port")
    parser.add_argument("--result", type=Path, help="unique JSON result path")
    parser.add_argument(
        "--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--shared-repetitions",
        type=int,
        default=DEFAULT_SHARED_REPETITIONS,
        help="repeated shared context lines; default targets a 50–73k token prompt",
    )
    parser.add_argument(
        "--min-context-tokens", type=int, default=DEFAULT_MIN_CONTEXT_TOKENS
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS
    )
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args(argv)
    if args.shared_repetitions < 2:
        parser.error("--shared-repetitions must be at least 2")
    if args.mtp_k < 2:
        parser.error("--mtp-k must be at least 2")
    if args.min_context_tokens < 1 or args.max_context_tokens < args.min_context_tokens:
        parser.error("--max-context-tokens must be >= --min-context-tokens >= 1")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")
    if args.reasoning_effort.strip() == "":
        parser.error("--reasoning-effort must not be empty")
    return args


def request_controls(args: argparse.Namespace) -> dict[str, Any]:
    controls = {
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }
    return {key: value for key, value in controls.items() if value is not None}
def default_result_path() -> Path:
    name = f"multiturn-latency-{time.time_ns()}-{os.getpid()}.json"
    return REPO_ROOT / "outputs" / name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        print(f"qw binary does not exist: {binary}", file=sys.stderr)
        return 2
    result_path = (args.result or default_result_path()).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        print(f"refusing to overwrite result file: {result_path}", file=sys.stderr)
        return 2
    log_path = result_path.with_suffix(result_path.suffix + ".server.log")
    if log_path.exists():
        print(f"refusing to overwrite server log: {log_path}", file=sys.stderr)
        return 2

    host = "127.0.0.1"
    port = args.port or reserve_port(host)
    run_started = timestamp()
    controls = request_controls(args)
    scenarios: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "running",
        "definitions": {
            "ttft": TTFT_DEFINITION,
            "semantic_ttft": SEMANTIC_TTFT_DEFINITION,
            "stream_termination": TERMINATION_DEFINITION,
            "strict_termination_bound_ns": TERMINATION_BOUND_NS,
        },
        "decoder_policy": args.decoder,
        "mtp_k": args.mtp_k,
        "request_controls": controls,
        "cache_defaults": {
            "memory_bytes": 2 * 1024 * 1024 * 1024,
            "filesystem_bytes": 16 * 1024 * 1024 * 1024,
            "capacity_flags_omitted": True,
        },
        "run_started": run_started,
        "shared_context": {
            "repetitions": args.shared_repetitions,
            "minimum_prompt_tokens": args.min_context_tokens,
            "maximum_prompt_tokens": args.max_context_tokens,
        },
        "scenarios": scenarios,
    }

    process: subprocess.Popen[bytes] | None = None
    server_log: Any = None
    exit_code = 1
    with tempfile.TemporaryDirectory(prefix="qw-multiturn-latency-cache-") as cache_directory:
        def command_for(run_port: int) -> list[str]:
            command = [
                os.fspath(GPU_LOCK),
                "--",
                os.fspath(binary),
                "serve",
                "--bind",
                f"{host}:{run_port}",
                "--model-id",
                MODEL_ID,
                "--prefix-cache-directory",
                cache_directory,
                "--decoder",
                args.decoder,
                "--no-file-logging",
                "--output-format",
                "json",
            ]
            if args.decoder == "mtp":
                command.extend(["--mtp-k", str(args.mtp_k)])
            if args.dflash_draft_model is not None:
                command.extend(
                    ["--dflash-draft-model", os.fspath(args.dflash_draft_model.expanduser().resolve())]
                )
            if args.model is not None:
                command.extend(["--model", os.fspath(args.model.expanduser().resolve())])
            return command

        def launch(run_port: int, append_log: bool) -> tuple[subprocess.Popen[bytes], Any]:
            mode = "ab" if append_log else "xb"
            log_handle = log_path.open(mode)
            try:
                child = subprocess.Popen(
                    command_for(run_port),
                    cwd=REPO_ROOT,
                    env=os.environ
                    | {
                        "RUST_LOG": "info,qw_server=trace,qw_prefix_cache=trace,qw_runtime=debug",
                        "QW_GPU_LOCK_SESSION": f"multiturn-latency-{os.getpid()}",
                    },
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except BaseException:
                log_handle.close()
                raise
            return child, log_handle

        def record(
            name: str, request: dict[str, Any], run_port: int
        ) -> dict[str, Any]:
            result = post_stream(host, run_port, request, args.request_timeout)
            result["name"] = name
            scenarios.append(result)
            return result

        report["server"] = {
            "command": command_for(port),
            "host": host,
            "port": port,
            "cache_directory": cache_directory,
            "log_path": os.fspath(log_path),
            "capacity_flags_omitted": True,
        }
        shared = shared_context(args.shared_repetitions)
        report["shared_context"]["sha256"] = hashlib.sha256(
            shared.encode("utf-8")
        ).hexdigest()
        report["shared_context"]["bytes"] = len(shared.encode("utf-8"))

        try:
            process, server_log = launch(port, append_log=False)
            report["server"]["pid"] = process.pid
            report["server"]["ready_at"] = wait_until_ready(
                process, host, port, args.startup_timeout
            )

            request_plan: list[tuple[str, dict[str, Any]]] = []
            seed_request = first_turn_request(shared, args.max_tokens, controls)
            request_plan.append(("first_turn", seed_request))
            seed = record("first_turn", seed_request, port)

            boundary_request = turn_boundary_request(
                shared,
                seed["assistant_text"],
                seed["assistant_reasoning_text"],
                args.max_tokens,
                controls,
            )
            request_plan.append(("turn_boundary", boundary_request))
            boundary = record("turn_boundary", boundary_request, port)

            discovery_request = intra_turn_request(
                shared, "structural discovery", "DISCOVERY", args.max_tokens, controls
            )
            request_plan.append(("intra_turn_discovery", discovery_request))
            discovery = record("intra_turn_discovery", discovery_request, port)

            probe_request = intra_turn_request(
                shared, "structural probe", "PROBE", args.max_tokens, controls
            )
            request_plan.append(("intra_turn_probe", probe_request))
            probe = record("intra_turn_probe", probe_request, port)

            tool_request = tool_turn_request(shared, args.max_tokens, controls)
            request_plan.append(("tool_turn", tool_request))
            tool_result = record("tool_turn", tool_request, port)

            next_request = next_user_turn_request(
                shared,
                tool_result["assistant_text"],
                tool_result["assistant_reasoning_text"],
                args.max_tokens,
                controls,
            )
            request_plan.append(("next_user_turn", next_request))
            manifests_before_restart = persisted_manifest_state(cache_directory)
            next_result = record("next_user_turn", next_request, port)
            next_result["filesystem_persistence"] = wait_for_persisted_manifest(
                cache_directory, manifests_before_restart
            )

            stop_server(process)
            server_log.close()
            process = None
            server_log = None

            restart_port = reserve_port(host)
            process, server_log = launch(restart_port, append_log=True)
            report["server"]["restart"] = {
                "command": command_for(restart_port),
                "port": restart_port,
                "pid": process.pid,
                "ready_at": wait_until_ready(
                    process, host, restart_port, args.startup_timeout
                ),
            }
            filesystem_request = next_request
            request_plan.append(("filesystem_restore", filesystem_request))
            record("filesystem_restore", filesystem_request, restart_port)

            report["request_plan"] = [
                {
                    "name": name,
                    "sha256": request_digest(request),
                    "message_count": len(request["messages"]),
                    "provided_policy_fields": sorted(
                        key for key in REQUEST_POLICY_FIELDS if key in request
                    ),
                }
                for name, request in request_plan
            ]
            report["checks"] = validate_scenarios(scenarios)
            report["checks"].update(
                validate_context_window(
                    scenarios, args.min_context_tokens, args.max_context_tokens
                )
            )
            report["status"] = "passed"
            exit_code = 0
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                stop_server(process)
                report["server"]["exit_status"] = process.returncode
            if server_log is not None:
                server_log.close()
            report["run_finished"] = timestamp()

    events = server_trace_events(log_path) if log_path.exists() else []
    report["server_trace"] = events
    report["server_cache_trace"] = [
        event
        for event in events
        if event.get("event", "").startswith("cache.")
        or event.get("phase", "").startswith("cache.")
    ]
    attach_cache_traces(scenarios, report["server_cache_trace"])
    attach_server_traces(scenarios, events)
    if exit_code == 0:
        try:
            report["checks"].update(validate_cache_traces(scenarios))
            report["checks"].update(validate_extended_scenarios(scenarios))
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            exit_code = 1
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"result: {result_path}")
    print(f"server log: {log_path}")
    for scenario in scenarios:
        print(
            f"{scenario['name']}: ttft_ns={scenario['latency_ns']['ttft']} "
            f"semantic_ttft_ns={scenario['latency_ns']['semantic_ttft']} "
            f"last_semantic_to_terminal_ns={scenario['latency_ns']['last_semantic_to_terminal']} "
            f"terminal_to_done_ns={scenario['latency_ns']['terminal_to_done']} "
            f"done_to_eof_ns={scenario['latency_ns']['done_to_eof']} "
            f"cached_tokens={scenario['cache_evidence']['cached_tokens']}/"
            f"{scenario['cache_evidence']['prompt_tokens']}"
        )
    if exit_code:
        print(report.get("error", "latency smoke failed"), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
