"""Durable EvalScope/Tau2 bridge for τ³ banking runs.

The installed EvalScope τ³ adapter currently drops reasoning content while converting
Tau2 messages and catches every ``run_task`` exception as a zero reward.  This module
is imported by the runner before evaluation and replaces only those hooks; the
third-party installation remains untouched.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from contextvars import ContextVar
from typing import Any

from openai import APIStatusError

from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages.chat_message import dict_to_chat_message
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.models.utils.openai import openai_chat_message, openai_chat_tools
from evalscope.api.tool.tool_info import ToolInfo
from evalscope.constants import EvalType
from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.utils import llm_utils as tau_llm_utils


class Tau3AdapterError(RuntimeError):
    """An error with a structured classification for capture/reporting."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
SIMULATOR_EMPTY_RESPONSE_RETRIES = 3
SIMULATOR_STEP_PROTOCOL = (
    "\n\n## Simulation-step response protocol\n"
    "Each invocation is a new simulation step, including invocations after tool results. "
    "Return either visible text or at least one tool call on every invocation, never an empty response. "
    "If the interaction is finished, use the termination marker already specified in these guidelines.\n"
)
MODEL_DICT: dict[str, Any] = {"agent": None, "user": None}
_CAPTURE_PATH: Path | None = None
_CURRENT_TASK_ID: ContextVar[str | None] = ContextVar("tau3_task_id", default=None)


def configure_capture(path: str | os.PathLike[str] | None) -> None:
    """Enable opt-in JSONL capture at a unique path."""

    global _CAPTURE_PATH
    capture_path = Path(path).expanduser() if path else None
    if capture_path is not None:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with capture_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError as error:
            raise ValueError(f"capture path already exists; choose a unique path: {capture_path}") from error
    _CAPTURE_PATH = capture_path


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

def _capture(record: dict[str, Any]) -> None:
    if _CAPTURE_PATH is None:
        return
    if _CURRENT_TASK_ID.get() is not None and "task_id" not in record:
        record = {**record, "task_id": _CURRENT_TASK_ID.get()}
    with _CAPTURE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")




def _reasoning_from_tau_message(message: Message) -> str | None:
    raw = message.raw_data or {}
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list) and choices:
        raw_message = choices[0].get("message") if isinstance(choices[0], dict) else None
        raw_content = raw_message.get("content") if isinstance(raw_message, dict) else None
        if isinstance(raw_content, list):
            parts = [
                part.get("reasoning")
                for part in raw_content
                if isinstance(part, dict)
                and part.get("type") == "reasoning"
                and part.get("reasoning")
            ]
            if parts:
                return str(parts[-1])
        if isinstance(raw_message, dict) and isinstance(raw_message.get("reasoning"), str):
            return raw_message["reasoning"]
    return None


def _trajectory_message(source: Message) -> Any:
    """Build a report artifact without changing the participant's role.

    EvalScope has no user-role tool-call field. Preserve those actions in
    metadata, using an empty content list only for a tool-only report message.
    This representation is never used as model input; the full Tau2 result
    remains the authoritative trajectory.
    """
    payload = tau_llm_utils.to_litellm_messages([source])[0]
    if isinstance(source, UserMessage) and source.tool_calls:
        payload["metadata"] = {
            "tau2_user_tool_calls": [call.model_dump(mode="json") for call in source.tool_calls]
        }
        if payload.get("content") is None:
            payload["content"] = []
    if isinstance(source, AssistantMessage):
        reasoning = _reasoning_from_tau_message(source)
        if reasoning:
            payload["reasoning"] = reasoning
        if payload.get("content") is None and payload.get("tool_calls"):
            payload["content"] = ""
    return dict_to_chat_message(payload)

def _tau_messages_for_model(messages: list[Message]) -> list[Any]:
    """Use Tau2's converter, adding EvalScope's explicit reasoning field.

    ``dict_to_chat_message`` converts ``reasoning`` into ``ContentReasoning``;
    EvalScope's OpenAI serializer then emits it as ``reasoning_content`` according
    to ``GenerateConfig.reasoning_history``.  Empty assistant/user responses are
    rejected rather than turned into fabricated text.
    """

    base_messages = tau_llm_utils.to_litellm_messages(messages)
    converted: list[Any] = []
    for source, payload in zip(messages, base_messages):
        has_calls = bool(payload.get("tool_calls"))
        content = payload.get("content")
        if source.role in {"assistant", "user"} and not has_calls and not (isinstance(content, str) and content.strip()):
            raise Tau3AdapterError(
                "model_output_invalid",
                f"empty {source.role} message has neither text nor tool calls",
            )
        if source.role == "assistant":
            reasoning = _reasoning_from_tau_message(source)
            if reasoning:
                payload["reasoning"] = reasoning
            if content is None and has_calls:
                payload["content"] = ""
        converted.append(dict_to_chat_message(payload))
    return converted

def _request_record(model: str, messages: list[Message], tools: list[Tool] | None, tool_choice: Any) -> dict[str, Any]:
    payloads = tau_llm_utils.to_litellm_messages(messages)
    return {
        "model": model,
        "tau_messages": payloads,
        "tau_tools": [tool.openai_schema for tool in tools] if tools else None,
        "tool_choice": tool_choice,
    }
def _is_typed_model_output_error(exc: Exception) -> bool:
    """Recognize only the server's typed non-retryable output-contract error."""

    if not isinstance(exc, APIStatusError) or getattr(exc, "status_code", None) != 422:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    error = body.get("error", body)
    return (
        isinstance(error, dict)
        and error.get("type") == "model_output_error"
        and error.get("code") == "invalid_model_output"
    )


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, Tau3AdapterError):
        return exc.kind
    if _is_typed_model_output_error(exc):
        return "model_output_invalid"
    return "infrastructure"


def patched_generate(
    model: str,
    messages: list[Message],
    tools: list[Tool] | None = None,
    tool_choice: Any = None,
    **kwargs: Any,
) -> AssistantMessage:
    """Generate one Tau2 response while preserving reasoning and raw evidence."""

    oa_model = MODEL_DICT.get(model) or MODEL_DICT.get("user")
    if oa_model is None:
        raise Tau3AdapterError("adapter_error", f"model {model!r} is not configured")
    request = _request_record(model, messages, tools, tool_choice)
    request["call_name"] = kwargs.get("call_name")
    simulator = model == "user" and request["call_name"] == "user_simulator_response"
    try:
        model_input = _tau_messages_for_model(messages)
        if simulator:
            system = next((message for message in model_input if message.role == "system"), None)
            if system is None:
                raise Tau3AdapterError("adapter_error", "user simulator requires its system guidelines")
            system.content = system.text + SIMULATOR_STEP_PROTOCOL
        request["wire_messages"] = [
            openai_chat_message(message, reasoning_format="reasoning_field")
            for message in model_input
        ]
        generation_tools = [ToolInfo.model_validate(tool.openai_schema["function"]) for tool in tools] if tools else None
        request["wire_tools"] = openai_chat_tools(generation_tools) if generation_tools else None
    except Exception as exc:
        kind = _classify_exception(exc)
        _capture({"event": "error", "kind": kind, "error_type": type(exc).__name__, "error": str(exc), **request})
        raise Tau3AdapterError(kind, str(exc)) from exc
    generation_tool_choice = tool_choice if tool_choice is not None else ("auto" if tools else None)
    attempts = 1 + SIMULATOR_EMPTY_RESPONSE_RETRIES if simulator else 1
    for attempt in range(1, attempts + 1):
        attempt_record = {**request, "attempt": attempt}
        _capture({"event": "request", **attempt_record})
        try:
            completion = oa_model.generate(
                input=model_input,
                tools=generation_tools,
                tool_choice=generation_tool_choice,
            )
        except Exception as exc:
            kind = _classify_exception(exc)
            _capture({"event": "error", "kind": kind, "error_type": type(exc).__name__, "error": str(exc), **attempt_record})
            raise Tau3AdapterError(kind, str(exc)) from exc

        _capture({"event": "response", "response": completion.model_dump(mode="json", exclude_none=False), **attempt_record})
        if completion.error:
            _capture({"event": "error", "kind": "model_output_invalid", "error": "model completion reported an error", **attempt_record})
            raise Tau3AdapterError("model_output_invalid", "model completion reported an error")
        if not completion.choices:
            _capture({"event": "error", "kind": "model_output_invalid", "error": "model returned no choices", **attempt_record})
            raise Tau3AdapterError("model_output_invalid", "model returned no choices")
        choice = completion.choices[0]
        message = choice.message
        if (
            attempt < attempts
            and choice.stop_reason == "stop"
            and not completion.error
            and not message.text.strip()
            and not message.tool_calls
        ):
            _capture({
                "event": "protocol_retry",
                "kind": "simulator_empty_response",
                "reason": "clean stop without visible text or tool calls",
                "next_attempt": attempt + 1,
                **attempt_record,
            })
            continue
        break
    tool_calls = []
    for tool_call in message.tool_calls or []:
        arguments = tool_call.function.arguments
        if not isinstance(arguments, dict):
            _capture({"event": "error", "kind": "model_output_invalid", "error": "EvalScope tool arguments were not an object", **attempt_record})
            raise Tau3AdapterError("model_output_invalid", "EvalScope tool arguments were not an object")
        tool_calls.append(ToolCall(id=tool_call.id, name=tool_call.function.name, arguments=arguments))
    content = message.text
    if not content.strip() and not tool_calls:
        _capture({"event": "error", "kind": "model_output_invalid", "error": "model response had no text or tool calls", **attempt_record})
        raise Tau3AdapterError("model_output_invalid", "model response had no text or tool calls")
    raw_data = completion.model_dump(mode="json", exclude_none=False)
    perf = message.perf_metrics
    if perf is not None:
        raw_data["_perf_metrics"] = perf.model_dump(mode="json")
    usage = completion.usage.model_dump(exclude_none=True) if completion.usage is not None else None
    return AssistantMessage(
        role="assistant",
        content=content or None,
        tool_calls=tool_calls or None,
        cost=None,
        usage=usage,
        raw_data=raw_data,
    )


def _build_model(agent_model: Any, adapter_instance: Any) -> None:
    from evalscope.api.model import GenerateConfig, get_model

    if adapter_instance.extra_params.get("simulator_empty_response_retries", SIMULATOR_EMPTY_RESPONSE_RETRIES) != SIMULATOR_EMPTY_RESPONSE_RETRIES:
        raise Tau3AdapterError("adapter_error", "simulator_empty_response_retries must match the fixed three-retry policy")
    if adapter_instance.extra_params.get("simulator_step_protocol", SIMULATOR_STEP_PROTOCOL) != SIMULATOR_STEP_PROTOCOL:
        raise Tau3AdapterError("adapter_error", "simulator_step_protocol must match the fixed step clarification")

    user_server = get_model(
        model=adapter_instance.user_model,
        eval_type=EvalType.OPENAI_API,
        base_url=adapter_instance.api_base,
        api_key=adapter_instance.api_key,
        config=GenerateConfig(**adapter_instance.generation_config),
        model_args={"max_retries": 5},
    )
    MODEL_DICT["user"] = user_server
    MODEL_DICT["agent"] = agent_model
    original = getattr(tau_llm_utils, "generate", None)
    if original is None:
        raise Tau3AdapterError("adapter_error", "tau2.utils.llm_utils.generate not found")
    tau_llm_utils.generate = patched_generate
    for name, module in list(sys.modules.items()):
        if not isinstance(name, str) or not name.startswith("tau2") or module is None:
            continue
        for attr, value in list(vars(module).items()):
            if value is original:
                try:
                    setattr(module, attr, patched_generate)
                except Exception:
                    pass


def predict(model: Any, sample: Any, adapter_instance: Any) -> InferenceResult:
    """Run Tau2 without converting exceptions into unlabelled reward zeroes."""

    domain = sample.subset_key
    task_data = {key: value for key, value in sample.metadata.items() if key != "_domain"}
    task = Task.model_validate(task_data)
    _CURRENT_TASK_ID.set(task.id)
    try:
        _build_model(model, adapter_instance)
    except Exception as exc:
        kind = _classify_exception(exc)
        _capture({"event": "task_error", "kind": kind, "error_type": type(exc).__name__, "error": str(exc)})
        if isinstance(exc, Tau3AdapterError):
            raise
        raise Tau3AdapterError(kind, str(exc)) from exc
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import run_task

    run_kwargs: dict[str, Any] = {
        "domain": domain,
        "task": task,
        "agent": "llm_agent",
        "user": "user_simulator",
        "llm_agent": "agent",
        "llm_user": "user",
        "evaluation_type": EvaluationType.ALL_WITH_NL_ASSERTIONS,
    }
    if domain == "banking_knowledge":
        run_kwargs["retrieval_config"] = adapter_instance.retrieval_config
        run_kwargs["retrieval_config_kwargs"] = adapter_instance.retrieval_config_kwargs
    try:
        result = run_task(**run_kwargs)
    except Tau3AdapterError:
        raise
    except Exception as exc:
        kind = _classify_exception(exc)
        _capture({"event": "task_error", "kind": kind, "error_type": type(exc).__name__, "error": str(exc), "task_id": task.id})
        raise Tau3AdapterError(kind, str(exc)) from exc

    # Preserve completed simulation and judge evidence before optional report
    # projection, so a report-schema failure cannot discard the finished task.
    _capture({"event": "task_result", "result": result.model_dump(mode="json"), "task_id": task.id})

    task_result = result.reward_info.model_dump()
    task_result["status"] = "completed"
    sample.metadata["task_result"] = task_result
    raw_messages = result.messages or []
    agent_messages = []
    for raw in raw_messages:
        try:
            agent_messages.append(_trajectory_message(raw))
        except Exception as exc:
            _capture({"event": "trajectory_error", "kind": "adapter_error", "error": str(exc), "task_id": task.id})
            raise Tau3AdapterError("adapter_error", str(exc)) from exc

    output = ModelOutput(
        model=model.name,
        choices=[ChatCompletionChoice.from_content(result.model_dump_json(indent=2))],
    )
    return InferenceResult(output=output, messages=agent_messages or None)


def install() -> None:
    """Install the repo-owned hooks into the installed τ³ adapter."""

    from evalscope.benchmarks.tau_bench.tau3_bench import generation
    from evalscope.api.registry import BENCHMARK_REGISTRY
    from evalscope.benchmarks.tau_bench.tau3_bench import tau3_bench_adapter  # Register metadata before extending it.

    BENCHMARK_REGISTRY.get("tau3_bench").extra_params["simulator_empty_response_retries"] = {
        "type": "int",
        "description": "Fixed additional clean-stop empty user-simulator attempts; never agent or judge retries.",
        "value": SIMULATOR_EMPTY_RESPONSE_RETRIES,
    }
    BENCHMARK_REGISTRY.get("tau3_bench").extra_params["simulator_step_protocol"] = {
        "type": "str",
        "description": "Exact simulator-only clarification of Tau2's nonempty step contract.",
        "value": SIMULATOR_STEP_PROTOCOL,
    }

    generation.predict = predict
    generation.patched_generate = patched_generate


def _validate_cached_records(cache, dataset, model_name: str) -> tuple[int, int]:
    """Validate native cache rows without changing or synthesizing any record."""
    from evalscope.api.evaluator.cache import ModelResult, ReviewResult
    from evalscope.utils.io_utils import jsonl_to_list
    from tau2.data_model.simulation import SimulationRun

    fields = {"role", "content", "tool_calls", "tool_call_id", "metadata"}
    predictions = {}
    prediction_path = cache.get_prediction_cache_path("banking_knowledge")
    for row in jsonl_to_list(prediction_path) if Path(prediction_path).is_file() else []:
        prediction = ModelResult.model_validate(row)
        index = prediction.index
        if type(row["index"]) is not int or not 0 <= index < len(dataset) or index in predictions:
            raise ValueError("resume prediction has a duplicate or invalid dataset index")
        sample = dataset[index]
        metadata = prediction.metadata or {}
        if sample.id != index or {key: value for key, value in metadata.items() if key != "task_result"} != sample.metadata:
            raise ValueError(f"resume prediction {index} does not match the current dataset task/order")
        output = prediction.model_output
        if prediction.model != model_name or output is None or output.model != model_name or output.error or len(output.choices) != 1:
            raise ValueError(f"resume prediction {index} is not a successful model report")
        if output.choices[0].stop_reason != "stop":
            raise ValueError(f"resume prediction {index} is not the canonical report wrapper")
        simulation = SimulationRun.model_validate_json(output.choices[0].message.text)
        if simulation.task_id != sample.metadata["id"] or simulation.reward_info is None:
            raise ValueError(f"resume prediction {index} lacks its matching canonical task result")
        reward = simulation.reward_info.reward
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError(f"resume prediction {index} has an invalid canonical reward")
        expected_result = {**simulation.reward_info.model_dump(mode="json"), "status": "completed"}
        if metadata.get("task_result") != expected_result:
            raise ValueError(f"resume prediction {index} lacks verified completed-result metadata")
        raw_messages = simulation.messages or []
        if len(prediction.messages) != len(raw_messages) or any(
            saved.model_dump(include=fields, exclude_none=True) != _trajectory_message(raw).model_dump(include=fields, exclude_none=True)
            for saved, raw in zip(prediction.messages, raw_messages)
        ):
            raise ValueError(f"resume prediction {index} has a lossy or mismatched trajectory")
        predictions[index] = (prediction.to_task_state(dataset), simulation)

    reviewed = set()
    review_path = cache.get_review_cache_path("banking_knowledge")
    for row in jsonl_to_list(review_path) if Path(review_path).is_file() else []:
        review = ReviewResult.model_validate(row)
        index = review.index
        if type(row["index"]) is not int or index in reviewed or index not in predictions:
            raise ValueError("resume review has a duplicate, invalid, or orphan dataset index")
        state, simulation = predictions[index]
        sample_score = review.sample_score
        score = sample_score.score
        scored_simulation = SimulationRun.model_validate_json(score.prediction)
        if (
            review.target != state.target
            or review.agent_trace != state.agent_trace
            or [message.model_dump(include=fields, exclude_none=True) for message in review.messages]
            != [message.model_dump(include=fields, exclude_none=True) for message in state.messages]
            or sample_score.sample_id != state.sample_id
            or sample_score.group_id != state.group_id
            or sample_score.generation_index != 0
            or sample_score.sample_metadata != state.metadata
            or score.status != "success"
            or score.value != {"acc": simulation.reward_info.reward}
            or score.metadata != {"task_result": state.metadata["task_result"]}
            or scored_simulation.task_id != simulation.task_id
            or scored_simulation.termination_reason != simulation.termination_reason
            or scored_simulation.reward_info != simulation.reward_info
            or [_trajectory_message(message).model_dump(include=fields, exclude_none=True) for message in scored_simulation.messages or []]
            != [message.model_dump(include=fields, exclude_none=True) for message in state.messages]
        ):
            raise ValueError(f"resume review {index} does not match its canonical scored result")
        reviewed.add(index)
    return len(predictions), len(reviewed)


__all__ = ["Tau3AdapterError", "configure_capture", "install"]
