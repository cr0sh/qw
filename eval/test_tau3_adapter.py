import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from evalscope.api.dataset.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput, get_model
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_message
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage, ToolCall
from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.user.user_simulator import UserSimulator, UserState

from eval import tau3_adapter


class Typed422Handler(BaseHTTPRequestHandler):
    requests = 0

    def do_POST(self):
        type(self).requests += 1
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        body = json.dumps({
            "error": {
                "message": "generated tool-call output was invalid",
                "type": "model_output_error",
                "code": "invalid_model_output",
            }
        }).encode()
        self.send_response(422)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass

def chat_response(content=None, *, finish_reason="stop", tool_calls=None, refusal=None):
    return {"choices": [{
        "index": 0, "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls, "refusal": refusal},
    }]}


@contextmanager
def completion_server(responses):
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payloads.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            body = json.dumps({
                "id": f"reply-{len(payloads)}", "object": "chat.completion",
                "created": 0, "model": "simulator-protocol",
                **responses[min(len(payloads) - 1, len(responses) - 1)],
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = get_model(
        model="simulator-protocol", eval_type=EvalType.OPENAI_API,
        base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test",
        config=GenerateConfig(retries=1, temperature=0, max_tokens=128, reasoning_effort="low"),
        model_args={"max_retries": 0},
    )
    try:
        yield model, payloads
    finally:
        model.api.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def check_email(folder: str) -> str:
    """Read an email folder."""
    raise AssertionError("Generation must not execute simulator tools")




class Tau3AdapterTests(unittest.TestCase):
    def tearDown(self):
        tau3_adapter.MODEL_DICT["user"] = None
        tau3_adapter.MODEL_DICT["agent"] = None
        tau3_adapter.configure_capture(None)

    def test_reasoning_round_trip_uses_evalscope_reasoning_field(self):
        message = AssistantMessage(
            role="assistant",
            content="answer",
            raw_data={
                "choices": [{"message": {"content": [
                    {"type": "reasoning", "reasoning": "prior thought"},
                    {"type": "text", "text": "answer"},
                ]}}],
            },
        )
        converted = tau3_adapter._tau_messages_for_model([message])
        self.assertIsInstance(converted[0], ChatMessageAssistant)
        self.assertEqual(converted[0].text, "answer")
        self.assertEqual(converted[0].content[0].reasoning, "prior thought")
        wire = openai_chat_message(converted[0], reasoning_format="reasoning_field")
        self.assertEqual(wire["content"].strip(), "answer")
        self.assertEqual(wire["reasoning_content"], "prior thought")

    def test_empty_user_response_is_rejected_without_placeholder(self):
        with self.assertRaises(tau3_adapter.Tau3AdapterError) as ctx:
            tau3_adapter._tau_messages_for_model([UserMessage(role="user", content=" ")])
        self.assertEqual(ctx.exception.kind, "model_output_invalid")

    def test_completed_prediction_preserves_user_role_and_action_losslessly(self):
        action = ToolCall(
            id="user-action", name="check_email",
            arguments={"folder": "inbox"}, requestor="user",
        )
        source = UserMessage(role="user", content=None, tool_calls=[action])
        task = Task(id="report-action", user_scenario={"instructions": "Check the inbox."})
        completed = SimulationRun(
            id="report-run", task_id=task.id,
            start_time="2026-01-01T00:00:00", end_time="2026-01-01T00:00:01",
            duration=1.0, termination_reason="user_stop",
            reward_info=RewardInfo(reward=1.0), messages=[source],
        )
        sample = Sample(input="Check the inbox.", subset_key="mock", metadata=task.model_dump())
        with patch.object(tau3_adapter, "_build_model"), patch("tau2.run.run_task", return_value=completed):
            prediction = tau3_adapter.predict(SimpleNamespace(name="report-model"), sample, None)
        restored_result = SimulationRun.model_validate_json(prediction.output.choices[0].message.text)
        self.assertEqual(restored_result.termination_reason, completed.termination_reason)
        self.assertEqual(restored_result.messages[0].tool_calls, [action])
        self.assertEqual(sample.metadata["task_result"]["status"], "completed")
        restored = ChatMessageUser.model_validate_json(prediction.messages[0].model_dump_json())
        self.assertEqual(restored.role, "user")
        self.assertEqual(restored.content, [])
        self.assertEqual(
            restored.metadata["tau2_user_tool_calls"],
            [action.model_dump(mode="json")],
        )
        self.assertIsNone(source.content)
        self.assertEqual(source.tool_calls, [action])
        # A reporting representation must never silently rewrite model input
        # into an assistant action. Tau2 owns any perspective conversion.
        with self.assertRaises(tau3_adapter.Tau3AdapterError):
            tau3_adapter._tau_messages_for_model([source])

    def test_simulator_empty_recovery_preserves_payload_and_advances_state_once(self):
        responses = [chat_response(None), chat_response(" \n"), chat_response(""), chat_response("###STOP###")]
        with completion_server(responses) as (model, payloads), tempfile.TemporaryDirectory() as directory:
            tau3_adapter.MODEL_DICT["user"] = model
            capture = Path(directory) / "attempts.jsonl"
            tau3_adapter.configure_capture(capture)
            simulator = UserSimulator(llm="user", tools=[Tool(check_email)])
            state = UserState(system_messages=[SystemMessage(role="system", content="Act as the customer.")], messages=[])
            incoming = AssistantMessage(role="assistant", content="Your request is complete.")
            with patch("tau2.user.user_simulator.generate", tau3_adapter.patched_generate):
                answer, state = simulator.generate_next_message(incoming, state)
            self.assertEqual(answer.content, "###STOP###")
            self.assertEqual(state.messages, [incoming, answer])
            self.assertEqual(payloads, [payloads[0]] * 4)
            self.assertEqual(state.system_messages[0].content, "Act as the customer.")
            self.assertEqual(payloads[0]["messages"][0]["content"], state.system_messages[0].content + tau3_adapter.SIMULATOR_STEP_PROTOCOL)
            self.assertEqual(payloads[0]["tools"][0]["function"]["name"], "check_email")
            rows = [json.loads(line) for line in capture.read_text().splitlines()]
            for row, payload in zip([row for row in rows if row["event"] == "request"], payloads):
                self.assertEqual(row["wire_messages"], payload["messages"])
                self.assertEqual(row["wire_tools"], payload["tools"])
                self.assertNotIn("tools", row)
            self.assertEqual([row["attempt"] for row in rows if row["event"] == "request"], [1, 2, 3, 4])
            self.assertEqual([row["next_attempt"] for row in rows if row["event"] == "protocol_retry"], [2, 3, 4])

    def test_step_clarification_preserves_task_and_history_and_excludes_agent_judge(self):
        messages = [
            SystemMessage(role="system", content="Goal: compare cards; do not submit an application."),
            UserMessage(role="user", content="Which card has no annual fee?"),
        ]
        original = [message.model_dump() for message in messages]
        for role, call_name in [("user", "user_simulator_response"), ("agent", "agent_response"), ("user", "judge")]:
            with self.subTest(role=role, call_name=call_name), completion_server([chat_response("Continue comparing.")]) as (model, payloads):
                tau3_adapter.MODEL_DICT[role] = model
                tau3_adapter.patched_generate(model=role, messages=messages, call_name=call_name)
                expected_system = messages[0].content
                if call_name == "user_simulator_response":
                    expected_system += tau3_adapter.SIMULATOR_STEP_PROTOCOL
                self.assertEqual(payloads[0]["messages"], [
                    {"role": "system", "content": expected_system},
                    {"role": "user", "content": messages[1].content},
                ])
                self.assertEqual([message.model_dump() for message in messages], original)

    def test_empty_recovery_is_bounded_and_never_retries_agent_judge_or_nonstop(self):
        cases = [
            ("user", "user_simulator_response", chat_response(), 4),
            ("agent", "user_simulator_response", chat_response(), 1),
            ("user", "judge", chat_response(), 1),
            ("user", None, chat_response(), 1),
            ("user", "user_simulator_response", chat_response(finish_reason="length"), 1),
            ("user", "user_simulator_response", {"choices": []}, 1),
        ]
        for role, call_name, response, requests in cases:
            with self.subTest(role=role, call_name=call_name, response=response):
                with completion_server([response]) as (model, payloads):
                    tau3_adapter.MODEL_DICT[role] = model
                    with self.assertRaises(tau3_adapter.Tau3AdapterError):
                        tau3_adapter.patched_generate(
                            model=role, call_name=call_name,
                            messages=[SystemMessage(role="system", content="Act as the customer."), UserMessage(role="user", content="Continue.")],
                        )
                    self.assertEqual(len(payloads), requests)

    def test_visible_truncated_agent_response_is_preserved_without_retry(self):
        with completion_server([chat_response("Partial visible answer", finish_reason="length")]) as (model, payloads):
            tau3_adapter.MODEL_DICT["agent"] = model
            answer = tau3_adapter.patched_generate(
                model="agent", call_name="agent_response",
                messages=[UserMessage(role="user", content="Explain the card options.")],
            )
            self.assertEqual(answer.content, "Partial visible answer")
            self.assertEqual(len(payloads), 1)

    def test_visible_response_refusal_and_tool_action_are_not_resampled(self):
        action = {"id": "action-1", "type": "function", "function": {"name": "check_email", "arguments": '{"folder":"spam"}'}}
        cases = [
            (chat_response("A possibly wrong answer."), "A possibly wrong answer.", False),
            (chat_response(refusal="I cannot help."), "I cannot help.", False),
            (chat_response(tool_calls=[action], finish_reason="tool_calls"), None, True),
        ]
        for response, content, has_tool in cases:
            with self.subTest(response=response), completion_server([response]) as (model, payloads):
                tau3_adapter.MODEL_DICT["user"] = model
                answer = tau3_adapter.patched_generate(
                    model="user", call_name="user_simulator_response",
                    messages=[SystemMessage(role="system", content="Act as the customer."), UserMessage(role="user", content="Continue.")], tools=[Tool(check_email)],
                )
                self.assertEqual(answer.content, content)
                self.assertEqual(bool(answer.tool_calls), has_tool)
                if has_tool:
                    self.assertEqual(answer.tool_calls[0].arguments, {"folder": "spam"})
                self.assertEqual(len(payloads), 1)

    def test_completion_error_with_visible_content_aborts_without_retry(self):
        completion = ModelOutput.from_content(model="error-diagnostic", content="Visible but failed.", error="upstream failure")
        with patch.object(tau3_adapter, "MODEL_DICT", {"user": SimpleNamespace(generate=lambda **kwargs: completion)}):
            with patch.object(tau3_adapter, "_capture") as capture:
                with self.assertRaises(tau3_adapter.Tau3AdapterError):
                    tau3_adapter.patched_generate(
                        model="user", call_name="user_simulator_response",
                        messages=[SystemMessage(role="system", content="Act as the customer."), UserMessage(role="user", content="Continue.")],
                    )
                self.assertEqual([call.args[0]["event"] for call in capture.call_args_list], ["request", "response", "error"])

    def test_typed_422_stops_sdk_retries_after_one_real_http_request(self):
        Typed422Handler.requests = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), Typed422Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = get_model(
                model="typed422-diagnostic",
                eval_type=EvalType.OPENAI_API,
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="diagnostic-only",
                config=GenerateConfig(retries=1, max_tokens=8),
                model_args={"max_retries": 5},
            )
            tau3_adapter.MODEL_DICT["user"] = model
            with tempfile.TemporaryDirectory() as directory:
                capture = Path(directory) / "capture.jsonl"
                tau3_adapter.configure_capture(capture)
                with self.assertRaises(tau3_adapter.Tau3AdapterError) as ctx:
                    tau3_adapter.patched_generate(
                        model="user",
                        call_name="user_simulator_response",
                        messages=[SystemMessage(role="system", content="Act as the customer."), UserMessage(role="user", content="hello")],
                    )
                self.assertEqual(ctx.exception.kind, "model_output_invalid")
                rows = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual([row["event"] for row in rows], ["request", "error"])
                self.assertEqual(rows[1]["kind"], "model_output_invalid")
            self.assertEqual(Typed422Handler.requests, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)



if __name__ == "__main__":
    unittest.main()
