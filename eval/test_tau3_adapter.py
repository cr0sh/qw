import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from evalscope.api.dataset.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import GenerateConfig, get_model
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_message
from tau2.data_model.message import AssistantMessage, UserMessage, ToolCall
from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.data_model.tasks import Task

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
        self.assertIn("empty user", str(ctx.exception))

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
                        messages=[UserMessage(role="user", content="hello")],
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
