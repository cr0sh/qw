import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evalscope.api.messages import ContentReasoning, ContentText, ChatMessageAssistant
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelOutput, get_model
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_message
from tau2.data_model.message import AssistantMessage, UserMessage

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


class FakeModel:
    def __init__(self, output):
        self.output = output

    def generate(self, **_kwargs):
        return self.output


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
            tau3_adapter._tau_messages_for_model([UserMessage(role="user", content=None)])
        self.assertEqual(ctx.exception.kind, "model_output_invalid")
        self.assertIn("empty user", str(ctx.exception))

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

    def test_response_schema_keeps_reasoning_in_raw_model_output(self):
        output = ModelOutput(
            model="user",
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[ContentReasoning(reasoning="thinking"), ContentText(text="done")]
                ),
                stop_reason="stop",
            )],
        )
        tau3_adapter.MODEL_DICT["user"] = FakeModel(output)
        result = tau3_adapter.patched_generate(
            model="user",
            messages=[UserMessage(role="user", content="hello")],
        )
        self.assertEqual(result.content, "done")
        raw_content = result.raw_data["choices"][0]["message"]["content"]
        self.assertEqual(raw_content[0]["type"], "reasoning")
        self.assertEqual(raw_content[0]["reasoning"], "thinking")


if __name__ == "__main__":
    unittest.main()
