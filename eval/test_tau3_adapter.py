import json
from pathlib import Path
import tempfile
import unittest

from evalscope.api.messages import ContentReasoning, ContentText, ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import ChatCompletionChoice, ModelOutput
from evalscope.models.utils.openai import openai_chat_message
from tau2.data_model.message import AssistantMessage, UserMessage

from eval import tau3_adapter


class FakeModel:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
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
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "reasoning", "reasoning": "prior thought"},
                                {"type": "text", "text": "answer"},
                            ]
                        }
                    }
                ]
            },
        )
        converted = tau3_adapter._tau_messages_for_model([message])
        self.assertIsInstance(converted[0], ChatMessageAssistant)
        self.assertEqual(converted[0].text, "answer")
        self.assertEqual(converted[0].content[0].type, "reasoning")
        self.assertEqual(converted[0].content[0].reasoning, "prior thought")
        wire = openai_chat_message(converted[0], reasoning_format="reasoning_field")
        self.assertEqual(wire["content"].strip(), "answer")
        self.assertEqual(wire["reasoning_content"], "prior thought")

    def test_empty_user_response_is_rejected_without_placeholder(self):
        with self.assertRaises(tau3_adapter.Tau3AdapterError) as ctx:
            tau3_adapter._tau_messages_for_model([UserMessage(role="user", content=None)])
        self.assertEqual(ctx.exception.kind, "model_output_invalid")
        self.assertIn("empty user", str(ctx.exception))

    def test_invalid_model_output_capture_has_one_request_and_one_error(self):
        fake = FakeModel(error=RuntimeError("HTTP 422 model_output_error invalid_model_output"))
        tau3_adapter.MODEL_DICT["user"] = fake
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.jsonl"
            tau3_adapter.configure_capture(capture)
            with self.assertRaises(tau3_adapter.Tau3AdapterError) as ctx:
                tau3_adapter.patched_generate(
                    model="user",
                    messages=[UserMessage(role="user", content="hello")],
                )
            self.assertEqual(ctx.exception.kind, "model_output_invalid")
            self.assertEqual(fake.calls, 1)
            rows = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["request", "error"])
            self.assertEqual(rows[1]["kind"], "model_output_invalid")
            self.assertNotIn("api_key", json.dumps(rows))

    def test_response_schema_keeps_reasoning_in_raw_model_output(self):
        output = ModelOutput(
            model="user",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=[ContentReasoning(reasoning="thinking"), ContentText(text="done")]
                    ),
                    stop_reason="stop",
                )
            ],
        )
        fake = FakeModel(output=output)
        tau3_adapter.MODEL_DICT["user"] = fake
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
