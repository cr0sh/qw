"""Security and reasoning-history boundaries for the native evaluator integration."""

import os
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

from eval.run_tau3_banking import _load_simulator_credentials


class SimulatorCredentialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "eval").mkdir()
        self.credential_file = root / "eval" / ".env.local"
        self.credential_file.write_text("DEEPSEEK_API_KEY='synthetic-${SECRET_SUFFIX}'\n")
        self.credential_file.chmod(0o600)
        git = patch(
            "eval.run_tau3_banking.subprocess.run",
            return_value=CompletedProcess([], 0, stdout=str(root / ".git") + "\n"),
        )
        git.start()
        self.addCleanup(git.stop)
        environment = patch.dict(os.environ, {"SECRET_SUFFIX": "must-not-expand"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_private_credential_is_literal_and_available_to_native_environment_fallback(self):
        _load_simulator_credentials()
        self.assertEqual(os.environ["OPENAI_API_KEY"], "synthetic-${SECRET_SUFFIX}")
        self.assertEqual(os.environ["EVALSCOPE_API_KEY"], "synthetic-${SECRET_SUFFIX}")

    def test_public_or_symlinked_credentials_are_rejected_without_loading(self):
        self.credential_file.chmod(0o644)
        with self.assertRaises(SystemExit):
            _load_simulator_credentials()
        self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertNotIn("EVALSCOPE_API_KEY", os.environ)

        private_target = self.credential_file.with_name("private-key")
        self.credential_file.rename(private_target)
        private_target.chmod(0o600)
        self.credential_file.symlink_to(private_target)
        with self.assertRaises(SystemExit):
            _load_simulator_credentials()
        self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertNotIn("EVALSCOPE_API_KEY", os.environ)

    def test_missing_or_empty_credentials_cannot_use_an_inherited_key(self):
        os.environ["OPENAI_API_KEY"] = "unrelated-inherited-key"
        for contents in ("", "DEEPSEEK_API_KEY=\n"):
            with self.subTest(contents=contents):
                self.credential_file.write_text(contents)
                with self.assertRaises(SystemExit):
                    _load_simulator_credentials()
        self.credential_file.unlink()
        with self.assertRaises(SystemExit):
            _load_simulator_credentials()


class NativeReasoningRoundTripTests(unittest.TestCase):
    def test_private_reasoning_survives_tau_bridge_and_next_tool_turn(self):
        import json
        from evalscope.api.messages.chat_message import dict_to_chat_message
        from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
        from evalscope.models.utils.openai import openai_chat_choices, openai_chat_message
        from tau2.data_model.message import AssistantMessage, ToolCall
        from tau2.utils.llm_utils import to_litellm_messages

        reasoning = "\nReasoning retained exactly across a tool turn.\n"
        message = dict_to_chat_message({
            "role": "assistant",
            "content": None,
            "reasoning": reasoning,
            "tool_calls": [{
                "id": "call-search",
                "type": "function",
                "function": {"name": "KB_search", "arguments": {"query": "accounts"}},
            }],
        })
        choice = ChatCompletionChoice(message=message, stop_reason="tool_calls")
        completion = ModelOutput(model="test", choices=[choice])
        visible = openai_chat_choices(completion.choices, include_reasoning=False)[0].message
        native = AssistantMessage(
            role="assistant",
            content=visible.content,
            tool_calls=[
                ToolCall(id=call.id, name=call.function.name,
                         arguments=json.loads(call.function.arguments))
                for call in visible.tool_calls
            ],
            raw_data=completion.model_dump(),
        )
        history = to_litellm_messages([native])
        wire = openai_chat_message(
            dict_to_chat_message(history[0]), reasoning_format="reasoning_field"
        )
        self.assertEqual(wire["reasoning_content"], reasoning)
        self.assertNotIn(reasoning, native.content or "")
        self.assertNotIn(reasoning, wire.get("content") or "")
        self.assertEqual(wire["tool_calls"][0]["id"], "call-search")
        self.assertEqual(
            json.loads(wire["tool_calls"][0]["function"]["arguments"]),
            {"query": "accounts"},
        )


if __name__ == "__main__":
    unittest.main()
