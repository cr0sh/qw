import copy
import json
import io
import logging
import os
import multiprocessing
from pathlib import Path
import tempfile
import shutil
import signal
import threading
import traceback
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from evalscope.api.dataset.dataset import MemoryDataset, Sample
from evalscope.api.evaluator.cache import CacheManager
from evalscope.api.evaluator.state import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput, get_model
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_message
from evalscope.utils.io_utils import OutputsStructure
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


def cache_fixture_dataset(count=2):
    samples = []
    for index in range(count):
        task = Task(
            id="cached-zero" if index == 0 else ("pending" if index == 1 else f"pending-{index}"),
            user_scenario={"instructions": "Compare available cards." if index == 0 else "Pending task"},
        )
        samples.append(Sample(
            id=index, group_id=index, input="Compare available cards." if index == 0 else "Pending task",
            subset_key="banking_knowledge", metadata=task.model_dump(),
        ))
    return MemoryDataset(samples)


def cache_fixture_record(dataset, index=0):
    sample = dataset[index]
    message = AssistantMessage(
        role="assistant", content="I still need details.",
        raw_data={"choices": [{"message": {"content": [
            {"type": "reasoning", "reasoning": "Need more information."},
            {"type": "text", "text": "I still need details."},
        ]}}]},
    )
    user_tool = UserMessage(role="user", tool_calls=[ToolCall(id="lookup", name="lookup", arguments={}, requestor="user")])
    result = SimulationRun(
        id="cache-fixture" if index == 0 else f"cache-fixture-{index}", task_id=sample.metadata["id"], start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01", duration=1.0,
        termination_reason="max_steps", reward_info=RewardInfo(reward=0.0), messages=[message, user_tool],
    )
    task_result = {**result.reward_info.model_dump(mode="json"), "status": "completed"}
    state = TaskState(
        model="cache-fixture", sample=sample.model_copy(update={"metadata": {**sample.metadata, "task_result": task_result}}),
        output=ModelOutput.from_content(model="cache-fixture", content=result.model_dump_json()),
        messages=[tau3_adapter._trajectory_message(item) for item in result.messages], completed=True,
    )
    score = SampleScore(
        sample_id=index, group_id=index, generation_index=0, sample_metadata=state.metadata,
        score=Score(value={"acc": 0.0}, prediction=result.model_dump_json(), metadata={"task_result": task_result}),
    )
    return state, score


def completed_cache_fixture(directory):
    dataset = cache_fixture_dataset()
    state, score = cache_fixture_record(dataset)
    cache = CacheManager(OutputsStructure(directory), "cache-fixture", "tau3_bench")
    cache.save_prediction_cache("banking_knowledge", state)
    cache.save_review_cache("banking_knowledge", state, score)
    cache.close()
    return cache, dataset


def native_resume_fixture(run_dir, dataset):
    from evalscope import TaskConfig
    from evalscope.api.registry import get_benchmark
    from evalscope.benchmarks.tau_bench.tau3_bench.tau3_bench_adapter import Tau3BenchAdapter
    from evalscope.evaluation_versioning import (
        ResolvedBenchmarkSpec, build_evaluation_identity, build_generated_evaluation_metadata,
    )

    config = TaskConfig(
        model="cache-fixture", api_url="http://unused.invalid/v1", api_key="test",
        eval_type="openai_api", datasets=["tau3_bench"], eval_batch_size=3,
        dataset_args={"tau3_bench": {"subset_list": ["banking_knowledge"]}},
        work_dir=str(run_dir), repeats=1,
    )
    # Supply a deterministic dataset, not a network/model-backed loader; native
    # registry metadata, identity generation and all cache serializers remain real.
    with patch.object(Tau3BenchAdapter, "_prepare_data_dir"):
        benchmark = get_benchmark("tau3_bench", config)
    benchmark.load_dataset = lambda: {"banking_knowledge": dataset}
    meta = benchmark.benchmark_meta
    specs = {"tau3_bench": ResolvedBenchmarkSpec.from_meta(meta, config)}
    identity = build_evaluation_identity(specs, {"tau3_bench": meta.evaluation_version}, config)
    return config, benchmark, build_generated_evaluation_metadata(specs, identity)


def killed_native_writer(run_dir_string, connection):
    from eval.run_tau3_banking import _campaign_writer
    from evalscope.utils.tqdm_utils.progress_tracker import ProgressTracker

    tau3_adapter.install()
    run_dir = Path(run_dir_string)
    with _campaign_writer(run_dir.parent):
        dataset = cache_fixture_dataset(97)
        outputs = OutputsStructure(str(run_dir))
        config, _, generated = native_resume_fixture(run_dir, dataset)
        config.dump_yaml(outputs.configs_dir, generated)
        tracker = ProgressTracker(str(run_dir), pipeline="eval", write_interval=0, total_count=97)
        cache = CacheManager(outputs, "cache-fixture", "tau3_bench")
        for index in (0, 1):
            state, score = cache_fixture_record(dataset, index)
            cache.save_prediction_cache("banking_knowledge", state)
            if index == 1:
                cache.delete_review_cache("banking_knowledge")
            cache.save_review_cache("banking_knowledge", state, score)
            tracker.update()
        prediction = Path(cache.get_prediction_cache_path("banking_knowledge"))
        staging = Path(next(iter(cache._review_reruns.values())))
        # A subsequent native append can be interrupted between payload and LF.
        fragment = b'{"index":2,"unfinished":"\xf0\x9f'
        for path in (prediction, staging):
            with path.open("ab") as stream:
                stream.write(fragment)
                stream.flush()
                os.fsync(stream.fileno())
        connection.send({"staging": str(staging), "fragment": fragment})
        connection.recv()  # Parent SIGKILLs us: no close(), finalizer, or fake error state.




class Tau3AdapterTests(unittest.TestCase):
    def setUp(self):
        tau3_adapter.install()
        self.models_token = tau3_adapter._TASK_MODELS.set(None)
        self.task_token = tau3_adapter._CURRENT_TASK_ID.set(None)

    def tearDown(self):
        tau3_adapter._TASK_MODELS.reset(self.models_token)
        tau3_adapter._CURRENT_TASK_ID.reset(self.task_token)
        tau3_adapter.configure_capture(None)

    def test_sigkill_recovers_native_records_staging_and_stale_running_under_lock(self):
        from eval.run_tau3_banking import _campaign_writer, _validate_resume

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            context = multiprocessing.get_context("spawn")
            parent, child_connection = context.Pipe()
            child = context.Process(target=killed_native_writer, args=(str(run_dir), child_connection))
            child.start()
            child_connection.close()
            try:
                self.assertTrue(parent.poll(30), "native child did not acknowledge its durable records")
                ready = parent.recv()
                before_progress = (run_dir / "progress.json").read_bytes()
                with self.assertRaises(SystemExit):
                    with _campaign_writer(run_dir.parent):
                        self.fail("an active native writer must retain exclusive ownership")
                self.assertEqual((run_dir / "progress.json").read_bytes(), before_progress)
                child.kill()
                child.join(timeout=10)
                self.assertEqual(child.exitcode, -signal.SIGKILL)
            finally:
                if child.is_alive():
                    child.kill()
                    child.join(timeout=10)
                parent.close()
            self.assertEqual(json.loads(before_progress)["status"], "running")
            dataset = cache_fixture_dataset(97)
            original_metadata = [copy.deepcopy(sample.metadata) for sample in dataset]
            config, benchmark, _ = native_resume_fixture(run_dir, dataset)
            with _campaign_writer(run_dir.parent) as lock, patch(
                "evalscope.api.registry.get_benchmark", return_value=benchmark
            ):
                self.assertEqual(_validate_resume(config, run_dir, lock), (2, 2))
                self.assertEqual([sample.metadata for sample in dataset], original_metadata)
                cache = CacheManager(OutputsStructure(str(run_dir), is_make=False), "cache-fixture", "tau3_bench")
                restored, remaining = cache.filter_prediction_cache("banking_knowledge", copy.deepcopy(dataset))
                self.assertEqual({state.sample_id for state in restored}, {0, 1})
                self.assertEqual({sample.id for sample in remaining}, set(range(2, 97)))
                scores, pending = cache.filter_review_cache("banking_knowledge", restored)
                self.assertEqual({score.sample_id for score in scores}, {0, 1})
                self.assertEqual([score.score.value for score in scores], [{"acc": 0.0}, {"acc": 0.0}])
                self.assertEqual(pending, [])
                archived = list(run_dir.glob("recovery-evidence/*/reviews/cache-fixture/" + Path(ready["staging"]).name))
                self.assertEqual(len(archived), 1)
                self.assertTrue(archived[0].read_bytes().endswith(ready["fragment"]))
                self.assertEqual((run_dir / "progress.json").read_bytes(), before_progress)
                # Simulate death after canonical publication but before source
                # staging cleanup: an identical surviving stage is idempotent.
                shutil.copy2(archived[0], ready["staging"])
                self.assertEqual(_validate_resume(config, run_dir, lock), (2, 2))
                self.assertEqual(tau3_adapter._validate_cached_records(cache, dataset, "cache-fixture"), (2, 2))

    def test_native_recovery_rejects_middle_corruption_conflicts_orphans_and_unknown_temps(self):
        from eval.run_tau3_banking import _campaign_writer

        for corruption in ("middle", "conflict", "orphan", "unrelated", "garbage_tail"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory) / "run"
                cache, dataset = completed_cache_fixture(str(run_dir))
                prediction = Path(cache.get_prediction_cache_path("banking_knowledge"))
                review = Path(cache.get_review_cache_path("banking_knowledge"))
                if corruption == "middle":
                    original = prediction.read_bytes()
                    prediction.write_bytes(original + b'{"index":\n' + original)
                elif corruption == "garbage_tail":
                    prediction.write_bytes(prediction.read_bytes() + b"not a native object")
                elif corruption in {"conflict", "orphan"}:
                    state, score = cache_fixture_record(dataset, 0 if corruption == "conflict" else 1)
                    if corruption == "conflict":
                        score.score.value = {"acc": 1.0}
                    cache.delete_review_cache("banking_knowledge")
                    cache.save_review_cache("banking_knowledge", state, score)
                    cache.close()
                else:
                    shutil.copy2(review, str(review) + ".tmp")
                sources = [prediction, review, *review.parent.glob(review.name + ".*")]
                before = {path: path.read_bytes() for path in sources}
                with _campaign_writer(run_dir.parent), self.assertRaises(ValueError):
                    tau3_adapter._recover_cached_records(cache, dataset, "cache-fixture", run_dir)
                self.assertEqual({path: path.read_bytes() for path in sources}, before)
                self.assertFalse((run_dir / "recovery-evidence").exists())

    def test_native_cache_sync_failure_is_not_acknowledged_as_a_saved_prediction(self):
        from eval.run_tau3_banking import _campaign_writer, _validate_resume
        from evalscope.utils.tqdm_utils.progress_tracker import ProgressTracker

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            dataset = cache_fixture_dataset(97)
            outputs = OutputsStructure(str(run_dir))
            config, benchmark, generated = native_resume_fixture(run_dir, dataset)
            config.dump_yaml(outputs.configs_dir, generated)
            ProgressTracker(str(run_dir), pipeline="eval", total_count=97)
            cache = CacheManager(outputs, "cache-fixture", "tau3_bench")
            state, _ = cache_fixture_record(dataset)
            with patch.object(tau3_adapter.os, "fsync", side_effect=OSError("injected durability failure")):
                with self.assertRaisesRegex(OSError, "injected durability failure"):
                    cache.save_prediction_cache("banking_knowledge", state)
            cache.close()
            # A complete surviving record can be validated and made durable on
            # recovery; the original failing save must not silently acknowledge it.
            with _campaign_writer(run_dir.parent) as lock, patch(
                "evalscope.api.registry.get_benchmark", return_value=benchmark
            ):
                self.assertEqual(_validate_resume(config, run_dir, lock), (1, 0))

    def test_failed_progress_publication_preserves_previous_native_snapshot(self):
        from evalscope.utils.tqdm_utils.progress_tracker import ProgressTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = ProgressTracker(directory, pipeline="eval", total_count=97)
            path = Path(directory) / "progress.json"
            before = path.read_bytes()
            with patch.object(tau3_adapter.os, "fsync", side_effect=OSError("injected metadata failure")):
                with self.assertRaisesRegex(OSError, "injected metadata failure"):
                    tracker.set_status("completed")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(before)["status"], "running")

    def test_interleaved_predictions_isolate_roles_capture_and_failure_cleanup(self):
        barrier = threading.Barrier(2, timeout=10)
        secret = "fake-concurrent-simulator-secret"
        replies = {"left": "Left account: " + "L" * 20000, "right": "Right account: " + "R" * 20000}
        observed = {}

        def run_simulation(*, task, **kwargs):
            barrier.wait()
            answers = []
            for role in ("agent", "user"):
                answer = tau3_adapter.patched_generate(
                    model=role, messages=[UserMessage(role="user", content=f"{task.id}: {secret}")],
                )
                answers.append(answer)
                barrier.wait()
            observed[task.id] = [answer.content for answer in answers]
            if task.id == "left":
                raise ValueError(f"simulation failed with {secret}")
            return SimulationRun(
                id="concurrent-right", task_id=task.id, start_time="2026-01-01T00:00:00",
                end_time="2026-01-01T00:00:01", duration=1.0,
                termination_reason="max_steps", reward_info=RewardInfo(reward=0.0), messages=answers,
            )

        def predict_one(task_id, agent_model):
            task = Task(id=task_id, user_scenario={"instructions": "Compare cards."})
            sample = Sample(input="Compare cards.", subset_key="mock", metadata=task.model_dump())
            instance = SimpleNamespace(
                user_model="right" if task_id == "left" else "left",
                api_base="http://unused.invalid", generation_config={},
            )
            try:
                result = tau3_adapter.predict(agent_model, sample, instance)
                outcome = SimulationRun.model_validate_json(result.output.choices[0].message.text)
            except tau3_adapter.Tau3AdapterError as error:
                outcome = error
            # A worker may serve another sample after success OR failure. Neither
            # model access nor task attribution may survive the completed call.
            with self.assertRaises(tau3_adapter.Tau3AdapterError):
                tau3_adapter.patched_generate(model="agent", messages=[UserMessage(role="user", content="outside task")])
            tau3_adapter._capture({"event": "worker_released", "worker": task_id})
            return outcome

        with completion_server([chat_response(replies["left"])]) as (left, _), \
                completion_server([chat_response(replies["right"])]) as (right, _), \
                tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "concurrent.jsonl"
            tau3_adapter.configure_capture(capture)
            models = {"left": left, "right": right}
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": secret}), \
                    patch("evalscope.api.model.get_model", side_effect=lambda **kwargs: models[kwargs["model"]]), \
                    patch("tau2.run.run_task", side_effect=run_simulation), \
                    ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(predict_one, task_id, models[task_id]) for task_id in ("left", "right")]
                failed, completed = [future.result(timeout=30) for future in futures]
            self.assertEqual(observed, {
                "left": [replies["left"], replies["right"]],
                "right": [replies["right"], replies["left"]],
            })
            self.assertIsInstance(failed, tau3_adapter.Tau3AdapterError)
            self.assertNotIn(secret, str(failed))
            self.assertEqual(completed.reward_info.reward, 0.0)
            text = capture.read_text()
            self.assertNotIn(secret, text)
            rows = [json.loads(line) for line in text.splitlines()]
            for task_id in ("left", "right"):
                responses = [row for row in rows if row.get("task_id") == task_id and row["event"] == "response"]
                self.assertEqual([ModelOutput.model_validate(row["response"]).choices[0].message.text for row in responses], observed[task_id])
                requests = [row for row in rows if row.get("task_id") == task_id and row["event"] == "request"]
                self.assertEqual([row["wire_messages"][0]["content"] for row in requests], [f"{task_id}: [REDACTED]"] * 2)
            self.assertEqual([row["task_id"] for row in rows if row["event"] == "task_error"], ["left"])
            self.assertEqual([row["task_id"] for row in rows if row["event"] == "task_result"], ["right"])
            released = [row for row in rows if row["event"] == "worker_released"]
            self.assertEqual({row["worker"] for row in released}, {"left", "right"})
            self.assertTrue(all("task_id" not in row for row in released))

    def test_provider_error_redacts_runtime_credential_from_logs_capture_and_traceback(self):
        from tau2.orchestrator.orchestrator import logger as tau_logger

        sentinel = "fake-deepseek-credential-for-regression"
        log_output = io.StringIO()
        tau_log_output = io.StringIO()
        tau_sink = tau_logger.add(tau_log_output, diagnose=True, backtrace=True)
        logger = logging.getLogger("evalscope.fake_provider_error")
        handler = logging.StreamHandler(log_output)
        logger.addHandler(handler)
        old_propagate = logger.propagate
        logger.propagate = False
        try:
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": sentinel}), tempfile.TemporaryDirectory() as directory:
                tau3_adapter._protect_runtime_logs()
                capture = Path(directory) / "private-errors.jsonl"
                tau3_adapter.configure_capture(capture)

                def fail(**kwargs):
                    try:
                        raise ValueError(f"provider rejected credential {sentinel}")
                    except ValueError:
                        logger.exception("provider error for %s", sentinel)
                        raise

                tau3_adapter._TASK_MODELS.set({"user": SimpleNamespace(generate=fail)})
                simulator = UserSimulator(llm="user")
                state = UserState(
                    system_messages=[SystemMessage(role="system", content="Act as the customer.")],
                    messages=[],
                )
                incoming = AssistantMessage(role="assistant", content="How can I help?")
                try:
                    with patch("tau2.user.user_simulator.generate", tau3_adapter.patched_generate):
                        simulator.generate_next_message(incoming, state)
                except tau3_adapter.Tau3AdapterError as error:
                    self.assertEqual(error.kind, "infrastructure")
                    error_trace = "".join(traceback.format_exception(error))
                    tau_logger.exception("Tau2 simulator provider failure")
                else:
                    self.fail("The provider failure must abort generation")
                for text in (error_trace, capture.read_text(), log_output.getvalue(), tau_log_output.getvalue()):
                    self.assertNotIn(sentinel, text)
                    self.assertIn("provider", text)
                    self.assertIn("[REDACTED]", text)
                self.assertEqual(capture.stat().st_mode & 0o077, 0)
        finally:
            tau_logger.remove(tau_sink)
            logger.removeHandler(handler)
            handler.close()
            logger.propagate = old_propagate

    def test_outer_prediction_loguru_redacts_provider_initialization_failure(self):
        from tau2.orchestrator.orchestrator import logger as tau_logger

        sentinel = "fake-deepseek-initialization-credential"
        rendered = io.StringIO()
        sink = tau_logger.add(rendered, diagnose=True, backtrace=True)
        task = Task(id="privacy", user_scenario={"instructions": "Compare cards."})
        sample = SimpleNamespace(subset_key="banking_knowledge", metadata=task.model_dump())

        def fail(*args):
            raise ValueError(f"provider initialization rejected {sentinel}")

        try:
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": sentinel}), patch.object(tau3_adapter, "_build_model", fail):
                try:
                    tau3_adapter.predict(SimpleNamespace(), sample, SimpleNamespace())
                except tau3_adapter.Tau3AdapterError as error:
                    self.assertEqual(error.kind, "infrastructure")
                    tau_logger.exception("Outer Tau2 prediction initialization failed")
                else:
                    self.fail("Initialization failure must abort prediction")
                self.assertNotIn(sentinel, rendered.getvalue())
                self.assertIn("[REDACTED]", rendered.getvalue())
        finally:
            tau_logger.remove(sink)

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

    def test_simulator_preserves_canonical_prompt_and_advances_state_once(self):
        with completion_server([chat_response("###STOP###")]) as (model, payloads), tempfile.TemporaryDirectory() as directory:
            tau3_adapter._TASK_MODELS.set({"user": model})
            capture = Path(directory) / "attempts.jsonl"
            tau3_adapter.configure_capture(capture)
            simulator = UserSimulator(llm="user", tools=[Tool(check_email)])
            state = UserState(system_messages=[SystemMessage(role="system", content="Act as the customer.")], messages=[])
            incoming = AssistantMessage(role="assistant", content="Your request is complete.")
            with patch("tau2.user.user_simulator.generate", tau3_adapter.patched_generate):
                answer, state = simulator.generate_next_message(incoming, state)
            self.assertEqual(answer.content, "###STOP###")
            self.assertEqual(state.messages, [incoming, answer])
            self.assertEqual(len(payloads), 1)
            self.assertEqual(state.system_messages[0].content, "Act as the customer.")
            self.assertEqual(payloads[0]["messages"][0]["content"], state.system_messages[0].content)
            self.assertEqual(payloads[0]["tools"][0]["function"]["name"], "check_email")
            rows = [json.loads(line) for line in capture.read_text().splitlines()]
            for row, payload in zip([row for row in rows if row["event"] == "request"], payloads):
                self.assertEqual(row["wire_messages"], payload["messages"])
                self.assertEqual(row["wire_tools"], payload["tools"])
                self.assertNotIn("tools", row)

    def test_generation_preserves_task_and_history_for_simulator_agent_and_judge(self):
        messages = [
            SystemMessage(role="system", content="Goal: compare cards; do not submit an application."),
            UserMessage(role="user", content="Which card has no annual fee?"),
        ]
        original = [message.model_dump() for message in messages]
        for role, call_name in [("user", "user_simulator_response"), ("agent", "agent_response"), ("user", "judge")]:
            with self.subTest(role=role, call_name=call_name), completion_server([chat_response("Continue comparing.")]) as (model, payloads):
                tau3_adapter._TASK_MODELS.set({role: model})
                tau3_adapter.patched_generate(model=role, messages=messages, call_name=call_name)
                self.assertEqual(len(payloads), 1)
                self.assertEqual(payloads[0]["messages"], [
                    {"role": "system", "content": messages[0].content},
                    {"role": "user", "content": messages[1].content},
                ])
                self.assertEqual([message.model_dump() for message in messages], original)

    def test_invalid_simulator_completion_aborts_once_without_user_reply(self):
        cases = [
            chat_response(None),
            chat_response(" \n"),
            chat_response(""),
            chat_response(finish_reason="length"),
            {"choices": []},
        ]
        for response in cases:
            with self.subTest(response=response), completion_server([response, chat_response("###STOP###")]) as (model, payloads):
                tau3_adapter._TASK_MODELS.set({"user": model})
                simulator = UserSimulator(llm="user", tools=[Tool(check_email)])
                state = UserState(
                    system_messages=[SystemMessage(role="system", content="Act as the customer.")],
                    messages=[UserMessage(role="user", content="Please check my inbox.")],
                )
                original_system = [message.model_dump() for message in state.system_messages]
                original_history = list(state.messages)
                incoming = AssistantMessage(role="assistant", content="Your request is complete.")
                with patch("tau2.user.user_simulator.generate", tau3_adapter.patched_generate):
                    with self.assertRaises(tau3_adapter.Tau3AdapterError):
                        simulator.generate_next_message(incoming, state)
                self.assertEqual(len(payloads), 1)
                self.assertEqual([message.model_dump() for message in state.system_messages], original_system)
                self.assertEqual(state.messages, original_history + [incoming])

    def test_visible_truncated_agent_response_is_preserved_without_retry(self):
        with completion_server([chat_response("Partial visible answer", finish_reason="length")]) as (model, payloads):
            tau3_adapter._TASK_MODELS.set({"agent": model})
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
                tau3_adapter._TASK_MODELS.set({"user": model})
                answer = tau3_adapter.patched_generate(
                    model="user", call_name="user_simulator_response",
                    messages=[SystemMessage(role="system", content="Act as the customer."), UserMessage(role="user", content="Continue.")], tools=[Tool(check_email)],
                )
                self.assertEqual(answer.content, content)
                self.assertEqual(bool(answer.tool_calls), has_tool)
                if has_tool:
                    self.assertEqual(answer.tool_calls[0].arguments, {"folder": "spam"})
                self.assertEqual(len(payloads), 1)

    def test_completion_error_with_visible_content_aborts_once_without_user_reply(self):
        completion = ModelOutput.from_content(model="error-diagnostic", content="Visible but failed.", error="upstream failure")
        model = SimpleNamespace(generate=lambda **kwargs: completion)
        simulator = UserSimulator(llm="user")
        state = UserState(
            system_messages=[SystemMessage(role="system", content="Act as the customer.")],
            messages=[UserMessage(role="user", content="Please check my inbox.")],
        )
        original_history = list(state.messages)
        incoming = AssistantMessage(role="assistant", content="Your request is complete.")
        tau3_adapter._TASK_MODELS.set({"user": model})
        with patch.object(model, "generate", wraps=model.generate) as generate:
            with patch("tau2.user.user_simulator.generate", tau3_adapter.patched_generate):
                with self.assertRaises(tau3_adapter.Tau3AdapterError) as ctx:
                    simulator.generate_next_message(incoming, state)
            self.assertEqual(ctx.exception.kind, "model_output_invalid")
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(state.messages, original_history + [incoming])

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
            tau3_adapter._TASK_MODELS.set({"user": model})
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

    def test_banking_dataset_validation_rejects_missing_empty_and_duplicate_tasks(self):
        from eval.run_tau3_banking import _validate_banking_dataset

        duplicate = Sample(input="Duplicate", metadata={"id": "task_001"})
        for datasets, expected_count in [
            ({}, 1),
            ({"banking_knowledge": []}, 97),
            ({"banking_knowledge": [duplicate, duplicate]}, 2),
        ]:
            with self.subTest(datasets=datasets), self.assertRaises(ValueError):
                _validate_banking_dataset(datasets, expected_count)

    def test_native_cache_reuses_completed_zero_reward_max_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, dataset = completed_cache_fixture(directory)
            review_path = Path(cache.get_review_cache_path("banking_knowledge"))
            review = json.loads(review_path.read_text())
            review["messages"][0]["id"] = "regenerated-report-message-id"
            review_path.write_text(json.dumps(review) + "\n")
            self.assertEqual(tau3_adapter._validate_cached_records(cache, dataset, "cache-fixture"), (1, 1))
            restored, remaining = cache.filter_prediction_cache("banking_knowledge", dataset)
            self.assertEqual([sample.id for sample in remaining], [1])
            self.assertEqual(SimulationRun.model_validate_json(restored[0].output.message.text).termination_reason, "max_steps")
            scores, pending_review = cache.filter_review_cache("banking_knowledge", restored)
            self.assertEqual(scores[0].score.value, {"acc": 0.0})
            self.assertEqual(pending_review, [])

    def test_resume_rejects_duplicate_or_unproven_cache_records(self):
        for corruption in ["duplicate_prediction", "invalid_index", "missing_completed_status", "duplicate_review", "orphan_review"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                cache, dataset = completed_cache_fixture(directory)
                prediction_path = Path(cache.get_prediction_cache_path("banking_knowledge"))
                review_path = Path(cache.get_review_cache_path("banking_knowledge"))
                prediction = json.loads(prediction_path.read_text())
                review = json.loads(review_path.read_text())
                if corruption == "duplicate_prediction":
                    prediction_path.write_text(prediction_path.read_text() * 2)
                elif corruption == "invalid_index":
                    prediction["index"] = -1
                    prediction_path.write_text(json.dumps(prediction) + "\n")
                elif corruption == "missing_completed_status":
                    del prediction["metadata"]["task_result"]["status"]
                    prediction_path.write_text(json.dumps(prediction) + "\n")
                elif corruption == "duplicate_review":
                    review_path.write_text(review_path.read_text() * 2)
                else:
                    review["index"] = 1
                    review_path.write_text(json.dumps(review) + "\n")
                with self.assertRaises(ValueError):
                    tau3_adapter._validate_cached_records(cache, dataset, "cache-fixture")

    def test_resume_rejects_lost_reasoning_and_mismatched_review(self):
        for corruption in ["lost_reasoning", "lost_user_tool", "wrong_reward", "wrong_trajectory", "different_canonical_run"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                cache, dataset = completed_cache_fixture(directory)
                path = Path(cache.get_prediction_cache_path("banking_knowledge") if corruption in {"lost_reasoning", "lost_user_tool"} else cache.get_review_cache_path("banking_knowledge"))
                row = json.loads(path.read_text())
                if corruption == "lost_reasoning":
                    row["messages"][0]["content"] = [
                        content for content in row["messages"][0]["content"] if content["type"] != "reasoning"
                    ]
                elif corruption == "lost_user_tool":
                    row["messages"][1]["metadata"] = {}
                elif corruption == "wrong_reward":
                    row["sample_score"]["score"]["value"]["acc"] = 1.0
                elif corruption == "different_canonical_run":
                    scored_run = json.loads(row["sample_score"]["score"]["prediction"])
                    scored_run["seed"] = 43
                    row["sample_score"]["score"]["prediction"] = json.dumps(scored_run)
                else:
                    row["messages"][0]["content"] = "Unrelated trajectory"
                path.write_text(json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    tau3_adapter._validate_cached_records(cache, dataset, "cache-fixture")



if __name__ == "__main__":
    unittest.main()
