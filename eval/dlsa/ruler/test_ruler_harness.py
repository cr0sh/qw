"""Tests of the RULER harness: upstream's scoring, resuming, the request body, the summary and log parsing.

python3 -m unittest discover -s eval/dlsa/ruler -p 'test_*.py'
"""
import contextlib
import http.server
import io
import json
import os
import sys
import socketserver
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ruler  # noqa: E402
import server_log_stats  # noqa: E402
import summarize  # noqa: E402
from ruler_score import metric_for_task, postprocess_pred, prompt_set_hash, prompt_text, score_row  # noqa: E402


class ScoringTests(unittest.TestCase):
    def test_postprocess_strips_and_turns_control_characters_into_newlines(self):
        self.assertEqual(postprocess_pred("  \t7210606\x00\x1fdone \n"), "7210606\n\ndone")
        self.assertEqual(postprocess_pred("\x07"), "")

    def test_match_ignores_case_and_markdown_around_the_answer(self):
        self.assertEqual(score_row("niah_single_1", "The number is **7210606**.", ["7210606"]), 1.0)
        self.assertEqual(score_row("niah_single_3", "ANSWER: AbC-123", ["abc-123"]), 1.0)
        self.assertEqual(score_row("niah_single_1", "7210605", ["7210606"]), 0.0)

    def test_qa_takes_any_reference_and_the_rest_the_fraction_found(self):
        self.assertIs(metric_for_task("qa_squad").__name__, "string_match_part")
        self.assertIs(metric_for_task("qa_hotpot").__name__, "string_match_part")
        self.assertIs(metric_for_task("niah_multivalue").__name__, "string_match_all")
        self.assertEqual(score_row("qa_hotpot", "It was Paris", ["London", "paris"]), 1.0)
        self.assertEqual(score_row("niah_multivalue", "1, 2 and 3", ["1", "2", "3", "4"]), 0.75)
        words = [f"word{i}" for i in range(10)]
        self.assertAlmostEqual(score_row("cwe", " ".join(words[:7]), words), 0.7)
        self.assertEqual(score_row("vt", "", ["X"]), 0.0)
        with self.assertRaises(KeyError):
            metric_for_task("niah_single_4")

    def test_prompt_joins_with_one_space_only_where_neither_side_has_whitespace(self):
        self.assertEqual(prompt_text({"input": "Q?", "gen_prefix": "Answer:"}), "Q? Answer:")
        self.assertEqual(prompt_text({"input": "Q? ", "gen_prefix": "Answer:"}), "Q? Answer:")
        self.assertEqual(prompt_text({"input": "Q?\n", "gen_prefix": "A"}), "Q?\nA")
        self.assertEqual(prompt_set_hash([{"input": "Q? ", "gen_prefix": "A"}]), prompt_set_hash([{"input": "Q?", "gen_prefix": "A"}]))


class Stub(http.server.BaseHTTPRequestHandler):
    """A chat endpoint that answers from `Stub.script`: a list of (status, body) per request, then 200s."""

    script = []
    bodies = []
    reply = "The special magic number — **{answer}**."

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.bodies.append(body)
        status, payload = Stub.script.pop(0) if Stub.script else (200, None)
        if payload is None:
            answer = body["messages"][0]["content"].split("#")[1]
            payload = {
                "id": f"chatcmpl-{len(Stub.bodies)}",
                "choices": [
                    {"message": {"content": Stub.reply.format(answer=answer), "reasoning_content": ""}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "prompt_tokens_details": {"cached_tokens": 0}},
            }
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


class LocalServer(http.server.ThreadingHTTPServer):
    """The stub's server, binding without HTTPServer's reverse lookup of the host (seconds on macOS)."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "data")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(os.path.join(self.root, "ruler"))
        self.write_cell(4096)
        self.server = LocalServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        Stub.script, Stub.bodies = [], []
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def write_cell(self, length, gold=1000):
        with open(os.path.join(self.root, "ruler", f"niah_single_1_{length}.jsonl"), "w") as sink:
            for i in range(3):
                row = {"index": i, "input": f"needle #{1000 + i}# hidden", "outputs": [str(gold + i)],
                       "gen_prefix": "It is", "length": 50, "max_length": length}
                sink.write(json.dumps(row) + "\n")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def run_main(self, *extra):
        argv = ["--base-url", self.url, "--data-root", self.root, "--out", self.out, "--tasks", "niah_single_1",
                "--lengths", "4096", "--rows", "3", "--label", "dense", *extra]
        with mock.patch.object(run_ruler.time, "sleep"), contextlib.redirect_stderr(io.StringIO()):
            return run_ruler.main(argv)

    def records(self):
        return run_ruler.load_records(os.path.join(self.out, "niah_single_1_4096.jsonl"))

    def test_request_body_has_exactly_upstreams_keys(self):
        body = run_ruler.request_body({"input": "Q", "gen_prefix": "A"}, "qwen3.8-27b", 4096)
        self.assertEqual(
            body,
            {
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "Q A"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    def test_run_scores_records_and_writes_meta(self):
        self.assertEqual(self.run_main(), 0)
        records = self.records()
        self.assertEqual([r["index"] for r in sorted(records, key=lambda r: r["index"])], [0, 1, 2])
        self.assertTrue(all(r["score"] == 1.0 and r["error"] is None and r["attempt"] == 1 for r in records))
        self.assertEqual(records[0]["prompt_tokens"], 11)
        self.assertEqual(records[0]["cached_tokens"], 0)
        self.assertEqual(Stub.bodies[0], run_ruler.request_body({"input": "needle #1000# hidden", "gen_prefix": "It is"}, "qwen3.8-27b", 4096))
        with open(os.path.join(self.out, "meta.json")) as source:
            meta = json.load(source)
        self.assertEqual(meta["label"], "dense")
        self.assertIn("unavailable", meta["invocations"][0]["server_models"])
        self.assertIn("niah_single_1_4096", meta["invocations"][0]["data"])

    def test_resume_skips_finished_rows_and_reruns_errored_ones(self):
        self.assertEqual(self.run_main("--rows", "1"), 0)
        path = os.path.join(self.out, "niah_single_1_4096.jsonl")
        with open(path, "ab") as sink:
            sink.write(json.dumps({"index": 1, "error": "timeout", "score": None}).encode() + b"\n")
            # torn by a kill inside a multibyte character (the first two bytes of an em dash)
            sink.write(b'{"index": 2, "content": "The number \xe2\x80')
        Stub.bodies.clear()
        self.assertEqual(self.run_main(), 0)
        sent = sorted(b["messages"][0]["content"] for b in Stub.bodies)
        self.assertEqual(sent, ["needle #1001# hidden It is", "needle #1002# hidden It is"])
        latest = {r["index"]: r for r in self.records()}
        self.assertEqual(sorted(latest), [0, 1, 2])
        self.assertEqual(latest[0]["attempt"], 1)
        self.assertEqual(latest[1]["attempt"], 2)
        self.assertEqual(latest[2]["attempt"], 1)
        self.assertIn("—", latest[2]["content"])
        # The torn line was closed, so every record after it sits on a line of its own
        with open(path, "rb") as source:
            self.assertEqual(sum(1 for _ in source), 5)
        # A finished run sends nothing more
        Stub.bodies.clear()
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(Stub.bodies, [])

    def test_server_errors_retry_and_client_errors_do_not(self):
        Stub.script = [(500, {"error": "busy"}), (400, {"error": "bad"})]
        self.assertEqual(self.run_main("--concurrency", "1"), 1)
        errored = [r for r in self.records() if r["error"]]
        self.assertEqual(len(errored), 1)
        self.assertTrue(errored[0]["error"].startswith("http 400"))
        self.assertEqual(errored[0]["http_attempts"], 2)
        self.assertEqual(len(Stub.bodies), 4)

    def test_concurrent_rows_each_append_one_whole_record(self):
        self.assertEqual(self.run_main("--concurrency", "3"), 0)
        self.assertEqual(sorted(r["index"] for r in self.records()), [0, 1, 2])

    def test_another_configuration_is_refused_in_the_same_directory(self):
        self.assertEqual(self.run_main(), 0)
        with self.assertRaises(SystemExit):
            self.run_main("--label", "dlsa")

    def test_changed_data_is_refused(self):
        self.assertEqual(self.run_main(), 0)
        self.write_cell(4096, gold=2000)  # same indices, other gold answers
        Stub.bodies.clear()
        with self.assertRaises(SystemExit) as refused:
            self.run_main()
        self.assertIn("use another --out for other data", str(refused.exception))
        with self.assertRaises(SystemExit):
            self.run_main("--dry-run")
        self.assertEqual(Stub.bodies, [])

    def test_more_rows_and_another_length_resume_on_unchanged_data(self):
        self.assertEqual(self.run_main("--rows", "2"), 0)
        self.assertEqual(self.run_main("--rows", "3"), 0)
        self.assertEqual(len(Stub.bodies), 3)
        self.write_cell(32768)
        self.assertEqual(self.run_main("--lengths", "4096", "32768"), 0)
        self.assertEqual(len(Stub.bodies), 6)
        longer = run_ruler.load_records(os.path.join(self.out, "niah_single_1_32768.jsonl"))
        self.assertEqual(sorted(r["index"] for r in longer), [0, 1, 2])
        self.assertEqual(self.run_main("--lengths", "4096", "32768"), 0)
        self.assertEqual(len(Stub.bodies), 6)

    def test_dry_run_sends_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertEqual(Stub.bodies, [])
        self.assertIn('"enable_thinking": false', printed.getvalue())
        self.assertFalse(os.path.exists(self.out))


def record(task, index, score, length=4096, content=None, error=None, latency=10.0, finish="stop"):
    return {
        "task": task, "length": length, "index": index, "score": score, "error": error, "content": content or str(score),
        "latency_s": latency, "prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 0, "finish_reason": finish,
        "started_at": "2026-11-06T00:00:00Z", "finished_at": "2026-11-06T00:00:10Z",
    }


class SummarizeTests(unittest.TestCase):
    def test_paired_means_and_deltas_use_rows_every_run_finished(self):
        dense = summarize.final_records(
            [record("vt", 0, 1.0), record("vt", 1, 1.0), record("vt", 2, 0.0), record("cwe", 0, 0.5)]
        )
        dlsa = summarize.final_records(
            [
                record("vt", 0, 1.0),
                record("vt", 1, None, error="timeout"),
                record("vt", 1, 0.0),  # the rerun that finished counts
                record("vt", 2, None, error="timeout"),  # never finished: unpaired
                record("cwe", 0, 0.25, finish="length"),
            ]
        )
        table = summarize.score_table([("dense", {}, dense), ("dlsa", {}, dlsa)], 4096)
        rows = {r["task"]: r for r in table["tasks"]}
        self.assertEqual(rows["vt"]["n"], [3, 2])
        self.assertAlmostEqual(rows["vt"]["own"][0], 2 / 3)
        self.assertEqual(rows["vt"]["paired_n"], 2)
        self.assertEqual(rows["vt"]["paired"], [1.0, 0.5])
        self.assertEqual(table["paired_mean"], [0.75, 0.375])
        self.assertAlmostEqual(table["own_mean"][0], (2 / 3 + 0.5) / 2)
        self.assertEqual(table["paper"], 0.974)
        flagged = summarize.flagged([("dense", {}, dense), ("dlsa", {}, dlsa)])
        self.assertEqual([(f["task"], f["index"]) for f in flagged], [("vt", 2), ("cwe", 0)])
        agreement = {r["task"]: r for r in summarize.agreement(("dense", {}, dense), ("dlsa", {}, dlsa), 4096)}
        self.assertEqual(agreement["vt"]["b_lower"], [1])
        self.assertEqual(agreement["vt"]["content_differs"], 1)

    def test_timing_extrapolates_per_task_means(self):
        final = summarize.final_records([record("vt", 0, 1.0, latency=36.0), record("cwe", 0, 1.0, latency=72.0)])
        (row,) = summarize.timing_table([("dense", {}, final)], 4096, 100)
        self.assertEqual(row["mean_latency_s"], 54.0)
        # vt 36 s, cwe 72 s, the 11 tasks not run yet 54 s each: 100 rows of each
        self.assertAlmostEqual(row["estimate_h"], 100 * (36 + 72 + 11 * 54) / 3600)

    def test_main_renders_markdown_from_result_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, scores in (("dense", [1.0, 1.0]), ("dlsa", [1.0, 0.0])):
                os.makedirs(os.path.join(tmp, label))
                with open(os.path.join(tmp, label, "meta.json"), "w") as sink:
                    json.dump({"label": label}, sink)
                with open(os.path.join(tmp, label, "vt_4096.jsonl"), "w") as sink:
                    for i, s in enumerate(scores):
                        sink.write(json.dumps(record("vt", i, s)) + "\n")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                summarize.main(["--runs", os.path.join(tmp, "dense"), os.path.join(tmp, "dlsa"), "--json", os.path.join(tmp, "s.json")])
            text = printed.getvalue()
            self.assertIn("| vt | 1.000 (2) | 0.500 (2) | 2 | 1.000 | 0.500 | -0.500 |", text)
            self.assertIn("### Agreement 4096: dlsa vs dense", text)
            self.assertTrue(text.endswith("|\n") and not text.endswith("\n\n"))
            with open(os.path.join(tmp, "s.json")) as source:
                self.assertEqual(json.load(source)["scores"][0]["paired_mean"], [1.0, 0.5])

    def test_a_record_torn_inside_a_multibyte_character_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "vt_4096.jsonl"), "wb") as sink:
                sink.write(json.dumps(record("vt", 0, 1.0, content="café — 1"), ensure_ascii=False).encode() + b"\n")
                sink.write(b'{"index": 1, "content": "\xe2\x80')
            label, _, final = summarize.load_run(f"x={tmp}")
        self.assertEqual(label, "x")
        self.assertEqual(list(final), [(4096, "vt", 0)])
        self.assertEqual(final[(4096, "vt", 0)]["content"], "café — 1")


class LogTests(unittest.TestCase):
    def test_requests_group_by_response_id(self):
        span = {"name": "generation", "response_id": "chatcmpl-a"}
        lines = [
            "gpu-lock: acquired",
            json.dumps({"level": "DEBUG", "fields": {"phase": "dflash2.completed", "stats": "Dflash2GenerationStats { accepted_draft_tokens: 137, proposed_draft_tokens: 236, prefill_time: 6.88s, verify_width_rounds: [72, 5], speculative_rounds: 77 }"}, "span": span}),
            json.dumps({"level": "INFO", "fields": {"event": "prefill.complete", "chat_id": "chatcmpl-a", "tps": 236.6, "total_tokens": 3970, "prefilled_tokens": 3970, "prefix_reused_tokens": 0}, "span": span}),
            json.dumps({"level": "INFO", "fields": {"event": "decode.complete", "chat_id": "chatcmpl-a", "tps": 22.7, "total_tokens": 3977, "decoded_tokens": 7, "prefix_reused_tokens": 0}, "span": span}),
            json.dumps({"level": "DEBUG", "fields": {"phase": "dlsa.request", "sparse_rows": 12, "keys_read": 900, "rollbacks": 1, "index_ms": 3.5}}),
            json.dumps({"level": "DEBUG", "fields": {"phase": "model.complete", "mtp_proposed_draft_tokens": 10, "mtp_accepted_draft_tokens": 4}, "spans": [{"name": "server.request"}, {"name": "generation", "response_id": "chatcmpl-b"}]}),
            json.dumps({"level": "INFO", "fields": {"event": "prefill.complete", "chat_id": "chatcmpl-b", "tps": 200.0, "total_tokens": 32649, "prefilled_tokens": 32649, "prefix_reused_tokens": 0}}),
        ]
        a, b = server_log_stats.requests(lines)
        self.assertEqual((a["id"], a["prefill_tps"], a["prompt_tokens"], a["decoded_tokens"], a["decode_tps"]), ("chatcmpl-a", 236.6, 3970, 7, 22.7))
        self.assertEqual((a["decoder"], a["accepted"], a["proposed"], a["rounds"]), ("dflash2", 137, 236, 77))
        self.assertEqual(a["dlsa"]["sparse_rows"], 12)
        self.assertEqual(a["dlsa"]["rollbacks"], 1)
        self.assertEqual((b["decoder"], b["acceptance"], b["prompt_tokens"]), ("mtp", 0.4, 32649))
        text = "\n".join(server_log_stats.render([a, b]))
        self.assertIn("draft acceptance: 141/246", text)
        self.assertIn("dlsa.request over 1 requests", text)


if __name__ == "__main__":
    unittest.main()
