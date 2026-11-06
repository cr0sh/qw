# RULER accuracy of D-LSA on QW

Scripts that replicate upstream's RULER evaluation of D-LSA (Qwen3.8-27B, 13 tasks, 4k and 32k) against QW's
OpenAI-compatible server, and summarize dense, D-LSA and PCA-map runs side by side.

Upstream: [Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) @
`5ebea732d639576d2a7579b0de40e6252f9001f5` (Apache-2.0). File:line cites below are into that commit.

| file | what | runs with |
|---|---|---|
| `prepare_ruler.py` | generate the data files with upstream's driver on macOS; check them against the pinned hashes | the pinned venv (generation); stdlib (`--check-hashes`) |
| `run_ruler.py` | send the rows to a server, score them, resumably | Python 3 stdlib |
| `ruler_score.py` | upstream's prompt join, prompt-set hash and metric | Python 3 stdlib |
| `summarize.py` | markdown tables over several configurations | Python 3 stdlib |
| `server_log_stats.py` | per-request prefill / decode / draft acceptance / D-LSA stats from a QW JSON log | Python 3 stdlib |
| `test_ruler_harness.py` | `python3 -m unittest discover -s eval/dlsa/ruler -p 'test_*.py'` | Python 3 stdlib |

The data files are not committed (they are generated, 167 MB for the 26 cells); only the scripts and these notes are.

## Results

Roadmap task 7's runs (2026-11-06, M4 Max; dense 20 rows per task at 4k and 10 at 32k, D-LSA 10 at 32k and 2 at 4k)
are in [`results/task7/`](results/task7/): `dense/` and `dlsa/` (per-row records and `meta.json`), `summary.md` and
`summary.json` (`summarize.py`), and `server-log-{dense,dlsa}.json` (`server_log_stats.py`). They are read in
[`docs/dlsa.md`](../../../docs/dlsa.md) §16: at 32k, mean 0.968 for both, identical on 12 of 13 tasks, cwe +0.010
(2 of 130 rows changing score, one each way). The records are resumable runs: raising `--rows` with the same `--out`
adds rows to them.

## The protocol upstream runs, and how it is replicated

- **Data**: NVIDIA/RULER's own generators at the card's commit `c3f5e3b4` with upstream's patch, seed 42,
  prompts sized in Qwen3.8-27B's tokenizer (`src/lsa/evaluation/accuracy/data/ruler.py`). Files
  `<root>/ruler/<task>_<length>.jsonl`, 100 rows each, fields `index, input, outputs, gen_prefix, length,
  max_length`; one manifest per file under `<root>/manifest/`. An N-row set is the first N rows of the file.
- **Tasks**: niah_single_1/2/3, niah_multikey_1/2/3, niah_multiquery, niah_multivalue, vt, cwe, fwe, qa_squad,
  qa_hotpot; lengths 4096 and 32768.
- **Prompt**: lm-eval's, as upstream builds it (`tasks.py:217` target delimiter `" "`, `runner.py:950` one user
  turn, no system prompt, no template rendered by the client): `input`, then `gen_prefix`, with one space
  between them only where neither side has whitespace at the seam. The qa_squad / qa_hotpot rows whose
  question ends in a space get no second space; a plain `input + " " + gen_prefix` misses 4 of the 30 pinned
  prompt-set hashes, this join matches all 30 (`ruler_score.prompt_text`).
- **Request** (`client.py:106-113`, `sampling.py:41-56`, `configs/evaluation/accuracy/qwen38_27b/_base.yaml:99-104`):
  `POST /v1/chat/completions` with exactly
  `{"model", "messages": [{"role": "user", "content": P}], "temperature": 0.0, "top_p": 1.0, "max_tokens": 4096,
  "chat_template_kwargs": {"enable_thinking": false}}`; no stop strings (`client.py:75`).
- **Retries** (`client.py:132-160`): 3 attempts with backoff min(2^attempt, 30) s on connection errors, HTTP 5xx
  and malformed replies; a 4xx and a timeout are not retried (a timed-out row is sent again by a continued run).
  Timeout 14400 s.
- **Score** (`src/lsa/evaluation/accuracy/grading/ruler.py:7-37`, lm-eval's `postprocess_pred`): only
  `choices[0].message.content` is graded. `p = re.sub(r"[\x00-\x1f]", "\n", reply.strip()).strip()`; qa_squad and
  qa_hotpot score `string_match_part` (1 if any gold string is a case-insensitive substring of p), the other 11
  tasks `string_match_all` (the fraction of gold strings found). Rows cut at `max_tokens` are scored as they
  are. A task's score is the mean over its rows without an error; a length's is the mean of its 13 task scores.
- **Reference**: upstream README.md:178-183, Qwen3.8-27B D-LSA (Flat): 0.974 at 4k, 0.968 at 32k (the paper:
  0.975 / 0.966). No per-task or dense numbers are published as text.
- **Concurrency**: upstream's client keeps 8 requests in flight (`client.py:84`); QW runs one sequence at a time
  on this machine, so `run_ruler.py` sends one at a time by default (`--concurrency 1`); every request stands
  alone, so the number in flight changes no prompt.

## Preparing the data (macOS)

```sh
uv venv --python 3.12 v
uv pip install --python v/bin/python tokenizers==0.23.2 transformers==5.17.0 numpy==2.5.3 scipy==1.18.1 \
  wonderwords==3.0.1 nltk==3.10.3 pyyaml hydra-core submitit tqdm huggingface-hub
v/bin/python eval/dlsa/ruler/prepare_ruler.py --lsa-clone LSA --root DATA --tokenizer Qwen/Qwen3.8-27b \
  --lengths 4096 32768 --samples 100
HF_HUB_OFFLINE=1 v/bin/python eval/dlsa/ruler/prepare_ruler.py ...      # once the tokenizer is cached
python3 eval/dlsa/ruler/prepare_ruler.py --root DATA \
  --check-hashes LSA/src/lsa/evaluation/accuracy/data/hashes_qwen38_27b.json   # expect 30/30 MATCH
```

- The pins matter: tokenizers 0.23.3 changes 3 cells.
- Upstream's `lsa data prepare ruler --tokenizer Qwen/Qwen3.8-27b` fails on macOS: `os.sched_getaffinity`
  does not exist (`ruler.py:270,381`), and `publish()` uses GNU `mv --no-clobber --no-target-directory`
  (`ruler.py:559`). `prepare_ruler.py` supplies the first and replaces the second with a non-overwriting rename,
  without editing upstream's files, and calls `ruler.prepare_cell` per cell instead of going through submitit.
- Upstream's `ensure()` installs and verifies the prerequisites (default `DATA/ruler-nvidia-<patch>` and
  `DATA/nltk_data`; `RULER_NVIDIA` and `NLTK_DATA` override): NVIDIA/RULER at `c3f5e3b4` with upstream's
  `ruler.patch`, `PaulGrahamEssays.json` and `squad.json` from upstream's release `ruler-sources-v1`,
  `english_words.json` from NVIDIA/RULER @ c3f5e3b4 (LFS), `hotpotqa.json` from the CMU host or its HF mirror,
  each checked against its sha256 (`ruler.py:83-106`, `verify()` at `ruler.py:448`).
- Hash check: the sha256 of each prompt, first 16 hex digits, concatenated over the first N rows, sha256 again,
  first 16 (`tasks.py:534-568`), against `ruler_<task>_<length>:first<N>` in `hashes_qwen38_27b.json`.

## Servers, one configuration each

Every server command goes through `./gpu-lock --`, built with the dlsa feature, unquantized KV (D-LSA requires
it, and the dense baseline must match). Keep the decoder the same across configurations (pass `--decoder`
explicitly, e.g. `--decoder baseline`), so that only the attention differs.

```sh
./gpu-lock -- qw serve --no-kv-quantization --bind 127.0.0.1:8000 --output-format json > dense.log 2>&1
./gpu-lock -- qw serve --dlsa --no-kv-quantization --bind 127.0.0.1:8000 --output-format json > dlsa.log 2>&1
./gpu-lock -- qw serve --dlsa --dlsa-map-kind pca --no-kv-quantization --bind 127.0.0.1:8000 --output-format json > pca.log 2>&1
```

- The prefix cache stays on as QW ships; each record keeps `cached_tokens`, and the timing table shows its mean,
  so a reused prefix is visible. `--disable-prefix-cache` turns it off for a timing-clean run.
- For `server_log_stats.py` to see the speculative decoder and D-LSA statistics on stdout, start the server with
  `RUST_LOG=info,qw_server::engine=debug,qw_runtime::provider=debug` (stdout logs at `RUST_LOG`, INFO by default,
  and those events are DEBUG). Without `--no-file-logging` the daily file `~/.cache/qw/log/qw.log.<date>` already
  holds them: it logs at `--persistent-log-filter` / `QW_LOG`, by default `debug,qw_server=trace,qw_runtime=trace,...`.

## Running, scoring, summarizing

```sh
python3 eval/dlsa/ruler/run_ruler.py --data-root DATA --out results/dense --label dense --dry-run
python3 eval/dlsa/ruler/run_ruler.py --data-root DATA --out results/dense --label dense --rows 100 \
  --note "qw <commit>, --decoder baseline"
python3 eval/dlsa/ruler/summarize.py --runs dense=results/dense dlsa=results/dlsa pca=results/pca \
  --extrapolate 100 --json results/summary.json
python3 eval/dlsa/ruler/server_log_stats.py dlsa.log
```

- `run_ruler.py` scores every reply as it lands and appends it to `<out>/<task>_<length>.jsonl` (fsynced):
  `index, task, length, prompt_tokens, completion_tokens, cached_tokens, finish_reason, content,
  reasoning_content, score, latency_s, started_at, finished_at, error, attempt, http_attempts, response_id`.
  `attempt` counts the row's records so far; `response_id` is the id QW logs the request under.
- Rows go one length at a time, cheapest first (`--order as-given` keeps `--lengths` order), then task, then
  index. Started again with the same `--out`, it skips rows that have a record without an error and sends the
  errored ones again; a line torn by a kill is skipped and closed. So stop it any time, and grow a run by raising
  `--rows` (4 rows are the first 4 of the 100). It exits 1 while any row ends in an error.
- `<out>/meta.json` holds the label, model and `max_tokens` (a run with others is refused in that directory), the
  request's sampling fields, and per invocation: note, server URL, data root, rows, lengths, tasks, each file's
  sha256 with its manifest's and its prompt-set hash, the harness's `git describe`, and the server's
  `/v1/models` reply (QW has no such route; the error is kept instead).
- Timings on this machine (dense, measured 2026-11-06): a 4k row ~18 s (prefill ~237 tok/s), a 32k row
  ~145 s (~226 tok/s), decode ~23 tok/s: 100 rows x 13 tasks is ~6.5 h at 4k and ~52 h at 32k per configuration.
  `summarize.py --extrapolate N` turns the rows done into an estimate for N rows per task.
- `summarize.py` prints, per length, each configuration's own mean (n rows) and the means over the rows every
  configuration finished ("paired"), with the delta to the first (dense) run and to the upstream D-LSA number;
  then timing, the rows that errored or stopped at `max_tokens`, and a per-task agreement table between two
  configurations (`--compare A B`, default the first two): rows whose scores differ, and rows whose reply text
  differs.
