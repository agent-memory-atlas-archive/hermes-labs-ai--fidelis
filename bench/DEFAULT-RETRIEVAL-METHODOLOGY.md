# Default retrieval evaluation for 0.3.0rc1

This measures retrieval through the redesigned candidate's actual `POST /query`
handler, the route used by `fidelis_recall` with its default `mode="fast"` and
`limit=5`. It does not evaluate generated answers, the thorough hybrid path,
write acceptance, or deployment reliability. The release proof covers installation
and MCP separately.

## Dataset and corpus

The input is `longmemeval_s_cleaned.json` from
[xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
The artifact records its SHA-256. The dataset contains 500 questions; the fixed
selection excludes question IDs containing `_abs`, leaving all 470 answerable
questions in original order. Abstention behavior is not measured.

Each question searches only its own haystack, isolated by the server's configured
`user_id` filter. Each session is one record: all turns in original order, each
formatted as `role: content`, separated by newlines. Neither dates, answers,
question categories nor gold session labels enter the embedded text. The
identifier mapping is used only after retrieval for scoring. Corpus metadata
contains a session ID, but the candidate does not rank using it.

The runner seeds real mem0/Chroma records directly with full session text and
vectors obtained by the candidate's `_embed_bounded` function and default Ollama
`nomic-embed-text` embedder. This preserves the actual embedding fallback behavior
for oversized records without substituting benchmark chunking or prefix logic.
It bypasses the write gate: these are preloaded, unversioned retrieval records,
not a claim that every benchmark conversation passes the public store endpoint.

## Runtime and reproducibility

The runner creates a fresh isolated HOME, Chroma store and queue for every run;
it serves the candidate's `make_handler` on loopback at a port of at least 19440.
It makes real HTTP requests containing only `{"text": question}`. Default
retrieval, filtering, temporal annotation and result limiting all run in the
candidate implementation. It does not duplicate its ranking algorithm.

Corpus vectors can be reused between smoke and full runs. Cache keys include
exact text, model digest, model details, embedder source and `_embed_bounded`
source. Query embeddings and every retrieval output are fresh. No historical
ranking output is consumed. The release run also reused vectors after a
docstring-only sanitization: both complete source-file hashes were bound to
their run artifacts, the embedding function AST with its docstring removed
was identical, and the same model/dependency identity and exact old embedder
cache keys were verified. The [cache provenance receipt](eval-cache-provenance-0.3.0rc1.json)
records the hashes and vector counts. This changes no executable embedding
behavior; a cold reproduction can compute the same vectors without this cache. Source hashes, dependencies, model identity and
per-question ranked session IDs are retained in the JSON artifact. The model
is local; generation is forbidden in the memory object's LLM implementation.
The default route has no LLM generation or remote inference stage.

Run a five-question smoke first, then all 470 if indexing and requests succeed:

```sh
python bench/longmemeval_default.py --data /path/to/longmemeval_s_cleaned.json \
  --state /path/to/isolated-eval-cache --output smoke.json --limit 5
python bench/longmemeval_default.py --data /path/to/longmemeval_s_cleaned.json \
  --state /path/to/isolated-eval-cache --output results.json
```

Run with the candidate installed or `PYTHONPATH=src`, using its dependencies and
a local Ollama service with `nomic-embed-text`. The runner never opens the user's
normal memory store. Failures remain in the denominator as zero-credit rows;
they are never silently dropped. Any failure returns a nonzero exit status.

## Metrics and interpretation

- **Hit@1 / Hit@5:** fraction of questions with at least one gold session among
  the first one / five returned records.
- **Session recall@5:** per-question fraction of its gold sessions retrieved
  among the first five, macro-averaged over all questions.
- **All-answer-sessions@5:** fraction of questions for which every gold session
  appears among the first five results.

There are 170 questions with one gold session and 300 with multiple gold
sessions. Three questions require six sessions, so the default five-result cap
cannot retrieve all their gold sessions. These questions remain in the denominator.

These metrics differ on questions requiring multiple sessions. None is QA
accuracy. A retrieval hit does not prove the evidence was sufficient to answer
correctly. Reported latency covers the HTTP query only, excluding ingestion;
it is a local operational observation, not a cross-machine performance promise.

The historical 83.2% result came from a different benchmark pipeline. This
candidate measurement uses whole-session records and default vector retrieval,
so it is not a controlled before/after comparison with that pipeline, with
chunked hybrid retrieval, or with published QA benchmarks. The redesign should
not inherit the historical headline as evidence. Full LLM/QA evaluation remains
post-release work.

## Source-bound release run

The [final default-path run](results-default-0.3.0rc1.json) completed all 470
eligible questions with zero errors after a [healthy five-question smoke](results-default-smoke-0.3.0rc1.json).
The runner and candidate source hashes are recorded; candidate sources remained
unchanged throughout the full run. This is the whole-session, unversioned-corpus
protocol described above, not a before/after comparison with historical results.

| Metric | Result |
| --- | --- |
| Hit@1 | 78.94% (371/470) |
| Hit@5 | 94.68% (445/470) |
| Macro session recall@5 | 89.07% |
| All-answer-sessions@5 | 81.91% (385/470) |

These are retrieval metrics, not answer accuracy. No LLM/QA evaluation is included.
