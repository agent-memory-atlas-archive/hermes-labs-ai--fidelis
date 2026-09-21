"""Fresh retrieval-only evaluation of the candidate's real default HTTP handler.

One whole session per record; no chunking, reranking, LLM or historical rankings.
Corpus vectors are content-addressed for reuse within this invocation/smoke->full.
Run with --data DATA --state NEW_DIR --output RESULT [--limit 5].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sqlite3
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=470)
    parser.add_argument("--port", type=int, default=19449)
    parser.add_argument("--mode", choices=("fast", "thorough"), default="fast")
    parser.add_argument("--warm-query", action="store_true")
    args = parser.parse_args()
    if args.mode == "thorough" and args.limit > 5:
        parser.error("thorough comparison is bounded to five smoke questions")
    if args.port < 19440:
        parser.error("isolated port must be >=19440")
    args.state.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="run-", dir=args.state.resolve()))
    os.environ.update(
        HOME=str(home),
        COGITO_STORE_PATH=str(home / "store"),
        FIDELIS_QUEUE_DIR=str(home / "queue"),
        FIDELIS_PORT=str(args.port),
        MEM0_TELEMETRY="false",
        ANONYMIZED_TELEMETRY="false",
    )
    from fidelis import __version__
    from fidelis.config import load, mem0_config
    from fidelis.degrade import _embed_bounded
    from fidelis.server import make_handler
    from mem0 import Memory

    dataset_bytes = args.data.read_bytes()
    dataset = json.loads(dataset_bytes)
    entries = [e for e in dataset if "_abs" not in e["question_id"]]
    assert len(dataset) == 500 and len(entries) == 470
    assert len({e["question_id"] for e in entries}) == 470
    entries = entries[: args.limit]
    config_file = home / "config.json"
    config_file.write_text("{}")
    cfg = load(config_file)
    cfg.update(store_path=str(home / "store"), collection="longmemeval_default")
    memory = Memory.from_config(mem0_config(cfg))

    # Fail closed if any generation path is accidentally reached.
    def forbidden(*a, **kw):
        raise RuntimeError("LLM generation is forbidden in this evaluation")

    memory.llm.generate_response = forbidden
    tags = json.load(urllib.request.urlopen(cfg["ollama_url"] + "/api/tags"))
    model = next(
        m
        for m in tags["models"]
        if m["name"] == cfg["embed_model"] + ":latest" or m["name"] == cfg["embed_model"]
    )
    model = {k: model[k] for k in ("name", "digest", "details")}
    key_prefix = json.dumps(model, sort_keys=True)
    cache = sqlite3.connect(args.state / "embeddings.sqlite")
    cache.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
    )
    source = Path(__file__).resolve().parents[1] / "src" / "fidelis"
    hashes = {
        str(p.relative_to(source)): digest(p.read_bytes()) for p in sorted(source.rglob("*.py"))
    }
    result = {
        "schema": "fidelis.default-retrieval-eval/v1",
        "version": __version__,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
            "file": args.data.name,
            "sha256": digest(dataset_bytes),
            "total": 500,
            "eligible": 470,
            "selection": "question_id does not contain '_abs'; original order",
        },
        "source_sha256": hashes,
        "source_tree_sha256": digest(json.dumps(hashes, sort_keys=True).encode()),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "model": model,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("mem0ai", "chromadb", "ollama", "numpy", "pydantic")
        },
        "protocol": {
            "route": "POST /query"
            if args.mode == "fast"
            else "POST /recall_hybrid (tier=zero_llm, top_k=5)",
            "mode": args.mode,
            "request": '{"text": question}; default limit=5'
            if args.mode == "fast"
            else '{"text": question, "limit":5, "top_k":5, "tier":"zero_llm"}',
            "corpus": "one full role-prefixed session per record; no dates or gold labels in text",
            "indexing": "direct mem0 Chroma insertion with candidate _embed_bounded; write gate not evaluated",
            "embedding_cache": "content + model digest + embedder and bounded function source hashes; no retrieval output cache",
            "llm_calls": 0,
            "temporal_metadata": "absent; legacy-compatible unversioned corpus",
            "comparison": "not directly comparable with historical chunked hybrid pipeline or QA accuracy",
        },
        "rows": [],
    }
    embed_code = digest(
        __import__("inspect").getsource(type(memory.embedding_model)).encode()
        + __import__("inspect").getsource(_embed_bounded).encode()
    )

    def save():
        rows = result["rows"]
        n = len(rows)
        result["questions_evaluated"] = n
        result["errors"] = sum(bool(r.get("error")) for r in rows)
        result["metrics"] = (
            {
                m: sum(r[m] for r in rows) / n
                for m in ("hit_at_1", "hit_at_5", "recall_at_5", "all_answer_sessions_at_5")
            }
            if n
            else {}
        )
        result["complete"] = n == len(entries)
        if n:
            result["query_seconds_median"] = statistics.median(
                r.get("query_seconds", 0) for r in rows
            )
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for number, entry in enumerate(entries, 1):
        qid = entry["question_id"]
        gold = set(entry["answer_session_ids"])
        row = {
            "question_id": qid,
            "question_type": entry["question_type"],
            "gold_session_ids": sorted(gold),
            "ranked_session_ids": [],
            "hit_at_1": 0,
            "hit_at_5": 0,
            "recall_at_5": 0.0,
            "all_answer_sessions_at_5": 0,
        }
        server = None
        try:
            cfg["user_id"] = qid
            vectors, ids, payloads = [], [], []
            for i, (sid, session) in enumerate(
                zip(entry["haystack_session_ids"], entry["haystack_sessions"])
            ):
                text = "\n".join(f"{t['role']}: {t['content']}" for t in session)
                key = digest((key_prefix + embed_code + text).encode())
                found = cache.execute(
                    "SELECT vector FROM embeddings WHERE key=?", (key,)
                ).fetchone()
                if found:
                    vector = json.loads(found[0])
                else:
                    vector = _embed_bounded(memory.embedding_model, text, memory_action="add")
                    cache.execute("INSERT INTO embeddings VALUES (?,?)", (key, json.dumps(vector)))
                    cache.commit()
                vectors.append(vector)
                ids.append(f"{qid}-{i}")
                payloads.append({"data": text, "user_id": qid, "session_id": sid})
            memory.vector_store.insert(vectors=vectors, ids=ids, payloads=payloads)
            server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(memory, cfg))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            request = urllib.request.Request(
                f"http://127.0.0.1:{args.port}"
                + ("/query" if args.mode == "fast" else "/recall_hybrid"),
                data=json.dumps(
                    {"text": entry["question"]}
                    if args.mode == "fast"
                    else {"text": entry["question"], "limit": 5, "top_k": 5, "tier": "zero_llm"}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            row["query_seconds"] = time.perf_counter() - start
            if args.warm_query:
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=120) as response:
                    warm_body = json.load(response)
                row["warm_query_seconds"] = time.perf_counter() - start
                row["warm_ranking_unchanged"] = warm_body.get("memories") == body.get("memories")
            row["method"] = body.get("method", "default-query")
            if body.get("error"):
                raise RuntimeError(body["error"])
            mapping = dict(zip(ids, entry["haystack_session_ids"]))
            ranking = [mapping[m["id"]] for m in body["memories"]]
            row["ranked_session_ids"] = ranking
            row["scores"] = [m.get("score") for m in body["memories"]]
            row.update(
                hit_at_1=int(bool(gold & set(ranking[:1]))),
                hit_at_5=int(bool(gold & set(ranking[:5]))),
                recall_at_5=len(gold & set(ranking[:5])) / len(gold),
                all_answer_sessions_at_5=int(gold <= set(ranking[:5])),
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if server:
                server.shutdown()
                server.server_close()
                thread.join()
        result["rows"].append(row)
        save()
        print(
            f"{number}/{len(entries)} {qid} hit@5={row['hit_at_5']} error={row.get('error', 'none')}",
            flush=True,
        )
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["source_unchanged_during_run"] = hashes == {
        str(p.relative_to(source)): digest(p.read_bytes()) for p in sorted(source.rglob("*.py"))
    }
    result["valid"] = (
        result["complete"] and not result["errors"] and result["source_unchanged_during_run"]
    )
    save()
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
