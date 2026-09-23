#!/usr/bin/env python3
"""Bounded, synthetic sustained-write/restart/restore proof for issue #62.

Runs only against a fresh temporary Chroma store and queue. No external model,
network service, private memory, or credential is used.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["MEM0_TELEMETRY"] = "False"
os.environ["FIDELIS_RETRIEVAL_TELEMETRY"] = "0"
os.environ["COGITO_RELATION_ENVELOPES_V1"] = "1"
os.environ["COGITO_TEMPORAL_V1"] = "1"


def _load_dependencies():
    from mem0.vector_stores.chroma import ChromaDB

    from fidelis import degrade, server
    from fidelis.degrade import configure_temporal, replay_queue

    return ChromaDB, degrade, server, configure_temporal, replay_queue


ChromaDB, degrade, server, configure_temporal, replay_queue = _load_dependencies()

N_DIRECT = 120
N_CORRECTIONS = 12
N_QUEUED = 16
N_DUPLICATES = 20
USER = "issue-62-synthetic-agent"
NAMESPACE = uuid.UUID("a62e4e89-bd75-5bb1-a9fc-b00000000062")


class DeterministicEmbedder:
    DIM = 512

    def __init__(self):
        self.fail = False

    @classmethod
    def vector(cls, text):
        vec = [0.0] * cls.DIM
        for token in re.findall(r"[a-z0-9]+", str(text).lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % cls.DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / norm for value in vec]

    def embed(self, text, *args, **kwargs):
        del args, kwargs
        if self.fail:
            raise ConnectionError("synthetic embedding outage")
        return self.vector(text)


class Harness:
    def __init__(self, memory, cfg, port):
        self.memory, self.cfg, self.port = memory, cfg, port

    def post(self, route, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{route}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            result = error.code, json.loads(error.read())
        return result[0], result[1], (time.perf_counter() - start) * 1000


def percentiles(values):
    vals = sorted(values)

    def p(q):
        return round(vals[min(len(vals) - 1, math.ceil(q * len(vals)) - 1)], 3)

    return {"p50_ms": p(0.50), "p95_ms": p(0.95)}


def peak_rss_bytes():
    # macOS reports bytes; other supported Unix systems report KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def restore_worker(source, queue, restore, manifest_path):
    """Fresh-process restart, backup, restore, replay, and recall verification."""
    source, queue, restore = Path(source), Path(queue), Path(restore)
    source_status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/fidelis"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status:
        raise RuntimeError("proof requires an unchanged tracked src/fidelis tree")
    manifest = json.loads(Path(manifest_path).read_text())
    expected_before = {record["id"]: record["text"] for record in manifest["direct"]}
    expected_before.update(
        {record["superseder_id"]: record["text"] for record in manifest["corrections"]}
    )
    expected = dict(expected_before)
    expected.update({record["id"]: record["text"] for record in manifest["queued"]})
    rss_before = peak_rss_bytes()

    # Reopen the writer's persistent store in this new interpreter before making
    # the backup. Confirm queued items also survived the process boundary.
    source_memory = SimpleNamespace(
        embedding_model=DeterministicEmbedder(),
        vector_store=ChromaDB(collection_name="issue-62-synthetic", path=str(source)),
    )
    assert source_memory.vector_store.collection.count() == len(expected_before)
    source_rows = source_memory.vector_store.collection.get(include=["metadatas"])
    source_payloads = dict(zip(source_rows["ids"], source_rows["metadatas"]))
    assert set(source_payloads) == set(expected_before)
    assert all(source_payloads[rid].get("data") == text for rid, text in expected_before.items())
    queued_before = list(queue.glob("*.json"))
    assert len(queued_before) == len(manifest["queued"])
    del source_memory
    gc.collect()

    shutil.copytree(source, restore / "store")
    shutil.copy2(Path(str(source) + ".temporal.sqlite"), restore / "store.temporal.sqlite")
    shutil.copytree(queue, restore / "queue")
    restored_queue = restore / "queue"
    os.environ["COGITO_QUEUE_DIR"] = str(restored_queue)
    os.environ["FIDELIS_QUEUE_DIR"] = str(restored_queue)
    restored_memory = SimpleNamespace(
        embedding_model=DeterministicEmbedder(),
        vector_store=ChromaDB(collection_name="issue-62-synthetic", path=str(restore / "store")),
    )
    cfg = {
        "user_id": USER,
        "collection": "issue-62-synthetic",
        "store_path": str(restore / "store"),
        "write_gate": True,
    }
    configure_temporal(restore / "store", cfg)
    replay = replay_queue(restored_memory, USER)
    assert replay.get("replayed") == len(manifest["queued"])
    assert replay.get("remaining") == 0 and replay.get("dead_lettered") == 0, replay
    index = degrade.temporal_index()
    assert index is not None
    edge_pairs = {(target, superseder) for target, superseder, _ in index.edges()}
    expected_edges = {
        (record["target_id"], record["superseder_id"]) for record in manifest["corrections"]
    }
    assert expected_edges <= edge_pairs
    assert restored_memory.vector_store.collection.count() == len(expected)
    rows = restored_memory.vector_store.collection.get(include=["metadatas"])
    payloads = dict(zip(rows["ids"], rows["metadatas"]))
    assert set(payloads) == set(expected)
    assert all(payloads[rid].get("data") == text for rid, text in expected.items())

    recall_ms = []
    for item in manifest["direct"][:60]:
        start = time.perf_counter()
        result = restored_memory.vector_store.search(
            query=item["text"],
            vectors=[DeterministicEmbedder.vector(item["text"])],
            top_k=5,
            filters={"user_id": USER},
        )
        recall_ms.append((time.perf_counter() - start) * 1000)
        assert any(row.payload.get("data") == item["text"] for row in result), item["id"]

    remaining = list(restored_queue.glob("*.json"))
    dead = (
        list((restored_queue / "dead").glob("*.json")) if (restored_queue / "dead").exists() else []
    )
    store_bytes = sum(
        path.stat().st_size for path in (restore / "store").rglob("*") if path.is_file()
    )
    queue_bytes = sum(
        path.stat().st_size for path in (restore / "queue").rglob("*") if path.is_file()
    )
    request_count = N_DIRECT + N_DUPLICATES + N_CORRECTIONS + N_QUEUED
    errors = manifest["write_errors"]
    rss_after = peak_rss_bytes()
    report = {
        "proof": "bounded synthetic sustained-write restart and restore",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "proof_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_tree_clean": True,
        "scope": {
            "direct": N_DIRECT,
            "duplicate_attempts": N_DUPLICATES,
            "corrections": N_CORRECTIONS,
            "queued_during_embedding_outage": N_QUEUED,
        },
        "accounting": {
            "unique_records_after_restore": len(expected),
            "duplicate_attempts": len(manifest["duplicates"]),
            "correction_edges_expected": len(expected_edges),
            "correction_edges_observed": sum(edge in edge_pairs for edge in expected_edges),
            "queued_before_restart": len(queued_before),
            "replayed": replay.get("replayed"),
            "queue_remaining": len(remaining),
            "dead_lettered": len(dead),
            "write_errors": errors,
            "requests": request_count,
            "error_rate": errors / request_count,
            "maximum_queue_depth": len(queued_before),
        },
        "identifiers": {
            "direct": [item["id"] for item in manifest["direct"]],
            "duplicates": manifest["duplicates"],
            "corrections": manifest["corrections"],
            "queued": manifest["queued"],
        },
        "latency": {
            "store_http": percentiles(manifest["write_ms"]),
            "restored_vector_recall": percentiles(recall_ms),
        },
        "growth_bytes": {
            "restored_chroma_store": store_bytes,
            "restored_queue": queue_bytes,
            "restored_store_plus_queue": store_bytes + queue_bytes,
            "process_peak_rss_before_bytes": rss_before,
            "process_peak_rss_after_bytes": rss_after,
            "process_peak_rss_delta_bytes": max(0, rss_after - rss_before),
        },
        "recall_queries": len(recall_ms),
        "restored_exact_texts_verified": len(expected),
        "limitations": [
            "Synthetic local workload; timings and storage and peak-RSS growth are machine-specific.",
            "This bounded proof does not establish long-duration uptime or solve issue #60's store-wide budget.",
            "Recall uses the installed Chroma vector-store search API; it does not validate HTTP /query's separate top_k adapter.",
        ],
    }
    configure_temporal(None)
    return report


def main():
    with tempfile.TemporaryDirectory(prefix="fidelis-issue-62-") as temp:
        base = Path(temp)
        store = base / "store"
        queue = base / "queue"
        restore = base / "restore"
        os.environ["COGITO_QUEUE_DIR"] = str(queue)
        os.environ["FIDELIS_QUEUE_DIR"] = str(queue)
        embedder = DeterministicEmbedder()
        memory = SimpleNamespace(
            embedding_model=embedder,
            vector_store=ChromaDB(collection_name="issue-62-synthetic", path=str(store)),
        )
        cfg = {
            "user_id": USER,
            "collection": "issue-62-synthetic",
            "store_path": str(store),
            "recall_limit": 10,
            "ephemera_filter": False,
            "supersession_pointers_path": None,
            "ollama_url": "http://127.0.0.1:9",
            "filter_endpoint": "",
        }
        configure_temporal(store, cfg)
        httpd = server._BoundedThreadingHTTPServer(
            ("127.0.0.1", 0), server.make_handler(memory, cfg)
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        h = Harness(memory, cfg, httpd.server_port)
        write_ms = []
        errors = 0
        direct = []
        correction_pairs = []
        queued = []
        duplicates = []
        try:
            for i in range(N_DIRECT):
                rid = uuid.uuid5(NAMESPACE, f"direct-{i}").hex
                text = f"Synthetic issue 62 record {i:04d} describes component {i % 17} status {i % 23}."
                status, body, elapsed = h.post("/store", {"text": text, "id": rid})
                write_ms.append(elapsed)
                if status != 200 or body.get("status") != "stored":
                    errors += 1
                    raise AssertionError((status, body))
                direct.append({"id": body["id"], "text": text})
            for i in range(N_DUPLICATES):
                prior = direct[i]
                status, body, elapsed = h.post(
                    "/store",
                    {"text": prior["text"], "id": uuid.uuid5(NAMESPACE, f"duplicate-{i}").hex},
                )
                write_ms.append(elapsed)
                if (
                    status != 200
                    or body.get("status") != "duplicate"
                    or body.get("id") != prior["id"]
                ):
                    errors += 1
                    raise AssertionError((status, body))
                duplicates.append(
                    {
                        "attempted_id": uuid.uuid5(NAMESPACE, f"duplicate-{i}").hex,
                        "existing_id": body["id"],
                    }
                )
            for i in range(N_CORRECTIONS):
                prior = direct[i]
                rid = uuid.uuid5(NAMESPACE, f"correction-{i}").hex
                text = f"Synthetic correction {i:04d} supersedes {prior['id']} with revised component status {i % 19}."
                status, body, elapsed = h.post(
                    "/store", {"text": text, "id": rid, "supersedes": [prior["id"]]}
                )
                write_ms.append(elapsed)
                if status != 200 or body.get("status") != "stored":
                    errors += 1
                    raise AssertionError((status, body))
                correction_pairs.append(
                    {"target_id": prior["id"], "superseder_id": body["id"], "text": text}
                )
            embedder.fail = True
            for i in range(N_QUEUED):
                rid = uuid.uuid5(NAMESPACE, f"queued-{i}").hex
                text = f"Synthetic queued record {i:04d} survives a simulated embedding outage with category {i % 7}."
                status, body, elapsed = h.post("/store", {"text": text, "id": rid})
                write_ms.append(elapsed)
                if status != 200 or body.get("status") != "queued":
                    errors += 1
                    raise AssertionError((status, body))
                queued.append({"id": body["id"], "text": text})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)
        before_restart_count = memory.vector_store.collection.count()
        assert before_restart_count == N_DIRECT + N_CORRECTIONS
        assert len(list(queue.glob("*.json"))) == N_QUEUED
        configure_temporal(None)
        # Drop the HTTP handler and writer-side Chroma client before launching
        # a fresh verifier, so the child never depends on live parent handles.
        del h, httpd, thread, memory, embedder
        gc.collect()
        manifest_path = base / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "direct": direct,
                    "duplicates": duplicates,
                    "corrections": correction_pairs,
                    "queued": queued,
                    "write_ms": write_ms,
                    "write_errors": errors,
                },
                sort_keys=True,
            )
        )
        worker_env = os.environ.copy()
        worker_env["COGITO_QUEUE_DIR"] = str(queue)
        worker_env["FIDELIS_QUEUE_DIR"] = str(queue)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--restore-worker",
                str(store),
                str(queue),
                str(restore),
                str(manifest_path),
            ],
            cwd=ROOT,
            env=worker_env,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--restore-worker":
        result = restore_worker(*sys.argv[2:])
        print(json.dumps(result, sort_keys=True))
    else:
        main()
