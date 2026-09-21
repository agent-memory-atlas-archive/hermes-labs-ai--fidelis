"""PACKET F2 — backend read endpoints for navigation (docs/TIME-AWARE-SPEC.md).

`/get` (F2a), `/recent` (F2b), `/stats` (F2c): all read-only, same-user where
applicable, sidecar-backed, fail-open when the sidecar is unavailable.

Harness: a REAL mem0 Chroma vector store under `tmp_path`, a real sqlite
temporal sidecar, and the real HTTP handler from `fidelis.server.make_handler`
on an ephemeral port -- the same real-Chroma + hashing-embedder pattern as
`tests/test_superseded_not_dropped.py` (itself mirroring
`tests/test_time_aware_acceptance.py`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from fidelis import degrade, recall_b, recall_hybrid, server
from fidelis.degrade import configure_temporal
from fidelis.relation_envelope import FEATURE_ENV

TEXT_A = "The Corvid gateway service listens on port 6611."
TEXT_B = "The Corvid gateway service was moved to port 6622."
TEXT_C = "The Corvid gateway service was moved again to port 6633."
USER = "browse-agent"


class HashingEmbedder:
    DIM = 512

    def __init__(self) -> None:
        self.fail = False

    @classmethod
    def vector(cls, text: str) -> list[float]:
        vec = [0.0] * cls.DIM
        for token in re.findall(r"[a-z0-9]+", str(text).lower()):
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % cls.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed(self, text, *args, **kwargs):
        del args, kwargs
        if self.fail:
            raise ConnectionError("embedder unreachable (simulated)")
        return self.vector(text)


class Harness:
    def __init__(self, memory, cfg: dict, queue_dir: Path):
        self.memory = memory
        self.cfg = cfg
        self.queue_dir = queue_dir
        self.store_path = Path(cfg["store_path"])
        self.sidecar = self.store_path.parent / (self.store_path.name + ".temporal.sqlite")
        self.httpd = None
        self.port = 0

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as err:
            raw = err.read()
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, {"_raw": raw.decode("utf-8", "replace")}

    def get(self, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=30
            ) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as err:
            raw = err.read()
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, {"_raw": raw.decode("utf-8", "replace")}

    def store_ok(self, text: str, **fields) -> dict:
        status, body = self.post("/store", {"text": text, **fields})
        assert status == 200, (status, body)
        assert body.get("status") == "stored", body
        return body


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from mem0.vector_stores.chroma import ChromaDB

    home = tmp_path / "home"
    home.mkdir()
    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COGITO_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("FIDELIS_RETRIEVAL_TELEMETRY", "0")
    monkeypatch.setenv("FIDELIS_RETRIEVAL_TELEMETRY_LOG", str(tmp_path / "telemetry.jsonl"))
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.delenv("COGITO_TEMPORAL_V1", raising=False)
    monkeypatch.delenv(FEATURE_ENV, raising=False)
    monkeypatch.delenv("COGITO_HERMENEUTICS_EVIDENCE_STATUS", raising=False)
    from fidelis import telemetry

    monkeypatch.setattr(telemetry, "_LOG_PATH", tmp_path / "escalation.log", raising=False)
    monkeypatch.setattr(recall_b, "_batch_embed", lambda texts, cfg: None)
    monkeypatch.setattr(
        recall_hybrid, "_embed_docs",
        lambda texts, cfg: [HashingEmbedder.vector(t) for t in texts],
    )
    monkeypatch.setattr(
        recall_hybrid, "_embed_queries",
        lambda texts, cfg: [HashingEmbedder.vector(t) for t in texts],
    )

    store_path = tmp_path / "store"
    embedder = HashingEmbedder()
    memory = SimpleNamespace(
        embedding_model=embedder,
        vector_store=ChromaDB(collection_name="browse-tests", path=str(store_path)),
    )
    cfg = {
        "user_id": USER,
        "collection": "browse-tests",
        "store_path": str(store_path),
        "recall_limit": 30,
        "ephemera_filter": False,
        "supersession_pointers_path": None,
        "ollama_url": "http://127.0.0.1:9",
        "filter_endpoint": "",
    }
    h = Harness(memory, cfg, queue_dir)
    configure_temporal(store_path, cfg)
    httpd = server._BoundedThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(memory, cfg))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    h.httpd = httpd
    h.port = int(httpd.server_port)
    try:
        yield h
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)
        configure_temporal(None)
        assert degrade.temporal_index() is None


# ── /get (F2a) ───────────────────────────────────────────────────────────────


def test_get_unknown_id_is_404(harness):
    status, body = harness.post("/get", {"id": "no-such-id"})
    assert status == 404, (status, body)
    assert "no-such-id" in body["error"]


def test_get_missing_id_argument_is_400(harness):
    status, body = harness.post("/get", {})
    assert status == 400, (status, body)


def test_get_isolated_record_has_empty_chains(harness):
    a = harness.store_ok(TEXT_A)
    status, body = harness.post("/get", {"id": a["id"]})
    assert status == 200, (status, body)
    assert body["id"] == a["id"]
    assert body["text"] == TEXT_A
    assert body["recorded_at"] == a["recorded_at"]
    assert body["temporal"]["status"] == "current"
    assert body["supersedes"] == []
    assert body["superseded_by"] == []
    assert body["index"] == "ok"


def test_get_chain_both_directions(harness):
    a = harness.store_ok(TEXT_A)
    b = harness.store_ok(TEXT_B, supersedes=[a["id"]])
    c = harness.store_ok(TEXT_C, supersedes=[b["id"]])

    _, from_a = harness.post("/get", {"id": a["id"]})
    assert from_a["supersedes"] == []
    forward_ids = [e["id"] for e in from_a["superseded_by"]]
    assert forward_ids == [b["id"], c["id"]], from_a["superseded_by"]  # oldest first, newest (current) last
    assert from_a["superseded_by"][0]["status"] == "superseded"
    assert from_a["superseded_by"][1]["status"] == "current"
    assert from_a["superseded_by"][1]["text"] == TEXT_C

    _, from_c = harness.post("/get", {"id": c["id"]})
    assert from_c["superseded_by"] == []
    backward_ids = [e["id"] for e in from_c["supersedes"]]
    assert backward_ids == [b["id"], a["id"]], from_c["supersedes"]  # nearest ancestor first, oldest last
    assert from_c["supersedes"][0]["text"] == TEXT_B
    assert from_c["supersedes"][1]["text"] == TEXT_A


def test_get_branching_superseded_by(harness):
    a = harness.store_ok(TEXT_A)
    b = harness.store_ok(TEXT_B, supersedes=[a["id"]])
    d = harness.store_ok("The Corvid gateway service branch note.", supersedes=[a["id"]])

    _, from_a = harness.post("/get", {"id": a["id"]})
    ids = {e["id"] for e in from_a["superseded_by"]}
    assert ids == {b["id"], d["id"]}


def test_get_other_user_record_is_404(harness):
    a = harness.store_ok(TEXT_A)
    row = harness.memory.vector_store.get(a["id"])
    payload = dict(row.payload)
    payload["user_id"] = "someone-else"
    harness.memory.vector_store.collection.update(ids=[a["id"]], metadatas=[payload])

    status, body = harness.post("/get", {"id": a["id"]})
    assert status == 404, (status, body)


def test_get_cycle_safe(harness):
    """A malformed/adversarial mutual-supersession declaration must not hang
    the chain walk -- inserted directly, bypassing the API (which forbids
    only literal self-supersession, not a mutual A<->B cycle)."""
    vector = HashingEmbedder.vector(TEXT_A)
    harness.memory.vector_store.insert(
        vectors=[vector],
        payloads=[{
            "data": TEXT_A, "user_id": USER, "temporal_schema": "fidelis.temporal/v1",
            "recorded_at": "2026-01-01T00:00:00Z", "recorded_at_source": "write",
            "content_sha256": hashlib.sha256(TEXT_A.encode()).hexdigest(),
            "supersedes_json": json.dumps(["cycle-b"]),
        }],
        ids=["cycle-a"],
    )
    harness.memory.vector_store.insert(
        vectors=[HashingEmbedder.vector(TEXT_B)],
        payloads=[{
            "data": TEXT_B, "user_id": USER, "temporal_schema": "fidelis.temporal/v1",
            "recorded_at": "2026-01-02T00:00:00Z", "recorded_at_source": "write",
            "content_sha256": hashlib.sha256(TEXT_B.encode()).hexdigest(),
            "supersedes_json": json.dumps(["cycle-a"]),
        }],
        ids=["cycle-b"],
    )
    degrade.temporal_index().record_write(
        "cycle-a", hashlib.sha256(TEXT_A.encode()).hexdigest(), "2026-01-01T00:00:00Z", ["cycle-b"],
    )
    degrade.temporal_index().record_write(
        "cycle-b", hashlib.sha256(TEXT_B.encode()).hexdigest(), "2026-01-02T00:00:00Z", ["cycle-a"],
    )

    status, body = harness.post("/get", {"id": "cycle-a"})
    assert status == 200, (status, body)
    # Terminates (no timeout/hang) and does not fabricate an unbounded chain.
    assert len(body["supersedes"]) <= 16
    assert len(body["superseded_by"]) <= 16


def test_get_sidecar_unavailable_still_returns_record(harness):
    a = harness.store_ok(TEXT_A)
    configure_temporal(None)  # simulate sidecar unavailable

    status, body = harness.post("/get", {"id": a["id"]})

    assert status == 200, (status, body)
    assert body["text"] == TEXT_A
    assert body["index"] == "unavailable"
    assert body["supersedes"] == []
    assert body["superseded_by"] == []


# ── /recent (F2b) ────────────────────────────────────────────────────────────


def test_recent_newest_first_default_limit(harness):
    a = harness.store_ok(TEXT_A)
    b = harness.store_ok(TEXT_B)
    c = harness.store_ok(TEXT_C)

    status, body = harness.post("/recent", {"limit": 10})

    assert status == 200, (status, body)
    ids = [r["id"] for r in body["records"]]
    assert ids == [c["id"], b["id"], a["id"]]
    assert body["records"][0]["text"] == TEXT_C
    assert body["index"] == "ok"


def test_recent_respects_limit(harness):
    harness.store_ok(TEXT_A)
    harness.store_ok(TEXT_B)
    harness.store_ok(TEXT_C)

    status, body = harness.post("/recent", {"limit": 1})
    assert status == 200
    assert len(body["records"]) == 1
    assert body["records"][0]["text"] == TEXT_C


def test_recent_since_lower_bound(harness):
    a = harness.store_ok(TEXT_A)
    b = harness.store_ok(TEXT_B)

    status, body = harness.post("/recent", {"limit": 10, "since": b["recorded_at"]})
    assert status == 200, (status, body)
    ids = [r["id"] for r in body["records"]]
    assert ids == [b["id"]]
    assert a["id"] not in ids


def test_recent_invalid_since_is_400(harness):
    status, body = harness.post("/recent", {"limit": 10, "since": "not a date"})
    assert status == 400, (status, body)
    assert body.get("error")


def test_recent_kind_corrections_only(harness):
    a = harness.store_ok(TEXT_A)
    harness.store_ok(TEXT_B)  # not a correction
    c = harness.store_ok(TEXT_C, supersedes=[a["id"]])

    status, body = harness.post("/recent", {"limit": 10, "kind": "corrections"})

    assert status == 200, (status, body)
    assert len(body["records"]) == 1
    assert body["records"][0]["id"] == c["id"]
    assert body["records"][0]["supersedes"] == [a["id"]]


def test_recent_invalid_kind_is_400(harness):
    status, body = harness.post("/recent", {"limit": 10, "kind": "bogus"})
    assert status == 400, (status, body)


def test_recent_invalid_limit_is_400(harness):
    status, body = harness.post("/recent", {"limit": 0})
    assert status == 400, (status, body)
    status, body = harness.post("/recent", {"limit": "many"})
    assert status == 400, (status, body)


def test_recent_sidecar_unavailable_says_so(harness):
    harness.store_ok(TEXT_A)
    configure_temporal(None)

    status, body = harness.post("/recent", {"limit": 10})

    assert status == 200, (status, body)
    assert body["records"] == []
    assert body["index"] == "unavailable"
    assert body.get("note")


# ── /stats (F2c) ─────────────────────────────────────────────────────────────


def test_stats_basic_counts(harness):
    a = harness.store_ok(TEXT_A)
    harness.store_ok(TEXT_B, supersedes=[a["id"]])
    harness.store_ok(TEXT_C, valid_to="2020-01-01T00:00:00Z")

    status, body = harness.get("/stats")

    assert status == 200, (status, body)
    assert body["total"] == 3
    assert body["indexed"] == 3
    assert body["current"] == 1
    assert body["superseded"] == 1
    assert body["expired"] == 1
    assert body["not_yet_valid"] == 0
    assert body["no_recorded_at"] == 0
    assert body["oldest_recorded_at"] is not None
    assert body["newest_recorded_at"] is not None
    assert body["queued"] == 0
    assert body["dead_letter"] == 0
    assert body["index"] == "ok"


def test_stats_counts_legacy_no_recorded_at_records(harness):
    harness.store_ok(TEXT_A)
    harness.memory.vector_store.insert(
        vectors=[HashingEmbedder.vector("legacy note")],
        payloads=[{"data": "legacy note", "user_id": USER}],
        ids=["legacy-1"],
    )

    status, body = harness.get("/stats")

    assert status == 200, (status, body)
    assert body["total"] == 2
    assert body["indexed"] == 1
    assert body["no_recorded_at"] == 1


def test_stats_never_blocks_when_sidecar_unavailable(harness):
    harness.store_ok(TEXT_A)
    configure_temporal(None)

    status, body = harness.get("/stats")

    assert status == 200, (status, body)
    assert body["index"] == "unavailable"
    assert body.get("note")
    # Total (a plain vector-store count) is still honestly reported.
    assert body["total"] == 1
