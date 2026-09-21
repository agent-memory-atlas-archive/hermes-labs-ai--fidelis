"""PACKET F1 — write-path truthfulness (docs/TIME-AWARE-SPEC.md).

Three behaviours, all at the real HTTP `/store` boundary plus one
`fidelis.degrade.safe_add` unit test for the store-unavailable fallback:

(a) A declared `supersedes` id must exist, for the SAME user, or the write
    is rejected (400, nothing written or queued) naming the unknown id.
    Superseding an already-superseded record is allowed (chains/branches)
    but the response must say so via `superseded_targets_already_superseded`.
    If the existence lookup itself raises (store unavailable), the write
    must not be rejected -- it falls through to the normal
    queue-on-dependency-failure path, and replay re-validates before
    inserting, dead-lettering with a clear `last_error` if the id still
    does not exist.
(b) A `stored` `/store` response must echo `recorded_at` and any of
    `valid_from`/`valid_to`/`event_at`/`supersedes` that were declared, in
    normalised ISO form, so a caller can confirm without a recall.
(c) A caller-supplied `id` that is NOT going to be honoured (relation
    envelopes off, the default) must come back with `"id_ignored": true`.

Harness: a REAL mem0 Chroma vector store under `tmp_path`, a real sqlite
temporal sidecar, and the real HTTP handler from `fidelis.server.make_handler`
on an ephemeral port -- the same real-Chroma + hashing-embedder pattern as
`tests/test_superseded_not_dropped.py` (itself mirroring
`tests/test_time_aware_acceptance.py`, which this file does not edit).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from fidelis import degrade, recall_b, recall_hybrid, server
from fidelis.degrade import configure_temporal, replay_queue, safe_add
from fidelis.relation_envelope import FEATURE_ENV

TEXT_A = "The Meridian gateway service listens on port 7711."
TEXT_B = "The Meridian gateway service was moved to port 7722."
TEXT_C = "The Meridian gateway service was moved again to port 7733."
USER = "truthfulness-agent"
UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"
TRUNCATED_ID = "a571cd95-7dbd-47e9-93ca"


# ── deterministic embedder (mirrors tests/test_superseded_not_dropped.py) ───


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
            raise ConnectionError("embedder unreachable (simulated dependency failure)")
        return self.vector(text)


class Harness:
    def __init__(self, memory, cfg: dict, queue_dir):
        self.memory = memory
        self.cfg = cfg
        self.queue_dir = queue_dir
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

    def store(self, text: str, **fields) -> tuple[int, dict]:
        return self.post("/store", {"text": text, **fields})

    def store_ok(self, text: str, **fields) -> dict:
        status, body = self.store(text, **fields)
        assert status == 200, (status, body)
        assert body.get("status") == "stored", body
        return body

    def count(self) -> int:
        return int(self.memory.vector_store.collection.count())

    def payload(self, record_id: str) -> dict:
        row = self.memory.vector_store.get(record_id)
        return dict(row.payload or {})

    def queue_files(self) -> list:
        if not self.queue_dir.exists():
            return []
        return sorted(p for p in self.queue_dir.rglob("*.json") if p.is_file())

    def dead_files(self) -> list:
        dead = self.queue_dir / "dead"
        if not dead.exists():
            return []
        return sorted(dead.glob("*.json"))


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
        vector_store=ChromaDB(collection_name="write-truthfulness", path=str(store_path)),
    )
    cfg = {
        "user_id": USER,
        "collection": "write-truthfulness",
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


# ── (a) supersedes existence validation ─────────────────────────────────────


def test_supersedes_unknown_id_is_400_and_writes_nothing(harness):
    harness.store_ok(TEXT_A)
    count = harness.count()

    status, body = harness.store(TEXT_B, supersedes=[UNKNOWN_ID])

    assert status == 400, (status, body)
    assert isinstance(body.get("error"), str)
    assert UNKNOWN_ID in body["error"]
    assert harness.count() == count
    assert harness.queue_files() == []


def test_supersedes_truncated_id_is_400_and_writes_nothing(harness):
    harness.store_ok(TEXT_A)
    count = harness.count()

    status, body = harness.store(TEXT_B, supersedes=[TRUNCATED_ID])

    assert status == 400, (status, body)
    assert TRUNCATED_ID in body["error"]
    assert harness.count() == count
    assert harness.queue_files() == []


def test_supersedes_mixed_known_and_unknown_names_only_unknown(harness):
    a = harness.store_ok(TEXT_A)
    count = harness.count()

    status, body = harness.store(TEXT_B, supersedes=[a["id"], UNKNOWN_ID])

    assert status == 400, (status, body)
    assert UNKNOWN_ID in body["error"]
    assert a["id"] not in body["error"]
    assert harness.count() == count


def test_supersedes_id_belonging_to_another_user_is_unknown(harness):
    # A record for a different user must not be a valid supersede target --
    # existence is scoped same-user, never leaked cross-user.
    status, body = harness.post("/store", {"text": TEXT_A})
    assert status == 200 and body["status"] == "stored"
    foreign_id = body["id"]
    # Overwrite user_id on the stored payload directly (simulating a record
    # that belongs to someone else) since /store always uses the server's
    # configured user_id.
    row = harness.memory.vector_store.get(foreign_id)
    payload = dict(row.payload)
    payload["user_id"] = "someone-else"
    harness.memory.vector_store.collection.update(ids=[foreign_id], metadatas=[payload])

    status, body = harness.store(TEXT_B, supersedes=[foreign_id])

    assert status == 400, (status, body)
    assert foreign_id in body["error"]


def test_supersedes_chain_of_already_superseded_is_allowed_and_flagged(harness):
    a = harness.store_ok(TEXT_A)
    b = harness.store_ok(TEXT_B, supersedes=[a["id"]])

    c = harness.store_ok(TEXT_C, supersedes=[b["id"]])

    # b was current (not yet superseded) at the time c superseded it.
    assert "superseded_targets_already_superseded" not in c
    # Now supersede A again (already superseded by B) in the same call as a
    # fresh branch off C -- A is "already superseded" and must be named.
    branch = harness.store_ok(
        "The Meridian gateway service branch note.", supersedes=[a["id"], c["id"]]
    )
    assert branch["superseded_targets_already_superseded"] == [a["id"]]


def test_store_unavailable_during_verification_falls_through_to_queue_then_dead_letters(
    harness, monkeypatch
):
    """The existence lookup itself raising must never reject a write -- it
    queues instead, and replay re-validates: if the id still does not exist,
    the record is dead-lettered with a clear last_error, never silently
    dropped and never silently accepted as current."""
    original_get = harness.memory.vector_store.get
    original_insert = harness.memory.vector_store.insert
    calls = {"n": 0}

    def flaky_get(record_id):
        calls["n"] += 1
        raise RuntimeError("chroma unavailable (simulated)")

    def flaky_insert(**kwargs):
        raise RuntimeError("chroma unavailable (simulated)")

    monkeypatch.setattr(harness.memory.vector_store, "get", flaky_get)
    # The embedder is fine; only the store's .get() (used for verification)
    # is down. Since the vector store itself is unreachable, the actual
    # insert further down must also fail for this scenario to be coherent.
    monkeypatch.setattr(harness.memory.vector_store, "insert", flaky_insert)

    status, body = harness.store(TEXT_B, supersedes=[UNKNOWN_ID])

    assert status == 200, (status, body)
    assert body["status"] == "queued", body
    assert calls["n"] >= 1
    files = [p for p in harness.queue_files()]
    assert len(files) == 1
    queued_rec = json.loads(files[0].read_text())
    assert queued_rec["metadata"]["_pending_supersedes_check"] == [UNKNOWN_ID]

    # Store comes back, id STILL does not exist -> dead-letter, not silently
    # accepted, not stuck retrying forever.
    monkeypatch.setattr(harness.memory.vector_store, "get", original_get)
    monkeypatch.setattr(harness.memory.vector_store, "insert", original_insert)
    result = replay_queue(harness.memory, user_id=USER)

    assert result["dead_lettered"] == 1, result
    assert result["replayed"] == 0
    dead = harness.dead_files()
    assert len(dead) == 1
    dead_rec = json.loads(dead[0].read_text())
    assert UNKNOWN_ID in dead_rec["last_error"]
    assert harness.count() == 0


def test_store_unavailable_during_verification_then_id_exists_on_replay(harness, monkeypatch):
    a = harness.store_ok(TEXT_A)

    def flaky_get(record_id):
        raise RuntimeError("chroma unavailable (simulated)")

    def flaky_insert(**kwargs):
        raise RuntimeError("chroma unavailable (simulated)")

    real_get = harness.memory.vector_store.get
    real_insert = harness.memory.vector_store.insert
    monkeypatch.setattr(harness.memory.vector_store, "get", flaky_get)
    monkeypatch.setattr(harness.memory.vector_store, "insert", flaky_insert)

    status, body = harness.store(TEXT_B, supersedes=[a["id"]])
    assert body["status"] == "queued", body

    monkeypatch.setattr(harness.memory.vector_store, "get", real_get)
    monkeypatch.setattr(harness.memory.vector_store, "insert", real_insert)
    result = replay_queue(harness.memory, user_id=USER)

    assert result["replayed"] == 1, result
    assert result["dead_lettered"] == 0
    hits = harness.payload(body["id"])
    assert json.loads(hits["supersedes_json"]) == [a["id"]]


def test_safe_add_direct_store_unavailable_marks_pending_check(tmp_path, monkeypatch):
    """Unit-level check of fidelis.degrade.safe_add's own contract, no HTTP."""
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("COGITO_QUEUE_DIR", str(tmp_path / "queue"))
    configure_temporal(tmp_path / "store", {})
    try:
        class _FlakyVectorStore:
            def get(self, record_id):
                raise RuntimeError("store unavailable")

            def insert(self, **kwargs):
                raise RuntimeError("store unavailable")

        memory = SimpleNamespace(
            embedding_model=SimpleNamespace(embed=lambda text, **kw: [0.0, 0.0]),
            vector_store=_FlakyVectorStore(),
        )
        result = safe_add(
            memory, "some text", "agent", kind="store",
            metadata={"supersedes": ["nonexistent-id"]},
        )
        assert result["status"] == "queued"
        qfile = next((tmp_path / "queue").glob("*.json"))
        rec = json.loads(qfile.read_text())
        assert rec["metadata"]["_pending_supersedes_check"] == ["nonexistent-id"]
    finally:
        configure_temporal(None)


# ── (b) /store echoes what was recorded ─────────────────────────────────────


def test_store_response_echoes_recorded_at_always(harness):
    body = harness.store_ok(TEXT_A)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$", body["recorded_at"])
    for key in ("valid_from", "valid_to", "event_at", "supersedes"):
        assert key not in body, f"undeclared {key} was echoed"


def test_store_response_echoes_declared_window_and_supersedes(harness):
    a = harness.store_ok(TEXT_A)

    body = harness.store_ok(
        TEXT_B,
        valid_from="2026-01-01",
        valid_to="2026-06-01T00:00:00Z",
        event_at="2026-01-15",
        supersedes=[a["id"]],
    )

    assert body["valid_from"] == "2026-01-01T00:00:00Z"
    assert body["valid_to"] == "2026-06-01T00:00:00Z"
    assert body["event_at"] == "2026-01-15T00:00:00Z"
    assert body["supersedes"] == [a["id"]]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$", body["recorded_at"])


# ── (c) id_ignored truthfulness ─────────────────────────────────────────────


def test_store_id_argument_ignored_by_default_is_flagged(harness):
    body = harness.store_ok(TEXT_A, id="my-stable-id")

    assert body["id_ignored"] is True
    assert body["id"] != "my-stable-id"


def test_store_id_argument_honoured_with_relation_envelopes_is_not_flagged(harness, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV, "1")

    body = harness.store_ok(TEXT_A, id="my-stable-id-2")

    assert body["id"] == "my-stable-id-2"
    assert "id_ignored" not in body


def test_store_without_id_argument_never_flagged(harness):
    body = harness.store_ok(TEXT_A)
    assert "id_ignored" not in body
