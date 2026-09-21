"""Acceptance regression for docs/TIME-AWARE-SPEC.md (contract v1).

Written from the contract, not from the implementation. Every assertion here
traces to a clause of the spec: the "Acceptance scenario" steps 1-10 and the
"Invariants" section.

Harness: a REAL mem0 Chroma vector store persisted under ``tmp_path`` (the same
``mem0.vector_stores.chroma.ChromaDB`` class the server boots with), a real
sqlite temporal sidecar next to it, and the real HTTP handler from
``fidelis.server.make_handler`` on an ephemeral port. Only the embedding model
is replaced — by a deterministic bag-of-words hashing embedder — so nothing
needs Ollama, a network, or an LLM, yet recall genuinely ranks by similarity
and ids/payloads round-trip through a real store.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from fidelis import degrade, mcp_server, recall_b, recall_hybrid, server
from fidelis.degrade import configure_temporal, replay_queue, safe_add
from fidelis.relation_envelope import FEATURE_ENV
from fidelis.temporal import parse_instant
from fidelis.temporal_recall import temporal_view

USER = "acceptance-agent"
TEXT_A = "Fidelis server listens on port 19420."
TEXT_B = "Fidelis server listens on port 19555."
QUERY = "which port does the fidelis server listen on"
RECALL_ENDPOINTS = ["/query", "/recall", "/recall_hybrid"]
RECORDED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
CALLER_TEMPORAL_KEYS = ("event_at", "valid_from", "valid_to", "supersedes_json", "source")


# ── deterministic embedder ──────────────────────────────────────────────────


class HashingEmbedder:
    """Bag-of-words feature hashing, L2-normalised. No model, no network."""

    DIM = 512

    def __init__(self) -> None:
        self.fail = False

    @classmethod
    def vector(cls, text: str) -> list[float]:
        vec = [0.0] * cls.DIM
        for token in re.findall(r"[a-z0-9]+", str(text).lower()):
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]  # listens ~ listen
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


# ── harness ─────────────────────────────────────────────────────────────────


class Harness:
    def __init__(self, tmp_path: Path, memory, embedder, cfg, queue_dir: Path):
        self.tmp_path = tmp_path
        self.memory = memory
        self.embedder = embedder
        self.cfg = cfg
        self.queue_dir = queue_dir
        self.store_path = Path(cfg["store_path"])
        self.sidecar = self.store_path.parent / (self.store_path.name + ".temporal.sqlite")
        self.httpd = None
        self.port = 0

    # HTTP -------------------------------------------------------------------
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

    def recall(self, endpoint: str, query: str, **extra) -> list[dict]:
        body = {"text": query, "limit": 10, **extra}
        if endpoint == "/recall_hybrid":
            body["top_k"] = 10
        status, response = self.post(endpoint, body)
        assert status == 200, (endpoint, status, response)
        return response["memories"]

    # store / queue introspection ---------------------------------------------
    def count(self) -> int:
        return int(self.memory.vector_store.collection.count())

    def payload(self, record_id: str) -> dict:
        row = self.memory.vector_store.get(record_id)
        return dict(row.payload or {})

    def queue_files(self) -> list[Path]:
        if not self.queue_dir.exists():
            return []
        return sorted(p for p in self.queue_dir.rglob("*") if p.is_file())


def _by_id(memories: list[dict]) -> dict[str, dict]:
    return {m["id"]: m for m in memories if m.get("id")}


def _position(memories: list[dict], record_id: str) -> int:
    ids = [m.get("id") for m in memories]
    assert record_id in ids, f"record {record_id} missing from recall: {memories}"
    return ids.index(record_id)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from mem0.vector_stores.chroma import ChromaDB

    # Isolation: nothing may touch the real home, queue, or telemetry logs.
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

    # No Ollama: secondary embedding calls inside the retrievers are made
    # deterministic (same approach as tests/test_relation_envelope_e2e.py).
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
        vector_store=ChromaDB(collection_name="time-aware-acceptance", path=str(store_path)),
    )
    cfg = {
        "user_id": USER,
        "collection": "time-aware-acceptance",
        "store_path": str(store_path),
        "recall_limit": 30,
        "ephemera_filter": False,
        "supersession_pointers_path": None,
        "ollama_url": "http://127.0.0.1:9",  # never reachable; never contacted
        "filter_endpoint": "",
    }
    h = Harness(tmp_path, memory, embedder, cfg, queue_dir)
    configure_temporal(store_path, cfg)
    httpd = server._BoundedThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(memory, cfg)
    )
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
        # Reset module-level state so nothing leaks into other tests.
        configure_temporal(None)
        assert degrade.temporal_index() is None


def _store_a_then_b(h: Harness) -> tuple[dict, dict]:
    a = h.store_ok(TEXT_A)
    time.sleep(0.05)  # leave a real clock gap for the as_of midpoint
    b = h.store_ok(TEXT_B, supersedes=[a["id"]])
    return a, b


def _midpoint(a: dict, b: dict) -> str:
    ta, tb = parse_instant(a["recorded_at"]), parse_instant(b["recorded_at"])
    assert ta < tb, "recorded_at must be strictly ordered for the as_of step"
    mid = ta + (tb - ta) / 2
    assert ta <= mid < tb
    return mid.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds") + "Z"


# ── Acceptance scenario, steps 1-10 ─────────────────────────────────────────


def test_step01_store_returns_stored_with_recorded_at(harness):
    before = datetime.now(timezone.utc) - timedelta(seconds=2)
    status, body = harness.store(TEXT_A)
    after = datetime.now(timezone.utc) + timedelta(seconds=2)

    assert status == 200
    assert body["status"] == "stored"
    assert body.get("id")
    assert RECORDED_AT_RE.match(body["recorded_at"]), body["recorded_at"]
    assert before <= parse_instant(body["recorded_at"]) <= after
    assert harness.count() == 1

    payload = harness.payload(body["id"])
    assert payload["data"] == TEXT_A
    assert payload["user_id"] == USER
    assert payload["temporal_schema"] == "fidelis.temporal/v1"
    assert payload["recorded_at"] == body["recorded_at"]
    assert payload["recorded_at_source"] == "write"
    assert payload["content_sha256"] == hashlib.sha256(TEXT_A.strip().encode("utf-8")).hexdigest()
    # Invariant 3: nothing the caller did not declare.
    for key in CALLER_TEMPORAL_KEYS:
        assert key not in payload, f"undeclared field {key} was invented"


def test_step01b_caller_cannot_set_recorded_at(harness):
    status, body = harness.store(TEXT_A, recorded_at="1999-01-01T00:00:00Z")
    # Contract: "Callers cannot set it." Either refuse the write or ignore the value.
    if status == 200:
        assert body["status"] == "stored"
        assert parse_instant(harness.payload(body["id"])["recorded_at"]).year != 1999
        assert parse_instant(body["recorded_at"]).year != 1999
    else:
        assert status == 400
        assert harness.count() == 0


def test_step02_store_same_text_again_is_duplicate(harness):
    first = harness.store_ok(TEXT_A)
    count = harness.count()
    payload_before = harness.payload(first["id"])

    status, again = harness.store(TEXT_A)

    assert again["status"] == "duplicate", (status, again)
    assert again["id"] == first["id"]
    assert harness.count() == count == 1
    assert harness.payload(first["id"]) == payload_before
    assert harness.queue_files() == []


def test_step03_store_b_superseding_a(harness):
    a = harness.store_ok(TEXT_A)
    payload_a = harness.payload(a["id"])

    b = harness.store_ok(TEXT_B, supersedes=[a["id"]])

    assert b["id"] != a["id"]
    assert RECORDED_AT_RE.match(b["recorded_at"])
    assert harness.count() == 2
    payload_b = harness.payload(b["id"])
    assert payload_b["data"] == TEXT_B
    assert isinstance(payload_b["supersedes_json"], str)
    assert json.loads(payload_b["supersedes_json"]) == [a["id"]]
    # Append-only: declaring supersession must not rewrite the target.
    assert harness.payload(a["id"]) == payload_a


@pytest.mark.parametrize("endpoint", RECALL_ENDPOINTS)
def test_step04_recall_ranks_b_above_superseded_a(harness, endpoint):
    a, b = _store_a_then_b(harness)

    memories = harness.recall(endpoint, QUERY)

    hits = _by_id(memories)
    assert a["id"] in hits, "superseded record A was silently dropped"
    assert b["id"] in hits
    assert _position(memories, b["id"]) < _position(memories, a["id"])
    assert hits[a["id"]]["temporal"]["status"] == "superseded"
    assert hits[a["id"]]["temporal"]["superseded_by"] == [b["id"]]
    assert hits[b["id"]]["temporal"]["status"] == "current"
    assert hits[b["id"]]["temporal"]["superseded_by"] == []
    assert hits[a["id"]]["temporal"]["index"] == "ok"
    assert hits[a["id"]]["temporal"]["recorded_at"] == a["recorded_at"]
    assert hits[b["id"]]["temporal"]["recorded_at"] == b["recorded_at"]
    assert hits[a["id"]]["temporal"]["recorded_at_known"] is True
    # Both texts byte-identical to what was stored.
    assert hits[a["id"]]["text"] == TEXT_A
    assert hits[b["id"]]["text"] == TEXT_B


@pytest.mark.parametrize("endpoint", RECALL_ENDPOINTS)
def test_step05_recall_as_of_between_a_and_b(harness, endpoint):
    a, b = _store_a_then_b(harness)
    as_of = _midpoint(a, b)

    memories = harness.recall(endpoint, QUERY, as_of=as_of)

    hits = _by_id(memories)
    assert b["id"] not in hits, "B was recorded after as_of and must be absent"
    assert TEXT_B not in [m.get("text") for m in memories]
    assert a["id"] in hits
    assert hits[a["id"]]["temporal"]["status"] == "current"
    assert hits[a["id"]]["temporal"]["superseded_by"] == []
    assert parse_instant(hits[a["id"]]["temporal"]["evaluated_at"]) == parse_instant(as_of)
    assert hits[a["id"]]["text"] == TEXT_A


@pytest.mark.parametrize("endpoint", RECALL_ENDPOINTS)
def test_step05b_invalid_as_of_is_400(harness, endpoint):
    harness.store_ok(TEXT_A)
    body = {"text": QUERY, "limit": 5, "as_of": "last week"}
    status, response = harness.post(endpoint, body)
    assert status == 400, (endpoint, status, response)
    assert response.get("error")


def test_step06_expired_record_is_annotated_and_demoted(harness):
    a, b = _store_a_then_b(harness)
    text_c = "Temporary migration override: fidelis server listens on port 18000."
    past = "2020-01-01T00:00:00Z"
    c = harness.store_ok(text_c, valid_to=past)
    assert parse_instant(harness.payload(c["id"])["valid_to"]) == parse_instant(past)

    # A query C matches best, so its demotion is visible rather than incidental.
    query = "temporary migration override port for the fidelis server"
    raw_order = harness.recall("/query", query, historical=True)
    assert _position(raw_order, c["id"]) == 0, "precondition: C is the most relevant hit"

    memories = harness.recall("/query", query)
    hits = _by_id(memories)
    assert hits[c["id"]]["temporal"]["status"] == "expired"
    assert parse_instant(hits[c["id"]]["temporal"]["valid_to"]) == parse_instant(past)
    assert hits[c["id"]]["text"] == text_c
    current = [m["id"] for m in memories if m["temporal"]["status"] == "current"]
    assert b["id"] in current
    for record_id in current:
        assert _position(memories, record_id) < _position(memories, c["id"])
    # Demoted, not removed; the other stale record is still there too.
    assert a["id"] in hits


def test_step07_valid_from_after_valid_to_is_400(harness):
    harness.store_ok(TEXT_A)
    count = harness.count()

    status, body = harness.store(
        "Fidelis maintenance window for the port migration.",
        valid_from="2026-02-01", valid_to="2026-01-01",
    )

    assert status == 400, (status, body)
    assert isinstance(body.get("error"), str) and body["error"]
    assert harness.count() == count
    assert harness.queue_files() == []


def test_step08_harness_envelope_is_422_rejected(harness):
    harness.store_ok(TEXT_A)
    count = harness.count()
    queue_before = harness.queue_files()
    tag = "system" + "-reminder"
    envelope_text = "<" + tag + ">\nFidelis server listens on port 19420.\n</" + tag + ">"

    status, body = harness.store(envelope_text)

    assert status == 422, (status, body)
    assert body["status"] == "rejected"
    assert body["reason"] == "harness_envelope"
    assert harness.count() == count
    assert harness.queue_files() == queue_before == []
    texts = [m["text"] for m in harness.recall("/query", QUERY)]
    assert envelope_text not in texts


def test_step09_dependency_failure_queues_then_replay_keeps_queue_time(harness):
    harness.store_ok(TEXT_A)
    count = harness.count()
    text = "Fidelis replay queue drains every sixty seconds."

    harness.embedder.fail = True
    status, body = harness.store(text)
    assert body["status"] == "queued", (status, body)
    assert status == 200
    assert harness.count() == count, "a queued write must not be in the store yet"
    files = [p for p in harness.queue_files() if p.suffix == ".json"]
    assert len(files) == 1

    # Make queue time unmistakably different from replay time.
    record = json.loads(files[0].read_text())
    assert record["text"] == text
    assert "ts" in record, "queue record carries no original queue time"
    original_ts = 1750000000.0  # 2025-06-15T15:06:40Z
    record["ts"] = original_ts
    files[0].write_text(json.dumps(record))

    harness.embedder.fail = False
    replay_started = datetime.now(timezone.utc)
    result = replay_queue(harness.memory, user_id=USER)

    assert result["replayed"] == 1, result
    assert result["remaining"] == 0
    assert harness.count() == count + 1
    payload = harness.payload(body["id"])
    assert payload["data"] == text
    assert payload["recorded_at_source"] == "queue"
    assert RECORDED_AT_RE.match(payload["recorded_at"])
    recorded_at = parse_instant(payload["recorded_at"])
    assert recorded_at == datetime.fromtimestamp(original_ts, timezone.utc)
    assert recorded_at < replay_started - timedelta(days=30)
    assert payload["temporal_schema"] == "fidelis.temporal/v1"
    assert payload["content_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert [p for p in harness.queue_files() if p.suffix == ".json"] == []

    # And the replayed record is recallable with the original time.
    hits = _by_id(harness.recall("/query", "replay queue drains"))
    assert parse_instant(hits[body["id"]]["temporal"]["recorded_at"]) == recorded_at


def test_step09b_safe_add_direct_queue_then_replay(harness):
    """Same step through fidelis.degrade.safe_add directly (no HTTP)."""
    text = "Fidelis sidecar is a rebuildable cache."
    harness.embedder.fail = True
    queued = safe_add(harness.memory, text, USER, kind="store")
    assert queued["status"] == "queued"
    assert harness.count() == 0

    (qfile,) = [p for p in harness.queue_files() if p.suffix == ".json"]
    record = json.loads(qfile.read_text())
    record["ts"] = 1700000000.0  # 2023-11-14T22:13:20Z
    qfile.write_text(json.dumps(record))

    harness.embedder.fail = False
    assert replay_queue(harness.memory, user_id=USER)["replayed"] == 1
    payload = harness.payload(queued["id"])
    assert parse_instant(payload["recorded_at"]) == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert payload["recorded_at_source"] == "queue"


def test_step09c_replay_without_queue_time_says_replay(harness):
    text = "Fidelis queue records written by older clients have no timestamp."
    harness.embedder.fail = True
    queued = safe_add(harness.memory, text, USER, kind="store")
    (qfile,) = [p for p in harness.queue_files() if p.suffix == ".json"]
    record = json.loads(qfile.read_text())
    record.pop("ts", None)
    record.pop("queued_at", None)
    qfile.write_text(json.dumps(record))

    harness.embedder.fail = False
    before = datetime.now(timezone.utc) - timedelta(seconds=2)
    assert replay_queue(harness.memory, user_id=USER)["replayed"] == 1
    payload = harness.payload(queued["id"])
    assert payload["recorded_at_source"] == "replay"
    assert parse_instant(payload["recorded_at"]) >= before


def test_step09d_queued_supersession_survives_replay(harness):
    a = harness.store_ok(TEXT_A)
    harness.embedder.fail = True
    status, queued = harness.store(TEXT_B, supersedes=[a["id"]])
    assert queued["status"] == "queued", (status, queued)
    harness.embedder.fail = False
    assert replay_queue(harness.memory, user_id=USER)["replayed"] == 1

    assert json.loads(harness.payload(queued["id"])["supersedes_json"]) == [a["id"]]
    hits = _by_id(harness.recall("/query", QUERY))
    assert hits[a["id"]]["temporal"]["status"] == "superseded"
    assert hits[a["id"]]["temporal"]["superseded_by"] == [queued["id"]]


@pytest.mark.parametrize("endpoint", RECALL_ENDPOINTS)
def test_step10_legacy_payload_is_current_with_unknown_recorded_at(harness, endpoint):
    legacy_text = "Legacy note: fidelis server port was chosen by the owner."
    harness.memory.vector_store.insert(
        vectors=[HashingEmbedder.vector(legacy_text)],
        payloads=[{"data": legacy_text, "user_id": USER}],
        ids=["legacy-row-0001"],
    )
    harness.store_ok(TEXT_A)

    memories = harness.recall(endpoint, "fidelis server port owner legacy note")

    legacy = [m for m in memories if m.get("text") == legacy_text]
    assert len(legacy) == 1, memories
    temporal = legacy[0]["temporal"]
    assert temporal["recorded_at_known"] is False
    assert temporal["recorded_at"] is None
    assert temporal["status"] == "current"
    # The legacy row itself was not touched by reading it.
    assert harness.payload("legacy-row-0001") == {"data": legacy_text, "user_id": USER}

    # Legacy rows stay visible under as_of (flagged, not guessed).
    past = harness.recall(endpoint, "fidelis server port owner legacy note",
                          as_of="2001-01-01T00:00:00Z")
    assert legacy_text in [m.get("text") for m in past]
    assert TEXT_A not in [m.get("text") for m in past]


# ── Invariants ──────────────────────────────────────────────────────────────


TRICKY_TEXT = (
    "  \tCafé naïve ☕ 日本語 \U0001f680 — on 2024-03-15 the  fidelis   server\n"
    "\tmoved to port 19420 at 2024-03-15T09:30:00Z;  trailing   spaces kept \n "
)


@pytest.mark.parametrize("endpoint", RECALL_ENDPOINTS)
def test_invariant_a_byte_identity_and_no_date_promotion(harness, endpoint):
    stored = harness.store_ok(TRICKY_TEXT)
    plain = harness.store_ok(TEXT_A)

    payload = harness.payload(stored["id"])
    assert payload["data"] == TRICKY_TEXT
    assert payload["data"].encode("utf-8") == TRICKY_TEXT.encode("utf-8")
    for key in ("event_at", "valid_from", "valid_to"):
        assert key not in payload, f"in-text date was promoted to {key}"
    assert payload["content_sha256"] == hashlib.sha256(
        TRICKY_TEXT.strip().encode("utf-8")
    ).hexdigest()

    memories = harness.recall(endpoint, "fidelis server moved to port 19420")
    hits = _by_id(memories)
    assert hits[stored["id"]]["text"] == TRICKY_TEXT
    assert hits[plain["id"]]["text"] == TEXT_A
    temporal = hits[stored["id"]]["temporal"]
    assert temporal["event_at"] is None
    assert temporal["valid_from"] is None
    assert temporal["valid_to"] is None
    assert temporal["status"] == "current"
    # Every stored text comes back exactly.
    stored_texts = {TRICKY_TEXT, TEXT_A}
    assert {m["text"] for m in memories} == stored_texts


def test_invariant_b_append_only_after_supersession(harness):
    a = harness.store_ok(TEXT_A)
    snapshot = harness.payload(a["id"])

    b = harness.store_ok(TEXT_B, supersedes=[a["id"]])
    for endpoint in RECALL_ENDPOINTS:
        harness.recall(endpoint, QUERY)
        harness.recall(endpoint, QUERY, as_of=b["recorded_at"])

    assert harness.count() == 2
    assert harness.payload(a["id"]) == snapshot
    assert harness.payload(a["id"])["data"] == TEXT_A
    ids = set(harness.memory.vector_store.collection.get()["ids"])
    assert ids == {a["id"], b["id"]}


@pytest.mark.parametrize("endpoint", RECALL_ENDPOINTS)
def test_invariant_c_superseded_never_silently_dropped(harness, endpoint):
    a, b = _store_a_then_b(harness)
    # A query that favours A on raw relevance: still present, still demoted.
    query = "fidelis server listens on port 19420"

    memories = harness.recall(endpoint, query)
    hits = _by_id(memories)
    assert a["id"] in hits and b["id"] in hits
    assert hits[a["id"]]["temporal"]["status"] == "superseded"
    assert _position(memories, b["id"]) < _position(memories, a["id"])

    # historical=True skips reordering but keeps the annotation.
    historical = harness.recall(endpoint, query, historical=True)
    assert _by_id(historical)[a["id"]]["temporal"]["status"] == "superseded"
    assert {a["id"], b["id"]} <= set(_by_id(historical))

    # The only exclusion is as_of, and it excludes B (recorded later), not A.
    as_of_hits = _by_id(harness.recall(endpoint, query, as_of=_midpoint(a, b)))
    assert a["id"] in as_of_hits and b["id"] not in as_of_hits

    # as_of at/after B: both visible again, A superseded again.
    later = _by_id(harness.recall(endpoint, query, as_of=b["recorded_at"]))
    assert later[a["id"]]["temporal"]["status"] == "superseded"
    assert b["id"] in later


def _assert_passthrough_unavailable(memories: list[dict], expected_texts: set[str]):
    assert {m["text"] for m in memories} >= expected_texts
    for m in memories:
        assert "temporal" in m, f"hit lost its temporal annotation: {m}"
        assert m["temporal"]["index"] == "unavailable"


def test_invariant_d_fail_open_corrupt_sidecar_at_startup(harness):
    a, b = _store_a_then_b(harness)
    assert harness.sidecar.exists(), "sidecar not at the contract path"

    configure_temporal(None)  # server stopped
    for extra in ("-wal", "-shm"):
        Path(str(harness.sidecar) + extra).unlink(missing_ok=True)
    harness.sidecar.write_bytes(b"this is not a sqlite database" * 64)
    configure_temporal(harness.store_path, harness.cfg)  # server restarted

    for endpoint in RECALL_ENDPOINTS:
        _assert_passthrough_unavailable(harness.recall(endpoint, QUERY), {TEXT_A, TEXT_B})
    # Fail-open does not extend to losing writes: a store still persists.
    count = harness.count()
    status, body = harness.store("Fidelis keeps writing when the sidecar is corrupt.")
    assert status == 200 and body["status"] == "stored", (status, body)
    assert harness.count() == count + 1


def test_invariant_d_fail_open_sidecar_corrupted_while_running(harness):
    _store_a_then_b(harness)
    for extra in ("-wal", "-shm"):
        Path(str(harness.sidecar) + extra).unlink(missing_ok=True)
    harness.sidecar.write_bytes(b"\x00garbage-not-sqlite\xff" * 512)

    for endpoint in RECALL_ENDPOINTS:
        status, response = harness.post(
            endpoint, {"text": QUERY, "limit": 10, "top_k": 10}
        )
        assert status == 200, (endpoint, status, response)
        memories = response["memories"]
        assert {m["text"] for m in memories} >= {TEXT_A, TEXT_B}
        for m in memories:
            assert "temporal" in m, f"{endpoint}: hit passed through without temporal: {m}"
            assert m["temporal"]["index"] in ("ok", "unavailable")


def test_invariant_d_fail_open_sidecar_with_damaged_pages(harness):
    """Header intact, table pages trashed: the sidecar opens, then reads fail."""
    _store_a_then_b(harness)
    configure_temporal(None)  # server stopped; closing checkpoints the WAL
    for extra in ("-wal", "-shm"):
        Path(str(harness.sidecar) + extra).unlink(missing_ok=True)
    raw = bytearray(harness.sidecar.read_bytes())
    page_size = int.from_bytes(raw[16:18], "big") or 65536
    assert len(raw) > page_size, "precondition: sidecar has table pages to damage"
    raw[page_size:] = b"\xde\xad\xbe\xef" * ((len(raw) - page_size) // 4)
    harness.sidecar.write_bytes(bytes(raw))
    configure_temporal(harness.store_path, harness.cfg)  # server restarted

    for endpoint in RECALL_ENDPOINTS:
        status, response = harness.post(
            endpoint, {"text": QUERY, "limit": 10, "top_k": 10}
        )
        assert status == 200, (endpoint, status, response)
        _assert_passthrough_unavailable(response["memories"], {TEXT_A, TEXT_B})


def test_invariant_d_fail_open_index_read_error(harness):
    """A sidecar whose reads raise: hits pass through, index == 'unavailable'."""

    class BrokenIndex:
        def find_by_sha(self, sha):
            raise sqlite3.DatabaseError("database disk image is malformed")

        def supersession_index(self):
            raise sqlite3.DatabaseError("database disk image is malformed")

    a, b = _store_a_then_b(harness)
    hits = [
        {"id": a["id"], "text": TEXT_A, "score": 0.9},
        {"id": b["id"], "text": TEXT_B, "score": 0.8},
    ]
    original = json.loads(json.dumps(hits))

    viewed = temporal_view(hits, memory=harness.memory, index=BrokenIndex())

    assert hits == original, "input hits were mutated"
    assert [m["text"] for m in viewed] == [TEXT_A, TEXT_B]
    _assert_passthrough_unavailable(viewed, {TEXT_A, TEXT_B})


def test_invariant_d_fail_open_deleted_sidecar(harness):
    a, b = _store_a_then_b(harness)

    configure_temporal(None)  # server stopped
    for extra in ("", "-wal", "-shm"):
        Path(str(harness.sidecar) + extra).unlink(missing_ok=True)
    assert not harness.sidecar.exists()

    # No sidecar bound: documented pass-through.
    memories = harness.recall("/query", QUERY)
    _assert_passthrough_unavailable(memories, {TEXT_A, TEXT_B})

    # Restart against the store whose sidecar went missing. The payload fields
    # are the source of truth; the sidecar is a rebuildable cache. Either the
    # supersession is known again (rebuilt) or the hit says the index is
    # unavailable. Reporting A as "current" with index "ok" is a guessed status.
    configure_temporal(harness.store_path, harness.cfg)
    hits = _by_id(harness.recall("/query", QUERY))
    temporal_a = hits[a["id"]]["temporal"]
    assert temporal_a["index"] == "unavailable" or (
        temporal_a["status"] == "superseded" and temporal_a["superseded_by"] == [b["id"]]
    ), f"missing sidecar silently reported superseded A as: {temporal_a}"


INVALID_DECLARATIONS = {
    "unparseable_event_at": {"event_at": "last week"},
    "unparseable_valid_from": {"valid_from": "not-a-date"},
    "unparseable_valid_to": {"valid_to": "2026-13-45"},
    "integer_instant": {"event_at": 1750000000},
    "valid_from_equals_valid_to": {"valid_from": "2026-01-01", "valid_to": "2026-01-01"},
    "valid_from_after_valid_to": {
        "valid_from": "2026-06-01T00:00:00Z", "valid_to": "2026-01-01T00:00:00Z",
    },
    "too_many_supersedes": {"supersedes": [f"record-{i:03d}" for i in range(65)]},
    "supersedes_empty_string_member": {"supersedes": ["record-001", ""]},
    "supersedes_not_a_list": {"supersedes": {"id": "record-001"}},
    "source_too_long": {"source": "s" * 513},
}


@pytest.mark.parametrize("name", sorted(INVALID_DECLARATIONS))
def test_invariant_e_invalid_declaration_is_400_and_writes_nothing(harness, name):
    harness.store_ok(TEXT_A)
    count = harness.count()

    status, body = harness.store(
        "Fidelis write that carries an invalid temporal declaration.",
        **INVALID_DECLARATIONS[name],
    )

    assert status == 400, (name, status, body)
    assert isinstance(body.get("error"), str) and body["error"]
    assert harness.count() == count
    assert harness.queue_files() == []


def test_invariant_e_self_supersession_is_400(harness, monkeypatch):
    # A caller can only name its own record id when it supplies a stable id,
    # which the server honours with relation envelopes on.
    monkeypatch.setenv(FEATURE_ENV, "1")
    harness.store_ok(TEXT_A)
    count = harness.count()

    status, body = harness.store(
        TEXT_B, id="port-fact-0002", supersedes=["port-fact-0002"]
    )

    assert status == 400, (status, body)
    assert isinstance(body.get("error"), str) and body["error"]
    assert harness.count() == count
    assert harness.queue_files() == []


def test_invariant_e_64_supersedes_ids_is_accepted(harness):
    # PACKET F1 (docs/TIME-AWARE-SPEC.md): supersedes ids are now validated to
    # exist for this user at write time, so the 64-id maximum is exercised
    # here against 64 REAL records rather than synthetic nonexistent ids —
    # the format/count limit is what this test pins, not the now-removed
    # permissive acceptance of unknown ids.
    targets = [harness.store_ok(f"record {i:03d} fact.")["id"] for i in range(64)]
    body = harness.store_ok(TEXT_B, supersedes=targets)
    assert len(json.loads(harness.payload(body["id"])["supersedes_json"])) == 64


def test_invariant_f_rejected_writes_are_never_queued(harness):
    tag = "task" + "-notification"
    rejected_texts = [
        "<" + tag + ">agent finished</" + tag + ">",
        "note canary-probe-1234 left behind by a health probe",
        "You are QA for a drafted fix. Answer ONLY: YES or NO",
    ]
    for text in rejected_texts:
        status, body = harness.store(text)
        assert status == 422, (text, status, body)
        assert body["status"] == "rejected"
        assert body.get("reason")

    # Even while the dependency is down, a rejection is not turned into a queue entry.
    harness.embedder.fail = True
    status, body = harness.store(rejected_texts[0])
    assert status == 422 and body["status"] == "rejected", (status, body)
    direct = safe_add(harness.memory, rejected_texts[1], USER, kind="store")
    assert direct["status"] == "rejected"
    harness.embedder.fail = False

    assert harness.count() == 0
    assert harness.queue_files() == []
    assert not harness.queue_dir.exists() or list(harness.queue_dir.iterdir()) == []


def test_invariant_f_secret_is_rejected_without_echo(harness):
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    status, body = harness.store("deploy token for the fidelis mirror: " + secret)
    assert status == 422, (status, body)
    assert body["reason"] == "secret_like"
    assert secret not in json.dumps(body)
    assert harness.count() == 0
    assert harness.queue_files() == []


def test_invariant_f_prose_about_markup_is_accepted(harness):
    body = harness.store_ok("The store contained task-notification tags before the gate.")
    assert harness.payload(body["id"])["data"].startswith("The store contained")


def test_invariant_g_dedup_skipped_when_validity_or_supersedes_declared(harness):
    first = harness.store_ok(TEXT_A)

    second = harness.store_ok(TEXT_A, valid_from="2026-01-01T00:00:00Z")
    assert second["id"] != first["id"]
    assert harness.count() == 2

    third = harness.store_ok(TEXT_A, valid_to="2031-01-01")
    fourth = harness.store_ok(TEXT_A, supersedes=[first["id"]])
    assert len({first["id"], second["id"], third["id"], fourth["id"]}) == 4
    assert harness.count() == 4
    for record_id in (first["id"], second["id"], third["id"], fourth["id"]):
        assert harness.payload(record_id)["data"] == TEXT_A
    assert parse_instant(harness.payload(second["id"])["valid_from"]) == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )

    # A plain re-store with no declaration is still a duplicate — of the oldest
    # copy that is still CURRENT. `first` was superseded by `fourth` above, so
    # it no longer makes a new write redundant (maintainer review 2026-09-20:
    # a once-superseded fact must stay re-assertable).
    status, again = harness.store(TEXT_A)
    assert again["status"] == "duplicate", (status, again)
    assert again["id"] == second["id"]
    assert harness.count() == 4


def test_invariant_h_kill_switch_writes_no_temporal_fields(harness, monkeypatch):
    monkeypatch.setenv("COGITO_TEMPORAL_V1", "0")
    configure_temporal(harness.store_path, harness.cfg)

    status, body = harness.store(TEXT_A)

    assert status == 200 and body["status"] == "stored", (status, body)
    payload = harness.payload(body["id"])
    assert payload == {"data": TEXT_A, "user_id": USER}
    direct = safe_add(harness.memory, TEXT_B, USER, kind="store")
    assert direct["status"] == "stored"
    assert harness.payload(direct["id"]) == {"data": TEXT_B, "user_id": USER}
    # Recall keeps working over such rows.
    assert TEXT_A in [m["text"] for m in harness.recall("/query", QUERY)]


def test_invariant_temporal_fields_written_with_relation_envelopes_on(harness, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV, "1")
    body = harness.store_ok(TEXT_A, id="port-fact-0001", event_at="2026-05-01")
    payload = harness.payload(body["id"])
    assert payload["data"] == TEXT_A
    assert payload["temporal_schema"] == "fidelis.temporal/v1"
    assert payload["recorded_at_source"] == "write"
    assert parse_instant(payload["event_at"]) == datetime(2026, 5, 1, tzinfo=timezone.utc)
    for value in payload.values():
        assert isinstance(value, (str, int, float, bool)), "payload must stay Chroma-safe scalars"


# ── (i) MCP honesty ─────────────────────────────────────────────────────────


def _claims_stored(rendered) -> bool:
    text = rendered if isinstance(rendered, str) else json.dumps(rendered)
    lowered = text.lower()
    if isinstance(rendered, dict) and rendered.get("status") == "stored":
        return True
    return lowered.startswith("stored") or "stored memory" in lowered


def test_invariant_i_mcp_store_never_claims_stored_when_not_persisted(monkeypatch):
    responses = [
        {"status": "rejected", "reason": "harness_envelope", "detail": None,
         "queued_total": 0, "http_status": 422},
        {"status": "duplicate", "id": "existing-record-42", "extracted": [], "queued_total": 0},
        {"status": "queued", "id": "queued-record-77",
         "reason": "ConnectionError: embedder unreachable", "queued_total": 1},
        {"status": "stored", "id": "fresh-record-99", "extracted": ["x"],
         "recorded_at": "2026-09-20T10:00:00Z", "queued_total": 0},
    ]
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append((path, payload))
        return responses[len(calls) - 1]

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    monkeypatch.delenv("COGITO_HERMENEUTICS_EVIDENCE_STATUS", raising=False)

    rejected = mcp_server._tool_store({"text": "some harness text"})
    duplicate = mcp_server._tool_store({"text": TEXT_A})
    queued = mcp_server._tool_store({"text": TEXT_A})
    stored = mcp_server._tool_store({"text": TEXT_B, "supersedes": ["existing-record-42"]})

    assert [path for path, _ in calls] == ["/store"] * 4
    assert calls[3][1]["supersedes"] == ["existing-record-42"]

    assert not _claims_stored(rejected), rejected
    assert "reject" in str(rejected).lower()
    assert "harness_envelope" in str(rejected)

    assert not _claims_stored(duplicate), duplicate
    assert "existing-record-42" in str(duplicate)
    assert "fresh" not in str(duplicate)

    assert not _claims_stored(queued), queued
    assert "queued" in str(queued).lower()
    assert "queued-record-77" in str(queued)

    assert _claims_stored(stored), stored
    assert "fresh-record-99" in str(stored)


def test_invariant_i_mcp_forwards_temporal_declarations(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout=30.0):
        seen.update(payload)
        return {"status": "stored", "id": "r1", "recorded_at": "2026-09-20T10:00:00Z"}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    mcp_server._tool_store({
        "text": TEXT_B, "event_at": "2026-09-01", "valid_from": "2026-09-01",
        "valid_to": "2027-01-01", "supersedes": ["a1"], "source": "ops log",
    })
    assert seen["event_at"] == "2026-09-01"
    assert seen["valid_from"] == "2026-09-01"
    assert seen["valid_to"] == "2027-01-01"
    assert seen["supersedes"] == ["a1"]
    assert seen["source"] == "ops log"


def test_invariant_i_mcp_recall_text_marks_superseded_and_expired(monkeypatch):
    memories = [
        {"id": "rec-B", "text": TEXT_B, "score": 0.91,
         "temporal": {"status": "current", "recorded_at": "2026-09-20T10:00:05Z",
                      "recorded_at_known": True, "superseded_by": [], "valid_from": None,
                      "valid_to": None, "event_at": None,
                      "evaluated_at": "2026-09-20T11:00:00Z", "index": "ok"}},
        {"id": "rec-A", "text": TEXT_A, "score": 0.92,
         "temporal": {"status": "superseded", "recorded_at": "2026-09-20T10:00:00Z",
                      "recorded_at_known": True, "superseded_by": ["rec-B"],
                      "valid_from": None, "valid_to": None, "event_at": None,
                      "evaluated_at": "2026-09-20T11:00:00Z", "index": "ok"}},
        {"id": "rec-C", "text": "Temporary override on port 18000.", "score": 0.5,
         "temporal": {"status": "expired", "recorded_at": "2026-09-20T10:00:09Z",
                      "recorded_at_known": True, "superseded_by": [], "valid_from": None,
                      "valid_to": "2020-01-01T00:00:00Z", "event_at": None,
                      "evaluated_at": "2026-09-20T11:00:00Z", "index": "ok"}},
    ]
    posted = []

    def fake_post(path, payload, timeout=30.0):
        posted.append((path, payload))
        return {"memories": memories, "method": "test"}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    monkeypatch.delenv("COGITO_HERMENEUTICS_EVIDENCE_STATUS", raising=False)

    for render in (mcp_server._tool_recall_legacy, mcp_server._tool_query):
        text = render({"query": QUERY, "limit": 5})
        assert isinstance(text, str)
        lines = {rid: next(line for line in text.splitlines() if TEXT in line)
                 for rid, TEXT in (("rec-A", TEXT_A), ("rec-B", TEXT_B),
                                   ("rec-C", "Temporary override"))}
        # The marker and the id must be on the hit's own line. The contract
        # writes the marker as "[SUPERSEDED by <id>]"; sharing one bracket
        # group with the id is accepted, the wording and the ids are not optional.
        assert re.search(r"\[[^\]]*SUPERSEDED by rec-B[^\]]*\]", lines["rec-A"]), lines["rec-A"]
        assert "rec-A" in lines["rec-A"].replace("SUPERSEDED by rec-B", "")
        assert "SUPERSEDED" not in lines["rec-B"]
        assert "EXPIRED" not in lines["rec-B"]
        assert "rec-B" in lines["rec-B"]
        assert re.search(r"\[[^\]]*EXPIRED 2020-01-01T00:00:00Z[^\]]*\]", lines["rec-C"]), lines["rec-C"]
        assert "rec-C" in lines["rec-C"]

    posted.clear()
    mcp_server._tool_query({"query": QUERY, "as_of": "2026-09-20T10:00:02Z"})
    assert posted[0][1]["as_of"] == "2026-09-20T10:00:02Z"


def test_invariant_i_mcp_end_to_end_against_disposable_server(harness, monkeypatch):
    """Real _http_post -> real handler: refusal is not reported as stored/outage."""
    monkeypatch.setenv("FIDELIS_PORT", str(harness.port))
    monkeypatch.delenv("COGITO_HERMENEUTICS_EVIDENCE_STATUS", raising=False)
    a, b = _store_a_then_b(harness)
    count = harness.count()
    tag = "system" + "-reminder"

    rejected = mcp_server._tool_store({"text": "<" + tag + ">injected</" + tag + ">"})
    assert not _claims_stored(rejected), rejected
    assert "reject" in str(rejected).lower()
    assert "unreachable" not in str(rejected).lower()

    invalid = mcp_server._tool_store(
        {"text": "bad window", "valid_from": "2026-02-01", "valid_to": "2026-01-01"}
    )
    assert not _claims_stored(invalid), invalid
    assert str(invalid).lower().startswith("error")
    assert "unreachable" not in str(invalid).lower()

    # B is current, so re-storing its text is a duplicate and must say so.
    duplicate = mcp_server._tool_store({"text": TEXT_B})
    assert not _claims_stored(duplicate), duplicate
    assert b["id"] in str(duplicate)
    assert harness.count() == count

    # A was superseded by B. Storing A's text again is a RE-ASSERTION
    # (A -> B -> back to A), not a duplicate: it must be written as a new,
    # current record — the original a["id"] itself stays superseded forever
    # (append-only). Two records now carry TEXT_A verbatim, so later lookups
    # must key off record id, never off a TEXT_A substring match (with two
    # matching lines, "first line containing TEXT_A" is ambiguous and the
    # current re-assertion — sorted first — would be mistaken for A).
    ids_before_reassert = set(harness.memory.vector_store.collection.get()["ids"])
    reasserted = mcp_server._tool_store({"text": TEXT_A})
    assert _claims_stored(reasserted), reasserted
    assert a["id"] not in str(reasserted)
    assert harness.count() == count + 1
    ids_after_reassert = set(harness.memory.vector_store.collection.get()["ids"])
    new_ids = ids_after_reassert - ids_before_reassert
    assert len(new_ids) == 1, new_ids
    (reasserted_id,) = new_ids
    assert reasserted_id != a["id"]
    assert reasserted_id in str(reasserted)

    rendered = mcp_server._tool_query({"query": QUERY, "limit": 5})
    lines = rendered.splitlines()

    line_a = next(line for line in lines if f"id={a['id']}" in line)
    assert "SUPERSEDED by " + b["id"] in line_a
    assert a["id"] in line_a

    # The re-asserted copy is a DIFFERENT, current record: no SUPERSEDED
    # marker, even though it carries the same verbatim text as A.
    line_reasserted = next(line for line in lines if f"id={reasserted_id}" in line)
    assert "SUPERSEDED" not in line_reasserted
    assert TEXT_A in line_reasserted
