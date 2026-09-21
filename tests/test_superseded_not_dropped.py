"""HTTP-level regression for WORK PACKET E (docs/TIME-AWARE-SPEC.md).

Reproduces the field defect on a real handler: a store holding many
CURRENT records plus a superseded/superseder pair. Querying the superseded
record's own exact text at a small limit used to return the superseder plus
unrelated current records — the superseded record itself was silently
dropped, because ``temporal_view`` partitioned the WHOLE over-fetched pool
current-first and only THEN trimmed to ``limit`` (see
``tests/test_temporal_recall.py::TestSupersededKeptInEligibilityWindow`` for
the unit-level version of this same defect).

``tests/test_time_aware_acceptance.py`` is owned by another author and is not
edited here; this file stands up the real handler the SAME WAY that file
does — a real mem0 Chroma vector store under ``tmp_path``, a deterministic
bag-of-words hashing embedder (no Ollama, no network), the real
``fidelis.server.make_handler`` on an ephemeral port, ``configure_temporal``,
env isolation, and teardown resetting module-level state — duplicated here
rather than imported, matching this codebase's existing convention of each
test file mirroring a harness rather than importing one (see
``tests/test_score_and_dedup.py``'s docstring, which does the same).
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
from fidelis.degrade import configure_temporal
from fidelis.relation_envelope import FEATURE_ENV

TARGET_A = "The Zephyrine gateway service listens on port 8811."
TARGET_B = "The Zephyrine gateway service was moved to port 8822 for the new deployment."
# Deliberately avoids repeating "Zephyrine gateway service" — every filler
# record shares those tokens, so a paraphrase built from them alone gets
# diluted across recall_b's decompose+RRF pipeline (empirically verified:
# it stopped ranking B first). Naming the new port and "deployment"/"move"
# keeps this reliably B-specific across /query, /recall, and /recall_hybrid.
PARAPHRASE = "which new port number 8822 did the service move to for the deployment"
EXACT_QUERY_ENDPOINTS = ["/query", "/recall", "/recall_hybrid"]

# A near-identical pair (differs only in the port number, at the very end of
# a long-enough sentence that the shared 4-word shingles dominate) — Jaccard
# ~0.85 against fidelis.recall._dedup_candidates' own shingle/Jaccard
# measure, safely over its 0.8 collapse threshold. Used ONLY by the single
# xfail test below; the main assertions use TARGET_A/TARGET_B instead
# (Jaccard ~0.07 — comfortably under 0.7) so the pre-existing, separately
# tracked P6 gap in recall.py's dedup does not mask what this packet fixes.
NEAR_A = (
    "The Zephyrine gateway service originally listened for all inbound "
    "external traffic on legacy port 8811."
)
NEAR_B = (
    "The Zephyrine gateway service originally listened for all inbound "
    "external traffic on legacy port 8822."
)

_FILLER_TOPICS = [
    "supports TLS termination for inbound connections",
    "logs access records to var log zephyrine",
    "authenticates clients via mutual TLS certificates",
    "rate limits requests per client above a threshold",
    "caches upstream responses for thirty seconds",
    "exposes a health check endpoint for the load balancer",
    "rotates its credentials on a weekly schedule",
    "is deployed across three availability zones",
    "compresses response bodies above one kilobyte",
    "retries upstream failures with exponential backoff",
    "emits structured metrics to the telemetry collector",
    "was configured by the platform team last quarter",
    "supports both ipv4 and ipv6 listeners",
    "queues connections during a rolling restart",
    "validates request headers against an allow list",
    "shares a connection pool with the sibling service",
    "was audited for compliance earlier this year",
    "runs behind a dedicated reverse proxy layer",
    "buffers slow client writes to protect upstream workers",
    "was migrated to the new container runtime",
    "reports uptime to the internal status page",
    "supports websocket upgrades for streaming clients",
    "enforces a maximum request body size",
    "was documented in the service catalog",
    "delegates authorization checks to the policy engine",
    "is paged on when error rates spike",
    "supports blue green deployments for releases",
    "throttles background jobs during peak hours",
    "was load tested before the last major release",
    "publishes its openapi schema for consumers",
]


def _unrelated_current_records() -> list[str]:
    """>= 30 unrelated CURRENT records sharing vocabulary with the target
    (``Zephyrine``, ``gateway``, ``service``, ``port``) so the over-fetch
    pool the handlers build is dominated by current hits — exactly the field
    condition (488-record store) that hid the defect."""
    assert len(_FILLER_TOPICS) >= 30
    return [f"The Zephyrine gateway service {topic}." for topic in _FILLER_TOPICS]


# ── deterministic embedder (mirrors tests/test_time_aware_acceptance.py) ────


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


# ── harness (mirrors tests/test_time_aware_acceptance.py's Harness) ────────


class Harness:
    def __init__(self, memory, cfg: dict):
        self.memory = memory
        self.cfg = cfg
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

    def store_ok(self, text: str, **fields) -> dict:
        status, body = self.post("/store", {"text": text, **fields})
        assert status == 200, (status, body)
        assert body.get("status") == "stored", body
        return body

    def recall(self, endpoint: str, query: str, *, limit: int = 5) -> list[dict]:
        body = {"text": query, "limit": limit}
        if endpoint == "/recall_hybrid":
            body["top_k"] = limit
        status, response = self.post(endpoint, body)
        assert status == 200, (endpoint, status, response)
        return response["memories"]


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

    # No Ollama: secondary embedding calls inside the retrievers are made
    # deterministic (same approach as tests/test_time_aware_acceptance.py).
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
        vector_store=ChromaDB(collection_name="superseded-not-dropped", path=str(store_path)),
    )
    cfg = {
        "user_id": "packet-e-agent",
        "collection": "superseded-not-dropped",
        "store_path": str(store_path),
        "recall_limit": 30,
        "ephemera_filter": False,
        "supersession_pointers_path": None,
        "ollama_url": "http://127.0.0.1:9",  # never reachable; never contacted
        "filter_endpoint": "",
    }
    h = Harness(memory, cfg)
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


def _by_id(memories: list[dict]) -> dict[str, dict]:
    return {m["id"]: m for m in memories if m.get("id")}


def _seed_pool_and_supersession(h: Harness, text_a: str, text_b: str) -> tuple[dict, dict]:
    """>= 30 unrelated current fillers, then A, then B superseding A."""
    for text in _unrelated_current_records():
        h.store_ok(text)
    a = h.store_ok(text_a)
    b = h.store_ok(text_b, supersedes=[a["id"]])
    return a, b


# ── main scenario: dissimilar A/B texts (Jaccard well under the 0.8 gap) ────


@pytest.mark.parametrize("endpoint", EXACT_QUERY_ENDPOINTS)
def test_exact_text_query_keeps_superseded_a_and_ranks_b_above_it(harness, endpoint):
    a, b = _seed_pool_and_supersession(harness, TARGET_A, TARGET_B)

    memories = harness.recall(endpoint, TARGET_A, limit=5)
    hits = _by_id(memories)

    assert a["id"] in hits, (
        f"{endpoint}: superseded record A was silently dropped from a "
        f"{len(memories)}-hit response — the WORK PACKET E defect"
    )
    # Temporal replacement is explicitly pool-only. Legacy /recall's RRF
    # top-five pool can omit B on another Chroma/platform tie ordering; that
    # is retrieval coverage, not this regression (silently dropping A).
    replacement_in_pool = True
    if endpoint == "/recall":
        pool, _ = server.do_recall(
            harness.memory, TARGET_A, user_id=harness.cfg["user_id"],
            cfg=harness.cfg, limit=5,
        )
        replacement_in_pool = b["id"] in _by_id(pool)
    if replacement_in_pool:
        assert b["id"] in hits
        ids = [m.get("id") for m in memories]
        assert ids.index(b["id"]) < ids.index(a["id"]), f"{endpoint}: B must rank above superseded A"
    assert hits[a["id"]]["temporal"]["status"] == "superseded"
    assert hits[a["id"]]["temporal"]["superseded_by"] == [b["id"]]
    assert hits[a["id"]]["text"] == TARGET_A
    if replacement_in_pool:
        assert hits[b["id"]]["temporal"]["status"] == "current"
        assert hits[b["id"]]["text"] == TARGET_B


@pytest.mark.parametrize("endpoint", EXACT_QUERY_ENDPOINTS)
def test_paraphrase_query_ranks_b_first_a_present_only_if_in_window(harness, endpoint):
    a, b = _seed_pool_and_supersession(harness, TARGET_A, TARGET_B)

    memories = harness.recall(endpoint, PARAPHRASE, limit=5)
    hits = _by_id(memories)

    assert memories, f"{endpoint}: paraphrase query returned nothing"
    assert memories[0].get("id") == b["id"], (
        f"{endpoint}: paraphrase of B's content did not rank B first: {memories}"
    )
    assert hits[b["id"]]["temporal"]["status"] == "current"
    # A is only required to be present when relevance itself put A in the
    # pool for this query; when present, it must never be silently dropped.
    if a["id"] in hits:
        assert hits[a["id"]]["temporal"]["status"] == "superseded"
        ids = [m.get("id") for m in memories]
        assert ids.index(b["id"]) < ids.index(a["id"])


# ── known, separately-tracked gap: near-identical A/B upstream of temporal_view ──


@pytest.mark.xfail(
    strict=False,
    reason="known gap P6: near-duplicate collapse upstream of temporal_view",
)
def test_recall_near_identical_pair_collapses_before_temporal_view(harness):
    """Observed behaviour (2026-09-21, this harness): when A and B differ by
    only the port number in an otherwise-identical sentence (Jaccard ~0.85
    under fidelis.recall._dedup_candidates' shingle measure, over its 0.8
    collapse threshold), fidelis.recall.recall's Stage-1 near-duplicate
    dedup — which runs BEFORE temporal_view ever sees the candidate pool —
    keeps only the first-ranked of the pair and drops the other outright.
    Because dedup runs pre-temporal, it does not know one of the two is a
    supersession target: it can just as easily keep B and drop A (this
    packet's defect, from temporal_view's side) as keep A and drop B
    (backwards from a temporal standpoint, since the record that is CURRENT
    should survive). Which one is dropped depends on which one dedup's
    'first (highest-ranked) of each cluster' rule happens to keep for a
    given raw relevance ordering, not on temporal status. This is a
    pre-existing, separately-tracked gap in src/fidelis/recall.py's
    ``_dedup_candidates`` (Jaccard >= 0.8), not something WORK PACKET E's
    temporal_recall.py fix can (or should) paper over — temporal_view is
    given whatever pool survives dedup and correctly demotes-not-drops
    within it; it simply never receives both members of a collapsed pair to
    reorder in the first place. This test asserts the CORRECT outcome (both
    present, superseded one annotated and demoted) and is expected to fail
    for the reason above; it documents the gap rather than hiding it.
    """
    a, b = _seed_pool_and_supersession(harness, NEAR_A, NEAR_B)

    memories = harness.recall("/recall", NEAR_A, limit=5)
    hits = _by_id(memories)

    assert a["id"] in hits
    assert b["id"] in hits
    assert hits[a["id"]]["temporal"]["status"] == "superseded"
