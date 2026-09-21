"""Focused contracts for additive comparative relation envelopes."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from fidelis.cogito_hermeneutics import (
    apply_hermeneutics,
    plan_retrieval,
)
from fidelis.degrade import safe_add
from fidelis.relation_envelope import (
    DECLARATIONS_FIELD,
    EXTRACTOR_VERSION,
    FEATURE_ENV,
    PAYLOAD_FIELD,
    SCHEMA_VERSION,
    build_relation_envelope,
    canonical_envelope_json,
    envelope_from_metadata,
    expose_linked_raw_evidence,
    relation_envelopes_enabled,
    visible_payload_metadata,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "relation_envelopes_v1.json"
)


@pytest.fixture
def records() -> dict[str, dict]:
    payload = json.loads(FIXTURE.read_text())
    return {record["id"]: record for record in payload["records"]}


class _Embedder:
    def embed(self, text, **kwargs):
        return [float(len(text)), 1.0]


class _VectorStore:
    def __init__(self):
        self.inserts = []
        self.search_ids = None
        self.search_queries = []
        self.get_calls = []

    def insert(self, **kwargs):
        self.inserts.append(kwargs)

    def get(self, vector_id):
        self.get_calls.append(vector_id)
        for inserted in self.inserts:
            if inserted["ids"][0] == vector_id:
                return type(
                    "Stored",
                    (),
                    {
                        "id": vector_id,
                        "payload": inserted["payloads"][0],
                        "score": 0.0,
                    },
                )()
        raise KeyError(vector_id)

    def search(self, query, vectors, top_k, filters):
        self.search_queries.append(query)
        selected = (
            self.search_ids
            if self.search_ids is not None
            else [inserted["ids"][0] for inserted in self.inserts]
        )
        hits = []
        for rank, vector_id in enumerate(selected[:top_k]):
            for inserted in self.inserts:
                if inserted["ids"][0] == vector_id:
                    hits.append(
                        type(
                            "SearchHit",
                            (),
                            {
                                "id": vector_id,
                                "payload": inserted["payloads"][0],
                                "score": float(rank) / 10.0,
                            },
                        )()
                    )
                    break
        return hits


class _Memory:
    def __init__(self):
        self.embedding_model = _Embedder()
        self.vector_store = _VectorStore()


class _BrokenEmbedder:
    def embed(self, text, **kwargs):
        raise ConnectionError("embedding unavailable")


def _ingested_candidate(memory: _Memory, index: int) -> dict:
    inserted = memory.vector_store.inserts[index]
    payload = inserted["payloads"][0]
    return {
        "id": inserted["ids"][0],
        "text": payload["data"],
        "score": 1.0 - (index * 0.1),
        "metadata": {
            key: value for key, value in payload.items() if key != "data"
        },
    }


def test_feature_is_default_off_and_validates_values():
    assert relation_envelopes_enabled({}) is False
    assert relation_envelopes_enabled({FEATURE_ENV: "true"}) is True
    with pytest.raises(ValueError):
        relation_envelopes_enabled({FEATURE_ENV: "sometimes"})


def test_envelope_is_deterministic_versioned_and_raw_faithful(records):
    record = records["meridian-v2"]
    first = build_relation_envelope(
        record_id=record["id"],
        text=record["text"],
        metadata=record["metadata"],
    )
    second = build_relation_envelope(
        record_id=record["id"],
        text=record["text"],
        metadata=record["metadata"],
    )
    assert canonical_envelope_json(first) == canonical_envelope_json(second)
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["extractor"]["version"] == EXTRACTOR_VERSION
    assert first["extractor"]["runtime_llm_used"] is False
    assert first["record"]["text_sha256"] == hashlib.sha256(
        record["text"].encode()
    ).hexdigest()
    assert first["record"]["raw_text_preserved"] is True
    assert first["provenance"]["collection"] == "fixture-relations"
    assert first["provenance"]["source_pointer"] == (
        "fixture://meridian/current"
    )


def test_literal_entity_time_role_and_raw_declared_link(records):
    record = records["meridian-v2"]
    envelope = build_relation_envelope(
        record_id=record["id"],
        text=record["text"],
        metadata=record["metadata"],
    )
    entity = envelope["entity_spans"][0]
    assert record["text"][entity["start"]:entity["end"]] == (
        "Project Meridian"
    )
    assert envelope["event_time"]["expressions"][0]["text"] == "2026-07-26"
    assert envelope["candidate_roles"] == [
        "may_support_current",
        "may_support_change",
    ]
    assert envelope["linked_raw_record_ids"] == ["meridian-v1"]
    assert envelope["relations"][0]["status"] == (
        "declared_in_raw_text_not_truth_verified"
    )


def test_ambiguous_and_unrelated_records_remain_unresolved(records):
    ambiguous = records["meridian-ambiguous"]
    unresolved = build_relation_envelope(
        record_id=ambiguous["id"],
        text=ambiguous["text"],
        metadata=ambiguous["metadata"],
    )
    assert unresolved["ambiguity_status"] == "unresolved"
    assert unresolved["linked_raw_record_ids"] == []
    assert unresolved["relations"][0]["status"] == "unresolved_ambiguous"

    unrelated = records["unrelated"]
    envelope = build_relation_envelope(
        record_id=unrelated["id"],
        text=unrelated["text"],
        metadata=unrelated["metadata"],
    )
    assert envelope["candidate_roles"] == ["unknown"]
    assert envelope["relations"] == []
    assert "typed_relation" in envelope["unresolved_fields"]


def test_source_metadata_declaration_requires_literal_raw_support():
    text = (
        "On 2026-07-26, Fidelis Alpha2 is active locally from the repaired "
        "lineage."
    )
    metadata = {
        "source": "frozen-record",
        DECLARATIONS_FIELD: [
            {
                "type": "changes_from",
                "target_record_id": "prior-record",
                "support_text": "active locally from the repaired lineage",
            },
            {
                "type": "supersedes",
                "target_record_id": "fabricated",
                "support_text": "words not present in raw text",
            },
        ],
    }
    envelope = build_relation_envelope(
        record_id="current-record",
        text=text,
        metadata=metadata,
    )
    assert envelope["linked_raw_record_ids"] == ["prior-record"]
    assert len(envelope["relations"]) == 1
    assert envelope["relations"][0]["status"] == (
        "declared_in_source_metadata_with_literal_raw_span_"
        "not_truth_verified"
    )
    span = envelope["relations"][0]["source_span"]
    assert text[span["start"]:span["end"]] == span["text"]


def test_feature_off_preserves_exact_legacy_vector_payload(
    monkeypatch, records
):
    # Both features off: the payload is byte-identical to the legacy shape.
    # This is the rollback guarantee for COGITO_TEMPORAL_V1=0.
    monkeypatch.delenv(FEATURE_ENV, raising=False)
    monkeypatch.setenv("COGITO_TEMPORAL_V1", "0")
    memory = _Memory()
    record = records["meridian-v2"]
    result = safe_add(
        memory,
        record["text"],
        "agent",
        kind="store",
        record_id=record["id"],
        metadata=record["metadata"],
    )
    inserted = memory.vector_store.inserts[0]
    assert inserted["payloads"] == [
        {"data": record["text"], "user_id": "agent"}
    ]
    assert inserted["ids"][0] != record["id"]
    assert "relation_envelope" not in result


def test_relation_off_temporal_default_adds_only_temporal_fields(
    monkeypatch, records
):
    # Relation envelopes off, temporal stamping at its default (on): the raw
    # text is untouched, no relation field leaks, and the write is time-stamped.
    monkeypatch.delenv(FEATURE_ENV, raising=False)
    monkeypatch.delenv("COGITO_TEMPORAL_V1", raising=False)
    memory = _Memory()
    record = records["meridian-v2"]
    result = safe_add(
        memory,
        record["text"],
        "agent",
        kind="store",
        record_id=record["id"],
        metadata=record["metadata"],
    )
    payload = memory.vector_store.inserts[0]["payloads"][0]
    assert payload["data"] == record["text"]
    assert payload["user_id"] == "agent"
    assert not [key for key in payload if key.startswith("relation_")]
    assert payload["temporal_schema"] == "fidelis.temporal/v1"
    assert payload["recorded_at"].endswith("Z")
    assert payload["recorded_at_source"] == "write"
    assert result["recorded_at"] == payload["recorded_at"]
    assert "relation_envelope" not in result


def test_feature_on_stores_scalar_transport_and_preserves_raw(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    memory = _Memory()
    record = records["meridian-v2"]
    result = safe_add(
        memory,
        record["text"],
        "agent",
        kind="store",
        record_id=record["id"],
        metadata=record["metadata"],
    )
    payload = memory.vector_store.inserts[0]["payloads"][0]
    assert memory.vector_store.inserts[0]["ids"] == [record["id"]]
    assert payload["data"] == record["text"]
    assert isinstance(payload[PAYLOAD_FIELD], str)
    decoded = envelope_from_metadata(payload)
    assert decoded == result["relation_envelope"]
    assert decoded["record"]["id"] == record["id"]
    assert payload["relation_envelope_schema"] == SCHEMA_VERSION


def test_canonical_http_store_transports_relation_envelope(
    monkeypatch, tmp_path, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(tmp_path / "queue"))
    from fidelis.server import make_handler

    memory = _Memory()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            memory,
            {"user_id": "agent", "collection": "fixture-relations"},
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    record = records["meridian-v2"]
    body = json.dumps(
        {
            "id": record["id"],
            "text": record["text"],
            "metadata": record["metadata"],
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/store",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result["status"] == "stored"
    assert result["id"] == record["id"]
    assert result["relation_envelope"]["schema_version"] == SCHEMA_VERSION
    inserted = memory.vector_store.inserts[0]
    assert inserted["payloads"][0]["data"] == record["text"]
    assert envelope_from_metadata(inserted["payloads"][0]) == (
        result["relation_envelope"]
    )


def test_planner_contract_is_additive_only_when_flag_on(monkeypatch):
    query = (
        "What changed in Project Meridian, and what did it use before?"
    )
    monkeypatch.delenv(FEATURE_ENV, raising=False)
    legacy = plan_retrieval(query).to_dict()
    assert "evidence_needs" not in legacy
    assert "relation_envelope_schema" not in legacy

    monkeypatch.setenv(FEATURE_ENV, "1")
    plan = plan_retrieval(query)
    assert plan.mode == "mixed"
    assert plan.evidence_needs == ("prior", "current")
    assert plan.to_dict()["evidence_needs"] == ["prior", "current"]


def test_ambiguous_comparative_subject_does_not_request_links(
    monkeypatch,
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    plan = plan_retrieval(
        "What changed previously in the Hermeneutics integration?"
    )
    assert plan.evidence_needs == ()
    assert "comparative_ambiguous" in plan.cues


def test_feature_off_hides_persisted_relation_transport(records):
    record = records["meridian-v2"]
    from fidelis.relation_envelope import ingestion_payload

    payload, _ = ingestion_payload(
        text=record["text"],
        user_id="agent",
        record_id=record["id"],
        metadata=record["metadata"],
    )
    assert visible_payload_metadata(
        payload,
        enabled=False,
    ) == {"user_id": "agent"}
    enabled = visible_payload_metadata(payload, enabled=True)
    assert enabled["record_id"] == record["id"]
    assert enabled[PAYLOAD_FIELD]


def test_retrieval_facing_packet_exposes_only_declared_admitted_link(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    memory = _Memory()
    for record_id in ("meridian-v1", "meridian-v2", "unrelated"):
        record = records[record_id]
        safe_add(
            memory,
            record["text"],
            "agent",
            kind="store",
            record_id=record["id"],
            metadata=record["metadata"],
        )
    candidates = [
        _ingested_candidate(memory, index) for index in range(3)
    ]
    plan = plan_retrieval(
        "What changed in Project Meridian, and what did it use before?"
    )
    result = apply_hermeneutics(
        query=(
            "What changed in Project Meridian, and what did it use before?"
        ),
        candidates=candidates,
        plan=plan,
        limit=3,
        retrieval_method="fixture",
    )
    packet = result["declared_relation_evidence"]
    assert packet["status"] == "declared_linked_evidence"
    assert [record["id"] for record in packet["records"]] == [
        "meridian-v1",
        "meridian-v2",
    ]
    assert [record["text"] for record in packet["records"]] == [
        records["meridian-v1"]["text"],
        records["meridian-v2"]["text"],
    ]
    assert all(
        record["id"] != "unrelated" for record in packet["records"]
    )


def test_canonical_hybrid_admits_declared_link_inside_equal_budget(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    from fidelis import recall_hybrid as hybrid

    memory = _Memory()
    for record_id in ("meridian-v1", "meridian-v2", "unrelated"):
        record = records[record_id]
        safe_add(
            memory,
            record["text"],
            "agent",
            kind="store",
            record_id=record["id"],
            metadata=record["metadata"],
        )
    memory.vector_store.search_ids = ["meridian-v2", "unrelated"]
    monkeypatch.setattr(
        hybrid,
        "_embed_docs",
        lambda texts, cfg: [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        hybrid,
        "_embed_queries",
        lambda texts, cfg: [[1.0, 1.0] for _ in texts],
    )
    query = (
        "What changed in Project Meridian, and what did it use before?"
    )
    admitted, method = hybrid.recall_hybrid(
        memory,
        query,
        user_id="agent",
        cfg={"recall_limit": 2},
        limit=2,
        tier="zero_llm",
        evidence_needs=("prior", "current"),
    )
    assert len(admitted) == 2
    assert {record["id"] for record in admitted} == {
        "meridian-v1",
        "meridian-v2",
    }
    assert "links1" in method
    assert memory.vector_store.search_queries[0] == query
    assert memory.vector_store.get_calls == ["meridian-v1"]


def test_canonical_hybrid_without_declared_need_preserves_dense_pool(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    from fidelis import recall_hybrid as hybrid

    memory = _Memory()
    for record_id in ("meridian-v1", "meridian-v2", "unrelated"):
        record = records[record_id]
        safe_add(
            memory,
            record["text"],
            "agent",
            kind="store",
            record_id=record["id"],
            metadata=record["metadata"],
        )
    memory.vector_store.search_ids = ["meridian-v2", "unrelated"]
    monkeypatch.setattr(
        hybrid,
        "_embed_docs",
        lambda texts, cfg: [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        hybrid,
        "_embed_queries",
        lambda texts, cfg: [[1.0, 1.0] for _ in texts],
    )
    admitted, method = hybrid.recall_hybrid(
        memory,
        "Project Meridian status",
        user_id="agent",
        cfg={"recall_limit": 2},
        limit=2,
        tier="zero_llm",
    )
    assert {record["id"] for record in admitted} == {
        "meridian-v2",
        "unrelated",
    }
    assert "links" not in method
    assert memory.vector_store.get_calls == []


def test_explicit_comparative_orientation_drives_link_admission(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    from fidelis import recall_hybrid as hybrid
    from fidelis import server

    memory = _Memory()
    for record_id in ("meridian-v1", "meridian-v2", "unrelated"):
        record = records[record_id]
        safe_add(
            memory,
            record["text"],
            "agent",
            kind="store",
            record_id=record["id"],
            metadata=record["metadata"],
        )
    memory.vector_store.search_ids = ["meridian-v2", "unrelated"]
    monkeypatch.setattr(
        hybrid,
        "_embed_docs",
        lambda texts, cfg: [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        hybrid,
        "_embed_queries",
        lambda texts, cfg: [[1.0, 1.0] for _ in texts],
    )
    query = (
        "What changed in Project Meridian, and what did it use before?"
    )
    result = server._run_cogito_hermeneutics(
        memory,
        {
            "user_id": "agent",
            "recall_limit": 30,
            "ephemera_filter": False,
            "supersession_pointers_path": None,
        },
        {"text": query, "automatic": True, "limit": 3},
    )
    assert result["plan"]["evidence_needs"] == ["prior", "current"]
    assert "links1" in result["method"]
    assert result["candidate_count"] == 3
    packet = result["declared_relation_evidence"]
    assert packet["status"] == "declared_linked_evidence"
    assert {record["id"] for record in packet["records"]} == {
        "meridian-v1",
        "meridian-v2",
    }
    assert all(
        record["text"]
        in {
            records["meridian-v1"]["text"],
            records["meridian-v2"]["text"],
        }
        for record in packet["records"]
    )


def test_no_relation_or_missing_target_exposes_no_packet(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    memory = _Memory()
    for record_id in ("meridian-v2", "unrelated"):
        record = records[record_id]
        safe_add(
            memory,
            record["text"],
            "agent",
            kind="store",
            record_id=record["id"],
            metadata=record["metadata"],
        )
    candidates = [_ingested_candidate(memory, index) for index in range(2)]
    packet = expose_linked_raw_evidence(
        evidence_needs=("prior", "current"),
        candidates=candidates,
    )
    assert packet["status"] == (
        "no_declared_relation_in_admitted_candidates"
    )
    assert packet["records"] == []


def test_queued_relation_ingest_replays_with_same_id_and_envelope(
    monkeypatch, tmp_path, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(tmp_path / "queue"))
    broken = _Memory()
    broken.embedding_model = _BrokenEmbedder()
    record = records["meridian-v2"]
    queued = safe_add(
        broken,
        record["text"],
        "agent",
        kind="store",
        record_id=record["id"],
        metadata=record["metadata"],
    )
    assert queued["status"] == "queued"
    assert queued["id"] == record["id"]

    from fidelis.degrade import replay_queue

    working = _Memory()
    replayed = replay_queue(working, user_id="agent")
    assert replayed["replayed"] == 1
    inserted = working.vector_store.inserts[0]
    assert inserted["ids"] == [record["id"]]
    assert inserted["payloads"][0]["data"] == record["text"]
    envelope = envelope_from_metadata(inserted["payloads"][0])
    assert envelope["record"]["id"] == record["id"]


def test_relation_replay_rejects_duplicate_id_with_different_raw_text(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    memory = _Memory()
    record = records["meridian-v2"]
    safe_add(
        memory,
        record["text"],
        "agent",
        kind="store",
        record_id=record["id"],
        metadata=record["metadata"],
    )

    def duplicate_insert(**kwargs):
        raise ValueError("duplicate id already exists")

    memory.vector_store.insert = duplicate_insert
    from fidelis.degrade import _replay_verbatim

    with pytest.raises(
        RuntimeError,
        match="different raw evidence",
    ):
        _replay_verbatim(
            memory,
            {
                "id": record["id"],
                "text": record["text"] + " conflicting",
                "user_id": "agent",
                "metadata": record["metadata"],
            },
        )


def test_tampered_raw_text_hash_cannot_expose_linked_packet(
    monkeypatch, records
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    memory = _Memory()
    for record_id in ("meridian-v1", "meridian-v2"):
        record = records[record_id]
        safe_add(
            memory,
            record["text"],
            "agent",
            kind="store",
            record_id=record["id"],
            metadata=record["metadata"],
        )
    candidates = [_ingested_candidate(memory, index) for index in range(2)]
    candidates[0]["text"] = "tampered source text"
    packet = expose_linked_raw_evidence(
        evidence_needs=("prior", "current"),
        candidates=candidates,
    )
    assert packet["records"] == []


def test_unknown_schema_is_backcompat_ignored():
    assert envelope_from_metadata(
        {
            PAYLOAD_FIELD: json.dumps(
                {"schema_version": "fidelis.relation-envelope/v999"}
            )
        }
    ) is None
