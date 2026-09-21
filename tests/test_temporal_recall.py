"""Unit tests for fidelis.temporal_recall (docs/TIME-AWARE-SPEC.md).

temporal_recall.py was written directly against the contract and had no unit
tests of its own before this file. Everything here uses small, local fakes —
a fake ``memory`` whose ``vector_store.get(id)`` returns an object with
``.payload``, and whose ``vector_store.collection.get(where=..., include=...)``
returns a ``{"ids": [...], "metadatas": [...]}`` mapping — plus the REAL
``fidelis.temporal_index.TemporalIndex`` backed by a throwaway sqlite file
under ``tmp_path``. No network, no Ollama, no real Chroma store.
"""

from __future__ import annotations

import copy
import json
import logging
from types import SimpleNamespace

import pytest

from fidelis.temporal import SCHEMA, SupersessionIndex, content_sha256, format_instant, parse_instant
from fidelis.temporal_index import TemporalIndex
from fidelis.temporal_recall import (
    _SCRATCH,
    carry_payload,
    overfetch,
    parse_as_of,
    reconcile_index,
    temporal_view,
)


# ── small fakes ──────────────────────────────────────────────────────────────


class _FakeCollection:
    """Stand-in for mem0's ``vector_store.collection`` (a chromadb.Collection)."""

    def __init__(self, store: "_FakeVectorStore") -> None:
        self._store = store
        self.calls = 0
        self.raises: Exception | None = None

    def get(self, where=None, include=None):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        ids = list(self._store.rows.keys())
        metadatas = [dict(self._store.rows[i]) for i in ids]
        return {"ids": ids, "metadatas": metadatas}


class _FakeVectorStore:
    """Stand-in for mem0's ``vector_store`` — just ``.get`` and ``.collection``."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.get_calls: list[str] = []
        self.raises: Exception | None = None
        self.collection = _FakeCollection(self)

    def get(self, record_id):
        self.get_calls.append(record_id)
        if self.raises is not None:
            raise self.raises
        payload = self.rows.get(record_id)
        if payload is None:
            return None
        return SimpleNamespace(id=record_id, payload=dict(payload))


class _FakeMemory:
    def __init__(self) -> None:
        self.vector_store = _FakeVectorStore()


@pytest.fixture
def memory() -> _FakeMemory:
    return _FakeMemory()


@pytest.fixture
def index_factory(tmp_path):
    created: list[TemporalIndex] = []

    def _make(name: str = "sidecar.temporal.sqlite") -> TemporalIndex:
        idx = TemporalIndex(tmp_path / name)
        created.append(idx)
        return idx

    yield _make
    for idx in created:
        idx.close()


@pytest.fixture
def index(index_factory) -> TemporalIndex:
    return index_factory()


def _carried(**fields) -> dict:
    """A minimal STORED payload (as carry_payload would receive from a
    retriever that already holds it), spread into a hit dict."""
    base = {"temporal_schema": SCHEMA, "recorded_at": "2026-01-01T00:00:00Z"}
    base.update(fields)
    return carry_payload(base)


# ── (a) parse_as_of ──────────────────────────────────────────────────────────


def test_parse_as_of_none_and_empty_mean_now():
    assert parse_as_of(None) is None
    assert parse_as_of("") is None


def test_parse_as_of_valid_iso():
    assert parse_as_of("2026-01-01T00:00:00Z") == parse_instant("2026-01-01T00:00:00Z")
    assert parse_as_of("2026-01-01") == parse_instant("2026-01-01")


def test_parse_as_of_garbage_raises_value_error():
    with pytest.raises(ValueError):
        parse_as_of("last week")
    with pytest.raises(ValueError):
        parse_as_of(1750000000)


# ── (b) overfetch ────────────────────────────────────────────────────────────


def test_overfetch_unchanged_without_as_of():
    assert overfetch(10, None) == 10
    assert overfetch(1, None) == 1
    assert overfetch(1000, None) == 1000


def test_overfetch_deeper_with_as_of():
    now = parse_instant("2026-01-01T00:00:00Z")
    assert overfetch(10, now) == 30  # max(10*3, 10+10) = 30
    assert overfetch(1, now) == 11  # max(3, 11) = 11


def test_overfetch_capped_at_100_with_as_of():
    now = parse_instant("2026-01-01T00:00:00Z")
    assert overfetch(40, now) == 100
    assert overfetch(10_000, now) == 100


# ── (c) carry_payload ────────────────────────────────────────────────────────


def test_carry_payload_keeps_only_temporal_fields():
    payload = {
        "data": "some verbatim text",
        "user_id": "agent",
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "recorded_at_source": "write",
        "content_sha256": "abc123",
        "event_at": "2026-01-01T00:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_to": "2027-01-01T00:00:00Z",
        "supersedes_json": "[]",
        "source": "unit-test",
        "some_unrelated_extra_field": "zzz",
    }
    out = carry_payload(payload)
    assert set(out.keys()) == {_SCRATCH}
    scratch = out[_SCRATCH]
    assert "data" not in scratch
    assert "user_id" not in scratch
    assert "some_unrelated_extra_field" not in scratch
    assert scratch["temporal_schema"] == SCHEMA
    assert scratch["recorded_at"] == "2026-01-01T00:00:00Z"
    assert scratch["source"] == "unit-test"


@pytest.mark.parametrize("bad", [None, "not a mapping", 42, [1, 2, 3]])
def test_carry_payload_non_mapping_is_empty(bad):
    assert carry_payload(bad) == {_SCRATCH: {}}


def test_carried_payload_hit_causes_zero_vector_store_get_calls(memory, index):
    hit = {"id": "rec-1", "text": "hello world", "score": 0.9, **_carried()}
    out = temporal_view([hit], memory=memory, index=index)
    assert memory.vector_store.get_calls == []
    assert len(out) == 1
    assert out[0]["text"] == "hello world"


# ── (d) id recovery by content hash ──────────────────────────────────────────


def test_bare_hit_recovers_id_and_temporal_fields_by_hash(memory, index):
    text = "The deploy target is staging."
    sha = content_sha256(text)
    memory.vector_store.rows["rec-1"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "recorded_at_source": "write",
        "content_sha256": sha,
        "data": text,
        "user_id": "agent",
    }
    index.record_write("rec-1", sha, "2026-01-01T00:00:00Z", [])

    hit = {"text": text, "score": 0.7}
    out = temporal_view([hit], memory=memory, index=index)

    assert out[0]["id"] == "rec-1"
    assert out[0]["temporal"]["recorded_at_known"] is True
    assert out[0]["temporal"]["recorded_at"] == "2026-01-01T00:00:00Z"
    assert memory.vector_store.get_calls == ["rec-1"]


def test_duplicate_text_hash_never_guesses_an_id(memory, index):
    text = "The deploy target is staging."
    sha = content_sha256(text)
    for rid, ts in (("rec-1", "2026-01-01T00:00:00Z"), ("rec-2", "2026-02-01T00:00:00Z")):
        memory.vector_store.rows[rid] = {
            "temporal_schema": SCHEMA,
            "recorded_at": ts,
            "content_sha256": sha,
            "data": text,
        }
        index.record_write(rid, sha, ts, [])

    hit = {"text": text, "score": 0.7}
    out = temporal_view([hit], memory=memory, index=index)

    assert "id" not in out[0]
    assert out[0]["temporal"]["recorded_at_known"] is False
    # Two ids share this hash — picking either would be a guess, so no
    # store lookup is even attempted.
    assert memory.vector_store.get_calls == []


# ── (e) hits that already carry temporal fields; retriever payload survives ──


def test_hit_with_temporal_under_payload_key_used_as_is(memory, index):
    hit = {
        "id": "rec-9",
        "text": "some text",
        "score": 0.5,
        "payload": {
            "temporal_schema": SCHEMA,
            "recorded_at": "2026-03-01T00:00:00Z",
            "content_sha256": "zzz",
        },
    }
    out = temporal_view([hit], memory=memory, index=index)
    assert memory.vector_store.get_calls == []
    assert out[0]["temporal"]["recorded_at"] == "2026-03-01T00:00:00Z"
    assert out[0]["payload"] == hit["payload"]


def test_hit_with_temporal_under_metadata_key_used_as_is(memory, index):
    hit = {
        "id": "rec-10",
        "text": "some other text",
        "score": 0.5,
        "metadata": {
            "temporal_schema": SCHEMA,
            "recorded_at": "2026-04-01T00:00:00Z",
        },
    }
    out = temporal_view([hit], memory=memory, index=index)
    assert memory.vector_store.get_calls == []
    assert out[0]["temporal"]["recorded_at"] == "2026-04-01T00:00:00Z"
    assert out[0]["metadata"] == hit["metadata"]
    assert "payload" not in out[0]


def test_retriever_payload_without_temporal_fields_survives_id_lookup(memory, index):
    """A retriever's OWN payload (unrelated to temporal) must come back
    untouched even when temporal_view has to fetch temporal fields by id —
    the internal scratch mapping must never leak into / replace it."""
    memory.vector_store.rows["rec-42"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-05-01T00:00:00Z",
        "content_sha256": "sha-42",
        "data": "the fetched fact",
    }
    retriever_payload = {"source": "retriever-x", "extra": 123}
    hit = {
        "id": "rec-42",
        "text": "the fetched fact",
        "score": 0.8,
        "payload": dict(retriever_payload),
    }

    out = temporal_view([hit], memory=memory, index=index)

    assert out[0]["payload"] == retriever_payload
    assert out[0]["temporal"]["recorded_at"] == "2026-05-01T00:00:00Z"
    assert memory.vector_store.get_calls == ["rec-42"]


# ── (f) evaluated_at presence + byte-stability ───────────────────────────────


def test_evaluated_at_absent_without_as_of_present_with_as_of(memory, index):
    hit = {"id": "rec-1", "text": "t", "score": 0.5, **_carried()}

    out_no_as_of = temporal_view([hit], memory=memory, index=index)
    assert "evaluated_at" not in out_no_as_of[0]["temporal"]

    as_of = parse_instant("2026-06-01T00:00:00Z")
    out_with_as_of = temporal_view([hit], memory=memory, index=index, as_of=as_of)
    assert out_with_as_of[0]["temporal"]["evaluated_at"] == format_instant(as_of)


def test_repeated_calls_without_as_of_are_byte_stable(memory, index):
    hit = {"id": "rec-1", "text": "t", "score": 0.5, **_carried()}
    first = temporal_view([hit], memory=memory, index=index)
    second = temporal_view([hit], memory=memory, index=index)
    assert first == second


# ── (g) fail-open ─────────────────────────────────────────────────────────


def test_fail_open_index_none(memory):
    hits = [
        {"id": "r1", "text": "alpha", "score": 0.9},
        {"text": "beta", "score": 0.5},
    ]
    out = temporal_view(hits, memory=memory, index=None)
    assert len(out) == 2
    assert [h["text"] for h in out] == ["alpha", "beta"]
    for h in out:
        assert h["temporal"]["index"] == "unavailable"


def test_fail_open_supersession_index_raises(memory):
    class BrokenSupersessionIndex:
        def ids_by_sha(self, sha):
            return []

        def supersession_index(self):
            raise RuntimeError("sidecar unreadable")

    hits = [{"id": "r1", "text": "alpha", "score": 0.9}]
    out = temporal_view(hits, memory=memory, index=BrokenSupersessionIndex())
    assert len(out) == 1
    assert out[0]["text"] == "alpha"
    assert out[0]["temporal"]["index"] == "unavailable"


def test_fail_open_ids_by_sha_raises(memory):
    class BrokenIdsBySha:
        def ids_by_sha(self, sha):
            raise RuntimeError("index lookup boom")

        def supersession_index(self):
            return SupersessionIndex()

    hits = [{"text": "alpha only text, no id", "score": 0.9}]
    out = temporal_view(hits, memory=memory, index=BrokenIdsBySha())
    assert len(out) == 1
    assert out[0]["text"] == "alpha only text, no id"
    assert "temporal" in out[0]


def test_fail_open_vector_store_get_raises(index):
    text = "alpha only text for lookup"
    sha = content_sha256(text)
    index.record_write("rec-1", sha, "2026-01-01T00:00:00Z", [])

    memory = _FakeMemory()
    memory.vector_store.raises = RuntimeError("store lookup boom")

    hits = [{"text": text, "score": 0.9}]
    out = temporal_view(hits, memory=memory, index=index)
    assert len(out) == 1
    assert out[0]["text"] == text
    assert "temporal" in out[0]


# ── (h) non-mapping passthrough + no mutation ───────────────────────────────


def test_non_mapping_items_pass_through_untouched(memory, index):
    items = ["not-a-dict", 42, None, ["nested", "list"]]
    out = temporal_view(list(items), memory=memory, index=index)
    assert out == items


def test_input_list_and_dicts_are_not_mutated(memory, index):
    hit = {"id": "rec-1", "text": "hello", "score": 0.9}
    hits = [hit]
    original = copy.deepcopy(hits)

    temporal_view(hits, memory=memory, index=index)

    assert hits == original
    assert hit == original[0]


# ── (i) eligibility window + pool-only replacement (WORK PACKET E) ──────────
#
# The test this section used to contain asserted that the limit is applied
# AFTER current-first ordering of the WHOLE pool, such that "a superseded hit
# ranked first by relevance ends up after a current one, and limit=1 returns
# the current one". That pinned the DEFECT this packet fixes (it was
# specified wrongly by the lead): partitioning the entire over-fetched pool
# before trimming silently drops a superseded record whenever the pool holds
# >= `limit` current hits — which is the normal case against a real store,
# not the edge case. See docs/TIME-AWARE-SPEC.md ("eligibility window" /
# "pool-only replacement") and TestSupersededKeptInEligibilityWindow below
# for the full defect scenario.


def test_limit1_pulls_the_superseder_in_when_it_is_in_the_pool(memory, index):
    # 'B' supersedes 'A'. Relevance ranks A first even though it is stale.
    index.record_write("A", "sha-A", "2026-01-01T00:00:00Z", [])
    index.record_write("B", "sha-B", "2026-01-02T00:00:00Z", ["A"])

    hits = [
        {"id": "A", "text": "old fact", "score": 0.99, **_carried(recorded_at="2026-01-01T00:00:00Z")},
        {"id": "B", "text": "new fact", "score": 0.10, **_carried(recorded_at="2026-01-02T00:00:00Z")},
    ]

    out = temporal_view(hits, memory=memory, index=index, limit=1)

    # The eligibility window is [A] (top-1 by relevance). B is in the POOL,
    # so it is pulled in to keep the replacement visible: length =
    # limit(1) + 1 pulled, B ranked above A, A still present and annotated.
    assert [h["id"] for h in out] == ["B", "A"]
    assert len(out) == 2
    assert out[1]["temporal"]["status"] == "superseded"
    assert out[1]["temporal"]["superseded_by"] == ["B"]
    assert out[0]["pulled_by"] == {"supersedes": "A"}


def test_limit1_unrelated_current_hit_never_displaces_the_superseded_one(memory, index):
    index.record_write("A", "sha-A", "2026-01-01T00:00:00Z", [])
    # A's superseder is declared in the sidecar but never appears in THIS
    # pool (e.g. it fell outside the recall's relevance window entirely) —
    # there is nothing to pull in from.
    index.record_write("phantom", "sha-phantom", "2026-01-02T00:00:00Z", ["A"])

    hits = [
        {"id": "A", "text": "old fact", "score": 0.99, **_carried(recorded_at="2026-01-01T00:00:00Z")},
        {"id": "C", "text": "unrelated current fact", "score": 0.50, **_carried(recorded_at="2026-01-03T00:00:00Z")},
    ]

    out = temporal_view(hits, memory=memory, index=index, limit=1)

    # A is ranked first by relevance, so it — not the unrelated current C —
    # is the eligibility window. An unrelated current record must never
    # displace the relevant superseded one just because it would sort
    # "current first" if it were in the window.
    assert [h["id"] for h in out] == ["A"]
    assert out[0]["temporal"]["status"] == "superseded"


def _rec(hit_id: str, ts: str, **extra) -> dict:
    """A pool hit carrying its own temporal payload (no store/index lookup
    needed to resolve it) — id + score + a carried scratch payload."""
    return {
        "id": hit_id,
        "text": f"content for {hit_id}",
        "score": 0.5,
        **_carried(recorded_at=ts, **extra),
    }


class TestSupersededKeptInEligibilityWindow:
    """WORK PACKET E — the field defect: a store with A ('...port 8811.')
    then B ('...port 8822...', supersedes=[A]) returned B plus four unrelated
    CURRENT hits for a query on A's exact text at limit=5 — A was not in the
    response at all. Cause: ``temporal_view`` handed the WHOLE over-fetched
    pool to ``apply_temporal``, which stable-partitions current-first over
    the entire pool and only THEN trims to ``limit``; any store with >=
    `limit` current hits pushes a superseded/expired hit past the cut with
    no trace (violates invariants 4 and 5 in docs/TIME-AWARE-SPEC.md).

    Fix: the eligibility window is the first `limit` hits BY RELEVANCE
    (invariant 4), computed before any current-first reordering. A
    superseded record that survives into the window then pulls its newest
    superseder INTO VIEW from the POOL ONLY (never a fresh store fetch,
    which would bypass the ephemera/user filters already applied upstream
    and could surface a stale mid-chain record — an earlier version that did
    fetch by id was removed on review). The result may exceed `limit` by at
    most the number of records pulled in this way.
    """

    N = 20

    def _seed(self, index, *, supersedes_map=None, expired_ranks=(), extra_edges=()):
        """`N` unrelated 'current' pool hits, hit-1..hit-N, with strictly
        increasing `recorded_at`. `supersedes_map={7: [1]}` means hit-7
        supersedes hit-1. `extra_edges` are `(superseder_id, ts,
        [target_ids])` sidecar edges for a superseder that is NOT added to
        the returned pool — it exists in the store/sidecar but fell outside
        this recall's relevance-selected pool."""
        supersedes_map = supersedes_map or {}
        hits = []
        for i in range(1, self.N + 1):
            hit_id = f"hit-{i}"
            ts = f"2026-01-01T00:{i:02d}:00Z"
            targets = [f"hit-{t}" for t in supersedes_map.get(i, [])]
            index.record_write(hit_id, f"sha-{hit_id}", ts, targets)
            extra = {"valid_to": "2020-01-01T00:00:00Z"} if i in expired_ranks else {}
            hits.append(_rec(hit_id, ts, **extra))
        for superseder_id, ts, targets in extra_edges:
            index.record_write(superseder_id, f"sha-{superseder_id}", ts, targets)
        return hits

    def test_superseded_rank1_pulls_its_superseder_into_the_window(self, memory, index):
        hits = self._seed(index, supersedes_map={7: [1]})

        out = temporal_view(hits, memory=memory, index=index, limit=5)
        ids = [h["id"] for h in out]

        assert set(ids) <= {f"hit-{i}" for i in range(1, self.N + 1)}, "record from outside the pool"
        assert "hit-1" in ids, "superseded hit #1 was silently dropped"
        assert "hit-7" in ids
        hit1 = next(h for h in out if h["id"] == "hit-1")
        hit7 = next(h for h in out if h["id"] == "hit-7")
        assert hit1["temporal"]["status"] == "superseded"
        assert hit1["temporal"]["superseded_by"] == ["hit-7"]
        assert hit7.get("pulled_by") == {"supersedes": "hit-1"}
        assert ids.index("hit-7") < ids.index("hit-1")
        assert len(out) <= 6  # limit(5) + 1 pulled

    def test_superseder_absent_from_pool_nothing_pulled(self, memory, index):
        hits = self._seed(
            index, extra_edges=[("phantom-superseder", "2026-01-01T00:30:00Z", ["hit-1"])]
        )

        out = temporal_view(hits, memory=memory, index=index, limit=5)
        ids = [h["id"] for h in out]

        assert "hit-1" in ids
        assert "phantom-superseder" not in ids, "never fetched from the store"
        hit1 = next(h for h in out if h["id"] == "hit-1")
        assert hit1["temporal"]["status"] == "superseded"
        assert hit1["temporal"]["superseded_by"] == ["phantom-superseder"]
        assert not any("pulled_by" in h for h in out)
        assert len(out) == 5

    def test_expired_record_demoted_within_window(self, memory, index):
        hits = self._seed(index, expired_ranks={1})

        out = temporal_view(hits, memory=memory, index=index, limit=5)
        ids = [h["id"] for h in out]

        assert ids.count("hit-1") == 1
        hit1 = next(h for h in out if h["id"] == "hit-1")
        assert hit1["temporal"]["status"] == "expired"
        current_positions = [i for i, h in enumerate(out) if h["temporal"]["status"] == "current"]
        assert all(p < ids.index("hit-1") for p in current_positions)
        assert len(out) == 5

    def test_limit_none_whole_pool_no_growth(self, memory, index):
        hits = self._seed(index, supersedes_map={7: [1]})

        out = temporal_view(hits, memory=memory, index=index, limit=None)
        ids = [h["id"] for h in out]

        assert len(out) == self.N
        assert set(ids) == {f"hit-{i}" for i in range(1, self.N + 1)}
        assert not any("pulled_by" in h for h in out)  # already in the (unbounded) window
        hit1 = next(h for h in out if h["id"] == "hit-1")
        assert hit1["temporal"]["status"] == "superseded"
        assert ids.index("hit-7") < ids.index("hit-1")

    def test_as_of_between_writes_hit1_current_hit7_absent_nothing_pulled(self, memory, index):
        hits = self._seed(index, supersedes_map={7: [1]})
        as_of = parse_instant("2026-01-01T00:04:30Z")  # strictly between hit-1 and hit-7

        out = temporal_view(hits, memory=memory, index=index, limit=5, as_of=as_of)
        ids = [h["id"] for h in out]

        assert "hit-7" not in ids, "hit-7 was recorded after as_of; must be invisible"
        assert "hit-1" in ids
        hit1 = next(h for h in out if h["id"] == "hit-1")
        assert hit1["temporal"]["status"] == "current"
        assert hit1["temporal"]["superseded_by"] == []
        assert not any("pulled_by" in h for h in out)
        assert len(out) == 4  # only hit-1..hit-4 are visible as of this instant

    def test_three_link_chain_pulls_the_newest_not_the_middle(self, memory, index):
        index.record_write("chain-a", "sha-a", "2026-01-01T00:01:00Z", [])
        index.record_write("chain-b", "sha-b", "2026-01-01T00:02:00Z", ["chain-a"])
        index.record_write("chain-c", "sha-c", "2026-01-01T00:03:00Z", ["chain-b"])
        pool = [
            _rec("chain-a", "2026-01-01T00:01:00Z"),
            _rec("filler-1", "2026-01-01T01:01:00Z"),
            _rec("filler-2", "2026-01-01T01:02:00Z"),
            _rec("chain-b", "2026-01-01T00:02:00Z"),
            _rec("chain-c", "2026-01-01T00:03:00Z"),
        ]

        out = temporal_view(pool, memory=memory, index=index, limit=3)
        ids = [h["id"] for h in out]

        assert "chain-b" not in ids, "the middle link must not be pulled"
        assert "chain-c" in ids, "the newest superseder must be pulled"
        assert "chain-a" in ids
        chain_a = next(h for h in out if h["id"] == "chain-a")
        chain_c = next(h for h in out if h["id"] == "chain-c")
        assert chain_a["temporal"]["status"] == "superseded"
        assert chain_c.get("pulled_by") == {"supersedes": "chain-a"}
        assert ids.index("chain-c") < ids.index("chain-a")
        assert len(out) == 4  # limit(3) + 1 pulled

    def test_cycle_terminates_without_crash(self, memory, index):
        # Contradictory data (each declares it supersedes the other) must
        # not infinite-loop the chain walk.
        index.record_write("cyc-a", "sha-cyc-a", "2026-01-01T00:01:00Z", ["cyc-b"])
        index.record_write("cyc-b", "sha-cyc-b", "2026-01-01T00:02:00Z", ["cyc-a"])
        pool = [
            _rec("cyc-a", "2026-01-01T00:01:00Z"),
            _rec("filler-1", "2026-01-01T01:01:00Z"),
            _rec("cyc-b", "2026-01-01T00:02:00Z"),
        ]

        out = temporal_view(pool, memory=memory, index=index, limit=2)
        ids = [h["id"] for h in out]

        assert set(ids) <= {"cyc-a", "cyc-b", "filler-1"}
        assert len(out) <= 3

    def test_historical_true_relevance_order_annotated_no_pull(self, memory, index):
        hits = self._seed(index, supersedes_map={7: [1]})

        out = temporal_view(hits, memory=memory, index=index, limit=5, historical=True)
        ids = [h["id"] for h in out]

        # Pure relevance order, top-limit, annotated only — historical never
        # reorders and (decision, documented here: recommended "no") never
        # pulls a replacement in either.
        assert ids == [f"hit-{i}" for i in range(1, 6)]
        assert len(out) == 5
        hit1 = next(h for h in out if h["id"] == "hit-1")
        assert hit1["temporal"]["status"] == "superseded"
        assert not any("pulled_by" in h for h in out)


# ── (j) reconcile_index ──────────────────────────────────────────────────────


def test_reconcile_additive_repair_then_noop_once_per_instance(memory, index):
    memory.vector_store.rows["rec-1"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "content_sha256": "sha-1",
        "data": "fact one",
    }
    memory.vector_store.rows["rec-2"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-02T00:00:00Z",
        "content_sha256": "sha-2",
        "data": "fact two",
    }
    assert index.record_ids() == []

    repaired = reconcile_index(memory, index)
    assert repaired == 2
    assert sorted(index.record_ids()) == ["rec-1", "rec-2"]

    repaired_again = reconcile_index(memory, index)
    assert repaired_again is None
    assert memory.vector_store.collection.calls == 1  # scanned only once


def test_reconcile_detects_ghost_id_despite_equal_counts_additive(memory, index_factory):
    memory.vector_store.rows["S"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "content_sha256": "sha-s",
        "data": "unindexed store row",
    }
    idx = index_factory("additive.temporal.sqlite")
    idx.record_write("G", "sha-ghost", "2020-01-01T00:00:00Z", [])
    assert idx.record_ids() == ["G"]

    repaired = reconcile_index(memory, idx, full=False)

    assert repaired == 1
    assert set(idx.record_ids()) == {"G", "S"}  # additive: G is left alone


def test_reconcile_full_rebuild_clears_ghost_id(memory, index_factory):
    memory.vector_store.rows["S"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "content_sha256": "sha-s",
        "data": "unindexed store row",
    }
    idx = index_factory("full.temporal.sqlite")
    idx.record_write("G", "sha-ghost", "2020-01-01T00:00:00Z", [])
    assert idx.record_ids() == ["G"]

    repaired = reconcile_index(memory, idx, full=True)

    assert repaired == 1
    assert set(idx.record_ids()) == {"S"}  # full rebuild: ghost gone


def test_reconcile_failing_scan_logs_warning_and_does_not_rescan(memory, index, caplog):
    memory.vector_store.collection.raises = RuntimeError("scan exploded")

    with caplog.at_level(logging.WARNING, logger="fidelis.temporal_recall"):
        result = reconcile_index(memory, index)
    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert memory.vector_store.collection.calls == 1

    result_again = reconcile_index(memory, index)
    assert result_again is None
    assert memory.vector_store.collection.calls == 1  # no re-scan


def test_reconcile_additive_never_deletes_existing_rows_and_edges(memory, index):
    memory.vector_store.rows["new-in-store"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-03-01T00:00:00Z",
        "content_sha256": "sha-new",
        "data": "brand-new fact",
    }
    index.record_write("existing-a", "sha-a", "2026-01-01T00:00:00Z", [])
    index.record_write("existing-b", "sha-b", "2026-01-02T00:00:00Z", ["existing-a"])
    assert index.is_superseded("existing-a")

    repaired = reconcile_index(memory, index, full=False)

    assert repaired == 1
    assert set(index.record_ids()) == {"existing-a", "existing-b", "new-in-store"}
    assert index.is_superseded("existing-a")


def test_reconcile_restores_supersession_edge_declared_in_store_row(memory, index):
    memory.vector_store.rows["target-old"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "content_sha256": "sha-old",
        "data": "the old fact",
    }
    memory.vector_store.rows["superseder-new"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-02-01T00:00:00Z",
        "content_sha256": "sha-new",
        "data": "the new fact",
        "supersedes_json": json.dumps(["target-old"]),
    }
    assert not index.is_superseded("target-old")

    repaired = reconcile_index(memory, index, full=False)

    assert repaired == 2
    assert index.is_superseded("target-old")


def test_reconcile_malformed_supersedes_json_does_not_abort_other_rows(memory, index):
    memory.vector_store.rows["bad-row"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-01T00:00:00Z",
        "content_sha256": "sha-bad",
        "data": "row with malformed supersedes_json",
        "supersedes_json": "{not valid json",
    }
    memory.vector_store.rows["good-row"] = {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-01-02T00:00:00Z",
        "content_sha256": "sha-good",
        "data": "row with clean fields",
    }

    repaired = reconcile_index(memory, index, full=False)

    assert repaired == 2
    assert set(index.record_ids()) == {"bad-row", "good-row"}


# ── (k) removed feature is actually gone; its replacement is present ────────
#
# An earlier attempt at "keep the replacement in view" fetched a missing
# superseder from the STORE by id (helpers named `_pull_missing_superseders`
# / `_terminal_superseder`). That was removed on review: it bypassed the
# ephemera filter and the user filter, could surface a mid-chain stale
# record, and could displace the superseded hit it was meant to accompany.
# Those two names must stay gone. WORK PACKET E reintroduces the FEATURE with
# a pool-only helper under a different name (`_replacements_from_pool`), and
# `pulled_by` is now a LEGITIMATE key on a hit that was pulled in this way —
# it is no longer pinned absent, only pinned to the two removed names.


def test_pull_missing_superseders_symbols_are_absent():
    import fidelis.temporal_recall as module

    assert not hasattr(module, "_pull_missing_superseders")
    assert not hasattr(module, "_terminal_superseder")
    assert "_pull_missing_superseders" not in dir(module)
    assert "_terminal_superseder" not in dir(module)
    # The replacement helper (WORK PACKET E) exists under its own name.
    assert hasattr(module, "_replacements_from_pool")


def test_no_hit_carries_pulled_by_when_nothing_was_pulled(memory, index):
    # `pulled_by` is a legitimate key now (see comment above), but it must
    # only appear on a hit that was actually pulled in from the pool. These
    # two hits have no supersession relationship at all, so neither should
    # carry it.
    hits = [
        {"id": "r1", "text": "alpha", "score": 0.9},
        {"text": "beta", "score": 0.5},
    ]
    out = temporal_view(hits, memory=memory, index=index)
    for h in out:
        assert "pulled_by" not in h
        if isinstance(h.get("temporal"), dict):
            assert "pulled_by" not in h["temporal"]
