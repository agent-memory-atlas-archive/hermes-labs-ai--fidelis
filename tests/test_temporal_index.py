"""Contract tests for fidelis.temporal_index (docs/TIME-AWARE-SPEC.md).

Every sqlite file lives under pytest's tmp_path; nothing here touches the
live store or server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from fidelis import temporal_index
from fidelis.temporal_index import TemporalIndex, open_index, sidecar_path

T1 = "2026-09-20T10:00:00Z"
T2 = "2026-09-20T11:00:00Z"
T3 = "2026-09-20T12:00:00Z"


@pytest.fixture(autouse=True)
def _reset_warned():
    temporal_index._warned_paths.clear()
    yield
    temporal_index._warned_paths.clear()


@pytest.fixture
def idx(tmp_path):
    index = TemporalIndex(tmp_path / "store.temporal.sqlite")
    yield index
    index.close()


# --- path rule ---------------------------------------------------------------

def test_sidecar_path_rule(tmp_path):
    store = tmp_path / "chroma_db"
    assert sidecar_path(store) == tmp_path / "chroma_db.temporal.sqlite"
    assert sidecar_path(str(store)) == tmp_path / "chroma_db.temporal.sqlite"
    # suffixes on the store name are kept, not replaced
    assert sidecar_path(tmp_path / "mem.v2.db").name == "mem.v2.db.temporal.sqlite"
    assert isinstance(sidecar_path("relative/store"), Path)
    assert sidecar_path("relative/store") == Path("relative") / "store.temporal.sqlite"


# --- schema / pragmas --------------------------------------------------------

def test_schema_wal_and_busy_timeout(idx):
    conn = idx._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    cols = {r[1]: r for r in conn.execute("PRAGMA table_info(records)")}
    # PACKET F2c: valid_from/valid_to were added (nullable) so /stats can
    # compute current/superseded/expired/not-yet-valid counts from this
    # sidecar alone; see test_migration_adds_valid_from_valid_to_columns.
    assert list(cols) == ["record_id", "sha", "recorded_at", "valid_from", "valid_to"]
    assert cols["record_id"][5] == 1  # primary key
    scols = {r[1]: r[5] for r in conn.execute("PRAGMA table_info(supersedes)")}
    assert list(scols) == ["target_id", "superseder_id", "recorded_at"]
    assert scols["target_id"] and scols["superseder_id"] and not scols["recorded_at"]
    indexed = [
        conn.execute(f"PRAGMA index_info({r[1]!r})").fetchall()
        for r in conn.execute("PRAGMA index_list(records)")
    ]
    assert any([c[2] for c in cols_] == ["sha"] for cols_ in indexed)


def test_reopen_preserves_contents(tmp_path):
    p = tmp_path / "s.temporal.sqlite"
    a = TemporalIndex(p)
    a.record_write("A", "sha-a", T1, [])
    a.record_write("B", "sha-b", T2, ["A"])
    a.close()
    b = TemporalIndex(p)
    try:
        assert b.find_by_sha("sha-a") == "A"
        assert b.edges() == [("A", "B", T2)]
    finally:
        b.close()


def test_close_is_idempotent(tmp_path):
    index = TemporalIndex(tmp_path / "c.sqlite")
    index.close()
    index.close()


# --- record_write / find_by_sha ----------------------------------------------

def test_find_by_sha_missing(idx):
    assert idx.find_by_sha("nope") is None


def test_record_write_and_find(idx):
    idx.record_write("A", "sha-a", T1, [])
    assert idx.find_by_sha("sha-a") == "A"
    assert idx.count() == 1


def test_record_write_is_idempotent(idx):
    idx.record_write("B", "sha-b", T2, ["A"])
    idx.record_write("B", "sha-b", T2, ["A"])
    idx.record_write("B", "sha-b", T2, ["A", "A"])
    assert idx.count() == 1
    assert idx.edges() == [("A", "B", T2)]


def test_first_write_of_an_id_wins(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("A", "sha-other", T3, [])
    assert idx.find_by_sha("sha-a") == "A"
    assert idx.find_by_sha("sha-other") is None


def test_find_by_sha_returns_oldest(idx):
    idx.record_write("zz-late", "same", T3, [])
    idx.record_write("mm-early", "same", T1, [])
    idx.record_write("aa-mid", "same", T2, [])
    assert idx.find_by_sha("same") == "mm-early"


def test_find_by_sha_tie_breaks_on_smallest_id(idx):
    idx.record_write("id-b", "same", T1, [])
    idx.record_write("id-a", "same", T1, [])
    idx.record_write("id-c", "same", T1, [])
    assert idx.find_by_sha("same") == "id-a"


def test_find_by_sha_mixed_precision_is_chronological(idx):
    # "…:00Z" sorts AFTER "…:00.500Z" lexically but is the earlier instant.
    idx.record_write("frac", "same", "2026-09-20T10:00:00.500Z", [])
    idx.record_write("whole", "same", "2026-09-20T10:00:00Z", [])
    assert idx.find_by_sha("same") == "whole"


def test_unparseable_recorded_at_sorts_last_not_crash(idx):
    idx.record_write("junk", "same", "not-a-date", [])
    idx.record_write("good", "same", T3, [])
    assert idx.find_by_sha("same") == "good"


def test_supersedes_none_and_bare_string(idx):
    idx.record_write("A", "sha-a", T1, None)  # type: ignore[arg-type]
    idx.record_write("B", "sha-b", T2, "A")  # type: ignore[arg-type]
    assert idx.edges() == [("A", "B", T2)]


def test_bad_supersedes_entries_are_ignored(idx):
    idx.record_write("B", "sha-b", T2, ["A", "", None, 7, "B"])  # type: ignore[list-item]
    # empty / non-str dropped; self-edge B->B dropped
    assert idx.edges() == [("A", "B", T2)]


# --- edges -------------------------------------------------------------------

def test_edges_empty(idx):
    assert idx.edges() == []


def test_edges_carry_superseder_recorded_at(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, ["A"])
    idx.record_write("C", "sha-c", T3, ["A", "B"])
    edges = idx.edges()
    assert edges == [("A", "B", T2), ("A", "C", T3), ("B", "C", T3)]
    assert all(isinstance(e, tuple) and len(e) == 3 for e in edges)
    assert all(isinstance(x, str) for e in edges for x in e)


def test_edge_to_unknown_target_is_kept(idx):
    # The target may be a legacy row the sidecar has never seen.
    idx.record_write("B", "sha-b", T2, ["legacy-row"])
    assert idx.edges() == [("legacy-row", "B", T2)]


# --- supersession_index (needs the sibling module) ---------------------------

def test_supersession_index_from_edges(idx):
    temporal = pytest.importorskip("fidelis.temporal")
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, ["A"])
    idx.record_write("C", "sha-c", T3, ["A"])
    si = idx.supersession_index()
    assert isinstance(si, temporal.SupersessionIndex)
    assert sorted(si.superseders("A")) == ["B", "C"]
    assert si.superseders("B") == []
    between = temporal.parse_instant("2026-09-20T11:30:00Z")
    assert si.superseders("A", as_of=between) == ["B"]


def test_supersession_index_uses_from_edges(idx, monkeypatch):
    """Data path without the sibling module: a stand-in receives edges()."""
    import sys
    import types

    seen = {}

    class FakeSI:
        @classmethod
        def from_edges(cls, edges):
            seen["edges"] = list(edges)
            return cls()

    fake = types.ModuleType("fidelis.temporal")
    fake.SupersessionIndex = FakeSI
    monkeypatch.setitem(sys.modules, "fidelis.temporal", fake)
    idx.record_write("B", "sha-b", T2, ["A"])
    assert isinstance(idx.supersession_index(), FakeSI)
    assert seen["edges"] == [("A", "B", T2)]


# --- rebuild -----------------------------------------------------------------

def _payload(sha, at, supersedes=None, **extra):
    p = {"data": "text", "user_id": "u", "content_sha256": sha, "recorded_at": at}
    if supersedes is not None:
        p["supersedes_json"] = json.dumps(supersedes)
    p.update(extra)
    return p


def test_rebuild_indexes_and_returns_count(idx):
    n = idx.rebuild([
        ("A", _payload("sha-a", T1)),
        ("B", _payload("sha-b", T2, ["A"])),
    ])
    assert n == 2
    assert idx.find_by_sha("sha-a") == "A"
    assert idx.edges() == [("A", "B", T2)]


def test_rebuild_skips_legacy_rows(idx):
    n = idx.rebuild([
        ("legacy-1", {"data": "old", "user_id": "u"}),
        ("legacy-2", {"data": "old", "content_sha256": "x"}),      # no recorded_at
        ("legacy-3", {"data": "old", "recorded_at": T1}),          # no sha
        ("legacy-4", {"content_sha256": "", "recorded_at": T1}),
        ("legacy-5", None),
        ("legacy-6", "not a mapping"),
        (None, _payload("sha-n", T1)),
        "garbage-row",
        ("A", _payload("sha-a", T1)),
    ])
    assert n == 1
    assert idx.count() == 1


def test_rebuild_replaces_prior_contents(idx):
    idx.record_write("OLD", "sha-old", T1, ["OLDER"])
    n = idx.rebuild([("A", _payload("sha-a", T2))])
    assert n == 1
    assert idx.find_by_sha("sha-old") is None
    assert idx.edges() == []


def test_rebuild_is_idempotent(idx):
    rows = [("A", _payload("sha-a", T1)), ("B", _payload("sha-b", T2, ["A"])),
            ("A", _payload("sha-dupe", T3))]
    assert idx.rebuild(rows) == 2
    first = (idx.count(), idx.edges(), idx.find_by_sha("sha-a"))
    assert idx.rebuild(rows) == 2
    assert (idx.count(), idx.edges(), idx.find_by_sha("sha-a")) == first
    assert idx.find_by_sha("sha-dupe") is None


def test_rebuild_empty(idx):
    idx.record_write("A", "sha-a", T1, [])
    assert idx.rebuild([]) == 0
    assert idx.count() == 0


@pytest.mark.parametrize("bad", ["{not json", "[\"A\"", "null", "{\"a\": 1}", "42", "[1, 2]"])
def test_rebuild_survives_malformed_supersedes_json(idx, bad):
    n = idx.rebuild([
        ("A", _payload("sha-a", T1)),
        ("B", {"content_sha256": "sha-b", "recorded_at": T2, "supersedes_json": bad}),
        ("C", _payload("sha-c", T3, ["A"])),
    ])
    assert n == 3
    assert idx.find_by_sha("sha-b") == "B"  # indexed, just without edges
    assert idx.edges() == [("A", "C", T3)]


def test_rebuild_accepts_a_generator(idx):
    gen = ((f"id-{i}", _payload(f"sha-{i}", T1)) for i in range(50))
    assert idx.rebuild(gen) == 50


def test_rebuild_is_atomic_when_source_iterator_dies(idx):
    idx.record_write("KEEP", "sha-keep", T1, ["X"])

    def rows():
        yield ("A", _payload("sha-a", T2))
        raise RuntimeError("store iterator died")

    with pytest.raises(RuntimeError):
        idx.rebuild(rows())
    assert idx.find_by_sha("sha-keep") == "KEEP"
    assert idx.edges() == [("X", "KEEP", T1)]
    assert idx.find_by_sha("sha-a") is None


def test_rebuild_is_one_transaction_other_connection_never_sees_empty(tmp_path):
    p = tmp_path / "atomic.sqlite"
    index = TemporalIndex(p)
    try:
        index.rebuild([(f"id-{i}", _payload(f"sha-{i}", T1)) for i in range(200)])
        stop = threading.Event()
        seen: list[int] = []

        def watch():
            conn = sqlite3.connect(str(p), timeout=5.0)
            try:
                while not stop.is_set():
                    seen.append(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
            finally:
                conn.close()

        t = threading.Thread(target=watch)
        t.start()
        for _ in range(10):
            index.rebuild([(f"id-{i}", _payload(f"sha-{i}", T1)) for i in range(200)])
        stop.set()
        t.join()
        assert seen and set(seen) == {200}
    finally:
        index.close()


# --- thread safety -----------------------------------------------------------

def test_threaded_hammer(idx):
    threads_n, ops = 8, 200
    errors: list[BaseException] = []
    start = threading.Barrier(threads_n)

    def worker(t: int) -> None:
        try:
            start.wait()
            for i in range(ops):
                rid = f"t{t}-r{i}"
                sha = f"sha-{t}-{i}"
                prev = [f"t{t}-r{i - 1}"] if i else []
                idx.record_write(rid, sha, T1, prev)
                assert idx.find_by_sha(sha) == rid
                # shared sha written by every thread: oldest == smallest id on ties
                idx.record_write(f"shared-{t:02d}-{i}", f"shared-{i}", T2, [])
                assert idx.find_by_sha(f"shared-{i}") is not None
        except BaseException as e:  # noqa: BLE001 — surfaced below
            errors.append(e)

    workers = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=120)
    assert not any(w.is_alive() for w in workers)
    assert errors == []
    assert idx.count() == threads_n * ops * 2
    assert len(idx.edges()) == threads_n * (ops - 1)
    assert idx.find_by_sha("shared-7") == "shared-00-7"


def test_two_index_objects_on_one_file_from_threads(tmp_path):
    p = tmp_path / "two.sqlite"
    a, b = TemporalIndex(p), TemporalIndex(p)
    errors: list[BaseException] = []

    def worker(index, tag):
        try:
            for i in range(100):
                index.record_write(f"{tag}-{i}", f"sha-{tag}-{i}", T1, [])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(a, "a")),
          threading.Thread(target=worker, args=(b, "b"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    try:
        assert errors == []
        assert a.count() == 200 and b.count() == 200
    finally:
        a.close()
        b.close()


# --- migration: valid_from/valid_to columns (PACKET F2c) --------------------

def test_migration_adds_valid_from_valid_to_columns(tmp_path):
    p = tmp_path / "old.temporal.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE records (record_id TEXT PRIMARY KEY, sha TEXT, recorded_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE supersedes (target_id TEXT, superseder_id TEXT, recorded_at TEXT,"
        " PRIMARY KEY(target_id, superseder_id))"
    )
    conn.execute(
        "INSERT INTO records VALUES ('OLD', 'sha-old', ?)", (T1,)
    )
    conn.commit()
    conn.close()

    index = TemporalIndex(p)
    try:
        cols = {r[1] for r in index._conn.execute("PRAGMA table_info(records)")}
        assert {"valid_from", "valid_to"} <= cols
        # A pre-existing row survives the migration with NULL new columns.
        assert index.find_by_sha("sha-old") == "OLD"
        row = index._conn.execute(
            "SELECT valid_from, valid_to FROM records WHERE record_id='OLD'"
        ).fetchone()
        assert row == (None, None)
    finally:
        index.close()


def test_migration_is_idempotent_on_reopen(tmp_path):
    p = tmp_path / "reopen.temporal.sqlite"
    a = TemporalIndex(p)
    a.record_write("A", "sha-a", T1, [], valid_from="2026-01-01T00:00:00Z")
    a.close()
    b = TemporalIndex(p)  # migration runs again; must not fail or drop data
    try:
        row = b._conn.execute(
            "SELECT valid_from FROM records WHERE record_id='A'"
        ).fetchone()
        assert row == ("2026-01-01T00:00:00Z",)
    finally:
        b.close()


# --- recent / corrections / oldest_newest / status_counts (PACKET F2) -------

def test_recent_newest_first_and_limit(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, [])
    idx.record_write("C", "sha-c", T3, [])
    assert idx.recent(limit=10) == [("C", T3), ("B", T2), ("A", T1)]
    assert idx.recent(limit=2) == [("C", T3), ("B", T2)]
    assert idx.recent(limit=0) == []


def test_recent_since_is_an_inclusive_lower_bound(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, [])
    idx.record_write("C", "sha-c", T3, [])
    assert idx.recent(limit=10, since=T2) == [("C", T3), ("B", T2)]
    assert idx.recent(limit=10, since=T3) == [("C", T3)]
    assert idx.recent(limit=10, since="2099-01-01T00:00:00Z") == []


def test_recent_mixed_precision_sorts_chronologically(idx):
    idx.record_write("frac", "sha-f", "2026-09-20T10:00:00.500Z", [])
    idx.record_write("whole", "sha-w", "2026-09-20T10:00:01Z", [])
    assert idx.recent(limit=10) == [
        ("whole", "2026-09-20T10:00:01Z"), ("frac", "2026-09-20T10:00:00.500Z"),
    ]


def test_recent_empty_index(idx):
    assert idx.recent(limit=10) == []


def test_corrections_only_records_that_supersede(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, ["A"])
    idx.record_write("C", "sha-c", T3, [])
    assert idx.corrections(limit=10) == [("B", T2, ["A"])]


def test_corrections_groups_multiple_targets_under_one_superseder(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T1, [])
    idx.record_write("C", "sha-c", T2, ["A", "B"])
    assert idx.corrections(limit=10) == [("C", T2, ["A", "B"])]


def test_corrections_newest_first_and_since_lower_bound(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, ["A"])
    idx.record_write("C", "sha-c", T3, ["B"])
    assert idx.corrections(limit=10) == [("C", T3, ["B"]), ("B", T2, ["A"])]
    assert idx.corrections(limit=10, since=T2) == [("C", T3, ["B"]), ("B", T2, ["A"])]
    assert idx.corrections(limit=10, since=T3) == [("C", T3, ["B"])]
    assert idx.corrections(limit=1) == [("C", T3, ["B"])]


def test_oldest_newest_empty(idx):
    assert idx.oldest_newest() == (None, None)


def test_oldest_newest_mixed_precision(idx):
    idx.record_write("mid", "sha-m", T2, [])
    idx.record_write("frac", "sha-f", "2026-09-20T10:00:00.100Z", [])
    idx.record_write("late", "sha-l", T3, [])
    assert idx.oldest_newest() == ("2026-09-20T10:00:00.100Z", T3)


def test_status_counts_current_only(idx):
    idx.record_write("A", "sha-a", T1, [])
    idx.record_write("B", "sha-b", T2, [])
    assert idx.status_counts(now=T3) == {
        "current": 2, "superseded": 0, "expired": 0, "not_yet_valid": 0,
    }


def test_status_counts_superseded_takes_precedence(idx):
    idx.record_write("A", "sha-a", T1, [], valid_to="2099-01-01T00:00:00Z")
    idx.record_write("B", "sha-b", T2, ["A"])
    counts = idx.status_counts(now=T3)
    assert counts["superseded"] == 1
    assert counts["current"] == 1


def test_status_counts_expired_and_not_yet_valid(idx):
    idx.record_write("expired", "sha-e", T1, [], valid_to="2020-01-01T00:00:00Z")
    idx.record_write("future", "sha-f", T1, [], valid_from="2099-01-01T00:00:00Z")
    idx.record_write("current", "sha-c", T1, [])
    counts = idx.status_counts(now="2026-09-21T00:00:00Z")
    assert counts == {"current": 1, "superseded": 0, "expired": 1, "not_yet_valid": 1}


def test_status_counts_empty_index(idx):
    assert idx.status_counts(now=T1) == {
        "current": 0, "superseded": 0, "expired": 0, "not_yet_valid": 0,
    }


# --- open_index fail-open ----------------------------------------------------

def test_open_index_happy_path(tmp_path):
    store = tmp_path / "chroma_db"
    index = open_index(store)
    try:
        assert isinstance(index, TemporalIndex)
        assert index.path == sidecar_path(store)
        assert sidecar_path(store).exists()
        assert not store.exists()  # never creates or touches the store itself
    finally:
        index.close()


def test_open_index_parent_is_a_regular_file(tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    with caplog.at_level(logging.WARNING, logger="fidelis.temporal_index"):
        assert open_index(blocker / "chroma_db") is None
        assert open_index(blocker / "chroma_db") is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert blocker.read_text() == "i am a file, not a directory"


def test_open_index_garbage_sidecar(tmp_path, caplog):
    store = tmp_path / "chroma_db"
    garbage = b"this is definitely not a sqlite database\x00\xff" * 64
    sidecar_path(store).write_bytes(garbage)
    with caplog.at_level(logging.WARNING, logger="fidelis.temporal_index"):
        assert open_index(store) is None
        assert open_index(store) is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    # fail-open means hands-off: the broken file is left for an operator
    assert sidecar_path(store).read_bytes() == garbage


def test_open_index_bad_argument_returns_none():
    assert open_index(None) is None  # type: ignore[arg-type]


def test_direct_constructor_raises_on_garbage(tmp_path):
    p = tmp_path / "bad.sqlite"
    p.write_bytes(b"garbage" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        TemporalIndex(p)
