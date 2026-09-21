"""Unit tests for fidelis.temporal against docs/TIME-AWARE-SPEC.md (contract v1)."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from fidelis import temporal
from fidelis.temporal import (
    SCHEMA,
    SupersessionIndex,
    apply_temporal,
    build_temporal_fields,
    content_sha256,
    format_instant,
    parse_instant,
    temporal_status,
    visible_as_of,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def _dt(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


# --------------------------------------------------------------------------
# parse_instant / format_instant / content_sha256
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-03-04", _dt(2026, 3, 4)),
        ("2026-03-04T05:06:07Z", _dt(2026, 3, 4, 5, 6, 7)),
        ("2026-03-04T05:06:07z", _dt(2026, 3, 4, 5, 6, 7)),
        ("2026-03-04T05:06:07+00:00", _dt(2026, 3, 4, 5, 6, 7)),
        ("2026-03-04T05:06:07+02:00", _dt(2026, 3, 4, 3, 6, 7)),
        ("2026-03-04T05:06:07-0530", _dt(2026, 3, 4, 10, 36, 7)),
        ("2026-03-04T05:06:07", _dt(2026, 3, 4, 5, 6, 7)),
        ("2026-03-04 05:06:07", _dt(2026, 3, 4, 5, 6, 7)),
        ("2026-03-04T05:06Z", _dt(2026, 3, 4, 5, 6)),
        ("2026-03-04T05:06:07.5Z", _dt(2026, 3, 4, 5, 6, 7, 500000)),
        ("2026-03-04T05:06:07.123456Z", _dt(2026, 3, 4, 5, 6, 7, 123456)),
        ("  2026-03-04  ", _dt(2026, 3, 4)),
        (datetime(2026, 3, 4, 5, 6, 7), _dt(2026, 3, 4, 5, 6, 7)),
        (
            datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone(timedelta(hours=2))),
            _dt(2026, 3, 4, 3, 6, 7),
        ),
    ],
)
def test_parse_instant_accepts(value, expected):
    got = parse_instant(value)
    assert got == expected
    assert got.tzinfo is not None
    assert got.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "value",
    [
        None, "", "   ", 0, 1726833600, 1.5, True, b"2026-03-04", [], {},
        "last week", "yesterday", "now", "garbage",
        "2026-13-01", "2026-02-30", "2026-3-4", "20260304", "2026-W10-3",
        "2026-03-04T25:00:00Z", "2026-03-04T05:06:07+99:00",
        "2026-03-04T05:06:07Zjunk", "March 4 2026", "2026/03/04",
    ],
)
def test_parse_instant_rejects(value):
    with pytest.raises(ValueError):
        parse_instant(value)


def test_format_instant_is_utc_z_and_round_trips():
    assert format_instant(_dt(2026, 3, 4, 5, 6, 7)) == "2026-03-04T05:06:07Z"
    offset = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone(timedelta(hours=2)))
    assert format_instant(offset) == "2026-03-04T03:06:07Z"
    assert format_instant(datetime(2026, 3, 4)) == "2026-03-04T00:00:00Z"
    fine = _dt(2026, 3, 4, 5, 6, 7, 120000)
    assert format_instant(fine) == "2026-03-04T05:06:07.120000Z"
    assert parse_instant(format_instant(fine)) == fine


def test_content_sha256_hashes_stripped_utf8():
    expected = hashlib.sha256("héllo wörld".encode("utf-8")).hexdigest()
    assert content_sha256("  héllo wörld\n") == expected
    assert content_sha256("héllo wörld") == expected
    assert content_sha256("a") != content_sha256("b")


# --------------------------------------------------------------------------
# build_temporal_fields
# --------------------------------------------------------------------------

def test_build_minimal_fields_only_system_keys():
    fields = build_temporal_fields("  some text ", now=NOW)
    assert fields == {
        "temporal_schema": SCHEMA,
        "recorded_at": "2026-09-20T12:00:00Z",
        "recorded_at_source": "write",
        "content_sha256": content_sha256("some text"),
    }
    assert SCHEMA == "fidelis.temporal/v1"


def test_build_full_declaration_is_scalar_and_normalised():
    declared = {
        "event_at": "2026-01-01",
        "valid_from": "2026-01-01T00:00:00+02:00",
        "valid_to": datetime(2026, 6, 1, tzinfo=UTC),
        "supersedes": ["a", "b", "a", "c", "b"],
        "source": "ops runbook",
    }
    before = copy.deepcopy(declared)
    fields = build_temporal_fields("t", now=NOW, declared=declared)
    assert declared == before
    assert fields["event_at"] == "2026-01-01T00:00:00Z"
    assert fields["valid_from"] == "2025-12-31T22:00:00Z"
    assert fields["valid_to"] == "2026-06-01T00:00:00Z"
    assert fields["supersedes_json"] == '["a","b","c"]'
    assert fields["source"] == "ops runbook"
    assert all(isinstance(v, str) for v in fields.values())


def test_build_supersedes_accepts_single_string():
    fields = build_temporal_fields("t", now=NOW, declared={"supersedes": "rec-1"})
    assert json.loads(fields["supersedes_json"]) == ["rec-1"]
    assert " " not in fields["supersedes_json"]


def test_build_undeclared_optionals_are_absent_not_guessed():
    # A date inside the text is never promoted to event_at / valid_*.
    fields = build_temporal_fields(
        "On 2026-01-05 the port changed.", now=NOW,
        declared={"event_at": None, "supersedes": None, "supersedes_x": []},
    )
    for key in ("event_at", "valid_from", "valid_to", "supersedes_json", "source"):
        assert key not in fields


def test_build_ignores_unknown_keys_and_caller_recorded_at():
    fields = build_temporal_fields(
        "t", now=NOW,
        declared={
            "relation_declarations_v2": [{"x": 1}],
            "recorded_at": "1999-01-01",
            "temporal_schema": "evil/v9",
            "content_sha256": "0" * 64,
            "whatever": object(),
        },
    )
    assert set(fields) == {
        "temporal_schema", "recorded_at", "recorded_at_source", "content_sha256",
    }
    assert fields["recorded_at"] == "2026-09-20T12:00:00Z"
    assert fields["temporal_schema"] == SCHEMA
    assert fields["content_sha256"] == content_sha256("t")


def test_build_replay_path_keeps_original_queue_time():
    fields = build_temporal_fields(
        "t", now=NOW, recorded_at="2026-09-19T08:30:00Z",
        recorded_at_source="queue",
    )
    assert fields["recorded_at"] == "2026-09-19T08:30:00Z"
    assert fields["recorded_at_source"] == "queue"
    replay = build_temporal_fields("t", now=NOW, recorded_at_source="replay")
    assert replay["recorded_at"] == "2026-09-20T12:00:00Z"
    assert replay["recorded_at_source"] == "replay"


@pytest.mark.parametrize(
    "declared, fragment",
    [
        ({"event_at": "last week"}, "event_at"),
        ({"valid_from": ""}, "valid_from"),
        ({"valid_to": 1726833600}, "valid_to"),
        ({"valid_from": "2026-02-01", "valid_to": "2026-01-01"}, "valid_from"),
        ({"valid_from": "2026-01-01", "valid_to": "2026-01-01"}, "valid_from"),
        ({"supersedes": 7}, "supersedes"),
        ({"supersedes": {"a": 1}}, "supersedes"),
        ({"supersedes": ["ok", ""]}, "supersedes"),
        ({"supersedes": ["ok", "   "]}, "supersedes"),
        ({"supersedes": ["ok", 3]}, "supersedes"),
        ({"supersedes": ""}, "supersedes"),
        ({"supersedes": [f"id-{i}" for i in range(65)]}, "64"),
        ({"source": 42}, "source"),
        ({"source": ["a"]}, "source"),
        ({"source": "x" * 513}, "512"),
    ],
)
def test_build_rejects_invalid_declarations(declared, fragment):
    with pytest.raises(ValueError) as excinfo:
        build_temporal_fields("t", now=NOW, declared=declared)
    assert fragment in str(excinfo.value)


def test_build_limits_are_inclusive():
    ids = [f"id-{i}" for i in range(64)]
    fields = build_temporal_fields(
        "t", now=NOW, declared={"supersedes": ids, "source": "x" * 512}
    )
    assert json.loads(fields["supersedes_json"]) == ids
    assert len(fields["source"]) == 512


def test_build_rejects_self_supersession():
    with pytest.raises(ValueError, match="supersede itself"):
        build_temporal_fields(
            "t", now=NOW, declared={"supersedes": ["a", "me"]}, record_id="me"
        )
    ok = build_temporal_fields(
        "t", now=NOW, declared={"supersedes": ["a"]}, record_id="me"
    )
    assert ok["supersedes_json"] == '["a"]'


def test_build_rejects_bad_system_arguments():
    with pytest.raises(ValueError):
        build_temporal_fields("t", now=NOW, recorded_at_source="caller")
    with pytest.raises(ValueError):
        build_temporal_fields("t", now=NOW, recorded_at="not a time")
    with pytest.raises(ValueError):
        build_temporal_fields("t", now=None)
    with pytest.raises(ValueError):
        build_temporal_fields("t", now=NOW, declared=["event_at"])
    with pytest.raises(ValueError):
        build_temporal_fields(None, now=NOW)


# --------------------------------------------------------------------------
# SupersessionIndex
# --------------------------------------------------------------------------

def test_index_superseders_and_as_of_cutoff():
    index = SupersessionIndex()
    index.add("A", "C", "2026-03-01T00:00:00Z")
    index.add("A", "B", "2026-02-01T00:00:00Z")
    assert index.superseders("A") == ["B", "C"]
    assert index.superseders("A", as_of="2026-02-15") == ["B"]
    assert index.superseders("A", as_of="2026-02-01T00:00:00Z") == ["B"]
    assert index.superseders("A", as_of="2026-01-31T23:59:59Z") == []
    assert index.superseders("missing") == []
    assert index.superseders(None) == []


def test_index_is_idempotent_and_ignores_self_edges():
    index = SupersessionIndex.from_edges(
        [
            ("A", "B", "2026-02-01T00:00:00Z"),
            ("A", "B", "2026-02-01T00:00:00Z"),
            ("A", "A", "2026-02-01T00:00:00Z"),
            ("bad-row",),
            None,
        ]
    )
    assert index.superseders("A") == ["B"]
    assert len(index) == 1


def test_index_unknown_superseder_time_never_assumed_inside_as_of():
    index = SupersessionIndex.from_edges([("A", "B", "corrupt"), ("A", "C", None)])
    assert index.superseders("A") == ["B", "C"]
    assert index.superseders("A", as_of="2030-01-01") == []


# --------------------------------------------------------------------------
# temporal_status
# --------------------------------------------------------------------------

EMPTY = SupersessionIndex()


def test_status_shape_for_plain_current_record():
    payload = build_temporal_fields("t", now=_dt(2026, 1, 1))
    status = temporal_status(payload, "A", index=EMPTY, now=NOW)
    assert status == {
        "status": "current",
        "recorded_at": "2026-01-01T00:00:00Z",
        "recorded_at_known": True,
        "superseded_by": [],
        "valid_from": None,
        "valid_to": None,
        "event_at": None,
        "evaluated_at": "2026-09-20T12:00:00Z",
        "index": "ok",
    }


def test_status_half_open_validity_boundaries():
    payload = {
        "recorded_at": "2026-01-01T00:00:00Z",
        "valid_from": "2026-02-01T00:00:00Z",
        "valid_to": "2026-03-01T00:00:00Z",
    }

    def at(instant):
        return temporal_status(payload, "A", index=EMPTY, now=instant)["status"]

    assert at("2026-01-31T23:59:59Z") == "not_yet_valid"
    assert at("2026-02-01T00:00:00Z") == "current"        # T == valid_from
    assert at("2026-02-28T23:59:59Z") == "current"
    assert at("2026-03-01T00:00:00Z") == "expired"        # T == valid_to
    assert at("2026-04-01T00:00:00Z") == "expired"


def test_status_precedence():
    index = SupersessionIndex.from_edges([("A", "B", "2026-01-02T00:00:00Z")])
    expired = {"valid_to": "2026-02-01T00:00:00Z"}
    future = {"valid_from": "2027-01-01T00:00:00Z"}
    # A window written past the write gate: both expired and not_yet_valid.
    both = {"valid_from": "2027-01-01T00:00:00Z", "valid_to": "2026-02-01T00:00:00Z"}

    def status(payload, idx):
        return temporal_status(payload, "A", index=idx, now=NOW)["status"]

    assert status(expired, index) == "superseded"
    assert status(future, index) == "superseded"
    assert status(both, index) == "superseded"
    assert status(both, EMPTY) == "expired"
    assert status(future, EMPTY) == "not_yet_valid"
    assert status({}, EMPTY) == "current"
    out = temporal_status(expired, "A", index=index, now=NOW)
    assert out["superseded_by"] == ["B"]
    assert out["valid_to"] == "2026-02-01T00:00:00Z"


def test_status_as_of_ignores_later_superseder():
    index = SupersessionIndex.from_edges([("A", "B", "2026-06-01T00:00:00Z")])
    payload = {"recorded_at": "2026-01-01T00:00:00Z"}
    before = temporal_status(
        payload, "A", index=index, now=NOW, as_of="2026-03-01T00:00:00Z"
    )
    assert before["status"] == "current"
    assert before["superseded_by"] == []
    assert before["evaluated_at"] == "2026-03-01T00:00:00Z"
    after = temporal_status(payload, "A", index=index, now=NOW, as_of="2026-07-01")
    assert after["status"] == "superseded"
    assert after["superseded_by"] == ["B"]
    today = temporal_status(payload, "A", index=index, now=NOW)
    assert today["status"] == "superseded"
    assert today["evaluated_at"] == "2026-09-20T12:00:00Z"


def test_status_as_of_evaluates_validity_at_as_of():
    payload = {"valid_to": "2026-05-01T00:00:00Z"}
    assert temporal_status(payload, "A", index=EMPTY, now=NOW)["status"] == "expired"
    then = temporal_status(payload, "A", index=EMPTY, now=NOW, as_of="2026-04-01")
    assert then["status"] == "current"


def test_status_index_none_is_unavailable_not_an_error():
    payload = {"recorded_at": "2026-01-01T00:00:00Z", "valid_to": "2026-02-01"}
    out = temporal_status(payload, "A", index=None, now=NOW)
    assert out["index"] == "unavailable"
    assert out["superseded_by"] == []
    assert out["status"] == "expired"           # validity still evaluated
    assert temporal_status({}, "A", index=None, now=NOW)["status"] == "current"


def test_status_broken_index_fails_open():
    class Broken:
        def superseders(self, target_id, *, as_of=None):
            raise RuntimeError("sidecar gone")

    out = temporal_status({}, "A", index=Broken(), now=NOW)
    assert out["index"] == "unavailable"
    assert out["status"] == "current"


def test_status_legacy_and_corrupt_payloads_tolerated():
    legacy = {"data": "old row", "user_id": "agent"}
    out = temporal_status(legacy, "L", index=EMPTY, now=NOW)
    assert out["status"] == "current"
    assert out["recorded_at"] is None
    assert out["recorded_at_known"] is False

    corrupt = {"recorded_at": "not-a-date", "valid_to": 12345, "event_at": ""}
    out = temporal_status(corrupt, "X", index=EMPTY, now=NOW)
    assert out["recorded_at_known"] is False
    assert out["recorded_at"] is None
    assert out["valid_to"] is None and out["event_at"] is None
    assert out["status"] == "current"

    assert temporal_status(None, None, index=EMPTY, now=NOW)["status"] == "current"


def test_status_invalid_as_of_raises():
    with pytest.raises(ValueError):
        temporal_status({}, "A", index=EMPTY, now=NOW, as_of="last week")


# --------------------------------------------------------------------------
# visible_as_of
# --------------------------------------------------------------------------

def test_visible_as_of():
    payload = {"recorded_at": "2026-03-01T00:00:00Z"}
    assert visible_as_of(payload, "2026-03-01T00:00:00Z") is True
    assert visible_as_of(payload, "2026-04-01") is True
    assert visible_as_of(payload, "2026-02-28T23:59:59Z") is False
    assert visible_as_of({}, "2000-01-01") is True
    assert visible_as_of({"recorded_at": "garbage"}, "2000-01-01") is True
    assert visible_as_of(None, "2000-01-01") is True
    with pytest.raises(ValueError):
        visible_as_of(payload, "whenever")


# --------------------------------------------------------------------------
# apply_temporal
# --------------------------------------------------------------------------

def _mem(rid, text, **payload):
    return {"id": rid, "text": text, "score": 0.5, "payload": payload}


def _ids(memories):
    return [m["id"] for m in memories]


def _scenario():
    """Relevance order: A(superseded) B C(expired) D E(not yet valid) F."""
    index = SupersessionIndex.from_edges([("A", "B", "2026-02-01T00:00:00Z")])
    memories = [
        _mem("A", "port 19420", recorded_at="2026-01-01T00:00:00Z"),
        _mem("B", "port 19555", recorded_at="2026-02-01T00:00:00Z",
             supersedes_json='["A"]'),
        _mem("C", "temp note", recorded_at="2026-01-05T00:00:00Z",
             valid_to="2026-01-10T00:00:00Z"),
        _mem("D", "plain", recorded_at="2026-01-06T00:00:00Z"),
        _mem("E", "future", recorded_at="2026-01-07T00:00:00Z",
             valid_from="2027-01-01T00:00:00Z"),
        _mem("F", "plain two", recorded_at="2026-01-08T00:00:00Z"),
    ]
    return index, memories


def test_apply_stable_partition_preserves_relevance_order():
    index, memories = _scenario()
    out = apply_temporal(memories, index=index, now=NOW)
    assert _ids(out) == ["B", "D", "F", "A", "C", "E"]
    by_id = {m["id"]: m["temporal"] for m in out}
    assert by_id["A"]["status"] == "superseded"
    assert by_id["A"]["superseded_by"] == ["B"]
    assert by_id["B"]["status"] == "current"
    assert by_id["C"]["status"] == "expired"
    assert by_id["E"]["status"] == "not_yet_valid"
    assert len(out) == len(memories)             # never silently drop


def test_apply_historical_leaves_order_untouched_but_annotates():
    index, memories = _scenario()
    out = apply_temporal(memories, index=index, now=NOW, historical=True)
    assert _ids(out) == ["A", "B", "C", "D", "E", "F"]
    assert out[0]["temporal"]["status"] == "superseded"


def test_apply_does_not_mutate_inputs_and_text_is_byte_identical():
    index, memories = _scenario()
    memories[0]["text"] = "  Fidelis listens on pört 19420.\r\n\t"
    before = copy.deepcopy(memories)
    out = apply_temporal(memories, index=index, now=NOW, as_of="2026-06-01")
    assert memories == before
    assert all("temporal" not in m for m in memories)
    originals = {m["id"]: m for m in memories}
    for m in out:
        assert m is not originals[m["id"]]
        assert m["text"] == originals[m["id"]]["text"]
        assert m["text"].encode("utf-8") == originals[m["id"]]["text"].encode("utf-8")
        assert m["score"] == 0.5
        assert m["payload"] == originals[m["id"]]["payload"]


def test_apply_as_of_excludes_later_records_and_their_supersession():
    index, memories = _scenario()
    out = apply_temporal(memories, index=index, now=NOW, as_of="2026-01-15T00:00:00Z")
    assert "B" not in _ids(out)                  # recorded after as_of
    by_id = {m["id"]: m["temporal"] for m in out}
    assert by_id["A"]["status"] == "current"     # superseder did not exist yet
    assert by_id["A"]["superseded_by"] == []
    assert by_id["C"]["status"] == "expired"
    assert _ids(out) == ["A", "D", "F", "C", "E"]
    assert all(t["evaluated_at"] == "2026-01-15T00:00:00Z" for t in by_id.values())


def test_apply_without_as_of_never_excludes():
    index, memories = _scenario()
    future = _mem("Z", "from the future", recorded_at="2030-01-01T00:00:00Z")
    out = apply_temporal([future] + memories, index=index, now=NOW)
    assert "Z" in _ids(out) and len(out) == 7


def test_apply_legacy_record_visible_flagged_and_sorted_after_known_under_as_of():
    legacy = {"id": "L", "text": "legacy", "payload": {"data": "legacy", "user_id": "u"}}
    corrupt = _mem("X", "corrupt", recorded_at="????")
    known = _mem("K", "known", recorded_at="2026-01-01T00:00:00Z")
    gone = _mem("G", "expired", recorded_at="2026-01-01T00:00:00Z", valid_to="2026-01-02")
    memories = [legacy, gone, corrupt, known]

    out = apply_temporal(memories, index=EMPTY, now=NOW, as_of="2026-06-01")
    assert _ids(out) == ["K", "L", "X", "G"]
    flags = {m["id"]: m["temporal"]["recorded_at_known"] for m in out}
    assert flags == {"K": True, "L": False, "X": False, "G": True}
    assert out[1]["temporal"]["status"] == "current"

    # Without as_of, unknown recorded_at does not affect ordering.
    plain = apply_temporal(memories, index=EMPTY, now=NOW)
    assert _ids(plain) == ["L", "X", "K", "G"]


def test_apply_index_none_passes_hits_through_annotated():
    _, memories = _scenario()
    out = apply_temporal(memories, index=None, now=NOW)
    assert all(m["temporal"]["index"] == "unavailable" for m in out)
    assert {m["id"]: m["temporal"]["status"] for m in out}["A"] == "current"
    assert _ids(out) == ["A", "B", "D", "F", "C", "E"]


def test_apply_reads_fields_from_metadata_or_top_level():
    expired = {"recorded_at": "2026-01-01T00:00:00Z", "valid_to": "2026-01-02T00:00:00Z"}
    memories = [
        {"id": "M", "text": "via metadata", "metadata": dict(expired)},
        {"id": "T", "text": "via top level", **expired},
        {"id": "P", "text": "via payload", "payload": dict(expired)},
        {"id": "E", "text": "payload lacks fields", "payload": {"data": "x"},
         "metadata": dict(expired)},
        {"text": "no id, no fields"},
    ]
    out = apply_temporal(memories, index=EMPTY, now=NOW)
    assert [m.get("id") for m in out] == [None, "M", "T", "P", "E"]
    for m in out[1:]:
        assert m["temporal"]["status"] == "expired"
        assert m["temporal"]["valid_to"] == "2026-01-02T00:00:00Z"
    assert out[0]["temporal"]["status"] == "current"
    assert out[0]["temporal"]["recorded_at_known"] is False


def test_apply_empty_and_invalid_as_of():
    assert apply_temporal([], index=EMPTY, now=NOW) == []
    with pytest.raises(ValueError):
        apply_temporal([], index=EMPTY, now=NOW, as_of="tomorrow-ish")


def test_module_has_no_io_imports():
    for name in ("os", "sqlite3", "socket", "urllib", "pathlib", "subprocess"):
        assert not hasattr(temporal, name)
