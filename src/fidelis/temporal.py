"""Time-aware memory: pure temporal logic (contract: docs/TIME-AWARE-SPEC.md).

The store is append-only and verbatim, so time is never written INTO a record's
text and never used to rewrite or delete one. Instead every verbatim insert is
stamped with a few scalar payload fields (:func:`build_temporal_fields`) and
currency is a LOOKUP evaluated at read time (:func:`temporal_status`,
:func:`apply_temporal`).

Design rules this module enforces:

* Never invent a date, relation, or status. A field the caller did not declare
  and the system did not observe is absent. Legacy rows with no ``recorded_at``
  are reported as ``recorded_at_known: False`` rather than guessed.
* Relevance establishes eligibility before recency affects ordering. The only
  exclusion is ``as_of`` visibility, and only when the caller asked for it;
  everything else is annotate-and-demote inside the incoming hit set. There is
  no decay or half-life formula.
* Fail-open reads, fail-loud writes. Write-time validation raises
  ``ValueError`` with a precise message; read-time evaluation of a STORED
  payload tolerates missing/corrupt values and a missing supersession index
  (``"index": "unavailable"``).

No I/O, no network, standard library only. The sqlite sidecar that feeds
:class:`SupersessionIndex` lives in ``temporal_index.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any


SCHEMA = "fidelis.temporal/v1"

MAX_SUPERSEDES = 64
MAX_SOURCE_CHARS = 512
RECORDED_AT_SOURCES = ("write", "queue", "replay")

_INSTANT_FIELDS = ("event_at", "valid_from", "valid_to")
# Keys whose presence marks a mapping as "the one carrying temporal fields".
# ``source`` is deliberately absent: hits use that name for other things.
_TEMPORAL_KEYS = frozenset(
    ("temporal_schema", "recorded_at", "recorded_at_source", "content_sha256",
     "supersedes_json") + _INSTANT_FIELDS
)

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Explicit shape gate so acceptance does not drift with the interpreter:
# 3.11+ ``fromisoformat`` takes basic/week formats that 3.10 rejects.
_DATETIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[Tt ]"
    r"(?P<hm>\d{2}:\d{2})(?::(?P<sec>\d{2})(?:\.(?P<frac>\d{1,6}))?)?"
    r"(?P<tz>[Zz]|[+-]\d{2}:?\d{2})?$"
)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"instant out of range: {dt!r}") from exc


def parse_instant(value: Any) -> datetime:
    """Parse a declared instant into a tz-aware UTC ``datetime``.

    Accepts a ``datetime``, ``"YYYY-MM-DD"`` (midnight UTC), or full ISO-8601
    with ``Z`` or a numeric offset. Naive values are assumed UTC. Anything
    else — ``""``, ``None``, numbers, ``"last week"`` — raises ``ValueError``;
    relative or fuzzy dates are never resolved on the caller's behalf.
    """
    if isinstance(value, datetime):
        return _to_utc(value)
    if not isinstance(value, str):
        raise ValueError(
            f"invalid instant {value!r}: expected ISO-8601 string or datetime"
        )
    raw = value.strip()
    if not raw:
        raise ValueError("invalid instant '': empty string")
    try:
        if _DATE_ONLY_RE.match(raw):
            return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        match = _DATETIME_RE.match(raw)
        if match is None:
            raise ValueError("not ISO-8601")
        # Rebuild in the one shape every supported interpreter parses.
        frac = (match.group("frac") or "").ljust(6, "0")
        tz = match.group("tz") or ""
        if tz in ("Z", "z"):
            tz = "+00:00"
        elif tz and ":" not in tz:
            tz = f"{tz[:3]}:{tz[3:]}"
        normalised = (
            f"{match.group('date')}T{match.group('hm')}:"
            f"{match.group('sec') or '00'}.{frac}{tz}"
        )
        return _to_utc(datetime.fromisoformat(normalised))
    except ValueError as exc:
        raise ValueError(
            f"invalid instant {value!r}: expected YYYY-MM-DD or ISO-8601 "
            "with Z/offset"
        ) from exc


def format_instant(dt: Any) -> str:
    """Canonical wire form: UTC, ``Z`` suffix, second precision or finer."""
    instant = parse_instant(dt)
    spec = "microseconds" if instant.microsecond else "seconds"
    return instant.replace(tzinfo=None).isoformat(timespec=spec) + "Z"


def content_sha256(text: str) -> str:
    """sha256 hex of ``text.strip()`` as UTF-8 (the dedup key)."""
    if not isinstance(text, str):
        raise ValueError("content_sha256 requires str text")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _normalise_supersedes(value: Any, record_id: Any) -> list[str]:
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(
            "supersedes must be a record-id string or a list of record-id "
            f"strings, got {type(value).__name__}"
        )
    if len(items) > MAX_SUPERSEDES:
        raise ValueError(
            f"supersedes lists {len(items)} ids; maximum is {MAX_SUPERSEDES}"
        )
    ids: list[str] = []
    for position, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"supersedes[{position}] must be a non-empty string, "
                f"got {item!r}"
            )
        item = item.strip()
        if item not in ids:
            ids.append(item)
    if record_id is not None and str(record_id) in ids:
        raise ValueError(f"record {str(record_id)!r} cannot supersede itself")
    return ids


def build_temporal_fields(
    text: str,
    *,
    now: Any,
    declared: Mapping[str, Any] | None = None,
    recorded_at_source: str = "write",
    recorded_at: Any = None,
    record_id: Any = None,
) -> dict[str, str]:
    """Build the Chroma-safe scalar payload fields for one verbatim insert.

    ``recorded_at`` comes from the system clock (``now``); callers cannot set
    it through ``declared``. The ``recorded_at`` parameter exists only for the
    queue-replay path, which must keep the ORIGINAL queue time. Optional fields
    appear in the result only when declared (``None`` counts as undeclared).
    Unknown ``declared`` keys are ignored — other features own them. ``text``
    is hashed, never altered or mined for dates.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    if declared is None:
        declared = {}
    elif not isinstance(declared, Mapping):
        raise ValueError("declared temporal fields must be a mapping")
    if recorded_at_source not in RECORDED_AT_SOURCES:
        raise ValueError(
            f"recorded_at_source must be one of {RECORDED_AT_SOURCES}, "
            f"got {recorded_at_source!r}"
        )

    stamp = parse_instant(now if recorded_at is None else recorded_at)
    fields: dict[str, str] = {
        "temporal_schema": SCHEMA,
        "recorded_at": format_instant(stamp),
        "recorded_at_source": recorded_at_source,
        "content_sha256": content_sha256(text),
    }

    instants: dict[str, datetime] = {}
    for name in _INSTANT_FIELDS:
        value = declared.get(name)
        if value is None:
            continue
        try:
            instants[name] = parse_instant(value)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        fields[name] = format_instant(instants[name])
    if (
        "valid_from" in instants
        and "valid_to" in instants
        and instants["valid_from"] >= instants["valid_to"]
    ):
        raise ValueError(
            f"valid_from ({fields['valid_from']}) must be earlier than "
            f"valid_to ({fields['valid_to']}); the window is half-open"
        )

    supersedes = declared.get("supersedes")
    if supersedes is not None:
        ids = _normalise_supersedes(supersedes, record_id)
        if ids:
            fields["supersedes_json"] = json.dumps(ids, separators=(",", ":"))

    source = declared.get("source")
    if source is not None:
        if not isinstance(source, str):
            raise ValueError(
                f"source must be a string, got {type(source).__name__}"
            )
        if len(source) > MAX_SOURCE_CHARS:
            raise ValueError(
                f"source is {len(source)} chars; maximum is {MAX_SOURCE_CHARS}"
            )
        if source:
            fields["source"] = source
    return fields


def _stored_instant(payload: Mapping[str, Any], name: str) -> datetime | None:
    """Read-time parse of a STORED field: corrupt or absent means unknown."""
    try:
        return parse_instant(payload.get(name))
    except ValueError:
        return None


class SupersessionIndex:
    """In-memory view: target_id -> [(superseder_id, superseder_recorded_at)].

    Built from the sidecar (or any edge iterable) once per read; supersession
    is then a dictionary lookup. A superseder only counts at reference instant
    ``as_of`` if it had been recorded by then, which is what lets an as-of
    query see the world before a later correction existed.
    """

    def __init__(self) -> None:
        self._edges: dict[str, list[tuple[str, datetime | None]]] = {}

    def add(self, target_id: Any, superseder_id: Any, recorded_at: Any) -> None:
        if target_id is None or superseder_id is None:
            return
        target, by = str(target_id), str(superseder_id)
        if not target or not by or target == by:
            return
        try:
            when: datetime | None = parse_instant(recorded_at)
        except ValueError:
            when = None
        edges = self._edges.setdefault(target, [])
        for position, (existing, existing_when) in enumerate(edges):
            if existing == by:
                # Idempotent rebuilds; a known time beats an unknown one.
                if existing_when is None and when is not None:
                    edges[position] = (by, when)
                return
        edges.append((by, when))

    def superseders(self, target_id: Any, *, as_of: Any = None) -> list[str]:
        """Superseder ids, oldest first (unknown times last, insertion-stable).

        With ``as_of``, only superseders KNOWN to be recorded at or before it;
        an unknown time is never assumed to fall inside the window.
        """
        if target_id is None:
            return []
        edges = self._edges.get(str(target_id), [])
        if as_of is not None:
            limit = parse_instant(as_of)
            edges = [e for e in edges if e[1] is not None and e[1] <= limit]
        ordered = sorted(
            edges, key=lambda e: (e[1] is None, e[1] or datetime.min.replace(
                tzinfo=timezone.utc))
        )
        return [by for by, _ in ordered]

    def __len__(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    @classmethod
    def from_edges(cls, edges: Iterable[Any]) -> "SupersessionIndex":
        """Build from ``(target, by, recorded_at_str)`` rows; bad rows skipped."""
        index = cls()
        for edge in edges or ():
            try:
                target, by, recorded_at = edge
            except (TypeError, ValueError):
                continue
            index.add(target, by, recorded_at)
        return index


def temporal_status(
    payload: Mapping[str, Any] | None,
    record_id: Any,
    *,
    index: SupersessionIndex | None,
    now: Any,
    as_of: Any = None,
) -> dict[str, Any]:
    """Evaluate one stored payload at reference instant T = ``as_of`` or ``now``.

    Precedence: superseded > expired > not_yet_valid > current. Validity is
    half-open: valid at ``valid_from``, expired at ``valid_to``. ``index=None``
    (or an index that errors) reports ``"index": "unavailable"`` and leaves
    supersession unevaluated. Only an invalid ``now``/``as_of`` raises.
    """
    if not isinstance(payload, Mapping):
        payload = {}
    reference = parse_instant(now if as_of is None else as_of)

    recorded_at = _stored_instant(payload, "recorded_at")
    valid_from = _stored_instant(payload, "valid_from")
    valid_to = _stored_instant(payload, "valid_to")
    event_at = _stored_instant(payload, "event_at")

    superseded_by: list[str] = []
    index_state = "unavailable"
    if index is not None:
        try:
            if record_id is not None:
                superseded_by = [
                    str(by)
                    for by in index.superseders(
                        str(record_id),
                        as_of=None if as_of is None else reference,
                    )
                ]
            index_state = "ok"
        except Exception:
            superseded_by = []

    if superseded_by:
        status = "superseded"
    elif valid_to is not None and reference >= valid_to:
        status = "expired"
    elif valid_from is not None and reference < valid_from:
        status = "not_yet_valid"
    else:
        status = "current"

    def _out(instant: datetime | None) -> str | None:
        return None if instant is None else format_instant(instant)

    return {
        "status": status,
        "recorded_at": _out(recorded_at),
        "recorded_at_known": recorded_at is not None,
        "superseded_by": superseded_by,
        "valid_from": _out(valid_from),
        "valid_to": _out(valid_to),
        "event_at": _out(event_at),
        "evaluated_at": format_instant(reference),
        "index": index_state,
    }


def visible_as_of(payload: Mapping[str, Any] | None, as_of: Any) -> bool:
    """False only when ``recorded_at`` is KNOWN and later than ``as_of``.

    Legacy rows with unknown ``recorded_at`` stay visible — flagged by
    ``recorded_at_known: False``, never guessed out of the result.
    """
    if as_of is None or not isinstance(payload, Mapping):
        return True
    limit = parse_instant(as_of)
    recorded_at = _stored_instant(payload, "recorded_at")
    return recorded_at is None or recorded_at <= limit


def _temporal_payload(memory: Mapping[str, Any]) -> Mapping[str, Any]:
    """The mapping holding temporal fields: payload, metadata, else top level."""
    nested = [
        memory.get(key) for key in ("payload", "metadata")
        if isinstance(memory.get(key), Mapping)
    ]
    for candidate in nested:
        if _TEMPORAL_KEYS.intersection(candidate):
            return candidate
    if nested and not _TEMPORAL_KEYS.intersection(memory):
        return nested[0]
    return memory


def apply_temporal(
    memories: Iterable[Mapping[str, Any]],
    *,
    index: SupersessionIndex | None,
    now: Any,
    as_of: Any = None,
    historical: bool = False,
) -> list:
    """Annotate (and, unless ``historical``, demote) a relevance-ordered hit list.

    1. With ``as_of``, drop records not yet recorded then — the ONLY exclusion.
    2. Attach ``temporal`` on a shallow COPY; inputs and ``text`` are untouched.
    3. Stable partition: ``current`` first, the rest after, each keeping the
       incoming relevance order. Under ``as_of``, current records whose
       ``recorded_at`` is unknown follow the known ones.
    """
    parse_instant(now)
    if as_of is not None:
        as_of = parse_instant(as_of)

    annotated: list = []
    for memory in memories or ():
        if not isinstance(memory, Mapping):
            annotated.append(memory)  # never silently drop; nothing to annotate
            continue
        payload = _temporal_payload(memory)
        if as_of is not None and not visible_as_of(payload, as_of):
            continue
        copy = dict(memory)
        copy["temporal"] = temporal_status(
            payload, memory.get("id"), index=index, now=now, as_of=as_of
        )
        annotated.append(copy)
    if historical:
        return annotated

    def _rank(memory: Any) -> int:
        if not isinstance(memory, Mapping):
            return 0
        temporal = memory["temporal"]
        if temporal["status"] != "current":
            return 2
        if as_of is not None and not temporal["recorded_at_known"]:
            return 1
        return 0

    return sorted(annotated, key=_rank)  # sorted() is stable
