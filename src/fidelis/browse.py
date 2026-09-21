"""Read-side navigation over the time-aware store (PACKET F2,
docs/TIME-AWARE-SPEC.md).

Backs three read-only HTTP endpoints in ``fidelis/server.py``:

  POST /get     -- one record by id, plus its supersession chain both ways.
  POST /recent  -- newest records by ``recorded_at``, or corrections only.
  GET  /stats   -- cheap, sidecar-backed store-wide counts.

All three are same-user, fail-open when the temporal sidecar is unavailable
(they say so via an ``"index"``/``"sidecar"`` field rather than falling back
to a full vector-store scan), and never invent a value a payload does not
carry (docs/TIME-AWARE-SPEC.md invariant 3).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from fidelis.temporal import format_instant, parse_instant, temporal_status

# Bound on both chain walks in get_record: enough for any realistic
# correction history while keeping a broken/cyclic declaration graph from
# ever looping (each hop only steps to ids not already visited).
MAX_CHAIN_HOPS = 16


def _payload_for_user(memory, record_id: str, user_id: str) -> dict[str, Any] | None:
    """The stored payload for ``record_id`` if it exists and belongs to
    ``user_id``; ``None`` otherwise (unknown or another user's record)."""
    row = memory.vector_store.get(record_id)
    payload = dict(getattr(row, "payload", None) or {})
    if not payload.get("data") or payload.get("user_id") != user_id:
        return None
    return payload


def _declared_supersedes(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("supersedes_json")
    if not isinstance(raw, str) or not raw:
        return []
    try:
        ids = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(ids, list):
        return []
    return [i for i in ids if isinstance(i, str) and i]


def _safe_supersession_index(index):
    """``(SupersessionIndex-or-None, "ok"|"unavailable")``. Never raises."""
    if index is None:
        return None, "unavailable"
    try:
        return index.supersession_index(), "ok"
    except Exception:  # noqa: fail-open -- a broken sidecar must not break a read
        return None, "unavailable"


def _render_chain_entry(payload: Mapping[str, Any], record_id: str, *, supersession, now) -> dict[str, Any]:
    status = temporal_status(payload, record_id, index=supersession, now=now)
    return {
        "id": record_id,
        "text": payload.get("data", ""),
        "status": status["status"],
        "recorded_at": status["recorded_at"],
    }


def _sort_chain(entries: list[dict[str, Any]], *, newest_last: bool) -> list[dict[str, Any]]:
    """Chronological order; entries with an unknown ``recorded_at`` always
    sort last (both directions), since there is nothing to order them by."""
    known = [e for e in entries if e["recorded_at"] is not None]
    unknown = [e for e in entries if e["recorded_at"] is None]
    known.sort(key=lambda e: parse_instant(e["recorded_at"]), reverse=not newest_last)
    return known + unknown


def _walk_backward(
    memory, user_id: str, start_payload: Mapping[str, Any], *, supersession, now, max_hops: int,
) -> list[dict[str, Any]]:
    """Ids the record (transitively) replaced, oldest last.

    Walks each visited record's OWN declared ``supersedes_json`` -- no
    sidecar lookup needed for this direction, since a record's declaration
    of what it replaced lives on its own payload. Cycle-safe (each id
    visited at most once) and bounded to ``max_hops`` BFS levels. An id in
    the chain that no longer exists, or belongs to another user, is skipped
    rather than fabricated.
    """
    frontier = _declared_supersedes(start_payload)
    visited: set[str] = set()
    collected: list[dict[str, Any]] = []
    hops = 0
    while frontier and hops < max_hops:
        hops += 1
        next_frontier: list[str] = []
        for target_id in frontier:
            if target_id in visited:
                continue
            visited.add(target_id)
            payload = _payload_for_user(memory, target_id, user_id)
            if payload is None:
                continue
            collected.append(
                _render_chain_entry(payload, target_id, supersession=supersession, now=now)
            )
            next_frontier.extend(_declared_supersedes(payload))
        frontier = next_frontier
    return _sort_chain(collected, newest_last=False)


def _walk_forward(
    memory, user_id: str, record_id: str, *, supersession, now, max_hops: int,
) -> list[dict[str, Any]]:
    """Ids that (transitively) replaced the record, newest last.

    Walks forward via the supersession index (who declares superseding
    whom); a record with no known superseders returns ``[]``. Cycle-safe
    and bounded the same way as :func:`_walk_backward`.
    """
    if supersession is None:
        return []
    frontier = [record_id]
    visited = {record_id}
    collected: list[dict[str, Any]] = []
    hops = 0
    while frontier and hops < max_hops:
        hops += 1
        next_frontier: list[str] = []
        for current_id in frontier:
            try:
                superseders = supersession.superseders(current_id)
            except Exception:  # noqa: fail-open -- a broken chain walk stops here, not the read
                superseders = []
            for sup_id in superseders:
                if sup_id in visited:
                    continue
                visited.add(sup_id)
                payload = _payload_for_user(memory, sup_id, user_id)
                if payload is None:
                    continue
                collected.append(
                    _render_chain_entry(payload, sup_id, supersession=supersession, now=now)
                )
                next_frontier.append(sup_id)
        frontier = next_frontier
    return _sort_chain(collected, newest_last=True)


def get_record(
    memory, index, record_id: str, user_id: str, *, now: datetime | None = None,
) -> dict[str, Any] | None:
    """One record verbatim, its temporal status, and its supersession chain
    in both directions. ``None`` when the id is unknown or belongs to a
    different user (PACKET F2a is same-user only).
    """
    now = now or datetime.now(timezone.utc)
    payload = _payload_for_user(memory, record_id, user_id)
    if payload is None:
        return None
    supersession, index_state = _safe_supersession_index(index)
    status = temporal_status(payload, record_id, index=supersession, now=now)
    return {
        "id": record_id,
        "text": payload.get("data", ""),
        "source": payload.get("source"),
        "recorded_at": status["recorded_at"],
        "temporal": status,
        "supersedes": _walk_backward(
            memory, user_id, payload, supersession=supersession, now=now,
            max_hops=MAX_CHAIN_HOPS,
        ),
        "superseded_by": _walk_forward(
            memory, user_id, record_id, supersession=supersession, now=now,
            max_hops=MAX_CHAIN_HOPS,
        ),
        "index": index_state,
    }


def recent_records(
    memory, index, *, user_id: str, limit: int = 10, since: str | None = None,
    kind: str = "all", now: datetime | None = None,
) -> dict[str, Any]:
    """Newest records by ``recorded_at`` (or corrections only), sidecar-
    backed: only the sidecar's lightweight id/time/edge tables are scanned,
    and payloads are fetched by id for just the page returned -- never a
    full vector-store scan.

    Fail-open: when the sidecar is unavailable, returns an empty page with
    ``"index": "unavailable"`` rather than falling back to a store scan, and
    says so via ``"note"``.
    """
    now = now or datetime.now(timezone.utc)
    if index is None:
        return {
            "kind": kind,
            "limit": limit,
            "records": [],
            "index": "unavailable",
            "note": (
                "temporal sidecar unavailable; recency browsing is not "
                "possible without it (never a full store scan)."
            ),
        }
    supersession, index_state = _safe_supersession_index(index)
    records: list[dict[str, Any]] = []
    if kind == "corrections":
        try:
            triples = index.corrections(limit=limit, since=since)
        except Exception:  # noqa: fail-open -- a broken sidecar read must not break /recent
            return {
                "kind": kind, "limit": limit, "records": [], "index": "unavailable",
                "note": "temporal sidecar read failed; recency browsing is not possible.",
            }
        for superseder_id, _recorded_at, target_ids in triples:
            payload = _payload_for_user(memory, superseder_id, user_id)
            if payload is None:
                continue  # sidecar/store disagreement (another user, or deleted) -- skip, don't fabricate
            entry = _render_chain_entry(payload, superseder_id, supersession=supersession, now=now)
            entry["source"] = payload.get("source")
            entry["supersedes"] = target_ids
            records.append(entry)
    else:
        try:
            pairs = index.recent(limit=limit, since=since)
        except Exception:  # noqa: fail-open -- a broken sidecar read must not break /recent
            return {
                "kind": kind, "limit": limit, "records": [], "index": "unavailable",
                "note": "temporal sidecar read failed; recency browsing is not possible.",
            }
        for record_id, _recorded_at in pairs:
            payload = _payload_for_user(memory, record_id, user_id)
            if payload is None:
                continue
            entry = _render_chain_entry(payload, record_id, supersession=supersession, now=now)
            entry["source"] = payload.get("source")
            records.append(entry)
    return {"kind": kind, "limit": limit, "records": records, "index": index_state}


def store_stats(memory, index, *, now: datetime | None = None) -> dict[str, Any]:
    """Cheap, sidecar-backed store-wide counts (PACKET F2c). Total count
    comes from the vector store's own O(1) collection count; every other
    figure is sidecar-only, so a slow/unavailable sidecar never blocks the
    health probe -- it just reports ``"index": "unavailable"``.
    """
    now = now or datetime.now(timezone.utc)
    try:
        total = int(memory.vector_store.collection.count())
    except Exception:  # noqa: fail-open -- stats must never raise into a health probe
        total = None
    if index is None:
        return {
            "total": total,
            "index": "unavailable",
            "note": "temporal sidecar unavailable; per-status counts are not available.",
        }
    try:
        indexed = index.count()
        status_counts = index.status_counts(now=format_instant(now))
        oldest, newest = index.oldest_newest()
    except Exception:  # noqa: fail-open -- a broken sidecar read must not break /stats
        return {
            "total": total,
            "index": "unavailable",
            "note": "temporal sidecar read failed; per-status counts are not available.",
        }
    return {
        "total": total,
        "indexed": indexed,
        "no_recorded_at": (max(0, total - indexed) if total is not None else None),
        **status_counts,
        "oldest_recorded_at": oldest,
        "newest_recorded_at": newest,
        "index": "ok",
    }
