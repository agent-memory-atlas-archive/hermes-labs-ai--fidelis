"""Read-side temporal view over heterogeneous recall hits.

Recall paths disagree about what a hit carries: ``recall_hybrid`` and
``recall_b`` return ``id`` + ``metadata``; ``/query``, ``/recall`` and the
vector-only fallback return bare ``{"text", "score"}``. Rather than thread
record ids through every retriever, this module recovers them after the fact:
stored text is verbatim, so ``content_sha256(text)`` finds the record in the
temporal sidecar, and the vector store returns its payload by id.

Everything here is fail-open. A missing sidecar, an unknown hit, or a store
lookup error leaves the hit in place, annotated ``recorded_at_known: False``
— never dropped, never guessed (docs/TIME-AWARE-SPEC.md, invariants 3 and 6).
Only an invalid caller-supplied ``as_of`` raises, so the server can 400.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from fidelis.temporal import SCHEMA, apply_temporal, content_sha256, parse_instant

logger = logging.getLogger(__name__)

_TEMPORAL_FIELDS = (
    "temporal_schema", "recorded_at", "recorded_at_source", "content_sha256",
    "event_at", "valid_from", "valid_to", "supersedes_json", "source",
)
# Scaffolding keys injected for apply_temporal and removed before returning.
_SCRATCH = "_temporal_payload"
_PARKED = "_retriever_payload"


def parse_as_of(value: Any) -> datetime | None:
    """Validate a caller-supplied ``as_of``; None/"" means "now"."""
    if value is None or value == "":
        return None
    return parse_instant(value)


def overfetch(limit: int, as_of: Any) -> int:
    """As-of filtering removes hits recorded later; fetch deeper so the
    trimmed result is not starved. No effect on ordinary recall."""
    return limit if as_of is None else min(max(limit * 3, limit + 10), 100)


_RECONCILED_FLAG = "_fidelis_reconciled"


def reconcile_index(memory: Any, index: Any, *, full: bool = False) -> int | None:
    """Heal the sidecar from the store. The payloads are the source of truth.

    A sidecar that was deleted and recreated empty, or that missed a write,
    would report a superseded record as ``current`` with ``index: ok`` —
    confidently wrong. Comparison is by record-id SET: equal counts can hide a
    ghost row on one side and an unindexed row on the other.

    ``full=False`` (any time): ADDITIVE only — index the store rows the
    sidecar lacks, never delete. Safe beside concurrent writers: a write that
    lands mid-repair cannot be lost, at worst it is indexed twice, which
    ``record_write`` makes a no-op.
    ``full=True`` (server boot only, before any writer thread exists): a set
    difference in either direction triggers a transactional rebuild, which
    also clears ghost rows.

    Attempted once per index instance whatever the outcome — a failing scan
    must not re-run on every recall. Returns records repaired/rebuilt, or None.
    """
    if index is None or memory is None:
        return None
    if not full and getattr(index, _RECONCILED_FLAG, False):
        return None
    try:
        setattr(index, _RECONCILED_FLAG, True)
    except Exception:  # noqa: an index that cannot carry the flag is not retried either
        return None
    try:
        collection = memory.vector_store.collection
        rows = collection.get(where={"temporal_schema": SCHEMA}, include=["metadatas"])
        ids = [str(i) for i in rows.get("ids") or []]
        metadatas = rows.get("metadatas") or []
        indexed = set(index.record_ids())
        if set(ids) == indexed:
            return None
        if full:
            repaired = index.rebuild(zip(ids, metadatas))
        else:
            repaired = 0
            for record_id, meta in zip(ids, metadatas):
                meta = meta or {}
                if record_id in indexed or not meta.get("content_sha256") or not meta.get("recorded_at"):
                    continue
                try:
                    supersedes = json.loads(meta.get("supersedes_json") or "[]")
                except ValueError:
                    supersedes = []
                index.record_write(
                    record_id, meta["content_sha256"], meta["recorded_at"],
                    supersedes if isinstance(supersedes, list) else [],
                    valid_from=meta.get("valid_from"),
                    valid_to=meta.get("valid_to"),
                )
                repaired += 1
        logger.warning(
            "[fidelis] temporal index healed from store (%s): %s records",
            "rebuild" if full else "additive", repaired,
        )
        return repaired
    except Exception as exc:  # noqa: fail-open — a stale cache is reported, never fatal
        logger.warning("[fidelis] temporal index could not be reconciled: %s", exc)
        return None


def carry_payload(payload: Any) -> dict[str, Any]:
    """Temporal fields of a stored payload the caller ALREADY holds, keyed for
    ``temporal_view``. Spread into a hit so the view does not refetch the same
    record by id — one point-get per over-fetched hit adds up fast."""
    payload = payload if isinstance(payload, Mapping) else {}
    return {_SCRATCH: {k: payload[k] for k in _TEMPORAL_FIELDS if k in payload}}


def _has_temporal(mapping: Any) -> bool:
    return isinstance(mapping, Mapping) and "recorded_at" in mapping


def _stored_payload(memory: Any, record_id: str) -> Mapping[str, Any]:
    row = memory.vector_store.get(record_id)
    return dict(getattr(row, "payload", None) or {})


def _resolve(hit: Mapping[str, Any], memory: Any, index: Any) -> dict[str, Any]:
    """Shallow copy of ``hit`` with ``id`` and a scratch temporal payload."""
    out = dict(hit)
    if isinstance(hit.get(_SCRATCH), Mapping):
        return out  # caller carried the stored payload through; nothing to look up
    for key in ("payload", "metadata"):
        if _has_temporal(hit.get(key)):
            out[_SCRATCH] = hit[key]
            return out
    if _has_temporal(hit):
        out[_SCRATCH] = hit
        return out

    record_id = hit.get("id")
    try:
        if not record_id and index is not None:
            text = hit.get("text")
            if isinstance(text, str) and text:
                # Identical text can legitimately exist more than once (a
                # re-assertion with a new validity window). Then the hash does
                # not identify ONE record, and picking one would be a guess.
                matches = index.ids_by_sha(content_sha256(text))
                record_id = matches[0] if len(matches) == 1 else None
        if record_id and memory is not None:
            payload = _stored_payload(memory, str(record_id))
            out[_SCRATCH] = {k: payload[k] for k in _TEMPORAL_FIELDS if k in payload}
    except Exception as exc:  # noqa: fail-open — leave the hit unannotated-by-lookup
        logger.debug("[fidelis] temporal lookup skipped: %s", exc)
    if record_id:
        out["id"] = str(record_id)
    out.setdefault(_SCRATCH, {})
    return out


def _pool_chain_terminal(
    record_id: str,
    *,
    supersession: Any,
    pool_ids: set,
    as_of: datetime | None,
    max_hops: int = 8,
) -> str | None:
    """Walk the supersession chain from ``record_id`` toward its newest
    superseder, stepping only to ids already present in ``pool_ids`` — this
    never queries the store (WORK PACKET E, "pool-only replacement rule").

    Returns the id of the last such superseder reached (the chain's
    terminal, present-in-pool replacement), or ``None`` when ``record_id``
    has no superseder present in the pool. Bounded to ``max_hops`` steps;
    each id is visited at most once, so a cyclic declaration (A supersedes
    B, B supersedes A) terminates instead of looping.
    """
    current = record_id
    visited = {record_id}
    terminal: str | None = None
    for _ in range(max_hops):
        try:
            superseders = supersession.superseders(current, as_of=as_of)
        except Exception:  # noqa: fail-open — a broken chain walk loses the pull, not the recall
            break
        candidates = [s for s in superseders if s in pool_ids and s not in visited]
        if not candidates:
            break
        # superseders() is sorted oldest-first, unknown-time last; the last
        # present-in-pool candidate is the newest one available in the pool.
        current = candidates[-1]
        visited.add(current)
        terminal = current
    return terminal


def _replacements_from_pool(
    window: list,
    pool_by_id: dict,
    *,
    supersession: Any,
    as_of: datetime | None,
) -> list[tuple[str, str]]:
    """``(terminal_id, superseded_id)`` pairs for replacements the window is
    missing (WORK PACKET E step 4 — "keep the replacement in view").

    For each hit IN THE WINDOW whose status is ``superseded``, follow the
    supersession chain to its newest superseder that is itself present in
    ``pool_by_id``. Nothing is ever fetched from the store: only records
    already retrieved by relevance in this same request are eligible — a
    prior version that fetched a missing superseder by id was removed on
    review because it bypassed the ephemera/user filters already applied to
    the pool and could surface a stale mid-chain record. A terminal already
    in the window, or already returned for an earlier window record, is not
    repeated.
    """
    if supersession is None:
        return []
    window_ids = {
        hit.get("id") for hit in window if isinstance(hit, Mapping) and hit.get("id")
    }
    pool_ids = set(pool_by_id)
    pulled: set[str] = set()
    replacements: list[tuple[str, str]] = []
    for hit in window:
        if not isinstance(hit, Mapping):
            continue
        record_id = hit.get("id")
        temporal = hit.get("temporal")
        if not record_id or not isinstance(temporal, Mapping):
            continue
        if temporal.get("status") != "superseded":
            continue
        terminal = _pool_chain_terminal(
            str(record_id), supersession=supersession, pool_ids=pool_ids, as_of=as_of,
        )
        if terminal is None or terminal in window_ids or terminal in pulled:
            continue
        pulled.add(terminal)
        replacements.append((terminal, str(record_id)))
    return replacements


def temporal_view(
    memories: list,
    *,
    memory: Any,
    index: Any,
    as_of: datetime | None = None,
    historical: bool = False,
    limit: int | None = None,
    now: datetime | None = None,
) -> list:
    """Annotate, as-of filter, and order a relevance-ranked list (WORK PACKET E).

    Eligibility is decided by RELEVANCE before recency reorders anything
    (docs/TIME-AWARE-SPEC.md invariant 4): the window is the first ``limit``
    records of the pool, IN INCOMING ORDER, after the as-of exclusion (the
    only permitted drop, invariant 5). A superseded record that survives
    into the window then pulls its newest superseder INTO VIEW, but only
    from records already retrieved by relevance in THIS pool — never a fresh
    store fetch (see :func:`_replacements_from_pool`). The result may
    therefore be up to ``len(pulled)`` longer than ``limit``: dropping
    either half of a supersession pair would lose exactly the information
    this feature exists to give. ``historical=True`` is a pure, unreordered
    relevance view, annotated only — it does not pull a replacement in.
    """
    reconcile_index(memory, index)
    try:
        resolved = [
            _resolve(hit, memory, index) if isinstance(hit, Mapping) else hit
            for hit in memories or []
        ]
        # apply_temporal reads fields from "payload"; present the scratch
        # mapping under that name and park any real payload the retriever
        # supplied so it can be put back untouched.
        staged = []
        for hit in resolved:
            if isinstance(hit, Mapping):
                hit = dict(hit)
                if "payload" in hit:
                    hit[_PARKED] = hit["payload"]
                hit["payload"] = hit.pop(_SCRATCH)
            staged.append(hit)
        # A sidecar that opens but cannot be read is the same as no sidecar:
        # hits are still annotated, with "index": "unavailable".
        supersession = None
        if index is not None:
            try:
                supersession = index.supersession_index()
            except Exception as exc:  # noqa: fail-open — damaged sidecar
                logger.warning("[fidelis] temporal index unreadable: %s", exc)

        effective_now = now or datetime.now(timezone.utc)

        # Steps 1-2: annotate + the ONLY permitted exclusion (as-of
        # visibility). historical=True here means "annotate, do not reorder
        # yet" — apply_temporal's own contract — because relevance order is
        # exactly what the eligibility window (step 3) must be built from.
        pool = apply_temporal(
            staged, index=supersession, now=effective_now, as_of=as_of, historical=True,
        )

        # Step 3: eligibility window. Relevance decides who is in play
        # BEFORE any recency-based reordering (invariant 4); `limit=None`
        # means the whole pool is eligible, so nothing can ever be missing
        # from the window and step 4 is a no-op.
        window = list(pool[:limit]) if limit is not None else list(pool)

        if historical:
            # historical is a pure relevance view: annotated, top-limit, and
            # (decision, documented here) it never pulls a replacement in —
            # pulling is a recency-aware "keep the demoted-but-relevant fact
            # visible" behavior, which historical mode has no place for.
            viewed = window
        else:
            pool_by_id = {
                hit["id"]: hit for hit in pool if isinstance(hit, Mapping) and hit.get("id")
            }
            # Step 4: pool-only replacement.
            replacements = _replacements_from_pool(
                window, pool_by_id, supersession=supersession, as_of=as_of,
            )
            for terminal_id, superseded_id in replacements:
                pulled = dict(pool_by_id[terminal_id])
                pulled["pulled_by"] = {"supersedes": superseded_id}
                window.append(pulled)

            # Step 5: reorder WITHIN the window only — current first,
            # relevance order preserved inside each group, exactly
            # apply_temporal's stable partition (its own contract: correct
            # for an already-relevance-selected set, which `window` now is).
            # Step 6: no further trim is needed — `window` was already cut
            # to `limit` before any pull, so its length here is already
            # <= limit + len(replacements); trimming further would re-drop
            # exactly what step 4 just restored.
            viewed = apply_temporal(
                window, index=supersession, now=effective_now, as_of=as_of, historical=False,
            )
    except Exception as exc:  # noqa: fail-open — recall must survive any temporal fault
        logger.warning("[fidelis] temporal view unavailable: %s", exc)
        # Still say so on every hit: an unannotated hit is indistinguishable
        # from a current one, which is exactly the ambiguity to avoid.
        passthrough = [
            {**hit, "temporal": {"index": "unavailable", "recorded_at_known": False}}
            if isinstance(hit, Mapping) and "temporal" not in hit else hit
            for hit in memories or []
        ]
        return passthrough[:limit] if limit else passthrough

    out = []
    for hit in viewed:
        if isinstance(hit, Mapping):
            hit = dict(hit)
            hit.pop("payload", None)
            # Without as_of, "evaluated_at" is wall-clock and would make every
            # response unique; responses must stay byte-stable across calls.
            if as_of is None and isinstance(hit.get("temporal"), Mapping):
                hit["temporal"] = {
                    k: v for k, v in hit["temporal"].items() if k != "evaluated_at"
                }
            if _PARKED in hit:
                hit["payload"] = hit.pop(_PARKED)
        out.append(hit)
    return out
