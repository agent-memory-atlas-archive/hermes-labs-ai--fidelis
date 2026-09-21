"""Graceful-degradation layer for fidelis writes.

When the upstream LLM (Ollama / mem0) is unreachable, we MUST NOT lose the
write. Instead, queue it locally as JSONL and let a sync job replay later.

This module exists because of the 2026-04-19 incident: Ollama's socket layer
broke under Python 3.14, every `cogito add` returned HTTP 500, and a full
session of memory was silently lost. The Hermes Seal v1 made this gap explicit.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from fidelis import write_gate
from fidelis.relation_envelope import (
    ingestion_payload,
    relation_envelopes_enabled,
)
from fidelis.temporal import build_temporal_fields
from fidelis.temporal_index import TemporalIndex, open_index

MAX_ATTEMPTS = 5

TEMPORAL_ENV = "COGITO_TEMPORAL_V1"
_FALSE = {"0", "false", "no", "off"}
# Caller-declared temporal keys, carried in the same ``metadata`` mapping the
# relation envelope uses. See docs/TIME-AWARE-SPEC.md.
_TEMPORAL_DECLARED_KEYS = ("event_at", "valid_from", "valid_to", "supersedes", "source")
# Declaring any of these gives a re-assertion new temporal meaning, so it is a
# new record rather than a duplicate of identical text.
_DEDUP_EXEMPT_KEYS = ("event_at", "valid_from", "valid_to", "supersedes")

# Set once by the server after config load. None means no sidecar: writes are
# still stamped, but exact-duplicate detection and the supersession index are
# skipped (the payload fields stay the source of truth; the index is rebuildable).
_temporal_index: TemporalIndex | None = None
_gate_config: dict[str, object] = {}


def temporal_enabled() -> bool:
    return os.environ.get(TEMPORAL_ENV, "1").strip().lower() not in _FALSE


def configure_temporal(
    store_path: str | Path | None,
    cfg: Mapping[str, object] | None = None,
) -> TemporalIndex | None:
    """Bind the write path to a store's temporal sidecar and gate config."""
    global _temporal_index, _gate_config
    if _temporal_index is not None:
        _temporal_index.close()
    _temporal_index = open_index(store_path) if store_path and temporal_enabled() else None
    _gate_config = {"write_gate": (cfg or {}).get("write_gate", True)}
    return _temporal_index


def temporal_index() -> TemporalIndex | None:
    return _temporal_index


def reconcile_temporal_index(memory) -> int | None:
    """Full sidecar reconcile. Call at server boot ONLY, before writer threads
    exist — it may rebuild. See temporal_recall.reconcile_index."""
    from fidelis.temporal_recall import reconcile_index

    return reconcile_index(memory, _temporal_index, full=True)


def _declared_temporal(metadata: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    declared = {
        key: metadata[key]
        for key in _TEMPORAL_DECLARED_KEYS
        if metadata.get(key) is not None
    }
    # ``source`` is also a relation-envelope provenance field and may be a
    # non-string there; only a plain string is a temporal declaration.
    if not isinstance(declared.get("source"), str):
        declared.pop("source", None)
    return declared


def _supersedes_targets_exist(
    memory, user_id: str, ids: list[str],
) -> tuple[list[str], list[str]]:
    """``(unknown_ids, already_superseded_ids)`` for a declared ``supersedes``
    list, checked against the SAME user's records only (docs/TIME-AWARE-SPEC.md).

    ``memory.vector_store.get(id)`` returns ``None`` (mem0's Chroma wrapper)
    for a missing id; a record found for a different ``user_id`` is treated
    as unknown too, rather than leaking cross-user existence. Superseding an
    already-superseded record is allowed (chains/branches); its id is
    reported back so the caller can say so.

    Propagates any exception the lookup itself raises, UNCHANGED, so the
    caller can tell "definitely missing" (empty/non-empty list, no raise)
    from "could not check right now" (raise) -- these get different
    treatment (reject vs. fall through to the queue).
    """
    unknown: list[str] = []
    already: list[str] = []
    for target_id in ids:
        row = memory.vector_store.get(target_id)
        payload = dict(getattr(row, "payload", None) or {})
        if not payload.get("data") or payload.get("user_id") != user_id:
            unknown.append(target_id)
            continue
        if _temporal_index is not None:
            try:
                if _temporal_index.is_superseded(target_id):
                    already.append(target_id)
            except Exception:  # noqa: fail-open -- a broken index must not block a write
                pass
    return unknown, already


def _pre_write(
    text: str,
    metadata: Mapping[str, object] | None,
    record_id: str | None,
    *,
    memory,
    user_id: str,
) -> tuple[dict | None, dict[str, str], list[str] | None, list[str]]:
    """Gate and validate a write BEFORE any dependency call.

    Runs outside safe_add's try-block on purpose: a rejected write must never
    be queued, and an invalid temporal declaration must surface as ValueError
    (HTTP 400), not be swallowed into the dependency-failure queue path.

    Returns ``(rejection_result_or_None, temporal_fields,
    pending_supersedes_check, already_superseded_targets)``.

    ``pending_supersedes_check`` is the declared ``supersedes`` id list when
    their existence could not be verified because the lookup itself raised
    (store unavailable) -- the write must NOT be rejected for that; the ids
    ride along on a queued record so ``replay_queue`` can re-validate them
    and dead-letter with a clear reason if they still do not exist.
    ``already_superseded_targets`` lists declared ``supersedes`` ids that
    exist but are themselves already superseded -- allowed (chains/
    branches), reported so the caller can say so truthfully.

    Raises ``ValueError`` when a declared ``supersedes`` id definitely does
    not exist for this user (mapped to HTTP 400 by the caller), exactly like
    any other invalid temporal declaration.
    """
    decision = write_gate.evaluate(text, config=_gate_config)
    if not decision.accept:
        return (
            {"status": "rejected", "reason": decision.reason, "detail": decision.detail},
            {}, None, [],
        )
    if not temporal_enabled():
        return None, {}, None, []
    fields = build_temporal_fields(
        text,
        now=datetime.now(timezone.utc),
        declared=_declared_temporal(metadata),
        record_id=record_id,
    )
    pending_check: list[str] | None = None
    already_superseded: list[str] = []
    supersedes_json = fields.get("supersedes_json")
    if supersedes_json:
        ids = json.loads(supersedes_json)
        try:
            unknown, already_superseded = _supersedes_targets_exist(memory, user_id, ids)
        except Exception:  # noqa: fail-open -- store unavailable; re-verify on replay
            pending_check = ids
        else:
            if unknown:
                raise ValueError(
                    "supersedes: unknown record id(s) for this user: "
                    + ", ".join(unknown)
                )
    return None, fields, pending_check, already_superseded


def _find_duplicate(memory, user_id: str, fields: Mapping[str, str],
                    metadata: Mapping[str, object] | None) -> str | None:
    """Existing record id holding byte-identical text, or None.

    The sidecar is a cache, so a hit only counts when the store still holds
    that row for this user — otherwise we would report a write as present
    when nothing is persisted.
    """
    if _temporal_index is None or not fields:
        return None
    declared = _declared_temporal(metadata)
    if any(key in declared for key in _DEDUP_EXEMPT_KEYS):
        return None
    try:
        # Only a record that is still CURRENT makes a new write redundant.
        # If every copy of this text has since been superseded, storing it
        # again is a re-assertion (A -> B -> back to A), not a duplicate.
        existing = next(
            (rid for rid in _temporal_index.ids_by_sha(fields["content_sha256"])
             if not _temporal_index.is_superseded(rid)),
            None,
        )
        if not existing:
            return None
        row = memory.vector_store.get(existing)
        payload = dict(getattr(row, "payload", None) or {})
    except Exception:  # noqa: fail-open — an index/lookup fault must not block a write
        return None
    if not payload.get("data") or payload.get("user_id") != user_id:
        return None
    return existing


def _index_write(mid: str, fields: Mapping[str, str], result: dict | None = None) -> None:
    if _temporal_index is None or not fields:
        return
    try:
        _temporal_index.record_write(
            mid,
            fields["content_sha256"],
            fields["recorded_at"],
            json.loads(fields.get("supersedes_json") or "[]"),
            valid_from=fields.get("valid_from"),
            valid_to=fields.get("valid_to"),
        )
    except Exception:  # noqa: the vector is already persisted; the index is rebuildable
        if result is not None:
            result["temporal_index"] = "failed"


def _queue_dir() -> Path:
    """Resolve queue directory from env at call time so tests can isolate."""
    env = os.environ.get("FIDELIS_QUEUE_DIR") or os.environ.get("COGITO_QUEUE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cogito" / "queue"


def _dead_dir() -> Path:
    return _queue_dir() / "dead"


# Backwards-compat module-level paths. These resolve at import time and are kept
# so existing callers / scripts that reference QUEUE_DIR still work, but the
# functions below always use _queue_dir() to honour env overrides set after
# import (test isolation).
QUEUE_DIR = _queue_dir()
DEAD_DIR = _dead_dir()


def _ensure_queue() -> Path:
    qdir = _queue_dir()
    qdir.mkdir(parents=True, exist_ok=True)
    return qdir


def _ensure_dead() -> Path:
    ddir = _dead_dir()
    ddir.mkdir(parents=True, exist_ok=True)
    return ddir


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: write to sibling tmp then os.replace.

    SIGKILL between write and replace leaves either the old file (if it
    existed) or no file — never a half-written corrupt JSON.
    """
    tmp = path.parent / (path.name + ".tmp")
    data = json.dumps(payload).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def queue_write(
    text: str,
    user_id: str,
    kind: str = "add",
    *,
    record_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Append a memory write to the local queue and return its id."""
    qdir = _ensure_queue()
    mid = record_id or str(uuid.uuid4())
    rec = {
        "id": mid, "ts": time.time(), "kind": kind, "user_id": user_id,
        "text": text, "attempts": 0,
    }
    if metadata is not None:
        rec["metadata"] = dict(metadata)
    qfile = qdir / f"{int(rec['ts'])}-{mid}.json"
    _atomic_write_json(qfile, rec)
    return mid


def queued_count() -> int:
    qdir = _queue_dir()
    if not qdir.exists():
        return 0
    return len([p for p in qdir.glob("*.json") if p.is_file()])


def dead_count() -> int:
    ddir = _dead_dir()
    if not ddir.exists():
        return 0
    return len(list(ddir.glob("*.json")))


def _embed_bounded(embedding_model, text: str, **kwargs):
    """Embed text, shrinking the input on failure so an oversized memory is never lost.

    nomic-embed-text returns HTTP 500 past its ~2048-token context, and token density
    varies (dense markdown/structured text hits the cap at far fewer chars than prose),
    so a fixed char bound is unreliable — try the full text, then progressively shorter
    prefixes. The full text is still stored verbatim in the payload; only the vector (a
    retrieval key) is computed from the prefix. This closes the embed-layer twin of the
    noop:0b silent-loss bug: large verbatim writes used to 500 on embed and queue forever
    (the 7-15KB [voice:user] dumps that filled ~/.cogito/queue and the dead-letter dir).
    """
    candidates = [text]
    for limit in (6000, 4000, 2000, 1000, 400):
        if len(text) > limit:
            candidates.append(text[:limit])
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            return embedding_model.embed(candidate, **kwargs)
        except Exception as e:  # noqa: retry with a shorter prefix; re-raised below if every size fails
            last_exc = e
    raise last_exc  # type: ignore[misc]  # all sizes failed → genuine dependency outage, let safe_add queue it


def _usable_extracted_memories(result) -> list[str]:
    """Return only non-blank facts from mem0's optional result envelope."""
    if not isinstance(result, dict):
        return []
    extracted = []
    for item in result.get("results") or []:
        if not isinstance(item, dict):
            continue
        value = item.get("memory")
        if isinstance(value, str) and value.strip():
            extracted.append(value)
    return extracted



def safe_add(
    memory,
    text: str,
    user_id: str,
    kind: str = "add",
    *,
    record_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict:
    """Try to write through mem0; on dependency failure, queue locally.

    Returns:
      {"status": "stored", "extracted": [...]}  — happy path
      {"status": "queued", "id": "...", "reason": "..."}  — fallback path
      {"status": "rejected", "reason": code}  — write gate; nothing written or queued
      {"status": "duplicate", "id": existing}  — identical text already stored

    Raises ValueError for an invalid temporal declaration (caller maps to 400) --
    including a declared ``supersedes`` id that does not exist for this user.
    """
    relation_enabled = relation_envelopes_enabled()
    store_mid = record_id if relation_enabled and record_id else str(uuid.uuid4())
    rejected, temporal_fields, pending_supersedes_check, already_superseded = _pre_write(
        text, metadata, store_mid if kind == "store" else None,
        memory=memory, user_id=user_id,
    )
    if rejected is not None:
        return rejected
    if kind == "store":
        duplicate_of = _find_duplicate(memory, user_id, temporal_fields, metadata)
        if duplicate_of is not None:
            return {"status": "duplicate", "id": duplicate_of, "extracted": []}
    try:
        if kind == "store":
            mid = store_mid
            vector = _embed_bounded(memory.embedding_model, text)
            envelope = None
            if relation_enabled:
                payload, envelope = ingestion_payload(
                    text=text,
                    user_id=user_id,
                    record_id=mid,
                    metadata=metadata,
                )
            else:
                # Exact legacy payload when the feature is disabled.
                payload = {"data": text, "user_id": user_id}
            payload.update(temporal_fields)
            memory.vector_store.insert(
                vectors=[vector],
                payloads=[payload],
                ids=[mid],
            )
            result = {"status": "stored", "id": mid, "extracted": [text]}
            if temporal_fields:
                result["recorded_at"] = temporal_fields["recorded_at"]
                # Echo exactly what was recorded, normalised, so a caller can
                # confirm without a recall (docs/TIME-AWARE-SPEC.md, F1b).
                for key in ("valid_from", "valid_to", "event_at"):
                    if key in temporal_fields:
                        result[key] = temporal_fields[key]
                if "supersedes_json" in temporal_fields:
                    result["supersedes"] = json.loads(temporal_fields["supersedes_json"])
            if already_superseded:
                result["superseded_targets_already_superseded"] = already_superseded
            if envelope is not None:
                result["relation_envelope"] = envelope
            _index_write(mid, temporal_fields, result)
            return result
        result = memory.add(text, user_id=user_id)
        extracted = _usable_extracted_memories(result)
        if not extracted:
            # mem0 swallows LLM-extraction failures internally (logs "LLM
            # extraction failed", returns [] instead of raising) — e.g. when
            # llm_model points at a missing Ollama model. Without this fallback
            # the caller gets {"status": "stored"} while nothing was written:
            # silent loss on the live /add path, the exact failure this module
            # exists to prevent. Degrade to the same verbatim insert the
            # kind=="store" path and replay_queue() already use.
            mid = str(uuid.uuid4())
            vector = _embed_bounded(memory.embedding_model, text)
            memory.vector_store.insert(
                vectors=[vector],
                payloads=[{"data": text, "user_id": user_id, **temporal_fields}],
                ids=[mid],
            )
            result = {"status": "stored", "id": mid, "extracted": [text],
                      "degraded": "verbatim-fallback-empty-extraction"}
            _index_write(mid, temporal_fields, result)
            return result
        return {"status": "stored", "extracted": extracted}
    except Exception as e:
        # Temporal declarations must survive the queue even when relation
        # envelopes are off, or a replayed write would lose its validity window.
        queue_metadata = metadata if relation_enabled else (_declared_temporal(metadata) or None)
        if pending_supersedes_check:
            # The existence check could not be completed (store unavailable);
            # carry the ids forward so replay_queue re-validates them before
            # inserting, instead of writing an unverified supersession.
            queue_metadata = dict(queue_metadata or {})
            queue_metadata["_pending_supersedes_check"] = pending_supersedes_check
        mid = queue_write(
            text,
            user_id,
            kind=kind,
            record_id=record_id if relation_enabled else None,
            metadata=queue_metadata,
        )
        return {"status": "queued", "id": mid, "reason": f"{type(e).__name__}: {e}"}


def replay_queue(memory, user_id: str) -> dict:
    """Replay queued writes through mem0. Removes successfully replayed records.

    Two-stage fallback for "add"-kind items: first try the full mem0.add() path
    (which uses the LLM for fact extraction). If that fails (most commonly
    because the configured LLM model is not available on Ollama), fall back to
    a direct verbatim vector insert — equivalent to /store. The user gets
    something recallable instead of a write that's stuck in the queue forever.

    Per-item retry budget: each failure increments rec["attempts"]. After
    MAX_ATTEMPTS the record is moved to the dead-letter queue so a permanently
    poisoned write cannot keep the replay loop hot forever.
    """
    qdir = _queue_dir()
    if not qdir.exists():
        return {"replayed": 0, "remaining": 0, "replayed_verbatim": 0, "failed": 0, "dead_lettered": 0}
    replayed = 0
    replayed_verbatim = 0
    failed = 0
    dead_lettered = 0
    ddir = _dead_dir()
    for qfile in sorted(qdir.glob("*.json")):
        if not qfile.is_file():
            continue
        try:
            rec = json.loads(qfile.read_text())
        except Exception:  # noqa: silent — corrupted queue file moved to dead-letter for inspection
            _move_to_dead(qfile)
            dead_lettered += 1
            continue
        # Writes queued before the gate existed (or by another client) are
        # gated here. A rejected record is dead-lettered for inspection, never
        # inserted and never silently unlinked.
        decision = write_gate.evaluate(str(rec.get("text") or ""), config=_gate_config)
        if not decision.accept:
            rec["last_error"] = f"rejected: {decision.reason}"
            _ensure_dead()
            _atomic_write_json(ddir / qfile.name, rec)
            try:
                qfile.unlink()
            except OSError:  # noqa: silent — file already moved
                pass
            dead_lettered += 1
            continue
        # A record queued because its declared supersedes ids could not be
        # verified (store was unavailable, docs/TIME-AWARE-SPEC.md F1a) is
        # re-checked here, now that we are attempting replay. If the store is
        # reachable and the id(s) still do not exist, dead-letter with a
        # clear reason instead of writing an unverified supersession or
        # retrying forever. If verification itself still fails (store still
        # down), fall through to the normal replay attempt below, which will
        # fail the same way and count against the ordinary retry budget.
        rec_metadata = rec.get("metadata")
        pending_ids = (
            rec_metadata.get("_pending_supersedes_check")
            if isinstance(rec_metadata, dict)
            else None
        )
        if pending_ids:
            try:
                unknown, _ = _supersedes_targets_exist(memory, rec.get("user_id"), pending_ids)
            except Exception:  # noqa: fail-open -- still unverifiable; try the normal replay path
                unknown = None
            if unknown:
                rec["last_error"] = (
                    "rejected: supersedes id(s) not found: " + ", ".join(unknown)
                )
                _ensure_dead()
                _atomic_write_json(ddir / qfile.name, rec)
                try:
                    qfile.unlink()
                except OSError:  # noqa: silent — file already moved
                    pass
                dead_lettered += 1
                continue
        try:
            if rec.get("kind") == "store":
                _replay_verbatim(memory, rec)
            else:
                # The extraction LLM can fail two ways: raise (model unreachable)
                # OR return zero facts without raising (e.g. llm_model=noop:0b).
                # Both must trigger the verbatim fallback, else the queued write
                # is silently dropped on replay — the exact loss this layer exists
                # to prevent.
                add_result = None
                try:
                    add_result = memory.add(rec["text"], user_id=rec["user_id"])
                except Exception:
                    add_result = None
                if not _usable_extracted_memories(add_result):
                    _replay_verbatim(memory, rec)
                    replayed_verbatim += 1
            qfile.unlink()
            replayed += 1
        except Exception as e:
            attempts = int(rec.get("attempts", 0)) + 1
            rec["attempts"] = attempts
            rec["last_error"] = f"{type(e).__name__}: {e}"
            if attempts >= MAX_ATTEMPTS:
                _ensure_dead()
                _atomic_write_json(ddir / qfile.name, rec)
                try:
                    qfile.unlink()
                except OSError:  # noqa: silent — file already moved
                    pass
                dead_lettered += 1
            else:
                _atomic_write_json(qfile, rec)
                failed += 1
    remaining = len([p for p in qdir.glob("*.json") if p.is_file()])
    return {
        "replayed": replayed,
        "replayed_verbatim": replayed_verbatim,
        "failed": failed,
        "dead_lettered": dead_lettered,
        "remaining": remaining,
    }


def _move_to_dead(qfile: Path) -> None:
    _ensure_dead()
    target = _dead_dir() / qfile.name
    try:
        qfile.rename(target)
    except OSError:  # noqa: silent — best-effort dead-letter; if rename fails the file stays for next sweep
        pass


def _replay_temporal_fields(rec: dict, mid: str) -> dict[str, str]:
    """Temporal fields for a replayed record.

    ``recorded_at`` is the ORIGINAL enqueue time (``ts``), so a write delayed
    by an outage does not look newer than it is. A record with no usable
    ``ts`` is stamped at replay time and says so.
    """
    if not temporal_enabled():
        return {}
    try:
        queued_at = datetime.fromtimestamp(float(rec["ts"]), timezone.utc)
        origin = "queue"
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        queued_at = datetime.now(timezone.utc)
        origin = "replay"
    return build_temporal_fields(
        rec["text"],
        now=queued_at,
        declared=_declared_temporal(rec.get("metadata")),
        record_id=mid,
        recorded_at=queued_at,
        recorded_at_source=origin,
    )


def _replay_verbatim(memory, rec: dict) -> None:
    """Insert a queued record directly into the vector store without LLM
    extraction. Used as the safe-add fast path AND as the LLM-fallback path
    inside replay_queue.

    Idempotent under retry: chroma's insert with a duplicate id raises; we
    catch and treat as success because the prior partial-attempt already
    persisted the vector.
    """
    mid = rec.get("id") or str(uuid.uuid4())
    vector = _embed_bounded(memory.embedding_model, rec["text"])
    if relation_envelopes_enabled():
        payload, _ = ingestion_payload(
            text=rec["text"],
            user_id=rec["user_id"],
            record_id=mid,
            metadata=rec.get("metadata"),
        )
    else:
        payload = {"data": rec["text"], "user_id": rec["user_id"]}
    temporal_fields = _replay_temporal_fields(rec, mid)
    payload.update(temporal_fields)
    try:
        memory.vector_store.insert(
            vectors=[vector],
            payloads=[payload],
            ids=[mid],
        )
        _index_write(mid, temporal_fields)
    except Exception as e:
        # chromadb raises on duplicate ids — that's a successful prior insert.
        # For relation records with caller-stable IDs, verify the prior payload
        # before calling the retry idempotent.  Otherwise an ID collision could
        # silently discard different raw evidence.
        from chromadb.errors import DuplicateIDError
        duplicate_markers = ("existing embedding id", "duplicate embedding id",
                             "duplicate id", "id already exists")
        if isinstance(e, DuplicateIDError) or any(marker in str(e).lower() for marker in duplicate_markers):
            if relation_envelopes_enabled() and rec.get("metadata") is not None:
                existing = memory.vector_store.get(mid)
                existing_payload = dict(getattr(existing, "payload", None) or {})
                expected_payload, _ = ingestion_payload(
                    text=rec["text"],
                    user_id=rec["user_id"],
                    record_id=mid,
                    metadata=rec.get("metadata"),
                )
                if (
                    existing_payload.get("data") != expected_payload["data"]
                    or existing_payload.get("relation_envelope_v1_json")
                    != expected_payload["relation_envelope_v1_json"]
                ):
                    raise RuntimeError(
                        "duplicate relation record id has different raw evidence"
                    ) from e
            return
        raise
