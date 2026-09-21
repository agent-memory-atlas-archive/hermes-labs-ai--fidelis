"""Bounded retrieval telemetry kept separate from the memory corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fidelis.retrieval-telemetry/v1"
FAILURE_CATEGORIES = {
    "memory_not_invoked",
    "query_too_broad",
    "wrong_time_scope",
    "clear_corpus_noise",
    "relevant_record_absent",
    "relevant_record_ranked_low",
    "useful_evidence_unused",
    "no_supported_evidence",
    "success",
}
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def _enabled() -> bool:
    raw = os.environ.get("FIDELIS_RETRIEVAL_TELEMETRY", "1").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FIDELIS_RETRIEVAL_TELEMETRY must be a boolean")


def _path() -> Path:
    configured = os.environ.get("FIDELIS_RETRIEVAL_TELEMETRY_LOG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cogito" / "retrieval-telemetry.jsonl"


def redact_query(value: str) -> str:
    """Retain the bounded query while removing obvious inline credentials."""

    return _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)[:1000]


def _record_id(record: dict[str, Any], index: int) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    explicit = (
        record.get("record_id")
        or record.get("id")
        or record.get("memory_id")
        or metadata.get("record_id")
        or metadata.get("id")
        or metadata.get("memory_id")
    )
    if explicit:
        return str(explicit)
    identity = "\x00".join(
        (
            str(record.get("text") or record.get("memory") or ""),
            str(record.get("source_pointer") or metadata.get("source_pointer") or ""),
            str(record.get("created_at") or metadata.get("created_at") or ""),
        )
    )
    return f"sha256:{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def build_event(
    *,
    session_id: str | None,
    turn_id: str | None,
    invoked: bool,
    brief: dict[str, Any] | None,
    plan: dict[str, Any],
    first_query: str,
    first_records: list[dict[str, Any]],
    refinement_reason: str | None,
    refined_query: str | None,
    second_records: list[dict[str, Any]],
    result: dict[str, Any],
    failure_category: str,
) -> dict[str, Any]:
    category = (
        failure_category
        if failure_category in FAILURE_CATEGORIES
        else "no_supported_evidence"
    )
    returned = result.get("records")
    if not isinstance(returned, list):
        returned = []
    excluded = result.get("clear_noise_control", {}).get("excluded", [])
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": (session_id or "host-process")[:128],
        "turn_id": (turn_id or "unavailable")[:128],
        "recall_invoked": invoked,
        "retrieval_brief_version": (brief or {}).get("version"),
        "brief_newly_loaded": bool((brief or {}).get("loaded_newly")),
        "entity_or_referent": plan.get("entity_or_referent"),
        "purpose": plan.get("purpose"),
        "time_scope": plan.get("time_scope"),
        "evidence_needed": plan.get("evidence_needed", []),
        "first_query": redact_query(first_query),
        "first_returned_ids": [
            _record_id(record, index)
            for index, record in enumerate(first_records)
        ],
        "refinement_reason": refinement_reason,
        "refined_query": redact_query(refined_query or "") or None,
        "second_returned_ids": [
            _record_id(record, index)
            for index, record in enumerate(second_records)
        ],
        "returned": [
            {
                "record_id": _record_id(record, index),
                "score": record.get("score"),
            }
            for index, record in enumerate(returned)
        ],
        "selected_or_used_ids": [
            _record_id(record, index)
            for index, record in enumerate(returned)
        ],
        "excluded_clear_noise": excluded,
        "refinement_occurred": bool(refined_query),
        "final_disposition": (
            "use_supported_evidence"
            if returned
            else "not_needed"
            if not invoked
            else "abstain"
        ),
        "failure_category": category,
    }


def record(event: dict[str, Any]) -> None:
    """Append one privacy-bounded event. Telemetry never breaks retrieval."""

    try:
        if not _enabled():
            return
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        persisted = {**event, "recorded_at_unix": time.time()}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(persisted, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    except Exception:
        return
