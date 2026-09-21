"""Bounded, deterministic evidence-status presentation.

This module does not retrieve, rank, select, summarize, or establish truth.
It classifies only the raw records already selected by Cogito Hermeneutics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
from typing import Any, Mapping

FEATURE_ENV = "COGITO_HERMENEUTICS_EVIDENCE_STATUS"
SCHEMA_VERSION = "cogito-evidence-envelope/v1"
MCP_MAX_BYTES = 65_536
LIMITATION = (
    "Status describes only the selected evidence returned by the declared "
    "bounded searches. It does not establish truth, corpus completeness, or "
    "absence elsewhere."
)
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_COMPARISON_CUES = (
    "change",
    "changed",
    "changes",
    "different",
    "difference",
    "compare",
    "compared",
    "versus",
    " vs ",
    "then and now",
)
_PRIOR_CUES = (
    "before",
    "previous",
    "previously",
    "prior",
    "earlier",
    "used to",
    "formerly",
    "old",
    "originally",
    "then",
)
_PRIOR_MARKERS = _PRIOR_CUES + ("baseline", "superseded")
_CURRENT_MARKERS = (
    "current",
    "currently",
    "now",
    "today",
    "new",
    "changed",
    "change",
    "replaced",
    "integrated",
    "enabled",
    "active",
)
_VERIFIED_CURRENT = {"verified_current"}
_ANCHOR_STOPWORDS = {
    "about", "after", "all", "also", "and", "are", "before", "between",
    "change", "changed", "changes", "compare", "compared", "current", "did",
    "decide", "decided", "decision", "difference", "different", "does",
    "earlier", "formerly", "from", "happened",
    "has", "have", "how", "into", "now", "old", "originally", "previous",
    "previously", "prior", "system", "than", "that", "the", "then", "thing",
    "rationale", "reason", "this", "today", "used", "versus", "what", "when",
    "where", "which", "why", "with", "it", "choose", "chose", "chosen",
}
_REFERENTIAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:what|how)\s+about\s+"
    r"(?:it|that|this|that\s+work|that\s+thing)\??\s*$|"
    r"^\s*(?:and\s+)?what\s+(?:about|before)\??\s*$",
    re.IGNORECASE,
)
_DECISION_RATIONALE_RE = re.compile(
    r"(?:\bwhy\b.*\b(?:decid\w*|cho(?:ose|se|sen)|select\w*|approv\w*|"
    r"reject\w*)\b|"
    r"\b(?:decision|choice)\b.*\b(?:why|reason|rationale)\b|"
    r"\b(?:what did we decide|which decision)\b.*\b(?:and why|rationale)\b)",
    re.IGNORECASE | re.DOTALL,
)
_DECISION_MARKERS = (
    "decided", "decision", "chose", "chosen", "selected", "approved",
    "rejected", "agreed",
)
_RATIONALE_MARKERS = (
    "because", "rationale", "reason", "reasons", "due to", "so that",
    "in order to",
)


def evidence_status_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the fail-closed feature-flag state."""

    source = os.environ if environ is None else environ
    raw = str(source.get(FEATURE_ENV, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw and raw not in {"0", "false", "no", "off"}:
        logger.warning("invalid %s value; bridge disabled", FEATURE_ENV)
    return False


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text.casefold()} "
    for phrase in phrases:
        if " " in phrase.strip():
            if phrase in padded:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", padded):
            return True
    return False


def _anchors(text: str) -> list[str]:
    anchors: list[str] = []
    for token in _WORD_RE.findall(text.casefold()):
        if len(token) < 3 or token in _ANCHOR_STOPWORDS:
            continue
        if token not in anchors:
            anchors.append(token)
    return anchors[:24]


def _is_comparison(text: str, anchors: list[str]) -> bool:
    return bool(
        _contains_phrase(text, _COMPARISON_CUES)
        and _contains_phrase(text, _PRIOR_CUES)
        and anchors
    )


def _query_shape(query: str, recent_turns: list[str] | None) -> tuple[str, list[str]]:
    current_anchors = _anchors(query)
    if _is_comparison(query, current_anchors):
        return "prior_current_comparison", current_anchors
    if _DECISION_RATIONALE_RE.search(query):
        return "decision_rationale", current_anchors
    if _REFERENTIAL_FOLLOWUP_RE.fullmatch(query):
        for turn in reversed(list(recent_turns or [])[-4:]):
            turn_anchors = _anchors(turn)
            if _is_comparison(turn, turn_anchors):
                return "prior_current_comparison", turn_anchors
            if _DECISION_RATIONALE_RE.search(turn):
                return "decision_rationale", turn_anchors
    return "other", current_anchors


def _stable_id(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return str(
        record.get("record_id")
        or record.get("id")
        or record.get("memory_id")
        or metadata.get("record_id")
        or ""
    )


def _provenance(record: Mapping[str, Any]) -> Any:
    group = record.get("exact_text_group")
    if isinstance(group, Mapping) and group.get("all_provenance"):
        return group.get("all_provenance")
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, Mapping) else None


_SOURCE_IDENTITY_FIELDS = {
    "source",
    "source_kind",
    "source_pointer",
    "source_path",
    "file_path",
    "path",
    "thread_id",
    "session_id",
}


def _mapping_has_source_identity(value: Mapping[str, Any]) -> bool:
    if any(value.get(field) for field in _SOURCE_IDENTITY_FIELDS):
        return True
    metadata = value.get("metadata")
    return bool(
        isinstance(metadata, Mapping)
        and any(metadata.get(field) for field in _SOURCE_IDENTITY_FIELDS)
    )


def _has_provenance(record: Mapping[str, Any]) -> bool:
    if _mapping_has_source_identity(record):
        return True
    provenance = _provenance(record)
    if isinstance(provenance, Mapping):
        return _mapping_has_source_identity(provenance)
    if isinstance(provenance, list):
        return any(
            isinstance(item, Mapping) and _mapping_has_source_identity(item)
            for item in provenance
        )
    return False


def _current_truth_status(record: Mapping[str, Any]) -> str:
    disposition = record.get("current_truth_disposition")
    if isinstance(disposition, Mapping) and disposition.get("status"):
        return str(disposition["status"])
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("current_truth_status"):
        return str(metadata["current_truth_status"])
    return "unknown"


def _is_superseded(record: Mapping[str, Any]) -> bool:
    if record.get("supersession"):
        return True
    status = _current_truth_status(record).casefold()
    return "supersed" in status or "historical_only" in status


def _roles(
    record: Mapping[str, Any],
    anchors: list[str],
    required: list[str],
) -> list[str]:
    text = str(record.get("text") or record.get("raw_text") or "")
    lower = text.casefold()
    common = bool(
        _stable_id(record)
        and text
        and _has_provenance(record)
        and any(re.search(rf"\b{re.escape(anchor)}\b", lower) for anchor in anchors)
    )
    if not common:
        return []
    roles: list[str] = []
    if "prior" in required and _contains_phrase(text, _PRIOR_MARKERS):
        roles.append("prior")
    status = _current_truth_status(record).casefold()
    if "current" in required and not _is_superseded(record) and (
        status in _VERIFIED_CURRENT or _contains_phrase(text, _CURRENT_MARKERS)
    ):
        roles.append("current")
    if "decision" in required and _contains_phrase(text, _DECISION_MARKERS):
        roles.append("decision")
    if "rationale" in required and _contains_phrase(text, _RATIONALE_MARKERS):
        roles.append("rationale")
    return roles


def _provenance_signature(record: Mapping[str, Any]) -> str:
    provenance = _provenance(record)
    try:
        return json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""


def _ambiguity(
    query: str,
    anchors: list[str],
    result: Mapping[str, Any],
    record_roles: list[tuple[Mapping[str, Any], list[str]]],
) -> dict[str, Any]:
    incumbent = result.get("ambiguity")
    if not isinstance(incumbent, Mapping):
        plan = result.get("plan")
        incumbent = plan.get("ambiguity") if isinstance(plan, Mapping) else None
    if (
        isinstance(incumbent, Mapping)
        and incumbent.get("state") not in {None, "", "none"}
    ):
        return {
            "state": "ambiguous_referent",
            "reason": str(
                incumbent.get("reason") or "incumbent_reported_ambiguity"
            ),
        }
    if not anchors:
        return {
            "state": "ambiguous_referent",
            "reason": "no_literal_subject_anchor",
        }
    alternatives = re.search(
        r"\b([A-Za-z0-9_.-]{3,})\s+or\s+([A-Za-z0-9_.-]{3,})\b",
        query,
        re.IGNORECASE,
    )
    if alternatives:
        left, right = (
            alternatives.group(1).casefold(),
            alternatives.group(2).casefold(),
        )
        disambiguator = re.search(
            rf"\b(?:specifically|meaning|for)\s+"
            rf"({re.escape(left)}|{re.escape(right)})\b",
            query,
            re.IGNORECASE,
        )
        if not disambiguator:
            matched: dict[str, set[str]] = {left: set(), right: set()}
            for record, roles in record_roles:
                if not roles:
                    continue
                text = str(record.get("text") or "").casefold()
                signature = _provenance_signature(record)
                if not signature:
                    continue
                for alternative in (left, right):
                    if re.search(rf"\b{re.escape(alternative)}\b", text):
                        matched[alternative].add(signature)
            if matched[left] and matched[right] and any(
                left_signature != right_signature
                for left_signature in matched[left]
                for right_signature in matched[right]
            ):
                return {
                    "state": "ambiguous_referent",
                    "reason": (
                        "provenance_distinct_explicit_alternatives_without_"
                        "disambiguator"
                    ),
                }
    return {"state": "none", "reason": None}


def _evidence_item(record: Mapping[str, Any], roles: list[str]) -> dict[str, Any]:
    item = dict(record)
    text = str(record.get("text") or record.get("raw_text") or "")
    adjustments = record.get("retrieval_adjustments")
    if not isinstance(adjustments, Mapping):
        adjustments = {}
    identity_linked = bool(_stable_id(record) and _has_provenance(record))
    item.update(
        {
            "id": _stable_id(record),
            "identity_linked": identity_linked,
            "identity_strength": (
                "stable_source_linked" if identity_linked else "unlinked"
            ),
            "raw_text": text,
            "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "raw_text_bytes": len(text.encode("utf-8")),
            "raw_text_complete": True,
            "provenance": _provenance(record),
            "current_truth_status": _current_truth_status(record),
            "duplicate_relation": record.get("exact_text_group") or {},
            "external_score": record.get("score"),
            "may_support": roles,
            "selection_reason": list(adjustments.get("reasons", []))[:8],
            "truncated": bool(record.get("truncated", False)),
        }
    )
    return item


def _healthy_source() -> dict[str, Any]:
    return {
        "source": "atomic_memory",
        "requested_path": "recall_hybrid",
        "actual_path": "recall_hybrid",
        "state": "healthy",
        "failure_class": None,
        "authority_limitation": None,
    }


def build_evidence_envelope(
    *,
    query: str,
    result: Mapping[str, Any],
    source_health: list[dict[str, Any]] | None = None,
    recent_turns: list[str] | None = None,
) -> dict[str, Any]:
    """Wrap an incumbent result without changing its selected evidence."""

    records_value = result.get("records", result.get("memories", []))
    records = records_value if isinstance(records_value, list) else []
    legacy_status = str(result.get("retrieval_status") or "")
    not_needed = legacy_status == "not_needed"
    shape, anchors = _query_shape(query, recent_turns)
    required = (
        ["prior", "current"]
        if shape == "prior_current_comparison"
        else ["decision", "rationale"]
        if shape == "decision_rationale"
        else []
    )

    evidence: list[dict[str, Any]] = []
    covered_set: set[str] = set()
    record_roles: list[tuple[Mapping[str, Any], list[str]]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        roles = _roles(record, anchors, required) if required else []
        record_roles.append((record, roles))
        covered_set.update(roles)
        evidence.append(_evidence_item(record, roles))

    ambiguity = (
        _ambiguity(query, anchors, result, record_roles)
        if shape in {"prior_current_comparison", "decision_rationale"}
        else {"state": "none", "reason": None}
    )

    covered = [facet for facet in required if facet in covered_set]
    missing = [facet for facet in required if facet not in covered_set]
    health = list(source_health or ([] if not_needed else [_healthy_source()]))
    if (
        not not_needed
        and any(not item["identity_linked"] for item in evidence)
    ):
        health.append(
            {
                "source": "selected_evidence",
                "requested_path": "stable_identity_and_source_provenance",
                "actual_path": "unlinked_selected_record",
                "state": "error",
                "failure_class": "EvidenceIdentityUnavailable",
                "authority_limitation": (
                    "Selected raw evidence lacked stable identity or source "
                    "provenance."
                ),
            }
        )
    degraded = any(
        item.get("state") in {"unavailable", "error", "degraded_fallback"}
        for item in health
    )

    if not_needed:
        status = "not_needed"
        required = []
        covered = []
        missing = []
        health = []
    elif degraded:
        status = "degraded_evidence"
    elif ambiguity["state"] != "none":
        status = "incomplete_evidence"
    elif required and not covered:
        status = "no_supported_result"
    elif required and missing:
        status = "incomplete_evidence"
    elif evidence:
        status = "supported_evidence"
    else:
        status = "no_supported_result"

    trace = result.get("retrieval_trace")
    plan = result.get("plan")
    if not isinstance(plan, Mapping):
        plan = {}
    trace_truncated = (
        bool(trace.get("trace_truncated"))
        if isinstance(trace, Mapping)
        else False
    )
    selected_limit = int(plan.get("initial_depth") or len(evidence))
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "query_shape": shape,
        "required_facets": required,
        "covered_facets": covered,
        "missing_facets": missing,
        "ambiguity": ambiguity,
        "bounds": {
            "selected_count": len(evidence),
            "selected_limit": selected_limit,
            "trace_included": isinstance(trace, Mapping),
            "truncated": trace_truncated or any(item["truncated"] for item in evidence),
        },
        "source_health": health,
        "evidence": evidence,
        "trace": trace if isinstance(trace, Mapping) else None,
        "limitation": LIMITATION,
        "method": result.get("method"),
        "plan": dict(plan),
        "legacy_retrieval_status": legacy_status,
        "candidate_count": result.get("candidate_count", len(records)),
        "shown_count": len(evidence),
    }
    for key in (
        "retrieval_brief",
        "retrieval_attempts",
        "retrieval_telemetry",
    ):
        if key in result:
            envelope[key] = result[key]
    return envelope


def _json_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def bound_mcp_envelope(
    envelope: Mapping[str, Any],
    *,
    max_bytes: int = MCP_MAX_BYTES,
) -> dict[str, Any]:
    """Bound MCP presentation while retaining full-text hashes and identity."""

    bounded = copy.deepcopy(dict(envelope))
    if _json_size(bounded) <= max_bytes:
        return bounded

    if bounded.get("trace") is not None:
        bounded["trace"] = None
        bounded["bounds"]["trace_included"] = False
        bounded["bounds"]["trace_omitted_for_mcp_bound"] = True

    evidence = bounded.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    while _json_size(bounded) > max_bytes:
        candidates = [
            item for item in evidence
            if isinstance(item, dict) and str(item.get("raw_text") or "")
        ]
        if not candidates:
            break
        largest = max(candidates, key=lambda item: len(str(item["raw_text"]).encode("utf-8")))
        raw = str(largest["raw_text"])
        encoded = raw.encode("utf-8")
        keep = max(0, len(encoded) // 2)
        truncated = encoded[:keep].decode("utf-8", errors="ignore")
        largest["raw_text"] = truncated
        if largest.get("text") == raw:
            largest["text"] = truncated
        largest["raw_text_complete"] = False
        largest["truncated"] = True
        bounded["bounds"]["truncated"] = True

    if _json_size(bounded) <= max_bytes:
        return bounded

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "degraded_evidence",
        "query_shape": bounded.get("query_shape", "other"),
        "required_facets": bounded.get("required_facets", []),
        "covered_facets": [],
        "missing_facets": bounded.get("required_facets", []),
        "ambiguity": bounded.get(
            "ambiguity", {"state": "none", "reason": None}
        ),
        "bounds": {
            "selected_count": len(evidence),
            "selected_limit": bounded.get("bounds", {}).get("selected_limit", 0),
            "trace_included": False,
            "truncated": True,
        },
        "source_health": [
            {
                "source": "mcp_presentation",
                "requested_path": "structured_evidence_envelope",
                "actual_path": "fail_closed",
                "state": "error",
                "failure_class": "MCPEnvelopeLimitExceeded",
                "authority_limitation": (
                    "Identity/provenance could not be retained inside the MCP cap."
                ),
            }
        ],
        "evidence": [],
        "trace": None,
        "limitation": LIMITATION,
    }


def compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
