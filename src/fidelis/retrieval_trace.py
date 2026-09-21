"""Opt-in, behavior-neutral retrieval tracing for Cogito Hermeneutics.

Trace is off by default. It records only bounded hashes, enums, counts, ranks,
timings, and score components; raw queries, source text, local paths, prompts,
and exception messages are intentionally excluded.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


TRACE_SCHEMA_VERSION = "retrieval-trace/v1"
PLAN_SCHEMA_VERSION = "retrieval-plan/v2"
CANDIDATE_SCHEMA_VERSION = "candidate-envelope/v1"
RESPONSE_SCHEMA_VERSION = "retrieval-response/v1.1"
DEFAULT_TRACE_MAX_BYTES = 32768
MIN_TRACE_MAX_BYTES = 1024
MAX_TRACE_MAX_BYTES = 65536


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_requirements(query: str) -> list[str]:
    """Trace-only deterministic query-shape signals; never evidence claims."""
    patterns = (
        (
            "comparison",
            r"\b(?:changed?|change|differ(?:ence|ent)?|compare|versus|vs\.?)\b",
        ),
        (
            "prior_relation",
            r"\b(?:before|previously|previous|prior|earlier|used to)\b",
        ),
        (
            "current_relation",
            r"\b(?:current|currently|now|latest|today|remain)\b",
        ),
        ("timeline", r"\b(?:timeline|chronolog\w*|over time|when)\b"),
        (
            "source_request",
            r"\b(?:source|provenance|record|evidence|which file)\b",
        ),
        (
            "decision_relation",
            r"\b(?:decid\w*|decision|rationale|why)\b",
        ),
    )
    return [
        requirement
        for requirement, pattern in patterns
        if re.search(pattern, query, re.IGNORECASE)
    ]


class RetrievalTrace:
    """Collect a bounded additive trace without affecting retrieval decisions."""

    def __init__(
        self,
        query: str,
        *,
        max_bytes: int = DEFAULT_TRACE_MAX_BYTES,
    ):
        self.query_hash = _stable_hash(query)
        self.max_bytes = max(
            MIN_TRACE_MAX_BYTES,
            min(int(max_bytes), MAX_TRACE_MAX_BYTES),
        )
        self.events: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.selection: list[dict[str, Any]] = []
        self.plan: dict[str, Any] | None = None

    @staticmethod
    def probe_id(subquery: str, index: int) -> str:
        return f"probe-{index + 1}-{_stable_hash(subquery)[:12]}"

    @staticmethod
    def candidate_id_hash(candidate_key: str) -> str:
        return _stable_hash(candidate_key)[:20]

    def set_plan(self, plan: Any, query: str) -> None:
        legacy_plan = plan.to_dict()
        detected_entities = legacy_plan.pop("detected_entities", [])
        self.plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "legacy_plan": legacy_plan,
            "detected_entity_hashes": [
                _stable_hash(str(entity))[:20]
                for entity in detected_entities
            ],
            "query_requirements": _query_requirements(query),
            "requirements_are_query_shape_only": True,
            "candidate_evidence_or_truth_inferred": False,
        }

    def event(self, stage: str, status: str, **fields: Any) -> None:
        allowed = {
            "probe_id",
            "requested_count",
            "returned_count",
            "unique_count",
            "pool_limit",
            "saturated",
            "failure_class",
            "fallback_path",
            "available",
            "run_count",
            "candidate_count",
            "latency_ms",
            "source_kind",
        }
        event = {
            "event_id": f"event-{len(self.events) + 1}",
            "stage": str(stage),
            "status": str(status),
        }
        for key, value in fields.items():
            if key in allowed and value is not None:
                event[key] = value
        self.events.append(event)

    def record_candidates(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self.candidates = [
            {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "candidate_id_hash": self.candidate_id_hash(
                    str(row["candidate_key"])
                ),
                "final_rank": int(row["final_rank"]),
                "fused_score": round(float(row["fused_score"]), 6),
                "vector_cosine": round(float(row["vector_cosine"]), 6),
                "admission_probe_ids": list(
                    dict.fromkeys(row.get("admission_probe_ids", []))
                ),
                "score_is_trace_only": True,
            }
            for row in rows
        ]

    def record_selection(self, records: list[dict[str, Any]]) -> None:
        selected: list[dict[str, Any]] = []
        for rank, record in enumerate(records, 1):
            record_id = str(
                record.get("record_id")
                or record.get("id")
                or record.get("memory_id")
                or ""
            )
            adjustments = record.get("retrieval_adjustments")
            if not isinstance(adjustments, dict):
                adjustments = {}
            selected.append(
                {
                    "candidate_id_hash": self.candidate_id_hash(record_id),
                    "selected_rank": rank,
                    "selection_reasons": [
                        str(reason)
                        for reason in adjustments.get("reasons", [])
                    ][:8],
                    "low_information_signals": [
                        str(reason)
                        for reason in adjustments.get(
                            "low_information_reasons", []
                        )
                    ][:8],
                    "signals_are_diagnostic_only": True,
                }
            )
        self.selection = selected

    def _payload(
        self,
        *,
        events: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        selection: list[dict[str, Any]],
        truncated: bool,
        dropped_candidates: int,
        dropped_events: int,
        dropped_selection: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "query_hash": self.query_hash,
            "plan": self.plan,
            "events": events,
            "candidates": candidates,
            "selection": selection,
            "trace_truncated": truncated,
            "dropped": {
                "candidate_count": dropped_candidates,
                "event_count": dropped_events,
                "selection_count": dropped_selection,
            },
            "privacy": {
                "raw_query_included": False,
                "raw_source_text_included": False,
                "local_paths_included": False,
                "exception_messages_included": False,
            },
            "semantics": {
                "complete_or_true_claimed": False,
                "candidate_role_signals_are_diagnostic_only": True,
                "external_score_contract_changed": False,
            },
        }

    @staticmethod
    def _size(payload: dict[str, Any]) -> int:
        return len(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def finalize(self) -> dict[str, Any]:
        events = list(self.events)
        candidates = list(self.candidates)
        selection = list(self.selection)
        dropped_candidates = 0
        dropped_events = 0
        dropped_selection = 0
        payload = self._payload(
            events=events,
            candidates=candidates,
            selection=selection,
            truncated=False,
            dropped_candidates=0,
            dropped_events=0,
            dropped_selection=0,
        )
        while self._size(payload) > self.max_bytes and candidates:
            candidates.pop()
            dropped_candidates += 1
            payload = self._payload(
                events=events,
                candidates=candidates,
                selection=selection,
                truncated=True,
                dropped_candidates=dropped_candidates,
                dropped_events=dropped_events,
                dropped_selection=dropped_selection,
            )
        while self._size(payload) > self.max_bytes and selection:
            selection.pop()
            dropped_selection += 1
            payload = self._payload(
                events=events,
                candidates=candidates,
                selection=selection,
                truncated=True,
                dropped_candidates=dropped_candidates,
                dropped_events=dropped_events,
                dropped_selection=dropped_selection,
            )
        while self._size(payload) > self.max_bytes and events:
            events.pop()
            dropped_events += 1
            payload = self._payload(
                events=events,
                candidates=candidates,
                selection=selection,
                truncated=True,
                dropped_candidates=dropped_candidates,
                dropped_events=dropped_events,
                dropped_selection=dropped_selection,
            )
        if self._size(payload) > self.max_bytes:
            payload["plan"] = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "trace_plan_truncated": True,
            }
            payload["trace_truncated"] = True
        return payload
