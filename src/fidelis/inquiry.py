"""Deterministic, bounded inquiry over Fidelis's canonical hybrid retriever.

The inquiry layer plans evidence collection and ranks retrieved records.  It
does not answer the user's question, rewrite record text, mutate the corpus, or
call a generative model.  Candidate admission remains owned by
``recall_hybrid(..., tier="zero_llm")``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence


INQUIRY_SCHEMA_VERSION = "fidelis.inquiry-result/v1"
ASK_SCHEMA_VERSION = "fidelis.structured-ask/v1"
POLICY_SCHEMA_VERSION = "fidelis.inquiry-policy/v1"
ALLOWED_OPERATIONS = ("verify_claim", "evaluate_decision", "trace_evolution")
MAX_PASSES = 3

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_VAGUE_RE = re.compile(
    r"^\s*(?:what about (?:it|that)|and that|this|that|it|why|how)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_TRACE_RE = re.compile(
    r"\b(?:trace|evolv\w*|timeline|over time|supersed\w*|progression|history)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:why|decid\w*|decision|choose|chosen|switch\w*|chang\w*|"
    r"justif\w*|successful|afterward|outcome|paused?)\b",
    re.IGNORECASE,
)
_VERIFY_RE = re.compile(
    r"\b(?:verify|confirm|claim|prove|support\w*|true|completed|preserved|"
    r"deployed|measured)\b",
    re.IGNORECASE,
)
_OUTCOME_RE = re.compile(
    r"\b(?:afterward|later|outcome|result|successful|success|justif\w*|"
    r"what happened next|given what happened)\b",
    re.IGNORECASE,
)
_CONTRADICTION_RE = re.compile(
    r"\b(?:contradict\w*|qualif\w*|reversal|reversed|counterevidence)\b",
    re.IGNORECASE,
)
_HIGH_CERTAINTY_RE = re.compile(
    r"\b(?:verify|prove|high confidence|exact|justif\w*|authoritative)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a",
    "about",
    "after",
    "afterward",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "by",
    "claim",
    "complete",
    "completed",
    "did",
    "do",
    "does",
    "evidence",
    "for",
    "from",
    "given",
    "happened",
    "how",
    "in",
    "including",
    "is",
    "it",
    "later",
    "of",
    "on",
    "our",
    "that",
    "the",
    "this",
    "to",
    "trace",
    "verify",
    "was",
    "we",
    "were",
    "what",
    "when",
    "whether",
    "why",
    "with",
}

_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "direct_support": (
        "completed",
        "measured",
        "confirmed",
        "matched",
        "preserves",
        "returns",
        "uses deterministic",
    ),
    "primary_evidence": ("receipt", "log", "manifest", "controlled benchmark"),
    "independent_support": ("independent", "separate", "following month"),
    "contradiction": (
        "contradict",
        "however",
        "also increased",
        "also delayed",
        "regression",
        "reversal",
        "qualif",
        "rising",
    ),
    "prior_state": ("before the change", "previously", "prior", "used the", "initially"),
    "trigger": (
        "trigger",
        "incident",
        "repeated",
        "exceeded",
        "timeouts",
        "mismatches",
        "amplified",
    ),
    "rationale": (
        "rationale",
        "because",
        "chosen to",
        "selected",
        "so ",
        "to prevent",
        "to remove",
    ),
    "alternatives": ("alternatives", "considered", "instead of"),
    "decision": (
        "approved",
        "changed",
        "decision",
        "owner required",
        "owner paused",
        "selected",
        "switch",
        "replaced",
    ),
    "expected_outcome": ("expected outcome", "expected benefit", "targeted"),
    "later_outcome": (
        "following month",
        "after ",
        "later ",
        "fell from",
        "caught ",
        "resulted",
    ),
    "early_state": ("early", "initial", "originally"),
    "intermediate_state": ("intermediate", "during the transition"),
    "transition": ("transition", "changed after", "replaced", "migrated"),
    "superseded_state": ("superseded", "old ", "initial "),
    "current_state": ("current", "now ", "latest"),
    "unresolved_branch": ("unresolved", "unknown", "not yet decided"),
}

_ROLE_ORDER = (
    "direct_support",
    "primary_evidence",
    "independent_support",
    "prior_state",
    "trigger",
    "rationale",
    "alternatives",
    "decision",
    "expected_outcome",
    "later_outcome",
    "contradiction",
    "early_state",
    "intermediate_state",
    "transition",
    "superseded_state",
    "current_state",
    "unresolved_branch",
)

_COMPONENTS = (
    "vector_similarity",
    "lexical_fit",
    "temporal_fit",
    "role_fit",
    "source_specificity",
    "pointer_confidence",
    "novelty",
    "source_diversity",
    "relation_to_selected",
    "redundancy_penalty",
    "supersession_penalty",
    "ambiguity_penalty",
)


@dataclass(frozen=True)
class StructuredAsk:
    raw_ask: str
    subject: tuple[str, ...]
    operation: str | None
    evidence_needed: tuple[str, ...]
    time_scope: str | None
    certainty_requirement: str
    disposition: str
    uncertainties: tuple[str, ...]
    parser: str = "deterministic-zero-llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASK_SCHEMA_VERSION,
            "raw_ask": self.raw_ask,
            "subject": list(self.subject),
            "operation": self.operation,
            "evidence_needed": list(self.evidence_needed),
            "time_scope": self.time_scope,
            "certainty_requirement": self.certainty_requirement,
            "disposition": self.disposition,
            "uncertainties": list(self.uncertainties),
            "parser": self.parser,
        }


@dataclass(frozen=True)
class InquiryPolicy:
    candidate_sources: tuple[str, ...]
    ranking_weights: Mapping[str, float]
    role_weights: Mapping[str, float]
    initial_candidate_count: int
    maximum_passes: int
    maximum_selected_records: int
    temporal_scope: str | None
    required_evidence_roles: tuple[str, ...]
    contradiction_required: bool
    source_diversity_required: bool
    stopping_conditions: tuple[str, ...]
    policy_id: str
    compiler: str = "deterministic-zero-llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "candidate_sources": list(self.candidate_sources),
            "ranking_weights": dict(self.ranking_weights),
            "role_weights": dict(self.role_weights),
            "initial_candidate_count": self.initial_candidate_count,
            "maximum_passes": self.maximum_passes,
            "maximum_selected_records": self.maximum_selected_records,
            "temporal_scope": self.temporal_scope,
            "required_evidence_roles": list(self.required_evidence_roles),
            "contradiction_required": self.contradiction_required,
            "source_diversity_required": self.source_diversity_required,
            "stopping_conditions": list(self.stopping_conditions),
            "compiler": self.compiler,
        }


Retriever = Callable[
    [str, int, tuple[str, ...], int],
    tuple[list[dict[str, Any]], str],
]


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _WORD_RE.findall(text))


def _subject_tokens(query: str) -> tuple[str, ...]:
    named = [
        token
        for token in _WORD_RE.findall(query)
        if (
            any(char.isupper() for char in token[1:])
            or token[:1].isupper()
            or token.casefold() in {"sqlite", "postgresql", "fidelis", "redis"}
        )
        and token.casefold() not in _STOPWORDS
    ]
    content = [
        token
        for token in _WORD_RE.findall(query)
        if len(token) >= 4 and token.casefold() not in _STOPWORDS
    ]
    return _dedupe(named + content)[:6]


def _operation_for(query: str) -> str | None:
    if _VAGUE_RE.fullmatch(query):
        return None
    if _TRACE_RE.search(query):
        return "trace_evolution"
    if _DECISION_RE.search(query):
        return "evaluate_decision"
    if _VERIFY_RE.search(query):
        return "verify_claim"
    return None


def _evidence_needs(query: str, operation: str | None) -> tuple[str, ...]:
    if operation == "verify_claim":
        return (
            "direct_support",
            "primary_evidence",
            "independent_support",
            "contradiction",
        )
    if operation == "evaluate_decision":
        if _OUTCOME_RE.search(query):
            return (
                "prior_state",
                "trigger",
                "rationale",
                "alternatives",
                "decision",
                "expected_outcome",
                "later_outcome",
                "contradiction",
            )
        return (
            "prior_state",
            "trigger",
            "rationale",
            "alternatives",
            "decision",
        )
    if operation == "trace_evolution":
        needs = [
            "early_state",
            "intermediate_state",
            "transition",
            "superseded_state",
            "current_state",
            "unresolved_branch",
        ]
        return tuple(needs)
    return ()


def parse_ask(
    query: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> StructuredAsk:
    """Parse one natural-language ask into a closed, uncertainty-bearing shape."""

    raw = str(query)
    local = raw.strip()
    supplied = dict(overrides or {})
    operation = supplied.get("operation", _operation_for(local))
    if operation not in (*ALLOWED_OPERATIONS, None):
        raise ValueError(f"unsupported inquiry operation: {operation}")

    subject_value = supplied.get("subject")
    if isinstance(subject_value, str):
        subject = (subject_value.strip(),) if subject_value.strip() else ()
    elif isinstance(subject_value, (list, tuple)):
        subject = _dedupe(tuple(str(item).strip() for item in subject_value))[:8]
    else:
        subject = _subject_tokens(local)

    evidence_value = supplied.get("evidence_needed")
    if isinstance(evidence_value, (list, tuple)):
        evidence_needed = _dedupe(
            tuple(str(item) for item in evidence_value if str(item) in _ROLE_ORDER)
        )
    else:
        evidence_needed = _evidence_needs(local, operation)

    time_scope = supplied.get("time_scope")
    if time_scope is None:
        if operation == "trace_evolution" or (
            operation == "evaluate_decision" and _OUTCOME_RE.search(local)
        ):
            time_scope = "before_and_after"
        elif operation == "evaluate_decision":
            time_scope = "decision_context"
        elif operation == "verify_claim":
            time_scope = "relevant_recorded_time"

    certainty = str(
        supplied.get(
            "certainty_requirement",
            "high" if _HIGH_CERTAINTY_RE.search(local) else "normal",
        )
    )
    disposition = str(
        supplied.get(
            "disposition",
            "retrieve" if operation and subject and local else "abstain",
        )
    )
    if disposition not in {"retrieve", "abstain"}:
        raise ValueError(f"unsupported inquiry disposition: {disposition}")

    uncertainties: list[str] = []
    if not local:
        uncertainties.append("raw_ask")
    if operation is None:
        uncertainties.append("operation")
    if not subject:
        uncertainties.append("subject")
    if not evidence_needed and operation is not None:
        uncertainties.append("evidence_needed")
    if uncertainties and "disposition" not in supplied:
        disposition = "abstain"

    return StructuredAsk(
        raw_ask=raw,
        subject=subject,
        operation=operation,
        evidence_needed=evidence_needed,
        time_scope=str(time_scope) if time_scope is not None else None,
        certainty_requirement=certainty,
        disposition=disposition,
        uncertainties=tuple(uncertainties),
    )


def _default_component_weights(ask: StructuredAsk) -> dict[str, float]:
    weights = {
        "vector_similarity": 0.20,
        "lexical_fit": 0.10,
        "temporal_fit": 0.08,
        "role_fit": 0.25,
        "source_specificity": 0.08,
        "pointer_confidence": 0.08,
        "novelty": 0.08,
        "source_diversity": 0.05,
        # Relation-linked admission and ranking stay owned by the canonical
        # retriever's validated, feature-gated path. Inquiry must not infer a
        # relation effect from arbitrary legacy metadata.
        "relation_to_selected": 0.0,
        "redundancy_penalty": 0.14,
        "supersession_penalty": 0.12,
        "ambiguity_penalty": 0.10,
    }
    if ask.operation == "verify_claim":
        weights.update(
            role_fit=0.28,
            source_specificity=0.12,
            pointer_confidence=0.12,
            vector_similarity=0.16,
        )
    elif ask.operation == "evaluate_decision":
        if _OUTCOME_RE.search(ask.raw_ask):
            weights.update(
                temporal_fit=0.16,
                role_fit=0.29,
                vector_similarity=0.14,
                novelty=0.10,
            )
        else:
            weights.update(
                temporal_fit=0.09,
                role_fit=0.31,
                lexical_fit=0.13,
                vector_similarity=0.16,
            )
    elif ask.operation == "trace_evolution":
        weights.update(
            temporal_fit=0.19,
            role_fit=0.28,
            novelty=0.11,
            vector_similarity=0.13,
            supersession_penalty=0.03,
        )
    return weights


def _role_weights(ask: StructuredAsk) -> dict[str, float]:
    weights = {role: 1.0 for role in ask.evidence_needed}
    if ask.operation == "verify_claim":
        weights.update(direct_support=1.0, primary_evidence=0.98, contradiction=0.88)
    elif ask.operation == "evaluate_decision":
        if _OUTCOME_RE.search(ask.raw_ask):
            weights.update(
                rationale=1.0,
                expected_outcome=0.96,
                later_outcome=1.0,
                contradiction=0.96,
                decision=0.92,
                trigger=0.78,
            )
        else:
            weights.update(
                trigger=1.0,
                rationale=1.0,
                decision=0.98,
                alternatives=0.90,
                later_outcome=0.45,
            )
    elif ask.operation == "trace_evolution":
        weights.update(
            early_state=0.94,
            transition=1.0,
            current_state=1.0,
            superseded_state=0.88,
        )
    return weights


def _required_roles(ask: StructuredAsk) -> tuple[str, ...]:
    if ask.operation == "verify_claim":
        return ("direct_support", "primary_evidence")
    if ask.operation == "evaluate_decision":
        required = ["prior_state", "decision", "rationale_or_trigger"]
        if _OUTCOME_RE.search(ask.raw_ask):
            required.append("later_outcome")
        return tuple(required)
    if ask.operation == "trace_evolution":
        return ("early_state", "transition", "current_state")
    return ()


def compile_policy(
    ask: StructuredAsk,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> InquiryPolicy:
    """Compile an inspectable ask-specific policy with bounded budgets."""

    supplied = dict(overrides or {})
    component_weights = _default_component_weights(ask)
    weight_overrides = supplied.get("ranking_weights")
    if isinstance(weight_overrides, Mapping):
        for name, value in weight_overrides.items():
            if name in _COMPONENTS:
                component_weights[name] = max(0.0, min(float(value), 2.0))

    roles = _role_weights(ask)
    role_overrides = supplied.get("role_weights")
    if isinstance(role_overrides, Mapping):
        for name, value in role_overrides.items():
            if name in _ROLE_ORDER:
                roles[name] = max(0.0, min(float(value), 2.0))

    required = supplied.get("required_evidence_roles")
    if isinstance(required, (list, tuple)):
        required_roles = _dedupe(
            tuple(
                str(role)
                for role in required
                if str(role) in (*_ROLE_ORDER, "rationale_or_trigger")
            )
        )
    else:
        required_roles = _required_roles(ask)

    maximum_passes = max(1, min(int(supplied.get("maximum_passes", MAX_PASSES)), MAX_PASSES))
    maximum_selected = max(
        1,
        min(int(supplied.get("maximum_selected_records", 8)), 12),
    )
    initial_count = max(
        5,
        min(int(supplied.get("initial_candidate_count", 36)), 100),
    )
    contradiction_required = bool(
        supplied.get(
            "contradiction_required",
            ask.operation == "verify_claim"
            or (
                ask.operation == "evaluate_decision"
                and bool(_OUTCOME_RE.search(ask.raw_ask) or _CONTRADICTION_RE.search(ask.raw_ask))
            ),
        )
    )
    diversity_required = bool(
        supplied.get(
            "source_diversity_required",
            ask.certainty_requirement == "high",
        )
    )
    sources = supplied.get("candidate_sources")
    if isinstance(sources, (list, tuple)):
        candidate_sources = _dedupe(tuple(str(source) for source in sources))
    else:
        candidate_sources = ()

    identity = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "compiler": "deterministic-zero-llm",
        "ask": ask.to_dict(),
        "candidate_sources": candidate_sources,
        "weights": component_weights,
        "role_weights": roles,
        "initial_candidate_count": initial_count,
        "passes": maximum_passes,
        "selected": maximum_selected,
        "temporal_scope": ask.time_scope,
        "required": required_roles,
        "contradiction": contradiction_required,
        "diversity": diversity_required,
        "stopping_conditions": (
            "structural_sufficiency",
            "maximum_passes",
            "maximum_selected_records",
            "unresolved_ask",
        ),
    }
    policy_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return InquiryPolicy(
        candidate_sources=candidate_sources,
        ranking_weights=component_weights,
        role_weights=roles,
        initial_candidate_count=initial_count,
        maximum_passes=maximum_passes,
        maximum_selected_records=maximum_selected,
        temporal_scope=ask.time_scope,
        required_evidence_roles=required_roles,
        contradiction_required=contradiction_required,
        source_diversity_required=diversity_required,
        stopping_conditions=(
            "structural_sufficiency",
            "maximum_passes",
            "maximum_selected_records",
            "unresolved_ask",
        ),
        policy_id=policy_id,
    )


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("text") or candidate.get("memory") or "")


def _metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw = candidate.get("metadata")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _record_id(candidate: Mapping[str, Any], index: int = 0) -> str:
    del index
    metadata = _metadata(candidate)
    for value in (
        candidate.get("record_id"),
        candidate.get("id"),
        metadata.get("record_id"),
        metadata.get("id"),
    ):
        if value:
            return str(value)
    digest = hashlib.sha256(_candidate_text(candidate).encode("utf-8")).hexdigest()[:16]
    return f"content-{digest}"


def _source_pointer(candidate: Mapping[str, Any]) -> str | None:
    metadata = _metadata(candidate)
    for value in (
        candidate.get("source_pointer"),
        metadata.get("source_pointer"),
        metadata.get("path"),
        metadata.get("url"),
    ):
        if value:
            return str(value)
    return None


def _candidate_source(candidate: Mapping[str, Any]) -> str:
    metadata = _metadata(candidate)
    return str(
        metadata.get("source")
        or metadata.get("source_kind")
        or candidate.get("source")
        or "unknown"
    )


def _record_sources(record: Mapping[str, Any]) -> set[str]:
    sources = {_candidate_source(record)}
    underlying = record.get("underlying_records")
    if isinstance(underlying, list):
        sources.update(
            str(item.get("source") or "unknown")
            for item in underlying
            if isinstance(item, Mapping)
        )
    return {source for source in sources if source != "unknown"}


def evidence_roles(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return declared roles plus conservative deterministic text cues."""

    metadata = _metadata(candidate)
    declared = (
        candidate.get("evidence_roles")
        or candidate.get("evidence_role")
        or metadata.get("evidence_roles")
        or metadata.get("evidence_role")
    )
    if isinstance(declared, str):
        roles = [declared]
    elif isinstance(declared, (list, tuple)):
        roles = [str(role) for role in declared]
    else:
        roles = []
    text = _candidate_text(candidate).casefold()
    for role, patterns in _ROLE_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            roles.append(role)
    if metadata.get("primary") is True:
        roles.append("primary_evidence")
    supersession = candidate.get("supersession")
    if isinstance(supersession, Mapping) and str(supersession.get("status", "")).upper() in {
        "SUPERSEDED",
        "RETRACTED",
    }:
        roles.append("superseded_state")
    return _dedupe(tuple(role for role in roles if role in _ROLE_ORDER))


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _vector_similarity(candidate: Mapping[str, Any]) -> float:
    try:
        score = float(candidate.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    if score < 0.0:
        score = (score + 1.0) / 2.0
    return max(0.0, min(score, 1.0))


def _temporal_fit(roles: set[str], temporal_scope: str | None) -> float:
    if temporal_scope == "before_and_after":
        if roles & {
            "prior_state",
            "early_state",
            "transition",
            "later_outcome",
            "current_state",
            "superseded_state",
        }:
            return 1.0
        return 0.25
    if temporal_scope == "decision_context":
        return 1.0 if roles & {"prior_state", "trigger", "rationale", "decision"} else 0.35
    return 0.65


def _source_specificity(candidate: Mapping[str, Any]) -> float:
    metadata = _metadata(candidate)
    if metadata.get("primary") is True:
        return 1.0
    source = _candidate_source(candidate).casefold()
    if any(
        marker in source
        for marker in (
            "receipt",
            "log",
            "benchmark",
            "metrics",
            "config",
            "manifest",
            "review",
            "contract",
        )
    ):
        return 0.9
    if source in {"unknown", "chat", "reference", "summary", "glossary"}:
        return 0.25
    return 0.65


def _pointer_confidence(candidate: Mapping[str, Any], index: int) -> float:
    has_id = bool(_record_id(candidate, index))
    has_pointer = bool(_source_pointer(candidate))
    if has_id and has_pointer:
        return 1.0
    if has_id:
        return 0.65
    return 0.0


def _relation_score(
    candidate: Mapping[str, Any],
    selected_ids: set[str],
) -> float:
    metadata = _metadata(candidate)
    values: list[str] = []
    for field in ("related_record_ids", "relation_targets", "target_record_id"):
        value = metadata.get(field)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
    return 1.0 if selected_ids & set(values) else (0.2 if selected_ids else 0.0)


def score_candidate(
    candidate: Mapping[str, Any],
    *,
    ask: StructuredAsk,
    policy: InquiryPolicy,
    target_roles: Sequence[str],
    selected: Sequence[Mapping[str, Any]] = (),
    index: int = 0,
) -> dict[str, Any]:
    """Score one candidate with inspectable deterministic components."""

    text = _candidate_text(candidate)
    text_tokens = _tokens(text)
    ask_tokens = tuple(
        token for token in _tokens(" ".join((*ask.subject, ask.raw_ask))) if token not in _STOPWORDS
    )
    roles = set(evidence_roles(candidate))
    target = set(target_roles)
    matched = roles & target
    selected_texts = [_tokens(str(record.get("text") or "")) for record in selected]
    overlap = max((_jaccard(text_tokens, other) for other in selected_texts), default=0.0)
    selected_sources = {_candidate_source(record) for record in selected}
    selected_ids = {_record_id(record, i) for i, record in enumerate(selected)}
    role_weight = max((float(policy.role_weights.get(role, 1.0)) for role in matched), default=0.0)
    supersession = candidate.get("supersession")
    superseded = (
        isinstance(supersession, Mapping)
        and str(supersession.get("status", "")).upper() in {"SUPERSEDED", "RETRACTED"}
    )
    components = {
        "vector_similarity": _vector_similarity(candidate),
        "lexical_fit": _jaccard(ask_tokens, text_tokens),
        "temporal_fit": _temporal_fit(roles, policy.temporal_scope),
        "role_fit": min(1.0, role_weight),
        "source_specificity": _source_specificity(candidate),
        "pointer_confidence": _pointer_confidence(candidate, index),
        "novelty": max(0.0, 1.0 - overlap),
        "source_diversity": (
            1.0 if _candidate_source(candidate) not in selected_sources else 0.25
        ),
        "relation_to_selected": _relation_score(candidate, selected_ids),
        "redundancy_penalty": overlap,
        "supersession_penalty": (
            0.0 if "superseded_state" in target else (1.0 if superseded else 0.0)
        ),
        "ambiguity_penalty": 0.0 if roles else 0.65,
    }
    positive = sum(
        components[name] * float(policy.ranking_weights.get(name, 0.0))
        for name in _COMPONENTS
        if not name.endswith("_penalty")
    )
    penalties = sum(
        components[name] * float(policy.ranking_weights.get(name, 0.0))
        for name in _COMPONENTS
        if name.endswith("_penalty")
    )
    return {
        "record_id": _record_id(candidate, index),
        "final_score": round(positive - penalties, 6),
        "components": {name: round(float(value), 6) for name, value in components.items()},
        "evidence_roles": list(evidence_roles(candidate)),
    }


def _expanded_required(required: Sequence[str]) -> set[str]:
    expanded: set[str] = set()
    for role in required:
        if role == "rationale_or_trigger":
            expanded.update(("rationale", "trigger"))
        else:
            expanded.add(role)
    return expanded


def _target_roles(
    ask: StructuredAsk,
    policy: InquiryPolicy,
    pass_number: int,
    missing_roles: Sequence[str],
) -> tuple[str, ...]:
    missing = set(missing_roles)
    if pass_number == 1:
        if ask.operation == "verify_claim":
            preferred = ("direct_support", "primary_evidence")
        elif ask.operation == "evaluate_decision":
            preferred = (
                "prior_state",
                "trigger",
                "rationale",
                "alternatives",
                "decision",
                "expected_outcome",
            )
        elif ask.operation == "trace_evolution":
            preferred = ("early_state", "current_state")
        else:
            preferred = ()
    elif pass_number == 2:
        if ask.operation == "verify_claim":
            preferred = ("contradiction", "independent_support")
        elif ask.operation == "evaluate_decision" and _OUTCOME_RE.search(ask.raw_ask):
            preferred = ("later_outcome", "contradiction")
        elif ask.operation == "trace_evolution":
            preferred = ("transition", "intermediate_state", "superseded_state")
        else:
            preferred = tuple(missing_roles)
    else:
        preferred = tuple(missing_roles)
        if policy.contradiction_required:
            preferred = (*preferred, "contradiction")
    roles = [role for role in preferred if role in missing or role == "contradiction"]
    if not roles:
        roles = list(missing)
    if not roles:
        roles = list(_expanded_required(policy.required_evidence_roles))
    return _dedupe(tuple(roles))


def _pass_query(ask: StructuredAsk, pass_number: int, roles: Sequence[str]) -> str:
    if pass_number == 1:
        return ask.raw_ask.strip()
    subject = " ".join(ask.subject) or "unresolved subject"
    role_text = " ".join(role.replace("_", " ") for role in roles)
    return f"{subject} {role_text} evidence".strip()


def _required_satisfied(required: str, covered: set[str]) -> bool:
    if required == "rationale_or_trigger":
        return bool({"rationale", "trigger"} & covered)
    return required in covered


def _record_time(record: Mapping[str, Any]) -> str | None:
    metadata = _metadata(record)
    for value in (
        record.get("recorded_at"),
        record.get("created_at"),
        metadata.get("recorded_at"),
        metadata.get("created_at"),
        metadata.get("timestamp"),
    ):
        if value:
            return str(value)
    return None


def _parse_record_time(value: str) -> float | None:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _chronology_valid(
    records: Sequence[Mapping[str, Any]],
) -> tuple[bool | None, str]:
    by_role: dict[str, list[float]] = {}
    for record in records:
        timestamp = _record_time(record)
        if not timestamp:
            continue
        parsed_timestamp = _parse_record_time(timestamp)
        if parsed_timestamp is None:
            return None, "not_evaluable_invalid_timestamp"
        for role in evidence_roles(record):
            by_role.setdefault(role, []).append(parsed_timestamp)
    comparisons = (
        ("prior_state", "later_outcome"),
        ("early_state", "current_state"),
        ("trigger", "decision"),
    )
    checked = 0
    for earlier, later in comparisons:
        if by_role.get(earlier) and by_role.get(later):
            checked += 1
            if min(by_role[earlier]) > max(by_role[later]):
                return False, "timestamp_conflict"
    if checked:
        return True, "timestamps"
    return None, "not_evaluable_no_comparable_timestamps"


def _structural_sufficiency(
    ask: StructuredAsk,
    policy: InquiryPolicy,
    selected: Sequence[Mapping[str, Any]],
    *,
    contradiction_searched: bool,
    source_diversity_attempted: bool,
) -> dict[str, Any]:
    covered = {role for record in selected for role in evidence_roles(record)}
    required_missing = [
        role
        for role in policy.required_evidence_roles
        if not _required_satisfied(role, covered)
    ]
    source_pointer_present = any(
        role in {"direct_support", "primary_evidence"}
        and bool(_source_pointer(record))
        for record in selected
        for role in evidence_roles(record)
    )
    chronology_valid, chronology_basis = _chronology_valid(selected)
    checks: dict[str, bool] = {"required_roles": not required_missing}
    if chronology_valid is not None:
        checks["chronology_valid"] = chronology_valid
    if ask.operation == "verify_claim":
        checks.update(
            direct_evidence_found="direct_support" in covered,
            source_pointer_present=source_pointer_present,
            contradiction_searched=(
                contradiction_searched if policy.contradiction_required else True
            ),
            source_diversity_found=(
                len(
                    {
                        source
                        for record in selected
                        for source in _record_sources(record)
                    }
                )
                >= 2
                if policy.source_diversity_required
                else True
            ),
        )
    elif ask.operation == "evaluate_decision":
        checks.update(
            prior_state_found="prior_state" in covered,
            decision_found="decision" in covered,
            rationale_or_trigger_found=bool({"rationale", "trigger"} & covered),
            later_outcome_found=(
                "later_outcome" in covered if _OUTCOME_RE.search(ask.raw_ask) else True
            ),
            contradiction_searched=(
                contradiction_searched if policy.contradiction_required else True
            ),
        )
    elif ask.operation == "trace_evolution":
        state_count = len(
            covered
            & {
                "early_state",
                "intermediate_state",
                "superseded_state",
                "current_state",
            }
        )
        checks.update(
            two_temporal_states_found=state_count >= 2,
            transition_found="transition" in covered,
            current_or_latest_found="current_state" in covered,
            superseded_states_preserved=all(
                _candidate_text(record)
                for record in selected
                if "superseded_state" in evidence_roles(record)
            ),
        )
    return {
        "sufficient": bool(checks) and all(checks.values()),
        "checks": checks,
        "required_missing": required_missing,
        "chronology_basis": chronology_basis,
    }


def _result_record(
    candidate: Mapping[str, Any],
    ranking: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    record_id = _record_id(candidate, index)
    pointer = _source_pointer(candidate)
    text = _candidate_text(candidate)
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    result = {
        "record_id": record_id,
        "text": text,
        "source_pointer": pointer,
        "source": _candidate_source(candidate),
        "evidence_roles": list(evidence_roles(candidate)),
        "score": ranking["final_score"],
        "score_components": dict(ranking["components"]),
        "text_sha256": text_sha256,
        "evidence_receipt": {
            "schema_version": "fidelis.inquiry-evidence-receipt/v1",
            "record_id": record_id,
            "source_pointer": pointer,
            "text_sha256": text_sha256,
        },
        "underlying_records": [
            {
                "record_id": record_id,
                "source_pointer": pointer,
                "source": _candidate_source(candidate),
                "evidence_roles": list(evidence_roles(candidate)),
            }
        ],
    }
    for field in ("metadata", "created_at", "recorded_at", "supersession"):
        if candidate.get(field) is not None:
            result[field] = candidate[field]
    return result


def _guidance(
    *,
    covered: Sequence[str],
    missing: Sequence[str],
    stop_reason: str,
) -> str:
    found_text = ", ".join(role.replace("_", " ") for role in covered) or "no required role"
    missing_text = ", ".join(role.replace("_", " ") for role in missing) or "no required role"
    if stop_reason == "structurally_sufficient":
        return f"Found {found_text}, so retrieval stopped because structural sufficiency was reached."
    if stop_reason == "unresolved_ask":
        return "Found no resolved operation or subject, so retrieval stopped."
    return (
        f"Found {found_text}, but missing {missing_text}, so retrieval stopped "
        "because the pass budget was exhausted."
    )


def run_inquiry(
    memory: Any,
    cfg: Mapping[str, Any],
    query: str,
    *,
    policy: str | Mapping[str, Any] = "auto",
    include_trace: bool = True,
    overrides: Mapping[str, Any] | None = None,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    """Run one bounded inquiry against the canonical hybrid candidate owner."""

    if policy != "auto" and not isinstance(policy, Mapping):
        raise ValueError("policy must be 'auto' or a mapping of constrained overrides")
    supplied = dict(overrides or {})
    if isinstance(policy, Mapping):
        supplied.update(policy)
    ask = parse_ask(query, overrides=supplied)
    compiled = compile_policy(ask, overrides=supplied)

    if retriever is None:
        from fidelis.recall_hybrid import recall_hybrid

        def canonical_retriever(
            pass_query: str,
            limit: int,
            target_roles: tuple[str, ...],
            pass_number: int,
        ) -> tuple[list[dict[str, Any]], str]:
            del pass_number
            return recall_hybrid(
                memory,
                pass_query,
                user_id=str(cfg["user_id"]),
                cfg=dict(cfg),
                limit=limit,
                tier="zero_llm",
                top_k=limit,
                evidence_needs=target_roles,
            )

        retriever = canonical_retriever

    initial_missing = list(ask.evidence_needed)
    if ask.disposition == "abstain":
        result = {
            "schema_version": INQUIRY_SCHEMA_VERSION,
            "ask": ask.to_dict(),
            "policy": compiled.to_dict(),
            "records": [],
            "covered_roles": [],
            "missing_roles": initial_missing,
            "structural_sufficiency": {
                "sufficient": False,
                "checks": {},
                "required_missing": list(compiled.required_evidence_roles),
                "chronology_basis": "not_evaluated",
            },
            "disposition": "abstain",
            "stop_reason": "unresolved_ask",
            "guidance": _guidance(
                covered=(),
                missing=initial_missing,
                stop_reason="unresolved_ask",
            ),
            "runtime": {
                "generative_llm_calls": 0,
                "candidate_owner": "recall_hybrid",
                "candidate_tier": "zero_llm",
                "corpus_mutations": 0,
            },
        }
        if include_trace:
            result["trace"] = {"passes": [], "records_inspected": 0}
        return result

    selected: list[dict[str, Any]] = []
    selected_by_hash: dict[str, dict[str, Any]] = {}
    passes: list[dict[str, Any]] = []
    expanded_required = _expanded_required(compiled.required_evidence_roles)
    needed_roles = _dedupe(
        (
            *ask.evidence_needed,
            *(
                role
                for role in _ROLE_ORDER
                if role in expanded_required
            ),
        )
    )
    missing = list(needed_roles)
    contradiction_searched = False
    diversity_attempted = False
    inspected = 0
    stop_reason = "budget_exhausted"
    sufficiency: dict[str, Any] = {
        "sufficient": False,
        "checks": {},
        "required_missing": list(compiled.required_evidence_roles),
        "chronology_basis": "not_evaluated",
    }

    for pass_number in range(1, compiled.maximum_passes + 1):
        missing_before = list(missing)
        targets = _target_roles(ask, compiled, pass_number, missing_before)
        pass_query = _pass_query(ask, pass_number, targets)
        contradiction_searched = contradiction_searched or "contradiction" in targets
        diversity_attempted = diversity_attempted or (
            compiled.source_diversity_required
            and ("independent_support" in targets or pass_number > 1)
        )
        candidates, method = retriever(
            pass_query,
            compiled.initial_candidate_count,
            targets,
            pass_number,
        )
        candidates = list(candidates)
        inspected += len(candidates)
        admitted_candidates = [
            candidate
            for candidate in candidates
            if (
                not compiled.candidate_sources
                or _candidate_source(candidate) in compiled.candidate_sources
            )
        ]
        ranked = [
            (
                candidate,
                score_candidate(
                    candidate,
                    ask=ask,
                    policy=compiled,
                    target_roles=targets,
                    selected=selected,
                    index=index,
                ),
                index,
            )
            for index, candidate in enumerate(admitted_candidates)
            if _candidate_text(candidate)
        ]
        ranked.sort(
            key=lambda item: (
                -float(item[1]["final_score"]),
                str(item[1]["record_id"]),
            )
        )
        selected_this_pass: list[str] = []
        covered_before = {
            role for record in selected for role in evidence_roles(record)
        }
        for candidate, ranking, index in ranked:
            text = _candidate_text(candidate)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            record_id = _record_id(candidate, index)
            pointer = _source_pointer(candidate)
            if digest in selected_by_hash:
                group = selected_by_hash[digest]["underlying_records"]
                if not any(
                    item["record_id"] == record_id
                    and item.get("source_pointer") == pointer
                    for item in group
                ):
                    group.append(
                        {
                            "record_id": record_id,
                            "source_pointer": pointer,
                            "source": _candidate_source(candidate),
                            "evidence_roles": list(evidence_roles(candidate)),
                        }
                    )
                union_roles = {
                    role
                    for item in group
                    for role in item.get("evidence_roles", [])
                }
                selected_by_hash[digest]["evidence_roles"] = [
                    role for role in _ROLE_ORDER if role in union_roles
                ]
                continue
            roles = set(evidence_roles(candidate))
            covered_now = {
                role for record in selected for role in evidence_roles(record)
            }
            role_contributes = (
                roles
                & set(targets)
                & (set(needed_roles) - covered_now)
            )
            selected_sources_now = {
                source
                for record in selected
                for source in _record_sources(record)
            }
            source = _candidate_source(candidate)
            diversity_contributes = (
                compiled.source_diversity_required
                and len(selected_sources_now) < 2
                and source != "unknown"
                and source not in selected_sources_now
                and bool(roles & set(targets))
            )
            if not role_contributes and not diversity_contributes:
                continue
            if len(selected) >= compiled.maximum_selected_records:
                stop_reason = "selection_budget_exhausted"
                break
            record = _result_record(candidate, ranking, index)
            selected.append(record)
            selected_by_hash[digest] = record
            selected_this_pass.append(record_id)

        covered = {
            role
            for record in selected
            for role in evidence_roles(record)
            if role in needed_roles
        }
        missing = [role for role in needed_roles if role not in covered]
        sufficiency = _structural_sufficiency(
            ask,
            compiled,
            selected,
            contradiction_searched=contradiction_searched,
            source_diversity_attempted=diversity_attempted,
        )
        pass_stop_reason = None
        if sufficiency["sufficient"]:
            stop_reason = "structurally_sufficient"
            pass_stop_reason = stop_reason
        elif len(selected) >= compiled.maximum_selected_records:
            stop_reason = "selection_budget_exhausted"
            pass_stop_reason = stop_reason
        elif pass_number == compiled.maximum_passes:
            stop_reason = "budget_exhausted"
            pass_stop_reason = stop_reason
        passes.append(
            {
                "pass_number": pass_number,
                "target_evidence_roles": list(targets),
                "query": pass_query,
                "filters": {
                    "temporal_scope": compiled.temporal_scope,
                    "candidate_sources": list(compiled.candidate_sources),
                },
                "retrieval_method": method,
                "candidate_ids": [
                    _record_id(candidate, index)
                    for index, candidate in enumerate(candidates)
                ],
                "ranking": [ranking for _, ranking, _ in ranked],
                "selected_ids": selected_this_pass,
                "missing_roles_before": missing_before,
                "missing_roles_after": list(missing),
                "covered_roles_before": sorted(covered_before),
                "stop_reason": pass_stop_reason,
            }
        )
        if pass_stop_reason:
            break

    covered_roles = [
        role
        for role in _ROLE_ORDER
        if any(role in evidence_roles(record) for record in selected)
    ]
    core_supported = False
    covered_set = set(covered_roles)
    if ask.operation == "verify_claim":
        core_supported = "direct_support" in covered_set
    elif ask.operation == "evaluate_decision":
        core_supported = (
            "decision" in covered_set
            and bool({"rationale", "trigger"} & covered_set)
        )
    elif ask.operation == "trace_evolution":
        core_supported = (
            "transition" in covered_set
            and len(covered_set & {"early_state", "current_state", "superseded_state"}) >= 2
        )
    disposition = (
        "supported_evidence"
        if sufficiency["sufficient"]
        else "partial_evidence"
        if core_supported
        else "abstain"
    )
    result = {
        "schema_version": INQUIRY_SCHEMA_VERSION,
        "ask": ask.to_dict(),
        "policy": compiled.to_dict(),
        "records": selected,
        "covered_roles": covered_roles,
        "missing_roles": list(missing),
        "structural_sufficiency": sufficiency,
        "disposition": disposition,
        "stop_reason": stop_reason,
        "guidance": _guidance(
            covered=covered_roles,
            missing=missing,
            stop_reason=stop_reason,
        ),
        "runtime": {
            "generative_llm_calls": 0,
            "candidate_owner": "recall_hybrid",
            "candidate_tier": "zero_llm",
            "corpus_mutations": 0,
        },
    }
    if include_trace:
        result["trace"] = {
            "passes": passes,
            "records_inspected": inspected,
            "records_returned": len(selected),
        }
    return result


def _server_url(server_url: str | None = None) -> str:
    if server_url:
        return server_url.rstrip("/")
    explicit = os.environ.get("FIDELIS_SERVER_URL")
    if explicit:
        return explicit.rstrip("/")
    port = int(
        os.environ.get("FIDELIS_PORT")
        or os.environ.get("COGITO_PORT", "19420")
    )
    return f"http://127.0.0.1:{port}"


def inquire(
    *,
    query: str,
    policy: str | Mapping[str, Any] = "auto",
    include_trace: bool = False,
    overrides: Mapping[str, Any] | None = None,
    server_url: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Submit one inquiry to the local Fidelis service.

    Ordinary retrieval APIs are unchanged; inquiry is opt-in.
    """

    body = json.dumps(
        {
            "query": query,
            "policy": policy,
            "include_trace": include_trace,
            "overrides": dict(overrides or {}),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{_server_url(server_url)}/inquire",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fidelis inquiry failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Fidelis server unavailable at {_server_url(server_url)}") from exc


__all__ = [
    "ALLOWED_OPERATIONS",
    "ASK_SCHEMA_VERSION",
    "INQUIRY_SCHEMA_VERSION",
    "MAX_PASSES",
    "POLICY_SCHEMA_VERSION",
    "InquiryPolicy",
    "StructuredAsk",
    "compile_policy",
    "evidence_roles",
    "inquire",
    "parse_ask",
    "run_inquiry",
    "score_candidate",
]
