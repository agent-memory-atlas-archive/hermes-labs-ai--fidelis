"""Cogito Hermeneutics V0.4 integrated read-time controls.

This module is deliberately additive. It never mutates the memory corpus and
can be bypassed by using Fidelis's existing ``/recall_hybrid`` endpoint.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from fidelis.relation_envelope import (
    SCHEMA_VERSION as RELATION_SCHEMA_VERSION,
    expose_linked_raw_evidence,
    relation_envelopes_enabled,
)


ENGINE_NAME = "cogito-hermeneutics"
ENGINE_VERSION = "0.4.0"
AUTOMATIC_TIMEOUT_MS = 100
RETRIEVAL_BRIEF_VERSION = "fidelis-retrieval-brief/v1"
_BRIEF_SESSION_LIMIT = 2048
_BRIEF_SESSIONS: OrderedDict[str, None] = OrderedDict()
_BRIEF_LOCK = threading.Lock()

_WORD_RE = re.compile(r"[a-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b")
_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|good morning|good afternoon)[.!?\s]*$",
    re.IGNORECASE,
)
_UNRELATED_ACTION_RE = re.compile(
    r"^\s*(?:write|draft|refactor|create|build|implement|edit|translate|"
    r"generate|compose|fix|run|test|review|explain|summarize)\b",
    re.IGNORECASE,
)
_MEMORY_INTENT_RE = re.compile(
    r"\b(?:memory|memories|remember|recall|awareness|orient(?:ation)?|"
    r"what (?:have|had) we|what we have|have been doing|last few days|"
    r"recent work|work lately|catch (?:me )?up|previous (?:chat|task|work)|"
    r"earlier (?:chat|task|work)|we discussed|we decided|side[\s-]?chat|"
    r"lost (?:chat|task|thread)|assistant (?:message|output)|last output|"
    r"conversation log|task log|post[- ]v?\d+(?:\.\d+)? roadmap)\b",
    re.IGNORECASE,
)
_PAST_WORK_RE = re.compile(
    r"\b(?:we|our|my)\b.*\b(?:work(?:ed|ing)?|project|task|thread|chat|"
    r"conversation|release|implementation|integration|decision|plan|"
    r"status|history|timeline)\b|"
    r"\b(?:past|previous|earlier|recent|last)\b.*\b(?:work|project|task|"
    r"thread|chat|conversation|release|implementation|integration|"
    r"decision|plan|status)\b|"
    r"\b(?:current|latest|prior|previous)\s+(?:status|state|version)\s+"
    r"of\s+(?:our|my|the)\b|"
    r"\b(?:timeline|history)\s+of\s+(?:our|my)\b|"
    r"\b(?:project|repository|repo|service|integration|release|deployment|"
    r"pipeline)\s+[A-Za-z0-9_.-]{3,}\b",
    re.IGNORECASE,
)
_CHANGING_STATE_RE = re.compile(
    r"\b(?:what changed|what has changed|current(?:ly)?|latest|now|"
    r"before|previously|prior|then and now|current versus prior|"
    r"current vs prior|since (?:our|the|last)|over time)\b",
    re.IGNORECASE,
)
_DECISION_RATIONALE_RE = re.compile(
    r"(?:\bwhy\b.*\b(?:decid\w*|cho(?:ose|se|sen)|select\w*|approv\w*|"
    r"reject\w*)\b|"
    r"\b(?:decision|choice)\b.*\b(?:why|reason|rationale)\b|"
    r"\b(?:what did we decide|which decision)\b.*\b(?:and why|rationale)\b)",
    re.IGNORECASE | re.DOTALL,
)
_HISTORICAL_RE = re.compile(
    r"\b(?:timeline|chronolog\w*|histor\w*|as[- ]of|over time|previous|earlier)\b",
    re.IGNORECASE,
)
_COMPARATIVE_NEED_RE = re.compile(
    r"(?:\bwhat changed\b.*\b(?:before|previously|prior)\b|"
    r"\b(?:compare|comparison)\b.*\b(?:before|prior|previous)\b.*"
    r"\b(?:now|current|after)\b|"
    r"\bchange(?:d)?\s+from\b.*\bto\b)",
    re.IGNORECASE | re.DOTALL,
)
_SUBJECT_STOPWORDS = {
    "about", "and", "before", "changed", "changes", "compare",
    "comparison", "current", "currently", "did", "earlier", "from",
    "history", "integration",
    "implementation", "now", "pipeline", "previous", "previously", "prior",
    "project", "service", "setup", "system", "the", "then", "timeline",
    "use", "used", "what", "when", "where", "which", "with", "work",
}
_VAGUE_SUBJECT_RE = re.compile(
    r"\b(?:it|that|this|the (?:[A-Za-z0-9_.-]+\s+)?"
    r"(?:integration|project|system|setup|work|implementation|pipeline|"
    r"service))\b",
    re.IGNORECASE,
)
_SESSION_OR_PROVENANCE_RE = re.compile(
    r"\b(?:session|chat|task|thread|source|provenance|where did|which file)\b",
    re.IGNORECASE,
)
_REFERENTIAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:what|how)\s+about\s+"
    r"(?:it|that|this|that\s+work|that\s+thing)\??\s*$|"
    r"^\s*(?:and\s+)?what\s+(?:about|before)\??\s*$",
    re.IGNORECASE,
)
_DOMAIN_PHRASES = (
    "cogito hermeneutics",
    "cogito-hermeneutics",
    "cogito ergo",
    "cogito-ergo",
    "fidelis",
    "cogito",
    "hermes",
    "agent gorgon",
    "linguistic attractors",
)
_LOW_INFORMATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("raw_tool_output", re.compile(r"^\s*(?:\[[^\]]+\]\s*)*\[?tool_(?:result|call)\b", re.I)),
    ("generic_session_state", re.compile(r"^\s*(?:user|assistant)?\s*:?\s*(?:task|session) (?:is|was) (?:active|running|complete)\b", re.I)),
    ("generic_progress", re.compile(r"^\s*(?:working on it|still working|done|completed|passed|failed)\W*$", re.I)),
    ("bare_path", re.compile(r"^\s*(?:/|~[/\\])\S+\s*$")),
    ("placeholder", re.compile(r"^\s*(?:todo|tbd|n/?a|unknown|none|null|placeholder)\W*$", re.I)),
)
_CLEAR_NOISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "extraction_prompt",
        re.compile(
            r"^\s*(?:(?:output|return|respond with)\s+(?:only\s+)?"
            r"(?:bullet points?|json|facts?|a summary)(?:\s+from\s+(?:the\s+)?"
            r"(?:following\s+)?(?:source|text|input))?|"
            r"(?:extract|summari[sz]e|list)\s+(?:the\s+)?(?:key\s+)?"
            r"(?:facts?|points?|content)(?:\s+from\s+(?:the\s+)?"
            r"(?:following\s+)?(?:source|text|input|text above))?)"
            r"(?:\s+and\s+return\s+(?:json|bullet points?)\s+only)?[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "prompt_scaffolding",
        re.compile(
            r"^\s*(?:system|assistant|user)\s+(?:prompt|instructions?)\s*:"
            r"|^\s*you are (?:a |an )?(?:helpful )?(?:ai |language-model )?"
            r"assistant\b.{0,120}$"
            r"|^\s*your task is to (?:extract|summari[sz]e|output|return|list)\b"
            r".{0,160}$"
            r"|^\s*follow these instructions\b.{0,160}$",
            re.I | re.S,
        ),
    ),
    (
        "failed_or_empty_summary",
        re.compile(
            r"^\s*(?:summary\s+)?(?:generation\s+)?failed\b"
            r"|^\s*(?:failed|unable)\s+to\s+summari[sz]e\b"
            r"|^\s*no\s+summary\s+(?:generated|available)\W*$",
            re.I,
        ),
    ),
    (
        "placeholder_or_redacted",
        re.compile(
            r"^\s*(?:\[?\s*redacted\s*\]?|<\s*(?:placeholder|redacted)\s*>|"
            r"\[?\s*placeholder\s*\]?)\W*$",
            re.I,
        ),
    ),
    (
        "processing_boilerplate",
        re.compile(
            r"^\s*(?:processing|analy[sz]ing|extracting)\s+"
            r"(?:input|request|source|text)\b.{0,80}$",
            re.I,
        ),
    ),
    (
        "malformed_fragment",
        re.compile(
            r"^\s*(?:\{\s*\"[^\"]+\"\s*:|\[\s*\{?)\s*$"
            r"|^\s*(?:\.\.\.|---+|===+)\s*$",
            re.S,
        ),
    ),
)

_STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "been",
    "by", "did", "do", "does", "for", "from", "give", "had", "has", "have",
    "how", "i", "in", "is", "it", "last", "me", "my", "of", "on", "or",
    "our", "recent", "show", "status", "tell", "that", "the", "these", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "with", "work", "you", "your",
}

_FEATURE_ENV = {
    "retrieval_quality": (
        "COGITO_HERMENEUTICS_RETRIEVAL_QUALITY",
        "COGITO_V04_RETRIEVAL_QUALITY",
    ),
    "exact_text_dedup": (
        "COGITO_HERMENEUTICS_EXACT_TEXT_DEDUP",
        "COGITO_V04_EXACT_TEXT_DEDUP",
    ),
    "low_information_downweight": (
        "COGITO_HERMENEUTICS_LOW_INFORMATION",
        "COGITO_V04_LOW_INFORMATION",
    ),
    "valid_negative_abstention": (
        "COGITO_HERMENEUTICS_ABSTENTION",
        "COGITO_V04_ABSTENTION",
    ),
    "clear_noise_control": (
        "COGITO_HERMENEUTICS_CLEAR_NOISE",
        "COGITO_V04_CLEAR_NOISE",
    ),
    "current_truth_precedence": (
        "COGITO_HERMENEUTICS_CURRENT_TRUTH",
        "COGITO_V04_CURRENT_TRUTH",
    ),
}


def _env_bool(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, on/off")


def load_retrieval_brief(session_id: str | None) -> dict[str, Any]:
    """Load the compact brief once per bounded process-local host session."""

    key = (session_id or "host-process").strip()[:128] or "host-process"
    with _BRIEF_LOCK:
        loaded_newly = key not in _BRIEF_SESSIONS
        if loaded_newly:
            _BRIEF_SESSIONS[key] = None
            while len(_BRIEF_SESSIONS) > _BRIEF_SESSION_LIMIT:
                _BRIEF_SESSIONS.popitem(last=False)
        else:
            _BRIEF_SESSIONS.move_to_end(key)
    result: dict[str, Any] = {
        "version": RETRIEVAL_BRIEF_VERSION,
        "loaded_newly": loaded_newly,
        "session_id": key,
    }
    if loaded_newly:
        result["brief"] = [
            "Name the specific entity or referent and why it matters now.",
            "Bound the time scope and requested evidence type.",
            "Use one narrow concrete query; treat records as evidence, not conclusions.",
            "Refine at most once when the first packet is broad, stale, noisy, or insufficient.",
        ]
    return result


@dataclass(frozen=True)
class QualityFlags:
    """Independently reversible Cogito Hermeneutics V0.4 controls."""

    retrieval_quality: bool = True
    exact_text_dedup: bool = True
    low_information_downweight: bool = True
    valid_negative_abstention: bool = False
    clear_noise_control: bool = True
    current_truth_precedence: bool = True

    @classmethod
    def from_environment(cls) -> "QualityFlags":
        defaults = cls()
        values: dict[str, bool] = {}
        for field_name, aliases in _FEATURE_ENV.items():
            raw_name = next((name for name in aliases if name in os.environ), None)
            values[field_name] = (
                getattr(defaults, field_name)
                if raw_name is None
                else _env_bool(raw_name, os.environ[raw_name])
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "exact_text_dedup": self.exact_text_dedup,
            "low_information_downweight": self.low_information_downweight,
            "valid_negative_abstention": self.valid_negative_abstention,
            "current_truth_precedence": self.current_truth_precedence,
            "environment_controls": {
                name: aliases[0]
                for name, aliases in _FEATURE_ENV.items()
                if name not in {"retrieval_quality", "clear_noise_control"}
            },
        }
        if self.retrieval_quality:
            result["retrieval_quality"] = True
            result["clear_noise_control"] = self.clear_noise_control
            result["environment_controls"].update(
                {
                    "retrieval_quality": _FEATURE_ENV["retrieval_quality"][0],
                    "clear_noise_control": _FEATURE_ENV["clear_noise_control"][0],
                }
            )
        return result


@dataclass(frozen=True)
class RetrievalPlan:
    mode: str
    initial_depth: int
    ordering: str
    cues: tuple[str, ...]
    detected_entities: tuple[str, ...]
    automatic: bool
    evidence_needs: tuple[str, ...] = ()
    entity_or_referent: str | None = None
    purpose: str | None = None
    time_scope: str | None = None
    query: str | None = None
    disposition: str = "retrieve"
    relation_envelopes_active: bool = False
    planner: str = "deterministic-zero-llm"
    budget_ms: int = AUTOMATIC_TIMEOUT_MS

    def to_dict(self) -> dict[str, Any]:
        result = {
            "mode": self.mode,
            "initial_depth": self.initial_depth,
            "ordering": self.ordering,
            "cues": list(self.cues),
            "detected_entities": list(self.detected_entities),
            "automatic": self.automatic,
            "planner": self.planner,
            "budget_ms": self.budget_ms,
            "disposition": self.disposition,
        }
        if self.entity_or_referent:
            result["entity_or_referent"] = self.entity_or_referent
        if self.purpose:
            result["purpose"] = self.purpose
        if self.time_scope:
            result["time_scope"] = self.time_scope
        if self.evidence_needs:
            result["evidence_needed"] = list(self.evidence_needs)
        if self.query:
            result["query"] = self.query
        if self.relation_envelopes_active:
            result["evidence_needs"] = list(self.evidence_needs)
            result["relation_envelope_schema"] = RELATION_SCHEMA_VERSION
        return result

    def to_legacy_dict(self) -> dict[str, Any]:
        result = {
            "mode": self.mode,
            "initial_depth": self.initial_depth,
            "ordering": self.ordering,
            "cues": list(self.cues),
            "detected_entities": list(self.detected_entities),
            "automatic": self.automatic,
            "planner": self.planner,
            "budget_ms": self.budget_ms,
        }
        if self.relation_envelopes_active:
            result["evidence_needs"] = list(self.evidence_needs)
            result["relation_envelope_schema"] = RELATION_SCHEMA_VERSION
        return result


def _literal_subject_tokens(
    message: str,
    recent_turns: list[str],
) -> tuple[str, ...]:
    """Return bounded literal subject tokens, without resolving a referent."""

    tokens = [
        token
        for token in _WORD_RE.findall(message.casefold())
        if len(token) >= 3 and token not in _SUBJECT_STOPWORDS
    ]
    if tokens:
        return tuple(dict.fromkeys(tokens))[:8]
    for turn in reversed(recent_turns):
        prior = [
            token
            for token in _WORD_RE.findall(turn.casefold())
            if len(token) >= 3 and token not in _SUBJECT_STOPWORDS
        ]
        if prior:
            return tuple(dict.fromkeys(prior))[:8]
    return ()


def _comparative_subject_is_ambiguous(
    message: str,
    recent_turns: list[str],
) -> bool:
    """Conservatively preserve unresolved comparative referents.

    Literal query tokens or bounded recent-turn tokens may identify a subject.
    The function never chooses between explicit alternatives and never treats a
    generic noun (for example, ``the integration``) as a resolved entity.
    """

    lowered = message.casefold()
    if re.search(r"\b[a-z0-9_.-]{3,}\s+or\s+[a-z0-9_.-]{3,}\b", lowered):
        if not re.search(r"\b(?:specifically|meaning|for)\s+[a-z0-9_.-]{3,}\b", lowered):
            return True
    tokens = _literal_subject_tokens(message, recent_turns)
    if not tokens:
        return True
    if _VAGUE_SUBJECT_RE.search(message):
        local_tokens = _literal_subject_tokens(message, [])
        explicit_named_project = bool(
            re.search(r"\bProject\s+[A-Z][A-Za-z0-9_.-]{2,}\b", message)
        )
        if explicit_named_project or len(local_tokens) >= 2:
            return False
        return not any(_literal_subject_tokens(turn, []) for turn in recent_turns)
    return False


def _structured_plan_fields(
    message: str,
    *,
    mode: str,
    detected_entities: tuple[str, ...],
    cues: tuple[str, ...],
    evidence_needs: tuple[str, ...],
) -> dict[str, Any]:
    lowered = message.casefold()
    entity: str | None = None
    for phrase in _DOMAIN_PHRASES:
        if phrase in lowered:
            entity = " ".join(part.capitalize() for part in phrase.split())
            if phrase == "fidelis":
                entity = "Fidelis"
            elif phrase in {"cogito hermeneutics", "cogito-hermeneutics"}:
                entity = "Cogito Hermeneutics"
            break
    if entity is None:
        entity = next(
            (
                value
                for value in detected_entities
                if value.casefold() not in {"what", "when", "where", "which", "show", "recall"}
            ),
            None,
        )

    purpose_by_mode = {
        "semantic": "retrieve evidence about the named subject",
        "current": "establish the latest supported recorded state",
        "timeline": "recover the relevant historical sequence",
        "decision": "recover the recorded decision and rationale",
        "referential": "resolve the current conversational referent from records",
        "exhaustive": "inspect the bounded recorded history",
        "mixed": "compare the relevant recorded states and evidence",
    }
    time_scope_by_mode = {
        "current": "latest supported recorded state",
        "timeline": "historical records",
        "decision": "decision time and relevant prior context",
        "mixed": "relevant prior and current records",
    }
    needs = list(evidence_needs)
    if not needs:
        if mode == "current":
            needs = ["current-state record", "record pointer"]
        elif mode == "timeline":
            needs = ["historical records", "record pointers"]
        elif mode == "decision" or "decision" in cues:
            needs = ["decision record", "rationale evidence"]
        elif mode != "none":
            needs = ["verbatim record", "record pointer"]
    return {
        "entity_or_referent": entity,
        "purpose": purpose_by_mode.get(mode),
        "time_scope": time_scope_by_mode.get(mode),
        "evidence_needs": tuple(needs),
        "query": message.strip() if mode != "none" else None,
        "disposition": "retrieve" if mode != "none" else "abstain",
    }


def plan_retrieval(
    message: str,
    *,
    recent_turns: list[str] | None = None,
    automatic: bool = True,
    explicit_mode: str | None = None,
) -> RetrievalPlan:
    """Decide whether and how to retrieve using bounded deterministic cues."""

    started = time.perf_counter()
    local = message.strip()
    recent = list(recent_turns or [])[-4:]
    cues: list[str] = []
    relations_active = relation_envelopes_enabled()
    referential_followup = bool(_REFERENTIAL_FOLLOWUP_RE.fullmatch(local))
    inherited_comparative = bool(
        referential_followup
        and any(
            _COMPARATIVE_NEED_RE.search(turn)
            and not _comparative_subject_is_ambiguous(turn, [])
            for turn in recent
        )
    )
    comparative_shape = bool(
        _COMPARATIVE_NEED_RE.search(local) or inherited_comparative
    )
    comparative_ambiguous = (
        bool(_COMPARATIVE_NEED_RE.search(local))
        and _comparative_subject_is_ambiguous(local, recent)
    )
    comparative_need = (
        relations_active and comparative_shape and not comparative_ambiguous
    )
    decision_rationale_need = bool(
        _DECISION_RATIONALE_RE.search(local)
        or (
            referential_followup
            and any(_DECISION_RATIONALE_RE.search(turn) for turn in recent)
        )
    )
    checks = {
        "exhaustive": r"\b(?:all|every|everything|full|exhaustive|complete history)\b",
        "timeline": r"\b(?:timeline|chronolog\w*|histor\w*|as[- ]of|over time|what happened|when|last few days|recent work)\b",
        "current": r"\b(?:current|currently|now|today|latest recorded status|live status)\b",
        "decision": r"\b(?:decid\w*|decision|chosen|why did we|owner)\b",
        "referential": r"\b(?:we discussed|talked about|mentioned before|previously|earlier|that work|that thing|our last chat)\b",
        "correction": r"\b(?:correct\w*|supersed\w*|invalidat\w*|contradict\w*|actually|was wrong)\b",
    }
    for cue, pattern in checks.items():
        if re.search(pattern, local, re.I):
            cues.append(cue)
    if comparative_need:
        cues.append("comparative")
    elif relations_active and comparative_shape and comparative_ambiguous:
        cues.append("comparative_ambiguous")
    if decision_rationale_need:
        cues.append("decision_rationale")
    if referential_followup and recent:
        cues.append("referential")

    memory_intent = bool(_MEMORY_INTENT_RE.search(local))
    domain_intent = any(phrase in local.lower() for phrase in _DOMAIN_PHRASES)
    unrelated_action = bool(_UNRELATED_ACTION_RE.search(local))
    source_bound_intent = bool(
        _PAST_WORK_RE.search(local)
        or memory_intent
        or domain_intent
        or (
            comparative_shape
            and any(_PAST_WORK_RE.search(turn) for turn in recent)
        )
        or (
            referential_followup
            and any(
                _PAST_WORK_RE.search(turn)
                or _MEMORY_INTENT_RE.search(turn)
                or _HISTORICAL_RE.search(turn)
                or _COMPARATIVE_NEED_RE.search(turn)
                or _DECISION_RATIONALE_RE.search(turn)
                for turn in recent
            )
        )
        or (
            "decision" in cues
            and re.search(r"\b(?:we|our|my)\b", local, re.IGNORECASE)
        )
    )
    critical_state_intent = bool(
        source_bound_intent
        and (
            _CHANGING_STATE_RE.search(local)
            or _HISTORICAL_RE.search(local)
            or "decision" in cues
            or decision_rationale_need
        )
    )
    # Comparative wording alone is common in general-knowledge questions.
    # Linked admission is licensed only when the turn is also source-bound.
    comparative_need = comparative_need and (
        source_bound_intent or not automatic
    )

    if explicit_mode:
        mode = explicit_mode
    elif not automatic:
        mode = "semantic"
        cues.append("explicit_recall")
    elif not local or _GREETING_RE.fullmatch(local):
        mode = "none"
    elif unrelated_action and not source_bound_intent:
        mode = "none"
    elif comparative_need:
        mode = "mixed"
    elif "exhaustive" in cues:
        mode = "exhaustive"
    else:
        primary = [
            cue
            for cue in ("timeline", "current", "decision", "referential")
            if cue in cues
        ]
        if len(primary) > 1 or "correction" in cues:
            mode = "mixed" if source_bound_intent else "none"
        elif primary:
            mode = primary[0] if source_bound_intent else "none"
        elif critical_state_intent:
            mode = "mixed"
        elif memory_intent:
            mode = "referential"
        elif domain_intent:
            mode = "semantic"
        elif source_bound_intent:
            mode = "referential"
        else:
            mode = "none"

    depths = {
        "none": 0,
        "semantic": 5,
        "current": 5,
        "timeline": 12,
        "decision": 8,
        "referential": 8,
        "exhaustive": 25,
        "mixed": 12,
    }
    orderings = {
        "none": "none",
        "semantic": "relevance_then_information_gain",
        "current": "current_truth_then_relevance",
        "timeline": "relevance_then_chronology",
        "decision": "relevance_then_decision_evidence",
        "referential": "relevance_then_information_gain",
        "exhaustive": "relevance_then_chronology",
        "mixed": "current_truth_then_relevance_then_chronology",
    }
    elapsed_ms = (time.perf_counter() - started) * 1000
    if elapsed_ms > AUTOMATIC_TIMEOUT_MS:
        mode = "none"
        cues = ["planner_timeout"]

    detected_entities = tuple(
        dict.fromkeys(_ENTITY_RE.findall(" ".join(recent + [local])))
    )[-8:]
    evidence_needs = (
        ("prior", "current")
        if comparative_need
        else ("decision", "rationale")
        if decision_rationale_need and source_bound_intent
        else ()
    )
    structured = _structured_plan_fields(
        local,
        mode=mode,
        detected_entities=detected_entities,
        cues=tuple(dict.fromkeys(cues)),
        evidence_needs=evidence_needs,
    )
    return RetrievalPlan(
        mode=mode,
        initial_depth=depths[mode],
        ordering=orderings[mode],
        cues=tuple(dict.fromkeys(cues)),
        detected_entities=detected_entities,
        automatic=automatic,
        evidence_needs=structured["evidence_needs"],
        entity_or_referent=structured["entity_or_referent"],
        purpose=structured["purpose"],
        time_scope=structured["time_scope"],
        query=structured["query"],
        disposition=structured["disposition"],
        relation_envelopes_active=relations_active,
    )


def _terms(text: str) -> set[str]:
    return {term for term in _WORD_RE.findall(text.lower()) if term not in _STOPWORDS}


def _record_id(item: dict[str, Any], index: int) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for value in (
        item.get("record_id"),
        item.get("id"),
        item.get("memory_id"),
        metadata.get("record_id"),
        metadata.get("id"),
        metadata.get("memory_id"),
    ):
        if value:
            return str(value)
    identity = "\x00".join(
        (
            str(item.get("text") or ""),
            str(item.get("source_pointer") or metadata.get("source_pointer") or ""),
            str(item.get("created_at") or metadata.get("created_at") or ""),
        )
    )
    return f"sha256:{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def candidate_record_id(item: dict[str, Any], index: int = 0) -> str:
    """Expose the same stable packet identity used by the hermeneutics layer."""

    return _record_id(item, index)


def _provenance(item: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "record_id": _record_id(item, index),
        "source_pointer": item.get("source_pointer") or metadata.get("source_pointer"),
        "source_span": item.get("source_span") or metadata.get("source_span"),
        "recorded_at": (
            item.get("recorded_at")
            or item.get("created_at")
            or metadata.get("recorded_at")
            or metadata.get("created_at")
        ),
        "authority": item.get("authority") or metadata.get("authority"),
        "score": item.get("score"),
    }


def _low_information_reasons(text: str, item: dict[str, Any]) -> list[str]:
    reasons = [name for name, pattern in _LOW_INFORMATION_PATTERNS if pattern.search(text)]
    if item.get("ephemera"):
        reasons.append("current_truth_ephemera_pattern")
    terms = _terms(text)
    if len(terms) <= 2 and len(text) < 80:
        reasons.append("very_low_lexical_information")
    return sorted(set(reasons))


def clear_noise_reasons(text: str, item: dict[str, Any] | None = None) -> list[str]:
    """Return only deterministic, high-confidence non-memory artifact labels."""

    value = text.strip()
    if not value:
        return ["empty_record"]
    reasons = [name for name, pattern in _CLEAR_NOISE_PATTERNS if pattern.search(value)]
    metadata = (item or {}).get("metadata")
    if isinstance(metadata, dict) and metadata.get("extraction_status") in {
        "failed",
        "empty",
    }:
        reasons.append("failed_or_empty_summary")
    return sorted(set(reasons))


def choose_refinement_reason(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    plan: RetrievalPlan,
    enabled: bool = True,
) -> str | None:
    """License no more than one narrower attempt from observable first results."""

    if not enabled or plan.mode == "none" or plan.relation_envelopes_active:
        return None
    nonempty = [
        candidate
        for candidate in candidates[:5]
        if str(candidate.get("text") or candidate.get("memory") or "").strip()
    ]
    if not nonempty:
        return (
            "no_supported_evidence"
            if plan.entity_or_referent
            else None
        )
    noise_count = sum(
        bool(
            clear_noise_reasons(
                str(candidate.get("text") or candidate.get("memory") or ""),
                candidate,
            )
        )
        for candidate in nonempty
    )
    if noise_count >= 2 and noise_count / len(nonempty) >= 0.4:
        return "clear_corpus_noise"
    if plan.mode != "timeline":
        stale_count = sum(bool(candidate.get("supersession")) for candidate in nonempty)
        if stale_count >= 2 and stale_count / len(nonempty) >= 0.5:
            return "stale_results"
    query_terms = _terms(query)
    broad_terms = {"happened", "history", "recent", "status", "work"}
    if (
        plan.entity_or_referent
        and len(query_terms) <= 4
        and bool(query_terms & broad_terms)
    ):
        return "query_too_broad"
    if plan.evidence_needs:
        packet_terms = {
            term
            for candidate in nonempty
            for term in _terms(str(candidate.get("text") or candidate.get("memory") or ""))
        }
        evidence_markers = {
            "prior": {"prior", "previous", "previously", "before", "earlier"},
            "current": {"current", "currently", "now", "latest"},
            "decision": {"decision", "decided", "chosen", "selected"},
            "rationale": {"rationale", "reason", "because", "why"},
        }
        requested = {
            marker
            for need in plan.evidence_needs
            for marker in evidence_markers
            if marker in _terms(need)
        }
        missing = {
            marker
            for marker in requested
            if not (packet_terms & evidence_markers[marker])
        }
        if requested and missing:
            return "missing_requested_evidence_type"
    return None


def build_refined_query(query: str, plan: RetrievalPlan, reason: str) -> str | None:
    """Build one deterministic narrower query without inventing aliases or facts."""

    parts: list[str] = []
    if plan.entity_or_referent:
        parts.append(plan.entity_or_referent)
    if plan.mode == "current":
        parts.extend(("current", "recorded", "state"))
    elif plan.mode == "timeline":
        parts.extend(("historical", "record", "evidence"))
    elif plan.mode == "decision" or "decision" in plan.cues:
        parts.extend(("decision", "rationale", "evidence"))
    elif reason == "clear_corpus_noise":
        parts.extend(("recorded", "decision", "evidence"))
    if plan.mode not in {"current", "timeline", "decision"}:
        for need in plan.evidence_needs:
            parts.extend(
                term
                for term in _WORD_RE.findall(need.casefold())
                if term not in _STOPWORDS
            )
    parts.extend(
        term
        for term in _WORD_RE.findall(query.casefold())
        if term not in _STOPWORDS
        and term not in {
            "before",
            "changed",
            "happened",
            "history",
            "recent",
            "status",
            "work",
        }
    )
    unique_parts: list[str] = []
    seen_parts: set[str] = set()
    for part in parts:
        key = part.casefold()
        if key in seen_parts:
            continue
        seen_parts.add(key)
        unique_parts.append(part)
    refined = " ".join(unique_parts).strip()
    if not refined or refined.casefold() == query.strip().casefold():
        return None
    return refined


def merge_candidate_attempts(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union two bounded attempts while retaining the first packet's evidence."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate([*second, *first]):
        identity = _record_id(candidate, index)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(candidate)
    return merged


def _exact_query_phrase_hits(query: str, text: str) -> int:
    words = _WORD_RE.findall(query.lower())
    phrases = {
        " ".join(words[index:index + width])
        for width in (2, 3)
        for index in range(0, len(words) - width + 1)
    }
    lower = text.lower()
    return sum(phrase in lower for phrase in phrases)


def apply_hermeneutics(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    plan: RetrievalPlan,
    limit: int = 5,
    flags: QualityFlags | None = None,
    retrieval_method: str = "unknown",
    protect_codex_history: bool = False,
) -> dict[str, Any]:
    """Apply Cogito Hermeneutics V0.4 controls to a bounded candidate set."""

    started = time.perf_counter()
    active_flags = flags or QualityFlags.from_environment()
    clear_noise_enabled = (
        active_flags.retrieval_quality and active_flags.clear_noise_control
    )
    plan_payload = (
        plan.to_dict()
        if active_flags.retrieval_quality
        else plan.to_legacy_dict()
    )
    if plan.mode == "none":
        return {
            "engine": {"name": ENGINE_NAME, "version": ENGINE_VERSION},
            "query": query,
            "plan": plan_payload,
            "retrieval_status": "not_needed",
            "memories": [],
            "records": [],
            "method": f"{retrieval_method}|{ENGINE_NAME}-v{ENGINE_VERSION}",
            "feature_flags": active_flags.to_dict(),
            "source_fidelity": True,
            "candidate_count": 0,
            "shown_count": 0,
            "more_available": False,
            "latency": {"hermeneutics_ms": round((time.perf_counter() - started) * 1000, 3)},
        }

    prepared: list[dict[str, Any]] = []
    historical_request = bool(_HISTORICAL_RE.search(query))
    session_request = bool(_SESSION_OR_PROVENANCE_RE.search(query))
    for index, raw in enumerate(candidates):
        text = str(raw.get("text") or raw.get("memory") or "")
        if not text:
            continue
        item = dict(raw)
        item["text"] = text
        item["_input_index"] = index
        item["_record_id"] = _record_id(item, index)
        item["_provenance"] = _provenance(item, index)
        item["_low_information"] = _low_information_reasons(text, item)
        item["_clear_noise"] = clear_noise_reasons(text, item)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        item["_protected_history"] = (
            protect_codex_history
            and metadata.get("source_kind") == "codex_task_history"
        )
        score = item.get("score")
        base_score = float(score) if isinstance(score, (int, float)) else 1.0 / (index + 1)
        adjustment = 0.0
        reasons: list[str] = []
        if (
            active_flags.current_truth_precedence
            and item.get("supersession")
            and not historical_request
        ):
            adjustment -= 2.0
            reasons.append("superseded_or_retracted_current_truth_penalty")
        item["_base_score"] = base_score
        item["_rank_adjustment"] = adjustment
        item["_adjustment_reasons"] = reasons
        prepared.append(item)

    has_direct_evidence = any(
        not item["_low_information"]
        and (not active_flags.retrieval_quality or not item["_clear_noise"])
        for item in prepared
    )
    if active_flags.low_information_downweight and has_direct_evidence:
        for item in prepared:
            if item["_low_information"] and plan.mode not in {"timeline", "exhaustive"}:
                item["_rank_adjustment"] -= 0.2 if session_request else 0.4
                item["_adjustment_reasons"].append(
                    "low_information_penalty_session_query_but_retained"
                    if session_request
                    else "low_information_penalty_with_direct_evidence"
                )
            phrase_hits = _exact_query_phrase_hits(query, item["text"])
            if phrase_hits:
                item["_rank_adjustment"] += min(0.24, phrase_hits * 0.08)
                item["_adjustment_reasons"].append("exact_query_phrase_support_bonus")

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        key = item["text"] if active_flags.exact_text_dedup else f"{item['_record_id']}:{item['_input_index']}"
        groups.setdefault(key, []).append(item)

    representatives: list[dict[str, Any]] = []
    duplicate_excess = 0
    for group in groups.values():
        representative = max(
            group,
            key=lambda item: (
                item["_protected_history"],
                item["_base_score"] + item["_rank_adjustment"],
                -item["_input_index"],
            ),
        )
        duplicate_excess += max(0, len(group) - 1)
        representative = dict(representative)
        representative["_group"] = group
        representatives.append(representative)

    representatives.sort(
        key=lambda item: (
            -(item["_base_score"] + item["_rank_adjustment"]),
            item["_input_index"],
        )
    )
    excluded_clear_noise = [
        {
            "record_id": item["_record_id"],
            "reasons": item["_clear_noise"],
        }
        for item in representatives
        if clear_noise_enabled and item["_clear_noise"]
    ]
    eligible_representatives = [
        item
        for item in representatives
        if not (clear_noise_enabled and item["_clear_noise"])
    ]
    shown_depth = min(limit, plan.initial_depth or limit)
    shown = eligible_representatives[:shown_depth]
    protected_history = [
        item for item in eligible_representatives if item["_protected_history"]
    ]
    if protected_history and not any(item["_protected_history"] for item in shown):
        if shown:
            shown[-1] = protected_history[0]
        else:
            shown = protected_history[:1]
        shown.sort(
            key=lambda item: (
                -(item["_base_score"] + item["_rank_adjustment"]),
                item["_input_index"],
            )
        )
    records: list[dict[str, Any]] = []
    for item in shown:
        group = item["_group"]
        record = {
            key: value
            for key, value in item.items()
            if not key.startswith("_") and key not in {"memory"}
        }
        record["record_id"] = item["_record_id"]
        record["exact_text_group"] = {
            "duplicate_count": len(group),
            "suppressed_count": max(0, len(group) - 1),
            "all_provenance": [
                member["_provenance"] for member in sorted(group, key=lambda member: member["_input_index"])
            ],
            "near_duplicates_merged": False,
        }
        record["retrieval_adjustments"] = {
            "low_information_reasons": item["_low_information"],
            "rank_adjustment": round(item["_rank_adjustment"], 4),
            "reasons": item["_adjustment_reasons"],
        }
        if active_flags.retrieval_quality:
            record["retrieval_adjustments"]["clear_noise_reasons"] = item[
                "_clear_noise"
            ]
        record["current_truth_disposition"] = {
            "status": (
                "historical_record_preserved_with_supersession"
                if item.get("supersession") and historical_request
                else "historical_only_not_current_truth"
                if item.get("supersession")
                else "not_flagged_by_current_truth_controls"
            ),
            "control": item.get("supersession"),
        }
        records.append(record)

    source_fidelity = all(
        record["text"] == next(item["text"] for item in shown if item["_record_id"] == record["record_id"])
        for record in records
    )
    result = {
        "engine": {"name": ENGINE_NAME, "version": ENGINE_VERSION},
        "query": query,
        "plan": plan_payload,
        "retrieval_status": "ok" if records else "no_supported_result_in_bounded_window",
        "memories": records,
        "records": records,
        "method": f"{retrieval_method}|{ENGINE_NAME}-v{ENGINE_VERSION}",
        "feature_flags": active_flags.to_dict(),
        "exact_text_duplicate_control": {
            "enabled": active_flags.exact_text_dedup,
            "candidate_count_before_collapse": len(prepared),
            "representative_count_after_collapse": len(representatives),
            "suppressed_excess_count": duplicate_excess,
            "all_provenance_retained_behind_representatives": True,
            "near_duplicates_merged": False,
        },
        "low_information_control": {
            "enabled": active_flags.low_information_downweight,
            "direct_evidence_present": has_direct_evidence,
            "flagged_record_count": sum(bool(item["_low_information"]) for item in prepared),
        },
        "abstention": {
            "enabled": active_flags.valid_negative_abstention,
            "status": "experimental_disabled_by_default",
            "withheld_records": 0,
        },
        "source_fidelity": source_fidelity,
        "candidate_count": len(candidates),
        "shown_count": len(records),
        "more_available": len(eligible_representatives) > len(records),
        "latency": {"hermeneutics_ms": round((time.perf_counter() - started) * 1000, 3)},
    }
    if active_flags.retrieval_quality:
        result["clear_noise_control"] = {
            "enabled": clear_noise_enabled,
            "policy": "reversible_packet_exclusion_raw_records_preserved",
            "excluded": excluded_clear_noise,
            "excluded_count": len(excluded_clear_noise),
            "corpus_records_deleted": 0,
        }
    if plan.relation_envelopes_active:
        result["declared_relation_evidence"] = expose_linked_raw_evidence(
            evidence_needs=plan.evidence_needs,
            candidates=candidates,
            limit=max(2, min(limit * 2, 8)),
        )
    return result
