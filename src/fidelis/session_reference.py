"""Deterministic parsing of natural references to recent local work.

A user who says "the other agent today that audited Agent Kickstart" is naming
two things a retrieval score cannot see: a time window and, sometimes, a
literal session or task identity. This module turns that phrasing into explicit,
testable signals, and defines the fixed score adjustments the shared session
recall path applies on top of them.

Everything here is pure. There is no model call, no service call and no clock
side effect that a caller cannot inject, so the same query plus the same
`now`/`tzinfo` always yields the same decision.

Deliberate boundaries:

* Adjustments only ever RE-RANK candidates that already carry independent
  lexical evidence. The single exception is a literal identity match, which is
  an identity lookup rather than a similarity judgement.
* An out-of-window candidate is demoted, never removed. An explicitly named
  older date becomes the window itself, so naming an older referent cannot be
  suppressed by newer work.
* Nothing here reports whether a session or task is running. Live state is a
  control-plane question and stored evidence cannot answer it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

# Fixed adjustments. The identity bonus is deliberately larger than any score a
# similarity match can reach (base similarity 1.0 + literal-phrase 1.0 + window
# 0.75), so naming an id is an identity lookup and not a hint. The window terms
# are large enough that a stale but lexically similar record cannot outrank
# same-window evidence, and small enough that they cannot invent relevance.
EXACT_IDENTITY_BONUS = 4.0
IN_WINDOW_BONUS = 0.75
OUT_OF_WINDOW_PENALTY = -0.5

# Share of a query's content terms that must appear in a candidate before it may
# be reported as a match at all.
EVIDENCE_FLOOR_RATIO = 0.34

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_LABELLED_ID_RE = re.compile(
    r"\b(?:session|task|thread|conversation|chat|run|id)\s*(?:id)?\s*[:=#]?\s*"
    r"([0-9a-fA-F]{8,})\b"
)
# An all-digit token after "run"/"id" is a count, a row number or a date stamp,
# not a session identity. Requiring a hex letter keeps an accidental number from
# being granted identity dominance and an evidence-floor bypass.
_HEX_LETTER_RE = re.compile(r"[a-fA-F]")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TODAY_RE = re.compile(
    r"\b(?:today|today's|todays|earlier today|this morning|this afternoon|"
    r"this evening|tonight|just now)\b",
    re.IGNORECASE,
)
_YESTERDAY_RE = re.compile(
    r"\b(?:yesterday|yesterday's|yesterdays|last night)\b", re.IGNORECASE
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*")
_TIME_WORDS = frozenset({
    "today", "todays", "tonight", "yesterday", "yesterdays", "morning",
    "afternoon", "evening", "night", "earlier", "recent", "recently", "just",
    "now", "ago", "day", "days", "week", "weeks", "month", "months", "date",
})
_STOPWORDS = frozenset({
    "about", "after", "again", "also", "and", "another", "any", "are", "around",
    "because", "been", "before", "being", "but", "can", "come", "could", "did",
    "does", "done", "during", "each", "else", "even", "every", "find", "for",
    "from", "get", "give", "had", "has", "have", "her", "his", "how", "into",
    "its", "let", "like", "look", "made", "make", "many", "may", "me", "more",
    "most", "much", "must", "need", "not", "one", "only", "other", "our", "out",
    "over", "please", "pull", "put", "same", "see", "she", "should", "show",
    "some", "such", "sure", "take", "tell", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "used", "using", "very", "want", "was", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your",
})
_RETROSPECTIVE_SCAFFOLD = frozenset({
    "doing", "work", "working", "worked", "up",
})
_RETROSPECTIVE_TOPIC_RE = re.compile(
    r"\bwhat\s+(?:"
    r"was\s+i\s+(?:doing|working\s+on|up\s+to)|"
    r"were\s+we\s+(?:doing|working\s+on)|"
    r"have\s+i\s+been\s+(?:doing|working\s+on)|"
    r"did\s+(?:i|we)\s+(?:do|work\s+on)"
    r")\b",
    re.IGNORECASE,
)
_CANONICAL_TOPIC_ALIASES = {"linter": "lintlang"}


def canonical_topic_term(token: str) -> str:
    """Return the one supported canonical project-topic spelling."""
    lowered = token.lower()
    return _CANONICAL_TOPIC_ALIASES.get(lowered, lowered)


@dataclass(frozen=True)
class SessionReference:
    """What a query literally asked for, in explicit terms."""

    query: str
    window_start: str | None = None
    window_end: str | None = None
    window_label: str = ""
    explicit_ids: tuple[str, ...] = ()
    content_terms: tuple[str, ...] = field(default_factory=tuple)
    tz_offset: str = ""

    @property
    def has_window(self) -> bool:
        return self.window_start is not None and self.window_end is not None

    def covers(self, local_day: str) -> bool:
        """True when a candidate's local calendar date falls in the window."""
        if not self.has_window or not local_day:
            return False
        return self.window_start <= local_day[:10] <= self.window_end

    def matches_identity(self, *identifiers: str) -> str:
        """Return the identifier a caller named literally, or ""."""
        if not self.explicit_ids:
            return ""
        for identifier in identifiers:
            if not identifier:
                continue
            lowered = identifier.lower()
            for named in self.explicit_ids:
                if lowered == named or (len(named) >= 8 and lowered.startswith(named)):
                    return identifier
        return ""

    def to_dict(self) -> dict:
        return {
            "window_label": self.window_label,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "explicit_ids": list(self.explicit_ids),
            "timezone_offset": self.tz_offset,
        }


def _resolve_now(now: datetime | None, tzinfo) -> datetime:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if tzinfo is None:
        tzinfo = datetime.now().astimezone().tzinfo or timezone.utc
    return now.astimezone(tzinfo)


def content_terms(text: str) -> tuple[str, ...]:
    """Terms that carry topic. Time words and stopwords are excluded because
    they describe *when*, not *what*, and must not count as evidence.

    A first-person retrospective may contain scaffolding such as "doing" or
    "working". When it also names a concrete topic, that scaffolding cannot
    stand in for topical evidence. The one canonical alias is intentionally
    narrow: users commonly call the LintLang project "the linter" while its
    retained evidence uses the project name.
    """
    seen: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < 3 or token in _STOPWORDS or token in _TIME_WORDS:
            continue
        if _UUID_RE.fullmatch(token):
            continue
        token = canonical_topic_term(token)
        if token not in seen:
            seen.append(token)
    if _RETROSPECTIVE_TOPIC_RE.search(text):
        concrete = [term for term in seen if term not in _RETROSPECTIVE_SCAFFOLD]
        if concrete:
            return tuple(concrete)
    return tuple(seen)


def parse_session_reference(
    query: str,
    *,
    now: datetime | None = None,
    tzinfo=None,
) -> SessionReference:
    """Extract the window and literal identities a query named.

    Dates are resolved in the caller's timezone when one is supplied, otherwise
    in the local timezone, so "today" means the caller's calendar day rather
    than the UTC day a record happens to be stamped with.
    """
    local_now = _resolve_now(now, tzinfo)
    today: date = local_now.date()

    explicit_ids: list[str] = []
    for match in _UUID_RE.finditer(query):
        candidate = match.group(0).lower()
        if candidate not in explicit_ids:
            explicit_ids.append(candidate)
    for match in _LABELLED_ID_RE.finditer(query):
        candidate = match.group(1).lower()
        if _HEX_LETTER_RE.search(candidate) is None:
            continue
        if candidate not in explicit_ids and not any(
            existing.startswith(candidate) for existing in explicit_ids
        ):
            explicit_ids.append(candidate)

    window_start: str | None = None
    window_end: str | None = None
    label = ""

    iso_match = _ISO_DATE_RE.search(query)
    if iso_match is not None:
        try:
            named = date(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )
        except ValueError:
            named = None
        if named is not None:
            # An explicit date is the most specific thing a caller can say, so
            # it wins over a relative word appearing in the same query.
            window_start = window_end = named.isoformat()
            label = "date"
    if label == "" and _TODAY_RE.search(query):
        window_start = window_end = today.isoformat()
        label = "today"
    elif label == "" and _YESTERDAY_RE.search(query):
        window_start = window_end = (today - timedelta(days=1)).isoformat()
        label = "yesterday"

    return SessionReference(
        query=query,
        window_start=window_start,
        window_end=window_end,
        window_label=label,
        explicit_ids=tuple(explicit_ids),
        content_terms=content_terms(query),
        tz_offset=local_now.strftime("%z") or "+0000",
    )


def parse_timestamp(timestamp: str) -> datetime | None:
    """Parse a stored timestamp into an aware datetime, or None.

    A naive timestamp is read as UTC because that is how session ingestion
    writes it. An unparseable value yields None rather than a guess — callers
    must not fabricate a date from a string they could not read.
    """
    if not timestamp:
        return None
    raw = timestamp.strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def local_day(timestamp: str, *, tzinfo=None) -> str:
    """Render a stored timestamp as a calendar date in the caller's timezone."""
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return ""
    if tzinfo is None:
        tzinfo = datetime.now().astimezone().tzinfo or timezone.utc
    return parsed.astimezone(tzinfo).date().isoformat()


def sort_epoch(timestamp: str) -> float:
    """Absolute ordering key for a timestamp; -inf when it cannot be read.

    Derived from the parsed instant, so it stays monotonic in real time
    regardless of the timezone a date is later rendered in.
    """
    parsed = parse_timestamp(timestamp)
    return parsed.timestamp() if parsed is not None else float("-inf")


# Inflectional suffixes, longest first. Stripping these lets "audited" match
# "audit" without the looser behaviour of an arbitrary prefix comparison, which
# also matched unrelated words ("solver" against "solvent").
_SUFFIX_RULES = (
    ("ies", "y"),
    ("ing", ""),
    ("es", ""),
    ("ed", ""),
    ("s", ""),
)


def _stem(word: str) -> str:
    for suffix, replacement in _SUFFIX_RULES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def _document_terms(text_lower: str) -> frozenset[str]:
    """Tokens of a candidate's text, plus their sub-tokens and stems.

    Matching against this set rather than the raw string keeps a term from being
    "found" inside an unrelated longer word ("audit" inside "auditorium"), while
    sub-tokens still let a term match a compound like "retry-policy".
    """
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(text_lower):
        pieces = [token, *re.split(r"[._-]", token)]
        for piece in pieces:
            if piece:
                piece = canonical_topic_term(piece)
                terms.add(piece)
                terms.add(_stem(piece))
    return frozenset(terms)


def _term_present(term: str, doc_terms: frozenset[str]) -> bool:
    if term in doc_terms:
        return True
    return _stem(term) in doc_terms


def evidence_ratio(reference: SessionReference, text: str) -> float:
    """Share of the query's content terms present in a candidate's text."""
    terms = reference.content_terms
    if not terms:
        return 0.0
    doc_terms = _document_terms(text.lower())
    present = sum(1 for term in terms if _term_present(term, doc_terms))
    return present / len(terms)


def meets_evidence_floor(reference: SessionReference, text: str) -> bool:
    """Whether a candidate may be reported as a match at all.

    This is the guard against answering "which session was that?" with the
    nearest available session when the work being described never happened.
    """
    return evidence_ratio(reference, text) >= EVIDENCE_FLOOR_RATIO


def reference_adjustment(
    reference: SessionReference,
    *,
    timestamp: str,
    identifiers: tuple[str, ...] = (),
    tzinfo=None,
) -> tuple[float, tuple[str, ...]]:
    """Fixed score delta and human-readable reasons for one candidate."""
    delta = 0.0
    reasons: list[str] = []

    named = reference.matches_identity(*identifiers)
    if named:
        delta += EXACT_IDENTITY_BONUS
        reasons.append("exact_identity_match")

    if reference.has_window:
        day = local_day(timestamp, tzinfo=tzinfo)
        if not day:
            reasons.append("window_requested_no_timestamp")
        elif reference.covers(day):
            delta += IN_WINDOW_BONUS
            reasons.append(
                "same_day_match" if reference.window_label == "today"
                else f"in_window_match:{reference.window_label or 'window'}"
            )
        else:
            delta += OUT_OF_WINDOW_PENALTY
            reasons.append("outside_requested_window")

    return delta, tuple(reasons)
