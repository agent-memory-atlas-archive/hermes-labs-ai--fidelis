"""Shared recall for natural references to recent local work.

Both clients ask the same kind of question — "what was that agent doing earlier
today?" — and both already have local evidence for it in two different places:
the ingested session index (Claude Code and Codex records) and the read-only
Codex task-history adapter. This module is the one place that reads both,
applies the deterministic reference rules in `fidelis.session_reference`, and
returns a compact answer that names the source exactly.

What a caller gets back is an identity, a real timestamp, the literal capture
class of the underlying evidence, and a pointer to that evidence — enough to
decide which record to open, without reading a transcript to find out.

Two honesty rules are structural rather than advisory:

* `capture_class` is copied verbatim from the source. A Codex prompt-history
  record stays `USER_PROMPTS_ONLY` and a telemetry-only task stays
  `telemetry_partial_no_assistant_text`. Nothing here can present partial
  capture as a full transcript.
* No candidate carries a live/active/idle status. Stored evidence cannot answer
  whether a session is running, so this path does not answer it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fidelis.session_reference import (
    SessionReference,
    local_day,
    meets_evidence_floor,
    parse_session_reference,
    reference_adjustment,
    sort_epoch,
)

MAX_EXCERPT_CHARS = 320
MAX_CANDIDATES = 10

CLAUDE_SESSION_CLASS = "claude_code_session"
CODEX_SESSION_CLASS = "codex_session"
CODEX_TASK_CLASS = "codex_task_history"

_SOURCE_CLASSES = {
    "claude_code": CLAUDE_SESSION_CLASS,
    "codex": CODEX_SESSION_CLASS,
}


def _redact_home(pointer: str) -> str:
    """Collapse the user's home prefix so a pointer stays usable without
    printing an absolute private path."""
    if not pointer:
        return ""
    home = os.path.expanduser("~")
    if home and home != "/" and pointer.startswith(home):
        return "~" + pointer[len(home):]
    return pointer


def _source_class(session_source: str) -> str:
    return _SOURCE_CLASSES.get(session_source, f"{session_source or 'unknown'}_session")


def _excerpt(text: str) -> str:
    """One bounded line. The wording is preserved; only layout is collapsed, so
    a candidate stays identifiable at a glance in a terminal or a tool result."""
    return " ".join(str(text or "").split())[:MAX_EXCERPT_CHARS]


@dataclass(frozen=True)
class RecentWorkCandidate:
    """One local record that could be the work a caller referred to."""

    source_class: str
    identity: str
    timestamp: str
    local_date: str
    capture_class: str
    assistant_text_available: bool
    evidence_pointer: str
    excerpt: str
    score: float
    match_reasons: tuple[str, ...]
    # Present for ingested sessions; empty/0 for a Codex task, which has no
    # project path or turn count of its own.
    project_path: str = ""
    turn_count: int = 0

    def to_dict(self) -> dict:
        return {
            "source_class": self.source_class,
            "identity": self.identity,
            "timestamp": self.timestamp,
            "local_date": self.local_date,
            "capture_class": self.capture_class,
            "assistant_text_available": self.assistant_text_available,
            "evidence_pointer": self.evidence_pointer,
            "excerpt": self.excerpt,
            "score": round(self.score, 4),
            "match_reasons": list(self.match_reasons),
            "project_path": self.project_path,
            "turn_count": self.turn_count,
        }


@dataclass(frozen=True)
class RecentWorkResult:
    query: str
    reference: SessionReference
    candidates: tuple[RecentWorkCandidate, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "reference": self.reference.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
            "live_status_available": False,
        }


def _session_evidence_text(result) -> str:
    """Delegate to the retrieval-side definition so both stages judge a record
    on identical text."""
    from fidelis.recall_sessions import session_evidence_text

    return session_evidence_text(
        result.matched_chunk, getattr(result, "turns", None) or []
    )


def rank_recent_work(
    query: str,
    *,
    session_results=(),
    task_results=(),
    reference: SessionReference | None = None,
    now: datetime | None = None,
    tzinfo=None,
    limit: int = 5,
) -> RecentWorkResult:
    """Merge and rank already-fetched candidates from both local sources.

    Pure with respect to services, so the whole ranking contract is testable
    without ChromaDB, Ollama or a Codex install.

    `session_results` whose `reference_applied` flag is set are taken to have the
    reference adjustment already folded into their score by `query_sessions`
    (which is the only component that sees the whole session corpus and can
    therefore keep an in-window record from being cut before ranking). Anything
    else is adjusted here, so each adjustment is applied exactly once.
    """
    if reference is None:
        reference = parse_session_reference(query, now=now, tzinfo=tzinfo)

    # The floor exists to stop a *reference* ("which session was that?") from
    # being answered with the nearest record. A plain topical search makes no
    # such claim, so applying it there would only discard legitimate paraphrase
    # matches that semantic search is meant to find.
    apply_floor = reference.has_window or bool(reference.explicit_ids)

    limit = max(1, min(int(limit), MAX_CANDIDATES))
    rows: list[tuple[float, float, str, RecentWorkCandidate]] = []
    floor_rejected = 0

    for result in session_results or ():
        identity = result.session_id or ""
        identity_named = bool(reference.matches_identity(identity))
        if apply_floor and not identity_named and not meets_evidence_floor(
            reference, _session_evidence_text(result)
        ):
            floor_rejected += 1
            continue
        score = float(result.score)
        reasons = tuple(getattr(result, "match_reasons", ()) or ())
        if not getattr(result, "reference_applied", False):
            delta, extra = reference_adjustment(
                reference,
                timestamp=result.end_ts or result.start_ts,
                identifiers=(identity,),
                tzinfo=tzinfo,
            )
            score += delta
            reasons = reasons + extra
        timestamp = result.end_ts or result.start_ts or ""
        rows.append((
            score,
            sort_epoch(timestamp),
            identity,
            RecentWorkCandidate(
                source_class=_source_class(result.session_source),
                identity=identity,
                timestamp=timestamp,
                local_date=local_day(timestamp, tzinfo=tzinfo),
                capture_class=result.capture_completeness,
                assistant_text_available=not str(
                    result.capture_completeness
                ).startswith("USER_PROMPTS_ONLY"),
                evidence_pointer=_redact_home(result.source_locator or ""),
                excerpt=_excerpt(result.matched_chunk),
                score=score,
                match_reasons=reasons,
                project_path=result.project_path,
                turn_count=int(result.turn_count or 0),
            ),
        ))

    for result in task_results or ():
        identity = result.thread_id or ""
        identity_named = bool(reference.matches_identity(identity))
        if apply_floor and not identity_named and not meets_evidence_floor(
            reference, result.text
        ):
            floor_rejected += 1
            continue
        timestamp = result.recorded_at or ""
        delta, reasons = reference_adjustment(
            reference,
            timestamp=timestamp,
            identifiers=(identity,),
            tzinfo=tzinfo,
        )
        score = float(result.score) + delta
        pointer = _redact_home(result.source_pointer or "")
        if result.source_span:
            pointer = f"{pointer}#{result.source_span}" if pointer else result.source_span
        rows.append((
            score,
            sort_epoch(timestamp),
            identity,
            RecentWorkCandidate(
                source_class=CODEX_TASK_CLASS,
                identity=identity,
                timestamp=timestamp,
                local_date=local_day(timestamp, tzinfo=tzinfo),
                capture_class=result.evidence_class,
                assistant_text_available=result.assistant_text_available,
                evidence_pointer=pointer,
                excerpt=_excerpt(result.text),
                score=score,
                match_reasons=reasons,
            ),
        ))

    # Deterministic order: score, then newer evidence, then identity as a final
    # tiebreak so the result never depends on store scan order.
    rows.sort(key=lambda row: (-round(row[0], 6), -row[1], row[2]))

    # One row per record. A rollout can match on several messages, and a caller
    # deciding *which* session to open gains nothing from the same id repeated.
    deduplicated: list[tuple[float, int, str, RecentWorkCandidate]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row[3].source_class, row[3].identity)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)
    rows = deduplicated

    notes: list[str] = []
    if reference.window_label:
        notes.append(
            f"time reference '{reference.window_label}' resolved to "
            f"{reference.window_start}..{reference.window_end} "
            f"(UTC offset {reference.tz_offset})"
        )
    if reference.explicit_ids:
        notes.append("explicit identity requested; identity match outranks similarity")
    if floor_rejected:
        notes.append(
            f"{floor_rejected} candidate(s) withheld for insufficient lexical evidence"
        )
    if not rows:
        notes.append(
            "no local record met the evidence floor for this reference"
            if apply_floor
            else "no local record matched this query"
        )
    notes.append("live session/task status is not derivable from stored evidence")

    return RecentWorkResult(
        query=query,
        reference=reference,
        candidates=tuple(candidate for _, _, _, candidate in rows[:limit]),
        notes=tuple(notes),
    )


def recall_recent_work(
    query: str,
    *,
    limit: int = 5,
    now: datetime | None = None,
    tzinfo=None,
    codex_home: Path | None = None,
) -> RecentWorkResult:
    """Answer a recent-work reference from both local sources.

    Each source is consulted independently and a failure in one never hides the
    other, matching the existing behaviour of the shared session tool.
    """
    reference = parse_session_reference(query, now=now, tzinfo=tzinfo)
    fetch_limit = max(limit, 5)

    session_results = []
    session_error: str | None = None
    try:
        from fidelis.recall_sessions import query_sessions

        session_results = query_sessions(
            query, top_k=fetch_limit, reference=reference, tzinfo=tzinfo
        )
    except Exception as exc:  # one source failing must not hide the other
        session_error = type(exc).__name__

    task_results = []
    task_error: str | None = None
    try:
        from fidelis.codex_history import search_codex_history

        kwargs = {"limit": fetch_limit}
        if codex_home is not None:
            kwargs["codex_home"] = codex_home
        task_results = search_codex_history(query, **kwargs)
    except Exception as exc:  # same isolation in the other direction
        task_results = []
        task_error = type(exc).__name__

    result = rank_recent_work(
        query,
        session_results=session_results,
        task_results=task_results,
        reference=reference,
        now=now,
        tzinfo=tzinfo,
        limit=limit,
    )
    # An unavailable source is said out loud in both directions. Silence would
    # make a broken adapter indistinguishable from "there is no such work".
    prefix: tuple[str, ...] = ()
    if session_error is not None:
        prefix += (f"ingested session index unavailable ({session_error})",)
    if task_error is not None:
        prefix += (f"local Codex task evidence unavailable ({task_error})",)
    if prefix:
        result = RecentWorkResult(
            query=result.query,
            reference=result.reference,
            candidates=result.candidates,
            notes=prefix + result.notes,
        )
    return result
