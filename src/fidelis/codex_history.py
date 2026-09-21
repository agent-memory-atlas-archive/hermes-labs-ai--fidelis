"""Bounded, read-only retrieval over local Codex task history.

Normal durable recall remains the Fidelis corpus. This adapter is consulted
only for explicit task/session/provenance questions so a missing Codex task
index entry does not make the underlying local evidence invisible.

Two evidence classes are kept distinct:

* saved rollout messages: literal user or assistant text with a file pointer;
* telemetry-only tasks: exact retained user prompts plus assistant item IDs and
  reasoning-summary labels, explicitly marked as missing assistant prose.

The adapter never mutates Codex state, creates no index, and does not inspect
Claude's private session or hook directories.
"""

from __future__ import annotations

import heapq
import json
import os
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE)
_THREAD_RE = re.compile(r"thread_id=([0-9a-f-]{36})")
_TURN_RE = re.compile(r'\bid: "([0-9a-f-]{36})"')
_USER_TEXT_RE = re.compile(
    r'Text \{ text: "((?:\\.|[^"\\])*)", text_elements:',
    re.DOTALL,
)
_ASSISTANT_ITEM_RE = re.compile(
    r'Output item item_type="message" item_id="([^"]+)"'
)
_REASONING_SUMMARY_RE = re.compile(r"summary=(\[[^\r\n]*\]) summaryPartCount=")
_HISTORY_INTENT_RE = re.compile(
    r"(?:\b(?:side[\s-]?chat|last output|lost chat|lost task|"
    r"conversation log|task log)\b|"
    r"\b(?:find|show|recover|locate|have|see)\b.{0,40}"
    r"\bassistant (?:message|output)\b|"
    r"\bassistant (?:message|output)\b.{0,40}"
    r"\b(?:roadmap|post[- ]v?\d|task|thread|session|chat|lost|last)\b|"
    r"\bcodex\b.{0,40}\b"
    r"(?:task|thread|session|chat|assistant|output|history|log|provenance)\b|"
    r"\b(?:task|thread|session|chat|assistant|output|history|log|provenance)\b"
    r".{0,40}\bcodex\b|"
    r"\b(?:task|thread|session)\s+[0-9a-f-]{8,36}\b|"
    r"\b(?:task|thread|session)\b.{0,40}\b"
    r"(?:discussed|mentioned|contained|assistant|output|history|log)\b)",
    re.IGNORECASE,
)
_NON_HISTORY_ACTION_RE = re.compile(
    r"^\s*(?:write|draft|create|build|design|make|generate|compose|show|"
    r"explain|summarize|review)\b",
    re.IGNORECASE,
)
_EXPLICIT_HISTORY_QUALIFIER_RE = re.compile(
    r"(?:\b(?:codex|lost|previous|prior|earlier|past|last|"
    r"side[\s-]?chat|we discussed|we mentioned)\b|"
    r"\b(?:task|thread|session)\s+[0-9a-f-]{8,36}\b)",
    re.IGNORECASE,
)
# A caller who names a time window or a literal id together with a noun that
# denotes a unit of agent work is asking a provenance question even without the
# word "Codex" — "the other agent today that audited X".
#
# Restricted to those nouns on purpose. Verbs like run/did/worked/reviewed and
# deictics like other/earlier are among the most common words in ordinary
# developer prose ("run the tests today", "review the diff I fixed today"), and
# admitting them would pay for rollout, sqlite and desktop-log scans on queries
# that are not about provenance at all.
_RECENT_WORK_REFERENT_RE = re.compile(
    r"\b(?:agent|agents|session|sessions|task|tasks|thread|threads|"
    r"chat|chats|rollout|rollouts)\b",
    re.IGNORECASE,
)
# The same question asked about oneself carries no such noun — "what was I doing
# on the linter today". Matched as a whole first-person past-work construction
# rather than by its words, so "how do I use the linter today" (present tense,
# instructional) and "should I work on the linter today" (forward-looking) stay
# out. The named agent as an explicit subject is the same retrospective
# construction — "what did codex work on yesterday" — and is matched the same
# whole-construction way, so modals ("can codex work on this today") and
# imperatives ("use codex to fix this today") stay out; "codex" is deliberately
# NOT added to the recent-work noun list above. Still requires a date or id
# reference before any file is read.
_RETROSPECTIVE_INTENT_RE = re.compile(
    r"\bwhat\s+(?:"
    r"was\s+(?:i|codex)\s+(?:doing|working\s+on|up\s+to)|"
    r"were\s+we\s+(?:doing|working\s+on)|"
    r"(?:have\s+i|has\s+codex)\s+been\s+(?:doing|working\s+on)|"
    r"did\s+(?:i|we|codex)\s+(?:do|work\s+on)"
    r")\b",
    re.IGNORECASE,
)
# A command that leans on a prior artifact — "run the script we had created
# yesterday" — names the artifact and its past creation relationship instead of
# a unit-of-work noun. Matched as a determiner + artifact noun + relative
# past-creation clause, so imperatives that merely mention an artifact
# ("create a script today") and plain undated action requests ("run the tests
# today") stay out. Still requires a date or id reference before any file is
# read.
_PRIOR_ARTIFACT_INTENT_RE = re.compile(
    r"\b(?:the|that|this|our)\s+(?:script|tool|report|command)\b"
    r".{0,40}?\b(?:we|i|you|codex)\s+(?:had\s+|have\s+)?"
    r"(?:created|built|made|wrote|set\s+up)\b",
    re.IGNORECASE,
)
_INTERNAL_PROMPT_PREFIXES = (
    "# Overview\n\nGenerate 0 to 3 hyperpersonalized suggestions",
    (
        "You are a helpful assistant. You will be presented with a user prompt, "
        "and your job is to provide a short title"
    ),
)

_STOPWORDS = {
    "about",
    "after",
    "again",
    "assistant",
    "before",
    "chat",
    "could",
    "does",
    "find",
    "from",
    "have",
    "last",
    "message",
    "output",
    "please",
    "show",
    "side",
    "task",
    "that",
    "the",
    "this",
    "thread",
    "what",
    "where",
    "with",
}


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def is_codex_history_query(text: str) -> bool:
    """Return whether a query is specifically about Codex task provenance.

    Deliberately unchanged: `fidelis.server` also uses this predicate to promote
    an automatic /recall plan from `none` to `referential`, so widening it here
    would silently change retrieval for that separate HTTP path. The broader
    admission for recent-work phrasing lives in `should_search_task_history`,
    which only this adapter consults.
    """

    if not _HISTORY_INTENT_RE.search(text):
        return False
    return not (
        _NON_HISTORY_ACTION_RE.search(text)
        and not _EXPLICIT_HISTORY_QUALIFIER_RE.search(text)
    )


def _is_recent_work_reference(text: str) -> bool:
    """Whether a query names a time window or literal id for a unit of work.

    Satisfied either by a noun denoting that unit ("agent", "session", "task"),
    by an explicit retrospective construction, which names the same thing
    without a noun for it, or by a prior-artifact construction ("the script we
    had created yesterday"), which names the unit of work by its product.
    """

    if not (
        _RECENT_WORK_REFERENT_RE.search(text)
        or _RETROSPECTIVE_INTENT_RE.search(text)
        or _PRIOR_ARTIFACT_INTENT_RE.search(text)
    ):
        return False
    from fidelis.session_reference import parse_session_reference

    reference = parse_session_reference(text)
    return reference.has_window or bool(reference.explicit_ids)


def should_search_task_history(text: str) -> bool:
    """Whether this adapter should read local Codex task evidence for a query.

    A superset of `is_codex_history_query`: it also admits a natural reference to
    recent work. Admitting a query cannot invent evidence — term scoring still
    has to find a real match, and the shared recall path applies its own
    evidence floor on top.
    """

    if is_codex_history_query(text):
        return True
    if not _is_recent_work_reference(text):
        return False
    return not (
        _NON_HISTORY_ACTION_RE.search(text)
        and not _EXPLICIT_HISTORY_QUALIFIER_RE.search(text)
    )


def _query_terms(text: str) -> set[str]:
    from fidelis.session_reference import content_terms, parse_session_reference

    terms = {
        token for token in content_terms(text) if token not in _STOPWORDS
    }
    terms.update(parse_session_reference(text).explicit_ids)
    terms.update(version.lower() for version in _VERSION_RE.findall(text))
    return terms


def _score(query: str, text: str) -> float:
    from fidelis.session_reference import canonical_topic_term

    query_terms = _query_terms(query)
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    document_terms = {
        canonical_topic_term(token) for token in _WORD_RE.findall(text_lower)
    }
    overlap = sum(
        term in text_lower
        or term in document_terms
        or (term.startswith("v") and term[1:] in text_lower)
        for term in query_terms
    )
    if not overlap:
        return 0.0
    coverage = overlap / len(query_terms)
    phrase_bonus = 0.15 if query.strip().lower() in text_lower else 0.0
    version_bonus = 0.1 if any(version in text_lower for version in _VERSION_RE.findall(query.lower())) else 0.0
    return min(0.99, 0.35 + 0.5 * coverage + phrase_bonus + version_bonus)


def _message_text(payload: dict) -> tuple[str, str, str] | None:
    if payload.get("type") != "message":
        return None
    role = str(payload.get("role") or "")
    if role not in {"user", "assistant"}:
        return None
    chunks: list[str] = []
    for content in payload.get("content") or []:
        if not isinstance(content, dict):
            continue
        text = content.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    if not chunks:
        return None
    return role, "\n".join(chunks), str(payload.get("phase") or "")


@dataclass(frozen=True)
class CodexHistoryResult:
    thread_id: str
    role: str
    text: str
    score: float
    source_pointer: str
    source_span: str | None
    evidence_class: str
    recorded_at: str | None = None
    assistant_text_available: bool = True

    def to_memory(self) -> dict:
        identity = f"codex-history:{self.thread_id}:{self.source_span or 'task'}"
        return {
            "id": identity,
            "record_id": identity,
            "text": self.text,
            "score": round(self.score, 4),
            "source_pointer": self.source_pointer,
            "source_span": self.source_span,
            "recorded_at": self.recorded_at,
            "authority": "local_codex_task_evidence",
            "metadata": {
                "source_kind": "codex_task_history",
                "thread_id": self.thread_id,
                "role": self.role,
                "evidence_class": self.evidence_class,
                "assistant_text_available": self.assistant_text_available,
            },
        }


def _rollout_files(
    sessions_root: Path,
    *,
    max_files: int,
    max_age_days: int,
) -> list[Path]:
    if not sessions_root.exists() or max_files <= 0:
        return []
    now = time.time()
    cutoff = now - max_age_days * 86_400
    newest: list[tuple[float, str]] = []
    for day_offset in range(max_age_days + 1):
        day = time.gmtime(now - day_offset * 86_400)
        day_root = (
            sessions_root
            / f"{day.tm_year:04d}"
            / f"{day.tm_mon:02d}"
            / f"{day.tm_mday:02d}"
        )
        try:
            with os.scandir(day_root) as entries:
                for entry in entries:
                    if not entry.name.endswith(".jsonl"):
                        continue
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if not entry.is_file(follow_symlinks=False) or stat.st_mtime < cutoff:
                        continue
                    candidate = (stat.st_mtime, entry.path)
                    if len(newest) < max_files:
                        heapq.heappush(newest, candidate)
                    elif candidate > newest[0]:
                        heapq.heapreplace(newest, candidate)
        except OSError:
            continue
    newest.sort(reverse=True)
    return [Path(path) for _modified_at, path in newest]


def _thread_id_from_rollout(path: Path, payload: dict) -> str:
    if payload.get("type") == "session_meta":
        candidate = (payload.get("payload") or {}).get("id")
        if candidate:
            return str(candidate)
    match = re.search(r"([0-9a-f-]{36})\.jsonl$", path.name)
    return match.group(1) if match else path.stem


def _search_rollouts(
    query: str,
    *,
    sessions_root: Path,
    limit: int,
    max_files: int,
    max_age_days: int,
    max_bytes: int,
) -> list[CodexHistoryResult]:
    results: list[CodexHistoryResult] = []
    bytes_read = 0
    for path in _rollout_files(
        sessions_root,
        max_files=max_files,
        max_age_days=max_age_days,
    ):
        if bytes_read >= max_bytes:
            break
        thread_id = ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    bytes_read += len(line.encode("utf-8", errors="ignore"))
                    if bytes_read > max_bytes:
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = event.get("payload") or {}
                    if not thread_id:
                        thread_id = _thread_id_from_rollout(path, event)
                    if event.get("type") != "response_item":
                        continue
                    message = _message_text(payload)
                    if not message:
                        continue
                    role, text, phase = message
                    score = _score(query, f"{thread_id}\n{path.name}\n{text}")
                    exact_thread_match = bool(
                        thread_id and thread_id.lower() in query.lower()
                    )
                    if exact_thread_match:
                        score = max(score, 0.85)
                    if score <= 0:
                        continue
                    if role == "assistant":
                        score = min(0.99, score + 0.02)
                    if phase == "final_answer":
                        score = min(0.99, score + 0.06)
                    elif phase == "commentary":
                        score = max(0.0, score - 0.04)
                    if exact_thread_match:
                        # Literal saved text is stronger evidence than a
                        # telemetry-only partial for the same exact task.
                        score = max(score, 0.98)
                        if phase == "final_answer":
                            score = 0.99
                    results.append(
                        CodexHistoryResult(
                            thread_id=thread_id,
                            role=role,
                            text=text[:4_000],
                            score=score,
                            source_pointer=str(path),
                            source_span=f"line:{line_number}",
                            evidence_class="saved_rollout_message",
                            recorded_at=str(event.get("timestamp") or "") or None,
                            assistant_text_available=True,
                        )
                    )
        except OSError:
            continue
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def _decode_rust_debug_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")


def _telemetry_search_terms(query: str) -> list[str]:
    terms = sorted(_query_terms(query), key=lambda term: (-len(term), term))
    versions = [version.lower() for version in _VERSION_RE.findall(query)]
    version_variants = versions + [version[1:] for version in versions]
    prioritized = version_variants + [
        term for term in terms if term not in version_variants
    ]
    return prioritized[:6]


def _matching_telemetry_threads(
    connection: sqlite3.Connection,
    *,
    query: str,
    cutoff: int,
    row_limit: int = 500,
    max_prompt_chars: int = 8_192,
) -> dict[str, list[str]]:
    terms = _telemetry_search_terms(query)
    if not terms:
        return {}
    predicates = " OR ".join(
        "instr(lower(feedback_log_body), ?) > 0" for _ in terms
    )
    candidate_rows = connection.execute(
        "SELECT thread_id, MAX(ts) AS matched_ts"
        " FROM logs"
        " WHERE ts >= ?"
        "   AND target = 'codex_core::session::handlers'"
        "   AND thread_id IS NOT NULL"
        "   AND instr(feedback_log_body, 'op: UserInput') > 0"
        f"   AND ({predicates})"
        " GROUP BY thread_id"
        " ORDER BY matched_ts DESC"
        " LIMIT ?",
        (cutoff, *terms, row_limit),
    ).fetchall()
    thread_ids = [str(row[0]) for row in candidate_rows]
    if not thread_ids:
        return {}
    threads: dict[str, list[str]] = {thread_id: [] for thread_id in thread_ids}
    placeholders = ", ".join("?" for _ in thread_ids)
    prompt_rows = connection.execute(
        "SELECT thread_id, substr(feedback_log_body, 1, ?)"
        " FROM logs"
        f" WHERE thread_id IN ({placeholders})"
        "   AND ts >= ?"
        "   AND target = 'codex_core::session::handlers'"
        "   AND instr(feedback_log_body, 'op: UserInput') > 0"
        " ORDER BY ts DESC, ts_nanos DESC, id DESC"
        " LIMIT ?",
        (max_prompt_chars, *thread_ids, cutoff, row_limit),
    ).fetchall()
    for thread_id, body in prompt_rows:
        match = _USER_TEXT_RE.search(str(body))
        if not match:
            continue
        text = _decode_rust_debug_string(match.group(1)).strip()
        if not text:
            continue
        if text.startswith(_INTERNAL_PROMPT_PREFIXES):
            continue
        prompts = threads[str(thread_id)]
        if text not in prompts:
            prompts.append(text)
    return {thread_id: prompts for thread_id, prompts in threads.items() if prompts}


def _telemetry_task_result(
    connection: sqlite3.Connection,
    *,
    query: str,
    thread_id: str,
    prompts: Iterable[str],
    source_db: Path,
    cutoff: int,
    row_limit: int = 200,
    max_body_chars: int = 8_192,
    desktop_summaries: Iterable[str] = (),
) -> CodexHistoryResult | None:
    rows = connection.execute(
        "SELECT ts, feedback_log_body"
        " FROM ("
        "   SELECT ts, ts_nanos, id,"
        "          substr(feedback_log_body, 1, ?) AS feedback_log_body"
        "   FROM logs"
        "   WHERE thread_id = ? AND ts >= ?"
        "   ORDER BY ts DESC, ts_nanos DESC, id DESC"
        "   LIMIT ?"
        " ) AS recent_rows"
        " ORDER BY ts ASC, ts_nanos ASC, id ASC",
        (max_body_chars, thread_id, cutoff, row_limit),
    ).fetchall()
    summaries: list[str] = list(desktop_summaries)
    assistant_item_ids: list[str] = []
    prompt_list = list(prompts)
    first_ts: int | None = None
    for timestamp, body in rows:
        first_ts = int(timestamp) if first_ts is None else first_ts
        body = str(body)
        prompt_match = _USER_TEXT_RE.search(body)
        if prompt_match:
            prompt = _decode_rust_debug_string(prompt_match.group(1)).strip()
            if (
                prompt
                and not prompt.startswith(_INTERNAL_PROMPT_PREFIXES)
                and prompt not in prompt_list
            ):
                prompt_list.append(prompt)
        item_match = _ASSISTANT_ITEM_RE.search(body)
        if item_match and item_match.group(1) not in assistant_item_ids:
            assistant_item_ids.append(item_match.group(1))
        summary_match = _REASONING_SUMMARY_RE.search(body)
        if not summary_match:
            continue
        try:
            values = json.loads(summary_match.group(1))
        except json.JSONDecodeError:
            continue
        for value in values:
            cleaned = str(value).strip("* ")
            if cleaned and cleaned not in summaries:
                summaries.append(cleaned)

    evidence_text = "\n".join([thread_id, *prompt_list, *summaries])
    score = _score(query, evidence_text)
    if thread_id.lower() in query.lower():
        score = max(score, 0.98)
    if score <= 0:
        return None
    sections = [
        f"Recovered partial Codex task {thread_id}.",
        "Exact retained user prompts:",
        *[f"- {prompt}" for prompt in prompt_list[:8]],
    ]
    if summaries:
        sections.extend(
            [
                "Retained assistant reasoning-summary labels:",
                *[f"- {summary}" for summary in summaries[-16:]],
            ]
        )
    if assistant_item_ids:
        sections.append(f"Final retained assistant item ID: {assistant_item_ids[-1]}.")
    sections.append(
        "The literal assistant response text is unavailable in telemetry; "
        "do not quote this partial recovery as a verbatim assistant message."
    )
    recorded_at = (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(first_ts))
        if first_ts is not None
        else None
    )
    return CodexHistoryResult(
        thread_id=thread_id,
        role="mixed_partial",
        text="\n".join(sections)[:8_000],
        score=score,
        source_pointer=f"{source_db}#thread={thread_id}",
        source_span="telemetry-task-summary",
        evidence_class="telemetry_partial_no_assistant_text",
        recorded_at=recorded_at,
        assistant_text_available=False,
    )


def _desktop_reasoning_summaries(
    thread_ids: Iterable[str],
    *,
    logs_root: Path | None,
    max_age_days: int,
    max_bytes: int = 32 * 1024 * 1024,
    max_files: int = 200,
    max_entries: int = 2_000,
) -> dict[str, list[str]]:
    wanted = set(thread_ids)
    if (
        not wanted
        or logs_root is None
        or not logs_root.exists()
        or max_files <= 0
        or max_entries <= 0
    ):
        return {}
    cutoff = time.time() - max_age_days * 86_400
    files: list[tuple[float, Path]] = []
    pending_directories = [logs_root]
    entries_seen = 0
    while (
        pending_directories
        and len(files) < max_files
        and entries_seen < max_entries
    ):
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > max_entries:
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending_directories.append(Path(entry.path))
                        elif (
                            entry.name.endswith(".log")
                            and entry.is_file(follow_symlinks=False)
                        ):
                            modified_at = entry.stat(follow_symlinks=False).st_mtime
                            if modified_at >= cutoff:
                                files.append((modified_at, Path(entry.path)))
                                if len(files) >= max_files:
                                    break
                    except OSError:
                        continue
        except OSError:
            continue
    files.sort(key=lambda item: item[0], reverse=True)
    summaries: dict[str, list[str]] = {}
    bytes_read = 0
    for _, path in files:
        if bytes_read >= max_bytes:
            break
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    bytes_read += len(line.encode("utf-8", errors="ignore"))
                    if bytes_read > max_bytes:
                        break
                    if "Reasoning summary item completed" not in line:
                        continue
                    thread_id = next(
                        (candidate for candidate in wanted if candidate in line),
                        None,
                    )
                    if thread_id is None:
                        continue
                    match = _REASONING_SUMMARY_RE.search(line)
                    if not match:
                        continue
                    try:
                        values = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        continue
                    collected = summaries.setdefault(thread_id, [])
                    for value in values:
                        cleaned = str(value).strip("* ")
                        if cleaned and cleaned not in collected:
                            collected.append(cleaned)
        except OSError:
            continue
    return summaries


def _search_telemetry(
    query: str,
    *,
    logs_db: Path,
    limit: int,
    max_age_days: int,
    desktop_logs_root: Path | None = None,
) -> list[CodexHistoryResult]:
    if not logs_db.exists():
        return []
    try:
        connection = sqlite3.connect(
            f"file:{logs_db}?mode=ro",
            uri=True,
            timeout=1.0,
        )
    except sqlite3.Error:
        return []
    try:
        connection.execute("PRAGMA query_only=ON")
        cutoff = int(time.time()) - max_age_days * 86_400
        threads = _matching_telemetry_threads(
            connection,
            query=query,
            cutoff=cutoff,
        )
        desktop_summaries = _desktop_reasoning_summaries(
            threads,
            logs_root=desktop_logs_root,
            max_age_days=max_age_days,
        )
        results = [
            result
            for thread_id, prompts in threads.items()
            if (
                result := _telemetry_task_result(
                    connection,
                    query=query,
                    thread_id=thread_id,
                    prompts=prompts,
                    source_db=logs_db,
                    cutoff=cutoff,
                    desktop_summaries=desktop_summaries.get(thread_id, ()),
                )
            )
            is not None
        ]
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def search_codex_history(
    query: str,
    *,
    limit: int = 5,
    codex_home: Path | None = None,
    max_files: int = 300,
    max_age_days: int = 30,
    max_bytes: int = 64 * 1024 * 1024,
) -> list[CodexHistoryResult]:
    """Search recent Codex rollouts and telemetry under strict local bounds."""

    if not query.strip() or not should_search_task_history(query):
        return []
    # A rollout can contain several matching messages from one task. Fetch a
    # small bounded surplus so those messages cannot consume the caller's whole
    # task-identity limit before the final identity deduplication below.
    candidate_limit = min(max(limit * 5, 5), 25)
    root = codex_home or _codex_home()
    rollout_results = _search_rollouts(
        query,
        sessions_root=root / "sessions",
        limit=candidate_limit,
        max_files=max_files,
        max_age_days=max_age_days,
        max_bytes=max_bytes,
    )
    telemetry_results = _search_telemetry(
        query,
        logs_db=root / "logs_2.sqlite",
        limit=candidate_limit,
        max_age_days=max_age_days,
        desktop_logs_root=(
            Path.home() / "Library" / "Logs" / "com.openai.codex"
            if root == _codex_home()
            else None
        ),
    )
    combined = rollout_results + telemetry_results
    combined.sort(
        key=lambda item: (
            item.score,
            item.evidence_class == "saved_rollout_message",
        ),
        reverse=True,
    )
    deduplicated: list[CodexHistoryResult] = []
    seen: set[str] = set()
    for result in combined:
        key = result.thread_id
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(result)
        if len(deduplicated) >= limit:
            break
    return deduplicated
