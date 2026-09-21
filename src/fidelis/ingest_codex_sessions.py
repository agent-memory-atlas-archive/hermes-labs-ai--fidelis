"""Local Codex session ingestion for Fidelis.

Native rollout JSONL files preserve user/assistant messages and structured tool
events. Codex desktop prompt history is a bounded fallback for side chats that
have no native rollout; those records are explicitly labelled USER_PROMPTS_ONLY.

Nothing runs automatically. The caller chooses Codex ingestion explicitly via
``fidelis sessions ingest --source codex`` (or ``--source all``). Prompt-history
fallbacks require the additional ``--include-prompt-history`` flag.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from fidelis.ingest_claude_sessions import (
    COGITO_SESSIONS_DIR,
    _excluded,
    _get_collection,
    _included,
    _load_exclude,
    _load_include,
    _load_ledger,
    _load_mode,
    _save_ledger,
    _store_session,
)

CODEX_SESSIONS = Path.home() / ".codex" / "sessions"
CODEX_ARCHIVED_SESSIONS = Path.home() / ".codex" / "archived_sessions"
CODEX_GLOBAL_STATE = Path.home() / ".codex" / ".codex-global-state.json"
MAX_TURNS_PER_SESSION = 200
MAX_ACTIONS_PER_SESSION = 500
THREAD_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


@dataclass(frozen=True)
class CodexSession:
    session_id: str
    project_path: str
    turns: list[dict]
    actions: list[dict]
    capture_completeness: str
    source_path: str
    source_sha256: str = ""


def _extract_message_text(content) -> str:
    """Extract visible text from a Codex response_item message."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text", "text"}:
            text = item.get("text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


def _output_fingerprint(output) -> tuple[int, str]:
    """Return size and hash without retaining possibly sensitive tool output."""
    if isinstance(output, str):
        payload = output.encode("utf-8", errors="replace")
    else:
        payload = json.dumps(
            output, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
    return len(payload), hashlib.sha256(payload).hexdigest()


def _parse_codex_jsonl(path: Path) -> CodexSession | None:
    """Parse one native Codex rollout into messages plus bounded action metadata."""
    path_match = THREAD_ID_RE.search(path.name)
    session_id = path_match.group(0) if path_match else ""
    project_path = "codex-unassigned"
    metadata_seen = False
    turns: list[dict] = []
    actions: list[dict] = []
    action_by_call_id: dict[str, dict] = {}
    truncated = False
    source_hasher = hashlib.sha256()

    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                source_hasher.update(raw_line)
                line = raw_line.decode("utf-8", errors="replace")
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                record_type = obj.get("type")
                payload = obj.get("payload")
                if not isinstance(payload, dict):
                    continue

                if record_type == "session_meta":
                    if not metadata_seen:
                        if not session_id:
                            session_id = str(
                                payload.get("id") or payload.get("session_id") or ""
                            )
                        project_path = str(
                            payload.get("cwd") or "codex-unassigned"
                        )
                        metadata_seen = True
                    continue

                if record_type != "response_item":
                    continue
                payload_type = payload.get("type")
                timestamp = str(obj.get("timestamp") or "")

                if payload_type == "message":
                    role = payload.get("role")
                    if role not in {"user", "assistant"}:
                        continue
                    text = _extract_message_text(payload.get("content"))
                    if not text:
                        continue
                    if len(turns) >= MAX_TURNS_PER_SESSION:
                        truncated = True
                        continue
                    turn = {"role": role, "content": text, "ts": timestamp}
                    phase = payload.get("phase")
                    if isinstance(phase, str) and phase:
                        turn["phase"] = phase
                    turns.append(turn)
                    continue

                if payload_type in {"custom_tool_call", "function_call"}:
                    if len(actions) >= MAX_ACTIONS_PER_SESSION:
                        truncated = True
                        continue
                    call_id = str(payload.get("call_id") or payload.get("id") or "")
                    action = {
                        "type": "tool_call",
                        "name": str(payload.get("name") or "unknown"),
                        "call_id": call_id,
                        "status": str(payload.get("status") or "recorded"),
                        "ts": timestamp,
                    }
                    actions.append(action)
                    if call_id:
                        action_by_call_id[call_id] = action
                    continue

                if payload_type in {"custom_tool_call_output", "function_call_output"}:
                    call_id = str(payload.get("call_id") or "")
                    action = action_by_call_id.get(call_id)
                    if action is not None:
                        size, digest = _output_fingerprint(payload.get("output"))
                        action["result"] = "recorded"
                        action["output_bytes"] = size
                        action["output_sha256"] = digest
    except (OSError, PermissionError):
        return None

    if not session_id or not turns:
        return None
    return CodexSession(
        session_id=session_id,
        project_path=project_path,
        turns=turns,
        actions=actions,
        capture_completeness=(
            "NATIVE_MESSAGES_TRUNCATED" if truncated else "NATIVE_MESSAGES"
        ),
        source_path=str(path),
        source_sha256=source_hasher.hexdigest(),
    )


def _native_paths(since: datetime | None = None) -> Iterator[Path]:
    """Yield active and archived rollout files, newest first."""
    paths: list[Path] = []
    if CODEX_SESSIONS.exists():
        paths.extend(CODEX_SESSIONS.rglob("*.jsonl"))
    if CODEX_ARCHIVED_SESSIONS.exists():
        paths.extend(CODEX_ARCHIVED_SESSIONS.glob("*.jsonl"))
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths:
        if since is not None:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < since:
                continue
        yield path


def _peek_codex_identity(path: Path) -> tuple[str, str] | None:
    """Read only the leading session metadata used for pre-ingest filtering."""
    path_match = THREAD_ID_RE.search(path.name)
    path_id = path_match.group(0) if path_match else ""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if obj.get("type") == "response_item":
                    return None
                if obj.get("type") != "session_meta":
                    continue
                payload = obj.get("payload")
                if not isinstance(payload, dict):
                    return None
                session_id = path_id or str(
                    payload.get("id") or payload.get("session_id") or ""
                )
                if not THREAD_ID_RE.fullmatch(session_id):
                    return None
                return session_id, str(payload.get("cwd") or "codex-unassigned")
    except (OSError, PermissionError):
        return None
    return None


def _selected_subject(
    project_path: str,
    session_id: str,
    mode: str,
    include,
    exclude,
) -> bool:
    subject = f"{project_path} {session_id}"
    if mode == "opt-in":
        return _included(subject, include)
    return not _excluded(subject, exclude)


def _native_sessions(
    since: datetime | None = None,
    *,
    mode: str = "opt-out",
    include=None,
    exclude=None,
    thread_id: str | None = None,
) -> dict[str, CodexSession]:
    """Return one newest native record per exact Codex thread ID."""
    sessions: dict[str, CodexSession] = {}
    for path in _native_paths(since=since):
        identity = _peek_codex_identity(path)
        if identity is None:
            continue
        candidate_id, project_path = identity
        if thread_id is not None and candidate_id != thread_id:
            continue
        if not _selected_subject(
            project_path,
            candidate_id,
            mode,
            include or [],
            exclude or [],
        ):
            continue
        parsed = _parse_codex_jsonl(path)
        if parsed and parsed.session_id not in sessions:
            sessions[parsed.session_id] = parsed
    return sessions


def _native_thread_ids() -> set[str]:
    """Resolve IDs from every native path without parsing private message text."""
    ids = set()
    for path in _native_paths(since=None):
        match = THREAD_ID_RE.search(path.name)
        if match:
            ids.add(match.group(0))
    return ids


def _prompt_history_sessions(
    native_ids: set[str],
    since: datetime | None = None,
    *,
    mode: str = "opt-out",
    include=None,
    exclude=None,
    exact_thread_id: str | None = None,
) -> dict[str, CodexSession]:
    """Read user-only desktop prompt histories not represented by native JSONL.

    Codex desktop does not expose per-prompt timestamps here. ``since`` can only
    be applied to the state file as a whole; the completeness label keeps this
    limitation visible after ingestion.
    """
    if not CODEX_GLOBAL_STATE.exists():
        return {}
    if since is not None:
        modified = datetime.fromtimestamp(
            CODEX_GLOBAL_STATE.stat().st_mtime, tz=timezone.utc
        )
        if modified < since:
            return {}
    try:
        state = json.loads(CODEX_GLOBAL_STATE.read_text())
    except (OSError, PermissionError, json.JSONDecodeError):
        return {}

    histories = (
        state.get("electron-persisted-atom-state", {}).get("prompt-history", {})
    )
    assignments = state.get("thread-project-assignments", {})
    if not isinstance(histories, dict):
        return {}

    sessions: dict[str, CodexSession] = {}
    for candidate_id, prompts in histories.items():
        if (
            not isinstance(candidate_id, str)
            or not THREAD_ID_RE.fullmatch(candidate_id)
            or candidate_id in native_ids
            or not isinstance(prompts, list)
        ):
            continue
        if exact_thread_id is not None and candidate_id != exact_thread_id:
            continue
        assignment = assignments.get(candidate_id, {})
        project_path = (
            str(assignment.get("cwd") or "codex-unassigned")
            if isinstance(assignment, dict)
            else "codex-unassigned"
        )
        if not _selected_subject(
            project_path,
            candidate_id,
            mode,
            include or [],
            exclude or [],
        ):
            continue
        prompt_texts = [
            prompt for prompt in prompts
            if isinstance(prompt, str) and prompt.strip()
        ]
        if not prompt_texts:
            continue
        truncated = len(prompt_texts) > MAX_TURNS_PER_SESSION
        retained_prompts = prompt_texts[-MAX_TURNS_PER_SESSION:]
        turns = [
            {"role": "user", "content": prompt, "ts": ""}
            for prompt in retained_prompts
        ]
        prompt_digest = hashlib.sha256(
            json.dumps(
                prompt_texts,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        sessions[candidate_id] = CodexSession(
            session_id=candidate_id,
            project_path=project_path,
            turns=turns,
            actions=[],
            capture_completeness=(
                "USER_PROMPTS_ONLY_TRUNCATED"
                if truncated
                else "USER_PROMPTS_ONLY"
            ),
            source_path=str(CODEX_GLOBAL_STATE),
            source_sha256=prompt_digest,
        )
    return sessions


def _selected(session: CodexSession, mode: str, include, exclude) -> bool:
    """Apply the existing project privacy posture to Codex cwd plus thread ID."""
    return _selected_subject(
        session.project_path,
        session.session_id,
        mode,
        include,
        exclude,
    )


def _stable_chroma_id(session_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fidelis:codex:{session_id}"))


def _make_codex_ingest_hash(session: CodexSession) -> str:
    """Hash the full bounded Codex record so later turns/actions trigger update."""
    payload = {
        "session_id": session.session_id,
        "capture_completeness": session.capture_completeness,
        "turns": [
            {
                "role": turn.get("role", ""),
                "content": turn.get("content", ""),
                "phase": turn.get("phase", ""),
            }
            for turn in session.turns
        ],
        "actions": session.actions,
        "source_sha256": session.source_sha256,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def ingest(
    since: datetime | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    exclude: list[str] | None = None,
    mode: str | None = None,
    include: list[str] | None = None,
    include_prompt_history: bool = False,
    thread_id: str | None = None,
) -> dict:
    """Ingest native Codex sessions and optional user-only side-chat fallbacks."""
    mode = _load_mode() if mode is None else mode
    exclude = _load_exclude() if exclude is None else exclude
    include = _load_include() if include is None else include
    if thread_id is not None and not THREAD_ID_RE.fullmatch(thread_id):
        raise ValueError("Codex thread ID must be a UUID")
    if include_prompt_history and since is not None:
        raise ValueError(
            "prompt-history fallback requires an all-history run because "
            "Codex desktop does not preserve per-prompt timestamps"
        )
    if mode == "opt-in" and not include:
        print(
            "opt-in mode: no projects included yet — set FIDELIS_SESSIONS_INCLUDE=... "
            "to choose what to index"
        )
        return {
            "scanned": 0, "skipped_dedup": 0, "skipped_empty": 0,
            "stored": 0, "updated": 0, "errors": 0,
        }

    native = _native_sessions(
        since=since,
        mode=mode,
        include=include,
        exclude=exclude,
        thread_id=thread_id,
    )
    sessions = dict(native)
    if include_prompt_history:
        sessions.update(
            _prompt_history_sessions(
                set(native) | _native_thread_ids(),
                since=since,
                mode=mode,
                include=include,
                exclude=exclude,
                exact_thread_id=thread_id,
            )
        )
    # Loaders apply these gates before extracting message text. Repeat the
    # predicate here as defense in depth for injected/custom loader results.
    candidates = [
        session
        for session in sessions.values()
        if (thread_id is None or session.session_id == thread_id)
        and _selected(session, mode=mode, include=include, exclude=exclude)
    ]
    candidates.sort(key=lambda session: session.session_id)

    label = "would index" if dry_run else "Indexing"
    fallback_count = sum(
        session.capture_completeness.startswith("USER_PROMPTS_ONLY")
        for session in candidates
    )
    print(
        f"{label} {len(candidates)} Codex sessions "
        f"({fallback_count} user-prompt-only fallbacks)…"
    )

    ledger = _load_ledger()
    col = None if dry_run else _get_collection()
    if not dry_run:
        try:
            os.chmod(Path.home() / ".cogito", 0o700)
        except OSError:
            pass
    stats = {
        "scanned": 0, "skipped_dedup": 0, "skipped_empty": 0,
        "stored": 0, "updated": 0, "errors": 0,
    }

    for session in candidates:
        stats["scanned"] += 1
        if not session.turns:
            stats["skipped_empty"] += 1
            continue
        ingest_hash = _make_codex_ingest_hash(session)
        if ingest_hash in ledger:
            stats["skipped_dedup"] += 1
            continue

        chroma_id = _stable_chroma_id(session.session_id)
        previous_hashes = [key for key, value in ledger.items() if value == chroma_id]
        if dry_run:
            stats["updated" if previous_hashes else "stored"] += 1
            if verbose:
                print(
                    f"  DRY-RUN {session.session_id[:8]} "
                    f"{session.capture_completeness} {len(session.turns)} turns"
                )
            continue
        try:
            _store_session(
                col,
                session_id=session.session_id,
                project_path=session.project_path,
                turns=session.turns,
                ingest_hash=ingest_hash,
                session_source="codex",
                capture_completeness=session.capture_completeness,
                actions=session.actions,
                action_trace_level="METADATA_ONLY" if session.actions else "NONE",
                source_locator=session.source_path,
                chroma_id=chroma_id,
                upsert=True,
            )
            for previous_hash in previous_hashes:
                ledger.pop(previous_hash, None)
            ledger[ingest_hash] = chroma_id
            stats["updated" if previous_hashes else "stored"] += 1
        except Exception as exc:
            stats["errors"] += 1
            if verbose:
                print(f"  ERROR {session.session_id[:8]}: {exc}")

    if not dry_run:
        COGITO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _save_ledger(ledger)
    return stats
