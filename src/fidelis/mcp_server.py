"""Minimal MCP server bundled with fidelis — exposes recall + health as MCP tools.

Implements the JSON-RPC stdio transport for the Claude Code MCP protocol. Relies
on a running fidelis-server (port 19420 by default) for the actual recall.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

from fidelis import __version__
from fidelis.context import context_packet, plan_context
from fidelis.evidence_status import (
    MCP_MAX_BYTES,
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
)
from fidelis.evidence_status import (
    bound_mcp_envelope,
    build_evidence_envelope,
    compact_json,
    evidence_status_enabled,
)
from fidelis.relation_envelope import (
    DECLARATIONS_FIELD,
    EXTRACTOR_VERSION as RELATION_EXTRACTOR_VERSION,
    RELATION_SEMANTICS_VERSION,
    SCHEMA_VERSION as RELATION_SCHEMA_VERSION,
    SUPPORTED_RELATION_TYPES,
    relation_envelopes_enabled,
)

_MCP_LEGACY_PROTOCOL = "2024-11-05"
_MCP_STRUCTURED_PROTOCOL = "2025-06-18"
_MCP_BATCH_PROTOCOL = "2025-03-26"
_SUPPORTED_PROTOCOLS = (_MCP_STRUCTURED_PROTOCOL, _MCP_BATCH_PROTOCOL, _MCP_LEGACY_PROTOCOL)
_negotiated_protocol = _MCP_LEGACY_PROTOCOL
_STORE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_MAX_STORE_METADATA_BYTES = 16_384
_MAX_RELATION_DECLARATIONS = 16


def _exit_when_orphaned(poll_secs: float = 5.0) -> None:
    """stdin EOF is not reliable when the parent is SIGKILLed — the blocking
    read never returns and the process leaks. Reparenting to PID 1 is the
    unambiguous parent-death signal."""
    while True:
        if os.getppid() == 1:
            os._exit(0)
        time.sleep(poll_secs)


def _server_url() -> str:
    port = os.environ.get("FIDELIS_PORT", os.environ.get("COGITO_PORT", "19420"))
    return f"http://127.0.0.1:{port}"


def _host_identity(arguments: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    if arguments.get("session_id"):
        result["session_id"] = str(arguments["session_id"])[:128]
    if arguments.get("turn_id"):
        result["turn_id"] = str(arguments["turn_id"])[:128]
    return result


class _ToolError(str):
    """A tool-result string that must be surfaced to the client as isError:true.

    Subclasses str so every existing text-equality assertion, alias, and
    truncation/rendering path keeps working byte-for-byte; only the dispatcher
    treats it differently (sets the JSON-RPC result's isError flag)."""


# Best-effort record of the most recent backend call within one tools/call
# dispatch, for the stderr log line only. The stdio loop is sequential (one
# request handled at a time), so a plain module-level dict is sufficient.
_LAST_BACKEND_CALL: dict[str, object] = {"path": None, "status": None}

# Diagnostic logging must never be able to stall the request/response path
# (verifier finding, 2026-09-21): a client that pipes stderr without
# draining it fills the OS pipe, and a blocking write then hangs the whole
# single-threaded dispatch loop -- worse than the crash this file fixes
# elsewhere. _make_stderr_nonblocking() (called once from main(), never at
# import time, so it cannot change how pytest's own stderr behaves for
# in-process unit tests) puts fd 2 in non-blocking mode; _write_log_line()
# below then does a raw, best-effort os.write and silently drops the line
# on any failure instead of blocking or raising.
_LOG_STATE = {"dropped": 0}


def _make_stderr_nonblocking() -> None:
    try:
        os.set_blocking(2, False)
    except (OSError, ValueError):  # noqa: not fatal -- logging just stays best-effort
        pass


def _write_log_line(line: str) -> bool:
    """Best-effort, non-blocking single write of one line to fd 2.

    A line this short (see the 300-char detail cap below) is well under the
    POSIX PIPE_BUF guarantee (>=512 bytes), so on a pipe this write either
    fully succeeds or raises EAGAIN -- never a corrupting partial write."""
    data = (line + "\n").encode("utf-8", errors="replace")
    try:
        os.write(2, data)
        return True
    except (BlockingIOError, OSError):
        return False


def _log_call(
    tool_or_method: str,
    outcome: str,
    elapsed_ms: float,
    *,
    backend_path: str | None = None,
    http_status: int | None = None,
    detail: str | None = None,
) -> None:
    """One diagnostic line to stderr per tool call or protocol error.

    Format: ts pid tool-or-method outcome-class elapsed-ms backend-path
    http-status. Never logs query text, memory text, or call arguments —
    stdout is the only protocol channel and stderr carries identifiers and
    timings only. Never blocks and never raises: a full/absent reader on
    the other end of stderr silently drops log lines instead of stalling a
    response."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fields = [
        ts,
        f"pid={os.getpid()}",
        str(tool_or_method or "?"),
        outcome,
        f"{elapsed_ms:.1f}ms",
        backend_path or "-",
        str(http_status) if http_status is not None else "-",
    ]
    if _LOG_STATE["dropped"] and _write_log_line(
        f"({_LOG_STATE['dropped']} earlier log lines dropped -- reader was not draining stderr)"
    ):
        _LOG_STATE["dropped"] = 0
    if not _write_log_line(" ".join(fields)):
        _LOG_STATE["dropped"] += 1
    if detail and not _write_log_line(f"    detail: {detail[:300]}"):
        _LOG_STATE["dropped"] += 1


def _http_post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_server_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    _LAST_BACKEND_CALL["path"] = path
    _LAST_BACKEND_CALL["status"] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _LAST_BACKEND_CALL["status"] = resp.status
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # The server answered: a 400/422 carries a JSON verdict (invalid
        # declaration, write-gate rejection). Reporting it as "unreachable"
        # would make a deliberate refusal look like an outage.
        _LAST_BACKEND_CALL["status"] = e.code
        try:
            body = json.loads(e.read())
        except Exception:  # noqa: non-JSON error body
            body = {}
        if not isinstance(body, dict):
            body = {}
        body.setdefault("http_status", e.code)
        if "error" not in body and body.get("status") != "rejected":
            body["error"] = f"fidelis-server returned HTTP {e.code}"
        return body
    except (socket.timeout, TimeoutError):
        # A read timeout is TimeoutError/socket.timeout, not a URLError
        # subclass — it must be caught separately or it escapes uncaught.
        return {
            "error": f"fidelis-server timed out after {timeout}s calling {path}"
        }
    except urllib.error.URLError as e:
        return {"error": f"fidelis-server unreachable at {_server_url()}: {e}"}


def _http_get(path: str, timeout: float = 5.0) -> dict:
    _LAST_BACKEND_CALL["path"] = path
    _LAST_BACKEND_CALL["status"] = None
    try:
        with urllib.request.urlopen(f"{_server_url()}{path}", timeout=timeout) as resp:
            _LAST_BACKEND_CALL["status"] = resp.status
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Mirror _http_post: a structured error body (e.g. the 503 the
        # server answers while still loading) is a verdict, not an outage.
        _LAST_BACKEND_CALL["status"] = e.code
        try:
            body = json.loads(e.read())
        except Exception:  # noqa: non-JSON error body
            body = {}
        if not isinstance(body, dict):
            body = {}
        body.setdefault("http_status", e.code)
        body.setdefault("error", f"fidelis-server returned HTTP {e.code}")
        return body
    except (socket.timeout, TimeoutError):
        return {
            "error": f"fidelis-server timed out after {timeout}s calling {path}"
        }
    except urllib.error.URLError as e:
        return {"error": f"fidelis-server unreachable at {_server_url()}: {e}"}


# PACKET F3 -- MCP surface v2 (real-user panel synthesis + research/memory-
# mcp-surface.md + MCP-PRD.md D1-D6). Removed from the agent-facing surface,
# with the reason each was removed:
#   - all cogito_* aliases: byte-identical duplicates of their fidelis_*
#     counterparts (confirmed by four independent testers); pure clutter.
#   - fidelis_query: folded into fidelis_recall's default ("fast") mode --
#     uses the direct vector retrieval path.
#   - fidelis_inquire: testers found it abstains on natural questions and
#     labels zero-overlap records "direct support" -- "more dangerous than
#     silence" than a plain miss.
#   - fidelis_orient: silently misclassifies "needs memory" turns as
#     not_needed and drops ids/status markers a caller needs to act safely.
#   - fidelis_capabilities: folds into fidelis_health (no agent decision
#     depended on it that fidelis_health cannot answer).
# A removed name is never silently repointed at new behaviour (PRD D4: "a
# name never changes meaning") -- calling one returns -32602 naming its
# replacement, handled in _handle below.
_REMOVED_TOOLS: dict[str, str] = {
    "fidelis_query": "fidelis_recall (default mode is the fast path)",
    "fidelis_inquire": "fidelis_recall",
    "fidelis_orient": "fidelis_recall",
    "fidelis_capabilities": "fidelis_health",
    "cogito_recall": "fidelis_recall",
    "cogito_orient": "fidelis_recall",
    "cogito_inquire": "fidelis_recall",
    "cogito_query": "fidelis_recall",
    "cogito_add": "fidelis_store",
    "cogito_health": "fidelis_health",
    "cogito_capabilities": "fidelis_health",
}
# cogito_recall_sessions / cogito_recall_both stay a plain "unknown tool"
# (no entry here): they have no direct replacement -- PRD D5 brings sessions
# recall back later only as an HTTP proxy, never as a tool name again.

_READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
_APPEND_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}

_STORE_METADATA_SCHEMA = {
    "type": "object",
    "description": (
        "Optional provenance metadata with at most 16 "
        "relation_declarations_v2 entries"
    ),
    "properties": {
        "relation_declarations_v2": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(SUPPORTED_RELATION_TYPES),
                    },
                    "target_record_id": {"type": "string"},
                    "support_text": {"type": "string"},
                    "declared_by": {
                        "type": "string",
                        "enum": ["caller", "source"],
                    },
                },
                "required": [
                    "type",
                    "target_record_id",
                    "support_text",
                    "declared_by",
                ],
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": True,
}


def _fidelis_store_tool() -> dict:
    """fidelis_store's schema. The `id` argument is listed only when it is
    actually honoured in this deployment's default configuration (relation
    envelopes on) -- otherwise it would silently be ignored, exactly the
    bug T4 found (docs/TIME-AWARE-SPEC.md, PACKET F1c)."""
    properties: dict[str, object] = {}
    if relation_envelopes_enabled():
        properties["id"] = {
            "type": "string",
            "minLength": 3,
            "maxLength": 128,
            "description": "Optional stable record identifier.",
        }
    properties.update(
        {
            "text": {
                "type": "string",
                "description": "Atomic fact to store verbatim.",
            },
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 64,
                "description": (
                    "Ids this memory replaces (from a prior fidelis_recall "
                    "or fidelis_get result). Each must exist for this user "
                    "or the write is rejected."
                ),
            },
            "valid_from": {
                "type": "string",
                "description": "Optional ISO-8601 start of validity (inclusive).",
            },
            "valid_to": {
                "type": "string",
                "description": "Optional ISO-8601 end of validity (exclusive).",
            },
            "event_at": {
                "type": "string",
                "description": "Optional ISO-8601 time the described thing happened.",
            },
            "source": {
                "type": "string",
                "maxLength": 512,
                "description": "Optional provenance, e.g. a URL or file path.",
            },
            "metadata": _STORE_METADATA_SCHEMA,
        }
    )
    return {
        "name": "fidelis_store",
        "description": (
            "Remember or store one new, self-contained memory verbatim -- "
            "exact text, not a summary. Optional valid_from/valid_to/"
            "event_at/source, and supersedes (ids this replaces; unknown "
            "or another user's ids are rejected). The reply echoes "
            "recorded_at and any declared window so you can confirm "
            "without a recall. Do not use this to correct something "
            "already in memory -- use fidelis_correct with the old id "
            "instead of storing a near-duplicate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["text"],
        },
        "annotations": _APPEND_ANNOTATIONS,
    }


def _build_tools_v2() -> list[dict]:
    """The v2 agent-facing tool inventory (PACKET F3).

    PRD D2: one schema for every negotiated protocol version -- the legacy
    2024-11-05 inventory freeze is lifted once, deliberately, here; both
    "2024-11-05" and "2025-06-18" clients see the identical list (D3: only
    whether `structuredContent` accompanies a call's result differs by
    version, handled in _handle, not here).

    Built fresh per call, not cached in a module global, because two things
    legitimately vary at runtime and must be read at tools/list time, not
    import time, or a long-lived process could serve a stale inventory
    after the environment changes underneath it:
      - COGITO_RELATION_ENVELOPES_V1 (fidelis_store's `id` argument, F1c)
      - FIDELIS_ORIENT_ROUTE (whether fidelis_route is listed at all, D4)
    """
    tools = [
        {
            "name": "fidelis_recall",
            "description": (
                "Search stored memories for `query`. Call before "
                "answering questions about prior work, decisions, or past "
                'conversations. Default mode="fast" (vector search, '
                '~0.1s); use mode="thorough" only when a fast result '
                "looks wrong or stakes are high (~3s, hybrid search). "
                "Current facts list first, superseded/expired after, "
                "marked. Every hit shows its id, status, and recorded "
                "date. Fix a stale hit with fidelis_correct; see its "
                "history with fidelis_get. Do not use this to store a "
                "new fact -- use fidelis_store instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max hits to return.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "thorough"],
                        "default": "fast",
                        "description": (
                            '"fast": vector search (default). "thorough": '
                            "hybrid search, slower, for a second opinion."
                        ),
                    },
                    "as_of": {
                        "type": "string",
                        "description": (
                            "Optional ISO-8601 date or instant. Answer as "
                            "memory stood then: records written later are "
                            "excluded and later supersessions are ignored."
                        ),
                    },
                    "historical": {
                        "type": "boolean",
                        "description": (
                            "Keep relevance order; do not demote superseded "
                            "or expired records."
                        ),
                    },
                },
                "required": ["query"],
            },
            "annotations": _READ_ONLY_ANNOTATIONS,
        },
        _fidelis_store_tool(),
        {
            "name": "fidelis_correct",
            "description": (
                "Correct a memory that is wrong or outdated, in one call: "
                "stores `text` as a new record superseding `id`. The old "
                "record is kept, marked superseded, and stays visible in "
                "fidelis_get's history -- nothing is deleted. `id` must "
                "come from a prior fidelis_recall or fidelis_get result; an "
                "unknown id is rejected. If `id` is already superseded, "
                "this reports the current replacement instead of writing, "
                "unless force=true. Do not use this for a brand-new, "
                "unrelated fact -- use fidelis_store instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": (
                            "Id of the record being corrected (from a prior "
                            "fidelis_recall or fidelis_get result)."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "The corrected fact, verbatim.",
                    },
                    "valid_from": {
                        "type": "string",
                        "description": "Optional ISO-8601 start of validity (inclusive).",
                    },
                    "valid_to": {
                        "type": "string",
                        "description": "Optional ISO-8601 end of validity (exclusive).",
                    },
                    "event_at": {
                        "type": "string",
                        "description": "Optional ISO-8601 time the described thing happened.",
                    },
                    "source": {
                        "type": "string",
                        "maxLength": 512,
                        "description": "Optional provenance, e.g. a URL or file path.",
                    },
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Supersede `id` anyway even if something else "
                            "already superseded it."
                        ),
                    },
                },
                "required": ["id", "text"],
            },
            "annotations": _APPEND_ANNOTATIONS,
        },
        {
            "name": "fidelis_get",
            "description": (
                "Fetch exactly one memory by id, verbatim, with its "
                "correction chain in both directions: what it replaced "
                "(oldest last) and what replaced it (newest last). Use "
                'this to resolve a "SUPERSEDED by <id>" marker from '
                "fidelis_recall, or to confirm an id before passing it to "
                "fidelis_correct. An unknown id is an error. Do not use "
                "this to find a memory by topic -- use fidelis_recall "
                "instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Record id."},
                },
                "required": ["id"],
            },
            "annotations": _READ_ONLY_ANNOTATIONS,
        },
        {
            "name": "fidelis_recent",
            "description": (
                "List memories by recency, no search query needed -- good "
                'for "what changed recently" or catching up after time '
                "away. `since` (ISO-8601) is a lower bound: only records "
                'at or after it. kind="corrections" shows only memories '
                "that corrected an older one, with the id(s) they "
                "replaced. Do not use this to find a memory about a "
                "topic -- use fidelis_recall instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Max records to return.",
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Optional ISO-8601 instant; only records at or "
                            "after it."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["all", "corrections"],
                        "default": "all",
                        "description": (
                            '"all": every recent record. "corrections": '
                            "only records that superseded another."
                        ),
                    },
                },
            },
            "annotations": _READ_ONLY_ANNOTATIONS,
        },
        {
            "name": "fidelis_health",
            "description": (
                "Check whether the memory backend is up, and see "
                "store-wide stats (current/superseded/expired counts, "
                "oldest/newest record, queue and dead-letter depth) in "
                "one call. Call this after any other fidelis_* tool "
                "returns isError, before retrying. Do not call this to "
                "search memory -- use fidelis_recall for that."
            ),
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": _READ_ONLY_ANNOTATIONS,
        },
    ]
    return tools


# Static snapshot for introspection/`hasattr` convenience only (e.g.
# `import fidelis.mcp_server; "fidelis_recall" in {t["name"] for t in
# mcp_server.TOOLS}`). The wire protocol (`tools/list`, in _handle below)
# always calls _build_tools_v2() fresh so an env change is reflected
# immediately, never this stale snapshot.
TOOLS = _build_tools_v2()


def _tool_recall_legacy(arguments: dict) -> str:
    query = arguments.get("query", "")
    limit = max(1, min(int(arguments.get("limit", 5)), 5))
    try:
        res = _http_post(
            "/orient",
            {
                "text": query,
                "limit": limit,
                "automatic": False,
                **_host_identity(arguments),
                **_temporal_args(arguments),
            },
        )
    except Exception:
        res = {"error": "orient raised"}
    # A 400 is the server's verdict on the request (e.g. invalid as_of); the
    # fallbacks would fail the same way, so surface it instead of retrying.
    if res.get("error") and res.get("http_status") != 400:
        try:
            res = _http_post(
                "/recall_hybrid",
                {"text": query, "limit": limit, "top_k": limit, "tier": "zero_llm",
                 **_temporal_args(arguments)},
            )
        except Exception:
            res = {"error": "recall_hybrid raised"}
    if res.get("error") and res.get("http_status") != 400:
        res = _http_post(
            "/recall", {"text": query, "limit": limit, **_temporal_args(arguments)}
        )
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    memories = res.get("memories", [])
    if not memories:
        return "no memories found."
    header = f"{len(memories)} memories"
    if arguments.get("as_of"):
        header += f" as of {arguments['as_of']}"
    lines = [f"{header}:"]
    for i, m in enumerate(memories[:limit], 1):
        text = m.get("text", "")
        score = m.get("score", "")
        label = _memory_evidence_label(m) + _temporal_label(m)
        lines.append(f"  [{i}] (score {score}){label} {text[:500]}")
    return "\n".join(lines)


def _legacy_result(res: dict, limit: int) -> dict:
    memories = res.get("memories", [])
    if not isinstance(memories, list):
        memories = []
    return {
        **res,
        "records": memories[:limit],
        "memories": memories[:limit],
        "retrieval_status": (
            res.get("retrieval_status")
            or ("ok" if memories else "no_supported_result_in_bounded_window")
        ),
        "shown_count": len(memories[:limit]),
        "candidate_count": len(memories),
        "plan": res.get("plan") or {"mode": "semantic", "initial_depth": limit},
    }


def _tool_recall_structured(arguments: dict) -> dict:
    query = arguments.get("query", "")
    limit = max(1, min(int(arguments.get("limit", 5)), 5))
    try:
        res = _http_post(
            "/orient",
            {
                "text": query,
                "limit": limit,
                "automatic": False,
                **_host_identity(arguments),
                **_temporal_args(arguments),
            },
        )
    except Exception:
        res = {"error": "orient raised"}
    if res.get("schema_version") == EVIDENCE_SCHEMA_VERSION:
        return bound_mcp_envelope(res)

    requested_path = "orient"
    actual_path = "orient"
    fallback_count = 0
    failure_class = None
    if res.get("error"):
        failure_class = "CanonicalOrientError"
        fallback_count += 1
        actual_path = "recall_hybrid"
        try:
            res = _http_post(
                "/recall_hybrid",
                {
                    "text": query,
                    "limit": limit,
                    "top_k": limit,
                    "tier": "zero_llm",
                    **_temporal_args(arguments),
                },
            )
        except Exception:
            res = {"error": "recall_hybrid raised"}
    if res.get("error"):
        failure_class = "HybridFallbackError"
        fallback_count += 1
        actual_path = "recall"
        try:
            res = _http_post(
                "/recall",
                {"text": query, "limit": limit, **_temporal_args(arguments)},
            )
        except Exception as exc:
            failure_class = type(exc).__name__
            actual_path = "none"
            res = {"error": "terminal recall fallback raised"}
    if res.get("error"):
        failure_class = failure_class or "AllRecallPathsError"
        actual_path = "none"

    health = [
        {
            "source": "atomic_memory",
            "requested_path": requested_path,
            "actual_path": actual_path,
            "state": (
                "error"
                if res.get("error")
                else "degraded_fallback"
                if fallback_count
                else "healthy"
            ),
            "failure_class": failure_class,
            "authority_limitation": (
                "No recall path returned evidence."
                if res.get("error")
                else "MCP used a non-canonical fallback path."
                if fallback_count
                else None
            ),
        }
    ]
    return bound_mcp_envelope(
        build_evidence_envelope(
            query=query,
            result=_legacy_result(res, limit),
            source_health=health,
        )
    )


def _tool_recall(arguments: dict) -> str | dict:
    if evidence_status_enabled():
        return _tool_recall_structured(arguments)
    return _tool_recall_legacy(arguments)


def _temporal_label(memory: dict) -> str:
    """Record id plus any non-current temporal state, for text renderings.

    The id is shown so an agent can supersede this record later; the marker
    is shown so a stale hit is never read as current.

    Also surfaces the LEGACY regex-pointers supersession marker
    (`fidelis.supersession.mark_superseded`'s `memory["supersession"]` dict)
    — a separate, older mechanism from the temporal system above. As of
    2026-09-21 the /query, /recall, /recall_b, /recall_hybrid response paths
    use `mark_superseded` (metadata-only) instead of `annotate_superseded`
    (which used to rewrite `text` to "[STATUS:id] note || original: <text>"),
    per docs/TIME-AWARE-SPEC.md invariant 2 (stored text is byte-identical
    in responses). This label is where that status/note become visible to
    the reader instead.
    """
    temporal = memory.get("temporal")
    legacy = memory.get("supersession")
    has_temporal_signal = isinstance(temporal, dict) and not (
        temporal.get("status") in (None, "current") and not temporal.get("recorded_at")
    )
    has_legacy_signal = isinstance(legacy, dict) and bool(legacy.get("status"))
    # Label only when there is something to say: a legacy row with no
    # recorded time, no temporal marker, and no legacy supersession pointer
    # renders byte-identically to pre-temporal output.
    if not has_temporal_signal and not has_legacy_signal:
        return ""
    parts = []
    if memory.get("id"):
        parts.append(f"id={memory['id']}")
    if has_temporal_signal:
        status = temporal.get("status")
        if status == "superseded":
            by = ", ".join(str(b) for b in temporal.get("superseded_by") or [])
            parts.append(f"SUPERSEDED by {by}" if by else "SUPERSEDED")
        elif status == "expired":
            parts.append(f"EXPIRED {temporal.get('valid_to')}")
        elif status == "not_yet_valid":
            parts.append(f"NOT YET VALID until {temporal.get('valid_from')}")
        if temporal.get("recorded_at"):
            parts.append(f"recorded {temporal['recorded_at']}")
        elif temporal.get("recorded_at_known") is False:
            parts.append("recorded: unknown")
    if has_legacy_signal:
        status = legacy.get("status") or "SUPERSEDED"
        rid = legacy.get("id")
        note = legacy.get("note")
        tag = f"{status}:{rid}" if rid else str(status)
        parts.append(f"{tag} — {note}" if note else tag)
    return f" [{'; '.join(parts)}]" if parts else ""


def _temporal_args(arguments: dict) -> dict:
    """Optional as_of / historical passthrough for recall-type tools."""
    out: dict = {}
    if arguments.get("as_of"):
        out["as_of"] = str(arguments["as_of"])
    if arguments.get("historical"):
        out["historical"] = True
    return out


def _memory_evidence_label(memory: dict) -> str:
    metadata = memory.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("source_kind") != "codex_task_history":
        return ""
    thread_id = str(metadata.get("thread_id") or "unknown")
    if metadata.get("assistant_text_available") is False:
        evidence = "Codex telemetry partial; assistant prose unavailable"
    else:
        evidence = "literal saved Codex message"
    return f" [{evidence}; task={thread_id}]"


def _tool_orient_legacy(arguments: dict) -> str:
    message = arguments.get("message", "")
    limit = max(1, min(int(arguments.get("limit", 5)), 5))
    recent_turns = arguments.get("recent_turns", [])
    if not isinstance(recent_turns, list):
        recent_turns = []
    res = _http_post(
        "/orient",
        {
            "text": message,
            "limit": limit,
            "automatic": True,
            "recent_turns": [str(turn) for turn in recent_turns[-4:]],
            **_host_identity(arguments),
        },
    )
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    if res.get("retrieval_status") == "not_needed":
        return "memory retrieval not needed."
    memories = res.get("memories", [])
    if not memories:
        return "memory retrieval ran; no supported result found."
    lines = [f"{len(memories)} memories ({res.get('method', 'cogito-hermeneutics')}):"]
    for i, memory in enumerate(memories[:limit], 1):
        label = _memory_evidence_label(memory)
        lines.append(f"  [{i}]{label} {str(memory.get('text', ''))[:500]}")
    return "\n".join(lines)


def _tool_orient_structured(arguments: dict) -> dict:
    message = arguments.get("message", "")
    limit = max(1, min(int(arguments.get("limit", 5)), 5))
    recent_turns = arguments.get("recent_turns", [])
    if not isinstance(recent_turns, list):
        recent_turns = []
    res = _http_post(
        "/orient",
        {
            "text": message,
            "limit": limit,
            "automatic": True,
            "recent_turns": [str(turn) for turn in recent_turns[-4:]],
            **_host_identity(arguments),
        },
    )
    if res.get("schema_version") == EVIDENCE_SCHEMA_VERSION:
        return bound_mcp_envelope(res)
    health = [
        {
            "source": "atomic_memory",
            "requested_path": "orient",
            "actual_path": "orient" if not res.get("error") else "none",
            "state": "healthy" if not res.get("error") else "error",
            "failure_class": "CanonicalOrientError" if res.get("error") else None,
            "authority_limitation": (
                "Canonical orientation did not return evidence."
                if res.get("error")
                else None
            ),
        }
    ]
    return bound_mcp_envelope(
        build_evidence_envelope(
            query=message,
            result=_legacy_result(res, limit),
            source_health=health,
            recent_turns=[str(turn) for turn in recent_turns[-4:]],
        )
    )


def _tool_orient(arguments: dict) -> str:
    utterance = str(arguments.get("utterance", ""))
    recent_turns = arguments.get("recent_turns", [])
    if not isinstance(recent_turns, list):
        recent_turns = []
    plan = plan_context(
        utterance,
        recent_turns=[str(turn) for turn in recent_turns[-4:]],
        entity_hint=arguments.get("entity"),
    )
    if plan.disposition == "abstain":
        return json.dumps(context_packet(plan, []), ensure_ascii=False)
    limit = max(1, min(int(arguments.get("limit", 5)), 20))
    res = _http_post(
        "/recall_hybrid",
        {"text": plan.retrieval_query, "limit": limit, "tier": "zero_llm"},
    )
    if res.get("error"):
        packet = context_packet(plan, [])
        packet["evidence_status"] = "unavailable"
        packet["error"] = res["error"]
        return json.dumps(packet, ensure_ascii=False)
    return json.dumps(
        context_packet(plan, res.get("memories", [])[:limit]),
        ensure_ascii=False,
    )



def _tool_inquire(arguments: dict) -> dict | str:
    allowed = {
        "query",
        "operation",
        "maximum_passes",
        "maximum_selected_records",
        "include_trace",
    }
    unknown = set(arguments) - allowed
    if unknown:
        return _ToolError(
            "error: unsupported inquiry fields: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    query_value = arguments.get("query")
    if (
        not isinstance(query_value, str)
        or not 3 <= len(query_value.strip()) <= 4096
    ):
        return _ToolError(
            "error: query must be a string containing 3 to 4096 characters"
        )
    query = query_value.strip()
    policy: dict[str, int] = {}
    if "maximum_passes" in arguments:
        maximum_passes = arguments["maximum_passes"]
        if (
            isinstance(maximum_passes, bool)
            or not isinstance(maximum_passes, int)
            or not 1 <= maximum_passes <= 3
        ):
            return _ToolError("error: maximum_passes must be an integer from 1 to 3")
        policy["maximum_passes"] = maximum_passes
    if "maximum_selected_records" in arguments:
        maximum_selected = arguments["maximum_selected_records"]
        if (
            isinstance(maximum_selected, bool)
            or not isinstance(maximum_selected, int)
            or not 1 <= maximum_selected <= 12
        ):
            return _ToolError(
                "error: maximum_selected_records must be an integer "
                "from 1 to 12"
            )
        policy["maximum_selected_records"] = maximum_selected
    overrides: dict[str, str] = {}
    operation = arguments.get("operation")
    if operation is not None:
        operation = str(operation)
        if operation not in {
            "verify_claim",
            "evaluate_decision",
            "trace_evolution",
        }:
            return _ToolError("error: unsupported inquiry operation")
        overrides["operation"] = operation
    include_trace = arguments.get("include_trace", False)
    if not isinstance(include_trace, bool):
        return _ToolError("error: include_trace must be a boolean")
    res = _http_post(
        "/inquire",
        {
            "query": query,
            "policy": policy or "auto",
            "overrides": overrides,
            "include_trace": include_trace,
        },
    )
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    return res


def _bound_inquiry_mcp_result(
    result: dict,
    *,
    max_bytes: int = MCP_MAX_BYTES,
) -> dict:
    bounded = json.loads(json.dumps(result))

    def size() -> int:
        return len(compact_json(bounded).encode("utf-8"))

    if size() <= max_bytes:
        return bounded
    trace_was_present = "trace" in bounded
    bounded.pop("trace", None)
    bounded["mcp_bounds"] = {
        "maximum_bytes": max_bytes,
        "trace_included": False,
        "trace_omitted_for_mcp_bound": trace_was_present,
        "records_truncated": False,
        "fail_closed": False,
    }
    if size() <= max_bytes:
        return bounded

    records = bounded.get("records")
    if not isinstance(records, list):
        records = []
    while size() > max_bytes:
        candidates = [
            record
            for record in records
            if isinstance(record, dict) and str(record.get("text") or "")
        ]
        if not candidates:
            break
        largest = max(
            candidates,
            key=lambda record: len(str(record["text"]).encode("utf-8")),
        )
        raw = str(largest["text"])
        raw_bytes = raw.encode("utf-8")
        keep = max(0, len(raw_bytes) // 2)
        largest["text"] = raw_bytes[:keep].decode("utf-8", errors="ignore")
        largest["text_complete"] = False
        largest.setdefault("text_bytes", len(raw_bytes))
        largest["truncated_for_mcp"] = True
        bounded["mcp_bounds"]["records_truncated"] = True

    if size() <= max_bytes:
        return bounded

    raw_ask = str(result.get("ask", {}).get("raw_ask", ""))
    return {
        "schema_version": result.get(
            "schema_version", "fidelis.inquiry-result/v1"
        ),
        "ask": {
            "raw_ask_sha256": hashlib.sha256(
                raw_ask.encode("utf-8")
            ).hexdigest(),
            "operation": result.get("ask", {}).get("operation"),
        },
        "records": [],
        "covered_roles": [],
        "missing_roles": list(result.get("missing_roles", [])),
        "structural_sufficiency": {
            "sufficient": False,
            "checks": {},
            "required_missing": list(result.get("missing_roles", [])),
            "chronology_basis": "not_evaluated",
        },
        "disposition": "abstain",
        "stop_reason": "mcp_presentation_limit",
        "guidance": (
            "The inquiry packet exceeded the MCP presentation bound; use "
            "Python, HTTP, or CLI output for the full pointer-bound packet."
        ),
        "runtime": dict(result.get("runtime", {})),
        "mcp_bounds": {
            "maximum_bytes": max_bytes,
            "trace_included": False,
            "trace_omitted_for_mcp_bound": trace_was_present,
            "records_truncated": True,
            "fail_closed": True,
        },
    }


def _tool_query(arguments: dict) -> str:
    query = arguments.get("query", "")
    limit = int(arguments.get("limit", 5))
    res = _http_post(
        "/query", {"text": query, "limit": limit, **_temporal_args(arguments)}
    )
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    memories = res.get("memories", [])
    if not memories:
        return "no memories found."
    lines = [f"{len(memories)} memories:"]
    for i, m in enumerate(memories[:limit], 1):
        text = m.get("text", "")
        score = m.get("score", "")
        lines.append(f"  [{i}] (score {score}){_temporal_label(m)} {text[:300]}")
    return "\n".join(lines)


def _validated_store_arguments(arguments: dict) -> tuple[dict | None, str | None]:
    text = str(arguments.get("text", "")).strip()
    if not text:
        return None, "text is required"
    payload: dict = {"text": text}
    record_id = arguments.get("id")
    if record_id is not None:
        if not isinstance(record_id, str) or not _STORE_ID_RE.fullmatch(record_id):
            return None, "id must be a 3-128 character stable identifier"
        payload["id"] = record_id
    # Temporal declarations are forwarded as-is; the server owns their
    # validation and answers 400 with the precise reason.
    for key in ("event_at", "valid_from", "valid_to", "supersedes", "source"):
        if arguments.get(key) is not None:
            payload[key] = arguments[key]
    metadata = arguments.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            return None, "metadata must be an object"
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > _MAX_STORE_METADATA_BYTES:
            return None, "metadata exceeds 16384 bytes"
        declarations = metadata.get(DECLARATIONS_FIELD, [])
        if not isinstance(declarations, list):
            return None, f"metadata.{DECLARATIONS_FIELD} must be an array"
        if len(declarations) > _MAX_RELATION_DECLARATIONS:
            return None, f"metadata.{DECLARATIONS_FIELD} exceeds 16 entries"
        for index, declaration in enumerate(declarations):
            if not isinstance(declaration, dict):
                return None, f"relation declaration {index} must be an object"
            if set(declaration) != {
                "type",
                "target_record_id",
                "support_text",
                "declared_by",
            }:
                return None, f"relation declaration {index} has invalid fields"
            if declaration.get("declared_by") not in {"caller", "source"}:
                return None, f"relation declaration {index} has invalid declared_by"
            for field in ("type", "target_record_id", "support_text"):
                value = declaration.get(field)
                if not isinstance(value, str) or not value.strip():
                    return None, f"relation declaration {index} requires {field}"
            if declaration["type"] not in SUPPORTED_RELATION_TYPES:
                return None, (
                    f"relation declaration {index} has unsupported type"
                )
            target = declaration["target_record_id"]
            if not _STORE_ID_RE.fullmatch(target):
                return None, f"relation declaration {index} has invalid target_record_id"
            if declaration["support_text"] not in text:
                return None, f"relation declaration {index} support_text is not verbatim"
        payload["metadata"] = metadata
    return payload, None


def _tool_store(arguments: dict) -> str | dict:
    payload, error = _validated_store_arguments(arguments)
    if error or payload is None:
        return _ToolError(f"error: {error}")
    res = _http_post("/store", payload)
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    # Report what actually happened. Only "stored" means a vector was
    # written — a missing or unrecognised status must never default to it.
    status = res.get("status") or "unknown"
    if status == "rejected":
        return _ToolError(
            f"NOT stored — rejected by write gate ({res.get('reason')}). "
            "Nothing was written or queued."
        )
    if status == "duplicate":
        return f"already stored (identical text): {res.get('id')} — nothing new written"
    has_declarations = bool(
        isinstance(arguments.get("metadata"), dict)
        and arguments["metadata"].get(DECLARATIONS_FIELD)
    )
    # Only negotiated structured responses can report queued declarations below.
    # Older protocols must never render a pending write as stored.
    if status == "queued" and not (
        has_declarations and _negotiated_protocol == _MCP_STRUCTURED_PROTOCOL
    ):
        return (
            f"NOT yet stored — queued for replay as {res.get('id')} "
            f"({res.get('reason')}). It is not recallable until replay succeeds."
        )
    if status not in {"stored", "queued"}:
        # A missing status, or one this build does not recognise, is never a
        # completed write — do not let it fall through to "stored memory: ...".
        return _ToolError(
            "outcome unknown — recall the text or store again; identical "
            "text is deduplicated"
        )
    stored_id = res.get("id", "unknown")
    if res.get("recorded_at") and not has_declarations:
        # Echo exactly what was recorded (docs/TIME-AWARE-SPEC.md F1b) so a
        # caller can confirm without a recall, and say so when the `id`
        # argument was not honoured (F1c) rather than let that pass silently.
        supersedes = res.get("supersedes") or payload.get("supersedes") or []
        if isinstance(supersedes, str):
            supersedes = [supersedes]
        note = f" | supersedes: {', '.join(supersedes)}" if supersedes else ""
        window_parts = [
            f"{key}: {res[key]}"
            for key in ("valid_from", "valid_to", "event_at")
            if res.get(key)
        ]
        window_note = f" | {' | '.join(window_parts)}" if window_parts else ""
        already = res.get("superseded_targets_already_superseded")
        already_note = (
            f" | already-superseded target(s) re-superseded: {', '.join(already)}"
            if already
            else ""
        )
        id_ignored_note = (
            " | id ignored (relation envelopes not enabled)"
            if res.get("id_ignored")
            else ""
        )
        return (
            f"stored memory: {stored_id} | recorded_at: {res['recorded_at']}"
            f"{note}{window_note}{already_note}{id_ignored_note}"
        )
    declaration_count = len(
        arguments.get("metadata", {}).get(DECLARATIONS_FIELD, [])
        if isinstance(arguments.get("metadata"), dict)
        else []
    )
    if declaration_count == 0:
        return f"stored memory: {stored_id}"
    envelope = res.get("relation_envelope")
    if not isinstance(envelope, dict):
        envelope = {}
    relations = envelope.get("relations")
    if not isinstance(relations, list):
        relations = []
    visible_relations = [
        {
            "type": relation.get("type"),
            "target_record_id": relation.get("target_record_id"),
            "status": relation.get("status"),
            "authority": (
                relation.get("declaration", {}).get("authority")
                if isinstance(relation.get("declaration"), dict)
                else None
            ),
        }
        for relation in relations
        if (
            isinstance(relation, dict)
            and isinstance(relation.get("declaration"), dict)
            and relation["declaration"].get("origin")
            == "source_metadata_explicit_relation"
        )
    ]
    unresolved_count = sum(
        str(relation.get("status") or "").startswith("unresolved")
        for relation in visible_relations
    )
    disposition = (
        "unresolved"
        if unresolved_count
        else "accepted"
        if visible_relations
        else "no_relation_materialized"
    )
    relation_declarations = {
        "disposition": disposition,
        "declared_count": declaration_count,
        "materialized_count": len(visible_relations),
        "unresolved_count": unresolved_count,
        "relations": visible_relations,
        "authority": "declaration_only_not_truth_verified",
    }
    if _negotiated_protocol == _MCP_STRUCTURED_PROTOCOL:
        return {
            # Reuse the already-resolved status (never re-defaults to
            # "stored" on a missing key — see the guard above).
            "status": status,
            "id": stored_id,
            "relation_declarations": relation_declarations,
        }
    return (
        f"stored memory: {stored_id} | relation declarations: "
        f"{disposition} ({len(visible_relations)} materialized, "
        f"{unresolved_count} unresolved; declaration only, not truth verified)"
    )


def _tool_health(arguments: dict) -> str:
    """fidelis_health (PACKET F3): up/down plus F2c store-wide stats in one
    call. `calibrated`/`snapshot` are dropped from this rendering -- testers
    could not interpret them and no agent decision depends on them (T4)."""
    res = _http_get("/health")
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    line1 = (
        f"status: {res.get('status')} | memories: {res.get('count')} | "
        f"version: {res.get('version')}"
    )
    stats = _http_get("/stats")
    if stats.get("error") or stats.get("index") == "unavailable":
        return (
            f"{line1}\nstats: unavailable (temporal index down) | "
            f"queued: {res.get('queued', 0)}"
        )
    line2 = (
        f"current: {stats.get('current')} | superseded: {stats.get('superseded')} | "
        f"expired: {stats.get('expired')} | not_yet_valid: {stats.get('not_yet_valid')} | "
        f"no_recorded_at: {stats.get('no_recorded_at')} | "
        f"queued: {stats.get('queued')} | dead_letter: {stats.get('dead_letter')}"
    )
    return f"{line1}\n{line2}"


# ── PACKET F3 v2 handlers: fidelis_recall (fast/thorough), fidelis_correct,
# fidelis_get, fidelis_recent ────────────────────────────────────────────


def _v2_hit_line(memory: dict, index: int) -> str:
    """One fidelis_recall hit line: id and recorded date always shown (per
    the real-user panel, the current get-by-id-later workflow needed the id
    on every line, not just non-current ones); a status marker only when
    not current; the raw similarity score labelled, not bare, and never
    implied to be comparable across modes/hits from different code paths
    (three testers read unexplained score/order mismatches as a bug)."""
    text = str(memory.get("text", ""))
    record_id = memory.get("id") or "unknown"
    temporal = memory.get("temporal") if isinstance(memory.get("temporal"), dict) else {}
    status = temporal.get("status", "current")
    parts = [f"id={record_id}"]
    if status == "superseded":
        by = ", ".join(str(b) for b in temporal.get("superseded_by") or [])
        parts.append(f"SUPERSEDED by {by}" if by else "SUPERSEDED")
    elif status == "expired":
        parts.append(f"EXPIRED {temporal.get('valid_to')}")
    elif status == "not_yet_valid":
        parts.append(f"NOT YET VALID until {temporal.get('valid_from')}")
    recorded_at = temporal.get("recorded_at")
    parts.append(f"recorded {recorded_at}" if recorded_at else "recorded: unknown")
    score = memory.get("score")
    if score is not None:
        parts.append(f"similarity={score}")
    return f"  [{index}] ({'; '.join(parts)}) {text}"


def _tool_fidelis_recall(arguments: dict) -> str:
    """fidelis_recall (PACKET F3): default mode="fast" hits POST /query
    without a generative LLM; mode="thorough" hits POST /recall_hybrid, zero_llm tier.
    Never cascades to a slower path on failure -- reports the error."""
    query = arguments.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return _ToolError("error: query is required")
    try:
        limit = max(1, min(int(arguments.get("limit", 5)), 20))
    except (TypeError, ValueError):
        return _ToolError("error: limit must be an integer")
    mode = arguments.get("mode", "fast")
    if mode not in ("fast", "thorough"):
        return _ToolError(f"error: mode must be 'fast' or 'thorough', got {mode!r}")
    temporal_args = _temporal_args(arguments)
    if mode == "fast":
        res = _http_post("/query", {"text": query, "limit": limit, **temporal_args})
    else:
        res = _http_post(
            "/recall_hybrid",
            {
                "text": query, "limit": limit, "top_k": limit,
                "tier": "zero_llm", **temporal_args,
            },
        )
    if res.get("error"):
        # On backend failure, never silently fall back to a slower path --
        # report the error so the caller can retry or escalate itself.
        return _ToolError(f"error: {res['error']}")
    memories = res.get("memories", [])[:limit]
    if not memories:
        return f"no memories found (mode={mode})."
    current_count = sum(
        1 for m in memories
        if not isinstance(m.get("temporal"), dict) or m["temporal"].get("status", "current") == "current"
    )
    noncurrent_count = len(memories) - current_count
    header = (
        f"{len(memories)} memories (mode={mode}): current facts first; "
        f"superseded/expired/not-yet-valid shown after, marked "
        f"({current_count} current, {noncurrent_count} other)."
    )
    lines = [header] + [_v2_hit_line(m, i) for i, m in enumerate(memories, 1)]
    return "\n".join(lines)


def _tool_fidelis_correct(arguments: dict) -> str:
    """fidelis_correct (PACKET F3, NEW): recall -> copy id -> store, in one
    call. Validates `id` exists (F1a, via the /store 400 path) and checks
    for an already-superseded id BEFORE writing so the caller can redirect
    to the current replacement, unless force=true."""
    old_id = arguments.get("id")
    if not isinstance(old_id, str) or not old_id.strip():
        return _ToolError("error: id is required")
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        return _ToolError("error: text is required")
    force = bool(arguments.get("force", False))

    existing = _http_post("/get", {"id": old_id})
    if existing.get("error"):
        return _ToolError(f"error: {existing['error']}")
    existing_temporal = existing.get("temporal") or {}
    if existing_temporal.get("status") == "superseded" and not force:
        current_ids = existing_temporal.get("superseded_by") or []
        current = current_ids[-1] if current_ids else "unknown"
        return _ToolError(
            f"error: {old_id} is already superseded by {current} -- correct "
            f"{current} instead, or pass force=true to supersede {old_id} anyway"
        )

    payload: dict = {"text": text, "supersedes": [old_id]}
    for key in ("valid_from", "valid_to", "event_at", "source"):
        if arguments.get(key) is not None:
            payload[key] = arguments[key]
    res = _http_post("/store", payload)
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    status = res.get("status") or "unknown"
    if status == "rejected":
        return _ToolError(
            f"NOT stored — rejected ({res.get('reason')}). Nothing corrected."
        )
    if status not in {"stored", "queued"}:
        return _ToolError(
            "outcome unknown — recall or fidelis_get the old id to check "
            "whether the correction landed"
        )
    new_id = res.get("id", "unknown")
    if status == "queued":
        return (
            f"NOT yet stored — queued as {new_id}, will supersede {old_id} "
            f"once replay succeeds ({res.get('reason')})."
        )
    return f"corrected: {new_id} supersedes {old_id}"


def _render_chain_entries(entries: list, label: str) -> list[str]:
    if not entries:
        return [f"{label}: (nothing)"]
    lines = [f"{label}:"]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", ""))
        lines.append(f"  {entry.get('id')} [{entry.get('status')}] {text}")
    return lines


def _tool_fidelis_get(arguments: dict) -> str:
    """fidelis_get (PACKET F3, NEW): one record by id, via F2a, rendered
    compactly with its correction chain in both directions."""
    record_id = arguments.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        return _ToolError("error: id is required")
    res = _http_post("/get", {"id": record_id})
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    temporal = res.get("temporal") or {}
    lines = [
        f"id: {res.get('id')}",
        f"text: {res.get('text', '')}",
    ]
    if res.get("source"):
        lines.append(f"source: {res['source']}")
    status = temporal.get("status", "current")
    status_line = f"status: {status}"
    if status == "superseded":
        by = ", ".join(str(b) for b in temporal.get("superseded_by") or [])
        if by:
            status_line += f" (superseded by {by})"
    elif status == "expired":
        status_line += f" (valid_to {temporal.get('valid_to')})"
    elif status == "not_yet_valid":
        status_line += f" (valid_from {temporal.get('valid_from')})"
    lines.append(status_line)
    lines.append(f"recorded_at: {temporal.get('recorded_at') or 'unknown'}")
    lines.extend(_render_chain_entries(res.get("supersedes") or [], "supersedes (oldest last)"))
    lines.extend(
        _render_chain_entries(res.get("superseded_by") or [], "superseded_by (newest last)")
    )
    if res.get("index") == "unavailable":
        lines.append("(temporal index unavailable -- chain may be incomplete)")
    return "\n".join(lines)


def _tool_fidelis_recent(arguments: dict) -> str:
    """fidelis_recent (PACKET F3, NEW): newest records by recorded_at, or
    corrections only, via F2b."""
    try:
        limit = max(1, min(int(arguments.get("limit", 10)), 50))
    except (TypeError, ValueError):
        return _ToolError("error: limit must be an integer")
    kind = arguments.get("kind", "all")
    if kind not in ("all", "corrections"):
        return _ToolError(f"error: kind must be 'all' or 'corrections', got {kind!r}")
    payload: dict = {"limit": limit, "kind": kind}
    since = arguments.get("since")
    if since is not None:
        payload["since"] = since
    res = _http_post("/recent", payload)
    if res.get("error"):
        return _ToolError(f"error: {res['error']}")
    records = res.get("records", [])
    if not records:
        note = res.get("note")
        return f"0 memories.{f' {note}' if note else ''}"
    since_note = f" since {since}" if since else ""
    header = f"{len(records)} memories{since_note} (kind={kind}):"
    lines = [header]
    for i, r in enumerate(records, 1):
        status = r.get("status", "current")
        marker = "" if status == "current" else f" [{status}]"
        recorded = r.get("recorded_at") or "unknown"
        corrects = ""
        if kind == "corrections" and r.get("supersedes"):
            corrects = f" (supersedes {', '.join(r['supersedes'])})"
        text = str(r.get("text", ""))
        lines.append(
            f"  [{i}] id={r.get('id')}{marker} recorded {recorded}{corrects} {text}"
        )
    return "\n".join(lines)


def _local_feature_state() -> dict:
    def _state(check) -> dict:
        try:
            return {"valid": True, "enabled": bool(check())}
        except ValueError as exc:
            return {"valid": False, "enabled": None, "error": str(exc)}

    return {
        "evidence_status": _state(evidence_status_enabled),
        "relation_envelopes": _state(relation_envelopes_enabled),
    }


def _tool_capabilities(arguments: dict) -> dict:
    service = _http_get("/capabilities")
    service_health = (
        "unavailable" if service.get("error") else str(service.get("readiness", "unknown"))
    )
    return {
        "schema_version": "fidelis.runtime-capabilities/v1",
        "package_version": __version__,
        "mcp": {
            "negotiated_protocol": _negotiated_protocol,
            "structured_content": _negotiated_protocol == _MCP_STRUCTURED_PROTOCOL,
            "local_feature_state": _local_feature_state(),
        },
        "service": service,
        "source_health": {
            "service": service_health,
            "service_and_mcp_flag_parity_asserted": False,
        },
        "relation_contract": {
            "schema_version": RELATION_SCHEMA_VERSION,
            "extractor_version": RELATION_EXTRACTOR_VERSION,
            "semantics_version": RELATION_SEMANTICS_VERSION,
        },
        "client_tool_invocation_required": True,
    }


def _ranked_recent_work_lines(query, reference, session_results, task_results, limit):
    """Render the shared cross-client answer for a recent-work reference.

    This block is what both clients read to pick a record: exact identity,
    source class, real timestamp, the literal capture class of the evidence and
    a pointer to it. It never restates a whole transcript, and it stays silent
    rather than naming the nearest session when nothing actually matched.
    """
    from fidelis.recall_recent import rank_recent_work

    ranked = rank_recent_work(
        query,
        session_results=session_results,
        task_results=task_results,
        reference=reference,
        limit=limit,
    )
    header = "ranked cross-client match (identity and time aware)"
    if reference.window_label:
        header += (
            f" | window {reference.window_label}="
            f"{reference.window_start}..{reference.window_end}"
            f" (UTC offset {reference.tz_offset})"
        )
    lines = [f"{header}:"]
    if not ranked.candidates:
        lines.append(
            "  none met the evidence floor; no match is asserted for this reference"
        )
    for index, candidate in enumerate(ranked.candidates, 1):
        lines.append(
            f"  [{index}] {candidate.source_class} | id={candidate.identity} | "
            f"{candidate.timestamp or 'unknown-timestamp'} | "
            f"capture={candidate.capture_class} | "
            f"assistant_text={'yes' if candidate.assistant_text_available else 'no'} | "
            f"score={candidate.score:.3f}"
            + (f" | why={','.join(candidate.match_reasons)}" if candidate.match_reasons else "")
        )
        if candidate.evidence_pointer:
            lines.append(f"      evidence: {candidate.evidence_pointer}")
        if candidate.excerpt:
            lines.append(f"      {candidate.excerpt[:200]}")
    lines.append("  live session/task status is not derivable from stored evidence.")
    return lines


def _tool_recall_sessions(arguments: dict) -> str:
    query = str(arguments.get("query", "")).strip()
    limit = max(1, min(int(arguments.get("limit", 3)), 5))
    if not query:
        return "error: query is required"
    from fidelis.session_reference import parse_session_reference

    reference = parse_session_reference(query)
    session_error: str | None = None
    try:
        from fidelis.recall_sessions import query_sessions

        results = query_sessions(query, top_k=limit, reference=reference)
    except Exception as exc:
        results = []
        session_error = str(exc)
    try:
        from fidelis.codex_history import search_codex_history

        codex_results = search_codex_history(query, limit=limit)
    except Exception:
        codex_results = []
    if not results and not codex_results:
        if session_error:
            return (
                f"ingested session recall unavailable: {session_error}; "
                f"no Codex task memories found for: {query}"
            )
        return f"no session or Codex task memories found for: {query}"
    lines = _ranked_recent_work_lines(
        query, reference, results, codex_results, limit
    )
    # The pre-existing section keeps its original meaning: similarity-only score
    # and order. Reference adjustments belong to the ranked block above, so a
    # same-day but off-topic record cannot displace the best semantic match here.
    results = sorted(results, key=lambda item: item.semantic_score, reverse=True)
    lines.append(f"{len(results)} ingested session result(s):")
    if session_error:
        lines.append(f"  ingested session recall unavailable: {session_error}")
    for index, result in enumerate(results, 1):
        lines.append(
            f"  [{index}] session={result.session_id[:8]} | "
            f"score={result.semantic_score:.3f} | turns={result.turn_count} | "
            f"{result.start_ts[:10]}"
        )
        lines.append(f"      {result.matched_chunk[:300]}")
    lines.append(f"{len(codex_results)} local Codex task result(s):")
    for index, result in enumerate(codex_results, 1):
        availability = (
            "literal message"
            if result.assistant_text_available
            else "telemetry partial; assistant prose unavailable"
        )
        lines.append(
            f"  [{index}] task={result.thread_id} | score={result.score:.3f} | "
            f"{availability}"
        )
        lines.append(f"      {result.text[:500]}")
    return "\n".join(lines)


def _sanitize_auxiliary_session_text(text: str) -> tuple[str, bool]:
    """Redact local exception details from the flag-on auxiliary envelope."""

    redacted = False
    lines: list[str] = []
    for line in text.splitlines():
        if "ingested session recall unavailable:" in line:
            prefix = line.split("ingested session recall unavailable:", 1)[0]
            lines.append(
                f"{prefix}ingested session recall unavailable; details redacted"
            )
            redacted = True
        else:
            lines.append(line)
    return "\n".join(lines), redacted


def _tool_recall_both(arguments: dict) -> str | dict:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return "error: query is required"
    atomic = _tool_recall({"query": query, "limit": 3})
    sessions = _tool_recall_sessions({"query": query, "limit": 3})
    if evidence_status_enabled() and isinstance(atomic, dict):
        combined = dict(atomic)
        safe_sessions, session_error = _sanitize_auxiliary_session_text(
            sessions
        )
        session_bytes = safe_sessions.encode("utf-8")
        shown = session_bytes[:8192].decode("utf-8", errors="ignore")
        combined["auxiliary_sources"] = {
            "sessions": {
                "authority": "separate_auxiliary_session_source",
                "text": shown,
                "text_sha256": hashlib.sha256(session_bytes).hexdigest(),
                "text_bytes": len(session_bytes),
                "text_complete": shown == safe_sessions,
                "truncated": shown != safe_sessions,
                "error_details_redacted": session_error,
            }
        }
        if session_error:
            combined["status"] = "degraded_evidence"
            combined.setdefault("source_health", []).append(
                {
                    "source": "session_memory",
                    "requested_path": "session_and_codex_history",
                    "actual_path": "unavailable_or_partial",
                    "state": "error",
                    "failure_class": "AuxiliarySessionSourceError",
                    "authority_limitation": (
                        "Auxiliary session evidence was unavailable or partial."
                    ),
                }
            )
        if shown != safe_sessions:
            combined.setdefault("bounds", {})["truncated"] = True
        return bound_mcp_envelope(combined)
    return f"=== ATOMIC (facts) ===\n{atomic}\n\n=== SESSIONS ===\n{sessions}"


# PACKET F3: the dispatch table for the v2 surface. `_tool_recall`,
# `_tool_orient`, `_tool_orient_legacy`, `_tool_orient_structured`,
# `_tool_inquire`, `_tool_query`, and `_tool_capabilities` are UNCHANGED and
# stay importable/callable directly (other tests exercise them that way —
# PRD D2/D4: the functions are unchanged, only their reachability via a
# tools/call NAME changed) but are no longer reachable through this table;
# calling their old tool names now returns -32602 naming the replacement
# (see _REMOVED_TOOLS and _handle). No cogito_* alias is listed or
# dispatchable (all were byte-identical duplicates, confirmed by four
# independent testers).
TOOL_HANDLERS = {
    "fidelis_recall": _tool_fidelis_recall,
    "fidelis_store": _tool_store,
    "fidelis_correct": _tool_fidelis_correct,
    "fidelis_get": _tool_fidelis_get,
    "fidelis_recent": _tool_fidelis_recent,
    "fidelis_health": _tool_health,
    # fidelis_route is deliberately absent here -- it is dispatchable only
    # when FIDELIS_ORIENT_ROUTE=notion, checked dynamically in _handle so
    # its meaning never depends on which table entries happen to exist.
    # cogito_recall_sessions / cogito_recall_both are intentionally absent —
    # see the comment above _build_tools_v2 (R5/D5).
}


def _send(payload: dict | list) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _invalid_params(rid, method_or_tool: str) -> dict:
    _log_call(method_or_tool, "invalid_params", 0.0)
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32602, "message": "params must be an object"},
    }


def _handle(req: dict) -> dict | None:
    global _negotiated_protocol
    method = req.get("method", "")
    if not isinstance(method, str):
        method = str(method)
    rid = req.get("id")

    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}} if rid is not None else None
    if method == "initialize":
        # "params" absent means "no params" (fine); present-but-not-an-object
        # (e.g. JSON null) is the malformed case a client can actually send.
        if "params" in req and not isinstance(req["params"], dict):
            return _invalid_params(rid, "initialize")
        params = req.get("params") or {}
        requested = str(params.get("protocolVersion") or "")
        _negotiated_protocol = (
            requested if requested in _SUPPORTED_PROTOCOLS else _MCP_LEGACY_PROTOCOL
        )
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": _negotiated_protocol,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fidelis", "version": __version__},
            },
        }
    if method == "tools/list":
        # PRD D2: one schema for every negotiated protocol version -- built
        # fresh so an env change (relation envelopes, FIDELIS_ORIENT_ROUTE)
        # is reflected immediately, never a stale snapshot.
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"tools": _build_tools_v2()},
        }
    if method == "tools/call":
        if "params" in req and not isinstance(req["params"], dict):
            return _invalid_params(rid, "tools/call")
        params = req.get("params") or {}
        tool_name = params.get("name", "")
        if not isinstance(tool_name, str):
            tool_name = str(tool_name)
        if "arguments" in params and not isinstance(params["arguments"], dict):
            return _invalid_params(rid, tool_name or "tools/call")
        arguments = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            if tool_name in _REMOVED_TOOLS:
                _log_call(tool_name, "unknown_tool", 0.0)
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {
                        "code": -32602,
                        "message": (
                            f"{tool_name} was removed: use "
                            f"{_REMOVED_TOOLS[tool_name]}"
                        ),
                    },
                }
            _log_call(tool_name or "?", "unknown_tool", 0.0)
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32602, "message": f"unknown tool: {tool_name}"},
            }
        start = time.monotonic()
        _LAST_BACKEND_CALL["path"] = None
        _LAST_BACKEND_CALL["status"] = None
        try:
            output = handler(arguments)
            is_error = isinstance(output, _ToolError)
            _log_call(
                tool_name,
                "tool_error" if is_error else "ok",
                (time.monotonic() - start) * 1000,
                backend_path=_LAST_BACKEND_CALL["path"],
                http_status=_LAST_BACKEND_CALL["status"],
            )
            if isinstance(output, dict):
                bounded = (
                    _bound_inquiry_mcp_result(output)
                    if tool_name in {"fidelis_inquire", "cogito_inquire"}
                    else bound_mcp_envelope(output)
                )
                result = {
                    "content": [
                        {"type": "text", "text": compact_json(bounded)}
                    ],
                }
                if _negotiated_protocol == _MCP_STRUCTURED_PROTOCOL:
                    result["structuredContent"] = bounded
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": result,
                }
            result = {"content": [{"type": "text", "text": str(output)}]}
            if is_error:
                result["isError"] = True
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": result,
            }
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            if evidence_status_enabled() and tool_name in {
                "fidelis_recall",
                "cogito_recall",
                "fidelis_orient",
                "cogito_orient",
                "cogito_recall_both",
            }:
                _log_call(
                    tool_name, "fail_closed_degraded", elapsed_ms,
                    detail=type(e).__name__,
                )
                query = str(
                    arguments.get("query")
                    or arguments.get("message")
                    or ""
                )
                degraded = bound_mcp_envelope(
                    build_evidence_envelope(
                        query=query,
                        result={
                            "retrieval_status": "error",
                            "records": [],
                            "memories": [],
                            "plan": {"mode": "unknown", "initial_depth": 0},
                            "method": "mcp_handler_fail_closed",
                        },
                        source_health=[
                            {
                                "source": "mcp_handler",
                                "requested_path": tool_name,
                                "actual_path": "fail_closed",
                                "state": "error",
                                "failure_class": type(e).__name__,
                                "authority_limitation": (
                                    "The requested MCP evidence path failed."
                                ),
                            }
                        ],
                    )
                )
                result = {
                    "content": [
                        {"type": "text", "text": compact_json(degraded)}
                    ],
                }
                if _negotiated_protocol == _MCP_STRUCTURED_PROTOCOL:
                    result["structuredContent"] = degraded
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": result,
                }
            # Never send str(exception) to the client — it can carry a path,
            # an argument value, or other internal detail. The full detail
            # goes to the log only; the client gets a fixed message plus pid.
            _log_call(
                tool_name, "internal_error", elapsed_ms,
                detail=f"{type(e).__name__}: {e}",
            )
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {
                    "code": -32603,
                    "message": f"internal error (see mcp server log, pid {os.getpid()})",
                },
            }
    if method == "notifications/initialized":
        return None  # no response needed for notifications
    _log_call(method or "?", "method_not_found", 0.0)
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def _dispatch(msg) -> dict | None:
    """Validate and route one already-JSON-parsed message (never a list).

    Never raises: any failure becomes a JSON-RPC error, or — for a
    notification, i.e. a message with no "id" key at all, however malformed
    — no response at all. This is what keeps a single bad input line from
    ever being able to end the process (R1)."""
    if not isinstance(msg, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "invalid request: expected a JSON object",
            },
        }
    is_notification = "id" not in msg
    try:
        resp = _handle(msg)
    except Exception as e:  # noqa: last-resort guard, a bad line must never kill the process
        _log_call(
            str(msg.get("method") or "?"), "internal_error", 0.0,
            detail=f"{type(e).__name__}: {e}",
        )
        resp = {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "error": {
                "code": -32603,
                "message": f"internal error (see mcp server log, pid {os.getpid()})",
            },
        }
    return None if is_notification else resp


def _dispatch_batch(items: list) -> None:
    """A JSON-RPC batch (a bare JSON array). MCP protocol revision
    2025-03-26 required batch support; 2025-06-18 dropped it. Simplest
    correct, non-crashing handling: answer each element as if it had arrived
    on its own line, replying with a JSON array and omitting notifications.
    An empty array is itself an invalid batch (JSON-RPC 2.0 spec): reply with
    a single Response object, not an array."""
    if not items:
        _send(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "invalid request: empty batch"},
            }
        )
        return
    responses = [resp for resp in (_dispatch(item) for item in items) if resp is not None]
    if responses:
        _send(responses)


def main() -> int:
    _make_stderr_nonblocking()
    threading.Thread(target=_exit_when_orphaned, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            _log_call("?", "parse_error", 0.0)
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "parse error: invalid JSON"},
                }
            )
            continue
        if isinstance(parsed, list):
            if _negotiated_protocol == _MCP_BATCH_PROTOCOL:
                _dispatch_batch(parsed)
            else:
                _send({"jsonrpc": "2.0", "id": None, "error": {
                    "code": -32600, "message": "batches unsupported by negotiated protocol"}})
            continue
        resp = _dispatch(parsed)
        if resp is not None:
            _send(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
