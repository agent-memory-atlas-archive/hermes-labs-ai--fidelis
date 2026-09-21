"""PACKET F3 — MCP surface v2 (docs/TIME-AWARE-SPEC.md, MCP-PRD.md D1-D6,
research/memory-mcp-surface.md, reports/SYNTHESIS.md).

Covers:
  - exact tool inventory, identical for both negotiated protocol versions
    (PRD D2: the legacy 2024-11-05 inventory freeze is lifted once, here),
    with and without FIDELIS_ORIENT_ROUTE=notion.
  - every removed name (all cogito_* aliases, fidelis_query, fidelis_inquire,
    fidelis_orient, fidelis_capabilities) -> -32602 naming its replacement.
  - fidelis_recall: default mode hits /query, mode="thorough" hits
    /recall_hybrid; the ordering header; id + status marker on every hit.
  - fidelis_correct: happy path, unknown id, already-superseded (force).
  - fidelis_get: both chain directions, 404.
  - fidelis_recent: limit/since/kind passthrough, corrections rendering.
  - fidelis_health: stats rendering, isError when the backend is down.
  - every tool: annotations present, description <= ~90 words with a
    "do not use" clause.
  - a lexical-proxy "selection" smoke test (explicitly NOT a model test).
"""

from __future__ import annotations

import re

import pytest

from fidelis import mcp_server


# ── tool inventory ───────────────────────────────────────────────────────


EXPECTED_TOOL_SPECS = {
    "fidelis_recall": {"required": ["query"]},
    "fidelis_store": {"required": ["text"]},
    "fidelis_correct": {"required": ["id", "text"]},
    "fidelis_get": {"required": ["id"]},
    "fidelis_recent": {"required": []},
    "fidelis_health": {"required": []},
}
EXPECTED_ORDER = [
    "fidelis_recall", "fidelis_store", "fidelis_correct",
    "fidelis_get", "fidelis_recent", "fidelis_health",
]


def _list_tools(protocol_version: str, monkeypatch=None) -> list[dict]:
    mcp_server._handle(
        {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": protocol_version},
        }
    )
    resp = mcp_server._handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    return resp["result"]["tools"]


@pytest.mark.parametrize("protocol_version", ["2024-11-05", "2025-06-18"])
def test_inventory_identical_for_both_protocol_versions(monkeypatch, protocol_version):
    monkeypatch.delenv("FIDELIS_ORIENT_ROUTE", raising=False)
    tools = _list_tools(protocol_version)
    names = [t["name"] for t in tools]
    assert names == EXPECTED_ORDER, names
    for tool in tools:
        spec = EXPECTED_TOOL_SPECS[tool["name"]]
        assert tool["inputSchema"].get("required", []) == spec["required"], tool["name"]


def test_inventory_identical_across_both_versions_directly():
    legacy = {t["name"]: t for t in _list_tools("2024-11-05")}
    modern = {t["name"]: t for t in _list_tools("2025-06-18")}
    assert set(legacy) == set(modern) == set(EXPECTED_ORDER)
    for name in EXPECTED_ORDER:
        assert legacy[name] == modern[name], name


def test_no_cogito_alias_listed():
    tools = _list_tools("2025-06-18")
    names = {t["name"] for t in tools}
    assert not any(name.startswith("cogito_") for name in names), names


def test_removed_names_absent_without_notion_route(monkeypatch):
    monkeypatch.delenv("FIDELIS_ORIENT_ROUTE", raising=False)
    names = {t["name"] for t in _list_tools("2025-06-18")}
    assert "fidelis_route" not in names
    assert "fidelis_orient" not in names
    assert "fidelis_query" not in names
    assert "fidelis_inquire" not in names
    assert "fidelis_capabilities" not in names


def test_private_route_never_listed(monkeypatch):
    monkeypatch.setenv("FIDELIS_ORIENT_ROUTE", "notion")
    names = {t["name"] for t in _list_tools("2025-06-18")}
    assert "fidelis_route" not in names
    assert "fidelis_orient" not in names, "a name never changes meaning (PRD D4)"

    monkeypatch.delenv("FIDELIS_ORIENT_ROUTE")
    names_off = {t["name"] for t in _list_tools("2025-06-18")}
    assert "fidelis_route" not in names_off


def test_fidelis_store_id_argument_listed_only_with_relation_envelopes_on(monkeypatch):
    monkeypatch.delenv("COGITO_RELATION_ENVELOPES_V1", raising=False)
    tools_off = {t["name"]: t for t in _list_tools("2025-06-18")}
    assert "id" not in tools_off["fidelis_store"]["inputSchema"]["properties"]

    monkeypatch.setenv("COGITO_RELATION_ENVELOPES_V1", "1")
    tools_on = {t["name"]: t for t in _list_tools("2025-06-18")}
    assert "id" in tools_on["fidelis_store"]["inputSchema"]["properties"]


# ── removed names -> -32602 naming the replacement ──────────────────────


REMOVED_NAMES_AND_REPLACEMENTS = [
    ("fidelis_query", "fidelis_recall"),
    ("fidelis_inquire", "fidelis_recall"),
    ("fidelis_orient", "fidelis_recall"),
    ("fidelis_capabilities", "fidelis_health"),
    ("cogito_recall", "fidelis_recall"),
    ("cogito_orient", "fidelis_recall"),
    ("cogito_inquire", "fidelis_recall"),
    ("cogito_query", "fidelis_recall"),
    ("cogito_add", "fidelis_store"),
    ("cogito_health", "fidelis_health"),
    ("cogito_capabilities", "fidelis_health"),
]


@pytest.mark.parametrize(("removed", "replacement"), REMOVED_NAMES_AND_REPLACEMENTS)
def test_removed_name_returns_32602_naming_replacement(removed, replacement):
    resp = mcp_server._handle(
        {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": removed, "arguments": {}},
        }
    )
    assert "result" not in resp, resp
    assert resp["error"]["code"] == -32602, resp
    assert removed in resp["error"]["message"]
    assert replacement in resp["error"]["message"]


def test_fidelis_route_not_dispatchable_without_notion_env(monkeypatch):
    monkeypatch.delenv("FIDELIS_ORIENT_ROUTE", raising=False)
    resp = mcp_server._handle(
        {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "fidelis_route", "arguments": {"message": "hi"}},
        }
    )
    assert resp["error"]["code"] == -32602, resp


def test_private_route_never_dispatchable(monkeypatch):
    monkeypatch.setenv("FIDELIS_ORIENT_ROUTE", "notion")
    resp = mcp_server._handle(
        {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "fidelis_route", "arguments": {"message": "open notion please"}},
        }
    )
    assert resp["error"]["code"] == -32602, resp


# ── fidelis_recall: fast/thorough dispatch, ordering header, hit lines ──


def test_recall_default_mode_hits_query(monkeypatch):
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append(path)
        return {"memories": []}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    mcp_server._tool_fidelis_recall({"query": "anything"})
    assert calls == ["/query"]


def test_recall_thorough_mode_hits_recall_hybrid(monkeypatch):
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append((path, payload))
        return {"memories": []}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    mcp_server._tool_fidelis_recall({"query": "anything", "mode": "thorough"})
    assert calls[0][0] == "/recall_hybrid"
    assert calls[0][1]["tier"] == "zero_llm"


def test_recall_invalid_mode_is_error(monkeypatch):
    result = mcp_server._tool_fidelis_recall({"query": "q", "mode": "medium"})
    assert isinstance(result, mcp_server._ToolError)


def test_recall_header_explains_ordering_and_hit_lines_carry_id_and_status(monkeypatch):
    memories = [
        {
            "id": "rec-b", "text": "current fact", "score": 0.9,
            "temporal": {"status": "current", "recorded_at": "2026-09-21T10:00:00Z"},
        },
        {
            "id": "rec-a", "text": "old fact", "score": 0.95,
            "temporal": {
                "status": "superseded", "recorded_at": "2026-09-20T10:00:00Z",
                "superseded_by": ["rec-b"],
            },
        },
    ]
    monkeypatch.setattr(
        mcp_server, "_http_post", lambda path, payload, timeout=30.0: {"memories": memories}
    )
    output = mcp_server._tool_fidelis_recall({"query": "q"})
    lines = output.splitlines()
    header = lines[0]
    assert "current" in header.lower()
    assert "superseded" in header.lower() or "after" in header.lower()

    hit_lines = lines[1:]
    line_b = next(line for line in hit_lines if "current fact" in line)
    assert "id=rec-b" in line_b
    assert "recorded 2026-09-21T10:00:00Z" in line_b

    line_a = next(line for line in hit_lines if "old fact" in line)
    assert "id=rec-a" in line_a
    assert "SUPERSEDED by rec-b" in line_a
    assert "recorded 2026-09-20T10:00:00Z" in line_a


def test_recall_backend_failure_is_isError_not_silent_fallback(monkeypatch):
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append(path)
        return {"error": "fidelis-server unreachable"}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    result = mcp_server._tool_fidelis_recall({"query": "q"})
    assert isinstance(result, mcp_server._ToolError)
    assert calls == ["/query"], "must not cascade to a slower path on failure"


def test_recall_no_hits(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "_http_post", lambda path, payload, timeout=30.0: {"memories": []}
    )
    result = mcp_server._tool_fidelis_recall({"query": "q"})
    assert not isinstance(result, mcp_server._ToolError)
    assert "no memories" in result.lower()


# ── fidelis_correct ──────────────────────────────────────────────────────


def test_correct_happy_path_one_store_call_with_supersedes(monkeypatch):
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append((path, payload))
        if path == "/get":
            return {
                "id": payload["id"], "text": "old text",
                "temporal": {"status": "current"},
            }
        assert path == "/store"
        return {"status": "stored", "id": "new-id", "recorded_at": "2026-09-21T00:00:00Z"}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    result = mcp_server._tool_fidelis_correct({"id": "old-id", "text": "corrected text"})

    assert not isinstance(result, mcp_server._ToolError), result
    store_calls = [c for c in calls if c[0] == "/store"]
    assert len(store_calls) == 1
    assert store_calls[0][1]["supersedes"] == ["old-id"]
    assert store_calls[0][1]["text"] == "corrected text"
    assert "new-id" in result
    assert "old-id" in result


def test_correct_unknown_id_is_isError_naming_the_id(monkeypatch):
    def fake_post(path, payload, timeout=30.0):
        assert path == "/get"
        return {"error": f"no record found for id {payload['id']!r}"}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    result = mcp_server._tool_fidelis_correct({"id": "no-such-id", "text": "x"})

    assert isinstance(result, mcp_server._ToolError)
    assert "no-such-id" in result


def test_correct_already_superseded_is_explanatory_non_write_without_force(monkeypatch):
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append(path)
        assert path == "/get"
        return {
            "id": "old-id", "text": "old text",
            "temporal": {"status": "superseded", "superseded_by": ["mid-id", "current-id"]},
        }

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    result = mcp_server._tool_fidelis_correct({"id": "old-id", "text": "x"})

    assert isinstance(result, mcp_server._ToolError)
    assert "old-id" in result
    assert "current-id" in result
    assert calls == ["/get"], "must not write when already superseded and not forced"


def test_correct_already_superseded_with_force_still_writes(monkeypatch):
    calls = []

    def fake_post(path, payload, timeout=30.0):
        calls.append((path, payload))
        if path == "/get":
            return {
                "id": "old-id", "text": "old text",
                "temporal": {"status": "superseded", "superseded_by": ["current-id"]},
            }
        return {"status": "stored", "id": "forced-id", "recorded_at": "2026-09-21T00:00:00Z"}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    result = mcp_server._tool_fidelis_correct(
        {"id": "old-id", "text": "x", "force": True}
    )

    assert not isinstance(result, mcp_server._ToolError), result
    store_calls = [c for c in calls if c[0] == "/store"]
    assert len(store_calls) == 1
    assert store_calls[0][1]["supersedes"] == ["old-id"]


# ── fidelis_get ──────────────────────────────────────────────────────────


def test_get_renders_both_chain_directions(monkeypatch):
    record = {
        "id": "c-id", "text": "current text", "source": "ops log",
        "temporal": {"status": "current", "recorded_at": "2026-09-21T00:00:00Z"},
        "supersedes": [
            {"id": "b-id", "text": "middle text", "status": "superseded", "recorded_at": "2026-09-20T00:00:00Z"},
            {"id": "a-id", "text": "oldest text", "status": "superseded", "recorded_at": "2026-09-19T00:00:00Z"},
        ],
        "superseded_by": [],
        "index": "ok",
    }
    monkeypatch.setattr(mcp_server, "_http_post", lambda path, payload, timeout=30.0: record)
    output = mcp_server._tool_fidelis_get({"id": "c-id"})

    assert "c-id" in output
    assert "current text" in output
    assert "supersedes" in output.lower()
    assert "b-id" in output and "middle text" in output
    assert "a-id" in output and "oldest text" in output
    assert "superseded_by" in output.lower()


def test_get_unknown_id_is_isError(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "_http_post",
        lambda path, payload, timeout=30.0: {"error": "no record found for id 'ghost'"},
    )
    result = mcp_server._tool_fidelis_get({"id": "ghost"})
    assert isinstance(result, mcp_server._ToolError)
    assert "ghost" in result


# ── fidelis_recent ───────────────────────────────────────────────────────


def test_recent_passes_limit_since_kind_through(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout=30.0):
        seen.update(path=path, payload=payload)
        return {"records": [], "kind": payload["kind"]}

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    mcp_server._tool_fidelis_recent({"limit": 3, "since": "2026-09-01T00:00:00Z", "kind": "corrections"})

    assert seen["path"] == "/recent"
    assert seen["payload"]["limit"] == 3
    assert seen["payload"]["since"] == "2026-09-01T00:00:00Z"
    assert seen["payload"]["kind"] == "corrections"


def test_recent_renders_corrections_with_ids_they_replaced(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "_http_post",
        lambda path, payload, timeout=30.0: {
            "records": [
                {"id": "new-id", "text": "new text", "status": "current",
                 "recorded_at": "2026-09-21T00:00:00Z", "supersedes": ["old-id"]},
            ],
        },
    )
    output = mcp_server._tool_fidelis_recent({"kind": "corrections"})
    assert "new-id" in output
    assert "old-id" in output
    assert "supersedes old-id" in output


# ── fidelis_health ───────────────────────────────────────────────────────


def test_health_renders_stats(monkeypatch):
    def fake_get(path, timeout=5.0):
        if path == "/health":
            return {"status": "ok", "count": 42, "version": "0.1.0b1", "queued": 0}
        assert path == "/stats"
        return {
            "current": 30, "superseded": 10, "expired": 1, "not_yet_valid": 1,
            "no_recorded_at": 0, "queued": 0, "dead_letter": 0, "index": "ok",
        }

    monkeypatch.setattr(mcp_server, "_http_get", fake_get)
    output = mcp_server._tool_health({})
    assert "status: ok" in output
    assert "current: 30" in output
    assert "superseded: 10" in output
    assert "dead_letter: 0" in output


def test_health_isError_when_backend_down(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "_http_get",
        lambda path, timeout=5.0: {"error": "fidelis-server unreachable at http://127.0.0.1:19420"},
    )
    result = mcp_server._tool_health({})
    assert isinstance(result, mcp_server._ToolError)
    assert "unreachable" in result.lower()


def test_health_stats_unavailable_is_not_isError(monkeypatch):
    def fake_get(path, timeout=5.0):
        if path == "/health":
            return {"status": "ok", "count": 5, "version": "0.1.0b1", "queued": 2}
        return {"index": "unavailable", "note": "sidecar unavailable"}

    monkeypatch.setattr(mcp_server, "_http_get", fake_get)
    result = mcp_server._tool_health({})
    assert not isinstance(result, mcp_server._ToolError)
    assert "unavailable" in result.lower()


# ── annotations + description shape ─────────────────────────────────────


@pytest.mark.parametrize("protocol_version", ["2024-11-05", "2025-06-18"])
def test_every_tool_has_annotations_and_bounded_description_with_not_clause(
    monkeypatch, protocol_version,
):
    monkeypatch.setenv("FIDELIS_ORIENT_ROUTE", "notion")
    for tool in _list_tools(protocol_version):
        assert "annotations" in tool, tool["name"]
        for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            assert key in tool["annotations"], tool["name"]
        description = tool["description"]
        word_count = len(description.split())
        assert word_count <= 95, (tool["name"], word_count)
        lowered = description.lower()
        assert "do not" in lowered or "not use" in lowered, (
            f"{tool['name']} description lacks a when-NOT-to-use clause"
        )


# ── selection smoke test (lexical proxy, NOT a model test) ──────────────
#
# This is a trivial keyword-overlap ranker over tool name + description,
# with common English stopwords removed -- nothing like a real model's tool
# selection. It exists only to pin that the six descriptions stay lexically
# distinguishable for a small set of realistic agent intents (in the spirit
# of Cognee's own pinned tool-search recall benchmark, research/memory-mcp-
# surface.md §C). A failure here means two descriptions have drifted close
# enough in vocabulary that even simple word overlap cannot tell them
# apart -- it does NOT predict what an actual LLM would pick.

_STOPWORDS = frozenset(
    "a an the this that these those is are was were be been being do does "
    "did to of in on at for with about and or not no use using used call "
    "calling it its i you your we our what when how me by from as one "
    "only if than need please".split()
)


def _tokenize(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOPWORDS and len(word) > 1
    }


def _lexical_rank(intent: str, tools: list[dict]) -> str:
    """Name of the tool with the most stopword-filtered token overlap with
    `intent` against `name + description`; ties break alphabetically."""
    intent_tokens = _tokenize(intent)
    scored = sorted(
        (
            (len(intent_tokens & _tokenize(tool["name"] + " " + tool["description"])), tool["name"])
            for tool in tools
        ),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return scored[0][1]


AGENT_INTENTS = [
    ("search past decisions about the migration", "fidelis_recall"),
    ("find prior work on the deploy port", "fidelis_recall"),
    ("an old memory is wrong, correct it now", "fidelis_correct"),
    ("that memory is outdated, fix it", "fidelis_correct"),
    ("what changed recently", "fidelis_recent"),
    ("list recent memories since yesterday", "fidelis_recent"),
    ("I need to fetch one specific memory by its id", "fidelis_get"),
    ("fetch the memory with id xyz", "fidelis_get"),
    ("is the memory server up", "fidelis_health"),
    ("check backend health and stats", "fidelis_health"),
    ("please remember this for later, verbatim", "fidelis_store"),
    ("remember that the meeting moved to friday", "fidelis_store"),
]


@pytest.mark.parametrize(("intent", "expected_tool"), AGENT_INTENTS)
def test_selection_lexical_proxy_ranks_intended_tool_first(monkeypatch, intent, expected_tool):
    monkeypatch.delenv("FIDELIS_ORIENT_ROUTE", raising=False)
    tools = _list_tools("2025-06-18")
    assert _lexical_rank(intent, tools) == expected_tool


def test_exposed_memory_tools_return_full_verbatim_text(monkeypatch):
    text = "A long stored memory. " * 60 + "END-OF-MEMORY"
    old = "Previous correction context. " * 30 + "END-OF-OLD"
    def post(path, payload, timeout=30.0):
        if path == "/query":
            return {"memories": [{"id": "new-id", "text": text}]}
        if path == "/recent":
            return {"records": [{"id": "new-id", "text": text}]}
        return {"id": "new-id", "text": text,
                "supersedes": [{"id": "old-id", "text": old, "status": "superseded"}]}
    monkeypatch.setattr(mcp_server, "_http_post", post)
    assert text in mcp_server._tool_fidelis_recall({"query": "memory"})
    assert text in mcp_server._tool_fidelis_recent({})
    got = mcp_server._tool_fidelis_get({"id": "new-id"})
    assert text in got and old in got
