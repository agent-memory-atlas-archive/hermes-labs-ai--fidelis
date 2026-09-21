"""Disposable HTTP/MCP proof for evidence-bound comparative link admission."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from fidelis import recall_hybrid, server
from fidelis.relation_envelope import (
    DECLARATIONS_FIELD,
    FEATURE_ENV,
    SCHEMA_VERSION,
)


ROOT = Path(__file__).parents[1]
QUERY = "What changed in Atlas Relay, and what did it use before?"

# One _log_call line per tool call/protocol error is now expected on stderr
# (PRD condition 1 / R17 stub); this matches its exact format so a stray
# print(), bare exception repr, or third-party warning still fails the test
# (independent-verifier recommendation, 2026-09-21).
_LOG_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z pid=\d+ \S+ \S+ [\d.]+ms \S+ \S+$"
)


class _Embedder:
    def embed(self, text, **kwargs):
        del kwargs
        return [float(len(str(text))), 1.0]


class _VectorStore:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.order: list[str] = []
        self.get_calls: list[str] = []
        self.search_calls: list[str] = []
        self.mode = "normal"

    def insert(self, *, vectors, payloads, ids):
        del vectors
        record_id = str(ids[0])
        self.records[record_id] = dict(payloads[0])
        if record_id not in self.order:
            self.order.append(record_id)

    def get(self, vector_id):
        stable_id = str(vector_id)
        self.get_calls.append(stable_id)
        payload = self.records[stable_id]
        return type(
            "Stored",
            (),
            {"id": stable_id, "payload": payload, "score": 0.0},
        )()

    def search(self, *, query, vectors, top_k, filters):
        del vectors, filters
        self.search_calls.append(str(query))
        if self.mode == "empty":
            return []
        selected = [
            record_id
            for record_id in self.order
            if record_id != "atlas-prior"
        ]
        hits = []
        for rank, record_id in enumerate(selected[:top_k]):
            hits.append(
                type(
                    "Hit",
                    (),
                    {
                        "id": record_id,
                        "payload": self.records[record_id],
                        "score": float(rank) / 100.0,
                    },
                )()
            )
        return hits


class _Memory:
    def __init__(self):
        self.embedding_model = _Embedder()
        self.vector_store = _VectorStore()


def _post(port: int, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _mcp_exchange(port: int, *, relations: bool) -> list[dict]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["FIDELIS_PORT"] = str(port)
    env["COGITO_HERMENEUTICS_EVIDENCE_STATUS"] = "1"
    if relations:
        env[FEATURE_ENV] = "1"
    else:
        env.pop(FEATURE_ENV, None)
    # PACKET F3 (MCP surface v2): `cogito_recall` is removed (all `cogito_*`
    # aliases were byte-identical duplicates, confirmed by the real-user
    # panel), so only `fidelis_recall` is exercised here now.
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "fidelis_recall",
                "arguments": {"query": QUERY, "limit": 5},
            },
        },
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "fidelis.mcp_server"],
        input="\n".join(json.dumps(request) for request in requests) + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=True,
    )
    # Every non-empty stderr line must be a well-formed _log_call line (or
    # its indented "detail:" continuation) -- tighter than "no crash noise":
    # also catches a stray print(), a bare exception repr, or a leaked
    # warning (independent-verifier recommendation, 2026-09-21).
    for stderr_line in completed.stderr.splitlines():
        if not stderr_line.strip():
            continue
        assert _LOG_LINE_RE.match(stderr_line) or stderr_line.startswith("    detail: "), (
            stderr_line
        )
    return [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def test_disposable_ingest_http_mcp_and_rollback(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(FEATURE_ENV, "1")
    monkeypatch.setenv("COGITO_HERMENEUTICS_EVIDENCE_STATUS", "1")
    monkeypatch.setattr(
        recall_hybrid,
        "_embed_docs",
        lambda texts, cfg: [[1.0, 1.0] for _ in texts],
    )
    monkeypatch.setattr(
        recall_hybrid,
        "_embed_queries",
        lambda texts, cfg: [[1.0, 1.0] for _ in texts],
    )

    memory = _Memory()
    httpd = server._BoundedThreadingHTTPServer(
        ("127.0.0.1", 0),
        server.make_handler(
            memory,
            {
                "user_id": "agent",
                "collection": "disposable-relation-e2e",
                "recall_limit": 30,
                "ephemera_filter": False,
                "supersession_pointers_path": None,
            },
        ),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = int(httpd.server_port)

    prior_text = (
        "On 2025-02-01, Atlas Relay previously used batch dispatch."
    )
    current_text = (
        "On 2026-07-26, Atlas Relay currently uses streaming dispatch. "
        "The migration replaced batch dispatch."
    )
    records = [
        {
            "id": "atlas-prior",
            "text": prior_text,
            "metadata": {
                "source": "atlas-operations-log",
                "source_pointer": "local://atlas/prior",
                "entity_literals": ["Atlas Relay"],
            },
        },
        {
            "id": "atlas-current",
            "text": current_text,
            "metadata": {
                "source": "atlas-change-ledger",
                "source_pointer": "local://atlas/current",
                "entity_literals": ["Atlas Relay"],
                DECLARATIONS_FIELD: [
                    {
                        "type": "supersedes",
                        "target_record_id": "atlas-prior",
                        "support_text": (
                            "Atlas Relay currently uses streaming dispatch."
                        ),
                        "declared_by": "source",
                    }
                ],
            },
        },
    ]
    records.extend(
        {
            "id": f"distractor-{index:02d}",
            "text": (
                f"On 2026-06-{(index % 28) + 1:02d}, Garden Node {index} "
                "records irrigation pressure."
            ),
            "metadata": {
                "source": "garden-log",
                "source_pointer": f"local://garden/{index}",
            },
        }
        for index in range(29)
    )

    try:
        stored = [
            _post(port, "/store", record)
            for record in records
        ]
        assert all(item["status"] == "stored" for item in stored)
        assert stored[1]["relation_envelope"]["schema_version"] == SCHEMA_VERSION
        relation = stored[1]["relation_envelope"]["relations"][0]
        assert relation["source_span"]["text"] in current_text
        assert relation["target_record_id"] == "atlas-prior"

        feature_on = _post(
            port,
            "/orient",
            {"text": QUERY, "automatic": False, "limit": 5},
        )
        assert feature_on["status"] == "supported_evidence"
        assert "links1" in feature_on["method"]
        assert feature_on["covered_facets"] == ["prior", "current"]
        assert {item["id"] for item in feature_on["evidence"]} >= {
            "atlas-prior",
            "atlas-current",
        }
        linked = {
            item["record_id"]: item
            for item in feature_on["evidence"]
            if item["record_id"] in {"atlas-prior", "atlas-current"}
        }
        assert linked["atlas-prior"]["raw_text"] == prior_text
        assert linked["atlas-current"]["raw_text"] == current_text
        assert linked["atlas-prior"]["raw_text_sha256"] == hashlib.sha256(
            prior_text.encode()
        ).hexdigest()
        assert linked["atlas-current"]["raw_text_sha256"] == hashlib.sha256(
            current_text.encode()
        ).hexdigest()
        assert memory.vector_store.get_calls == ["atlas-prior"]

        calls_before_ambiguous = len(memory.vector_store.get_calls)
        ambiguous = _post(
            port,
            "/orient",
            {
                "text": (
                    "What changed in the Hermeneutics integration, and what "
                    "did it do before?"
                ),
                "automatic": True,
                "limit": 5,
            },
        )
        assert "comparative_ambiguous" in ambiguous["plan"]["cues"]
        assert ambiguous["plan"]["evidence_needs"] == []
        assert len(memory.vector_store.get_calls) == calls_before_ambiguous

        memory.vector_store.mode = "empty"
        no_coverage = _post(
            port,
            "/orient",
            {
                "text": (
                    "What changed in Quartz Loom, and what did it use before?"
                ),
                "automatic": False,
                "limit": 5,
            },
        )
        assert no_coverage["status"] == "no_supported_result"
        assert no_coverage["evidence"] == []
        memory.vector_store.mode = "normal"

        searches_before_bypass = len(memory.vector_store.search_calls)
        bypass = _post(
            port,
            "/orient",
            {
                "text": "Translate this sentence into French.",
                "automatic": True,
                "limit": 5,
            },
        )
        assert bypass["status"] == "not_needed"
        assert len(memory.vector_store.search_calls) == searches_before_bypass

        # PACKET F3 (MCP surface v2): fidelis_recall's default fast path
        # (POST /query) does not follow relation-envelope links -- the fake
        # vector store's search() here deliberately excludes "atlas-prior"
        # from direct search results (only reachable via the link the OLD
        # orient-cascade path followed), so the relation-envelope-LINKED
        # evidence this section used to prove through the `fidelis_recall`
        # tool name is still fully proven above via the HTTP /orient
        # assertions, and separately in
        # tests/test_evidence_status_integration.py /
        # tests/test_evidence_status_adversarial.py, which call the
        # unchanged `_tool_recall`/`_tool_recall_structured` functions
        # directly (PRD D2: the functions are unchanged, only their
        # reachability via the `fidelis_recall` NAME changed). This section
        # now proves what is still true of the new tool: it is listed (not
        # `cogito_recall`, which is removed), dispatchable over a real
        # subprocess against a real backend, and returns the directly-
        # searchable record without error.
        mcp_on = _mcp_exchange(port, relations=True)
        assert mcp_on[0]["result"]["protocolVersion"] == "2025-06-18"
        tool_names = {
            item["name"] for item in mcp_on[1]["result"]["tools"]
        }
        assert "fidelis_recall" in tool_names
        assert "cogito_recall" not in tool_names
        for response in mcp_on[2:]:
            assert "error" not in response, response
            assert response["result"].get("isError") is not True, response
            text = " ".join(
                c.get("text", "") for c in response["result"].get("content", [])
            )
            assert "atlas-current" in text

        monkeypatch.delenv(FEATURE_ENV)
        gets_before_rollback = len(memory.vector_store.get_calls)
        rollback_http = _post(
            port,
            "/orient",
            {"text": QUERY, "automatic": False, "limit": 5},
        )
        assert "links" not in rollback_http["method"]
        assert "atlas-prior" not in {
            item["id"] for item in rollback_http["evidence"]
        }
        assert len(memory.vector_store.get_calls) == gets_before_rollback
        assert "evidence_needs" not in rollback_http["plan"]

        mcp_off = _mcp_exchange(port, relations=False)
        for response in mcp_off[2:]:
            assert "error" not in response, response
            assert response["result"].get("isError") is not True, response
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
