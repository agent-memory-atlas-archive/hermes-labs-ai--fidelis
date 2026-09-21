"""Public write-fallback contract, exercised without Ollama or ChromaDB."""

from __future__ import annotations

import argparse
import io
import json

import pytest
from unittest.mock import MagicMock

from fidelis import cli
from fidelis.server import make_handler


class _Hit:
    def __init__(self, text: str):
        self.payload = {"data": text, "user_id": "agent"}
        self.score = 0.0


class _VectorStore:
    def __init__(self):
        self.rows: list[tuple[str, str]] = []

    def insert(self, *, vectors, payloads, ids):
        del vectors
        self.rows.extend((record_id, payload["data"]) for record_id, payload in zip(ids, payloads))

    def search(self, *, query, vectors, top_k, filters):
        del query, vectors, filters
        return [_Hit(text) for _, text in self.rows[:top_k]]


class _Embedder:
    def embed(self, text, *args, **kwargs):
        del text, args, kwargs
        return [0.0] * 8


class _EmptyExtractionMemory:
    def __init__(self):
        self.embedding_model = _Embedder()
        self.vector_store = _VectorStore()

    def add(self, text, user_id):
        del text, user_id
        return {"results": []}


def _post_to_handler(handler_cls, path: str, body: dict) -> dict:
    payload = json.dumps(body).encode()

    class _Capture:
        def __init__(self):
            self.written: list[bytes] = []

        def write(self, data):
            self.written.append(data)

        def flush(self):
            pass

    handler = handler_cls.__new__(handler_cls)
    handler.wfile = _Capture()
    handler.rfile = io.BytesIO(payload)
    handler.headers = {
        "Content-Length": str(len(payload)),
        "Content-Type": "application/json",
    }
    handler.path = path
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "POST"
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None
    handler.do_POST()
    return json.loads(b"".join(handler.wfile.written))


def test_add_fallback_is_explicit_and_recallable_deterministically():
    memory = _EmptyExtractionMemory()
    handler_cls = make_handler(memory, {"user_id": "agent", "vocab_map": {}})

    stored = _post_to_handler(handler_cls, "/add", {"text": "synthetic durable fact"})

    assert stored["status"] == "stored"
    assert stored["count"] == 1
    assert stored["memories"] == ["synthetic durable fact"]
    assert stored["degraded"] == "verbatim-fallback-empty-extraction"
    assert stored["id"]

    first = _post_to_handler(handler_cls, "/query", {"text": "durable", "limit": 3})
    second = _post_to_handler(handler_cls, "/query", {"text": "durable", "limit": 3})
    assert first == second
    assert first["memories"][0]["text"] == "synthetic durable fact"


def test_cli_degraded_write_exits_zero_with_stable_status(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_post",
        lambda path, payload: {
            "status": "stored",
            "count": 1,
            "memories": [payload["text"]],
            "degraded": "verbatim-fallback-empty-extraction",
            "id": "fallback-uuid",
        },
    )

    cli.cmd_add(argparse.Namespace(text=["raw", "input"], extract=True))

    captured = capsys.readouterr()
    assert captured.out.strip() == (
        "status=stored degraded=verbatim-fallback-empty-extraction "
        "id=fallback-uuid count=1"
    )
    assert "stored verbatim" in captured.err.lower()
    assert "Added" not in captured.out


def test_cli_successful_extraction_output_is_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_post",
        lambda path, payload: {
            "status": "stored",
            "count": 1,
            "memories": ["extracted fact"],
        },
    )

    cli.cmd_add(argparse.Namespace(text=["raw", "input"], extract=True))

    captured = capsys.readouterr()
    assert "Added 1 memories." in captured.out
    assert "extracted fact" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("status", ["stored", "duplicate", "rejected", "unknown"])
def test_cli_verbatim_write_reports_actual_outcome(monkeypatch, capsys, status):
    monkeypatch.setattr(cli, "_post", lambda *a, **kw: {"status": status, "id": "record-one", "reason": "test refusal"})
    if status in {"rejected", "unknown"}:
        with pytest.raises(SystemExit) as exc:
            cli._store_verbatim("durable test fact")
        assert exc.value.code == 1
    else:
        cli._store_verbatim("durable test fact")
    output = capsys.readouterr()
    if status == "stored":
        assert "Stored 1 memory" in output.out
    else:
        assert "Stored 1 memory" not in output.out
    if status == "duplicate":
        assert "Duplicate" in output.out
    if status == "rejected":
        assert "Rejected" in output.err


@pytest.mark.parametrize("protocol", ["2024-11-05", "2025-03-26", "2025-06-18"])
def test_mcp_queued_relation_write_never_claims_stored(monkeypatch, protocol):
    from fidelis import mcp_server as mcp

    monkeypatch.setattr(mcp, "_negotiated_protocol", protocol)
    monkeypatch.setattr(mcp, "_http_post", lambda *a: {
        "status": "queued", "id": "pending-record", "reason": "store unavailable",
    })
    text = "Atlas now uses streaming dispatch."
    result = mcp._tool_store({
        "text": text,
        "metadata": {mcp.DECLARATIONS_FIELD: [{
            "type": "supersedes", "target_record_id": "atlas-prior",
            "support_text": text, "declared_by": "caller",
        }]},
    })
    if protocol == "2025-06-18":
        assert result["status"] == "queued"
    else:
        assert "NOT yet stored" in result
        assert "stored memory:" not in result
