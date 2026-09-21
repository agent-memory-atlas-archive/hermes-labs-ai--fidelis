"""Liveness must not depend on Chroma, Ollama, or readiness probes."""

from __future__ import annotations

import io
import json
import threading
import time
from unittest.mock import MagicMock

from fidelis.server import make_handler


def _invoke_get(handler_cls: type, path: str) -> dict:
    handler = handler_cls.__new__(handler_cls)
    handler.wfile = io.BytesIO()
    handler.rfile = io.BytesIO()
    handler.headers = {"Content-Length": "0"}
    handler.path = path
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "GET"
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None

    handler.do_GET()
    return json.loads(handler.wfile.getvalue())


def _handler(memory: MagicMock) -> type:
    return make_handler(
        memory,
        {
            "user_id": "test-agent",
            "port": 0,
            "query_threshold": 250.0,
            "recall_limit": 5,
        },
    )


def test_live_does_not_touch_memory_dependencies() -> None:
    memory = MagicMock()
    memory.vector_store.collection.count.side_effect = AssertionError(
        "liveness must not call Chroma"
    )
    memory.embedding_model.embed.side_effect = AssertionError(
        "liveness must not call Ollama"
    )

    payload = _invoke_get(_handler(memory), "/live")
    assert payload["status"] == "alive"
    assert payload["readiness"] == "ready"
    memory.vector_store.collection.count.assert_not_called()
    memory.embedding_model.embed.assert_not_called()


def test_health_does_not_wait_for_slow_readiness_refresh() -> None:
    memory = MagicMock()
    release = threading.Event()

    def slow_count() -> int:
        release.wait(timeout=1.0)
        return 42

    memory.vector_store.collection.count.side_effect = slow_count
    memory.embedding_model.embed.return_value = [0.0]
    memory.vector_store.search.return_value = []
    handler_cls = _handler(memory)

    started = time.monotonic()
    payload = _invoke_get(handler_cls, "/health")
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.2, f"health blocked on readiness probe for {elapsed:.3f}s"
    assert payload["status"] == "degraded"


def test_primed_health_cache_serves_last_deep_result_without_reprobing() -> None:
    memory = MagicMock()
    memory.vector_store.collection.count.return_value = 42
    memory.embedding_model.embed.return_value = [0.0]
    memory.vector_store.search.return_value = []
    handler_cls = _handler(memory)

    handler_cls.prime_health()
    deadline = time.monotonic() + 1.0
    payload = {"probe_refreshing": True}
    while payload["probe_refreshing"] and time.monotonic() < deadline:
        time.sleep(0.01)
        payload = _invoke_get(handler_cls, "/health")
    assert memory.vector_store.collection.count.call_count == 1
    assert payload["status"] == "ok"
    memory.vector_store.collection.count.reset_mock()
    memory.vector_store.collection.count.side_effect = AssertionError(
        "fresh health cache must not re-enter Chroma"
    )

    payload = _invoke_get(handler_cls, "/health")

    assert payload["status"] == "ok"
    assert payload["count"] == 42
    memory.vector_store.collection.count.assert_not_called()


def test_stale_success_refreshes_to_degraded_off_path(monkeypatch) -> None:
    monkeypatch.setenv("FIDELIS_HEALTH_CACHE_TTL_SECS", "0")
    memory = MagicMock()
    memory.vector_store.collection.count.return_value = 42
    memory.embedding_model.embed.return_value = [0.0]
    memory.vector_store.search.return_value = []
    handler_cls = _handler(memory)
    handler_cls.prime_health()
    deadline = time.monotonic() + 1.0
    while not memory.vector_store.search.called and time.monotonic() < deadline:
        time.sleep(0.01)
    assert memory.vector_store.search.called

    memory.vector_store.collection.count.side_effect = RuntimeError("store unavailable")
    first = _invoke_get(handler_cls, "/health")

    assert first["status"] == "degraded"
    deadline = time.monotonic() + 1.0
    while memory.vector_store.collection.count.call_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    second = _invoke_get(handler_cls, "/health")
    assert second["status"] == "degraded"
    assert second["count"] == -1


def test_health_priming_does_not_block_on_slow_dependencies() -> None:
    memory = MagicMock()
    release = threading.Event()

    def slow_count() -> int:
        release.wait(timeout=1.0)
        return 42

    memory.vector_store.collection.count.side_effect = slow_count
    handler_cls = _handler(memory)

    started = time.monotonic()
    handler_cls.prime_health()
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.2, f"health priming blocked for {elapsed:.3f}s"


def test_refresh_thread_start_failure_is_degraded(monkeypatch) -> None:
    memory = MagicMock()
    handler_cls = _handler(memory)

    def fail_start(_thread) -> None:
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    payload = _invoke_get(handler_cls, "/health")

    assert payload["status"] == "degraded"
    assert payload["probe_refreshing"] is False
    assert payload["search_error"] == "refresh_start_failed"


def test_stuck_refresh_is_superseded_without_late_cache_overwrite(monkeypatch) -> None:
    monkeypatch.setenv("FIDELIS_HEALTH_REFRESH_MAX_AGE_SECS", "0")
    memory = MagicMock()
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0

    def count() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            release_first.wait(timeout=1.0)
            return 41
        return 42

    memory.vector_store.collection.count.side_effect = count
    memory.embedding_model.embed.return_value = [0.0]
    memory.vector_store.search.return_value = []
    handler_cls = _handler(memory)

    handler_cls.prime_health()
    assert first_entered.wait(timeout=1.0)
    handler_cls.prime_health()
    deadline = time.monotonic() + 1.0
    while calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    payload = _invoke_get(handler_cls, "/health")
    assert payload["status"] == "ok"
    assert payload["count"] == 42

    release_first.set()
    time.sleep(0.02)
    payload = _invoke_get(handler_cls, "/health")
    assert payload["count"] == 42


def test_post_write_invalidation_never_serves_old_count() -> None:
    memory = MagicMock()
    refresh_entered = threading.Event()
    release_refresh = threading.Event()
    calls = 0

    def count() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 42
        refresh_entered.set()
        release_refresh.wait(timeout=1.0)
        return 43

    memory.vector_store.collection.count.side_effect = count
    memory.embedding_model.embed.return_value = [0.0]
    memory.vector_store.search.return_value = []
    handler_cls = _handler(memory)
    handler_cls.prime_health()
    deadline = time.monotonic() + 1.0
    while memory.vector_store.collection.count.call_count < 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    handler_cls.invalidate_health_cache()
    assert refresh_entered.wait(timeout=1.0)
    pending = _invoke_get(handler_cls, "/health")
    assert pending["status"] == "degraded"
    assert pending["count"] == -1

    release_refresh.set()
    deadline = time.monotonic() + 1.0
    current = pending
    while current["probe_refreshing"] and time.monotonic() < deadline:
        time.sleep(0.01)
        current = _invoke_get(handler_cls, "/health")
    assert current["status"] == "ok"
    assert current["count"] == 43


def test_post_write_invalidation_supersedes_an_inflight_old_count() -> None:
    memory = MagicMock()
    old_probe_entered = threading.Event()
    release_old_probe = threading.Event()
    calls = 0

    def count() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            old_probe_entered.set()
            release_old_probe.wait(timeout=1.0)
            return 42
        return 43

    memory.vector_store.collection.count.side_effect = count
    memory.embedding_model.embed.return_value = [0.0]
    memory.vector_store.search.return_value = []
    handler_cls = _handler(memory)

    handler_cls.prime_health()
    assert old_probe_entered.wait(timeout=1.0)
    handler_cls.invalidate_health_cache()
    deadline = time.monotonic() + 1.0
    current = {"probe_refreshing": True}
    while current["probe_refreshing"] and time.monotonic() < deadline:
        time.sleep(0.01)
        current = _invoke_get(handler_cls, "/health")

    assert calls >= 2
    assert current["status"] == "ok"
    assert current["count"] == 43

    release_old_probe.set()
    time.sleep(0.02)
    assert _invoke_get(handler_cls, "/health")["count"] == 43
