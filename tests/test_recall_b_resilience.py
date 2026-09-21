"""
Unit tests for sub-query search resilience in fidelis.recall_b.

recall_b decomposes a query into several sub-queries and runs _pool_search()
(direct memory.vector_store.search(), bypassing mem0's broken
memory.search()/score_and_rank wrapper — see recall_b._pool_search's
docstring) on each, merging with RRF. A single sub-query's embed/search can
fail — the embedding model (nomic-embed-text) returns HTTP 500 past ~2048
tokens, and Ollama can time out transiently. A failed sub-query is skipped
and RRF covers from the sub-queries that succeeded — mirroring the graceful
degradation already in _batch_embed (embed failure -> RRF order) and recall.py's
filter (LLM down -> passthrough). _pool_search itself never raises (internal
try/except returns {"results": []} on any embed/search failure); recall_b's
per-subquery try/except is therefore a second, redundant layer of the same
guarantee.

_batch_embed is monkeypatched to None throughout so the cosine-rerank stage takes
its no-network RRF fallback — these tests touch no Ollama, no ChromaDB.
"""

from __future__ import annotations

import sys

from fidelis.recall_b import recall_b


def _no_network_rerank(monkeypatch):
    # Force _cosine_rerank's "embedding failed -> RRF order" branch so no test
    # needs a live Ollama embed endpoint.
    mod = sys.modules["fidelis.recall_b"]
    monkeypatch.setattr(mod, "_batch_embed", lambda texts, cfg: None)


class _FakeHit:
    """Stand-in for mem0 vector_store.search() hits (.payload / .score)."""

    def __init__(self, text: str, score: float = 0.5):
        self.payload = {"data": text}
        self.score = score


class _FakeEmbeddingModel:
    def embed(self, text, memory_action=None):
        return [0.0, 0.0, 0.0, 0.0]


def _fake_memory(vector_store):
    """Assemble a minimal mem0-shaped object: .embedding_model + .vector_store
    (the interface _pool_search uses)."""
    mem = type("FakeMemory", (), {})()
    mem.embedding_model = _FakeEmbeddingModel()
    mem.vector_store = vector_store
    return mem


def test_failing_subquery_is_skipped_recall_still_returns(monkeypatch):
    _no_network_rerank(monkeypatch)

    class FlakyVectorStore:
        def __init__(self):
            self.calls = []

        def search(self, *, query, vectors, top_k, filters):
            self.calls.append(query)
            if "token" in query:  # the full query + any sub-query containing 'token' 500s
                raise RuntimeError("simulated nomic-embed 500 (oversized)")
            return [_FakeHit(f"hit::{query}")]

    vs = FlakyVectorStore()
    mem = _fake_memory(vs)
    out, method = recall_b(mem, "embedding token limit failure", user_id="agent", cfg={})

    # The failing sub-query family WAS attempted (the original full query contains 'token')...
    assert any("token" in c for c in vs.calls)
    # ...yet recall did not crash and still returned results from the survivors.
    assert out, "recall_b returned nothing despite token-free sub-queries succeeding"
    assert method.startswith("decompose_")


def test_all_subqueries_fail_returns_empty_not_crash(monkeypatch):
    _no_network_rerank(monkeypatch)

    class AllFailVectorStore:
        def search(self, *, query, vectors, top_k, filters):
            raise RuntimeError("every embed 500s")

    out, method = recall_b(_fake_memory(AllFailVectorStore()), "anything at all here", user_id="agent", cfg={})
    assert out == []
    assert method == "no_candidates"


def test_none_search_result_is_handled(monkeypatch):
    _no_network_rerank(monkeypatch)

    class NoneVectorStore:
        def search(self, *, query, vectors, top_k, filters):
            return None  # exercises the (raw or []) guard in _pool_search

    out, method = recall_b(_fake_memory(NoneVectorStore()), "some query words here", user_id="agent", cfg={})
    assert out == []
    assert method == "no_candidates"


def test_normal_path_unaffected_by_guard(monkeypatch):
    _no_network_rerank(monkeypatch)

    class OKVectorStore:
        def search(self, *, query, vectors, top_k, filters):
            return [_FakeHit("the verbatim answer")]

    out, method = recall_b(_fake_memory(OKVectorStore()), "what is the answer", user_id="agent", cfg={})
    assert out
    assert any("verbatim answer" in m["text"] for m in out)
    assert method.startswith("decompose_")
