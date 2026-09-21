"""
recall_b — zero-LLM structural query decomposition + multi-query vector fusion.

Strategy:
  1. Strip question scaffolding (what, why, how, describe, etc.) from query.
  2. Extract key tokens by removing stop words.
  3. Expand tokens using vocab_map (plain-English → technical terms bridge).
  4. Generate sub-queries: original → stripped phrase → expansion terms → bigrams → trigrams → tokens.
  5. Run up to MAX_SUBQUERIES vector searches independently.
  6. Merge with Reciprocal Rank Fusion (RRF); deduplicate by text.
  7. Cosine-rerank merged candidates against original query embedding.
  8. Apply score floor — return empty if top result below threshold.

No LLM at runtime. vocab_map written once by `fidelis calibrate`. Target latency: <200ms.
"""

from __future__ import annotations

import json as _json
import math
import re
import time
import urllib.request
from typing import Any

from fidelis.relation_envelope import (
    relation_envelopes_enabled,
    visible_payload_metadata,
)

from fidelis.degrade import _embed_bounded


# Words that don't contribute to retrieval; stripped before sub-query generation.
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "and", "or", "but", "if", "while", "because",
    "so", "not", "no", "nor", "yet", "than", "then", "about",
    # Question scaffolding words — add semantic noise without adding signal
    "cause", "caused", "describe", "explain", "tell", "give", "show",
    "find", "get", "make", "use", "used", "using", "their", "its",
    "did", "was", "were", "been", "being", "can", "cannot",
})

# Minimum token length to include as an individual sub-query.
_MIN_TOKEN_LEN = 4

# Max number of sub-queries to issue against the vector store.
MAX_SUBQUERIES = 8

# RRF constant — k=60 is empirically stable across retrieval tasks.
_RRF_K = 60

# Cosine rerank weight — blend RRF coverage with cosine precision.
# 0.0 = pure RRF order, 1.0 = pure cosine order.
# calibrated: 0.7 matched LongMemEval_S benchmark (470q); lower values reduce precision.
_COSINE_WEIGHT = 0.7

# Score floor — if top result cosine is below this AND the gap between
# top score and mean score is below _SCORE_GAP_MIN, return empty.
# calibrated: 0.25 floor + 0.1 gap derived from nomic-embed-text cosine distributions
# on fidelis dev set; fires <5% of queries in normal operation.
_COSINE_FLOOR = 0.25
_SCORE_GAP_MIN = 0.1

# Ollama embedding endpoint (same model as mem0 vector store).
_EMBED_MODEL = "nomic-embed-text"


def _tokenize(text: str) -> list[str]:
    """Lower-case, remove punctuation, split on whitespace."""
    return re.sub(r"[^\w\s-]", " ", text.lower()).split()


def _key_tokens(text: str) -> list[str]:
    """Return tokens that aren't stop words, preserving order."""
    return [t for t in _tokenize(text) if t not in _STOP and len(t) >= 2]


def _expand_with_vocab_map(
    tokens: list[str],
    query: str,
    vocab_map: dict[str, list[str]],
) -> list[str]:
    """
    Return additional search terms from vocab_map.

    Checks individual tokens, adjacent bigrams from key tokens, and
    substrings of the query against vocab_map keys (case-insensitive).
    Returns deduplicated expansion terms (each term separately, not joined).
    """
    expansion: list[str] = []
    seen: set[str] = set()
    query_lower = query.lower()

    def _add_terms(terms: list[str]) -> None:
        for t in terms:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                expansion.append(t)

    # Check individual tokens
    for tok in tokens:
        if tok in vocab_map:
            _add_terms(vocab_map[tok])

    # Check bigrams of key tokens
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in vocab_map:
            _add_terms(vocab_map[bigram])

    # Check any vocab_map key that appears as a substring in the query
    for key, terms in vocab_map.items():
        if key in query_lower and key not in {t for t in tokens}:
            _add_terms(terms)

    return expansion


def _build_subqueries(
    query: str,
    vocab_map: dict[str, list[str]] | None = None,
) -> tuple[list[str], bool]:
    """
    Generate up to MAX_SUBQUERIES search strings from query.

    Priority order:
      1. Original query (always first — preserves full semantic vector)
      2. Stripped key-token phrase (stop words removed)
      3. Vocabulary expansion terms (if vocab_map provided and matches found)
      4. Adjacent bigrams of key tokens
      5. All adjacent trigrams (for longer queries)
      6. Individual key tokens (length >= MIN_TOKEN_LEN)

    Returns (subqueries, expanded) where expanded=True if vocab expansion was used.
    """
    seen: set[str] = set()
    result: list[str] = []

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            result.append(q)

    _add(query)

    tokens = _key_tokens(query)
    if not tokens:
        return result, False

    # Stripped phrase
    _add(" ".join(tokens))

    # Vocab expansion terms (inserted before bigrams for priority)
    expanded = False
    if vocab_map:
        expansion_terms = _expand_with_vocab_map(tokens, query, vocab_map)
        for term in expansion_terms:
            _add(term)
        expanded = bool(expansion_terms)

    # All adjacent bigrams
    for i in range(len(tokens) - 1):
        _add(f"{tokens[i]} {tokens[i+1]}")

    # All adjacent trigrams (if query is long enough)
    for i in range(len(tokens) - 2):
        _add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    # Individual significant tokens
    for t in tokens:
        if len(t) >= _MIN_TOKEN_LEN:
            _add(t)

    return result[:MAX_SUBQUERIES], expanded


def _rrf_merge(
    runs: list[list[dict]],
    limit: int,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Each run is a list of {"text": str, "score": float} ordered best-first.
    Returns a unified list ordered by RRF score, deduplicated by text.
    """
    scores: dict[str, float] = {}
    canonical: dict[str, dict] = {}

    for run in runs:
        for rank, item in enumerate(run, 1):
            text = item.get("text", "")
            if not text:
                continue
            key = _candidate_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            if key not in canonical:
                canonical[key] = item

    merged = sorted(canonical.values(), key=lambda x: scores[_candidate_key(x)], reverse=True)
    return merged[:limit]


def _candidate_key(item: dict) -> str:
    """Stable identity when available; exact text is the legacy fallback."""
    return str(item.get("id") or item.get("record_id") or item.get("text") or "")


def _batch_embed(texts: list[str], cfg: dict[str, Any]) -> list[list[float]] | None:
    """Batch-embed texts via Ollama /api/embed. Returns list of vectors or None."""
    url = cfg.get("ollama_url", "http://localhost:11434")
    try:
        body = _json.dumps({"model": _EMBED_MODEL, "input": texts}).encode()
        req = urllib.request.Request(
            f"{url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        vecs = data.get("embeddings", [])
        return vecs if len(vecs) == len(texts) else None
    except Exception:
        return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure Python, no numpy dependency."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_rerank(
    query: str,
    candidates: list[dict],
    rrf_scores: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[list[dict], bool]:
    """
    Re-rank candidates by blending RRF score with cosine similarity to original query.

    Returns (reranked_list, was_filtered) where was_filtered=True if score floor
    triggered and the list is empty.
    """
    if not candidates:
        return [], False

    # Batch embed: query + all candidate texts in one call
    texts = [query] + [c["text"] for c in candidates]
    vecs = _batch_embed(texts, cfg)
    if vecs is None:
        # Embedding failed — fall back to RRF order
        return candidates, False

    query_vec = vecs[0]
    candidate_vecs = vecs[1:]

    # Compute cosine similarity for each candidate
    cosine_scores: list[float] = []
    for cv in candidate_vecs:
        cosine_scores.append(_cosine_sim(query_vec, cv))

    # Normalize RRF scores to [0, 1]
    rrf_vals = [rrf_scores.get(_candidate_key(c), 0.0) for c in candidates]
    rrf_max = max(rrf_vals) if rrf_vals else 1.0
    rrf_norm = [v / rrf_max if rrf_max > 0 else 0.0 for v in rrf_vals]

    # Blend: final = (1 - weight) * rrf_normalized + weight * cosine
    w = _COSINE_WEIGHT
    blended = [
        (1.0 - w) * rrf_n + w * cos
        for rrf_n, cos in zip(rrf_norm, cosine_scores)
    ]

    # Score floor gate: if top cosine is low AND no clear winner, return empty
    top_cosine = max(cosine_scores) if cosine_scores else 0.0
    mean_cosine = sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0
    score_gap = top_cosine - mean_cosine
    if top_cosine < _COSINE_FLOOR and score_gap < _SCORE_GAP_MIN:
        return [], True

    # Sort candidates by blended score descending
    paired = list(zip(candidates, blended, cosine_scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for cand, blend_score, cos_score in paired:
        cand = dict(cand)  # copy
        cand["score"] = round(cos_score, 4)
        reranked.append(cand)

    return reranked, False


def _pool_search(
    memory: Any,
    query: str,
    user_id: str,
    top_k: int,
    trace: Any | None = None,
    probe_id: str | None = None,
) -> dict[str, list[dict]]:
    """Direct vector-store candidate search — bypasses ``mem0.Memory.search()``.

    ``memory.search()`` routes through mem0 2.0.x's ``score_and_rank``, which
    (verified empirically, see ``server.py`` ``/query`` handler) returns a
    broken ``score=1.0`` for every result regardless of similarity. That bug
    doesn't just mislabel the score field — because RRF fusion here uses the
    *order* results come back in as each run's rank, a corrupted ranking
    corrupts fusion input before BM25/RRF/cosine ever runs. ``/query`` already
    works around this by embedding the query itself and calling
    ``memory.vector_store.search()`` directly; this helper does the same
    thing so ``recall_b`` (and ``recall_hybrid``, which imports this) get a
    properly-ordered candidate pool.

    Root-caused + measured 2026-07-20 against the live 30-query
    ``gold-set-v1.jsonl`` benchmark: swapping this in raised recall_b's R@5
    from 6.67% to 30.0% (simulated via the equivalent /query-backed pool
    before this patch landed). See ~/ai-infra/tools/loops/retrieval-hit-rate
    receipts for the pre/post numbers.

    Returns the same ``{"results": [{"memory": str, "score": float}, ...]}``
    shape ``memory.search()`` returns, so callers need no other changes.
    """
    started = time.perf_counter()
    try:
        qv = _embed_bounded(memory.embedding_model, query, memory_action="search")
        raw = memory.vector_store.search(
            query=query, vectors=[qv], top_k=top_k,
            filters={"user_id": user_id},
        )
        results = []
        for r in raw or []:
            payload = dict(r.payload or {})
            text = payload.get("data", "")
            if not text:
                continue
            record_id = (
                getattr(r, "id", None)
                or payload.get("id")
                or payload.get("memory_id")
                or payload.get("record_id")
            )
            results.append({
                "memory": text,
                "id": str(record_id) if record_id else None,
                "metadata": visible_payload_metadata(
                    payload,
                    enabled=relation_envelopes_enabled(),
                ),
                # mem0 2.x already returns a similarity, 1/(1+distance), in (0, 1];
                # larger = closer. Clamp only — same convention /query uses.
                "score": round(max(0.0, min(1.0, r.score or 0.0)), 3),
            })
    except Exception as exc:
        # Never propagate — a single sub-query's embed/search failure (Ollama
        # 500 on an oversized sub-query, transient timeout, malformed response)
        # must not abort the whole recall. Callers treat empty results the
        # same as a skipped sub-query.
        if trace is not None:
            try:
                trace.event(
                    "dense_admission",
                    "error",
                    probe_id=probe_id,
                    requested_count=top_k,
                    returned_count=0,
                    saturated=False,
                    failure_class=type(exc).__name__,
                    latency_ms=round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                )
            except Exception:
                pass
        return {"results": []}
    if trace is not None:
        try:
            trace.event(
                "dense_admission",
                "ok",
                probe_id=probe_id,
                requested_count=top_k,
                returned_count=len(results),
                saturated=len(results) >= top_k,
                latency_ms=round(
                    (time.perf_counter() - started) * 1000,
                    3,
                ),
            )
        except Exception:
            pass
    return {"results": results}


def recall_b(
    memory: Any,
    query: str,
    user_id: str,
    cfg: dict[str, Any],
    limit: int | None = None,
) -> tuple[list[dict], str]:
    """
    Zero-LLM recall.  Returns (memories, method).

    memories: list of {"text": str, "score": float}
    method:   "decompose_N" or "decompose_N_v" (v = vocab expansion applied)
    """
    limit = limit or cfg.get("recall_limit", 50)
    per_query_limit = min(limit, 20)  # per-sub-query cap; RRF widens coverage
    vocab_map: dict[str, list[str]] = cfg.get("vocab_map", {})

    subqueries, expanded = _build_subqueries(query, vocab_map if vocab_map else None)
    runs: list[list[dict]] = []

    for sq in subqueries:
        try:
            raw = _pool_search(memory, sq, user_id, per_query_limit)
        except Exception:
            # A single sub-query embed/search failure (Ollama 500 on an oversized
            # sub-query, transient timeout) must not abort the whole recall — skip it
            # and let RRF cover from the sub-queries that succeeded. Mirrors the
            # graceful-degradation already in _batch_embed and recall.py's filter.
            continue
        candidates = []
        for r in (raw or {}).get("results", []):
            if not r.get("memory"):
                continue
            candidate = {
                "text": r.get("memory", ""),
                "score": round(r.get("score", 9999), 3),
            }
            for key in ("id", "metadata", "source_pointer", "created_at", "recorded_at"):
                if r.get(key) is not None:
                    candidate[key] = r[key]
            candidates.append(candidate)
        if candidates:
            runs.append(candidates)

    if not runs:
        return [], "no_candidates"

    merged = _rrf_merge(runs, limit=limit)
    suffix = "_v" if expanded else ""

    # Build RRF score map for blending
    rrf_scores: dict[str, float] = {}
    for run in runs:
        for rank, item in enumerate(run, 1):
            key = _candidate_key(item)
            if key:
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    # Cosine rerank against original query
    reranked, filtered = _cosine_rerank(query, merged, rrf_scores, cfg)
    if filtered:
        return [], f"below_threshold_{len(runs)}{suffix}"

    method = f"decompose_{len(runs)}{suffix}"
    return reranked[:limit], method
