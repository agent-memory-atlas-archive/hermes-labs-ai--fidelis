"""
cogito server — HTTP server keeping memory warm in process.

Endpoints:
  GET  /health
       → {"status": "ok", "count": N, "version": "..."}

  POST /query   {"text": "...", "limit": 5}
       → {"memories": [{"text": "...", "score": N}]}
       Narrow search, L2 threshold filter only. Fast.

  POST /recall  {"text": "...", "limit": 50, "threshold": 400}
       → {"memories": [...], "method": "filter"|"fallback_*"}
       Broad search + cheap-LLM integer-pointer filter. Smart.

  POST /recall_hybrid  {"text": "...", "limit": 50, "tier": "filter", "top_k": 5}
       → {"memories": [...], "method": "hybrid_*|..."}
       BM25 + dense + RRF + tiered LLM escalation.
       tier is one of: "zero_llm" (default, local, no retrieval LLM) |
       "filter" (experimental) | "flagship" (experimental). Historical
       flagship benchmark claims are retired because of harness defects.

  POST /orient  {"text": "...", "limit": 5, "automatic": true}
       → Cogito Hermeneutics V0.4 plan + provenance-preserving memories.
       ``automatic=true`` may return ``retrieval_status=not_needed`` without
       searching. ``automatic=false`` is the explicit-recall path.

  POST /store   {"text": "...", "id": "<optional id>", "metadata": {...}}
       → {"id": "...", "text": "..."}
       Write one memory verbatim — no extraction LLM, agent decides content.
       This is the preferred write path. Use /add only if you want mem0
       extraction to summarise raw text for you.

  POST /add     {"text": "..."}
       → {"count": N, "memories": [...]}
       Feeds text through mem0's extraction LLM before storing. Use when
       you have raw/unstructured text and want automatic summarisation.

Start:
  fidelis-server                        # uses .cogito.json or env vars
  fidelis-server --config /path/to.json
  fidelis-server --port 19420
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import time
import os
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fidelis import __version__
from fidelis.browse import get_record, recent_records, store_stats
from fidelis.codex_history import is_codex_history_query, search_codex_history
from fidelis.cogito_hermeneutics import (
    ENGINE_NAME as HERMENEUTICS_ENGINE,
)
from fidelis.cogito_hermeneutics import (
    ENGINE_VERSION as HERMENEUTICS_VERSION,
)
from fidelis.cogito_hermeneutics import (
    QualityFlags,
    apply_hermeneutics,
    build_refined_query,
    candidate_record_id,
    choose_refinement_reason,
    load_retrieval_brief,
    merge_candidate_attempts,
    plan_retrieval,
)
from fidelis.config import load, mem0_config
from fidelis.degrade import (
    _embed_bounded,
    configure_temporal,
    dead_count,
    queued_count,
    reconcile_temporal_index,
    replay_queue,
    safe_add,
    temporal_index,
)
from fidelis.evidence_status import (
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
    build_evidence_envelope,
    evidence_status_enabled,
)
from fidelis.inquiry import (
    ASK_SCHEMA_VERSION,
    INQUIRY_SCHEMA_VERSION,
    MAX_PASSES,
    POLICY_SCHEMA_VERSION,
    run_inquiry,
)
from fidelis.recall import recall as do_recall
from fidelis.recall_b import recall_b as do_recall_b
from fidelis.recall_hybrid import recall_hybrid as do_recall_hybrid
from fidelis.retrieval_trace import (
    DEFAULT_TRACE_MAX_BYTES,
    RetrievalTrace,
)
from fidelis.retrieval_telemetry import build_event as build_retrieval_event
from fidelis.retrieval_telemetry import record as record_retrieval_event
from fidelis.relation_envelope import (
    DECLARATIONS_FIELD,
    EXTRACTOR_VERSION as RELATION_EXTRACTOR_VERSION,
    RELATION_SEMANTICS_VERSION,
    SCHEMA_VERSION as RELATION_SCHEMA_VERSION,
    SUPPORTED_RELATION_TYPES,
    relation_envelopes_enabled,
)
from fidelis.snapshot import _read_snapshot, _snapshot_path
from fidelis.supersession import (
    filter_ephemera,
    mark_ephemera,
    mark_superseded,
)
from fidelis.temporal import format_instant
from fidelis.temporal_recall import carry_payload, overfetch, parse_as_of, temporal_view

logger = logging.getLogger("cogito.server")

_TEMPORAL_STORE_KEYS = ("event_at", "valid_from", "valid_to", "supersedes", "source")


def _hit_id(hit) -> dict:
    """``{"id": ...}`` when a vector-store hit exposes one, else ``{}``.

    Not every store result carries an id; the temporal view recovers a missing
    one by content hash, so its absence must never break a recall.
    """
    record_id = getattr(hit, "id", None)
    return {"id": str(record_id)} if record_id else {}


def _temporal_request(data: dict):
    """(as_of, historical) from a recall body. ValueError on a bad ``as_of``."""
    return parse_as_of(data.get("as_of")), bool(data.get("historical", False))

_MAX_STORE_METADATA_BYTES = 16_384
_MAX_RELATION_DECLARATIONS = 16
_PUBLIC_INQUIRY_POLICY_KEYS = {
    "maximum_passes",
    "maximum_selected_records",
}
_PUBLIC_INQUIRY_OVERRIDE_KEYS = {"operation"}


def _feature_state(check) -> dict:
    try:
        return {"valid": True, "enabled": bool(check())}
    except ValueError as exc:
        return {"valid": False, "enabled": None, "error": str(exc)}


def _runtime_capabilities(readiness: str) -> dict:
    return {
        "schema_version": "fidelis.runtime-capabilities/v1",
        "package_version": __version__,
        "engine": {
            "name": HERMENEUTICS_ENGINE,
            "version": HERMENEUTICS_VERSION,
            "zero_runtime_generative_llm_default": True,
        },
        "readiness": readiness,
        "features": {
            "bounded_inquiry": {
                "valid": True,
                "enabled": True,
                "opt_in": True,
                "maximum_passes": MAX_PASSES,
                "ask_schema_version": ASK_SCHEMA_VERSION,
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "result_schema_version": INQUIRY_SCHEMA_VERSION,
                "transports": ["python", "http", "cli", "mcp"],
                "mcp_protocols": ["2025-06-18"],
            },
            "evidence_status": {
                **_feature_state(evidence_status_enabled),
                "schema_version": EVIDENCE_SCHEMA_VERSION,
            },
            "relation_envelopes": {
                **_feature_state(relation_envelopes_enabled),
                "schema_version": RELATION_SCHEMA_VERSION,
                "extractor_version": RELATION_EXTRACTOR_VERSION,
                "semantics_version": RELATION_SEMANTICS_VERSION,
            },
        },
        "client_tool_invocation_required": True,
        "claim_limit": (
            "Capability state does not establish corpus coverage, evidence "
            "truth, completeness, or invocation outside a tested host."
        ),
    }


def _validate_store_transport(data: dict, text: str) -> str | None:
    record_id = data.get("id")
    if record_id is not None and (
        not isinstance(record_id, str)
        or not 3 <= len(record_id) <= 128
        or not all(ch.isalnum() or ch in "_.:-" for ch in record_id)
    ):
        return "invalid stable id"
    metadata = data.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return "metadata must be an object"
    if len(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ) > _MAX_STORE_METADATA_BYTES:
        return "metadata exceeds 16384 bytes"
    declarations = metadata.get(DECLARATIONS_FIELD, [])
    if not isinstance(declarations, list):
        return f"metadata.{DECLARATIONS_FIELD} must be an array"
    if len(declarations) > _MAX_RELATION_DECLARATIONS:
        return f"metadata.{DECLARATIONS_FIELD} exceeds 16 entries"
    for declaration in declarations:
        if not isinstance(declaration, dict):
            return "relation declaration must be an object"
        if set(declaration) != {
            "type",
            "target_record_id",
            "support_text",
            "declared_by",
        }:
            return "relation declaration has invalid fields"
        if declaration.get("declared_by") not in {"caller", "source"}:
            return "relation declaration has invalid declared_by"
        if not all(
            isinstance(declaration.get(field), str)
            and bool(declaration[field].strip())
            for field in ("type", "target_record_id", "support_text")
        ):
            return "relation declaration has missing fields"
        if declaration["type"] not in SUPPORTED_RELATION_TYPES:
            return "relation declaration has unsupported type"
        if declaration["support_text"] not in text:
            return "relation declaration support_text is not verbatim"
    return None


def _validate_inquiry_transport(
    policy: object,
    overrides: object,
    include_trace: object,
) -> str | None:
    if policy != "auto" and not isinstance(policy, dict):
        return "policy must be 'auto' or an object"
    if isinstance(policy, dict):
        unknown = set(policy) - _PUBLIC_INQUIRY_POLICY_KEYS
        if unknown:
            return (
                "unsupported public policy fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        maximum_passes = policy.get("maximum_passes")
        if maximum_passes is not None and (
            isinstance(maximum_passes, bool)
            or not isinstance(maximum_passes, int)
            or not 1 <= maximum_passes <= MAX_PASSES
        ):
            return f"maximum_passes must be an integer from 1 to {MAX_PASSES}"
        maximum_selected = policy.get("maximum_selected_records")
        if maximum_selected is not None and (
            isinstance(maximum_selected, bool)
            or not isinstance(maximum_selected, int)
            or not 1 <= maximum_selected <= 12
        ):
            return "maximum_selected_records must be an integer from 1 to 12"
    if overrides is not None and not isinstance(overrides, dict):
        return "overrides must be an object"
    if isinstance(overrides, dict):
        unknown = set(overrides) - _PUBLIC_INQUIRY_OVERRIDE_KEYS
        if unknown:
            return (
                "unsupported public override fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        operation = overrides.get("operation")
        if operation is not None and operation not in {
            "verify_claim",
            "evaluate_decision",
            "trace_evolution",
        }:
            return "unsupported inquiry operation"
    if not isinstance(include_trace, bool):
        return "include_trace must be a boolean"
    return None

# Shared executor for /recall's bounded decompose pipeline. Deliberately
# module-level: a `with ThreadPoolExecutor(...)` per request looks equivalent
# but __exit__ calls shutdown(wait=True), which blocks the handler thread until
# the submitted task ACTUALLY finishes — the result(timeout=...) "timeout" was
# cosmetic (verified empirically 2026-07-18), so slow recalls held their
# threads hostage anyway. Bounded workers also cap how many orphaned slow
# futures can pile up after their callers have timed out and moved on.
_RECALL_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="fidelis-recall"
)


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Cap concurrent in-flight request THREADS (not just accept rate).

    ThreadingMixIn.process_request returns as soon as the worker thread is
    spawned, so a semaphore released there would bound nothing; hold it for
    the worker's full lifetime by releasing in process_request_thread.
    Saturation behavior: refuse (close) new connections after a short wait
    instead of stacking unbounded threads the server will never service —
    clients already treat connection failure like a timeout (D4a)."""

    _MAX_INFLIGHT = 64

    def __init__(self, *args, **kwargs):
        import threading
        self._inflight = threading.Semaphore(self._MAX_INFLIGHT)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._inflight.acquire(timeout=5):
            logger.warning("[fidelis] saturated (%d in-flight); refusing connection",
                           self._MAX_INFLIGHT)
            try:
                self.shutdown_request(request)
            except Exception:
                pass
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._inflight.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._inflight.release()


class MemoryHolder:
    """Thread-safe, once-only, retry-on-failure lazy initializer for the mem0
    Memory object.

    Registry inspectors (Glama et al.) start the container's console-script
    entry point without a reachable Ollama and only probe GET /health. The
    real Memory object depends on mem0's ollama embedder, whose
    `_ensure_model_exists()` raises a ConnectionError when Ollama is down
    (server.py `_boot`, historically called eagerly before the HTTP server
    even bound). This holder defers that call to first use by a request
    handler that actually needs the store, caches a successful construction
    forever, and caches the last failure only until the next request asks
    for the memory again — so a transient outage does not permanently
    poison the process; each request after a failure gets a fresh attempt.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._memory: object | None = None
        self._last_error: Exception | None = None
        self._last_error_at: float = 0.0
        # Minimum seconds between retry attempts after a failure. Prevents
        # a burst of concurrent requests (or a registry inspector probing
        # /query, /recall, etc. in a tight loop) from retrying `_boot` —
        # which dials Ollama over the network — once per request. A single
        # request past the cooldown gets the real retry (holding `_lock`);
        # everything else in that window reuses the cached failure.
        self._retry_cooldown_s: float = self._parse_retry_cooldown(
            os.environ.get("FIDELIS_MEMORY_RETRY_COOLDOWN_SECS")
        )

    @staticmethod
    def _parse_retry_cooldown(raw: str | None, default: float = 5.0) -> float:
        """Parse FIDELIS_MEMORY_RETRY_COOLDOWN_SECS defensively.

        A malformed, negative, NaN, or infinite value must never abort
        startup — fall back to the documented default instead."""
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value) or value < 0:
            return default
        return value

    @property
    def ready(self) -> bool:
        return self._memory is not None

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def get(self) -> object:
        """Return the constructed Memory, building it on first call, or
        retrying no more than once per cooldown window after a failure.
        Raises the underlying exception (fresh or cached) on failure —
        never caches a permanent poison state, but never hammers Ollama
        on every single request either."""
        with self._lock:
            if self._memory is not None:
                return self._memory
            if (
                self._last_error is not None
                and time.monotonic() - self._last_error_at < self._retry_cooldown_s
            ):
                raise self._last_error
            try:
                memory = _boot(self._cfg)
                configure_temporal(self._cfg.get("store_path"), self._cfg)
                reconcile_temporal_index(memory)
            except Exception as e:
                self._last_error = e
                # Timestamp the failure itself, not the moment we entered
                # this call — _boot() dials Ollama over the network and can
                # take a while to time out, so measuring from entry would
                # under-count the cooldown window.
                self._last_error_at = time.monotonic()
                raise
            self._memory = memory
            self._last_error = None
            return memory

    def __getattr__(self, name):
        return getattr(self.get(), name)


def _boot(cfg: dict) -> object:
    """Import mem0 from wherever it's installed and return a Memory instance."""
    # mem0's supported telemetry switch must be set before importing mem0.
    # Its PostHog worker is non-daemon in the currently pinned dependency and
    # otherwise keeps Python alive after Fidelis has completed clean shutdown.
    # The installer already emits the same setting; keep direct module/CLI
    # launches private and shutdown-safe as well.
    os.environ.setdefault("MEM0_TELEMETRY", "False")

    # Support venv via COGITO_SITE_PACKAGES or system install
    site = os.environ.get("COGITO_SITE_PACKAGES")
    if site and site not in sys.path:
        sys.path.insert(0, site)

    os.environ.setdefault("MEM0_TELEMETRY", "False")

    from mem0 import Memory  # type: ignore

    m = Memory.from_config(mem0_config(cfg))
    _bound_ollama_timeouts(m)
    return m


def _bound_ollama_timeouts(memory: object, secs: float = 10.0) -> None:
    """mem0's Ollama LLM/embedder clients (ollama.Client -> httpx.Client)
    default to timeout=None (infinite). A slow or overloaded Ollama then hangs
    a request thread forever, and ThreadingHTTPServer spawns a thread per
    connection with no bound — together the confirmed root cause of the D4a
    starvation (CLOSE_WAIT pileup, erratic tail latency, canary restarts;
    DEEPDIVE-20260718). No mem0/ollama constructor knob exists for this, so
    bound the already-constructed httpx.Client post hoc — httpx.Client.timeout
    has a public setter, making this a supported mutation."""
    try:
        import httpx
    except Exception:
        return
    for attr in ("embedding_model", "llm"):
        inner = getattr(getattr(memory, attr, None), "client", None)
        inner = getattr(inner, "_client", None)
        if isinstance(inner, httpx.Client):
            inner.timeout = httpx.Timeout(secs)


def _chroma_index_lag(cfg: dict) -> int | None:
    """Uncompacted WAL backlog for the configured collection's vector segment.

    /health's "queued" only counts fidelis's own degrade queue (Ollama-down
    fallback) and stayed 0 on 2026-07-22 while 100+ writes sat in chroma's
    embeddings_queue awaiting the vector segment's threshold flush. Those
    writes are already searchable — chroma serves WAL entries alongside the
    HNSW segment, and that day's "fresh-write lag" traced to ranking, not
    index visibility — so index_lag measures compaction debt, not a search
    gap. Reads chroma's sqlite read-only: counts cfg["collection"]'s own WAL
    rows (topic carries the collection id) past its VECTOR segment's applied
    watermark. Both sides must be collection-filtered — a global MAX(seq_id)
    minuend charges this collection for any later write to a sibling
    collection, and reported the full ~103k WAL for a collection with no
    watermark row. A missing watermark row means nothing applied yet: all of
    the collection's queued rows count. None = unreadable store or unknown
    collection (never fails the health check over it).
    """
    try:
        db = Path(cfg["store_path"]).expanduser() / "chroma.sqlite3"
        if not db.exists():
            return None
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
        try:
            coll = con.execute(
                "SELECT id FROM collections WHERE name = ?", (cfg["collection"],)
            ).fetchone()
            if coll is None:
                return None
            row = con.execute(
                "SELECT COUNT(*) FROM embeddings_queue q"
                " WHERE q.topic LIKE '%' || ?"
                "   AND q.seq_id > COALESCE((SELECT m.seq_id FROM max_seq_id m"
                "        JOIN segments s ON CAST(m.segment_id AS TEXT) = s.id"
                "        WHERE s.scope = 'VECTOR' AND s.collection = ?), 0)",
                (coll[0], coll[0]),
            ).fetchone()
            return int(row[0]) if row else None
        finally:
            con.close()
    except Exception:
        logger.debug("health: chroma index-lag probe failed", exc_info=True)
        return None


def _run_cogito_hermeneutics(memory: object, cfg: dict, data: dict) -> dict:
    """Run the integrated V0.4 read layer without corpus mutation."""
    text = str(data.get("text") or "")
    limit = max(1, min(int(data.get("limit", 5)), 50))
    automatic = bool(data.get("automatic", True))
    session_id = str(data.get("session_id") or "host-process")[:128]
    turn_id = str(data.get("turn_id") or "unavailable")[:128]
    recent_turns = data.get("recent_turns")
    if not isinstance(recent_turns, list):
        recent_turns = []
    explicit_mode = data.get("mode")
    status_enabled = evidence_status_enabled()
    source_health: list[dict] = []
    trace = None
    if data.get("trace") is True:
        try:
            trace_max_bytes = int(
                data.get("trace_max_bytes", DEFAULT_TRACE_MAX_BYTES)
            )
        except (TypeError, ValueError):
            trace_max_bytes = DEFAULT_TRACE_MAX_BYTES
        trace = RetrievalTrace(text, max_bytes=trace_max_bytes)
    plan = plan_retrieval(
        text,
        recent_turns=[str(turn) for turn in recent_turns[-4:]],
        automatic=automatic,
        explicit_mode=str(explicit_mode) if explicit_mode else None,
    )
    history_intent = is_codex_history_query(text)
    if history_intent and plan.mode == "none" and not explicit_mode:
        plan = plan_retrieval(
            text,
            recent_turns=[str(turn) for turn in recent_turns[-4:]],
            automatic=automatic,
            explicit_mode="referential",
        )
    if trace is not None:
        trace.set_plan(plan, text)

    def _with_trace(result: dict) -> dict:
        if trace is None:
            return result
        try:
            trace.record_selection(result.get("records", []))
            traced = dict(result)
            traced["retrieval_trace"] = trace.finalize()
            return traced
        except Exception:
            # Observability must never alter the canonical retrieval result.
            return result

    def _finalize(result: dict) -> dict:
        traced = _with_trace(result)
        if not status_enabled:
            return traced
        return build_evidence_envelope(
            query=text,
            result=traced,
            source_health=source_health,
            recent_turns=[str(turn) for turn in recent_turns[-4:]],
        )

    flags = QualityFlags.from_environment()
    if plan.mode == "none":
        if trace is not None:
            trace.event(
                "orientation",
                "not_needed",
                source_kind="planner",
            )
        result = apply_hermeneutics(
            query=text,
            candidates=[],
            plan=plan,
            limit=limit,
            flags=flags,
            retrieval_method="not_invoked",
        )
        if not flags.retrieval_quality:
            return _finalize(result)
        event = build_retrieval_event(
            session_id=session_id,
            turn_id=turn_id,
            invoked=False,
            brief=None,
            plan=plan.to_dict(),
            first_query=text,
            first_records=[],
            refinement_reason=None,
            refined_query=None,
            second_records=[],
            result=result,
            failure_category="memory_not_invoked",
        )
        record_retrieval_event(event)
        result["retrieval_telemetry"] = event
        return _finalize(result)

    brief = (
        load_retrieval_brief(session_id)
        if flags.retrieval_quality
        else None
    )
    first_query = plan.query or text
    refinement_reason: str | None = None
    refined_query: str | None = None
    second_candidates: list[dict] = []
    candidate_limit = max(30, limit * 6)
    source_health.append(
        {
            "source": "atomic_memory",
            "requested_path": "recall_hybrid",
            "actual_path": "recall_hybrid",
            "state": "healthy",
            "failure_class": None,
            "authority_limitation": None,
        }
    )
    try:
        recall_kwargs = {
            "user_id": cfg["user_id"],
            "cfg": cfg,
            "limit": candidate_limit,
            "tier": "zero_llm",
            "top_k": candidate_limit,
        }
        if trace is not None:
            recall_kwargs["trace"] = trace
        if plan.relation_envelopes_active:
            recall_kwargs["evidence_needs"] = plan.evidence_needs
        candidates, method = do_recall_hybrid(
            memory,
            first_query,
            **recall_kwargs,
        )
    except Exception as exc:
        if trace is not None:
            trace.event(
                "hybrid_retrieval",
                "error",
                failure_class=type(exc).__name__,
                fallback_path="recall_b",
            )
        logger.exception("[fidelis] Cogito Hermeneutics hybrid candidate retrieval failed")
        try:
            candidates, method = do_recall_b(
                memory,
                first_query,
                user_id=cfg["user_id"],
                cfg=cfg,
                limit=candidate_limit,
            )
            method = f"{method}|hybrid_error_fallback"
            source_health[0] = {
                "source": "atomic_memory",
                "requested_path": "recall_hybrid",
                "actual_path": "recall_b",
                "state": "degraded_fallback",
                "failure_class": type(exc).__name__,
                "authority_limitation": (
                    "Legacy fallback does not preserve canonical hybrid authority."
                ),
            }
            if trace is not None:
                trace.event(
                    "hybrid_retrieval",
                    "fallback_ok",
                    fallback_path="recall_b",
                    candidate_count=len(candidates),
                )
        except Exception as fallback_exc:
            source_health[0] = {
                "source": "atomic_memory",
                "requested_path": "recall_hybrid",
                "actual_path": (
                    "codex_history_fail_open"
                    if history_intent
                    else "none"
                ),
                "state": "degraded_fallback" if history_intent else "error",
                "failure_class": type(fallback_exc).__name__,
                "authority_limitation": (
                    "Both canonical hybrid and legacy atomic fallback failed."
                ),
            }
            if trace is not None:
                trace.event(
                    "hybrid_retrieval",
                    "fallback_error",
                    failure_class=type(fallback_exc).__name__,
                    fallback_path=(
                        "codex_history_fail_open"
                        if history_intent
                        else "raise"
                    ),
                )
            if not history_intent and not status_enabled:
                raise
            if not history_intent:
                result = apply_hermeneutics(
                    query=first_query,
                    candidates=[],
                    plan=plan,
                    limit=limit,
                    flags=flags,
                    retrieval_method="memory_retrievers_unavailable",
                )
                if not flags.retrieval_quality:
                    return _finalize(result)
                result["retrieval_brief"] = brief
                event = build_retrieval_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    invoked=True,
                    brief=brief,
                    plan=plan.to_dict(),
                    first_query=first_query,
                    first_records=[],
                    refinement_reason=None,
                    refined_query=None,
                    second_records=[],
                    result=result,
                    failure_category="no_supported_evidence",
                )
                record_retrieval_event(event)
                result["retrieval_telemetry"] = event
                return _finalize(result)
            logger.exception(
                "[fidelis] memory retrievers unavailable; continuing bounded Codex history"
            )
            candidates = []
            method = "memory_retrievers_unavailable|codex_history_fail_open"

    first_candidates = list(candidates)
    refinement_reason = choose_refinement_reason(
        query=first_query,
        candidates=first_candidates,
        plan=plan,
        enabled=flags.retrieval_quality,
    )
    if refinement_reason:
        refined_query = build_refined_query(first_query, plan, refinement_reason)
    if refined_query:
        try:
            second_candidates, second_method = do_recall_hybrid(
                memory,
                refined_query,
                **recall_kwargs,
            )
            candidates = merge_candidate_attempts(
                first_candidates,
                second_candidates,
            )
            method = f"{method}|refined_once:{second_method}"
        except Exception as exc:
            logger.warning(
                "[fidelis] bounded refinement failed: %s",
                type(exc).__name__,
            )
            method = f"{method}|refinement_error:{type(exc).__name__}"
            candidates = first_candidates

    if history_intent:
        history_error: Exception | None = None
        try:
            history_candidate_limit = min(50, max(20, limit * 4))
            history_results = search_codex_history(
                text,
                limit=history_candidate_limit,
            )
        except Exception as exc:
            history_error = exc
            logger.exception("[fidelis] bounded Codex history retrieval failed")
            history_results = []
        source_health.append(
            {
                "source": "codex_task_history",
                "requested_path": "codex_history",
                "actual_path": "codex_history",
                "state": (
                    "error"
                    if history_error is not None
                    else "healthy"
                    if history_results
                    else "empty"
                ),
                "failure_class": (
                    type(history_error).__name__
                    if history_error is not None
                    else None
                ),
                "authority_limitation": (
                    "Codex task history was unavailable."
                    if history_error is not None
                    else None
                ),
            }
        )
        if history_results:
            candidates.extend(result.to_memory() for result in history_results)
            method = f"{method}+codex_history:{len(history_results)}"
        if trace is not None:
            trace.event(
                "codex_history",
                "ok" if history_results else "empty",
                returned_count=len(history_results),
                source_kind="codex_task_history",
            )

    candidates = mark_ephemera(candidates, cfg)
    candidates = mark_superseded(candidates, cfg)
    final_query = refined_query or first_query
    result = apply_hermeneutics(
        query=final_query,
        candidates=candidates,
        plan=plan,
        limit=limit,
        flags=flags,
        retrieval_method=method,
        protect_codex_history=history_intent,
    )
    if flags.retrieval_quality:
        result["retrieval_brief"] = brief
        result["retrieval_attempts"] = {
            "maximum_attempts": 2,
            "attempt_count": 2 if refined_query else 1,
            "first_query": first_query,
            "first_returned_ids": [
                candidate_record_id(item, index)
                for index, item in enumerate(first_candidates)
            ],
            "refinement_reason": refinement_reason,
            "refined_query": refined_query,
            "second_returned_ids": [
                candidate_record_id(item, index)
                for index, item in enumerate(second_candidates)
            ],
            "final_disposition": (
                "use_supported_evidence" if result.get("records") else "abstain"
            ),
        }
        event = build_retrieval_event(
            session_id=session_id,
            turn_id=turn_id,
            invoked=True,
            brief=brief,
            plan=plan.to_dict(),
            first_query=first_query,
            first_records=first_candidates,
            refinement_reason=refinement_reason,
            refined_query=refined_query,
            second_records=second_candidates,
            result=result,
            failure_category=(
                "success" if result.get("records") else "no_supported_evidence"
            ),
        )
        record_retrieval_event(event)
        result["retrieval_telemetry"] = event
    if trace is not None:
        trace.event(
            "evidence_packet",
            "ok" if result.get("records") else "empty",
            candidate_count=result.get("candidate_count", 0),
            returned_count=result.get("shown_count", 0),
        )
    return _finalize(result)


def make_handler(memory: object, cfg: dict) -> type:
    user_id: str = cfg["user_id"]
    # FIDELIS_DECOMPOSE_TIMEOUT_SECS: max seconds for /recall sub-query pipeline.
    # Default 8s preserves existing behavior in normal cases; kicks in only on slow-call edges.
    _decompose_timeout: float = float(os.environ.get("FIDELIS_DECOMPOSE_TIMEOUT_SECS", 8))

    # /health must exercise the search path it vouches for: during the
    # 2026-07-02 store corruption it reported "ok" (count() worked) while every
    # /query and /recall raised — clients saw a healthy server and silently got
    # zero recall. Canary is cached so the 60s launchd probe doesn't embed
    # every minute.
    _canary = {"ts": 0.0, "ok": True, "err": ""}
    _CANARY_TTL_SECS = 300.0
    # Count/readiness freshness is intentionally much shorter than the search
    # canary cadence. Reusing the 300s search TTL here allowed a failed Chroma
    # store to inherit a stale successful /health status for five minutes.
    _HEALTH_CACHE_TTL_SECS = float(
        os.environ.get("FIDELIS_HEALTH_CACHE_TTL_SECS", "10")
    )
    _HEALTH_REFRESH_MAX_AGE_SECS = float(
        os.environ.get("FIDELIS_HEALTH_REFRESH_MAX_AGE_SECS", "30")
    )
    _health_lock = threading.Lock()
    _health_cache = {
        "ts": 0.0,
        "status": "degraded",
        "count": -1,
        "search_ok": False,
        "search_err": "initializing",
        "refreshing": False,
        "refresh_started_at": 0.0,
        "refresh_generation": 0,
    }

    def _search_canary() -> tuple[bool, str]:
        import time as _ct
        now = _ct.time()
        if now - _canary["ts"] < _CANARY_TTL_SECS:
            return _canary["ok"], _canary["err"]
        try:
            qv = _embed_bounded(memory.embedding_model, "health canary", memory_action="search")  # type: ignore
            memory.vector_store.search(  # type: ignore
                query="health canary", vectors=[qv], top_k=1,
                filters={"user_id": user_id},
            )
            _canary.update(ts=now, ok=True, err="")
        except Exception:
            logger.exception("health: canary search failed")
            _canary.update(ts=now, ok=False, err="search_failing")
        return _canary["ok"], _canary["err"]

    def _probe_runtime_health() -> tuple[str, int, bool, str]:
        try:
            count = memory.vector_store.collection.count()  # type: ignore
        except Exception as exc:
            count = -1
            logger.warning("health: chroma count failed: %s", exc)
        search_ok, search_err = _search_canary()
        status = "ok" if (count >= 0 and search_ok) else "degraded"
        return status, count, search_ok, search_err

    def _refresh_health_cache(generation: int) -> None:
        try:
            status, count, search_ok, search_err = _probe_runtime_health()
            import time as _ht
            with _health_lock:
                if generation != _health_cache["refresh_generation"]:
                    return
                _health_cache.update(
                    ts=_ht.time(),
                    status=status,
                    count=count,
                    search_ok=search_ok,
                    search_err=search_err,
                )
        finally:
            with _health_lock:
                if generation == _health_cache["refresh_generation"]:
                    _health_cache.update(
                        refreshing=False,
                        refresh_started_at=0.0,
                    )

    def _start_health_refresh(*, force: bool = False) -> bool:
        """Start one off-path readiness refresh; return whether it was started."""
        import time as _ht
        now = _ht.time()
        with _health_lock:
            if _health_cache["refreshing"]:
                refresh_age = now - float(_health_cache["refresh_started_at"])
                if not force and refresh_age < _HEALTH_REFRESH_MAX_AGE_SECS:
                    return False
                logger.warning(
                    "health: superseding readiness refresh generation %s after %.1fs",
                    _health_cache["refresh_generation"],
                    refresh_age,
                )
            generation = int(_health_cache["refresh_generation"]) + 1
            _health_cache.update(
                status="degraded",
                refreshing=True,
                refresh_started_at=now,
                refresh_generation=generation,
            )
        try:
            threading.Thread(
                target=_refresh_health_cache,
                args=(generation,),
                daemon=True,
                name="fidelis-health-refresh",
            ).start()
        except Exception:
            with _health_lock:
                if generation == _health_cache["refresh_generation"]:
                    _health_cache.update(
                        status="degraded",
                        search_ok=False,
                        search_err="refresh_start_failed",
                        refreshing=False,
                        refresh_started_at=0.0,
                    )
            logger.exception("health: could not start readiness refresh")
            return False
        return True

    def _invalidate_health_cache() -> None:
        """Make post-mutation count explicitly unknown until an off-path refresh."""
        with _health_lock:
            _health_cache.update(
                ts=0.0,
                status="degraded",
                count=-1,
            )
        # A pre-write refresh may already have captured the old count. Force a
        # new generation so its late result cannot make the mutation look stale.
        _start_health_refresh(force=True)

    def _prime_health() -> None:
        """Begin deep readiness off-path without delaying the real handler."""
        _start_health_refresh()

    def _runtime_health() -> tuple[str, int, bool, str, float | None, bool]:
        """Return cached readiness immediately and refresh stale data off-path."""
        import time as _ht
        now = _ht.time()
        with _health_lock:
            ts = float(_health_cache["ts"])
            stale = now - ts >= _HEALTH_CACHE_TTL_SECS
        if stale:
            _start_health_refresh()
        with _health_lock:
            current_ts = float(_health_cache["ts"])
            refreshing = bool(_health_cache["refreshing"])
            cached_status = str(_health_cache["status"])
            result = (
                "degraded" if refreshing else cached_status,
                int(_health_cache["count"]),
                bool(_health_cache["search_ok"]),
                str(_health_cache["search_err"]),
                round(now - current_ts, 3) if current_ts > 0 else None,
                refreshing,
            )
        return result

    class Handler(BaseHTTPRequestHandler):
        prime_health = staticmethod(_prime_health)
        invalidate_health_cache = staticmethod(_invalidate_health_cache)

        def log_message(self, fmt, *args):  # suppress default logging
            pass

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected during write")
                return

        _MAX_BODY = 1_048_576  # 1 MB

        def _read_body(self) -> dict | None:
            n = int(self.headers.get("Content-Length", 0))
            if n > self._MAX_BODY:
                return None  # signal rejection
            raw = self.rfile.read(n)
            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if isinstance(memory, MemoryHolder) and not memory.ready:
                if self.path == "/health":
                    self._json({
                        "status": "degraded" if memory.last_error else "ok",
                        "count": -1, "queued": queued_count(),
                        "version": __version__, "store_loaded": False,
                        "calibrated": bool(cfg.get("vocab_map")),
                        "snapshot": _snapshot_path(cfg).exists(),
                    })
                    return
                if self.path not in ("/live", "/snapshot", "/capabilities"):
                    try:
                        memory.get()
                    except Exception:
                        self._json({"error": "memory store unavailable"}, 503)
                        return
            try:
                if self.path == "/live":
                    # Dependency-free process liveness for supervisors. Keep
                    # this separate from /health: readiness deliberately
                    # exercises Chroma count/search and may block briefly
                    # behind a concurrent write. Restart automation must not
                    # turn that transient readiness delay into a kill loop.
                    self._json({
                        "status": "alive",
                        "readiness": "ready",
                        "version": __version__,
                    })
                elif self.path == "/health":
                    # Read count directly from chroma. Previous code used
                    # get_all with top_k=10000 which (a) silently capped the
                    # reported count at 10000 and (b) hammered Ollama on
                    # every probe. mem0 2.0's ChromaDB wrapper exposes the
                    # underlying chromadb.Collection as `.collection`; its
                    # `.count()` is O(1).
                    status, count, search_ok, search_err, probe_age, refreshing = _runtime_health()
                    snap_path = _snapshot_path(cfg)
                    payload = {
                        "status": status,
                        "count": count,
                        "queued": queued_count(),
                        "index_lag": _chroma_index_lag(cfg),
                        "probe_age_s": probe_age,
                        "probe_refreshing": refreshing,
                        "version": __version__,
                        "cogito_hermeneutics": {
                            "name": HERMENEUTICS_ENGINE,
                            "version": HERMENEUTICS_VERSION,
                            "active": True,
                            "feature_flags": QualityFlags.from_environment().to_dict(),
                        },
                        "calibrated": bool(cfg.get("vocab_map")),
                        "snapshot": snap_path.exists(),
                    }
                    if isinstance(memory, MemoryHolder):
                        payload["store_loaded"] = memory.ready
                    payload["capabilities"] = _runtime_capabilities(
                        str(payload["status"])
                    )
                    if not search_ok:
                        payload["search_error"] = search_err
                    self._json(payload)
                elif self.path == "/capabilities":
                    readiness, _, _, _, _, _ = _runtime_health()
                    self._json(_runtime_capabilities(readiness))
                elif self.path == "/snapshot":
                    text = _read_snapshot(cfg)
                    if text is None:
                        self._json({"error": "no snapshot — run `cogito snapshot` first"}, 404)
                    else:
                        self._json({"snapshot": text, "path": str(_snapshot_path(cfg))})
                elif self.path == "/replay":
                    # Manually drain the queue. Useful to call after fixing a
                    # transient Ollama outage. The server also auto-drains on
                    # startup and periodically via the background thread.
                    result = replay_queue(memory, user_id=user_id)  # type: ignore
                    self._json(result)
                elif self.path == "/stats":
                    # PACKET F2c: cheap, sidecar-backed store-wide counts.
                    # Deliberately a separate endpoint from /health -- /health's
                    # own readiness probe already does its own Chroma
                    # count/search and must never be slowed down or risk a
                    # regression from stats logic sharing its code path.
                    stats = store_stats(memory, temporal_index())  # type: ignore
                    stats["queued"] = queued_count()
                    stats["dead_letter"] = dead_count()
                    self._json(stats)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                # Full traceback to the log — a bare class name ("InternalError")
                # gave zero diagnostic signal during the 2026-07-02 store-corruption
                # incident; the client still gets only the class name.
                logger.exception("[fidelis] %s %s failed: %s", self.command, self.path, e)
                try:
                    self._json({"error": f"internal error: {type(e).__name__}"}, 500)
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("client disconnected before error response could be sent")

        def do_POST(self):
            if isinstance(memory, MemoryHolder):
                try:
                    memory.get()
                except Exception as exc:
                    logger.warning("memory store unavailable: %s", type(exc).__name__)
                    self._json({"error": "memory store unavailable"}, 503)
                    return
            try:
                data = self._read_body()
                if data is None:
                    self._json({"error": "request body too large"}, 413)
                    return
                if not data and self.path not in ("/add", "/store"):
                    self._json({"error": "invalid json"}, 400)
                    return

                if self.path == "/query":
                    text = data.get("text", "")
                    limit = int(data.get("limit", 5))
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": []})
                        return
                    try:
                        as_of, historical = _temporal_request(data)
                    except ValueError as e:
                        self._json({"error": f"invalid as_of: {e}"}, 400)
                        return
                    # Bypass mem0.Memory.search wrapper: it routes through
                    # score_and_rank which (in mem0 2.0.x) returns broken
                    # score=1.0 for every result regardless of similarity.
                    # Verified empirically — same query, when we go directly to
                    # vector_store.search, returns proper distances (the actual
                    # text-match record scores 0.5878 vs unrelated at 1.07+).
                    qv = _embed_bounded(memory.embedding_model, text, memory_action="search")  # type: ignore
                    # Over-fetch 4x so ephemera filtering (50.5% of the store,
                    # measured 2026-07-18) doesn't starve the response.
                    raw = memory.vector_store.search(  # type: ignore
                        query=text, vectors=[qv], top_k=max(overfetch(limit, as_of) * 4, 20),
                        filters={"user_id": user_id},
                    )
                    memories = [
                        {
                            **_hit_id(r), **carry_payload(r.payload),
                            "text": (r.payload or {}).get("data", ""),
                            # mem0 2.x already returns a similarity, 1/(1+distance), in (0, 1];
                            # larger = closer. Clamp only; do not re-derive it from a distance.
                            "score": round(max(0.0, min(1.0, r.score or 0.0)), 3),
                        }
                        for r in raw
                        if (r.payload or {}).get("data")
                    ]
                    # Temporal view runs on verbatim text, before the legacy
                    # supersession prefix rewrites it.
                    memories = temporal_view(
                        filter_ephemera(memories, cfg), memory=memory,
                        index=temporal_index(), as_of=as_of,
                        historical=historical, limit=limit,
                    )
                    self._json({"memories": mark_superseded(memories, cfg)})

                elif self.path == "/recall":
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    since = data.get("since")
                    try:
                        as_of, historical = _temporal_request(data)
                    except ValueError as e:
                        self._json({"error": f"invalid as_of: {e}"}, 400)
                        return
                    degraded = False
                    _fut = _RECALL_POOL.submit(
                        do_recall, memory, text,
                        user_id=user_id, cfg=cfg, limit=overfetch(limit, as_of), since=since,
                    )
                    try:
                        memories, method = _fut.result(timeout=_decompose_timeout)
                    except concurrent.futures.TimeoutError:
                        # Decompose pipeline timed out — fall back to vector-only single query.
                        # Bypass mem0.Memory.search wrapper (broken score_and_rank in 2.0.0
                        # returns score=1.0 for all results); call vector_store.search directly.
                        logger.warning(
                            "[fidelis] /recall decompose timeout (>%ss) for query '%s'; returning vector-only fallback",
                            _decompose_timeout, text[:50],
                        )
                        qv = memory.embedding_model.embed(text, memory_action="search")
                        raw = memory.vector_store.search(
                            query=text, vectors=[qv], top_k=overfetch(limit, as_of),
                            filters={"user_id": user_id},
                        )
                        memories = [
                            {
                                **_hit_id(r), **carry_payload(r.payload),
                                "text": (r.payload or {}).get("data", ""),
                                "score": round(max(0.0, min(1.0, r.score or 0.0)), 3),
                            }
                            for r in raw
                            if (r.payload or {}).get("data")
                        ]
                        method = "vector-only-fallback"
                        degraded = True
                    print(f"[cogito] /recall '{text[:50]}' → {len(memories)} results ({method})", flush=True)
                    memories = temporal_view(
                        filter_ephemera(memories, cfg), memory=memory,
                        index=temporal_index(), as_of=as_of,
                        historical=historical, limit=limit,
                    )
                    resp: dict = {"memories": mark_superseded(memories, cfg), "method": method}
                    if degraded:
                        resp["degraded"] = True
                    self._json(resp)

                elif self.path == "/recall_b":
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    memories, method = do_recall_b(
                        memory, text, user_id=user_id, cfg=cfg,
                        limit=limit,
                    )
                    memories = filter_ephemera(memories, cfg)
                    print(f"[cogito] /recall_b '{text[:50]}' → {len(memories)} results ({method})", flush=True)
                    self._json({"memories": mark_superseded(memories, cfg), "method": method})

                elif self.path in ("/orient", "/cogito-hermeneutics"):
                    text = str(data.get("text") or "")
                    if not text or len(text.strip()) < 3:
                        self._json({"error": "query must contain at least 3 characters"}, 400)
                        return
                    try:
                        as_of, historical = _temporal_request(data)
                    except ValueError as e:
                        self._json({"error": f"invalid as_of: {e}"}, 400)
                        return
                    result = _run_cogito_hermeneutics(memory, cfg, data)
                    if isinstance(result.get("memories"), list):
                        # Timeline plans own their chronological order; every
                        # other mode gets current-first within the relevant set.
                        timeline = (result.get("plan") or {}).get("mode") == "timeline"
                        result["memories"] = temporal_view(
                            result["memories"], memory=memory,
                            index=temporal_index(), as_of=as_of,
                            historical=historical or timeline,
                        )
                        result["shown_count"] = len(result["memories"])
                    print(
                        f"[fidelis] /orient '{text[:50]}' mode={result['plan']['mode']} "
                        f"→ {result['shown_count']} results ({result['method']})",
                        flush=True,
                    )
                    self._json(result)

                elif self.path == "/inquire":
                    query_value = (
                        data["query"]
                        if "query" in data
                        else data.get("text")
                    )
                    if (
                        not isinstance(query_value, str)
                        or not 3 <= len(query_value.strip()) <= 4096
                    ):
                        self._json(
                            {
                                "error": (
                                    "query must be a string containing "
                                    "3 to 4096 characters"
                                )
                            },
                            400,
                        )
                        return
                    query = query_value.strip()
                    policy = data.get("policy", "auto")
                    overrides = data.get("overrides")
                    include_trace = data.get("include_trace", False)
                    inquiry_error = _validate_inquiry_transport(
                        policy,
                        overrides,
                        include_trace,
                    )
                    if inquiry_error:
                        self._json({"error": inquiry_error}, 400)
                        return

                    def inquiry_retriever(
                        pass_query: str,
                        candidate_limit: int,
                        target_roles: tuple[str, ...],
                        pass_number: int,
                    ) -> tuple[list[dict], str]:
                        del pass_number
                        records, retrieval_method = do_recall_hybrid(
                            memory,
                            pass_query,
                            user_id=user_id,
                            cfg=cfg,
                            limit=candidate_limit,
                            tier="zero_llm",
                            top_k=candidate_limit,
                            evidence_needs=target_roles,
                        )
                        records = mark_ephemera(records, cfg)
                        records = mark_superseded(records, cfg)
                        return records, retrieval_method

                    result = run_inquiry(
                        memory,
                        cfg,
                        query,
                        policy=policy,
                        include_trace=include_trace,
                        overrides=overrides,
                        retriever=inquiry_retriever,
                    )
                    print(
                        f"[fidelis] /inquire '{query[:50]}' "
                        f"operation={result['ask']['operation']} "
                        f"passes={len(result.get('trace', {}).get('passes', []))} "
                        f"→ {len(result['records'])} results "
                        f"({result['stop_reason']})",
                        flush=True,
                    )
                    self._json(result)

                elif self.path == "/recall_hybrid":
                    # BM25 + dense + RRF + tiered LLM escalation.
                    # Default tier: local zero_llm.
                    # Opt-in filter/flagship remain experimental.
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    tier = data.get("tier", "zero_llm")
                    top_k = int(data.get("top_k", 5))
                    if tier not in ("zero_llm", "filter", "flagship"):
                        self._json({"error": f"invalid tier: {tier}"}, 400)
                        return
                    try:
                        as_of, historical = _temporal_request(data)
                    except ValueError as e:
                        self._json({"error": f"invalid as_of: {e}"}, 400)
                        return
                    # Over-fetch 3x for the ephemera filter, trim back to top_k.
                    memories, method = do_recall_hybrid(
                        memory, text, user_id=user_id, cfg=cfg,
                        limit=limit, tier=tier, top_k=max(overfetch(top_k, as_of) * 3, 15),
                    )
                    memories = temporal_view(
                        filter_ephemera(memories, cfg), memory=memory,
                        index=temporal_index(), as_of=as_of,
                        historical=historical, limit=top_k,
                    )
                    print(f"[cogito] /recall_hybrid '{text[:50]}' tier={tier} → {len(memories)} results ({method})", flush=True)
                    self._json({"memories": mark_superseded(memories, cfg), "method": method})

                elif self.path == "/store":
                    # Verbatim write — agent decides content, no extraction LLM.
                    # Uses safe_add: queues locally if dependency (Ollama) is down.
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"error": "no text"}, 400)
                        return
                    transport_error = _validate_store_transport(data, text)
                    if transport_error:
                        self._json({"error": transport_error}, 400)
                        return
                    declared_metadata = data.get("metadata")
                    if not isinstance(declared_metadata, dict):
                        declared_metadata = {}
                    declared_metadata = dict(declared_metadata)
                    # Temporal declarations are top-level request fields and
                    # are honored whether or not relation envelopes are on.
                    temporal_declared = {
                        key: data[key] for key in _TEMPORAL_STORE_KEYS
                        if data.get(key) is not None
                    }
                    declared_metadata.update(temporal_declared)
                    try:
                        if relation_envelopes_enabled():
                            declared_metadata.setdefault(
                                "collection", cfg.get("collection")
                            )
                            result = safe_add(  # type: ignore
                                memory,
                                text,
                                user_id,
                                kind="store",
                                record_id=(
                                    str(data["id"])
                                    if data.get("id")
                                    else None
                                ),
                                metadata=declared_metadata,
                            )
                        elif temporal_declared:
                            result = safe_add(  # type: ignore
                                memory, text, user_id, kind="store",
                                metadata=temporal_declared,
                            )
                        else:
                            # Preserve the exact legacy call when nothing is declared.
                            result = safe_add(memory, text, user_id, kind="store")  # type: ignore
                    except ValueError as e:
                        # Invalid temporal declaration: fail loud, write nothing.
                        self._json({"error": str(e)}, 400)
                        return
                    if result.get("status") == "rejected":
                        self._json({**result, "queued_total": queued_count()}, 422)
                        return
                    if result.get("status") != "queued":
                        _invalidate_health_cache()
                    response = {**result, "queued_total": queued_count()}
                    # The documented `id` argument is only honoured when relation
                    # envelopes are on (it becomes the record's stable id there);
                    # otherwise a caller-supplied id is silently replaced by a
                    # server UUID. Say so instead of letting that pass silently
                    # (docs/TIME-AWARE-SPEC.md F1c).
                    if data.get("id") and not relation_envelopes_enabled():
                        response["id_ignored"] = True
                    self._json(response)

                elif self.path == "/add":
                    # Uses safe_add: queues locally if Ollama is unreachable.
                    text = data.get("text", "")
                    if not text:
                        self._json({"error": "no text"}, 400)
                        return
                    result = safe_add(memory, text, user_id, kind="add")  # type: ignore
                    if result["status"] == "rejected":
                        self._json({**result, "queued_total": queued_count()}, 422)
                    elif result["status"] == "queued":
                        self._json({
                            "status": "queued",
                            "id": result["id"],
                            "reason": result["reason"],
                            "queued_total": queued_count(),
                        }, 202)  # 202 Accepted: write deferred
                    else:
                        _invalidate_health_cache()
                        extracted = result.get("extracted", [])
                        resp = {
                            "status": "stored",
                            "count": len(extracted),
                            "memories": extracted,
                        }
                        if result.get("degraded"):
                            # Surface safe_add's extraction→verbatim fallback so
                            # clients can tell the user extraction did NOT run;
                            # dropping it here made the CLI report degraded
                            # writes as successful extraction.
                            resp["degraded"] = result["degraded"]
                            resp["id"] = result.get("id")
                        self._json(resp)

                elif self.path == "/get":
                    # PACKET F2a: one record by id, with its supersession
                    # chain in both directions. Read-only, same-user, never
                    # scans the store beyond the single lookup plus each
                    # chain hop's own point-get.
                    record_id = data.get("id")
                    if not isinstance(record_id, str) or not record_id:
                        self._json({"error": "id is required"}, 400)
                        return
                    record = get_record(memory, temporal_index(), record_id, user_id)  # type: ignore
                    if record is None:
                        self._json(
                            {"error": f"no record found for id {record_id!r}"}, 404
                        )
                        return
                    self._json(record)

                elif self.path == "/recent":
                    # PACKET F2b: newest records by recorded_at, or
                    # corrections only. Sidecar-backed paging -- never a full
                    # store scan.
                    limit = data.get("limit", 10)
                    try:
                        limit = int(limit)
                    except (TypeError, ValueError):
                        self._json({"error": "limit must be an integer"}, 400)
                        return
                    if limit < 1:
                        self._json({"error": "limit must be >= 1"}, 400)
                        return
                    kind = data.get("kind", "all")
                    if kind not in ("all", "corrections"):
                        self._json({"error": f"invalid kind: {kind}"}, 400)
                        return
                    since = data.get("since")
                    if since is not None:
                        try:
                            since = format_instant(parse_as_of(since))
                        except ValueError as e:
                            self._json({"error": f"invalid since: {e}"}, 400)
                            return
                    result = recent_records(  # type: ignore
                        memory, temporal_index(), user_id=user_id, limit=limit,
                        since=since, kind=kind,
                    )
                    self._json(result)

                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                # Full traceback to the log — a bare class name ("InternalError")
                # gave zero diagnostic signal during the 2026-07-02 store-corruption
                # incident; the client still gets only the class name.
                logger.exception("[fidelis] %s %s failed: %s", self.command, self.path, e)
                try:
                    self._json({"error": f"internal error: {type(e).__name__}"}, 500)
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("client disconnected before error response could be sent")

    return Handler


def main():
    # Configure root logging once. Without this, `logger.warning(...)` calls
    # are silent in launchd / systemd / MCP contexts because nothing else in
    # the stack calls basicConfig. FIDELIS_LOG_LEVEL overrides per-deployment.
    logging.basicConfig(
        level=os.environ.get("FIDELIS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=False,  # respect any pre-existing config (tests, parent app)
    )

    parser = argparse.ArgumentParser(description="fidelis memory server")
    parser.add_argument("--config", help="Path to .cogito.json")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    cfg = load(args.config)
    if args.port:
        cfg["port"] = args.port

    src = cfg.get("_config_file", "defaults + env")
    print(f"[cogito] Starting server v{__version__} (config: {src})", flush=True)

    # Lazy memory construction: the historical eager `_boot(cfg)` call here
    # required a reachable Ollama before the HTTP server even bound, so a
    # registry inspector (Glama et al.) that starts this entry point with no
    # Ollama running never got past this line before crashing — GET /health
    # was unreachable. MemoryHolder defers the real construction to first
    # use by a request handler and is thread-safe / retry-on-failure (see
    # its docstring above). This preserves identical behavior when Ollama IS
    # reachable: the first request just pays the one-time construction cost
    # instead of it happening before bind.
    memory = MemoryHolder(cfg)
    stop_event = threading.Event()

    # Background replay thread — sweeps the queue every 60s. Items that failed
    # at write time (Ollama momentarily unreachable, embed timeout) get retried
    # without requiring a server restart. The first sweep runs ~5s after server
    # start so HTTP serving is up immediately rather than blocking on a long
    # drain. Items stay in the queue across server restarts.
    def _replay_loop():
        stop_event.wait(5)  # let serve_forever() bind first
        # Exponential backoff: base 60s, doubles on no-progress sweeps,
        # capped at 30 min. Resets to base on any successful replay.
        # Prevents the forever-warm-LLM heat bug when the queue is
        # non-empty but every item keeps failing (e.g. Ollama model
        # missing or unreachable). Combined with MAX_ATTEMPTS dead-letter
        # in degrade.py, the queue cannot stay hot indefinitely.
        BASE = 60
        MAX = 1800
        sleep_s = BASE
        while not stop_event.is_set():
            try:
                pending = queued_count()
                if pending > 0:
                    # Sweeping requires the real Memory object; if it can't
                    # be constructed yet (Ollama still unreachable), this
                    # raises and the outer except below backs off — the
                    # queue simply waits for a later sweep, same as before.
                    active_memory = memory.get()
                    result = replay_queue(active_memory, user_id=cfg["user_id"])
                    print(
                        f"[fidelis] queue sweep: replayed={result.get('replayed', 0)} "
                        f"(verbatim_fallback={result.get('replayed_verbatim', 0)}) "
                        f"failed={result.get('failed', 0)} "
                        f"dead_lettered={result.get('dead_lettered', 0)} "
                        f"remaining={result.get('remaining', 0)} "
                        f"next_sweep_s={sleep_s}",
                        flush=True,
                    )
                    if result.get("replayed", 0) > 0:
                        sleep_s = BASE
                    else:
                        sleep_s = min(sleep_s * 2, MAX)
                else:
                    sleep_s = BASE
            except Exception as e:
                logger.debug("background replay tick failed: %s", e)
                sleep_s = min(sleep_s * 2, MAX)
            stop_event.wait(sleep_s)
    replay_thread = threading.Thread(target=_replay_loop, daemon=True, name="fidelis-replay")
    replay_thread.start()

    port = cfg["port"]
    handler = make_handler(memory, cfg)
    httpd = _BoundedThreadingHTTPServer((args.host, port), handler)

    # Graceful-shutdown signal handlers. SIGTERM is what launchd/systemd send
    # on `launchctl bootout` or `systemctl stop`; SIGINT is Ctrl-C. We must
    # call httpd.shutdown() (which returns once the serve loop has cleanly
    # finished any in-flight requests) and then close the chromadb client so
    # its SQLite WAL is checkpointed to disk. Without this, a hard OS reboot
    # mid-write can leave the store in an inconsistent state — exactly the
    # data-integrity hazard a memory product cannot afford.
    import signal
    _shutdown_done = threading.Event()

    def _shutdown(signum, frame):
        if _shutdown_done.is_set():
            return
        _shutdown_done.set()
        stop_event.set()
        logger.warning("received signal %s — shutting down gracefully", signum)
        # httpd.shutdown() blocks until the serve loop returns; must not be
        # called from the same thread as serve_forever (deadlocks). Spawn it.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[fidelis] Listening on {args.host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        # Always-runs cleanup, even on unhandled exceptions. Closes the
        # underlying chromadb client (and its SQLite handle) so any pending
        # WAL frames are checkpointed before process exit.
        stop_event.set()
        replay_thread.join(timeout=5)
        try:
            httpd.server_close()
        except Exception as e:  # noqa: silent — best-effort socket close
            logger.debug("httpd.server_close() raised: %s", e)
        try:
            # Only touch vector_store if Memory was actually constructed —
            # a server that shuts down before ever handling a memory-backed
            # request (e.g. the registry inspector's health-only probe) has
            # nothing to checkpoint.
            if memory.ready:
                active_memory = memory.get()
                client = getattr(active_memory.vector_store, "client", None)
                if client is not None and hasattr(client, "_admin_client"):
                    # chromadb PersistentClient — let GC trigger __del__ checkpoint
                    pass
        except Exception as e:  # noqa: silent — chromadb internals may shift across versions; fall back to GC
            logger.debug("chromadb close hook raised: %s", e)
        print("[fidelis] Stopped cleanly.", flush=True)


if __name__ == "__main__":
    main()
