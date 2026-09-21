"""
fidelis sessions — searchable, purge-able session memory (MVP-A).

Wraps existing capability (query_sessions, ingest) and adds the purge path.
Honesty rules (non-negotiable, carried from the v1 spec):
  • Never print retrieval-benchmark numbers in product output (label only:
    "searchable session history — best on multi-turn queries").
  • Deletion claims stop at the INDEX. Claude/Codex source files and backups are
    NEVER touched.
  • No redaction-as-privacy-control. The honest levers are: don't-index
    (deny-list / --dry-run) and delete-after (purge).

Pure helpers (_iso_before, _collect_purge_targets, _filter_before) are separated
so they can be tested without a live ChromaDB / Ollama — "experiential contact"
with the LOGIC, not the services.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

COGITO_SESSIONS_DIR = Path.home() / ".cogito" / "session_ingest"

BACKUP_BOUNDARY = (
    "Source files in ~/.claude/projects, ~/.codex/sessions, "
    "~/.codex/archived_sessions, Codex desktop prompt history, and backups in "
    "~/Backups/claude-sessions are NOT touched by design."
)
PRIVACY_NOTICE = (
    "Note: sessions contain full conversation history, stored locally in ~/.cogito "
    "(readable only by your user). Run `fidelis sessions purge` to remove. " + BACKUP_BOUNDARY
)


# ── Pure helpers (testable without services) ──────────────────────────────────

def _valid_iso_date(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _iso_before(end_ts: str, before: str) -> bool:
    """True if end_ts is strictly before the `before` date (YYYY-MM-DD).
    ISO-8601 lexicographic comparison is valid for the stored `...Z` format,
    so we avoid ChromaDB's broken string `$lt` where-filter entirely."""
    if not end_ts or not _valid_iso_date(before):
        return False
    return end_ts[:10] < before[:10]


def _collect_purge_targets(
    metadatas: list[dict], ids: list[str], mode: str, before: str | None
) -> tuple[list[str], list[str]]:
    """Return (target_ids, target_hashes) for purge. Pure: takes already-fetched
    ChromaDB rows, never calls the DB. mode is 'all' or 'before'.
    Collects ingest_hash too — required for ledger cleanup (spec step 3)."""
    target_ids: list[str] = []
    target_hashes: list[str] = []
    for cid, meta in zip(ids, metadatas):
        if mode == "all":
            keep = True
        elif mode == "before":
            keep = _iso_before(meta.get("end_ts", ""), before or "")
        else:
            keep = False
        if keep:
            target_ids.append(cid)
            h = meta.get("ingest_hash")
            if h:
                target_hashes.append(h)
    return target_ids, target_hashes


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rewrite_purge_ledger(
    ledger_path: Path,
    target_hashes: list[str],
    target_ids: list[str],
) -> dict[str, str]:
    """Atomically remove purge targets and return the original ledger."""
    original = json.loads(ledger_path.read_text())
    if not isinstance(original, dict):
        raise ValueError("session ingest ledger must be a JSON object")
    updated = {
        ingest_hash: chroma_id
        for ingest_hash, chroma_id in original.items()
        if ingest_hash not in target_hashes and chroma_id not in target_ids
    }
    tmp_path = ledger_path.with_name(f".{ledger_path.name}.purge-tmp")
    tmp_path.write_text(json.dumps(updated, indent=2))
    tmp_path.replace(ledger_path)
    return original


def _restore_purge_ledger(ledger_path: Path, original: dict[str, str]) -> None:
    tmp_path = ledger_path.with_name(f".{ledger_path.name}.restore-tmp")
    tmp_path.write_text(json.dumps(original, indent=2))
    tmp_path.replace(ledger_path)


# ── purge (the locked contract — §4 of the v1 spec) ───────────────────────────

def cmd_purge(args) -> int:
    if bool(args.all) == bool(args.before):
        print("Error: pass exactly one of --all or --before YYYY-MM-DD")
        return 2
    if args.before and not _valid_iso_date(args.before):
        print("Error: --before must be a valid YYYY-MM-DD date")
        return 2

    from fidelis.ingest_claude_sessions import _get_collection  # lazy: avoids chromadb import at CLI start

    col = _get_collection()
    entries = col.get(where={"mem_type": "session"}, include=["metadatas"])
    ids = entries.get("ids", []) or []
    metas = entries.get("metadatas", []) or []

    mode = "all" if args.all else "before"
    target_ids, target_hashes = _collect_purge_targets(metas, ids, mode, args.before)

    if not target_ids:
        print("No matching sessions to purge.")
        return 0

    # oldest/newest of the MATCHED set (not the whole corpus) for an honest summary
    target_set = set(target_ids)
    ends = sorted(
        m.get("end_ts", "")
        for cid, m in zip(ids, metas)
        if cid in target_set and m.get("end_ts")
    )
    span = f"{ends[0][:10]}..{ends[-1][:10]}" if ends else "unknown"

    if args.dry_run:
        print(f"DRY-RUN: would remove {len(target_ids)} sessions (span {span}). Nothing written.")
        print(BACKUP_BOUNDARY)
        return 0

    if not args.yes:
        print(f"About to remove {len(target_ids)} sessions from the search index (span {span}).")
        print(BACKUP_BOUNDARY)
        resp = input(f"Delete {len(target_ids)} sessions? [y/N] ").strip().lower()
        if resp != "y":  # default N — accidental Enter must not delete
            print("Aborted.")
            return 1

    # ledger cleanup: ingested.json maps {ingest_hash: chroma_id}
    ledger_path = COGITO_SESSIONS_DIR / "ingested.json"
    original_ledger: dict[str, str] | None = None
    if ledger_path.exists():
        try:
            # Remove the dedup entries first. If the following Chroma delete
            # fails, leaving the ledger entry absent is recoverable (the next
            # ingest upserts the still-present stable row). The reverse order
            # can lose searchability indefinitely after a ledger write fault.
            original_ledger = _rewrite_purge_ledger(
                ledger_path,
                target_hashes,
                target_ids,
            )
        except Exception as exc:
            print(f"Error: purge ledger update failed; no sessions removed ({type(exc).__name__}).")
            return 1

    try:
        col.delete(ids=target_ids)
    except Exception as exc:
        if original_ledger is not None:
            try:
                _restore_purge_ledger(ledger_path, original_ledger)
            except Exception as restore_exc:
                print(
                    "Warning: index delete failed and ledger restore also failed "
                    f"({type(restore_exc).__name__}); re-ingest is safe because "
                    "stable IDs are upserted."
                )
        print(f"Error: session index delete failed ({type(exc).__name__}).")
        return 1

    # append purge log
    COGITO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COGITO_SESSIONS_DIR / "purge_log.jsonl", "a") as f:
        f.write(json.dumps({
            "ts": _now_iso(), "mode": mode, "count": len(target_ids),
            "session_ids": target_ids,
        }) + "\n")

    print(f"Removed {len(target_ids)} sessions from the search index. {BACKUP_BOUNDARY}")
    return 0


# ── ingest / search / list / stats (thin wrappers over existing code) ─────────

def _is_first_run() -> bool:
    """First-run = no dedup ledger yet (nothing has ever been ingested).
    Reads the on-disk ledger, so it works without a live ChromaDB/Ollama."""
    from fidelis.ingest_claude_sessions import _load_ledger
    try:
        return not _load_ledger()
    except Exception:  # noqa: best-effort; a missing/corrupt ledger means first run
        return True


def cmd_ingest(args) -> int:
    from fidelis.ingest_claude_sessions import ingest as ingest_claude
    from fidelis.ingest_codex_sessions import ingest as ingest_codex
    from datetime import datetime as _dt, timezone as _tz
    since = None
    default_window = False
    source = getattr(args, "source", "claude")
    include_prompt_history = getattr(args, "include_prompt_history", False)
    thread_id = getattr(args, "thread_id", None)
    if thread_id and source != "codex":
        print("--thread-id requires --source codex")
        return 2
    if include_prompt_history and source == "claude":
        print("--include-prompt-history requires --source codex or --source all")
        return 2
    if include_prompt_history and not args.all:
        print(
            "--include-prompt-history requires --all because Codex desktop "
            "does not preserve per-prompt timestamps"
        )
        return 2
    if args.since and not _valid_iso_date(args.since):
        print("Error: --since must be a valid YYYY-MM-DD date")
        return 2
    if not args.all:
        # default: last 7 days (spec) unless --since given
        if args.since:
            since = _dt.fromisoformat(args.since).replace(tzinfo=_tz.utc)
        else:
            from datetime import timedelta
            since = _dt.now(_tz.utc) - timedelta(days=7)
            default_window = True

    # B3: the pitch is "keep & search your sessions", but the default sweep is only
    # 7 days. On a first run that silently leaves all older history un-indexed. Nudge
    # (don't silently change the default): tell the user older sessions exist and how
    # to get them, without over-engineering a full-history scan into the default path.
    if default_window and _is_first_run():
        full_history_command = "fidelis sessions ingest --all"
        if source != "claude":
            full_history_command = f"fidelis sessions ingest --source {source} --all"
        print(
            "First run: indexing only the last 7 days. Your older selected-source "
            f"sessions are NOT indexed yet — run `{full_history_command}` "
            "to index your full history."
        )

    runs = []
    if source in {"claude", "all"}:
        runs.append(ingest_claude(since=since, dry_run=args.dry_run, verbose=args.verbose))
    if source in {"codex", "all"}:
        runs.append(ingest_codex(
            since=since,
            dry_run=args.dry_run,
            verbose=args.verbose,
            include_prompt_history=include_prompt_history,
            thread_id=thread_id,
        ))
    keys = {"scanned", "stored", "updated", "skipped_dedup", "skipped_empty", "errors"}
    stats = {key: sum(int(run.get(key, 0)) for run in runs) for key in keys}
    print(f"scanned={stats['scanned']} stored={stats['stored']} "
          f"updated={stats['updated']} "
          f"skipped_dedup={stats['skipped_dedup']} skipped_empty={stats['skipped_empty']} "
          f"errors={stats['errors']}")
    if not args.dry_run:
        print(PRIVACY_NOTICE)
    return 0


def cmd_search(args) -> int:
    """Search local session history.

    Routes through the shared recent-work path so a phrase like "today" or an
    explicit session/task id resolves the same way here as it does for an MCP
    client, and so local Codex task evidence is not invisible when a matching
    index record is missing.
    """
    from fidelis.recall_recent import recall_recent_work
    from fidelis.recall_sessions import query_sessions

    if args.raw:
        # Unchanged contract: a top-level array of ingested session records,
        # ranked by similarity alone, exactly as before this feature existed.
        # `json.load(...)[0]["session_id"]` keeps working. Cross-client results
        # are a different shape and live behind --raw-cross-client.
        results = query_sessions(args.query, top_k=args.limit)
        print(json.dumps([item.to_dict() for item in results], indent=2))
        return 0

    result = recall_recent_work(args.query, limit=args.limit)
    if getattr(args, "raw_cross_client", False):
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if not result.candidates:
        print(
            "No matching sessions. (searchable session history — best on "
            "multi-turn queries)"
        )
        for note in result.notes:
            print(f"  note: {note}")
        return 0
    for index, candidate in enumerate(result.candidates, 1):
        detail = f"{candidate.turn_count} turns  " if candidate.turn_count else ""
        print(
            f"[{index}] score {candidate.score:.4f}  "
            f"{candidate.local_date or 'unknown-date'}  {detail}"
            f"{candidate.source_class}/{candidate.capture_class}  "
            f"id={candidate.identity}"
            + (f"  {candidate.project_path}" if candidate.project_path else "")
        )
        if candidate.evidence_pointer:
            print(f"    evidence: {candidate.evidence_pointer}")
        print(f"    {candidate.excerpt}")
        if candidate.match_reasons:
            print(f"    why: {', '.join(candidate.match_reasons)}")
    for note in result.notes:
        print(f"note: {note}")
    return 0


def cmd_list(args) -> int:
    if args.since and not _valid_iso_date(args.since):
        print("Error: --since must be a valid YYYY-MM-DD date")
        return 2
    from fidelis.ingest_claude_sessions import _get_collection
    col = _get_collection()
    entries = col.get(where={"mem_type": "session"}, include=["metadatas"])
    metas = entries.get("metadatas", []) or []
    rows = sorted(metas, key=lambda m: m.get("start_ts", ""))
    if args.since:
        rows = [m for m in rows if m.get("start_ts", "")[:10] >= args.since[:10]]
    rows = rows[: args.limit]
    for m in rows:
        print(f"{m.get('start_ts','')[:10]}  {m.get('session_id','')[:8]}  "
              f"{m.get('turn_count','?')} turns  "
              f"{m.get('session_source','claude_code')}/"
              f"{m.get('capture_completeness','NATIVE_MESSAGES')}  "
              f"{m.get('project_path','')}")
    print(f"\n{len(rows)} sessions.")
    return 0


def cmd_stats(args) -> int:
    if args.since and not _valid_iso_date(args.since):
        print("Error: --since must be a valid YYYY-MM-DD date")
        return 2
    from fidelis.ingest_claude_sessions import _get_collection
    col = _get_collection()
    entries = col.get(where={"mem_type": "session"}, include=["metadatas"])
    metas = entries.get("metadatas", []) or []
    if args.since:
        metas = [m for m in metas if m.get("start_ts", "")[:10] >= args.since]
    if not metas:
        if args.since:
            print(f"No indexed sessions match --since {args.since}.")
        else:
            print("No sessions indexed. Run `fidelis sessions ingest`.")
        return 0
    total_turns = sum(int(m.get("turn_count", 0) or 0) for m in metas)
    ends = sorted(m.get("end_ts", "") for m in metas if m.get("end_ts"))
    projects: dict[str, int] = {}
    sources: dict[str, int] = {}
    completeness: dict[str, int] = {}
    for m in metas:
        projects[m.get("project_path", "?")] = projects.get(m.get("project_path", "?"), 0) + 1
        source = m.get("session_source", "claude_code")
        level = m.get("capture_completeness", "NATIVE_MESSAGES")
        sources[source] = sources.get(source, 0) + 1
        completeness[level] = completeness.get(level, 0) + 1
    print(f"indexed sessions: {len(metas)}")
    print(f"date range: {ends[0][:10] if ends else '?'} .. {ends[-1][:10] if ends else '?'}")
    print(f"total turns: {total_turns}  avg/session: {total_turns // max(len(metas),1)}")
    print("by source: " + ", ".join(f"{key}={value}" for key, value in sorted(sources.items())))
    print(
        "by completeness: "
        + ", ".join(f"{key}={value}" for key, value in sorted(completeness.items()))
    )
    print("by project:")
    for p, n in sorted(projects.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:>5}  {p}")
    print("\nTool/skill call breakdown is not available in this version.")
    return 0
