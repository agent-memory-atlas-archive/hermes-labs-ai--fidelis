"""SQLite sidecar for time-aware memory (TIME-AWARE-SPEC v1).

The vector store is append-only and its payload fields (``content_sha256``,
``recorded_at``, ``supersedes_json``) are the source of truth. Two questions
are too expensive to answer by scanning payloads on every request:

* "has this exact text been stored before?"   (dedup, by content sha)
* "which records replace record X?"           (supersession, read time)

This module is a REBUILDABLE CACHE answering exactly those two. It lives next
to the store as ``<store-name>.temporal.sqlite`` and can be deleted at any
time; :meth:`TemporalIndex.rebuild` regenerates it from payloads.

Fail-open: :func:`open_index` returns ``None`` on any failure (unwritable
directory, corrupt file) and logs one warning per path. Callers treat ``None``
as "index unavailable" — recall still works, hits are just flagged
``temporal.index: "unavailable"``. A broken sidecar is never deleted or
overwritten here; repairing it is an explicit operator action.

Thread safety: the server uses a threaded HTTP handler. One connection is
shared (``check_same_thread=False``) and every public method holds a lock for
its whole statement sequence; cursors are per-call and never shared.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # lazy at runtime — fidelis.temporal lands independently
    from fidelis.temporal import SupersessionIndex

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".temporal.sqlite"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS records ("
    " record_id TEXT PRIMARY KEY, sha TEXT, recorded_at TEXT,"
    " valid_from TEXT, valid_to TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_records_sha ON records(sha)",
    "CREATE TABLE IF NOT EXISTS supersedes ("
    " target_id TEXT, superseder_id TEXT, recorded_at TEXT,"
    " PRIMARY KEY(target_id, superseder_id))",
)
# ``valid_from``/``valid_to`` (PACKET F2c) let /stats compute current /
# superseded / expired / not-yet-valid counts from this lightweight sidecar
# alone, without touching the vector store's payloads. Added via migration
# below so a sidecar file created before this change still opens.
_RECORDS_MIGRATED_COLUMNS = ("valid_from", "valid_to")

# Sidecar paths already warned about; open_index is called per request on
# some paths and must not flood the log while the sidecar stays broken.
_warned_paths: set[str] = set()
_warned_lock = threading.Lock()


def sidecar_path(store_path: str | Path) -> Path:
    p = Path(store_path)
    return p.parent / (p.name + SIDECAR_SUFFIX)


def _parse_or_none(recorded_at: str | None) -> datetime | None:
    """Best-effort ISO-8601 parse (``Z`` or offset, any sub-second precision).

    ``None`` on anything unparseable -- never guessed into a date. Shared by
    :func:`_instant_key` and the status/paging queries below so there is one
    definition of "what counts as a valid instant" in this module.
    """
    raw = (recorded_at or "").strip()
    if not raw:
        return None
    try:
        s = raw
        if s[-1:] in ("Z", "z"):
            s = s[:-1] + "+00:00"
        # 3.10's fromisoformat only takes 3- or 6-digit fractions; pad.
        head, sep, tail = s.partition(".")
        if sep:
            digits = ""
            while tail and tail[0].isdigit():
                digits, tail = digits + tail[0], tail[1:]
            s = head + "." + (digits + "000000")[:6] + tail
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, IndexError):
        return None


def _instant_key(recorded_at: str, record_id: str) -> tuple:
    """Sort key for "oldest first".

    ``recorded_at`` is ISO-8601 with a ``Z`` suffix at second precision OR
    finer, and mixed precision does not sort lexically ("…:00Z" > "…:00.5Z"),
    so compare parsed instants. Unparseable values sort last, by raw string —
    never guessed into a date.
    """
    raw = recorded_at or ""
    dt = _parse_or_none(raw)
    if dt is None:
        return (1, datetime.max.replace(tzinfo=timezone.utc), raw, record_id)
    return (0, dt, raw, record_id)


def _clean_ids(supersedes: object, self_id: str) -> list[str]:
    """Normalise a supersedes declaration to a de-duplicated list of ids.

    A bare string is one id (not an iterable of characters). Non-string and
    empty entries are ignored. A self-reference is ignored too: the contract
    makes the caller reject it, and an edge X->X would mark a record as
    replaced by itself forever.
    """
    if isinstance(supersedes, str):
        supersedes = [supersedes]
    if not isinstance(supersedes, (list, tuple)):
        return []
    out: list[str] = []
    for t in supersedes:
        if isinstance(t, str) and t and t != self_id and t not in out:
            out.append(t)
    return out


class TemporalIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit, transactions are explicit below.
        self._conn = sqlite3.connect(
            str(self.path), timeout=5.0, check_same_thread=False,
            isolation_level=None,
        )
        try:
            # A non-sqlite file connects fine and only fails here.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._migrate()
        except Exception:
            self._conn.close()
            raise

    def _migrate(self) -> None:
        """Add columns a sidecar created before this change is missing.

        ``CREATE TABLE IF NOT EXISTS`` does not add columns to an existing
        table, so a pre-existing sidecar file needs an explicit, idempotent
        ``ALTER TABLE``. Existing rows get ``NULL`` valid_from/valid_to until
        the next write or rebuild -- acceptable for a rebuildable cache.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(records)")}
        for column in _RECORDS_MIGRATED_COLUMNS:
            if column not in existing:
                self._conn.execute(f"ALTER TABLE records ADD COLUMN {column} TEXT")

    # -- writes --------------------------------------------------------------

    def _insert(self, record_id: str, sha: str, recorded_at: str,
                targets: list[str], *, valid_from: str | None = None,
                valid_to: str | None = None) -> None:
        # INSERT OR IGNORE: the store is append-only, so the first write of an
        # id is the truth and a replayed/retried write must be a no-op.
        self._conn.execute(
            "INSERT OR IGNORE INTO records(record_id, sha, recorded_at, valid_from, valid_to)"
            " VALUES (?,?,?,?,?)",
            (record_id, sha, recorded_at, valid_from, valid_to),
        )
        self._conn.executemany(
            "INSERT OR IGNORE INTO supersedes(target_id, superseder_id, recorded_at)"
            " VALUES (?,?,?)",
            [(t, record_id, recorded_at) for t in targets],
        )

    def record_write(self, record_id: str, sha: str, recorded_at: str,
                     supersedes: list[str] | None = None, *,
                     valid_from: str | None = None, valid_to: str | None = None) -> None:
        """Index one persisted record. Idempotent per ``record_id``.

        ``valid_from``/``valid_to`` are optional and stored only so
        :meth:`status_counts` can compute current/expired/not-yet-valid
        counts from this sidecar alone (PACKET F2c); they are never
        consulted for supersession, dedup, or any other existing behaviour.
        """
        targets = _clean_ids(supersedes, record_id)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._insert(
                    record_id, sha, recorded_at, targets,
                    valid_from=valid_from, valid_to=valid_to,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def rebuild(self, rows: Iterable[tuple[str, Mapping[str, object] | None]]) -> int:
        """Replace the whole index from ``(record_id, payload)`` pairs.

        Legacy rows (no ``content_sha256`` / ``recorded_at``) are skipped. A
        malformed ``supersedes_json`` costs that row its edges, not the
        rebuild. ``rows`` is fully consumed BEFORE the transaction opens, so a
        store iterator that dies halfway leaves the previous index intact.
        Returns the number of records indexed.
        """
        parsed: dict[str, tuple[str, str, list[str], str | None, str | None]] = {}
        for row in rows:
            try:
                record_id, payload = row
            except (TypeError, ValueError):
                continue
            if not isinstance(record_id, str) or not record_id:
                continue
            if not isinstance(payload, Mapping):
                continue
            sha = payload.get("content_sha256")
            recorded_at = payload.get("recorded_at")
            if not (isinstance(sha, str) and sha
                    and isinstance(recorded_at, str) and recorded_at):
                continue
            if record_id in parsed:
                continue  # first occurrence wins, same as record_write
            targets: list[str] = []
            raw = payload.get("supersedes_json")
            if isinstance(raw, str) and raw:
                try:
                    targets = _clean_ids(json.loads(raw), record_id)
                except ValueError:
                    logger.warning(
                        "temporal index rebuild: malformed supersedes_json on %s; "
                        "record indexed without edges", record_id)
            valid_from = payload.get("valid_from")
            valid_to = payload.get("valid_to")
            parsed[record_id] = (
                sha, recorded_at, targets,
                valid_from if isinstance(valid_from, str) else None,
                valid_to if isinstance(valid_to, str) else None,
            )

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM supersedes")
                self._conn.execute("DELETE FROM records")
                for record_id, (sha, recorded_at, targets, valid_from, valid_to) in parsed.items():
                    self._insert(
                        record_id, sha, recorded_at, targets,
                        valid_from=valid_from, valid_to=valid_to,
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return len(parsed)

    # -- reads ---------------------------------------------------------------

    def find_by_sha(self, sha: str) -> str | None:
        """Oldest record_id stored with this content sha (ties: smallest id)."""
        with self._lock:
            found = self._conn.execute(
                "SELECT record_id, recorded_at FROM records WHERE sha = ?", (sha,)
            ).fetchall()
        if not found:
            return None
        return min(found, key=lambda r: _instant_key(r[1], r[0]))[0]

    def ids_by_sha(self, sha: str) -> list[str]:
        """Every record_id stored with this content sha, oldest first.

        More than one means the text alone does not identify a record (a
        re-assertion with a new validity window); callers must not guess.
        """
        with self._lock:
            found = self._conn.execute(
                "SELECT record_id, recorded_at FROM records WHERE sha = ?", (sha,)
            ).fetchall()
        found.sort(key=lambda r: _instant_key(r[1], r[0]))
        return [r[0] for r in found]

    def edges(self) -> list[tuple[str, str, str]]:
        """All ``(target_id, superseder_id, superseder_recorded_at)`` edges."""
        with self._lock:
            found = self._conn.execute(
                "SELECT target_id, superseder_id, recorded_at FROM supersedes"
            ).fetchall()
        found.sort(key=lambda e: (_instant_key(e[2], e[1])[:2], e[0], e[1]))
        return [(t, s, r) for t, s, r in found]

    def is_superseded(self, record_id: str) -> bool:
        """True when any indexed record declares it supersedes ``record_id``."""
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM supersedes WHERE target_id = ? LIMIT 1", (record_id,)
            ).fetchone() is not None

    def record_ids(self) -> list[str]:
        """Every indexed record_id — for set comparison against the store."""
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT record_id FROM records")]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def recent(self, *, limit: int, since: str | None = None) -> list[tuple[str, str]]:
        """``(record_id, recorded_at)`` pairs, newest first, up to ``limit``
        (PACKET F2b, backing ``POST /recent``).

        ``since`` (an ISO-8601 instant) is a LOWER bound: only records
        recorded at or after it ("what's new since X"), not a paging cursor.
        Reads the whole ``records`` table -- id/sha/time only, no text or
        vectors -- so this stays cheap at scale without touching the vector
        store; sorted with the same instant-aware key ``find_by_sha``/
        ``edges`` use, not a naive SQL ``ORDER BY`` (mixed second/microsecond
        precision does not sort lexically). A record never indexed (legacy,
        no recorded_at) is never returned here -- see :meth:`status_counts`
        for that count.
        """
        with self._lock:
            rows = self._conn.execute("SELECT record_id, recorded_at FROM records").fetchall()
        if since is not None:
            floor = _parse_or_none(since)
            if floor is not None:
                rows = [r for r in rows if (_parse_or_none(r[1]) or datetime.min.replace(
                    tzinfo=timezone.utc)) >= floor]
        rows.sort(key=lambda r: _instant_key(r[1], r[0]), reverse=True)
        return [(r[0], r[1]) for r in rows[: max(0, limit)]]

    def corrections(
        self, *, limit: int, since: str | None = None,
    ) -> list[tuple[str, str, list[str]]]:
        """``(superseder_id, recorded_at, [target_ids])`` triples, one row per
        record that declares ``supersedes``, newest first, up to ``limit``
        (PACKET F2b, backing ``POST /recent`` with ``kind="corrections"``).
        ``since`` is the same lower bound as :meth:`recent`. Reads the whole
        ``supersedes`` table only.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT target_id, superseder_id, recorded_at FROM supersedes"
            ).fetchall()
        grouped: dict[str, tuple[str, list[str]]] = {}
        for target_id, superseder_id, recorded_at in rows:
            _, targets = grouped.setdefault(superseder_id, (recorded_at, []))
            targets.append(target_id)
        triples = [
            (superseder_id, recorded_at, sorted(targets))
            for superseder_id, (recorded_at, targets) in grouped.items()
        ]
        if since is not None:
            floor = _parse_or_none(since)
            if floor is not None:
                triples = [
                    t for t in triples
                    if (_parse_or_none(t[1]) or datetime.min.replace(tzinfo=timezone.utc)) >= floor
                ]
        triples.sort(key=lambda t: _instant_key(t[1], t[0]), reverse=True)
        return triples[: max(0, limit)]

    def oldest_newest(self) -> tuple[str | None, str | None]:
        """``(oldest, newest)`` recorded_at among indexed records, instant-
        aware, or ``(None, None)`` when empty (PACKET F2c, backing stats)."""
        with self._lock:
            rows = self._conn.execute("SELECT record_id, recorded_at FROM records").fetchall()
        if not rows:
            return None, None
        rows.sort(key=lambda r: _instant_key(r[1], r[0]))
        return rows[0][1], rows[-1][1]

    def status_counts(self, *, now: str) -> dict[str, int]:
        """Cheap current/superseded/expired/not-yet-valid counts over every
        indexed record, evaluated at ``now`` (an ISO-8601 instant), backing
        ``GET /stats`` (PACKET F2c). Reads only this sidecar: ``record_write``
        and ``rebuild`` already carry ``valid_from``/``valid_to`` precisely
        so this never touches the vector store's payloads. Precedence
        matches :func:`fidelis.temporal.temporal_status`: superseded >
        expired > not_yet_valid > current. A record whose ``now`` cannot be
        parsed is never guessed at -- callers always pass a formatted instant.
        """
        reference = _parse_or_none(now)
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, valid_from, valid_to FROM records"
            ).fetchall()
            superseded_ids = {
                row[0]
                for row in self._conn.execute("SELECT DISTINCT target_id FROM supersedes")
            }
        counts = {"current": 0, "superseded": 0, "expired": 0, "not_yet_valid": 0}
        for record_id, valid_from, valid_to in rows:
            if record_id in superseded_ids:
                counts["superseded"] += 1
                continue
            valid_to_dt = _parse_or_none(valid_to)
            valid_from_dt = _parse_or_none(valid_from)
            if reference is not None and valid_to_dt is not None and reference >= valid_to_dt:
                counts["expired"] += 1
            elif reference is not None and valid_from_dt is not None and reference < valid_from_dt:
                counts["not_yet_valid"] += 1
            else:
                counts["current"] += 1
        return counts

    def supersession_index(self) -> SupersessionIndex:
        # Imported here, not at module top: this sidecar must stay importable
        # (dedup keeps working) even if fidelis.temporal is absent or broken.
        from fidelis.temporal import SupersessionIndex

        return SupersessionIndex.from_edges(self.edges())

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: silent — closing a cache must never raise into shutdown
                pass


def open_index(store_path: str | Path) -> TemporalIndex | None:
    """Open (creating if needed) the sidecar for ``store_path``.

    Returns ``None`` on ANY failure — reads must fail open. Warns once per
    sidecar path per process.
    """
    try:
        path = sidecar_path(store_path)
        return TemporalIndex(path)
    except Exception as e:
        key = str(store_path)
        with _warned_lock:
            first = key not in _warned_paths
            _warned_paths.add(key)
        if first:
            logger.warning(
                "temporal index unavailable for %s (%s: %s); continuing without it",
                key, type(e).__name__, e)
        return None
