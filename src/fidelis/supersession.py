"""Read-time supersession annotation (DEEPDIVE-20260718 R2).

The store is append-only by design; currency is a LOOKUP, not a property of
the store. This module consults a machine-readable pointers file (built by
an operator and selected with ``supersession_pointers_path``)
and ANNOTATES retracted/superseded/caveated hits in recall responses so no
consumer can unknowingly cite a dead claim. Hits are never dropped on pull
paths — an agent asking about a retracted claim should see the retraction,
not silence. (Push channels apply their own stricter filtering offline.)

Fail-open: missing/corrupt pointers file, or ``supersession_pointers_path``
set falsy in config, means responses pass through untouched.
"""

from __future__ import annotations

import json
import os
import re

_DEFAULT_PATH = None

# (path, mtime) -> compiled rules; single-entry cache, reloaded on file change.
_cache: dict = {"key": None, "rules": None}


def _load_rules(path: str):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime)
    if _cache["key"] == key:
        return _cache["rules"]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rules = [
            (re.compile(r["pattern"], re.I), r.get("status", "SUPERSEDED"),
             r.get("note", ""), r.get("id", ""))
            for r in data.get("retracted", [])
        ]
    except Exception:
        return None
    _cache["key"] = key
    _cache["rules"] = rules
    return rules


def _load_ephemera(path: str):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime, "ephemera")
    if _cache.get("ekey") == key:
        return _cache.get("epatterns")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pats = [re.compile(p, re.I) for p in data.get("ephemera_patterns", [])]
    except Exception:
        return None
    _cache["ekey"] = key
    _cache["epatterns"] = pats
    return pats


# mem_type=session rows are stored as flattened turn-pair dumps, e.g.
# "User: You are QA for a drafted fix...\nAssistant: YES\nUser: ..."
# (see ingest_claude_sessions.py::_session_to_text) and some hand-tagged
# rows carry a leading bracket tag, e.g. "[voice:user] [tool_result: ...]".
# ephemera_patterns anchor on unprefixed content (``^\s*You are``,
# ``^\s*\[?tool_result``), so those anchors never see position 0 of the raw
# stored string once an envelope precedes it. _strip_envelope peels one
# leading "User:"/"Assistant:" role-prefix or one leading "[tag]" bracket
# off a string; filter_ephemera applies it per-line (each turn lives on its
# own line) so the anchors line up with each turn's actual content start.
_ENVELOPE_RE = re.compile(r"^\s*(?:(?:User|Assistant)\s*:\s*|\[[^\[\]]{1,40}\]\s*)", re.I)


def _strip_envelope(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _ENVELOPE_RE.sub("", s, count=1)
    return s


def _is_ephemera(text: str, pats) -> bool:
    """True if any ephemera pattern matches ``text`` as stored, OR matches
    once a leading storage envelope (role-prefix / bracket-tag) is stripped
    — checked both on the whole string and line-by-line, since multi-turn
    session dumps carry one envelope per line."""
    if not text:
        return False
    if any(p.search(text) for p in pats):
        return True
    stripped_whole = _strip_envelope(text)
    if stripped_whole != text and any(p.search(stripped_whole) for p in pats):
        return True
    for line in text.splitlines():
        stripped_line = _strip_envelope(line)
        if stripped_line != line and any(p.search(stripped_line) for p in pats):
            return True
    return False


def filter_ephemera(memories: list, cfg: dict, limit: int | None = None) -> list:
    """Drop hits that are task-state junk (raw tool dumps, scaffold-prompt
    templates, 'user is currently...' ephemera) per the pointers file's
    ephemera_patterns. Measured 2026-07-18 (P3, n=200 stratified): 50.5% of
    the store is ephemera, so unfiltered top-k is majority junk for many
    queries. Call sites over-fetch so filtering doesn't starve the response.
    Fail-open like annotate_superseded. Set cfg key
    ``ephemera_filter: false`` to disable."""
    if cfg.get("ephemera_filter", True) is False:
        return memories[:limit] if limit else memories
    path = cfg.get("supersession_pointers_path", _DEFAULT_PATH)
    if not path:
        return memories[:limit] if limit else memories
    pats = _load_ephemera(os.path.expanduser(path))
    if not pats:
        return memories[:limit] if limit else memories
    kept = [m for m in memories if not _is_ephemera(m.get("text") or "", pats)]
    return kept[:limit] if limit else kept


def mark_ephemera(memories: list, cfg: dict) -> list:
    """Attach ephemera metadata without dropping or rewriting source text.

    The legacy read paths intentionally use :func:`filter_ephemera`. Cogito
    Hermeneutics instead consumes this structured marker as a low-information
    signal so V0.4 can downweight noisy rows while retaining provenance.
    """
    if cfg.get("ephemera_filter", True) is False:
        return memories
    path = cfg.get("supersession_pointers_path", _DEFAULT_PATH)
    if not path:
        return memories
    pats = _load_ephemera(os.path.expanduser(path))
    if not pats:
        return memories
    out = []
    for memory in memories:
        text = memory.get("text") or ""
        if _is_ephemera(text, pats):
            memory = dict(memory)
            memory["ephemera"] = {
                "matched": True,
                "source": "current_truth_ephemera_pattern",
            }
        out.append(memory)
    return out


def mark_superseded(memories: list, cfg: dict) -> list:
    """Attach supersession metadata without changing verbatim source text."""
    path = cfg.get("supersession_pointers_path", _DEFAULT_PATH)
    if not path:
        return memories
    rules = _load_rules(os.path.expanduser(path))
    if not rules:
        return memories
    out = []
    for m in memories:
        text = m.get("text") or ""
        hit = next(((status, note, rid) for rx, status, note, rid in rules
                    if rx.search(text)), None)
        if hit:
            status, note, rid = hit
            m = dict(m)
            m["supersession"] = {"status": status, "id": rid, "note": note}
        out.append(m)
    return out


def annotate_superseded(memories: list, cfg: dict) -> list:
    """Prefix supersession status for legacy consumers.

    Cogito Hermeneutics uses :func:`mark_superseded` so source text remains
    byte-identical while the same control is available as structured metadata.
    """
    out = []
    for m in mark_superseded(memories, cfg):
        hit = m.get("supersession")
        if hit:
            m = dict(m)
            status, note, rid = hit["status"], hit["note"], hit["id"]
            text = m.get("text") or ""
            m["text"] = f"[{status}:{rid}] {note} || original: {text}"
        out.append(m)
    return out
