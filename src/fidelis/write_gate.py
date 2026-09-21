"""Write-time admission gate for the memory store (TIME-AWARE-SPEC v1).

The store is append-only, so junk that gets in stays in. The 2026-09-20 audit
found the live store 60-70% machine exhaust: thousands of copies of one
sub-agent prompt template, raw tool-result dumps, per-turn session markers,
health-probe residue, and memories carrying literal harness control markup.
Read-time filtering (``supersession.filter_ephemera``) hides that after the
fact; this module refuses it at the door.

Design rules:

* **Pure and deterministic.** No I/O, no clock, no network, no model. The same
  text always yields the same decision, so a rejection is reproducible and a
  caller can be told exactly which stable reason code fired.
* **Match structure, not vocabulary.** A durable fact that merely DISCUSSES
  harness tags, canary probes, or yes/no prompts in prose must be accepted.
  Every rule therefore keys on shape: an angle bracket directly followed by a
  control-tag name, a probe id with its id body, a template opener at the
  START of a turn, a key prefix followed by a realistic key body.
* **Secrets are never negotiable.** ``config["write_gate"] == False`` turns
  off the exhaust rules but not ``secret_like``, and ``detail`` never echoes
  the matched secret (only the rule name, offset, and length).
* **Fail-open on non-text.** Input that is not a ``str`` is not this gate's
  problem; it is accepted here and left to the caller's own validation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

# Stable reason codes — callers and the HTTP layer surface these verbatim.
HARNESS_ENVELOPE = "harness_envelope"
SCAFFOLD_PROMPT = "scaffold_prompt"
PROBE_RESIDUE = "probe_residue"
SESSION_EXHAUST = "session_exhaust"
SECRET_LIKE = "secret_like"

REASONS = (
    HARNESS_ENVELOPE,
    SCAFFOLD_PROMPT,
    PROBE_RESIDUE,
    SESSION_EXHAUST,
    SECRET_LIKE,
)


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    reason: str | None = None
    detail: str | None = None


_ACCEPT = GateDecision(accept=True, reason=None, detail=None)


# --- storage envelope -------------------------------------------------------
# Session rows are flattened turn-pair dumps ("User: ...\nAssistant: ...") and
# hand-tagged rows carry a leading bracket tag ("[voice:user] ..."). Same
# convention as supersession._strip_envelope, re-implemented locally so this
# module stays import-free. One difference: a bracket that IS a tool_result
# dump is never peeled, otherwise "[tool_result: ok]" would strip itself away
# before the harness rule could see it.
_ENVELOPE_RE = re.compile(
    r"^\s*(?:(?:User|Assistant)\s*:\s*|\[(?!\s*tool_result)[^\[\]]{1,40}\]\s*)",
    re.I,
)


def _strip_envelope(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _ENVELOPE_RE.sub("", s, count=1)
    return s


def _turn_starts(text: str) -> list[str]:
    """Strings whose position 0 is the start of a turn's actual content.

    Always the whole text with its envelope peeled; plus every line that
    carried an envelope of its own (multi-turn dumps put one turn per line).
    A plain continuation line in the middle of prose is deliberately NOT a
    turn start, so start-anchored rules cannot fire on a sentence that just
    happens to wrap there.
    """
    starts = [_strip_envelope(text)]
    for line in text.splitlines():
        stripped = _strip_envelope(line)
        if stripped != line:
            starts.append(stripped)
    return starts


# --- harness_envelope -------------------------------------------------------
# Tag STRUCTURE: "<", optional "/", a control-tag name, then a tag terminator
# (whitespace, "/", ">"). The bare word "system-reminder" in a sentence has no
# angle bracket and does not match; "<tool_use_id" has no terminator after the
# name and does not match either.
_HARNESS_TAG_NAMES = (
    "task-notification",
    "system-reminder",
    "function_calls",
    "tool_use",
    "tool_result",
)
_HARNESS_TAG_RE = re.compile(
    "<"
    + r"/?(?P<name>"
    + "|".join(re.escape(n) for n in _HARNESS_TAG_NAMES)
    + r"|antml:[A-Za-z_][\w.\-]*"
    + r")(?=[\s/>])",
    re.I,
)
_TOOL_RESULT_BRACKET_RE = re.compile(r"^\s*\[\s*tool_result", re.I)

# --- scaffold_prompt --------------------------------------------------------
_SCAFFOLD_START_RE = re.compile(
    r"^\s*(?:You are QA for\b|Extract discrete, atomic\b|Output only imperative rules\b)",
    re.I,
)
# The colon and the word ONLY are what make this a template instruction rather
# than prose ("answer yes or no when asked a yes/no question" stays legal).
_SCAFFOLD_ANYWHERE_RE = re.compile(r"\bAnswer\s+ONLY\s*:\s*YES\s+or\s+NO\b", re.I)

# --- probe_residue ----------------------------------------------------------
# Requires the id body after the final hyphen: "canary-probe-ad4f3ff1" is
# residue, "the canary probe" and a dangling "canary-probe-" prefix are not.
_PROBE_RE = re.compile(r"canary-probe-[A-Za-z0-9]", re.I)

# --- session_exhaust --------------------------------------------------------
_SESSION_RE = re.compile(r"^\s*SESSION \d{4}-\d{2}-\d{2} .*transcript evidence")

# --- secret_like ------------------------------------------------------------
# Each pattern demands a realistic key BODY; a bare prefix mention ("tokens
# start with ghp_") is documentation, not a secret. The left guards stop a
# prefix from matching inside an ordinary word ("task-...", "risk-...").
_SECRET_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block",
     re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("sk_api_key",
     re.compile(r"(?<![A-Za-z0-9_\-])sk-[A-Za-z0-9_\-]{20,}")),
    ("github_token",
     re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws_access_key_id",
     re.compile(r"(?<![A-Za-z0-9])AKIA[A-Z0-9]{16}(?![A-Za-z0-9])")),
    ("slack_token",
     re.compile(r"(?<![A-Za-z0-9_\-])xox[bp]-[A-Za-z0-9\-]{20,}")),
)


def _check_secret(text: str) -> GateDecision | None:
    for name, rx in _SECRET_RES:
        m = rx.search(text)
        if m:
            # Rule name + position only. Never m.group(): detail is logged
            # and returned over HTTP, and must not become a second leak.
            return GateDecision(
                False, SECRET_LIKE,
                f"{name} at offset {m.start()} ({m.end() - m.start()} chars, redacted)",
            )
    return None


def _check_harness(text: str, starts: list[str]) -> GateDecision | None:
    # LEADING tags only. Raw harness dumps begin with the control tag (after an
    # optional role/[voice:*] prefix, which ``starts`` has already peeled). A
    # tag quoted mid-sentence or shown in a code block is the owner writing
    # ABOUT harnesses — rejecting that is data loss with a 422.
    for s in starts:
        m = _HARNESS_TAG_RE.match(s.lstrip())
        if m:
            return GateDecision(False, HARNESS_ENVELOPE,
                                f"leading control tag '{m.group('name').lower()}'")
    if any(_TOOL_RESULT_BRACKET_RE.match(s) for s in starts):
        return GateDecision(False, HARNESS_ENVELOPE, "leading [tool_result bracket")
    return None


def _check_scaffold(text: str, starts: list[str]) -> GateDecision | None:
    for s in starts:
        m = _SCAFFOLD_START_RE.match(s)
        if m:
            return GateDecision(False, SCAFFOLD_PROMPT,
                                f"template opener '{m.group(0).strip()}'")
    if _SCAFFOLD_ANYWHERE_RE.search(text):
        return GateDecision(False, SCAFFOLD_PROMPT, "template instruction 'Answer ONLY: YES or NO'")
    return None


def _check_probe(text: str, starts: list[str]) -> GateDecision | None:
    m = _PROBE_RE.search(text)
    if m:
        return GateDecision(False, PROBE_RESIDUE, f"probe id at offset {m.start()}")
    return None


def _check_session(text: str, starts: list[str]) -> GateDecision | None:
    if any(_SESSION_RE.match(s) for s in starts):
        return GateDecision(False, SESSION_EXHAUST, "per-turn SESSION transcript-evidence marker")
    return None


# Order is part of the contract only in that secret_like goes first (handled
# in evaluate); the rest run most-structural first so the reported reason is
# the most specific one.
_EXHAUST_CHECKS = (_check_harness, _check_scaffold, _check_probe, _check_session)


def _gate_enabled(config: Mapping[str, object] | None) -> bool:
    if not isinstance(config, Mapping):
        return True
    return config.get("write_gate", True) is not False


def evaluate(text: str, *, config: Mapping[str, object] | None = None) -> GateDecision:
    """Decide whether ``text`` may be persisted.

    Returns ``GateDecision(accept=True, reason=None, detail=None)`` or a
    rejection carrying one of the stable codes in :data:`REASONS`.
    ``secret_like`` is evaluated first and is NEVER negotiable: it survives
    ``write_gate: false`` and there is no config key of any kind — including
    a would-be ``write_gate_secrets`` — that can disable it (policy
    2026-09-21: a prior, unrequested opt-out was removed for exactly this
    reason).
    """
    if not isinstance(text, str) or not text:
        return _ACCEPT
    # Always on. Secret refusal has no OFF switch, by design (see module
    # docstring "Secrets are never negotiable"); do not add one.
    hit = _check_secret(text)
    if hit is not None:
        return hit
    if not _gate_enabled(config):
        return _ACCEPT
    starts = _turn_starts(text)
    for check in _EXHAUST_CHECKS:
        hit = check(text, starts)
        if hit is not None:
            return hit
    return _ACCEPT
