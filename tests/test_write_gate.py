"""Contract tests for fidelis.write_gate (docs/TIME-AWARE-SPEC.md).

Every harness tag and fake credential below is assembled by concatenation so
this file never contains raw control markup or anything a secret scanner
would flag.
"""

from __future__ import annotations

import dataclasses

import pytest

from fidelis import write_gate
from fidelis.write_gate import GateDecision, evaluate

LT, GT = "<", ">"


def _open(name: str) -> str:
    return LT + name + GT


def _close(name: str) -> str:
    return LT + "/" + name + GT


TAG_NAMES = [
    "task-notification",
    "system-reminder",
    "function_calls",
    "tool_use",
    "tool_result",
    "antml:" + "invoke",
    "antml:" + "parameter",
]

FAKE_SECRETS = {
    "sk": "sk-" + "aB3d" * 6,
    "sk_ant": "sk-" + "ant-api03-" + "Zz9_" * 8,
    "ghp": "ghp_" + "a" * 36,
    "gho": "gho_" + "B7" * 18,
    "akia": "AKIA" + "Q7" * 8,
    "xoxb": "xoxb-" + "1234567890-" * 2 + "abcdefAB",
    "xoxp": "xoxp-" + "9" * 24,
    "pem": "-----BEGIN " + "RSA PRIVATE KEY" + "-----\nMIIEow" + "A" * 40,
    "pem_plain": "-----BEGIN " + "PRIVATE KEY" + "-----",
    "pem_openssh": "-----BEGIN " + "OPENSSH PRIVATE KEY" + "-----",
}


# --- shape of the decision ---------------------------------------------------

def test_decision_is_frozen_dataclass_with_contract_fields():
    assert dataclasses.is_dataclass(GateDecision)
    assert [f.name for f in dataclasses.fields(GateDecision)] == ["accept", "reason", "detail"]
    d = evaluate("Fidelis server listens on port 19420.")
    assert d == GateDecision(accept=True, reason=None, detail=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.accept = False  # type: ignore[misc]


def test_reason_codes_are_exactly_the_contract_set():
    assert set(write_gate.REASONS) == {
        "harness_envelope", "scaffold_prompt", "probe_residue",
        "session_exhaust", "secret_like",
    }


def test_config_is_keyword_only():
    with pytest.raises(TypeError):
        evaluate("hello", {"write_gate": False})  # type: ignore[misc]


def test_deterministic():
    text = "User: You are QA for a drafted fix."
    assert evaluate(text) == evaluate(text)


@pytest.mark.parametrize("value", ["", None, 17, b"bytes"])
def test_non_text_and_empty_fail_open(value):
    assert evaluate(value).accept is True  # type: ignore[arg-type]


# --- precision: prose ABOUT exhaust must be accepted -------------------------

ACCEPTED_PROSE = [
    "The store audit on 2026-09-20 found task-notification tags inside stored memories.",
    "Decision: session hooks must send one marker per session, not one per turn.",
    "The canary probe runs every 300 seconds and only reads.",
    "Maya prefers terse replies; answer yes or no when asked a yes/no question.",
    "GitHub tokens start with ghp_ and must never be stored.",
    # more of the same family
    "Fidelis server listens on port 19420.",
    "system-reminder and tool_result blocks are harness markup, never memory.",
    "The tool_use count for that run was 33; tool_result payloads were large.",
    "Probe ids are named canary-probe- plus eight hex characters.",
    "OpenAI keys begin with sk- and AWS key ids begin with AKIA.",
    "Slack bot tokens look like xoxb- followed by digits.",
    "The task-risk-assessment-for-the-quarterly-planning doc is in Notion.",
    "In the 2026-09-20 session we gathered transcript evidence for the audit.",
    "The QA sub-agent is told it is QA for a drafted fix and must reply tersely.",
    "Rule: if a < b and b > c then skip; tool_use is unaffected.",
    "Generic type List<str> and the tag <b>bold</b> are fine.",
    "We use " + LT + "tool_use_id" + GT + " as a placeholder name in the docs.",
    "[voice:user] I want memory to be verbatim and append-only.",
    "User: my sister's birthday is 2026-11-03.\nAssistant: Noted.",
    "Private keys must stay in the keychain; never paste a PRIVATE KEY block.",
    "  Extract is a verb. Later line:\nExtract discrete, atomic facts is how the old prompt began.",
]


@pytest.mark.parametrize("text", ACCEPTED_PROSE)
def test_prose_about_exhaust_is_accepted(text):
    d = evaluate(text)
    assert d.accept is True, (d.reason, d.detail)
    assert d.reason is None and d.detail is None


# --- harness_envelope --------------------------------------------------------

@pytest.mark.parametrize("name", TAG_NAMES)
@pytest.mark.parametrize("maker", [_open, _close])
def test_harness_tags_rejected(name, maker):
    # A raw harness dump LEADS with the control tag.
    d = evaluate(maker(name) + " trailing words.")
    assert (d.accept, d.reason) == (False, "harness_envelope")
    assert d.detail


@pytest.mark.parametrize("name", TAG_NAMES)
@pytest.mark.parametrize("maker", [_open, _close])
def test_harness_tag_quoted_mid_text_is_accepted(name, maker):
    # The same tag quoted inside a sentence is the owner writing ABOUT
    # harnesses. Rejecting it would be data loss with a 422 (maintainer
    # review 2026-09-20), so only leading tags are gated.
    d = evaluate("Some memory text " + maker(name) + " trailing words.")
    assert d.accept is True, (d.reason, d.detail)


def test_harness_tag_in_code_fence_is_accepted():
    text = "Note on harness design:\n```\n" + _open("tool_result") + "example" + _close("tool_result") + "\n```"
    assert evaluate(text).accept is True


def test_harness_tag_with_attributes_and_self_closing():
    assert evaluate(LT + "antml:" + 'invoke name="Read"' + GT).reason == "harness_envelope"
    assert evaluate(LT + "tool_result" + "/" + GT + " x").reason == "harness_envelope"
    assert evaluate(LT + "tool_use" + "\n id=1" + GT).reason == "harness_envelope"


def test_harness_tag_inside_role_prefixed_dump():
    text = "User: hi\nAssistant: " + _open("system-reminder") + "be nice" + _close("system-reminder")
    assert evaluate(text).reason == "harness_envelope"


@pytest.mark.parametrize("text", [
    "[tool_result: exit 0] total 48",
    "  [tool_result] ok",
    "[voice:user] [tool_result: 200 OK]",
    "User: run it\nAssistant: [tool_result: done]",
])
def test_leading_tool_result_bracket_rejected(text):
    assert evaluate(text).reason == "harness_envelope"


def test_tool_result_bracket_mid_sentence_is_prose():
    assert evaluate("Dumps that begin with [tool_result are junk.").accept is True


# --- scaffold_prompt ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "You are QA for a drafted fix. Answer ONLY: YES or NO",
    "User: You are QA for a drafted fix.\nAssistant: YES",
    "[session] User: You are QA for a drafted fix.",
    "Assistant: ok\nUser: You are QA for the next patch.",
    "Given the diff below. Answer ONLY: YES or NO",
    "answer only:  yes or no",
    "Extract discrete, atomic facts from the conversation below.",
    "User: Extract discrete, atomic facts.",
    "Output only imperative rules, one per line.",
])
def test_scaffold_prompts_rejected(text):
    d = evaluate(text)
    assert (d.accept, d.reason) == (False, "scaffold_prompt")


# --- probe_residue -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "User mentioned 'canary-probe-ad4f3ff1'",
    "canary-probe-0",
    "note: CANARY-PROBE-AD4F3FF1 seen",
])
def test_probe_residue_rejected(text):
    d = evaluate(text)
    assert (d.accept, d.reason) == (False, "probe_residue")


# --- session_exhaust ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "SESSION 2026-09-20 17:57 bcd13a4d: transcript evidence — 3 file(s)… 33 tool calls",
    "[auto] SESSION 2026-09-20 17:57 bcd13a4d: transcript evidence",
    "User: SESSION 2026-01-02 00:00 x: transcript evidence",
])
def test_session_exhaust_rejected(text):
    d = evaluate(text)
    assert (d.accept, d.reason) == (False, "session_exhaust")


@pytest.mark.parametrize("text", [
    "The SESSION 2026-09-20 17:57 marker said transcript evidence, which is noise.",
    "SESSION 2026-09-20 17:57 bcd13a4d: decided to ship the gate.",
    "SESSION 2026-09-20 notes\ntranscript evidence is kept in the vault.",
])
def test_session_lookalikes_accepted(text):
    assert evaluate(text).accept is True


# --- secret_like -------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(FAKE_SECRETS))
def test_secrets_rejected_and_never_echoed(kind):
    secret = FAKE_SECRETS[kind]
    d = evaluate("deploy note: the credential is " + secret + " keep safe")
    assert (d.accept, d.reason) == (False, "secret_like")
    assert d.detail
    assert secret not in d.detail
    # no fragment of the key body leaks either
    body = secret.split("\n")[0][-12:]
    assert body not in d.detail
    assert secret[:8] not in d.detail


@pytest.mark.parametrize("text", [
    "sk-" + "a" * 19,            # body too short
    "ghp_" + "a" * 29,
    "gho_" + "a" * 10,
    "AKIA" + "Q7" * 7,           # 14 chars
    "AKIA" + "q7" * 8,           # lowercase body
    "xoxb-" + "1" * 19,
    "-----BEGIN PUBLIC KEY-----",
    "-----BEGIN CERTIFICATE-----",
    "disk-" + "a" * 30,          # 'sk-' inside a word
])
def test_near_miss_secrets_accepted(text):
    d = evaluate(text)
    assert d.accept is True, (d.reason, d.detail)


def test_secret_like_is_evaluated_first():
    text = _open("system-reminder") + " canary-probe-ad4f3ff1 " + FAKE_SECRETS["ghp"]
    assert evaluate(text).reason == "secret_like"


# --- config ------------------------------------------------------------------

REJECTED_NON_SECRET = [
    _open("task-notification") + " x",
    "[tool_result: ok]",
    "You are QA for a drafted fix. Answer ONLY: YES or NO",
    "User mentioned 'canary-probe-ad4f3ff1'",
    "SESSION 2026-09-20 17:57 bcd13a4d: transcript evidence — 3 file(s)",
]


@pytest.mark.parametrize("text", REJECTED_NON_SECRET)
def test_write_gate_false_disables_exhaust_rules(text):
    assert evaluate(text).accept is False
    assert evaluate(text, config={"write_gate": False}) == GateDecision(True, None, None)


def test_write_gate_false_keeps_secret_like():
    d = evaluate("token " + FAKE_SECRETS["akia"], config={"write_gate": False})
    assert (d.accept, d.reason) == (False, "secret_like")


def test_write_gate_secrets_opt_out_cannot_disable_secret_detection():
    """Owner decision B (2026-09-21): secret refusal is never negotiable.

    No config key -- including one purpose-built to sound like it should --
    may disable secret_like. The opt-out `write_gate_secrets: False` was
    added without being requested and must have no effect, alone or
    combined with `write_gate: False`."""
    fake_key = "AKIA" + "Q7" * 8
    d = evaluate("token " + fake_key, config={"write_gate_secrets": False})
    assert (d.accept, d.reason) == (False, "secret_like")
    d = evaluate(
        "token " + fake_key,
        config={"write_gate": False, "write_gate_secrets": False},
    )
    assert (d.accept, d.reason) == (False, "secret_like")


@pytest.mark.parametrize("config", [
    None, {}, {"write_gate": True}, {"write_gate": None}, {"write_gate": 0},
    {"other": False},
])
def test_gate_stays_on_unless_explicitly_false(config):
    assert evaluate("[tool_result: ok]", config=config).reason == "harness_envelope"


def test_input_is_not_mutated_or_normalised():
    text = "  User:  Fidelis listens on 19420.  \n"
    before = str(text)
    evaluate(text)
    assert text == before
