# Fidelis Memory

This extension connects to the local Fidelis service. Retrieve original recorded
text when a turn depends on earlier work, decisions, or changing facts.

- `fidelis_recall`: query relevant records; default fast mode uses no generative
  LLM. Use `as_of` for a historical validity date and `mode: "thorough"` only when
  broader hybrid retrieval is useful.
- `fidelis_store`: retain a fact the user intends to keep. Read the acknowledgement:
  stored, duplicate, rejected, and queued are different outcomes.
- `fidelis_correct`: create a replacement linked to an existing ID. The original
  stays in history. Do not silently overwrite a conflicting record.
- `fidelis_get`: fetch full text and correction links for a known ID.
- `fidelis_recent`: browse recent records or corrections.
- `fidelis_health`: inspect availability; an unloaded or unreachable store is not
  an empty store.

Preserve qualifiers when quoting. Read temporal status: superseded records are
history, not current evidence. Scores are ranking signals, not truth confidence.
An empty search cannot prove that a fact was never recorded. Store no secrets.
When memory is unavailable, report that and do not invent a remembered answer.
Unrelated turns need no tool call. The client decides when memory is invoked.

Setup and reference: https://github.com/hermes-labs-ai/fidelis
