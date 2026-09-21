# Fidelis — agent integration guide

The current pre-release is 0.3.0rc1. Install `fidelis-memory`; import and CLI names
remain `fidelis`. Read [README.md](README.md) and the
[API reference](docs/full-reference.md) for setup and request shapes.

The six MCP tools are `fidelis_recall`, `fidelis_store`, `fidelis_correct`,
`fidelis_get`, `fidelis_recent`, and `fidelis_health`. Default recall calls
`/query` using local embeddings without a generative LLM. Thorough recall is
explicit and uses the zero-LLM hybrid tier. Optional legacy filters/extraction
are separate model-using paths.

Recall when an answer depends on recorded work. Preserve original wording,
qualifiers, stable IDs, source metadata, and temporal status. A recorded claim
is not verified truth. A low score or empty result does not prove absence.
Use get-by-ID for complete text and correction chains. Store only intended
durable facts; correct through links, retaining the original record as history.
Read write acknowledgements literally; queued writes have not landed yet.

The HTTP service binds loopback by default. `COGITO_USER_ID` is a local storage
namespace, not authenticated identity. Do not represent it as tenant isolation.
`FIDELIS_PORT` configures both clients and server, with `COGITO_PORT` as fallback.
The data directory remains `~/.cogito/` for compatibility.

Do not reuse historical benchmark headlines as evidence for the new default
retrieval path. Current claims must link to the current candidate evaluation.
