"""
fidelis — faithful memory retrieval for AI agents.

Default recall uses local vector search without a generative LLM, followed
by deterministic Cogito Hermeneutics controls. Optional thorough recall adds
zero-LLM hybrid fusion; BM25 requires the hybrid extra.

Optional LLM tiers (experimental): ``filter`` and ``flagship``. The filter
LLM outputs only integer indices (e.g. [3, 7, 12]) — never memory text —
so it cannot corrupt or hallucinate into the content returned to the agent.
Fidelity is structural, not a prompting convention.

fidelis was previously published as ``cogito-ergo`` (0.0.8 and 0.3.0 on PyPI).
Data paths and env var names retain the ``cogito`` prefix for continuity
with existing deployments.
"""

__version__ = "0.3.0rc1"

from fidelis.inquiry import inquire  # noqa: F401
from fidelis.recall import recall  # noqa: F401

# Note: avoid shadowing the ``fidelis.recall_hybrid`` submodule. Users who want
# the function directly can do ``from fidelis.recall_hybrid import recall_hybrid``.
