# Releasing Fidelis Memory

## Version strategy

Fidelis uses Semantic Versioning for the installable `fidelis-memory` package.
While the major version is zero, the public API is still evolving:

- `0.3.0rc1` is the redesigned memory pre-release: six MCP tools, explicit
  corrections and validity dates, with public install safeguards retained.
- Breaking pre-1.0 changes require a minor version and migration notes.
- `1.0.0` requires an explicitly stable public contract.

The archived `cogito-ergo` entries in `CHANGELOG.md` describe a predecessor
package. They do not set the current Fidelis package version.

## Coordinated version surfaces

Every release must carry the same version in:

- `pyproject.toml` and `src/fidelis/__init__.py`;
- `server.json`, `gemini-extension.json`, and the Claude plugin manifests;
- `CITATION.cff` and `codemeta.json`;
- current install commands in `README.md`, `llms.txt`, and
  `docs/full-reference.md`, plus container metadata; and
- the public-install assertions in `tests/test_public_install_truth.py`.

Historical changelog text and protocol-marker versions are not mechanically
rewritten.

## Release format

Use this order in the GitHub release body:

1. **Why this release:** one sentence naming the user outcome.
2. **Who it is for:** link to the user-fit matrix and state the supported OS and
   client boundary.
3. **Install or upgrade:** one pinned command plus the native Gemini route when
   relevant.
4. **What changed:** only shipped behavior, packaging, and documentation.
5. **Evidence:** CI/release checks and the release-readiness record.
6. **Known limits:** link to the README section; never hide a non-fit.
7. **Next:** link to the outcome-gated roadmap without presenting it as shipped.

For this candidate use [`releases/0.3.0rc1.md`](releases/0.3.0rc1.md) and
mark the GitHub release as a pre-release.

## Release sequence

1. Merge the coordinated version PR only after repository checks, Hermes Gate,
   and a bounded independent review pass.
2. Create annotated tag `vX.Y.Z` at the exact merged commit and push the tag.
3. Dispatch `.github/workflows/release.yml` with that existing tag. It verifies
   the tag/version match, lints, tests, builds, runs `twine check`, publishes to
   PyPI by OIDC, then publishes `server.json` to the MCP Registry.
4. Verify the GitHub Release, PyPI version, MCP Registry version, source archive,
   and Gemini extension manifest from public URLs. A green workflow without
   those read-backs is not a finished release.
5. Test one fresh wheel install and one client start from the public artifacts.
6. If publication fails after PyPI accepted the version, never overwrite it.
   Fix forward with a patch release and make the incomplete coordinate explicit
   in the release notes.
