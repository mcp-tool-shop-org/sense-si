# Ship Gate

> No repo is "done" until every applicable line is checked.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

**Detected:** `[all]` `[pypi]` `[mcp]` `[cli]`

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) (2026-04-09)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) (2026-04-09)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output (2026-04-09)
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-04-09)

### Default safety posture

- [x] `[cli|mcp|desktop]` SKIP: no dangerous actions — all tools are read-only image analysis
- [x] `[cli|mcp|desktop]` File operations constrained to known directories (2026-04-09)
- [x] `[mcp]` Network egress: zero at inference. One exception — the **first run** downloads ~1.6 GB of weights from HuggingFace; pre-populate the cache and set `AI_EYES_MODEL_DIR` for a zero-egress deployment. The earlier unqualified "off by default" line was inaccurate. (2026-08-20)
- [x] `[mcp]` Stack traces never exposed — structured error results only (2026-04-09)

## B. Error Handling

- [x] `[all]` Errors are actionable `ToolError` message strings with embedded hints (missing path → check path; invalid image; GPU OOM → try `AI_EYES_DTYPE=float16` / `AI_EYES_DEVICE=cpu`); raw errors logged at DEBUG, never leaked to the caller. Deliberate message-string form (not a `{code,message,hint}` dict) — right for an LLM caller. (2026-07-07)
- [x] `[cli]` Exit codes: 0 ok · 1 user error · 2 runtime error · 3 partial success (2026-04-09)
- [x] `[cli]` No raw stack traces without `--debug` (2026-04-09)
- [x] `[mcp]` Tool errors return structured results — server never crashes on bad input (2026-04-09)
- [x] `[mcp]` State/config corruption degrades gracefully (stale data over crash) (2026-04-09)
- [x] `[desktop]` SKIP: not a desktop app
- [x] `[vscode]` SKIP: not a VS Code extension

## C. Operator Docs

- [x] `[all]` README is current: what it does, install, usage, supported platforms + runtime versions (2026-04-09)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-04-09)
- [x] `[all]` LICENSE file present and repo states support status (2026-04-09)
- [x] `[cli]` SKIP: STDIO server, no interactive CLI flags beyond entry point
- [x] `[cli|mcp|desktop]` Logging levels: configurable via `AI_EYES_LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR), stderr, `ai_eyes_mcp` logger (2026-07-07)
- [x] `[mcp]` All 7 tools documented with description + parameters (README table + reference; docstrings) (2026-07-07)
- [x] `[complex]` SKIP: no background daemons or operational modes

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists and PASSES end-to-end: imports, 7-tool registration from the single `EXPECTED_TOOL_NAMES` source, cold-status gate, wheel build, `pytest -m "not dogfood"` (51 passed). Re-dated only after a green run — it had been failing since v1.1.0 added two tools. (2026-08-20)
- [x] `[all]` Version in manifest (1.1.0) matches git tag v1.1.0 (2026-07-07)
- [x] `[all]` Dependency scanning: `pip-audit` runs in CI (advisory/non-blocking — heavy ML deps carry many non-actionable advisories) (2026-07-07)
- [x] `[all]` SKIP: private internal tool — Dependabot not added (per Actions-cost policy); dep updates handled manually
- [x] `[npm]` SKIP: not an npm package
- [x] `[pypi]` `python_requires` set (2026-04-09)
- [x] `[pypi]` Clean wheel + sdist build (2026-04-09)
- [x] `[vsix]` SKIP: not a VS Code extension
- [x] `[desktop]` SKIP: not a desktop app

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header — `docs/logo.png`, dark-card wordmark, replaced the line-art SVG (2026-08-20)
- [ ] `[all]` Translations — **SKIP justification is STALE.** The reason on file was "internal instrument (Anthropic hand-off)"; the repo is **PUBLIC** with a public description and 7 topics. Whether a public org tool gets the standard 8-language treatment is a Director call, not a default. Flagged 2026-08-20, not silently honoured and not silently overridden.
- [x] `[org]` Landing page + 5-page Starlight handbook at `site/` (blue accent, Pagefind search, local logo). The prior SKIP was justified as "internal"; the repo is PUBLIC, so the justification was stale rather than the decision being wrong. Build verified: 6 pages, search index, all handbook routes present. (2026-08-20)
- [x] `[all]` GitHub repo metadata: description + 7 topics set via `gh repo edit`; description refreshed for v1.2.0 (2026-08-20)

---

## Gate Rules

**Hard gate (A-D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."
