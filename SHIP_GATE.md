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
- [x] `[mcp]` Network egress off by default (2026-04-09)
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

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-04-09)
- [x] `[all]` Version in manifest (1.1.0) matches git tag v1.1.0 (2026-07-07)
- [x] `[all]` Dependency scanning: `pip-audit` runs in CI (advisory/non-blocking — heavy ML deps carry many non-actionable advisories) (2026-07-07)
- [x] `[all]` SKIP: private internal tool — Dependabot not added (per Actions-cost policy); dep updates handled manually
- [x] `[npm]` SKIP: not an npm package
- [x] `[pypi]` `python_requires` set (2026-04-09)
- [x] `[pypi]` Clean wheel + sdist build (2026-04-09)
- [x] `[vsix]` SKIP: not a VS Code extension
- [x] `[desktop]` SKIP: not a desktop app

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (committed SVG, theme-aware) (2026-07-07)
- [x] `[all]` SKIP: internal instrument (Anthropic hand-off) — no public translations
- [x] `[org]` SKIP: internal — no public landing page (docs/ handbook instead)
- [x] `[all]` GitHub repo metadata: description + topics set via `gh repo edit` (2026-07-07)

---

## Gate Rules

**Hard gate (A-D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."
