# ai-eyes-mcp: how it works

Mapped at 2026-09-30 from commit 1199865 by Atlas 1.24.0.

## What this is

6 parts, mostly Python (11 files), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and shell (1). Work enters through 3 doors; CI, Deploy site to GitHub Pages and ai-eyes-mcp each reach 1 part, and CI is followed because a pull request goes through it. It deploys a site to GitHub Pages. People run ai-eyes-mcp.

## What changed since 2026-09-25 (f8d83a6)

- CI's pull request trigger now also names `codecov.yml`.
- CI's push trigger now also names `codecov.yml`.
- CI now also runs src/ai_eyes_mcp/__init__.py, src/ai_eyes_mcp/engine.py and src/ai_eyes_mcp/server.py.
- And 1 more change to a door.
- 1 file added and 2 changed content, across 2 parts.

## What comes in

1. **CI.** On a pull request touching 7 paths; on a push touching 7 paths; or by hand. Runs src/ai_eyes_mcp/__init__.py, src/ai_eyes_mcp/engine.py and src/ai_eyes_mcp/server.py.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **ai-eyes-mcp** (a command people run). Runs src/ai_eyes_mcp/server.py.

## What happens through CI

1. The workflow runs src/ai_eyes_mcp/__init__.py, src/ai_eyes_mcp/engine.py and src/ai_eyes_mcp/server.py in src.
2. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**ai-eyes-mcp** (a command people run) runs src/ai_eyes_mcp/server.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since 3 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

6 test files run in no workflow: tests/test_ci_gates.py, tests/test_edge_cases.py, tests/test_engine_ci.py and 3 more.

verify.sh runs in no workflow.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, docs/, the repository root and site/. Nothing in this repository writes to them.

## Where to start

.github/workflows/ci.yml → src/ai_eyes_mcp/server.py → src/ai_eyes_mcp/engine.py

Read those in order to follow one pull request end to end.

## What this map cannot see

- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
