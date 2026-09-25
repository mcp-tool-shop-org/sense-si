# ai-eyes-mcp: how it works

Mapped at 2026-09-25 from commit f8d83a6.

## What this is

6 parts, mostly Python (11 files), TypeScript (2) and JavaScript (1). Work enters through 3 doors; the busiest is CI, which reaches 2 parts. People run ai-eyes-mcp.

## What changed since the last map

This is the first map.

## What comes in

1. **CI.** On a pull request touching 6 paths; on a push touching 6 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **ai-eyes-mcp** (a command people run). Runs src/ai_eyes_mcp/server.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (3 files).

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

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, docs/, the repository root and site/. Nothing in this repository writes to them.

## Where to start

src/ai_eyes_mcp/server.py

Read those in order to follow one run of ai-eyes-mcp end to end. This path follows ai-eyes-mcp (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- Statistics confidence is low: fewer than 20 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
