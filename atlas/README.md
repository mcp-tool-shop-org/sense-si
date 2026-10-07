# sense-si: how it works

Mapped at 2026-10-07 from commit e2b6139 by Atlas 1.24.0.

## What this is

5 parts, mostly Python (8 files). Work enters through 1 door; the busiest is CI, which reaches 2 parts.

## What changed since 2026-10-07 (52198bb)

- CI's push trigger now also names `main`.
- 1 file changed content, across 1 part.

## What comes in

1. **CI.** On a pull request; on a push to main touching 3 paths; or by hand. Runs packages/decisions/tests/ and packages/ears/tests/.

## What happens through CI

1. The workflow runs packages/decisions/tests/ in decisions and packages/ears/tests/ in ears.

## Who reads the results

CI writes nothing this map can see.

## What breaks what

- **decisions** is imported by 1 part (ears) and sits on the path of 1 door.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, docs/ and the repository root. Nothing in this repository writes to them.

## Where to start

.github/workflows/ci.yml → packages/decisions/src/decisions/__init__.py → packages/decisions/src/decisions/client.py

Read those in order to follow one pull request end to end.

## What this map cannot see

- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
