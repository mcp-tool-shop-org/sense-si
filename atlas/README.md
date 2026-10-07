# sense-si: how it works

Mapped at 2026-10-07 from commit 5ad4c96 by Atlas 1.24.0.

## What this is

6 parts, mostly Python (10 files). Work enters through 1 door; the busiest is CI, which reaches 3 parts.

## What changed since 2026-10-07 (14b3ef7)

Nothing structural changed since 2026-10-07; 1 file changed content.

## What comes in

1. **CI.** On a pull request; on a push to main touching 5 paths; or by hand. Runs tools/coverage_bar.py, packages/decisions/tests/ and packages/ears/tests/.

## What happens through CI

1. The workflow runs packages/decisions/tests/ in decisions, packages/ears/tests/ in ears and tools/coverage_bar.py in tools.
2. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## What breaks what

- **decisions** is imported by 1 part (ears) and sits on the path of 1 door.
- **tools** is imported only from tests, by 1 part (decisions), and sits on the path of 1 door.

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

- 1 read goes to a path its caller passes, not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
