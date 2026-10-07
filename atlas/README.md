# sense-si: how it works

Mapped at 2026-10-07 from commit e4e89f5 by Atlas 1.24.0.

## What this is

7 parts, mostly Python (24 files), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and shell (1). Work enters through 3 doors; the busiest is CI, which reaches 4 parts. People run ai-eyes and ai-eyes-mcp.

## What changed since 2026-10-07 (503d57a)

- ears now imports tools, which closes the cycle ears → tools → ears.
- tools now imports decisions, which closes the cycle tools → decisions → tools.
- tools now imports ears, which closes the cycle tools → ears → tools.
- docs/calibration.md is now written by tools/phrase_calibration.py.
- docs/calibration/corp.svg is now written by tools/phrase_calibration.py.
- docs/calibration/phrases.json is now written by tools/phrase_calibration.py.
- And 4 more new writers and readers of places.
- docs was authored and is now mixed.
- 5 files added and 11 changed content, across 5 parts.

## What comes in

1. **CI.** On a pull request; on a push to main touching 5 paths; or by hand. Runs tools/coverage_bar.py, packages/decisions/tests/, packages/ears/tests/ and 8 more.
2. **ai-eyes** (a command people run). Runs packages/eyes/src/ai_eyes_mcp/server.py.
3. **ai-eyes-mcp** (a command people run). Runs packages/eyes/src/ai_eyes_mcp/server.py.

## What happens through CI

1. The workflow runs packages/decisions/tests/ in decisions, packages/ears/tests/ in ears, packages/eyes/tests/ in eyes and tools/coverage_bar.py in tools.
2. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**ai-eyes** (a command people run) runs packages/eyes/src/ai_eyes_mcp/server.py.

**ai-eyes-mcp** (a command people run) runs packages/eyes/src/ai_eyes_mcp/server.py.

## What breaks what

- **decisions** is imported by 2 parts (ears, tools) and sits on the path of 1 door.
- **ears** is imported by 1 part (tools) and sits on the path of 1 door.
- **eyes** is imported by no other part and sits on the path of 3 doors.
- **tools** is imported only from tests, by 2 parts (decisions, ears), and sits on the path of 1 door.

## What tends to change together

- **src/ai_eyes_mcp/server.py** and **tests/test_server_ci.py** changed together in 7 of 14 commits.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 2 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Every tracked place code writes here is edited by people too; see Hand-authored.

## Hand-authored

People write .github/ and the repository root. Nothing in this repository writes to them.

- **docs/calibration.md** is written by tools/phrase_calibration.py from inputs this repository does not keep, and by people.

## Where to start

packages/eyes/src/ai_eyes_mcp/server.py → packages/eyes/src/ai_eyes_mcp/engine.py

Read those in order to follow one run of ai-eyes end to end. This path follows ai-eyes (a command people run) from its entry, since CI runs only tests and scripts that import no code here.

## What this map cannot see

- 4 reads use paths built at run time and are not named here.
- 3 writes go to places this repository does not track, so they are not listed as generated.
- 1 write and 3 reads go to a path their caller passes, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
