---
title: Honest Measurement
description: truncated, incomplete, confidence — the fields that exist so the tool can decline to overstate.
sidebar:
  order: 3
---

Most of this tool's design budget went into **not overstating**. These are the
mechanisms, and the reasoning behind each.

## `truncated`

The text encoder holds **64 tokens**. A longer query is silently truncated by
the tokenizer, and the score then reflects only the prefix.

The character cap does not protect you: `'cat dog fox ' * 39` is 468 characters —
under the 500-character limit — and tokenizes to **119 ids**. So a query can pass
every length check and still be scored on roughly half of itself.

Every text-scoring tool returns `truncated`. When it is `true`, **treat the score
as incomplete rather than as a number.** The warning also goes to the log, but a
log line is not the result channel — an MCP caller would never see it.

## `incomplete`

Returned by `image_compare` and `image_rank` when you did not supply
`baselines`. It means: *this is a measurement, not a verdict.*

The instrument knows the number. What it does not know is what counts as "close"
in **your** image domain, and that floor genuinely does not transfer. Rather than
picking one and being quietly wrong outside the style it was tuned on, it reports
the measurement and flags that you have not given it enough to judge.

## `confidence`

`image_verify` returns a band describing the **measured gap** — `high`,
`moderate`, or `low — inconclusive`. It is not a restatement of the score and it
is not invented certainty. A small margin produces `low — inconclusive` even when
the target technically ranked first, because that is what the measurement
supports.

## Numbers never contradict the verdict beside them

Displayed values are rounded for readability, but **never far enough to disagree
with the decision printed next to them**. Rounding escalates precision until the
displayed numbers still imply the verdict.

Concretely, these payloads are impossible:

- `{ "present": true, "margin": 0.0 }`
- `{ "present": true, "score": 0.02, "threshold": 0.02 }`

And a measured non-zero never prints as `0` — a sigmoid of `4.38e-08` renders as
`4.3813e-08`, not as a calibrated-looking zero. The self-test's negative side is
the clearest place to see this.

## What was declined, and why it matters here

`image_score_batch` scores one image per forward pass. Stacking them is
**1.65–1.95×** faster at good chunk sizes (and 0.47× — *slower* — at 100), which
is a real win that was measured and then turned down.

The reason: **the score depends on the batch size.** The same image, same query,
same pinned revision returns a different value at batch size 1, 2, 4 and 8 —
because batched matrix multiplication reduces in a different order. That is
arithmetic, not a bug, and no batch size except 1 reproduces the single-image
score.

The difference is small, but it is **not invisible**: because tiny scores print
at five significant digits rather than collapsing to zero, 4 of 11 test fixtures
print a *different number* at batch size 8. And since `image_contains` stays at
batch size 1, shipping it would mean `image_contains(x)` and `image_score_batch`
printing different numbers for the same image, same query, same revision.

Two calibrations in one instrument. Naming the batch size in the payload would
*explain* that; it would not reconcile it. So the loop stays.

The evidence is executable: `test_stacking_divergence_is_payload_visible`.
