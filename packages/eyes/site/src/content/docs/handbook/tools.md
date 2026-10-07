---
title: Tools
description: All eight MCP tools, and the question each one actually answers.
sidebar:
  order: 2
---

Eight tools. The useful way to choose between them is by the **question you can
honestly ask**, not by the shape of the return value.

| You want to know | Use | Because |
|---|---|---|
| Does this image match a description I supply? | `image_contains` | Raw sigmoid. You must threshold it. |
| Which of my labels fits best? | `image_classify` | Relative ranking, insensitive to absolute magnitude. |
| Is my hypothesis better than these alternatives? | `image_verify` | **Relative verdict.** Usually what you actually wanted. |
| How similar are these two images? | `image_compare` | Cosine similarity, with a floor you supply. |
| Which of these N look like my reference? | `image_rank` | Encodes the reference once. Can return nothing. |
| Score many images against one query | `image_score_batch` | Encodes the text once. Max 100. |
| Is the instrument working? | `eyes_selftest` | Known orderings on bundled images. |
| What is loaded, on what? | `eyes_status` | Does not trigger a load. |

## Prefer relative over absolute

`image_contains` returns a raw score you must compare against a threshold — and
**absolute scores are query-phrasing sensitive**. The same image can score
10–100× higher for a style-matched phrase than for a generic one, and a
threshold tuned on sprites will be wrong on photographs.

So: reach for `image_contains` only when you control both the wording and the
image style. Otherwise use `image_verify` (rank a target against contrasts you
supply) or `image_classify` (rank your labels against each other). Both are
insensitive to the magnitude problem because they compare like with like.

## `image_verify`

```
image_verify(image_path, target, alternatives)
```

`alternatives` is **required, ≥1** — this verb is relative by construction and
there is no absolute mode. Returns the decision, the `margin`, and a
`confidence` band that describes the measured gap rather than asserting
certainty: `high`, `moderate`, or `low — inconclusive`.

## `image_compare` and `image_rank`

Both accept optional `baselines` — pairs of images that are **not** a match in
your style.

Without baselines you get a number and `incomplete: true`. That is deliberate.
Measured across six pairs of *different* characters in one sprite style, cosine
similarity ranged **0.698–0.836**, against 1.0 for an image against itself. A
cutoff drawn from that range would be wrong for photographs, screenshots or
3D renders — so the tool declines to invent one and asks you for the contrast,
exactly as `image_verify` asks for alternatives.

With baselines, `image_rank` returns only candidates above your floor. If none
clear it, `matches` is **empty**. A ranking verb that always returns *k* results
is a confident answer in a regime with no signal.

## `image_score_batch`

Encodes the query once and scores each image. Per-item failures are collected
into `error_details` rather than failing the call.

It scores one image at a time deliberately. A stacked forward pass is 1.65–1.95×
faster at good chunk sizes — and slower at 100 — but the score would then depend
on how many unrelated images shared the request. See
[Honest Measurement](../honest-measurement/).
