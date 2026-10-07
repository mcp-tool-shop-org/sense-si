---
title: ai-eyes Handbook
description: A grounded visual evaluator for LLM callers — what it measures, what it refuses to say, and why.
sidebar:
  order: 0
---

`ai-eyes` is an MCP server that wraps **SigLIP2** as a measurement instrument.

## The problem it exists for

When a model needs to check something about an image — *does this sprite have a
sword?*, *is there a login button?* — generative vision models answer
confidently and are sometimes simply wrong. They are completing a narrative, not
reporting an observation. In the run that motivated this tool, LLaVA 13B reported
"the character is holding a greatsword" on images with no weapon, at 0.90
confidence, on every crop.

SigLIP2 is **discriminative**. It does not generate text. It measures similarity
between an image and a description you supply and returns a calibrated sigmoid
score. It cannot narrate, so it cannot narrate wrongly.

## The constraint that shapes everything else

> It measures one image-text pair. It does not describe the image. It never
> states something it did not measure.

That is not a slogan — it is the rule every design decision here was checked
against, and several plausible features were declined by it:

- A score is meaningless unless you know which weights produced it, so the model
  revision is **pinned** and **reported in every payload that carries a number**.
- A query longer than the encoder's 64-token capacity is scored on its *prefix*,
  so the payload says `truncated: true` instead of presenting part as whole.
- Similarity thresholds do not transfer between image styles, so the tools that
  need one **take it from you** and return `incomplete` when you do not supply it.
- A batched-forward optimisation worth up to 1.95× was **declined** because it
  made a score depend on how many unrelated images shared the request.

## Where to go next

- **[Getting Started](./getting-started/)** — install, register with an MCP
  client, make a first call.
- **[Tools](./tools/)** — all eight, with the questions each one actually answers.
- **[Honest Measurement](./honest-measurement/)** — `truncated`, `incomplete`,
  `confidence`, and when the instrument refuses.
- **[Reproducibility](./reproducibility/)** — the pinned revision, and running
  fully offline.

## When to use something else

ai-eyes answers *"is this claim about the pixels true?"* It scores a hypothesis
you supply. It cannot tell you what is in an image you have not described.

For captions, descriptions, or text read off an image, use
[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight) — a generative
instrument whose output can hallucinate by construction. The two are a
deliberate pair: **plain-sight describes, ai-eyes measures.** Bring anything
load-bearing from one back to the other.
