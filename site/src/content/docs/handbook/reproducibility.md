---
title: Reproducibility
description: The pinned model revision, why an unpinned one is a correctness bug, and running fully offline.
sidebar:
  order: 4
---

## A score is only meaningful if you know which weights produced it

The model revision is **pinned to a specific commit SHA**, passed to every load,
and **reported in every payload that contains a number**.

```
e8e487298228002f3d8a82e0cd5c8ea9c567f57f
```

Before this was pinned, the model resolved to whatever the HuggingFace default
branch pointed at **at download time**. Two installs made weeks apart could
return different scores for identical input, with no signal and no way for a
caller to tell. For a tool whose entire claim is honest measurement, that is not
a nice-to-have — it is the claim failing silently.

## Overriding it

`AI_EYES_MODEL_REVISION` is honoured **only as a different 40-character hex
commit SHA.** That is operator intent, and it is supported.

`main`, a tag, a short SHA, or an empty string is a **hard load failure with an
actionable message** — not a fallback to the pin. Falling back would reintroduce
exactly the ambiguity the pin removes: you would believe you had selected a
revision and quietly have the default.

## Verifying what you actually loaded

`eyes_status` and `eyes_selftest` both report the **resolved** revision — what
the loaded model says about itself, not the constant the code was compiled with.
If those could ever differ, the resolved one is the true answer.

```
eyes_status()
→ { "model_id": "google/siglip2-so400m-patch14-384",
    "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    "device": "cuda", "dtype": "float32", "loaded": true }
```

The pin is verified end-to-end: installed from a built wheel, outside the source
tree, the self-test reproduces the same calibration values byte-for-byte as a
development checkout on the same SHA.

## Running fully offline

All *inference* is local. The single network call in the tool's life is the
first-run model download.

To eliminate it: pre-populate the HuggingFace cache and point
`AI_EYES_MODEL_DIR` at it. The tool then makes **zero** network calls at any
point — suitable for an air-gapped or egress-audited deployment.

Be precise about this when you assess it: "no network egress" is true of
inference and false of first run. The distinction matters to exactly the people
who ask the question.

## What reproducibility does *not* cover

Pinning the weights makes the same input produce the same number **on the same
code path**. It does not make different code paths agree with each other — which
is why `image_score_batch` scores one image at a time rather than batching, and
why that decision is documented in
[Honest Measurement](../honest-measurement/) rather than buried.
