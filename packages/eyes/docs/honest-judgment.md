# Honest judgment — getting reliable verdicts

SigLIP2 is honest but **query‑phrasing sensitive**: the *absolute* sigmoid score for the same image swings widely with wording. A style‑matched phrase can score 10–100× higher than a generic one, and a good threshold for one image style does NOT transfer to another.

This is not a bug — it's how the model calibrates. The fix is to use it *relatively*.

## Prefer relative ranking

| Need | Use | Why |
|---|---|---|
| "Is X the best description here?" | `image_verify` | Decision + margin + confidence, relative to alternatives |
| "Which of these labels fits?" | `image_classify` | Ranks candidates; the argmax is robust to absolute magnitude |
| "How similar are these two images?" | `image_compare` | Cosine of embeddings — no query phrasing at all |
| A raw number you'll threshold yourself | `image_contains` | Only when you control both the query wording and the image style |

## The confidence band (`image_verify`)

`image_verify` derives `confidence` from the **margin magnitude** between the target and the best alternative:

- **high** — |margin| ≥ 0.3 (decisive — present *or* absent)
- **moderate** — 0.1 ≤ |margin| < 0.3
- **low — inconclusive** — |margin| < 0.1 (target and best alternative are close; treat as unknown, not a confident answer)

A near‑zero score is a **valid answer** ("not confidently present"), not an error.

## Worked example

On a knight sprite:

- `image_contains("a knight with a sword and shield")` → **0.885** (high — but the number depends on the phrasing).
- `image_verify(target="a knight with a sword and shield", alternatives=["a goblin cook", "a bard"])` → **present, margin 0.885, high** — the *decision* is robust regardless of the absolute number.

On a goblin cook, the same `image_verify(target="a knight with a sword", alternatives=["a goblin cook"])` → **absent** (the cook wins its own label by 0.9996) — an honest negative with high confidence.
