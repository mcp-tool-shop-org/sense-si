# The 7 tools

All tools take absolute image paths and return structured JSON. Scores are independent **sigmoid** probabilities (0–1), NOT softmax — multiple queries can score high on the same image.

## `image_contains(image_path, query, threshold=0.02)`

Raw sigmoid score for "does the image match this description?" Returns `{present, score, threshold, query, elapsed_ms}`.

> ⚠ The absolute score is query‑phrasing sensitive — see [Honest judgment](honest-judgment.md). For a robust yes/no, prefer `image_verify`.

## `image_classify(image_path, labels)`

Scores the image against N candidate labels (≤20), sorted high→low. Returns `{scores, best, best_score, elapsed_ms}`. Relative ranking — the reliable path for "which of these is it?".

## `image_compare(image_a, image_b)`

Cosine similarity of the two images' SigLIP2 embeddings, in [-1, 1]. Returns `{similarity, image_a, image_b, elapsed_ms}`. Use for pose/render matching, duplicate detection.

## `image_score_batch(image_paths, query, threshold=0.02)`

Scores many images (≤100) against one query; the text is encoded once. Returns per‑image results + `best_path`/`best_score`. A bad path in the batch is captured in `error_details` (sanitized message), not a crash.

## `image_verify(image_path, target, alternatives)` ⭐

**The honest verdict.** Ranks `target` against caller‑supplied `alternatives` (required, ≥1) and returns a DECISION + margin + confidence — robust to phrasing because it's *relative*. Returns `{present, target, target_score, best_alternative, best_alternative_score, margin, confidence}`.

```
image_verify("knight.png", "a knight with a sword", ["a cook", "a merchant"])
→ { present: true, margin: 0.88, confidence: "high", ... }
```

## `eyes_selftest()`

Runs decisive known orderings on bundled reference images to prove the install loaded + is calibrated. Returns `{passed, checks, model_id, device, versions}`. See [Install & validate](install-and-validate.md).

## `eyes_status()`

Health check — model, device, dtype, loaded state, versions, `scoring_guidance`. Does NOT trigger a model load. When unloaded, includes a `note` about first‑call latency.
