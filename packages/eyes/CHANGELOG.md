# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-08-20

Pin, honesty fields, one new ranking verb, and relative verdicts on
image-image comparison. Additive: no documented behaviour was removed.

### Added

- **The model revision is pinned** (`e8e487298228002f3d8a82e0cd5c8ea9c567f57f`)
  and hard-refuses branch names, tags, and empty values — an override is
  honoured only as a different 40-character SHA. Previously the model resolved
  to whatever the hub default branch pointed at *at download time*, so two
  installs weeks apart could return different scores for identical input with
  no signal. **Every payload that returns a model number now carries
  `revision`**, so a score can name the weights that produced it.
- `truncated` on every text-scoring tool, and on standalone `Score` (a float
  subclass carrying `.truncated` and `.revision`). A query past the encoder's
  64-token capacity is scored on its prefix; the caller can now see that
  without reading stderr.
- **`image_rank`** — one reference, many candidates, top-k with margins,
  encoding the reference once. Without caller baselines the ranking is a
  measurement (`incomplete`); with them, candidates at or below the
  "these are different" floor are not matches, and `matches` is empty when
  nothing clears it.
- `image_compare` accepts optional `baselines` — pairs that are *not* a match
  in your style. Without them, `incomplete: true`; with them, `separated` is
  relative to that floor. **No hardcoded band**: measured similarity between
  six pairs of different characters in one sprite style was 0.698–0.836, and a
  cutoff drawn from that would not transfer to photos or screenshots.
- `similarities_to_reference` engine primitive (embed the canon once).
- In-memory image-embedding memo (bounded LRU, default 64, `AI_EYES_EMBED_CACHE`).
  Baselines were re-embedded on every call — 36.5 ms each, 219 ms per call for a
  three-pair floor — to recompute a number that cannot have changed. Keyed on
  path + mtime + size, never path alone, so a rewritten file is re-measured
  rather than served stale. Returns a private copy so a caller mutating the
  vector cannot poison the memo. In-memory only: no disk, no sidecar, no index.
  Moves no score.
- Caller-facing honesty contract on `eyes_status.scoring_guidance`, reciprocal
  with plain-sight's — that tool describes, this one measures.

### Changed

- `image_classify` ranks on raw scores, not 4-dp rounded values. Two labels
  within 1e-4 previously collapsed to a tie that `max()` resolved by *caller
  argument order*, so the same call with labels swapped could report a
  different `best`.
- `AI_EYES_MODEL_ID` is honoured by a standalone `SigLIPEngine()`. It was the
  only configuration constant not read at module scope, so the MCP surface
  respected it and the documented library path silently ignored it.
- Out-of-range `AI_EYES_DEFAULT_THRESHOLD` falls back to 0.02 with a warning.
  A numeric-but-invalid value such as `1.5` was previously kept, which made
  both thresholded tools reject their own default on every call.
- Logging is configured on the standalone import path, so `AI_EYES_LOG_LEVEL`
  works there — including the truncation warning it exists to surface.

### Fixed

- First-load and eager-import failures no longer leak hub or CUDA internals;
  an unresolvable model id exits cleanly instead of dumping a traceback.
- `image_compare` / `image_rank` baselines validate the **type** of each pair,
  not only its length. `len("ab") == 2` is true, so `"ab"` was indexed by
  character into two garbage paths — silently, from a plausible typo. Bytes,
  `[[1, 2]]`, a dict of two and a set of two additionally leaked a raw
  `TypeError` / `KeyError` from outside the tool's try/except.
- Display rounding can no longer contradict the verdict printed beside it —
  `margin: 0.0` with `present: true`, or `{present: true, score: 0.02,
  threshold: 0.02}`, are both gone. A measured non-zero never prints as `0`.
- `verify.sh` and CI share one tool-set source and select by marker; the
  script had asserted a five-tool set since v1.1.0 shipped seven.

### Held

- **Stacked batch forward is held, on evidence.** `score_batch` still runs a
  per-image loop. A stacked forward is 1.65×–1.95× faster at good chunk sizes
  (and *slower* at 100), but it is not the same number — and the difference
  **reaches the payload**: 4 of 11 fixtures print a different value at batch
  size 8, because `display_round` keeps five significant digits for scores too
  small for 4-dp rounding and SigLIP2 scores non-matching images at 1e-12–1e-5.
  Shipping it would put two calibrations in one server, since `image_contains`
  stays at batch size 1 — the same image, query and revision would print
  different numbers depending on which verb you called. The design is
  implementable; declining it is a product decision, and the evidence is
  executable: `test_stacking_divergence_is_payload_visible`.

### Internal

- The version is pinned across `pyproject.toml`, `__init__.py` and the README
  by a CI gate, with a floor check so agreement cannot be satisfied by a
  placeholder.
- The tool-registration anchor is explicit (`SHIPPED_TOOL_COUNT`) and no longer
  stale — eight tools ship, while three test names still said "seven".

## [1.1.0] - 2026-07-07

Hardening + capability release (internal). Two new tools, a HIGH-severity
compatibility fix, and a test suite taken from "green theater" to real.

### Added

- `image_verify` tool — honest **relative** verdict: ranks a target hypothesis
  against caller-supplied alternatives and returns a decision + margin +
  confidence band, robust to SigLIP's query-phrasing sensitivity (prefer this
  over thresholding a raw `image_contains` score).
- `eyes_selftest` tool — self-proving calibration check: runs decisive known
  orderings on bundled reference images (shipped as package data) to confirm the
  install loaded correctly and is calibrated.
- `AI_EYES_LOG_LEVEL` — configurable stderr log verbosity.
- `AI_EYES_EAGER_LOAD` — load the model at startup so a broken model/cache fails
  fast instead of on the first tool call.

### Fixed

- **[HIGH] transformers 5.x compatibility** — `get_image_features` returns an
  output object (not a bare tensor) on transformers 5.x, which crashed
  `image_compare` / `embed_image` on every current install. Fixed
  version-agnostically (tensor on 4.x, `.pooler_output` on 5.x).
- **Long-query crash** — a query over the text encoder's 64-token limit raised
  in the forward pass; now truncated (with a warning) instead of crashing.
- **Thread-safety** — forward passes are serialized by a per-engine lock
  (FastMCP runs sync tools in a worker threadpool, so concurrent calls shared
  the model). The prior "thread-safe via torch.no_grad" claim was corrected.
- Empty-list guards (`score_multi([])`, `score_batch([])`), NaN-safe threshold
  validation, and sanitized batch error messages (no raw path/stack leak).

### Changed

- Consistent, actionable tool error hints via a single mapper (missing path,
  invalid image, GPU OOM → "try AI_EYES_DTYPE=float16 / AI_EYES_DEVICE=cpu").
- Query-phrasing guidance made prominent (README, tool docstrings,
  `eyes_status.scoring_guidance`).
- Test suite de-theatered: was 11 pass / 92 skip on any non-original rig
  (hardcoded absolute fixture/cache paths); now **132 passed / 0 skipped**, real
  GPU inference, with own-IP fixtures plus clean-room-install, MCP-protocol e2e,
  and determinism guards.

## [1.0.0] - 2026-04-09

### Added

- SigLIP2 SO400M vision engine with lazy model loading
- `image_contains` tool — sigmoid score for "does image contain X?"
- `image_classify` tool — score image against N candidate labels
- `image_compare` tool — cosine similarity between two images
- `image_score_batch` tool — score N images against one query
- `eyes_status` tool — health check (model, device, VRAM)
- FastMCP v3 server with STDIO transport
- Configurable model, device, cache dir, and threshold via environment variables

[1.2.0]: https://github.com/mcp-tool-shop-org/ai-eyes-mcp/releases/tag/v1.2.0
[1.1.0]: https://github.com/mcp-tool-shop-org/ai-eyes-mcp/releases/tag/v1.1.0
[1.0.0]: https://github.com/mcp-tool-shop-org/ai-eyes-mcp/releases/tag/v1.0.0
