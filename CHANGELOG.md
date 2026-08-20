# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- In-memory image-embedding memo (`embed_image`, bounded LRU, default 64
  entries, `AI_EYES_EMBED_CACHE` overrides). A caller's baseline pairs were
  re-embedded on every `image_compare` / `image_rank` call — 36.5 ms per
  embedding, so a three-pair floor cost 219 ms per call to recompute a number
  that cannot have changed. Keyed on path + mtime + size, never path alone, so
  a rewritten file is re-measured rather than served stale. In-memory only:
  no disk, no sidecar, no index file. Returns a private copy, so a caller
  mutating the vector cannot poison the memo. Moves no score — `embed_image`
  is bit-identical across repeat calls on the pin.

### Fixed

- `image_compare` / `image_rank` baselines validate the TYPE of each pair, not
  only its length. `len("ab") == 2` is true, so the string `"ab"` was indexed
  by character and became two garbage paths resolved against the server's
  working directory — silently, from a plausible typo. Bytes, `[[1, 2]]`, a
  dict of two and a set of two additionally leaked a raw `TypeError` /
  `KeyError` from outside the tool's try/except, bypassing the actionable
  error shape.

### Held

- **Stacked batch forward (F-W5-ENGINE-001) remains held.** `score_batch` still
  runs a per-image loop. A stacked forward is 1.65x–1.95x faster at good chunk
  sizes but is not the same number, and — contrary to the assumption this was
  scoped on — the difference REACHES THE PAYLOAD: 4 of 11 vendored fixtures
  print a different value at batch size 8, because `display_round` keeps five
  significant digits for scores too small to survive 4-decimal rounding and
  SigLIP2 scores non-matching images at 1e-12..1e-5. Shipping it would put two
  calibrations in one server, since `image_contains` stays at batch size 1.
  The design is implementable — a fixed batch size is bit-reproducible and
  padding content provably does not affect the real images' scores — so this is
  a product decision, not an engineering gap. Evidence is executable:
  `test_stacking_divergence_is_payload_visible`.

### Internal

- Version is pinned across `pyproject.toml`, `__init__.py` and the README by a
  CI gate, plus a floor gate so agreement is not satisfiable by a placeholder.
- The tool-registration anchor is explicit (`SHIPPED_TOOL_COUNT`) and no longer
  stale: eight tools ship, while three test names still said "seven".

## [1.2.0] - 2026-08-20

Pin, honesty fields, one new ranking verb, and relative verdicts on
image-image comparison. Additive: no documented behaviour was removed.
Not 2.0.0. Not a patch.

### Added

- Model revision is pinned (`e8e487298228002f3d8a82e0cd5c8ea9c567f57f`) and
  hard-refuses branch names. Every tool payload that returns a model number
  includes `revision`.
- `truncated` on text-scoring tools and on standalone `Score` (a float
  subclass carrying `.truncated` and `.revision`).
- `image_rank` — one reference, many candidates, top-k with margins.
  Without caller baselines the ranking is a measurement (`incomplete`);
  with baselines, candidates at or below the 'these are different' floor
  are not matches (`nothing_close` when the list is empty).
- `image_compare` accepts optional `baselines` (pairs that are not a match
  in this style). Without them `incomplete: true`; with them, `separated`
  is relative to that floor. No hardcoded 0.70–0.84 band.
- `similarities_to_reference` engine primitive (embed the canon once).
- Caller-facing honesty contract on `eyes_status.scoring_guidance`.

### Changed

- `image_classify` ranks on raw scores, not 4-dp rounded values.
- `AI_EYES_MODEL_ID` is honoured by standalone `SigLIPEngine()`.
- Out-of-range `AI_EYES_DEFAULT_THRESHOLD` falls back to 0.02 with a warning.
- Logging is configured on the standalone import path (`AI_EYES_LOG_LEVEL`).

### Fixed

- First-load and eager-import errors no longer leak hub/CUDA internals.
- `verify.sh` / CI use one tool-set source and `pytest -m "not dogfood"`.

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
