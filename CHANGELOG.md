# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[1.1.0]: https://github.com/mcp-tool-shop-org/ai-eyes-mcp/releases/tag/v1.1.0
[1.0.0]: https://github.com/mcp-tool-shop-org/ai-eyes-mcp/releases/tag/v1.0.0
