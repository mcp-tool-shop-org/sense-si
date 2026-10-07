---
title: Getting Started
description: Install ai-eyes, register it with an MCP client, and make a first measurement.
sidebar:
  order: 1
---

## Requirements

- Python **3.10+**
- A CUDA GPU is recommended. Measured VRAM: **~4.3 GB** at the default
  `float32`, ~2.2 GB with `AI_EYES_DTYPE=float16`.
- CPU works and is roughly 10× slower.
- The model downloads **~1.6 GB** on first use.

The model is 1136M parameters. "SO400M" in the model id names the vision tower,
not the assembled model — a figure worth having right when you size a card.

## Install

```bash
git clone https://github.com/mcp-tool-shop-org/ai-eyes-mcp
cd ai-eyes-mcp
pip install -e .
```

## Register with an MCP client

```json
{
  "mcpServers": {
    "ai-eyes": {
      "command": "ai-eyes-mcp",
      "env": { "AI_EYES_MODEL_DIR": "/path/to/model/cache" }
    }
  }
}
```

Or run it directly as a module: `python -m ai_eyes_mcp`.

## Prove the install before you trust a number

```
eyes_selftest()
```

This runs known orderings on bundled reference images and reports the model
revision it used. It is the fastest way to distinguish "the tool is working" from
"the tool is returning plausible numbers from the wrong weights."

A healthy result looks like this — note that the negative side reports at full
resolution rather than collapsing to `0`:

```json
{
  "passed": true,
  "checks": [
    { "name": "armed_vs_unarmed", "measured_a": 0.921, "measured_b": 4.3813e-08, "ok": true }
  ],
  "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
}
```

## First measurement

```
image_contains("sprite.png", "a knight with a sword")
```

```json
{ "present": true, "score": 0.6847, "threshold": 0.02,
  "truncated": false, "revision": "e8e4872…" }
```

## First-call cost

The model loads lazily, so the **first** image call pays ~10–20s on GPU and
subsequent calls are ~100ms. Set `AI_EYES_EAGER_LOAD=1` to pay it at server
start instead, which also surfaces a broken cache immediately rather than on
first use.

`eyes_status` reports state without loading the model — but it is **not free on
a cold server**: the first call pays a one-off library import (measured ~10s,
~2ms once warm).
