# Install & validate

## Install

```
pip install <ai_eyes_mcp wheel>     # or, from a checkout:  pip install -e .
```

Requires Python ≥3.10 and the declared deps (`torch`, `transformers`, `fastmcp`, `pillow`, `numpy`). The SigLIP2 model (`google/siglip2-so400m-patch14-384`, ~1.6 GB) downloads on first use to the HuggingFace cache; set `AI_EYES_MODEL_DIR` / `HF_HOME` to point at an existing cache.

## Run

```
ai-eyes-mcp            # installed entry point
python -m ai_eyes_mcp  # equivalent
```

It's a STDIO MCP server. Add it to your MCP client config with `command` = the venv's python and `args` = `["-m", "ai_eyes_mcp"]`.

## Validate the install — `eyes_selftest`

An instrument should prove itself. Call `eyes_selftest()` (no args): it runs the model on **bundled reference images** (shipped as package data, so this works from a `pip install`, not just a source checkout) and confirms decisive known orderings:

- an armed knight scores far higher for a weapon query than an unarmed cook,
- a cheetah photo scores higher for "a cheetah" than "a bus",
- an image is more similar to itself than to a different one.

```json
{ "passed": true,
  "checks": [ { "name": "armed_vs_unarmed", "ok": true, "measured_a": 0.885, "measured_b": 0.0 }, ... ],
  "model_id": "google/siglip2-so400m-patch14-384", "device": "cuda", ... }
```

`passed: true` means the weights loaded correctly and SigLIP2 is calibrated as expected — the one-call smoke test for a receiving operator.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `AI_EYES_MODEL_DIR` | HF default | Model cache (hub) directory |
| `AI_EYES_DEVICE` | cuda if available, else cpu | torch device |
| `AI_EYES_DTYPE` | full precision | `float16` / `bfloat16` to halve VRAM |
| `AI_EYES_LOG_LEVEL` | WARNING | stderr log verbosity |
| `AI_EYES_EAGER_LOAD` | off | load at startup (fail fast on a broken cache) |
| `AI_EYES_DEFAULT_THRESHOLD` | 0.02 | default `image_contains` threshold |
