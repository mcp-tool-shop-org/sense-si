# ai-eyes-mcp

**Version:** 1.0.0

Grounded visual evaluator MCP server. Gives Claude honest image judgment via SigLIP2.

## The Problem

When Claude needs to verify what's in an image — "does this sprite have a sword?", "is there a login button?" — generative VLMs (LLaVA, GPT-4V) hallucinate confident answers. They complete narratives, not observations. LLaVA 13B reported "the character is holding a greatsword" on images where no weapon existed, at 0.90 confidence, on every single crop.

## The Solution

SigLIP2 is a discriminative vision model. It doesn't generate text — it measures similarity between an image and a text description, returning a calibrated sigmoid score. When the weapon is present, the score is 10-100x higher than when it's absent. When it can't tell, the score is low. It doesn't hallucinate.

This MCP server wraps SigLIP2 as tools that any Claude workflow can call.

## Tools

| Tool | What it does |
|------|-------------|
| `image_contains` | "Does this image contain X?" → sigmoid score |
| `image_classify` | Score image against N candidate labels |
| `image_compare` | Cosine similarity between two images |
| `image_score_batch` | Score N images against one query |
| `image_verify` | Honest RELATIVE verdict: target vs alternatives → decision + margin + confidence |
| `eyes_selftest` | Self-test on bundled reference images (proves install + calibration) |
| `eyes_status` | Health check: model, device, loaded state |

## Quick Start

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

Or run as a module: `python -m ai_eyes_mcp`

### Claude Code config

```json
{
  "mcpServers": {
    "ai-eyes": {
      "command": "ai-eyes-mcp",
      "env": {
        "AI_EYES_MODEL_DIR": "/path/to/model/cache"
      }
    }
  }
}
```

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | HuggingFace model |
| `AI_EYES_MODEL_DIR` | HF default cache | Model cache directory |
| `AI_EYES_DEVICE` | `auto` (cuda if available, else cpu) | torch device |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Default threshold for `image_contains` |
| `AI_EYES_LOG_LEVEL` | `WARNING` | Log verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | unset | If truthy, load the model at startup so a broken model/cache fails fast (not on the first tool call) |
| `AI_EYES_DTYPE` | full precision | `float16` / `bfloat16` to halve VRAM |

**Logging:** The server logs under the `ai_eyes_mcp` logger to **stderr** (stdout is the MCP protocol channel). Set the level with `AI_EYES_LOG_LEVEL` (above), or attach your own handlers to `logging.getLogger("ai_eyes_mcp")`.

**First call:** the model loads lazily — the **first** image tool call downloads/loads SigLIP2 (~10–20s on GPU; longer on the first-ever download), and subsequent calls are ~100ms. Set `AI_EYES_EAGER_LOAD=1` to load at server start instead, or call `eyes_status` (which reports `loaded` without triggering a load).

## How Scores Work

SigLIP2 uses **sigmoid** scoring, not softmax. Each image-text pair gets an independent probability (0-1):

- **High score** (>0.1): Strong visual match — the described object is likely present
- **Low score** (<0.01): No match — the object is not visible
- **Mid score** (0.01-0.1): Ambiguous — may need human review

Scores are NOT relative. Multiple queries can score high on the same image (e.g., an image with both a sword and a shield).

### ⚠ Query phrasing matters — prefer `image_classify` for robust decisions

SigLIP2 sigmoid scores are **query-phrasing sensitive**: the absolute score for the *same* image swings widely with wording (a style-matched phrase can score 10–100× higher than a generic one). A fixed `threshold` therefore needs query engineering per use case, and thresholds do **not** transfer across image styles.

For robust yes/no decisions across varied inputs, prefer **`image_classify`** — it *ranks* candidate labels against each other and is insensitive to absolute score magnitude. Reach for `image_contains` with a tuned threshold only when you control both the query wording and the image style. `eyes_status` echoes this in its `scoring_guidance` field.

The default threshold (`0.02`) is a permissive floor, not a universal cutoff — tune it for your queries and image style, or use `image_classify`.

## Architecture

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` is the core — it owns model loading, device selection, and all
inference logic. `server.py` never touches torch directly; it delegates
everything to the engine. This means you can `from ai_eyes_mcp.engine import
SigLIPEngine` and use it in any Python script without pulling in FastMCP.

The 5-to-8 direction mapping used by the sprite pipeline is NOT this tool's
concern. ai-eyes-mcp evaluates images; direction mapping lives in the
downstream consumer (Sprite Foundry / Game Foundry OS).

## Tool Reference

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

Check if an image contains something described by the query. Returns an independent sigmoid score (0-1).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | yes | Absolute path to image file |
| `query` | string | yes | What to look for (e.g., "a person holding a sword") |
| `threshold` | float | no | Score threshold for present verdict (default 0.02) |

Returns: `{present: bool, score: float, threshold: float, query: string}`

### `image_classify`

```
image_classify(image_path, labels)
```

Score an image against multiple candidate labels. Returns independent sigmoid scores — NOT softmax.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | yes | Absolute path to image file |
| `labels` | string[] | yes | Candidate labels to score (max 20) |

Returns: `{scores: {label: float}, best: string, best_score: float}`

### `image_compare`

```
image_compare(image_a, image_b)
```

Compute visual similarity between two images using cosine similarity of SigLIP2 embeddings.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_a` | string | yes | Absolute path to first image |
| `image_b` | string | yes | Absolute path to second image |

Returns: `{similarity: float, image_a: string, image_b: string}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

Score multiple images against a single query. Max 100 images per call.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_paths` | string[] | yes | List of absolute image paths |
| `query` | string | yes | What to look for |
| `threshold` | float | no | Score threshold (default 0.02) |

Returns: `{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}]}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

Honest **relative** verdict — ranks `target` against caller-supplied `alternatives` (required, ≥1) and returns a decision + margin + confidence. Robust to SigLIP's query-phrasing sensitivity because it's relative, not an absolute threshold. For a raw score use `image_contains`; for full ranking use `image_classify`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | yes | Absolute path to the image |
| `target` | string | yes | The hypothesis to verify |
| `alternatives` | string[] | yes | Contrast alternatives to rank against (≥1) |

Returns: `{present, target, target_score, best_alternative, best_alternative_score, margin, confidence}` — `confidence` is `high` / `moderate` / `low — inconclusive`, describing the measured gap.

### `eyes_status`

```
eyes_status()
```

Check server status. Does not trigger model loading.

Returns: `{model_id, device, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

### `eyes_selftest`

```
eyes_selftest()
```

Runs the model on a few bundled reference images and confirms the expected orderings hold — proves the install loaded correctly and SigLIP2 is calibrated. Loads the model if not already loaded.

Returns: `{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

When `loaded` is true, also returns `parameters` (e.g., '400M') and `vram_mb` (CUDA only).

## Security and Trust

This tool operates **locally only**.

- **Data touched:** Local image files (read-only), HuggingFace model cache (downloaded once)
- **No network egress** at runtime — model downloads happen once on first use, then all inference is local
- **No secrets handling** — does not read, store, or transmit credentials or API keys
- **No telemetry** — nothing is collected or sent
- **No file mutation** — image files are opened read-only, never modified
- **No dangerous actions** — no delete, kill, or restart operations
- **Structured errors only** — stack traces never exposed to clients

## Requirements

- Python >= 3.10
- CUDA GPU recommended (~800MB VRAM at FP16)
- CPU fallback available (slower, ~10x)
- Model downloads ~1.6GB on first use

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run CI-safe tests (no model required)
pytest tests/test_edge_cases.py -v

# Run all tests (requires SigLIP2 model + GPU)
pytest tests/ -v

# Full verify: imports, tools, build, edge-case tests
bash verify.sh
```

## License

MIT

---

Built by [MCP Tool Shop](https://mcp-tool-shop.github.io/)
