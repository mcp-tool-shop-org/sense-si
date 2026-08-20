<p align="center">
  <img src="logo.png" alt="ai-eyes" width="320">
</p>

# ai-eyes handbook

ai-eyes is an MCP server that gives an LLM (Claude) **honest, grounded image judgment** — it *measures*, it doesn't narrate.

## The thesis: measurement, not narration

When an LLM needs to verify what's in an image, generative vision models (LLaVA, GPT‑4V) **complete a narrative**: asked "is there a sword?", they tend to answer "yes, the character wields a greatsword" — confidently, whether or not a weapon is present. LLaVA‑13B reported a weapon on *every* sprite crop, including unarmed ones, at ~0.90 confidence.

ai-eyes wraps **SigLIP2** — a *discriminative* vision model. It doesn't generate text; it returns a calibrated similarity score between an image and a text description. Weapon‑present crops score 10–100× higher than weapon‑absent ones. When it can't tell, the score is low. **It has no narrative to complete, so it doesn't hallucinate.**

## Pages

- [The 7 tools](tools.md) — what each does, with examples.
- [Honest judgment](honest-judgment.md) — how to get reliable verdicts (prefer relative ranking over absolute thresholds).
- [Install & validate](install-and-validate.md) — set up + prove the install is calibrated with `eyes_selftest`.
- [Architecture](architecture.md) — image + query → SigLIP2 → sigmoid → verdict.

## At a glance

- **7 tools**, STDIO MCP server (FastMCP v3), SigLIP2 SO400M (`google/siglip2-so400m-patch14-384`).
- **Deterministic** (no sampling) — same image + query → identical score.
- ~100 ms/query on GPU; runs on CPU too. First call loads the model (~10–20 s).
- Read-only: it analyzes image paths you give it. No writes, no telemetry, no network egress by default (the one-time model download excepted).
