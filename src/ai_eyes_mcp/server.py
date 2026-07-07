"""
ai-eyes MCP server — grounded visual evaluator.

Exposes SigLIP2 as MCP tools for honest image judgment.
Returns calibrated scores, not prose. Measures, not narrates.

Tools:
  image_contains    — "Does this image contain X?"
  image_classify    — Score image against N candidate labels
  image_compare     — Cosine similarity between two images
  image_score_batch — Score N images against one query
  eyes_status       — Health check
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ai_eyes_mcp.engine import SigLIPEngine, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DEFAULT_CACHE_DIR, DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_THRESHOLD

logger = logging.getLogger("ai_eyes_mcp")

# Configurable verbosity — AI_EYES_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
# (default WARNING). Logs go to STDERR only: STDOUT is the MCP STDIO protocol
# channel and must never be polluted.
_log_level = os.environ.get("AI_EYES_LOG_LEVEL", "WARNING").strip().upper()
logger.setLevel(getattr(logging, _log_level, logging.WARNING))
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# ---------------------------------------------------------------------------
# Server + engine setup
#
# NOTE (F-PB-007): The engine is constructed at module level deliberately.
# For an MCP server the module IS the server process — lazy construction
# would just defer the same work to the first tool call with no benefit.
# CUDA init happens here so startup failures surface immediately.
# ---------------------------------------------------------------------------

mcp = FastMCP(name="ai-eyes")

engine = SigLIPEngine(
    model_id=os.environ.get("AI_EYES_MODEL_ID", DEFAULT_MODEL_ID),
    revision=os.environ.get("AI_EYES_MODEL_REVISION", DEFAULT_MODEL_REVISION),
    cache_dir=DEFAULT_CACHE_DIR,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool
def image_contains(
    image_path: Annotated[str, Field(description="Absolute path to the image file")],
    query: Annotated[str, Field(description="What to look for (e.g. 'a person holding a sword', 'a login button', 'a red car')")],
    threshold: Annotated[float, Field(description="Score threshold for 'present' verdict (default 0.02; scores are query-phrasing sensitive — see docstring)")] = DEFAULT_THRESHOLD,
) -> dict:
    """Check if an image contains something described by the query.

    Returns an independent sigmoid score (0-1) — higher means stronger visual match.
    The score is NOT relative to other queries. Each query is evaluated on its own.

    Uses SigLIP2 discriminative vision — measures visual similarity, does not
    generate text or hallucinate. If the image doesn't clearly match, the score
    will be low.

    NOTE — sigmoid scores are query-phrasing sensitive: absolute scores swing
    widely with wording, so a fixed ``threshold`` needs query engineering per
    use case. For robust yes/no decisions across varied inputs, prefer
    ``image_classify`` (relative ranking of candidate labels), which is
    insensitive to absolute score magnitude.
    """
    t0 = time.perf_counter()
    if not (0.0 <= threshold <= 1.0):  # NaN-safe: a NaN threshold fails the chained compare
        raise ToolError("threshold must be between 0.0 and 1.0 (sigmoid score range)")
    image_path = str(Path(image_path).resolve())
    if not engine.loaded:
        try:
            engine._ensure_loaded()
        except Exception as e:
            raise ToolError(f"Model not loaded: {e}. Check AI_EYES_MODEL_DIR, AI_EYES_DEVICE, and network.")
    try:
        score = engine.score(image_path, query)
    except FileNotFoundError:
        raise ToolError(f"Image not found: {image_path}")
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}")
    except Exception as e:
        raise ToolError(f"Scoring failed: {e}")

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_contains completed in %.3fs", elapsed / 1000)
    return {
        "present": score > threshold,
        "score": round(score, 4),
        "threshold": threshold,
        "query": query,
        "elapsed_ms": elapsed,
    }


@mcp.tool
def image_classify(
    image_path: Annotated[str, Field(description="Absolute path to the image file")],
    labels: Annotated[list[str], Field(description="Candidate labels to score (e.g. ['cat', 'dog', 'bird'])")],
) -> dict:
    """Score an image against multiple candidate labels.

    Returns independent sigmoid scores for EACH label — these are NOT softmax.
    Multiple labels can score high simultaneously (e.g. an image with both a
    cat and a dog). A label scoring low means it's NOT confidently present.

    This is measurement, not conversation. The model looks at the pixels and
    reports similarity to each text description.
    """
    if not labels:
        raise ToolError("At least one label is required")
    if len(labels) > 20:
        raise ToolError("Maximum 20 labels per call")

    t0 = time.perf_counter()
    image_path = str(Path(image_path).resolve())
    if not engine.loaded:
        try:
            engine._ensure_loaded()
        except Exception as e:
            raise ToolError(f"Model not loaded: {e}. Check AI_EYES_MODEL_DIR, AI_EYES_DEVICE, and network.")
    try:
        scores = engine.score_multi(image_path, labels)
    except FileNotFoundError:
        raise ToolError(f"Image not found: {image_path}")
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}")
    except Exception as e:
        raise ToolError(f"Classification failed: {e}")

    rounded = {k: round(v, 4) for k, v in scores.items()}
    sorted_scores = dict(sorted(rounded.items(), key=lambda kv: kv[1], reverse=True))
    best_label = max(rounded, key=rounded.get)

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_classify completed in %.3fs", elapsed / 1000)
    return {
        "scores": sorted_scores,
        "best": best_label,
        "best_score": rounded[best_label],
        "elapsed_ms": elapsed,
    }


@mcp.tool
def image_compare(
    image_a: Annotated[str, Field(description="Absolute path to the first image")],
    image_b: Annotated[str, Field(description="Absolute path to the second image")],
) -> dict:
    """Compute visual similarity between two images.

    Returns cosine similarity (-1 to 1) of their SigLIP2 embeddings.
    Higher values mean the images look more alike to the vision model.

    Use cases: comparing sprite poses, checking if two renders match,
    detecting visual duplicates.
    """
    t0 = time.perf_counter()
    image_a = str(Path(image_a).resolve())
    image_b = str(Path(image_b).resolve())
    if not engine.loaded:
        try:
            engine._ensure_loaded()
        except Exception as e:
            raise ToolError(f"Model not loaded: {e}. Check AI_EYES_MODEL_DIR, AI_EYES_DEVICE, and network.")
    try:
        similarity = engine.compare(image_a, image_b)
    except FileNotFoundError as e:
        raise ToolError(str(e))
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}")
    except Exception as e:
        raise ToolError(f"Comparison failed: {e}")

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_compare completed in %.3fs", elapsed / 1000)
    return {
        "similarity": round(similarity, 4),
        "image_a": image_a,
        "image_b": image_b,
        "elapsed_ms": elapsed,
    }


@mcp.tool
def image_score_batch(
    image_paths: Annotated[list[str], Field(description="List of absolute image file paths")],
    query: Annotated[str, Field(description="What to look for in each image")],
    threshold: Annotated[float, Field(description="Score threshold for 'present' verdict")] = DEFAULT_THRESHOLD,
) -> dict:
    """Score multiple images against a single query.

    Returns per-image scores and a summary. Useful for batch evaluation,
    cherry-picking the best match, or scanning a directory of images.
    """
    if not image_paths:
        raise ToolError("At least one image path is required")
    if len(image_paths) > 100:
        raise ToolError("Maximum 100 images per batch")
    if not (0.0 <= threshold <= 1.0):  # NaN-safe: a NaN threshold fails the chained compare
        raise ToolError("threshold must be between 0.0 and 1.0 (sigmoid score range)")

    t0 = time.perf_counter()
    image_paths = [str(Path(p).resolve()) for p in image_paths]
    if not engine.loaded:
        try:
            engine._ensure_loaded()
        except Exception as e:
            raise ToolError(f"Model not loaded: {e}. Check AI_EYES_MODEL_DIR, AI_EYES_DEVICE, and network.")
    results = []
    errors = []

    # Encode text ONCE via the engine, then score each image with pre-encoded text
    try:
        text_inputs = engine._encode_text(query)
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}")
    except Exception as e:
        raise ToolError(f"Text encoding failed: {e}")

    for path in image_paths:
        try:
            score = engine._score_with_text_inputs(path, text_inputs)
            results.append({
                "path": path,
                "score": round(score, 4),
                "present": score > threshold,
            })
        except FileNotFoundError:
            errors.append({"path": path, "error": "not found"})
        except ValueError as e:
            # Sanitized: never leak raw PIL/exception text (may include absolute
            # paths / internals) into the response. "invalid image" covers
            # corrupt, unsupported-format, and decompression-bomb cases.
            logger.debug("batch item failed (invalid image): %s", e)
            errors.append({"path": path, "error": "invalid image"})
        except Exception as e:
            logger.debug("batch item failed: %s", e)
            errors.append({"path": path, "error": "scoring failed"})

    present_count = sum(1 for r in results if r["present"])

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_score_batch completed in %.3fs", elapsed / 1000)
    response = {
        "query": query,
        "threshold": threshold,
        "total": len(image_paths),
        "scored": len(results),
        "present": present_count,
        "absent": len(results) - present_count,
        "errors": len(errors),
        "results": results,
        "error_details": errors if errors else None,
        "elapsed_ms": elapsed,
    }
    if results:
        best = max(results, key=lambda r: r["score"])
        response["best_path"] = best["path"]
        response["best_score"] = best["score"]
    return response


@mcp.tool
def eyes_status() -> dict:
    """Check ai-eyes server status.

    Returns model info, device, and whether the model is currently loaded.
    The model loads lazily on first tool call — this tool does NOT trigger loading.
    """
    t0 = time.perf_counter()
    result = engine.status()
    result["scoring_guidance"] = (
        "Sigmoid scores are query-phrasing sensitive — absolute thresholds need "
        "per-use-case tuning. For robust decisions, prefer image_classify "
        "(relative ranking), which is insensitive to absolute score magnitude."
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("eyes_status completed in %.3fs", elapsed / 1000)
    result["elapsed_ms"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()
