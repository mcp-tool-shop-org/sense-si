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

from ai_eyes_mcp.engine import (
    SigLIPEngine,
    DEFAULT_CACHE_DIR,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_THRESHOLD,
    round_preserving_gt,
    display_round,
    configure_logging,
    confidence_from_magnitude,
    round_margin,
)

logger = logging.getLogger("ai_eyes_mcp")
configure_logging()

# ---------------------------------------------------------------------------
# Server + engine setup
#
# NOTE (F-PB-007): The engine is constructed at module level deliberately.
# For an MCP server the module IS the server process — lazy construction
# would just defer the same work to the first tool call with no benefit.
# CUDA init happens here so startup failures surface immediately.
# ---------------------------------------------------------------------------

mcp = FastMCP(name="ai-eyes")


def _construct_engine() -> SigLIPEngine:
    """Build the process engine. Eager-load failures must not dump a traceback
    (W1-COORD-007 / SHIP_GATE B)."""
    try:
        return SigLIPEngine(
            # model_id / revision / cache / device / dtype all resolve inside
            # the engine from env-or-default. One source of truth (W1-COORD-005).
            revision=os.environ.get("AI_EYES_MODEL_REVISION"),
            cache_dir=DEFAULT_CACHE_DIR,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
        )
    except Exception as exc:
        logger.debug("engine construction failed", exc_info=exc)
        sys.stderr.write(
            "Failed to initialize ai-eyes engine. "
            "Check AI_EYES_MODEL_ID, AI_EYES_MODEL_DIR, AI_EYES_DEVICE. "
            "Set AI_EYES_LOG_LEVEL=DEBUG for the traceback.\n"
        )
        raise SystemExit(1) from None


engine = _construct_engine()


def _parse_baseline_pairs(baselines) -> list[tuple[str, str]] | None:
    """Caller-supplied 'these are not a match' pairs. None/empty → no floor."""
    if not baselines:
        return None
    pairs: list[tuple[str, str]] = []
    for item in baselines:
        if item is None or len(item) != 2:
            raise ToolError(
                "each baseline must be a pair of image paths [path_a, path_b] "
                "that are NOT a match in this style"
            )
        pairs.append((str(Path(item[0]).resolve()), str(Path(item[1]).resolve())))
    return pairs


def _separation(similarity: float, floor: float) -> dict:
    margin = similarity - floor
    return {
        "separated": similarity > floor,
        "margin": round_margin(margin),
        "baseline_max": float(floor),
        "confidence": confidence_from_magnitude(abs(margin)),
    }


# ---------------------------------------------------------------------------
# Error mapping — one place so every tool returns the same actionable shape
# ---------------------------------------------------------------------------

def _tool_error(exc: Exception) -> ToolError:
    """Map an engine exception to a consistent, actionable ToolError.

    The message tells an LLM caller what to DO, not just what broke. Raw
    exception text is forwarded only where it is already actionable (missing
    path, bad input); the catch-all is sanitized (unactionable internals can
    leak paths / stack detail) and the real error is logged at DEBUG.
    """
    if isinstance(exc, FileNotFoundError):
        return ToolError(f"{exc} — check the path exists and points to a readable image file.")
    if isinstance(exc, ValueError):
        # Engine ValueErrors are already actionable (empty / too-long query,
        # corrupt or unsupported image).
        return ToolError(f"Invalid input: {exc}")
    if "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError":
        return ToolError(
            "GPU out of memory. Try AI_EYES_DTYPE=float16 (halves VRAM), a smaller "
            "batch, or AI_EYES_DEVICE=cpu."
        )
    logger.debug("unexpected engine error", exc_info=exc)
    return ToolError(
        "Scoring failed (unexpected internal error). Set AI_EYES_LOG_LEVEL=DEBUG "
        "on the server for the full traceback."
    )


def _load_error(exc: Exception) -> ToolError:
    """Sanitized load failure — never interpolate the raw exception (W1-SERVER-004)."""
    logger.debug("model load failed", exc_info=exc)
    return ToolError(
        "Model not loaded. Check AI_EYES_MODEL_DIR, AI_EYES_DEVICE, and network. "
        "Set AI_EYES_LOG_LEVEL=DEBUG on the server for the full traceback."
    )


def _ensure_engine_ready() -> None:
    if engine.loaded:
        return
    try:
        engine._ensure_loaded()
    except Exception as exc:
        raise _load_error(exc) from None


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
    _ensure_engine_ready()
    try:
        score = engine.score(image_path, query)
        truncated = engine.query_truncated(query)
    except Exception as e:
        raise _tool_error(e) from None

    present = score > threshold
    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_contains completed in %.3fs", elapsed / 1000)
    return {
        "present": present,
        "score": round_preserving_gt(score, threshold),
        "threshold": threshold,
        "query": query,
        "truncated": truncated,
        "revision": engine._resolved_revision,
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
    cat and a dog). ``best`` is the highest of the labels you supplied, not a
    presence verdict.

    Absolute scores are query-phrasing sensitive: a low number can mean a weak
    phrase rather than absence. Compare labels to each other. For a relative
    yes/no against caller-supplied contrasts, prefer ``image_verify``.
    """
    if not labels:
        raise ToolError("At least one label is required")
    if len(labels) > 20:
        raise ToolError("Maximum 20 labels per call")

    t0 = time.perf_counter()
    image_path = str(Path(image_path).resolve())
    _ensure_engine_ready()
    try:
        scores = engine.score_multi(image_path, labels)
        truncated = any(engine.query_truncated(label) for label in labels)
    except Exception as e:
        raise _tool_error(e) from None

    # Rank on full precision (Class 1). Rounding first collapses gaps < 1e-4
    # and max() then follows caller label order — W1-COORD-009.
    best_label = max(scores, key=scores.get)
    ranked = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    displayed = None
    for digits in range(4, 16):
        candidate = {k: display_round(v, digits) for k, v in ranked.items()}
        if candidate[best_label] >= max(candidate.values()):
            displayed = candidate
            break
    if displayed is None:
        displayed = ranked

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_classify completed in %.3fs", elapsed / 1000)
    return {
        "scores": displayed,
        "best": best_label,
        "best_score": displayed[best_label],
        "truncated": truncated,
        "revision": engine._resolved_revision,
        "elapsed_ms": elapsed,
    }


@mcp.tool
def image_compare(
    image_a: Annotated[str, Field(description="Absolute path to the first image")],
    image_b: Annotated[str, Field(description="Absolute path to the second image")],
    baselines: Annotated[
        list[list[str]] | None,
        Field(
            description=(
                "Pairs of images that are NOT a match in this style. A-B is "
                "separated only if its similarity exceeds this caller-supplied "
                "floor. Omit for a raw measurement (incomplete — not a verdict)."
            )
        ),
    ] = None,
) -> dict:
    """Compute visual similarity between two images.

    Returns cosine similarity (-1 to 1) of their SigLIP2 embeddings.
    Without ``baselines`` the number is a measurement, not a match verdict
    (``incomplete: true``) — similarity floors do not transfer across styles.
    With baselines, ``separated`` reports whether A-B exceeds the caller-supplied
    'these are different' floor.
    """
    t0 = time.perf_counter()
    image_a = str(Path(image_a).resolve())
    image_b = str(Path(image_b).resolve())
    pairs = _parse_baseline_pairs(baselines)
    _ensure_engine_ready()
    try:
        similarity = engine.compare(image_a, image_b)
        floor = None
        if pairs:
            floor = max(engine.compare(x, y) for x, y in pairs)
    except Exception as e:
        raise _tool_error(e) from None

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("image_compare completed in %.3fs", elapsed / 1000)
    payload = {
        "similarity": round(similarity, 4),
        "image_a": image_a,
        "image_b": image_b,
        "revision": engine._resolved_revision,
        "incomplete": floor is None,
        "elapsed_ms": elapsed,
    }
    if floor is None:
        payload["separated"] = None
        payload["margin"] = None
        payload["baseline_max"] = None
        payload["confidence"] = None
    else:
        payload.update(_separation(float(similarity), float(floor)))
    return payload


@mcp.tool
def image_rank(
    reference: Annotated[str, Field(description="Absolute path to the reference image")],
    candidates: Annotated[list[str], Field(description="Candidate image paths to rank against the reference")],
    k: Annotated[int, Field(description="Maximum matches to return")] = 5,
    baselines: Annotated[
        list[list[str]] | None,
        Field(
            description=(
                "Pairs of images that are NOT a match in this style. Candidates "
                "at or below this floor are not close. Omit for a raw top-k "
                "measurement (incomplete — not a verdict)."
            )
        ),
    ] = None,
) -> dict:
    """Rank candidates by cosine similarity to one reference.

    Encodes the reference once. Returns top-k with margins. Without baselines
    the ranking is a measurement (``incomplete: true``), not 'these match'.
    With baselines, only candidates whose similarity exceeds the caller-supplied
    'different' floor are matches; if none do, ``matches`` is empty and
    ``nothing_close`` is true.
    """
    if not candidates:
        raise ToolError("At least one candidate is required")
    if len(candidates) > 100:
        raise ToolError("Maximum 100 candidates per call")
    if k < 1:
        raise ToolError("k must be >= 1")
    t0 = time.perf_counter()
    reference = str(Path(reference).resolve())
    candidates = [str(Path(p).resolve()) for p in candidates]
    pairs = _parse_baseline_pairs(baselines)
    _ensure_engine_ready()
    try:
        sims = engine.similarities_to_reference(reference, candidates)
        floor = None
        if pairs:
            floor = max(engine.compare(x, y) for x, y in pairs)
    except Exception as e:
        raise _tool_error(e) from None

    ranked = sorted(zip(candidates, sims), key=lambda kv: kv[1], reverse=True)
    matches = []
    if floor is None:
        take = ranked[:k]
        for i, (path, sim) in enumerate(take):
            nxt = take[i + 1][1] if i + 1 < len(take) else None
            matches.append({
                "path": path,
                "similarity": round(float(sim), 4),
                "margin_to_next": None if nxt is None else round(float(sim) - float(nxt), 4),
                "separated": None,
            })
        nothing_close = False
    else:
        above = [(p, s) for p, s in ranked if float(s) > float(floor)]
        take = above[:k]
        for i, (path, sim) in enumerate(take):
            nxt = take[i + 1][1] if i + 1 < len(take) else None
            matches.append({
                "path": path,
                "similarity": round(float(sim), 4),
                "margin_to_next": None if nxt is None else round(float(sim) - float(nxt), 4),
                "separated": True,
                "margin_to_baseline": round_margin(float(sim) - float(floor)),
            })
        nothing_close = len(above) == 0

    elapsed = round((time.perf_counter() - t0) * 1000)
    return {
        "reference": reference,
        "k": k,
        "matches": matches,
        "incomplete": floor is None,
        "nothing_close": nothing_close,
        "baseline_max": None if floor is None else float(floor),
        "revision": engine._resolved_revision,
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
    _ensure_engine_ready()
    results = []
    errors = []

    # Encode text ONCE via the engine, then score each image with pre-encoded text
    try:
        text_inputs = engine._encode_text(query)
        truncated = engine.query_truncated(query)
    except Exception as e:
        raise _tool_error(e) from None

    for path in image_paths:
        try:
            score = engine._score_with_text_inputs(path, text_inputs)
            present = score > threshold
            results.append({
                "path": path,
                "score": round_preserving_gt(score, threshold),
                "present": present,
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
        "truncated": truncated,
        "revision": engine._resolved_revision,
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
    if not result.get("loaded"):
        result["note"] = (
            "Model not loaded yet — the first image tool call loads SigLIP2 "
            "(~10-20s on GPU); subsequent calls are ~100ms. Set AI_EYES_EAGER_LOAD=1 "
            "to load at server start instead."
        )
    # The honesty contract. Reciprocal of plain-sight's _HONESTY_GUIDANCE, which
    # sends callers here when a claim needs measuring rather than narrating. The
    # two strings are a matched pair across repos — neither side rewords alone.
    result["scoring_guidance"] = (
        "ai-eyes measures one image-text pair. It does not describe the image and "
        "it does not complete a narrative. A score is a sigmoid for the query AS "
        "TOKENIZED, on the pinned SigLIP2 revision named in this payload. It is "
        "not a probability that 'X is in the photo' independent of wording.\n"
        "Treat these as incomplete, not as a number: truncated=true (the score is "
        "of a prefix, not your full query); a confidence containing 'inconclusive'; "
        "any payload with no revision.\n"
        "Do not threshold a raw image_contains score across image styles. Prefer "
        "image_verify (relative, against contrasts you supply) or image_classify "
        "(ranking).\n"
        "For a caption, use plain-sight. For a claim about pixels, use ai-eyes."
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.debug("eyes_status completed in %.3fs", elapsed / 1000)
    result["elapsed_ms"] = elapsed
    return result


@mcp.tool
def image_verify(
    image_path: Annotated[str, Field(description="Absolute path to the image file")],
    target: Annotated[str, Field(description="The hypothesis to verify (e.g. 'a knight with a sword')")],
    alternatives: Annotated[list[str], Field(description="Contrast alternatives to rank the target against — REQUIRED (this is a relative verdict), e.g. ['a goblin cook', 'a bard']")],
) -> dict:
    """Honest RELATIVE verdict: is `target` a better description of the image than
    the caller-supplied `alternatives`?

    Unlike image_contains (which returns a raw sigmoid you must threshold),
    image_verify returns a DECISION + margin + confidence by RANKING the target
    against alternatives — robust to SigLIP's query-phrasing sensitivity because
    it is relative, not absolute. `alternatives` is required (>=1); for a raw
    score use image_contains, for full ranking use image_classify.

    Returns `{present, target, target_score, best_alternative,
    best_alternative_score, margin, confidence}`. `confidence` describes the
    measured gap ("high" / "moderate" / "low — inconclusive"), not invented
    certainty.
    """
    t0 = time.perf_counter()
    image_path = str(Path(image_path).resolve())
    _ensure_engine_ready()
    try:
        result = engine.verify(image_path, target, alternatives)
        result["truncated"] = engine.query_truncated(target) or any(
            engine.query_truncated(a) for a in alternatives
        )
        result["revision"] = engine._resolved_revision
    except Exception as e:
        raise _tool_error(e) from None
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


@mcp.tool
def eyes_selftest() -> dict:
    """Self-test the instrument: run the model on a few bundled reference images
    and confirm the expected orderings hold — proves the install loaded correctly
    and SigLIP2 is calibrated. Loads the model if it isn't already.

    Returns `{passed, checks: [{name, expected, measured_a, measured_b, ok}],
    model_id, device, torch_version, transformers_version}`.
    """
    t0 = time.perf_counter()
    try:
        result = engine.selftest()
    except Exception as e:
        raise _tool_error(e) from None
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()
