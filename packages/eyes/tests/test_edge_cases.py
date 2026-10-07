"""
Edge-case tests — error handling and boundary conditions.

Tests that the engine and tools fail gracefully with correct error types.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fastmcp.exceptions import ToolError

from ai_eyes_mcp.engine import SigLIPEngine
from ai_eyes_mcp.server import (
    image_contains,
    image_classify,
    image_compare,
    image_score_batch,
)


FAKE_PATH = "F:/nonexistent/totally_fake_image.png"  # intentionally nonexistent on any rig
# Portable: the repo's own pyproject.toml — a real file that is NOT an image.
# (Was a hardcoded F:/ path that didn't exist on this rig, so the "non-image"
# tests were silently exercising the file-not-found path instead — Wave 0 fix.)
NON_IMAGE_PATH = str(Path(__file__).resolve().parent.parent / "pyproject.toml")


# ===========================================================================
# Engine-level errors
# ===========================================================================

@pytest.mark.dogfood
class TestEngineErrors:

    def test_score_missing_file(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.score(FAKE_PATH, "anything")

    def test_score_multi_missing_file(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.score_multi(FAKE_PATH, ["a", "b"])

    def test_embed_missing_file(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.embed_image(FAKE_PATH)

    def test_compare_missing_first(self, engine, photo_cheetah):
        # Image-agnostic — re-pointed to a vendored photo in Wave 0.
        with pytest.raises(FileNotFoundError):
            engine.compare(FAKE_PATH, photo_cheetah)

    def test_compare_missing_second(self, engine, photo_cheetah):
        # Image-agnostic — re-pointed to a vendored photo in Wave 0.
        with pytest.raises(FileNotFoundError):
            engine.compare(photo_cheetah, FAKE_PATH)

    def test_score_invalid_image_type(self, engine):
        """Passing a non-image file (pyproject.toml) must raise ValueError.

        The engine's _load_image catches PIL's UnidentifiedImageError and
        re-raises ValueError ("Cannot open image ..."). Asserting ValueError
        specifically — not the broad OSError, which FileNotFoundError also
        satisfies — ensures this exercises non-image handling, not file-not-found.
        """
        with pytest.raises(ValueError, match="Cannot open image"):
            engine.score(NON_IMAGE_PATH, "anything")

    def test_score_batch_missing_file(self, engine):
        """score_batch with a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            engine.score_batch(["F:/nonexistent/fake.png"], "test")

    def test_score_directory_path(self, engine, tmp_path):
        """Passing a directory path instead of a file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not a file"):
            engine.score(str(tmp_path), "anything")


# ===========================================================================
# Tool-level errors (ToolError)
# ===========================================================================

@pytest.mark.dogfood
class TestToolErrors:

    def test_contains_missing_file(self):
        with pytest.raises(ToolError, match="not found"):
            image_contains(FAKE_PATH, "anything")

    def test_classify_missing_file(self):
        with pytest.raises(ToolError, match="not found"):
            image_classify(FAKE_PATH, ["a", "b"])

    def test_classify_empty_labels(self, photo_cheetah):
        with pytest.raises(ToolError, match="At least one label"):
            image_classify(photo_cheetah, [])

    def test_classify_too_many_labels(self, photo_cheetah):
        labels = [f"label_{i}" for i in range(21)]
        with pytest.raises(ToolError, match="Maximum 20"):
            image_classify(photo_cheetah, labels)

    def test_compare_missing_image_a(self, photo_cheetah):
        # Image-agnostic — re-pointed to a vendored photo in Wave 0.
        with pytest.raises(ToolError, match="not found"):
            image_compare(FAKE_PATH, photo_cheetah)

    def test_compare_missing_image_b(self, photo_cheetah):
        # Image-agnostic — re-pointed to a vendored photo in Wave 0.
        with pytest.raises(ToolError, match="not found"):
            image_compare(photo_cheetah, FAKE_PATH)

    def test_batch_empty_list(self):
        with pytest.raises(ToolError, match="At least one"):
            image_score_batch([], "anything")

    def test_batch_too_many(self):
        paths = [f"F:/fake_{i}.png" for i in range(101)]
        with pytest.raises(ToolError, match="Maximum 100"):
            image_score_batch(paths, "anything")

    def test_batch_empty_query(self, photo_cheetah):
        """image_score_batch must reject an empty query like its sibling tools.

        Regression guard: it previously bypassed _validate_query by calling
        engine._encode_text() directly instead of score/score_multi.
        """
        with pytest.raises(ToolError, match="empty"):
            image_score_batch([photo_cheetah], "")

    def test_contains_invalid_image_type(self):
        """Passing a non-image file should raise ToolError."""
        with pytest.raises(ToolError, match="Cannot open image"):
            image_contains(NON_IMAGE_PATH, "anything")

    def test_contains_empty_query(self, photo_cheetah):
        """Empty query string should raise ToolError at tool level.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        with pytest.raises(ToolError, match="empty"):
            image_contains(photo_cheetah, "")

    def test_contains_overlength_query(self, photo_cheetah):
        """Query exceeding 500 chars should raise ToolError at tool level.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        long_query = "x" * 501
        with pytest.raises(ToolError, match="too long"):
            image_contains(photo_cheetah, long_query)

    def test_classify_empty_label_in_list(self, photo_cheetah):
        """A list containing an empty string label should raise ToolError.

        The engine deduplicates and validates each query via _validate_query,
        which rejects empty strings. The tool wraps that as ToolError.
        """
        with pytest.raises(ToolError, match="empty"):
            image_classify(photo_cheetah, ["cheetah", "", "bus"])

    def test_compare_non_image_as_image_a(self, photo_cheetah):
        """Passing a non-image file as image_a should raise ToolError.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        with pytest.raises(ToolError, match="Cannot open image"):
            image_compare(NON_IMAGE_PATH, photo_cheetah)

    def test_compare_non_image_as_image_b(self, photo_cheetah):
        """Passing a non-image file as image_b should raise ToolError.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        with pytest.raises(ToolError, match="Cannot open image"):
            image_compare(photo_cheetah, NON_IMAGE_PATH)

    def test_batch_non_image_captured_in_errors(self, photo_cheetah):
        """Non-image file in batch should be captured in error_details, not crash.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        result = image_score_batch(
            [photo_cheetah, NON_IMAGE_PATH],
            "test query",
        )
        assert result["errors"] == 1, f"Expected 1 error, got {result['errors']}"
        assert result["scored"] == 1, f"Expected 1 scored, got {result['scored']}"
        assert result["error_details"] is not None
        # Path.resolve() normalizes separators on Windows (/ → \)
        assert Path(result["error_details"][0]["path"]) == Path(NON_IMAGE_PATH).resolve()


# ===========================================================================
# Boundary values
# ===========================================================================

@pytest.mark.dogfood
class TestBoundaryValues:

    def test_single_label_classify(self, photo_cheetah):
        """One label should still work."""
        result = image_classify(photo_cheetah, ["cheetah"])
        assert result["best"] == "cheetah"
        assert len(result["scores"]) == 1

    def test_twenty_labels_classify(self, photo_cheetah):
        """Exactly 20 labels should work (the limit)."""
        labels = [f"label_{i}" for i in range(20)]
        result = image_classify(photo_cheetah, labels)
        assert len(result["scores"]) == 20

    def test_single_image_batch(self, photo_cheetah):
        """Single image in batch should work."""
        result = image_score_batch([photo_cheetah], "cat")
        assert result["total"] == 1
        assert result["scored"] == 1

    def test_moderately_long_query(self, engine, photo_cheetah):
        """Reasonably long query shouldn't crash."""
        query = "a detailed description of a cheetah running through the african savanna"
        score = engine.score(photo_cheetah, query)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_query_rejected(self, engine, photo_cheetah):
        """Empty string query should be rejected with ValueError."""
        with pytest.raises(ValueError, match="empty"):
            engine.score(photo_cheetah, "")

    def test_whitespace_only_query_rejected(self, engine, photo_cheetah):
        """Whitespace-only query should be rejected with ValueError matching 'empty'."""
        with pytest.raises(ValueError, match="empty"):
            engine.score(photo_cheetah, "   ")

    def test_unicode_query(self, engine, photo_cheetah):
        """Unicode queries shouldn't crash."""
        score = engine.score(photo_cheetah, "un guépard rapide 🐆")
        assert isinstance(score, float)

    def test_score_range(self, engine, photo_cheetah):
        """Sigmoid score should always be in [0, 1].
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        for query in ["sword", "nothing", "xyzzy", "a", "the quick brown fox"]:
            score = engine.score(photo_cheetah, query)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for query '{query}'"

    def test_zero_threshold(self, photo_cheetah):
        """threshold=0.0 means any positive score triggers present=True.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0; query matches
        the photo so the rounded score is reliably above zero.)"""
        result = image_contains(photo_cheetah, "a cheetah", threshold=0.0)
        assert result["score"] > 0.0, "A matching image should score above zero"
        assert result["present"] is True, (
            f"Score {result['score']} > threshold 0.0 but present is False"
        )

    def test_negative_threshold_rejected(self, photo_cheetah):
        """threshold=-1.0 should be rejected — sigmoid scores are [0, 1].
        (Image-agnostic — threshold is validated before the image is read.)"""
        with pytest.raises(ToolError, match="between 0.0 and 1.0"):
            image_contains(photo_cheetah, "pixel art knight", threshold=-1.0)

    def test_duplicate_labels_in_classify(self, photo_cheetah):
        """Duplicate labels are deduplicated by score_multi before scoring.

        Passing ["cat", "cat", "dog"] results in only 2 entries because
        the engine deduplicates queries before evaluation, so the returned
        dict has one entry per unique label.
        """
        result = image_classify(photo_cheetah, ["cat", "cat", "dog"])
        assert len(result["scores"]) == 2, (
            f"Expected 2 deduped entries, got {len(result['scores'])}: {result['scores']}"
        )


# ===========================================================================
# Stage B — proactive hardening
# ===========================================================================

class TestProactiveHardening:
    """Defensive-coding + observability guards (Stage B)."""

    # --- E-05: degenerate-input guards (fire before model load) ---

    @pytest.mark.dogfood
    def test_score_multi_empty_queries_rejected(self, engine):
        """score_multi([]) must raise a clear error, not reach the tokenizer."""
        with pytest.raises(ValueError, match="At least one query"):
            engine.score_multi("unused.png", [])

    @pytest.mark.dogfood
    def test_score_batch_empty_paths_rejected(self, engine):
        """score_batch([], q) must raise a clear error, not return []."""
        with pytest.raises(ValueError, match="At least one image"):
            engine.score_batch([], "a query")

    # --- E-06: token-truncation observability ---

    @pytest.mark.dogfood
    def test_long_query_logs_truncation_warning(self, engine, photo_cheetah, caplog):
        """A query that tokenizes past the text-encoder limit must WARN — otherwise
        the score silently reflects only the first N tokens."""
        long_query = "cat dog fox " * 39  # ~117 tokens, 468 chars (< MAX_QUERY_LENGTH)
        with caplog.at_level(logging.WARNING, logger="ai_eyes_mcp"):
            score = engine.score(photo_cheetah, long_query)
        # Must be truncated and scored — NOT crash the forward pass
        # (>64 tokens raises "Sequence length ... > max_position_embeddings").
        assert isinstance(score, float) and 0.0 <= score <= 1.0
        assert any("truncat" in r.getMessage().lower() for r in caplog.records), (
            "expected a truncation warning for an over-length query"
        )

    @pytest.mark.dogfood
    def test_short_query_no_truncation_warning(self, engine, photo_cheetah, caplog):
        with caplog.at_level(logging.WARNING, logger="ai_eyes_mcp"):
            engine.score(photo_cheetah, "a cheetah")
        assert not any("truncat" in r.getMessage().lower() for r in caplog.records)

    # --- F2: batch error messages are sanitized (no raw path/exception leak) ---

    @pytest.mark.dogfood
    def test_batch_error_message_is_sanitized(self, photo_cheetah):
        """A bad image in a batch must yield a sanitized classification, not a raw
        exception string that leaks the resolved path / PIL internals."""
        result = image_score_batch([photo_cheetah, NON_IMAGE_PATH], "test")
        assert result["errors"] == 1
        assert result["scored"] == 1  # degrade, don't abort the batch
        msg = result["error_details"][0]["error"]
        assert msg in ("not found", "invalid image", "scoring failed"), (
            f"batch error message must be a sanitized classification, got {msg!r}"
        )

    # --- AI_EYES_LOG_LEVEL: configurable verbosity (closes a SHIP_GATE gap) ---

    def test_log_level_env_configures_logger(self):
        env = os.environ.copy()
        env["AI_EYES_LOG_LEVEL"] = "DEBUG"
        r = subprocess.run(
            [sys.executable, "-c",
             "import ai_eyes_mcp.server, logging; "
             "lvl = logging.getLogger('ai_eyes_mcp').level; "
             "assert lvl == logging.DEBUG, lvl"],
            env=env, capture_output=True, text=True, timeout=90,
        )
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"

    # --- E-04: eager load surfaces a broken model at construction ---

    def test_eager_load_surfaces_failure_at_construction(self):
        """AI_EYES_EAGER_LOAD makes a broken model/cache fail at construction
        (server start), not on the first tool call."""
        env = os.environ.copy()
        env["AI_EYES_EAGER_LOAD"] = "1"
        env["HF_HUB_OFFLINE"] = "1"  # fail fast, no network
        r = subprocess.run(
            [sys.executable, "-c",
             "from ai_eyes_mcp.engine import SigLIPEngine; "
             "SigLIPEngine(model_id='ai-eyes-nonexistent/does-not-exist-xyz')"],
            env=env, capture_output=True, text=True, timeout=90,
        )
        assert r.returncode != 0, (
            "eager load of a nonexistent model should fail at construction, not silently succeed"
        )


# ===========================================================================
# Stage C — LLM-caller error-hint actionability
# ===========================================================================

class TestErrorHints:
    """Errors an LLM caller can hit must carry an ACTIONABLE hint (what to DO),
    consistently, and must not leak raw internals."""

    def test_oom_error_gives_actionable_hint(self):
        from ai_eyes_mcp.server import _tool_error
        err = _tool_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        m = str(err).lower()
        assert "out of memory" in m
        assert "float16" in m or "cpu" in m, "OOM error must tell the caller what to try"

    def test_generic_error_is_sanitized_and_points_to_logs(self):
        from ai_eyes_mcp.server import _tool_error
        err = _tool_error(RuntimeError("internal detail C:/secret/path/module.py:42 boom"))
        m = str(err)
        assert "C:/secret/path" not in m, "raw internals must not leak to the caller"
        assert "AI_EYES_LOG_LEVEL" in m, "an unexpected error should point to how to debug it"

    def test_not_found_error_stays_actionable(self):
        from ai_eyes_mcp.server import _tool_error
        err = _tool_error(FileNotFoundError("Image not found: /x/y.png"))
        m = str(err).lower()
        assert "not found" in m
        assert "path" in m  # hint: check the path

    def test_invalid_input_error_passthrough(self):
        from ai_eyes_mcp.server import _tool_error
        err = _tool_error(ValueError("Query must not be empty"))
        assert "empty" in str(err).lower()
