"""
Edge-case tests — error handling and boundary conditions.

Tests that the engine and tools fail gracefully with correct error types.
"""

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


FAKE_PATH = "F:/nonexistent/totally_fake_image.png"
NON_IMAGE_PATH = "F:/AI/ai-eyes-mcp/pyproject.toml"


# ===========================================================================
# Engine-level errors
# ===========================================================================

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

    def test_compare_missing_first(self, engine, knight_sword_front):
        with pytest.raises(FileNotFoundError):
            engine.compare(FAKE_PATH, knight_sword_front)

    def test_compare_missing_second(self, engine, knight_sword_front):
        with pytest.raises(FileNotFoundError):
            engine.compare(knight_sword_front, FAKE_PATH)

    def test_score_invalid_image_type(self, engine):
        """Passing a non-image file (e.g. pyproject.toml) should raise.

        PIL raises UnidentifiedImageError (subclass of OSError) for files
        that aren't valid images. The engine also raises ValueError for
        decompression bombs. Accept either.
        """
        with pytest.raises((ValueError, OSError)):
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

    def test_compare_missing_image_a(self, knight_sword_front):
        with pytest.raises(ToolError, match="not found"):
            image_compare(FAKE_PATH, knight_sword_front)

    def test_compare_missing_image_b(self, knight_sword_front):
        with pytest.raises(ToolError, match="not found"):
            image_compare(knight_sword_front, FAKE_PATH)

    def test_batch_empty_list(self):
        with pytest.raises(ToolError, match="At least one"):
            image_score_batch([], "anything")

    def test_batch_too_many(self):
        paths = [f"F:/fake_{i}.png" for i in range(101)]
        with pytest.raises(ToolError, match="Maximum 100"):
            image_score_batch(paths, "anything")

    def test_contains_invalid_image_type(self):
        """Passing a non-image file should raise ToolError."""
        with pytest.raises(ToolError):
            image_contains(NON_IMAGE_PATH, "anything")

    def test_contains_empty_query(self, knight_sword_front):
        """Empty query string should raise ToolError at tool level."""
        with pytest.raises(ToolError, match="empty"):
            image_contains(knight_sword_front, "")

    def test_contains_overlength_query(self, knight_sword_front):
        """Query exceeding 500 chars should raise ToolError at tool level."""
        long_query = "x" * 501
        with pytest.raises(ToolError, match="too long"):
            image_contains(knight_sword_front, long_query)

    def test_classify_empty_label_in_list(self, photo_cheetah):
        """A list containing an empty string label should raise ToolError.

        The engine deduplicates and validates each query via _validate_query,
        which rejects empty strings. The tool wraps that as ToolError.
        """
        with pytest.raises(ToolError, match="empty"):
            image_classify(photo_cheetah, ["cheetah", "", "bus"])

    def test_compare_non_image_as_image_a(self, knight_sword_front):
        """Passing a non-image file as image_a should raise ToolError."""
        with pytest.raises(ToolError):
            image_compare(NON_IMAGE_PATH, knight_sword_front)

    def test_compare_non_image_as_image_b(self, knight_sword_front):
        """Passing a non-image file as image_b should raise ToolError."""
        with pytest.raises(ToolError):
            image_compare(knight_sword_front, NON_IMAGE_PATH)

    def test_batch_non_image_captured_in_errors(self, knight_sword_front):
        """Non-image file in batch should be captured in error_details, not crash."""
        result = image_score_batch(
            [knight_sword_front, NON_IMAGE_PATH],
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

    def test_score_range(self, engine, knight_sword_front):
        """Sigmoid score should always be in [0, 1]."""
        for query in ["sword", "nothing", "xyzzy", "a", "the quick brown fox"]:
            score = engine.score(knight_sword_front, query)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for query '{query}'"

    def test_zero_threshold(self, knight_sword_front):
        """threshold=0.0 means any positive score triggers present=True."""
        result = image_contains(knight_sword_front, "pixel art knight", threshold=0.0)
        assert result["score"] > 0.0, "Knight should score above zero"
        assert result["present"] is True, (
            f"Score {result['score']} > threshold 0.0 but present is False"
        )

    def test_negative_threshold_rejected(self, knight_sword_front):
        """threshold=-1.0 should be rejected — sigmoid scores are [0, 1]."""
        with pytest.raises(ToolError, match="between 0.0 and 1.0"):
            image_contains(knight_sword_front, "pixel art knight", threshold=-1.0)

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
