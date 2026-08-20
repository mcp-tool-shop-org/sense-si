"""
MCP tool-level dogfood tests — exercise the server tools directly.

These call the tool functions as the MCP layer would, verifying response
shapes, field names, and that the tools correctly wrap the engine.
"""

import asyncio

import pytest

from fastmcp.exceptions import ToolError

pytestmark = pytest.mark.dogfood

from ai_eyes_mcp import EXPECTED_TOOL_NAMES
from ai_eyes_mcp.server import (
    mcp,
    image_contains,
    image_classify,
    image_compare,
    image_score_batch,
    image_verify,
    eyes_selftest,
    eyes_status,
)


# ===========================================================================
# image_contains
# ===========================================================================

class TestImageContains:

    def test_returns_expected_shape(self, photo_cheetah):
        # Image-agnostic shape check — re-pointed to a vendored photo in Wave 0.
        result = image_contains(photo_cheetah, "a sword")
        assert "present" in result
        assert "score" in result
        assert "threshold" in result
        assert "query" in result
        assert isinstance(result["present"], bool)
        assert isinstance(result["score"], float)

    def test_present_when_armed(self, knight_sword_front):
        """Style-matched query triggers detection."""
        result = image_contains(knight_sword_front, "a knight with a sword and shield")
        assert result["present"] is True, f"Knight w/ sword not detected: {result}"

    def test_absent_when_unarmed(self, goblin_cook_front):
        result = image_contains(goblin_cook_front, "a knight with a sword and shield", threshold=0.01)
        assert result["present"] is False, f"Goblin cook false-positive for sword: {result}"

    def test_custom_threshold(self, knight_sword_front):
        """Same query, different threshold — score stays constant."""
        low = image_contains(knight_sword_front, "a knight with a sword and shield", threshold=0.001)
        high = image_contains(knight_sword_front, "a knight with a sword and shield", threshold=0.99)
        assert low["present"] is True
        assert high["present"] is False
        # Exact equality is safe here: both calls evaluate the same image+query
        # through the same engine, producing the same round(score, 4) value.
        # There is no randomness or non-determinism between the two calls.
        assert low["score"] == high["score"]  # same score, different threshold

    def test_query_echoed(self, photo_cheetah):
        result = image_contains(photo_cheetah, "a spotted cat")
        assert result["query"] == "a spotted cat"

    def test_threshold_echoed(self, photo_cheetah):
        result = image_contains(photo_cheetah, "test", threshold=0.42)
        assert result["threshold"] == 0.42

    def test_default_threshold_wired(self, photo_cheetah):
        """Calling without explicit threshold must use the 0.02 default.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        result = image_contains(photo_cheetah, "a sword")
        assert result["threshold"] == 0.02


# ===========================================================================
# image_classify
# ===========================================================================

class TestImageClassify:

    def test_returns_expected_shape(self, photo_cheetah):
        result = image_classify(photo_cheetah, ["cheetah", "bus"])
        assert "scores" in result
        assert "best" in result
        assert "best_score" in result
        assert isinstance(result["scores"], dict)
        assert len(result["scores"]) == 2

    def test_all_labels_present(self, photo_cheetah):
        labels = ["cat", "dog", "bird", "fish"]
        result = image_classify(photo_cheetah, labels)
        for label in labels:
            assert label in result["scores"]

    def test_best_matches_max(self, photo_lion):
        result = image_classify(photo_lion, ["lion", "cheetah", "bus"])
        max_label = max(result["scores"], key=result["scores"].get)
        assert result["best"] == max_label
        assert result["best_score"] == result["scores"][max_label]

    def test_cheetah_classified_correctly(self, photo_cheetah):
        result = image_classify(photo_cheetah, ["cheetah", "bus", "tower", "sword"])
        assert result["best"] == "cheetah"

    def test_scores_are_rounded(self, photo_bus):
        result = image_classify(photo_bus, ["bus", "car"])
        for score in result["scores"].values():
            # Check 4 decimal places max
            assert score == round(score, 4)


# ===========================================================================
# image_compare
# ===========================================================================

class TestImageCompare:

    def test_returns_expected_shape(self, photo_cheetah, photo_lion):
        # Image-agnostic shape check — re-pointed to vendored photos in Wave 0.
        result = image_compare(photo_cheetah, photo_lion)
        assert "similarity" in result
        assert "image_a" in result
        assert "image_b" in result
        assert isinstance(result["similarity"], float)

    def test_self_similarity(self, photo_cheetah):
        # Image-agnostic — re-pointed to a vendored photo in Wave 0.
        result = image_compare(photo_cheetah, photo_cheetah)
        assert result["similarity"] > 0.99

    def test_paths_echoed(self, photo_cheetah, photo_lion):
        result = image_compare(photo_cheetah, photo_lion)
        assert result["image_a"] == photo_cheetah
        assert result["image_b"] == photo_lion

    def test_symmetry(self, photo_cheetah, photo_bus):
        ab = image_compare(photo_cheetah, photo_bus)
        ba = image_compare(photo_bus, photo_cheetah)
        assert abs(ab["similarity"] - ba["similarity"]) < 0.001


# ===========================================================================
# image_score_batch
# ===========================================================================

class TestImageScoreBatch:

    def test_returns_expected_shape(self, photo_cheetah, photo_lion):
        # Image-agnostic shape check — re-pointed to vendored photos in Wave 0.
        result = image_score_batch(
            [photo_cheetah, photo_lion],
            "a sword",
        )
        assert "query" in result
        assert "total" in result
        assert "scored" in result
        assert "present" in result
        assert "absent" in result
        assert "errors" in result
        assert "results" in result
        assert "threshold" in result
        assert result["total"] == 2
        assert result["scored"] == 2

    def test_results_have_per_image_data(self, photo_cheetah, photo_bus):
        # Image-agnostic — re-pointed to vendored photos in Wave 0.
        result = image_score_batch(
            [photo_cheetah, photo_bus],
            "test query",
        )
        for r in result["results"]:
            assert "path" in r
            assert "score" in r
            assert "present" in r

    def test_present_count(self, knight_sword_front, goblin_cook_front, photo_bus):
        result = image_score_batch(
            [knight_sword_front, goblin_cook_front, photo_bus],
            "a knight with a sword and shield",
            threshold=0.02,
        )
        # Knight should be present with style-matched query, others not
        assert result["present"] >= 1, "At least the knight should be present"
        assert result["total"] == 3

    def test_handles_missing_in_batch(self, photo_cheetah):
        # Image-agnostic (missing-path handling) — re-pointed in Wave 0.
        result = image_score_batch(
            [photo_cheetah, "F:/nonexistent/fake_image.png"],
            "test",
        )
        assert result["errors"] == 1
        assert result["scored"] == 1
        assert result["error_details"] is not None
        # Path.resolve() normalizes separators on Windows (/ → \)
        from pathlib import Path
        assert Path(result["error_details"][0]["path"]) == Path("F:/nonexistent/fake_image.png").resolve()

    def test_query_echoed(self, photo_cheetah):
        result = image_score_batch([photo_cheetah], "a big cat")
        assert result["query"] == "a big cat"

    def test_threshold_flips_verdicts(self, knight_sword_front, goblin_cook_front, hero_bard_front):
        """Low threshold → all present; high threshold → none present."""
        paths = [knight_sword_front, goblin_cook_front, hero_bard_front]
        query = "a fantasy character"

        low = image_score_batch(paths, query, threshold=0.0001)
        high = image_score_batch(paths, query, threshold=0.9999)

        assert low["present"] == 3, f"All should be present at ultra-low threshold: {low}"
        assert low["absent"] == 0
        assert high["present"] == 0, f"None should be present at ultra-high threshold: {high}"
        assert high["absent"] == 3


# ===========================================================================
# eyes_status
# ===========================================================================

class TestEyesStatus:

    def test_returns_expected_shape(self):
        result = eyes_status()
        assert "model_id" in result
        assert "device" in result
        assert "loaded" in result
        assert "cache_dir" in result

    def test_reports_siglip2(self):
        result = eyes_status()
        assert "siglip2" in result["model_id"].lower()


# ===========================================================================
# image_verify — contrastive honest verdict
# ===========================================================================

class TestImageVerify:
    """Relative verdict: target vs caller-supplied alternatives -> decision +
    margin + confidence that DESCRIBES the measured gap (not invented precision)."""

    def test_present_high_confidence(self, knight_sword_front):
        r = image_verify(knight_sword_front, "a knight with a sword and shield",
                         ["a goblin cook", "a bard"])
        assert r["present"] is True
        assert r["margin"] > 0.3
        assert r["confidence"] == "high"
        assert r["target"] == "a knight with a sword and shield"

    def test_absent_when_target_loses(self, goblin_cook_front):
        r = image_verify(goblin_cook_front, "a knight with a sword", ["a goblin cook"])
        assert r["present"] is False  # the cook wins its own label
        assert r["margin"] < 0
        assert r["best_alternative"] == "a goblin cook"

    def test_low_confidence_on_near_tie(self, photo_lion):
        # "a lion" (~0.017) barely beats "a big cat" (~0.010) — present, but the
        # gap is tiny, so the verdict must read as low / inconclusive.
        r = image_verify(photo_lion, "a lion", ["a big cat"])
        assert abs(r["margin"]) < 0.1
        assert "low" in r["confidence"] or "inconclusive" in r["confidence"]

    def test_requires_alternatives(self, knight_sword_front):
        with pytest.raises(ToolError, match="RELATIVE"):
            image_verify(knight_sword_front, "a knight", [])

    def test_target_not_allowed_to_beat_itself(self, knight_sword_front):
        # Duplicating the target in alternatives must not let it "beat" itself.
        r = image_verify(knight_sword_front, "a knight with a sword and shield",
                         ["a knight with a sword and shield", "a goblin cook"])
        assert r["best_alternative"] != "a knight with a sword and shield"
        assert r["present"] is True

    def test_return_shape(self, knight_sword_front):
        r = image_verify(knight_sword_front, "a knight", ["a cook"])
        for k in ("present", "target", "target_score", "best_alternative",
                  "best_alternative_score", "margin", "confidence"):
            assert k in r


# ===========================================================================
# eyes_selftest — self-proving calibration check
# ===========================================================================

class TestEyesSelftest:
    """The instrument verifies itself on bundled reference images."""

    def test_selftest_passes_on_real_model(self):
        r = eyes_selftest()
        assert r["passed"] is True, f"selftest failed: {r}"
        assert len(r["checks"]) >= 3
        assert all(c["ok"] for c in r["checks"]), r["checks"]

    def test_selftest_reports_model_info(self):
        r = eyes_selftest()
        assert "siglip2" in r["model_id"].lower()
        assert r["device"]


# ===========================================================================
# MCP protocol end-to-end (in-memory client -> transport -> tool -> engine)
# ===========================================================================

class TestMCPProtocolE2E:
    """Exercise tools over the REAL MCP protocol via the FastMCP in-memory
    client — validates registration -> transport -> tool -> engine, not just the
    direct-function-call path the other tests use."""

    def _list(self):
        from fastmcp import Client

        async def go():
            async with Client(mcp) as c:
                return await c.list_tools()
        return asyncio.run(go())

    def _call(self, name, args):
        from fastmcp import Client

        async def go():
            async with Client(mcp) as c:
                return await c.call_tool(name, args)
        return asyncio.run(go())

    def test_list_tools_over_protocol(self):
        assert len({t.name for t in self._list()}) == len(EXPECTED_TOOL_NAMES)

    def test_image_contains_over_protocol(self, photo_cheetah):
        r = self._call("image_contains", {"image_path": photo_cheetah, "query": "a cheetah"})
        assert "present" in r.data and isinstance(r.data["score"], float)

    def test_image_verify_over_protocol(self, knight_sword_front):
        r = self._call("image_verify", {
            "image_path": knight_sword_front,
            "target": "a knight with a sword and shield",
            "alternatives": ["a goblin cook"],
        })
        assert r.data["present"] is True and "confidence" in r.data

    def test_eyes_selftest_over_protocol(self):
        r = self._call("eyes_selftest", {})
        assert r.data["passed"] is True

    def test_bad_input_over_protocol_raises_not_crashes(self):
        # A missing image path must surface as a ToolError over the protocol, not
        # crash the server. (Client.call_tool raises ToolError by default.)
        with pytest.raises(ToolError):
            self._call("image_contains", {"image_path": "F:/nonexistent/x.png", "query": "a cat"})


# ===========================================================================
# Determinism — it's an instrument; scores must be reproducible (bit-identical)
# ===========================================================================

class TestDeterminismInstrument:

    def test_score_bit_identical_across_3_calls(self, engine, photo_cheetah):
        q = "a cheetah running across the savanna"
        vals = [engine.score(photo_cheetah, q) for _ in range(3)]
        assert vals[0] == vals[1] == vals[2], f"nondeterministic score: {vals}"

    def test_verify_bit_identical_across_3_calls(self, engine, knight_sword_front):
        outs = [engine.verify(knight_sword_front, "a knight", ["a cook"]) for _ in range(3)]
        assert outs[0] == outs[1] == outs[2], f"nondeterministic verify: {outs}"


# ===========================================================================
# MCP tool registration
# ===========================================================================

class TestMCPToolRegistration:
    """Verify the FastMCP server has exactly the expected tools registered.

    This test runs without GPU — it only inspects the registration metadata.
    Uses the same async _list_tools() API as verify.sh.
    """

    EXPECTED_TOOLS = EXPECTED_TOOL_NAMES

    def _get_tools(self):
        return asyncio.run(mcp._list_tools())

    def test_tool_count(self):
        """Registered tools must match EXPECTED_TOOL_NAMES."""
        tools = self._get_tools()
        assert len(tools) == len(EXPECTED_TOOL_NAMES), (
            f"Expected {len(EXPECTED_TOOL_NAMES)} registered tools, got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    def test_tool_names_match(self):
        """Registered tool names must match the expected set."""
        tools = self._get_tools()
        names = {t.name for t in tools}
        assert names == self.EXPECTED_TOOLS, (
            f"Tool name mismatch.\n"
            f"  Expected: {self.EXPECTED_TOOLS}\n"
            f"  Got:      {names}\n"
            f"  Missing:  {self.EXPECTED_TOOLS - names}\n"
            f"  Extra:    {names - self.EXPECTED_TOOLS}"
        )
