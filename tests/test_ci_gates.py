"""CI-safe gates — no model, no GPU.

These tests exist so `pytest -m "not dogfood"` actually protects the
tool-registration contract and the dogfood split. They must fail on a
tree where verify.sh still asserts a five-tool set or CI still names
one test file instead of the marker selector.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SHIPPED_TOOLS = {
    "image_contains",
    "image_classify",
    "image_compare",
    "image_score_batch",
    "image_verify",
    "eyes_selftest",
    "eyes_status",
}


def test_canonical_constant_matches_shipped_seven():
    from ai_eyes_mcp import EXPECTED_TOOL_NAMES

    assert set(EXPECTED_TOOL_NAMES) == SHIPPED_TOOLS


def test_live_registration_matches_shipped_seven():
    """The server must register exactly the v1.1.0 seven-tool set."""
    from ai_eyes_mcp import EXPECTED_TOOL_NAMES
    from ai_eyes_mcp.server import mcp

    names = {t.name for t in asyncio.run(mcp._list_tools())}
    assert names == EXPECTED_TOOL_NAMES, (
        f"registered {sorted(names)} != shipped {sorted(EXPECTED_TOOL_NAMES)}"
    )


def test_verify_sh_asserts_the_shipped_seven():
    """verify.sh must not carry a stale inline tool set (W1-CITOOL-001).

    One source of truth: both the script and CI import EXPECTED_TOOL_NAMES
    rather than each maintaining a frozenset literal.
    """
    text = (REPO / "verify.sh").read_text(encoding="utf-8")
    assert "EXPECTED_TOOL_NAMES" in text, (
        "verify.sh must import EXPECTED_TOOL_NAMES; an inline expected = "
        "{...} drifted to five tools while seven register"
    )
    # The v1.0.0 five-tool literal must not be the assertion.
    five = (
        "{'image_contains', 'image_classify', 'image_compare', "
        "'image_score_batch', 'eyes_status'}"
    )
    assert five not in text.replace(" ", ""), (
        "verify.sh still contains the v1.0.0 five-tool expected set"
    )


def test_ci_yml_uses_canonical_tool_names():
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "EXPECTED_TOOL_NAMES" in text, (
        "ci.yml must import EXPECTED_TOOL_NAMES rather than a second hand-maintained set"
    )


def test_ci_and_verify_select_not_dogfood():
    """W1-CITOOL-003: both gates must use the dogfood marker, not a filename."""
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    verify = (REPO / "verify.sh").read_text(encoding="utf-8")
    selector = re.compile(r"pytest\s+-m\s+[\"']not dogfood[\"']")
    assert selector.search(ci), (
        "ci.yml must run `pytest -m \"not dogfood\"` so CI-safe tests in "
        "any file run, not only tests/test_edge_cases.py"
    )
    assert selector.search(verify), (
        "verify.sh must run `pytest -m \"not dogfood\"` so the local gate "
        "matches CI"
    )
    assert "pytest tests/test_edge_cases.py" not in ci, (
        "ci.yml still pins a single test file by name"
    )
    assert "pytest tests/test_edge_cases.py" not in verify, (
        "verify.sh still pins a single test file by name"
    )
