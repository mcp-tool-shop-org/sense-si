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

import pytest

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


def test_dev_extra_includes_hatchling():
    """W1-COORD-010: verify.sh runs `python -m build --no-isolation`, which
    needs hatchling on the import path. The documented `pip install -e ".[dev]"`
    path must provision it."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover — 3.10
        import tomli as tomllib
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(d.split(">=")[0].split("==")[0] == "hatchling" for d in dev), (
        f"dev extra must include hatchling so verify.sh --no-isolation can build; got {dev}"
    )


def test_cold_status_gate_passes_fresh_engine():
    from ai_eyes_mcp.engine import SigLIPEngine, assert_cold_status

    assert_cold_status(SigLIPEngine().status())


def test_cold_status_gate_rejects_loaded_lie():
    """W1-CITOOL-002: the gate must be able to go red if loaded lies."""
    from ai_eyes_mcp.engine import SigLIPEngine, assert_cold_status

    s = SigLIPEngine().status()
    s["loaded"] = True
    with pytest.raises(AssertionError, match="loaded"):
        assert_cold_status(s)


def test_cold_status_gate_rejects_wrong_revision():
    from ai_eyes_mcp.engine import SigLIPEngine, assert_cold_status

    s = SigLIPEngine().status()
    s["revision"] = "main"
    with pytest.raises(AssertionError, match="revision"):
        assert_cold_status(s)


def test_verify_and_ci_use_cold_status_gate():
    verify = (REPO / "verify.sh").read_text(encoding="utf-8")
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "assert_cold_status" in verify
    assert "assert_cold_status" in ci


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
