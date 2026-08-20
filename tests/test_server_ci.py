"""CI-safe server tests — no GPU, no weights.

Stub the module-level engine so these exercise the tool layer only.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from fastmcp.exceptions import ToolError

from ai_eyes_mcp.engine import PINNED_MODEL_REVISION
from ai_eyes_mcp.server import (
    image_classify,
    image_contains,
    image_score_batch,
    image_verify,
    _tool_error,
)

_PIN = PINNED_MODEL_REVISION


def _stub_loaded(monkeypatch, server):
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))
    monkeypatch.setattr(server.engine, "_resolved_revision", _PIN, raising=False)


def test_classify_best_uses_raw_scores_not_rounded(monkeypatch):
    """W1-COORD-009: labels that round to the same 4dp must not flip `best`
    based on caller order. Ranking is the only thing image_classify does."""
    from ai_eyes_mcp import server

    raw = {"generic": 0.123441, "knight": 0.123449}
    monkeypatch.setattr(
        server.engine,
        "score_multi",
        lambda path, labels: {lab: raw[lab] for lab in labels},
    )
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: False)
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))

    first = image_classify("unused.png", ["generic", "knight"])
    swapped = image_classify("unused.png", ["knight", "generic"])
    assert first["best"] == "knight", (
        f"true best is knight (0.123449 > 0.123441); got {first['best']!r} "
        f"from caller order generic-first. payload={first}"
    )
    assert swapped["best"] == "knight"
    assert first["best"] == swapped["best"]
    # displayed scores stay consistent with that choice
    assert first["scores"]["knight"] >= first["scores"]["generic"]
    assert first["best_score"] == first["scores"]["knight"]


def test_contains_displayed_score_agrees_with_present(monkeypatch):
    """W1-SERVER-001: raw 0.02004 vs threshold 0.02 must not emit score 0.02."""
    from ai_eyes_mcp import server

    monkeypatch.setattr(server.engine, "score", lambda path, query: 0.02004)
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: False)
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))

    r = image_contains("unused.png", "a query", threshold=0.02)
    assert r["present"] is True
    assert r["score"] > r["threshold"], (
        f"payload contradicts itself: {r}"
    )


def test_batch_displayed_score_agrees_with_present(monkeypatch):
    from ai_eyes_mcp import server

    monkeypatch.setattr(
        server.engine, "_score_with_text_inputs", lambda path, text: 0.02004
    )
    monkeypatch.setattr(server.engine, "_encode_text", lambda query: {})
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: False)
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))

    r = image_score_batch(["unused.png"], "a query", threshold=0.02)
    assert r["results"][0]["present"] is True
    assert r["results"][0]["score"] > r["threshold"]


def test_contains_exposes_truncated_flag(monkeypatch):
    """W1-SERVER-003: truncation must be on the MCP payload, not only stderr."""
    from ai_eyes_mcp import server

    monkeypatch.setattr(server.engine, "score", lambda path, query: 0.1)
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: True)
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))

    r = image_contains("unused.png", "cat dog fox " * 39)
    assert r["truncated"] is True


def test_classify_exposes_truncated_flag(monkeypatch):
    from ai_eyes_mcp import server

    monkeypatch.setattr(
        server.engine, "score_multi", lambda path, labels: {labels[0]: 0.2, labels[1]: 0.1}
    )
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: query == "long")
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))

    r = image_classify("unused.png", ["short", "long"])
    assert r["truncated"] is True


def test_verify_exposes_truncated_flag(monkeypatch):
    from ai_eyes_mcp import server

    monkeypatch.setattr(
        server.engine,
        "verify",
        lambda path, target, alts: {
            "present": True,
            "target": target,
            "target_score": 0.9,
            "best_alternative": alts[0],
            "best_alternative_score": 0.1,
            "margin": 0.8,
            "confidence": "high",
        },
    )
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: True)
    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))

    r = image_verify("unused.png", "a knight", ["a cook"])
    assert r["truncated"] is True


def test_load_failure_does_not_forward_raw_exception(monkeypatch):
    """W1-SERVER-004: first-load errors go through the sanitizer."""
    from ai_eyes_mcp import server

    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: False))

    def boom():
        raise RuntimeError("cache E:\\secret\\hub.py huggingface.co/models")

    monkeypatch.setattr(server.engine, "_ensure_loaded", boom)

    with pytest.raises(ToolError) as ei:
        image_contains("unused.png", "a query")
    msg = str(ei.value)
    assert "E:\\secret" not in msg
    assert "huggingface.co" not in msg
    assert "hub.py" not in msg


def test_eager_server_import_does_not_leak_traceback():
    """W1-COORD-007: EAGER_LOAD failure at import must not dump site-packages frames."""
    env = os.environ.copy()
    env["AI_EYES_EAGER_LOAD"] = "1"
    env["AI_EYES_MODEL_ID"] = "definitely/not-a-real-model-xyz"
    env["HF_HUB_OFFLINE"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", "from ai_eyes_mcp.server import mcp"],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert r.returncode != 0
    leaked = (r.stdout or "") + (r.stderr or "")
    assert "site-packages" not in leaked, leaked[-800:]
    assert "Traceback (most recent call last)" not in leaked, leaked[-800:]


def test_eyes_status_includes_revision():
    """W1-COORD-008: a status payload that cannot name the weights is incomplete."""
    from ai_eyes_mcp.server import eyes_status

    r = eyes_status()
    assert "revision" in r
    assert r["revision"] == "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    assert len(r["revision"]) == 40


def test_scoring_payloads_name_the_resolved_revision(monkeypatch):
    """W5-SERVER-001 / W5-TESTS-002: every scoring payload names the pin.

    scoring_guidance tells callers to treat a payload with no revision as
    incomplete. This gate must fail if any scoring tool omits the field or
    echoes a constant that is not the resolved revision.
    """
    from ai_eyes_mcp import server

    _stub_loaded(monkeypatch, server)
    monkeypatch.setattr(server.engine, "score", lambda path, query: 0.1)
    monkeypatch.setattr(server.engine, "query_truncated", lambda query: False)
    monkeypatch.setattr(
        server.engine,
        "score_multi",
        lambda path, labels: {lab: 0.1 for lab in labels},
    )
    monkeypatch.setattr(
        server.engine,
        "verify",
        lambda path, target, alts: {
            "present": True,
            "target": target,
            "target_score": 0.9,
            "best_alternative": alts[0],
            "best_alternative_score": 0.1,
            "margin": 0.8,
            "confidence": "high",
        },
    )
    monkeypatch.setattr(server.engine, "_encode_text", lambda query: {})
    monkeypatch.setattr(server.engine, "_score_with_text_inputs", lambda path, text: 0.1)

    payloads = {
        "image_contains": image_contains("unused.png", "a query"),
        "image_classify": image_classify("unused.png", ["a", "b"]),
        "image_verify": image_verify("unused.png", "a", ["b"]),
        "image_score_batch": image_score_batch(["unused.png"], "a query"),
    }
    missing = [name for name, p in payloads.items() if "revision" not in p]
    assert not missing, f"scoring payloads missing revision: {missing}"
    wrong = [
        name
        for name, p in payloads.items()
        if p.get("revision") != _PIN
    ]
    assert not wrong, (
        f"revision is not the resolved SHA on {wrong}: "
        + ", ".join(f"{n}={payloads[n].get('revision')!r}" for n in wrong)
    )


def test_classify_docstring_does_not_equate_low_score_with_absence():
    """W1-SERVER-002: the MCP API must not claim a low score means 'not present'."""
    doc = image_classify.__doc__ or ""
    assert "NOT confidently present" not in doc
    assert "not confidently present" not in doc.lower()
    assert "phrasing" in doc.lower() or "relative" in doc.lower()
