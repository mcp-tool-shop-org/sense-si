"""CI-safe server tests — no GPU, no weights.

Stub the module-level engine so these exercise the tool layer only.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from fastmcp.exceptions import ToolError

from ai_eyes_mcp import EXPECTED_TOOL_NAMES
from ai_eyes_mcp.engine import PINNED_MODEL_REVISION
from ai_eyes_mcp.server import (
    image_classify,
    image_compare,
    image_contains,
    image_rank,
    image_score_batch,
    image_verify,
    eyes_selftest,
    eyes_status,
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


def test_every_model_backed_payload_names_the_resolved_revision(monkeypatch):
    """W5-SERVER-001 / W5-TESTS-002: ANY payload with a model number names the pin.

    scoring_guidance says treat a payload with no revision as incomplete.
    The list is EXPECTED_TOOL_NAMES so a new tool inherits the assertion
    instead of quietly escaping a hand-enumerated four.
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
    monkeypatch.setattr(server.engine, "compare", lambda a, b: 0.8)
    monkeypatch.setattr(
        server.engine, "similarities_to_reference", lambda ref, cands: [0.5] * len(cands)
    )
    monkeypatch.setattr(
        server.engine,
        "selftest",
        lambda: {
            "passed": True,
            "checks": [],
            "model_id": "google/siglip2-so400m-patch14-384",
            "revision": _PIN,
            "device": "cpu",
            "torch_version": "x",
            "transformers_version": "x",
        },
    )
    monkeypatch.setattr(
        server.engine,
        "status",
        lambda: {
            "model_id": "google/siglip2-so400m-patch14-384",
            "revision": _PIN,
            "device": "cpu",
            "loaded": True,
        },
    )

    callers = {
        "image_contains": lambda: image_contains("unused.png", "a query"),
        "image_classify": lambda: image_classify("unused.png", ["a", "b"]),
        "image_compare": lambda: image_compare("unused.png", "other.png"),
        "image_score_batch": lambda: image_score_batch(["unused.png"], "a query"),
        "image_verify": lambda: image_verify("unused.png", "a", ["b"]),
        "eyes_selftest": eyes_selftest,
        "eyes_status": eyes_status,
        "image_rank": lambda: image_rank("ref.png", ["a.png", "b.png"], k=1),
    }
    uncovered = EXPECTED_TOOL_NAMES - set(callers)
    extra = set(callers) - EXPECTED_TOOL_NAMES
    assert not uncovered and not extra, (
        f"gate dispatch must equal EXPECTED_TOOL_NAMES; "
        f"uncovered={sorted(uncovered)} extra={sorted(extra)}"
    )

    payloads = {name: fn() for name, fn in callers.items()}
    missing = [name for name, p in payloads.items() if "revision" not in p]
    assert not missing, f"payloads missing revision: {missing}"
    wrong = [name for name, p in payloads.items() if p.get("revision") != _PIN]
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


def test_image_rank_incomplete_without_baselines(monkeypatch):
    """F-W5-SERVER-003: without baselines, top-k is a measurement not a verdict."""
    from ai_eyes_mcp import server

    _stub_loaded(monkeypatch, server)
    monkeypatch.setattr(
        server.engine,
        "similarities_to_reference",
        lambda ref, cands: [0.9, 0.4],
    )
    r = image_rank("ref.png", ["hit.png", "miss.png"], k=2)
    assert r["incomplete"] is True
    assert r["revision"] == _PIN
    assert len(r["matches"]) == 2
    assert r["matches"][0]["path"].endswith("hit.png")
    assert r["matches"][0]["similarity"] >= r["matches"][1]["similarity"]


def test_image_rank_nothing_close_returns_empty_matches(monkeypatch):
    """A verb that always returns top-k is confident output with no signal."""
    from ai_eyes_mcp import server

    _stub_loaded(monkeypatch, server)
    monkeypatch.setattr(
        server.engine,
        "similarities_to_reference",
        lambda ref, cands: [0.70, 0.65],
    )
    monkeypatch.setattr(server.engine, "compare", lambda a, b: 0.80)
    r = image_rank(
        "ref.png",
        ["a.png", "b.png"],
        k=5,
        baselines=[["x.png", "y.png"]],
    )
    assert r["incomplete"] is False
    assert r["matches"] == []
    assert r["nothing_close"] is True
    assert r["revision"] == _PIN


def test_compare_incomplete_without_baselines(monkeypatch):
    """F-W5-SERVER-002: no caller baseline → no verdict, only the measurement."""
    from ai_eyes_mcp import server

    _stub_loaded(monkeypatch, server)
    monkeypatch.setattr(server.engine, "compare", lambda a, b: 0.82)
    r = image_compare("a.png", "b.png")
    assert r["incomplete"] is True
    assert r["separated"] is None
    assert "similarity" in r
    assert r["revision"] == _PIN


def test_compare_separated_uses_caller_baseline_not_a_fixed_band(monkeypatch):
    from ai_eyes_mcp import server

    _stub_loaded(monkeypatch, server)
    calls = []

    def fake_compare(a, b):
        calls.append(1)
        # First call is A-B, later calls are baseline pairs.
        return 0.95 if len(calls) == 1 else 0.80

    monkeypatch.setattr(server.engine, "compare", fake_compare)
    r = image_compare("a.png", "b.png", baselines=[["x.png", "y.png"]])
    assert r["incomplete"] is False
    assert r["separated"] is True
    assert r["baseline_max"] == 0.80
    assert r["margin"] > 0


# ---------------------------------------------------------------------------
# Baseline pair parsing — the guard validates the TYPE, not just the length
# ---------------------------------------------------------------------------


def test_baseline_pair_rejects_two_character_string():
    """A plausible typo must not become two silent garbage paths.

    ``len(item) != 2`` is False for the string ``"ab"``, which then gets
    indexed by CHARACTER — ``_parse_baseline_pairs(["ab"])`` returned
    ``[('<cwd>/a', '<cwd>/b')]``: two paths the caller never wrote, resolved
    against the server's working directory, fed straight into a comparison
    that decides ``separated``.
    """
    from ai_eyes_mcp.server import _parse_baseline_pairs

    with pytest.raises(ToolError, match="pair of image paths"):
        _parse_baseline_pairs(["ab"])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([b"xy"], id="bytes"),
        pytest.param([[1, 2]], id="non-string-elements"),
        pytest.param([{"a": 1, "b": 2}], id="dict-of-two"),
        pytest.param([{1, 2}], id="set-of-two"),
        pytest.param([["only-one"]], id="one-element"),
        pytest.param([["a", "b", "c"]], id="three-elements"),
        pytest.param([None], id="none"),
        pytest.param([["a", ""]], id="empty-path"),
    ],
)
def test_baseline_pair_rejects_wrong_shapes_with_actionable_error(bad):
    """Everything that is not a pair of path strings raises the SAME shape.

    Sweeping from the claim ("each baseline must be a pair of image paths")
    rather than from the one reported case: bytes, ints, dicts and sets all
    satisfied ``len(item) == 2`` or failed the length check late, and leaked a
    raw ``TypeError``/``KeyError`` instead. ``_parse_baseline_pairs`` runs
    BEFORE the try/except in both ``image_compare`` and ``image_rank``, so
    those escaped without ever passing through ``_tool_error``.
    """
    from ai_eyes_mcp.server import _parse_baseline_pairs

    with pytest.raises(ToolError):
        _parse_baseline_pairs(bad)


def test_baseline_pair_accepts_lists_and_tuples_of_two_paths():
    """The guard must not have been tightened into rejecting valid input."""
    from ai_eyes_mcp.server import _parse_baseline_pairs

    for good in ([["a.png", "b.png"]], [("a.png", "b.png")]):
        pairs = _parse_baseline_pairs(good)
        assert len(pairs) == 1
        assert pairs[0][0].endswith("a.png") and pairs[0][1].endswith("b.png")
    assert _parse_baseline_pairs(None) is None
    assert _parse_baseline_pairs([]) is None


def test_baseline_floor_is_not_re_embedded_on_every_call(monkeypatch, tmp_path):
    """F-W5-SERVER-004: the caller's baseline pairs are the same images
    every call, and were re-embedded every time.

    Runs the real ``engine.compare`` (numpy dot) over a stubbed embedder, so
    this counts actual forward passes without a GPU. Two ``image_compare``
    calls with one baseline pair touch four distinct images; the second call
    must add no new embeddings, and must return the identical payload.
    """
    import numpy as np

    from ai_eyes_mcp import server
    from ai_eyes_mcp.engine import SigLIPEngine

    _stub_loaded(monkeypatch, server)
    seen: list[str] = []
    vecs = {
        "a.png": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "b.png": np.array([0.9, 0.436, 0.0], dtype=np.float32),
        "x.png": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "y.png": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    for name in vecs:
        (tmp_path / name).write_bytes(b"stub")

    def fake(self, image_path):
        seen.append(os.path.basename(image_path))
        return vecs[os.path.basename(image_path)]

    monkeypatch.setattr(SigLIPEngine, "_embed_image_uncached", fake)
    server.engine._embed_cache.clear()

    def call():
        return image_compare(
            str(tmp_path / "a.png"),
            str(tmp_path / "b.png"),
            baselines=[[str(tmp_path / "x.png"), str(tmp_path / "y.png")]],
        )

    first = call()
    after_first = len(seen)
    second = call()

    assert after_first == 4, f"first call should embed 4 images, got {seen}"
    assert len(seen) == 4, (
        f"second call re-embedded {len(seen) - after_first} images that cannot "
        f"have changed: {seen[after_first:]}"
    )
    for field in ("similarity", "separated", "baseline_max", "margin"):
        assert first[field] == second[field], (
            f"{field} moved between identical calls: "
            f"{first[field]!r} -> {second[field]!r}"
        )
