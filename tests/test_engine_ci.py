"""CI-safe engine tests — no GPU, no weights.

These pin measurement honesty (Class 1 / Class 2), the load lock, and
status field labeling. They stub the model/processor; they must go red
on unmodified engine.py.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from ai_eyes_mcp.engine import SigLIPEngine


# ---------------------------------------------------------------------------
# Class 1 — displayed numbers must imply the verdict
# ---------------------------------------------------------------------------


def _stub_score_multi(mapping):
    def fake(self, image_path, queries):
        return {q: mapping[q] for q in queries}
    return fake


def test_verify_tiny_margin_does_not_display_as_zero(monkeypatch):
    """W1-ENGINE-003: present true + margin 0.0 is a self-contradicting payload."""
    e = SigLIPEngine()
    monkeypatch.setattr(
        SigLIPEngine,
        "score_multi",
        lambda self, path, queries: {queries[0]: 0.50002, queries[1]: 0.50000},
    )
    r = e.verify("unused.png", "target", ["alt"])
    assert r["present"] is True
    assert r["margin"] > 0
    assert r["target_score"] > r["best_alternative_score"], (
        f"displayed scores contradict present=True: {r}"
    )


def test_verify_confidence_band_matches_displayed_margin(monkeypatch):
    """A 0.29999 gap is 'moderate'; displaying 0.3 would read as 'high'."""
    e = SigLIPEngine()
    monkeypatch.setattr(
        SigLIPEngine,
        "score_multi",
        lambda self, path, queries: {queries[0]: 0.5, queries[1]: 0.5 - 0.29999},
    )
    r = e.verify("unused.png", "target", ["alt"])
    assert "moderate" in r["confidence"]
    assert abs(r["margin"]) < 0.3, (
        f"displayed margin {r['margin']} would imply high, but confidence is moderate"
    )


def test_display_round_never_prints_zero_for_nonzero_sigmoid():
    """W1-COORD-003: selftest measured_b: 0 for a tiny sigmoid is a calibrated-looking lie."""
    from ai_eyes_mcp.engine import display_round

    rounded = display_round(1.23e-8, ndigits=5)
    assert rounded != 0
    assert rounded != 0.0
    assert float(rounded) > 0


# ---------------------------------------------------------------------------
# Class 2 — truncation is a returned fact, not a stderr aside
# ---------------------------------------------------------------------------


class _FakeTok:
    def __init__(self, n_tokens: int):
        self.n_tokens = n_tokens

    def __call__(self, query, truncation=False, add_special_tokens=True):
        return {"input_ids": list(range(self.n_tokens))}


class _FakeTextCfg:
    max_position_embeddings = 64


class _FakeCfg:
    text_config = _FakeTextCfg()
    max_position_embeddings = None


def _engine_with_tokenizer(n_tokens: int) -> SigLIPEngine:
    e = SigLIPEngine()
    e._processor = type("P", (), {"tokenizer": _FakeTok(n_tokens)})()
    e._model = type("M", (), {"config": _FakeCfg()})()
    return e


def test_query_truncated_true_when_over_encoder_limit():
    """W1-ENGINE-002: 80 tokens against a 64-position encoder is truncated."""
    e = _engine_with_tokenizer(80)
    assert e.query_truncated("unused") is True


def test_query_truncated_false_when_under_limit():
    e = _engine_with_tokenizer(10)
    assert e.query_truncated("a cheetah") is False


def test_char_cap_does_not_bound_tokens():
    """The 500-char cap still admits queries that tokenize past 64."""
    from ai_eyes_mcp.engine import MAX_QUERY_LENGTH

    q = "cat dog fox " * 39
    assert len(q) < MAX_QUERY_LENGTH
    e = _engine_with_tokenizer(119)
    assert e.query_truncated(q) is True


# ---------------------------------------------------------------------------
# W1-ENGINE-001 — load lock, not the forward lock
# ---------------------------------------------------------------------------


def test_ensure_loaded_single_flight(monkeypatch):
    """Two concurrent first-calls must load the model once, not twice."""
    e = SigLIPEngine()
    loads = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def to(self, device):
            time.sleep(0.15)
            loads.append(threading.current_thread().name)
            return self

    class FakeProc:
        pass

    def fake_processor(*a, **k):
        return FakeProc()

    def fake_model(*a, **k):
        return FakeModel()

    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(e._ensure_loaded) for _ in range(2)]
        for f in futs:
            f.result()

    assert len(loads) == 1, f"loaded {len(loads)} times (threads={loads})"
    assert e.loaded


def test_load_lock_is_not_the_forward_lock():
    """Holding the forward lock across a 10-20s load would stall every caller."""
    e = SigLIPEngine()
    assert hasattr(e, "_load_lock")
    assert e._load_lock is not e._forward_lock


# ---------------------------------------------------------------------------
# W1-ENGINE-004 — dtype is measured, not echoed
# ---------------------------------------------------------------------------


def test_status_dtype_reads_parameter_dtype_not_constructor_string():
    e = SigLIPEngine(dtype="fp16")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    e._model = M()
    e._processor = object()
    s = e.status()
    assert "float32" in str(s["dtype"]), (
        f"status dtype echoed constructor {e.dtype!r} instead of parameter dtype; got {s['dtype']!r}"
    )
    assert s["dtype"] != "fp16"


# ---------------------------------------------------------------------------
# W1-COORD-002 — vram_mb is this engine's load delta, not process-wide
# ---------------------------------------------------------------------------


def test_vram_mb_is_load_delta_not_current_process_allocation(monkeypatch):
    e = SigLIPEngine(device="cuda")
    mem = [100 * 1024 * 1024]  # 100 MiB

    def fake_allocated(*_a, **_k):
        return mem[0]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", fake_allocated)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def to(self, device):
            mem[0] = 500 * 1024 * 1024  # 500 MiB after load
            return self

    import transformers

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: FakeModel()
    )

    e._ensure_loaded()
    mem[0] = 900 * 1024 * 1024  # another consumer allocates after load
    s = e.status()
    assert "vram_mb" in s
    # Delta at load was 400 MiB; process-wide now would be 900.
    assert s["vram_mb"] == 400, (
        f"vram_mb should be the load delta (400), not process allocated "
        f"({s['vram_mb']})"
    )


# ---------------------------------------------------------------------------
# W1-COORD-008 — revision is a 40-char SHA, passed unconditionally
# ---------------------------------------------------------------------------

_BLESSED_SHA = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
_OTHER_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _fake_pretrained_pair(capture: dict):
    """Stub AutoModel/AutoProcessor.from_pretrained and record kwargs."""

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.config = type("C", (), {"_commit_hash": capture.get("resolved")})()

        def eval(self):
            return self

        def to(self, device):
            return self

    def fake_model(*_a, **k):
        capture["model_kwargs"] = k
        return FakeModel()

    def fake_proc(*_a, **k):
        capture["proc_kwargs"] = k
        return object()

    return fake_model, fake_proc


def test_default_revision_is_forty_char_hex():
    from ai_eyes_mcp.engine import DEFAULT_MODEL_REVISION, PINNED_MODEL_REVISION

    assert DEFAULT_MODEL_REVISION == _BLESSED_SHA
    assert PINNED_MODEL_REVISION == _BLESSED_SHA
    assert DEFAULT_MODEL_REVISION not in (None, "main", "")
    assert len(DEFAULT_MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_MODEL_REVISION)


def test_from_pretrained_always_receives_revision(monkeypatch):
    capture: dict = {}
    fake_model, fake_proc = _fake_pretrained_pair(capture)
    import transformers

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_proc)

    e = SigLIPEngine()
    e._ensure_loaded()
    assert "revision" in capture["model_kwargs"], (
        "from_pretrained was called without revision= — that is the float"
    )
    assert capture["model_kwargs"]["revision"] == _BLESSED_SHA
    assert capture["proc_kwargs"]["revision"] == _BLESSED_SHA


def test_valid_sha_override_reaches_from_pretrained(monkeypatch):
    capture: dict = {"resolved": _OTHER_SHA}
    fake_model, fake_proc = _fake_pretrained_pair(capture)
    import transformers

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_proc)

    e = SigLIPEngine(revision=_OTHER_SHA)
    e._ensure_loaded()
    assert capture["model_kwargs"]["revision"] == _OTHER_SHA
    assert e.status()["revision"] == _OTHER_SHA


@pytest.mark.parametrize("bad", ["main", "", "v1.0", "e8e4872", "MAIN"])
def test_non_sha_revision_raises_actionable_error(bad):
    with pytest.raises(ValueError, match=r"40-character hex|commit SHA") as ei:
        SigLIPEngine(revision=bad)
    msg = str(ei.value)
    assert "main" in msg.lower() or "SHA" in msg or "sha" in msg.lower()
    assert "AI_EYES_MODEL_REVISION" in msg or "revision" in msg.lower()


def test_env_main_raises_at_construction(monkeypatch):
    monkeypatch.setenv("AI_EYES_MODEL_REVISION", "main")
    with pytest.raises(ValueError, match=r"40-character hex|commit SHA"):
        SigLIPEngine()


def test_status_includes_revision_when_unloaded():
    e = SigLIPEngine()
    assert e.loaded is False
    s = e.status()
    assert s["revision"] == _BLESSED_SHA


# ---------------------------------------------------------------------------
# W1-COORD-006 / W1-COORD-005 / W1-ENGINE-005 — standalone contract
# ---------------------------------------------------------------------------

import os
import subprocess
import sys


def _engine_subprocess(code: str, env_extra: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_extra)
    # Isolate from the parent process's already-imported engine defaults.
    env.pop("AI_EYES_EAGER_LOAD", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_standalone_engine_honours_log_level():
    """W1-COORD-006: importing engine.py alone must honour AI_EYES_LOG_LEVEL."""
    r = _engine_subprocess(
        "import logging; from ai_eyes_mcp import engine; "
        "log = logging.getLogger('ai_eyes_mcp'); "
        "assert log.level == logging.DEBUG, log.level; "
        "assert log.handlers, 'no handlers on standalone path'; "
        "print('ok')",
        {"AI_EYES_LOG_LEVEL": "DEBUG"},
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_configure_logging_is_idempotent():
    from ai_eyes_mcp.engine import configure_logging

    log = __import__("logging").getLogger("ai_eyes_mcp")
    n = len(log.handlers)
    configure_logging()
    configure_logging()
    assert len(log.handlers) == n
    assert n >= 1


def test_standalone_engine_honours_model_id_env():
    """W1-COORD-005: SigLIPEngine() must read AI_EYES_MODEL_ID."""
    r = _engine_subprocess(
        "from ai_eyes_mcp.engine import SigLIPEngine; "
        "print(SigLIPEngine().model_id, end='')",
        {"AI_EYES_MODEL_ID": "TEST/should-appear"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == "TEST/should-appear", r.stdout


def test_score_object_behaves_as_float_and_carries_qualifiers():
    """W5-ENGINE-003: standalone score is a float that still names its weights."""
    from ai_eyes_mcp.engine import Score

    s = Score(0.02004, truncated=True, revision=_BLESSED_SHA)
    assert isinstance(s, float)
    assert s > 0.02
    assert float(s) == 0.02004
    assert s.truncated is True
    assert s.revision == _BLESSED_SHA


def test_engine_score_returns_score_with_qualifiers(monkeypatch):
    """score() itself, not a helper, must attach truncated + resolved revision."""
    from ai_eyes_mcp.engine import Score, SigLIPEngine

    e = SigLIPEngine()
    e._resolved_revision = _BLESSED_SHA
    monkeypatch.setattr(e, "_validate_query", lambda q: None)
    monkeypatch.setattr(e, "_load_image", lambda p: object())
    monkeypatch.setattr(e, "_warn_if_truncated", lambda q: True)

    class _Tensors(dict):
        def to(self, device):
            return self

    class _Proc:
        def __call__(self, **_k):
            return _Tensors()

    class _Out:
        logits_per_image = __import__("torch").tensor([[0.0]])  # sigmoid -> 0.5

    class _Model:
        def __call__(self, **_k):
            return _Out()

    e._processor = _Proc()
    e._model = _Model()

    s = e.score("unused.png", "a query")
    assert isinstance(s, Score)
    assert isinstance(s, float)
    assert s.truncated is True
    assert s.revision == _BLESSED_SHA
    assert abs(float(s) - 0.5) < 1e-6


def test_missing_weights_helper_does_not_treat_oom_as_absent():
    """W1-TESTS-001: CUDA OOM / ImportError must fail the suite, not skip-green."""
    from tests.conftest import is_missing_weights_error

    assert is_missing_weights_error(OSError("We couldn't find google/siglip2 in the cache"))
    assert is_missing_weights_error(OSError("Offline mode: file not found"))
    assert not is_missing_weights_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_missing_weights_error(ImportError("No module named transformers"))
    assert not is_missing_weights_error(RuntimeError("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"))


def test_out_of_range_threshold_falls_back_with_warning():
    """W1-ENGINE-005: 1.5 must not become the default; banana already falls back."""
    r = _engine_subprocess(
        "from ai_eyes_mcp.engine import DEFAULT_THRESHOLD; "
        "print(DEFAULT_THRESHOLD, end='')",
        {"AI_EYES_DEFAULT_THRESHOLD": "1.5"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == "0.02", r.stdout
    assert "AI_EYES_DEFAULT_THRESHOLD" in (r.stderr or "") or "threshold" in (r.stderr or "").lower()
