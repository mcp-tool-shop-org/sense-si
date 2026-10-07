"""CI-safe coverage of the forward wiring and the tool error branches.

No weights. The dogfood suite is what runs the real model.
"""

from __future__ import annotations

import importlib
import logging
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from fastmcp.exceptions import ToolError
from PIL import Image, UnidentifiedImageError

from ai_eyes_mcp.engine import (
    PINNED_MODEL_REVISION,
    Score,
    SigLIPEngine,
    _TokenIdConfigWarningFilter,
    assert_cold_status,
    display_round,
    round_margin,
    round_pair_preserving_order,
    round_preserving_gt,
    validate_model_revision,
)

_PIN = PINNED_MODEL_REVISION


class _Tensors(dict):
    def to(self, device):
        return self


class _Proc:
    def __init__(self):
        self.tokenizer = None

    def __call__(self, **_kwargs):
        return _Tensors()


class _Out:
    def __init__(self):
        self.logits_per_image = torch.zeros(1, 16)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.config = type(
            "C",
            (),
            {
                "_commit_hash": _PIN,
                "text_config": type("T", (), {"max_position_embeddings": 64})(),
            },
        )()
        self.cast = None

    def eval(self):
        return self

    def to(self, device):
        return self

    def half(self):
        self.cast = "float16"
        return self

    def bfloat16(self):
        self.cast = "bfloat16"
        return self

    def __call__(self, **_kwargs):
        return _Out()

    def get_image_features(self, **_kwargs):
        return torch.ones(1, 4)


def _wired(**kwargs) -> SigLIPEngine:
    engine = SigLIPEngine(**kwargs)
    engine._processor = _Proc()
    engine._model = _Model()
    engine._resolved_revision = _PIN
    engine._load_image = lambda _path: object()
    return engine


def _ready(monkeypatch):
    from ai_eyes_mcp import server

    monkeypatch.setattr(type(server.engine), "loaded", property(lambda self: True))
    monkeypatch.setattr(server.engine, "_resolved_revision", _PIN, raising=False)
    return server


def test_score_multi_dedupes_and_refuses_an_empty_list():
    engine = _wired()
    scores = engine.score_multi("img.png", ["a cat", "a cat", "a bus"])
    assert list(scores) == ["a cat", "a bus"]
    assert all(isinstance(value, Score) for value in scores.values())
    with pytest.raises(ValueError, match="At least one query"):
        engine.score_multi("img.png", [])


def test_score_batch_logs_progress_and_refuses_an_empty_list(caplog):
    engine = _wired()
    paths = [f"img-{i}.png" for i in range(26)]
    # configure_logging attaches a stderr handler and turns propagation off,
    # so the root caplog handler never sees this record.
    log = logging.getLogger("ai_eyes_mcp")
    log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="ai_eyes_mcp"):
            scores = engine.score_batch(paths, "a knight")
    finally:
        log.removeHandler(caplog.handler)
    assert len(scores) == 26
    assert any("Batch progress: 25/26" in message for message in caplog.messages)
    with pytest.raises(ValueError, match="At least one image"):
        engine.score_batch([], "a knight")


def test_embed_cache_hit_is_a_private_copy_and_a_missing_file_is_uncached(tmp_path):
    engine = _wired()
    image = tmp_path / "one.png"
    image.write_bytes(b"not-a-real-image")
    engine._embed_image_uncached = lambda _path: np.ones(4, dtype=np.float32)
    first = engine.embed_image(str(image))
    second = engine.embed_image(str(image))
    assert np.array_equal(first, second)
    second[0] = 0
    assert first[0] == 1

    missing = engine.embed_image(str(tmp_path / "gone.png"))
    assert missing.shape == (4,)


def test_embed_cache_can_be_turned_off(monkeypatch, tmp_path):
    import ai_eyes_mcp.engine as engine_module

    monkeypatch.setattr(engine_module, "EMBED_CACHE_MAX", 0)
    engine = _wired()
    image = tmp_path / "one.png"
    image.write_bytes(b"x")
    calls = {"n": 0}

    def uncached(_path):
        calls["n"] += 1
        return np.ones(4, dtype=np.float32)

    engine._embed_image_uncached = uncached
    engine.embed_image(str(image))
    engine.embed_image(str(image))
    assert calls["n"] == 2
    assert engine._embed_cache == {}


def test_embed_accepts_a_pooled_output_object():
    engine = _wired()

    class _Pooled:
        pooler_output = torch.ones(1, 4)

    engine._model.get_image_features = lambda **_kwargs: _Pooled()
    vector = engine._embed_image_uncached("img.png")
    assert vector.shape == (4,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_similarities_need_a_candidate_and_verify_needs_a_real_contrast():
    engine = _wired()
    engine.embed_image = lambda _path: np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="At least one candidate"):
        engine.similarities_to_reference("ref.png", [])
    sims = engine.similarities_to_reference("ref.png", ["a.png", "b.png"])
    assert sims == [1.0, 1.0]

    engine.score_multi = lambda _path, queries: {query: Score(0.2) for query in queries}
    with pytest.raises(ValueError, match="at least one contrast"):
        engine.verify("img.png", "a knight", [])
    with pytest.raises(ValueError, match="differs from the target"):
        engine.verify("img.png", "a knight", ["a knight"])


def test_selftest_orders_the_bundled_pairs():
    engine = _wired()

    def score(_path, query):
        if query == "a cheetah":
            return 0.9
        if query == "a bus":
            return 0.1
        if "knight.png" in _path:
            return 0.8
        return 0.2

    def compare(left, right):
        return 1.0 if left == right else 0.2

    engine.score = score
    engine.compare = compare
    report = engine.selftest()
    assert report["passed"] is True
    assert [check["name"] for check in report["checks"]] == [
        "armed_vs_unarmed",
        "photo_true_vs_wrong_label",
        "self_vs_cross_similarity",
    ]
    assert report["revision"] == _PIN


def test_load_image_refuses_a_directory_a_missing_file_and_a_bad_file(monkeypatch, tmp_path):
    engine = SigLIPEngine()
    folder = tmp_path / "dir"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="not a file"):
        engine._load_image(str(folder))
    with pytest.raises(FileNotFoundError, match="not found"):
        engine._load_image(str(tmp_path / "missing.png"))

    image = tmp_path / "bad.png"
    image.write_bytes(b"x")

    def bomb(_path):
        raise Image.DecompressionBombError("too big")

    monkeypatch.setattr(Image, "open", bomb)
    with pytest.raises(ValueError, match="decompression bomb"):
        engine._load_image(str(image))

    def corrupt(_path):
        raise UnidentifiedImageError("nope")

    monkeypatch.setattr(Image, "open", corrupt)
    with pytest.raises(ValueError, match="Cannot open image"):
        engine._load_image(str(image))


def test_load_applies_dtype_records_cache_dir_and_clears_a_failed_load(monkeypatch, tmp_path):
    import transformers

    made: dict = {}

    def fake_model(*_args, **kwargs):
        made["kwargs"] = kwargs
        return _Model()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Proc())
    log = logging.getLogger("transformers")
    handler = logging.NullHandler()
    log.addHandler(handler)
    try:
        half = SigLIPEngine(dtype="float16", cache_dir=str(tmp_path))
        half._ensure_loaded()
        assert half._model.cast == "float16"
        assert made["kwargs"]["cache_dir"] == str(tmp_path)

        bf = SigLIPEngine(dtype="bfloat16")
        bf._ensure_loaded()
        assert bf._model.cast == "bfloat16"

        unknown = SigLIPEngine(dtype="fp32")
        unknown._ensure_loaded()
        assert unknown._model.cast is None
    finally:
        log.removeHandler(handler)

    def boom(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", boom)
    failed = SigLIPEngine()
    with pytest.raises(RuntimeError, match="offline"):
        failed._ensure_loaded()
    assert failed._model is None
    assert failed._processor is None


def test_truncation_check_never_raises():
    engine = _wired()

    class _Tok:
        def __call__(self, query, truncation, add_special_tokens):
            count = 80 if "long" in query else 3
            return {"input_ids": [0] * count}

    engine._processor.tokenizer = _Tok()
    assert engine._warn_if_truncated("a long query") is True
    assert engine._warn_if_truncated("a cat") is False

    engine._processor.tokenizer = None
    assert engine._warn_if_truncated("a cat") is False

    engine._processor.tokenizer = _Tok()
    engine._model.config = type("C", (), {"text_config": None})()
    assert engine._warn_if_truncated("a long query") is False

    class _BadTok:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("tokenizer broke")

    engine._model = _Model()
    engine._processor.tokenizer = _BadTok()
    assert engine._warn_if_truncated("a cat") is False


def test_token_id_filter_drops_only_the_benign_config_warning():
    filt = _TokenIdConfigWarningFilter()

    def record(text: str) -> logging.LogRecord:
        return logging.LogRecord("transformers", logging.WARNING, __file__, 1, text, (), None)

    assert filt.filter(record("bos_token_id must be set")) is False
    assert filt.filter(record("eos_token_id must be set")) is False
    assert filt.filter(record("the weights failed to load")) is True


def test_eager_load_runs_at_construction(monkeypatch):
    monkeypatch.setenv("AI_EYES_EAGER_LOAD", "1")
    monkeypatch.setattr(SigLIPEngine, "_ensure_loaded", lambda self: setattr(self, "eager", True))
    engine = SigLIPEngine()
    assert engine.eager is True


def test_cold_status_rejects_the_wrong_model_and_revision_edges():
    status = SigLIPEngine().status()
    status["model_id"] = "other/model"
    with pytest.raises(AssertionError, match="model_id"):
        assert_cold_status(status)
    assert validate_model_revision(None) == _PIN
    with pytest.raises(ValueError, match="40-character hex"):
        validate_model_revision(12345)


def test_rounding_falls_back_to_the_raw_value_when_every_digit_flips(monkeypatch):
    import ai_eyes_mcp.engine as engine_module

    monkeypatch.setattr(engine_module, "display_round", lambda value, ndigits=4: 0.0)
    assert round_preserving_gt(0.2, 0.1) == 0.2
    assert round_pair_preserving_order(0.2, 0.1) == (0.2, 0.1)
    assert round_margin(0.5) == 0.5
    # The real helper still refuses to print a non-zero as zero.
    assert display_round(1e-8) != 0.0


def test_bad_env_defaults_fall_back_and_are_restored(monkeypatch):
    import ai_eyes_mcp.engine as engine_module

    monkeypatch.setenv("AI_EYES_EMBED_CACHE", "nope")
    monkeypatch.setenv("AI_EYES_DEFAULT_THRESHOLD", "nope")
    try:
        importlib.reload(engine_module)
        assert engine_module.EMBED_CACHE_MAX == 64
        assert engine_module.DEFAULT_THRESHOLD == 0.02
        monkeypatch.setenv("AI_EYES_DEFAULT_THRESHOLD", "1.5")
        importlib.reload(engine_module)
        assert engine_module.DEFAULT_THRESHOLD == 0.02
    finally:
        monkeypatch.delenv("AI_EYES_EMBED_CACHE", raising=False)
        monkeypatch.delenv("AI_EYES_DEFAULT_THRESHOLD", raising=False)
        importlib.reload(engine_module)
        # Reload replaces SigLIPEngine. The server still holds the instance
        # it built at import, and later tests patch the class that exists now.
        import ai_eyes_mcp.server as server_module

        server_module.SigLIPEngine = engine_module.SigLIPEngine
        server_module.engine = server_module._construct_engine()


def test_module_entry_point_calls_server_main(monkeypatch):
    from ai_eyes_mcp import server

    called = {}
    monkeypatch.setattr(server, "main", lambda: called.setdefault("ran", True))
    path = Path(server.__file__).resolve().parent / "__main__.py"
    runpy.run_path(str(path), run_name="__main__")
    assert called["ran"] is True


def test_construct_engine_exits_on_a_bad_revision(monkeypatch, capsys):
    from ai_eyes_mcp import server

    monkeypatch.setenv("AI_EYES_MODEL_REVISION", "main")
    with pytest.raises(SystemExit) as caught:
        server._construct_engine()
    assert caught.value.code == 1
    assert "Failed to initialize" in capsys.readouterr().err


def test_tools_refuse_bad_input_and_map_engine_errors(monkeypatch):
    server = _ready(monkeypatch)
    from ai_eyes_mcp.server import (
        eyes_selftest,
        image_classify,
        image_compare,
        image_contains,
        image_rank,
        image_score_batch,
        image_verify,
    )

    with pytest.raises(ToolError, match="threshold"):
        image_contains("img.png", "a cat", threshold=1.5)
    monkeypatch.setattr(server.engine, "score", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(server.engine, "query_truncated", lambda _q: False)
    with pytest.raises(ToolError, match="Scoring failed"):
        image_contains("img.png", "a cat")

    with pytest.raises(ToolError, match="At least one label"):
        image_classify("img.png", [])
    with pytest.raises(ToolError, match="Maximum 20"):
        image_classify("img.png", [f"label-{i}" for i in range(21)])
    monkeypatch.setattr(
        server.engine, "score_multi", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("empty"))
    )
    with pytest.raises(ToolError, match="Invalid input"):
        image_classify("img.png", ["a cat"])

    monkeypatch.setattr(server.engine, "compare", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ToolError, match="Scoring failed"):
        image_compare("a.png", "b.png")

    with pytest.raises(ToolError, match="At least one candidate"):
        image_rank("ref.png", [])
    with pytest.raises(ToolError, match="Maximum 100"):
        image_rank("ref.png", [f"{i}.png" for i in range(101)])
    with pytest.raises(ToolError, match="k must be"):
        image_rank("ref.png", ["a.png"], k=0)
    monkeypatch.setattr(
        server.engine,
        "similarities_to_reference",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    with pytest.raises(ToolError, match="check the path"):
        image_rank("ref.png", ["a.png"])

    with pytest.raises(ToolError, match="At least one image"):
        image_score_batch([], "a cat")
    with pytest.raises(ToolError, match="Maximum 100"):
        image_score_batch([f"{i}.png" for i in range(101)], "a cat")
    with pytest.raises(ToolError, match="threshold"):
        image_score_batch(["a.png"], "a cat", threshold=2)
    monkeypatch.setattr(
        server.engine, "_encode_text", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("empty"))
    )
    with pytest.raises(ToolError, match="Invalid input"):
        image_score_batch(["a.png"], "a cat")

    monkeypatch.setattr(server.engine, "verify", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ToolError, match="Scoring failed"):
        image_verify("img.png", "a knight", ["a cook"])
    monkeypatch.setattr(server.engine, "selftest", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ToolError, match="Scoring failed"):
        eyes_selftest()


def test_classify_keeps_raw_scores_when_rounding_cannot_preserve_the_best(monkeypatch):
    server = _ready(monkeypatch)
    from ai_eyes_mcp import server as server_module
    from ai_eyes_mcp.server import image_classify

    monkeypatch.setattr(
        server.engine,
        "score_multi",
        lambda _path, labels: {"a cat": 0.9, "a bus": 0.1},
    )
    monkeypatch.setattr(server.engine, "query_truncated", lambda _q: False)

    def inverted(value, _digits=4):
        return 0.0 if value > 0.5 else 1.0

    monkeypatch.setattr(server_module, "display_round", inverted)
    result = image_classify("img.png", ["a cat", "a bus"])
    assert result["best"] == "a cat"
    assert result["scores"]["a cat"] == 0.9


def test_compare_and_rank_use_caller_baselines(monkeypatch):
    server = _ready(monkeypatch)
    from ai_eyes_mcp.server import image_compare, image_rank

    def compare(left, right):
        if left.endswith("a.png"):
            return 0.9
        return 0.1

    monkeypatch.setattr(server.engine, "compare", compare)
    compared = image_compare("a.png", "b.png", baselines=[["c.png", "d.png"]])
    assert compared["incomplete"] is False
    assert compared["baseline_max"] == 0.1

    monkeypatch.setattr(server.engine, "compare", lambda _left, _right: 0.5)
    monkeypatch.setattr(server.engine, "similarities_to_reference", lambda _ref, paths: [0.2, 0.8])
    ranked = image_rank("ref.png", ["low.png", "high.png"], baselines=[["c.png", "d.png"]])
    assert ranked["nothing_close"] is False
    assert len(ranked["matches"]) == 1
    assert ranked["matches"][0]["separated"] is True

    monkeypatch.setattr(server.engine, "similarities_to_reference", lambda _ref, paths: [0.0])
    quiet = image_rank("ref.png", ["low.png"], baselines=[["c.png", "d.png"]])
    assert quiet["nothing_close"] is True
    assert quiet["matches"] == []


def test_score_batch_sorts_errors_and_names_a_best_path(monkeypatch):
    server = _ready(monkeypatch)
    from ai_eyes_mcp.server import image_score_batch

    monkeypatch.setattr(server.engine, "_encode_text", lambda _query: {})
    monkeypatch.setattr(server.engine, "query_truncated", lambda _query: True)

    def score_one(path, _text, truncated=False):
        if "missing" in path:
            raise FileNotFoundError(path)
        if "bad" in path:
            raise ValueError("corrupt")
        if "boom" in path:
            raise RuntimeError("nope")
        return Score(0.5, truncated=truncated, revision=_PIN)

    monkeypatch.setattr(server.engine, "_score_with_text_inputs", score_one)
    mixed = image_score_batch(["ok.png", "missing.png", "bad.png", "boom.png"], "a cat")
    assert mixed["errors"] == 3
    assert mixed["scored"] == 1
    assert mixed["truncated"] is True
    assert mixed["best_path"].endswith("ok.png")
    assert {item["error"] for item in mixed["error_details"]} == {
        "not found",
        "invalid image",
        "scoring failed",
    }

    empty = image_score_batch(["missing.png"], "a cat")
    assert "best_path" not in empty
    assert empty["error_details"][0]["error"] == "not found"


def test_server_main_starts_the_stdio_server(monkeypatch):
    from ai_eyes_mcp import server

    monkeypatch.setattr(server.mcp, "run", lambda: None)
    server.main()
