"""Kev stays stubbed in CI. The live checkpoint is opt-in."""

import json
import os
import sys

import pytest

from decisions import DecisionRequest, DecisionsError, NoulQuestion
from decisions.client import PINNED_MODEL
from decisions.kev import (
    DEFAULT_MEMORY_FRACTION,
    KEV_BASE,
    KEV_BASE_REVISION,
    KEV_JUDGMENT,
    KEV_LIBRARY_REVISION,
    KEV_MODEL,
    KEV_REVISION,
    KEV_UNANSWERED_BAND,
    apply_memory_cap,
    create_kev_client,
    fit_lomo,
    resolve_memory_fraction,
    serve_capped,
)


def test_the_pin_is_a_revision_hash_and_not_jevs_band():
    assert KEV_REVISION == "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
    assert len(KEV_REVISION) == 40
    assert KEV_MODEL.endswith(KEV_REVISION)
    assert KEV_BASE == "Qwen/Qwen3.5-4B-Base"
    assert KEV_BASE_REVISION == "1001bb4d826a52d1f399e183466143f4da7b741b"
    assert KEV_LIBRARY_REVISION == "5e42a7a03f28134853dd3ff77461457e921e5ec1"
    assert KEV_UNANSWERED_BAND == (0.0, 1.0)
    assert KEV_JUDGMENT == "insufficient-evidence"
    assert DEFAULT_MEMORY_FRACTION == 0.82
    source = open(os.path.join(os.path.dirname(__file__), "..", "src", "decisions", "kev.py"), encoding="utf-8").read()
    assert "0.35" not in source
    assert "0.65" not in source
    assert "jev-1.13" not in source
    assert "openjev" not in source.lower()


def test_memory_fraction_defaults_and_refuses_a_bad_cap(monkeypatch):
    monkeypatch.delenv("KEV_MEMORY_FRACTION", raising=False)
    assert resolve_memory_fraction(None) == 0.82
    assert resolve_memory_fraction(0.5) == 0.5
    monkeypatch.setenv("KEV_MEMORY_FRACTION", "0.7")
    assert resolve_memory_fraction(None) == 0.7
    with pytest.raises(DecisionsError, match="outside"):
        resolve_memory_fraction(0.0)
    with pytest.raises(DecisionsError, match="outside"):
        resolve_memory_fraction(1.1)
    monkeypatch.setenv("KEV_MEMORY_FRACTION", "lots")
    with pytest.raises(DecisionsError, match="not a fraction"):
        resolve_memory_fraction(None)
    monkeypatch.delenv("KEV_MEMORY_FRACTION", raising=False)
    with pytest.raises(DecisionsError, match="not a number"):
        resolve_memory_fraction(True)
    with pytest.raises(DecisionsError, match="not a number"):
        resolve_memory_fraction("0.5")
    with pytest.raises(DecisionsError, match="outside"):
        resolve_memory_fraction(float("nan"))
    assert resolve_memory_fraction(1) == 1.0


def test_the_cap_is_applied_only_when_cuda_is_there():
    seen = {}

    def setter(fraction, device):
        seen["fraction"] = fraction
        seen["device"] = device

    assert apply_memory_cap(cuda_available=False, set_fraction=setter) == 0.82
    assert seen == {}
    assert apply_memory_cap(0.7, cuda_available=True, set_fraction=setter) == 0.7
    assert seen == {"fraction": 0.7, "device": 0}


def test_default_cuda_and_setter_do_not_need_a_gpu(monkeypatch):
    import decisions.kev as kev

    class Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            Cuda.seen = (fraction, device)

    fake = type("Torch", (), {"cuda": Cuda})()
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert kev._default_cuda() is False
    kev._default_set_fraction(0.82, 0)
    assert Cuda.seen == (0.82, 0)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert kev._default_cuda() is False


def test_an_uncapped_call_uses_the_fake_torch_and_not_the_card(monkeypatch):
    class Cuda:
        available = True
        seen = None

        @staticmethod
        def is_available():
            return Cuda.available

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            Cuda.seen = (fraction, device)

    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {"cuda": Cuda})())
    assert apply_memory_cap() == 0.82
    assert Cuda.seen == (0.82, 0)
    Cuda.available = False
    Cuda.seen = None
    assert apply_memory_cap() == 0.82
    assert Cuda.seen is None


def test_serve_refuses_an_unpinned_run_and_launches_the_pin(monkeypatch):
    import runpy

    recorded = {}

    def run_module(name, run_name):
        recorded["name"] = name
        recorded["argv"] = sys.argv[:]

    monkeypatch.setattr(runpy, "run_module", run_module)
    with pytest.raises(DecisionsError, match="not pinned"):
        serve_capped(argv=["--run", "jaredpalmer/kev-9b@v1.0"], cuda_available=False)
    with pytest.raises(DecisionsError, match="not pinned"):
        serve_capped(argv=["--port", "8009"], cuda_available=False)
    with pytest.raises(DecisionsError, match="not pinned"):
        serve_capped(argv=["--run"], cuda_available=False)
    with pytest.raises(DecisionsError, match="beyond this machine"):
        serve_capped(argv=["--run", KEV_MODEL, "--host", "0.0.0.0"], cuda_available=False)
    with pytest.raises(DecisionsError, match="beyond this machine"):
        serve_capped(argv=["--run", KEV_MODEL, "--host=0.0.0.0"], cuda_available=False)
    with pytest.raises(DecisionsError, match="beyond this machine"):
        serve_capped(argv=["--run", KEV_MODEL, "--host"], cuda_available=False)
    launched = {}

    def runner(argv):
        launched["argv"] = list(argv)

    serve_capped(argv=["--run", KEV_MODEL, "--port", "8009"], cuda_available=False, runner=runner)
    assert launched["argv"] == ["--run", KEV_MODEL, "--port", "8009"]
    serve_capped(
        argv=["--run", KEV_MODEL, "--host=127.0.0.1"],
        cuda_available=False,
        runner=runner,
    )
    assert launched["argv"][-1] == "--host=127.0.0.1"
    applied = serve_capped(port=8009, cuda_available=False)
    assert applied == 0.82
    assert recorded["name"] == "kev.serve"
    assert recorded["argv"][0] == "kev.serve"
    assert "--run" in recorded["argv"]
    assert KEV_MODEL in recorded["argv"]
    assert "127.0.0.1" in recorded["argv"]
    assert sys.argv[0] != "kev.serve"


def test_client_posts_systemone_and_stamps_the_revision():
    sent = {}

    def fetch(url, init):
        sent["url"] = url
        sent["body"] = json.loads(init["body"])
        raw = json.dumps(
            {
                "model": KEV_MODEL,
                "answers": {"moves": {"type": "noul", "noul": 0.82}},
                "usage": {"input_tokens": 40},
            }
        )
        return 200, raw

    result = create_kev_client(base_url="http://127.0.0.1:8009", fetch_impl=fetch)(
        DecisionRequest(state={"phrase": "la"}, questions={"moves": NoulQuestion(instructions="moving?", true="yes", false="no")})
    )
    assert sent["url"] == "http://127.0.0.1:8009/v1/systemone"
    assert sent["body"]["model"] == KEV_MODEL
    assert sent["body"]["questions"]["moves"]["type"] == "noul"
    assert result.model == KEV_MODEL
    assert result.dated == KEV_REVISION
    assert result.answers["moves"].noul == 0.82
    assert result.cost is None
    assert result.input_tokens == 40


def test_client_accepts_a_base_that_already_names_the_path():
    def fetch(url, init):
        fetch.url = url
        return 200, json.dumps({"model": KEV_MODEL, "answers": {"moves": {"noul": 0.5}}})

    create_kev_client(base_url="http://127.0.0.1:9/v1/systemone", fetch_impl=fetch)(
        DecisionRequest(
            state="x",
            questions={"moves": NoulQuestion(instructions="q", true="a", false="b")},
            model=KEV_MODEL,
        )
    )
    assert fetch.url == "http://127.0.0.1:9/v1/systemone"


def test_client_refuses_a_third_model_a_bad_echo_and_a_bad_body():
    client = create_kev_client(fetch_impl=lambda url, init: (200, "{}"))
    with pytest.raises(DecisionsError, match="refused model"):
        client(DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")}, model="other"))
    with pytest.raises(DecisionsError, match="no questions"):
        client(DecisionRequest(state="x", questions={}))
    with pytest.raises(DecisionsError, match="HTTP 500"):
        create_kev_client(fetch_impl=lambda url, init: (500, ""))(
            DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
        )
    with pytest.raises(DecisionsError, match="non-JSON"):
        create_kev_client(fetch_impl=lambda url, init: (200, "nope"))(
            DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
        )
    with pytest.raises(DecisionsError, match="non-object"):
        create_kev_client(fetch_impl=lambda url, init: (200, "[]"))(
            DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
        )
    echoed = json.dumps({"model": PINNED_MODEL, "answers": {"m": {"noul": 0.2}}})
    with pytest.raises(DecisionsError, match="on the answer"):
        create_kev_client(fetch_impl=lambda url, init: (200, echoed))(
            DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
        )
    missing = json.dumps({"model": KEV_MODEL, "answers": []})
    with pytest.raises(DecisionsError, match="is missing"):
        create_kev_client(fetch_impl=lambda url, init: (200, missing))(
            DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
        )
    flagged = json.dumps({"model": KEV_MODEL, "answers": {"m": {"noul": 0.2}}, "usage": []})
    quiet = create_kev_client(fetch_impl=lambda url, init: (200, flagged))(
        DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
    )
    assert quiet.input_tokens is None


def test_client_uses_the_local_fetch_when_no_stub_is_passed(monkeypatch):
    import decisions.kev as kev

    def fetch(url, init):
        fetch.url = url
        body = {"model": KEV_MODEL, "answers": {"m": {"noul": 0.4}}, "usage": {"input_tokens": True}}
        return 200, json.dumps(body)

    monkeypatch.setattr(kev, "_default_fetch", fetch)
    result = create_kev_client()(
        DecisionRequest(state="x", questions={"m": NoulQuestion(instructions="q", true="a", false="b")})
    )
    assert fetch.url == "http://127.0.0.1:8009/v1/systemone"
    assert result.input_tokens is None


def test_default_fetch_reads_a_body_and_reports_a_down_server(monkeypatch):
    import decisions.kev as kev

    class Response:
        status = 200

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        urlopen.timeout = timeout
        return Response()

    monkeypatch.setattr(kev.urllib.request, "urlopen", urlopen)
    status, text = kev._default_fetch(
        "http://127.0.0.1:9/v1/systemone",
        {"body": "{}", "headers": {}, "timeout": 3},
    )
    assert status == 200
    assert urlopen.timeout == 3
    assert "ok" in text

    def http_err(request, timeout):
        raise kev.urllib.error.HTTPError("http://127.0.0.1", 503, "no", None, None)

    monkeypatch.setattr(kev.urllib.request, "urlopen", http_err)
    status, text = kev._default_fetch(
        "http://127.0.0.1:9/v1/systemone",
        {"body": "{}", "headers": {}, "timeout": 3},
    )
    assert status == 503
    assert text == ""

    def down(request, timeout):
        raise kev.urllib.error.URLError("down")

    monkeypatch.setattr(kev.urllib.request, "urlopen", down)
    with pytest.raises(DecisionsError, match="cannot reach"):
        kev._default_fetch(
            "http://127.0.0.1:9/v1/systemone",
            {"body": "{}", "headers": {}, "timeout": 3},
        )


def test_lomo_fits_a_temperature_per_mix_and_does_not_adopt():
    rows = []
    for mix, label in (("a", 1), ("b", 0)):
        for _ in range(4):
            rows.append({"mix": mix, "y": label, "p": 0.82})
    fitted = fit_lomo(rows)
    assert set(fitted["temperatures"]) == {"a", "b"}
    assert fitted["judgment"] == "insufficient-evidence"
    assert "band" not in fitted
    assert len(fitted["fold_brier"]) == 2
    with pytest.raises(DecisionsError, match="two mixes"):
        fit_lomo([{"mix": "only", "y": 1, "p": 0.8}, {"mix": "only", "y": 0, "p": 0.2}])
    with pytest.raises(DecisionsError, match="more than one row"):
        fit_lomo([{"mix": "only", "y": 1, "p": 0.8}])


def test_one_class_training_does_not_invent_a_temperature():
    rows = [
        {"mix": "marked", "y": 1, "p": 0.9},
        {"mix": "marked", "y": 0, "p": 0.2},
        {"mix": "pad", "y": 1, "p": 0.8},
    ]
    fitted = fit_lomo(rows)
    # Holding out the marked mix leaves a one-class pad, so that temperature stays 1.
    assert fitted["temperatures"]["marked"] == 1.0


@pytest.mark.kev
def test_live_checkpoint_answers_one_question():
    """Opt-in. The server must already be the capped pin. This test does not load the model."""
    if os.environ.get("KEV_LIVE") != "1":
        pytest.skip("opt-in: set KEV_LIVE=1 and point KEV_URL at a server that is already the capped pin")
    client = create_kev_client(base_url=os.environ.get("KEV_URL", "http://127.0.0.1:8009"))
    result = client(
        DecisionRequest(
            state={"note": "fixture"},
            questions={"moves": NoulQuestion(instructions="is it moving?", true="yes", false="no")},
            model=KEV_MODEL,
            dated=KEV_REVISION,
        )
    )
    assert result.dated == KEV_REVISION
    assert 0.0 <= result.answers["moves"].noul <= 1.0
