import json

import pytest

from decisions import (
    PINNED_DATE,
    PINNED_MODEL,
    ChoiceQuestion,
    DecisionRequest,
    DecisionsError,
    NoulQuestion,
    ScoreQuestion,
    create_decisions_client,
    validate_answer,
)
from decisions.client import FetchResponse
from decisions.kev import KEV_MODEL


def _response(status: int, body: str, ok: bool | None = None) -> FetchResponse:
    return FetchResponse(ok=status < 400 if ok is None else ok, status=status, text=lambda: body)


def _client(fetch, retries=3):
    return create_decisions_client(api_key="sk-or-TEST", fetch_impl=fetch, retries=retries, sleep=lambda _s: None)


NOUL = {"moves": NoulQuestion(instructions="does the world move?", true="yes", false="no")}


def test_import_does_not_require_a_key():
    import decisions.client as client

    assert client.PINNED_MODEL == PINNED_MODEL
    assert not hasattr(client, "default_client")


def test_empty_key_builds_nothing():
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        raise AssertionError("fetch")

    with pytest.raises(DecisionsError, match="not set"):
        create_decisions_client(api_key="  ", fetch_impl=fetch)
    assert calls["n"] == 0


def test_posts_the_pin_and_stamps_the_date_on_the_answer():
    sent = {}

    def fetch(url, init):
        sent["url"] = url
        sent["body"] = json.loads(init["body"])
        sent["auth"] = init["headers"]["Authorization"]
        raw = json.dumps(
            {
                "model": PINNED_MODEL,
                "answers": {"moves": {"type": "noul", "noul": 0.94}},
                "usage": {"cost": 0.0001, "input_tokens": 1200},
            }
        )
        return _response(200, raw)

    result = _client(fetch)(
        DecisionRequest(state="transcript", questions={"moves": NOUL["moves"]})
    )
    assert sent["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert sent["auth"] == "Bearer sk-or-TEST"
    assert sent["body"]["model"] == PINNED_MODEL
    assert "dated" not in sent["body"]
    assert sent["body"]["questions"]["moves"]["type"] == "noul"
    assert result.model == PINNED_MODEL
    assert result.dated == PINNED_DATE
    assert result.answers["moves"].noul == 0.94
    assert result.cost == 0.0001
    assert result.input_tokens == 1200


@pytest.mark.parametrize(
    "model",
    ["jev-router", "typesafe/jev-router", PINNED_DATE, "typesafe/jev-1.13 ", "", KEV_MODEL],
)
def test_refuses_anything_but_the_pin_before_the_request(model):
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        raise AssertionError("fetch")

    client = _client(fetch)
    with pytest.raises(DecisionsError, match="refused"):
        client(DecisionRequest(model=model, state="s", questions=NOUL))
    assert calls["n"] == 0


def test_refuses_a_router_that_comes_back_on_the_answer():
    def fetch(url, init):
        raw = json.dumps({"model": "jev-router", "answers": {"moves": {"noul": 0.99}}})
        return _response(200, raw)

    with pytest.raises(DecisionsError, match="not replayable"):
        _client(fetch)(DecisionRequest(state="s", questions=NOUL))


def test_accepts_the_dated_snapshot_when_the_answer_echoes_it():
    def fetch(url, init):
        raw = json.dumps(
            {
                "model": f"typesafe/{PINNED_DATE}",
                "answers": {"moves": {"noul": 0.4}},
                "usage": {"cost": 0.0002},
            }
        )
        return _response(200, raw)

    result = _client(fetch)(DecisionRequest(state="s", questions=NOUL))
    assert result.model == PINNED_MODEL
    assert result.dated == PINNED_DATE
    assert result.answers["moves"].noul == 0.4
    assert result.cost == 0.0002


def test_refuses_a_different_snapshot_on_the_answer():
    def fetch(url, init):
        raw = json.dumps(
            {"model": "typesafe/jev-1.13-19990101", "answers": {"moves": {"noul": 0.4}}}
        )
        return _response(200, raw)

    with pytest.raises(DecisionsError, match="not replayable"):
        _client(fetch)(DecisionRequest(state="s", questions=NOUL))


def test_retries_429_and_5xx_but_not_other_4xx():
    calls = {"n": 0}

    def flaky(url, init):
        calls["n"] += 1
        if calls["n"] < 3:
            return _response(503, "nope")
        raw = json.dumps({"answers": {"moves": {"noul": 0.1}}})
        return _response(200, raw)

    result = _client(flaky)(DecisionRequest(state="s", questions=NOUL))
    assert result.answers["moves"].noul == 0.1
    assert calls["n"] == 3

    bad = {"n": 0}

    def terminal(url, init):
        bad["n"] += 1
        return _response(401, "nope")

    with pytest.raises(DecisionsError, match="HTTP 401") as caught:
        _client(terminal)(DecisionRequest(state="s", questions=NOUL))
    assert caught.value.status == 401
    assert bad["n"] == 1


def test_refuses_a_request_with_no_questions_before_calling():
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        raise AssertionError("fetch")

    with pytest.raises(DecisionsError, match="no questions"):
        _client(fetch)(DecisionRequest(state="s", questions={}))
    assert calls["n"] == 0


def test_rejects_a_malformed_answer_instead_of_guessing():
    def fetch(url, init):
        raw = json.dumps({"answers": {"moves": {"type": "noul", "noul": 1.7}}})
        return _response(200, raw)

    with pytest.raises(DecisionsError, match="noul"):
        _client(fetch)(DecisionRequest(state="s", questions=NOUL))


def test_choice_must_be_one_of_the_options():
    question = ChoiceQuestion(instructions="which?", criteria={"a": "A", "b": "B"})
    answer = validate_answer(
        question,
        {"type": "choice", "choice": "b", "confidence": 0.7, "probabilities": {"a": 0.3, "b": 0.7}},
        "x",
    )
    assert answer.choice == "b"
    assert answer.confidence == 0.7
    with pytest.raises(DecisionsError, match="not one of a, b"):
        validate_answer(question, {"type": "choice", "choice": "c"}, "x")


def test_score_must_be_numeric():
    question = ScoreQuestion(instructions="how much?", criteria=("low", "mid", "high"))
    answer = validate_answer(
        question,
        {"type": "score", "score": 1.8, "confidence": 0.9, "legend": {"0": "low"}},
        "x",
    )
    assert answer.score == 1.8
    with pytest.raises(DecisionsError, match="numeric"):
        validate_answer(question, {"type": "score"}, "x")


def test_questions_wire_the_shape_the_api_expects():
    assert NoulQuestion(instructions="x").to_wire() == {"type": "noul", "instructions": "x"}
    one_sided = NoulQuestion(instructions="x", false="no").to_wire()
    assert one_sided["criteria"] == {"true": None, "false": "no"}
    assert ScoreQuestion(instructions="how?", criteria=("low", "high")).to_wire() == {
        "type": "score",
        "instructions": "how?",
        "criteria": ["low", "high"],
    }


def test_answers_that_are_not_objects_or_not_probabilities_are_refused():
    with pytest.raises(DecisionsError, match="missing"):
        validate_answer(NOUL["moves"], None, "moves")
    with pytest.raises(DecisionsError, match="noul"):
        validate_answer(NOUL["moves"], {"noul": True}, "moves")
    question = ScoreQuestion(instructions="how much?", criteria=("low", "high"))
    with pytest.raises(DecisionsError, match="numeric"):
        validate_answer(question, {"score": True}, "x")
    with pytest.raises(DecisionsError, match="numeric"):
        validate_answer(question, {"score": float("inf")}, "x")
    score = validate_answer(
        question,
        {"score": 1, "confidence": "high", "legend": ["low"], "probabilities": [1]},
        "x",
    )
    assert score.confidence is None
    assert score.legend is None
    assert score.probabilities is None
    choice = ChoiceQuestion(instructions="which?", criteria={"a": "A"})
    parsed = validate_answer(choice, {"choice": "a", "confidence": True, "probabilities": "no"}, "x")
    assert parsed.choice == "a"
    assert parsed.confidence is None
    assert parsed.probabilities is None


def test_a_refused_date_never_calls():
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        raise AssertionError("fetch")

    with pytest.raises(DecisionsError, match="date stamp"):
        _client(fetch)(DecisionRequest(state="s", questions=NOUL, dated="nope"))
    assert calls["n"] == 0


@pytest.mark.parametrize(
    ("status", "hint"),
    [(402, "credits exhausted"), (413, "state too large"), (400, "not retried")],
)
def test_client_errors_are_not_retried(status, hint):
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        return _response(status, "nope")

    with pytest.raises(DecisionsError, match=f"HTTP {status}") as caught:
        _client(fetch)(DecisionRequest(state="s", questions=NOUL))
    assert hint in caught.value.hint
    assert caught.value.status == status
    assert calls["n"] == 1


def test_retries_stop_when_the_budget_is_spent():
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        return _response(429, "later")

    with pytest.raises(DecisionsError, match="HTTP 429") as caught:
        _client(fetch, retries=0)(DecisionRequest(state="s", questions=NOUL))
    assert calls["n"] == 1
    assert "retried" in caught.value.hint


def test_a_dead_transport_or_a_bad_body_is_not_turned_into_an_answer():
    def boom(url, init):
        raise OSError("down")

    with pytest.raises(DecisionsError, match="cannot reach"):
        _client(boom, retries=0)(DecisionRequest(state="s", questions=NOUL))

    def garbage(url, init):
        return _response(200, "not-json")

    with pytest.raises(DecisionsError, match="non-JSON"):
        _client(garbage, retries=0)(DecisionRequest(state="s", questions=NOUL))

    def array(url, init):
        return _response(200, "[]")

    with pytest.raises(DecisionsError, match="non-object"):
        _client(array, retries=0)(DecisionRequest(state="s", questions=NOUL))

    def answers(url, init):
        return _response(200, json.dumps({"answers": ["nope"]}))

    with pytest.raises(DecisionsError, match="missing"):
        _client(answers, retries=0)(DecisionRequest(state="s", questions=NOUL))


def test_usage_that_is_not_a_number_is_left_blank():
    def fetch(url, init):
        raw = json.dumps({"answers": {"moves": {"noul": 0.2}}, "usage": {"cost": True, "input_tokens": True}})
        return _response(200, raw)

    result = _client(fetch)(DecisionRequest(state="s", questions=NOUL))
    assert result.answers["moves"].noul == 0.2
    assert result.cost is None
    assert result.input_tokens is None


def test_the_default_transport_reads_a_response(monkeypatch):
    class Response:
        status = 200

        def read(self):
            return json.dumps({"answers": {"moves": {"noul": 0.3}}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        assert request.full_url == "https://example.test/decisions"
        assert timeout == 5
        assert request.data
        return Response()

    monkeypatch.setattr("decisions.client.urllib.request.urlopen", urlopen)
    client = create_decisions_client(
        api_key="sk-or-TEST",
        base_url="https://example.test",
        retries=0,
        attempt_timeout_s=5,
    )
    result = client(DecisionRequest(state="s", questions=NOUL))
    assert result.answers["moves"].noul == 0.3


def test_the_default_transport_reports_an_http_error(monkeypatch):
    import io
    import urllib.error

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            "https://example.test/decisions",
            402,
            "pay",
            hdrs=None,
            fp=io.BytesIO(b"broke"),
        )

    monkeypatch.setattr("decisions.client.urllib.request.urlopen", urlopen)
    client = create_decisions_client(
        api_key="sk-or-TEST",
        base_url="https://example.test",
        retries=0,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(DecisionsError, match="HTTP 402") as caught:
        client(DecisionRequest(state="s", questions=NOUL))
    assert "credits" in caught.value.hint


def test_the_default_transport_reports_an_unreachable_host(monkeypatch):
    import urllib.error

    def urlopen(request, timeout):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("decisions.client.urllib.request.urlopen", urlopen)
    client = create_decisions_client(
        api_key="sk-or-TEST",
        base_url="https://example.test",
        retries=0,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(DecisionsError, match="cannot reach"):
        client(DecisionRequest(state="s", questions=NOUL))


def test_omitted_sleep_pauses_between_retries(monkeypatch):
    import time

    slept = []
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))
    calls = {"n": 0}

    def fetch(url, init):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(503, "nope")
        raw = json.dumps({"answers": {"moves": {"noul": 0.4}}})
        return _response(200, raw)

    client = create_decisions_client(api_key="sk-or-TEST", fetch_impl=fetch, retries=1)
    result = client(DecisionRequest(state="s", questions=NOUL))
    assert result.answers["moves"].noul == 0.4
    assert slept == [1.0]
