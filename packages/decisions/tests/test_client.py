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


@pytest.mark.parametrize("model", ["jev-router", "typesafe/jev-router", PINNED_DATE, "typesafe/jev-1.13 ", ""])
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
