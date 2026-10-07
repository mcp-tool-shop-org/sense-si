import json

import pytest

from decisions import (
    PINNED_DATE,
    PINNED_MODEL,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    NoulAnswer,
    create_decisions_client,
)
from decisions.client import FetchResponse

from ai_ears.questions import load_question_result, phrase_clean, which_take
from ai_ears.record import EarsError, HearingRecord, Measure

INSTRUCTIONS = "Compare the measurements in the record."
CLEAN = "Is the pitch median within 20 cents and the onset offset within 40 ms?"


def _measure(value: float, unit: str) -> Measure:
    instrument = "pyin" if unit == "cents" else "rise_onset"
    return Measure(value=value, unit=unit, instrument=instrument, revision="fixture")


def _take(take_id: str, cents: float, ms: float) -> HearingRecord:
    return HearingRecord(
        take_id=take_id,
        phrase_id="fixture-phrase",
        pitch=(_measure(cents, "cents"),),
        timing=(_measure(ms, "ms"),),
    )


def _stub(answer):
    def client(request: DecisionRequest) -> DecisionResult:
        client.request = request
        return DecisionResult(
            model=PINNED_MODEL,
            dated=PINNED_DATE,
            answers={next(iter(request.questions)): answer},
            cost=0.0002,
        )

    return client


def test_which_take_returns_probabilities_and_no_winner():
    client = _stub(ChoiceAnswer(choice="take-b", confidence=0.82, probabilities={"take-a": 0.2, "take-b": 0.8}))
    result = which_take(
        [_take("take-a", cents=-30, ms=70), _take("take-b", cents=-2, ms=6)],
        instructions=INSTRUCTIONS,
        client=client,
    )
    body = result.to_dict()
    assert body["probabilities"] == {"take-a": 0.2, "take-b": 0.8}
    assert body["model"] == PINNED_MODEL
    assert body["dated"] == PINNED_DATE
    assert body["uncertain"] is False
    assert body["cost"] == 0.0002
    assert "choice" not in body
    assert "winner" not in body
    assert "pass" not in body
    assert "fail" not in body
    state = client.request.state
    assert state["takes"][0]["pitch"][0]["value"] == -30
    assert "verdict" not in json.dumps(state)


def test_missing_confidence_is_too_close_to_call():
    client = _stub(ChoiceAnswer(choice="take-a", probabilities={"take-a": 0.51, "take-b": 0.49}))
    result = which_take(
        [_take("take-a", cents=0, ms=1), _take("take-b", cents=1, ms=2)],
        instructions=INSTRUCTIONS,
        client=client,
    )
    assert result.uncertain is True
    assert result.confidence is None


def test_phrase_clean_reports_p_yes_and_the_band():
    middling = _stub(NoulAnswer(noul=0.5))
    inside = phrase_clean(
        _take("take-a", cents=-2, ms=6),
        instructions=CLEAN,
        true="median within 20 cents and onset within 40 ms",
        false="either measurement sits outside those bounds",
        client=middling,
    )
    assert inside.probabilities == {"yes": 0.5}
    assert inside.uncertain is True
    assert "no" not in inside.to_dict()["probabilities"]

    sure = _stub(NoulAnswer(noul=0.92))
    outside = phrase_clean(
        _take("take-a", cents=-2, ms=6),
        instructions=CLEAN,
        true="median within 20 cents and onset within 40 ms",
        false="either measurement sits outside those bounds",
        client=sure,
    )
    assert outside.uncertain is False
    assert outside.to_dict()["band"] == [0.35, 0.65]


def test_band_edges_are_uncertain():
    for probability in (0.35, 0.65):
        result = phrase_clean(
            _take("take-a", cents=0, ms=0),
            instructions=CLEAN,
            true="holds",
            false="does not hold",
            client=_stub(NoulAnswer(noul=probability)),
        )
        assert result.uncertain is True


def test_a_gate_word_in_the_question_never_calls():
    def client(request):
        raise AssertionError("called")

    with pytest.raises(EarsError, match="pass or fail"):
        phrase_clean(
            _take("take-a", cents=0, ms=0),
            instructions="Does this phrase fail the timing gate?",
            true="yes",
            false="no",
            client=client,
        )


def test_load_rejects_pass_and_fail_and_a_winner():
    good = phrase_clean(
        _take("take-a", cents=0, ms=0),
        instructions=CLEAN,
        true="holds",
        false="does not hold",
        client=_stub(NoulAnswer(noul=0.2)),
    ).to_dict()
    assert load_question_result(good).probabilities["yes"] == 0.2
    with pytest.raises(EarsError, match="pass"):
        load_question_result({**good, "pass": True})
    with pytest.raises(EarsError, match="fail"):
        load_question_result({**good, "fail": False})
    with pytest.raises(EarsError, match="unexpected"):
        load_question_result({**good, "winner": "take-a"})


def test_which_take_through_the_real_client_never_leaves_the_machine():
    sent = {}

    def fetch(url, init):
        sent["body"] = json.loads(init["body"])
        raw = json.dumps(
            {
                "model": PINNED_MODEL,
                "answers": {
                    "which_take": {
                        "type": "choice",
                        "choice": "take-a",
                        "confidence": 0.4,
                        "probabilities": {"take-a": 0.55, "take-b": 0.45},
                    }
                },
                "usage": {"cost": 0.0001},
            }
        )
        return FetchResponse(ok=True, status=200, text=lambda: raw)

    client = create_decisions_client(api_key="sk-or-TEST", fetch_impl=fetch, sleep=lambda _s: None)
    result = which_take(
        [_take("take-a", cents=1, ms=2), _take("take-b", cents=3, ms=4)],
        instructions=INSTRUCTIONS,
        client=client,
    )
    assert sent["body"]["model"] == PINNED_MODEL
    assert "dated" not in sent["body"]
    assert result.dated == PINNED_DATE
    assert result.uncertain is True
    assert "choice" not in result.to_dict()
    assert result.probabilities == {"take-a": 0.55, "take-b": 0.45}
