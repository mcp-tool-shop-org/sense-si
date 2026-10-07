import json

import pytest

from decisions import (
    KEV_MODEL,
    KEV_REVISION,
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


def _pair():
    return [_take("take-a", cents=0, ms=0), _take("take-b", cents=1, ms=1)]


def _result(answer, model=PINNED_MODEL, dated=PINNED_DATE, cost=0.0):
    def client(request):
        return DecisionResult(model=model, dated=dated, answers={"which_take": answer, "phrase_clean": answer}, cost=cost)

    return client


def test_empty_instructions_never_call():
    def client(request):
        raise AssertionError("called")

    with pytest.raises(EarsError, match="empty"):
        which_take(_pair(), instructions="  ", client=client)
    with pytest.raises(EarsError, match="empty"):
        phrase_clean(_take("take-a", cents=0, ms=0), instructions="", true="yes", false="no", client=client)


def test_which_take_refuses_a_set_that_is_not_one_phrase():
    def client(request):
        raise AssertionError("called")

    with pytest.raises(EarsError, match="at least two"):
        which_take([_take("take-a", cents=0, ms=0)], instructions=INSTRUCTIONS, client=client)
    with pytest.raises(EarsError, match="unique"):
        which_take(
            [_take("take-a", cents=0, ms=0), _take("take-a", cents=1, ms=1)],
            instructions=INSTRUCTIONS,
            client=client,
        )
    other = HearingRecord(take_id="take-b", phrase_id="other", pitch=(), timing=())
    with pytest.raises(EarsError, match="one phrase"):
        which_take([_take("take-a", cents=0, ms=0), other], instructions=INSTRUCTIONS, client=client)


def test_which_take_refuses_an_unpinned_or_shapeless_answer():
    records = _pair()
    probs = {"take-a": 0.6, "take-b": 0.4}
    with pytest.raises(EarsError, match="pinned"):
        which_take(records, instructions=INSTRUCTIONS, client=_result(ChoiceAnswer(choice="take-a", probabilities=probs), model="other"))
    with pytest.raises(EarsError, match="pinned"):
        which_take(records, instructions=INSTRUCTIONS, client=_result(ChoiceAnswer(choice="take-a", probabilities=probs), dated="nope"))
    with pytest.raises(EarsError, match="no probabilities"):
        which_take(records, instructions=INSTRUCTIONS, client=_result(ChoiceAnswer(choice="take-a")))
    with pytest.raises(EarsError, match="no probabilities"):
        which_take(records, instructions=INSTRUCTIONS, client=_result(NoulAnswer(noul=0.4)))
    with pytest.raises(EarsError, match="do not match"):
        which_take(
            records,
            instructions=INSTRUCTIONS,
            client=_result(ChoiceAnswer(choice="take-a", probabilities={"take-a": 1.0})),
        )
    with pytest.raises(EarsError, match="not a number"):
        which_take(
            records,
            instructions=INSTRUCTIONS,
            client=_result(ChoiceAnswer(choice="take-a", probabilities={"take-a": True, "take-b": 0.2})),
        )
    with pytest.raises(EarsError, match="outside"):
        which_take(
            records,
            instructions=INSTRUCTIONS,
            client=_result(ChoiceAnswer(choice="take-a", probabilities={"take-a": 1.2, "take-b": 0.0})),
        )
    with pytest.raises(EarsError, match="pass or fail"):
        which_take(records, instructions="the take passed", client=_result(NoulAnswer(noul=0.1)))


def test_phrase_clean_refuses_blank_criteria_and_a_choice_answer():
    record = _take("take-a", cents=0, ms=0)
    client = _stub(NoulAnswer(noul=0.1))
    with pytest.raises(EarsError, match="true and false"):
        phrase_clean(record, instructions=CLEAN, true="  ", false="no", client=client)
    with pytest.raises(EarsError, match="true and false"):
        phrase_clean(record, instructions=CLEAN, true="yes", false="", client=client)
    with pytest.raises(EarsError, match="must differ"):
        phrase_clean(record, instructions=CLEAN, true="same", false=" same ", client=client)
    with pytest.raises(EarsError, match="not a probability"):
        phrase_clean(record, instructions=CLEAN, true="yes", false="no", client=_result(ChoiceAnswer(choice="yes")))
    with pytest.raises(EarsError, match="pinned"):
        phrase_clean(
            record,
            instructions=CLEAN,
            true="yes",
            false="no",
            client=_result(NoulAnswer(noul=0.2), dated="nope"),
        )


def test_a_result_without_a_cost_omits_it_and_a_loaded_confidence_is_kept():
    record = _take("take-a", cents=0, ms=0)

    def client(request):
        return DecisionResult(
            model=PINNED_MODEL,
            dated=PINNED_DATE,
            answers={"phrase_clean": NoulAnswer(noul=0.1)},
            cost=None,
        )

    body = phrase_clean(record, instructions=CLEAN, true="yes", false="no", client=client).to_dict()
    assert "cost" not in body
    assert "confidence" not in body
    loaded = load_question_result({**body, "confidence": 0.9})
    assert loaded.confidence == 0.9


def test_an_explicit_jev_band_still_marks_a_clear_probability():
    result = phrase_clean(
        _take("take-a", cents=0, ms=0),
        instructions=CLEAN,
        true="yes",
        false="no",
        client=_stub(NoulAnswer(noul=0.5)),
        band=(0.1, 0.2),
    )
    assert result.band == (0.1, 0.2)
    assert result.uncertain is False
    assert result.to_dict()["band"] == [0.1, 0.2]


def test_kev_phrase_clean_stays_unanswered_and_refuses_a_passed_band():
    record = _take("take-a", cents=0, ms=0)

    def client(request):
        return DecisionResult(
            model=KEV_MODEL,
            dated=KEV_REVISION,
            answers={"phrase_clean": NoulAnswer(noul=0.92)},
            cost=None,
        )

    result = phrase_clean(record, instructions=CLEAN, true="yes", false="no", client=client)
    assert result.uncertain is True
    assert result.band == (0.0, 1.0)
    assert result.model == KEV_MODEL
    assert result.dated == KEV_REVISION
    assert result.probabilities == {"yes": 0.92}
    assert result.to_dict()["band"] == [0.0, 1.0]
    with pytest.raises(EarsError, match="Kev threshold"):
        phrase_clean(
            record,
            instructions=CLEAN,
            true="yes",
            false="no",
            client=client,
            band=(0.35, 0.65),
        )
    with pytest.raises(EarsError, match="pinned"):
        phrase_clean(
            record,
            instructions=CLEAN,
            true="yes",
            false="no",
            client=_result(NoulAnswer(noul=0.92), model=KEV_MODEL, dated=PINNED_DATE),
        )


def test_kev_which_take_stays_unanswered():
    probs = {"take-a": 0.7, "take-b": 0.3}
    answer = ChoiceAnswer(choice="take-a", probabilities=probs, confidence=0.9)
    result = which_take(
        _pair(),
        instructions=INSTRUCTIONS,
        client=_result(answer, model=KEV_MODEL, dated=KEV_REVISION),
    )
    assert result.uncertain is True
    assert result.band == (0.0, 1.0)
    assert result.probabilities == probs
    with pytest.raises(EarsError, match="Kev threshold"):
        which_take(
            _pair(),
            instructions=INSTRUCTIONS,
            client=_result(answer, model=KEV_MODEL, dated=KEV_REVISION),
            band=(0.35, 0.65),
        )
    with pytest.raises(EarsError, match="pinned"):
        which_take(
            _pair(),
            instructions=INSTRUCTIONS,
            client=_result(answer, model="other", dated=KEV_REVISION),
        )
