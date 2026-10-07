"""Two questions. Both return probabilities. Neither returns a winner or a gate."""

from __future__ import annotations

import re
from dataclasses import dataclass

from decisions import (
    KEV_MODEL,
    KEV_REVISION,
    KEV_UNANSWERED_BAND,
    PINNED_DATE,
    PINNED_MODEL,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
)

from ai_ears.record import EarsError, HearingRecord, reject_gate_keys

# Kept after docs/calibration.md. 124 sung phrases, 73 clean and 51 not clean.
# Geifman and El-Yaniv 2017, target risk 0.10, δ = 0.001. No interior band in
# 1000 of 1000 inner resamples, so these edges stay. The product interval is
# closed: 0.35 and 0.65 are too close to call. The study would have abstained
# only strictly inside its edges.
DEFAULT_BAND = (0.35, 0.65)
_GATE_WORD = re.compile(r"\b(pass|fail|passed|failed)\b", re.IGNORECASE)


@dataclass(frozen=True)
class QuestionResult:
    question: str
    model: str
    dated: str
    probabilities: dict[str, float]
    uncertain: bool
    band: tuple[float, float]
    confidence: float | None = None
    cost: float | None = None

    def to_dict(self) -> dict:
        body: dict = {
            "question": self.question,
            "model": self.model,
            "dated": self.dated,
            "probabilities": self.probabilities,
            "uncertain": self.uncertain,
            "band": [self.band[0], self.band[1]],
        }
        if self.confidence is not None:
            body["confidence"] = self.confidence
        if self.cost is not None:
            body["cost"] = self.cost
        reject_gate_keys(body, "question result")
        return body


def load_question_result(payload: dict) -> QuestionResult:
    reject_gate_keys(payload, "question result")
    allowed = {
        "question",
        "model",
        "dated",
        "probabilities",
        "uncertain",
        "band",
        "confidence",
        "cost",
    }
    extra = set(payload) - allowed
    if extra:
        raise EarsError(f"question result has unexpected fields: {', '.join(sorted(extra))}")
    band = payload["band"]
    return QuestionResult(
        question=payload["question"],
        model=payload["model"],
        dated=payload["dated"],
        probabilities={key: float(value) for key, value in payload["probabilities"].items()},
        uncertain=bool(payload["uncertain"]),
        band=(float(band[0]), float(band[1])),
        confidence=None if payload.get("confidence") is None else float(payload["confidence"]),
        cost=None if payload.get("cost") is None else float(payload["cost"]),
    )


def _in_band(value: float, band: tuple[float, float]) -> bool:
    low, high = band
    return low <= value <= high


def _check_instructions(instructions: str) -> None:
    if not instructions.strip():
        raise EarsError("the question instructions are empty")
    if _GATE_WORD.search(instructions):
        raise EarsError("the question asks Jev to pass or fail")


def _check_pin(result: DecisionResult) -> str:
    if result.model == PINNED_MODEL and result.dated == PINNED_DATE:
        return "jev"
    if result.model == KEV_MODEL and result.dated == KEV_REVISION:
        return "kev"
    raise EarsError("decision result is not a pinned engine")


def _answered_band(
    engine: str, band: tuple[float, float] | None
) -> tuple[tuple[float, float], bool]:
    """Jev uses its band. Kev has no adopted cut, so the question stays unanswered."""
    if engine == "kev":
        if band is not None:
            raise EarsError("a Kev threshold is fitted on Kev's own folds")
        return KEV_UNANSWERED_BAND, True
    return (DEFAULT_BAND if band is None else band), False


def _probabilities(raw: dict, expected: set[str]) -> dict[str, float]:
    if set(raw) != expected:
        raise EarsError("choice probabilities do not match the takes")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EarsError(f"probability for {key} is not a number")
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise EarsError(f"probability for {key} is outside 0..1")
        out[key] = number
    return out


def which_take(
    records: list[HearingRecord] | tuple[HearingRecord, ...],
    *,
    instructions: str,
    client,
    band: tuple[float, float] | None = None,
) -> QuestionResult:
    """Ask which take. Return a probability per take, and no winner."""
    _check_instructions(instructions)
    if len(records) < 2:
        raise EarsError("which_take needs at least two takes")
    phrase = records[0].phrase_id
    ids = [record.take_id for record in records]
    if len(set(ids)) != len(ids):
        raise EarsError("take ids must be unique")
    if any(record.phrase_id != phrase for record in records):
        raise EarsError("which_take compares takes of one phrase")
    request = DecisionRequest(
        state={"phrase_id": phrase, "takes": [record.to_state() for record in records]},
        questions={
            "which_take": ChoiceQuestion(
                instructions=instructions,
                criteria={
                    record.take_id: f"measurements for {record.take_id} in state.takes"
                    for record in records
                },
            )
        },
    )
    result = client(request)
    engine = _check_pin(result)
    used, unanswered = _answered_band(engine, band)
    answer = result.answers["which_take"]
    if not isinstance(answer, ChoiceAnswer) or not answer.probabilities:
        raise EarsError("which_take answer has no probabilities")
    probabilities = _probabilities(dict(answer.probabilities), set(ids))
    confidence = answer.confidence
    uncertain = True if unanswered else (confidence is None or _in_band(confidence, used))
    return QuestionResult(
        question="which_take",
        model=result.model,
        dated=result.dated,
        probabilities=probabilities,
        uncertain=uncertain,
        band=used,
        confidence=confidence,
        cost=result.cost,
    )


def phrase_clean(
    record: HearingRecord,
    *,
    instructions: str,
    true: str,
    false: str,
    client,
    band: tuple[float, float] | None = None,
) -> QuestionResult:
    """Ask whether the caller's criterion holds. Return P(yes), not a gate."""
    _check_instructions(instructions)
    if not true.strip() or not false.strip():
        raise EarsError("phrase_clean needs true and false criteria")
    if true.strip() == false.strip():
        raise EarsError("true and false criteria must differ")
    request = DecisionRequest(
        state=record.to_state(),
        questions={
            "phrase_clean": NoulQuestion(instructions=instructions, true=true, false=false),
        },
    )
    result = client(request)
    engine = _check_pin(result)
    used, unanswered = _answered_band(engine, band)
    answer = result.answers["phrase_clean"]
    if not isinstance(answer, NoulAnswer):
        raise EarsError("phrase_clean answer is not a probability")
    probability = answer.noul
    uncertain = True if unanswered else _in_band(probability, used)
    return QuestionResult(
        question="phrase_clean",
        model=result.model,
        dated=result.dated,
        probabilities={"yes": probability},
        uncertain=uncertain,
        band=used,
        cost=result.cost,
    )
