"""OpenRouter Decisions API, pinned to one dated Jev.

The transport is the playtest client's shape: a state, named questions of
three kinds, and a validated answer. This copy adds the pin. The model
string on the wire is ``typesafe/jev-1.13``. The date is recorded on the
result and is not sent as the model name. An answer may echo
``typesafe/jev-1.13-20260917``. That echo is this pin. ``jev-router`` and
any other model string are refused. A request that names them fails before
any request.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, Union

PINNED_MODEL = "typesafe/jev-1.13"
PINNED_DATE = "jev-1.13-20260917"
# The request sends PINNED_MODEL. The API may echo the dated snapshot.
ACCEPTED_ANSWER_MODELS = frozenset({PINNED_MODEL, f"typesafe/{PINNED_DATE}"})
DEFAULT_BASE = "https://openrouter.ai/api/alpha"


class DecisionsError(Exception):
    code = "E_DECISIONS"

    def __init__(self, message: str, hint: str, status: int | None = None) -> None:
        super().__init__(message)
        self.hint = hint
        self.status = status


def _is_prob(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0.0 <= float(value) <= 1.0


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    true: str | None = None
    false: str | None = None

    def to_wire(self) -> dict:
        body: dict = {"type": "noul", "instructions": self.instructions}
        if self.true is not None or self.false is not None:
            body["criteria"] = {"true": self.true, "false": self.false}
        return body


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: Mapping[str, str]

    def to_wire(self) -> dict:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: Sequence[str]

    def to_wire(self) -> dict:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


DecisionQuestion = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: str = "noul"


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float | None = None
    probabilities: Mapping[str, float] | None = None
    type: str = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float | None = None
    legend: Mapping[str, str] | None = None
    probabilities: Mapping[str, float] | None = None
    type: str = "score"


DecisionAnswer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True)
class DecisionRequest:
    state: object
    questions: Mapping[str, DecisionQuestion]
    model: str = PINNED_MODEL
    dated: str = PINNED_DATE


@dataclass(frozen=True)
class DecisionResult:
    model: str
    dated: str
    answers: Mapping[str, DecisionAnswer]
    cost: float | None = None
    input_tokens: int | None = None


@dataclass
class FetchResponse:
    ok: bool
    status: int
    text: Callable[[], str] = field(default=lambda: "")


FetchInit = dict
Fetch = Callable[[str, FetchInit], FetchResponse]


def require_pinned(model: str) -> None:
    """Reject every model except the dated pin. The date itself is not a model name."""
    if model != PINNED_MODEL:
        raise DecisionsError(
            f"refused model {model!r}",
            "only typesafe/jev-1.13 is accepted; jev-router is not replayable",
        )


def validate_answer(question: DecisionQuestion, answer: object, answer_id: str) -> DecisionAnswer:
    """Check one answer against the question it answers. A bad shape is an error."""

    def bad(why: str) -> DecisionsError:
        return DecisionsError(
            f'answer "{answer_id}" {why}',
            "the decision model returned an unexpected shape; the Decisions API is alpha",
        )

    if not isinstance(answer, dict):
        raise bad("is missing")
    if isinstance(question, NoulQuestion):
        noul = answer.get("noul")
        if not _is_prob(noul):
            raise bad(f'has no probability in "noul" (got {json.dumps(noul)})')
        return NoulAnswer(noul=float(noul))
    if isinstance(question, ChoiceQuestion):
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in question.criteria:
            listed = ", ".join(question.criteria)
            raise bad(f"chose {json.dumps(choice)}, not one of {listed}")
        confidence = answer.get("confidence")
        probabilities = answer.get("probabilities")
        return ChoiceAnswer(
            choice=choice,
            confidence=float(confidence) if _is_prob(confidence) else None,
            probabilities=probabilities if isinstance(probabilities, dict) else None,
        )
    score = answer.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise bad(f'has no numeric "score" (got {json.dumps(score)})')
    confidence = answer.get("confidence")
    legend = answer.get("legend")
    probabilities = answer.get("probabilities")
    return ScoreAnswer(
        score=float(score),
        confidence=float(confidence) if _is_prob(confidence) else None,
        legend=legend if isinstance(legend, dict) else None,
        probabilities=probabilities if isinstance(probabilities, dict) else None,
    )


def _default_fetch(url: str, init: FetchInit) -> FetchResponse:
    request = urllib.request.Request(
        url,
        data=init["body"].encode("utf-8"),
        headers=init["headers"],
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=init["timeout"]) as response:
            raw = response.read().decode("utf-8")
            return FetchResponse(ok=True, status=response.status, text=lambda: raw)
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        return FetchResponse(ok=False, status=err.code, text=lambda: raw)
    except urllib.error.URLError as err:
        raise OSError(str(err.reason)) from err


def create_decisions_client(
    api_key: str,
    *,
    base_url: str = DEFAULT_BASE,
    fetch_impl: Fetch | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] | None = None,
    attempt_timeout_s: float = 60.0,
) -> Callable[[DecisionRequest], DecisionResult]:
    """Build a client. An empty key fails here, before any request exists."""
    key = api_key.strip()
    if not key:
        raise DecisionsError(
            "OPENROUTER_API_KEY is not set",
            "the live client is built only when the caller passes a key",
        )
    fetch = fetch_impl or _default_fetch
    pause = sleep or (lambda seconds: __import__("time").sleep(seconds))
    base = base_url.rstrip("/")

    def client(request: DecisionRequest) -> DecisionResult:
        require_pinned(request.model)
        if request.dated != PINNED_DATE:
            raise DecisionsError(
                f"refused date stamp {request.dated!r}",
                f"judgments are stamped {PINNED_DATE}",
            )
        if len(request.questions) == 0:
            raise DecisionsError("no questions to ask", "pass at least one question")
        # The date stays off the wire. The API model slug is the pin.
        body = json.dumps(
            {
                "model": PINNED_MODEL,
                "state": request.state,
                "questions": {qid: question.to_wire() for qid, question in request.questions.items()},
            }
        )
        last_err: DecisionsError | None = None
        for attempt in range(retries + 1):
            if attempt > 0:
                pause(1.0 * 2 ** (attempt - 1))
            status = 0
            text = ""
            try:
                response = fetch(
                    f"{base}/decisions",
                    {
                        "method": "POST",
                        "headers": {
                            "Authorization": f"Bearer {key}",
                            "content-type": "application/json",
                            "HTTP-Referer": "https://github.com/mcp-tool-shop-org/sense-si",
                            "X-Title": "sense-si",
                        },
                        "body": body,
                        "timeout": attempt_timeout_s,
                    },
                )
                status = response.status
                text = response.text()
                if not response.ok:
                    if status == 401:
                        hint = "OPENROUTER_API_KEY is missing or invalid"
                    elif status == 402:
                        hint = "OpenRouter credits exhausted"
                    elif status == 413:
                        hint = "state too large for the model; shorten the record"
                    elif status == 429 or status >= 500:
                        hint = "transient; retried"
                    else:
                        hint = "not retried"
                    last_err = DecisionsError(
                        f"HTTP {status} from the Decisions API for {PINNED_MODEL}: {text[:300]}",
                        hint,
                        status,
                    )
                    if status != 429 and status < 500:
                        raise last_err
                    continue
            except DecisionsError:
                raise
            except Exception as err:
                last_err = DecisionsError(
                    f"cannot reach the Decisions API: {err}",
                    "check connectivity; retried",
                    status or None,
                )
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                last_err = DecisionsError(
                    "non-JSON body from the Decisions API",
                    "transient; retried",
                    status,
                )
                continue
            if not isinstance(parsed, dict):
                last_err = DecisionsError(
                    "non-object body from the Decisions API",
                    "transient; retried",
                    status,
                )
                continue
            returned = parsed.get("model", PINNED_MODEL)
            if returned not in ACCEPTED_ANSWER_MODELS:
                raise DecisionsError(
                    f"refused model {returned!r} on the answer; not replayable",
                    "the response model does not match the pin",
                )
            answers = parsed.get("answers")
            if not isinstance(answers, dict):
                answers = {}
            checked: dict[str, DecisionAnswer] = {}
            for qid, question in request.questions.items():
                checked[qid] = validate_answer(question, answers.get(qid), qid)
            usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
            cost = usage.get("cost")
            input_tokens = usage.get("input_tokens")
            return DecisionResult(
                model=PINNED_MODEL,
                dated=PINNED_DATE,
                answers=checked,
                cost=cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
                input_tokens=input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else None,
            )
        raise last_err or DecisionsError("exhausted retries", "see the previous errors")

    return client
