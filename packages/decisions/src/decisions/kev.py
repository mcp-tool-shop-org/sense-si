"""Local Kev-4B engine. Same request shape as Jev, served at /v1/systemone.

The checkpoint is pinned by revision, the way Jev is pinned by snapshot.
Thresholds are fitted on leave-one-mix-out folds of Kev's own probabilities.
Jev's uncertain band is not a Kev threshold. The phrase-clean layer stays
insufficient evidence until new labels arrive. A non-commercial sibling
checkpoint is not imported here.
"""

from __future__ import annotations

import json
import math
import os
import runpy
import sys
import urllib.error
import urllib.request
from typing import Callable, Mapping, Sequence

from decisions.client import (
    DecisionRequest,
    DecisionResult,
    DecisionsError,
    validate_answer,
)

KEV_REPO = "jaredpalmer/kev-4b"
# Hub tag v1.0. Adapter and pointer head on Qwen/Qwen3.5-4B-Base.
KEV_REVISION = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
KEV_BASE = "Qwen/Qwen3.5-4B-Base"
KEV_BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
# kev.serve revision the capped run used. The checkpoint pin is KEV_REVISION.
KEV_LIBRARY_REVISION = "5e42a7a03f28134853dd3ff77461457e921e5ec1"
KEV_MODEL = f"{KEV_REPO}@{KEV_REVISION}"
# No adopted cut. Every probability sits inside this range, which means the
# question is unanswered. It is not a fitted threshold.
KEV_UNANSWERED_BAND = (0.0, 1.0)
KEV_JUDGMENT = "insufficient-evidence"
# 9B peaked at 27.8 GB under 0.82. Uncapped, 4B's allocator grew to 31.9 GB
# and tripped 2 of the 3 samples that make the VRAM watchdog kill a process.
DEFAULT_MEMORY_FRACTION = 0.82
DEFAULT_BASE = "http://127.0.0.1:8009"
_MEMORY_ENV = "KEV_MEMORY_FRACTION"
_LOGIT_CLIP = 1e-6
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def resolve_memory_fraction(fraction: float | None = None) -> float:
    """Argument, then KEV_MEMORY_FRACTION, then the safe default."""
    if fraction is None and _MEMORY_ENV in os.environ:
        raw = os.environ[_MEMORY_ENV].strip()
        try:
            fraction = float(raw)
        except ValueError as err:
            raise DecisionsError(
                f"{_MEMORY_ENV} is not a fraction",
                "set it to a number between 0 and 1, or leave it unset",
            ) from err
    if fraction is None:
        fraction = DEFAULT_MEMORY_FRACTION
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise DecisionsError("memory fraction is not a number", "pass a float in (0, 1]")
    number = float(fraction)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise DecisionsError(
            f"memory fraction {number} is outside (0, 1]",
            "the cap has to leave the process some of the card, and not more than all of it",
        )
    return number


def _default_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _default_set_fraction(fraction: float, device: int) -> None:
    import torch

    torch.cuda.set_per_process_memory_fraction(fraction, device)


def apply_memory_cap(
    fraction: float | None = None,
    *,
    device: int = 0,
    set_fraction: Callable[[float, int], None] | None = None,
    cuda_available: bool | None = None,
) -> float:
    """Cap this process before Kev allocates. A CPU process records the fraction."""
    number = resolve_memory_fraction(fraction)
    available = _default_cuda() if cuda_available is None else cuda_available
    if not available:
        return number
    setter = _default_set_fraction if set_fraction is None else set_fraction
    setter(number, device)
    return number


def _argv_pins_run(argv: Sequence[str]) -> bool:
    args = list(argv)
    if "--run" not in args:
        return False
    index = args.index("--run")
    return index + 1 < len(args) and args[index + 1] == KEV_MODEL


def _argv_is_local(argv: Sequence[str]) -> bool:
    args = list(argv)
    for index, arg in enumerate(args):
        if arg == "--host":
            if index + 1 >= len(args):
                return False
            return args[index + 1] in _LOCAL_HOSTS
        if arg.startswith("--host="):
            return arg.split("=", 1)[1] in _LOCAL_HOSTS
    return True


def _run_kev_serve(argv: Sequence[str]) -> None:
    previous = sys.argv[:]
    sys.argv = ["kev.serve", *argv]
    try:
        runpy.run_module("kev.serve", run_name="__main__")
    finally:
        sys.argv = previous


def serve_capped(
    fraction: float | None = None,
    *,
    port: int = 8009,
    argv: Sequence[str] | None = None,
    runner: Callable[[Sequence[str]], None] | None = None,
    set_fraction: Callable[[float, int], None] | None = None,
    cuda_available: bool | None = None,
) -> float:
    """Set the memory cap, then serve the pinned checkpoint in this process.

    The default argv binds 127.0.0.1. The fraction is set here, in the
    process that loads the checkpoint. A child process does not inherit it.
    """
    applied = apply_memory_cap(
        fraction,
        set_fraction=set_fraction,
        cuda_available=cuda_available,
    )
    args = list(argv) if argv is not None else [
        "--run",
        KEV_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if not _argv_pins_run(args):
        raise DecisionsError(
            "Kev serve is not pinned to the checkpoint revision",
            f"pass --run {KEV_MODEL}",
        )
    if not _argv_is_local(args):
        raise DecisionsError(
            "Kev serve does not bind beyond this machine",
            "pass --host 127.0.0.1",
        )
    (runner or _run_kev_serve)(args)
    return applied


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/systemone"):
        return base
    return f"{base}/v1/systemone"


def _wire_model(request: DecisionRequest) -> None:
    """The local engine sends the Kev pin. A third model string is refused."""
    from decisions.client import PINNED_MODEL

    if request.model not in {PINNED_MODEL, KEV_MODEL}:
        raise DecisionsError(
            f"refused model {request.model!r}",
            f"the local engine is {KEV_MODEL}",
        )


def _default_fetch(url: str, init: dict) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=init["body"].encode("utf-8"),
        headers=init["headers"],
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=init["timeout"]) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except urllib.error.URLError as err:
        raise DecisionsError(
            "cannot reach Kev",
            "the capped server was not running",
        ) from err


def create_kev_client(
    *,
    base_url: str = DEFAULT_BASE,
    fetch_impl: Callable[[str, dict], tuple[int, str]] | None = None,
    attempt_timeout_s: float = 120.0,
) -> Callable[[DecisionRequest], DecisionResult]:
    """Talk to /v1/systemone. No key. The tests pass a stub fetch."""
    fetch = fetch_impl or _default_fetch
    url = _endpoint(base_url)

    def client(request: DecisionRequest) -> DecisionResult:
        _wire_model(request)
        if len(request.questions) == 0:
            raise DecisionsError("no questions to ask", "pass at least one question")
        body = json.dumps(
            {
                "model": KEV_MODEL,
                "state": request.state,
                "questions": {
                    qid: question.to_wire() for qid, question in request.questions.items()
                },
            }
        )
        status, text = fetch(
            url,
            {
                "method": "POST",
                "headers": {"content-type": "application/json"},
                "body": body,
                "timeout": attempt_timeout_s,
            },
        )
        if status != 200:
            raise DecisionsError(
                f"HTTP {status} from Kev",
                "the capped server refused the request",
                status,
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as err:
            raise DecisionsError("non-JSON body from Kev", "the server did not answer") from err
        if not isinstance(parsed, dict):
            raise DecisionsError("non-object body from Kev", "the server did not answer")
        returned = parsed.get("model")
        if returned != KEV_MODEL:
            raise DecisionsError(
                f"refused model {returned!r} on the answer",
                "the response is not the pinned checkpoint",
            )
        answers = parsed.get("answers")
        if not isinstance(answers, dict):
            answers = {}
        checked = {}
        for qid, question in request.questions.items():
            checked[qid] = validate_answer(question, answers.get(qid), qid)
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        input_tokens = usage.get("input_tokens")
        return DecisionResult(
            model=KEV_MODEL,
            dated=KEV_REVISION,
            answers=checked,
            cost=None,
            input_tokens=input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else None,
        )

    return client


def _clip(probability: float) -> float:
    return min(1.0 - _LOGIT_CLIP, max(_LOGIT_CLIP, float(probability)))


def _scale(probability: float, temperature: float) -> float:
    logit = math.log(_clip(probability) / (1.0 - _clip(probability)))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _fit_temperature(probs: Sequence[float], labels: Sequence[int]) -> float:
    """NLL on this fold's training mixes. One class has nothing to fit."""
    if len(set(labels)) < 2:
        return 1.0
    candidates = [1.0]
    for step in range(-40, 41):
        candidates.append(10 ** (step / 20.0))
    best_nll = math.inf
    best_t = 1.0
    for temperature in candidates:
        nll = 0.0
        for probability, label in zip(probs, labels):
            scaled = _scale(probability, temperature)
            picked = scaled if label else 1.0 - scaled
            nll -= math.log(max(picked, _LOGIT_CLIP))
        closer = abs(temperature - 1.0) < abs(best_t - 1.0)
        if nll < best_nll - 1e-12 or (abs(nll - best_nll) <= 1e-12 and closer):
            best_nll = nll
            best_t = temperature
    return best_t


def fit_lomo(rows: Sequence[Mapping[str, object]]) -> dict:
    """Leave-one-mix-out temperature on Kev probabilities. Does not adopt a band."""
    if len(rows) < 2:
        raise DecisionsError("leave-one-mix-out needs more than one row", "pass the scored phrases")
    mixes: list[str] = []
    for row in rows:
        mix = str(row["mix"])
        if mix not in mixes:
            mixes.append(mix)
    if len(mixes) < 2:
        raise DecisionsError("leave-one-mix-out needs two mixes", "group the phrases by mix")
    temperatures: dict[str, float] = {}
    fold_brier: list[float] = []
    fold_base: list[float] = []
    calibrated: list[float | None] = [None] * len(rows)
    base_rate: list[float | None] = [None] * len(rows)
    for held in mixes:
        train = [row for row in rows if str(row["mix"]) != held]
        test_at = [index for index, row in enumerate(rows) if str(row["mix"]) == held]
        labels = [int(row["y"]) for row in train]
        temperature = _fit_temperature([float(row["p"]) for row in train], labels)
        temperatures[held] = temperature
        rate = sum(labels) / len(labels)
        model_loss = []
        base_loss = []
        for index in test_at:
            probability = _scale(float(rows[index]["p"]), temperature)
            label = int(rows[index]["y"])
            calibrated[index] = probability
            base_rate[index] = rate
            model_loss.append((probability - label) ** 2)
            base_loss.append((rate - label) ** 2)
        fold_brier.append(sum(model_loss) / len(model_loss))
        fold_base.append(sum(base_loss) / len(base_loss))
    scored = [float(value) for value in calibrated]
    bases = [float(value) for value in base_rate]
    labels = [int(row["y"]) for row in rows]
    return {
        "temperatures": temperatures,
        "fold_brier": fold_brier,
        "fold_base_brier": fold_base,
        "brier": sum((p - y) ** 2 for p, y in zip(scored, labels)) / len(labels),
        "base_brier": sum((p - y) ** 2 for p, y in zip(bases, labels)) / len(labels),
        "judgment": KEV_JUDGMENT,
    }
