"""Phrase-calibration rule, locked before any phrase is scored.

The rule is the R&D entry 2026-10-07-calibrating-probabilistic-decision-layers.
Do not retune the constants below after seeing labels, probabilities, or a score.
A live call spends OpenRouter only under the $0.25 cap, and only for
typesafe/jev-1.13. The record sent to Jev has an empty marks list.

Band scale: phrase_clean compares the raw P(yes) with the band. Temperature
scaling is fit for the reported probabilities only. It is not applied to the
band, and it is not wired into the question.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _extra in (
    ROOT / "packages" / "ears" / "src",
    ROOT / "packages" / "decisions" / "src",
):
    _text = str(_extra)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ai_ears.questions import _GATE_WORD, phrase_clean  # noqa: E402
from ai_ears.record import EarsError, from_jam_take  # noqa: E402
from decisions import PINNED_DATE, PINNED_MODEL, create_decisions_client  # noqa: E402

# Locked 2026-10-07, before any test phrase was scored.
SEED = 20261007
FOLDS = 5
REPEATS = 10
BOOTSTRAP = 1000
OUTER_FRACTION = 0.30
DELTA = 0.001  # Geifman and El-Yaniv 2017, their experiment's confidence
TARGET_RISK = 0.10
ACCURACY_DELTA = 0.05  # one-sided 95% Clopper-Pearson floor on the outer set
MARK_LOOKBACK_S = 1.0
MARK_LOOKAHEAD_S = 0.2
EDGE_SWING = 0.10
NO_BAND_FRACTION = 0.05
MIN_CLASS = 30
TEMPERATURE_LO = 0.05
TEMPERATURE_HI = 20.0
TEMPERATURE_STEPS = 201
CAP_USD = 0.25
EXPECTED_PHRASES = 124
PLACEHOLDER_BAND = (0.35, 0.65)
INSTRUCTIONS = "Is this phrase clean?"
TRUE_CRITERION = "the phrase is clean"
FALSE_CRITERION = "the phrase is not clean"
TAKE_ID = r"^[A-Za-z0-9._-]{1,80}$"
VOCAL_ROOT = Path(r"E:/AI/ajs-fullsong/tmp/vocal-clock")
MARKS_PATH = VOCAL_ROOT / "review" / "director-marks-2026-10-07.json"
MIXES = (
    ("amazing-grace-new-britain", "phrase16", "sing"),
    ("amazing-grace-new-britain", "phrase16w", "sing"),
    ("amazing-grace-new-britain", "phrase16w2", "sing"),
    ("amazing-grace-new-britain", "pad16", "pad"),
    ("america-the-beautiful-materna", "phrase16w", "sing"),
    ("america-the-beautiful-materna", "phrase16w2", "sing"),
    ("america-the-beautiful-materna", "pad16", "pad"),
)

CACHE_PATH = Path(os.environ.get("TEMP", ".")) / "sense-si-jev-calibration-cache.json"
DOC_PATH = ROOT / "docs" / "calibration.md"
SUMMARY_PATH = ROOT / "docs" / "calibration" / "summary.json"
PHRASES_PATH = ROOT / "docs" / "calibration" / "phrases.json"
CORP_PATH = ROOT / "docs" / "calibration" / "corp.svg"


def mark_overlaps(start: float, end: float, mark_t: float) -> bool:
    """Closed overlap of [start, end] with [t - 1s, t + 0.2s]."""
    lo = mark_t - MARK_LOOKBACK_S
    hi = mark_t + MARK_LOOKAHEAD_S
    return lo <= end and hi >= start


def phrase_label(start: float, end: float, times: list[float]) -> int:
    """1 is clean. Any overlapping mark makes the phrase not clean."""
    for mark_t in times:
        if mark_overlaps(start, end, mark_t):
            return 0
    return 1


def _number_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    left, right = value
    if isinstance(left, bool) or isinstance(right, bool):
        return None
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    if not math.isfinite(left) or not math.isfinite(right) or left > right:
        return None
    return float(left), float(right)


def slice_phrase(timing: dict, pitch: dict, start: float, end: float) -> tuple[dict, dict]:
    """Keep the syllables and notes that sit in the phrase. The copies are shallow."""
    table = []
    for row in timing.get("table") or []:
        if not isinstance(row, dict):
            continue
        t_score = row.get("t_score")
        if isinstance(t_score, bool) or not isinstance(t_score, (int, float)):
            continue
        if start <= float(t_score) <= end:
            table.append(row)
    rows = []
    for row in pitch.get("rows") or []:
        if not isinstance(row, dict):
            continue
        window = _number_pair(row.get("window"))
        if window is None:
            continue
        if window[0] <= end and window[1] >= start:
            rows.append(row)
    sliced_timing = dict(timing)
    sliced_timing["table"] = table
    sliced_pitch = dict(pitch)
    sliced_pitch["rows"] = rows
    return sliced_timing, sliced_pitch


def binom_cdf(errors: int, n: int, prob: float) -> float:
    """P(X <= errors) for X ~ Binomial(n, prob), in log space."""
    if errors < 0:
        return 0.0
    if errors >= n:
        return 1.0
    if prob <= 0.0:
        return 1.0
    if prob >= 1.0:
        return 0.0
    logs = []
    for j in range(errors + 1):
        logs.append(
            math.lgamma(n + 1)
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * math.log(prob)
            + (n - j) * math.log(1.0 - prob)
        )
    top = max(logs)
    return math.exp(top) * sum(math.exp(item - top) for item in logs)


def clopper_upper(errors: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound. This is Lemma 3.1's B*."""
    if n <= 0:
        raise ValueError("Clopper-Pearson needs at least one trial")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must sit between 0 and 1")
    if errors <= 0:
        return 1.0 - delta ** (1.0 / n)
    if errors >= n:
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        # The CDF falls as the probability rises, so a CDF above delta means the bound is higher.
        if binom_cdf(errors, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def clopper_accuracy_lower(correct: int, n: int, delta: float = ACCURACY_DELTA) -> float | None:
    if n <= 0:
        return None
    return 1.0 - clopper_upper(n - correct, n, delta)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    ez = math.exp(logit)
    return ez / (1.0 + ez)


def apply_temperature(probability: float, temperature: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped / (1.0 - clipped))
    return _sigmoid(logit / temperature)


def temperature_grid() -> list[float]:
    lo = math.log(TEMPERATURE_LO)
    hi = math.log(TEMPERATURE_HI)
    grid = [
        math.exp(lo + (hi - lo) * step / (TEMPERATURE_STEPS - 1))
        for step in range(TEMPERATURE_STEPS)
    ]
    grid.append(1.0)
    return grid


def _nll(probs: list[float], labels: list[int], temperature: float) -> float:
    total = 0.0
    for probability, label in zip(probs, labels):
        calibrated = min(max(apply_temperature(probability, temperature), 1e-12), 1.0 - 1e-12)
        total += -(label * math.log(calibrated) + (1 - label) * math.log(1.0 - calibrated))
    return total / len(probs)


def fit_temperature(probs: list[float], labels: list[int]) -> float:
    """One-parameter temperature. Ties go to the value closest to 1."""
    if not probs:
        return 1.0
    best_t = 1.0
    best_loss = _nll(probs, labels, 1.0)
    for temperature in temperature_grid():
        loss = _nll(probs, labels, temperature)
        closer = abs(loss - best_loss) <= 1e-12 and abs(temperature - 1.0) < abs(best_t - 1.0)
        if loss < best_loss - 1e-12 or closer:
            best_loss = loss
            best_t = temperature
    return best_t


def selective_bound(probs: list[float], labels: list[int], delta: float) -> dict | None:
    """Selection with guaranteed risk, on raw P(yes).

    Probes are the full sample plus ceil(log2 n) binary-search thresholds
    (Geifman and El-Yaniv, Algorithm 1). Each probe is charged delta/(k+1)
    so the full-sample probe is inside the union bound. Full coverage that
    already meets the target is not an interior band: there is no reject
    region to estimate. Otherwise the lowest passing threshold on the path
    is the band, which is the narrowest one that path can certify.
    """
    n = len(probs)
    if n < 2 or len(labels) != n:
        return None
    kappa = [max(p, 1.0 - p) for p in probs]
    order = sorted(range(n), key=lambda i: (kappa[i], i))
    steps = math.ceil(math.log2(n))
    delta_i = delta / (steps + 1)

    def probe(theta: float) -> tuple[float, int, int]:
        selected = [i for i in range(n) if kappa[i] >= theta]
        errors = sum(1 for i in selected if (probs[i] >= 0.5) != bool(labels[i]))
        return clopper_upper(errors, len(selected), delta_i), len(selected), errors

    minimum = kappa[order[0]]
    full_bound, full_n, full_errors = probe(minimum)
    if full_bound < TARGET_RISK:
        return {
            "interior": False,
            "theta": minimum,
            "low": None,
            "high": None,
            "bound": full_bound,
            "selected": full_n,
            "errors": full_errors,
        }
    best: dict | None = None
    zmin = 1
    zmax = n
    for _ in range(steps):
        z = math.ceil((zmin + zmax) / 2)
        theta = kappa[order[z - 1]]
        bound, selected, errors = probe(theta)
        if bound < TARGET_RISK:
            if best is None or theta < best["theta"]:
                best = {
                    "theta": theta,
                    "bound": bound,
                    "selected": selected,
                    "errors": errors,
                }
            zmax = z
        else:
            zmin = z
    if best is None:
        return None
    interior = any(item < best["theta"] for item in kappa)
    if not interior:
        return {
            "interior": False,
            "theta": best["theta"],
            "low": None,
            "high": None,
            "bound": best["bound"],
            "selected": best["selected"],
            "errors": best["errors"],
        }
    return {
        "interior": True,
        "theta": best["theta"],
        "low": 1.0 - best["theta"],
        "high": best["theta"],
        "bound": best["bound"],
        "selected": best["selected"],
        "errors": best["errors"],
    }


def pav(probs: list[float], labels: list[int]) -> list[float]:
    """Pool-adjacent-violators isotonic fit. Diagram only, not a recalibrator."""
    order = sorted(range(len(probs)), key=lambda i: (probs[i], i))
    blocks: list[list[int]] = []
    means: list[float] = []
    for index in order:
        blocks.append([index])
        means.append(float(labels[index]))
        while len(means) >= 2 and means[-2] > means[-1]:
            block = blocks.pop()
            means.pop()
            blocks[-1].extend(block)
            means[-1] = sum(float(labels[j]) for j in blocks[-1]) / len(blocks[-1])
    fitted = [0.0] * len(probs)
    for block, mean in zip(blocks, means):
        for index in block:
            fitted[index] = mean
    return fitted


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("median of an empty sample")
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def assign_outer(labels: list[int], seed: int = SEED) -> set[int]:
    """Stratified holdout. Band fitting never sees these indices."""
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(labels):
        groups[label].append(index)
    outer: set[int] = set()
    for label in (0, 1):
        order = groups[label][:]
        rng.shuffle(order)
        hold = int(round(OUTER_FRACTION * len(order)))
        if len(order) >= 2:
            hold = min(max(hold, 1), len(order) - 1)
        else:
            hold = 0
        outer.update(order[:hold])
    return outer


def assign_folds(labels: list[int], indices: list[int], repeat: int) -> dict[int, int]:
    rng = random.Random(SEED + 1 + repeat)
    groups: dict[int, list[int]] = {0: [], 1: []}
    for index in indices:
        groups[labels[index]].append(index)
    assignment: dict[int, int] = {}
    for label in (0, 1):
        order = groups[label][:]
        rng.shuffle(order)
        for offset, index in enumerate(order):
            assignment[index] = offset % FOLDS
    return assignment


def _subset(probs: list[float], labels: list[int], indices: list[int]) -> tuple[list[float], list[int]]:
    return [probs[i] for i in indices], [labels[i] for i in indices]


def _phrase_bootstrap(
    phrase_values: list[list[float]], seed: int, draws_n: int
) -> tuple[float | None, float | None, float | None]:
    """Resample phrases. Each phrase contributes the mean of its held-out values."""
    if not phrase_values:
        return None, None, None
    point = sum(sum(item) / len(item) for item in phrase_values) / len(phrase_values)
    rng = random.Random(seed)
    draws = []
    count = len(phrase_values)
    for _ in range(draws_n):
        total = 0.0
        for _pick in range(count):
            chosen = phrase_values[rng.randrange(count)]
            total += sum(chosen) / len(chosen)
        draws.append(total / count)
    return point, percentile(draws, 0.025), percentile(draws, 0.975)


def bootstrap_edges(
    probs: list[float], labels: list[int], indices: list[int], seed: int, draws_n: int
) -> dict:
    """Refit the band rule on stratified resamples of the inner phrases."""
    groups = {
        0: [i for i in indices if labels[i] == 0],
        1: [i for i in indices if labels[i] == 1],
    }
    rng = random.Random(seed)
    lows: list[float] = []
    highs: list[float] = []
    no_interior = 0
    for _ in range(draws_n):
        drawn: list[int] = []
        for label in (0, 1):
            pool = groups[label]
            if not pool:
                continue
            for _pick in range(len(pool)):
                drawn.append(pool[rng.randrange(len(pool))])
        if len(drawn) < 2:
            no_interior += 1
            continue
        result = selective_bound(
            [probs[i] for i in drawn],
            [labels[i] for i in drawn],
            DELTA,
        )
        if result is None or not result["interior"]:
            no_interior += 1
            continue
        lows.append(result["low"])
        highs.append(result["high"])
    return {"low": lows, "high": highs, "no_interior": no_interior, "n": draws_n}


def decide(stats: dict) -> dict:
    """Adopt a band only when every locked gate passes. Otherwise keep 0.35-0.65."""
    reasons: list[str] = []
    if stats["n_clean"] < MIN_CLASS or stats["n_not_clean"] < MIN_CLASS:
        reasons.append("class-count")
    draws = stats["n_bootstrap"]
    if draws <= 0 or stats["n_no_interior"] / draws > NO_BAND_FRACTION:
        reasons.append("no-interior-fraction")
    lows = stats["edge_low"]
    highs = stats["edge_high"]
    if (
        len(lows) < 2
        or percentile(highs, 0.975) - percentile(highs, 0.025) > EDGE_SWING
        or percentile(lows, 0.975) - percentile(lows, 0.025) > EDGE_SWING
    ):
        reasons.append("edge-width")
    accuracy = stats["outer_accuracy"]
    lower = stats["outer_accuracy_lower"]
    if stats["outer_answered"] < 1 or accuracy is None or accuracy < 1.0 - TARGET_RISK:
        reasons.append("selective-accuracy")
    if lower is None or lower < 1.0 - TARGET_RISK:
        reasons.append("selective-accuracy-interval")
    adopted = not reasons
    return {
        "adopted": adopted,
        "reasons": reasons,
        "low": median(lows) if adopted else None,
        "high": median(highs) if adopted else None,
        "keep": list(PLACEHOLDER_BAND),
    }


def run_study(probs: list[float], labels: list[int], bootstrap: int = BOOTSTRAP) -> dict:
    """Score probabilities. The band is never scored on the phrases that set it."""
    if len(probs) != len(labels) or any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 or 1, one per probability")
    outer = assign_outer(labels)
    inner = [i for i in range(len(labels)) if i not in outer]
    raw_oof: list[list[float]] = [[] for _ in labels]
    temp_oof: list[list[float]] = [[] for _ in labels]
    calibrated_oof: list[list[float]] = [[] for _ in labels]
    for repeat in range(REPEATS):
        folds = assign_folds(labels, inner, repeat)
        for fold in range(FOLDS):
            train = [i for i in inner if folds[i] != fold]
            test = [i for i in inner if folds[i] == fold]
            if not test or not train:
                continue
            temperature = fit_temperature(*_subset(probs, labels, train))
            for index in test:
                calibrated = apply_temperature(probs[index], temperature)
                raw_oof[index].append((probs[index] - labels[index]) ** 2)
                temp_oof[index].append((calibrated - labels[index]) ** 2)
                calibrated_oof[index].append(calibrated)
    inner_raw = [raw_oof[i] for i in inner if raw_oof[i]]
    inner_temp = [temp_oof[i] for i in inner if temp_oof[i]]
    brier_raw, brier_raw_lo, brier_raw_hi = _phrase_bootstrap(inner_raw, SEED + 20011, bootstrap)
    brier_temp, brier_temp_lo, brier_temp_hi = _phrase_bootstrap(inner_temp, SEED + 20012, bootstrap)
    # One temperature for the outer probabilities, fit on the whole inner set.
    outer_temperature = fit_temperature(*_subset(probs, labels, inner))
    outer_raw_phrases = [[(probs[i] - labels[i]) ** 2] for i in sorted(outer)]
    outer_temp_phrases = [
        [(apply_temperature(probs[i], outer_temperature) - labels[i]) ** 2] for i in sorted(outer)
    ]
    outer_brier_raw, outer_brier_raw_lo, outer_brier_raw_hi = _phrase_bootstrap(
        outer_raw_phrases, SEED + 20013, bootstrap
    )
    outer_brier_temp, outer_brier_temp_lo, outer_brier_temp_hi = _phrase_bootstrap(
        outer_temp_phrases, SEED + 20014, bootstrap
    )
    edges = bootstrap_edges(probs, labels, inner, SEED + 10007, bootstrap)
    candidate = None
    if len(edges["low"]) >= 2:
        candidate = (median(edges["low"]), median(edges["high"]))
    answered = 0
    correct = 0
    if candidate is not None:
        low, high = candidate
        for index in sorted(outer):
            probability = probs[index]
            if low < probability < high:
                continue
            answered += 1
            if (probability >= 0.5) == bool(labels[index]):
                correct += 1
    accuracy = (correct / answered) if answered else None
    corp_probs = []
    corp_labels = []
    for index in inner:
        if calibrated_oof[index]:
            corp_probs.append(sum(calibrated_oof[index]) / len(calibrated_oof[index]))
            corp_labels.append(labels[index])
    stats = {
        "n": len(labels),
        "n_clean": sum(labels),
        "n_not_clean": len(labels) - sum(labels),
        "n_inner": len(inner),
        "n_outer": len(outer),
        "outer": sorted(outer),
        "brier_raw": brier_raw,
        "brier_raw_low": brier_raw_lo,
        "brier_raw_high": brier_raw_hi,
        "brier_temperature": brier_temp,
        "brier_temperature_low": brier_temp_lo,
        "brier_temperature_high": brier_temp_hi,
        "outer_brier_raw": outer_brier_raw,
        "outer_brier_raw_low": outer_brier_raw_lo,
        "outer_brier_raw_high": outer_brier_raw_hi,
        "outer_brier_temperature": outer_brier_temp,
        "outer_brier_temperature_low": outer_brier_temp_lo,
        "outer_brier_temperature_high": outer_brier_temp_hi,
        "outer_temperature": outer_temperature,
        "edge_low": edges["low"],
        "edge_high": edges["high"],
        "n_bootstrap": edges["n"],
        "n_no_interior": edges["no_interior"],
        "outer_answered": answered,
        "outer_correct": correct,
        "outer_accuracy": accuracy,
        "outer_accuracy_lower": clopper_accuracy_lower(correct, answered),
        "candidate_low": None if candidate is None else candidate[0],
        "candidate_high": None if candidate is None else candidate[1],
        "corp": _corp(corp_probs, corp_labels, bootstrap),
        "oof_temperature_mean": [
            (sum(calibrated_oof[i]) / len(calibrated_oof[i])) if calibrated_oof[i] else None
            for i in range(len(labels))
        ],
    }
    stats["decision"] = decide(stats)
    return stats


def _corp(probs: list[float], labels: list[int], draws_n: int) -> dict:
    if not probs:
        return {"probs": [], "fitted": [], "band_low": [], "band_high": [], "outside": 0}
    fitted = pav(probs, labels)
    rng = random.Random(SEED + 30011)
    columns = [[] for _ in probs]
    for _ in range(draws_n):
        drawn = [1 if rng.random() < probs[i] else 0 for i in range(len(probs))]
        resampled = pav(probs, drawn)
        for index, value in enumerate(resampled):
            columns[index].append(value)
    band_low = [percentile(column, 0.025) for column in columns]
    band_high = [percentile(column, 0.975) for column in columns]
    outside = sum(
        1
        for index, value in enumerate(fitted)
        if value < band_low[index] or value > band_high[index]
    )
    return {
        "probs": probs,
        "fitted": fitted,
        "band_low": band_low,
        "band_high": band_high,
        "outside": outside,
    }


_TAKE_RE = re.compile(TAKE_ID)
_RESULT_START = "<!-- result -->"
_RESULT_END = "<!-- /result -->"


def assert_question() -> None:
    for text in (INSTRUCTIONS, TRUE_CRITERION, FALSE_CRITERION):
        if _GATE_WORD.search(text):
            raise EarsError("the question asks Jev to pass or fail")


def build_phrase(song: str, mix: str, phrase: dict, timing: dict, pitch: dict, scores: dict):
    take = phrase.get("take")
    if not isinstance(take, str) or _TAKE_RE.match(take) is None:
        raise EarsError(f"{song}/{mix} take id is not a short token")
    index = phrase.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise EarsError(f"{song}/{mix} phrase index is not an int")
    start = float(phrase["start"])
    end = float(phrase["end"])
    sliced_timing, sliced_pitch = slice_phrase(timing, pitch, start, end)
    record = from_jam_take(
        take_id=take,
        phrase_id=f"{song}/{mix}/{index}",
        timing=sliced_timing,
        pitch=sliced_pitch,
        phrase_scores=scores,
        phrase_index=str(index),
    )
    state = record.to_state()
    if state.get("marks"):
        raise EarsError("marks leaked into the record")
    keys: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(str(key).casefold())
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(state)
    if "verdict" in keys:
        raise EarsError("a verdict leaked into the record")
    if "review-marks" in json.dumps(state):
        raise EarsError("marks schema leaked into the record")
    return record, state, start, end, take


def _scores_path(song: str, kind: str) -> Path:
    folder = "sing-pad" if kind == "pad" else "sing"
    return VOCAL_ROOT / folder / song / "phrase-scores.json"


def _mark_times() -> dict[tuple[str, str], list[float]]:
    document = json.loads(MARKS_PATH.read_text(encoding="utf-8"))
    document.pop("note", None)
    entries = document.get("entries")
    sequence = entries.values() if isinstance(entries, dict) else entries
    found: dict[tuple[str, str], list[float]] = {}
    for entry in sequence or []:
        if not isinstance(entry, dict):
            raise EarsError("a marks entry is not an object")
        entry.pop("note", None)
        rel = str(entry.get("dir", "")).replace("\\", "/").rstrip("/")
        matched = None
        for song, mix, _kind in MIXES:
            if rel.endswith(f"{song}/{mix}"):
                matched = (song, mix)
                break
        if matched is None:
            tail = "/".join(rel.split("/")[-2:])
            raise EarsError(f"marks entry {tail} is not one of the seven mixes")
        if matched in found:
            raise EarsError(f"marks entry {matched[0]}/{matched[1]} is repeated")
        times = []
        for mark in entry.get("marks") or []:
            if isinstance(mark, dict) and "t" in mark:
                mark.pop("note", None)
                times.append(float(mark["t"]))
        found[matched] = times
    missing = [(song, mix) for song, mix, _kind in MIXES if (song, mix) not in found]
    if missing:
        raise EarsError("a mix has no marks entry")
    return found


def load_items() -> list[dict]:
    assert_question()
    times = _mark_times()
    items = []
    for song, mix, kind in MIXES:
        mix_dir = VOCAL_ROOT / "sing" / song / mix
        plan = json.loads((mix_dir / "plan.json").read_text(encoding="utf-8"))
        timing = json.loads((mix_dir / "receipt.json").read_text(encoding="utf-8"))
        pitch = json.loads((mix_dir / "pitch.json").read_text(encoding="utf-8"))
        scores = json.loads(_scores_path(song, kind).read_text(encoding="utf-8"))
        for phrase in plan["phrases"]:
            record, state, start, end, take = build_phrase(song, mix, phrase, timing, pitch, scores)
            index = phrase["index"]
            items.append(
                {
                    "id": f"{song}/{mix}/{index}",
                    "song": song,
                    "mix": mix,
                    "index": index,
                    "take": take,
                    "start": start,
                    "end": end,
                    "label": phrase_label(start, end, times[(song, mix)]),
                    "record": record,
                    "timing_n": len(state["timing"]),
                    "pitch_n": len(state["pitch"]),
                    "bytes": len(json.dumps(state)),
                }
            )
    if len(items) != EXPECTED_PHRASES:
        raise EarsError(f"expected {EXPECTED_PHRASES} phrases, found {len(items)}")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise EarsError("phrase ids are not unique")
    sizes = sorted(item["timing_n"] for item in items)
    if sizes[len(sizes) // 2] == 0:
        raise EarsError("the phrase slice kept no timing rows")
    if any(item["bytes"] > 80000 for item in items):
        raise EarsError("a phrase record is too large to send")
    return items


def _dry_line(items: list[dict]) -> str:
    clean = sum(item["label"] for item in items)
    sizes = sorted(item["bytes"] for item in items)
    empty = sum(1 for item in items if item["timing_n"] == 0 and item["pitch_n"] == 0)
    return (
        f"DRY n={len(items)} clean={clean} not_clean={len(items) - clean} "
        f"bytes_median={sizes[len(sizes) // 2]} empty={empty}"
    )


def _cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    calls = payload.get("calls")
    return calls if isinstance(calls, dict) else {}


def _save_cache(calls: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps({"model": PINNED_MODEL, "dated": PINNED_DATE, "calls": calls}, indent=2),
        encoding="utf-8",
    )


def call_jev(items: list[dict]) -> bool:
    assert_question()
    if os.environ.get("OPENROUTER_API_KEY", "").strip() == "":
        print("FAILED no-key")
        return False
    client = create_decisions_client(os.environ["OPENROUTER_API_KEY"])
    calls = _cache()
    spent = sum(float(row.get("cost") or 0.0) for row in calls.values())
    last = None
    for item in items:
        if item["id"] in calls:
            continue
        if spent >= CAP_USD or (last is not None and spent + last > CAP_USD):
            print(f"CAP spent={spent:.4f} last={0.0 if last is None else last:.4f}")
            return False
        result = phrase_clean(
            item["record"],
            instructions=INSTRUCTIONS,
            true=TRUE_CRITERION,
            false=FALSE_CRITERION,
            client=client,
        )
        if result.model != PINNED_MODEL or result.dated != PINNED_DATE:
            print("FAILED pin")
            return False
        if result.cost is None:
            print("FAILED no-cost")
            return False
        cost = float(result.cost)
        calls[item["id"]] = {"p_yes": float(result.probabilities["yes"]), "cost": cost}
        _save_cache(calls)
        spent += cost
        last = cost
        if spent > CAP_USD:
            print(f"CAP spent={spent:.4f} last={last:.4f}")
            return False
    print("DONE")
    return True


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _interval(low: object, high: object) -> str:
    return f"{_fmt(low)} to {_fmt(high)}"


def _reason_text(reason: str) -> str:
    return {
        "class-count": "a class has fewer than 30 phrases",
        "no-interior-fraction": "more than 5% of the inner resamples had no interior band",
        "edge-width": "a 95% interval on a band edge is wider than 0.10",
        "selective-accuracy": "outer accuracy is under 90%, or the outer band answered nothing",
        "selective-accuracy-interval": "the outer 95% lower bound on accuracy is under 90%",
    }.get(reason, reason)


def _svg(corp: dict) -> str:
    order = sorted(range(len(corp["probs"])), key=lambda i: (corp["probs"][i], i))
    left, right, top, bottom = 56, 600, 28, 372

    def x_of(probability: float) -> float:
        return left + (right - left) * probability

    def y_of(probability: float) -> float:
        return bottom - (bottom - top) * probability

    def step(key: str) -> str:
        if not order:
            return ""
        commands = [f"M {x_of(corp['probs'][order[0]]):.2f} {y_of(corp[key][order[0]]):.2f}"]
        for index in order[1:]:
            commands.append(f"H {x_of(corp['probs'][index]):.2f} V {y_of(corp[key][index]):.2f}")
        commands.append(f"H {x_of(1):.2f}")
        return " ".join(commands)

    band = ""
    if order:
        forward = " ".join(
            f"L {x_of(corp['probs'][i]):.2f} {y_of(corp['band_high'][i]):.2f}" for i in order
        )
        backward = " ".join(
            f"L {x_of(corp['probs'][i]):.2f} {y_of(corp['band_low'][i]):.2f}" for i in reversed(order)
        )
        first = order[0]
        band = (
            f"M {x_of(corp['probs'][first]):.2f} {y_of(corp['band_high'][first]):.2f} "
            f"{forward} {backward} Z"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
  <rect width="640" height="420" fill="#ffffff"/>
  <path d="{band}" fill="#d9e4f5"/>
  <line x1="{x_of(0):.2f}" y1="{y_of(0):.2f}" x2="{x_of(1):.2f}" y2="{y_of(1):.2f}" stroke="#888888"/>
  <path d="{step('fitted')}" fill="none" stroke="#1f4e79" stroke-width="2"/>
  <text x="320" y="408" text-anchor="middle" font-family="sans-serif" font-size="14">P(yes), temperature-scaled, inner out-of-fold</text>
  <text x="16" y="200" font-family="sans-serif" font-size="14" transform="rotate(-90 16 200)">observed frequency</text>
</svg>
"""


def _public_summary(stats: dict, spent: float, calls: int) -> dict:
    decision = stats["decision"]
    lows = stats["edge_low"]
    highs = stats["edge_high"]
    edge = None
    if len(lows) >= 2:
        edge = {
            "low": {
                "median": median(lows),
                "p2.5": percentile(lows, 0.025),
                "p97.5": percentile(lows, 0.975),
            },
            "high": {
                "median": median(highs),
                "p2.5": percentile(highs, 0.025),
                "p97.5": percentile(highs, 0.975),
            },
            "shipped": False,
        }
    return {
        "model": PINNED_MODEL,
        "dated": PINNED_DATE,
        "n": stats["n"],
        "n_clean": stats["n_clean"],
        "n_not_clean": stats["n_not_clean"],
        "n_inner": stats["n_inner"],
        "n_outer": stats["n_outer"],
        "brier_raw_inner": {
            "point": stats["brier_raw"],
            "low": stats["brier_raw_low"],
            "high": stats["brier_raw_high"],
        },
        "brier_temperature_inner": {
            "point": stats["brier_temperature"],
            "low": stats["brier_temperature_low"],
            "high": stats["brier_temperature_high"],
        },
        "brier_raw_outer": {
            "point": stats["outer_brier_raw"],
            "low": stats["outer_brier_raw_low"],
            "high": stats["outer_brier_raw_high"],
        },
        "brier_temperature_outer": {
            "point": stats["outer_brier_temperature"],
            "low": stats["outer_brier_temperature_low"],
            "high": stats["outer_brier_temperature_high"],
        },
        "corp_outside": stats["corp"]["outside"],
        "corp_n": len(stats["corp"]["probs"]),
        "edge_diagnostic": edge,
        "no_interior_replicates": stats["n_no_interior"],
        "bootstrap": stats["n_bootstrap"],
        "outer_answered": stats["outer_answered"],
        "outer_correct": stats["outer_correct"],
        "outer_accuracy": stats["outer_accuracy"],
        "outer_accuracy_lower": stats["outer_accuracy_lower"],
        "decision": decision,
        "cost_usd": round(spent, 6),
        "calls": calls,
        "placeholder_band": list(PLACEHOLDER_BAND),
    }


def _result_markdown(summary: dict) -> str:
    decision = summary["decision"]
    reasons = decision["reasons"]
    if decision["adopted"]:
        outcome = (
            f"The held-out rule adopts a band from {decision['low']:.4f} to {decision['high']:.4f}. "
            "That number is not in the question yet. The placeholder stays until its own pull request."
        )
    else:
        why = "; ".join(_reason_text(reason) for reason in reasons)
        outcome = (
            "The data does not support a band yet. The placeholder 0.35–0.65 stays. "
            f"The gates that fired: {why}."
        )
    edge = summary["edge_diagnostic"]
    if edge is None:
        edge_line = "No interior band showed up often enough to quote an edge."
    else:
        edge_line = (
            "Inner-resample edges, not shipped: "
            f"low {_interval(edge['low']['p2.5'], edge['low']['p97.5'])} "
            f"(median {_fmt(edge['low']['median'])}), "
            f"high {_interval(edge['high']['p2.5'], edge['high']['p97.5'])} "
            f"(median {_fmt(edge['high']['median'])})."
        )
    raw = summary["brier_raw_inner"]
    scaled = summary["brier_temperature_inner"]
    return "\n".join(
        [
            f"Model `{summary['model']}`, stamp `{summary['dated']}`. "
            f"{summary['calls']} calls, ${summary['cost_usd']:.4f} of the $0.25 cap.",
            "",
            f"Phrases: {summary['n']}. Clean: {summary['n_clean']}. Not clean: {summary['n_not_clean']}. "
            f"Inner {summary['n_inner']}, outer holdout {summary['n_outer']}.",
            "",
            f"Inner out-of-fold Brier, raw P(yes): {_fmt(raw['point'])} "
            f"(bootstrap {_interval(raw['low'], raw['high'])}).",
            f"Inner out-of-fold Brier, temperature-scaled: {_fmt(scaled['point'])} "
            f"(bootstrap {_interval(scaled['low'], scaled['high'])}).",
            f"CORP diagram: docs/calibration/corp.svg. "
            f"{summary['corp_outside']} of {summary['corp_n']} inner phrases sit outside the 95% consistency band.",
            "",
            edge_line,
            f"Outer phrases answered by the diagnostic band: {summary['outer_answered']}. "
            f"Accuracy {_fmt(summary['outer_accuracy'])}, "
            f"95% lower bound {_fmt(summary['outer_accuracy_lower'])}.",
            "",
            outcome,
            "",
            "Binned ECE is not reported. Isotonic regression is the CORP diagram only.",
        ]
    )


def _forbid(text: str) -> None:
    for needle in ("C:\\Users", "C:/Users", "ajs-fullsong", "director-marks", "OPENROUTER", "100.64."):
        if needle in text:
            raise EarsError("a report file contains a local path or a key marker")


def write_report(items: list[dict]) -> None:
    calls = _cache()
    missing = [item["id"] for item in items if item["id"] not in calls]
    if missing:
        print(f"FAILED incomplete {len(missing)}")
        raise SystemExit(2)
    probs = [float(calls[item["id"]]["p_yes"]) for item in items]
    labels = [item["label"] for item in items]
    stats = run_study(probs, labels)
    spent = sum(float(calls[item["id"]]["cost"]) for item in items)
    summary = _public_summary(stats, spent, len(items))
    phrases = []
    outer = set(stats["outer"])
    for index, item in enumerate(items):
        probability = float(calls[item["id"]]["p_yes"])
        if index in outer:
            scaled = apply_temperature(probability, stats["outer_temperature"])
            fit = "outer-holdout"
        else:
            scaled = stats["oof_temperature_mean"][index]
            fit = "inner-oof"
        phrases.append(
            {
                "song": item["song"],
                "mix": item["mix"],
                "index": item["index"],
                "start": item["start"],
                "end": item["end"],
                "clean": item["label"] == 1,
                "held_out": index in outer,
                "p_yes": probability,
                "p_yes_temperature": scaled,
                "temperature_fit": fit,
            }
        )
    body = _result_markdown(summary)
    _forbid(body)
    summary_text = json.dumps(summary, indent=2)
    phrases_text = json.dumps(
        {
            "model": PINNED_MODEL,
            "dated": PINNED_DATE,
            "question": INSTRUCTIONS,
            "clean_means": "no review-mark window [t - 1.0, t + 0.2] overlaps the phrase",
            "phrases": phrases,
        },
        indent=2,
    )
    svg = _svg(stats["corp"])
    _forbid(summary_text)
    _forbid(phrases_text)
    _forbid(svg)
    document = DOC_PATH.read_text(encoding="utf-8")
    if _RESULT_START not in document or _RESULT_END not in document:
        raise EarsError("the calibration doc has no result marker")
    before, rest = document.split(_RESULT_START, 1)
    _old, after = rest.split(_RESULT_END, 1)
    DOC_PATH.write_text(
        f"{before}{_RESULT_START}\n\n{body}\n\n{_RESULT_END}{after}",
        encoding="utf-8",
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary_text + "\n", encoding="utf-8")
    PHRASES_PATH.write_text(phrases_text + "\n", encoding="utf-8")
    CORP_PATH.write_text(svg, encoding="utf-8")
    print("REPORT")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    try:
        if command == "--dry-run":
            print(_dry_line(load_items()))
            return 0
        if command == "--call":
            items = load_items()
            if not call_jev(items):
                return 2
            write_report(items)
            return 0
        if command == "--report":
            write_report(load_items())
            return 0
    except EarsError as err:
        print(f"FAILED {err}")
        return 2
    print("usage: phrase_calibration.py --dry-run|--call|--report")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

