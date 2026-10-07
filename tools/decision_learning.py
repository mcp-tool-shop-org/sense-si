"""Follow-up to the phrase calibration. The band in the question stays 0.35–0.65.

Locked before this file scored a phrase. Do not retune the constants after
seeing a Brier score, a fold, or a Jev probability.

The pinned Decisions API returns a probability. It does not return rule text,
so the FeatLLM arm is a logistic regression on rules written once from the
feature definitions. Jev is not asked to invent them.

New Jev calls stay inside the original $0.25 cap, counting the calibration
spend that is already recorded.

The winner split is leave-one-mix-out on every phrase. The ungrouped
repeated cross-validation is not a claim.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import phrase_calibration as calibration  # noqa: E402
from decisions import (  # noqa: E402
    PINNED_DATE,
    PINNED_MODEL,
    DecisionRequest,
    DecisionsError,
    NoulQuestion,
    create_decisions_client,
)

SEED = calibration.SEED
FOLDS = calibration.FOLDS
REPEATS = calibration.REPEATS
HOLDOUT_SEED = 20261008
HOLDOUT_N = 20
COLLAR_S = 0.5
RESOLVE_BRIER = 0.02
LABEL_BOOTSTRAP_SEED = SEED + 40001
EDGE_BOOTSTRAP_SEED = SEED + 40002
PRIOR_SPEND_USD = 0.021653
SAFETY_USD = 0.002
CAP_USD = 0.25
UNIT_PRIOR = PRIOR_SPEND_USD / calibration.EXPECTED_PHRASES
STATE_BYTE_CAP = 80_000

FEATURE_NAMES = (
    "timing_n",
    "undated_n",
    "undated_fraction",
    "timing_abs_median",
    "timing_dated_n",
    "pitch_n",
    "unvoiced_n",
    "untrackable_n",
    "null_median_n",
    "pitch_abs_median",
    "tuning_abs_offset",
    "tuning_scatter",
    "tuning_missing",
    "has_transcript",
    "words",
    "matched",
    "match_ratio",
    "intelligibility",
    "intelligibility_missing",
    "notes",
    "mean_abs_cents",
    "mean_abs_cents_missing",
)
TIMING_FEATURES = FEATURE_NAMES[:5]
PITCH_FEATURES = FEATURE_NAMES[5:13]
TRANSCRIPT_FEATURES = FEATURE_NAMES[13:]
JOIN_FEATURES = ("joins", "switches")
SEGMENT_FEATURES = (
    "air_ms_max",
    "shift_diff_ms_max",
    "shift_spread_ms",
    "stretch_min",
    "stretch_max",
    "stretch_missing",
    "segment_boundary_s",
    "segment_boundary_missing",
)
MEASURE_FEATURES = (
    "spectral_jump_max",
    "repeat_similarity_max",
    "click_z_max",
    "f0_step_cents_max",
    "f0_step_missing",
    "pct_max",
    "octave_jumps",
    "pitch_step_cents_max",
    "voicing_flips",
)
EVIDENCE_FEATURES = JOIN_FEATURES + SEGMENT_FEATURES + MEASURE_FEATURES
TIMING_PITCH_FEATURES = TIMING_FEATURES + PITCH_FEATURES
PLUS_JOINS_FEATURES = TIMING_PITCH_FEATURES + JOIN_FEATURES
PLUS_SEGMENT_FEATURES = PLUS_JOINS_FEATURES + SEGMENT_FEATURES
PLUS_MEASURE_FEATURES = PLUS_SEGMENT_FEATURES + MEASURE_FEATURES
ALL_FEATURES = FEATURE_NAMES + EVIDENCE_FEATURES
GROUP_SEED = SEED + 70001
PROBE_SEED = SEED + 60000
MIX_PROBE_MARGIN = 0.20
# Published on this branch before the folds were grouped. Not a claim.
UNGROUPED_TREE = (
    "Ungrouped repeated cross-validation on the 104, shallow tree minus logistic, "
    "was -0.0719 (-0.1298 to -0.0141). Unclaimed: possible mix leakage."
)
# Compared with logistic on every feature, including the evidence family.
# Every other extended row is compared with logistic on the twenty-two
# receipt features. Locked before the grouped rerun.
# The only cut-placement mix. Secondary analysis leaves it out of both sides.
# Declared before that analysis was scored. The primary table keeps it.
CUT_PLACEMENT_MIX = "amazing-grace-new-britain:phrase16"
DISPLAY_NAMES = {
    "gbdt": "shallow tree, 22 receipt features",
    "gbdt-all": "shallow tree, 41 features",
}
EVIDENCE_COMPARED = frozenset(
    {
        "logistic-timing-pitch",
        "logistic-plus-joins",
        "logistic-plus-segment",
        "logistic-plus-measures",
        "gbdt-all",
        "tabpfn-all",
    }
)

# Shot answers use the extended-window label. full-16 cannot fit the state cap.
CELLS = (
    ("pruned", 0),
    ("rows", 0),
    ("pruned", 4),
    ("pruned", 8),
    ("pruned", 16),
    ("full", 4),
    ("full", 8),
    ("rows", 4),
    ("rows", 8),
    ("rows", 16),
)

GBDT_TREES = 50
GBDT_DEPTH = 2
GBDT_RATE = 0.1
LR_C = 1.0

CACHE_PATH = Path(os.environ.get("TEMP", ".")) / "sense-si-jev-serial-cache.json"
DOC_PATH = ROOT / "docs" / "decision-learning.md"
SUMMARY_PATH = ROOT / "docs" / "decision-learning" / "summary.json"
PHRASES_PATH = ROOT / "docs" / "decision-learning" / "phrases.json"
_RESULT_START = "<!-- result -->"
_RESULT_END = "<!-- /result -->"


def strict_overlaps(start: float, end: float, mark_t: float) -> bool:
    """The mark time itself falls inside the phrase. No lag."""
    return start <= mark_t <= end


def strict_label(start: float, end: float, times: list[float]) -> int:
    for mark_t in times:
        if strict_overlaps(start, end, mark_t):
            return 0
    return 1


def tag_phrase(start: float, end: float, times: list[float]) -> str:
    """certain, edge, or mid. Disagreement is edge. A short phrase has no mid."""
    extended = calibration.phrase_label(start, end, times)
    strict = strict_label(start, end, times)
    if strict != extended:
        return "edge"
    counted = [mark_t for mark_t in times if calibration.mark_overlaps(start, end, mark_t)]
    if extended == 0 and counted:
        interior = start + COLLAR_S < end - COLLAR_S
        if interior and all(start + COLLAR_S < mark_t < end - COLLAR_S for mark_t in counted):
            return "mid"
        return "edge"
    return "certain"


def frozen_test_indices(n: int, k: int = HOLDOUT_N, seed: int = HOLDOUT_SEED) -> list[int]:
    """Shot-pool exclusion for this serialisation sweep. Not a test."""
    if k >= n:
        raise ValueError("the holdout has to leave a training phrase")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return sorted(order[:k])


def room_for_cell(spent_new: float, unit: float, n: int = calibration.EXPECTED_PHRASES) -> bool:
    return PRIOR_SPEND_USD + SAFETY_USD + spent_new + n * unit <= CAP_USD


def _betacf(a: float, b: float, x: float) -> float:
    front = math.exp(
        a * math.log(x)
        + b * math.log(1.0 - x)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    ) / a
    f = c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 201):
        num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        d = 1.0 + num * d
        c = 1.0 + num / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        c = 1.0 + num / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * f


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betacf(b, a, 1.0 - x)
    return _betacf(a, b, x)


def student_t_975(df: int) -> float:
    """Two-sided 95% critical value of Student's t."""
    if df < 1:
        raise ValueError("t degrees of freedom start at 1")
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        x = df / (df + mid * mid)
        tail = 0.5 * regularized_beta(x, df / 2.0, 0.5)
        if tail > 0.025:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nadeau_bengio(
    differences: list[float], n_train: int, n_test: int, ratio: float | None = None
) -> dict:
    """Corrected interval for a fold difference. Claim only past 0.02."""
    count = len(differences)
    mean = sum(differences) / count
    if count < 2 or n_train < 1 or n_test < 1:
        return {"mean": mean, "low": None, "high": None, "resolvable": False}
    variance = sum((item - mean) ** 2 for item in differences) / (count - 1)
    used = (n_test / n_train) if ratio is None else ratio
    corrected = (1.0 / count + used) * variance
    half = student_t_975(count - 1) * math.sqrt(corrected)
    low, high = mean - half, mean + half
    resolvable = abs(mean) >= RESOLVE_BRIER and (high < 0.0 or low > 0.0)
    return {"mean": mean, "low": low, "high": high, "resolvable": resolvable}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _median_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(abs(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _obs_number(observations: tuple, key: str) -> float | None:
    for name, value in observations:
        if name == key:
            return _finite(value)
    return None


def feature_row(record) -> dict[str, float]:
    """Numeric families only. Transcript text is not a feature."""
    undated = [row for row in record.timing if row.measurement_state == "undated"]
    dated = [
        row.value
        for row in record.timing
        if row.measurement_state != "undated" and _finite(row.value) is not None
    ]
    timing_n = len(record.timing)
    unvoiced = sum(1 for row in record.pitch if row.measurement_state == "unvoiced")
    untrackable = sum(1 for row in record.pitch if row.measurement_state == "untrackable")
    null_median = sum(1 for row in record.pitch if row.value is None)
    pitch_values = [row.value for row in record.pitch if _finite(row.value) is not None]
    tuning_missing = 1.0 if record.tuning is None else 0.0
    offset = None if record.tuning is None else _finite(record.tuning.global_offset_cents)
    scatter = None if record.tuning is None else _finite(record.tuning.scatter_sd_cents)
    transcript = record.transcript
    has_transcript = 0.0 if transcript is None else 1.0
    observations = () if transcript is None else transcript.observations
    words = _obs_number(observations, "words")
    matched = _obs_number(observations, "matched")
    intelligibility = _obs_number(observations, "intelligibility")
    notes = _obs_number(observations, "notes")
    cents = _obs_number(observations, "mean_abs_cents")
    word_count = 0.0 if words is None else words
    matched_count = 0.0 if matched is None else matched
    ratio = matched_count / word_count if word_count else 0.0
    return {
        "timing_n": float(timing_n),
        "undated_n": float(len(undated)),
        "undated_fraction": (len(undated) / timing_n) if timing_n else 0.0,
        "timing_abs_median": _median_abs(dated),
        "timing_dated_n": float(len(dated)),
        "pitch_n": float(len(record.pitch)),
        "unvoiced_n": float(unvoiced),
        "untrackable_n": float(untrackable),
        "null_median_n": float(null_median),
        "pitch_abs_median": _median_abs(pitch_values),
        "tuning_abs_offset": 0.0 if offset is None else abs(offset),
        "tuning_scatter": 0.0 if scatter is None else scatter,
        "tuning_missing": tuning_missing,
        "has_transcript": has_transcript,
        "words": word_count,
        "matched": matched_count,
        "match_ratio": ratio,
        "intelligibility": 0.0 if intelligibility is None else intelligibility,
        "intelligibility_missing": 1.0 if intelligibility is None else 0.0,
        "notes": 0.0 if notes is None else notes,
        "mean_abs_cents": 0.0 if cents is None else cents,
        "mean_abs_cents_missing": 1.0 if cents is None else 0.0,
    }


def _or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def evidence_feature_row(evidence) -> dict[str, float]:
    """Phrase-level join and segment numbers. A missing reading is 0 plus a flag."""
    if evidence is None:
        blank = {name: 0.0 for name in EVIDENCE_FEATURES}
        blank["stretch_missing"] = 1.0
        blank["segment_boundary_missing"] = 1.0
        blank["f0_step_missing"] = 1.0
        return blank
    flips = sum(1 for join in evidence.at_joins if join.voicing_flip)
    return {
        "joins": float(evidence.joins),
        "switches": float(evidence.switches),
        "air_ms_max": _or_zero(evidence.air_ms_max),
        "shift_diff_ms_max": _or_zero(evidence.shift_diff_ms_max),
        "shift_spread_ms": _or_zero(evidence.shift_spread_ms),
        "stretch_min": _or_zero(evidence.stretch_min),
        "stretch_max": _or_zero(evidence.stretch_max),
        "stretch_missing": 1.0 if evidence.stretch_min is None or evidence.stretch_max is None else 0.0,
        "segment_boundary_s": _or_zero(evidence.segment_boundary_s),
        "segment_boundary_missing": 1.0 if evidence.segment_boundary_s is None else 0.0,
        "spectral_jump_max": _or_zero(evidence.spectral_jump_max),
        "repeat_similarity_max": _or_zero(evidence.repeat_similarity_max),
        "click_z_max": _or_zero(evidence.click_z_max),
        "f0_step_cents_max": _or_zero(evidence.f0_step_cents_max),
        "f0_step_missing": 1.0 if evidence.f0_step_cents_max is None else 0.0,
        "pct_max": _or_zero(evidence.pct_max),
        "octave_jumps": float(evidence.octave_jumps),
        "pitch_step_cents_max": _or_zero(evidence.pitch_step_cents_max),
        "voicing_flips": float(flips),
    }


def rule_row(features: dict[str, float]) -> dict[str, float]:
    """Nine rules, written once from the definitions. Not fit to a label."""
    return {
        "any_undated": 1.0 if features["undated_n"] >= 1.0 else 0.0,
        "many_undated": 1.0 if features["undated_fraction"] >= 0.25 else 0.0,
        "any_unpitched": 1.0 if features["unvoiced_n"] + features["untrackable_n"] >= 1.0 else 0.0,
        "any_null_pitch": 1.0 if features["null_median_n"] >= 1.0 else 0.0,
        "wide_pitch": 1.0 if features["pitch_abs_median"] >= 100.0 else 0.0,
        "wide_timing": 1.0 if features["timing_abs_median"] >= 100.0 else 0.0,
        "low_match": 1.0 if features["words"] >= 1.0 and features["match_ratio"] < 0.5 else 0.0,
        "wide_listener_cents": 1.0 if features["mean_abs_cents"] >= 100.0 else 0.0,
        "listener_notes": 1.0 if features["notes"] >= 1.0 else 0.0,
    }


RULE_NAMES = (
    "any_undated",
    "many_undated",
    "any_unpitched",
    "any_null_pitch",
    "wide_pitch",
    "wide_timing",
    "low_match",
    "wide_listener_cents",
    "listener_notes",
)


def _vector(row: dict[str, float], names: tuple[str, ...]) -> list[float]:
    return [float(row[name]) for name in names]


def _standardize(
    train: list[list[float]], test: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    width = len(train[0])
    means = [sum(row[col] for row in train) / len(train) for col in range(width)]
    scales = []
    for col, mean in enumerate(means):
        second = sum((row[col] - mean) ** 2 for row in train) / len(train)
        scales.append(math.sqrt(second) or 1.0)

    def apply(rows: list[list[float]]) -> list[list[float]]:
        return [[(row[col] - means[col]) / scales[col] for col in range(width)] for row in rows]

    return apply(train), apply(test)


def brier(probs: list[float], labels: list[int]) -> float:
    return sum((prob - label) ** 2 for prob, label in zip(probs, labels)) / len(probs)


def log_loss(probs: list[float], labels: list[int]) -> float:
    total = 0.0
    for prob, label in zip(probs, labels):
        clipped = min(1.0 - 1e-6, max(1e-6, prob))
        total += -(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped))
    return total / len(probs)


def _predict_constant(labels: list[int], n_test: int) -> list[float]:
    rate = sum(labels) / len(labels) if labels else 0.5
    return [rate] * n_test


def _fit_sklearn(name: str, train: list[list[float]], labels: list[int], test: list[list[float]], seed: int):
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    x_train = np.asarray(train, dtype=float)
    y_train = np.asarray(labels, dtype=int)
    x_test = np.asarray(test, dtype=float)
    if len(set(labels)) < 2:
        return _predict_constant(labels, len(test))
    if name == "logistic":
        model = LogisticRegression(C=LR_C, solver="lbfgs", max_iter=500)
        model.fit(x_train, y_train)
    elif name == "gbdt":
        model = GradientBoostingClassifier(
            n_estimators=GBDT_TREES,
            max_depth=GBDT_DEPTH,
            learning_rate=GBDT_RATE,
            random_state=seed,
        )
        model.fit(x_train, y_train)
    else:
        raise ValueError(name)
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0] * len(test)
    column = classes.index(1)
    return [float(row[column]) for row in probabilities]


def _fit_tabpfn(train: list[list[float]], labels: list[int], test: list[list[float]]):
    # A batch run must not block on a browser. Weights that are not already
    # available fail the fit, and the caller records not-run.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    import numpy as np
    from tabpfn import TabPFNClassifier

    if len(set(labels)) < 2:
        return _predict_constant(labels, len(test))
    model = TabPFNClassifier()
    model.fit(np.asarray(train, dtype=float), np.asarray(labels, dtype=int))
    probabilities = model.predict_proba(np.asarray(test, dtype=float))
    classes = list(model.classes_)
    column = classes.index(1)
    return [float(row[column]) for row in probabilities]


def repeated_predictions(
    rows: list[list[float]],
    labels: list[int],
    indices: list[int],
    model: str,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Out-of-fold probabilities, plus per-fold Brier and log loss in fold order."""
    losses_brier: list[list[float]] = [[] for _ in rows]
    fold_brier: list[float] = []
    fold_log: list[float] = []
    for repeat in range(REPEATS):
        assignment = calibration.assign_folds(labels, indices, repeat)
        for fold in range(FOLDS):
            test_index = [index for index in indices if assignment[index] == fold]
            train_index = [index for index in indices if assignment[index] != fold]
            if not test_index or not train_index:
                continue
            train = [rows[index] for index in train_index]
            test = [rows[index] for index in test_index]
            y_train = [labels[index] for index in train_index]
            y_test = [labels[index] for index in test_index]
            seed = SEED + 1 + repeat
            if model == "base-rate":
                predicted = _predict_constant(y_train, len(test_index))
            elif model == "tabpfn":
                predicted = _fit_tabpfn(train, y_train, test)
            elif model == "logistic":
                standardized_train, standardized_test = _standardize(train, test)
                predicted = _fit_sklearn(model, standardized_train, y_train, standardized_test, seed)
            else:
                predicted = _fit_sklearn(model, train, y_train, test, seed)
            for index, prob in zip(test_index, predicted):
                losses_brier[index].append(prob)
            fold_brier.append(brier(predicted, y_test))
            fold_log.append(log_loss(predicted, y_test))
    return losses_brier, fold_brier, fold_log


def fixed_fold_scores(probs: list[float], labels: list[int], indices: list[int]) -> tuple[list[float], list[float]]:
    """Score one saved probability on the same folds the fitted models use."""
    fold_brier: list[float] = []
    fold_log: list[float] = []
    for repeat in range(REPEATS):
        assignment = calibration.assign_folds(labels, indices, repeat)
        for fold in range(FOLDS):
            test_index = [index for index in indices if assignment[index] == fold]
            if not test_index:
                continue
            predicted = [probs[index] for index in test_index]
            y_test = [labels[index] for index in test_index]
            fold_brier.append(brier(predicted, y_test))
            fold_log.append(log_loss(predicted, y_test))
    return fold_brier, fold_log


def mix_name(item: dict) -> str:
    return f"{item['song']}:{item['mix']}"


def leave_one_mix_out(items: list[dict]) -> list[tuple[str, list[int], list[int]]]:
    """Each mix is the test fold once. Training never sees that mix."""
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        key = mix_name(item)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(index)
    everyone = list(range(len(items)))
    folds = []
    for key in order:
        test = groups[key]
        held = set(test)
        train = [index for index in everyone if index not in held]
        folds.append((key, train, test))
    return folds


def _predict_fold(model: str, train, y_train, test, seed: int) -> list[float]:
    if model == "base-rate":
        return _predict_constant(y_train, len(test))
    if model == "tabpfn":
        return _fit_tabpfn(train, y_train, test)
    if model == "logistic":
        standardized_train, standardized_test = _standardize(train, test)
        return _fit_sklearn(model, standardized_train, y_train, standardized_test, seed)
    return _fit_sklearn(model, train, y_train, test, seed)


def grouped_predictions(rows, labels, items, model: str):
    """Leave-one-mix-out probabilities. Each phrase is scored once."""
    losses: list[list[float]] = [[] for _ in rows]
    fold_brier: list[float] = []
    fold_log: list[float] = []
    within: list[dict] = []
    for fold_index, (key, train_index, test_index) in enumerate(leave_one_mix_out(items)):
        if not train_index or not test_index:
            continue
        predicted = _predict_fold(
            model,
            [rows[index] for index in train_index],
            [labels[index] for index in train_index],
            [rows[index] for index in test_index],
            GROUP_SEED + fold_index,
        )
        y_test = [labels[index] for index in test_index]
        fold_value = brier(predicted, y_test)
        fold_brier.append(fold_value)
        fold_log.append(log_loss(predicted, y_test))
        for index, prob in zip(test_index, predicted):
            losses[index].append(prob)
        within.append({"mix": key, "n": len(test_index), "brier": fold_value, "log_loss": fold_log[-1]})
    return losses, fold_brier, fold_log, within


def fixed_grouped_scores(probs: list[float], labels: list[int], items: list[dict]):
    fold_brier: list[float] = []
    fold_log: list[float] = []
    within: list[dict] = []
    for key, _train, test_index in leave_one_mix_out(items):
        predicted = [probs[index] for index in test_index]
        y_test = [labels[index] for index in test_index]
        fold_value = brier(predicted, y_test)
        fold_brier.append(fold_value)
        fold_log.append(log_loss(predicted, y_test))
        within.append({"mix": key, "n": len(test_index), "brier": fold_value, "log_loss": fold_log[-1]})
    return fold_brier, fold_log, within


def compare_grouped(candidate: list[float], reference: list[float], items: list[dict]) -> dict:
    """Nadeau–Bengio on leave-one-mix-out, using the mean test/train ratio."""
    differences = [item - ref for item, ref in zip(candidate, reference)]
    ratios = [
        len(test) / len(train)
        for _key, train, test in leave_one_mix_out(items)
        if train and test
    ]
    ratio = sum(ratios) / len(ratios) if ratios else 1.0
    compared = nadeau_bengio(differences, 1, 1, ratio=ratio)
    compared["ratio"] = ratio
    return compared


def _within_mean(within: list[dict]) -> float | None:
    if not within:
        return None
    return sum(row["brier"] for row in within) / len(within)


def mix_identified(accuracy: float, majority: float) -> bool:
    """Locked before the probe is scored. Majority rate plus 0.20."""
    return accuracy >= majority + MIX_PROBE_MARGIN


def _assign_groups(labels: list[int], repeat: int) -> list[int]:
    rng = random.Random(PROBE_SEED + repeat)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    assignment = [0] * len(labels)
    for indices in groups.values():
        order = indices[:]
        rng.shuffle(order)
        for offset, index in enumerate(order):
            assignment[index] = offset % FOLDS
    return assignment


def _fit_gbdt_class(train, labels, test, seed: int) -> list[int]:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=GBDT_TREES,
        max_depth=GBDT_DEPTH,
        learning_rate=GBDT_RATE,
        random_state=seed,
    )
    model.fit(np.asarray(train, dtype=float), np.asarray(labels, dtype=int))
    predicted = model.predict(np.asarray(test, dtype=float))
    return [int(value) for value in predicted]


def mix_probe(items: list[dict], names: tuple[str, ...]) -> dict:
    """Can the same tree name the mix from these features? Every mix is in train and test."""
    missing = _model_missing("gbdt")
    if missing:
        return {"status": "not-run", "reason": missing, "n_features": len(names)}
    rows = _rows(items, names, "features")
    keys: list[str] = []
    labels: list[int] = []
    for item in items:
        key = mix_name(item)
        if key not in keys:
            keys.append(key)
        labels.append(keys.index(key))
    majority = max(labels.count(label) for label in set(labels)) / len(labels)
    correct = 0
    total = 0
    per_mix = {label: [0, 0] for label in range(len(keys))}
    try:
        for repeat in range(REPEATS):
            assignment = _assign_groups(labels, repeat)
            for fold in range(FOLDS):
                test_index = [index for index in range(len(items)) if assignment[index] == fold]
                train_index = [index for index in range(len(items)) if assignment[index] != fold]
                if not test_index or not train_index:
                    continue
                predicted = _fit_gbdt_class(
                    [rows[index] for index in train_index],
                    [labels[index] for index in train_index],
                    [rows[index] for index in test_index],
                    PROBE_SEED + repeat * FOLDS + fold,
                )
                for index, guess in zip(test_index, predicted):
                    total += 1
                    per_mix[labels[index]][1] += 1
                    if guess == labels[index]:
                        correct += 1
                        per_mix[labels[index]][0] += 1
    except Exception:
        return {"status": "not-run", "reason": "fit", "n_features": len(names)}
    accuracy = correct / total if total else 0.0
    recalls = [hits / count for hits, count in per_mix.values() if count]
    macro = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "status": "ok",
        "n_features": len(names),
        "accuracy": accuracy,
        "majority": majority,
        "macro_recall": macro,
        "identified": mix_identified(accuracy, majority),
    }


def _fold_sizes(n_items: int) -> tuple[int, int]:
    n_test = int(round(n_items / FOLDS))
    n_test = min(max(n_test, 1), n_items - 1)
    return n_items - n_test, n_test


def compare_folds(candidate: list[float], reference: list[float], n_items: int) -> dict:
    differences = [item - ref for item, ref in zip(candidate, reference)]
    n_train, n_test = _fold_sizes(n_items)
    compared = nadeau_bengio(differences, n_train, n_test)
    compared["reference"] = "logistic"
    return compared


def cosine_neighbors(
    query: int,
    pool: list[int],
    rows: list[list[float]],
    k: int,
) -> list[int]:
    """Similarity inside the pool. The query is not an example of itself."""
    others = [index for index in pool if index != query]
    if k <= 0 or not others:
        return []
    width = len(rows[0])
    means = [sum(rows[index][col] for index in others) / len(others) for col in range(width)]
    scales = []
    for col, mean in enumerate(means):
        second = sum((rows[index][col] - mean) ** 2 for index in others) / len(others)
        scales.append(math.sqrt(second) or 1.0)

    def project(index: int) -> list[float]:
        return [(rows[index][col] - means[col]) / scales[col] for col in range(width)]

    query_row = project(query)
    query_norm = math.sqrt(sum(value * value for value in query_row)) or 1.0
    scored = []
    for index in others:
        row = project(index)
        norm = math.sqrt(sum(value * value for value in row)) or 1.0
        cosine = sum(left * right for left, right in zip(query_row, row)) / (query_norm * norm)
        scored.append((-cosine, index))
    scored.sort()
    return [index for _score, index in scored[:k]]


def _answer(label: int) -> str:
    return "clean" if label == 1 else "not clean"


def _row_measurement(measure) -> dict:
    return {
        "value": measure.value,
        "unit": measure.unit,
        "measurement_state": measure.measurement_state,
    }


def serialised_state(kind: str, record, features: dict[str, float], examples: list[dict]) -> dict:
    if kind == "full":
        # This sweep's full state was fixed before the evidence family existed.
        # A resumed call has to keep sending that state.
        body = record.to_state()
        body.pop("evidence", None)
    elif kind == "pruned":
        body = {
            "take_id": record.take_id,
            "phrase_id": record.phrase_id,
            "marks": [],
            "features": {key: features[key] for key in FEATURE_NAMES},
        }
    elif kind == "rows":
        numbers = {
            key: features[key]
            for key in (
                "words",
                "matched",
                "match_ratio",
                "intelligibility",
                "notes",
                "mean_abs_cents",
            )
        }
        body = {
            "take_id": record.take_id,
            "phrase_id": record.phrase_id,
            "marks": [],
            "timing": [_row_measurement(row) for row in record.timing],
            "pitch": [_row_measurement(row) for row in record.pitch],
            "transcript_numbers": numbers,
        }
    else:
        raise ValueError(kind)
    if examples:
        body["examples"] = examples
    return body


def _contains_text(value: object) -> bool:
    if isinstance(value, dict):
        if "text" in value:
            return True
        return any(_contains_text(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item) for item in value)
    return False


def state_ok(state: dict, kind: str) -> bool:
    if state.get("marks"):
        return False
    if kind != "full" and _contains_text(state):
        return False
    encoded = json.dumps(state)
    if "review-marks" in encoded or len(encoded.encode("utf-8")) > STATE_BYTE_CAP:
        return False
    return True


def cell_name(kind: str, shots: int) -> str:
    return f"{kind}-{shots}"


def _rows(items: list[dict], names: tuple[str, ...], source: str) -> list[list[float]]:
    return [_vector(item[source], names) for item in items]


def _oof_mean(buckets: list[list[float]]) -> list[float | None]:
    return [(sum(bucket) / len(bucket)) if bucket else None for bucket in buckets]


def bootstrap_delta(left: list[float], right: list[float], seed: int) -> dict:
    point = sum(left) / len(left) - sum(right) / len(right)
    rng = random.Random(seed)
    draws = []
    n = len(left)
    for _ in range(1000):
        chosen = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(left[i] for i in chosen) / n - sum(right[i] for i in chosen) / n)
    low = calibration.percentile(draws, 0.025)
    high = calibration.percentile(draws, 0.975)
    return {
        "mean": point,
        "low": low,
        "high": high,
        "resolvable": abs(point) >= RESOLVE_BRIER and (high < 0.0 or low > 0.0),
    }


def edge_shift(losses: list[float], tags: list[str], seed: int) -> dict:
    """How much dropping edge phrases moves mean squared error."""
    kept = [loss for loss, tag in zip(losses, tags) if tag != "edge"]
    if not losses or not kept:
        return {"mean": None, "low": None, "high": None, "resolvable": False}
    point = sum(losses) / len(losses) - sum(kept) / len(kept)
    rng = random.Random(seed)
    draws = []
    n = len(losses)
    for _ in range(1000):
        chosen = [rng.randrange(n) for _ in range(n)]
        sample = [losses[index] for index in chosen]
        kept_sample = [losses[index] for index in chosen if tags[index] != "edge"]
        if not kept_sample:
            continue
        draws.append(sum(sample) / n - sum(kept_sample) / len(kept_sample))
    if len(draws) < 2:
        return {"mean": point, "low": None, "high": None, "resolvable": False}
    low = calibration.percentile(draws, 0.025)
    high = calibration.percentile(draws, 0.975)
    return {
        "mean": point,
        "low": low,
        "high": high,
        "resolvable": abs(point) >= RESOLVE_BRIER and (high < 0.0 or low > 0.0),
    }


def _load_evidence(song: str, mix: str, index: int):
    from ai_ears.record import phrase_evidence

    path = calibration.VOCAL_ROOT / "sing" / song / mix / "phrase-evidence.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("dir", None)
    return phrase_evidence(document, index)


def prepare() -> list[dict]:
    from dataclasses import replace

    calibration.assert_question()
    items = calibration.load_items()
    times = calibration._mark_times()
    published = json.loads(calibration.PHRASES_PATH.read_text(encoding="utf-8"))
    saved = {
        f"{row['song']}/{row['mix']}/{row['index']}": float(row["p_yes"])
        for row in published["phrases"]
    }
    # The retired random 20 stays out of the shot pool only, so a resumed
    # serialisation call does not change examples in the middle of a cell.
    shot_pool = set(frozen_test_indices(len(items)))
    prepared = []
    for index, item in enumerate(items):
        key = (item["song"], item["mix"])
        evidence = _load_evidence(item["song"], item["mix"], item["index"])
        record = replace(item["record"], evidence=evidence)
        features = feature_row(record)
        features.update(evidence_feature_row(evidence))
        prepared.append(
            {
                "id": item["id"],
                "song": item["song"],
                "mix": item["mix"],
                "index": item["index"],
                "start": item["start"],
                "end": item["end"],
                "record": record,
                "label_extended": item["label"],
                "label_strict": strict_label(item["start"], item["end"], times[key]),
                "tag": tag_phrase(item["start"], item["end"], times[key]),
                "future_test": index in shot_pool,
                "features": features,
                "rules": rule_row(features),
                "p_yes_full_0": saved[item["id"]],
            }
        )
    return prepared


def _model_missing(name: str) -> str | None:
    try:
        if name == "tabpfn":
            import tabpfn  # noqa: F401
        elif name in {"logistic", "gbdt", "featllm"}:
            import sklearn.linear_model  # noqa: F401
            if name == "gbdt":
                import sklearn.ensemble  # noqa: F401
        return None
    except ImportError:
        return "import"


def _label_values(items: list[dict], label_key: str) -> list[int]:
    field = {"extended": "label_extended", "strict": "label_strict"}.get(label_key)
    if field is None:
        raise ValueError(label_key)
    return [int(item[field]) for item in items]


def _fit_scope(items: list[dict], indices: list[int], label_key: str, model: str, names, source: str) -> dict:
    missing = _model_missing(model if model != "featllm" else "logistic")
    if model not in {"base-rate", "logistic", "gbdt", "tabpfn", "featllm"}:
        raise ValueError(model)
    if missing:
        return {"model": model, "status": "not-run", "reason": missing}
    labels = _label_values(items, label_key)
    rows = _rows(items, names, source)
    try:
        buckets, fold_brier, fold_log, within = grouped_predictions(
            rows, labels, items, "logistic" if model == "featllm" else model
        )
    except Exception:
        return {"model": model, "status": "not-run", "reason": "fit"}
    oof = _oof_mean(buckets)
    known = [index for index in indices if oof[index] is not None]
    return {
        "model": model,
        "status": "ok",
        "brier": brier([oof[index] for index in known], [labels[index] for index in known]),
        "log_loss": log_loss([oof[index] for index in known], [labels[index] for index in known]),
        "within_brier": _within_mean(within),
        "within": within,
        "fold_brier": fold_brier,
        "fold_log_loss": fold_log,
        "oof": [None if value is None else value for value in oof],
    }


def _once(items: list[dict], train: list[int], test: list[int], label_key: str, model: str, names, source: str) -> dict:
    missing = _model_missing("logistic" if model == "featllm" else model)
    if missing or model == "base-rate":
        if model != "base-rate":
            return {"model": model, "status": "not-run", "reason": missing}
    labels = _label_values(items, label_key)
    rows = _rows(items, names, source)
    y_train = [labels[index] for index in train]
    y_test = [labels[index] for index in test]
    try:
        if model == "base-rate":
            predicted = _predict_constant(y_train, len(test))
        elif model == "tabpfn":
            predicted = _fit_tabpfn([rows[index] for index in train], y_train, [rows[index] for index in test])
        elif model in {"logistic", "featllm"}:
            train_rows, test_rows = _standardize(
                [rows[index] for index in train], [rows[index] for index in test]
            )
            predicted = _fit_sklearn("logistic", train_rows, y_train, test_rows, SEED)
        else:
            predicted = _fit_sklearn(
                "gbdt", [rows[index] for index in train], y_train, [rows[index] for index in test], SEED
            )
    except Exception:
        return {"model": model, "status": "not-run", "reason": "fit"}
    return {
        "model": model,
        "status": "ok",
        "brier": brier(predicted, y_test),
        "log_loss": log_loss(predicted, y_test),
        "n": len(test),
    }


def _label_movement(items: list[dict]) -> dict:
    probs = [item["p_yes_full_0"] for item in items]
    extended = [item["label_extended"] for item in items]
    strict = [item["label_strict"] for item in items]
    extended_loss = [(prob - label) ** 2 for prob, label in zip(probs, extended)]
    strict_loss = [(prob - label) ** 2 for prob, label in zip(probs, strict)]
    kept = [index for index, item in enumerate(items) if item["tag"] != "edge"]
    kept_loss = [extended_loss[index] for index in kept]
    movement = bootstrap_delta(extended_loss, strict_loss, LABEL_BOOTSTRAP_SEED)
    tags = [item["tag"] for item in items]
    without = None if not kept_loss else sum(kept_loss) / len(kept_loss)
    return {
        "brier_extended": sum(extended_loss) / len(extended_loss),
        "brier_strict": sum(strict_loss) / len(strict_loss),
        "brier_without_edge": without,
        "extended_minus_strict": movement,
        "edge_shift": edge_shift(extended_loss, tags, EDGE_BOOTSTRAP_SEED),
        "n_edge": sum(1 for item in items if item["tag"] == "edge"),
        "n_mid": sum(1 for item in items if item["tag"] == "mid"),
        "n_certain": sum(1 for item in items if item["tag"] == "certain"),
        "n_extended_clean": sum(extended),
        "n_strict_clean": sum(strict),
        "n_flip": sum(left != right for left, right in zip(extended, strict)),
    }


def _versus(candidate: list[float], reference: list[float] | None, ref_name: str, items: list[dict]) -> dict | None:
    if not reference or len(candidate) != len(reference):
        return None
    compared = compare_grouped(candidate, reference, items)
    compared["reference"] = ref_name
    return compared


def _attach_versus(result: dict, receipt_folds, evidence_folds, items: list[dict]) -> None:
    """Receipt rows against the 22-feature logistic. Evidence rows against logistic-all."""
    folds = result.get("fold_brier")
    name = result.get("name")
    if result.get("status") != "ok" or result.get("label") != "extended" or not folds:
        result["versus_logistic"] = None
        return
    if name in {"logistic", "logistic-strict"}:
        result["versus_logistic"] = None
        return
    if name == "logistic-all":
        result["versus_logistic"] = _versus(folds, receipt_folds, "logistic", items)
        return
    if name in EVIDENCE_COMPARED:
        result["versus_logistic"] = _versus(folds, evidence_folds, "logistic-all", items)
        return
    result["versus_logistic"] = _versus(folds, receipt_folds, "logistic", items)


def warp_items(items: list[dict]) -> list[dict]:
    """Warp-placement mixes only. The cut-placement mix is not a member."""
    return [item for item in items if mix_name(item) != CUT_PLACEMENT_MIX]


def _pooled_gap(tree: dict, base: dict) -> float | None:
    """Phrase-weighted Brier, tree minus base rate. Not the fold mean."""
    tree_brier = tree.get("brier")
    base_brier = base.get("brier")
    if isinstance(tree_brier, bool) or isinstance(base_brier, bool):
        return None
    if not isinstance(tree_brier, (int, float)) or not isinstance(base_brier, (int, float)):
        return None
    return float(tree_brier) - float(base_brier)


def _attach_base(fitted: list[dict], items: list[dict]) -> None:
    """41-feature tree minus the base rate. Same bar. Declared before the interval."""
    base = next((row for row in fitted if row["name"] == "base-rate" and row.get("status") == "ok"), None)
    tree = next((row for row in fitted if row["name"] == "gbdt-all" and row.get("status") == "ok"), None)
    if base is None or tree is None:
        return
    compared = _versus(tree["fold_brier"], base["fold_brier"], "base-rate", items)
    if compared is None:
        return
    pooled = _pooled_gap(tree, base)
    if pooled is not None:
        compared["pooled"] = pooled
    tree["versus_base"] = compared


def _stamp_pooled(models: list[dict] | None) -> None:
    """Fill the pooled gap from stored Briers. Does not refit."""
    if not models:
        return
    base = next((row for row in models if row.get("name") == "base-rate" and row.get("status") == "ok"), None)
    tree = next((row for row in models if row.get("name") == "gbdt-all" and row.get("status") == "ok"), None)
    if base is None or tree is None:
        return
    versus = tree.get("versus_base")
    if not isinstance(versus, dict):
        return
    pooled = _pooled_gap(tree, base)
    if pooled is not None:
        versus["pooled"] = pooled


def _fit_table(items: list[dict]) -> dict:
    """Leave-one-mix-out for one set of phrases."""
    everyone = list(range(len(items)))
    specs = (
        ("base-rate", "base-rate", "extended", FEATURE_NAMES, "features", "prevalence"),
        ("logistic", "logistic", "extended", FEATURE_NAMES, "features", "receipt"),
        ("logistic-all", "logistic", "extended", ALL_FEATURES, "features", "all"),
        ("logistic-timing", "logistic", "extended", TIMING_FEATURES, "features", "timing"),
        ("logistic-pitch", "logistic", "extended", PITCH_FEATURES, "features", "pitch"),
        ("logistic-transcript", "logistic", "extended", TRANSCRIPT_FEATURES, "features", "transcript"),
        ("logistic-timing-pitch", "logistic", "extended", TIMING_PITCH_FEATURES, "features", "timing-pitch"),
        ("logistic-plus-joins", "logistic", "extended", PLUS_JOINS_FEATURES, "features", "plus-joins"),
        ("logistic-plus-segment", "logistic", "extended", PLUS_SEGMENT_FEATURES, "features", "plus-segment"),
        ("logistic-plus-measures", "logistic", "extended", PLUS_MEASURE_FEATURES, "features", "plus-measures"),
        ("gbdt", "gbdt", "extended", FEATURE_NAMES, "features", "receipt"),
        ("gbdt-all", "gbdt", "extended", ALL_FEATURES, "features", "all"),
        ("tabpfn", "tabpfn", "extended", FEATURE_NAMES, "features", "receipt"),
        ("tabpfn-all", "tabpfn", "extended", ALL_FEATURES, "features", "all"),
        ("featllm", "featllm", "extended", RULE_NAMES, "rules", "rules"),
        ("logistic-strict", "logistic", "strict", FEATURE_NAMES, "features", "receipt"),
    )
    fitted = []
    for name, model, label_key, names, source, family in specs:
        result = _fit_scope(items, everyone, label_key, model, names, source)
        result["name"] = name
        result["label"] = label_key
        result["family"] = family
        result["n_features"] = 0 if model == "base-rate" else len(names)
        fitted.append(result)
    receipt_folds = next(
        (row["fold_brier"] for row in fitted if row["name"] == "logistic" and row.get("status") == "ok"),
        None,
    )
    evidence_folds = next(
        (row["fold_brier"] for row in fitted if row["name"] == "logistic-all" and row.get("status") == "ok"),
        None,
    )
    for result in fitted:
        _attach_versus(result, receipt_folds, evidence_folds, items)
    _attach_base(fitted, items)
    labels = [item["label_extended"] for item in items]
    probs = [item["p_yes_full_0"] for item in items]
    jev_folds, jev_logs, within = fixed_grouped_scores(probs, labels, items)
    jev = {
        "name": "jev-full-0",
        "model": "jev",
        "status": "ok",
        "label": "extended",
        "family": "published",
        "n_features": 0,
        "brier": brier(probs, labels),
        "log_loss": log_loss(probs, labels),
        "within_brier": _within_mean(within),
        "within": within,
        "fold_brier": jev_folds,
        "fold_log_loss": jev_logs,
    }
    _attach_versus(jev, receipt_folds, evidence_folds, items)
    return {
        "n_scored": len(everyone),
        "n_mixes": len(leave_one_mix_out(items)),
        "models": fitted,
        "jev": jev,
    }


def run_models(items: list[dict]) -> dict:
    """Primary leave-one-mix-out, then the declared warp-placement secondary."""
    shot = [index for index, item in enumerate(items) if item["future_test"]]
    primary = _fit_table(items)
    secondary = _fit_table(warp_items(items))
    return {
        "n_scored": primary["n_scored"],
        "n_mixes": primary["n_mixes"],
        "shot_pool_n": len(shot),
        "models": primary["models"],
        "jev": primary["jev"],
        "warp": secondary,
        "holdout": [],
        "labels": _label_movement(items),
        "mix_probe": {
            "receipt": mix_probe(items, FEATURE_NAMES),
            "all": mix_probe(items, ALL_FEATURES),
        },
    }


def _example(kind: str, item: dict) -> dict:
    payload = serialised_state(kind, item["record"], item["features"], [])
    payload["answer"] = _answer(item["label_extended"])
    return payload


def _cell_state(items: list[dict], index: int, kind: str, shots: int, pool: list[int], rows: list[list[float]]) -> dict:
    neighbors = cosine_neighbors(index, pool, rows, shots)
    examples = [_example(kind, items[neighbor]) for neighbor in neighbors]
    return serialised_state(kind, items[index]["record"], items[index]["features"], examples)


def _cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    return cells if isinstance(cells, dict) else {}


def _save_cache(cells: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps({"model": PINNED_MODEL, "dated": PINNED_DATE, "cells": cells}, indent=2),
        encoding="utf-8",
    )


def _spent(cells: dict) -> float:
    total = 0.0
    for rows in cells.values():
        if not isinstance(rows, dict):
            continue
        for row in rows.values():
            if not isinstance(row, dict):
                continue
            cost = row.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                continue
            total += float(cost)
    return total


def _mark_infeasible(cells: dict, name: str) -> None:
    current = cells.get(name)
    if not isinstance(current, dict):
        current = {}
    current["_status"] = {"infeasible": True}
    cells[name] = current
    _save_cache(cells)


def call_cells(items: list[dict]) -> bool:
    calibration.assert_question()
    if os.environ.get("OPENROUTER_API_KEY", "").strip() == "":
        print("FAILED no-key")
        return False
    client = create_decisions_client(os.environ["OPENROUTER_API_KEY"])
    cells = _cache()
    pool = [index for index, item in enumerate(items) if not item["future_test"]]
    rows = _rows(items, FEATURE_NAMES, "features")
    unit = UNIT_PRIOR
    for kind, shots in CELLS:
        name = cell_name(kind, shots)
        done = cells.get(name) if isinstance(cells.get(name), dict) else {}
        status = done.get("_status")
        if isinstance(status, dict) and status.get("infeasible"):
            continue
        if items and all(item["id"] in done for item in items):
            costs = [
                float(row["cost"])
                for row in done.values()
                if isinstance(row, dict) and row.get("cost") is not None
            ]
            if costs:
                unit = sum(costs) / len(costs)
            continue
        pending = [index for index, item in enumerate(items) if item["id"] not in done]
        if not pending:
            continue
        if not room_for_cell(_spent(cells), max(unit, UNIT_PRIOR), len(pending)):
            print(f"CAP spent={PRIOR_SPEND_USD + _spent(cells):.4f} last={unit:.4f}")
            return False
        probe = _cell_state(items, pending[0], kind, shots, pool, rows)
        if not state_ok(probe, kind):
            _mark_infeasible(cells, name)
            continue
        last = None
        for index in pending:
            state = _cell_state(items, index, kind, shots, pool, rows)
            if not state_ok(state, kind):
                _mark_infeasible(cells, name)
                break
            spent = _spent(cells)
            if not room_for_cell(spent, last if last is not None else unit, 1):
                print(f"CAP spent={PRIOR_SPEND_USD + spent:.4f} last={unit:.4f}")
                return False
            try:
                probability, cost = _ask(client, state)
            except DecisionsError as err:
                print(f"FAILED decisions status={err.status} {err.hint}")
                return False
            if cost is None:
                print("FAILED no-cost")
                return False
            last = float(cost)
            unit = last
            cells.setdefault(name, {})
            cells[name][items[index]["id"]] = {"p_yes": probability, "cost": last}
            _save_cache(cells)
            if not room_for_cell(_spent(cells), 0.0, 0):
                print(f"CAP spent={PRIOR_SPEND_USD + _spent(cells):.4f} last={last:.4f}")
                return False
    print("DONE")
    return True


def _ask(client, state: dict) -> tuple[float, float | None]:
    result = client(
        DecisionRequest(
            state=state,
            questions={
                "phrase_clean": NoulQuestion(
                    instructions=calibration.INSTRUCTIONS,
                    true=calibration.TRUE_CRITERION,
                    false=calibration.FALSE_CRITERION,
                )
            },
        )
    )
    if result.model != PINNED_MODEL or result.dated != PINNED_DATE:
        raise DecisionsError("refused pin on the answer", "the response is not the pinned Jev")
    return float(result.answers["phrase_clean"].noul), result.cost


def _cell_brier(items: list[dict], calls: dict) -> float | None:
    if not isinstance(calls, dict) or any(item["id"] not in calls for item in items):
        return None
    if any(not isinstance(calls[item["id"]], dict) or "p_yes" not in calls[item["id"]] for item in items):
        return None
    probs = [float(calls[item["id"]]["p_yes"]) for item in items]
    labels = [item["label_extended"] for item in items]
    return brier(probs, labels)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _claim(compared: dict | None) -> str:
    if not compared or compared.get("mean") is None:
        return "not compared"
    if compared.get("resolvable"):
        return "the corrected interval clears 0.02"
    return "not resolvable"


def _complete_probs(items: list[dict], calls: object) -> list[float] | None:
    if not items or not isinstance(calls, dict):
        return None
    probs = []
    for item in items:
        row = calls.get(item["id"])
        if not isinstance(row, dict) or "p_yes" not in row:
            return None
        probs.append(float(row["p_yes"]))
    return probs


def _max_call_cost(cells: dict) -> float:
    highest = UNIT_PRIOR
    for rows in cells.values():
        if not isinstance(rows, dict):
            continue
        for key, row in rows.items():
            if key == "_status" or not isinstance(row, dict):
                continue
            cost = row.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                continue
            highest = max(highest, float(cost))
    return highest


def serialisation_table(items: list[dict], cells: dict, receipt_folds: list[float] | None) -> list[dict]:
    """Grouped scores for each completed cell. Partial cells are not a Brier."""
    labels = [item["label_extended"] for item in items]
    rows = []
    for kind, shots in CELLS:
        name = cell_name(kind, shots)
        probs = _complete_probs(items, cells.get(name))
        if probs is None:
            rows.append({"name": name, "status": "partial"})
            continue
        folds, logs, within = fixed_grouped_scores(probs, labels, items)
        row = {
            "name": name,
            "status": "ok",
            "brier": brier(probs, labels),
            "log_loss": log_loss(probs, labels),
            "within_brier": _within_mean(within),
            "within": within,
            "fold_brier": folds,
            "fold_log_loss": logs,
        }
        if receipt_folds is not None and len(folds) == len(receipt_folds):
            compared = compare_grouped(folds, receipt_folds, items)
            compared["reference"] = "logistic"
            row["versus_logistic"] = compared
        rows.append(row)
    return rows


def _receipt_folds(study: dict) -> list[float] | None:
    for result in study.get("models") or []:
        if result.get("name") == "logistic" and result.get("status") == "ok":
            folds = result.get("fold_brier")
            if isinstance(folds, list):
                return folds
    return None


def _versus_bits(label: str, versus: dict | None) -> str:
    if not versus:
        return ""
    return (
        f" Minus {label}: {_fmt(versus.get('mean'))} "
        f"({_fmt(versus.get('low'))} to {_fmt(versus.get('high'))}), {_claim(versus)}."
    )


def _base_bits(versus: dict | None) -> str:
    """Pooled gap beside the fold interval. A pooled gap past 0.02 is not a claim."""
    if not versus:
        return ""
    pooled = versus.get("pooled")
    gap = ""
    if isinstance(pooled, (int, float)) and not isinstance(pooled, bool):
        mark = "past 0.02" if abs(float(pooled)) >= RESOLVE_BRIER else "under 0.02"
        gap = f"pooled {_fmt(float(pooled))}, {mark}. "
    low = versus.get("low")
    high = versus.get("high")
    claim = _claim(versus)
    crosses = (
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and not isinstance(low, bool)
        and not isinstance(high, bool)
        and float(low) < 0.0 < float(high)
    )
    if claim == "not resolvable" and crosses:
        claim = "the interval includes zero, so it is not claimed"
    return (
        f" Minus base-rate: {gap}"
        f"Fold mean {_fmt(versus.get('mean'))} "
        f"({_fmt(low)} to {_fmt(high)}), {claim}."
    )


def _score_line(name: str, result: dict) -> str:
    shown = DISPLAY_NAMES.get(name, name)
    if result.get("status") != "ok":
        return f"{shown}: not run ({result.get('reason', 'missing')})."
    versus = result.get("versus_logistic")
    extra = _versus_bits(versus.get("reference", "logistic"), versus) if versus else ""
    extra += _base_bits(result.get("versus_base"))
    within = result.get("within") or []
    within_bits = ""
    if within:
        within_bits = " Within mix: " + ", ".join(f"{row['mix']} {_fmt(row['brier'])}" for row in within) + "."
    features = ""
    count = result.get("n_features")
    if isinstance(count, int) and count > 0:
        features = f" Features: {count}."
    return (
        f"{shown}: pooled Brier {_fmt(result.get('brier'))}, "
        f"within-mix mean {_fmt(result.get('within_brier'))}, "
        f"log loss {_fmt(result.get('log_loss'))}.{features}{extra}{within_bits}"
    )


def _probe_line(label: str, probe: dict | None) -> str:
    if not probe or probe.get("status") != "ok":
        reason = (probe or {}).get("reason", "missing")
        return f"Mix probe, {label}: not run ({reason})."
    sentence = (
        f"Mix probe, {label}: accuracy {_fmt(probe.get('accuracy'))}, "
        f"majority {_fmt(probe.get('majority'))}, "
        f"macro recall {_fmt(probe.get('macro_recall'))}."
    )
    if probe.get("identified"):
        return sentence + " The same tree names the mix. The pooled score is a mix detector."
    return sentence + " It does not clear the identification bar."


def render_result(items: list[dict], study: dict, cells: dict) -> str:
    labels = study["labels"]
    edge_sentence = (
        "That difference clears 0.02."
        if labels["edge_shift"]["resolvable"]
        else "The edge cases do not move the result by a resolvable amount."
    )
    lines = [
        f"Model `{PINNED_MODEL}`, stamp `{PINNED_DATE}`. "
        f"The question band is unchanged at 0.35–0.65.",
        "",
        f"Phrases: {len(items)}. Extended-window clean: {labels['n_extended_clean']}. "
        f"Strict-window clean: {labels['n_strict_clean']}. "
        f"Labels flip on {labels['n_flip']} phrases.",
        f"Tags: certain {labels['n_certain']}, edge {labels['n_edge']}, mid {labels['n_mid']}.",
        f"Jev full record, 0 shots, extended labels, Brier {_fmt(labels['brier_extended'])}. "
        f"The same probabilities on the strict labels: {_fmt(labels['brier_strict'])}. "
        f"Without the edge phrases: {_fmt(labels['brier_without_edge'])}.",
        "Extended minus strict: "
        f"{_fmt(labels['extended_minus_strict']['mean'])} "
        f"({_fmt(labels['extended_minus_strict']['low'])} to {_fmt(labels['extended_minus_strict']['high'])}). "
        + (
            "That difference clears 0.02."
            if labels["extended_minus_strict"]["resolvable"]
            else "That difference is not resolvable at this n."
        ),
        "Dropping the edge phrases moves Brier by "
        f"{_fmt(labels['edge_shift']['mean'])} "
        f"({_fmt(labels['edge_shift']['low'])} to {_fmt(labels['edge_shift']['high'])}). "
        + edge_sentence
        + " Dropping them is a sensitivity check. The defects sit at phrase edges, "
        "where joins and segment boundaries fall. It is not a cleaned label.",
        "",
        f"Leave-one-mix-out on {len(items)} phrases. Each mix is the test fold once, "
        "and training never sees that mix. Pooled Brier scores every phrase once. "
        "Within-mix Brier is the score on that mix. The within-mix mean is the unweighted "
        "mean of those scores, and a mix whose phrases are all one class stays in it. "
        "Brier is defined on a single class. AUC is not computed for that mean. "
        "The padded mixes carry no marks, so a within-mix AUC is undefined. "
        "A Brier gap under 0.02 is not resolvable. The Nadeau–Bengio interval has to clear zero as well. "
        "The test/train ratio is the mean of the per-fold ratios.",
        UNGROUPED_TREE,
        "",
    ]
    for result in study["models"] + [study["jev"]]:
        lines.append(_score_line(str(result.get("name", result.get("model"))), result))
    warp = study.get("warp")
    if warp:
        lines.extend(
            [
                "",
                "Secondary analysis, warp-placement mixes only. "
                f"{CUT_PLACEMENT_MIX} stays in the primary table and is left out of this one. "
                f"Leave-one-mix-out on {warp['n_scored']} phrases, {warp['n_mixes']} mixes. "
                "The same claim bar applies.",
                "",
            ]
        )
        for result in warp["models"] + [warp["jev"]]:
            lines.append(_score_line(str(result.get("name", result.get("model"))), result))
    probes = study.get("mix_probe") or {}
    if probes:
        lines.extend(["", _probe_line("receipt features", probes.get("receipt")), _probe_line("all features", probes.get("all"))])
    else:
        lines.extend(["", "Mix probe was not run."])
    lines.extend(
        [
            "",
            "The random 20 drawn with seed 20261008 stays out of the shot pool for this sweep. "
            "It is not a test. `future_test` in the phrase table marks that pool.",
            "",
            "Serialisation cells, leave-one-mix-out, extended labels. "
            "This sweep's Jev states were fixed before the evidence block, and they do not include it. "
            "A completed cell is compared with the receipt logistic.",
            "",
        ]
    )
    table = serialisation_table(items, cells, _receipt_folds(study))
    completed = [row for row in table if row["status"] == "ok"]
    if not completed:
        lines.append("No new serialisation cell was completed.")
    else:
        pooled = [labels["brier_extended"]]
        lines.append(f"full-0: pooled Brier {_fmt(labels['brier_extended'])}.")
        if study.get("jev", {}).get("status") == "ok":
            lines[-1] = _score_line("full-0", study["jev"])
        for row in completed:
            lines.append(_score_line(row["name"], row))
            pooled.append(row["brier"])
        spread = max(pooled) - min(pooled)
        lines.append(
            f"Spread {_fmt(spread)}. "
            + (
                "The spread clears 0.02."
                if spread >= RESOLVE_BRIER
                else "The spread is not resolvable at this n."
            )
        )
    if items:
        missing = [row["name"] for row in table if row["status"] != "ok"]
        if missing and not room_for_cell(_spent(cells), _max_call_cost(cells), calibration.EXPECTED_PHRASES):
            lines.append(
                f"Not sent: {', '.join(missing)}. "
                "The next cell's estimate would have passed the $0.25 cap, so the run stopped."
            )
        elif missing:
            lines.append(f"Not scored: {', '.join(missing)}.")
    spent = _spent(cells)
    if spent <= 0.0:
        spend = (
            f"No new Jev calls. The recorded calibration spend is ${PRIOR_SPEND_USD:.4f} "
            "of the $0.25 cap."
        )
    else:
        total = PRIOR_SPEND_USD + spent
        relation = "under" if total <= CAP_USD else "over"
        spend = (
            f"New Jev calls cost ${spent:.4f}. With the recorded calibration spend of "
            f"${PRIOR_SPEND_USD:.4f}, the total is ${total:.4f}, {relation} the $0.25 cap."
        )
    lines.extend(
        [
            "",
            spend,
            "full-16 is not attempted: sixteen full records do not fit the state cap.",
            "The local models read phrase-evidence.json "
            "(schema ai-jam-sessions/phrase-evidence/v1, revision 1). "
            "This sweep's Jev states do not include that block.",
            "Jev is a tested negative. Hosted Jev and local OpenJev, on these "
            "124 records, neither beat the base rate once calibrated: Brier 0.252 "
            "and 0.251 against 0.242. Within-mix AUC is 0.57 for hosted Jev and "
            "0.59 for OpenJev. The serialisation cells are the spend record. "
            "They are not a reason to continue. The evidence-family rows are "
            "the grouped comparison.",
            "The next labels are not drawn by uncertainty sampling. "
            "About 30% of each later round is random. "
            "The random 20 is not held out of a later training set.",
            "The phrase-clean decision layer stays insufficient evidence. "
            "What comes next is new labels: a blind re-mark from about 2026-10-21, "
            "and marks on more mixes once the review moves into the cockpit. Not a new model.",
        ]
    )
    return "\n".join(lines)


def _public_phrase(item: dict, cells: dict) -> dict:
    body = {
        "song": item["song"],
        "mix": item["mix"],
        "index": item["index"],
        "start": item["start"],
        "end": item["end"],
        "label_extended": item["label_extended"] == 1,
        "label_strict": item["label_strict"] == 1,
        "tag": item["tag"],
        "future_test": item["future_test"],
        "p_yes_full_0": item["p_yes_full_0"],
    }
    for kind, shots in CELLS:
        name = cell_name(kind, shots)
        calls = cells.get(name)
        if isinstance(calls, dict) and isinstance(calls.get(item["id"]), dict) and "p_yes" in calls[item["id"]]:
            body[f"p_yes_{name.replace('-', '_')}"] = calls[item["id"]]["p_yes"]
    return body


def _forbid(text: str) -> None:
    for needle in ("C:\\Users", "C:/Users", "ajs-fullsong", "director-marks", "OPENROUTER", "100.64."):
        if needle in text:
            raise RuntimeError("a report file contains a local path or a key marker")


def _public_model(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "oof"}


def _public_table(table: dict | None) -> dict | None:
    if not table:
        return None
    return {
        "n_scored": table["n_scored"],
        "n_mixes": table["n_mixes"],
        "models": [_public_model(result) for result in table["models"]],
        "jev": _public_model(table["jev"]),
    }


def write_report(items: list[dict], study: dict) -> None:
    cells = _cache()
    _stamp_pooled(study.get("models"))
    warp = study.get("warp")
    if isinstance(warp, dict):
        _stamp_pooled(warp.get("models"))
    summary = {
        "model": PINNED_MODEL,
        "dated": PINNED_DATE,
        "band": [0.35, 0.65],
        "holdout_seed": HOLDOUT_SEED,
        "holdout_n": HOLDOUT_N,
        "prior_spend_usd": PRIOR_SPEND_USD,
        "new_spend_usd": _spent(cells),
        "labels": study["labels"],
        "models": [
            {key: value for key, value in result.items() if key != "oof"}
            for result in study["models"]
        ],
        "split": "leave-one-mix-out",
        "shot_pool_n": study.get("shot_pool_n", HOLDOUT_N),
        "jev": study["jev"],
        "holdout": study.get("holdout", []),
        "mix_probe": study.get("mix_probe"),
        "warp": _public_table(study.get("warp")),
        "cells": [cell_name(kind, shots) for kind, shots in CELLS],
        "serialisation": serialisation_table(items, cells, _receipt_folds(study)),
    }
    phrases = {
        "model": PINNED_MODEL,
        "dated": PINNED_DATE,
        "phrases": [_public_phrase(item, cells) for item in items],
    }
    body = render_result(items, study, cells)
    _forbid(body)
    summary_text = json.dumps(summary, indent=2)
    phrases_text = json.dumps(phrases, indent=2)
    _forbid(summary_text)
    _forbid(phrases_text)
    document = DOC_PATH.read_text(encoding="utf-8")
    before, rest = document.split(_RESULT_START, 1)
    _old, after = rest.split(_RESULT_END, 1)
    DOC_PATH.write_text(f"{before}{_RESULT_START}\n\n{body}\n\n{_RESULT_END}{after}", encoding="utf-8")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary_text + "\n", encoding="utf-8")
    PHRASES_PATH.write_text(phrases_text + "\n", encoding="utf-8")
    print("REPORT")


def _save_study(study: dict) -> None:
    path = Path(os.environ.get("TEMP", ".")) / "sense-si-decision-study.json"
    stored = json.loads(json.dumps(study))
    path.write_text(json.dumps(stored), encoding="utf-8")


def _load_study() -> dict:
    path = Path(os.environ.get("TEMP", ".")) / "sense-si-decision-study.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    try:
        if command == "--local":
            items = prepare()
            study = run_models(items)
            _save_study(study)
            write_report(items, study)
            return 0
        if command == "--call":
            items = prepare()
            finished = call_cells(items)
            study = _load_study()
            write_report(items, study)
            return 0 if finished else 2
        if command == "--report":
            items = prepare()
            write_report(items, _load_study())
            return 0
    except DecisionsError as err:
        print(f"FAILED decisions status={err.status} {err.hint}")
        return 2
    except Exception as err:
        print(f"FAILED {type(err).__name__}")
        return 2
    print("usage: decision_learning.py --local|--call|--report")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
