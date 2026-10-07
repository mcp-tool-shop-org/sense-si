"""Follow-up to the phrase calibration. The band in the question stays 0.35–0.65.

Locked before this file scored a phrase. Do not retune the constants after
seeing a Brier score, a fold, or a Jev probability.

The pinned Decisions API returns a probability. It does not return rule text,
so the FeatLLM arm is a logistic regression on rules written once from the
feature definitions. Jev is not asked to invent them.

New Jev calls stay inside the original $0.25 cap, counting the calibration
spend that is already recorded.
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
    """A fixed random holdout for later rounds. Not a stratified split."""
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


def nadeau_bengio(differences: list[float], n_train: int, n_test: int) -> dict:
    """Corrected interval for a repeated-CV difference. Claim only past 0.02."""
    count = len(differences)
    mean = sum(differences) / count
    if count < 2 or n_train < 1 or n_test < 1:
        return {"mean": mean, "low": None, "high": None, "resolvable": False}
    variance = sum((item - mean) ** 2 for item in differences) / (count - 1)
    corrected = (1.0 / count + n_test / n_train) * variance
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
        body = record.to_state()
    elif kind == "pruned":
        body = {
            "take_id": record.take_id,
            "phrase_id": record.phrase_id,
            "marks": [],
            "features": features,
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


def prepare() -> list[dict]:
    calibration.assert_question()
    items = calibration.load_items()
    times = calibration._mark_times()
    published = json.loads(calibration.PHRASES_PATH.read_text(encoding="utf-8"))
    saved = {
        f"{row['song']}/{row['mix']}/{row['index']}": float(row["p_yes"])
        for row in published["phrases"]
    }
    holdout = set(frozen_test_indices(len(items)))
    prepared = []
    for index, item in enumerate(items):
        key = (item["song"], item["mix"])
        features = feature_row(item["record"])
        prepared.append(
            {
                "id": item["id"],
                "song": item["song"],
                "mix": item["mix"],
                "index": item["index"],
                "start": item["start"],
                "end": item["end"],
                "record": item["record"],
                "label_extended": item["label"],
                "label_strict": strict_label(item["start"], item["end"], times[key]),
                "tag": tag_phrase(item["start"], item["end"], times[key]),
                "future_test": index in holdout,
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
        buckets, fold_brier, fold_log = repeated_predictions(
            rows, labels, indices, "logistic" if model == "featllm" else model
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


def run_models(items: list[dict]) -> dict:
    train = [index for index, item in enumerate(items) if not item["future_test"]]
    holdout = [index for index, item in enumerate(items) if item["future_test"]]
    specs = (
        ("base-rate", "extended", FEATURE_NAMES, "features"),
        ("logistic", "extended", FEATURE_NAMES, "features"),
        ("logistic-timing", "extended", TIMING_FEATURES, "features"),
        ("logistic-pitch", "extended", PITCH_FEATURES, "features"),
        ("logistic-transcript", "extended", TRANSCRIPT_FEATURES, "features"),
        ("gbdt", "extended", FEATURE_NAMES, "features"),
        ("tabpfn", "extended", FEATURE_NAMES, "features"),
        ("featllm", "extended", RULE_NAMES, "rules"),
        ("logistic", "strict", FEATURE_NAMES, "features"),
    )
    fitted = []
    logistic_folds = None
    for model, label_key, names, source in specs:
        name = model if label_key == "extended" else f"{model}-strict"
        result = _fit_scope(items, train, label_key, model.split("-")[0] if model.startswith("logistic") else model, names, source)
        result["name"] = name
        result["label"] = label_key
        result["family"] = {
            "logistic-timing": "timing",
            "logistic-pitch": "pitch",
            "logistic-transcript": "transcript",
        }.get(model, "all" if model != "featllm" else "rules")
        if name == "logistic" and result.get("status") == "ok":
            logistic_folds = result["fold_brier"]
        fitted.append(result)
    for result in fitted:
        if result.get("status") == "ok" and logistic_folds is not None and result["name"] != "logistic":
            if result["label"] == "extended" and len(result["fold_brier"]) == len(logistic_folds):
                result["versus_logistic"] = compare_folds(result["fold_brier"], logistic_folds, len(train))
            else:
                result["versus_logistic"] = None
        else:
            result["versus_logistic"] = None
    labels = [item["label_extended"] for item in items]
    probs = [item["p_yes_full_0"] for item in items]
    jev_folds, jev_logs = fixed_fold_scores(probs, labels, train)
    jev = {
        "name": "jev-full-0",
        "status": "ok",
        "brier": brier([probs[index] for index in train], [labels[index] for index in train]),
        "log_loss": log_loss([probs[index] for index in train], [labels[index] for index in train]),
        "fold_brier": jev_folds,
        "fold_log_loss": jev_logs,
        "versus_logistic": None
        if logistic_folds is None or len(jev_folds) != len(logistic_folds)
        else compare_folds(jev_folds, logistic_folds, len(train)),
    }
    holdout_scores = [
        _once(items, train, holdout, "extended", model, names, source)
        for model, names, source in (
            ("base-rate", FEATURE_NAMES, "features"),
            ("logistic", FEATURE_NAMES, "features"),
            ("gbdt", FEATURE_NAMES, "features"),
            ("tabpfn", FEATURE_NAMES, "features"),
            ("featllm", RULE_NAMES, "rules"),
        )
    ]
    return {
        "n_train": len(train),
        "n_holdout": len(holdout),
        "models": fitted,
        "jev": jev,
        "holdout": holdout_scores,
        "labels": _label_movement(items),
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


def render_result(items: list[dict], study: dict, cells: dict) -> str:
    labels = study["labels"]
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
        + (
            "That difference clears 0.02."
            if labels["edge_shift"]["resolvable"]
            else "The edge cases do not move the result by a resolvable amount."
        ),
        "",
        "Repeated stratified 5-fold, 10 repeats, on the phrases that are not in the frozen 20. "
        "A Brier gap under 0.02 is not resolvable. The Nadeau–Bengio interval has to clear zero as well.",
        "",
    ]
    for result in study["models"] + [study["jev"]]:
        if result.get("status") != "ok":
            lines.append(f"{result.get('name', result.get('model'))}: not run ({result.get('reason', 'missing')}).")
            continue
        versus = result.get("versus_logistic")
        extra = ""
        if versus:
            extra = (
                f" Minus logistic: {_fmt(versus['mean'])} "
                f"({_fmt(versus['low'])} to {_fmt(versus['high'])}), {_claim(versus)}."
            )
        lines.append(
            f"{result.get('name', result.get('model'))}: Brier {_fmt(result['brier'])}, "
            f"log loss {_fmt(result['log_loss'])}.{extra}"
        )
    lines.extend(["", "The frozen 20 are scored once. They are not used to claim a winner.", ""])
    for result in study["holdout"]:
        if result.get("status") != "ok":
            lines.append(f"Holdout {result['model']}: not run ({result.get('reason', 'missing')}).")
        else:
            lines.append(
                f"Holdout {result['model']}: Brier {_fmt(result['brier'])}, "
                f"log loss {_fmt(result['log_loss'])}, n={result['n']}."
            )
    scored = [("full-0", labels["brier_extended"])]
    for kind, shots in CELLS:
        name = cell_name(kind, shots)
        value = _cell_brier(items, cells.get(name))
        if value is not None:
            scored.append((name, value))
    lines.extend(["", "Serialisation cells, extended labels, all 124 phrases.", ""])
    if len(scored) == 1:
        lines.append("No new serialisation cell was completed.")
    else:
        values = [value for _name, value in scored]
        spread = max(values) - min(values)
        for name, value in scored:
            lines.append(f"{name}: Brier {_fmt(value)}.")
        lines.append(
            f"Spread {_fmt(spread)}. "
            + (
                "The spread clears 0.02."
                if spread >= RESOLVE_BRIER
                else "The spread is not resolvable at this n."
            )
        )
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
            "Join features are not in these receipts, so no model sees them.",
            "The next labels are not drawn by uncertainty sampling. "
            "About 30% of each later round is random, and the frozen 20 stay out of training.",
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


def write_report(items: list[dict], study: dict) -> None:
    cells = _cache()
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
        "jev": study["jev"],
        "holdout": study["holdout"],
        "cells": [cell_name(kind, shots) for kind, shots in CELLS],
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
