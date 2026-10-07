"""The calibration rule, with no network and no review file."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import phrase_calibration as rule


def test_mark_window_is_closed_at_both_ends():
    assert rule.mark_overlaps(10.0, 12.0, 11.0)
    assert rule.mark_overlaps(10.0, 12.0, 9.8)
    assert rule.mark_overlaps(10.0, 12.0, 13.0)
    assert not rule.mark_overlaps(10.0, 12.0, 9.799)
    assert not rule.mark_overlaps(10.0, 12.0, 13.001)
    assert rule.phrase_label(10.0, 12.0, [9.799, 13.001]) == 1
    assert rule.phrase_label(10.0, 12.0, [9.8]) == 0


def test_clopper_matches_the_zero_error_form_and_a_two_trial_case():
    delta = 0.05
    assert rule.clopper_upper(0, 100, delta) == 1.0 - delta ** (1.0 / 100)
    assert rule.clopper_upper(2, 2, 0.25) == 1.0
    assert abs(rule.clopper_upper(1, 2, 0.25) - math.sqrt(0.75)) < 1e-6
    assert abs(rule.binom_cdf(0, 10, 0.2) - 0.8**10) < 1e-12


def test_zero_errors_need_a_large_sample_at_the_locked_confidence():
    assert rule.clopper_upper(0, 100, 0.001) < 0.10
    assert rule.clopper_upper(0, 50, 0.001) > 0.10


def test_temperature_sharpens_an_underconfident_probability_and_leaves_a_coin_flip():
    sharp = rule.fit_temperature([0.6] * 30 + [0.4] * 30, [1] * 30 + [0] * 30)
    assert sharp < 1.0
    coin = rule.fit_temperature([0.5] * 20, [1] * 10 + [0] * 10)
    assert coin == 1.0


def test_full_coverage_is_not_an_interior_band_and_a_small_sample_certifies_nothing():
    perfect = rule.selective_bound([0.99] * 50 + [0.01] * 50, [1] * 50 + [0] * 50, rule.DELTA)
    assert perfect is not None and perfect["interior"] is False
    small = rule.selective_bound([0.99] * 20 + [0.01] * 20, [1] * 20 + [0] * 20, rule.DELTA)
    assert small is None


def test_errors_at_low_confidence_open_an_interior_band():
    probs = [0.99] * 50 + [0.01] * 50 + [0.6] * 20
    labels = [1] * 50 + [0] * 50 + [0] * 10 + [1] * 10
    band = rule.selective_bound(probs, labels, rule.DELTA)
    assert band is not None and band["interior"] is True
    assert band["low"] < 0.1 and band["high"] > 0.9


def test_pav_pools_a_reversal_and_keeps_an_increase():
    pooled = rule.pav([0.2, 0.8], [1, 0])
    assert pooled == [0.5, 0.5]
    rising = rule.pav([0.2, 0.8], [0, 1])
    assert rising == [0.0, 1.0]


def test_outer_split_is_stratified_and_leaves_the_inner_set():
    labels = [1] * 40 + [0] * 20
    first = rule.assign_outer(labels)
    second = rule.assign_outer(labels)
    assert first == second
    assert len(first) == int(round(0.30 * 40)) + int(round(0.30 * 20))
    assert first.isdisjoint(set(range(len(labels))) - first)
    assert any(labels[i] == 1 for i in first) and any(labels[i] == 0 for i in first)
    assert any(labels[i] == 1 for i in range(len(labels)) if i not in first)


def test_slice_keeps_a_boundary_row_and_an_overlapping_note():
    timing = {"table": [{"t_score": 1.0}, {"t_score": 2.0}, {"t_score": 2.01}, {"t_score": None}]}
    pitch = {"rows": [{"window": [0.5, 1.0]}, {"window": [3.0, 4.0]}, {"window": [2.0, 1.0]}]}
    sliced_timing, sliced_pitch = rule.slice_phrase(timing, pitch, 1.0, 2.0)
    assert [row["t_score"] for row in sliced_timing["table"]] == [1.0, 2.0]
    assert sliced_pitch["rows"] == [{"window": [0.5, 1.0]}]


def test_a_built_phrase_carries_no_marks():
    timing = {
        "table": [{"id": "a", "err_ms": -3.0, "t_score": 1.2, "method": "rise"}],
        "detector": {"band_hz": [80, 800]},
    }
    pitch = {
        "rows": [{"id": "a", "cents_median": 4.0, "window": [1.0, 1.5]}],
        "tracker": "pyin",
        "global_offset_cents": 1.0,
        "scatter_sd_cents": 2.0,
    }
    scores = {
        "listener": {"model": "fixture"},
        "takes": {"take-01": {"phrases": {"0": {"heard": "la", "words": 1}}}},
    }
    phrase = {"take": "take-01", "index": 0, "start": 1.0, "end": 2.0}
    record, state, start, end, take = rule.build_phrase(
        "song", "mix", phrase, timing, pitch, scores
    )
    assert record.marks == ()
    assert state["marks"] == []
    assert len(state["timing"]) == 1
    assert len(state["pitch"]) == 1
    assert start == 1.0 and end == 2.0 and take == "take-01"
    assert "verdict" not in state


def test_question_text_has_no_gate_word():
    rule.assert_question()


def test_decide_adopts_only_when_every_gate_passes():
    adopted = rule.decide(
        {
            "n_clean": 40,
            "n_not_clean": 40,
            "n_bootstrap": 100,
            "n_no_interior": 0,
            "edge_low": [0.20] * 100,
            "edge_high": [0.80] * 100,
            "outer_answered": 40,
            "outer_accuracy": 1.0,
            "outer_accuracy_lower": 0.93,
        }
    )
    assert adopted["adopted"] is True
    assert adopted["low"] == 0.20 and adopted["high"] == 0.80
    swinging = rule.decide(
        {
            "n_clean": 40,
            "n_not_clean": 40,
            "n_bootstrap": 100,
            "n_no_interior": 0,
            "edge_low": [0.05, 0.40],
            "edge_high": [0.60, 0.95],
            "outer_answered": 40,
            "outer_accuracy": 1.0,
            "outer_accuracy_lower": 0.93,
        }
    )
    assert swinging["adopted"] is False
    assert "edge-width" in swinging["reasons"]
    short = rule.decide(
        {
            "n_clean": 29,
            "n_not_clean": 40,
            "n_bootstrap": 100,
            "n_no_interior": 0,
            "edge_low": [0.20] * 100,
            "edge_high": [0.80] * 100,
            "outer_answered": 40,
            "outer_accuracy": 1.0,
            "outer_accuracy_lower": 0.93,
        }
    )
    assert short["adopted"] is False
    assert "class-count" in short["reasons"]
    assert short["keep"] == [0.35, 0.65]


def test_study_on_a_perfect_sample_does_not_invent_an_interior_band():
    probs = [0.99] * 40 + [0.01] * 40
    labels = [1] * 40 + [0] * 40
    stats = rule.run_study(probs, labels, bootstrap=20)
    assert stats["n_clean"] == 40 and stats["n_not_clean"] == 40
    assert stats["n_inner"] + stats["n_outer"] == 80
    assert stats["decision"]["adopted"] is False
    assert "no-interior-fraction" in stats["decision"]["reasons"]
    assert stats["brier_raw"] < 0.01
