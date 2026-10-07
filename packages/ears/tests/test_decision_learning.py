"""The decision-learning rule, with no network and no review file."""

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import decision_learning as study
from ai_ears.record import HearingRecord, JoinReading, Measure, PhraseEvidence, ReviewMark, Transcript


def _features(**overrides) -> dict[str, float]:
    row = {name: 0.0 for name in study.FEATURE_NAMES}
    row.update(overrides)
    return row


def _record(marks=()) -> HearingRecord:
    return HearingRecord(
        take_id="take-a",
        phrase_id="phrase-a",
        timing=(
            Measure(
                value=20.0,
                unit="ms",
                instrument="onset",
                revision="r",
                measurement_state="rise_onset",
                observations=(("lyric", "la"),),
            ),
        ),
        pitch=(
            Measure(
                value=None,
                unit="cents",
                instrument="pyin",
                revision="r",
                measurement_state="unvoiced",
            ),
        ),
        transcript=Transcript(
            text="advisory",
            instrument="listener",
            revision="r",
            observations=(
                ("words", 4),
                ("matched", 1),
                ("notes", 2),
                ("mean_abs_cents", 40),
            ),
        ),
        marks=marks,
    )


def test_strict_window_ignores_the_lag_and_tags_follow_the_locked_priority():
    assert study.COLLAR_S == 0.5
    assert study.strict_label(10.0, 12.0, [9.8]) == 1
    assert study.strict_label(10.0, 12.0, [10.0]) == 0
    assert study.strict_label(10.0, 12.0, [12.0]) == 0
    assert study.tag_phrase(10.0, 12.0, [9.8]) == "edge"
    assert study.tag_phrase(10.0, 12.0, [11.0]) == "mid"
    assert study.tag_phrase(10.0, 12.0, [10.2]) == "edge"
    assert study.tag_phrase(10.0, 12.0, [11.0, 10.2]) == "edge"
    assert study.tag_phrase(10.0, 10.4, [10.2]) == "edge"
    assert study.tag_phrase(10.0, 12.0, []) == "certain"
    assert study.tag_phrase(10.0, 12.0, [1.0]) == "certain"


def test_frozen_holdout_is_twenty_sorted_indices_and_leaves_the_rest():
    first = study.frozen_test_indices(124)
    assert first == study.frozen_test_indices(124)
    assert first == sorted(first)
    assert len(first) == 20
    assert len(set(first)) == 20
    train = [index for index in range(124) if index not in set(first)]
    assert len(train) == 104
    assert set(first).isdisjoint(train)
    assert first != study.frozen_test_indices(124, seed=20261009)


def test_student_t_and_nadeau_bengio_match_the_locked_small_case():
    assert abs(study.student_t_975(1) - 12.7062047364) < 1e-6
    assert abs(study.student_t_975(10) - 2.22813885199) < 1e-6
    wide = study.nadeau_bengio([0.1, -0.1], 80, 20)
    half = study.student_t_975(1) * (0.015**0.5)
    assert abs(wide["mean"]) < 1e-12
    assert abs(wide["high"] - half) < 1e-9
    assert abs(wide["low"] + half) < 1e-9
    assert wide["resolvable"] is False
    claimed = study.nadeau_bengio([0.05] * 20, 80, 20)
    assert abs(claimed["mean"] - 0.05) < 1e-12
    assert abs(claimed["low"] - 0.05) < 1e-9
    assert claimed["resolvable"] is True
    small = study.nadeau_bengio([0.01] * 20, 80, 20)
    assert small["resolvable"] is False
    assert study.nadeau_bengio([0.2], 80, 20)["resolvable"] is False


def test_cap_counts_the_recorded_calibration_spend():
    assert study.PRIOR_SPEND_USD == 0.021653
    assert study.SAFETY_USD == 0.002
    assert study.CAP_USD == 0.25
    assert abs(study.UNIT_PRIOR - (0.021653 / 124)) < 1e-15
    assert study.room_for_cell(0.0, study.UNIT_PRIOR, 124)
    assert not study.room_for_cell(0.23, study.UNIT_PRIOR, 1)
    assert ("full", 16) not in study.CELLS
    assert ("full", 0) not in study.CELLS
    assert study.CELLS[0] == ("pruned", 0)
    assert ("rows", 16) in study.CELLS


def test_pruned_and_row_states_carry_no_text_and_empty_marks():
    record = _record()
    features = study.feature_row(record)
    assert set(features) == set(study.FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features.values())
    assert "la" not in json.dumps(features)
    pruned = study.serialised_state("pruned", record, features, [])
    assert pruned["marks"] == []
    pruned_text = json.dumps(pruned)
    assert "text" not in pruned_text
    assert "lyric" not in pruned_text
    assert "la" not in pruned_text
    assert study.state_ok(pruned, "pruned")
    rows = study.serialised_state("rows", record, features, [])
    assert rows["marks"] == []
    assert "observations" not in rows["timing"][0]
    assert "text" not in json.dumps(rows)
    assert study.state_ok(rows, "rows")
    full = study.serialised_state("full", record, features, [])
    assert full["transcript"]["text"] == "advisory"
    assert study.state_ok(full, "full")
    assert not study.state_ok(full, "rows")
    marked = study.serialised_state(
        "full",
        _record(marks=(ReviewMark(t=1.0, note="", level="mark"),)),
        features,
        [],
    )
    assert not study.state_ok(marked, "full")
    needle = dict(pruned)
    needle["phrase_id"] = "review-marks"
    assert not study.state_ok(needle, "pruned")
    huge = dict(pruned)
    huge["pad"] = "x" * (study.STATE_BYTE_CAP + 1)
    assert not study.state_ok(huge, "pruned")


def test_rules_are_the_nine_thresholds_and_a_missing_cent_does_not_fire():
    quiet = _features()
    fired = study.rule_row(quiet)
    assert set(fired) == set(study.RULE_NAMES)
    assert all(value == 0.0 for value in fired.values())
    loud = _features(
        undated_n=1,
        undated_fraction=0.25,
        unvoiced_n=1,
        null_median_n=1,
        pitch_abs_median=100,
        timing_abs_median=100,
        words=4,
        match_ratio=0.49,
        mean_abs_cents=100,
        notes=1,
    )
    rules = study.rule_row(loud)
    assert all(value == 1.0 for value in rules.values())
    assert study.rule_row(_features(words=0, match_ratio=0.0))["low_match"] == 0.0
    heard = study.feature_row(_record())
    heard_rules = study.rule_row(heard)
    assert heard_rules["any_unpitched"] == 1.0
    assert heard_rules["any_null_pitch"] == 1.0
    assert heard_rules["low_match"] == 1.0
    assert heard_rules["listener_notes"] == 1.0
    assert heard_rules["wide_listener_cents"] == 0.0
    assert heard_rules["wide_timing"] == 0.0


def test_similarity_shots_skip_the_query_and_break_ties_by_index():
    rows = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    assert study.cosine_neighbors(2, [0, 1, 2, 3], rows, 3) == [3, 0, 1]
    assert study.cosine_neighbors(2, [0, 1, 2, 3], rows, 0) == []


def test_base_rate_reads_the_extended_and_strict_fields():
    items = []
    for index in range(20):
        features = _features(undated_fraction=float(index))
        items.append(
            {
                "song": "a" if index < 12 else "b",
                "mix": "one" if index < 12 else "two",
                "label_extended": 1 if index < 12 else 0,
                "label_strict": 0 if index < 12 else 1,
                "features": features,
                "rules": study.rule_row(features),
            }
        )
    train = list(range(16))
    fitted = study._fit_scope(items, train, "extended", "base-rate", study.FEATURE_NAMES, "features")
    once = study._once(items, train, [16, 17, 18, 19], "strict", "base-rate", study.FEATURE_NAMES, "features")
    assert fitted["status"] == "ok"
    assert once["status"] == "ok"
    assert fitted["brier"] > 0.0
    assert len(fitted["fold_brier"]) == 2
    assert fitted["within_brier"] is not None


def test_zero_spend_does_not_claim_a_call_and_an_edge_drop_can_be_unresolvable():
    movement = {"mean": 0.0, "low": -0.01, "high": 0.01, "resolvable": False}
    text = study.render_result(
        [],
        {
            "labels": {
                "n_extended_clean": 1,
                "n_strict_clean": 1,
                "n_flip": 0,
                "n_certain": 1,
                "n_edge": 0,
                "n_mid": 0,
                "brier_extended": 0.25,
                "brier_strict": 0.25,
                "brier_without_edge": 0.25,
                "extended_minus_strict": movement,
                "edge_shift": movement,
            },
            "models": [],
            "jev": {"name": "jev-full-0", "status": "not-run", "reason": "missing"},
            "holdout": [],
        },
        {},
    )
    assert "No new Jev calls." in text
    assert "New Jev calls cost" not in text
    assert "not move the result by a resolvable amount" in text
    assert "Unclaimed: possible mix leakage" in text
    assert "not a cleaned label" in text
    assert "Join features are not in these receipts" not in text
    assert "stay out of training" not in text
    assert "tested negative" in text
    shift = study.edge_shift([0.0, 0.0, 1.0], ["certain", "certain", "edge"], 1)
    assert abs(shift["mean"] - (1.0 / 3.0)) < 1e-12
    assert shift["resolvable"] is False
    assert study.edge_shift([1.0], ["edge"], 1)["mean"] is None


def test_leave_one_mix_out_holds_the_whole_mix_and_the_probe_bar_is_locked():
    items = [
        {"song": "a", "mix": "one"},
        {"song": "a", "mix": "two"},
        {"song": "a", "mix": "one"},
        {"song": "b", "mix": "one"},
    ]
    folds = study.leave_one_mix_out(items)
    assert [key for key, _train, _test in folds] == ["a:one", "a:two", "b:one"]
    assert folds[0][1] == [1, 3]
    assert folds[0][2] == [0, 2]
    assert study.MIX_PROBE_MARGIN == 0.20
    assert study.mix_identified(0.36, 0.16) is True
    assert study.mix_identified(0.359, 0.16) is False
    compared = study.compare_grouped(
        [0.1, 0.1],
        [0.0, 0.0],
        [
            {"song": "s", "mix": "a"},
            {"song": "s", "mix": "a"},
            {"song": "s", "mix": "b"},
        ],
    )
    assert abs(compared["ratio"] - 1.25) < 1e-12
    assert compared["resolvable"] is True
    assert "words" not in study.PLUS_MEASURE_FEATURES
    assert "spectral_jump_max" in study.PLUS_MEASURE_FEATURES
    assert "words" in study.ALL_FEATURES
    assert "gbdt-all" in study.EVIDENCE_COMPARED
    assert "gbdt" not in study.EVIDENCE_COMPARED


def _evidence() -> PhraseEvidence:
    return PhraseEvidence(
        instrument="phrase-evidence",
        revision="rev",
        joins=1,
        switches=0,
        air_ms_max=3.0,
        shift_diff_ms_max=None,
        shift_spread_ms=4.0,
        stretch_min=None,
        stretch_max=1.0,
        segment_boundary_s=None,
        spectral_jump_max=5.0,
        repeat_similarity_max=0.2,
        click_z_max=1.0,
        f0_step_cents_max=None,
        pct_max=0.9,
        octave_jumps=1,
        pitch_step_cents_max=10.0,
        at_joins=(
            JoinReading(
                t=1.0,
                spectral_jump=5.0,
                repeat_similarity=0.2,
                click_z=1.0,
                step_cents=None,
                spectral_jump_pct=1.0,
                repeat_similarity_pct=0.5,
                click_z_pct=0.1,
                step_cents_pct=None,
                octave=False,
                voicing_flip=True,
                switch=False,
                air_ms=3.0,
                shift_diff_ms=None,
                stretch=None,
            ),
        ),
    )


def test_evidence_features_flag_the_three_missing_readings():
    blank = study.evidence_feature_row(None)
    assert blank["stretch_missing"] == 1.0
    assert blank["segment_boundary_missing"] == 1.0
    assert blank["f0_step_missing"] == 1.0
    assert blank["joins"] == 0.0
    row = study.evidence_feature_row(_evidence())
    assert row["joins"] == 1.0
    assert row["voicing_flips"] == 1.0
    assert row["shift_diff_ms_max"] == 0.0
    assert row["stretch_missing"] == 1.0
    assert row["f0_step_missing"] == 1.0
    assert row["segment_boundary_missing"] == 1.0
    assert row["spectral_jump_max"] == 5.0


def test_full_state_omits_evidence_and_pruned_keeps_the_receipt_features():
    record = replace(_record(), evidence=_evidence())
    features = _features(timing_n=2.0)
    features["joins"] = 9.0
    full = study.serialised_state("full", record, features, [])
    pruned = study.serialised_state("pruned", record, features, [])
    assert "evidence" not in full
    assert full["marks"] == []
    assert set(pruned["features"]) == set(study.FEATURE_NAMES)
    assert pruned["features"]["timing_n"] == 2.0
    assert "joins" not in pruned["features"]
    assert study.state_ok(full, "full")
    assert study.state_ok(pruned, "pruned")
