import json
from pathlib import Path

import pytest

from ai_ears.record import (
    EarsError,
    HearingRecord,
    from_jam_take,
    reject_gate_keys,
    review_marks,
    transcript_from_scores,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _keys(value):
    found = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_keys(item))
    return found


def _placed(**extra) -> dict:
    return from_jam_take(
        take_id="take-01",
        phrase_id="0",
        timing=_load("receipt.json"),
        pitch=_load("pitch.json"),
        phrase_scores=_load("phrase-scores.json"),
        **extra,
    ).to_state()


def test_placed_receipt_keeps_measurements_and_drops_the_gate():
    raw_timing = _load("receipt.json")
    raw_pitch = _load("pitch.json")
    assert raw_timing["verdict"] == "FAIL"
    assert raw_timing["table"][0]["pass"] is True
    assert "cents_swift" not in raw_pitch["rows"][0]
    state = _placed()
    names = {key.casefold() for key in _keys(state)}
    assert "pass" not in names
    assert "fail" not in names
    assert "verdict" not in names
    assert "status" not in names
    assert "global_pass" not in names
    assert "scatter_warn" not in names
    assert "pitch_fails" not in names

    first = state["timing"][0]
    assert first["value"] == -9.06
    assert first["instrument"] == "rise_onset"
    assert first["incomplete"] is False
    assert first["observations"]["dip_db"] == 49.0
    assert first["observations"]["t_score"] == pytest.approx(11.666666666666666)
    assert first["observations"]["t_vowel"] == pytest.approx(11.657603688035838)
    assert "band_hz" in first["revision"]
    assert first["cross_check"]["instrument"] == "hubertfa"
    assert first["cross_check"]["value"] == -71.1
    assert first["cross_check"]["offset_ms"] == -16.8
    assert first["cross_check"]["measured_from"] == 107
    assert first["cross_check"]["incomplete"] is True
    assert first["jam_interpretation"] == "disputed"
    assert "jam_interpretation" not in state["timing"][1]
    assert state["timing"][1]["cross_check"]["value"] == 3.6

    note = state["pitch"][0]
    assert note["instrument"] == "pyin"
    assert note["value"] == -0.0 or note["value"] == 0.0
    assert note["observations"]["voiced_fraction"] == 1.0
    assert note["observations"]["cents_sd"] == 43.8
    assert note["incomplete"] is True
    assert "measurement_state" not in note
    untrackable = state["pitch"][2]
    assert untrackable["label"] == "v44"
    assert untrackable["measurement_state"] == "untrackable"
    assert untrackable["observations"]["state_reason"] == "mean/median split 47 c"
    assert state["tuning"]["global_offset_cents"] == 6.2
    assert state["tuning"]["scatter_sd_cents"] == 17.0
    assert state["tuning"]["instrument"] == "pyin"
    assert "global_fail_cents" not in state["tuning"]
    assert state["provenance"]["vocal_sha256"].startswith("2b3c5633954e")
    assert state["provenance"]["onsets_from"].endswith("phrase16w2/receipt.json")
    assert state["transcript"]["text"] == "amazing grace how sweet the sound"
    assert state["transcript"]["advisory"] is True
    assert "Qwen3-Omni" in state["transcript"]["instrument"]
    assert "mmproj" in state["transcript"]["revision"]
    assert state["transcript"]["observations"]["intelligibility"] == 1.0


def test_raw_take_has_no_aligner_reading():
    state = from_jam_take(
        take_id="take-01",
        phrase_id="0",
        timing=_load("verify-energy.json"),
    ).to_state()
    row = state["timing"][0]
    assert row["value"] == -178.63
    assert row["observations"]["dip_db"] == 26.1
    assert "measurement_state" not in row
    assert "state_reason" not in row["observations"]
    assert "cross_check" not in row
    assert "hubertfa" not in json.dumps(state["timing"])
    assert row["incomplete"] is False


def test_a_null_onset_is_undated_and_the_aligner_stays_on_the_receipt():
    raw = _load("undated-verify-energy.json")
    assert raw["table"][0]["err_ms"] is None
    assert raw["table"][0]["reason"] == "no-rise-in-window"
    state = from_jam_take(take_id="take-01", phrase_id="0", timing=raw).to_state()
    row = state["timing"][0]
    assert row["measurement_state"] == "undated"
    assert row["value"] is None
    assert row["instrument"] == "rise_onset"
    assert row["incomplete"] is False
    assert "band_hz" in row["revision"]
    assert row["observations"]["t_score"] == 82.5
    assert row["observations"]["state_reason"] == "no-rise-in-window"
    assert row["observations"]["peak"] == pytest.approx(0.14281909496166775)
    assert "t_vowel" not in row["observations"]
    assert "dip_db" not in row["observations"]
    assert "cross_check" not in row
    assert "pass" not in {key.casefold() for key in _keys(state)}

    placed = from_jam_take(
        take_id="take-01",
        phrase_id="0",
        timing=_load("undated-receipt.json"),
    ).to_state()
    kept = placed["timing"][0]
    assert kept["measurement_state"] == "undated"
    assert kept["value"] is None
    assert kept["observations"]["state_reason"] == "no-rise-in-window"
    assert kept["cross_check"]["instrument"] == "hubertfa"
    assert kept["cross_check"]["value"] == -277.4
    assert kept["cross_check"]["incomplete"] is True
    assert kept["cross_check"]["offset_ms"] == -14.3
    assert kept["cross_check"]["measured_from"] == 107
    assert kept["jam_interpretation"] == "both_off"

    blank = from_jam_take(
        take_id="t",
        phrase_id="0",
        timing=_row(err_ms=None, reason=4),
    ).to_state()["timing"][0]
    assert blank["measurement_state"] == "undated"
    assert blank["value"] is None
    assert "state_reason" not in blank.get("observations", {})


def test_tracker_name_is_read_from_the_receipt():
    pitch = _load("pitch.json")
    pitch["tracker"] = "yin"
    state = from_jam_take(take_id="take-01", phrase_id="0", timing=_load("receipt.json"), pitch=pitch).to_state()
    assert state["pitch"][0]["instrument"] == "yin"
    assert state["tuning"]["instrument"] == "yin"


def test_swift_fields_are_not_a_cross_check():
    pitch = _load("pitch.json")
    pitch["rows"][0]["cents_swift"] = -1.0
    pitch["rows"][0]["tracker_disagree"] = True
    state = from_jam_take(take_id="take-01", phrase_id="0", timing=_load("receipt.json"), pitch=pitch).to_state()
    assert "swiftf0" not in json.dumps(state)
    assert "cross_check" not in state["pitch"][0]


def test_review_mark_keeps_the_level():
    document = {
        "schema": "ai-jam-sessions/review-marks/v1",
        "entries": {
            "phrase16w2": {
                "marks": [
                    {"t": 12.5, "cats": ["pitch"], "note": "sits high", "by": {"name": "reviewer", "level": "musician"}}
                ]
            }
        },
    }
    state = _placed(marks=document)
    assert state["marks"][0]["level"] == "musician"
    assert state["marks"][0]["t"] == 12.5
    assert state["marks"][0]["role"] == "exhibit"
    assert review_marks(document)[0].cats == ("pitch",)


def test_listener_transcript_must_stay_advisory():
    from ai_ears.record import Transcript

    with pytest.raises(EarsError, match="advisory"):
        Transcript(text="la", instrument="qwen3-omni", revision="q4", advisory=False)


def test_gate_keys_are_rejected_on_a_result_shaped_object():
    with pytest.raises(EarsError, match="pass"):
        reject_gate_keys({"probabilities": {"yes": 0.9}, "pass": True})
    with pytest.raises(EarsError, match="fail"):
        reject_gate_keys({"fail": False})


def test_take_id_cannot_be_a_gate_word():
    with pytest.raises(EarsError, match="collides"):
        HearingRecord(take_id="pass", phrase_id="p")


def _row(**extra):
    body = {"id": "n0", "err_ms": 1.0}
    body.update(extra)
    return {"table": [body]}


def test_a_record_needs_both_ids():
    with pytest.raises(EarsError, match="take id and a phrase id"):
        HearingRecord(take_id="", phrase_id="p")
    with pytest.raises(EarsError, match="take id and a phrase id"):
        HearingRecord(take_id="t", phrase_id="")


def test_receipts_that_are_not_measurements_are_refused():
    with pytest.raises(EarsError, match="timing receipt must be an object"):
        from_jam_take(take_id="t", phrase_id="0", timing=[])
    with pytest.raises(EarsError, match="no table"):
        from_jam_take(take_id="t", phrase_id="0", timing={"detector": {}})
    with pytest.raises(EarsError, match="no err_ms"):
        from_jam_take(take_id="t", phrase_id="0", timing={"table": [{"id": "n0"}]})
    timing = _row()
    with pytest.raises(EarsError, match="pitch receipt must be an object"):
        from_jam_take(take_id="t", phrase_id="0", timing=timing, pitch=[])
    with pytest.raises(EarsError, match="no rows"):
        from_jam_take(take_id="t", phrase_id="0", timing=timing, pitch={"tracker": None})
    with pytest.raises(EarsError, match="no cents_median"):
        from_jam_take(take_id="t", phrase_id="0", timing=timing, pitch={"rows": [{"id": "v"}], "tracker": None})


def test_an_empty_table_and_a_blank_hash_leave_no_provenance():
    state = from_jam_take(
        take_id="t",
        phrase_id="0",
        timing={"table": None, "detector": {}},
    ).to_state()
    assert state["timing"] == []
    assert "provenance" not in state


def test_skipped_observations_and_a_non_rise_method_stay_measurements():
    state = from_jam_take(
        take_id="take-01",
        phrase_id="0",
        timing={
            "table": [
                {
                    "id": "n0",
                    "err_ms": 2.0,
                    "method": "energy",
                    "lyric": "",
                    "t_score": True,
                    "t_vowel": None,
                    "peak": "",
                    "aligner_err_ms": True,
                    "cross_check": "nope",
                }
            ],
            "detector": "not-a-config",
            "vocal_sha256": "aaa",
            "checks": {"aligner_cross_check": {"offset_ms": True, "measured_from": 1.5}},
        },
    ).to_state()
    row = state["timing"][0]
    assert row["instrument"] == "energy"
    assert row["incomplete"] is True
    assert "observations" not in row
    assert "cross_check" not in row
    assert "jam_interpretation" not in row
    assert state["provenance"]["vocal_sha256"] == "aaa"
    assert "timing_vocal_sha256" not in state["provenance"]


def test_pitch_states_and_a_different_vocal_hash_are_kept():
    state = from_jam_take(
        take_id="take-01",
        phrase_id="0",
        timing=_row(aligner_err_ms=5.0) | {"vocal_sha256": "timing-sha", "checks": {"aligner_cross_check": {}}},
        pitch={
            "tracker": " pyin ",
            "vocal_sha256": "pitch-sha",
            "onsets_from": "clock t_sec",
            "global_offset_cents": True,
            "scatter_sd_cents": "wide",
            "rows": [
                {
                    "id": "v0",
                    "cents_median": 3,
                    "status": "flat",
                    "reason": 4,
                    "lyric": None,
                    "voiced_fraction": True,
                    "cents_sd": None,
                },
                {"id": "v1", "cents_median": 0, "status": "untrackable", "reason": 4},
                {"id": "v2", "cents_median": -1, "status": 1},
            ],
        },
    ).to_state()
    assert state["pitch"][0]["instrument"] == "pyin"
    assert state["pitch"][0]["measurement_state"] == "flat"
    assert "state_reason" not in state["pitch"][0].get("observations", {})
    assert state["pitch"][1]["measurement_state"] == "untrackable"
    assert "measurement_state" not in state["pitch"][2]
    assert "global_offset_cents" not in state["tuning"]
    assert "scatter_sd_cents" not in state["tuning"]
    assert state["provenance"]["vocal_sha256"] == "pitch-sha"
    assert state["provenance"]["timing_vocal_sha256"] == "timing-sha"
    assert state["provenance"]["onsets_from"] == "clock t_sec"
    cross = state["timing"][0]["cross_check"]
    assert cross["value"] == 5.0
    assert "offset_ms" not in cross
    assert "measured_from" not in cross


def test_phrase_scores_keep_a_string_note_and_skip_empty_values():
    scores = {
        "listener": {"model": "listener"},
        "takes": {
            "take-01": {
                "phrases": {
                    "0": {
                        "heard": "la",
                        "words": None,
                        "matched": True,
                        "notes": "two",
                        "mean_abs_cents": 1.5,
                        "intelligibility": {"not": "a number"},
                    }
                }
            }
        },
    }
    state = from_jam_take(
        take_id="take-01",
        phrase_id="0",
        timing=_row(),
        phrase_scores=scores,
    ).to_state()
    observed = state["transcript"]["observations"]
    assert observed["notes"] == "two"
    assert observed["mean_abs_cents"] == 1.5
    assert "words" not in observed
    assert "matched" not in observed

    bare = {
        "listener": {"model": "listener"},
        "takes": {"take-01": {"phrases": {"3": {"heard": ""}}}},
    }
    quiet = from_jam_take(
        take_id="take-01",
        phrase_id="3",
        timing=_row(),
        phrase_scores=bare,
        phrase_index="3",
    ).to_state()
    assert quiet["transcript"]["text"] == ""
    assert "observations" not in quiet["transcript"]


def test_phrase_scores_and_marks_that_are_the_wrong_shape_are_refused():
    timing = _row()
    with pytest.raises(EarsError, match="no listener"):
        from_jam_take(take_id="take-01", phrase_id="0", timing=timing, phrase_scores={"takes": {}})
    with pytest.raises(EarsError, match="no phrase"):
        from_jam_take(
            take_id="take-01",
            phrase_id="0",
            timing=timing,
            phrase_scores={"listener": {}, "takes": {"take-01": {"phrases": {}}}},
        )
    with pytest.raises(EarsError, match="no phrase"):
        transcript_from_scores({"listener": {}, "takes": {"take-01": None}}, "take-01", "0")

    assert review_marks([{"t": 1.25, "note": "late"}])[0].level == "listener"
    assert review_marks({"entries": [{"t": 2.0, "cats": ["time"]}]})[0].t == 2.0
    marks = review_marks(
        {"entries": {"phrase": [{"t": 3.0}], "other": {"marks": []}, "skipped": "nope"}}
    )
    assert [mark.t for mark in marks] == [3.0]
    with pytest.raises(EarsError, match="unknown marks schema"):
        review_marks({"schema": "other/v1", "entries": []})
    with pytest.raises(EarsError, match="no entries"):
        review_marks({"schema": "ai-jam-sessions/review-marks/v1"})
    with pytest.raises(EarsError, match="must be a document"):
        review_marks("nope")
    with pytest.raises(EarsError, match="needs a time"):
        review_marks([{"note": "x"}])
    with pytest.raises(EarsError, match="needs a time"):
        review_marks(["nope"])
