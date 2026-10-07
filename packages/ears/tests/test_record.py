import json
from pathlib import Path

import pytest

from ai_ears.record import (
    EarsError,
    HearingRecord,
    from_jam_take,
    reject_gate_keys,
    review_marks,
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
    assert "cross_check" not in row
    assert "hubertfa" not in json.dumps(state["timing"])
    assert row["incomplete"] is False


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
