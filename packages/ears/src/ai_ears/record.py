"""A hearing record is text and numbers. It is not a verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass

GATE_KEYS = frozenset({"pass", "fail"})
PITCH_VERDICTS = frozenset({"pass", "warn", "fail"})
PITCH_STATES = frozenset({"untrackable", "unvoiced"})
JAM_INTERPRETATIONS = frozenset({"rescued", "both_off", "unconfirmed", "disputed"})
DETECTOR_KEYS = ("band_hz", "rise_fraction", "env_win_s", "env_hop_s", "slope_min_db")
LISTENER_KEYS = ("mmproj", "model", "prompt", "seed", "temperature")
PHRASE_OBSERVATIONS = ("words", "matched", "intelligibility", "notes", "mean_abs_cents")
MARKS_SCHEMA = "ai-jam-sessions/review-marks/"


class EarsError(Exception):
    pass


def reject_gate_keys(value: object, where: str = "payload") -> None:
    """A pass or fail field is a gate. The record and the question result refuse both."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in GATE_KEYS:
                raise EarsError(f"{where} carries a {key!r} field")
            reject_gate_keys(item, where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_gate_keys(item, where)


def _revision(config: dict) -> str:
    if not config:
        return ""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def detector_revision(detector: object) -> str:
    """The timing instrument's revision is its configuration, not a git sha."""
    if not isinstance(detector, dict):
        return ""
    return _revision({key: detector[key] for key in DETECTOR_KEYS if key in detector})


@dataclass(frozen=True)
class CrossCheck:
    """A second instrument's reading. Jam's rescued/disputed token is not stored here."""

    instrument: str
    revision: str
    value: float
    incomplete: bool = False
    offset_ms: float | None = None
    measured_from: int | None = None

    def to_state(self) -> dict:
        body: dict = {
            "instrument": self.instrument,
            "revision": self.revision,
            "value": self.value,
            "incomplete": self.incomplete,
        }
        if self.offset_ms is not None:
            body["offset_ms"] = self.offset_ms
        if self.measured_from is not None:
            body["measured_from"] = self.measured_from
        return body


@dataclass(frozen=True)
class Measure:
    value: float | None
    unit: str
    instrument: str
    revision: str
    incomplete: bool = False
    label: str = ""
    cross_check: CrossCheck | None = None
    observations: tuple[tuple[str, float | str], ...] = ()
    measurement_state: str | None = None
    jam_interpretation: str | None = None

    def to_state(self) -> dict:
        body: dict = {
            "value": self.value,
            "unit": self.unit,
            "instrument": self.instrument,
            "revision": self.revision,
            "incomplete": self.incomplete,
        }
        if self.label:
            body["label"] = self.label
        if self.observations:
            body["observations"] = {key: value for key, value in self.observations}
        if self.measurement_state:
            body["measurement_state"] = self.measurement_state
        if self.jam_interpretation:
            body["jam_interpretation"] = self.jam_interpretation
        if self.cross_check is not None:
            body["cross_check"] = self.cross_check.to_state()
        return body


@dataclass(frozen=True)
class Transcript:
    text: str
    instrument: str
    revision: str
    advisory: bool = True
    observations: tuple[tuple[str, float | str | int], ...] = ()

    def __post_init__(self) -> None:
        if self.advisory is not True:
            raise EarsError("a listener transcript is advisory")

    def to_state(self) -> dict:
        body: dict = {
            "text": self.text,
            "instrument": self.instrument,
            "revision": self.revision,
            "advisory": self.advisory,
        }
        if self.observations:
            body["observations"] = {key: value for key, value in self.observations}
        return body


@dataclass(frozen=True)
class ReviewMark:
    t: float
    note: str
    level: str
    cats: tuple[str, ...] = ()
    name: str = ""

    def to_state(self) -> dict:
        return {
            "t": self.t,
            "cats": list(self.cats),
            "note": self.note,
            "level": self.level,
            "name": self.name,
            "role": "exhibit",
        }


@dataclass(frozen=True)
class TakeTuning:
    instrument: str
    revision: str
    incomplete: bool
    global_offset_cents: float | None = None
    scatter_sd_cents: float | None = None

    def to_state(self) -> dict:
        body: dict = {
            "instrument": self.instrument,
            "revision": self.revision,
            "incomplete": self.incomplete,
        }
        if self.global_offset_cents is not None:
            body["global_offset_cents"] = self.global_offset_cents
        if self.scatter_sd_cents is not None:
            body["scatter_sd_cents"] = self.scatter_sd_cents
        return body


@dataclass(frozen=True)
class Provenance:
    vocal_sha256: str = ""
    onsets_from: str = ""
    timing_vocal_sha256: str = ""

    def to_state(self) -> dict:
        body: dict = {}
        if self.vocal_sha256:
            body["vocal_sha256"] = self.vocal_sha256
        if self.onsets_from:
            body["onsets_from"] = self.onsets_from
        if self.timing_vocal_sha256 and self.timing_vocal_sha256 != self.vocal_sha256:
            body["timing_vocal_sha256"] = self.timing_vocal_sha256
        return body


EVIDENCE_SCHEMA = "ai-jam-sessions/phrase-evidence/v1"
EVIDENCE_INSTRUMENT = "phrase-evidence"


@dataclass(frozen=True)
class JoinReading:
    """One join, and its percentile against the non-join controls in the take."""

    t: float
    spectral_jump: float | None
    repeat_similarity: float | None
    click_z: float | None
    step_cents: float | None
    spectral_jump_pct: float | None
    repeat_similarity_pct: float | None
    click_z_pct: float | None
    step_cents_pct: float | None
    octave: bool
    voicing_flip: bool
    switch: bool
    air_ms: float | None
    shift_diff_ms: float | None
    stretch: float | None

    def to_state(self) -> dict:
        return {
            "t": self.t,
            "spectral_jump": self.spectral_jump,
            "repeat_similarity": self.repeat_similarity,
            "click_z": self.click_z,
            "step_cents": self.step_cents,
            "spectral_jump_pct": self.spectral_jump_pct,
            "repeat_similarity_pct": self.repeat_similarity_pct,
            "click_z_pct": self.click_z_pct,
            "step_cents_pct": self.step_cents_pct,
            "octave": self.octave,
            "voicing_flip": self.voicing_flip,
            "switch": self.switch,
            "air_ms": self.air_ms,
            "shift_diff_ms": self.shift_diff_ms,
            "stretch": self.stretch,
        }


@dataclass(frozen=True)
class PhraseEvidence:
    """Phrase-level join and segment readings. The file's directory is not stored."""

    instrument: str
    revision: str
    joins: int
    switches: int
    air_ms_max: float | None
    shift_diff_ms_max: float | None
    shift_spread_ms: float | None
    stretch_min: float | None
    stretch_max: float | None
    segment_boundary_s: float | None
    spectral_jump_max: float | None
    repeat_similarity_max: float | None
    click_z_max: float | None
    f0_step_cents_max: float | None
    pct_max: float | None
    octave_jumps: int
    pitch_step_cents_max: float | None
    at_joins: tuple[JoinReading, ...] = ()

    def to_state(self) -> dict:
        return {
            "instrument": self.instrument,
            "revision": self.revision,
            "joins": self.joins,
            "switches": self.switches,
            "air_ms_max": self.air_ms_max,
            "shift_diff_ms_max": self.shift_diff_ms_max,
            "shift_spread_ms": self.shift_spread_ms,
            "stretch_min": self.stretch_min,
            "stretch_max": self.stretch_max,
            "segment_boundary_s": self.segment_boundary_s,
            "spectral_jump_max": self.spectral_jump_max,
            "repeat_similarity_max": self.repeat_similarity_max,
            "click_z_max": self.click_z_max,
            "f0_step_cents_max": self.f0_step_cents_max,
            "pct_max": self.pct_max,
            "octave_jumps": self.octave_jumps,
            "pitch_step_cents_max": self.pitch_step_cents_max,
            "at_joins": [item.to_state() for item in self.at_joins],
        }


@dataclass(frozen=True)
class HearingRecord:
    take_id: str
    phrase_id: str
    timing: tuple[Measure, ...] = ()
    pitch: tuple[Measure, ...] = ()
    transcript: Transcript | None = None
    marks: tuple[ReviewMark, ...] = ()
    tuning: TakeTuning | None = None
    provenance: Provenance | None = None
    evidence: PhraseEvidence | None = None

    def __post_init__(self) -> None:
        if not self.take_id or not self.phrase_id:
            raise EarsError("a record needs a take id and a phrase id")
        if self.take_id.casefold() in GATE_KEYS:
            raise EarsError(f"take id {self.take_id!r} collides with a gate field")

    def to_state(self) -> dict:
        body: dict = {
            "take_id": self.take_id,
            "phrase_id": self.phrase_id,
            "timing": [item.to_state() for item in self.timing],
            "pitch": [item.to_state() for item in self.pitch],
            "marks": [item.to_state() for item in self.marks],
        }
        if self.transcript is not None:
            body["transcript"] = self.transcript.to_state()
        if self.tuning is not None:
            body["tuning"] = self.tuning.to_state()
        if self.provenance is not None and self.provenance.to_state():
            body["provenance"] = self.provenance.to_state()
        if self.evidence is not None:
            body["evidence"] = self.evidence.to_state()
        reject_gate_keys(body, "hearing record")
        return body


def _obs(row: dict, keys: tuple[str, ...]) -> tuple[tuple[str, float | str], ...]:
    found: list[tuple[str, float | str]] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            found.append((key, float(value)))
        elif isinstance(value, str) and value:
            found.append((key, value))
    return tuple(found)


def _timing_rows(receipt: dict) -> tuple[Measure, ...]:
    if "table" not in receipt:
        raise EarsError("timing receipt has no table")
    revision = detector_revision(receipt.get("detector"))
    incomplete = revision == ""
    aligner = (receipt.get("checks") or {}).get("aligner_cross_check") or {}
    offset = aligner.get("offset_ms")
    measured = aligner.get("measured_from")
    rows: list[Measure] = []
    for row in receipt.get("table") or []:
        if "err_ms" not in row:
            raise EarsError(f"clock row {row.get('id')!r} has no err_ms")
        # A missing method still means the rise detector looked and found no onset.
        method = str(row.get("method") or "rise")
        instrument = "rise_onset" if method == "rise" else method
        cross = None
        aligner_err = row.get("aligner_err_ms")
        if isinstance(aligner_err, (int, float)) and not isinstance(aligner_err, bool):
            cross = CrossCheck(
                instrument="hubertfa",
                revision="",
                value=float(aligner_err),
                incomplete=True,
                offset_ms=float(offset) if isinstance(offset, (int, float)) and not isinstance(offset, bool) else None,
                measured_from=int(measured) if isinstance(measured, int) and not isinstance(measured, bool) else None,
            )
        token = row.get("cross_check")
        interpretation = token if isinstance(token, str) and token in JAM_INTERPRETATIONS else None
        err = row["err_ms"]
        state = None
        reason = ""
        if err is None:
            state = "undated"
            raw_reason = row.get("reason")
            if isinstance(raw_reason, str) and raw_reason:
                reason = raw_reason
            value = None
        else:
            value = float(err)
        observations = list(_obs(row, ("lyric", "t_score", "t_vowel", "dip_db", "peak")))
        if reason:
            observations.append(("state_reason", reason))
        rows.append(
            Measure(
                value=value,
                unit="ms",
                instrument=instrument,
                revision=revision,
                incomplete=incomplete,
                label=str(row.get("id") or ""),
                cross_check=cross,
                observations=tuple(observations),
                measurement_state=state,
                jam_interpretation=interpretation,
            )
        )
    return tuple(rows)


def _pitch_rows(receipt: dict) -> tuple[tuple[Measure, ...], TakeTuning]:
    if "rows" not in receipt:
        raise EarsError("pitch receipt has no rows")
    tracker = receipt.get("tracker")
    instrument = tracker.strip() if isinstance(tracker, str) else ""
    rows: list[Measure] = []
    for row in receipt.get("rows") or []:
        if "cents_median" not in row:
            raise EarsError(f"pitch row {row.get('id')!r} has no cents_median")
        status = row.get("status")
        state = None
        reason = ""
        if isinstance(status, str):
            token = status.casefold()
            if token in PITCH_STATES:
                state = token
                if isinstance(row.get("reason"), str):
                    reason = row["reason"]
            elif token not in PITCH_VERDICTS:
                state = token
        observations = list(_obs(row, ("lyric", "voiced_fraction", "cents_sd")))
        if reason:
            observations.append(("state_reason", reason))
        median = row["cents_median"]
        # Unvoiced or untrackable can come back with no median. That is the state, not a zero.
        if median is None and state in PITCH_STATES:
            value = None
        elif median is None:
            raise EarsError(f"pitch row {row.get('id')!r} has no numeric cents_median")
        else:
            value = float(median)
        rows.append(
            Measure(
                value=value,
                unit="cents",
                instrument=instrument,
                revision="",
                incomplete=True,
                label=str(row.get("id") or ""),
                observations=tuple(observations),
                measurement_state=state,
            )
        )
    offset = receipt.get("global_offset_cents")
    scatter = receipt.get("scatter_sd_cents")
    # The receipt names the tracker and does not pin a build, so the revision stays empty.
    tuning = TakeTuning(
        instrument=instrument,
        revision="",
        incomplete=True,
        global_offset_cents=float(offset) if isinstance(offset, (int, float)) and not isinstance(offset, bool) else None,
        scatter_sd_cents=float(scatter) if isinstance(scatter, (int, float)) and not isinstance(scatter, bool) else None,
    )
    return tuple(rows), tuning


def _sha(receipt: dict) -> str:
    artifacts = receipt.get("artifacts") if isinstance(receipt.get("artifacts"), dict) else {}
    value = receipt.get("vocal_sha256") or artifacts.get("vocal_sha256") or ""
    return value if isinstance(value, str) else ""


def transcript_from_scores(document: dict, take_id: str, phrase_index: str = "0") -> Transcript:
    """The listener block is the transcript's instrument revision. pitch_fails is a gate tally and is not read."""
    listener = document.get("listener")
    if not isinstance(listener, dict):
        raise EarsError("phrase scores have no listener")
    try:
        phrase = document["takes"][take_id]["phrases"][str(phrase_index)]
    except (KeyError, TypeError) as err:
        raise EarsError(f"phrase scores have no phrase {phrase_index!r} on {take_id}") from err
    revision = _revision({key: listener[key] for key in LISTENER_KEYS if key in listener})
    observations: list[tuple[str, float | str | int]] = []
    for key in PHRASE_OBSERVATIONS:
        value = phrase.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            observations.append((key, value))
        elif isinstance(value, float):
            observations.append((key, value))
        elif isinstance(value, str):
            observations.append((key, value))
    return Transcript(
        text=str(phrase.get("heard") or ""),
        instrument=str(listener.get("model") or ""),
        revision=revision,
        observations=tuple(observations),
    )


def review_marks(document: dict | list) -> tuple[ReviewMark, ...]:
    """Read ai-jam-sessions/review-marks/v1. A missing level counts as a listener, which is that schema's own default."""
    if isinstance(document, list):
        raw = document
    elif isinstance(document, dict):
        schema = document.get("schema")
        if schema is not None and not str(schema).startswith(MARKS_SCHEMA):
            raise EarsError(f"unknown marks schema {schema!r}")
        raw = []
        entries = document.get("entries")
        if isinstance(entries, dict):
            for entry in entries.values():
                if isinstance(entry, dict):
                    raw.extend(entry.get("marks") or [])
                elif isinstance(entry, list):
                    raw.extend(entry)
        elif isinstance(entries, list):
            raw.extend(entries)
        else:
            raise EarsError("review marks have no entries")
    else:
        raise EarsError("review marks must be a document")
    marks: list[ReviewMark] = []
    for mark in raw:
        if not isinstance(mark, dict) or "t" not in mark:
            raise EarsError("a review mark needs a time")
        by = mark.get("by") if isinstance(mark.get("by"), dict) else {}
        level = by.get("level") or "listener"
        cats = mark.get("cats") or []
        marks.append(
            ReviewMark(
                t=float(mark["t"]),
                note=str(mark.get("note") or ""),
                level=str(level),
                cats=tuple(str(cat) for cat in cats),
                name=str(by.get("name") or ""),
            )
        )
    return tuple(marks)


def from_jam_take(
    *,
    take_id: str,
    phrase_id: str,
    timing: dict,
    pitch: dict | None = None,
    phrase_scores: dict | None = None,
    phrase_index: str = "0",
    marks: dict | list | None = None,
) -> HearingRecord:
    """Copy measurements off the receipts jam actually writes.

    ``timing`` is a ``receipt.json`` (placed vocal) or a ``verify-energy.json``
    (raw take). ``pitch`` is a ``pitch.json``. Gate fields are not read.
    ``cents_swift`` is not a field on these receipts and is ignored if present.
    """
    if not isinstance(timing, dict):
        raise EarsError("timing receipt must be an object")
    timing_rows = _timing_rows(timing)
    pitch_rows: tuple[Measure, ...] = ()
    tuning = None
    if pitch is not None:
        if not isinstance(pitch, dict):
            raise EarsError("pitch receipt must be an object")
        pitch_rows, tuning = _pitch_rows(pitch)
    transcript = None
    if phrase_scores is not None:
        transcript = transcript_from_scores(phrase_scores, take_id, phrase_index)
    parsed_marks = review_marks(marks) if marks is not None else ()
    pitch_sha = _sha(pitch or {})
    timing_sha = _sha(timing)
    onsets = ""
    if isinstance(pitch, dict) and isinstance(pitch.get("onsets_from"), str):
        onsets = pitch["onsets_from"]
    provenance = Provenance(
        vocal_sha256=pitch_sha or timing_sha,
        onsets_from=onsets,
        timing_vocal_sha256=timing_sha,
    )
    record = HearingRecord(
        take_id=take_id,
        phrase_id=phrase_id,
        timing=timing_rows,
        pitch=pitch_rows,
        transcript=transcript,
        marks=parsed_marks,
        tuning=tuning,
        provenance=provenance,
    )
    json.dumps(record.to_state())
    return record


def _evidence_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _evidence_count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EarsError(f"phrase evidence {where} is not a count")
    return value


def evidence_revision(document: dict) -> str:
    """The instrument revision is the schema, the file revision, and the parameters."""
    instruments = document.get("instruments") if isinstance(document.get("instruments"), dict) else {}
    body = {
        "schema": document.get("schema"),
        "revision": document.get("revision"),
        "f0": instruments.get("f0"),
        "mel": instruments.get("mel"),
        "params": instruments.get("params"),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def phrase_evidence(document: dict, index: int) -> PhraseEvidence:
    """Read one phrase from phrase-evidence.json. The directory and any marks are refused."""
    if not isinstance(document, dict):
        raise EarsError("phrase evidence must be an object")
    if document.get("schema") != EVIDENCE_SCHEMA:
        raise EarsError("phrase evidence schema is not the pinned v1")
    if str(document.get("revision")) != "1":
        raise EarsError("phrase evidence revision is not 1")
    if "marks" in document:
        raise EarsError("phrase evidence carries marks")
    phrases = document.get("phrases")
    if not isinstance(phrases, list):
        raise EarsError("phrase evidence has no phrases")
    found = None
    for row in phrases:
        if isinstance(row, dict) and row.get("index") == index:
            found = row
            break
    if not isinstance(found, dict):
        raise EarsError("phrase evidence has no phrase at that index")
    if "marks" in found:
        raise EarsError("phrase evidence carries marks")
    readings: list[JoinReading] = []
    raw_joins = found.get("at_joins")
    if not isinstance(raw_joins, list):
        raise EarsError("phrase evidence joins are not a list")
    for join in raw_joins:
        if not isinstance(join, dict) or _evidence_number(join.get("t")) is None:
            raise EarsError("a join reading needs a time")
        readings.append(
            JoinReading(
                t=float(join["t"]),
                spectral_jump=_evidence_number(join.get("spectral_jump")),
                repeat_similarity=_evidence_number(join.get("repeat_similarity")),
                click_z=_evidence_number(join.get("click_z")),
                step_cents=_evidence_number(join.get("step_cents")),
                spectral_jump_pct=_evidence_number(join.get("spectral_jump_pct")),
                repeat_similarity_pct=_evidence_number(join.get("repeat_similarity_pct")),
                click_z_pct=_evidence_number(join.get("click_z_pct")),
                step_cents_pct=_evidence_number(join.get("step_cents_pct")),
                octave=join.get("octave") is True,
                voicing_flip=join.get("voicing_flip") is True,
                switch=join.get("switch") is True,
                air_ms=_evidence_number(join.get("air_ms")),
                shift_diff_ms=_evidence_number(join.get("shift_diff_ms")),
                stretch=_evidence_number(join.get("stretch")),
            )
        )
    evidence = PhraseEvidence(
        instrument=EVIDENCE_INSTRUMENT,
        revision=evidence_revision(document),
        joins=_evidence_count(found.get("joins"), "joins"),
        switches=_evidence_count(found.get("switches"), "switches"),
        air_ms_max=_evidence_number(found.get("air_ms_max")),
        shift_diff_ms_max=_evidence_number(found.get("shift_diff_ms_max")),
        shift_spread_ms=_evidence_number(found.get("shift_spread_ms")),
        stretch_min=_evidence_number(found.get("stretch_min")),
        stretch_max=_evidence_number(found.get("stretch_max")),
        segment_boundary_s=_evidence_number(found.get("segment_boundary_s")),
        spectral_jump_max=_evidence_number(found.get("spectral_jump_max")),
        repeat_similarity_max=_evidence_number(found.get("repeat_similarity_max")),
        click_z_max=_evidence_number(found.get("click_z_max")),
        f0_step_cents_max=_evidence_number(found.get("f0_step_cents_max")),
        pct_max=_evidence_number(found.get("pct_max")),
        octave_jumps=_evidence_count(found.get("octave_jumps"), "octave_jumps"),
        pitch_step_cents_max=_evidence_number(found.get("pitch_step_cents_max")),
        at_joins=tuple(readings),
    )
    if evidence.joins != len(readings):
        raise EarsError("phrase evidence join count does not match the readings")
    state = evidence.to_state()
    if "dir" in state or "marks" in state:
        raise EarsError("phrase evidence kept a path or a mark")
    reject_gate_keys(state, "phrase evidence")
    return evidence
