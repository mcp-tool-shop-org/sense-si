# Contract

sense-si holds instruments and a question client. The instruments measure. The client returns probabilities. Nothing in this repo emits pass or fail.

## Eyes

ai-eyes answers "is this claim about the pixels true?" with SigLIP2. `image_verify` already returns a relative decision, a margin, and a confidence band. That decision stays inside eyes. Jev is not wired under it. Eyes evaluates images and nothing else, and it has no opinion about what a consumer does with the number.

The package name is `ai-eyes`. The code lives in `packages/eyes`, and the import stays `ai_eyes_mcp`. The primary console script is `ai-eyes`. `ai-eyes-mcp` stays as a second entry point until the Claude Code MCP config is switched to `ai-eyes`, and comes out in a later pull request. The payloads stay as they are.

## Ears

A hearing record is the only thing Jev is allowed to see for a sound. One take, one phrase. The fields are timing, pitch, an advisory transcript, and optional review marks.

Every number carries the instrument id and a revision. An incomplete measurement is flagged. The value is not dropped and it is not presented as a finished reading.

Review marks are an exhibit. They are not the onset reference. Hand marks were rejected for that job, and nothing here requires them.

The listener's transcript is advisory text. The question sent with a record must not tell Jev that the phrase passed or failed. The numbers are the evidence.

This is not a CLAP-style score of a caption against a waveform. The record is text and numbers from instruments that already exist. A discriminative audio model can be added later as another instrument id. The questions do not need it.

### Timing

Vowel-onset offset in milliseconds against the score clock. A row whose `err_ms` is null is the measurement state `undated`: the detector found no vowel onset, the value is null, and the row's reason stays beside `t_score`. A row whose method is `rise`, including a row with no method, is recorded as `rise_onset`. The receipt's `detector` object (band, rise fraction, envelope window and hop, slope) is that instrument's revision. `aligner_err_ms`, when the gate ran with the aligner, is the HubertFA cross-check, already corrected by the offset in `checks.aligner_cross_check`. That offset and the syllable count it was measured on travel with the cross-check. The receipt does not pin a HubertFA build, so the cross-check stays incomplete. Jam's `cross_check` token (rescued, both_off, unconfirmed, disputed) is kept as jam's interpretation, separate from the numbers. A raw `verify-energy.json` has no aligner reading, and none is invented.

`t_score`, `t_vowel`, `dip_db`, and `peak` stay on the row. They are measurements.

### Pitch

The receipt's top-level `tracker` is the instrument id. On the files this adapter was checked against, that value is `pyin`. Cents against the score are the median of the steady middle of the note. `voiced_fraction` and `cents_sd` stay on the note. `global_offset_cents` and `scatter_sd_cents` stay on the take. `untrackable` and `unvoiced` stay as a measurement state, because they say no pitch could be measured. PASS, WARN, and FAIL do not.

The pitch receipt does not pin a tracker build, so those numbers are flagged incomplete. `vocal_sha256` and `onsets_from` stay as provenance. They tie the reading to the audio and to the timing receipt. They are not a tracker revision.

### Two questions

Callers supply the record and the question. Jev returns numbers, not an explanation.

**Which take.** A choice among the take ids handed in. The result is a probability per take, and confidence when the model gave one. There is no winner field for a script to apply.

**Is this phrase clean.** A yes/no question. The caller writes what true and false mean. The result is P(yes). "Clean" is not a scale hidden in the package. P(yes) inside the uncertain band, 0.35 to 0.65 inclusive, is reported as too close to call. That band is a placeholder. It gets replaced when a calibration run on sung audio measures it, the way ai-playtest set its band from Jev's measured calibration.

A score question waits until there is an ordered scale and labels. One call does not discover a new defect.

Probabilities here are not a claim that Jev is calibrated on sung audio. A text benchmark does not transfer. Calibration waits on an audio answer key.

## Decisions

The only model string accepted is `typesafe/jev-1.13`. Every answer payload also stores the date `jev-1.13-20260917`. That date is ours, written beside the model so a later run can be compared with this one. It is not sent as a substitute model name. `jev-router`, a bare alias, or any other string fails before a request is made. A router would break replay.

The live client is built only when the caller passes a key. Importing the package does not build one, and the tests never pass a live key. The allowance to spend OpenRouter on Jev in ai-playtest does not extend to this repo.

The answer schema has probabilities, the model pin, the date, the uncertain band, and the cost when a call reported one. It has no `pass` field and no `fail` field. A payload that carries either is rejected.

A confident probability that disagrees with a jam gate is a mark for a person to read. It does not override the gate, and it does not pick the take. A consumer that thresholds Jev into a gate is outside this tool.

## What stays outside

ai-jam-sessions still owns rendering, the timing gate, the pitch gate, take picking, and the mix. si-jam-sessions has no singer and is not a dependency. SoulX, HubertFA, and the listener stay where they are. NUS-48E error figures stay private and are not copied into this repo.
