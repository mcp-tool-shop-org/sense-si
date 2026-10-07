# Contract

sense-si holds instruments and a question client. The instruments measure. The client returns probabilities. Nothing in this repo emits pass or fail.

## Eyes

ai-eyes answers "is this claim about the pixels true?" with SigLIP2. `image_verify` already returns a relative decision, a margin, and a confidence band. That decision stays inside eyes. Jev is not wired under it. Eyes evaluates images and nothing else, and it has no opinion about what a consumer does with the number.

The package name is `ai-eyes`. The code lives in `packages/eyes`, and the import stays `ai_eyes_mcp`. The console script is `ai-eyes`. The payloads stay as they are.

## Ears

A hearing record is the only thing Jev is allowed to see for a sound. One take, one phrase. The fields are timing, pitch, an advisory transcript, and optional review marks.

Every number carries the instrument id and a revision. An incomplete measurement is flagged. The value is not dropped and it is not presented as a finished reading.

Review marks are an exhibit. They are not the onset reference. Hand marks were rejected for that job, and nothing here requires them.

The listener's transcript is advisory text. The question sent with a record must not tell Jev that the phrase passed or failed. The numbers are the evidence.

This is not a CLAP-style score of a caption against a waveform. The record is text and numbers from instruments that already exist. A discriminative audio model can be added later as another instrument id. The questions do not need it.

### Joins

A join is a boundary between consecutive placed cuts of one take. The jam file `phrase-evidence.json`, schema `ai-jam-sessions/phrase-evidence/v1`, revision 1, carries the join count, the switch count, the segment readings, and the maxima of four join measurements for each phrase: a spectral jump, a repeat or skip, a click or noise burst, and an F0 break. Percentiles against the non-join controls in that take sit on each join. The fields are specified in `docs/join-evidence.md`. This repo maps that file. It does not compute the audio, and it does not invent a missing reading.

### Timing

Vowel-onset offset in milliseconds against the score clock. A row whose `err_ms` is null is the measurement state `undated`: the detector found no vowel onset, the value is null, and the row's reason stays beside `t_score`. A row whose method is `rise`, including a row with no method, is recorded as `rise_onset`. The receipt's `detector` object (band, rise fraction, envelope window and hop, slope) is that instrument's revision. `aligner_err_ms`, when the gate ran with the aligner, is the HubertFA cross-check, already corrected by the offset in `checks.aligner_cross_check`. That offset and the syllable count it was measured on travel with the cross-check. The receipt does not pin a HubertFA build, so the cross-check stays incomplete. Jam's `cross_check` token (rescued, both_off, unconfirmed, disputed) is kept as jam's interpretation, separate from the numbers. A raw `verify-energy.json` has no aligner reading, and none is invented.

`t_score`, `t_vowel`, `dip_db`, and `peak` stay on the row. They are measurements.

### Pitch

The receipt's top-level `tracker` is the instrument id. On the files this adapter was checked against, that value is `pyin`. Cents against the score are the median of the steady middle of the note. `voiced_fraction` and `cents_sd` stay on the note. `global_offset_cents` and `scatter_sd_cents` stay on the take. `untrackable` and `unvoiced` stay as a measurement state, because they say no pitch could be measured. A null median on those rows stays null. PASS, WARN, and FAIL do not.

The pitch receipt does not pin a tracker build, so those numbers are flagged incomplete. `vocal_sha256` and `onsets_from` stay as provenance. They tie the reading to the audio and to the timing receipt. They are not a tracker revision.

### Two questions

Callers supply the record and the question. Jev returns numbers, not an explanation.

**Which take.** A choice among the take ids handed in. The result is a probability per take, and confidence when the model gave one. There is no winner field for a script to apply.

**Is this phrase clean.** A yes/no question. The caller writes what true and false mean. The result is P(yes). "Clean" is not a scale hidden in the package. P(yes) from 0.35 to 0.65 inclusive is reported as too close to call. A calibration on 124 sung phrases, 73 clean and 51 not clean, did not support a narrower band. The method is selective classification with the upper confidence bound of Geifman and El-Yaniv (2017), target risk 0.10, δ = 0.001. No interior band appeared in 1000 of 1000 resamples of the inner phrases, so 0.35–0.65 stays. The scores and the rule are in `docs/calibration.md`. The product interval is closed, so the endpoints count as too close to call. The study would have abstained only strictly inside its edges.

A score question waits until there is an ordered scale and labels. One call does not discover a new defect.

Probabilities here are not a claim that Jev is calibrated. The sung-audio study's Brier score did not support reading P(yes) as a frequency. A text benchmark does not transfer.

## Decisions

The local engine is Kev-4B. The checkpoint is `jaredpalmer/kev-4b` at revision `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`. The client posts to `/v1/systemone` on this machine, sends that pin, and accepts only an echo of the same pin. No key. Kev is Apache-2.0. The base it was adapted from is `Qwen/Qwen3.5-4B-Base` at `1001bb4d826a52d1f399e183466143f4da7b741b`. That revision is provenance, not a phrase-clean threshold. The serve implementation measured with the cap is library revision `5e42a7a03f28134853dd3ff77461457e921e5ec1`. It is not vendored. The helper's default listen address is `127.0.0.1` port 8009.

Before the checkpoint allocates, the process that loads it sets a PyTorch per-process memory fraction. The default is 0.82. An argument overrides `KEV_MEMORY_FRACTION`, which overrides the default. A fraction outside (0, 1] is refused. On a CPU process the fraction is recorded and CUDA is not called. A child process does not inherit the fraction.

Hosted Jev stays the optional comparison. That client sends `typesafe/jev-1.13` and nothing else. Every Jev answer payload stores that model and the date `jev-1.13-20260917`. The date is ours, written beside the model so a later run can be compared with this one. It is not sent as a substitute model name. An answer may echo `typesafe/jev-1.13-20260917`. That echo is this same pin, and it is accepted. `jev-router`, a bare alias, or any other string is refused. A request that names one of those fails before a call is made. A router would break replay.

The live Jev client is built only when the caller passes a key. Importing the package does not build one, and the tests never pass a live key. The allowance to spend OpenRouter on Jev in ai-playtest does not extend to this repo. A live Kev call is a separate opt-in against a server that is already running the capped pin. The suite does not load the checkpoint.

OpenJev (`openjev/openjev`, CC BY-NC) stays research-only. It is not imported and it is not a dependency.

Kev's probabilities are not Jev's. The 0.35–0.65 band is the Jev question's band. A Kev result is stored with the unanswered range 0–1 and `uncertain` true. That range is the unanswered state, not a fitted threshold. A caller who passes a band into a Kev question is refused. When a Kev threshold exists, it is fitted on Kev's own leave-one-mix-out folds. The phrase-clean layer stays insufficient evidence until new labels arrive. The research seat measured this checkpoint against hosted Jev on the same 124 phrases (rnd v1.1.0.0.0, `experiments/openjev-vs-jev/results/compare-all.json`): correlation 0.67, about 0.43 seconds a phrase, and once calibrated the probabilities sat at the base rate. The engine change is cost, speed, and the Apache-2.0 licence.

The answer schema has probabilities, the model pin, the date, the uncertain band, and the cost when a call reported one. It has no `pass` field and no `fail` field. A payload that carries either is rejected.

A confident probability that disagrees with a jam gate is a mark for a person to read. It does not override the gate, and it does not pick the take. A consumer that thresholds a probability into a gate is outside this tool.

## What stays outside

ai-jam-sessions still owns rendering, the timing gate, the pitch gate, take picking, and the mix. si-jam-sessions has no singer and is not a dependency. SoulX, HubertFA, and the listener stay where they are. NUS-48E error figures stay private and are not copied into this repo.
