# Jam receipt adapter

Ears does not import ai-jam-sessions. `from_jam_take` reads the JSON jam already writes and copies measurements only.

The fixtures under `packages/ears/tests/fixtures/` are cuts from `phrase16w2/receipt.json`, `phrase16w2/pitch.json`, `take-01/verify-energy.json`, and `phrase-scores.json` in the amazing-grace-new-britain run, taken on 2026-10-07. Tables are the first few rows, plus pitch row `v44` so an untrackable note is in the cut. Absolute paths in `verify-energy.json` artifacts were shortened to the worktree-relative form. The listener block and the phrase measurements in `phrase-scores.json` are unchanged. No review-marks export was in that run, so marks are covered by a document that matches schema `ai-jam-sessions/review-marks/v1`.

## Timing

`receipt.json` is a placed vocal, from `vocal_clock.py verify`. `verify-energy.json` is a raw take, in `<run>/take-NN/`. Neither file is named `vocal-clock.receipt.json`.

Each table row's `err_ms` is the timing value. `method` `rise` is recorded as `rise_onset`. Any other method is kept as its own instrument id. `t_score`, `t_vowel`, `dip_db`, and `peak` are kept.

`aligner_err_ms` is the HubertFA reading, already corrected for the offset measured on that run. `checks.aligner_cross_check.offset_ms` and `measured_from` are stored with it. A raw energy file has no `aligner_err_ms`, and the adapter does not invent one. `stt_err_ms` is not an aligner reading and is not read.

The row's `cross_check` value (rescued, both_off, unconfirmed, disputed) is jam's interpretation of the two instruments. It is kept under `jam_interpretation`. It is not a measurement and it is not a gate.

`detector` holds `band_hz`, `rise_fraction`, `env_win_s`, `env_hop_s`, and `slope_min_db`. That object is the timing revision. The prose `principle` is not part of the revision.

Dropped, because they are the gate: `verdict`, row `pass`, and the `checks` entries other than the aligner offset and the count it was measured on.

## Pitch

`pitch.json` names its tracker. Top-level `tracker` is the instrument id. The checked file says `pyin`. The adapter does not hard-code that name.

`cents_median` is the note value. `voiced_fraction` and `cents_sd` are kept. `global_offset_cents` and `scatter_sd_cents` are kept on the take. `vocal_sha256` and `onsets_from` are kept as provenance.

`status` values `untrackable` and `unvoiced` are a measurement state, with the row's `reason` beside them. `PASS`, `WARN`, and `FAIL` are dropped. Also dropped: `global_pass`, `global_fail_cents`, `scatter_warn`, and `per_note`.

`cents_swift` and `tracker_disagree` are not fields on jam's main-branch receipts. If a row carries them, they are ignored.

The pitch receipt does not pin a tracker build. Those numbers are flagged incomplete. The sha and `onsets_from` do not fill that revision.

## Listener

`phrase-scores.json` holds the transcript. `takes.<take>.phrases.<i>.heard` is the text. `words`, `matched`, `intelligibility`, `notes`, and `mean_abs_cents` are kept. `pitch_fails` is a gate tally and is not read.

The top-level `listener` object (model, mmproj, prompt, temperature, seed) is the transcript's revision. The transcript stays advisory.

## Review marks

Schema `ai-jam-sessions/review-marks/v1`, from `scripts/review_marks.py`. Each mark has `t`, `cats`, `note`, and `by.name` / `by.level`. The level is kept. A mark with no level is recorded as `listener`, which is that script's own default. Marks stay an exhibit. They are not the onset reference.
