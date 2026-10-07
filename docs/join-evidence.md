# Join evidence

The jam session writes `phrase-evidence.json` into a reviewed mix folder.
The schema is `ai-jam-sessions/phrase-evidence/v1`, revision 1. This page is
the contract for the fields this repo maps. The numbers are not computed
here. The hearing record stores them under the instrument id `phrase-evidence`.
The instrument revision is the canonical JSON of the schema, the file
revision, the F0 instrument, the mel settings, and the parameters. The file's
directory is not copied into the record.

Each phrase carries `joins`, `switches`, `air_ms_max`, `shift_diff_ms_max`,
`shift_spread_ms`, `stretch_min`, `stretch_max`, `segment_boundary_s`, and the
maxima `spectral_jump_max`, `repeat_similarity_max`, `click_z_max`,
`f0_step_cents_max`, `pct_max`, `octave_jumps`, and `pitch_step_cents_max`.
Each join under `at_joins` carries the four measurements, the percentile of
each against the non-join controls in that take, and `octave`, `voicing_flip`,
and `switch`.

A join is the boundary between two consecutive placed cuts in one take. A
non-join control is another time in that same take, kept clear of the joins
by the file's `control_clear_s`. Word text is not a feature. A placed cut is
not itself one of these four readings.

Each feature is a measurement. When the jam session writes it, the number
names its instrument and revision. A missing reading stays missing. This repo
does not fill one in, and it does not fit a threshold on a literature figure.
Thresholds wait on this studio's own marks.

1. Spectral jump. An MFCC or a log-mel vector, taken over 5–15 ms on either
   side of the boundary, divided by that take's ordinary frame-to-frame
   distance.
2. Repeat or skip. A ridge in the log-mel cross-similarity, with the aligner's
   deletions as a second reading of the same boundary.
3. Click and noise. Energy and a high-pass first difference, as z-scores on
   the take, plus spectral flatness.
4. F0. A jump of more than one semitone, a ratio at an octave, or a flip
   between voiced and unvoiced.

Pitch for that fourth reading is not taken through a GPL Praat binding in
this package. The jam session writes the feature. This package does not
compute the audio.
