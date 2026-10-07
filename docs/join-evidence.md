# Join evidence

The jam session's receipt will carry four measurements for each known join,
and one non-join control from the same take. This page is the contract for
those fields. The numbers are not computed in this repo. The phrase receipts
read by `docs/decision-learning.md` do not contain them, and no model in that
study sees a join feature.

A join is the boundary between two consecutive placed cuts in one take. A
non-join control is another time in that same take, at least 50 ms from every
join. Word text is not a feature. A placed cut is not itself one of these
four readings.

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
