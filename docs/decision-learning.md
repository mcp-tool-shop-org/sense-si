# Decision learning

A second study of the same 124 sung phrases. The uncertain band in
`phrase_clean` stays 0.35–0.65. This page does not change the tuple, and it
does not retune the calibration constants.

The calibration already spent $0.021653 of the $0.25 cap. New Jev calls in
this study stay inside what remains. A further $0.002 is held back for one
uncached refusal from that first run. The README still says a live call waits
on a separate yes. This page does not enable the client.

The rule below was locked before any phrase in this study was scored.

## Rule

Two labels are derived from the same time marks. The strict label is not clean
when a mark time falls inside `[start, end]`. Both ends are closed. The
extended label is the calibration window `[t - 1.0, t + 0.2]`, also closed.
There is no new measured lag. The marks are whole seconds, and that one-second
lookback is the window the calibration already uses.

A phrase is `edge` when the two labels disagree. Otherwise a not-clean phrase
whose every counted mark sits more than 0.5 seconds inside both ends is `mid`.
The 0.5 second collar is half the one-second mark resolution. It is not tuned.
A phrase shorter than one second has no such interior, so an agreed not-clean
label on it is `edge`. Every other phrase, including an agreed clean phrase,
is `certain`.

The published full-record, zero-shot probabilities are scored on both labels.
Dropping the edge phrases is scored the same way. A label movement is claimed
only when the absolute Brier difference is at least 0.02 and a 1000-draw
phrase bootstrap excludes zero. The extended-minus-strict draw uses seed
`20261007 + 40001`. The edge-drop draw uses seed `20261007 + 40002`. A smaller
difference is reported, and it is not claimed.

Twenty phrases are a frozen test set. The draw is a plain shuffle with seed
`20261008`, then the first 20 indices in sorted order. It is not stratified.
Those phrases stay out of every fit, every shot, and every winner claim. They
are scored once, from one fit on the other 104. The repeated cross-validation
below is those 104 only.

The comparison is repeated stratified 5-fold, 10 repeats, the same fold
assignment as the calibration. The headline Brier is the mean squared error of
the out-of-fold probability, one number per phrase. Log loss is reported beside
it. A candidate is compared with the L2 logistic regression on the per-fold
Brier scores. The difference is candidate minus logistic. Lower is better.

The interval is the Nadeau–Bengio correction for repeated cross-validation.
With `k * r` fold differences, `s^2` is their sample variance. The corrected
variance of the mean is `(1 / (k * r) + n_test / n_train) * s^2`. The 95%
interval uses the Student t 0.975 quantile on `k * r - 1` degrees of freedom.
`n_test` is `n / 5`, rounded, and clamped so both sides are non-empty. A win
is claimed only when the absolute mean difference is at least 0.02 and that
interval excludes zero. Otherwise the numbers are reported and no winner is
named. The same rule is required before calling a model better than the
published Jev scores. Those scores are fixed. Jev is not refit on the folds.

At this n, a Brier gap under about 0.02 is not resolvable.

### Features

Twenty-two numbers, from the measurements only. No lyric, no mark, no gate
field, no rank. A missing numeric becomes 0, and the pitch and listener
families keep a missing flag where one is listed.

Timing: count, undated count, undated fraction, absolute median of the dated
offsets, dated count. Pitch: count, unvoiced count, untrackable count, null
median count, absolute median of the numeric cents, absolute tuning offset,
tuning scatter, tuning missing. Transcript: present or not, word count,
matched count, match ratio, intelligibility, intelligibility missing, note
count, mean absolute cents, and that cents value missing.

The timing, pitch, and transcript families are each fit alone with the L2
logistic regression. The tree and TabPFN see all twenty-two. Join features are
not in these receipts. No model sees them. The receipt contract is
`docs/join-evidence.md`.

### Models

The reference is L2 logistic regression, `C = 1`, `lbfgs`, at most 500
iterations, standardized on the training fold. A column with zero training
variance is left unscaled. A shallow gradient-boosted tree uses 50 trees,
depth 2, and learning rate 0.1, on the raw features, with no grid search.
TabPFN is the library default classifier, no tuning and no older checkpoint
substituted in. A batch run does not open a browser to fetch weights. If that
default cannot be imported or cannot fit, its status is not-run and the other
models still run. These three are study tools. They
are not package dependencies, and the tests do not import them.

The base-rate model predicts the training-fold prevalence. A constant
probability of 0.5 has Brier score 0.25. A one-class training fold predicts
its prevalence and does not fit.

### One rule arm

The pinned Decisions API returns a probability. It does not return rule text,
so Jev is not asked to write rules. The nine rules below are written once from
the feature definitions, before any fit, and they are not revised after a
score. A logistic regression then predicts from those nine bits. In the tables
this arm is called `featllm`.

- `any_undated`: at least one undated timing row.
- `many_undated`: at least a quarter of the timing rows are undated.
- `any_unpitched`: at least one unvoiced or untrackable pitch row.
- `any_null_pitch`: at least one null pitch median.
- `wide_pitch`: absolute pitch median at least 100 cents.
- `wide_timing`: absolute timing median at least 100 milliseconds.
- `low_match`: at least one word, and fewer than half matched.
- `wide_listener_cents`: listener mean absolute cents at least 100, else 0
  when that reading is missing.
- `listener_notes`: at least one listener note.

### Serialisation

Jev is scored again on the same phrases, with the record's shape as a
variable. Three shapes:

- `full` is the hearing record's own state. The zero-shot full state is the
  calibration, already paid, and it is not sent again.
- `pruned` is the take id, the phrase id, an empty marks list, and the
  twenty-two features.
- `rows` is the take id, the phrase id, an empty marks list, and the timing
  and pitch measurements as value, unit, and measurement state. Transcript
  counts travel with them. Transcript text does not.

Shots are 0, 4, 8, or 16 examples, chosen by cosine similarity of the
twenty-two features to the phrase being scored. The features are standardized
on the example pool only. That pool is the 104 phrases outside the frozen
test, and it excludes the phrase itself. A tie goes to the lower index. An
example's answer is the word `clean` or `not clean` from the extended label.
The question strings stay the calibration question.

The cells, in order, are pruned-0, rows-0, pruned-4, pruned-8, pruned-16,
full-4, full-8, rows-4, rows-8, rows-16. full-16 is not attempted. Sixteen
full records do not fit an 80,000 byte state. A state with a non-empty marks
list, the review-marks marker, or a text field on anything other than the full
record, is not sent. If the first pending state of a cell fails that check,
the cell is infeasible and the run moves on. A budget estimate that would pass
$0.25 stops the run. Later cells are not skipped ahead to. A cell is scored
only when every phrase has a probability. The spread is the published full-0
Brier and the completed cells, on all 124 extended labels. A spread under 0.02
is not resolvable.

### Later labels

No uncertainty sampling on this set. It is a cold start. Later rounds are
about 20 phrases, diversity-led, with about 30% of each round drawn at random.
The frozen 20 stay out of that training.

## Result

<!-- result -->

Not run.

<!-- /result -->
