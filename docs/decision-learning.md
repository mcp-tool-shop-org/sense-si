# Decision learning

A second study of the same 124 sung phrases. The uncertain band in
`phrase_clean` stays 0.35–0.65. This page does not change the tuple, and it
does not retune the calibration constants.

The calibration already spent $0.021653 of the $0.25 cap. New Jev calls in
this study stay inside what remains. A further $0.002 is held back for one
uncached refusal from that first run. The README still says a live call waits
on a separate yes. This page does not enable the client.

The label rule, the claim bar, and the model constants below were locked
before any phrase in this study was scored. The split was replaced before the
grouped rerun. Ungrouped repeated cross-validation on the 104 is not a claim.
The shallow tree minus logistic on that split was -0.0719 (-0.1298 to -0.0141).
Unclaimed: possible mix leakage.

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
Dropping the edge phrases is a sensitivity check, not a cleaned label. The
defects sit at phrase edges, where joins and segment boundaries fall. The
shift is still reported. A label movement is claimed only when the absolute
Brier difference is at least 0.02 and a 1000-draw phrase bootstrap excludes
zero. The extended-minus-strict draw uses seed `20261007 + 40001`. The
edge-drop draw uses seed `20261007 + 40002`. A smaller difference is reported,
and it is not claimed.

The winner split is leave-one-mix-out on all 124 phrases. Each mix is the test
fold once. Training never sees that mix. The test mix is not chosen after
looking at which mix has the most marks. Pooled Brier is the mean squared
error of those held-out probabilities, one number per phrase. Within-mix
Brier is the score on that mix. The within-mix mean is the unweighted mean
of those Brier scores, and a mix whose phrases are all one class stays in it.
Brier is defined on a single class. AUC is not computed for that mean. Both
padded mixes carry no marks, so a within-mix AUC is undefined there. Log loss
is reported beside the pooled score.

A random 20, drawn with seed `20261008` as a plain shuffle and then the first
20 indices in sorted order, is the shot pool for this serialisation sweep. It
is not a test. Phrases from mixes the model has already seen are not a held-out
set. `future_test` in the phrase table marks that pool.

The comparison uses the seven leave-one-mix-out fold scores. The difference is
candidate minus the reference logistic. Lower is better. Rows that use only
the twenty-two receipt features are compared with L2 logistic regression on
those twenty-two. `logistic-all` is those twenty-two plus the evidence family,
and it is compared with the receipt logistic. The cumulative evidence rows,
and the tree and TabPFN that see every feature, are compared with `logistic-all`.

The shallow tree is two rows. One uses the twenty-two receipt features. That
is the tree behind the ungrouped difference. The other uses all 41 features.
The 41-feature tree is also compared with the base rate. The difference is
tree minus base rate. The same 0.02 bar applies. A pooled gap past 0.02 is
not a claim unless the corrected interval excludes zero.

### Warp-placement secondary

Declared before those scores were computed. The primary table stays
leave-one-mix-out on all seven mixes. `amazing-grace-new-britain:phrase16`
is the only cut-placement mix, and it stays in the primary table.

The secondary analysis removes that mix from training and from test.
Leave-one-mix-out then runs on the six warp-placement mixes, 108 phrases.
Three of those mixes have 16 phrases, so a fold's ratio is 16/92. Three have
20 phrases, so a fold's ratio is 20/88. The corrected interval uses those six
differences and 5 degrees of freedom. The models and the claim bar are the
same. The tree's fold seed is `20261007 + 70001` plus the fold index in
first-seen order among the six.

On the primary seven mixes, the interval is the Nadeau–Bengio correction.
With one difference per mix, `s^2` is their sample variance on 6 degrees of
freedom. The corrected variance of the mean is `(1 / 7 + r) * s^2`, where
`r` is the mean of the per-fold `n_test / n_train` ratios. Four mixes have 16
phrases, so that fold's ratio is 16/108. Three mixes have 20 phrases, so that
fold's ratio is 20/104. The 95%
interval uses the Student t 0.975 quantile. A win is claimed only when the
absolute mean difference is at least 0.02 and that interval excludes zero.
Otherwise the numbers are reported and no winner is named. The same rule is
required before calling a model better than the published Jev scores. Those
scores are fixed. Jev is not refit on the folds.

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
logistic regression. The tree and TabPFN are fit twice: once on the twenty-two,
and once on every feature below.

Join and segment numbers come from `phrase-evidence.json`, schema
`ai-jam-sessions/phrase-evidence/v1`, revision 1. The instrument id is
`phrase-evidence`. Its revision is the canonical JSON of the schema, the file
revision, the F0 instrument, the mel settings, and the parameters. This repo
maps that file. It does not compute the audio. The contract is
`docs/join-evidence.md`.

The evidence features, locked before the grouped rerun, are:

- Joins: `joins`, `switches`.
- Segment: `air_ms_max`, `shift_diff_ms_max`, `shift_spread_ms`, `stretch_min`,
  `stretch_max`, `stretch_missing`, `segment_boundary_s`,
  `segment_boundary_missing`.
- Join measurements: `spectral_jump_max`, `repeat_similarity_max`, `click_z_max`,
  `f0_step_cents_max`, `f0_step_missing`, `pct_max`, `octave_jumps`,
  `pitch_step_cents_max`, `voicing_flips`.

`voicing_flips` counts joins whose `voicing_flip` is true. A missing stretch
pair, segment boundary, or F0 step is 0 plus its missing flag. Any other
missing evidence number is 0 with no flag. The file's directory is not a feature.

The ablation is cumulative and it does not include the transcript family.
The rows are timing and pitch, then those plus joins, then plus the segment
readings, then plus the join measurements. The nine-rule arm does not gain
evidence rules after a score.

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

The tree's fold seed is `20261007 + 70001` plus the fold index. A leakage
probe fits the same tree to name the mix from the features. It is repeated
stratified 5-fold, 10 repeats, stratified by mix, so every mix is in train
and in test. That is the question the probe asks. Its seed is
`20261007 + 60000`. It is run on the twenty-two receipt features and on the
full set, separately. It identifies the mix when its accuracy is at least the
majority rate plus 0.20. The largest mix is 20 of 124. If it identifies the
mix, the pooled score is a mix detector, and the result says so. The probe is
not itself a winner claim.

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

- `full` is the hearing record's own state, without the evidence block. This
  sweep's full state was fixed before that block existed. The zero-shot full
  state is the calibration, already paid, and it is not sent again.
- `pruned` is the take id, the phrase id, an empty marks list, and the
  twenty-two receipt features. Evidence keys are not added to a resumed call.
- `rows` is the take id, the phrase id, an empty marks list, and the timing
  and pitch measurements as value, unit, and measurement state. Transcript
  counts travel with them. Transcript text does not.

Shots are 0, 4, 8, or 16 examples, chosen by cosine similarity of the
twenty-two receipt features to the phrase being scored. The features are
standardized on the example pool only. That pool is the 104 phrases outside
the random 20, and it excludes the phrase itself. The pool is not a test, and
the examples are not redrawn in the middle of a cell. A tie goes to the lower
index. An example's answer is the word `clean` or `not clean` from the
extended label. The question strings stay the calibration question.

The cells, in order, are pruned-0, rows-0, pruned-4, pruned-8, pruned-16,
full-4, full-8, rows-4, rows-8, rows-16. full-16 is not attempted. Sixteen
full records do not fit an 80,000 byte state. A state with a non-empty marks
list, the review-marks marker, or a text field on anything other than the full
record, is not sent. If the first pending state of a cell fails that check,
the cell is infeasible and the run moves on. A budget estimate that would pass
$0.25 stops the run. Later cells are not skipped ahead to. A cell is scored
only when every phrase has a probability. Each completed cell is scored by
leave-one-mix-out, pooled and within mix, and compared with the receipt
logistic. The spread is the published full-0 Brier and the completed cells'
pooled Briers. A spread under 0.02 is not resolvable. A cell that was not
sent is not given a Brier.

Hosted Jev and a local OpenJev were compared on these 124 records in the
research seat's entry `2026-10-07-openjev-and-open-jev-alternatives`. Once
calibrated, neither beats the base rate: Brier 0.252 and 0.251 against 0.242.
Within-mix AUC is 0.57 for hosted Jev and 0.59 for OpenJev. Jev is a tested
negative. The serialisation cells are the spend record. They are not a reason to
continue. The evidence families under leave-one-mix-out are the comparison
this page adds.

### Later labels

No uncertainty sampling on this set. It is a cold start. Later rounds are
about 20 phrases, diversity-led, with about 30% of each round drawn at random.
The random 20 is not held out of that training. It only stayed out of this
sweep's shot pool. The phrase-clean decision layer stays insufficient
evidence. What comes next is new labels: a blind re-mark from about
2026-10-21, and marks on more mixes once the review moves into the cockpit.
Not a new model.

## Result

<!-- result -->

Model `typesafe/jev-1.13`, stamp `jev-1.13-20260917`. The question band is unchanged at 0.35–0.65.

Phrases: 124. Extended-window clean: 73. Strict-window clean: 87. Labels flip on 14 phrases.
Tags: certain 73, edge 45, mid 6.
Jev full record, 0 shots, extended labels, Brier 0.3341. The same probabilities on the strict labels: 0.3863. Without the edge phrases: 0.4801.
Extended minus strict: -0.0523 (-0.0798 to -0.0272). That difference clears 0.02.
Dropping the edge phrases moves Brier by -0.1460 (-0.1811 to -0.1113). That difference clears 0.02. Dropping them is a sensitivity check. The defects sit at phrase edges, where joins and segment boundaries fall. It is not a cleaned label.

Leave-one-mix-out on 124 phrases. Each mix is the test fold once, and training never sees that mix. Pooled Brier scores every phrase once. Within-mix Brier is the score on that mix. The within-mix mean is the unweighted mean of those scores, and a mix whose phrases are all one class stays in it. Brier is defined on a single class. AUC is not computed for that mean. The padded mixes carry no marks, so a within-mix AUC is undefined. A Brier gap under 0.02 is not resolvable. The Nadeau–Bengio interval has to clear zero as well. The test/train ratio is the mean of the per-fold ratios.
Ungrouped repeated cross-validation on the 104, shallow tree minus logistic, was -0.0719 (-0.1298 to -0.0141). Unclaimed: possible mix leakage.

base-rate: pooled Brier 0.2777, within-mix mean 0.2785, log loss 0.7513. Minus logistic: -0.2291 (-0.5080 to 0.0499), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.4236, amazing-grace-new-britain:phrase16w 0.2470, amazing-grace-new-britain:phrase16w2 0.2470, amazing-grace-new-britain:pad16 0.2230, america-the-beautiful-materna:phrase16w 0.3285, america-the-beautiful-materna:phrase16w2 0.2402, america-the-beautiful-materna:pad16 0.2405.
logistic: pooled Brier 0.5019, within-mix mean 0.5076, log loss 2.7188. Features: 22. Within mix: amazing-grace-new-britain:phrase16 0.9375, amazing-grace-new-britain:phrase16w 0.3264, amazing-grace-new-britain:phrase16w2 0.3721, amazing-grace-new-britain:pad16 0.5722, america-the-beautiful-materna:phrase16w 0.3780, america-the-beautiful-materna:phrase16w2 0.2710, america-the-beautiful-materna:pad16 0.6960.
logistic-all: pooled Brier 0.3109, within-mix mean 0.3224, log loss 1.8295. Features: 41. Minus logistic: -0.1852 (-0.4015 to 0.0310), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.9374, amazing-grace-new-britain:phrase16w 0.1978, amazing-grace-new-britain:phrase16w2 0.3003, amazing-grace-new-britain:pad16 0.2085, america-the-beautiful-materna:phrase16w 0.1700, america-the-beautiful-materna:phrase16w2 0.1771, america-the-beautiful-materna:pad16 0.2655.
logistic-timing: pooled Brier 0.3359, within-mix mean 0.3419, log loss 0.9294. Features: 5. Minus logistic: -0.1657 (-0.4264 to 0.0949), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.7388, amazing-grace-new-britain:phrase16w 0.2878, amazing-grace-new-britain:phrase16w2 0.2574, amazing-grace-new-britain:pad16 0.2690, america-the-beautiful-materna:phrase16w 0.4171, america-the-beautiful-materna:phrase16w2 0.2403, america-the-beautiful-materna:pad16 0.1828.
logistic-pitch: pooled Brier 0.3882, within-mix mean 0.4000, log loss 2.1518. Features: 8. Minus logistic: -0.1076 (-0.3205 to 0.1053), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.9375, amazing-grace-new-britain:phrase16w 0.3285, amazing-grace-new-britain:phrase16w2 0.3486, amazing-grace-new-britain:pad16 0.3500, america-the-beautiful-materna:phrase16w 0.2830, america-the-beautiful-materna:phrase16w2 0.2671, america-the-beautiful-materna:pad16 0.2853.
logistic-transcript: pooled Brier 0.2938, within-mix mean 0.2957, log loss 0.7881. Features: 9. Minus logistic: -0.2119 (-0.4869 to 0.0631), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.4233, amazing-grace-new-britain:phrase16w 0.2599, amazing-grace-new-britain:phrase16w2 0.2990, amazing-grace-new-britain:pad16 0.2621, america-the-beautiful-materna:phrase16w 0.3401, america-the-beautiful-materna:phrase16w2 0.2190, america-the-beautiful-materna:pad16 0.2666.
logistic-timing-pitch: pooled Brier 0.4379, within-mix mean 0.4486, log loss 2.4820. Features: 13. Minus logistic-all: 0.1262 (-0.0261 to 0.2785), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.9375, amazing-grace-new-britain:phrase16w 0.3171, amazing-grace-new-britain:phrase16w2 0.3258, amazing-grace-new-britain:pad16 0.5452, america-the-beautiful-materna:phrase16w 0.3557, america-the-beautiful-materna:phrase16w2 0.2713, america-the-beautiful-materna:pad16 0.3874.
logistic-plus-joins: pooled Brier 0.3964, within-mix mean 0.4071, log loss 2.2965. Features: 15. Minus logistic-all: 0.0848 (-0.0485 to 0.2180), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.9375, amazing-grace-new-britain:phrase16w 0.2844, amazing-grace-new-britain:phrase16w2 0.3037, amazing-grace-new-britain:pad16 0.4350, america-the-beautiful-materna:phrase16w 0.3634, america-the-beautiful-materna:phrase16w2 0.2821, america-the-beautiful-materna:pad16 0.2439.
logistic-plus-segment: pooled Brier 0.3276, within-mix mean 0.3394, log loss 2.0637. Features: 23. Minus logistic-all: 0.0171 (-0.0623 to 0.0964), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.9375, amazing-grace-new-britain:phrase16w 0.1950, amazing-grace-new-britain:phrase16w2 0.3307, amazing-grace-new-britain:pad16 0.2598, america-the-beautiful-materna:phrase16w 0.2794, america-the-beautiful-materna:phrase16w2 0.1900, america-the-beautiful-materna:pad16 0.1836.
logistic-plus-measures: pooled Brier 0.2831, within-mix mean 0.2968, log loss 1.3952. Features: 32. Minus logistic-all: -0.0256 (-0.0902 to 0.0390), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.9358, amazing-grace-new-britain:phrase16w 0.1985, amazing-grace-new-britain:phrase16w2 0.2833, amazing-grace-new-britain:pad16 0.1915, america-the-beautiful-materna:phrase16w 0.1658, america-the-beautiful-materna:phrase16w2 0.1688, america-the-beautiful-materna:pad16 0.1335.
shallow tree, 22 receipt features: pooled Brier 0.3288, within-mix mean 0.3429, log loss 1.0586. Features: 22. Minus logistic: -0.1647 (-0.4352 to 0.1057), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.8839, amazing-grace-new-britain:phrase16w 0.2340, amazing-grace-new-britain:phrase16w2 0.2460, amazing-grace-new-britain:pad16 0.4452, america-the-beautiful-materna:phrase16w 0.2407, america-the-beautiful-materna:phrase16w2 0.2577, america-the-beautiful-materna:pad16 0.0926.
shallow tree, 41 features: pooled Brier 0.2542, within-mix mean 0.2608, log loss 0.7171. Features: 41. Minus logistic-all: -0.0616 (-0.2864 to 0.1633), not resolvable. Minus base-rate: pooled -0.0235, past 0.02. Fold mean -0.0177 (-0.1209 to 0.0854), the interval includes zero, so it is not claimed. Within mix: amazing-grace-new-britain:phrase16 0.5386, amazing-grace-new-britain:phrase16w 0.2330, amazing-grace-new-britain:phrase16w2 0.2752, amazing-grace-new-britain:pad16 0.2023, america-the-beautiful-materna:phrase16w 0.2511, america-the-beautiful-materna:phrase16w2 0.2070, america-the-beautiful-materna:pad16 0.1185.
tabpfn: not run (fit).
tabpfn-all: not run (fit).
featllm: pooled Brier 0.2812, within-mix mean 0.2837, log loss 0.7773. Features: 9. Minus logistic: -0.2239 (-0.4922 to 0.0444), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.4761, amazing-grace-new-britain:phrase16w 0.2335, amazing-grace-new-britain:phrase16w2 0.2903, amazing-grace-new-britain:pad16 0.2142, america-the-beautiful-materna:phrase16w 0.3197, america-the-beautiful-materna:phrase16w2 0.2273, america-the-beautiful-materna:pad16 0.2250.
logistic-strict: pooled Brier 0.3978, within-mix mean 0.4037, log loss 2.3675. Features: 22. Within mix: amazing-grace-new-britain:phrase16 0.8750, amazing-grace-new-britain:phrase16w 0.2485, amazing-grace-new-britain:phrase16w2 0.2481, amazing-grace-new-britain:pad16 0.4231, america-the-beautiful-materna:phrase16w 0.2307, america-the-beautiful-materna:phrase16w2 0.1907, america-the-beautiful-materna:pad16 0.6095.
jev-full-0: pooled Brier 0.3341, within-mix mean 0.3310, log loss 0.8834. Minus logistic: -0.1766 (-0.5935 to 0.2403), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.0981, amazing-grace-new-britain:phrase16w 0.3298, amazing-grace-new-britain:phrase16w2 0.3356, amazing-grace-new-britain:pad16 0.4651, america-the-beautiful-materna:phrase16w 0.2132, america-the-beautiful-materna:phrase16w2 0.3506, america-the-beautiful-materna:pad16 0.5246.

Secondary analysis, warp-placement mixes only. amazing-grace-new-britain:phrase16 stays in the primary table and is left out of this one. Leave-one-mix-out on 108 phrases, 6 mixes. The same claim bar applies.

base-rate: pooled Brier 0.2531, within-mix mean 0.2503, log loss 0.7068. Minus logistic: 0.0343 (-0.0964 to 0.1651), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2610, amazing-grace-new-britain:phrase16w2 0.2610, amazing-grace-new-britain:pad16 0.1531, america-the-beautiful-materna:phrase16w 0.4125, america-the-beautiful-materna:phrase16w2 0.2467, america-the-beautiful-materna:pad16 0.1674.
logistic: pooled Brier 0.2141, within-mix mean 0.2160, log loss 0.6158. Features: 22. Within mix: amazing-grace-new-britain:phrase16w 0.2405, amazing-grace-new-britain:phrase16w2 0.3047, amazing-grace-new-britain:pad16 0.1529, america-the-beautiful-materna:phrase16w 0.3107, america-the-beautiful-materna:phrase16w2 0.2870, america-the-beautiful-materna:pad16 0.0000.
logistic-all: pooled Brier 0.1936, within-mix mean 0.1959, log loss 0.7941. Features: 41. Minus logistic: -0.0200 (-0.0541 to 0.0140), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2219, amazing-grace-new-britain:phrase16w2 0.3110, amazing-grace-new-britain:pad16 0.1183, america-the-beautiful-materna:phrase16w 0.2900, america-the-beautiful-materna:phrase16w2 0.2342, america-the-beautiful-materna:pad16 0.0000.
logistic-timing: pooled Brier 0.2590, within-mix mean 0.2574, log loss 0.7230. Features: 5. Minus logistic: 0.0414 (-0.0884 to 0.1713), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2472, amazing-grace-new-britain:phrase16w2 0.2793, amazing-grace-new-britain:pad16 0.2020, america-the-beautiful-materna:phrase16w 0.4161, america-the-beautiful-materna:phrase16w2 0.2327, america-the-beautiful-materna:pad16 0.1669.
logistic-pitch: pooled Brier 0.2105, within-mix mean 0.2105, log loss 0.5906. Features: 8. Minus logistic: -0.0054 (-0.0511 to 0.0402), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2486, amazing-grace-new-britain:phrase16w2 0.2871, amazing-grace-new-britain:pad16 0.0975, america-the-beautiful-materna:phrase16w 0.3427, america-the-beautiful-materna:phrase16w2 0.2873, america-the-beautiful-materna:pad16 0.0000.
logistic-transcript: pooled Brier 0.2726, within-mix mean 0.2712, log loss 0.7596. Features: 9. Minus logistic: 0.0553 (-0.0842 to 0.1948), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2948, amazing-grace-new-britain:phrase16w2 0.3038, amazing-grace-new-britain:pad16 0.1784, america-the-beautiful-materna:phrase16w 0.4160, america-the-beautiful-materna:phrase16w2 0.2325, america-the-beautiful-materna:pad16 0.2020.
logistic-timing-pitch: pooled Brier 0.2185, within-mix mean 0.2205, log loss 0.6157. Features: 13. Minus logistic-all: 0.0246 (-0.0152 to 0.0643), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2218, amazing-grace-new-britain:phrase16w2 0.3182, amazing-grace-new-britain:pad16 0.1761, america-the-beautiful-materna:phrase16w 0.3253, america-the-beautiful-materna:phrase16w2 0.2816, america-the-beautiful-materna:pad16 0.0000.
logistic-plus-joins: pooled Brier 0.2262, within-mix mean 0.2279, log loss 0.6432. Features: 15. Minus logistic-all: 0.0319 (-0.0194 to 0.0833), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2192, amazing-grace-new-britain:phrase16w2 0.3216, amazing-grace-new-britain:pad16 0.1876, america-the-beautiful-materna:phrase16w 0.3537, america-the-beautiful-materna:phrase16w2 0.2849, america-the-beautiful-materna:pad16 0.0000.
logistic-plus-segment: pooled Brier 0.1968, within-mix mean 0.2018, log loss 0.6063. Features: 23. Minus logistic-all: 0.0059 (-0.0587 to 0.0706), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.1985, amazing-grace-new-britain:phrase16w2 0.3689, amazing-grace-new-britain:pad16 0.1730, america-the-beautiful-materna:phrase16w 0.2789, america-the-beautiful-materna:phrase16w2 0.1917, america-the-beautiful-materna:pad16 0.0000.
logistic-plus-measures: pooled Brier 0.1913, within-mix mean 0.1934, log loss 0.7624. Features: 32. Minus logistic-all: -0.0025 (-0.0088 to 0.0038), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2159, amazing-grace-new-britain:phrase16w2 0.3056, amazing-grace-new-britain:pad16 0.1165, america-the-beautiful-materna:phrase16w 0.2941, america-the-beautiful-materna:phrase16w2 0.2284, america-the-beautiful-materna:pad16 0.0000.
shallow tree, 22 receipt features: pooled Brier 0.1962, within-mix mean 0.1993, log loss 0.5831. Features: 22. Minus logistic: -0.0167 (-0.0876 to 0.0542), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2245, amazing-grace-new-britain:phrase16w2 0.2486, amazing-grace-new-britain:pad16 0.2083, america-the-beautiful-materna:phrase16w 0.2457, america-the-beautiful-materna:phrase16w2 0.2537, america-the-beautiful-materna:pad16 0.0147.
shallow tree, 41 features: pooled Brier 0.1939, within-mix mean 0.1975, log loss 0.5824. Features: 41. Minus logistic-all: 0.0016 (-0.0686 to 0.0718), not resolvable. Minus base-rate: pooled -0.0592, past 0.02. Fold mean -0.0528 (-0.1784 to 0.0729), the interval includes zero, so it is not claimed. Within mix: amazing-grace-new-britain:phrase16w 0.2416, amazing-grace-new-britain:phrase16w2 0.2774, amazing-grace-new-britain:pad16 0.1727, america-the-beautiful-materna:phrase16w 0.2326, america-the-beautiful-materna:phrase16w2 0.2155, america-the-beautiful-materna:pad16 0.0455.
tabpfn: not run (fit).
tabpfn-all: not run (fit).
featllm: pooled Brier 0.2545, within-mix mean 0.2533, log loss 0.7388. Features: 9. Minus logistic: 0.0373 (-0.0918 to 0.1664), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.2781, amazing-grace-new-britain:phrase16w2 0.3026, amazing-grace-new-britain:pad16 0.1464, america-the-beautiful-materna:phrase16w 0.3966, america-the-beautiful-materna:phrase16w2 0.2233, america-the-beautiful-materna:pad16 0.1727.
logistic-strict: pooled Brier 0.1486, within-mix mean 0.1519, log loss 0.4254. Features: 22. Within mix: amazing-grace-new-britain:phrase16w 0.1546, amazing-grace-new-britain:phrase16w2 0.2598, amazing-grace-new-britain:pad16 0.1280, america-the-beautiful-materna:phrase16w 0.1709, america-the-beautiful-materna:phrase16w2 0.1978, america-the-beautiful-materna:pad16 0.0000.
jev-full-0: pooled Brier 0.3690, within-mix mean 0.3698, log loss 0.9606. Minus logistic: 0.1539 (-0.1967 to 0.5044), not resolvable. Within mix: amazing-grace-new-britain:phrase16w 0.3298, amazing-grace-new-britain:phrase16w2 0.3356, amazing-grace-new-britain:pad16 0.4651, america-the-beautiful-materna:phrase16w 0.2132, america-the-beautiful-materna:phrase16w2 0.3506, america-the-beautiful-materna:pad16 0.5246.

Mix probe, receipt features: accuracy 0.4766, majority 0.1613, macro recall 0.4850. The same tree names the mix. The pooled score is a mix detector.
Mix probe, all features: accuracy 0.7226, majority 0.1613, macro recall 0.7170. The same tree names the mix. The pooled score is a mix detector.

The random 20 drawn with seed 20261008 stays out of the shot pool for this sweep. It is not a test. `future_test` in the phrase table marks that pool.

Serialisation cells, leave-one-mix-out, extended labels. This sweep's Jev states were fixed before the evidence block, and they do not include it. A completed cell is compared with the receipt logistic.

full-0: pooled Brier 0.3341, within-mix mean 0.3310, log loss 0.8834. Minus logistic: -0.1766 (-0.5935 to 0.2403), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.0981, amazing-grace-new-britain:phrase16w 0.3298, amazing-grace-new-britain:phrase16w2 0.3356, amazing-grace-new-britain:pad16 0.4651, america-the-beautiful-materna:phrase16w 0.2132, america-the-beautiful-materna:phrase16w2 0.3506, america-the-beautiful-materna:pad16 0.5246.
pruned-0: pooled Brier 0.2595, within-mix mean 0.2609, log loss 0.7178. Minus logistic: -0.2467 (-0.6054 to 0.1119), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.3733, amazing-grace-new-britain:phrase16w 0.2691, amazing-grace-new-britain:phrase16w2 0.3260, amazing-grace-new-britain:pad16 0.1179, america-the-beautiful-materna:phrase16w 0.3543, america-the-beautiful-materna:phrase16w2 0.2472, america-the-beautiful-materna:pad16 0.1386.
rows-0: pooled Brier 0.2708, within-mix mean 0.2729, log loss 0.7456. Minus logistic: -0.2348 (-0.5842 to 0.1147), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.3889, amazing-grace-new-britain:phrase16w 0.2758, amazing-grace-new-britain:phrase16w2 0.3764, amazing-grace-new-britain:pad16 0.1132, america-the-beautiful-materna:phrase16w 0.3306, america-the-beautiful-materna:phrase16w2 0.2423, america-the-beautiful-materna:pad16 0.1828.
pruned-4: pooled Brier 0.1719, within-mix mean 0.1698, log loss 0.5289. Minus logistic: -0.3378 (-0.7426 to 0.0670), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.1217, amazing-grace-new-britain:phrase16w 0.1763, amazing-grace-new-britain:phrase16w2 0.2158, amazing-grace-new-britain:pad16 0.0996, america-the-beautiful-materna:phrase16w 0.2172, america-the-beautiful-materna:phrase16w2 0.2677, america-the-beautiful-materna:pad16 0.0903.
pruned-8: pooled Brier 0.1881, within-mix mean 0.1863, log loss 0.5694. Minus logistic: -0.3213 (-0.7088 to 0.0661), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.1397, amazing-grace-new-britain:phrase16w 0.1773, amazing-grace-new-britain:phrase16w2 0.2420, amazing-grace-new-britain:pad16 0.1296, america-the-beautiful-materna:phrase16w 0.2276, america-the-beautiful-materna:phrase16w2 0.2517, america-the-beautiful-materna:pad16 0.1360.
pruned-16: pooled Brier 0.1804, within-mix mean 0.1789, log loss 0.5487. Minus logistic: -0.3287 (-0.7054 to 0.0480), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.1738, amazing-grace-new-britain:phrase16w 0.1664, amazing-grace-new-britain:phrase16w2 0.2310, amazing-grace-new-britain:pad16 0.0970, america-the-beautiful-materna:phrase16w 0.2218, america-the-beautiful-materna:phrase16w2 0.2455, america-the-beautiful-materna:pad16 0.1167.
full-4: pooled Brier 0.2366, within-mix mean 0.2334, log loss 0.6760. Minus logistic: -0.2742 (-0.6923 to 0.1439), not resolvable. Within mix: amazing-grace-new-britain:phrase16 0.0989, amazing-grace-new-britain:phrase16w 0.2398, amazing-grace-new-britain:phrase16w2 0.2589, amazing-grace-new-britain:pad16 0.2359, america-the-beautiful-materna:phrase16w 0.2031, america-the-beautiful-materna:phrase16w2 0.3654, america-the-beautiful-materna:pad16 0.2318.
Spread 0.1622. The spread clears 0.02.
Not sent: full-8, rows-4, rows-8, rows-16. The next cell's estimate would have passed the $0.25 cap, so the run stopped.

New Jev calls cost $0.1579. With the recorded calibration spend of $0.0217, the total is $0.1795, under the $0.25 cap.
full-16 is not attempted: sixteen full records do not fit the state cap.
The local models read phrase-evidence.json (schema ai-jam-sessions/phrase-evidence/v1, revision 1). This sweep's Jev states do not include that block.
Jev is a tested negative. Hosted Jev and local OpenJev, on these 124 records, neither beat the base rate once calibrated: Brier 0.252 and 0.251 against 0.242. Within-mix AUC is 0.57 for hosted Jev and 0.59 for OpenJev. The serialisation cells are the spend record. They are not a reason to continue. The evidence-family rows are the grouped comparison.
The next labels are not drawn by uncertainty sampling. About 30% of each later round is random. The random 20 is not held out of a later training set.
The phrase-clean decision layer stays insufficient evidence. What comes next is new labels: a blind re-mark from about 2026-10-21, and marks on more mixes once the review moves into the cockpit. Not a new model.

<!-- /result -->
