# Phrase calibration

One capped study of the question "is this phrase clean?" The uncertain band in
`phrase_clean` is still the placeholder 0.35–0.65. This page does not change it.

The README still says a live call waits on a separate yes. This study is that
yes: `typesafe/jev-1.13`, stamp `jev-1.13-20260917`, cap $0.25, key from the
environment. An answer that echoes `typesafe/jev-1.13-20260917` is this pin.
The study still records `typesafe/jev-1.13` and the date. It does not enable
the client. Importing the package still spends nothing.

The rule below was locked before any phrase was scored. It follows the R&D
entry `2026-10-07-calibrating-probabilistic-decision-layers`.

## Rule

A phrase is not clean when a review mark's window `[t - 1.0, t + 0.2]` overlaps
the phrase's `[start, end]`. Both intervals are closed. The mark is the label.
The record Jev sees is the phrase's sliced receipt and pitch, plus that phrase's
listener transcript. The marks list on that record is empty. A mix verdict is
not a label.

The split is stratified, seed `20261007`. Thirty percent of each class is an
outer holdout. The band rule is fit only on the inner phrases. Repeated
stratified 5-fold, 10 repeats, scores the inner phrases. A phrase is never
scored with a temperature or a fold-band fit on itself.

Recalibration is temperature scaling, one parameter, fit by negative log
likelihood on a log grid from 0.05 to 20, ties closest to 1 (Guo, Pleiss, Sun,
and Weinberger, 2017). Platt, beta, and isotonic are not used as
recalibrators. The product compares the raw P(yes) with the band, so the band
is chosen on the raw probability. Temperature is reported, and it is not wired
into the question.

The headline score is the inner out-of-fold Brier score, raw and
temperature-scaled, each with a phrase-level bootstrap 95% interval (1000
resamples). The outer holdout has its own Brier. The reliability diagram is
CORP: pool-adjacent-violators on the inner out-of-fold temperature-scaled
probabilities, with a 95% consistency band from Bernoulli resamples of those
probabilities (Dimitriadis, Gneiting, and Jordan, 2021). Isotonic regression
appears only in that diagram. Binned ECE is not reported.

"Too close to call" is a risk-coverage threshold (Geifman and El-Yaniv, 2017,
Algorithm 1). Confidence is `max(p, 1-p)`. The bound is the Clopper-Pearson
upper bound in their Lemma 3.1, at δ = 0.001, target risk 0.10. Probes are the
full inner resample plus `ceil(log2 n)` binary-search thresholds, and each
probe is charged δ divided by that probe count. If the full sample already
meets the bound, there is no interior band. Otherwise the lowest passing
threshold on the path is the band `(1-θ, θ)`. The study abstains when `p` is
strictly inside that interval.

The band's edges are the medians of 1000 stratified resamples of the inner
phrases. They swing when a 95% interval is wider than 0.10, or when more than
5% of resamples have no interior band. They are not adopted when either class
has fewer than 30 phrases on the full set. They are not adopted unless the
outer holdout, scored only with that median band, has accuracy of at least 90%
and a one-sided 95% Clopper-Pearson lower bound of at least 90%. If any gate
fires, the data does not support a band yet, and 0.35–0.65 stays.

## Result

<!-- result -->

Model `typesafe/jev-1.13`, stamp `jev-1.13-20260917`. 124 calls, $0.0217 of the $0.25 cap.

Phrases: 124. Clean: 73. Not clean: 51. Inner 87, outer holdout 37.

Inner out-of-fold Brier, raw P(yes): 0.3304 (bootstrap 0.2825 to 0.3760).
Inner out-of-fold Brier, temperature-scaled: 0.2518 (bootstrap 0.2493 to 0.2544).
CORP diagram: docs/calibration/corp.svg. 33 of 87 inner phrases sit outside the 95% consistency band.

No interior band showed up often enough to quote an edge.
Outer phrases answered by the diagnostic band: 0. Accuracy n/a, 95% lower bound n/a.

The data does not support a band yet. The placeholder 0.35–0.65 stays. The gates that fired: more than 5% of the inner resamples had no interior band; a 95% interval on a band edge is wider than 0.10; outer accuracy is under 90%, or the outer band answered nothing; the outer 95% lower bound on accuracy is under 90%.

Binned ECE is not reported. Isotonic regression is the CORP diagram only.

<!-- /result -->
