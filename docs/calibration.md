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

Not run.

<!-- /result -->
