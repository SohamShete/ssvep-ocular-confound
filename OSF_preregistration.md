# Pre-registration: Ocular contribution to target decoding in SSVEP-BCI benchmarks

**Authors:** Soham Sanjyot Shete, Ananta [surname], Suraj Prakash Sahoo
School of Electronics Engineering (SENSE), Vellore Institute of Technology, Vellore

**Date:** [fill in on submission]
**Registration type:** Secondary analysis of existing public data
**Data access status:** Registration following preliminary analysis of the data (see §2)

---

## 1. Study information

### 1.1 Research question

In the SSVEP Benchmark dataset (Wang et al., 2017, IEEE TNSRE 25:1746–1752), 40 targets
occupy fixed positions in a 5x8 grid and participants are instructed to fixate the cued
target. Target identity is therefore confounded by design with gaze direction: the label
"target 17" and the label "eyes directed to +6.51 deg horizontal, -4.98 deg vertical" are
the same label and cannot be separated within this dataset.

We ask three questions:

- **RQ1.** How much target-identity information is present in activity low-pass filtered
  below 4 Hz — beneath the entire 8.0–15.8 Hz stimulus range and all harmonics — recorded
  from the standard nine-channel parieto-occipital montage used throughout the SSVEP-BCI
  literature?
- **RQ2.** Is that information attributable to ocular volume conduction, or to a
  non-ocular source such as anticipatory slow potentials or covert spatial attention?
- **RQ3.** Does removing the low-frequency component change reported accuracy differently
  across decoding method families (filter-bank vs. broadband deep networks)?

### 1.2 Hypotheses

Directional, numbered, and stated before the confirmatory analyses below.

- **H1.** Target identity is decodable above the permutation null from the nine-channel
  occipital montage, below 4 Hz, in the pre-stimulus window (samples 0–124), where no
  stimulus has yet appeared and no SSVEP can exist.
- **H2.** The same effect is present in the 0.14–0.64 s analysis window used for
  short-window decoding in the literature.
- **H3.** Decoding accuracy declines monotonically with increasing window onset, tracking
  the magnitude of the measured frontopolar gaze gradient rather than its sign.
- **H4.** The scalp distribution of the discriminative activation pattern is consistent
  with an anterior (ocular) generator: frontopolar maximum, monotonic posterior falloff,
  no independent occipital focus.
- **H5.** A forward simulation injecting the *measured* occipital gaze gradient into
  surrogate data with matched noise reproduces the observed occipital decoding accuracy
  (sufficiency of the ocular account).
- **H6.** Decoded vertical position scales linearly with true vertical eccentricity in
  degrees of visual angle, with a slope consistent with the independently measured
  microvolt-per-degree gradient.
- **H7.** Broadband deep-network baselines lose more accuracy under low-frequency removal
  than filter-bank methods, whose lowest sub-band begins at 8 Hz and already excludes the
  affected band.
- **H8.** The pattern in H1–H3 replicates on the BETA dataset (Liu et al., 2020), which
  uses a different speller geometry.

**We do not hypothesise, and will not claim, that published SSVEP accuracies are
inflated.** That claim requires evidence we do not expect this design to produce.

---

## 2. Prior access to the data — full disclosure

This is a re-analysis of a public dataset that we have already partially analysed. We
state exactly what has been observed so that confirmatory and exploratory claims can be
separated by any reader.

**Already performed and observed, on all 35 subjects, prior to this registration:**

1. FBCCA baseline reproduction at 0.5 / 0.7 / 1.0 s windows.
2. Frontopolar ocular gradient measurement (HEOG and VEOG, microvolts per degree) and
   its time course.
3. Ocular decoding pilot: 40-class, row and column accuracy across four windows, two
   channel sets, raw and EOG-regressed conditions, with 200-permutation nulls.

Results from (1)–(3) are **exploratory** and will be reported as such. They motivated
H1–H3, which are therefore *not* independent confirmations. We will state this in the
manuscript in plain language rather than presenting these as pre-registered predictions.

**Not yet performed, and genuinely confirmatory under this registration:**
H4, H5, H6, H7, H8; the occipital microvolt-per-degree measurement; the mastoid
re-reference control; the high-pass model fitting; all per-subject inferential statistics;
and all multiple-comparison correction.

---

## 3. Data

**Primary dataset.** Benchmark (Wang et al., 2017). 35 participants, 64 channels,
250 Hz, 6 blocks x 40 trials, 6 s epochs (0.5 s pre-stimulus, 5 s stimulation, 0.5 s
post). Obtained from the Tsinghua BCI Lab distribution
(bci.med.tsinghua.edu.cn/download.html), retrieved [date]. Acquisition passband
0.15–200 Hz, vertex reference, 50 Hz notch.

**Replication dataset.** BETA (Liu et al., 2020, Front. Neurosci. 14:627). 70
participants, same 40 targets, QWERTY-like speller, recorded outside a shielded room.

**Channel sets, verified by name against 64-channels.loc:**
- OCC-9 (zero-based): 47, 53, 54, 55, 56, 57, 60, 61, 62 = Pz, PO5, PO3, POz, PO4, PO6,
  O1, Oz, O2
- FRONT-5: 0, 1, 2, 3, 4 = Fp1, Fpz, Fp2, AF3, AF4
- EOG surrogate regressors: 0, 1, 2, 3, 4, 5, 13 = Fp1, Fpz, Fp2, AF3, AF4, F7, F8

**Epoching.** Analysis onset sample 160 = 125 (cue) + 35 (~140 ms visual latency).
Trial indexing: C-order reshape of (40 targets, 6 blocks), so target = index // 6.

**Layout structure.** Verified from Freq_Phase.mat: freq = 8.0 + 1.0*column +
0.2*row on a row-major 5x8 grid. Column correlates with stimulus frequency at
r = +0.992 and is therefore **not interpretable as evidence of gaze during stimulation**;
row correlates at r = +0.123 and is the only spatial axis separable from frequency. All
dissociation claims rest on the row axis. In the pre-stimulus window no stimulus exists,
so both axes are frequency-free and both are interpretable there.

---

## 4. Analysis plan

### 4.1 Preprocessing

Zero-phase 4th-order Butterworth low-pass at 4 Hz applied to the full epoch, then windows
sliced. Temporal decimation to 12.5 Hz (the 4 Hz low-pass requires only 8 Hz by Nyquist).
No trial rejection, no ICA in the primary analysis.

### 4.2 Decoding (H1, H2, H3)

Shrinkage LDA (Ledoit–Wolf, lsqr solver), leave-one-subject-out cross-validation.
Normalisation fitted on training folds only. Windows: pre-stimulus (0–124),
0.14–0.64 s, 0.14–1.14 s, 1.14–2.64 s, plus a sweep at 0.2–5.0 s for H3.

Chance is established empirically by label permutation within subject, not assumed at
1/40. **1000 permutations** for every number reported in the manuscript; the permutation
p-value floor is 1/(n+1), so 200 permutations cannot support claims below p = 0.005.
Row and column accuracy each receive their own permutation null; the 40-class null does
not transfer to them.

### 4.3 Mechanism (H4, H5, H6)

- **H4.** Decode from all 64 channels; convert LDA weight vectors to activation patterns
  by multiplying by the data covariance (Haufe et al., 2014, NeuroImage 87:96–110), since
  weights of linear models are not neurophysiologically interpretable. Plot topographically.
  Separately, decode from each channel alone and plot the accuracy map.
- **H5.** Measure microvolts per degree at each OCC-9 channel by regressing mean
  low-frequency amplitude on eccentricity. Build a forward model injecting exactly that
  gradient into surrogate occipital data with noise matched to the measured residual
  spectrum, run the identical decoder, and compare simulated to observed accuracy.
- **H6.** Regress decoded vertical position (in degrees) on true vertical eccentricity.
  Report slope with bootstrap CI and compare against the measured gradient.

### 4.4 Controls

- **EOG regression.** Unsupervised, no labels used, therefore cannot leak. Seven frontal
  regressors plus intercept, fitted per subject. Stated limitation: the surrogates are
  frontal EEG, so genuine frontal brain activity correlated with occipital activity is
  also removed; the control is conservative and may over-correct. We additionally
  **quantify the adequacy of the control** by measuring how much of the *known* ocular
  gradient it removes, rather than assuming it is complete.
- **Reference control.** The recording is vertex-referenced, so ocular potential at Cz is
  subtracted into every channel including occipital. We re-reference to linked mastoids
  (M1, M2; indices 32, 42) and repeat the primary decoding.
- **High-pass model.** Fit first- and second-order high-pass step responses to the
  group-mean gradient time course and report which fits and at what corner frequency. We
  note in advance that a single-pole high-pass predicts monotonic decay without sign
  reversal, so an observed reversal would indicate a higher-order response.

### 4.5 Method-family comparison (H7)

FBCCA, EEGNet, and one deep network, each under four conditions: raw, 6 Hz high-pass,
EOG regression, ICA correction. Spatial filters and ICA weights fitted on training folds
only. Ensemble DNN is excluded on documented compute grounds (its authors report ~45 h
for one window length) and its published numbers cited instead.

### 4.6 Replication (H8)

BETA, identical pipeline, no parameter changes. Speller geometry recomputed separately
because the layout differs.

---

## 5. Inference criteria

- **Unit of analysis is the subject, not the trial.** Per-subject accuracies, Wilcoxon
  signed-rank against the permutation null, bootstrap 95% CIs. Trial-pooled z-scores
  inflate with sample size and will not be used as primary evidence.
- **Multiple comparisons.** Benjamini–Hochberg FDR at q = 0.05 within each hypothesis
  family. Families are declared in advance: (a) decoding by window and montage, (b)
  gradient measurements, (c) method-family comparisons.
- **Null claims.** Any claim that an effect is absent will be supported by a Bayes factor
  (BF01 > 3 as the threshold for evidence of absence), not by a non-significant p-value.
- **Effect sizes** with bootstrap CIs reported for every comparison, significant or not.
- **Seeds.** At least 5 seeds where stochastic; variation across seeds and across subjects
  reported separately, never pooled.

## 6. Exclusions

No participant, block, or trial exclusions are planned. If any file fails the integrity
checks in verify_data.py (wrong shape, non-finite values, flat channels), that subject is
excluded and the exclusion reported with its reason. No exclusion will be made on the
basis of decoding outcome.

## 7. Sample size

Fixed by the datasets: 35 (Benchmark) and 70 (BETA). No stopping rule applies. A
sensitivity analysis will report the smallest effect detectable at 80% power given these
sample sizes.

## 8. Contingencies and outcome-independent value

We commit in advance to reporting the result whichever way it falls:

- Occipital effect present and ocular in origin -> benchmark accuracy conflates cortical
  and ocular sources; method families differentially affected.
- Occipital effect present but surviving all ocular controls -> the source is not ocular;
  we report that honestly and do not describe it as an ocular confound.
- No occipital effect -> first rigorous demonstration that the field's most-used benchmark
  is free of ocular leakage in the standard montage.

## 9. Code and data availability

All analysis code will be public at [repository URL] with a Zenodo DOI minted at
submission. Per-subject results will be released as supplementary tables. A preprint will
be posted at submission.

## 10. Declarations

Data are public and were obtained under the original distribution terms; Wang et al.
(2017) is cited as the data source. The study involves no new human data collection. AI
assistance was used for code review, error checking and argument critique; all text is
author-written and all authors are accountable for every claim and number. No competing
interests.
