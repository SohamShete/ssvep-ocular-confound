# SSVEP Ocular-Confound Study — Complete Handoff

**Written 8 September 2026. Paste this whole file into a new chat as the first message.**

Upload alongside it: `verify_data.py`, `verify_claims.py`, `fbcca_baseline.py`,
`measure_gradient.py`, `ocular_pilot.py`, `topography.py`, and the result CSVs
(`fbcca_results_2.csv`, `gradient_results.csv`, `gradient_timecourse.csv`,
`ocular_pilot_35.csv`).

---

## 0. How to use this document

Sections 1–7 are settled fact: numbers already measured, traps already found,
mistakes already made. Do not re-derive them.

Section 8 is the decision tree. **The paper's framing is not fixed.** Three
pending analyses each have the power to change the title, the claim set and the
target venue. Section 8 says exactly which result implies which paper. Read it
before writing any prose.

Sections 9–13 are process: statistics, schedule, venue, integrity, next actions.

---

## 1. People and constraints

| | |
|---|---|
| Soham Sanjyot Shete | B.Tech ECE, VIT Vellore, 2024–2028. Hardware/PCB/embedded (NGX Technologies internship: isolated AC-DC, ESP32, BOM sourcing; Kineton VLSI internship; Team VAUV AUV electronics, ROS2). Owns pipeline, repo, draft. **First author.** |
| Adya Razdan | B.Tech ECE, VIT, 2024–2028. STM32, Python, MATLAB, DSP; Brose India internship (automotive tier-1); prior ML on biosignals. Owns analysis scripts and figures. **Second author.** |
| Dr. Suraj Prakash Sahoo | Assistant Professor, SENSE, VIT Vellore. h-index 13, 445 citations. Deep learning / pattern recognition / human action recognition / ECG. Publishes in IEEE TIM, IEEE TETCI, Expert Systems with Applications, Displays, DSP. **Last author, corresponding.** |
| Ananta Jaiswal | ECE, VIT, class of 2027. Was a third member; group currently inactive. May rejoin. Not currently contributing. |

**Important about the supervisor:** Suraj has no EEG or BCI publications. He will
vet ML and signal processing competently but cannot tell you whether prior work
exists in this literature — that burden is entirely yours. His one adjacent
paper is *GA-EMDNet: Graph Attention-guided Eye Movement Detection Network from
EMG Sensor signals* (IEEE Sensors Letters, 2026) — read it, it gives shared
vocabulary for the ocular discussion. He has published in **IEEE TIM**, which is
where CCA-Net appeared, so he knows that venue's review culture from the inside.

**Context:** this began as an AI/ML course project. The ML deliverable is
analysis A4 (method-family comparison, §8.5) — genuine ML, scientifically
motivated, not bolted on. Do not add TRCA/TDCA/architecture search to satisfy
the course; A4 covers it.

**Timeline:** submit by end of November 2026. *Published* by November is
impossible — first editorial decision takes 2–4 months, acceptance 7–13 months.
Submission by late November is comfortable. Twelve weeks available, eight needed.

**Hardware:** Windows 11, Lenovo Legion, Intel Core Ultra 7 255HX, 32 GB RAM,
RTX 5060 (8 GB, Blackwell sm_120). Python 3.13 venv at `C:\ssvep\venv_313`.
Sufficient for everything planned. Nothing needs a cluster.

---

## 2. The paper as currently framed

**One sentence.** The 40 targets sit at fixed screen positions, so "target 17"
and "eyes at +6.51° horizontal, −4.98° vertical" are the same label; we measure
how much target information is gaze rather than cortex, and hand the field a
control procedure for it.

**Why the design cannot separate them.** Subjects are instructed to fixate the
cued target. Target identity is confounded with gaze direction *by construction*.
Every accuracy figure in this literature, including the ITR records, was measured
under that confound.

**The test.** Low-pass below 4 Hz — beneath the entire 8.0–15.8 Hz stimulus range
and every harmonic — then attempt to decode target anyway. Nothing surviving that
filter can be an SSVEP.

**What is NOT claimed, and must not be:**
- That SSVEP is invalid.
- That gaze-dependence was unknown (the field discusses it openly; the BETA paper
  itself says its paradigm "falls into the category of dependent BCI where
  subjects were instructed to redirect their gaze," limiting patient use).
- That artifacts inflating decoding accuracy is a new discovery.
- **That published SSVEP accuracies are inflated.** Not shown. The effect inside
  the real analysis window is small in absolute terms, and FBCCA's lowest
  sub-band starts at 8 Hz so it discards the affected band by accident.

---

## 3. Established results — all measured, all reproducible

### 3.1 Pipeline validation (the gate) — PASSED

FBCCA, 35 subjects, 9-channel occipital montage, 40 classes, chance 2.5%:

| Window | Mean ± SEM | Range | ITR |
|---|---|---|---|
| 0.5 s | 21.39 ± 2.16% | 3.3–59.6% | 25.1 bpm |
| 0.7 s | 41.51 ± 3.41% | 5.0–87.1% | 62.6 bpm |
| **1.0 s** | **65.44 ± 3.79%** | 14.6–96.2% | 102.6 bpm |

Published FBCCA is 60–65% at 1.0 s. Gate passed.

**A leakage bug was found and fixed.** The original script applied
`sosfiltfilt` to the full 1500-sample trial and *then* sliced the analysis
window. Zero-phase filtering runs backwards, dragging SSVEP from seconds 1–5
into the 0–1 s window. This inflated 5-subject accuracy to 83.92%. Fixed by
slicing the window first, then filtering:

```python
# WRONG - leaks future data backwards into the window
filtered = [scipy.signal.sosfiltfilt(sos, occ, axis=1)[:, onset:onset+n, :]
            for sos in bank]

# CORRECT - cut first, then filter
window_occ = occ[:, onset:onset + n_samples, :]
filtered = [scipy.signal.sosfiltfilt(sos, window_occ, axis=1) for sos in bank]
```

Mention this in the paper's Methods. It is a real trap for short-window SSVEP work.

### 3.2 Dataset integrity — ALL CHECKS PASS

35 subjects, 105–107 MB each, shape (64, 1500, 40, 6), std 8.7–29.1 µV.
40 frequencies 8.0–15.8 Hz in 0.2 Hz steps. 40 phases, 0–4.712 rad.

**Channel mapping verified by name against `64-channels.loc`:**
- OCC-9 (0-based) `[47, 53, 54, 55, 56, 57, 60, 61, 62]` = PZ PO5 PO3 POZ PO4 PO6 O1 OZ O2
- FRONT-5 (0-based) `[0, 1, 2, 3, 4]` = FP1 FPZ FP2 AF3 AF4
- Also useful: F7 = 5, F8 = 13, M1 = 32, M2 = 42

**Trial indexing:** C-order reshape of (40 targets, 6 blocks) → `target = idx // 6`.
**Analysis onset:** sample 160 = 125 (cue) + 35 (≈140 ms visual latency).

### 3.3 Layout structure — the structural constraint

Verified from `Freq_Phase.mat`: `freq = 8.0 + 1.0×col + 0.2×row` on a row-major 5×8 grid.

| Axis | r with frequency | Consequence |
|---|---|---|
| Column (horizontal) | **+0.992** | Confounded. Column decoding during stimulation is NOT evidence of gaze. |
| Row (vertical) | **+0.123** | The only spatial axis separable from frequency. All dissociation claims rest on it. |

**Crucial exception, easy to miss:** in the **pre-stimulus window no stimulus
exists**, so no frequency information exists, so *both* axes are clean there.
The column confound applies only to the stimulation windows. State this
explicitly in Methods — it protects your strongest measurement.

Awkward but must be owned in Methods: horizontal EOG is the large clean signal
but its axis is confounded; the interpretable axis is vertical, where the ocular
signal is weaker. Reviewers who know EOG will spot this immediately.

### 3.4 Speller geometry — independently reproduced

23.6″ 1920×1080 at 70 cm → screen 52.25 × 29.39 cm, pixel pitch 0.02721 cm/px
(horizontal and vertical agree). Matrix 1510×1037 px, stimuli 140 px.

- Columns (8): −14.91, −10.77, −6.51, −2.18, +2.18, +6.51, +10.77, +14.91° — span 29.8°, **adjacent step 4.26°**
- Rows (5): −9.89, −4.98, 0.00, +4.98, +9.89° — span 19.8°, **adjacent step 4.94°**

**All inputs remain UNVERIFIED against primary sources.** Monitor size, 70 cm
distance, matrix pixel dimensions and stimulus size were taken on trust from the
project plan. Confirm each against Wang et al. 2017 and Chen et al. 2015 Methods
before any of it enters a manuscript. Everything above collapses if one is wrong.

### 3.5 Ocular gradient — MEASURED, 35 subjects

Frontopolar, pre-stimulus window (samples 80–125, i.e. −0.18 to 0.00 s):

| | HEOG (columns) | VEOG (rows) |
|---|---|---|
| Mean | **−1.551 µV/deg** | **−1.512 µV/deg** |
| SD / SEM | 0.663 / 0.112 | 0.933 / 0.158 |
| Bootstrap 95% CI | [−1.78, −1.34] | [−1.81, −1.20] |
| Same sign across subjects | **35/35** | 32/35 |
| Wilcoxon vs zero | p = 5.8 × 10⁻¹¹ | p = 1.5 × 10⁻⁹ |

**Report the 35/35 sign consistency, not mean ± SEM.** It is the stronger
statement and immune to distributional assumptions.

Implied microvolt steps: HEOG adjacent column 6.61 µV, extreme 23.1 µV; VEOG
adjacent row 7.47 µV, extreme 14.95 µV.

**Delete the derived amplitude table from the plan.** It assumed 3.0–15.0 µV/deg
based on periorbital EOG literature. Measured frontopolar value is 1.55 — the
assumptions overestimated by 2–10×. Report measured, drop derived. A measured
number is unattackable where a derived one is not.

### 3.6 Time course and polarity reversal — 35 subjects

Group-mean HEOG gradient (µV/deg) vs time from stimulus onset:

```
-0.40  -0.186   |  0.80  +0.077   |  1.80  +0.345
-0.20  -1.651   |  1.00  +0.202   |  2.00  +0.310
 0.00  -1.250   |  1.20  +0.289   |  2.60  +0.146
 0.20  -0.739   |  1.40  +0.337   |  3.00  +0.085
 0.40  -0.376   |  1.60  +0.352 (peak)
 0.60  -0.114   |
```

**Zero crossing: group mean 0.71 s, bootstrap 95% CI [0.60, 0.83].
All 35/35 subjects cross.** Per-subject median 0.69 s, IQR [0.51, 0.86].

Quote the CI, never a bare "0.72 s" — you do not have two-decimal precision, and
the 200 ms sliding window plus the 4 Hz low-pass impulse response both limit
temporal resolution.

### 3.7 CRITICAL CORRECTION to the project plan's mechanism claim

The plan states that a sustained gaze offset through a **single-pole** 0.15 Hz
high-pass "must decay and undershoot." **This is wrong.** A step through a
first-order high-pass gives a monotone exponential decay to zero and never
crosses. Undershoot requires two or more poles.

So your data are *stronger* than the stated model, not explained by it. The decay
timescale is broadly consistent with a 0.15 Hz corner (τ = 1.0610 s; residuals
87.6% at 0.14 s, 62.4% at 0.50 s, 54.7% at 0.64 s, 34.1% at 1.14 s, 15.2% at
2.00 s, 0.9% at 5.00 s — all reproduced exactly). But the reversal implies the
amplifier high-pass is higher-order than single-pole.

**Action required:** find the actual SynAmps2 filter order. The Benchmark Readme
says only "passband 0.15 Hz to 200 Hz." Check the Neuroscan hardware manual or
email the Tsinghua contact (lbc14@tsinghua.org.cn is responsive on BETA issues).
Meanwhile, fit first- and second-order high-pass step responses to the measured
group time course and report which fits, at what corner frequency. That converts
a breakable claim into a measurement.

Incidental evidence the value is not fixed lab-wide: the **eldBETA** BIDS
metadata reports the same SynAmps2 high-pass as **0.1 Hz**, while Benchmark and
BETA both report **0.15 Hz**. Do not assume.

### 3.8 Ocular decoding — 35 subjects, LOSO, 200 permutations, decim 20

Shrinkage LDA (Ledoit-Wolf, lsqr), leave-one-subject-out, normaliser fitted on
training folds only. Nulls by within-subject label permutation.

**RAW condition:**

| Window | Set | 40-class | row | col |
|---|---|---|---|---|
| pre-stim | FRONT-5 | 11.00% (z=45.74) | 32.17% (z=28.34) | 30.73% (z=48.72) |
| pre-stim | **OCC-9** | **5.75% (z=18.45)** | **27.58% (z=17.37)** | 19.13% (z=18.04) |
| 0.14–0.64 | FRONT-5 | 6.79% (z=22.66) | 25.79% (z=13.23) | 23.49% (z=27.90) |
| 0.14–0.64 | **OCC-9** | **4.57% (z=13.39)** | **25.86% (z=13.13)** | 16.74% (z=10.84) |
| 0.14–1.14 | FRONT-5 | 6.87% (z=24.78) | 25.43% (z=13.44) | 23.61% (z=30.43) |
| 0.14–1.14 | OCC-9 | 4.50% (z=12.01) | 24.87% (z=11.08) | 17.08% (z=12.57) |
| 1.14–2.64 | FRONT-5 | 5.45% (z=17.41) | 23.76% (z=8.67) | 20.00% (z=22.72) |
| 1.14–2.64 | OCC-9 | 3.07% (z=3.11) | 21.60% (z=3.73) | 14.45% (z=4.96) |

Nulls landed at 2.44–2.55% (40-class), 19.95–20.14% (row), 12.37–12.57% (col) —
exactly at theoretical chance, which validates the permutation machinery.

**EOG-REGRESSED condition** (2 regressors only: Fp1−Fp2 and Fpz):

| Window | Set | 40-class | row | col |
|---|---|---|---|---|
| pre-stim | FRONT-5 | 5.79% (z=18.55) | 21.18% (z=2.61) | 23.48% (z=28.86) |
| pre-stim | OCC-9 | 4.56% (z=12.96) | 24.77% (z=10.56) | 16.61% (z=10.95) |
| 0.14–0.64 | FRONT-5 | 3.54% (z=5.85) | 20.10% (z=0.23, n.s.) | 17.63% (z=13.75) |
| 0.14–0.64 | OCC-9 | 3.99% (z=9.05) | 24.55% (z=10.22) | 15.51% (z=7.41) |
| 0.14–1.14 | FRONT-5 | 3.94% (z=9.15) | 19.88% (z=−0.21, n.s.) | 18.87% (z=17.33) |
| 0.14–1.14 | OCC-9 | 3.73% (z=7.13) | 23.86% (z=8.28) | 15.48% (z=8.23) |
| 1.14–2.64 | FRONT-5 | 3.23% (z=4.19) | 19.44% (n.s.) | 15.94% (z=9.66) |
| 1.14–2.64 | OCC-9 | 2.64% (z=0.79, n.s.) | 20.93% (z=2.08) | 13.11% (z=1.70) |

**Two findings that matter more than any single number:**

**(a) The effect is inside the real analysis window.** At 0.14–0.64 s — where the
field operates and every ITR record is set — OCC-9 gives 4.57% and row 25.86%,
z = 13.4 and 13.1. Earlier 5-subject runs had this marginal (z ≈ 2.5) and the
project plan told you not to claim it. At n=35 it is not marginal.

**(b) THE 2-REGRESSOR EOG CONTROL IS INADEQUATE. Demonstrated, not speculated.**

Percentage of excess-over-null removed, pre-stimulus:

| | removed |
|---|---|
| FRONT-5 row | 91% — **arithmetically forced**, Fpz *is* a regressor |
| FRONT-5 col | **40%** — known-ocular signal, control fails |
| OCC-9 row | 36% |
| OCC-9 col | 38% |
| OCC-9 40-class | 36% |

FRONT-5 column in the pre-stimulus window is horizontal gaze and nothing else —
no flicker exists. It is as unambiguously ocular as EEG gets. After regressing
out Fp1−Fp2 and Fpz it still decodes at 23.48%, **z = 28.86**. Why: Fpz becomes
identically zero (it is a regressor), and Fp1−Fp2 being a regressor forces
Fp1_resid = Fp2_resid, killing the frontopolar differential — but AF3 and AF4
carry HEOG too and were never in the surrogate. The signal relocated.

**Therefore: OCC-9 surviving at ~65% is exactly what an ocular source passed
through a control that leaves 60% of known-ocular signal standing would produce.
The control cannot establish mechanism in either direction.** Do not write
"ocular confirmed." Do not write "not ocular."

### 3.9 A mistake I made, recorded so it is not repeated

On the 5-subject run, the occipital row effect went to chance after EOG
regression and I concluded the mechanism test had passed and the source was
ocular. At 35 subjects it does not collapse (24.77%, z = 10.56). **That was an
over-read of underpowered data.** The lesson generalises: nothing from a
5-subject LOSO run should drive an interpretive claim. Treat 5-subject runs
purely as smoke tests for code correctness.

---

## 4. Literature position — checked, but not yet systematic

**Gate result: no prior work found quantifying the gaze/target confound on the
Benchmark or BETA datasets.** Provisional — this was four web searches, not a
systematic review. Do Scopus and Web of Science through the VIT library with
documented search strings and dates in week 1, and keep the log for the paper.

**Three findings that change the framing:**

1. **There is a template paper in a neighbouring field.** The KU Leuven group did
   this for auditory attention decoding: they built a gaze-controlled dataset and
   showed spatial AAD methods fail to reach above-chance performance on it,
   indicating previously reported results were mainly driven by eye-gaze
   confounds in existing datasets. Same claim shape, same field-level
   consequence. Read it, cite it, frame your Introduction the same way. It is
   also proof this kind of paper gets published and read.

2. **Stimulus-free-window decoding is NOT your invention.** It is standard
   practice in cognitive-neuroscience MVPA; methods guides state plainly that the
   baseline window quantifies the gap between theoretical and observed chance
   because no effects are expected before stimulus onset. The project plan calls
   this "novel as a protocol." **It is not, and claiming it would be caught
   immediately.** Reframe as: *importing an established MVPA control into
   SSVEP-BCI benchmarking, where it is not used.* Still a real contribution,
   much safer. Prior work also shows artifact correction may be essential to
   minimise confounds that artificially inflate decoding accuracy, and that
   artifactual decoding arises when eye movements are unevenly distributed across
   conditions — cite generously; it strengthens the narrow claim.

3. **The BETA paper acknowledges gaze-dependence itself.** Cite it as your
   foundation, never as your finding.

**Remaining genuine novelty:** quantified speller geometry; the measured
−1.55 µV/deg frontopolar gradient (35/35); the 0.71 s polarity reversal (35/35);
sub-4 Hz occipital decoding inside the standard analysis window; **and the
demonstration that a standard two-regressor EOG regression control leaves ~60% of
known-ocular signal intact** — that last is a methods finding other people need.

**Mandatory reading before writing:** Haufe et al. 2014, NeuroImage 87:96–110.
Weight vectors of linear models are not neurophysiologically interpretable. Plot
activation patterns (weights × data covariance), never weights. Reviewers in this
field check.

---

## 5. Data on disk — and the traps

### 5.1 Benchmark (primary) — WORKING
`C:\ssvep\data\` — S1.mat … S35.mat flat, plus `Freq_Phase.mat`.
`64-channels.loc` in the same folder. All integrity checks pass.
Cite Wang et al. 2017, IEEE TNSRE 25:1746–1752. Obtained from
bci.med.tsinghua.edu.cn/download.html — record the retrieval date.

### 5.2 BETA (replication) — NOT YET DOWNLOADED CORRECTLY

**THE SINGLE MOST IMPORTANT TRAP IN THIS PROJECT.**

BETA ships in two versions. The main release was **band-pass filtered 3–100 Hz
with eegfilt before distribution**. A separate directory, **"BETA Database
(wof)"**, holds the same data with no filtering; it was added in the October 2021
update. Your entire method lives below 4 Hz.

Two BETA files were spot-checked and they came from *different* releases:

| File | 0.5–2 Hz power | 8–15 Hz power | verdict |
|---|---|---|---|
| S4.mat | 67.20 | 0.63 | unfiltered, 1/f intact — CORRECT |
| S10.mat | 0.001 | 2.50 | high-passed near 3 Hz — USELESS |

On the filtered release your analysis returns chance, you conclude the effect
does not generalise, and the actual cause is that the band was deleted before you
loaded the file. **Download only "BETA Database (wof)."**

Add this check to `verify_data.py` and run it on every BETA file:

```python
import scipy.signal, numpy as np
def lowfreq_intact(x, fs=250.0):
    """x: (n_time,) or (n_time, n_trials). Returns (ratio, ok)."""
    fq, P = scipy.signal.welch(x, fs=fs, nperseg=min(512, x.shape[0]), axis=0)
    if P.ndim > 1:
        P = P.mean(axis=1)
    lo = P[(fq >= 0.5) & (fq < 2)].mean()
    hi = P[(fq >= 8) & (fq < 15)].mean()
    ratio = lo / hi
    # Unfiltered EEG is 1/f dominated: ratio >> 1. A 3 Hz high-pass drives it to ~0.
    return ratio, ratio > 1.5
```

**BETA format differences that will silently break the Benchmark pipeline:**

```python
# BETA is a MATLAB struct, not a plain array
d    = scipy.io.loadmat(path)['data'][0, 0]
eeg  = d['EEG']            # (64, n_time, 4, 40)
sup  = d['suppl_info'][0, 0]
# suppl_info fields: sub, age, gender, chan, freqs, phases,
#                    bci_quotient, srate  (+ wide_snr, narrow_snr on some)
```

| | Benchmark | BETA |
|---|---|---|
| Tensor order | channel × time × **target × block** | channel × time × **block × condition** |
| Trial index → target | `target = idx // 6` | **`target = idx % 40`** |
| Blocks | 6 | 4 |
| Trial length | 1500 samples, all subjects | **750 (S1–S15), 1000 (S16–S70)** |
| Stimulation | 5 s | 2 s (S1–S15), 3 s (S16–S70) |
| Visual latency | ~140 ms → onset 160 | **~130 ms → onset 158** |
| Pre/post | 0.5 s / 0.5 s | 0.5 s / 0.5 s |
| Recording | shielded room | outside shielded room, lower SNR |

Getting the dimension order wrong decodes noise while everything appears to run
fine. **The 1.14–2.64 s late window does not exist for BETA S1–S15.**

**BETA geometry is given directly in degrees in the paper** — do not derive from
pixels. 27-inch ASUS MG279Q, 1920×1080, 60 Hz. QWERTY layout, five rows, 30 px
spacing. Squares **3.1° × 3.1°** (136×136 px), space bar **21° × 3.1°**
(966×136 px). Frequency mapping: `f_k = 8.0 + (k−1)×0.2`, phase
`Φ_k = (k−1)×0.5π`, where k indexes dot, comma, backspace, then a–z, then 0–9,
then space.

**Phases were corrected in the October 2021 update.** Current correct values
begin [1.5π, 0, 0.5π, 1π, 1.5π, 0, …]. The spot-checked files have the corrected
values. There is also a corrigendum to Eq. 4 of the paper (`P(k·f)` → `P(k·f_n)`).

Download routes, in order: (1) bci.med.tsinghua.edu.cn/download.html — cite this;
(2) figshare collection 5034338; (3) Hugging Face `Bingchuan/BETA`, posted by the
first author, useful if the Tsinghua site is down (its own page has flagged the
lab site as under maintenance). Cite Liu et al. 2020, Front. Neurosci. 14:627
regardless of route.

### 5.3 Datasets downloaded by mistake — DELETE, do not use

- **MAMEM III** (figshare 3413851, files `U0xx*.mat`). Mislabelled "UCSD" in the
  team chat — **these are different datasets.** UCSD is Nakanishi et al. 2015
  (10 subjects, 12 targets, 8 channels, 256 Hz), which is what CCA-Net
  benchmarks on. MAMEM III is CERTH's Emotiv EPOC set: 11 subjects, 5 targets,
  6.66–12 Hz, 14 channels (AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4) at
  128 Hz, `prefilter` recorded as NaN. **Only O1 and O2 from your montage
  exist**, there is no Fp1/Fp2, and with 5 fixed positions at 5 fixed
  frequencies neither spatial axis is separable. Cannot replicate the headline.
- **Wearable SSVEP** (`S007.mat` shape (8,710,2,10,12), `Impedance.mat`,
  `Subjects_Information.mat`). 102 subjects but only 8 channels, all occipital
  (POz PO3 PO4 PO5 PO6 Oz O1 O2). No frontal channels at all, so no EOG
  surrogate and no possible control.
- **eldBETA** (`README`, `CHANGES`, `participants.tsv`, `dataset_description.json`,
  `task-ssvep_*.json`). 9 targets, elderly cohort, online paradigm with feedback,
  4 s cue. Different paradigm.
- **Cross-Session Collaborative BCI** (`62-channels.loc`). RSVP target/non-target.
  Unrelated.

**Stop downloading datasets.** You need Benchmark (have it) and BETA wof (get it).

---

## 6. Scripts — exact current state

All in `C:\ssvep\`. Activate with `cd C:\ssvep` then `venv_313\Scripts\activate`;
the prompt must show `(venv_313)`.

| Script | State | Purpose |
|---|---|---|
| `verify_claims.py` | unmodified, run, matches | Recomputes geometry and τ table from first principles. No data needed. |
| `verify_data.py` | has layout check §2b + channel-name check. **All 14 checks pass.** | File integrity, stimulus metadata, layout structure, channel mapping. |
| `fbcca_baseline.py` | **PATCHED** — window sliced before filtering (§3.1) | The gate. Calibration-free FBCCA. |
| `measure_gradient.py` | unmodified. **Measures only Fp1/Fpz/Fp2.** | µV/deg and time course. Needs occipital extension (§8.2). |
| `ocular_pilot.py` | **PATCHED TWICE. Patch 2 NOT YET RUN.** | The core decoding analysis. |
| `topography.py` | **NEW, NOT YET RUN.** | Per-channel accuracy map + Haufe activation patterns. |

**`ocular_pilot.py` patch 1 (run, results in §3.8):** separate permutation nulls
for row and column out of the same loop (the 40-class null does not transfer);
`--decim` default changed 10 → 20 (data is low-passed at 4 Hz so Nyquist needs
only 8 Hz; 12.5 Hz loses nothing and quarters runtime).

**`ocular_pilot.py` patch 2 (applied, NOT yet run) — this is the priority run:**

```python
OCC_ANT_IDX  = np.array([47])                       # Pz  (most anterior of the 9)
OCC_POST_IDX = np.array([60, 61, 62])               # O1 Oz O2 (the pole)
EOG_REG_IDX  = np.array([0, 1, 2, 3, 4, 5, 13])     # Fp1 Fpz Fp2 AF3 AF4 F7 F8
```

Seven raw frontal channels span every differential and common-mode combination
of the frontal montage including the classic F7/F8 horizontal pair, fixing the
inadequacy in §3.8(b). FRONT-5 is **skipped** in the regressed condition because
its channels *are* the regressors, making the residual zero by construction.
Stated limitation for the paper: seven frontal regressors may also remove genuine
frontal-occipital brain activity, so the control now errs toward
over-correction — which is the conservative direction.

**Runtime notes.** 35 subjects, 200 permutations, decim 20: ~2.5 h total, but the
1.14–2.64 s cells took 1927 s and 2913 s versus ~200–400 s for everything else.
Watch for thermal throttling — plug in, close Chrome, elevate the chassis.
Permutation p cannot fall below 1/(n+1), so 200 gives a floor of 0.005; use
`--n_perm 1000` for anything reported, at minimum on the pre-stimulus cells.

---

## 7. Exact commands, in order

```bat
cd C:\ssvep
venv_313\Scripts\activate

:: already done, re-run only to reconfirm
python verify_claims.py
python verify_data.py --data_dir "C:\ssvep\data" --loc_file "C:\ssvep\data\64-channels.loc"
python fbcca_baseline.py --data_dir "C:\ssvep\data" --out_csv fbcca_results_2.csv

:: PRIORITY 1 - topography, smoke test then full
pip install matplotlib
python topography.py --data_dir "C:\ssvep\data" --loc_file "C:\ssvep\data\64-channels.loc" --max_subjects 5
python topography.py --data_dir "C:\ssvep\data" --loc_file "C:\ssvep\data\64-channels.loc"

:: PRIORITY 2 - pilot with the 7-regressor surrogate and ANT/POST subsets
python ocular_pilot.py --data_dir "C:\ssvep\data" --max_subjects 5
python ocular_pilot.py --data_dir "C:\ssvep\data" --max_subjects 35 --n_perm 1000 --out_csv ocular_pilot_v2.csv
```

Later, when needed: `pip install pandas seaborn statsmodels pingouin h5py tqdm pyyaml`
(pingouin covers Wilcoxon, bootstrap CIs and Bayes factors). PyTorch only for
A4, week 5 — CUDA 13.0 is the PyPI default and covers sm_120; do not pin cu128,
those binaries were removed. Then `pip freeze > requirements.txt` and commit.

---

## 8. THE DECISION TREE — the paper's focus is not fixed

Three pending analyses each change the paper. Do not draft prose until they are in.

### 8.1 Topography and per-channel map — DECIDES THE MECHANISM

`topography.py` decodes row from each of the 64 channels alone (empirical, no
model assumptions) and from all 64 jointly, converting weights to activation
patterns via the Haufe transform. Quote the **FRONT-5 / OCC-9 ratio**.

| Outcome | The paper becomes |
|---|---|
| **Frontal peak, monotonic posterior falloff, no independent occipital focus** | *Strong version.* Gaze position is decodable from the standard occipital montage; benchmark accuracy conflates cortical and ocular sources. Title: "Gaze position is decodable from occipital EEG in a stimulus-free window of the SSVEP Benchmark dataset." Mechanism settled; Discussion writes itself. **Target: JNE.** |
| **Independent occipital peak** | *Different and arguably more interesting paper.* A frontal dipole cannot produce this — something posterior contributes (anticipatory slow potentials, covert spatial attention). Report honestly; **do not call it an ocular confound.** Title shifts to "Non-SSVEP target information in stimulus-free windows of SSVEP benchmarks." Still JNE, possibly stronger. |
| **Diffuse, no clear structure** | Fall back to the protocol paper: "Stimulus-free-window decoding as a validity control for SSVEP-BCI benchmarks" — reframed per §4.2 as importing an established MVPA control, not inventing one. Moderate but real. |

### 8.2 Occipital µV/deg + sufficiency simulation — DECIDES WHETHER OCULAR IS ENOUGH

Never measured. The plan's own words: this is "the one parameter nobody has
measured." Extend `measure_gradient.py` to the OCC-9 indices, then inject exactly
the measured gradient into surrogate occipital data with noise matched to the
measured residual spectrum, run the identical decoder, and compare simulated to
observed (27.58% row, pre-stimulus).

- **Simulation reproduces the observed accuracy** → ocular volume conduction is
  *sufficient*. Combined with 8.1 frontal peak, mechanism is closed.
- **Simulation lands far below observed** → something else contributes, even if
  the topography looks frontal. This is the quantitative version of the argument
  and much harder to dismiss than a qualitative map.

### 8.3 Mastoid re-reference — COULD REFRAME THE WHOLE PAPER

The recording is **vertex (Cz) referenced**, so ocular potential present at Cz is
subtracted into every channel including occipital. That is a volume-conduction
path invisible to any frontal regressor, and it is the leading benign explanation
for the pattern in §3.8. Re-reference to linked mastoids (M1 = 32, M2 = 42) and
repeat the primary decoding.

- **Effect drops substantially on mastoid reference** → the paper becomes a
  *methods paper about reference choice*: "Vertex referencing injects ocular
  potential into occipital channels in SSVEP benchmarks." Narrower but very
  clean, highly actionable, and an easy sell to **J. Neuroscience Methods** or
  **IEEE TIM**. This is a genuinely good outcome, not a consolation prize.
- **Effect survives** → local occipital pickup, and the original framing holds.

### 8.4 High-pass order fit — REPAIRS THE MECHANISM PARAGRAPH

Per §3.7. Fit 1-pole and 2-pole high-pass step responses to the group time
course; report which fits and the implied corner. Also verify the amplifier's
actual filter order from primary sources. Without this the Discussion's strongest
paragraph is breakable.

### 8.5 Method-family comparison (A4) — THE ML DELIVERABLE

FBCCA, EEGNet, and one deep network (SSVEPformer or Güney DNN), each under four
conditions: raw / 6 Hz high-pass / EOG regression / ICA correction. Spatial
filters and ICA weights fitted on training folds only.

Hypothesis: FBCCA's lowest sub-band starts at 8 Hz and discards the affected band
by accident, while broadband deep networks see it — so deep models should lose
more accuracy under low-frequency removal. If true, that is differential
vulnerability across method families and it connects your confound to numbers
people care about.

**Exclude Ensemble DNN** on documented compute grounds (its authors report ~80 min
per fine-tuned model and ~45 h total for one window length on a GTX 1660). Cite
their published numbers and say why.

### 8.6 BETA replication — DECIDES SCOPE

Identical pipeline, no parameter changes, **wof release only** (§5.2).
- Replicates → "single dataset, could be idiosyncratic" is dead. Two datasets,
  different speller geometries, same effect. Strongest possible version.
- Does not replicate → paper becomes Benchmark-specific. Weaker but still valid,
  and the geometry difference (QWERTY vs 5×8) becomes the discussion point.

### 8.7 Analyses worth adding if weeks 3–4 run ahead

- **Linear scaling in degrees.** Regress decoded vertical position on true
  eccentricity in degrees. A corneo-retinal dipole predicts linearity with a
  slope comparable to the measured µV/deg. Attention has no reason to scale
  linearly with visual angle. Elegant and physiologically specific.
- **Saccade-locked re-epoching.** Detect saccade onset per trial from the frontal
  differential and re-epoch to it rather than to stimulus onset. If the gradient
  time course sharpens, the effect is tied to the eye movement itself.
- **Correlate |gradient| against window-wise accuracy.** The row trend declines
  monotonically (27.58 → 25.86 → 24.87 → 21.60) while the gradient reverses sign;
  a sign-blind linear classifier should track *magnitude*. Report that rather
  than treating the plan's predicted non-monotonicity as a failed prediction.
  Makes a good figure and a real quantitative link.

### 8.8 Do NOT add

TRCA, TDCA, accuracy-chasing, architecture search, hyperparameter tuning,
extra datasets beyond BETA, Ensemble DNN. None answers the research question and
each costs a week. Reviewers do not reject papers for being simple; they reject
them for overclaiming and unresolved confounds.

---

## 9. Statistics — non-negotiable requirements

- **Subject is the unit of analysis, not trial.** The z-scores in §3.8 are
  inflated by pooling 8400 trials; z = 45.74 is not ten times more convincing
  than z = 4.5. Report per-subject accuracies, Wilcoxon signed-rank against the
  permutation null, and bootstrap 95% CIs. `ocular_pilot.py` does not currently
  emit per-subject accuracy — add it; reviewers will ask for the supplement.
- **≥1000 permutations** for anything reported. The p-floor is 1/(n+1).
- **Benjamini–Hochberg FDR at q = 0.05 within declared families.** You currently
  run 48 tests uncorrected. Families: (a) decoding by window and montage,
  (b) gradient measurements, (c) method-family comparisons. At these z values
  almost everything survives, so it costs nothing and closes an objection.
- **Bayes factors for every null claim** (BF₀₁ > 3 as the threshold). The plan
  lists "at chance throughout" as a publishable outcome and several cells now
  report it — a non-significant p cannot support absence of effect.
- **Leakage statement:** normalisers, spatial filters and ICA weights fitted on
  training folds only. In `topography.py` the full-data refit used for the
  activation pattern never touches an accuracy number.
- **≥5 seeds** where stochastic; report variation across seeds and across
  subjects separately. Conflating them is a common and fatal error.
- **Cluster-based permutation** for the time-resolved gradient.
- Effect sizes with bootstrap CIs for every comparison, significant or not.

---

## 10. Schedule — twelve weeks to submission

| Week | Deliverable |
|---|---|
| Sep 8–14 | OSF pre-registration filed. Authorship by email. Systematic literature search logged. Amplifier filter order confirmed. **topography.py run.** |
| Sep 15–21 | `ocular_pilot.py` patch-2 full run. Occipital µV/deg. Mastoid re-reference. |
| Sep 22–28 | Window sweep 0.2–5.0 s. Per-subject stats, Wilcoxon, bootstrap, BH-FDR, Bayes factors. High-pass order fit. |
| Sep 29–Oct 5 | Sufficiency simulation. Screen-space confusion matrices. Figures 1–4 drafted. |
| Oct 6–12 | Baselines under LOSO: FBCCA, EEGNet, one DNN. |
| Oct 13–19 | A4 correction conditions. **AIML course deliverable.** |
| Oct 20–26 | BETA (wof) replication. |
| Oct 27–Nov 2 | Statistics consolidated, figures final. |
| Nov 3–9 | Full first draft. Write in this order: figures → Methods → Results → Discussion → Introduction → Abstract → Title. |
| Nov 10–16 | Adya and Suraj review, revise. |
| Nov 17–23 | Repo runs clean on a fresh machine. Zenodo DOI. arXiv preprint. |
| Nov 24–30 | Cover letter, submit. |

Four weeks of slack against the plan's eight-week estimate.

**Figures — five, no more:**
1. 5×8 layout with visual angles annotated, plus the frequency map showing column–frequency confounding.
2. Time-resolved gradient with the fitted high-pass prediction overlaid and analysis windows shaded. **Best figure.**
3. Decoding accuracy vs window, occipital and frontal, with permutation null bands and per-subject lines.
4. Topography: per-channel accuracy map and activation pattern side by side.
5. Baseline accuracy before and after ocular correction, by method family.

Vector format. Colourblind-safe palettes. State in every caption what the error
bars are. Show per-subject data — bar charts with SEM hide everything that matters.

---

## 11. Venue

**JNE first, IEEE TIM second.** Both Q1, both SCIE-indexed, both count
identically for VIT incentives and for Suraj's promotion file. The prestige gap
is not meaningful to any admissions committee.

- **JNE** — better fit. The BCI readership who must change their protocols read
  it. Higher citation potential precisely because citations come from people who
  need to run your control. Free via the subscription route; gold OA is
  £1,960 / $2,700 — do not choose it.
- **IEEE TIM** — Suraj has published there, CCA-Net appeared there, so the
  editors have handled this literature. That is a real acceptance-probability
  advantage on a first submission, and acceptance probability dominates
  everything else. **Let Suraj make the call.**

Then: J. Neuroscience Methods (free, explicitly a methods venue — and the natural
home if §8.3 reframes the paper), IEEE TNSRE ($2,160 mandatory gold OA, no free
route; India is lower-middle-income so do not assume discount eligibility),
eNeuro / Imaging Neuroscience, Scientific Reports.

**Do not target Nature-family venues.** A single-modality re-analysis of public
EEG will not clear their novelty bar and a desk-reject cycle costs months.
Avoid MDPI and anything promising sub-three-week review.

**One journal at a time.** Simultaneous submission is misconduct under IEEE, IOP,
Elsevier and Springer policy. A preprint is not a submission — post one.

**On "most cited":** nobody can engineer that. There is one structural lever:
papers that impose an obligation on a field — *run this control, report this
number* — accumulate citations by necessity rather than merit. The KU Leuven
paper is that in AAD. This can be that in SSVEP-BCI. It happens by being right,
narrow, and impossible to ignore. Not by being big.

---

## 12. Integrity — three things settled

**Pre-registration.** A full draft exists (`OSF_preregistration.md`). File it at
osf.io: create project → Registrations tab → **OSF Preregistration** template →
when asked about existing data select **"Registration following preliminary
analysis of the data."** That option is honest and necessary: the FBCCA baseline,
gradient measurement and ocular pilot have already been run on all 35 subjects,
so H1–H3 are exploratory, and the confirmatory weight sits on the pending
analyses in §8. No embargo — the public timestamp is the whole point. Put the DOI
in Methods and the cover letter. **Do not backdate anything.** A reviewer who
sees an honest disclosure trusts the rest of the paper; one who catches a
concealed one rejects it and remembers the name.

**Conference papers.** Suraj wants the journal paper first, then 6–7 conference
presentations. **The direction matters.** IEEE policy allows conference papers to
*evolve into* journal papers with sufficient new material — conference first,
journal second. The reverse is prior publication. Thresholds where stated: IEEE
~30% new content, IET 30%, ACM 25%, IEEE PES 60%. Overlap is checked
mechanically by CrossCheck/iThenticate, and ComSoc imposes rejection plus a
minimum six-month submission ban. Seven archival conference papers cannot each
clear a 30% bar against the same journal paper.

What *is* fine, and probably what he means: **non-archival presentation** —
posters, abstract-only tracks, society meetings, institutional symposia. Seven
presentations, seven CV lines, zero exposure. Ask him one narrow question: *"For
the conferences afterwards, are we submitting full papers to proceedings, or
presenting as posters and abstracts?"* Framed as scheduling, not challenge.
Note also that JNE is subscription with a copyright transfer, so reusing figures
in IEEE proceedings later needs IOP's permission too.

Legitimate route to multiple outputs: (1) this confound paper; (2) a separable
methods paper on the EOG-regression control inadequacy; (3) **a real-time
embedded implementation** — decoder on STM32 or Jetson, fixed-point or CMSIS-DSP,
hard latency budget, FreeRTOS or Zephyr, measured jitter and WCET. That third one
is genuine embedded engineering, publishable on its own (IEEE Embedded Systems
Letters or a real-time systems venue), reuses the decoder so marginal cost is
low, and is the only one of the three that actually demonstrates the skills for
automotive/embedded/RTOS graduate admissions. This paper demonstrates statistical
rigour, signal processing and reproducibility — to a TUM, RWTH or CMU committee
it reads as "capable researcher, unrelated area."

**AI use.** Journals require disclosure, not zero use. AI cannot be an author.
Use it for code review, error checking and argument critique — not for generating
submitted text. Keep dated drafts and git history as evidence of authorship.
Detectors are unreliable and biased against non-native English writing, so you
face elevated false positives on your own prose; the defence is provenance, not
rewriting. Never copy method descriptions from papers you reproduce — paraphrase
that follows a source's sentence structure still flags. **Verify every citation
against the actual paper**; a fabricated reference caught in review is
unrecoverable in that submission.

---

## 13. Do these next, in this order

1. **`topography.py`, 5 subjects then 35.** Report the FRONT-5 / OCC-9 ratio and
   where the activation pattern peaks. This is the gate for §8.1 and it decides
   the paper's framing.
2. **File the OSF pre-registration.** Two hours. Must be timestamped before the
   §8 confirmatory analyses carry weight.
3. **Authorship by email**, both of you plus Suraj, dated.
4. **`ocular_pilot.py` patch 2, full run, `--n_perm 1000`.**
5. **Systematic literature search** in Scopus and Web of Science, logged.
6. **Amplifier filter order** — Neuroscan manual or email lbc14@tsinghua.org.cn.
7. **Download BETA (wof)** and verify with the PSD check in §5.2. Not needed
   until week 7, but confirm access now so October holds no surprises.

The single highest-value thing you can do right now is item 1. Everything in the
Discussion depends on how that map looks.
