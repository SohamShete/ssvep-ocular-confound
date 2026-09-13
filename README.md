# SSVEP Ocular Confound Study

Empirical evaluation of ocular artifact contributions to Steady-State Visual Evoked Potential (SSVEP) signals across 35 subjects.

## Summary Results

* **Frontal EOG Variance Explained ($R^2$):** $36.03\% \pm 13.78\%$ (Range: 18.92% – 68.20%)
* **Raw SSVEP Decoding Accuracy:** 16.94%
* **Residual SSVEP Decoding Accuracy:** 17.06%
* **Paired Comparison:** $t(34) = -0.3805$, $p = 0.7060$ (No significant degradation after EOG subtraction)

## Repository Structure

* `eog_7regressor.py` — Fits 7-regressor frontal EOG model (`Fp1`, `Fpz`, `Fp2`, `AF3`, `AF4`, `F7`, `F8`).
* `eog_residual_fbcca.py` — Calculates residual variance ratios per subject.
* `evaluate_residual_decoding.py` — Evaluates FBCCA target decoding accuracy on raw vs. residual signals.
* `OSF_preregistration.md` — Formal preregistration document.
