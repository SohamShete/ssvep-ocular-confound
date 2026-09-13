"""
FBCCA baseline on the Tsinghua Benchmark SSVEP dataset.

WHAT THIS DOES
    Calibration-free SSVEP classification. It looks at each 40-class trial and
    guesses which flicker frequency the subject was staring at, without ever
    training on that subject.

WHY YOU RUN IT FIRST
    It is a correctness check, not a contribution. Published FBCCA lands around
    60-65% accuracy at a 1.0 s window on 40 classes. If your number is far off,
    your preprocessing is wrong and every later result is noise. Chance is 2.5%.

QUICK TEST (do this first, ~1 minute):
    python fbcca_baseline.py --data_dir "C:\\ssvep\\data" --max_subjects 2 --windows 1.0

FULL RUN (~5-20 minutes for 35 subjects):
    python fbcca_baseline.py --data_dir "C:\\ssvep\\data"

Reference: Chen et al., "Filter bank canonical correlation analysis for
implementing a high-speed SSVEP-based BCI", J. Neural Eng. 12(4), 2015.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import scipy.io
import scipy.signal


# ---------------------------------------------------------------------------
# Dataset constants (Benchmark / Wang et al. 2017)
# Verify these against the dataset Readme before trusting any result.
# ---------------------------------------------------------------------------
FS = 250.0                # sampling rate after the provider's downsampling
N_SAMPLES_TRIAL = 1500    # 6.0 s per trial
N_PRE_SAMPLES = 125       # 0.5 s cue period before stimulus onset
VISUAL_LATENCY_S = 0.14   # ~140 ms, per the dataset paper
N_BLOCKS = 6
N_TARGETS = 40

# Standard 9-channel parieto-occipital montage:
# PZ, PO5, PO3, POz, PO4, PO6, O1, Oz, O2  (ZERO-BASED indices)
# Cross-check these against your 64-channels.loc file BY NAME before trusting.
OCC_IDX = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])
OCC_NAMES = ["PZ", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]

# FBCCA hyperparameters (Chen et al. 2015)
N_SUBBANDS = 5
N_HARMONICS = 5
FB_HIGHCUT = 90.0         # Hz; below Nyquist (125 Hz)
FB_A, FB_B = 1.25, 0.25   # sub-band weights w(n) = n^-a + b


def build_filter_bank():
    """Chebyshev Type I bandpass, one per sub-band. Sub-band m starts at 8*m Hz."""
    nyq = FS / 2.0
    return [
        scipy.signal.cheby1(N=4, rp=0.5,
                            Wn=[(8.0 * m) / nyq, FB_HIGHCUT / nyq],
                            btype="bandpass", output="sos")
        for m in range(1, N_SUBBANDS + 1)
    ]


def make_references(freqs, n_samples):
    """Sine-cosine templates, shape (n_targets, 2*Nh, n_samples).

    No stimulus phase is used: FBCCA is phase-blind. That is why it discards
    the phase information JFPM encodes, and why template methods beat it.
    """
    t = np.arange(n_samples) / FS
    refs = np.zeros((len(freqs), 2 * N_HARMONICS, n_samples))
    for i, f in enumerate(freqs):
        for h in range(1, N_HARMONICS + 1):
            refs[i, 2 * (h - 1)] = np.sin(2 * np.pi * h * f * t)
            refs[i, 2 * (h - 1) + 1] = np.cos(2 * np.pi * h * f * t)
    return refs


def prepare_reference_qs(refs):
    """Pre-compute the orthonormal basis of each reference template once."""
    out = []
    for k in range(refs.shape[0]):
        Y = refs[k].T
        Y = Y - Y.mean(axis=0, keepdims=True)
        Q, _ = np.linalg.qr(Y)
        out.append(Q)
    return out


def max_canonical_corr_q(X, Qy):
    """Largest canonical correlation between X and a pre-orthonormalised Qy."""
    X = X - X.mean(axis=0, keepdims=True)
    Qx, Rx = np.linalg.qr(X)
    d = np.abs(np.diag(Rx))
    if d.size == 0 or d.max() == 0:
        return 0.0
    keep = int((d > 1e-10 * d.max()).sum())
    if keep == 0:
        return 0.0
    s = np.linalg.svd(Qx[:, :keep].T @ Qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0))


def itr_bpm(accuracy, n_targets, total_time_s):
    """Wolpaw ITR in bits/min. total_time_s includes gaze-shift time."""
    P = accuracy
    if P <= 1.0 / n_targets:
        return 0.0
    if P >= 1.0:
        bits = np.log2(n_targets)
    else:
        bits = (np.log2(n_targets) + P * np.log2(P)
                + (1 - P) * np.log2((1 - P) / (n_targets - 1)))
    return bits * 60.0 / total_time_s


def find_subject_files(data_dir):
    """Return S*.mat files sorted numerically. Ignores anything else."""
    pat = re.compile(r"^S(\d+)\.mat$", re.IGNORECASE)
    found = []
    for f in os.listdir(data_dir):
        m = pat.match(f)
        if m:
            found.append((int(m.group(1)), f))
    found.sort()
    return [f for _, f in found]


def main():
    ap = argparse.ArgumentParser(
        description="FBCCA baseline on the Benchmark SSVEP dataset.")
    ap.add_argument("--data_dir", required=True,
                    help='Folder with S*.mat and Freq_Phase.mat, e.g. "C:\\ssvep\\data"')
    ap.add_argument("--windows", type=float, nargs="+", default=[0.5, 0.7, 1.0],
                    help="Signal lengths in seconds (default: 0.5 0.7 1.0)")
    ap.add_argument("--max_subjects", type=int, default=None,
                    help="Only process the first N subjects. Use 2 for a quick test.")
    ap.add_argument("--gaze_shift", type=float, default=0.5,
                    help="Gaze-shift time added to the window for ITR (default 0.5)")
    ap.add_argument("--out_csv", default="fbcca_results.csv")
    args = ap.parse_args()

    # ---- checks with human-readable errors -------------------------------
    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: folder not found: {args.data_dir}\n"
                 f"Check the path. Use quotes around it if it contains spaces.")

    freq_path = os.path.join(args.data_dir, "Freq_Phase.mat")
    if not os.path.isfile(freq_path):
        sys.exit(f"ERROR: Freq_Phase.mat not found in {args.data_dir}\n"
                 f"It must sit in the SAME folder as the S*.mat files.")

    subject_files = find_subject_files(args.data_dir)
    if not subject_files:
        sys.exit(f"ERROR: no S*.mat files found in {args.data_dir}\n"
                 f"Expected files named S1.mat, S2.mat, ...")
    if args.max_subjects:
        subject_files = subject_files[:args.max_subjects]

    freqs = scipy.io.loadmat(freq_path)["freqs"].flatten()
    if len(freqs) != N_TARGETS:
        sys.exit(f"ERROR: expected {N_TARGETS} frequencies, found {len(freqs)}")

    print("=" * 66)
    print("FBCCA baseline")
    print(f"  data folder : {args.data_dir}")
    print(f"  subjects    : {len(subject_files)}  ({subject_files[0]} ... {subject_files[-1]})")
    print(f"  windows     : {args.windows} s")
    print(f"  channels    : {', '.join(OCC_NAMES)}")
    print(f"  freq range  : {freqs.min():.1f} - {freqs.max():.1f} Hz, {N_TARGETS} targets")
    print(f"  chance level: {100.0 / N_TARGETS:.1f}%")
    print("=" * 66)

    bank = build_filter_bank()
    weights = np.array([(n ** -FB_A) + FB_B for n in range(1, N_SUBBANDS + 1)])
    onset = N_PRE_SAMPLES + int(round(VISUAL_LATENCY_S * FS))  # sample 160

    rows = []
    for win_s in args.windows:
        n_samples = int(round(win_s * FS))
        if onset + n_samples > N_SAMPLES_TRIAL:
            print(f"  skipping {win_s}s: window runs past end of trial")
            continue

        refs = make_references(freqs, n_samples)
        ref_qs = prepare_reference_qs(refs)

        print(f"\n--- window {win_s:.1f} s  (samples {onset}:{onset + n_samples}) ---")
        per_subject = []
        t_start = time.time()

        for si, fname in enumerate(subject_files):
            t0 = time.time()
            try:
                data = scipy.io.loadmat(os.path.join(args.data_dir, fname))["data"]
            except Exception as e:
                sys.exit(f"ERROR reading {fname}: {e}\n"
                         f"The file may be incomplete. Re-download it.")

            if data.shape != (64, N_SAMPLES_TRIAL, N_TARGETS, N_BLOCKS):
                sys.exit(f"ERROR: {fname} has shape {data.shape}, "
                         f"expected (64, 1500, 40, 6)")

            # Select channels, flatten trials: (9, 1500, 240)
            occ = data[OCC_IDX].reshape(len(OCC_IDX), N_SAMPLES_TRIAL, -1)

            # Cut the window FIRST, then filter, to prevent data leakage
            window_occ = occ[:, onset:onset + n_samples, :]
            filtered = [
                scipy.signal.sosfiltfilt(sos, window_occ, axis=1)
                for sos in bank
            ]

            correct = 0
            n_trials = N_TARGETS * N_BLOCKS
            for tr in range(n_trials):
                # occ was reshaped from (..., 40 targets, 6 blocks) in C order,
                # so flat index tr maps to target = tr // N_BLOCKS.
                target = tr // N_BLOCKS
                scores = np.zeros(N_TARGETS)
                Xs = [f[:, :, tr].T for f in filtered]   # (n_samples, 9) each
                for k in range(N_TARGETS):
                    acc = 0.0
                    for m in range(N_SUBBANDS):
                        rho = max_canonical_corr_q(Xs[m], ref_qs[k])
                        acc += weights[m] * rho * rho
                    scores[k] = acc
                correct += int(np.argmax(scores) == target)

            acc = correct / n_trials
            per_subject.append(acc)
            rows.append((win_s, fname[:-4], acc))

            elapsed = time.time() - t0
            done, total = si + 1, len(subject_files)
            eta = (time.time() - t_start) / done * (total - done)
            print(f"  [{done:2d}/{total}] {fname:<9} acc = {acc * 100:6.2f}%   "
                  f"({elapsed:5.1f}s, ETA {eta / 60:4.1f} min)")

        accs = np.array(per_subject)
        mean = accs.mean()
        sem = accs.std(ddof=1) / np.sqrt(len(accs)) if len(accs) > 1 else 0.0
        print(f"  MEAN {mean * 100:.2f} +/- {sem * 100:.2f}% (SEM)   "
              f"range {accs.min() * 100:.1f}-{accs.max() * 100:.1f}%   "
              f"ITR {itr_bpm(mean, N_TARGETS, win_s + args.gaze_shift):.1f} bpm")

        if abs(win_s - 1.0) < 1e-9:
            if 0.55 <= mean <= 0.72:
                print("  --> In the expected range for FBCCA at 1.0 s. Pipeline looks sound.")
            elif mean < 0.10:
                print("  --> NEAR CHANCE. Something is fundamentally wrong: check "
                      "channel indices, epoch window, and that data is not shuffled.")
            else:
                print("  --> OUTSIDE the expected 60-65% range. Investigate before "
                      "proceeding. Do not build on this.")

    out_path = os.path.abspath(args.out_csv)
    with open(out_path, "w") as fh:
        fh.write("window_s,subject,accuracy\n")
        for w, s, a in rows:
            fh.write(f"{w},{s},{a:.6f}\n")
    print(f"\nPer-subject results written to:\n  {out_path}")


if __name__ == "__main__":
    main()
