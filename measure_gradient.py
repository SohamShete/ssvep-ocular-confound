"""
Measure the ocular gradient and its time course on the Benchmark dataset.

Produces the data behind Figures 1-2 of the paper:
  * measured microvolts-per-degree at frontopolar sites, per subject
  * time-resolved gradient showing the decay AND POLARITY REVERSAL predicted
    by the amplifier's 0.15 Hz high-pass (tau = 1.06 s)

Reported values from S15 (single subject, real data), for comparison:
  pre-stimulus HEOG slope  -1.554 uV/deg,  r = -0.992
  time course: -1.514 at -0.08 s -> zero crossing 0.72 s -> +0.547 at 2.00 s

USAGE
    python measure_gradient.py --data_dir "C:\\ssvep\\data"
    python measure_gradient.py --data_dir "C:\\ssvep\\data" --max_subjects 5
"""

import argparse
import os
import re
import sys

import numpy as np
import scipy.io
import scipy.signal
from scipy.stats import pearsonr

FS = 250.0
N_TRIAL, N_TARGETS, N_BLOCKS = 1500, 40, 6
N_PRE = 125
ONSET = N_PRE + 35                      # +140 ms visual latency

# VERIFIED from Freq_Phase.mat: freq = 8.0 + 1.0*col + 0.2*row, row-major 5x8
TGT_ROW = np.repeat(np.arange(5), 8)
TGT_COL = np.tile(np.arange(8), 5)

# Visual angle of each column/row centre. Depends on the acquisition geometry
# (23.6" 1920x1080 at 70 cm, 1510x1037 px matrix, 140 px stimuli).
# CONFIRM these against Wang et al. 2017 / Chen et al. 2015 before publishing.
COL_DEG = np.array([-14.91, -10.77, -6.51, -2.18, 2.18, 6.51, 10.77, 14.91])
ROW_DEG = np.array([-9.89, -4.98, 0.0, 4.98, 9.89])

FP1, FPZ, FP2 = 0, 1, 2
TAU = 1.0 / (2 * np.pi * 0.15)          # 1.0610 s


def find_subjects(d):
    pat = re.compile(r"^S(\d+)\.mat$", re.IGNORECASE)
    got = [(int(m.group(1)), f) for f in os.listdir(d) if (m := pat.match(f))]
    return [f for _, f in sorted(got)]


def load_eog(path, lp_hz):
    """Return baseline-corrected HEOG and VEOG, each (n_time, 40, 6)."""
    data = scipy.io.loadmat(path)["data"]
    if data.shape != (64, N_TRIAL, N_TARGETS, N_BLOCKS):
        sys.exit(f"ERROR: {os.path.basename(path)} shape {data.shape}")
    sos = scipy.signal.butter(4, lp_hz / (FS / 2), btype="lowpass", output="sos")
    x = scipy.signal.sosfiltfilt(sos, data[[FP1, FPZ, FP2]], axis=1)
    x = x - x[:, :10].mean(axis=1, keepdims=True)     # first 40 ms baseline
    return x[0] - x[2], x[1]                          # HEOG, VEOG


def gradient(sig, lo, hi, axis_idx, axis_deg, n_lev):
    """Slope (uV/deg) and Pearson r of mean amplitude against eccentricity."""
    m = [sig[lo:hi][:, axis_idx == k, :].mean() for k in range(n_lev)]
    r, p = pearsonr(axis_deg, m)
    return float(np.polyfit(axis_deg, m, 1)[0]), float(r), float(p), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--lowpass", type=float, default=4.0,
                    help="Below every stimulus frequency (8.0-15.8 Hz)")
    ap.add_argument("--out_csv", default="gradient_results.csv")
    ap.add_argument("--timecourse_csv", default="gradient_timecourse.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: folder not found: {args.data_dir}")
    subs = find_subjects(args.data_dir)
    if args.max_subjects:
        subs = subs[:args.max_subjects]
    if not subs:
        sys.exit("ERROR: no S*.mat files found")

    print("=" * 76)
    print("OCULAR GRADIENT MEASUREMENT")
    print(f"  subjects : {len(subs)}   low-pass : {args.lowpass} Hz   tau : {TAU:.4f} s")
    print("=" * 76)

    # ---------------- per-subject gradients -----------------------------
    rows, tc_rows = [], []
    print(f"\n{'subject':<9} {'HEOG uV/deg':>12} {'r':>8} {'p':>9} | "
          f"{'VEOG uV/deg':>12} {'r':>8} {'p':>9}")
    print("-" * 76)
    for f in subs:
        heog, veog = load_eog(os.path.join(args.data_dir, f), args.lowpass)
        # end of the cue period: saccade complete, stimulus not yet on
        sh, rh, ph, _ = gradient(heog, 80, 125, TGT_COL, COL_DEG, 8)
        sv, rv, pv, _ = gradient(veog, 80, 125, TGT_ROW, ROW_DEG, 5)
        print(f"{f[:-4]:<9} {sh:>12.3f} {rh:>8.3f} {ph:>9.4f} | "
              f"{sv:>12.3f} {rv:>8.3f} {pv:>9.4f}")
        rows.append((f[:-4], sh, rh, ph, sv, rv, pv))

        # ------------- time course (200 ms window, 40 ms step) ----------
        win, step = 50, 10
        for c in range(win // 2, N_TRIAL - win // 2, step):
            s_, r_, _, _ = gradient(heog, c - win // 2, c + win // 2,
                                    TGT_COL, COL_DEG, 8)
            tc_rows.append((f[:-4], (c - N_PRE) / FS, s_, r_))

    a = np.array([r[1] for r in rows])
    b = np.array([r[4] for r in rows])
    print("-" * 76)
    print(f"{'MEAN':<9} {a.mean():>12.3f} {'':>8} {'':>9} | {b.mean():>12.3f}")
    if len(a) > 1:
        print(f"{'SD':<9} {a.std(ddof=1):>12.3f} {'':>8} {'':>9} | {b.std(ddof=1):>12.3f}")
    print("\nThe periorbital EOG literature quotes 14-20 uV/deg. Frontopolar sites")
    print("measure far less. REPORT YOUR MEASURED VALUE, not a derived one.")

    # ---------------- group time course ---------------------------------
    print("\n" + "=" * 76)
    print("TIME-RESOLVED HEOG GRADIENT (group mean)")
    print("tau = 1.06 s predicts decay AND polarity reversal, not simple decay")
    print("=" * 76)
    tc = np.array([(t, s, r) for _, t, s, r in tc_rows])
    times = np.unique(tc[:, 0])
    prev, crossings = None, []
    print(f"{'t from onset (s)':>17} {'slope uV/deg':>13} {'r':>8}")
    for t in times:
        m = tc[np.isclose(tc[:, 0], t)]
        s_, r_ = m[:, 1].mean(), m[:, 2].mean()
        if prev is not None and np.sign(s_) != np.sign(prev):
            crossings.append(t)
        prev = s_
        if abs(t % 0.4) < 0.02 or abs(t) < 0.02:
            bar = "#" * min(int(abs(s_) * 18), 24)
            print(f"{t:>17.2f} {s_:>13.3f} {r_:>8.3f}  {'-' if s_ < 0 else '+'}{bar}")
    print(f"\npolarity reversal at t = {[round(c, 2) for c in crossings[:4]]} s")
    print("If a reversal appears near 0.5-1.0 s, that is the high-pass signature.")
    print("It means long windows are not clean - they are OPPOSITELY contaminated.")

    with open(args.out_csv, "w") as fh:
        fh.write("subject,heog_slope_uv_per_deg,heog_r,heog_p,"
                 "veog_slope_uv_per_deg,veog_r,veog_p\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")
    with open(args.timecourse_csv, "w") as fh:
        fh.write("subject,t_from_onset_s,slope_uv_per_deg,r\n")
        for r in tc_rows:
            fh.write(",".join(str(v) for v in r) + "\n")
    print(f"\nWritten:\n  {os.path.abspath(args.out_csv)}"
          f"\n  {os.path.abspath(args.timecourse_csv)}")


if __name__ == "__main__":
    main()
