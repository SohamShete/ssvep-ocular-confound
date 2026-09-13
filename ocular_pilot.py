"""
Ocular-only decoding pilot, with the HEOG-regression control.

THE QUESTION
    Can target identity be decoded from activity low-passed below 4 Hz - beneath
    the entire 8.0-15.8 Hz stimulus range and every harmonic? If yes, that
    information is not SSVEP.

WHAT DECIDES THE PAPER
    FRONT-5 above chance is near-guaranteed and is NOT a finding; reviewers call
    it trivially known. The OCC-9 number is the headline. And the EOG-REGRESSED
    condition decides the mechanism:

      OCC-9 raw above chance, collapses after regression -> ocular. Confirmed.
      OCC-9 raw above chance, SURVIVES regression        -> NOT ocular. Probably
        anticipatory slow potentials or spatial attention. A different, possibly
        more interesting paper - but you must not call it an ocular confound.

    A reviewer WILL raise the anticipatory-potential alternative. Run this
    before drafting the Discussion.

REFERENCE (S15, single subject, leave-one-block-out, raw condition)
    pre-stimulus  FRONT-5 10.00%  OCC-9 7.50%  (chance 2.5%, z = 6.01 / 6.21)
    0.14-0.64 s   FRONT-5  5.42%  OCC-9 3.75%  (z = 5.67 / 2.46)
    0.14-1.14 s                   OCC-9 3.33%  (z = 1.55, n.s.)

USAGE
    pip install scikit-learn
    python ocular_pilot.py --data_dir "C:\\ssvep\\data" --max_subjects 5
    python ocular_pilot.py --data_dir "C:\\ssvep\\data" --n_perm 200
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import scipy.io
import scipy.signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut

FS = 250.0
N_TRIAL, N_TARGETS, N_BLOCKS = 1500, 40, 6
N_PRE = 125
LATENCY_S = 0.14
ONSET = N_PRE + int(round(LATENCY_S * FS))          # sample 160

FRONT_IDX = np.array([0, 1, 2, 3, 4])                     # Fp1 Fpz Fp2 AF3 AF4
OCC_IDX = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62])  # PZ..O2
FP1, FPZ, FP2 = 0, 1, 2

# VERIFIED from Freq_Phase.mat: freq = 8.0 + 1.0*col + 0.2*row, row-major 5x8.
#
# CRITICAL CONSEQUENCE (measured, not assumed):
#   column vs frequency  r = +0.992  -> CONFOUNDED. Column decoding CANNOT
#                                       distinguish gaze from frequency and is
#                                       not evidence of gaze.
#   row    vs frequency  r = +0.123  -> USABLE. The only spatial axis separable
#                                       from frequency.
N_ROWS, N_COLS = 5, 8
TGT_ROW = np.repeat(np.arange(N_ROWS), N_COLS)
TGT_COL = np.tile(np.arange(N_COLS), N_ROWS)

WINDOWS = {
    "pre-stim_-0.50_to_0.00s": (0, N_PRE),
    "window_0.14_to_0.64s": (ONSET, ONSET + 125),
    "window_0.14_to_1.14s": (ONSET, ONSET + 250),
    "late_1.14_to_2.64s": (ONSET + 250, ONSET + 625),
}


def find_subjects(d):
    pat = re.compile(r"^S(\d+)\.mat$", re.IGNORECASE)
    got = [(int(m.group(1)), f) for f in os.listdir(d) if (m := pat.match(f))]
    return [f for _, f in sorted(got)]


def regress_out_eog(x, eog):
    """Remove EOG-correlated variance from every channel.

    x   : (n_chan, n_time, n_trials)
    eog : (n_eog,  n_time, n_trials)

    Coefficients are fitted per subject over all timepoints and trials. This is
    UNSUPERVISED - no labels are used - so it cannot leak. Limitation to state
    in the paper: the surrogates are built from frontopolar EEG, so genuine
    frontal brain activity correlated with occipital activity is also removed.
    That makes the control CONSERVATIVE - it can over-correct, not under-correct.
    """
    nc, nt, ntr = x.shape
    X = x.reshape(nc, -1).T                       # (samples, n_chan)
    E = eog.reshape(eog.shape[0], -1).T           # (samples, n_eog)
    E = np.column_stack([E, np.ones(len(E))])     # + intercept
    beta, *_ = np.linalg.lstsq(E, X, rcond=None)
    return (X - E @ beta).T.reshape(nc, nt, ntr)


def load_subject(path, lp_hz, regress):
    """Return {channel_set: (n_chan, n_time, n_trials)}, labels, block ids."""
    data = scipy.io.loadmat(path)["data"]
    if data.shape != (64, N_TRIAL, N_TARGETS, N_BLOCKS):
        sys.exit(f"ERROR: {os.path.basename(path)} shape {data.shape}, "
                 f"expected (64, 1500, 40, 6)")

    sos = scipy.signal.butter(4, lp_hz / (FS / 2), btype="lowpass", output="sos")
    filt = scipy.signal.sosfiltfilt(sos, data, axis=1)
    # (n_chan, n_time, 240); C-order reshape -> target = idx // N_BLOCKS
    flat = filt.reshape(64, N_TRIAL, -1)

    if regress:
        eog = np.stack([flat[FP1] - flat[FP2], flat[FPZ]])
        occ = regress_out_eog(flat[OCC_IDX], eog)
        front = regress_out_eog(flat[FRONT_IDX], eog)
    else:
        occ, front = flat[OCC_IDX], flat[FRONT_IDX]

    labels = np.arange(flat.shape[2]) // N_BLOCKS
    blocks = np.arange(flat.shape[2]) % N_BLOCKS
    return {"FRONT-5": front, "OCC-9": occ}, labels, blocks


def window_features(sig, lo, hi, decim=20):
    seg = sig[:, lo:hi, :][:, ::decim, :]
    return seg.transpose(2, 0, 1).reshape(seg.shape[2], -1).astype(np.float64)


def decode(F, y, groups, rng=None):
    """Leave-one-group-out. Normaliser fitted on training folds only."""
    yy = y.copy()
    if rng is not None:
        for g in np.unique(groups):
            m = groups == g
            yy[m] = rng.permutation(yy[m])
    pred = np.zeros_like(yy)
    for tr, te in LeaveOneGroupOut().split(F, yy, groups):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit((F[tr] - mu) / sd, yy[tr])
        pred[te] = clf.predict((F[te] - mu) / sd)
    return pred, yy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_subjects", type=int, default=5)
    ap.add_argument("--lowpass", type=float, default=4.0)
    ap.add_argument("--n_perm", type=int, default=200,
                    help="Permutations for the null. >=200 for anything reported.")
    ap.add_argument("--skip_regression", action="store_true")
    ap.add_argument("--decim", type=int, default=20,
                    help="Temporal downsampling before the classifier. The data "
                         "are low-passed at 4 Hz, so Nyquist needs only 8 Hz; "
                         "decim=20 gives 12.5 Hz and loses nothing, while "
                         "quartering the runtime versus decim=10.")
    ap.add_argument("--out_csv", default="ocular_pilot_results.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: folder not found: {args.data_dir}")
    subs = find_subjects(args.data_dir)[:args.max_subjects]
    if not subs:
        sys.exit("ERROR: no S*.mat files found")
    if args.n_perm < 200:
        print(f"WARNING: n_perm={args.n_perm}. The permutation p cannot fall below "
              f"1/(n+1) = {1 / (args.n_perm + 1):.4f}. Use >=200 for reported "
              f"results.\n")

    grouping = "leave-one-subject-out" if len(subs) > 1 else "leave-one-block-out"
    conditions = ["raw"] if args.skip_regression else ["raw", "EOG-regressed"]

    print("=" * 78)
    print("OCULAR-ONLY DECODING PILOT")
    print(f"  subjects  : {len(subs)}  ({', '.join(s[:-4] for s in subs)})")
    print(f"  low-pass  : {args.lowpass} Hz   (all stimuli are 8.0-15.8 Hz)")
    print(f"  conditions: {', '.join(conditions)}")
    print(f"  chance    : 40-class 2.50% | row 20.00% | column 12.50%")
    print(f"  CV        : {grouping}")
    print("=" * 78)

    rows = []
    for cond in conditions:
        cache = {}
        for si, f in enumerate(subs):
            sets, y, blk = load_subject(os.path.join(args.data_dir, f),
                                        args.lowpass, cond != "raw")
            cache[si] = (sets, y, blk)

        print(f"\n{'#' * 24} CONDITION: {cond} {'#' * 24}")
        for wname, (lo, hi) in WINDOWS.items():
            if hi > N_TRIAL:
                continue
            print(f"\n  {wname}")
            for cname in ("FRONT-5", "OCC-9"):
                t0 = time.time()
                Fs, ys, gs = [], [], []
                for si in cache:
                    sets, y, blk = cache[si]
                    Fs.append(window_features(sets[cname], lo, hi, args.decim))
                    ys.append(y)
                    gs.append(np.full(len(y), si) if len(subs) > 1 else blk)
                F, y, g = np.vstack(Fs), np.concatenate(ys), np.concatenate(gs)

                pred, yt = decode(F, y, g)
                acc = (pred == yt).mean()
                rowa = (TGT_ROW[pred] == TGT_ROW[yt]).mean()
                cola = (TGT_COL[pred] == TGT_COL[yt]).mean()

                # One permutation loop, three nulls. row_acc is the axis the
                # dissociation claim rests on (row vs freq r = 0.12), so it
                # needs its own null -- the 40-class null does not transfer.
                null, null_row, null_col = [], [], []
                for i in range(args.n_perm):
                    pp, tt = decode(F, y, g, np.random.default_rng(1000 + i))
                    null.append((pp == tt).mean())
                    null_row.append((TGT_ROW[pp] == TGT_ROW[tt]).mean())
                    null_col.append((TGT_COL[pp] == TGT_COL[tt]).mean())
                null = np.array(null)
                null_row = np.array(null_row)
                null_col = np.array(null_col)

                floor = 1.0 / (args.n_perm + 1)

                def assess(obs, nul):
                    zz = (obs - nul.mean()) / (nul.std(ddof=1) + 1e-12)
                    pp_ = (np.sum(nul >= obs) + 1) / (args.n_perm + 1)
                    if pp_ <= floor + 1e-12:
                        return zz, pp_, (zz > 3.0), f"p<={floor:.4f}"
                    return zz, pp_, (pp_ <= 0.05), f"p={pp_:.4f}"

                z, p_emp, sig, pstr = assess(acc, null)
                z_row, p_row, sig_row, pstr_row = assess(rowa, null_row)
                z_col, p_col, _, _ = assess(cola, null_col)

                verdict = "ABOVE CHANCE" if (sig and acc > null.mean()) else "at chance"
                verdict_row = "ABOVE CHANCE" if (sig_row and rowa > null_row.mean()) else "at chance"

                print(f"    {cname:<8} 40-cls {acc * 100:6.2f}% (null {null.mean() * 100:5.2f}%, "
                      f"z={z:6.2f}, {pstr}) {verdict}")
                print(f"    {'':<8} row     {rowa * 100:6.2f}% (null {null_row.mean() * 100:5.2f}%, "
                      f"z={z_row:6.2f}, {pstr_row}) {verdict_row}")
                print(f"    {'':<8} col     {cola * 100:6.2f}% (null {null_col.mean() * 100:5.2f}%, "
                      f"z={z_col:6.2f}) -- confounded with frequency DURING "
                      f"stimulation only; clean in the pre-stimulus window")
                print(f"    {'':<8} ({time.time() - t0:.0f}s)")
                rows.append((cond, wname, cname, acc, rowa, cola,
                             null.mean(), z, p_emp,
                             null_row.mean(), z_row, p_row,
                             null_col.mean(), z_col, p_col))

    with open(args.out_csv, "w") as fh:
        fh.write("condition,window,channels,acc,row_acc,col_acc,"
                 "null_mean,z,p_emp,"
                 "null_row_mean,z_row,p_row,"
                 "null_col_mean,z_col,p_col\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")
    print(f"\nWritten to {os.path.abspath(args.out_csv)}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  FRONT-5 above chance : expected. NOT a finding on its own.")
    print("  OCC-9   above chance : THE headline. The standard pipeline carries")
    print("                         non-SSVEP target information.")
    print()
    print("  RAW vs EOG-REGRESSED decides the mechanism:")
    print("    OCC-9 collapses after regression -> ocular. Confirmed.")
    print("    OCC-9 survives regression        -> NOT ocular. Likely anticipatory")
    print("      slow potentials or spatial attention. Report honestly; do not")
    print("      call it an ocular confound.")
    print()
    print("  col_acc is CONFOUNDED: column correlates with stimulus frequency at")
    print("  r = 0.992, so high column accuracy is expected from frequency alone.")
    print("  row_acc (row vs freq r = 0.12) is the interpretable spatial result.")
    print()
    print("  Window trend: the 0.15 Hz high-pass (tau = 1.06 s) makes the ocular")
    print("  offset decay AND REVERSE SIGN. In S15 the HEOG gradient ran -1.51")
    print("  uV/deg at -0.08 s, crossed zero at 0.72 s, and reached +0.55 at")
    print("  2.00 s. Expect a non-monotonic trend, not a simple decline.")


if __name__ == "__main__":
    main()
