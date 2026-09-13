"""
Where on the scalp does the sub-4 Hz target information live?

v3 CHANGES (over the patched v2 that produced topo_smoke_*):
  * --reference {vertex,average,mastoid,all}. The recording is VERTEX (Cz)
    referenced, so Cz is identically zero and any distant source grows with
    distance from Cz. In the n=5 smoke test the four most peripheral electrodes
    (CB2, CB1, M1, M2) were the four best channels and accuracy correlated with
    distance from Cz at r = +0.51. A Cz-referenced map therefore cannot answer
    the anterior-vs-posterior question in Section 8.1 on its own.
    Data are loaded ONCE; the reference transform is applied on the fly.
  * Reports accuracy vs distance-from-Cz (Pearson + Spearman) per window per
    reference. If the radial structure is reference-driven it collapses under
    average reference. That is the diagnostic.
  * Reports M1/M2/CB1/CB2 explicitly. If the mastoids carry target information,
    the linked-mastoid control (Section 8.3, prereg Section 4.4) subtracts a
    signal-carrying reference into every channel and is not interpretable.
  * FRONT-5 / OCC-9 ratio reported as EXCESS OVER CHANCE. Accuracy is floored
    at chance, so a raw ratio compresses the difference: on synthetic data with
    a purely frontal generator, raw 1.66x vs excess 3.97x.
  * Per-subject accuracies written to .npy (subject is the unit of analysis).
  * Electrode radii normalised so the montage fills the head outline.
  * float32 cache: 35 subjects = 3.2 GB rather than 6.4 GB.

NO permutation null here, deliberately. Section 3.8 already established with 200
permutations that the row null lands at 19.95-20.14%, i.e. exactly theoretical
chance, under this balanced LOSO design. State that in Methods and cite it.

WHAT IS DECODED
    Row (vertical position, 5 classes, chance 20%). Row is the only spatial axis
    separable from stimulus frequency (r = 0.123 vs 0.992 for column). In the
    pre-stimulus window no stimulus exists, so column is clean there too and is
    reported; during stimulation column is confounded and is NOT interpretable.

USAGE
    python topography.py --data_dir "C:\\ssvep\\data" --loc_file "C:\\ssvep\\data\\64-channels.loc" --reference all
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import scipy.io
import scipy.signal
from scipy.stats import pearsonr, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut

FS = 250.0
N_TRIAL, N_TARGETS, N_BLOCKS, N_CHAN = 1500, 40, 6, 64
N_PRE = 125
ONSET = N_PRE + 35                       # sample 160, after ~140 ms latency

M1_IDX, M2_IDX = 32, 42                  # verified by name in verify_data.py

TGT_ROW = np.repeat(np.arange(5), 8)
TGT_COL = np.tile(np.arange(8), 5)

WINDOWS = {
    "pre-stim_-0.50_to_0.00s": (0, N_PRE),
    "window_0.14_to_0.64s": (ONSET, ONSET + 125),
}

OCC = {"PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"}
FRONT = {"FP1", "FPZ", "FP2", "AF3", "AF4"}
RIM = ["M1", "M2", "CB1", "CB2"]


def find_subjects(d):
    pat = re.compile(r"^S(\d+)\.mat$", re.IGNORECASE)
    got = [(int(m.group(1)), f) for f in os.listdir(d) if (m := pat.match(f))]
    return [f for _, f in sorted(got)]


def read_locations(loc_path):
    """EEGLAB .loc -> labels and 2D plotting coordinates.

    Format: index, theta (deg), radius, label. Theta 0 is the nose, negative is
    the left hemisphere. Nose-up plot: x = r*sin(theta), y = r*cos(theta).
    Radii are rescaled so the outermost electrode sits at 0.88 of the unit head
    circle; the raw .loc radii use a convention where the outer ring is ~0.5 and
    the montage would otherwise plot as a small blob inside an empty head.
    """
    labels, xs, ys = {}, {}, {}
    with open(loc_path, "r", errors="ignore") as fh:
        for line in fh:
            p = line.replace("`", "").split()
            if len(p) >= 4 and p[0].lstrip("-").isdigit():
                i = int(p[0]) - 1                     # file is 1-based
                th = np.radians(float(p[1]))
                r = float(p[2])
                labels[i] = p[-1]
                xs[i] = r * np.sin(th)
                ys[i] = r * np.cos(th)
    if len(labels) < N_CHAN:
        sys.exit(f"ERROR: parsed only {len(labels)} channels from {loc_path}, "
                 f"expected {N_CHAN}. Check the file.")
    order = sorted(labels)
    X = np.array([xs[i] for i in order])
    Y = np.array([ys[i] for i in order])
    rmax = np.hypot(X, Y).max()
    if rmax > 0:
        X, Y = X * (0.88 / rmax), Y * (0.88 / rmax)
    return [labels[i] for i in order], X, Y


def load_subject(path, lp_hz=4.0):
    """All 64 channels, low-passed, flattened to (64, n_time, 240), float32.

    Filtering is applied to the FULL epoch before slicing, matching the
    pre-registration and ocular_pilot.py. Zero-phase filtering spreads energy
    backwards, so the last ~2 decimated samples of the pre-stimulus window carry
    some post-onset low-frequency content (measured: a step at sample 125
    contributes 2.6% of its amplitude to the pre-stim window mean, but 35-48% at
    samples 120-124). SSVEP itself cannot leak: 8 Hz is attenuated 48 dB by the
    filtfilt pass. Disclose this and run the robustness check with the pre-stim
    window ending at sample 100.
    """
    data = scipy.io.loadmat(path)["data"]
    if data.shape != (N_CHAN, N_TRIAL, N_TARGETS, N_BLOCKS):
        sys.exit(f"ERROR: {os.path.basename(path)} shape {data.shape}, "
                 f"expected ({N_CHAN}, {N_TRIAL}, {N_TARGETS}, {N_BLOCKS})")
    sos = scipy.signal.butter(4, lp_hz / (FS / 2), btype="lowpass", output="sos")
    filt = scipy.signal.sosfiltfilt(sos, data, axis=1)
    return filt.reshape(N_CHAN, N_TRIAL, -1).astype(np.float32)


def reference_signal(sig, mode):
    """Return the (n_time, n_trials) signal to subtract from every channel.

    vertex  : nothing subtracted; the data are already Cz-referenced.
    average : mean across all 64 channels (includes M1, M2, CB1, CB2 - stated).
    mastoid : mean of M1 and M2 (linked mastoids, prereg Section 4.4).
    """
    if mode == "vertex":
        return None
    if mode == "average":
        return sig.mean(axis=0)
    if mode == "mastoid":
        return sig[[M1_IDX, M2_IDX]].mean(axis=0)
    raise ValueError(mode)


def decode(F, y, groups, return_model=False):
    """Leave-one-subject-out. Normaliser fitted on training folds only.

    If return_model, also refit on all data to get weights and covariance for
    the activation-pattern transform. That refit is used ONLY for the pattern,
    never for an accuracy number, so it cannot leak into any reported score.
    """
    pred = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(F, y, groups):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit((F[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((F[te] - mu) / sd)
    if not return_model:
        return pred, None
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = (F - mu) / sd
    full = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, y)
    # Haufe transform: A = Cov(X) W. Weights are not interpretable; patterns are.
    W = full.coef_.T                                  # (n_feat, n_class)
    A = np.cov(Z, rowvar=False) @ W                   # (n_feat, n_class)
    return pred, A


def topoplot(ax, vals, x, y, labels, title, cmap, highlight=()):
    """Interpolated scalp map. Head outline drawn as a unit circle."""
    from scipy.interpolate import griddata
    n = 200
    gx, gy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    gz = griddata((x, y), vals, (gx, gy), method="cubic")
    gz[gx ** 2 + gy ** 2 > 1.0] = np.nan              # clip to the head
    im = ax.contourf(gx, gy, gz, levels=25, cmap=cmap)
    ax.contour(gx, gy, gz, levels=8, colors="k", linewidths=0.3, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1.5)     # head
    ax.plot([-0.09, 0, 0.09], [0.995, 1.11, 0.995], "k-", lw=1.5)   # nose
    ax.scatter(x, y, c="k", s=5, zorder=3)
    for i, lab in enumerate(labels):
        if lab.upper() in highlight:
            ax.scatter(x[i], y[i], facecolors="none", edgecolors="w",
                       s=70, lw=1.6, zorder=4)
            ax.annotate(lab, (x[i], y[i]), fontsize=6, color="w",
                        xytext=(2, 3), textcoords="offset points", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--loc_file", required=True,
                    help="64-channels.loc, needed for electrode positions")
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--lowpass", type=float, default=4.0)
    ap.add_argument("--decim", type=int, default=20,
                    help="12.5 Hz after a 4 Hz low-pass; Nyquist needs only 8")
    ap.add_argument("--reference", default="all",
                    choices=["vertex", "average", "mastoid", "all"])
    ap.add_argument("--out_prefix", default="topography")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: folder not found: {args.data_dir}")
    if not os.path.isfile(args.loc_file):
        sys.exit(f"ERROR: loc file not found: {args.loc_file}")

    labels, px, py = read_locations(args.loc_file)
    up = [l.upper() for l in labels]
    for want, idx in [("M1", M1_IDX), ("M2", M2_IDX)]:
        if up[idx] != want:
            sys.exit(f"ERROR: index {idx} is {up[idx]}, expected {want}. "
                     f"The mastoid reference would be wrong. Fix the indices.")
    if "CZ" not in up:
        sys.exit("ERROR: no CZ in the montage; the distance diagnostic needs it.")
    cz = up.index("CZ")
    dist_cz = np.hypot(px - px[cz], py - py[cz])

    subs = find_subjects(args.data_dir)
    if args.max_subjects:
        subs = subs[:args.max_subjects]
    if len(subs) < 3:
        sys.exit(f"ERROR: need >= 3 subjects for LOSO, found {len(subs)}")

    refs = ["vertex", "average", "mastoid"] if args.reference == "all" \
        else [args.reference]

    print("=" * 78)
    print("SCALP TOPOGRAPHY OF SUB-4 Hz TARGET INFORMATION")
    print(f"  subjects   : {len(subs)}   low-pass : {args.lowpass} Hz")
    print(f"  references : {', '.join(refs)}")
    print(f"  chance     : row 20.00%   column 12.50%")
    print("=" * 78)

    print("\nloading subjects (once; references applied on the fly)", end="",
          flush=True)
    t_load = time.time()
    cache = []
    for f in subs:
        cache.append(load_subject(os.path.join(args.data_dir, f), args.lowpass))
        print(".", end="", flush=True)
    print(f" done ({time.time() - t_load:.0f}s, "
          f"{sum(s.nbytes for s in cache) / 1e9:.2f} GB resident)")

    y_row = np.concatenate([TGT_ROW[np.arange(240) // N_BLOCKS]] * len(subs))
    y_col = np.concatenate([TGT_COL[np.arange(240) // N_BLOCKS]] * len(subs))
    groups = np.concatenate([np.full(240, i) for i in range(len(subs))])
    subs_idx = np.unique(groups)

    rows_out = []
    for ref in refs:
        refsig = [reference_signal(s, ref) for s in cache]

        def chan(si, ch, lo, hi):
            seg = cache[si][ch, lo:hi, :]
            if refsig[si] is not None:
                seg = seg - refsig[si][lo:hi, :]
            return seg

        def allchan(si, lo, hi):
            seg = cache[si][:, lo:hi, :]
            if refsig[si] is not None:
                seg = seg - refsig[si][None, lo:hi, :]
            return seg

        print("\n" + "=" * 78)
        print(f"REFERENCE: {ref.upper()}")
        print("=" * 78)

        for wname, (lo, hi) in WINDOWS.items():
            print(f"\n### {wname} " + "#" * 30)
            t0 = time.time()
            print("  [1/2] decoding each channel alone (64 fits x LOSO)...")
            acc_row = np.zeros(N_CHAN)
            acc_col = np.zeros(N_CHAN)
            percorrect = {}
            for ch in range(N_CHAN):
                seg = np.vstack([chan(si, ch, lo, hi)[::args.decim, :].T
                                 for si in range(len(subs))]).astype(np.float64)
                pr = decode(seg, y_row, groups)[0]
                percorrect[ch] = (pr == y_row)
                acc_row[ch] = percorrect[ch].mean()
                acc_col[ch] = (decode(seg, y_col, groups)[0] == y_col).mean()
                if (ch + 1) % 16 == 0:
                    print(f"        {ch + 1}/64 channels ({time.time() - t0:.0f}s)")

            order = np.argsort(-acc_row)
            print("\n        top 10 channels by ROW accuracy (chance 20%):")
            for i in order[:10]:
                tag = "OCC-9" if up[i] in OCC else ("FRONT" if up[i] in FRONT
                                                    else ("rim" if up[i] in RIM else ""))
                print(f"          {labels[i]:<5} {acc_row[i] * 100:6.2f}%  {tag}")

            occ_m = np.mean([acc_row[i] for i in range(N_CHAN) if up[i] in OCC])
            fro_m = np.mean([acc_row[i] for i in range(N_CHAN) if up[i] in FRONT])
            exc = (fro_m - 0.20) / (occ_m - 0.20) if occ_m > 0.20 else float("nan")
            print(f"\n        FRONT-5 mean {fro_m * 100:.2f}%   "
                  f"OCC-9 mean {occ_m * 100:.2f}%")
            print(f"        ratio raw {fro_m / occ_m:.2f}x   "
                  f"EXCESS-OVER-CHANCE {exc:.2f}x   <-- quote this one")

            # ---- reference diagnostic ----------------------------------
            rp, pp = pearsonr(dist_cz, acc_row)
            rs, ps = spearmanr(dist_cz, acc_row)
            print(f"\n        [DIAGNOSTIC] accuracy vs distance-from-Cz: "
                  f"r = {rp:+.3f} (p={pp:.1e}), rho = {rs:+.3f}")
            print(f"        rim channels: " + "  ".join(
                f"{c} {acc_row[up.index(c)] * 100:.2f}%" for c in RIM if c in up)
                + f"   |  CZ {acc_row[cz] * 100:.2f}%")

            # ---- per-subject (subject is the unit of analysis) ----------
            for tag, chset in [("FRONT-5", FRONT), ("OCC-9", OCC)]:
                chans = [i for i in range(N_CHAN) if up[i] in chset]
                per = np.array([np.mean([percorrect[c][groups == si].mean()
                                         for c in chans]) for si in subs_idx])
                print(f"        per-subject {tag:<8} row acc: mean {per.mean() * 100:.2f}%"
                      f"  median {np.median(per) * 100:.2f}%"
                      f"  above chance in {int((per > 0.20).sum())}/{len(per)}")
                np.save(f"{args.out_prefix}_{ref}_{wname.split('_')[0]}"
                        f"_{tag}_persubject.npy", per)

            # ---- 2. joint decode + activation pattern -------------------
            print("  [2/2] joint 64-channel decode + activation pattern...")
            Fm = np.vstack([allchan(si, lo, hi).mean(axis=1).T
                            for si in range(len(subs))]).astype(np.float64)
            pred, A = decode(Fm, y_row, groups, return_model=True)
            joint = (pred == y_row).mean()
            pattern = np.linalg.norm(A, axis=1)
            pattern = pattern / pattern.max()
            print(f"        joint 64-ch row accuracy {joint * 100:.2f}% (chance 20%)")
            pk = int(np.argmax(pattern))
            print(f"        activation pattern peaks at {labels[pk]}")
            occ_p = np.mean([pattern[i] for i in range(N_CHAN) if up[i] in OCC])
            fro_p = np.mean([pattern[i] for i in range(N_CHAN) if up[i] in FRONT])
            print(f"        pattern amplitude: FRONT-5 {fro_p:.3f}  "
                  f"OCC-9 {occ_p:.3f}  ratio {fro_p / occ_p:.2f}x")

            for i in range(N_CHAN):
                rows_out.append((ref, wname, labels[i], px[i], py[i], dist_cz[i],
                                 acc_row[i], acc_col[i], pattern[i]))

            # ---- figure -------------------------------------------------
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
                im1 = topoplot(axes[0], acc_row * 100, px, py, labels,
                               f"Row accuracy per channel (%)\n{ref} ref | {wname}",
                               "viridis", highlight=OCC | FRONT)
                fig.colorbar(im1, ax=axes[0], fraction=0.045,
                             label="% (chance 20)")
                im2 = topoplot(axes[1], pattern, px, py, labels,
                               f"Activation pattern (Haufe), normalised\n{ref} ref | {wname}",
                               "magma", highlight=OCC | FRONT)
                fig.colorbar(im2, ax=axes[1], fraction=0.045, label="a.u.")
                fig.tight_layout()
                png = f"{args.out_prefix}_{ref}_{wname.split('_')[0]}.png"
                fig.savefig(png, dpi=220)
                plt.close(fig)
                print(f"        figure -> {os.path.abspath(png)}")
            except ImportError:
                print("        matplotlib not installed; CSV written, no figure.")

    csv = f"{args.out_prefix}_results.csv"
    with open(csv, "w") as fh:
        fh.write("reference,window,channel,x,y,dist_from_cz,"
                 "row_acc,col_acc,activation_pattern\n")
        for r in rows_out:
            fh.write(",".join(str(v) for v in r) + "\n")
    print(f"\nWritten to {os.path.abspath(csv)}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  FIRST look at the distance-from-Cz diagnostic, not the front/back")
    print("  ratio. The recording is vertex-referenced, so Cz is identically")
    print("  zero and a radial (centre -> rim) gradient is expected from the")
    print("  reference alone, independently of where the source is.")
    print()
    print("    Radial r LARGE at vertex, SMALL at average reference")
    print("      -> the vertex map was reference-driven. Read Section 8.1 off")
    print("         the AVERAGE-reference map only.")
    print("    Radial r LARGE under BOTH references")
    print("      -> not a reference artifact. Peripheral pickup (neck/temporalis")
    print("         EMG, far-field ocular) is the next hypothesis, and neither")
    print("         FRONT-5 nor OCC-9 is where the effect lives.")
    print()
    print("  THEN read Section 8.1 off the average-reference map:")
    print("    frontal peak, monotonic posterior falloff -> ocular volume")
    print("      conduction is CONSISTENT with the map. Not established by it:")
    print("      sufficiency still needs the Section 8.2 simulation.")
    print("    independent occipital peak -> a frontal dipole cannot produce it.")
    print("      Report honestly; do not call it an ocular confound.")
    print("    diffuse -> fall back to the protocol paper (Section 8.1 row 3).")
    print()
    print("  M1 / M2 decide whether Section 8.3 is usable. If they are above")
    print("  chance, linked-mastoid re-referencing subtracts a signal-carrying")
    print("  reference into every channel and the control is uninterpretable.")
    print("  Use average reference instead and amend prereg Section 4.4 BEFORE")
    print("  filing.")
    print()
    print("  Quote the EXCESS-OVER-CHANCE ratio. Patterns, not weights (Haufe")
    print("  et al. 2014) -- state this in Methods, reviewers check.")


if __name__ == "__main__":
    main()
