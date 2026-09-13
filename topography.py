"""
Scalp topography of sub-4 Hz target information -- v5, reference-free + inferential.

WHY v5 EXISTS
    The 35-subject v3 run showed that every result depends on the reference:
      accuracy vs distance-from-Cz   vertex +0.49 / +0.55   average +0.52 / +0.47
                                     mastoid -0.70 / -0.68
      midline monotonicity rho       vertex +0.43 / -0.10   average +0.43 / +0.19
                                     mastoid +0.95 / +1.00
    Vertex and average both put the zero point mid-scalp, which turns a
    monotonic front-to-back gradient into a U-shaped accuracy profile
    (minimum at CPz, rising again to Oz). Mastoid moves the zero to the sides
    and monotonicity returns -- but the mastoids carry target information
    themselves (M1 31.3%, M2 31.5% at vertex ref, 2nd and 3rd of 64), and
    linked-mastoid re-referencing RAISED OCC-9 pre-stimulus accuracy by
    +4.17 points, 95% CI [+2.86,+5.48], p=4.5e-6, in 30/35 subjects.
    So no potential-based reference can settle Section 8.1.

    CSD (surface Laplacian, Perrin et al. spherical splines via MNE) is
    reference-free by construction: the transform's rows sum to zero, so any
    spatially uniform component -- including the whole reference question --
    cancels. That is what v5 adds.

    CAVEAT, from the transform's own structure. Off-diagonal/diagonal ratio is
    highest at M1 (13.2), M2 (12.7), Fp1 (12.2), Fp2 (12.0), F7/F8, Fpz (9.1):
    these sit at the montage edge and their CSD depends heavily on
    interpolation from distant neighbours. OCC-9 is interior by comparison.
    CSD therefore answers "is there an INDEPENDENT POSTERIOR GENERATOR?" and
    NOT "how large is the frontal source?". Do not quote a CSD front/back ratio.

ALSO NEW
    * 4 windows, including a pre-stimulus window trimmed to sample 100. The 4 Hz
      zero-phase low-pass is applied to the full epoch (as pre-registered) and
      filtfilt spreads energy backwards: measured, a step at sample 125 puts
      2.6% of its amplitude into the pre-stim window MEAN but 35-48% into
      samples 120-124. If prestim and prestimTRIM agree, leakage is not driving
      the result and you can say so with a number. SSVEP itself cannot leak:
      8 Hz is -48 dB through the filtfilt pass.
    * SET-LEVEL decoding with LOCAL variants. "OCC-9-local" removes the OCC-9
      mean at each timepoint, so only the gradient WITHIN the occipital patch
      survives. A far-field ocular source is near-uniform across that patch and
      should lose most of its accuracy; a local occipital generator should not.
      Reference-free logic, no head model needed. Independent of CSD, so the two
      cross-check each other.
    * 1000-permutation nulls on ALL 64 channels, parallelised. These also supply
      nulls for the DERIVED statistics (excess ratio, radial r, anterior-
      posterior coefficient controlling for radius, midline rho), which have no
      analytic null because per-channel accuracies are not independent.
    * Per-subject Wilcoxon against the PER-SUBJECT permutation null, bootstrap
      95% CIs, sign consistency, Cohen's d, BH-FDR at q=0.05 within declared
      families, BF01 for any claim of absence (Section 9 requires all of these).
    * Column decoded ONLY in the pre-stimulus windows. Column correlates with
      stimulus frequency at r=+0.992, so during stimulation it is confounded and
      uninterpretable; computing it there would burn a quarter of the runtime to
      produce a number nobody can use.

STAGING -- DO THIS
    Stage 1: --n_perm 0     ~20 min.  Gives the CSD answer today.
    Stage 2: --n_perm 1000  ~7-12 h on 24 cores. Everything reportable.
    Run stage 1 first: if CSD shows an independent occipital focus, the paper
    changes and so does what is worth permuting.

WHAT THIS STILL CANNOT DO
    It cannot establish that ocular activity is SUFFICIENT to produce the
    observed occipital accuracy. A frontal maximum is consistent with volume
    conduction, not proof of it. Sufficiency is Section 8.2's forward
    simulation, which injects the MEASURED occipital microvolt-per-degree
    gradient into surrogate data and asks whether the same decoder reaches the
    same number. Do not write "ocular confirmed" off any topography.

USAGE
    pip install mne pingouin
    python topography.py --data_dir "C:\\ssvep\\data" --loc_file "C:\\ssvep\\data\\64-channels.loc" --n_perm 0    --out_prefix topo35s1
    python topography.py --data_dir "C:\\ssvep\\data" --loc_file "C:\\ssvep\\data\\64-channels.loc" --n_perm 1000 --out_prefix topo35s2
"""

import argparse
import csv as _csv
import json
import os
import re
import sys
import time

import numpy as np
import scipy.io
import scipy.signal
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from joblib import Parallel, delayed

FS = 250.0
N_TRIAL, N_TARGETS, N_BLOCKS, N_CHAN = 1500, 40, 6, 64
N_PRE = 125
ONSET = N_PRE + 35
M1_IDX, M2_IDX = 32, 42
CH_ROW, CH_COL = 0.20, 0.125

TGT_ROW = np.repeat(np.arange(5), 8)
TGT_COL = np.tile(np.arange(8), 5)

WINDOWS = {
    "prestim_-0.50_to_0.00s":      (0, N_PRE),
    "prestimTRIM_-0.50_to_-0.10s": (0, 100),
    "stim_0.14_to_0.64s":          (ONSET, ONSET + 125),
    "late_1.14_to_2.64s":          (ONSET + 250, ONSET + 625),
}

OCC = {"PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"}
FRONT = {"FP1", "FPZ", "FP2", "AF3", "AF4"}
RIM = ["M1", "M2", "CB1", "CB2"]
MIDLINE = ["FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ"]

# CB1/CB2 are the Chinese-convention inferior-occipital sites; I1/I2 in the
# 10-5 system are the accepted equivalents and are what standard_1005 calls them.
MNE_SUB = {"CB1": "I1", "CB2": "I2"}


# --------------------------------------------------------------------------
def find_subjects(d):
    pat = re.compile(r"^S(\d+)\.mat$", re.IGNORECASE)
    got = [(int(m.group(1)), f) for f in os.listdir(d) if (m := pat.match(f))]
    return [f for _, f in sorted(got)]


def read_locations(loc_path):
    """EEGLAB .loc -> labels and 2D plot coordinates. Nose-up.

    Radii rescaled so the outermost electrode sits at 0.88 of the unit head
    circle; raw .loc radii use an outer ring of ~0.5 and the montage would
    otherwise plot as a small blob inside an empty head.
    """
    labels, xs, ys = {}, {}, {}
    with open(loc_path, "r", errors="ignore") as fh:
        for line in fh:
            p = line.replace("`", "").split()
            if len(p) >= 4 and p[0].lstrip("-").isdigit():
                i = int(p[0]) - 1
                th = np.radians(float(p[1]))
                r = float(p[2])
                labels[i] = p[-1]
                xs[i] = r * np.sin(th)
                ys[i] = r * np.cos(th)
    if len(labels) < N_CHAN:
        sys.exit(f"ERROR: parsed only {len(labels)} channels from {loc_path}, "
                 f"expected {N_CHAN}.")
    o = sorted(labels)
    X = np.array([xs[i] for i in o]); Y = np.array([ys[i] for i in o])
    rmax = np.hypot(X, Y).max()
    if rmax > 0:
        X, Y = X * (0.88 / rmax), Y * (0.88 / rmax)
    return [labels[i] for i in o], X, Y


def csd_matrix(labels):
    """64x64 surface-Laplacian transform from MNE's spherical splines.

    CSD is linear, so pushing the identity through recovers the matrix once and
    it then applies as a single matmul. Positions come from standard_1005 by
    channel NAME -- this assumes the Tsinghua cap follows standard 10-10
    positions, which its labels assert. Row sums are checked against zero:
    that is what makes the transform reference-free.
    """
    try:
        import mne
    except ImportError:
        print("  [WARN] mne not installed -> CSD SKIPPED. Run: pip install mne")
        return None
    mne.set_log_level("ERROR")
    mont = mne.channels.make_standard_montage("standard_1005")
    canon = {c.upper(): c for c in mont.ch_names}
    names = []
    for l in labels:
        key = MNE_SUB.get(l.upper(), l.upper())
        if key not in canon:
            print(f"  [WARN] {l} not in standard_1005 -> CSD SKIPPED")
            return None
        names.append(canon[key])
    info = mne.create_info(names, FS, "eeg")
    info.set_montage(mont)
    ep = mne.EpochsArray(np.eye(N_CHAN)[np.newaxis], info, verbose=False)
    T = mne.preprocessing.compute_current_source_density(ep, copy=True).get_data()[0]
    rs = np.abs(T.sum(axis=1)).max()
    if rs > 1e-6:
        sys.exit(f"ERROR: CSD row sums {rs:.2e}, not reference-free. Aborting.")
    d = np.abs(np.diag(T)); ratio = (np.abs(T).sum(1) - d) / d
    print(f"  CSD transform OK (row sums {rs:.1e}). Least reliable sites "
          f"(edge of montage, off-diag/diag): " +
          ", ".join(f"{labels[i]} {ratio[i]:.1f}"
                    for i in np.argsort(-ratio)[:6]))
    return T


def load_subject(path, lp_hz=4.0):
    data = scipy.io.loadmat(path)["data"]
    if data.shape != (N_CHAN, N_TRIAL, N_TARGETS, N_BLOCKS):
        sys.exit(f"ERROR: {os.path.basename(path)} shape {data.shape}, expected "
                 f"({N_CHAN}, {N_TRIAL}, {N_TARGETS}, {N_BLOCKS})")
    sos = scipy.signal.butter(4, lp_hz / (FS / 2), btype="lowpass", output="sos")
    filt = scipy.signal.sosfiltfilt(sos, data, axis=1)
    return filt.reshape(N_CHAN, N_TRIAL, -1).astype(np.float32)


def apply_reference(seg, mode, T):
    """seg: (64, n_time, n_trials). Returns the referenced/transformed data."""
    if mode == "vertex":
        return seg
    if mode == "average":
        return seg - seg.mean(axis=0, keepdims=True)
    if mode == "mastoid":
        return seg - seg[[M1_IDX, M2_IDX]].mean(axis=0, keepdims=True)
    if mode == "csd":
        return np.einsum("ij,jtr->itr", T, seg)
    raise ValueError(mode)


# --------------------------------------------------------------------------
def loso_predict(F, y, groups):
    pred = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(F, y, groups):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit((F[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((F[te] - mu) / sd)
    return pred


def one_pass(Fch, Fsets, Fm, y_row, y_col, groups, subs_idx, do_col,
             perm_seed=None):
    """One full sweep: 64 single channels + set-level decodes + joint 64-ch.

    perm_seed=None -> observed. Otherwise labels are permuted WITHIN SUBJECT,
    preserving per-subject class balance and the LOSO structure. Row and column
    receive the SAME permutation so the two stay paired.
    """
    yr, yc = y_row.copy(), y_col.copy()
    if perm_seed is not None:
        rng = np.random.default_rng(perm_seed)
        for s in subs_idx:
            m = groups == s
            p = rng.permutation(int(m.sum()))
            yr[m] = yr[m][p]; yc[m] = yc[m][p]
    ns = len(subs_idx)
    acc_r = np.zeros(N_CHAN); acc_c = np.full(N_CHAN, np.nan)
    ps_r = np.zeros((N_CHAN, ns))
    for ch in range(N_CHAN):
        cr = loso_predict(Fch[ch], yr, groups) == yr
        acc_r[ch] = cr.mean()
        for j, s in enumerate(subs_idx):
            ps_r[ch, j] = cr[groups == s].mean()
        if do_col:
            acc_c[ch] = (loso_predict(Fch[ch], yc, groups) == yc).mean()
    set_acc, set_ps = {}, {}
    for name, F in Fsets.items():
        c = loso_predict(F, yr, groups) == yr
        set_acc[name] = c.mean()
        set_ps[name] = np.array([c[groups == s].mean() for s in subs_idx])
    cj = loso_predict(Fm, yr, groups) == yr
    joint = cj.mean()
    joint_ps = np.array([cj[groups == s].mean() for s in subs_idx])
    return acc_r, acc_c, ps_r, set_acc, set_ps, joint, joint_ps


def haufe_pattern(Fm, y, groups):
    """Refit on all data purely for the pattern; never used for an accuracy
    number, so it cannot leak. Haufe et al. 2014 NeuroImage 87:96-110: A=Cov(X)W.
    Weights of linear models are not neurophysiologically interpretable."""
    mu, sd = Fm.mean(0), Fm.std(0) + 1e-9
    Z = (Fm - mu) / sd
    f = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, y)
    p = np.linalg.norm(np.cov(Z, rowvar=False) @ f.coef_.T, axis=1)
    return p / p.max()


# ---- derived statistics ---------------------------------------------------
def excess_ratio(a, fi, oi):
    f, o = a[fi].mean() - CH_ROW, a[oi].mean() - CH_ROW
    return f / o if o > 1e-9 else np.nan


def ap_coef(a, y, d):
    X = np.column_stack([np.ones_like(y), y, d])
    b, *_ = np.linalg.lstsq(X, a, rcond=None)
    return b[1], b[2]


def midline_rho(a, y, mi):
    return spearmanr(y[mi], a[mi])[0]


# ---- inference ------------------------------------------------------------
def bh_fdr(p, q=0.05):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    passed = p[o] <= q * np.arange(1, n + 1) / n
    k = int(np.max(np.where(passed)[0])) + 1 if passed.any() else 0
    crit = p[o][k - 1] if k else 0.0
    return p <= crit, crit


def boot_ci(x, n=20000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bf01(x, mu0=0.0):
    """Evidence FOR the null. Section 9: BF01 > 3 before any absence claim."""
    try:
        import pingouin as pg
        t, _ = ttest_1samp(x, mu0)
        return 1.0 / float(pg.bayesfactor_ttest(t, len(x)))
    except Exception:
        return np.nan


def perm_p(obs, null, two_sided=False):
    null = np.asarray(null)
    if null.size == 0 or not np.isfinite(obs):
        return np.nan
    if two_sided:
        c = np.sum(np.abs(null - np.median(null)) >= abs(obs - np.median(null)))
    else:
        c = np.sum(null >= obs)
    return (c + 1) / (len(null) + 1)


# --------------------------------------------------------------------------
def topoplot(ax, vals, x, y, labels, title, cmap, highlight=()):
    from scipy.interpolate import griddata
    gx, gy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    gz = griddata((x, y), vals, (gx, gy), method="cubic")
    gz[gx ** 2 + gy ** 2 > 1.0] = np.nan
    im = ax.contourf(gx, gy, gz, levels=25, cmap=cmap)
    ax.contour(gx, gy, gz, levels=8, colors="k", linewidths=0.3, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1.5)
    ax.plot([-0.09, 0, 0.09], [0.995, 1.11, 0.995], "k-", lw=1.5)
    ax.scatter(x, y, c="k", s=5, zorder=3)
    for i, l in enumerate(labels):
        if l.upper() in highlight:
            ax.scatter(x[i], y[i], facecolors="none", edgecolors="w", s=70,
                       lw=1.6, zorder=4)
            ax.annotate(l, (x[i], y[i]), fontsize=6, color="w",
                        xytext=(2, 3), textcoords="offset points", zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--loc_file", required=True)
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--lowpass", type=float, default=4.0)
    ap.add_argument("--decim", type=int, default=20)
    ap.add_argument("--reference", default="all",
                    choices=["vertex", "average", "mastoid", "csd", "all"])
    ap.add_argument("--n_perm", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out_prefix", default="topo")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: folder not found: {args.data_dir}")
    if not os.path.isfile(args.loc_file):
        sys.exit(f"ERROR: loc file not found: {args.loc_file}")

    labels, px, py = read_locations(args.loc_file)
    up = [l.upper() for l in labels]
    for want, i in [("M1", M1_IDX), ("M2", M2_IDX)]:
        if up[i] != want:
            sys.exit(f"ERROR: index {i} is {up[i]}, expected {want}. The mastoid "
                     f"reference would be wrong. Fix the indices.")
    if "CZ" not in up:
        sys.exit("ERROR: no CZ in the montage; the radial diagnostic needs it.")
    cz = up.index("CZ")
    dist_cz = np.hypot(px - px[cz], py - py[cz])
    fi = [i for i in range(N_CHAN) if up[i] in FRONT]
    oi = [i for i in range(N_CHAN) if up[i] in OCC]
    ri = [up.index(c) for c in RIM if c in up]
    mi = [up.index(c) for c in MIDLINE if c in up]
    if len(fi) != 5 or len(oi) != 9:
        sys.exit(f"ERROR: matched {len(fi)} FRONT-5 and {len(oi)} OCC-9 by name; "
                 f"expected 5 and 9. Check the .loc file.")

    refs = ["vertex", "average", "mastoid", "csd"] if args.reference == "all" \
        else [args.reference]
    T = csd_matrix(labels) if "csd" in refs else None
    if "csd" in refs and T is None:
        refs = [r for r in refs if r != "csd"]

    subs = find_subjects(args.data_dir)
    if args.max_subjects:
        subs = subs[:args.max_subjects]
    if len(subs) < 3:
        sys.exit(f"ERROR: need >= 3 subjects for LOSO, found {len(subs)}")
    ns = len(subs)

    print("=" * 78)
    print("SUB-4 Hz TARGET INFORMATION: TOPOGRAPHY (v5, reference-free)")
    print(f"  subjects   : {ns}   low-pass {args.lowpass} Hz   decim {args.decim}")
    print(f"  references : {', '.join(refs)}")
    print(f"  windows    : {', '.join(WINDOWS)}")
    print(f"  midline    : {[labels[i] for i in mi]}")
    print(f"  perms      : {args.n_perm}" + ("   <-- STAGE 1, nothing here is "
          "reportable" if args.n_perm == 0 else
          f"   (p floor {1 / (args.n_perm + 1):.4f})"))
    print(f"  chance     : row {CH_ROW*100:.2f}%   col {CH_COL*100:.2f}%")
    print("=" * 78)

    print("\nloading (once; transforms applied on the fly)", end="", flush=True)
    t0 = time.time(); cache = []
    for f in subs:
        cache.append(load_subject(os.path.join(args.data_dir, f), args.lowpass))
        print(".", end="", flush=True)
    print(f" done ({time.time()-t0:.0f}s, "
          f"{sum(s.nbytes for s in cache)/1e9:.2f} GB resident)")

    y_row = np.concatenate([TGT_ROW[np.arange(240) // N_BLOCKS]] * ns)
    y_col = np.concatenate([TGT_COL[np.arange(240) // N_BLOCKS]] * ns)
    groups = np.concatenate([np.full(240, i) for i in range(ns)])
    subs_idx = np.unique(groups)

    chan_rows, summ_rows, ps_store = [], [], {}

    for ref in refs:
        for wname, (lo, hi) in WINDOWS.items():
            do_col = wname.startswith("prestim")
            tag = f"{ref}|{wname}"
            print("\n" + "=" * 78)
            print(f"{tag}   samples {lo}:{hi}   column decoded: {do_col}")
            print("=" * 78)
            tw = time.time()

            nf = len(range(0, hi - lo, args.decim))
            Fch = np.empty((N_CHAN, 240 * ns, nf))
            Fm = np.empty((240 * ns, N_CHAN))
            loc_sets = {"FRONT-5": fi, "OCC-9": oi, "RIM-4": ri}
            Fsets = {k: np.empty((240 * ns, len(v) * nf))
                     for k, v in loc_sets.items()}
            for k in ("FRONT-5-local", "OCC-9-local"):
                Fsets[k] = np.empty((240 * ns, len(loc_sets[k.split("-local")[0]]) * nf))
            for si in range(ns):
                seg = apply_reference(cache[si][:, lo:hi, :], ref, T)
                sl = slice(si * 240, (si + 1) * 240)
                Fch[:, sl, :] = seg[:, ::args.decim, :].transpose(0, 2, 1)
                Fm[sl, :] = seg.mean(axis=1).T
                for k, idxs in loc_sets.items():
                    Fsets[k][sl] = seg[idxs][:, ::args.decim, :] \
                        .transpose(2, 0, 1).reshape(240, -1)
                for k in ("FRONT-5", "OCC-9"):
                    s2 = seg[loc_sets[k]]
                    s2 = s2 - s2.mean(axis=0, keepdims=True)   # patch-local only
                    Fsets[k + "-local"][sl] = s2[:, ::args.decim, :] \
                        .transpose(2, 0, 1).reshape(240, -1)
            print(f"  features: channel {Fch.shape}  joint {Fm.shape}  "
                  f"({(Fch.nbytes + sum(v.nbytes for v in Fsets.values()))/1e6:.0f} MB)")

            acc_r, acc_c, ps_r, set_acc, set_ps, joint, joint_ps = one_pass(
                Fch, Fsets, Fm, y_row, y_col, groups, subs_idx, do_col)
            pattern = haufe_pattern(Fm, y_row, groups)
            t_obs = time.time() - tw
            print(f"  observed pass {t_obs:.0f}s")

            o_exc = excess_ratio(acc_r, fi, oi)
            o_rad = pearsonr(dist_cz, acc_r)[0]
            o_ap, o_radb = ap_coef(acc_r, py, dist_cz)
            o_mid = midline_rho(acc_r, py, mi)

            nul = {}
            if args.n_perm > 0:
                print(f"  {args.n_perm} permutations on {args.n_jobs} jobs "
                      f"(rough ETA {t_obs*args.n_perm/max(1,(os.cpu_count() if args.n_jobs==-1 else args.n_jobs))/3600:.1f} h)...")
                tp = time.time()
                out = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=1)(
                    delayed(one_pass)(Fch, Fsets, Fm, y_row, y_col, groups,
                                      subs_idx, do_col, 100000 + k)
                    for k in range(args.n_perm))
                nul["r"] = np.array([o[0] for o in out])
                nul["c"] = np.array([o[1] for o in out])
                nul["ps"] = np.array([o[2] for o in out])
                nul["set"] = {k: np.array([o[3][k] for o in out]) for k in Fsets}
                nul["setps"] = {k: np.array([o[4][k] for o in out]) for k in Fsets}
                nul["joint"] = np.array([o[5] for o in out])
                nul["jointps"] = np.array([o[6] for o in out])
                nul["exc"] = np.array([excess_ratio(a, fi, oi) for a in nul["r"]])
                nul["rad"] = np.array([pearsonr(dist_cz, a)[0] for a in nul["r"]])
                nul["ap"] = np.array([ap_coef(a, py, dist_cz)[0] for a in nul["r"]])
                nul["mid"] = np.array([midline_rho(a, py, mi) for a in nul["r"]])
                print(f"  permutations {(time.time()-tp)/60:.1f} min")

            # ---- report ---------------------------------------------------
            print("\n  top 12 channels, ROW accuracy (chance 20%):")
            for i in np.argsort(-acc_r)[:12]:
                t_ = ("OCC-9" if up[i] in OCC else "FRONT" if up[i] in FRONT
                      else "rim" if up[i] in RIM else "")
                pv = f"  p={perm_p(acc_r[i], nul['r'][:, i]):.4f}" if nul else ""
                print(f"    {labels[i]:<5} {acc_r[i]*100:6.2f}%  {t_:<5}{pv}")

            print(f"\n  FRONT-5 {acc_r[fi].mean()*100:.2f}%   "
                  f"OCC-9 {acc_r[oi].mean()*100:.2f}%   "
                  f"RIM-4 {acc_r[ri].mean()*100:.2f}%   CZ {acc_r[cz]*100:.2f}%")
            print(f"  EXCESS-OVER-CHANCE front/occ {o_exc:.2f}x" +
                  (f"  p={perm_p(o_exc, nul['exc']):.4f}" if nul else "") +
                  ("   [do NOT quote under CSD: Fp* are edge sites]"
                   if ref == "csd" else ""))

            print("\n  [SHAPE / REFERENCE DIAGNOSTICS]")
            print(f"    radial  acc vs dist-from-Cz   r = {o_rad:+.3f}" +
                  (f"  p={perm_p(abs(o_rad), np.abs(nul['rad']), True):.4f}" if nul else ""))
            print(f"    OLS acc ~ y + dist_cz         y {o_ap:+.5f}" +
                  (f"  p={perm_p(abs(o_ap), np.abs(nul['ap']), True):.4f}" if nul else "") +
                  f"   dist {o_radb:+.5f}")
            print(f"    midline monotonicity rho      {o_mid:+.3f}" +
                  (f"  p={perm_p(abs(o_mid), np.abs(nul['mid']), True):.4f}" if nul else ""))
            print("    midline chain (ant->post): " + "  ".join(
                f"{labels[i]} {acc_r[i]*100:.1f}" for i in mi))
            print("    rim: " + "  ".join(f"{labels[i]} {acc_r[i]*100:.2f}%"
                                          for i in ri))

            print("\n  [SET-LEVEL, multichannel]   -local = set mean removed, "
                  "so only the WITHIN-PATCH gradient remains")
            for k in ["FRONT-5", "FRONT-5-local", "OCC-9", "OCC-9-local", "RIM-4"]:
                per = set_ps[k]
                base = nul["setps"][k].mean(0) if nul else np.full(ns, CH_ROW)
                lo_c, hi_c = boot_ci(per)
                try:
                    wp = wilcoxon(per, base, alternative="greater").pvalue
                except Exception:
                    wp = np.nan
                dd = (per - base).mean() / ((per - base).std(ddof=1) + 1e-12)
                pv = f"  perm p={perm_p(set_acc[k], nul['set'][k]):.4f}" if nul else ""
                print(f"    {k:<15} {set_acc[k]*100:6.2f}%  95%CI "
                      f"[{lo_c*100:.2f},{hi_c*100:.2f}]  >null {int((per>base).sum())}/{ns}"
                      f"  W p={wp:.2e}  d={dd:.2f}  BF01={bf01(per-base):.3g}{pv}")
                ps_store[f"{ref}|{wname}|{k}"] = per
                summ_rows.append(dict(reference=ref, window=wname, set=k,
                    acc=set_acc[k], ci_lo=lo_c, ci_hi=hi_c,
                    n_above_null=int((per > base).sum()), n_subj=ns,
                    wilcoxon_p=wp, cohens_d=dd, bf01=bf01(per - base),
                    perm_p=perm_p(set_acc[k], nul["set"][k]) if nul else np.nan,
                    excess_ratio=o_exc, radial_r=o_rad, ap_coef=o_ap,
                    midline_rho=o_mid, joint_acc=joint))

            print(f"\n  joint 64-ch {joint*100:.2f}%" +
                  (f"  perm p={perm_p(joint, nul['joint']):.4f}" if nul else "") +
                  f"   pattern peaks at {labels[int(np.argmax(pattern))]}"
                  f"   FRONT-5 {pattern[fi].mean():.3f}  OCC-9 {pattern[oi].mean():.3f}")

            for i in range(N_CHAN):
                chan_rows.append(dict(reference=ref, window=wname,
                    channel=labels[i], x=px[i], y=py[i], dist_cz=dist_cz[i],
                    row_acc=acc_r[i], col_acc=acc_c[i], pattern=pattern[i],
                    row_p=perm_p(acc_r[i], nul["r"][:, i]) if nul else np.nan,
                    col_p=perm_p(acc_c[i], nul["c"][:, i]) if (nul and do_col) else np.nan,
                    row_null_mean=nul["r"][:, i].mean() if nul else np.nan))
            np.save(f"{args.out_prefix}_{ref}_{wname}_perchan_persubj.npy", ps_r)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axx = plt.subplots(1, 2, figsize=(9, 4.4))
                i1 = topoplot(axx[0], acc_r * 100, px, py, labels,
                              f"Row accuracy per channel (%)\n{tag}",
                              "viridis", highlight=OCC | FRONT)
                fig.colorbar(i1, ax=axx[0], fraction=.045, label="% (chance 20)")
                i2 = topoplot(axx[1], pattern, px, py, labels,
                              f"Activation pattern (Haufe)\n{tag}",
                              "magma", highlight=OCC | FRONT)
                fig.colorbar(i2, ax=axx[1], fraction=.045, label="a.u.")
                fig.tight_layout()
                png = f"{args.out_prefix}_{ref}_{wname}.png"
                fig.savefig(png, dpi=220); plt.close(fig)
                print(f"  figure -> {os.path.abspath(png)}")
            except ImportError:
                print("  matplotlib missing; CSVs written, no figure.")
            print(f"  cell total {(time.time()-tw)/60:.1f} min")

    if args.n_perm > 0:
        for lbl, key in [("(a) per-channel row", "row_p"),
                         ("(a) per-channel col", "col_p"),
                         ("(c) set-level", "perm_p")]:
            src = summ_rows if key == "perm_p" else chan_rows
            ps = np.array([r[key] for r in src], float)
            k = np.isfinite(ps)
            if not k.any():
                continue
            sig, crit = bh_fdr(ps[k])
            print(f"\nBH-FDR q=0.05, family {lbl}: {int(sig.sum())}/{int(k.sum())} "
                  f"survive (critical p={crit:.5f})")
            j = 0
            for r in src:
                if np.isfinite(r[key]):
                    r[key + "_fdr"] = bool(sig[j]); j += 1
                else:
                    r[key + "_fdr"] = None

    for nm, rws in [("channels", chan_rows), ("summary", summ_rows)]:
        with open(f"{args.out_prefix}_{nm}.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rws[0].keys()))
            w.writeheader(); w.writerows(rws)
    np.savez(f"{args.out_prefix}_persubject.npz", **ps_store)
    with open(f"{args.out_prefix}_provenance.json", "w") as fh:
        json.dump(dict(subjects=[s[:-4] for s in subs], n_perm=args.n_perm,
                       lowpass=args.lowpass, decim=args.decim, references=refs,
                       windows={k: list(v) for k, v in WINDOWS.items()},
                       front_idx=fi, occ_idx=oi, rim_idx=ri, m1=M1_IDX, m2=M2_IDX,
                       midline=[labels[i] for i in mi],
                       csd_from="mne standard_1005, CB1->I1, CB2->I2"), fh, indent=2)
    print(f"\nWritten: {args.out_prefix}_channels.csv, _summary.csv, "
          f"_persubject.npz, _provenance.json")

    print("\n" + "=" * 78)
    print("READ IN THIS ORDER")
    print("=" * 78)
    print("  1. CSD, OCC-9 and the occipital channels. CSD is reference-free, so")
    print("     this is the only cell that can answer Section 8.1.")
    print("       occipital CSD focus present, survives -> INDEPENDENT POSTERIOR")
    print("         GENERATOR. A frontal dipole cannot make a local Laplacian")
    print("         peak. Section 8.1 row 2. Anticipatory slow potentials or")
    print("         spatial attention. Do NOT call it an ocular confound.")
    print("       occipital CSD at chance -> no local posterior source; the")
    print("         occipital signal is spatially smooth, consistent with a")
    print("         far-field generator. Section 8.1 row 1, but CONSISTENT WITH")
    print("         is not SUFFICIENT -- that is still Section 8.2.")
    print("  2. OCC-9-local vs OCC-9. Independent cross-check on the same")
    print("     question with no head model: removing the patch mean removes any")
    print("     spatially uniform far-field component.")
    print("       -local collapses -> the information was uniform across the")
    print("         patch, i.e. far-field. Agrees with a null CSD result.")
    print("       -local survives -> local occipital gradient. Should agree with")
    print("         a positive CSD result. IF CSD AND -local DISAGREE, believe")
    print("         neither and say so; they rest on different assumptions.")
    print("  3. prestim vs prestimTRIM. Agreement -> backward filter leakage is")
    print("     not driving the pre-stimulus result.")
    print("  4. Do not quote a CSD front/back ratio. Fp1/Fpz/Fp2 and M1/M2 are")
    print("     montage-edge sites where CSD is interpolation-dominated.")
    print("  5. Sign consistency (n/35 above the permutation null) before any")
    print("     p-value; Section 3.5. Effect sizes and CIs for everything.")


if __name__ == "__main__":
    main()
