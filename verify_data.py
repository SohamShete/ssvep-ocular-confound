"""
Verify the Benchmark dataset folder before running anything else.

Catches, in about a minute, the problems that otherwise surface fifteen
minutes into a run or - worse - silently produce wrong numbers:

  * missing or truncated subject files
  * files still buried in p1/p2/p3/p4 subfolders
  * wrong array shape or dtype
  * stimulus frequencies that do not match the published values
  * CHANNEL INDICES THAT DO NOT MATCH THE CHANNEL NAMES

That last check is the important one. Every analysis in this project depends
on indices [47, 53, 54, 55, 56, 57, 60, 61, 62] actually being PZ PO5 PO3
POz PO4 PO6 O1 Oz O2. If the montage file orders channels differently, every
result is wrong and nothing else will tell you.

USAGE
    python verify_data.py --data_dir "C:\\ssvep\\data"
    python verify_data.py --data_dir "C:\\ssvep\\data" --loc_file "C:\\ssvep\\64-channels.loc"
"""

import argparse
import os
import re
import sys

import numpy as np
import scipy.io

N_TRIAL, N_TARGETS, N_BLOCKS, N_CHAN = 1500, 40, 6, 64
EXPECTED_N_SUBJECTS = 35

OCC_IDX = [47, 53, 54, 55, 56, 57, 60, 61, 62]
OCC_NAMES = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
FRONT_IDX = [0, 1, 2, 3, 4]
FRONT_NAMES = ["FP1", "FPZ", "FP2", "AF3", "AF4"]

PASS, FAIL, WARN = [], [], []


def ok(m):
    PASS.append(m)
    print(f"  [ OK ] {m}")


def bad(m):
    FAIL.append(m)
    print(f"  [FAIL] {m}")


def warn(m):
    WARN.append(m)
    print(f"  [WARN] {m}")


def check_channels(loc_path):
    print("\n[3] CHANNEL MONTAGE  (the check that matters most)")
    if not loc_path or not os.path.isfile(loc_path):
        warn("64-channels.loc not supplied or not found. Channel indices are "
             "UNVERIFIED -- download it from the Tsinghua page and re-run with "
             "--loc_file. Do not trust any result until this passes.")
        return

    labels = {}
    with open(loc_path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.replace("`", "").split()
            if len(parts) >= 4 and parts[0].lstrip("-").isdigit():
                labels[int(parts[0]) - 1] = parts[-1].upper()   # file is 1-based

    if len(labels) < N_CHAN:
        warn(f"parsed only {len(labels)} channels from the .loc file; expected 64")

    for tag, idx, names in [("OCC-9", OCC_IDX, OCC_NAMES),
                            ("FRONT-5", FRONT_IDX, FRONT_NAMES)]:
        got = [labels.get(i, "?") for i in idx]
        if got == names:
            ok(f"{tag} indices map to {', '.join(got)}")
        else:
            bad(f"{tag} MISMATCH.\n"
                f"         expected {names}\n"
                f"         actually {got}\n"
                f"         -> Fix the indices in the scripts before running anything.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--loc_file", default=None,
                    help="Path to 64-channels.loc (strongly recommended)")
    ap.add_argument("--quick", action="store_true",
                    help="Skip loading array contents (checks names/sizes only)")
    args = ap.parse_args()

    print("=" * 70)
    print("BENCHMARK DATASET VERIFICATION")
    print(f"folder: {args.data_dir}")
    print("=" * 70)

    # ---- 1. folder and file inventory ---------------------------------
    print("\n[1] FILES")
    if not os.path.isdir(args.data_dir):
        print(f"  [FAIL] folder does not exist: {args.data_dir}")
        sys.exit(1)

    entries = os.listdir(args.data_dir)
    subdirs = [e for e in entries
               if os.path.isdir(os.path.join(args.data_dir, e))]
    if subdirs:
        warn(f"subfolders present ({', '.join(subdirs[:5])}). Files must be FLAT "
             f"in one folder -- move every S*.mat up one level.")

    pat = re.compile(r"^S(\d+)\.mat$", re.IGNORECASE)
    found = {int(m.group(1)): f for f in entries if (m := pat.match(f))}

    if "Freq_Phase.mat" in entries:
        ok("Freq_Phase.mat present")
    else:
        bad("Freq_Phase.mat MISSING -- must sit beside the S*.mat files")

    if not found:
        bad("no S*.mat files found")
        sys.exit(1)

    nums = sorted(found)
    ok(f"{len(nums)} subject files: S{nums[0]}..S{nums[-1]}")
    missing = [n for n in range(1, EXPECTED_N_SUBJECTS + 1) if n not in found]
    if missing:
        warn(f"missing {len(missing)} of {EXPECTED_N_SUBJECTS}: "
             f"S{', S'.join(map(str, missing[:12]))}"
             f"{' ...' if len(missing) > 12 else ''}. "
             f"You need all 35 for results comparable to published work.")
    else:
        ok(f"all {EXPECTED_N_SUBJECTS} subjects present")

    sizes = {n: os.path.getsize(os.path.join(args.data_dir, f)) / 1e6
             for n, f in found.items()}
    small = {n: s for n, s in sizes.items() if s < 50}
    if small:
        bad(f"suspiciously small (likely truncated download): "
            + ", ".join(f"S{n} {s:.0f}MB" for n, s in sorted(small.items())))
    else:
        ok(f"file sizes {min(sizes.values()):.0f}-{max(sizes.values()):.0f} MB, "
           f"all plausible")

    # ---- 2. stimulus metadata -----------------------------------------
    print("\n[2] STIMULUS METADATA")
    fp = os.path.join(args.data_dir, "Freq_Phase.mat")
    if os.path.isfile(fp):
        try:
            md = scipy.io.loadmat(fp)
            freqs = md["freqs"].flatten()
            phases = md["phases"].flatten()
            expected = np.arange(40) * 0.2 + 8.0
            if len(freqs) != 40:
                bad(f"freqs has {len(freqs)} entries, expected 40")
            elif np.allclose(np.sort(freqs), expected, atol=1e-6):
                ok(f"40 frequencies, 8.0-15.8 Hz in 0.2 Hz steps, as published")
            else:
                bad(f"frequencies do not match published values "
                    f"(min {freqs.min():.2f}, max {freqs.max():.2f}). "
                    f"Wrong dataset?")
            if len(phases) == 40:
                ok(f"40 phases, range {phases.min():.3f}-{phases.max():.3f} rad")
            else:
                bad(f"phases has {len(phases)} entries, expected 40")
        except Exception as e:
            bad(f"could not read Freq_Phase.mat: {e}")

    # ---- 2b. layout structure (Step 0, automated) ----------------------
    print("\n[2b] LAYOUT STRUCTURE  (Step 0)")
    if os.path.isfile(fp):
        try:
            f40 = scipy.io.loadmat(fp)["freqs"].flatten()
            if len(f40) == 40:
                row = np.repeat(np.arange(5), 8)
                col = np.tile(np.arange(8), 5)
                recon = 8.0 + 1.0 * col + 0.2 * row
                if np.allclose(np.sort(f40), np.sort(recon), atol=1e-6):
                    if np.allclose(f40, recon, atol=1e-6):
                        ok("freq = 8.0 + 1.0*col + 0.2*row on a row-major 5x8 grid")
                    else:
                        warn("frequencies match the 5x8 set but not in row-major "
                             "order. Re-derive the mapping from Fig. 1 of "
                             "Wang et al. 2017 before any row/column analysis.")
                r_col = np.corrcoef(f40, col)[0, 1]
                r_row = np.corrcoef(f40, row)[0, 1]
                print(f"         column vs frequency  r = {r_col:+.3f}   "
                      f"{'CONFOUNDED - not evidence of gaze' if abs(r_col) > 0.3 else 'usable'}")
                print(f"         row    vs frequency  r = {r_row:+.3f}   "
                      f"{'CONFOUNDED - not evidence of gaze' if abs(r_row) > 0.3 else 'usable'}")
                if abs(r_col) > 0.3 and abs(r_row) > 0.3:
                    bad("NEITHER spatial axis is separable from frequency. "
                        "The dissociation analysis is not possible on this layout.")
                else:
                    ok("at least one spatial axis is separable from frequency")
        except Exception as e:
            warn(f"layout check failed: {e}")

    # ---- 3. channel montage -------------------------------------------
    loc = args.loc_file
    if loc is None:
        for cand in ["64-channels.loc", os.path.join("..", "64-channels.loc")]:
            p = os.path.join(args.data_dir, cand)
            if os.path.isfile(p):
                loc = p
                break
    check_channels(loc)

    # ---- 4. array contents --------------------------------------------
    print("\n[4] ARRAY CONTENTS")
    if args.quick:
        warn("--quick given; array contents not checked")
    else:
        to_check = nums[:3] + ([nums[-1]] if len(nums) > 3 else [])
        for n in to_check:
            f = found[n]
            try:
                d = scipy.io.loadmat(os.path.join(args.data_dir, f))["data"]
            except Exception as e:
                bad(f"{f}: cannot read ({e})")
                continue
            if d.shape != (N_CHAN, N_TRIAL, N_TARGETS, N_BLOCKS):
                bad(f"{f}: shape {d.shape}, expected "
                    f"({N_CHAN}, {N_TRIAL}, {N_TARGETS}, {N_BLOCKS})")
                continue
            sd = float(np.std(d))
            if not np.isfinite(d).all():
                bad(f"{f}: contains NaN or Inf")
            elif sd < 1e-6:
                bad(f"{f}: array is flat (std={sd:.2e}) -- corrupt")
            elif not (1.0 < sd < 200.0):
                warn(f"{f}: std={sd:.2f} uV is outside the usual 5-40 uV range")
            else:
                ok(f"{f}: shape OK, std {sd:.2f} uV, "
                   f"range [{d.min():.0f}, {d.max():.0f}] uV")

    # ---- summary -------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failures")
    print("=" * 70)
    if FAIL:
        print("\nDO NOT PROCEED. Fix the failures above first:")
        for m in FAIL:
            print(f"  - {m.splitlines()[0]}")
        sys.exit(1)
    if WARN:
        print("\nUsable, but read the warnings. Especially any unverified")
        print("channel mapping -- that one silently corrupts every result.")
    else:
        print("\nAll checks passed. Run fbcca_baseline.py next.")


if __name__ == "__main__":
    main()
