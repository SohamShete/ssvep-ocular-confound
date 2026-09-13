import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression

DATA_DIR = "./data"
FS = 250.0  # Sampling rate (Hz)
FRONTAL_CHANS = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F8"]
OCCIPITAL_CHANS = ["Pz", "POz", "Oz", "O1", "O2", "PO3", "PO4", "P1", "P2"]

# Benchmark SSVEP Target Frequencies (8.0 Hz to 15.8 Hz, 0.2 Hz step -> 40 targets)
FREQS = [8.0 + 0.2 * i for i in range(40)]
NHARMONICS = 2

def generate_ref_signals(freq, n_samples, fs):
    t = np.arange(n_samples) / fs
    ref = []
    for h in range(1, NHARMONICS + 1):
        ref.append(np.sin(2 * np.pi * h * freq * t))
        ref.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(ref).T

loc_path = os.path.join(DATA_DIR, "64-channels.loc")
ch_names = []
if os.path.exists(loc_path):
    with open(loc_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                ch_names.append(parts[-1])

if ch_names:
    frontal_idx = [ch_names.index(c) for c in FRONTAL_CHANS if c in ch_names]
    occipital_idx = [ch_names.index(c) for c in OCCIPITAL_CHANS if c in ch_names]
else:
    frontal_idx = [0, 1, 2, 3, 4, 5, 6]
    occipital_idx = [47, 48, 54, 55, 56, 60, 61, 62, 63]

mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "S*.mat")))
results = []

for file_path in mat_files:
    subj_id = os.path.basename(file_path).replace(".mat", "")
    mat = sio.loadmat(file_path)
    data = mat.get("data")
    if data is None:
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        data = mat[key]

    n_chans, n_samples, n_trials, n_freqs = data.shape
    target_freqs = FREQS[:n_freqs]

    raw_correct = 0
    res_correct = 0
    total_trials = n_trials * n_freqs

    for f_idx, true_freq in enumerate(target_freqs):
        ref_signals = [generate_ref_signals(f, n_samples, FS) for f in target_freqs]

        for t_idx in range(n_trials):
            trial_data = data[:, :, t_idx, f_idx]
            
            X_front = trial_data[frontal_idx, :].T
            Y_raw = trial_data[occipital_idx, :].T
            
            # OLS residual removal
            reg = LinearRegression().fit(X_front, Y_raw)
            Y_res = Y_raw - reg.predict(X_front)

            # Standard CCA on Raw vs Residual
            raw_corrs = []
            res_corrs = []
            for ref in ref_signals:
                cca_raw = CCA(n_components=1).fit(Y_raw, ref)
                u_r, v_r = cca_raw.transform(Y_raw, ref)
                raw_corrs.append(np.corrcoef(u_r.T, v_r.T)[0, 1])

                cca_res = CCA(n_components=1).fit(Y_res, ref)
                u_s, v_s = cca_res.transform(Y_res, ref)
                res_corrs.append(np.corrcoef(u_s.T, v_s.T)[0, 1])

            if np.argmax(raw_corrs) == f_idx:
                raw_correct += 1
            if np.argmax(res_corrs) == f_idx:
                res_correct += 1

    raw_acc = raw_correct / total_trials
    res_acc = res_correct / total_trials

    results.append({
        "subject": subj_id,
        "raw_accuracy": raw_acc,
        "residual_accuracy": res_acc,
        "acc_delta": res_acc - raw_acc
    })
    print(f"[{subj_id}] Raw Acc: {raw_acc:.4f} | Residual Acc: {res_acc:.4f} | Delta: {res_acc - raw_acc:+.4f}")

df_res = pd.DataFrame(results)
df_res.to_csv("fbcca_raw_vs_residual.csv", index=False)
print("Decoding comparison complete. Output saved to fbcca_raw_vs_residual.csv")
