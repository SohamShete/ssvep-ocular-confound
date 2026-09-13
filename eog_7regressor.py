import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import LinearRegression

DATA_DIR = "./data"
FRONTAL_CHANS = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F8"]
OCCIPITAL_CHANS = ["Pz", "POz", "Oz", "O1", "O2", "PO3", "PO4", "P1", "P2"]

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
print(f"Loaded {len(mat_files)} datasets. Running 7-regressor OLS model...")

results = []
for file_path in mat_files:
    subj_id = os.path.basename(file_path).replace(".mat", "")
    mat = sio.loadmat(file_path)
    data = mat.get("data")
    if data is None:
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        data = mat[key]

    n_chans, n_samples, n_trials, n_freqs = data.shape
    
    r2_scores = []
    for f_idx in range(n_freqs):
        for t_idx in range(n_trials):
            trial_data = data[:, :, t_idx, f_idx]
            
            X = trial_data[frontal_idx, :].T
            Y = trial_data[occipital_idx, :].T
            
            reg = LinearRegression().fit(X, Y)
            r2_scores.append(reg.score(X, Y))
            
    avg_r2 = float(np.mean(r2_scores))
    results.append({"subject": subj_id, "mean_r2": avg_r2})
    print(f"[{subj_id}] 7-Regressor Mean R^2: {avg_r2:.4f}")

df_res = pd.DataFrame(results)
df_res.to_csv("eog_7regressor_results.csv", index=False)
print("7-regressor confirmatory analysis complete. Output saved to eog_7regressor_results.csv")