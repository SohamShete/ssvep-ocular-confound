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
results = []

for file_path in mat_files:
    subj_id = os.path.basename(file_path).replace(".mat", "")
    mat = sio.loadmat(file_path)
    data = mat.get("data")
    if data is None:
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        data = mat[key]

    n_chans, n_samples, n_trials, n_freqs = data.shape
    
    var_explained_list = []
    for f_idx in range(n_freqs):
        for t_idx in range(n_trials):
            trial_data = data[:, :, t_idx, f_idx]
            X = trial_data[frontal_idx, :].T
            Y = trial_data[occipital_idx, :].T
            
            reg = LinearRegression().fit(X, Y)
            Y_pred = reg.predict(X)
            Y_res = Y - Y_pred
            
            var_total = np.var(Y, axis=0).sum()
            var_res = np.var(Y_res, axis=0).sum()
            if var_total > 0:
                var_explained_list.append(1.0 - (var_res / var_total))

    results.append({
        "subject": subj_id,
        "variance_explained_ratio": float(np.mean(var_explained_list))
    })
    print(f"[{subj_id}] Processed residual analysis.")

df_res = pd.DataFrame(results)
df_res.to_csv("eog_residual_analysis.csv", index=False)
print("Residual analysis complete. Output saved to eog_residual_analysis.csv")
