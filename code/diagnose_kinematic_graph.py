import numpy as np
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import load_pahaw, CONFIG, compute_te_matrix
from test_kinematic_within_subject import (
    build_kinematic_within_subject_ngs, extract_kinematic_features, KINEMATIC_KEYS
)

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
row = part_df.iloc[0]
print(f"Inspecting subject {row['participant_id']} ({row['label']})\n")

svc_data = row['svc_data']
n_windows = 20
n = len(svc_data)
win_size = n // n_windows

feat_rows = []
for wi in range(n_windows):
    start = wi * win_size
    end = start + win_size if wi < n_windows - 1 else n
    window = svc_data[start:end]
    feats = extract_kinematic_features(window)
    feat_rows.append([feats[k] for k in KINEMATIC_KEYS])

X_windows = np.array(feat_rows)

print("── 1. Raw window-level feature correlation matrix ──")
print("(off-diagonal values near ±1 = mathematically/trivially coupled features)\n")
corr = np.corrcoef(X_windows.T)
print(f"{'':15}" + "".join(f"{k[:8]:>10}" for k in KINEMATIC_KEYS))
for i, k in enumerate(KINEMATIC_KEYS):
    print(f"{k:15}" + "".join(f"{corr[i,j]:10.2f}" for j in range(len(KINEMATIC_KEYS))))

print("\n── 2. Raw TE matrix (before thresholding) ──")
X_z = StandardScaler().fit_transform(X_windows)
te_matrix = compute_te_matrix(X_z)
print(f"{'':15}" + "".join(f"{k[:8]:>10}" for k in KINEMATIC_KEYS))
for i, k in enumerate(KINEMATIC_KEYS):
    print(f"{k:15}" + "".join(f"{te_matrix[i,j]:10.3f}" for j in range(len(KINEMATIC_KEYS))))

print("\n── 3. Which edges survive the 25th-percentile threshold? ──")
nonzero = te_matrix[te_matrix > 0]
if len(nonzero) > 0:
    thr = np.percentile(nonzero, 25)
    print(f"Threshold (25th pct of nonzero TE): {thr:.4f}")
    surviving = np.argwhere(te_matrix > thr)
    print(f"Surviving edges: {len(surviving)} / {len(KINEMATIC_KEYS)**2 - len(KINEMATIC_KEYS)} possible")
    for i, j in surviving:
        print(f"  {KINEMATIC_KEYS[i]:15} -> {KINEMATIC_KEYS[j]:15}  TE={te_matrix[i,j]:.4f}")
else:
    print("No nonzero TE values at all.")
