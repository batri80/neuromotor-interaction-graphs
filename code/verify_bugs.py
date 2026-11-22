import numpy as np
from ngs_pipeline_v2 import load_pahaw, CONFIG
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
row = part_df.iloc[0]
svc_data = row['svc_data']
print(f"Subject {row['participant_id']} ({row['label']}), {len(svc_data)} raw samples\n")

print("── 1. Raw timestamp column: units check ──")
t = svc_data[:, 2]
dt_all = np.diff(t)
print(f"First 10 raw t values: {t[:10]}")
print(f"First 10 dt values: {dt_all[:10]}")
print(f"Median dt: {np.median(dt_all):.6f}")
print(f"If dt ~ 0.0067 -> units are SECONDS (150Hz). If dt ~ 6.7 or ~7 -> units are MILLISECONDS.\n")

print("── 2. Button/surface state ──")
button = svc_data[:, 3]
print(f"Unique button values in this recording: {np.unique(button)}")
print(f"Fraction on-surface (button=1): {np.mean(button==1):.4f}\n")

print("── 3. a_std vs jerk_std: identical or just rank-tied? ──")
n_windows = 20
n = len(svc_data)
win_size = n // n_windows
a_std_vals, jerk_std_vals = [], []
for wi in range(n_windows):
    start = wi * win_size
    end = start + win_size if wi < n_windows - 1 else n
    window = svc_data[start:end]
    feats = extract_kinematic_features(window)
    a_std_vals.append(feats['a_std'])
    jerk_std_vals.append(feats['jerk_std'])

a_std_vals = np.array(a_std_vals)
jerk_std_vals = np.array(jerk_std_vals)
print(f"a_std values:    {np.round(a_std_vals, 4)}")
print(f"jerk_std values: {np.round(jerk_std_vals, 4)}")
print(f"Exactly equal element-wise? {np.allclose(a_std_vals, jerk_std_vals)}")
print(f"Max abs difference: {np.max(np.abs(a_std_vals - jerk_std_vals)):.6f}")
print(f"Correlation: {np.corrcoef(a_std_vals, jerk_std_vals)[0,1]:.6f}")
