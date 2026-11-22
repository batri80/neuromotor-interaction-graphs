import numpy as np
import pandas as pd
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from scipy import stats

from ngs_pipeline_v2 import NGS_CORE

ngs_df = pd.read_pickle('within_subject_kinematic_ngs_df.pkl')
X_full = ngs_df[NGS_CORE].values
y_full = ngs_df['label_int'].values
n = len(y_full)

B = 5000


def nested_select_fit_predict(X, y, alpha=0.05):
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X):
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        pd_mask, hc_mask = ytr == 1, ytr == 0
        keep = []
        for i in range(Xtr.shape[1]):
            if pd_mask.sum() >= 3 and hc_mask.sum() >= 3:
                try:
                    _, p = stats.mannwhitneyu(Xtr[pd_mask, i], Xtr[hc_mask, i])
                except ValueError:
                    p = 1.0
            else:
                p = 1.0
            if p < alpha:
                keep.append(i)
        if not keep:
            keep = list(range(Xtr.shape[1]))
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr[:, keep])
        Xte_s = sc.transform(Xte[:, keep])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr_s, ytr)
        yt.append(y[te][0])
        yp.append(clf.predict_proba(Xte_s)[0, 1])
    return roc_auc_score(yt, yp)


obs_auc = nested_select_fit_predict(X_full, y_full)
print(f"Observed AUC = {obs_auc:.4f}")
print(f"Running {B} full-procedure permutations...\n")

rng = np.random.RandomState(0)
perm_aucs = np.empty(B)
t0 = time.time()
n_exceed_running = 0

for b in range(B):
    y_perm = rng.permutation(y_full)
    perm_aucs[b] = nested_select_fit_predict(X_full, y_perm)
    if perm_aucs[b] >= obs_auc:
        n_exceed_running += 1

    if (b+1) % 250 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (b+1) * (B - b - 1)
        running_p = (1 + n_exceed_running) / (b + 2)
        print(f"  {b+1}/{B}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m  "
              f"running p~{running_p:.4f}", flush=True)

n_exceed = int(np.sum(perm_aucs >= obs_auc))
p_value = (1 + n_exceed) / (B + 1)

print(f"\n{'='*60}")
print(f"Observed AUC = {obs_auc:.4f}")
print(f"Permutation null: mean={perm_aucs.mean():.4f}, std={perm_aucs.std():.4f}")
print(f"Permutations >= observed: {n_exceed}/{B}")
print(f"p = {p_value:.4f}")
print(f"{'='*60}")

with open('permutation_test_full_results.pkl', 'wb') as f:
    pickle.dump({'obs_auc': obs_auc, 'perm_aucs': perm_aucs,
                 'p_value': p_value, 'B': B}, f)
print("\nSaved: permutation_test_full_results.pkl")
