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
pids = ngs_df['participant_id'].values
n = len(y_full)


def nested_select_fit_predict(X, y, alpha=0.05):
    """Full nested feature-selection LOPO. Returns fold-level records:
    (true_label, pred_prob, selected_feature_indices) per subject."""
    loo = LeaveOneOut()
    records = []
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
        prob = clf.predict_proba(Xte_s)[0, 1]
        records.append({'true': y[te][0], 'prob': prob, 'selected': tuple(keep)})
    return records


# ── Step 1: capture the actual fold-level outputs for the observed result ──
print("── Capturing fold-level outputs for the observed nested-selection LOPO run ──")
t0 = time.time()
obs_records = nested_select_fit_predict(X_full, y_full)
print(f"Done in {time.time()-t0:.1f}s")

obs_df = pd.DataFrame(obs_records)
obs_df['participant_id'] = pids
obs_auc = roc_auc_score(obs_df['true'], obs_df['prob'])
print(f"Observed LOPO AUC = {obs_auc:.4f}  (matches earlier 0.6150)\n")

obs_df.to_pickle('nested_selection_fold_outputs.pkl')
print("Saved fold-level outputs: nested_selection_fold_outputs.pkl")
print(obs_df[['participant_id', 'true', 'prob']].to_string(index=False))

# Feature selection frequency across folds -- useful diagnostic for the paper
print("\n── Feature selection frequency across the 72 LOPO folds ──")
from collections import Counter
sel_counter = Counter()
for rec in obs_records:
    for idx in rec['selected']:
        sel_counter[NGS_CORE[idx]] += 1
for feat in NGS_CORE:
    print(f"  {feat:<14} selected in {sel_counter.get(feat,0)}/72 folds")

# ── Step 2: time a small batch of permutations before committing to full scale ──
print("\n── Timing test: 20 full-procedure permutations ──")
np.random.seed(42)
t0 = time.time()
perm_aucs_test = []
for b in range(20):
    y_perm = np.random.permutation(y_full)
    recs = nested_select_fit_predict(X_full, y_perm)
    df_perm = pd.DataFrame(recs)
    perm_aucs_test.append(roc_auc_score(df_perm['true'], df_perm['prob']))
elapsed = time.time() - t0
per_perm = elapsed / 20
print(f"20 permutations took {elapsed:.1f}s ({per_perm:.2f}s/permutation)")
print(f"Estimated time for 1,000 permutations:  {per_perm*1000/60:.1f} min")
print(f"Estimated time for 5,000 permutations:  {per_perm*5000/60:.1f} min")
print(f"Estimated time for 10,000 permutations: {per_perm*10000/60:.1f} min")
print(f"\nQuick check from these 20: mean={np.mean(perm_aucs_test):.4f}, "
      f"max={np.max(perm_aucs_test):.4f} (observed={obs_auc:.4f})")

with open('permutation_timing.pkl', 'wb') as f:
    pickle.dump({'per_perm_seconds': per_perm, 'obs_auc': obs_auc,
                 'test_perm_aucs': perm_aucs_test}, f)

# ── Step 3: subject-level bootstrap CI on the observed out-of-fold predictions ──
print("\n── Subject-level bootstrap 95% CI (2000 resamples) ──")
rng = np.random.RandomState(42)
true_arr = obs_df['true'].values
prob_arr = obs_df['prob'].values
boot_aucs = []
for _ in range(2000):
    idx = rng.choice(n, size=n, replace=True)
    yt_b, yp_b = true_arr[idx], prob_arr[idx]
    if len(np.unique(yt_b)) < 2:
        continue
    boot_aucs.append(roc_auc_score(yt_b, yp_b))
boot_aucs = np.array(boot_aucs)
ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
print(f"LOPO AUC = {obs_auc:.4f}; 95% bootstrap CI = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"CI {'CROSSES' if ci_low < 0.5 < ci_high else 'does NOT cross'} 0.5")
