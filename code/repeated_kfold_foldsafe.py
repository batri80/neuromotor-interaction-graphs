"""
Fold-safe repeated Stratified K-Fold CV -- faster primary protocol.
Same leakage/circularity guarantees as lopo_cv_fold_safe, but O(n_folds)
reference-graph rebuilds instead of O(n_subjects), so it's ~7x cheaper
for n=72, k=10.

Also prints progress so a long run is visible rather than silent.
"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import time

from foldsafe_ngs import build_fold_safe_ngs


def repeated_kfold_foldsafe(part_df, feat_cols, ngs_keys, build_nig_fn, extract_ngs_fn,
                             clf_factory, n_splits=5, n_repeats=10, random_state=42,
                             verbose=True):
    """
    clf_factory: zero-arg callable returning a fresh sklearn classifier
                 (e.g. lambda: LogisticRegression(max_iter=1000, random_state=42))

    Returns dict with per-repeat-fold AUCs and pooled predictions.
    """
    y_all = part_df['label_int'].values
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=random_state)
    splits = list(rskf.split(np.zeros(len(part_df)), y_all))
    n_total = len(splits)

    fold_aucs = []
    all_true, all_prob = [], []
    t0 = time.time()

    for fi, (train_idx, test_idx) in enumerate(splits):
        feats = build_fold_safe_ngs(part_df, feat_cols, ngs_keys,
                                     build_nig_fn, extract_ngs_fn,
                                     train_idx=list(train_idx), test_idx=list(test_idx))
        feat_cols_out = [c for c in feats.columns if c not in
                         ('participant_id', 'label', 'label_int')]

        test_ids = set(part_df.iloc[list(test_idx)]['participant_id'])
        train_rows = feats[~feats['participant_id'].isin(test_ids)]
        test_rows  = feats[feats['participant_id'].isin(test_ids)]

        Xtr, ytr = train_rows[feat_cols_out].values, train_rows['label_int'].values
        Xte, yte = test_rows[feat_cols_out].values,  test_rows['label_int'].values

        sc = StandardScaler()
        clf = clf_factory()
        clf.fit(sc.fit_transform(Xtr), ytr)
        try:
            prob = clf.predict_proba(sc.transform(Xte))[:, 1]
        except Exception:
            prob = clf.predict(sc.transform(Xte)).astype(float)

        auc = roc_auc_score(yte, prob) if len(np.unique(yte)) > 1 else np.nan
        fold_aucs.append(auc)
        all_true.extend(yte.tolist())
        all_prob.extend(prob.tolist())

        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (fi + 1) * (n_total - fi - 1)
            print(f"  fold {fi+1}/{n_total}  AUC={auc:.4f}  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    fold_aucs = np.array(fold_aucs)
    pooled_auc = roc_auc_score(all_true, all_prob)

    return {
        'fold_aucs': fold_aucs,
        'mean_auc': np.nanmean(fold_aucs),
        'std_auc': np.nanstd(fold_aucs),
        'pooled_auc': pooled_auc,
        'y_true_pooled': np.array(all_true),
        'y_prob_pooled': np.array(all_prob),
    }
