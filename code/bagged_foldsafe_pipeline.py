"""
Full fold-safe BAGGED population-graph pipeline.

Combines two components that were validated separately:
  1. Fold-safety: reference graphs built from TRAINING subjects only,
     per-subject features as insertion deltas (no label ever used to
     pick which reference a test subject compares against).
  2. Bagging: the reference graph itself is now a 50-bootstrap-resample
     AVERAGE (not a single-shot estimate) -- directly targeting the
     estimation-variance mechanism confirmed in test_bagging_mechanism.py
     (56-89% variance reduction on 6/7 metrics, both groups).

Cost tradeoff: the reference is bagged (50x), but each subject's
insertion-delta is a single-shot graph build against that stable
reference (not re-bagged per subject) -- keeps this tractable while
still implementing the confirmed mechanism.
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

B_BAG = 25


def bagged_reference_ngs(X, B, rng):
    n = len(X)
    accum = {k: [] for k in NGS_CORE}
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < 4:
            continue
        Xz = StandardScaler().fit_transform(X[idx])
        G = build_nig(Xz, label='boot')
        ngs = extract_ngs(G)
        for k in NGS_CORE:
            accum[k].append(ngs.get(k, 0.0))
    return {k: (np.mean(v) if v else 0.0) for k, v in accum.items()}


def insertion_delta(X_ref, x_subject, ref_ngs, keys):
    X_aug = np.vstack([X_ref, x_subject])
    Xz = StandardScaler().fit_transform(X_aug)
    G = build_nig(Xz, label='insert')
    ngs_aug = extract_ngs(G)
    return np.array([ngs_aug.get(k, 0.0) - ref_ngs.get(k, 0.0) for k in keys])


def build_fold_features(part_df, feat_cols, train_idx, test_idx, rng):
    """Bagged reference (from training only) + insertion delta for every
    subject in train_idx + test_idx. Returns DataFrame."""
    X_all = part_df[feat_cols].values
    labels = part_df['label'].values

    pd_train = [i for i in train_idx if labels[i] == 'PD']
    hc_train = [i for i in train_idx if labels[i] == 'HC']
    X_pd_train = X_all[pd_train]
    X_hc_train = X_all[hc_train]

    ngs_pd_ref = bagged_reference_ngs(X_pd_train, B_BAG, rng)
    ngs_hc_ref = bagged_reference_ngs(X_hc_train, B_BAG, rng)

    rows = []
    for i in list(train_idx) + list(test_idx):
        x_s = X_all[i:i+1]
        # leave-true-self-out for training-set members of a reference group
        if i in pd_train:
            X_pd_ref_i = X_all[[j for j in pd_train if j != i]]
            ref_i = bagged_reference_ngs(X_pd_ref_i, B_BAG, rng) if len(X_pd_ref_i) >= 4 else ngs_pd_ref
            d_pd = insertion_delta(X_pd_ref_i, x_s, ref_i, NGS_CORE)
            d_hc = insertion_delta(X_hc_train, x_s, ngs_hc_ref, NGS_CORE)
        elif i in hc_train:
            X_hc_ref_i = X_all[[j for j in hc_train if j != i]]
            ref_i = bagged_reference_ngs(X_hc_ref_i, B_BAG, rng) if len(X_hc_ref_i) >= 4 else ngs_hc_ref
            d_hc = insertion_delta(X_hc_ref_i, x_s, ref_i, NGS_CORE)
            d_pd = insertion_delta(X_pd_train, x_s, ngs_pd_ref, NGS_CORE)
        else:
            d_pd = insertion_delta(X_pd_train, x_s, ngs_pd_ref, NGS_CORE)
            d_hc = insertion_delta(X_hc_train, x_s, ngs_hc_ref, NGS_CORE)

        row = {'participant_id': part_df.iloc[i]['participant_id'],
               'label_int': 1 if part_df.iloc[i]['label'] == 'PD' else 0}
        for k, key in enumerate(NGS_CORE):
            row[f'{key}_dPD'] = d_pd[k]
            row[f'{key}_dHC'] = d_hc[k]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    feat_cols = CONFIG['feature_names']
    n = len(part_df)
    y_all = (part_df['label'] == 'PD').astype(int).values

    print("── Repeated 5-fold CV, bagged fold-safe reference ──")
    print("(This re-bags the reference from scratch every fold -- expect "
          "several minutes per fold given B=50 bootstrap graph builds x2 groups)\n")

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)  # single pass: 5 folds, fast pilot
    splits = list(rskf.split(np.zeros(n), y_all))

    fold_aucs = []
    all_t, all_p = [], []
    t0 = time.time()

    for fi, (train_idx, test_idx) in enumerate(splits):
        rng = np.random.RandomState(1000 + fi)
        feats = build_fold_features(part_df, feat_cols, list(train_idx), list(test_idx), rng)
        feat_cols_out = [c for c in feats.columns if c not in ('participant_id', 'label_int')]

        test_ids = set(part_df.iloc[list(test_idx)]['participant_id'])
        train_rows = feats[~feats['participant_id'].isin(test_ids)]
        test_rows = feats[feats['participant_id'].isin(test_ids)]

        Xtr, ytr = train_rows[feat_cols_out].values, train_rows['label_int'].values
        Xte, yte = test_rows[feat_cols_out].values, test_rows['label_int'].values

        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(Xtr), ytr)
        prob = clf.predict_proba(sc.transform(Xte))[:, 1]

        if len(np.unique(yte)) > 1:
            fold_aucs.append(roc_auc_score(yte, prob))
        all_t.extend(yte.tolist()); all_p.extend(prob.tolist())

        elapsed = time.time() - t0
        eta = elapsed / (fi+1) * (len(splits) - fi - 1)
        print(f"  fold {fi+1}/{len(splits)}  AUC={fold_aucs[-1]:.4f}  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    fold_aucs = np.array(fold_aucs)
    pooled_auc = roc_auc_score(all_t, all_p)
    print(f"\nMean AUC={fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}")
    print(f"Pooled AUC={pooled_auc:.4f}")
    print(f"\nFor reference: same architecture WITHOUT bagging (single-shot) gave "
          f"mean=0.488, pooled=0.496 (chance)")
