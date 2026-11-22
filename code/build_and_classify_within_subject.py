import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from ngs_pipeline_v2 import (
    load_pahaw, CONFIG, render_svc_quartile, extract_phi10,
    build_nig, extract_ngs, NGS_CORE
)

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 6      # confirmed via sweep + worst-case stress test

N_WINDOWS = 10
MIN_SAMPLES_PER_WINDOW = 50


def build_within_subject_ngs(svc_data, n_windows=N_WINDOWS, size=(512, 512)):
    if svc_data is None or len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None
    feat_rows = []
    for wi in range(n_windows):
        img = render_svc_quartile(svc_data, wi, n_q=n_windows, size=size)
        feats = extract_phi10(img)
        feat_rows.append([feats[k] for k in
                           ['Eb', 'Eg', 'Em', 'sigma_d', 'Ag', 'rho_s', 'Hs', 'Hg', 'SIA', 'Omega']])
    X_windows = np.array(feat_rows)
    X_z = StandardScaler().fit_transform(X_windows)
    G = build_nig(X_z, label='within_subject')
    return extract_ngs(G)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    print(f"Building within-subject NGS for all {len(part_df)} subjects "
          f"(N_WINDOWS={N_WINDOWS}, te_bins={CONFIG['te_bins']})...")
    t0 = time.time()

    rows = []
    n_excluded = 0
    for _, row in part_df.iterrows():
        ngs = build_within_subject_ngs(row['svc_data'])
        if ngs is None:
            n_excluded += 1
            print(f"  EXCLUDED (too short): {row['participant_id']} ({row['label']})")
            continue
        out = {'participant_id': row['participant_id'], 'label': row['label'],
               'label_int': 1 if row['label'] == 'PD' else 0}
        out.update({k: ngs.get(k, 0.0) for k in NGS_CORE})
        rows.append(out)

    ngs_ws_df = pd.DataFrame(rows)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"{len(ngs_ws_df)}/{len(part_df)} subjects included, {n_excluded} excluded.\n")

    ngs_ws_df.to_pickle('within_subject_ngs_df.pkl')

    # Sanity: PD/HC counts after exclusion
    print(ngs_ws_df['label'].value_counts(), "\n")

    X = ngs_ws_df[NGS_CORE].values
    y = ngs_ws_df['label_int'].values

    # ── Standard LOPO-CV (valid now -- no cross-subject leakage possible) ──
    print("Running standard LOPO-CV (should take seconds, not minutes)...")
    loo = LeaveOneOut()
    y_true, y_prob = [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr, y[tr])
        y_true.append(y[te][0])
        y_prob.append(clf.predict_proba(Xte)[0, 1])

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens, spec = tp/(tp+fn), tn/(tn+fp)
    print(f"LOPO-CV: AUC={auc:.4f}  Acc={acc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}\n")

    # ── Standard repeated 10x5-fold CV (valid now, and cheap) ──
    print("Running repeated 10x5-fold CV...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs = []
    all_true, all_prob = [], []
    for tr, te in rskf.split(X, y):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], prob))
        all_true.extend(y[te].tolist())
        all_prob.extend(prob.tolist())

    fold_aucs = np.array(fold_aucs)
    pooled_auc = roc_auc_score(all_true, all_prob)
    print(f"Repeated 10x5-fold: mean AUC={fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}  "
          f"pooled AUC={pooled_auc:.4f}\n")

    with open('within_subject_results.pkl', 'wb') as f:
        pickle.dump({
            'lopo': {'auc': auc, 'acc': acc, 'sens': sens, 'spec': spec,
                     'y_true': y_true, 'y_prob': y_prob},
            'kfold': {'mean_auc': fold_aucs.mean(), 'std_auc': fold_aucs.std(),
                      'pooled_auc': pooled_auc, 'fold_aucs': fold_aucs},
            'n_excluded': n_excluded, 'n_included': len(ngs_ws_df),
        }, f)

    print("Saved: within_subject_ngs_df.pkl, within_subject_results.pkl")
