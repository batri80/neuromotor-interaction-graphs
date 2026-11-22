import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy import stats

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8      # paper's original default; corrected features may tolerate it

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15


def build_kinematic_within_subject_ngs(svc_data, n_windows=N_WINDOWS):
    if svc_data is None or len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None
    n = len(svc_data)
    win_size = n // n_windows
    feat_rows = []
    for wi in range(n_windows):
        start = wi * win_size
        end = start + win_size if wi < n_windows - 1 else n
        feats = extract_kinematic_features(svc_data[start:end])
        if feats is None:
            return None
        feat_rows.append([feats[k] for k in KINEMATIC_KEYS])
    X_windows = np.array(feat_rows)
    if not np.all(np.isfinite(X_windows)):
        return None
    X_z = StandardScaler().fit_transform(X_windows)
    G = build_nig(X_z, label='kinematic_within_subject')
    return extract_ngs(G)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    part_df['svc_len'] = part_df['svc_data'].apply(lambda s: len(s) if s is not None else 0)

    # ── Worst-case stress test ──
    print("── Stress test: 8 shortest eligible recordings ──\n")
    eligible = part_df[part_df['svc_len'] >= N_WINDOWS * MIN_SAMPLES_PER_WINDOW]
    shortest = eligible.nsmallest(8, 'svc_len')
    n_degenerate = 0
    for _, row in shortest.iterrows():
        ngs = build_kinematic_within_subject_ngs(row['svc_data'])
        vals = [ngs.get(k, 0.0) for k in NGS_CORE] if ngs else []
        is_deg = ngs is None or all(abs(v) < 1e-9 for v in vals)
        if is_deg:
            n_degenerate += 1
        flag = "  <-- DEGENERATE/FAILED" if is_deg else ""
        print(f"  {row['participant_id']} ({row['label']}, {row['svc_len']} samples){flag}")
    print(f"\nDegenerate: {n_degenerate}/8\n")

    if n_degenerate > 0:
        print("STOPPING: stress test failed. Fix before proceeding to full build.")
        exit(1)

    # ── Full 72-subject build ──
    print("── Building within-subject kinematic NGS for all subjects ──")
    t0 = time.time()
    rows, n_excluded = [], 0
    for _, row in part_df.iterrows():
        ngs = build_kinematic_within_subject_ngs(row['svc_data'])
        if ngs is None:
            n_excluded += 1
            continue
        out = {'participant_id': row['participant_id'], 'label': row['label'],
               'label_int': 1 if row['label'] == 'PD' else 0}
        out.update({k: ngs.get(k, 0.0) for k in NGS_CORE})
        rows.append(out)

    ngs_df = pd.DataFrame(rows)
    print(f"Done in {(time.time()-t0)/60:.1f} min. "
          f"{len(ngs_df)}/{len(part_df)} included, {n_excluded} excluded.")
    print(ngs_df['label'].value_counts(), "\n")
    ngs_df.to_pickle('within_subject_kinematic_ngs_df.pkl')

    # ── Univariate signal check (most informative diagnostic from before) ──
    print("── Univariate raw-feature AUC (no model, no CV) ──")
    y = ngs_df['label_int'].values
    for k in NGS_CORE:
        x = ngs_df[k].values
        try:
            auc_raw = roc_auc_score(y, x)
        except Exception:
            auc_raw = float('nan')
        r, p = stats.mannwhitneyu(ngs_df[ngs_df.label=='PD'][k], ngs_df[ngs_df.label=='HC'][k])
        print(f"{k:<14} raw AUC={auc_raw:.3f}   Mann-Whitney p={p:.4f}")
    print()

    # ── Classification ──
    X = ngs_df[NGS_CORE].values

    print("── LOPO-CV ──")
    loo = LeaveOneOut()
    y_true, y_prob = [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        y_true.append(y[te][0])
        y_prob.append(clf.predict_proba(sc.transform(X[te]))[0, 1])
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"AUC={auc:.4f}  Acc={acc:.4f}  Sens={tp/(tp+fn):.4f}  Spec={tn/(tn+fp):.4f}\n")

    print("── Repeated 10x5-fold CV ──")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs, all_t, all_p = [], [], []
    for tr, te in rskf.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], prob))
        all_t.extend(y[te].tolist()); all_p.extend(prob.tolist())
    fold_aucs = np.array(fold_aucs)
    print(f"Mean AUC={fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}  "
          f"Pooled AUC={roc_auc_score(all_t, all_p):.4f}")

    with open('within_subject_kinematic_results.pkl', 'wb') as f:
        pickle.dump({'lopo_auc': auc, 'lopo_acc': acc,
                     'kfold_mean': fold_aucs.mean(), 'kfold_std': fold_aucs.std(),
                     'kfold_pooled': roc_auc_score(all_t, all_p)}, f)
