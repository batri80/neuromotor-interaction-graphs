import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'


def evaluate(X, y, clf_factory, label=""):
    """Standard LOPO + repeated 10x5-fold, no feature selection inside."""
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        clf = clf_factory()
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        yt.append(y[te][0])
        try:
            yp.append(clf.predict_proba(sc.transform(X[te]))[0, 1])
        except AttributeError:
            yp.append(clf.decision_function(sc.transform(X[te]))[0])
    lopo_auc = roc_auc_score(yt, yp)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs, all_t, all_p = [], [], []
    for tr, te in rskf.split(X, y):
        sc = StandardScaler()
        clf = clf_factory()
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        try:
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
        except AttributeError:
            prob = clf.decision_function(sc.transform(X[te]))
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], prob))
        all_t.extend(y[te].tolist()); all_p.extend(prob.tolist())
    kfold_pooled = roc_auc_score(all_t, all_p)

    print(f"{label:<45} LOPO AUC={lopo_auc:.4f}   K-fold pooled AUC={kfold_pooled:.4f}")
    return lopo_auc, kfold_pooled


def evaluate_nested_feature_selection(X, y, feature_names, alpha=0.05):
    """Feature selection done WITHIN each training fold only -- no leakage."""
    def select_and_fit(Xtr, ytr, Xte):
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
            keep = list(range(Xtr.shape[1]))  # fallback: keep all if none significant
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr[:, keep])
        Xte_s = sc.transform(Xte[:, keep])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr_s, ytr)
        return clf, Xte_s, keep

    loo = LeaveOneOut()
    yt, yp, selected_counts = [], [], []
    for tr, te in loo.split(X):
        clf, Xte_s, keep = select_and_fit(X[tr], y[tr], X[te])
        yt.append(y[te][0])
        yp.append(clf.predict_proba(Xte_s)[0, 1])
        selected_counts.append(len(keep))
    lopo_auc = roc_auc_score(yt, yp)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs, all_t, all_p = [], [], []
    for tr, te in rskf.split(X, y):
        clf, Xte_s, keep = select_and_fit(X[tr], y[tr], X[te])
        prob = clf.predict_proba(Xte_s)[:, 1]
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], prob))
        all_t.extend(y[te].tolist()); all_p.extend(prob.tolist())
    kfold_pooled = roc_auc_score(all_t, all_p)

    print(f"{'Nested feature selection (LR, per-fold)':<45} "
          f"LOPO AUC={lopo_auc:.4f}   K-fold pooled AUC={kfold_pooled:.4f}   "
          f"(avg features kept: {np.mean(selected_counts):.1f}/{X.shape[1]})")
    return lopo_auc, kfold_pooled


if __name__ == '__main__':
    print("="*90)
    print("EXPERIMENT B: Random Forest / SVM on existing kinematic NGS (N_WINDOWS=20)")
    print("="*90)
    ngs_df = pd.read_pickle('within_subject_kinematic_ngs_df.pkl')
    X = ngs_df[NGS_CORE].values
    y = ngs_df['label_int'].values

    evaluate(X, y, lambda: LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression (baseline)")
    evaluate(X, y, lambda: RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
    evaluate(X, y, lambda: SVC(kernel='rbf', probability=True, random_state=42), "SVM (RBF)")

    print("\n" + "="*90)
    print("EXPERIMENT A: Nested feature selection (fold-safe, no leakage)")
    print("="*90)
    evaluate_nested_feature_selection(X, y, NGS_CORE)

    print("\n" + "="*90)
    print("EXPERIMENT C: N_WINDOWS=30 rebuild")
    print("="*90)
    CONFIG['te_bins'] = 8
    N_WINDOWS_30 = 30
    MIN_SAMPLES_PER_WINDOW = 15

    def build_ngs(svc_data, n_windows):
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
        Xw = np.array(feat_rows)
        if not np.all(np.isfinite(Xw)):
            return None
        Xz = StandardScaler().fit_transform(Xw)
        G = build_nig(Xz, label='k30')
        return extract_ngs(G)

    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    t0 = time.time()
    rows30, n_exc = [], 0
    for _, row in part_df.iterrows():
        ngs = build_ngs(row['svc_data'], N_WINDOWS_30)
        if ngs is None:
            n_exc += 1
            continue
        out = {'label_int': 1 if row['label']=='PD' else 0}
        out.update({k: ngs.get(k, 0.0) for k in NGS_CORE})
        rows30.append(out)
    ngs30_df = pd.DataFrame(rows30)
    print(f"Built in {(time.time()-t0)/60:.1f} min. {len(ngs30_df)}/{len(part_df)} included, {n_exc} excluded.")
    X30 = ngs30_df[NGS_CORE].values
    y30 = ngs30_df['label_int'].values
    evaluate(X30, y30, lambda: LogisticRegression(max_iter=1000, random_state=42), "N_WINDOWS=30, LR")

    print("\n" + "="*90)
    print("EXPERIMENT D: Hybrid (kinematic NGS + per-subject Phi10 scalars)")
    print("="*90)
    feat_cols_phi10 = CONFIG['feature_names']
    part_df_idx = part_df.set_index('participant_id')
    hybrid_rows = []
    for _, row in ngs_df.iterrows():
        pid = row['participant_id']
        if pid not in part_df_idx.index:
            continue
        phi10_vals = part_df_idx.loc[pid, feat_cols_phi10]
        out = {'label_int': row['label_int']}
        out.update({k: row[k] for k in NGS_CORE})
        out.update({f'phi10_{c}': phi10_vals[c] for c in feat_cols_phi10})
        hybrid_rows.append(out)
    hybrid_df = pd.DataFrame(hybrid_rows)
    hybrid_cols = NGS_CORE + [f'phi10_{c}' for c in feat_cols_phi10]
    Xh = hybrid_df[hybrid_cols].values
    yh = hybrid_df['label_int'].values
    print(f"Hybrid feature count: {Xh.shape[1]} ({len(NGS_CORE)} NGS + {len(feat_cols_phi10)} Phi10)")
    evaluate(Xh, yh, lambda: LogisticRegression(max_iter=1000, random_state=42), "Hybrid, LR")
    evaluate(Xh, yh, lambda: RandomForestClassifier(n_estimators=100, random_state=42), "Hybrid, Random Forest")

    print("\n" + "="*90)
    print("SUMMARY -- compare all honestly against baseline LOPO=0.5718 / K-fold pooled=0.5916")
    print("="*90)
