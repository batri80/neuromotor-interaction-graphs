import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import networkx as nx

from ngs_pipeline_v2 import load_pahaw, CONFIG as CONFIG_PAHAW, extract_ngs, NGS_CORE
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG_PAHAW['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG_PAHAW['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG_PAHAW['te_threshold_pct'] = 25

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15


# ── Generic graph builder: same thresholding/graph logic, swappable edge method ──
def build_nig_generic(X, matrix_fn, feat_names):
    M = matrix_fn(X)
    nz = M[M > 0]
    thr = np.percentile(nz, CONFIG_PAHAW['te_threshold_pct']) if len(nz) > 0 else 0.0
    G = nx.DiGraph()
    for i, f in enumerate(feat_names):
        G.add_node(i, name=f)
    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            if i != j and M[i, j] > thr:
                G.add_edge(i, j, weight=M[i, j])
    return G


def pearson_matrix(X):
    n = X.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = np.corrcoef(X[:, i], X[:, j])[0, 1]
                M[i, j] = abs(r) if np.isfinite(r) else 0.0
    return M


def mi_matrix(X):
    n = X.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    mi = mutual_info_regression(X[:, i:i+1], X[:, j], random_state=0)[0]
                except Exception:
                    mi = 0.0
                M[i, j] = max(mi, 0.0)
    return M


from ngs_pipeline_v2 import build_nig as te_build_nig  # the real TE version, unchanged


def build_subject_ngs_method(svc_data, method, n_windows=N_WINDOWS):
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

    if method == 'TE':
        G = te_build_nig(Xz, label='te')
    elif method == 'pearson':
        G = build_nig_generic(Xz, pearson_matrix, KINEMATIC_KEYS)
    elif method == 'MI':
        G = build_nig_generic(Xz, mi_matrix, KINEMATIC_KEYS)
    return extract_ngs(G)


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
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(Xtr[:, keep]), ytr)
        yt.append(y[te][0])
        yp.append(clf.predict_proba(sc.transform(Xte[:, keep]))[0, 1])
    return roc_auc_score(yt, yp)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG_PAHAW['pahaw_root'], CONFIG_PAHAW['corpus_path'])

    print("="*80)
    print("EDGE-CONSTRUCTION COMPARISON: TE vs Pearson vs MI (real, not estimated)")
    print("Applied to the honest within-subject kinematic pipeline")
    print("="*80)

    for method in ['TE', 'pearson', 'MI']:
        t0 = time.time()
        rows = []
        for _, row in part_df.iterrows():
            ngs = build_subject_ngs_method(row['svc_data'], method)
            if ngs is None:
                continue
            out = {'label_int': 1 if row['label'] == 'PD' else 0}
            out.update({k: ngs.get(k, 0.0) for k in NGS_CORE})
            rows.append(out)
        df = pd.DataFrame(rows)
        X = df[NGS_CORE].values
        y = df['label_int'].values
        auc = nested_select_fit_predict(X, y)
        print(f"{method:<10} built in {time.time()-t0:.1f}s, n={len(df)}, "
              f"nested-selection LOPO AUC={auc:.4f}")

    print("\n" + "="*80)
    print("CROSS-DATASET TRANSFER TEST (shared Phi10 feature space)")
    print("Train on PaHaW, test on NewHandPD, and vice versa")
    print("="*80)

    from ngs_p4_newhandpd import load_newhandpd, CONFIG as CONFIG_NHPD
    CONFIG_NHPD['healthy_root'] = './healthy'
    CONFIG_NHPD['patient_root'] = './patient'
    _, part_df_nhpd = load_newhandpd(CONFIG_NHPD['healthy_root'], CONFIG_NHPD['patient_root'])

    feat_cols = CONFIG_PAHAW['feature_names']  # same Phi10 names in both files
    X_pahaw = part_df[feat_cols].values
    y_pahaw = (part_df['label'] == 'PD').astype(int).values
    X_nhpd = part_df_nhpd[feat_cols].values
    y_nhpd = part_df_nhpd['label_int'].values

    sc = StandardScaler().fit(X_pahaw)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(sc.transform(X_pahaw), y_pahaw)
    prob = clf.predict_proba(sc.transform(X_nhpd))[:, 1]
    auc_p2n = roc_auc_score(y_nhpd, prob)
    print(f"Train PaHaW -> Test NewHandPD:  AUC={auc_p2n:.4f}")

    sc2 = StandardScaler().fit(X_nhpd)
    clf2 = LogisticRegression(max_iter=1000, random_state=42)
    clf2.fit(sc2.transform(X_nhpd), y_nhpd)
    prob2 = clf2.predict_proba(sc2.transform(X_pahaw))[:, 1]
    auc_n2p = roc_auc_score(y_pahaw, prob2)
    print(f"Train NewHandPD -> Test PaHaW:  AUC={auc_n2p:.4f}")

    print(f"\nFor reference: within-dataset Phi10 baseline AUCs were "
          f"PaHaW=0.541 (paper), NewHandPD=0.907 (paper)")
