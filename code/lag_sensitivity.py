import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from scipy import stats
import networkx as nx

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE, transfer_entropy, extract_ngs
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8  # pre-specified reference, unchanged

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15
PRUNE_PCT = 25  # pre-specified reference, unchanged


def compute_te_matrix_with_lag(X, lag):
    """Same as compute_te_matrix, but with an explicit, configurable lag
    (the original function always used the transfer_entropy default lag=1)."""
    n = X.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = transfer_entropy(X[:, i], X[:, j], lag=lag)
    return M


def build_nig_custom_lag(X, lag, prune_pct=PRUNE_PCT):
    M = compute_te_matrix_with_lag(X, lag)
    nz = M[M > 0]
    thr = np.percentile(nz, prune_pct) if len(nz) > 0 else 0.0
    G = nx.DiGraph()
    for i in range(M.shape[0]):
        G.add_node(i)
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            if i != j and M[i, j] > thr:
                G.add_edge(i, j, weight=M[i, j])
    return G


def build_subject_ngs(svc_data, lag, n_windows=N_WINDOWS):
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
    if len(Xz) < lag + 2:  # guard, matches transfer_entropy's own internal check
        return None
    G = build_nig_custom_lag(Xz, lag)
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
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    print("="*70)
    print("TE-LAG SENSITIVITY (N_WINDOWS=20, te_bins=8, prune_pct=25 fixed)")
    print("="*70)

    results = []
    for lag in [1, 2, 3, 4, 5]:
        t0 = time.time()
        rows, n_excluded = [], 0
        for _, row in part_df.iterrows():
            ngs = build_subject_ngs(row['svc_data'], lag)
            if ngs is None:
                n_excluded += 1
                continue
            out = {'label_int': 1 if row['label'] == 'PD' else 0}
            out.update({k: ngs.get(k, 0.0) for k in NGS_CORE})
            rows.append(out)
        df = pd.DataFrame(rows)
        X = df[NGS_CORE].values
        y = df['label_int'].values
        auc = nested_select_fit_predict(X, y)
        elapsed = time.time() - t0
        results.append((lag, auc, len(df), n_excluded))
        note = " <-- reference (pre-specified, paper default)" if lag == 1 else ""
        print(f"lag={lag:<3} n={len(df):<3} excluded={n_excluded:<3} "
              f"LOPO AUC={auc:.4f}  ({elapsed:.1f}s){note}")

    aucs = [r[1] for r in results]
    print(f"\nRange: {min(aucs):.4f} - {max(aucs):.4f}  (spread: {max(aucs)-min(aucs):.4f})")
    print("\nSame discipline as the other sweeps: report the full spread honestly.")
    print("lag=1 is the pre-specified reference used everywhere else in the paper.")
