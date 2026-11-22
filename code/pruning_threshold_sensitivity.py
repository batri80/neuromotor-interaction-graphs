import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from scipy import stats
import networkx as nx

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE, compute_te_matrix, extract_ngs
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8   # pre-specified reference value, unchanged

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15


def build_nig_custom_threshold(X, prune_pct):
    """Same as build_nig, but with a configurable prune percentile instead
    of the hardcoded 25th."""
    M = compute_te_matrix(X)
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


def build_subject_ngs(svc_data, prune_pct, n_windows=N_WINDOWS):
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
    G = build_nig_custom_threshold(Xz, prune_pct)
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
    print("PRUNING-THRESHOLD SENSITIVITY (N_WINDOWS=20, te_bins=8 fixed)")
    print("="*70)

    results = []
    for prune_pct in [10, 15, 20, 25, 30, 40]:
        t0 = time.time()
        rows = []
        for _, row in part_df.iterrows():
            ngs = build_subject_ngs(row['svc_data'], prune_pct)
            if ngs is None:
                continue
            out = {'label_int': 1 if row['label'] == 'PD' else 0}
            out.update({k: ngs.get(k, 0.0) for k in NGS_CORE})
            rows.append(out)
        df = pd.DataFrame(rows)
        X = df[NGS_CORE].values
        y = df['label_int'].values
        auc = nested_select_fit_predict(X, y)
        elapsed = time.time() - t0
        results.append((prune_pct, auc))
        note = " <-- reference (pre-specified, paper default)" if prune_pct == 25 else ""
        print(f"prune_pct={prune_pct:<4} LOPO AUC={auc:.4f}  ({elapsed:.1f}s){note}")

    aucs = [r[1] for r in results]
    print(f"\nRange: {min(aucs):.4f} - {max(aucs):.4f}  (spread: {max(aucs)-min(aucs):.4f})")
    print("\nSame discipline as the te_bins sweep: report the full spread honestly.")
    print("The pre-specified reference (25th pct) result is the one used everywhere")
    print("else in the paper -- do not substitute a different threshold post hoc")
    print("even if it scores higher here.")
