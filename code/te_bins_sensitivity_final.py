import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from scipy import stats

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15


def build_subject_ngs(svc_data, n_windows=N_WINDOWS):
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
    G = build_nig(Xz, label='kin')
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
    print(f"TE-BIN SENSITIVITY, N_WINDOWS={N_WINDOWS} fixed, corrected kinematic pipeline")
    print("="*70)

    results = []
    for te_bins in [3, 4, 5, 6, 8]:
        CONFIG['te_bins'] = te_bins
        t0 = time.time()
        rows = []
        for _, row in part_df.iterrows():
            ngs = build_subject_ngs(row['svc_data'])
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
        results.append((te_bins, auc, len(df)))
        print(f"te_bins={te_bins:<3} n={len(df):<3} LOPO AUC={auc:.4f}  ({elapsed:.1f}s)")

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'te_bins':<10}{'LOPO AUC':<12}{'note'}")
    for te_bins, auc, n in results:
        note = "reference (paper default)" if te_bins == 8 else ""
        print(f"{te_bins:<10}{auc:<12.4f}{note}")

    aucs = [r[1] for r in results]
    print(f"\nRange: {min(aucs):.4f} - {max(aucs):.4f}  (spread: {max(aucs)-min(aucs):.4f})")
