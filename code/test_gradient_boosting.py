import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS
from ngs_pipeline_v2 import compute_te_matrix, build_nig, extract_ngs

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8

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
    G = build_nig(Xz, label='s')
    return extract_ngs(G)


def gb_lopo_unselected(X, y):
    """Gradient Boosting, all 7 NGS features, no selection -- matches how
    LR/RF/SVM were tested as 'unselected' baselines in the Robustness table."""
    loo = LeaveOneOut()
    yt, yp, yhat = [], [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        yt.append(y[te][0])
        prob = clf.predict_proba(sc.transform(X[te]))[0, 1]
        yp.append(prob)
        yhat.append(int(prob >= 0.5))
    auc = roc_auc_score(yt, yp)
    acc = accuracy_score(yt, yhat)
    return auc, acc


def gb_lopo_nested_selection(X, y, alpha=0.05):
    """Gradient Boosting WITH our nested fold-safe feature selection --
    the direct, apples-to-apples comparison against our primary
    nested-selection + LR result (AUC=0.6150)."""
    loo = LeaveOneOut()
    yt, yp, yhat = [], [], []
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
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(sc.fit_transform(Xtr[:, keep]), ytr)
        yt.append(y[te][0])
        prob = clf.predict_proba(sc.transform(Xte[:, keep]))[0, 1]
        yp.append(prob)
        yhat.append(int(prob >= 0.5))
    auc = roc_auc_score(yt, yp)
    acc = accuracy_score(yt, yhat)
    return auc, acc


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    print("Building within-subject NGS features (same pipeline as primary result)...")
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
    print(f"n={len(df)}\n")

    print("="*70)
    print("GRADIENT BOOSTING on within-subject NGS, LOPO-CV")
    print("(same protocol used throughout this paper's Robustness table)")
    print("="*70)

    auc_un, acc_un = gb_lopo_unselected(X, y)
    print(f"Gradient Boosting, unselected (7 features):  AUC={auc_un:.4f}  Acc={acc_un:.4f}")

    auc_ns, acc_ns = gb_lopo_nested_selection(X, y)
    print(f"Gradient Boosting, nested selection:          AUC={auc_ns:.4f}  Acc={acc_ns:.4f}")

    print(f"\nFor reference, this paper's existing results on the same representation:")
    print(f"  Within-subject NGS, Logistic Regression (unselected):  AUC=0.5718")
    print(f"  Within-subject NGS, Random Forest (unselected):        AUC=0.5961")
    print(f"  Within-subject NGS, SVM RBF (unselected):               AUC=0.5552")
    print(f"  Within-subject NGS, nested selection + LR (best):       AUC=0.6150")
    print(f"\nKashid & Deshpande (2026), their own pipeline, PaHaW:")
    print(f"  Gradient Boosting (their features, their protocol):     Acc=0.696")
    print("\nThis test isolates classifier choice from feature representation:")
    print("same NGS features, same LOPO protocol, only the classifier differs.")
    print("If GB underperforms our nested-selection LR here, that's evidence")
    print("their stronger result reflects their feature set / protocol, not")
    print("Gradient Boosting being intrinsically better suited to this problem.")
