"""
Whole-recording kinematic baseline (no windowing, no TE, no graph) under
LOPO (reproduction of Table 11) and repeated 10x5 stratified CV, which
removes the leave-one-out class-composition effect.

Usage:  python r2_baseline_kfold.py [--pahaw ...] [--corpus ...]
Output: r2_baseline_kfold_results.txt
"""
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import ngs_pipeline_v2 as base
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

ap = argparse.ArgumentParser()
ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
a = ap.parse_args()
df = base.load_pahaw(a.pahaw, a.corpus)
X, y = [], []
for _, r in df.iterrows():
    fe = extract_kinematic_features(r['svc_data'])
    if fe is None: continue
    X.append([fe[k] for k in KINEMATIC_KEYS]); y.append(1 if r['label'] == 'PD' else 0)
X, y = np.array(X, float), np.array(y)
ok = np.all(np.isfinite(X), axis=1); X, y = X[ok], y[ok]
clfs = {'LR': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'RF': lambda: RandomForestClassifier(100, random_state=42),
        'SVM': lambda: SVC(kernel='rbf', probability=True, random_state=42)}
f = open('r2_baseline_kfold_results.txt', 'w')
def out(s=''): print(s); f.write(s + '\n')
out(f"n={len(y)} PD={y.sum()} HC={(y==0).sum()}  (Table 11: LR 0.4807, RF 0.5544, SVM 0.2905; NGS nested 10x5 = 0.6249)")
for name, mk in clfs.items():
    s = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        s[te] = make_pipeline(StandardScaler(), mk()).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    lopo = roc_auc_score(y, s)
    fold, oo_s, oo_y = [], [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(X, y):
        p = make_pipeline(StandardScaler(), mk()).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fold.append(roc_auc_score(y[te], p)); oo_s += list(p); oo_y += list(y[te])
    out(f"{name:<4} LOPO={lopo:.4f}   10x5 mean={np.mean(fold):.4f} sd={np.std(fold):.4f}  pooled={roc_auc_score(oo_y, oo_s):.4f}")
f.close()
