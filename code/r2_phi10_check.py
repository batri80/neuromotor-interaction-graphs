"""
Why is within-PaHaW Phi10 AUC below chance (0.360 +/- 0.023, single 5-fold
split)? Reports, for the rendered-image Phi10 features:
  1. the original single 5-fold value (reproduction),
  2. repeated 10x5-fold: mean of fold AUCs and pooled out-of-fold AUC,
  3. the same with a strongly regularised LR (C=0.01) and with RF,
  4. univariate AUC of every feature on the whole sample (orientation check),
  5. how often each LR coefficient changes sign across folds.

Usage:  python r2_phi10_check.py [--pahaw ...] [--corpus ...]
Output: r2_phi10_check_results.txt
"""
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import ngs_pipeline_v2 as base
from ngs_pipeline_v2 import CONFIG

ap = argparse.ArgumentParser()
ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
a = ap.parse_args()
df = base.load_pahaw(a.pahaw, a.corpus)
feat = CONFIG['feature_names']
X = df[feat].values.astype(float); y = (df.label == 'PD').astype(int).values
f = open('r2_phi10_check_results.txt', 'w')
def out(s=''): print(s); f.write(s + '\n')
out(f"n={len(y)} PD={y.sum()} HC={(y==0).sum()}")

def run(cv, mk, score):
    fold, oof_s, oof_y, coefs = [], [], [], []
    for tr, te in cv.split(X, y):
        m = mk().fit(X[tr], y[tr])
        s = score(m, X[te]); fold.append(roc_auc_score(y[te], s))
        oof_s += list(s); oof_y += list(y[te])
        if hasattr(m[-1], 'coef_'): coefs.append(m[-1].coef_[0])
    return np.mean(fold), np.std(fold), roc_auc_score(oof_y, oof_s), np.array(coefs)

lr = lambda C=1.0: (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=42)))
dec = lambda m, Z: m.decision_function(Z)
prob = lambda m, Z: m.predict_proba(Z)[:, 1]
m0, s0, p0, _ = run(StratifiedKFold(5, shuffle=True, random_state=42), lr(), dec)
out(f"1. single 5-fold (as reported): mean={m0:.3f} sd={s0:.3f}  pooled={p0:.3f}")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
m1, s1, p1, C1 = run(rcv, lr(), dec)
out(f"2. repeated 10x5, LR C=1:    mean={m1:.3f} sd={s1:.3f}  pooled-oof={p1:.3f}")
m2, s2, p2, _ = run(rcv, lr(0.01), dec)
out(f"3a. repeated 10x5, LR C=0.01: mean={m2:.3f} sd={s2:.3f}  pooled-oof={p2:.3f}")
m3, s3, p3, _ = run(rcv, lambda: make_pipeline(StandardScaler(), RandomForestClassifier(300, random_state=42)), prob)
out(f"3b. repeated 10x5, RF:         mean={m3:.3f} sd={s3:.3f}  pooled-oof={p3:.3f}")
out("4. univariate whole-sample AUC (PD positive; <0.5 means lower in PD):")
for k, n in enumerate(feat):
    out(f"   {n:<8} {roc_auc_score(y, X[:, k]):.3f}")
out("5. LR coefficient sign changes across the 50 folds (share of folds with the minority sign):")
for k, n in enumerate(feat):
    pos = np.mean(C1[:, k] > 0); out(f"   {n:<8} {min(pos, 1-pos):.2f}")
f.close()
