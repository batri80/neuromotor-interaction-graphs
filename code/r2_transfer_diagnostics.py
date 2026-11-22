"""
Diagnostics for all six cross-dataset transfers (PaHaW, NewHandPD,
Parkinson's Drawings), Phi10 features, protocol unchanged from
third_dataset_transfer.py: StandardScaler fit on the source only, Logistic
Regression trained on the full source, evaluated on the full target.

For each ordered pair it reports
  AUC_prob   AUC from predicted probabilities (as in the manuscript; ties
             arise when probabilities round to exactly 0 or 1),
  AUC_logit  AUC from the decision function (log-odds), which is tie-free and
             shows the model's actual ranking,
  distinct   number of distinct probability values / n_test,
  sat        % of target predictions with probability exactly 0 or 1,
  maxmed|z|  largest per-feature median |z| of the target after source scaling
             (a measure of how far outside the training range the target lies).

Usage:  python r2_transfer_diagnostics.py
Output: r2_transfer_diagnostics.txt
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ngs_p4_newhandpd import load_newhandpd
from ngs_pipeline_v2 import load_pahaw, CONFIG as CONFIG_PAHAW
from third_dataset_transfer import load_third_dataset, THIRD_DATASET_ROOT

feat = CONFIG_PAHAW['feature_names']
log = open('r2_transfer_diagnostics.txt', 'w')
def out(s=''):
    print(s); log.write(s + '\n'); log.flush()

pa = load_pahaw('./PaHaW/PaHaW_public', './PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
_, nh = load_newhandpd('./healthy', './patient')
dr = load_third_dataset(THIRD_DATASET_ROOT)
data = {
    'PaHaW': (pa[feat].values.astype(float), (pa['label'] == 'PD').astype(int).values),
    'NewHandPD': (nh[feat].values.astype(float), nh['label_int'].values),
    'Drawings': (dr[feat].values.astype(float), dr['label_int'].values),
}

out(f"{'Train -> Test':<24}{'AUC_prob':>9}{'AUC_logit':>10}{'distinct':>10}{'sat':>7}{'maxmed|z|':>11}  feature")
for s_name, (Xs, ys) in data.items():
    for t_name, (Xt, yt) in data.items():
        if s_name == t_name:
            continue
        sc = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(Xs), ys)
        Zt = sc.transform(Xt)
        prob = clf.predict_proba(Zt)[:, 1]
        logit = clf.decision_function(Zt)
        medz = np.median(np.abs(Zt), axis=0)
        k = int(np.argmax(medz))
        sat = 100 * np.mean((prob == 0.0) | (prob == 1.0))
        out(f"{s_name + ' -> ' + t_name:<24}{roc_auc_score(yt, prob):>9.4f}"
            f"{roc_auc_score(yt, logit):>10.4f}{len(np.unique(prob)):>5}/{len(prob):<4}"
            f"{sat:>6.0f}%{medz[k]:>11.1f}  {feat[k]}")

out("\nWithin-dataset reference (5-fold stratified CV, scaler fit on training folds):")
from sklearn.model_selection import StratifiedKFold
for name, (X, y) in data.items():
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    out(f"  {name:<10} AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
log.close()
