"""
Why is NewHandPD -> PaHaW transfer exactly AUC = 0.5000?

Re-runs that one transfer exactly as in third_dataset_transfer.py
(StandardScaler fit on the source, Logistic Regression, Phi10 features) and
reports:
  1. the distribution of the predicted probabilities on PaHaW (how many are
     identical / saturated at 0 or 1 in floating point);
  2. how far PaHaW's features fall outside NewHandPD's range after the
     source-fitted scaling (z-scores);
  3. the AUC computed from the decision function (log-odds) instead of the
     probabilities; the log-odds keep their ordering even when the
     probabilities round to exactly 0 or 1, so this AUC shows the model's
     actual ranking of PaHaW participants;
  4. for reference, the reverse direction (PaHaW -> NewHandPD) the same way.

Usage:  python r2_check_nhpd_to_pahaw.py
Output: r2_check_nhpd_to_pahaw.txt
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ngs_p4_newhandpd import CONFIG as CONFIG_NHPD, load_newhandpd
from ngs_pipeline_v2 import load_pahaw, CONFIG as CONFIG_PAHAW

CONFIG_PAHAW['pahaw_root'] = './PaHaW/PaHaW_public'
CONFIG_PAHAW['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
feat = CONFIG_PAHAW['feature_names']

log = open('r2_check_nhpd_to_pahaw.txt', 'w')
def out(s=''):
    print(s); log.write(s + '\n'); log.flush()

pa = load_pahaw(CONFIG_PAHAW['pahaw_root'], CONFIG_PAHAW['corpus_path'])
Xp, yp = pa[feat].values.astype(float), (pa['label'] == 'PD').astype(int).values
_, nh = load_newhandpd('./healthy', './patient')
Xn, yn = nh[feat].values.astype(float), nh['label_int'].values

def diagnose(name_tr, Xtr, ytr, name_te, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(Xtr), ytr)
    Zte = sc.transform(Xte)
    prob = clf.predict_proba(Zte)[:, 1]
    logit = clf.decision_function(Zte)
    out(f"\n{name_tr} -> {name_te}  (n_test={len(yte)})")
    out(f"  AUC from probabilities (as in the manuscript): {roc_auc_score(yte, prob):.4f}")
    out(f"  AUC from decision function (log-odds):         {roc_auc_score(yte, logit):.4f}")
    out(f"  probabilities: min={prob.min():.3e}  max={prob.max():.6f}  "
        f"distinct values={len(np.unique(prob))}/{len(prob)}")
    out(f"  exactly 1.0: {int(np.sum(prob == 1.0))}   exactly 0.0: {int(np.sum(prob == 0.0))}   "
        f"predicted PD (p>0.5): {int(np.sum(prob > 0.5))}")
    out(f"  log-odds: min={logit.min():.1f}  median={np.median(logit):.1f}  max={logit.max():.1f}")
    out("  target features after source scaling, median |z| per feature:")
    out("   " + "  ".join(f"{f}={np.median(np.abs(Zte[:, k])):.1f}" for k, f in enumerate(feat)))

diagnose("NewHandPD", Xn, yn, "PaHaW", Xp, yp)
diagnose("PaHaW", Xp, yp, "NewHandPD", Xn, yn)
log.close()
