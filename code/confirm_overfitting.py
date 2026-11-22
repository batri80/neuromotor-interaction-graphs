import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from ngs_pipeline_v2 import NGS_CORE

ngs_ws_df = pd.read_pickle('within_subject_ngs_df.pkl')
X = ngs_ws_df[NGS_CORE].values
y = ngs_ws_df['label_int'].values

loo = LeaveOneOut()

for C in [1.0, 0.1, 0.01, 0.001]:
    y_true, y_prob = [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(Xtr, y[tr])
        y_true.append(y[te][0])
        y_prob.append(clf.predict_proba(Xte)[0, 1])
    auc = roc_auc_score(y_true, y_prob)
    print(f"C={C:<8} (stronger reg. as C decreases)  LOPO AUC={auc:.4f}")

print("\nIf AUC climbs back toward 0.5 as C decreases (stronger regularization),")
print("that confirms overfitting-on-noise was driving the below-chance result.")
print("If AUC stays stubbornly below 0.5 even under heavy regularization,")
print("that would need a different explanation.")
