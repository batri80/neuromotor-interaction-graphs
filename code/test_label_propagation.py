"""
Transductive label propagation, evaluated honestly.

Legitimacy check (read before trusting the numbers):
  - The similarity graph over subjects is built from FEATURE VECTORS ONLY.
    No subject's true label is ever used to build the graph.
  - sklearn's LabelPropagation/LabelSpreading are inherently transductive:
    at fit time they see ALL feature vectors, with the held-out subject's
    label masked as -1. This is the standard, textbook use of these
    algorithms -- not a workaround.
  - Evaluation is still LOPO: exactly one label is hidden per iteration,
    predicted from the graph structure + all other known labels, exactly
    parallel to every other LOPO run today.
  - This produces a TRANSDUCTIVE result: it tells us how well label
    propagation recovers a hidden label when the full population's
    features are visible. It does NOT tell us how well a trained model
    would generalize to a brand-new, never-before-seen subject collected
    tomorrow (an inductive claim). Both are legitimate scientific
    questions -- but a transductive result must be reported and discussed
    as such, not conflated with the earlier inductive classifier numbers.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph

from ngs_pipeline_v2 import NGS_CORE

ngs_df = pd.read_pickle('within_subject_kinematic_ngs_df.pkl')
X_raw = ngs_df[NGS_CORE].values
y = ngs_df['label_int'].values
n = len(y)

X = StandardScaler().fit_transform(X_raw)


def lopo_label_propagation(X, y, method='propagation', kernel='knn', n_neighbors=7, gamma=20, alpha=0.8):
    yt, yp = [], []
    for i in range(len(y)):
        y_masked = y.copy()
        y_masked[i] = -1   # hide this subject's label -- graph itself is untouched
        if method == 'propagation':
            model = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, gamma=gamma, max_iter=1000)
        else:
            model = LabelSpreading(kernel=kernel, n_neighbors=n_neighbors, gamma=gamma, alpha=alpha, max_iter=1000)
        model.fit(X, y_masked)
        prob = model.predict_proba([X[i]])[0]
        # predict_proba columns follow model.classes_ order
        pd_col = list(model.classes_).index(1) if 1 in model.classes_ else None
        yt.append(y[i])
        yp.append(prob[pd_col] if pd_col is not None else 0.5)
    return roc_auc_score(yt, yp), np.array(yt), np.array(yp)


print("── Transductive label propagation, LOPO evaluation ──\n")

configs = [
    ("LabelPropagation, knn(k=7)",  dict(method='propagation', kernel='knn', n_neighbors=7)),
    ("LabelPropagation, knn(k=5)",  dict(method='propagation', kernel='knn', n_neighbors=5)),
    ("LabelPropagation, knn(k=10)", dict(method='propagation', kernel='knn', n_neighbors=10)),
    ("LabelPropagation, rbf",        dict(method='propagation', kernel='rbf', gamma=20)),
    ("LabelSpreading, knn(k=7)",     dict(method='spreading', kernel='knn', n_neighbors=7, alpha=0.8)),
    ("LabelSpreading, rbf",          dict(method='spreading', kernel='rbf', gamma=20, alpha=0.8)),
]

results = []
for name, kwargs in configs:
    auc, yt, yp = lopo_label_propagation(X, y, **kwargs)
    print(f"{name:<32} LOPO AUC={auc:.4f}")
    results.append((name, auc))

print(f"\nFor reference, today's other LOPO results:")
print(f"  Kinematic NGS + LR (baseline)        0.5718")
print(f"  Kinematic NGS + RF                    0.5961")
print(f"  Nested feature selection + LR         0.6150")

best = max(results, key=lambda r: r[1])
print(f"\nBest transductive config: {best[0]} (AUC={best[1]:.4f})")
print("\nIMPORTANT: this is a transductive result (see docstring). If it beats")
print("the inductive numbers, that's a real and reportable finding, but it")
print("answers a different question than 'will this generalize to a new")
print("subject' -- report it as such, not as a drop-in replacement.")
