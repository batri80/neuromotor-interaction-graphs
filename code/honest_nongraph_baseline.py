import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from ngs_pipeline_v2 import load_pahaw, CONFIG
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'


def whole_recording_features(svc_data):
    """No windowing, no TE, no graph -- just the 10 kinematic features
    computed once over the ENTIRE recording. This is the fairest possible
    'no graph transformation' baseline: same raw kinematic signal, same
    feature definitions, just without the within-subject TE/NGS step."""
    if svc_data is None or len(svc_data) < 20:
        return None
    feats = extract_kinematic_features(svc_data)
    if feats is None:
        return None
    return [feats[k] for k in KINEMATIC_KEYS]


def evaluate(X, y, clf_factory, label):
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        clf = clf_factory()
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        yt.append(y[te][0])
        try:
            yp.append(clf.predict_proba(sc.transform(X[te]))[0, 1])
        except AttributeError:
            yp.append(clf.decision_function(sc.transform(X[te]))[0])
    auc = roc_auc_score(yt, yp)
    print(f"{label:<45} LOPO AUC={auc:.4f}")
    return auc


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    rows = []
    for _, row in part_df.iterrows():
        feats = whole_recording_features(row['svc_data'])
        if feats is None:
            continue
        rows.append({'label_int': 1 if row['label'] == 'PD' else 0,
                     **{k: v for k, v in zip(KINEMATIC_KEYS, feats)}})
    df = pd.DataFrame(rows)
    X = df[KINEMATIC_KEYS].values
    y = df['label_int'].values

    print("="*70)
    print("HONEST NON-GRAPH BASELINE")
    print("Same 10 kinematic features, whole-recording aggregate, NO windowing,")
    print("NO Transfer Entropy, NO graph construction at all.")
    print(f"n={len(df)}")
    print("="*70)

    evaluate(X, y, lambda: LogisticRegression(max_iter=1000, random_state=42), "Whole-recording kinematic, LR")
    evaluate(X, y, lambda: RandomForestClassifier(n_estimators=100, random_state=42), "Whole-recording kinematic, RF")
    evaluate(X, y, lambda: SVC(kernel='rbf', probability=True, random_state=42), "Whole-recording kinematic, SVM")

    print(f"\nFor reference, the within-subject NGS (graph) results:")
    print(f"  Nested-selection LR (best):     0.6150")
    print(f"  Unselected LR:                  0.5718")
    print(f"  Unselected RF:                  0.5961")
    print(f"  Unselected SVM:                 0.5552")
    print("\nIf this whole-recording baseline scores CLOSE to or ABOVE the graph")
    print("results, that's important context: it would suggest the graph")
    print("transformation isn't adding much beyond the raw aggregated kinematics.")
    print("If it scores clearly BELOW, that supports the graph transformation")
    print("capturing real structure beyond simple aggregates. Report honestly")
    print("either way -- this is exactly the comparison R2.3 was asking for.")
