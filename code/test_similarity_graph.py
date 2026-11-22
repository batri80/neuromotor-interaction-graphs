"""
Structural redesign: subject-as-node similarity graph.

Nodes = subjects. Edges = feature similarity (RBF kernel), NOT TE, NOT
group-conditional. No label is used anywhere in graph construction, so
there is no circularity risk by design -- this sidesteps the population-
pooling instability entirely rather than patching around it.

New NGS = each subject's own POSITION in this similarity graph:
  - weighted_degree: how similar this subject is to everyone else, summed
  - clustering: how tightly this subject's neighborhood is interconnected
  - eigenvector_centrality: influence-weighted importance in the graph
  - avg_knn_dist: average distance to k nearest neighbors (simple, robust)

Tested transductively first (graph built from all subjects' features, no
labels) since that's fast and tells us if there's anything here worth the
extra rigor of a fully fold-safe inductive version.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'


def build_similarity_graph_features(X, k_neighbors=10, gamma=None):
    """X: (n_subjects, n_features), already standardized.
    Returns (n_subjects, 4) array of graph-position features per subject."""
    n = X.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    S = rbf_kernel(X, X, gamma=gamma)  # similarity matrix, values in (0,1]
    np.fill_diagonal(S, 0)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if S[i, j] > 0.01:  # drop negligible edges for a cleaner graph
                G.add_edge(i, j, weight=S[i, j])

    weighted_degree = np.array([G.degree(i, weight='weight') for i in range(n)])
    clustering = np.array([nx.clustering(G, i, weight='weight') for i in range(n)])
    try:
        eig_cent = nx.eigenvector_centrality_numpy(G, weight='weight')
        eig_cent = np.array([eig_cent.get(i, 0.0) for i in range(n)])
    except Exception:
        eig_cent = np.zeros(n)

    # average distance to k nearest neighbors (simple, robust, complementary)
    avg_knn_dist = np.zeros(n)
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -np.inf
        top_k = np.argsort(sims)[-k_neighbors:]
        avg_knn_dist[i] = 1.0 - np.mean(S[i, top_k])  # convert similarity->distance-like

    return np.column_stack([weighted_degree, clustering, eig_cent, avg_knn_dist])


GRAPH_FEAT_NAMES = ['weighted_degree', 'clustering', 'eig_centrality', 'avg_knn_dist']


def evaluate(X, y, label=""):
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        yt.append(y[te][0])
        yp.append(clf.predict_proba(sc.transform(X[te]))[0, 1])
    lopo_auc = roc_auc_score(yt, yp)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    all_t, all_p = [], []
    for tr, te in rskf.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
        all_t.extend(y[te].tolist()); all_p.extend(prob.tolist())
    kfold_pooled = roc_auc_score(all_t, all_p)

    print(f"{label:<50} LOPO AUC={lopo_auc:.4f}   K-fold pooled AUC={kfold_pooled:.4f}")
    return lopo_auc, kfold_pooled


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    y = (part_df['label'] == 'PD').astype(int).values

    print("="*90)
    print("Feature source A: raw per-subject Phi10 scalars (10-dim, already trusted)")
    print("="*90)
    X_phi10 = part_df[CONFIG['feature_names']].values
    X_phi10_z = StandardScaler().fit_transform(X_phi10)

    evaluate(X_phi10_z, y, "Raw Phi10 directly (no graph, sanity baseline)")

    for k in [5, 10, 15]:
        graph_feats = build_similarity_graph_features(X_phi10_z, k_neighbors=k)
        evaluate(graph_feats, y, f"Similarity-graph position features (k={k}), from Phi10")

    print("\n" + "="*90)
    print("Feature source B: within-subject kinematic NGS (7-dim, from earlier)")
    print("="*90)
    ngs_df = pd.read_pickle('within_subject_kinematic_ngs_df.pkl')
    X_kin = ngs_df[NGS_CORE].values
    y_kin = ngs_df['label_int'].values
    X_kin_z = StandardScaler().fit_transform(X_kin)

    evaluate(X_kin_z, y_kin, "Within-subject kinematic NGS directly (reference, =0.5718/0.5916)")

    for k in [5, 10, 15]:
        graph_feats = build_similarity_graph_features(X_kin_z, k_neighbors=k)
        evaluate(graph_feats, y_kin, f"Similarity-graph position features (k={k}), from kinematic NGS")

    print("\n" + "="*90)
    print("Feature source C: combined (Phi10 + kinematic NGS similarity-graph features)")
    print("="*90)
    common_pids = set(part_df['participant_id']) & set(ngs_df['participant_id'])
    part_idx = part_df.set_index('participant_id')
    ngs_idx = ngs_df.set_index('participant_id')
    combo_rows = []
    for pid in sorted(common_pids):
        row = {'label_int': int(part_idx.loc[pid, 'label'] == 'PD')}
        for c in CONFIG['feature_names']:
            row[f'phi10_{c}'] = part_idx.loc[pid, c]
        for c in NGS_CORE:
            row[f'ngs_{c}'] = ngs_idx.loc[pid, c]
        combo_rows.append(row)
    combo_df = pd.DataFrame(combo_rows)
    combo_cols = [c for c in combo_df.columns if c != 'label_int']
    X_combo = StandardScaler().fit_transform(combo_df[combo_cols].values)
    y_combo = combo_df['label_int'].values

    for k in [5, 10, 15]:
        graph_feats = build_similarity_graph_features(X_combo, k_neighbors=k)
        evaluate(graph_feats, y_combo, f"Similarity-graph position features (k={k}), from combined")
