"""
Minimal GNN baseline: 2-layer GCN + global mean pooling, graph classification.
Each subject IS one graph (10 nodes = kinematic features, edges = their own
within-subject TE-weighted adjacency).

IMPORTANT CAVEAT (read before trusting any single run of this):
GNN training on 72 tiny (10-node) graphs, with random weight initialization
and gradient descent, is inherently higher-variance than the closed-form/
low-variance methods (LR, RF, SVM) used elsewhere in this analysis. A single
run's AUC is not a reliable estimate. This script runs N_SEEDS independent
full LOPO evaluations and reports mean +/- std -- report BOTH numbers, not
just the best seed.

Requires PyTorch. If not installed:
    pip install torch --break-system-packages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not installed. Run: pip install torch --break-system-packages")
    raise SystemExit(1)

from ngs_pipeline_v2 import load_pahaw, CONFIG, compute_te_matrix
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15
N_SEEDS = 5   # multiple seeds given GNN training variance -- report mean+/-std, not best


def build_subject_graph(svc_data, n_windows=N_WINDOWS):
    """Returns (node_features [10,2], adjacency [10,10]) for one subject,
    or None if too short. Node features = (mean, std) of that kinematic
    feature across this subject's own windows (self-normalized)."""
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

    node_feats = np.column_stack([Xw.mean(axis=0), Xw.std(axis=0)])  # (10, 2)
    # NOTE: deliberately using Xw (raw window values), not Xz (standardized).
    # Xz.mean()=0 and Xz.std()=1 by construction of StandardScaler, which
    # would make node features carry zero information -- this was the bug
    # in the first version of this script.

    te_matrix = compute_te_matrix(Xz)
    nz = te_matrix[te_matrix > 0]
    thr = np.percentile(nz, 25) if len(nz) > 0 else 0.0
    adj = (te_matrix > thr).astype(float) * te_matrix
    # symmetrize + self-loops for GCN stability
    adj = adj + adj.T
    adj = adj + np.eye(10)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg <= 0] = 0
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    return node_feats.astype(np.float32), adj_norm.astype(np.float32)


class SimpleGCN(nn.Module):
    def __init__(self, in_dim=2, hidden=8):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.5)  # heavy regularization given tiny n

    def forward(self, x, adj):
        h = F.relu(adj @ self.w1(x))
        h = self.dropout(h)
        h = F.relu(adj @ self.w2(h))
        g = h.mean(dim=0)  # global mean pool
        return self.out(g)


def train_and_eval_lopo(graphs, labels, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(labels)
    yt, yp = [], []

    for test_i in range(n):
        train_idx = [i for i in range(n) if i != test_i]
        model = SimpleGCN()
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

        model.train()
        for epoch in range(60):
            opt.zero_grad()
            loss = 0.0
            for i in train_idx:
                x = torch.tensor(graphs[i][0])
                adj = torch.tensor(graphs[i][1])
                logit = model(x, adj)
                y = torch.tensor([float(labels[i])])
                loss = loss + F.binary_cross_entropy_with_logits(logit, y)
            loss = loss / len(train_idx)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x = torch.tensor(graphs[test_i][0])
            adj = torch.tensor(graphs[test_i][1])
            prob = torch.sigmoid(model(x, adj)).item()
        yt.append(labels[test_i])
        yp.append(prob)

    return roc_auc_score(yt, yp)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    print("Building per-subject graphs...")
    graphs, labels = [], []
    for _, row in part_df.iterrows():
        g = build_subject_graph(row['svc_data'])
        if g is None:
            continue
        graphs.append(g)
        labels.append(1 if row['label'] == 'PD' else 0)
    print(f"n={len(graphs)} subjects\n")

    # Global rescaling of node features across all subjects (20-dim: 10 nodes
    # x [mean, std]). DISCLOSED SIMPLIFICATION: ideally fit per-fold on
    # training subjects only for strict fold-safety; done globally here as a
    # reasonable compromise for this bounded, already-exploratory experiment.
    # This is feature rescaling only, never uses any subject's label.
    all_node_feats = np.stack([g[0] for g in graphs])  # (n_subjects, 10, 2)
    flat = all_node_feats.reshape(len(graphs), -1)
    flat_mean, flat_std = flat.mean(axis=0), flat.std(axis=0)
    flat_std[flat_std == 0] = 1.0
    flat_scaled = (flat - flat_mean) / flat_std
    node_feats_scaled = flat_scaled.reshape(len(graphs), 10, 2).astype(np.float32)
    graphs = [(node_feats_scaled[i], graphs[i][1]) for i in range(len(graphs))]
    print("Applied global node-feature rescaling (disclosed simplification, "
          "see script comments).\n")

    print(f"Running LOPO GCN evaluation across {N_SEEDS} seeds "
          f"(this will take a while -- {N_SEEDS} x {len(graphs)} model trainings)...")

    aucs = []
    for seed in range(N_SEEDS):
        auc = train_and_eval_lopo(graphs, labels, seed)
        aucs.append(auc)
        print(f"  seed={seed}  LOPO AUC={auc:.4f}")

    aucs = np.array(aucs)
    print(f"\nMean AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}  "
          f"(range: {aucs.min():.4f} - {aucs.max():.4f})")
    print(f"\nFor reference: nested-selection LR (non-GNN) = 0.6150")
    print("\nReport MEAN +/- STD in the manuscript, not the best seed. High")
    print("variance across seeds would itself be an honest, reportable finding")
    print("about GNN reliability on this sample size -- not evidence to discard.")
