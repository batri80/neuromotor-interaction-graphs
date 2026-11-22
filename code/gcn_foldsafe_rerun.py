"""
Fold-safe rerun of the GCN comparison (node-feature rescaling fitted within each fold).

The original gnn_comparison.py rescaled node features GLOBALLY across
all 72 participants before the LOPO loop (disclosed as a simplification
in code comments, but not stated in the manuscript text). This version
moves that rescaling INSIDE each LOPO fold: scaling parameters (mean,
std) are computed from the 71 training participants only, then applied
unchanged to the held-out participant. This matches the same
fold-safety standard used everywhere else in the paper.

Everything else (graph construction, GCN architecture, training
protocol, 5 seeds) is identical to the original script, so any
difference in the result is attributable specifically to this fix.
"""

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15
N_SEEDS = 5


def build_subject_graph_raw(svc_data, n_windows=N_WINDOWS):
    """Same as before, EXCEPT node features are returned RAW (unscaled).
    Rescaling now happens later, inside the LOPO loop, fold-safely."""
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
    Xz_for_te = StandardScaler().fit_transform(Xw)  # per-subject z-score, only for TE estimation (unchanged, already fold-safe: uses only this subject's own windows)

    node_feats_raw = np.column_stack([Xw.mean(axis=0), Xw.std(axis=0)])  # (10, 2), RAW units

    te_matrix = compute_te_matrix(Xz_for_te)
    nz = te_matrix[te_matrix > 0]
    thr = np.percentile(nz, 25) if len(nz) > 0 else 0.0
    adj = (te_matrix > thr).astype(float) * te_matrix
    adj = adj + adj.T
    adj = adj + np.eye(10)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg <= 0] = 0
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    return node_feats_raw.astype(np.float32), adj_norm.astype(np.float32)


class SimpleGCN(nn.Module):
    def __init__(self, in_dim=2, hidden=8):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        h = F.relu(adj @ self.w1(x))
        h = self.dropout(h)
        h = F.relu(adj @ self.w2(h))
        g = h.mean(dim=0)
        return self.out(g)


def train_and_eval_lopo_foldsafe(graphs_raw, labels, seed):
    """graphs_raw: list of (raw_node_feats [10,2], adj_norm [10,10]).
    Rescaling is fit fresh in EVERY fold, using only the 71 training
    participants' raw node features, then applied to both train and
    the held-out participant."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(labels)
    yt, yp = [], []

    for test_i in range(n):
        train_idx = [i for i in range(n) if i != test_i]

        # Fold-safe rescaling: fit on training participants' raw node features only
        train_feats_flat = np.stack([graphs_raw[i][0] for i in train_idx])  # (71, 10, 2)
        flat = train_feats_flat.reshape(len(train_idx), -1)
        flat_mean, flat_std = flat.mean(axis=0), flat.std(axis=0)
        flat_std[flat_std == 0] = 1.0

        def rescale(node_feats_raw):
            f = node_feats_raw.reshape(1, -1)
            f_scaled = (f - flat_mean) / flat_std
            return f_scaled.reshape(10, 2).astype(np.float32)

        model = SimpleGCN()
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

        model.train()
        for epoch in range(60):
            opt.zero_grad()
            loss = 0.0
            for i in train_idx:
                x = torch.tensor(rescale(graphs_raw[i][0]))
                adj = torch.tensor(graphs_raw[i][1])
                logit = model(x, adj)
                y = torch.tensor([float(labels[i])])
                loss = loss + F.binary_cross_entropy_with_logits(logit, y)
            loss = loss / len(train_idx)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x = torch.tensor(rescale(graphs_raw[test_i][0]))
            adj = torch.tensor(graphs_raw[test_i][1])
            prob = torch.sigmoid(model(x, adj)).item()
        yt.append(labels[test_i])
        yp.append(prob)

    return roc_auc_score(yt, yp)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    print("Building per-subject graphs (raw node features, unscaled)...")
    graphs_raw, labels = [], []
    for _, row in part_df.iterrows():
        g = build_subject_graph_raw(row['svc_data'])
        if g is None:
            continue
        graphs_raw.append(g)
        labels.append(1 if row['label'] == 'PD' else 0)
    print(f"n={len(graphs_raw)} subjects\n")

    print(f"Running FOLD-SAFE LOPO GCN evaluation across {N_SEEDS} seeds")
    print("(rescaling now fit fresh per fold on training participants only)...")

    aucs = []
    for seed in range(N_SEEDS):
        auc = train_and_eval_lopo_foldsafe(graphs_raw, labels, seed)
        aucs.append(auc)
        print(f"  seed={seed}  LOPO AUC={auc:.4f}")

    aucs = np.array(aucs)
    print(f"\nFold-safe mean AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}  "
          f"(range: {aucs.min():.4f} - {aucs.max():.4f})")
    print(f"\nFor reference, original (globally-rescaled) result: 0.3602 +/- 0.0197")
    print("\nIf these numbers are similar, that confirms the global-rescaling")
    print("simplification was inconsequential, as expected (feature-only, not")
    print("label leakage, and the result was already below chance). Report")
    print("whichever number set you use consistently in the manuscript.")
