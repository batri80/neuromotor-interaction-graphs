"""
Shared helpers for the second-round revision analyses (INS-D-26-7505R1).

Everything here reuses the published pipeline unchanged:
  * windowing and kinematic features  -> test_kinematic_within_subject.py
  * TE estimator, NIG thresholding, NGS -> ngs_pipeline_v2.py
The only additions are (i) a fast path that discretises each node series once
and reuses the codes (numerically identical to ngs_pipeline_v2.transfer_entropy,
verified by check_te_equivalence()), (ii) source-shuffled surrogate TE for bias
estimation, and (iii) small evaluation utilities.
"""

import random
from collections import Counter

import numpy as np
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

import ngs_pipeline_v2 as base
from ngs_pipeline_v2 import CONFIG, NGS_CORE, extract_ngs, _disc
from test_kinematic_within_subject import extract_kinematic_features, KINEMATIC_KEYS

# Pre-specified reference configuration (unchanged from the manuscript)
N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15
TE_BINS = 8
TE_LAG = 1
PRUNE_PCT = 25
CONFIG['te_bins'] = TE_BINS


# ─────────────────────────────────────────────────────────────────────────────
# Windowing (identical to full_kinematic_pipeline / lag_sensitivity)
# ─────────────────────────────────────────────────────────────────────────────
def window_features(svc_data, n_windows=N_WINDOWS):
    """Return the (n_windows, 10) raw kinematic feature matrix, or None
    (same exclusion rules as the manuscript pipeline), plus a reason string."""
    if svc_data is None:
        return None, 'no Task-1 recording'
    if len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None, f'recording too short ({len(svc_data)} < {n_windows*MIN_SAMPLES_PER_WINDOW} samples)'
    n = len(svc_data)
    win = n // n_windows
    rows = []
    for wi in range(n_windows):
        s = wi * win
        e = s + win if wi < n_windows - 1 else n
        f = extract_kinematic_features(svc_data[s:e])
        if f is None:
            return None, f'feature extraction failed in window {wi}'
        rows.append([f[k] for k in KINEMATIC_KEYS])
    Xw = np.array(rows, dtype=float)
    if not np.all(np.isfinite(Xw)):
        return None, 'non-finite kinematic feature'
    return Xw, 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# Transfer entropy on pre-discretised codes
# ─────────────────────────────────────────────────────────────────────────────
def discretise_columns(Xz, bins=TE_BINS):
    """Equal-frequency discretisation of every column, exactly as
    ngs_pipeline_v2._disc does inside transfer_entropy."""
    return np.column_stack([_disc(Xz[:, j], bins) for j in range(Xz.shape[1])])


def te_codes(xd, yd, lag=TE_LAG):
    """Plug-in TE(X->Y) in nats from integer codes. Same formula and clipping
    as ngs_pipeline_v2.transfer_entropy (history length 1, lag `lag`)."""
    if len(xd) < lag + 2:
        return 0.0
    yt, yt1, xt1 = yd[lag:], yd[:-lag], xd[:-lag]
    N = len(yt)
    if N < 2:
        return 0.0
    jyy = Counter(zip(yt, yt1))
    jxxy = Counter(zip(yt, yt1, xt1))
    my1 = Counter(yt1)
    jyx1 = Counter(zip(yt1, xt1))
    te = 0.0
    for (a, b, c), cnt in jxxy.items():
        p = cnt / N
        py = jyy[(a, b)] / N
        px = jyx1[(b, c)] / N
        pm = my1[b] / N
        r = (p * pm) / (py * px)
        if r > 0:
            te += p * np.log(r)
    return max(te, 0.0)


def te_matrix_codes(D, lag=TE_LAG):
    n = D.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = te_codes(D[:, i], D[:, j], lag)
    return M


def surrogate_te(D, n_surr, rng, lag=TE_LAG):
    """Source-shuffled surrogates. For each directed pair (i->j) the source
    series i is randomly permuted, destroying any i->j temporal dependence
    while keeping both marginals (fixed by equal-frequency binning) and the
    target's own history. Returns array (n_surr, n, n)."""
    n = D.shape[1]
    S = np.zeros((n_surr, n, n))
    for s in range(n_surr):
        for i in range(n):
            xs = D[rng.permutation(len(D)), i]
            for j in range(n):
                if i != j:
                    S[s, i, j] = te_codes(xs, D[:, j], lag)
    return S


def surrogate_te_shift(D, min_shift=3, lag=TE_LAG):
    """Circular time-shift surrogates. The source series is rotated by every
    shift k in [min_shift, n - min_shift]; this destroys the alignment between
    source and target while preserving the source's own autocorrelation
    (unlike shuffling). All admissible shifts are used, so the result is
    deterministic. Returns array (n_shifts, n, n)."""
    T, n = D.shape
    shifts = list(range(min_shift, T - min_shift + 1))
    S = np.zeros((len(shifts), n, n))
    for s, k in enumerate(shifts):
        for i in range(n):
            xs = np.roll(D[:, i], k)
            for j in range(n):
                if i != j:
                    S[s, i, j] = te_codes(xs, D[:, j], lag)
    return S


def check_te_equivalence(Xz, lag=TE_LAG):
    """Confirm the fast path reproduces ngs_pipeline_v2.compute_te_matrix."""
    ref = np.zeros((Xz.shape[1], Xz.shape[1]))
    for i in range(Xz.shape[1]):
        for j in range(Xz.shape[1]):
            if i != j:
                ref[i, j] = base.transfer_entropy(Xz[:, i], Xz[:, j], lag=lag)
    fast = te_matrix_codes(discretise_columns(Xz), lag)
    return float(np.max(np.abs(ref - fast)))


# ─────────────────────────────────────────────────────────────────────────────
# Graph + NGS from a TE matrix (same thresholding rule as build_nig)
# ─────────────────────────────────────────────────────────────────────────────
def graph_from_matrix(M, prune_pct=PRUNE_PCT):
    nz = M[M > 0]
    thr = np.percentile(nz, prune_pct) if len(nz) > 0 else 0.0
    G = nx.DiGraph()
    G.add_nodes_from(range(M.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            if i != j and M[i, j] > thr:
                G.add_edge(i, j, weight=M[i, j])
    return G


def ngs_from_matrix(M, seed=0):
    """NGS vector (NGS_CORE order). The small-world null graphs in
    ngs_pipeline_v2 use Python's global `random`; seeding it makes every
    value reproducible."""
    random.seed(seed)
    d = extract_ngs(graph_from_matrix(M))
    return np.array([d.get(k, 0.0) for k in NGS_CORE], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (nested selection identical to lag_sensitivity.py)
# ─────────────────────────────────────────────────────────────────────────────
def _select(Xtr, ytr, alpha=0.05):
    keep = []
    for i in range(Xtr.shape[1]):
        try:
            _, p = stats.mannwhitneyu(Xtr[ytr == 1, i], Xtr[ytr == 0, i])
        except ValueError:
            p = 1.0
        if p < alpha:
            keep.append(i)
    return keep or list(range(Xtr.shape[1]))


def nested_lopo_auc(X, y, return_scores=False):
    yt, yp = [], []
    for tr, te in LeaveOneOut().split(X):
        keep = _select(X[tr], y[tr])
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr][:, keep]), y[tr])
        yt.append(y[te][0])
        yp.append(clf.predict_proba(sc.transform(X[te][:, keep]))[0, 1])
    auc = roc_auc_score(yt, yp)
    return (auc, np.array(yt), np.array(yp)) if return_scores else auc


def nested_kfold_auc(X, y, n_splits=5, n_repeats=10, seed=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    aucs = []
    for tr, te in rskf.split(X, y):
        keep = _select(X[tr], y[tr])
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr][:, keep]), y[tr])
        p = clf.predict_proba(sc.transform(X[te][:, keep]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def permutation_p(X, y, observed, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    null = np.array([nested_lopo_auc(X, rng.permutation(y)) for _ in range(n_perm)])
    return (1 + np.sum(null >= observed)) / (n_perm + 1), null


def bootstrap_ci(y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx_pd, idx_hc = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(idx_pd, len(idx_pd)),
                            rng.choice(idx_hc, len(idx_hc))])
        out.append(roc_auc_score(y[b], s[b]))
    return np.percentile(out, [2.5, 97.5])


def group_comparison(X, y, names=NGS_CORE):
    """Mann-Whitney per metric with Bonferroni and BH-FDR (m = 7)."""
    p = np.array([stats.mannwhitneyu(X[y == 1, k], X[y == 0, k])[1]
                  for k in range(X.shape[1])])
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    q = np.empty(m)
    q[order] = q_sorted
    bonf = np.minimum(p * len(p), 1.0)
    return [(n, p[k], q[k], bonf[k]) for k, n in enumerate(names)]


def load_cohort(pahaw_root, corpus_path):
    """Load PaHaW with the manuscript's loader and build the window matrices.
    Returns ids, labels (PD=1), list of raw window matrices, and the loader
    DataFrame (for demographics)."""
    part_df = base.load_pahaw(pahaw_root, corpus_path)
    ids, y, mats = [], [], []
    for _, row in part_df.iterrows():
        Xw, why = window_features(row['svc_data'])
        if Xw is None:
            continue
        ids.append(row['participant_id'])
        y.append(1 if row['label'] == 'PD' else 0)
        mats.append(Xw)
    return ids, np.array(y), mats, part_df
