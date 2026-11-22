"""
Native-resolution Neuromotor Interaction Graphs.

Why: the window-aggregated NIG gives each participant only 19 transitions
(20 windows, lag 1). The round-2 analyses showed that this, not the idea of a
coupling graph, is the bottleneck: edge weights are bias-dominated (plug-in)
and graph descriptors are unreliable within a participant (bootstrap R < 0.2
for every estimator). PaHaW records the pen at ~130 Hz, so each spiral
contains hundreds to thousands of samples. Here the graph is estimated from
the kinematic signals themselves, sample by sample, which is the direct
analogue of functional connectivity estimated from multichannel neural time
series.

Pre-specified design (fixed before looking at any group result):
  Nodes (10 instantaneous kinematic channels, on-surface samples only):
      speed, tangential acceleration, jerk, angular velocity (direction-change
      rate), pressure, pressure rate, azimuth, altitude, azimuth rate,
      altitude rate.
      Derivatives are BACKWARD differences on a causal 3-sample moving average,
      so no channel contains information from the future (centred filters
      would leak future samples and create spurious TE).
  Edges: linear-Gaussian TE (= Granger causality in nats; Barnett et al. 2009)
      after a Gaussian-copula transform of every channel within participant
      (Ince et al. 2017). Model order p = 5 samples (~40 ms) for both the
      target's own history and the source; only time points whose whole
      history lies inside the same pen-down stroke are used. Edge test:
      nested-model F test, df = (p, N - 2p - 1).
  Bias: circular-shift surrogates of the source (30 random shifts between
      10% and 90% of the series), corrected TE = TE - mean(surrogate), >= 0.
  Graph, NGS, pruning (25th pct), nested-selection LR: unchanged.

Reported:
  0. Samples and usable transitions per participant.
  1. Edge level: % edges with F-test p<0.05 (chance 5%), bias/raw ratio,
     bias PD vs HC, and the edge-level group contrast (mean TE and % significant
     edges per participant, PD vs HC, Mann-Whitney).
  2. Reliability: split-half (first vs second half of the on-surface samples):
     per-metric ICC(1) across participants and Spearman-Brown-corrected
     correlation. Directly comparable to the window-level bootstrap R.
  3. Group comparison (FDR/Bonferroni) and classification (LOPO, 10x5-fold,
     bootstrap CI, permutation p) for raw and bias-corrected weights.
  4. Recording-length control: PD write more slowly, so recordings differ in
     length. Correlation of NGS with sample count, and a full re-analysis with
     every participant truncated to the shortest recording's sample count.

Usage:
  python r2_native_nig.py [--order 5] [--perm 1000] [--no-copula]
Run the default first. --order 3 and --order 10 are sensitivity checks.
Outputs: r2_native_nig_results_p{order}{_nocop}.txt and a matching .csv
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

import r2_common as rc
from r2_common import NGS_CORE
import ngs_pipeline_v2 as base

CHANNELS = ['speed', 'acc_tan', 'jerk', 'ang_vel', 'pressure', 'p_rate',
            'azimuth', 'altitude', 'az_rate', 'alt_rate']


# ─────────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────────
def _causal_ma(x, k=3):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _bdiff(x, dt):
    d = np.zeros_like(x)
    d[1:] = np.diff(x) / dt[1:]
    d[0] = d[1] if len(d) > 1 else 0.0
    return d


def stroke_signals(seg):
    """seg: (n,7) on-surface samples of one stroke -> (n,10) channels."""
    Y, X, t, _, az, alt, p = seg.T
    dt = np.diff(t, prepend=t[0])
    pos = dt[dt > 0]
    med = np.median(pos) if len(pos) else 7.5
    dt[dt <= 0] = med
    dt = dt / 1000.0  # ms -> s
    Xs, Ys = _causal_ma(X), _causal_ma(Y)
    ps, azs, alts = _causal_ma(p), _causal_ma(az), _causal_ma(alt)
    vx, vy = _bdiff(Xs, dt), _bdiff(Ys, dt)
    speed = np.hypot(vx, vy)
    acc = _bdiff(speed, dt)
    jerk = _bdiff(acc, dt)
    theta = np.unwrap(np.arctan2(vy, vx))
    angv = _bdiff(theta, dt)
    return np.column_stack([speed, acc, jerk, angv, ps, _bdiff(ps, dt),
                            azs, alts, _bdiff(azs, dt), _bdiff(alts, dt)])


def participant_series(svc, min_len):
    """Concatenated channel matrix of all pen-down strokes and the stroke
    boundaries (start indices). Strokes shorter than min_len are dropped."""
    on = svc[:, 3] == 1
    idx = np.where(on)[0]
    if len(idx) == 0:
        return None, None
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(idx)]])
    blocks, bounds, pos = [], [], 0
    for s, e in zip(starts, ends):
        seg = svc[idx[s:e]]
        if len(seg) < min_len:
            continue
        blocks.append(stroke_signals(seg))
        bounds.append((pos, pos + len(seg)))
        pos += len(seg)
    if not blocks:
        return None, None
    return np.vstack(blocks), bounds


def clip_bounds(bounds, n, p):
    return [(a, min(b, n)) for a, b in bounds if a < n and min(b, n) - a > p]


def transform(S, copula=True):
    Z = (S - S.mean(0)) / np.where(S.std(0) > 0, S.std(0), 1.0)
    if copula:
        n = len(Z)
        Z = stats.norm.ppf(np.apply_along_axis(stats.rankdata, 0, Z) / (n + 1))
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# Granger / Gaussian TE with stroke-respecting lags
# ─────────────────────────────────────────────────────────────────────────────
def lag_rows(bounds, p):
    """Indices t such that t-p..t-1 lie in the same stroke as t."""
    return np.concatenate([np.arange(a + p, b) for a, b in bounds if b - a > p])


def lagmat(x, rows, p):
    return np.column_stack([x[rows - k] for k in range(1, p + 1)])


def te_matrix(Z, bounds, p, src=None):
    """TE(i->j) for all pairs. `src` optionally replaces source columns
    (used for surrogates). Returns TE matrix (nats) and F-test p-values."""
    n = Z.shape[1]
    rows = lag_rows(bounds, p)
    N = len(rows)
    one = np.ones((N, 1))
    src = Z if src is None else src
    Lsrc = [lagmat(src[:, i], rows, p) for i in range(n)]
    M, P = np.zeros((n, n)), np.ones((n, n))
    for j in range(n):
        yt = Z[rows, j]
        Ar = np.hstack([one, lagmat(Z[:, j], rows, p)])
        cr, *_ = np.linalg.lstsq(Ar, yt, rcond=None)
        rss_r = float(np.sum((yt - Ar @ cr) ** 2))
        for i in range(n):
            if i == j:
                continue
            Af = np.hstack([Ar, Lsrc[i]])
            cf, *_ = np.linalg.lstsq(Af, yt, rcond=None)
            rss_f = float(np.sum((yt - Af @ cf) ** 2))
            if rss_f <= 1e-12 or rss_r <= 1e-12:
                continue
            M[i, j] = max(0.5 * np.log(rss_r / rss_f), 0.0)
            F = ((rss_r - rss_f) / p) / (rss_f / (N - 2 * p - 1))
            P[i, j] = stats.f.sf(F, p, N - 2 * p - 1)
    return M, P, N


def shift_null(Z, bounds, p, n_shift, rng):
    T = len(Z)
    S = []
    for _ in range(n_shift):
        k = int(rng.integers(int(0.1 * T), int(0.9 * T)))
        S.append(te_matrix(Z, bounds, p, src=np.roll(Z, k, axis=0))[0])
    return np.mean(S, axis=0)


def icc1(a, b):
    X = np.column_stack([a, b])
    n, k = X.shape
    gm = X.mean()
    msb = k * np.sum((X.mean(1) - gm) ** 2) / (n - 1)
    msw = np.sum((X - X.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--order', type=int, default=5)
    ap.add_argument('--perm', type=int, default=1000)
    ap.add_argument('--shifts', type=int, default=30)
    ap.add_argument('--no-copula', action='store_true')
    args = ap.parse_args()
    p, cop = args.order, not args.no_copula
    sfx = f"_p{p}" + ('' if cop else '_nocop')

    log = open(f'r2_native_nig_results{sfx}.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    part_df = base.load_pahaw(args.pahaw, args.corpus)
    rng = np.random.default_rng(2026)
    min_len = 2 * p + 5
    off = ~np.eye(10, dtype=bool)

    ids, y, Mraw, Meff, H1, H2, SERIES = [], [], [], [], [], [], []
    nsamp, ntrans, fsig, raw_m, bias_m = [], [], [], [], []
    t0 = time.time()
    for _, row in part_df.iterrows():
        svc = row['svc_data']
        if svc is None or rc.window_features(svc)[0] is None:
            continue   # same 72-participant cohort as the manuscript
        S, bounds = participant_series(svc, min_len)
        if S is None or len(S) < 200:
            out(f"  excluded {row['participant_id']}: too few on-surface samples")
            continue
        Z = transform(S, cop)
        M, P, N = te_matrix(Z, bounds, p)
        bias = shift_null(Z, bounds, p, args.shifts, rng)
        # split-half: first vs second half of the concatenated samples
        mid = len(Z) // 2
        b1 = [(a, min(b, mid)) for a, b in bounds if a < mid and min(b, mid) - a > p]
        b2 = [(max(a, mid) - mid, b - mid) for a, b in bounds if b > mid and b - max(a, mid) > p]
        Z1, Z2 = transform(S[:mid], cop), transform(S[mid:], cop)
        H1.append(te_matrix(Z1, b1, p)[0]); H2.append(te_matrix(Z2, b2, p)[0])

        SERIES.append((S, bounds))
        ids.append(row['participant_id']); y.append(1 if row['label'] == 'PD' else 0)
        Mraw.append(M); Meff.append(np.maximum(M - bias, 0.0))
        nsamp.append(len(S)); ntrans.append(N)
        fsig.append((P[off] < 0.05).mean())
        raw_m.append(M[off].mean()); bias_m.append(bias[off].mean())
    y = np.array(y)
    nsamp, ntrans, fsig, raw_m, bias_m = map(np.array, (nsamp, ntrans, fsig, raw_m, bias_m))

    out(f"Native-resolution NIG: Gaussian TE{' (copula)' if cop else ''}, order p={p}, "
        f"{args.shifts} circular-shift surrogates  [{(time.time()-t0)/60:.1f} min]")
    out(f"n={len(y)} (PD={int(y.sum())}, HC={int((y==0).sum())})")
    out(f"\n0. On-surface samples per participant: min={nsamp.min()}  median={int(np.median(nsamp))}  "
        f"max={nsamp.max()}")
    out(f"   usable transitions per participant: min={ntrans.min()}  median={int(np.median(ntrans))}  "
        f"(window-level design: 19)")

    out(f"   samples PD median={int(np.median(nsamp[y==1]))}  HC median={int(np.median(nsamp[y==0]))}  "
        f"PD vs HC p={stats.mannwhitneyu(nsamp[y==1], nsamp[y==0])[1]:.4f}")

    out("\n1. Edge level")
    out(f"   edges with F-test p<0.05: {100*fsig.mean():.1f}%  (chance 5.0%)   "
        f"PD {100*fsig[y==1].mean():.1f}%  HC {100*fsig[y==0].mean():.1f}%   "
        f"PD vs HC p={stats.mannwhitneyu(fsig[y==1], fsig[y==0])[1]:.4f}")
    out(f"   raw TE mean={raw_m.mean():.5f} nats; surrogate (bias) mean={bias_m.mean():.5f}; "
        f"bias/raw={np.mean(bias_m/np.maximum(raw_m,1e-12)):.3f}")
    out(f"   bias PD={bias_m[y==1].mean():.5f}  HC={bias_m[y==0].mean():.5f}  "
        f"PD vs HC p={stats.mannwhitneyu(bias_m[y==1], bias_m[y==0])[1]:.4f}")
    eff_m = np.array([M[off].mean() for M in Meff])
    out(f"   corrected mean TE  PD={eff_m[y==1].mean():.5f}  HC={eff_m[y==0].mean():.5f}  "
        f"PD vs HC p={stats.mannwhitneyu(eff_m[y==1], eff_m[y==0])[1]:.4f}")

    out("\n   Strongest edges (group-mean corrected TE, top 10):")
    G = np.mean(Meff, axis=0)
    order = np.dstack(np.unravel_index(np.argsort(-G, axis=None), G.shape))[0]
    shown = 0
    for i, j in order:
        if i == j:
            continue
        a = np.array([M[i, j] for M in Meff])
        out(f"     {CHANNELS[i]:>9} -> {CHANNELS[j]:<9} mean={G[i,j]:.4f}  "
            f"PD={a[y==1].mean():.4f} HC={a[y==0].mean():.4f}  "
            f"p={stats.mannwhitneyu(a[y==1], a[y==0])[1]:.4f}")
        shown += 1
        if shown == 10:
            break

    X1 = np.array([rc.ngs_from_matrix(M, seed=k) for k, M in enumerate(H1)])
    X2 = np.array([rc.ngs_from_matrix(M, seed=k) for k, M in enumerate(H2)])
    out("\n2. Split-half reliability of NGS (first vs second half of the recording)")
    out(f"   {'metric':<12} {'ICC(1)':>7} {'r':>7} {'r_SB':>7}")
    for k, m in enumerate(NGS_CORE):
        r = stats.spearmanr(X1[:, k], X2[:, k])[0]
        out(f"   {m:<12} {icc1(X1[:,k], X2[:,k]):>7.3f} {r:>7.3f} {2*r/(1+r) if r>-1 else np.nan:>7.3f}")
    out("   (window-level bootstrap R for Eff/E_spectral/Ricci: plug-in 0.05/0.03/0.02, "
        "Gaussian-copula 0.11/0.06/0.11)")

    Xr = np.array([rc.ngs_from_matrix(M, seed=k) for k, M in enumerate(Mraw)])
    Xe = np.array([rc.ngs_from_matrix(M, seed=k) for k, M in enumerate(Meff)])
    out("\n3. Rank agreement raw vs corrected: " + "  ".join(
        f"{m}={stats.spearmanr(Xr[:,k], Xe[:,k])[0]:+.2f}" for k, m in enumerate(NGS_CORE)))
    for tag, X in [('raw', Xr), ('bias-corrected', Xe)]:
        out(f"\n   Group comparison, {tag} (Mann-Whitney, m=7):")
        for m, pv, q, b in rc.group_comparison(X, y):
            flag = ('FDR*' if q < 0.05 else '') + (' Bonf*' if b < 0.05 else '')
            d = X[y==1, NGS_CORE.index(m)].mean() - X[y==0, NGS_CORE.index(m)].mean()
            out(f"     {m:<12} p={pv:.4f}  q_FDR={q:.4f}  p_Bonf={b:.4f}  "
                f"{'PD>HC' if d>0 else 'PD<HC'}  {flag}")
        auc, yt, ys = rc.nested_lopo_auc(X, y, return_scores=True)
        km, ks = rc.nested_kfold_auc(X, y)
        lo, hi = rc.bootstrap_ci(yt, ys)
        pv, _ = rc.permutation_p(X, y, auc, args.perm)
        out(f"   Classification ({tag}): LOPO AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
            f"10x5-fold={km:.4f}±{ks:.4f}  permutation p={pv:.4f} ({args.perm})")

    out("\n4. Recording-length control")
    out("   Spearman correlation of each NGS metric (corrected) with on-surface sample count:")
    out("   " + "  ".join(f"{m}={stats.spearmanr(Xe[:,k], nsamp)[0]:+.2f}"
                         for k, m in enumerate(NGS_CORE)))
    Nc = int(nsamp.min())
    out(f"   Length-matched re-analysis: every participant truncated to the first {Nc} "
        f"on-surface samples (shortest recording in the cohort)")
    Xt = []
    for k, (S, bounds) in enumerate(SERIES):
        Zt = transform(S[:Nc], cop)
        bt = clip_bounds(bounds, Nc, p)
        Mt = te_matrix(Zt, bt, p)[0]
        bias_t = shift_null(Zt, bt, p, args.shifts, rng)
        Xt.append(rc.ngs_from_matrix(np.maximum(Mt - bias_t, 0.0), seed=k))
    Xt = np.array(Xt)
    out("   Rank agreement full vs length-matched: " + "  ".join(
        f"{m}={stats.spearmanr(Xe[:,k], Xt[:,k])[0]:+.2f}" for k, m in enumerate(NGS_CORE)))
    out("   Group comparison, length-matched bias-corrected (Mann-Whitney, m=7):")
    for m, pv, q, b in rc.group_comparison(Xt, y):
        flag = ('FDR*' if q < 0.05 else '') + (' Bonf*' if b < 0.05 else '')
        d = Xt[y==1, NGS_CORE.index(m)].mean() - Xt[y==0, NGS_CORE.index(m)].mean()
        out(f"     {m:<12} p={pv:.4f}  q_FDR={q:.4f}  p_Bonf={b:.4f}  "
            f"{'PD>HC' if d>0 else 'PD<HC'}  {flag}")
    auc, yt_, ys_ = rc.nested_lopo_auc(Xt, y, return_scores=True)
    km, ks = rc.nested_kfold_auc(Xt, y)
    lo, hi = rc.bootstrap_ci(yt_, ys_)
    pv, _ = rc.permutation_p(Xt, y, auc, args.perm)
    out(f"   Classification (length-matched): LOPO AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
        f"10x5-fold={km:.4f}±{ks:.4f}  permutation p={pv:.4f} ({args.perm})")

    rows = []
    for k, pid in enumerate(ids):
        r = {'participant_id': pid, 'label': int(y[k]), 'n_samples': nsamp[k],
             'n_transitions': ntrans[k], 'frac_sig_edges': fsig[k],
             'raw_te_mean': raw_m[k], 'bias_mean': bias_m[k]}
        r.update({f'raw_{m}': Xr[k, j] for j, m in enumerate(NGS_CORE)})
        r.update({f'eff_{m}': Xe[k, j] for j, m in enumerate(NGS_CORE)})
        r.update({f'lenmatched_{m}': Xt[k, j] for j, m in enumerate(NGS_CORE)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'r2_native_nig{sfx}.csv', index=False)
    np.savez(f'r2_native_nig_matrices{sfx}.npz', raw=np.array(Mraw), eff=np.array(Meff),
             y=y, ids=np.array(ids), channels=np.array(CHANNELS))
    out(f"\nSaved r2_native_nig_results{sfx}.txt, .csv and _matrices .npz  "
        f"[total {(time.time()-t0)/60:.1f} min]")
    log.close()


if __name__ == '__main__':
    main()
