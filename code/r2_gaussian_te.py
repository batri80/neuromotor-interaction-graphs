"""
Small-sample alternative estimator: linear-Gaussian transfer entropy.

For Gaussian variables TE(X->Y) equals half the log ratio of the residual
variances of two nested regressions (Barnett, Barrett & Seth, 2009, PRL 103,
238701), i.e. it is Granger causality expressed in nats:

    restricted:  y_{t+l} = a + b*y_t + e_r
    full:        y_{t+l} = a + b*y_t + c*x_t + e_f
    TE(X->Y)   = 0.5 * ln( var(e_r) / var(e_f) )

It needs 3 parameters per edge instead of a 512-cell joint histogram, so its
variance at 19 transitions is far lower than the plug-in estimator's. Edge
significance has an exact parametric test (nested-model F test, df = 1, N-3).

Everything else is unchanged from the manuscript pipeline: same participants,
20 windows, same 10 kinematic features z-scored within participant, lag 1,
same 25th-percentile pruning, same NGS, same nested-selection LR.

The script runs, in order:
  A. Edge-level evidence: fraction of edges with F-test p<0.05 (chance 5%),
     raw vs circular-shift null TE (bias), PD vs HC.
  B. Group comparison (FDR/Bonferroni) and classification (LOPO, 10x5-fold,
     bootstrap CI, 1000-permutation p) for raw and shift-null-corrected
     Gaussian TE.
  C. Moving-block bootstrap (100 replicates, block 4) of the raw Gaussian TE:
     descriptor reliability R, group-effect stability, AUC across worlds.
  D. The same reliability numbers for the plug-in estimator are already in
     r2_window_bootstrap_results_raw.txt for direct comparison.

Options: --copula applies a Gaussian-copula transform (ranks -> normal scores)
to each feature within participant before estimation (Ince et al., 2017,
Hum. Brain Mapp.), which makes the estimator robust to skewed features.
Run once without and once with --copula.

Usage:
  python r2_gaussian_te.py [--perm 1000] [--reps 100] [--copula]
Outputs: r2_gaussian_te_results[_copula].txt, r2_gaussian_te_ngs[_copula].csv
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import r2_common as rc
from r2_common import NGS_CORE

SELECTED = ['Eff', 'E_spectral', 'Ricci_OR']


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────
def prepare(Xw, copula=False):
    Xz = StandardScaler().fit_transform(Xw)
    if copula:
        n = Xz.shape[0]
        R = np.apply_along_axis(stats.rankdata, 0, Xz)
        Xz = stats.norm.ppf(R / (n + 1))
        Xz = StandardScaler().fit_transform(Xz)
    return Xz


def _rss(A, b):
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ coef
    return float(r @ r)


def gauss_te_pair(x, y, lag=rc.TE_LAG):
    """Returns (TE in nats, F-test p-value) for X->Y."""
    yt, yp, xp = y[lag:], y[:-lag], x[:-lag]
    N = len(yt)
    one = np.ones(N)
    rss_r = _rss(np.column_stack([one, yp]), yt)
    rss_f = _rss(np.column_stack([one, yp, xp]), yt)
    if rss_f <= 1e-15 or rss_r <= 1e-15:
        return 0.0, 1.0
    te = 0.5 * np.log(rss_r / rss_f)
    F = (rss_r - rss_f) / (rss_f / (N - 3))
    p = 1.0 - stats.f.cdf(F, 1, N - 3)
    return max(te, 0.0), p


def gauss_te_matrix(Xz, lag=rc.TE_LAG):
    n = Xz.shape[1]
    M, P = np.zeros((n, n)), np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j], P[i, j] = gauss_te_pair(Xz[:, i], Xz[:, j], lag)
    return M, P


def gauss_shift_null(Xz, min_shift=3, lag=rc.TE_LAG):
    T, n = Xz.shape
    shifts = range(min_shift, T - min_shift + 1)
    S = []
    for k in shifts:
        M = np.zeros((n, n))
        for i in range(n):
            xs = np.roll(Xz[:, i], k)
            for j in range(n):
                if i != j:
                    M[i, j] = gauss_te_pair(xs, Xz[:, j], lag)[0]
        S.append(M)
    return np.array(S)


def block_resample(n, L, rng):
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--perm', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--block', type=int, default=4)
    ap.add_argument('--copula', action='store_true')
    args = ap.parse_args()

    sfx = '_copula' if args.copula else ''
    log = open(f'r2_gaussian_te_results{sfx}.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    ids, y, mats, _ = rc.load_cohort(args.pahaw, args.corpus)
    off = ~np.eye(10, dtype=bool)
    out(f"Linear-Gaussian TE{' (Gaussian-copula transformed)' if args.copula else ''}; "
        f"n={len(y)} (PD={int(y.sum())}, HC={int((y==0).sum())}); W={rc.N_WINDOWS}, "
        f"lag={rc.TE_LAG} -> {rc.N_WINDOWS - rc.TE_LAG} transitions; prune={rc.PRUNE_PCT}th pct")

    # ── A. edge-level evidence ───────────────────────────────────────────────
    t0 = time.time()
    Mraw, Meff, fsig, raw_m, bias_m, null_sig = [], [], [], [], [], []
    for Xw in mats:
        Xz = prepare(Xw, args.copula)
        M, P = gauss_te_matrix(Xz)
        S = gauss_shift_null(Xz)
        bias = S.mean(axis=0)
        Mraw.append(M)
        Meff.append(np.maximum(M - bias, 0.0))
        fsig.append((P[off] < 0.05).mean())
        null_sig.append((M[off] > S.max(axis=0)[off]).mean())
        raw_m.append(M[off].mean()); bias_m.append(bias[off].mean())
    raw_m, bias_m, fsig = map(np.array, (raw_m, bias_m, fsig))
    out(f"\nA. Edge-level evidence  [{(time.time()-t0)/60:.1f} min]")
    out(f"  edges with F-test p<0.05: mean {100*fsig.mean():.1f}%  (chance 5.0%)  "
        f"PD {100*fsig[y==1].mean():.1f}%  HC {100*fsig[y==0].mean():.1f}%")
    out(f"  edges above ALL 15 circular-shift nulls: mean {100*np.mean(null_sig):.1f}%  "
        f"(chance ~6.3%)")
    out(f"  raw TE mean={raw_m.mean():.4f} nats; shift-null (bias) mean={bias_m.mean():.4f}; "
        f"bias/raw={np.mean(bias_m/raw_m):.3f}")
    out(f"  raw  PD={raw_m[y==1].mean():.4f}  HC={raw_m[y==0].mean():.4f}   "
        f"bias PD={bias_m[y==1].mean():.4f}  HC={bias_m[y==0].mean():.4f}   "
        f"bias PD vs HC p={stats.mannwhitneyu(bias_m[y==1], bias_m[y==0])[1]:.4f}")

    # ── B. group comparison and classification ───────────────────────────────
    Xr = np.array([rc.ngs_from_matrix(M, seed=k) for k, M in enumerate(Mraw)])
    Xe = np.array([rc.ngs_from_matrix(M, seed=k) for k, M in enumerate(Meff)])
    out("\nB. Rank agreement raw vs corrected (Spearman):  " +
        "  ".join(f"{m}={stats.spearmanr(Xr[:,k], Xe[:,k])[0]:+.2f}"
                  for k, m in enumerate(NGS_CORE)))
    for tag, X in [('raw Gaussian TE', Xr), ('shift-null-corrected Gaussian TE', Xe)]:
        out(f"\n  Group comparison, {tag} (Mann-Whitney, m=7):")
        for m, p, q, b in rc.group_comparison(X, y):
            flag = ('FDR*' if q < 0.05 else '') + (' Bonf*' if b < 0.05 else '')
            out(f"    {m:<12} p={p:.4f}  q_FDR={q:.4f}  p_Bonf={b:.4f}  {flag}")
        auc, yt, ys = rc.nested_lopo_auc(X, y, return_scores=True)
        km, ks = rc.nested_kfold_auc(X, y)
        lo, hi = rc.bootstrap_ci(yt, ys)
        p, _ = rc.permutation_p(X, y, auc, args.perm)
        out(f"  Classification: LOPO AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
            f"10x5-fold={km:.4f}±{ks:.4f}  permutation p={p:.4f} ({args.perm})")

    # ── C. bootstrap reliability (raw Gaussian TE) ───────────────────────────
    out(f"\nC. Moving-block bootstrap, {args.reps} replicates, block {args.block}")
    rng = np.random.default_rng(12345)
    B = np.zeros((len(mats), args.reps, len(NGS_CORE)))
    t0 = time.time()
    for s, Xw in enumerate(mats):
        for r in range(args.reps):
            idx = block_resample(len(Xw), args.block, rng)
            M, _ = gauss_te_matrix(prepare(Xw[idx], args.copula))
            B[s, r] = rc.ngs_from_matrix(M, seed=s * 100000 + r)
        if (s + 1) % 12 == 0:
            out(f"  {s+1}/{len(mats)} participants ({(time.time()-t0)/60:.1f} min)")
    out(f"  {'metric':<12} {'within-SD':>10} {'between-SD':>11} {'R':>6}   FDR-sig worlds")
    sig = {m: 0 for m in NGS_CORE}
    for r in range(args.reps):
        for m, p, q, b in rc.group_comparison(B[:, r, :], y):
            sig[m] += q < 0.05
    for k, m in enumerate(NGS_CORE):
        wv = B[:, :, k].var(axis=1, ddof=1).mean()
        bv = B[:, :, k].mean(axis=1).var(ddof=1)
        star = '  <- retained in manuscript' if m in SELECTED else ''
        out(f"  {m:<12} {np.sqrt(wv):>10.4g} {np.sqrt(bv):>11.4g} {bv/(bv+wv+1e-30):>6.3f}"
            f"   {sig[m]/args.reps:6.1%}{star}")
    aucs = np.array([rc.nested_lopo_auc(B[:, r, :], y) for r in range(args.reps)])
    out(f"  AUC across worlds: median={np.median(aucs):.4f}  IQR=[{np.percentile(aucs,25):.4f}, "
        f"{np.percentile(aucs,75):.4f}]  P(AUC>0.5)={np.mean(aucs>0.5):.3f}")

    rows = []
    for k, pid in enumerate(ids):
        row = {'participant_id': pid, 'label': int(y[k])}
        row.update({f'raw_{m}': Xr[k, j] for j, m in enumerate(NGS_CORE)})
        row.update({f'eff_{m}': Xe[k, j] for j, m in enumerate(NGS_CORE)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'r2_gaussian_te_ngs{sfx}.csv', index=False)
    out(f"\nSaved r2_gaussian_te_results{sfx}.txt and r2_gaussian_te_ngs{sfx}.csv")
    log.close()


if __name__ == '__main__':
    main()
