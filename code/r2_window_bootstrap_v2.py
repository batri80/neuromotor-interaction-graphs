"""
Window-level bootstrap with NO artificial transitions (corrected version of
r2_window_bootstrap.py and section C of r2_gaussian_te.py).

The moving-block bootstrap resamples blocks of L consecutive windows. In the
previous version the resampled blocks were concatenated and every adjacent
pair was used as a transition, so the joins between blocks (e.g. window 4 ->
window 9) entered the TE estimate as if they were real. Here lagged pairs are
formed only WITHIN each sampled block; transitions across block joins are
excluded. With 20 windows, L=4 gives 5 blocks x 3 = 15 genuine transitions
per replicate (L=5: 4 x 4 = 16).

Estimators: plugin (8-bin equal-frequency, as in the manuscript), gauss
(linear-Gaussian TE), copula (Gaussian-copula linear TE).

Reported (same layout as before, so values are directly comparable):
  descriptor reliability R = var_between / (var_between + mean var_within),
  % of bootstrap worlds with an FDR-significant group difference,
  LOPO AUC across bootstrap worlds (a temporal-perturbation analysis, not a
  participant-level bootstrap).

Usage:
  python r2_window_bootstrap_v2.py --estimator plugin
  python r2_window_bootstrap_v2.py --estimator gauss
  python r2_window_bootstrap_v2.py --estimator copula
  [--reps 100] [--block 4]
Output: r2_window_bootstrap_v2_<estimator>_L<block>.txt
"""

import argparse
import time
from collections import Counter

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

import r2_common as rc
from r2_common import NGS_CORE

SELECTED = ['Eff', 'E_spectral', 'Ricci_OR']


def sample_blocks(n, L, rng):
    """Return list of index arrays, one per sampled block (enough blocks to
    cover n windows)."""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=k)
    return [np.arange(s, s + L) for s in starts]


def within_block_rows(blocks, lag):
    """Positions t in the concatenated sequence such that t and t-lag come
    from the same block."""
    rows, pos = [], 0
    for b in blocks:
        rows.extend(range(pos + lag, pos + len(b)))
        pos += len(b)
    return np.array(rows, dtype=int)


# ── plug-in TE on selected rows (same formula as ngs_pipeline_v2) ───────────
def te_plugin_rows(xd, yd, rows, lag):
    yt, yt1, xt1 = yd[rows], yd[rows - lag], xd[rows - lag]
    N = len(rows)
    if N < 2:
        return 0.0
    jyy = Counter(zip(yt, yt1)); jxxy = Counter(zip(yt, yt1, xt1))
    my1 = Counter(yt1); jyx1 = Counter(zip(yt1, xt1))
    te = 0.0
    for (a, b, c), cnt in jxxy.items():
        p = cnt / N
        r = (p * (my1[b] / N)) / ((jyy[(a, b)] / N) * (jyx1[(b, c)] / N))
        if r > 0:
            te += p * np.log(r)
    return max(te, 0.0)


# ── linear-Gaussian TE on selected rows ─────────────────────────────────────
def te_gauss_rows(x, y, rows, lag):
    yt, yp, xp = y[rows], y[rows - lag], x[rows - lag]
    one = np.ones(len(rows))
    def rss(A):
        c, *_ = np.linalg.lstsq(A, yt, rcond=None)
        r = yt - A @ c
        return float(r @ r)
    rr = rss(np.column_stack([one, yp])); rf = rss(np.column_stack([one, yp, xp]))
    if rr <= 1e-15 or rf <= 1e-15:
        return 0.0
    return max(0.5 * np.log(rr / rf), 0.0)


def te_matrix_rows(Xw_res, rows, estimator, lag=rc.TE_LAG):
    Xz = StandardScaler().fit_transform(Xw_res)
    n = Xz.shape[1]
    M = np.zeros((n, n))
    if estimator == 'plugin':
        D = rc.discretise_columns(Xz)
        for i in range(n):
            for j in range(n):
                if i != j:
                    M[i, j] = te_plugin_rows(D[:, i], D[:, j], rows, lag)
        return M
    if estimator == 'copula':
        R = np.apply_along_axis(stats.rankdata, 0, Xz)
        Xz = StandardScaler().fit_transform(stats.norm.ppf(R / (len(Xz) + 1)))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = te_gauss_rows(Xz[:, i], Xz[:, j], rows, lag)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--estimator', choices=['plugin', 'gauss', 'copula'], default='plugin')
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--block', type=int, default=4)
    args = ap.parse_args()

    fname = f'r2_window_bootstrap_v2_{args.estimator}_L{args.block}.txt'
    log = open(fname, 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    ids, y, mats, _ = rc.load_cohort(args.pahaw, args.corpus)
    rng = np.random.default_rng(12345)
    probe = within_block_rows(sample_blocks(rc.N_WINDOWS, args.block, rng), rc.TE_LAG)
    out(f"Estimator={args.estimator}; n={len(y)} (PD={int(y.sum())}, HC={int((y==0).sum())}); "
        f"{args.reps} replicates; block length {args.block}; "
        f"{len(probe)} genuine within-block transitions per replicate "
        f"(junction transitions excluded; original design: 19)")

    rng = np.random.default_rng(12345)
    B = np.zeros((len(mats), args.reps, len(NGS_CORE)))
    t0 = time.time()
    for s, Xw in enumerate(mats):
        for r in range(args.reps):
            blocks = sample_blocks(len(Xw), args.block, rng)
            idx = np.concatenate(blocks)
            rows = within_block_rows(blocks, rc.TE_LAG)
            M = te_matrix_rows(Xw[idx], rows, args.estimator)
            B[s, r] = rc.ngs_from_matrix(M, seed=s * 100000 + r)
        if (s + 1) % 12 == 0:
            out(f"  {s+1}/{len(mats)} participants ({(time.time()-t0)/60:.1f} min)")

    out(f"\n  {'metric':<12} {'within-SD':>10} {'between-SD':>11} {'R':>6}   FDR-sig worlds")
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
    out(f"\n  LOPO AUC across bootstrap worlds (temporal perturbation, not a participant "
        f"bootstrap): median={np.median(aucs):.4f}  IQR=[{np.percentile(aucs,25):.4f}, "
        f"{np.percentile(aucs,75):.4f}]  P(AUC>0.5)={np.mean(aucs>0.5):.3f}")
    out(f"\nSaved {fname}  [total {(time.time()-t0)/60:.1f} min]")
    log.close()


if __name__ == '__main__':
    main()
