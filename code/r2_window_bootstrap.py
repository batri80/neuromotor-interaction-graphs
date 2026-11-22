"""
R4.1 (second half) — How TE estimation uncertainty propagates to the seven
NGS descriptors and to classification.

For each participant the sequence of 20 window feature vectors is resampled
with a moving-block bootstrap (block length L, default 4; blocks keep the
within-block temporal order so lag-1 transitions remain meaningful). Each
replicate is z-scored, its TE matrix and NIG are rebuilt with the reference
settings, and the NGS vector is recomputed.

Reported:
  * per metric: within-participant bootstrap SD, between-participant SD and a
    reliability coefficient R = var_between / (var_between + mean var_within)
    (an ICC(1)-type ratio; R near 1 = descriptor dominated by real
    between-participant differences, near 0 = dominated by estimation noise);
  * for the three metrics retained by nested selection, the fraction of
    bootstrap "worlds" in which the PD-HC difference stays FDR-significant;
  * the distribution of LOPO AUC over bootstrap worlds (in world r every
    participant contributes replicate r), i.e. how much the headline AUC
    moves purely because of TE estimation noise.

Run with --corrected to apply the surrogate bias correction inside every
replicate (slower; uses --surrogates per replicate).

Usage:
  python r2_window_bootstrap.py --pahaw ./PaHaW/PaHaW_public \
      --corpus ./PaHaW/PaHaW_files/corpus_PaHaW.xlsx [--reps 100] [--block 4]
      [--corrected --surrogates 50]
Outputs: r2_window_bootstrap_results.txt, r2_window_bootstrap_ngs.npz
"""

import argparse
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

import r2_common as rc
from r2_common import NGS_CORE

SELECTED = ['Eff', 'E_spectral', 'Ricci_OR']


def block_resample(n, L, rng):
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
    return idx


def replicate_ngs(Xw, idx, corrected, n_surr, rng, seed):
    Xz = StandardScaler().fit_transform(Xw[idx])
    D = rc.discretise_columns(Xz)
    M = rc.te_matrix_codes(D)
    if corrected:
        M = np.maximum(M - rc.surrogate_te(D, n_surr, rng).mean(axis=0), 0.0)
    return rc.ngs_from_matrix(M, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--block', type=int, default=4)
    ap.add_argument('--corrected', action='store_true')
    ap.add_argument('--surrogates', type=int, default=50)
    args = ap.parse_args()

    tag = 'bias-corrected' if args.corrected else 'raw'
    log = open(f'r2_window_bootstrap_results_{tag}.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    ids, y, mats, _ = rc.load_cohort(args.pahaw, args.corpus)
    n_sub, R = len(mats), args.reps
    out(f"Cohort n={n_sub} (PD={int(y.sum())}, HC={int((y==0).sum())}); "
        f"{R} moving-block bootstrap replicates per participant, block length "
        f"{args.block}; TE weights: {tag}")

    rng = np.random.default_rng(12345)
    B = np.zeros((n_sub, R, len(NGS_CORE)))
    t0 = time.time()
    for s, Xw in enumerate(mats):
        for r in range(R):
            idx = block_resample(len(Xw), args.block, rng)
            B[s, r] = replicate_ngs(Xw, idx, args.corrected, args.surrogates,
                                    rng, seed=s * 100000 + r)
        if (s + 1) % 12 == 0:
            out(f"  {s+1}/{n_sub} participants done ({(time.time()-t0)/60:.1f} min)")
    np.savez('r2_window_bootstrap_ngs_%s.npz' % tag, B=B, y=y, ids=np.array(ids))

    out("\nDescriptor reliability under TE re-estimation:")
    out(f"  {'metric':<12} {'within-SD':>10} {'between-SD':>11} {'R':>6}")
    for k, m in enumerate(NGS_CORE):
        within_var = B[:, :, k].var(axis=1, ddof=1).mean()
        between_var = B[:, :, k].mean(axis=1).var(ddof=1)
        Rk = between_var / (between_var + within_var + 1e-30)
        out(f"  {m:<12} {np.sqrt(within_var):>10.4g} {np.sqrt(between_var):>11.4g} {Rk:>6.3f}")

    out("\nGroup effect stability across bootstrap worlds (FDR q<0.05, m=7):")
    sig_counts = {m: 0 for m in NGS_CORE}
    for r in range(R):
        for m, p, q, b in rc.group_comparison(B[:, r, :], y):
            sig_counts[m] += q < 0.05
    for m in NGS_CORE:
        star = '  <- retained by nested selection' if m in SELECTED else ''
        out(f"  {m:<12} significant in {sig_counts[m]/R:6.1%} of worlds{star}")

    out("\nLOPO AUC across bootstrap worlds (nested selection, LR):")
    aucs = np.array([rc.nested_lopo_auc(B[:, r, :], y) for r in range(R)])
    out(f"  median={np.median(aucs):.4f}  IQR=[{np.percentile(aucs,25):.4f}, "
        f"{np.percentile(aucs,75):.4f}]  2.5-97.5%=[{np.percentile(aucs,2.5):.4f}, "
        f"{np.percentile(aucs,97.5):.4f}]  P(AUC>0.5)={np.mean(aucs>0.5):.3f}")
    Xbar = B.mean(axis=1)
    out(f"  AUC using each participant's bootstrap-mean NGS: "
        f"{rc.nested_lopo_auc(Xbar, y):.4f}")
    out(f"\nTotal time {(time.time()-t0)/60:.1f} min")
    log.close()


if __name__ == '__main__':
    main()
