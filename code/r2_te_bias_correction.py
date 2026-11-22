"""
R2.1 / R4.1 — Small-sample bias of the plug-in TE estimator and its effect
on the NGS descriptors and on classification.

What it does (reference configuration unchanged: W=20, B=8, lag=1, prune=25th pct):
  1. Reproduces the reference raw-TE result (LOPO AUC should be ~0.6150).
  2. For every participant and every directed edge, estimates the finite-sample
     bias of the plug-in estimator from source-shuffled surrogates and forms the
     bias-corrected ("effective") TE = TE_raw - mean(TE_surrogate), floored at 0.
  3. Reports the size of the bias (mean surrogate TE vs raw TE), how uniform it
     is across edges/participants, and how many edges exceed their own
     surrogate 95th percentile.
  4. Rebuilds every NIG from the corrected weights with the same pruning rule,
     recomputes NGS, and reports: per-metric rank agreement with the raw NGS
     (Spearman across participants), the FDR-corrected group comparison, nested
     LOPO AUC, repeated 10x5-fold AUC, subject-level bootstrap CI and a
     full-procedure permutation p-value.
  5. Repeats steps 2-4 for B in {3, 4} (smaller tables, 27/64 cells) so the
     response can show whether conclusions depend on the bin count once bias
     is removed.

Usage:
  python r2_te_bias_correction.py --pahaw ./PaHaW/PaHaW_public \
      --corpus ./PaHaW/PaHaW_files/corpus_PaHaW.xlsx [--surrogates 200] [--perm 1000]
Outputs: r2_te_bias_results.txt and r2_te_bias_ngs.csv in the current folder.
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import r2_common as rc
from r2_common import NGS_CORE


def per_subject(Xw, n_surr, bins, seed, null='shuffle'):
    rng = np.random.default_rng(seed)
    Xz = StandardScaler().fit_transform(Xw)
    D = rc.discretise_columns(Xz, bins)
    M_raw = rc.te_matrix_codes(D)
    S = (rc.surrogate_te_shift(D) if null == 'shift'
         else rc.surrogate_te(D, n_surr, rng))
    bias = S.mean(axis=0)
    M_eff = np.maximum(M_raw - bias, 0.0)
    off = ~np.eye(M_raw.shape[0], dtype=bool)
    p95 = np.percentile(S, 95, axis=0)
    return {
        'M_raw': M_raw, 'M_eff': M_eff,
        'raw_mean': M_raw[off].mean(), 'bias_mean': bias[off].mean(),
        'bias_cv_edges': bias[off].std() / (bias[off].mean() + 1e-12),
        'frac_bias': (bias[off] / (M_raw[off] + 1e-12)).clip(0, 5).mean(),
        'n_sig_edges': int((M_raw[off] > p95[off]).sum()),
        'n_pos_eff': int((M_eff[off] > 0).sum()),
    }


def evaluate(name, X, y, n_perm, out):
    auc, yt, ys = rc.nested_lopo_auc(X, y, return_scores=True)
    km, ks = rc.nested_kfold_auc(X, y)
    lo, hi = rc.bootstrap_ci(yt, ys)
    line = (f"{name:<34} LOPO AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
            f"10x5-fold AUC={km:.4f}±{ks:.4f}")
    if n_perm:
        p, _ = rc.permutation_p(X, y, auc, n_perm)
        line += f"  permutation p={p:.4f} ({n_perm} perms)"
    out(line)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--surrogates', type=int, default=200)
    ap.add_argument('--perm', type=int, default=1000)
    ap.add_argument('--bins', type=int, nargs='*', default=[8, 3, 4])
    ap.add_argument('--null', choices=['shuffle', 'shift'], default='shuffle',
                    help='shuffle: permuted source; shift: circularly time-shifted '
                         'source (keeps source autocorrelation; all shifts 3..17)')
    args = ap.parse_args()

    suffix = '' if args.null == 'shuffle' else '_shift'
    log = open(f'r2_te_bias_results{suffix}.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    ids, y, mats, _ = rc.load_cohort(args.pahaw, args.corpus)
    out(f"Cohort: n={len(y)}  PD={int(y.sum())}  HC={int((y==0).sum())}")
    out(f"Windows W={rc.N_WINDOWS}, lag={rc.TE_LAG} -> {rc.N_WINDOWS - rc.TE_LAG} "
        f"transitions per participant; prune={rc.PRUNE_PCT}th pct; "
        f"null={args.null}" + (f", surrogates={args.surrogates}" if args.null == 'shuffle' else ", all circular shifts 3..17"))

    err = rc.check_te_equivalence(StandardScaler().fit_transform(mats[0]))
    out(f"Fast TE path vs ngs_pipeline_v2.transfer_entropy: max |diff| = {err:.2e}")
    if err > 1e-10:
        out("ABORT: fast path does not reproduce the published estimator.")
        sys.exit(1)

    all_rows = []
    for bins in args.bins:
        out('\n' + '=' * 78)
        out(f"B = {bins} equal-frequency bins  "
            f"(joint table p(y_t+1, y_t, x_t) has {bins**3} cells for 19 samples)")
        out('=' * 78)
        t0 = time.time()
        res = [per_subject(Xw, args.surrogates, bins, seed=k, null=args.null)
               for k, Xw in enumerate(mats)]
        out(f"[TE + surrogates computed in {(time.time()-t0)/60:.1f} min]")

        raw = np.array([r['raw_mean'] for r in res])
        bias = np.array([r['bias_mean'] for r in res])
        out("\nEstimator bias (per-participant means over the 90 directed edges):")
        out(f"  raw plug-in TE          mean={raw.mean():.4f} nats  (SD across participants {raw.std():.4f})")
        out(f"  surrogate (bias) TE     mean={bias.mean():.4f} nats  (SD across participants {bias.std():.4f})")
        out(f"  bias / raw              mean={np.mean([r['frac_bias'] for r in res]):.3f}")
        out(f"  CV of bias across edges mean={np.mean([r['bias_cv_edges'] for r in res]):.3f}  "
            f"(small = bias nearly uniform across edges)")
        out(f"  bias PD vs HC           Mann-Whitney p="
            f"{stats.mannwhitneyu(bias[y==1], bias[y==0])[1]:.4f}  "
            f"(non-significant = bias does not differ by group)")
        sig = np.array([r['n_sig_edges'] for r in res])
        out(f"  edges above own surrogate 95th pct: mean {sig.mean():.1f}/90 "
            f"(chance ~4.5)")
        out(f"  bias mean PD={bias[y==1].mean():.4f}  HC={bias[y==0].mean():.4f}; "
            f"raw mean PD={raw[y==1].mean():.4f}  HC={raw[y==0].mean():.4f}")
        pos = np.array([r['n_pos_eff'] for r in res])
        out(f"  edges with positive corrected TE:   mean {pos.mean():.1f}/90")

        X_raw = np.array([rc.ngs_from_matrix(r['M_raw'], seed=k) for k, r in enumerate(res)])
        X_eff = np.array([rc.ngs_from_matrix(r['M_eff'], seed=k) for k, r in enumerate(res)])

        out("\nRank agreement of each NGS metric, raw vs bias-corrected (Spearman across participants):")
        for k, m in enumerate(NGS_CORE):
            rho = stats.spearmanr(X_raw[:, k], X_eff[:, k])[0]
            out(f"  {m:<12} rho={rho:+.3f}")

        for tag, X in [('raw', X_raw), ('bias-corrected', X_eff)]:
            out(f"\nGroup comparison ({tag} TE), Mann-Whitney with FDR / Bonferroni (m=7):")
            for m, p, q, b in rc.group_comparison(X, y):
                flag = ('FDR*' if q < 0.05 else '') + (' Bonf*' if b < 0.05 else '')
                out(f"  {m:<12} p={p:.4f}  q_FDR={q:.4f}  p_Bonf={b:.4f}  {flag}")

        out("\nClassification (nested fold-safe selection, LR):")
        evaluate(f"B={bins} raw TE", X_raw, y, args.perm if bins == 8 else 0, out)
        evaluate(f"B={bins} bias-corrected TE", X_eff, y, args.perm, out)

        for k, pid in enumerate(ids):
            row = {'participant_id': pid, 'label': int(y[k]), 'bins': bins,
                   'raw_te_mean': res[k]['raw_mean'], 'bias_mean': res[k]['bias_mean']}
            row.update({f'raw_{m}': X_raw[k, j] for j, m in enumerate(NGS_CORE)})
            row.update({f'eff_{m}': X_eff[k, j] for j, m in enumerate(NGS_CORE)})
            all_rows.append(row)

    pd.DataFrame(all_rows).to_csv(f'r2_te_bias_ngs{suffix}.csv', index=False)
    out(f"\nSaved r2_te_bias_results{suffix}.txt and r2_te_bias_ngs{suffix}.csv")
    log.close()


if __name__ == '__main__':
    main()
