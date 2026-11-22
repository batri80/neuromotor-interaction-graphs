"""
Tempo-adjusted (and tempo + age adjusted) group comparison of the
multi-task fixed-segment NIG descriptors.

Reads r2_multitask_nig_L500_p5.csv (written by r2_multitask_nig.py).

Pre-specified model:
  * each descriptor is rank-based inverse-normal transformed (robust to the
    heavy tails of E_spectral and Ricci);
  * covariate set A: log(number of 500-sample segments) = writing amount/tempo;
  * covariate set B: A + age (PD are older than HC in PaHaW);
  * group comparison: Mann-Whitney on the residuals (FDR / Bonferroni, m=7);
  * classification: nested LOPO and 10x5-fold where the covariate regression
    is fitted on the TRAINING fold only and applied to the held-out
    participant(s), followed by the same fold-safe Mann-Whitney selection and
    logistic regression; permutation test repeats the whole procedure.

Usage:  python r2_tempo_adjusted.py [--csv r2_multitask_nig_L500_p5.csv] [--perm 1000]
Output: r2_tempo_adjusted_results.txt
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import r2_common as rc
from r2_common import NGS_CORE


def inv_normal(x):
    return stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x))


def fold_fit_predict(Xtr, Ctr, ytr, Xte, Cte):
    reg = LinearRegression().fit(Ctr, Xtr)
    Rtr, Rte = Xtr - reg.predict(Ctr), Xte - reg.predict(Cte)
    keep = rc._select(Rtr, ytr)
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(sc.fit_transform(Rtr[:, keep]), ytr)
    return clf.predict_proba(sc.transform(Rte[:, keep]))[:, 1]


def lopo_auc(X, C, y):
    s = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        s[te] = fold_fit_predict(X[tr], C[tr], y[tr], X[te], C[te])
    return roc_auc_score(y, s), s


def kfold_auc(X, C, y):
    out = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                          random_state=42).split(X, y):
        out.append(roc_auc_score(y[te], fold_fit_predict(X[tr], C[tr], y[tr], X[te], C[te])))
    return np.mean(out), np.std(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='r2_multitask_nig_L500_p5.csv')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--perm', type=int, default=1000)
    args = ap.parse_args()

    log = open('r2_tempo_adjusted_results.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    P = pd.read_csv(args.csv, dtype={'participant_id': str}).set_index('participant_id')
    corpus = pd.read_excel(args.corpus)
    corpus['participant_id'] = corpus['ID'].apply(lambda v: f"{int(v):05d}")
    P = P.join(corpus.set_index('participant_id')['Age'])
    y = P.label.values.astype(int)
    X = np.column_stack([inv_normal(P[m].values) for m in NGS_CORE])
    logseg = np.log(P.n_seg.values.astype(float))
    age = P.Age.values.astype(float)

    out(f"n={len(y)} (PD={y.sum()}, HC={(y==0).sum()})")
    out(f"Covariates: log(segments) PD median={np.median(logseg[y==1]):.2f} HC={np.median(logseg[y==0]):.2f} "
        f"p={stats.mannwhitneyu(logseg[y==1], logseg[y==0])[1]:.4f};  "
        f"age PD={age[y==1].mean():.1f} HC={age[y==0].mean():.1f} "
        f"p={stats.mannwhitneyu(age[y==1], age[y==0])[1]:.4f}")
    out("Spearman rho of each descriptor with age: " + "  ".join(
        f"{m}={stats.spearmanr(P[m], age)[0]:+.2f}" for m in NGS_CORE))

    for tag, C in [('A: adjusted for log(segments)', logseg[:, None]),
                   ('B: adjusted for log(segments) + age', np.column_stack([logseg, age]))]:
        R = X - LinearRegression().fit(C, X).predict(C)
        out(f"\n{tag}")
        out("  Group comparison on residuals (Mann-Whitney, m=7):")
        for m, pv, q, b in rc.group_comparison(R, y):
            k = NGS_CORE.index(m)
            d = np.median(R[y == 1, k]) - np.median(R[y == 0, k])
            flag = ('FDR*' if q < 0.05 else '') + (' Bonf*' if b < 0.05 else '')
            out(f"    {m:<12} p={pv:.4f}  q_FDR={q:.4f}  p_Bonf={b:.4f}  "
                f"{'PD>HC' if d > 0 else 'PD<HC'}  {flag}")
        auc, s = lopo_auc(X, C, y)
        km, ks = kfold_auc(X, C, y)
        lo, hi = rc.bootstrap_ci(y, s)
        rng = np.random.default_rng(0)
        null = np.array([lopo_auc(X, C, rng.permutation(y))[0] for _ in range(args.perm)])
        pv = (1 + np.sum(null >= auc)) / (args.perm + 1)
        out(f"  Classification (in-fold covariate adjustment): LOPO AUC={auc:.4f}  "
            f"95% CI [{lo:.4f}, {hi:.4f}]  10x5-fold={km:.4f}±{ks:.4f}  "
            f"permutation p={pv:.4f} ({args.perm})")
    log.close()


if __name__ == '__main__':
    main()
