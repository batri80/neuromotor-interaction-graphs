"""
H&Y correlation of the reliable multi-task descriptors (Appendix B.3),
for the revised Section 5.12. Spearman rho with H&Y in the PD group, raw and
partial on log(number of segments) (pen-down writing duration), BH-FDR over
the seven descriptors.

Usage:  python r2_hy_multitask.py [--csv r2_multitask_nig_L500_p5.csv]
                                  [--corpus ./PaHaW/PaHaW_files/corpus_PaHaW.xlsx]
Output: r2_hy_multitask_results.txt
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
from ngs_pipeline_v2 import NGS_CORE

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='r2_multitask_nig_L500_p5.csv')
ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
a = ap.parse_args()
P = pd.read_csv(a.csv, dtype={'participant_id': str}).set_index('participant_id')
c = pd.read_excel(a.corpus); c['participant_id'] = c['ID'].apply(lambda v: f"{int(v):05d}")
P = P.join(c.set_index('participant_id')['UPDRS V'].rename('HY'))
P = P[(P.label == 1) & P.HY.notna()]
hy, ls = P.HY.values.astype(float), np.log(P.n_seg.values.astype(float))

def partial_spearman(x, y, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, _ = stats.pearsonr(ex, ey)
    n = len(x); t = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), n - 3)

rows = []
for m in NGS_CORE:
    r, p = stats.spearmanr(P[m], hy)
    rp, pp = partial_spearman(P[m].values, hy, ls)
    rows.append((m, r, p, rp, pp))
q = bh_fdr([r[2] for r in rows])
qp = bh_fdr([r[4] for r in rows])
with open('r2_hy_multitask_results.txt', 'w') as f:
    def out(s): print(s); f.write(s + '\n')
    out(f"PD participants with H&Y: n={len(P)}   H&Y counts: {P.HY.value_counts().sort_index().to_dict()}")
    out(f"Spearman rho of H&Y with log(segments): {stats.spearmanr(hy, ls)[0]:+.3f} (p={stats.spearmanr(hy, ls)[1]:.3f})")
    out(f"{'descriptor':<12}{'rho':>8}{'p':>8}{'q':>8}   {'rho|dur':>8}{'p':>8}{'q':>8}")
    for (m, r, p, rp, pp), qq, qq2 in zip(rows, q, qp):
        out(f"{m:<12}{r:>+8.3f}{p:>8.3f}{qq:>8.3f}   {rp:>+8.3f}{pp:>8.3f}{qq2:>8.3f}")
