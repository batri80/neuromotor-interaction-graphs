"""
Table 5 (window-level plug-in NGS, W=20, B=8) repeated with age as a
covariate. For each descriptor: Mann-Whitney (reproduces Table 5) and an
OLS of the rank-transformed descriptor on group + age (rank ANCOVA);
BH-FDR over the seven descriptors.

Usage:  python r2_table5_age.py [--pahaw ...] [--corpus ...]
Output: r2_table5_age_results.txt
"""
import argparse
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import r2_common as rc
from ngs_pipeline_v2 import NGS_CORE

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)

def ols_group(r, g, z):
    """OLS of r on [1, g, z]; returns coefficient and two-sided p for g."""
    Xd = np.column_stack([np.ones_like(r), g, z])
    beta, *_ = np.linalg.lstsq(Xd, r, rcond=None)
    res = r - Xd @ beta; dof = len(r) - Xd.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(Xd.T @ Xd)
    t = beta[1] / np.sqrt(cov[1, 1])
    return beta[1], 2 * stats.t.sf(abs(t), dof)

ap = argparse.ArgumentParser()
ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
a = ap.parse_args()
ids, y, mats, part_df = rc.load_cohort(a.pahaw, a.corpus)
age_map = dict(zip(part_df.participant_id, part_df.age))
age = np.array([age_map[i] for i in ids], float)
N = np.array([rc.ngs_from_matrix(rc.te_matrix_codes(rc.discretise_columns(StandardScaler().fit_transform(Xw)))) for Xw in mats])
f = open('r2_table5_age_results.txt', 'w')
def out(s=''): print(s); f.write(s + '\n')
out(f"n={len(y)}  age PD={age[y==1].mean():.1f} HC={age[y==0].mean():.1f}  MW p={stats.mannwhitneyu(age[y==1], age[y==0])[1]:.4f}")
p_mw, p_adj, b_adj = [], [], []
for k, m in enumerate(NGS_CORE):
    p_mw.append(stats.mannwhitneyu(N[y==1, k], N[y==0, k], alternative='two-sided')[1])
    r = stats.rankdata(N[:, k]); r = (r - r.mean()) / r.std()
    b, p = ols_group(r, y.astype(float), age)
    p_adj.append(p); b_adj.append(b)
q_mw = bh_fdr(p_mw); q_adj = bh_fdr(p_adj)
out(f"{'descriptor':<12}{'MW p':>8}{'q':>8}   {'group|age (rank SD)':>20}{'p':>8}{'q':>8}")
for k, m in enumerate(NGS_CORE):
    out(f"{m:<12}{p_mw[k]:>8.4f}{q_mw[k]:>8.4f}   {b_adj[k]:>+20.3f}{p_adj[k]:>8.4f}{q_adj[k]:>8.4f}")
f.close()
