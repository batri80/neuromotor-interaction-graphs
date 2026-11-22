"""
Tests whether the ORIGINAL per-subject population-insertion construction
(build_subject_ngs, used to generate Table 4/12's group-comparison
statistics in the original submission) produces artificially deflated
within-group variance -- which would mechanically inflate the reported
significance (d=-1.00, p<0.001) independent of any real biological
effect.

Mechanism under test: build_subject_ngs builds every subject's graph
from their ENTIRE group (all 36 members) plus a duplicated row of
themselves. Since ~97% of the input is shared across all subjects in
the same group, their resulting NGS values may cluster artificially
tightly, deflating within-group SD and inflating the between-group test.

Fix under test: a fold-safe (leave-self-out) reconstruction, where each
subject's reference population EXCLUDES themselves entirely (their own
group, minus themselves), using an insertion-delta relative to that
excluded-self reference. This uses the exact same logic validated
earlier for the classification leakage fix, applied here to
the group-comparison variance question specifically.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
feat_cols = CONFIG['feature_names']
X_all = part_df[feat_cols].values
labels = part_df['label'].values
n = len(part_df)

pd_idx = [i for i in range(n) if labels[i] == 'PD']
hc_idx = [i for i in range(n) if labels[i] == 'HC']


# ---------- METHOD A: original leaky construction (matches Table 4/12) ----------
def original_leaky_ngs(subject_idx, group_idx):
    """Exact original construction: full group (including subject) + duplicate."""
    grp_mat = X_all[group_idx]
    sub_mat = np.vstack([grp_mat, X_all[subject_idx:subject_idx+1]])
    Xz = StandardScaler().fit_transform(sub_mat)
    G = build_nig(Xz, label='leaky')
    return extract_ngs(G)


# ---------- METHOD B: fold-safe, leave-self-out construction ----------
def foldsafe_ngs_value(subject_idx, group_idx):
    """Reference population EXCLUDES the subject entirely; insertion delta
    relative to that excluded-self reference (same logic as the
    classification fix validated earlier)."""
    ref_idx = [j for j in group_idx if j != subject_idx]
    X_ref = X_all[ref_idx]
    Xz_ref = StandardScaler().fit_transform(X_ref)
    G_ref = build_nig(Xz_ref, label='ref')
    ngs_ref = extract_ngs(G_ref)

    X_aug = np.vstack([X_ref, X_all[subject_idx:subject_idx+1]])
    Xz_aug = StandardScaler().fit_transform(X_aug)
    G_aug = build_nig(Xz_aug, label='aug')
    ngs_aug = extract_ngs(G_aug)

    return {k: ngs_aug.get(k, 0.0) - ngs_ref.get(k, 0.0) for k in NGS_CORE}


if __name__ == '__main__':
    print("Computing original leaky per-subject NGS (matches Table 4/12 method)...")
    leaky_pd = [original_leaky_ngs(i, pd_idx) for i in pd_idx]
    leaky_hc = [original_leaky_ngs(i, hc_idx) for i in hc_idx]

    print("Computing fold-safe (leave-self-out delta) per-subject NGS...")
    fs_pd = [foldsafe_ngs_value(i, pd_idx) for i in pd_idx]
    fs_hc = [foldsafe_ngs_value(i, hc_idx) for i in hc_idx]

    print("\n" + "="*90)
    print("SMALL-WORLD COEFFICIENT: variance and significance comparison")
    print("="*90)

    metric = 'SmallWorld'
    leaky_pd_vals = np.array([d[metric] for d in leaky_pd])
    leaky_hc_vals = np.array([d[metric] for d in leaky_hc])
    fs_pd_vals = np.array([d[metric] for d in fs_pd])
    fs_hc_vals = np.array([d[metric] for d in fs_hc])

    def cohens_d(a, b):
        n1, n2 = len(a), len(b)
        pooled_std = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
        return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

    print("\n--- METHOD A: Original leaky construction (Table 4/12 method) ---")
    print(f"PD: mean={leaky_pd_vals.mean():.4f} SD={leaky_pd_vals.std(ddof=1):.4f}")
    print(f"HC: mean={leaky_hc_vals.mean():.4f} SD={leaky_hc_vals.std(ddof=1):.4f}")
    d_leaky = cohens_d(leaky_pd_vals, leaky_hc_vals)
    u, p_leaky = stats.mannwhitneyu(leaky_pd_vals, leaky_hc_vals)
    print(f"Cohen's d = {d_leaky:.3f}   Mann-Whitney p = {p_leaky:.6f}")

    print("\n--- METHOD B: Fold-safe (leave-self-out delta) construction ---")
    print(f"PD: mean={fs_pd_vals.mean():.4f} SD={fs_pd_vals.std(ddof=1):.4f}")
    print(f"HC: mean={fs_hc_vals.mean():.4f} SD={fs_hc_vals.std(ddof=1):.4f}")
    d_fs = cohens_d(fs_pd_vals, fs_hc_vals)
    u2, p_fs = stats.mannwhitneyu(fs_pd_vals, fs_hc_vals)
    print(f"Cohen's d = {d_fs:.3f}   Mann-Whitney p = {p_fs:.6f}")

    print("\n" + "="*90)
    print("INTERPRETATION")
    print("="*90)
    sd_ratio_pd = fs_pd_vals.std(ddof=1) / leaky_pd_vals.std(ddof=1) if leaky_pd_vals.std(ddof=1) > 0 else float('nan')
    sd_ratio_hc = fs_hc_vals.std(ddof=1) / leaky_hc_vals.std(ddof=1) if leaky_hc_vals.std(ddof=1) > 0 else float('nan')
    print(f"Fold-safe/leaky SD ratio -- PD: {sd_ratio_pd:.2f}x   HC: {sd_ratio_hc:.2f}x")
    print(f"(ratio > 1 means fold-safe shows MORE variance, i.e. the leaky method WAS")
    print(f" artificially deflating within-group variance, as hypothesised)")
    print(f"\nOriginal leaky significance:   d={d_leaky:.3f}, p={p_leaky:.6f}")
    print(f"Fold-safe significance:        d={d_fs:.3f}, p={p_fs:.6f}")
    print("\nIf fold-safe d/significance collapses toward the corrected within-subject")
    print("result (d=+0.066, p=0.665, non-significant) rather than replicating the")
    print("original claim, that confirms the population-level 'Finding 3' claim was")
    print("ALSO an artifact of the leaky construction, not an independently valid result.")
