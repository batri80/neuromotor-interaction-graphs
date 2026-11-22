"""
Direct mechanism test: is population-graph instability an ESTIMATION
VARIANCE problem (which bagging would fix) or something bagging can't
touch?

Method: repeatedly draw random 80% subsamples of the PD group (simulating
what a K-fold training set looks like across different folds). For each
subsample, compute NGS two ways:
  (a) single-shot: one graph, as the original design did
  (b) bagged: 50 bootstrap resamples of the subsample, graph+NGS each,
      averaged

If bagging meaningfully reduces the SPREAD of NGS values across the M
simulated folds, that confirms estimation variance is the mechanism and
a full bagged pipeline is worth building. If not, bagging can't rescue
this, and that's a decisive, efficient answer.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

M = 30            # simulated folds
B = 50            # bootstrap resamples per bagged estimate
SUBSAMPLE_FRAC = 0.8


def single_shot_ngs(X):
    Xz = StandardScaler().fit_transform(X)
    G = build_nig(Xz, label='single')
    return extract_ngs(G)


def bagged_ngs(X, B, rng):
    n = len(X)
    accum = {k: [] for k in NGS_CORE}
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < 4:
            continue
        Xz = StandardScaler().fit_transform(X[idx])
        G = build_nig(Xz, label='boot')
        ngs = extract_ngs(G)
        for k in NGS_CORE:
            accum[k].append(ngs.get(k, 0.0))
    return {k: (np.mean(v) if v else 0.0) for k, v in accum.items()}


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    feat_cols = CONFIG['feature_names']

    for group_label in ['PD', 'HC']:
        print(f"\n{'='*80}\nGroup: {group_label}\n{'='*80}")
        Xg = part_df[part_df['label'] == group_label][feat_cols].values
        n = len(Xg)
        n_sub = int(n * SUBSAMPLE_FRAC)

        rng = np.random.RandomState(42)
        single_results = {k: [] for k in NGS_CORE}
        bagged_results = {k: [] for k in NGS_CORE}

        for m in range(M):
            idx = rng.choice(n, size=n_sub, replace=False)
            Xsub = Xg[idx]
            ss = single_shot_ngs(Xsub)
            bg = bagged_ngs(Xsub, B, rng)
            for k in NGS_CORE:
                single_results[k].append(ss.get(k, 0.0))
                bagged_results[k].append(bg.get(k, 0.0))

        print(f"{'Metric':<14}{'SingleShot std':>16}{'Bagged std':>14}{'Var. reduction':>16}")
        for k in NGS_CORE:
            ss_std = np.std(single_results[k])
            bg_std = np.std(bagged_results[k])
            reduction = 100 * (1 - bg_std / ss_std) if ss_std > 1e-9 else float('nan')
            print(f"{k:<14}{ss_std:>16.4f}{bg_std:>14.4f}{reduction:>15.1f}%")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("Large, consistent variance reduction (e.g. >30-40%) across most")
    print("metrics -> bagging targets a real estimation-variance mechanism;")
    print("worth building the full fold-safe bagged classification pipeline.")
    print("Small/inconsistent reduction -> the instability isn't primarily")
    print("an estimation-variance problem bagging can fix; stop here.")
