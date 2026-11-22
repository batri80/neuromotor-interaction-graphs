"""
Confidence intervals for the multi-task participant-level reliability
(section 1b of r2_multitask_nig.py), computed from the saved segment file,
so nothing needs to be re-estimated.

For each descriptor: odd-numbered vs even-numbered segment means per
participant (length-matched by construction) ->
  ICC(1) (one-way random effects, absolute agreement, single measure),
  Spearman r, Spearman-Brown r_SB = 2r/(1+r),
  and the same ICC after a rank-based inverse-normal transform of each half
  (removes the influence of heavy tails, e.g. E_spectral and Ricci),
each with a 95% participant-bootstrap CI (2000 resamples).

Usage:  python r2_multitask_reliability_ci.py [--csv r2_multitask_nig_segments_L500_p5.csv]
Output: r2_multitask_reliability_ci.txt
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from r2_common import NGS_CORE
from r2_native_nig import icc1


def inv_normal(x):
    return stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x))


def stats_of(a, b):
    r = stats.spearmanr(a, b)[0]
    return (icc1(a, b), r, 2 * r / (1 + r) if r > -1 else np.nan,
            icc1(inv_normal(a), inv_normal(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='r2_multitask_nig_segments_L500_p5.csv')
    ap.add_argument('--min-segments', type=int, default=4)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()
    log = open('r2_multitask_reliability_ci.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    seg = pd.read_csv(args.csv, dtype={'participant_id': str})
    per = seg.groupby('participant_id').size()
    seg = seg[seg.participant_id.isin(per[per >= args.min_segments].index)]
    seg = seg.sort_values(['participant_id', 'task', 'seg']).copy()
    seg['ordinal'] = seg.groupby('participant_id').cumcount()
    odd = seg[seg.ordinal % 2 == 0].groupby('participant_id')[NGS_CORE].mean()
    even = seg[seg.ordinal % 2 == 1].groupby('participant_id')[NGS_CORE].mean()
    ids = odd.index.intersection(even.index)
    out(f"Multi-task participant-level reliability, odd vs even segment means (n={len(ids)})")
    out("ICC(1): one-way random effects, absolute agreement, single measure; r: Spearman;")
    out("r_SB = 2r/(1+r); ICC_rank: ICC(1) after rank-based inverse-normal transform;")
    out(f"[95% participant-bootstrap CI, {args.boot} resamples]\n")
    rng = np.random.default_rng(0)
    for m in NGS_CORE:
        a, b = odd.loc[ids, m].values, even.loc[ids, m].values
        est = stats_of(a, b)
        bs = np.array([stats_of(a[i], b[i]) for i in
                       (rng.integers(0, len(a), len(a)) for _ in range(args.boot))])
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        out(f"{m:<12} ICC={est[0]:6.3f} [{lo[0]:6.3f},{hi[0]:6.3f}]  "
            f"r={est[1]:6.3f} [{lo[1]:6.3f},{hi[1]:6.3f}]  "
            f"r_SB={est[2]:6.3f} [{lo[2]:6.3f},{hi[2]:6.3f}]  "
            f"ICC_rank={est[3]:6.3f} [{lo[3]:6.3f},{hi[3]:6.3f}]")
    log.close()


if __name__ == '__main__':
    main()
