"""
Length-matched split-half reliability of native-resolution NGS.

Concern: in r2_native_nig.py every descriptor correlates with recording length
(|rho| 0.38-0.83), and both halves of a participant share that length, so the
split-half ICC (0.51-0.90) could partly reflect length rather than a stable
participant trait. Here every segment compared has EXACTLY the same number of
on-surface samples for every participant, so length cannot contribute.

Designs (same estimator, order, copula, pruning, NGS as r2_native_nig.py):
  D1  start-matched: first 2L samples of each recording, split into two
      consecutive segments of L samples (L = half the shortest recording,
      so all 72 participants are included).
  D2  spread-matched: segment A = first L samples of the first half of the
      recording, segment B = first L samples of the second half. The two
      segments come from different parts of the spiral but have equal length.
  D3  longer segments: as D2 with L = 1000 for the participants whose halves
      contain at least 1000 samples (subset; n reported).

For each design: per-metric ICC(1), Spearman r and Spearman-Brown r_SB
between the two segments, plus the correlation of each segment's descriptors
with the participant's full recording length (should now be ~0 if length was
the driver).

Usage:  python r2_native_reliability_lenmatched.py [--order 5]
Output: r2_native_reliability_lenmatched_p5.txt
"""

import argparse
import time

import numpy as np
from scipy import stats

import r2_common as rc
from r2_common import NGS_CORE
import ngs_pipeline_v2 as base
from r2_native_nig import participant_series, transform, te_matrix, icc1


def segment(S, bounds, a, b, p):
    """Samples a..b-1 of the concatenated series, with stroke bounds re-based."""
    segb = [(max(s, a) - a, min(e, b) - a) for s, e in bounds
            if e > a and s < b and min(e, b) - max(s, a) > p]
    return S[a:b], segb


def ngs_of(S, bounds, p, seed):
    Z = transform(S, True)
    M = te_matrix(Z, bounds, p)[0]
    return rc.ngs_from_matrix(M, seed=seed)


def boot_ci(a, b, n_boot=2000, seed=0):
    """Participant-level bootstrap 95% CIs for ICC(1) and Spearman r."""
    rng = np.random.default_rng(seed)
    iccs, rs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        iccs.append(icc1(a[i], b[i])); rs.append(stats.spearmanr(a[i], b[i])[0])
    return np.nanpercentile(iccs, [2.5, 97.5]), np.nanpercentile(rs, [2.5, 97.5])


def report(out, name, A, B, lengths):
    out(f"\n{name}  (n={len(A)})")
    out("   ICC(1) = one-way random effects, absolute agreement, single measure;")
    out("   r = Spearman rank correlation; r_SB = 2r/(1+r); [95% participant-bootstrap CI]")
    for k, m in enumerate(NGS_CORE):
        r = stats.spearmanr(A[:, k], B[:, k])[0]
        rl = stats.spearmanr(A[:, k], lengths)[0]
        ci_i, ci_r = boot_ci(A[:, k], B[:, k], seed=k)
        out(f"   {m:<12} ICC={icc1(A[:,k], B[:,k]):6.3f} [{ci_i[0]:6.3f},{ci_i[1]:6.3f}]  "
            f"r={r:6.3f} [{ci_r[0]:6.3f},{ci_r[1]:6.3f}]  "
            f"r_SB={2*r/(1+r) if r > -1 else np.nan:6.3f}  rho(length)={rl:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--order', type=int, default=5)
    ap.add_argument('--long', type=int, default=1000)
    args = ap.parse_args()
    p = args.order

    log = open(f'r2_native_reliability_lenmatched_p{p}.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    part_df = base.load_pahaw(args.pahaw, args.corpus)
    series = []
    for _, row in part_df.iterrows():
        svc = row['svc_data']
        if svc is None or rc.window_features(svc)[0] is None:
            continue
        S, bounds = participant_series(svc, 2 * p + 5)
        if S is not None:
            series.append((S, bounds))
    lengths = np.array([len(S) for S, _ in series])
    L = int(lengths.min()) // 2
    out(f"Length-matched split-half reliability, native NIG (Gaussian-copula TE, p={p})")
    out(f"n={len(series)}; shortest recording {lengths.min()} samples -> L={L} samples per segment")

    t0 = time.time()
    A1, B1, A2, B2 = [], [], [], []
    for k, (S, b) in enumerate(series):
        A1.append(ngs_of(*segment(S, b, 0, L, p), p, k))
        B1.append(ngs_of(*segment(S, b, L, 2 * L, p), p, k))
        h = len(S) // 2
        A2.append(ngs_of(*segment(S, b, 0, L, p), p, k))
        B2.append(ngs_of(*segment(S, b, h, h + L, p), p, k))
    report(out, f"D1 start-matched: samples 1-{L} vs {L+1}-{2*L}",
           np.array(A1), np.array(B1), lengths)
    report(out, f"D2 spread-matched: first {L} samples of each half",
           np.array(A2), np.array(B2), lengths)

    A3, B3, L3 = [], [], []
    for k, (S, b) in enumerate(series):
        h = len(S) // 2
        if h >= args.long:
            A3.append(ngs_of(*segment(S, b, 0, args.long, p), p, k))
            B3.append(ngs_of(*segment(S, b, h, h + args.long, p), p, k))
            L3.append(len(S))
    if len(A3) >= 10:
        report(out, f"D3 spread-matched, L={args.long} (participants with halves >= {args.long})",
               np.array(A3), np.array(B3), np.array(L3))
    out("\nReference (r2_native_nig.py, unmatched halves): ICC "
        "H_graph 0.83, Eff 0.86, Modularity 0.67, Lambda2 0.90, SmallWorld 0.51, "
        "E_spectral 0.66, Ricci 0.90")
    out(f"\n[total {(time.time()-t0)/60:.1f} min]")
    log.close()


if __name__ == '__main__':
    main()
