"""
Multi-task, fixed-segment Neuromotor Interaction Graphs.

Diagnosis from the round-2 analyses:
  (1) one spiral gives one noisy graph per participant;
  (2) graph descriptors track recording length (|rho| up to 0.83), and this
      inflated the apparent split-half reliability;
  (3) once length is matched, a single segment's descriptor reliability is low.

Remedy (fixed before looking at any group result):
  * Use all eight PaHaW tasks recorded for each participant
    (1 spiral, 2-4 letters, 5-7 words, 8 sentence), not only the spiral.
  * Cut each task's pen-down signal into non-overlapping segments of exactly
    L on-surface samples (default 500), respecting stroke boundaries; every
    graph is estimated from the same amount of data, so length cannot vary
    and the estimator bias is the same for every segment by construction.
  * Estimate one native-resolution NIG per segment (Gaussian-copula linear TE,
    order p=5, same 10 kinematic channels, same pruning and NGS as
    r2_native_nig.py).
  * Participant descriptor = mean over all of that participant's segments.
    By Spearman-Brown, averaging K segments raises reliability from r to
    K r / (1 + (K-1) r).

Reported:
  0. Segments per task and per participant; participants per task.
  1. Single-segment reliability (adjacent segment pairs within a task) and
     participant-level reliability: odd-numbered vs even-numbered segments,
     each half averaged -> ICC, Spearman r, Spearman-Brown r_SB.
     These halves are length-matched by construction.
  2. Cross-task consistency: spiral-only mean vs mean over tasks 2-8.
  3. Confound checks: correlation of each participant-level descriptor with
     number of segments and with total writing duration.
  4. Group comparison (FDR/Bonferroni, m=7) and classification (nested LOPO,
     10x5-fold, bootstrap CI, 1000-permutation p), all-task mean (primary) and
     spiral-only mean (secondary).

Usage:  python r2_multitask_nig.py [--seglen 500] [--order 5] [--perm 1000]
Outputs: r2_multitask_nig_results_L500_p5.txt, r2_multitask_nig_L500_p5.csv,
         r2_multitask_nig_segments_L500_p5.csv (one row per segment)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import r2_common as rc
from r2_common import NGS_CORE
import ngs_pipeline_v2 as base
from r2_native_nig import participant_series, transform, te_matrix, icc1

TASKS = {1: 'spiral', 2: 'l', 3: 'le', 4: 'les', 5: 'lektorka',
         6: 'porovnat', 7: 'nepopadnout', 8: 'sentence'}


def fixed_segments(S, bounds, L, p):
    """Non-overlapping segments of exactly L concatenated pen-down samples.
    Returns list of (S_seg, bounds_seg); remainder < L is discarded."""
    out = []
    for k in range(len(S) // L):
        a, b = k * L, (k + 1) * L
        sb = [(max(s, a) - a, min(e, b) - a) for s, e in bounds
              if e > a and s < b and min(e, b) - max(s, a) > p]
        if sb and sum(e - s for s, e in sb) > 3 * p + 20:
            out.append((S[a:b], sb))
    return out


def seg_ngs(Sseg, bseg, p, seed):
    M = te_matrix(transform(Sseg, True), bseg, p)[0]
    return rc.ngs_from_matrix(M, seed=seed)


def rel_line(out, m, a, b):
    r = stats.spearmanr(a, b)[0]
    out(f"   {m:<12} ICC={icc1(a, b):>6.3f}  r={r:>6.3f}  r_SB={2*r/(1+r) if r > -1 else np.nan:>6.3f}")


def evaluate(out, tag, X, y, n_perm):
    out(f"\n   Group comparison, {tag} (Mann-Whitney, m=7):")
    for m, pv, q, b in rc.group_comparison(X, y):
        k = NGS_CORE.index(m)
        d = np.median(X[y == 1, k]) - np.median(X[y == 0, k])
        flag = ('FDR*' if q < 0.05 else '') + (' Bonf*' if b < 0.05 else '')
        out(f"     {m:<12} p={pv:.4f}  q_FDR={q:.4f}  p_Bonf={b:.4f}  "
            f"{'PD>HC' if d > 0 else 'PD<HC'}  {flag}")
    auc, yt, ys = rc.nested_lopo_auc(X, y, return_scores=True)
    km, ks = rc.nested_kfold_auc(X, y)
    lo, hi = rc.bootstrap_ci(yt, ys)
    pv, _ = rc.permutation_p(X, y, auc, n_perm)
    out(f"   Classification ({tag}): LOPO AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
        f"10x5-fold={km:.4f}±{ks:.4f}  permutation p={pv:.4f} ({n_perm})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--seglen', type=int, default=500)
    ap.add_argument('--order', type=int, default=5)
    ap.add_argument('--perm', type=int, default=1000)
    ap.add_argument('--min-segments', type=int, default=4)
    args = ap.parse_args()
    L, p = args.seglen, args.order
    sfx = f"_L{L}_p{p}"

    log = open(f'r2_multitask_nig_results{sfx}.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    # Same 72-participant cohort as the manuscript (Task-1 present and usable)
    part_df = base.load_pahaw(args.pahaw, args.corpus)
    cohort = [(r['participant_id'], 1 if r['label'] == 'PD' else 0)
              for _, r in part_df.iterrows()
              if r['svc_data'] is not None and rc.window_features(r['svc_data'])[0] is not None]
    root = Path(args.pahaw)

    t0 = time.time()
    seg_rows = []
    for k, (pid, lab) in enumerate(cohort):
        for task in TASKS:
            f = root / pid / f"{pid}__{task}_1.svc"
            if not f.exists():
                continue
            svc = base.read_svc(f)
            if len(svc) == 0:
                continue
            S, bounds = participant_series(svc, 2 * p + 5)
            if S is None:
                continue
            for si, (Ss, bs) in enumerate(fixed_segments(S, bounds, L, p)):
                v = seg_ngs(Ss, bs, p, seed=k * 1000 + task * 100 + si)
                row = {'participant_id': pid, 'label': lab, 'task': task, 'seg': si}
                row.update(dict(zip(NGS_CORE, v)))
                seg_rows.append(row)
        if (k + 1) % 12 == 0:
            out(f"  {k+1}/{len(cohort)} participants ({(time.time()-t0)/60:.1f} min)")
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(f'r2_multitask_nig_segments{sfx}.csv', index=False)

    out(f"\nMulti-task fixed-segment NIG: L={L} on-surface samples per segment, "
        f"Gaussian-copula TE p={p}")
    out("\n0. Segments")
    for task, name in TASKS.items():
        st = seg[seg.task == task]
        out(f"   task {task} ({name:<11}): {st.participant_id.nunique():>2} participants, "
            f"{len(st):>4} segments")
    per = seg.groupby('participant_id').size()
    out(f"   per participant: min={per.min()}  median={int(per.median())}  max={per.max()}")
    keep = per[per >= args.min_segments].index
    out(f"   participants with >= {args.min_segments} segments: {len(keep)} / {len(cohort)}")
    seg = seg[seg.participant_id.isin(keep)]

    # ── 1. reliability ──────────────────────────────────────────────────────
    out("\n1a. Single-segment reliability (consecutive segment pairs within a task):")
    pairs = []
    for (pid, task), g in seg.groupby(['participant_id', 'task']):
        g = g.sort_values('seg')
        if len(g) >= 2:
            pairs.append((g.iloc[0][NGS_CORE].values.astype(float),
                          g.iloc[1][NGS_CORE].values.astype(float), pid))
    # one pair per participant (first task with >= 2 segments) to keep independence
    seen, A, B = set(), [], []
    for a, b, pid in pairs:
        if pid not in seen:
            seen.add(pid); A.append(a); B.append(b)
    A, B = np.array(A), np.array(B)
    out(f"   (n={len(A)} participants)")
    for k, m in enumerate(NGS_CORE):
        rel_line(out, m, A[:, k], B[:, k])

    seg = seg.sort_values(['participant_id', 'task', 'seg']).copy()
    seg['ordinal'] = seg.groupby('participant_id').cumcount()
    odd = seg[seg.ordinal % 2 == 0].groupby('participant_id')[NGS_CORE].mean()
    even = seg[seg.ordinal % 2 == 1].groupby('participant_id')[NGS_CORE].mean()
    common = odd.index.intersection(even.index)
    out(f"\n1b. Participant-level reliability: mean of odd vs mean of even segments "
        f"(length-matched by construction; n={len(common)})")
    for m in NGS_CORE:
        rel_line(out, m, odd.loc[common, m].values, even.loc[common, m].values)
    out("   (r_SB is the reliability of the full all-segment mean used below)")

    # ── 2. cross-task consistency ───────────────────────────────────────────
    sp = seg[seg.task == 1].groupby('participant_id')[NGS_CORE].mean()
    ot = seg[seg.task != 1].groupby('participant_id')[NGS_CORE].mean()
    ci = sp.index.intersection(ot.index)
    out(f"\n2. Cross-task consistency: spiral-only mean vs tasks 2-8 mean (n={len(ci)})")
    for m in NGS_CORE:
        rel_line(out, m, sp.loc[ci, m].values, ot.loc[ci, m].values)

    # ── 3. confounds ────────────────────────────────────────────────────────
    P = seg.groupby('participant_id').agg(label=('label', 'first'),
                                          n_seg=('seg', 'size'))
    P = P.join(seg.groupby('participant_id')[NGS_CORE].mean())
    out("\n3. Confound check: Spearman rho of participant mean descriptor with number of segments")
    out("   " + "  ".join(f"{m}={stats.spearmanr(P[m], P.n_seg)[0]:+.2f}" for m in NGS_CORE))
    out(f"   segments PD median={int(P[P.label==1].n_seg.median())}  "
        f"HC median={int(P[P.label==0].n_seg.median())}  "
        f"p={stats.mannwhitneyu(P[P.label==1].n_seg, P[P.label==0].n_seg)[1]:.4f}")

    # ── 4. group comparison and classification ─────────────────────────────
    y = P.label.values.astype(int)
    out(f"\n4. Group comparison and classification (n={len(y)}: PD={y.sum()}, HC={(y==0).sum()})")
    evaluate(out, 'all-task mean (primary)', P[NGS_CORE].values, y, args.perm)
    Ps = P.loc[P.index.intersection(sp.index)]
    evaluate(out, 'spiral-only mean (secondary)', sp.loc[Ps.index, NGS_CORE].values,
             Ps.label.values.astype(int), args.perm)

    P.to_csv(f'r2_multitask_nig{sfx}.csv')
    out(f"\nSaved r2_multitask_nig_results{sfx}.txt and CSVs  [total {(time.time()-t0)/60:.1f} min]")
    log.close()


if __name__ == '__main__':
    main()
