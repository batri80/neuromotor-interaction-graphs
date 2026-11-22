"""
R4.2 — Recompute Table 4 from the participants actually analysed and
document every exclusion.

PaHaW: lists corpus counts, which participants lack a Task-1 (spiral) file,
which are excluded by the within-subject windowing rule (W=20 windows x 15
samples = 300 samples minimum, or failed/non-finite features), and then
prints age, sex, H&Y and LED for the analysed PD and HC groups.

NewHandPD (optional, --newhandpd): counts distinct participant IDs per class
from the image filenames using the same rule as the manuscript code
(text before the first underscore of the file stem) and lists every ID with
its image count, so the PD count (31 in the public release vs 32 in Table 4)
can be checked.

Usage:
  python r2_audit_cohort.py --pahaw ./PaHaW/PaHaW_public \
      --corpus ./PaHaW/PaHaW_files/corpus_PaHaW.xlsx \
      [--newhandpd /path/containing/PatientSpiral-resized_and_HealthySpiral-resized]
Output: r2_audit_cohort_results.txt
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ngs_pipeline_v2 import read_svc
import r2_common as rc


def find_col(df, *cands):
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--nh-patient', default=None,
                    help='NewHandPD patient folder (searched recursively)')
    ap.add_argument('--nh-healthy', default=None,
                    help='NewHandPD healthy folder (searched recursively)')
    args = ap.parse_args()

    log = open('r2_audit_cohort_results.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    corpus = pd.read_excel(args.corpus)
    out(f"Corpus columns: {list(corpus.columns)}")
    corpus['ID_str'] = corpus['ID'].apply(lambda x: f"{int(x):05d}")
    corpus['grp'] = corpus['Disease'].map({'PD': 'PD', 'H': 'HC'})
    out(f"Corpus: PD={int((corpus.grp=='PD').sum())}  HC={int((corpus.grp=='HC').sum())}")

    root = Path(args.pahaw)
    status = []
    for _, r in corpus.iterrows():
        f = root / r['ID_str'] / f"{r['ID_str']}__1_1.svc"
        if not f.exists():
            status.append('no Task-1 recording'); continue
        Xw, why = rc.window_features(read_svc(f))
        status.append(why)
    corpus['status'] = status

    out("\nExclusions:")
    excl = corpus[corpus.status != 'ok']
    if len(excl) == 0:
        out("  none")
    for _, r in excl.iterrows():
        out(f"  {r['ID_str']} ({r['grp']}): {r['status']}")

    an = corpus[corpus.status == 'ok']
    out(f"\nAnalysed: PD={int((an.grp=='PD').sum())}  HC={int((an.grp=='HC').sum())}")

    age, sex = find_col(an, 'Age'), find_col(an, 'Sex', 'Gender')
    hy, led = find_col(an, 'UPDRS V', 'H&Y', 'HY'), find_col(an, 'LED')
    for g in ['PD', 'HC']:
        d = an[an.grp == g]
        out(f"\n{g} (n={len(d)})")
        if age:
            out(f"  Age mean±SD: {d[age].mean():.1f} ± {d[age].std(ddof=1):.1f}")
        if sex:
            out(f"  Sex counts: {dict(Counter(d[sex].astype(str)))}")
        else:
            out("  Sex: no sex column found in corpus — please check manually")
        if g == 'PD' and hy:
            out(f"  H&Y distribution: {dict(sorted(Counter(d[hy]).items()))}  "
                f"(sum={int(d[hy].notna().sum())})")
        if g == 'PD' and led:
            out(f"  LED mean±SD: {d[led].mean():.0f} ± {d[led].std(ddof=1):.0f}")

    if args.nh_patient or args.nh_healthy:
        out("\n" + "=" * 70 + "\nNewHandPD participant IDs (rule: stem.split('_')[0])")
        exts = {'.jpg', '.jpeg', '.png'}
        for cls, root_dir in [('PD', args.nh_patient), ('HC', args.nh_healthy)]:
            if not root_dir:
                continue
            root_p = Path(root_dir)
            imgs = sorted(p for p in root_p.rglob('*') if p.suffix.lower() in exts)
            out(f"\n{cls}: {len(imgs)} images under {root_p}")
            by_folder = defaultdict(list)
            for p in imgs:
                by_folder[str(p.parent.relative_to(root_p))].append(p)
            ids_all = defaultdict(int)
            for fd, files in sorted(by_folder.items()):
                ids = Counter(f.stem.split('_')[0] for f in files)
                out(f"  folder '{fd}': {len(files)} images, {len(ids)} distinct IDs; "
                    f"sample names: {[f.name for f in files[:4]]}")
                for k, v in ids.items():
                    ids_all[k] += v
            out(f"  {cls} total: {len(ids_all)} distinct IDs across all folders")
            out("  " + ", ".join(f"{k}({v})" for k, v in sorted(ids_all.items())))
            counts = list(ids_all.values())
            if counts:
                mode = max(set(counts), key=counts.count)
                odd = {k: v for k, v in ids_all.items() if v != mode}
                if odd:
                    out(f"  IDs whose image count differs from the usual {mode} "
                        f"(possible split/duplicate IDs): {odd}")
    log.close()


if __name__ == '__main__':
    main()
