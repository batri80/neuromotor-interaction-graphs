import numpy as np
import pandas as pd
from scipy import stats

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

print("── Checking available columns for H&Y stage ──")
print(list(part_df.columns))
print()

# Try common column name variants
hy_col = None
for candidate in ['H&Y', 'HY', 'hy_stage', 'HY_stage', 'hoehn_yahr', 'updrs_v', 'H_Y']:
    if candidate in part_df.columns:
        hy_col = candidate
        break

if hy_col is None:
    print("Could not find H&Y column automatically. Available columns printed above --")
    print("please identify the correct column name and I'll adjust the script.")
else:
    print(f"Using H&Y column: '{hy_col}'\n")

    pd_df = part_df[(part_df['label'] == 'PD') & part_df[hy_col].notna()].copy()
    print(f"PD subjects with H&Y stage available: {len(pd_df)}\n")

    ngs_ws_df = pd.read_pickle('within_subject_kinematic_ngs_df.pkl')

    merged = pd_df[['participant_id', hy_col]].merge(
        ngs_ws_df, on='participant_id', how='inner')
    print(f"Merged (within-subject NGS + H&Y): n={len(merged)}\n")

    print("── Within-subject kinematic NGS vs H&Y stage (bypasses rendering entirely) ──")
    print(f"{'Metric':<14}{'Spearman rho':<15}{'p-value':<12}{'Bonferroni p'}")
    results = []
    for k in NGS_CORE:
        rho, p = stats.spearmanr(merged[hy_col], merged[k])
        results.append((k, rho, p))
        print(f"{k:<14}{rho:<15.4f}{p:<12.4f}{min(p*7, 1.0):.4f}")

    print(f"\nFor reference, ORIGINAL rendered-image pipeline (population-pooled,")
    print(f"Instability Score): rho=-0.056, p=0.748")
    print(f"ORIGINAL rendered-image pipeline (best NGS metric, SmallWorld): "
          f"rho=+0.306, p=0.070, Bonferroni p=0.487")

    best = max(results, key=lambda r: abs(r[1]))
    print(f"\nBest within-subject kinematic correlation: {best[0]}, rho={best[1]:.4f}, p={best[2]:.4f}")
    print("\nIf this is meaningfully stronger than the rendered-pipeline result,")
    print("that's direct evidence FOR the attenuation-by-rendering hypothesis.")
    print("If similar or weaker, that's evidence AGAINST it -- severity signal")
    print("may simply not be present regardless of rendering, and R3.8's premise")
    print("would need to be revised rather than explained.")
