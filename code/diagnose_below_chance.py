import pandas as pd
import numpy as np
from scipy import stats

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
part_df['svc_len'] = part_df['svc_data'].apply(lambda s: len(s) if s is not None else 0)

ngs_ws_df = pd.read_pickle('within_subject_ngs_df.pkl')

merged = ngs_ws_df.merge(part_df[['participant_id', 'svc_len']], on='participant_id')

print("── 1. Does recording length differ by group? ──")
pd_len = merged[merged['label'] == 'PD']['svc_len']
hc_len = merged[merged['label'] == 'HC']['svc_len']
print(f"PD: mean={pd_len.mean():.0f}  median={pd_len.median():.0f}  std={pd_len.std():.0f}")
print(f"HC: mean={hc_len.mean():.0f}  median={hc_len.median():.0f}  std={hc_len.std():.0f}")
u, p = stats.mannwhitneyu(pd_len, hc_len)
print(f"Mann-Whitney U test: p={p:.4f}  {'*** CONFOUND LIKELY ***' if p < 0.05 else '(not significant)'}\n")

print("── 2. Do NGS features correlate with recording length? ──")
for k in NGS_CORE:
    r, p = stats.spearmanr(merged['svc_len'], merged[k])
    flag = "  <-- STRONG LENGTH CORRELATION" if abs(r) > 0.3 and p < 0.05 else ""
    print(f"{k:<14} spearman r={r:+.3f}  p={p:.4f}{flag}")

print("\n── 3. Sanity check: label encoding / predict_proba column order ──")
print(f"label_int values for PD rows: {merged[merged['label']=='PD']['label_int'].unique()}")
print(f"label_int values for HC rows: {merged[merged['label']=='HC']['label_int'].unique()}")
print("(Expect PD=1, HC=0 consistently -- if reversed anywhere, that alone "
      "would explain an inverted AUC.)\n")

print("── 4. Quick manual single-feature check ──")
print("If a feature has a real group difference, its own AUC (used alone, "
      "no CV) should NOT be ~0.5. If it IS informative but our CV pipeline "
      "gives inverted-looking results, that's a bug, not absence of signal.")
from sklearn.metrics import roc_auc_score
y = merged['label_int'].values
for k in NGS_CORE:
    x = merged[k].values
    try:
        auc_raw = roc_auc_score(y, x)
    except Exception:
        auc_raw = float('nan')
    print(f"{k:<14} raw single-feature AUC (feature value directly as score) = {auc_raw:.3f}")
