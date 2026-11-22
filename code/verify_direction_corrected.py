import pandas as pd
import numpy as np
from scipy import stats

from ngs_pipeline_v2 import NGS_CORE

ngs_df = pd.read_pickle('within_subject_kinematic_ngs_df.pkl')

pd_df = ngs_df[ngs_df['label'] == 'PD']
hc_df = ngs_df[ngs_df['label'] == 'HC']

print(f"n_PD={len(pd_df)}, n_HC={len(hc_df)}\n")
print(f"{'Metric':<14}{'PD mean±SD':<20}{'HC mean±SD':<20}{'Cohen d':<10}{'direction':<14}{'p (MWU)'}")

for k in NGS_CORE:
    pd_vals = pd_df[k].values
    hc_vals = hc_df[k].values
    pd_mean, pd_std = pd_vals.mean(), pd_vals.std()
    hc_mean, hc_std = hc_vals.mean(), hc_vals.std()

    # pooled SD for Cohen's d
    n1, n2 = len(pd_vals), len(hc_vals)
    pooled_std = np.sqrt(((n1-1)*pd_vals.std(ddof=1)**2 + (n2-1)*hc_vals.std(ddof=1)**2) / (n1+n2-2))
    d = (pd_mean - hc_mean) / pooled_std if pooled_std > 0 else 0.0
    direction = "higher in PD" if d > 0 else "lower in PD"

    _, p = stats.mannwhitneyu(pd_vals, hc_vals)

    print(f"{k:<14}{pd_mean:>8.4f}±{pd_std:<9.4f}{hc_mean:>8.4f}±{hc_std:<9.4f}"
          f"{d:>8.3f}  {direction:<14}{p:.4f}")
