import numpy as np
from sklearn.linear_model import LogisticRegression

from ngs_p4_newhandpd import load_newhandpd, CONFIG, build_nig, extract_ngs, NGS_CORE
from repeated_kfold_foldsafe import repeated_kfold_foldsafe

# ADJUST THESE to wherever the NewHandPD folders actually live -- run
#   find . -iname "*HealthySpiral*" -o -iname "*PatientSpiral*"
# first if unsure.
CONFIG['healthy_root'] = './healthy'
CONFIG['patient_root'] = './patient'

img_df, part_df = load_newhandpd(CONFIG['healthy_root'], CONFIG['patient_root'])
feat_cols = CONFIG['feature_names']

print(f"Loaded {len(part_df)} subjects: "
      f"{(part_df['label']=='PD').sum()} PD, {(part_df['label']=='HC').sum()} HC\n")

print("── Fast fold-safe repeated 5-fold CV (the honest stress test) ──")
results = repeated_kfold_foldsafe(
    part_df, feat_cols, NGS_CORE, build_nig, extract_ngs,
    clf_factory=lambda: LogisticRegression(max_iter=1000, random_state=42),
    n_splits=5, n_repeats=10
)
print(f"\nMean AUC={results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
print(f"Pooled AUC={results['pooled_auc']:.4f}")
print(f"\nFor reference, the manuscript's claimed NewHandPD result: "
      f"NGS solo AUC=0.909, Hybrid AUC=0.985 (both from the LEAKED pipeline)")
