import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Reuses nested_selection_fold_outputs.pkl written by permutation_and_bootstrap.py
# (participant_id, true, prob, selected) from the permutation-test run.
obs_df = pd.read_pickle('nested_selection_fold_outputs.pkl')

y_true = obs_df['true'].values
y_prob = obs_df['prob'].values

print("="*70)
print("CALIBRATION ANALYSIS")
print("="*70)

brier = brier_score_loss(y_true, y_prob)
print(f"Brier score: {brier:.4f}  (0=perfect, 0.25=uninformative-constant-0.5 baseline)")

# Reliability diagram data (5 bins given small n)
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy='quantile')
print("\nReliability table (quantile-binned, 5 bins):")
print(f"{'Bin mean predicted prob':<28}{'Observed fraction positive'}")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"{mp:<28.4f}{fp:.4f}")

print("\n" + "="*70)
print("FAILURE-CASE ANALYSIS")
print("="*70)

y_pred = (y_prob >= 0.5).astype(int)
correct = (y_true == y_pred)
n_correct = correct.sum()
n_total = len(correct)
print(f"Correct: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)\n")

misclassified_idx = np.where(~correct)[0]
print(f"{'participant_id':<16}{'true':<6}{'pred':<6}{'prob(PD)':<10}{'confidence'}")
for i in misclassified_idx:
    pid = obs_df.iloc[i]['participant_id']
    true_lbl = 'PD' if y_true[i] == 1 else 'HC'
    pred_lbl = 'PD' if y_pred[i] == 1 else 'HC'
    conf = abs(y_prob[i] - 0.5) * 2
    conf_tag = "confidently wrong" if conf > 0.5 else "borderline"
    print(f"{pid:<16}{true_lbl:<6}{pred_lbl:<6}{y_prob[i]:<10.3f}{conf_tag}")

n_confident_wrong = sum(1 for i in misclassified_idx if abs(y_prob[i]-0.5)*2 > 0.5)
pd_idx = np.where(y_true == 1)[0]
hc_idx = np.where(y_true == 0)[0]
pd_errors = (~correct[pd_idx]).sum()
hc_errors = (~correct[hc_idx]).sum()

print(f"\nTotal misclassified: {len(misclassified_idx)}/{n_total}")
print(f"Confidently wrong: {n_confident_wrong}  Borderline: {len(misclassified_idx)-n_confident_wrong}")
print(f"PD misclassified: {pd_errors}/{len(pd_idx)}   HC misclassified: {hc_errors}/{len(hc_idx)}")

print("\nFor the manuscript's Section 5 (Supplementary or main text robustness")
print("subsection), report: n misclassified, PD/HC split, confident/borderline")
print("split, and Brier score. This closes R1.3's failure-case-analysis gap.")
