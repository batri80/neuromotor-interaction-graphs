"""
Inspect per-subject prediction pattern from the completed fold-safe LOPO run.
Can be run while the K-fold run is still going --
it only reads the already-saved pickle, no graph computation needed, so it's instant.
"""

import pickle
import numpy as np
from ngs_pipeline_v2 import load_pahaw, CONFIG

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

with open('foldsafe_lopo_results.pkl', 'rb') as f:
    R = pickle.load(f)

y_true = R['y_true']
y_pred = R['y_pred']
y_prob = R['y_prob']

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

print(f"Overall: AUC={R['auc']:.4f}  Acc={R['acc']:.4f}  Sens={R['sens']:.4f}  Spec={R['spec']:.4f}\n")

correct = (y_true == y_pred)
n_correct = correct.sum()
n_total = len(correct)
print(f"Correct: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)\n")

# Misclassified subjects, with their predicted probability (how confidently wrong)
print("── Misclassified subjects ──")
print(f"{'idx':<5}{'participant_id':<16}{'true':<6}{'pred':<6}{'prob(PD)':<10}{'confidence'}")
misclassified_idx = np.where(~correct)[0]
for i in misclassified_idx:
    pid = part_df.iloc[i]['participant_id']
    true_lbl = 'PD' if y_true[i] == 1 else 'HC'
    pred_lbl = 'PD' if y_pred[i] == 1 else 'HC'
    conf = abs(y_prob[i] - 0.5) * 2  # 0 = coin-flip, 1 = maximally confident
    conf_tag = "confidently wrong" if conf > 0.5 else "borderline"
    print(f"{i:<5}{pid:<16}{true_lbl:<6}{pred_lbl:<6}{y_prob[i]:<10.3f}{conf_tag}")

print(f"\nTotal misclassified: {len(misclassified_idx)}")
n_confident_wrong = sum(1 for i in misclassified_idx if abs(y_prob[i]-0.5)*2 > 0.5)
print(f"Confidently wrong (>0.75 or <0.25 prob): {n_confident_wrong}")
print(f"Borderline (close to 0.5): {len(misclassified_idx) - n_confident_wrong}")

# Split misclassification rate by true class
pd_idx = np.where(y_true == 1)[0]
hc_idx = np.where(y_true == 0)[0]
pd_errors = (~correct[pd_idx]).sum()
hc_errors = (~correct[hc_idx]).sum()
print(f"\nPD subjects misclassified: {pd_errors}/{len(pd_idx)}")
print(f"HC subjects misclassified: {hc_errors}/{len(hc_idx)}")

# Distribution of predicted probabilities for correct vs incorrect -- broad-based
# good performance should show correct predictions clustered near 0/1, not just
# barely over the 0.5 threshold.
print("\n── Probability distribution ──")
print(f"Correct predictions  -- prob range: [{y_prob[correct].min():.3f}, {y_prob[correct].max():.3f}], "
      f"mean dist from 0.5: {np.mean(np.abs(y_prob[correct]-0.5)):.3f}")
if (~correct).sum() > 0:
    print(f"Incorrect predictions -- prob range: [{y_prob[~correct].min():.3f}, {y_prob[~correct].max():.3f}], "
          f"mean dist from 0.5: {np.mean(np.abs(y_prob[~correct]-0.5)):.3f}")
