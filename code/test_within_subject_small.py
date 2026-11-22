"""
SMALL-SCALE TEST: within-subject temporal NIG construction.

Instead of pooling across subjects (the leaky/circular design), this builds
each subject's NIG entirely from their OWN 150Hz spiral recording, split
into N_WINDOWS time windows in real chronological order. TE is estimated
across windows (real time axis, real lag=1), not across subjects (fake
"time axis" via arbitrary participant order).

No other subject's data ever touches this computation -> leakage is
impossible by construction, not just avoided by careful fold management.

Run this FIRST on a handful of subjects (2-3 PD, 2-3 HC) to sanity-check
before committing to a full n=72 rebuild.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import (
    load_pahaw, CONFIG, render_svc_quartile, extract_phi10,
    build_nig, extract_ngs, NGS_CORE
)

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

N_WINDOWS = 10          # start modest; raise later if this works and data supports it
MIN_SAMPLES_PER_WINDOW = 50   # same guard the existing quartile code uses


def build_within_subject_ngs(svc_data, n_windows=N_WINDOWS, size=(512, 512)):
    """
    Returns (ngs_dict, window_feat_matrix, n_windows_used) for ONE subject,
    built entirely from their own recording. window_feat_matrix has shape
    (n_windows, 10) -- one Phi-10 vector per chronological window.
    """
    if svc_data is None or len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None, None, 0

    feat_rows = []
    for wi in range(n_windows):
        img = render_svc_quartile(svc_data, wi, n_q=n_windows, size=size)
        feats = extract_phi10(img)
        feat_rows.append([feats[k] for k in
                           ['Eb', 'Eg', 'Em', 'sigma_d', 'Ag', 'rho_s', 'Hs', 'Hg', 'SIA', 'Omega']])

    X_windows = np.array(feat_rows)  # (n_windows, 10)

    # Standardize ACROSS WINDOWS (this subject's own within-recording variation),
    # not across subjects -- this is the key structural change.
    X_z = StandardScaler().fit_transform(X_windows)

    G = build_nig(X_z, label='within_subject')
    ngs = extract_ngs(G)
    return ngs, X_windows, n_windows


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    # Pick a small, mixed test set: first 3 PD and first 3 HC subjects with
    # enough samples for N_WINDOWS windows.
    pd_rows = part_df[part_df['label'] == 'PD']
    hc_rows = part_df[part_df['label'] == 'HC']

    test_subjects = []
    for _, row in pd_rows.iterrows():
        if row.get('svc_data') is not None and len(row['svc_data']) >= N_WINDOWS * MIN_SAMPLES_PER_WINDOW:
            test_subjects.append(row)
        if len(test_subjects) >= 3:
            break
    n_pd_found = len(test_subjects)
    for _, row in hc_rows.iterrows():
        if row.get('svc_data') is not None and len(row['svc_data']) >= N_WINDOWS * MIN_SAMPLES_PER_WINDOW:
            test_subjects.append(row)
        if len(test_subjects) >= n_pd_found + 3:
            break

    print(f"Testing on {len(test_subjects)} subjects "
          f"({n_pd_found} PD, {len(test_subjects)-n_pd_found} HC), N_WINDOWS={N_WINDOWS}\n")

    results = []
    for row in test_subjects:
        pid = row['participant_id']
        label = row['label']
        ngs, X_windows, n_used = build_within_subject_ngs(row['svc_data'])
        if ngs is None:
            print(f"{pid} ({label}): SKIPPED — not enough samples")
            continue
        results.append((pid, label, ngs, X_windows))
        print(f"{pid} ({label}):")
        for k in NGS_CORE:
            print(f"    {k:<14} = {ngs.get(k, float('nan')):.4f}")
        # quick degeneracy check: are the window feature rows actually
        # different from each other, or did rendering collapse to near-identical
        # sub-images (which would make TE meaningless)?
        row_std = X_windows.std(axis=0)
        print(f"    window-to-window feature std (should be >0, not tiny): "
              f"{np.round(row_std, 3)}")
        print()

    print(f"\nCompleted {len(results)}/{len(test_subjects)} subjects successfully.")
    print("Look for: (1) no NaN/zero NGS values, (2) window std clearly >0 "
          "(windows are actually different from each other, not degenerate), "
          "(3) any qualitative PD vs HC separation worth a closer look (not "
          "a formal test yet, just an eyeball check at this scale).")
