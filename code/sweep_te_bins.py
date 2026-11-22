import numpy as np
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import (
    load_pahaw, CONFIG, render_svc_quartile, extract_phi10,
    build_nig, extract_ngs, NGS_CORE
)

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

N_WINDOWS = 10
MIN_SAMPLES_PER_WINDOW = 50


def build_within_subject_ngs(svc_data, n_windows, size=(512, 512)):
    if svc_data is None or len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None
    feat_rows = []
    for wi in range(n_windows):
        img = render_svc_quartile(svc_data, wi, n_q=n_windows, size=size)
        feats = extract_phi10(img)
        feat_rows.append([feats[k] for k in
                           ['Eb', 'Eg', 'Em', 'sigma_d', 'Ag', 'rho_s', 'Hs', 'Hg', 'SIA', 'Omega']])
    X_windows = np.array(feat_rows)
    X_z = StandardScaler().fit_transform(X_windows)
    G = build_nig(X_z, label='within_subject')
    return extract_ngs(G)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    pd_rows = part_df[part_df['label'] == 'PD']
    hc_rows = part_df[part_df['label'] == 'HC']

    test_subjects = []
    for _, row in pd_rows.iterrows():
        if row.get('svc_data') is not None and len(row['svc_data']) >= N_WINDOWS * MIN_SAMPLES_PER_WINDOW:
            test_subjects.append(row)
        if len(test_subjects) >= 3:
            break
    n_pd = len(test_subjects)
    for _, row in hc_rows.iterrows():
        if row.get('svc_data') is not None and len(row['svc_data']) >= N_WINDOWS * MIN_SAMPLES_PER_WINDOW:
            test_subjects.append(row)
        if len(test_subjects) >= n_pd + 3:
            break

    print(f"Sweeping te_bins on {len(test_subjects)} subjects, N_WINDOWS={N_WINDOWS}\n")

    for te_bins in [3, 4, 5, 6, 8]:
        CONFIG['te_bins'] = te_bins
        n_degenerate = 0
        rows_summary = []
        for row in test_subjects:
            ngs = build_within_subject_ngs(row['svc_data'], N_WINDOWS)
            if ngs is None:
                continue
            vals = [ngs.get(k, 0.0) for k in NGS_CORE]
            is_degenerate = all(abs(v) < 1e-9 for v in vals)
            if is_degenerate:
                n_degenerate += 1
            rows_summary.append((row['participant_id'], row['label'], is_degenerate))

        print(f"te_bins={te_bins}: {n_degenerate}/{len(test_subjects)} subjects degenerate (all-zero NGS)")
        for pid, label, deg in rows_summary:
            flag = "  <-- DEGENERATE" if deg else ""
            print(f"    {pid} ({label}){flag}")
        print()
