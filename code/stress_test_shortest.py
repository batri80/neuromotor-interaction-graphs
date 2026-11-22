import numpy as np
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import (
    load_pahaw, CONFIG, render_svc_quartile, extract_phi10,
    build_nig, extract_ngs, NGS_CORE
)

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 6          # chosen from the sweep

N_WINDOWS = 10
MIN_SAMPLES_PER_WINDOW = 50


def build_within_subject_ngs(svc_data, n_windows, size=(512, 512)):
    if svc_data is None or len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None, None
    feat_rows = []
    for wi in range(n_windows):
        img = render_svc_quartile(svc_data, wi, n_q=n_windows, size=size)
        feats = extract_phi10(img)
        feat_rows.append([feats[k] for k in
                           ['Eb', 'Eg', 'Em', 'sigma_d', 'Ag', 'rho_s', 'Hs', 'Hg', 'SIA', 'Omega']])
    X_windows = np.array(feat_rows)
    X_z = StandardScaler().fit_transform(X_windows)
    G = build_nig(X_z, label='within_subject')
    return extract_ngs(G), X_windows


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    part_df['svc_len'] = part_df['svc_data'].apply(lambda s: len(s) if s is not None else 0)

    # Worst-case stress test: the 8 shortest recordings that still clear the
    # N_WINDOWS=10 floor (i.e. the subjects with the LEAST data to fill 6 bins with)
    eligible = part_df[part_df['svc_len'] >= N_WINDOWS * MIN_SAMPLES_PER_WINDOW]
    shortest = eligible.nsmallest(8, 'svc_len')

    print(f"Testing te_bins={CONFIG['te_bins']}, N_WINDOWS={N_WINDOWS} on the "
          f"8 SHORTEST eligible recordings (worst case for TE sparsity):\n")

    n_degenerate = 0
    for _, row in shortest.iterrows():
        pid, label, svc_len = row['participant_id'], row['label'], row['svc_len']
        ngs, X_windows = build_within_subject_ngs(row['svc_data'], N_WINDOWS)
        vals = [ngs.get(k, 0.0) for k in NGS_CORE]
        is_degenerate = all(abs(v) < 1e-9 for v in vals)
        if is_degenerate:
            n_degenerate += 1
        flag = "  <-- DEGENERATE" if is_degenerate else ""
        print(f"{pid} ({label}, {svc_len} samples = {svc_len/150:.1f}s){flag}")
        for k in NGS_CORE:
            print(f"    {k:<14} = {ngs.get(k, float('nan')):.4f}")
        print()

    print(f"Degenerate: {n_degenerate}/{len(shortest)} (worst-case subjects)")
