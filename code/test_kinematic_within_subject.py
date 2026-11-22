"""
SMALL-SCALE TEST: direct kinematic (not image-based) within-subject features.

Computes features straight from the raw (Y, X, t, button, azimuth, altitude,
pressure) stream per time-window -- no image rendering step at all. This
needs far fewer raw samples per window than legible image rendering did,
so it should support many more windows per subject, giving TE a longer,
finer-grained temporal sequence to work with.

10 features per window, parallel in spirit to Phi-10 but kinematic-native:
  v_mean, v_std       : speed statistics
  a_mean, a_std        : acceleration statistics
  jerk_std             : movement smoothness
  p_mean, p_std        : pen pressure statistics
  surface_ratio        : fraction of samples on-surface (button=1)
  angle_std            : combined azimuth/altitude variability
  tremor_power         : spectral power in the 4-6 Hz band of the speed signal
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from ngs_pipeline_v2 import load_pahaw, CONFIG, build_nig, extract_ngs, NGS_CORE

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

KINEMATIC_MIN_SAMPLES_PER_WINDOW = 15   # much lower bar than image rendering needed


def extract_kinematic_features(window):
    """window: np.ndarray (n, 7) = [Y, X, t, button, azimuth, altitude, pressure]
    NOTE: t is in MILLISECONDS in PaHaW's SVC format (confirmed empirically:
    median dt ~ 7-8 units). All time-based calculations below use dt_s (seconds)."""
    if len(window) < 4:
        return None

    Y, X, t, button, azimuth, altitude, pressure = window.T
    dt_ms = np.diff(t)
    dt_ms[dt_ms <= 0] = np.median(dt_ms[dt_ms > 0]) if np.any(dt_ms > 0) else 8.0
    dt_s = dt_ms / 1000.0   # <-- the fix: convert ms -> s before any rate calc

    dx, dy = np.diff(X), np.diff(Y)
    speed = np.sqrt(dx**2 + dy**2) / dt_s

    if len(speed) < 3:
        return None

    accel = np.diff(speed) / dt_s[1:]

    v_mean, v_std = np.mean(speed), np.std(speed)
    a_mean, a_std = np.mean(accel), np.std(accel)
    p_mean, p_std = np.mean(pressure), np.std(pressure)
    angle_std = np.std(azimuth) + np.std(altitude)

    # Tremor-band (4-6 Hz) power from the speed signal's periodogram.
    # fs now correctly in Hz since dt_s is in seconds.
    fs = 1.0 / np.median(dt_s) if np.median(dt_s) > 0 else 125.0
    if len(speed) >= 8:
        freqs = np.fft.rfftfreq(len(speed), d=1.0/fs)
        power = np.abs(np.fft.rfft(speed - speed.mean()))**2
        band = (freqs >= 4) & (freqs <= 6)
        tremor_power = power[band].sum() if band.any() else 0.0
    else:
        tremor_power = 0.0

    # Direction-change (curvature) -- captures spiral geometry, not speed dynamics.
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi  # wrap to [-pi, pi]
    curvature_mean = np.mean(np.abs(dtheta)) if len(dtheta) > 0 else 0.0

    # Path straightness -- displacement / path length within this window.
    # Low value = tight looping/tremor-like path; high value = smooth stroke.
    path_length = np.sum(np.sqrt(dx**2 + dy**2))
    displacement = np.sqrt((X[-1]-X[0])**2 + (Y[-1]-Y[0])**2)
    path_straightness = displacement / (path_length + 1e-6)

    return {
        'v_mean': v_mean, 'v_std': v_std,
        'a_mean': a_mean, 'a_std': a_std,
        'p_mean': p_mean, 'p_std': p_std,
        'angle_std': angle_std,
        'tremor_power': tremor_power,
        'curvature_mean': curvature_mean,
        'path_straightness': path_straightness,
    }


KINEMATIC_KEYS = ['v_mean', 'v_std', 'a_mean', 'a_std',
                  'p_mean', 'p_std', 'angle_std', 'tremor_power',
                  'curvature_mean', 'path_straightness']


def build_kinematic_within_subject_ngs(svc_data, n_windows):
    if svc_data is None or len(svc_data) < n_windows * KINEMATIC_MIN_SAMPLES_PER_WINDOW:
        return None, None
    n = len(svc_data)
    win_size = n // n_windows
    feat_rows = []
    for wi in range(n_windows):
        start = wi * win_size
        end = start + win_size if wi < n_windows - 1 else n
        window = svc_data[start:end]
        feats = extract_kinematic_features(window)
        if feats is None:
            return None, None
        feat_rows.append([feats[k] for k in KINEMATIC_KEYS])

    X_windows = np.array(feat_rows)
    if not np.all(np.isfinite(X_windows)):
        return None, X_windows  # flag for inspection

    X_z = StandardScaler().fit_transform(X_windows)
    G = build_nig(X_z, label='kinematic_within_subject')
    ngs = extract_ngs(G)
    return ngs, X_windows


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    pd_rows = part_df[part_df['label'] == 'PD']
    hc_rows = part_df[part_df['label'] == 'HC']

    test_subjects = list(pd_rows.head(3).itertuples()) + list(hc_rows.head(3).itertuples())

    print("── Testing kinematic feature extraction at several N_WINDOWS ──\n")
    for n_windows in [10, 15, 20, 25, 30]:
        n_ok, n_excluded, n_nonfinite = 0, 0, 0
        for row in test_subjects:
            ngs, X_windows = build_kinematic_within_subject_ngs(row.svc_data, n_windows)
            if ngs is None and X_windows is None:
                n_excluded += 1
            elif ngs is None:
                n_nonfinite += 1
            else:
                n_ok += 1
        print(f"N_WINDOWS={n_windows:<4} ok={n_ok}  excluded(too short)={n_excluded}  "
              f"non-finite={n_nonfinite}   (of {len(test_subjects)})")

    print("\n── Full NGS output at N_WINDOWS=20 (te_bins default) ──\n")
    for row in test_subjects:
        ngs, X_windows = build_kinematic_within_subject_ngs(row.svc_data, 20)
        pid, label = row.participant_id, row.label
        if ngs is None:
            print(f"{pid} ({label}): SKIPPED")
            continue
        print(f"{pid} ({label}):")
        for k in NGS_CORE:
            print(f"    {k:<14} = {ngs.get(k, float('nan')):.4f}")
        is_degenerate = all(abs(ngs.get(k, 0.0)) < 1e-9 for k in NGS_CORE)
        if is_degenerate:
            print("    <-- DEGENERATE (all zero)")
        print()
