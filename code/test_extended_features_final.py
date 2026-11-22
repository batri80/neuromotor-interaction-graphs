"""
LAST EXPERIMENT OF THIS SESSION.

Extends our own Phi10_kin (10-dim) to a 12-dim feature vector by adding
two additional kinematic concepts that are:
  (a) well-established in the general handwriting-kinematics literature,
  (b) verifiably present in Drotar et al. (2016)'s own feature table
      (which we have primary-source access to -- see Table 2 of that
      paper: "number of changes in velocity direction" and "jerk"),
  (c) conceptually distinct from anything already in our Phi10_kin.

This is explicitly OUR OWN feature extension, grounded in verifiable,
general literature -- NOT an attempt to reconstruct Kashid & Deshpande's
specific TPS/AWCI/PIC/SIR/MMV formulas, which are not disclosed in their
abstract and which we do not have access to. No claim is made that this
tests "their" features.

New features:
  ncv_rel   : number of local direction-changes in the speed signal,
              relative to window duration (motor-irregularity measure,
              distinct from our existing angle-based curvature feature)
  jerk_mean : mean absolute jerk (rate of change of acceleration) --
              one kinematic derivative order beyond our existing
              acceleration features, standard in the literature
              (Drotar et al. 2016, Table 2)

Everything else (windowing, TE construction, thresholding, nested
selection, LOPO-CV protocol) is IDENTICAL to the primary 10-feature
pipeline, so this isolates the effect of the two added features alone.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from ngs_pipeline_v2 import load_pahaw, CONFIG, NGS_CORE, build_nig, extract_ngs

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG['te_bins'] = 8

N_WINDOWS = 20
MIN_SAMPLES_PER_WINDOW = 15


def extract_extended_kinematic_features(window):
    """Same 10 base features as the primary pipeline, plus ncv_rel and
    jerk_mean. Kept numerically consistent with extract_kinematic_features
    (same dt handling, same conventions)."""
    if len(window) < 5:
        return None

    Y, X, t, button, azimuth, altitude, pressure = window.T
    dt_ms = np.diff(t)
    dt_ms[dt_ms <= 0] = np.median(dt_ms[dt_ms > 0]) if np.any(dt_ms > 0) else 8.0
    dt_s = dt_ms / 1000.0

    dx, dy = np.diff(X), np.diff(Y)
    speed = np.sqrt(dx**2 + dy**2) / dt_s
    if len(speed) < 4:
        return None

    accel = np.diff(speed) / dt_s[1:]
    if len(accel) < 3:
        return None

    v_mean, v_std = np.mean(speed), np.std(speed)
    a_mean, a_std = np.mean(accel), np.std(accel)
    p_mean, p_std = np.mean(pressure), np.std(pressure)
    angle_std = np.std(azimuth) + np.std(altitude)

    fs = 1.0 / np.median(dt_s) if np.median(dt_s) > 0 else 125.0
    if len(speed) >= 8:
        freqs = np.fft.rfftfreq(len(speed), d=1.0/fs)
        power = np.abs(np.fft.rfft(speed - speed.mean()))**2
        band = (freqs >= 4) & (freqs <= 6)
        tremor_power = power[band].sum() if band.any() else 0.0
    else:
        tremor_power = 0.0

    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
    curvature_mean = np.mean(np.abs(dtheta)) if len(dtheta) > 0 else 0.0

    path_length = np.sum(np.sqrt(dx**2 + dy**2))
    displacement = np.sqrt((X[-1]-X[0])**2 + (Y[-1]-Y[0])**2)
    path_straightness = displacement / (path_length + 1e-6)

    # --- New feature 1: NCV (number of changes in velocity direction),
    #     relative to window duration. Counts local extrema (sign changes
    #     in the derivative) of the speed signal. Drotar et al. 2016,
    #     Table 2: "number of changes in velocity direction (NCV)".
    dspeed = np.diff(speed)
    sign_changes = np.sum(np.diff(np.sign(dspeed)) != 0)
    window_duration_s = np.sum(dt_s)
    ncv_rel = sign_changes / window_duration_s if window_duration_s > 0 else 0.0

    # --- New feature 2: mean absolute jerk. Drotar et al. 2016, Table 2:
    #     "jerk: rate at which the acceleration of a pen changes with time".
    jerk = np.diff(accel) / dt_s[2:]
    jerk_mean = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0.0

    return {
        'v_mean': v_mean, 'v_std': v_std,
        'a_mean': a_mean, 'a_std': a_std,
        'p_mean': p_mean, 'p_std': p_std,
        'angle_std': angle_std,
        'tremor_power': tremor_power,
        'curvature_mean': curvature_mean,
        'path_straightness': path_straightness,
        'ncv_rel': ncv_rel,
        'jerk_mean': jerk_mean,
    }


EXTENDED_KEYS = ['v_mean', 'v_std', 'a_mean', 'a_std',
                 'p_mean', 'p_std', 'angle_std', 'tremor_power',
                 'curvature_mean', 'path_straightness',
                 'ncv_rel', 'jerk_mean']

EXTENDED_NGS_CORE = NGS_CORE  # NGS extraction (7 metrics) is graph-size-agnostic


def build_subject_ngs_extended(svc_data, n_windows=N_WINDOWS):
    if svc_data is None or len(svc_data) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None
    n = len(svc_data)
    win_size = n // n_windows
    feat_rows = []
    for wi in range(n_windows):
        start = wi * win_size
        end = start + win_size if wi < n_windows - 1 else n
        feats = extract_extended_kinematic_features(svc_data[start:end])
        if feats is None:
            return None
        feat_rows.append([feats[k] for k in EXTENDED_KEYS])
    Xw = np.array(feat_rows)
    if not np.all(np.isfinite(Xw)):
        return None
    Xz = StandardScaler().fit_transform(Xw)
    G = build_nig(Xz, label='extended')
    return extract_ngs(G)


def nested_select_fit_predict(X, y, alpha=0.05):
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X):
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        pd_mask, hc_mask = ytr == 1, ytr == 0
        keep = []
        for i in range(Xtr.shape[1]):
            if pd_mask.sum() >= 3 and hc_mask.sum() >= 3:
                try:
                    _, p = stats.mannwhitneyu(Xtr[pd_mask, i], Xtr[hc_mask, i])
                except ValueError:
                    p = 1.0
            else:
                p = 1.0
            if p < alpha:
                keep.append(i)
        if not keep:
            keep = list(range(Xtr.shape[1]))
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(Xtr[:, keep]), ytr)
        yt.append(y[te][0])
        yp.append(clf.predict_proba(sc.transform(Xte[:, keep]))[0, 1])
    return roc_auc_score(yt, yp)


if __name__ == '__main__':
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

    print("Building 12-dim extended within-subject NGS (10 original + NCV + jerk)...")
    rows, n_excluded = [], 0
    for _, row in part_df.iterrows():
        ngs = build_subject_ngs_extended(row['svc_data'])
        if ngs is None:
            n_excluded += 1
            continue
        out = {'label_int': 1 if row['label'] == 'PD' else 0}
        out.update({k: ngs.get(k, 0.0) for k in EXTENDED_NGS_CORE})
        rows.append(out)
    df = pd.DataFrame(rows)
    print(f"n={len(df)}  excluded={n_excluded}\n")

    X = df[EXTENDED_NGS_CORE].values
    y = df['label_int'].values

    auc_extended = nested_select_fit_predict(X, y)

    print("="*70)
    print("RESULT: 12-node graph (original 10 + NCV + jerk), nested-selection LOPO-CV")
    print("="*70)
    print(f"AUC = {auc_extended:.4f}")
    print(f"\nFor reference, primary 10-node result (this paper): AUC = 0.6150")
    print(f"Difference: {auc_extended - 0.6150:+.4f}")
    print("\nReport this honestly regardless of direction. This is our own")
    print("extension using verified, general kinematic concepts (Drotar et al.")
    print("2016, Table 2) -- not a reconstruction of Kashid & Deshpande's")
    print("specific, undisclosed feature formulas.")
