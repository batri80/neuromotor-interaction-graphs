"""
Cross-dataset transfer, rerun with EXPLICIT source-only preprocessing
(verification of source-only preprocessing) -- v2, with image loading fully implemented.

WHAT'S COMPLETE:
  - Folder-walking and label assignment for NewHandPD and Parkinson's
    Drawings, using the directory structures documented in the
    manuscript (Section 4.1) and the standard, publicly documented
    layout of the Kaggle 'kmader/parkinsons-drawings' dataset.
  - The core fix itself: an explicit StandardScaler fit ONLY on the
    source dataset, applied unchanged to the target -- this is the
    actual logic being verified.

WHAT STILL NEEDS ONE LINE FROM YOU:
  - The Phi10 feature-extraction call (extract_phi10_from_image below).
    Definition 3.1 in the manuscript involves multi-scale parameters
    (weights w_k, number of scales, frequency band B) that aren't fully
    pinned to specific numeric values in what I have direct access to.
    Rather than guess at them -- which risks silently testing a
    DIFFERENT feature computation than the one that actually produced
    your original Table 15 numbers -- please import and call your
    existing compute_phi10-equivalent function there. It should already
    exist in your codebase (likely in ngs_pipeline_v2.py or similar,
    since it's what generated the original results).

If your NewHandPD/Drawings folder paths differ from what's below,
adjust NEWHANDPD_ROOT and DRAWINGS_ROOT accordingly.
"""

import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ngs_pipeline_v2 import load_pahaw, CONFIG
# ---- Plug in your existing Phi10 feature-extraction function here ----
# Example (adjust to your actual function name/signature):
# from ngs_pipeline_v2 import compute_phi10
from PIL import Image

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
feat_cols = CONFIG['feature_names']

# Adjust these two paths if your local folder layout differs
NEWHANDPD_ROOT = './'      # expects PatientSpiral-resized/, HealthySpiral-resized/ here
DRAWINGS_ROOT  = './third_dataset/spiral/'  # expects training/{healthy,parkinson}, testing/{healthy,parkinson}


def extract_phi10_from_image(image_path):
    """
    *** THIS IS THE ONE PIECE TO PLUG IN ***
    Should return a length-10 array matching CONFIG['feature_names']
    column order (same Phi10 formulas as Definition 3.1 / Table 1).
    Replace the body below with a call to your existing function, e.g.:

        img = Image.open(image_path).convert('L')
        return compute_phi10(np.array(img))

    Left as a stub for now so this script fails loudly (not silently
    with wrong numbers) until you wire in the real function.
    """
    raise NotImplementedError(
        "Plug in your existing Phi10 feature-extraction function here "
        "(see docstring). Not implemented deliberately -- guessing at "
        "Definition 3.1's scale/weight parameters risks silently testing "
        "a different feature computation than your original pipeline."
    )


def load_pahaw_phi10():
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    X = part_df[feat_cols].values
    y = (part_df['label'] == 'PD').astype(int).values
    return X, y


def load_newhandpd_phi10():
    """NewHandPD: images in PatientSpiral-resized/ (PD) and
    HealthySpiral-resized/ (HC), per manuscript Section 4.1."""
    patient_dir = os.path.join(NEWHANDPD_ROOT, 'PatientSpiral-resized')
    healthy_dir = os.path.join(NEWHANDPD_ROOT, 'HealthySpiral-resized')

    rows, labels = [], []
    for label_val, folder in [(1, patient_dir), (0, healthy_dir)]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected NewHandPD folder not found: {folder}\n"
                f"Adjust NEWHANDPD_ROOT at the top of this script."
            )
        image_paths = sorted(glob.glob(os.path.join(folder, '*.png')) +
                              glob.glob(os.path.join(folder, '*.jpg')))
        for p in image_paths:
            feats = extract_phi10_from_image(p)
            rows.append(feats)
            labels.append(label_val)

    X = np.array(rows)
    y = np.array(labels)
    return X, y


def load_drawings_phi10():
    """Parkinson's Drawings (Kaggle kmader/parkinsons-drawings): standard
    layout is <root>/training/{healthy,parkinson}/*.png and
    <root>/testing/{healthy,parkinson}/*.png. Both splits are pooled
    here since the manuscript's cross-dataset transfer uses the full
    dataset, not its internal train/test split (that split is used only
    for the separate within-dataset sanity check reported alongside
    Table 15)."""
    rows, labels = [], []
    for split in ['training', 'testing']:
        for label_name, label_val in [('parkinson', 1), ('healthy', 0)]:
            folder = os.path.join(DRAWINGS_ROOT, split, label_name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Expected Parkinson's Drawings folder not found: {folder}\n"
                    f"Adjust DRAWINGS_ROOT at the top of this script."
                )
            image_paths = sorted(glob.glob(os.path.join(folder, '*.png')) +
                                  glob.glob(os.path.join(folder, '*.jpg')))
            for p in image_paths:
                feats = extract_phi10_from_image(p)
                rows.append(feats)
                labels.append(label_val)

    X = np.array(rows)
    y = np.array(labels)
    return X, y


def transfer_auc_source_only_scaling(X_source, y_source, X_target, y_target):
    """The actual fix being verified: scaler fit ONLY on source, applied
    unchanged to target. No target statistic used anywhere before
    scoring."""
    scaler = StandardScaler().fit(X_source)
    X_source_scaled = scaler.transform(X_source)
    X_target_scaled = scaler.transform(X_target)  # target never contributes to fitting

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_source_scaled, y_source)
    prob = clf.predict_proba(X_target_scaled)[:, 1]
    return roc_auc_score(y_target, prob)


if __name__ == '__main__':
    print("Loading all three datasets' Phi10 features...")
    datasets = {}
    datasets['PaHaW'] = load_pahaw_phi10()
    datasets['NewHandPD'] = load_newhandpd_phi10()
    datasets['Drawings'] = load_drawings_phi10()

    for name, (X, y) in datasets.items():
        print(f"  {name}: n={len(y)}, PD={int(y.sum())}, HC={len(y)-int(y.sum())}")

    print("\n" + "="*70)
    print("CROSS-DATASET TRANSFER MATRIX (explicit source-only scaling)")
    print("="*70)

    names = list(datasets.keys())
    print(f"{'Train -> Test':<25}{'AUC':<10}")
    matrix = {}
    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            X_s, y_s = datasets[src]
            X_t, y_t = datasets[tgt]
            auc = transfer_auc_source_only_scaling(X_s, y_s, X_t, y_t)
            matrix[(src, tgt)] = auc
            print(f"{src+' -> '+tgt:<25}{auc:<10.4f}")

    print("\nFor reference, current manuscript Table 15 values:")
    print("  PaHaW->NewHandPD: 0.6714   PaHaW->Drawings: 0.5418")
    print("  NewHandPD->PaHaW: 0.5000   NewHandPD->Drawings: 0.5998")
    print("  Drawings->PaHaW:  0.4707   Drawings->NewHandPD: 0.7741")
    print("\nIf these match closely, that confirms source-only scaling was")
    print("already (implicitly) what the original pipeline did, and the fix")
    print("is purely a documentation addition to the manuscript text. If they")
    print("differ meaningfully, Table 15 needs updating to these values --")
    print("report back either way.")
