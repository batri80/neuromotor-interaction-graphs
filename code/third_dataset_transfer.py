import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ngs_p4_newhandpd import CONFIG as CONFIG_NHPD, extract_phi10, load_newhandpd
from ngs_pipeline_v2 import load_pahaw, CONFIG as CONFIG_PAHAW

CONFIG_PAHAW['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG_PAHAW['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
CONFIG_NHPD['healthy_root'] = './healthy'
CONFIG_NHPD['patient_root'] = './patient'

THIRD_DATASET_ROOT = Path('./third_dataset/dataset')
IMAGE_SIZE = CONFIG_NHPD['image_size']
feat_cols = CONFIG_PAHAW['feature_names']  # same Phi10 names everywhere


def load_third_dataset(root):
    """Image-level loading (no assumed subject grouping -- naming scheme
    undocumented, so we don't risk a false participant structure)."""
    rows = []
    for split in ['train', 'test']:
        for label_name, label_int in [('healthy', 0), ('parkinson', 1)]:
            folder = root / split / label_name
            if not folder.exists():
                continue
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                    continue
                try:
                    pil = Image.open(img_path).convert('RGB')
                    if pil.size != IMAGE_SIZE:
                        pil = pil.resize(IMAGE_SIZE, Image.LANCZOS)
                    arr = np.array(pil, dtype=np.uint8)
                    feat = extract_phi10(arr)
                    feat['label_int'] = label_int
                    feat['filename'] = img_path.name
                    feat['split'] = split
                    rows.append(feat)
                except Exception as e:
                    print(f"  [skip] {img_path.name}: {e}")
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("── Loading third dataset (Zham et al. spiral drawings, image-level) ──")
    third_df = load_third_dataset(THIRD_DATASET_ROOT)
    print(f"Loaded {len(third_df)} images: "
          f"{(third_df['label_int']==1).sum()} PD, {(third_df['label_int']==0).sum()} HC\n")

    X_third = third_df[feat_cols].values
    y_third = third_df['label_int'].values

    print("── Loading PaHaW and NewHandPD (participant-level, already validated) ──")
    part_df_pahaw = load_pahaw(CONFIG_PAHAW['pahaw_root'], CONFIG_PAHAW['corpus_path'])
    X_pahaw = part_df_pahaw[feat_cols].values
    y_pahaw = (part_df_pahaw['label'] == 'PD').astype(int).values

    _, part_df_nhpd = load_newhandpd(CONFIG_NHPD['healthy_root'], CONFIG_NHPD['patient_root'])
    X_nhpd = part_df_nhpd[feat_cols].values
    y_nhpd = part_df_nhpd['label_int'].values

    print("\n" + "="*80)
    print("THREE-WAY CROSS-DATASET TRANSFER (Phi10 feature space, shared across all)")
    print("="*80)

    def train_test(name_tr, X_tr, y_tr, name_te, X_te, y_te):
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(sc.transform(X_tr), y_tr)
        prob = clf.predict_proba(sc.transform(X_te))[:, 1]
        auc = roc_auc_score(y_te, prob)
        print(f"Train {name_tr:<12} -> Test {name_te:<12}  AUC={auc:.4f}  (n_test={len(y_te)})")
        return auc

    train_test("PaHaW", X_pahaw, y_pahaw, "NewHandPD", X_nhpd, y_nhpd)
    train_test("NewHandPD", X_nhpd, y_nhpd, "PaHaW", X_pahaw, y_pahaw)
    print()
    train_test("PaHaW", X_pahaw, y_pahaw, "ThirdDataset", X_third, y_third)
    train_test("NewHandPD", X_nhpd, y_nhpd, "ThirdDataset", X_third, y_third)
    train_test("ThirdDataset", X_third, y_third, "PaHaW", X_pahaw, y_pahaw)
    train_test("ThirdDataset", X_third, y_third, "NewHandPD", X_nhpd, y_nhpd)

    print("\n── Within-dataset (ThirdDataset) sanity check ──")
    print("(train/test split as bundled in the repo, if meaningful)")
    train_mask = third_df['split'] == 'train'
    test_mask = third_df['split'] == 'test'
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        train_test("ThirdDataset-train", X_third[train_mask], y_third[train_mask],
                    "ThirdDataset-test", X_third[test_mask], y_third[test_mask])
