"""
Which Parkinson's Drawings images entered the transfer analysis?
Counts the rows returned by load_third_dataset by drawing type and label,
and, independently, the image files on disk.

Usage:  python r2_check_drawings.py
Output: printed counts
"""
import os, collections
from third_dataset_transfer import load_third_dataset, THIRD_DATASET_ROOT

dr = load_third_dataset(THIRD_DATASET_ROOT)
print(f"rows loaded: {len(dr)}   columns: {list(dr.columns)[:20]}")
pathcols = [c for c in dr.columns if dr[c].dtype == object and dr[c].astype(str).str.contains(r'\.(png|jpg|jpeg)', case=False).any()]
if pathcols:
    s = dr[pathcols[0]].astype(str).str.lower()
    kind = s.map(lambda p: 'spiral' if 'spiral' in p else ('wave' if 'wave' in p else 'other'))
    print(collections.Counter(zip(kind, dr['label_int'])))
else:
    print("no path column found; label counts:", dr['label_int'].value_counts().to_dict())
cnt = collections.Counter()
for root, _, files in os.walk(THIRD_DATASET_ROOT):
    n = sum(fn.lower().endswith(('.png', '.jpg', '.jpeg')) for fn in files)
    if n: cnt[os.path.relpath(root, THIRD_DATASET_ROOT)] += n
print("image files on disk by folder:")
for k, v in sorted(cnt.items()): print(f"   {k:<45} {v}")
