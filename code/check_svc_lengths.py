from ngs_pipeline_v2 import load_pahaw, CONFIG
import numpy as np

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])

lengths = []
for _, row in part_df.iterrows():
    svc = row.get('svc_data')
    if svc is not None:
        lengths.append(len(svc))

lengths = np.array(lengths)
print(f"n subjects with svc_data: {len(lengths)}")
print(f"min={lengths.min()}  max={lengths.max()}  mean={lengths.mean():.0f}  median={np.median(lengths):.0f}")
print(f"25th pct={np.percentile(lengths,25):.0f}  75th pct={np.percentile(lengths,75):.0f}")

# at 150Hz, convert to seconds
print(f"\nAt 150Hz recording rate:")
print(f"min={lengths.min()/150:.1f}s  median={np.median(lengths)/150:.1f}s  max={lengths.max()/150:.1f}s")

# For different window counts, how many samples per window at the WORST case (min length)?
print(f"\nSamples-per-window at min-length subject, for various N_WINDOWS:")
for n_w in [10, 15, 20, 25, 30, 40, 50]:
    spw = lengths.min() // n_w
    n_subjects_excluded = (lengths < n_w * 50).sum()
    print(f"  N_WINDOWS={n_w:<4} min samples/window={spw:<6} "
          f"subjects excluded (below 50 samples/window floor)={n_subjects_excluded}/{len(lengths)}")
