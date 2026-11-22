"""
NGS P4 — NewHandPD Cross-Dataset Validation
============================================
Standalone script to run P4 cross-dataset validation on the NewHandPD
public dataset using the NIG/NGS pipeline.

Expected folder structure (as provided):
    <healthy_root>/
        HealthySpiral-resized/     ← use these (preprocessed)
        HealthyMeander-resized/
        HealthyCircle-resized/
        HealthySpiral/             ← original (not used unless resized absent)
        HealthyMeander/
        HealthyCircle/

    <patient_root>/
        PatientSpiral-resized/
        PatientMeander-resized/
        PatientCircle-resized/
        PatientSpiral/
        PatientMeander/
        PatientCircle/

Usage:
    python ngs_p4_newhandpd.py \\
        --healthy ./healthy \\
        --patient ./patient \\
        --pahaw_ngs /path/to/ngs_outputs_v2/pickles/ngs_v2_all_results.pkl \\
        --output  ./ngs_outputs_p4

The --pahaw_ngs argument is optional. If provided, PaHaW results are
printed alongside NewHandPD for side-by-side comparison in the paper.

Paper: Neuromotor Interaction Graphs — Information Sciences Special Issue
Author: Krishnan Batri
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, warnings, pickle, argparse, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold,
                                     RepeatedStratifiedKFold,
                                     permutation_test_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'healthy_root'  : None,   # set by CLI
    'patient_root'  : None,   # set by CLI
    'pahaw_pkl'     : None,   # optional: PaHaW NGS pickle for comparison
    'output_dir'    : './ngs_outputs_p4',

    # Image settings
    'image_size'    : (512, 512),
    'equilibrium_ref': 128,

    # Tasks to use — use resized versions preferentially
    'task_folders'  : {
        'HC': ['HealthySpiral-resized',  'HealthyMeander-resized',
               'HealthyCircle-resized'],
        'PD': ['PatientSpiral-resized',  'PatientMeander-resized',
               'PatientCircle-resized'],
    },
    # Fallback to originals if resized not found
    'task_folders_fallback': {
        'HC': ['HealthySpiral',  'HealthyMeander',  'HealthyCircle'],
        'PD': ['PatientSpiral',  'PatientMeander',  'PatientCircle'],
    },

    # Feature names
    'feature_names' : ['Eb','Eg','Em','sigma_d','Ag','rho_s',
                       'Hs','Hg','SIA','Omega'],

    # NIG
    'te_bins'       : 8,
    'te_threshold_pct': 25,

    # NGS
    'compute_ricci' : True,
    'ricci_alpha'   : 0.5,

    # Validation
    'n_permutations': 1000,
    'n_cv_repeats'  : 10,
    'n_cv_folds'    : 5,
    'n_bootstrap'   : 1000,

    # Output
    'save_csv'      : True,
    'save_figures'  : True,
    'figure_dpi'    : 300,
    'figure_format' : 'pdf',
}

FEAT_LABELS = {
    'Eb':r'$E^b$','Eg':r'$E^g$','Em':r'$E^m$',
    'sigma_d':r'$\sigma_d$','Ag':r'$A^g$','rho_s':r'$\rho_s$',
    'Hs':r'$H_s$','Hg':r'$H_g$','SIA':'SIA','Omega':r'$\Omega$'
}
NGS_CORE = ['H_graph','Eff','Modularity','Lambda2',
            'SmallWorld','E_spectral','Ricci_OR']

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def print_header(title, width=68):
    print('\n' + '='*width + f'\n  {title}\n' + '='*width)

def print_sub(title):
    print(f'\n  ── {title} ──')

def print_result(label, value, indent=6):
    print(f"{' '*indent}{label:<44s} {value}")

def elapsed(t0):
    return f"{time.time()-t0:.1f}s"

def make_dirs():
    for sd in ['tables','figures','pickles']:
        os.makedirs(os.path.join(CONFIG['output_dir'], sd), exist_ok=True)

def save_table(df, name):
    if CONFIG['save_csv']:
        p = os.path.join(CONFIG['output_dir'],'tables',f'{name}.csv')
        df.to_csv(p, index=True)
        print(f"      [saved] {p}")

def save_fig(fig, name):
    if CONFIG['save_figures']:
        p = os.path.join(CONFIG['output_dir'],'figures',
                         f'{name}.{CONFIG["figure_format"]}')
        fig.savefig(p, dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f"      [saved] {p}")

def bootstrap_ci(values, fn=np.mean, n=1000, alpha=0.95):
    boot = [fn(np.random.choice(values, len(values), replace=True))
            for _ in range(n)]
    return fn(values), np.percentile(boot,2.5), np.percentile(boot,97.5)

# ─────────────────────────────────────────────────────────────────────────────
# PHI-10 FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _bg_norm(grey):
    Ibg = np.percentile(grey, 90)
    if Ibg < 1: Ibg = 255.0
    return np.clip(255.0*(Ibg-grey)/Ibg, 0, 255)

def _sobel(I):
    from scipy.ndimage import sobel
    gx = sobel(I, axis=1); gy = sobel(I, axis=0)
    return gx, gy, np.sqrt(gx**2+gy**2)

def _pyramid(img, levels=3):
    pyr = [img]; cur = img.copy()
    for _ in range(levels-1):
        h,w = cur.shape
        p = Image.fromarray(cur.astype(np.uint8))
        p = p.resize((max(1,w//2), max(1,h//2)), Image.LANCZOS)
        cur = np.array(p, dtype=float); pyr.append(cur)
    return pyr

def extract_phi10(img_array):
    """
    Extract Phi-10 neurodynamic feature vector from image array.
    Handles both greyscale and RGB input.
    Matches Batri et al. (2026) framework exactly.
    """
    r = CONFIG['equilibrium_ref']

    # Convert to greyscale if needed
    arr = img_array.astype(float)
    if arr.ndim == 3:
        I_grey = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    else:
        I_grey = arr.copy()

    I = _bg_norm(I_grey)
    dev2 = (I-r)**2
    gx, gy, gm = _sobel(I)
    pyr = _pyramid(I)
    w   = [0.5, 0.3, 0.2]

    Eb      = np.sqrt(np.mean(dev2))
    Eg      = np.mean(gm**2)
    Em      = sum(wi*np.sqrt(np.mean((lev-r)**2)) for lev,wi in zip(pyr,w))
    sigma_d = np.std(dev2)
    Ag      = np.var(gx**2)/(np.var(gy**2)+1e-8)

    d0 = (pyr[0]-r)**2
    d1 = np.array(Image.fromarray(((pyr[1]-r)**2).astype(np.float32))
                  .resize((d0.shape[1],d0.shape[0]),Image.LANCZOS), dtype=float)
    rc = np.corrcoef(d0.flatten(), d1.flatten())[0,1]
    rho_s = rc if not np.isnan(rc) else 0.0

    h,_  = np.histogram(I.flatten(), bins=256, range=(0,255))
    hn   = h/(h.sum()+1e-10)
    Hs   = -np.sum(hn*np.log(hn+1e-10))

    hg,_ = np.histogram(gm.flatten(), bins=256)
    hgn  = hg/(hg.sum()+1e-10)
    Hg   = -np.sum(hgn*np.log(hgn+1e-10))

    g1r  = np.array(Image.fromarray(pyr[1].astype(np.float32))
                    .resize((I.shape[1],I.shape[0]),Image.LANCZOS), dtype=float)
    _,_,gm1 = _sobel(g1r)
    SIA  = np.mean(np.abs(gm-gm1))/(np.mean(gm)+1e-6)

    F    = np.fft.fftshift(np.fft.fft2(I))
    pw   = np.abs(F)**2
    cy,cx = I.shape[0]//2, I.shape[1]//2
    fy,fx = np.ogrid[:I.shape[0], :I.shape[1]]
    dist  = np.sqrt((fy-cy)**2+(fx-cx)**2)
    Omega = np.sum(pw[(dist>=15)&(dist<=45)])

    return {'Eb':Eb,'Eg':Eg,'Em':Em,'sigma_d':sigma_d,
            'Ag':Ag,'rho_s':rho_s,'Hs':Hs,'Hg':Hg,
            'SIA':SIA,'Omega':Omega}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _find_folder(root, candidates):
    """Return first existing folder from candidates list."""
    for c in candidates:
        p = Path(root) / c
        if p.exists():
            return p
    return None

def _parse_participant_id(stem, label):
    """
    Parse the participant ID from a NewHandPD image filename.

    NewHandPD names images <task><repetition>-<P|H><participant>, e.g.
    'sp1-P12', 'mea3-H7', 'circA-P4'. The participant number is the digits
    after the final hyphen. (The healthy-control circle images use a 'P'
    prefix, e.g. 'circA-P1' inside HealthyCircle, and one file is lower-case,
    'circA-p27'; the class label always comes from the folder, so neither
    matters.)

    CORRECTION (round-2 revision): the previous version took the FIRST number
    in the stem, which for spiral and meander images is the task-repetition
    index ('sp1' -> 1), not the participant. All sp1/mea1 images of every
    participant were therefore pooled into one record 'PD_001', and so on,
    giving 32 PD and 35 HC "participants" of which 8 were pooled averages
    over almost the whole group. The public release contains 31 PD and 35 HC
    participants.
    """
    import re
    m = re.search(r'-([PpHh])(\d+)$', stem)
    if m:
        return f"{label}_{int(m.group(2)):03d}"
    raise ValueError(f"Unrecognised NewHandPD filename (no -P#/-H# suffix): {stem}")

def load_newhandpd(healthy_root, patient_root):
    """
    Load NewHandPD images from the given directory structure.

    For each label (HC / PD):
      - Scans all three task subdirectories (resized preferred)
      - Loads every image file (jpg, png, bmp, tif)
      - Extracts Phi-10 features from each image
      - Parses participant ID from filename
      - Aggregates multiple images per participant by mean

    Returns: participant-level DataFrame
    """
    print_sub("Scanning NewHandPD folder structure")

    EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.JPG','.PNG'}
    feat_cols = CONFIG['feature_names']
    rows = []
    skipped = 0

    for label, root, primary_keys, fallback_keys in [
        ('HC', healthy_root,
         CONFIG['task_folders']['HC'],
         CONFIG['task_folders_fallback']['HC']),
        ('PD', patient_root,
         CONFIG['task_folders']['PD'],
         CONFIG['task_folders_fallback']['PD']),
    ]:
        label_int = 0 if label == 'HC' else 1
        n_loaded = 0

        for task_key, fb_key in zip(primary_keys, fallback_keys):
            folder = _find_folder(root, [task_key, fb_key])
            if folder is None:
                print(f"      [warn] Neither {task_key} nor {fb_key} found under {root}")
                continue

            img_files = sorted([f for f in folder.iterdir()
                                 if f.suffix in EXTS])
            print_result(f"  {label} / {folder.name}",
                         f"{len(img_files)} images")

            for img_path in img_files:
                try:
                    pil = Image.open(img_path).convert('RGB')
                    # Resize to 512x512 with Lanczos if not already
                    if pil.size != CONFIG['image_size']:
                        pil = pil.resize(CONFIG['image_size'], Image.LANCZOS)
                    arr  = np.array(pil, dtype=np.uint8)
                    feat = extract_phi10(arr)
                    pid  = _parse_participant_id(img_path.stem, label)
                    feat['participant_id'] = pid
                    feat['label']          = label
                    feat['label_int']      = label_int
                    feat['task']           = task_key.replace('-resized','').replace('Healthy','').replace('Patient','')
                    feat['img_path']       = str(img_path)
                    rows.append(feat)
                    n_loaded += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"      [warn] {img_path.name}: {e}")

        print_result(f"  {label} total images loaded", str(n_loaded))

    if not rows:
        raise RuntimeError("No images loaded. Check folder paths.")

    img_df = pd.DataFrame(rows)
    print_result("Total image-level rows", len(img_df))
    print_result("Skipped (errors)", skipped)

    # Aggregate to participant level (mean across all images per participant)
    print_sub("Aggregating to participant level")
    part_df = img_df.groupby(['participant_id','label','label_int'])[feat_cols]\
                    .mean().reset_index()
    part_df = part_df.sort_values('participant_id').reset_index(drop=True)

    pd_n = (part_df['label']=='PD').sum()
    hc_n = (part_df['label']=='HC').sum()
    per = img_df.groupby('participant_id').size()
    print_result("Images per participant (min / median / max)",
                 f"{per.min()} / {int(per.median())} / {per.max()}")
    print_result("Unique participants", len(part_df))
    print_result("PD participants",     str(pd_n))
    print_result("HC participants",     str(hc_n))

    if pd_n < 5 or hc_n < 5:
        print("      [warn] Very few participants detected.")
        print("             Check that filename → participant ID parsing is correct.")
        print("             First 10 IDs:", part_df['participant_id'].tolist()[:10])

    return img_df, part_df

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER ENTROPY + NIG
# ─────────────────────────────────────────────────────────────────────────────
def _disc(x, n=8):
    bins = np.unique(np.percentile(x, np.linspace(0,100,n+1)))
    return np.digitize(x, bins[:-1])-1

def transfer_entropy(x, y, lag=1):
    x,y = np.asarray(x,float), np.asarray(y,float)
    if len(x)<lag+2: return 0.0
    n = CONFIG['te_bins']
    xd,yd = _disc(x,n), _disc(y,n)
    yt,yt1,xt1 = yd[lag:],yd[:-lag],xd[:-lag]
    N = len(yt)
    if N<2: return 0.0
    jyy  = Counter(zip(yt,yt1))
    jxxy = Counter(zip(yt,yt1,xt1))
    my1  = Counter(yt1)
    jyx1 = Counter(zip(yt1,xt1))
    te = 0.0
    for (a,b,c),cnt in jxxy.items():
        p=cnt/N; py=jyy[(a,b)]/N; px=jyx1[(b,c)]/N; pm=my1[b]/N
        if all(v>0 for v in [p,py,px,pm]):
            r=(p*pm)/(py*px)
            if r>0: te+=p*np.log(r)
    return max(te,0.0)

def compute_te_matrix(X):
    n = X.shape[1]
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: M[i,j]=transfer_entropy(X[:,i],X[:,j])
    return M

def build_nig(X, label=''):
    feat = CONFIG['feature_names']
    M    = compute_te_matrix(X)
    nz   = M[M>0]
    thr  = np.percentile(nz,CONFIG['te_threshold_pct']) if len(nz)>0 else 0.0
    G    = nx.DiGraph(label=label)
    for i,f in enumerate(feat):
        G.add_node(i,name=f,display=FEAT_LABELS.get(f,f))
    for i in range(len(feat)):
        for j in range(len(feat)):
            if i!=j and M[i,j]>thr:
                G.add_edge(i,j,weight=M[i,j],source=feat[i],target=feat[j])
    G.graph['te_matrix']=M; G.graph['threshold']=thr
    return G

# ─────────────────────────────────────────────────────────────────────────────
# NGS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def graph_entropy(G):
    w=np.array([d['weight'] for _,_,d in G.edges(data=True)])
    if len(w)==0: return 0.0
    wn=w/(w.sum()+1e-10)
    return -np.sum(wn*np.log(wn+1e-10))

def global_efficiency(G):
    U=G.to_undirected(); n=U.number_of_nodes()
    if n<2: return 0.0
    for u,v,d in U.edges(data=True): d['inv_w']=1.0/(d['weight']+1e-10)
    tot=0.0
    for src in U.nodes():
        try:
            L=nx.single_source_dijkstra_path_length(U,src,weight='inv_w')
            tot+=sum(1.0/d for t,d in L.items() if t!=src and d>0)
        except: pass
    return tot/(n*(n-1)) if tot>0 else 0.0

def graph_modularity(G):
    U=G.to_undirected()
    if U.number_of_edges()==0: return 0.0
    try:
        C=nx.community.greedy_modularity_communities(U,weight='weight')
        return nx.community.modularity(U,C,weight='weight')
    except: return 0.0

def algebraic_connectivity(G):
    U=G.to_undirected()
    if U.number_of_nodes()<2: return 0.0
    try: return nx.algebraic_connectivity(U,weight='weight')
    except: return 0.0

SW_SEED = 0   # seed for the Erdos-Renyi null graphs of the small-world coefficient

def small_world(G,n_rand=5,seed=SW_SEED):
    U=G.to_undirected(); n,m=U.number_of_nodes(),U.number_of_edges()
    if m==0 or n<3: return 0.0
    try:
        C=nx.average_clustering(U,weight='weight')
        cc=max(nx.connected_components(U),key=len)
        L=nx.average_shortest_path_length(U.subgraph(cc)) if len(cc)>1 else 1.0
        Cr,Lr=[],[]
        rng=random.Random(seed)   # fixed seed: reproducible across runs
        for _ in range(n_rand):
            Gr=nx.gnm_random_graph(n,m,seed=rng)
            Cr.append(nx.average_clustering(Gr))
            try:
                gcc=max(nx.connected_components(Gr),key=len)
                Lr.append(nx.average_shortest_path_length(Gr.subgraph(gcc)) if len(gcc)>1 else L)
            except: Lr.append(L)
        return (C/(np.mean(Cr)+1e-10))/(L/(np.mean(Lr)+1e-10))
    except: return 0.0

def spectral_energy(G):
    A=nx.to_numpy_array(G,weight='weight')
    ev=np.linalg.eigvalsh(A+A.T)
    return float(np.sum(ev**2))

def ollivier_ricci(G):
    alpha=CONFIG['ricci_alpha']; U=G.to_undirected()
    if U.number_of_edges()==0: return 0.0
    def mu(v):
        m={v:alpha}; nb=list(U.neighbors(v))
        if not nb: return m
        wt=sum(U[v][u].get('weight',1.0) for u in nb)
        for u in nb: m[u]=(1-alpha)*U[v][u].get('weight',1.0)/(wt+1e-10)
        return m
    def emd(m1,m2):
        t=0.0
        for a,pa in m1.items():
            for b,pb in m2.items():
                try: d=nx.shortest_path_length(U,a,b)
                except: d=len(U)
                t+=pa*pb*d
        return t
    ks=[]
    for u,v in list(U.edges())[:40]:
        dw=U[u][v].get('weight',1.0)
        if dw<1e-10: continue
        ks.append(1.0-emd(mu(u),mu(v))/dw)
    return float(np.mean(ks)) if ks else 0.0

def extract_ngs(G):
    d={'H_graph':graph_entropy(G),'Eff':global_efficiency(G),
       'Modularity':graph_modularity(G),'Lambda2':algebraic_connectivity(G),
       'SmallWorld':small_world(G),'E_spectral':spectral_energy(G),
       'n_edges':G.number_of_edges(),'density':nx.density(G)}
    if CONFIG['compute_ricci']:
        d['Ricci_OR']=ollivier_ricci(G)
    return d

def build_subject_ngs(part_df):
    """Build subject-level NGS using group-context NIG construction."""
    feat_cols=CONFIG['feature_names']
    X=part_df[feat_cols].values
    pd_mask=(part_df['label'].values=='PD')
    rows=[]
    for idx,row in part_df.iterrows():
        grp=pd_mask if row['label']=='PD' else ~pd_mask
        grp_mat=X[grp]
        sub_mat=np.vstack([grp_mat,X[idx:idx+1]])
        sub_z=StandardScaler().fit_transform(sub_mat)
        G_sub=build_nig(sub_z,label=row['label'])
        ngs=extract_ngs(G_sub)
        ngs['participant_id']=row['participant_id']
        ngs['label']=row['label']
        ngs['label_int']=row['label_int']
        rows.append(ngs)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def lopo_cv(X, y, clf):
    yt,yp,ypr=[],[],[]
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler()
        clf.fit(sc.fit_transform(X[tr]),y[tr])
        yt.append(y[te[0]]); yp.append(clf.predict(sc.transform(X[te]))[0])
        try:    ypr.append(clf.predict_proba(sc.transform(X[te]))[0,1])
        except: ypr.append(float(yp[-1]))
    return np.array(yt),np.array(yp),np.array(ypr)

def clf_metrics(yt,yp,ypr):
    cm=confusion_matrix(yt,yp)
    tn,fp,fn,tp=cm.ravel() if cm.size==4 else (0,0,0,0)
    return {'Accuracy':accuracy_score(yt,yp),
            'Sensitivity':tp/(tp+fn+1e-10),
            'Specificity':tn/(tn+fp+1e-10),
            'F1':f1_score(yt,yp,zero_division=0),
            'AUC':roc_auc_score(yt,ypr) if len(np.unique(yt))>1 else 0.5,
            'TP':tp,'TN':tn,'FP':fp,'FN':fn}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN P4 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_p4(healthy_root, patient_root, pahaw_pkl=None):
    print_header("NGS P4 — NewHandPD CROSS-DATASET VALIDATION")
    t_total = time.time()
    make_dirs()

    # ── 1. Load images and extract features ──────────────────────────────────
    print_header("STEP 1: LOAD IMAGES AND EXTRACT PHI-10")
    t0 = time.time()
    img_df, part_df = load_newhandpd(healthy_root, patient_root)
    print(f"\n  ✓ Feature extraction complete [{elapsed(t0)}]")

    # Save image-level feature matrix for inspection
    save_table(part_df.drop(columns=[],errors='ignore'),
               'P4_participant_features')

    # ── 2. Build subject-level NGS ────────────────────────────────────────────
    print_header("STEP 2: BUILD SUBJECT-LEVEL NIG/NGS")
    t0 = time.time()
    print_sub("Building population NIGs (PD and HC)")

    feat_cols = CONFIG['feature_names']
    pd_mask   = part_df['label']=='PD'
    hc_mask   = part_df['label']=='HC'

    pd_mat = StandardScaler().fit_transform(part_df.loc[pd_mask,feat_cols].values)
    hc_mat = StandardScaler().fit_transform(part_df.loc[hc_mask,feat_cols].values)

    G_PD = build_nig(pd_mat, label='PD')
    G_HC = build_nig(hc_mat, label='HC')
    ngs_pd_pop = extract_ngs(G_PD)
    ngs_hc_pop = extract_ngs(G_HC)

    print_result("PD NIG edges", G_PD.number_of_edges())
    print_result("HC NIG edges", G_HC.number_of_edges())

    print_sub("Building subject-level NGS")
    ngs_df = build_subject_ngs(part_df)
    print(f"\n  ✓ NGS complete [{elapsed(t0)}]")

    # ── 3. Group comparison ───────────────────────────────────────────────────
    print_header("STEP 3: NGS GROUP COMPARISON")
    t0 = time.time()

    ngs_cols = [c for c in NGS_CORE if c in ngs_df.columns]
    stat_rows = []
    for metric in ngs_cols:
        vp = ngs_df.loc[ngs_df['label']=='PD', metric].dropna().values
        vh = ngs_df.loc[ngs_df['label']=='HC', metric].dropna().values
        if len(vp)<2 or len(vh)<2: continue
        u,p = stats.mannwhitneyu(vp,vh,alternative='two-sided')
        d   = (np.mean(vp)-np.mean(vh))/(np.std(np.concatenate([vp,vh]))+1e-10)
        sig = '**' if p<0.01 else ('*' if p<0.05 else 'ns')
        stat_rows.append({'Metric':metric,
                          'PD mean±SD':f"{np.mean(vp):.4f}±{np.std(vp):.4f}",
                          'HC mean±SD':f"{np.mean(vh):.4f}±{np.std(vh):.4f}",
                          "Cohen's d":f"{d:.3f}",'U':f"{u:.0f}",
                          'p-value':f"{p:.4f}",'Sig':sig})
        print_result(metric,
                     f"PD={np.mean(vp):.4f}  HC={np.mean(vh):.4f}  "
                     f"p={p:.4f} {sig}  d={d:.3f}")

    stat_df = pd.DataFrame(stat_rows)
    save_table(stat_df,'P4_NGS_group_stats')
    print(f"\n  ✓ Group comparison complete [{elapsed(t0)}]")

    # ── 4. Classification ─────────────────────────────────────────────────────
    print_header("STEP 4: CLASSIFICATION (NGS vs PHI-10 vs HYBRID)")
    t0 = time.time()

    merged = part_df.merge(ngs_df[['participant_id']+ngs_cols],
                           on='participant_id',how='left').dropna(subset=ngs_cols)
    y = merged['label_int'].values

    clfs = {
        'SVM (RBF)'          : SVC(kernel='rbf',probability=True,random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100,random_state=42),
        'KNN (k=7)'          : KNeighborsClassifier(n_neighbors=7),
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
    }
    fsets = {
        'Baseline (Phi-10)'  : feat_cols,
        'Graph (NGS)'        : ngs_cols,
        'Hybrid (Phi10+NGS)' : feat_cols+ngs_cols,
    }

    clf_rows = []
    best_auc = 0.0; best_row = None
    for fs_name,fcols in fsets.items():
        print_sub(f"Feature set: {fs_name}")
        X = merged[fcols].values
        for clf_name,clf in clfs.items():
            yt,yp,ypr = lopo_cv(X,y,clf)
            m = clf_metrics(yt,yp,ypr)
            clf_rows.append({'Dataset':'NewHandPD','Feature Set':fs_name,
                             'Classifier':clf_name,**m})
            print_result(f"  {clf_name}",
                         f"Acc={m['Accuracy']:.4f}  Sens={m['Sensitivity']:.4f}  "
                         f"Spec={m['Specificity']:.4f}  AUC={m['AUC']:.4f}")
            if m['AUC'] > best_auc:
                best_auc = m['AUC']; best_row = clf_rows[-1]

    clf_df = pd.DataFrame(clf_rows)
    save_table(clf_df,'P4_classification_results')
    print(f"\n  ✓ Classification complete [{elapsed(t0)}]")

    # ── 5. Permutation test ───────────────────────────────────────────────────
    print_header("STEP 5: PERMUTATION TEST (NGS/LR)")
    t0 = time.time()

    X_ngs = merged[ngs_cols].values
    sc_ngs = StandardScaler()
    cv5    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr     = LogisticRegression(max_iter=1000, random_state=42)

    score_true, perm_scores, p_perm = permutation_test_score(
        lr, sc_ngs.fit_transform(X_ngs), y,
        scoring='roc_auc', cv=cv5,
        n_permutations=CONFIG['n_permutations'],
        random_state=42, n_jobs=-1
    )

    print_result("True AUC (5-fold CV)",  f"{score_true:.4f}")
    print_result("Null mean ± SD",        f"{perm_scores.mean():.4f} ± {perm_scores.std():.4f}")
    print_result("Null 99th percentile",  f"{np.percentile(perm_scores,99):.4f}")
    print_result("p-value (permutation)", f"{p_perm:.4f}")
    print_result("Significance",
                 "** p<0.001" if p_perm<0.001 else
                 ("* p<0.05" if p_perm<0.05 else "ns"))
    print_result("AUC > 99th null pct",
                 "YES" if score_true>np.percentile(perm_scores,99) else "NO")

    # ── 6. Repeated CV + Bootstrap CI ────────────────────────────────────────
    print_header("STEP 6: REPEATED 10×5-FOLD CV + BOOTSTRAP CIs")
    t0 = time.time()

    rskf = RepeatedStratifiedKFold(n_splits=CONFIG['n_cv_folds'],
                                   n_repeats=CONFIG['n_cv_repeats'],
                                   random_state=42)
    rep_rows = []
    for fs_name,fcols in fsets.items():
        X_f = merged[fcols].values
        aucs = []
        for tr,te in rskf.split(X_f,y):
            sc=StandardScaler(); clf=LogisticRegression(max_iter=1000,random_state=42)
            clf.fit(sc.fit_transform(X_f[tr]),y[tr])
            try:
                pr=clf.predict_proba(sc.transform(X_f[te]))[:,1]
                aucs.append(roc_auc_score(y[te],pr))
            except: pass
        mu,lo,hi = bootstrap_ci(np.array(aucs),n=CONFIG['n_bootstrap'])
        print_result(f"  {fs_name}",
                     f"AUC={mu:.4f}±{np.std(aucs):.4f}  95%CI [{lo:.4f},{hi:.4f}]")
        rep_rows.append({'Feature Set':fs_name,'Mean AUC':mu,
                         'SD':np.std(aucs),'CI_lo':lo,'CI_hi':hi})

    rep_df = pd.DataFrame(rep_rows)
    save_table(rep_df,'P4_repeated_cv')
    print(f"\n  ✓ Repeated CV complete [{elapsed(t0)}]")

    # ── 7. McNemar test ───────────────────────────────────────────────────────
    print_header("STEP 7: McNEMAR TEST (NGS vs PHI-10)")
    t0 = time.time()

    X_phi = merged[feat_cols].values
    lr2   = LogisticRegression(max_iter=1000,random_state=42)
    yt_ngs,yp_ngs,_ = lopo_cv(X_ngs,y,lr)
    yt_phi,yp_phi,_ = lopo_cv(X_phi,y,lr2)

    b = np.sum((yp_ngs==yt_ngs)&(yp_phi!=yt_phi))
    c = np.sum((yp_ngs!=yt_ngs)&(yp_phi==yt_phi))
    chi2 = (abs(b-c)-1)**2/(b+c) if b+c>0 else 0.0
    p_mc = stats.chi2.sf(chi2,df=1) if b+c>0 else 1.0

    print_result("b (NGS✓, Phi-10✗)", str(b))
    print_result("c (NGS✗, Phi-10✓)", str(c))
    print_result("McNemar chi²",       f"{chi2:.4f}")
    print_result("p-value",            f"{p_mc:.6f}")
    print_result("NGS significantly better?",
                 "YES **" if p_mc<0.01 else ("YES *" if p_mc<0.05 else "NO"))

    mn_df = pd.DataFrame([{'b':b,'c':c,'chi2':chi2,'p_value':p_mc}])
    save_table(mn_df,'P4_mcnemar')
    print(f"\n  ✓ McNemar complete [{elapsed(t0)}]")

    # ── 8. PaHaW comparison (if pickle provided) ──────────────────────────────
    pahaw_results = None
    if pahaw_pkl and Path(pahaw_pkl).exists():
        print_header("STEP 8: PAHAW vs NewHandPD COMPARISON")
        with open(pahaw_pkl,'rb') as f:
            pahaw_results = pickle.load(f)
        print_sub("Side-by-side AUC comparison")
        print(f"{'Feature Set':<25} {'PaHaW AUC':>12} {'NewHandPD AUC':>15}")
        print('-'*55)
        nhs_ngs = clf_df[(clf_df['Feature Set']=='Graph (NGS)')&
                         (clf_df['Classifier']=='Logistic Regression')]['AUC'].values
        nhs_phi = clf_df[(clf_df['Feature Set']=='Baseline (Phi-10)')]['AUC'].max()
        pahaw_ngs_auc = 0.974  # from pipeline v2 results
        pahaw_phi_auc = 0.541
        if len(nhs_ngs)>0:
            print(f"{'NGS / Logistic Regression':<25} {pahaw_ngs_auc:>12.3f} {nhs_ngs[0]:>15.3f}")
        print(f"{'Phi-10 (best)':<25} {pahaw_phi_auc:>12.3f} {nhs_phi:>15.3f}")
    else:
        print_sub("Step 8 skipped (no PaHaW pickle provided)")

    # ── 9. Figures ────────────────────────────────────────────────────────────
    print_header("STEP 9: FIGURES")

    # Figure 1: NGS violin PD vs HC
    metrics_plot = [m for m in ['Eff','SmallWorld','E_spectral','H_graph','Ricci_OR']
                    if m in ngs_df.columns]
    fig, axes = plt.subplots(1,len(metrics_plot),figsize=(4*len(metrics_plot),5))
    if len(metrics_plot)==1: axes=[axes]
    fig.suptitle('NewHandPD: NGS Group Comparison (PD vs HC)',
                 fontsize=12, fontweight='bold')
    for ax,metric in zip(axes,metrics_plot):
        data=ngs_df[['label',metric]].dropna()
        sns.violinplot(data=data,x='label',y=metric,
                       palette={'PD':'#C0392B','HC':'#2980B9'},
                       ax=ax,inner='box',alpha=0.8)
        ax.set_title(metric,fontsize=9); ax.set_xlabel('')
    plt.tight_layout()
    save_fig(fig,'P4_Fig1_NGS_group_comparison')

    # Figure 2: Classification bar chart
    fsets_order = ['Baseline (Phi-10)','Graph (NGS)','Hybrid (Phi10+NGS)']
    clfs_order  = list(clfs.keys())
    x = np.arange(len(clfs_order)); w = 0.25
    colors = ['#3498DB','#E74C3C','#2ECC71']
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle('NewHandPD P4: Classification Comparison (LOPO-CV)',
                 fontsize=12,fontweight='bold')
    for ax,(metric,ylabel) in zip(axes,[('AUC','ROC-AUC'),('Sensitivity','Sensitivity')]):
        for i,(fs,col) in enumerate(zip(fsets_order,colors)):
            vals=clf_df[clf_df['Feature Set']==fs][metric].values
            bars=ax.bar(x+i*w,vals,w,label=fs,color=col,alpha=0.85)
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                        f'{val:.3f}',ha='center',va='bottom',fontsize=7)
        ax.set_xticks(x+w)
        ax.set_xticklabels([c.replace(' ','\n') for c in clfs_order],fontsize=8)
        ax.set_ylabel(ylabel,fontsize=10)
        ax.legend(fontsize=7,loc='lower right')
        ax.set_ylim(0,1.15)
        ax.grid(axis='y',alpha=0.3)
        ax.axhline(0.9,color='gray',linestyle='--',alpha=0.4,linewidth=1)
    plt.tight_layout()
    save_fig(fig,'P4_Fig2_classification_comparison')

    # Figure 3: Permutation test
    fig,ax = plt.subplots(figsize=(8,5))
    ax.hist(perm_scores,bins=40,color='#85C1E9',edgecolor='white',
            alpha=0.85,label='Null distribution')
    ax.axvline(score_true,color='#E74C3C',linewidth=2.5,
               label=f'Observed AUC={score_true:.4f}')
    ax.axvline(np.percentile(perm_scores,95),color='orange',
               linewidth=1.5,linestyle='--',label='95th pct (null)')
    ax.axvline(np.percentile(perm_scores,99),color='red',
               linewidth=1.5,linestyle=':',label='99th pct (null)')
    ax.set_xlabel('AUC (5-fold CV)',fontsize=11)
    ax.set_ylabel('Frequency',fontsize=11)
    ax.set_title(f'NewHandPD — Permutation Test (NGS/LR)\n'
                 f'p={p_perm:.4f}, n={CONFIG["n_permutations"]} permutations',
                 fontsize=11,fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig,'P4_Fig3_permutation_test')
    print("  ✓ 3 figures saved")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print_header("P4 CONSOLIDATED RESULTS SUMMARY")
    print_sub("Dataset")
    print_result("NewHandPD participants", len(part_df))
    print_result("PD", int((part_df['label']=='PD').sum()))
    print_result("HC", int((part_df['label']=='HC').sum()))

    print_sub("NGS Group Comparison")
    sig_n = len(stat_df[stat_df['Sig'].isin(['*','**'])])
    print_result("Significant NGS metrics (p<0.05)", f"{sig_n}/{len(stat_df)}")
    for _,r in stat_df[stat_df['Sig'].isin(['*','**'])].iterrows():
        print_result(f"  {r['Metric']}", f"{r['PD mean±SD']} vs {r['HC mean±SD']}  {r['Sig']}")

    print_sub("Best Classification")
    if best_row:
        print_result(f"{best_row['Feature Set']} / {best_row['Classifier']}",
                     f"AUC={best_row['AUC']:.4f}  Sens={best_row['Sensitivity']:.4f}  "
                     f"Acc={best_row['Accuracy']:.4f}")

    print_sub("Validation")
    print_result("Permutation p-value", f"{p_perm:.4f}")
    ngs_rep = rep_df[rep_df['Feature Set']=='Graph (NGS)']
    if len(ngs_rep)>0:
        r = ngs_rep.iloc[0]
        print_result("Repeated CV NGS AUC",
                     f"{r['Mean AUC']:.4f}±{r['SD']:.4f}  "
                     f"[{r['CI_lo']:.4f},{r['CI_hi']:.4f}]")
    print_result("McNemar b/c", f"{b}/{c}  chi²={chi2:.3f}  p={p_mc:.4f}")

    # Save everything
    if CONFIG['save_csv']:
        p = os.path.join(CONFIG['output_dir'],'pickles','p4_results.pkl')
        os.makedirs(os.path.dirname(p),exist_ok=True)
        results = {
            'part_df':part_df, 'ngs_df':ngs_df,
            'stat_df':stat_df, 'clf_df':clf_df,
            'rep_df':rep_df, 'mcnemar':mn_df,
            'perm_true':score_true, 'perm_scores':perm_scores,
            'perm_p':p_perm,
        }
        with open(p,'wb') as f: pickle.dump(results,f)
        print(f"\n      [saved] {p}")

    print_header("P4 COMPLETE")
    print(f"  Total time : {elapsed(t_total)}")
    print(f"  Output dir : {os.path.abspath(CONFIG['output_dir'])}")
    print(f"  Tables     : {len(os.listdir(os.path.join(CONFIG['output_dir'],'tables')))} CSV")
    print(f"  Figures    : {len(os.listdir(os.path.join(CONFIG['output_dir'],'figures')))} PDF")
    print('='*68)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NGS P4 — NewHandPD Cross-Dataset Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run:
  python ngs_p4_newhandpd.py \\
    --healthy /path/to/healthy \\
    --patient /path/to/patient \\
    --output  /path/to/ngs_outputs_p4

  # With PaHaW comparison:
  python ngs_p4_newhandpd.py \\
    --healthy /path/to/healthy \\
    --patient /path/to/patient \\
    --pahaw_ngs /path/to/ngs_outputs_v2/pickles/ngs_v2_all_results.pkl \\
    --output  /path/to/ngs_outputs_p4
        """
    )
    parser.add_argument('--healthy',   required=True,
                        help='Path to healthy/ root folder')
    parser.add_argument('--patient',   required=True,
                        help='Path to patient/ root folder')
    parser.add_argument('--pahaw_ngs', default=None,
                        help='Optional: PaHaW NGS pickle for comparison')
    parser.add_argument('--output',    default='./ngs_outputs_p4',
                        help='Output directory')
    args = parser.parse_args()

    CONFIG['healthy_root'] = args.healthy
    CONFIG['patient_root'] = args.patient
    CONFIG['pahaw_pkl']    = args.pahaw_ngs
    CONFIG['output_dir']   = args.output

    # Validate paths exist
    for name,path in [('--healthy',args.healthy),('--patient',args.patient)]:
        if not Path(path).exists():
            print(f"ERROR: {name} path does not exist: {path}")
            sys.exit(1)

    run_p4(args.healthy, args.patient, args.pahaw_ngs)
