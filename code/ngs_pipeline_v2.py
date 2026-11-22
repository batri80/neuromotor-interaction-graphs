"""
NGS EXPERIMENTAL PIPELINE v2.0
================================
Neuromotor Interaction Graphs and Graph Signatures for Explainable
Assessment of Parkinsonian Handwriting Dynamics

Priority-ordered experimental tracks:

  P1 : Validation          Permutation test (1000 shuffles) + Repeated 5×CV
                           (10 repeats) + McNemar test (NGS vs Phi-10)
  P2 : Feature Importance  Sequential forward selection + LR coefficient
                           analysis + single-metric AUC ranking + SHAP
  P3 : Temporal Quartiles  TRUE within-task Q1→Q4 segmentation from SVC
                           time series (Corollary 1 first real test)
  P4 : NewHandPD           Cross-dataset validation (if images available)
  P5 : IS Recalibration    Recompute z-score from PaHaW HC; fix severity
  P6 : NGS Severity        Graph topology (SmallWorld, E_spectral) vs H&Y

Dataset : PaHaW (37 PD, 38 HC; SVC kinematic recordings; H&Y staging)
          NewHandPD (31 PD, 35 HC; images; optional — P4)

Usage   : python ngs_pipeline_v2.py
          --pahaw  /path/to/PaHaW/PaHaW_public
          --corpus /path/to/PaHaW/PaHaW_files/corpus_PaHaW.xlsx
          --newhandpd /path/to/NewHandPD   [optional, for P4]
          --output /path/to/output_dir
          --skip P3 P4                     [skip slow tracks if needed]

Paper   : Neuromotor Interaction Graphs: Inferring Functional Motor Control
          Networks from Handwriting Dynamics for Explainable Parkinson's
          Disease Assessment
Target  : Information Sciences — Special Issue: Graph-based Solutions for AI
Author  : Krishnan Batri
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, re, sys, time, warnings, pickle, argparse, random
from pathlib import Path
from collections import Counter
from itertools import combinations

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
                                     RepeatedStratifiedKFold, permutation_test_score)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths (overridden by CLI)
    'pahaw_root'         : './PaHaW/PaHaW_public',
    'corpus_path'        : './PaHaW/PaHaW_files/corpus_PaHaW.xlsx',
    'newhandpd_root'     : None,
    'output_dir'         : './ngs_outputs_v2',

    # Image rendering
    'image_size'         : (512, 512),
    'equilibrium_ref'    : 128,
    'stroke_width'       : 3,
    'pressure_modulate'  : True,

    # Features
    'feature_names'      : ['Eb','Eg','Em','sigma_d','Ag','rho_s',
                            'Hs','Hg','SIA','Omega'],

    # NIG
    'te_bins'            : 8,
    'te_lag'             : 1,
    'te_threshold_pct'   : 25,

    # NGS
    'compute_ricci'      : True,
    'ricci_alpha'        : 0.5,

    # P1 Validation
    'n_permutations'     : 1000,
    'n_cv_repeats'       : 10,
    'n_cv_folds'         : 5,
    'bootstrap_n'        : 1000,

    # P3 Temporal
    'n_quartiles'        : 4,
    'pahaw_spiral_task'  : 1,

    # P6 Severity
    'hy_rho_target'      : 0.35,

    # Output
    'save_pkl'           : True,
    'save_csv'           : True,
    'save_figures'       : True,
    'figure_dpi'         : 300,
    'figure_format'      : 'pdf',
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
def print_header(title, width=72):
    print('\n'+'='*width+f'\n  {title}\n'+'='*width)

def print_sub(title):
    print(f'\n  ── {title} ──')

def print_result(label, value, indent=6):
    print(f"{' '*indent}{label:<44s} {value}")

def elapsed(t0):
    return f"{time.time()-t0:.1f}s"

def make_dirs():
    for sd in ['tables','figures','pickles','rendered']:
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
# SVC READING & IMAGE RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def read_svc(path):
    """Read PaHaW SVC file → np.ndarray (n, 7).
    Columns: Y, X, timestamp, button_state, azimuth, altitude, pressure"""
    rows = []
    with open(path, 'r') as f:
        f.readline()  # skip count line
        for line in f:
            v = line.strip().split()
            if v:
                rows.append([float(x) for x in v])
    return np.array(rows) if rows else np.zeros((0, 7))

def render_to_image(svc_data, size=(512,512)):
    """
    Render SVC kinematic data to greyscale handwriting image.
    Uses on-surface strokes (button_state=1) with pressure-modulated width.
    Returns np.ndarray (H, W) uint8.
    """
    if len(svc_data) == 0:
        return np.full(size, 255, dtype=np.uint8)

    mask = svc_data[:, 3] == 1
    data = svc_data[mask] if mask.sum() > 1 else svc_data
    if len(data) < 2:
        return np.full(size, 255, dtype=np.uint8)

    x, y = data[:, 1], data[:, 0]
    pressure = data[:, 6]
    pad = 20
    W, H = size

    x_r = x.max()-x.min() if x.max()>x.min() else 1
    y_r = y.max()-y.min() if y.max()>y.min() else 1
    xn = ((x-x.min())/x_r*(W-2*pad)+pad).astype(int).clip(0, W-1)
    yn = ((y-y.min())/y_r*(H-2*pad)+pad).astype(int).clip(0, H-1)
    p_r = pressure.max()-pressure.min() if pressure.max()>pressure.min() else 1
    pn  = (pressure-pressure.min())/p_r

    img  = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    sw   = CONFIG['stroke_width']
    for i in range(len(xn)-1):
        w = max(1, int(sw*(0.5+pn[i]))) if CONFIG['pressure_modulate'] else sw
        draw.line([(int(xn[i]),int(yn[i])),(int(xn[i+1]),int(yn[i+1]))],
                  fill=0, width=w)
    return np.array(img, dtype=np.uint8)

def render_svc_quartile(svc_data, q_idx, n_q=4, size=(512,512)):
    """
    Render one temporal quartile of an SVC recording.
    q_idx: 0-based quartile index (0=Q1, 1=Q2, 2=Q3, 3=Q4)
    Splits by time index (not timestamp) for equal sample counts.
    """
    n = len(svc_data)
    q_size = n // n_q
    start  = q_idx * q_size
    end    = start + q_size if q_idx < n_q-1 else n
    return render_to_image(svc_data[start:end], size=size)

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
    Extract Phi-10 neurodynamic feature vector.
    Eb, Eg, Em, sigma_d, Ag, rho_s, Hs, Hg, SIA, Omega
    Matches Batri et al. (2026) framework exactly.
    """
    r = CONFIG['equilibrium_ref']
    I = _bg_norm(img_array.astype(float) if img_array.ndim==2
                 else 0.299*img_array[:,:,0]+0.587*img_array[:,:,1]+0.114*img_array[:,:,2])
    dev2   = (I-r)**2
    gx,gy,gm = _sobel(I)
    pyr    = _pyramid(I)
    w      = [0.5, 0.3, 0.2]

    Eb = np.sqrt(np.mean(dev2))
    Eg = np.mean(gm**2)
    Em = sum(wi*np.sqrt(np.mean((lev-r)**2)) for lev,wi in zip(pyr,w))
    sigma_d = np.std(dev2)
    Ag = np.var(gx**2)/(np.var(gy**2)+1e-8)

    d0 = (pyr[0]-r)**2
    d1 = np.array(Image.fromarray(((pyr[1]-r)**2).astype(np.float32))
                  .resize((d0.shape[1],d0.shape[0]),Image.LANCZOS), dtype=float)
    rc = np.corrcoef(d0.flatten(), d1.flatten())[0,1]
    rho_s = rc if not np.isnan(rc) else 0.0

    h,_ = np.histogram(I.flatten(), bins=256, range=(0,255))
    hn  = h/(h.sum()+1e-10)
    Hs  = -np.sum(hn*np.log(hn+1e-10))

    hg,_ = np.histogram(gm.flatten(), bins=256)
    hgn  = hg/(hg.sum()+1e-10)
    Hg   = -np.sum(hgn*np.log(hgn+1e-10))

    g1r = np.array(Image.fromarray(pyr[1].astype(np.float32))
                   .resize((I.shape[1],I.shape[0]),Image.LANCZOS), dtype=float)
    _,_,gm1 = _sobel(g1r)
    SIA = np.mean(np.abs(gm-gm1))/(np.mean(gm)+1e-6)

    F = np.fft.fftshift(np.fft.fft2(I))
    pw = np.abs(F)**2
    cy,cx = I.shape[0]//2, I.shape[1]//2
    fy,fx = np.ogrid[:I.shape[0], :I.shape[1]]
    dist  = np.sqrt((fy-cy)**2+(fx-cx)**2)
    Omega = np.sum(pw[(dist>=15)&(dist<=45)])

    return {'Eb':Eb,'Eg':Eg,'Em':Em,'sigma_d':sigma_d,
            'Ag':Ag,'rho_s':rho_s,'Hs':Hs,'Hg':Hg,
            'SIA':SIA,'Omega':Omega}

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER ENTROPY & NIG CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def _disc(x, n=8):
    bins = np.unique(np.percentile(x, np.linspace(0,100,n+1)))
    return np.digitize(x, bins[:-1])-1

def transfer_entropy(x, y, lag=1):
    """TE(X→Y) via equal-frequency binning across population."""
    x,y = np.asarray(x,float), np.asarray(y,float)
    if len(x)<lag+2: return 0.0
    n  = CONFIG['te_bins']
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
        p = cnt/N; py=jyy[(a,b)]/N; px=jyx1[(b,c)]/N; pm=my1[b]/N
        if all(v>0 for v in [p,py,px,pm]):
            r = (p*pm)/(py*px)
            if r>0: te += p*np.log(r)
    return max(te, 0.0)

def compute_te_matrix(X):
    """Full n_feat×n_feat directed TE matrix from (n_participants, n_feat)."""
    n = X.shape[1]
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: M[i,j] = transfer_entropy(X[:,i], X[:,j])
    return M

def build_nig(X, label=''):
    """Build Neuromotor Interaction Graph (NIG) from feature matrix."""
    feat = CONFIG['feature_names']
    M    = compute_te_matrix(X)
    nz   = M[M>0]
    thr  = np.percentile(nz, CONFIG['te_threshold_pct']) if len(nz)>0 else 0.0
    G    = nx.DiGraph(label=label)
    for i,f in enumerate(feat):
        G.add_node(i, name=f, display=FEAT_LABELS.get(f,f))
    for i in range(len(feat)):
        for j in range(len(feat)):
            if i!=j and M[i,j]>thr:
                G.add_edge(i,j, weight=M[i,j],
                           source=feat[i], target=feat[j])
    G.graph['te_matrix'] = M
    G.graph['threshold'] = thr
    return G

# ─────────────────────────────────────────────────────────────────────────────
# NGS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def graph_entropy(G):
    w = np.array([d['weight'] for _,_,d in G.edges(data=True)])
    if len(w)==0: return 0.0
    wn = w/(w.sum()+1e-10)
    return -np.sum(wn*np.log(wn+1e-10))

def global_efficiency(G):
    U = G.to_undirected()
    n = U.number_of_nodes()
    if n<2: return 0.0
    for u,v,d in U.edges(data=True):
        d['inv_w'] = 1.0/(d['weight']+1e-10)
    tot = 0.0
    for src in U.nodes():
        try:
            L = nx.single_source_dijkstra_path_length(U,src,weight='inv_w')
            tot += sum(1.0/d for t,d in L.items() if t!=src and d>0)
        except: pass
    return tot/(n*(n-1)) if tot>0 else 0.0

def graph_modularity(G):
    U = G.to_undirected()
    if U.number_of_edges()==0: return 0.0
    try:
        C = nx.community.greedy_modularity_communities(U, weight='weight')
        return nx.community.modularity(U, C, weight='weight')
    except: return 0.0

def algebraic_connectivity(G):
    U = G.to_undirected()
    if U.number_of_nodes()<2: return 0.0
    try: return nx.algebraic_connectivity(U, weight='weight')
    except: return 0.0

SW_SEED = 0   # seed for the Erdos-Renyi null graphs of the small-world coefficient

def small_world(G, n_rand=5, seed=SW_SEED):
    U = G.to_undirected()
    n,m = U.number_of_nodes(), U.number_of_edges()
    if m==0 or n<3: return 0.0
    try:
        C = nx.average_clustering(U, weight='weight')
        cc = max(nx.connected_components(U), key=len)
        L  = nx.average_shortest_path_length(U.subgraph(cc)) if len(cc)>1 else 1.0
        Cr,Lr = [],[]
        rng = random.Random(seed)   # fixed seed: reproducible across runs
        for _ in range(n_rand):
            Gr = nx.gnm_random_graph(n,m, seed=rng)
            Cr.append(nx.average_clustering(Gr))
            try:
                gcc = max(nx.connected_components(Gr), key=len)
                Lr.append(nx.average_shortest_path_length(Gr.subgraph(gcc))
                          if len(gcc)>1 else L)
            except: Lr.append(L)
        return (C/(np.mean(Cr)+1e-10)) / (L/(np.mean(Lr)+1e-10))
    except: return 0.0

def spectral_energy(G):
    A  = nx.to_numpy_array(G, weight='weight')
    ev = np.linalg.eigvalsh(A+A.T)
    return float(np.sum(ev**2))

def ollivier_ricci(G):
    alpha = CONFIG['ricci_alpha']
    U = G.to_undirected()
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
    d = {
        'H_graph'   : graph_entropy(G),
        'Eff'       : global_efficiency(G),
        'Modularity': graph_modularity(G),
        'Lambda2'   : algebraic_connectivity(G),
        'SmallWorld': small_world(G),
        'E_spectral': spectral_energy(G),
        'n_edges'   : G.number_of_edges(),
        'density'   : nx.density(G),
    }
    if CONFIG['compute_ricci']:
        d['Ricci_OR'] = ollivier_ricci(G)
    return d

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def get_classifiers():
    return {
        'SVM (RBF)'          : SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN (k=7)'          : KNeighborsClassifier(n_neighbors=7),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

def lopo_cv(X, y, clf):
    """Leave-One-Participant-Out CV. Returns (y_true, y_pred, y_prob)."""
    yt,yp,ypr = [],[],[]
    for tr,te in LeaveOneOut().split(X):
        sc = StandardScaler()
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        yt.append(y[te[0]])
        yp.append(clf.predict(sc.transform(X[te]))[0])
        try:    ypr.append(clf.predict_proba(sc.transform(X[te]))[0,1])
        except: ypr.append(float(yp[-1]))
    return np.array(yt),np.array(yp),np.array(ypr)

def clf_metrics(yt, yp, ypr):
    cm = confusion_matrix(yt,yp)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    return {
        'Accuracy'   : accuracy_score(yt,yp),
        'Sensitivity': tp/(tp+fn+1e-10),
        'Specificity': tn/(tn+fp+1e-10),
        'F1'         : f1_score(yt,yp,zero_division=0),
        'AUC'        : roc_auc_score(yt,ypr) if len(np.unique(yt))>1 else 0.5,
        'TP':tp,'TN':tn,'FP':fp,'FN':fn
    }

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_pahaw(pahaw_root, corpus_path):
    """
    Load PaHaW SVC files (spiral task), render to images,
    extract Phi-10 features, merge with corpus metadata.
    Returns participant-level DataFrame.
    """
    print_sub("Loading PaHaW corpus metadata")
    corpus = pd.read_excel(corpus_path)
    corpus['ID_str'] = corpus['ID'].apply(lambda x: f"{int(x):05d}")
    pd_corp = corpus[corpus['Disease']=='PD'].copy()
    hc_corp = corpus[corpus['Disease']=='H'].copy()
    print_result("PD participants", len(pd_corp))
    print_result("HC participants", len(hc_corp))
    print_result("H&Y distribution",
                 str(pd_corp['UPDRS V'].value_counts().sort_index().to_dict()))

    print_sub(f"Rendering Task {CONFIG['pahaw_spiral_task']} (spiral) SVC → images")
    root = Path(pahaw_root)
    task_str = f"__{CONFIG['pahaw_spiral_task']}_1.svc"
    rows = []

    for pid_dir in sorted(root.iterdir()):
        if not pid_dir.is_dir(): continue
        pid   = pid_dir.name
        svc_f = pid_dir / f"{pid}{task_str}"
        if not svc_f.exists(): continue

        svc_data = read_svc(svc_f)
        img      = render_to_image(svc_data)
        feats    = extract_phi10(img)
        feats['participant_id'] = pid
        feats['svc_data']       = svc_data  # keep for P3

        if pid in pd_corp['ID_str'].values:
            row = pd_corp[pd_corp['ID_str']==pid].iloc[0]
            feats.update({'label':'PD','label_int':1,
                          'HY':row['UPDRS V'],'age':row['Age'],
                          'LED':row['LED'],'disease_dur':row['Length of PD']})
        elif pid in hc_corp['ID_str'].values:
            row = hc_corp[hc_corp['ID_str']==pid].iloc[0]
            feats.update({'label':'HC','label_int':0,
                          'HY':np.nan,'age':row['Age'],
                          'LED':np.nan,'disease_dur':np.nan})
        else:
            continue
        rows.append(feats)

    df = pd.DataFrame(rows).reset_index(drop=True)
    print_result("Participants loaded", len(df))
    print_result("PD", int((df['label']=='PD').sum()))
    print_result("HC", int((df['label']=='HC').sum()))
    return df

def build_subject_ngs(part_df):
    """
    LEGACY - DO NOT USE FOR INFERENCE (see manuscript Section 5.10).

    Build subject-level NGS vectors using group-context NIG construction.
    Each participant's NIG is built from their group's feature matrix
    (population context). This construction uses the group label of every
    participant, including held-out ones, and deflates within-group variance;
    the per-subject group comparisons and classification it produced in the
    first submission are invalid. It is kept only so that the leakage audit
    of Section 5.10 can be reproduced. The valid individual-level pipeline is
    the within-subject formulation (test_kinematic_within_subject.py,
    r2_common.py).
    """
    warnings.warn("build_subject_ngs is the legacy leaky construction "
                  "(manuscript Section 5.10); use it only to reproduce the "
                  "leakage audit.", stacklevel=2)
    feat_cols = CONFIG['feature_names']
    X = part_df[feat_cols].values
    pd_mask = part_df['label'].values == 'PD'
    hc_mask = ~pd_mask
    rows = []

    for idx, row in part_df.iterrows():
        grp = pd_mask if row['label']=='PD' else hc_mask
        grp_mat = X[grp]
        sub_mat  = np.vstack([grp_mat, X[idx:idx+1]])
        sub_z    = StandardScaler().fit_transform(sub_mat)
        G_sub    = build_nig(sub_z, label=row['label'])
        ngs      = extract_ngs(G_sub)
        ngs['participant_id'] = row['participant_id']
        ngs['label']          = row['label']
        ngs['label_int']      = row['label_int']
        ngs['HY']             = row.get('HY', np.nan)
        ngs['age']            = row.get('age', np.nan)
        ngs['LED']            = row.get('LED', np.nan)
        rows.append(ngs)

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# P1 — VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def run_P1(part_df, ngs_df, all_results):
    """
    P1: Validate AUC=0.966 with three complementary methods.

    V1: Permutation test (1000 shuffles, p-value for NGS/LR AUC)
    V2: Repeated 5-fold stratified CV (10 repeats) — mean AUC ± SD
    V3: McNemar test — NGS vs Phi-10 significance
    """
    print_header("P1: VALIDATION — PERMUTATION + REPEATED CV + McNEMAR")
    t0 = time.time()

    feat_cols = CONFIG['feature_names']
    ngs_cols  = [c for c in ngs_df.columns
                 if c not in ['participant_id','label','label_int',
                              'HY','age','LED','n_edges','density']]
    merged = part_df.merge(ngs_df[['participant_id']+ngs_cols],
                           on='participant_id', how='left').dropna(subset=ngs_cols)
    y = merged['label_int'].values

    # Best classifier from v1: NGS / Logistic Regression
    clf_best = LogisticRegression(max_iter=1000, random_state=42)
    clf_base = LogisticRegression(max_iter=1000, random_state=42)

    X_ngs  = merged[ngs_cols].values
    X_phi  = merged[feat_cols].values
    sc_ngs = StandardScaler()
    sc_phi = StandardScaler()

    # ── V1: Permutation Test ─────────────────────────────────────────────────
    print_sub("V1: Permutation Test (1000 shuffles — NGS/LR)")
    cv_perm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    score_true, perm_scores, p_val = permutation_test_score(
        clf_best, sc_ngs.fit_transform(X_ngs), y,
        scoring='roc_auc', cv=cv_perm,
        n_permutations=CONFIG['n_permutations'],
        random_state=42, n_jobs=-1
    )

    print_result("True AUC (5-fold CV)",    f"{score_true:.4f}")
    print_result("Permutation AUC mean",    f"{perm_scores.mean():.4f} ± {perm_scores.std():.4f}")
    print_result("p-value (permutation)",   f"{p_val:.4f}")
    print_result("Significance",
                 "** p<0.001" if p_val<0.001 else
                 ("* p<0.05"  if p_val<0.05  else "ns"))
    print_result("AUC > 99th pct of null",
                 f"{'YES' if score_true>np.percentile(perm_scores,99) else 'NO'} "
                 f"(99th pct = {np.percentile(perm_scores,99):.4f})")

    # ── V2: Repeated Stratified K-Fold ───────────────────────────────────────
    print_sub(f"V2: Repeated {CONFIG['n_cv_folds']}-fold CV "
              f"({CONFIG['n_cv_repeats']} repeats) — all feature sets")

    rskf = RepeatedStratifiedKFold(
        n_splits=CONFIG['n_cv_folds'],
        n_repeats=CONFIG['n_cv_repeats'],
        random_state=42
    )

    rep_rows = []
    for fs_name, X_fs in [('Phi-10', X_phi), ('NGS', X_ngs),
                           ('Hybrid', np.hstack([X_phi, X_ngs]))]:
        aucs = []
        for tr, te in rskf.split(X_fs, y):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(sc.fit_transform(X_fs[tr]), y[tr])
            try:
                pr = clf.predict_proba(sc.transform(X_fs[te]))[:,1]
                aucs.append(roc_auc_score(y[te], pr))
            except: pass

        mu,lo,hi = bootstrap_ci(np.array(aucs), n=CONFIG['bootstrap_n'])
        print_result(f"  {fs_name}",
                     f"AUC = {mu:.4f} ± {np.std(aucs):.4f}  "
                     f"95%CI [{lo:.4f},{hi:.4f}]")
        rep_rows.append({'Feature Set':fs_name, 'Mean AUC':mu,
                         'SD':np.std(aucs), 'CI_lo':lo, 'CI_hi':hi,
                         'n_folds':len(aucs)})

    rep_df = pd.DataFrame(rep_rows)
    save_table(rep_df, 'P1_V2_repeated_cv')

    # ── V3: McNemar Test ─────────────────────────────────────────────────────
    print_sub("V3: McNemar Test — NGS vs Phi-10 (LOPO-CV)")
    yt_ngs, yp_ngs, _ = lopo_cv(X_ngs, y, clf_best)
    yt_phi, yp_phi, _ = lopo_cv(X_phi, y, clf_base)

    # McNemar contingency table
    b = np.sum((yp_ngs==yt_ngs) & (yp_phi!=yt_phi))  # NGS correct, Phi wrong
    c = np.sum((yp_ngs!=yt_ngs) & (yp_phi==yt_phi))  # NGS wrong, Phi correct
    # McNemar statistic with continuity correction
    if b+c > 0:
        chi2 = (abs(b-c)-1)**2 / (b+c)
        p_mc = stats.chi2.sf(chi2, df=1)
    else:
        chi2, p_mc = 0.0, 1.0

    print_result("NGS correct, Phi-10 wrong (b)", str(b))
    print_result("NGS wrong, Phi-10 correct (c)", str(c))
    print_result("McNemar chi2", f"{chi2:.4f}")
    print_result("p-value", f"{p_mc:.4f}")
    print_result("NGS significantly better?",
                 "YES **" if p_mc<0.01 else ("YES *" if p_mc<0.05 else "NO"))

    mn_df = pd.DataFrame([{'b':b,'c':c,'chi2':chi2,'p_value':p_mc}])
    save_table(mn_df, 'P1_V3_mcnemar')

    # Summary permutation figure
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(perm_scores, bins=40, color='#90CAF9', edgecolor='white',
            alpha=0.85, label='Null distribution (permuted)')
    ax.axvline(score_true, color='#d62728', linewidth=2.5,
               label=f'Observed AUC = {score_true:.4f}')
    ax.axvline(np.percentile(perm_scores,95), color='orange',
               linewidth=1.5, linestyle='--', label='95th percentile (null)')
    ax.axvline(np.percentile(perm_scores,99), color='red',
               linewidth=1.5, linestyle=':', label='99th percentile (null)')
    ax.set_xlabel('AUC (5-fold CV)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Permutation Test — NGS/Logistic Regression\n'
                 f'p = {p_val:.4f}, n = {CONFIG["n_permutations"]} permutations',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    save_fig(fig, 'P1_permutation_test')

    all_results['P1'] = {
        'perm_true':score_true, 'perm_scores':perm_scores,
        'perm_p':p_val, 'rep_df':rep_df, 'mcnemar':mn_df
    }
    print(f"\n  ✓ P1 complete [{elapsed(t0)}]")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# P2 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def run_P2(ngs_df, all_results):
    """
    P2: Identify which NGS metrics drive AUC=0.966.

    E1: Single-metric AUC ranking (each NGS metric alone)
    E2: Sequential Forward Selection (build up minimal sufficient set)
    E3: LR coefficient analysis (standardised weights)
    E4: SHAP values (feature contributions per participant)
    """
    print_header("P2: NGS FEATURE IMPORTANCE")
    t0 = time.time()

    ngs_cols = [c for c in NGS_CORE if c in ngs_df.columns]
    y = ngs_df['label_int'].values
    X = ngs_df[ngs_cols].values

    # ── E1: Single-metric AUC ranking ────────────────────────────────────────
    print_sub("E1: Single-metric AUC (LOPO-CV, LR)")
    e1_rows = []
    for i, col in enumerate(ngs_cols):
        xi = X[:, i:i+1]
        yt,_,ypr = lopo_cv(xi, y, LogisticRegression(max_iter=1000,random_state=42))
        auc = roc_auc_score(yt,ypr) if len(np.unique(yt))>1 else 0.5
        mw_u, mw_p = stats.mannwhitneyu(
            X[y==1,i], X[y==0,i], alternative='two-sided')
        sig = '**' if mw_p<0.01 else ('*' if mw_p<0.05 else 'ns')
        e1_rows.append({'NGS Metric':col,'Solo AUC':auc,
                        'MW p-value':mw_p,'Sig':sig})
        print_result(f"  {col}", f"AUC={auc:.4f}  p={mw_p:.4f} {sig}")

    e1_df = pd.DataFrame(e1_rows).sort_values('Solo AUC', ascending=False)
    save_table(e1_df, 'P2_E1_single_metric_auc')

    # ── E2: Sequential Forward Selection ─────────────────────────────────────
    print_sub("E2: Sequential Forward Selection (LR, 5-fold CV)")
    remaining = list(range(len(ngs_cols)))
    selected  = []
    sfs_rows  = []
    best_auc  = 0.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    while remaining:
        step_best_auc = 0.0
        step_best_idx = None
        for idx in remaining:
            candidate = selected + [idx]
            Xc = StandardScaler().fit_transform(X[:, candidate])
            aucs = []
            for tr,te in cv.split(Xc, y):
                clf = LogisticRegression(max_iter=1000,random_state=42)
                clf.fit(Xc[tr], y[tr])
                try:
                    pr = clf.predict_proba(Xc[te])[:,1]
                    aucs.append(roc_auc_score(y[te],pr))
                except: pass
            m = np.mean(aucs) if aucs else 0.0
            if m > step_best_auc:
                step_best_auc = m
                step_best_idx = idx
        if step_best_idx is None: break
        selected.append(step_best_idx)
        remaining.remove(step_best_idx)
        feat_added = ngs_cols[step_best_idx]
        gain = step_best_auc - best_auc
        best_auc = step_best_auc
        sfs_rows.append({'Step':len(selected), 'Feature Added':feat_added,
                         'Cumulative AUC':step_best_auc, 'Gain':gain})
        print_result(f"  Step {len(selected)}: +{feat_added}",
                     f"AUC={step_best_auc:.4f}  gain={gain:+.4f}")

    sfs_df = pd.DataFrame(sfs_rows)
    save_table(sfs_df, 'P2_E2_sequential_forward_selection')

    # ── E3: LR Coefficient Analysis ──────────────────────────────────────────
    print_sub("E3: Logistic Regression Standardised Coefficients")
    sc = StandardScaler()
    Xz = sc.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xz, y)
    coefs = lr.coef_[0]
    coef_df = pd.DataFrame({
        'NGS Metric' : ngs_cols,
        'Coefficient': coefs,
        'Abs_Coef'   : np.abs(coefs)
    }).sort_values('Abs_Coef', ascending=False)
    save_table(coef_df, 'P2_E3_lr_coefficients')
    for _,row in coef_df.iterrows():
        direction = '↑ PD' if row['Coefficient']>0 else '↓ PD'
        print_result(f"  {row['NGS Metric']}",
                     f"coef={row['Coefficient']:+.4f}  {direction}")

    # ── E4: Feature Importance (SHAP or |coef| fallback) ────────────────────
    print_sub("E4: Feature Importance (SHAP or |coefficient| fallback)")
    shap_df = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.LinearExplainer(
                lr, Xz, feature_perturbation='interventional')
            shap_vals = explainer.shap_values(Xz)
            mean_abs  = np.abs(shap_vals).mean(axis=0)
            shap_df   = pd.DataFrame({
                'NGS Metric' : ngs_cols,
                'Mean |SHAP|': mean_abs,
                'Source'     : 'SHAP'
            }).sort_values('Mean |SHAP|', ascending=False)
            print("      [SHAP] Using SHAP LinearExplainer")
        except Exception as e:
            print(f"      [warn] SHAP failed: {e}")
    if shap_df is None:
        shap_df = pd.DataFrame({
            'NGS Metric' : coef_df['NGS Metric'].values,
            'Mean |SHAP|': coef_df['Abs_Coef'].values,
            'Source'     : '|LR coefficient|'
        }).sort_values('Mean |SHAP|', ascending=False)
        print("      [fallback] Using |LR coefficient| as importance proxy")
        print("      [note] pip install shap==0.41.0  (for SHAP support)")
    save_table(shap_df, 'P2_E4_importance')
    for _, row in shap_df.iterrows():
        print_result(f"  {row['NGS Metric']}",
                     f"importance={row['Mean |SHAP|']:.5f}  [{row['Source']}]")

    # ── Figures ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('P2: NGS Feature Importance Analysis', fontsize=12, fontweight='bold')

    # E1: solo AUC bar
    ax = axes[0]
    cols_sorted = e1_df['NGS Metric'].tolist()
    aucs_sorted = e1_df['Solo AUC'].tolist()
    colors = ['#d62728' if a>0.6 else '#90CAF9' for a in aucs_sorted]
    ax.barh(cols_sorted[::-1], aucs_sorted[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Solo AUC (LOPO-CV)', fontsize=10)
    ax.set_title('E1: Single-Metric AUC', fontsize=10)
    for i,(c,a) in enumerate(zip(cols_sorted[::-1], aucs_sorted[::-1])):
        ax.text(a+0.005, i, f'{a:.3f}', va='center', fontsize=8)

    # E2: SFS trajectory
    ax = axes[1]
    ax.plot(sfs_df['Step'], sfs_df['Cumulative AUC'],
            'o-', color='#1f77b4', linewidth=2, markersize=8)
    for _,row in sfs_df.iterrows():
        ax.annotate(row['Feature Added'],
                    (row['Step'], row['Cumulative AUC']),
                    textcoords='offset points', xytext=(5,5), fontsize=7)
    ax.set_xlabel('Number of Features', fontsize=10)
    ax.set_ylabel('AUC (5-fold CV)', fontsize=10)
    ax.set_title('E2: Sequential Forward Selection', fontsize=10)
    ax.grid(True, alpha=0.3)

    # E3: LR coefficients
    ax = axes[2]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    bar_colors = ['#d62728' if coefs[i]>0 else '#1f77b4' for i in sorted_idx]
    ax.barh([ngs_cols[i] for i in sorted_idx[::-1]],
            [coefs[i] for i in sorted_idx[::-1]],
            color=bar_colors[::-1], alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardised LR Coefficient', fontsize=10)
    ax.set_title('E3: LR Coefficients\n(+ = higher in PD)', fontsize=10)

    plt.tight_layout()
    save_fig(fig, 'P2_feature_importance')

    all_results['P2'] = {
        'e1_df':e1_df, 'sfs_df':sfs_df,
        'coef_df':coef_df, 'shap_df':shap_df
    }
    print(f"\n  ✓ P2 complete [{elapsed(t0)}]")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# P3 — TRUE TEMPORAL QUARTILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_P3(part_df, all_results):
    """
    P3: True Corollary 1 test — within-task temporal quartile NIG evolution.

    Each participant's spiral SVC time series is split into Q1–Q4
    by sample index (equal temporal segments at 150 Hz).
    Each quartile is rendered as a separate image → Phi-10 extracted →
    Population NIG built per quartile per group → NGS tracked Q1→Q4.

    This is the first genuine test of temporal neuromotor graph degradation.
    """
    print_header("P3: TRUE TEMPORAL QUARTILE ANALYSIS (Corollary 1)")
    t0 = time.time()

    feat_cols = CONFIG['feature_names']
    n_q = CONFIG['n_quartiles']

    # Build per-participant per-quartile feature vectors
    print_sub("Rendering SVC quartiles Q1–Q4 per participant")
    q_feat_rows = []
    for _, row in part_df.iterrows():
        svc_data = row.get('svc_data')
        if svc_data is None or len(svc_data) < n_q*50:
            print(f"      [skip] {row['participant_id']}: insufficient samples")
            continue
        for qi in range(n_q):
            img  = render_svc_quartile(svc_data, qi, n_q=n_q)
            feat = extract_phi10(img)
            feat['participant_id'] = row['participant_id']
            feat['label']          = row['label']
            feat['label_int']      = row['label_int']
            feat['quartile']       = f'Q{qi+1}'
            feat['q_idx']          = qi + 1
            q_feat_rows.append(feat)

    q_df = pd.DataFrame(q_feat_rows)
    print_result("Total quartile feature rows", len(q_df))

    # Build population NIG per quartile per group
    print_sub("Building NIG per quartile (Q1–Q4) per group")
    q_ngs_pd, q_ngs_hc = {}, {}
    for qi in range(1, n_q+1):
        qname = f'Q{qi}'
        qd    = q_df[q_df['q_idx']==qi]
        pd_mat = qd[qd['label']=='PD'][feat_cols].values
        hc_mat = qd[qd['label']=='HC'][feat_cols].values

        if len(pd_mat) > 2:
            G_pd = build_nig(StandardScaler().fit_transform(pd_mat),
                             label=f'PD_{qname}')
            q_ngs_pd[qname] = extract_ngs(G_pd)
            q_ngs_pd[qname]['G'] = G_pd
        else:
            q_ngs_pd[qname] = {k:np.nan for k in NGS_CORE}

        if len(hc_mat) > 2:
            G_hc = build_nig(StandardScaler().fit_transform(hc_mat),
                             label=f'HC_{qname}')
            q_ngs_hc[qname] = extract_ngs(G_hc)
            q_ngs_hc[qname]['G'] = G_hc
        else:
            q_ngs_hc[qname] = {k:np.nan for k in NGS_CORE}

        print_result(f"  {qname} H(G) PD/HC",
                     f"{q_ngs_pd[qname].get('H_graph',np.nan):.4f} / "
                     f"{q_ngs_hc[qname].get('H_graph',np.nan):.4f}")
        print_result(f"  {qname} Eff PD/HC",
                     f"{q_ngs_pd[qname].get('Eff',np.nan):.4f} / "
                     f"{q_ngs_hc[qname].get('Eff',np.nan):.4f}")

    # Slope analysis — Corollary 1 test
    print_sub("Corollary 1: Slope Test (Q1→Q4)")
    quarters = ['Q1','Q2','Q3','Q4']

    def slope(vals):
        xy = [(i,v) for i,v in enumerate(vals) if not np.isnan(v)]
        if len(xy)<2: return np.nan
        xs,ys = zip(*xy)
        m,*_ = stats.linregress(xs,ys)
        return m

    slope_rows = []
    for metric in ['H_graph','Eff','SmallWorld','Ricci_OR']:
        pd_traj = [q_ngs_pd[q].get(metric,np.nan) for q in quarters]
        hc_traj = [q_ngs_hc[q].get(metric,np.nan) for q in quarters]
        s_pd = slope(pd_traj)
        s_hc = slope(hc_traj)
        expected_dir = 'PD>HC' if metric=='H_graph' else 'PD<HC'
        supported = (s_pd>s_hc) if metric=='H_graph' else (s_pd<s_hc)
        slope_rows.append({
            'Metric':metric, 'PD slope':s_pd, 'HC slope':s_hc,
            'Expected':expected_dir, 'Supported':supported
        })
        print_result(f"  {metric} slope PD/HC",
                     f"{s_pd:+.5f} / {s_hc:+.5f}  "
                     f"{'SUPPORTED ✓' if supported else 'NOT SUPPORTED'}"
                     f"  [expected {expected_dir}]")

    slope_df = pd.DataFrame(slope_rows)
    n_supported = slope_df['Supported'].sum()
    print_result("Corollary 1 overall",
                 f"{n_supported}/{len(slope_df)} metrics support hypothesis")

    # Build trajectory table
    traj_rows = []
    for q in quarters:
        for grp,qd in [('PD',q_ngs_pd),('HC',q_ngs_hc)]:
            traj_rows.append({'Quartile':q,'Group':grp,
                              **{k:qd[q].get(k,np.nan) for k in NGS_CORE}})
    traj_df = pd.DataFrame(traj_rows)
    save_table(traj_df, 'P3_temporal_quartile_trajectory')
    save_table(slope_df, 'P3_corollary1_slope_test')

    # Figure — temporal trajectory
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('P3: Temporal NIG Evolution — Within-Task Quartiles Q1→Q4\n'
                 'TRUE Corollary 1 Test (spiral SVC segmented by time)',
                 fontsize=11, fontweight='bold')

    metrics_plot = [
        ('H_graph',    'Graph Entropy H(G)',       'Expected ↑ PD'),
        ('Eff',        'Global Efficiency',         'Expected ↓ PD'),
        ('SmallWorld', 'Small-World Coefficient',   'Expected ↓ PD'),
        ('Ricci_OR',   'Ollivier-Ricci Curvature',  'Expected ↓ PD'),
    ]
    x = [1,2,3,4]
    for ax,(metric,ylabel,note) in zip(axes,metrics_plot):
        pd_vals=[q_ngs_pd[q].get(metric,np.nan) for q in quarters]
        hc_vals=[q_ngs_hc[q].get(metric,np.nan) for q in quarters]
        ax.plot(x,pd_vals,'o-',color='#d62728',linewidth=2,
                markersize=8,label='PD',zorder=3)
        ax.plot(x,hc_vals,'s--',color='#1f77b4',linewidth=2,
                markersize=8,label='HC',zorder=3)
        # trend lines
        for vals,col in [(pd_vals,'#d62728'),(hc_vals,'#1f77b4')]:
            xy=[(xi,v) for xi,v in zip(x,vals) if not np.isnan(v)]
            if len(xy)>=2:
                xs,ys=zip(*xy); m,b,*_=stats.linregress(xs,ys)
                ax.plot(x,[m*xi+b for xi in x],'--',color=col,alpha=0.35)
        ax.set_xticks(x); ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
        ax.set_xlabel('Quartile',fontsize=9)
        ax.set_ylabel(ylabel,fontsize=9)
        ax.set_title(f"{ylabel}\n{note}",fontsize=9)
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'P3_temporal_quartile_trajectory')

    all_results['P3'] = {
        'q_ngs_pd':q_ngs_pd, 'q_ngs_hc':q_ngs_hc,
        'traj_df':traj_df, 'slope_df':slope_df,
        'n_supported':n_supported
    }
    print(f"\n  ✓ P3 complete [{elapsed(t0)}]")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# P4 — NewHandPD CROSS-DATASET VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def run_P4(newhandpd_root, all_results):
    """
    LEGACY (first submission). Superseded by ngs_p4_newhandpd.py; uses the
    legacy build_subject_ngs construction and is not used for any result
    reported in revision 2.
    P4: Cross-dataset validation on NewHandPD (public, 66 participants).
    Applies the NIG/NGS pipeline independently to NewHandPD images.
    Compares classification AUC with PaHaW result.
    """
    print_header("P4: NewHandPD CROSS-DATASET VALIDATION")
    t0 = time.time()

    root = Path(newhandpd_root)
    pd_dir = root / 'PD'
    hc_dir = root / 'HC'

    if not pd_dir.exists() or not hc_dir.exists():
        print("      [skip] NewHandPD images not found.")
        print(f"      Place images in {pd_dir} and {hc_dir}")
        all_results['P4'] = {'status':'skipped'}
        return all_results

    feat_cols = CONFIG['feature_names']
    rows = []
    for label, folder, label_int in [('PD',pd_dir,1),('HC',hc_dir,0)]:
        imgs = sorted(list(folder.glob('*.jpg'))+
                      list(folder.glob('*.png'))+
                      list(folder.glob('*.JPG')))
        for img_path in imgs:
            # NewHandPD participants are identified by the -P<n>/-H<n> suffix
            # of the filename (corrected in revision 2; the first submission
            # split on '_' and mis-grouped images). Superseded by
            # ngs_p4_newhandpd.py, which is the loader used for all reported
            # NewHandPD results.
            m_id = re.search(r'-([PpHh])(\d+)$', img_path.stem)
            if m_id is None:
                raise ValueError(f"Unrecognised NewHandPD filename: {img_path.name}")
            pid = f"{label}_{int(m_id.group(2)):03d}"
            try:
                pil = Image.open(img_path).convert('RGB')
                pil = pil.resize(CONFIG['image_size'], Image.LANCZOS)
                feat = extract_phi10(np.array(pil))
                feat.update({'participant_id':pid,'label':label,'label_int':label_int})
                rows.append(feat)
            except Exception as e:
                print(f"      [warn] {img_path.name}: {e}")

    if not rows:
        print("      [skip] No images loaded from NewHandPD.")
        all_results['P4'] = {'status':'no_images'}
        return all_results

    nh_df = pd.DataFrame(rows)
    # Aggregate to participant level
    part_nh = nh_df.groupby(['participant_id','label','label_int'])[feat_cols]\
                   .mean().reset_index()
    print_result("NewHandPD participants", len(part_nh))
    print_result("PD", int((part_nh['label']=='PD').sum()))
    print_result("HC", int((part_nh['label']=='HC').sum()))

    # Build subject-level NGS
    ngs_nh = build_subject_ngs(part_nh)
    ngs_cols = [c for c in NGS_CORE if c in ngs_nh.columns]
    merged   = part_nh.merge(ngs_nh[['participant_id']+ngs_cols],
                             on='participant_id').dropna(subset=ngs_cols)
    y  = merged['label_int'].values
    X  = merged[ngs_cols].values
    Xp = merged[feat_cols].values

    # LOPO-CV comparison
    P4_rows = []
    for fs,Xf in [('Phi-10',Xp),('NGS',X),('Hybrid',np.hstack([Xp,X]))]:
        yt,yp,ypr = lopo_cv(Xf,y,LogisticRegression(max_iter=1000,random_state=42))
        m = clf_metrics(yt,yp,ypr)
        P4_rows.append({'Dataset':'NewHandPD','Feature Set':fs,**m})
        print_result(f"  {fs}",
                     f"AUC={m['AUC']:.4f}  Sens={m['Sensitivity']:.4f}  "
                     f"Acc={m['Accuracy']:.4f}")

    P4_df = pd.DataFrame(P4_rows)
    save_table(P4_df, 'P4_newhandpd_validation')

    all_results['P4'] = {'P4_df':P4_df, 'status':'complete'}
    print(f"\n  ✓ P4 complete [{elapsed(t0)}]")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# P5 — IS RECALIBRATION FOR PAHAW
# ─────────────────────────────────────────────────────────────────────────────
def run_P5(part_df, all_results):
    """
    P5: Recalibrate Instability Score using PaHaW HC statistics.

    The IS failed (rho=-0.048) in v1 because z-score parameters were
    imported from NewHandPD HC distributions. Here we compute mu/SD
    from PaHaW HC participants' rendered images — the correct reference.

    IS = mean(Eg+, SIA+, Hs+, Omega+) where x+ = (x-mu_HC)/sd_HC
    """
    print_header("P5: IS RECALIBRATION USING PAHAW HC STATISTICS")
    t0 = time.time()

    feat_cols = CONFIG['feature_names']
    hc_df = part_df[part_df['label']=='HC']
    pd_df = part_df[part_df['label']=='PD']

    # Compute HC training statistics from PaHaW HC participants
    IS_components = ['Eg','SIA','Hs','Omega']
    hc_mu = {c: hc_df[c].mean() for c in IS_components}
    hc_sd = {c: hc_df[c].std()  for c in IS_components}

    print_sub("PaHaW HC calibration statistics")
    for c in IS_components:
        print_result(f"  {c}  mu / SD",
                     f"{hc_mu[c]:.4f} / {hc_sd[c]:.4f}")

    # Compute IS for all PD participants
    def compute_IS(row):
        zs = [(row[c]-hc_mu[c])/(hc_sd[c]+1e-10) for c in IS_components]
        return float(np.mean(zs))

    pd_df = pd_df.copy()
    pd_df['IS'] = pd_df.apply(compute_IS, axis=1)
    pd_hy = pd_df.dropna(subset=['HY','IS'])

    print_sub("Recalibrated IS vs H&Y Stage")
    rho, p_val = stats.spearmanr(pd_hy['IS'], pd_hy['HY'])
    sig = '**' if p_val<0.01 else ('*' if p_val<0.05 else 'ns')
    print_result("Spearman rho(IS_recal, H&Y)",
                 f"rho={rho:.3f}  p={p_val:.4f}  {sig}")
    print_result("Target (rho>0.35, p<0.05)",
                 "MET ✓" if rho>CONFIG['hy_rho_target'] and p_val<0.05
                 else "NOT MET")

    # Partial correlations
    def partial_sp(x, y, cov):
        m,b,*_ = stats.linregress(cov, x)
        rx = x-(m*cov+b)
        m2,b2,*_ = stats.linregress(cov, y)
        ry = y-(m2*cov+b2)
        return stats.spearmanr(rx, ry)

    pd_age = pd_hy.dropna(subset=['age'])
    pd_led = pd_hy.dropna(subset=['LED'])
    if len(pd_age)>5:
        ra,pa = partial_sp(pd_age['IS'].values,pd_age['HY'].values,
                           pd_age['age'].values)
        print_result("rho | controlling age", f"{ra:.3f}  p={pa:.4f}")
    if len(pd_led)>5:
        rl,pl = partial_sp(pd_led['IS'].values,pd_led['HY'].values,
                           pd_led['LED'].values)
        print_result("rho | controlling LED", f"{rl:.3f}  p={pl:.4f}")

    print_sub("IS by H&Y stage")
    for stage in sorted(pd_hy['HY'].unique()):
        grp = pd_hy[pd_hy['HY']==stage]['IS']
        print_result(f"  H&Y {stage} (n={len(grp)})",
                     f"IS = {grp.mean():.3f} ± {grp.std():.3f}")

    sev_df = pd_hy[['participant_id','HY','IS','age','LED']].copy()
    save_table(sev_df, 'P5_IS_recalibrated_severity')

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'P5: Recalibrated IS vs H&Y — PaHaW PD (n={len(pd_hy)})\n'
                 f'ρ={rho:.3f}, p={p_val:.4f}',
                 fontsize=11, fontweight='bold')
    ax = axes[0]
    ax.scatter(pd_hy['IS'], pd_hy['HY'],
               c=pd_hy['HY'], cmap='YlOrRd', s=60, alpha=0.8, zorder=3)
    m,b,*_ = stats.linregress(pd_hy['IS'].values, pd_hy['HY'].values)
    xs = np.linspace(pd_hy['IS'].min(), pd_hy['IS'].max(), 100)
    ax.plot(xs, m*xs+b, '--', color='gray', alpha=0.7)
    ax.set_xlabel('IS (recalibrated)', fontsize=10)
    ax.set_ylabel('H&Y Stage', fontsize=10)
    ax.set_title(f'IS vs H&Y  (ρ={rho:.3f})', fontsize=10)

    ax = axes[1]
    stages = sorted(pd_hy['HY'].unique())
    bp = [pd_hy[pd_hy['HY']==s]['IS'].values for s in stages]
    ax.violinplot(bp, positions=range(len(stages)), showmedians=True)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([f"H&Y {s}\n(n={len(pd_hy[pd_hy['HY']==s])})"
                        for s in stages], fontsize=8)
    ax.set_ylabel('IS (recalibrated)', fontsize=10)
    ax.set_title('IS by H&Y Stage', fontsize=10)

    plt.tight_layout()
    save_fig(fig, 'P5_IS_recalibrated')

    all_results['P5'] = {'sev_df':sev_df,'rho':rho,'p_val':p_val}
    print(f"\n  ✓ P5 complete [{elapsed(t0)}]")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# P6 — NGS-BASED SEVERITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_P6(ngs_df, all_results):
    """
    P6: Graph topology as severity biomarker.
    Test whether individual NGS metrics correlate with H&Y stage.
    This is novel — no prior work uses graph topology for PD severity.
    """
    print_header("P6: NGS-BASED SEVERITY ANALYSIS (Graph Topology vs H&Y)")
    t0 = time.time()

    pd_ngs = ngs_df[ngs_df['label']=='PD'].dropna(subset=['HY'])
    print_result("PD participants with H&Y", len(pd_ngs))

    ngs_cols = [c for c in NGS_CORE if c in ngs_df.columns]
    P6_rows  = []

    print_sub("Spearman rho(NGS metric, H&Y) — uncorrected and Bonferroni")
    n_tests = len(ngs_cols)
    for metric in ngs_cols:
        vals = pd_ngs[metric].dropna()
        hy   = pd_ngs.loc[vals.index, 'HY']
        if len(vals) < 5:
            continue
        rho, p = stats.spearmanr(vals, hy)
        p_bonf = min(p*n_tests, 1.0)
        sig_raw  = '**' if p<0.01 else ('*' if p<0.05 else 'ns')
        sig_bonf = '**' if p_bonf<0.01 else ('*' if p_bonf<0.05 else 'ns')
        P6_rows.append({'NGS Metric':metric, 'rho':rho,
                        'p_raw':p, 'p_Bonferroni':p_bonf,
                        'Sig (raw)':sig_raw, 'Sig (Bonf)':sig_bonf})
        print_result(f"  {metric}",
                     f"rho={rho:+.3f}  p={p:.4f}{sig_raw}  "
                     f"p_Bonf={p_bonf:.4f}{sig_bonf}")

    P6_df = pd.DataFrame(P6_rows).sort_values('p_raw')
    save_table(P6_df, 'P6_NGS_severity_correlations')

    # Partial correlations for top metric
    best = P6_df.iloc[0]
    print_sub(f"Partial correlations for top metric: {best['NGS Metric']}")
    metric = best['NGS Metric']
    pd_age = pd_ngs.dropna(subset=['age', metric])
    pd_led = pd_ngs.dropna(subset=['LED', metric])

    def partial_sp(x, y, cov):
        m,b,*_ = stats.linregress(cov,x); rx=x-(m*cov+b)
        m2,b2,*_=stats.linregress(cov,y); ry=y-(m2*cov+b2)
        return stats.spearmanr(rx,ry)

    if len(pd_age)>5:
        ra,pa = partial_sp(pd_age[metric].values, pd_age['HY'].values,
                           pd_age['age'].values)
        print_result(f"  rho({metric}, H&Y) | age", f"{ra:+.3f}  p={pa:.4f}")
    if len(pd_led)>5:
        rl,pl = partial_sp(pd_led[metric].values, pd_led['HY'].values,
                           pd_led['LED'].values)
        print_result(f"  rho({metric}, H&Y) | LED", f"{rl:+.3f}  p={pl:.4f}")

    # Figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('P6: NGS Graph Topology vs H&Y Severity (PaHaW PD)',
                 fontsize=12, fontweight='bold')
    axes_flat = axes.flatten()

    for i, (ax, row) in enumerate(zip(axes_flat, P6_df.itertuples())):
        metric = row._1  # 'NGS Metric'
        vals   = pd_ngs[metric].dropna()
        hy_    = pd_ngs.loc[vals.index, 'HY']
        ax.scatter(vals, hy_, alpha=0.6, s=50,
                   c=hy_, cmap='YlOrRd')
        m_,b_,*_ = stats.linregress(vals.values, hy_.values)
        xs_ = np.linspace(vals.min(), vals.max(), 100)
        ax.plot(xs_, m_*xs_+b_, '--', color='gray', alpha=0.7)
        sig_ = row._6  # 'Sig (raw)'
        ax.set_title(f"{metric}\nρ={row.rho:.3f} p={row.p_raw:.3f}{sig_}",
                     fontsize=9)
        ax.set_xlabel(metric, fontsize=8)
        ax.set_ylabel('H&Y', fontsize=8)

    for ax in axes_flat[len(P6_df):]:
        ax.axis('off')

    plt.tight_layout()
    save_fig(fig, 'P6_NGS_severity')

    all_results['P6'] = {'P6_df':P6_df}
    print(f"\n  ✓ P6 complete [{elapsed(t0)}]")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDATED SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_results):
    print_header("CONSOLIDATED RESULTS SUMMARY FOR MANUSCRIPT")

    if 'P1' in all_results and all_results['P1']:
        r = all_results['P1']
        print_sub("P1: Validation")
        print_result("Permutation AUC (true)",    f"{r['perm_true']:.4f}")
        print_result("Permutation p-value",        f"{r['perm_p']:.4f}")
        rep = r['rep_df']
        for _,row in rep.iterrows():
            print_result(f"  Repeated CV — {row['Feature Set']}",
                         f"AUC={row['Mean AUC']:.4f}±{row['SD']:.4f} "
                         f"[{row['CI_lo']:.4f},{row['CI_hi']:.4f}]")
        mc = r['mcnemar'].iloc[0]
        print_result("McNemar NGS vs Phi-10",
                     f"chi2={mc['chi2']:.3f}  p={mc['p_value']:.4f}")

    if 'P2' in all_results and all_results['P2']:
        r = all_results['P2']
        print_sub("P2: Feature Importance")
        top = r['e1_df'].iloc[0]
        print_result("Top solo metric",
                     f"{top['NGS Metric']}  AUC={top['Solo AUC']:.4f}")
        print_result("Top SHAP metric",
                     f"{r['shap_df'].iloc[0]['NGS Metric']}")
        sfs = r['sfs_df']
        best_step = sfs.loc[sfs['Cumulative AUC'].idxmax()]
        print_result("SFS optimal set",
                     f"{int(best_step['Step'])} features  "
                     f"AUC={best_step['Cumulative AUC']:.4f}")

    if 'P3' in all_results and all_results['P3']:
        r = all_results['P3']
        print_sub("P3: Temporal Quartiles (Corollary 1)")
        print_result("Metrics supporting Corollary 1",
                     f"{r['n_supported']}/{len(r['slope_df'])}")

    if 'P5' in all_results and all_results['P5']:
        r = all_results['P5']
        print_sub("P5: IS Recalibrated Severity")
        print_result("rho(IS_recal, H&Y)",
                     f"{r['rho']:.3f}  p={r['p_val']:.4f}  "
                     f"{'MET ✓' if r['rho']>0.35 and r['p_val']<0.05 else 'NOT MET'}")

    if 'P6' in all_results and all_results['P6']:
        r = all_results['P6']
        print_sub("P6: NGS Severity Biomarkers")
        sig = r['P6_df'][r['P6_df']['Sig (raw)'].isin(['*','**'])]
        print_result("Significant NGS–H&Y correlations",
                     f"{len(sig)}/{len(r['P6_df'])}")
        if len(sig)>0:
            best = sig.iloc[0]
            print_result("Best",
                         f"{best['NGS Metric']}  rho={best['rho']:+.3f}  "
                         f"p={best['p_raw']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='NGS Pipeline v2.0')
    parser.add_argument('--pahaw',     default=CONFIG['pahaw_root'])
    parser.add_argument('--corpus',    default=CONFIG['corpus_path'])
    parser.add_argument('--newhandpd', default=None)
    parser.add_argument('--output',    default=CONFIG['output_dir'])
    parser.add_argument('--skip',      nargs='*', default=[],
                        help='Tracks to skip, e.g. --skip P3 P4')
    args = parser.parse_args()

    CONFIG['pahaw_root']    = args.pahaw
    CONFIG['corpus_path']   = args.corpus
    CONFIG['newhandpd_root']= args.newhandpd
    CONFIG['output_dir']    = args.output

    skip = set(args.skip or [])

    print_header("NGS EXPERIMENTAL PIPELINE v2.0", width=72)
    print("  Priority-ordered: P1→P2→P3→P4→P5→P6")
    print(f"  Output: {CONFIG['output_dir']}")
    if skip: print(f"  Skipping: {skip}")

    pipeline_t0 = time.time()
    make_dirs()
    all_results = {}

    # ── Data Loading ──────────────────────────────────────────────────────────
    print_header("DATA LOADING")
    t0 = time.time()
    part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
    print_sub("Building subject-level NGS")
    ngs_df  = build_subject_ngs(part_df)
    print(f"\n  ✓ Data loaded [{elapsed(t0)}]")

    # ── P1: Validation ────────────────────────────────────────────────────────
    if 'P1' not in skip:
        all_results = run_P1(part_df, ngs_df, all_results)

    # ── P2: Feature Importance ────────────────────────────────────────────────
    if 'P2' not in skip:
        all_results = run_P2(ngs_df, all_results)

    # ── P3: Temporal Quartiles ────────────────────────────────────────────────
    if 'P3' not in skip:
        all_results = run_P3(part_df, all_results)

    # ── P4: NewHandPD ─────────────────────────────────────────────────────────
    if 'P4' not in skip:
        nhroot = CONFIG.get('newhandpd_root')
        all_results = run_P4(nhroot or './data/NewHandPD', all_results)

    # ── P5: IS Recalibration ──────────────────────────────────────────────────
    if 'P5' not in skip:
        all_results = run_P5(part_df, all_results)

    # ── P6: NGS Severity ──────────────────────────────────────────────────────
    if 'P6' not in skip:
        all_results = run_P6(ngs_df, all_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(all_results)

    if CONFIG['save_pkl']:
        p = os.path.join(CONFIG['output_dir'],'pickles','ngs_v2_all_results.pkl')
        safe = {}
        for k,v in all_results.items():
            if isinstance(v,dict):
                safe[k]={sk:sv for sk,sv in v.items()
                         if not isinstance(sv,(nx.Graph,nx.DiGraph,np.ndarray))}
            else: safe[k]=v
        with open(p,'wb') as f: pickle.dump(safe,f)
        print(f"\n      [saved] {p}")

    print_header("PIPELINE v2.0 COMPLETE")
    print(f"  Total time : {elapsed(pipeline_t0)}")
    print(f"  Output     : {os.path.abspath(CONFIG['output_dir'])}")
    t_count = len(os.listdir(os.path.join(CONFIG['output_dir'],'tables')))
    f_count = len(os.listdir(os.path.join(CONFIG['output_dir'],'figures')))
    print(f"  Tables     : {t_count} CSV files")
    print(f"  Figures    : {f_count} PDF figures")
    print()
    print("  Usage examples:")
    print("  # Full run:")
    print("  python ngs_pipeline_v2.py --pahaw /path/PaHaW_public "
          "--corpus /path/corpus_PaHaW.xlsx")
    print("  # Skip slow tracks:")
    print("  python ngs_pipeline_v2.py ... --skip P3 P4")
    print("  # With NewHandPD:")
    print("  python ngs_pipeline_v2.py ... --newhandpd /path/NewHandPD")
    print('='*72)

if __name__ == '__main__':
    main()