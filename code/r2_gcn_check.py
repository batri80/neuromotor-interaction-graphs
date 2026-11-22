"""
R2.2 — Is the below-chance GCN AUC (0.3651 ± 0.0226) an evaluation error or
a genuine modelling failure?

Uses the manuscript's fold-safe GCN code unchanged (graph construction,
architecture, optimiser, 60 epochs, weight decay 1e-3, fold-safe rescaling)
from gcn_foldsafe_rerun.py, and adds four checks:

  C1  Label and score orientation: PD is coded 1, the score is
      sigmoid(logit) = P(PD), and AUC is roc_auc_score(y, P(PD)). The script
      asserts this and also prints AUC(1 - score) for transparency.
  C2  In-sample (training) AUC of the fitted models. If orientation were
      inverted anywhere in the pipeline, training AUC would also be < 0.5.
  C3  Same GCN under repeated stratified 5-fold CV (5 x 5, class-balanced
      folds). If the LOPO value is a leave-one-out artefact, this should move
      to ~0.5; if the model genuinely learns an anti-signal it stays < 0.5.
  C4  Model-free demonstration of the leave-one-out artefact: a heavily
      regularised logistic regression on pure-noise features with the same
      36/36 design, under LOPO vs stratified 5-fold. (Removing one
      participant tilts the training base rate against that participant's
      class; a model whose output tracks the training base rate therefore
      ranks held-out participants backwards.) This part needs no data.

Usage:
  python r2_gcn_check.py --pahaw ./PaHaW/PaHaW_public \
      --corpus ./PaHaW/PaHaW_files/corpus_PaHaW.xlsx [--seeds 5] [--only-c4]
Output: r2_gcn_check_results.txt
"""

import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


def c4_noise_demo(out, n_per_class=36, n_feat=7, n_sims=200):
    rng = np.random.default_rng(0)
    y = np.array([1] * n_per_class + [0] * n_per_class)
    res = {}
    for C in [1e-4, 1e-2, 1.0]:
        lopo, kf = [], []
        for _ in range(n_sims):
            X = rng.standard_normal((len(y), n_feat))
            s = np.zeros(len(y))
            for tr, te in LeaveOneOut().split(X):
                clf = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr])
                s[te] = clf.predict_proba(X[te])[:, 1]
            lopo.append(roc_auc_score(y, s))
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                           random_state=int(rng.integers(1e9)))
            fa = []
            for tr, te in rskf.split(X, y):
                clf = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr])
                fa.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
            kf.append(np.mean(fa))
        res[C] = (np.mean(lopo), np.std(lopo), np.mean(kf), np.std(kf))
        out(f"  C={C:<7g} LOPO AUC = {res[C][0]:.4f} ± {res[C][1]:.4f}   "
            f"stratified 5-fold AUC = {res[C][2]:.4f} ± {res[C][3]:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
    ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--only-c4', action='store_true')
    args = ap.parse_args()

    log = open('r2_gcn_check_results.txt', 'w')
    def out(s=''):
        print(s); log.write(s + '\n'); log.flush()

    out("C4  Leave-one-out artefact on pure noise (36 PD / 36 HC, 7 features, 200 simulations)")
    c4_noise_demo(out)
    if args.only_c4:
        log.close(); return

    import torch
    import torch.nn.functional as F
    import gcn_foldsafe_rerun as g
    from ngs_pipeline_v2 import load_pahaw

    part_df = load_pahaw(args.pahaw, args.corpus)
    graphs, labels, names = [], [], []
    for _, row in part_df.iterrows():
        gr = g.build_subject_graph_raw(row['svc_data'])
        if gr is None:
            continue
        graphs.append(gr)
        labels.append(1 if row['label'] == 'PD' else 0)
        names.append(row['label'])
    y = np.array(labels)
    # C1: orientation — PD must be the positive class (1) everywhere
    assert set(np.unique(y)) == {0, 1}
    assert all((n == 'PD') == (lab == 1) for n, lab in zip(names, labels))
    out(f"\nCohort n={len(y)}  PD(=1)={int(y.sum())}  HC(=0)={int((y==0).sum())}")

    def fit_predict(train_idx, test_idx, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        flat = np.stack([graphs[i][0] for i in train_idx]).reshape(len(train_idx), -1)
        mu, sd = flat.mean(0), flat.std(0); sd[sd == 0] = 1.0
        resc = lambda f: ((f.reshape(1, -1) - mu) / sd).reshape(10, 2).astype(np.float32)
        model = g.SimpleGCN()
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        model.train()
        for _ in range(60):
            opt.zero_grad()
            loss = 0.0
            for i in train_idx:
                logit = model(torch.tensor(resc(graphs[i][0])), torch.tensor(graphs[i][1]))
                loss = loss + F.binary_cross_entropy_with_logits(
                    logit, torch.tensor([float(y[i])]))
            (loss / len(train_idx)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = lambda i: torch.sigmoid(model(torch.tensor(resc(graphs[i][0])),
                                                  torch.tensor(graphs[i][1]))).item()
            te = np.array([score(i) for i in test_idx])
            tr = np.array([score(i) for i in train_idx])
        return te, tr

    out("\nC1/C2  LOPO, fold-safe (identical to manuscript protocol)")
    lopo_aucs, train_aucs = [], []
    t0 = time.time()
    for seed in range(args.seeds):
        s = np.zeros(len(y)); tr_auc = []
        for i in range(len(y)):
            tr_idx = [k for k in range(len(y)) if k != i]
            te, tr = fit_predict(tr_idx, [i], seed)
            s[i] = te[0]
            tr_auc.append(roc_auc_score(y[tr_idx], tr))
        a = roc_auc_score(y, s)
        lopo_aucs.append(a); train_aucs.append(np.mean(tr_auc))
        out(f"  seed {seed}: held-out AUC(P(PD))={a:.4f}  AUC(1-P(PD))={1-a:.4f}  "
            f"mean training AUC={np.mean(tr_auc):.4f}  "
            f"score range [{s.min():.3f}, {s.max():.3f}]  SD={s.std():.4f}")
    out(f"  LOPO mean {np.mean(lopo_aucs):.4f} ± {np.std(lopo_aucs):.4f}; "
        f"training AUC mean {np.mean(train_aucs):.4f}  ({(time.time()-t0)/60:.1f} min)")

    out("\nC3  Repeated stratified 5-fold (5 x 5), fold-safe")
    kf_means = []
    for seed in range(args.seeds):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        fa = []
        for tr, te in rskf.split(np.zeros(len(y)), y):
            p, _ = fit_predict(list(tr), list(te), seed)
            fa.append(roc_auc_score(y[te], p))
        kf_means.append(np.mean(fa))
        out(f"  seed {seed}: mean fold AUC={np.mean(fa):.4f} ± {np.std(fa):.4f}")
    out(f"  5-fold mean {np.mean(kf_means):.4f} ± {np.std(kf_means):.4f}")
    out(f"\nTotal {(time.time()-t0)/60:.1f} min")
    log.close()


if __name__ == '__main__':
    main()
