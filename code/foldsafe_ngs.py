"""
FOLD-SAFE NGS CONSTRUCTION
==========================
Drop-in replacement for `build_subject_ngs()` in ngs_pipeline_v2.py /
ngs_pahaw_pipeline.py / ngs_p4_newhandpd.py.

Fixes two stacked problems in the original implementation:

  1. LABEL LEAKAGE (population pooling): the original built each subject's
     graph using ALL subjects of their group (including test-fold subjects
     in every CV split), so held-out subjects' features encoded information
     about held-out subjects.

  2. TRUE-LABEL CIRCULARITY: the original selected which group's population
     graph to embed a subject in using `row['label']` -- i.e. the answer
     the classifier is supposed to predict. This makes classification
     impossible in principle for a genuinely unseen subject, independent
     of the pooling issue above.

FIX: for each CV fold, build two REFERENCE graphs from TRAINING subjects
only (G_PD_train, G_HC_train). For every subject -- train or test -- compute
a feature vector as the graph-signature DELTA caused by inserting that
subject into EACH reference graph. Both deltas are always computed, for
every subject, regardless of true label, so nothing about the label is
used to build the feature. This removes both the pooling leak and the
circularity.

Requires: build_nig(), extract_ngs(), NGS_CORE, CONFIG from the original
pipeline file (imported, not redefined, to keep TE/graph logic identical).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _reference_graph_ngs(build_nig_fn, extract_ngs_fn, X_group, label):
    """Build a population graph from a training-only group matrix and
    extract its NGS. Returns (graph, ngs_dict)."""
    G = build_nig_fn(X_group, label=label)
    ngs = extract_ngs_fn(G)
    return G, ngs


def _delta_ngs(build_nig_fn, extract_ngs_fn, X_group, x_subject, ref_ngs, label, ngs_keys):
    """Insert one subject's (already-scaled) feature row into a training-only
    group matrix, rebuild the graph, and return the NGS delta vs. the
    reference (pre-insertion) graph. x_subject: shape (1, n_feat)."""
    X_aug = np.vstack([X_group, x_subject])
    G_aug = build_nig_fn(X_aug, label=f"{label}+subj")
    ngs_aug = extract_ngs_fn(G_aug)
    return np.array([ngs_aug.get(k, 0.0) - ref_ngs.get(k, 0.0) for k in ngs_keys])


def build_fold_safe_ngs(part_df, feat_cols, ngs_keys, build_nig_fn, extract_ngs_fn,
                         train_idx, test_idx=None):
    """
    Compute leakage-free, non-circular subject-level NGS delta features
    for one CV fold.

    Parameters
    ----------
    part_df    : full participant DataFrame (must contain feat_cols, 'label',
                 'label_int', 'participant_id')
    feat_cols  : list of Phi-10 column names used as graph nodes
    ngs_keys   : list of NGS metric names to extract (e.g. NGS_CORE)
    build_nig_fn, extract_ngs_fn : imported unmodified from the original
                 pipeline file -- graph/TE logic itself is untouched
    train_idx  : integer positional indices (into part_df) for this fold's
                 TRAINING subjects
    test_idx   : integer positional indices for this fold's TEST subjects.
                 If None, features are produced for train_idx only (use
                 this to build a fold's training feature matrix once,
                 then call again with test_idx set to score held-out
                 subjects against the SAME frozen reference graphs).

    Returns
    -------
    DataFrame indexed by participant_id with columns:
        {metric}_dPD, {metric}_dHC   for each metric in ngs_keys
        label, label_int
    Note: the true label of a test subject is NEVER read in this function
    except to attach it to the output row for downstream scoring -- it is
    never used to decide which reference graph a subject's delta is
    computed against. Both deltas are always computed for every subject.
    """
    X_all = part_df[feat_cols].values
    labels = part_df['label'].values

    train_mask = np.zeros(len(part_df), dtype=bool)
    train_mask[train_idx] = True

    # Scaler fit on TRAINING data only (pooled across both groups, since a
    # subject's group is not known at feature-construction time for test
    # subjects -- fitting per-group would itself require the label).
    scaler = StandardScaler().fit(X_all[train_mask])
    X_all_z = scaler.transform(X_all)

    # Reference (training-only) group graphs.
    pd_train_idx = [i for i in train_idx if labels[i] == 'PD']
    hc_train_idx = [i for i in train_idx if labels[i] == 'HC']
    X_pd_train = X_all_z[pd_train_idx]
    X_hc_train = X_all_z[hc_train_idx]

    _, ngs_pd_ref = _reference_graph_ngs(build_nig_fn, extract_ngs_fn, X_pd_train, 'PD_train')
    _, ngs_hc_ref = _reference_graph_ngs(build_nig_fn, extract_ngs_fn, X_hc_train, 'HC_train')

    subject_idx = list(train_idx) + (list(test_idx) if test_idx is not None else [])
    rows = []
    for i in subject_idx:
        x_s = X_all_z[i:i+1]

        # For a TRAINING subject who is themselves a member of the reference
        # population, remove them first (true leave-one-out) so their own
        # delta isn't measured against a graph that already contains them.
        if i in pd_train_idx:
            X_pd_ref_i = X_all_z[[j for j in pd_train_idx if j != i]]
            d_pd = _delta_ngs(build_nig_fn, extract_ngs_fn, X_pd_ref_i, x_s,
                               _reference_graph_ngs(build_nig_fn, extract_ngs_fn, X_pd_ref_i, 'PD_loo')[1],
                               'PD', ngs_keys)
            d_hc = _delta_ngs(build_nig_fn, extract_ngs_fn, X_hc_train, x_s, ngs_hc_ref, 'HC', ngs_keys)
        elif i in hc_train_idx:
            X_hc_ref_i = X_all_z[[j for j in hc_train_idx if j != i]]
            d_hc = _delta_ngs(build_nig_fn, extract_ngs_fn, X_hc_ref_i, x_s,
                               _reference_graph_ngs(build_nig_fn, extract_ngs_fn, X_hc_ref_i, 'HC_loo')[1],
                               'HC', ngs_keys)
            d_pd = _delta_ngs(build_nig_fn, extract_ngs_fn, X_pd_train, x_s, ngs_pd_ref, 'PD', ngs_keys)
        else:
            # Genuine held-out test subject: never in either reference set.
            d_pd = _delta_ngs(build_nig_fn, extract_ngs_fn, X_pd_train, x_s, ngs_pd_ref, 'PD', ngs_keys)
            d_hc = _delta_ngs(build_nig_fn, extract_ngs_fn, X_hc_train, x_s, ngs_hc_ref, 'HC', ngs_keys)

        row = {'participant_id': part_df.iloc[i]['participant_id'],
               'label': part_df.iloc[i]['label'],
               'label_int': part_df.iloc[i]['label_int']}
        for k, key in enumerate(ngs_keys):
            row[f'{key}_dPD'] = d_pd[k]
            row[f'{key}_dHC'] = d_hc[k]
        rows.append(row)

    return pd.DataFrame(rows)


def lopo_cv_fold_safe(part_df, feat_cols, ngs_keys, build_nig_fn, extract_ngs_fn, clf):
    """
    Full leave-one-participant-out CV using fold-safe NGS features.
    Rebuilds reference graphs fresh for every held-out subject (n calls,
    each O(group_size) TE computations -- expect this to be noticeably
    slower than the original single-pass build_subject_ngs; for n=72
    subjects x 10-node graphs this is still tractable, just budget for it).

    Returns arrays (y_true, y_pred, y_prob) exactly like the original
    lopo_cv(), so it drops into run_P1 with minimal changes.
    """
    from sklearn.metrics import roc_auc_score
    n = len(part_df)
    y_true, y_pred, y_prob = [], [], []

    for test_i in range(n):
        train_idx = [j for j in range(n) if j != test_i]
        feats = build_fold_safe_ngs(part_df, feat_cols, ngs_keys,
                                     build_nig_fn, extract_ngs_fn,
                                     train_idx=train_idx, test_idx=[test_i])
        feat_cols_out = [c for c in feats.columns if c not in
                         ('participant_id', 'label', 'label_int')]

        train_rows = feats[feats['participant_id'] != part_df.iloc[test_i]['participant_id']]
        test_row = feats[feats['participant_id'] == part_df.iloc[test_i]['participant_id']]

        Xtr = train_rows[feat_cols_out].values
        ytr = train_rows['label_int'].values
        Xte = test_row[feat_cols_out].values

        sc = StandardScaler()
        clf.fit(sc.fit_transform(Xtr), ytr)
        y_true.append(part_df.iloc[test_i]['label_int'])
        y_pred.append(clf.predict(sc.transform(Xte))[0])
        try:
            y_prob.append(clf.predict_proba(sc.transform(Xte))[0, 1])
        except Exception:
            y_prob.append(float(y_pred[-1]))

    return np.array(y_true), np.array(y_pred), np.array(y_prob)
