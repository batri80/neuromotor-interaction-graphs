# Neuromotor Interaction Graphs (NIG)

Code for the paper

> K. Batri, S. Lakshmi, T. R. Mahesh, A. Shankar, A. R. Onteddu, W. Viriyasitavat.
> **Neuromotor Interaction Graphs: Inferring Functional Motor Control Networks from
> Handwriting Dynamics for Explainable Parkinson's Disease Assessment.**
> *Information Sciences* (under revision, manuscript INS-D-26-7505).

A Neuromotor Interaction Graph is a directed graph whose nodes are handwriting
biomarkers and whose edges are Transfer Entropy (TE) between them. It is estimated
separately for each participant from the time course of kinematic features within a
single recording, and is summarised by seven Neuromotor Graph Signature (NGS)
descriptors: graph entropy, global efficiency, modularity, algebraic connectivity,
small-world coefficient, spectral energy and an Ollivier–Ricci-type curvature.

The repository contains every script used to produce the tables and figures of the
revised manuscript, including the leakage audit of the first submission, the estimator
bias and reliability analysis (Section 5.13) and the alternative designs of Appendix B.
No data are included; see [Data](#data).

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

PyTorch is needed only for the GCN comparison (`gnn_comparison.py`,
`gcn_foldsafe_rerun.py`, `r2_gcn_check.py`, `debug_gnn.py`).

## Data

The datasets are not redistributed here. Obtain them from their providers and place
them next to the scripts as follows (all scripts run from inside `code/`):

```
code/
├── PaHaW/
│   ├── PaHaW_public/                 # one folder per participant (*.svc files)
│   └── PaHaW_files/corpus_PaHaW.xlsx # metadata: diagnosis, age, sex, H&Y, LED
├── healthy/                          # NewHandPD: HealthySpiral, HealthyMeander, HealthyCircle
├── patient/                          # NewHandPD: PatientSpiral, PatientMeander, PatientCircle
└── third_dataset/dataset/            # Parkinson's Drawings: {train,test}/{healthy,parkinson}
```

* **PaHaW**: research licence from CEITEC Masaryk University (Drotár et al., 2016).
  Task 1 (Archimedean spiral) is the primary task; Appendix B.3 uses all eight tasks.
  The tablet samples at about 125 Hz. 72 participants have a Task 1 recording
  (36 PD, 36 HC); 00080 (PD), 00061 and 00089 (HC) have none.
* **NewHandPD**: <https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/>.
  Participants are identified from the `-P<n>` / `-H<n>` suffix of the image filename
  (31 PD and 35 HC, nine images each).
* **Parkinson's Drawings** (Zham et al., 2017):
  <https://www.kaggle.com/datasets/kmader/parkinsons-drawings>. Spiral and wave
  drawings, 204 images (102 healthy, 102 Parkinsonian), evaluated at image level.

The second-round scripts (`r2_*.py`) accept `--pahaw` and `--corpus` if your folders
differ from this layout; the first-round scripts read the paths set in `CONFIG` at the
top of each file, which default to this layout. `ngs_p4_newhandpd.py` takes
`--healthy` and `--patient`.

## Reproducing the manuscript

Run the scripts from inside `code/`. Each writes its results to the console and, where
noted, to a text, CSV or pickle file in the current folder.

### Core modules (imported by the scripts below)

| File | Content |
|---|---|
| `ngs_pipeline_v2.py` | PaHaW loading and rendering, Φ10 image features, population-pooled NIG (Algorithm 1), the seven NGS descriptors (Section 3.3) |
| `test_kinematic_within_subject.py` | The ten instantaneous kinematic features (Table 3) and their window-level computation |
| `r2_common.py` | Within-subject design: W = 20 windows, plug-in TE (B = 8 equal-frequency bins, lag 1, nats), per-participant NGS, LOPO, nested repeated CV, permutation test, bootstrap CI |
| `ngs_p4_newhandpd.py` | NewHandPD loader (corrected participant grouping) and NewHandPD pipeline |
| `third_dataset_transfer.py` | Parkinson's Drawings loader and cross-dataset transfer |

### Main text

| Manuscript item | Script(s) |
|---|---|
| Table 4, Section 4.1 (cohort) | `r2_audit_cohort.py`, `check_svc_lengths.py` |
| Fig. 1 | `make_fig1_framework.py` |
| Fig. 2 | `r2_fig2_redraw.py` (supersedes `make_fig2_real_data.py`) |
| Table 5, Section 5.1 (group comparison) | `build_and_classify_within_subject.py`, `verify_direction_corrected.py`; age-adjusted check: `r2_table5_age.py` |
| Sections 5.2–5.3, Table 6 (LOPO, nested repeated CV, permutation test, bootstrap CI) | `improvement_experiments.py`, `permutation_and_bootstrap.py` (writes `nested_selection_fold_outputs.pkl`), `full_permutation_test.py`, `repeated_kfold_foldsafe.py` |
| Section 5.4, Table 7 (feature-selection frequency) | `permutation_and_bootstrap.py` |
| Section 5.5, Table 8 (TE vs Pearson vs MI edges) | `edge_comparison_and_transfer.py` |
| Section 5.6, Table 9 (bins, pruning, lag) | `te_bins_sensitivity_final.py`, `pruning_threshold_sensitivity.py`, `lag_sensitivity.py` (bin sweep on a subset: `sweep_te_bins.py`, `stress_test_shortest.py`) |
| Section 5.6 (undirected-weight convention, curvature edge limit) | `r2_undirected_rule.py` |
| Section 5.7, Table 10 (calibration, error analysis) | `calibration_and_failure_analysis.py` (reads the pickle above), `inspect_lopo_errors.py` |
| Section 5.8, Table 11 (robustness) | non-graph baseline `honest_nongraph_baseline.py` (repeated 5-fold version: `r2_baseline_kfold.py`); GCN `gnn_comparison.py`, `gcn_foldsafe_rerun.py`, `r2_gcn_check.py`; `test_extended_features_final.py`, `test_gradient_boosting.py`, `test_label_propagation.py`, `test_similarity_graph.py` |
| Section 5.10, Tables 13–14 (leakage audit of the first submission) | `test_population_variance_artifact.py`, `foldsafe_ngs.py`, `bagged_foldsafe_pipeline.py`, `test_bagging_mechanism.py`, `test_newhandpd_foldsafe.py`, `confirm_overfitting.py`, `diagnose_below_chance.py`, `verify_bugs.py` |
| Section 5.12 (severity) | `test_hy_severity_mechanism.py`, `r2_hy_multitask.py` |
| Section 5.13, Table 15 (estimator bias and reliability) | `r2_te_bias_correction.py` (shuffle null; `--null shift` for the time-shift null), `r2_window_bootstrap_v2.py` |
| Section 6 (cross-dataset transfer), Table 16 | `ngs_p4_newhandpd.py`, `third_dataset_transfer.py`, `r2_check_nhpd_to_pahaw.py`, `r2_transfer_diagnostics.py`, `r2_phi10_check.py`, `r2_check_drawings.py` |

### Appendix B

| Item | Script(s) |
|---|---|
| B.1 linear-Gaussian and Gaussian-copula TE | `r2_gaussian_te.py` (add `--copula` for the copula variant) |
| B.2 native-resolution NIG, Table B1 | `r2_native_nig.py`, then `r2_native_reliability_lenmatched.py` |
| B.3 multi-task aggregation, Table B2 | `r2_multitask_nig.py` (writes `r2_multitask_nig_L500_p5.csv`), then `r2_multitask_reliability_ci.py` |
| B.4 adjustment for writing duration and age | `r2_tempo_adjusted.py` (reads the CSV above) |

Further run notes for the second-round scripts are in `code/RUN_ROUND2.md` and
`code/RUN_TEMPLATE_FIXES.md`.

### Superseded and diagnostic scripts

These are kept for transparency; no reported result depends on them.

| Script | Status |
|---|---|
| `r2_window_bootstrap.py` | Superseded by `r2_window_bootstrap_v2.py` (the first version allowed transitions across block junctions) |
| `crossdataset_sourceonly_verify_v2.py` | Verification of source-only scaling in the transfer experiments; superseded by `third_dataset_transfer.py`, which already scales on the source only |
| `make_fig2_real_data.py` | First-round Figure 2; superseded by `r2_fig2_redraw.py` |
| `debug_gnn.py`, `diagnose_kinematic_graph.py`, `test_within_subject_small.py`, `full_kinematic_pipeline.py` | Development and sanity checks |

## Implementation notes

* **Legacy construction.** `build_subject_ngs()` and `run_P4()` in `ngs_pipeline_v2.py`
  reproduce the per-subject construction of the first submission, which inserts each
  participant into a group-conditional reference graph. It leaks label information and
  deflates within-group variance (manuscript Section 5.10). Both functions now emit a
  warning and are kept only so that the leakage audit can be reproduced.
* **NewHandPD participant grouping.** The first-submission loader grouped images by the
  drawing index instead of the participant. The corrected loader in
  `ngs_p4_newhandpd.py` uses the `-P<n>` / `-H<n>` suffix and raises an error for any
  filename it cannot parse.
* **Seeds.** The Erdős–Rényi null graphs of the small-world coefficient use a fixed seed
  (`SW_SEED = 0` in `ngs_pipeline_v2.py` and `ngs_p4_newhandpd.py`), so the coefficient
  is reproducible from run to run. Mutual-information edges use `random_state=0`;
  classifiers and CV splits use `random_state=42`.
* **Undirected conversion.** Efficiency, modularity, algebraic connectivity, the
  small-world coefficient and the curvature are computed on `G.to_undirected()`. When a
  pair of features is connected in both directions, NetworkX keeps the weight of the edge
  added last, i.e. the edge from the higher-indexed to the lower-indexed feature.
  `r2_undirected_rule.py` repeats the primary analysis with the mean and the maximum of
  the two weights (manuscript Section 5.6).
* **Curvature.** The curvature descriptor is an Ollivier–Ricci-type curvature computed on
  the symmetrised graph over at most the first 40 edges, with laziness α = 0.5,
  TE-proportional transition probabilities, and the expected hop distance under the
  product of the two neighbourhood measures as an upper bound on the Wasserstein-1
  cost, normalised by the TE weight of the edge.
* **Sampling rate.** Some early diagnostic scripts print durations using a nominal
  150 Hz rate; the analyses themselves work on sample counts and timestamps, and the
  actual rate is about 125 Hz.

## Citation

If you use this code, please cite the paper above and the archived release
(see `CITATION.cff`).

## Licence

MIT; see `LICENSE`.
