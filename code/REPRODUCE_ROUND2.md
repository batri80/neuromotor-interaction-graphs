# Second-round analyses: run order

Run from inside `code/` with the data layout described in the top-level README.
Each script prints its results and writes the file listed.

```bash
# Cohort audit (Table 4)
python r2_audit_cohort.py --nh-healthy ./healthy --nh-patient ./patient
#   -> r2_audit_cohort_results.txt

# Estimator bias (Section 5.13, Table 15)
python r2_te_bias_correction.py                  # shuffled-source null
python r2_te_bias_correction.py --null shift     # circular time-shift null
#   -> r2_te_bias_results.txt, r2_te_bias_results_shift.txt

# Window-level reliability (Section 5.13), moving-block bootstrap without junction transitions
python r2_window_bootstrap_v2.py --estimator plugin
python r2_window_bootstrap_v2.py --estimator copula
#   -> r2_window_bootstrap_v2_plugin_L4.txt, r2_window_bootstrap_v2_copula_L4.txt

# GCN orientation check (Section 5.8)
python r2_gcn_check.py                           # -> r2_gcn_check_results.txt

# Appendix B.1: linear-Gaussian and Gaussian-copula TE
python r2_gaussian_te.py
python r2_gaussian_te.py --copula                # -> r2_gaussian_te_results*.txt

# Appendix B.2: native-resolution NIG (Table B1)
python r2_native_nig.py                          # -> r2_native_nig_results_p5.txt
python r2_native_reliability_lenmatched.py       # -> r2_native_reliability_lenmatched_p5.txt

# Appendix B.3: multi-task aggregation (Table B2)
python r2_multitask_nig.py                       # -> r2_multitask_nig_results_L500_p5.txt and CSVs
python r2_multitask_reliability_ci.py            # -> r2_multitask_reliability_ci.txt

# Appendix B.4: adjustment for pen-down writing duration and age
python r2_tempo_adjusted.py                      # -> r2_tempo_adjusted_results.txt

# Section 5.12: H&Y correlation of the multi-task descriptors
python r2_hy_multitask.py                        # -> r2_hy_multitask_results.txt

# Figure 2
python r2_fig2_redraw.py                         # -> Fig2_NIG_PD_vs_HC.pdf/.png, r2_fig2_values.txt

# NewHandPD (corrected loader) and cross-dataset transfer (Section 6, Table 16)
python ngs_p4_newhandpd.py --healthy ./healthy --patient ./patient
python third_dataset_transfer.py
python r2_check_nhpd_to_pahaw.py
python r2_transfer_diagnostics.py
python r2_check_drawings.py

# Additional checks referred to in the text
python r2_phi10_check.py                         # within-PaHaW Phi10 AUC (Section 6.3)
python r2_baseline_kfold.py                      # non-graph baseline under repeated 5-fold CV
python r2_table5_age.py                          # Table 5 with age as a covariate
```

`r2_window_bootstrap.py` is the first version of the window bootstrap, which allowed
transitions across block junctions; it is superseded by `r2_window_bootstrap_v2.py`
and its results are not used.
