# Code Review: PAPER_EXPERIMENTS
**Date:** 2026-02-22
**Reviewer:** Claude (expert ML + spectral graph theory review)
**Status:** COMPLETE — all issues resolved or deferred with decision

---

## CRITICAL Issues

| ID | Status | File | Description |
|----|--------|------|-------------|
| C-1 | ✅ FIXED | `src/utils.py:520`, `master_training.py` | Double self-loop in SGC diffusion: computes `D^{-1/2}(A+2I)D^{-1/2}` instead of `D^{-1/2}(A+I)D^{-1/2}` |
| C-2 | ⬜ TODO | `src/utils.py`, `master_training.py` | `spectral_rownorm_alpha0.0` != `restricted_rownorm_mlp`: different architectures (bias, intermediate normalization) |
| C-3 | ✅ FIXED | `master_training.py:308-320` | Fisher score: denominator uses mean of per-class std instead of mean of per-class variance -- dimensionally inconsistent |
| C-4 | ✅ NOTED | `master_analytics.py:188-193` | Span verification is trivially 0 when `d_eff >= n_train` (always for most datasets) -- metric is vacuous |

---

## MAJOR Issues

| ID | Status | File | Description |
|----|--------|------|-------------|
| M-1 | ✅ NOTED | `master_training.py:860-861` | `remaining_gap_pp` algebraically simplifies to `SGC_acc - RowNorm_acc` -- not a meaningful "gap" quantity |
| M-2 | ✅ FIXED | `master_analytics.py:337-374` | Random splits: fallback always uses fixed split `train_idx`, so spectral analysis is on the wrong nodes |
| M-3 | ✅ FIXED | `master_analytics.py:107-124` | Condition number of U uses all `n` nodes; condition number of X uses only `n_train` nodes -- incomparable |
| M-4 | ✅ FIXED | `master_training.py:930-937` | Top-level `fisher_score` in JSON is always split-0 value, not the mean across splits |
| M-5 | ✅ NOTED | `master_training.py:257-266` | `speed_to_90/95/99` measures fraction of each model's own peak accuracy, not absolute threshold -- incommensurable across methods |
| M-6 | ✅ FIXED | `generate_paper_artifacts_v2.py:502` | Recovery rate condition `logmag_acc > 0` is always true; negative recovery accepted silently |

---

## MINOR Issues

| ID | Status | File | Description |
|----|--------|------|-------------|
| m-1 | ✅ NOTED | `src/utils.py:83-95` | `StandardMLP` has no dropout -- massive overfitting (Cora: train=100%, test=38%). Creates architectural confound vs. other models |
| m-2 | ✅ NOTED | `generate_paper_artifacts_v2.py` | All artifact functions hardcode `split_type='fixed'` -- no way to generate random-split artifacts |
| m-3 | ✅ FIXED | `generate_paper_artifacts_v2.py:762` | Section 6 functions hardcoded to CiteSeer only |
| m-4 | ✅ FIXED | `generate_paper_artifacts_v2.py:591` | Figure 5.1 y-axis: `min(accs)*0.95` becomes 0 if a method is missing (stored as 0 default) |
| m-5 | ✅ FIXED | `src/utils.py:SpectralRowNormMLP` | Docstring claims "RowNorm with eigenvalue weighting" but architecture differs from RowNormMLP |
| m-6 | ✅ NOTED | `master_analytics.py:171-172` | `spectral_separability` values typically <1 (inter < intra distance) but name implies higher=better; not documented |

---

## Review Log

*(filled in as each item is investigated)*

---

### C-1: Double Self-Loop in SGC Diffusion
- **Status:** ✅ FIXED
- **Finding:** Confirmed. `build_graph_matrices` adds self-loops to adj, then `compute_sgc_normalized_adjacency` added them again. Result: diagonal = `2/(deg+2)` instead of correct `1/(deg+1)`. For a degree-4 node: 0.333 vs correct 0.200 (67% overweight on self).
- **Fix:** Removed `adj = adj + sp.eye(adj.shape[0])` from `compute_sgc_normalized_adjacency`. Updated docstring. Added test `test_diagonal_equals_1_over_deg_plus_1`.
- **Impact:** All experiments must be re-run with corrected diffusion operator.

### C-2: SpectralRowNorm alpha=0 != RowNormMLP
- **Status:** 🔄 DEFERRED
- **Finding:** Confirmed architectural differences: SpectralRowNormMLP has bias terms, no intermediate re-normalization, has Dropout(0.5). RowNormMLP has no bias, re-normalizes after every hidden layer, no dropout. 1440 vs 1507 params.
- **Decision:** Deferred -- depends on whether paper claims alpha=0 = RowNorm baseline.

### C-3: Fisher Score Denominator (std vs variance)
- **Status:** ✅ FIXED
- **Finding:** Confirmed. `_fisher_on_magnitudes` used `M_c.std()` (units: magnitude) in denominator instead of `M_c.var()` (units: magnitude^2). This made the Fisher score dimensionally inconsistent (units: magnitude instead of dimensionless) and not scale-invariant.
- **Fix:** Changed `class_stds / M_c.std()` to `class_vars / M_c.var()` and `cs.mean()` to `cv.mean()`. Result is now a true dimensionless ratio, scale-invariant. Old buggy score was ~0.46x the correct value (sqrt factor).
- **Impact:** All Fisher diagnostic values in PAPER_RESULTS are wrong. Must re-run analytics after training re-run.

### C-4: Span Verification Vacuous
- **Status:** ✅ NOTED
- **Finding:** When d_eff >= n_train (true for all datasets: e.g. Cora d_eff=1427, n_train=140), pinv(U_tr) is a right-inverse of U_tr, so U_tr @ pinv(U_tr) = I exactly. span_error is always ~0 regardless of whether U truly spans X.
- **Decision:** Not reporting this metric in paper. Added prominent WARNING comment in master_analytics.py explaining why it is meaningless.

### M-1: remaining_gap_pp = SGC - RowNorm
- **Status:** ✅ NOTED
- **Finding:** remaining_gap_pp = Part A - Part B = (SGC - Std) - (RowNorm - Std) = SGC - RowNorm. The Std terms cancel algebraically. The value is mathematically valid (the unrecovered portion of SGC's advantage), but the multi-term formula obscured this.
- **Fix:** Simplified the formula to `sgc_mlp_acc - restr_rn_acc` with an explanatory comment. Value is identical to before.

### M-2: Wrong train_idx for Random Splits in Analytics
- **Status:** ✅ FIXED
- **Finding:** Confirmed. .npz matrix caching is disabled in master_training.py, so the fallback in master_analytics.py always runs. The fallback set `train_idx = split_idx['train_idx']` (the dataset's fixed split) even when SPLIT_TYPE='random'. The spectral analysis (condition numbers, separability) was then computed on the wrong set of nodes.
- **Fix:** Compute the correct `train_idx` once before the k-loop, replicating master_training.py's split 0 generation exactly (`np.random.seed(0)`, shuffle, take first 60%). Removed the incorrect override in the fallback branch.

### M-3: U vs X Condition Numbers on Different Node Sets
- **Status:** ✅ FIXED
- **Finding:** `sv_U` was computed on full `U` (n x d_eff rows), `sv_X` on `X_tr` (n_train x d_raw rows). Different matrix shapes make singular values and condition numbers incommensurable.
- **Fix:** Compute `sv_U` on `U_tr = U[train_idx]` (same n_train rows as X_tr). Now both condition numbers are on matching n_train x d matrices and can be compared directly.

### M-4: fisher_score Always Split-0
- **Status:** ✅ FIXED
- **Finding:** `fisher_agg['fisher_score']` stored `fisher_diags[0]['fisher_score']` (split 0 only). `mean_fisher_score` existed alongside it but anyone reading the top-level key got the wrong value.
- **Fix:** Changed `fisher_score` to store `np.mean(fisher_scores)` (mean across all splits). Removed redundant `mean_fisher_score` key. Per-split values still accessible via `per_split`.

### M-5: speed_to_90 Relative Not Absolute
- **Status:** ✅ NOTED
- **Finding:** `speed_to(0.90)` computes epochs to reach `0.90 * this_model's_own_max_acc`. If RowNorm peaks at 85% and Std peaks at 60%, their "speed_to_90" targets are 76.5% and 54% respectively -- not comparable.
- **Fix:** Renamed keys to `speed_to_90_pct_of_peak` etc. to make the relative nature explicit. Added warning comment. Cross-method convergence comparison should use `checkpoint_val_accs` instead.

### M-6: Recovery Rate Condition Always True
- **Status:** ✅ FIXED
- **Finding:** `logmag_acc > 0` used to detect method presence, but any trained classifier gets >0%, so it is true whenever the method ran. Also, `if part_b5_logmag` is Python-falsy when value is exactly 0.0.
- **Fix:** Replaced `> 0` sentinel with explicit `'log_magnitude' in exps` key check. `part_b5_logmag` now uses `is not None` guard. Recovery rate now computes even when negative (negative recovery is valid data -- logmag also hurt). Added comment documenting that negative recovery is possible.

### m-1: StandardMLP No Dropout
- **Status:** ✅ NOTED
- **Finding:** Confirmed. StandardMLP has no dropout. On small datasets like Cora it overfits severely (train ~100%, test ~38%). RowNormMLP benefits implicitly from row-normalisation acting as regularisation, creating an unfair comparison.
- **Decision:** Adding dropout requires experiment rerun. Added prominent WARNING docstring explaining the confound. StandardMLP should only be used to measure the RowNorm effect (Part B), not as a general baseline.

### m-2: Hardcoded split_type='fixed'
- **Status:** ✅ NOTED
- **Finding:** All load_* helper functions default to `split_type='fixed'` and all section functions call them without overriding. Random-split artifacts cannot currently be generated.
- **Decision:** Paper uses fixed splits as canonical results. Accepted as current limitation; add random-split support if needed later.

### m-3: Section 6 Hardcoded to CiteSeer
- **Status:** ✅ FIXED
- **Finding:** Both section 6 functions hardcoded dataset='citeseer'.
- **Fix:** Both functions now loop over all DATASETS, skipping any without dynamics data. Figure produces one PDF per dataset plus a combined multi-panel PDF. Table has all datasets as row groups separated by \midrule, with Dataset/Method/Speed90/Speed95/Speed99/AUC columns.

### m-4: Figure 5.1 y-axis Bug
- **Status:** ✅ FIXED
- **Finding:** `accs = [std_acc, rn_acc, logmag_acc, dual_acc]` included 0-default values for missing methods. `min(accs)*0.95 = 0` whenever any method is absent, forcing y-axis to start at 0.
- **Fix:** Only include methods that are actually present in results. Build `accs` list dynamically by checking `'log_magnitude' in exps`. `min(accs)` now uses only real accuracy values.

### m-5: SpectralRowNormMLP Docstring Misleading
- **Status:** ✅ FIXED
- **Finding:** Docstring said "RowNorm with eigenvalue weighting" and implied `alpha=0 = standard RowNorm`. Architecture actually differs: has bias=True, normalises input only (not intermediate), has Dropout(0.5).
- **Fix:** Rewrote docstring to accurately describe the architecture, explicitly list the three differences from RowNormMLP, and reference C-2 in CODE_REVIEW.md.

### m-6: spectral_separability Not Documented
- **Status:** ✅ NOTED
- **Finding:** `spectral_separability = inter_centroid_dist / intra_mean_dist`. On real datasets this is typically <1 because class clusters overlap. The name implies higher=better but the range and expected values were not documented.
- **Fix:** Added inline comment explaining the formula, interpretation (>1 = well-separated, <1 = overlap), and that typical real-dataset values are small but higher is still better.
