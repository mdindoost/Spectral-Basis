# Eigenvalue Distribution Analysis: Final Report

**Date:** 2026-02-18
**Hypothesis tested:** Eigenvalue distribution of the restricted Laplacian determines whether RowNorm succeeds.

---

## Hypothesis

**Claim:** Well-separated eigenvalues → diverse eigenvector directions → RowNorm succeeds.
**Counter-claim:** Clustered eigenvalues → nearly parallel eigenvectors → RowNorm fails.

**Metrics tested:**
1. **Condition number κ** = λ_max / λ_min (large κ → well-separated)
2. **Eigenvalue entropy H(λ)** = −Σ p_i log p_i where p_i = λ_i / Σλ_j (high H → spread-out spectrum)
3. **Effective rank** = number of eigenvalues capturing 95% of trace

**RowNorm recovery** = (Rayleigh-Ritz MLP+RowNorm test acc) − (Rayleigh-Ritz MLP test acc) × 100 pp
Source: `investigation_whitening_rownorm` at k=10.

---

## Data: All 9 Datasets

| Dataset | Family | κ (k=10) | H(λ) (k=10) | Eff. Rank | RN Fixed | RN Random |
|---------|--------|----------|-------------|-----------|----------|-----------|
| cora | citation | 66.69 | 6.998 | 1329 | +2.84pp | +4.64pp |
| citeseer | citation | 122.77 | 7.489 | 2158 | −0.03pp | +9.22pp |
| pubmed | citation | 8.26 | 6.152 | 486 | −0.84pp | −0.67pp |
| coauthor-cs | coauthorship | 72.68 | 8.705 | 6663 | +4.43pp | +6.27pp |
| coauthor-physics | coauthorship | 66.72 | 8.851 | 7523 | +5.25pp | +4.49pp |
| amazon-computers | product | 40.57 | 6.038 | 444 | +15.10pp | +15.45pp |
| amazon-photo | product | 40,821,672 | 6.344 | 731 | +8.42pp | +9.30pp |
| ogbn-arxiv | large | 9.09 | 4.706 | 126 | +64.73pp | +56.67pp |
| wikics | wikipedia | 5,735.84 | 4.742 | 299 | +12.90pp | +29.89pp |

**RowNorm recovery averages:** Fixed = +12.53pp, Random = +15.03pp (N=9)

---

## 1. Correlation Results (k=10)

### Condition Number κ vs RowNorm Recovery

| Split | Pearson r | p-value | Spearman ρ | Significance |
|-------|-----------|---------|------------|--------------|
| Fixed | −0.076 | 0.846 | +0.050 | ns |
| Random | −0.120 | 0.758 | +0.233 | ns |

→ **No correlation.** Condition number does not predict RowNorm recovery.

### Eigenvalue Entropy H(λ) vs RowNorm Recovery

| Split | Pearson r | p-value | Spearman ρ | Significance |
|-------|-----------|---------|------------|--------------|
| Fixed | −0.584 | 0.099 | −0.583 | borderline ns |
| Random | −0.696 | 0.037 | −0.700 | **p < 0.05*** |

→ **Weak-to-moderate negative correlation.** Higher entropy (more spread-out spectrum) predicts *lower* RowNorm recovery. This is **opposite to the hypothesis direction.**

### Effective Rank vs RowNorm Recovery

| Split | Pearson r | p-value | Spearman ρ | Significance |
|-------|-----------|---------|------------|--------------|
| Fixed | −0.325 | 0.394 | −0.583 | borderline ns |
| Random | −0.406 | 0.278 | −0.700 | **p < 0.05*** |

→ **Negative trend** (Spearman significant for random, Pearson not). Higher effective rank predicts lower recovery.

---

## 2. Dataset Patterns

**Highest κ (most separated eigenvalues):**
- amazon-photo: κ = 40,821,672 (extreme outlier due to disconnected components and diffusion)
- wikics: κ = 5,736
- citeseer: κ = 123

**Lowest κ (most clustered eigenvalues):**
- ogbn-arxiv: κ = 9.09
- pubmed: κ = 8.26
- amazon-computers: κ = 40.57

**RowNorm recovery by κ:** No clear pattern. ogbn-arxiv (lowest κ = 9.09) has the highest recovery (+64.73pp), while pubmed (κ = 8.26) has negative recovery (−0.84pp). High-κ datasets show mixed results.

**By entropy H(λ):**
- Low entropy (concentrated spectra): ogbn-arxiv (H=4.71), wikics (H=4.74) — these have higher RowNorm recovery (64.7pp and 12.9pp respectively)
- High entropy (spread-out spectra): coauthor-physics (H=8.85), coauthor-cs (H=8.71) — these have lower recovery (5.2pp and 4.4pp)

---

## 3. Hypothesis Validation

**The geometric diversity hypothesis is NOT confirmed.**

The data shows the opposite trend from what was predicted:
- **Condition number κ** shows no correlation with RowNorm recovery (r ≈ −0.08 to −0.12)
- **Entropy H(λ)** shows a *negative* correlation (r ≈ −0.58 to −0.70), meaning datasets with *more* eigenvalue diversity (higher entropy) tend to have *lower* RowNorm recovery
- **Effective rank** shows a similar negative trend

The most striking counterexample: **ogbn-arxiv** has highly clustered eigenvalues (κ=9.09, H=4.71, eff_rank=126) yet the highest RowNorm recovery (+64.73pp fixed). This single dataset strongly contradicts the geometric diversity hypothesis.

**Caveat — ogbn-arxiv outlier effect:**
ogbn-arxiv's +64.73pp recovery is extreme (Rayleigh-Ritz MLP: 5.86% → with RowNorm: 70.6%). This large recovery likely reflects a pathological scale normalization failure that RowNorm corrects, rather than geometric diversity per se. The extremely low baseline MLP performance (5.86% for 40 classes) suggests scale issues unrelated to eigenvalue distribution.

---

## 4. Key Findings

1. **RowNorm recovery range:** −0.84pp to +64.73pp (fixed), −0.67pp to +56.67pp (random). High variance across datasets.

2. **No simple predictor from eigenvalue distribution.** The condition number, entropy, and effective rank of the restricted Laplacian eigenproblem do not reliably predict RowNorm benefit.

3. **The hypothesis is falsified.** Well-separated eigenvalues do not cause RowNorm to work better. If anything, the trend is opposite (though ogbn-arxiv may be driving this via outlier effects).

4. **Statistical caveats:** N=9 datasets limits power. For p < 0.05 with N=9, need |r| > 0.666. Only entropy/effective_rank achieve this for random splits.

5. **Practical implication:** RowNorm recovery is likely driven by factors other than eigenvalue geometry — such as the scale of the raw features, dataset-specific conditioning, or the relationship between feature scale and the Laplacian structure.

---

## 5. Should This Go in the Paper?

**Recommendation: Yes, as a negative result with nuance.**

- The negative result is informative: it rules out eigenvalue distribution as the primary driver of RowNorm success
- The entropy trend (while opposite to hypothesis) is worth reporting
- The ogbn-arxiv outlier warrants separate investigation (why does RR MLP fail so catastrophically without RowNorm on ogbn-arxiv?)
- The analysis strengthens the empirical contribution by showing RowNorm benefit is not a simple geometric property

---

## Files Generated

| File | Description |
|------|-------------|
| `data/eigenvalue_analysis_complete.json` | All 9 datasets: eigenvalues at baseline + k=2,4,6,8,10 |
| `data/summary_table.csv` | Per-dataset condition numbers and spreads |
| `eigenvalue_metrics_complete.csv` | k=10 metrics + RowNorm recovery for all 9 datasets |
| `correlation_results.txt` | Statistical correlation results |
| `plots/figure1_eigenvalue_spectra_all.png` | 3×3 eigenvalue spectra (all 9 datasets, k=10) |
| `plots/figure2_kappa_vs_recovery_fixed.png` | κ vs RowNorm recovery, fixed splits |
| `plots/figure2b_kappa_vs_recovery_random.png` | κ vs RowNorm recovery, random splits |
| `plots/figure3_entropy_vs_recovery_fixed.png` | H(λ) vs RowNorm recovery, fixed splits |
| `plots/figure3b_entropy_vs_recovery_random.png` | H(λ) vs RowNorm recovery, random splits |
| `plots/figure4_metrics_heatmap.png` | Comprehensive metrics heatmap |

---

## Validation Checklist

- ✓ All 9 datasets processed (amazon-photo, coauthor-physics, wikics added)
- ✓ Used `compute_restricted_eigenvectors()` (correct function name from utils.py)
- ✓ Used `sgc_precompute()` for diffusion (k=10 steps)
- ✓ Full graph (not LCC) used for eigenvalue computation (matches existing 6-dataset analysis)
- ✓ RowNorm recovery from Whitening Verification at k=10 LCC
- ✓ LCC RowNorm recovery averaged: fixed=+12.53pp, random=+15.03pp
- ✓ No NaN or Inf (coauthor-physics κ well-defined; amazon-photo extreme κ due to diffusion spreading)
- ✓ Existing 6 datasets preserved from original JSON
- ✓ Correlation results include p-values and significance tests

---

*Generated by `experiments/complete_eigenvalue_analysis.py`*
