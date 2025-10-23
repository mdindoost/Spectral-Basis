# Random Subspace Experiment

## Research Question

**What determines whether RowNorm works on restricted eigenvectors?**

We know:
- **Investigation 1** (true eigenvectors): RowNorm **overperforms** (+15-40%)
- **Investigation 2** (restricted eigenvectors from X): RowNorm **underperforms** (-9.3% average)

**Is the problem**:
1. Fundamental limitation of restricted eigenvectors? OR
2. Poor conditioning of engineered features X?

## Experimental Design

### Generate Random Subspaces

Instead of using engineered features X, use **random Gaussian features**:

```python
X_r ~ N(0,1)  # Shape: (num_nodes, dimension)
```

Where `dimension = rank(X)` to ensure fair comparison.

### Run Three Experiments

For each random subspace X_r:

- **(a) X_r → StandardScaler → Standard MLP** [Baseline]
- **(b) V_r → StandardScaler → Standard MLP** [Basis sensitivity]
- **(c) V_r → RowNorm MLP** [RowNorm on random]

Where V_r = restricted eigenvectors computed from X_r.

### Control for Randomness

- Generate **10 different random subspaces** (different seeds)
- **5 training runs per subspace** (random initialization)
- **Total: 50 runs per experiment** (10 × 5)

### Compare to Original

Run same experiments on engineered features X and compare:

| Metric | Engineered X | Random X_r |
|--------|--------------|------------|
| Direction A (RowNorm effect) | ? | ? |
| Baseline accuracy | ? | ? |
| Variance across subspaces | N/A | ? |

## Expected Outcomes

### Outcome 1: RowNorm Works on Random

**Direction A improves**: Random X_r shows positive RowNorm effect.

**Interpretation**:
- Problem is with engineered feature conditioning
- Random Gaussian features are isotropic and well-conditioned
- Restricted eigenvectors work better with clean subspaces

**Implication**: Feature engineering introduces harmful structure.

---

### Outcome 2: RowNorm Fails on Random Too

**Direction A still negative**: Random X_r shows no improvement.

**Interpretation**:
- Problem is fundamental to restricted eigenvectors
- Neither engineered nor random subspaces help
- RowNorm requires true spectral structure (full eigenvectors)

**Implication**: RowNorm not salvageable for restricted eigenvectors.

---

### Outcome 3: Random Outperforms Engineered 🤯

**Experiment (a)**: X_r accuracy > X accuracy.

**Interpretation**:
- Graph structure (via random projection + Rayleigh-Ritz) captures task-relevant info
- Feature engineering may be over-fitting or introducing biases
- Unsupervised graph-based features rival supervised engineering

**Implication**: Major finding! Feature engineering may be redundant.

---

### Outcome 4: High Variance

**Large std across random subspaces**: Results vary significantly.

**Interpretation**:
- Specific random subspace matters
- Some directions work better than others
- Need to understand favorable subspaces

**Implication**: Random subspace selection is important.

## Usage

### Run Single Dataset

```bash
python experiments/investigation2_random_subspaces.py [dataset]
```

Examples:
```bash
python experiments/investigation2_random_subspaces.py ogbn-arxiv
python experiments/investigation2_random_subspaces.py wikics
python experiments/investigation2_random_subspaces.py coauthor-physics
```

### Run All Datasets

```bash
chmod +x scripts/run_random_all_datasets.sh
./scripts/run_random_all_datasets.sh
```

**Estimated time**: 2-6 hours depending on hardware.

### Generate Summary

```bash
python scripts/summarize_random_subspaces.py
```

This produces:
1. Complete comparison table (engineered vs random)
2. Key findings analysis
3. Variance statistics
4. Interpretation guide

## Output Structure

```
results/investigation2_random_subspaces/
└── <dataset>/
    └── fixed_splits/
        ├── metrics/
        │   └── results_complete.json    # All results
        └── plots/                        # (Future: convergence plots)
```

### Results JSON Structure

```json
{
  "dataset": "ogbn-arxiv",
  "dimension": 128,
  "num_random_subspaces": 10,
  "num_seeds_per_subspace": 5,
  
  "original_features": {
    "a_X_std": {"test_acc_mean": 0.5231, "test_acc_std": 0.0021, ...},
    "b_U_std": {...},
    "c_U_row": {...}
  },
  
  "random_subspaces": [
    // 10 entries, one per random subspace
    {"a_X_std": {...}, "b_U_std": {...}, "c_U_row": {...}},
    ...
  ],
  
  "aggregated_random": {
    "a_X_std": {
      "mean_across_subspaces": 0.5123,
      "std_across_subspaces": 0.0156,
      "min": 0.4987,
      "max": 0.5289
    },
    ...
  },
  
  "comparisons": {
    "direction_A_original": -6.5,      // RowNorm effect on X
    "direction_A_random": 2.3,         // RowNorm effect on X_r
    "direction_B_original": -3.4,      // Basis sensitivity on X
    "direction_B_random": -1.8,        // Basis sensitivity on X_r
    "baseline_diff": -0.0108           // X_r - X accuracy
  }
}
```

## Implementation Details

### Random Subspace Generation

```python
def generate_random_subspace(num_nodes, dimension, seed):
    rng = np.random.RandomState(seed)
    X_r = rng.randn(num_nodes, dimension).astype(np.float64)
    return X_r
```

**Notes**:
- Raw Gaussian (not normalized)
- Same dimension as rank(X) after QR decomposition
- Different seed for each subspace

### Restricted Eigenvector Computation

Same as Investigation 2:
1. QR decomposition of X_r (handles potential rank issues)
2. Project Laplacian: L_r = Q^T L Q
3. Project degree: D_r = Q^T D Q
4. Solve: (L_r)v = λ(D_r)v
5. Map back: U_r = Q @ V

### Training Protocol

Identical to Investigation 2:
- 200 epochs
- Hidden dimension: 256
- Batch size: 128
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Loss: CrossEntropyLoss

## Key Comparisons

### 1. RowNorm Effect: Engineered vs Random

| Dataset | Original X | Random X_r | Improvement |
|---------|------------|------------|-------------|
| ...     | -6.5%      | +2.3%      | **+8.8pp** |

**Question**: Does RowNorm work better on random features?

---

### 2. Baseline Accuracy: Engineered vs Random

| Dataset | Engineered X | Random X_r | Difference |
|---------|--------------|------------|------------|
| ...     | 52.31%       | 51.23%     | -1.08pp    |

**Question**: Can random features match engineered features?

---

### 3. Variance Across Subspaces

| Dataset | Exp (a) Std | Exp (b) Std | Exp (c) Std |
|---------|-------------|-------------|-------------|
| ...     | 0.0156      | 0.0234      | 0.0189      |

**Question**: How sensitive are results to random subspace choice?

## What to Look For

### If RowNorm improves significantly:

✓ **Hypothesis supported**: Engineered features have poor conditioning
- Investigate covariance structure of X vs X_r
- Compare eigenvalue distributions
- Consider random projection preprocessing

### If random matches/exceeds engineered:

🤯 **Major finding**: Graph structure sufficient!
- Feature engineering may be over-fitting
- Consider unsupervised graph-based features
- This would be publication-worthy on its own

### If high variance across subspaces:

📊 **Subspace matters**: Need to understand why
- Analyze successful vs unsuccessful subspaces
- Look for patterns in eigenvalue distributions
- Consider guided random projection

### If results similar to Investigation 2:

✗ **Fundamental limitation**: RowNorm needs true eigenvectors
- Neither engineered nor random help
- Focus on other solutions (different architectures, etc.)

## Professor Koutis's Intuition

From his email:
> "It'd be surprising if we can in an unsupervised way construct features X_r that are better than what has been engineered into X"

**He suspects**:
- Random features might work better due to isotropy
- Engineered features may have harmful correlations/anisotropy
- This is a clever test to isolate the problem

## Timeline

**Per dataset** (approximate):
- Small (Cora, CiteSeer): 10-15 minutes
- Medium (PubMed, WikiCS, Amazon): 20-30 minutes
- Large (ogbn-arxiv): 30-45 minutes
- Very large (Coauthor): 60-90 minutes

**All 9 datasets**: 2-6 hours total (depends on GPU).

## Next Steps After Results

**Based on outcome**:

1. **If Outcome 1** → Analyze feature conditioning
2. **If Outcome 3** → Write up major finding
3. **If Outcome 4** → Study subspace patterns
4. **If negative** → Focus on other solutions

**Then**:
- Share results with Professor Koutis
- Discuss interpretation
- Plan next experiments or writing

## Notes

- Uses same codebase as Investigation 2
- GPU strongly recommended (CPU will be very slow)
- Results are deterministic (fixed random seeds)
- Can interrupt and resume individual datasets

---

**Questions?** Check the code comments in `investigation2_random_subspaces.py` or ask Mohammad!
