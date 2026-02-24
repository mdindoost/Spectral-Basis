# MLP Sensitivity to Spectral Basis Representations
**PhD Research — Mohammad, NJIT, Advisor: Prof. Ioannis Koutis**

---

## PAPER NARRATIVE

> When node features are diffused through the graph (SGC: Â^k X) and then
> projected onto a restricted eigenvector basis (Rayleigh-Ritz), a performance
> gap opens up — the basis is sensitive to this transformation (Part A).
> Row normalization is a natural candidate to fix this gap, but it only
> sometimes works: on some datasets it recovers performance, on others it
> makes things worse (Part B can be positive or negative).
> The central research question is: **why and when does row normalization work?**
> To answer this, we propose and evaluate a set of models (LogMagnitude,
> DualStream, SpectralRowNorm, NestedSpheres) that progressively re-introduce
> magnitude and spectral information, and a set of metrics (Fisher score,
> spectral separability, condition numbers, singular value spectra) that
> characterise the geometry of the eigenvector space and predict recovery success.
> The paper provides a controlled, multi-dataset, multi-k empirical analysis
> of exactly where recovery succeeds, where it fails, and why.

**Do not deviate from this narrative without explicit researcher approval.**

---

## PYTHON ENVIRONMENT

Always use: `/home/md724/Spectral-Basis/venv/bin/python`
Never use: `python` or `python3` directly.

---

## KEY FILES

| File | Purpose |
|------|---------|
| `PAPER_EXPERIMENTS/master_training.py` | Runs all training experiments (Exp 1,3–7) |
| `PAPER_EXPERIMENTS/master_analytics.py` | Runs spectral + k-sensitivity analytics (Exp 2, 8) |
| `PAPER_EXPERIMENTS/generate_paper_artifacts_v2.py` | Generates all tables and figures |
| `src/graph_utils.py` | Dataset loading, graph construction, LCC, SGC diffusion, eigenvectors |
| `src/models.py` | All 9 model architectures + training helpers |
| `PAPER_EXPERIMENTS/PAPER_RESULTS/` | All trusted experiment results |
| `PAPER_EXPERIMENTS/PAPER_OUTPUT/` | Generated tables and figures |

---

## CRITICAL CONVENTION: GRAPH MATRICES

`build_graph_matrices()` returns `(adj, L, D)` where L=Laplacian, D=degree matrix.

```python
# CORRECT (master_training.py convention)
adj, L, D = build_graph_matrices(edge_index, num_nodes)
compute_restricted_eigenvectors(X_diffused, L, D, num_components)
```

Never swap L and D. D-orthonormality check: `U.T @ D @ U ≈ I` (error < 1e-6).

**Only trust results from:** `PAPER_EXPERIMENTS/PAPER_RESULTS/`
**Discard all results from earlier investigation runs** (`results/investigation_*/`).

---

## THREE CORE METRICS

```
Part A  = Acc(SGC+MLP) − Acc(Restricted+StandardMLP)   # how much does U hurt?
Part B  = Acc(Restricted+RowNorm) − Acc(Restricted+Std) # how much does RowNorm recover?
Gap     = Part A − Part B                               # what remains unrecovered?
```

- Part A > 0 → SGC beats U (eigenvector-detrimental regime)
- Part A < 0 → U beats SGC (eigenvector-beneficial regime, happens at high k)
- Part B < 0 → RowNorm makes things WORSE (observed on Cora/CiteSeer — this is a finding, not a failure)

**Anomaly awareness**: Before interpreting any result, check whether it is consistent across datasets and k values. If a metric (Part A, Part B, Gap, Fisher, separability, etc.) behaves in an unexpected direction for a specific dataset or k, treat it as a potential anomaly — investigate before reporting. Never average over or silently drop anomalous values; flag them explicitly.

---

## CANONICAL HYPERPARAMETERS

```
HIDDEN_DIM = 256        # hidden layer dimension
LR         = 0.01       # Adam learning rate
WD         = 5e-4       # weight decay
EPOCHS     = 500        # max epochs
PATIENCE   = 100        # early stopping on val_acc
SEEDS      = 15         # training seeds per config
SPLITS     = 1 (fixed) or 5 (random)
k sweep    = [1, 2, 4, 6, 8, 10, 12, 20, 30]
```

Never assume different hyperparameters without checking the code first.

---

## 9 DATASETS

`ogbn-arxiv, WikiCS, Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Coauthor-CS, Coauthor-Physics`

All experiments use the **Largest Connected Component (LCC)**.

---

## 9 MODELS (in order of complexity)

1. **SGC** — linear classifier on Â^k X
2. **StandardMLP** — 3-layer MLP on U, with bias
3. **RowNormMLP** — row-normalizes U at input and each layer, no bias
4. **CosineRowNormMLP** — RowNorm + cosine classifier
5. **LogMagnitudeMLP** — RowNorm + appends log(‖U_i‖)
6. **DualStreamMLP** — separate branches for direction and magnitude
7. **SpectralRowNormMLP** — scales U by λ^α before row-normalizing
8. **NestedSpheresMLP** — α×β sweep (5×5 grid)
9. **NestedSpheresClassifier** — full architecture combining spectral + magnitude

---

## 8 EXPERIMENTS & PAPER SECTIONS

| Exp | What it Measures | Paper Section |
|-----|-----------------|---------------|
| 1 | Part A/B/Gap across all k and all datasets | §3 |
| 2 | Spectral properties of U vs X (singular values, condition numbers, separability) | §4 |
| 3 | Fisher score of ‖U_i‖ — does magnitude carry class info? | §3 |
| 4 | LogMagnitude and DualStream accuracy at k=10 | §5 |
| 5 | Spectral α sweep at k=10 (α ∈ {−1, −0.5, 0, 0.5, 1}) | §5 |
| 6 | NestedSpheres 5×5 α×β sweep at k=10 | §5 |
| 7 | Training curves per epoch at k=10 for all 9 datasets | §6 |
| 8 | Aggregated k-sensitivity summary (loads from Exp 1 output) | §3 |

---

## STRICT RULES

1. Never trust results outside `PAPER_EXPERIMENTS/PAPER_RESULTS/`
2. Always verify D-orthonormality before interpreting U-based results
3. Distinguish: "we observe X" (empirical) vs "X suggests Y" (hypothesis) vs "X proves Y" (theoretical)
4. Never dismiss a 1–3pp difference as noise without checking `test_acc_std`. If difference > 2×std, treat it as real
5. Never smooth over negative results — Part B < 0 is a finding, not a failure
6. Never generate plots without first stating what they are intended to show
7. Never make assumptions about hyperparameters without checking the code
8. Never modify the research question under the guise of "improvement"

---

## NOTATION STANDARD

| Symbol | Meaning |
|--------|---------|
| Â | Normalized adjacency (symmetric, with self-loops) |
| X | Raw node features (n × d) |
| X_diff | Â^k X — diffused features |
| U | Restricted eigenvectors from Rayleigh-Ritz on X_diff (n × d_eff) |
| d_eff | Effective dimension after QR decomposition of X_diff |
| λ_i | Eigenvalues of the generalized problem L_r v = λ D_r v |
| α | Spectral weighting exponent in SpectralRowNormMLP |
| β | Magnitude weighting coefficient in NestedSpheres |
| k | Diffusion depth |
| n | Number of nodes after LCC extraction |
| Part A, Part B, Gap | As defined above — always capitalized |
