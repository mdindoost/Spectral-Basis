# MLP Sensitivity to Spectral Basis Representations

Research investigating how MLPs respond to different spectral basis representations of graph-structured data.



---

## Research Question

**Do MLPs exhibit different performance when node features are transformed to span-equivalent but basis-distinct representations?**

This project demonstrates that **basis choice matters**, not just information content. Even when two feature representations span the same subspace mathematically, MLPs achieve different performance depending on the basis used.

---

## Repository Structure

```
Spectral-Basis/
├── experiments/
│   ├── investigation1_true_eigenvectors.py    # Compare 6 models on true eigenvectors
│   ├── investigation2_restricted_eigenvectors.py # Compare X vs U (same span)
│   └── utils.py                                # Shared utilities & models
├── results/
│   ├── investigation1/
│   │   ├── ogbn-arxiv/    # Results per dataset
│   │   ├── cora/
│   │   ├── citeseer/
│   │   └── pubmed/
│   └── investigation2/
│       ├── ogbn-arxiv/
│       ├── cora/
│       ├── citeseer/
│       └── pubmed/
├── dataset/               # Downloaded datasets (auto-created)
├── run_all.sh            # Script to run all experiments
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/mdindoost/Spectral-Basis.git
cd Spectral-Basis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have issues with `torch-geometric`, specify your CUDA version:

```bash
# Check your PyTorch and CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Install PyG with matching CUDA version
pip install torch-geometric
```

### 4. Verify Installation

```bash
python -c "import torch, torch_geometric, ogb; print('✓ All dependencies installed')"
```

---

## Investigations

### Investigation 1: True Eigenvectors

**Purpose**: Demonstrate that RowNorm MLP exploits geometric properties of graph eigenvectors.

**Method**:
- Compute true graph eigenvectors: `eigsh(L, k=2×num_classes, M=D)`
- Small dimension (k=14 for Cora, k=32 for ogbn-arxiv)
- Compare 6 models/preprocessing approaches:
  1. Standard MLP (train-only StandardScaler)
  2. Standard MLP (full-data StandardScaler - with leakage)
  3. Standard MLP (no scaling)
  4. Standard MLP (eigenvalue-weighted)
  5. **RowNorm MLP** (no scaling, L2 normalization, no bias)
  6. Cosine Classifier MLP (angular distance)

**Key Finding**: RowNorm MLP significantly outperforms Standard MLP (~30-40% relative improvement)

**Why it matters**: Shows that row-normalization effectively exploits spectral structure.

---

### Investigation 2: X-Restricted Eigenvectors

**Purpose**: Test if basis representation matters when information content is identical.

**Method**:
- Solve restricted eigenproblem: `(X^T L X)v = λ(X^T D X)v` → `U = X @ V`
- **Critical insight**: `span(U) = span(X)` — same information, different basis
- Full dimension (k=128 for ogbn-arxiv, k=1433 for Cora)
- Compare 2 models:
  - **Model A**: Raw X with train-only StandardScaler → Standard MLP
  - **Model B**: Restricted eigenvectors U (no scaling) → RowNorm MLP

**Key Finding**: Small but real improvement (~1-3%) for RowNorm on U

**Why it matters**: Demonstrates that **MLPs are sensitive to basis choice**, not just information content. Even with identical spans, different bases yield different performance.

---

## Running Experiments

### Quick Start: Run Single Dataset

```bash
# Investigation 1 on Cora (fastest, ~2 minutes)
python experiments/investigation1_true_eigenvectors.py cora

# Investigation 2 on Cora (~15 minutes)
python experiments/investigation2_restricted_eigenvectors.py cora
```

### Run Specific Dataset

```bash
# Investigation 1
python experiments/investigation1_true_eigenvectors.py [dataset]

# Investigation 2  
python experiments/investigation2_restricted_eigenvectors.py [dataset]

# Available datasets: ogbn-arxiv, cora, citeseer, pubmed
```

### Run All Experiments (All Datasets)

```bash
# Make script executable
chmod +x run_all.sh

# Run both investigations on all 4 datasets (~2-3 hours total)
./run_all.sh
```

---

## Supported Datasets

| Dataset | Nodes | Edges | Classes | Features | k (Inv 1) | Train/Val/Test |
|---------|-------|-------|---------|----------|-----------|----------------|
| **ogbn-arxiv** | 169,343 | 1,166,243 | 40 | 128 | 32 | 90K/30K/49K |
| **Cora** | 2,708 | 10,556 | 7 | 1,433* | 14 | 140/500/1K |
| **CiteSeer** | 3,327 | 9,104 | 6 | 3,703* | 12 | 120/500/1K |
| **PubMed** | 19,717 | 88,648 | 3 | 500 | 6 | 60/500/1K |

*Cora and CiteSeer use bag-of-words features which are rank-deficient (see [Investigation 2 Notes](#investigation-2-technical-notes))

**k = min(32, 2 × num_classes)** for Investigation 1

---

## Results Organization

Results are automatically saved to dataset-specific directories:

```
results/
├── investigation1/
│   └── [dataset]/
│       ├── plots/
│       │   └── comparison.png          # Validation acc & training loss curves
│       └── metrics/
│           └── results.json            # Test accuracies & all metrics
└── investigation2/
    └── [dataset]/
        ├── plots/
        │   └── comparison.png          # 2×3 grid: val acc, train loss, val loss
        └── metrics/
            └── results.json            # Test acc at best checkpoints
```

---

## Example Results

### Investigation 1 (ogbn-arxiv)

```
============================================================
SUMMARY
============================================================
Dataset: ogbn-arxiv
Features: 32 eigenvectors from (D−A)x = λ D x (D-orthonormal)

1. Standard MLP (train-only scaling):
   - Params: 84,520
   - Test  Acc: 0.4310

5. Row-Normalized MLP (Radial):
   - Params: 83,968
   - Test  Acc: 0.6006    ← +39.4% improvement!

============================================================
COMPARISON (Relative to Standard MLP)
============================================================
Row-Normalized vs Standard:     +0.1696 (+39.4%)
============================================================
```

**Interpretation**: RowNorm MLP dramatically outperforms Standard MLP on true eigenvectors because it exploits the geometric structure through radial projections.

---

### Investigation 2 (ogbn-arxiv)

```
============================================================
RESULTS - OGBN-ARXIV
============================================================
Standard MLP on X (scaled):
  best val loss  @ epoch  89: test acc = 0.7182 (val loss = 0.8534)
  best val acc   @ epoch 156: test acc = 0.7246 (val acc  = 0.7368)

RowNorm MLP on U (restricted eigenvectors):
  best val loss  @ epoch  94: test acc = 0.7305 (val loss = 0.8182)
  best val acc   @ epoch 163: test acc = 0.7402 (val acc  = 0.7455)

============================================================
COMPARISON (U vs X):
At best val loss checkpoint: +0.0123 (+1.7%)
At best val acc checkpoint:  +0.0156 (+2.2%)
============================================================
```

**Interpretation**: Even though span(U) = span(X) (identical information), the eigenvector basis (U) with RowNorm MLP achieves 1-2% better performance than raw features (X) with Standard MLP. This proves **basis choice matters**.

---

## Key Findings

### Main Result

**MLPs are sensitive to basis representation, not just information content.**

Even when two feature matrices span the same subspace (contain identical information), the choice of basis affects model performance:
- Investigation 1: ~30-40% improvement with RowNorm MLP on true eigenvectors
- Investigation 2: ~1-3% improvement with RowNorm MLP on restricted eigenvectors (same span as raw features)

### Why This Matters

**Practical Implications**:
- StandardScaler (common practice) may not be optimal preprocessing
- Spectral basis transformation can improve MLP performance
- Row-normalization is effective for spectral features

**Theoretical Implications**:
- Basis choice creates different optimization landscapes
- Geometric structure (radial vs. Cartesian) affects learning
- Information content ≠ representational quality

---

## Investigation 2: Technical Notes

### Rank-Deficient Features

**Cora and CiteSeer** use sparse bag-of-words features that are rank-deficient:

```
Cora:     1,433 nominal features → ~988 effective rank
CiteSeer: 3,703 nominal features → ~2,500 effective rank (estimated)
```

**Why this happens**: Many words never appear in the corpus, creating linearly dependent columns.

**How we handle it**: Added small regularization (1e-8 scale) to ensure numerical stability:

```python
reg = 1e-8 * trace(Dr) / d
Dr_regularized = Dr + reg * I
```

**Does this affect validity?** 
- ✅ No! The regularization is tiny (~0.00001% of eigenvalues)
- ✅ span(U) = span(X) still holds (just in lower-dimensional subspace)
- ✅ Both models see identical information
- ✅ Comparison remains scientifically valid

Expected warning for Cora:
```
⚠ Warning: X is rank-deficient (988 < 1433)
  The restricted eigenvectors span a 988-dimensional subspace.
```

This is **normal and expected** for bag-of-words features!

---

## Model Architectures

### Standard MLP
```python
class StandardMLP(nn.Module):
    # 3-layer MLP with bias and ReLU
    # Input → [Linear+bias → ReLU] → [Linear+bias → ReLU] → [Linear+bias] → Output
```

**Preprocessing**: StandardScaler (zero mean, unit variance per feature)

**Characteristics**:
- Works in Cartesian coordinates
- Requires proper scaling for training stability
- Standard ML practice

---

### RowNorm MLP (Radial)
```python
class RowNormMLP(nn.Module):
    # 3-layer MLP without bias, L2-normalized between layers
    # Input → [Normalize → Linear (no bias) → ReLU] × 3 → Output
```

**Preprocessing**: None (uses raw features)

**Characteristics**:
- Works in polar/radial coordinates (unit hypersphere)
- F.normalize(x, p=2, dim=1) projects to unit sphere
- No bias terms (incompatible with radial geometry)
- Exploits angular structure

---

## Training Configuration

### Investigation 1
- **Epochs**: 60
- **Batch size**: 128
- **Optimizer**: Adam (lr=0.01, weight_decay=5e-4)
- **Hidden dimension**: 256
- **Evaluation**: Final test accuracy

### Investigation 2
- **Epochs**: 200 (longer for convergence)
- **Batch size**: 128
- **Optimizer**: Adam (lr=0.01, weight_decay=5e-4)
- **Hidden dimension**: 256
- **Model selection**: Best by validation loss AND validation accuracy
- **Evaluation**: Test accuracy at both checkpoints

---

## Troubleshooting

### torch-geometric Installation Issues

```bash
# Option 1: Simple install (usually works)
pip install torch-geometric

# Option 2: Install with dependencies
pip install torch-scatter torch-sparse torch-geometric

# Option 3: Specify CUDA version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Memory Issues

If you run out of GPU memory on large datasets:

```bash
# Reduce batch size in the script (line ~200)
batch_size = 64  # instead of 128
```

### Dataset Download Fails

Datasets download automatically on first run. If download fails:

```bash
# Delete partial download and retry
rm -rf dataset/[dataset_name]
python experiments/investigation1_true_eigenvectors.py [dataset_name]
```

### Investigation 2 Fails with LinAlgError

This should be fixed with regularization. If you still see errors:

1. Check that you're using the updated `investigation2_restricted_eigenvectors.py`
2. The dataset might have extremely rank-deficient features
3. Try Investigation 1 instead, which works for all datasets

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{spectral-basis-mlp-2024,
  author = {Dindoost, Mohammad},
  title = {MLP Sensitivity to Spectral Basis Representations},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mdindoost/Spectral-Basis}}
}
```

---

## Contact

**Mohammad Dindoost**  
PhD Student, New Jersey Institute of Technology  
Email: md724@njit.edu  
GitHub: [@mdindoost](https://github.com/mdindoost)


---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **OGB Team**: For the ogbn-arxiv dataset
- **PyTorch Geometric Team**: For Planetoid datasets (Cora, CiteSeer, PubMed)
- **Professor Ioannis Koutis**: For research guidance and insights

---

## Quick Reference

### Run Everything
```bash
./run_all.sh
```

### Run Individual Experiments
```bash
# Fastest test (Cora, Investigation 1, ~2 min)
python experiments/investigation1_true_eigenvectors.py cora

# Slower test (ogbn-arxiv, Investigation 2, ~40 min)
python experiments/investigation2_restricted_eigenvectors.py ogbn-arxiv
```

### View Results
```bash
# List all results
ls results/investigation1/*/metrics/results.json
ls results/investigation2/*/metrics/results.json

# View specific result
cat results/investigation1/cora/metrics/results.json | python -m json.tool
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

**Last Updated**: Oct 9 - 2025