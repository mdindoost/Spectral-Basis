# AI Coding Agent Instructions for Spectral-Basis

## Project Overview

**Spectral-Basis** is a research codebase investigating how MLPs respond to different spectral basis representations of graph-structured data. The core research question: *When and why does using graph spectral properties (eigenvectors) as input features improve MLP performance?*

### Key Insight
The project tests whether restricted eigenvectors (eigenvectors of the Laplacian projected onto the feature space) and diffusion techniques can improve graph neural network performance, particularly when combined with RowNorm MLPs (neural networks with row normalization).

---

## Architecture & Data Flow

### Three Investigation Tracks

1. **Investigation 1** (`experiments/investigation1_*.py`)
   - **Question**: How do MLPs respond to *true graph eigenvectors* (computed via eigendecomposition)?
   - **Pattern**: Compare StandardMLP vs RowNormMLP on eigenvector inputs
   - **Expected Result**: RowNorm should outperform Standard by 5-10%
   - **Key Files**: `investigation1_true_eigenvectors.py`, `investigation1_true_eigenvectors_extended.py`

2. **Investigation 2** (`experiments/investigation2_*.py`)
   - **Question**: Do restricted eigenvectors (same span as original features but different basis) improve performance?
   - **Sub-branches**:
     - **Directions A&B**: Compare StandardMLP(X) vs StandardMLP(V) vs RowNormMLP(V), where V are restricted eigenvectors
     - **Diffused Engineered**: Apply diffusion (A^k @ X) before computing eigenvectors
     - **Random Subspaces**: Control experiment comparing against random orthonormal bases
   - **Key Files**: `investigation2_restricted_eigenvectors.py`, `investigation2_diffused_engineered.py`, `investigation2_random_subspaces.py`

3. **SGC Comparison** (`experiments/investigation_sgc_*.py`)
   - **Question**: How do these spectral representations compare to Simplified Graph Convolution?
   - **Variants**: standard aggregation, row normalization, truncated versions
   - **Key Files**: `investigation_sgc_comparison.py`, `investigation_sgc_rownorm.py`

### Data Processing Pipeline
```
Load Dataset (8 graphs: ogbn-arxiv, Cora, CiteSeer, PubMed, WikiCS, Amazon-Photo, Amazon-Computers, Coauthor-{CS,Physics})
    ↓
Build Graph Matrices (adjacency A, Laplacian L = D - A)
    ↓
Compute Spectral Features (eigenvectors, diffused features, or random bases)
    ↓
Normalize Features (StandardScaler per dataset)
    ↓
Train & Evaluate MLPs (3 architectures: StandardMLP, RowNormMLP, CosineRowNormMLP)
    ↓
Aggregate Results (mean/std across random seeds and splits)
    ↓
Correlate with Eigenvalue Properties (condition number, spectral gaps, etc.)
```

---

## Project-Specific Conventions & Patterns

### 1. **Model Architecture Variants** (`experiments/utils.py`)

Three core MLP classes with specific design patterns:

```python
# (a) StandardMLP: Traditional MLP with bias
class StandardMLP(nn.Module):
    fc1 = Linear(d_in, hidden)      # WITH bias
    fc2 = Linear(hidden, hidden)
    fc3 = Linear(hidden, d_out)
    forward: ReLU → ReLU → identity

# (b) RowNormMLP: Radial+Angular decomposition (NO bias)
class RowNormMLP(nn.Module):
    fc1 = Linear(d_in, hidden, bias=False)
    forward: normalize(x,2) → ReLU → normalize → ... 
    # Key: F.normalize(x, p=2, dim=1) before EVERY layer

# (c) CosineRowNormMLP: RowNorm + Cosine classifier
class CosineRowNormMLP(nn.Module):
    # Like RowNorm but final layer uses:
    # w_normalized = F.normalize(fc3.weight, p=2, dim=1)
    # logits = scale * (x @ w_normalized.T)
```

**When to use each**: 
- StandardMLP = baseline/control
- RowNormMLP = main hypothesis model (MLPs on spectral basis prefer normalization)
- CosineRowNormMLP = variant, used in specific ablations

### 2. **Dataset Handling** (`load_dataset` in `utils.py`)

Supports 9 datasets with inconsistent split conventions:

| Dataset | Type | Split Type | Notes |
|---------|------|-----------|-------|
| ogbn-arxiv | OGB | Official (train/val/test) | ~170k nodes, ~1.2M edges |
| Cora, CiteSeer, PubMed | Planetoid | Built-in masks | Small (<10k nodes) |
| WikiCS | PyG | 20 different splits, use split[0] | Must convert to undirected |
| Amazon-Photo, Amazon-Computers | PyG | None provided | Create 60/20/20 split, seed=42 |
| Coauthor-CS, Coauthor-Physics | PyG | None provided | Create 60/20/20 split, seed=42 |

**Pattern**: Always call `to_undirected()` for datasets that have directed edges; normalize node features with StandardScaler after loading.

### 3. **Restricted Eigenvector Computation** (Key Innovation)

The restricted eigenvector approach is central to Investigation 2:

```python
# Given: X (raw features), L (Laplacian)
# Goal: Find eigenvectors in span(X)

# Method: Galerkin projection
# 1. Project L into feature space: L_proj = X.T @ L @ X
# 2. Solve small eigendecomposition on L_proj
# 3. Back-project to get U (restricted eigenvectors)
U, eigenvals = eigh(L_proj)
V = X @ U  # V spans same space as X but has eigenvector properties
```

**Why**: This decouples the benefit of spectral structure from full-rank eigenvector computation—tests if MLPs prefer the *restricted* eigenvector direction/basis over raw features.

### 4. **Training Function Variants** (`utils.py`)

Two training patterns exist:

```python
# (A) train_simple: Runs epochs, returns final test accuracy
# - Used in: Investigation 1
# - Stops after fixed epochs, no validation-based selection

# (B) train_with_selection: Tracks best val loss AND best val accuracy
# - Used in: Investigation 2 (main)
# - Returns BOTH test@best_val_loss and test@best_val_acc
# - Critical: Use test@best_val_acc for final comparisons
```

**Pattern**: Most experiments use `train_with_selection` with 200 epochs, LR=0.01, weight_decay=5e-4, hidden_dim=256.

### 5. **Experiment Naming & Result Organization**

Directory structure follows strict convention:

```
results/
├── investigation1/{dataset}/              # True eigenvectors
│   ├── plots/                             # Performance curves
│   └── metrics/results_{seed}.json
├── investigation2/{dataset}/              # Restricted eigenvectors
├── investigation2_diffused_engineered/{dataset}/  # + Diffusion
├── investigation2_directions_AB/{dataset}/       # Main comparison
├── investigation2_random_subspaces/{dataset}/    # Control
├── investigation_sgc_comparison/{dataset}/       # vs SGC
└── eigenvalue_analysis/                  # Correlation with condition number
```

**Aggregation Pattern**: Each experiment runs multiple seeds (3-5) and multiple random splits (3-5), then aggregates with mean/std in a file like:
- `results_aggregated.json` (central results)
- `results_{seed}_split{i}.json` (per-seed raw results)

### 6. **Diffusion Patterns** (Investigation 2)

Two diffusion types implemented:

```python
# Standard Low-Pass Diffusion
X_diffused = A**k @ X  # Aggregate k-hop neighborhood features
# Intuition: Smooth features in graph space

# CMG High-Pass Diffusion (Complementary)
X_cmg = A**(k+1) @ X - A**k @ X  # Difference of diffusion levels
# Intuition: Extract high-frequency (boundary) information
```

**Typical sweep**: k ∈ {2, 4, 8} for standard; k ∈ {4, 8, 16} for CMG. Always try multiple k values—results are sensitive to diffusion strength.

### 7. **Eigenvalue Analysis** (`correlate_eigenvalues_performance.py`)

Research linking spectral properties to performance:

```python
# For each dataset, compute:
- Condition number κ = λ_max / λ_min (via SVD of L)
- Spectral gap: gap_ratio = (λ_2 - λ_1) / λ_1  # algebraic connectivity
- Rank: effective rank from singular values

# Then correlate against:
- Baseline restricted eigenvector performance
- Diffusion improvement (absolute and relative)
```

**Key Finding Pattern**: Often find negative correlation between condition number and baseline accuracy (ill-conditioned → poor baseline → large diffusion benefit).

---

## Critical Developer Workflows

### Running a Single Experiment

```bash
# Investigation 1: True eigenvectors
cd /home/md724/Spectral-Basis
python experiments/investigation1_true_eigenvectors.py cora

# Investigation 2: Restricted eigenvectors (main)
python experiments/investigation2_restricted_eigenvectors.py pubmed

# Investigation 2 with diffusion
python experiments/investigation2_diffused_engineered.py ogbn-arxiv --k-standard 2,4,8 --k-cmg 4,8,16

# SGC comparison
python experiments/investigation_sgc_comparison.py amazon-computers
```

### Batch Processing All Datasets

```bash
# From workspace root
bash scripts/run_random_all_datasets.sh  # Runs all experiments on all datasets

# Or manually loop:
for dataset in ogbn-arxiv cora citeseer pubmed wikics amazon-photo amazon-computers coauthor-cs coauthor-physics; do
    python experiments/investigation2_diffused_engineered.py $dataset
done
```

### Analyzing Results

```bash
# Summarize Investigation 2 Directions A&B results
python scripts/summarize_results.py

# Correlate eigenvalues with performance
python experiments/correlate_eigenvalues_performance.py

# Comprehensive ablation analysis
python scripts/analyze_extended_ablations.py

# Generate final figures and tables
python scripts/make_figs_and_tables.py
```

### PyTorch Compatibility Issue (Important!)

All experiments include this compatibility patch at the top:

```python
# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

**Why**: PyTorch 2.6+ added `weights_only=True` default in `torch.load()`, which breaks OGB dataset loading. This patch must be included in any new experiment files.

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Key packages:
# torch>=2.0.0, torch-geometric>=2.3.0, ogb>=1.3.6
# scipy, scikit-learn, numpy, matplotlib
```

---

## Integration Points & Dependencies

### External Data Sources
- **OGB (Open Graph Benchmark)**: `NodePropPredDataset('ogbn-arxiv')` - automatically downloads to `dataset/ogbn_arxiv/`
- **PyG Planetoid**: `Planetoid(name='Cora')` - downloads to `dataset/Cora/`
- **PyG Large Graphs**: WikiCS, Amazon, Coauthor - download on first load

### Cross-Component Dependencies

| Component | Depends On | Note |
|-----------|-----------|------|
| All experiments | `utils.py` | Must import models, dataset loader, training functions |
| Investigation 2 | `scipy.linalg.eigh()` | Restricted eigenvector computation |
| SGC experiments | `torch_geometric.transforms` | For graph convolution operations |
| Analysis scripts | `results/` JSON outputs | Aggregated metrics from experiments |

### Key Output Files Referenced

- `results/eigenvalue_analysis/data/eigenvalue_analysis_complete.json` - Input to correlation analysis
- `results/investigation2_diffused_engineered/*/summary/performance_comparison.json` - For result summaries
- `results/investigation2_directions_AB/*/metrics/results_aggregated.json` - Main Investigation 2 results

---

## Common Patterns & Anti-Patterns

### ✅ DO

- **Use `train_with_selection` for new experiments** - provides both best_val_loss and best_val_acc tracking
- **Always normalize features with StandardScaler** - even if load_dataset handles it, double-check
- **Run with multiple seeds (≥3)** - results have variance, aggregate with mean±std
- **Test on all 9 datasets** - conclusions must hold across diverse graph types
- **Sweep hyperparameters per dataset** - what works for ogbn-arxiv may not work for small graphs like Cora
- **Save intermediate results** - experiments are long; checkpoint after each seed

### ❌ DON'T

- **Forget row normalization in RowNormMLP layers** - F.normalize() must happen before AND after linear layers
- **Use test set for model selection** - only use validation metrics for early stopping/checkpointing
- **Mix directed and undirected graphs** - always call `to_undirected()` on directed datasets
- **Hardcode output paths** - use f'results/{experiment_name}/{dataset}/' format
- **Skip eigenvalue analysis** - condition number often explains why an experiment fails/succeeds
- **Ignore the PyTorch compatibility patch** - older PyTorch versions may break; new ones need the patch

### ⚠️ Gotchas

1. **WikiCS has 20 splits** - code defaults to split[0]; explicitly document if using different split
2. **Amazon & Coauthor have no official splits** - must create train/val/test; seed=42 for reproducibility
3. **Small datasets (Cora, CiteSeer)** - eigenvector computation can be numerically unstable; may need regularization
4. **Diffusion is expensive** - A^k @ X requires k sparse matrix multiplications; for large k use power iteration
5. **Model selection matters** - test@best_val_acc often differs from test@best_val_loss by 2-5%; report both
6. **Restricted eigenvectors are rank-deficient** - rank = min(d_raw, num_features); can cause issues in downstream analysis

---

## Documentation & References

- **README.md**: High-level project motivation and citation
- **Investigation Files**: Each has detailed docstring explaining the research question and expected findings
- **utils.py**: Comprehensive dataset loading logic with comments on split handling per dataset
- **Experiment Output**: JSON files in `results/` contain `experiment_config`, `model_hyperparameters`, and `results_summary`

---

## Getting Productive

To contribute a new analysis:

1. **Understand the research question** - read relevant investigation file's docstring
2. **Check if data exists** - review `results/` for intermediate files you can reuse
3. **Follow the naming convention** - name experiment `investigation{N}_{hypothesis}.py`
4. **Include PyTorch patch** - copy from existing file to maintain compatibility
5. **Use `load_dataset` for all 9 datasets** - ensures consistency
6. **Use `train_with_selection`** - standard training pattern in this codebase
7. **Aggregate results properly** - mean/std over 3+ seeds and 3+ splits
8. **Document findings in JSON** - include `experiment_config` and `results_summary` for downstream analysis
9. **Test on small dataset first** - Cora is fast; validate logic before running on ogbn-arxiv (takes hours)
10. **Commit results.** - the `/results` folder should be version controlled for reproducibility

Good luck! The spectral basis hypothesis is fascinating—look for that "aha" moment when RowNorm suddenly clicks on the eigenvector basis. 🎯
