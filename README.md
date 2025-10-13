# MLP Sensitivity to Spectral Basis Representations

Research investigating how MLPs respond to different spectral basis representations of graph-structured data.



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


---

### Investigation 2: X-Restricted Eigenvectors


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


---

### Investigation 2 (ogbn-arxiv)

---

## Key Findings


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