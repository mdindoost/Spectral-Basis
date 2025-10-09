"""
Investigation 1: True Eigenvectors
==================================

Usage:
    python experiments/investigation1_true_eigenvectors.py [dataset_name]
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed

Compares 6 different models on true graph eigenvectors.
Expected finding: RowNorm MLP >> Standard MLP (~5-10% improvement)
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    StandardMLP, RowNormMLP, CosineRowNormMLP,
    build_graph_matrices, train_simple, load_dataset
)

# ============================================================================
# Configuration
# ============================================================================
# Get dataset name from command line argument
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ogbn-arxiv'

# Validate dataset name
VALID_DATASETS = ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed']
if DATASET_NAME not in VALID_DATASETS:
    print(f"Error: Invalid dataset '{DATASET_NAME}'")
    print(f"Valid datasets: {', '.join(VALID_DATASETS)}")
    sys.exit(1)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*60)
print(f'INVESTIGATION 1: TRUE EIGENVECTORS - {DATASET_NAME.upper()}')
print('='*60)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('='*60)

# Create dataset-specific output directories
output_base = f'results/investigation1/{DATASET_NAME}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

from utils import load_dataset

(edge_index, node_features, labels, num_nodes, num_classes,
 train_idx, val_idx, test_idx) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Nodes: {num_nodes:,}')
print(f'Edges: {edge_index.shape[1]:,}')
print(f'Classes: {num_classes}')
if node_features is not None:
    print(f'Original features: {node_features.shape[1]}')
print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('Built: adjacency (with self-loops), degree matrix, Laplacian')

# ============================================================================
# 3. Compute True Eigenvectors
# ============================================================================
print('\n[3/6] Computing true eigenvectors...')
k = min(32, 2 * num_classes)
print(f'Solving (D-A)x = λDx for k={k} smallest eigenvalues...')

eigenvalues, eigenvectors = eigsh(
    L.astype(np.float64),
    k=k,
    M=D.astype(np.float64),
    which='SM',
    tol=1e-4
)

print(f'Eigenvalues range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]')

# Verify D-orthonormality
DX = D @ eigenvectors
G = eigenvectors.T @ DX
deviation = np.abs(G - np.eye(k)).max()
print(f'D-orthonormality check: max |X^T D X - I| = {deviation:.2e}')

X = eigenvectors.astype(np.float32)
print(f'Eigenvector matrix shape: {X.shape}')

# ============================================================================
# 4. Prepare Features for Different Models
# ============================================================================
print('\n[4/6] Preparing features for 6 models...')

# Labels
y_train = torch.from_numpy(labels[train_idx]).long().to(device)
y_val = torch.from_numpy(labels[val_idx]).long().to(device)
y_test = torch.from_numpy(labels[test_idx]).long().to(device)

# Model 1: Train-only StandardScaler
scaler = StandardScaler().fit(X[train_idx])
X_train_std = torch.from_numpy(scaler.transform(X[train_idx])).float().to(device)
X_val_std = torch.from_numpy(scaler.transform(X[val_idx])).float().to(device)
X_test_std = torch.from_numpy(scaler.transform(X[test_idx])).float().to(device)

# Model 2: Full-data StandardScaler (includes leakage)
scaler_full = StandardScaler().fit(X)
X_train_full = torch.from_numpy(scaler_full.transform(X[train_idx])).float().to(device)
X_val_full = torch.from_numpy(scaler_full.transform(X[val_idx])).float().to(device)
X_test_full = torch.from_numpy(scaler_full.transform(X[test_idx])).float().to(device)

# Model 3: No scaling
X_train_ns = torch.from_numpy(X[train_idx]).float().to(device)
X_val_ns = torch.from_numpy(X[val_idx]).float().to(device)
X_test_ns = torch.from_numpy(X[test_idx]).float().to(device)

# Model 4: Eigenvalue-weighted
weights = 1.0 / (0.01 + eigenvalues)
X_weighted = (X * weights.reshape(1, -1)).astype(np.float32)
X_train_w = torch.from_numpy(X_weighted[train_idx]).float().to(device)
X_val_w = torch.from_numpy(X_weighted[val_idx]).float().to(device)
X_test_w = torch.from_numpy(X_weighted[test_idx]).float().to(device)

# Models 5-6: Raw eigenvectors (no scaling, RowNorm handles internally)
X_train_rn = torch.from_numpy(X[train_idx]).float().to(device)
X_val_rn = torch.from_numpy(X[val_idx]).float().to(device)
X_test_rn = torch.from_numpy(X[test_idx]).float().to(device)

print('✓ Prepared features for all models')

# ============================================================================
# 5. Train All Models
# ============================================================================
print('\n[5/6] Training 6 models (60 epochs each)...')

hidden_dim = 256
batch_size = 128
epochs = 60

from torch.utils.data import DataLoader, TensorDataset

results = {}

# Model 1: Standard MLP (train-only scaling)
print('\n--- Model 1: Standard MLP (train-only scaling) ---')
train_loader = DataLoader(TensorDataset(X_train_std, y_train), batch_size=batch_size, shuffle=True)
model1 = StandardMLP(k, hidden_dim, num_classes)
results['standard_train_scaled'] = train_simple(
    model1, train_loader, X_val_std, y_val, X_test_std, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 2: Standard MLP (full-data scaling)
print('\n--- Model 2: Standard MLP (full-data scaling) ---')
train_loader = DataLoader(TensorDataset(X_train_full, y_train), batch_size=batch_size, shuffle=True)
model2 = StandardMLP(k, hidden_dim, num_classes)
results['standard_full_scaled'] = train_simple(
    model2, train_loader, X_val_full, y_val, X_test_full, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 3: Standard MLP (no scaling)
print('\n--- Model 3: Standard MLP (no scaling) ---')
train_loader = DataLoader(TensorDataset(X_train_ns, y_train), batch_size=batch_size, shuffle=True)
model3 = StandardMLP(k, hidden_dim, num_classes)
results['standard_no_scale'] = train_simple(
    model3, train_loader, X_val_ns, y_val, X_test_ns, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 4: Standard MLP (eigenvalue-weighted)
print('\n--- Model 4: Standard MLP (eigenvalue-weighted) ---')
train_loader = DataLoader(TensorDataset(X_train_w, y_train), batch_size=batch_size, shuffle=True)
model4 = StandardMLP(k, hidden_dim, num_classes)
results['standard_weighted'] = train_simple(
    model4, train_loader, X_val_w, y_val, X_test_w, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 5: RowNorm MLP
print('\n--- Model 5: RowNorm MLP (Radial) ---')
train_loader = DataLoader(TensorDataset(X_train_rn, y_train), batch_size=batch_size, shuffle=True)
model5 = RowNormMLP(k, hidden_dim, num_classes)
results['rownorm'] = train_simple(
    model5, train_loader, X_val_rn, y_val, X_test_rn, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 6: Cosine Classifier MLP
print('\n--- Model 6: Cosine Classifier MLP ---')
train_loader = DataLoader(TensorDataset(X_train_rn, y_train), batch_size=batch_size, shuffle=True)
model6 = CosineRowNormMLP(k, hidden_dim, num_classes)
results['cosine'] = train_simple(
    model6, train_loader, X_val_rn, y_val, X_test_rn, y_test,
    epochs=epochs, device=device, verbose=True
)

# ============================================================================
# 6. Generate Plots and Save Results
# ============================================================================
print('\n[6/6] Generating plots and saving results...')

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

model_names = [
    'Standard MLP (train-only scaled)',
    'Full-Data Scaled MLP',
    'No-Scaling MLP',
    'Eigenvalue-Weighted MLP',
    'Row-Normalized MLP',
    'Cosine Classifier MLP'
]

model_keys = [
    'standard_train_scaled',
    'standard_full_scaled',
    'standard_no_scale',
    'standard_weighted',
    'rownorm',
    'cosine'
]

# Plot validation accuracy
for name, key in zip(model_names, model_keys):
    axes[0].plot(results[key]['val_accs'], label=name, linewidth=2, alpha=0.85)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_title(f'Validation Accuracy - {DATASET_NAME}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, epochs])

# Plot training loss
for name, key in zip(model_names, model_keys):
    axes[1].plot(results[key]['train_losses'], label=name, linewidth=2, alpha=0.85)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Training Loss')
axes[1].set_title(f'Training Loss - {DATASET_NAME}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, epochs])

plt.tight_layout()
plot_path = f'{output_base}/plots/comparison.png'
plt.savefig(plot_path, dpi=150)
print(f'✓ Saved plot: {plot_path}')

# Save metrics
metrics = {
    'dataset': DATASET_NAME,
    'k_eigenvectors': int(k),
    'epochs': int(epochs),
    'hidden_dim': int(hidden_dim),
    'batch_size': int(batch_size),
    'test_accuracies': {
        key: float(results[key]['test_acc'])
        for key in model_keys
    },
    'final_val_accuracies': {
        key: float(results[key]['val_accs'][-1])
        for key in model_keys
    },
    'best_val_accuracies': {
        key: float(max(results[key]['val_accs']))
        for key in model_keys
    }
}

metrics_path = f'{output_base}/metrics/results.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'✓ Saved metrics: {metrics_path}')

# ============================================================================
# Print Summary
# ============================================================================
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Dataset: {DATASET_NAME}')
print(f'Features: {k} eigenvectors from (D−A)x = λ D x (D-orthonormal)')
print(f'Hidden dimension: {hidden_dim}')
print(f'Training epochs: {epochs}')
print(f'Batch size: {batch_size}')

model_configs = [
    ('Standard MLP (train-only scaling)', 'standard_train_scaled', model1),
    ('Full-Data Scaled MLP (all-data scaling - includes leakage)', 'standard_full_scaled', model2),
    ('No-Scaling MLP (raw eigenvectors)', 'standard_no_scale', model3),
    ('Eigenvalue-Weighted MLP (1/(0.01+λ))', 'standard_weighted', model4),
    ('Row-Normalized MLP (Radial)', 'rownorm', model5),
    ('Cosine Classifier MLP (Angular + Learnable Temperature)', 'cosine', model6),
]

for i, (name, key, model) in enumerate(model_configs, 1):
    num_params = sum(p.numel() for p in model.parameters())
    final_val_acc = results[key]['val_accs'][-1]
    best_val_acc = max(results[key]['val_accs'])
    test_acc = results[key]['test_acc']
    
    print(f'\n{i}. {name}:')
    print(f'   - Params: {num_params:,}')
    print(f'   - Final Val Acc: {final_val_acc:.4f}')
    print(f'   - Best  Val Acc: {best_val_acc:.4f}')
    print(f'   - Test  Acc: {test_acc:.4f}')
    
    # For Cosine Classifier, also print learned scale
    if key == 'cosine':
        print(f'   - Learned Scale: {model6.scale.item():.4f}')

print('\n' + '='*60)
print('COMPARISON (Relative to Standard MLP)')
print('='*60)

baseline = results['standard_train_scaled']['test_acc']
comparison_models = [
    ('Full-Data Scaled vs Standard', 'standard_full_scaled'),
    ('No-Scaling vs Standard', 'standard_no_scale'),
    ('Eigenvalue-Weighted vs Standard', 'standard_weighted'),
    ('Row-Normalized vs Standard', 'rownorm'),
    ('Cosine Classifier vs Standard', 'cosine'),
]

for name, key in comparison_models:
    diff = results[key]['test_acc'] - baseline
    pct = 100 * (results[key]['test_acc'] / baseline - 1)
    print(f'{name:38s}: {diff:+.4f} ({pct:+.1f}%)')

print('='*60)
print(f'✓ Investigation 1 complete for {DATASET_NAME}!')