"""
Investigation 2: X-Restricted Eigenvectors
==========================================

Usage:
    python experiments/investigation2_restricted_eigenvectors.py [dataset_name]
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed

Compares raw X vs restricted eigenvectors U (same span, different basis).
Expected finding: Small improvement (~1-3%) for RowNorm on U
"""

import os
import sys
import json
import torch
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    StandardMLP, RowNormMLP,
    build_graph_matrices, train_with_selection, load_dataset
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
print(f'INVESTIGATION 2: X-RESTRICTED EIGENVECTORS - {DATASET_NAME.upper()}')
print('='*60)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('='*60)

# Create dataset-specific output directories
output_base = f'results/investigation2/{DATASET_NAME}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx, val_idx, test_idx) = load_dataset(DATASET_NAME, root='./dataset')

if X_raw is None:
    print(f"ERROR: Dataset {DATASET_NAME} has no node features!")
    print("Investigation 2 requires node features to compute X-restricted eigenvectors.")
    sys.exit(1)

d_raw = X_raw.shape[1]

print(f'Nodes: {num_nodes:,}')
print(f'Edges: {edge_index.shape[1]:,}')
print(f'Classes: {num_classes}')
print(f'Raw feature dimension: {d_raw}')
print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('Built: adjacency (with self-loops), degree matrix, Laplacian')

# ============================================================================
# 3. Compute X-Restricted Eigenvectors
# ============================================================================
print('\n[3/6] Computing X-restricted eigenvectors...')
print(f'Projecting Laplacian into span(X) (dimension {d_raw})...')

X = X_raw.astype(np.float64)

# Project L and D into span(X)
LX = L @ X
DX = D @ X

Lr = X.T @ LX  # shape: (d_raw, d_raw)
Dr = X.T @ DX  # shape: (d_raw, d_raw)

# Symmetrize for numerical stability
Lr = 0.5 * (Lr + Lr.T)
Dr = 0.5 * (Dr + Dr.T)

print(f'Solving reduced generalized eigenproblem in R^{d_raw}...')

# Add small regularization to ensure Dr is positive definite
# This handles cases where X has linearly dependent columns
reg = 1e-8 * np.trace(Dr) / d_raw
Dr_reg = Dr + reg * np.eye(d_raw)

print(f'Added regularization: {reg:.2e} * I')

try:
    w, V = la.eigh(Lr, Dr_reg)  # Dense eigensolver with regularized Dr
    idx = np.argsort(w)
    w = w[idx]
    V = V[:, idx]
except np.linalg.LinAlgError as e:
    print(f'ERROR: Eigensolver failed even with regularization: {e}')
    print(f'This likely means the feature matrix has very low rank.')
    print(f'Try Investigation 1 (true eigenvectors) instead for {DATASET_NAME}.')
    sys.exit(1)

print(f'Eigenvalues range: [{w.min():.6f}, {w.max():.6f}]')

# Map back to node space
U = (X @ V).astype(np.float32)
print(f'Restricted eigenvector matrix U shape: {U.shape}')

# Check effective rank
from numpy.linalg import matrix_rank
rank_X = matrix_rank(X, tol=1e-6)
rank_U = matrix_rank(U, tol=1e-6)
print(f'Rank of X: {rank_X}/{d_raw}')
print(f'Rank of U: {rank_U}/{d_raw}')

if rank_X < d_raw:
    print(f'⚠ Warning: X is rank-deficient ({rank_X} < {d_raw})')
    print(f'  The restricted eigenvectors span a {rank_X}-dimensional subspace.')

print(f'✓ span(U) = span(X) — same subspace, different basis')

# Verify D-orthonormality of U
DU = (D @ U).astype(np.float64)
G = U.astype(np.float64).T @ DU
deviation = np.abs(G - np.eye(U.shape[1])).max()
print(f'D-orthonormality check: max |U^T D U - I| = {deviation:.2e}')

if deviation > 1e-4:
    print(f'⚠ Warning: Large D-orthonormality deviation (may be due to regularization or rank deficiency)')

# ============================================================================
# 4. Prepare Data
# ============================================================================
print('\n[4/6] Preparing data for two models...')

# Labels
y_train = torch.from_numpy(labels[train_idx]).long().to(device)
y_val = torch.from_numpy(labels[val_idx]).long().to(device)
y_test = torch.from_numpy(labels[test_idx]).long().to(device)

# Model A: Raw X with train-only StandardScaler
scaler = StandardScaler().fit(X_raw[train_idx])
X_train_std = torch.from_numpy(scaler.transform(X_raw[train_idx])).float().to(device)
X_val_std = torch.from_numpy(scaler.transform(X_raw[val_idx])).float().to(device)
X_test_std = torch.from_numpy(scaler.transform(X_raw[test_idx])).float().to(device)

# Model B: Restricted eigenvectors U (no scaling)
U_train = torch.from_numpy(U[train_idx]).float().to(device)
U_val = torch.from_numpy(U[val_idx]).float().to(device)
U_test = torch.from_numpy(U[test_idx]).float().to(device)

print('✓ Prepared features for both models')

# ============================================================================
# 5. Train Both Models
# ============================================================================
print('\n[5/6] Training 2 models (200 epochs each)...')

hidden_dim = 256
batch_size = 128
epochs = 200

from torch.utils.data import DataLoader, TensorDataset

# Model A: Standard MLP on X (scaled)
print('\n--- Model A: Standard MLP on Raw X (scaled) ---')
train_loader_X = DataLoader(TensorDataset(X_train_std, y_train), batch_size=batch_size, shuffle=True)
model_X = StandardMLP(d_raw, hidden_dim, num_classes)
res_X = train_with_selection(
    model_X, train_loader_X, X_val_std, y_val, X_test_std, y_test,
    epochs=epochs, lr=1e-2, weight_decay=5e-4, device=device, verbose=True
)

# Model B: RowNorm MLP on U (restricted eigenvectors)
print('\n--- Model B: RowNorm MLP on Restricted Eigenvectors U ---')
train_loader_U = DataLoader(TensorDataset(U_train, y_train), batch_size=batch_size, shuffle=True)
model_U = RowNormMLP(d_raw, hidden_dim, num_classes)
res_U = train_with_selection(
    model_U, train_loader_U, U_val, y_val, U_test, y_test,
    epochs=epochs, lr=1e-2, weight_decay=5e-4, device=device, verbose=True
)

# ============================================================================
# 6. Generate Plots and Save Results
# ============================================================================
print('\n[6/6] Generating plots and saving results...')

# Create 2x3 grid plot
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle(f'Investigation 2: X-Restricted Eigenvectors - {DATASET_NAME}', fontsize=14)

# Row 1: Standard MLP on X (scaled)
axes[0, 0].plot(res_X['val_accs'], linewidth=2)
axes[0, 0].set_title('X-scaled: Val Acc')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axvline(res_X['best_epoch_acc'], color='k', linestyle='--', alpha=0.5)

axes[0, 1].plot(res_X['train_losses'], linewidth=2)
axes[0, 1].set_title('X-scaled: Train Loss')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].plot(res_X['val_losses'], linewidth=2)
axes[0, 2].set_title('X-scaled: Val Loss')
axes[0, 2].grid(alpha=0.3)
axes[0, 2].axvline(res_X['best_epoch_loss'], color='k', linestyle='--', alpha=0.5)

# Row 2: RowNorm MLP on U (restricted eigenvectors)
axes[1, 0].plot(res_U['val_accs'], linewidth=2)
axes[1, 0].set_title('U-full: Val Acc')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axvline(res_U['best_epoch_acc'], color='k', linestyle='--', alpha=0.5)

axes[1, 1].plot(res_U['train_losses'], linewidth=2)
axes[1, 1].set_title('U-full: Train Loss')
axes[1, 1].grid(alpha=0.3)

axes[1, 2].plot(res_U['val_losses'], linewidth=2)
axes[1, 2].set_title('U-full: Val Loss')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].axvline(res_U['best_epoch_loss'], color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plot_path = f'{output_base}/plots/comparison.png'
plt.savefig(plot_path, dpi=150)
print(f'✓ Saved plot: {plot_path}')

# Save metrics
metrics = {
    'dataset': DATASET_NAME,
    'feature_dimension': int(d_raw),
    'epochs': int(epochs),
    'hidden_dim': int(hidden_dim),
    'batch_size': int(batch_size),
    'model_X': {
        'name': 'Standard MLP on X (scaled)',
        'best_val_loss': float(res_X['best_val_loss']),
        'best_val_acc': float(res_X['best_val_acc']),
        'best_epoch_loss': int(res_X['best_epoch_loss']),
        'best_epoch_acc': int(res_X['best_epoch_acc']),
        'test_at_best_val_loss': float(res_X['test_at_best_val_loss']),
        'test_at_best_val_acc': float(res_X['test_at_best_val_acc'])
    },
    'model_U': {
        'name': 'RowNorm MLP on U (restricted eigenvectors)',
        'best_val_loss': float(res_U['best_val_loss']),
        'best_val_acc': float(res_U['best_val_acc']),
        'best_epoch_loss': int(res_U['best_epoch_loss']),
        'best_epoch_acc': int(res_U['best_epoch_acc']),
        'test_at_best_val_loss': float(res_U['test_at_best_val_loss']),
        'test_at_best_val_acc': float(res_U['test_at_best_val_acc'])
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
print(f'RESULTS - {DATASET_NAME.upper()}')
print('='*60)
print('Standard MLP on X (scaled):')
print(f'  best val loss  @ epoch {res_X["best_epoch_loss"]+1:3d}: test acc = {res_X["test_at_best_val_loss"]:.4f} '
      f'(val loss = {res_X["best_val_loss"]:.4f})')
print(f'  best val acc   @ epoch {res_X["best_epoch_acc"]+1:3d}: test acc = {res_X["test_at_best_val_acc"]:.4f} '
      f'(val acc  = {res_X["best_val_acc"]:.4f})')

print('\nRowNorm MLP on U (restricted eigenvectors):')
print(f'  best val loss  @ epoch {res_U["best_epoch_loss"]+1:3d}: test acc = {res_U["test_at_best_val_loss"]:.4f} '
      f'(val loss = {res_U["best_val_loss"]:.4f})')
print(f'  best val acc   @ epoch {res_U["best_epoch_acc"]+1:3d}: test acc = {res_U["test_at_best_val_acc"]:.4f} '
      f'(val acc  = {res_U["best_val_acc"]:.4f})')

print('\n' + '='*60)
print('COMPARISON (U vs X):')
diff_best_loss = res_U['test_at_best_val_loss'] - res_X['test_at_best_val_loss']
diff_best_acc = res_U['test_at_best_val_acc'] - res_X['test_at_best_val_acc']
pct_best_loss = 100 * (res_U['test_at_best_val_loss'] / res_X['test_at_best_val_loss'] - 1)
pct_best_acc = 100 * (res_U['test_at_best_val_acc'] / res_X['test_at_best_val_acc'] - 1)
print(f'At best val loss checkpoint: {diff_best_loss:+.4f} ({pct_best_loss:+.1f}%)')
print(f'At best val acc checkpoint:  {diff_best_acc:+.4f} ({pct_best_acc:+.1f}%)')
print('='*60)
print(f'✓ Investigation 2 complete for {DATASET_NAME}!')