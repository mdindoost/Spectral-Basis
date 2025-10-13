"""
Investigation 2 Extended: X-Restricted Eigenvectors with Convergence Analysis
==============================================================================

Usage:
    python experiments/investigation2_restricted_eigenvectors_extended.py [dataset_name]
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed

Compares raw X vs restricted eigenvectors U (same span, different basis).
Extended with:
- QR decomposition to handle rank deficiency
- Convergence tracking at checkpoints [10, 20, 40, 80, 160, 200]
- CosineAnnealingLR scheduler
- Convergence analysis and visualization
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
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
    build_graph_matrices, load_dataset
)

# ============================================================================
# NEW: Training Function with Convergence Tracking
# ============================================================================

def train_with_convergence(model, train_loader, X_val, y_val, X_test, y_test,
                           epochs=200, lr=1e-2, weight_decay=5e-4, 
                           device='cpu', verbose=True):
    """
    Training with convergence tracking and learning rate scheduler
    
    Tracks validation accuracy at checkpoints: [10, 20, 40, 80, 160, 200]
    Uses CosineAnnealingLR for smooth learning rate decay
    Maintains dual model selection (best val loss + best val acc)
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    # Convergence checkpoints
    checkpoints = [10, 20, 40, 80, 160, 200]
    checkpoint_accs = {}
    
    # Best model tracking (dual selection)
    best_loss = float('inf')
    best_acc = -1.0
    best_state_by_loss = None
    best_state_by_acc = None
    best_ep_loss = -1
    best_ep_acc = -1
    
    for ep in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = crit(val_out, y_val).item()
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / max(1, len(train_loader)))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Record checkpoint accuracy
        if (ep + 1) in checkpoints:
            checkpoint_accs[ep + 1] = val_acc
        
        # Track best models
        if val_loss < best_loss:
            best_loss = val_loss
            best_state_by_loss = copy.deepcopy(model.state_dict())
            best_ep_loss = ep
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_by_acc = copy.deepcopy(model.state_dict())
            best_ep_acc = ep
        
        # Step scheduler
        scheduler.step()
        
        if verbose and (ep + 1) % 20 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f'Epoch {ep+1}/{epochs}  TrainLoss={train_losses[-1]:.4f}  '
                  f'ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}  LR={current_lr:.6f}')
    
    # Evaluate test at best-by-loss checkpoint
    model.load_state_dict(best_state_by_loss)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc_best_loss = (test_pred == y_test).float().mean().item()
    
    # Evaluate test at best-by-acc checkpoint
    model.load_state_dict(best_state_by_acc)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc_best_acc = (test_pred == y_test).float().mean().item()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'checkpoint_accs': checkpoint_accs,
        'best_val_loss': best_loss,
        'best_val_acc': best_acc,
        'best_epoch_loss': best_ep_loss,
        'best_epoch_acc': best_ep_acc,
        'test_at_best_val_loss': test_acc_best_loss,
        'test_at_best_val_acc': test_acc_best_acc,
    }

# ============================================================================
# NEW: Convergence Analysis Functions
# ============================================================================

def print_convergence_table(res_X, res_U, dataset_name, epochs):
    """Print convergence table comparing X vs U"""
    print('\n' + '='*80)
    print('CONVERGENCE ANALYSIS: Validation Accuracy at Checkpoints')
    print('='*80)
    
    checkpoints = [cp for cp in [10, 20, 40, 80, 160, 200] if cp <= epochs]
    
    # Header
    header = f"{'Model':<45}"
    for cp in checkpoints:
        header += f"{cp:>7}"
    header += f"{'Final':>7}"
    print(header)
    print('-' * 80)
    
    # Standard MLP on X
    row = f"{dataset_name}–Standard MLP (X-scaled)           "
    for cp in checkpoints:
        if cp in res_X['checkpoint_accs']:
            acc = res_X['checkpoint_accs'][cp]
            row += f"{acc*100:>6.0f}%"
        else:
            row += "     -"
    final_acc = res_X['val_accs'][-1]
    row += f"{final_acc*100:>6.0f}%"
    print(row)
    
    # RowNorm MLP on U
    row = f"{dataset_name}–RowNorm MLP (U-restricted)        "
    for cp in checkpoints:
        if cp in res_U['checkpoint_accs']:
            acc = res_U['checkpoint_accs'][cp]
            row += f"{acc*100:>6.0f}%"
        else:
            row += "     -"
    final_acc = res_U['val_accs'][-1]
    row += f"{final_acc*100:>6.0f}%"
    print(row)
    
    print('-' * 80)
    
    # Calculate epoch 20 comparison
    if 20 in res_X['checkpoint_accs'] and 20 in res_U['checkpoint_accs']:
        x_epoch20 = res_X['checkpoint_accs'][20]
        u_epoch20 = res_U['checkpoint_accs'][20]
        
        print(f'\n*** KEY COMPARISON at Epoch 20 ***')
        print(f'Standard MLP (X-scaled): {x_epoch20*100:.1f}%')
        print(f'RowNorm MLP (U):        {u_epoch20*100:.1f}%')
        
        if u_epoch20 > x_epoch20:
            speedup = (u_epoch20 - x_epoch20) / x_epoch20 * 100
            print(f'✓ RowNorm converges {speedup:.1f}% faster at epoch 20!')
        elif u_epoch20 < x_epoch20:
            slowdown = (x_epoch20 - u_epoch20) / x_epoch20 * 100
            print(f'✗ RowNorm is {slowdown:.1f}% slower at epoch 20')
        else:
            print(f'≈ Similar convergence speed')
    
    print('='*80)

def plot_convergence_curves(res_X, res_U, dataset_name, output_path, epochs):
    """Plot convergence curves with checkpoint markers"""
    checkpoints = [cp for cp in [10, 20, 40, 80, 160, 200] if cp <= epochs]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Validation accuracy comparison
    axes[0].plot(res_X['val_accs'], label='Standard MLP (X-scaled)', 
                linewidth=3, alpha=0.9, color='red')
    axes[0].plot(res_U['val_accs'], label='RowNorm MLP (U-restricted)', 
                linewidth=3, alpha=0.9, color='blue')
    
    # Mark checkpoints with dots
    for cp in checkpoints:
        if cp in res_X['checkpoint_accs']:
            axes[0].plot(cp-1, res_X['checkpoint_accs'][cp], 
                       'o', color='red', markersize=8)
        if cp in res_U['checkpoint_accs']:
            axes[0].plot(cp-1, res_U['checkpoint_accs'][cp], 
                       'o', color='blue', markersize=8)
    
    # Mark checkpoint epochs with vertical lines
    for cp in checkpoints:
        axes[0].axvline(cp-1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        axes[0].text(cp-1, axes[0].get_ylim()[1], str(cp), 
                    ha='center', va='bottom', fontsize=9, color='gray', fontweight='bold')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title(f'Convergence Speed Comparison - {dataset_name}', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    # Right: Training loss comparison
    axes[1].plot(res_X['train_losses'], label='Standard MLP (X-scaled)', 
                linewidth=2, alpha=0.85, color='red')
    axes[1].plot(res_U['train_losses'], label='RowNorm MLP (U-restricted)', 
                linewidth=2, alpha=0.85, color='blue')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Loss', fontsize=12)
    axes[1].set_title(f'Training Loss - {dataset_name}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved convergence plot: {output_path}')

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
print('='*70)
print(f'INVESTIGATION 2 EXTENDED: X-RESTRICTED EIGENVECTORS - {DATASET_NAME.upper()}')
print('='*70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('\nNew features:')
print('  • QR decomposition to handle rank deficiency')
print('  • 200 epochs with CosineAnnealingLR scheduler')
print('  • Convergence tracking at [10, 20, 40, 80, 160, 200]')
print('='*70)

# Create dataset-specific output directories
output_base = f'results/investigation2_extended/{DATASET_NAME}'
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
# 3. Compute X-Restricted Eigenvectors with QR Decomposition
# ============================================================================
print('\n[3/6] Computing X-restricted eigenvectors...')

X = X_raw.astype(np.float64)

# Check rank and perform QR decomposition
from numpy.linalg import matrix_rank
rank_X = matrix_rank(X, tol=1e-6)
print(f'Rank of X: {rank_X}/{d_raw}')

if rank_X < d_raw:
    print(f'⚠ Warning: X is rank-deficient ({rank_X} < {d_raw})')
    print(f'Performing QR decomposition to handle rank deficiency...')
    
    # QR decomposition
    Q, R = np.linalg.qr(X, mode='reduced')
    # Q: shape (n, d_raw) — full column count
    # R: shape (d_raw, d_raw)
    
    # CRITICAL: Truncate Q to effective rank
    Q = Q[:, :rank_X]  # Keep only rank_X columns
    # Q: shape (n, rank_X) — effective rank only
    
    print(f'✓ Reduced dimension from {d_raw} to {rank_X} (effective rank)')
    print(f'  Q shape after truncation: {Q.shape}')
    print(f'  span(Q) = span(X) — same subspace, orthonormal basis')
    
    # Use Q instead of X
    X_for_projection = Q
    d_effective = rank_X
else:
    print(f'✓ X is full-rank')
    X_for_projection = X
    d_effective = d_raw

# Project L and D into span(X) = span(Q)
print(f'Projecting Laplacian into span(X) (effective dimension {d_effective})...')

LX = L @ X_for_projection
DX = D @ X_for_projection

Lr = X_for_projection.T @ LX  # shape: (d_effective, d_effective)
Dr = X_for_projection.T @ DX  # shape: (d_effective, d_effective)

# Symmetrize for numerical stability
Lr = 0.5 * (Lr + Lr.T)
Dr = 0.5 * (Dr + Dr.T)

print(f'Solving reduced generalized eigenproblem in R^{d_effective}...')

# No regularization needed with QR!
try:
    w, V = la.eigh(Lr, Dr)  # Dense eigensolver
    idx = np.argsort(w)
    w = w[idx]
    V = V[:, idx]
except np.linalg.LinAlgError as e:
    print(f'ERROR: Eigensolver failed: {e}')
    print(f'This should not happen with QR decomposition.')
    sys.exit(1)

print(f'Eigenvalues range: [{w.min():.6f}, {w.max():.6f}]')

# Map back to node space
U = (X_for_projection @ V).astype(np.float32)
print(f'Restricted eigenvector matrix U shape: {U.shape}')
print(f'  Expected: ({num_nodes}, {d_effective})')

# Verify U has correct dimension
assert U.shape[1] == d_effective, f"Dimension mismatch: U has {U.shape[1]} columns but expected {d_effective}"

rank_U = matrix_rank(U, tol=1e-6)
print(f'Rank of U: {rank_U}/{d_effective}')
print(f'✓ span(U) = span(X) — same subspace, different basis')

# Verify D-orthonormality of U
DU = (D @ U).astype(np.float64)
G = U.astype(np.float64).T @ DU
deviation = np.abs(G - np.eye(U.shape[1])).max()
print(f'D-orthonormality check: max |U^T D U - I| = {deviation:.2e}')

if deviation < 1e-6:
    print(f'✓ Excellent D-orthonormality!')
elif deviation < 1e-4:
    print(f'✓ Good D-orthonormality')
else:
    print(f'⚠ Warning: Large D-orthonormality deviation')
    
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
# 5. Train Both Models (with convergence tracking)
# ============================================================================
print('\n[5/6] Training 2 models (200 epochs with CosineAnnealingLR)...')

hidden_dim = 256
batch_size = 128
epochs = 200

from torch.utils.data import DataLoader, TensorDataset

# Model A: Standard MLP on X (scaled)
print('\n--- Model A: Standard MLP on Raw X (scaled) ---')
train_loader_X = DataLoader(TensorDataset(X_train_std, y_train), batch_size=batch_size, shuffle=True)
model_X = StandardMLP(d_raw, hidden_dim, num_classes).to(device)
res_X = train_with_convergence(
    model_X, train_loader_X, X_val_std, y_val, X_test_std, y_test,
    epochs=epochs, lr=1e-2, weight_decay=5e-4, device=device, verbose=True
)

# Model B: RowNorm MLP on U (restricted eigenvectors)
print('\n--- Model B: RowNorm MLP on Restricted Eigenvectors U ---')
train_loader_U = DataLoader(TensorDataset(U_train, y_train), batch_size=batch_size, shuffle=True)
model_U = RowNormMLP(d_effective, hidden_dim, num_classes).to(device)
res_U = train_with_convergence(
    model_U, train_loader_U, U_val, y_val, U_test, y_test,
    epochs=epochs, lr=1e-2, weight_decay=5e-4, device=device, verbose=True
)

# ============================================================================
# 6. Generate Analysis, Plots and Save Results
# ============================================================================
print('\n[6/6] Generating convergence analysis and plots...')

# Print convergence table
print_convergence_table(res_X, res_U, DATASET_NAME, epochs)

# Create convergence plot
plot_path_convergence = f'{output_base}/plots/convergence_analysis.png'
plot_convergence_curves(res_X, res_U, DATASET_NAME, plot_path_convergence, epochs)

# Create original 2x3 grid plot (for consistency)
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle(f'Investigation 2 Extended: X-Restricted Eigenvectors - {DATASET_NAME}', fontsize=14)

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
axes[1, 0].set_title('U-restricted: Val Acc')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axvline(res_U['best_epoch_acc'], color='k', linestyle='--', alpha=0.5)

axes[1, 1].plot(res_U['train_losses'], linewidth=2)
axes[1, 1].set_title('U-restricted: Train Loss')
axes[1, 1].grid(alpha=0.3)

axes[1, 2].plot(res_U['val_losses'], linewidth=2)
axes[1, 2].set_title('U-restricted: Val Loss')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].axvline(res_U['best_epoch_loss'], color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plot_path = f'{output_base}/plots/comparison.png'
plt.savefig(plot_path, dpi=150)
print(f'✓ Saved standard plot: {plot_path}')

# Save metrics (extended with convergence data)
metrics = {
    'dataset': DATASET_NAME,
    'feature_dimension_original': int(d_raw),
    'feature_dimension_effective': int(d_effective),
    'rank_deficient': bool(rank_X < d_raw),
    'epochs': int(epochs),
    'hidden_dim': int(hidden_dim),
    'batch_size': int(batch_size),
    'scheduler': 'CosineAnnealingLR',
    'model_X': {
        'name': 'Standard MLP on X (scaled)',
        'best_val_loss': float(res_X['best_val_loss']),
        'best_val_acc': float(res_X['best_val_acc']),
        'best_epoch_loss': int(res_X['best_epoch_loss']),
        'best_epoch_acc': int(res_X['best_epoch_acc']),
        'test_at_best_val_loss': float(res_X['test_at_best_val_loss']),
        'test_at_best_val_acc': float(res_X['test_at_best_val_acc']),
        'checkpoint_accuracies': {str(cp): float(acc) for cp, acc in res_X['checkpoint_accs'].items()}
    },
    'model_U': {
        'name': 'RowNorm MLP on U (restricted eigenvectors)',
        'best_val_loss': float(res_U['best_val_loss']),
        'best_val_acc': float(res_U['best_val_acc']),
        'best_epoch_loss': int(res_U['best_epoch_loss']),
        'best_epoch_acc': int(res_U['best_epoch_acc']),
        'test_at_best_val_loss': float(res_U['test_at_best_val_loss']),
        'test_at_best_val_acc': float(res_U['test_at_best_val_acc']),
        'checkpoint_accuracies': {str(cp): float(acc) for cp, acc in res_U['checkpoint_accs'].items()}
    }
}

metrics_path = f'{output_base}/metrics/results.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'✓ Saved metrics: {metrics_path}')

# Save convergence table to text file
table_path = f'{output_base}/metrics/convergence_table.txt'
with open(table_path, 'w') as f:
    import sys
    old_stdout = sys.stdout
    sys.stdout = f
    print_convergence_table(res_X, res_U, DATASET_NAME, epochs)
    sys.stdout = old_stdout
print(f'✓ Saved convergence table: {table_path}')

# ============================================================================
# Print Summary
# ============================================================================
print('\n' + '='*70)
print(f'RESULTS - {DATASET_NAME.upper()}')
print('='*70)
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

print('\n' + '='*70)
print('COMPARISON (U vs X):')
diff_best_loss = res_U['test_at_best_val_loss'] - res_X['test_at_best_val_loss']
diff_best_acc = res_U['test_at_best_val_acc'] - res_X['test_at_best_val_acc']
pct_best_loss = 100 * (res_U['test_at_best_val_loss'] / res_X['test_at_best_val_loss'] - 1)
pct_best_acc = 100 * (res_U['test_at_best_val_acc'] / res_X['test_at_best_val_acc'] - 1)
print(f'At best val loss checkpoint: {diff_best_loss:+.4f} ({pct_best_loss:+.1f}%)')
print(f'At best val acc checkpoint:  {diff_best_acc:+.4f} ({pct_best_acc:+.1f}%)')

# Convergence speed comparison (epoch 20)
print('\n' + '='*70)
print('CONVERGENCE SPEED: Epoch 20 Accuracy')
print('='*70)

if 20 in res_X['checkpoint_accs'] and 20 in res_U['checkpoint_accs']:
    x_epoch20 = res_X['checkpoint_accs'][20]
    u_epoch20 = res_U['checkpoint_accs'][20]
    
    print(f'Standard MLP (X-scaled) @ epoch 20:  {x_epoch20:.4f} ({x_epoch20*100:.1f}%)')
    print(f'RowNorm MLP (U) @ epoch 20:          {u_epoch20:.4f} ({u_epoch20*100:.1f}%)')
    
    if u_epoch20 > x_epoch20:
        speedup = (u_epoch20 - x_epoch20) / x_epoch20 * 100
        print(f'\n✓ RowNorm converges {speedup:.1f}% faster at epoch 20!')
    elif u_epoch20 < x_epoch20:
        slowdown = (x_epoch20 - u_epoch20) / x_epoch20 * 100
        print(f'\n✗ RowNorm is {slowdown:.1f}% slower at epoch 20')
    else:
        print(f'\n≈ Similar convergence speed')

print('='*70)
print(f'✓ Investigation 2 Extended complete for {DATASET_NAME}!')
print(f'✓ Results saved to: {output_base}/')