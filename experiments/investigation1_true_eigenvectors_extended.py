"""
Investigation 1 Extended: True Eigenvectors with Convergence Analysis
======================================================================

Usage:
    python experiments/investigation1_true_eigenvectors_extended.py [dataset_name]
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed

Extended version with:
- k = 2×num_classes (no cap)
- 100 epochs (increased from 60)
- CosineAnnealingLR scheduler
- Convergence tracking at checkpoints [10, 20, 40, 80, 160, 200]
- Handling disconnected graphs
- Both convergence table and plot
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    StandardMLP, RowNormMLP, CosineRowNormMLP,
    build_graph_matrices, load_dataset
)

# ============================================================================
# Training Function with Convergence Tracking
# ============================================================================

def train_with_convergence(model, train_loader, X_val, y_val, X_test, y_test,
                           epochs=100, lr=1e-2, weight_decay=5e-4, 
                           device='cpu', verbose=True):
    """
    Training with learning rate scheduler and convergence tracking
    
    Tracks validation accuracy at checkpoints: [10, 20, 40, 80, 160, 200]
    Uses CosineAnnealingLR for smooth learning rate decay
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accs = []
    
    # Convergence checkpoints
    checkpoints = [10, 20, 40, 80, 160, 200]
    checkpoint_accs = {}
    
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
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / max(1, len(train_loader)))
        val_accs.append(val_acc)
        
        # Record checkpoint accuracy
        if (ep + 1) in checkpoints:
            checkpoint_accs[ep + 1] = val_acc
        
        # Step scheduler after validation
        scheduler.step()
        
        if verbose and (ep + 1) % 20 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f'Epoch {ep+1}/{epochs}  TrainLoss={train_losses[-1]:.4f}  '
                  f'ValAcc={val_acc:.4f}  LR={current_lr:.6f}')
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    return {
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'checkpoint_accs': checkpoint_accs
    }

# ============================================================================
# Convergence Analysis Functions
# ============================================================================

def print_convergence_table(results, model_names, model_keys, dataset_name, epochs):
    """Print convergence table highlighting key comparisons"""
    print('\n' + '='*90)
    print('CONVERGENCE ANALYSIS: Validation Accuracy at Checkpoints')
    print('='*90)
    
    checkpoints = [cp for cp in [10, 20, 40, 80, 160, 200] if cp <= epochs]
    
    # Header
    header = f"{'Model':<40}"
    for cp in checkpoints:
        header += f"{cp:>7}"
    header += f"{'Final':>7}"
    print(header)
    print('-' * 90)
    
    # Print all models
    for name, key in zip(model_names, model_keys):
        # Shorten name for table
        short_name = name.replace(' (train-only scaled)', '').replace(' MLP', '')
        row = f"{dataset_name}–{short_name:<35}"
        
        for cp in checkpoints:
            if cp in results[key]['checkpoint_accs']:
                acc = results[key]['checkpoint_accs'][cp]
                row += f"{acc*100:>6.0f}%"
            else:
                row += "     -"
        
        # Add final accuracy
        final_acc = results[key]['val_accs'][-1]
        row += f"{final_acc*100:>6.0f}%"
        print(row)
    
    print('-' * 90)
    
    # Highlight key comparison: Standard train-scaled vs RowNorm
    print('\n*** KEY COMPARISON: Standard MLP (train-scaled) vs Row-Normalized MLP ***')
    print('-' * 90)

    key_models = [
        ('Standard (train-scaled)', 'standard_train_scaled'),  # ← CORRECT!
        ('Row-Normalized', 'rownorm')
    ]
    
    header = f"{'Model':<40}"
    for cp in checkpoints:
        header += f"{cp:>7}"
    header += f"{'Final':>7}"
    print(header)
    print('-' * 90)
    
    for name, key in key_models:
        row = f"{dataset_name}–{name:<35}"
        
        for cp in checkpoints:
            if cp in results[key]['checkpoint_accs']:
                acc = results[key]['checkpoint_accs'][cp]
                row += f"{acc*100:>6.0f}%"
            else:
                row += "     -"
        
        final_acc = results[key]['val_accs'][-1]
        row += f"{final_acc*100:>6.0f}%"
        print(row)
    
    print('='*90)

def plot_convergence_curves(results, model_names, model_keys, dataset_name, 
                            output_path, epochs):
    """Plot convergence curves with checkpoint markers"""
    checkpoints = [cp for cp in [10, 20, 40, 80, 160, 200] if cp <= epochs]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: All models validation accuracy
    for name, key in zip(model_names, model_keys):
        axes[0].plot(results[key]['val_accs'], label=name, linewidth=2, alpha=0.85)
    
    # Mark checkpoint epochs
    for cp in checkpoints:
        axes[0].axvline(cp-1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title(f'Validation Accuracy - {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    # Add checkpoint labels at top
    for cp in checkpoints:
        axes[0].text(cp-1, axes[0].get_ylim()[1], str(cp), 
                    ha='center', va='bottom', fontsize=8, color='gray')
    
    # Right: Highlight key comparison (Standard no-scale vs RowNorm)
    key_models = [
        ('Standard MLP (train-only scaled)', 'standard_train_scaled', 'red'),
        ('Row-Normalized MLP', 'rownorm', 'blue')
    ]
    
    for name, key, color in key_models:
        axes[1].plot(results[key]['val_accs'], label=name, 
                    linewidth=3, alpha=0.9, color=color)
        
        # Mark checkpoints with dots
        for cp in checkpoints:
            if cp in results[key]['checkpoint_accs']:
                axes[1].plot(cp-1, results[key]['checkpoint_accs'][cp], 
                           'o', color=color, markersize=8)
    
    # Mark checkpoint epochs
    for cp in checkpoints:
        axes[1].axvline(cp-1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].set_title(f'Key Comparison: Convergence Speed - {dataset_name}', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    # Add checkpoint labels at top
    for cp in checkpoints:
        axes[1].text(cp-1, axes[1].get_ylim()[1], str(cp), 
                    ha='center', va='bottom', fontsize=9, color='gray', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved convergence plot: {output_path}')

# ============================================================================
# Configuration
# ============================================================================

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ogbn-arxiv'

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
print(f'INVESTIGATION 1 EXTENDED: TRUE EIGENVECTORS - {DATASET_NAME.upper()}')
print('='*70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('\nNew features:')
print('  • k = 2×num_classes (no cap)')
print('  • 100 epochs with CosineAnnealingLR scheduler')
print('  • Convergence tracking at [10, 20, 40, 80, 160, 200]')
print('  • Handling disconnected graphs')
print('='*70)

# Create output directories
output_base = f'results/investigation1_extended/{DATASET_NAME}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

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
# 3. Compute True Eigenvectors (with disconnected graph handling)
# ============================================================================
print('\n[3/6] Computing true eigenvectors...')

# Compute k = 2*num_classes (no cap)
k_target = 2 * num_classes
# Add buffer for potential zero eigenvalues (disconnected components)
k_compute = min(k_target + 10, num_nodes - 2)

print(f'Target eigenvectors: k = 2×{num_classes} = {k_target}')
print(f'Computing k = {k_compute} to handle potential disconnected components...')
print(f'Solving (D-A)x = λDx for smallest eigenvalues...')

eigenvalues, eigenvectors = eigsh(
    L.astype(np.float64),
    k=k_compute,
    M=D.astype(np.float64),
    which='SM',
    tol=1e-4
)

# Check for zero eigenvalues (disconnected components)
num_zero = np.sum(np.abs(eigenvalues) < 1e-6)

if num_zero > 0:
    print(f'⚠ Detected {num_zero} near-zero eigenvalues (disconnected components)')
    print(f'  Using eigenvectors [{num_zero+1}..{min(num_zero+k_target, k_compute)}]')
    
    # Skip zero eigenvalues and take upper k_target
    eigenvectors = eigenvectors[:, num_zero:min(num_zero+k_target, k_compute)]
    eigenvalues = eigenvalues[num_zero:min(num_zero+k_target, k_compute)]
else:
    print(f'✓ No disconnected components detected')
    eigenvectors = eigenvectors[:, :k_target]
    eigenvalues = eigenvalues[:k_target]

k = len(eigenvalues)
print(f'Final eigenvector count: k = {k}')
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
print('\n[5/6] Training 6 models (100 epochs with CosineAnnealingLR)...')

hidden_dim = 256
batch_size = 128
epochs = 100

results = {}

# Model 1: Standard MLP (train-only scaling)
print('\n--- Model 1: Standard MLP (train-only scaling) ---')
train_loader = DataLoader(TensorDataset(X_train_std, y_train), batch_size=batch_size, shuffle=True)
model1 = StandardMLP(k, hidden_dim, num_classes).to(device)
results['standard_train_scaled'] = train_with_convergence(
    model1, train_loader, X_val_std, y_val, X_test_std, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 2: Standard MLP (full-data scaling)
print('\n--- Model 2: Standard MLP (full-data scaling) ---')
train_loader = DataLoader(TensorDataset(X_train_full, y_train), batch_size=batch_size, shuffle=True)
model2 = StandardMLP(k, hidden_dim, num_classes).to(device)
results['standard_full_scaled'] = train_with_convergence(
    model2, train_loader, X_val_full, y_val, X_test_full, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 3: Standard MLP (no scaling) *** KEY COMPARISON ***
print('\n--- Model 3: Standard MLP (no scaling) *** KEY COMPARISON *** ---')
train_loader = DataLoader(TensorDataset(X_train_ns, y_train), batch_size=batch_size, shuffle=True)
model3 = StandardMLP(k, hidden_dim, num_classes).to(device)
results['standard_no_scale'] = train_with_convergence(
    model3, train_loader, X_val_ns, y_val, X_test_ns, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 4: Standard MLP (eigenvalue-weighted)
print('\n--- Model 4: Standard MLP (eigenvalue-weighted) ---')
train_loader = DataLoader(TensorDataset(X_train_w, y_train), batch_size=batch_size, shuffle=True)
model4 = StandardMLP(k, hidden_dim, num_classes).to(device)
results['standard_weighted'] = train_with_convergence(
    model4, train_loader, X_val_w, y_val, X_test_w, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 5: RowNorm MLP *** KEY COMPARISON ***
print('\n--- Model 5: RowNorm MLP (Radial) *** KEY COMPARISON *** ---')
train_loader = DataLoader(TensorDataset(X_train_rn, y_train), batch_size=batch_size, shuffle=True)
model5 = RowNormMLP(k, hidden_dim, num_classes).to(device)
results['rownorm'] = train_with_convergence(
    model5, train_loader, X_val_rn, y_val, X_test_rn, y_test,
    epochs=epochs, device=device, verbose=True
)

# Model 6: Cosine Classifier MLP
print('\n--- Model 6: Cosine Classifier MLP ---')
train_loader = DataLoader(TensorDataset(X_train_rn, y_train), batch_size=batch_size, shuffle=True)
model6 = CosineRowNormMLP(k, hidden_dim, num_classes).to(device)
results['cosine'] = train_with_convergence(
    model6, train_loader, X_val_rn, y_val, X_test_rn, y_test,
    epochs=epochs, device=device, verbose=True
)

# ============================================================================
# 6. Generate Analysis and Save Results
# ============================================================================
print('\n[6/6] Generating convergence analysis and plots...')

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

# Print convergence table
print_convergence_table(results, model_names, model_keys, DATASET_NAME, epochs)

# Create convergence plot
plot_path = f'{output_base}/plots/convergence_analysis.png'
plot_convergence_curves(results, model_names, model_keys, DATASET_NAME, plot_path, epochs)

# Create standard plots (validation accuracy + training loss)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

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
print(f'✓ Saved standard plot: {plot_path}')

# Save metrics
metrics = {
    'dataset': DATASET_NAME,
    'k_eigenvectors': int(k),
    'epochs': int(epochs),
    'hidden_dim': int(hidden_dim),
    'batch_size': int(batch_size),
    'scheduler': 'CosineAnnealingLR',
    'disconnected_components': int(num_zero) if num_zero > 0 else 0,
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
    },
    'checkpoint_accuracies': {
        key: {str(cp): float(acc) for cp, acc in results[key]['checkpoint_accs'].items()}
        for key in model_keys
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
    print_convergence_table(results, model_names, model_keys, DATASET_NAME, epochs)
    sys.stdout = old_stdout
print(f'✓ Saved convergence table: {table_path}')

# ============================================================================
# Print Summary
# ============================================================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Features: {k} eigenvectors from (D−A)x = λ D x (D-orthonormal)')
print(f'Hidden dimension: {hidden_dim}')
print(f'Training epochs: {epochs}')
print(f'Batch size: {batch_size}')
print(f'Scheduler: CosineAnnealingLR')
if num_zero > 0:
    print(f'Disconnected components: {num_zero} zero eigenvalues detected and handled')

models_info = [
    ('Standard MLP (train-only scaling)', 'standard_train_scaled', model1),
    ('Full-Data Scaled MLP', 'standard_full_scaled', model2),
    ('No-Scaling MLP', 'standard_no_scale', model3),
    ('Eigenvalue-Weighted MLP', 'standard_weighted', model4),
    ('Row-Normalized MLP (Radial)', 'rownorm', model5),
    ('Cosine Classifier MLP', 'cosine', model6),
]

for i, (name, key, model) in enumerate(models_info, 1):
    num_params = sum(p.numel() for p in model.parameters())
    final_val_acc = results[key]['val_accs'][-1]
    best_val_acc = max(results[key]['val_accs'])
    test_acc = results[key]['test_acc']
    
    print(f'\n{i}. {name}:')
    print(f'   - Params: {num_params:,}')
    print(f'   - Final Val Acc: {final_val_acc:.4f}')
    print(f'   - Best  Val Acc: {best_val_acc:.4f}')
    print(f'   - Test  Acc: {test_acc:.4f}')
    
    # Convergence speed (epoch 20 accuracy)
    if 20 in results[key]['checkpoint_accs']:
        epoch20_acc = results[key]['checkpoint_accs'][20]
        print(f'   - Epoch 20 Acc: {epoch20_acc:.4f} (convergence indicator)')
    
    if key == 'cosine':
        print(f'   - Learned Scale: {model6.scale.item():.4f}')

print('\n' + '='*70)
print('COMPARISON (Relative to Standard MLP train-scaled)')
print('='*70)

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

# Convergence speed comparison (epoch 20)
print('\n' + '='*70)
print('CONVERGENCE SPEED: Epoch 20 Accuracy (faster convergence indicator)')
print('='*70)

if 20 in results['standard_train_scaled']['checkpoint_accs'] and 20 in results['rownorm']['checkpoint_accs']:
    std_epoch20 = results['standard_train_scaled']['checkpoint_accs'][20]  # ← FIXED
    rn_epoch20 = results['rownorm']['checkpoint_accs'][20]
    
    print(f'Standard MLP (train-scaled) @ epoch 20:  {std_epoch20:.4f} ({std_epoch20*100:.1f}%)')
    print(f'Row-Normalized MLP @ epoch 20:         {rn_epoch20:.4f} ({rn_epoch20*100:.1f}%)')
    
    if rn_epoch20 > std_epoch20:
        speedup = (rn_epoch20 - std_epoch20) / std_epoch20 * 100
        print(f'\n✓ RowNorm converges {speedup:.1f}% faster at epoch 20!')
    else:
        print(f'\n✗ No significant convergence advantage detected')

print('='*70)
print(f'✓ Investigation 1 Extended complete for {DATASET_NAME}!')
print(f'✓ Results saved to: {output_base}/')