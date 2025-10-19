"""
Investigation 2 Extended Version 2: X-Restricted Eigenvectors with Statistical Rigor
=====================================================================================

Usage:
    # Fixed benchmark splits (default)
    python experiments/investigation2_extended_ver2.py [dataset_name]
    
    # Random 60/20/20 splits
    python experiments/investigation2_extended_ver2.py [dataset_name] --random-splits
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed

Improvements in v2:
- Multiple random seeds (5 for fixed splits, 15 for random splits)
- Support for both fixed benchmark splits and random 60/20/20 splits
- Adaptive batch size (128 for large datasets, full batch for small)
- Granular batch-level tracking for first 5 epochs
- Quantitative convergence metrics
- Enhanced plots with confidence bands and early convergence zoom
- QR decomposition to handle rank deficiency
- 200 epochs with CosineAnnealingLR scheduler

Expected finding: Small but real improvement from basis change despite identical span
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
# Configuration
# ============================================================================
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ogbn-arxiv'

# Check for random splits flag
USE_RANDOM_SPLITS = '--random-splits' in sys.argv

# Validate dataset name
VALID_DATASETS = ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed']
if DATASET_NAME not in VALID_DATASETS:
    print(f"Error: Invalid dataset '{DATASET_NAME}'")
    print(f"Valid datasets: {', '.join(VALID_DATASETS)}")
    sys.exit(1)

# Experimental parameters
NUM_SEEDS = 5  # Number of random seeds for statistical robustness
NUM_RANDOM_SPLITS = 3  # Number of random 60/20/20 splits
EPOCHS = 200  # Extended for high-dimensional features
HIDDEN_DIM = 256
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 5e-4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

split_type = 'random-splits' if USE_RANDOM_SPLITS else 'fixed-splits'
print('='*70)
print(f'INVESTIGATION 2 EXTENDED V2: X-RESTRICTED EIGENVECTORS - {DATASET_NAME.upper()}')
print('='*70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'\nSplit Type: {split_type.upper()}')
print(f'Number of seeds: {NUM_SEEDS}')
if USE_RANDOM_SPLITS:
    print(f'Number of random splits: {NUM_RANDOM_SPLITS}')
print(f'Epochs: {EPOCHS}')
print(f'Hidden dimension: {HIDDEN_DIM}')
print('='*70)

# Create output directories
output_base = f'results/investigation2_v2/{DATASET_NAME}/{split_type}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def get_batch_size(train_size):
    """
    Adaptive batch size strategy:
    - Large datasets (>256 samples): use batch_size=128
    - Small datasets (â‰¤256 samples): use full batch
    """
    if train_size > 256:
        return 128
    else:
        return train_size

def create_random_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Create random 60/20/20 train/val/test split
    
    Args:
        num_nodes: total number of nodes
        train_ratio: fraction for training (default 0.6)
        val_ratio: fraction for validation (default 0.2)
        seed: random seed for reproducibility
    
    Returns:
        train_idx, val_idx, test_idx: numpy arrays of indices
    """
    indices = np.arange(num_nodes)
    
    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        random_state=seed,
        shuffle=True
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=val_ratio_adjusted, 
        random_state=seed,
        shuffle=True
    )
    
    return train_idx, val_idx, test_idx

def compute_convergence_metrics(val_accs, epochs=200):
    """
    Compute convergence speed metrics
    
    Args:
        val_accs: list of validation accuracies
        epochs: total number of epochs
    
    Returns:
        dict with convergence metrics
    """
    val_accs = np.array(val_accs)
    final_acc = val_accs[-1]
    
    # Speed to reach X% of final accuracy
    def epochs_to_reach(target_pct):
        threshold = target_pct * final_acc
        indices = np.where(val_accs >= threshold)[0]
        return int(indices[0]) if len(indices) > 0 else epochs
    
    speed_90 = epochs_to_reach(0.90)
    speed_95 = epochs_to_reach(0.95)
    speed_99 = epochs_to_reach(0.99)
    
    # Area under curve (normalized)
    auc = np.trapz(val_accs) / epochs
    
    # Average improvement rate in first 20 epochs
    if len(val_accs) >= 20:
        conv_rate = (val_accs[19] - val_accs[0]) / 20
    else:
        conv_rate = 0.0
    
    return {
        'speed_to_90': speed_90,
        'speed_to_95': speed_95,
        'speed_to_99': speed_99,
        'auc': float(auc),
        'convergence_rate': float(conv_rate),
        'final_acc': float(final_acc)
    }

def train_with_granular_tracking(model, train_loader, X_val, y_val, X_test, y_test,
                                 epochs=200, lr=1e-2, weight_decay=5e-4, 
                                 device='cpu', verbose=True):
    """
    Training with granular convergence tracking and learning rate scheduler
    
    Tracks:
    - Batch-level validation accuracy (first 5 epochs)
    - Epoch-level validation accuracy and training loss
    - Checkpoint accuracies at [10, 20, 40, 80, 160, 200]
    - Convergence metrics
    
    Uses CosineAnnealingLR for smooth learning rate decay
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    # Batch-level tracking (first 5 epochs only)
    batch_level_tracking = []
    
    # Convergence checkpoints
    checkpoints = [10, 20, 40, 80, 160, 200]
    checkpoint_accs = {}
    
    for ep in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        
        for batch_idx, (bx, by) in enumerate(train_loader):
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            # Batch-level validation (first 5 epochs only)
            if ep < 5:
                model.eval()
                with torch.no_grad():
                    val_out = model(X_val)
                    val_pred = val_out.argmax(1)
                    val_acc = (val_pred == y_val).float().mean().item()
                
                batch_level_tracking.append({
                    'epoch': ep,
                    'batch': batch_idx,
                    'global_step': ep * len(train_loader) + batch_idx,
                    'val_acc': val_acc
                })
                model.train()
        
        # Epoch-level validation
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
        
        # Step scheduler
        scheduler.step()
        
        if verbose and (ep + 1) % 20 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f'  Epoch {ep+1}/{epochs}  TrainLoss={train_losses[-1]:.4f}  '
                  f'ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}  LR={current_lr:.6f}')
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    # Compute convergence metrics
    convergence_metrics = compute_convergence_metrics(val_accs, epochs)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'checkpoint_accs': checkpoint_accs,
        'batch_level_tracking': batch_level_tracking,
        'convergence_metrics': convergence_metrics
    }

def aggregate_results(results_list):
    """
    Aggregate results across multiple seeds/splits
    
    Args:
        results_list: list of result dicts from train_with_granular_tracking
    
    Returns:
        dict with mean and std for all metrics
    """
    # Test accuracies
    test_accs = [r['test_acc'] for r in results_list]
    
    # Validation accuracy curves
    val_accs_array = np.array([r['val_accs'] for r in results_list])
    
    # Validation loss curves
    val_losses_array = np.array([r['val_losses'] for r in results_list])
    
    # Training loss curves
    train_losses_array = np.array([r['train_losses'] for r in results_list])
    
    # Checkpoint accuracies
    all_checkpoints = {}
    checkpoints = [10, 20, 40, 80, 160, 200]
    for cp in checkpoints:
        cp_accs = [r['checkpoint_accs'].get(cp, np.nan) for r in results_list]
        cp_accs = [acc for acc in cp_accs if not np.isnan(acc)]
        if cp_accs:
            all_checkpoints[cp] = {
                'mean': float(np.mean(cp_accs)),
                'std': float(np.std(cp_accs))
            }
    
    # Convergence metrics
    conv_metrics = {}
    metric_keys = ['speed_to_90', 'speed_to_95', 'speed_to_99', 'auc', 'convergence_rate', 'final_acc']
    for key in metric_keys:
        values = [r['convergence_metrics'][key] for r in results_list]
        conv_metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    # Batch-level tracking (aggregate across seeds)
    if results_list[0]['batch_level_tracking']:
        batch_tracking_aggregated = {}
        max_steps = max([len(r['batch_level_tracking']) for r in results_list])
        
        for step in range(max_steps):
            step_accs = []
            for r in results_list:
                if step < len(r['batch_level_tracking']):
                    step_accs.append(r['batch_level_tracking'][step]['val_acc'])
            
            if step_accs:
                batch_tracking_aggregated[step] = {
                    'global_step': results_list[0]['batch_level_tracking'][step]['global_step'],
                    'mean_val_acc': float(np.mean(step_accs)),
                    'std_val_acc': float(np.std(step_accs))
                }
    else:
        batch_tracking_aggregated = {}
    
    return {
        'test_acc': {
            'mean': float(np.mean(test_accs)),
            'std': float(np.std(test_accs)),
            'all': test_accs
        },
        'val_accs': {
            'mean': val_accs_array.mean(axis=0),
            'std': val_accs_array.std(axis=0)
        },
        'val_losses': {
            'mean': val_losses_array.mean(axis=0),
            'std': val_losses_array.std(axis=0)
        },
        'train_losses': {
            'mean': train_losses_array.mean(axis=0),
            'std': train_losses_array.std(axis=0)
        },
        'checkpoint_accs': all_checkpoints,
        'convergence_metrics': conv_metrics,
        'batch_level_tracking': batch_tracking_aggregated
    }

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_convergence_with_confidence(aggregated_results, model_names, model_keys,
                                    dataset_name, save_path, epochs):
    """
    Plot validation accuracy curves with confidence bands (mean Â± std)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = ['#d62728', '#1f77b4']  # red, blue
    
    for idx, (name, key) in enumerate(zip(model_names, model_keys)):
        results = aggregated_results[key]
        mean_val = results['val_accs']['mean']
        std_val = results['val_accs']['std']
        
        x = np.arange(len(mean_val))
        
        # Validation accuracy
        axes[0].plot(x, mean_val, label=name, linewidth=2, color=colors[idx], alpha=0.9)
        axes[0].fill_between(x, mean_val - std_val, mean_val + std_val, 
                            alpha=0.2, color=colors[idx])
        
        # Training loss
        mean_loss = results['train_losses']['mean']
        std_loss = results['train_losses']['std']
        axes[1].plot(x, mean_loss, label=name, linewidth=2, color=colors[idx], alpha=0.9)
        axes[1].fill_between(x, mean_loss - std_loss, mean_loss + std_loss,
                            alpha=0.2, color=colors[idx])
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title(f'Validation Accuracy (Mean Â± Std) - {dataset_name}', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Loss', fontsize=12)
    axes[1].set_title(f'Training Loss (Mean Â± Std) - {dataset_name}', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'âœ“ Saved: {save_path}')
    plt.close()

def plot_early_convergence_zoom(aggregated_results, dataset_name, save_path):
    """
    Plot batch-level convergence for first 5 epochs with confidence bands
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728', '#1f77b4']
    labels = ['Standard MLP (X-scaled)', 'RowNorm MLP (U-restricted)']
    
    for idx, key in enumerate(['standard_scaled', 'rownorm_restricted']):
        batch_data = aggregated_results[key]['batch_level_tracking']
        
        if batch_data:
            steps = [batch_data[i]['global_step'] for i in sorted(batch_data.keys())]
            means = [batch_data[i]['mean_val_acc'] for i in sorted(batch_data.keys())]
            stds = [batch_data[i]['std_val_acc'] for i in sorted(batch_data.keys())]
            
            means = np.array(means)
            stds = np.array(stds)
            
            ax.plot(steps, means, label=labels[idx], linewidth=2, 
                   color=colors[idx], alpha=0.9)
            ax.fill_between(steps, means - stds, means + stds,
                           alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Batch (Global Step)', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'Early Convergence: First 5 Epochs (Batch-Level) - {dataset_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'âœ“ Saved: {save_path}')
    plt.close()



def plot_original_style_grid(aggregated_results, dataset_name, save_path, epochs):
    """
    Plot 1x3 grid:
      Col 1: Validation Accuracy (both models)
      Col 2: Training Loss (both models)
      Col 3: Validation Loss (both models)
    Clean layout with legend below plots and non-overlapping title.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Model configs: (key, label, color)
    cfgs = [
        ('standard_scaled', 'Standard MLP (X-scaled)', '#d62728'),   # red
        ('rownorm_restricted', 'RowNorm MLP (U-restricted)', '#1f77b4')  # blue
    ]

    def plot_with_band(ax, y_mean, y_std, label, color):
        x = np.arange(len(y_mean))
        ax.plot(x, y_mean, label=label, linewidth=2.2, color=color)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

    # --- (1) Validation Accuracy ---
    ax = axes[0]
    for key, label, color in cfgs:
        res = aggregated_results[key]
        plot_with_band(ax, res['val_accs']['mean'], res['val_accs']['std'], label, color)
    ax.set_title('Validation Accuracy', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(alpha=0.3)

    # --- (2) Training Loss ---
    ax = axes[1]
    for key, label, color in cfgs:
        res = aggregated_results[key]
        plot_with_band(ax, res['train_losses']['mean'], res['train_losses']['std'], label, color)
    ax.set_title('Training Loss', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)

    # --- (3) Validation Loss ---
    ax = axes[2]
    for key, label, color in cfgs:
        res = aggregated_results[key]
        plot_with_band(ax, res['val_losses']['mean'], res['val_losses']['std'], label, color)
    ax.set_title('Validation Loss', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)

    # Legend BELOW plots (non-overlapping)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=11,
        handlelength=3.5,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, -0.05)  # move below plots
    )

    # Title at top with enough space
    plt.suptitle(
        f'Investigation 2 Extended: X-Restricted Eigenvectors — {dataset_name}',
        fontsize=15,
        y=1.02,
        fontweight='bold'
    )

    # Adjust layout to fit everything neatly
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(bottom=0.18)  # space for legend
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_path}')
    plt.close()

# ============================================================================
# Table Printing Functions
# ============================================================================

def print_convergence_table(aggregated_results, model_names, model_keys, dataset_name, epochs):
    """
    Print validation accuracy at checkpoints
    """
    print('\n' + '='*70)
    print(f'CONVERGENCE TABLE: {dataset_name.upper()}')
    print('='*70)
    print(f'Validation Accuracy (%) at Checkpoints [Mean Â± Std]')
    print('-'*70)
    
    checkpoints = [10, 20, 40, 80, 160, 200]
    header = f"{'Model':<30} " + " ".join([f"Ep{cp:>3}" for cp in checkpoints]) + f"  Final"
    print(header)
    print('-'*70)
    
    for name, key in zip(model_names, model_keys):
        results = aggregated_results[key]
        row = f"{name:<30}"
        
        for cp in checkpoints:
            if cp in results['checkpoint_accs']:
                mean = results['checkpoint_accs'][cp]['mean'] * 100
                std = results['checkpoint_accs'][cp]['std'] * 100
                row += f" {mean:5.1f}Â±{std:3.1f}"
            else:
                row += "    -    "
        
        final_mean = results['test_acc']['mean'] * 100
        final_std = results['test_acc']['std'] * 100
        row += f"  {final_mean:5.1f}Â±{final_std:3.1f}"
        
        print(row)
    
    print('='*70)

def print_convergence_metrics_table(aggregated_results, model_names, model_keys, dataset_name):
    """
    Print convergence speed metrics
    """
    print('\n' + '='*70)
    print(f'CONVERGENCE METRICS: {dataset_name.upper()}')
    print('='*70)
    print(f"{'Model':<30} {'90%':>10} {'95%':>10} {'99%':>10} {'AUC':>8}")
    print('-'*70)
    
    for name, key in zip(model_names, model_keys):
        metrics = aggregated_results[key]['convergence_metrics']
        
        s90_m = metrics['speed_to_90']['mean']
        s90_s = metrics['speed_to_90']['std']
        s95_m = metrics['speed_to_95']['mean']
        s95_s = metrics['speed_to_95']['std']
        s99_m = metrics['speed_to_99']['mean']
        s99_s = metrics['speed_to_99']['std']
        auc_m = metrics['auc']['mean']
        auc_s = metrics['auc']['std']
        
        print(f"{name:<30} {s90_m:4.0f}Â±{s90_s:3.0f}   {s95_m:4.0f}Â±{s95_s:3.0f}   "
              f"{s99_m:4.0f}Â±{s99_s:3.0f}   {auc_m:.3f}Â±{auc_s:.3f}")
    
    print('='*70)
    print('Lower epochs-to-threshold is better (faster convergence)')
    print('='*70)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

if X_raw is None:
    print(f"ERROR: Dataset {DATASET_NAME} has no node features!")
    print("Investigation 2 requires node features to compute X-restricted eigenvectors.")
    sys.exit(1)

d_raw = X_raw.shape[1]

print(f'Nodes: {num_nodes:,}')
print(f'Edges: {edge_index.shape[1]:,}')
print(f'Classes: {num_classes}')
print(f'Raw feature dimension: {d_raw}')

# Determine number of split iterations
if USE_RANDOM_SPLITS:
    num_split_iterations = NUM_RANDOM_SPLITS
    print(f'\nUsing {NUM_RANDOM_SPLITS} random 60/20/20 splits')
else:
    num_split_iterations = 1
    print(f'\nUsing fixed benchmark splits')
    print(f'Train: {len(train_idx_fixed):,} | Val: {len(val_idx_fixed):,} | Test: {len(test_idx_fixed):,}')

# Determine batch size (consistent across all splits)
if USE_RANDOM_SPLITS:
    typical_train_size = int(num_nodes * 0.6)
    batch_size = get_batch_size(typical_train_size)
else:
    batch_size = get_batch_size(len(train_idx_fixed))

print(f'\nBatch size: {batch_size}', end='')
if batch_size < 256:
    print(' (full batch gradient descent)')
else:
    print(f' (mini-batch SGD)')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('Built: adjacency (with self-loops), degree matrix, Laplacian')

# ============================================================================
# 3. Compute X-Restricted Eigenvectors with QR Decomposition
# ============================================================================
print('\n[3/6] Computing X-restricted eigenvectors with QR decomposition...')

X = X_raw.astype(np.float64)

# Check rank and perform QR decomposition
from numpy.linalg import matrix_rank
rank_X = matrix_rank(X, tol=1e-6)
print(f'Rank of X: {rank_X}/{d_raw}')

if rank_X < d_raw:
    print(f'âš ï¸  Warning: X is rank-deficient ({rank_X} < {d_raw})')
    print(f'Performing QR decomposition to handle rank deficiency...')
    
    # QR decomposition
    Q, R = np.linalg.qr(X, mode='reduced')
    
    # CRITICAL: Truncate Q to effective rank
    Q = Q[:, :rank_X]
    
    print(f'âœ“ Reduced dimension from {d_raw} to {rank_X} (effective rank)')
    print(f'  Q shape after truncation: {Q.shape}')
    print(f'  span(Q) = span(X) â€” same subspace, orthonormal basis')
    
    X_for_projection = Q
    d_effective = rank_X
else:
    print(f'âœ“ X is full-rank')
    X_for_projection = X
    d_effective = d_raw

# Project L and D into span(X)
print(f'Projecting Laplacian into span(X) (effective dimension {d_effective})...')

LX = L @ X_for_projection
DX = D @ X_for_projection

Lr = X_for_projection.T @ LX
Dr = X_for_projection.T @ DX

# Symmetrize for numerical stability
Lr = 0.5 * (Lr + Lr.T)
Dr = 0.5 * (Dr + Dr.T)

print(f'Solving reduced generalized eigenproblem in R^{d_effective}...')

# No regularization needed with QR!
try:
    w, V = la.eigh(Lr, Dr)
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

rank_U = matrix_rank(U, tol=1e-6)
print(f'Rank of U: {rank_U}/{d_effective}')
print(f'âœ“ span(U) = span(X) â€” same subspace, different basis')

# Verify D-orthonormality of U
DU = (D @ U).astype(np.float64)
G = U.astype(np.float64).T @ DU
deviation = np.abs(G - np.eye(U.shape[1])).max()
print(f'D-orthonormality check: max |U^T D U - I| = {deviation:.2e}')

if deviation < 1e-6:
    print(f'âœ“ Excellent D-orthonormality!')
elif deviation < 1e-4:
    print(f'âœ“ Good D-orthonormality')
else:
    print(f'âš ï¸  Warning: Large D-orthonormality deviation')

# ============================================================================
# 4. Prepare Labels (Common for All Splits)
# ============================================================================
print('\n[4/6] Preparing labels...')
print('âœ“ Labels prepared')

# ============================================================================
# 5. Train Models with Multiple Seeds and Splits
# ============================================================================
print(f'\n[5/6] Training 2 models with {NUM_SEEDS} random seeds each...')
if USE_RANDOM_SPLITS:
    print(f'Across {NUM_RANDOM_SPLITS} random splits')
    print(f'Total runs: {2 * NUM_SEEDS * NUM_RANDOM_SPLITS} = {2}(models) Ã— {NUM_SEEDS}(seeds) Ã— {NUM_RANDOM_SPLITS}(splits)')
else:
    print(f'Total runs: {2 * NUM_SEEDS} = {2}(models) Ã— {NUM_SEEDS}(seeds)')

print(f'\nModel dimensions:')
print(f'  Model A (Standard MLP): input_dim = {d_raw} (original X features)')
print(f'  Model B (RowNorm MLP):  input_dim = {d_effective} (restricted eigenvectors U)')
if d_raw != d_effective:
    print(f'  â†’ Different input dimensions due to QR decomposition')

model_configs = [
    ('Standard MLP on X (train-scaled)', 'standard_scaled', 'X_scaled'),
    ('RowNorm MLP on U (restricted)', 'rownorm_restricted', 'U_restricted'),
]

# Storage for results across all seeds and splits
all_results = {key: [] for _, key, _ in model_configs}

# Outer loop: splits
for split_idx in range(num_split_iterations):
    
    # Get train/val/test indices for this split
    if USE_RANDOM_SPLITS:
        print(f'\n{"="*70}')
        print(f'RANDOM SPLIT {split_idx+1}/{NUM_RANDOM_SPLITS}')
        print(f'{"="*70}')
        
        train_idx, val_idx, test_idx = create_random_split(
            num_nodes, 
            train_ratio=0.6, 
            val_ratio=0.2, 
            seed=split_idx
        )
        print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')
    else:
        train_idx = train_idx_fixed
        val_idx = val_idx_fixed
        test_idx = test_idx_fixed
    
    # Prepare labels for this split
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)
    y_test = torch.from_numpy(labels[test_idx]).long().to(device)
    
    # Prepare features for Model A: Raw X with train-only StandardScaler
    scaler = StandardScaler().fit(X_raw[train_idx])
    X_train_std = torch.from_numpy(scaler.transform(X_raw[train_idx])).float().to(device)
    X_val_std = torch.from_numpy(scaler.transform(X_raw[val_idx])).float().to(device)
    X_test_std = torch.from_numpy(scaler.transform(X_raw[test_idx])).float().to(device)
    
    # Prepare features for Model B: Restricted eigenvectors U (no scaling)
    # Note: U has dimension d_effective (after QR), while X has dimension d_raw
    U_train = torch.from_numpy(U[train_idx]).float().to(device)
    U_val = torch.from_numpy(U[val_idx]).float().to(device)
    U_test = torch.from_numpy(U[test_idx]).float().to(device)
    
    # Inner loop: seeds
    for seed_idx in range(NUM_SEEDS):
        seed = seed_idx if not USE_RANDOM_SPLITS else (split_idx * NUM_SEEDS + seed_idx)
        
        print(f'\n--- Split {split_idx+1}/{num_split_iterations}, Seed {seed_idx+1}/{NUM_SEEDS} (global seed={seed}) ---')
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Model A: Standard MLP on X (scaled)
        print('\nModel A: Standard MLP on X (train-scaled)...')
        
        train_loader_X = DataLoader(
            TensorDataset(X_train_std, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        
        model_std = StandardMLP(d_raw, HIDDEN_DIM, num_classes)  # Use d_raw (original dimension)
        
        # Sanity check
        assert X_train_std.shape[1] == d_raw, f"Dimension mismatch: X has {X_train_std.shape[1]} but model expects {d_raw}"
        result_std = train_with_granular_tracking(
            model_std, train_loader_X, X_val_std, y_val, X_test_std, y_test,
            epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            device=device, verbose=False
        )
        all_results['standard_scaled'].append(result_std)
        print(f'  Test Acc: {result_std["test_acc"]:.4f}')
        
        # Model B: RowNorm MLP on U (restricted)
        print('\nModel B: RowNorm MLP on U (restricted)...')
        
        train_loader_U = DataLoader(
            TensorDataset(U_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        
        model_rn = RowNormMLP(d_effective, HIDDEN_DIM, num_classes)
        
        # Sanity check
        assert U_train.shape[1] == d_effective, f"Dimension mismatch: U has {U_train.shape[1]} but model expects {d_effective}"
        result_rn = train_with_granular_tracking(
            model_rn, train_loader_U, U_val, y_val, U_test, y_test,
            epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            device=device, verbose=False
        )
        all_results['rownorm_restricted'].append(result_rn)
        print(f'  Test Acc: {result_rn["test_acc"]:.4f}')

# ============================================================================
# 6. Aggregate Results and Generate Outputs
# ============================================================================
print(f'\n[6/6] Aggregating results and generating outputs...')

if USE_RANDOM_SPLITS:
    print(f'Aggregating across {NUM_RANDOM_SPLITS} splits Ã— {NUM_SEEDS} seeds = {NUM_RANDOM_SPLITS * NUM_SEEDS} runs per model')
else:
    print(f'Aggregating across {NUM_SEEDS} seeds')

# Aggregate results
aggregated_results = {}
for model_name, model_key, _ in model_configs:
    print(f'Aggregating: {model_name}...')
    aggregated_results[model_key] = aggregate_results(all_results[model_key])

model_names = [name for name, _, _ in model_configs]
model_keys = [key for _, key, _ in model_configs]

# Generate plots
print('\nGenerating plots...')

# Plot 1: Original 2Ã—3 grid (from Investigation 2 Extended)
plot_path_grid = f'{output_base}/plots/original_2x3_grid.png'
plot_original_style_grid(aggregated_results, DATASET_NAME, plot_path_grid, EPOCHS)

# Plot 2: Convergence with confidence bands (from Investigation 1 Ver2)
plot_path_convergence = f'{output_base}/plots/convergence_with_confidence.png'
plot_convergence_with_confidence(aggregated_results, model_names, model_keys,
                                 DATASET_NAME, plot_path_convergence, EPOCHS)

# Plot 3: Early convergence zoom (from Investigation 1 Ver2)
plot_path_zoom = f'{output_base}/plots/early_convergence_zoom.png'
plot_early_convergence_zoom(aggregated_results, DATASET_NAME, plot_path_zoom)

# Print tables
print_convergence_table(aggregated_results, model_names, model_keys, DATASET_NAME, EPOCHS)
print_convergence_metrics_table(aggregated_results, model_names, model_keys, DATASET_NAME)

# Save metrics
metrics = {
    'dataset': DATASET_NAME,
    'split_type': split_type,
    'num_seeds': NUM_SEEDS,
    'num_splits': num_split_iterations,
    'total_runs_per_model': NUM_SEEDS * num_split_iterations,
    'd_raw': int(d_raw),
    'd_effective': int(d_effective),
    'rank_deficiency': bool(rank_X < d_raw),
    'epochs': int(EPOCHS),
    'hidden_dim': int(HIDDEN_DIM),
    'batch_size': int(batch_size),
    'learning_rate': float(LEARNING_RATE),
    'weight_decay': float(WEIGHT_DECAY),
    'models': {}
}

for model_name, model_key, _ in model_configs:
    metrics['models'][model_key] = {
        'name': model_name,
        'test_accuracy': aggregated_results[model_key]['test_acc'],
        'convergence_metrics': aggregated_results[model_key]['convergence_metrics'],
        'checkpoint_accuracies': aggregated_results[model_key]['checkpoint_accs']
    }

metrics_path = f'{output_base}/metrics/results_aggregated.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\nâœ“ Saved aggregated metrics: {metrics_path}')

# ============================================================================
# Print Final Summary
# ============================================================================
print('\n' + '='*70)
print('FINAL SUMMARY')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Split type: {split_type}')
print(f'Raw feature dimension: {d_raw}')
print(f'Effective dimension (after QR): {d_effective}')
if rank_X < d_raw:
    print(f'Rank deficiency handled: {d_raw} â†’ {d_effective}')
print(f'span(U) = span(X) â€” same subspace, different basis')
print(f'Hidden dimension: {HIDDEN_DIM}')
print(f'Training epochs: {EPOCHS}')
print(f'Batch size: {batch_size}')
print(f'Random seeds: {NUM_SEEDS} (fixed), {NUM_SEEDS * NUM_RANDOM_SPLITS} total (random)')

print('\n--- MODEL PERFORMANCE ---')
for model_name, model_key, _ in model_configs:
    test_acc = aggregated_results[model_key]['test_acc']
    final_val_acc = aggregated_results[model_key]['convergence_metrics']['final_acc']
    
    # Determine correct input dimension for parameter count
    input_dim = d_raw if "Standard" in model_name else d_effective
    
    print(f'\n{model_name}:')
    print(f'  Test Accuracy:  {test_acc["mean"]:.4f} Â± {test_acc["std"]:.4f}')
    print(f'  Final Val Acc:  {final_val_acc["mean"]:.4f} Â± {final_val_acc["std"]:.4f}')
    print(f'  Input dim: {input_dim}')
    print(f'  Params: {sum(p.numel() for p in (StandardMLP(input_dim, HIDDEN_DIM, num_classes) if "Standard" in model_name else RowNormMLP(input_dim, HIDDEN_DIM, num_classes)).parameters()):,}')

# Compare models
std_test = aggregated_results['standard_scaled']['test_acc']['mean']
rn_test = aggregated_results['rownorm_restricted']['test_acc']['mean']
gap = rn_test - std_test
gap_pct = 100 * gap / std_test

print('\n--- COMPARISON ---')
print(f'RowNorm vs Standard: {gap:+.4f} ({gap_pct:+.1f}%)')
if abs(gap) < 0.01:
    print('Result: Small difference (basis sensitivity minimal)')
elif gap > 0:
    print('Result: RowNorm better (basis change helps despite same span)')
else:
    print('Result: Standard better (scaling more important than basis)')

print('\n' + '='*70)
print('âœ“ Investigation 2 Extended Ver2 Complete!')
print('='*70)