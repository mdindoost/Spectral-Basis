"""
Investigation 1 Extended Version 2: True Eigenvectors with Statistical Rigor
============================================================================

Usage:
    # Fixed benchmark splits (default)
    python experiments/investigation1_extended_ver2.py [dataset_name]
    
    # Random 60/20/20 splits
    python experiments/investigation1_extended_ver2.py [dataset_name] --random-splits
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed

Improvements in v2:
- NetworkX for proper connected component detection
- Multiple random seeds (5 runs) with aggregated statistics
- Adaptive batch size (128 for large datasets, full batch for small)
- Granular batch-level tracking for first 5 epochs
- Quantitative convergence metrics
- Enhanced plots with confidence bands and early convergence zoom
- Support for both fixed benchmark splits and random 60/20/20 splits

Expected finding: RowNorm MLP converges significantly faster than Standard MLP
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx

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
EPOCHS = 100
HIDDEN_DIM = 256
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 5e-4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

split_type = 'random-splits' if USE_RANDOM_SPLITS else 'fixed-splits'
print('='*70)
print(f'INVESTIGATION 1 EXTENDED V2: TRUE EIGENVECTORS - {DATASET_NAME.upper()}')
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
output_base = f'results/investigation1_v2/{DATASET_NAME}/{split_type}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def get_batch_size(train_size):
    """
    Adaptive batch size strategy:
    - Large datasets (>256 samples): use batch_size=128
    - Small datasets (≤256 samples): use full batch
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

def compute_convergence_metrics(val_accs):
    """
    Compute quantitative convergence metrics
    
    Returns:
        dict with:
        - speed_to_90: epochs to reach 90% of final accuracy
        - speed_to_95: epochs to reach 95% of final accuracy
        - speed_to_99: epochs to reach 99% of final accuracy
        - auc: area under convergence curve (normalized)
        - convergence_rate: average improvement per epoch (first 20 epochs)
        - final_acc: final validation accuracy
    """
    val_accs = np.array(val_accs)
    final_acc = val_accs[-1]
    
    # Speed to reach X% of final accuracy
    threshold_90 = 0.90 * final_acc
    threshold_95 = 0.95 * final_acc
    threshold_99 = 0.99 * final_acc
    
    speed_to_90 = np.argmax(val_accs >= threshold_90) if np.any(val_accs >= threshold_90) else len(val_accs)
    speed_to_95 = np.argmax(val_accs >= threshold_95) if np.any(val_accs >= threshold_95) else len(val_accs)
    speed_to_99 = np.argmax(val_accs >= threshold_99) if np.any(val_accs >= threshold_99) else len(val_accs)
    
    # Area under curve (normalized by number of epochs)
    auc = np.trapz(val_accs) / len(val_accs)
    
    # Convergence rate (first 20 epochs)
    early_phase = min(20, len(val_accs))
    early_accs = val_accs[:early_phase]
    convergence_rate = (early_accs[-1] - early_accs[0]) / len(early_accs) if len(early_accs) > 1 else 0.0
    
    return {
        'speed_to_90': int(speed_to_90),
        'speed_to_95': int(speed_to_95),
        'speed_to_99': int(speed_to_99),
        'auc': float(auc),
        'convergence_rate': float(convergence_rate),
        'final_acc': float(final_acc)
    }

def train_with_granular_tracking(model, train_loader, X_val, y_val, X_test, y_test,
                                 epochs=100, lr=1e-2, weight_decay=5e-4, 
                                 device='cpu', verbose=True):
    """
    Training with:
    - Batch-level tracking for first 5 epochs
    - Epoch-level tracking for all epochs
    - Checkpoint tracking at [10, 20, 40, 80, 160, 200]
    - CosineAnnealingLR scheduler
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    
    # Epoch-level tracking
    train_losses = []
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
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / max(1, len(train_loader)))
        val_accs.append(val_acc)
        
        # Record checkpoint accuracy
        if (ep + 1) in checkpoints:
            checkpoint_accs[ep + 1] = val_acc
        
        # Step scheduler
        scheduler.step()
        
        if verbose and (ep + 1) % 20 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f'  Epoch {ep+1}/{epochs}  TrainLoss={train_losses[-1]:.4f}  '
                  f'ValAcc={val_acc:.4f}  LR={current_lr:.6f}')
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    # Compute convergence metrics
    convergence_metrics = compute_convergence_metrics(val_accs)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'checkpoint_accs': checkpoint_accs,
        'batch_level_tracking': batch_level_tracking,
        'convergence_metrics': convergence_metrics
    }

def aggregate_results(results_list):
    """
    Aggregate results across multiple seeds
    
    Args:
        results_list: list of result dicts from train_with_granular_tracking
    
    Returns:
        dict with mean and std for all metrics
    """
    # Test accuracies
    test_accs = [r['test_acc'] for r in results_list]
    
    # Validation accuracy curves
    val_accs_array = np.array([r['val_accs'] for r in results_list])
    
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
            'mean': val_accs_array.mean(axis=0).tolist(),
            'std': val_accs_array.std(axis=0).tolist()
        },
        'train_losses': {
            'mean': train_losses_array.mean(axis=0).tolist(),
            'std': train_losses_array.std(axis=0).tolist()
        },
        'checkpoint_accs': all_checkpoints,
        'convergence_metrics': conv_metrics,
        'batch_tracking': batch_tracking_aggregated
    }

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_convergence_with_confidence(aggregated_results, model_names, model_keys, 
                                    dataset_name, output_path, epochs):
    """
    Plot convergence curves with confidence bands (±1 std)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    # Left: Validation accuracy with confidence bands
    for idx, (name, key) in enumerate(zip(model_names, model_keys)):
        mean_accs = np.array(aggregated_results[key]['val_accs']['mean'])
        std_accs = np.array(aggregated_results[key]['val_accs']['std'])
        
        color = colors[idx % len(colors)]
        axes[0].plot(mean_accs, label=name, linewidth=2, alpha=0.85, color=color)
        axes[0].fill_between(range(len(mean_accs)), 
                            mean_accs - std_accs,
                            mean_accs + std_accs,
                            alpha=0.2, color=color)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title(f'Validation Accuracy (Mean ± Std) - {dataset_name}', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    # Right: Training loss with confidence bands
    for idx, (name, key) in enumerate(zip(model_names, model_keys)):
        mean_losses = np.array(aggregated_results[key]['train_losses']['mean'])
        std_losses = np.array(aggregated_results[key]['train_losses']['std'])
        
        color = colors[idx % len(colors)]
        axes[1].plot(mean_losses, label=name, linewidth=2, alpha=0.85, color=color)
        axes[1].fill_between(range(len(mean_losses)), 
                            mean_losses - std_losses,
                            mean_losses + std_losses,
                            alpha=0.2, color=color)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Loss', fontsize=12)
    axes[1].set_title(f'Training Loss (Mean ± Std) - {dataset_name}', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')

def plot_early_convergence_zoom(aggregated_results, dataset_name, output_path):
    """
    Plot first 5 epochs at batch-level granularity
    Focus on Standard (train-scaled) vs RowNorm comparison
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Extract batch-level data for key comparison
    key_models = [
        ('Standard MLP (train-scaled)', 'standard_train_scaled', 'red'),
        ('Row-Normalized MLP', 'rownorm', 'blue')
    ]
    
    for name, key, color in key_models:
        batch_data = aggregated_results[key]['batch_tracking']
        if batch_data:
            steps = [batch_data[i]['global_step'] for i in sorted(batch_data.keys())]
            mean_accs = [batch_data[i]['mean_val_acc'] for i in sorted(batch_data.keys())]
            std_accs = [batch_data[i]['std_val_acc'] for i in sorted(batch_data.keys())]
            
            ax.plot(steps, mean_accs, label=name, linewidth=2.5, alpha=0.9, color=color)
            ax.fill_between(steps, 
                           np.array(mean_accs) - np.array(std_accs),
                           np.array(mean_accs) + np.array(std_accs),
                           alpha=0.2, color=color)
    
    ax.set_xlabel('Batch (Global Step)', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'Early Convergence: First 5 Epochs (Batch-Level) - {dataset_name}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')

def print_convergence_table(aggregated_results, model_names, model_keys, dataset_name, epochs):
    """
    Print convergence table with mean ± std
    """
    print('\n' + '='*90)
    print('CONVERGENCE ANALYSIS: Validation Accuracy at Checkpoints (Mean ± Std)')
    print('='*90)
    
    checkpoints = [cp for cp in [10, 20, 40, 80, 160, 200] if cp <= epochs]
    
    # Header
    header = f"{'Model':<40}"
    for cp in checkpoints:
        header += f"{cp:>10}"
    header += f"{'Final':>10}"
    print(header)
    print('-' * 90)
    
    # Print all models
    for name, key in zip(model_names, model_keys):
        short_name = name.replace(' (train-only scaled)', '').replace(' MLP', '')
        row = f"{dataset_name}–{short_name:<35}"
        
        for cp in checkpoints:
            if cp in aggregated_results[key]['checkpoint_accs']:
                mean = aggregated_results[key]['checkpoint_accs'][cp]['mean']
                std = aggregated_results[key]['checkpoint_accs'][cp]['std']
                row += f"{mean*100:>5.1f}±{std*100:.1f}%"
            else:
                row += "         -"
        
        # Final accuracy
        final_mean = aggregated_results[key]['test_acc']['mean']
        final_std = aggregated_results[key]['test_acc']['std']
        row += f"{final_mean*100:>5.1f}±{final_std*100:.1f}%"
        print(row)
    
    print('-' * 90)
    
    # Key comparison at epoch 20
    print('\n*** KEY COMPARISON at Epoch 20 ***')
    key_models = [
        ('Standard MLP (train-scaled)', 'standard_train_scaled'),
        ('Row-Normalized MLP', 'rownorm')
    ]
    
    if 20 in aggregated_results['standard_train_scaled']['checkpoint_accs'] and \
       20 in aggregated_results['rownorm']['checkpoint_accs']:
        std_mean = aggregated_results['standard_train_scaled']['checkpoint_accs'][20]['mean']
        rn_mean = aggregated_results['rownorm']['checkpoint_accs'][20]['mean']
        
        print(f'Standard MLP (train-scaled): {std_mean*100:.1f}%')
        print(f'Row-Normalized MLP:         {rn_mean*100:.1f}%')
        
        if rn_mean > std_mean:
            speedup = (rn_mean - std_mean) / std_mean * 100
            print(f'✓ RowNorm converges {speedup:.1f}% faster at epoch 20!')
        else:
            print(f'✗ No convergence advantage detected')
    
    print('='*90)

def print_convergence_metrics_table(aggregated_results, model_names, model_keys, dataset_name):
    """
    Print quantitative convergence metrics
    """
    print('\n' + '='*100)
    print('QUANTITATIVE CONVERGENCE METRICS (Mean ± Std)')
    print('='*100)
    
    print(f"{'Metric':<30} | {'Standard MLP':<20} | {'RowNorm MLP':<20} | {'Improvement':<15}")
    print('-' * 100)
    
    std_key = 'standard_train_scaled'
    rn_key = 'rownorm'
    
    metrics = [
        ('Final Test Acc (%)', 'final_acc', True),
        ('Speed to 90% (epochs)', 'speed_to_90', False),
        ('Speed to 95% (epochs)', 'speed_to_95', False),
        ('Speed to 99% (epochs)', 'speed_to_99', False),
        ('AUC (normalized)', 'auc', True),
        ('Convergence Rate', 'convergence_rate', True)
    ]
    
    for metric_name, metric_key, higher_better in metrics:
        std_mean = aggregated_results[std_key]['convergence_metrics'][metric_key]['mean']
        std_std = aggregated_results[std_key]['convergence_metrics'][metric_key]['std']
        rn_mean = aggregated_results[rn_key]['convergence_metrics'][metric_key]['mean']
        rn_std = aggregated_results[rn_key]['convergence_metrics'][metric_key]['std']
        
        if 'acc' in metric_key.lower() or metric_key == 'final_acc':
            std_str = f"{std_mean*100:>6.2f}±{std_std*100:.2f}%"
            rn_str = f"{rn_mean*100:>6.2f}±{rn_std*100:.2f}%"
            if higher_better:
                improvement = ((rn_mean - std_mean) / std_mean * 100)
                imp_str = f"{improvement:+.1f}%"
            else:
                improvement = ((std_mean - rn_mean) / std_mean * 100)
                imp_str = f"{improvement:+.1f}%"
        elif 'speed' in metric_key.lower():
            std_str = f"{std_mean:>8.1f}±{std_std:.1f}"
            rn_str = f"{rn_mean:>8.1f}±{rn_std:.1f}"
            speedup = std_mean / rn_mean if rn_mean > 0 else 1.0
            imp_str = f"{speedup:.2f}x faster"
        else:
            std_str = f"{std_mean:>9.4f}±{std_std:.4f}"
            rn_str = f"{rn_mean:>9.4f}±{rn_std:.4f}"
            if higher_better:
                improvement = ((rn_mean - std_mean) / std_mean * 100)
                imp_str = f"{improvement:+.1f}%"
            else:
                improvement = ((std_mean - rn_mean) / std_mean * 100)
                imp_str = f"{improvement:+.1f}%"
        
        print(f"{metric_name:<30} | {std_str:<20} | {rn_str:<20} | {imp_str:<15}")
    
    print('='*100)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

(edge_index, node_features, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Nodes: {num_nodes:,}')
print(f'Edges: {edge_index.shape[1]:,}')
print(f'Classes: {num_classes}')
if node_features is not None:
    print(f'Original features: {node_features.shape[1]}')

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
    # Estimate based on typical split size
    typical_train_size = int(num_nodes * 0.6)
    batch_size = get_batch_size(typical_train_size)
else:
    batch_size = get_batch_size(len(train_idx_fixed))

print(f'\nBatch size: {batch_size}', end='')
if batch_size == len(train_idx_fixed) if not USE_RANDOM_SPLITS else batch_size > 256:
    print(' (full batch gradient descent)' if batch_size < 256 else f' ({batch_size} samples per batch)')
else:
    print(f' (mini-batch SGD)')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('Built: adjacency (with self-loops), degree matrix, Laplacian')

# ============================================================================
# 3. Compute True Eigenvectors with Proper Component Detection
# ============================================================================
print('\n[3/6] Computing true eigenvectors...')

# Step 1: Count connected components using NetworkX
print('Detecting connected components with NetworkX...')
G = nx.from_scipy_sparse_array(adj)
ncc = nx.number_connected_components(G)
print(f'Number of connected components: {ncc}')

# Step 2: Compute k = 2*num_classes eigenvectors
k_target = 2 * num_classes
k_compute = ncc + k_target

print(f'Computing {k_compute} eigenvectors (ncc={ncc} + k_target={k_target})...')
print(f'Solving (D-A)x = λDx for smallest eigenvalues...')

eigenvalues, eigenvectors = eigsh(
    L.astype(np.float64),
    k=k_compute,
    M=D.astype(np.float64),
    which='SM',
    tol=1e-4
)

# Step 3: Drop first ncc eigenvalues (connected components)
if ncc > 0:
    print(f'Dropping first {ncc} eigenvalues (connected component eigenvectors)...')
    eigenvalues = eigenvalues[ncc:]
    eigenvectors = eigenvectors[:, ncc:]

k = len(eigenvalues)
print(f'Using {k} eigenvectors for classification')
print(f'Eigenvalues range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]')

# Verify D-orthonormality
DX = D @ eigenvectors
G_matrix = eigenvectors.T @ DX
deviation = np.abs(G_matrix - np.eye(k)).max()
print(f'D-orthonormality check: max |X^T D X - I| = {deviation:.2e}')

X = eigenvectors.astype(np.float32)
print(f'Eigenvector matrix shape: {X.shape}')

# ============================================================================
# 4. Prepare Labels (Common for All Splits)
# ============================================================================
print('\n[4/6] Preparing labels...')
print('✓ Labels prepared')

# ============================================================================
# 5. Train All Models with Multiple Seeds and Splits
# ============================================================================
print(f'\n[5/6] Training 6 models with {NUM_SEEDS} random seeds each...')
if USE_RANDOM_SPLITS:
    print(f'Across {NUM_RANDOM_SPLITS} random splits')
    print(f'Total runs: {6 * NUM_SEEDS * NUM_RANDOM_SPLITS} = {6}(models) × {NUM_SEEDS}(seeds) × {NUM_RANDOM_SPLITS}(splits)')
else:
    print(f'Total runs: {6 * NUM_SEEDS} = {6}(models) × {NUM_SEEDS}(seeds)')

model_configs = [
    ('Standard MLP (train-only scaled)', 'standard_train_scaled', 'scaled_train'),
    ('Full-Data Scaled MLP', 'standard_full_scaled', 'scaled_full'),
    ('No-Scaling MLP', 'standard_no_scale', 'no_scale'),
    ('Eigenvalue-Weighted MLP', 'standard_weighted', 'weighted'),
    ('Row-Normalized MLP', 'rownorm', 'rownorm'),
    ('Cosine Classifier MLP', 'cosine', 'cosine'),
]

# Storage for results across all seeds and splits
all_results = {key: [] for _, key, _ in model_configs}

from torch.utils.data import DataLoader, TensorDataset

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
    
    # Inner loop: seeds
    for seed_idx in range(NUM_SEEDS):
        if USE_RANDOM_SPLITS:
            print(f'\n  Seed {seed_idx+1}/{NUM_SEEDS} (Split {split_idx+1}/{NUM_RANDOM_SPLITS})')
        else:
            print(f'\n{"="*70}')
            print(f'SEED {seed_idx+1}/{NUM_SEEDS}')
            print(f'{"="*70}')
        
        # Set all random seeds
        torch.manual_seed(seed_idx)
        np.random.seed(seed_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_idx)
        
        # Prepare features for this seed
        # Model 1: Train-only StandardScaler
        scaler = StandardScaler().fit(X[train_idx])
        X_train_std = torch.from_numpy(scaler.transform(X[train_idx])).float().to(device)
        X_val_std = torch.from_numpy(scaler.transform(X[val_idx])).float().to(device)
        X_test_std = torch.from_numpy(scaler.transform(X[test_idx])).float().to(device)
        
        # Model 2: Full-data StandardScaler
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
        
        # Models 5-6: Raw eigenvectors (RowNorm handles internally)
        X_train_rn = torch.from_numpy(X[train_idx]).float().to(device)
        X_val_rn = torch.from_numpy(X[val_idx]).float().to(device)
        X_test_rn = torch.from_numpy(X[test_idx]).float().to(device)
        
        # Labels for this split
        y_train = torch.from_numpy(labels[train_idx]).long().to(device)
        y_val = torch.from_numpy(labels[val_idx]).long().to(device)
        y_test = torch.from_numpy(labels[test_idx]).long().to(device)
        
        # Train each model
        for model_name, model_key, feature_type in model_configs:
            if USE_RANDOM_SPLITS:
                print(f'    → {model_name}')
            else:
                print(f'\n--- {model_name} (Seed {seed_idx+1}) ---')
            
            # Select features
            if feature_type == 'scaled_train':
                X_tr, X_v, X_te = X_train_std, X_val_std, X_test_std
                model = StandardMLP(k, HIDDEN_DIM, num_classes).to(device)
            elif feature_type == 'scaled_full':
                X_tr, X_v, X_te = X_train_full, X_val_full, X_test_full
                model = StandardMLP(k, HIDDEN_DIM, num_classes).to(device)
            elif feature_type == 'no_scale':
                X_tr, X_v, X_te = X_train_ns, X_val_ns, X_test_ns
                model = StandardMLP(k, HIDDEN_DIM, num_classes).to(device)
            elif feature_type == 'weighted':
                X_tr, X_v, X_te = X_train_w, X_val_w, X_test_w
                model = StandardMLP(k, HIDDEN_DIM, num_classes).to(device)
            elif feature_type == 'rownorm':
                X_tr, X_v, X_te = X_train_rn, X_val_rn, X_test_rn
                model = RowNormMLP(k, HIDDEN_DIM, num_classes).to(device)
            elif feature_type == 'cosine':
                X_tr, X_v, X_te = X_train_rn, X_val_rn, X_test_rn
                model = CosineRowNormMLP(k, HIDDEN_DIM, num_classes).to(device)
            
            # Create data loader
            train_loader = DataLoader(TensorDataset(X_tr, y_train), 
                                     batch_size=batch_size, shuffle=True)
            
            # Train with granular tracking
            verbose = not USE_RANDOM_SPLITS  # Only verbose for fixed splits
            results = train_with_granular_tracking(
                model, train_loader, X_v, y_val, X_te, y_test,
                epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                device=device, verbose=verbose
            )
            
            all_results[model_key].append(results)

# ============================================================================
# 6. Aggregate Results and Generate Outputs
# ============================================================================
print(f'\n[6/6] Aggregating results and generating outputs...')

if USE_RANDOM_SPLITS:
    print(f'Aggregating across {NUM_RANDOM_SPLITS} splits × {NUM_SEEDS} seeds = {NUM_RANDOM_SPLITS * NUM_SEEDS} runs per model')
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

# Plot 1: Convergence with confidence bands
plot_path_convergence = f'{output_base}/plots/convergence_with_confidence.png'
plot_convergence_with_confidence(aggregated_results, model_names, model_keys,
                                 DATASET_NAME, plot_path_convergence, EPOCHS)

# Plot 2: Early convergence zoom (batch-level)
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
    'k_eigenvectors': int(k),
    'ncc_components': int(ncc),
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
print(f'\n✓ Saved aggregated metrics: {metrics_path}')

# ============================================================================
# Print Final Summary
# ============================================================================
print('\n' + '='*70)
print('FINAL SUMMARY')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Split type: {split_type}')
print(f'Features: {k} eigenvectors from (D−A)x = λ D x (D-orthonormal)')
print(f'Connected components: {ncc}')
if USE_RANDOM_SPLITS:
    print(f'Statistical robustness: {NUM_RANDOM_SPLITS} random splits × {NUM_SEEDS} seeds = {NUM_RANDOM_SPLITS * NUM_SEEDS} runs/model')
else:
    print(f'Statistical robustness: {NUM_SEEDS} random seeds')
print(f'Batch size: {batch_size}')
print(f'Training epochs: {EPOCHS}')

print('\nTest Accuracy (Mean ± Std):')
for model_name, model_key, _ in model_configs:
    mean_acc = aggregated_results[model_key]['test_acc']['mean']
    std_acc = aggregated_results[model_key]['test_acc']['std']
    print(f'  {model_name:45s}: {mean_acc*100:5.2f}% ± {std_acc*100:.2f}%')

print('\n' + '='*70)
print('COMPARISON (Relative to Standard MLP with train-only scaling)')
print('='*70)

baseline_mean = aggregated_results['standard_train_scaled']['test_acc']['mean']

comparison_models = [
    ('Full-Data Scaled vs Standard', 'standard_full_scaled'),
    ('No-Scaling vs Standard', 'standard_no_scale'),
    ('Eigenvalue-Weighted vs Standard', 'standard_weighted'),
    ('Row-Normalized vs Standard', 'rownorm'),
    ('Cosine Classifier vs Standard', 'cosine'),
]

for name, key in comparison_models:
    model_mean = aggregated_results[key]['test_acc']['mean']
    diff = model_mean - baseline_mean
    pct = 100 * (model_mean / baseline_mean - 1)
    print(f'{name:38s}: {diff:+.4f} ({pct:+.1f}%)')

print('='*70)
print(f'✓ Investigation 1 Extended V2 complete for {DATASET_NAME} ({split_type})!')
print(f'✓ All results saved to: {output_base}/')
print('='*70)