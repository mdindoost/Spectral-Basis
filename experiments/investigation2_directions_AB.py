"""
Investigation 2: Directions A & B - Complete Basis Sensitivity Study
====================================================================

Three experiments to separate basis effects from model effects:
  (a) X → StandardScaler → Standard MLP        [Baseline]
  (b) V → StandardScaler → Standard MLP        [NEW - Isolates basis effect]
  (c) V → RowNorm MLP (no scaling)            [Tests model on restricted basis]

Direction A (b vs c): Does RowNorm help with restricted eigenvectors?
Direction B (a vs b): Does basis choice matter with identical preprocessing?

Usage:
    # Fixed benchmark splits (default)
    python experiments/investigation2_directions_AB.py [dataset_name]
    
    # Random 60/20/20 splits
    python experiments/investigation2_directions_AB.py [dataset_name] --random-splits
    
    dataset_name: ogbn-arxiv (default), cora, citeseer, pubmed
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
USE_RANDOM_SPLITS = '--random-splits' in sys.argv

VALID_DATASETS = ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed']
if DATASET_NAME not in VALID_DATASETS:
    print(f"Error: Invalid dataset '{DATASET_NAME}'")
    print(f"Valid datasets: {', '.join(VALID_DATASETS)}")
    sys.exit(1)

# Experimental parameters
NUM_SEEDS = 5
NUM_RANDOM_SPLITS = 5 if USE_RANDOM_SPLITS else 1

# Hyperparameters
HIDDEN_DIM = 256
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*70)
print(f'INVESTIGATION 2: DIRECTIONS A & B - {DATASET_NAME.upper()}')
print('='*70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Random splits: {"Yes" if USE_RANDOM_SPLITS else "No (fixed benchmark)"}')
print('='*70)

# Create output directories
split_type = 'random_splits' if USE_RANDOM_SPLITS else 'fixed_splits'
output_base = f'results/investigation2_directions_AB/{DATASET_NAME}/{split_type}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def create_random_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create random 60/20/20 split"""
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

def compute_convergence_metrics(val_accs, epochs):
    """Compute convergence speed metrics"""
    final_acc = val_accs[-1]
    
    # Speed to thresholds
    thresholds = [0.90, 0.95, 0.99]
    speeds = []
    for thresh in thresholds:
        target = final_acc * thresh
        epoch = next((i for i, acc in enumerate(val_accs) if acc >= target), epochs)
        speeds.append(epoch)
    
    # AUC (normalized)
    auc = np.trapz(val_accs, dx=1) / (epochs * final_acc) if final_acc > 0 else 0
    
    # Convergence rate (first 20 epochs)
    rate = (val_accs[min(19, len(val_accs)-1)] - val_accs[0]) / min(20, len(val_accs)) if len(val_accs) > 0 else 0
    
    return {
        'speed_to_90': speeds[0],
        'speed_to_95': speeds[1],
        'speed_to_99': speeds[2],
        'auc': auc,
        'convergence_rate': rate,
        'final_acc': final_acc
    }

def train_with_granular_tracking(model, train_loader, X_val, y_val, X_test, y_test,
                                 epochs=200, lr=1e-2, weight_decay=5e-4, device='cpu',
                                 batch_size=128, track_batches=True):
    """Train with granular batch-level tracking for first 5 epochs"""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    crit = nn.CrossEntropyLoss()
    
    train_losses, val_accs = [], []
    checkpoint_accs = {}
    batch_level_tracking = []
    
    global_step = 0
    
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (bx, by) in enumerate(train_loader):
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            
            # Batch-level tracking for first 5 epochs
            if track_batches and ep < 5:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val).argmax(1)
                    val_acc = (val_pred == y_val).float().mean().item()
                batch_level_tracking.append({
                    'epoch': ep,
                    'batch': batch_idx,
                    'global_step': global_step,
                    'val_acc': val_acc
                })
                model.train()
            
            global_step += 1
        
        # Epoch-level validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = crit(val_out, y_val).item()
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(epoch_loss / max(1, len(train_loader)))
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Checkpoint accuracies
        if ep + 1 in [10, 20, 40, 80, 160, 200]:
            checkpoint_accs[ep + 1] = val_acc
    
    # Final test accuracy
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    # Convergence metrics
    convergence_metrics = compute_convergence_metrics(val_accs, epochs)
    
    return {
        'val_accs': val_accs,
        'train_losses': train_losses,
        'test_acc': test_acc,
        'checkpoint_accs': checkpoint_accs,
        'batch_level_tracking': batch_level_tracking,
        'convergence_metrics': convergence_metrics
    }

def aggregate_results(results_list):
    """Aggregate results across multiple seeds and splits"""
    test_accs = [r['test_acc'] for r in results_list]
    val_accs_array = np.array([r['val_accs'] for r in results_list])
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
    
    # Batch-level tracking
    batch_tracking_aggregated = {}
    if results_list[0]['batch_level_tracking']:
        max_steps = max([len(r['batch_level_tracking']) for r in results_list])
        for step in range(max_steps):
            step_accs = []
            for r in results_list:
                if step < len(r['batch_level_tracking']):
                    step_accs.append(r['batch_level_tracking'][step]['val_acc'])
            if step_accs:
                batch_tracking_aggregated[step] = {
                    'global_step': step,
                    'mean_val_acc': float(np.mean(step_accs)),
                    'std_val_acc': float(np.std(step_accs))
                }
    
    return {
        'test_acc': {'mean': float(np.mean(test_accs)), 'std': float(np.std(test_accs))},
        'val_accs': {'mean': np.mean(val_accs_array, axis=0), 'std': np.std(val_accs_array, axis=0)},
        'train_losses': {'mean': np.mean(train_losses_array, axis=0), 'std': np.std(train_losses_array, axis=0)},
        'checkpoint_accs': all_checkpoints,
        'convergence_metrics': conv_metrics,
        'batch_level_tracking': batch_tracking_aggregated
    }

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_direction_A(aggregated_results, dataset_name, save_path, epochs):
    """Plot Direction A: (b) vs (c) - Model sensitivity on same basis V"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = ['#ff7f0e', '#2ca02c']  # orange, green
    models = [
        ('Standard MLP on V (scaled)', 'standard_V_scaled'),
        ('RowNorm MLP on V (unscaled)', 'rownorm_V')
    ]
    
    for idx, (name, key) in enumerate(models):
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
    axes[0].set_title(f'Direction A: Model Sensitivity (Same Basis V) - {dataset_name}', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Loss', fontsize=12)
    axes[1].set_title(f'Training Loss - {dataset_name}', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_path}')
    plt.close()

def plot_direction_B(aggregated_results, dataset_name, save_path, epochs):
    """Plot Direction B: (a) vs (b) - Basis sensitivity with same preprocessing"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = ['#d62728', '#ff7f0e']  # red, orange
    models = [
        ('Standard MLP on X (scaled)', 'standard_X_scaled'),
        ('Standard MLP on V (scaled)', 'standard_V_scaled')
    ]
    
    for idx, (name, key) in enumerate(models):
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
    axes[0].set_title(f'Direction B: Basis Sensitivity (Same Preprocessing) - {dataset_name}', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Loss', fontsize=12)
    axes[1].set_title(f'Training Loss - {dataset_name}', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_path}')
    plt.close()

def plot_all_three(aggregated_results, dataset_name, save_path, epochs):
    """Plot all three experiments together"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # red, orange, green
    models = [
        ('(a) Standard MLP on X (scaled)', 'standard_X_scaled'),
        ('(b) Standard MLP on V (scaled)', 'standard_V_scaled'),
        ('(c) RowNorm MLP on V (unscaled)', 'rownorm_V')
    ]
    
    for idx, (name, key) in enumerate(models):
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
    axes[0].set_title(f'All Three Experiments - {dataset_name}', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, epochs])
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Loss', fontsize=12)
    axes[1].set_title(f'Training Loss - {dataset_name}', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, epochs])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_path}')
    plt.close()

def plot_early_convergence_all(aggregated_results, dataset_name, save_path):
    """Plot batch-level convergence for all three models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    models = [
        ('(a) Standard MLP on X', 'standard_X_scaled'),
        ('(b) Standard MLP on V', 'standard_V_scaled'),
        ('(c) RowNorm MLP on V', 'rownorm_V')
    ]
    
    for idx, (name, key) in enumerate(models):
        batch_data = aggregated_results[key]['batch_level_tracking']
        if batch_data:
            steps = [batch_data[s]['global_step'] for s in sorted(batch_data.keys())]
            means = [batch_data[s]['mean_val_acc'] for s in sorted(batch_data.keys())]
            stds = [batch_data[s]['std_val_acc'] for s in sorted(batch_data.keys())]
            
            ax.plot(steps, means, label=name, linewidth=2, color=colors[idx], alpha=0.9)
            ax.fill_between(steps, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Batch (Global Step)', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'Early Convergence: First 5 Epochs (Batch-Level) - {dataset_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_path}')
    plt.close()

# ============================================================================
# Table Printing Functions
# ============================================================================

def print_summary_table(aggregated_results, dataset_name):
    """Print comprehensive summary table"""
    print('\n' + '='*100)
    print(f'SUMMARY TABLE: {dataset_name.upper()}')
    print('='*100)
    print(f"{'Model':<40} {'Test Acc':>12} {'Ep20 ValAcc':>12} {'AUC':>10}")
    print('-'*100)
    
    models = [
        ('(a) Standard MLP on X (scaled)', 'standard_X_scaled'),
        ('(b) Standard MLP on V (scaled)', 'standard_V_scaled'),
        ('(c) RowNorm MLP on V (unscaled)', 'rownorm_V')
    ]
    
    for name, key in models:
        results = aggregated_results[key]
        test_mean = results['test_acc']['mean'] * 100
        test_std = results['test_acc']['std'] * 100
        
        ep20_acc = results['checkpoint_accs'].get(20, {})
        if ep20_acc:
            ep20_mean = ep20_acc['mean'] * 100
            ep20_std = ep20_acc['std'] * 100
            ep20_str = f"{ep20_mean:5.2f}±{ep20_std:4.2f}%"
        else:
            ep20_str = "N/A"
        
        auc_mean = results['convergence_metrics']['auc']['mean']
        auc_std = results['convergence_metrics']['auc']['std']
        
        print(f"{name:<40} {test_mean:5.2f}±{test_std:4.2f}%   {ep20_str:>12}   {auc_mean:.3f}±{auc_std:.3f}")
    
    print('='*100)

def print_direction_A_analysis(aggregated_results, dataset_name):
    """Print Direction A analysis: (b) vs (c)"""
    print('\n' + '='*100)
    print('DIRECTION A: MODEL SENSITIVITY (Same Basis V)')
    print('='*100)
    print('Question: Does RowNorm MLP help with restricted eigenvectors V?')
    print('Comparison: (b) Standard MLP on V vs (c) RowNorm MLP on V')
    print('-'*100)
    
    b_results = aggregated_results['standard_V_scaled']
    c_results = aggregated_results['rownorm_V']
    
    b_test = b_results['test_acc']['mean']
    c_test = c_results['test_acc']['mean']
    
    print(f"\n{'Metric':<30} {'(b) Standard V':>15} {'(c) RowNorm V':>15} {'Difference':>15}")
    print('-'*100)
    
    # Test accuracy
    b_test_str = f"{b_test*100:.2f}±{b_results['test_acc']['std']*100:.2f}%"
    c_test_str = f"{c_test*100:.2f}±{c_results['test_acc']['std']*100:.2f}%"
    diff = (c_test - b_test) / b_test * 100
    print(f"{'Test Accuracy':<30} {b_test_str:>15} {c_test_str:>15} {diff:+.1f}%")
    
    # Convergence speed
    b_speed = b_results['convergence_metrics']['speed_to_90']['mean']
    c_speed = c_results['convergence_metrics']['speed_to_90']['mean']
    speedup = b_speed / c_speed if c_speed > 0 else 1.0
    print(f"{'Speed to 90% (epochs)':<30} {b_speed:>15.1f} {c_speed:>15.1f} {speedup:>14.2f}x")
    
    print('\n' + '='*100)
    if c_test > b_test:
        print('✓ RESULT: RowNorm MLP shows advantage on restricted eigenvectors')
    else:
        print('✗ RESULT: RowNorm MLP does NOT show advantage on restricted eigenvectors')
    print('='*100)

def print_direction_B_analysis(aggregated_results, dataset_name):
    """Print Direction B analysis: (a) vs (b)"""
    print('\n' + '='*100)
    print('DIRECTION B: BASIS SENSITIVITY (Same Preprocessing)')
    print('='*100)
    print('Question: Does basis choice (X vs V) matter with identical preprocessing?')
    print('Comparison: (a) Standard MLP on X vs (b) Standard MLP on V')
    print('-'*100)
    
    a_results = aggregated_results['standard_X_scaled']
    b_results = aggregated_results['standard_V_scaled']
    
    a_test = a_results['test_acc']['mean']
    b_test = b_results['test_acc']['mean']
    
    print(f"\n{'Metric':<30} {'(a) Standard X':>15} {'(b) Standard V':>15} {'Difference':>15}")
    print('-'*100)
    
    # Test accuracy
    a_test_str = f"{a_test*100:.2f}±{a_results['test_acc']['std']*100:.2f}%"
    b_test_str = f"{b_test*100:.2f}±{b_results['test_acc']['std']*100:.2f}%"
    diff = (b_test - a_test) / a_test * 100
    print(f"{'Test Accuracy':<30} {a_test_str:>15} {b_test_str:>15} {diff:+.1f}%")
    
    # AUC comparison
    a_auc = a_results['convergence_metrics']['auc']['mean']
    b_auc = b_results['convergence_metrics']['auc']['mean']
    auc_diff = (b_auc - a_auc) / a_auc * 100
    print(f"{'AUC (normalized)':<30} {a_auc:>15.3f} {b_auc:>15.3f} {auc_diff:+.1f}%")
    
    print('\n' + '='*100)
    print('KEY INSIGHT: span(X) = span(V) — they contain IDENTICAL information!')
    if abs(diff) > 1.0:
        print(f'✓ RESULT: Basis choice DOES matter ({abs(diff):.1f}% difference despite same information)')
        print('   → MLPs are sensitive to how data is represented, not just what information is present')
    else:
        print(f'○ RESULT: Basis choice has minimal effect ({abs(diff):.1f}% difference)')
        print('   → For this dataset, MLPs are relatively basis-invariant')
    print('='*100)
def print_isolated_effects_table(aggregated_results, dataset_name):
    """
    Print table showing isolated basis and model effects
    Tests whether effects are additive or interactive
    """
    print('\n' + '='*100)
    print('ISOLATED EFFECTS ANALYSIS')
    print('='*100)
    print('Breaking down the performance differences into pure basis and pure model effects.')
    print('-'*100)
    
    a_test = aggregated_results['standard_X_scaled']['test_acc']['mean']
    b_test = aggregated_results['standard_V_scaled']['test_acc']['mean']
    c_test = aggregated_results['rownorm_V']['test_acc']['mean']
    
    # Pure basis effect (Direction B: same model, different basis)
    basis_effect = (b_test - a_test) / a_test * 100
    
    # Pure model effect (Direction A: same basis, different model)
    model_effect = (c_test - b_test) / b_test * 100
    
    # Combined effect (confounded)
    combined_effect = (c_test - a_test) / a_test * 100
    
    # Expected combined if effects are additive
    expected_combined = basis_effect + model_effect
    
    print(f"\n{'Effect Type':<35} | {'Comparison':<15} | {'Magnitude':<12} | {'Interpretation':<30}")
    print('-'*100)
    print(f"{'Pure Basis Effect':<35} | {'(a) → (b)':<15} | {basis_effect:+6.2f}%     | {'X → V (same preprocessing)':<30}")
    print(f"{'Pure Model Effect':<35} | {'(b) → (c)':<15} | {model_effect:+6.2f}%     | {'Standard → RowNorm (same basis)':<30}")
    print(f"{'Combined Effect (measured)':<35} | {'(a) → (c)':<15} | {combined_effect:+6.2f}%     | {'Both basis + model change':<30}")
    print(f"{'Expected if additive':<35} | {'basis + model':<15} | {expected_combined:+6.2f}%     | {'Sum of isolated effects':<30}")
    print('-'*100)
    
    # Check additivity
    interaction = combined_effect - expected_combined
    print(f"\n{'Interaction Term':<35} | {'measured - expected':<15} | {interaction:+6.2f}%")
    
    if abs(interaction) < 1.0:
        print("✓ Effects are approximately ADDITIVE (interaction < 1%)")
        print("  → Basis and model effects are independent")
        print("  → Can predict combined performance from individual effects")
    elif abs(interaction) < 3.0:
        print("◯ Weak interaction between basis and model (1-3%)")
        print("  → Some synergy/antagonism present but minor")
    else:
        print("⚠ Strong INTERACTION between basis and model (>3%)")
        if interaction > 0:
            print("  → Synergistic: Combined effect GREATER than sum of parts")
            print("  → RowNorm benefits MORE from restricted eigenvectors than expected")
        else:
            print("  → Antagonistic: Combined effect LESS than sum of parts")
            print("  → RowNorm benefits LESS from restricted eigenvectors than expected")
    
    print('='*100)
    
def interpret_results(aggregated_results, dataset_name):
    """Interpret results against expectations"""
    print('\n' + '='*100)
    print('INTERPRETATION: EXPECTED VS OBSERVED')
    print('='*100)
    
    a_test = aggregated_results['standard_X_scaled']['test_acc']['mean']
    b_test = aggregated_results['standard_V_scaled']['test_acc']['mean']
    c_test = aggregated_results['rownorm_V']['test_acc']['mean']
    
    basis_effect = (b_test - a_test) / a_test * 100
    model_effect = (c_test - b_test) / b_test * 100
    
    print("\nDirection A Expectations:")
    print("- Expected: Small RowNorm advantage (~1-3%) on restricted eigenvectors")
    print(f"- Observed: {model_effect:+.1f}%")
    
    if model_effect > 1.0:
        print("✓ CONFIRMED: RowNorm advantage persists with restricted eigenvectors")
        print("  → Investigation 1 findings transfer to X-restricted case")
    elif abs(model_effect) < 1.0:
        print("◯ INCONCLUSIVE: Minimal difference between models")
        print("  → Restricted eigenvectors may lack geometric properties RowNorm exploits")
    else:
        print("✗ UNEXPECTED: RowNorm underperforms on restricted eigenvectors")
        print(f"  → {abs(model_effect):.1f}% degradation suggests incompatibility")
    
    print("\nDirection B Expectations:")
    print("- Expected: Small basis sensitivity (~1-3%) despite span(X) = span(V)")
    print(f"- Observed: {basis_effect:+.1f}%")
    
    if abs(basis_effect) > 2.0:
        print("✓ CONFIRMED: Significant basis sensitivity detected")
        print("  → MLPs are sensitive to basis representation, not just information content")
    elif abs(basis_effect) < 1.0:
        print("✗ UNEXPECTED: Minimal basis sensitivity")
        print("  → Basis choice may not matter for this dataset")
    else:
        print("◯ MODERATE: Small but measurable basis sensitivity")
    
    # Check if effects are additive
    combined_effect = (c_test - a_test) / a_test * 100
    expected_combined = basis_effect + model_effect
    interaction = combined_effect - expected_combined
    
    print("\nEffect Additivity Check:")
    print(f"- Basis effect (a→b):     {basis_effect:+6.2f}%")
    print(f"- Model effect (b→c):     {model_effect:+6.2f}%")
    print(f"- Sum of effects:         {expected_combined:+6.2f}%")
    print(f"- Observed combined (a→c):{combined_effect:+6.2f}%")
    print(f"- Interaction term:       {interaction:+6.2f}%")
    
    if abs(interaction) < 1.0:
        print("\n✓ Effects are approximately ADDITIVE - minimal interaction")
        print("  → Basis and model effects are independent")
    else:
        print("\n⚠ Effects show INTERACTION - not purely additive")
        print("  → Basis choice and model architecture interact")

def print_key_takeaways(aggregated_results, dataset_name):
    """Print concise key takeaways"""
    print('\n' + '='*100)
    print('KEY TAKEAWAYS')
    print('='*100)

    a_test = aggregated_results['standard_X_scaled']['test_acc']['mean']
    b_test = aggregated_results['standard_V_scaled']['test_acc']['mean']
    c_test = aggregated_results['rownorm_V']['test_acc']['mean']

    basis_effect = abs((b_test - a_test) / a_test * 100)
    model_effect = (c_test - b_test) / b_test * 100

    print(f'\n1. Basis Sensitivity (Direction B):')
    if basis_effect > 2.0:
        print(f'   ✓ CONFIRMED: Basis choice matters ({basis_effect:.1f}% difference)')
        print(f'     Despite span(X) = span(V), MLPs are basis-sensitive')
    else:
        print(f'   ✗ Minimal basis sensitivity ({basis_effect:.1f}% difference)')
        print(f'     Basis choice does not strongly affect performance')

    print(f'\n2. RowNorm Effectiveness (Direction A):')
    if model_effect > 1.0:
        print(f'   ✓ RowNorm advantage persists: +{model_effect:.1f}%')
        print(f'     Restricted eigenvectors preserve geometric properties RowNorm needs')
    elif abs(model_effect) < 1.0:
        print(f'   ◯ Neutral: {model_effect:+.1f}% (minimal difference)')
    else:
        print(f'   ✗ RowNorm does not help: {model_effect:.1f}%')
        print(f'     Restricted eigenvectors may lack properties from Investigation 1')

    print(f'\n3. Research Contribution:')
    print(f'   - Experiment (b) successfully isolates pure basis effect')
    print(f'   - Can now separate basis sensitivity from model architecture effects')
    print(f'   - Results provide evidence for {dataset_name} dataset characteristics')
    
    print('='*100)
# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

if X_raw is None:
    print(f"ERROR: Dataset {DATASET_NAME} has no node features!")
    sys.exit(1)

d_raw = X_raw.shape[1]
print(f'Nodes: {num_nodes:,}')
print(f'Edges: {edge_index.shape[1]:,}')
print(f'Classes: {num_classes}')
print(f'Raw feature dimension: {d_raw}')

# ============================================================================
# 2. Compute Restricted Eigenvectors with QR Decomposition
# ============================================================================
print(f'\n[2/6] Computing X-restricted eigenvectors...')

try:
    adj, D, L = build_graph_matrices(edge_index, num_nodes)
except Exception as e:
    print(f'ERROR: Failed to build graph matrices: {e}')
    sys.exit(1)

# QR decomposition for rank deficiency
print('Performing QR decomposition of X...')
try:
    Q, R = np.linalg.qr(X_raw.astype(np.float64))
    rank_X = np.sum(np.abs(np.diag(R)) > 1e-10)
    d_effective = rank_X
    
    print(f'Original dimension: {d_raw}')
    print(f'Effective rank: {rank_X}/{d_raw} ({rank_X/d_raw*100:.1f}%)')
    
    if rank_X < d_raw:
        deficiency_pct = (1 - rank_X/d_raw) * 100
        print(f'⚠ Rank deficiency detected: {deficiency_pct:.1f}% rank loss')
        print(f'  Using {d_effective} dimensions (dropping {d_raw - d_effective} redundant)')
        Q = Q[:, :d_effective]
        
        # Warn if severe rank deficiency
        if rank_X < 0.5 * d_raw:
            print(f'⚠⚠ SEVERE rank deficiency: Only {rank_X/d_raw*100:.1f}% of dimensions retained!')
            print(f'     Results may be unreliable. Consider using Investigation 1 instead.')
    else:
        print(f'✓ Full rank: No rank deficiency')
        
except np.linalg.LinAlgError as e:
    print(f'ERROR: QR decomposition failed: {e}')
    print('This dataset may have numerical issues. Try Investigation 1 instead.')
    sys.exit(1)

# Compute restricted eigenproblem
print(f'\nComputing restricted eigenvectors in R^{d_effective}...')
try:
    LQ = (L @ Q).astype(np.float64)
    DQ = (D @ Q).astype(np.float64)
    
    Lr = Q.T @ LQ
    Dr = Q.T @ DQ
    
    # Symmetrize
    Lr = 0.5 * (Lr + Lr.T)
    Dr = 0.5 * (Dr + Dr.T)
    
    # Check condition number before regularization
    cond_Dr = np.linalg.cond(Dr)
    print(f'Condition number of D_r: {cond_Dr:.2e}')
    
    if cond_Dr > 1e10:
        print(f'⚠ WARNING: D_r is ill-conditioned (cond={cond_Dr:.2e})')
        print(f'  Applying stronger regularization...')
        epsilon = 1e-6 * np.trace(Dr) / d_effective
    else:
        epsilon = 1e-8 * np.trace(Dr) / d_effective
    
    print(f'Regularization: ε = {epsilon:.2e}')
    Dr = Dr + epsilon * np.eye(d_effective)
    
    # Solve generalized eigenproblem
    print('Solving generalized eigenproblem (L_r, D_r)...')
    w, V = la.eigh(Lr, Dr)
    idx = np.argsort(w)
    w = w[idx]
    V = V[:, idx]
    
    print(f'Eigenvalue range: [{w[0]:.6f}, {w[-1]:.6f}]')
    
    # Check for negative eigenvalues
    num_negative = np.sum(w < -1e-10)
    if num_negative > 0:
        print(f'⚠ WARNING: {num_negative} negative eigenvalues detected!')
        print(f'  This may indicate numerical issues.')
    
except np.linalg.LinAlgError as e:
    print(f'ERROR: Eigensolver failed: {e}')
    print('Try increasing regularization or using Investigation 1.')
    sys.exit(1)

# Map back to node space
try:
    U = (Q @ V).astype(np.float32)
    print(f'✓ Mapped back to node space: U.shape = {U.shape}')
except Exception as e:
    print(f'ERROR: Failed to map eigenvectors back to node space: {e}')
    sys.exit(1)

# Verify D-orthonormality
print('\nVerifying D-orthonormality...')
try:
    DU = (D @ U).astype(np.float64)
    G = U.astype(np.float64).T @ DU
    dev = np.abs(G - np.eye(U.shape[1])).max()
    dev_frobenius = np.linalg.norm(G - np.eye(U.shape[1]), 'fro')
    
    print(f'D-orthonormality check:')
    print(f'  max |U^T D U - I|     = {dev:.2e}')
    print(f'  ||U^T D U - I||_F     = {dev_frobenius:.2e}')
    
    # Quality assessment
    if dev < 1e-6:
        quality = 'EXCELLENT'
    elif dev < 1e-4:
        quality = 'GOOD'
    elif dev < 1e-2:
        quality = 'ACCEPTABLE'
    else:
        quality = 'POOR'
    
    print(f'  Quality: {quality}')
    
    if dev > 1e-4:
        print(f'⚠ WARNING: D-orthonormality error exceeds 1e-4!')
        print(f'  This may affect RowNorm MLP performance.')
        print(f'  Consider increasing regularization or using Investigation 1.')
    else:
        print(f'✓ D-orthonormality verified (error < 1e-4)')
        
except Exception as e:
    print(f'ERROR: D-orthonormality verification failed: {e}')
    sys.exit(1)

print(f'\n✓ Restricted eigenvectors U computed successfully')
print(f'✓ span(U) = span(X) — same subspace, different basis')
print(f'✓ Ready for experiments (a), (b), (c)')

# Save D-orthonormality deviation for later reference
D_ORTHONORMALITY_DEV = dev
D_ORTHONORMALITY_QUALITY = quality

# ============================================================================
# 3. Prepare Data Splits
# ============================================================================
print(f'\n[3/6] Preparing data splits...')

num_split_iterations = NUM_RANDOM_SPLITS if USE_RANDOM_SPLITS else 1

# ============================================================================
# 4. Prepare Labels
# ============================================================================
print(f'\n[4/6] Preparing labels...')
print('✓ Labels prepared')

# ============================================================================
# 5. Train All Models with Multiple Seeds and Splits
# ============================================================================
print(f'\n[5/6] Training 3 models...')
print(f'Total runs: {3 * NUM_SEEDS * num_split_iterations}')

model_configs = [
    ('(a) Standard MLP on X (scaled)', 'standard_X_scaled', 'X', True),
    ('(b) Standard MLP on V (scaled)', 'standard_V_scaled', 'V', True),
    ('(c) RowNorm MLP on V (unscaled)', 'rownorm_V', 'V', False),
]

all_results = {key: [] for _, key, _, _ in model_configs}

# Determine batch size
if len(train_idx_fixed) > 256:
    batch_size = 128
    print(f'Batch size: {batch_size} (large dataset)')
else:
    batch_size = len(train_idx_fixed)
    print(f'Batch size: {batch_size} (full batch for small dataset)')

# Outer loop: splits
for split_idx in range(num_split_iterations):
    
    if USE_RANDOM_SPLITS:
        print(f'\n{"="*70}')
        print(f'RANDOM SPLIT {split_idx+1}/{NUM_RANDOM_SPLITS}')
        print(f'{"="*70}')
        train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
    else:
        print(f'\n{"="*70}')
        print('USING FIXED BENCHMARK SPLITS')
        print(f'{"="*70}')
        train_idx, val_idx, test_idx = train_idx_fixed, val_idx_fixed, test_idx_fixed
    
    print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')
    
    # Prepare labels
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)
    y_test = torch.from_numpy(labels[test_idx]).long().to(device)
    
    # Inner loop: random seeds
    for seed_idx in range(NUM_SEEDS):
        seed = split_idx * NUM_SEEDS + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f'\n--- Split {split_idx+1}/{num_split_iterations}, Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed}) ---')
        
        # Train each model
        for model_name, model_key, feature_type, use_scaling in model_configs:
            print(f'\nTraining: {model_name}...')
            
            # Prepare features
            if feature_type == 'X' and use_scaling:
                # (a) X with StandardScaler
                scaler = StandardScaler().fit(X_raw[train_idx])
                X_train = torch.from_numpy(scaler.transform(X_raw[train_idx])).float().to(device)
                X_val = torch.from_numpy(scaler.transform(X_raw[val_idx])).float().to(device)
                X_test = torch.from_numpy(scaler.transform(X_raw[test_idx])).float().to(device)
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
                model = StandardMLP(d_raw, HIDDEN_DIM, num_classes)
                
            elif feature_type == 'V' and use_scaling:
                # (b) V with StandardScaler [NEW EXPERIMENT!]
                print('  → NEW EXPERIMENT (b): StandardScaler on restricted eigenvectors V')
                scaler = StandardScaler().fit(U[train_idx])
                V_train = torch.from_numpy(scaler.transform(U[train_idx])).float().to(device)
                V_val = torch.from_numpy(scaler.transform(U[val_idx])).float().to(device)
                V_test = torch.from_numpy(scaler.transform(U[test_idx])).float().to(device)
                
                # Check scaling statistics
                print(f'  V scaling stats: mean={scaler.mean_[:5]} ... , std={scaler.scale_[:5]} ...')
                
                train_loader = DataLoader(TensorDataset(V_train, y_train), batch_size=batch_size, shuffle=True)
                model = StandardMLP(d_effective, HIDDEN_DIM, num_classes)
                
            elif feature_type == 'V' and not use_scaling:
                # (c) V with RowNorm MLP
                V_train = torch.from_numpy(U[train_idx]).float().to(device)
                V_val = torch.from_numpy(U[val_idx]).float().to(device)
                V_test = torch.from_numpy(U[test_idx]).float().to(device)
                train_loader = DataLoader(TensorDataset(V_train, y_train), batch_size=batch_size, shuffle=True)
                model = RowNormMLP(d_effective, HIDDEN_DIM, num_classes)
            
            # Train
            if feature_type == 'X' and use_scaling:
                results = train_with_granular_tracking(
                    model, train_loader, X_val, y_val, X_test, y_test,
                    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    device=device, batch_size=batch_size
                )
            elif feature_type == 'V' and use_scaling:
                results = train_with_granular_tracking(
                    model, train_loader, V_val, y_val, V_test, y_test,
                    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    device=device, batch_size=batch_size
                )
            else:  # V without scaling
                results = train_with_granular_tracking(
                    model, train_loader, V_val, y_val, V_test, y_test,
                    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    device=device, batch_size=batch_size
                )
            
            all_results[model_key].append(results)
            
            # Print detailed results for this run
            test_acc = results["test_acc"]
            final_val = results["val_accs"][-1]
            ep20_val = results["checkpoint_accs"].get(20, 0.0)
            
            print(f'  ✓ Test acc: {test_acc:.4f} | Final val: {final_val:.4f} | Ep20 val: {ep20_val:.4f}')
            
            # Special attention to experiment (b)
            if model_key == 'standard_V_scaled':
                print(f'  🔍 EXPERIMENT (b) completed - this is the NEW experiment!')
        
        # After training all three models for this seed, print comparison
        print(f'\n--- Seed {seed_idx+1} Summary ---')
        a_acc = all_results['standard_X_scaled'][-1]['test_acc']
        b_acc = all_results['standard_V_scaled'][-1]['test_acc']
        c_acc = all_results['rownorm_V'][-1]['test_acc']
        
        print(f'  (a) Standard X: {a_acc:.4f}')
        print(f'  (b) Standard V: {b_acc:.4f}  [Diff from (a): {(b_acc-a_acc)*100:+.2f}%]')
        print(f'  (c) RowNorm V:  {c_acc:.4f}  [Diff from (b): {(c_acc-b_acc)*100:+.2f}%]')
        
        # Quick interpretation
        if abs(b_acc - a_acc) > 0.01:
            print(f'  → Basis effect (a vs b): {abs((b_acc-a_acc)/a_acc*100):.1f}% difference')
        else:
            print(f'  → Basis effect minimal: {abs((b_acc-a_acc)/a_acc*100):.1f}% difference')
            
        if c_acc > b_acc:
            print(f'  → RowNorm helps: +{(c_acc-b_acc)/b_acc*100:.1f}%')
        else:
            print(f'  → RowNorm hurts: {(c_acc-b_acc)/b_acc*100:.1f}%')

# ============================================================================
# 6. Aggregate Results and Generate Outputs
# ============================================================================
print(f'\n[6/6] Aggregating results and generating outputs...')

# Print running statistics before aggregation
print('\n' + '='*70)
print('RUNNING STATISTICS (Before Aggregation)')
print('='*70)
for model_name, model_key, _, _ in model_configs:
    test_accs = [r['test_acc'] for r in all_results[model_key]]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    min_acc = np.min(test_accs)
    max_acc = np.max(test_accs)
    
    print(f'\n{model_name}:')
    print(f'  Mean: {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'  Range: [{min_acc:.4f}, {max_acc:.4f}]')
    print(f'  N runs: {len(test_accs)}')
    
    if model_key == 'standard_V_scaled':
        print(f'  🔍 This is experiment (b) - the NEW experiment for Direction B!')

print('='*70)

aggregated_results = {}
for model_name, model_key, _, _ in model_configs:
    print(f'Aggregating: {model_name}...')
    aggregated_results[model_key] = aggregate_results(all_results[model_key])

# Generate plots
print('\nGenerating plots...')

plot_direction_A(aggregated_results, DATASET_NAME, 
                f'{output_base}/plots/direction_A_model_sensitivity.png', EPOCHS)

plot_direction_B(aggregated_results, DATASET_NAME,
                f'{output_base}/plots/direction_B_basis_sensitivity.png', EPOCHS)

plot_all_three(aggregated_results, DATASET_NAME,
              f'{output_base}/plots/all_three_experiments.png', EPOCHS)

plot_early_convergence_all(aggregated_results, DATASET_NAME,
                           f'{output_base}/plots/early_convergence_all.png')

# Print analyses
print_summary_table(aggregated_results, DATASET_NAME)
print_direction_A_analysis(aggregated_results, DATASET_NAME)
print_direction_B_analysis(aggregated_results, DATASET_NAME)
print_isolated_effects_table(aggregated_results, DATASET_NAME)


interpret_results(aggregated_results, DATASET_NAME)
print_key_takeaways(aggregated_results, DATASET_NAME)

# Additional sanity checks
print('\n' + '='*100)
print('SANITY CHECKS & WARNINGS')
print('='*100)

a_test = aggregated_results['standard_X_scaled']['test_acc']['mean']
b_test = aggregated_results['standard_V_scaled']['test_acc']['mean']
c_test = aggregated_results['rownorm_V']['test_acc']['mean']

# Check 1: All models above random baseline
random_baseline = 1.0 / num_classes
print(f'\n1. Random Baseline Check (1/{num_classes} = {random_baseline:.4f}):')
if a_test < random_baseline * 1.5:
    print(f'   ⚠ WARNING: (a) Standard X only {a_test/random_baseline:.2f}x better than random!')
if b_test < random_baseline * 1.5:
    print(f'   ⚠ WARNING: (b) Standard V only {b_test/random_baseline:.2f}x better than random!')
if c_test < random_baseline * 1.5:
    print(f'   ⚠ WARNING: (c) RowNorm V only {c_test/random_baseline:.2f}x better than random!')
if a_test >= random_baseline * 1.5 and b_test >= random_baseline * 1.5 and c_test >= random_baseline * 1.5:
    print(f'   ✓ All models significantly above random baseline')

# Check 2: Experiment (b) behaving reasonably
print(f'\n2. Experiment (b) Validity Check:')
b_std = aggregated_results['standard_V_scaled']['test_acc']['std']
if b_std > 0.05:
    print(f'   ⚠ WARNING: (b) has high variance (std={b_std:.4f})')
    print(f'      Training may be unstable. Consider more runs or different hyperparameters.')
else:
    print(f'   ✓ (b) has acceptable variance (std={b_std:.4f})')

diff_ab = abs(b_test - a_test) / a_test
if diff_ab > 0.20:
    print(f'   ⚠ LARGE difference between (a) and (b): {diff_ab*100:.1f}%')
    print(f'      Despite span(X)=span(V), this is a strong basis sensitivity signal.')
elif diff_ab < 0.01:
    print(f'   ✓ (a) and (b) very similar ({diff_ab*100:.1f}% diff) - minimal basis sensitivity')
else:
    print(f'   ✓ (a) and (b) show {diff_ab*100:.1f}% difference - moderate basis sensitivity')

# Check 3: Direction A (RowNorm on V)
print(f'\n3. Direction A Validity Check (RowNorm on restricted eigenvectors):')
diff_bc = (c_test - b_test) / b_test
if diff_bc < -0.20:
    print(f'   ⚠ RowNorm MUCH worse on V than Standard ({diff_bc*100:.1f}%)')
    print(f'      Restricted eigenvectors may not have geometric properties RowNorm needs.')
elif diff_bc > 0.05:
    print(f'   ✓ RowNorm shows advantage on V (+{diff_bc*100:.1f}%)')
    print(f'      Investigation 1 findings transfer to restricted eigenvectors!')
else:
    print(f'   ○ RowNorm similar to Standard on V ({diff_bc*100:.1f}%)')
    print(f'      No strong model preference for restricted eigenvectors.')

# Check 4: Variance comparison
print(f'\n4. Variance Analysis:')
a_std = aggregated_results['standard_X_scaled']['test_acc']['std']
c_std = aggregated_results['rownorm_V']['test_acc']['std']
if c_std > 2 * a_std:
    print(f'   ⚠ RowNorm variance ({c_std:.4f}) >> Standard variance ({a_std:.4f})')
    print(f'      RowNorm may be less stable on this dataset.')
else:
    print(f'   ✓ Variances comparable: Standard={a_std:.4f}, RowNorm={c_std:.4f}')

# Check 5: D-orthonormality quality impact
print(f'\n5. Feature Quality Assessment:')
print(f'   D-orthonormality error: {D_ORTHONORMALITY_DEV:.2e} ({D_ORTHONORMALITY_QUALITY})')
if D_ORTHONORMALITY_DEV > 1e-4:
    print(f'   ⚠ Poor D-orthonormality may explain RowNorm underperformance')
    print(f'      RowNorm MLP requires well-conditioned D-orthonormal features')
else:
    print(f'   ✓ D-orthonormality is good - RowNorm results are reliable')

print('='*100)

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
    'models': {}
}

for model_name, model_key, _, _ in model_configs:
    metrics['models'][model_key] = {
        'name': model_name,
        'test_accuracy': aggregated_results[model_key]['test_acc'],
        'convergence_metrics': aggregated_results[model_key]['convergence_metrics']
    }

metrics_path = f'{output_base}/metrics/results_aggregated.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\n✓ Saved aggregated metrics: {metrics_path}')

print('\n' + '='*70)
print('✓ INVESTIGATION 2 DIRECTIONS A & B COMPLETE!')
print('='*70)
print(f'Results saved to: {output_base}/')
print('\nNext steps:')
print('1. Review Direction A: Does RowNorm help with restricted eigenvectors?')
print('2. Review Direction B: Does basis choice matter?')
print('='*70)
