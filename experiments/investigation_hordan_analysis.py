"""
Investigation: Hordan et al. Framework Analysis
===============================================

Testing theoretical predictions from "Spectral Graph Neural Networks are 
Incomplete on Graphs with a Simple Spectrum" (Hordan et al., 2025)

Core Insight from Hordan:
- Sign-invariant operations (like squaring, or analogously RowNorm) can 
  fundamentally lose discriminative information
- Theorem 3: EPNN is complete when eigenvectors have < n zeros
- Solution: Preserve magnitude/sign information (equivariant features)

Connection to Our Research:
- RowNorm discards magnitude → analogous to Hordan's sign-invariant squaring
- If magnitude carries class information, RowNorm causes information loss
- This explains "the gap" that RowNorm cannot fully close

Experiments:
1. Sparsity Analysis - Tests Theorem 3 condition (completeness when few zeros)
2. Magnitude-Class Correlation - Tests if RowNorm discards class-relevant info
3. Reference Node Analysis - Tests if "anchor nodes" (no zeros) exist
4. Equivariant Fix Test - Tests if preserving magnitude closes the gap

Matrices Analyzed:
- features: Raw node features
- X_diffused: SGC-diffused features (S^k X)  
- U: Restricted eigenvectors from Rayleigh-Ritz

Usage:
    python investigation_hordan_analysis.py
    python investigation_hordan_analysis.py --datasets cora citeseer
    python investigation_hordan_analysis.py --k_diffusion 2 10
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import from existing utils.py
from utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
    StandardMLP,
    RowNormMLP,
    LogMagnitudeMLP,  # This is our "MagnitudePreservingMLP" - Hordan's fix!
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Hordan Framework Analysis')
parser.add_argument('--datasets', nargs='+', type=str, 
                   default=['cora', 'citeseer', 'pubmed', 'wikics', 
                           'amazon-computers', 'amazon-photo',
                           'coauthor-cs', 'coauthor-physics', 'ogbn-arxiv'],
                   help='Datasets to analyze')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 10],
                   help='Diffusion steps to test')
parser.add_argument('--splits', type=str, choices=['random', 'fixed'], default='random',
                   help='Use random (60/20/20) or fixed splits')
parser.add_argument('--num_splits', type=int, default=5,
                   help='Number of random splits (only used if --splits random)')
parser.add_argument('--num_seeds', type=int, default=3,
                   help='Number of seeds per split for equivariant fix test')
parser.add_argument('--output_dir', type=str, default='results/hordan_analysis',
                   help='Output directory')
args = parser.parse_args()

DATASETS = args.datasets
K_DIFFUSION_VALUES = args.k_diffusion
SPLIT_TYPE = args.splits
NUM_SPLITS = args.num_splits if args.splits == 'random' else 1
NUM_SEEDS = args.num_seeds
OUTPUT_DIR = args.output_dir

# Near-zero thresholds to test
NEAR_ZERO_THRESHOLDS = {
    'abs_1e-10': ('absolute', 1e-10),
    'abs_1e-6': ('absolute', 1e-6),
    'abs_1e-3': ('absolute', 1e-3),
    'rel_1e-6': ('relative_max', 1e-6),
    'rel_1e-3': ('relative_max', 1e-3),
}

# Training parameters for equivariant fix test
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

print('='*80)
print('HORDAN FRAMEWORK ANALYSIS')
print('='*80)
print(f'Datasets: {DATASETS}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'Split type: {SPLIT_TYPE} ({NUM_SPLITS} splits)')
print(f'Seeds per split: {NUM_SEEDS}')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# Experiment 1: Sparsity Analysis
# ============================================================================

def count_zeros(X, threshold_type, threshold_value):
    """Count zeros/near-zeros in matrix"""
    if threshold_type == 'absolute':
        near_zero_mask = np.abs(X) < threshold_value
    elif threshold_type == 'relative_max':
        max_val = np.max(np.abs(X))
        near_zero_mask = np.abs(X) < threshold_value * max_val
    else:
        raise ValueError(f'Unknown threshold type: {threshold_type}')
    
    return near_zero_mask

def analyze_sparsity(X, name, thresholds):
    """
    Analyze sparsity patterns in matrix X
    
    Tests Hordan's Theorem 3: EPNN complete when < n zeros
    """
    n, d = X.shape
    total_entries = n * d
    
    results = {
        'name': name,
        'n_nodes': n,
        'd_features': d,
        'total_entries': total_entries,
        'thresholds': {}
    }
    
    for thresh_name, (thresh_type, thresh_val) in thresholds.items():
        near_zero_mask = count_zeros(X, thresh_type, thresh_val)
        
        # Global statistics
        num_zeros = np.sum(near_zero_mask)
        sparsity_ratio = num_zeros / total_entries
        
        # Row-wise statistics (critical for Theorem 3)
        zeros_per_row = np.sum(near_zero_mask, axis=1)
        rows_with_no_zeros = np.sum(zeros_per_row == 0)
        rows_with_all_zeros = np.sum(zeros_per_row == d)
        max_zeros_in_row = np.max(zeros_per_row)
        min_zeros_in_row = np.min(zeros_per_row)
        mean_zeros_per_row = np.mean(zeros_per_row)
        
        # Column-wise statistics
        zeros_per_col = np.sum(near_zero_mask, axis=0)
        cols_with_no_zeros = np.sum(zeros_per_col == 0)
        cols_with_all_zeros = np.sum(zeros_per_col == n)
        
        # Theorem 3 condition: fewer than n zeros total
        theorem3_satisfied = (num_zeros < n)
        
        # Alternative condition: at least one row with no zeros
        has_reference_row = (rows_with_no_zeros > 0)
        
        results['thresholds'][thresh_name] = {
            'threshold_type': thresh_type,
            'threshold_value': float(thresh_val),
            'num_zeros': int(num_zeros),
            'sparsity_ratio': float(sparsity_ratio),
            'zeros_per_row_mean': float(mean_zeros_per_row),
            'zeros_per_row_max': int(max_zeros_in_row),
            'zeros_per_row_min': int(min_zeros_in_row),
            'rows_with_no_zeros': int(rows_with_no_zeros),
            'rows_with_no_zeros_pct': float(rows_with_no_zeros / n * 100),
            'rows_with_all_zeros': int(rows_with_all_zeros),
            'cols_with_no_zeros': int(cols_with_no_zeros),
            'cols_with_all_zeros': int(cols_with_all_zeros),
            'theorem3_satisfied': theorem3_satisfied,
            'has_reference_row': has_reference_row,
        }
    
    return results

# ============================================================================
# Experiment 2: Magnitude-Class Correlation
# ============================================================================

def analyze_magnitude_class_correlation(X, labels, name):
    """
    Analyze if magnitude carries class information
    
    If magnitude is predictive of class, RowNorm discards useful info
    """
    n = X.shape[0]
    num_classes = len(np.unique(labels))
    
    # Compute row magnitudes
    magnitudes = np.linalg.norm(X, axis=1)
    
    # Handle zero magnitudes
    magnitudes = np.clip(magnitudes, 1e-10, None)
    log_magnitudes = np.log(magnitudes)
    
    results = {
        'name': name,
        'n_nodes': n,
        'num_classes': num_classes,
        'magnitude_stats': {
            'mean': float(np.mean(magnitudes)),
            'std': float(np.std(magnitudes)),
            'min': float(np.min(magnitudes)),
            'max': float(np.max(magnitudes)),
            'log_mean': float(np.mean(log_magnitudes)),
            'log_std': float(np.std(log_magnitudes)),
        },
        'per_class_magnitude': {},
        'separability_metrics': {}
    }
    
    # Per-class magnitude statistics
    class_magnitudes = []
    for c in range(num_classes):
        mask = labels == c
        class_mag = magnitudes[mask]
        class_log_mag = log_magnitudes[mask]
        
        results['per_class_magnitude'][f'class_{c}'] = {
            'count': int(np.sum(mask)),
            'mag_mean': float(np.mean(class_mag)),
            'mag_std': float(np.std(class_mag)),
            'log_mag_mean': float(np.mean(class_log_mag)),
            'log_mag_std': float(np.std(class_log_mag)),
        }
        class_magnitudes.append(class_mag)
    
    # Compute separability metrics
    
    # 1. Discretized Mutual Information
    n_bins = min(50, n // 100)
    n_bins = max(10, n_bins)
    mag_discretized = np.digitize(magnitudes, np.linspace(magnitudes.min(), magnitudes.max(), n_bins))
    
    try:
        mi = mutual_info_score(labels, mag_discretized)
        label_counts = np.bincount(labels)
        label_probs = label_counts / len(labels)
        label_entropy = -np.sum(label_probs * np.log(label_probs + 1e-10))
        nmi = mi / (label_entropy + 1e-10)
    except:
        mi = 0.0
        nmi = 0.0
    
    results['separability_metrics']['mutual_info'] = float(mi)
    results['separability_metrics']['normalized_mutual_info'] = float(nmi)
    
    # 2. Between-class variance / Within-class variance (Fisher criterion)
    overall_mean = np.mean(magnitudes)
    
    between_class_var = 0.0
    within_class_var = 0.0
    
    for c in range(num_classes):
        mask = labels == c
        class_mag = magnitudes[mask]
        n_c = len(class_mag)
        
        if n_c > 0:
            class_mean = np.mean(class_mag)
            between_class_var += n_c * (class_mean - overall_mean) ** 2
            within_class_var += np.sum((class_mag - class_mean) ** 2)
    
    between_class_var /= n
    within_class_var /= n
    
    fisher_ratio = between_class_var / (within_class_var + 1e-10)
    
    results['separability_metrics']['between_class_var'] = float(between_class_var)
    results['separability_metrics']['within_class_var'] = float(within_class_var)
    results['separability_metrics']['fisher_ratio'] = float(fisher_ratio)
    
    # 3. Simple magnitude-only classification accuracy (1-NN on magnitude)
    sorted_idx = np.argsort(magnitudes)
    sorted_labels = labels[sorted_idx]
    
    correct = 0
    for i in range(1, n-1):
        pred_label = sorted_labels[i-1] if abs(magnitudes[sorted_idx[i]] - magnitudes[sorted_idx[i-1]]) < \
                                            abs(magnitudes[sorted_idx[i]] - magnitudes[sorted_idx[i+1]]) \
                     else sorted_labels[i+1]
        if pred_label == sorted_labels[i]:
            correct += 1
    
    magnitude_1nn_acc = correct / (n - 2)
    results['separability_metrics']['magnitude_1nn_accuracy'] = float(magnitude_1nn_acc)
    
    # Random baseline
    random_baseline = 1.0 / num_classes
    results['separability_metrics']['random_baseline'] = float(random_baseline)
    
    # Is magnitude predictive?
    results['separability_metrics']['magnitude_is_predictive'] = (magnitude_1nn_acc > random_baseline * 1.5)
    
    return results, magnitudes

# ============================================================================
# Experiment 3: Reference Node Analysis
# ============================================================================

def analyze_reference_nodes(X, labels, name, thresholds):
    """
    Analyze "reference nodes" - nodes that have no zeros (Theorem 3)
    
    These nodes can serve as anchors for reconstruction
    """
    n, d = X.shape
    num_classes = len(np.unique(labels))
    
    results = {
        'name': name,
        'n_nodes': n,
        'd_features': d,
        'num_classes': num_classes,
        'thresholds': {}
    }
    
    for thresh_name, (thresh_type, thresh_val) in thresholds.items():
        near_zero_mask = count_zeros(X, thresh_type, thresh_val)
        zeros_per_row = np.sum(near_zero_mask, axis=1)
        
        # Reference nodes: rows with no zeros
        reference_mask = (zeros_per_row == 0)
        reference_indices = np.where(reference_mask)[0]
        num_reference = len(reference_indices)
        
        # Low-sparsity nodes: rows with < 10% zeros
        low_sparsity_mask = (zeros_per_row < d * 0.1)
        low_sparsity_indices = np.where(low_sparsity_mask)[0]
        num_low_sparsity = len(low_sparsity_indices)
        
        # Class distribution of reference nodes
        if num_reference > 0:
            ref_labels = labels[reference_indices]
            ref_class_dist = {}
            for c in range(num_classes):
                ref_class_dist[f'class_{c}'] = int(np.sum(ref_labels == c))
            
            ref_counts = np.array([ref_class_dist[f'class_{c}'] for c in range(num_classes)])
            ref_entropy = -np.sum((ref_counts/num_reference + 1e-10) * np.log(ref_counts/num_reference + 1e-10))
            max_entropy = np.log(num_classes)
            ref_balance = ref_entropy / max_entropy
        else:
            ref_class_dist = {f'class_{c}': 0 for c in range(num_classes)}
            ref_balance = 0.0
        
        results['thresholds'][thresh_name] = {
            'num_reference_nodes': num_reference,
            'pct_reference_nodes': float(num_reference / n * 100),
            'num_low_sparsity_nodes': num_low_sparsity,
            'pct_low_sparsity_nodes': float(num_low_sparsity / n * 100),
            'reference_class_distribution': ref_class_dist,
            'reference_class_balance': float(ref_balance),
            'has_reference_nodes': num_reference > 0,
            'reference_nodes_sufficient': num_reference >= num_classes,
        }
    
    return results

# ============================================================================
# Experiment 4: Equivariant Fix Test
# ============================================================================

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs, lr, weight_decay, device):
    """Train model and return best validation accuracy and corresponding test accuracy"""
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_pred = model(X_test_t).argmax(dim=1)
                    best_test_acc = (test_pred == y_test_t).float().mean().item()
    
    return best_val_acc, best_test_acc

def run_equivariant_fix_test(X_diffused, U, labels, train_idx, val_idx, test_idx,
                             num_classes, num_seeds, device):
    """
    Test if preserving magnitude (Hordan's solution) closes the gap
    
    Compares:
    1. SGC + StandardMLP (baseline, on diffused features)
    2. Restricted + StandardMLP (Part A - shows basis sensitivity)
    3. Restricted + RowNormMLP (Part B - RowNorm helps)
    4. Restricted + LogMagnitudeMLP (Hordan fix - should close gap)
    
    Note: LogMagnitudeMLP from utils.py is the "MagnitudePreservingMLP" - 
          it concatenates log-magnitude to RowNorm features, exactly as
          Hordan's equiEPNN preserves equivariant features.
    """
    results = {
        'sgc_standard': [],
        'restricted_standard': [],
        'restricted_rownorm': [],
        'restricted_magnitude_preserving': [],
    }
    
    d_diffused = X_diffused.shape[1]
    d_restricted = U.shape[1]
    
    # Prepare data
    scaler_diffused = StandardScaler()
    X_diff_train = scaler_diffused.fit_transform(X_diffused[train_idx])
    X_diff_val = scaler_diffused.transform(X_diffused[val_idx])
    X_diff_test = scaler_diffused.transform(X_diffused[test_idx])
    
    scaler_restricted = StandardScaler()
    U_train = scaler_restricted.fit_transform(U[train_idx])
    U_val = scaler_restricted.transform(U[val_idx])
    U_test = scaler_restricted.transform(U[test_idx])
    
    # For RowNorm and LogMagnitude, use raw (unscaled) restricted eigenvectors
    U_raw_train = U[train_idx]
    U_raw_val = U[val_idx]
    U_raw_test = U[test_idx]
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. SGC + StandardMLP
        model1 = StandardMLP(d_diffused, HIDDEN_DIM, num_classes)
        val_acc, test_acc = train_and_evaluate(
            model1, X_diff_train, y_train, X_diff_val, y_val, X_diff_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        results['sgc_standard'].append({'val_acc': val_acc, 'test_acc': test_acc})
        
        # 2. Restricted + StandardMLP
        model2 = StandardMLP(d_restricted, HIDDEN_DIM, num_classes)
        val_acc, test_acc = train_and_evaluate(
            model2, U_train, y_train, U_val, y_val, U_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        results['restricted_standard'].append({'val_acc': val_acc, 'test_acc': test_acc})
        
        # 3. Restricted + RowNormMLP
        model3 = RowNormMLP(d_restricted, HIDDEN_DIM, num_classes)
        val_acc, test_acc = train_and_evaluate(
            model3, U_raw_train, y_train, U_raw_val, y_val, U_raw_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        results['restricted_rownorm'].append({'val_acc': val_acc, 'test_acc': test_acc})
        
        # 4. Restricted + LogMagnitudeMLP (Hordan fix)
        model4 = LogMagnitudeMLP(d_restricted, HIDDEN_DIM, num_classes)
        val_acc, test_acc = train_and_evaluate(
            model4, U_raw_train, y_train, U_raw_val, y_val, U_raw_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        results['restricted_magnitude_preserving'].append({'val_acc': val_acc, 'test_acc': test_acc})
    
    return results


def run_equivariant_fix_test_multi_split(X_diffused, U, labels, num_nodes, num_classes,
                                          fixed_split_idx, split_type, num_splits, 
                                          num_seeds, device):
    """
    Run equivariant fix test across multiple splits and aggregate results
    """
    all_results = {
        'sgc_standard': [],
        'restricted_standard': [],
        'restricted_rownorm': [],
        'restricted_magnitude_preserving': [],
    }
    
    for split_i in range(num_splits):
        # Generate split indices
        if split_type == 'fixed' and fixed_split_idx is not None:
            train_idx = fixed_split_idx['train_idx']
            val_idx = fixed_split_idx['val_idx']
            test_idx = fixed_split_idx['test_idx']
        else:
            # Random 60/20/20 split
            np.random.seed(split_i)
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        if split_i == 0:
            print(f'    Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}')
        
        # Run for this split
        split_results = run_equivariant_fix_test(
            X_diffused, U, labels, train_idx, val_idx, test_idx,
            num_classes, num_seeds, device
        )
        
        # Accumulate results
        for method in all_results.keys():
            all_results[method].extend(split_results[method])
    
    # Aggregate results
    aggregated = {}
    for method, runs in all_results.items():
        test_accs = [r['test_acc'] for r in runs]
        aggregated[method] = {
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
            'num_runs': len(test_accs),
        }
    
    # Compute gaps
    sgc_acc = aggregated['sgc_standard']['test_acc_mean']
    restricted_std_acc = aggregated['restricted_standard']['test_acc_mean']
    restricted_rn_acc = aggregated['restricted_rownorm']['test_acc_mean']
    restricted_mp_acc = aggregated['restricted_magnitude_preserving']['test_acc_mean']
    
    gaps = {
        'part_a_gap': restricted_std_acc - sgc_acc,
        'part_b_improvement': restricted_rn_acc - restricted_std_acc,
        'hordan_fix_improvement': restricted_mp_acc - restricted_rn_acc,
        'total_vs_baseline': restricted_mp_acc - sgc_acc,
        'magnitude_preserving_vs_rownorm': restricted_mp_acc - restricted_rn_acc,
    }
    
    return aggregated, gaps

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_magnitude_distributions(all_magnitude_data, output_path):
    """Plot magnitude distributions across datasets and matrices"""
    datasets = list(all_magnitude_data.keys())
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        return
    
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 4*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    matrix_names = ['features', 'X_diffused', 'U']
    
    for i, dataset in enumerate(datasets):
        for j, mat_name in enumerate(matrix_names):
            ax = axes[i, j]
            
            if mat_name in all_magnitude_data[dataset]:
                data = all_magnitude_data[dataset][mat_name]
                magnitudes = data['magnitudes']
                labels = data['labels']
                num_classes = len(np.unique(labels))
                
                for c in range(min(num_classes, 10)):
                    class_mag = magnitudes[labels == c]
                    ax.hist(class_mag, bins=50, alpha=0.5, label=f'Class {c}', density=True)
                
                ax.set_xlabel('Magnitude')
                ax.set_ylabel('Density')
                ax.set_title(f'{dataset}\n{mat_name}')
                if j == 2:
                    ax.legend(fontsize=6, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_sparsity_summary(all_sparsity_data, output_path):
    """Plot sparsity summary across datasets"""
    thresh_key = 'abs_1e-6'
    
    datasets = list(all_sparsity_data.keys())
    matrix_names = ['features', 'X_diffused', 'U']
    
    data_to_plot = {mat: {'sparsity': [], 'ref_nodes': []} for mat in matrix_names}
    
    for dataset in datasets:
        for mat_name in matrix_names:
            if mat_name in all_sparsity_data[dataset]:
                sparsity = all_sparsity_data[dataset][mat_name]['thresholds'][thresh_key]['sparsity_ratio'] * 100
                ref_nodes = all_sparsity_data[dataset][mat_name]['thresholds'][thresh_key]['rows_with_no_zeros_pct']
                data_to_plot[mat_name]['sparsity'].append(sparsity)
                data_to_plot[mat_name]['ref_nodes'].append(ref_nodes)
            else:
                data_to_plot[mat_name]['sparsity'].append(0)
                data_to_plot[mat_name]['ref_nodes'].append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    ax1 = axes[0]
    for i, mat_name in enumerate(matrix_names):
        ax1.bar(x + i*width, data_to_plot[mat_name]['sparsity'], width, label=mat_name)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Sparsity Ratio (%)')
    ax1.set_title(f'Sparsity Ratio (threshold={thresh_key})')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    for i, mat_name in enumerate(matrix_names):
        ax2.bar(x + i*width, data_to_plot[mat_name]['ref_nodes'], width, label=mat_name)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Rows with No Zeros (%)')
    ax2.set_title(f'Reference Nodes (Theorem 3)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_equivariant_fix_results(all_fix_results, output_path):
    """Plot equivariant fix test results"""
    datasets = list(all_fix_results.keys())
    
    methods = ['sgc_standard', 'restricted_standard', 'restricted_rownorm', 'restricted_magnitude_preserving']
    method_labels = ['SGC+Std', 'Restr+Std', 'Restr+RN', 'Restr+LogMag']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        accs = []
        stds = []
        for dataset in datasets:
            if dataset in all_fix_results and method in all_fix_results[dataset]:
                accs.append(all_fix_results[dataset][method]['test_acc_mean'] * 100)
                stds.append(all_fix_results[dataset][method]['test_acc_std'] * 100)
            else:
                accs.append(0)
                stds.append(0)
        
        ax.bar(x + i*width, accs, width, label=label, color=color, yerr=stds, capsize=3)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Equivariant Fix Test: Does Preserving Magnitude Close the Gap?')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Main Analysis Loop
# ============================================================================

def analyze_dataset(dataset_name, k_diffusion, split_type, num_splits, num_seeds, device):
    """Run all Hordan analyses for a single dataset"""
    print(f'\n{"="*70}')
    print(f'ANALYZING: {dataset_name} (k={k_diffusion}, {split_type} splits)')
    print(f'{"="*70}')
    
    results = {
        'dataset': dataset_name,
        'k_diffusion': k_diffusion,
        'sparsity': {},
        'magnitude_correlation': {},
        'reference_nodes': {},
        'equivariant_fix': None,
        'metadata': {}
    }
    
    magnitude_data = {}
    
    try:
        # Load dataset using utils.py
        print('\n[1/6] Loading dataset...')
        (edge_index, features, labels, num_nodes, num_classes,
         train_idx, val_idx, test_idx) = load_dataset(dataset_name)
        
        print(f'  Nodes: {num_nodes}, Features: {features.shape[1]}, Classes: {num_classes}')
        
        # Build graph matrices using utils.py
        # Note: build_graph_matrices returns (adj, L, D)
        print('\n[2/6] Building graph matrices...')
        adj, L, D = build_graph_matrices(edge_index, num_nodes)
        
        # Extract LCC using utils.py
        print('\n[3/6] Extracting largest connected component...')
        lcc_mask = get_largest_connected_component_nx(adj)
        lcc_size = lcc_mask.sum()
        print(f'  LCC size: {lcc_size} ({lcc_size/num_nodes*100:.1f}%)')
        
        # Package splits
        split_idx = None
        if train_idx is not None:
            split_idx = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
        
        # Extract subgraph using utils.py
        adj, features, labels, split_idx = extract_subgraph(adj, features, labels, lcc_mask, split_idx)
        
        # Rebuild matrices for LCC
        adj_coo = adj.tocoo()
        edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
        adj, L, D = build_graph_matrices(edge_index_lcc, adj.shape[0])
        
        num_nodes = features.shape[0]
        results['metadata']['num_nodes_lcc'] = num_nodes
        results['metadata']['num_features'] = features.shape[1]
        results['metadata']['num_classes'] = num_classes
        results['metadata']['num_components'] = 1
        results['metadata']['split_type'] = split_type
        results['metadata']['num_splits'] = num_splits
        
        # Create splits if not available
        if split_idx is None:
            np.random.seed(42)
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        else:
            train_idx = split_idx['train_idx']
            val_idx = split_idx['val_idx']
            test_idx = split_idx['test_idx']
        
        print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
        
        # Store fixed split info (may be overridden by random splits in experiment)
        fixed_split_idx = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        
        # Compute diffused features using utils.py
        print(f'\n[4/6] Computing diffused features (k={k_diffusion})...')
        A_sgc = compute_sgc_normalized_adjacency(adj)
        features_dense = features.toarray() if sp.issparse(features) else features
        X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diffusion)
        print(f'  X_diffused shape: {X_diffused.shape}')
        
        # Compute restricted eigenvectors using utils.py
        print('\n[5/6] Computing restricted eigenvectors...')
        U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(X_diffused, L, D, num_components=0)
        print(f'  U shape: {U.shape}, d_eff: {d_eff}, ortho_error: {ortho_error:.2e}')
        results['metadata']['d_effective'] = d_eff
        results['metadata']['ortho_error'] = float(ortho_error)
        
        # ====================================================================
        # Run Experiments
        # ====================================================================
        
        print('\n[6/6] Running Hordan experiments...')
        
        matrices = {
            'features': features_dense.astype(np.float32),
            'X_diffused': X_diffused.astype(np.float32),
            'U': U,
        }
        
        # Experiment 1: Sparsity Analysis
        print('\n  [Exp 1] Sparsity Analysis...')
        for mat_name, mat in matrices.items():
            results['sparsity'][mat_name] = analyze_sparsity(mat, mat_name, NEAR_ZERO_THRESHOLDS)
            print(f'    {mat_name}: {results["sparsity"][mat_name]["thresholds"]["abs_1e-6"]["sparsity_ratio"]*100:.2f}% sparse, '
                  f'{results["sparsity"][mat_name]["thresholds"]["abs_1e-6"]["rows_with_no_zeros_pct"]:.1f}% ref nodes')
        
        # Experiment 2: Magnitude-Class Correlation
        print('\n  [Exp 2] Magnitude-Class Correlation...')
        for mat_name, mat in matrices.items():
            mag_results, magnitudes = analyze_magnitude_class_correlation(mat, labels, mat_name)
            results['magnitude_correlation'][mat_name] = mag_results
            magnitude_data[mat_name] = {'magnitudes': magnitudes, 'labels': labels}
            
            fisher = mag_results['separability_metrics']['fisher_ratio']
            nmi = mag_results['separability_metrics']['normalized_mutual_info']
            print(f'    {mat_name}: Fisher={fisher:.4f}, NMI={nmi:.4f}')
        
        # Experiment 3: Reference Node Analysis
        print('\n  [Exp 3] Reference Node Analysis...')
        for mat_name, mat in matrices.items():
            results['reference_nodes'][mat_name] = analyze_reference_nodes(mat, labels, mat_name, NEAR_ZERO_THRESHOLDS)
        
        # Experiment 4: Equivariant Fix Test
        print('\n  [Exp 4] Equivariant Fix Test...')
        print(f'    Running {num_splits} splits × {num_seeds} seeds = {num_splits * num_seeds} total runs')
        fix_results, gaps = run_equivariant_fix_test_multi_split(
            X_diffused, U, labels, num_nodes, num_classes,
            fixed_split_idx, split_type, num_splits, num_seeds, device
        )
        results['equivariant_fix'] = {
            'results': fix_results,
            'gaps': gaps
        }
        
        print(f'    SGC+Std:        {fix_results["sgc_standard"]["test_acc_mean"]*100:.2f}%')
        print(f'    Restr+Std:      {fix_results["restricted_standard"]["test_acc_mean"]*100:.2f}%')
        print(f'    Restr+RN:       {fix_results["restricted_rownorm"]["test_acc_mean"]*100:.2f}%')
        print(f'    Restr+LogMag:   {fix_results["restricted_magnitude_preserving"]["test_acc_mean"]*100:.2f}%')
        print(f'    Part A Gap:     {gaps["part_a_gap"]*100:+.2f}pp')
        print(f'    Hordan Fix:     {gaps["magnitude_preserving_vs_rownorm"]*100:+.2f}pp')
        
        return results, magnitude_data
        
    except Exception as e:
        print(f'  ERROR: {str(e)}')
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    
    all_results = {}
    all_magnitude_data = {}
    all_sparsity_data = {}
    all_fix_results = {}
    
    for dataset_name in DATASETS:
        for k_diff in K_DIFFUSION_VALUES:
            key = f'{dataset_name}_k{k_diff}'
            
            results, mag_data = analyze_dataset(dataset_name, k_diff, SPLIT_TYPE, NUM_SPLITS, NUM_SEEDS, device)
            
            if results is not None:
                all_results[key] = results
                all_magnitude_data[key] = mag_data
                all_sparsity_data[key] = results['sparsity']
                if results['equivariant_fix'] is not None:
                    all_fix_results[key] = results['equivariant_fix']['results']
    
    # ========================================================================
    # Generate Summary Tables
    # ========================================================================
    
    print('\n' + '='*80)
    print('SUMMARY: SPARSITY ANALYSIS (threshold=abs_1e-6)')
    print('='*80)
    print(f'{"Dataset":<25} {"Matrix":<12} {"Sparsity%":<12} {"RefNodes%":<12} {"Thm3":<8}')
    print('-'*80)
    
    for key, sparsity_data in all_sparsity_data.items():
        for mat_name in ['features', 'X_diffused', 'U']:
            if mat_name in sparsity_data:
                data = sparsity_data[mat_name]['thresholds']['abs_1e-6']
                thm3 = '✓' if data['theorem3_satisfied'] else '✗'
                print(f'{key:<25} {mat_name:<12} {data["sparsity_ratio"]*100:<12.2f} {data["rows_with_no_zeros_pct"]:<12.1f} {thm3:<8}')
    
    print('\n' + '='*80)
    print('SUMMARY: MAGNITUDE-CLASS CORRELATION')
    print('='*80)
    print(f'{"Dataset":<25} {"Matrix":<12} {"Fisher":<12} {"NMI":<12} {"Predictive":<12}')
    print('-'*80)
    
    for key, results in all_results.items():
        for mat_name in ['features', 'X_diffused', 'U']:
            if mat_name in results['magnitude_correlation']:
                data = results['magnitude_correlation'][mat_name]['separability_metrics']
                pred = '✓' if data['magnitude_is_predictive'] else '✗'
                print(f'{key:<25} {mat_name:<12} {data["fisher_ratio"]:<12.4f} {data["normalized_mutual_info"]:<12.4f} {pred:<12}')
    
    print('\n' + '='*80)
    print('SUMMARY: EQUIVARIANT FIX TEST')
    print('='*80)
    print(f'{"Dataset":<25} {"SGC+Std":<12} {"Rst+Std":<12} {"Rst+RN":<12} {"Rst+LogMag":<12} {"HordanFix":<12}')
    print('-'*80)
    
    for key, results in all_results.items():
        if results['equivariant_fix'] is not None:
            fix = results['equivariant_fix']
            res = fix['results']
            gaps = fix['gaps']
            
            print(f'{key:<25} '
                  f'{res["sgc_standard"]["test_acc_mean"]*100:<12.2f} '
                  f'{res["restricted_standard"]["test_acc_mean"]*100:<12.2f} '
                  f'{res["restricted_rownorm"]["test_acc_mean"]*100:<12.2f} '
                  f'{res["restricted_magnitude_preserving"]["test_acc_mean"]*100:<12.2f} '
                  f'{gaps["magnitude_preserving_vs_rownorm"]*100:+.2f}pp')
    
    # ========================================================================
    # Generate Visualizations
    # ========================================================================
    
    print('\n' + '='*80)
    print('GENERATING VISUALIZATIONS')
    print('='*80)
    
    if all_sparsity_data:
        plot_sparsity_summary(all_sparsity_data, f'{OUTPUT_DIR}/plots/sparsity_summary.png')
        print(f'✓ Saved: {OUTPUT_DIR}/plots/sparsity_summary.png')
    
    if all_fix_results:
        plot_equivariant_fix_results(all_fix_results, f'{OUTPUT_DIR}/plots/equivariant_fix_results.png')
        print(f'✓ Saved: {OUTPUT_DIR}/plots/equivariant_fix_results.png')
    
    if all_magnitude_data:
        sample_data = dict(list(all_magnitude_data.items())[:4])
        if sample_data:
            plot_magnitude_distributions(sample_data, f'{OUTPUT_DIR}/plots/magnitude_distributions.png')
            print(f'✓ Saved: {OUTPUT_DIR}/plots/magnitude_distributions.png')
    
    # ========================================================================
    # Save Results to JSON
    # ========================================================================
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(all_results)
    
    with open(f'{OUTPUT_DIR}/hordan_analysis_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f'\n✓ Results saved: {OUTPUT_DIR}/hordan_analysis_results.json')
    
    # ========================================================================
    # Final Interpretation
    # ========================================================================
    
    print('\n' + '='*80)
    print('INTERPRETATION: HORDAN FRAMEWORK ANALYSIS')
    print('='*80)
    
    print('''
Key Questions Answered:

1. SPARSITY (Theorem 3 Condition):
   - If sparsity is HIGH and few reference nodes exist → Hordan mechanism likely
   - If sparsity is LOW with many reference nodes → Other factors dominate

2. MAGNITUDE-CLASS CORRELATION:
   - HIGH Fisher ratio / NMI → Magnitude carries class info → RowNorm hurts
   - LOW Fisher ratio / NMI → Magnitude not discriminative → RowNorm safe

3. EQUIVARIANT FIX TEST:
   - If LogMagnitudeMLP >> RowNormMLP → Hordan explanation confirmed
   - If LogMagnitudeMLP ≈ RowNormMLP → Other factors cause the gap

Look for patterns:
- Datasets where magnitude IS predictive but RowNorm still helps → Complex interaction
- Datasets where sparsity differs between X_diffused and U → Basis transformation creates/removes zeros
- Datasets where Hordan fix closes gap → Direct evidence of information loss through normalization
''')
    
    print(f'\nAll results saved to: {OUTPUT_DIR}/')
    print('='*80)