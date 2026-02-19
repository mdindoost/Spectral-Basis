"""
================================================================================
LABEL SMOOTHNESS DIAGNOSTIC: y^T L y Analysis
================================================================================

This script computes the Dirichlet energy (label smoothness) of class labels
on the graph and correlates it with Part A (basis sensitivity) results.

Hypothesis: Low label smoothness (labels align with graph structure) should
predict when eigenvectors perform well vs poorly.

Usage (run from repo root):
    python experiments/compute_label_smoothness_diagnostic.py --split_type fixed
    python experiments/compute_label_smoothness_diagnostic.py --split_type random
    python experiments/compute_label_smoothness_diagnostic.py --split_type both

Output:
    results/label_smoothness_diagnostic/
    ├── fixed_splits/
    │   ├── label_smoothness_results.json
    │   ├── label_smoothness_vs_partA.pdf
    │   ├── per_class_analysis.pdf
    │   └── comprehensive_report.txt
    └── random_splits/
        └── ...

Author: Mohammad Dindoost
Date: February 2026
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Fix for PyTorch 2.6 + OGB compatibility
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import from utils
try:
    from utils import load_dataset, build_graph_matrices
    UTILS_AVAILABLE = True
except ImportError:
    print("ERROR: Could not import from utils.py")
    print("Make sure utils.py is in the same directory or in PYTHONPATH")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Label Smoothness Diagnostic')
parser.add_argument('--split_type', type=str, 
                   choices=['fixed', 'random', 'both'],
                   default='both',
                   help='Which split type to analyze')
parser.add_argument('--k_value', type=int, default=10,
                   help='Which k value to use for Part A (default: 10)')
parser.add_argument('--results_dir', type=str,
                   default='results/investigation_sgc_antiSmoothing',
                   help='Base directory containing Part A results')
parser.add_argument('--output_dir', type=str,
                   default='results/label_smoothness_diagnostic',
                   help='Output directory for diagnostic results')
parser.add_argument('--data_root', type=str,
                   default='./dataset',
                   help='Root directory for datasets')

args = parser.parse_args()

SPLIT_TYPE = args.split_type
K_VALUE = args.k_value
RESULTS_DIR = args.results_dir
OUTPUT_DIR = args.output_dir
DATA_ROOT = args.data_root

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plotting style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')
sns.set_palette('husl')

print('='*80)
print('LABEL SMOOTHNESS DIAGNOSTIC: y^T L y Analysis')
print('='*80)
print(f'Split type: {SPLIT_TYPE}')
print(f'K value for Part A: {K_VALUE}')
print(f'Results directory: {RESULTS_DIR}')
print(f'Output directory: {OUTPUT_DIR}')
print(f'Data root: {DATA_ROOT}')
print('='*80)
print()

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASETS = [
    'ogbn-arxiv',
    'cora',
    'citeseer',
    'pubmed',
    'wikics',
    'amazon-photo',
    'amazon-computers',
    'coauthor-cs',
    'coauthor-physics'
]

# ============================================================================
# Label Smoothness Computation Functions
# ============================================================================

def compute_label_smoothness_multiclass(L, D, labels, num_classes):
    """
    Compute label smoothness for multi-class classification using Rayleigh quotient.
    
    For multi-class, we use one-hot encoding Y and compute:
        Energy_raw = Tr(Y^T L Y)
        Normalization = Tr(Y^T D Y)
        Energy_normalized = Energy_raw / Normalization (Rayleigh quotient)
    
    The normalized energy is in [0, 2] for normalized Laplacian:
        - 0 = perfect smoothness (disconnected clusters)
        - 1 = random labels
        - 2 = worst case (all edges cross class boundaries)
    
    Args:
        L: Graph Laplacian (n x n sparse or dense)
        D: Degree matrix (n x n sparse or dense)
        labels: Label vector (n,)
        num_classes: Number of classes
    
    Returns:
        dict with energy metrics and per-class breakdown
    """
    n = len(labels)

    # Compute Tr(Y^T L Y) and Tr(Y^T D Y) using sparse mat-vec products
    # Tr(Y^T M Y) = sum_c y_c^T M y_c, where y_c is the c-th column of Y
    # This avoids converting L/D to dense (infeasible for large graphs)
    deg = D.diagonal() if sp.issparse(D) else np.diag(D)
    volume = float(np.sum(deg))

    energy_raw = 0.0
    normalization = 0.0
    per_class = {}

    for c in range(num_classes):
        y_c = (labels == c).astype(np.float64)
        n_c = int(y_c.sum())

        # Energy for this class: y_c^T L y_c
        energy_c = float(y_c @ (L @ y_c))

        # Volume for this class: y_c^T D y_c = sum of degrees of class c nodes
        volume_c = float(y_c @ (D @ y_c))

        energy_raw += energy_c
        normalization += volume_c

        # Normalized energy for this class
        if volume_c > 1e-10:
            energy_c_norm = energy_c / volume_c
        else:
            energy_c_norm = float('inf')

        per_class[int(c)] = {
            'count': n_c,
            'energy_raw': float(energy_c),
            'volume': float(volume_c),
            'energy_normalized': float(energy_c_norm)
        }

    # Rayleigh quotient (normalized energy)
    if normalization > 1e-10:
        energy_normalized = energy_raw / normalization
    else:
        energy_normalized = float('inf')

    # Alternative normalization: by graph volume
    if volume > 1e-10:
        energy_per_volume = energy_raw / volume
    else:
        energy_per_volume = float('inf')

    return {
        'energy_raw': float(energy_raw),
        'normalization': float(normalization),
        'energy_normalized': float(energy_normalized),
        'energy_per_volume': float(energy_per_volume),
        'volume': float(volume),
        'per_class': per_class
    }

def compute_label_smoothness_binary_all_pairs(L, D, labels, num_classes):
    """
    Compute label smoothness for all pairwise class comparisons.
    
    For each pair of classes (c1, c2), create binary indicator:
        y[i] = +1 if label[i] == c1
        y[i] = -1 if label[i] == c2
        y[i] = 0 otherwise
    
    Then compute y^T L y / y^T D y
    
    This measures how well each class boundary aligns with graph structure.
    
    Args:
        L: Graph Laplacian (n x n)
        D: Degree matrix (n x n)
        labels: Label vector (n,)
        num_classes: Number of classes
    
    Returns:
        dict with pairwise smoothness metrics
    """
    n = len(labels)

    pairwise = {}

    for c1 in range(num_classes):
        for c2 in range(c1 + 1, num_classes):
            # Create binary indicator
            y = np.zeros(n, dtype=np.float64)
            mask_c1 = (labels == c1)
            mask_c2 = (labels == c2)

            y[mask_c1] = 1.0
            y[mask_c2] = -1.0

            n_c1 = int(mask_c1.sum())
            n_c2 = int(mask_c2.sum())

            if n_c1 == 0 or n_c2 == 0:
                continue

            # Compute energy (sparse mat-vec, no dense conversion)
            energy_raw = float(y @ (L @ y))
            normalization = float(y @ (D @ y))
            
            if normalization > 1e-10:
                energy_normalized = energy_raw / normalization
            else:
                energy_normalized = float('inf')
            
            pair_key = f'{c1}_{c2}'
            pairwise[pair_key] = {
                'class1': int(c1),
                'class2': int(c2),
                'count1': int(n_c1),
                'count2': int(n_c2),
                'energy_raw': float(energy_raw),
                'normalization': float(normalization),
                'energy_normalized': float(energy_normalized)
            }
    
    # Average pairwise energy
    if pairwise:
        avg_pairwise_energy = np.mean([v['energy_normalized'] 
                                       for v in pairwise.values() 
                                       if not np.isinf(v['energy_normalized'])])
    else:
        avg_pairwise_energy = float('nan')
    
    return {
        'pairwise': pairwise,
        'avg_pairwise_energy': float(avg_pairwise_energy)
    }

# ============================================================================
# Part A Results Loading
# ============================================================================

def load_part_a_results(results_dir, dataset, split_type, k_value):
    """
    Load Part A results from investigation_sgc_antiSmoothing.
    
    Part A = Restricted+StandardMLP - SGC+MLP
    
    Args:
        results_dir: Base results directory
        dataset: Dataset name
        split_type: 'fixed' or 'random'
        k_value: Which k value to load
    
    Returns:
        dict with Part A metrics, or None if not found
    """
    # Construct path
    dataset_dir = f'{dataset}_{split_type}_lcc'
    k_dir = f'k{k_value}'
    results_file = os.path.join(results_dir, dataset_dir, k_dir, 
                                'metrics', 'results.json')
    
    if not os.path.exists(results_file):
        print(f'  ⚠️  Results not found: {results_file}')
        return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        results_dict = data.get('results', {})
        
        # Extract accuracies
        sgc_mlp = results_dict.get('sgc_mlp_baseline', {}).get('test_acc_mean', None)
        restricted_std = results_dict.get('restricted_standard_mlp', {}).get('test_acc_mean', None)
        restricted_row = results_dict.get('full_rownorm_mlp', {}).get('test_acc_mean', None)
        
        if sgc_mlp is None or restricted_std is None:
            print(f'  ⚠️  Missing required keys in: {results_file}')
            return None
        
        # Compute Part A (in percentage points)
        part_a = (restricted_std - sgc_mlp) * 100.0
        
        # Also compute Part B and Gap if available
        if restricted_row is not None:
            part_b = (restricted_row - restricted_std) * 100.0
            gap = (restricted_row - sgc_mlp) * 100.0
        else:
            part_b = None
            gap = None
        
        return {
            'sgc_mlp': sgc_mlp * 100.0,
            'restricted_std': restricted_std * 100.0,
            'restricted_row': restricted_row * 100.0 if restricted_row else None,
            'part_a': part_a,
            'part_b': part_b,
            'gap': gap,
            'k': k_value
        }
        
    except Exception as e:
        print(f'  ❌ Error loading {results_file}: {e}')
        return None

def find_available_k_values(results_dir, dataset, split_type):
    """
    Find all available k values for a dataset.
    """
    dataset_dir = os.path.join(results_dir, f'{dataset}_{split_type}_lcc')
    
    if not os.path.exists(dataset_dir):
        return []
    
    k_values = []
    for item in os.listdir(dataset_dir):
        if item.startswith('k') and os.path.isdir(os.path.join(dataset_dir, item)):
            try:
                k = int(item[1:])
                k_values.append(k)
            except ValueError:
                continue
    
    return sorted(k_values)

# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def analyze_label_smoothness_single_split(split_type, k_value):
    """
    Run complete analysis for one split type.
    """
    print(f'\n{"="*80}')
    print(f'ANALYZING: {split_type.upper()} SPLITS')
    print(f'{"="*80}\n')
    
    # Output directory for this split type
    output_split_dir = os.path.join(OUTPUT_DIR, f'{split_type}_splits')
    os.makedirs(output_split_dir, exist_ok=True)
    
    results_list = []
    
    for dataset_name in DATASETS:
        print(f'\n{"-"*80}')
        print(f'Dataset: {dataset_name}')
        print(f'{"-"*80}')
        
        # ====================================================================
        # Step 1: Load dataset and build graph
        # ====================================================================
        print(f'\n[1/3] Loading dataset and building graph...')
        
        try:
            edge_index, X, labels, num_nodes, num_classes, train_idx, val_idx, test_idx = \
                load_dataset(dataset_name, root=DATA_ROOT)
            
            # Build graph matrices
            adj, L, D = build_graph_matrices(edge_index, num_nodes)

            # Extract LCC to match partAB experiments (results are from LCC subgraph)
            G = nx.from_scipy_sparse_array(adj)
            num_components = nx.number_connected_components(G)
            print(f'  Components: {num_components}')
            components = list(nx.connected_components(G))
            if num_components > 1:
                largest_cc = max(components, key=len)
                lcc_mask = np.zeros(num_nodes, dtype=bool)
                lcc_mask[sorted(list(largest_cc))] = True
                print(f'  LCC: {lcc_mask.sum()}/{num_nodes} nodes ({lcc_mask.sum()/num_nodes*100:.1f}%)')

                # Extract subgraph
                adj_sub = adj[lcc_mask][:, lcc_mask]
                labels = labels[lcc_mask]
                X = X[lcc_mask]
                num_nodes = int(lcc_mask.sum())
                num_classes = len(np.unique(labels))

                # Rebuild graph matrices for LCC
                adj_coo = adj_sub.tocoo()
                edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
                adj, L, D = build_graph_matrices(edge_index_lcc, num_nodes)

            print(f'  ✓ Nodes: {num_nodes}')
            print(f'  ✓ Edges: {adj.nnz // 2}')
            print(f'  ✓ Classes: {num_classes}')
            print(f'  ✓ Features: {X.shape[1]}')
            
        except Exception as e:
            print(f'  ❌ Error loading dataset: {e}')
            continue
        
        # ====================================================================
        # Step 2: Compute label smoothness
        # ====================================================================
        print(f'\n[2/3] Computing label smoothness...')
        
        try:
            # Multi-class smoothness (primary metric)
            smoothness_multi = compute_label_smoothness_multiclass(
                L, D, labels, num_classes
            )
            
            print(f'  Label Smoothness (Rayleigh quotient):')
            print(f'    Energy (normalized): {smoothness_multi["energy_normalized"]:.6f}')
            print(f'    Energy (raw): {smoothness_multi["energy_raw"]:.2f}')
            print(f'    Normalization term: {smoothness_multi["normalization"]:.2f}')
            print(f'    Graph volume: {smoothness_multi["volume"]:.2f}')
            
            # Interpretation
            energy_norm = smoothness_multi['energy_normalized']
            if energy_norm < 0.5:
                interp = 'VERY SMOOTH (strong graph-label alignment)'
            elif energy_norm < 1.0:
                interp = 'SMOOTH (moderate graph-label alignment)'
            elif energy_norm < 1.5:
                interp = 'ROUGH (weak graph-label alignment)'
            else:
                interp = 'VERY ROUGH (poor graph-label alignment)'
            
            print(f'    Interpretation: {interp}')
            
            # Pairwise smoothness
            smoothness_pairwise = compute_label_smoothness_binary_all_pairs(
                L, D, labels, num_classes
            )
            
            print(f'  Average pairwise smoothness: {smoothness_pairwise["avg_pairwise_energy"]:.6f}')
            
        except Exception as e:
            print(f'  ❌ Error computing smoothness: {e}')
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # Step 3: Load Part A results
        # ====================================================================
        print(f'\n[3/3] Loading Part A results...')
        
        # First, check what k values are available
        available_k = find_available_k_values(RESULTS_DIR, dataset_name, split_type)
        
        if not available_k:
            print(f'  ⚠️  No results found for {dataset_name} ({split_type})')
            continue
        
        print(f'  Available k values: {available_k}')
        
        # Use requested k_value or max available
        if k_value in available_k:
            use_k = k_value
        else:
            use_k = max(available_k)
            print(f'  ⚠️  k={k_value} not found, using k={use_k}')
        
        part_a_data = load_part_a_results(RESULTS_DIR, dataset_name, split_type, use_k)
        
        if part_a_data is None:
            print(f'  ⚠️  Could not load Part A results')
            continue
        
        print(f'  ✓ Part A (k={use_k}): {part_a_data["part_a"]:+.2f}pp')
        if part_a_data['part_b'] is not None:
            print(f'  ✓ Part B: {part_a_data["part_b"]:+.2f}pp')
            print(f'  ✓ Gap: {part_a_data["gap"]:+.2f}pp')
        
        # ====================================================================
        # Collect results
        # ====================================================================
        results_list.append({
            'dataset': dataset_name,
            'num_nodes': num_nodes,
            'num_edges': adj.nnz // 2,
            'num_classes': num_classes,
            'num_components': num_components,
            'num_features': X.shape[1],
            'label_smoothness_normalized': smoothness_multi['energy_normalized'],
            'label_smoothness_raw': smoothness_multi['energy_raw'],
            'label_smoothness_per_volume': smoothness_multi['energy_per_volume'],
            'avg_pairwise_smoothness': smoothness_pairwise['avg_pairwise_energy'],
            'part_a': part_a_data['part_a'],
            'part_b': part_a_data['part_b'],
            'gap': part_a_data['gap'],
            'sgc_mlp': part_a_data['sgc_mlp'],
            'restricted_std': part_a_data['restricted_std'],
            'restricted_row': part_a_data['restricted_row'],
            'k_value': use_k,
            'smoothness_per_class': smoothness_multi['per_class'],
            'smoothness_pairwise': smoothness_pairwise['pairwise']
        })
    
    # ========================================================================
    # Analysis and Visualization
    # ========================================================================
    if not results_list:
        print(f'\n⚠️  No results collected for {split_type} splits')
        return None
    
    df = pd.DataFrame(results_list)
    
    # Save raw results
    results_file = os.path.join(output_split_dir, 'label_smoothness_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f'\n✓ Saved results: {results_file}')
    
    # Compute correlation
    correlation = df['label_smoothness_normalized'].corr(df['part_a'])
    correlation_pairwise = df['avg_pairwise_smoothness'].corr(df['part_a'])
    
    print(f'\n{"="*80}')
    print(f'CORRELATION ANALYSIS ({split_type.upper()} SPLITS)')
    print(f'{"="*80}')
    print(f'Label Smoothness (multi-class) vs Part A: {correlation:.3f}')
    print(f'Label Smoothness (pairwise avg) vs Part A: {correlation_pairwise:.3f}')
    
    if abs(correlation) > 0.6:
        strength = 'STRONG'
    elif abs(correlation) > 0.3:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'
    
    direction = 'negative' if correlation < 0 else 'positive'
    
    print(f'\nInterpretation: {strength} {direction} correlation')
    
    if correlation < -0.5:
        print('  → Low smoothness (rough labels) → Negative Part A')
        print('  → High smoothness (smooth labels) → Positive Part A')
        print('  ✓ Hypothesis SUPPORTED: Label smoothness predicts basis sensitivity')
    elif correlation > 0.5:
        print('  → Low smoothness → Positive Part A')
        print('  → High smoothness → Negative Part A')
        print('  ⚠️  Unexpected pattern (opposite of hypothesis)')
    else:
        print('  ⚠️  Label smoothness is NOT a strong predictor of Part A')
    
    print(f'{"="*80}')
    
    # ========================================================================
    # Visualization 1: Scatter plot with correlation
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(df['label_smoothness_normalized'], df['part_a'], 
                        s=150, alpha=0.6, edgecolors='black', linewidths=1.5)
    
    # Add dataset labels
    for _, row in df.iterrows():
        ax.annotate(row['dataset'], 
                   (row['label_smoothness_normalized'], row['part_a']),
                   xytext=(7, 7), textcoords='offset points', 
                   fontsize=9, alpha=0.8)
    
    # Add trend line
    z = np.polyfit(df['label_smoothness_normalized'], df['part_a'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['label_smoothness_normalized'].min(), 
                        df['label_smoothness_normalized'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
           label=f'Linear fit (r={correlation:.3f})')
    
    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.3, linewidth=1, 
              label='Random labels threshold')
    
    # Labels and title
    ax.set_xlabel('Label Smoothness (y^T L y / y^T D y)\n← More Smooth    |    More Rough →', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Part A Gap (pp)\n← Eigenvectors Worse    |    Eigenvectors Better →', 
                 fontsize=12, fontweight='bold')
    ax.set_title(f'Label Smoothness vs MLP Basis Sensitivity\n'
                f'{split_type.upper()} Splits | k={df.iloc[0]["k_value"]} | '
                f'Correlation: {correlation:.3f}',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add interpretation text box
    textstr = f'Correlation: {correlation:.3f} ({strength} {direction})\n'
    textstr += f'Range: [{df["label_smoothness_normalized"].min():.3f}, {df["label_smoothness_normalized"].max():.3f}]\n'
    textstr += f'Mean: {df["label_smoothness_normalized"].mean():.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_split_dir, 'label_smoothness_vs_partA.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved plot: {plot_file}')
    plt.close()
    
    # ========================================================================
    # Visualization 2: Per-class analysis
    # ========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= 9:
            break
        
        ax = axes[idx]
        
        # Extract per-class data
        per_class = row['smoothness_per_class']
        classes = sorted([int(k) for k in per_class.keys()])
        energies = [per_class[c]['energy_normalized'] for c in classes]
        counts = [per_class[c]['count'] for c in classes]
        
        # Bar plot
        bars = ax.bar(classes, energies, alpha=0.7, edgecolor='black')
        
        # Color bars by energy level
        for bar, energy in zip(bars, energies):
            if energy < 0.5:
                bar.set_color('green')
            elif energy < 1.0:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # Add count labels
        for c, energy, count in zip(classes, energies, counts):
            ax.text(c, energy, f'n={count}', ha='center', va='bottom', 
                   fontsize=7, rotation=0)
        
        ax.set_xlabel('Class', fontsize=10)
        ax.set_ylabel('Label Smoothness', fontsize=10)
        ax.set_title(f'{row["dataset"]} (Part A: {row["part_a"]:+.1f}pp)', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(classes)
    
    # Remove empty subplots
    for idx in range(len(df), 9):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Per-Class Label Smoothness Analysis\n{split_type.upper()} Splits',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_file = os.path.join(output_split_dir, 'per_class_analysis.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved plot: {plot_file}')
    plt.close()
    
    # ========================================================================
    # Visualization 3: Comprehensive dashboard
    # ========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main scatter plot (larger)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Color by Part A sign
    colors = ['red' if x < 0 else 'green' for x in df['part_a']]
    scatter = ax1.scatter(df['label_smoothness_normalized'], df['part_a'],
                         c=colors, s=200, alpha=0.6, edgecolors='black', linewidths=2)
    
    for _, row in df.iterrows():
        ax1.annotate(row['dataset'], 
                    (row['label_smoothness_normalized'], row['part_a']),
                    xytext=(7, 7), textcoords='offset points', fontsize=9)
    
    # Trend line
    z = np.polyfit(df['label_smoothness_normalized'], df['part_a'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['label_smoothness_normalized'].min(), 
                        df['label_smoothness_normalized'].max(), 100)
    ax1.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2)
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Label Smoothness', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Part A Gap (pp)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Main Finding (r={correlation:.3f})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Distribution of smoothness values
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['label_smoothness_normalized'], bins=8, alpha=0.7, 
            edgecolor='black', color='skyblue')
    ax2.axvline(df['label_smoothness_normalized'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {df["label_smoothness_normalized"].mean():.3f}')
    ax2.set_xlabel('Label Smoothness', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Distribution of Part A values
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(df['part_a'], bins=8, alpha=0.7, edgecolor='black', color='lightcoral')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(df['part_a'].mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {df["part_a"].mean():.2f}pp')
    ax3.set_xlabel('Part A Gap (pp)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Distribution', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Summary statistics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Min', 'Max', 'Mean', 'Std'],
        ['Label Smoothness', 
         f'{df["label_smoothness_normalized"].min():.4f}',
         f'{df["label_smoothness_normalized"].max():.4f}',
         f'{df["label_smoothness_normalized"].mean():.4f}',
         f'{df["label_smoothness_normalized"].std():.4f}'],
        ['Part A (pp)',
         f'{df["part_a"].min():.2f}',
         f'{df["part_a"].max():.2f}',
         f'{df["part_a"].mean():.2f}',
         f'{df["part_a"].std():.2f}'],
        ['',  '', '', '', ''],
        ['Correlation', '', '', f'{correlation:.4f}', ''],
        ['Datasets analyzed', '', '', f'{len(df)}', ''],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.0, 0.8, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Label Smoothness Diagnostic Dashboard\n'
                f'{split_type.upper()} Splits | k={df.iloc[0]["k_value"]}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plot_file = os.path.join(output_split_dir, 'comprehensive_dashboard.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved plot: {plot_file}')
    plt.close()
    
    # ========================================================================
    # Generate comprehensive text report
    # ========================================================================
    report_file = os.path.join(output_split_dir, 'comprehensive_report.txt')
    
    with open(report_file, 'w') as f:
        f.write('='*80 + '\n')
        f.write('LABEL SMOOTHNESS DIAGNOSTIC REPORT\n')
        f.write('='*80 + '\n\n')
        
        f.write(f'Split Type: {split_type.upper()}\n')
        f.write(f'K Value: {df.iloc[0]["k_value"]}\n')
        f.write(f'Datasets Analyzed: {len(df)}\n')
        f.write(f'Analysis Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('='*80 + '\n')
        f.write('HYPOTHESIS\n')
        f.write('='*80 + '\n\n')
        
        f.write('Label smoothness (y^T L y / y^T D y) should predict Part A gaps:\n')
        f.write('  - Low smoothness (labels rough on graph) → Negative Part A\n')
        f.write('  - High smoothness (labels smooth on graph) → Positive Part A\n\n')
        
        f.write('='*80 + '\n')
        f.write('MAIN FINDINGS\n')
        f.write('='*80 + '\n\n')
        
        f.write(f'Correlation: {correlation:.4f} ({strength} {direction})\n\n')
        
        if correlation < -0.5:
            f.write('✓ HYPOTHESIS SUPPORTED\n')
            f.write('  Label smoothness is a strong predictor of basis sensitivity.\n')
            f.write('  Datasets with rough labels (high energy) show negative Part A,\n')
            f.write('  while datasets with smooth labels show positive Part A.\n\n')
        elif abs(correlation) < 0.3:
            f.write('✗ HYPOTHESIS NOT SUPPORTED\n')
            f.write('  Label smoothness is NOT a strong predictor of Part A.\n')
            f.write('  Other factors likely dominate basis sensitivity.\n\n')
        else:
            f.write('≈ PARTIAL SUPPORT\n')
            f.write('  Moderate correlation suggests label smoothness plays a role\n')
            f.write('  but is not the sole determinant of basis sensitivity.\n\n')
        
        f.write('='*80 + '\n')
        f.write('DETAILED RESULTS\n')
        f.write('='*80 + '\n\n')
        
        f.write(f'{"Dataset":<20} {"Smoothness":<12} {"Part A":<10} {"Interpretation"}\n')
        f.write('-'*80 + '\n')
        
        for _, row in df.iterrows():
            smoothness = row['label_smoothness_normalized']
            part_a = row['part_a']
            
            if smoothness < 0.5:
                smooth_label = 'Very Smooth'
            elif smoothness < 1.0:
                smooth_label = 'Smooth'
            elif smoothness < 1.5:
                smooth_label = 'Rough'
            else:
                smooth_label = 'Very Rough'
            
            f.write(f'{row["dataset"]:<20} {smoothness:<12.4f} {part_a:<10.2f} {smooth_label}\n')
        
        f.write('\n' + '='*80 + '\n')
        f.write('SUMMARY STATISTICS\n')
        f.write('='*80 + '\n\n')
        
        f.write('Label Smoothness:\n')
        f.write(f'  Mean: {df["label_smoothness_normalized"].mean():.4f}\n')
        f.write(f'  Std:  {df["label_smoothness_normalized"].std():.4f}\n')
        f.write(f'  Min:  {df["label_smoothness_normalized"].min():.4f} ({df.loc[df["label_smoothness_normalized"].idxmin(), "dataset"]})\n')
        f.write(f'  Max:  {df["label_smoothness_normalized"].max():.4f} ({df.loc[df["label_smoothness_normalized"].idxmax(), "dataset"]})\n\n')
        
        f.write('Part A Gap:\n')
        f.write(f'  Mean: {df["part_a"].mean():.2f}pp\n')
        f.write(f'  Std:  {df["part_a"].std():.2f}pp\n')
        f.write(f'  Min:  {df["part_a"].min():.2f}pp ({df.loc[df["part_a"].idxmin(), "dataset"]})\n')
        f.write(f'  Max:  {df["part_a"].max():.2f}pp ({df.loc[df["part_a"].idxmax(), "dataset"]})\n\n')
        
        f.write('='*80 + '\n')
        f.write('INTERPRETATION GUIDE\n')
        f.write('='*80 + '\n\n')
        
        f.write('Label Smoothness (Rayleigh Quotient):\n')
        f.write('  [0.0, 0.5)  = Very Smooth (strong graph-label alignment)\n')
        f.write('  [0.5, 1.0)  = Smooth (moderate alignment)\n')
        f.write('  [1.0, 1.5)  = Rough (weak alignment)\n')
        f.write('  [1.5, 2.0]  = Very Rough (poor alignment)\n\n')
        
        f.write('Part A Gap:\n')
        f.write('  Positive = Eigenvectors outperform diffused features\n')
        f.write('  Negative = Diffused features outperform eigenvectors\n')
        f.write('  Near zero = No basis sensitivity (span equivalence holds)\n\n')
        
        f.write('='*80 + '\n')
        f.write('END OF REPORT\n')
        f.write('='*80 + '\n')
    
    print(f'✓ Saved report: {report_file}')
    
    return df

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    
    split_types = []
    if SPLIT_TYPE == 'both':
        split_types = ['fixed', 'random']
    else:
        split_types = [SPLIT_TYPE]
    
    all_results = {}
    
    for split_type in split_types:
        df = analyze_label_smoothness_single_split(split_type, K_VALUE)
        if df is not None:
            all_results[split_type] = df
    
    # ========================================================================
    # Combined analysis if both splits were analyzed
    # ========================================================================
    if len(all_results) == 2:
        print(f'\n{"="*80}')
        print('COMBINED ANALYSIS: FIXED vs RANDOM SPLITS')
        print(f'{"="*80}\n')
        
        df_fixed = all_results['fixed']
        df_random = all_results['random']
        
        # Merge on dataset
        df_combined = pd.merge(
            df_fixed[['dataset', 'label_smoothness_normalized', 'part_a']],
            df_random[['dataset', 'label_smoothness_normalized', 'part_a']],
            on='dataset',
            suffixes=('_fixed', '_random')
        )
        
        print(f'Correlation (Fixed):  {df_fixed["label_smoothness_normalized"].corr(df_fixed["part_a"]):.3f}')
        print(f'Correlation (Random): {df_random["label_smoothness_normalized"].corr(df_random["part_a"]):.3f}')
        
        # Check if smoothness is consistent across splits
        smoothness_corr = df_combined['label_smoothness_normalized_fixed'].corr(
            df_combined['label_smoothness_normalized_random']
        )
        print(f'\nLabel smoothness consistency (fixed vs random): {smoothness_corr:.3f}')
        
        if smoothness_corr > 0.95:
            print('  ✓ Label smoothness is highly consistent across split types')
        elif smoothness_corr > 0.8:
            print('  ≈ Label smoothness shows good consistency')
        else:
            print('  ⚠️  Label smoothness varies between split types')
        
        # Comparative plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (split_type, df) in enumerate([('Fixed', df_fixed), ('Random', df_random)]):
            ax = axes[idx]
            
            colors = ['red' if x < 0 else 'green' for x in df['part_a']]
            ax.scatter(df['label_smoothness_normalized'], df['part_a'],
                      c=colors, s=150, alpha=0.6, edgecolors='black', linewidths=1.5)
            
            for _, row in df.iterrows():
                ax.annotate(row['dataset'], 
                           (row['label_smoothness_normalized'], row['part_a']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Trend line
            z = np.polyfit(df['label_smoothness_normalized'], df['part_a'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['label_smoothness_normalized'].min(),
                                df['label_smoothness_normalized'].max(), 100)
            corr = df['label_smoothness_normalized'].corr(df['part_a'])
            ax.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2,
                   label=f'r={corr:.3f}')
            
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Label Smoothness', fontsize=11, fontweight='bold')
            ax.set_ylabel('Part A Gap (pp)', fontsize=11, fontweight='bold')
            ax.set_title(f'{split_type} Splits', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Label Smoothness vs Part A: Split Type Comparison',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        plot_file = os.path.join(OUTPUT_DIR, 'split_comparison.pdf')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'\n✓ Saved comparison plot: {plot_file}')
        plt.close()
    
    # ========================================================================
    # Final summary
    # ========================================================================
    print(f'\n{"="*80}')
    print('ANALYSIS COMPLETE')
    print(f'{"="*80}')
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'Datasets analyzed: {len(DATASETS)}')
    print(f'Split types: {", ".join(all_results.keys())}')
    print('\nGenerated files:')
    for split_type in all_results.keys():
        split_dir = os.path.join(OUTPUT_DIR, f'{split_type}_splits')
        print(f'\n  {split_type} splits ({split_dir}):')
        print(f'    - label_smoothness_results.json')
        print(f'    - label_smoothness_vs_partA.pdf')
        print(f'    - per_class_analysis.pdf')
        print(f'    - comprehensive_dashboard.pdf')
        print(f'    - comprehensive_report.txt')
    
    if len(all_results) == 2:
        print(f'\n  Combined:')
        print(f'    - split_comparison.pdf')
    
    print(f'\n{"="*80}')

if __name__ == '__main__':
    main()