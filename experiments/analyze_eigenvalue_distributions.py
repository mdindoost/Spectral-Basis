"""
Eigenvalue Distribution Analysis
=================================

Purpose: Analyze eigenvalue distributions of restricted eigenproblems
to understand when RowNorm works vs fails.

Research Questions:
1. How are eigenvalues distributed in restricted eigenproblems?
2. How does diffusion change eigenvalue distributions?
3. Can eigenvalue properties predict RowNorm success?

Outputs:
- Eigenvalue spectrum plots (before/after diffusion)
- Distribution histograms
- Statistical metrics (spread, clustering, condition)
- Correlation with baseline performance

Usage:
    python analyze_eigenvalue_distributions.py
    
    # Analyze specific datasets
    python analyze_eigenvalue_distributions.py --datasets ogbn-arxiv pubmed cora
    
    # Test specific k values
    python analyze_eigenvalue_distributions.py --k_values 2 4 8
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import pandas as pd
import networkx as nx
from collections import defaultdict

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import load_dataset, build_graph_matrices

# ============================================================================
# Configuration
# ============================================================================

# Parse arguments
DATASETS = ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 
            'wikics', 'amazon-computers', 'coauthor-cs', 'coauthor-physics']

if '--datasets' in sys.argv:
    idx = sys.argv.index('--datasets')
    DATASETS = sys.argv[idx+1:idx+1+len([x for x in sys.argv[idx+1:] if not x.startswith('--')])]

K_VALUES = [2, 4, 6, 8, 10]  # Diffusion steps to test
if '--k_values' in sys.argv:
    idx = sys.argv.index('--k_values')
    K_VALUES = [int(x) for x in sys.argv[idx+1:idx+1+3] if x.isdigit()]

# Output directory
OUTPUT_DIR = 'results/eigenvalue_analysis'
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)

print('='*80)
print('EIGENVALUE DISTRIBUTION ANALYSIS')
print('='*80)
print(f'Datasets: {DATASETS}')
print(f'Diffusion steps to test: {K_VALUES}')
print(f'Output: {OUTPUT_DIR}')
print('='*80)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# Helper Functions
# ============================================================================

def compute_normalized_adjacency(adj, D):
    """Compute symmetric normalized adjacency for diffusion"""
    deg = np.array(D.diagonal())
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return A_norm

def apply_diffusion(A_norm, X, k, verbose=False):
    """Apply k-step graph diffusion"""
    X_diffused = X.copy()
    for i in range(k):
        X_diffused = A_norm @ X_diffused
        if verbose and (i+1) % max(1, k//4) == 0:
            print(f'    Diffusion step {i+1}/{k}')
    return X_diffused

def compute_restricted_eigenvalues(X, L, D, num_components=0, eps_base=1e-8):
    """
    Compute eigenvalues of restricted eigenproblem
    
    Args:
        X: Feature matrix
        L: Laplacian matrix
        D: Degree matrix
        num_components: Number of connected components (eigenvalues to drop)
        eps_base: Regularization base
    
    Returns:
        eigenvalues_all: All sorted eigenvalues (including near-zero)
        eigenvalues_valid: Valid eigenvalues (after dropping components)
        condition_number: max/min eigenvalue ratio (using valid eigenvalues)
        effective_rank: rank after QR
        num_near_zero: number of near-zero eigenvalues
    """
    # QR decomposition
    Q, R = la.qr(X, mode='economic')
    rank_X = np.linalg.matrix_rank(R, tol=1e-10)
    
    if rank_X < X.shape[1]:
        Q = Q[:, :rank_X]
    
    d_eff = rank_X
    
    # Project Laplacian
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize
    eps = eps_base * np.trace(D_r) / d_eff
    D_r = D_r + eps * np.eye(d_eff)
    
    # Solve
    eigenvalues_all, _ = la.eigh(L_r, D_r)
    eigenvalues_all = np.sort(eigenvalues_all)
    
    # Count near-zero eigenvalues (likely from connected components)
    ZERO_THRESHOLD = 1e-8
    num_near_zero = np.sum(eigenvalues_all < ZERO_THRESHOLD)
    
    # Drop near-zero eigenvalues (connected component eigenvalues)
    # Use the maximum of: detected components OR computed near-zero eigenvalues
    num_to_drop = max(num_components, num_near_zero)
    
    if num_to_drop > 0:
        eigenvalues_valid = eigenvalues_all[num_to_drop:]
    else:
        eigenvalues_valid = eigenvalues_all
    
    # Condition number (using valid eigenvalues only)
    if len(eigenvalues_valid) > 0 and eigenvalues_valid[-1] > ZERO_THRESHOLD:
        condition = eigenvalues_valid[-1] / eigenvalues_valid[0]
    else:
        condition = np.inf
    
    return eigenvalues_all, eigenvalues_valid, condition, d_eff, num_near_zero

def compute_eigenvalue_statistics(eigenvalues):
    """
    Compute comprehensive statistics on eigenvalue distribution
    
    Args:
        eigenvalues: VALID eigenvalues (near-zero already removed)
    
    Note: This function expects eigenvalues that have already been cleaned
          (near-zero/component eigenvalues removed)
    """
    eigs = eigenvalues  # Assume already filtered
    
    if len(eigs) == 0:
        return {
            'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0,
            'range': 0, 'iqr': 0, 'skewness': 0, 'gap_ratio': 0,
            'num_valid': 0, 'condition': np.inf
        }
    
    stats = {
        'mean': float(np.mean(eigs)),
        'median': float(np.median(eigs)),
        'std': float(np.std(eigs)),
        'min': float(np.min(eigs)),
        'max': float(np.max(eigs)),
        'range': float(np.max(eigs) - np.min(eigs)),
        'iqr': float(np.percentile(eigs, 75) - np.percentile(eigs, 25)),
        'num_valid': int(len(eigs)),
        'condition': float(np.max(eigs) / np.min(eigs)) if np.min(eigs) > 0 else np.inf
    }
    
    # Skewness (measure of clustering)
    if stats['std'] > 0:
        stats['skewness'] = float(np.mean(((eigs - stats['mean']) / stats['std'])**3))
    else:
        stats['skewness'] = 0
    
    # Gap ratio (largest gap / median gap)
    if len(eigs) > 1:
        gaps = np.diff(eigs)
        if len(gaps) > 0 and np.median(gaps) > 0:
            stats['gap_ratio'] = float(np.max(gaps) / np.median(gaps))
        else:
            stats['gap_ratio'] = 0
    else:
        stats['gap_ratio'] = 0
    
    return stats

# ============================================================================
# Main Analysis Loop
# ============================================================================

all_results = {}

for dataset_name in DATASETS:
    print(f'\n{"="*80}')
    print(f'ANALYZING: {dataset_name.upper()}')
    print(f'{"="*80}')
    
    try:
        # Load dataset
        print('[1/5] Loading dataset...')
        data_tuple = load_dataset(dataset_name)
        edge_index, X_raw, labels, num_nodes, num_classes, train_idx, val_idx, test_idx = data_tuple
        
        if X_raw is None:
            print(f'⚠️  Skipping {dataset_name}: no node features')
            continue
        
        if torch.is_tensor(X_raw):
            X_raw = X_raw.numpy()
        
        X = X_raw.astype(np.float64)
        
        print(f'  Nodes: {num_nodes:,}')
        print(f'  Features: {X.shape[1]}')
        print(f'  Classes: {num_classes}')
        
        # Build graph matrices
        print('[2/5] Building graph matrices...')
        adj, L, D = build_graph_matrices(edge_index, num_nodes)
        A_norm = compute_normalized_adjacency(adj, D)
        
        # Count connected components
        print('[2.5/5] Detecting connected components...')
        G = nx.from_scipy_sparse_array(adj)
        num_components = nx.number_connected_components(G)
        print(f'  Number of connected components: {num_components}')
        if num_components > 1:
            print(f'  ⚠️  Graph is disconnected! Will drop {num_components} eigenvalues.')
        
        # Store results for this dataset
        dataset_results = {
            'metadata': {
                'num_nodes': num_nodes,
                'num_features': X.shape[1],
                'num_classes': num_classes,
                'num_components': num_components
            },
            'baseline': {},
            'diffused': {}
        }
        
        # Analyze baseline (no diffusion)
        print('[3/5] Analyzing baseline restricted eigenproblem...')
        eigs_all, eigs_valid, cond_baseline, rank_baseline, num_near_zero = \
            compute_restricted_eigenvalues(X, L, D, num_components)
        stats_baseline = compute_eigenvalue_statistics(eigs_valid)
        
        dataset_results['baseline'] = {
            'eigenvalues_all': eigs_all.tolist(),
            'eigenvalues_valid': eigs_valid.tolist(),
            'num_near_zero': int(num_near_zero),
            'num_dropped': max(num_components, num_near_zero),
            'condition': float(cond_baseline),
            'rank': int(rank_baseline),
            'statistics': stats_baseline
        }
        
        print(f'  Total eigenvalues computed: {len(eigs_all)}')
        print(f'  Near-zero eigenvalues: {num_near_zero}')
        print(f'  Components: {num_components}')
        print(f'  Eigenvalues dropped: {max(num_components, num_near_zero)}')
        print(f'  Valid eigenvalues: {len(eigs_valid)}')
        print(f'  Baseline condition number: {cond_baseline:.2e}')
        print(f'  Eigenvalue range (valid): [{stats_baseline["min"]:.6f}, {stats_baseline["max"]:.6f}]')
        print(f'  Eigenvalue spread (std): {stats_baseline["std"]:.6f}')
        
        # Analyze with diffusion
        print('[4/5] Analyzing with diffusion...')
        for k in K_VALUES:
            print(f'  k={k}:')
            X_diffused = apply_diffusion(A_norm, X, k, verbose=False)
            eigs_all_d, eigs_valid_d, cond_diffused, rank_diffused, num_near_zero_d = \
                compute_restricted_eigenvalues(X_diffused, L, D, num_components)
            stats_diffused = compute_eigenvalue_statistics(eigs_valid_d)
            
            dataset_results['diffused'][k] = {
                'eigenvalues_all': eigs_all_d.tolist(),
                'eigenvalues_valid': eigs_valid_d.tolist(),
                'num_near_zero': int(num_near_zero_d),
                'num_dropped': max(num_components, num_near_zero_d),
                'condition': float(cond_diffused),
                'rank': int(rank_diffused),
                'statistics': stats_diffused
            }
            
            # Compare to baseline
            cond_change = (cond_diffused - cond_baseline) / cond_baseline * 100
            spread_change = (stats_diffused['std'] - stats_baseline['std']) / stats_baseline['std'] * 100
            
            print(f'    Valid eigenvalues: {len(eigs_valid_d)}')
            print(f'    Condition: {cond_diffused:.2e} ({cond_change:+.1f}%)')
            print(f'    Spread: {stats_diffused["std"]:.6f} ({spread_change:+.1f}%)')
        
        all_results[dataset_name] = dataset_results
        
        # Create visualizations
        print('[5/5] Creating visualizations...')
        
        # Get valid eigenvalues for plotting
        eigs_baseline_plot = np.array(dataset_results['baseline']['eigenvalues_valid'])
        
        # Figure 1: Eigenvalue Spectra (VALID eigenvalues only)
        fig, axes = plt.subplots(1, len(K_VALUES)+1, figsize=(4*(len(K_VALUES)+1), 4))
        
        # Baseline
        ax = axes[0]
        ax.plot(eigs_baseline_plot, 'o-', markersize=3, linewidth=1, alpha=0.7)
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        num_dropped = dataset_results['baseline']['num_dropped']
        title_baseline = f'Baseline\nκ={cond_baseline:.1e}'
        if num_dropped > 0:
            title_baseline += f'\n({num_dropped} dropped)'
        ax.set_title(title_baseline)
        ax.grid(True, alpha=0.3)
        
        # Diffused
        for i, k in enumerate(K_VALUES):
            ax = axes[i+1]
            eigs = np.array(dataset_results['diffused'][k]['eigenvalues_valid'])
            cond = dataset_results['diffused'][k]['condition']
            num_dropped_k = dataset_results['diffused'][k]['num_dropped']
            ax.plot(eigs, 'o-', markersize=3, linewidth=1, alpha=0.7, color='C1')
            ax.set_xlabel('Index')
            title_k = f'Diffused k={k}\nκ={cond:.1e}'
            if num_dropped_k > 0:
                title_k += f'\n({num_dropped_k} dropped)'
            ax.set_title(title_k)
            ax.grid(True, alpha=0.3)
        
        suptitle = f'{dataset_name}: Eigenvalue Spectra'
        if num_components > 1:
            suptitle += f' ({num_components} components)'
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/{dataset_name}_spectra.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top left: Baseline histogram
        ax = axes[0, 0]
        if len(eigs_baseline_plot) > 0:
            ax.hist(eigs_baseline_plot, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Eigenvalue')
        ax.set_ylabel('Count')
        title = 'Baseline Distribution'
        if num_dropped > 0:
            title += f' ({num_dropped} near-zero dropped)'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Top right: Best diffused histogram
        ax = axes[0, 1]
        best_k = K_VALUES[-1]
        eigs_diffused_plot = np.array(dataset_results['diffused'][best_k]['eigenvalues_valid'])
        num_dropped_best = dataset_results['diffused'][best_k]['num_dropped']
        if len(eigs_diffused_plot) > 0:
            ax.hist(eigs_diffused_plot, bins=50, alpha=0.7, edgecolor='black', color='C1')
        ax.set_xlabel('Eigenvalue')
        title = f'Diffused k={best_k} Distribution'
        if num_dropped_best > 0:
            title += f' ({num_dropped_best} dropped)'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Bottom left: Log-scale spectra comparison
        ax = axes[1, 0]
        ax.semilogy(eigs_baseline_plot, 'o-', markersize=3, linewidth=1, alpha=0.7, label='Baseline')
        for k in K_VALUES:
            eigs = np.array(dataset_results['diffused'][k]['eigenvalues_valid'])
            ax.semilogy(eigs, 'o-', markersize=3, linewidth=1, alpha=0.5, label=f'k={k}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue (log scale)')
        ax.set_title('Spectra Comparison (Log Scale, valid eigenvalues)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Condition number evolution
        ax = axes[1, 1]
        cond_nums = [cond_baseline] + [dataset_results['diffused'][k]['condition'] for k in K_VALUES]
        k_vals = [0] + K_VALUES
        ax.plot(k_vals, cond_nums, 'o-', markersize=8, linewidth=2)
        ax.set_xlabel('Diffusion Steps (k)')
        ax.set_ylabel('Condition Number')
        ax.set_title('Condition Number vs Diffusion')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        suptitle = f'{dataset_name}: Eigenvalue Analysis'
        if num_components > 1:
            suptitle += f' ({num_components} components)'
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/{dataset_name}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Saved plots for {dataset_name}')
        
    except Exception as e:
        print(f'❌ Error processing {dataset_name}: {e}')
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# Cross-Dataset Analysis
# ============================================================================

print(f'\n{"="*80}')
print('CROSS-DATASET ANALYSIS')
print(f'{"="*80}')

if len(all_results) > 0:
    # Create summary table
    summary_data = []
    
    for dataset_name, results in all_results.items():
        row = {
            'Dataset': dataset_name,
            'Nodes': results['metadata']['num_nodes'],
            'Features': results['metadata']['num_features'],
            'Components': results['metadata']['num_components'],
            'Dropped_Eigs': results['baseline']['num_dropped'],
            'Valid_Eigs': len(results['baseline']['eigenvalues_valid']),
            'Baseline_Cond': results['baseline']['condition'],
            'Baseline_Spread': results['baseline']['statistics']['std'],
            'Baseline_Skewness': results['baseline']['statistics']['skewness'],
        }
        
        # Add diffusion results
        for k in K_VALUES:
            if k in results['diffused']:
                row[f'k{k}_Cond'] = results['diffused'][k]['condition']
                row[f'k{k}_Spread'] = results['diffused'][k]['statistics']['std']
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Save summary table
    df.to_csv(f'{OUTPUT_DIR}/data/summary_table.csv', index=False)
    print(f'✓ Saved summary table')
    
    # Print summary
    print('\nCONDITION NUMBERS (Lower = Better Conditioned):')
    print('-'*80)
    print(f'{"Dataset":<20s} {"Components":>5s} {"Dropped":>8s} {"Valid":>6s} {"Condition":>12s}')
    print('-'*80)
    for _, row in df.iterrows():
        print(f"{row['Dataset']:20s} {row['Components']:>5d} {row['Dropped_Eigs']:>8d} "
              f"{row['Valid_Eigs']:>6d} {row['Baseline_Cond']:>12.2e}")
    print('-'*80)
    
    # Warn about highly disconnected graphs
    high_component_datasets = df[df['Components'] > 10]['Dataset'].tolist()
    if high_component_datasets:
        print(f'\n⚠️  WARNING: Highly disconnected graphs detected:')
        for ds in high_component_datasets:
            n_comp = df[df['Dataset']==ds]['Components'].iloc[0]
            print(f'    {ds}: {n_comp} components')
        print(f'   These datasets may have unreliable eigenvalue statistics.')
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Condition numbers
    ax = axes[0, 0]
    datasets = df['Dataset'].tolist()
    baseline_conds = df['Baseline_Cond'].tolist()
    x_pos = np.arange(len(datasets))
    
    colors = ['green' if c < 100 else 'orange' if c < 1000 else 'red' for c in baseline_conds]
    ax.bar(x_pos, baseline_conds, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_yscale('log')
    ax.set_title('Baseline Condition Numbers')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Well-conditioned (<100)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate (100-1000)'),
        Patch(facecolor='red', alpha=0.7, label='Ill-conditioned (>1000)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # Plot 2: Spread (std of eigenvalues)
    ax = axes[0, 1]
    spreads = df['Baseline_Spread'].tolist()
    ax.bar(x_pos, spreads, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Eigenvalue Spread (std)')
    ax.set_title('Baseline Eigenvalue Spread')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Diffusion effect on condition
    ax = axes[1, 0]
    for i, dataset in enumerate(datasets):
        conds = [df.loc[i, 'Baseline_Cond']]
        for k in K_VALUES:
            col = f'k{k}_Cond'
            if col in df.columns:
                conds.append(df.loc[i, col])
        k_vals = [0] + K_VALUES[:len(conds)-1]
        ax.plot(k_vals, conds, 'o-', label=dataset, markersize=6, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Diffusion Steps (k)')
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_yscale('log')
    ax.set_title('Condition Number Evolution with Diffusion')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Feature dimension vs condition
    ax = axes[1, 1]
    dims = df['Features'].tolist()
    ax.scatter(dims, baseline_conds, s=100, alpha=0.7, edgecolor='black')
    for i, dataset in enumerate(datasets):
        ax.annotate(dataset, (dims[i], baseline_conds[i]), 
                   fontsize=8, ha='center', va='bottom')
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Feature Dimension vs Condition')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Dataset Eigenvalue Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/cross_dataset_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved cross-dataset summary')

# Save all results
results_file = f'{OUTPUT_DIR}/data/eigenvalue_analysis_complete.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n{"="*80}')
print('ANALYSIS COMPLETE')
print(f'{"="*80}')
print(f'Results saved to: {OUTPUT_DIR}')
print(f'Plots: {OUTPUT_DIR}/plots/')
print(f'Data: {OUTPUT_DIR}/data/')
print(f'{"="*80}')