"""
Investigation: Geometry Analysis of X_diffused vs U
====================================================

Key Question: What geometric properties differ between X and U that cause
              linear classifiers to fail catastrophically on U?

Analyses:
1. Condition Number - How "stretched" is the data?
2. Column Scales - Do dimensions have different magnitudes?
3. Gram Matrix Structure - How do features relate to each other?
4. Class Geometry - How are classes distributed in each space?
5. Effective Dimensionality - How many dimensions matter?

Goal: Understand WHAT is broken and HOW to fix it.

Usage:
    python investigation_geometry.py
    python investigation_geometry.py --datasets cora citeseer pubmed
"""

import os
import json
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Geometry Analysis')
parser.add_argument('--datasets', nargs='+', type=str, 
                   default=['cora', 'citeseer', 'pubmed', 'wikics', 
                           'amazon-computers', 'amazon-photo',
                           'coauthor-cs', 'coauthor-physics', 'ogbn-arxiv'],
                   help='Datasets to analyze')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 10],
                   help='Diffusion steps to test')
parser.add_argument('--output_dir', type=str, default='results/geometry_analysis',
                   help='Output directory')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

print('='*80)
print('GEOMETRY ANALYSIS: X_diffused vs U')
print('='*80)

# ============================================================================
# Geometry Analysis Functions
# ============================================================================

def analyze_condition_number(X, U, name):
    """
    Condition number = σ_max / σ_min (ratio of largest to smallest singular value)
    
    High condition number means:
    - Data is "stretched" along some directions
    - Numerical instability
    - Different effective learning rates per dimension
    """
    print(f'\n--- Condition Number Analysis ({name}) ---')
    
    # Compute singular values
    try:
        sv_X = la.svdvals(X)
        sv_U = la.svdvals(U)
        
        cond_X = sv_X[0] / sv_X[-1] if sv_X[-1] > 1e-10 else np.inf
        cond_U = sv_U[0] / sv_U[-1] if sv_U[-1] > 1e-10 else np.inf
        
        print(f'  X_diffused:')
        print(f'    σ_max = {sv_X[0]:.4f}, σ_min = {sv_X[-1]:.6f}')
        print(f'    Condition number = {cond_X:.2f}')
        
        print(f'  U:')
        print(f'    σ_max = {sv_U[0]:.4f}, σ_min = {sv_U[-1]:.6f}')
        print(f'    Condition number = {cond_U:.2f}')
        
        print(f'  Ratio (U/X): {cond_U/cond_X:.2f}x')
        
        return {
            'X_cond': float(cond_X),
            'U_cond': float(cond_U),
            'X_sv_max': float(sv_X[0]),
            'X_sv_min': float(sv_X[-1]),
            'U_sv_max': float(sv_U[0]),
            'U_sv_min': float(sv_U[-1]),
            'sv_X': sv_X.tolist(),
            'sv_U': sv_U.tolist(),
        }
    except Exception as e:
        print(f'  Error: {e}')
        return None


def analyze_column_scales(X, U, name):
    """
    Column scales = L2 norm of each column (feature dimension)
    
    If scales vary wildly:
    - Some features dominate others
    - Linear classifier weights must compensate
    - Gradient descent has different effective learning rates
    """
    print(f'\n--- Column Scale Analysis ({name}) ---')
    
    X_col_norms = np.linalg.norm(X, axis=0)
    U_col_norms = np.linalg.norm(U, axis=0)
    
    print(f'  X_diffused column norms:')
    print(f'    Min: {X_col_norms.min():.6f}')
    print(f'    Max: {X_col_norms.max():.6f}')
    print(f'    Mean: {X_col_norms.mean():.6f}')
    print(f'    Std: {X_col_norms.std():.6f}')
    print(f'    Ratio (max/min): {X_col_norms.max()/X_col_norms.min():.2f}')
    
    print(f'  U column norms:')
    print(f'    Min: {U_col_norms.min():.6f}')
    print(f'    Max: {U_col_norms.max():.6f}')
    print(f'    Mean: {U_col_norms.mean():.6f}')
    print(f'    Std: {U_col_norms.std():.6f}')
    print(f'    Ratio (max/min): {U_col_norms.max()/U_col_norms.min():.2f}')
    
    return {
        'X_col_min': float(X_col_norms.min()),
        'X_col_max': float(X_col_norms.max()),
        'X_col_mean': float(X_col_norms.mean()),
        'X_col_ratio': float(X_col_norms.max()/X_col_norms.min()),
        'U_col_min': float(U_col_norms.min()),
        'U_col_max': float(U_col_norms.max()),
        'U_col_mean': float(U_col_norms.mean()),
        'U_col_ratio': float(U_col_norms.max()/U_col_norms.min()),
        'X_col_norms': X_col_norms.tolist(),
        'U_col_norms': U_col_norms.tolist(),
    }


def analyze_row_norms(X, U, labels, name):
    """
    Row norms = L2 norm of each sample
    
    For classification:
    - If row norms vary by class → magnitude carries class info
    - If row norms are similar → magnitude is noise
    """
    print(f'\n--- Row Norm Analysis ({name}) ---')
    
    X_row_norms = np.linalg.norm(X, axis=1)
    U_row_norms = np.linalg.norm(U, axis=1)
    
    print(f'  X_diffused row norms:')
    print(f'    Min: {X_row_norms.min():.4f}, Max: {X_row_norms.max():.4f}')
    print(f'    Mean: {X_row_norms.mean():.4f}, Std: {X_row_norms.std():.4f}')
    print(f'    Ratio: {X_row_norms.max()/X_row_norms.min():.2f}')
    
    print(f'  U row norms:')
    print(f'    Min: {U_row_norms.min():.4f}, Max: {U_row_norms.max():.4f}')
    print(f'    Mean: {U_row_norms.mean():.4f}, Std: {U_row_norms.std():.4f}')
    print(f'    Ratio: {U_row_norms.max()/U_row_norms.min():.2f}')
    
    # Per-class analysis
    unique_classes = np.unique(labels)
    print(f'\n  Per-class mean row norms:')
    print(f'    {"Class":<8} {"X_mean":<12} {"U_mean":<12}')
    
    for c in unique_classes[:5]:  # First 5 classes
        mask = labels == c
        X_class_mean = X_row_norms[mask].mean()
        U_class_mean = U_row_norms[mask].mean()
        print(f'    {c:<8} {X_class_mean:<12.4f} {U_class_mean:<12.4f}')
    
    return {
        'X_row_min': float(X_row_norms.min()),
        'X_row_max': float(X_row_norms.max()),
        'X_row_mean': float(X_row_norms.mean()),
        'X_row_ratio': float(X_row_norms.max()/X_row_norms.min()),
        'U_row_min': float(U_row_norms.min()),
        'U_row_max': float(U_row_norms.max()),
        'U_row_mean': float(U_row_norms.mean()),
        'U_row_ratio': float(U_row_norms.max()/U_row_norms.min()),
    }


def analyze_gram_matrix(X, U, name):
    """
    Gram matrix G = X^T X (or U^T U)
    
    The Gram matrix captures:
    - Feature correlations (off-diagonal)
    - Feature variances (diagonal)
    - Overall "shape" of the data cloud
    """
    print(f'\n--- Gram Matrix Analysis ({name}) ---')
    
    # Compute Gram matrices
    G_X = X.T @ X
    G_U = U.T @ U
    
    # Normalize by number of samples for comparison
    n = X.shape[0]
    G_X_norm = G_X / n
    G_U_norm = G_U / n
    
    # Diagonal (variances)
    diag_X = np.diag(G_X_norm)
    diag_U = np.diag(G_U_norm)
    
    print(f'  G_X diagonal (feature variances):')
    print(f'    Min: {diag_X.min():.6f}, Max: {diag_X.max():.6f}')
    print(f'    Ratio: {diag_X.max()/(diag_X.min() + 1e-10):.2f}')
    
    print(f'  G_U diagonal (feature variances):')
    print(f'    Min: {diag_U.min():.6f}, Max: {diag_U.max():.6f}')
    print(f'    Ratio: {diag_U.max()/(diag_U.min() + 1e-10):.2f}')
    
    # Off-diagonal (correlations) - use correct indices for each matrix
    triu_idx_X = np.triu_indices(G_X_norm.shape[0], k=1)
    triu_idx_U = np.triu_indices(G_U_norm.shape[0], k=1)
    
    off_diag_X = G_X_norm[triu_idx_X]
    off_diag_U = G_U_norm[triu_idx_U]
    
    print(f'  G_X off-diagonal (feature correlations):')
    print(f'    Mean: {off_diag_X.mean():.6f}, Std: {off_diag_X.std():.6f}')
    
    print(f'  G_U off-diagonal (feature correlations):')
    print(f'    Mean: {off_diag_U.mean():.6f}, Std: {off_diag_U.std():.6f}')
    
    # Eigenvalues of Gram matrices
    eig_X = la.eigvalsh(G_X_norm)[::-1]  # Descending
    eig_U = la.eigvalsh(G_U_norm)[::-1]
    
    print(f'  G_X eigenvalues (top 5): {eig_X[:5]}')
    print(f'  G_U eigenvalues (top 5): {eig_U[:5]}')
    
    # Effective rank (how many eigenvalues matter)
    def effective_rank(eigenvalues):
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0
        p = eigenvalues / eigenvalues.sum()
        return np.exp(-np.sum(p * np.log(p + 1e-10)))
    
    eff_rank_X = effective_rank(eig_X)
    eff_rank_U = effective_rank(eig_U)
    
    print(f'  Effective rank:')
    print(f'    X: {eff_rank_X:.2f} / {len(diag_X)}')
    print(f'    U: {eff_rank_U:.2f} / {len(diag_U)}')
    
    return {
        'X_var_min': float(diag_X.min()),
        'X_var_max': float(diag_X.max()),
        'X_var_ratio': float(diag_X.max()/(diag_X.min() + 1e-10)),
        'U_var_min': float(diag_U.min()),
        'U_var_max': float(diag_U.max()),
        'U_var_ratio': float(diag_U.max()/(diag_U.min() + 1e-10)),
        'X_eff_rank': float(eff_rank_X),
        'U_eff_rank': float(eff_rank_U),
        'G_X_eig': eig_X.tolist(),
        'G_U_eig': eig_U.tolist(),
    }


def analyze_class_geometry(X, U, labels, name):
    """
    Class geometry: How are classes distributed in each space?
    
    Measures:
    - Between-class distance (separability)
    - Within-class variance (compactness)
    - Fisher's criterion (separability / compactness)
    """
    print(f'\n--- Class Geometry Analysis ({name}) ---')
    
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    def compute_class_stats(data, labels):
        """Compute class means and within-class scatter"""
        class_means = []
        within_scatter = 0
        
        for c in unique_classes:
            mask = labels == c
            class_data = data[mask]
            mean = class_data.mean(axis=0)
            class_means.append(mean)
            
            # Within-class scatter
            centered = class_data - mean
            within_scatter += (centered ** 2).sum()
        
        class_means = np.array(class_means)
        global_mean = data.mean(axis=0)
        
        # Between-class scatter
        between_scatter = 0
        for i, c in enumerate(unique_classes):
            n_c = (labels == c).sum()
            diff = class_means[i] - global_mean
            between_scatter += n_c * (diff ** 2).sum()
        
        return class_means, within_scatter, between_scatter
    
    X_means, X_within, X_between = compute_class_stats(X, labels)
    U_means, U_within, U_between = compute_class_stats(U, labels)
    
    # Fisher criterion: between / within (higher = more separable)
    fisher_X = X_between / (X_within + 1e-10)
    fisher_U = U_between / (U_within + 1e-10)
    
    print(f'  X_diffused:')
    print(f'    Within-class scatter:  {X_within:.4f}')
    print(f'    Between-class scatter: {X_between:.4f}')
    print(f'    Fisher criterion:      {fisher_X:.6f}')
    
    print(f'  U:')
    print(f'    Within-class scatter:  {U_within:.4f}')
    print(f'    Between-class scatter: {U_between:.4f}')
    print(f'    Fisher criterion:      {fisher_U:.6f}')
    
    print(f'  Fisher ratio (X/U): {fisher_X/fisher_U:.2f}')
    
    # Pairwise class distances
    def mean_pairwise_distance(class_means):
        n = len(class_means)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total += np.linalg.norm(class_means[i] - class_means[j])
                count += 1
        return total / count
    
    X_class_dist = mean_pairwise_distance(X_means)
    U_class_dist = mean_pairwise_distance(U_means)
    
    print(f'  Mean pairwise class distance:')
    print(f'    X: {X_class_dist:.4f}')
    print(f'    U: {U_class_dist:.4f}')
    print(f'    Ratio (X/U): {X_class_dist/U_class_dist:.2f}')
    
    return {
        'X_within': float(X_within),
        'X_between': float(X_between),
        'X_fisher': float(fisher_X),
        'U_within': float(U_within),
        'U_between': float(U_between),
        'U_fisher': float(fisher_U),
        'fisher_ratio': float(fisher_X/fisher_U),
        'X_class_dist': float(X_class_dist),
        'U_class_dist': float(U_class_dist),
    }


def analyze_after_normalization(X, U, labels, name):
    """
    What happens after different normalizations?
    
    This tells us what transformation "fixes" the geometry.
    """
    print(f'\n--- Effect of Normalization ({name}) ---')
    
    results = {}
    
    # 1. StandardScaler (zero mean, unit variance per feature)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    U_std = scaler.fit_transform(U)
    
    cond_X_std = np.linalg.cond(X_std)
    cond_U_std = np.linalg.cond(U_std)
    
    print(f'  After StandardScaler:')
    print(f'    X condition: {cond_X_std:.2f}')
    print(f'    U condition: {cond_U_std:.2f}')
    
    results['X_std_cond'] = float(cond_X_std)
    results['U_std_cond'] = float(cond_U_std)
    
    # 2. Row normalization (unit L2 norm per sample)
    X_row = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    U_row = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
    
    cond_X_row = np.linalg.cond(X_row)
    cond_U_row = np.linalg.cond(U_row)
    
    print(f'  After Row Normalization:')
    print(f'    X condition: {cond_X_row:.2f}')
    print(f'    U condition: {cond_U_row:.2f}')
    
    results['X_row_cond'] = float(cond_X_row)
    results['U_row_cond'] = float(cond_U_row)
    
    # 3. Column normalization (unit L2 norm per feature)
    X_col = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-10)
    U_col = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    
    cond_X_col = np.linalg.cond(X_col)
    cond_U_col = np.linalg.cond(U_col)
    
    print(f'  After Column Normalization:')
    print(f'    X condition: {cond_X_col:.2f}')
    print(f'    U condition: {cond_U_col:.2f}')
    
    results['X_col_cond'] = float(cond_X_col)
    results['U_col_cond'] = float(cond_U_col)
    
    # 4. Both (column then row)
    X_both = X_col / (np.linalg.norm(X_col, axis=1, keepdims=True) + 1e-10)
    U_both = U_col / (np.linalg.norm(U_col, axis=1, keepdims=True) + 1e-10)
    
    cond_X_both = np.linalg.cond(X_both)
    cond_U_both = np.linalg.cond(U_both)
    
    print(f'  After Column + Row Normalization:')
    print(f'    X condition: {cond_X_both:.2f}')
    print(f'    U condition: {cond_U_both:.2f}')
    
    results['X_both_cond'] = float(cond_X_both)
    results['U_both_cond'] = float(cond_U_both)
    
    return results


def analyze_eigenvalue_scaling(U, eigenvalues, name):
    """
    How do the restricted eigenvalues relate to U's column scales?
    """
    print(f'\n--- Eigenvalue-Scale Relationship ({name}) ---')
    
    U_col_norms = np.linalg.norm(U, axis=0)
    
    # Correlation between eigenvalue and column norm
    if len(eigenvalues) == len(U_col_norms):
        corr = np.corrcoef(eigenvalues, U_col_norms)[0, 1]
        print(f'  Correlation(eigenvalue, column_norm): {corr:.4f}')
        
        print(f'\n  Sample eigenvalue vs column norm:')
        print(f'    {"Idx":<6} {"Eigenvalue":<15} {"Col Norm":<15}')
        for i in [0, 1, 2, len(eigenvalues)//2, -3, -2, -1]:
            idx = i if i >= 0 else len(eigenvalues) + i
            print(f'    {idx:<6} {eigenvalues[idx]:<15.6f} {U_col_norms[idx]:<15.6f}')
        
        return {
            'eigenvalue_colnorm_corr': float(corr),
            'eigenvalues': eigenvalues.tolist(),
        }
    else:
        print(f'  Dimension mismatch: eigenvalues={len(eigenvalues)}, U_cols={len(U_col_norms)}')
        return None


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_singular_value_spectrum(sv_X, sv_U, name, output_path):
    """Plot singular value spectrum comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1 = axes[0]
    ax1.plot(sv_X, 'b-', label='X_diffused', linewidth=2)
    ax1.plot(sv_U, 'r-', label='U', linewidth=2)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title(f'{name}: Singular Value Spectrum', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Log scale
    ax2 = axes[1]
    ax2.semilogy(sv_X, 'b-', label='X_diffused', linewidth=2)
    ax2.semilogy(sv_U, 'r-', label='U', linewidth=2)
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_ylabel('Singular Value (log)', fontsize=12)
    ax2.set_title(f'{name}: Singular Value Spectrum (log)', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_column_norms(X_norms, U_norms, name, output_path):
    """Plot column norm distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(X_norms, bins=50, alpha=0.7, label='X_diffused', color='blue')
    ax1.hist(U_norms, bins=50, alpha=0.7, label='U', color='red')
    ax1.set_xlabel('Column Norm', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{name}: Column Norm Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sorted
    ax2 = axes[1]
    ax2.plot(np.sort(X_norms)[::-1], 'b-', label='X_diffused', linewidth=2)
    ax2.plot(np.sort(U_norms)[::-1], 'r-', label='U', linewidth=2)
    ax2.set_xlabel('Rank', fontsize=12)
    ax2.set_ylabel('Column Norm', fontsize=12)
    ax2.set_title(f'{name}: Sorted Column Norms', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gram_eigenvalues(eig_X, eig_U, name, output_path):
    """Plot Gram matrix eigenvalue spectrum"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear
    ax1 = axes[0]
    ax1.plot(eig_X[:50], 'b-', label='G_X', linewidth=2)
    ax1.plot(eig_U[:50], 'r-', label='G_U', linewidth=2)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title(f'{name}: Gram Matrix Eigenvalues (top 50)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Log
    ax2 = axes[1]
    ax2.semilogy(np.array(eig_X) + 1e-10, 'b-', label='G_X', linewidth=2)
    ax2.semilogy(np.array(eig_U) + 1e-10, 'r-', label='G_U', linewidth=2)
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue (log)', fontsize=12)
    ax2.set_title(f'{name}: Gram Matrix Eigenvalues (log)', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_dataset(dataset_name, k_diffusion):
    """Run full geometry analysis for a dataset"""
    print(f'\n{"="*70}')
    print(f'ANALYZING: {dataset_name} (k={k_diffusion})')
    print(f'{"="*70}')
    
    results = {'dataset': dataset_name, 'k': k_diffusion}
    
    try:
        # Load and preprocess
        print('\n[1/3] Loading dataset...')
        (edge_index, features, labels, num_nodes, num_classes,
         train_idx, val_idx, test_idx) = load_dataset(dataset_name)
        
        print('\n[2/3] Building graph and extracting LCC...')
        adj, L, D = build_graph_matrices(edge_index, num_nodes)
        lcc_mask = get_largest_connected_component_nx(adj)
        
        split_idx = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx} if train_idx is not None else None
        adj, features, labels, split_idx = extract_subgraph(adj, features, labels, lcc_mask, split_idx)
        
        adj_coo = adj.tocoo()
        edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
        adj, L, D = build_graph_matrices(edge_index_lcc, adj.shape[0])
        
        num_nodes = features.shape[0]
        print(f'  LCC nodes: {num_nodes}, Classes: {num_classes}')
        
        print('\n[3/3] Computing X_diffused and U...')
        A_sgc = compute_sgc_normalized_adjacency(adj)
        features_dense = features.toarray() if sp.issparse(features) else features
        X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diffusion)
        
        U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(X_diffused, L, D, num_components=0)
        print(f'  X shape: {X_diffused.shape}, U shape: {U.shape}')
        
        results['num_nodes'] = num_nodes
        results['num_classes'] = num_classes
        results['X_shape'] = list(X_diffused.shape)
        results['U_shape'] = list(U.shape)
        
        # Run all analyses
        name = f'{dataset_name}_k{k_diffusion}'
        
        cond_results = analyze_condition_number(X_diffused, U, name)
        if cond_results:
            results['condition'] = cond_results
        
        col_results = analyze_column_scales(X_diffused, U, name)
        results['column_scales'] = col_results
        
        row_results = analyze_row_norms(X_diffused, U, labels, name)
        results['row_norms'] = row_results
        
        gram_results = analyze_gram_matrix(X_diffused, U, name)
        results['gram_matrix'] = gram_results
        
        class_results = analyze_class_geometry(X_diffused, U, labels, name)
        results['class_geometry'] = class_results
        
        norm_results = analyze_after_normalization(X_diffused, U, labels, name)
        results['normalization'] = norm_results
        
        eig_results = analyze_eigenvalue_scaling(U, eigenvalues, name)
        if eig_results:
            results['eigenvalue_scaling'] = eig_results
        
        # Generate plots
        if cond_results and 'sv_X' in cond_results:
            plot_singular_value_spectrum(
                np.array(cond_results['sv_X']), 
                np.array(cond_results['sv_U']),
                name, f'{OUTPUT_DIR}/plots/{name}_singular_values.png'
            )
        
        plot_column_norms(
            np.array(col_results['X_col_norms']),
            np.array(col_results['U_col_norms']),
            name, f'{OUTPUT_DIR}/plots/{name}_column_norms.png'
        )
        
        if 'G_X_eig' in gram_results:
            plot_gram_eigenvalues(
                gram_results['G_X_eig'],
                gram_results['G_U_eig'],
                name, f'{OUTPUT_DIR}/plots/{name}_gram_eigenvalues.png'
            )
        
        return results
        
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    
    all_results = {}
    
    for dataset_name in args.datasets:
        for k in args.k_diffusion:
            key = f'{dataset_name}_k{k}'
            results = analyze_dataset(dataset_name, k)
            if results:
                all_results[key] = results
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print('\n' + '='*100)
    print('SUMMARY: KEY GEOMETRIC DIFFERENCES')
    print('='*100)
    
    print(f'\n{"Dataset":<20} {"X_cond":<12} {"U_cond":<12} {"Ratio":<10} {"X_col_ratio":<12} {"U_col_ratio":<12}')
    print('-'*80)
    
    for name, res in all_results.items():
        if 'condition' in res:
            x_cond = res['condition']['X_cond']
            u_cond = res['condition']['U_cond']
            ratio = u_cond / x_cond
            x_col = res['column_scales']['X_col_ratio']
            u_col = res['column_scales']['U_col_ratio']
            print(f'{name:<20} {x_cond:<12.2f} {u_cond:<12.2f} {ratio:<10.2f} {x_col:<12.2f} {u_col:<12.2f}')
    
    print('\n' + '='*100)
    print('SUMMARY: CLASS GEOMETRY')
    print('='*100)
    
    print(f'\n{"Dataset":<20} {"X_Fisher":<12} {"U_Fisher":<12} {"Ratio(X/U)":<12} {"X_class_dist":<14} {"U_class_dist":<14}')
    print('-'*90)
    
    for name, res in all_results.items():
        if 'class_geometry' in res:
            cg = res['class_geometry']
            print(f'{name:<20} {cg["X_fisher"]:<12.6f} {cg["U_fisher"]:<12.6f} {cg["fisher_ratio"]:<12.2f} {cg["X_class_dist"]:<14.4f} {cg["U_class_dist"]:<14.4f}')
    
    print('\n' + '='*100)
    print('SUMMARY: EFFECT OF NORMALIZATIONS ON CONDITION NUMBER')
    print('='*100)
    
    print(f'\n{"Dataset":<20} {"Raw_U":<10} {"StdScaler":<10} {"RowNorm":<10} {"ColNorm":<10} {"Both":<10}')
    print('-'*70)
    
    for name, res in all_results.items():
        if 'normalization' in res and 'condition' in res:
            raw = res['condition']['U_cond']
            std = res['normalization']['U_std_cond']
            row = res['normalization']['U_row_cond']
            col = res['normalization']['U_col_cond']
            both = res['normalization']['U_both_cond']
            print(f'{name:<20} {raw:<10.2f} {std:<10.2f} {row:<10.2f} {col:<10.2f} {both:<10.2f}')
    
    # ========================================================================
    # Key Findings
    # ========================================================================
    
    print('\n' + '='*100)
    print('KEY FINDINGS')
    print('='*100)
    
    # Average ratios
    if all_results:
        cond_ratios = [res['condition']['U_cond']/res['condition']['X_cond'] 
                       for res in all_results.values() if 'condition' in res]
        fisher_ratios = [res['class_geometry']['fisher_ratio'] 
                         for res in all_results.values() if 'class_geometry' in res]
        
        print(f'''
1. CONDITION NUMBER:
   - Mean U/X ratio: {np.mean(cond_ratios):.2f}x
   - U is {np.mean(cond_ratios):.0f}x more ill-conditioned than X
   - This causes numerical instability in linear classifiers

2. CLASS SEPARABILITY:
   - Mean Fisher ratio (X/U): {np.mean(fisher_ratios):.2f}
   - Classes are {np.mean(fisher_ratios):.0f}x MORE separable in X than U
   - This directly explains why linear classifiers fail

3. WHAT FIXES IT:
   - Check the normalization table above
   - The normalization that brings U's condition number closest to X's
     is likely the "fix" we need
''')
    
    # Save results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(f'{OUTPUT_DIR}/geometry_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f'\n✓ Results saved: {OUTPUT_DIR}/geometry_results.json')
    
    print('\n' + '='*100)
    print('ANALYSIS COMPLETE')
    print('='*100)