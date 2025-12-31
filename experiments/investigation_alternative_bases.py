"""
Investigation: Alternative Bases for the Same Span
===================================================

Key Question: Is class geometry collapse specific to Rayleigh-Ritz,
              or does ANY orthonormal basis of span(X) collapse classes?

We test multiple bases that all span the same subspace as X_diffused:

1. X_diffused (original) - baseline with good class geometry
2. Q from QR decomposition - Gram-Schmidt orthonormalization  
3. U from SVD (left singular vectors) - PCA basis
4. U_RR from Rayleigh-Ritz - graph Laplacian eigenvectors (current method)
5. Random orthonormal basis - random rotation of Q

All bases span the SAME subspace, but have different coordinate systems.
We measure class geometry (Fisher criterion) for each.

If only Rayleigh-Ritz collapses classes → problem is graph alignment
If ALL orthonormal bases collapse classes → problem is orthonormalization itself

Usage:
    python investigation_alternative_bases.py
"""

import os
import json
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

parser = argparse.ArgumentParser(description='Alternative Bases Investigation')
parser.add_argument('--datasets', nargs='+', type=str, 
                #    default=['cora', 'citeseer', 'pubmed', 'wikics', 
                #            'amazon-computers', 'amazon-photo',
                #            'coauthor-cs', 'coauthor-physics', 'ogbn-arxiv'],
                   default=[ 'ogbn-arxiv'],
                   help='Datasets to analyze')
parser.add_argument('--k_diffusion', type=int, default=2,
                   help='Diffusion steps')
parser.add_argument('--num_random_bases', type=int, default=5,
                   help='Number of random orthonormal bases to test')
parser.add_argument('--output_dir', type=str, default='results/alternative_bases',
                   help='Output directory')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

print('='*80)
print('ALTERNATIVE BASES INVESTIGATION')
print('='*80)
print('Question: Is class collapse specific to Rayleigh-Ritz or all orthonormal bases?')
print('='*80)

# ============================================================================
# Basis Construction Functions
# ============================================================================

def construct_qr_basis(X):
    """
    QR decomposition: X = Q @ R
    Q is orthonormal and spans the same space as X
    
    This is Gram-Schmidt orthonormalization - a "neutral" orthonormal basis
    that doesn't consider the graph structure.
    """
    Q, R = np.linalg.qr(X)
    return Q, "QR (Gram-Schmidt)"


def construct_svd_basis(X):
    """
    SVD: X = U @ S @ V^T
    U contains left singular vectors (orthonormal columns)
    
    This is the PCA basis - directions of maximum variance.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, "SVD (PCA)"


def construct_random_orthonormal_basis(X, seed=42):
    """
    Random orthonormal basis for span(X)
    
    Method: 
    1. Get Q from QR decomposition
    2. Apply random orthogonal rotation: Q_random = Q @ R_random
       where R_random is a random orthogonal matrix
    """
    np.random.seed(seed)
    Q, _ = np.linalg.qr(X)
    d = Q.shape[1]
    
    # Generate random orthogonal matrix via QR of random matrix
    random_matrix = np.random.randn(d, d)
    R_random, _ = np.linalg.qr(random_matrix)
    
    Q_random = Q @ R_random
    return Q_random, f"Random Orthonormal (seed={seed})"


def construct_rayleigh_ritz_basis(X, L, D):
    """
    Rayleigh-Ritz restricted eigenvectors (current method)
    
    These are eigenvectors of the graph Laplacian restricted to span(X).
    """
    U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(X, L, D, num_components=0)
    return U, "Rayleigh-Ritz (Graph Eigenvectors)"


def construct_class_optimal_basis(X, labels):
    """
    Supervised baseline: LDA-like basis that maximizes class separation
    within the constraint of spanning the same space as X.
    
    Method:
    1. Get Q from QR (orthonormal basis for span(X))
    2. Compute class means directly in Q space
    3. Find rotation that aligns principal axes with class-discriminative directions
    """
    Q, _ = np.linalg.qr(X)
    d = Q.shape[1]
    unique_classes = np.unique(labels)
    
    # Compute class means directly in Q coordinates
    # Q[labels == c] gives nodes of class c in Q basis
    class_means_Q = np.array([Q[labels == c].mean(axis=0) for c in unique_classes])
    
    # Find principal directions of class means (directions that separate classes)
    class_means_centered = class_means_Q - class_means_Q.mean(axis=0)
    
    # SVD to find directions of maximum class separation
    U_lda, S_lda, Vt_lda = np.linalg.svd(class_means_centered, full_matrices=False)
    
    # Rotation matrix: columns of Vt_lda.T are the principal directions
    # We rotate Q so its first columns align with class-discriminative directions
    # Q_optimal = Q @ Vt_lda.T would work if Vt_lda was square
    
    # Since Vt_lda is (num_classes, d), we need to extend it to a full rotation
    # For simplicity, just reorder Q's columns by their class-discriminative power
    
    # Alternative: project Q onto class-discriminative subspace first
    # This is a simplified version - just return Q rotated by available directions
    
    num_dirs = min(len(unique_classes) - 1, d)  # At most (C-1) discriminative directions
    
    if num_dirs < d:
        # Extend Vt_lda to a full orthogonal matrix
        # Fill remaining directions with orthogonal complement
        V_partial = Vt_lda.T  # (d, num_classes)
        
        # Use QR to extend to full orthogonal basis
        V_full = np.eye(d)
        V_full[:, :num_dirs] = V_partial[:, :num_dirs]
        V_full, _ = np.linalg.qr(V_full)  # Orthogonalize
    else:
        V_full = Vt_lda.T
    
    Q_optimal = Q @ V_full
    
    return Q_optimal, "Class-Optimal (Supervised)"


# ============================================================================
# Geometry Analysis Functions
# ============================================================================

def compute_fisher_criterion(X, labels):
    """
    Fisher criterion = between-class variance / within-class variance
    Higher = better class separability
    """
    unique_classes = np.unique(labels)
    global_mean = X.mean(axis=0)
    
    within_scatter = 0
    between_scatter = 0
    
    for c in unique_classes:
        mask = labels == c
        class_data = X[mask]
        class_mean = class_data.mean(axis=0)
        n_c = mask.sum()
        
        # Within-class scatter
        centered = class_data - class_mean
        within_scatter += (centered ** 2).sum()
        
        # Between-class scatter  
        diff = class_mean - global_mean
        between_scatter += n_c * (diff ** 2).sum()
    
    fisher = between_scatter / (within_scatter + 1e-10)
    return fisher


def compute_class_centroid_distances(X, labels):
    """
    Mean pairwise distance between class centroids
    """
    unique_classes = np.unique(labels)
    class_means = np.array([X[labels == c].mean(axis=0) for c in unique_classes])
    
    total_dist = 0
    count = 0
    for i in range(len(class_means)):
        for j in range(i+1, len(class_means)):
            total_dist += np.linalg.norm(class_means[i] - class_means[j])
            count += 1
    
    return total_dist / count if count > 0 else 0


def compute_condition_number(X):
    """Condition number = σ_max / σ_min"""
    sv = la.svdvals(X)
    if sv[-1] < 1e-10:
        return np.inf
    return sv[0] / sv[-1]


def test_linear_classifier(X, labels, num_splits=5):
    """
    Test linear classifier accuracy on representation
    """
    accuracies = []
    n = len(labels)
    
    for seed in range(num_splits):
        np.random.seed(seed)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * n)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    return np.mean(accuracies), np.std(accuracies)


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_dataset(dataset_name, k_diffusion, num_random_bases):
    """Analyze all basis types for a dataset"""
    print(f'\n{"="*70}')
    print(f'ANALYZING: {dataset_name} (k={k_diffusion})')
    print(f'{"="*70}')
    
    results = {'dataset': dataset_name, 'k': k_diffusion, 'bases': {}}
    
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
        
        print('\n[3/3] Computing X_diffused...')
        A_sgc = compute_sgc_normalized_adjacency(adj)
        features_dense = features.toarray() if sp.issparse(features) else features
        X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diffusion)
        print(f'  X_diffused shape: {X_diffused.shape}')
        
        # ================================================================
        # Construct all bases
        # ================================================================
        
        bases = {}
        
        # 1. Original X_diffused (not orthonormal, but baseline)
        bases['X_diffused'] = (X_diffused, "X_diffused (Original)")
        
        # 2. QR basis
        Q, name = construct_qr_basis(X_diffused)
        bases['QR'] = (Q, name)
        
        # 3. SVD/PCA basis
        U_svd, name = construct_svd_basis(X_diffused)
        bases['SVD'] = (U_svd, name)
        
        # 4. Rayleigh-Ritz basis
        U_rr, name = construct_rayleigh_ritz_basis(X_diffused, L, D)
        bases['Rayleigh-Ritz'] = (U_rr, name)
        
        # 5. Random orthonormal bases
        for i in range(num_random_bases):
            Q_rand, name = construct_random_orthonormal_basis(X_diffused, seed=i)
            bases[f'Random_{i}'] = (Q_rand, name)
        
        # 6. Class-optimal basis (supervised, for comparison)
        Q_opt, name = construct_class_optimal_basis(X_diffused, labels)
        bases['Class-Optimal'] = (Q_opt, name)
        
        # ================================================================
        # Analyze each basis
        # ================================================================
        
        print(f'\n{"Basis":<25} {"Fisher":<12} {"Class Dist":<12} {"Cond Num":<12} {"Linear Acc":<12}')
        print('-'*75)
        
        for basis_key, (basis_matrix, basis_name) in bases.items():
            fisher = compute_fisher_criterion(basis_matrix, labels)
            class_dist = compute_class_centroid_distances(basis_matrix, labels)
            cond_num = compute_condition_number(basis_matrix)
            
            # Linear classifier accuracy
            lin_acc, lin_std = test_linear_classifier(basis_matrix, labels)
            
            print(f'{basis_key:<25} {fisher:<12.6f} {class_dist:<12.4f} {cond_num:<12.2f} {lin_acc*100:<.2f}%')
            
            results['bases'][basis_key] = {
                'name': basis_name,
                'fisher': float(fisher),
                'class_dist': float(class_dist),
                'condition_number': float(cond_num) if not np.isinf(cond_num) else 'inf',
                'linear_accuracy': float(lin_acc),
                'linear_std': float(lin_std),
                'shape': list(basis_matrix.shape),
            }
        
        # ================================================================
        # Compute collapse ratios relative to X_diffused
        # ================================================================
        
        x_fisher = results['bases']['X_diffused']['fisher']
        x_class_dist = results['bases']['X_diffused']['class_dist']
        
        print(f'\n{"Basis":<25} {"Fisher Collapse (X/B)":<22} {"Dist Collapse (X/B)":<22}')
        print('-'*70)
        
        for basis_key, data in results['bases'].items():
            if basis_key == 'X_diffused':
                continue
            
            fisher_collapse = x_fisher / (data['fisher'] + 1e-10)
            dist_collapse = x_class_dist / (data['class_dist'] + 1e-10)
            
            results['bases'][basis_key]['fisher_collapse'] = float(fisher_collapse)
            results['bases'][basis_key]['dist_collapse'] = float(dist_collapse)
            
            print(f'{basis_key:<25} {fisher_collapse:<22.2f} {dist_collapse:<22.2f}')
        
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
        results = analyze_dataset(dataset_name, args.k_diffusion, args.num_random_bases)
        if results:
            all_results[dataset_name] = results
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print('\n' + '='*100)
    print('SUMMARY: FISHER COLLAPSE BY BASIS TYPE')
    print('='*100)
    
    # Collect collapse ratios for each basis type
    basis_types = ['QR', 'SVD', 'Rayleigh-Ritz', 'Random_0', 'Class-Optimal']
    
    print(f'\n{"Dataset":<20}', end='')
    for bt in basis_types:
        print(f'{bt:<15}', end='')
    print()
    print('-'*95)
    
    for dataset_name, results in all_results.items():
        print(f'{dataset_name:<20}', end='')
        for bt in basis_types:
            if bt in results['bases'] and 'fisher_collapse' in results['bases'][bt]:
                collapse = results['bases'][bt]['fisher_collapse']
                print(f'{collapse:<15.2f}', end='')
            else:
                print(f'{"N/A":<15}', end='')
        print()
    
    # ========================================================================
    # Key Finding
    # ========================================================================
    
    print('\n' + '='*100)
    print('KEY FINDING: Is collapse specific to Rayleigh-Ritz?')
    print('='*100)
    
    # Average collapse by basis type
    collapse_by_type = {bt: [] for bt in basis_types}
    
    for dataset_name, results in all_results.items():
        for bt in basis_types:
            if bt in results['bases'] and 'fisher_collapse' in results['bases'][bt]:
                collapse_by_type[bt].append(results['bases'][bt]['fisher_collapse'])
    
    print(f'\n{"Basis Type":<25} {"Mean Collapse":<15} {"Interpretation":<40}')
    print('-'*80)
    
    for bt in basis_types:
        if len(collapse_by_type[bt]) > 0:
            mean_collapse = np.mean(collapse_by_type[bt])
            
            if mean_collapse < 2:
                interp = "Preserves class geometry"
            elif mean_collapse < 10:
                interp = "Mild collapse"
            elif mean_collapse < 100:
                interp = "Significant collapse"
            else:
                interp = "Severe collapse"
            
            print(f'{bt:<25} {mean_collapse:<15.2f} {interp:<40}')
    
    # ========================================================================
    # Conclusion
    # ========================================================================
    
    rr_collapse = np.mean(collapse_by_type['Rayleigh-Ritz']) if collapse_by_type['Rayleigh-Ritz'] else 0
    qr_collapse = np.mean(collapse_by_type['QR']) if collapse_by_type['QR'] else 0
    svd_collapse = np.mean(collapse_by_type['SVD']) if collapse_by_type['SVD'] else 0
    random_collapse = np.mean(collapse_by_type['Random_0']) if collapse_by_type['Random_0'] else 0
    
    print('\n' + '='*100)
    print('CONCLUSION')
    print('='*100)
    
    if rr_collapse > 2 * max(qr_collapse, svd_collapse, random_collapse):
        print('''
FINDING: Rayleigh-Ritz causes SIGNIFICANTLY MORE collapse than other bases.

The class geometry collapse is SPECIFIC to Rayleigh-Ritz, not inherent
to orthonormalization. This suggests the problem is the alignment with
graph smoothness rather than the mathematical structure of orthonormal bases.

IMPLICATION: We can potentially find a better basis for span(X) that:
1. Is orthonormal (for numerical stability)
2. Preserves class geometry (unlike Rayleigh-Ritz)
3. Still captures graph structure (useful for GNNs)
''')
    elif all(c > 10 for c in [qr_collapse, svd_collapse, random_collapse] if c > 0):
        print('''
FINDING: ALL orthonormal bases cause significant collapse.

The class geometry collapse is INHERENT to orthonormalization, not specific
to Rayleigh-Ritz. This suggests that X_diffused's non-orthogonal structure
is essential for preserving class information.

IMPLICATION: Orthonormalizing X_diffused fundamentally destroys class geometry.
The solution may require working with non-orthonormal bases or finding
a way to preserve class structure during orthonormalization.
''')
    else:
        print('''
FINDING: Mixed results - need further analysis.

Some bases preserve class geometry better than others. This suggests
there may be a "sweet spot" between graph alignment and class preservation.
''')
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
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
    
    with open(f'{OUTPUT_DIR}/alternative_bases_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f'\n✓ Results saved: {OUTPUT_DIR}/alternative_bases_results.json')
    
    # ========================================================================
    # Generate Plot
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Fisher collapse by basis type
    ax1 = axes[0]
    basis_order = ['QR', 'SVD', 'Random_0', 'Rayleigh-Ritz', 'Class-Optimal']
    x_pos = np.arange(len(basis_order))
    
    mean_collapses = [np.mean(collapse_by_type[bt]) if collapse_by_type[bt] else 0 for bt in basis_order]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    ax1.bar(x_pos, mean_collapses, color=colors)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(basis_order, rotation=45, ha='right')
    ax1.set_ylabel('Mean Fisher Collapse (X/Basis)')
    ax1.set_title('Class Geometry Collapse by Basis Type')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No collapse')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Per-dataset comparison (Rayleigh-Ritz vs QR)
    ax2 = axes[1]
    datasets = list(all_results.keys())
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    rr_collapses = [all_results[d]['bases']['Rayleigh-Ritz']['fisher_collapse'] 
                   if 'Rayleigh-Ritz' in all_results[d]['bases'] else 0 for d in datasets]
    qr_collapses = [all_results[d]['bases']['QR']['fisher_collapse'] 
                   if 'QR' in all_results[d]['bases'] else 0 for d in datasets]
    
    ax2.bar(x_pos - width/2, rr_collapses, width, label='Rayleigh-Ritz', color='red')
    ax2.bar(x_pos + width/2, qr_collapses, width, label='QR', color='blue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylabel('Fisher Collapse (X/Basis)')
    ax2.set_title('Rayleigh-Ritz vs QR Collapse per Dataset')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/basis_comparison.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {OUTPUT_DIR}/plots/basis_comparison.png')
    plt.close()
    
    print('\n' + '='*100)
    print('INVESTIGATION COMPLETE')
    print('='*100)