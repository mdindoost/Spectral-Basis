"""
Compute Feature Matrix Ranks Across All Datasets
=================================================

This script computes:
- Rank of feature matrix X
- Full dimensionality
- Rank deficiency (if any)
- Condition number
- Singular value statistics

Usage:
    python compute_feature_ranks.py


"""

import os
import numpy as np
import torch
from scipy import linalg as la
from utils import load_dataset

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    'ogbn-arxiv',
    'cora',
    'citeseer',
    'pubmed',
    'wikics',
    'amazon-computers',
    'coauthor-cs',
    'coauthor-physics'
]

OUTPUT_FILE = 'results/feature_ranks_summary.txt'
os.makedirs('results', exist_ok=True)

# Tolerance for rank computation
RANK_TOLERANCE = 1e-10

# ============================================================================
# Rank Computation Functions
# ============================================================================

def compute_rank_statistics(X, tolerance=1e-10):
    """
    Compute comprehensive rank statistics for feature matrix X.
    Uses QR decomposition method (same as compute_restricted_eigenvectors).
    
    Args:
        X: Feature matrix (n x d)
        tolerance: Threshold for considering singular values as zero
    
    Returns:
        dict with statistics
    """
    n, d = X.shape
    
    # Method 1: QR decomposition (same as old compute_restricted_eigenvectors)
    print(f'  Computing rank via QR decomposition...')
    from scipy import linalg as la
    Q, R = la.qr(X, mode='economic')
    rank_qr = np.linalg.matrix_rank(R, tol=tolerance)
    
    # Method 2: SVD for condition number and singular value analysis
    print(f'  Computing SVD for condition number...')
    # For large matrices, only compute singular values (not full SVD)
    if n > 5000 or d > 5000:
        # Use sparse SVD for large matrices
        s = np.linalg.svd(X, compute_uv=False)
    else:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Compute rank from singular values (should match QR method)
    rank_svd = np.sum(s > tolerance)
    
    # Use QR rank as primary (more numerically stable)
    rank = rank_qr
    
    # Rank deficiency
    rank_deficiency = d - rank
    
    # Condition number (ratio of largest to smallest non-zero singular value)
    if rank > 0:
        s_nonzero = s[s > tolerance]
        condition_number = s_nonzero[0] / s_nonzero[-1] if len(s_nonzero) > 0 else np.inf
    else:
        condition_number = np.inf
    
    # Singular value statistics
    s_max = s[0] if len(s) > 0 else 0
    s_min = s[min(rank-1, len(s)-1)] if rank > 0 and len(s) > 0 else 0
    s_mean = np.mean(s[:rank]) if rank > 0 else 0
    s_median = np.median(s[:rank]) if rank > 0 else 0
    
    # Effective rank (number of singular values > 1% of max)
    effective_rank_01 = np.sum(s > 0.01 * s_max)
    
    # Number of near-zero singular values
    num_near_zero = np.sum(s <= tolerance)
    
    return {
        'num_nodes': n,
        'num_features': d,
        'rank': rank,
        'rank_qr': rank_qr,
        'rank_svd': rank_svd,
        'rank_deficiency': rank_deficiency,
        'rank_ratio': rank / d if d > 0 else 0,
        'condition_number': condition_number,
        'singular_values': {
            'max': float(s_max),
            'min': float(s_min),
            'mean': float(s_mean),
            'median': float(s_median),
            'num_near_zero': int(num_near_zero)
        },
        'effective_rank_01': int(effective_rank_01),
    }

# ============================================================================
# Main Analysis
# ============================================================================

print('='*70)
print('FEATURE MATRIX RANK ANALYSIS')
print('='*70)
print(f'Datasets: {len(DATASETS)}')
print(f'Rank tolerance: {RANK_TOLERANCE}')
print('='*70)

results = {}

for dataset_name in DATASETS:
    print(f'\n{dataset_name.upper()}')
    print('-'*70)
    
    try:
        # Load dataset
        print(f'  Loading dataset...')
        data_tuple = load_dataset(dataset_name)
        
        print(f'  Data tuple length: {len(data_tuple)}')
        
        # Extract features (CORRECT ORDER from load_dataset)
        if len(data_tuple) == 8:
            # Order: edge_index, node_features, labels, num_nodes, num_classes, train_idx, val_idx, test_idx
            edge_index = data_tuple[0]      # (2, num_edges)
            X = data_tuple[1]               # (num_nodes, num_features) <- THIS IS WHAT WE WANT
            labels = data_tuple[2]
            num_nodes = data_tuple[3]
            num_classes = data_tuple[4]
        else:
            raise ValueError(f"Unexpected data format: {len(data_tuple)} elements")
        
        # Convert to numpy if needed
        if torch.is_tensor(X):
            X = X.numpy()
        
        print(f'  Feature matrix X shape: {X.shape}')
        print(f'  Expected: ({num_nodes}, num_features)')
        print(f'  Edge index shape: {edge_index.shape if hasattr(edge_index, "shape") else "N/A"}')
        
        # Sanity check
        if X.shape[0] != num_nodes:
            raise ValueError(f"X.shape[0]={X.shape[0]} but num_nodes={num_nodes}. Data loading error!")
        
        print(f'  Verified: {X.shape[0]} nodes × {X.shape[1]} features')
        
        # Compute rank statistics
        stats = compute_rank_statistics(X, tolerance=RANK_TOLERANCE)
        
        # Display results
        print(f'  Rank (QR method): {stats["rank_qr"]}/{stats["num_features"]}')
        print(f'  Rank (SVD method): {stats["rank_svd"]}/{stats["num_features"]}')
        if stats['rank_qr'] != stats['rank_svd']:
            print(f'  WARNING: QR and SVD ranks differ!')
        
        if stats['rank_deficiency'] > 0:
            print(f'  Rank deficiency: {stats["rank_deficiency"]} ({stats["rank_ratio"]*100:.1f}% rank)')
        else:
            print(f'  Full rank ✓')
        print(f'  Condition number: {stats["condition_number"]:.2e}')
        print(f'  Effective rank (>1% σ_max): {stats["effective_rank_01"]}')
        
        results[dataset_name] = stats
        
    except Exception as e:
        print(f'  ERROR: {e}')
        results[dataset_name] = {'error': str(e)}

# ============================================================================
# Generate Summary Report
# ============================================================================

print('\n' + '='*70)
print('SUMMARY REPORT')
print('='*70)

# Create summary table
summary_lines = []
summary_lines.append('='*100)
summary_lines.append('FEATURE MATRIX RANKS ACROSS ALL DATASETS')
summary_lines.append('='*100)
summary_lines.append('')
summary_lines.append(f'{"Dataset":<20} {"Nodes":>10} {"Features":>10} {"Rank":>10} {"Deficiency":>12} {"Rank %":>10} {"Cond. #":>12}')
summary_lines.append('-'*100)

for dataset_name in DATASETS:
    if dataset_name in results and 'error' not in results[dataset_name]:
        stats = results[dataset_name]
        
        rank_pct = f"{stats['rank_ratio']*100:.1f}%"
        cond = f"{stats['condition_number']:.2e}"
        
        # Highlight rank deficiency if present
        deficiency_str = str(stats['rank_deficiency'])
        if stats['rank_deficiency'] > 0:
            deficiency_str = f"*{deficiency_str}*"
        
        summary_lines.append(
            f"{dataset_name:<20} "
            f"{stats['num_nodes']:>10} "
            f"{stats['num_features']:>10} "
            f"{stats['rank']:>10} "
            f"{deficiency_str:>12} "
            f"{rank_pct:>10} "
            f"{cond:>12}"
        )

summary_lines.append('='*100)
summary_lines.append('* Indicates rank deficiency (rank < num_features)')
summary_lines.append('')

# Detailed statistics
summary_lines.append('')
summary_lines.append('DETAILED STATISTICS')
summary_lines.append('='*100)

for dataset_name in DATASETS:
    if dataset_name in results and 'error' not in results[dataset_name]:
        stats = results[dataset_name]
        
        summary_lines.append('')
        summary_lines.append(f'{dataset_name.upper()}')
        summary_lines.append('-'*70)
        summary_lines.append(f'  Nodes: {stats["num_nodes"]}')
        summary_lines.append(f'  Features: {stats["num_features"]}')
        summary_lines.append(f'  Rank (QR): {stats["rank_qr"]} ({stats["rank_ratio"]*100:.2f}%)')
        summary_lines.append(f'  Rank (SVD): {stats["rank_svd"]}')
        if stats['rank_qr'] != stats['rank_svd']:
            summary_lines.append(f'  WARNING: Methods disagree!')
        summary_lines.append(f'  Rank deficiency: {stats["rank_deficiency"]}')
        summary_lines.append(f'  Condition number: {stats["condition_number"]:.2e}')
        summary_lines.append(f'  Effective rank (>1% max): {stats["effective_rank_01"]}')
        summary_lines.append(f'  Singular values:')
        summary_lines.append(f'    Max: {stats["singular_values"]["max"]:.4f}')
        summary_lines.append(f'    Min: {stats["singular_values"]["min"]:.2e}')
        summary_lines.append(f'    Mean: {stats["singular_values"]["mean"]:.4f}')
        summary_lines.append(f'    Median: {stats["singular_values"]["median"]:.4f}')
        summary_lines.append(f'    Near-zero (<{RANK_TOLERANCE}): {stats["singular_values"]["num_near_zero"]}')

summary_lines.append('')
summary_lines.append('='*100)

# Print to console
for line in summary_lines:
    print(line)

# Save to file
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(summary_lines))

print(f'\nResults saved to: {OUTPUT_FILE}')
print('='*70)