"""
Investigation 2: Random Subspace Experiment
============================================

Tests whether RowNorm works on restricted eigenvectors from RANDOM subspaces
instead of engineered features X.

Research Question:
- Does RowNorm underperform because of:
  (1) Fundamental limitation of restricted eigenvectors? OR
  (2) Poor conditioning of engineered features X?

Experiment Design:
- Generate X_r ~ N(0,1) with dimension = rank(X)
- Compute restricted eigenvectors V_r from X_r
- Run experiments (a), (b), (c) on X_r
- Compare to results on engineered X
- Control randomness with 3 different X_r initializations

Usage:
    # Fixed benchmark splits (default)
    python experiments/investigation2_random_subspaces.py [dataset_name]
    
    # Random 60/20/20 splits
    python experiments/investigation2_random_subspaces.py [dataset_name] --random-splits
    
Examples:
    python experiments/investigation2_random_subspaces.py ogbn-arxiv
    python experiments/investigation2_random_subspaces.py wikics --random-splits
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
USE_RANDOM_SPLITS = '--random-splits' in sys.argv  # Support for random splits flag

# Experimental parameters - MATCHING investigation2_directions_AB.py
NUM_RANDOM_SUBSPACES = 3   # Number of random subspace initializations
NUM_SEEDS = 5              # Training seeds per subspace
NUM_RANDOM_SPLITS = 5 if USE_RANDOM_SPLITS else 1  # Number of random data splits

# Hyperparameters
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Set device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory - split-specific like investigation2_directions_AB.py
split_type = 'random_splits' if USE_RANDOM_SPLITS else 'fixed_splits'
output_base = f'results/investigation2_random_subspaces/{DATASET_NAME}/{split_type}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

print('='*70)
print('INVESTIGATION 2: RANDOM SUBSPACE EXPERIMENT')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Random splits: {"Yes" if USE_RANDOM_SPLITS else "No (fixed benchmark)"}')
print(f'Random subspaces: {NUM_RANDOM_SUBSPACES}')
print(f'Training seeds per subspace: {NUM_SEEDS}')
if USE_RANDOM_SPLITS:
    print(f'Data splits: {NUM_RANDOM_SPLITS}')
    print(f'Total runs per experiment: {NUM_RANDOM_SUBSPACES * NUM_SEEDS * NUM_RANDOM_SPLITS}')
else:
    print(f'Total runs per experiment: {NUM_RANDOM_SUBSPACES * NUM_SEEDS}')
print(f'Device: {device}')
print('='*70)

# ============================================================================
# Helper Functions
# ============================================================================

def create_random_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=0):
    """Create random train/val/test split - MATCHING investigation2_directions_AB.py"""
    np.random.seed(seed)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

def generate_random_subspace(num_nodes, dimension, seed):
    """
    Generate random Gaussian subspace
    
    Args:
        num_nodes: number of graph nodes
        dimension: subspace dimension (should match rank(X))
        seed: random seed for reproducibility
    
    Returns:
        X_r: (num_nodes, dimension) random feature matrix
    """
    rng = np.random.RandomState(seed)
    X_r = rng.randn(num_nodes, dimension).astype(np.float64)
    return X_r

def compute_restricted_eigenvectors(X, L, D, eps_base=1e-8):
    """
    Compute restricted eigenvectors from features X
    
    Args:
        X: (n, d) feature matrix (can be engineered or random)
        L: (n, n) Laplacian matrix
        D: (n, n) degree matrix
        eps_base: regularization base
    
    Returns:
        U: (n, d_eff) D-orthonormal restricted eigenvectors
        eigenvalues: (d_eff,) eigenvalue array
        d_effective: effective dimension after QR
    """
    # QR decomposition to handle potential rank deficiency
    Q, R = la.qr(X, mode='economic')
    rank_X = np.linalg.matrix_rank(R, tol=1e-6)
    
    if rank_X < X.shape[1]:
        print(f'  Rank deficiency: {X.shape[1]} → {rank_X}')
        Q = Q[:, :rank_X]
    
    d_effective = Q.shape[1]
    
    # Project Laplacian and degree matrices
    LQ = (L @ Q).astype(np.float64)
    DQ = (D @ Q).astype(np.float64)
    
    L_r = Q.T @ LQ
    D_r = Q.T @ DQ
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularization
    trace_Dr = np.trace(D_r)
    eps = eps_base * (trace_Dr / d_effective)
    D_r = D_r + eps * np.eye(d_effective)
    
    # Solve generalized eigenvalue problem
    eigenvalues, V = la.eigh(L_r, D_r)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Map back to node space
    U = (Q @ V).astype(np.float32)
    
    # Verify D-orthonormality
    DU = (D @ U).astype(np.float64)
    G = U.astype(np.float64).T @ DU
    deviation = np.abs(G - np.eye(d_effective)).max()
    print(f'  D-orthonormality: max|U^T D U - I| = {deviation:.2e}')
    
    return U, eigenvalues, d_effective

def train_model_single_run(model, train_loader, X_val, y_val, X_test, y_test,
                           epochs, lr, weight_decay, device):
    """
    Train model for one run with validation tracking
    
    Returns:
        dict with test_acc, val_accs, train_losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss / len(train_loader))
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_pred = val_output.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
            val_accs.append(val_acc)
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    return {
        'test_acc': test_acc,
        'val_accs': val_accs,
        'train_losses': train_losses
    }

def run_three_experiments(X, U, labels, train_idx, val_idx, test_idx,
                         num_seeds, epochs, hidden_dim, batch_size, 
                         lr, weight_decay, device):
    """
    Run experiments (a), (b), (c) with multiple random seeds
    
    Args:
        X: raw features (engineered or random)
        U: restricted eigenvectors from X
        labels, train_idx, val_idx, test_idx: dataset splits
        num_seeds: number of random initialization seeds
        
    Returns:
        dict with aggregated results for all three experiments
    """
    num_classes = int(labels.max()) + 1
    d_in = X.shape[1]
    
    results = {
        'a_X_std': [],
        'b_U_std': [],
        'c_U_row': []
    }
    
    # Prepare labels
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)
    y_test = torch.from_numpy(labels[test_idx]).long().to(device)
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # ================================================================
        # Experiment (a): X → StandardScaler → Standard MLP
        # ================================================================
        scaler_a = StandardScaler().fit(X[train_idx])
        X_train_a = torch.from_numpy(scaler_a.transform(X[train_idx])).float().to(device)
        X_val_a = torch.from_numpy(scaler_a.transform(X[val_idx])).float().to(device)
        X_test_a = torch.from_numpy(scaler_a.transform(X[test_idx])).float().to(device)
        
        loader_a = DataLoader(TensorDataset(X_train_a, y_train), 
                             batch_size=batch_size, shuffle=True)
        
        model_a = StandardMLP(d_in, hidden_dim, num_classes)
        result_a = train_model_single_run(model_a, loader_a, X_val_a, y_val, 
                                         X_test_a, y_test, epochs, lr, 
                                         weight_decay, device)
        results['a_X_std'].append(result_a)
        
        # ================================================================
        # Experiment (b): U → StandardScaler → Standard MLP
        # ================================================================
        scaler_b = StandardScaler().fit(U[train_idx])
        U_train_b = torch.from_numpy(scaler_b.transform(U[train_idx])).float().to(device)
        U_val_b = torch.from_numpy(scaler_b.transform(U[val_idx])).float().to(device)
        U_test_b = torch.from_numpy(scaler_b.transform(U[test_idx])).float().to(device)
        
        loader_b = DataLoader(TensorDataset(U_train_b, y_train),
                             batch_size=batch_size, shuffle=True)
        
        model_b = StandardMLP(U.shape[1], hidden_dim, num_classes)
        result_b = train_model_single_run(model_b, loader_b, U_val_b, y_val,
                                         U_test_b, y_test, epochs, lr,
                                         weight_decay, device)
        results['b_U_std'].append(result_b)
        
        # ================================================================
        # Experiment (c): U → RowNorm MLP
        # ================================================================
        U_train_c = torch.from_numpy(U[train_idx]).float().to(device)
        U_val_c = torch.from_numpy(U[val_idx]).float().to(device)
        U_test_c = torch.from_numpy(U[test_idx]).float().to(device)
        
        loader_c = DataLoader(TensorDataset(U_train_c, y_train),
                             batch_size=batch_size, shuffle=True)
        
        model_c = RowNormMLP(U.shape[1], hidden_dim, num_classes)
        result_c = train_model_single_run(model_c, loader_c, U_val_c, y_val,
                                         U_test_c, y_test, epochs, lr,
                                         weight_decay, device)
        results['c_U_row'].append(result_c)
        
        print(f'    Seed {seed+1}/{num_seeds}: (a)={result_a["test_acc"]:.4f} '
              f'(b)={result_b["test_acc"]:.4f} (c)={result_c["test_acc"]:.4f}')
    
    # Aggregate across seeds
    aggregated = {}
    for exp_name in ['a_X_std', 'b_U_std', 'c_U_row']:
        test_accs = [r['test_acc'] for r in results[exp_name]]
        aggregated[exp_name] = {
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
            'val_accs_mean': np.mean([r['val_accs'] for r in results[exp_name]], axis=0).tolist(),
            'val_accs_std': np.std([r['val_accs'] for r in results[exp_name]], axis=0).tolist(),
        }
    
    return aggregated

# ============================================================================
# Main Experiment
# ============================================================================

print('\n[1/6] Loading dataset...')
(edge_index, X_original, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Loaded: {num_nodes} nodes, {num_classes} classes, {X_original.shape[1]} features')

print('\n[2/6] Building graph matrices...')
adj, L, D = build_graph_matrices(edge_index, num_nodes)
print(f'Graph matrices built: adj={adj.shape}, L={L.shape}, D={D.shape}')

print('\n[3/6] Processing original engineered features X...')
# QR to get effective rank
Q_orig, R_orig = la.qr(X_original, mode='economic')
rank_X_orig = np.linalg.matrix_rank(R_orig, tol=1e-6)
print(f'Original features: dim={X_original.shape[1]}, rank={rank_X_orig}')

if rank_X_orig < X_original.shape[1]:
    print(f'Rank deficiency: using effective dimension {rank_X_orig}')
    dimension = rank_X_orig
else:
    dimension = X_original.shape[1]

# ============================================================================
# [4/6] Run experiments on ORIGINAL engineered features
# ============================================================================
print(f'\n[4/6] Running experiments on ORIGINAL engineered features...')

# Compute restricted eigenvectors from original X ONCE (used across all splits)
print('Computing restricted eigenvectors from original X...')
U_original, evals_orig, d_eff_orig = compute_restricted_eigenvectors(
    X_original, L, D
)

# Determine batch size
if len(train_idx_fixed) > 256:
    batch_size = 128
else:
    batch_size = len(train_idx_fixed)

print(f'Batch size: {batch_size}')

# Storage for results across splits
results_original_all_splits = []

# Loop over data splits
for split_idx in range(NUM_RANDOM_SPLITS):
    if USE_RANDOM_SPLITS:
        print(f'\n--- Random Split {split_idx+1}/{NUM_RANDOM_SPLITS} ---')
        train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
    else:
        print(f'\n--- Using Fixed Benchmark Split ---')
        train_idx, val_idx, test_idx = train_idx_fixed, val_idx_fixed, test_idx_fixed
    
    print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')
    
    # Run experiments on this split
    result_split = run_three_experiments(
        X_original, U_original, labels, train_idx, val_idx, test_idx,
        NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
        LEARNING_RATE, WEIGHT_DECAY, device
    )
    
    results_original_all_splits.append(result_split)

# Aggregate across splits
results_original = {}
for exp_name in ['a_X_std', 'b_U_std', 'c_U_row']:
    test_accs = [r[exp_name]['test_acc_mean'] for r in results_original_all_splits]
    results_original[exp_name] = {
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
    }

print('\n' + '='*70)
print('ORIGINAL FEATURES RESULTS (aggregated across splits)')
print('='*70)
print(f"(a) X_std: {results_original['a_X_std']['test_acc_mean']:.4f} ± {results_original['a_X_std']['test_acc_std']:.4f}")
print(f"(b) U_std: {results_original['b_U_std']['test_acc_mean']:.4f} ± {results_original['b_U_std']['test_acc_std']:.4f}")
print(f"(c) U_row: {results_original['c_U_row']['test_acc_mean']:.4f} ± {results_original['c_U_row']['test_acc_std']:.4f}")

# ============================================================================
# [5/6] Run experiments on RANDOM subspaces
# ============================================================================
print('\n[5/6] Running experiments on RANDOM subspaces...')
print(f'Generating {NUM_RANDOM_SUBSPACES} random subspaces with dimension={dimension}')

results_random_all_subspaces = []

for subspace_idx in range(NUM_RANDOM_SUBSPACES):
    print(f'\n{"="*70}')
    print(f'RANDOM SUBSPACE {subspace_idx+1}/{NUM_RANDOM_SUBSPACES}')
    print(f'{"="*70}')
    
    # Generate random features
    print(f'Generating X_r ~ N(0,1) with seed={subspace_idx}...')
    X_random = generate_random_subspace(num_nodes, dimension, seed=subspace_idx)
    
    # Compute restricted eigenvectors from random X_r ONCE
    print('Computing restricted eigenvectors from X_r...')
    U_random, evals_rand, d_eff_rand = compute_restricted_eigenvectors(
        X_random, L, D
    )
    
    # Storage for this subspace across splits
    results_random_all_splits = []
    
    # Loop over data splits
    for split_idx in range(NUM_RANDOM_SPLITS):
        if USE_RANDOM_SPLITS:
            print(f'\n  Split {split_idx+1}/{NUM_RANDOM_SPLITS}')
            train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
        else:
            train_idx, val_idx, test_idx = train_idx_fixed, val_idx_fixed, test_idx_fixed
        
        # Run experiments
        result_split = run_three_experiments(
            X_random, U_random, labels, train_idx, val_idx, test_idx,
            NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
            LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        results_random_all_splits.append(result_split)
    
    # Aggregate across splits for this subspace
    result_subspace = {}
    for exp_name in ['a_X_std', 'b_U_std', 'c_U_row']:
        test_accs = [r[exp_name]['test_acc_mean'] for r in results_random_all_splits]
        result_subspace[exp_name] = {
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
        }
    
    results_random_all_subspaces.append(result_subspace)
    
    print(f'\nSubspace {subspace_idx+1} summary (across {NUM_RANDOM_SPLITS} split(s)):')
    print(f"  (a) X_r_std: {result_subspace['a_X_std']['test_acc_mean']:.4f} ± {result_subspace['a_X_std']['test_acc_std']:.4f}")
    print(f"  (b) U_r_std: {result_subspace['b_U_std']['test_acc_mean']:.4f} ± {result_subspace['b_U_std']['test_acc_std']:.4f}")
    print(f"  (c) U_r_row: {result_subspace['c_U_row']['test_acc_mean']:.4f} ± {result_subspace['c_U_row']['test_acc_std']:.4f}")

# ============================================================================
# [6/6] Aggregate Across Random Subspaces
# ============================================================================
print('\n[6/6] Aggregating results across random subspaces...')

aggregated_random = {}
for exp_name in ['a_X_std', 'b_U_std', 'c_U_row']:
    # Collect means across subspaces
    means = [r[exp_name]['test_acc_mean'] for r in results_random_all_subspaces]
    
    aggregated_random[exp_name] = {
        'mean_across_subspaces': float(np.mean(means)),
        'std_across_subspaces': float(np.std(means)),
        'min': float(np.min(means)),
        'max': float(np.max(means))
    }

# ============================================================================
# Print Final Summary
# ============================================================================
print('\n' + '='*70)
print('FINAL RESULTS SUMMARY')
print('='*70)

print(f'\nDataset: {DATASET_NAME}')
print(f'Split type: {split_type}')
print(f'Feature dimension: {dimension}')
print(f'Random subspaces tested: {NUM_RANDOM_SUBSPACES}')
print(f'Data splits per condition: {NUM_RANDOM_SPLITS}')
print(f'Training seeds per split: {NUM_SEEDS}')

print('\n' + '-'*70)
print('ORIGINAL ENGINEERED FEATURES (X)')
print('-'*70)
print(f"(a) X → StandardScaler → Standard MLP:  {results_original['a_X_std']['test_acc_mean']:.4f} ± {results_original['a_X_std']['test_acc_std']:.4f}")
print(f"(b) U → StandardScaler → Standard MLP:  {results_original['b_U_std']['test_acc_mean']:.4f} ± {results_original['b_U_std']['test_acc_std']:.4f}")
print(f"(c) U → RowNorm MLP:                    {results_original['c_U_row']['test_acc_mean']:.4f} ± {results_original['c_U_row']['test_acc_std']:.4f}")

basis_effect_orig = ((results_original['b_U_std']['test_acc_mean'] - results_original['a_X_std']['test_acc_mean']) 
                     / results_original['a_X_std']['test_acc_mean'] * 100)
model_effect_orig = ((results_original['c_U_row']['test_acc_mean'] - results_original['b_U_std']['test_acc_mean'])
                     / results_original['b_U_std']['test_acc_mean'] * 100)

print(f'\nDirection B (basis sensitivity):  {basis_effect_orig:+.1f}%')
print(f'Direction A (RowNorm effect):     {model_effect_orig:+.1f}%')

print('\n' + '-'*70)
print(f'RANDOM GAUSSIAN FEATURES (X_r) - Averaged over {NUM_RANDOM_SUBSPACES} subspaces')
print('-'*70)
print(f"(a) X_r → StandardScaler → Standard MLP:  {aggregated_random['a_X_std']['mean_across_subspaces']:.4f} ± {aggregated_random['a_X_std']['std_across_subspaces']:.4f}")
print(f"    Range: [{aggregated_random['a_X_std']['min']:.4f}, {aggregated_random['a_X_std']['max']:.4f}]")
print(f"(b) U_r → StandardScaler → Standard MLP:  {aggregated_random['b_U_std']['mean_across_subspaces']:.4f} ± {aggregated_random['b_U_std']['std_across_subspaces']:.4f}")
print(f"    Range: [{aggregated_random['b_U_std']['min']:.4f}, {aggregated_random['b_U_std']['max']:.4f}]")
print(f"(c) U_r → RowNorm MLP:                    {aggregated_random['c_U_row']['mean_across_subspaces']:.4f} ± {aggregated_random['c_U_row']['std_across_subspaces']:.4f}")
print(f"    Range: [{aggregated_random['c_U_row']['min']:.4f}, {aggregated_random['c_U_row']['max']:.4f}]")

basis_effect_rand = ((aggregated_random['b_U_std']['mean_across_subspaces'] - aggregated_random['a_X_std']['mean_across_subspaces'])
                     / aggregated_random['a_X_std']['mean_across_subspaces'] * 100)
model_effect_rand = ((aggregated_random['c_U_row']['mean_across_subspaces'] - aggregated_random['b_U_std']['mean_across_subspaces'])
                     / aggregated_random['b_U_std']['mean_across_subspaces'] * 100)

print(f'\nDirection B (basis sensitivity):  {basis_effect_rand:+.1f}%')
print(f'Direction A (RowNorm effect):     {model_effect_rand:+.1f}%')

print('\n' + '='*70)
print('KEY COMPARISONS')
print('='*70)

# Comparison 1: RowNorm effect on engineered vs random
print('\n1. Direction A (RowNorm Effect):')
print(f'   Original X:  {model_effect_orig:+.1f}%')
print(f'   Random X_r:  {model_effect_rand:+.1f}%')
improvement = model_effect_rand - model_effect_orig
print(f'   Improvement: {improvement:+.1f}pp')
if model_effect_rand > 0 and model_effect_orig < 0:
    print('   → ✓ RowNorm WORKS on random but NOT on engineered features!')
elif improvement > 5:
    print('   → ✓ RowNorm works MUCH BETTER on random features')
elif improvement > 2:
    print('   → ✓ RowNorm works better on random features')
elif abs(improvement) < 2:
    print('   → ≈ Similar RowNorm effect on both')
else:
    print('   → ✗ RowNorm does not benefit from random features')

# Comparison 2: Engineered vs random baseline
print('\n2. Baseline Accuracy (Experiment a):')
print(f'   Original X:  {results_original["a_X_std"]["test_acc_mean"]:.4f}')
print(f'   Random X_r:  {aggregated_random["a_X_std"]["mean_across_subspaces"]:.4f}')
diff = aggregated_random['a_X_std']['mean_across_subspaces'] - results_original['a_X_std']['test_acc_mean']
print(f'   Difference:  {diff:+.4f}')
if abs(diff) < 0.02:
    print(f'   → ≈ Similar performance')
elif diff > 0.02:
    print(f'   → 🤯 Random features OUTPERFORM engineered!')
else:
    print(f'   → ✓ Engineered features better (expected)')

# Comparison 3: Variance analysis
print('\n3. Variance Across Random Subspaces:')
print(f"   (a) X_r std: {aggregated_random['a_X_std']['std_across_subspaces']:.4f}")
print(f"   (b) U_r std: {aggregated_random['b_U_std']['std_across_subspaces']:.4f}")
print(f"   (c) U_r std: {aggregated_random['c_U_row']['std_across_subspaces']:.4f}")
max_std = max(aggregated_random['a_X_std']['std_across_subspaces'],
              aggregated_random['b_U_std']['std_across_subspaces'],
              aggregated_random['c_U_row']['std_across_subspaces'])
if max_std > 0.05:
    print('   → ⚠️  HIGH variance: specific random subspace matters!')
elif max_std > 0.02:
    print('   → ≈ MODERATE variance')
else:
    print('   → ✓ LOW variance: results consistent across random subspaces')

print('\n' + '='*70)

# Save results
results_dict = {
    'dataset': DATASET_NAME,
    'split_type': split_type,
    'dimension': int(dimension),
    'num_random_subspaces': NUM_RANDOM_SUBSPACES,
    'num_seeds_per_subspace': NUM_SEEDS,
    'num_data_splits': NUM_RANDOM_SPLITS,
    'original_features': results_original,
    'random_subspaces': results_random_all_subspaces,
    'aggregated_random': aggregated_random,
    'comparisons': {
        'direction_A_original': float(model_effect_orig),
        'direction_A_random': float(model_effect_rand),
        'direction_A_improvement': float(improvement),
        'direction_B_original': float(basis_effect_orig),
        'direction_B_random': float(basis_effect_rand),
        'baseline_diff': float(diff)
    }
}

save_path = f'{output_base}/metrics/results_complete.json'
with open(save_path, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f'✓ Results saved: {save_path}')
print(f'✓ Experiment complete for {DATASET_NAME}!')
print('='*70)