"""
Investigation 2: Graph-Diffused Random Features
================================================

Purpose: Test whether graph structure (via diffusion) can create spectral advantages
even without semantic content.

Research Question:
- Random features (R): No semantics, no structure → catastrophic failure (-52pp)
- Graph-diffused (R*): No semantics, YES structure → ???

Hypothesis (from Yiannis):
"We will see an improvement because the 'diffusion' produces semi-convergent eigenvectors.
We will also likely see that rowNorm helps."

Usage:
    # Random splits (default)
    python experiments/investigation2_graph_diffused.py [dataset_name]
    
    # With specific k value
    python experiments/investigation2_graph_diffused.py [dataset_name] --k 32
    
Examples:
    python experiments/investigation2_graph_diffused.py ogbn-arxiv
    python experiments/investigation2_graph_diffused.py cora --k 16
"""

import os
import sys
import json
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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

# Parse k parameter
K_DIFFUSION = 16  # Default
if '--k' in sys.argv:
    k_idx = sys.argv.index('--k')
    if k_idx + 1 < len(sys.argv):
        K_DIFFUSION = int(sys.argv[k_idx + 1])

# Experimental parameters
NUM_SEEDS = 5              # Training seeds
NUM_RANDOM_SPLITS = 5      # Random data splits
NUM_RANDOM_SUBSPACES = 3   # Different random initializations

# Hyperparameters (matching Investigation 2)
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Set device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
output_base = f'results/investigation2_graph_diffused/{DATASET_NAME}/k{K_DIFFUSION}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

print('='*70)
print('INVESTIGATION 2: GRAPH-DIFFUSED RANDOM FEATURES')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion steps (k): {K_DIFFUSION}')
print(f'Random subspaces: {NUM_RANDOM_SUBSPACES}')
print(f'Random splits: {NUM_RANDOM_SPLITS}')
print(f'Training seeds per configuration: {NUM_SEEDS}')
print(f'Total runs per experiment: {NUM_RANDOM_SUBSPACES * NUM_SEEDS * NUM_RANDOM_SPLITS}')
print(f'Device: {device}')
print('='*70)

# ============================================================================
# Helper Functions
# ============================================================================

def create_random_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=0):
    """Create random train/val/test split"""
    np.random.seed(seed)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

def compute_normalized_adjacency(adj, D):
    """
    Compute symmetric normalized adjacency: A = D^(-1/2) Adj D^(-1/2)
    
    This is the diffusion operator (NOT the Laplacian)
    """
    deg = np.array(D.diagonal())
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return A_norm

def apply_graph_diffusion(A_norm, R, k, verbose=True):
    """
    Apply k-step graph diffusion: R* = A^k @ R
    
    Args:
        A_norm: Normalized adjacency matrix (sparse)
        R: Random features (n x d)
        k: Number of diffusion steps
        verbose: Print timing info
    
    Returns:
        R_star: Diffused features (n x d)
        time_elapsed: Computation time in seconds
    """
    start_time = time.time()
    
    R_star = R.copy()
    for i in range(k):
        R_star = A_norm @ R_star
        if verbose and (i + 1) % 8 == 0:
            print(f'  Diffusion step {i+1}/{k}...')
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f'  Diffusion completed in {elapsed:.2f}s')
    
    return R_star, elapsed

def compute_restricted_eigenvectors(X, L, D, eps_base=1e-7):
    """
    Compute restricted eigenvectors from features X
    
    Returns:
        U: (n, d_eff) D-orthonormal restricted eigenvectors
        eigenvalues: (d_eff,) eigenvalue array
        d_effective: effective dimension after QR
        ortho_error: D-orthonormality verification error
    """
    # QR decomposition
    Q, R_qr = la.qr(X, mode='economic')
    rank_X = np.linalg.matrix_rank(R_qr, tol=1e-10)
    Q = Q[:, :rank_X]
    
    print(f'  Rank: {rank_X}/{X.shape[1]}')
    
    # Project into feature subspace
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize
    eps = eps_base * np.trace(D_r) / rank_X
    D_r = D_r + eps * np.eye(rank_X)
    
    # Solve generalized eigenproblem
    eigenvalues, V = la.eigh(L_r, D_r)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Map back to node space
    U = Q @ V
    
    # Verify D-orthonormality
    DU = D @ U
    G = U.T @ DU
    ortho_error = np.abs(G - np.eye(rank_X)).max()
    
    print(f'  D-orthonormality error: {ortho_error:.2e}')
    
    return U.astype(np.float32), eigenvalues, rank_X, ortho_error

def train_model_simple(model, train_loader, X_val, y_val, X_test, y_test,
                       epochs, lr, weight_decay, device):
    """
    Simple training loop (no checkpointing)
    
    Returns:
        test_acc: Final test accuracy
        val_accs: Validation accuracy per epoch
        train_losses: Training loss per epoch
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accs = []
    
    for ep in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / len(train_loader))
        val_accs.append(val_acc)
    
    # Final test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    return test_acc, val_accs, train_losses

def run_three_experiments(features_dict, labels, train_idx, val_idx, test_idx,
                          num_seeds, epochs, hidden_dim, batch_size,
                          lr, weight_decay, device):
    """
    Run experiments (a), (b), (c) on given features
    
    Args:
        features_dict: Dict with keys 'raw', 'restricted'
        labels: Node labels
        train/val/test_idx: Data splits
        
    Returns:
        results: Dict with test accuracies for each experiment
    """
    X = features_dict['raw']
    U = features_dict['restricted']
    
    # Prepare labels
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)
    y_test = torch.from_numpy(labels[test_idx]).long().to(device)
    
    num_classes = len(np.unique(labels))
    
    results = {}
    
    # ========================================================================
    # Experiment (a): Raw features → StandardScaler → Standard MLP
    # ========================================================================
    print('  Exp (a): Raw → StandardScaler → Standard MLP')
    
    test_accs_a = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Scale
        scaler_a = StandardScaler().fit(X[train_idx])
        X_train = torch.from_numpy(scaler_a.transform(X[train_idx])).float().to(device)
        X_val = torch.from_numpy(scaler_a.transform(X[val_idx])).float().to(device)
        X_test = torch.from_numpy(scaler_a.transform(X[test_idx])).float().to(device)
        
        # Train
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                  batch_size=batch_size, shuffle=True)
        model = StandardMLP(X.shape[1], hidden_dim, num_classes)
        
        test_acc, _, _ = train_model_simple(
            model, train_loader, X_val, y_val, X_test, y_test,
            epochs, lr, weight_decay, device
        )
        test_accs_a.append(test_acc)
    
    results['a_raw_std'] = {
        'test_acc_mean': float(np.mean(test_accs_a)),
        'test_acc_std': float(np.std(test_accs_a)),
    }
    print(f'    Test acc: {results["a_raw_std"]["test_acc_mean"]:.4f} ± {results["a_raw_std"]["test_acc_std"]:.4f}')
    
    # ========================================================================
    # Experiment (b): Restricted → StandardScaler → Standard MLP
    # ========================================================================
    print('  Exp (b): Restricted → StandardScaler → Standard MLP')
    
    test_accs_b = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Scale
        scaler_b = StandardScaler().fit(U[train_idx])
        U_train = torch.from_numpy(scaler_b.transform(U[train_idx])).float().to(device)
        U_val = torch.from_numpy(scaler_b.transform(U[val_idx])).float().to(device)
        U_test = torch.from_numpy(scaler_b.transform(U[test_idx])).float().to(device)
        
        # Train
        train_loader = DataLoader(TensorDataset(U_train, y_train),
                                  batch_size=batch_size, shuffle=True)
        model = StandardMLP(U.shape[1], hidden_dim, num_classes)
        
        test_acc, _, _ = train_model_simple(
            model, train_loader, U_val, y_val, U_test, y_test,
            epochs, lr, weight_decay, device
        )
        test_accs_b.append(test_acc)
    
    results['b_restricted_std'] = {
        'test_acc_mean': float(np.mean(test_accs_b)),
        'test_acc_std': float(np.std(test_accs_b)),
    }
    print(f'    Test acc: {results["b_restricted_std"]["test_acc_mean"]:.4f} ± {results["b_restricted_std"]["test_acc_std"]:.4f}')
    
    # ========================================================================
    # Experiment (c): Restricted → RowNorm MLP
    # ========================================================================
    print('  Exp (c): Restricted → RowNorm MLP')
    
    test_accs_c = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # No scaling for RowNorm
        U_train = torch.from_numpy(U[train_idx]).float().to(device)
        U_val = torch.from_numpy(U[val_idx]).float().to(device)
        U_test = torch.from_numpy(U[test_idx]).float().to(device)
        
        # Train
        train_loader = DataLoader(TensorDataset(U_train, y_train),
                                  batch_size=batch_size, shuffle=True)
        model = RowNormMLP(U.shape[1], hidden_dim, num_classes)
        
        test_acc, _, _ = train_model_simple(
            model, train_loader, U_val, y_val, U_test, y_test,
            epochs, lr, weight_decay, device
        )
        test_accs_c.append(test_acc)
    
    results['c_restricted_rownorm'] = {
        'test_acc_mean': float(np.mean(test_accs_c)),
        'test_acc_std': float(np.std(test_accs_c)),
    }
    print(f'    Test acc: {results["c_restricted_rownorm"]["test_acc_mean"]:.4f} ± {results["c_restricted_rownorm"]["test_acc_std"]:.4f}')
    
    return results

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/7] Loading {DATASET_NAME}...')

(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

if X_raw is None:
    print(f"ERROR: Dataset {DATASET_NAME} has no node features!")
    sys.exit(1)

d_raw = X_raw.shape[1]
print(f'Nodes: {num_nodes}, Classes: {num_classes}, Features: {d_raw}')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print(f'\n[2/7] Building graph matrices...')

adj, D, L = build_graph_matrices(edge_index, num_nodes)

print(f'Edges: {adj.nnz // 2}')  # Undirected, so divide by 2

# Compute normalized adjacency (diffusion operator)
print('Computing normalized adjacency: A = D^(-1/2) Adj D^(-1/2)...')
A_norm = compute_normalized_adjacency(adj, D)
print(f'A_norm: {A_norm.shape}, nnz: {A_norm.nnz}')

# ============================================================================
# 3. Determine dimension and batch size
# ============================================================================
dimension = d_raw
print(f'\nUsing dimension d = {dimension} (same as raw features)')

# Adaptive batch size
num_train_approx = int(0.6 * num_nodes)
batch_size = 128 if num_train_approx > 256 else num_train_approx
print(f'Batch size: {batch_size}')

# ============================================================================
# 4. Baseline: Engineered Features (Reference)
# ============================================================================
print(f'\n[3/7] Computing baseline on ENGINEERED features X...')

baseline_results = []

for split_idx in range(NUM_RANDOM_SPLITS):
    print(f'\nSplit {split_idx+1}/{NUM_RANDOM_SPLITS}')
    
    train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
    print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
    
    # Compute restricted eigenvectors from engineered X
    print('Computing restricted eigenvectors from X...')
    U_engineered, evals, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_raw, L, D
    )
    
    features_dict = {
        'raw': X_raw,
        'restricted': U_engineered
    }
    
    # Run experiments
    result = run_three_experiments(
        features_dict, labels, train_idx, val_idx, test_idx,
        NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
        LEARNING_RATE, WEIGHT_DECAY, device
    )
    
    baseline_results.append(result)

# Aggregate baseline
baseline_aggregated = {}
for exp_name in ['a_raw_std', 'b_restricted_std', 'c_restricted_rownorm']:
    test_accs = [r[exp_name]['test_acc_mean'] for r in baseline_results]
    baseline_aggregated[exp_name] = {
        'mean': float(np.mean(test_accs)),
        'std': float(np.std(test_accs)),
    }

print('\n' + '='*70)
print('BASELINE RESULTS (Engineered X)')
print('='*70)
for exp_name, exp_label in [
    ('a_raw_std', '(a) X → StandardScaler → Standard MLP'),
    ('b_restricted_std', '(b) U → StandardScaler → Standard MLP'),
    ('c_restricted_rownorm', '(c) U → RowNorm MLP')
]:
    mean = baseline_aggregated[exp_name]['mean']
    std = baseline_aggregated[exp_name]['std']
    print(f'{exp_label}: {mean*100:.2f}% ± {std*100:.2f}%')

# ============================================================================
# 5. Random Features (No Diffusion) - Reference
# ============================================================================
print(f'\n[4/7] Computing RANDOM features (no diffusion)...')

random_no_diffusion = []

for subspace_idx in range(NUM_RANDOM_SUBSPACES):
    print(f'\nRandom subspace {subspace_idx+1}/{NUM_RANDOM_SUBSPACES}')
    
    # Generate random features
    R = np.random.randn(num_nodes, dimension).astype(np.float64)
    print(f'Generated R: {R.shape}')
    
    subspace_results = []
    
    for split_idx in range(NUM_RANDOM_SPLITS):
        train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
        
        # Compute restricted eigenvectors
        print(f'Split {split_idx+1}: Computing restricted eigenvectors...')
        U_random, evals, d_eff, ortho_err = compute_restricted_eigenvectors(
            R, L, D
        )
        
        features_dict = {
            'raw': R,
            'restricted': U_random
        }
        
        # Run experiments
        result = run_three_experiments(
            features_dict, labels, train_idx, val_idx, test_idx,
            NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
            LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        subspace_results.append(result)
    
    random_no_diffusion.append(subspace_results)

# Aggregate random (no diffusion)
random_aggregated = {}
for exp_name in ['a_raw_std', 'b_restricted_std', 'c_restricted_rownorm']:
    all_accs = []
    for subspace_results in random_no_diffusion:
        for result in subspace_results:
            all_accs.append(result[exp_name]['test_acc_mean'])
    
    random_aggregated[exp_name] = {
        'mean': float(np.mean(all_accs)),
        'std': float(np.std(all_accs)),
    }

print('\n' + '='*70)
print('RANDOM (No Diffusion) RESULTS')
print('='*70)
for exp_name, exp_label in [
    ('a_raw_std', '(a) R → StandardScaler → Standard MLP'),
    ('b_restricted_std', '(b) U_random → StandardScaler → Standard MLP'),
    ('c_restricted_rownorm', '(c) U_random → RowNorm MLP')
]:
    mean = random_aggregated[exp_name]['mean']
    std = random_aggregated[exp_name]['std']
    print(f'{exp_label}: {mean*100:.2f}% ± {std*100:.2f}%')

# ============================================================================
# 6. Graph-Diffused Random Features
# ============================================================================
print(f'\n[5/7] Computing GRAPH-DIFFUSED features (k={K_DIFFUSION})...')

diffused_results_all = []
diffusion_times = []

for subspace_idx in range(NUM_RANDOM_SUBSPACES):
    print(f'\n{"="*70}')
    print(f'DIFFUSED SUBSPACE {subspace_idx+1}/{NUM_RANDOM_SUBSPACES}')
    print(f'{"="*70}')
    
    # Generate random features
    R = np.random.randn(num_nodes, dimension).astype(np.float64)
    print(f'Generated R: {R.shape}')
    
    # Apply diffusion
    print(f'Applying {K_DIFFUSION}-step diffusion...')
    R_star, diff_time = apply_graph_diffusion(A_norm, R, K_DIFFUSION, verbose=True)
    diffusion_times.append(diff_time)
    
    print(f'R*: {R_star.shape}')
    print(f'Norm change: ||R||={np.linalg.norm(R):.2f} → ||R*||={np.linalg.norm(R_star):.2f}')
    
    subspace_results = []
    
    for split_idx in range(NUM_RANDOM_SPLITS):
        print(f'\nSplit {split_idx+1}/{NUM_RANDOM_SPLITS}')
        train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
        
        # Compute restricted eigenvectors from diffused features
        print('Computing restricted eigenvectors from R*...')
        U_diffused, evals, d_eff, ortho_err = compute_restricted_eigenvectors(
            R_star, L, D
        )
        
        features_dict = {
            'raw': R_star,
            'restricted': U_diffused
        }
        
        # Run experiments
        result = run_three_experiments(
            features_dict, labels, train_idx, val_idx, test_idx,
            NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
            LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        subspace_results.append(result)
    
    diffused_results_all.append(subspace_results)

# Aggregate diffused results
diffused_aggregated = {}
for exp_name in ['a_raw_std', 'b_restricted_std', 'c_restricted_rownorm']:
    all_accs = []
    for subspace_results in diffused_results_all:
        for result in subspace_results:
            all_accs.append(result[exp_name]['test_acc_mean'])
    
    diffused_aggregated[exp_name] = {
        'mean': float(np.mean(all_accs)),
        'std': float(np.std(all_accs)),
    }

print('\n' + '='*70)
print(f'GRAPH-DIFFUSED (k={K_DIFFUSION}) RESULTS')
print('='*70)
for exp_name, exp_label in [
    ('a_raw_std', '(a) R* → StandardScaler → Standard MLP'),
    ('b_restricted_std', '(b) U_diffused → StandardScaler → Standard MLP'),
    ('c_restricted_rownorm', '(c) U_diffused → RowNorm MLP')
]:
    mean = diffused_aggregated[exp_name]['mean']
    std = diffused_aggregated[exp_name]['std']
    print(f'{exp_label}: {mean*100:.2f}% ± {std*100:.2f}%')

avg_diff_time = np.mean(diffusion_times)
print(f'\nAverage diffusion time: {avg_diff_time:.2f}s')

# ============================================================================
# 7. Analysis and Comparison
# ============================================================================
print(f'\n[6/7] Comparative Analysis...')

print('\n' + '='*70)
print('KEY COMPARISONS')
print('='*70)

# Comparison 1: Diffusion effect on raw features (Exp a)
print('\n1. Effect of Diffusion on Raw Features [Exp (a)]:')
eng_a = baseline_aggregated['a_raw_std']['mean']
rand_a = random_aggregated['a_raw_std']['mean']
diff_a = diffused_aggregated['a_raw_std']['mean']

print(f'   Engineered X:       {eng_a*100:.2f}%')
print(f'   Random (no diff):   {rand_a*100:.2f}%')
print(f'   Diffused R*:        {diff_a*100:.2f}%')
print(f'   ')
print(f'   Diffusion improvement: {(diff_a - rand_a)*100:+.2f}pp')
print(f'   Gap to engineered:     {(eng_a - diff_a)*100:+.2f}pp')

if diff_a > rand_a + 0.05:
    print(f'   → ✓ Diffusion creates substantial improvement!')
elif diff_a > rand_a + 0.02:
    print(f'   → ✓ Diffusion helps moderately')
else:
    print(f'   → ✗ Diffusion has minimal effect')

# Comparison 2: RowNorm effect on diffused features (b vs c)
print('\n2. RowNorm Effect on Diffused Features [Exp (b) vs (c)]:')
diff_b = diffused_aggregated['b_restricted_std']['mean']
diff_c = diffused_aggregated['c_restricted_rownorm']['mean']
rownorm_effect = (diff_c - diff_b) / diff_b * 100

print(f'   Standard MLP on U_diffused: {diff_b*100:.2f}%')
print(f'   RowNorm MLP on U_diffused:  {diff_c*100:.2f}%')
print(f'   RowNorm effect:             {rownorm_effect:+.1f}%')

# Compare to random (no diffusion)
rand_b = random_aggregated['b_restricted_std']['mean']
rand_c = random_aggregated['c_restricted_rownorm']['mean']
rand_rownorm_effect = (rand_c - rand_b) / rand_b * 100

print(f'\n   For comparison, RowNorm on random (no diffusion): {rand_rownorm_effect:+.1f}%')

if rownorm_effect > 2 and rownorm_effect > rand_rownorm_effect:
    print(f'   → ✓ RowNorm WORKS on diffused features!')
else:
    print(f'   → ✗ RowNorm does not help diffused features')

# Comparison 3: Hierarchy
print('\n3. Feature Quality Hierarchy:')
print(f'   True Eigenvectors (Inv1):  [Not tested here]')
print(f'   Engineered X (baseline):   {eng_a*100:.2f}%')
print(f'   Diffused R* (new):         {diff_a*100:.2f}%')
print(f'   Random R (no structure):   {rand_a*100:.2f}%')

# ============================================================================
# 8. Save Results
# ============================================================================
print(f'\n[7/7] Saving results...')

results_dict = {
    'dataset': DATASET_NAME,
    'k_diffusion': K_DIFFUSION,
    'dimension': dimension,
    'num_random_subspaces': NUM_RANDOM_SUBSPACES,
    'num_splits': NUM_RANDOM_SPLITS,
    'num_seeds': NUM_SEEDS,
    'total_runs_per_exp': NUM_RANDOM_SUBSPACES * NUM_RANDOM_SPLITS * NUM_SEEDS,
    'avg_diffusion_time_seconds': float(avg_diff_time),
    'baseline_engineered': baseline_aggregated,
    'random_no_diffusion': random_aggregated,
    'graph_diffused': diffused_aggregated,
    'comparisons': {
        'diffusion_improvement_raw': float((diff_a - rand_a) * 100),
        'gap_to_engineered': float((eng_a - diff_a) * 100),
        'rownorm_effect_diffused': float(rownorm_effect),
        'rownorm_effect_random': float(rand_rownorm_effect),
    }
}

save_path = f'{output_base}/metrics/results_complete.json'
with open(save_path, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f'✓ Results saved: {save_path}')

# ============================================================================
# 9. Generate Summary Plot
# ============================================================================
print('\nGenerating summary plot...')

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

experiments = ['(a) Raw', '(b) Restricted\nStandard', '(c) Restricted\nRowNorm']
x_pos = np.arange(len(experiments))
width = 0.25

# Baseline (engineered)
baseline_vals = [
    baseline_aggregated['a_raw_std']['mean'],
    baseline_aggregated['b_restricted_std']['mean'],
    baseline_aggregated['c_restricted_rownorm']['mean']
]

# Random (no diffusion)
random_vals = [
    random_aggregated['a_raw_std']['mean'],
    random_aggregated['b_restricted_std']['mean'],
    random_aggregated['c_restricted_rownorm']['mean']
]

# Diffused
diffused_vals = [
    diffused_aggregated['a_raw_std']['mean'],
    diffused_aggregated['b_restricted_std']['mean'],
    diffused_aggregated['c_restricted_rownorm']['mean']
]

ax.bar(x_pos - width, baseline_vals, width, label='Engineered X', color='#2E7D32', alpha=0.8)
ax.bar(x_pos, random_vals, width, label='Random (no diffusion)', color='#C62828', alpha=0.8)
ax.bar(x_pos + width, diffused_vals, width, label=f'Diffused k={K_DIFFUSION}', color='#1565C0', alpha=0.8)

ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_xlabel('Experiment', fontsize=12)
ax.set_title(f'Graph Diffusion Effect - {DATASET_NAME}', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(experiments)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = f'{output_base}/plots/comparison_bar.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'✓ Plot saved: {plot_path}')
plt.close()

# ============================================================================
# Final Summary
# ============================================================================
print('\n' + '='*70)
print('EXPERIMENT COMPLETE')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion steps: {K_DIFFUSION}')
print(f'Total runs: {NUM_RANDOM_SUBSPACES * NUM_RANDOM_SPLITS * NUM_SEEDS} per experiment')
print(f'\nKey Finding:')
print(f'  Diffusion improvement: {(diff_a - rand_a)*100:+.2f}pp')
print(f'  RowNorm on diffused:   {rownorm_effect:+.1f}%')
print(f'\nResults saved to: {output_base}/')
print('='*70)
