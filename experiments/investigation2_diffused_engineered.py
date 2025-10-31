"""
Investigation 2: Diffused Engineered Features
==============================================

Purpose: Test whether diffusing engineered features X before computing restricted 
eigenvectors improves RowNorm MLP performance.

Two diffusion types:
1. Standard (low-pass): X_diff = A^k @ X
2. CMG (high-pass): X_cmg = A^(k+1) @ X - A^k @ X

Usage:
    # Run both diffusion types (default)
    python experiments/investigation2_diffused_engineered.py [dataset_name]
    
    # Run only standard diffusion
    python experiments/investigation2_diffused_engineered.py [dataset_name] --diffusion standard
    
    # Custom k values
    python experiments/investigation2_diffused_engineered.py ogbn-arxiv --k-standard 2,4,8 --k-cmg 4,8,16

Examples:
    python experiments/investigation2_diffused_engineered.py ogbn-arxiv
    python experiments/investigation2_diffused_engineered.py cora --diffusion standard
"""

import os
import sys
import json
import copy
import time
import argparse
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Investigation 2: Diffused Engineered Features')
parser.add_argument('dataset', type=str, nargs='?', default='ogbn-arxiv',
                    help='Dataset name (default: ogbn-arxiv)')
parser.add_argument('--diffusion', type=str, choices=['standard', 'cmg', 'both'], 
                    default='both', help='Diffusion type to run (default: both)')
parser.add_argument('--k-standard', type=str, default='2,4,8',
                    help='k values for standard diffusion (comma-separated, default: 2,4,8)')
parser.add_argument('--k-cmg', type=str, default='4,8,16',
                    help='k values for CMG diffusion (comma-separated, default: 4,8,16)')
args = parser.parse_args()

DATASET_NAME = args.dataset

# Parse k values
K_VALUES_STANDARD = [int(k) for k in args.k_standard.split(',')]
K_VALUES_CMG = [int(k) for k in args.k_cmg.split(',')]

# Experimental parameters (matching Investigation 2)
NUM_SEEDS = 3          # Was 5, now 3
NUM_RANDOM_SPLITS = 3  # Was 5, now 3

# Hyperparameters (matching previous experiments)
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Set device and seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
output_base = f'results/investigation2_diffused_engineered/{DATASET_NAME}'
os.makedirs(f'{output_base}/baseline/metrics', exist_ok=True)
os.makedirs(f'{output_base}/baseline/plots', exist_ok=True)
os.makedirs(f'{output_base}/summary', exist_ok=True)

print('='*70)
print('INVESTIGATION 2: DIFFUSED ENGINEERED FEATURES')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion types: {args.diffusion}')
if args.diffusion in ['standard', 'both']:
    print(f'Standard k values: {K_VALUES_STANDARD}')
if args.diffusion in ['cmg', 'both']:
    print(f'CMG k values: {K_VALUES_CMG}')
print(f'Random splits: {NUM_RANDOM_SPLITS}')
print(f'Seeds per split: {NUM_SEEDS}')
print(f'Total runs per experiment: {NUM_SEEDS * NUM_RANDOM_SPLITS}')
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
    """
    deg = np.array(D.diagonal())
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return A_norm

def apply_standard_diffusion(A_norm, X, k, verbose=False):
    """
    Apply standard k-step diffusion (low-pass filter): X_diff = A^k @ X
    
    Args:
        A_norm: Normalized adjacency matrix (sparse)
        X: Features (n x d)
        k: Number of diffusion steps
        verbose: Print progress
    
    Returns:
        X_diffused: Diffused features
        time_elapsed: Computation time
    """
    start_time = time.time()
    
    X_diffused = X.copy()
    for i in range(k):
        X_diffused = A_norm @ X_diffused
        if verbose and (i+1) % max(1, k//4) == 0:
            print(f'  Diffusion step {i+1}/{k}...')
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f'  Completed in {elapsed:.2f}s')
        print(f'  Norm change: ||X||={np.linalg.norm(X):.2f} → ||X_diff||={np.linalg.norm(X_diffused):.2f}')
    
    return X_diffused, elapsed

def apply_cmg_diffusion(A_norm, X, k, verbose=False):
    """
    Apply CMG diffusion (high-pass filter): X_cmg = A^(k+1) @ X - A^k @ X
    
    Args:
        A_norm: Normalized adjacency matrix (sparse)
        X: Features (n x d)
        k: Base diffusion steps
        verbose: Print progress
    
    Returns:
        X_cmg: CMG-diffused features
        time_elapsed: Computation time
    """
    start_time = time.time()
    
    # Compute A^k @ X
    X_k = X.copy()
    for i in range(k):
        X_k = A_norm @ X_k
        if verbose and (i+1) % max(1, k//4) == 0:
            print(f'  Diffusion step {i+1}/{k}...')
    
    # Compute A^(k+1) @ X
    if verbose:
        print(f'  Computing A^(k+1) @ X...')
    X_k_plus_1 = A_norm @ X_k
    
    # CMG: difference
    X_cmg = X_k_plus_1 - X_k
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f'  Completed in {elapsed:.2f}s')
        print(f'  Norm: ||X_cmg||={np.linalg.norm(X_cmg):.2f}')
    
    return X_cmg, elapsed

def compute_restricted_eigenvectors(X, L, D, eps_base=1e-8):
    """
    Compute restricted eigenvectors from features X
    
    Args:
        X: (n, d) feature matrix
        L: (n, n) Laplacian matrix
        D: (n, n) degree matrix
        eps_base: regularization base
    
    Returns:
        U: (n, d_eff) D-orthonormal restricted eigenvectors
        eigenvalues: (d_eff,) eigenvalue array
        d_effective: effective dimension after QR
        ortho_error: D-orthonormality deviation
    """
    # QR decomposition
    Q, R = la.qr(X, mode='economic')
    rank_X = np.linalg.matrix_rank(R, tol=1e-10)
    
    if rank_X < X.shape[1]:
        print(f'  Warning: Rank deficiency detected ({rank_X}/{X.shape[1]})')
        Q = Q[:, :rank_X]
    
    d_effective = rank_X
    
    # Project Laplacian into feature subspace
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize
    eps = eps_base * np.trace(D_r) / d_effective
    D_r = D_r + eps * np.eye(d_effective)
    
    # Solve generalized eigenproblem
    eigenvalues, V = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Map to node space
    U = Q @ V
    
    # Verify D-orthonormality
    DU = D @ U
    G = U.T @ DU
    ortho_error = np.abs(G - np.eye(d_effective)).max()
    
    return U.astype(np.float32), eigenvalues, d_effective, ortho_error

def train_model_single_run(model, train_loader, X_val, y_val, X_test, y_test,
                           epochs, lr, weight_decay, device):
    """
    Train model for one random seed
    
    Returns:
        dict with test_acc, val_accs, train_losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    val_accs = []
    train_losses = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        val_accs.append(val_acc)
        train_losses.append(total_loss / max(1, len(train_loader)))
    
    # Test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    return {
        'test_acc': test_acc,
        'val_accs': val_accs,
        'train_losses': train_losses
    }

def run_three_experiments(X_raw, U, labels, train_idx, val_idx, test_idx,
                          num_seeds, epochs, hidden_dim, batch_size,
                          lr, weight_decay, device, num_classes):
    """
    Run experiments (a), (b), (c) for given features
    
    Returns:
        dict with results for all three experiments
    """
    results_all_seeds = {
        'a_raw_std': [],
        'b_restricted_std': [],
        'c_restricted_rownorm': []
    }
    
    for seed in range(num_seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        
        # Prepare labels
        y_train = torch.from_numpy(labels[train_idx]).long().to(device)
        y_val = torch.from_numpy(labels[val_idx]).long().to(device)
        y_test = torch.from_numpy(labels[test_idx]).long().to(device)
        
        # ========== Experiment (a): X_raw → StandardScaler → Standard MLP ==========
        scaler_a = StandardScaler().fit(X_raw[train_idx])
        X_train_a = torch.from_numpy(scaler_a.transform(X_raw[train_idx])).float().to(device)
        X_val_a = torch.from_numpy(scaler_a.transform(X_raw[val_idx])).float().to(device)
        X_test_a = torch.from_numpy(scaler_a.transform(X_raw[test_idx])).float().to(device)
        
        train_loader_a = DataLoader(TensorDataset(X_train_a, y_train), 
                                     batch_size=batch_size, shuffle=True)
        
        model_a = StandardMLP(X_raw.shape[1], hidden_dim, num_classes)
        result_a = train_model_single_run(model_a, train_loader_a, X_val_a, y_val, 
                                          X_test_a, y_test, epochs, lr, weight_decay, device)
        results_all_seeds['a_raw_std'].append(result_a)
        
        # ========== Experiment (b): U → StandardScaler → Standard MLP ==========
        scaler_b = StandardScaler().fit(U[train_idx])
        U_train_b = torch.from_numpy(scaler_b.transform(U[train_idx])).float().to(device)
        U_val_b = torch.from_numpy(scaler_b.transform(U[val_idx])).float().to(device)
        U_test_b = torch.from_numpy(scaler_b.transform(U[test_idx])).float().to(device)
        
        train_loader_b = DataLoader(TensorDataset(U_train_b, y_train),
                                     batch_size=batch_size, shuffle=True)
        
        model_b = StandardMLP(U.shape[1], hidden_dim, num_classes)
        result_b = train_model_single_run(model_b, train_loader_b, U_val_b, y_val,
                                          U_test_b, y_test, epochs, lr, weight_decay, device)
        results_all_seeds['b_restricted_std'].append(result_b)
        
        # ========== Experiment (c): U → RowNorm MLP ==========
        U_train_c = torch.from_numpy(U[train_idx]).float().to(device)
        U_val_c = torch.from_numpy(U[val_idx]).float().to(device)
        U_test_c = torch.from_numpy(U[test_idx]).float().to(device)
        
        train_loader_c = DataLoader(TensorDataset(U_train_c, y_train),
                                     batch_size=batch_size, shuffle=True)
        
        model_c = RowNormMLP(U.shape[1], hidden_dim, num_classes)
        result_c = train_model_single_run(model_c, train_loader_c, U_val_c, y_val,
                                          U_test_c, y_test, epochs, lr, weight_decay, device)
        results_all_seeds['c_restricted_rownorm'].append(result_c)
    
    # Aggregate across seeds
    aggregated = {}
    for exp_name, results_list in results_all_seeds.items():
        test_accs = [r['test_acc'] for r in results_list]
        aggregated[exp_name] = {
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
        }
    
    return aggregated

def aggregate_results(results_list):
    """
    Aggregate results across multiple splits
    
    Args:
        results_list: list of result dicts from run_three_experiments
    
    Returns:
        dict with aggregated statistics
    """
    aggregated = {}
    
    for exp_name in ['a_raw_std', 'b_restricted_std', 'c_restricted_rownorm']:
        all_accs = [r[exp_name]['test_acc_mean'] for r in results_list]
        aggregated[exp_name] = {
            'mean': float(np.mean(all_accs)),
            'std': float(np.std(all_accs)),
        }
    
    return aggregated

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

d_raw = X_raw.shape[1]
print(f'Nodes: {num_nodes:,}, Classes: {num_classes}, Features: {d_raw}')
print(f'Train: {len(train_idx_fixed):,}, Val: {len(val_idx_fixed):,}, Test: {len(test_idx_fixed):,}')

# Determine batch size
batch_size = 128 if len(train_idx_fixed) > 256 else len(train_idx_fixed)
print(f'Batch size: {batch_size}')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('Built: adjacency (with self-loops), degree matrix, Laplacian')

# Compute normalized adjacency for diffusion
print('Computing normalized adjacency A = D^(-1/2) Adj D^(-1/2)...')
A_norm = compute_normalized_adjacency(adj, D)
print(f'A_norm: {A_norm.shape}, nnz: {A_norm.nnz:,}')

# ============================================================================
# 3. Baseline (No Diffusion)
# ============================================================================
print('\n[3/6] Computing BASELINE (no diffusion)...')

# Compute restricted eigenvectors from raw X
print('Computing restricted eigenvectors from raw X...')
U_baseline, evals, d_eff, ortho_err = compute_restricted_eigenvectors(
                                                X_raw.astype(np.float64), L, D
)
print(f'  Effective dimension: {d_eff}/{d_raw}')
print(f'  D-orthonormality error: {ortho_err:.2e}')
    
baseline_results_all_splits = []

for split_idx in range(NUM_RANDOM_SPLITS):
    print(f'\nSplit {split_idx+1}/{NUM_RANDOM_SPLITS}')
    train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
    
    # Run experiments
    result = run_three_experiments(
        X_raw, U_baseline, labels, train_idx, val_idx, test_idx,
        NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
        LEARNING_RATE, WEIGHT_DECAY, device, num_classes
    )
    
    baseline_results_all_splits.append(result)

# Aggregate baseline
baseline_aggregated = aggregate_results(baseline_results_all_splits)

print('\n' + '='*70)
print('BASELINE RESULTS (No Diffusion)')
print('='*70)
for exp_name, exp_label in [
    ('a_raw_std', '(a) X → StandardScaler → Standard MLP'),
    ('b_restricted_std', '(b) U → StandardScaler → Standard MLP'),
    ('c_restricted_rownorm', '(c) U → RowNorm MLP')
]:
    mean = baseline_aggregated[exp_name]['mean']
    std = baseline_aggregated[exp_name]['std']
    print(f'{exp_label}: {mean*100:.2f}% ± {std*100:.2f}%')

# Save baseline
baseline_save = {
    'dataset': DATASET_NAME,
    'num_seeds': NUM_SEEDS,
    'num_splits': NUM_RANDOM_SPLITS,
    'results': baseline_aggregated
}
with open(f'{output_base}/baseline/metrics/results.json', 'w') as f:
    json.dump(baseline_save, f, indent=2)

# ============================================================================
# 4. Standard Diffusion Experiments
# ============================================================================

standard_results = {}

if args.diffusion in ['standard', 'both']:
    print('\n[4/6] Running STANDARD DIFFUSION experiments...')
    
    for k in K_VALUES_STANDARD:
        print(f'\n{"="*70}')
        print(f'STANDARD DIFFUSION: k={k}')
        print(f'{"="*70}')
        
        # Create output directory
        output_k = f'{output_base}/standard_diffusion/k{k}'
        os.makedirs(f'{output_k}/metrics', exist_ok=True)
        
        diffusion_times = []
        # Apply standard diffusion
        print(f'Applying {k}-step standard diffusion...')
        X_diffused, diff_time = apply_standard_diffusion(
                A_norm, X_raw.astype(np.float64), k, verbose=True
            )
        diffusion_times.append(diff_time)

        # Compute restricted eigenvectors from diffused X
        print('Computing restricted eigenvectors from X_diffused...')
        U_diffused, evals, d_eff, ortho_err = compute_restricted_eigenvectors(
                X_diffused, L, D
        )
        print(f'  Effective dimension: {d_eff}/{d_raw}')
        print(f'  D-orthonormality error: {ortho_err:.2e}')
            
        results_all_splits = []
        
        
        for split_idx in range(NUM_RANDOM_SPLITS):
            print(f'\nSplit {split_idx+1}/{NUM_RANDOM_SPLITS}')
            train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
            
         
            # Run experiments
            result = run_three_experiments(
                X_diffused.astype(np.float32), U_diffused, labels, 
                train_idx, val_idx, test_idx,
                NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
                LEARNING_RATE, WEIGHT_DECAY, device, num_classes
            )
            
            results_all_splits.append(result)
        
        # Aggregate
        aggregated = aggregate_results(results_all_splits)
        standard_results[k] = aggregated
        
        avg_time = np.mean(diffusion_times)
        
        print(f'\n{"="*70}')
        print(f'STANDARD DIFFUSION k={k} RESULTS')
        print(f'{"="*70}')
        for exp_name, exp_label in [
            ('a_raw_std', '(a) X_diff → StandardScaler → Standard MLP'),
            ('b_restricted_std', '(b) U_diff → StandardScaler → Standard MLP'),
            ('c_restricted_rownorm', '(c) U_diff → RowNorm MLP')
        ]:
            mean = aggregated[exp_name]['mean']
            std = aggregated[exp_name]['std']
            print(f'{exp_label}: {mean*100:.2f}% ± {std*100:.2f}%')
        print(f'Average diffusion time: {avg_time:.2f}s')
        
        # Save
        save_data = {
            'dataset': DATASET_NAME,
            'diffusion_type': 'standard',
            'k': k,
            'num_seeds': NUM_SEEDS,
            'num_splits': NUM_RANDOM_SPLITS,
            'avg_diffusion_time': float(avg_time),
            'results': aggregated
        }
        with open(f'{output_k}/metrics/results.json', 'w') as f:
            json.dump(save_data, f, indent=2)

# ============================================================================
# 5. CMG Diffusion Experiments
# ============================================================================

cmg_results = {}

if args.diffusion in ['cmg', 'both']:
    print('\n[5/6] Running CMG DIFFUSION experiments...')
    
    for k in K_VALUES_CMG:
        print(f'\n{"="*70}')
        print(f'CMG DIFFUSION: k={k}')
        print(f'{"="*70}')
        
        # Create output directory
        output_k = f'{output_base}/cmg_diffusion/k{k}'
        os.makedirs(f'{output_k}/metrics', exist_ok=True)
        
        # Apply CMG diffusion
        print(f'Applying CMG diffusion (k={k})...')
        X_cmg, diff_time = apply_cmg_diffusion(
            A_norm, X_raw.astype(np.float64), k, verbose=True
        )
        diffusion_times.append(diff_time)
            
            
        # Compute restricted eigenvectors from CMG-diffused X
        print('Computing restricted eigenvectors from X_cmg...')
        U_cmg, evals, d_eff, ortho_err = compute_restricted_eigenvectors(
                X_cmg, L, D
        )
        print(f'  Effective dimension: {d_eff}/{d_raw}')
        print(f'  D-orthonormality error: {ortho_err:.2e}')
            
        
        results_all_splits = []
        diffusion_times = []
        
        for split_idx in range(NUM_RANDOM_SPLITS):
            print(f'\nSplit {split_idx+1}/{NUM_RANDOM_SPLITS}')
            train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
            

            
            # Run experiments
            result = run_three_experiments(
                X_cmg.astype(np.float32), U_cmg, labels,
                train_idx, val_idx, test_idx,
                NUM_SEEDS, EPOCHS, HIDDEN_DIM, batch_size,
                LEARNING_RATE, WEIGHT_DECAY, device, num_classes
            )
            
            results_all_splits.append(result)
        
        # Aggregate
        aggregated = aggregate_results(results_all_splits)
        cmg_results[k] = aggregated
        
        avg_time = np.mean(diffusion_times)
        
        print(f'\n{"="*70}')
        print(f'CMG DIFFUSION k={k} RESULTS')
        print(f'{"="*70}')
        for exp_name, exp_label in [
            ('a_raw_std', '(a) X_cmg → StandardScaler → Standard MLP'),
            ('b_restricted_std', '(b) U_cmg → StandardScaler → Standard MLP'),
            ('c_restricted_rownorm', '(c) U_cmg → RowNorm MLP')
        ]:
            mean = aggregated[exp_name]['mean']
            std = aggregated[exp_name]['std']
            print(f'{exp_label}: {mean*100:.2f}% ± {std*100:.2f}%')
        print(f'Average diffusion time: {avg_time:.2f}s')
        
        # Save
        save_data = {
            'dataset': DATASET_NAME,
            'diffusion_type': 'cmg',
            'k': k,
            'num_seeds': NUM_SEEDS,
            'num_splits': NUM_RANDOM_SPLITS,
            'avg_diffusion_time': float(avg_time),
            'results': aggregated
        }
        with open(f'{output_k}/metrics/results.json', 'w') as f:
            json.dump(save_data, f, indent=2)

# ============================================================================
# 6. Summary and Comparison
# ============================================================================
print('\n[6/6] Generating summary and comparison...')

print('\n' + '='*70)
print('COMPLETE RESULTS SUMMARY')
print('='*70)

# Baseline
print('\nBASELINE (No Diffusion):')
baseline_a = baseline_aggregated['a_raw_std']['mean']
baseline_c = baseline_aggregated['c_restricted_rownorm']['mean']
print(f'  (a) X → Standard MLP:        {baseline_a*100:.2f}%')
print(f'  (c) U → RowNorm MLP:         {baseline_c*100:.2f}%')
print(f'  RowNorm effect:              {(baseline_c - baseline_a)*100:+.2f}pp')

# Standard diffusion
if args.diffusion in ['standard', 'both'] and standard_results:
    print('\nSTANDARD DIFFUSION (Low-Pass Filter):')
    for k in K_VALUES_STANDARD:
        std_a = standard_results[k]['a_raw_std']['mean']
        std_c = standard_results[k]['c_restricted_rownorm']['mean']
        print(f'  k={k}:')
        print(f'    (a) X_diff → Standard:   {std_a*100:.2f}% ({(std_a - baseline_a)*100:+.2f}pp)')
        print(f'    (c) U_diff → RowNorm:    {std_c*100:.2f}% ({(std_c - baseline_c)*100:+.2f}pp)')
        print(f'    RowNorm effect:          {(std_c - std_a)*100:+.2f}pp')

# CMG diffusion
if args.diffusion in ['cmg', 'both'] and cmg_results:
    print('\nCMG DIFFUSION (High-Pass Filter):')
    for k in K_VALUES_CMG:
        cmg_a = cmg_results[k]['a_raw_std']['mean']
        cmg_c = cmg_results[k]['c_restricted_rownorm']['mean']
        print(f'  k={k}:')
        print(f'    (a) X_cmg → Standard:    {cmg_a*100:.2f}% ({(cmg_a - baseline_a)*100:+.2f}pp)')
        print(f'    (c) U_cmg → RowNorm:     {cmg_c*100:.2f}% ({(cmg_c - baseline_c)*100:+.2f}pp)')
        print(f'    RowNorm effect:          {(cmg_c - cmg_a)*100:+.2f}pp')

# Save complete summary
summary = {
    'dataset': DATASET_NAME,
    'baseline': baseline_aggregated,
    'standard_diffusion': standard_results if args.diffusion in ['standard', 'both'] else {},
    'cmg_diffusion': cmg_results if args.diffusion in ['cmg', 'both'] else {}
}

with open(f'{output_base}/summary/complete_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n✓ Complete results saved to: {output_base}/summary/complete_results.json')

# ============================================================================
# 7. Generate Comparison Plots
# ============================================================================

if args.diffusion in ['standard', 'both'] and standard_results:
    print('\nGenerating standard diffusion comparison plot...')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Baseline
    ax.axhline(y=baseline_c * 100, color='black', linestyle='--', 
               label='Baseline RowNorm', linewidth=2, alpha=0.7)
    
    # Standard diffusion results
    k_vals = sorted(standard_results.keys())
    accuracies = [standard_results[k]['c_restricted_rownorm']['mean'] * 100 for k in k_vals]
    
    ax.plot(k_vals, accuracies, marker='o', markersize=8, linewidth=2,
            label='Standard Diffusion + RowNorm', color='#1565C0')
    
    ax.set_xlabel('Diffusion Steps (k)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Standard Diffusion Effect - {DATASET_NAME}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_base}/summary/standard_diffusion_comparison.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {output_base}/summary/standard_diffusion_comparison.png')
    plt.close()

if args.diffusion in ['cmg', 'both'] and cmg_results:
    print('Generating CMG diffusion comparison plot...')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Baseline
    ax.axhline(y=baseline_c * 100, color='black', linestyle='--',
               label='Baseline RowNorm', linewidth=2, alpha=0.7)
    
    # CMG diffusion results
    k_vals = sorted(cmg_results.keys())
    accuracies = [cmg_results[k]['c_restricted_rownorm']['mean'] * 100 for k in k_vals]
    
    ax.plot(k_vals, accuracies, marker='s', markersize=8, linewidth=2,
            label='CMG Diffusion + RowNorm', color='#D32F2F')
    
    ax.set_xlabel('Diffusion Parameter (k)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'CMG Diffusion Effect - {DATASET_NAME}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_base}/summary/cmg_diffusion_comparison.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {output_base}/summary/cmg_diffusion_comparison.png')
    plt.close()

# ============================================================================
# Final Summary
# ============================================================================
print('\n' + '='*70)
print('EXPERIMENT COMPLETE')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Total runs: {NUM_SEEDS * NUM_RANDOM_SPLITS} per experiment')
if args.diffusion in ['standard', 'both']:
    print(f'Standard diffusion tested: k = {K_VALUES_STANDARD}')
if args.diffusion in ['cmg', 'both']:
    print(f'CMG diffusion tested: k = {K_VALUES_CMG}')
print(f'\nResults saved to: {output_base}/')
print('='*70)
