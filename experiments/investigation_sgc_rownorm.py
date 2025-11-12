"""
SGC Comparison: RowNorm MLP Variant
====================================

Compare:
1. SGC Baseline: X → A^k @ X → Logistic Regression
2. RowNorm MLP: X → A^k @ X → Restricted Eigenvectors → RowNorm MLP

Usage:
    python experiments/investigation_sgc_rownorm.py [dataset] --k [2,4,8,16]
    
Examples:
    python experiments/investigation_sgc_rownorm.py ogbn-arxiv
    python experiments/investigation_sgc_rownorm.py cora --k 2 4 8
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    load_dataset, build_graph_matrices, RowNormMLP
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='SGC vs RowNorm MLP Comparison')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--k', nargs='+', type=int, default=[2, 4, 6, 8, 10],
                   help='Propagation steps to test')
args = parser.parse_args()

DATASET_NAME = args.dataset
K_VALUES = args.k

# Experimental parameters
NUM_RANDOM_SPLITS = 5
NUM_SEEDS = 5

# Training hyperparameters
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Output
output_base = f'results/investigation_sgc_rownorm/{DATASET_NAME}'
os.makedirs(f'{output_base}/plots', exist_ok=True)

print('='*70)
print('SGC VS ROWNORM MLP COMPARISON')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'K values: {K_VALUES}')
print(f'Random splits: {NUM_RANDOM_SPLITS}')
print(f'Seeds per split: {NUM_SEEDS}')
print(f'Total runs per experiment: {NUM_RANDOM_SPLITS * NUM_SEEDS}')
print(f'Device: {device}')
print('='*70)

# ============================================================================
# Helper Functions
# ============================================================================

def compute_sgc_normalized_adjacency(adj):
    """
    Compute SGC-style normalized adjacency: A_hat = D^(-1/2) A_hat D^(-1/2)
    where A_hat = A + I (self-loops already added by build_graph_matrices)
    """
    adj = sp.coo_matrix(adj)
    
    # Compute degree matrix
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Symmetric normalization
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sgc_precompute(features, adj_normalized, degree, verbose=True):
    """
    SGC feature propagation: X_tilde = A_hat^K @ X
    """
    start_time = time.time()
    
    # Convert to torch tensor
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    
    # Convert sparse matrix to torch sparse tensor
    adj_normalized = adj_normalized.tocoo()
    indices = torch.LongTensor(np.vstack([adj_normalized.row, adj_normalized.col]))
    values = torch.FloatTensor(adj_normalized.data)
    shape = adj_normalized.shape
    adj_sparse = torch.sparse_coo_tensor(indices, values, shape)
    
    # Iterative propagation
    features_prop = features
    for i in range(degree):
        features_prop = torch.spmm(adj_sparse, features_prop)
        if verbose and (i + 1) % 2 == 0:
            print(f'  Propagation step {i+1}/{degree} complete')
    
    precompute_time = time.time() - start_time
    return features_prop.numpy(), precompute_time

def compute_restricted_eigenvectors(X, L, D, eps_base=1e-8):
    """
    Compute D-orthonormal restricted eigenvectors from features X
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

# ============================================================================
# Models
# ============================================================================

class SGC(nn.Module):
    """SGC: Single linear layer (logistic regression) with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)

# RowNormMLP is imported from utils.py

# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, train_features, train_labels, val_features, val_labels,
                epochs, lr, weight_decay, device, model_type='sgc'):
    """
    Train model (SGC or RowNorm MLP)
    """
    model = model.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Use scheduler for RowNorm MLP (like Investigation 1)
    if model_type == 'rownorm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    best_val_acc = 0
    history = {'train_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Forward
        optimizer.zero_grad()
        output = model(train_features)
        loss = criterion(output, train_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if model_type == 'rownorm':
            scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_features)
            val_pred = val_output.argmax(dim=1)
            val_acc = (val_pred == val_labels).float().mean().item()
        model.train()
        
        history['train_loss'].append(loss.item())
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    train_time = time.time() - start_time
    return model, best_val_acc, train_time, history

def test_model(model, test_features, test_labels, device):
    """Evaluate model on test set"""
    model.eval()
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        output = model(test_features)
        pred = output.argmax(dim=1)
        acc = (pred == test_labels).float().mean().item()
    
    return acc

def aggregate_results(results):
    """Aggregate results across multiple runs"""
    test_accs = [r['test_acc'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    train_times = [r['train_time'] for r in results]
    
    return {
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'val_acc_mean': float(np.mean(val_accs)),
        'val_acc_std': float(np.std(val_accs)),
        'train_time_mean': float(np.mean(train_times)),
        'train_time_std': float(np.std(train_times))
    }

# ============================================================================
# Main Experiment Functions
# ============================================================================

def run_sgc_baseline(X_diffused, labels, train_idx, val_idx, test_idx,
                     num_classes, num_seeds, device):
    """
    Experiment 1: SGC Baseline
    X_diffused → Logistic Regression (with bias)
    """
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepare data
        X_train = torch.FloatTensor(X_diffused[train_idx])
        X_val = torch.FloatTensor(X_diffused[val_idx])
        X_test = torch.FloatTensor(X_diffused[test_idx])
        y_train = torch.LongTensor(labels[train_idx])
        y_val = torch.LongTensor(labels[val_idx])
        y_test = torch.LongTensor(labels[test_idx])
        
        # Create model
        model = SGC(X_diffused.shape[1], num_classes)
        
        # Train
        model, val_acc, train_time, history = train_model(
            model, X_train, y_train, X_val, y_val,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, model_type='sgc'
        )
        
        # Test
        test_acc = test_model(model, X_test, y_test, device)
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results

def run_rownorm_mlp(X_diffused, L, D, labels, train_idx, val_idx, test_idx,
                    num_classes, num_seeds, device):
    """
    Experiment 2: RowNorm MLP Variant
    X_diffused → Restricted Eigenvectors → RowNorm MLP
    """
    # Compute restricted eigenvectors from diffused features
    print('  Computing restricted eigenvectors...')
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D
    )
    print(f'  Effective dimension: {d_eff}')
    print(f'  D-orthonormality error: {ortho_err:.2e}')
    
    if ortho_err > 1e-4:
        print(f'  WARNING: Large orthonormality error!')
    
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepare data (restricted eigenvectors)
        X_train = torch.FloatTensor(U[train_idx])
        X_val = torch.FloatTensor(U[val_idx])
        X_test = torch.FloatTensor(U[test_idx])
        y_train = torch.LongTensor(labels[train_idx])
        y_val = torch.LongTensor(labels[val_idx])
        y_test = torch.LongTensor(labels[test_idx])
        
        # Create RowNorm MLP (from utils.py)
        model = RowNormMLP(d_eff, HIDDEN_DIM, num_classes)
        
        # Train
        model, val_acc, train_time, history = train_model(
            model, X_train, y_train, X_val, y_val,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, model_type='rownorm'
        )
        
        # Test
        test_acc = test_model(model, X_test, y_test, device)
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results, d_eff, ortho_err

# ============================================================================
# Load Data
# ============================================================================

print('\n[1/6] Loading dataset...')
data = load_dataset(DATASET_NAME)
edge_index, features, labels, num_nodes, num_classes, _, _, _ = data

print(f'Nodes: {num_nodes}')
print(f'Classes: {num_classes}')
print(f'Features: {features.shape[1]}')

# ============================================================================
# Build Graph Matrices
# ============================================================================

print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('Built: adjacency (with self-loops), degree matrix, Laplacian')

# Compute SGC normalized adjacency
print('Computing SGC-style normalized adjacency...')
A_sgc = compute_sgc_normalized_adjacency(adj)
print('✓ A_sgc = D^(-1/2) (A + I) D^(-1/2)')

# ============================================================================
# Run Experiments for Each K
# ============================================================================

all_results = {}

for K in K_VALUES:
    print(f'\n{"="*70}')
    print(f'K = {K} (Propagation Steps)')
    print(f'{"="*70}')
    
    # Create output directory
    output_k = f'{output_base}/k{K}'
    os.makedirs(f'{output_k}/metrics', exist_ok=True)
    
    # ========================================================================
    # SGC Precompute
    # ========================================================================
    print(f'\n[3/6] SGC precompute: X_tilde = A_sgc^{K} @ X')
    
    if sp.issparse(features):
        X_features = features.toarray()
    else:
        X_features = features.numpy() if isinstance(features, torch.Tensor) else features
    
    X_diffused, precompute_time = sgc_precompute(
        X_features, A_sgc, K, verbose=True
    )
    print(f'Diffused features shape: {X_diffused.shape}')
    print(f'Precompute time: {precompute_time:.2f}s')
    
    # ========================================================================
    # Run Experiments Across Random Splits
    # ========================================================================
    
    sgc_results_all = []
    rownorm_results_all = []
    
    for split_idx in range(NUM_RANDOM_SPLITS):
        print(f'\n{"="*70}')
        print(f'Random Split {split_idx+1}/{NUM_RANDOM_SPLITS} (60/20/20)')
        print(f'{"="*70}')
        
        # Create random split
        np.random.seed(split_idx)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
        
        # Experiment 1: SGC Baseline
        print('\n[4/6] Experiment 1: SGC Baseline')
        print('X_diffused → Logistic Regression (with bias)')
        sgc_results = run_sgc_baseline(
            X_diffused, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        sgc_results_all.extend(sgc_results)
        
        sgc_agg = aggregate_results(sgc_results)
        print(f'SGC Test Acc: {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2: RowNorm MLP
        print('\n[5/6] Experiment 2: RowNorm MLP Variant')
        print('X_diffused → Restricted Eigenvectors → RowNorm MLP')
        rownorm_results, d_eff, ortho_err = run_rownorm_mlp(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        rownorm_results_all.extend(rownorm_results)
        
        rownorm_agg = aggregate_results(rownorm_results)
        print(f'RowNorm MLP Test Acc: {rownorm_agg["test_acc_mean"]*100:.2f}% ± {rownorm_agg["test_acc_std"]*100:.2f}%')
        
        # Comparison
        improvement = (rownorm_agg['test_acc_mean'] - sgc_agg['test_acc_mean']) * 100
        print(f'\nImprovement: {improvement:+.2f}pp')
    
    # ========================================================================
    # Aggregate Results
    # ========================================================================
    print(f'\n[6/6] Aggregating results for K={K}...')
    
    sgc_final = aggregate_results(sgc_results_all)
    rownorm_final = aggregate_results(rownorm_results_all)
    
    print(f'\n{"="*70}')
    print(f'FINAL RESULTS (K={K}, {len(sgc_results_all)} total runs)')
    print(f'{"="*70}')
    print(f'SGC Baseline:    {sgc_final["test_acc_mean"]*100:.2f}% ± {sgc_final["test_acc_std"]*100:.2f}%')
    print(f'RowNorm MLP:     {rownorm_final["test_acc_mean"]*100:.2f}% ± {rownorm_final["test_acc_std"]*100:.2f}%')
    
    improvement_final = (rownorm_final['test_acc_mean'] - sgc_final['test_acc_mean']) * 100
    relative_improvement = (rownorm_final['test_acc_mean'] / sgc_final['test_acc_mean'] - 1) * 100
    
    print(f'\nAbsolute Improvement: {improvement_final:+.2f}pp')
    print(f'Relative Improvement: {relative_improvement:+.2f}%')
    print(f'\nPrecompute time: {precompute_time:.2f}s')
    print(f'SGC train time: {sgc_final["train_time_mean"]:.2f}s ± {sgc_final["train_time_std"]:.2f}s')
    print(f'RowNorm train time: {rownorm_final["train_time_mean"]:.2f}s ± {rownorm_final["train_time_std"]:.2f}s')
    
    # Save results
    results_dict = {
        'dataset': DATASET_NAME,
        'k': K,
        'num_splits': NUM_RANDOM_SPLITS,
        'num_seeds': NUM_SEEDS,
        'total_runs': len(sgc_results_all),
        'precompute_time': float(precompute_time),
        'sgc_baseline': sgc_final,
        'rownorm_mlp': rownorm_final,
        'improvement': {
            'absolute_pp': float(improvement_final),
            'relative_percent': float(relative_improvement)
        }
    }
    
    save_path = f'{output_k}/metrics/results.json'
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f'\n✓ Results saved: {save_path}')
    all_results[K] = results_dict

# ============================================================================
# Generate Comparison Plots
# ============================================================================

if len(K_VALUES) > 1:
    print(f'\n{"="*70}')
    print('Generating comparison plots...')
    print(f'{"="*70}')
    
    k_vals = sorted(all_results.keys())
    sgc_accs = [all_results[k]['sgc_baseline']['test_acc_mean'] * 100 for k in k_vals]
    sgc_stds = [all_results[k]['sgc_baseline']['test_acc_std'] * 100 for k in k_vals]
    rownorm_accs = [all_results[k]['rownorm_mlp']['test_acc_mean'] * 100 for k in k_vals]
    rownorm_stds = [all_results[k]['rownorm_mlp']['test_acc_std'] * 100 for k in k_vals]
    improvements = [all_results[k]['improvement']['absolute_pp'] for k in k_vals]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(k_vals, sgc_accs, yerr=sgc_stds, marker='o', linewidth=2,
                 capsize=5, label='SGC Baseline', color='#1f77b4')
    ax1.errorbar(k_vals, rownorm_accs, yerr=rownorm_stds, marker='s', linewidth=2,
                 capsize=5, label='RowNorm MLP', color='#ff7f0e')
    ax1.set_xlabel('Propagation Steps (K)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title(f'SGC vs RowNorm MLP - {DATASET_NAME}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(k_vals)
    
    ax2.bar(k_vals, improvements, color='#2ca02c', alpha=0.7, width=0.4)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Propagation Steps (K)', fontsize=12)
    ax2.set_ylabel('Improvement (pp)', fontsize=12)
    ax2.set_title(f'RowNorm MLP Improvement Over SGC', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(k_vals)
    
    for k, imp in zip(k_vals, improvements):
        ax2.text(k, imp + 0.1 if imp > 0 else imp - 0.3, f'{imp:+.2f}',
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = f'{output_base}/plots/k_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'✓ Plot saved: {plot_path}')
    plt.close()

# ============================================================================
# Final Summary
# ============================================================================

print(f'\n{"="*70}')
print('EXPERIMENT COMPLETE')
print(f'{"="*70}')
print(f'Dataset: {DATASET_NAME}')
print(f'K values tested: {K_VALUES}')
print(f'Random splits: {NUM_RANDOM_SPLITS}')
print(f'Total runs per K: {NUM_SEEDS * NUM_RANDOM_SPLITS}')

if len(K_VALUES) > 1:
    print(f'\nBest K for SGC: {k_vals[np.argmax(sgc_accs)]} ({max(sgc_accs):.2f}%)')
    print(f'Best K for RowNorm MLP: {k_vals[np.argmax(rownorm_accs)]} ({max(rownorm_accs):.2f}%)')
    print(f'Maximum improvement: {max(improvements):+.2f}pp at K={k_vals[np.argmax(improvements)]}')

print(f'\nResults saved to: {output_base}/')
print(f'{"="*70}')
