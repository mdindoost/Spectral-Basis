"""
SGC with Truncated Restricted Eigenvectors
===========================================

Tests Yiannis's hypothesis: Truncate restricted eigenvectors to k=nclasses or 
k=2nclasses to remove noise and keep only signal.

For each dataset, tests:
1. SGC Baseline: X → A^k @ X → Logistic Regression (with bias)
2. Full eigenvectors (k=all): X → A^k @ X → Restricted Eigs (all d) → RowNorm → Classifier
3. Truncated (k=nclasses): X → A^k @ X → Restricted Eigs → Keep k=nclasses → RowNorm → Classifier
4. Truncated (k=2nclasses): X → A^k @ X → Restricted Eigs → Keep k=2nclasses → RowNorm → Classifier

Each variant tested with two classifiers:
- Logistic Regression (no bias)
- MLP (256 hidden, bias=False)

Usage:
    python experiments/investigation_sgc_truncated.py [dataset] --k_diffusion [values]
    
Examples:
    python experiments/investigation_sgc_truncated.py ogbn-arxiv
    python experiments/investigation_sgc_truncated.py cora --k_diffusion 2 4 8
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

parser = argparse.ArgumentParser(description='SGC with Truncated Restricted Eigenvectors')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 4, 6, 8, 10],
                   help='Diffusion propagation steps to test')
args = parser.parse_args()

DATASET_NAME = args.dataset
K_DIFFUSION_VALUES = args.k_diffusion

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
output_base = f'results/investigation_sgc_truncated/{DATASET_NAME}'
os.makedirs(f'{output_base}/plots', exist_ok=True)

print('='*80)
print('SGC WITH TRUNCATED RESTRICTED EIGENVECTORS')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'Truncation variants: [all, nclasses, 2nclasses]')
print(f'Classifiers: [Logistic, MLP]')
print(f'Random splits: {NUM_RANDOM_SPLITS}')
print(f'Seeds per split: {NUM_SEEDS}')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# Helper Functions
# ============================================================================

def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency"""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sgc_precompute(features, adj_normalized, degree, verbose=True):
    """SGC feature propagation"""
    start_time = time.time()
    
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    
    adj_normalized = adj_normalized.tocoo()
    indices = torch.LongTensor(np.vstack([adj_normalized.row, adj_normalized.col]))
    values = torch.FloatTensor(adj_normalized.data)
    shape = adj_normalized.shape
    adj_sparse = torch.sparse_coo_tensor(indices, values, shape)
    
    features_prop = features
    for i in range(degree):
        features_prop = torch.spmm(adj_sparse, features_prop)
        if verbose and (i + 1) % 2 == 0:
            print(f'  Propagation step {i+1}/{degree}')
    
    precompute_time = time.time() - start_time
    return features_prop.numpy(), precompute_time

def compute_restricted_eigenvectors(X, L, D, eps_base=1e-8):
    """Compute D-orthonormal restricted eigenvectors"""
    # QR decomposition
    Q, R = la.qr(X, mode='economic')
    rank_X = np.linalg.matrix_rank(R, tol=1e-10)
    
    if rank_X < X.shape[1]:
        print(f'  Rank deficiency: {rank_X}/{X.shape[1]}')
        Q = Q[:, :rank_X]
    
    d_effective = rank_X
    
    # Project Laplacian
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
    
    # Sort by eigenvalue (ascending)
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
    """SGC: Logistic regression with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)

class RowNormLogistic(nn.Module):
    """Logistic regression with row normalization, no bias"""
    def __init__(self, nfeat, nclass):
        super(RowNormLogistic, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=False)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.W(x)

# RowNormMLP imported from utils

# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, train_features, train_labels, val_features, val_labels,
                epochs, lr, weight_decay, device, use_scheduler=False):
    """Train model"""
    model = model.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    best_val_acc = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_features)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        
        if use_scheduler:
            scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_features)
            val_pred = val_output.argmax(dim=1)
            val_acc = (val_pred == val_labels).float().mean().item()
        model.train()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    train_time = time.time() - start_time
    return model, best_val_acc, train_time

def test_model(model, test_features, test_labels, device):
    """Evaluate model"""
    model.eval()
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        output = model(test_features)
        pred = output.argmax(dim=1)
        acc = (pred == test_labels).float().mean().item()
    
    return acc

def aggregate_results(results):
    """Aggregate results"""
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
    """SGC Baseline: Logistic regression on diffused features"""
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train = torch.FloatTensor(X_diffused[train_idx])
        X_val = torch.FloatTensor(X_diffused[val_idx])
        X_test = torch.FloatTensor(X_diffused[test_idx])
        y_train = torch.LongTensor(labels[train_idx])
        y_val = torch.LongTensor(labels[val_idx])
        y_test = torch.LongTensor(labels[test_idx])
        
        model = SGC(X_diffused.shape[1], num_classes)
        
        model, val_acc, train_time = train_model(
            model, X_train, y_train, X_val, y_val,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, use_scheduler=False
        )
        
        test_acc = test_model(model, X_test, y_test, device)
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results

def run_truncated_experiment(X_diffused, L, D, labels, train_idx, val_idx, test_idx,
                             num_classes, num_seeds, device, 
                             truncation='all', classifier='logistic'):
    """
    Run experiment with truncated restricted eigenvectors
    
    Args:
        truncation: 'all', 'nclasses', or '2nclasses'
        classifier: 'logistic' or 'mlp'
    """
    # Compute full restricted eigenvectors
    U_full, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D
    )
    
    # Truncate based on option
    if truncation == 'nclasses':
        k = num_classes
    elif truncation == '2nclasses':
        k = 2 * num_classes
    else:  # 'all'
        k = d_eff
    
    # Ensure k doesn't exceed available eigenvectors
    k = min(k, d_eff)
    
    U = U_full[:, :k]
    
    print(f'  Truncation: {truncation} → k={k} (from {d_eff} total)')
    print(f'  Ortho error: {ortho_err:.2e}')
    
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train = torch.FloatTensor(U[train_idx])
        X_val = torch.FloatTensor(U[val_idx])
        X_test = torch.FloatTensor(U[test_idx])
        y_train = torch.LongTensor(labels[train_idx])
        y_val = torch.LongTensor(labels[val_idx])
        y_test = torch.LongTensor(labels[test_idx])
        
        # Create model
        if classifier == 'logistic':
            model = RowNormLogistic(k, num_classes)
            use_scheduler = False
        elif classifier == 'mlp':
            model = RowNormMLP(k, HIDDEN_DIM, num_classes)
            use_scheduler = True
        else:
            raise ValueError(f'Unknown classifier: {classifier}')
        
        # Train
        model, val_acc, train_time = train_model(
            model, X_train, y_train, X_val, y_val,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, use_scheduler=use_scheduler
        )
        
        # Test
        test_acc = test_model(model, X_test, y_test, device)
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results, k, ortho_err

# ============================================================================
# Load Data
# ============================================================================

print('\n[1/5] Loading dataset...')
data = load_dataset(DATASET_NAME)
edge_index, features, labels, num_nodes, num_classes, _, _, _ = data

print(f'Nodes: {num_nodes}')
print(f'Classes: {num_classes}')
print(f'Features: {features.shape[1]}')

# ============================================================================
# Build Graph Matrices
# ============================================================================

print('\n[2/5] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print('✓ Built: adjacency, degree matrix, Laplacian')

A_sgc = compute_sgc_normalized_adjacency(adj)
print('✓ Computed SGC-style normalized adjacency')

# ============================================================================
# Run Experiments for Each K
# ============================================================================

all_results = {}

for k_diff in K_DIFFUSION_VALUES:
    print(f'\n{"="*80}')
    print(f'DIFFUSION K = {k_diff}')
    print(f'{"="*80}')
    
    # Create output directory
    output_k = f'{output_base}/k{k_diff}'
    os.makedirs(f'{output_k}/metrics', exist_ok=True)
    
    # ========================================================================
    # SGC Precompute
    # ========================================================================
    print(f'\n[3/5] SGC precompute: A^{k_diff} @ X')
    
    if sp.issparse(features):
        X_features = features.toarray()
    else:
        X_features = features.numpy() if isinstance(features, torch.Tensor) else features
    
    X_diffused, precompute_time = sgc_precompute(
        X_features, A_sgc, k_diff, verbose=True
    )
    print(f'✓ Diffused features: {X_diffused.shape}')
    print(f'✓ Time: {precompute_time:.2f}s')
    
    # ========================================================================
    # Run All Experiments Across Splits
    # ========================================================================
    
    # Storage for results
    experiments = {
        'sgc_baseline': [],
        'full_logistic': [],
        'full_mlp': [],
        'nclasses_logistic': [],
        'nclasses_mlp': [],
        '2nclasses_logistic': [],
        '2nclasses_mlp': []
    }
    
    metadata = {}
    
    for split_idx in range(NUM_RANDOM_SPLITS):
        print(f'\n{"="*80}')
        print(f'SPLIT {split_idx+1}/{NUM_RANDOM_SPLITS}')
        print(f'{"="*80}')
        
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
        print('\n[4/5] Experiment 1: SGC Baseline')
        sgc_results = run_sgc_baseline(
            X_diffused, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        experiments['sgc_baseline'].extend(sgc_results)
        
        sgc_agg = aggregate_results(sgc_results)
        print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2: Full Eigenvectors + Logistic
        print('\n[5/5] Experiment 2a: Full Eigenvectors + Logistic')
        full_log_results, k_full, ortho = run_truncated_experiment(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='all', classifier='logistic'
        )
        experiments['full_logistic'].extend(full_log_results)
        metadata['k_full'] = int(k_full)
        
        full_log_agg = aggregate_results(full_log_results)
        print(f'→ {full_log_agg["test_acc_mean"]*100:.2f}% ± {full_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 3: Full Eigenvectors + MLP
        print('\nExperiment 2b: Full Eigenvectors + MLP')
        full_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='all', classifier='mlp'
        )
        experiments['full_mlp'].extend(full_mlp_results)
        
        full_mlp_agg = aggregate_results(full_mlp_results)
        print(f'→ {full_mlp_agg["test_acc_mean"]*100:.2f}% ± {full_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 4: k=nclasses + Logistic
        print('\nExperiment 3a: Truncated (k=nclasses) + Logistic')
        nc_log_results, k_nc, _ = run_truncated_experiment(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='nclasses', classifier='logistic'
        )
        experiments['nclasses_logistic'].extend(nc_log_results)
        metadata['k_nclasses'] = int(k_nc)
        
        nc_log_agg = aggregate_results(nc_log_results)
        print(f'→ {nc_log_agg["test_acc_mean"]*100:.2f}% ± {nc_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 5: k=nclasses + MLP
        print('\nExperiment 3b: Truncated (k=nclasses) + MLP')
        nc_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='nclasses', classifier='mlp'
        )
        experiments['nclasses_mlp'].extend(nc_mlp_results)
        
        nc_mlp_agg = aggregate_results(nc_mlp_results)
        print(f'→ {nc_mlp_agg["test_acc_mean"]*100:.2f}% ± {nc_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 6: k=2nclasses + Logistic
        print('\nExperiment 4a: Truncated (k=2nclasses) + Logistic')
        nc2_log_results, k_2nc, _ = run_truncated_experiment(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='2nclasses', classifier='logistic'
        )
        experiments['2nclasses_logistic'].extend(nc2_log_results)
        metadata['k_2nclasses'] = int(k_2nc)
        
        nc2_log_agg = aggregate_results(nc2_log_results)
        print(f'→ {nc2_log_agg["test_acc_mean"]*100:.2f}% ± {nc2_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 7: k=2nclasses + MLP
        print('\nExperiment 4b: Truncated (k=2nclasses) + MLP')
        nc2_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='2nclasses', classifier='mlp'
        )
        experiments['2nclasses_mlp'].extend(nc2_mlp_results)
        
        nc2_mlp_agg = aggregate_results(nc2_mlp_results)
        print(f'→ {nc2_mlp_agg["test_acc_mean"]*100:.2f}% ± {nc2_mlp_agg["test_acc_std"]*100:.2f}%')
    
    # ========================================================================
    # Aggregate All Results
    # ========================================================================
    print(f'\n{"="*80}')
    print(f'FINAL RESULTS (k_diffusion={k_diff})')
    print(f'{"="*80}')
    
    final_results = {}
    for exp_name, exp_results in experiments.items():
        final_results[exp_name] = aggregate_results(exp_results)
    
    # Print summary
    print(f'\nSGC Baseline: {final_results["sgc_baseline"]["test_acc_mean"]*100:.2f}%')
    print(f'\nFull Eigenvectors (k={metadata["k_full"]}):')
    print(f'  Logistic: {final_results["full_logistic"]["test_acc_mean"]*100:.2f}%')
    print(f'  MLP:      {final_results["full_mlp"]["test_acc_mean"]*100:.2f}%')
    print(f'\nTruncated k={metadata["k_nclasses"]} (nclasses):')
    print(f'  Logistic: {final_results["nclasses_logistic"]["test_acc_mean"]*100:.2f}%')
    print(f'  MLP:      {final_results["nclasses_mlp"]["test_acc_mean"]*100:.2f}%')
    print(f'\nTruncated k={metadata["k_2nclasses"]} (2nclasses):')
    print(f'  Logistic: {final_results["2nclasses_logistic"]["test_acc_mean"]*100:.2f}%')
    print(f'  MLP:      {final_results["2nclasses_mlp"]["test_acc_mean"]*100:.2f}%')
    
    # Compute improvements over SGC
    sgc_acc = final_results["sgc_baseline"]["test_acc_mean"]
    improvements = {}
    for exp_name in experiments.keys():
        if exp_name != 'sgc_baseline':
            improvements[exp_name] = float((final_results[exp_name]["test_acc_mean"] - sgc_acc) * 100)
    
    print(f'\nImprovements over SGC:')
    print(f'  Full Logistic:      {improvements["full_logistic"]:+.2f}pp')
    print(f'  Full MLP:           {improvements["full_mlp"]:+.2f}pp')
    print(f'  nclasses Logistic:  {improvements["nclasses_logistic"]:+.2f}pp')
    print(f'  nclasses MLP:       {improvements["nclasses_mlp"]:+.2f}pp')
    print(f'  2nclasses Logistic: {improvements["2nclasses_logistic"]:+.2f}pp')
    print(f'  2nclasses MLP:      {improvements["2nclasses_mlp"]:+.2f}pp')
    
    # Save results
    results_dict = {
        'dataset': DATASET_NAME,
        'k_diffusion': k_diff,
        'num_splits': NUM_RANDOM_SPLITS,
        'num_seeds': NUM_SEEDS,
        'total_runs': NUM_RANDOM_SPLITS * NUM_SEEDS,
        'precompute_time': float(precompute_time),
        'metadata': metadata,
        'results': final_results,
        'improvements': improvements
    }
    
    save_path = f'{output_k}/metrics/results.json'
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f'\n✓ Results saved: {save_path}')
    all_results[k_diff] = results_dict

# ============================================================================
# Final Summary
# ============================================================================

print(f'\n{"="*80}')
print('EXPERIMENT COMPLETE')
print(f'{"="*80}')
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion k values tested: {K_DIFFUSION_VALUES}')
print(f'Truncation variants: [all, nclasses, 2nclasses]')
print(f'Classifiers: [Logistic, MLP]')
print(f'Total configurations: {len(K_DIFFUSION_VALUES)} × 7 = {len(K_DIFFUSION_VALUES) * 7}')
print(f'\nResults saved to: {output_base}/')
print(f'{"="*80}')
