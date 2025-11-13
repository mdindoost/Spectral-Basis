"""
SGC with Truncated Restricted Eigenvectors (FIXED for disconnected components)
===============================================================================

Tests Yiannis's hypothesis with PROPER handling of disconnected graph components.

Usage:
    python experiments/investigation_sgc_truncated_fixed.py [dataset] --k_diffusion [values]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import networkx as nx
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

parser = argparse.ArgumentParser(description='SGC with Truncated Restricted Eigenvectors (Fixed)')
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
output_base = f'results/investigation_sgc_truncated_fixed/{DATASET_NAME}'
os.makedirs(f'{output_base}/plots', exist_ok=True)

print('='*80)
print('SGC WITH TRUNCATED RESTRICTED EIGENVECTORS (FIXED)')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
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

def compute_restricted_eigenvectors_fixed(X, L, D, num_components, max_eigs=None, eps_base=1e-8):
    """
    Compute D-orthonormal restricted eigenvectors with PROPER disconnected component handling.
    
    Args:
        X: Feature matrix (n x d)
        L: Laplacian matrix (n x n)
        D: Degree matrix (n x n)
        num_components: Number of disconnected components (from NetworkX)
        max_eigs: Maximum number of eigenvectors to compute (None = all)
        eps_base: Regularization parameter
    
    Returns:
        U: (n x k_effective) D-orthonormal restricted eigenvectors (component eigenvectors removed)
        eigenvalues: (k_effective,) eigenvalue array (component eigenvalues removed)
        k_full: Full number of eigenvectors computed before dropping
        k_effective: Number of eigenvectors after dropping component eigenvectors
        ortho_error: D-orthonormality deviation
    """
    print(f'  Computing restricted eigenvectors (handling {num_components} components)...')
    
    # QR decomposition
    Q, R = la.qr(X, mode='economic')
    rank_X = np.linalg.matrix_rank(R, tol=1e-10)
    
    if rank_X < X.shape[1]:
        print(f'  Rank deficiency: {rank_X}/{X.shape[1]}')
        Q = Q[:, :rank_X]
    
    d_full = rank_X
    
    # Determine how many eigenvectors to compute
    if max_eigs is not None:
        k_compute = min(num_components + max_eigs, d_full)
    else:
        k_compute = d_full
    
    print(f'  Computing {k_compute} eigenvectors (ncc={num_components} + target={k_compute-num_components})...')
    
    # Project Laplacian into feature subspace
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize
    eps = eps_base * np.trace(D_r) / d_full
    D_r = D_r + eps * np.eye(d_full)
    
    # Solve generalized eigenproblem for k_compute eigenvectors
    eigenvalues_full, V_full = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue (ascending - smallest first)
    idx = np.argsort(eigenvalues_full)
    eigenvalues_full = eigenvalues_full[idx]
    V_full = V_full[:, idx]
    
    # Take first k_compute
    eigenvalues_full = eigenvalues_full[:k_compute]
    V_full = V_full[:, :k_compute]
    
    print(f'  Eigenvalue range (all {k_compute}): [{eigenvalues_full.min():.6e}, {eigenvalues_full.max():.6e}]')
    
    # Drop first num_components eigenvalues (disconnected components)
    if num_components > 0:
        print(f'  Dropping first {num_components} eigenvalues (component eigenvectors)...')
        eigenvalues = eigenvalues_full[num_components:]
        V = V_full[:, num_components:]
    else:
        eigenvalues = eigenvalues_full
        V = V_full
    
    k_effective = len(eigenvalues)
    print(f'  Using {k_effective} eigenvectors after dropping components')
    print(f'  Eigenvalue range (after dropping): [{eigenvalues.min():.6e}, {eigenvalues.max():.6e}]')
    
    # Map to node space
    U = Q @ V
    
    # Verify D-orthonormality
    DU = D @ U
    G = U.T @ DU
    ortho_error = np.abs(G - np.eye(k_effective)).max()
    
    return U.astype(np.float32), eigenvalues, k_compute, k_effective, ortho_error

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
    """SGC Baseline"""
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

def run_truncated_experiment(X_diffused, L, D, num_components, labels, 
                             train_idx, val_idx, test_idx,
                             num_classes, num_seeds, device, 
                             truncation='all', classifier='logistic'):
    """
    Run experiment with truncated restricted eigenvectors
    
    Args:
        truncation: 'all', 'nclasses', or '2nclasses'
        classifier: 'logistic' or 'mlp'
    """
    # Determine max_eigs based on truncation
    if truncation == 'nclasses':
        max_eigs = num_classes
    elif truncation == '2nclasses':
        max_eigs = 2 * num_classes
    else:  # 'all'
        max_eigs = None
    
    # Compute restricted eigenvectors (FIXED for components)
    U, eigenvalues, k_full, k_effective, ortho_err = compute_restricted_eigenvectors_fixed(
        X_diffused, L, D, num_components, max_eigs=max_eigs
    )
    
    # Further truncate if requested (after component removal)
    if truncation == 'nclasses':
        k = min(num_classes, k_effective)
    elif truncation == '2nclasses':
        k = min(2 * num_classes, k_effective)
    else:  # 'all'
        k = k_effective
    
    U_final = U[:, :k]
    
    print(f'  Truncation: {truncation} → k={k} (from {k_effective} after dropping {num_components} components)')
    print(f'  Ortho error: {ortho_err:.2e}')
    
    if ortho_err > 1e-4:
        print(f'  ⚠️  WARNING: Large orthonormality error!')
    
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train = torch.FloatTensor(U_final[train_idx])
        X_val = torch.FloatTensor(U_final[val_idx])
        X_test = torch.FloatTensor(U_final[test_idx])
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
print('✓ Built: adjacency, degree matrix, Laplacian')

# ============================================================================
# COUNT DISCONNECTED COMPONENTS (CRITICAL FIX!)
# ============================================================================

print('\n[3/6] Detecting disconnected components with NetworkX...')
G = nx.from_scipy_sparse_array(adj)
num_components = nx.number_connected_components(G)
print(f'✓ Number of connected components: {num_components}')

if num_components > 1:
    print(f'⚠️  Graph has {num_components} disconnected components!')
    print(f'   Will compute extra eigenvectors and drop first {num_components}')

# ============================================================================
# Compute SGC Normalized Adjacency
# ============================================================================

print('\n[4/6] Computing SGC-style normalized adjacency...')
A_sgc = compute_sgc_normalized_adjacency(adj)
print('✓ A_sgc = D^(-1/2) (A + I) D^(-1/2)')

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
    
    # SGC Precompute
    print(f'\n[5/6] SGC precompute: A^{k_diff} @ X')
    
    if sp.issparse(features):
        X_features = features.toarray()
    else:
        X_features = features.numpy() if isinstance(features, torch.Tensor) else features
    
    X_diffused, precompute_time = sgc_precompute(
        X_features, A_sgc, k_diff, verbose=True
    )
    print(f'✓ Diffused features: {X_diffused.shape}')
    print(f'✓ Time: {precompute_time:.2f}s')
    
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
        print('\n[6/6] Experiment 1: SGC Baseline')
        sgc_results = run_sgc_baseline(
            X_diffused, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        experiments['sgc_baseline'].extend(sgc_results)
        
        sgc_agg = aggregate_results(sgc_results)
        print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2: Full + Logistic
        print('\nExperiment 2a: Full Eigenvectors + Logistic')
        full_log_results, k_full, ortho = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='all', classifier='logistic'
        )
        experiments['full_logistic'].extend(full_log_results)
        metadata['k_full'] = k_full
        
        full_log_agg = aggregate_results(full_log_results)
        print(f'→ {full_log_agg["test_acc_mean"]*100:.2f}% ± {full_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 3: Full + MLP
        print('\nExperiment 2b: Full Eigenvectors + MLP')
        full_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='all', classifier='mlp'
        )
        experiments['full_mlp'].extend(full_mlp_results)
        
        full_mlp_agg = aggregate_results(full_mlp_results)
        print(f'→ {full_mlp_agg["test_acc_mean"]*100:.2f}% ± {full_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 4: k=nclasses + Logistic
        print('\nExperiment 3a: Truncated (k=nclasses) + Logistic')
        nc_log_results, k_nc, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='nclasses', classifier='logistic'
        )
        experiments['nclasses_logistic'].extend(nc_log_results)
        metadata['k_nclasses'] = k_nc
        
        nc_log_agg = aggregate_results(nc_log_results)
        print(f'→ {nc_log_agg["test_acc_mean"]*100:.2f}% ± {nc_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 5: k=nclasses + MLP
        print('\nExperiment 3b: Truncated (k=nclasses) + MLP')
        nc_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='nclasses', classifier='mlp'
        )
        experiments['nclasses_mlp'].extend(nc_mlp_results)
        
        nc_mlp_agg = aggregate_results(nc_mlp_results)
        print(f'→ {nc_mlp_agg["test_acc_mean"]*100:.2f}% ± {nc_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 6: k=2nclasses + Logistic
        print('\nExperiment 4a: Truncated (k=2nclasses) + Logistic')
        nc2_log_results, k_2nc, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='2nclasses', classifier='logistic'
        )
        experiments['2nclasses_logistic'].extend(nc2_log_results)
        metadata['k_2nclasses'] = k_2nc
        
        nc2_log_agg = aggregate_results(nc2_log_results)
        print(f'→ {nc2_log_agg["test_acc_mean"]*100:.2f}% ± {nc2_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 7: k=2nclasses + MLP
        print('\nExperiment 4b: Truncated (k=2nclasses) + MLP')
        nc2_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='2nclasses', classifier='mlp'
        )
        experiments['2nclasses_mlp'].extend(nc2_mlp_results)
        
        nc2_mlp_agg = aggregate_results(nc2_mlp_results)
        print(f'→ {nc2_mlp_agg["test_acc_mean"]*100:.2f}% ± {nc2_mlp_agg["test_acc_std"]*100:.2f}%')
    
    # Aggregate results
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
    
    # Compute improvements
    sgc_acc = final_results["sgc_baseline"]["test_acc_mean"]
    improvements = {}
    for exp_name in experiments.keys():
        if exp_name != 'sgc_baseline':
            improvements[exp_name] = (final_results[exp_name]["test_acc_mean"] - sgc_acc) * 100
    
    print(f'\nImprovements over SGC:')
    for exp_name, imp in improvements.items():
        print(f'  {exp_name:25}: {imp:+.2f}pp')
    
    # Save results
    results_dict = {
        'dataset': DATASET_NAME,
        'num_components': num_components,
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
print(f'Components: {num_components}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'\nResults saved to: {output_base}/')
print(f'{"="*80}')