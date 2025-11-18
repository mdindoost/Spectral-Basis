"""
SGC with Truncated Restricted Eigenvectors (FIXED for disconnected components)
===============================================================================

Tests Yiannis's hypothesis with PROPER handling of disconnected graph components.

Updated: Added --splits and --component options per Yiannis's request

Usage:
    # Original (random splits, whole graph)
    python experiments/investigation_sgc_truncated_v2.py [dataset] --k_diffusion [values]
    
    # With fixed splits
    python experiments/investigation_sgc_truncated_v2.py [dataset] --k_diffusion 2 --splits fixed
    
    # With largest connected component
    python experiments/investigation_sgc_truncated_v2.py [dataset] --k_diffusion 2 --component lcc
    
    # Both options
    python experiments/investigation_sgc_truncated_v2.py [dataset] --k_diffusion 2 --splits fixed --component lcc
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

parser = argparse.ArgumentParser(description='SGC with Truncated Restricted Eigenvectors (Ver 2.0)')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 4, 6, 8, 10],
                   help='Diffusion propagation steps to test')
parser.add_argument('--splits', type=str, choices=['random', 'fixed'], default='random',
                   help='Use random or fixed splits')
parser.add_argument('--component', type=str, choices=['whole', 'lcc'], default='whole',
                   help='Use whole graph or largest connected component')
args = parser.parse_args()

DATASET_NAME = args.dataset
K_DIFFUSION_VALUES = args.k_diffusion
SPLIT_TYPE = args.splits
COMPONENT_TYPE = args.component

# Experimental parameters
NUM_RANDOM_SPLITS = 5 if SPLIT_TYPE == 'random' else 1
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
output_base = f'results/investigation_sgc_truncated_fixed/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}'
os.makedirs(f'{output_base}/plots', exist_ok=True)

print('='*80)
print('SGC WITH TRUNCATED RESTRICTED EIGENVECTORS (FIXED)')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# Helper Functions
# ============================================================================

def get_largest_connected_component_nx(adj):
    """Extract largest connected component using networkx (original method)"""
    # Convert to networkx graph
    G = nx.from_scipy_sparse_array(adj)
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    # Find largest
    largest_cc = max(components, key=len)
    lcc_nodes = sorted(list(largest_cc))
    
    # Create mask
    lcc_mask = np.zeros(adj.shape[0], dtype=bool)
    lcc_mask[lcc_nodes] = True
    
    print(f'\nConnected Components Analysis (NetworkX):')
    print(f'  Total components: {len(components)}')
    print(f'  Largest component size: {len(largest_cc)} nodes')
    print(f'  Total nodes: {adj.shape[0]}')
    print(f'  LCC percentage: {len(largest_cc)/adj.shape[0]*100:.2f}%')
    
    return lcc_mask

def extract_subgraph(adj, features, labels, mask, split_idx=None):
    """Extract subgraph for nodes in mask"""
    node_indices = np.where(mask)[0]
    
    # Create mapping from old to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
    
    # Extract adjacency (keep only edges within subgraph)
    adj_sub = adj[mask][:, mask]
    
    # Extract features and labels
    features_sub = features[mask]
    labels_sub = labels[mask]
    
    # Remap split indices if provided
    split_idx_sub = None
    if split_idx is not None:
        split_idx_sub = {}
        for split_name in ['train_idx', 'val_idx', 'test_idx']:
            if split_name in split_idx:
                old_indices = split_idx[split_name]
                # Only keep indices that are in the mask
                new_indices = []
                for old_idx in old_indices:
                    if mask[old_idx]:
                        new_indices.append(old_to_new[old_idx])
                split_idx_sub[split_name] = np.array(new_indices)
                
                print(f'  {split_name}: {len(old_indices)} -> {len(new_indices)} nodes')
    
    return adj_sub, features_sub, labels_sub, split_idx_sub

def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency"""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def sgc_precompute(features, adj_normalized, degree):
    """Apply SGC precomputation: (D^(-1/2) A D^(-1/2))^k X"""
    for i in range(degree):
        features = adj_normalized @ features
    return features

def compute_restricted_eigenvectors_fixed(X, L, D, num_components, max_eigs=None):
    """
    Compute restricted eigenvectors with FIXED component handling
    
    Args:
        X: Feature matrix
        L: Laplacian
        D: Degree matrix  
        num_components: Number of disconnected components (eigenvalues to drop)
        max_eigs: Maximum eigenvectors to compute (None = all)
    """
    num_nodes, dimension = X.shape
    
    # QR decomposition
    Q, R = np.linalg.qr(X)
    rank_X = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    if rank_X < dimension:
        Q = Q[:, :rank_X]
        dimension = rank_X
    
    # Determine how many to compute
    if max_eigs is None:
        k_compute = dimension + num_components
    else:
        k_compute = min(max_eigs + num_components, dimension + num_components)
    
    k_compute = min(k_compute, num_nodes - 1)
    
    # Project Laplacian
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize
    eps = 1e-10 * np.trace(D_r) / dimension
    D_r = D_r + eps * np.eye(dimension)
    
    # Solve generalized eigenproblem
    eigenvalues_full, V_full = la.eigh(L_r, D_r)
    
    # Sort
    idx = np.argsort(eigenvalues_full)
    eigenvalues_full = eigenvalues_full[idx]
    V_full = V_full[:, idx]
    
    # Take first k_compute
    eigenvalues_full = eigenvalues_full[:k_compute]
    V_full = V_full[:, :k_compute]
    
    # Drop first num_components eigenvalues
    if num_components > 0:
        eigenvalues = eigenvalues_full[num_components:]
        V = V_full[:, num_components:]
    else:
        eigenvalues = eigenvalues_full
        V = V_full
    
    k_effective = len(eigenvalues)
    
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
    """SGC baseline: simple logistic regression with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)

class RowNormLogistic(nn.Module):
    """Our variant: RowNorm + logistic regression, no bias"""
    def __init__(self, nfeat, nclass):
        super(RowNormLogistic, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=False)
        nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.W(x)

# ============================================================================
# Training Functions
# ============================================================================

def train_and_test(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                   epochs, lr, weight_decay, device, use_scheduler=False):
    """Train model and return validation + test accuracy"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    patience = 20
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if use_scheduler:
            scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_pred = val_output.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
            
            test_output = model(X_test)
            test_pred = test_output.argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    train_time = time.time() - start_time
    
    return best_val_acc, best_test_acc, train_time

def aggregate_results(results_list):
    """Aggregate results"""
    test_accs = [r['test_acc'] for r in results_list]
    return {
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'test_acc_min': float(np.min(test_accs)),
        'test_acc_max': float(np.max(test_accs))
    }

# ============================================================================
# Experiment Functions
# ============================================================================

def run_sgc_baseline(X_diffused, labels, train_idx, val_idx, test_idx,
                     num_classes, num_seeds, device):
    """Experiment 1: SGC Baseline"""
    results = []
    
    X_train = X_diffused[train_idx]
    y_train = labels[train_idx]
    X_val = X_diffused[val_idx]
    y_val = labels[val_idx]
    X_test = X_diffused[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = SGC(X_diffused.shape[1], num_classes).to(device)
        
        val_acc, test_acc, train_time = train_and_test(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, use_scheduler=False
        )
        
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
    """Run experiment with truncated restricted eigenvectors"""
    # Determine max_eigs based on truncation
    if truncation == 'nclasses':
        max_eigs = num_classes
    elif truncation == '2nclasses':
        max_eigs = 2 * num_classes
    else:  # 'all'
        max_eigs = None
    
    # Compute restricted eigenvectors
    U, eigenvalues, k_full, k_effective, ortho_err = compute_restricted_eigenvectors_fixed(
        X_diffused, L, D, num_components, max_eigs=max_eigs
    )
    
    # Further truncate if requested
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
    
    # Train models
    results = []
    
    X_train = U_final[train_idx]
    y_train = labels[train_idx]
    X_val = U_final[val_idx]
    y_val = labels[val_idx]
    X_test = U_final[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if classifier == 'logistic':
            model = RowNormLogistic(U_final.shape[1], num_classes).to(device)
        else:  # 'mlp'
            model = RowNormMLP(U_final.shape[1], HIDDEN_DIM, num_classes).to(device)
        
        val_acc, test_acc, train_time = train_and_test(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, use_scheduler=False
        )
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results, k, ortho_err

# ============================================================================
# Main Experiment
# ============================================================================

print('\n[1/6] Loading dataset...')
(edge_index, features_original, labels_original, num_nodes_dataset, num_classes_dataset,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Loaded: {num_nodes_dataset} nodes, {num_classes_dataset} classes')

# Package fixed splits
if train_idx_fixed is not None:
    split_idx_original = {
        'train_idx': train_idx_fixed,
        'val_idx': val_idx_fixed,
        'test_idx': test_idx_fixed
    }
else:
    split_idx_original = None

# ============================================================================
# Build Initial Graph Matrices (from original whole graph)
# ============================================================================

print('\n[2/6] Building graph matrices from edge_index...')
adj_from_edge, D_orig, L_orig = build_graph_matrices(edge_index, num_nodes_dataset)
print(f'Built adjacency matrix: {adj_from_edge.shape}')

# ============================================================================
# Apply Component Selection
# ============================================================================

if COMPONENT_TYPE == 'lcc':
    print('\n[3/6] Extracting largest connected component...')
    lcc_mask = get_largest_connected_component_nx(adj_from_edge)
    adj, features, labels, split_idx = extract_subgraph(
        adj_from_edge, features_original, labels_original, lcc_mask, split_idx_original
    )
    
    # Rebuild graph matrices for LCC
    print('Rebuilding graph matrices for LCC...')
    adj_coo = adj.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj_final, D, L = build_graph_matrices(edge_index_lcc, adj.shape[0])
    adj = adj_final
else:
    print('\n[3/6] Using whole graph...')
    adj = adj_from_edge
    features = features_original
    labels = labels_original
    split_idx = split_idx_original
    D = D_orig
    L = L_orig

num_nodes = features.shape[0]
num_classes = len(np.unique(labels))

print(f'\nDataset Info (after component selection):')
print(f'  Nodes: {num_nodes}')
print(f'  Features: {features.shape[1]}')
print(f'  Classes: {num_classes}')
print(f'  Edges: {adj.nnz // 2}')

# ============================================================================
# Check for fixed splits availability
# ============================================================================

if SPLIT_TYPE == 'fixed':
    if split_idx is None:
        print(f'\n❌ ERROR: Dataset {DATASET_NAME} does not have fixed splits!')
        print('Please use --splits random instead.')
        sys.exit(1)
    print('✓ Using fixed splits from dataset')

# Count components using NetworkX
print(f'\n[4/6] Analyzing graph connectivity...')
G = nx.from_scipy_sparse_array(adj)
num_components = nx.number_connected_components(G)
print(f'Connected components: {num_components}')

if num_components >= features.shape[1]:
    print(f'\n❌ ERROR: Too many components ({num_components}) for feature dimension ({features.shape[1]})')
    print('Cannot proceed - no eigenvectors would remain after dropping components')
    sys.exit(1)

# ============================================================================
# Run Experiments for Each k Value
# ============================================================================

all_results = {}

for k_diff in K_DIFFUSION_VALUES:
    print(f'\n{"="*80}')
    print(f'DIFFUSION k={k_diff}')
    print(f'{"="*80}')
    
    output_k = f'{output_base}/k{k_diff}'
    os.makedirs(f'{output_k}/metrics', exist_ok=True)
    
    # Apply SGC diffusion
    print('\n[5/6] Computing SGC-style normalized adjacency...')
    A_sgc = compute_sgc_normalized_adjacency(adj)
    print('✓ A_sgc = D^(-1/2) (A + I) D^(-1/2)')
    
    print(f'\n[6/6] Precomputing diffusion (k={k_diff})...')
    features_dense = features.toarray() if sp.issparse(features) else features
    
    start_time = time.time()
    X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diff)
    precompute_time = time.time() - start_time
    
    print(f'✓ Precomputation done in {precompute_time:.2f}s')
    print(f'  X_diffused shape: {X_diffused.shape}')
    
    # Initialize experiment storage
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
    
    print(f'\n[7/7] Running experiments...')
    
    for split_idx_iter in range(NUM_RANDOM_SPLITS):
        print(f'\n{"="*70}')
        print(f'SPLIT {split_idx_iter+1}/{NUM_RANDOM_SPLITS}')
        print(f'{"="*70}')
        
        # Get split indices
        if SPLIT_TYPE == 'fixed':
            train_idx = split_idx['train_idx']
            val_idx = split_idx['val_idx']
            test_idx = split_idx['test_idx']
        else:
            # Create random split
            np.random.seed(split_idx_iter)
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
        
        # Experiment 1: SGC Baseline
        print('\nExperiment 1: SGC Baseline')
        sgc_results = run_sgc_baseline(
            X_diffused, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        experiments['sgc_baseline'].extend(sgc_results)
        
        sgc_agg = aggregate_results(sgc_results)
        print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2a: Full + Logistic
        print('\nExperiment 2a: Full Eigenvectors + Logistic')
        full_log_results, k_full, ortho = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='all', classifier='logistic'
        )
        experiments['full_logistic'].extend(full_log_results)
        metadata['k_full'] = k_full
        
        full_log_agg = aggregate_results(full_log_results)
        print(f'→ {full_log_agg["test_acc_mean"]*100:.2f}% ± {full_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2b: Full + MLP
        print('\nExperiment 2b: Full Eigenvectors + MLP')
        full_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='all', classifier='mlp'
        )
        experiments['full_mlp'].extend(full_mlp_results)
        
        full_mlp_agg = aggregate_results(full_mlp_results)
        print(f'→ {full_mlp_agg["test_acc_mean"]*100:.2f}% ± {full_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 3a: k=nclasses + Logistic
        print('\nExperiment 3a: Truncated (k=nclasses) + Logistic')
        nc_log_results, k_nc, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='nclasses', classifier='logistic'
        )
        experiments['nclasses_logistic'].extend(nc_log_results)
        metadata['k_nclasses'] = k_nc
        
        nc_log_agg = aggregate_results(nc_log_results)
        print(f'→ {nc_log_agg["test_acc_mean"]*100:.2f}% ± {nc_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 3b: k=nclasses + MLP
        print('\nExperiment 3b: Truncated (k=nclasses) + MLP')
        nc_mlp_results, _, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='nclasses', classifier='mlp'
        )
        experiments['nclasses_mlp'].extend(nc_mlp_results)
        
        nc_mlp_agg = aggregate_results(nc_mlp_results)
        print(f'→ {nc_mlp_agg["test_acc_mean"]*100:.2f}% ± {nc_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 4a: k=2nclasses + Logistic
        print('\nExperiment 4a: Truncated (k=2nclasses) + Logistic')
        nc2_log_results, k_2nc, _ = run_truncated_experiment(
            X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device, truncation='2nclasses', classifier='logistic'
        )
        experiments['2nclasses_logistic'].extend(nc2_log_results)
        metadata['k_2nclasses'] = k_2nc
        
        nc2_log_agg = aggregate_results(nc2_log_results)
        print(f'→ {nc2_log_agg["test_acc_mean"]*100:.2f}% ± {nc2_log_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 4b: k=2nclasses + MLP
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
        'split_type': SPLIT_TYPE,
        'component_type': COMPONENT_TYPE,
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
print(f'Configuration: {SPLIT_TYPE} splits, {COMPONENT_TYPE} graph')
print(f'Components: {num_components}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'\nResults saved to: {output_base}/')
print(f'{"="*80}')