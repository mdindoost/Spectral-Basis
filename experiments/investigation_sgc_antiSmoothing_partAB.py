"""
Investigation: SGC with Global Anti-Smoothing
==============================================

New path:
- SGC uses diffusion but degrades at high k (over-smoothing)
- We ask: Does S = span(diffused_k(X)) contain more information that SGC doesn't exploit?
- Answer: Yes! Restricted eigenvectors give a better basis of S
- RowNorm (Radial classifier) unlocks this information
- Method IMPROVES with high k → "Global Anti-Smoothing"

Pipeline: X → diffusion_k → restricted eigenvectors → RowNorm/Radial classifier

New experiments:
1. SGC + MLP baseline (to show improvement isn't just from using MLP)
2. Johnson-Lindenstrauss random projections (better than truncation)
   - Binary {+1,-1} projection
   - Random orthogonal projection
3. Parameter efficiency analysis

Usage:
    # Random splits, whole graph
    python experiments/investigation_sgc_antiSmoothing.py [dataset] --k_diffusion 2 4 6 8 10
    
    # Fixed splits
    python experiments/investigation_sgc_antiSmoothing.py [dataset] --k_diffusion 2 4 6 8 10 --splits fixed
    
    # Largest connected component
    python experiments/investigation_sgc_antiSmoothing.py [dataset] --k_diffusion 2 4 6 8 10 --component lcc
    
Examples:
    python experiments/investigation_sgc_antiSmoothing.py ogbn-arxiv --k_diffusion 2 4 6 8 10 --splits fixed --component lcc
    python experiments/investigation_sgc_antiSmoothing.py wikics --k_diffusion 2 4 6 8 10
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    load_dataset, build_graph_matrices, StandardMLP, RowNormMLP
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='SGC with Global Anti-Smoothing')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 4, 6, 8, 10],
                   help='Diffusion propagation steps to test')
parser.add_argument('--splits', type=str, choices=['random', 'fixed'], default='random',
                   help='Use random or fixed splits')
parser.add_argument('--component', type=str, choices=['whole', 'lcc'], default='lcc',
                   help='Use whole graph or largest connected component')
args = parser.parse_args()

DATASET_NAME = args.dataset
K_DIFFUSION_VALUES = args.k_diffusion
SPLIT_TYPE = args.splits
COMPONENT_TYPE = args.component

# Experimental parameters
NUM_RANDOM_SPLITS = 5 if SPLIT_TYPE == 'random' else 1
NUM_SEEDS = 5

# Training hyperparameters (same as previous experiments)
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Output
output_base = f'results/investigation_sgc_antiSmoothing/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

print('='*80)
print('SGC WITH GLOBAL ANTI-SMOOTHING')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Device: {device}')
print('='*80)

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
# Helper Functions
# ============================================================================

def get_largest_connected_component_nx(adj):
    """Extract largest connected component using networkx"""
    G = nx.from_scipy_sparse_array(adj)
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    lcc_nodes = sorted(list(largest_cc))
    
    lcc_mask = np.zeros(adj.shape[0], dtype=bool)
    lcc_mask[lcc_nodes] = True
    
    print(f'\nConnected Components Analysis:')
    print(f'  Total components: {len(components)}')
    print(f'  Largest component size: {len(largest_cc)} nodes')
    print(f'  Total nodes: {adj.shape[0]}')
    print(f'  LCC percentage: {len(largest_cc)/adj.shape[0]*100:.2f}%')
    
    return lcc_mask

def extract_subgraph(adj, features, labels, mask, split_idx=None):
    """Extract subgraph for nodes in mask"""
    node_indices = np.where(mask)[0]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
    
    adj_sub = adj[mask][:, mask]
    features_sub = features[mask]
    labels_sub = labels[mask]
    
    split_idx_sub = None
    if split_idx is not None:
        split_idx_sub = {}
        for split_name in ['train_idx', 'val_idx', 'test_idx']:
            if split_name in split_idx:
                old_indices = split_idx[split_name]
                new_indices = []
                for old_idx in old_indices:
                    if mask[old_idx]:
                        new_indices.append(old_to_new[old_idx])
                split_idx_sub[split_name] = np.array(new_indices)
                print(f'  {split_name}: {len(old_indices)} -> {len(new_indices)} nodes')
    
    return adj_sub, features_sub, labels_sub, split_idx_sub

def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency: D^(-1/2) (A + I) D^(-1/2)"""
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops
    adj = sp.coo_matrix(adj)
    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def sgc_precompute(features, adj_normalized, degree):
    """Apply SGC precomputation: (D^(-1/2) (A+I) D^(-1/2))^k X"""
    for i in range(degree):
        features = adj_normalized @ features
    return features

def compute_restricted_eigenvectors(X, L, D, num_components=0):
    """
    Compute restricted eigenvectors with proper component handling
    
    Args:
        X: Feature matrix
        L: Laplacian
        D: Degree matrix
        num_components: Number of disconnected components (eigenvalues to drop)
    
    Returns:
        U: Restricted eigenvectors (D-orthonormal)
        eigenvalues: Eigenvalues
        d_effective: Effective dimension
        ortho_error: D-orthonormality error
    """
    num_nodes, dimension = X.shape
    
    # QR decomposition for rank handling
    Q, R = np.linalg.qr(X)
    rank_X = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    if rank_X < dimension:
        Q = Q[:, :rank_X]
        dimension = rank_X
    
    d_effective = dimension
    
    # Project Laplacian
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize
    eps_base = 1e-10
    eps = eps_base * np.trace(D_r) / d_effective
    D_r = D_r + eps * np.eye(d_effective)
    
    # Solve generalized eigenproblem
    eigenvalues, V = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Eigenvalue range check: normalized Laplacian eigenvalues must lie in [0, 2]
    # Violation means L and D were passed swapped to this function
    eig_min = float(eigenvalues.min())
    eig_max = float(eigenvalues.max())
    if eig_min < -0.01 or eig_max > 2.1:
        import warnings
        warnings.warn(
            f"EIGENVALUE ALARM: values out of [0, 2] range: "
            f"min={eig_min:.4f}, max={eig_max:.4f}. "
            f"L and D may be swapped!",
            stacklevel=2
        )

    # Drop component eigenvalues if needed
    if num_components > 0:
        eigenvalues = eigenvalues[num_components:]
        V = V[:, num_components:]
    
    # Map to node space
    U = Q @ V
    
    # Verify D-orthonormality
    DU = D @ U
    G = U.T @ DU
    ortho_error = np.abs(G - np.eye(U.shape[1])).max()
    
    return U.astype(np.float32), eigenvalues, U.shape[1], ortho_error

def random_binary_projection(Y, target_dim, seed=42):
    """
    Johnson-Lindenstrauss projection with binary {+1, -1} matrix
    
    Args:
        Y: Input matrix (n × d)
        target_dim: Target dimension d'
        seed: Random seed
    
    Returns:
        Y_compressed: (n × d')
    """
    np.random.seed(seed)
    d = Y.shape[1]
    
    # Random {+1, -1} matrix
    H = np.random.choice([1, -1], size=(d, target_dim)).astype(np.float32)
    H = H / np.sqrt(target_dim)  # Normalize for distance preservation
    
    Y_compressed = Y @ H
    return Y_compressed

def random_orthogonal_projection(Y, target_dim, seed=42):
    """
    Johnson-Lindenstrauss projection with random orthogonal matrix
    
    Args:
        Y: Input matrix (n × d)
        target_dim: Target dimension d'
        seed: Random seed
    
    Returns:
        Y_compressed: (n × d')
    """
    np.random.seed(seed)
    d = Y.shape[1]
    
    # Random Gaussian matrix
    H = np.random.randn(d, target_dim).astype(np.float32)
    
    # QR decomposition to make orthogonal
    H, _ = np.linalg.qr(H)
    
    Y_compressed = Y @ H
    return Y_compressed

def aggregate_results(results):
    """Aggregate results across seeds"""
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    train_times = [r['train_time'] for r in results]
    
    return {
        'val_acc_mean': float(np.mean(val_accs)),
        'val_acc_std': float(np.std(val_accs)),
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'train_time_mean': float(np.mean(train_times)),
        'train_time_std': float(np.std(train_times))
    }

def train_and_test(model, X_train, y_train, X_val, y_val, X_test, y_test,
                   epochs, lr, weight_decay, device, use_scheduler=False):
    """Train and test a model"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if use_scheduler:
            scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_pred = val_output.argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
                
                test_output = model(X_test_t)
                test_pred = test_output.argmax(dim=1)
                test_acc = (test_pred == y_test_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
    
    train_time = time.time() - start_time
    return best_val_acc, best_test_acc, train_time

# ============================================================================
# Experiment Functions
# ============================================================================

def run_sgc_baseline(X_diffused, labels, train_idx, val_idx, test_idx,
                     num_classes, num_seeds, device):
    """Experiment: SGC Baseline (Logistic Regression)"""
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

def run_sgc_mlp_baseline(X_diffused, labels, train_idx, val_idx, test_idx,
                         num_classes, num_seeds, device):
    """
    NEW EXPERIMENT: SGC + MLP Baseline
    Apply StandardScaler to diffused features, train Standard MLP
    """
    results = []
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_diffused[train_idx])
    X_val_scaled = scaler.transform(X_diffused[val_idx])
    X_test_scaled = scaler.transform(X_diffused[test_idx])
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = StandardMLP(X_diffused.shape[1], HIDDEN_DIM, num_classes).to(device)
        
        val_acc, test_acc, train_time = train_and_test(
            model, X_scaled, labels[train_idx], 
            X_val_scaled, labels[val_idx],
            X_test_scaled, labels[test_idx],
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, use_scheduler=False
        )
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results

def run_restricted_standard_mlp(X_diffused, L, D, num_components, labels,
                                train_idx, val_idx, test_idx, num_classes,
                                num_seeds, device):
    """
    NEW FOR PART A/B FRAMEWORK
    Restricted Eigenvectors + StandardScaler + Standard MLP
    """
    # Compute restricted eigenvectors
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D, num_components
    )
    
    # Apply StandardScaler
    scaler = StandardScaler()
    U_train_scaled = scaler.fit_transform(U[train_idx])
    U_val_scaled = scaler.transform(U[val_idx])
    U_test_scaled = scaler.transform(U[test_idx])
    
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = StandardMLP(d_eff, HIDDEN_DIM, num_classes).to(device)
        
        val_acc, test_acc, train_time = train_and_test(
            model, U_train_scaled, labels[train_idx],
            U_val_scaled, labels[val_idx],
            U_test_scaled, labels[test_idx],
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, use_scheduler=False
        )
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results, d_eff, ortho_err

def run_restricted_eigenvectors(X_diffused, L, D, num_components, labels,
                                train_idx, val_idx, test_idx, num_classes,
                                num_seeds, device, classifier='mlp'):
    """
    Experiment: Full Restricted Eigenvectors + RowNorm
    """
    # Compute restricted eigenvectors
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D, num_components
    )
    
    results = []
    
    X_train = U[train_idx]
    y_train = labels[train_idx]
    X_val = U[val_idx]
    y_val = labels[val_idx]
    X_test = U[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if classifier == 'logistic':
            model = RowNormLogistic(U.shape[1], num_classes).to(device)
        else:  # 'mlp'
            model = RowNormMLP(U.shape[1], HIDDEN_DIM, num_classes).to(device)
        
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
    
    return results, d_eff, ortho_err

def run_jl_projection(X_diffused, L, D, num_components, labels,
                     train_idx, val_idx, test_idx, num_classes,
                     num_seeds, device, target_dim, projection_type='binary'):
    """
    NEW EXPERIMENT: Johnson-Lindenstrauss Random Projection
    
    Args:
        projection_type: 'binary' or 'orthogonal'
        target_dim: Target dimension after projection
    """
    # Compute restricted eigenvectors
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D, num_components
    )
    
    # Apply JL projection
    if projection_type == 'binary':
        U_compressed = random_binary_projection(U, target_dim, seed=42)
    else:  # 'orthogonal'
        U_compressed = random_orthogonal_projection(U, target_dim, seed=42)
    
    results = []
    
    X_train = U_compressed[train_idx]
    y_train = labels[train_idx]
    X_val = U_compressed[val_idx]
    y_val = labels[val_idx]
    X_test = U_compressed[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = RowNormMLP(target_dim, HIDDEN_DIM, num_classes).to(device)
        
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
    
    return results, target_dim, ortho_err

# ============================================================================
# Main Experiment
# ============================================================================

print('\n[1/7] Loading dataset...')
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
# Build Graph Matrices
# ============================================================================

print('\n[2/7] Building graph matrices from edge_index...')
adj_from_edge, D_orig, L_orig = build_graph_matrices(edge_index, num_nodes_dataset)
print(f'Built adjacency matrix: {adj_from_edge.shape}')

# ============================================================================
# Apply Component Selection
# ============================================================================

if COMPONENT_TYPE == 'lcc':
    print('\n[3/7] Extracting largest connected component...')
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
    print('\n[3/7] Using whole graph...')
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

# Check for fixed splits
if SPLIT_TYPE == 'fixed':
    if split_idx is None:
        print(f'\n❌ ERROR: Dataset {DATASET_NAME} does not have fixed splits!')
        print('Please use --splits random instead.')
        sys.exit(1)
    print('✓ Using fixed splits from dataset')

# Count components
print(f'\n[4/7] Analyzing graph connectivity...')
G = nx.from_scipy_sparse_array(adj)
num_components = nx.number_connected_components(G)
print(f'Connected components: {num_components}')

if num_components >= features.shape[1]:
    print(f'\n❌ ERROR: Too many components ({num_components}) for feature dimension ({features.shape[1]})')
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
    print('\n[5/7] Computing SGC-style normalized adjacency...')
    A_sgc = compute_sgc_normalized_adjacency(adj)
    print('✓ A_sgc = D^(-1/2) (A + I) D^(-1/2)')
    
    print(f'\n[6/7] Precomputing diffusion (k={k_diff})...')
    features_dense = features.toarray() if sp.issparse(features) else features
    
    start_time = time.time()
    X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diff)
    precompute_time = time.time() - start_time
    
    print(f'✓ Precomputation done in {precompute_time:.2f}s')
    print(f'  X_diffused shape: {X_diffused.shape}')
    
    # Initialize experiment storage
    experiments = {
        'sgc_baseline': [],
        'sgc_mlp_baseline': [],  # NEW
        'restricted_standard_mlp': [],  # NEW FOR PART A/B
        'full_rownorm_mlp': [],
        'jl_binary_nc': [],      # NEW: JL binary, d'=nc
        'jl_binary_2nc': [],     # NEW: JL binary, d'=2nc
        'jl_binary_4nc': [],     # NEW: JL binary, d'=4nc
        'jl_ortho_nc': [],       # NEW: JL orthogonal, d'=nc
        'jl_ortho_2nc': [],      # NEW: JL orthogonal, d'=2nc
        'jl_ortho_4nc': [],      # NEW: JL orthogonal, d'=4nc
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
        print('\n[1/9] Experiment: SGC Baseline (Logistic)')
        sgc_results = run_sgc_baseline(
            X_diffused, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        experiments['sgc_baseline'].extend(sgc_results)
        sgc_agg = aggregate_results(sgc_results)
        print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2: SGC + MLP Baseline (NEW)
        print('\n[2/9] Experiment: SGC + MLP Baseline (NEW)')
        sgc_mlp_results = run_sgc_mlp_baseline(
            X_diffused, labels, train_idx, val_idx, test_idx,
            num_classes, NUM_SEEDS, device
        )
        experiments['sgc_mlp_baseline'].extend(sgc_mlp_results)
        sgc_mlp_agg = aggregate_results(sgc_mlp_results)
        print(f'→ {sgc_mlp_agg["test_acc_mean"]*100:.2f}% ± {sgc_mlp_agg["test_acc_std"]*100:.2f}%')
        
        # Experiment 2.5: Restricted + StandardScaler + MLP (NEW FOR PART A/B)
        print('\n[2.5/9] Experiment: Restricted Eigenvectors + StandardScaler + Standard MLP (NEW)')
        print('  (Bridge between Part A and Part B)')
        restricted_std_results, d_eff_std, ortho_err_std = run_restricted_standard_mlp(
            X_diffused, L, D, num_components, labels,
            train_idx, val_idx, test_idx, num_classes,
            NUM_SEEDS, device
        )
        experiments['restricted_standard_mlp'].extend(restricted_std_results)
        metadata['d_restricted_std'] = d_eff_std
        metadata['ortho_error_std'] = float(ortho_err_std)
        restricted_std_agg = aggregate_results(restricted_std_results)
        print(f'→ {restricted_std_agg["test_acc_mean"]*100:.2f}% ± {restricted_std_agg["test_acc_std"]*100:.2f}%')
        print(f'  D-orthonormality error: {ortho_err_std:.2e}')
        
        # Experiment 3: Full Restricted Eigenvectors + RowNorm MLP
        print('\n[3/9] Experiment: Full Restricted Eigenvectors + RowNorm MLP')
        full_results, d_eff, ortho_err = run_restricted_eigenvectors(
            X_diffused, L, D, num_components, labels,
            train_idx, val_idx, test_idx, num_classes,
            NUM_SEEDS, device, classifier='mlp'
        )
        experiments['full_rownorm_mlp'].extend(full_results)
        metadata['d_full'] = d_eff
        metadata['ortho_error'] = float(ortho_err)
        full_agg = aggregate_results(full_results)
        print(f'→ {full_agg["test_acc_mean"]*100:.2f}% ± {full_agg["test_acc_std"]*100:.2f}%')
        print(f'  D-orthonormality error: {ortho_err:.2e}')
        
        # JL Projection Experiments (NEW)
        target_dims = {
            'nc': num_classes,
            '2nc': 2 * num_classes,
            '4nc': 4 * num_classes
        }
        
        for dim_name, target_dim in target_dims.items():
            if target_dim > d_eff:
                print(f'\n[JL {dim_name}] Skipping: target_dim={target_dim} > d_eff={d_eff}')
                continue
            
            # Binary JL
            print(f'\n[4-6/9] Experiment: JL Binary Projection (d\'={dim_name}={target_dim})')
            jl_binary_results, _, _ = run_jl_projection(
                X_diffused, L, D, num_components, labels,
                train_idx, val_idx, test_idx, num_classes,
                NUM_SEEDS, device, target_dim, projection_type='binary'
            )
            exp_name = f'jl_binary_{dim_name}'
            experiments[exp_name].extend(jl_binary_results)
            jl_binary_agg = aggregate_results(jl_binary_results)
            print(f'→ {jl_binary_agg["test_acc_mean"]*100:.2f}% ± {jl_binary_agg["test_acc_std"]*100:.2f}%')
            
            # Orthogonal JL
            print(f'\n[7-9/9] Experiment: JL Orthogonal Projection (d\'={dim_name}={target_dim})')
            jl_ortho_results, _, _ = run_jl_projection(
                X_diffused, L, D, num_components, labels,
                train_idx, val_idx, test_idx, num_classes,
                NUM_SEEDS, device, target_dim, projection_type='orthogonal'
            )
            exp_name = f'jl_ortho_{dim_name}'
            experiments[exp_name].extend(jl_ortho_results)
            jl_ortho_agg = aggregate_results(jl_ortho_results)
            print(f'→ {jl_ortho_agg["test_acc_mean"]*100:.2f}% ± {jl_ortho_agg["test_acc_std"]*100:.2f}%')
    
    # ========================================================================
    # Aggregate and Save Results for this k
    # ========================================================================
    
    print(f'\n{"="*80}')
    print(f'AGGREGATED RESULTS FOR k={k_diff}')
    print(f'{"="*80}')
    
    final_results = {}
    for exp_name, exp_results in experiments.items():
        if len(exp_results) > 0:
            final_results[exp_name] = aggregate_results(exp_results)
    
    # Display results
    print('\nBaselines:')
    print(f'  SGC (Logistic):        {final_results["sgc_baseline"]["test_acc_mean"]*100:.2f}%')
    print(f'  SGC + MLP (NEW):       {final_results["sgc_mlp_baseline"]["test_acc_mean"]*100:.2f}%')
    
    print('\nOur Methods:')
    print(f'  Restricted + StandardMLP (Part A/B Bridge): {final_results["restricted_standard_mlp"]["test_acc_mean"]*100:.2f}%')
    print(f'  Restricted + RowNorm MLP:                   {final_results["full_rownorm_mlp"]["test_acc_mean"]*100:.2f}%')

    print('\nJL Binary Projections:')
    for dim_name in ['nc', '2nc', '4nc']:
        exp_name = f'jl_binary_{dim_name}'
        if exp_name in final_results:
            print(f'  d\'={dim_name:4}: {final_results[exp_name]["test_acc_mean"]*100:.2f}%')
    
    print('\nJL Orthogonal Projections:')
    for dim_name in ['nc', '2nc', '4nc']:
        exp_name = f'jl_ortho_{dim_name}'
        if exp_name in final_results:
            print(f'  d\'={dim_name:4}: {final_results[exp_name]["test_acc_mean"]*100:.2f}%')
    
    # Compute improvements
    sgc_acc = final_results["sgc_baseline"]["test_acc_mean"]
    sgc_mlp_acc = final_results["sgc_mlp_baseline"]["test_acc_mean"]
    
    improvements = {}
    for exp_name in final_results.keys():
        if exp_name not in ['sgc_baseline', 'sgc_mlp_baseline']:
            improvements[exp_name] = {
                'vs_sgc': (final_results[exp_name]["test_acc_mean"] - sgc_acc) * 100,
                'vs_sgc_mlp': (final_results[exp_name]["test_acc_mean"] - sgc_mlp_acc) * 100
            }
    
    print('\nImprovements:')
    for exp_name, imps in improvements.items():
        print(f'  {exp_name:25}: vs SGC: {imps["vs_sgc"]:+.2f}pp, vs SGC+MLP: {imps["vs_sgc_mlp"]:+.2f}pp')
    
    # Parameter count analysis
    feature_dim = X_diffused.shape[1]
    param_counts = {
        'sgc_baseline': feature_dim * num_classes,
        'sgc_mlp_baseline': feature_dim * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * num_classes,
        'restricted_standard_mlp': metadata.get('d_restricted_std', d_eff) * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * num_classes,  # ADD THIS
        'full_rownorm_mlp': d_eff * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * num_classes,
    }
    
    # Add JL param counts
    for dim_name, target_dim in target_dims.items():
        if target_dim <= d_eff:
            param_counts[f'jl_binary_{dim_name}'] = target_dim * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * num_classes
            param_counts[f'jl_ortho_{dim_name}'] = target_dim * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * num_classes
    
    print('\nParameter Counts:')
    for exp_name, count in param_counts.items():
        if exp_name in final_results:
            print(f'  {exp_name:25}: {count:,} parameters')
    
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
        'improvements': improvements,
        'parameter_counts': param_counts
    }
    
    save_path = f'{output_k}/metrics/results.json'
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f'\n✓ Results saved: {save_path}')
    all_results[k_diff] = results_dict

# ============================================================================
# Generate Summary Plots
# ============================================================================

print(f'\n{"="*80}')
print('GENERATING SUMMARY PLOTS')
print(f'{"="*80}')

# Plot 1: K-value trajectories for all methods
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

k_vals = sorted(all_results.keys())
colors = {
    'sgc_baseline': '#1f77b4',
    'sgc_mlp_baseline': '#ff7f0e',
    'restricted_standard_mlp': '#17becf',  # Part AB
    'full_rownorm_mlp': '#2ca02c',
    'jl_binary_2nc': '#d62728',
    'jl_ortho_2nc': '#9467bd'
}

labels = {
    'sgc_baseline': 'SGC (Logistic)',
    'sgc_mlp_baseline': 'SGC + MLP',
    'restricted_standard_mlp': 'Restricted + StandardMLP',  # Part AB
    'full_rownorm_mlp': 'Full + RowNorm MLP',
    'jl_binary_2nc': 'JL Binary (2nc)',
    'jl_ortho_2nc': 'JL Orthogonal (2nc)'
}

for exp_name in colors.keys():
    accs = []
    for k in k_vals:
        if exp_name in all_results[k]['results']:
            accs.append(all_results[k]['results'][exp_name]['test_acc_mean'] * 100)
        else:
            accs.append(None)
    
    if any(a is not None for a in accs):
        ax.plot(k_vals, accs, marker='o', label=labels[exp_name], 
                color=colors[exp_name], linewidth=2, markersize=8)

ax.set_xlabel('Diffusion Steps (k)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'Global Anti-Smoothing Effect - {DATASET_NAME}', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_base}/plots/k_trajectories_all_methods.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {output_base}/plots/k_trajectories_all_methods.png')
plt.close()

# Plot 2: Parameter efficiency (accuracy vs parameters)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Use results from best k
best_k = max(all_results.keys())
best_results = all_results[best_k]

methods_to_plot = ['sgc_mlp_baseline', 'restricted_standard_mlp', 'full_rownorm_mlp', 'jl_binary_2nc', 'jl_ortho_2nc']
x_params = []
y_accs = []
method_names = []

for exp_name in methods_to_plot:
    if exp_name in best_results['results'] and exp_name in best_results['parameter_counts']:
        x_params.append(best_results['parameter_counts'][exp_name])
        y_accs.append(best_results['results'][exp_name]['test_acc_mean'] * 100)
        method_names.append(labels[exp_name])

ax.scatter(x_params, y_accs, s=200, alpha=0.6)

for i, name in enumerate(method_names):
    ax.annotate(name, (x_params[i], y_accs[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Number of Parameters', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'Parameter Efficiency (k={best_k}) - {DATASET_NAME}', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_base}/plots/parameter_efficiency.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {output_base}/plots/parameter_efficiency.png')
plt.close()

# ============================================================================
# Final Summary
# ============================================================================

print(f'\n{"="*80}')
print('EXPERIMENT COMPLETE - WITH ADDED NEW PART A/B FRAMEWORK')
print(f'{"="*80}')
print(f'Dataset: {DATASET_NAME}')
print(f'Configuration: {SPLIT_TYPE} splits, {COMPONENT_TYPE} graph')
print(f'Components: {num_components}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')

# Find best methods
best_k = max(all_results.keys())
best_sgc_acc = all_results[best_k]['results']['sgc_baseline']['test_acc_mean'] * 100
best_sgc_mlp_acc = all_results[best_k]['results']['sgc_mlp_baseline']['test_acc_mean'] * 100
best_restricted_std_acc = all_results[best_k]['results']['restricted_standard_mlp']['test_acc_mean'] * 100
best_full_acc = all_results[best_k]['results']['full_rownorm_mlp']['test_acc_mean'] * 100

print(f'\nKey Results at k={best_k}:')
print(f'  SGC Baseline:               {best_sgc_acc:.2f}%')
print(f'  SGC + MLP:                  {best_sgc_mlp_acc:.2f}%')
print(f'  Restricted + StandardMLP:   {best_restricted_std_acc:.2f}%')
print(f'  Restricted + RowNorm:       {best_full_acc:.2f}%')

print(f'\nPart A - Basis Sensitivity:')
print(f'  SGC+MLP → Restricted+StandardMLP: {best_restricted_std_acc - best_sgc_mlp_acc:+.2f}pp')
if abs(best_restricted_std_acc - best_sgc_mlp_acc) > 0.5:
    print(f'  ✓ Basis choice matters (span is identical but performance differs)')
else:
    print(f'  ✗ No significant basis sensitivity detected')

print(f'\nPart B - RowNorm Improvement:')
print(f'  Restricted+StandardMLP → Restricted+RowNorm: {best_full_acc - best_restricted_std_acc:+.2f}pp')
if best_full_acc > best_restricted_std_acc:
    print(f'  ✓ RowNorm improves over StandardMLP on eigenvectors')
else:
    print(f'  ✗ RowNorm does not improve (StandardMLP is better)')

print(f'\nThe Gap - How much does Part B fix Part A?:')
gap_a = best_restricted_std_acc - best_sgc_mlp_acc
gap_b = best_full_acc - best_restricted_std_acc

gap_final = best_full_acc - best_sgc_mlp_acc
if gap_a < 0:  # If Part A degraded performance
    if gap_b > 0:  # If Part B improved
        if gap_final > 0:
            print(f'  ✓ Fully recovered: {gap_final:+.2f}pp (beat baseline!)')
        else:
            pct_fixed = (gap_b / abs(gap_a)) * 100
            print(f'  Partially recovered: {gap_final:+.2f}pp ({pct_fixed:.0f}% of gap closed)')
    else:  # If Part B also degraded
        print(f'  ✗ Part B made it worse: {gap_final:+.2f}pp (total degradation)')
else:
    print(f'  Final improvement: {gap_final:+.2f}pp vs SGC+MLP')

# Check for anti-smoothing effect
k_sorted = sorted(all_results.keys())
sgc_first = all_results[k_sorted[0]]['results']['sgc_baseline']['test_acc_mean'] * 100
sgc_last = all_results[k_sorted[-1]]['results']['sgc_baseline']['test_acc_mean'] * 100
full_first = all_results[k_sorted[0]]['results']['full_rownorm_mlp']['test_acc_mean'] * 100
full_last = all_results[k_sorted[-1]]['results']['full_rownorm_mlp']['test_acc_mean'] * 100

print(f'\nAnti-Smoothing Effect:')
print(f'  SGC: {sgc_first:.2f}% (k={k_sorted[0]}) → {sgc_last:.2f}% (k={k_sorted[-1]}) [{sgc_last-sgc_first:+.2f}pp]')
print(f'  Ours: {full_first:.2f}% (k={k_sorted[0]}) → {full_last:.2f}% (k={k_sorted[-1]}) [{full_last-full_first:+.2f}pp]')

if (sgc_last - sgc_first) < 0 and (full_last - full_first) > 0:
    print(f'  ✓ Anti-smoothing confirmed: Our method improves while SGC degrades!')
elif (sgc_last - sgc_first) < 0 and (full_last - full_first) < 0:
    print(f'  ✗ Both methods degrade (anti-smoothing failed)')
else:
    print(f'  → Mixed pattern (investigate further)')

print(f'\nResults saved to: {output_base}/')
print(f'{"="*80}')
