"""
===================================================================================
INVESTIGATION 4: SPECTRAL NORMALIZATION (NESTED SPHERES)
===================================================================================

Research Question: Does eigenvalue weighting (nested spheres) improve 
classification on restricted eigenvectors?

Hypothesis: Current RowNorm treats all eigenvector dimensions equally, 
discarding eigenvalue structure. Each eigenvector should be weighted by its
eigenvalue BEFORE normalization, creating "nested spheres" where different
eigenvectors contribute with different magnitudes based on their spectral importance.

Mathematical Formulation:
    Standard RowNorm:     V_norm = V / ||V||_row
    Spectral RowNorm:     V_weighted = V * f(Λ)
                          V_norm = V_weighted / ||V_weighted||_row
    
    where f(Λ) = Λ^alpha for different alpha values:
        alpha = 0:   Standard RowNorm (baseline)
        alpha = 0.5: Square root eigenvalue weighting
        alpha = 1.0: Linear eigenvalue weighting
        alpha = -0.5: Inverse square root weighting
        alpha = -1.0: Inverse eigenvalue weighting

Framework Extension:
    Part A: Basis Sensitivity (SGC+MLP vs Restricted+StandardMLP)
    Part B: RowNorm Improvement (Restricted+StandardMLP vs Restricted+RowNorm)
    Part B.5: Magnitude Preservation (Restricted+RowNorm vs Restricted+LogMag)
    Part B.6: Spectral Structure (NEW - Does eigenvalue weighting help?)

Methods Compared:
    1. SGC Baseline (logistic regression)
    2. SGC + MLP (Part A baseline)
    3. Restricted + StandardMLP (Part A endpoint)
    4. Restricted + RowNorm (Part B - alpha=0)
    5. Restricted + Log-Magnitude (Part B.5 - best from Inv3)
    6. Restricted + Spectral RowNorm (alpha=0.5) (NEW)
    7. Restricted + Spectral RowNorm (alpha=1.0) (NEW)
    8. Restricted + Spectral RowNorm (alpha=-0.5) (NEW)
    9. Restricted + Spectral RowNorm (alpha=-1.0) (NEW)
    10. Restricted + Nested Spheres (alpha=0.5, beta=1.0) (NEW - FULL)

Author: Mohammad
Date: November 2025
===================================================================================
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
    load_dataset, build_graph_matrices, compute_restricted_eigenvectors,
    StandardMLP, RowNormMLP, LogMagnitudeMLP
)

# ============================================================================
# Models (following investigation3 pattern - define locally)
# ============================================================================

class SGC(nn.Module):
    """SGC: Logistic regression with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)


class SpectralRowNormMLP(nn.Module):
    """
    Spectral RowNorm MLP with eigenvalue weighting (Nested Spheres)
    
    V_weighted = V * (eigenvalues ** alpha)
    V_normalized = V_weighted / ||V_weighted||_row
    """
    def __init__(self, input_dim, hidden_dim, num_classes, eigenvalues, alpha=0.0):
        super().__init__()
        
        # Eigenvalue weighting
        if abs(alpha) < 1e-8:  # alpha=0 is standard RowNorm
            eigenvalue_weights = torch.ones(input_dim)
        else:
            eigenvalues_safe = torch.abs(eigenvalues) + 1e-8
            eigenvalue_weights = eigenvalues_safe ** alpha
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('eigenvalue_weights', eigenvalue_weights)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, X):
        # Weight by eigenvalues
        X_weighted = X * self.eigenvalue_weights
        
        # Row normalization
        X_norm = torch.norm(X_weighted, dim=1, keepdim=True)
        X_normalized = X_weighted / (X_norm + 1e-10)
        
        return self.mlp(X_normalized)


class NestedSpheresClassifier(nn.Module):
    """
    Full Nested Spheres architecture combining:
    1. Eigenvalue weighting (spectral structure)
    2. Magnitude preservation (node importance)
    
    V_weighted = V * (eigenvalues ** alpha)
    M = ||V_weighted||_row
    V_normalized = V_weighted / M
    X_augmented = [V_normalized, beta * log(M)]
    """
    def __init__(self, input_dim, hidden_dim, num_classes, eigenvalues, alpha=0.5, beta=1.0):
        super().__init__()
        
        # Eigenvalue weighting
        if abs(alpha) < 1e-8:
            eigenvalue_weights = torch.ones(input_dim)
        else:
            eigenvalues_safe = torch.abs(eigenvalues) + 1e-8
            eigenvalue_weights = eigenvalues_safe ** alpha
        
        self.register_buffer('eigenvalue_weights', eigenvalue_weights)
        self.beta = beta
        
        # MLP (input_dim + 1 because we add log-magnitude)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, X):
        # Weight by eigenvalues (nested spheres)
        X_weighted = X * self.eigenvalue_weights
        
        # Compute magnitude
        M = torch.norm(X_weighted, dim=1, keepdim=True)
        
        # Normalize
        X_normalized = X_weighted / (M + 1e-10)
        
        # Augment with log-magnitude
        log_M = torch.log(M + 1e-10)
        X_augmented = torch.cat([X_normalized, self.beta * log_M], dim=1)
        
        return self.mlp(X_augmented)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Investigation 4: Spectral Normalization')
parser.add_argument('--dataset', type=str, default='coauthor-cs',
                   help='Dataset name')
parser.add_argument('--k_diffusion', type=int, default=10,
                   help='Number of diffusion steps')
parser.add_argument('--split_type', type=str, choices=['fixed', 'random'],
                   default='fixed', help='Type of data split')
parser.add_argument('--component_type', type=str, choices=['lcc', 'full'],
                   default='lcc', help='Use largest connected component or full graph')
parser.add_argument('--num_seeds', type=int, default=5,
                   help='Number of random seeds')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

DATASET_NAME = args.dataset
K_DIFFUSION = args.k_diffusion
SPLIT_TYPE = args.split_type
COMPONENT_TYPE = args.component_type
NUM_SEEDS = args.num_seeds
device = torch.device(args.device)

# Hyperparameters (same as previous investigations)
HIDDEN_DIM = 256
HIDDEN_MAG = 32
EPOCHS = 500
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Alpha values to test
ALPHA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

print('='*80)
print('INVESTIGATION 4: SPECTRAL NORMALIZATION (NESTED SPHERES)')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion steps (k): {K_DIFFUSION}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Number of seeds: {NUM_SEEDS}')
print(f'Device: {device}')
print(f'Alpha values: {ALPHA_VALUES}')
print('='*80)

# ============================================================================
# Helper Functions
# ============================================================================

def aggregate_results(results_list):
    """Aggregate results from multiple seeds"""
    val_accs = [r['val_acc'] for r in results_list]
    test_accs = [r['test_acc'] for r in results_list]
    train_times = [r['train_time'] for r in results_list]
    
    return {
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs),
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        'train_time_mean': np.mean(train_times),
        'train_time_std': np.std(train_times)
    }


def train_and_test(model, X_train, y_train, X_val, y_val, X_test, y_test,
                   epochs, lr, weight_decay, device, use_scheduler=False):
    """Train and test a model (EXACT MATCH to partAB.py)"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=50, verbose=False
        )
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    patience_limit = 100
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
            
            test_logits = model(X_test_t)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == y_test_t).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if use_scheduler:
            scheduler.step(val_acc)
        
        if patience_counter >= patience_limit:
            break
    
    train_time = time.time() - start_time
    
    return best_val_acc, best_test_acc, train_time


def get_largest_connected_component_nx(adj):
    """Extract largest connected component using NetworkX (EXACT MATCH to partAB.py)"""
    G = nx.from_scipy_sparse_array(adj)
    components = list(nx.connected_components(G))
    
    if len(components) == 1:
        return np.ones(adj.shape[0], dtype=bool)
    
    largest_component = max(components, key=len)
    mask = np.zeros(adj.shape[0], dtype=bool)
    mask[list(largest_component)] = True
    
    return mask


def extract_subgraph(adj, features, labels, mask, split_idx):
    """Extract subgraph for nodes in mask (EXACT MATCH to investigation3)"""
    node_indices = np.where(mask)[0]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
    
    adj_sub = adj[mask][:, mask]
    features_sub = features[mask]
    labels_sub = labels[mask]
    
    split_idx_sub = None
    if split_idx is not None:
        split_idx_sub = {}
        for split_name, indices in split_idx.items():
            mask_indices = np.isin(indices, node_indices)
            old_indices = indices[mask_indices]
            new_indices = np.array([old_to_new[idx] for idx in old_indices])
            split_idx_sub[split_name] = new_indices
    
    return adj_sub, features_sub, labels_sub, split_idx_sub


def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency (EXACT MATCH to partAB.py)"""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def sgc_precompute(features, adj_normalized, degree):
    """Apply SGC precomputation (EXACT MATCH to partAB.py)"""
    for i in range(degree):
        features = adj_normalized @ features
    return features





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
    """SGC + MLP Baseline - FIXED: No data leakage"""
    results = []
    
    # CRITICAL FIX: Fit scaler on TRAINING data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_diffused[train_idx])
    X_val_scaled = scaler.transform(X_diffused[val_idx])
    X_test_scaled = scaler.transform(X_diffused[test_idx])
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = StandardMLP(X_diffused.shape[1], HIDDEN_DIM, num_classes)
        
        val_acc, test_acc, train_time = train_and_test(
            model, X_train_scaled, labels[train_idx],
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
    """Restricted Eigenvectors + StandardScaler + MLP - FIXED: No data leakage"""
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D, num_components
    )
    
    # CRITICAL FIX: Fit scaler on TRAINING data only
    scaler = StandardScaler()
    U_train_scaled = scaler.fit_transform(U[train_idx])
    U_val_scaled = scaler.transform(U[val_idx])
    U_test_scaled = scaler.transform(U[test_idx])
    
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = StandardMLP(d_eff, HIDDEN_DIM, num_classes)
        
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
    
    return results, d_eff, ortho_err, eigenvalues


def run_restricted_rownorm_mlp(X_diffused, L, D, num_components, labels,
                              train_idx, val_idx, test_idx, num_classes,
                              num_seeds, device):
    """Restricted Eigenvectors + RowNorm MLP (alpha=0, standard RowNorm)"""
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
        
        model = RowNormMLP(d_eff, HIDDEN_DIM, num_classes)
        
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
    
    return results, d_eff, ortho_err, eigenvalues


def run_log_magnitude(X_diffused, L, D, num_components, labels,
                     train_idx, val_idx, test_idx, num_classes,
                     num_seeds, device):
    """Log-Magnitude Augmented MLP (best from Investigation 3)"""
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
        
        model = LogMagnitudeMLP(d_eff, HIDDEN_DIM, num_classes)
        
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
    
    return results, d_eff, ortho_err, eigenvalues


def run_spectral_rownorm(X_diffused, L, D, num_components, labels,
                        train_idx, val_idx, test_idx, num_classes,
                        num_seeds, device, alpha):
    """Spectral RowNorm MLP with eigenvalue weighting"""
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
    
    # Convert eigenvalues to torch tensor
    eigenvalues_torch = torch.FloatTensor(eigenvalues)
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = SpectralRowNormMLP(d_eff, HIDDEN_DIM, num_classes, 
                                   eigenvalues_torch, alpha=alpha)
        
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
    
    return results, d_eff, ortho_err, eigenvalues


def run_nested_spheres(X_diffused, L, D, num_components, labels,
                      train_idx, val_idx, test_idx, num_classes,
                      num_seeds, device, alpha, beta):
    """Nested Spheres Classifier (eigenvalue weighting + magnitude)"""
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
    
    # Convert eigenvalues to torch tensor
    eigenvalues_torch = torch.FloatTensor(eigenvalues)
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = NestedSpheresClassifier(d_eff, HIDDEN_DIM, num_classes,
                                       eigenvalues_torch, alpha=alpha, beta=beta)
        
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
    
    return results, d_eff, ortho_err, eigenvalues


# ============================================================================
# Main Experiment
# ============================================================================

# Create output directory
output_base = f'results/investigation4_spectral_normalization/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}/k{K_DIFFUSION}'
os.makedirs(f'{output_base}/metrics', exist_ok=True)
os.makedirs(f'{output_base}/plots', exist_ok=True)

print('\n[1/6] Loading dataset...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx, val_idx, test_idx) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Nodes: {num_nodes:,}, Features: {X_raw.shape[1]}, Classes: {num_classes}')

# Build graph matrices
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)

# Extract LCC if requested (EXACT MATCH to partAB.py)
if COMPONENT_TYPE == 'lcc':
    print('\n[3/6] Extracting largest connected component...')
    lcc_mask = get_largest_connected_component_nx(adj)
    
    split_idx_original = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_original
    )
    
    # Rebuild graph matrices for LCC
    print('Rebuilding graph matrices for LCC...')
    adj_coo = adj.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj_final, D, L = build_graph_matrices(edge_index_lcc, adj.shape[0])
    adj = adj_final
    
    num_components = 0  # LCC has 1 component
else:
    print('\n[3/6] Using whole graph...')
    num_components = len(list(nx.connected_components(nx.from_scipy_sparse_array(adj)))) - 1

# Update variables after component selection
num_nodes = X_raw.shape[0]
num_classes = len(np.unique(labels))

if SPLIT_TYPE == 'fixed':
    train_idx = split_idx['train_idx']
    val_idx = split_idx['val_idx']
    test_idx = split_idx['test_idx']

# Compute normalized adjacency (EXACT MATCH to partAB.py)
print('\n[4/6] Computing SGC-style normalized adjacency...')
A_sgc = compute_sgc_normalized_adjacency(adj)
print('✓ A_sgc = D^(-1/2) (A + I) D^(-1/2)')

# Diffuse features (EXACT MATCH to partAB.py)
print(f'\n[5/6] Precomputing diffusion (k={K_DIFFUSION})...')
features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw
X_diffused = sgc_precompute(features_dense.copy(), A_sgc, K_DIFFUSION)
print(f'✓ X_diffused shape: {X_diffused.shape}')

# Handle splits
if SPLIT_TYPE == 'random':
    print(f'\n[6/6] Creating random 60/20/20 split...')
    np.random.seed(0)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
else:
    print(f'\n[6/6] Using fixed splits...')

print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

# Run experiments
print(f'\n{"="*80}')
print(f'RUNNING EXPERIMENTS WITH {NUM_SEEDS} SEEDS')
print(f'{"="*80}')

experiments = {}
metadata = {
    'dataset': DATASET_NAME,
    'k_diffusion': K_DIFFUSION,
    'split_type': SPLIT_TYPE,
    'component_type': COMPONENT_TYPE,
    'num_nodes': num_nodes,
    'num_features': X_raw.shape[1],
    'num_classes': num_classes,
    'num_components': num_components,
    'num_seeds': NUM_SEEDS
}

# Experiment 1: SGC Baseline
print('\n[1/10] SGC Baseline (Logistic)')
sgc_results = run_sgc_baseline(
    X_diffused, labels, train_idx, val_idx, test_idx,
    num_classes, NUM_SEEDS, device
)
experiments['sgc_baseline'] = sgc_results
sgc_agg = aggregate_results(sgc_results)
print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')

# Experiment 2: SGC + MLP
print('\n[2/10] SGC + MLP Baseline')
sgc_mlp_results = run_sgc_mlp_baseline(
    X_diffused, labels, train_idx, val_idx, test_idx,
    num_classes, NUM_SEEDS, device
)
experiments['sgc_mlp_baseline'] = sgc_mlp_results
sgc_mlp_agg = aggregate_results(sgc_mlp_results)
print(f'→ {sgc_mlp_agg["test_acc_mean"]*100:.2f}% ± {sgc_mlp_agg["test_acc_std"]*100:.2f}%')

# Experiment 3: Restricted + StandardMLP
print('\n[3/10] Restricted + StandardMLP')
restricted_std_results, d_eff_std, ortho_err_std, eigenvalues_std = run_restricted_standard_mlp(
    X_diffused, L, D, num_components, labels,
    train_idx, val_idx, test_idx, num_classes,
    NUM_SEEDS, device
)
experiments['restricted_standard_mlp'] = restricted_std_results
metadata['d_restricted'] = d_eff_std
metadata['ortho_error'] = float(ortho_err_std)
restricted_std_agg = aggregate_results(restricted_std_results)
print(f'→ {restricted_std_agg["test_acc_mean"]*100:.2f}% ± {restricted_std_agg["test_acc_std"]*100:.2f}%')
print(f'  D-orthonormality error: {ortho_err_std:.2e}')

# Experiment 4: Restricted + RowNorm (alpha=0)
print('\n[4/10] Restricted + RowNorm (alpha=0, standard)')
rownorm_results, d_eff_rn, ortho_err_rn, eigenvalues_rn = run_restricted_rownorm_mlp(
    X_diffused, L, D, num_components, labels,
    train_idx, val_idx, test_idx, num_classes,
    NUM_SEEDS, device
)
experiments['restricted_rownorm'] = rownorm_results
rownorm_agg = aggregate_results(rownorm_results)
print(f'→ {rownorm_agg["test_acc_mean"]*100:.2f}% ± {rownorm_agg["test_acc_std"]*100:.2f}%')

# Experiment 5: Log-Magnitude
print('\n[5/10] Log-Magnitude Augmented (best from Inv3)')
logmag_results, d_eff_lm, ortho_err_lm, eigenvalues_lm = run_log_magnitude(
    X_diffused, L, D, num_components, labels,
    train_idx, val_idx, test_idx, num_classes,
    NUM_SEEDS, device
)
experiments['log_magnitude'] = logmag_results
logmag_agg = aggregate_results(logmag_results)
print(f'→ {logmag_agg["test_acc_mean"]*100:.2f}% ± {logmag_agg["test_acc_std"]*100:.2f}%')

# Experiments 6-9: Spectral RowNorm with different alpha values
spectral_results = {}
exp_counter = 6
for alpha in ALPHA_VALUES:
    if alpha == 0.0:
        continue  # Already done as standard RowNorm
    
    print(f'\n[{exp_counter}/10] Spectral RowNorm (alpha={alpha})')
    
    results, d_eff, ortho_err, eigenvalues = run_spectral_rownorm(
        X_diffused, L, D, num_components, labels,
        train_idx, val_idx, test_idx, num_classes,
        NUM_SEEDS, device, alpha=alpha
    )
    
    exp_name = f'spectral_rownorm_alpha{alpha}'
    experiments[exp_name] = results
    spectral_results[alpha] = results
    
    agg = aggregate_results(results)
    print(f'→ {agg["test_acc_mean"]*100:.2f}% ± {agg["test_acc_std"]*100:.2f}%')
    
    exp_counter += 1

# Experiment 10: Nested Spheres (full architecture)
print(f'\n[10/10] Nested Spheres Classifier (alpha=0.5, beta=1.0)')
nested_results, d_eff_nested, ortho_err_nested, eigenvalues_nested = run_nested_spheres(
    X_diffused, L, D, num_components, labels,
    train_idx, val_idx, test_idx, num_classes,
    NUM_SEEDS, device, alpha=0.5, beta=1.0
)
experiments['nested_spheres'] = nested_results
nested_agg = aggregate_results(nested_results)
print(f'→ {nested_agg["test_acc_mean"]*100:.2f}% ± {nested_agg["test_acc_std"]*100:.2f}%')

# ============================================================================
# Results Summary
# ============================================================================

print(f'\n{"="*80}')
print('RESULTS SUMMARY')
print(f'{"="*80}')

final_results = {}
for exp_name, exp_results in experiments.items():
    final_results[exp_name] = aggregate_results(exp_results)

print(f'\n{"Method":<35} {"Test Acc":<15} {"Std":<10}')
print('-' * 60)

method_names = {
    'sgc_baseline': 'SGC Baseline (Logistic)',
    'sgc_mlp_baseline': 'SGC + MLP',
    'restricted_standard_mlp': 'Restricted + StandardMLP',
    'restricted_rownorm': 'Restricted + RowNorm (α=0)',
    'log_magnitude': 'Log-Magnitude Augmented',
    'spectral_rownorm_alpha-1.0': 'Spectral RowNorm (α=-1.0)',
    'spectral_rownorm_alpha-0.5': 'Spectral RowNorm (α=-0.5)',
    'spectral_rownorm_alpha0.5': 'Spectral RowNorm (α=0.5)',
    'spectral_rownorm_alpha1.0': 'Spectral RowNorm (α=1.0)',
    'nested_spheres': 'Nested Spheres (α=0.5, β=1.0)'
}

for exp_name in method_names.keys():
    if exp_name in final_results:
        agg = final_results[exp_name]
        display_name = method_names[exp_name]
        print(f'{display_name:<35} {agg["test_acc_mean"]*100:>6.2f}%        {agg["test_acc_std"]*100:>5.2f}%')

# ============================================================================
# Framework Analysis
# ============================================================================

print(f'\n{"="*80}')
print('FRAMEWORK ANALYSIS (Part A/B/B.5/B.6)')
print(f'{"="*80}')

sgc_mlp_acc = final_results['sgc_mlp_baseline']['test_acc_mean'] * 100
restricted_std_acc = final_results['restricted_standard_mlp']['test_acc_mean'] * 100
rownorm_acc = final_results['restricted_rownorm']['test_acc_mean'] * 100
logmag_acc = final_results['log_magnitude']['test_acc_mean'] * 100

# Find best spectral rownorm
best_spectral_alpha = None
best_spectral_acc = -1
for alpha in ALPHA_VALUES:
    if alpha == 0.0:
        continue
    exp_name = f'spectral_rownorm_alpha{alpha}'
    if exp_name in final_results:
        acc = final_results[exp_name]['test_acc_mean'] * 100
        if acc > best_spectral_acc:
            best_spectral_acc = acc
            best_spectral_alpha = alpha

nested_acc = final_results['nested_spheres']['test_acc_mean'] * 100

# Compute deltas
part_a = restricted_std_acc - sgc_mlp_acc
part_b = rownorm_acc - restricted_std_acc
part_b5 = logmag_acc - restricted_std_acc
part_b6_spectral = best_spectral_acc - restricted_std_acc
part_b6_nested = nested_acc - restricted_std_acc

gap_rownorm = rownorm_acc - sgc_mlp_acc
gap_logmag = logmag_acc - sgc_mlp_acc
gap_spectral = best_spectral_acc - sgc_mlp_acc
gap_nested = nested_acc - sgc_mlp_acc

print(f'\nPart A (Basis Sensitivity):')
print(f'  SGC+MLP → Restricted+StandardMLP: {part_a:+.2f}pp')

print(f'\nPart B (RowNorm):')
print(f'  Restricted+StandardMLP → Restricted+RowNorm: {part_b:+.2f}pp')

print(f'\nPart B.5 (Magnitude Preservation):')
print(f'  Restricted+StandardMLP → Log-Magnitude: {part_b5:+.2f}pp')

print(f'\nPart B.6 (Spectral Structure - NEW):')
print(f'  Best Spectral RowNorm (α={best_spectral_alpha}): {part_b6_spectral:+.2f}pp over StandardMLP')
print(f'  Nested Spheres (full): {part_b6_nested:+.2f}pp over StandardMLP')

if part_b6_spectral > part_b:
    print(f'  ✓ Spectral RowNorm beats standard RowNorm by {part_b6_spectral - part_b:.2f}pp')
else:
    print(f'  ✗ Spectral RowNorm does not beat standard RowNorm ({part_b6_spectral - part_b:+.2f}pp)')

if part_b6_nested > part_b5:
    print(f'  ✓ Nested Spheres beats Log-Magnitude by {part_b6_nested - part_b5:.2f}pp')
else:
    print(f'  ✗ Nested Spheres does not beat Log-Magnitude ({part_b6_nested - part_b5:+.2f}pp)')

print(f'\nThe Gap (Final vs SGC+MLP Baseline):')
print(f'  RowNorm:              {gap_rownorm:+.2f}pp')
print(f'  Log-Magnitude:        {gap_logmag:+.2f}pp')
print(f'  Spectral RowNorm:     {gap_spectral:+.2f}pp')
print(f'  Nested Spheres:       {gap_nested:+.2f}pp')

if gap_nested > 0:
    print(f'  ✓ Nested Spheres CLOSES THE GAP! (+{gap_nested:.2f}pp)')
elif gap_nested > gap_logmag:
    print(f'  ◐ Nested Spheres partially closes The Gap (improvement over Log-Mag: {gap_nested - gap_logmag:+.2f}pp)')
else:
    print(f'  ✗ The Gap persists ({gap_nested:+.2f}pp)')

# Save framework analysis
framework_analysis = {
    'part_a': float(part_a),
    'part_b_rownorm': float(part_b),
    'part_b5_logmag': float(part_b5),
    'part_b6_spectral_rownorm': float(part_b6_spectral),
    'part_b6_nested_spheres': float(part_b6_nested),
    'gap_rownorm': float(gap_rownorm),
    'gap_logmag': float(gap_logmag),
    'gap_spectral_rownorm': float(gap_spectral),
    'gap_nested_spheres': float(gap_nested),
    'best_spectral_alpha': float(best_spectral_alpha) if best_spectral_alpha is not None else None
}

# Save all results
all_results = {
    'experiments': final_results,
    'metadata': metadata,
    'framework_analysis': framework_analysis,
    'eigenvalues': eigenvalues_std.tolist()
}

with open(f'{output_base}/metrics/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n✓ Saved results: {output_base}/metrics/results.json')

# ============================================================================
# Visualizations
# ============================================================================

print('\nGenerating visualizations...')

# Plot 1: Alpha sweep
fig, ax = plt.subplots(figsize=(10, 6))

alphas_tested = []
accs = []
for alpha in ALPHA_VALUES:
    if alpha == 0.0:
        exp_name = 'restricted_rownorm'
    else:
        exp_name = f'spectral_rownorm_alpha{alpha}'
    
    if exp_name in final_results:
        alphas_tested.append(alpha)
        accs.append(final_results[exp_name]['test_acc_mean'] * 100)

ax.plot(alphas_tested, accs, marker='o', linewidth=2, markersize=10, color='steelblue')
ax.axhline(y=sgc_mlp_acc, color='red', linestyle='--', label='SGC+MLP Baseline', linewidth=2)
ax.axhline(y=logmag_acc, color='green', linestyle='--', label='Log-Magnitude', linewidth=2)
ax.set_xlabel('Alpha (Eigenvalue Weighting Exponent)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'Spectral RowNorm: Alpha Sweep - {DATASET_NAME}', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_base}/plots/alpha_sweep.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {output_base}/plots/alpha_sweep.png')
plt.close()

# Plot 2: Method comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))

methods = ['SGC+MLP', 'Restricted+Std', 'RowNorm', 'Log-Mag', 
           f'Spectral (α={best_spectral_alpha})', 'Nested Spheres']
accuracies = [sgc_mlp_acc, restricted_std_acc, rownorm_acc, logmag_acc, 
              best_spectral_acc, nested_acc]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'Method Comparison - {DATASET_NAME}', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig(f'{output_base}/plots/method_comparison.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {output_base}/plots/method_comparison.png')
plt.close()

print(f'\n{"="*80}')
print('EXPERIMENT COMPLETE')
print(f'{"="*80}')