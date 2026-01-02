"""
Investigation: Does RowNorm Fix Standard Whitening?
====================================================

Direction B from Professor Koutis's email:
"It would be worth exploring if Row Normalization fixes the whitening proposed in [Wadia et al.], 
and not just our kind of whitening. If it does, then we have a bigger claim."

PURPOSE:
    This script runs experiments and saves raw results to JSON.
    It does NOT interpret results or draw conclusions.
    
    For analysis and conclusions, run: scripts/analyze_whitening_rownorm.py

Experimental Framework:
    For each dataset:
        1. Original features X (or diffused A^k X)
        2. Train-whitening: PCA(X), ZCA(X) computed from train set
        3. Full-whitening: ZCA(X) computed from ALL data (Wadia's method)
        4. Spectral: Rayleigh-Ritz restricted eigenvectors U
        
        For each feature variant:
            - Linear classifier (with bias)
            - Linear + RowNorm (no bias)
            - MLP (with StandardScaler for original only)
            - MLP + RowNorm

Output:
    results/investigation_whitening_rownorm/{dataset}_{split}_{component}_k{k}/metrics/results.json

Usage:
    python investigation_whitening_rownorm.py cora --splits fixed --component lcc
    python investigation_whitening_rownorm.py cora --splits fixed --component lcc --k_diffusion 2
    
    # Run all datasets
    for dataset in ogbn-arxiv cora citeseer pubmed wikics amazon-photo amazon-computers coauthor-cs coauthor-physics; do
        for k in 0 2 10; do
            python investigation_whitening_rownorm.py $dataset --splits fixed --component lcc --k_diffusion $k
        done
    done

Author: Mohammad Dindoost
Date: December 2024
Reference: Wadia et al. "Whitening and Second Order Optimization Both Make Information 
           in the Dataset Unusable During Training" (ICML 2021)
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
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import from utils (adjust path as needed)
try:
    from utils import (
        load_dataset, 
        build_graph_matrices,  # Returns (adj, L, D) - note order!
        StandardMLP, 
        RowNormMLP,
        compute_restricted_eigenvectors,  # Use utils version for consistency
        extract_subgraph,  # Use utils version for consistency
    )
    UTILS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import from utils.py. Make sure it's in the path.")
    UTILS_AVAILABLE = False
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Direction B: Does RowNorm Fix Standard Whitening?')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--splits', type=str, choices=['random', 'fixed'], default='fixed',
                   help='Use random or fixed splits')
parser.add_argument('--component', type=str, choices=['whole', 'lcc'], default='lcc',
                   help='Use whole graph or largest connected component')
parser.add_argument('--k_diffusion', type=int, default=0,
                   help='Diffusion steps (0 = no diffusion, use raw features)')
args = parser.parse_args()

DATASET_NAME = args.dataset
SPLIT_TYPE = args.splits
COMPONENT_TYPE = args.component
K_DIFFUSION = args.k_diffusion

# Experimental parameters
NUM_RANDOM_SPLITS = 5 if SPLIT_TYPE == 'random' else 1
NUM_SEEDS = 5

# Training hyperparameters
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Regularization for whitening (handles rank-deficient matrices)
# Updated to 1e-6 for proper whitening per Professor Koutis's verification
# With eps=1e-6, Gram matrix K approaches identity (Wadia's condition satisfied)
WHITENING_EPS = 1e-6  # Proper whitening; verified K eigenvalues ≈ 1

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Output directory
output_base = f'results/investigation_whitening_rownorm/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}'
if K_DIFFUSION > 0:
    output_base += f'_k{K_DIFFUSION}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

print('='*80)
print('DIRECTION B: DOES ROWNORM FIX STANDARD WHITENING?')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Diffusion k: {K_DIFFUSION}')
print(f'Whitening eps: {WHITENING_EPS}')
print(f'Device: {device}')
print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('='*80)

# ============================================================================
# Model Architectures
# ============================================================================

class LinearClassifier(nn.Module):
    """Standard logistic regression with bias"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x):
        return self.fc(x)

class RowNormLinear(nn.Module):
    """Logistic regression with row normalization, no bias"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.fc(x)

# StandardMLP and RowNormMLP imported from utils.py

# ============================================================================
# Whitening Methods
# ============================================================================

def pca_whiten(X, train_idx, eps=WHITENING_EPS):
    """
    PCA Whitening (train-whitening: compute transform from training data only)
    
    Whitening transformation: X_white = (X - μ) @ V @ Λ^{-1/2}
    where V, Λ are eigenvectors/eigenvalues of training covariance
    
    Args:
        X: (n, d) feature matrix
        train_idx: indices of training nodes
        eps: regularization for numerical stability
    
    Returns:
        X_white: (n, d) whitened features
        diagnostics: dict with whitening statistics
    """
    n, d = X.shape
    
    # Compute mean and covariance from TRAINING data only
    X_train = X[train_idx]
    mu = X_train.mean(axis=0)
    X_centered = X - mu  # Center ALL data using training mean
    
    # Covariance of training data
    X_train_centered = X_train - mu
    F = X_train_centered.T @ X_train_centered / (len(train_idx) - 1)  # (d, d)
    
    # Eigendecomposition
    eigenvalues, V = np.linalg.eigh(F)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Regularize small eigenvalues
    eigenvalues_reg = np.maximum(eigenvalues, eps)
    
    # Whitening transformation
    W = V @ np.diag(1.0 / np.sqrt(eigenvalues_reg))  # (d, d)
    X_white = X_centered @ W
    
    # Diagnostics
    diagnostics = {
        'method': 'PCA',
        'eigenvalues_original': eigenvalues.copy(),
        'eigenvalues_regularized': eigenvalues_reg.copy(),
        'condition_number_before': eigenvalues[0] / (eigenvalues[-1] + 1e-10),
        'condition_number_after': eigenvalues_reg[0] / eigenvalues_reg[-1],
        'rank_deficient': np.sum(eigenvalues < eps),
    }
    
    return X_white.astype(np.float32), diagnostics

def zca_whiten(X, train_idx, eps=WHITENING_EPS):
    """
    ZCA Whitening (Zero-phase Component Analysis)
    
    ZCA stays closest to original data: X_white = (X - μ) @ V @ Λ^{-1/2} @ V^T
    
    Args:
        X: (n, d) feature matrix
        train_idx: indices of training nodes
        eps: regularization for numerical stability
    
    Returns:
        X_white: (n, d) whitened features
        diagnostics: dict with whitening statistics
    """
    n, d = X.shape
    
    # Compute mean and covariance from TRAINING data only
    X_train = X[train_idx]
    mu = X_train.mean(axis=0)
    X_centered = X - mu
    
    # Covariance of training data
    X_train_centered = X_train - mu
    F = X_train_centered.T @ X_train_centered / (len(train_idx) - 1)
    
    # Eigendecomposition
    eigenvalues, V = np.linalg.eigh(F)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Regularize
    eigenvalues_reg = np.maximum(eigenvalues, eps)
    
    # ZCA whitening: W = V @ Λ^{-1/2} @ V^T = F^{-1/2}
    W = V @ np.diag(1.0 / np.sqrt(eigenvalues_reg)) @ V.T
    X_white = X_centered @ W
    
    diagnostics = {
        'method': 'ZCA',
        'eigenvalues_original': eigenvalues.copy(),
        'eigenvalues_regularized': eigenvalues_reg.copy(),
        'condition_number_before': eigenvalues[0] / (eigenvalues[-1] + 1e-10),
        'condition_number_after': eigenvalues_reg[0] / eigenvalues_reg[-1],
        'rank_deficient': np.sum(eigenvalues < eps),
    }
    
    return X_white.astype(np.float32), diagnostics

def full_zca_whiten(X, eps=WHITENING_EPS):
    """
    Full ZCA Whitening (compute transform from ALL data - Wadia's extreme case)
    
    WARNING: This is the "cheating" scenario that Wadia shows destroys generalization
    when n <= d. Included for completeness.
    
    Args:
        X: (n, d) feature matrix
        eps: regularization
    
    Returns:
        X_white: whitened features
        diagnostics: dict
    """
    n, d = X.shape
    
    mu = X.mean(axis=0)
    X_centered = X - mu
    
    # Covariance of ALL data
    F = X_centered.T @ X_centered / (n - 1)
    
    eigenvalues, V = np.linalg.eigh(F)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    eigenvalues_reg = np.maximum(eigenvalues, eps)
    
    W = V @ np.diag(1.0 / np.sqrt(eigenvalues_reg)) @ V.T
    X_white = X_centered @ W
    
    diagnostics = {
        'method': 'Full-ZCA',
        'eigenvalues_original': eigenvalues.copy(),
        'eigenvalues_regularized': eigenvalues_reg.copy(),
        'condition_number_before': eigenvalues[0] / (eigenvalues[-1] + 1e-10),
        'condition_number_after': eigenvalues_reg[0] / eigenvalues_reg[-1],
        'rank_deficient': np.sum(eigenvalues < eps),
        'n_vs_d': 'n <= d (severe damage expected)' if n <= d else 'n > d (moderate damage expected)',
    }
    
    return X_white.astype(np.float32), diagnostics

# ============================================================================
# Rayleigh-Ritz (Your Method) - Using utils.compute_restricted_eigenvectors
# ============================================================================
# Note: compute_restricted_eigenvectors is imported from utils.py
# It returns: U, eigenvalues, d_effective, ortho_error

# ============================================================================
# Whitening Diagnostics (from our discussion)
# ============================================================================

def analyze_whitening_effect(X, X_white, name=""):
    """
    Analyze whether transformation has whitening-like effect.
    
    Compares feature covariance F = X^T X before and after transformation.
    """
    d = X.shape[1]
    d_white = X_white.shape[1]
    
    # Feature covariance
    F_X = X.T @ X
    F_white = X_white.T @ X_white
    
    # Normalize for comparison
    F_X_norm = F_X / (np.trace(F_X) / d + 1e-10) 
    F_white_norm = F_white / (np.trace(F_white) / d_white + 1e-10)
    
    # Distance from identity
    dist_X = np.linalg.norm(F_X_norm - np.eye(d), 'fro')
    dist_white = np.linalg.norm(F_white_norm - np.eye(d_white), 'fro')
    
    # Eigenvalue analysis
    eig_X = np.linalg.eigvalsh(F_X)
    eig_white = np.linalg.eigvalsh(F_white)
    
    eig_X_pos = eig_X[eig_X > 1e-10]
    eig_white_pos = eig_white[eig_white > 1e-10]
    
    # Condition number
    cond_X = eig_X_pos[-1] / eig_X_pos[0] if len(eig_X_pos) > 0 else float('inf')
    cond_white = eig_white_pos[-1] / eig_white_pos[0] if len(eig_white_pos) > 0 else float('inf')
    
    # Entropy
    p_X = eig_X_pos / eig_X_pos.sum()
    p_white = eig_white_pos / eig_white_pos.sum()
    
    entropy_X = -np.sum(p_X * np.log(p_X + 1e-10))
    entropy_white = -np.sum(p_white * np.log(p_white + 1e-10))
    
    max_entropy_X = np.log(len(eig_X_pos))
    max_entropy_white = np.log(len(eig_white_pos))
    
    return {
        'name': name,
        'dim_original': d,
        'dim_whitened': d_white,
        'dist_from_I_original': float(dist_X),
        'dist_from_I_whitened': float(dist_white),
        'condition_original': float(cond_X),
        'condition_whitened': float(cond_white),
        'entropy_original': float(entropy_X),
        'entropy_whitened': float(entropy_white),
        'max_entropy_original': float(max_entropy_X),
        'max_entropy_whitened': float(max_entropy_white),
        'uniformity_original': float(entropy_X / max_entropy_X) if max_entropy_X > 0 else 0,
        'uniformity_whitened': float(entropy_white / max_entropy_white) if max_entropy_white > 0 else 0,
        'whitening_indicators': int(dist_white < dist_X) + int(cond_white < cond_X) + int(entropy_white > entropy_X),
    }

def analyze_gram_matrix(X, name=""):
    """
    Analyze the Gram matrix K = XX^T (n×n sample-sample matrix).

    THIS IS WHAT WADIA SAYS DETERMINES GENERALIZATION!

    Wadia's key prediction:
      - n > d: K retains structure, some generalization possible
      - n ≤ d: After full whitening, K → Identity, complete collapse expected

    Our notation (X is n×d):
      - F = X^T @ X (d×d) - feature covariance (what whitening targets)
      - K = X @ X^T (n×n) - Gram matrix (what determines generalization!)

    Args:
        X: (n, d) feature matrix (samples × features)
        name: identifier for this analysis

    Returns:
        dict with Gram matrix statistics including Wadia criterion check
    """
    n, d = X.shape

    # Center the data
    X_centered = X - X.mean(axis=0)

    # For Gram matrix eigenvalues, we use the trick that eigenvalues of XX^T
    # equal eigenvalues of X^T X (plus zeros if n > d)
    # This is computationally cheaper when d < n

    # Compute feature covariance F = X^T X / (n-1) which is (d×d)
    F = X_centered.T @ X_centered / (n - 1)
    eig_F = np.linalg.eigvalsh(F)
    eig_F = np.sort(eig_F)[::-1]  # Descending order

    # Gram matrix K = XX^T / (n-1) has same non-zero eigenvalues as F
    # K is (n×n), F is (d×d)
    # Non-zero eigenvalues: min(n, d)

    if n > d:
        # K has d non-zero eigenvalues (same as F) plus (n-d) zeros
        eig_K = eig_F  # Non-zero part
    else:
        # K is n×n, can have at most n non-zero eigenvalues
        # But F is d×d with d > n, so F has n non-zero eigenvalues
        eig_K = eig_F[:n]

    # Filter positive eigenvalues for statistics
    eig_K_pos = eig_K[eig_K > 1e-10]
    eig_F_pos = eig_F[eig_F > 1e-10]

    # Wadia criterion: after proper whitening, K eigenvalues should be ≈ 1
    stats = {
        'name': name,
        'n_samples': n,
        'n_features': d,
        'wadia_regime': 'n <= d (collapse expected)' if n <= d else 'n > d (structure retained)',

        # Feature covariance F = X^T X statistics
        'F_shape': f'{d}x{d}',
        'F_n_nonzero_eig': len(eig_F_pos),
        'F_eig_mean': float(eig_F_pos.mean()) if len(eig_F_pos) > 0 else 0,
        'F_eig_std': float(eig_F_pos.std()) if len(eig_F_pos) > 0 else 0,
        'F_dist_from_1_mean': float(np.abs(eig_F_pos - 1).mean()) if len(eig_F_pos) > 0 else float('inf'),

        # Gram matrix K = XX^T statistics (THIS IS KEY!)
        'K_shape': f'{n}x{n}',
        'K_n_nonzero_eig': len(eig_K_pos),
        'K_eig_min': float(eig_K_pos.min()) if len(eig_K_pos) > 0 else 0,
        'K_eig_max': float(eig_K_pos.max()) if len(eig_K_pos) > 0 else 0,
        'K_eig_mean': float(eig_K_pos.mean()) if len(eig_K_pos) > 0 else 0,
        'K_eig_std': float(eig_K_pos.std()) if len(eig_K_pos) > 0 else 0,
        'K_dist_from_1_mean': float(np.abs(eig_K_pos - 1).mean()) if len(eig_K_pos) > 0 else float('inf'),
        'K_dist_from_1_max': float(np.abs(eig_K_pos - 1).max()) if len(eig_K_pos) > 0 else float('inf'),
        'K_condition_number': float(eig_K_pos.max() / eig_K_pos.min()) if len(eig_K_pos) > 0 else float('inf'),

        # Wadia criterion check
        'K_is_approximately_identity': bool(np.abs(eig_K_pos - 1).mean() < 0.1) if len(eig_K_pos) > 0 else False,

        # First and last eigenvalues for inspection
        'K_eig_first_10': eig_K_pos[:10].tolist() if len(eig_K_pos) > 0 else [],
        'K_eig_last_10': eig_K_pos[-10:].tolist() if len(eig_K_pos) > 0 else [],
    }

    return stats

# ============================================================================
# Graph and Data Utilities
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
    print(f'  Largest component: {len(largest_cc)} nodes ({len(largest_cc)/adj.shape[0]*100:.1f}%)')
    
    return lcc_mask

# Note: extract_subgraph is imported from utils.py

def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency"""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def sgc_precompute(features, adj_normalized, degree):
    """Apply SGC diffusion"""
    for _ in range(degree):
        features = adj_normalized @ features
    return features

# ============================================================================
# Training Functions
# ============================================================================

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs, lr, weight_decay, device):
    """Train model and return best validation accuracy and corresponding test accuracy"""
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
                
                test_pred = model(X_test_t).argmax(dim=1)
                test_acc = (test_pred == y_test_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
    
    train_time = time.time() - start_time
    
    return best_val_acc, best_test_acc, train_time

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
    }

# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(features, labels, train_idx, val_idx, test_idx, 
                   num_classes, model_class, model_kwargs, num_seeds, 
                   use_standard_scaler=False, description=""):
    """
    Run experiment with given features and model.
    
    Args:
        features: (n, d) feature matrix
        labels: (n,) labels
        train_idx, val_idx, test_idx: split indices
        num_classes: number of classes
        model_class: model class to instantiate
        model_kwargs: kwargs for model (except input_dim)
        num_seeds: number of random seeds
        use_standard_scaler: whether to apply StandardScaler
        description: experiment description for logging
    """
    results = []
    
    input_dim = features.shape[1]
    
    # Prepare data
    if use_standard_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_val = scaler.transform(features[val_idx])
        X_test = scaler.transform(features[test_idx])
    else:
        X_train = features[train_idx]
        X_val = features[val_idx]
        X_test = features[test_idx]
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_class(input_dim, **model_kwargs)
        
        val_acc, test_acc, train_time = train_and_evaluate(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results

# ============================================================================
# Main Experiment
# ============================================================================

print('\n[1/7] Loading dataset...')
edge_index, features, labels, num_nodes, num_classes, train_idx, val_idx, test_idx = load_dataset(DATASET_NAME)

print(f'  Nodes: {num_nodes}')
print(f'  Features: {features.shape[1]}')
print(f'  Classes: {num_classes}')
print(f'  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}')

# Build graph matrices
print('\n[2/7] Building graph matrices...')
adj, L, D = build_graph_matrices(edge_index, num_nodes)

# Handle connected components
if COMPONENT_TYPE == 'lcc':
    print('\n[3/7] Extracting largest connected component...')
    lcc_mask = get_largest_connected_component_nx(adj)
    
    split_idx_dict = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    adj, features, labels, split_idx_dict = extract_subgraph(adj, features, labels, lcc_mask, split_idx_dict)
    
    train_idx = split_idx_dict['train_idx']
    val_idx = split_idx_dict['val_idx']
    test_idx = split_idx_dict['test_idx']
    
    num_nodes = features.shape[0]
    num_components = 1
    
    # Rebuild L and D for LCC
    deg = np.array(adj.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - adj
    
    print(f'  LCC nodes: {num_nodes}')
    print(f'  LCC Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}')
else:
    print('\n[3/7] Using whole graph...')
    G = nx.from_scipy_sparse_array(adj)
    num_components = nx.number_connected_components(G)
    print(f'  Connected components: {num_components}')

# Convert to numpy if needed
if isinstance(features, torch.Tensor):
    features = features.numpy()
if isinstance(labels, torch.Tensor):
    labels = labels.numpy()

# Apply diffusion if requested
if K_DIFFUSION > 0:
    print(f'\n[3.5/7] Applying SGC diffusion (k={K_DIFFUSION})...')
    adj_normalized = compute_sgc_normalized_adjacency(adj)
    features = sgc_precompute(features, adj_normalized, K_DIFFUSION)
    if isinstance(features, np.matrix):
        features = np.asarray(features)

X = features.astype(np.float32)
n, d = X.shape

print(f'\nData dimensions: n={n} samples, d={d} features')
print(f'Wadia regime: {"n <= d (severe whitening damage expected)" if n <= d else "n > d (moderate damage expected)"}')

# ============================================================================
# Prepare Feature Variants
# ============================================================================

print('\n[4/7] Preparing feature variants...')

# Separate features into:
# 1. Split-independent: computed once (original, full_zca, rayleigh_ritz)
# 2. Split-dependent: must be recomputed per split (pca_whiten, zca_whiten)

feature_variants_fixed = {}  # Computed once, doesn't depend on train split
feature_variants_per_split = {}  # Will be computed inside split loop

# 1. Original features (split-independent)
feature_variants_fixed['original'] = {
    'features': X,
    'diagnostics': None
}
print('  ✓ Original features')

# 2-3. PCA/ZCA whitening - behavior depends on split type
if SPLIT_TYPE == 'fixed':
    # For fixed splits: compute once using the fixed train_idx
    try:
        X_pca, diag_pca = pca_whiten(X, train_idx)
        feature_variants_fixed['pca_whiten'] = {
            'features': X_pca,
            'diagnostics': diag_pca
        }
        print(f'  ✓ PCA whitening (rank-deficient: {diag_pca["rank_deficient"]} dims)')
    except Exception as e:
        print(f'  ✗ PCA whitening failed: {e}')
    
    try:
        X_zca, diag_zca = zca_whiten(X, train_idx)
        feature_variants_fixed['zca_whiten'] = {
            'features': X_zca,
            'diagnostics': diag_zca
        }
        print(f'  ✓ ZCA whitening (rank-deficient: {diag_zca["rank_deficient"]} dims)')
    except Exception as e:
        print(f'  ✗ ZCA whitening failed: {e}')
else:
    # For random splits: mark as per-split computation
    feature_variants_per_split['pca_whiten'] = True
    feature_variants_per_split['zca_whiten'] = True
    print('  ○ PCA whitening (will be computed per split)')
    print('  ○ ZCA whitening (will be computed per split)')

# 4. Full ZCA whitening (Wadia's extreme case - uses ALL data, split-independent)
try:
    X_full_zca, diag_full_zca = full_zca_whiten(X)
    feature_variants_fixed['full_zca_whiten'] = {
        'features': X_full_zca,
        'diagnostics': diag_full_zca
    }
    print(f'  ✓ Full ZCA whitening (WARNING: uses all data)')
except Exception as e:
    print(f'  ✗ Full ZCA whitening failed: {e}')

# 5. Rayleigh-Ritz restricted eigenvectors (uses graph structure, split-independent)
try:
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D, num_components)
    feature_variants_fixed['rayleigh_ritz'] = {
        'features': U.astype(np.float32),  # Ensure float32 for PyTorch consistency
        'diagnostics': {
            'method': 'Rayleigh-Ritz',
            'd_effective': d_eff,
            'ortho_error': float(ortho_err),
            'eigenvalues': eigenvalues.tolist()[:10]  # First 10 for logging
        }
    }
    print(f'  ✓ Rayleigh-Ritz (d_eff={d_eff}, ortho_err={ortho_err:.2e})')
except Exception as e:
    print(f'  ✗ Rayleigh-Ritz failed: {e}')

# ============================================================================
# Whitening Effect Analysis (for fixed variants only)
# ============================================================================

print('\n[5/7] Analyzing whitening effects...')

whitening_analysis = {}
for variant_name, variant_data in feature_variants_fixed.items():
    if variant_name != 'original' and variant_data.get('features') is not None:
        analysis = analyze_whitening_effect(X, variant_data['features'], variant_name)
        whitening_analysis[variant_name] = analysis

        print(f'\n  {variant_name}:')
        print(f'    Distance from I: {analysis["dist_from_I_original"]:.2f} → {analysis["dist_from_I_whitened"]:.2f}')
        print(f'    Condition number: {analysis["condition_original"]:.2f} → {analysis["condition_whitened"]:.2f}')
        print(f'    Uniformity: {analysis["uniformity_original"]*100:.1f}% → {analysis["uniformity_whitened"]*100:.1f}%')
        print(f'    Whitening indicators: {analysis["whitening_indicators"]}/3')

        # Warn about numerical instability
        if analysis["condition_whitened"] > 1e10:
            print(f'    ⚠️  WARNING: Extreme condition number ({analysis["condition_whitened"]:.2e}) may cause numerical issues')

if feature_variants_per_split:
    print(f'\n  Note: {list(feature_variants_per_split.keys())} will be analyzed per split')

# ============================================================================
# GRAM MATRIX ANALYSIS (Wadia Criterion Verification)
# ============================================================================

print('\n' + '='*70)
print('GRAM MATRIX ANALYSIS (Wadia Criterion Verification)')
print('='*70)
print(f'\nWadia\'s key insight: Gram matrix K = XX^T determines generalization')
print(f'  - Our data: n={n} samples, d={d} features')
print(f'  - Regime: {"n ≤ d → K should become identity (COLLAPSE EXPECTED!)" if n <= d else "n > d → K retains structure"}')

gram_matrix_analysis = {}

# Analyze original features
print('\n  Original features:')
gram_orig = analyze_gram_matrix(X, 'original')
gram_matrix_analysis['original'] = gram_orig
print(f'    K mean|eig-1|: {gram_orig["K_dist_from_1_mean"]:.4f}')
print(f'    K condition #: {gram_orig["K_condition_number"]:.2f}')

# Analyze Full ZCA (Wadia's method - most important!)
if 'full_zca_whiten' in feature_variants_fixed:
    print('\n  Full ZCA whitening (Wadia\'s method):')
    X_full_zca = feature_variants_fixed['full_zca_whiten']['features']
    gram_full_zca = analyze_gram_matrix(X_full_zca, 'full_zca')
    gram_matrix_analysis['full_zca'] = gram_full_zca
    print(f'    K mean|eig-1|: {gram_full_zca["K_dist_from_1_mean"]:.4f}')
    print(f'    K condition #: {gram_full_zca["K_condition_number"]:.2f}')
    print(f'    K ≈ Identity?: {gram_full_zca["K_is_approximately_identity"]}')

    # Wadia criterion check
    print('\n  *** WADIA CRITERION CHECK ***')
    if gram_full_zca["K_is_approximately_identity"]:
        print(f'    ✓ K IS approaching identity (mean|eig-1| = {gram_full_zca["K_dist_from_1_mean"]:.4f} < 0.1)')
        if n <= d:
            print(f'    → Wadia predicts COLLAPSE in n ≤ d regime')
            print(f'    → If no collapse observed, likely due to Adam optimizer vs standard GD')
    else:
        print(f'    ✗ K is NOT identity (mean|eig-1| = {gram_full_zca["K_dist_from_1_mean"]:.4f})')
        print(f'    → Whitening incomplete, may explain lack of collapse')

# Analyze Rayleigh-Ritz
if 'rayleigh_ritz' in feature_variants_fixed:
    print('\n  Rayleigh-Ritz:')
    X_rr = feature_variants_fixed['rayleigh_ritz']['features']
    gram_rr = analyze_gram_matrix(X_rr, 'rayleigh_ritz')
    gram_matrix_analysis['rayleigh_ritz'] = gram_rr
    print(f'    K mean|eig-1|: {gram_rr["K_dist_from_1_mean"]:.4f}')
    print(f'    K condition #: {gram_rr["K_condition_number"]:.2f}')

# ============================================================================
# Run All Experiments
# ============================================================================

print('\n[6/7] Running experiments...')

all_results = {}

# Define model configurations
# Note: use_scaler flag will be overridden for whitened features
model_configs = [
    ('Linear', LinearClassifier, {'num_classes': num_classes}, False),
    ('Linear+RowNorm', RowNormLinear, {'num_classes': num_classes}, False),
    ('MLP', StandardMLP, {'hidden_dim': HIDDEN_DIM, 'output_dim': num_classes}, True),
    ('MLP+RowNorm', RowNormMLP, {'hidden_dim': HIDDEN_DIM, 'output_dim': num_classes}, False),
]

# Whitened variants should NOT use StandardScaler (they're already normalized)
whitened_variants = {'pca_whiten', 'zca_whiten', 'full_zca_whiten', 'rayleigh_ritz'}

for split_iter in range(NUM_RANDOM_SPLITS):
    print(f'\n{"="*70}')
    print(f'SPLIT {split_iter+1}/{NUM_RANDOM_SPLITS}')
    print(f'{"="*70}')
    
    # Get split indices
    if SPLIT_TYPE == 'random':
        np.random.seed(split_iter)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        current_train_idx = indices[:train_size]
        current_val_idx = indices[train_size:train_size + val_size]
        current_test_idx = indices[train_size + val_size:]
    else:
        current_train_idx = train_idx
        current_val_idx = val_idx
        current_test_idx = test_idx
    
    # Build feature variants for this split
    # Start with fixed variants (don't depend on split)
    feature_variants = dict(feature_variants_fixed)
    
    # For random splits: compute train-whitening using THIS split's training data
    if SPLIT_TYPE == 'random' and feature_variants_per_split:
        print(f'\n  Computing train-whitening for this split...')
        
        if 'pca_whiten' in feature_variants_per_split:
            try:
                X_pca, diag_pca = pca_whiten(X, current_train_idx)
                feature_variants['pca_whiten'] = {
                    'features': X_pca,
                    'diagnostics': diag_pca
                }
                print(f'    ✓ PCA whitening (rank-deficient: {diag_pca["rank_deficient"]} dims)')
            except Exception as e:
                print(f'    ✗ PCA whitening failed: {e}')
        
        if 'zca_whiten' in feature_variants_per_split:
            try:
                X_zca, diag_zca = zca_whiten(X, current_train_idx)
                feature_variants['zca_whiten'] = {
                    'features': X_zca,
                    'diagnostics': diag_zca
                }
                print(f'    ✓ ZCA whitening (rank-deficient: {diag_zca["rank_deficient"]} dims)')
            except Exception as e:
                print(f'    ✗ ZCA whitening failed: {e}')
    
    # Run experiments for each feature variant
    for variant_name, variant_data in feature_variants.items():
        features_variant = variant_data['features']
        
        print(f'\n  Feature variant: {variant_name} (dim={features_variant.shape[1]})')
        
        for model_name, model_class, model_kwargs, use_scaler in model_configs:
            exp_key = f'{variant_name}_{model_name}'
            
            if exp_key not in all_results:
                all_results[exp_key] = []
            
            # Override use_scaler for whitened features (they're already normalized)
            actual_use_scaler = use_scaler and (variant_name not in whitened_variants)
            
            results = run_experiment(
                features_variant, labels,
                current_train_idx, current_val_idx, current_test_idx,
                num_classes, model_class, model_kwargs, NUM_SEEDS,
                use_standard_scaler=actual_use_scaler,
                description=f'{variant_name} + {model_name}'
            )
            
            all_results[exp_key].extend(results)
            
            agg = aggregate_results(results)
            print(f'    {model_name:15}: {agg["test_acc_mean"]*100:.2f}% ± {agg["test_acc_std"]*100:.2f}%')

# ============================================================================
# Aggregate and Analyze Results
# ============================================================================

print('\n[7/7] Aggregating results...')

final_results = {}
for exp_key, results in all_results.items():
    final_results[exp_key] = aggregate_results(results)

# ============================================================================
# Compute Key Metrics
# ============================================================================

print('\n' + '='*80)
print('RESULTS SUMMARY')
print('='*80)

# Organize results by feature variant and model
variants = ['original', 'pca_whiten', 'zca_whiten', 'full_zca_whiten', 'rayleigh_ritz']
models = ['Linear', 'Linear+RowNorm', 'MLP', 'MLP+RowNorm']

print('\n' + '-'*80)
print('TEST ACCURACY (%) - Mean ± Std')
print('-'*80)
print(f'{"Feature Variant":<20} {"Linear":>12} {"Linear+RN":>12} {"MLP":>12} {"MLP+RN":>12}')
print('-'*80)

for variant in variants:
    if f'{variant}_Linear' in final_results:
        row = f'{variant:<20}'
        for model in models:
            key = f'{variant}_{model}'
            if key in final_results:
                acc = final_results[key]['test_acc_mean'] * 100
                std = final_results[key]['test_acc_std'] * 100
                row += f' {acc:>5.2f}±{std:<4.2f}'
            else:
                row += f' {"N/A":>12}'
        print(row)

# ============================================================================
# Compute Whitening Damage and RowNorm Recovery
# ============================================================================

print('\n' + '-'*80)
print('WHITENING DAMAGE (Original - Whitened) in percentage points')
print('-'*80)

whitening_methods = ['pca_whiten', 'zca_whiten', 'full_zca_whiten', 'rayleigh_ritz']

for model in ['Linear', 'MLP']:
    print(f'\n{model}:')
    orig_key = f'original_{model}'
    if orig_key in final_results:
        orig_acc = final_results[orig_key]['test_acc_mean']
        
        for whitening in whitening_methods:
            white_key = f'{whitening}_{model}'
            if white_key in final_results:
                white_acc = final_results[white_key]['test_acc_mean']
                damage = (orig_acc - white_acc) * 100
                print(f'  {whitening:<20}: {damage:+.2f}pp {"(damaged)" if damage > 0 else "(improved)"}')

print('\n' + '-'*80)
print('ROWNORM RECOVERY (WithRowNorm - WithoutRowNorm) in percentage points')
print('-'*80)

for base_model, rownorm_model in [('Linear', 'Linear+RowNorm'), ('MLP', 'MLP+RowNorm')]:
    print(f'\n{base_model} → {rownorm_model}:')
    
    for variant in variants:
        base_key = f'{variant}_{base_model}'
        rownorm_key = f'{variant}_{rownorm_model}'
        
        if base_key in final_results and rownorm_key in final_results:
            base_acc = final_results[base_key]['test_acc_mean']
            rownorm_acc = final_results[rownorm_key]['test_acc_mean']
            recovery = (rownorm_acc - base_acc) * 100
            print(f'  {variant:<20}: {recovery:+.2f}pp {"(recovery)" if recovery > 0 else "(no benefit)"}')

# ============================================================================
# Key Finding: Does RowNorm Generalize?
# ============================================================================

# Note: Interpretation and conclusions are done in scripts/analyze_whitening_rownorm.py
# This script only collects and saves raw experimental results.

print('\n' + '='*80)
print('RESULTS SUMMARY')
print('='*80)

# Print raw results for quick reference (no interpretation)
print('\nTest Accuracies (see JSON for full details):')
for key, val in sorted(final_results.items()):
    print(f'  {key}: {val["test_acc_mean"]*100:.2f}% ± {val["test_acc_std"]*100:.2f}%')


# ============================================================================
# Save Results
# ============================================================================

results_dict = {
    'metadata': {
        'dataset': DATASET_NAME,
        'split_type': SPLIT_TYPE,
        'component_type': COMPONENT_TYPE,
        'k_diffusion': K_DIFFUSION,
        'num_nodes': num_nodes,
        'num_features': d,
        'num_classes': num_classes,
        'num_random_splits': NUM_RANDOM_SPLITS,
        'num_seeds': NUM_SEEDS,
        'epochs': EPOCHS,
        'whitening_eps': WHITENING_EPS,
        'wadia_regime': 'n <= d (collapse expected)' if n <= d else 'n > d (structure retained)',
        'timestamp': datetime.now().isoformat(),
    },
    'final_results': final_results,
    'whitening_analysis': whitening_analysis,
    'gram_matrix_analysis': gram_matrix_analysis,  # Wadia criterion verification
    # For random splits, per-split whitening diagnostics vary; we save fixed variants only
    'feature_diagnostics': {k: v.get('diagnostics') for k, v in feature_variants_fixed.items() if v.get('diagnostics')},
}

results_file = f'{output_base}/metrics/results.json'
with open(results_file, 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)
print(f'\n✓ Results saved to {results_file}')

print('\n' + '='*80)
print('EXPERIMENT COMPLETE')
print('='*80)
