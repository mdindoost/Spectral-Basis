"""
===================================================================================
INVESTIGATION 3: MAGNITUDE PRESERVATION IN SPECTRAL REPRESENTATIONS
===================================================================================

Research Question: Can we improve Part B by explicitly preserving magnitude 
information that RowNorm discards?

Hypothesis: x = m · x̂ where m = ||x||₂ (magnitude) and x̂ = x/||x|| (direction)
RowNorm discards m completely. Can we design classifiers that exploit both?

Framework Extension:
    Part A: Basis Sensitivity (SGC+MLP vs Restricted+StandardMLP)
    Part B: RowNorm Improvement (Restricted+StandardMLP vs Restricted+RowNorm)
    Part B.5: Magnitude-Aware (NEW - Does magnitude preservation help?)

Methods Compared:
    1. SGC Baseline (logistic regression)
    2. SGC + MLP (Part A baseline)
    3. Restricted + StandardMLP (Part A endpoint)
    4. Restricted + RowNorm (Part B - current best)
    5. Magnitude-Only MLP (NEW - ablation to test if magnitude is informative)
    6. Log-Magnitude Augmented MLP (NEW - simple magnitude preservation)
    7. Dual-Stream MLP (NEW - separate direction/magnitude processing)

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
    load_dataset, build_graph_matrices, StandardMLP, RowNormMLP
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Investigation 3: Magnitude Preservation')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--k', type=int, default=10,
                   help='Diffusion steps (default: 10)')
parser.add_argument('--splits', type=str, choices=['random', 'fixed'], default='fixed',
                   help='Use random or fixed splits')
parser.add_argument('--component', type=str, choices=['whole', 'lcc'], default='lcc',
                   help='Use whole graph or largest connected component')

args = parser.parse_args()

DATASET_NAME = args.dataset
K_DIFFUSION = args.k
SPLIT_TYPE = args.splits
COMPONENT_TYPE = args.component

# Experimental parameters (same as partAB)
NUM_RANDOM_SPLITS = 5 if SPLIT_TYPE == 'random' else 1
NUM_SEEDS = 5

# Training hyperparameters (same as partAB)
EPOCHS = 200
HIDDEN_DIM = 256
HIDDEN_MAG = 32  # For magnitude stream in dual-stream architecture
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Output
output_base = f'results/investigation3_magnitude/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}/k{K_DIFFUSION}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)
os.makedirs(f'{output_base}/diagnostics', exist_ok=True)

print('='*80)
print('INVESTIGATION 3: MAGNITUDE PRESERVATION IN SPECTRAL REPRESENTATIONS')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion k: {K_DIFFUSION}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# Models (following partAB.py pattern)
# ============================================================================

class SGC(nn.Module):
    """SGC: Logistic regression with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)


class MagnitudeOnlyMLP(nn.Module):
    """
    ABLATION: Use only magnitude ||x||₂ for classification.
    Tests if magnitude contains discriminative information.
    """
    def __init__(self, num_classes, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, X):
        M = torch.norm(X, dim=1, keepdim=True)
        return self.mlp(M)


class LogMagnitudeMLP(nn.Module):
    """
    Augment RowNorm features with log-magnitude: [x̂, log(||x||)]
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
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
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-10)
        M = torch.norm(X, dim=1, keepdim=True)
        log_M = torch.log(M + 1e-10)
        X_augmented = torch.cat([X_norm, log_M], dim=1)
        return self.mlp(X_augmented)


class DualStreamMLP(nn.Module):
    """
    Process direction and magnitude in separate streams.
    """
    def __init__(self, input_dim, hidden_dir, hidden_mag, num_classes):
        super().__init__()
        
        self.mlp_direction = nn.Sequential(
            nn.Linear(input_dim, hidden_dir),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dir, hidden_dir // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.mlp_magnitude = nn.Sequential(
            nn.Linear(1, hidden_mag),
            nn.ReLU(),
            nn.Linear(hidden_mag, hidden_mag)
        )
        
        self.classifier = nn.Linear(hidden_dir // 2 + hidden_mag, num_classes)
    
    def forward(self, X):
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-10)
        M = torch.norm(X, dim=1, keepdim=True)
        
        h_dir = self.mlp_direction(X_norm)
        h_mag = self.mlp_magnitude(M)
        
        h_combined = torch.cat([h_dir, h_mag], dim=1)
        return self.classifier(h_combined)


# ============================================================================
# Helper Functions (from partAB.py)
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
        for split_name, indices in split_idx.items():
            mask_indices = np.isin(indices, node_indices)
            old_indices = indices[mask_indices]
            new_indices = np.array([old_to_new[idx] for idx in old_indices])
            split_idx_sub[split_name] = new_indices
    
    return adj_sub, features_sub, labels_sub, split_idx_sub


def compute_sgc_normalized_adjacency(adj):
    """
    Compute SGC-style normalized adjacency: D^(-1/2) (A + I) D^(-1/2)
    EXACT COPY FROM partAB.py
    """
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops
    adj = sp.coo_matrix(adj)
    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def sgc_precompute(features, adj_normalized, degree):
    """
    Apply SGC precomputation: (D^(-1/2) (A+I) D^(-1/2))^k X
    EXACT COPY FROM partAB.py
    """
    for i in range(degree):
        features = adj_normalized @ features
    return features


def compute_restricted_eigenvectors(X, L, D, num_components=0):
    """
    Compute restricted eigenvectors with proper component handling
    EXACT COPY FROM partAB.py
    
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
    
    # Map back to full space
    U = Q @ V
    
    # Verify D-orthonormality
    G = U.T @ (D @ U)
    ortho_error = np.max(np.abs(G - np.eye(len(eigenvalues))))
    
    return U, eigenvalues, len(eigenvalues), ortho_error


# ============================================================================
# Diagnostic Functions
# ============================================================================

def analyze_magnitude_information(X, labels, num_classes, output_dir):
    """Phase 1 Diagnostics: Does magnitude contain discriminative information?"""
    print('\n' + '='*80)
    print('PHASE 1 DIAGNOSTICS: MAGNITUDE INFORMATION CONTENT')
    print('='*80)
    
    M = np.linalg.norm(X, axis=1)
    
    diagnostics = {
        'overall_mean': float(M.mean()),
        'overall_std': float(M.std()),
        'overall_min': float(M.min()),
        'overall_max': float(M.max()),
        'per_class': {}
    }
    
    print(f'\nOverall Magnitude Statistics:')
    print(f'  Mean: {M.mean():.4f}')
    print(f'  Std:  {M.std():.4f}')
    print(f'  Min:  {M.min():.4f}')
    print(f'  Max:  {M.max():.4f}')
    
    print(f'\nPer-Class Magnitude Analysis:')
    print(f'{"Class":<8} {"Count":<8} {"Mean":<12} {"Std":<12}')
    print('-' * 44)
    
    class_means = []
    class_stds = []
    
    for c in range(num_classes):
        mask = labels == c
        M_class = M[mask]
        
        class_mean = M_class.mean()
        class_std = M_class.std()
        
        class_means.append(class_mean)
        class_stds.append(class_std)
        
        diagnostics['per_class'][int(c)] = {
            'count': int(mask.sum()),
            'mean': float(class_mean),
            'std': float(class_std)
        }
        
        print(f'{c:<8} {mask.sum():<8} {class_mean:<12.4f} {class_std:<12.4f}')
    
    class_means = np.array(class_means)
    class_stds = np.array(class_stds)
    
    between_class_variance = class_means.var()
    within_class_variance = class_stds.mean()
    
    diagnostics['between_class_variance'] = float(between_class_variance)
    diagnostics['within_class_variance'] = float(within_class_variance)
    
    if within_class_variance > 0:
        fisher_score = between_class_variance / within_class_variance
    else:
        fisher_score = 0.0
    
    diagnostics['fisher_score'] = float(fisher_score)
    
    print(f'\nSeparability Analysis:')
    print(f'  Between-class variance: {between_class_variance:.4f}')
    print(f'  Within-class variance:  {within_class_variance:.4f}')
    print(f'  Fisher score:           {fisher_score:.4f}')
    
    if fisher_score > 0.1:
        print(f'  ✓ Magnitude appears discriminative (Fisher > 0.1)')
    else:
        print(f'  ✗ Magnitude may not be discriminative (Fisher < 0.1)')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    data_per_class = [M[labels == c] for c in range(num_classes)]
    bp = ax.boxplot(data_per_class, labels=range(num_classes))
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Magnitude ||x||₂', fontsize=12)
    ax.set_title('Magnitude Distribution per Class', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    x_pos = np.arange(num_classes)
    ax.bar(x_pos, class_means, yerr=class_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Mean Magnitude', fontsize=12)
    ax.set_title('Mean Magnitude per Class (±1 std)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range(num_classes))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/magnitude_per_class.png', dpi=150, bbox_inches='tight')
    print(f'\n✓ Saved: {output_dir}/magnitude_per_class.png')
    plt.close()
    
    with open(f'{output_dir}/magnitude_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f'✓ Saved: {output_dir}/magnitude_diagnostics.json')
    
    return diagnostics


# ============================================================================
# Training and Evaluation (from partAB.py)
# ============================================================================

def train_and_test(model, X_train, y_train, X_val, y_val, X_test, y_test,
                  epochs, lr, weight_decay, device, use_scheduler=False):
    """Standard training loop"""
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
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if use_scheduler:
            scheduler.step()
        
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


def aggregate_results(results):
    """Aggregate results across seeds"""
    if not results:
        return {'test_acc_mean': 0.0, 'test_acc_std': 0.0, 
                'val_acc_mean': 0.0, 'val_acc_std': 0.0}
    
    test_accs = [r['test_acc'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    
    return {
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs)
    }


# ============================================================================
# Experiment Functions (following partAB.py pattern)
# ============================================================================

def run_sgc_baseline(X_diffused, labels, train_idx, val_idx, test_idx,
                     num_classes, num_seeds, device):
    """SGC Baseline"""
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
    """SGC + MLP"""
    results = []
    
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
    """Restricted + StandardMLP"""
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D, num_components
    )
    
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


def run_restricted_rownorm(X_diffused, L, D, num_components, labels,
                           train_idx, val_idx, test_idx, num_classes,
                           num_seeds, device):
    """Restricted + RowNorm"""
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
        
        model = RowNormMLP(d_eff, HIDDEN_DIM, num_classes).to(device)
        
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


def run_magnitude_only(X_diffused, L, D, num_components, labels,
                       train_idx, val_idx, test_idx, num_classes,
                       num_seeds, device):
    """Magnitude-Only MLP (Ablation)"""
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
        
        model = MagnitudeOnlyMLP(num_classes, hidden_dim=64).to(device)
        
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


def run_log_magnitude(X_diffused, L, D, num_components, labels,
                     train_idx, val_idx, test_idx, num_classes,
                     num_seeds, device):
    """Log-Magnitude Augmented MLP"""
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
        
        model = LogMagnitudeMLP(d_eff, HIDDEN_DIM, num_classes).to(device)
        
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


def run_dual_stream(X_diffused, L, D, num_components, labels,
                   train_idx, val_idx, test_idx, num_classes,
                   num_seeds, device):
    """Dual-Stream MLP"""
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
        
        model = DualStreamMLP(d_eff, HIDDEN_DIM, HIDDEN_MAG, num_classes).to(device)
        
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


# ============================================================================
# Main Experiment Loop
# ============================================================================

print('\n[1/6] Loading dataset...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx, val_idx, test_idx) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Nodes: {num_nodes:,}, Features: {X_raw.shape[1]}, Classes: {num_classes}')

# Build graph matrices
print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)

# Extract LCC if requested (EXACT COPY FROM partAB.py)
if COMPONENT_TYPE == 'lcc':
    print('\n[3/6] Extracting largest connected component...')
    lcc_mask = get_largest_connected_component_nx(adj)
    
    split_idx_original = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_original
    )
    
    # Rebuild graph matrices for LCC (EXACT FROM partAB.py)
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

# Compute normalized adjacency (SAME AS partAB.py)
print('\n[4/6] Computing SGC-style normalized adjacency...')
A_sgc = compute_sgc_normalized_adjacency(adj)
print('✓ A_sgc = D^(-1/2) (A + I) D^(-1/2)')

# Diffuse features (SAME AS partAB.py)
print(f'\n[5/6] Precomputing diffusion (k={K_DIFFUSION})...')
features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw
X_diffused = sgc_precompute(features_dense.copy(), A_sgc, K_DIFFUSION)
print(f'✓ X_diffused shape: {X_diffused.shape}')

# Run diagnostics
print(f'\n[6/6] Running Phase 1 Diagnostics...')
diagnostics_diffused = analyze_magnitude_information(
    X_diffused, labels, num_classes, f'{output_base}/diagnostics'
)

# Initialize results storage
all_results = {
    'dataset': DATASET_NAME,
    'k_diffusion': K_DIFFUSION,
    'split_type': SPLIT_TYPE,
    'num_nodes': num_nodes,
    'num_classes': num_classes,
    'num_seeds': NUM_SEEDS,
    'diagnostics': {'diffused_features': diagnostics_diffused}
}

experiments = {
    'sgc_baseline': [],
    'sgc_mlp_baseline': [],
    'restricted_standard_mlp': [],
    'restricted_rownorm_mlp': [],
    'magnitude_only': [],
    'log_magnitude': [],
    'dual_stream': []
}

metadata = {}

# Run experiments
print(f'\n{"="*80}')
print(f'RUNNING EXPERIMENTS ({NUM_RANDOM_SPLITS} split(s) × {NUM_SEEDS} seeds)')
print(f'{"="*80}')

for split_idx_iter in range(NUM_RANDOM_SPLITS):
    print(f'\n{"="*80}')
    print(f'SPLIT {split_idx_iter + 1}/{NUM_RANDOM_SPLITS}')
    print(f'{"="*80}')
    
    if SPLIT_TYPE == 'fixed':
        train_idx_cur = train_idx
        val_idx_cur = val_idx
        test_idx_cur = test_idx
    else:
        np.random.seed(split_idx_iter)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx_cur = indices[:train_size]
        val_idx_cur = indices[train_size:train_size + val_size]
        test_idx_cur = indices[train_size + val_size:]
    
    print(f'Train: {len(train_idx_cur)}, Val: {len(val_idx_cur)}, Test: {len(test_idx_cur)}')
    
    # Experiment 1
    print('\n[1/7] SGC Baseline (Logistic)')
    sgc_results = run_sgc_baseline(
        X_diffused, labels, train_idx_cur, val_idx_cur, test_idx_cur,
        num_classes, NUM_SEEDS, device
    )
    experiments['sgc_baseline'].extend(sgc_results)
    sgc_agg = aggregate_results(sgc_results)
    print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')
    
    # Experiment 2
    print('\n[2/7] SGC + MLP')
    sgc_mlp_results = run_sgc_mlp_baseline(
        X_diffused, labels, train_idx_cur, val_idx_cur, test_idx_cur,
        num_classes, NUM_SEEDS, device
    )
    experiments['sgc_mlp_baseline'].extend(sgc_mlp_results)
    sgc_mlp_agg = aggregate_results(sgc_mlp_results)
    print(f'→ {sgc_mlp_agg["test_acc_mean"]*100:.2f}% ± {sgc_mlp_agg["test_acc_std"]*100:.2f}%')
    
    # Experiment 3
    print('\n[3/7] Restricted + StandardMLP')
    restricted_std_results, d_eff, ortho_err = run_restricted_standard_mlp(
        X_diffused, L, D, num_components, labels,
        train_idx_cur, val_idx_cur, test_idx_cur, num_classes,
        NUM_SEEDS, device
    )
    experiments['restricted_standard_mlp'].extend(restricted_std_results)
    metadata['d_restricted'] = d_eff
    metadata['ortho_error'] = float(ortho_err)
    restricted_std_agg = aggregate_results(restricted_std_results)
    print(f'→ {restricted_std_agg["test_acc_mean"]*100:.2f}% ± {restricted_std_agg["test_acc_std"]*100:.2f}%')
    print(f'  D-orthonormality error: {ortho_err:.2e}')
    
    # Experiment 4
    print('\n[4/7] Restricted + RowNorm')
    rownorm_results, _, _ = run_restricted_rownorm(
        X_diffused, L, D, num_components, labels,
        train_idx_cur, val_idx_cur, test_idx_cur, num_classes,
        NUM_SEEDS, device
    )
    experiments['restricted_rownorm_mlp'].extend(rownorm_results)
    rownorm_agg = aggregate_results(rownorm_results)
    print(f'→ {rownorm_agg["test_acc_mean"]*100:.2f}% ± {rownorm_agg["test_acc_std"]*100:.2f}%')
    
    # Experiment 5
    print('\n[5/7] Magnitude-Only MLP (Ablation)')
    magnitude_only_results, _, _ = run_magnitude_only(
        X_diffused, L, D, num_components, labels,
        train_idx_cur, val_idx_cur, test_idx_cur, num_classes,
        NUM_SEEDS, device
    )
    experiments['magnitude_only'].extend(magnitude_only_results)
    mag_only_agg = aggregate_results(magnitude_only_results)
    print(f'→ {mag_only_agg["test_acc_mean"]*100:.2f}% ± {mag_only_agg["test_acc_std"]*100:.2f}%')
    
    # Experiment 6
    print('\n[6/7] Log-Magnitude Augmented MLP')
    log_mag_results, _, _ = run_log_magnitude(
        X_diffused, L, D, num_components, labels,
        train_idx_cur, val_idx_cur, test_idx_cur, num_classes,
        NUM_SEEDS, device
    )
    experiments['log_magnitude'].extend(log_mag_results)
    log_mag_agg = aggregate_results(log_mag_results)
    print(f'→ {log_mag_agg["test_acc_mean"]*100:.2f}% ± {log_mag_agg["test_acc_std"]*100:.2f}%')
    
    # Experiment 7
    print('\n[7/7] Dual-Stream MLP')
    dual_stream_results, _, _ = run_dual_stream(
        X_diffused, L, D, num_components, labels,
        train_idx_cur, val_idx_cur, test_idx_cur, num_classes,
        NUM_SEEDS, device
    )
    experiments['dual_stream'].extend(dual_stream_results)
    dual_agg = aggregate_results(dual_stream_results)
    print(f'→ {dual_agg["test_acc_mean"]*100:.2f}% ± {dual_agg["test_acc_std"]*100:.2f}%')

# Aggregate and save results
print(f'\n{"="*80}')
print('FINAL RESULTS SUMMARY')
print(f'{"="*80}')

final_results = {}
for exp_name, exp_results in experiments.items():
    final_results[exp_name] = aggregate_results(exp_results)

print(f'\n{"Method":<30} {"Test Acc":<15} {"Std":<10}')
print('-' * 55)

method_names = {
    'sgc_baseline': 'SGC Baseline (Logistic)',
    'sgc_mlp_baseline': 'SGC + MLP',
    'restricted_standard_mlp': 'Restricted + StandardMLP',
    'restricted_rownorm_mlp': 'Restricted + RowNorm',
    'magnitude_only': 'Magnitude-Only (Ablation)',
    'log_magnitude': 'Log-Magnitude Augmented',
    'dual_stream': 'Dual-Stream'
}

for exp_name, display_name in method_names.items():
    agg = final_results[exp_name]
    print(f'{display_name:<30} {agg["test_acc_mean"]*100:>6.2f}%        {agg["test_acc_std"]*100:>5.2f}%')

# Part A/B/B.5 Analysis
print(f'\n{"="*80}')
print('PART A/B/B.5 FRAMEWORK ANALYSIS')
print(f'{"="*80}')

sgc_mlp_acc = final_results['sgc_mlp_baseline']['test_acc_mean'] * 100
restricted_std_acc = final_results['restricted_standard_mlp']['test_acc_mean'] * 100
rownorm_acc = final_results['restricted_rownorm_mlp']['test_acc_mean'] * 100
log_mag_acc = final_results['log_magnitude']['test_acc_mean'] * 100
dual_stream_acc = final_results['dual_stream']['test_acc_mean'] * 100

part_a = restricted_std_acc - sgc_mlp_acc
part_b_rownorm = rownorm_acc - restricted_std_acc
part_b_log_mag = log_mag_acc - restricted_std_acc
part_b_dual = dual_stream_acc - restricted_std_acc

gap_rownorm = rownorm_acc - sgc_mlp_acc
gap_log_mag = log_mag_acc - sgc_mlp_acc
gap_dual = dual_stream_acc - sgc_mlp_acc

print(f'\nPart A (Basis Sensitivity):')
print(f'  SGC+MLP → Restricted+StandardMLP: {part_a:+.2f}pp')

print(f'\nPart B (Spectral-Optimality):')
print(f'  RowNorm improvement:       {part_b_rownorm:+.2f}pp')
print(f'  Log-Magnitude improvement: {part_b_log_mag:+.2f}pp')
print(f'  Dual-Stream improvement:   {part_b_dual:+.2f}pp')

if part_b_log_mag > part_b_rownorm:
    print(f'  ✓ Log-Magnitude beats RowNorm by {part_b_log_mag - part_b_rownorm:.2f}pp')
if part_b_dual > part_b_rownorm:
    print(f'  ✓ Dual-Stream beats RowNorm by {part_b_dual - part_b_rownorm:.2f}pp')

print(f'\nThe Gap (Final vs Baseline):')
print(f'  RowNorm:       {gap_rownorm:+.2f}pp')
print(f'  Log-Magnitude: {gap_log_mag:+.2f}pp')
print(f'  Dual-Stream:   {gap_dual:+.2f}pp')

# Save results
all_results['experiments'] = final_results
all_results['metadata'] = metadata
all_results['framework_analysis'] = {
    'part_a': float(part_a),
    'part_b_rownorm': float(part_b_rownorm),
    'part_b_log_magnitude': float(part_b_log_mag),
    'part_b_dual_stream': float(part_b_dual),
    'gap_rownorm': float(gap_rownorm),
    'gap_log_magnitude': float(gap_log_mag),
    'gap_dual_stream': float(gap_dual)
}

with open(f'{output_base}/metrics/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n✓ Saved results: {output_base}/metrics/results.json')
print(f'\n{"="*80}')
print('EXPERIMENT COMPLETE')
print(f'{"="*80}')