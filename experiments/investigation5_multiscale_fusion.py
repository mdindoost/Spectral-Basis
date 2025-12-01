"""
===================================================================================
INVESTIGATION 5: MULTI-SCALE FEATURE FUSION
===================================================================================

Research Question: Does combining low-diffusion features (k=2) with high-diffusion
restricted eigenvectors (k=10) improve over either scale alone?

Hypothesis: Different diffusion scales capture complementary information:
    - Low diffusion (k=2): Preserves local structure and original features
    - High diffusion (k=10): Captures global structure via restricted eigenvectors
    - Fusion: Combines both scales for superior performance

Architecture:
    Branch 1 (Local):  SGC k=2 → Linear(bias=True) → ReLU → h1
    Branch 2 (Global): Diffuse k=10 → Restricted Eigenvectors → RowNorm 
                       → Linear(bias=False) → ReLU → h2
    Fusion: Concat[h1, h2] → 2-Layer MLP → Softmax

Key Design Choices:
    - Branch 1 WITH bias: SGC features can have arbitrary offset
    - Branch 2 WITHOUT bias: Preserves unit sphere geometry from RowNorm
    - Standard RowNorm (α=0): Testing multi-scale, not normalization variants

Baselines:
    1. SGC Baseline: Logistic regression on SGC k=2 features
    2. Branch 1 Only: SGC k=2 → Full MLP (with bias)
    3. Branch 2 Only: Restricted k=10 + RowNorm → Full MLP (no bias)
    4. Dual-Branch: Both branches fused

Author: Mohammad
Date: December 2025
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

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import load_dataset, build_graph_matrices

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Investigation 5: Multi-Scale Feature Fusion')
parser.add_argument('--dataset', type=str, default='coauthor-cs',
                   help='Dataset name')
parser.add_argument('--k_low', type=int, default=2,
                   help='Low diffusion steps (local features)')
parser.add_argument('--k_high', type=int, default=10,
                   help='High diffusion steps (global features)')
parser.add_argument('--split_type', type=str, choices=['fixed', 'random'],
                   default='fixed', help='Type of data split')
parser.add_argument('--component_type', type=str, choices=['lcc', 'full'],
                   default='lcc', help='Use largest connected component or full graph')
parser.add_argument('--num_seeds', type=int, default=5,
                   help='Number of random seeds')
parser.add_argument('--device', type=str, 
                   default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

DATASET_NAME = args.dataset
K_LOW = args.k_low
K_HIGH = args.k_high
SPLIT_TYPE = args.split_type
COMPONENT_TYPE = args.component_type
NUM_SEEDS = args.num_seeds
DEVICE = torch.device(args.device)

# Hyperparameters (same as previous investigations)
HIDDEN_DIM = 256
EPOCHS = 200  # Corrected from 500
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

print(f'\n{"="*80}')
print('INVESTIGATION 5: MULTI-SCALE FEATURE FUSION')
print(f'{"="*80}')
print(f'Dataset: {DATASET_NAME}')
print(f'Diffusion scales: k_low={K_LOW}, k_high={K_HIGH}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Device: {DEVICE}')
print(f'{"="*80}\n')

# ============================================================================
# Model Definitions (EXACT MATCH to previous investigations)
# ============================================================================

class SGC(nn.Module):
    """SGC: Logistic regression with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)


class Branch1MLP(nn.Module):
    """Branch 1 Only: SGC k=2 + Full MLP (WITH bias)"""
    def __init__(self, d_features, hidden_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_features, hidden_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        return self.mlp(h)


class Branch2MLP(nn.Module):
    """Branch 2 Only: Restricted k=10 + RowNorm + Full MLP (NO bias)"""
    def __init__(self, d_features, hidden_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_features, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        return self.mlp(h)


class DualBranchMLP(nn.Module):
    """Dual-Branch: Fusion of Branch 1 (WITH bias) and Branch 2 (NO bias)"""
    def __init__(self, d_sgc, d_eigenvec, hidden_dim, num_classes):
        super().__init__()
        self.branch1_linear = nn.Linear(d_sgc, hidden_dim, bias=True)
        self.branch2_linear = nn.Linear(d_eigenvec, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x_sgc, x_rownorm):
        h1 = F.relu(self.branch1_linear(x_sgc))
        h2 = F.relu(self.branch2_linear(x_rownorm))
        h_combined = torch.cat([h1, h2], dim=1)
        return self.mlp(h_combined)


# ============================================================================
# Feature Preparation (EXACT MATCH to previous investigations)
# ============================================================================

def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency: D^(-1/2) (A + I) D^(-1/2)"""
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops
    adj = sp.coo_matrix(adj)
    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def diffuse_features(X, A_norm, k):
    """k-step diffusion: A_norm^k @ X where A_norm = D^{-1/2}(A+I)D^{-1/2}"""
    X_diffused = X.copy()
    for _ in range(k):
        X_diffused = A_norm @ X_diffused
    return X_diffused


def compute_restricted_eigenvectors(X, L, D, num_components=0):
    """
    Compute restricted eigenvectors (EXACT MATCH to partAB.py)
    
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
    
    # Regularize (CRITICAL for numerical stability)
    eps_base = 1e-10
    eps = eps_base * np.trace(D_r) / d_effective
    D_r = D_r + eps * np.eye(d_effective)
    
    # Solve generalized eigenproblem
    eigenvalues, V = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
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


def prepare_multiscale_features(X, A, L, D, num_components, k_low, k_high):
    """Prepare features for dual-branch architecture"""
    # Compute SGC-style normalized adjacency (EXACT MATCH to previous code)
    A_norm = compute_sgc_normalized_adjacency(A)
    
    # Branch 1: Low-diffusion SGC
    print(f'\nBranch 1: Computing SGC k={k_low} features...')
    X_sgc_low = diffuse_features(X, A_norm, k_low)
    print(f'  Shape: {X_sgc_low.shape}')
    
    # Branch 2: High-diffusion restricted eigenvectors
    print(f'\nBranch 2: Computing Restricted Eigenvectors k={k_high} + RowNorm...')
    X_diffused_high = diffuse_features(X, A_norm, k_high)
    U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(
        X_diffused_high, L, D, num_components
    )
    print(f'  Eigenvectors shape: {U.shape}')
    print(f'  D-orthonormality error: {ortho_error:.2e}')
    
    if ortho_error > 1e-6:
        print(f'  ⚠ Warning: Orthonormality error {ortho_error:.2e} > 1e-6')
    
    # Standard RowNorm (α=0)
    V_rownorm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
    
    return X_sgc_low, V_rownorm, eigenvalues, ortho_error


# ============================================================================
# Training (EXACT MATCH to previous investigations - NO EARLY STOPPING!)
# ============================================================================

def train_and_test(model, X_train, y_train, X_val, y_val, X_test, y_test,
                  epochs, lr, weight_decay, device):
    """Train and evaluate model (single-branch only)"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        
        # Evaluate every 10 epochs (like original code)
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
    test_accs = [r['test_acc'] for r in results]
    return {
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        'val_acc_mean': np.mean([r['val_acc'] for r in results]),
        'val_acc_std': np.std([r['val_acc'] for r in results])
    }


# ============================================================================
# Experiment Runners (EXACT MATCH to previous investigations)
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
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
    
    return results


def run_branch1_mlp(X_sgc, labels, train_idx, val_idx, test_idx,
                   num_classes, num_seeds, device):
    """Experiment: Branch 1 Only (SGC k=2 + MLP with bias)"""
    results = []
    
    X_train = X_sgc[train_idx]
    y_train = labels[train_idx]
    X_val = X_sgc[val_idx]
    y_val = labels[val_idx]
    X_test = X_sgc[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = Branch1MLP(X_sgc.shape[1], HIDDEN_DIM, num_classes).to(device)
        
        val_acc, test_acc, train_time = train_and_test(
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


def run_branch2_mlp(V_rownorm, labels, train_idx, val_idx, test_idx,
                   num_classes, num_seeds, device):
    """Experiment: Branch 2 Only (Restricted k=10 + RowNorm + MLP no bias)"""
    results = []
    
    X_train = V_rownorm[train_idx]
    y_train = labels[train_idx]
    X_val = V_rownorm[val_idx]
    y_val = labels[val_idx]
    X_test = V_rownorm[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = Branch2MLP(V_rownorm.shape[1], HIDDEN_DIM, num_classes).to(device)
        
        val_acc, test_acc, train_time = train_and_test(
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


def run_dual_branch_mlp(X_sgc, V_rownorm, labels, train_idx, val_idx, test_idx,
                       num_classes, num_seeds, device):
    """Experiment: Dual-Branch (Multi-Scale Fusion)"""
    results = []
    
    # Prepare training data
    X_train_sgc = X_sgc[train_idx]
    X_train_rn = V_rownorm[train_idx]
    y_train = labels[train_idx]
    
    X_val_sgc = X_sgc[val_idx]
    X_val_rn = V_rownorm[val_idx]
    y_val = labels[val_idx]
    
    X_test_sgc = X_sgc[test_idx]
    X_test_rn = V_rownorm[test_idx]
    y_test = labels[test_idx]
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = DualBranchMLP(X_sgc.shape[1], V_rownorm.shape[1], 
                             HIDDEN_DIM, num_classes).to(device)
        
        # Convert to tensors
        X_train_sgc_t = torch.FloatTensor(X_train_sgc).to(device)
        X_train_rn_t = torch.FloatTensor(X_train_rn).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        X_val_sgc_t = torch.FloatTensor(X_val_sgc).to(device)
        X_val_rn_t = torch.FloatTensor(X_val_rn).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
        
        X_test_sgc_t = torch.FloatTensor(X_test_sgc).to(device)
        X_test_rn_t = torch.FloatTensor(X_test_rn).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)
        
        # Training (NO early stopping, like original code)
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=LEARNING_RATE, 
                                    weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_test_acc = 0.0
        
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_sgc_t, X_train_rn_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Evaluate every 10 epochs (like original)
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_sgc_t, X_val_rn_t)
                    test_logits = model(X_test_sgc_t, X_test_rn_t)
                    val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
                    test_acc = (test_logits.argmax(1) == y_test_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
        
        train_time = time.time() - start_time
        
        results.append({
            'seed': seed,
            'val_acc': best_val_acc,
            'test_acc': best_test_acc,
            'train_time': train_time
        })
    
    return results


# ============================================================================
# Main Experiment
# ============================================================================

output_base = f'results/investigation5_multiscale_fusion/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}/k{K_LOW}_k{K_HIGH}'
os.makedirs(f'{output_base}/metrics', exist_ok=True)
os.makedirs(f'{output_base}/plots', exist_ok=True)

print('\n[1/6] Loading dataset...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx, val_idx, test_idx) = load_dataset(DATASET_NAME, root='./dataset')
print(f'Nodes: {num_nodes:,}, Features: {X_raw.shape[1]}, Classes: {num_classes}')

print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)

# Extract LCC if requested
if COMPONENT_TYPE == 'lcc':
    print('\n[3/6] Extracting largest connected component...')
    
    # Convert edge_index to edge list (handle both numpy and torch)
    if isinstance(edge_index, np.ndarray):
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        edge_list = edge_index.T
    else:
        edge_list = edge_index.t().numpy()
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Find components
    components = list(nx.connected_components(G))
    num_components = len(components)
    print(f'  Total components: {num_components}')
    
    if num_components > 1:
        lcc = max(components, key=len)
        lcc_nodes = sorted(list(lcc))
        print(f'  LCC size: {len(lcc_nodes)} / {num_nodes} ({100*len(lcc_nodes)/num_nodes:.1f}%)')
        
        # Create mapping
        old_to_new = {old: new for new, old in enumerate(lcc_nodes)}
        
        # Filter data
        X_raw = X_raw[lcc_nodes]
        labels = labels[lcc_nodes]
        
        # Remap splits
        def remap_idx(idx):
            return np.array([old_to_new[i] for i in idx if i in old_to_new])
        
        if SPLIT_TYPE == 'fixed':
            train_idx = remap_idx(train_idx)
            val_idx = remap_idx(val_idx)
            test_idx = remap_idx(test_idx)
        
        # Rebuild graph
        subgraph_edges = []
        for i, j in edge_list:
            if i in old_to_new and j in old_to_new:
                subgraph_edges.append([old_to_new[i], old_to_new[j]])
        
        subgraph_edges = np.array(subgraph_edges).T
        num_nodes = len(lcc_nodes)
        adj, D, L = build_graph_matrices(torch.tensor(subgraph_edges), num_nodes)
        print(f'  New graph: {num_nodes} nodes, {subgraph_edges.shape[1]} edges')
        num_components = 1
    else:
        num_components = 1
        print('  Graph is already connected')
else:
    num_components = nx.number_connected_components(nx.from_edgelist(
        edge_index.T if isinstance(edge_index, np.ndarray) else edge_index.t().numpy()
    ))
    print(f'\n[3/6] Using full graph ({num_components} components)')

# Random splits
if SPLIT_TYPE == 'random':
    print('\n[4/6] Creating random splits...')
    np.random.seed(42)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
else:
    print(f'\n[4/6] Using fixed splits')
    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

# Prepare features
print('\n[5/6] Preparing multi-scale features...')
X_sgc_low, V_rownorm, eigenvalues, ortho_error = prepare_multiscale_features(
    X_raw, adj, L, D, num_components, K_LOW, K_HIGH
)

metadata = {
    'dataset': DATASET_NAME,
    'num_nodes': int(num_nodes),
    'num_classes': int(num_classes),
    'k_low': int(K_LOW),
    'k_high': int(K_HIGH),
    'd_sgc': int(X_sgc_low.shape[1]),
    'd_eigenvec': int(V_rownorm.shape[1]),
    'ortho_error': float(ortho_error)
}

# Run experiments
print('\n[6/6] Running experiments...\n')

print('[1/4] SGC Baseline...')
sgc_results = run_sgc_baseline(
    X_sgc_low, labels, train_idx, val_idx, test_idx,
    num_classes, NUM_SEEDS, DEVICE
)
sgc_agg = aggregate_results(sgc_results)
print(f'→ {sgc_agg["test_acc_mean"]*100:.2f}% ± {sgc_agg["test_acc_std"]*100:.2f}%')

print('\n[2/4] Branch 1 Only (SGC k=2 + MLP)...')
b1_results = run_branch1_mlp(
    X_sgc_low, labels, train_idx, val_idx, test_idx,
    num_classes, NUM_SEEDS, DEVICE
)
b1_agg = aggregate_results(b1_results)
print(f'→ {b1_agg["test_acc_mean"]*100:.2f}% ± {b1_agg["test_acc_std"]*100:.2f}%')

print('\n[3/4] Branch 2 Only (Restricted k=10 + RowNorm)...')
b2_results = run_branch2_mlp(
    V_rownorm, labels, train_idx, val_idx, test_idx,
    num_classes, NUM_SEEDS, DEVICE
)
b2_agg = aggregate_results(b2_results)
print(f'→ {b2_agg["test_acc_mean"]*100:.2f}% ± {b2_agg["test_acc_std"]*100:.2f}%')

print('\n[4/4] Dual-Branch (Multi-Scale Fusion)...')
dual_results = run_dual_branch_mlp(
    X_sgc_low, V_rownorm, labels, train_idx, val_idx, test_idx,
    num_classes, NUM_SEEDS, DEVICE
)
dual_agg = aggregate_results(dual_results)
print(f'→ {dual_agg["test_acc_mean"]*100:.2f}% ± {dual_agg["test_acc_std"]*100:.2f}%')

# Results summary
print(f'\n{"="*80}')
print('RESULTS SUMMARY')
print(f'{"="*80}\n')
print(f'{"Method":<40} {"Test Acc":<15} {"Std"}')
print('-' * 65)
print(f'{"SGC Baseline":<40} {sgc_agg["test_acc_mean"]*100:>6.2f}%        {sgc_agg["test_acc_std"]*100:>5.2f}%')
print(f'{"Branch 1 Only (SGC k=2 + MLP)":<40} {b1_agg["test_acc_mean"]*100:>6.2f}%        {b1_agg["test_acc_std"]*100:>5.2f}%')
print(f'{"Branch 2 Only (Restricted k=10 + RowNorm)":<40} {b2_agg["test_acc_mean"]*100:>6.2f}%        {b2_agg["test_acc_std"]*100:>5.2f}%')
print(f'{"Dual-Branch (Multi-Scale Fusion)":<40} {dual_agg["test_acc_mean"]*100:>6.2f}%        {dual_agg["test_acc_std"]*100:>5.2f}%')

# Analysis
print(f'\n{"="*80}')
print('FUSION ANALYSIS')
print(f'{"="*80}\n')

sgc_acc = sgc_agg['test_acc_mean'] * 100
b1_acc = b1_agg['test_acc_mean'] * 100
b2_acc = b2_agg['test_acc_mean'] * 100
dual_acc = dual_agg['test_acc_mean'] * 100
best_single = max(b1_acc, b2_acc)
fusion_gain = dual_acc - best_single

print(f'Best single branch:  {best_single:.2f}%')
print(f'Dual-branch:         {dual_acc:.2f}%')
print(f'Fusion gain:         {fusion_gain:+.2f}pp\n')

if fusion_gain > 0.5:
    print(f'✓ Multi-scale fusion HELPS! ({fusion_gain:+.2f}pp)')
elif fusion_gain > -0.5:
    print(f'≈ Multi-scale fusion neutral (±{abs(fusion_gain):.2f}pp)')
else:
    print(f'✗ Multi-scale fusion hurts ({fusion_gain:.2f}pp)')

# Save results
results_dict = {
    'metadata': metadata,
    'results': {
        'sgc_baseline': sgc_agg,
        'branch1_only': b1_agg,
        'branch2_only': b2_agg,
        'dual_branch': dual_agg
    },
    'analysis': {
        'best_single_branch': float(best_single),
        'dual_branch': float(dual_acc),
        'fusion_gain': float(fusion_gain)
    }
}

with open(f'{output_base}/metrics/results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f'\n✓ Results saved to: {output_base}/metrics/results.json')
print(f'\n{"="*80}')
print('EXPERIMENT COMPLETE')
print(f'{"="*80}')