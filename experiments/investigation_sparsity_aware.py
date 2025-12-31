"""
Investigation: Sparsity-Aware Classifiers
==========================================

Based on our finding that sparsity of U strongly predicts whether magnitude
preservation helps or hurts (r = -0.63, p = 0.005), we test classifiers that
adapt to sparsity.

Key Insight:
- Dense U → magnitude is stable/reliable → use it
- Sparse U → magnitude is noisy → ignore it

New Methods:
1. SparsityAwareMLP: Weights magnitude by row density
2. RobustMagnitudeMLP: Uses median instead of L2 norm (robust to outliers)
3. AdaptiveMagnitudeMLP: Learns when to trust magnitude

Baseline Comparisons:
- StandardMLP: No normalization
- RowNormMLP: Discards all magnitude
- LogMagnitudeMLP: Always adds magnitude (Hordan's approach)

Usage:
    python investigation_sparsity_aware.py
    python investigation_sparsity_aware.py --datasets cora citeseer pubmed
    python investigation_sparsity_aware.py --k_diffusion 2 10
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import from existing utils.py
from utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
    StandardMLP,
    RowNormMLP,
    LogMagnitudeMLP,
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Sparsity-Aware Classifier Investigation')
parser.add_argument('--datasets', nargs='+', type=str, 
                   default=['cora', 'citeseer', 'pubmed', 'wikics', 
                           'amazon-computers', 'amazon-photo',
                           'coauthor-cs', 'coauthor-physics', 'ogbn-arxiv'],
                   help='Datasets to analyze')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 10],
                   help='Diffusion steps to test')
parser.add_argument('--num_splits', type=int, default=5,
                   help='Number of random splits')
parser.add_argument('--num_seeds', type=int, default=3,
                   help='Number of seeds per split')
parser.add_argument('--output_dir', type=str, default='results/sparsity_aware',
                   help='Output directory')
args = parser.parse_args()

DATASETS = args.datasets
K_DIFFUSION_VALUES = args.k_diffusion
NUM_SPLITS = args.num_splits
NUM_SEEDS = args.num_seeds
OUTPUT_DIR = args.output_dir

# Training parameters
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

print('='*80)
print('SPARSITY-AWARE CLASSIFIER INVESTIGATION')
print('='*80)
print(f'Datasets: {DATASETS}')
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'Splits: {NUM_SPLITS}, Seeds per split: {NUM_SEEDS}')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# New Classifier Architectures
# ============================================================================

class SparsityAwareMLP(nn.Module):
    """
    Sparsity-Aware MLP: Weights magnitude contribution by row density
    
    Key idea: Trust magnitude only when the row is dense (few zeros)
    
    For each row:
        sparsity = fraction of near-zero entries
        density = 1 - sparsity
        
        x_normalized = x / ||x||
        log_magnitude = log(||x||)
        weighted_magnitude = density * log_magnitude
        
        output = MLP([x_normalized, weighted_magnitude])
    """
    def __init__(self, input_dim, hidden_dim, num_classes, sparsity_threshold=1e-6):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        
        # MLP with one extra input (weighted log-magnitude)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Compute per-row sparsity (fraction of near-zero entries)
        near_zero = (torch.abs(x) < self.sparsity_threshold).float()
        sparsity = near_zero.mean(dim=1, keepdim=True)  # [n, 1]
        density = 1.0 - sparsity  # [n, 1]
        
        # Row normalization
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Log-magnitude weighted by density
        log_magnitude = torch.log(x_norm + 1e-8)
        weighted_magnitude = density * log_magnitude
        
        # Concatenate
        x_augmented = torch.cat([x_normalized, weighted_magnitude], dim=1)
        
        # MLP
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class SparsityAwareMLP_V2(nn.Module):
    """
    Version 2: Also weight the normalized features by density
    
    Sparse rows get more aggressive normalization (pure direction)
    Dense rows keep more of their original scale information
    """
    def __init__(self, input_dim, hidden_dim, num_classes, sparsity_threshold=1e-6):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Compute per-row density
        near_zero = (torch.abs(x) < self.sparsity_threshold).float()
        sparsity = near_zero.mean(dim=1, keepdim=True)
        density = 1.0 - sparsity
        
        # Soft interpolation between original and normalized
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Sparse → use normalized, Dense → blend with original scale
        # x_blended = density * x + (1 - density) * x_normalized
        # Simplified: just use normalized but weight magnitude
        
        log_magnitude = torch.log(x_norm + 1e-8)
        weighted_magnitude = density * log_magnitude
        
        x_augmented = torch.cat([x_normalized, weighted_magnitude], dim=1)
        
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class RobustMagnitudeMLP(nn.Module):
    """
    Robust Magnitude MLP: Uses median absolute value instead of L2 norm
    
    L2 norm is sensitive to outliers (a few large values dominate)
    Median is robust - represents "typical" entry magnitude
    
    For sparse rows, L2 is dominated by the few non-zero entries
    Median better captures the overall signal level
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Row normalization (still use L2 for direction)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Robust magnitude: median of absolute values
        robust_mag = torch.median(torch.abs(x), dim=1, keepdim=True).values
        log_robust_mag = torch.log(robust_mag + 1e-8)
        
        x_augmented = torch.cat([x_normalized, log_robust_mag], dim=1)
        
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class TrimmedMagnitudeMLP(nn.Module):
    """
    Trimmed Magnitude MLP: Uses trimmed mean of absolute values
    
    Ignores the top and bottom 10% of entries when computing magnitude
    More robust than L2 norm but smoother than median
    """
    def __init__(self, input_dim, hidden_dim, num_classes, trim_fraction=0.1):
        super().__init__()
        self.trim_fraction = trim_fraction
        
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Row normalization
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Trimmed magnitude
        abs_x = torch.abs(x)
        n_trim = int(x.shape[1] * self.trim_fraction)
        
        if n_trim > 0:
            # Sort and trim
            sorted_abs, _ = torch.sort(abs_x, dim=1)
            trimmed = sorted_abs[:, n_trim:-n_trim]
            trimmed_mag = trimmed.mean(dim=1, keepdim=True)
        else:
            trimmed_mag = abs_x.mean(dim=1, keepdim=True)
        
        log_trimmed_mag = torch.log(trimmed_mag + 1e-8)
        
        x_augmented = torch.cat([x_normalized, log_trimmed_mag], dim=1)
        
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class AdaptiveMagnitudeMLP(nn.Module):
    """
    Adaptive Magnitude MLP: Learns when to trust magnitude
    
    A small gating network predicts how much to weight the magnitude
    based on the input features themselves.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        # Gating network: predicts magnitude weight (0-1)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Main MLP
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Compute gate value (how much to trust magnitude)
        gate_value = self.gate(x)  # [n, 1], range 0-1
        
        # Row normalization
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Gated magnitude
        log_magnitude = torch.log(x_norm + 1e-8)
        gated_magnitude = gate_value * log_magnitude
        
        x_augmented = torch.cat([x_normalized, gated_magnitude], dim=1)
        
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DensityOnlyMLP(nn.Module):
    """
    Density-Only MLP: Uses row density as a feature instead of magnitude
    
    Since we found sparsity predicts performance, maybe density itself
    is informative for classification.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, sparsity_threshold=1e-6):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Compute density
        near_zero = (torch.abs(x) < self.sparsity_threshold).float()
        density = 1.0 - near_zero.mean(dim=1, keepdim=True)
        
        # Row normalization
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Use density as feature instead of magnitude
        x_augmented = torch.cat([x_normalized, density], dim=1)
        
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============================================================================
# Linear Classifiers (to test if gap is MLP-specific)
# ============================================================================

class LinearClassifier(nn.Module):
    """
    Simple linear classifier (logistic regression)
    No hidden layers, just input -> output
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)


class RowNormLinear(nn.Module):
    """
    Linear classifier with row normalization
    Normalizes input to unit sphere, then linear classification
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.fc(x)


class CosineLinear(nn.Module):
    """
    Cosine similarity linear classifier
    Normalizes both input AND weights, uses temperature scaling
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.fc.weight, p=2, dim=1)
        return self.scale * (x @ w.t())


class LogMagnitudeLinear(nn.Module):
    """
    Linear classifier with log-magnitude feature
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim + 1, num_classes)
        
    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        log_magnitude = torch.log(x_norm + 1e-8)
        x_augmented = torch.cat([x_normalized, log_magnitude], dim=1)
        return self.fc(x_augmented)


class SparsityAwareLinear(nn.Module):
    """
    Linear classifier with sparsity-aware magnitude weighting
    """
    def __init__(self, input_dim, num_classes, sparsity_threshold=1e-6):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.fc = nn.Linear(input_dim + 1, num_classes)
        
    def forward(self, x):
        # Compute density
        near_zero = (torch.abs(x) < self.sparsity_threshold).float()
        density = 1.0 - near_zero.mean(dim=1, keepdim=True)
        
        # Row normalization
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Weighted magnitude
        log_magnitude = torch.log(x_norm + 1e-8)
        weighted_magnitude = density * log_magnitude
        
        x_augmented = torch.cat([x_normalized, weighted_magnitude], dim=1)
        return self.fc(x_augmented)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs, lr, weight_decay, device):
    """Train model and return best test accuracy (based on validation)"""
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
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_pred = model(X_test_t).argmax(dim=1)
                    best_test_acc = (test_pred == y_test_t).float().mean().item()
    
    return best_val_acc, best_test_acc


def run_classifier_comparison(X_diffused, U, labels, num_nodes, num_classes, 
                               num_splits, num_seeds, device):
    """
    Compare all classifier methods on BOTH X_diffused and U
    
    Key question: Does the basis sensitivity gap exist with linear classifiers too?
    If yes → fundamental representation issue
    If no → MLP-specific (optimization/gradient dynamics)
    """
    d_X = X_diffused.shape[1]
    d_U = U.shape[1]
    
    # Define all methods grouped by type
    methods = {}
    
    # === MLP methods on U (our main investigation) ===
    methods['U_StandardMLP'] = lambda: StandardMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_RowNormMLP'] = lambda: RowNormMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_LogMagnitudeMLP'] = lambda: LogMagnitudeMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_SparsityAwareMLP'] = lambda: SparsityAwareMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_RobustMagnitudeMLP'] = lambda: RobustMagnitudeMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_TrimmedMagnitudeMLP'] = lambda: TrimmedMagnitudeMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_AdaptiveMagnitudeMLP'] = lambda: AdaptiveMagnitudeMLP(d_U, HIDDEN_DIM, num_classes)
    methods['U_DensityOnlyMLP'] = lambda: DensityOnlyMLP(d_U, HIDDEN_DIM, num_classes)
    
    # === MLP methods on X_diffused (baseline) ===
    methods['X_StandardMLP'] = lambda: StandardMLP(d_X, HIDDEN_DIM, num_classes)
    methods['X_RowNormMLP'] = lambda: RowNormMLP(d_X, HIDDEN_DIM, num_classes)
    
    # === Linear classifiers on U ===
    methods['U_Linear'] = lambda: LinearClassifier(d_U, num_classes)
    methods['U_RowNormLinear'] = lambda: RowNormLinear(d_U, num_classes)
    methods['U_CosineLinear'] = lambda: CosineLinear(d_U, num_classes)
    methods['U_LogMagLinear'] = lambda: LogMagnitudeLinear(d_U, num_classes)
    methods['U_SparsityAwareLinear'] = lambda: SparsityAwareLinear(d_U, num_classes)
    
    # === Linear classifiers on X_diffused (baseline) ===
    methods['X_Linear'] = lambda: LinearClassifier(d_X, num_classes)
    methods['X_RowNormLinear'] = lambda: RowNormLinear(d_X, num_classes)
    methods['X_CosineLinear'] = lambda: CosineLinear(d_X, num_classes)
    methods['X_LogMagLinear'] = lambda: LogMagnitudeLinear(d_X, num_classes)
    
    results = {name: [] for name in methods.keys()}
    
    for split_i in range(num_splits):
        # Generate random 60/20/20 split
        np.random.seed(split_i)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        if split_i == 0:
            print(f'    Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}')
        
        # Prepare data for U
        U_train = U[train_idx]
        U_val = U[val_idx]
        U_test = U[test_idx]
        
        # Prepare data for X_diffused
        X_train = X_diffused[train_idx]
        X_val = X_diffused[val_idx]
        X_test = X_diffused[test_idx]
        
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        
        for seed in range(num_seeds):
            torch.manual_seed(seed + split_i * 100)
            np.random.seed(seed + split_i * 100)
            
            for method_name, model_fn in methods.items():
                model = model_fn()
                
                # Choose data based on method prefix
                if method_name.startswith('X_'):
                    data_train, data_val, data_test = X_train, X_val, X_test
                else:  # U_
                    data_train, data_val, data_test = U_train, U_val, U_test
                
                _, test_acc = train_and_evaluate(
                    model, data_train, y_train, data_val, y_val, data_test, y_test,
                    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
                )
                results[method_name].append(test_acc)
    
    # Aggregate
    aggregated = {}
    for method_name, accs in results.items():
        aggregated[method_name] = {
            'test_acc_mean': float(np.mean(accs)),
            'test_acc_std': float(np.std(accs)),
            'num_runs': len(accs),
        }
    
    return aggregated


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_dataset(dataset_name, k_diffusion, num_splits, num_seeds, device):
    """Run sparsity-aware classifier comparison for a dataset"""
    print(f'\n{"="*70}')
    print(f'ANALYZING: {dataset_name} (k={k_diffusion})')
    print(f'{"="*70}')
    
    results = {
        'dataset': dataset_name,
        'k_diffusion': k_diffusion,
        'metadata': {},
        'classifiers': None,
    }
    
    try:
        # Load and preprocess (same as Hordan analysis)
        print('\n[1/4] Loading dataset...')
        (edge_index, features, labels, num_nodes, num_classes,
         train_idx, val_idx, test_idx) = load_dataset(dataset_name)
        print(f'  Nodes: {num_nodes}, Features: {features.shape[1]}, Classes: {num_classes}')
        
        print('\n[2/4] Building graph and extracting LCC...')
        adj, L, D = build_graph_matrices(edge_index, num_nodes)
        lcc_mask = get_largest_connected_component_nx(adj)
        
        split_idx = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx} if train_idx is not None else None
        adj, features, labels, split_idx = extract_subgraph(adj, features, labels, lcc_mask, split_idx)
        
        adj_coo = adj.tocoo()
        edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
        adj, L, D = build_graph_matrices(edge_index_lcc, adj.shape[0])
        
        num_nodes = features.shape[0]
        print(f'  LCC nodes: {num_nodes}')
        
        print(f'\n[3/4] Computing diffused features and restricted eigenvectors...')
        A_sgc = compute_sgc_normalized_adjacency(adj)
        features_dense = features.toarray() if sp.issparse(features) else features
        X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diffusion)
        
        U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(X_diffused, L, D, num_components=0)
        print(f'  U shape: {U.shape}, d_eff: {d_eff}')
        
        # Compute sparsity of U
        near_zero = np.abs(U) < 1e-6
        sparsity_U = near_zero.mean()
        ref_nodes_pct = (near_zero.sum(axis=1) == 0).mean() * 100
        print(f'  U sparsity: {sparsity_U*100:.2f}%, Reference nodes: {ref_nodes_pct:.1f}%')
        
        results['metadata'] = {
            'num_nodes': num_nodes,
            'num_classes': num_classes,
            'd_eff': d_eff,
            'sparsity_U': float(sparsity_U),
            'ref_nodes_pct': float(ref_nodes_pct),
        }
        
        print(f'\n[4/4] Running classifier comparison ({num_splits} splits × {num_seeds} seeds)...')
        classifier_results = run_classifier_comparison(
            X_diffused, U, labels, num_nodes, num_classes,
            num_splits, num_seeds, device
        )
        results['classifiers'] = classifier_results
        
        # Print results grouped by type
        print('\n  MLP Results (on U):')
        mlp_methods = [k for k in classifier_results.keys() if 'MLP' in k and k.startswith('U_')]
        sorted_mlp = sorted([(k, classifier_results[k]) for k in mlp_methods], 
                           key=lambda x: x[1]['test_acc_mean'], reverse=True)
        for method_name, res in sorted_mlp:
            print(f'    {method_name:<25} {res["test_acc_mean"]*100:6.2f}% ± {res["test_acc_std"]*100:.2f}%')
        
        print('\n  MLP Results (on X_diffused):')
        mlp_X_methods = [k for k in classifier_results.keys() if 'MLP' in k and k.startswith('X_')]
        sorted_mlp_X = sorted([(k, classifier_results[k]) for k in mlp_X_methods], 
                             key=lambda x: x[1]['test_acc_mean'], reverse=True)
        for method_name, res in sorted_mlp_X:
            print(f'    {method_name:<25} {res["test_acc_mean"]*100:6.2f}% ± {res["test_acc_std"]*100:.2f}%')
        
        print('\n  Linear Classifier Results:')
        linear_methods = [k for k in classifier_results.keys() if 'Linear' in k]
        sorted_linear = sorted([(k, classifier_results[k]) for k in linear_methods], 
                              key=lambda x: x[1]['test_acc_mean'], reverse=True)
        for method_name, res in sorted_linear:
            print(f'    {method_name:<25} {res["test_acc_mean"]*100:6.2f}% ± {res["test_acc_std"]*100:.2f}%')
        
        # Compute gaps for quick reference
        if 'X_StandardMLP' in classifier_results and 'U_StandardMLP' in classifier_results:
            mlp_gap = (classifier_results['U_StandardMLP']['test_acc_mean'] - 
                      classifier_results['X_StandardMLP']['test_acc_mean']) * 100
            print(f'\n  MLP Gap (U - X): {mlp_gap:+.2f}pp')
        
        if 'X_Linear' in classifier_results and 'U_Linear' in classifier_results:
            linear_gap = (classifier_results['U_Linear']['test_acc_mean'] - 
                         classifier_results['X_Linear']['test_acc_mean']) * 100
            print(f'  Linear Gap (U - X): {linear_gap:+.2f}pp')
        
        return results
        
    except Exception as e:
        print(f'  ERROR: {str(e)}')
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    
    all_results = {}
    
    for dataset_name in DATASETS:
        for k_diff in K_DIFFUSION_VALUES:
            key = f'{dataset_name}_k{k_diff}'
            results = analyze_dataset(dataset_name, k_diff, NUM_SPLITS, NUM_SEEDS, device)
            
            if results is not None:
                all_results[key] = results
    
    # ========================================================================
    # Summary Table: MLP Comparison
    # ========================================================================
    
    print('\n' + '='*100)
    print('SUMMARY: MLP CLASSIFIER COMPARISON (on U)')
    print('='*100)
    
    # Get MLP method names (on U)
    mlp_method_names = ['U_StandardMLP', 'U_RowNormMLP', 'U_LogMagnitudeMLP', 
                        'U_SparsityAwareMLP', 'U_RobustMagnitudeMLP', 'U_TrimmedMagnitudeMLP',
                        'U_AdaptiveMagnitudeMLP', 'U_DensityOnlyMLP']
    
    # Header
    header = f'{"Dataset":<22} {"Sparse%":<8}'
    for name in mlp_method_names:
        short_name = name.replace('U_', '').replace('MLP', '').replace('Magnitude', 'Mag')[:8]
        header += f'{short_name:<9}'
    header += f'{"Best":<12}'
    print(header)
    print('-'*100)
    
    # Results rows
    for key, results in all_results.items():
        if results['classifiers'] is None:
            continue
            
        sparsity = results['metadata']['sparsity_U'] * 100
        row = f'{key:<22} {sparsity:<8.2f}'
        
        best_acc = 0
        best_method = ''
        
        for method_name in mlp_method_names:
            if method_name in results['classifiers']:
                acc = results['classifiers'][method_name]['test_acc_mean'] * 100
                row += f'{acc:<9.2f}'
                if acc > best_acc:
                    best_acc = acc
                    best_method = method_name.replace('U_', '')
            else:
                row += f'{"N/A":<9}'
        
        row += f'{best_method:<12}'
        print(row)
    
    # ========================================================================
    # Summary Table: LINEAR CLASSIFIER COMPARISON (Key Question!)
    # ========================================================================
    
    print('\n' + '='*100)
    print('SUMMARY: LINEAR CLASSIFIER COMPARISON (Does gap exist for linear too?)')
    print('='*100)
    
    linear_method_names_X = ['X_Linear', 'X_RowNormLinear', 'X_CosineLinear', 'X_LogMagLinear']
    linear_method_names_U = ['U_Linear', 'U_RowNormLinear', 'U_CosineLinear', 'U_LogMagLinear', 'U_SparsityAwareLinear']
    
    # Header
    header = f'{"Dataset":<22} '
    header += f'{"X_Lin":<8} {"X_RN":<8} {"X_Cos":<8} {"X_LM":<8} |'
    header += f'{"U_Lin":<8} {"U_RN":<8} {"U_Cos":<8} {"U_LM":<8} {"U_SA":<8} |'
    header += f'{"Gap(Lin)":<10}'
    print(header)
    print('-'*110)
    
    linear_gaps = []
    mlp_gaps = []
    
    for key, results in all_results.items():
        if results['classifiers'] is None:
            continue
        
        row = f'{key:<22} '
        
        # X methods
        for method_name in linear_method_names_X:
            if method_name in results['classifiers']:
                acc = results['classifiers'][method_name]['test_acc_mean'] * 100
                row += f'{acc:<8.2f}'
            else:
                row += f'{"N/A":<8}'
        
        row += '|'
        
        # U methods
        for method_name in linear_method_names_U:
            if method_name in results['classifiers']:
                acc = results['classifiers'][method_name]['test_acc_mean'] * 100
                row += f'{acc:<8.2f}'
            else:
                row += f'{"N/A":<8}'
        
        row += '|'
        
        # Compute linear gap (U_Linear - X_Linear)
        if 'X_Linear' in results['classifiers'] and 'U_Linear' in results['classifiers']:
            lin_gap = (results['classifiers']['U_Linear']['test_acc_mean'] - 
                      results['classifiers']['X_Linear']['test_acc_mean']) * 100
            row += f'{lin_gap:+.2f}pp'
            linear_gaps.append({'key': key, 'gap': lin_gap, 'sparsity': results['metadata']['sparsity_U']})
        
        # Also compute MLP gap for comparison
        if 'X_StandardMLP' in results['classifiers'] and 'U_StandardMLP' in results['classifiers']:
            mlp_gap = (results['classifiers']['U_StandardMLP']['test_acc_mean'] - 
                      results['classifiers']['X_StandardMLP']['test_acc_mean']) * 100
            mlp_gaps.append({'key': key, 'gap': mlp_gap, 'sparsity': results['metadata']['sparsity_U']})
        
        print(row)
    
    # ========================================================================
    # KEY ANALYSIS: Linear vs MLP gap comparison
    # ========================================================================
    
    print('\n' + '='*100)
    print('KEY ANALYSIS: Does the gap exist with linear classifiers?')
    print('='*100)
    
    print(f'\n{"Dataset":<25} {"MLP Gap":<15} {"Linear Gap":<15} {"Conclusion":<30}')
    print('-'*85)
    
    for i, (mlp_data, lin_data) in enumerate(zip(mlp_gaps, linear_gaps)):
        mlp_gap = mlp_data['gap']
        lin_gap = lin_data['gap']
        key = mlp_data['key']
        
        # Interpretation
        if abs(mlp_gap) < 2 and abs(lin_gap) < 2:
            conclusion = "No gap (both small)"
        elif mlp_gap < -5 and lin_gap < -5:
            conclusion = "Gap exists for BOTH"
        elif mlp_gap < -5 and abs(lin_gap) < 2:
            conclusion = "MLP-SPECIFIC gap!"
        elif abs(mlp_gap) < 2 and lin_gap < -5:
            conclusion = "Linear-specific gap (unusual)"
        else:
            conclusion = "Mixed pattern"
        
        print(f'{key:<25} {mlp_gap:+.2f}pp{"":<8} {lin_gap:+.2f}pp{"":<8} {conclusion:<30}')
    
    # Summary statistics
    if len(mlp_gaps) > 0 and len(linear_gaps) > 0:
        mean_mlp_gap = np.mean([d['gap'] for d in mlp_gaps])
        mean_lin_gap = np.mean([d['gap'] for d in linear_gaps])
        
        mlp_negative = sum(1 for d in mlp_gaps if d['gap'] < -2)
        lin_negative = sum(1 for d in linear_gaps if d['gap'] < -2)
        
        print(f'\nSummary:')
        print(f'  Mean MLP Gap:    {mean_mlp_gap:+.2f}pp')
        print(f'  Mean Linear Gap: {mean_lin_gap:+.2f}pp')
        print(f'  MLP shows gap (< -2pp):    {mlp_negative}/{len(mlp_gaps)} datasets')
        print(f'  Linear shows gap (< -2pp): {lin_negative}/{len(linear_gaps)} datasets')
        
        if mean_lin_gap > mean_mlp_gap + 5:
            print(f'\n  CONCLUSION: Gap is largely MLP-SPECIFIC!')
            print(f'              Linear classifiers do not show the same gap.')
            print(f'              This suggests the issue is MLP optimization, not representation.')
        elif abs(mean_lin_gap - mean_mlp_gap) < 3:
            print(f'\n  CONCLUSION: Gap exists for BOTH linear and MLP.')
            print(f'              This suggests a fundamental representation issue.')
        else:
            print(f'\n  CONCLUSION: Mixed results - needs further analysis.')
    
    # ========================================================================
    # Analysis: New MLP methods vs baselines
    # ========================================================================
    
    print('\n' + '='*100)
    print('ANALYSIS: NEW MLP METHODS VS BASELINES (on U)')
    print('='*100)
    
    improvements = {name: [] for name in ['U_SparsityAwareMLP', 'U_RobustMagnitudeMLP', 
                                          'U_TrimmedMagnitudeMLP', 'U_AdaptiveMagnitudeMLP', 
                                          'U_DensityOnlyMLP']}
    
    for key, results in all_results.items():
        if results['classifiers'] is None:
            continue
        
        rn_acc = results['classifiers']['U_RowNormMLP']['test_acc_mean']
        lm_acc = results['classifiers']['U_LogMagnitudeMLP']['test_acc_mean']
        best_baseline = max(rn_acc, lm_acc)
        
        for new_method in improvements.keys():
            if new_method in results['classifiers']:
                new_acc = results['classifiers'][new_method]['test_acc_mean']
                vs_rn = (new_acc - rn_acc) * 100
                vs_lm = (new_acc - lm_acc) * 100
                vs_best = (new_acc - best_baseline) * 100
                improvements[new_method].append({
                    'key': key,
                    'sparsity': results['metadata']['sparsity_U'],
                    'vs_rownorm': vs_rn,
                    'vs_logmag': vs_lm,
                    'vs_best_baseline': vs_best,
                })
    
    print(f'\n{"Method":<25} {"Mean vs RN":<15} {"Mean vs LogMag":<15} {"Win vs Best":<15}')
    print('-'*70)
    
    for method_name, data in improvements.items():
        if len(data) > 0:
            mean_vs_rn = np.mean([d['vs_rownorm'] for d in data])
            mean_vs_lm = np.mean([d['vs_logmag'] for d in data])
            wins = sum(1 for d in data if d['vs_best_baseline'] > 0)
            total = len(data)
            print(f'{method_name:<25} {mean_vs_rn:+.2f}pp{"":<8} {mean_vs_lm:+.2f}pp{"":<8} {wins}/{total}')
    
    # ========================================================================
    # Correlation with sparsity
    # ========================================================================
    
    print('\n' + '='*100)
    print('CORRELATION: IMPROVEMENT VS SPARSITY')
    print('='*100)
    
    for method_name, data in improvements.items():
        if len(data) >= 5:
            sparsities = [d['sparsity'] for d in data]
            vs_rn = [d['vs_rownorm'] for d in data]
            
            from scipy import stats
            corr, p_val = stats.pearsonr(sparsities, vs_rn)
            print(f'{method_name:<25} r = {corr:+.4f} (p = {p_val:.4f})')
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(all_results)
    
    with open(f'{OUTPUT_DIR}/sparsity_aware_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f'\n✓ Results saved: {OUTPUT_DIR}/sparsity_aware_results.json')
    
    # ========================================================================
    # Generate Plot
    # ========================================================================
    
    print('\n' + '='*100)
    print('GENERATING VISUALIZATIONS')
    print('='*100)
    
    # Plot 1: MLP Method comparison bar chart (on U)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    datasets = list(all_results.keys())
    x = np.arange(len(datasets))
    mlp_plot_methods = ['U_StandardMLP', 'U_RowNormMLP', 'U_LogMagnitudeMLP', 
                        'U_SparsityAwareMLP', 'U_RobustMagnitudeMLP', 'U_AdaptiveMagnitudeMLP']
    n_methods = len(mlp_plot_methods)
    width = 0.8 / n_methods
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    for i, method_name in enumerate(mlp_plot_methods):
        accs = []
        for key in datasets:
            if all_results[key]['classifiers'] and method_name in all_results[key]['classifiers']:
                accs.append(all_results[key]['classifiers'][method_name]['test_acc_mean'] * 100)
            else:
                accs.append(0)
        
        short_name = method_name.replace('U_', '').replace('MLP', '').replace('Magnitude', 'Mag')
        ax.bar(x + i*width - 0.4 + width/2, accs, width, label=short_name, color=colors[i])
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('MLP Classifier Comparison (on Restricted Eigenvectors U)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/mlp_comparison.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {OUTPUT_DIR}/plots/mlp_comparison.png')
    plt.close()
    
    # Plot 2: Linear vs MLP Gap Comparison (KEY PLOT!)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar chart comparing MLP and Linear gaps
    ax1 = axes[0]
    if len(mlp_gaps) > 0 and len(linear_gaps) > 0:
        dataset_keys = [d['key'] for d in mlp_gaps]
        mlp_gap_vals = [d['gap'] for d in mlp_gaps]
        lin_gap_vals = [d['gap'] for d in linear_gaps]
        
        x = np.arange(len(dataset_keys))
        width = 0.35
        
        ax1.bar(x - width/2, mlp_gap_vals, width, label='MLP Gap', color='#1f77b4')
        ax1.bar(x + width/2, lin_gap_vals, width, label='Linear Gap', color='#ff7f0e')
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Dataset', fontsize=12)
        ax1.set_ylabel('Gap (U - X) in pp', fontsize=12)
        ax1.set_title('Basis Sensitivity Gap: MLP vs Linear', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([k.replace('_k', '\nk=') for k in dataset_keys], fontsize=8)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
    
    # Right: Scatter plot MLP gap vs Linear gap
    ax2 = axes[1]
    if len(mlp_gaps) > 0 and len(linear_gaps) > 0:
        mlp_gap_vals = [d['gap'] for d in mlp_gaps]
        lin_gap_vals = [d['gap'] for d in linear_gaps]
        
        ax2.scatter(lin_gap_vals, mlp_gap_vals, s=100, alpha=0.7)
        
        # Diagonal line (if MLP gap == Linear gap)
        min_val = min(min(mlp_gap_vals), min(lin_gap_vals)) - 5
        max_val = max(max(mlp_gap_vals), max(lin_gap_vals)) + 5
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='MLP = Linear')
        
        # Add annotations
        for mlp_d, lin_d in zip(mlp_gaps, linear_gaps):
            ax2.annotate(mlp_d['key'].split('_')[0], 
                        (lin_d['gap'], mlp_d['gap']),
                        fontsize=8, alpha=0.7)
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Linear Gap (pp)', fontsize=12)
        ax2.set_ylabel('MLP Gap (pp)', fontsize=12)
        ax2.set_title('MLP Gap vs Linear Gap', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/mlp_vs_linear_gap.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {OUTPUT_DIR}/plots/mlp_vs_linear_gap.png')
    plt.close()
    
    # Plot 3: Improvement vs Sparsity scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    new_methods = ['U_SparsityAwareMLP', 'U_RobustMagnitudeMLP', 'U_TrimmedMagnitudeMLP',
                   'U_AdaptiveMagnitudeMLP', 'U_DensityOnlyMLP']
    
    for i, method_name in enumerate(new_methods):
        ax = axes[i]
        data = improvements[method_name]
        
        if len(data) > 0:
            sparsities = [d['sparsity'] * 100 for d in data]
            vs_rn = [d['vs_rownorm'] for d in data]
            
            ax.scatter(sparsities, vs_rn, s=100, alpha=0.7)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            for d in data:
                ax.annotate(d['key'].split('_')[0], 
                           (d['sparsity'] * 100, d['vs_rownorm']),
                           fontsize=7, alpha=0.7)
            
            # Regression line
            if len(sparsities) >= 3:
                z = np.polyfit(sparsities, vs_rn, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(sparsities), max(sparsities), 100)
                ax.plot(x_line, p(x_line), 'b--', alpha=0.5)
            
            ax.set_xlabel('Sparsity of U (%)', fontsize=10)
            ax.set_ylabel('Improvement vs RowNorm (pp)', fontsize=10)
            ax.set_title(method_name.replace('U_', '').replace('MLP', ''), fontsize=11)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/improvement_vs_sparsity.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {OUTPUT_DIR}/plots/improvement_vs_sparsity.png')
    plt.close()
    
    print('\n' + '='*100)
    print('INVESTIGATION COMPLETE')
    print('='*100)