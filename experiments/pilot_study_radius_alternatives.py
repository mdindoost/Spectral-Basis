"""
===================================================================================
PILOT STUDY: RADIUS-WEIGHTED TRAINING ALTERNATIVES
===================================================================================

Tests alternative radius definitions and weighting strategies on 3 representative
datasets to determine if full experiments are warranted.

Pilot Datasets:
  1. Coauthor-CS (fixed): Where radius worked (+0.86pp)
  2. Cora (fixed): Where radius failed (-0.11pp)  
  3. ogbn-arxiv (fixed): Marginal effect (+0.20pp)

Methods Tested:
  A. Radius Definitions (4):
     1. Current: ||V|| (raw eigenvector magnitude)
     2. Eigenvalue-weighted: ||V * λ^α|| (α=1.0)
     3. Degree-based: √D_ii
     4. Diffused: ||X_diffused||
  
  B. Weight Ranges (3):
     1. Current: [0.3, 1.0]
     2. Aggressive: [0.1, 1.0]
     3. Extreme: [0.05, 1.0]
  
  C. Learned Weights (1):
     - Small MLP learns weights from features

Total per dataset: 3 methods × (4 radii × 3 ranges + 1 learned) = 39 experiments

Decision Criteria:
  - If ANY alternative shows >1.0pp improvement on Cora/ogbn-arxiv → Full study
  - If all alternatives show <0.5pp improvement → Abandon, focus on publication
  - If one specific approach works → Full study with that approach only

Author: Mohammad
Date: December 2025
===================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np

# CRITICAL: Fix PyTorch 2.6 compatibility BEFORE importing torch or utils
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Now safe to import everything else
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.linalg as la
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway

# FIXED: Correct import path (not experiments.utils)
from experiments.utils import (
    load_dataset, build_graph_matrices,
    SGC, StandardMLP, RowNormMLP, LogMagnitudeMLP, NestedSpheresMLP,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute
)

# ============================================================================
# Configuration
# ============================================================================

# Pilot datasets (representative sample)
PILOT_DATASETS = ['coauthor-cs', 'cora', 'ogbn-arxiv']

K_DIFFUSION = 10
SPLIT_TYPE = 'fixed'
COMPONENT_TYPE = 'lcc'
NUM_SEEDS = 5
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test different weight ranges
WEIGHT_RANGES = [
    (0.3, 1.0),   # Current (Investigation 5)
    (0.1, 1.0),   # Aggressive
    (0.05, 1.0),  # Extreme
]

print(f'Device: {DEVICE}')

# ============================================================================
# Learned Weight Network (Model B) - NEW CLASSES
# ============================================================================

class LearnedWeightNetwork(nn.Module):
    """
    Small MLP that learns per-sample weights from features.
    From Yiannis's Model B.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive weights
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class RowNormMLPWithLearnedWeights(nn.Module):
    """
    RowNorm MLP where training weights are learned by a small network.
    Implements Yiannis's Model B.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Weight predictor (small capacity)
        self.weight_net = LearnedWeightNetwork(input_dim, hidden_dim=32)
    
    def forward(self, x):
        # Row normalization
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-10)
        
        # Predict weights
        weights = self.weight_net(x_normalized)
        
        # Classifier
        logits = self.classifier(x_normalized)
        
        return logits, weights


# ============================================================================
# Helper Functions
# ============================================================================

def compute_restricted_eigenvectors(X, L, D, num_components=0):
    """
    Compute X-restricted eigenvectors via Rayleigh-Ritz procedure.
    
    Returns:
        U: Eigenvector matrix (n x d_effective)
        eigenvalues: Corresponding eigenvalues
        d_effective: Effective dimension after rank check
        ortho_error: D-orthonormality error (should be < 1e-6)
    """
    num_nodes, dimension = X.shape
    
    # QR decomposition for numerical stability
    Q, R = np.linalg.qr(X)
    rank_X = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    if rank_X < dimension:
        Q = Q[:, :rank_X]
        dimension = rank_X
    
    d_effective = dimension
    
    # Project Laplacian and degree matrix onto column space of X
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize for numerical stability
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize D_r
    eps_base = 1e-10
    eps = eps_base * np.trace(D_r) / d_effective
    D_r = D_r + eps * np.eye(d_effective)
    
    # Solve generalized eigenvalue problem
    eigenvalues, V = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Skip first num_components (usually 0 for LCC)
    if num_components > 0:
        eigenvalues = eigenvalues[num_components:]
        V = V[:, num_components:]
    
    # Map back to original space
    U = Q @ V
    
    # Verify D-orthonormality: U^T D U should be identity
    G = U.T @ (D @ U)
    ortho_error = np.max(np.abs(G - np.eye(len(eigenvalues))))
    
    return U, eigenvalues, len(eigenvalues), ortho_error


def compute_radius(X):
    """Compute row-wise L2 norms (radius) - as Yiannis specified"""
    radius = np.linalg.norm(X, axis=1)
    return radius


# ============================================================================
# Alternative Radius Computation Functions
# ============================================================================

def compute_radius_raw(V):
    """Current baseline: Raw eigenvector magnitude ||V||"""
    return compute_radius(V)


def compute_radius_eigenvalue_weighted(V, eigenvalues, alpha=1.0):
    """Eigenvalue-weighted: ||V * λ^α|| (what Nested Spheres uses internally)"""
    eigenvalues_safe = np.abs(eigenvalues) + 1e-8
    weights = eigenvalues_safe ** alpha
    V_weighted = V * weights
    return compute_radius(V_weighted)


def compute_radius_degree(D):
    """Degree-based: √D_ii (structural centrality)"""
    # Extract diagonal of sparse degree matrix
    degrees = np.array(D.diagonal()).flatten()
    return np.sqrt(degrees + 1e-8)


def compute_radius_diffused(X_diffused):
    """Diffused: ||X_diffused|| (neighborhood-aware magnitude)"""
    return compute_radius(X_diffused)


def compute_radius_weights(radius, min_weight=0.3, max_weight=1.0):
    """
    Compute normalized radius weights for loss weighting.
    
    Weight range: [min_weight, max_weight] = [0.3, 1.0] as Yiannis specified.
    Low-radius nodes still contribute 30%.
    """
    r_min = radius.min()
    r_max = radius.max()
    
    if r_max - r_min < 1e-8:
        weights = np.ones_like(radius)
    else:
        # Scale radius to [0, 1]
        radius_scaled = (radius - r_min) / (r_max - r_min + 1e-8)
        # Map to [min_weight, max_weight]
        weights = min_weight + (max_weight - min_weight) * radius_scaled
    
    # Normalize to sum to 1
    weights_normalized = weights / weights.sum()
    
    return weights_normalized


# ============================================================================
# Radius Discriminability Analysis (NEW)
# ============================================================================

def analyze_radius_discriminability(radius, labels, num_classes):
    """
    Compute how well radius separates classes (like Fisher score for radius).
    Returns F-statistic and p-value from one-way ANOVA.
    """
    try:
        radius_by_class = [radius[labels == c] for c in range(num_classes)]
        f_stat, p_value = f_oneway(*radius_by_class)
        return f_stat, p_value
    except:
        return np.nan, np.nan


# ============================================================================
# Training Function
# ============================================================================

def train_and_test_with_weights(model, X_train, y_train, X_val, y_val, X_test, y_test,
                                 radius_weights, epochs, learning_rate, weight_decay, device,
                                 learned_weights=False):
    """
    Train with optional radius weighting or learned weighting.
    """
    model = model.to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    if radius_weights is not None and not learned_weights:
        radius_weights_t = torch.FloatTensor(radius_weights).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if learned_weights:
            # Model predicts both logits and weights
            logits, weights = model(X_train_t)
            # Normalize weights per batch
            weights_normalized = weights / (weights.sum() + 1e-8)
            # Weighted loss
            ce_loss = F.cross_entropy(logits, y_train_t, reduction='none')
            loss = (weights_normalized * ce_loss).sum()
        elif radius_weights is not None:
            # Fixed radius weights
            logits = model(X_train_t)
            ce_loss = F.cross_entropy(logits, y_train_t, reduction='none')
            loss = (radius_weights_t * ce_loss).sum()
        else:
            # Standard uniform weighting
            logits = model(X_train_t)
            loss = F.cross_entropy(logits, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if learned_weights:
                val_logits, _ = model(X_val_t)
                test_logits, _ = model(X_test_t)
            else:
                val_logits = model(X_val_t)
                test_logits = model(X_test_t)
            
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
            
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == y_test_t).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    
    return best_test_acc


# ============================================================================
# Pilot Experiment Function
# ============================================================================

def run_pilot_experiment(dataset_name, method_name, radius_type, weight_range,
                         use_learned_weights=False):
    """
    Run single pilot experiment.
    
    Args:
        dataset_name: Dataset to test
        method_name: 'rownorm', 'logmag', or 'nested'
        radius_type: 'raw', 'eigenvalue', 'degree', or 'diffused'
        weight_range: (min_weight, max_weight) tuple
        use_learned_weights: If True, use Model B instead of radius
    """
    print(f'\n{"="*80}')
    print(f'Dataset: {dataset_name}')
    print(f'Method: {method_name}')
    if use_learned_weights:
        print(f'Weighting: Learned (Model B)')
    else:
        print(f'Radius: {radius_type}, Range: {weight_range}')
    print(f'{"="*80}')
    
    # Load dataset - load_dataset() returns fixed splits only
    (edge_index, X_raw, labels, num_nodes, num_classes,
     train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(
        dataset_name, 
        root='./dataset'
    )
    
    print(f'Loaded: {num_nodes:,} nodes, {X_raw.shape[1]} features, {num_classes} classes')
    
    # Build graph matrices
    adj, D, L = build_graph_matrices(edge_index, num_nodes)
    
    # Extract LCC if requested
    if COMPONENT_TYPE == 'lcc':
        print('Extracting largest connected component...')
        lcc_mask = get_largest_connected_component_nx(adj)
        
        split_idx_original = {
            'train_idx': train_idx_fixed, 
            'val_idx': val_idx_fixed, 
            'test_idx': test_idx_fixed
        }
        adj, X_raw, labels, split_idx = extract_subgraph(
            adj, X_raw, labels, lcc_mask, split_idx_original
        )
        
        # Rebuild graph matrices for LCC
        adj_coo = adj.tocoo()
        edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
        adj, D, L = build_graph_matrices(edge_index_lcc, adj.shape[0])
        
        # Extract splits from LCC
        train_idx = split_idx['train_idx']
        val_idx = split_idx['val_idx']
        test_idx = split_idx['test_idx']
        num_components = 0
        
        print(f'LCC: {adj.shape[0]:,} nodes')
    else:
        import networkx as nx
        num_components = len(list(nx.connected_components(nx.from_scipy_sparse_array(adj)))) - 1
        train_idx = train_idx_fixed
        val_idx = val_idx_fixed
        test_idx = test_idx_fixed
    
    # Update variables
    num_nodes = X_raw.shape[0]
    num_classes = len(np.unique(labels))
    
    # Handle splits based on SPLIT_TYPE
    if SPLIT_TYPE == 'random':
        print('Creating random 60/20/20 split...')
        np.random.seed(0)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    else:
        print('Using fixed benchmark splits...')
    
    print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
    
    # Diffuse features
    print('Computing SGC diffusion...')
    A_sgc = compute_sgc_normalized_adjacency(adj)
    features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw
    X_diffused = sgc_precompute(features_dense.copy(), A_sgc, K_DIFFUSION)
    
    # Compute restricted eigenvectors
    print('Computing restricted eigenvectors...')
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diffused, L, D, num_components
    )
    
    print(f'Eigenvectors: d_eff={d_eff}, ortho_err={ortho_err:.2e}')
    
    # IMPROVED: Warn but don't skip (unless catastrophic)
    if ortho_err > 1e-4:
        warnings.warn(f'High D-orthonormality error: {ortho_err:.2e}')
        if ortho_err > 1e-2:
            print('   ⚠️  Orthonormality error too high, skipping...')
            return {
                'dataset': dataset_name,
                'method': method_name,
                'radius_type': radius_type if not use_learned_weights else 'learned',
                'weight_range': weight_range if not use_learned_weights else None,
                'test_acc_mean': np.nan,
                'test_acc_std': np.nan,
                'results': [],
                'error': f'High D-orthonormality error: {ortho_err:.2e}'
            }
    
    # Compute radius based on type
    if not use_learned_weights:
        if radius_type == 'raw':
            radius = compute_radius_raw(U)
        elif radius_type == 'eigenvalue':
            radius = compute_radius_eigenvalue_weighted(U, eigenvalues, alpha=1.0)
        elif radius_type == 'degree':
            radius = compute_radius_degree(D)
        elif radius_type == 'diffused':
            radius = compute_radius_diffused(X_diffused)
        else:
            raise ValueError(f'Unknown radius type: {radius_type}')
        
        # NEW: Analyze radius discriminability
        f_stat, p_value = analyze_radius_discriminability(radius, labels, num_classes)
        
        print(f'Radius: μ={radius.mean():.4f}, σ={radius.std():.4f}, range=[{radius.min():.4f}, {radius.max():.4f}]')
        print(f'Radius discriminability: F={f_stat:.4f}, p={p_value:.4e}')
    else:
        radius = None
        f_stat, p_value = np.nan, np.nan
    
    # Results storage
    results = []
    
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        if use_learned_weights:
            model = RowNormMLPWithLearnedWeights(d_eff, HIDDEN_DIM, num_classes)
            radius_weights = None
        else:
            if method_name == 'rownorm':
                model = RowNormMLP(d_eff, HIDDEN_DIM, num_classes)
            elif method_name == 'logmag':
                model = LogMagnitudeMLP(d_eff, HIDDEN_DIM, num_classes)
            elif method_name == 'nested':
                eigenvalues_torch = torch.FloatTensor(eigenvalues)
                model = NestedSpheresMLP(d_eff, HIDDEN_DIM, num_classes,
                                        eigenvalues_torch, alpha=1.0, beta=1.0)
            else:
                raise ValueError(f'Unknown method: {method_name}')
            
            # Compute weights
            min_weight, max_weight = weight_range
            radius_weights = compute_radius_weights(
                radius[train_idx], min_weight, max_weight
            )
        
        # Train
        test_acc = train_and_test_with_weights(
            model, U[train_idx], labels[train_idx],
            U[val_idx], labels[val_idx],
            U[test_idx], labels[test_idx],
            radius_weights, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE,
            learned_weights=use_learned_weights
        )
        
        results.append(test_acc)
        print(f'  Seed {seed}: {test_acc*100:.2f}%')
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    
    print(f'\nResult: {mean_acc:.2f}% ± {std_acc:.2f}%')
    
    return {
        'dataset': dataset_name,
        'method': method_name,
        'radius_type': radius_type if not use_learned_weights else 'learned',
        'weight_range': weight_range if not use_learned_weights else None,
        'test_acc_mean': mean_acc,
        'test_acc_std': std_acc,
        'results': results,
        'radius_f_stat': float(f_stat) if not np.isnan(f_stat) else None,
        'radius_p_value': float(p_value) if not np.isnan(p_value) else None
    }


# ============================================================================
# Main Pilot Study
# ============================================================================

if __name__ == '__main__':
    print('='*80)
    print('PILOT STUDY: RADIUS-WEIGHTED TRAINING ALTERNATIVES')
    print('='*80)
    print(f'\nDatasets: {PILOT_DATASETS}')
    print(f'Methods: RowNorm (SIMPLE), Log-Magnitude (COMPLEX), Nested Spheres (COMPLEX)')
    print(f'Radius types: raw, eigenvalue-weighted, degree-based, diffused')
    print(f'Weight ranges: {WEIGHT_RANGES}')
    print(f'Learned weights: Model B')
    print()
    
    # Storage
    all_results = []
    output_dir = 'results/pilot_study_radius'
    os.makedirs(output_dir, exist_ok=True)
    
    # IMPROVED: Progress tracking
    baseline_count = len(PILOT_DATASETS) * 3  # 3 methods
    alternative_radius_count = len(PILOT_DATASETS) * 3 * 3  # 3 datasets × 3 methods × 3 radii
    aggressive_count = len(PILOT_DATASETS) * 2  # 2 weight ranges
    learned_count = len(PILOT_DATASETS)  # 1 per dataset
    total_experiments = baseline_count + alternative_radius_count + aggressive_count + learned_count
    
    experiment_counter = 0
    
    # Baseline: Run Investigation 5 configuration first
    print('\n' + '='*80)
    print('BASELINE: Investigation 5 Configuration')
    print('='*80)
    
    for dataset in PILOT_DATASETS:
        for method in ['rownorm', 'logmag', 'nested']:
            experiment_counter += 1
            print(f'\n[{experiment_counter}/{total_experiments}] BASELINE EXPERIMENT')
            
            result = run_pilot_experiment(
                dataset, method, 'raw', (0.3, 1.0), use_learned_weights=False
            )
            all_results.append(result)
            
            # IMPROVED: Save intermediate results
            with open(f'{output_dir}/intermediate_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Test alternative radius definitions
    print('\n' + '='*80)
    print('ALTERNATIVE RADIUS DEFINITIONS')
    print('='*80)
    
    for dataset in PILOT_DATASETS:
        for method in ['rownorm', 'logmag', 'nested']:
            for radius_type in ['eigenvalue', 'degree', 'diffused']:
                experiment_counter += 1
                print(f'\n[{experiment_counter}/{total_experiments}] ALTERNATIVE RADIUS')
                
                result = run_pilot_experiment(
                    dataset, method, radius_type, (0.3, 1.0), use_learned_weights=False
                )
                all_results.append(result)
                
                with open(f'{output_dir}/intermediate_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2)
    
    # Test aggressive weight ranges (using best radius from above)
    print('\n' + '='*80)
    print('AGGRESSIVE WEIGHT RANGES')
    print('='*80)
    
    for dataset in PILOT_DATASETS:
        for weight_range in [(0.1, 1.0), (0.05, 1.0)]:
            experiment_counter += 1
            print(f'\n[{experiment_counter}/{total_experiments}] AGGRESSIVE WEIGHTING')
            
            result = run_pilot_experiment(
                dataset, 'rownorm', 'raw', weight_range, use_learned_weights=False
            )
            all_results.append(result)
            
            with open(f'{output_dir}/intermediate_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Test learned weights (Model B)
    print('\n' + '='*80)
    print('LEARNED WEIGHTS (MODEL B)')
    print('='*80)
    
    for dataset in PILOT_DATASETS:
        experiment_counter += 1
        print(f'\n[{experiment_counter}/{total_experiments}] LEARNED WEIGHTS')
        
        result = run_pilot_experiment(
            dataset, 'rownorm', None, None, use_learned_weights=True
        )
        all_results.append(result)
        
        with open(f'{output_dir}/intermediate_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Save final results
    final_results_file = f'{output_dir}/pilot_results_final.json'
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\n' + '='*80)
    print('PILOT STUDY COMPLETE')
    print('='*80)
    print(f'\nTotal experiments run: {len(all_results)}')
    print(f'Results saved to: {final_results_file}')
    print('\nNext steps:')
    print('  1. Run: python analyze_pilot_study.py')
    print('  2. Review decision criteria:')
    print('     - >1.0pp improvement on failures → Full study')
    print('     - <0.5pp improvement → Abandon, focus on publication')
    print('     - One approach works → Full study with that approach only')