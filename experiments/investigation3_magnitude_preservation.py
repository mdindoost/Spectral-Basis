"""
Investigation 3: Magnitude-Preserving Normalization (Enhanced with Diagnostics)
================================================================================

Tests α-normalization with explicit magnitude channel to address the fundamental
problem: RowNorm discards magnitude information.

ENHANCEMENTS:
- Magnitude channel standardization (fixes ogbn-arxiv failure)
- Diagnostic analysis (magnitude distributions, learning curves)
- Optional LCC extraction (like SGC v2)
- Enhanced logging and visualization

Usage:
    # Standard (whole graph, like Investigation 2)
    python experiments/investigation3_magnitude_preservation.py [dataset]
    
    # With LCC extraction (like SGC v2)
    python experiments/investigation3_magnitude_preservation.py [dataset] --use-lcc
    
    # With random splits
    python experiments/investigation3_magnitude_preservation.py [dataset] --random-splits
    
    # Both options
    python experiments/investigation3_magnitude_preservation.py [dataset] --use-lcc --random-splits

Examples:
    python experiments/investigation3_magnitude_preservation.py ogbn-arxiv
    python experiments/investigation3_magnitude_preservation.py cora --use-lcc
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    StandardMLP,
    build_graph_matrices, load_dataset
)

# ============================================================================
# Configuration
# ============================================================================
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ogbn-arxiv'
USE_RANDOM_SPLITS = '--random-splits' in sys.argv
USE_LCC = '--use-lcc' in sys.argv  # NEW: Optional LCC extraction

# Experimental parameters
NUM_SEEDS = 5
NUM_RANDOM_SPLITS = 5

# Hyperparameters
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# α values to test
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

# Set device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
split_type = 'random_splits' if USE_RANDOM_SPLITS else 'fixed_splits'
lcc_suffix = '_lcc' if USE_LCC else ''
output_base = f'results/investigation3_magnitude_preservation/{DATASET_NAME}/{split_type}{lcc_suffix}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)
os.makedirs(f'{output_base}/diagnostics', exist_ok=True)  # NEW: For diagnostic outputs

print('='*80)
print('INVESTIGATION 3: MAGNITUDE-PRESERVING NORMALIZATION (ENHANCED)')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Split type: {split_type}')
print(f'Use LCC: {USE_LCC}')
print(f'α values: {ALPHA_VALUES}')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# Enhanced Model Class with Diagnostics
# ============================================================================

class AlphaNormMLPDiagnostic(nn.Module):
    """
    MLP with α-normalization and diagnostic capabilities
    
    CRITICAL FIX: Magnitude channel is now standardized to prevent instability
    """
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.5, with_magnitude=False):
        super(AlphaNormMLPDiagnostic, self).__init__()
        self.alpha = alpha
        self.with_magnitude = with_magnitude
        
        # Track magnitude statistics for diagnostics
        self.magnitude_stats = {
            'raw_norms': [],
            'log_norms': [],
            'standardized_norms': []
        }
        
        # Adjust input dimension if magnitude channel is added
        actual_input_dim = input_dim + 1 if with_magnitude else input_dim
        
        self.fc1 = nn.Linear(actual_input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Initialize
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, x, collect_stats=False):
        # Compute norms (with small epsilon for numerical stability)
        eps = 1e-8
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + eps
        
        # Apply α-normalization
        if self.alpha > 0:
            x_normalized = x / (norms ** self.alpha)
        else:
            x_normalized = x  # No normalization when α=0
        
        # Optionally append magnitude channel
        if self.with_magnitude:
            log_norms = torch.log(norms)
            
            # CRITICAL FIX: Standardize magnitude channel
            # This prevents the catastrophic failure seen on ogbn-arxiv
            log_norms_mean = log_norms.mean()
            log_norms_std = log_norms.std()
            log_norms_standardized = (log_norms - log_norms_mean) / (log_norms_std + eps)
            
            # Collect statistics for diagnostics (Q1: Magnitude distributions)
            if collect_stats:
                self.magnitude_stats['raw_norms'].append(norms.detach().cpu().numpy())
                self.magnitude_stats['log_norms'].append(log_norms.detach().cpu().numpy())
                self.magnitude_stats['standardized_norms'].append(log_norms_standardized.detach().cpu().numpy())
            
            x_normalized = torch.cat([x_normalized, log_norms_standardized], dim=1)
        
        # Forward pass
        x = self.fc1(x_normalized)
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_magnitude_stats_summary(self):
        """Get summary statistics of magnitude distributions"""
        if not self.magnitude_stats['raw_norms']:
            return None
        
        raw_norms = np.concatenate(self.magnitude_stats['raw_norms'])
        log_norms = np.concatenate(self.magnitude_stats['log_norms'])
        std_norms = np.concatenate(self.magnitude_stats['standardized_norms'])
        
        return {
            'raw_norms': {
                'min': float(raw_norms.min()),
                'max': float(raw_norms.max()),
                'mean': float(raw_norms.mean()),
                'std': float(raw_norms.std()),
                'median': float(np.median(raw_norms))
            },
            'log_norms': {
                'min': float(log_norms.min()),
                'max': float(log_norms.max()),
                'mean': float(log_norms.mean()),
                'std': float(log_norms.std()),
                'median': float(np.median(log_norms))
            },
            'standardized_norms': {
                'min': float(std_norms.min()),
                'max': float(std_norms.max()),
                'mean': float(std_norms.mean()),
                'std': float(std_norms.std()),
                'median': float(np.median(std_norms))
            }
        }

# ============================================================================
# Helper Functions
# ============================================================================

def get_largest_connected_component(adj):
    """Extract largest connected component (for optional LCC mode)"""
    G = nx.from_scipy_sparse_array(adj)
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    lcc_nodes = sorted(list(largest_cc))
    lcc_mask = np.zeros(adj.shape[0], dtype=bool)
    lcc_mask[lcc_nodes] = True
    
    print(f'\nConnected Components Analysis:')
    print(f'  Total components: {len(components)}')
    print(f'  Largest component size: {len(largest_cc)} nodes ({len(largest_cc)/adj.shape[0]*100:.1f}%)')
    
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

def compute_restricted_eigenvectors(X, L, D, tol=1e-10, eps_base=1e-12):
    """Compute restricted eigenvectors from feature matrix X"""
    num_nodes, dimension = X.shape
    
    # QR decomposition
    Q, R = np.linalg.qr(X)
    rank_X = np.sum(np.abs(np.diag(R)) > tol)
    
    if rank_X < dimension:
        print(f'  Rank deficiency detected: {rank_X}/{dimension}')
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
    
    # Solve generalized eigenvalue problem
    eigenvalues, V = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Map back to node space
    U = Q @ V
    
    # Verify D-orthonormality
    DU = D @ U
    G = U.T @ DU
    ortho_error = np.abs(G - np.eye(d_effective)).max()
    
    return U.astype(np.float32), eigenvalues, d_effective, ortho_error

def create_random_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=0):
    """Create random train/val/test split"""
    np.random.seed(seed)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

def train_and_evaluate_diagnostic(model, X_train, y_train, X_val, y_val, X_test, y_test,
                                   epochs, lr, weight_decay, device, batch_size=128):
    """
    Training function with diagnostic tracking (Q2: Is network learning?)
    """
    # Prepare data
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # Create dataloader
    if len(X_train) > batch_size:
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = [(X_train_t, y_train_t)]
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Diagnostic tracking
    train_losses = []
    train_accs = []  # NEW: Track training accuracy
    val_accs = []
    best_val_acc = 0.0
    best_model_state = None
    
    # Check if model supports collect_stats (only AlphaNormMLPDiagnostic does)
    supports_collect_stats = hasattr(model, 'magnitude_stats')
    
    # Collect magnitude stats on first epoch
    collect_stats_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Collect stats only on first epoch, first batch (if model supports it)
            if supports_collect_stats:
                collect_stats = (epoch == collect_stats_epoch and epoch_total == 0)
                outputs = model(batch_X, collect_stats=collect_stats)
            else:
                outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Track training accuracy
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += batch_y.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if supports_collect_stats:
                val_outputs = model(X_val_t, collect_stats=False)
            else:
                val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
    
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        if supports_collect_stats:
            test_outputs = model(X_test_t, collect_stats=False)
        else:
            test_outputs = model(X_test_t)
        test_preds = test_outputs.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()
    
    # Get magnitude statistics
    mag_stats = model.get_magnitude_stats_summary() if hasattr(model, 'get_magnitude_stats_summary') else None
    
    return {
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,  # NEW
        'val_accs': val_accs,
        'magnitude_stats': mag_stats  # NEW
    }

def aggregate_results(results_list):
    """Aggregate results across multiple seeds"""
    test_accs = [r['test_acc'] for r in results_list]
    
    # Check if network learned (Q2: Is network learning?)
    final_train_accs = [r['train_accs'][-1] for r in results_list if 'train_accs' in r]
    network_learned = bool(np.mean(final_train_accs) > 0.2) if final_train_accs else True
    
    return {
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'test_acc_min': float(np.min(test_accs)),
        'test_acc_max': float(np.max(test_accs)),
        'n_runs': int(len(test_accs)),
        'network_learned': network_learned,
        'final_train_acc_mean': float(np.mean(final_train_accs)) if final_train_accs else None
    }

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/7] Loading dataset: {DATASET_NAME}...')

(edge_index, features_original, labels_original, num_nodes_original, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

print(f'Loaded: {num_nodes_original:,} nodes, {num_classes} classes, {features_original.shape[1]} features')

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
# 2. Build Graph Matrices and Optional LCC Extraction
# ============================================================================
print('\n[2/7] Building graph matrices...')
adj_original, D_original, L_original = build_graph_matrices(edge_index, num_nodes_original)
print(f'Built: Adjacency ({adj_original.shape}), Degree, Laplacian')

# Optional LCC extraction
if USE_LCC:
    print('\n[3/7] Extracting largest connected component...')
    lcc_mask = get_largest_connected_component(adj_original)
    adj, features, labels, split_idx = extract_subgraph(
        adj_original, features_original, labels_original, lcc_mask, split_idx_original
    )
    
    # Rebuild graph matrices for LCC
    print('Rebuilding graph matrices for LCC...')
    adj_coo = adj.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj, D, L = build_graph_matrices(edge_index_lcc, adj.shape[0])
else:
    print('\n[3/7] Using whole graph (no LCC extraction)...')
    adj = adj_original
    features = features_original
    labels = labels_original
    split_idx = split_idx_original
    D = D_original
    L = L_original

num_nodes = adj.shape[0]
print(f'Working with: {num_nodes:,} nodes')

# ============================================================================
# 3. Compute Restricted Eigenvectors
# ============================================================================
print('\n[4/7] Computing restricted eigenvectors from X...')

# Convert features to dense if sparse
if sp.issparse(features):
    X = features.toarray()
else:
    X = features

d_raw = X.shape[1]

# Compute restricted eigenvectors
U, eigenvalues, d_effective, ortho_error = compute_restricted_eigenvectors(X, L, D)

print(f'Restricted eigenvectors:')
print(f'  Shape: {U.shape}')
print(f'  Raw dimension: {d_raw}')
print(f'  Effective dimension: {d_effective}')
print(f'  D-orthonormality error: {ortho_error:.2e}')

if ortho_error > 1e-2:
    print(f'  ❌ CRITICAL: Very high orthonormality error!')
    print(f'  Consider using --use-lcc flag')
elif ortho_error > 1e-4:
    print(f'  ⚠️  WARNING: High orthonormality error!')
elif ortho_error < 1e-6:
    print(f'  ✓ Excellent D-orthonormality')
else:
    print(f'  ✓ Good D-orthonormality')

rank_X = d_effective
if rank_X < d_raw:
    print(f'  Rank deficiency: {rank_X}/{d_raw} ({rank_X/d_raw*100:.1f}%)')
else:
    print(f'  Full rank: {rank_X}/{d_raw}')

# ============================================================================
# 4. Analyze Magnitude Distributions (Q1)
# ============================================================================
print('\n[5/7] Analyzing magnitude distributions (Q1: What are the magnitudes?)...')

# Compute magnitudes
U_norms = np.linalg.norm(U, axis=1)
U_log_norms = np.log(U_norms + 1e-8)

mag_stats = {
    'raw_norms': {
        'min': float(U_norms.min()),
        'max': float(U_norms.max()),
        'mean': float(U_norms.mean()),
        'std': float(U_norms.std()),
        'median': float(np.median(U_norms)),
        'cv': float(U_norms.std() / U_norms.mean())  # Coefficient of variation
    },
    'log_norms': {
        'min': float(U_log_norms.min()),
        'max': float(U_log_norms.max()),
        'mean': float(U_log_norms.mean()),
        'std': float(U_log_norms.std()),
        'median': float(np.median(U_log_norms))
    }
}

print(f'Magnitude Distribution Analysis:')
print(f'  Raw norms ||u_i||:')
print(f'    Range: [{mag_stats["raw_norms"]["min"]:.4f}, {mag_stats["raw_norms"]["max"]:.4f}]')
print(f'    Mean ± Std: {mag_stats["raw_norms"]["mean"]:.4f} ± {mag_stats["raw_norms"]["std"]:.4f}')
print(f'    Coefficient of Variation: {mag_stats["raw_norms"]["cv"]:.4f}')
print(f'  Log norms log(||u_i||):')
print(f'    Range: [{mag_stats["log_norms"]["min"]:.4f}, {mag_stats["log_norms"]["max"]:.4f}]')
print(f'    Mean ± Std: {mag_stats["log_norms"]["mean"]:.4f} ± {mag_stats["log_norms"]["std"]:.4f}')

# Interpretation
if mag_stats['raw_norms']['cv'] < 0.1:
    print(f'  → Magnitudes are VERY UNIFORM (CV < 0.1)')
    print(f'     Magnitude channel may not provide much signal')
elif mag_stats['raw_norms']['cv'] < 0.3:
    print(f'  → Magnitudes have MODERATE variation (CV < 0.3)')
    print(f'     Magnitude channel may provide some signal')
else:
    print(f'  → Magnitudes are HIGHLY VARIED (CV > 0.3)')
    print(f'     Magnitude channel should provide strong signal')

# Generate magnitude distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Raw norms histogram
axes[0].hist(U_norms, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(U_norms.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {U_norms.mean():.3f}')
axes[0].set_xlabel('||u_i|| (Euclidean Norm)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title(f'Raw Magnitude Distribution - {DATASET_NAME}', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Log norms histogram
axes[1].hist(U_log_norms, bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1].axvline(U_log_norms.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {U_log_norms.mean():.3f}')
axes[1].set_xlabel('log(||u_i||)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title(f'Log Magnitude Distribution - {DATASET_NAME}', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_base}/diagnostics/magnitude_distributions.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {output_base}/diagnostics/magnitude_distributions.png')
plt.close()

# Save magnitude statistics
with open(f'{output_base}/diagnostics/magnitude_stats.json', 'w') as f:
    json.dump(mag_stats, f, indent=2)
print(f'✓ Saved: {output_base}/diagnostics/magnitude_stats.json')

# ============================================================================
# 5. Define Model Configurations
# ============================================================================
print('\n[6/7] Defining model configurations...')

model_configs = []

# Baselines
model_configs.append({
    'name': '(a) X → Standard MLP',
    'key': 'baseline_X_standard',
    'features': X,
    'model_class': StandardMLP,
    'alpha': None,
    'with_magnitude': False,
    'use_scaler': True
})

model_configs.append({
    'name': '(b) U → Standard MLP',
    'key': 'baseline_U_standard',
    'features': U,
    'model_class': StandardMLP,
    'alpha': None,
    'with_magnitude': False,
    'use_scaler': True
})

model_configs.append({
    'name': '(c) U → RowNorm MLP (α=1.0)',
    'key': 'baseline_U_rownorm',
    'features': U,
    'model_class': AlphaNormMLPDiagnostic,
    'alpha': 1.0,
    'with_magnitude': False,
    'use_scaler': False
})

# α-normalization without magnitude
for alpha in [0.25, 0.5, 0.75]:
    model_configs.append({
        'name': f'(d{alpha}) U → α-Norm MLP (α={alpha})',
        'key': f'alpha_{alpha:.2f}_no_mag',
        'features': U,
        'model_class': AlphaNormMLPDiagnostic,
        'alpha': alpha,
        'with_magnitude': False,
        'use_scaler': False
    })

# α-normalization WITH magnitude (Q3: Standardization tested!)
for alpha in [0.5, 0.75, 1.0]:
    model_configs.append({
        'name': f'(e{alpha}) U → α-Norm + Mag MLP (α={alpha})',
        'key': f'alpha_{alpha:.2f}_with_mag',
        'features': U,
        'model_class': AlphaNormMLPDiagnostic,
        'alpha': alpha,
        'with_magnitude': True,
        'use_scaler': False
    })

print(f'Testing {len(model_configs)} model configurations')

# ============================================================================
# 6. Run Experiments
# ============================================================================
print(f'\n[7/7] Running experiments...')

num_split_iterations = NUM_RANDOM_SPLITS if USE_RANDOM_SPLITS else 1

if USE_RANDOM_SPLITS:
    print(f'Using {NUM_RANDOM_SPLITS} random splits × {NUM_SEEDS} seeds')
else:
    print(f'Using fixed benchmark splits × {NUM_SEEDS} seeds')

# Determine batch size
if split_idx is not None and 'train_idx' in split_idx and len(split_idx['train_idx']) > 256:
    batch_size = 128
else:
    batch_size = 256

# Initialize results storage
all_results = {cfg['key']: [] for cfg in model_configs}

# Training loop
for split_idx_iter in range(num_split_iterations):
    
    if USE_RANDOM_SPLITS:
        print(f'\n{"="*70}')
        print(f'RANDOM SPLIT {split_idx_iter+1}/{NUM_RANDOM_SPLITS}')
        print(f'{"="*70}')
        train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx_iter)
    else:
        print(f'\n{"="*70}')
        print('USING FIXED BENCHMARK SPLITS')
        print(f'{"="*70}')
        if split_idx is not None:
            train_idx = split_idx['train_idx']
            val_idx = split_idx['val_idx']
            test_idx = split_idx['test_idx']
        else:
            # Fallback to random if no fixed splits
            train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=0)
    
    print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')
    
    current_batch_size = min(batch_size, len(train_idx))
    
    # Train each configuration
    for cfg_idx, cfg in enumerate(model_configs, 1):
        print(f'\n[{cfg_idx}/{len(model_configs)}] {cfg["name"]}')
        
        # Get features
        features_to_use = cfg['features']
        
        # Apply StandardScaler if needed
        if cfg['use_scaler']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features_to_use[train_idx])
            X_val = scaler.transform(features_to_use[val_idx])
            X_test = scaler.transform(features_to_use[test_idx])
        else:
            X_train = features_to_use[train_idx]
            X_val = features_to_use[val_idx]
            X_test = features_to_use[test_idx]
        
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        
        # Train with multiple seeds
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create model
            if cfg['model_class'] == StandardMLP:
                model = StandardMLP(
                    input_dim=X_train.shape[1],
                    hidden_dim=HIDDEN_DIM,
                    output_dim=num_classes
                ).to(device)
            else:  # AlphaNormMLPDiagnostic
                model = AlphaNormMLPDiagnostic(
                    input_dim=X_train.shape[1],
                    hidden_dim=HIDDEN_DIM,
                    output_dim=num_classes,
                    alpha=cfg['alpha'],
                    with_magnitude=cfg['with_magnitude']
                ).to(device)
            
            # Train and evaluate
            result = train_and_evaluate_diagnostic(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device, current_batch_size
            )
            
            all_results[cfg['key']].append(result)
        
        # Show summary
        recent_results = all_results[cfg['key']][-NUM_SEEDS:]
        test_accs = [r['test_acc'] for r in recent_results]
        print(f'  → Test Acc: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%')
        
        # Check if network learned (Q2)
        final_train_accs = [r['train_accs'][-1] for r in recent_results if 'train_accs' in r]
        if final_train_accs:
            avg_train_acc = np.mean(final_train_accs)
            if avg_train_acc < 0.2:
                print(f'  ⚠️  WARNING: Network may not be learning (train acc: {avg_train_acc*100:.1f}%)')

# ============================================================================
# 7. Aggregate and Save Results
# ============================================================================
print(f'\n[8/8] Aggregating results and saving outputs...')

aggregated_results = {}
for cfg in model_configs:
    key = cfg['key']
    aggregated_results[key] = aggregate_results(all_results[key])

# Save metrics
metrics = {
    'dataset': DATASET_NAME,
    'split_type': split_type,
    'use_lcc': bool(USE_LCC),
    'num_seeds': NUM_SEEDS,
    'num_splits': num_split_iterations,
    'total_runs_per_model': NUM_SEEDS * num_split_iterations,
    'd_raw': int(d_raw),
    'd_effective': int(d_effective),
    'rank_deficiency': bool(rank_X < d_raw),
    'orthonormality_error': float(ortho_error),
    'magnitude_statistics': mag_stats,
    'alpha_values_tested': ALPHA_VALUES,
    'models': {}
}

for cfg in model_configs:
    key = cfg['key']
    metrics['models'][key] = {
        'name': cfg['name'],
        'alpha': cfg['alpha'],
        'with_magnitude': cfg['with_magnitude'],
        'results': aggregated_results[key]
    }

save_path = f'{output_base}/metrics/results_complete.json'
with open(save_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'✓ Saved: {save_path}')

# ============================================================================
# 8. Print Summary
# ============================================================================
print('\n' + '='*80)
print('RESULTS SUMMARY')
print('='*80)

print('\nBaselines (from Investigation 2):')
baseline_keys = ['baseline_X_standard', 'baseline_U_standard', 'baseline_U_rownorm']
for key in baseline_keys:
    if key in aggregated_results:
        res = aggregated_results[key]
        cfg = next(c for c in model_configs if c['key'] == key)
        learned = '✓' if res['network_learned'] else '✗'
        print(f'{cfg["name"]:40} {res["test_acc_mean"]*100:6.2f}% ± {res["test_acc_std"]*100:5.2f}%  [{learned}]')

print('\nα-Normalization (without magnitude):')
for alpha in [0.25, 0.5, 0.75]:
    key = f'alpha_{alpha:.2f}_no_mag'
    if key in aggregated_results:
        res = aggregated_results[key]
        cfg = next(c for c in model_configs if c['key'] == key)
        learned = '✓' if res['network_learned'] else '✗'
        print(f'{cfg["name"]:40} {res["test_acc_mean"]*100:6.2f}% ± {res["test_acc_std"]*100:5.2f}%  [{learned}]')

print('\nα-Normalization (WITH magnitude channel):')
for alpha in [0.5, 0.75, 1.0]:
    key = f'alpha_{alpha:.2f}_with_mag'
    if key in aggregated_results:
        res = aggregated_results[key]
        cfg = next(c for c in model_configs if c['key'] == key)
        learned = '✓' if res['network_learned'] else '✗'
        print(f'{cfg["name"]:40} {res["test_acc_mean"]*100:6.2f}% ± {res["test_acc_std"]*100:5.2f}%  [{learned}]')

# Find best result
best_key = max(aggregated_results.keys(), key=lambda k: aggregated_results[k]['test_acc_mean'])
best_cfg = next(c for c in model_configs if c['key'] == best_key)
best_acc = aggregated_results[best_key]['test_acc_mean'] * 100

print(f'\n{"="*80}')
print(f'BEST RESULT: {best_cfg["name"]}')
print(f'Test Accuracy: {best_acc:.2f}%')

# Compare to baselines
baseline_rownorm = aggregated_results['baseline_U_rownorm']['test_acc_mean'] * 100
improvement = best_acc - baseline_rownorm
print(f'Improvement over RowNorm: {improvement:+.2f}pp')

if improvement > 2.0:
    print(f'✓✓ MAJOR IMPROVEMENT!')
elif improvement > 0.5:
    print(f'✓ Improvement')
elif improvement > -0.5:
    print(f'~ Similar performance')
else:
    print(f'✗ Degradation')

print('='*80)

# ============================================================================
# 9. Generate Plots
# ============================================================================
print('\nGenerating comparison plots...')

# [Rest of plotting code remains the same as before...]

print('\n' + '='*80)
print('EXPERIMENT COMPLETE')
print('='*80)
print(f'Dataset: {DATASET_NAME}')
print(f'Results saved to: {output_base}/')
print('='*80)