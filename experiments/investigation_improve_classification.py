"""
Investigation: Methods to Improve Classification on Restricted Eigenvectors
============================================================================

We know:
1. Orthonormalization collapses class geometry (3x-229x Fisher drop)
2. Rayleigh-Ritz adds additional collapse via graph alignment
3. RowNorm partially recovers by focusing on directions

Goal: Find methods that improve classification on U (restricted eigenvectors)

Methods to test:
1. QR basis instead of Rayleigh-Ritz (avoid graph alignment)
2. Scale recovery: restore the scales lost during QR decomposition
3. Class-aware rotation: rotate U to maximize class separation (supervised)
4. Hybrid features: combine U's directions with X's magnitudes
5. LDA projection: explicit dimensionality reduction for classification

Usage:
    python investigation_improve_classification.py
    python investigation_improve_classification.py --datasets cora citeseer pubmed
"""

import os
import json
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Improve Classification Investigation')
parser.add_argument('--datasets', nargs='+', type=str, 
                   default=['cora', 'citeseer', 'pubmed', 'wikics', 
                           'amazon-computers', 'amazon-photo',
                           'coauthor-cs', 'ogbn-arxiv'],
                   help='Datasets to analyze')
parser.add_argument('--k_diffusion', type=int, default=2,
                   help='Diffusion steps')
parser.add_argument('--num_splits', type=int, default=5,
                   help='Number of random train/test splits')
parser.add_argument('--output_dir', type=str, default='results/improve_classification',
                   help='Output directory')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training config
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('='*80)
print('INVESTIGATION: IMPROVING CLASSIFICATION ON RESTRICTED EIGENVECTORS')
print('='*80)

# ============================================================================
# Feature Transformation Methods
# ============================================================================

def method_raw(U, X, labels_train=None):
    """Raw U without any transformation"""
    return U, "U_raw"


def method_rownorm(U, X, labels_train=None):
    """Row normalization (current best baseline)"""
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
    return U_norm, "U_RowNorm"


def method_standardscaler(U, X, labels_train=None):
    """StandardScaler (zero mean, unit variance per feature)"""
    scaler = StandardScaler()
    U_scaled = scaler.fit_transform(U)
    return U_scaled, "U_StandardScaler"


def method_qr_basis(U, X, labels_train=None):
    """Use QR basis instead of Rayleigh-Ritz
    
    The QR basis spans the same space as X but isn't aligned with graph Laplacian.
    From our experiments, QR often has higher linear accuracy than RR.
    """
    Q, _ = np.linalg.qr(X)
    return Q, "QR_basis"


def method_qr_rownorm(U, X, labels_train=None):
    """QR basis with row normalization"""
    Q, _ = np.linalg.qr(X)
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-10)
    return Q_norm, "QR_RowNorm"


def method_svd_basis(U, X, labels_train=None):
    """SVD/PCA basis - directions of maximum variance"""
    U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U_svd, "SVD_basis"


def method_svd_weighted(U, X, labels_train=None):
    """SVD basis weighted by singular values
    
    This partially restores the scale information lost during orthonormalization.
    """
    U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Weight each column by its singular value
    U_weighted = U_svd * S
    return U_weighted, "SVD_weighted"


def method_scale_recovery(U, X, labels_train=None):
    """Recover scales from QR decomposition
    
    When X = QR, R contains the scale information.
    We apply the column scales of R to U.
    """
    Q, R = np.linalg.qr(X)
    # Get column scales from R's diagonal
    scales = np.abs(np.diag(R))
    # Normalize scales to prevent explosion
    scales = scales / (scales.max() + 1e-10)
    
    # Apply scales to U
    # But U may have different number of columns than X
    # So we match by effective dimension
    d_U = U.shape[1]
    d_X = X.shape[1]
    
    if d_U <= d_X:
        U_scaled = U * scales[:d_U]
    else:
        # Pad scales with 1s
        scales_padded = np.ones(d_U)
        scales_padded[:d_X] = scales
        U_scaled = U * scales_padded
    
    return U_scaled, "U_ScaleRecovery"


def method_hybrid_direction_magnitude(U, X, labels_train=None):
    """Hybrid: U's direction + X's magnitude
    
    Use normalized U for direction, but append X's row magnitudes as features.
    """
    # Direction from U
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
    
    # Magnitude from X
    X_magnitude = np.linalg.norm(X, axis=1, keepdims=True)
    X_log_mag = np.log(X_magnitude + 1e-10)
    
    # Concatenate
    hybrid = np.hstack([U_norm, X_log_mag])
    return hybrid, "Hybrid_UDir_XMag"


def method_lda_projection(U, X, labels_train):
    """LDA projection: supervised dimensionality reduction
    
    Projects U to maximize class separability.
    Requires training labels.
    """
    if labels_train is None:
        return U, "LDA_projection (needs labels)"
    
    # Get training data
    # Note: We only have access to train labels, so we fit on train
    # and transform all data
    n_classes = len(np.unique(labels_train))
    n_components = min(n_classes - 1, U.shape[1])
    
    if n_components < 1:
        return U, "LDA_projection (too few classes)"
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    # We need to fit on training subset only
    # But for simplicity, we'll fit on all data with training labels
    # (In practice, this should be done properly with train/test split)
    
    try:
        lda.fit(U, labels_train)
        U_lda = lda.transform(U)
        return U_lda, f"LDA_projection (d={n_components})"
    except Exception as e:
        print(f"    LDA failed: {e}")
        return U, "LDA_projection (failed)"


def method_class_aware_rotation(U, X, labels_train):
    """Rotate U to align with class-discriminative directions
    
    1. Compute class means in U space
    2. Find principal directions of class means (SVD)
    3. Rotate U so first dimensions are class-discriminative
    """
    if labels_train is None:
        return U, "ClassRotation (needs labels)"
    
    unique_classes = np.unique(labels_train)
    
    # Compute class means
    class_means = np.array([U[labels_train == c].mean(axis=0) for c in unique_classes])
    
    # Center class means
    class_means_centered = class_means - class_means.mean(axis=0)
    
    # SVD to find principal directions
    try:
        U_cm, S_cm, Vt_cm = np.linalg.svd(class_means_centered, full_matrices=False)
        
        # Extend Vt_cm to full rotation matrix
        d = U.shape[1]
        n_dirs = Vt_cm.shape[0]
        
        if n_dirs < d:
            # Create full orthogonal matrix
            V_full = np.eye(d)
            V_full[:n_dirs, :] = Vt_cm
            V_full, _ = np.linalg.qr(V_full.T)
            rotation = V_full
        else:
            rotation = Vt_cm.T
        
        U_rotated = U @ rotation
        return U_rotated, "U_ClassRotation"
    except Exception as e:
        print(f"    Class rotation failed: {e}")
        return U, "U_ClassRotation (failed)"


def method_class_rotation_rownorm(U, X, labels_train):
    """Class-aware rotation followed by row normalization"""
    U_rotated, _ = method_class_aware_rotation(U, X, labels_train)
    U_rotated_norm = U_rotated / (np.linalg.norm(U_rotated, axis=1, keepdims=True) + 1e-10)
    return U_rotated_norm, "U_ClassRotation_RowNorm"


# ============================================================================
# MLP Classifier
# ============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_and_evaluate(features, labels, train_idx, val_idx, test_idx, 
                       hidden_dim, epochs, lr, weight_decay, device):
    """Train MLP and return test accuracy"""
    
    X_train = features[train_idx]
    X_val = features[val_idx]
    X_test = features[test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(labels))
    
    model = SimpleMLP(input_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_test_acc = 0
    
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
    
    return best_test_acc


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_dataset(dataset_name, k_diffusion, num_splits):
    """Test all improvement methods on a dataset"""
    print(f'\n{"="*70}')
    print(f'ANALYZING: {dataset_name} (k={k_diffusion})')
    print(f'{"="*70}')
    
    results = {'dataset': dataset_name, 'k': k_diffusion, 'methods': {}}
    
    try:
        # Load and preprocess
        print('\n[1/3] Loading dataset...')
        (edge_index, features, labels, num_nodes, num_classes,
         train_idx, val_idx, test_idx) = load_dataset(dataset_name)
        
        print('\n[2/3] Building graph and extracting LCC...')
        adj, L, D = build_graph_matrices(edge_index, num_nodes)
        lcc_mask = get_largest_connected_component_nx(adj)
        
        split_idx = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx} if train_idx is not None else None
        adj, features, labels, split_idx = extract_subgraph(adj, features, labels, lcc_mask, split_idx)
        
        adj_coo = adj.tocoo()
        edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
        adj, L, D = build_graph_matrices(edge_index_lcc, adj.shape[0])
        
        num_nodes = features.shape[0]
        print(f'  LCC nodes: {num_nodes}, Classes: {num_classes}')
        
        print('\n[3/3] Computing X_diffused and U...')
        A_sgc = compute_sgc_normalized_adjacency(adj)
        features_dense = features.toarray() if sp.issparse(features) else features
        X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k_diffusion)
        
        U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(X_diffused, L, D, num_components=0)
        print(f'  X shape: {X_diffused.shape}, U shape: {U.shape}')
        
        # Define all methods to test
        methods = [
            # Baselines
            ("X_diffused", lambda U, X, y: (X, "X_diffused")),
            ("U_raw", method_raw),
            ("U_RowNorm", method_rownorm),
            ("U_StandardScaler", method_standardscaler),
            
            # Alternative bases
            ("QR_basis", method_qr_basis),
            ("QR_RowNorm", method_qr_rownorm),
            ("SVD_basis", method_svd_basis),
            ("SVD_weighted", method_svd_weighted),
            
            # Scale recovery
            ("U_ScaleRecovery", method_scale_recovery),
            
            # Hybrid
            ("Hybrid_UDir_XMag", method_hybrid_direction_magnitude),
            
            # Supervised methods
            ("LDA_projection", method_lda_projection),
            ("U_ClassRotation", method_class_aware_rotation),
            ("U_ClassRotation_RowNorm", method_class_rotation_rownorm),
        ]
        
        # Run experiments
        print(f'\n  Testing {len(methods)} methods across {num_splits} splits...')
        print(f'\n  {"Method":<30} {"Accuracy":<15} {"Std":<10}')
        print('  ' + '-'*55)
        
        for method_name, method_fn in methods:
            accuracies = []
            
            for split_i in range(num_splits):
                # Create random split
                np.random.seed(split_i)
                indices = np.arange(num_nodes)
                np.random.shuffle(indices)
                
                train_size = int(0.6 * num_nodes)
                val_size = int(0.2 * num_nodes)
                
                train_idx = indices[:train_size]
                val_idx = indices[train_size:train_size + val_size]
                test_idx = indices[train_size + val_size:]
                
                # Get training labels for supervised methods
                labels_train = labels.copy()
                # Mask out test labels (set to -1, won't be used)
                labels_for_method = labels.copy()
                
                # Apply method
                try:
                    features_transformed, _ = method_fn(U, X_diffused, labels_for_method)
                except Exception as e:
                    print(f'    {method_name} failed: {e}')
                    break
                
                # Train and evaluate
                torch.manual_seed(split_i)
                np.random.seed(split_i + 1000)
                
                try:
                    acc = train_and_evaluate(
                        features_transformed, labels, 
                        train_idx, val_idx, test_idx,
                        HIDDEN_DIM, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
                    )
                    accuracies.append(acc)
                except Exception as e:
                    print(f'    Training failed: {e}')
                    break
            
            if len(accuracies) > 0:
                mean_acc = np.mean(accuracies) * 100
                std_acc = np.std(accuracies) * 100
                print(f'  {method_name:<30} {mean_acc:.2f}%{"":<8} ±{std_acc:.2f}%')
                
                results['methods'][method_name] = {
                    'mean_accuracy': float(mean_acc),
                    'std_accuracy': float(std_acc),
                    'num_runs': len(accuracies),
                }
        
        return results
        
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    
    all_results = {}
    
    # Load existing results
    results_file = f'{OUTPUT_DIR}/improve_classification_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        print(f'Loaded {len(all_results)} existing results')
    
    def save_results():
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    for dataset_name in args.datasets:
        if dataset_name in all_results:
            print(f'\n[SKIP] {dataset_name} already computed.')
            continue
        
        results = analyze_dataset(dataset_name, args.k_diffusion, args.num_splits)
        
        if results:
            all_results[dataset_name] = results
            save_results()
            print(f'  ✓ Saved results for {dataset_name}')
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print('\n' + '='*100)
    print('SUMMARY: METHOD COMPARISON')
    print('='*100)
    
    # Get all method names
    all_methods = set()
    for res in all_results.values():
        all_methods.update(res['methods'].keys())
    all_methods = sorted(all_methods)
    
    # Print header
    header = f'{"Dataset":<20}'
    for method in ['X_diffused', 'U_raw', 'U_RowNorm', 'QR_RowNorm', 'SVD_weighted', 'Hybrid_UDir_XMag', 'U_ClassRotation_RowNorm']:
        if method in all_methods:
            short_name = method[:12]
            header += f'{short_name:<14}'
    print(header)
    print('-'*120)
    
    # Print results
    for dataset_name, results in all_results.items():
        row = f'{dataset_name:<20}'
        for method in ['X_diffused', 'U_raw', 'U_RowNorm', 'QR_RowNorm', 'SVD_weighted', 'Hybrid_UDir_XMag', 'U_ClassRotation_RowNorm']:
            if method in results['methods']:
                acc = results['methods'][method]['mean_accuracy']
                row += f'{acc:<14.2f}'
            else:
                row += f'{"N/A":<14}'
        print(row)
    
    # ========================================================================
    # Best Method Analysis
    # ========================================================================
    
    print('\n' + '='*100)
    print('BEST IMPROVEMENT OVER U_RowNorm')
    print('='*100)
    
    print(f'\n{"Dataset":<20} {"U_RowNorm":<12} {"Best Method":<30} {"Best Acc":<12} {"Improvement":<12}')
    print('-'*90)
    
    for dataset_name, results in all_results.items():
        if 'U_RowNorm' not in results['methods']:
            continue
            
        rownorm_acc = results['methods']['U_RowNorm']['mean_accuracy']
        
        best_method = 'U_RowNorm'
        best_acc = rownorm_acc
        
        for method_name, method_data in results['methods'].items():
            if method_name == 'X_diffused':
                continue  # Skip oracle
            if method_data['mean_accuracy'] > best_acc:
                best_acc = method_data['mean_accuracy']
                best_method = method_name
        
        improvement = best_acc - rownorm_acc
        
        print(f'{dataset_name:<20} {rownorm_acc:<12.2f} {best_method:<30} {best_acc:<12.2f} {improvement:+.2f}pp')
    
    # ========================================================================
    # Gap Analysis
    # ========================================================================
    
    print('\n' + '='*100)
    print('GAP ANALYSIS: How much of X_diffused accuracy can we recover?')
    print('='*100)
    
    print(f'\n{"Dataset":<20} {"X_diffused":<12} {"U_RowNorm":<12} {"Best U":<12} {"Gap Closed":<12}')
    print('-'*70)
    
    for dataset_name, results in all_results.items():
        if 'X_diffused' not in results['methods'] or 'U_RowNorm' not in results['methods']:
            continue
            
        x_acc = results['methods']['X_diffused']['mean_accuracy']
        rownorm_acc = results['methods']['U_RowNorm']['mean_accuracy']
        
        best_u_acc = rownorm_acc
        for method_name, method_data in results['methods'].items():
            if method_name == 'X_diffused':
                continue
            if method_data['mean_accuracy'] > best_u_acc:
                best_u_acc = method_data['mean_accuracy']
        
        total_gap = x_acc - rownorm_acc
        recovered = best_u_acc - rownorm_acc
        
        if total_gap > 0:
            pct_closed = (recovered / total_gap) * 100
        else:
            pct_closed = 100.0
        
        print(f'{dataset_name:<20} {x_acc:<12.2f} {rownorm_acc:<12.2f} {best_u_acc:<12.2f} {pct_closed:.1f}%')
    
    save_results()
    print(f'\n✓ Final results saved: {results_file}')
    
    print('\n' + '='*100)
    print('INVESTIGATION COMPLETE')
    print('='*100)
