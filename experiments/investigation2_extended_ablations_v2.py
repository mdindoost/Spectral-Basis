"""
Investigation 2 Extended: Ablation Studies on Basis Sensitivity
================================================================

Extended experiments to understand WHY basis sensitivity occurs in Direction B.

Three NEW experiments:
  (b') U → No scaling → Standard MLP          [Tests: Does StandardScaler hurt?]
  (b'') U_weighted → StandardScaler → MLP     [Tests: Do eigenvalues help?]
  PCA Analysis: X vs U geometry               [Tests: What properties differ?]

Usage:
    # Fixed benchmark splits (default)
    python investigation2_extended_ablations_v2.py [dataset_name]
    
    # Random 60/20/20 splits (for statistical robustness)
    python investigation2_extended_ablations_v2.py [dataset_name] --random-splits
    
Examples:
    python investigation2_extended_ablations_v2.py ogbn-arxiv
    python investigation2_extended_ablations_v2.py cora --random-splits
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import (
    StandardMLP, RowNormMLP,
    build_graph_matrices, load_dataset
)

# ============================================================================
# Configuration
# ============================================================================
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ogbn-arxiv'
USE_RANDOM_SPLITS = '--random-splits' in sys.argv

VALID_DATASETS = [
    'ogbn-arxiv',
    'cora',
    'citeseer',
    'pubmed',
    'wikics',
    'amazon-photo',
    'amazon-computers',
    'coauthor-cs',
    'coauthor-physics'
]

if DATASET_NAME not in VALID_DATASETS:
    print(f"Error: Invalid dataset '{DATASET_NAME}'")
    print(f"Valid datasets: {', '.join(VALID_DATASETS)}")
    sys.exit(1)

# Experimental parameters
NUM_SEEDS = 5
NUM_RANDOM_SPLITS = 5 if USE_RANDOM_SPLITS else 1

# Hyperparameters (MATCHING Investigation 2 Directions A&B)
HIDDEN_DIM = 256
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('='*70)
print(f'INVESTIGATION 2 EXTENDED: ABLATION STUDIES - {DATASET_NAME.upper()}')
print('='*70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'Random splits: {"Yes" if USE_RANDOM_SPLITS else "No (fixed benchmark)"}')
print('='*70)

# Create output directories
split_type = 'random_splits' if USE_RANDOM_SPLITS else 'fixed_splits'
output_base = f'results/investigation2_extended_ablations/{DATASET_NAME}/{split_type}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/8] Loading {DATASET_NAME}...')

(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_fixed, val_idx_fixed, test_idx_fixed) = load_dataset(DATASET_NAME, root='./dataset')

if X_raw is None:
    print(f"ERROR: Dataset {DATASET_NAME} has no node features!")
    print("Investigation 2 requires node features to compute X-restricted eigenvectors.")
    sys.exit(1)

d_raw = X_raw.shape[1]

print(f'\nDataset Statistics:')
print(f'  Nodes: {num_nodes:,}')
print(f'  Edges: {edge_index.shape[1]:,}')
print(f'  Classes: {num_classes}')
print(f'  Raw feature dimension: {d_raw}')
print(f'  Train: {len(train_idx_fixed):,} ({len(train_idx_fixed)/num_nodes*100:.1f}%)')
print(f'  Val: {len(val_idx_fixed):,} ({len(val_idx_fixed)/num_nodes*100:.1f}%)')
print(f'  Test: {len(test_idx_fixed):,} ({len(test_idx_fixed)/num_nodes*100:.1f}%)')

# ============================================================================
# 2. Build Graph Matrices
# ============================================================================
print(f'\n[2/8] Building graph matrices...')

adj, L, D = build_graph_matrices(edge_index, num_nodes)

print(f'Adjacency: {adj.shape}, nnz={adj.nnz:,}')
print(f'Laplacian: {L.shape}')
print(f'Degree matrix: {D.shape}')

# ============================================================================
# 3. Compute X-Restricted Eigenvectors using QR
# ============================================================================
print(f'\n[3/8] Computing X-restricted eigenvectors...')

# QR decomposition to handle potential rank deficiency
print('Performing QR decomposition on X...')
Q, R = np.linalg.qr(X_raw.astype(np.float64))
rank_X = np.sum(np.abs(np.diag(R)) > 1e-10)

print(f'Original dimension: {d_raw}')
print(f'Effective rank: {rank_X}/{d_raw}')

if rank_X < d_raw:
    print(f'⚠️  Rank deficiency detected: {d_raw - rank_X} dimensions removed')
    Q = Q[:, :rank_X]
    d_effective = rank_X
else:
    print('✓ Full rank - no dimension reduction needed')
    d_effective = d_raw

# Compute projected Laplacian and degree matrix
print('Computing projected matrices (Q^T L Q, Q^T D Q)...')
LQ = (L @ Q).astype(np.float64)
DQ = (D @ Q).astype(np.float64)

L_reduced = Q.T @ LQ
D_reduced = Q.T @ DQ

# Symmetrize for numerical stability
L_reduced = 0.5 * (L_reduced + L_reduced.T)
D_reduced = 0.5 * (D_reduced + D_reduced.T)

print(f'Reduced Laplacian: {L_reduced.shape}')
print(f'Reduced degree matrix: {D_reduced.shape}')

# Solve reduced generalized eigenproblem
print(f'Solving generalized eigenproblem in R^{d_effective}...')
eigenvalues, V = la.eigh(L_reduced, D_reduced)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
V = V[:, idx]

print(f'Eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]')

# Map back to node space
U = (Q @ V).astype(np.float32)
print(f'Restricted eigenvectors U: {U.shape}')

# Verify D-orthonormality
DU = (D @ U).astype(np.float64)
G = U.astype(np.float64).T @ DU
orthonormality_error = np.abs(G - np.eye(d_effective)).max()
print(f'D-orthonormality check: max |U^T D U - I| = {orthonormality_error:.2e}')

if orthonormality_error > 1e-4:
    print('⚠️  WARNING: Poor D-orthonormality! Results may be unreliable.')
elif orthonormality_error > 1e-6:
    print('⚠️  Acceptable D-orthonormality, but monitor results carefully.')
else:
    print('✓ Excellent D-orthonormality')

# ============================================================================
# 4. Create Eigenvalue-Weighted Features
# ============================================================================
print(f'\n[4/8] Creating eigenvalue-weighted features for Experiment 2...')

# Weight by 1/sqrt(λ) to normalize by "frequency"
weights = 1.0 / np.sqrt(eigenvalues + 1e-8)
U_weighted = U * weights[np.newaxis, :]

print(f'U_weighted shape: {U_weighted.shape}')
print(f'Weight range: [{weights.min():.2f}, {weights.max():.2f}]')

if np.isnan(U_weighted).any() or np.isinf(U_weighted).any():
    print('⚠️  WARNING: NaN or Inf values in U_weighted!')
else:
    print('✓ U_weighted is clean')

# ============================================================================
# 5. PCA Spectrum Analysis (No Training)
# ============================================================================
print(f'\n[5/8] EXPERIMENT 3: PCA Spectrum Analysis')
print('='*70)
print('Goal: Characterize geometric differences between X and U')
print('='*70)

def analyze_spectrum(features, name, train_idx):
    """Analyze eigenvalue spectrum and geometric properties"""
    X_train = features[train_idx]
    n, d = X_train.shape
    
    # Center the data
    X_centered = X_train - X_train.mean(axis=0)
    
    # Compute covariance matrix
    C = (X_centered.T @ X_centered) / n
    
    # Get eigenvalues (sorted descending)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = np.maximum(eigvals, 1e-12)
    
    # Metrics
    eigvals_normalized = eigvals / eigvals.sum()
    effective_rank = 1.0 / (eigvals_normalized ** 2).sum()
    condition_number = eigvals[0] / eigvals[-1]
    
    # Decay rate
    indices = np.arange(1, len(eigvals) + 1)
    mask = eigvals > 1e-10
    if mask.sum() > 10:
        log_indices = np.log(indices[mask][:min(50, mask.sum())])
        log_eigvals = np.log(eigvals[mask][:min(50, mask.sum())])
        decay_rate = -np.polyfit(log_indices, log_eigvals, 1)[0]
    else:
        decay_rate = 0.0
    
    # Entropy
    eigvals_prob = eigvals_normalized[eigvals_normalized > 1e-12]
    entropy = -(eigvals_prob * np.log(eigvals_prob)).sum()
    
    # Isotropy (CV)
    cv = eigvals.std() / eigvals.mean()
    
    # Variance explained
    cumsum = np.cumsum(eigvals_normalized)
    var_50 = np.searchsorted(cumsum, 0.50) + 1
    var_90 = np.searchsorted(cumsum, 0.90) + 1
    var_99 = np.searchsorted(cumsum, 0.99) + 1
    
    return {
        'name': name,
        'dimension': d,
        'effective_rank': effective_rank,
        'effective_rank_ratio': effective_rank / d,
        'condition_number': condition_number,
        'decay_rate': decay_rate,
        'entropy': entropy,
        'coefficient_variation': cv,
        'components_50': var_50,
        'components_90': var_90,
        'components_99': var_99,
        'eigenvalues': eigvals.tolist()[:100],  # First 100 only
        'eigenvalue_cumsum': cumsum.tolist()[:100]
    }, eigvals

# Analyze X and U
print('\nAnalyzing X (raw features)...')
results_X, eigvals_X = analyze_spectrum(X_raw, 'X', train_idx_fixed)
print(f"  Effective Rank: {results_X['effective_rank']:.1f} ({results_X['effective_rank_ratio']*100:.1f}%)")
print(f"  Condition Number: {results_X['condition_number']:.2e}")
print(f"  CV (isotropy): {results_X['coefficient_variation']:.3f}")

print('\nAnalyzing U (restricted eigenvectors)...')
results_U, eigvals_U = analyze_spectrum(U, 'U', train_idx_fixed)
print(f"  Effective Rank: {results_U['effective_rank']:.1f} ({results_U['effective_rank_ratio']*100:.1f}%)")
print(f"  Condition Number: {results_U['condition_number']:.2e}")
print(f"  CV (isotropy): {results_U['coefficient_variation']:.3f}")

# Store PCA results
pca_results = {
    'X': results_X,
    'U': results_U,
    'comparison': {
        'effective_rank_ratio': results_U['effective_rank_ratio'] / results_X['effective_rank_ratio'],
        'condition_number_ratio': results_U['condition_number'] / results_X['condition_number'],
        'isotropy_ratio': results_U['coefficient_variation'] / results_X['coefficient_variation']
    }
}

# ============================================================================
# 6. Training Function (Matching Investigation 2 style)
# ============================================================================

def train_with_granular_tracking(model, train_loader, X_val, y_val, X_test, y_test,
                                epochs=200, lr=0.01, weight_decay=5e-4, device='cpu',
                                batch_size=128):
    """Training with checkpoint tracking and batch-level metrics for first 5 epochs"""
    import copy
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_accs = [], []
    checkpoint_accs = {}
    batch_level_tracking = []
    
    checkpoints = [10, 20, 40, 80, 160, 200]
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Batch-level tracking for first 5 epochs
            if epoch < 5:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val).argmax(1)
                    val_acc = (val_pred == y_val).float().mean().item()
                
                global_step = epoch * len(train_loader) + batch_idx
                batch_level_tracking.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'global_step': global_step,
                    'val_acc': val_acc
                })
                model.train()
        
        # Epoch-level validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / len(train_loader))
        val_accs.append(val_acc)
        
        # Store checkpoint accuracies
        if (epoch + 1) in checkpoints:
            checkpoint_accs[epoch + 1] = val_acc
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    # Convergence metrics
    final_val_acc = val_accs[-1]
    thresholds = [0.90, 0.95, 0.99]
    convergence_metrics = {}
    
    for thresh in thresholds:
        target = thresh * final_val_acc
        epochs_to_reach = next((i for i, acc in enumerate(val_accs) if acc >= target), len(val_accs))
        convergence_metrics[f'speed_to_{int(thresh*100)}'] = epochs_to_reach
    
    # AUC
    auc = np.trapz(val_accs, dx=1) / len(val_accs)
    convergence_metrics['auc'] = auc / final_val_acc if final_val_acc > 0 else 0.0
    
    # Convergence rate
    if len(val_accs) >= 20:
        convergence_metrics['convergence_rate'] = (val_accs[19] - val_accs[0]) / 20
    else:
        convergence_metrics['convergence_rate'] = 0.0
    
    convergence_metrics['final_acc'] = final_val_acc
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'checkpoint_accs': checkpoint_accs,
        'batch_level_tracking': batch_level_tracking,
        'convergence_metrics': convergence_metrics
    }

def aggregate_results(results_list):
    """Aggregate results across multiple seeds"""
    test_accs = [r['test_acc'] for r in results_list]
    
    # Validation accuracy curves
    val_accs_array = np.array([r['val_accs'] for r in results_list])
    train_losses_array = np.array([r['train_losses'] for r in results_list])
    
    # Checkpoint accuracies
    all_checkpoints = {}
    checkpoints = [10, 20, 40, 80, 160, 200]
    for cp in checkpoints:
        cp_accs = [r['checkpoint_accs'].get(cp, np.nan) for r in results_list]
        cp_accs = [acc for acc in cp_accs if not np.isnan(acc)]
        if cp_accs:
            all_checkpoints[cp] = {
                'mean': float(np.mean(cp_accs)),
                'std': float(np.std(cp_accs))
            }
    
    # Convergence metrics
    conv_metrics = {}
    metric_keys = ['speed_to_90', 'speed_to_95', 'speed_to_99', 'auc', 'convergence_rate', 'final_acc']
    for key in metric_keys:
        values = [r['convergence_metrics'][key] for r in results_list]
        conv_metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    return {
        'test_acc': {
            'mean': float(np.mean(test_accs)),
            'std': float(np.std(test_accs)),
            'min': float(np.min(test_accs)),
            'max': float(np.max(test_accs))
        },
        'val_accs': {
            'mean': val_accs_array.mean(axis=0),
            'std': val_accs_array.std(axis=0)
        },
        'train_losses': {
            'mean': train_losses_array.mean(axis=0),
            'std': train_losses_array.std(axis=0)
        },
        'checkpoint_accs': all_checkpoints,
        'convergence_metrics': conv_metrics
    }

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

# ============================================================================
# 7. Train All Models with Multiple Seeds and Splits
# ============================================================================
print(f'\n[6/8] Training models across splits and seeds...')

if USE_RANDOM_SPLITS:
    print(f'Across {NUM_RANDOM_SPLITS} random splits × {NUM_SEEDS} seeds')
    print(f'Total runs: {5 * NUM_SEEDS * NUM_RANDOM_SPLITS} = 5(models) × {NUM_SEEDS}(seeds) × {NUM_RANDOM_SPLITS}(splits)')
else:
    print(f'Total runs: {5 * NUM_SEEDS} = 5(models) × {NUM_SEEDS}(seeds)')

# Model configurations - COMPLETE SET
model_configs = [
    # Original experiments from Direction B
    ('(a) Standard MLP on X (scaled)', 'standard_X_scaled', 'X', True, 'standard'),
    ('(b) Standard MLP on V (scaled)', 'standard_V_scaled', 'V', True, 'standard'),
    ('(c) RowNorm MLP on V (unscaled)', 'rownorm_V', 'V', False, 'rownorm'),
    
    # NEW EXPERIMENT 1: U without scaling
    ("(b') Standard MLP on V (unscaled)", 'standard_V_unscaled', 'V', False, 'standard'),
    
    # NEW EXPERIMENT 2: Eigenvalue-weighted U
    ("(b'') Standard MLP on V_weighted (scaled)", 'standard_V_weighted', 'V_weighted', True, 'standard'),
]

all_results = {key: [] for _, key, _, _, _ in model_configs}

# Determine batch size based on training set size
if len(train_idx_fixed) > 256:
    batch_size = 128
    print(f'Batch size: {batch_size} (dataset has {len(train_idx_fixed)} training samples)')
else:
    batch_size = len(train_idx_fixed)
    print(f'Batch size: {batch_size} (full batch for small dataset)')

num_split_iterations = NUM_RANDOM_SPLITS if USE_RANDOM_SPLITS else 1

# Outer loop: splits
for split_idx in range(num_split_iterations):
    
    if USE_RANDOM_SPLITS:
        print(f'\n{"="*70}')
        print(f'RANDOM SPLIT {split_idx+1}/{NUM_RANDOM_SPLITS}')
        print(f'{"="*70}')
        train_idx, val_idx, test_idx = create_random_split(num_nodes, seed=split_idx)
    else:
        print(f'\n{"="*70}')
        print('USING FIXED BENCHMARK SPLITS')
        print(f'{"="*70}')
        train_idx, val_idx, test_idx = train_idx_fixed, val_idx_fixed, test_idx_fixed
    
    print(f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}')
    
    # Parameter-to-sample ratio
    params_per_model = (d_effective * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * num_classes)
    ratio = params_per_model / len(train_idx)
    print(f'Parameter-to-sample ratio: {ratio:.1f}:1')
    if ratio > 10:
        print(f'⚠️  WARNING: High parameter-to-sample ratio! Expect overfitting.')
    
    # Prepare labels
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)
    y_test = torch.from_numpy(labels[test_idx]).long().to(device)
    
    # Inner loop: random seeds
    for seed_idx in range(NUM_SEEDS):
        seed = split_idx * NUM_SEEDS + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f'\n--- Split {split_idx+1}/{num_split_iterations}, Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed}) ---')
        
        # Train each model
        for model_name, model_key, feature_type, use_scaling, model_type in model_configs:
            print(f'\nTraining: {model_name}...')
            
            # Prepare features based on experiment type
            if feature_type == 'X':
                features = X_raw
                input_dim = d_raw
            elif feature_type == 'V':
                features = U
                input_dim = d_effective
            elif feature_type == 'V_weighted':
                features = U_weighted
                input_dim = d_effective
            
            # Apply scaling if needed
            if use_scaling:
                scaler = StandardScaler().fit(features[train_idx])
                F_train = torch.from_numpy(scaler.transform(features[train_idx])).float().to(device)
                F_val = torch.from_numpy(scaler.transform(features[val_idx])).float().to(device)
                F_test = torch.from_numpy(scaler.transform(features[test_idx])).float().to(device)
            else:
                F_train = torch.from_numpy(features[train_idx]).float().to(device)
                F_val = torch.from_numpy(features[val_idx]).float().to(device)
                F_test = torch.from_numpy(features[test_idx]).float().to(device)
            
            train_loader = DataLoader(TensorDataset(F_train, y_train), batch_size=batch_size, shuffle=True)
            
            # Create model
            if model_type == 'standard':
                model = StandardMLP(input_dim, HIDDEN_DIM, num_classes)
            else:  # rownorm
                model = RowNormMLP(input_dim, HIDDEN_DIM, num_classes)
            
            # Train
            results = train_with_granular_tracking(
                model, train_loader, F_val, y_val, F_test, y_test,
                epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                device=device, batch_size=batch_size
            )
            
            all_results[model_key].append(results)
            
            # Print summary
            test_acc = results["test_acc"]
            final_val = results["val_accs"][-1]
            ep20_val = results["checkpoint_accs"].get(20, 0.0)
            
            print(f'  ✓ Test acc: {test_acc:.4f} | Final val: {final_val:.4f} | Ep20 val: {ep20_val:.4f}')
        
        # After training all five models for this seed, print comparison
        print(f'\n--- Seed {seed_idx+1} Summary ---')
        a_acc = all_results['standard_X_scaled'][-1]['test_acc']
        b_acc = all_results['standard_V_scaled'][-1]['test_acc']
        c_acc = all_results['rownorm_V'][-1]['test_acc']
        bp_acc = all_results['standard_V_unscaled'][-1]['test_acc']
        bpp_acc = all_results['standard_V_weighted'][-1]['test_acc']
        
        print(f'  (a) X+scaled+Standard:     {a_acc:.4f}')
        print(f'  (b) V+scaled+Standard:     {b_acc:.4f}  [Dir B: {(b_acc-a_acc)*100:+.2f}%]')
        print(f"  (b') V+unscaled+Standard:  {bp_acc:.4f}  [vs b: {(bp_acc-b_acc)*100:+.2f}%]")
        print(f"  (b'') V_weighted+Standard: {bpp_acc:.4f}  [vs b: {(bpp_acc-b_acc)*100:+.2f}%]")
        print(f'  (c) V+RowNorm:             {c_acc:.4f}  [Dir A: {(c_acc-b_acc)*100:+.2f}%]')

# ============================================================================
# 8. Aggregate Results and Generate Outputs
# ============================================================================
print(f'\n[7/8] Aggregating results and generating outputs...')

aggregated_results = {}
for model_name, model_key, _, _, _ in model_configs:
    print(f'Aggregating: {model_name}...')
    aggregated_results[model_key] = aggregate_results(all_results[model_key])

# Helper function to convert numpy types to native Python types
def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Save aggregated metrics
metrics = {
    'dataset': DATASET_NAME,
    'split_type': split_type,
    'num_seeds': NUM_SEEDS,
    'num_splits': num_split_iterations,
    'total_runs_per_model': NUM_SEEDS * num_split_iterations,
    'd_raw': int(d_raw),
    'd_effective': int(d_effective),
    'rank_deficiency': bool(rank_X < d_raw),
    'orthonormality_error': float(orthonormality_error),
    'epochs': int(EPOCHS),
    'hidden_dim': int(HIDDEN_DIM),
    'batch_size': int(batch_size),
    'learning_rate': float(LEARNING_RATE),
    'weight_decay': float(WEIGHT_DECAY),
    'models': {},
    'pca_analysis': convert_to_serializable(pca_results)
}

for model_name, model_key, _, _, _ in model_configs:
    metrics['models'][model_key] = {
        'name': model_name,
        'test_accuracy': convert_to_serializable(aggregated_results[model_key]['test_acc']),
        'convergence_metrics': convert_to_serializable(aggregated_results[model_key]['convergence_metrics']),
        'checkpoint_accuracies': convert_to_serializable(aggregated_results[model_key]['checkpoint_accs'])
    }

metrics_path = f'{output_base}/metrics/results_aggregated.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\n✓ Saved aggregated metrics: {metrics_path}')

# ============================================================================
# Print Final Summary (MATCHING Investigation 2 style)
# ============================================================================
print('\n' + '='*70)
print('FINAL SUMMARY - EXTENDED ABLATION STUDIES')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Split type: {split_type}')
print(f'Raw feature dimension: {d_raw}')
print(f'Effective dimension (after QR): {d_effective}')
if rank_X < d_raw:
    print(f'Rank deficiency handled: {d_raw} → {d_effective}')
print(f'span(U) = span(X) — same subspace, different basis')
print(f'D-orthonormality: max |U^T D U - I| = {orthonormality_error:.2e}')
print(f'Total runs per model: {NUM_SEEDS * num_split_iterations}')

print('\n' + '='*70)
print('TEST ACCURACY RESULTS (Mean ± Std)')
print('='*70)

a_test = aggregated_results['standard_X_scaled']['test_acc']
b_test = aggregated_results['standard_V_scaled']['test_acc']
c_test = aggregated_results['rownorm_V']['test_acc']
bp_test = aggregated_results['standard_V_unscaled']['test_acc']
bpp_test = aggregated_results['standard_V_weighted']['test_acc']

print(f"(a) Standard MLP on X (scaled):            {a_test['mean']*100:5.2f}±{a_test['std']*100:4.2f}%")
print(f"(b) Standard MLP on V (scaled):            {b_test['mean']*100:5.2f}±{b_test['std']*100:4.2f}%")
print(f"(b') Standard MLP on V (unscaled):         {bp_test['mean']*100:5.2f}±{bp_test['std']*100:4.2f}%")
print(f"(b'') Standard MLP on V_weighted (scaled): {bpp_test['mean']*100:5.2f}±{bpp_test['std']*100:4.2f}%")
print(f"(c) RowNorm MLP on V (unscaled):           {c_test['mean']*100:5.2f}±{c_test['std']*100:4.2f}%")

print('\n' + '='*70)
print('DIRECTION B: BASIS SENSITIVITY (a vs b)')
print('='*70)
print('Same model (Standard MLP), same preprocessing (StandardScaler)')
print('Only difference: coordinate system (X vs V)')

basis_diff = b_test['mean'] - a_test['mean']
basis_pct = (basis_diff / a_test['mean']) * 100 if a_test['mean'] > 0 else 0

print(f"\nBasis Effect: {basis_diff:+.4f} ({basis_pct:+.1f}%)")
if abs(basis_pct) > 2:
    print('✓ SIGNIFICANT basis sensitivity detected')
    print('  → MLPs are sensitive to basis representation!')
else:
    print('○ Minimal basis sensitivity for this dataset')

print('\n' + '='*70)
print('EXPERIMENT 1: STANDARDSCALER EFFECT (b vs b\')')
print('='*70)
print('Same basis (V), same model (Standard MLP)')
print('Only difference: StandardScaler vs no scaling')

scaling_diff = bp_test['mean'] - b_test['mean']
scaling_pct = (scaling_diff / b_test['mean']) * 100 if b_test['mean'] > 0 else 0

print(f"\nScaling Effect: {scaling_diff:+.4f} ({scaling_pct:+.1f}%)")
if scaling_pct > 1:
    print('✓ Unscaled V performs BETTER')
    print('  → StandardScaler may destroy D-orthonormality')
elif abs(scaling_pct) < 0.5:
    print('○ Minimal scaling effect')
    print('  → Basis change itself is the issue')
else:
    print('✗ Unscaled V performs WORSE')
    print('  → StandardScaler helps despite breaking geometry')

print('\n' + '='*70)
print('EXPERIMENT 2: EIGENVALUE WEIGHTING EFFECT (b vs b\'\')')
print('='*70)
print('Same model (Standard MLP), same preprocessing (StandardScaler)')
print('Only difference: U vs U_weighted')

eigenval_diff = bpp_test['mean'] - b_test['mean']
eigenval_pct = (eigenval_diff / b_test['mean']) * 100 if b_test['mean'] > 0 else 0

print(f"\nEigenvalue Effect: {eigenval_diff:+.4f} ({eigenval_pct:+.1f}%)")
if eigenval_pct > 1:
    print('✓ Eigenvalue weighting HELPS')
    print('  → Restricted eigenvalues carry information')
elif abs(eigenval_pct) < 0.5:
    print('○ Minimal eigenvalue effect')
    print('  → All info is in basis, not scaling')
else:
    print('✗ Eigenvalue weighting HURTS')
    print('  → Restricted eigenvalues are noise')

print('\n' + '='*70)
print('DIRECTION A: MODEL SENSITIVITY (b vs c)')
print('='*70)
print('Same basis (V), different models')
print('Question: Does RowNorm help on restricted eigenvectors?')

model_diff = c_test['mean'] - b_test['mean']
model_pct = (model_diff / b_test['mean']) * 100 if b_test['mean'] > 0 else 0

print(f"\nModel Effect: {model_diff:+.4f} ({model_pct:+.1f}%)")
if model_pct > 1:
    print('✓ RowNorm advantage persists on restricted eigenvectors')
elif abs(model_pct) < 1:
    print('○ Minimal model difference')
else:
    print('✗ RowNorm does NOT help on restricted eigenvectors')

print('\n' + '='*70)
print('EXPERIMENT 3: PCA GEOMETRIC ANALYSIS')
print('='*70)
print(f"Effective Rank: X={results_X['effective_rank']:.1f}, U={results_U['effective_rank']:.1f}")
print(f"Condition Number: X={results_X['condition_number']:.2e}, U={results_U['condition_number']:.2e}")
print(f"Isotropy (CV): X={results_X['coefficient_variation']:.3f}, U={results_U['coefficient_variation']:.3f}")

if pca_results['comparison']['condition_number_ratio'] < 0.5:
    print('\n✓ U is better conditioned than X')
    print('  → Should be easier to optimize (paradox if performs worse!)')
elif pca_results['comparison']['condition_number_ratio'] > 2.0:
    print('\n✗ U is worse conditioned than X')
    print('  → May explain performance degradation')
else:
    print('\n○ Similar conditioning')

if pca_results['comparison']['isotropy_ratio'] < 0.7:
    print('✓ U is more isotropic than X')
    print('  → More uniform variance distribution')
elif pca_results['comparison']['isotropy_ratio'] > 1.4:
    print('✗ U is less isotropic than X')
    print('  → More skewed variance distribution')

print('\n' + '='*70)
print('OVERALL INTERPRETATION')
print('='*70)

# Determine dominant effect
effects = {
    'Basis change (a→b)': abs(basis_pct),
    'StandardScaler (b→b\')': abs(scaling_pct),
    'Eigenvalue weighting (b→b\'\')': abs(eigenval_pct),
    'Model change (b→c)': abs(model_pct)
}
dominant = max(effects, key=effects.get)
print(f'Dominant effect: {dominant} ({effects[dominant]:.1f}%)')

if abs(scaling_pct) > 1:
    print('\n🔍 StandardScaler appears to affect performance')
    if scaling_pct > 0:
        print('   → Consider using V WITHOUT standardization')
    else:
        print('   → StandardScaler is helping despite breaking geometry')

if abs(eigenval_pct) > 1:
    print('\n🔍 Eigenvalue weighting affects performance')
    if eigenval_pct > 0:
        print('   → Restricted eigenvalues carry learnable structure')
    else:
        print('   → Eigenvalue weighting may amplify noise')

print('\n' + '='*70)

# ============================================================================
# 9. Create Visualization
# ============================================================================
print(f'\n[8/8] Creating visualization...')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Extended Ablation Studies - {DATASET_NAME} ({split_type})', fontsize=14, fontweight='bold')

# Plot 1: Test Accuracy Comparison
ax = axes[0, 0]
names_short = ['(a)\nX', '(b)\nV', "(b')\nV\nunscaled", "(b'')\nV\nweighted", '(c)\nV\nRowNorm']
means = [
    aggregated_results['standard_X_scaled']['test_acc']['mean'],
    aggregated_results['standard_V_scaled']['test_acc']['mean'],
    aggregated_results['standard_V_unscaled']['test_acc']['mean'],
    aggregated_results['standard_V_weighted']['test_acc']['mean'],
    aggregated_results['rownorm_V']['test_acc']['mean']
]
stds = [
    aggregated_results['standard_X_scaled']['test_acc']['std'],
    aggregated_results['standard_V_scaled']['test_acc']['std'],
    aggregated_results['standard_V_unscaled']['test_acc']['std'],
    aggregated_results['standard_V_weighted']['test_acc']['std'],
    aggregated_results['rownorm_V']['test_acc']['std']
]

x = np.arange(len(names_short))
bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
              color=['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c'])
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=9)
ax.grid(axis='y', alpha=0.3)

for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.005,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Effect Decomposition
ax = axes[0, 1]
effect_names = ['Dir B\n(a→b)', 'Scaling\n(b→b\')', 'Eigenval\n(b→b\'\')', 'Dir A\n(b→c)']
effects = [
    basis_diff * 100,
    scaling_diff * 100,
    eigenval_diff * 100,
    model_diff * 100
]
colors = ['blue' if e < 0 else 'green' for e in effects]
bars = ax.bar(range(len(effect_names)), effects, color=colors, alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Effect Size (pp)')
ax.set_title('Effect Size Decomposition')
ax.set_xticks(range(len(effect_names)))
ax.set_xticklabels(effect_names, fontsize=9)
ax.grid(axis='y', alpha=0.3)

for bar, eff in zip(bars, effects):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.2 if eff > 0 else -0.2),
            f'{eff:+.1f}', ha='center',
            va='bottom' if eff > 0 else 'top', fontsize=9)

# Plot 3: PCA Eigenvalue Spectrum
ax = axes[0, 2]
k_plot = min(50, len(eigvals_X))
ax.semilogy(range(1, k_plot+1), eigvals_X[:k_plot], 'o-', label='X', alpha=0.7)
ax.semilogy(range(1, k_plot+1), eigvals_U[:k_plot], 's-', label='U', alpha=0.7)
ax.set_xlabel('Component Index')
ax.set_ylabel('Eigenvalue (log scale)')
ax.set_title('PCA Eigenvalue Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Cumulative Variance
ax = axes[1, 0]
cumsum_X = results_X['eigenvalue_cumsum']
cumsum_U = results_U['eigenvalue_cumsum']
ax.plot(cumsum_X[:k_plot], label='X', linewidth=2)
ax.plot(cumsum_U[:k_plot], label='U', linewidth=2)
ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance')
ax.set_title('Cumulative Variance Explained')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Geometric Properties
ax = axes[1, 1]
properties = ['Eff. Rank\nRatio', 'log10(Cond.\nNumber)', 'Isotropy\n(CV)']
x_vals = [
    results_X['effective_rank_ratio'],
    np.log10(results_X['condition_number']),
    results_X['coefficient_variation']
]
u_vals = [
    results_U['effective_rank_ratio'],
    np.log10(results_U['condition_number']),
    results_U['coefficient_variation']
]

x_pos = np.arange(len(properties))
width = 0.35
ax.bar(x_pos - width/2, x_vals, width, label='X', alpha=0.7)
ax.bar(x_pos + width/2, u_vals, width, label='U', alpha=0.7)
ax.set_ylabel('Value')
ax.set_title('Geometric Property Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(properties, fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 6: Validation Curves (first seed)
ax = axes[1, 2]
for i, (name_short, model_key) in enumerate(zip(names_short, [
    'standard_X_scaled', 'standard_V_scaled', 'standard_V_unscaled',
    'standard_V_weighted', 'rownorm_V'
])):
    val_accs = all_results[model_key][0]['val_accs']
    ax.plot(val_accs, label=name_short, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title(f'Training Curves (Seed 0)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = f'{output_base}/plots/extended_ablations.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f'✓ Visualization saved: {plot_file}')

print('\n' + '='*70)
print('EXTENDED ABLATION STUDY COMPLETE!')
print('='*70)
print(f'Results saved to: {output_base}/')
print(f'  - metrics/results_aggregated.json')
print(f'  - plots/extended_ablations.png')