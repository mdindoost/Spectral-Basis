"""
Investigation 2: Directions A & B - Complete Basis Sensitivity Study
Extended to support larger datasets
====================================================================

Three experiments to separate basis effects from model effects:
  (a) X → StandardScaler → Standard MLP        [Baseline]
  (b) V → StandardScaler → Standard MLP        [NEW - Isolates basis effect]
  (c) V → RowNorm MLP (no scaling)            [Tests model on restricted basis]

Direction A (b vs c): Does RowNorm help with restricted eigenvectors?
Direction B (a vs b): Does basis choice matter with identical preprocessing?

Supported Datasets:
- OGB: ogbn-arxiv
- Planetoid: cora, citeseer, pubmed
- WikiCS: wikics
- Amazon: amazon-photo, amazon-computers
- Coauthor: coauthor-cs, coauthor-physics

Usage:
    # Fixed benchmark splits (default)
    python experiments/investigation2_directions_AB.py [dataset_name]
    
    # Random 60/20/20 splits
    python experiments/investigation2_directions_AB.py [dataset_name] --random-splits
    
Examples:
    python experiments/investigation2_directions_AB.py ogbn-arxiv
    python experiments/investigation2_directions_AB.py wikics
    python experiments/investigation2_directions_AB.py coauthor-physics --random-splits
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

# EXTENDED: Now supports 9 datasets!
VALID_DATASETS = [
    'ogbn-arxiv',           # OGB
    'cora',                 # Planetoid
    'citeseer',             # Planetoid
    'pubmed',               # Planetoid
    'wikics',               # WikiCS
    'amazon-photo',         # Amazon
    'amazon-computers',     # Amazon
    'coauthor-cs',          # Coauthor
    'coauthor-physics'      # Coauthor
]

if DATASET_NAME not in VALID_DATASETS:
    print(f"Error: Invalid dataset '{DATASET_NAME}'")
    print(f"Valid datasets: {', '.join(VALID_DATASETS)}")
    sys.exit(1)

# Experimental parameters
NUM_SEEDS = 5
NUM_RANDOM_SPLITS = 5 if USE_RANDOM_SPLITS else 1

# Hyperparameters
HIDDEN_DIM = 256
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*70)
print(f'INVESTIGATION 2: DIRECTIONS A & B - {DATASET_NAME.upper()}')
print('='*70)
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'Random splits: {"Yes" if USE_RANDOM_SPLITS else "No (fixed benchmark)"}')
print('='*70)

# Create output directories
split_type = 'random_splits' if USE_RANDOM_SPLITS else 'fixed_splits'
output_base = f'results/investigation2_directions_AB/{DATASET_NAME}/{split_type}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print(f'\n[1/6] Loading {DATASET_NAME}...')

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
print(f'\n[2/6] Building graph matrices...')

adj, L, D = build_graph_matrices(edge_index, num_nodes)

print(f'Adjacency: {adj.shape}, nnz={adj.nnz:,}')
print(f'Laplacian: {L.shape}')
print(f'Degree matrix: {D.shape}')

# ============================================================================
# 3. Compute X-Restricted Eigenvectors using QR
# ============================================================================
print(f'\n[3/6] Computing X-restricted eigenvectors...')

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
# Helper Functions for Training with Granular Tracking
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
    checkpoint_accs = {}  # Store accuracy at specific epochs
    batch_level_tracking = []  # Batch-level validation for first 5 epochs
    
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
    
    # AUC (area under validation curve, normalized)
    auc = np.trapz(val_accs, dx=1) / len(val_accs)
    convergence_metrics['auc'] = auc / final_val_acc if final_val_acc > 0 else 0.0
    
    # Convergence rate (average improvement in first 20 epochs)
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
    
    # Batch-level tracking (aggregate if available)
    batch_tracking_aggregated = {}
    if results_list[0]['batch_level_tracking']:
        max_steps = max([len(r['batch_level_tracking']) for r in results_list])
        
        for step in range(max_steps):
            step_accs = []
            for r in results_list:
                if step < len(r['batch_level_tracking']):
                    step_accs.append(r['batch_level_tracking'][step]['val_acc'])
            
            if step_accs and step < len(results_list[0]['batch_level_tracking']):
                batch_tracking_aggregated[step] = {
                    'global_step': results_list[0]['batch_level_tracking'][step]['global_step'],
                    'mean_val_acc': float(np.mean(step_accs)),
                    'std_val_acc': float(np.std(step_accs))
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
        'convergence_metrics': conv_metrics,
        'batch_level_tracking': batch_tracking_aggregated
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
# 4. Prepare Data Splits
# ============================================================================
print(f'\n[4/6] Preparing data splits...')

num_split_iterations = NUM_RANDOM_SPLITS if USE_RANDOM_SPLITS else 1

# ============================================================================
# 5. Train All Models with Multiple Seeds and Splits
# ============================================================================
print(f'\n[5/6] Training 3 models (Directions A & B)...')
if USE_RANDOM_SPLITS:
    print(f'Across {NUM_RANDOM_SPLITS} random splits × {NUM_SEEDS} seeds')
    print(f'Total runs: {3 * NUM_SEEDS * NUM_RANDOM_SPLITS} = 3(models) × {NUM_SEEDS}(seeds) × {NUM_RANDOM_SPLITS}(splits)')
else:
    print(f'Total runs: {3 * NUM_SEEDS} = 3(models) × {NUM_SEEDS}(seeds)')

model_configs = [
    ('(a) Standard MLP on X (scaled)', 'standard_X_scaled', 'X', True),
    ('(b) Standard MLP on V (scaled)', 'standard_V_scaled', 'V', True),
    ('(c) RowNorm MLP on V (unscaled)', 'rownorm_V', 'V', False),
]

all_results = {key: [] for _, key, _, _ in model_configs}

# Determine batch size based on training set size
if len(train_idx_fixed) > 256:
    batch_size = 128
    print(f'Batch size: {batch_size} (dataset has {len(train_idx_fixed)} training samples)')
else:
    batch_size = len(train_idx_fixed)
    print(f'Batch size: {batch_size} (full batch for small dataset)')

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
        for model_name, model_key, feature_type, use_scaling in model_configs:
            print(f'\nTraining: {model_name}...')
            
            # Prepare features based on experiment type
            if feature_type == 'X' and use_scaling:
                # (a) X with StandardScaler
                scaler = StandardScaler().fit(X_raw[train_idx])
                X_train = torch.from_numpy(scaler.transform(X_raw[train_idx])).float().to(device)
                X_val = torch.from_numpy(scaler.transform(X_raw[val_idx])).float().to(device)
                X_test = torch.from_numpy(scaler.transform(X_raw[test_idx])).float().to(device)
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
                model = StandardMLP(d_raw, HIDDEN_DIM, num_classes)
                
                results = train_with_granular_tracking(
                    model, train_loader, X_val, y_val, X_test, y_test,
                    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    device=device, batch_size=batch_size
                )
                
            elif feature_type == 'V' and use_scaling:
                # (b) V with StandardScaler [KEY EXPERIMENT for Direction B!]
                print('  🔍 This is experiment (b) - isolates pure basis effect!')
                scaler = StandardScaler().fit(U[train_idx])
                V_train = torch.from_numpy(scaler.transform(U[train_idx])).float().to(device)
                V_val = torch.from_numpy(scaler.transform(U[val_idx])).float().to(device)
                V_test = torch.from_numpy(scaler.transform(U[test_idx])).float().to(device)
                train_loader = DataLoader(TensorDataset(V_train, y_train), batch_size=batch_size, shuffle=True)
                model = StandardMLP(d_effective, HIDDEN_DIM, num_classes)
                
                results = train_with_granular_tracking(
                    model, train_loader, V_val, y_val, V_test, y_test,
                    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    device=device, batch_size=batch_size
                )
                
            else:  # feature_type == 'V' and not use_scaling
                # (c) V with RowNorm MLP
                V_train = torch.from_numpy(U[train_idx]).float().to(device)
                V_val = torch.from_numpy(U[val_idx]).float().to(device)
                V_test = torch.from_numpy(U[test_idx]).float().to(device)
                train_loader = DataLoader(TensorDataset(V_train, y_train), batch_size=batch_size, shuffle=True)
                model = RowNormMLP(d_effective, HIDDEN_DIM, num_classes)
                
                results = train_with_granular_tracking(
                    model, train_loader, V_val, y_val, V_test, y_test,
                    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    device=device, batch_size=batch_size
                )
            
            all_results[model_key].append(results)
            
            # Print summary for this run
            test_acc = results["test_acc"]
            final_val = results["val_accs"][-1]
            ep20_val = results["checkpoint_accs"].get(20, 0.0)
            
            print(f'  ✓ Test acc: {test_acc:.4f} | Final val: {final_val:.4f} | Ep20 val: {ep20_val:.4f}')
        
        # After training all three models for this seed, print comparison
        print(f'\n--- Seed {seed_idx+1} Summary ---')
        a_acc = all_results['standard_X_scaled'][-1]['test_acc']
        b_acc = all_results['standard_V_scaled'][-1]['test_acc']
        c_acc = all_results['rownorm_V'][-1]['test_acc']
        
        print(f'  (a) Standard X: {a_acc:.4f}')
        print(f'  (b) Standard V: {b_acc:.4f}  [Diff from (a): {(b_acc-a_acc)*100:+.2f}%]')
        print(f'  (c) RowNorm V:  {c_acc:.4f}  [Diff from (b): {(c_acc-b_acc)*100:+.2f}%]')
        
        # Quick interpretation
        basis_effect = abs((b_acc-a_acc)/a_acc*100) if a_acc > 0 else 0
        if abs(b_acc - a_acc) > 0.01:
            print(f'  → Basis effect (a vs b): {basis_effect:.1f}% difference')
        else:
            print(f'  → Basis effect minimal: {basis_effect:.1f}% difference')
        
        if c_acc > b_acc:
            print(f'  → RowNorm helps: +{(c_acc-b_acc)/b_acc*100:.1f}%')
        else:
            print(f'  → RowNorm hurts: {(c_acc-b_acc)/b_acc*100:.1f}%')

# ============================================================================
# 6. Aggregate Results and Generate Outputs
# ============================================================================
print(f'\n[6/6] Aggregating results and generating outputs...')

aggregated_results = {}
for model_name, model_key, _, _ in model_configs:
    print(f'Aggregating: {model_name}...')
    aggregated_results[model_key] = aggregate_results(all_results[model_key])

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
    'models': {}
}

for model_name, model_key, _, _ in model_configs:
    metrics['models'][model_key] = {
        'name': model_name,
        'test_accuracy': aggregated_results[model_key]['test_acc'],
        'convergence_metrics': aggregated_results[model_key]['convergence_metrics'],
        'checkpoint_accuracies': aggregated_results[model_key]['checkpoint_accs']
    }

metrics_path = f'{output_base}/metrics/results_aggregated.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\n✓ Saved aggregated metrics: {metrics_path}')

# ============================================================================
# Print Final Summary
# ============================================================================
print('\n' + '='*70)
print('FINAL SUMMARY')
print('='*70)
print(f'Dataset: {DATASET_NAME}')
print(f'Split type: {split_type}')
print(f'Raw feature dimension: {d_raw}')
print(f'Effective dimension (after QR): {d_effective}')
if rank_X < d_raw:
    print(f'Rank deficiency handled: {d_raw} → {d_effective}')
print(f'span(U) = span(X) — same subspace, different basis')
print(f'D-orthonormality: max |U^T D U - I| = {orthonormality_error:.2e}')
print(f'Hidden dimension: {HIDDEN_DIM}')
print(f'Epochs: {EPOCHS}')
print(f'Total runs per model: {NUM_SEEDS * num_split_iterations}')

print('\n' + '='*70)
print('TEST ACCURACY RESULTS (Mean ± Std)')
print('='*70)

a_test = aggregated_results['standard_X_scaled']['test_acc']
b_test = aggregated_results['standard_V_scaled']['test_acc']
c_test = aggregated_results['rownorm_V']['test_acc']

print(f"(a) Standard MLP on X (scaled):  {a_test['mean']*100:5.2f}±{a_test['std']*100:4.2f}%")
print(f"(b) Standard MLP on V (scaled):  {b_test['mean']*100:5.2f}±{b_test['std']*100:4.2f}%")
print(f"(c) RowNorm MLP on V (unscaled): {c_test['mean']*100:5.2f}±{c_test['std']*100:4.2f}%")

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
print('EFFECT INDEPENDENCE')
print('='*70)

combined_diff = c_test['mean'] - a_test['mean']
combined_pct = (combined_diff / a_test['mean']) * 100 if a_test['mean'] > 0 else 0
expected_combined_pct = basis_pct + model_pct
interaction = combined_pct - expected_combined_pct

print(f"Basis effect (a→b):       {basis_pct:+6.2f}%")
print(f"Model effect (b→c):       {model_pct:+6.2f}%")
print(f"Expected combined:        {expected_combined_pct:+6.2f}%")
print(f"Observed combined (a→c):  {combined_pct:+6.2f}%")
print(f"Interaction term:         {interaction:+6.2f}%")

if abs(interaction) < 1:
    print('\n✓ Effects are ADDITIVE (independent)')
else:
    print(f'\n⚠ Effects show INTERACTION ({abs(interaction):.1f}%)')

print('\n' + '='*70)
print(f'✓ Investigation 2 Directions A&B complete for {DATASET_NAME}!')
print(f'✓ Results saved to: {output_base}/')
print('='*70)