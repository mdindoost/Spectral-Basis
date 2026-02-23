"""
===================================================================================
MASTER TRAINING EXPERIMENT
===================================================================================

Unified experiment covering:
  Exp 1:  Part A/B sweep across k values
          (SGC+MLP, Restricted+StandardMLP, Restricted+RowNorm)
  Exp 3:  Fisher diagnostic at ALL k values (training-set-only, no leakage)
          Also computes fisher_score_X_diffused and per-feature stats
  Exp 4:  Log-Magnitude and Dual-Stream at k=10
  Exp 5:  SpectralRowNorm alpha sweep at k=10 (α ∈ {-1.0, -0.5, 0.0, 0.5, 1.0})
  Exp 6:  NestedSpheres 5×5 (α × β) sweep at k=10
  Exp 7:  Optimization dynamics at k=10 for all 9 datasets
          All 3 core methods tracked: SGC+MLP, Restricted+StandardMLP, Restricted+RowNorm
          Logs: val_acc, train_acc, train_loss, val_loss per epoch + convergence metrics

Models imported from utils.py (src/utils.py):
  SGC, StandardMLP, RowNormMLP, LogMagnitudeMLP,
  DualStreamMLP, SpectralRowNormMLP, NestedSpheresClassifier

Graph/diffusion functions imported from utils.py:
  build_graph_matrices, get_largest_connected_component_nx, extract_subgraph,
  compute_sgc_normalized_adjacency, sgc_precompute, compute_restricted_eigenvectors

Training protocol (unified, canonical new paper numbers):
  All methods: 500 epochs, patience=100 early stopping on val_acc, eval every epoch

Graph convention (exact match to investigation3/4/partAB):
  adj, L, D = build_graph_matrices(...)
  # L holds Laplacian, D holds degree matrix (correct order matching src/utils.py)
  compute_restricted_eigenvectors(X_diffused, L, D, num_components)

k sweep: [1, 2, 4, 6, 8, 10, 12, 20, 30] (default)
Random splits: 5 splits × num_seeds seeds (matching investigation3)
Fixed splits:  1 split  × num_seeds seeds

Output:
  PAPER_RESULTS/{dataset}_{split}_{component}/k{k}/metrics/results.json
  PAPER_RESULTS/{dataset}_{split}_{component}/k{k}/diagnostics/fisher_diagnostic.json
      (all k values)
  PAPER_RESULTS/{dataset}_{split}_{component}/k{k}/training_curves/split{i}_dynamics.json
      (k=10 only, all datasets)

Author: Mohammad
===================================================================================

Usage:
  cd /home/md724/Spectral-Basis
  PYTHON=/home/md724/Spectral-Basis/venv/bin/python

  # Single dataset, full k sweep, fixed split (canonical paper run)
  $PYTHON PAPER_EXPERIMENTS/master_training.py cora

  # Single dataset, random splits
  $PYTHON PAPER_EXPERIMENTS/master_training.py cora --splits random

  # Single dataset, specific k values only
  $PYTHON PAPER_EXPERIMENTS/master_training.py citeseer --k_values 10

  # Run all 9 datasets (fixed + random) — full paper reproduction
  for ds in amazon-computers amazon-photo citeseer coauthor-cs coauthor-physics \
             cora ogbn-arxiv pubmed wikics; do
      $PYTHON PAPER_EXPERIMENTS/master_training.py $ds --splits fixed
      $PYTHON PAPER_EXPERIMENTS/master_training.py $ds --splits random
  done

  # Quick smoke test (1 seed, k=10 only)
  $PYTHON PAPER_EXPERIMENTS/master_training.py citeseer --k_values 10 --num_seeds 1

Arguments:
  dataset           Dataset name (positional, required)
  --k_values        Diffusion steps to run (default: 1 2 4 6 8 10 12 20 30)
  --splits          fixed | random (default: fixed)
  --component       lcc | whole (default: lcc)
  --num_seeds       Training seeds per config (default: 15)
  --device          CUDA device string (default: cuda). Raises error if CUDA unavailable.
===================================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import all utilities from src/utils.py
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _SRC_DIR)
from utils import (
    # Dataset
    load_dataset,
    # Graph construction and preprocessing
    build_graph_matrices, get_largest_connected_component_nx, extract_subgraph,
    compute_sgc_normalized_adjacency, sgc_precompute, compute_restricted_eigenvectors,
    # Models
    SGC, StandardMLP, RowNormMLP, LogMagnitudeMLP,
    DualStreamMLP, SpectralRowNormMLP, NestedSpheresClassifier,
)

# ============================================================================
# Training Function (unified protocol for all methods)
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                device, epochs=500, lr=0.01, weight_decay=5e-4, patience=100,
                track_dynamics=False):
    """Unified training protocol for all methods (canonical new paper numbers).

    Protocol:
      - Adam optimizer, lr=0.01, wd=5e-4
      - Full-batch training
      - Evaluate val and test every epoch
      - Early stopping: patience=100 epochs without val_acc improvement
      - Model selection: checkpoint test_acc at epoch with highest val_acc

    Returns:
      (best_val_acc, best_test_acc, best_train_acc, train_time, dynamics_or_None)

      best_train_acc: train accuracy at the best-val-acc epoch (eval mode, no dropout)

      dynamics_or_None: None if track_dynamics=False, else a dict with:
        val_acc, train_acc, train_loss, val_loss  — lists, length = actual epochs run
        speed_to_90, speed_to_95, speed_to_99    — epochs to reach % of best val_acc
        auc_normalized                            — area under val curve / (n_ep * max_acc)
        convergence_rate                          — mean improvement in first 20 epochs
        checkpoint_val_accs                       — {10, 20, 40, 80, 160, 200: float or None}
    """
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.LongTensor(y_val).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    y_te = torch.LongTensor(y_test).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc     = 0.0
    best_test_acc    = 0.0
    best_train_acc   = 0.0
    patience_counter = 0
    tr_eval_acc      = 0.0  # updated each epoch when track_dynamics

    if track_dynamics:
        dyn = {'val_acc': [], 'train_acc': [], 'train_loss': [], 'val_loss': []}
    else:
        dyn = None

    start = time.time()

    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits_tr = model(X_tr)
        tr_loss   = criterion(logits_tr, y_tr)
        tr_loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_logits  = model(X_va)
            val_loss_v  = criterion(val_logits, y_va).item()
            val_acc     = (val_logits.argmax(1) == y_va).float().mean().item()
            test_acc    = (model(X_te).argmax(1) == y_te).float().mean().item()
            if track_dynamics:
                tr_eval_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()

        if track_dynamics:
            dyn['val_acc'].append(float(val_acc))
            dyn['train_acc'].append(float(tr_eval_acc))
            dyn['train_loss'].append(float(tr_loss.item()))
            dyn['val_loss'].append(float(val_loss_v))

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            if track_dynamics:
                best_train_acc = float(tr_eval_acc)
            else:
                # Compute train acc at best val epoch (eval mode — no dropout)
                with torch.no_grad():
                    best_train_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Convergence metrics (only when dynamics tracked)
    if track_dynamics and dyn['val_acc']:
        n_ep    = len(dyn['val_acc'])
        max_acc = max(dyn['val_acc'])

        # NOTE: speed_to_XX measures epochs to reach XX% of THIS MODEL'S OWN peak accuracy.
        # These are NOT absolute thresholds and CANNOT be compared across methods with
        # different peak accuracies (e.g., RowNorm peak=85% vs Std peak=60% gives different
        # targets for speed_to_90). Use checkpoint_val_accs for cross-method comparisons.
        def speed_to(pct):
            target = pct * max_acc
            for i, v in enumerate(dyn['val_acc']):
                if v >= target:
                    return i + 1
            return None

        dyn['speed_to_90_pct_of_peak'] = speed_to(0.90)
        dyn['speed_to_95_pct_of_peak'] = speed_to(0.95)
        dyn['speed_to_99_pct_of_peak'] = speed_to(0.99)
        dyn['auc_normalized'] = (float(sum(dyn['val_acc'])) / (n_ep * max_acc)
                                  if max_acc > 0 else 0.0)
        if n_ep > 1:
            diffs = [dyn['val_acc'][i + 1] - dyn['val_acc'][i]
                     for i in range(min(19, n_ep - 1))]
            dyn['convergence_rate'] = float(np.mean(diffs))
        else:
            dyn['convergence_rate'] = 0.0

        checkpoints = [10, 20, 40, 80, 160, 200]
        dyn['checkpoint_val_accs'] = {
            str(ep): float(dyn['val_acc'][ep - 1]) if ep <= n_ep else None
            for ep in checkpoints
        }

    train_time = time.time() - start
    return best_val_acc, best_test_acc, best_train_acc, train_time, dyn


# ============================================================================
# Fisher Diagnostic (training-set-only)
# ============================================================================

def compute_fisher_diagnostic(U_train, y_train, num_classes, X_diffused_train=None):
    """Fisher score for magnitude discriminability on TRAINING SET ONLY.

    Improvement over investigation3 which uses all nodes (leakage risk).
    Fisher = between_class_variance(||row||) / within_class_variance(||row||)

    Args:
        U_train:          restricted eigenvectors for training nodes (n_train × d_eff)
        y_train:          integer labels (n_train,)
        num_classes:      number of classes
        X_diffused_train: optional diffused features (n_train × d_raw)
                          if provided, also computes fisher_score_X_diffused

    Returns dict with:
        fisher_score / fisher_score_U, between/within class variances,
        per_feature_U_stats (mean/std/max/top5 of per-column Fisher scores),
        and optionally fisher_score_X_diffused + per_feature_X_stats.
    """
    def _fisher_on_magnitudes(M, y, nc):
        # Unweighted Fisher = between_class_variance / within_class_variance.
        # All classes receive equal weight regardless of size. This is standard for
        # balanced datasets. For imbalanced datasets (e.g. ogbn-arxiv, 40 classes with
        # unequal sizes), small classes are over-represented. Disclose in paper.
        # Both numerator and denominator must be in units of magnitude² (dimensionless ratio).
        class_means, class_vars = [], []
        for c in range(nc):
            mask = (y == c)
            if mask.sum() == 0:
                class_means.append(0.0); class_vars.append(0.0)
            else:
                M_c = M[mask]
                class_means.append(float(M_c.mean()))
                class_vars.append(float(M_c.var()))   # variance, not std
        cm = np.array(class_means); cv = np.array(class_vars)
        bv = float(cm.var())        # variance of class means (magnitude²)
        wv = float(cv.mean())       # mean of within-class variances (magnitude²)
        return (bv / wv if wv > 0 else 0.0), bv, wv

    # Row-magnitude Fisher on U
    M_U = np.linalg.norm(U_train, axis=1)
    fs_U, bv_U, wv_U = _fisher_on_magnitudes(M_U, y_train, num_classes)

    # Per-feature Fisher for U (individual column absolute values)
    per_feat_scores = []
    for j in range(U_train.shape[1]):
        col = np.abs(U_train[:, j])
        fs_j, _, _ = _fisher_on_magnitudes(col, y_train, num_classes)
        per_feat_scores.append(float(fs_j))
    pf_arr = np.array(per_feat_scores)

    result = {
        'fisher_score':           float(fs_U),   # kept for backward compat
        'fisher_score_U':         float(fs_U),
        'between_class_variance': float(bv_U),
        'within_class_variance':  float(wv_U),
        'per_feature_U_stats': {
            'mean': float(pf_arr.mean()),
            'std':  float(pf_arr.std()),
            'max':  float(pf_arr.max()),
            'top5': [float(v) for v in sorted(per_feat_scores, reverse=True)[:5]]
        },
        'note': 'Computed on training set only to prevent data leakage'
    }

    # Row-magnitude Fisher on X_diffused (for comparison with U)
    if X_diffused_train is not None:
        M_X = np.linalg.norm(X_diffused_train, axis=1)
        fs_X, bv_X, wv_X = _fisher_on_magnitudes(M_X, y_train, num_classes)

        d_raw = X_diffused_train.shape[1]
        sample_cols = (np.random.choice(d_raw, min(500, d_raw), replace=False)
                       if d_raw > 500 else np.arange(d_raw))
        per_feat_X = []
        for j in sample_cols:
            col = np.abs(X_diffused_train[:, j])
            fs_j, _, _ = _fisher_on_magnitudes(col, y_train, num_classes)
            per_feat_X.append(float(fs_j))
        pf_X_arr = np.array(per_feat_X)

        result['fisher_score_X_diffused']  = float(fs_X)
        result['between_class_variance_X'] = float(bv_X)
        result['within_class_variance_X']  = float(wv_X)
        result['per_feature_X_stats'] = {
            'mean':           float(pf_X_arr.mean()),
            'std':            float(pf_X_arr.std()),
            'max':            float(pf_X_arr.max()),
            'top5':           [float(v) for v in sorted(per_feat_X, reverse=True)[:5]],
            'n_cols_sampled': int(len(sample_cols))
        }

    return result


# ============================================================================
# Result Aggregation
# ============================================================================

def aggregate_results(results_list):
    """Aggregate list of per-run result dicts."""
    val_accs   = [r['val_acc']   for r in results_list]
    test_accs  = [r['test_acc']  for r in results_list]
    train_accs = [r['train_acc'] for r in results_list if 'train_acc' in r]

    out = {
        'test_acc_mean':    float(np.mean(test_accs)),
        'test_acc_std':     float(np.std(test_accs)),
        'val_acc_mean':     float(np.mean(val_accs)),
        'val_acc_std':      float(np.std(val_accs)),
        'per_run_test_acc': [float(v) for v in test_accs],
        'n_runs':           len(test_accs)
    }
    if train_accs:
        out['train_acc_mean'] = float(np.mean(train_accs))
        out['train_acc_std']  = float(np.std(train_accs))
    if results_list and 'num_params' in results_list[0]:
        out['num_params']           = results_list[0]['num_params']
        out['param_to_train_ratio'] = results_list[0]['param_to_train_ratio']
    return out


# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description='Master Training Experiment')
parser.add_argument('dataset', type=str,
                    help='Dataset name (e.g. citeseer, cora, ogbn-arxiv, ...)')
parser.add_argument('--k_values', type=int, nargs='+',
                    default=[1, 2, 4, 6, 8, 10, 12, 20, 30],
                    help='Diffusion steps to sweep (default: 1 2 4 6 8 10 12 20 30)')
parser.add_argument('--splits', type=str, choices=['fixed', 'random'], default='fixed',
                    help='Split type (default: fixed)')
parser.add_argument('--component', type=str, choices=['lcc', 'whole'], default='lcc',
                    help='Graph component (default: lcc)')
parser.add_argument('--num_seeds', type=int, default=15,
                    help='Training seeds per configuration (default: 15)')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use (default: cuda). Must be a valid CUDA device.')
args = parser.parse_args()

DATASET_NAME   = args.dataset
K_VALUES       = sorted(args.k_values)
SPLIT_TYPE     = args.splits
COMPONENT_TYPE = args.component
NUM_SEEDS      = args.num_seeds

if not torch.cuda.is_available() and args.device == 'cuda':
    raise RuntimeError(
        "CUDA is not available on this machine. "
        "Check your CUDA installation (nvidia-smi, torch.version.cuda, etc.) "
        "and ensure the correct PyTorch+CUDA build is installed."
    )
device = torch.device(args.device)

# Fixed hyperparameters
HIDDEN_DIM = 256
HIDDEN_MAG = 32
LR         = 0.01
WD         = 5e-4
EPOCHS     = 500    # unified canonical protocol for all methods
PATIENCE   = 100    # unified canonical protocol for all methods

# SpectralRowNorm alpha sweep — includes 0.0 (= standard RowNorm) as explicit entry
ALPHA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

# NestedSpheres 5×5 alpha × beta grid
NESTED_ALPHA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
NESTED_BETA_VALUES  = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Number of random splits (matches investigation3: 5 for random, 1 for fixed)
NUM_RANDOM_SPLITS = 5 if SPLIT_TYPE == 'random' else 1

# Datasets for which optimization dynamics are tracked at k=10 (Exp 7) — all 9
DYNAMICS_DATASETS = {
    'amazon-computers', 'amazon-photo', 'citeseer',
    'coauthor-cs', 'coauthor-physics', 'cora',
    'ogbn-arxiv', 'pubmed', 'wikics'
}

# Output directory
OUTPUT_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'PAPER_RESULTS',
    f'{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}'
)

print('=' * 80)
print('MASTER TRAINING EXPERIMENT')
print('=' * 80)
print(f'Dataset:      {DATASET_NAME}')
print(f'k values:     {K_VALUES}')
print(f'Split type:   {SPLIT_TYPE}  ({NUM_RANDOM_SPLITS} split(s) x {NUM_SEEDS} seeds)')
print(f'Component:    {COMPONENT_TYPE}')
print(f'Device:       {device}')
print(f'Output base:  {OUTPUT_BASE}')
print('=' * 80)

# ============================================================================
# Step 1: Load dataset
# ============================================================================

print('\n[Step 1] Loading dataset...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(DATASET_NAME, root='./dataset')
print(f'Nodes: {num_nodes:,}, Features: {X_raw.shape[1]}, Classes: {num_classes}')

# ============================================================================
# Step 2: Build initial graph matrices
# src/utils.py returns (adj, L, D) where L=Laplacian, D=degree.
# Assigned correctly: L=Laplacian, D=degree (swap bug fixed).
# ============================================================================

print('\n[Step 2] Building graph matrices...')
adj, L, D = build_graph_matrices(edge_index, num_nodes)

# ============================================================================
# Step 3: Extract LCC (if requested)
# ============================================================================

if COMPONENT_TYPE == 'lcc':
    print('\n[Step 3] Extracting largest connected component...')
    lcc_mask = get_largest_connected_component_nx(adj)

    split_idx_original = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig
    }
    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_original
    )

    print('Rebuilding graph matrices for LCC...')
    # adj from extract_subgraph already has self-loops (inherited from the first
    # build_graph_matrices call). Feeding adj.tocoo() directly into build_graph_matrices
    # would include the (i,i) edges and trigger a second sp.eye addition — double self-loops.
    # Fix: strip self-loops before extracting the edge list so build_graph_matrices
    # adds them exactly once.
    adj_no_loops   = adj - sp.diags(adj.diagonal())
    adj_coo        = adj_no_loops.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj, L, D      = build_graph_matrices(edge_index_lcc, adj.shape[0])
    num_components = 0   # LCC has exactly 1 connected component

else:
    print('\n[Step 3] Using whole graph...')
    split_idx = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig
    }
    G_tmp = nx.from_scipy_sparse_array(adj)
    num_components = len(list(nx.connected_components(G_tmp))) - 1

num_nodes   = X_raw.shape[0]
num_classes = len(np.unique(labels))
print(f'After step 3: nodes={num_nodes:,}, classes={num_classes}, '
      f'disconnected_components_to_skip={num_components}')

# ============================================================================
# Step 4: Dense features (needed for SGC diffusion)
# ============================================================================

features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()

# ============================================================================
# Step 5: Prepare split index lists
# ============================================================================

print('\n[Step 4] Preparing splits...')

if SPLIT_TYPE == 'fixed':
    splits_list = [(0, split_idx['train_idx'], split_idx['val_idx'], split_idx['test_idx'])]
    tr, va, te = splits_list[0][1:]
    print(f'Fixed split: Train={len(tr)}, Val={len(va)}, Test={len(te)}')
else:
    splits_list = []
    for s_idx in range(NUM_RANDOM_SPLITS):
        np.random.seed(s_idx)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        n_tr = int(0.6 * num_nodes)
        n_va = int(0.2 * num_nodes)
        tr = indices[:n_tr]
        va = indices[n_tr:n_tr + n_va]
        te = indices[n_tr + n_va:]
        splits_list.append((s_idx, tr, va, te))
        print(f'Random split {s_idx}: Train={len(tr)}, Val={len(va)}, Test={len(te)}')

# ============================================================================
# Step 6: SGC normalized adjacency (computed once, reused across all k)
# ============================================================================

print('\n[Step 5] Computing SGC normalized adjacency...')
A_sgc = compute_sgc_normalized_adjacency(adj)
print('A_sgc = D^{-1/2}(A+I)D^{-1/2}')

track_dynamics = (DATASET_NAME in DYNAMICS_DATASETS)

# ============================================================================
# Step 7: Main k-sweep loop
# ============================================================================

for k in K_VALUES:
    print(f'\n{"="*80}')
    print(f'K = {k}')
    print(f'{"="*80}')

    X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k)
    print(f'  X_diffused shape: {X_diffused.shape}')

    # Create output directories — diagnostics for ALL k (Fisher computed at every k)
    k_dir = os.path.join(OUTPUT_BASE, f'k{k}')
    os.makedirs(os.path.join(k_dir, 'metrics'),     exist_ok=True)
    os.makedirs(os.path.join(k_dir, 'diagnostics'), exist_ok=True)
    # os.makedirs(os.path.join(k_dir, 'matrices'),    exist_ok=True)  # disabled: no .npz caching
    if k == 10 and track_dynamics:
        os.makedirs(os.path.join(k_dir, 'training_curves'), exist_ok=True)

    # Result accumulators
    method_keys = ['sgc_mlp_baseline', 'restricted_standard_mlp', 'restricted_rownorm_mlp']
    if k == 10:
        method_keys += ['log_magnitude', 'dual_stream']
        for alpha in ALPHA_VALUES:
            method_keys.append(f'spectral_rownorm_alpha{alpha}')
        for alpha in NESTED_ALPHA_VALUES:
            for beta in NESTED_BETA_VALUES:
                method_keys.append(f'nested_spheres_a{alpha}_b{beta}')
    all_results = {key: [] for key in method_keys}

    metadata_k   = {}
    fisher_diags = []

    # ── Split loop ──────────────────────────────────────────────────────────
    for split_i, tr_idx, va_idx, te_idx in splits_list:
        print(f'\n  --- Split {split_i + 1}/{NUM_RANDOM_SPLITS} '
              f'(train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}) ---')

        # Restricted eigenvectors: computed once per split per k
        print('  Computing restricted eigenvectors...')
        U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
            X_diffused, L, D, num_components
        )
        print(f'  d_effective={d_eff}, ortho_error={ortho_err:.2e}')

        # .npz matrix caching disabled (files were too large; analytics will recompute)
        # npz_path = os.path.join(k_dir, 'matrices', f'split{split_i}_matrices.npz')
        # np.savez_compressed(
        #     npz_path,
        #     U=U,
        #     X_diffused=X_diffused,
        #     eigenvalues=eigenvalues,
        #     train_idx=tr_idx,
        #     val_idx=va_idx,
        #     test_idx=te_idx
        # )
        # print(f'  Saved matrices: {npz_path}')

        # Set metadata once (split 0 only): d_eff and ortho_err are identical across
        # splits since U is recomputed from the same X_diffused, L, D every time.
        if not metadata_k:
            metadata_k.update({
                'd_effective':    int(d_eff),
                'ortho_error':    float(ortho_err),
                'num_components': int(num_components),
                'num_nodes':      int(num_nodes),
                'num_features':   int(X_raw.shape[1]),
                'num_classes':    int(num_classes)
            })

        # Standard-scaled features for SGC+MLP and Restricted+StandardMLP
        scaler_sgc = StandardScaler()
        X_sgc_tr   = scaler_sgc.fit_transform(X_diffused[tr_idx])
        X_sgc_va   = scaler_sgc.transform(X_diffused[va_idx])
        X_sgc_te   = scaler_sgc.transform(X_diffused[te_idx])

        scaler_std = StandardScaler()
        U_std_tr   = scaler_std.fit_transform(U[tr_idx])
        U_std_va   = scaler_std.transform(U[va_idx])
        U_std_te   = scaler_std.transform(U[te_idx])

        # Raw eigenvectors for RowNorm-based methods (no scaling)
        U_tr = U[tr_idx]
        U_va = U[va_idx]
        U_te = U[te_idx]

        eigenvalues_t = torch.FloatTensor(eigenvalues)

        # Fisher diagnostic — ALL k values, includes X_diffused comparison
        fisher_info = compute_fisher_diagnostic(
            U[tr_idx], labels[tr_idx], num_classes,
            X_diffused_train=X_diffused[tr_idx]
        )
        fisher_diags.append({'split': int(split_i), **fisher_info})

        dynamics_records = []  # Exp 7: per-epoch metrics for all 3 core methods

        # ── Seed loop ────────────────────────────────────────────────────────
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)

            do_dyn = (k == 10 and track_dynamics)

            # ── SGC + MLP ──────────────────────────────────────────────────
            model    = StandardMLP(X_diffused.shape[1], HIDDEN_DIM, num_classes)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            va, te, tr_acc, t, dyn = train_model(
                model, X_sgc_tr, labels[tr_idx],
                X_sgc_va, labels[va_idx],
                X_sgc_te, labels[te_idx],
                device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE,
                track_dynamics=do_dyn
            )
            all_results['sgc_mlp_baseline'].append({
                'seed': seed, 'split': split_i,
                'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                'num_params': n_params,
                'param_to_train_ratio': n_params / len(tr_idx)
            })
            if do_dyn and dyn is not None:
                dynamics_records.append({
                    'seed': seed, 'split': split_i, 'method': 'sgc_mlp_baseline', **dyn
                })

            # ── Restricted + StandardMLP ───────────────────────────────────
            torch.manual_seed(seed); np.random.seed(seed)
            model    = StandardMLP(d_eff, HIDDEN_DIM, num_classes)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            va, te, tr_acc, t, dyn = train_model(
                model, U_std_tr, labels[tr_idx],
                U_std_va, labels[va_idx],
                U_std_te, labels[te_idx],
                device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE,
                track_dynamics=do_dyn
            )
            all_results['restricted_standard_mlp'].append({
                'seed': seed, 'split': split_i,
                'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                'num_params': n_params,
                'param_to_train_ratio': n_params / len(tr_idx)
            })
            if do_dyn and dyn is not None:
                dynamics_records.append({
                    'seed': seed, 'split': split_i, 'method': 'restricted_standard_mlp', **dyn
                })

            # ── Restricted + RowNorm ───────────────────────────────────────
            torch.manual_seed(seed); np.random.seed(seed)
            model    = RowNormMLP(d_eff, HIDDEN_DIM, num_classes)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            va, te, tr_acc, t, dyn = train_model(
                model, U_tr, labels[tr_idx],
                U_va, labels[va_idx],
                U_te, labels[te_idx],
                device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE,
                track_dynamics=do_dyn
            )
            all_results['restricted_rownorm_mlp'].append({
                'seed': seed, 'split': split_i,
                'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                'num_params': n_params,
                'param_to_train_ratio': n_params / len(tr_idx)
            })
            if do_dyn and dyn is not None:
                dynamics_records.append({
                    'seed': seed, 'split': split_i, 'method': 'restricted_rownorm_mlp', **dyn
                })

            # ── k=10 extended methods ──────────────────────────────────────
            if k == 10:

                # Log-Magnitude
                torch.manual_seed(seed); np.random.seed(seed)
                model    = LogMagnitudeMLP(d_eff, HIDDEN_DIM, num_classes)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                va, te, tr_acc, t, _ = train_model(
                    model, U_tr, labels[tr_idx],
                    U_va, labels[va_idx],
                    U_te, labels[te_idx],
                    device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE
                )
                all_results['log_magnitude'].append({
                    'seed': seed, 'split': split_i,
                    'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                    'num_params': n_params,
                    'param_to_train_ratio': n_params / len(tr_idx)
                })

                # Dual-Stream
                torch.manual_seed(seed); np.random.seed(seed)
                model    = DualStreamMLP(d_eff, HIDDEN_DIM, HIDDEN_MAG, num_classes)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                va, te, tr_acc, t, _ = train_model(
                    model, U_tr, labels[tr_idx],
                    U_va, labels[va_idx],
                    U_te, labels[te_idx],
                    device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE
                )
                all_results['dual_stream'].append({
                    'seed': seed, 'split': split_i,
                    'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                    'num_params': n_params,
                    'param_to_train_ratio': n_params / len(tr_idx)
                })

                # SpectralRowNorm — all alpha values including 0.0
                for alpha in ALPHA_VALUES:
                    torch.manual_seed(seed); np.random.seed(seed)
                    model    = SpectralRowNormMLP(
                        d_eff, HIDDEN_DIM, num_classes, eigenvalues_t, alpha=alpha
                    )
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    va, te, tr_acc, t, _ = train_model(
                        model, U_tr, labels[tr_idx],
                        U_va, labels[va_idx],
                        U_te, labels[te_idx],
                        device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE
                    )
                    all_results[f'spectral_rownorm_alpha{alpha}'].append({
                        'seed': seed, 'split': split_i,
                        'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                        'num_params': n_params,
                        'param_to_train_ratio': n_params / len(tr_idx)
                    })

                # NestedSpheres — 5×5 alpha × beta grid
                for alpha in NESTED_ALPHA_VALUES:
                    for beta in NESTED_BETA_VALUES:
                        torch.manual_seed(seed); np.random.seed(seed)
                        model    = NestedSpheresClassifier(
                            d_eff, HIDDEN_DIM, num_classes, eigenvalues_t,
                            alpha=alpha, beta=beta
                        )
                        n_params = sum(p.numel() for p in model.parameters()
                                       if p.requires_grad)
                        va, te, tr_acc, t, _ = train_model(
                            model, U_tr, labels[tr_idx],
                            U_va, labels[va_idx],
                            U_te, labels[te_idx],
                            device, epochs=EPOCHS, lr=LR, weight_decay=WD, patience=PATIENCE
                        )
                        all_results[f'nested_spheres_a{alpha}_b{beta}'].append({
                            'seed': seed, 'split': split_i,
                            'val_acc': va, 'test_acc': te, 'train_acc': tr_acc,
                            'num_params': n_params,
                            'param_to_train_ratio': n_params / len(tr_idx)
                        })

        # ── End seed loop ────────────────────────────────────────────────────

        # Save dynamics for this split (k=10, dynamics_datasets)
        if k == 10 and track_dynamics and dynamics_records:
            dyn_path = os.path.join(k_dir, 'training_curves', f'split{split_i}_dynamics.json')
            with open(dyn_path, 'w') as f:
                json.dump(dynamics_records, f, indent=2)
            print(f'  Saved dynamics: {dyn_path}')

    # ── End split loop ───────────────────────────────────────────────────────

    # Aggregate results
    print(f'\n  Aggregating results for k={k} '
          f'({NUM_RANDOM_SPLITS} splits x {NUM_SEEDS} seeds = '
          f'{NUM_RANDOM_SPLITS * NUM_SEEDS} runs/method)...')

    aggregated = {name: aggregate_results(results)
                  for name, results in all_results.items() if results}

    print(f'\n  {"Method":<52} {"Test Acc":>10}  {"Std":>8}  {"n":>4}')
    print(f'  {"-"*78}')
    for name, agg in aggregated.items():
        print(f'  {name:<52} {agg["test_acc_mean"]*100:>10.2f}%  '
              f'{agg["test_acc_std"]*100:>7.2f}%  {agg["n_runs"]:>4}')

    # Framework analysis
    sgc_mlp_acc   = aggregated['sgc_mlp_baseline']['test_acc_mean'] * 100
    restr_std_acc = aggregated['restricted_standard_mlp']['test_acc_mean'] * 100
    restr_rn_acc  = aggregated['restricted_rownorm_mlp']['test_acc_mean'] * 100

    framework = {
        'sgc_mlp_acc_pct':            float(sgc_mlp_acc),
        'restricted_std_acc_pct':     float(restr_std_acc),
        'restricted_rownorm_acc_pct': float(restr_rn_acc),
        'part_a_pp':                  float(sgc_mlp_acc - restr_std_acc),
        'part_b_pp':                  float(restr_rn_acc - restr_std_acc),
        # remaining_gap_pp = Part A - Part B = (SGC - Std) - (RowNorm - Std) = SGC - RowNorm.
        # Algebraically this is just sgc_mlp_acc - restr_rn_acc — the portion of SGC's
        # advantage over Std that RowNorm does NOT recover. Stored in simplified form.
        'remaining_gap_pp':           float(sgc_mlp_acc - restr_rn_acc)
    }

    if k == 10:
        # Best spectral alpha (Part B.6) — selected on val_acc to avoid test-set leakage.
        # We identify the best alpha using val_acc_mean, then report its test_acc_mean.
        best_alpha     = None
        best_alpha_val = -1.0
        for alpha in ALPHA_VALUES:
            key = f'spectral_rownorm_alpha{alpha}'
            if key in aggregated and aggregated[key]['val_acc_mean'] > best_alpha_val:
                best_alpha_val = aggregated[key]['val_acc_mean']
                best_alpha     = alpha

        if best_alpha is not None:
            best_alpha_test_acc = aggregated[f'spectral_rownorm_alpha{best_alpha}']['test_acc_mean']
        else:
            best_alpha_test_acc = None

        framework['best_spectral_alpha']   = best_alpha
        framework['best_spectral_acc_pct'] = (float(best_alpha_test_acc * 100)
                                               if best_alpha_test_acc is not None else None)
        framework['part_b6_pp'] = (float(best_alpha_test_acc * 100 - restr_rn_acc)
                                    if best_alpha_test_acc is not None else None)

        # Part B.5
        if 'log_magnitude' in aggregated:
            framework['part_b5_logmag_pp'] = float(
                aggregated['log_magnitude']['test_acc_mean'] * 100 - restr_rn_acc
            )
        if 'dual_stream' in aggregated:
            framework['part_b5_dual_pp'] = float(
                aggregated['dual_stream']['test_acc_mean'] * 100 - restr_rn_acc
            )

        # Best NestedSpheres configuration — selected on val_acc to avoid test-set leakage
        best_ns_key = None
        best_ns_val = -1.0
        for alpha in NESTED_ALPHA_VALUES:
            for beta in NESTED_BETA_VALUES:
                key = f'nested_spheres_a{alpha}_b{beta}'
                if key in aggregated and aggregated[key]['val_acc_mean'] > best_ns_val:
                    best_ns_val = aggregated[key]['val_acc_mean']
                    best_ns_key = key
        if best_ns_key:
            best_ns_test_acc = aggregated[best_ns_key]['test_acc_mean']
            framework['best_nested_spheres_key']     = best_ns_key
            framework['best_nested_spheres_acc_pct'] = float(best_ns_test_acc * 100)
            framework['part_b6_nested_pp']           = float(best_ns_test_acc * 100 - restr_rn_acc)

    print(f'\n  Framework Analysis (k={k}):')
    print(f'    Part A (SGC+MLP - Restricted+Std): {framework["part_a_pp"]:+.3f} pp')
    print(f'    Part B (RowNorm - Restricted+Std): {framework["part_b_pp"]:+.3f} pp')
    print(f'    Remaining Gap:                     {framework["remaining_gap_pp"]:+.3f} pp')

    # Save results JSON
    output_json = {
        'dataset':               DATASET_NAME,
        'k':                     k,
        'split_type':            SPLIT_TYPE,
        'component_type':        COMPONENT_TYPE,
        'num_seeds':             NUM_SEEDS,
        'num_splits':            NUM_RANDOM_SPLITS,
        'total_runs_per_method': NUM_SEEDS * NUM_RANDOM_SPLITS,
        'metadata':              metadata_k,
        'experiments':           aggregated,
        'framework_analysis':    framework
    }

    results_path = os.path.join(k_dir, 'metrics', 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f'\n  Saved: {results_path}')

    # Save Fisher diagnostic — ALL k values
    if fisher_diags:
        fisher_scores = [d['fisher_score'] for d in fisher_diags]
        fisher_agg = {
            # fisher_score = mean across all splits (use this for reporting)
            'fisher_score':     float(np.mean(fisher_scores)),
            'std_fisher_score': float(np.std(fisher_scores)),
            'per_split':        fisher_diags,
            'note': 'Fisher score computed on training set only (no leakage)'
        }
        fisher_path = os.path.join(k_dir, 'diagnostics', 'fisher_diagnostic.json')
        with open(fisher_path, 'w') as f:
            json.dump(fisher_agg, f, indent=2)
        print(f'  Saved: {fisher_path}')

print(f'\n{"="*80}')
print('MASTER TRAINING EXPERIMENT COMPLETE')
print(f'{"="*80}')
print(f'Results saved to: {OUTPUT_BASE}')
