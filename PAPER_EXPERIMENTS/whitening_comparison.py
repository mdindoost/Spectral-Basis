"""
Whitening Comparison Experiment
================================
Compares five feature representations on benchmark graph datasets:
  1. original     — Â^k X (diffused features, no whitening)
  2. pca_whiten   — PCA whitening computed from training set only
  3. zca_whiten   — ZCA whitening computed from training set only
  4. full_zca     — ZCA computed from ALL nodes (Wadia et al. method)
  5. rayleigh_ritz — D-orthonormal restricted eigenvectors U (our method)

For each representation: MLP and MLP+RowNorm classifiers.

BUG FIX vs Archive/experiments/investigation_whitening_rownorm.py
------------------------------------------------------------------
The old script imported from experiments/utils.py which had:
    compute_sgc_normalized_adjacency(adj):
        adj = adj + sp.eye(adj.shape[0])   # ALWAYS added self-loops
Because build_graph_matrices already adds self-loops, this caused DOUBLE
self-loops in diffusion (A + 2I normalized instead of A + I normalized).
This script imports from src/graph_utils.py which does NOT re-add self-loops.
The LCC rebuild follows master_training.py exactly (strips loops, then
calls build_graph_matrices to add them exactly once).

Canonical hyperparameters:
  EPOCHS=500, PATIENCE=100, SEEDS=15, LR=0.01, WD=5e-4, HIDDEN_DIM=256

Usage:
    # Single dataset, all k values, fixed split
    $PYTHON PAPER_EXPERIMENTS/whitening_comparison.py cora

    # Random splits
    $PYTHON PAPER_EXPERIMENTS/whitening_comparison.py cora --splits random

    # Specific k values
    $PYTHON PAPER_EXPERIMENTS/whitening_comparison.py cora --k_values 1 10

    # Dry run (2 seeds, k=1 only, for smoke testing)
    $PYTHON PAPER_EXPERIMENTS/whitening_comparison.py cora --dry_run

    # Full sweep (all 9 datasets, fixed + random)
    for ds in cora citeseer pubmed ogbn-arxiv wikics amazon-photo amazon-computers coauthor-cs coauthor-physics; do
        $PYTHON PAPER_EXPERIMENTS/whitening_comparison.py $ds --splits fixed
        $PYTHON PAPER_EXPERIMENTS/whitening_comparison.py $ds --splits random
    done

Output:
    PAPER_EXPERIMENTS/PAPER_RESULTS/whitening_comparison/{dataset}_{split}_lcc/k{k}/results.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from datetime import datetime

# ── Canonical src imports (fixes double self-loop bug) ────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR   = os.path.join(_REPO_ROOT, 'src')
sys.path.insert(0, _SRC_DIR)

from graph_utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,   # does NOT re-add self-loops (bug fixed)
    sgc_precompute,
    compute_restricted_eigenvectors,
)
from models import StandardMLP, RowNormMLP

# ============================================================================
# Argument parsing
# ============================================================================

parser = argparse.ArgumentParser(description='Whitening comparison experiment')
parser.add_argument('dataset', type=str,
                    help='Dataset name (cora, citeseer, pubmed, ogbn-arxiv, wikics, '
                         'amazon-photo, amazon-computers, coauthor-cs, coauthor-physics)')
parser.add_argument('--splits', type=str, choices=['fixed', 'random'], default='fixed')
parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 10],
                    help='Diffusion depths to run (default: 1 2 10)')
parser.add_argument('--num_seeds', type=int, default=15,
                    help='Training seeds per config (default: 15)')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dry_run', action='store_true',
                    help='Smoke test: cora k=1 only, 2 seeds')
args = parser.parse_args()

DATASET_NAME  = args.dataset
SPLIT_TYPE    = args.splits
K_VALUES      = sorted(args.k_values)
NUM_SEEDS     = args.num_seeds
DRY_RUN       = args.dry_run

if DRY_RUN:
    K_VALUES  = [1]
    NUM_SEEDS = 2
    print('[DRY RUN] k=[1], seeds=2')

NUM_RANDOM_SPLITS = 5 if SPLIT_TYPE == 'random' else 1

# ── Canonical hyperparameters (match master_training.py) ──────────────────────
HIDDEN_DIM = 256
LR         = 0.01
WD         = 5e-4
EPOCHS     = 500
PATIENCE   = 100

WHITENING_EPS = 1e-6   # regularisation floor for whitening eigenvalues

# ── Device ────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available() and args.device == 'cuda':
    print('WARNING: CUDA unavailable, falling back to CPU')
    device = torch.device('cpu')
else:
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'PAPER_RESULTS', 'whitening_comparison',
    f'{DATASET_NAME}_{SPLIT_TYPE}_lcc'
)
os.makedirs(OUTPUT_BASE, exist_ok=True)

print('=' * 72)
print(f'Whitening Comparison — {DATASET_NAME}')
print(f'k values:   {K_VALUES}')
print(f'Split type: {SPLIT_TYPE}  ({NUM_RANDOM_SPLITS} split(s) × {NUM_SEEDS} seeds)')
print(f'Device:     {device}')
print(f'Output:     {OUTPUT_BASE}')
print('=' * 72)

# ============================================================================
# Training function (canonical protocol — mirrors master_training.py)
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    500 epochs, patience=100 early stopping on val_acc, Adam lr=0.01.
    Returns (best_val_acc, best_test_acc).
    """
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.LongTensor(y_val).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    y_te = torch.LongTensor(y_test).to(device)

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit  = nn.CrossEntropyLoss()

    best_val, best_test, patience_ctr = 0.0, 0.0, 0

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        crit(model(X_tr), y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_acc  = (model(X_va).argmax(1) == y_va).float().mean().item()
            test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()

        if val_acc > best_val:
            best_val, best_test, patience_ctr = val_acc, test_acc, 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    return float(best_val), float(best_test)


def run_seeds(model_fn, feat, labels, tr, va, te):
    """Run NUM_SEEDS seeds, return mean/std of test_acc."""
    results = []
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        va_acc, te_acc = train_model(
            model_fn(),
            feat[tr], labels[tr],
            feat[va], labels[va],
            feat[te], labels[te],
        )
        results.append({'seed': seed, 'val_acc': va_acc, 'test_acc': te_acc})
    test_accs = [r['test_acc'] for r in results]
    return {
        'runs':          results,
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std':  float(np.std(test_accs)),
    }

# ============================================================================
# Logistic regression (linear) classifier — not in src/models.py
# ============================================================================

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, x):
        return self.fc(x)


class RowNormLinear(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        return self.fc(F.normalize(x, p=2, dim=1))

# ============================================================================
# Whitening transforms
# ============================================================================

def pca_whiten(X, train_idx):
    """PCA whitening — transform computed from training set only."""
    mu = X[train_idx].mean(axis=0)
    Xc = X - mu
    Fc = Xc[train_idx].T @ Xc[train_idx] / (len(train_idx) - 1)
    evals, evecs = np.linalg.eigh(Fc)
    evals = evals[::-1]; evecs = evecs[:, ::-1]
    evals_reg = np.maximum(evals, WHITENING_EPS)
    W = evecs @ np.diag(1.0 / np.sqrt(evals_reg))
    return (Xc @ W).astype(np.float32), {
        'rank_deficient': int(np.sum(evals < WHITENING_EPS)),
        'condition_before': float(evals[0] / (evals[-1] + 1e-10)),
    }


def zca_whiten(X, train_idx):
    """ZCA whitening — stays in original feature space."""
    mu = X[train_idx].mean(axis=0)
    Xc = X - mu
    Fc = Xc[train_idx].T @ Xc[train_idx] / (len(train_idx) - 1)
    evals, evecs = np.linalg.eigh(Fc)
    evals = evals[::-1]; evecs = evecs[:, ::-1]
    evals_reg = np.maximum(evals, WHITENING_EPS)
    W = evecs @ np.diag(1.0 / np.sqrt(evals_reg)) @ evecs.T
    return (Xc @ W).astype(np.float32), {
        'rank_deficient': int(np.sum(evals < WHITENING_EPS)),
        'condition_before': float(evals[0] / (evals[-1] + 1e-10)),
    }


def full_zca_whiten(X):
    """ZCA computed from ALL nodes (Wadia et al.'s method — uses test data)."""
    mu = X.mean(axis=0)
    Xc = X - mu
    Fc = Xc.T @ Xc / (len(X) - 1)
    evals, evecs = np.linalg.eigh(Fc)
    evals = evals[::-1]; evecs = evecs[:, ::-1]
    evals_reg = np.maximum(evals, WHITENING_EPS)
    W = evecs @ np.diag(1.0 / np.sqrt(evals_reg)) @ evecs.T
    return (Xc @ W).astype(np.float32), {
        'rank_deficient': int(np.sum(evals < WHITENING_EPS)),
        'wadia_regime':   'n<=d (collapse expected)' if X.shape[0] <= X.shape[1]
                          else 'n>d (structure retained)',
    }


def gram_matrix_check(X):
    """Check Wadia criterion: are Gram matrix K=XX^T eigenvalues ≈ 1?"""
    Xc = X - X.mean(axis=0)
    n, d = Xc.shape
    Fc = Xc.T @ Xc / (n - 1)
    evals_F = np.linalg.eigvalsh(Fc)
    evals_F = np.sort(evals_F)[::-1]
    evals_K = evals_F[:n] if d > n else evals_F
    pos = evals_K[evals_K > 1e-10]
    if len(pos) == 0:
        return {'K_is_identity': False, 'K_dist_from_1_mean': float('inf')}
    dist = float(np.abs(pos - 1).mean())
    return {
        'K_is_approximately_identity': bool(dist < 0.1),
        'K_dist_from_1_mean':          dist,
        'K_condition_number':          float(pos.max() / pos.min()),
        'wadia_regime':                'n<=d' if n <= d else 'n>d',
    }

# ============================================================================
# Step 1: Load dataset
# ============================================================================

print('\n[1] Loading dataset...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_orig, val_orig, test_orig) = load_dataset(DATASET_NAME, root='./dataset')
print(f'    nodes={num_nodes:,}, features={X_raw.shape[1]}, classes={num_classes}')

# ============================================================================
# Step 2 + 3: Build graph matrices and extract LCC
#   Mirrors master_training.py lines 462-491 exactly.
# ============================================================================

print('\n[2] Building graph matrices and extracting LCC...')
adj, L, D = build_graph_matrices(edge_index, num_nodes)

lcc_mask = get_largest_connected_component_nx(adj)
split_idx_orig = {'train_idx': train_orig, 'val_idx': val_orig, 'test_idx': test_orig}
adj, X_raw, labels, split_idx = extract_subgraph(
    adj, X_raw, labels, lcc_mask, split_idx_orig)

# Strip self-loops before re-feeding into build_graph_matrices to avoid double addition
adj_no_loops   = adj - sp.diags(adj.diagonal())
adj_coo        = adj_no_loops.tocoo()
edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
adj, L, D      = build_graph_matrices(edge_index_lcc, adj.shape[0])
num_components = 0   # LCC has exactly 1 connected component

num_nodes_lcc = X_raw.shape[0]
print(f'    LCC nodes={num_nodes_lcc:,}')

# Dense features for diffusion
X_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw.astype(np.float64)

# Precompute normalized adjacency once (no self-loops added — already in adj)
A_hat = compute_sgc_normalized_adjacency(adj)

# ============================================================================
# Step 4: Prepare splits
# ============================================================================

print('\n[3] Preparing splits...')

if SPLIT_TYPE == 'fixed':
    splits_list = [(0,
                    split_idx['train_idx'],
                    split_idx['val_idx'],
                    split_idx['test_idx'])]
    tr, va, te = splits_list[0][1:]
    print(f'    Fixed: train={len(tr)}, val={len(va)}, test={len(te)}')
else:
    splits_list = []
    for s in range(NUM_RANDOM_SPLITS):
        np.random.seed(s)
        idx = np.arange(num_nodes_lcc)
        np.random.shuffle(idx)
        n_tr = int(0.6 * num_nodes_lcc)
        n_va = int(0.2 * num_nodes_lcc)
        splits_list.append((s, idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]))
    tr, va, te = splits_list[0][1:]
    print(f'    Random (5 splits): n_train≈{len(tr)}, n_val≈{len(va)}, n_test≈{len(te)}')

# ============================================================================
# Main loop: k → split → feature variant → model
# ============================================================================

all_k_results = {}

for k in K_VALUES:
    print(f'\n{"="*72}')
    print(f'k = {k}')
    print('='*72)

    # Diffuse once per k (split-independent)
    X_diff = sgc_precompute(X_dense.copy(), A_hat, k)
    print(f'  X_diff shape: {X_diff.shape}')

    k_records = []  # one entry per split

    for split_i, tr_idx, va_idx, te_idx in splits_list:
        print(f'\n  --- split {split_i} (train={len(tr_idx)}) ---')

        # ── Rayleigh-Ritz (split-independent geometry, but gram check uses train) ──
        U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
            X_diff, L, D, num_components)
        print(f'  d_eff={d_eff}, ortho_error={ortho_err:.2e}')
        if ortho_err > 1e-6:
            print(f'  WARNING: ortho_error={ortho_err:.2e} exceeds 1e-6 threshold')

        # ── Whitening variants (train-set variants computed per split) ────────────
        X_pca,  diag_pca  = pca_whiten(X_diff, tr_idx)
        X_zca,  diag_zca  = zca_whiten(X_diff, tr_idx)
        X_fzca, diag_fzca = full_zca_whiten(X_diff)

        variants = {
            'original':      X_diff.astype(np.float32),
            'pca_whiten':    X_pca,
            'zca_whiten':    X_zca,
            'full_zca':      X_fzca,
            'rayleigh_ritz': U.astype(np.float32),
        }

        # ── Gram matrix checks ────────────────────────────────────────────────────
        gram = {name: gram_matrix_check(feat[tr_idx])
                for name, feat in variants.items()}

        # ── Train all models ──────────────────────────────────────────────────────
        split_results = {}
        for vname, feat in variants.items():
            dim = feat.shape[1]
            model_configs = {
                'MLP':          lambda d=dim: StandardMLP(d, HIDDEN_DIM, num_classes),
                'MLP+RowNorm':  lambda d=dim: RowNormMLP(d, HIDDEN_DIM, num_classes),
                'Linear':       lambda d=dim: LinearClassifier(d, num_classes),
                'Linear+RN':    lambda d=dim: RowNormLinear(d, num_classes),
            }
            split_results[vname] = {}
            for mname, model_fn in model_configs.items():
                res = run_seeds(model_fn, feat, labels, tr_idx, va_idx, te_idx)
                split_results[vname][mname] = {
                    'test_acc_mean': res['test_acc_mean'],
                    'test_acc_std':  res['test_acc_std'],
                }
                print(f'    {vname:16s} {mname:14s}: '
                      f'{res["test_acc_mean"]*100:.2f}% ± {res["test_acc_std"]*100:.2f}%')

        k_records.append({
            'split_i':       split_i,
            'n_train':       len(tr_idx),
            'd_eff':         d_eff,
            'ortho_error':   float(ortho_err),
            'results':       split_results,
            'gram_checks':   gram,
            'whitening_diag': {
                'pca':  diag_pca,
                'zca':  diag_zca,
                'full_zca': diag_fzca,
            },
        })

    # ── Aggregate across splits ───────────────────────────────────────────────
    def aggregate_splits(records, vname, mname):
        means = [r['results'][vname][mname]['test_acc_mean'] for r in records]
        stds  = [r['results'][vname][mname]['test_acc_std']  for r in records]
        return {
            'test_acc_mean':       float(np.mean(means)),
            'test_acc_std_across_splits': float(np.std(means)),
            'test_acc_std_within_seeds':  float(np.mean(stds)),
        }

    aggregated = {}
    for vname in ['original', 'pca_whiten', 'zca_whiten', 'full_zca', 'rayleigh_ritz']:
        aggregated[vname] = {}
        for mname in ['MLP', 'MLP+RowNorm', 'Linear', 'Linear+RN']:
            aggregated[vname][mname] = aggregate_splits(k_records, vname, mname)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f'\n  SUMMARY k={k}')
    print(f'  {"Variant":16s}  {"MLP":>8}  {"MLP+RN":>8}  '
          f'{"Damage(MLP)":>12}  {"RN_Recovery":>12}')
    orig_mlp = aggregated['original']['MLP']['test_acc_mean']
    for vname in ['original', 'pca_whiten', 'zca_whiten', 'full_zca', 'rayleigh_ritz']:
        mlp  = aggregated[vname]['MLP']['test_acc_mean']
        rn   = aggregated[vname]['MLP+RowNorm']['test_acc_mean']
        dmg  = (orig_mlp - mlp) * 100
        rec  = (rn - mlp) * 100
        print(f'  {vname:16s}  {mlp*100:>7.2f}%  {rn*100:>7.2f}%  '
              f'{dmg:>+11.2f}pp  {rec:>+11.2f}pp')

    all_k_results[k] = {
        'splits':     k_records,
        'aggregated': aggregated,
    }

    # ── Save per-k results ────────────────────────────────────────────────────
    k_dir = os.path.join(OUTPUT_BASE, f'k{k}')
    os.makedirs(k_dir, exist_ok=True)
    out = {
        'metadata': {
            'dataset':      DATASET_NAME,
            'k':            k,
            'split_type':   SPLIT_TYPE,
            'num_splits':   NUM_RANDOM_SPLITS,
            'num_seeds':    NUM_SEEDS,
            'num_nodes_lcc': num_nodes_lcc,
            'num_classes':  num_classes,
            'epochs':       EPOCHS,
            'patience':     PATIENCE,
            'lr':           LR,
            'wd':           WD,
            'hidden_dim':   HIDDEN_DIM,
            'whitening_eps': WHITENING_EPS,
            'bug_fix':      'double_self_loop_fixed_imports_src_graph_utils',
            'timestamp':    datetime.now().isoformat(),
        },
        'aggregated': aggregated,
        'splits':     k_records,
    }
    path = os.path.join(k_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n  Saved: {path}')

print('\n' + '='*72)
print('COMPLETE')
print('='*72)
