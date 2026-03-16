"""
Exp 9: Softmax Regression Convergence
======================================
Compares training dynamics of a linear softmax classifier trained on:
  - X_diff = A^k X  (SGC-diffused features, raw)
  - Y = U           (restricted eigenvectors via Rayleigh-Ritz, raw)

These two representations span the same subspace by construction (Rayleigh-Ritz
produces U = Q @ V where Q is from thin QR of X_diff, so col(U) = col(X_diff)).
The experiment probes whether span-equivalence implies equivalent convergence for
gradient descent — and where it breaks down.

Two optimizers (SGD, Adam) × 2 feature sets (X_diff, Y) × k ∈ {1, 2, 10}.
Fixed split, 15 seeds, 500 epochs, NO early stopping, NO weight decay.

Per-epoch logs: train_loss, train_acc, val_acc, test_acc.

Step 1 (data saving — only when --save-arrays is passed):
  data/{dataset}_fixed_lcc/
    X_diff_k{k}.npy   — raw diffused features (n × d_raw)
    Y_k{k}.npy        — restricted eigenvectors U (n × d_eff)
    labels.npy        — node class labels (n,)
    split_idx.npy     — dict with keys 'train', 'val', 'test'

  NOTE: data/ files are large (60–300 MB per dataset) and are SAFE TO DELETE.
  They are recomputed on the fly from the raw dataset whenever --save-arrays is used.
  Do not commit them to git.

  Verified before saving:
    ortho_error  < 1e-6   (D-orthonormality: U^T D U ≈ I)
    mean_cc      > 0.99   (mean canonical correlation between span(X_diff) and span(U))

Step 2 (experiment):
  PAPER_EXPERIMENTS/PAPER_RESULTS/{dataset}_fixed_lcc/k{k}/softmax/
    exp9_sgd.json
    exp9_adam.json

Usage:
  cd /home/md724/Spectral-Basis
  venv/bin/python PAPER_EXPERIMENTS/exp9_softmax_convergence.py cora
  venv/bin/python PAPER_EXPERIMENTS/exp9_softmax_convergence.py cora --k_values 1 10
  venv/bin/python PAPER_EXPERIMENTS/exp9_softmax_convergence.py cora --device cpu
  venv/bin/python PAPER_EXPERIMENTS/exp9_softmax_convergence.py cora --save-arrays
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# PyTorch 2.6 + OGB compatibility (same patch as master_training.py)
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _SRC_DIR)
from graph_utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
)


# ============================================================================
# Argument parsing
# ============================================================================

parser = argparse.ArgumentParser(description='Exp 9: Softmax Regression Convergence')
parser.add_argument('dataset', type=str,
                    help='Dataset name (e.g. cora, amazon-computers, ...)')
parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 10],
                    help='Diffusion depths to run (default: 1 2 10)')
parser.add_argument('--num_seeds', type=int, default=15,
                    help='Training seeds per configuration (default: 15)')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device (default: cuda). Use --device cpu on machines without GPU.')
parser.add_argument('--save-arrays', action='store_true', default=False,
                    help='Save X_diff, Y, labels, split_idx as .npy files to data/. '
                         'Files are large (60–300 MB each) and safe to delete; omit this '
                         'flag when you only need the JSON results.')
args = parser.parse_args()

DATASET_NAME = args.dataset
K_VALUES     = sorted(args.k_values)
NUM_SEEDS    = args.num_seeds
SAVE_ARRAYS  = args.save_arrays
EPOCHS       = 500    # fixed, no early stopping — full trajectory is the point
LR           = 0.01   # canonical lr matching all other experiments
WD           = 0.0    # strictly zero — any regularization would confound the comparison

if not torch.cuda.is_available() and args.device == 'cuda':
    raise RuntimeError(
        "CUDA not available. Run with --device cpu or fix CUDA installation."
    )
device = torch.device(args.device)

# Paths — all relative to repo root so they work regardless of CWD
_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR     = os.path.join(_REPO_ROOT, 'data', f'{DATASET_NAME}_fixed_lcc')
RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'PAPER_RESULTS', f'{DATASET_NAME}_fixed_lcc')

if SAVE_ARRAYS:
    os.makedirs(DATA_DIR, exist_ok=True)

print('=' * 70)
print(f'EXP 9: Softmax Regression Convergence  —  {DATASET_NAME}')
print(f'k values: {K_VALUES}  |  seeds: {NUM_SEEDS}  |  epochs: {EPOCHS}')
print(f'weight_decay: {WD}  |  lr: {LR}  |  device: {device}')
print(f'save arrays:  {SAVE_ARRAYS}')
print('=' * 70)


# ============================================================================
# Steps 1–3: Load dataset, build graph, extract LCC
#
# This block is an EXACT copy of master_training.py steps 1–3 (lines 451–506).
# Any deviation would produce different LCC nodes / index remappings.
# ============================================================================

print('\n[1] Loading dataset...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(DATASET_NAME, root='./dataset')
print(f'    Nodes: {num_nodes:,}  Features: {X_raw.shape[1]}  Classes: {num_classes}')

print('\n[2] Building initial graph matrices...')
adj, L, D = build_graph_matrices(edge_index, num_nodes)

print('\n[3] Extracting LCC...')
lcc_mask = get_largest_connected_component_nx(adj)

# extract_subgraph remaps split indices from original-graph indices to LCC indices
split_idx_original = {
    'train_idx': train_idx_orig,
    'val_idx':   val_idx_orig,
    'test_idx':  test_idx_orig,
}
adj, X_raw, labels, split_idx = extract_subgraph(
    adj, X_raw, labels, lcc_mask, split_idx_original
)

# Rebuild matrices for LCC.
# CRITICAL: adj returned by extract_subgraph already contains self-loops (inherited
# from the initial build_graph_matrices call). Passing it directly to a second
# build_graph_matrices would add self-loops again → double self-loops → wrong D.
# Fix (identical to master_training.py:487-490): strip self-loops, re-extract edges,
# then call build_graph_matrices exactly once.
adj_no_loops   = adj - sp.diags(adj.diagonal())
adj_coo        = adj_no_loops.tocoo()
edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
adj, L, D      = build_graph_matrices(edge_index_lcc, adj.shape[0])

# LCC has exactly 1 connected component → num_components=0 (no trivial eigenvalues to skip)
num_components = 0

num_nodes   = X_raw.shape[0]
num_classes = len(np.unique(labels))
print(f'    LCC: {num_nodes:,} nodes, {num_classes} classes')

# Dense features needed for matrix multiply in sgc_precompute
features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()

# Fixed split — use LCC-remapped indices from extract_subgraph
# (matching master_training.py splits_list construction for fixed split)
tr_idx = split_idx['train_idx']
va_idx = split_idx['val_idx']
te_idx = split_idx['test_idx']
print(f'    Fixed split — train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}')

if SAVE_ARRAYS:
    # Save labels and split once (not per-k; they don't change with k)
    np.save(os.path.join(DATA_DIR, 'labels.npy'), labels)
    np.save(os.path.join(DATA_DIR, 'split_idx.npy'),
            {'train': tr_idx, 'val': va_idx, 'test': te_idx})
    # NOTE when loading split_idx: np.load('split_idx.npy', allow_pickle=True).item()
    print(f'    Saved labels.npy and split_idx.npy → {DATA_DIR}/')

# SGC normalized adjacency: D^{-1/2}(A+I)D^{-1/2}, computed once for all k
A_sgc = compute_sgc_normalized_adjacency(adj)


# ============================================================================
# Span alignment verification
#
# span(U) should equal span(X_diff) by construction. This checks the implementation.
#
# Method: compute orthonormal bases for each subspace via thin QR, then SVD of
# Q_U^T @ Q_X gives cosines of principal angles. Perfect alignment → all cosines = 1.
#
# d_eff = rank(X_diff): at high k, diffusion collapses features → rank < d_raw.
# We take only the first d_eff columns of Q_X (the rank-revealing columns).
# ============================================================================

def check_span_alignment(X_diff, U):
    """
    Returns (mean_cc, all_cc_list):
      mean_cc  — mean canonical correlation between span(X_diff) and span(U)
      all_cc   — list of all d_eff cosines (principal angles)

    Should give mean_cc > 0.99 if the Rayleigh-Ritz implementation is correct.
    """
    d_eff = U.shape[1]

    # Thin QR of X_diff → Q_X shape (n, d_raw); keep first d_eff columns
    # which span the rank-d_eff column space of X_diff.
    Q_X, _ = np.linalg.qr(X_diff)
    Q_X    = Q_X[:, :d_eff]

    # Thin QR of U → Q_U shape (n, d_eff)
    Q_U, _ = np.linalg.qr(U)

    # SVD of Q_U^T @ Q_X: singular values = cosines of principal angles
    _, sv, _ = np.linalg.svd(Q_U.T @ Q_X, full_matrices=False)
    sv = np.clip(sv, 0.0, 1.0)  # numerical clamp (svd can exceed 1 by ~1e-15)

    return float(sv.mean()), sv.tolist()


# ============================================================================
# Step 1: Compute and save data arrays for all k values
# ============================================================================

print('\n' + '=' * 70)
print('[Step 1] Computing and saving data arrays...')
print('=' * 70)

# Cache arrays for use in Step 2 (avoids recomputing diffusion twice)
arrays_by_k = {}

for k in K_VALUES:
    print(f'\n  k = {k}')

    X_diff = sgc_precompute(features_dense.copy(), A_sgc, k)

    # Rayleigh-Ritz: compute restricted eigenvectors on X_diff
    # Signature: compute_restricted_eigenvectors(X, L, D, num_components)
    # L=Laplacian, D=degree — order matches CLAUDE.md convention and graph_utils.py
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_diff, L, D, num_components
    )

    print(f'    d_eff = {d_eff}, X_diff.shape = {X_diff.shape}, U.shape = {U.shape}')

    # Verification 1: D-orthonormality — U^T D U should be identity (< 1e-6)
    print(f'    ortho_error = {ortho_err:.2e}', end='')
    if ortho_err >= 1e-6:
        raise ValueError(
            f'D-orthonormality check FAILED at k={k}: ortho_err={ortho_err:.2e}. '
            f'Check that L and D were not swapped in compute_restricted_eigenvectors call.'
        )
    print('  [PASS]')

    # Verification 2: span alignment — mean canonical correlation > 0.99
    mean_cc, all_cc = check_span_alignment(X_diff, U)
    print(f'    mean canonical correlation = {mean_cc:.6f}', end='')
    if mean_cc < 0.99:
        raise ValueError(
            f'Span check FAILED at k={k}: mean_cc={mean_cc:.6f}. '
            f'span(U) and span(X_diff) are not aligned — check Rayleigh-Ritz implementation.'
        )
    print('  [PASS]')

    if SAVE_ARRAYS:
        np.save(os.path.join(DATA_DIR, f'X_diff_k{k}.npy'), X_diff)
        np.save(os.path.join(DATA_DIR, f'Y_k{k}.npy'),      U)
        print(f'    Saved X_diff_k{k}.npy and Y_k{k}.npy')

    arrays_by_k[k] = {
        'X_diff':      X_diff,
        'U':           U,
        'eigenvalues': eigenvalues,
        'd_eff':       d_eff,
        'ortho_err':   ortho_err,
        'mean_cc':     mean_cc,
    }

print('\n[Step 1] All verifications passed.' + (' Data saved.' if SAVE_ARRAYS else ' (--save-arrays not set, skipping file writes.)'))


# ============================================================================
# Softmax regression model
#
# Single linear layer (weight + bias) → raw logits → CrossEntropyLoss applies
# log-softmax internally. This is the canonical PyTorch softmax regression.
# Xavier initialization matches the convention in SGC (models.py:34).
# ============================================================================

class SoftmaxRegression(nn.Module):
    """Linear softmax classifier (no hidden layers, no regularization)."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Raw logits — CrossEntropyLoss applies log-softmax internally
        return self.fc(x)


# ============================================================================
# Training loop
#
# Deliberately simpler than master_training.py train_model:
#   - No early stopping (we want the full 500-epoch trajectory)
#   - No weight decay (WD=0.0 strictly)
#   - Full-batch (same as master_training.py — all existing experiments are full-batch)
#   - Per-epoch: train_loss, train_acc, val_acc, test_acc all logged
# ============================================================================

def train_softmax(model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                  optimizer_name):
    """
    Train softmax regression for exactly EPOCHS epochs.

    Returns:
        dyn: dict of per-epoch lists (train_loss, train_acc, val_acc, test_acc)
        final_test_acc: test accuracy at the last epoch (epoch 500)
        peak_val_acc:   highest val accuracy seen at any epoch
    """
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t  = torch.LongTensor(y_tr).to(device)
    X_va_t  = torch.FloatTensor(X_va).to(device)
    y_va_t  = torch.LongTensor(y_va).to(device)
    X_te_t  = torch.FloatTensor(X_te).to(device)
    y_te_t  = torch.LongTensor(y_te).to(device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # WD=0.0 in both optimizers — no implicit regularization
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    dyn = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}

    for _epoch in range(EPOCHS):
        # Full-batch training step
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss   = criterion(logits, y_tr_t)
        loss.backward()
        optimizer.step()

        # Full evaluation every epoch — needed for convergence analysis
        model.eval()
        with torch.no_grad():
            tr_acc = (model(X_tr_t).argmax(1) == y_tr_t).float().mean().item()
            va_acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            te_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()

        dyn['train_loss'].append(float(loss.item()))
        dyn['train_acc'].append(float(tr_acc))
        dyn['val_acc'].append(float(va_acc))
        dyn['test_acc'].append(float(te_acc))

    return dyn, dyn['test_acc'][-1], max(dyn['val_acc'])


def epochs_to_fraction_of_peak(val_accs, fraction):
    """
    First epoch (1-indexed) where val_acc >= fraction * peak_val_acc.

    NOTE: these fractions are of each model's OWN peak, not absolute thresholds.
    Directly comparable between X_diff and Y only when their peak val_accs are similar.
    Same caveat applies as in master_training.py:215.
    """
    target = fraction * max(val_accs)
    for i, v in enumerate(val_accs):
        if v >= target:
            return i + 1  # 1-indexed
    return None  # never reached (can happen with oscillating SGD)


# ============================================================================
# Step 2: Run experiment — all k × optimizer combinations
# ============================================================================

print('\n' + '=' * 70)
print('[Step 2] Running softmax regression experiment...')
print('=' * 70)

for k in K_VALUES:
    arr   = arrays_by_k[k]
    X_diff = arr['X_diff']
    U      = arr['U']
    d_eff  = arr['d_eff']

    # Raw features — NO StandardScaler.
    # master_training.py scales X_diff and U for SGC+MLP / StandardMLP experiments
    # (lines 621-628). Here we deliberately skip scaling: we want to compare the raw
    # representations X_diff and U on equal footing, without any additional transformation
    # that could confound the convergence comparison.
    X_diff_tr = X_diff[tr_idx];  X_diff_va = X_diff[va_idx];  X_diff_te = X_diff[te_idx]
    U_tr      = U[tr_idx];       U_va      = U[va_idx];       U_te      = U[te_idx]

    k_dir = os.path.join(RESULTS_BASE, f'k{k}', 'softmax')
    os.makedirs(k_dir, exist_ok=True)

    for opt_name in ['sgd', 'adam']:
        print(f'\n  k={k}, optimizer={opt_name.upper()}')
        print(f'  X_diff: {X_diff_tr.shape}  Y: {U_tr.shape}')

        # Per-seed result lists
        xdiff_results = []
        y_results     = []

        # Accumulate dynamics across seeds (sum → divide at end for mean trajectory)
        xdiff_dyn_sum = {m: np.zeros(EPOCHS)
                         for m in ('train_loss', 'train_acc', 'val_acc', 'test_acc')}
        y_dyn_sum     = {m: np.zeros(EPOCHS)
                         for m in ('train_loss', 'train_acc', 'val_acc', 'test_acc')}

        for seed in range(NUM_SEEDS):
            # ── X_diff run ──────────────────────────────────────────────────
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_xd = SoftmaxRegression(X_diff.shape[1], num_classes)
            dyn_xd, final_te_xd, peak_va_xd = train_softmax(
                model_xd,
                X_diff_tr, labels[tr_idx],
                X_diff_va, labels[va_idx],
                X_diff_te, labels[te_idx],
                opt_name,
            )

            # ── Y (U) run — same seed for a fair comparison ─────────────────
            # Reseed identically so weight initialization differences between
            # the two models are not a confound.
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_y = SoftmaxRegression(d_eff, num_classes)
            dyn_y, final_te_y, peak_va_y = train_softmax(
                model_y,
                U_tr, labels[tr_idx],
                U_va, labels[va_idx],
                U_te, labels[te_idx],
                opt_name,
            )

            xdiff_results.append({
                'seed':                seed,
                'final_test_acc':      final_te_xd,
                'peak_val_acc':        peak_va_xd,
                'epoch_to_90pct_peak': epochs_to_fraction_of_peak(dyn_xd['val_acc'], 0.90),
                'epoch_to_95pct_peak': epochs_to_fraction_of_peak(dyn_xd['val_acc'], 0.95),
            })
            y_results.append({
                'seed':                seed,
                'final_test_acc':      final_te_y,
                'peak_val_acc':        peak_va_y,
                'epoch_to_90pct_peak': epochs_to_fraction_of_peak(dyn_y['val_acc'], 0.90),
                'epoch_to_95pct_peak': epochs_to_fraction_of_peak(dyn_y['val_acc'], 0.95),
            })

            for m in xdiff_dyn_sum:
                xdiff_dyn_sum[m] += np.array(dyn_xd[m])
                y_dyn_sum[m]     += np.array(dyn_y[m])

            if (seed + 1) % 5 == 0 or seed == NUM_SEEDS - 1:
                print(f'    seed {seed+1:2d}/{NUM_SEEDS} — '
                      f'X_diff final_test={final_te_xd*100:.2f}%  '
                      f'Y final_test={final_te_y*100:.2f}%')

        # ── Aggregate across seeds ───────────────────────────────────────────

        def _aggregate(results_list, dyn_sum):
            final_tes = [r['final_test_acc']      for r in results_list]
            peak_vas  = [r['peak_val_acc']         for r in results_list]
            ep90s     = [r['epoch_to_90pct_peak']  for r in results_list
                         if r['epoch_to_90pct_peak'] is not None]
            ep95s     = [r['epoch_to_95pct_peak']  for r in results_list
                         if r['epoch_to_95pct_peak'] is not None]
            # Mean dynamics trajectory (averaged over seeds)
            mean_dyn = {m: (dyn_sum[m] / NUM_SEEDS).tolist() for m in dyn_sum}
            return {
                # Naming matches master_training.py aggregate_results() output
                'test_acc_mean':            float(np.mean(final_tes)),
                'test_acc_std':             float(np.std(final_tes)),
                'val_acc_mean':             float(np.mean(peak_vas)),   # peak val across training
                'val_acc_std':              float(np.std(peak_vas)),
                'per_run_test_acc':         [float(v) for v in final_tes],
                'n_runs':                   NUM_SEEDS,
                # Convergence speed (fractions of each model's own peak, not absolute)
                'epoch_to_90pct_peak_mean': float(np.mean(ep90s)) if ep90s else None,
                'epoch_to_95pct_peak_mean': float(np.mean(ep95s)) if ep95s else None,
                'n_seeds_reached_90pct':    len(ep90s),
                'n_seeds_reached_95pct':    len(ep95s),
                # Full mean trajectory — stored for plotting convergence curves
                'mean_dynamics':            mean_dyn,
            }

        xdiff_agg = _aggregate(xdiff_results, xdiff_dyn_sum)
        y_agg     = _aggregate(y_results,     y_dyn_sum)

        # Flag unexpected accuracy gap.
        # Span identity (col(U) = col(X_diff)) implies a softmax model on X_diff can
        # represent any softmax model on U and vice versa (via a linear reparametrization).
        # Therefore FINAL test accuracies should converge to the same value.
        # A gap > 1pp at epoch 500 is unexpected and warrants investigation.
        delta_pp        = abs(xdiff_agg['test_acc_mean'] - y_agg['test_acc_mean']) * 100
        flag_unexpected = delta_pp > 1.0

        # ── Save JSON ────────────────────────────────────────────────────────

        output_json = {
            # Provenance (mirrors master_training.py results.json top-level fields)
            'dataset':        DATASET_NAME,
            'k':              k,
            'optimizer':      opt_name,
            'lr':             LR,
            'weight_decay':   WD,
            'epochs':         EPOCHS,
            'num_seeds':      NUM_SEEDS,
            'split_type':     'fixed',
            'component_type': 'lcc',
            # Metadata (mirrors master_training.py 'metadata' field)
            'metadata': {
                'num_nodes':    int(num_nodes),
                'num_classes':  int(num_classes),
                'x_diff_dim':   int(X_diff.shape[1]),
                'y_dim':        int(d_eff),
                'd_effective':  int(d_eff),
                'ortho_error':  float(arr['ortho_err']),
                'mean_cc':      float(arr['mean_cc']),
                'train_size':   int(len(tr_idx)),
                'val_size':     int(len(va_idx)),
                'test_size':    int(len(te_idx)),
            },
            # Flags
            'flag_unexpected_gap': bool(flag_unexpected),
            'test_acc_gap_pp':     float(delta_pp),
            # Results (mirrors master_training.py 'experiments' field)
            'experiments': {
                'X_diff': xdiff_agg,
                'Y':      y_agg,
            },
        }

        out_path = os.path.join(k_dir, f'exp9_{opt_name}.json')
        with open(out_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f'  Saved: {out_path}')

        # ── Print summary table ──────────────────────────────────────────────

        print(f'\n  ── k={k}  {opt_name.upper()} ────────────────────────────────────')
        print(f'  {"Feature":<10}  {"Test%":>8}  {"±Std":>7}  '
              f'{"Ep→90%":>8}  {"Ep→95%":>8}  {"PeakVal%":>10}')
        print(f'  {"-"*62}')
        for name, agg_d in [('X_diff', xdiff_agg), ('Y (U)', y_agg)]:
            ep90 = (f'{agg_d["epoch_to_90pct_peak_mean"]:.1f}'
                    if agg_d['epoch_to_90pct_peak_mean'] is not None else 'N/A')
            ep95 = (f'{agg_d["epoch_to_95pct_peak_mean"]:.1f}'
                    if agg_d['epoch_to_95pct_peak_mean'] is not None else 'N/A')
            print(f'  {name:<10}  {agg_d["test_acc_mean"]*100:>7.2f}%  '
                  f'±{agg_d["test_acc_std"]*100:>5.2f}%  '
                  f'{ep90:>8}  {ep95:>8}  '
                  f'{agg_d["val_acc_mean"]*100:>9.2f}%')
        print(f'  {"-"*62}')
        print(f'  |X_diff − Y| test gap: {delta_pp:.2f} pp', end='')
        if flag_unexpected:
            print(f'  *** UNEXPECTED (>1pp) — span identity says these should be equal ***')
        else:
            print(f'  (within 1pp — consistent with span identity)')

print(f'\n{"=" * 70}')
print('EXP 9 COMPLETE')
print(f'Results → PAPER_EXPERIMENTS/PAPER_RESULTS/{DATASET_NAME}_fixed_lcc/k*/softmax/')
print(f'Data    → data/{DATASET_NAME}_fixed_lcc/')
print(f'{"=" * 70}')
