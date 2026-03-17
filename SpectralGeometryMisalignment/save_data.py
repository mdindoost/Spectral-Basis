"""
SpectralGeometryMisalignment/save_data.py

Loads each dataset, extracts the LCC, computes X (raw features) and Y
(restricted eigenvectors via Rayleigh-Ritz), and saves as .npy files.

CONVENTION:
  L = D - A  (unnormalized Laplacian, NO self-loops)
  D = degree matrix (NO self-loops)
  Solve (Q^T L Q) V = (Q^T D Q) V Lambda  [Rayleigh-Ritz]
  Y = Q V  with Y^T D Y = I (D-orthonormal)
  span(Y) = span(X) exactly by construction.

Output: SpectralGeometryMisalignment/data/{dataset}/
  X.npy                         -- raw LCC features, shape (n, d)
  Y.npy                         -- restricted eigenvectors, shape (n, d_eff)
  labels.npy                    -- integer class labels, shape (n,)
  fixed_train_mask.npy          -- boolean, shape (n,)
  fixed_val_mask.npy            -- boolean, shape (n,)
  fixed_test_mask.npy           -- boolean, shape (n,)
  random{seed}_train_mask.npy   -- for seed in {0,1,2,3,4}
  random{seed}_val_mask.npy
  random{seed}_test_mask.npy

Fixed split convention (matches master_training.py exactly):
  Planetoid (cora, citeseer, pubmed): standard Planetoid masks
  Amazon/Coauthor: 60/20/20 with np.random.default_rng(42), same as master_training.py
  WikiCS: split index 0
  ogbn-arxiv: OGB official split

Random splits: stratified 60/20/20 with seeds 0,1,2,3,4 (on LCC nodes)

Usage:
  /home/md724/Spectral-Basis/venv/bin/python save_data.py
  /home/md724/Spectral-Basis/venv/bin/python save_data.py --datasets cora citeseer
"""

import os
import sys
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

# PyTorch 2.6 + OGB compatibility patch (must happen before any OGB import)
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from graph_utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_restricted_eigenvectors,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical dataset names (used for load_dataset and folder names)
# Folder uses underscores; load_dataset uses hyphens for multi-word names.
_DATASET_LOAD_NAMES = [
    'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'wikics',
    'amazon-computers', 'amazon-photo', 'coauthor-cs', 'coauthor-physics',
]

# Folder name → load name mapping
def _folder_name(ds: str) -> str:
    """Convert load name to folder name: hyphens → underscores."""
    return ds.replace('-', '_')

RANDOM_SEEDS  = [0, 1, 2, 3, 4]
DATASET_ROOT  = os.path.join(_REPO_ROOT, 'dataset')
DATA_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ── Graph helpers ─────────────────────────────────────────────────────────────

def build_no_selfloop_matrices(edge_index_lcc: np.ndarray, n: int):
    """
    Build L = D - A and D from adjacency WITHOUT self-loops.

    edge_index_lcc: (2, E) array of directed edges (should already be undirected)
    Returns: (L, D) as scipy sparse matrices.
    """
    edges = edge_index_lcc.T  # (E, 2)
    A = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float64),
         (edges[:, 0], edges[:, 1])),
        shape=(n, n)
    ).tocsr()
    A = A.maximum(A.T)  # symmetrise (safety — should already be undirected)
    deg = np.array(A.sum(axis=1)).ravel()
    D   = sp.diags(deg)
    L   = D - A
    return L, D


# ── Per-dataset processing ────────────────────────────────────────────────────

def process_dataset(dataset_name: str) -> dict:
    """Process one dataset and save all .npy files."""
    print(f'\n{"="*70}')
    print(f'Dataset: {dataset_name}')
    print(f'{"="*70}')

    # ── 1. Load full graph ────────────────────────────────────────────────────
    (edge_index, X_raw, labels, num_nodes, num_classes,
     train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(
        dataset_name, root=DATASET_ROOT
    )
    print(f'Full graph: {num_nodes:,} nodes, {X_raw.shape[1]} features, '
          f'{num_classes} classes')

    # ── 2. Build graph matrices (self-loops included) for LCC detection ────────
    adj_full, _, _ = build_graph_matrices(edge_index, num_nodes)

    # ── 3. Extract LCC ────────────────────────────────────────────────────────
    lcc_mask = get_largest_connected_component_nx(adj_full)
    pct = lcc_mask.mean() * 100
    print(f'LCC: {lcc_mask.sum():,}/{num_nodes:,} nodes ({pct:.1f}%)')

    split_idx_orig = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig,
    }
    adj_lcc, X_lcc, labels_lcc, split_idx_lcc = extract_subgraph(
        adj_full, X_raw, labels, lcc_mask, split_idx_orig
    )

    n = adj_lcc.shape[0]
    if sp.issparse(X_lcc):
        X_dense = np.asarray(X_lcc.todense(), dtype=np.float64)
    else:
        X_dense = X_lcc.copy().astype(np.float64)

    d           = X_dense.shape[1]
    labels_arr  = labels_lcc.astype(np.int64)

    # ── 4. Build L and D WITHOUT self-loops (Rayleigh-Ritz convention) ────────
    # strip self-loops from adj_lcc (which inherited them from build_graph_matrices)
    adj_lcc_csr  = adj_lcc.tocsr()
    adj_no_loops = adj_lcc_csr - sp.diags(adj_lcc_csr.diagonal())
    adj_nl_coo   = adj_no_loops.tocoo()
    edge_idx_lcc = np.vstack([adj_nl_coo.row, adj_nl_coo.col])  # (2, E)

    L, D = build_no_selfloop_matrices(edge_idx_lcc, n)

    # ── 5. Compute Y = restricted eigenvectors of X (k=0, no diffusion) ───────
    print('Computing restricted eigenvectors Y  [L and D have NO self-loops]...')
    Y, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_dense, L, D, num_components=0
    )
    print(f'  d_eff={d_eff}, ortho_error={ortho_err:.2e}')

    if ortho_err >= 1e-6:
        raise RuntimeError(
            f'D-orthonormality check FAILED for {dataset_name}: '
            f'max|Y^T D Y - I| = {ortho_err:.2e}  (threshold 1e-6)'
        )

    # ── 6. Fixed split masks ──────────────────────────────────────────────────
    # split_idx_lcc holds remapped LCC indices (integer arrays)
    tr_idx_fixed = split_idx_lcc['train_idx'] if split_idx_lcc else np.array([], dtype=int)
    va_idx_fixed = split_idx_lcc['val_idx']   if split_idx_lcc else np.array([], dtype=int)
    te_idx_fixed = split_idx_lcc['test_idx']  if split_idx_lcc else np.array([], dtype=int)

    fixed_train_mask = np.zeros(n, dtype=bool)
    fixed_val_mask   = np.zeros(n, dtype=bool)
    fixed_test_mask  = np.zeros(n, dtype=bool)
    fixed_train_mask[tr_idx_fixed] = True
    fixed_val_mask[va_idx_fixed]   = True
    fixed_test_mask[te_idx_fixed]  = True

    # Sanity: no node should appear in two splits
    assert not np.any(fixed_train_mask & fixed_val_mask),  'Train/Val overlap!'
    assert not np.any(fixed_train_mask & fixed_test_mask), 'Train/Test overlap!'
    assert not np.any(fixed_val_mask & fixed_test_mask),   'Val/Test overlap!'

    # ── 7. Random split masks (stratified 60/20/20 on LCC nodes) ─────────────
    all_indices   = np.arange(n)
    random_masks  = {}
    for seed in RANDOM_SEEDS:
        # 60 % train, 40 % temp  (stratified)
        tr, temp = train_test_split(
            all_indices, test_size=0.40,
            random_state=seed, stratify=labels_arr
        )
        # split temp 50/50 → 20 % val, 20 % test  (stratified)
        va, te = train_test_split(
            temp, test_size=0.50,
            random_state=seed, stratify=labels_arr[temp]
        )
        tr_m = np.zeros(n, dtype=bool); tr_m[tr] = True
        va_m = np.zeros(n, dtype=bool); va_m[va] = True
        te_m = np.zeros(n, dtype=bool); te_m[te] = True
        random_masks[seed] = (tr_m, va_m, te_m)

    # ── 8. Save all files ─────────────────────────────────────────────────────
    folder    = _folder_name(dataset_name)
    out_dir   = os.path.join(DATA_OUT_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'X.npy'),              X_dense.astype(np.float32))
    np.save(os.path.join(out_dir, 'Y.npy'),              Y.astype(np.float32))
    np.save(os.path.join(out_dir, 'labels.npy'),          labels_arr)
    np.save(os.path.join(out_dir, 'fixed_train_mask.npy'), fixed_train_mask)
    np.save(os.path.join(out_dir, 'fixed_val_mask.npy'),   fixed_val_mask)
    np.save(os.path.join(out_dir, 'fixed_test_mask.npy'),  fixed_test_mask)

    for seed, (tr_m, va_m, te_m) in random_masks.items():
        np.save(os.path.join(out_dir, f'random{seed}_train_mask.npy'), tr_m)
        np.save(os.path.join(out_dir, f'random{seed}_val_mask.npy'),   va_m)
        np.save(os.path.join(out_dir, f'random{seed}_test_mask.npy'),  te_m)

    print(f'  Saved to: {out_dir}/')

    # ── 9. Verification summary ───────────────────────────────────────────────
    print(f'\n  Summary:')
    print(f'    {"n":>10} {"d":>6} {"d_eff":>6} '
          f'{"train":>8} {"val":>8} {"test":>8}  {"ortho_err":>12}')
    print(f'    {n:>10,} {d:>6} {d_eff:>6} '
          f'{fixed_train_mask.sum():>8} {fixed_val_mask.sum():>8} '
          f'{fixed_test_mask.sum():>8}  {ortho_err:.2e}')
    print(f'    D-orthonormality: PASS (< 1e-6)')

    return {
        'dataset': dataset_name,
        'n': n, 'd': d, 'd_eff': d_eff,
        'ortho_err': float(ortho_err),
        'train': int(fixed_train_mask.sum()),
        'val':   int(fixed_val_mask.sum()),
        'test':  int(fixed_test_mask.sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Save X, Y, and split masks for SpectralGeometryMisalignment'
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Dataset names to process (default: all 9). '
             'Use hyphens, e.g. --datasets cora ogbn-arxiv amazon-photo'
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else _DATASET_LOAD_NAMES

    print('SpectralGeometryMisalignment — save_data.py')
    print(f'Output root: {DATA_OUT_ROOT}')
    print(f'Processing {len(datasets)} dataset(s): {datasets}')

    summaries = []
    for ds in datasets:
        info = process_dataset(ds)
        summaries.append(info)

    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)
    header = (f'  {"Dataset":<22} {"n":>8} {"d":>6} {"d_eff":>6} '
              f'{"train":>8} {"val":>8} {"test":>8}  {"ortho_err":>12}')
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for s in summaries:
        print(f'  {s["dataset"]:<22} {s["n"]:>8,} {s["d"]:>6} {s["d_eff"]:>6} '
              f'{s["train"]:>8} {s["val"]:>8} {s["test"]:>8}  {s["ortho_err"]:.2e}')

    print('\nAll datasets saved successfully.')
    print(f'Data directory: {DATA_OUT_ROOT}/')
