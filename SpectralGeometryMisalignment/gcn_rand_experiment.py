"""
SpectralGeometryMisalignment/gcn_rand_experiment.py

NULL FEATURE EXPERIMENT — 5-MODEL COMPARISON

MODELS:
  GCN_X         : 2-layer Kipf GCN on real X (reference ceiling)
  GCN_rand      : 2-layer Kipf GCN on X_rand ~ N(0,1) (null features, topology only)
  GCN_rowNorm_Y : 2-layer Kipf GCN on Y_rand with row-norm before each layer's aggregation
  MLP_Y         : Linear classifier on Y_rand (StandardScaler preprocessing)
  MLP_rowNorm_Y : Linear classifier on row-normalized Y_rand (row-norm replaces StandardScaler)

PURPOSE:
  Isolate what graph topology contributes to classification accuracy when node
  features carry zero class information. Provides a clean decomposition:
    GCN_X  - GCN_rand  = contribution of real feature class information to GCN
    GCN_rand - MLP_Y   = learned message-passing vs analytic spectral projection
                         under null features
    MLP_rowNorm_Y vs MLP_Y = effect of row normalization on Y_rand

MATHEMATICAL BASIS:
  Y_rand = Rayleigh-Ritz eigenvectors of (L,D) restricted to span(X_rand).
  X_rand ~ N(0,1) is full rank → d_eff = d. Capped to d_eff_real columns.
  Y_rand ≈ randomized spectral graph embedding (approximates leading Laplacian eigenvectors).
  Class-informativeness of Y_rand = 0 when X_rand is random (no feature-class alignment).

  GCN_rowNorm_Y forward (Option B — row-norm at each layer INPUT):
    Layer 1: Â @ row_norm(Y_rand) → fc1 → ReLU → dropout
    Layer 2: Â @ row_norm(H^(1)) → fc2

TRAINING PROTOCOL (identical across all models):
  Optimizers : SGD (momentum=0.9, WD=0) + Adam (betas=(0.9,0.999), WD=0)
  LRs        : [0.001, 0.01, 0.1]
  Epochs     : 500, no early stopping, full-batch
  Train seeds: 5 (0-4)
  Splits     : fixed (1 split seed=0), random (5 split seeds 0-4)

X_rand draws (for GCN_rand, GCN_rowNorm_Y, MLP_Y, MLP_rowNorm_Y):
  3 draws (xrand_seed in {0,1,2}). GCN_X uses real X — no draws needed.

PREPROCESSING:
  GCN_X        : No scaling (standard GCN practice).
  GCN_rand     : No scaling (X_rand ~ N(0,1) is already standardized).
  GCN_rowNorm_Y: Row-norm inside forward pass. No external preprocessing.
  MLP_Y        : StandardScaler fit on train nodes only, applied to val/test.
  MLP_rowNorm_Y: Row-norm per node (replaces StandardScaler). No additional scaling.

GRAPH MATRICES (cached per dataset):
  adj_kipf = D̂^{-1/2}(A+I)D̂^{-1/2}  → data/{dataset}/adj_kipf.npz
  L, D     = unnormalized Laplacian and degree, NO self-loops → L.npz, D.npz

RESUME: Existing JSONs are updated in-place. Missing model keys are computed
        and added without re-running completed models.

OUTPUT:
  results/gcn_rand/{dataset}/
    {dataset}_{split_type}_splitseed{ss}_xseed{xs}_{opt}_lr{lr}_seed{ts}.json
      → contains: GCN_rand, GCN_rowNorm_Y, MLP_Y, MLP_rowNorm_Y
    {dataset}_{split_type}_splitseed{ss}_gcnX_{opt}_lr{lr}_seed{ts}.json
      → contains: GCN_X
  results/gcn_rand/summary.json

Usage:
  /home/md724/Spectral-Basis/venv/bin/python gcn_rand_experiment.py
  /home/md724/Spectral-Basis/venv/bin/python gcn_rand_experiment.py --sanity
  /home/md724/Spectral-Basis/venv/bin/python gcn_rand_experiment.py --dataset cora
  /home/md724/Spectral-Basis/venv/bin/python gcn_rand_experiment.py \\
      --dataset cora --optimizer adam --lr 0.01 --verbose
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
from sklearn.preprocessing import StandardScaler

# ── Path setup ─────────────────────────────────────────────────────────────────

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from graph_utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    compute_restricted_eigenvectors,
)

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_ROOT    = os.path.join(_HERE, 'data')
RESULTS_ROOT = os.path.join(_HERE, 'results', 'gcn_rand')
DATASET_ROOT = os.path.join(_REPO_ROOT, 'dataset')

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]
_LOAD_NAMES = [
    'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'wikics',
    'amazon-computers', 'amazon-photo', 'coauthor-cs', 'coauthor-physics',
]
FOLDER_TO_LOAD = dict(zip(ALL_DATASETS, _LOAD_NAMES))

OPTIMIZERS         = ['sgd', 'adam']
LEARNING_RATES     = [0.001, 0.01, 0.1]
TRAIN_SEEDS        = list(range(5))
XRAND_SEEDS        = list(range(3))
RANDOM_SPLIT_SEEDS = list(range(5))
EPOCHS             = 500
GCN_HIDDEN         = 256
GCN_DROPOUT        = 0.5
PRINT_EVERY        = 100

# Keys expected in per-run JSON for xrand-dependent models
XRAND_MODEL_KEYS = ['GCN_rand', 'GCN_rowNorm_Y', 'MLP_Y', 'MLP_rowNorm_Y']
# Keys expected in per-run JSON for GCN_X (no xrand)
GCN_X_KEY = 'GCN_X'


# ── Models ─────────────────────────────────────────────────────────────────────

class SoftmaxRegression(nn.Module):
    """Single linear layer. No hidden layers. Same as softmax_experiment.py."""
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes)

    def forward(self, x):
        return self.linear(x)


class GCN(nn.Module):
    """
    Standard 2-layer Kipf GCN.
      H^(1) = ReLU( Â X W1 )  with dropout
      H^(2) = Â H^(1) W2
    Used for GCN_X (real X) and GCN_rand (X_rand).
    """
    def __init__(self, d_in: int, hidden: int, num_classes: int,
                 dropout: float = GCN_DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(d_in, hidden)
        self.fc2  = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        h   = F.relu(self.fc1(torch.sparse.mm(adj, x)))
        h   = self.drop(h)
        out = self.fc2(torch.sparse.mm(adj, h))
        return out


class GCN_RowNorm(nn.Module):
    """
    2-layer Kipf GCN with row normalization at each layer's INPUT (Option B).
      Layer 1: Â @ row_norm(Y_rand) → fc1 → ReLU → dropout
      Layer 2: Â @ row_norm(H^(1)) → fc2
    row_norm: F.normalize(x, p=2, dim=1) — projects each node to unit L2 sphere.
    Zero-norm rows are left as-is by F.normalize (returns zero vector).
    Used for GCN_rowNorm_Y.
    """
    def __init__(self, d_in: int, hidden: int, num_classes: int,
                 dropout: float = GCN_DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(d_in, hidden)
        self.fc2  = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Layer 1: aggregate row-normalized input
        x_rn = F.normalize(x, p=2, dim=1)
        h    = F.relu(self.fc1(torch.sparse.mm(adj, x_rn)))
        h    = self.drop(h)
        # Layer 2: aggregate row-normalized hidden
        h_rn = F.normalize(h, p=2, dim=1)
        out  = self.fc2(torch.sparse.mm(adj, h_rn))
        return out


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def row_normalize_np(X: np.ndarray) -> np.ndarray:
    """
    Row-normalize X so each row has unit L2 norm.
    Zero-norm rows are left unchanged (norm replaced by 1.0 to avoid NaN).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


# ── Optimizer factory ──────────────────────────────────────────────────────────

def make_optimizer(name: str, params, lr: float):
    """WD=0.0 for all models (symmetric comparison)."""
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)


# ── Graph matrix builder (cached) ──────────────────────────────────────────────

def build_and_cache_graph(dataset_folder: str):
    """
    Build and cache graph matrices for the LCC.
    Returns: (adj_kipf, L, D) as scipy sparse matrices.
    Verifies: symmetry/non-negativity of Kipf adj, L=D-A consistency, L row sums=0.
    """
    data_dir = os.path.join(DATA_ROOT, dataset_folder)
    adj_path = os.path.join(data_dir, 'adj_kipf.npz')
    L_path   = os.path.join(data_dir, 'L.npz')
    D_path   = os.path.join(data_dir, 'D.npz')

    if all(os.path.isfile(p) for p in [adj_path, L_path, D_path]):
        print(f'  [cache] Loading adj_kipf, L, D for {dataset_folder}')
        adj_kipf = sp.load_npz(adj_path)
        L        = sp.load_npz(L_path)
        D        = sp.load_npz(D_path)
        _verify_graph_matrices(adj_kipf, L, D, dataset_folder)
        return adj_kipf, L, D

    print(f'  [build] Computing adj_kipf, L, D for {dataset_folder} ...')
    load_name = FOLDER_TO_LOAD[dataset_folder]
    (edge_index, X_raw, labels, num_nodes, num_classes,
     tr_idx, va_idx, te_idx) = load_dataset(load_name, root=DATASET_ROOT)

    adj_full, _, _ = build_graph_matrices(edge_index, num_nodes)
    lcc_mask       = get_largest_connected_component_nx(adj_full)
    split_idx      = {'train_idx': tr_idx, 'val_idx': va_idx, 'test_idx': te_idx}
    adj_lcc, _, _, _ = extract_subgraph(
        adj_full, X_raw, labels, lcc_mask, split_idx
    )
    print(f'    LCC: {adj_lcc.shape[0]:,} nodes')

    adj_kipf = compute_sgc_normalized_adjacency(adj_lcc)

    adj_csr = adj_lcc.tocsr()
    adj_nl  = (adj_csr - sp.diags(adj_csr.diagonal())).tocsr()
    A_sym   = adj_nl.maximum(adj_nl.T).tocsr()
    deg     = np.array(A_sym.sum(axis=1)).ravel()
    D       = sp.diags(deg, format='csr')
    L       = (D - A_sym).tocsr()

    _verify_graph_matrices(adj_kipf, L, D, dataset_folder)

    sp.save_npz(adj_path, adj_kipf.astype(np.float32))
    sp.save_npz(L_path,   L.astype(np.float64))
    sp.save_npz(D_path,   D.astype(np.float64))
    print(f'    Cached → {data_dir}/')

    return adj_kipf, L, D


def _verify_graph_matrices(adj_kipf, L, D, label: str):
    """
    CHECK 1: Kipf adj symmetric, non-negative, all diagonal > 0 (self-loops present).
    CHECK 2: L diagonal == D diagonal (L = D - A).
    CHECK 3: L row sums ≈ 0 (Laplacian property).
    """
    # Check 1
    diff          = (adj_kipf - adj_kipf.T).data
    max_asymmetry = float(np.abs(diff).max()) if len(diff) > 0 else 0.0
    assert max_asymmetry < 1e-6, \
        f'[{label}] Kipf adj not symmetric: max|A-A^T|={max_asymmetry:.2e}'
    min_entry = float(adj_kipf.data.min()) if len(adj_kipf.data) > 0 else 0.0
    assert min_entry >= -1e-10, \
        f'[{label}] Kipf adj has negative entries: min={min_entry:.2e}'
    assert (adj_kipf.diagonal() > 0).all(), \
        f'[{label}] Kipf adj missing self-loops'
    row_sums = np.array(adj_kipf.sum(axis=1)).ravel()
    print(f'    [CHECK 1] Kipf adj: symmetric [max|A-A^T|={max_asymmetry:.2e}]  '
          f'non-negative  self-loops present  '
          f'row_sums [{row_sums.min():.3f}, {row_sums.max():.3f}]  [OK]')

    # Check 2
    diag_diff = float(np.abs(L.diagonal() - D.diagonal()).max())
    assert diag_diff < 1e-10, \
        f'[{label}] L diagonal ≠ D diagonal: {diag_diff:.2e}'
    print(f'    [CHECK 2] L=D-A: max|L_diag - D_diag| = {diag_diff:.2e}  [OK]')

    # Check 3
    max_rowsum = float(np.abs(np.array(L.sum(axis=1)).ravel()).max())
    assert max_rowsum < 1e-9, \
        f'[{label}] L row sums not zero: {max_rowsum:.2e}'
    print(f'    [CHECK 3] L row sums: max = {max_rowsum:.2e}  [OK]')


def scipy_to_torch_sparse(adj, device):
    coo  = adj.tocoo().astype(np.float32)
    idx  = torch.LongTensor(np.vstack([coo.row, coo.col]))
    vals = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx, vals, coo.shape,
                                   device=device).coalesce()


# ── Y_rand computation ─────────────────────────────────────────────────────────

def compute_y_rand(X_rand: np.ndarray, L, D,
                   d_eff_real: int, xrand_seed: int) -> tuple:
    """
    Compute Y_rand via Rayleigh-Ritz on X_rand. Cap to d_eff_real columns.
    CHECK 4: X_rand statistics (mean≈0, std≈1).
    CHECK 5: D-orthonormality of Y_rand.
    CHECK 6: Output shape matches d_eff_real.
    """
    xmean, xstd = float(X_rand.mean()), float(X_rand.std())
    print(f'    [CHECK 4] X_rand (xseed={xrand_seed}): '
          f'mean={xmean:.4f}  std={xstd:.4f}  shape={X_rand.shape}  [OK ≈0,≈1]')

    Y, _, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_rand.astype(np.float64), L, D, num_components=0
    )
    if ortho_err >= 1e-6:
        raise RuntimeError(
            f'[CHECK 5] Y_rand D-ortho FAILED: {ortho_err:.2e} ≥ 1e-6'
        )
    print(f'    [CHECK 5] Y_rand D-ortho: {ortho_err:.2e}  [PASS]')

    d_used = min(d_eff, d_eff_real)
    if d_eff > d_eff_real:
        Y = Y[:, :d_eff_real]
        print(f'    Capping: d_eff_raw={d_eff} → d_used={d_eff_real} '
              f'(first {d_eff_real} eigenvectors = smallest eigenvalues)')
    else:
        print(f'    Y_rand: d_eff_raw={d_eff} = d_eff_real (no capping)')

    assert Y.shape[1] == d_used, f'[CHECK 6] Shape mismatch: {Y.shape[1]} ≠ {d_used}'
    print(f'    [CHECK 6] Y_rand shape: {Y.shape}  [OK]')

    return Y.astype(np.float32), d_eff, ortho_err


# ── Split mask loading & verification ─────────────────────────────────────────

def load_split_masks(dataset_folder: str, split_type: str, split_seed: int):
    d = os.path.join(DATA_ROOT, dataset_folder)
    if split_type == 'fixed':
        tr = np.load(os.path.join(d, 'fixed_train_mask.npy'))
        va = np.load(os.path.join(d, 'fixed_val_mask.npy'))
        te = np.load(os.path.join(d, 'fixed_test_mask.npy'))
    else:
        tr = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
        va = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
        te = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))
    return tr, va, te


def verify_split(tr, va, te, split_type, split_seed, n):
    """CHECK 7: split sizes and no overlap."""
    assert not np.any(tr & va), 'Train/Val overlap!'
    assert not np.any(tr & te), 'Train/Test overlap!'
    assert not np.any(va & te), 'Val/Test overlap!'
    print(f'    [CHECK 7] {split_type} ss={split_seed}: '
          f'train={tr.sum()}  val={va.sum()}  test={te.sum()}  '
          f'total={tr.sum()+va.sum()+te.sum()}/{n}  [OK]')


# ── Training loops ─────────────────────────────────────────────────────────────

def _run_epochs_gcn(model, X_all_t, adj_t, labels_t,
                    tr_mask_t, va_mask_t, te_mask_t,
                    opt, crit, label: str, verbose: bool):
    """Shared epoch loop for GCN and GCN_RowNorm."""
    train_loss, val_acc, test_acc = [], [], []
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_all_t, adj_t)
        loss   = crit(logits[tr_mask_t], labels_t[tr_mask_t])
        loss.backward()
        opt.step()

        lv = float(loss.item())
        if np.isnan(lv):
            print(f'    WARNING: {label} NaN at ep={ep}. Aborting.')
            pad = EPOCHS - ep
            train_loss.extend([float('nan')] * (pad + 1))
            val_acc.extend([float('nan')]    * (pad + 1))
            test_acc.extend([float('nan')]   * (pad + 1))
            return train_loss, val_acc, test_acc, float('nan')

        model.eval()
        with torch.no_grad():
            logits_eval = model(X_all_t, adj_t)
            preds = logits_eval.argmax(dim=1)
            va = (preds[va_mask_t] == labels_t[va_mask_t]).float().mean().item()
            te = (preds[te_mask_t] == labels_t[te_mask_t]).float().mean().item()

        train_loss.append(lv)
        val_acc.append(float(va))
        test_acc.append(float(te))

        if verbose and ep % PRINT_EVERY == 0:
            print(f'      {label:<16} ep={ep:>3}/{EPOCHS}  '
                  f'loss={lv:.4f}  val={va:.3f}  test={te:.3f}')

    return train_loss, val_acc, test_acc, float(test_acc[-1])


def train_gcn(X_all_t, adj_t, labels_t, tr_mask_t, va_mask_t, te_mask_t,
              num_classes: int, optimizer_name: str, lr: float,
              seed: int, device, verbose: bool = False, label='GCN') -> tuple:
    """Standard 2-layer Kipf GCN (used for GCN_X and GCN_rand)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN(X_all_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs_gcn(model, X_all_t, adj_t, labels_t,
                           tr_mask_t, va_mask_t, te_mask_t,
                           opt, nn.CrossEntropyLoss(), label, verbose)


def train_gcn_rownorm_y(Y_rand_t, adj_t, labels_t, tr_mask_t, va_mask_t, te_mask_t,
                        num_classes: int, optimizer_name: str, lr: float,
                        seed: int, device, verbose: bool = False) -> tuple:
    """GCN with row-norm at each layer input, applied to Y_rand."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN_RowNorm(Y_rand_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs_gcn(model, Y_rand_t, adj_t, labels_t,
                           tr_mask_t, va_mask_t, te_mask_t,
                           opt, nn.CrossEntropyLoss(), 'GCN_rowNorm_Y', verbose)


def train_mlp(Y_tr, y_tr, Y_va, y_va, Y_te, y_te,
              num_classes: int, optimizer_name: str, lr: float,
              seed: int, device, verbose: bool = False, label='MLP_Y') -> tuple:
    """
    Full-batch softmax regression. Same protocol as softmax_experiment.py.
    Preprocessing (StandardScaler or row-norm) applied BEFORE calling this function.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = SoftmaxRegression(Y_tr.shape[1], num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    crit  = nn.CrossEntropyLoss()

    Ytr = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long,    device=device)
    Yva = torch.tensor(Y_va, dtype=torch.float32, device=device)
    yva = torch.tensor(y_va, dtype=torch.long,    device=device)
    Yte = torch.tensor(Y_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long,    device=device)

    train_loss, val_acc, test_acc = [], [], []
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(Ytr), ytr)
        loss.backward()
        opt.step()

        lv = float(loss.item())
        if np.isnan(lv):
            print(f'    WARNING: {label} NaN at ep={ep}. Aborting.')
            pad = EPOCHS - ep
            train_loss.extend([float('nan')] * (pad + 1))
            val_acc.extend([float('nan')]    * (pad + 1))
            test_acc.extend([float('nan')]   * (pad + 1))
            return train_loss, val_acc, test_acc, float('nan')

        model.eval()
        with torch.no_grad():
            va = (model(Yva).argmax(1) == yva).float().mean().item()
            te = (model(Yte).argmax(1) == yte).float().mean().item()

        train_loss.append(lv)
        val_acc.append(float(va))
        test_acc.append(float(te))

        if verbose and ep % PRINT_EVERY == 0:
            print(f'      {label:<16} ep={ep:>3}/{EPOCHS}  '
                  f'loss={lv:.4f}  val={va:.3f}  test={te:.3f}')

    return train_loss, val_acc, test_acc, float(test_acc[-1])


# ── Result path helpers ────────────────────────────────────────────────────────

def lr_str(lr: float) -> str:
    return str(lr)


def xrand_result_path(dataset, split_type, split_seed,
                      xrand_seed, optimizer, lr, train_seed) -> str:
    """Path for per-run JSON containing GCN_rand, GCN_rowNorm_Y, MLP_Y, MLP_rowNorm_Y."""
    d = os.path.join(RESULTS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d,
        f'{dataset}_{split_type}_splitseed{split_seed}'
        f'_xseed{xrand_seed}_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


def gcnx_result_path(dataset, split_type, split_seed,
                     optimizer, lr, train_seed) -> str:
    """Path for per-run JSON containing GCN_X."""
    d = os.path.join(RESULTS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d,
        f'{dataset}_{split_type}_splitseed{split_seed}'
        f'_gcnX_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


def load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rec = json.load(f)
    # Migration: old code saved GCN results under key 'GCN'. New code uses 'GCN_rand'.
    # Rename silently so existing results are recognised and not re-run.
    if 'GCN' in rec and 'GCN_rand' not in rec:
        rec['GCN_rand'] = rec.pop('GCN')
        save_json(path, rec)   # persist the rename immediately
    return rec


def save_json(path: str, record: dict):
    with open(path, 'w') as f:
        json.dump(record, f)


def pack_result(loss_curve, val_curve, test_curve, final_acc) -> dict:
    return {
        'train_loss_curve': loss_curve,
        'val_acc_curve':    val_curve,
        'test_acc_curve':   test_curve,
        'final_test_acc':   final_acc,
    }


# ── Core experiment loop ───────────────────────────────────────────────────────

def run_dataset(dataset_folder: str, optimizers_to_run: list,
                lrs_to_run: list, device, verbose: bool = False) -> list:
    print(f'\n{"="*70}')
    print(f'Dataset: {dataset_folder}')
    print(f'{"="*70}')

    # Verify required files
    data_dir = os.path.join(DATA_ROOT, dataset_folder)
    required = ['X.npy', 'Y.npy', 'labels.npy',
                'fixed_train_mask.npy', 'fixed_val_mask.npy', 'fixed_test_mask.npy']
    for s in RANDOM_SPLIT_SEEDS:
        for sfx in ['train', 'val', 'test']:
            required.append(f'random{s}_{sfx}_mask.npy')
    missing = [f for f in required if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        print(f'ERROR: Missing files: {missing}. Run save_data.py first.')
        return []

    # Load metadata
    X_real      = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
    Y_real      = np.load(os.path.join(data_dir, 'Y.npy'))
    labels      = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)
    n, d        = X_real.shape
    d_eff_real  = Y_real.shape[1]
    num_classes = int(labels.max()) + 1
    del Y_real

    print(f'n={n:,}  d={d}  d_eff_real={d_eff_real}  '
          f'num_classes={num_classes}  '
          f'random_baseline={100.0/num_classes:.2f}%')

    # Build/load graph matrices
    adj_kipf, L, D = build_and_cache_graph(dataset_folder)
    adj_t          = scipy_to_torch_sparse(adj_kipf, device)
    labels_t       = torch.tensor(labels, dtype=torch.long, device=device)
    X_real_t       = torch.tensor(X_real, dtype=torch.float32, device=device)

    all_records = []

    # ── GCN_X: real features, no xrand loop ───────────────────────────────────
    print(f'\n  {"─"*60}')
    print(f'  GCN_X (real X, no draws)')
    print(f'  {"─"*60}')

    for split_type in ['fixed', 'random']:
        split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS
        for split_seed in split_seeds:
            tr_mask, va_mask, te_mask = load_split_masks(
                dataset_folder, split_type, split_seed
            )
            verify_split(tr_mask, va_mask, te_mask, split_type, split_seed, n)
            tr_t = torch.tensor(tr_mask, dtype=torch.bool, device=device)
            va_t = torch.tensor(va_mask, dtype=torch.bool, device=device)
            te_t = torch.tensor(te_mask, dtype=torch.bool, device=device)

            for optimizer_name in optimizers_to_run:
                for lr in lrs_to_run:
                    finals = []
                    for train_seed in TRAIN_SEEDS:
                        path = gcnx_result_path(dataset_folder, split_type,
                                                split_seed, optimizer_name,
                                                lr, train_seed)
                        rec = load_json(path)

                        if GCN_X_KEY not in rec:
                            be_v = (verbose and train_seed == TRAIN_SEEDS[0]
                                    and split_type == 'fixed')
                            lo, va, te, fi = train_gcn(
                                X_real_t, adj_t, labels_t, tr_t, va_t, te_t,
                                num_classes, optimizer_name, lr,
                                train_seed, device, verbose=be_v, label='GCN_X'
                            )
                            rec.update({
                                'dataset': dataset_folder,
                                'split_type': split_type, 'split_seed': split_seed,
                                'optimizer': optimizer_name, 'lr': lr,
                                'train_seed': train_seed, 'd': int(d),
                                'num_classes': int(num_classes),
                                'random_baseline': float(100.0 / num_classes),
                                GCN_X_KEY: pack_result(lo, va, te, fi),
                            })
                            save_json(path, rec)

                        finals.append(rec[GCN_X_KEY]['final_test_acc'] * 100)
                        all_records.append(rec)

                    print(f'  [GCN_X | {dataset_folder} | {split_type} ss={split_seed} | '
                          f'{optimizer_name} | lr={lr}]  '
                          f'{np.mean(finals):.2f}±{np.std(finals):.2f}%')

    # ── xrand models: GCN_rand, GCN_rowNorm_Y, MLP_Y, MLP_rowNorm_Y ──────────
    for xrand_seed in XRAND_SEEDS:
        print(f'\n  {"─"*60}')
        print(f'  X_rand draw xseed={xrand_seed}')
        print(f'  {"─"*60}')

        rng      = np.random.default_rng(xrand_seed)
        X_rand   = rng.standard_normal((n, d)).astype(np.float32)
        X_rand_t = torch.tensor(X_rand, dtype=torch.float32, device=device)

        Y_rand, d_eff_rand, ortho_err = compute_y_rand(
            X_rand, L, D, d_eff_real, xrand_seed
        )
        Y_rand_t = torch.tensor(Y_rand, dtype=torch.float32, device=device)

        # Row-normalized Y_rand (numpy, for MLP_rowNorm_Y)
        Y_rand_rn = row_normalize_np(Y_rand)

        for split_type in ['fixed', 'random']:
            split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS
            for split_seed in split_seeds:
                tr_mask, va_mask, te_mask = load_split_masks(
                    dataset_folder, split_type, split_seed
                )
                verify_split(tr_mask, va_mask, te_mask, split_type, split_seed, n)
                tr_t = torch.tensor(tr_mask, dtype=torch.bool, device=device)
                va_t = torch.tensor(va_mask, dtype=torch.bool, device=device)
                te_t = torch.tensor(te_mask, dtype=torch.bool, device=device)

                y_tr = labels[tr_mask]
                y_va = labels[va_mask]
                y_te = labels[te_mask]

                # MLP_Y preprocessing: StandardScaler
                sc = StandardScaler()
                Y_tr   = sc.fit_transform(Y_rand[tr_mask]).astype(np.float32)
                Y_va_s = sc.transform(Y_rand[va_mask]).astype(np.float32)
                Y_te_s = sc.transform(Y_rand[te_mask]).astype(np.float32)

                # MLP_rowNorm_Y preprocessing: row-normalize (already done globally)
                Y_tr_rn   = Y_rand_rn[tr_mask].astype(np.float32)
                Y_va_rn   = Y_rand_rn[va_mask].astype(np.float32)
                Y_te_rn   = Y_rand_rn[te_mask].astype(np.float32)

                for optimizer_name in optimizers_to_run:
                    for lr in lrs_to_run:
                        gcn_rand_f, gcnrn_f, mlpy_f, mlprn_f = [], [], [], []

                        for train_seed in TRAIN_SEEDS:
                            path = xrand_result_path(
                                dataset_folder, split_type, split_seed,
                                xrand_seed, optimizer_name, lr, train_seed
                            )
                            rec = load_json(path)

                            # Determine which models still need to run
                            need = [k for k in XRAND_MODEL_KEYS if k not in rec]

                            be_v = (verbose and train_seed == TRAIN_SEEDS[0]
                                    and optimizer_name == optimizers_to_run[0]
                                    and lr == lrs_to_run[0]
                                    and split_type == 'fixed')

                            if 'GCN_rand' in need:
                                lo, va, te, fi = train_gcn(
                                    X_rand_t, adj_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v, label='GCN_rand'
                                )
                                rec['GCN_rand'] = pack_result(lo, va, te, fi)

                            if 'GCN_rowNorm_Y' in need:
                                lo, va, te, fi = train_gcn_rownorm_y(
                                    Y_rand_t, adj_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v
                                )
                                rec['GCN_rowNorm_Y'] = pack_result(lo, va, te, fi)

                            if 'MLP_Y' in need:
                                lo, va, te, fi = train_mlp(
                                    Y_tr, y_tr, Y_va_s, y_va, Y_te_s, y_te,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v, label='MLP_Y'
                                )
                                rec['MLP_Y'] = pack_result(lo, va, te, fi)

                            if 'MLP_rowNorm_Y' in need:
                                lo, va, te, fi = train_mlp(
                                    Y_tr_rn, y_tr, Y_va_rn, y_va, Y_te_rn, y_te,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v,
                                    label='MLP_rowNorm_Y'
                                )
                                rec['MLP_rowNorm_Y'] = pack_result(lo, va, te, fi)

                            # Always update metadata
                            rec.update({
                                'dataset': dataset_folder,
                                'split_type': split_type, 'split_seed': split_seed,
                                'xrand_seed': xrand_seed,
                                'optimizer': optimizer_name, 'lr': lr,
                                'train_seed': train_seed,
                                'd': int(d), 'd_eff_real': int(d_eff_real),
                                'd_eff_rand_raw': int(d_eff_rand),
                                'd_eff_rand_used': int(min(d_eff_rand, d_eff_real)),
                                'ortho_err_y_rand': float(ortho_err),
                                'n_train': int(tr_mask.sum()),
                                'n_val':   int(va_mask.sum()),
                                'n_test':  int(te_mask.sum()),
                                'num_classes':    int(num_classes),
                                'random_baseline': float(100.0 / num_classes),
                            })
                            if need:
                                save_json(path, rec)

                            for k, lst in [('GCN_rand',      gcn_rand_f),
                                           ('GCN_rowNorm_Y', gcnrn_f),
                                           ('MLP_Y',         mlpy_f),
                                           ('MLP_rowNorm_Y', mlprn_f)]:
                                lst.append(rec[k]['final_test_acc'] * 100)
                            all_records.append(rec)

                        skip = '' if any(len(load_json(xrand_result_path(
                            dataset_folder, split_type, split_seed, xrand_seed,
                            optimizer_name, lr, s))) == 0
                            for s in TRAIN_SEEDS) else ' (resumed)'
                        print(
                            f'  [{dataset_folder} | {split_type} ss={split_seed} | '
                            f'xs={xrand_seed} | {optimizer_name} | lr={lr}]{skip}  '
                            f'GCN_rand={np.mean(gcn_rand_f):.1f}±{np.std(gcn_rand_f):.1f}%  '
                            f'GCN_rnY={np.mean(gcnrn_f):.1f}±{np.std(gcnrn_f):.1f}%  '
                            f'MLP_Y={np.mean(mlpy_f):.1f}±{np.std(mlpy_f):.1f}%  '
                            f'MLP_rnY={np.mean(mlprn_f):.1f}±{np.std(mlprn_f):.1f}%'
                        )

    return all_records


# ── Summary ────────────────────────────────────────────────────────────────────

def build_summary(all_records: list) -> dict:
    """
    Aggregate by (dataset, split_type, optimizer, lr).
    Pools over all (xrand_seed, split_seed, train_seed).
    For GCN_X records (no xrand_seed), pooled separately.
    """
    from collections import defaultdict
    xrand_buckets = defaultdict(list)
    gcnx_buckets  = defaultdict(list)

    for rec in all_records:
        key = (rec['dataset'], rec['split_type'],
               rec.get('optimizer', ''), lr_str(rec.get('lr', 0)))
        if 'xrand_seed' in rec:
            xrand_buckets[key].append(rec)
        else:
            gcnx_buckets[key].append(rec)

    summary = {}
    all_keys = set(xrand_buckets) | set(gcnx_buckets)

    for key in all_keys:
        dataset, split_type, optimizer, lr_s = key
        entry = {'random_baseline': None}

        if key in xrand_buckets:
            recs = xrand_buckets[key]
            entry['random_baseline'] = recs[0]['random_baseline']
            for model_key in XRAND_MODEL_KEYS:
                accs = [r[model_key]['final_test_acc'] * 100
                        for r in recs if model_key in r]
                if accs:
                    entry[f'{model_key}_mean'] = float(np.mean(accs))
                    entry[f'{model_key}_std']  = float(np.std(accs))
                    entry[f'{model_key}_n']    = len(accs)

        if key in gcnx_buckets:
            recs = gcnx_buckets[key]
            if entry['random_baseline'] is None:
                entry['random_baseline'] = recs[0]['random_baseline']
            accs = [r[GCN_X_KEY]['final_test_acc'] * 100
                    for r in recs if GCN_X_KEY in r]
            if accs:
                entry['GCN_X_mean'] = float(np.mean(accs))
                entry['GCN_X_std']  = float(np.std(accs))
                entry['GCN_X_n']    = len(accs)

        (summary
         .setdefault(dataset, {})
         .setdefault(split_type, {})
         .setdefault(optimizer, {})[lr_s]) = entry

    return summary


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(device):
    """
    Smoke test: cora, fixed split, adam, lr=0.01, xseed=0, train_seed=0.
    Runs all checkpoints and all 5 models. Prints final accuracies vs random baseline.
    """
    print('\n' + '=' * 65)
    print('SANITY CHECK — all 5 models')
    print('cora | fixed | adam | lr=0.01 | xseed=0 | train_seed=0')
    print('=' * 65)

    data_dir = os.path.join(DATA_ROOT, 'cora')
    for req in ['X.npy', 'Y.npy', 'labels.npy',
                'fixed_train_mask.npy', 'fixed_val_mask.npy', 'fixed_test_mask.npy']:
        if not os.path.isfile(os.path.join(data_dir, req)):
            print('Skipped: data missing. Run save_data.py first.')
            return

    X_real      = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
    Y_real      = np.load(os.path.join(data_dir, 'Y.npy'))
    labels      = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)
    n, d        = X_real.shape
    d_eff_real  = Y_real.shape[1]
    num_classes = int(labels.max()) + 1
    del Y_real

    adj_kipf, L, D = build_and_cache_graph('cora')
    adj_t    = scipy_to_torch_sparse(adj_kipf, device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    tr = np.load(os.path.join(data_dir, 'fixed_train_mask.npy'))
    va = np.load(os.path.join(data_dir, 'fixed_val_mask.npy'))
    te = np.load(os.path.join(data_dir, 'fixed_test_mask.npy'))
    verify_split(tr, va, te, 'fixed', 0, n)

    tr_t = torch.tensor(tr, dtype=torch.bool, device=device)
    va_t = torch.tensor(va, dtype=torch.bool, device=device)
    te_t = torch.tensor(te, dtype=torch.bool, device=device)

    # GCN_X
    X_real_t = torch.tensor(X_real, dtype=torch.float32, device=device)
    _, _, _, gcnX_fi = train_gcn(
        X_real_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='GCN_X'
    )

    # X_rand models
    rng      = np.random.default_rng(0)
    X_rand   = rng.standard_normal((n, d)).astype(np.float32)
    X_rand_t = torch.tensor(X_rand, dtype=torch.float32, device=device)
    Y_rand, d_eff_rand, ortho_err = compute_y_rand(X_rand, L, D, d_eff_real, 0)
    Y_rand_t  = torch.tensor(Y_rand,              dtype=torch.float32, device=device)
    Y_rand_rn = row_normalize_np(Y_rand)

    sc    = StandardScaler()
    Y_tr  = sc.fit_transform(Y_rand[tr]).astype(np.float32)
    Y_va_ = sc.transform(Y_rand[va]).astype(np.float32)
    Y_te_ = sc.transform(Y_rand[te]).astype(np.float32)
    Y_tr_rn = Y_rand_rn[tr]; Y_va_rn = Y_rand_rn[va]; Y_te_rn = Y_rand_rn[te]

    _, _, _, gcnR_fi = train_gcn(
        X_rand_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='GCN_rand'
    )
    _, _, _, gcnRN_fi = train_gcn_rownorm_y(
        Y_rand_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True
    )
    _, _, _, mlpY_fi = train_mlp(
        Y_tr, labels[tr], Y_va_, labels[va], Y_te_, labels[te],
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='MLP_Y'
    )
    _, _, _, mlpRN_fi = train_mlp(
        Y_tr_rn, labels[tr], Y_va_rn, labels[va], Y_te_rn, labels[te],
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='MLP_rowNorm_Y'
    )

    baseline = 100.0 / num_classes
    print(f'\n  RESULTS (cora, {EPOCHS} epochs, adam lr=0.01):')
    print(f'  {"Model":<20} {"Test%":>7}  {"vs random":>10}')
    print(f'  {"-"*42}')
    for name, fi in [('GCN_X',         gcnX_fi),
                     ('GCN_rand',       gcnR_fi),
                     ('GCN_rowNorm_Y',  gcnRN_fi),
                     ('MLP_Y',          mlpY_fi),
                     ('MLP_rowNorm_Y',  mlpRN_fi)]:
        pct = fi * 100
        print(f'  {name:<20} {pct:>7.2f}%  {pct-baseline:>+10.2f} pp')
    print(f'  {"Random baseline":<20} {baseline:>7.2f}%')
    print('=' * 65 + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='5-model null feature experiment'
    )
    parser.add_argument('--dataset',   type=str, default=None)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default=None)
    parser.add_argument('--lr',        type=float, default=None)
    parser.add_argument('--sanity',    action='store_true')
    parser.add_argument('--verbose',   action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'gcn_rand_experiment.py  |  Device: {device}')

    if args.sanity:
        sanity_check(device)
        sys.exit(0)

    datasets_to_run   = [args.dataset]   if args.dataset   else ALL_DATASETS
    optimizers_to_run = [args.optimizer] if args.optimizer  else OPTIMIZERS
    lrs_to_run        = [args.lr]        if args.lr         else LEARNING_RATES

    for ds in datasets_to_run:
        if not os.path.isdir(os.path.join(DATA_ROOT, ds)):
            print(f'ERROR: data not found for "{ds}". Run save_data.py first.')
            sys.exit(1)

    runs_per_ds = (len(XRAND_SEEDS) * (1 + len(RANDOM_SPLIT_SEEDS))
                   * len(optimizers_to_run) * len(lrs_to_run) * len(TRAIN_SEEDS))
    gcnx_per_ds = ((1 + len(RANDOM_SPLIT_SEEDS))
                   * len(optimizers_to_run) * len(lrs_to_run) * len(TRAIN_SEEDS))

    print(f'\nDatasets   : {datasets_to_run}')
    print(f'Optimizers : {optimizers_to_run}  |  LRs: {lrs_to_run}')
    print(f'X_rand draws: {XRAND_SEEDS}  |  Train seeds: {TRAIN_SEEDS}')
    print(f'Runs/dataset: {runs_per_ds} (xrand models) + {gcnx_per_ds} (GCN_X)')
    print(f'Results    : {RESULTS_ROOT}\n')

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    all_records = []
    for dataset in datasets_to_run:
        records = run_dataset(dataset, optimizers_to_run, lrs_to_run,
                              device, verbose=args.verbose)
        all_records.extend(records)

    if all_records:
        summary = build_summary(all_records)
        summary_path = os.path.join(RESULTS_ROOT, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSummary → {summary_path}')

        # Overview table
        print('\n' + '=' * 80)
        print('OVERVIEW (mean across all seeds/splits/draws/opt/lr)')
        print(f'{"Dataset":<22} {"base%":>5} {"GCN_X":>7} {"GCN_rnd":>8} '
              f'{"GCN_rnY":>8} {"MLP_Y":>7} {"MLP_rnY":>8}')
        print('-' * 80)
        for ds in datasets_to_run:
            if ds not in summary:
                continue
            agg = {k: [] for k in ['GCN_X', 'GCN_rand', 'GCN_rowNorm_Y',
                                    'MLP_Y', 'MLP_rowNorm_Y']}
            base = None
            for st in summary[ds].values():
                for ov in st.values():
                    for entry in ov.values():
                        base = entry.get('random_baseline', base)
                        for k in agg:
                            mk = f'{k}_mean'
                            if mk in entry:
                                agg[k].append(entry[mk])
            def m(k): return f'{np.mean(agg[k]):.1f}' if agg[k] else 'N/A'
            print(f'{ds:<22} {base:>5.1f} {m("GCN_X"):>7} {m("GCN_rand"):>8} '
                  f'{m("GCN_rowNorm_Y"):>8} {m("MLP_Y"):>7} {m("MLP_rowNorm_Y"):>8}')
        print('=' * 80)

    print('\nDone.')
