"""
SpectralGeometryMisalignment/label_label_v2_experiment.py

EXPERIMENT 2-v2 — FULL LABEL INFORMATION + GNN vs TRANSFORMER COMPARISON
  (revised per Yiannis, April 2026)

CHANGE FROM v1:
  Adds 3 Transformer models alongside the 3 existing GCN models.
  GCN uses graph adjacency (local message passing).
  Transformer uses global self-attention — no graph structure.
  Same features (X_label_label, U_label_label), different architectures.

SCIENTIFIC QUESTION (new):
  Does the spectral reparameterization (U_label_label) interact differently
  with global attention (Transformer) vs local graph propagation (GCN)?
  Specifically: does U pre-spreading labels globally make the Transformer's
  job easier, harder, or the same compared to what it does for GCN?

SETUP (unchanged from v1):
  X_label_label : ALL nodes → one-hot class label. Deterministic.
  U_label_label : Rayleigh-Ritz of (L,D) on X_label_label. Computed once.
  span(X) = span(U) → all gaps are optimization geometry, not information.

6 MODELS:
  GCN_X           : 2-layer Kipf GCN on X_label_label
  GCN_U           : 2-layer Kipf GCN on U_label_label (StandardScaler)
  GCN_rowNorm_U   : 2-layer Kipf GCN on U_label_label (row-norm, Option B)
  Transformer_X   : 2-layer Transformer Encoder on X_label_label
  Transformer_U   : 2-layer Transformer Encoder on U_label_label (StandardScaler)
  Transformer_rowNorm_U : 2-layer Transformer Encoder on row-norm(U_label_label)

TRANSFORMER ARCHITECTURE:
  2-layer TransformerEncoder (pre-norm, multi-head self-attention).
  hidden_dim=256, n_heads=4 (64 dim per head), feedforward=512, dropout=0.5.
  No positional encoding — nodes have no natural order.
  No graph adjacency — pure global self-attention.
  Transformer_rowNorm_U: L2 row-normalization applied at the input only
    (before the first layer). The Transformer has its own LayerNorm internally;
    applying row-norm at every internal layer would conflict with that.

SCALABILITY:
  Full self-attention is O(N^2) memory. For large graphs this is infeasible.
  Threshold: FULLATT_MAX_NODES = 25000.
  Datasets with n > FULLATT_MAX_NODES: Transformer models stored as
    {'skipped': True, 'reason': 'n > FULLATT_MAX_NODES', 'n': ...}
  Feasibility by dataset (approximate):
    cora (~2.7K)           ✓   amazon_photo (~7.7K)    ✓
    citeseer (~2.1K)       ✓   amazon_computers (~13.7K) ✓
    wikics (~11.7K)        ✓   pubmed (~19.7K)         ✓ (borderline)
    coauthor_cs (~18K)     ✓ (borderline)
    coauthor_physics (~34K) ✗  ogbn_arxiv (~169K)      ✗

TRAINING (unchanged from v1):
  Optimizers : SGD (momentum=0.9, WD=0) + Adam (betas=(0.9,0.999), WD=0)
  LRs        : [0.001, 0.01, 0.1]
  Epochs     : 500, no early stopping, full-batch
  Loss       : CrossEntropyLoss on training nodes only.
  Train seeds: 5 (0–4)
  Splits     : 5 random 60/20/20 splits (seeds 0–4)

OUTPUT:
  results/label_label_v2/{dataset}/
    {dataset}_random_splitseed{ss}_{opt}_lr{lr}_seed{ts}.json
      → keys: GCN_X, GCN_U, GCN_rowNorm_U,
               Transformer_X, Transformer_U, Transformer_rowNorm_U

Usage:
  /home/md724/Spectral-Basis/venv/bin/python label_label_v2_experiment.py
  /home/md724/Spectral-Basis/venv/bin/python label_label_v2_experiment.py --sanity
  /home/md724/Spectral-Basis/venv/bin/python label_label_v2_experiment.py --dataset cora
  /home/md724/Spectral-Basis/venv/bin/python label_label_v2_experiment.py --gcn_only
  /home/md724/Spectral-Basis/venv/bin/python label_label_v2_experiment.py --transformer_only
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

from graph_utils import compute_restricted_eigenvectors

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_ROOT    = os.path.join(_HERE, 'data')
RESULTS_ROOT = os.path.join(_HERE, 'results', 'label_label_v2')

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

OPTIMIZERS         = ['sgd', 'adam']
LEARNING_RATES     = [0.001, 0.01, 0.1]
TRAIN_SEEDS        = list(range(5))
RANDOM_SPLIT_SEEDS = list(range(5))
EPOCHS             = 500
PRINT_EVERY        = 100

# GCN hyperparameters
GCN_HIDDEN  = 256
GCN_DROPOUT = 0.5

# Transformer hyperparameters
TRANS_HIDDEN  = 256    # must be divisible by TRANS_HEADS
TRANS_HEADS   = 4      # 256 / 4 = 64 dim per head
TRANS_FF_DIM  = 512    # feedforward dimension inside transformer
TRANS_LAYERS  = 2
TRANS_DROPOUT = 0.5

# Graphs with n > this skip full self-attention (O(N^2) memory infeasible)
FULLATT_MAX_NODES = 25000

GCN_KEYS   = ['GCN_X', 'GCN_U', 'GCN_rowNorm_U']
TRANS_KEYS = ['Transformer_X', 'Transformer_U', 'Transformer_rowNorm_U']
MODEL_KEYS = GCN_KEYS + TRANS_KEYS


# ── GCN Models ────────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """Standard 2-layer Kipf GCN."""
    def __init__(self, d_in, hidden, num_classes, dropout=GCN_DROPOUT):
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
    """2-layer Kipf GCN with row-norm at each layer's INPUT (Option B)."""
    def __init__(self, d_in, hidden, num_classes, dropout=GCN_DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(d_in, hidden)
        self.fc2  = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        x_rn = F.normalize(x, p=2, dim=1)
        h    = F.relu(self.fc1(torch.sparse.mm(adj, x_rn)))
        h    = self.drop(h)
        h_rn = F.normalize(h, p=2, dim=1)
        out  = self.fc2(torch.sparse.mm(adj, h_rn))
        return out


# ── Transformer Model ─────────────────────────────────────────────────────────

class NodeTransformer(nn.Module):
    """
    2-layer Transformer Encoder for node classification.
    Uses global self-attention over all N nodes — no graph adjacency.

    Architecture:
      1. Input projection: d_in → hidden_dim
      2. TransformerEncoder (2 layers, pre-norm):
           each layer: LayerNorm → MultiHeadAttention → residual
                       LayerNorm → FeedForward → residual
      3. Output classifier: hidden_dim → num_classes

    No positional encoding — nodes have no natural sequential order.
    The classifier is applied per-node to the output representations.

    row_norm (optional): if True, L2-normalize input before projection.
      This is the Transformer_rowNorm_U variant.
      Row-norm is applied at the INPUT only (before layer 1).
      The internal LayerNorm handles subsequent normalization needs.
    """
    def __init__(self, d_in, hidden, num_classes,
                 n_heads=TRANS_HEADS, ff_dim=TRANS_FF_DIM,
                 n_layers=TRANS_LAYERS, dropout=TRANS_DROPOUT,
                 row_norm=False):
        super().__init__()
        assert hidden % n_heads == 0, \
            f'hidden_dim={hidden} must be divisible by n_heads={n_heads}'
        self.row_norm   = row_norm
        self.input_proj = nn.Linear(d_in, hidden)
        self.input_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)
        self.classifier  = nn.Linear(hidden, num_classes)

    def forward(self, x):
        """
        x : (N, d_in) — all nodes
        returns logits : (N, num_classes)
        """
        if self.row_norm:
            x = F.normalize(x, p=2, dim=1)

        h = self.input_drop(self.input_proj(x))   # (N, hidden)
        h = h.unsqueeze(0)                         # (1, N, hidden) — batch of 1
        h = self.transformer(h)                    # (1, N, hidden)
        h = h.squeeze(0)                           # (N, hidden)
        return self.classifier(h)                  # (N, num_classes)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_optimizer(name, params, lr):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)


def one_hot_encode(labels, num_classes):
    oh = np.zeros((len(labels), num_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def scipy_to_torch_sparse(adj, device):
    coo  = adj.tocoo().astype(np.float32)
    idx  = torch.LongTensor(np.vstack([coo.row, coo.col]))
    vals = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx, vals, coo.shape,
                                   device=device).coalesce()


def load_split_masks(dataset_folder, split_seed):
    d = os.path.join(DATA_ROOT, dataset_folder)
    tr = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
    va = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
    te = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))
    return tr, va, te


def verify_split(tr, va, te, split_seed, n):
    assert not np.any(tr & va) and not np.any(tr & te) and not np.any(va & te)
    frac_tr = tr.sum() / n * 100
    print(f'    [CHECK split] random ss={split_seed}: '
          f'train={tr.sum()} ({frac_tr:.0f}%)  '
          f'val={va.sum()}  test={te.sum()}  [OK ≈60/20/20]')


def load_cached_graph(dataset_folder):
    data_dir = os.path.join(DATA_ROOT, dataset_folder)
    for name in ['adj_kipf.npz', 'L.npz', 'D.npz']:
        if not os.path.isfile(os.path.join(data_dir, name)):
            raise FileNotFoundError(
                f'Graph matrix not found: {os.path.join(data_dir, name)}\n'
                f'Run gcn_rand_experiment.py first.'
            )
    adj_kipf = sp.load_npz(os.path.join(data_dir, 'adj_kipf.npz'))
    L        = sp.load_npz(os.path.join(data_dir, 'L.npz'))
    D        = sp.load_npz(os.path.join(data_dir, 'D.npz'))
    print(f'  [cache] Loaded adj_kipf, L, D for {dataset_folder}')
    return adj_kipf, L, D


# ── Feature construction ──────────────────────────────────────────────────────

def build_x_label_label(labels, num_classes):
    oh = one_hot_encode(labels, num_classes)
    assert np.allclose(oh.sum(1), 1.0) and np.allclose(oh.max(1), 1.0)
    print(f'    [CHECK] X_label_label: all rows one-hot  shape={oh.shape}  [OK]')
    return oh


def compute_u_label_label(X_ll, L, D):
    n, num_classes = X_ll.shape
    U, _, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_ll.astype(np.float64), L, D, num_components=0
    )
    if ortho_err >= 1e-6:
        raise RuntimeError(f'U D-ortho FAILED: {ortho_err:.2e}')
    print(f'    [CHECK] U_label_label D-ortho: {ortho_err:.2e}  [PASS]  '
          f'd_eff={d_eff}  (expected {num_classes})')
    if d_eff != num_classes:
        print(f'    WARNING: d_eff={d_eff} ≠ num_classes={num_classes}.')
    assert U.shape == (n, d_eff)
    return U.astype(np.float32), d_eff, float(ortho_err)


# ── Training loops ─────────────────────────────────────────────────────────────

def _run_epochs_gcn(model, X_t, adj_t, labels_t,
                    tr_t, va_t, te_t, opt, label, verbose):
    """Training loop for GCN / GCN_RowNorm."""
    crit = nn.CrossEntropyLoss()
    train_loss, val_acc, test_acc = [], [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_t, adj_t)
        loss   = crit(logits[tr_t], labels_t[tr_t])
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
            preds = model(X_t, adj_t).argmax(dim=1)
            va = (preds[va_t] == labels_t[va_t]).float().mean().item()
            te = (preds[te_t] == labels_t[te_t]).float().mean().item()

        train_loss.append(lv)
        val_acc.append(float(va))
        test_acc.append(float(te))

        if verbose and ep % PRINT_EVERY == 0:
            print(f'      {label:<24} ep={ep:>3}/{EPOCHS}  '
                  f'loss={lv:.4f}  val={va:.3f}  test={te:.3f}')

    return train_loss, val_acc, test_acc, float(test_acc[-1])


def _run_epochs_transformer(model, X_t, labels_t,
                             tr_t, va_t, te_t, opt, label, verbose):
    """
    Training loop for NodeTransformer.
    No adj_t — Transformer does not use graph structure.
    """
    crit = nn.CrossEntropyLoss()
    train_loss, val_acc, test_acc = [], [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_t)                           # (N, num_classes)
        loss   = crit(logits[tr_t], labels_t[tr_t])  # train nodes only
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
            preds = model(X_t).argmax(dim=1)
            va = (preds[va_t] == labels_t[va_t]).float().mean().item()
            te = (preds[te_t] == labels_t[te_t]).float().mean().item()

        train_loss.append(lv)
        val_acc.append(float(va))
        test_acc.append(float(te))

        if verbose and ep % PRINT_EVERY == 0:
            print(f'      {label:<24} ep={ep:>3}/{EPOCHS}  '
                  f'loss={lv:.4f}  val={va:.3f}  test={te:.3f}')

    return train_loss, val_acc, test_acc, float(test_acc[-1])


def train_gcn(X_t, adj_t, labels_t, tr_t, va_t, te_t,
              num_classes, optimizer_name, lr, seed, device,
              verbose=False, label='GCN'):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN(X_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs_gcn(model, X_t, adj_t, labels_t, tr_t, va_t, te_t,
                           opt, label, verbose)


def train_gcn_rownorm(U_t, adj_t, labels_t, tr_t, va_t, te_t,
                      num_classes, optimizer_name, lr, seed, device, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN_RowNorm(U_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs_gcn(model, U_t, adj_t, labels_t, tr_t, va_t, te_t,
                           opt, 'GCN_rowNorm_U', verbose)


def train_transformer(X_t, labels_t, tr_t, va_t, te_t,
                      num_classes, optimizer_name, lr, seed, device,
                      verbose=False, label='Transformer', row_norm=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = NodeTransformer(X_t.shape[1], TRANS_HIDDEN, num_classes,
                            row_norm=row_norm).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs_transformer(model, X_t, labels_t, tr_t, va_t, te_t,
                                   opt, label, verbose)


# ── Result path helpers ────────────────────────────────────────────────────────

def lr_str(lr):
    return str(lr)


def result_path(dataset, split_seed, optimizer, lr, train_seed):
    d = os.path.join(RESULTS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d,
        f'{dataset}_random_splitseed{split_seed}'
        f'_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


def load_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path, record):
    with open(path, 'w') as f:
        json.dump(record, f)


def pack_result(loss_curve, val_curve, test_curve, final_acc):
    return {
        'train_loss_curve': loss_curve,
        'val_acc_curve':    val_curve,
        'test_acc_curve':   test_curve,
        'final_test_acc':   final_acc,
    }


def pack_skipped(reason, n):
    """Placeholder for Transformer models skipped due to graph size."""
    return {'skipped': True, 'reason': reason, 'n': int(n)}


# ── Core experiment loop ───────────────────────────────────────────────────────

def run_dataset(dataset_folder, optimizers_to_run, lrs_to_run, device,
                run_gcn=True, run_transformer=True, verbose=False):
    print(f'\n{"="*70}')
    print(f'Dataset: {dataset_folder}  [label_label_v2_experiment]')
    print(f'{"="*70}')

    data_dir = os.path.join(DATA_ROOT, dataset_folder)
    required = ['labels.npy']
    for s in RANDOM_SPLIT_SEEDS:
        for sfx in ['train', 'val', 'test']:
            required.append(f'random{s}_{sfx}_mask.npy')
    missing = [f for f in required if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        print(f'ERROR: Missing files: {missing}. Run save_data.py first.')
        return []

    labels      = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)
    n           = len(labels)
    num_classes = int(labels.max()) + 1

    transformer_feasible = (n <= FULLATT_MAX_NODES)
    print(f'n={n:,}  num_classes={num_classes}  '
          f'random_baseline={100.0/num_classes:.2f}%')
    if run_transformer:
        if transformer_feasible:
            print(f'  Transformer: ENABLED  (n={n:,} ≤ {FULLATT_MAX_NODES:,})')
        else:
            print(f'  Transformer: SKIPPED  (n={n:,} > {FULLATT_MAX_NODES:,} '
                  f'— O(N^2) attention infeasible)')

    adj_kipf, L, D = load_cached_graph(dataset_folder)
    adj_t          = scipy_to_torch_sparse(adj_kipf, device)
    labels_t       = torch.tensor(labels, dtype=torch.long, device=device)

    # ── Build features (deterministic, once per dataset) ──────────────────────
    X_ll   = build_x_label_label(labels, num_classes)
    X_ll_t = torch.tensor(X_ll, dtype=torch.float32, device=device)

    U_ll, d_eff, ortho_err = compute_u_label_label(X_ll, L, D)
    U_ll_t = torch.tensor(U_ll, dtype=torch.float32, device=device)

    print(f'  X_label_label={X_ll.shape}  U_label_label={U_ll.shape}  '
          f'ortho_err={ortho_err:.2e}')

    all_records = []

    for split_seed in RANDOM_SPLIT_SEEDS:
        tr_mask, va_mask, te_mask = load_split_masks(dataset_folder, split_seed)
        verify_split(tr_mask, va_mask, te_mask, split_seed, n)

        # StandardScaler fit on train nodes for GCN_U and Transformer_U
        sc = StandardScaler()
        sc.fit(U_ll[tr_mask])
        U_ll_sc   = sc.transform(U_ll).astype(np.float32)
        U_ll_sc_t = torch.tensor(U_ll_sc, dtype=torch.float32, device=device)

        tr_t = torch.tensor(tr_mask, dtype=torch.bool, device=device)
        va_t = torch.tensor(va_mask, dtype=torch.bool, device=device)
        te_t = torch.tensor(te_mask, dtype=torch.bool, device=device)

        for optimizer_name in optimizers_to_run:
            for lr in lrs_to_run:
                finals = {k: [] for k in MODEL_KEYS}

                for train_seed in TRAIN_SEEDS:
                    path = result_path(dataset_folder, split_seed,
                                       optimizer_name, lr, train_seed)
                    rec  = load_json(path)

                    # Determine which keys still need to be computed
                    gcn_need   = [k for k in GCN_KEYS
                                  if k not in rec and run_gcn]
                    trans_need = [k for k in TRANS_KEYS
                                  if k not in rec and run_transformer]

                    be_v = (verbose
                            and train_seed == TRAIN_SEEDS[0]
                            and split_seed == RANDOM_SPLIT_SEEDS[0])

                    # ── GCN models ──────────────────────────────────────────
                    if 'GCN_X' in gcn_need:
                        lo, va, te, fi = train_gcn(
                            X_ll_t, adj_t, labels_t, tr_t, va_t, te_t,
                            num_classes, optimizer_name, lr,
                            train_seed, device, verbose=be_v, label='GCN_X'
                        )
                        rec['GCN_X'] = pack_result(lo, va, te, fi)

                    if 'GCN_U' in gcn_need:
                        lo, va, te, fi = train_gcn(
                            U_ll_sc_t, adj_t, labels_t, tr_t, va_t, te_t,
                            num_classes, optimizer_name, lr,
                            train_seed, device, verbose=be_v, label='GCN_U'
                        )
                        rec['GCN_U'] = pack_result(lo, va, te, fi)

                    if 'GCN_rowNorm_U' in gcn_need:
                        lo, va, te, fi = train_gcn_rownorm(
                            U_ll_t, adj_t, labels_t, tr_t, va_t, te_t,
                            num_classes, optimizer_name, lr,
                            train_seed, device, verbose=be_v
                        )
                        rec['GCN_rowNorm_U'] = pack_result(lo, va, te, fi)

                    # ── Transformer models ──────────────────────────────────
                    if trans_need:
                        if not transformer_feasible:
                            # Mark all Transformer keys as skipped
                            skip_rec = pack_skipped(
                                f'n={n} > FULLATT_MAX_NODES={FULLATT_MAX_NODES}', n
                            )
                            for k in TRANS_KEYS:
                                if k not in rec:
                                    rec[k] = skip_rec
                        else:
                            if 'Transformer_X' in trans_need:
                                lo, va, te, fi = train_transformer(
                                    X_ll_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v,
                                    label='Transformer_X', row_norm=False
                                )
                                rec['Transformer_X'] = pack_result(lo, va, te, fi)

                            if 'Transformer_U' in trans_need:
                                lo, va, te, fi = train_transformer(
                                    U_ll_sc_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v,
                                    label='Transformer_U', row_norm=False
                                )
                                rec['Transformer_U'] = pack_result(lo, va, te, fi)

                            if 'Transformer_rowNorm_U' in trans_need:
                                lo, va, te, fi = train_transformer(
                                    U_ll_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v,
                                    label='Transformer_rowNorm_U', row_norm=True
                                )
                                rec['Transformer_rowNorm_U'] = pack_result(lo, va, te, fi)

                    # ── Metadata ────────────────────────────────────────────
                    rec.update({
                        'dataset':               dataset_folder,
                        'split_type':            'random',
                        'split_seed':            split_seed,
                        'optimizer':             optimizer_name,
                        'lr':                    lr,
                        'train_seed':            train_seed,
                        'n':                     int(n),
                        'num_classes':           int(num_classes),
                        'input_dim':             int(num_classes),
                        'd_eff':                 int(d_eff),
                        'ortho_err_U':           float(ortho_err),
                        'n_train':               int(tr_mask.sum()),
                        'n_val':                 int(va_mask.sum()),
                        'n_test':                int(te_mask.sum()),
                        'random_baseline':       float(100.0 / num_classes),
                        'transformer_feasible':  bool(transformer_feasible),
                        'fullatt_max_nodes':     FULLATT_MAX_NODES,
                    })

                    if gcn_need or trans_need:
                        save_json(path, rec)

                    # Collect finals for summary line
                    for k in MODEL_KEYS:
                        if k in rec and not rec[k].get('skipped', False):
                            finals[k].append(rec[k]['final_test_acc'] * 100)

                    all_records.append(rec)

                # Print summary line for this (split, opt, lr)
                def fmt(k):
                    v = finals[k]
                    if not v: return '  skip'
                    return f'{np.mean(v):.1f}±{np.std(v):.1f}%'

                gcn_str   = '  '.join(f'{k.replace("_rowNorm","")}={fmt(k)}'
                                      for k in GCN_KEYS)
                trans_str = '  '.join(f'{k.replace("Transformer_","T_")}={fmt(k)}'
                                      for k in TRANS_KEYS)
                print(f'  [{dataset_folder}|ss={split_seed}|{optimizer_name}|lr={lr}]  '
                      f'{gcn_str}  |  {trans_str}')

    return all_records


# ── Summary ────────────────────────────────────────────────────────────────────

def build_summary(all_records):
    from collections import defaultdict
    buckets = defaultdict(list)
    for rec in all_records:
        key = (rec['dataset'], rec.get('optimizer',''), lr_str(rec.get('lr',0)))
        buckets[key].append(rec)

    summary = {}
    for key, recs in buckets.items():
        dataset, optimizer, lr_s = key
        entry = {
            'random_baseline':       recs[0]['random_baseline'],
            'transformer_feasible':  recs[0].get('transformer_feasible', False),
        }
        for mk in MODEL_KEYS:
            accs = [r[mk]['final_test_acc'] * 100 for r in recs
                    if mk in r and not r[mk].get('skipped', False)]
            if accs:
                entry[f'{mk}_mean'] = float(np.mean(accs))
                entry[f'{mk}_std']  = float(np.std(accs))
                entry[f'{mk}_n']    = len(accs)
        (summary
         .setdefault(dataset, {})
         .setdefault(optimizer, {})[lr_s]) = entry
    return summary


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(device):
    print('\n' + '=' * 65)
    print('SANITY CHECK — label_label_v2_experiment, 6 models')
    print('cora | random ss=0 | adam | lr=0.01 | train_seed=0')
    print('=' * 65)

    data_dir = os.path.join(DATA_ROOT, 'cora')
    for req in ['labels.npy', 'random0_train_mask.npy',
                'random0_val_mask.npy', 'random0_test_mask.npy']:
        if not os.path.isfile(os.path.join(data_dir, req)):
            print(f'Skipped: data missing ({req}). Run save_data.py first.')
            return

    labels      = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)
    n           = len(labels)
    num_classes = int(labels.max()) + 1
    tr = np.load(os.path.join(data_dir, 'random0_train_mask.npy'))
    va = np.load(os.path.join(data_dir, 'random0_val_mask.npy'))
    te = np.load(os.path.join(data_dir, 'random0_test_mask.npy'))
    verify_split(tr, va, te, 0, n)

    adj_kipf, L, D = load_cached_graph('cora')
    adj_t    = scipy_to_torch_sparse(adj_kipf, device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    tr_t = torch.tensor(tr, dtype=torch.bool, device=device)
    va_t = torch.tensor(va, dtype=torch.bool, device=device)
    te_t = torch.tensor(te, dtype=torch.bool, device=device)

    X_ll   = build_x_label_label(labels, num_classes)
    X_ll_t = torch.tensor(X_ll, dtype=torch.float32, device=device)

    U_ll, d_eff, ortho_err = compute_u_label_label(X_ll, L, D)
    U_ll_t = torch.tensor(U_ll, dtype=torch.float32, device=device)
    sc = StandardScaler(); sc.fit(U_ll[tr])
    U_ll_sc_t = torch.tensor(sc.transform(U_ll).astype(np.float32),
                              dtype=torch.float32, device=device)

    results = {}
    for name, fn, kwargs in [
        ('GCN_X',               train_gcn,         dict(X_t=X_ll_t,   adj_t=adj_t, label='GCN_X')),
        ('GCN_U',               train_gcn,         dict(X_t=U_ll_sc_t,adj_t=adj_t, label='GCN_U')),
        ('GCN_rowNorm_U',       train_gcn_rownorm, dict(U_t=U_ll_t,   adj_t=adj_t)),
        ('Transformer_X',       train_transformer, dict(X_t=X_ll_t,              label='Transformer_X',       row_norm=False)),
        ('Transformer_U',       train_transformer, dict(X_t=U_ll_sc_t,           label='Transformer_U',       row_norm=False)),
        ('Transformer_rowNorm_U',train_transformer,dict(X_t=U_ll_t,              label='Transformer_rowNorm_U',row_norm=True)),
    ]:
        if 'adj_t' in kwargs and fn == train_gcn:
            _, _, _, fi = fn(kwargs['X_t'], kwargs['adj_t'], labels_t, tr_t, va_t, te_t,
                             num_classes, 'adam', 0.01, 0, device, verbose=True,
                             label=kwargs.get('label', 'GCN'))
        elif fn == train_gcn_rownorm:
            _, _, _, fi = fn(kwargs['U_t'], kwargs['adj_t'], labels_t, tr_t, va_t, te_t,
                             num_classes, 'adam', 0.01, 0, device, verbose=True)
        else:
            _, _, _, fi = fn(kwargs['X_t'], labels_t, tr_t, va_t, te_t,
                             num_classes, 'adam', 0.01, 0, device, verbose=True,
                             label=kwargs.get('label','T'), row_norm=kwargs.get('row_norm',False))
        results[name] = fi * 100

    baseline = 100.0 / num_classes
    print(f'\n  RESULTS (cora, {EPOCHS} epochs, adam lr=0.01, random ss=0):')
    print(f'  {"Model":<28} {"Test%":>7}  {"vs random":>12}')
    print(f'  {"-"*50}')
    for name, acc in results.items():
        print(f'  {name:<28} {acc:>7.2f}%  {acc - baseline:>+12.2f} pp')
    print(f'  {"Random baseline":<28} {baseline:>7.2f}%')
    print(f'\n  d_eff={d_eff}  ortho_err={ortho_err:.2e}  '
          f'Transformer: hidden={TRANS_HIDDEN} heads={TRANS_HEADS} layers={TRANS_LAYERS}')
    print(f'  NOTE: GCN_X expected near-perfect (test labels in features).')
    print('=' * 65 + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 2-v2: GCN vs Transformer on X/U_label_label'
    )
    parser.add_argument('--dataset',          type=str,   default=None)
    parser.add_argument('--optimizer',        type=str,   choices=['sgd', 'adam'], default=None)
    parser.add_argument('--lr',               type=float, default=None)
    parser.add_argument('--sanity',           action='store_true')
    parser.add_argument('--verbose',          action='store_true')
    parser.add_argument('--gcn_only',         action='store_true',
                        help='Run only GCN models (skip Transformer)')
    parser.add_argument('--transformer_only', action='store_true',
                        help='Run only Transformer models (skip GCN)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'label_label_v2_experiment.py  |  Device: {device}')
    print(f'v2: GCN + Transformer comparison on X_label_label / U_label_label')
    print(f'Transformer: hidden={TRANS_HIDDEN} heads={TRANS_HEADS} '
          f'layers={TRANS_LAYERS} ff={TRANS_FF_DIM} dropout={TRANS_DROPOUT}')
    print(f'Full-attention threshold: n ≤ {FULLATT_MAX_NODES:,} nodes')

    if args.sanity:
        sanity_check(device)
        sys.exit(0)

    run_gcn         = not args.transformer_only
    run_transformer = not args.gcn_only

    datasets_to_run   = [args.dataset]   if args.dataset   else ALL_DATASETS
    optimizers_to_run = [args.optimizer] if args.optimizer else OPTIMIZERS
    lrs_to_run        = [args.lr]        if args.lr        else LEARNING_RATES

    for ds in datasets_to_run:
        if not os.path.isdir(os.path.join(DATA_ROOT, ds)):
            print(f'ERROR: data not found for "{ds}". Run save_data.py first.')
            sys.exit(1)

    n_models    = (3 if run_gcn else 0) + (3 if run_transformer else 0)
    runs_per_ds = (len(RANDOM_SPLIT_SEEDS)
                   * len(optimizers_to_run) * len(lrs_to_run) * len(TRAIN_SEEDS))

    print(f'\nDatasets   : {datasets_to_run}')
    print(f'Optimizers : {optimizers_to_run}  |  LRs: {lrs_to_run}')
    print(f'Split seeds: {RANDOM_SPLIT_SEEDS} (random 60/20/20 only)')
    print(f'Train seeds: {TRAIN_SEEDS}')
    print(f'Models     : {n_models} per run (GCN={run_gcn}, Transformer={run_transformer})')
    print(f'Runs/dataset: {runs_per_ds} (×{n_models} = {runs_per_ds*n_models} total trainings)')
    print(f'Results    : {RESULTS_ROOT}\n')

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    all_records = []
    for dataset in datasets_to_run:
        records = run_dataset(dataset, optimizers_to_run, lrs_to_run, device,
                              run_gcn=run_gcn, run_transformer=run_transformer,
                              verbose=args.verbose)
        all_records.extend(records)

    if all_records:
        summary = build_summary(all_records)
        summary_path = os.path.join(RESULTS_ROOT, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSummary → {summary_path}')

        print('\n' + '=' * 90)
        print('OVERVIEW — label_label_v2_experiment  (NOTE: test labels in features)')
        print(f'{"Dataset":<22} {"base%":>5} '
              f'{"GCN_X":>8} {"GCN_U":>8} {"GCN_rnU":>9} | '
              f'{"Trans_X":>9} {"Trans_U":>9} {"T_rnU":>7}')
        print('-' * 90)
        for ds in datasets_to_run:
            if ds not in summary:
                continue
            agg  = {k: [] for k in MODEL_KEYS}
            base = None
            t_feasible = None
            for ov in summary[ds].values():
                for entry in ov.values():
                    base       = entry.get('random_baseline', base)
                    t_feasible = entry.get('transformer_feasible', t_feasible)
                    for k in agg:
                        mk = f'{k}_mean'
                        if mk in entry:
                            agg[k].append(entry[mk])

            def m(k):
                if not agg[k]:
                    return 'skip' if not t_feasible else 'N/A'
                return f'{np.mean(agg[k]):.1f}'

            print(f'{ds:<22} {base:>5.1f} '
                  f'{m("GCN_X"):>8} {m("GCN_U"):>8} {m("GCN_rowNorm_U"):>9} | '
                  f'{m("Transformer_X"):>9} {m("Transformer_U"):>9} '
                  f'{m("Transformer_rowNorm_U"):>7}')
        print('=' * 90)

    print('\nDone.')
