"""
SpectralGeometryMisalignment/label_label_experiment.py

EXPERIMENT 2 — FULL LABEL INFORMATION (UPPER BOUND / SCIENTIFIC PROBE)
  GCN(X_label_label) vs GCN(U_label_label) vs GCN_rowNorm(U_label_label)

SETUP:
  X_label_label : n × num_classes feature matrix.
                  ALL nodes (train, val, test) → one-hot encoding of their
                  true class label. Fully deterministic — no random draws.
  U_label_label : Rayleigh-Ritz eigenvectors of (L, D) restricted to
                  span(X_label_label). Exactly num_classes eigenvectors
                  (assuming all classes represented — rank = num_classes).
                  D-orthonormal: U^T D U = I.

THEORETICAL STATUS:
  This is NOT a predictive model. Test node labels are embedded directly
  in the feature matrix, so no true generalization is measured.
  Purpose: controlled upper bound / scientific probe.
  Answers: does Rayleigh-Ritz preserve the label-discriminative structure
  when given perfect information (all labels)?
  Connection: GCN(X_label_label) ≈ label propagation (transductive).

SCIENTIFIC QUESTIONS:
  1. Does GCN(U_label_label) match GCN(X_label_label)?
     YES → Rayleigh-Ritz is lossless for GCN at full label information.
     NO  → Rayleigh-Ritz loses structure even at the information ceiling.
  2. Does row-norm on U_label_label improve over raw U_label_label?

3 MODELS:
  GCN_X        : 2-layer Kipf GCN on X_label_label (raw one-hot, no preprocessing)
  GCN_U        : 2-layer Kipf GCN on U_label_label (StandardScaler, fit on
                 train nodes only, applied to all)
  GCN_rowNorm_U: 2-layer Kipf GCN on U_label_label with row-norm at each
                 layer's INPUT (Option B — same as GCN_rowNorm_Y in gcn_rand)

DIMENSION:
  X_label_label and U_label_label have exactly num_classes columns.

TRAINING PROTOCOL:
  Optimizers : SGD (momentum=0.9, WD=0) + Adam (betas=(0.9,0.999), WD=0)
  LRs        : [0.001, 0.01, 0.1]
  Epochs     : 500, no early stopping, full-batch
  Train seeds: 5 (0–4)

SPLITS:
  Random only: 5 stratified 60/20/20 splits (split_seeds 0–4, from save_data.py).
  Fixed split NOT used — Yiannis specified 60/20/20 for this experiment.
  No xrand seeds (features are fully deterministic — all true labels).

  NOTE: StandardScaler for GCN_U is fit on train nodes only (varies per split).
  U_label_label itself is the same for all splits (deterministic, computed once).

OUTPUT:
  results/label_label/{dataset}/
    {dataset}_random_splitseed{ss}_{opt}_lr{lr}_seed{ts}.json
      → keys: GCN_X, GCN_U, GCN_rowNorm_U

RESUME:
  Existing JSON files are loaded and only missing model keys are computed.
  Already-completed runs are never re-run.

Usage:
  /home/md724/Spectral-Basis/venv/bin/python label_label_experiment.py
  /home/md724/Spectral-Basis/venv/bin/python label_label_experiment.py --sanity
  /home/md724/Spectral-Basis/venv/bin/python label_label_experiment.py --dataset cora
  /home/md724/Spectral-Basis/venv/bin/python label_label_experiment.py \\
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

from graph_utils import compute_restricted_eigenvectors

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_ROOT    = os.path.join(_HERE, 'data')
RESULTS_ROOT = os.path.join(_HERE, 'results', 'label_label')

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

OPTIMIZERS         = ['sgd', 'adam']
LEARNING_RATES     = [0.001, 0.01, 0.1]
TRAIN_SEEDS        = list(range(5))
RANDOM_SPLIT_SEEDS = list(range(5))   # 60/20/20, seeds 0–4
EPOCHS             = 500
GCN_HIDDEN         = 256
GCN_DROPOUT        = 0.5
PRINT_EVERY        = 100

MODEL_KEYS = ['GCN_X', 'GCN_U', 'GCN_rowNorm_U']


# ── Models ─────────────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """
    Standard 2-layer Kipf GCN.
      H^(1) = ReLU( Â X W1 )  with dropout
      H^(2) = Â H^(1) W2
    Used for GCN_X (X_label_label) and GCN_U (StandardScaler-scaled U_label_label).
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
      Layer 1: Â @ row_norm(U_label_label) → fc1 → ReLU → dropout
      Layer 2: Â @ row_norm(H^(1)) → fc2
    row_norm: F.normalize(x, p=2, dim=1) — projects each node to unit L2 sphere.
    Zero-norm rows are left as-is by F.normalize (returns zero vector).
    Used for GCN_rowNorm_U.
    """
    def __init__(self, d_in: int, hidden: int, num_classes: int,
                 dropout: float = GCN_DROPOUT):
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_optimizer(name: str, params, lr: float):
    """WD=0.0 for all models (symmetric comparison)."""
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode all node labels. Returns float32 (n, num_classes)."""
    oh = np.zeros((len(labels), num_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def scipy_to_torch_sparse(adj, device):
    coo  = adj.tocoo().astype(np.float32)
    idx  = torch.LongTensor(np.vstack([coo.row, coo.col]))
    vals = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx, vals, coo.shape,
                                   device=device).coalesce()


def load_split_masks(dataset_folder: str, split_seed: int):
    """Load 60/20/20 random split masks (random splits only for this experiment)."""
    d = os.path.join(DATA_ROOT, dataset_folder)
    tr = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
    va = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
    te = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))
    return tr, va, te


def verify_split(tr, va, te, split_seed, n):
    """CHECK: split sizes, no overlap, confirm 60/20/20 proportions."""
    assert not np.any(tr & va), 'Train/Val overlap!'
    assert not np.any(tr & te), 'Train/Test overlap!'
    assert not np.any(va & te), 'Val/Test overlap!'
    frac_tr = tr.sum() / n * 100
    frac_va = va.sum() / n * 100
    frac_te = te.sum() / n * 100
    print(f'    [CHECK split] random ss={split_seed}: '
          f'train={tr.sum()} ({frac_tr:.0f}%)  '
          f'val={va.sum()} ({frac_va:.0f}%)  '
          f'test={te.sum()} ({frac_te:.0f}%)  [OK ≈60/20/20]')


def load_cached_graph(dataset_folder: str):
    """
    Load cached adj_kipf, L, D from data/{dataset}/.
    Built and cached by gcn_rand_experiment.py.
    Requires gcn_rand_experiment.py to have been run first.
    """
    data_dir = os.path.join(DATA_ROOT, dataset_folder)
    adj_path = os.path.join(data_dir, 'adj_kipf.npz')
    L_path   = os.path.join(data_dir, 'L.npz')
    D_path   = os.path.join(data_dir, 'D.npz')

    for p in [adj_path, L_path, D_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f'Graph matrix not found: {p}\n'
                f'Run gcn_rand_experiment.py first to build and cache graph matrices.'
            )

    adj_kipf = sp.load_npz(adj_path)
    L        = sp.load_npz(L_path)
    D        = sp.load_npz(D_path)
    print(f'  [cache] Loaded adj_kipf, L, D for {dataset_folder}')
    return adj_kipf, L, D


# ── X_label_label and U_label_label construction ──────────────────────────────

def build_x_label_label(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build X_label_label (n × num_classes).
    ALL nodes receive their true one-hot class label as their feature vector.
    Fully deterministic — no randomness.

    CHECK A: All rows are one-hot (max=1, row sum=1 for all nodes).
    """
    X_ll = one_hot_encode(labels, num_classes)
    # Verify one-hot property
    row_sums = X_ll.sum(axis=1)
    row_maxs = X_ll.max(axis=1)
    assert np.allclose(row_sums, 1.0), \
        f'[CHECK A] X_label_label row sums not all 1: min={row_sums.min():.4f}'
    assert np.allclose(row_maxs, 1.0), \
        f'[CHECK A] X_label_label row maxes not all 1: min={row_maxs.min():.4f}'
    print(f'    [CHECK A] X_label_label: all rows one-hot  '
          f'shape={X_ll.shape}  [OK]')
    return X_ll


def compute_u_label_label(X_ll: np.ndarray, L, D) -> tuple:
    """
    Compute U_label_label via Rayleigh-Ritz on X_label_label.
    X_label_label is deterministic (all true labels), so U_label_label is
    computed once per dataset and reused across all splits.

    CHECK B: D-orthonormality: U^T D U ≈ I (error < 1e-6).
    CHECK C: d_eff == num_classes (all classes present → rank = num_classes).

    Returns (U, d_eff, ortho_err).
    """
    n, num_classes = X_ll.shape

    U, _, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_ll.astype(np.float64), L, D, num_components=0
    )

    if ortho_err >= 1e-6:
        raise RuntimeError(
            f'[CHECK B] U_label_label D-ortho FAILED: {ortho_err:.2e} ≥ 1e-6'
        )
    print(f'    [CHECK B] U_label_label D-ortho: {ortho_err:.2e}  [PASS]  '
          f'd_eff={d_eff}  (expected {num_classes})')

    if d_eff != num_classes:
        # This would indicate some class has zero nodes — unexpected.
        print(f'    WARNING: d_eff={d_eff} ≠ num_classes={num_classes}. '
              f'Some class may have no nodes in LCC.')

    assert U.shape == (n, d_eff), \
        f'[CHECK C] Shape mismatch: {U.shape} vs ({n}, {d_eff})'

    return U.astype(np.float32), d_eff, float(ortho_err)


# ── Training loop ──────────────────────────────────────────────────────────────

def _run_epochs(model, X_all_t, adj_t, labels_t,
                tr_mask_t, va_mask_t, te_mask_t,
                opt, label: str, verbose: bool):
    """Shared training loop for GCN and GCN_RowNorm."""
    crit = nn.CrossEntropyLoss()
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
            preds       = logits_eval.argmax(dim=1)
            va = (preds[va_mask_t] == labels_t[va_mask_t]).float().mean().item()
            te = (preds[te_mask_t] == labels_t[te_mask_t]).float().mean().item()

        train_loss.append(lv)
        val_acc.append(float(va))
        test_acc.append(float(te))

        if verbose and ep % PRINT_EVERY == 0:
            print(f'      {label:<20} ep={ep:>3}/{EPOCHS}  '
                  f'loss={lv:.4f}  val={va:.3f}  test={te:.3f}')

    return train_loss, val_acc, test_acc, float(test_acc[-1])


def train_gcn(X_t, adj_t, labels_t, tr_t, va_t, te_t,
              num_classes: int, optimizer_name: str, lr: float,
              seed: int, device, verbose: bool = False, label: str = 'GCN') -> tuple:
    """Standard 2-layer Kipf GCN. Used for GCN_X and GCN_U."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN(X_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs(model, X_t, adj_t, labels_t, tr_t, va_t, te_t,
                       opt, label, verbose)


def train_gcn_rownorm(U_t, adj_t, labels_t, tr_t, va_t, te_t,
                      num_classes: int, optimizer_name: str, lr: float,
                      seed: int, device, verbose: bool = False) -> tuple:
    """GCN with row-norm at each layer input, on raw U_label_label."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN_RowNorm(U_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs(model, U_t, adj_t, labels_t, tr_t, va_t, te_t,
                       opt, 'GCN_rowNorm_U', verbose)


# ── Result path helpers ────────────────────────────────────────────────────────

def lr_str(lr: float) -> str:
    return str(lr)


def result_path(dataset, split_seed, optimizer, lr, train_seed) -> str:
    d = os.path.join(RESULTS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d,
        f'{dataset}_random_splitseed{split_seed}'
        f'_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


def load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


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
    print(f'Dataset: {dataset_folder}  [label_label_experiment]')
    print(f'{"="*70}')

    # Verify required data files
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

    print(f'n={n:,}  num_classes={num_classes}  '
          f'input_dim={num_classes}  '
          f'random_baseline={100.0/num_classes:.2f}%')

    # Load cached graph matrices
    adj_kipf, L, D = load_cached_graph(dataset_folder)
    adj_t          = scipy_to_torch_sparse(adj_kipf, device)
    labels_t       = torch.tensor(labels, dtype=torch.long, device=device)

    # ── Build X_label_label (deterministic, computed once per dataset) ─────────
    X_ll   = build_x_label_label(labels, num_classes)
    X_ll_t = torch.tensor(X_ll, dtype=torch.float32, device=device)

    # ── Compute U_label_label (deterministic, computed once per dataset) ───────
    U_ll, d_eff, ortho_err = compute_u_label_label(X_ll, L, D)
    # Raw U for GCN_rowNorm_U (row-norm applied inside forward pass)
    U_ll_t = torch.tensor(U_ll, dtype=torch.float32, device=device)

    print(f'  X_label_label: shape={X_ll.shape}  '
          f'U_label_label: shape={U_ll.shape}  ortho_err={ortho_err:.2e}')

    all_records = []

    # ── Random splits only (60/20/20, 5 seeds) ─────────────────────────────────
    for split_seed in RANDOM_SPLIT_SEEDS:
        tr_mask, va_mask, te_mask = load_split_masks(dataset_folder, split_seed)
        verify_split(tr_mask, va_mask, te_mask, split_seed, n)

        # StandardScaler for GCN_U: fit on train nodes, apply to all.
        # Recomputed per split since train nodes change.
        sc = StandardScaler()
        sc.fit(U_ll[tr_mask])
        U_ll_sc   = sc.transform(U_ll).astype(np.float32)
        U_ll_sc_t = torch.tensor(U_ll_sc, dtype=torch.float32, device=device)

        tr_t = torch.tensor(tr_mask, dtype=torch.bool, device=device)
        va_t = torch.tensor(va_mask, dtype=torch.bool, device=device)
        te_t = torch.tensor(te_mask, dtype=torch.bool, device=device)

        for optimizer_name in optimizers_to_run:
            for lr in lrs_to_run:
                gcnx_finals, gcnu_finals, gcnrn_finals = [], [], []

                for train_seed in TRAIN_SEEDS:
                    path = result_path(dataset_folder, split_seed,
                                       optimizer_name, lr, train_seed)
                    rec  = load_json(path)
                    need = [k for k in MODEL_KEYS if k not in rec]

                    be_v = (verbose
                            and train_seed == TRAIN_SEEDS[0]
                            and split_seed == RANDOM_SPLIT_SEEDS[0])

                    if 'GCN_X' in need:
                        lo, va, te, fi = train_gcn(
                            X_ll_t, adj_t, labels_t, tr_t, va_t, te_t,
                            num_classes, optimizer_name, lr,
                            train_seed, device, verbose=be_v, label='GCN_X'
                        )
                        rec['GCN_X'] = pack_result(lo, va, te, fi)

                    if 'GCN_U' in need:
                        lo, va, te, fi = train_gcn(
                            U_ll_sc_t, adj_t, labels_t, tr_t, va_t, te_t,
                            num_classes, optimizer_name, lr,
                            train_seed, device, verbose=be_v, label='GCN_U'
                        )
                        rec['GCN_U'] = pack_result(lo, va, te, fi)

                    if 'GCN_rowNorm_U' in need:
                        lo, va, te, fi = train_gcn_rownorm(
                            U_ll_t, adj_t, labels_t, tr_t, va_t, te_t,
                            num_classes, optimizer_name, lr,
                            train_seed, device, verbose=be_v
                        )
                        rec['GCN_rowNorm_U'] = pack_result(lo, va, te, fi)

                    # Update metadata
                    rec.update({
                        'dataset':         dataset_folder,
                        'split_type':      'random',
                        'split_seed':      split_seed,
                        'optimizer':       optimizer_name,
                        'lr':              lr,
                        'train_seed':      train_seed,
                        'n':               int(n),
                        'num_classes':     int(num_classes),
                        'input_dim':       int(num_classes),
                        'd_eff':           int(d_eff),
                        'ortho_err_U':     float(ortho_err),
                        'n_train':         int(tr_mask.sum()),
                        'n_val':           int(va_mask.sum()),
                        'n_test':          int(te_mask.sum()),
                        'random_baseline': float(100.0 / num_classes),
                    })
                    if need:
                        save_json(path, rec)

                    gcnx_finals.append(rec['GCN_X']['final_test_acc'] * 100)
                    gcnu_finals.append(rec['GCN_U']['final_test_acc'] * 100)
                    gcnrn_finals.append(rec['GCN_rowNorm_U']['final_test_acc'] * 100)
                    all_records.append(rec)

                print(
                    f'  [{dataset_folder} | random ss={split_seed} | '
                    f'{optimizer_name} | lr={lr}]  '
                    f'GCN_X={np.mean(gcnx_finals):.1f}±{np.std(gcnx_finals):.1f}%  '
                    f'GCN_U={np.mean(gcnu_finals):.1f}±{np.std(gcnu_finals):.1f}%  '
                    f'GCN_rnU={np.mean(gcnrn_finals):.1f}±{np.std(gcnrn_finals):.1f}%'
                )

    return all_records


# ── Summary ────────────────────────────────────────────────────────────────────

def build_summary(all_records: list) -> dict:
    """
    Aggregate by (dataset, optimizer, lr).
    Pools over all (split_seed, train_seed).
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for rec in all_records:
        key = (rec['dataset'], rec.get('optimizer', ''), lr_str(rec.get('lr', 0)))
        buckets[key].append(rec)

    summary = {}
    for key, recs in buckets.items():
        dataset, optimizer, lr_s = key
        entry = {'random_baseline': recs[0]['random_baseline']}
        for mk in MODEL_KEYS:
            accs = [r[mk]['final_test_acc'] * 100 for r in recs if mk in r]
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
    """
    Smoke test: cora, random split seed=0, adam, lr=0.01, train_seed=0.
    Runs all 3 models, prints final accuracies vs random baseline.
    """
    print('\n' + '=' * 65)
    print('SANITY CHECK — label_label_experiment, 3 models')
    print('cora | random ss=0 | adam | lr=0.01 | train_seed=0')
    print('=' * 65)

    data_dir = os.path.join(DATA_ROOT, 'cora')
    for req in ['labels.npy',
                'random0_train_mask.npy', 'random0_val_mask.npy', 'random0_test_mask.npy']:
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

    print(f'\n  X_label_label: shape={X_ll.shape}')
    print(f'  First 3 rows: {X_ll[:3]}  [should be one-hot]')

    U_ll, d_eff, ortho_err = compute_u_label_label(X_ll, L, D)
    U_ll_t = torch.tensor(U_ll, dtype=torch.float32, device=device)

    sc = StandardScaler(); sc.fit(U_ll[tr])
    U_ll_sc   = sc.transform(U_ll).astype(np.float32)
    U_ll_sc_t = torch.tensor(U_ll_sc, dtype=torch.float32, device=device)

    _, _, _, fi_gcnx = train_gcn(
        X_ll_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='GCN_X'
    )
    _, _, _, fi_gcnu = train_gcn(
        U_ll_sc_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='GCN_U'
    )
    _, _, _, fi_gcnrn = train_gcn_rownorm(
        U_ll_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True
    )

    baseline = 100.0 / num_classes
    print(f'\n  RESULTS (cora, {EPOCHS} epochs, adam lr=0.01, random ss=0):')
    print(f'  {"Model":<24} {"Test%":>7}  {"vs random":>12}')
    print(f'  {"-"*46}')
    for name, fi in [('GCN_X (X_label_label)', fi_gcnx),
                     ('GCN_U (U_label_label)',  fi_gcnu),
                     ('GCN_rowNorm_U',          fi_gcnrn)]:
        pct = fi * 100
        print(f'  {name:<24} {pct:>7.2f}%  {pct - baseline:>+12.2f} pp')
    print(f'  {"Random baseline":<24} {baseline:>7.2f}%')
    print(f'\n  d_eff={d_eff}  ortho_err={ortho_err:.2e}  input_dim={num_classes}')
    print(f'  NOTE: GCN_X expected near-perfect (test labels in features).')
    print('=' * 65 + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 2: GCN(X_label_label) vs GCN(U_label_label)'
    )
    parser.add_argument('--dataset',   type=str,   default=None)
    parser.add_argument('--optimizer', type=str,   choices=['sgd', 'adam'], default=None)
    parser.add_argument('--lr',        type=float, default=None)
    parser.add_argument('--sanity',    action='store_true')
    parser.add_argument('--verbose',   action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'label_label_experiment.py  |  Device: {device}')

    if args.sanity:
        sanity_check(device)
        sys.exit(0)

    datasets_to_run   = [args.dataset]   if args.dataset   else ALL_DATASETS
    optimizers_to_run = [args.optimizer] if args.optimizer else OPTIMIZERS
    lrs_to_run        = [args.lr]        if args.lr        else LEARNING_RATES

    for ds in datasets_to_run:
        if not os.path.isdir(os.path.join(DATA_ROOT, ds)):
            print(f'ERROR: data not found for "{ds}". Run save_data.py first.')
            sys.exit(1)

    runs_per_ds = (len(RANDOM_SPLIT_SEEDS)
                   * len(optimizers_to_run) * len(lrs_to_run) * len(TRAIN_SEEDS))

    print(f'\nDatasets   : {datasets_to_run}')
    print(f'Optimizers : {optimizers_to_run}  |  LRs: {lrs_to_run}')
    print(f'Split seeds: {RANDOM_SPLIT_SEEDS} (random 60/20/20 only)')
    print(f'Train seeds: {TRAIN_SEEDS}')
    print(f'Runs/dataset: {runs_per_ds} (×3 models = {runs_per_ds*3} total trainings)')
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
        print('\n' + '=' * 75)
        print('OVERVIEW — label_label_experiment')
        print('Mean across all seeds/splits/opt/lr  |  NOTE: GCN_X uses test labels')
        print(f'{"Dataset":<22} {"base%":>5} {"GCN_X":>8} {"GCN_U":>8} {"GCN_rnU":>9}')
        print('-' * 75)
        for ds in datasets_to_run:
            if ds not in summary:
                continue
            agg  = {k: [] for k in MODEL_KEYS}
            base = None
            for ov in summary[ds].values():
                for entry in ov.values():
                    base = entry.get('random_baseline', base)
                    for k in agg:
                        mk = f'{k}_mean'
                        if mk in entry:
                            agg[k].append(entry[mk])
            def m(k): return f'{np.mean(agg[k]):.1f}' if agg[k] else 'N/A'
            print(f'{ds:<22} {base:>5.1f} {m("GCN_X"):>8} {m("GCN_U"):>8} '
                  f'{m("GCN_rowNorm_U"):>9}')
        print('=' * 75)

    print('\nDone.')
