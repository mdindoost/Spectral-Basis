"""
SpectralGeometryMisalignment/label_rand_v2_experiment.py

EXPERIMENT 1-v2 — PARTIAL LABEL INFORMATION (revised per Yiannis, April 2026)

CHANGE FROM v1:
  Val/test random features are now a RANDOM PROBABILITY VECTOR (uniform simplex)
  instead of N(0,1) Gaussian noise.

  OLD: val/test → rng.standard_normal((n, num_classes))
  NEW: val/test → u ~ Uniform(0,1)^num_classes, then u / u.sum()
                  Each row sums to 1 — lies on the probability simplex.

  MOTIVATION (Yiannis): the random features should be probability vectors,
  matching the same space as the one-hot training features. N(0,1) features
  can have negative entries and unconstrained row norms, which is inconsistent
  with a label-probability interpretation.

  Loss is defined only on training labels (unchanged from v1 — already correct).

SETUP:
  X_label_rand_v2 : n × num_classes feature matrix.
                    Train nodes → one-hot class label.
                    Val/test nodes → random probability vector (sums to 1).
  U_label_rand_v2 : Rayleigh-Ritz eigenvectors of (L, D) restricted to
                    span(X_label_rand_v2). At most num_classes eigenvectors.
                    D-orthonormal: U^T D U = I.

THEORETICAL GUARANTEE:
  span(X_label_rand_v2) = span(U_label_rand_v2) by Rayleigh-Ritz construction.
  Any gap between GCN(X) and GCN(U) is optimization geometry, not information.

3 MODELS:
  GCN_X        : 2-layer Kipf GCN on X_label_rand_v2 (no preprocessing)
  GCN_U        : 2-layer Kipf GCN on U_label_rand_v2 (StandardScaler, train only)
  GCN_rowNorm_U: 2-layer Kipf GCN on U_label_rand_v2, row-norm at each
                 layer's INPUT (Option B)

TRAINING PROTOCOL: identical to v1.
  Optimizers : SGD (momentum=0.9, WD=0) + Adam (betas=(0.9,0.999), WD=0)
  LRs        : [0.001, 0.01, 0.1]
  Epochs     : 500, no early stopping, full-batch
  Loss       : CrossEntropyLoss on training nodes only.
  Train seeds: 5 (0–4)

XRAND SEEDS: 3 draws (xrand_seed in {0,1,2}).
SPLITS: 1 fixed + 5 random (60/20/20, seeds 0–4). Identical to v1.

OUTPUT:
  results/label_rand_v2/{dataset}/
    {dataset}_{split_type}_splitseed{ss}_xseed{xs}_{opt}_lr{lr}_seed{ts}.json
      → keys: GCN_X, GCN_U, GCN_rowNorm_U

Usage:
  /home/md724/Spectral-Basis/venv/bin/python label_rand_v2_experiment.py
  /home/md724/Spectral-Basis/venv/bin/python label_rand_v2_experiment.py --sanity
  /home/md724/Spectral-Basis/venv/bin/python label_rand_v2_experiment.py --dataset cora
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
RESULTS_ROOT = os.path.join(_HERE, 'results', 'label_rand_v2')

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

OPTIMIZERS         = ['sgd', 'adam']
LEARNING_RATES     = [0.001, 0.01, 0.1]
TRAIN_SEEDS        = list(range(5))
XRAND_SEEDS        = list(range(3))
RANDOM_SPLIT_SEEDS = list(range(5))
EPOCHS             = 500
GCN_HIDDEN         = 256
GCN_DROPOUT        = 0.5
PRINT_EVERY        = 100

MODEL_KEYS = ['GCN_X', 'GCN_U', 'GCN_rowNorm_U']


# ── Models ─────────────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """Standard 2-layer Kipf GCN. Used for GCN_X and GCN_U."""
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
    """
    2-layer Kipf GCN with row normalization at each layer's INPUT (Option B).
      Layer 1: Â @ row_norm(U) → fc1 → ReLU → dropout
      Layer 2: Â @ row_norm(H^(1)) → fc2
    """
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_optimizer(name, params, lr):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)


def one_hot_encode(labels_subset, num_classes):
    oh = np.zeros((len(labels_subset), num_classes), dtype=np.float32)
    oh[np.arange(len(labels_subset)), labels_subset] = 1.0
    return oh


def scipy_to_torch_sparse(adj, device):
    coo  = adj.tocoo().astype(np.float32)
    idx  = torch.LongTensor(np.vstack([coo.row, coo.col]))
    vals = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx, vals, coo.shape,
                                   device=device).coalesce()


def load_split_masks(dataset_folder, split_type, split_seed):
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
    assert not np.any(tr & va), 'Train/Val overlap!'
    assert not np.any(tr & te), 'Train/Test overlap!'
    assert not np.any(va & te), 'Val/Test overlap!'
    print(f'    [CHECK split] {split_type} ss={split_seed}: '
          f'train={tr.sum()}  val={va.sum()}  test={te.sum()}  '
          f'total={tr.sum()+va.sum()+te.sum()}/{n}  [OK]')


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


# ── X_label_rand_v2 construction ──────────────────────────────────────────────

def build_x_label_rand_v2(n, num_classes, labels, tr_mask, xrand_seed):
    """
    Build X_label_rand_v2 (n × num_classes).
      Train nodes    → one-hot encoding of their true label.
      Val/test nodes → random probability vector (uniform simplex).

    Probability simplex: sample u ~ Uniform(0,1)^num_classes, divide by row sum.
    This ensures each val/test feature row sums to 1 — same simplex as one-hot,
    but uninformative about the true class.

    CHECK A: val/test rows sum to 1 and are non-negative.
    CHECK B: train rows are one-hot (max=1, sum=1).
    """
    rng = np.random.default_rng(xrand_seed)

    # Sample uniform [0,1] for all nodes, normalize rows to sum to 1
    u    = rng.uniform(0.0, 1.0, size=(n, num_classes)).astype(np.float64)
    u   /= u.sum(axis=1, keepdims=True)          # each row sums to 1
    X_lr = u.astype(np.float32)

    # Override train nodes with true one-hot labels
    X_lr[tr_mask] = one_hot_encode(labels[tr_mask], num_classes)

    # CHECK A — val/test rows non-negative and sum to 1
    non_train = ~tr_mask
    if non_train.sum() > 0:
        row_sums = X_lr[non_train].sum(axis=1)
        assert np.all(X_lr[non_train] >= 0), '[CHECK A] Negative val/test features!'
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f'[CHECK A] Val/test rows not summing to 1: max_err={np.abs(row_sums-1).max():.2e}'

    # CHECK B — train rows are one-hot
    assert np.allclose(X_lr[tr_mask].sum(axis=1), 1.0, atol=1e-5), \
        '[CHECK B] Train rows not summing to 1!'
    assert np.allclose(X_lr[tr_mask].max(axis=1), 1.0, atol=1e-5), \
        '[CHECK B] Train rows not one-hot!'

    return X_lr


def compute_u_label_rand_v2(X_lr, L, D, xrand_seed, split_label):
    """Rayleigh-Ritz on X_label_rand_v2. D-orthonormality checked."""
    n, num_classes = X_lr.shape
    U, _, d_eff, ortho_err = compute_restricted_eigenvectors(
        X_lr.astype(np.float64), L, D, num_components=0
    )
    if ortho_err >= 1e-6:
        raise RuntimeError(
            f'[CHECK] U D-ortho FAILED (xseed={xrand_seed}, {split_label}): '
            f'{ortho_err:.2e} ≥ 1e-6'
        )
    print(f'    [CHECK] U D-ortho: {ortho_err:.2e}  [PASS]  '
          f'd_eff={d_eff}/{num_classes}  (xseed={xrand_seed}, {split_label})')
    assert U.shape == (n, d_eff)
    return U.astype(np.float32), d_eff, float(ortho_err)


# ── Training loop ──────────────────────────────────────────────────────────────

def _run_epochs(model, X_all_t, adj_t, labels_t,
                tr_mask_t, va_mask_t, te_mask_t,
                opt, label, verbose):
    crit = nn.CrossEntropyLoss()
    train_loss, val_acc, test_acc = [], [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_all_t, adj_t)
        loss   = crit(logits[tr_mask_t], labels_t[tr_mask_t])   # train nodes only
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
              num_classes, optimizer_name, lr, seed, device,
              verbose=False, label='GCN'):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN(X_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs(model, X_t, adj_t, labels_t, tr_t, va_t, te_t,
                       opt, label, verbose)


def train_gcn_rownorm(U_t, adj_t, labels_t, tr_t, va_t, te_t,
                      num_classes, optimizer_name, lr, seed, device, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN_RowNorm(U_t.shape[1], GCN_HIDDEN, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    return _run_epochs(model, U_t, adj_t, labels_t, tr_t, va_t, te_t,
                       opt, 'GCN_rowNorm_U', verbose)


# ── Result path helpers ────────────────────────────────────────────────────────

def lr_str(lr):
    return str(lr)


def result_path(dataset, split_type, split_seed, xrand_seed, optimizer, lr, train_seed):
    d = os.path.join(RESULTS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d,
        f'{dataset}_{split_type}_splitseed{split_seed}'
        f'_xseed{xrand_seed}_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


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


# ── Core experiment loop ───────────────────────────────────────────────────────

def run_dataset(dataset_folder, optimizers_to_run, lrs_to_run, device, verbose=False):
    print(f'\n{"="*70}')
    print(f'Dataset: {dataset_folder}  [label_rand_v2_experiment]')
    print(f'{"="*70}')

    data_dir = os.path.join(DATA_ROOT, dataset_folder)
    required = ['labels.npy',
                'fixed_train_mask.npy', 'fixed_val_mask.npy', 'fixed_test_mask.npy']
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
          f'X_dim={num_classes}  '
          f'random_baseline={100.0/num_classes:.2f}%')
    print(f'  [v2] Val/test features: random probability vector (uniform simplex)')

    adj_kipf, L, D = load_cached_graph(dataset_folder)
    adj_t          = scipy_to_torch_sparse(adj_kipf, device)
    labels_t       = torch.tensor(labels, dtype=torch.long, device=device)

    all_records = []

    for xrand_seed in XRAND_SEEDS:
        print(f'\n  {"─"*60}')
        print(f'  xrand_seed={xrand_seed}')
        print(f'  {"─"*60}')

        for split_type in ['fixed', 'random']:
            split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS

            for split_seed in split_seeds:
                tr_mask, va_mask, te_mask = load_split_masks(
                    dataset_folder, split_type, split_seed
                )
                verify_split(tr_mask, va_mask, te_mask, split_type, split_seed, n)

                split_label = f'{split_type}_ss{split_seed}'

                X_lr   = build_x_label_rand_v2(n, num_classes, labels,
                                               tr_mask, xrand_seed)
                X_lr_t = torch.tensor(X_lr, dtype=torch.float32, device=device)

                U_lr, d_eff, ortho_err = compute_u_label_rand_v2(
                    X_lr, L, D, xrand_seed, split_label
                )

                sc = StandardScaler()
                sc.fit(U_lr[tr_mask])
                U_lr_sc   = sc.transform(U_lr).astype(np.float32)
                U_lr_sc_t = torch.tensor(U_lr_sc, dtype=torch.float32, device=device)

                U_lr_t = torch.tensor(U_lr, dtype=torch.float32, device=device)

                tr_t = torch.tensor(tr_mask, dtype=torch.bool, device=device)
                va_t = torch.tensor(va_mask, dtype=torch.bool, device=device)
                te_t = torch.tensor(te_mask, dtype=torch.bool, device=device)

                for optimizer_name in optimizers_to_run:
                    for lr in lrs_to_run:
                        gcnx_finals, gcnu_finals, gcnrn_finals = [], [], []

                        for train_seed in TRAIN_SEEDS:
                            path = result_path(dataset_folder, split_type, split_seed,
                                               xrand_seed, optimizer_name, lr, train_seed)
                            rec  = load_json(path)
                            need = [k for k in MODEL_KEYS if k not in rec]

                            be_v = (verbose
                                    and train_seed == TRAIN_SEEDS[0]
                                    and split_type == 'fixed'
                                    and xrand_seed == XRAND_SEEDS[0])

                            if 'GCN_X' in need:
                                lo, va, te, fi = train_gcn(
                                    X_lr_t, adj_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v, label='GCN_X'
                                )
                                rec['GCN_X'] = pack_result(lo, va, te, fi)

                            if 'GCN_U' in need:
                                lo, va, te, fi = train_gcn(
                                    U_lr_sc_t, adj_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v, label='GCN_U'
                                )
                                rec['GCN_U'] = pack_result(lo, va, te, fi)

                            if 'GCN_rowNorm_U' in need:
                                lo, va, te, fi = train_gcn_rownorm(
                                    U_lr_t, adj_t, labels_t, tr_t, va_t, te_t,
                                    num_classes, optimizer_name, lr,
                                    train_seed, device, verbose=be_v
                                )
                                rec['GCN_rowNorm_U'] = pack_result(lo, va, te, fi)

                            rec.update({
                                'dataset':          dataset_folder,
                                'split_type':       split_type,
                                'split_seed':       split_seed,
                                'xrand_seed':       xrand_seed,
                                'optimizer':        optimizer_name,
                                'lr':               lr,
                                'train_seed':       train_seed,
                                'n':                int(n),
                                'num_classes':      int(num_classes),
                                'input_dim':        int(num_classes),
                                'd_eff':            int(d_eff),
                                'ortho_err_U':      float(ortho_err),
                                'n_train':          int(tr_mask.sum()),
                                'n_val':            int(va_mask.sum()),
                                'n_test':           int(te_mask.sum()),
                                'random_baseline':  float(100.0 / num_classes),
                                'rand_feature_type': 'simplex',  # v2 marker
                            })
                            if need:
                                save_json(path, rec)

                            gcnx_finals.append(rec['GCN_X']['final_test_acc'] * 100)
                            gcnu_finals.append(rec['GCN_U']['final_test_acc'] * 100)
                            gcnrn_finals.append(rec['GCN_rowNorm_U']['final_test_acc'] * 100)
                            all_records.append(rec)

                        print(
                            f'  [{dataset_folder} | {split_type} ss={split_seed} | '
                            f'xs={xrand_seed} | {optimizer_name} | lr={lr}]  '
                            f'GCN_X={np.mean(gcnx_finals):.1f}±{np.std(gcnx_finals):.1f}%  '
                            f'GCN_U={np.mean(gcnu_finals):.1f}±{np.std(gcnu_finals):.1f}%  '
                            f'GCN_rnU={np.mean(gcnrn_finals):.1f}±{np.std(gcnrn_finals):.1f}%'
                        )

    return all_records


# ── Summary ────────────────────────────────────────────────────────────────────

def build_summary(all_records):
    from collections import defaultdict
    buckets = defaultdict(list)
    for rec in all_records:
        key = (rec['dataset'], rec['split_type'],
               rec.get('optimizer', ''), lr_str(rec.get('lr', 0)))
        buckets[key].append(rec)

    summary = {}
    for key, recs in buckets.items():
        dataset, split_type, optimizer, lr_s = key
        entry = {'random_baseline': recs[0]['random_baseline']}
        for mk in MODEL_KEYS:
            accs = [r[mk]['final_test_acc'] * 100 for r in recs if mk in r]
            if accs:
                entry[f'{mk}_mean'] = float(np.mean(accs))
                entry[f'{mk}_std']  = float(np.std(accs))
                entry[f'{mk}_n']    = len(accs)
        (summary
         .setdefault(dataset, {})
         .setdefault(split_type, {})
         .setdefault(optimizer, {})[lr_s]) = entry
    return summary


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(device):
    print('\n' + '=' * 65)
    print('SANITY CHECK — label_rand_v2_experiment')
    print('cora | fixed | adam | lr=0.01 | xseed=0 | train_seed=0')
    print('Verifying: val/test rows sum to 1 (probability simplex)')
    print('=' * 65)

    data_dir = os.path.join(DATA_ROOT, 'cora')
    for req in ['labels.npy', 'fixed_train_mask.npy',
                'fixed_val_mask.npy', 'fixed_test_mask.npy']:
        if not os.path.isfile(os.path.join(data_dir, req)):
            print(f'Skipped: data missing ({req}). Run save_data.py first.')
            return

    labels      = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)
    n           = len(labels)
    num_classes = int(labels.max()) + 1
    tr = np.load(os.path.join(data_dir, 'fixed_train_mask.npy'))
    va = np.load(os.path.join(data_dir, 'fixed_val_mask.npy'))
    te = np.load(os.path.join(data_dir, 'fixed_test_mask.npy'))
    verify_split(tr, va, te, 'fixed', 0, n)

    adj_kipf, L, D = load_cached_graph('cora')
    adj_t    = scipy_to_torch_sparse(adj_kipf, device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    tr_t = torch.tensor(tr, dtype=torch.bool, device=device)
    va_t = torch.tensor(va, dtype=torch.bool, device=device)
    te_t = torch.tensor(te, dtype=torch.bool, device=device)

    X_lr   = build_x_label_rand_v2(n, num_classes, labels, tr, xrand_seed=0)
    X_lr_t = torch.tensor(X_lr, dtype=torch.float32, device=device)

    print(f'\n  X_label_rand_v2: shape={X_lr.shape}')
    non_train = ~tr
    print(f'  Val/test rows  — sum: min={X_lr[non_train].sum(1).min():.5f}  '
          f'max={X_lr[non_train].sum(1).max():.5f}  [should be ~1.0]')
    print(f'  Val/test rows  — min entry: {X_lr[non_train].min():.5f}  '
          f'[should be ≥0]')
    print(f'  Train rows (first 3): {X_lr[tr][:3]}  [should be one-hot]')

    U_lr, d_eff, ortho_err = compute_u_label_rand_v2(X_lr, L, D, 0, 'fixed_ss0')
    U_lr_t = torch.tensor(U_lr, dtype=torch.float32, device=device)
    sc = StandardScaler(); sc.fit(U_lr[tr])
    U_lr_sc   = sc.transform(U_lr).astype(np.float32)
    U_lr_sc_t = torch.tensor(U_lr_sc, dtype=torch.float32, device=device)

    _, _, _, fi_gcnx = train_gcn(
        X_lr_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='GCN_X'
    )
    _, _, _, fi_gcnu = train_gcn(
        U_lr_sc_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True, label='GCN_U'
    )
    _, _, _, fi_gcnrn = train_gcn_rownorm(
        U_lr_t, adj_t, labels_t, tr_t, va_t, te_t,
        num_classes, 'adam', 0.01, 0, device, verbose=True
    )

    baseline = 100.0 / num_classes
    print(f'\n  RESULTS (cora, {EPOCHS} epochs, adam lr=0.01, xseed=0):')
    print(f'  {"Model":<22} {"Test%":>7}  {"vs random":>12}')
    print(f'  {"-"*44}')
    for name, fi in [('GCN_X (simplex rand)', fi_gcnx),
                     ('GCN_U (U_label_rand)',  fi_gcnu),
                     ('GCN_rowNorm_U',         fi_gcnrn)]:
        pct = fi * 100
        print(f'  {name:<22} {pct:>7.2f}%  {pct - baseline:>+12.2f} pp')
    print(f'  {"Random baseline":<22} {baseline:>7.2f}%')
    print(f'\n  d_eff={d_eff}  ortho_err={ortho_err:.2e}  input_dim={num_classes}')
    print('=' * 65 + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 1-v2: label_rand with simplex random features'
    )
    parser.add_argument('--dataset',   type=str,   default=None)
    parser.add_argument('--optimizer', type=str,   choices=['sgd', 'adam'], default=None)
    parser.add_argument('--lr',        type=float, default=None)
    parser.add_argument('--sanity',    action='store_true')
    parser.add_argument('--verbose',   action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'label_rand_v2_experiment.py  |  Device: {device}')
    print(f'v2 change: val/test features = probability simplex (not Gaussian)')

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

    n_splits    = 1 + len(RANDOM_SPLIT_SEEDS)
    runs_per_ds = (len(XRAND_SEEDS) * n_splits
                   * len(optimizers_to_run) * len(lrs_to_run) * len(TRAIN_SEEDS))

    print(f'\nDatasets   : {datasets_to_run}')
    print(f'Optimizers : {optimizers_to_run}  |  LRs: {lrs_to_run}')
    print(f'Xrand seeds: {XRAND_SEEDS}  |  Train seeds: {TRAIN_SEEDS}')
    print(f'Splits     : 1 fixed + {len(RANDOM_SPLIT_SEEDS)} random = {n_splits}')
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

        print('\n' + '=' * 75)
        print('OVERVIEW — label_rand_v2_experiment  (simplex random features)')
        print(f'{"Dataset":<22} {"base%":>5} {"GCN_X":>8} {"GCN_U":>8} {"GCN_rnU":>9}')
        print('-' * 75)
        for ds in datasets_to_run:
            if ds not in summary:
                continue
            agg  = {k: [] for k in MODEL_KEYS}
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
            print(f'{ds:<22} {base:>5.1f} {m("GCN_X"):>8} {m("GCN_U"):>8} '
                  f'{m("GCN_rowNorm_U"):>9}')
        print('=' * 75)

    print('\nDone.')
