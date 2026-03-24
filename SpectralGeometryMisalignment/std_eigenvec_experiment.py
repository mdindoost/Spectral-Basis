"""
SpectralGeometryMisalignment/std_eigenvec_experiment.py

RESEARCH QUESTION:
  Does D-orthonormality specifically cause the convergence gap observed between
  X and Y, or does any Rayleigh-Ritz reparameterization hurt equally?

  Y     = D-orthonormal Rayleigh-Ritz eigenvectors  (Y^T D Y = I)
  U_std = Euclidean-orthonormal Rayleigh-Ritz eigenvectors  (U_std^T U_std = I)

  Both Y and U_std span the same subspace as X (by construction).
  The only difference is the inner product used for orthonormalization.

  If Ustd_minus_Y >> 0: standard orthonormality is less damaging →
    D-orthonormality is specifically the culprit.
  If Ustd_minus_Y ≈ 0: both are equally damaging →
    the damage comes from spectral reparameterization in general, not D.
  If Ustd_minus_Y < 0: D-orthonormal is actually better (unexpected).

FOUR CONDITIONS:
  X     : raw features           (from X.npy)
  Y     : D-orthonormal RR       (from Y.npy)
  U_std : Euclidean-orthonormal RR (from U_std.npy)
  U_rn  : row-normalized U_std   (computed on the fly)

TWO MODEL TYPES:
  linear : nn.Linear(d_in, num_classes), weight_decay=0.0
  mlp    : 3-layer MLP with hidden_dim=256, weight_decay=5e-4

TRAINING PROTOCOL:
  Optimizer: Adam (betas=(0.9,0.999), eps=1e-8)
  Learning rate: 0.01
  Epochs: 500, NO early stopping
  Full-batch (all training nodes in one forward pass)
  Training seeds: 15 (0–14)
  Splits: fixed (1 seed) + random (5 seeds, stratified 60/20/20)
  StandardScaler fit on training nodes only, per condition independently.
  For U_rn: StandardScaler applied AFTER row-normalization.

KEY GAPS IN OUTPUT:
  Ustd_minus_Y : U_std test acc − Y test acc  (primary research question)
  Urn_minus_Y  : U_rn test acc − Y test acc   (does RowNorm on standard help?)

OUTPUT:
  SpectralGeometryMisalignment/results/std_eigenvec/{model_type}/{dataset}/
    {dataset}_{split_type}_splitseed{s}_seed{train_seed}.json
  SpectralGeometryMisalignment/results/std_eigenvec/{model_type}/summary.json

Usage:
  # Run everything:
  /home/md724/Spectral-Basis/venv/bin/python std_eigenvec_experiment.py

  # Subset for testing / sanity check:
  /home/md724/Spectral-Basis/venv/bin/python std_eigenvec_experiment.py \\
      --dataset cora --model linear --sanity-only

  # Single dataset + model type:
  /home/md724/Spectral-Basis/venv/bin/python std_eigenvec_experiment.py \\
      --dataset cora --model mlp
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(_HERE, 'data')
RESULTS_ROOT = os.path.join(_HERE, 'results', 'std_eigenvec')

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

MODEL_TYPES        = ['linear', 'mlp']
LR                 = 0.01
HIDDEN_DIM         = 256   # matches master_training.py
EPOCHS             = 500
TRAIN_SEEDS        = list(range(15))
RANDOM_SPLIT_SEEDS = [0, 1, 2, 3, 4]

# Weight decay per model type (matches paper conventions)
WD = {'linear': 0.0, 'mlp': 5e-4}


# ── Models ────────────────────────────────────────────────────────────────────

class LinearModel(nn.Module):
    """Single linear layer — softmax regression."""
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """3-layer MLP matching master_training.py architecture."""
    def __init__(self, d_in: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_model(model_type: str, d_in: int, num_classes: int) -> nn.Module:
    if model_type == 'linear':
        return LinearModel(d_in, num_classes)
    elif model_type == 'mlp':
        return MLP(d_in, HIDDEN_DIM, num_classes)
    else:
        raise ValueError(f'Unknown model_type: {model_type}')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_arrays(dataset: str, split_type: str, split_seed: int):
    """
    Load X, Y, U_std, labels, and split masks from disk.
    Returns:
      X, Y, U_std : np.ndarray float32
      labels      : np.ndarray int64
      tr_mask, va_mask, te_mask : bool np.ndarray
    Exits with error message if any required file is missing.
    """
    d = os.path.join(DATA_ROOT, dataset)

    required = ['X.npy', 'Y.npy', 'U_std.npy', 'labels.npy']
    if split_type == 'fixed':
        required += ['fixed_train_mask.npy', 'fixed_val_mask.npy',
                     'fixed_test_mask.npy']
    else:
        for sfx in ['train', 'val', 'test']:
            required.append(f'random{split_seed}_{sfx}_mask.npy')

    missing = [f for f in required if not os.path.isfile(os.path.join(d, f))]
    if missing or not os.path.isdir(d):
        print(f'ERROR: Missing files for dataset "{dataset}":')
        for f in missing:
            print(f'  {os.path.join(d, f)}')
        if 'U_std.npy' in missing:
            print('  U_std.npy is missing — run save_data.py first to generate it.')
        else:
            print('  Run save_data.py first.')
        sys.exit(1)

    X     = np.load(os.path.join(d, 'X.npy')).astype(np.float32)
    Y     = np.load(os.path.join(d, 'Y.npy')).astype(np.float32)
    U_std = np.load(os.path.join(d, 'U_std.npy')).astype(np.float32)
    labels = np.load(os.path.join(d, 'labels.npy')).astype(np.int64)

    if split_type == 'fixed':
        tr_mask = np.load(os.path.join(d, 'fixed_train_mask.npy'))
        va_mask = np.load(os.path.join(d, 'fixed_val_mask.npy'))
        te_mask = np.load(os.path.join(d, 'fixed_test_mask.npy'))
    else:
        tr_mask = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
        va_mask = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
        te_mask = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))

    return X, Y, U_std, labels, tr_mask, va_mask, te_mask


# ── Preprocessing ─────────────────────────────────────────────────────────────

def row_normalize(U: np.ndarray) -> np.ndarray:
    """
    Row-normalize each row of U to unit L2 norm.
    Rows with norm < 1e-10 are left unchanged (zero-norm guard).
    Returns a new array (does not modify U in place).
    """
    norms = np.linalg.norm(U, axis=1, keepdims=True)   # (n, 1)
    U_rn  = U.copy()
    safe  = norms.ravel() >= 1e-10
    U_rn[safe] = U[safe] / norms[safe]
    return U_rn


def preprocess_all(X, Y, U_std, tr_mask, va_mask, te_mask):
    """
    Apply StandardScaler independently per condition.
    Scaler is fit on training nodes only, applied to val/test.
    U_rn is computed from U_std BEFORE scaling (row-norm is a geometric op).
    StandardScaler is then applied to U_rn independently.

    Returns:
      (X_tr, X_va, X_te),
      (Y_tr, Y_va, Y_te),
      (Us_tr, Us_va, Us_te),   # U_std scaled
      (Ur_tr, Ur_va, Ur_te)    # U_rn scaled
      All as np.float32.
    """
    def scale(arr):
        sc = StandardScaler()
        tr = sc.fit_transform(arr[tr_mask]).astype(np.float32)
        va = sc.transform(arr[va_mask]).astype(np.float32)
        te = sc.transform(arr[te_mask]).astype(np.float32)
        return tr, va, te

    U_rn = row_normalize(U_std)

    return (
        scale(X),
        scale(Y),
        scale(U_std),
        scale(U_rn),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(model_type: str, feat_tr, feat_va, feat_te,
              y_tr, y_va, y_te, num_classes: int, seed: int,
              device: torch.device, patience: int = None):
    """
    Train one model for up to EPOCHS epochs (full-batch).

    Linear model (patience=None):
      500 fixed epochs, no early stopping.
      final_test_acc = test accuracy at epoch 500.
      best_val_epoch = 500.

    MLP (patience=100):
      Early stopping on validation accuracy.
      final_test_acc = test accuracy at the epoch with best val accuracy.
      best_val_epoch = that epoch number.

    Returns:
      val_acc_curve  : list[float], length = epochs actually run
      final_test_acc : float
      best_val_epoch : int
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in  = feat_tr.shape[1]
    model = make_model(model_type, d_in, num_classes).to(device)
    opt   = torch.optim.Adam(
        model.parameters(), lr=LR,
        betas=(0.9, 0.999), eps=1e-8,
        weight_decay=WD[model_type],
    )
    crit = nn.CrossEntropyLoss()

    Xtr = torch.tensor(feat_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr,    dtype=torch.long,    device=device)
    Xva = torch.tensor(feat_va, dtype=torch.float32, device=device)
    yva = torch.tensor(y_va,    dtype=torch.long,    device=device)
    Xte = torch.tensor(feat_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te,    dtype=torch.long,    device=device)

    val_acc_curve  = []
    best_val_acc   = -1.0
    best_val_epoch = EPOCHS   # will be overwritten; default = last epoch
    best_test_acc  = 0.0
    no_improve     = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
            val_acc_curve.append(float(va_acc))

            if patience is not None:
                # MLP path: evaluate test at every epoch for best-val checkpoint
                te_acc = (model(Xte).argmax(1) == yte).float().mean().item()
                if va_acc > best_val_acc:
                    best_val_acc   = va_acc
                    best_val_epoch = epoch
                    best_test_acc  = te_acc
                    no_improve     = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break
            else:
                # Linear path: no early stopping, evaluate test only at epoch 500
                if epoch == EPOCHS:
                    best_test_acc  = (model(Xte).argmax(1) == yte).float().mean().item()
                    best_val_epoch = EPOCHS

    return val_acc_curve, float(best_test_acc), int(best_val_epoch)


# ── Output helpers ─────────────────────────────────────────────────────────────

def result_path(model_type: str, dataset: str, split_type: str,
                split_seed: int, train_seed: int) -> str:
    d = os.path.join(RESULTS_ROOT, model_type, dataset)
    os.makedirs(d, exist_ok=True)
    fname = (f'{dataset}_{split_type}_splitseed{split_seed}'
             f'_seed{train_seed}.json')
    return os.path.join(d, fname)


def save_record(record: dict, model_type: str, dataset: str,
                split_type: str, split_seed: int, train_seed: int):
    path = result_path(model_type, dataset, split_type, split_seed, train_seed)
    with open(path, 'w') as f:
        json.dump(record, f)


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_dataset(dataset: str, model_type: str, device: torch.device,
                seeds_override=None) -> list:
    """
    Run all splits × all train seeds for one (dataset, model_type).
    seeds_override: if set, use only these training seeds (for sanity check).
    Returns list of result records.
    """
    # Load arrays once (split masks differ, feature arrays do not)
    d     = os.path.join(DATA_ROOT, dataset)
    X_full    = np.load(os.path.join(d, 'X.npy')).astype(np.float32)
    Y_full    = np.load(os.path.join(d, 'Y.npy')).astype(np.float32)
    Ustd_full = np.load(os.path.join(d, 'U_std.npy')).astype(np.float32)
    labels    = np.load(os.path.join(d, 'labels.npy')).astype(np.int64)
    num_classes = int(labels.max()) + 1

    train_seeds = seeds_override if seeds_override is not None else TRAIN_SEEDS
    all_records = []

    # Early stopping for MLP (patience=100), none for linear — matches paper conventions.
    # Linear: test acc at epoch 500 (matches softmax_experiment.py).
    # MLP:    test acc at best-val-acc checkpoint (matches master_training.py).
    patience = 100 if model_type == 'mlp' else None

    for split_type in ['fixed', 'random']:
        split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS

        for split_seed in split_seeds:
            # Load split masks
            if split_type == 'fixed':
                tr_mask = np.load(os.path.join(d, 'fixed_train_mask.npy'))
                va_mask = np.load(os.path.join(d, 'fixed_val_mask.npy'))
                te_mask = np.load(os.path.join(d, 'fixed_test_mask.npy'))
            else:
                tr_mask = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
                va_mask = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
                te_mask = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))

            y_tr = labels[tr_mask]
            y_va = labels[va_mask]
            y_te = labels[te_mask]

            (X_tr, X_va, X_te), \
            (Y_tr, Y_va, Y_te), \
            (Us_tr, Us_va, Us_te), \
            (Ur_tr, Ur_va, Ur_te) = preprocess_all(
                X_full, Y_full, Ustd_full, tr_mask, va_mask, te_mask
            )

            for i, train_seed in enumerate(train_seeds):

                va_X,  te_X,  ep_X  = train_one(model_type,
                                                 X_tr,  X_va,  X_te,
                                                 y_tr, y_va, y_te,
                                                 num_classes, train_seed, device,
                                                 patience=patience)

                va_Y,  te_Y,  ep_Y  = train_one(model_type,
                                                 Y_tr,  Y_va,  Y_te,
                                                 y_tr, y_va, y_te,
                                                 num_classes, train_seed, device,
                                                 patience=patience)

                va_Us, te_Us, ep_Us = train_one(model_type,
                                                 Us_tr, Us_va, Us_te,
                                                 y_tr, y_va, y_te,
                                                 num_classes, train_seed, device,
                                                 patience=patience)

                va_Ur, te_Ur, ep_Ur = train_one(model_type,
                                                 Ur_tr, Ur_va, Ur_te,
                                                 y_tr, y_va, y_te,
                                                 num_classes, train_seed, device,
                                                 patience=patience)

                # Progress print every 5 seeds
                if (i + 1) % 5 == 0 or i == 0:
                    print(
                        f'  [{dataset} | {split_type} split{split_seed} | '
                        f'{model_type} | seed={train_seed}]  '
                        f'X={te_X*100:.1f}%  '
                        f'Y={te_Y*100:.1f}%  '
                        f'U_std={te_Us*100:.1f}%  '
                        f'U_rn={te_Ur*100:.1f}%'
                    )

                record = {
                    'dataset':    dataset,
                    'model_type': model_type,
                    'split_type': split_type,
                    'split_seed': split_seed,
                    'train_seed': train_seed,
                    'optimizer':  'adam',
                    'lr':         LR,
                    'X': {
                        'final_test_acc': float(te_X),
                        'val_acc_curve':  va_X,
                        'best_val_epoch': ep_X,
                    },
                    'Y': {
                        'final_test_acc': float(te_Y),
                        'val_acc_curve':  va_Y,
                        'best_val_epoch': ep_Y,
                    },
                    'U_std': {
                        'final_test_acc': float(te_Us),
                        'val_acc_curve':  va_Us,
                        'best_val_epoch': ep_Us,
                    },
                    'U_rn': {
                        'final_test_acc': float(te_Ur),
                        'val_acc_curve':  va_Ur,
                        'best_val_epoch': ep_Ur,
                    },
                    'gaps': {
                        'X_minus_Y':    float((te_X  - te_Y)  * 100),
                        'X_minus_Ustd': float((te_X  - te_Us) * 100),
                        'X_minus_Urn':  float((te_X  - te_Ur) * 100),
                        'Ustd_minus_Y': float((te_Us - te_Y)  * 100),
                        'Urn_minus_Y':  float((te_Ur - te_Y)  * 100),
                    },
                }
                save_record(record, model_type, dataset,
                            split_type, split_seed, train_seed)
                all_records.append(record)

            # Per-split summary across all seeds in this split
            recs_here = [r for r in all_records
                         if r['split_type'] == split_type
                         and r['split_seed'] == split_seed]
            mean_ustd_minus_y = float(np.mean(
                [r['gaps']['Ustd_minus_Y'] for r in recs_here]
            ))
            mean_x_minus_y = float(np.mean(
                [r['gaps']['X_minus_Y'] for r in recs_here]
            ))
            print(
                f'\n  [{dataset} | {split_type} split{split_seed} | {model_type}]  '
                f'X-Y={mean_x_minus_y:+.2f}pp  '
                f'Ustd-Y={mean_ustd_minus_y:+.2f}pp  '
                f'({"D-ortho culprit" if mean_ustd_minus_y > 3 else "both equally damaging" if abs(mean_ustd_minus_y) <= 3 else "D-ortho actually better"})\n'
            )

    return all_records


# ── Aggregate summary ─────────────────────────────────────────────────────────

def build_summary(all_records: list) -> dict:
    """
    Build summary keyed by (dataset, split_type).
    Aggregates over all (split_seed, train_seed) combinations.
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for rec in all_records:
        key = (rec['dataset'], rec['split_type'])
        buckets[key].append(rec)

    summary = {}
    for (dataset, split_type), recs in buckets.items():
        x_accs  = [r['X']['final_test_acc']    * 100 for r in recs]
        y_accs  = [r['Y']['final_test_acc']    * 100 for r in recs]
        us_accs = [r['U_std']['final_test_acc'] * 100 for r in recs]
        ur_accs = [r['U_rn']['final_test_acc']  * 100 for r in recs]

        entry = {
            'X_mean':    float(np.mean(x_accs)),
            'X_std':     float(np.std(x_accs)),
            'Y_mean':    float(np.mean(y_accs)),
            'Y_std':     float(np.std(y_accs)),
            'U_std_mean': float(np.mean(us_accs)),
            'U_std_std':  float(np.std(us_accs)),
            'U_rn_mean':  float(np.mean(ur_accs)),
            'U_rn_std':   float(np.std(ur_accs)),
            'gap_X_minus_Y_mean':    float(np.mean([r['gaps']['X_minus_Y']    for r in recs])),
            'gap_X_minus_Ustd_mean': float(np.mean([r['gaps']['X_minus_Ustd'] for r in recs])),
            'gap_X_minus_Urn_mean':  float(np.mean([r['gaps']['X_minus_Urn']  for r in recs])),
            'gap_Ustd_minus_Y_mean': float(np.mean([r['gaps']['Ustd_minus_Y'] for r in recs])),
            'gap_Urn_minus_Y_mean':  float(np.mean([r['gaps']['Urn_minus_Y']  for r in recs])),
            'n_seeds': len(recs),
        }
        summary.setdefault(dataset, {})[split_type] = entry

    return summary


def save_summary(summary: dict, model_type: str):
    d = os.path.join(RESULTS_ROOT, model_type)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {path}')


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(device: torch.device, model_types=None):
    """
    cora / fixed split / adam / lr=0.01 / seeds 0,1,2 only
    Both linear and MLP (or subset via model_types).

    Prints per-seed results and means.
    Interpretation guide for Ustd_minus_Y printed after results.
    """
    if model_types is None:
        model_types = MODEL_TYPES

    dataset     = 'cora'
    sanity_seeds = [0, 1, 2]

    d = os.path.join(DATA_ROOT, dataset)
    required = ['X.npy', 'Y.npy', 'U_std.npy', 'labels.npy',
                'fixed_train_mask.npy', 'fixed_val_mask.npy', 'fixed_test_mask.npy']
    missing = [f for f in required if not os.path.isfile(os.path.join(d, f))]
    if missing:
        print('Sanity check skipped: required files missing. Run save_data.py first.')
        for f in missing:
            print(f'  {os.path.join(d, f)}')
        return

    print('\n' + '=' * 70)
    print('SANITY CHECK: cora | fixed split | adam | lr=0.01 | seeds 0,1,2')
    print('=' * 70)

    for model_type in model_types:
        print(f'\n  Model: {model_type}')
        print(f'  {"Seed":>4}  {"X":>7}  {"Y":>7}  {"U_std":>7}  {"U_rn":>7}  '
              f'{"Ustd-Y":>8}  {"Urn-Y":>8}')
        print('  ' + '-' * 62)

        records = run_dataset(dataset, model_type, device,
                              seeds_override=sanity_seeds)
        # Filter to fixed split only
        fixed_recs = [r for r in records if r['split_type'] == 'fixed']

        ustd_minus_y_vals = []
        urn_minus_y_vals  = []

        for rec in fixed_recs:
            seed       = rec['train_seed']
            te_X       = rec['X']['final_test_acc']    * 100
            te_Y       = rec['Y']['final_test_acc']    * 100
            te_Us      = rec['U_std']['final_test_acc'] * 100
            te_Ur      = rec['U_rn']['final_test_acc']  * 100
            gap_us_y   = rec['gaps']['Ustd_minus_Y']
            gap_ur_y   = rec['gaps']['Urn_minus_Y']
            ustd_minus_y_vals.append(gap_us_y)
            urn_minus_y_vals.append(gap_ur_y)
            print(f'  {seed:>4}  {te_X:>6.2f}%  {te_Y:>6.2f}%  '
                  f'{te_Us:>6.2f}%  {te_Ur:>6.2f}%  '
                  f'{gap_us_y:>+7.2f}pp  {gap_ur_y:>+7.2f}pp')

        mean_us_y = float(np.mean(ustd_minus_y_vals))
        mean_ur_y = float(np.mean(urn_minus_y_vals))
        print(f'  {"MEAN":>4}  {"":>7}  {"":>7}  {"":>7}  {"":>7}  '
              f'{mean_us_y:>+7.2f}pp  {mean_ur_y:>+7.2f}pp')

    # Interpretation guide
    print('\n  INTERPRETATION GUIDE (Ustd_minus_Y):')
    print('   > +3pp : standard orthonormality is less damaging than D-orthonormality')
    print('            → D-orthonormality specifically causes the damage')
    print('    ≈ 0pp : both orthonormalizations are equally damaging')
    print('            → spectral reparameterization in general hurts, not D specifically')
    print('   < -3pp : D-orthonormal is actually better (unexpected)')
    print()
    print('  INTERPRETATION GUIDE (Urn_minus_Y):')
    print('   > +3pp : RowNorm on standard eigenvecs recovers more than RowNorm on Y')
    print('    ≈ 0pp : similar recovery for both eigenvector types')
    print('=' * 70 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Standard vs D-orthonormal eigenvector experiment'
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset to run (default: all 9). '
                             'Use underscore, e.g. ogbn_arxiv')
    parser.add_argument('--model',   type=str, choices=MODEL_TYPES, default=None,
                        help='Model type: linear or mlp (default: both)')
    parser.add_argument('--sanity-only', action='store_true',
                        help='Run sanity check only (cora | fixed | seeds 0,1,2)')
    args = parser.parse_args()

    device   = torch.device('cpu')
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    models   = [args.model]   if args.model   else MODEL_TYPES

    # Verify data files exist for all requested datasets
    for ds in datasets:
        d = os.path.join(DATA_ROOT, ds)
        for fname in ['X.npy', 'Y.npy', 'U_std.npy', 'labels.npy']:
            fpath = os.path.join(d, fname)
            if not os.path.isfile(fpath):
                print(f'ERROR: {fpath} not found.')
                if fname == 'U_std.npy':
                    print('  U_std.npy is missing — run save_data.py first.')
                else:
                    print('  Run save_data.py first.')
                sys.exit(1)

    if args.sanity_only:
        sanity_check(device, model_types=models)
        sys.exit(0)

    print('SpectralGeometryMisalignment — std_eigenvec_experiment.py')
    print(f'Datasets:    {datasets}')
    print(f'Model types: {models}')
    print(f'Optimizer:   Adam  lr={LR}')
    print(f'Epochs:      {EPOCHS}  (no early stopping)')
    print(f'Train seeds: {len(TRAIN_SEEDS)}  (0–14)')
    print(f'Results:     {RESULTS_ROOT}')
    print()

    for model_type in models:
        all_records = []

        for dataset in datasets:
            print(f'\n{"─"*70}')
            print(f'Config: {dataset} | {model_type}')
            print(f'{"─"*70}')

            records = run_dataset(dataset, model_type, device)
            all_records.extend(records)

            # Per-dataset overall summary
            us_y_gaps = [r['gaps']['Ustd_minus_Y'] for r in records]
            x_y_gaps  = [r['gaps']['X_minus_Y']    for r in records]
            print(
                f'  OVERALL [{dataset} | {model_type}]  '
                f'X-Y={np.mean(x_y_gaps):+.2f}±{np.std(x_y_gaps):.2f}pp  '
                f'Ustd-Y={np.mean(us_y_gaps):+.2f}±{np.std(us_y_gaps):.2f}pp'
            )

        summary = build_summary(all_records)
        save_summary(summary, model_type)

    # Sanity check at the end (cora | fixed | adam | seeds 0,1,2)
    sanity_check(device, model_types=models)

    print('Done.')
