"""
SpectralGeometryMisalignment/softmax_experiment.py

THEORETICAL GUARANTEE:
  Softmax regression with cross-entropy is convex. Since span(X) = span(Y)
  by Rayleigh-Ritz construction, the global optimum accuracy must be identical
  for both representations. This experiment measures CONVERGENCE SPEED.
  Final accuracy difference is a sanity check — it should be approximately zero.

MODEL:
  nn.Linear(d_in, num_classes)  only.
  No hidden layers. No dropout. No batch norm.
  Loss: nn.CrossEntropyLoss()
  Weight decay: STRICTLY 0.0 in optimizer (convexity requires this).

OPTIMIZERS:
  SGD  — momentum=0.9, weight_decay=0.0
  Adam — betas=(0.9,0.999), eps=1e-8, weight_decay=0.0

LEARNING RATES: [0.001, 0.01, 0.1] — same lr for X and Y.

TRAINING PROTOCOL:
  500 epochs fixed, NO early stopping.
  Full-batch (all training nodes in one forward pass).
  Record every epoch: train loss, val accuracy, test accuracy.
  15 training seeds per config.
  Split types: fixed (1 split seed) and random (5 split seeds × 15 train seeds).
  StandardScaler fit on training nodes only, applied to val/test.
  Fit separately for X and Y.

OUTPUT:
  SpectralGeometryMisalignment/results/softmax/{dataset}/
    {dataset}_{split_type}_splitseed{split_seed}_{optimizer}_lr{lr}_seed{train_seed}.json
  SpectralGeometryMisalignment/results/softmax/summary.json

Usage:
  # Run everything:
  /home/md724/Spectral-Basis/venv/bin/python softmax_experiment.py

  # Subset for testing:
  /home/md724/Spectral-Basis/venv/bin/python softmax_experiment.py \\
      --dataset cora --optimizer adam --lr 0.01
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

_HERE         = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT     = os.path.join(_HERE, 'data')
RESULTS_ROOT  = os.path.join(_HERE, 'results', 'softmax')

ALL_DATASETS  = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

OPTIMIZERS    = ['sgd', 'adam']
LEARNING_RATES = [0.001, 0.01, 0.1]
TRAIN_SEEDS   = list(range(15))
RANDOM_SPLIT_SEEDS = [0, 1, 2, 3, 4]
EPOCHS        = 500


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset_files(dataset: str, split_type: str, split_seed: int):
    """
    Load X, Y, labels and the requested split masks from disk.
    Returns:
      X, Y: np.ndarray float32
      labels: np.ndarray int64
      train_mask, val_mask, test_mask: boolean np.ndarray
    Raises SystemExit if any file is missing.
    """
    d = os.path.join(DATA_ROOT, dataset)

    required_files = ['X.npy', 'Y.npy', 'labels.npy']
    if split_type == 'fixed':
        required_files += ['fixed_train_mask.npy', 'fixed_val_mask.npy', 'fixed_test_mask.npy']
    else:
        for suffix in ['train', 'val', 'test']:
            required_files.append(f'random{split_seed}_{suffix}_mask.npy')

    missing = [f for f in required_files if not os.path.isfile(os.path.join(d, f))]
    if missing or not os.path.isdir(d):
        print(f'ERROR: Missing data files for dataset "{dataset}":')
        for f in missing:
            print(f'  {os.path.join(d, f)}')
        print('Please run save_data.py first.')
        sys.exit(1)

    X      = np.load(os.path.join(d, 'X.npy')).astype(np.float32)
    Y      = np.load(os.path.join(d, 'Y.npy')).astype(np.float32)
    labels = np.load(os.path.join(d, 'labels.npy')).astype(np.int64)

    if split_type == 'fixed':
        tr_mask = np.load(os.path.join(d, 'fixed_train_mask.npy'))
        va_mask = np.load(os.path.join(d, 'fixed_val_mask.npy'))
        te_mask = np.load(os.path.join(d, 'fixed_test_mask.npy'))
    else:
        tr_mask = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
        va_mask = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
        te_mask = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))

    return X, Y, labels, tr_mask, va_mask, te_mask


# ── Model ─────────────────────────────────────────────────────────────────────

class SoftmaxRegression(nn.Module):
    """Single linear layer — no hidden layers, no dropout, no batch norm."""
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes)

    def forward(self, x):
        return self.linear(x)


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_optimizer(name: str, params, lr: float):
    """
    name: 'sgd' or 'adam'
    weight_decay is STRICTLY 0.0 for convexity.
    """
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    elif name == 'adam':
        return torch.optim.Adam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
    else:
        raise ValueError(f'Unknown optimizer: {name}')


# ── Training ──────────────────────────────────────────────────────────────────

def train_softmax(X_tr, y_tr, X_va, y_va, X_te, y_te,
                  num_classes: int, optimizer_name: str, lr: float,
                  seed: int, device: torch.device):
    """
    Full-batch softmax regression for EPOCHS epochs with NO early stopping.

    Returns:
      train_loss_curve: list[float], length 500
      val_acc_curve:    list[float], length 500
      test_acc_curve:   list[float], length 500
      final_test_acc:   float   (epoch 500)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in  = X_tr.shape[1]
    model = SoftmaxRegression(d_in, num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    crit  = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long,    device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)
    yva = torch.tensor(y_va, dtype=torch.long,    device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long,    device=device)

    train_loss_curve = []
    val_acc_curve    = []
    test_acc_curve   = []

    for epoch in range(1, EPOCHS + 1):
        # Full-batch training step
        model.train()
        opt.zero_grad()
        logits   = model(Xtr)
        loss     = crit(logits, ytr)
        loss.backward()
        opt.step()

        # Evaluation (no gradient)
        model.eval()
        with torch.no_grad():
            tr_loss  = loss.item()
            va_acc   = (model(Xva).argmax(1) == yva).float().mean().item()
            te_acc   = (model(Xte).argmax(1) == yte).float().mean().item()

        train_loss_curve.append(float(tr_loss))
        val_acc_curve.append(float(va_acc))
        test_acc_curve.append(float(te_acc))

    return train_loss_curve, val_acc_curve, test_acc_curve, float(test_acc_curve[-1])


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X, Y, tr_mask, va_mask, te_mask):
    """
    StandardScaler fit on training nodes only, applied to val/test.
    Fit separately for X and Y.
    Returns: X_tr, X_va, X_te, Y_tr, Y_va, Y_te  (all np.float32)
    """
    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X[tr_mask]).astype(np.float32)
    X_va = scaler_X.transform(X[va_mask]).astype(np.float32)
    X_te = scaler_X.transform(X[te_mask]).astype(np.float32)

    scaler_Y = StandardScaler()
    Y_tr = scaler_Y.fit_transform(Y[tr_mask]).astype(np.float32)
    Y_va = scaler_Y.transform(Y[va_mask]).astype(np.float32)
    Y_te = scaler_Y.transform(Y[te_mask]).astype(np.float32)

    return X_tr, X_va, X_te, Y_tr, Y_va, Y_te


# ── Output helpers ────────────────────────────────────────────────────────────

def lr_str(lr: float) -> str:
    """Consistent lr → filename-safe string."""
    return str(lr)


def result_filename(dataset: str, split_type: str, split_seed: int,
                    optimizer: str, lr: float, train_seed: int) -> str:
    return (f'{dataset}_{split_type}_splitseed{split_seed}'
            f'_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


def save_result(dataset: str, split_type: str, split_seed: int,
                optimizer: str, lr: float, train_seed: int, record: dict):
    d = os.path.join(RESULTS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    fname = result_filename(dataset, split_type, split_seed, optimizer, lr, train_seed)
    with open(os.path.join(d, fname), 'w') as f:
        json.dump(record, f)


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_config(dataset: str, optimizer_name: str, lr: float, device: torch.device):
    """
    Run one (dataset, optimizer, lr) configuration across all split types and seeds.
    Prints progress every 100 epochs (via the training loop) and a final summary.
    Returns: list of all individual result records.
    """
    # ── Load X, Y once per dataset (split masks differ, data does not) ──────
    # We load with fixed split seed=0 just to get X and Y arrays.
    X_full = np.load(os.path.join(DATA_ROOT, dataset, 'X.npy')).astype(np.float32)
    Y_full = np.load(os.path.join(DATA_ROOT, dataset, 'Y.npy')).astype(np.float32)
    labels = np.load(os.path.join(DATA_ROOT, dataset, 'labels.npy')).astype(np.int64)
    num_classes = int(labels.max()) + 1

    all_records = []

    for split_type in ['fixed', 'random']:
        split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS

        for split_seed in split_seeds:
            # Load masks for this (split_type, split_seed)
            d = os.path.join(DATA_ROOT, dataset)
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

            X_tr, X_va, X_te, Y_tr, Y_va, Y_te = preprocess(
                X_full, Y_full, tr_mask, va_mask, te_mask
            )

            seed_X_gaps = []
            seed_Y_final = []
            seed_X_final = []

            for train_seed in TRAIN_SEEDS:

                # ── Train on X ──────────────────────────────────────────────
                tr_loss_X, va_acc_X, te_acc_X, final_te_X = train_softmax(
                    X_tr, y_tr, X_va, y_va, X_te, y_te,
                    num_classes, optimizer_name, lr, train_seed, device
                )

                # ── Train on Y ──────────────────────────────────────────────
                tr_loss_Y, va_acc_Y, te_acc_Y, final_te_Y = train_softmax(
                    Y_tr, y_tr, Y_va, y_va, Y_te, y_te,
                    num_classes, optimizer_name, lr, train_seed, device
                )

                gap_pp = (final_te_X - final_te_Y) * 100.0

                # Print progress snapshot (epoch 100, 200, ..., 500)
                for ep in [100, 200, 300, 400, 500]:
                    idx = ep - 1
                    print(
                        f'  [{dataset} | {split_type} split{split_seed} | '
                        f'{optimizer_name} | lr={lr} | seed={train_seed} | '
                        f'ep={ep}/{EPOCHS}]  '
                        f'X val={va_acc_X[idx]:.3f}  Y val={va_acc_Y[idx]:.3f}'
                    )

                record = {
                    'dataset':    dataset,
                    'split_type': split_type,
                    'split_seed': split_seed,
                    'optimizer':  optimizer_name,
                    'lr':         lr,
                    'train_seed': train_seed,
                    'X': {
                        'train_loss_curve': tr_loss_X,
                        'val_acc_curve':    va_acc_X,
                        'test_acc_curve':   te_acc_X,
                        'final_test_acc':   final_te_X,
                    },
                    'Y': {
                        'train_loss_curve': tr_loss_Y,
                        'val_acc_curve':    va_acc_Y,
                        'test_acc_curve':   te_acc_Y,
                        'final_test_acc':   final_te_Y,
                    },
                    'convergence_gap_pp': float(gap_pp),
                }
                save_result(dataset, split_type, split_seed,
                            optimizer_name, lr, train_seed, record)
                all_records.append(record)

                seed_X_gaps.append(gap_pp)
                seed_X_final.append(final_te_X * 100.0)
                seed_Y_final.append(final_te_Y * 100.0)

            # Per-split summary
            mean_gap = float(np.mean(seed_X_gaps))
            std_gap  = float(np.std(seed_X_gaps))
            print(
                f'\n  [{dataset} | {split_type} split_seed={split_seed} | '
                f'{optimizer_name} | lr={lr}]  '
                f'Final gap (X-Y): {mean_gap:+.3f} ± {std_gap:.3f} pp '
                f'(X={np.mean(seed_X_final):.2f}%  Y={np.mean(seed_Y_final):.2f}%)\n'
            )

    return all_records


# ── Aggregate summary ─────────────────────────────────────────────────────────

def build_summary(all_records: list) -> dict:
    """
    Build nested summary dict keyed by (dataset, split_type, optimizer, lr_str).
    Aggregates over all (split_seed, train_seed) for a given key.
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for rec in all_records:
        key = (rec['dataset'], rec['split_type'], rec['optimizer'], lr_str(rec['lr']))
        buckets[key].append(rec)

    summary = {}
    for (dataset, split_type, optimizer, lr_s), recs in buckets.items():
        x_accs   = [r['X']['final_test_acc'] * 100 for r in recs]
        y_accs   = [r['Y']['final_test_acc'] * 100 for r in recs]
        gaps_pp  = [r['convergence_gap_pp'] for r in recs]

        entry = {
            'X_final_acc_mean': float(np.mean(x_accs)),
            'X_final_acc_std':  float(np.std(x_accs)),
            'Y_final_acc_mean': float(np.mean(y_accs)),
            'Y_final_acc_std':  float(np.std(y_accs)),
            'gap_mean_pp':      float(np.mean(gaps_pp)),
            'gap_std_pp':       float(np.std(gaps_pp)),
            'n_seeds':          len(recs),
        }

        summary.setdefault(dataset, {}) \
               .setdefault(split_type, {}) \
               .setdefault(optimizer, {})[lr_s] = entry

    return summary


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(device: torch.device):
    """
    Dataset: cora, Split: fixed, Optimizer: Adam, LR: 0.01, Seeds: 0,1,2.
    Prints final test accuracy for X and Y at epoch 500 and the gap.
    """
    print('\n' + '=' * 60)
    print('SANITY CHECK: cora | fixed | adam | lr=0.01 | seeds 0,1,2')
    print('=' * 60)

    dataset = 'cora'
    d = os.path.join(DATA_ROOT, dataset)

    missing = [f for f in ['X.npy', 'Y.npy', 'labels.npy',
                            'fixed_train_mask.npy', 'fixed_val_mask.npy',
                            'fixed_test_mask.npy']
               if not os.path.isfile(os.path.join(d, f))]
    if missing:
        print('Sanity check skipped: data files missing. Run save_data.py first.')
        return

    X_full = np.load(os.path.join(d, 'X.npy')).astype(np.float32)
    Y_full = np.load(os.path.join(d, 'Y.npy')).astype(np.float32)
    labels = np.load(os.path.join(d, 'labels.npy')).astype(np.int64)
    tr_mask = np.load(os.path.join(d, 'fixed_train_mask.npy'))
    va_mask = np.load(os.path.join(d, 'fixed_val_mask.npy'))
    te_mask = np.load(os.path.join(d, 'fixed_test_mask.npy'))

    num_classes = int(labels.max()) + 1
    y_tr = labels[tr_mask]
    y_va = labels[va_mask]
    y_te = labels[te_mask]
    X_tr, X_va, X_te, Y_tr, Y_va, Y_te = preprocess(
        X_full, Y_full, tr_mask, va_mask, te_mask
    )

    gaps = []
    for seed in [0, 1, 2]:
        _, _, te_X, final_X = train_softmax(
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            num_classes, 'adam', 0.01, seed, device
        )
        _, _, te_Y, final_Y = train_softmax(
            Y_tr, y_tr, Y_va, y_va, Y_te, y_te,
            num_classes, 'adam', 0.01, seed, device
        )
        gap = (final_X - final_Y) * 100.0
        gaps.append(gap)
        print(f'  Seed {seed}: X test@500 = {final_X*100:.2f}%  '
              f'Y test@500 = {final_Y*100:.2f}%  gap = {gap:+.3f} pp')

    mean_gap = float(np.mean(gaps))
    print(f'\n  Mean gap (X - Y): {mean_gap:+.3f} pp')
    if abs(mean_gap) > 5.0:
        print(
            'WARNING: 500 epochs may be insufficient for convergence at this LR '
            '— consider extending epoch budget.'
        )
    print('=' * 60 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Softmax convergence experiment: X vs Y on all datasets'
    )
    parser.add_argument('--dataset',   type=str, default=None,
                        help='Single dataset to run (default: all)')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default=None,
                        help='Single optimizer to run (default: both)')
    parser.add_argument('--lr',        type=float, default=None,
                        help='Single learning rate to run (default: all three)')
    args = parser.parse_args()

    # Filter configurations based on optional flags
    datasets   = [args.dataset]   if args.dataset   else ALL_DATASETS
    optimizers = [args.optimizer] if args.optimizer  else OPTIMIZERS
    lrs        = [args.lr]        if args.lr         else LEARNING_RATES

    # Verify all requested datasets have data files
    for ds in datasets:
        d = os.path.join(DATA_ROOT, ds)
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, 'X.npy')):
            print(f'ERROR: Data not found for dataset "{ds}" at {d}')
            print('Please run save_data.py first.')
            sys.exit(1)

    device = torch.device('cpu')   # softmax regression — CPU is sufficient

    print('SpectralGeometryMisalignment — softmax_experiment.py')
    print(f'Datasets:   {datasets}')
    print(f'Optimizers: {optimizers}')
    print(f'LRs:        {lrs}')
    print(f'Epochs:     {EPOCHS}  (no early stopping)')
    print(f'Train seeds: {len(TRAIN_SEEDS)}  (0–14)')
    print(f'Results:    {RESULTS_ROOT}')
    print()

    all_records = []

    for dataset in datasets:
        for optimizer_name in optimizers:
            for lr in lrs:
                print(f'\n{"─"*70}')
                print(f'Config: {dataset} | {optimizer_name} | lr={lr}')
                print(f'{"─"*70}')
                records = run_config(dataset, optimizer_name, lr, device)
                all_records.extend(records)

                # Print final (dataset, optimizer, lr) summary across all splits/seeds
                X_finals = [r['X']['final_test_acc'] * 100 for r in records]
                Y_finals = [r['Y']['final_test_acc'] * 100 for r in records]
                gaps     = [r['convergence_gap_pp'] for r in records]
                print(
                    f'  OVERALL [{dataset} | {optimizer_name} | lr={lr}]  '
                    f'Final gap (X-Y): {np.mean(gaps):+.3f} ± {np.std(gaps):.3f} pp  '
                    f'(X={np.mean(X_finals):.2f}%  Y={np.mean(Y_finals):.2f}%)'
                )

    # Save aggregate summary
    summary = build_summary(all_records)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    summary_path = os.path.join(RESULTS_ROOT, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to: {summary_path}')

    # Sanity check (cora | fixed | adam | lr=0.01 | seeds 0,1,2)
    sanity_check(device)

    print('\nDone.')
