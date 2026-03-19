"""
SpectralGeometryMisalignment/d_wadia_experiment.py

D-Wadia control experiment.

PURPOSE
-------
Rayleigh-Ritz (Y) does two things simultaneously:
  1. D-orthogonalization: enforces U^T D U = I
  2. Basis change: rotates features into Laplacian eigenvector coordinates

This experiment adds two controls to isolate which effect causes the
performance gap observed in softmax regression:

  Z  Standard ZCA of centered X  →  Z^T Z = I_{d_eff_z}, PCA basis
     Tests: does Euclidean whitening break softmax regression?

  W  D-ZCA of X (Yiannis's formulation)  →  W^T D W = I_{d_eff_w}, D-PCA basis
     Tests: does D-orthogonalization alone (without Laplacian rotation) cause damage?

KEY COMPARISON: W_minus_Y gap (pp)
  W_minus_Y > 3pp  → Laplacian basis change causes extra damage beyond D-orthogonalization
  W_minus_Y 1–3pp  → inconclusive, both effects contribute
  W_minus_Y < 1pp  → D-orthogonalization alone explains the gap

TWO MODES
---------
  scaled   (PRIMARY): StandardScaler applied to all four conditions.
           Consistent with existing softmax_experiment.py for X and Y.
           All four comparisons (X−Y, X−Z, X−W, W−Y) are valid.

  unscaled (ROBUSTNESS CHECK): No preprocessing.
           Raw scale differences mean X comparisons are confounded.
           Only W_minus_Y is interpretable in this mode.

PROTOCOL
--------
  Model:   nn.Linear(d_in, num_classes), CrossEntropyLoss, weight_decay=0.0
  Optimizers: SGD (momentum=0.9), Adam (betas=(0.9,0.999), eps=1e-8)
  LRs:     [0.001, 0.01, 0.1]
  Epochs:  500, no early stopping, full-batch
  Seeds:   15 training seeds × (1 fixed + 5 random) splits = up to 90 runs per config
  Datasets: all 9

RESULTS
-------
  results/d_wadia/{mode}/{dataset}/
    {dataset}_{split_type}_splitseed{s}_{opt}_lr{lr}_seed{seed}.json
  results/d_wadia/{mode}/summary.json

USAGE
-----
  # Run save_data.py first (generates Z.npy and W.npy):
  /home/md724/Spectral-Basis/venv/bin/python save_data.py

  # Full experiment (both modes):
  /home/md724/Spectral-Basis/venv/bin/python d_wadia_experiment.py

  # Single config for testing:
  /home/md724/Spectral-Basis/venv/bin/python d_wadia_experiment.py \\
      --dataset cora --optimizer adam --lr 0.01 --mode scaled
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(_HERE, 'data')
RESULTS_ROOT = os.path.join(_HERE, 'results', 'd_wadia')

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

OPTIMIZERS         = ['sgd', 'adam']
LEARNING_RATES     = [0.001, 0.01, 0.1]
TRAIN_SEEDS        = list(range(15))
RANDOM_SPLIT_SEEDS = [0, 1, 2, 3, 4]
EPOCHS             = 500
CONDITIONS         = ['X', 'Y', 'Z', 'W']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset_arrays(dataset: str):
    """
    Load X, Y, Z, W, labels from disk.
    Exits with a clear message if Z.npy or W.npy are missing.
    """
    d = os.path.join(DATA_ROOT, dataset)

    required = ['X.npy', 'Y.npy', 'Z.npy', 'W.npy', 'labels.npy']
    missing  = [f for f in required if not os.path.isfile(os.path.join(d, f))]

    if not os.path.isdir(d) or missing:
        print(f'ERROR: Missing files for dataset "{dataset}":')
        for f in missing:
            print(f'  {os.path.join(d, f)}')
        zw = [f for f in missing if f in ('Z.npy', 'W.npy')]
        if zw:
            print('Run save_data.py first to generate Z and W.')
        sys.exit(1)

    X      = np.load(os.path.join(d, 'X.npy')).astype(np.float32)
    Y      = np.load(os.path.join(d, 'Y.npy')).astype(np.float32)
    Z      = np.load(os.path.join(d, 'Z.npy')).astype(np.float32)
    W      = np.load(os.path.join(d, 'W.npy')).astype(np.float32)
    labels = np.load(os.path.join(d, 'labels.npy')).astype(np.int64)
    return X, Y, Z, W, labels


def load_split_masks(dataset: str, split_type: str, split_seed: int):
    """Load boolean split masks for the given (split_type, split_seed)."""
    d = os.path.join(DATA_ROOT, dataset)
    if split_type == 'fixed':
        tr = np.load(os.path.join(d, 'fixed_train_mask.npy'))
        va = np.load(os.path.join(d, 'fixed_val_mask.npy'))
        te = np.load(os.path.join(d, 'fixed_test_mask.npy'))
    else:
        tr = np.load(os.path.join(d, f'random{split_seed}_train_mask.npy'))
        va = np.load(os.path.join(d, f'random{split_seed}_val_mask.npy'))
        te = np.load(os.path.join(d, f'random{split_seed}_test_mask.npy'))
    return tr, va, te


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_conditions(X, Y, Z, W, tr_mask, va_mask, te_mask, scaled: bool):
    """
    Returns dict: condition_name -> (feat_tr, feat_va, feat_te) as float32.

    scaled=True  (PRIMARY MODE):
      StandardScaler fit on training nodes only, applied to val/test.
      Each condition has an independent scaler.
      Applying StandardScaler to already-whitened Z and W causes minimal
      distortion (whitened features have near-unit column variance) but
      is required so that all four conditions receive identical preprocessing
      — in particular so that W_minus_Y is a fair comparison.

    scaled=False (ROBUSTNESS MODE):
      No preprocessing. Representations used as-is from save_data.py.
      NOTE: Only W_minus_Y is interpretable in this mode. X_minus_Z and
      X_minus_W are confounded by scale differences in raw X.
    """
    result = {}
    for name, feat in [('X', X), ('Y', Y), ('Z', Z), ('W', W)]:
        if scaled:
            sc = StandardScaler()
            tr = sc.fit_transform(feat[tr_mask]).astype(np.float32)
            va = sc.transform(feat[va_mask]).astype(np.float32)
            te = sc.transform(feat[te_mask]).astype(np.float32)
        else:
            tr = feat[tr_mask].astype(np.float32)
            va = feat[va_mask].astype(np.float32)
            te = feat[te_mask].astype(np.float32)
        result[name] = (tr, va, te)
    return result


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
    """weight_decay is STRICTLY 0.0 for convexity."""
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    elif name == 'adam':
        return torch.optim.Adam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
    raise ValueError(f'Unknown optimizer: {name}')


# ── Training ──────────────────────────────────────────────────────────────────

def train_softmax(X_tr, y_tr, X_va, y_va, X_te, y_te,
                  num_classes: int, optimizer_name: str, lr: float,
                  seed: int, device: torch.device):
    """
    Full-batch softmax regression for EPOCHS epochs, no early stopping.

    Returns:
      train_loss_curve  list[float] length 500
      val_acc_curve     list[float] length 500
      test_acc_curve    list[float] length 500
      final_test_acc    float  (epoch 500 test accuracy, 0–1 scale)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SoftmaxRegression(X_tr.shape[1], num_classes).to(device)
    opt   = make_optimizer(optimizer_name, model.parameters(), lr)
    crit  = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long,    device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)
    yva = torch.tensor(y_va, dtype=torch.long,    device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long,    device=device)

    train_loss_curve, val_acc_curve, test_acc_curve = [], [], []

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
            te_acc = (model(Xte).argmax(1) == yte).float().mean().item()

        train_loss_curve.append(float(loss.item()))
        val_acc_curve.append(float(va_acc))
        test_acc_curve.append(float(te_acc))

    return train_loss_curve, val_acc_curve, test_acc_curve, float(test_acc_curve[-1])


# ── Result I/O ────────────────────────────────────────────────────────────────

def lr_str(lr: float) -> str:
    return str(lr)


def result_filename(dataset, split_type, split_seed, optimizer, lr, train_seed):
    return (f'{dataset}_{split_type}_splitseed{split_seed}'
            f'_{optimizer}_lr{lr_str(lr)}_seed{train_seed}.json')


def save_result(mode, dataset, split_type, split_seed, optimizer, lr, train_seed, record):
    out_dir = os.path.join(RESULTS_ROOT, mode, dataset)
    os.makedirs(out_dir, exist_ok=True)
    fname = result_filename(dataset, split_type, split_seed, optimizer, lr, train_seed)
    with open(os.path.join(out_dir, fname), 'w') as f:
        json.dump(record, f)


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_config(dataset: str, optimizer_name: str, lr: float,
               mode: str, device: torch.device):
    """
    Run one (dataset, optimizer, lr, mode) configuration over all splits and
    training seeds.  Returns list of all individual result records.
    """
    scaled = (mode == 'scaled')

    X_full, Y_full, Z_full, W_full, labels = load_dataset_arrays(dataset)
    num_classes = int(labels.max()) + 1

    all_records = []

    for split_type in ['fixed', 'random']:
        split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS

        for split_seed in split_seeds:
            tr_mask, va_mask, te_mask = load_split_masks(dataset, split_type, split_seed)

            y_tr = labels[tr_mask]
            y_va = labels[va_mask]
            y_te = labels[te_mask]

            cond_data = preprocess_conditions(
                X_full, Y_full, Z_full, W_full,
                tr_mask, va_mask, te_mask, scaled
            )

            # Accumulate final accs per condition for split-level summary
            seed_finals = {c: [] for c in CONDITIONS}

            for train_seed in TRAIN_SEEDS:

                cond_outputs = {}
                for cond in CONDITIONS:
                    feat_tr, feat_va, feat_te = cond_data[cond]
                    tr_loss, va_acc, te_acc, final_te = train_softmax(
                        feat_tr, y_tr, feat_va, y_va, feat_te, y_te,
                        num_classes, optimizer_name, lr, train_seed, device
                    )
                    cond_outputs[cond] = {
                        'train_loss_curve': tr_loss,
                        'val_acc_curve':    va_acc,
                        'test_acc_curve':   te_acc,
                        'final_test_acc':   final_te,
                    }
                    seed_finals[cond].append(final_te * 100.0)

                fX = cond_outputs['X']['final_test_acc']
                fY = cond_outputs['Y']['final_test_acc']
                fZ = cond_outputs['Z']['final_test_acc']
                fW = cond_outputs['W']['final_test_acc']

                gaps = {
                    'X_minus_Y': float((fX - fY) * 100.0),
                    'X_minus_Z': float((fX - fZ) * 100.0),
                    'X_minus_W': float((fX - fW) * 100.0),
                    'W_minus_Y': float((fW - fY) * 100.0),
                }

                # Progress: print every 5 seeds
                if (train_seed + 1) % 5 == 0:
                    print(
                        f'  [{dataset} | {split_type} s{split_seed} | '
                        f'{optimizer_name} | lr={lr} | seed={train_seed:>2}]  '
                        f'X={fX*100:.1f}%  Y={fY*100:.1f}%  '
                        f'Z={fZ*100:.1f}%  W={fW*100:.1f}%  '
                        f'W−Y={gaps["W_minus_Y"]:+.1f}pp'
                    )

                record = {
                    'dataset':    dataset,
                    'mode':       mode,
                    'split_type': split_type,
                    'split_seed': split_seed,
                    'optimizer':  optimizer_name,
                    'lr':         lr,
                    'train_seed': train_seed,
                    'X':    cond_outputs['X'],
                    'Y':    cond_outputs['Y'],
                    'Z':    cond_outputs['Z'],
                    'W':    cond_outputs['W'],
                    'gaps': gaps,
                }
                save_result(mode, dataset, split_type, split_seed,
                            optimizer_name, lr, train_seed, record)
                all_records.append(record)

            # Per-(split_type, split_seed) summary
            means = {c: float(np.mean(seed_finals[c])) for c in CONDITIONS}
            stds  = {c: float(np.std(seed_finals[c]))  for c in CONDITIONS}
            print(
                f'\n  [{dataset} | {split_type} s{split_seed} | {optimizer_name} | lr={lr}]\n'
                f'  X={means["X"]:.2f}±{stds["X"]:.2f}  '
                f'Y={means["Y"]:.2f}±{stds["Y"]:.2f}  '
                f'Z={means["Z"]:.2f}±{stds["Z"]:.2f}  '
                f'W={means["W"]:.2f}±{stds["W"]:.2f}\n'
                f'  W−Y={means["W"]-means["Y"]:+.2f}pp  '
                f'X−W={means["X"]-means["W"]:+.2f}pp  '
                f'X−Z={means["X"]-means["Z"]:+.2f}pp  '
                f'X−Y={means["X"]-means["Y"]:+.2f}pp\n'
            )

    return all_records


# ── Summary builder ───────────────────────────────────────────────────────────

def build_summary(all_records: list) -> dict:
    """
    Aggregate over all (split_seed, train_seed) for each
    (dataset, split_type, optimizer, lr) key.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for rec in all_records:
        key = (rec['dataset'], rec['split_type'], rec['optimizer'], lr_str(rec['lr']))
        buckets[key].append(rec)

    summary = {}
    for (dataset, split_type, optimizer, lr_s), recs in buckets.items():
        entry = {}
        for cond in CONDITIONS:
            accs = [r[cond]['final_test_acc'] * 100.0 for r in recs]
            entry[f'{cond}_final_acc_mean'] = float(np.mean(accs))
            entry[f'{cond}_final_acc_std']  = float(np.std(accs))
        for gap_key in ['X_minus_Y', 'X_minus_Z', 'X_minus_W', 'W_minus_Y']:
            gaps = [r['gaps'][gap_key] for r in recs]
            entry[f'gap_{gap_key}_mean'] = float(np.mean(gaps))
            entry[f'gap_{gap_key}_std']  = float(np.std(gaps))
        entry['n_seeds'] = len(recs)
        summary.setdefault(dataset, {}) \
               .setdefault(split_type, {}) \
               .setdefault(optimizer, {})[lr_s] = entry
    return summary


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(mode: str, device: torch.device):
    """
    cora / fixed / adam / lr=0.01 / seeds 0, 1, 2 only.
    Prints final test accuracies for all four conditions and the W_minus_Y gap,
    with automated interpretation.
    """
    print('\n' + '=' * 68)
    print(f'SANITY CHECK  cora | fixed | adam | lr=0.01 | seeds 0,1,2 | {mode}')
    print('=' * 68)

    scaled   = (mode == 'scaled')
    dataset  = 'cora'
    d_path   = os.path.join(DATA_ROOT, dataset)
    required = ['X.npy', 'Y.npy', 'Z.npy', 'W.npy', 'labels.npy',
                'fixed_train_mask.npy', 'fixed_val_mask.npy', 'fixed_test_mask.npy']
    if any(not os.path.isfile(os.path.join(d_path, f)) for f in required):
        print('Sanity check skipped: data files missing.')
        return

    X_full, Y_full, Z_full, W_full, labels = load_dataset_arrays(dataset)
    tr_mask, va_mask, te_mask = load_split_masks(dataset, 'fixed', 0)
    num_classes = int(labels.max()) + 1
    y_tr = labels[tr_mask]
    y_va = labels[va_mask]
    y_te = labels[te_mask]

    cond_data = preprocess_conditions(
        X_full, Y_full, Z_full, W_full,
        tr_mask, va_mask, te_mask, scaled
    )

    wmy_gaps = []
    print(f'  {"Seed":>4}  {"X%":>7}  {"Y%":>7}  {"Z%":>7}  {"W%":>7}  {"W−Y pp":>8}')
    print(f'  {"----":>4}  {"-------":>7}  {"-------":>7}  {"-------":>7}  '
          f'{"-------":>7}  {"--------":>8}')

    for seed in [0, 1, 2]:
        finals = {}
        for cond in CONDITIONS:
            feat_tr, feat_va, feat_te = cond_data[cond]
            _, _, _, ft = train_softmax(
                feat_tr, y_tr, feat_va, y_va, feat_te, y_te,
                num_classes, 'adam', 0.01, seed, device
            )
            finals[cond] = ft * 100.0
        wmy = finals['W'] - finals['Y']
        wmy_gaps.append(wmy)
        print(f'  {seed:>4}  {finals["X"]:>7.2f}  {finals["Y"]:>7.2f}  '
              f'{finals["Z"]:>7.2f}  {finals["W"]:>7.2f}  {wmy:>+8.2f}')

    mean_wmy = float(np.mean(wmy_gaps))
    print(f'\n  Mean W−Y gap: {mean_wmy:+.2f} pp')

    if mean_wmy > 3.0:
        interp = ('Laplacian basis change causes extra damage beyond '
                  'D-orthogonalization alone')
    elif mean_wmy < 1.0:
        interp = ('D-orthogonalization alone explains the gap; '
                  'basis change is not the primary issue')
    else:
        interp = 'Inconclusive: both D-orthogonalization and basis change contribute'
    print(f'  → {interp}')

    if not scaled:
        print()
        print('  NOTE (unscaled mode): Only W_minus_Y is a valid comparison here.')
        print('  X_minus_Z and X_minus_W are confounded by scale differences in X.')

    print('=' * 68 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='D-Wadia control experiment: X vs Y vs Z (std ZCA) vs W (D-ZCA)'
    )
    parser.add_argument('--dataset',   type=str,   default=None,
                        help='Single dataset to run (default: all 9)')
    parser.add_argument('--optimizer', type=str,   choices=['sgd', 'adam'], default=None,
                        help='Single optimizer (default: both)')
    parser.add_argument('--lr',        type=float, default=None,
                        help='Single learning rate (default: all three)')
    parser.add_argument('--mode',      type=str,
                        choices=['scaled', 'unscaled', 'both'], default='both',
                        help='Preprocessing mode (default: both)')
    args = parser.parse_args()

    datasets   = [args.dataset]   if args.dataset   else ALL_DATASETS
    optimizers = [args.optimizer] if args.optimizer  else OPTIMIZERS
    lrs        = [args.lr]        if args.lr         else LEARNING_RATES
    modes      = ['scaled', 'unscaled'] if args.mode == 'both' else [args.mode]

    # Verify data files exist before starting
    for ds in datasets:
        dp = os.path.join(DATA_ROOT, ds)
        missing = [f for f in ['X.npy', 'Y.npy', 'Z.npy', 'W.npy']
                   if not os.path.isfile(os.path.join(dp, f))]
        if missing:
            print(f'ERROR: Missing files for dataset "{ds}": {missing}')
            print('Run save_data.py first to generate Z and W.')
            sys.exit(1)

    device = torch.device('cpu')

    print('SpectralGeometryMisalignment — d_wadia_experiment.py')
    print(f'Datasets:    {datasets}')
    print(f'Optimizers:  {optimizers}')
    print(f'LRs:         {lrs}')
    print(f'Modes:       {modes}')
    print(f'Epochs:      {EPOCHS}  (no early stopping)')
    print(f'Train seeds: {len(TRAIN_SEEDS)}  (0–14)')
    print(f'Results:     {RESULTS_ROOT}')
    print()

    for mode in modes:
        print(f'\n{"="*70}')
        print(f'MODE: {mode.upper()}')
        if mode == 'unscaled':
            print('NOTE: In unscaled mode only W_minus_Y is interpretable.')
            print('      X_minus_Z and X_minus_W are confounded by scale in raw X.')
        print(f'{"="*70}')

        all_records = []

        for dataset in datasets:
            for optimizer_name in optimizers:
                for lr in lrs:
                    print(f'\n{"─"*70}')
                    print(f'Config: {dataset} | {optimizer_name} | lr={lr} | {mode}')
                    print(f'{"─"*70}')

                    records = run_config(dataset, optimizer_name, lr, mode, device)
                    all_records.extend(records)

                    # Config-level summary across all splits and seeds
                    print(f'  OVERALL [{dataset} | {optimizer_name} | lr={lr} | {mode}]')
                    for cond in CONDITIONS:
                        accs = [r[cond]['final_test_acc'] * 100 for r in records]
                        print(f'    {cond}: {np.mean(accs):.2f} ± {np.std(accs):.2f}%')
                    wmy = [r['gaps']['W_minus_Y'] for r in records]
                    print(f'    W−Y: {np.mean(wmy):+.2f} ± {np.std(wmy):.2f} pp\n')

        # Save summary for this mode
        summary     = build_summary(all_records)
        summary_dir = os.path.join(RESULTS_ROOT, mode)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Summary saved to: {summary_path}')

        # Sanity check
        sanity_check(mode, device)

    print('Done.')
