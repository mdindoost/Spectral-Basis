"""
Resume the unscaled d_wadia experiment for coauthor_physics,
then rebuild unscaled/summary.json from all existing JSON files.

Usage:
    /home/md724/Spectral-Basis/venv/bin/python \
        SpectralGeometryMisalignment/resume_unscaled.py
"""
import os
import sys
import json
import numpy as np
import torch

# ── Import shared code from d_wadia_experiment.py ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from d_wadia_experiment import (
    ALL_DATASETS, TRAIN_SEEDS, RANDOM_SPLIT_SEEDS, RESULTS_ROOT,
    CONDITIONS, EPOCHS,
    load_dataset_arrays, load_split_masks, preprocess_conditions,
    train_softmax, save_result, result_filename, build_summary, lr_str,
)

OPTIMIZERS     = ['sgd', 'adam']
LEARNING_RATES = [0.001, 0.01, 0.1]
MODE           = 'unscaled'


def find_missing_files(dataset):
    """Return list of (split_type, split_seed, optimizer, lr, train_seed) tuples
    whose JSON files do not yet exist in the unscaled results directory."""
    missing = []
    for split_type in ['fixed', 'random']:
        split_seeds = [0] if split_type == 'fixed' else RANDOM_SPLIT_SEEDS
        for split_seed in split_seeds:
            for optimizer in OPTIMIZERS:
                for lr in LEARNING_RATES:
                    for train_seed in TRAIN_SEEDS:
                        fname = result_filename(dataset, split_type, split_seed,
                                                optimizer, lr, train_seed)
                        fpath = os.path.join(RESULTS_ROOT, MODE, dataset, fname)
                        if not os.path.isfile(fpath):
                            missing.append((split_type, split_seed,
                                            optimizer, lr, train_seed))
    return missing


def run_missing(dataset, missing, device):
    """Run only the missing (split_type, split_seed, optimizer, lr, seed) combos."""
    if not missing:
        print(f'  {dataset}: nothing missing, skipping.')
        return

    print(f'  {dataset}: {len(missing)} files to generate.')
    X_full, Y_full, Z_full, W_full, labels = load_dataset_arrays(dataset)
    num_classes = int(labels.max()) + 1

    # Group by (split_type, split_seed, optimizer, lr) to avoid reloading data
    from collections import defaultdict
    groups = defaultdict(list)
    for (st, ss, opt, lr, seed) in missing:
        groups[(st, ss, opt, lr)].append(seed)

    done = 0
    for (split_type, split_seed, optimizer_name, lr), seeds in sorted(groups.items()):
        tr_mask, va_mask, te_mask = load_split_masks(dataset, split_type, split_seed)
        y_tr = labels[tr_mask]
        y_va = labels[va_mask]
        y_te = labels[te_mask]
        cond_data = preprocess_conditions(
            X_full, Y_full, Z_full, W_full,
            tr_mask, va_mask, te_mask, scaled=False
        )
        for train_seed in sorted(seeds):
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
            record = {
                'dataset':    dataset,
                'mode':       MODE,
                'split_type': split_type,
                'split_seed': split_seed,
                'optimizer':  optimizer_name,
                'lr':         lr,
                'train_seed': train_seed,
                'X': cond_outputs['X'],
                'Y': cond_outputs['Y'],
                'Z': cond_outputs['Z'],
                'W': cond_outputs['W'],
                'gaps': gaps,
            }
            save_result(MODE, dataset, split_type, split_seed,
                        optimizer_name, lr, train_seed, record)
            done += 1
            if done % 15 == 0:
                print(f'    [{dataset} | {split_type} s{split_seed} | '
                      f'{optimizer_name} | lr={lr} | seed={train_seed:>2}]  '
                      f'X={fX*100:.1f}%  Y={fY*100:.1f}%  '
                      f'Z={fZ*100:.1f}%  W={fW*100:.1f}%  '
                      f'W-Y={gaps["W_minus_Y"]:+.1f}pp')

    print(f'  {dataset}: done. {done} new files written.')


def rebuild_summary():
    """Read all existing unscaled JSON files and rebuild summary.json."""
    unscaled_root = os.path.join(RESULTS_ROOT, MODE)
    all_records = []
    n_files = 0
    for dataset in ALL_DATASETS:
        ds_dir = os.path.join(unscaled_root, dataset)
        if not os.path.isdir(ds_dir):
            print(f'  WARNING: no directory for {dataset}, skipping.')
            continue
        for fname in os.listdir(ds_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(ds_dir, fname)
            with open(fpath) as f:
                record = json.load(f)
            all_records.append(record)
            n_files += 1
    print(f'  Loaded {n_files} JSON files across {len(ALL_DATASETS)} datasets.')
    summary = build_summary(all_records)
    summary_path = os.path.join(unscaled_root, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Summary saved to: {summary_path}')
    return summary


if __name__ == '__main__':
    device = torch.device('cpu')

    # Step 1: fill in missing files (only coauthor_physics expected to be incomplete)
    print('Step 1: checking for missing files...')
    for dataset in ALL_DATASETS:
        missing = find_missing_files(dataset)
        if missing:
            print(f'\n  {dataset}: {len(missing)} missing — running now...')
            run_missing(dataset, missing, device)
        else:
            print(f'  {dataset}: complete ({540} / 540)')

    # Step 2: rebuild summary.json from all existing files
    print('\nStep 2: rebuilding unscaled summary.json...')
    rebuild_summary()

    print('\nDone.')
