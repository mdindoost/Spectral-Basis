# -*- coding: utf-8 -*-
"""
Create tables and plots for Investigation 2 (Directions A & B), covering both fixed and random splits.

Outputs:
  tables/summary_<split>.csv
  tables/convergence_<split>.csv
  figures/acc_bars_<split>.png
  figures/basis_model_delta_<split>.png
  figures/convergence_bars_<split>.png
  figures/basis_delta_vs_rank_<split>.png  (optional if rank info available)

Assumes JSONs at:
  results/investigation2_directions_AB/<dataset>/{fixed_splits|random_splits}/metrics/results_aggregated.json
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
DATASETS = [
    ('ogbn-arxiv', 'OGB'),
    ('cora', 'Planetoid'),
    ('citeseer', 'Planetoid'),
    ('pubmed', 'Planetoid'),
    ('wikics', 'WikiCS'),
    ('amazon-photo', 'Amazon'),
    ('amazon-computers', 'Amazon'),
    ('coauthor-cs', 'Coauthor'),
    ('coauthor-physics', 'Coauthor'),
]
SPLITS = ['fixed_splits', 'random_splits']  # process both

ROOT = Path('results/investigation2_directions_AB')
OUT_TABLES = Path('tables')
OUT_FIGS = Path('figures')
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = {
    'a': 'standard_X_scaled',
    'b': 'standard_V_scaled',
    'c': 'rownorm_V',
}
MODEL_LABELS = {
    'a': '(a) X Std',
    'b': '(b) V Std',
    'c': '(c) V Row',
}

# -------------------------
# IO helpers
# -------------------------
def load_results_json(dataset: str, split: str) -> Optional[Dict[str, Any]]:
    path = ROOT / dataset / split / 'metrics' / 'results_aggregated.json'
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default

# -------------------------
# Aggregation
# -------------------------
def aggregate_one(dataset: str, source: str, split: str) -> Optional[Dict[str, Any]]:
    data = load_results_json(dataset, split)
    if data is None:
        return None

    def acc_block(model_key: str) -> Tuple[Optional[float], Optional[float]]:
        mean = safe_get(data, ['models', model_key, 'test_accuracy', 'mean'])
        std  = safe_get(data, ['models', model_key, 'test_accuracy', 'std'])
        return mean, std

    # Pull accuracy means/stds
    a_mean, a_std = acc_block(MODEL_KEYS['a'])
    b_mean, b_std = acc_block(MODEL_KEYS['b'])
    c_mean, c_std = acc_block(MODEL_KEYS['c'])

    # Effects
    basis_delta = None
    model_delta = None
    if a_mean is not None and a_mean > 0 and b_mean is not None:
        basis_delta = 100.0 * (b_mean - a_mean) / a_mean
    if b_mean is not None and b_mean > 0 and c_mean is not None:
        model_delta = 100.0 * (c_mean - b_mean) / b_mean

    # Rank info
    d_raw = safe_get(data, ['d_raw'])
    d_eff = safe_get(data, ['d_effective'])
    rank_def = safe_get(data, ['rank_deficiency'])
    ortho_err = safe_get(data, ['orthonormality_error'])

    # Convergence metrics (speed_to_90/95/99, auc)
    def conv_block(model_key: str, name: str):
        return safe_get(data, ['models', model_key, 'convergence_metrics', name, 'mean'])

    conv = {
        'a_speed90': conv_block(MODEL_KEYS['a'], 'speed_to_90'),
        'a_speed95': conv_block(MODEL_KEYS['a'], 'speed_to_95'),
        'a_speed99': conv_block(MODEL_KEYS['a'], 'speed_to_99'),
        'a_auc':     conv_block(MODEL_KEYS['a'], 'auc'),

        'b_speed90': conv_block(MODEL_KEYS['b'], 'speed_to_90'),
        'b_speed95': conv_block(MODEL_KEYS['b'], 'speed_to_95'),
        'b_speed99': conv_block(MODEL_KEYS['b'], 'speed_to_99'),
        'b_auc':     conv_block(MODEL_KEYS['b'], 'auc'),

        'c_speed90': conv_block(MODEL_KEYS['c'], 'speed_to_90'),
        'c_speed95': conv_block(MODEL_KEYS['c'], 'speed_to_95'),
        'c_speed99': conv_block(MODEL_KEYS['c'], 'speed_to_99'),
        'c_auc':     conv_block(MODEL_KEYS['c'], 'auc'),
    }

    return {
        'dataset': dataset,
        'source': source,
        'split': split,
        'd_raw': d_raw,
        'd_effective': d_eff,
        'rank_deficiency': rank_def,
        'orthonormality_error': ortho_err,

        'acc_a_mean': a_mean, 'acc_a_std': a_std,
        'acc_b_mean': b_mean, 'acc_b_std': b_std,
        'acc_c_mean': c_mean, 'acc_c_std': c_std,
        'basis_delta_pct': basis_delta,
        'model_delta_pct': model_delta,
        **conv
    }

def aggregate_all(split: str) -> pd.DataFrame:
    rows = []
    for dataset, source in DATASETS:
        row = aggregate_one(dataset, source, split)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Pretty columns for tables
    def fmt_dim(r):
        if pd.notnull(r['d_raw']) and pd.notnull(r['d_effective']) and r['d_raw'] != r['d_effective']:
            return f"{int(r['d_effective'])}/{int(r['d_raw'])}"
        elif pd.notnull(r['d_effective']):
            return str(int(r['d_effective']))
        return ""

    df['Dim'] = df.apply(fmt_dim, axis=1)
    df['RankDef'] = df['rank_deficiency'].map(lambda x: '⚠️' if bool(x) else '')
    return df

# -------------------------
# Tables
# -------------------------
def save_summary_table(df: pd.DataFrame, split: str):
    if df.empty:
        print(f"[{split}] No data found.")
        return

    tbl = pd.DataFrame({
        'Dataset': df['dataset'] + ' ' + df['RankDef'],
        'Source': df['source'],
        'Dim': df['Dim'],
        '(a) X Std': (df['acc_a_mean']*100).map(lambda x: f"{x:5.2f}%") + '±' + (df['acc_a_std']*100).map(lambda x: f"{x:4.2f}%"),
        '(b) V Std': (df['acc_b_mean']*100).map(lambda x: f"{x:5.2f}%") + '±' + (df['acc_b_std']*100).map(lambda x: f"{x:4.2f}%"),
        '(c) V Row': (df['acc_c_mean']*100).map(lambda x: f"{x:5.2f}%") + '±' + (df['acc_c_std']*100).map(lambda x: f"{x:4.2f}%"),
        'Basis Δ': df['basis_delta_pct'].map(lambda x: f"{x:+.1f}%" if pd.notnull(x) else ""),
        'Model Δ': df['model_delta_pct'].map(lambda x: f"{x:+.1f}%" if pd.notnull(x) else ""),
        'OrthoErr': df['orthonormality_error'].map(lambda x: f"{x:.2e}" if pd.notnull(x) else "")
    })

    out_csv = OUT_TABLES / f"summary_{split}.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"[{split}] Wrote table: {out_csv}")

def save_convergence_table(df: pd.DataFrame, split: str):
    if df.empty:
        return
    cols = [
        'dataset', 'source', 'Dim', 'RankDef',
        'a_speed90', 'a_speed95', 'a_speed99', 'a_auc',
        'b_speed90', 'b_speed95', 'b_speed99', 'b_auc',
        'c_speed90', 'c_speed95', 'c_speed99', 'c_auc',
    ]
    keep = [c for c in cols if c in df.columns]
    cdf = df[['dataset','source','Dim','RankDef']].copy()
    for c in keep:
        if c not in cdf.columns:
            cdf[c] = df[c] if c in df.columns else np.nan

    out_csv = OUT_TABLES / f"convergence_{split}.csv"
    cdf.to_csv(out_csv, index=False)
    print(f"[{split}] Wrote convergence table: {out_csv}")

# -------------------------
# Plots
# -------------------------
def plot_accuracy_bars(df: pd.DataFrame, split: str):
    if df.empty: return
    # datasets on x-axis, 3 bars per dataset for (a)(b)(c)
    labels = df['dataset'].tolist()
    a_vals = (df['acc_a_mean']*100).to_numpy()
    b_vals = (df['acc_b_mean']*100).to_numpy()
    c_vals = (df['acc_c_mean']*100).to_numpy()

    x = np.arange(len(labels))
    width = 0.26

    plt.figure(figsize=(max(9, len(labels)*0.9), 5))
    plt.bar(x - width, a_vals, width, label='(a) X Std')
    plt.bar(x,          b_vals, width, label='(b) V Std')
    plt.bar(x + width,  c_vals, width, label='(c) V Row')
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Accuracy by Model — {split.replace("_", " ")}')
    plt.legend()
    plt.tight_layout()
    out = OUT_FIGS / f"acc_bars_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

def plot_effect_bars(df: pd.DataFrame, split: str):
    if df.empty: return
    labels = df['dataset'].tolist()
    basis = df['basis_delta_pct'].to_numpy()
    model = df['model_delta_pct'].to_numpy()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(max(9, len(labels)*0.9), 5))
    plt.bar(x - width/2, basis, width, label='Basis Δ (b-a)/a %')
    plt.bar(x + width/2, model, width, label='Model Δ (c-b)/b %')
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('Relative Change (%)')
    plt.title(f'Basis and Model Effects — {split.replace("_", " ")}')
    plt.legend()
    plt.tight_layout()
    out = OUT_FIGS / f"basis_model_delta_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

def plot_convergence_bars(df: pd.DataFrame, split: str):
    if df.empty: return
    labels = df['dataset'].tolist()
    # any missing metrics remain NaN; matplotlib will skip
    def arr(prefix: str, metric: str):
        return df[f'{prefix}_{metric}'].to_numpy() if f'{prefix}_{metric}' in df.columns else np.full(len(labels), np.nan)

    a90, a95, a99, aAUC = arr('a','speed90'), arr('a','speed95'), arr('a','speed99'), arr('a','auc')
    b90, b95, b99, bAUC = arr('b','speed90'), arr('b','speed95'), arr('b','speed99'), arr('b','auc')
    c90, c95, c99, cAUC = arr('c','speed90'), arr('c','speed95'), arr('c','speed99'), arr('c','auc')

    # Plot speeds in epochs: grouped by model
    x = np.arange(len(labels))
    width = 0.22

    # speed_to_90
    plt.figure(figsize=(max(9, len(labels)*0.9), 5))
    plt.bar(x - width, a90, width, label='(a) speed@90')
    plt.bar(x,         b90, width, label='(b) speed@90')
    plt.bar(x + width, c90, width, label='(c) speed@90')
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('Epochs (lower is faster)')
    plt.title(f'Convergence: speed_to_90 — {split.replace("_", " ")}')
    plt.legend()
    plt.tight_layout()
    out = OUT_FIGS / f"convergence_speed90_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

    # speed_to_95
    plt.figure(figsize=(max(9, len(labels)*0.9), 5))
    plt.bar(x - width, a95, width, label='(a) speed@95')
    plt.bar(x,         b95, width, label='(b) speed@95')
    plt.bar(x + width, c95, width, label='(c) speed@95')
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('Epochs (lower is faster)')
    plt.title(f'Convergence: speed_to_95 — {split.replace("_", " ")}')
    plt.legend()
    plt.tight_layout()
    out = OUT_FIGS / f"convergence_speed95_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

    # speed_to_99 (only if you logged it—otherwise many NaNs, which is fine)
    plt.figure(figsize=(max(9, len(labels)*0.9), 5))
    plt.bar(x - width, a99, width, label='(a) speed@99')
    plt.bar(x,         b99, width, label='(b) speed@99')
    plt.bar(x + width, c99, width, label='(c) speed@99')
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('Epochs (lower is faster)')
    plt.title(f'Convergence: speed_to_99 — {split.replace("_", " ")}')
    plt.legend()
    plt.tight_layout()
    out = OUT_FIGS / f"convergence_speed99_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

    # AUC
    plt.figure(figsize=(max(9, len(labels)*0.9), 5))
    plt.bar(x - width, aAUC, width, label='(a) AUC')
    plt.bar(x,         bAUC, width, label='(b) AUC')
    plt.bar(x + width, cAUC, width, label='(c) AUC')
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('Normalized AUC (higher is better)')
    plt.title(f'Validation AUC — {split.replace("_", " ")}')
    plt.legend()
    plt.tight_layout()
    out = OUT_FIGS / f"convergence_auc_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

def plot_basis_vs_rank(df: pd.DataFrame, split: str):
    # Optional: correlate basis sensitivity with effective rank
    if df.empty: return
    if 'd_effective' not in df or df['basis_delta_pct'].isnull().all():
        return

    x = df['d_effective']
    y = df['basis_delta_pct']
    mask = (~x.isnull()) & (~y.isnull())
    if mask.sum() < 2:
        return

    plt.figure(figsize=(6.5, 5))
    plt.scatter(x[mask], y[mask])
    for _, r in df[mask].iterrows():
        plt.annotate(r['dataset'], (r['d_effective'], r['basis_delta_pct']), fontsize=8, xytext=(2,2), textcoords='offset points')
    plt.xlabel('Effective feature dimension (d_effective)')
    plt.ylabel('Basis Δ (%)')
    plt.title(f'Basis Sensitivity vs Effective Rank — {split.replace("_", " ")}')
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    out = OUT_FIGS / f"basis_delta_vs_rank_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[{split}] Wrote figure: {out}")

# -------------------------
# Main
# -------------------------
def main():
    for split in SPLITS:
        df = aggregate_all(split)

        # Save tables
        save_summary_table(df, split)
        save_convergence_table(df, split)

        # Plots
        plot_accuracy_bars(df, split)
        plot_effect_bars(df, split)
        plot_convergence_bars(df, split)
        plot_basis_vs_rank(df, split)

    print("\nDone. See 'tables/' and 'figures/'.")

if __name__ == "__main__":
    main()
