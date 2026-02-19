"""
Complete Eigenvalue Distribution Analysis
==========================================
Adds missing datasets (amazon-photo, coauthor-physics, wikics) to existing
eigenvalue analysis and runs full correlation analysis with RowNorm recovery.

Usage:
    python experiments/complete_eigenvalue_analysis.py
"""

import sys
import os
sys.path.append('/home/md724/Spectral-Basis/experiments')

import json
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import networkx as nx

from utils import (
    load_dataset,
    build_graph_matrices,
    compute_restricted_eigenvectors,
    sgc_precompute,
    compute_sgc_normalized_adjacency,
    get_largest_connected_component_nx,
)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = '/home/md724/Spectral-Basis'
OUTPUT_DIR = f'{BASE_DIR}/results/eigenvalue_analysis'
DATA_DIR = f'{OUTPUT_DIR}/data'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'
CACHE_DIR = f'{OUTPUT_DIR}/eigenvalue_cache'
WHITENING_DIR = f'{BASE_DIR}/results/investigation_whitening_rownorm'
DATASET_ROOT = f'{BASE_DIR}/dataset'

for d in [DATA_DIR, PLOTS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

DATASETS_ALL = [
    'cora', 'citeseer', 'pubmed',
    'coauthor-cs', 'coauthor-physics',
    'amazon-computers', 'amazon-photo',
    'ogbn-arxiv', 'wikics'
]

COMPLETED = ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 'amazon-computers', 'coauthor-cs']
TODO = ['amazon-photo', 'coauthor-physics', 'wikics']

K_VALUES = [2, 4, 6, 8, 10]
EIG_THRESHOLD = 1e-10

DATASET_FAMILIES = {
    'cora': 'citation', 'citeseer': 'citation', 'pubmed': 'citation',
    'coauthor-cs': 'coauthorship', 'coauthor-physics': 'coauthorship',
    'amazon-computers': 'product', 'amazon-photo': 'product',
    'ogbn-arxiv': 'large', 'wikics': 'wikipedia',
}

FAMILY_COLORS = {
    'citation': '#2196F3', 'coauthorship': '#4CAF50',
    'product': '#FF9800', 'large': '#9C27B0', 'wikipedia': '#F44336',
}

# ============================================================================
# Helper Functions
# ============================================================================

def compute_eigenvalue_stats(eigenvalues_sorted):
    """
    Compute statistics for an array of sorted eigenvalues.
    eigenvalues_sorted: full sorted array (may include near-zero values).
    Returns dict matching the existing JSON structure.
    """
    eigs_all = np.array(eigenvalues_sorted)
    eigs_valid = eigs_all[eigs_all > EIG_THRESHOLD]
    num_dropped = int(np.sum(eigs_all <= EIG_THRESHOLD))

    if len(eigs_valid) == 0:
        return {
            'eigenvalues_all': eigs_all.tolist(),
            'eigenvalues_valid': [],
            'num_near_zero': num_dropped,
            'num_dropped': num_dropped,
            'condition': float('inf'),
            'rank': 0,
            'statistics': {}
        }

    cond = float(eigs_valid[-1] / eigs_valid[0]) if eigs_valid[0] > 0 else float('inf')
    mean = float(eigs_valid.mean())
    std = float(eigs_valid.std())
    skewness = float(np.mean(((eigs_valid - mean) / (std + 1e-15)) ** 3))
    gap_ratio = float(eigs_valid[-1] / eigs_valid[-2]) if len(eigs_valid) > 1 else 1.0
    q25, q75 = np.percentile(eigs_valid, [25, 75])

    stats = {
        'mean': mean,
        'median': float(np.median(eigs_valid)),
        'std': std,
        'min': float(eigs_valid.min()),
        'max': float(eigs_valid.max()),
        'range': float(eigs_valid.max() - eigs_valid.min()),
        'iqr': float(q75 - q25),
        'num_valid': len(eigs_valid),
        'condition': cond,
        'skewness': skewness,
        'gap_ratio': gap_ratio,
    }

    return {
        'eigenvalues_all': eigs_all.tolist(),
        'eigenvalues_valid': eigs_valid.tolist(),
        'num_near_zero': num_dropped,
        'num_dropped': num_dropped,
        'condition': cond,
        'rank': len(eigs_all),
        'statistics': stats,
    }


def compute_hypothesis_metrics(eigenvalues_valid):
    """Compute condition number, entropy, and effective rank for hypothesis testing."""
    eigs = np.array(eigenvalues_valid)
    eigs = eigs[eigs > EIG_THRESHOLD]

    if len(eigs) == 0:
        return {'kappa': float('inf'), 'entropy': 0.0, 'effective_rank': 0}

    kappa = float(eigs[-1] / eigs[0]) if eigs[0] > 0 else float('inf')

    p = eigs / eigs.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-15)))

    cumsum = np.cumsum(eigs) / eigs.sum()
    eff_rank = int(np.searchsorted(cumsum, 0.95)) + 1

    return {'kappa': kappa, 'entropy': entropy, 'effective_rank': eff_rank}


def process_dataset(dataset_name):
    """
    Load full graph, compute restricted eigenproblem eigenvalues at baseline
    and k=2,4,6,8,10 diffusion steps. Returns data dict matching existing JSON.
    """
    cache_file = f'{CACHE_DIR}/{dataset_name}_result.json'
    if os.path.exists(cache_file):
        print(f'  Loading from cache: {cache_file}')
        with open(cache_file, 'r') as f:
            return json.load(f)

    print(f'  Loading dataset {dataset_name}...')
    edge_index, features, labels, num_nodes, num_classes, _, _, _ = load_dataset(
        dataset_name, root=DATASET_ROOT
    )

    # Convert to numpy
    if hasattr(features, 'numpy'):
        features = features.numpy()
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()
    features = features.astype(np.float64)

    print(f'  n={num_nodes}, d={features.shape[1]}, classes={num_classes}')

    # Build graph matrices on FULL graph
    print('  Building graph matrices...')
    adj, L, D = build_graph_matrices(edge_index, num_nodes)

    # Count connected components (using graph without extra self-loops)
    G = nx.from_scipy_sparse_array(adj)
    num_components = nx.number_connected_components(G)
    print(f'  Connected components: {num_components}')

    # Compute SGC-normalized adjacency for diffusion.
    # NOTE: compute_sgc_normalized_adjacency adds self-loops internally.
    # adj already has self-loops from build_graph_matrices. This mirrors
    # the protocol used in investigation_whitening_rownorm.py.
    print('  Computing normalized adjacency...')
    adj_normalized = compute_sgc_normalized_adjacency(adj)

    # ---- Baseline: restricted eigenproblem on raw features ----
    print('  Baseline (raw features)...')
    try:
        _, eigs_baseline, _, _ = compute_restricted_eigenvectors(
            features, L, D, num_components=0
        )
        eigs_baseline = np.sort(eigs_baseline)
    except Exception as e:
        print(f'  ERROR baseline: {e}')
        return None

    baseline_data = compute_eigenvalue_stats(eigs_baseline)

    # ---- Diffused: at k=2,4,6,8,10 ----
    diffused_data = {}
    X_curr = features.copy()
    prev_k = 0

    for k in K_VALUES:
        steps = k - prev_k
        print(f'  Applying {steps} more diffusion steps (total k={k})...')
        X_curr = sgc_precompute(X_curr, adj_normalized, steps)
        if isinstance(X_curr, np.matrix):
            X_curr = np.asarray(X_curr)
        prev_k = k

        try:
            _, eigs_k, _, _ = compute_restricted_eigenvectors(
                X_curr, L, D, num_components=0
            )
            eigs_k = np.sort(eigs_k)
            diffused_data[str(k)] = compute_eigenvalue_stats(eigs_k)
            cond = diffused_data[str(k)]['condition']
            print(f'    k={k}: {len(eigs_k)} eigenvalues, kappa={cond:.2f}')
        except Exception as e:
            print(f'    ERROR at k={k}: {e}')
            diffused_data[str(k)] = None

    result = {
        'metadata': {
            'num_nodes': int(num_nodes),
            'num_features': int(features.shape[1]),
            'num_classes': int(num_classes),
            'num_components': int(num_components),
        },
        'baseline': baseline_data,
        'diffused': diffused_data,
    }

    # Cache result
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    print(f'  Cached to: {cache_file}')

    return result


# ============================================================================
# Task 1: Complete Missing Datasets
# ============================================================================

print('=' * 70)
print('TASK 1: Complete Missing Datasets')
print('=' * 70)

existing_json_path = f'{DATA_DIR}/eigenvalue_analysis_complete.json'
print(f'Loading existing data from {existing_json_path}...')
with open(existing_json_path, 'r') as f:
    all_data = json.load(f)
print(f'Existing datasets: {list(all_data.keys())}')

for dataset in TODO:
    if dataset in all_data:
        print(f'\n{dataset}: already in JSON, skipping.')
        continue

    print(f'\nProcessing: {dataset}')
    print('-' * 50)
    result = process_dataset(dataset)
    if result is not None:
        all_data[dataset] = result
        print(f'  Added {dataset} to data.')
    else:
        print(f'  FAILED to process {dataset}.')

# Save updated JSON
with open(existing_json_path, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f'\nUpdated JSON saved: {existing_json_path}')
print(f'Total datasets: {len(all_data)}')

# ============================================================================
# Task 2: Update Summary Table
# ============================================================================

print('\n' + '=' * 70)
print('Updating summary table...')
print('=' * 70)

rows = []
for dataset in DATASETS_ALL:
    if dataset not in all_data:
        continue
    d = all_data[dataset]
    meta = d['metadata']
    bl = d['baseline']
    row = {
        'Dataset': dataset,
        'Nodes': meta['num_nodes'],
        'Features': meta['num_features'],
        'Components': meta['num_components'],
        'Dropped_Eigs': bl['num_dropped'],
        'Valid_Eigs': bl['statistics'].get('num_valid', bl['rank']),
        'Baseline_Cond': bl['condition'],
        'Baseline_Spread': bl['statistics'].get('std', float('nan')),
        'Baseline_Skewness': bl['statistics'].get('skewness', float('nan')),
    }
    for k in K_VALUES:
        diff = d['diffused'].get(str(k))
        if diff:
            row[f'k{k}_Cond'] = diff['condition']
            row[f'k{k}_Spread'] = diff['statistics'].get('std', float('nan'))
        else:
            row[f'k{k}_Cond'] = float('nan')
            row[f'k{k}_Spread'] = float('nan')
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv(f'{DATA_DIR}/summary_table.csv', index=False)
print(df_summary[['Dataset', 'Nodes', 'Baseline_Cond', 'k10_Cond']].to_string(index=False))

# ============================================================================
# Task 3: Load RowNorm Recovery Values
# ============================================================================

print('\n' + '=' * 70)
print('TASK 2: Load RowNorm Recovery Values (k=10, Whitening Rownorm)')
print('=' * 70)

rownorm_recovery = {}
for dataset in DATASETS_ALL:
    rownorm_recovery[dataset] = {}
    for split in ['fixed', 'random']:
        path = f'{WHITENING_DIR}/{dataset}_{split}_lcc_k10/metrics/results.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                res = json.load(f)
            fr = res['final_results']
            rr_mlp = fr['rayleigh_ritz_MLP']['test_acc_mean']
            rr_mlp_rn = fr['rayleigh_ritz_MLP+RowNorm']['test_acc_mean']
            recovery = (rr_mlp_rn - rr_mlp) * 100
            rownorm_recovery[dataset][split] = round(recovery, 4)
        else:
            rownorm_recovery[dataset][split] = None
            print(f'  MISSING: {dataset} {split} ({path})')

# Validate averages
fixed_vals = [v['fixed'] for v in rownorm_recovery.values() if v.get('fixed') is not None]
random_vals = [v['random'] for v in rownorm_recovery.values() if v.get('random') is not None]
print(f'\nRowNorm Recovery Summary:')
print(f'  Fixed splits avg:  {np.mean(fixed_vals):.2f}pp  (N={len(fixed_vals)})')
print(f'  Random splits avg: {np.mean(random_vals):.2f}pp  (N={len(random_vals)})')
print(f'\nPer-dataset values:')
for ds in DATASETS_ALL:
    fv = rownorm_recovery[ds].get('fixed')
    rv = rownorm_recovery[ds].get('random')
    fs = f'{fv:+.2f}pp' if fv is not None else 'MISSING'
    rs = f'{rv:+.2f}pp' if rv is not None else 'MISSING'
    print(f'  {ds:20s}: fixed={fs}, random={rs}')

# ============================================================================
# Task 4: Prepare Correlation Dataset
# ============================================================================

print('\n' + '=' * 70)
print('TASK 3: Correlation Analysis (k=10 eigenvalue metrics)')
print('=' * 70)

corr_rows = []
for dataset in DATASETS_ALL:
    if dataset not in all_data:
        print(f'  SKIP {dataset}: no eigenvalue data')
        continue

    d = all_data[dataset]
    diff_10 = d['diffused'].get('10')
    if diff_10 is None:
        print(f'  SKIP {dataset}: no k=10 diffused data')
        continue

    eigs_valid = np.array(diff_10['eigenvalues_valid'])
    h_metrics = compute_hypothesis_metrics(eigs_valid)

    fv = rownorm_recovery[dataset].get('fixed')
    rv = rownorm_recovery[dataset].get('random')

    corr_rows.append({
        'dataset': dataset,
        'family': DATASET_FAMILIES.get(dataset, 'other'),
        'kappa': h_metrics['kappa'],
        'entropy': h_metrics['entropy'],
        'effective_rank': h_metrics['effective_rank'],
        'rownorm_fixed': fv,
        'rownorm_random': rv,
        'n_nodes': d['metadata']['num_nodes'],
        'n_valid_eigs': len(eigs_valid),
    })

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(f'{OUTPUT_DIR}/eigenvalue_metrics_complete.csv', index=False)

print(f'\nCorrelation dataset ({len(df_corr)} datasets):')
cols = ['dataset', 'kappa', 'entropy', 'effective_rank', 'rownorm_fixed', 'rownorm_random']
print(df_corr[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# ============================================================================
# Correlation Statistics
# ============================================================================

print('\n--- Statistical Correlation Analysis ---\n')

metrics_to_test = [
    ('kappa', 'Condition Number (κ)'),
    ('entropy', 'Eigenvalue Entropy H(λ)'),
    ('effective_rank', 'Effective Rank (95%)'),
]

corr_results = {}
for metric, name in metrics_to_test:
    print(f'{name}:')
    corr_results[metric] = {}
    for split_col, split_name in [('rownorm_fixed', 'Fixed'), ('rownorm_random', 'Random')]:
        valid = df_corr[[metric, split_col]].dropna()
        n = len(valid)
        if n < 3:
            print(f'  {split_name}: Insufficient data (N={n})')
            continue
        r, p = pearsonr(valid[metric], valid[split_col])
        rho, p_sp = spearmanr(valid[metric], valid[split_col])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        corr_results[metric][split_col] = {
            'pearson_r': float(r), 'pearson_p': float(p),
            'spearman_rho': float(rho), 'spearman_p': float(p_sp),
            'n': n, 'significant': sig,
        }
        print(f'  {split_name}: Pearson r={r:+.3f} (p={p:.4f} {sig}), '
              f'Spearman ρ={rho:+.3f} (p={p_sp:.4f}), N={n}')
    print()

# Save correlation results text
corr_text_path = f'{OUTPUT_DIR}/correlation_results.txt'
with open(corr_text_path, 'w') as f:
    f.write('CORRELATION ANALYSIS: Eigenvalue Metrics vs RowNorm Recovery (k=10)\n')
    f.write('=' * 70 + '\n\n')
    for metric, name in metrics_to_test:
        f.write(f'{name}:\n')
        for split_col in ['rownorm_fixed', 'rownorm_random']:
            if split_col in corr_results.get(metric, {}):
                v = corr_results[metric][split_col]
                f.write(f'  {split_col}:  '
                        f'Pearson r={v["pearson_r"]:+.4f} (p={v["pearson_p"]:.6f} {v["significant"]}), '
                        f'Spearman ρ={v["spearman_rho"]:+.4f} (p={v["spearman_p"]:.6f}), '
                        f'N={v["n"]}\n')
        f.write('\n')
    f.write(f'\nRaw RowNorm Recovery Values:\n')
    f.write(f'{"Dataset":20s} {"Fixed":>10s} {"Random":>10s}\n')
    f.write('-' * 45 + '\n')
    for ds in DATASETS_ALL:
        fv = rownorm_recovery[ds].get('fixed')
        rv = rownorm_recovery[ds].get('random')
        f.write(f'{ds:20s} {(f"{fv:+.2f}pp" if fv is not None else "MISSING"):>10s} '
                f'{(f"{rv:+.2f}pp" if rv is not None else "MISSING"):>10s}\n')
    fv_mean = np.mean([v for v in [rownorm_recovery[d].get("fixed") for d in DATASETS_ALL] if v is not None])
    rv_mean = np.mean([v for v in [rownorm_recovery[d].get("random") for d in DATASETS_ALL] if v is not None])
    f.write(f'\nAverage fixed: {fv_mean:+.2f}pp\n')
    f.write(f'Average random: {rv_mean:+.2f}pp\n')

print(f'Saved: {corr_text_path}')

# ============================================================================
# FIGURES
# ============================================================================

print('\n' + '=' * 70)
print('TASK 4: Generating Figures')
print('=' * 70)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.dpi': 150,
})

# ---- Figure 1: Eigenvalue Spectra (3x3 grid) ----
print('\nFigure 1: Eigenvalue spectra 3x3 grid...')

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Eigenvalue Spectra at k=10 (Restricted Eigenproblem)',
             fontsize=14, fontweight='bold', y=1.01)

for ax, dataset in zip(axes.flat, DATASETS_ALL):
    if dataset not in all_data:
        ax.text(0.5, 0.5, f'{dataset}\n(no data)', ha='center', va='center',
                transform=ax.transAxes)
        continue

    diff_10 = all_data[dataset]['diffused'].get('10')
    if diff_10 is None:
        ax.text(0.5, 0.5, f'{dataset}\n(k=10 missing)', ha='center', va='center',
                transform=ax.transAxes)
        continue

    eigs = np.array(diff_10['eigenvalues_valid'])
    if len(eigs) == 0:
        ax.text(0.5, 0.5, f'{dataset}\n(no valid eigs)', ha='center', va='center',
                transform=ax.transAxes)
        continue

    h_metrics = compute_hypothesis_metrics(eigs)
    fam = DATASET_FAMILIES.get(dataset, 'other')
    color = FAMILY_COLORS.get(fam, '#666666')

    rr_fixed = rownorm_recovery[dataset].get('fixed')
    rr_val_str = f'{rr_fixed:+.1f}pp' if rr_fixed is not None else 'N/A'

    ax.plot(np.arange(len(eigs)), eigs, '-', color=color, linewidth=1.2, alpha=0.8)
    ax.fill_between(np.arange(len(eigs)), eigs, alpha=0.15, color=color)
    ax.set_title(f'{dataset}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Eigenvalue index', fontsize=9)
    ax.set_ylabel('λ', fontsize=9)

    # Annotate with metrics
    info = (f'κ={h_metrics["kappa"]:.1f}\n'
            f'H={h_metrics["entropy"]:.2f}\n'
            f'eff_rank={h_metrics["effective_rank"]}\n'
            f'RN recov={rr_val_str}')
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = f'{PLOTS_DIR}/figure1_eigenvalue_spectra_all.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig1_path}')


# ---- Figure 2: Kappa vs Recovery ----
def make_correlation_scatter(df, x_col, y_col, x_label, title, output_path, log_x=False):
    """Create a scatter plot with correlation statistics."""
    valid = df[[x_col, y_col, 'dataset', 'family']].dropna()
    if len(valid) < 2:
        print(f'  Skipping {output_path}: insufficient data')
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for fam in valid['family'].unique():
        sub = valid[valid['family'] == fam]
        color = FAMILY_COLORS.get(fam, '#666666')
        ax.scatter(sub[x_col], sub[y_col], c=color, s=120,
                   label=fam.title(), edgecolors='black', linewidth=0.8, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row['dataset'], (row[x_col], row[y_col]),
                        textcoords='offset points', xytext=(6, 4), fontsize=8)

    r, p = pearsonr(valid[x_col], valid[y_col])
    rho, p_sp = spearmanr(valid[x_col], valid[y_col])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

    # Trend line
    x_arr = valid[x_col].values
    y_arr = valid[y_col].values
    if log_x:
        x_fit = np.log10(x_arr + 1e-10)
    else:
        x_fit = x_arr
    z = np.polyfit(x_fit, y_arr, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
    if log_x:
        ax.plot(x_line, p_fit(np.log10(x_line + 1e-10)),
                'r--', alpha=0.6, linewidth=1.5, label='Trend')
    else:
        ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.6, linewidth=1.5, label='Trend')

    if log_x:
        ax.set_xscale('log')

    stat_text = (f'Pearson r = {r:+.3f} (p={p:.4f} {sig})\n'
                 f'Spearman ρ = {rho:+.3f} (p={p_sp:.4f})\nN = {len(valid)}')
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('RowNorm Recovery (pp)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


print('\nFigure 2: Kappa vs RowNorm Recovery...')
make_correlation_scatter(
    df_corr, 'kappa', 'rownorm_fixed',
    'Condition Number κ (log scale)', 'Condition Number vs RowNorm Recovery (Fixed Splits)',
    f'{PLOTS_DIR}/figure2_kappa_vs_recovery_fixed.png', log_x=True
)
make_correlation_scatter(
    df_corr, 'kappa', 'rownorm_random',
    'Condition Number κ (log scale)', 'Condition Number vs RowNorm Recovery (Random Splits)',
    f'{PLOTS_DIR}/figure2b_kappa_vs_recovery_random.png', log_x=True
)

print('\nFigure 3: Entropy vs RowNorm Recovery...')
make_correlation_scatter(
    df_corr, 'entropy', 'rownorm_fixed',
    'Eigenvalue Entropy H(λ)', 'Entropy vs RowNorm Recovery (Fixed Splits)',
    f'{PLOTS_DIR}/figure3_entropy_vs_recovery_fixed.png'
)
make_correlation_scatter(
    df_corr, 'entropy', 'rownorm_random',
    'Eigenvalue Entropy H(λ)', 'Entropy vs RowNorm Recovery (Random Splits)',
    f'{PLOTS_DIR}/figure3b_entropy_vs_recovery_random.png'
)

# ---- Figure 4: Heatmap ----
print('\nFigure 4: Metrics heatmap...')

heatmap_cols = ['kappa', 'entropy', 'effective_rank', 'rownorm_fixed', 'rownorm_random']
heatmap_labels = ['Cond. κ', 'Entropy H(λ)', 'Eff. Rank', 'RN Recov.\n(Fixed)', 'RN Recov.\n(Random)']

df_heat = df_corr.set_index('dataset')[heatmap_cols].copy()
df_heat = df_heat.reindex(DATASETS_ALL)

# Normalize each column for color display
df_norm = df_heat.copy()
for col in heatmap_cols:
    col_data = df_norm[col].dropna()
    if col_data.std() > 0:
        df_norm[col] = (df_norm[col] - col_data.min()) / (col_data.max() - col_data.min())

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(df_norm.values.astype(float), cmap='RdYlGn', aspect='auto',
               vmin=0, vmax=1)

ax.set_xticks(np.arange(len(heatmap_cols)))
ax.set_xticklabels(heatmap_labels, fontsize=10)
ax.set_yticks(np.arange(len(df_heat)))
ax.set_yticklabels(df_heat.index.tolist(), fontsize=10)

# Annotate cells
for i in range(len(df_heat)):
    for j in range(len(heatmap_cols)):
        val = df_heat.iloc[i, j]
        if pd.notna(val):
            if heatmap_cols[j] in ['rownorm_fixed', 'rownorm_random']:
                cell_text = f'{val:+.1f}'
            elif heatmap_cols[j] == 'kappa':
                cell_text = f'{val:.1f}'
            elif heatmap_cols[j] == 'effective_rank':
                cell_text = f'{int(val)}'
            else:
                cell_text = f'{val:.2f}'
            norm_val = df_norm.iloc[i, j]
            text_color = 'white' if (pd.notna(norm_val) and (norm_val < 0.2 or norm_val > 0.8)) else 'black'
            ax.text(j, i, cell_text, ha='center', va='center', fontsize=9, color=text_color)

plt.colorbar(im, ax=ax, label='Normalized value (0=min, 1=max)', fraction=0.03)
ax.set_title('Eigenvalue Metrics and RowNorm Recovery Heatmap (k=10)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
fig4_path = f'{PLOTS_DIR}/figure4_metrics_heatmap.png'
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig4_path}')

# ---- Figure 5: Eigenvalue spectra comparison (baseline vs k10) ----
print('\nExtra figure: Baseline vs k=10 eigenvalue evolution...')

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Eigenvalue Evolution: Baseline (raw) vs k=10 (diffused)',
             fontsize=14, fontweight='bold', y=1.01)

for ax, dataset in zip(axes.flat, DATASETS_ALL):
    if dataset not in all_data:
        ax.text(0.5, 0.5, f'{dataset}\n(no data)', ha='center', va='center',
                transform=ax.transAxes)
        continue

    d = all_data[dataset]
    fam = DATASET_FAMILIES.get(dataset, 'other')
    color = FAMILY_COLORS.get(fam, '#666666')

    bl_eigs = np.array(d['baseline']['eigenvalues_valid'])
    diff_10 = d['diffused'].get('10')
    if diff_10:
        k10_eigs = np.array(diff_10['eigenvalues_valid'])
    else:
        k10_eigs = np.array([])

    n_show = min(len(bl_eigs), len(k10_eigs), 200) if len(k10_eigs) > 0 else min(len(bl_eigs), 200)

    if n_show > 0:
        ax.plot(np.linspace(0, 1, min(len(bl_eigs), 200)),
                bl_eigs[np.round(np.linspace(0, len(bl_eigs)-1, 200)).astype(int)],
                '-', color='#666666', linewidth=1.2, alpha=0.7, label='Baseline')
    if len(k10_eigs) > 0:
        ax.plot(np.linspace(0, 1, min(len(k10_eigs), 200)),
                k10_eigs[np.round(np.linspace(0, len(k10_eigs)-1, 200)).astype(int)],
                '-', color=color, linewidth=1.5, label='k=10')

    ax.set_title(f'{dataset}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized index', fontsize=8)
    ax.set_ylabel('λ', fontsize=8)
    if dataset == 'cora':
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
extra_path = f'{PLOTS_DIR}/figure_baseline_vs_k10_comparison.png'
plt.savefig(extra_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {extra_path}')

# ============================================================================
# Task 5: Analysis Report
# ============================================================================

print('\n' + '=' * 70)
print('TASK 5: Writing Analysis Report')
print('=' * 70)

def fmt_corr(metric, split_col):
    if metric in corr_results and split_col in corr_results[metric]:
        v = corr_results[metric][split_col]
        return (f"r={v['pearson_r']:+.3f} (p={v['pearson_p']:.4f} {v['significant']}), "
                f"ρ={v['spearman_rho']:+.3f}")
    return 'N/A'

report_path = f'{OUTPUT_DIR}/analysis_report_final.md'
with open(report_path, 'w') as f:
    f.write('# Eigenvalue Distribution Analysis: Final Report\n\n')
    f.write('**Hypothesis:** Eigenvalue distribution of the restricted Laplacian '
            'determines whether RowNorm succeeds.\n\n')
    f.write('- Well-separated eigenvalues (high κ) → diverse eigenvector directions → RowNorm succeeds\n')
    f.write('- Clustered eigenvalues (low κ) → nearly parallel eigenvectors → RowNorm fails\n\n')

    f.write('## Datasets\n\n')
    f.write('| Dataset | Family | κ | H(λ) | Eff. Rank | RN Fixed | RN Random |\n')
    f.write('|---------|--------|---|------|-----------|----------|-----------|\n')
    for _, row in df_corr.iterrows():
        kf = f"{row['rownorm_fixed']:+.2f}pp" if pd.notna(row.get('rownorm_fixed')) else 'N/A'
        kr = f"{row['rownorm_random']:+.2f}pp" if pd.notna(row.get('rownorm_random')) else 'N/A'
        f.write(f"| {row['dataset']} | {row['family']} | {row['kappa']:.2f} | "
                f"{row['entropy']:.3f} | {int(row['effective_rank'])} | {kf} | {kr} |\n")
    f.write('\n')

    f.write('## Correlation Results (k=10)\n\n')
    f.write('| Metric | Fixed Splits | Random Splits |\n')
    f.write('|--------|-------------|---------------|\n')
    for metric, name in metrics_to_test:
        f.write(f'| {name} | {fmt_corr(metric, "rownorm_fixed")} | '
                f'{fmt_corr(metric, "rownorm_random")} |\n')
    f.write('\n')

    # Interpret results
    f.write('## Findings\n\n')

    # Check if any correlations are significant
    best_metric = None
    best_r = 0
    for metric, name in metrics_to_test:
        for split_col in ['rownorm_fixed', 'rownorm_random']:
            if metric in corr_results and split_col in corr_results[metric]:
                v = corr_results[metric][split_col]
                if abs(v['pearson_r']) > abs(best_r):
                    best_r = v['pearson_r']
                    best_metric = (metric, name, split_col)

    if best_metric and abs(best_r) > 0.5:
        f.write(f'**Strongest correlation:** {best_metric[1]} vs '
                f'{best_metric[2].replace("rownorm_", "").title()} splits: '
                f'r={best_r:+.3f}\n\n')

    f.write('### 1. Correlation Summary\n\n')
    any_sig = any(
        corr_results.get(m, {}).get(s, {}).get('significant', 'ns') != 'ns'
        for m, _ in metrics_to_test
        for s in ['rownorm_fixed', 'rownorm_random']
    )
    if any_sig:
        f.write('**Significant correlations found** (p < 0.05).\n\n')
    else:
        f.write('**No statistically significant correlations found** at p < 0.05 with N=9.\n'
                'Note: With N=9, Pearson |r| > 0.666 is needed for p < 0.05.\n\n')

    f.write('### 2. Dataset Patterns\n\n')
    if len(df_corr) > 0:
        high_kappa = df_corr.nlargest(3, 'kappa')['dataset'].tolist()
        low_kappa = df_corr.nsmallest(3, 'kappa')['dataset'].tolist()
        f.write(f'**Well-separated eigenvalues (high κ):** {", ".join(high_kappa)}\n\n')
        f.write(f'**Clustered eigenvalues (low κ):** {", ".join(low_kappa)}\n\n')

    f.write('### 3. Hypothesis Validation\n\n')
    f.write('The geometric diversity hypothesis posits that well-separated eigenvalues '
            'lead to diverse eigenvector directions, making RowNorm normalization more effective.\n\n')

    all_corr_vals = [
        corr_results.get(m, {}).get(s, {}).get('pearson_r', 0)
        for m, _ in metrics_to_test
        for s in ['rownorm_fixed', 'rownorm_random']
        if corr_results.get(m, {}).get(s) is not None
    ]
    if all_corr_vals:
        avg_r = np.mean(np.abs(all_corr_vals))
        f.write(f'Average |Pearson r| across all tested metrics and splits: **{avg_r:.3f}**\n\n')
        if avg_r > 0.6:
            f.write('→ **Moderate to strong evidence** supporting the hypothesis.\n\n')
        elif avg_r > 0.3:
            f.write('→ **Weak evidence** for the hypothesis. Effect exists but not strongly '
                    'statistically significant given small sample size (N=9).\n\n')
        else:
            f.write('→ **Insufficient evidence** for the hypothesis at this sample size.\n\n')

    f.write('### 4. Key Findings\n\n')
    f.write(f'- **RowNorm recovery range:** {min(fixed_vals):.2f}pp to {max(fixed_vals):.2f}pp '
            f'(fixed), mean={np.mean(fixed_vals):.2f}pp\n')
    f.write(f'- **RowNorm recovery range:** {min(random_vals):.2f}pp to {max(random_vals):.2f}pp '
            f'(random), mean={np.mean(random_vals):.2f}pp\n')
    f.write(f'- Datasets with N=9 limits statistical power; significant effects require |r| > 0.666\n')
    f.write(f'- ogbn-arxiv shows extreme RowNorm recovery (+{rownorm_recovery["ogbn-arxiv"]["fixed"]:.1f}pp), '
            f'which may drive correlations\n\n')

    f.write('## Files Generated\n\n')
    f.write('- `eigenvalue_metrics_complete.csv` - All 9 datasets with metrics\n')
    f.write('- `correlation_results.txt` - Statistical analysis\n')
    f.write('- `plots/figure1_eigenvalue_spectra_all.png` - 3x3 eigenvalue spectra\n')
    f.write('- `plots/figure2_kappa_vs_recovery_fixed.png` - κ vs Recovery (fixed)\n')
    f.write('- `plots/figure2b_kappa_vs_recovery_random.png` - κ vs Recovery (random)\n')
    f.write('- `plots/figure3_entropy_vs_recovery_fixed.png` - H(λ) vs Recovery (fixed)\n')
    f.write('- `plots/figure3b_entropy_vs_recovery_random.png` - H(λ) vs Recovery (random)\n')
    f.write('- `plots/figure4_metrics_heatmap.png` - Comprehensive heatmap\n\n')

    f.write('---\n*Generated by complete_eigenvalue_analysis.py*\n')

print(f'\nReport saved: {report_path}')

print('\n' + '=' * 70)
print('COMPLETE. Summary of outputs:')
print('=' * 70)
print(f'  JSON:     {existing_json_path}')
print(f'  Summary:  {DATA_DIR}/summary_table.csv')
print(f'  Metrics:  {OUTPUT_DIR}/eigenvalue_metrics_complete.csv')
print(f'  Corr:     {corr_text_path}')
print(f'  Report:   {report_path}')
print(f'  Plots:    {PLOTS_DIR}/')
print()
print('Datasets in JSON:', sorted(all_data.keys()))
