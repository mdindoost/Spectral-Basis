"""
Summarize Diffusion Experiments - Optimal k Selection
======================================================

Finds optimal k value for each dataset and diffusion type, then generates
comprehensive comparison tables and plots.

Usage:
    python scripts/summarize_diffusion_experiments_optimal.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    'ogbn-arxiv',
    'cora', 
    'citeseer',
    'pubmed',
    'wikics',
    'amazon-computers',
    'coauthor-cs',
    'coauthor-physics'
]

RESULTS_DIR = 'results/investigation2_diffused_engineered'
OUTPUT_DIR = 'results/investigation2_diffused_engineered/cross_dataset_summary_optimal'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_dataset_results(dataset_name):
    """Load complete_results.json for a dataset"""
    path = f'{RESULTS_DIR}/{dataset_name}/summary/complete_results.json'
    
    if not os.path.exists(path):
        print(f'⚠️  Missing: {dataset_name}')
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data

def find_optimal_k(diffusion_results, metric='c_restricted_rownorm'):
    """
    Find optimal k value for a diffusion type
    
    Args:
        diffusion_results: dict with k -> results mapping
        metric: which experiment to optimize (default: RowNorm on restricted)
    
    Returns:
        optimal_k: best k value
        optimal_acc: accuracy at optimal k
        all_k_values: list of (k, acc) tuples
    """
    if not diffusion_results:
        return None, None, []
    
    k_acc_pairs = []
    for k_str, results in diffusion_results.items():
        k = int(k_str)
        acc = results[metric]['mean'] * 100
        k_acc_pairs.append((k, acc))
    
    # Sort by k
    k_acc_pairs.sort(key=lambda x: x[0])
    
    # Find optimal
    optimal_k, optimal_acc = max(k_acc_pairs, key=lambda x: x[1])
    
    return optimal_k, optimal_acc, k_acc_pairs

def compute_improvement(baseline_acc, diffusion_acc):
    """Compute absolute and relative improvement"""
    abs_improvement = diffusion_acc - baseline_acc
    rel_improvement = (abs_improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    return abs_improvement, rel_improvement

# ============================================================================
# Load All Results
# ============================================================================

print('='*80)
print('LOADING RESULTS')
print('='*80)

all_results = {}
for dataset in DATASETS:
    results = load_dataset_results(dataset)
    if results:
        all_results[dataset] = results
        print(f'✓ Loaded: {dataset}')

print(f'\nFound results for {len(all_results)}/{len(DATASETS)} datasets\n')

# ============================================================================
# Analyze Optimal k Values
# ============================================================================

print('='*80)
print('FINDING OPTIMAL k VALUES FOR EACH DATASET')
print('='*80)

optimal_configs = {}

for dataset in sorted(all_results.keys()):
    data = all_results[dataset]
    
    baseline_c = data['baseline']['c_restricted_rownorm']['mean'] * 100
    
    # Find optimal k for standard diffusion
    std_optimal_k, std_optimal_acc, std_all = find_optimal_k(
        data.get('standard_diffusion', {}), 
        'c_restricted_rownorm'
    )
    
    # Find optimal k for CMG diffusion
    cmg_optimal_k, cmg_optimal_acc, cmg_all = find_optimal_k(
        data.get('cmg_diffusion', {}),
        'c_restricted_rownorm'
    )
    
    # Compute improvements
    if std_optimal_k is not None:
        std_abs_imp, std_rel_imp = compute_improvement(baseline_c, std_optimal_acc)
    else:
        std_abs_imp, std_rel_imp = None, None
    
    if cmg_optimal_k is not None:
        cmg_abs_imp, cmg_rel_imp = compute_improvement(baseline_c, cmg_optimal_acc)
    else:
        cmg_abs_imp, cmg_rel_imp = None, None
    
    optimal_configs[dataset] = {
        'baseline_c': baseline_c,
        'standard': {
            'optimal_k': std_optimal_k,
            'optimal_acc': std_optimal_acc,
            'abs_improvement': std_abs_imp,
            'rel_improvement': std_rel_imp,
            'all_k_values': std_all
        },
        'cmg': {
            'optimal_k': cmg_optimal_k,
            'optimal_acc': cmg_optimal_acc,
            'abs_improvement': cmg_abs_imp,
            'rel_improvement': cmg_rel_imp,
            'all_k_values': cmg_all
        }
    }
    
    print(f'\n{dataset}:')
    print(f'  Baseline RowNorm:     {baseline_c:6.2f}%')
    if std_optimal_k is not None:
        print(f'  Standard optimal:     k={std_optimal_k:2d} → {std_optimal_acc:6.2f}% ({std_abs_imp:+6.2f}pp, {std_rel_imp:+5.1f}%)')
    if cmg_optimal_k is not None:
        print(f'  CMG optimal:          k={cmg_optimal_k:2d} → {cmg_optimal_acc:6.2f}% ({cmg_abs_imp:+6.2f}pp, {cmg_rel_imp:+5.1f}%)')

# ============================================================================
# Generate Summary Tables
# ============================================================================

print('\n' + '='*80)
print('SUMMARY TABLE: OPTIMAL CONFIGURATIONS')
print('='*80)

# Table 1: Optimal k and improvements
header = f"{'Dataset':<20} {'Baseline':<10} {'Std k':<7} {'Std Acc':<10} {'Std Δ':<10} {'CMG k':<7} {'CMG Acc':<10} {'CMG Δ':<10}"
print(header)
print('-'*80)

summary_data = []

for dataset in sorted(optimal_configs.keys()):
    config = optimal_configs[dataset]
    baseline = config['baseline_c']
    
    row = f"{dataset:<20} {baseline:>6.2f}%   "
    
    # Standard diffusion
    if config['standard']['optimal_k'] is not None:
        std_k = config['standard']['optimal_k']
        std_acc = config['standard']['optimal_acc']
        std_delta = config['standard']['abs_improvement']
        row += f" k={std_k:<2d}   {std_acc:>6.2f}%   {std_delta:>+6.2f}pp  "
    else:
        row += f" {'N/A':<7} {'N/A':<10} {'N/A':<10}  "
    
    # CMG diffusion
    if config['cmg']['optimal_k'] is not None:
        cmg_k = config['cmg']['optimal_k']
        cmg_acc = config['cmg']['optimal_acc']
        cmg_delta = config['cmg']['abs_improvement']
        row += f" k={cmg_k:<2d}   {cmg_acc:>6.2f}%   {cmg_delta:>+6.2f}pp"
    else:
        row += f" {'N/A':<7} {'N/A':<10} {'N/A':<10}"
    
    print(row)
    
    summary_data.append({
        'dataset': dataset,
        'baseline': baseline,
        'std_optimal_k': config['standard']['optimal_k'],
        'std_optimal_acc': config['standard']['optimal_acc'],
        'std_delta': config['standard']['abs_improvement'],
        'cmg_optimal_k': config['cmg']['optimal_k'],
        'cmg_optimal_acc': config['cmg']['optimal_acc'],
        'cmg_delta': config['cmg']['abs_improvement']
    })

print('='*80)

# Table 2: Ranking by improvement magnitude
print('\n' + '='*80)
print('RANKING BY IMPROVEMENT MAGNITUDE (Standard Diffusion)')
print('='*80)

std_ranking = [(d, config['standard']['abs_improvement'], config['standard']['optimal_k']) 
               for d, config in optimal_configs.items() 
               if config['standard']['abs_improvement'] is not None]
std_ranking.sort(key=lambda x: x[1], reverse=True)

print(f"{'Rank':<6} {'Dataset':<20} {'Improvement':<15} {'Optimal k':<10}")
print('-'*80)
for rank, (dataset, improvement, k) in enumerate(std_ranking, 1):
    print(f"{rank:<6} {dataset:<20} {improvement:>+6.2f}pp ({improvement:>+5.1f}%)   k={k}")

print('\n' + '='*80)
print('RANKING BY IMPROVEMENT MAGNITUDE (CMG Diffusion)')
print('='*80)

cmg_ranking = [(d, config['cmg']['abs_improvement'], config['cmg']['optimal_k']) 
               for d, config in optimal_configs.items() 
               if config['cmg']['abs_improvement'] is not None]
cmg_ranking.sort(key=lambda x: x[1], reverse=True)

print(f"{'Rank':<6} {'Dataset':<20} {'Improvement':<15} {'Optimal k':<10}")
print('-'*80)
for rank, (dataset, improvement, k) in enumerate(cmg_ranking, 1):
    print(f"{rank:<6} {dataset:<20} {improvement:>+6.2f}pp ({improvement:>+5.1f}%)   k={k}")

# Table 3: k-value patterns
print('\n' + '='*80)
print('OPTIMAL k PATTERNS')
print('='*80)

print('\nStandard Diffusion:')
std_k_values = [config['standard']['optimal_k'] for config in optimal_configs.values() 
                if config['standard']['optimal_k'] is not None]
if std_k_values:
    print(f"  Mean optimal k: {np.mean(std_k_values):.1f}")
    print(f"  Median optimal k: {np.median(std_k_values):.0f}")
    print(f"  Range: [{min(std_k_values)}, {max(std_k_values)}]")

print('\nCMG Diffusion:')
cmg_k_values = [config['cmg']['optimal_k'] for config in optimal_configs.values() 
                if config['cmg']['optimal_k'] is not None]
if cmg_k_values:
    print(f"  Mean optimal k: {np.mean(cmg_k_values):.1f}")
    print(f"  Median optimal k: {np.median(cmg_k_values):.0f}")
    print(f"  Range: [{min(cmg_k_values)}, {max(cmg_k_values)}]")

# ============================================================================
# Generate Plots
# ============================================================================

print('\n' + '='*80)
print('GENERATING PLOTS')
print('='*80)

# Plot 1: Per-dataset k-value trajectories (2x4 grid)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, dataset in enumerate(sorted(optimal_configs.keys())):
    if idx >= 8:
        break
    
    ax = axes[idx]
    config = optimal_configs[dataset]
    baseline = config['baseline_c']
    
    # Baseline line
    ax.axhline(y=baseline, color='black', linestyle='--', 
               label='Baseline', linewidth=2, alpha=0.7)
    
    # Standard diffusion
    if config['standard']['all_k_values']:
        std_ks = [k for k, _ in config['standard']['all_k_values']]
        std_accs = [acc for _, acc in config['standard']['all_k_values']]
        ax.plot(std_ks, std_accs, marker='o', label='Standard Diff', 
                linewidth=2, color='#1565C0', markersize=6)
        
        # Mark optimal
        opt_k = config['standard']['optimal_k']
        opt_acc = config['standard']['optimal_acc']
        ax.scatter([opt_k], [opt_acc], s=150, color='#1565C0', 
                   edgecolors='red', linewidths=2, zorder=5)
    
    # CMG diffusion
    if config['cmg']['all_k_values']:
        cmg_ks = [k for k, _ in config['cmg']['all_k_values']]
        cmg_accs = [acc for _, acc in config['cmg']['all_k_values']]
        ax.plot(cmg_ks, cmg_accs, marker='s', label='CMG Diff',
                linewidth=2, color='#D32F2F', markersize=6)
        
        # Mark optimal
        opt_k = config['cmg']['optimal_k']
        opt_acc = config['cmg']['optimal_acc']
        ax.scatter([opt_k], [opt_acc], s=150, color='#D32F2F',
                   edgecolors='red', linewidths=2, zorder=5)
    
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.set_xlabel('k', fontsize=10)
    ax.set_ylabel('Test Acc (%)', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/all_datasets_k_trajectories.png', 
            dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/plots/all_datasets_k_trajectories.png')
plt.close()

# Plot 2: Optimal improvements bar chart
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

datasets_sorted = sorted(optimal_configs.keys())
datasets_labels = [d.replace('-', '\n') for d in datasets_sorted]
x_pos = np.arange(len(datasets_sorted))
width = 0.35

std_deltas = [optimal_configs[d]['standard']['abs_improvement'] 
              if optimal_configs[d]['standard']['abs_improvement'] is not None else 0 
              for d in datasets_sorted]
cmg_deltas = [optimal_configs[d]['cmg']['abs_improvement']
              if optimal_configs[d]['cmg']['abs_improvement'] is not None else 0
              for d in datasets_sorted]

bars1 = ax.bar(x_pos - width/2, std_deltas, width, 
               label='Standard (optimal k)', color='#1565C0', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, cmg_deltas, width,
               label='CMG (optimal k)', color='#D32F2F', alpha=0.8)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.5:  # Only label significant values
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_ylabel('Improvement over Baseline (pp)', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('Optimal Diffusion Effect Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets_labels, fontsize=9)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/optimal_improvements_comparison.png',
            dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/plots/optimal_improvements_comparison.png')
plt.close()

# Plot 3: Heatmap of all k values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

datasets_sorted = sorted(optimal_configs.keys())

# Standard diffusion heatmap
std_data = []
all_std_ks = sorted(set([k for d in datasets_sorted 
                         for k, _ in optimal_configs[d]['standard']['all_k_values']]))

for dataset in datasets_sorted:
    row = []
    k_acc_dict = {k: acc for k, acc in optimal_configs[dataset]['standard']['all_k_values']}
    baseline = optimal_configs[dataset]['baseline_c']
    for k in all_std_ks:
        if k in k_acc_dict:
            row.append(k_acc_dict[k] - baseline)  # Improvement
        else:
            row.append(np.nan)
    std_data.append(row)

std_data = np.array(std_data)
im1 = ax1.imshow(std_data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=45)
ax1.set_xticks(range(len(all_std_ks)))
ax1.set_xticklabels([f'k={k}' for k in all_std_ks])
ax1.set_yticks(range(len(datasets_sorted)))
ax1.set_yticklabels(datasets_sorted)
ax1.set_title('Standard Diffusion: Improvement (pp)', fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Improvement (pp)')

# CMG diffusion heatmap
cmg_data = []
all_cmg_ks = sorted(set([k for d in datasets_sorted 
                         for k, _ in optimal_configs[d]['cmg']['all_k_values']]))

for dataset in datasets_sorted:
    row = []
    k_acc_dict = {k: acc for k, acc in optimal_configs[dataset]['cmg']['all_k_values']}
    baseline = optimal_configs[dataset]['baseline_c']
    for k in all_cmg_ks:
        if k in k_acc_dict:
            row.append(k_acc_dict[k] - baseline)
        else:
            row.append(np.nan)
    cmg_data.append(row)

cmg_data = np.array(cmg_data)
im2 = ax2.imshow(cmg_data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=45)
ax2.set_xticks(range(len(all_cmg_ks)))
ax2.set_xticklabels([f'k={k}' for k in all_cmg_ks])
ax2.set_yticks(range(len(datasets_sorted)))
ax2.set_yticklabels(datasets_sorted)
ax2.set_title('CMG Diffusion: Improvement (pp)', fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Improvement (pp)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/diffusion_heatmap.png',
            dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/plots/diffusion_heatmap.png')
plt.close()

# ============================================================================
# Save Aggregated Results
# ============================================================================

print('\n' + '='*80)
print('SAVING RESULTS')
print('='*80)

output_data = {
    'optimal_configurations': optimal_configs,
    'summary_table': summary_data,
    'rankings': {
        'standard': [{'dataset': d, 'improvement': imp, 'optimal_k': k} 
                     for d, imp, k in std_ranking],
        'cmg': [{'dataset': d, 'improvement': imp, 'optimal_k': k}
                for d, imp, k in cmg_ranking]
    }
}

with open(f'{OUTPUT_DIR}/optimal_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f'✓ Saved: {OUTPUT_DIR}/optimal_results.json')

# ============================================================================
# Generate Text Report
# ============================================================================

with open(f'{OUTPUT_DIR}/summary_report.txt', 'w') as f:
    f.write('='*80 + '\n')
    f.write('DIFFUSION EXPERIMENT SUMMARY - OPTIMAL CONFIGURATIONS\n')
    f.write('='*80 + '\n\n')
    
    f.write('KEY FINDINGS:\n')
    f.write('-'*80 + '\n\n')
    
    # Top improvements
    f.write('TOP 3 IMPROVEMENTS (Standard Diffusion):\n')
    for rank, (dataset, improvement, k) in enumerate(std_ranking[:3], 1):
        f.write(f'  {rank}. {dataset}: +{improvement:.2f}pp at k={k}\n')
    
    f.write('\nTOP 3 IMPROVEMENTS (CMG Diffusion):\n')
    for rank, (dataset, improvement, k) in enumerate(cmg_ranking[:3], 1):
        f.write(f'  {rank}. {dataset}: +{improvement:.2f}pp at k={k}\n')
    
    # Failures
    f.write('\nDATASETS WHERE DIFFUSION HURTS:\n')
    for dataset, improvement, k in std_ranking:
        if improvement < 0:
            f.write(f'  - {dataset}: {improvement:.2f}pp at k={k}\n')
    
    f.write('\n' + '='*80 + '\n')
    f.write('COMPLETE RESULTS TABLE\n')
    f.write('='*80 + '\n\n')
    f.write(header + '\n')
    f.write('-'*80 + '\n')
    
    for dataset in sorted(optimal_configs.keys()):
        config = optimal_configs[dataset]
        baseline = config['baseline_c']
        
        line = f"{dataset:<20} {baseline:>6.2f}%   "
        
        if config['standard']['optimal_k'] is not None:
            std_k = config['standard']['optimal_k']
            std_acc = config['standard']['optimal_acc']
            std_delta = config['standard']['abs_improvement']
            line += f" k={std_k:<2d}   {std_acc:>6.2f}%   {std_delta:>+6.2f}pp  "
        else:
            line += f" {'N/A':<7} {'N/A':<10} {'N/A':<10}  "
        
        if config['cmg']['optimal_k'] is not None:
            cmg_k = config['cmg']['optimal_k']
            cmg_acc = config['cmg']['optimal_acc']
            cmg_delta = config['cmg']['abs_improvement']
            line += f" k={cmg_k:<2d}   {cmg_acc:>6.2f}%   {cmg_delta:>+6.2f}pp"
        else:
            line += f" {'N/A':<7} {'N/A':<10} {'N/A':<10}"
        
        f.write(line + '\n')

print(f'✓ Saved: {OUTPUT_DIR}/summary_report.txt')

print('\n' + '='*80)
print('SUMMARY COMPLETE')
print('='*80)
print(f'Results saved to: {OUTPUT_DIR}/')
print(f'  - optimal_results.json (full data)')
print(f'  - summary_report.txt (text summary)')
print(f'  - plots/all_datasets_k_trajectories.png')
print(f'  - plots/optimal_improvements_comparison.png')
print(f'  - plots/diffusion_heatmap.png')
print('='*80)

