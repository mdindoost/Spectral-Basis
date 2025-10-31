"""
Summarize Diffusion Experiments Across All Datasets
====================================================

Aggregates results from investigation2_diffused_engineered.py across all datasets
and generates comparison tables and plots.

Usage:
    python scripts/summarize_diffusion_experiments.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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
OUTPUT_DIR = 'results/investigation2_diffused_engineered/cross_dataset_summary'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load Results
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
# Generate Summary Table
# ============================================================================

print('='*80)
print('SUMMARY TABLE: BASELINE vs DIFFUSION')
print('='*80)

# Header
header = f"{'Dataset':<20} {'Baseline':<12} {'Std k=4':<12} {'Std Δ':<10} {'CMG k=8':<12} {'CMG Δ':<10}"
print(header)
print('-'*80)

summary_data = []

for dataset in sorted(all_results.keys()):
    data = all_results[dataset]
    
    # Baseline (c) RowNorm
    baseline_c = data['baseline']['c_restricted_rownorm']['mean'] * 100
    
    # Standard diffusion k=4 (if exists)
    std_k4_exists = '4' in data.get('standard_diffusion', {})
    if std_k4_exists:
        std_k4_c = data['standard_diffusion']['4']['c_restricted_rownorm']['mean'] * 100
        std_delta = std_k4_c - baseline_c
    else:
        std_k4_c = np.nan
        std_delta = np.nan
    
    # CMG diffusion k=8 (if exists)
    cmg_k8_exists = '8' in data.get('cmg_diffusion', {})
    if cmg_k8_exists:
        cmg_k8_c = data['cmg_diffusion']['8']['c_restricted_rownorm']['mean'] * 100
        cmg_delta = cmg_k8_c - baseline_c
    else:
        cmg_k8_c = np.nan
        cmg_delta = np.nan
    
    # Format row
    row = f"{dataset:<20} {baseline_c:>6.2f}%      "
    
    if not np.isnan(std_k4_c):
        row += f"{std_k4_c:>6.2f}%      {std_delta:>+6.2f}pp   "
    else:
        row += f"{'N/A':>12} {'N/A':>10}   "
    
    if not np.isnan(cmg_k8_c):
        row += f"{cmg_k8_c:>6.2f}%      {cmg_delta:>+6.2f}pp"
    else:
        row += f"{'N/A':>12} {'N/A':>10}"
    
    print(row)
    
    summary_data.append({
        'dataset': dataset,
        'baseline': baseline_c,
        'std_k4': std_k4_c,
        'std_delta': std_delta,
        'cmg_k8': cmg_k8_c,
        'cmg_delta': cmg_delta
    })

print('='*80)

# ============================================================================
# Generate Cross-Dataset Plots
# ============================================================================

print('\nGenerating cross-dataset plots...')

# Plot 1: Baseline vs Standard Diffusion (all k values)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, dataset in enumerate(sorted(all_results.keys())):
    if idx >= 8:
        break
    
    ax = axes[idx]
    data = all_results[dataset]
    
    baseline_c = data['baseline']['c_restricted_rownorm']['mean'] * 100
    
    # Baseline line
    ax.axhline(y=baseline_c, color='black', linestyle='--', 
               label='Baseline', linewidth=2, alpha=0.7)
    
    # Standard diffusion
    if 'standard_diffusion' in data and data['standard_diffusion']:
        k_vals = sorted([int(k) for k in data['standard_diffusion'].keys()])
        std_accs = [data['standard_diffusion'][str(k)]['c_restricted_rownorm']['mean'] * 100 
                    for k in k_vals]
        ax.plot(k_vals, std_accs, marker='o', label='Standard Diff', 
                linewidth=2, color='#1565C0')
    
    # CMG diffusion
    if 'cmg_diffusion' in data and data['cmg_diffusion']:
        k_vals = sorted([int(k) for k in data['cmg_diffusion'].keys()])
        cmg_accs = [data['cmg_diffusion'][str(k)]['c_restricted_rownorm']['mean'] * 100 
                    for k in k_vals]
        ax.plot(k_vals, cmg_accs, marker='s', label='CMG Diff',
                linewidth=2, color='#D32F2F')
    
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.set_xlabel('k', fontsize=10)
    ax.set_ylabel('Test Acc (%)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/all_datasets_diffusion_comparison.png', 
            dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/all_datasets_diffusion_comparison.png')
plt.close()

# Plot 2: Delta comparison (bar chart)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

datasets_clean = [d.replace('-', '\n') for d in sorted(all_results.keys())]
x_pos = np.arange(len(datasets_clean))
width = 0.35

std_deltas = [summary_data[i]['std_delta'] for i in range(len(summary_data))]
cmg_deltas = [summary_data[i]['cmg_delta'] for i in range(len(summary_data))]

# Handle NaN
std_deltas_clean = [d if not np.isnan(d) else 0 for d in std_deltas]
cmg_deltas_clean = [d if not np.isnan(d) else 0 for d in cmg_deltas]

ax.bar(x_pos - width/2, std_deltas_clean, width, 
       label='Standard k=4', color='#1565C0', alpha=0.8)
ax.bar(x_pos + width/2, cmg_deltas_clean, width,
       label='CMG k=8', color='#D32F2F', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_ylabel('Improvement over Baseline (pp)', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_title('Diffusion Effect Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets_clean, fontsize=9)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/diffusion_deltas_comparison.png',
            dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/diffusion_deltas_comparison.png')
plt.close()

# ============================================================================
# Save Aggregated Results
# ============================================================================

aggregated = {
    'summary_table': summary_data,
    'all_results': all_results
}

with open(f'{OUTPUT_DIR}/aggregated_results.json', 'w') as f:
    json.dump(aggregated, f, indent=2)

print(f'✓ Saved: {OUTPUT_DIR}/aggregated_results.json')

print('\n' + '='*80)
print('SUMMARY COMPLETE')
print('='*80)
print(f'Results saved to: {OUTPUT_DIR}/')
print('='*80)
