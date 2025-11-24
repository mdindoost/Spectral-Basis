"""
Analysis Script: Aggregate and Summarize Anti-Smoothing Results
================================================================

Creates publication-ready tables and figures from all experiments.

Usage:
    python experiments/analyze_antiSmoothing_results.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

RESULTS_BASE = 'results/investigation_sgc_antiSmoothing'
OUTPUT_DIR = 'results/investigation_sgc_antiSmoothing/summary'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets to analyze
DATASETS = [
    ('ogbn-arxiv', 'fixed', 'lcc'),
    ('wikics', 'fixed', 'lcc'),
    ('amazon-computers', 'fixed', 'lcc'),
    ('pubmed', 'random', 'lcc'),
    ('cora', 'random', 'lcc'),
    ('citeseer', 'random', 'lcc'),
]

K_VALUES = [2, 4, 6, 8, 10]

# ============================================================================
# Helper Functions
# ============================================================================

def load_results(dataset, splits, component, k):
    """Load results JSON for a specific configuration"""
    path = f'{RESULTS_BASE}/{dataset}_{splits}_{component}/k{k}/metrics/results.json'
    
    if not os.path.exists(path):
        print(f'Warning: Results not found for {dataset} (k={k})')
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

def format_accuracy(mean, std):
    """Format accuracy with standard deviation"""
    return f'{mean*100:.2f} ± {std*100:.2f}'

def format_improvement(improvement):
    """Format improvement with color coding"""
    return f'{improvement:+.2f}'

# ============================================================================
# Load All Results
# ============================================================================

print('='*80)
print('LOADING RESULTS')
print('='*80)

all_data = {}

for dataset, splits, component in DATASETS:
    print(f'\nLoading {dataset} ({splits} splits)...')
    
    all_data[dataset] = {}
    
    for k in K_VALUES:
        results = load_results(dataset, splits, component, k)
        if results:
            all_data[dataset][k] = results
            print(f'  k={k}: ✓')
        else:
            print(f'  k={k}: ✗')

# ============================================================================
# Table 1: Best Results Per Dataset
# ============================================================================

print('\n' + '='*80)
print('TABLE 1: BEST RESULTS PER DATASET')
print('='*80)

table1_data = []

for dataset, splits, component in DATASETS:
    if dataset not in all_data or len(all_data[dataset]) == 0:
        continue
    
    # Find best k for each method
    best_sgc = {'k': None, 'acc': 0}
    best_sgc_mlp = {'k': None, 'acc': 0}
    best_full = {'k': None, 'acc': 0}
    best_jl_binary = {'k': None, 'acc': 0}
    best_jl_ortho = {'k': None, 'acc': 0}
    
    for k in K_VALUES:
        if k not in all_data[dataset]:
            continue
        
        results = all_data[dataset][k]['results']
        
        # SGC
        if 'sgc_baseline' in results:
            acc = results['sgc_baseline']['test_acc_mean']
            if acc > best_sgc['acc']:
                best_sgc = {'k': k, 'acc': acc}
        
        # SGC + MLP
        if 'sgc_mlp_baseline' in results:
            acc = results['sgc_mlp_baseline']['test_acc_mean']
            if acc > best_sgc_mlp['acc']:
                best_sgc_mlp = {'k': k, 'acc': acc}
        
        # Full
        if 'full_rownorm_mlp' in results:
            acc = results['full_rownorm_mlp']['test_acc_mean']
            if acc > best_full['acc']:
                best_full = {'k': k, 'acc': acc}
        
        # JL Binary 2nc
        if 'jl_binary_2nc' in results:
            acc = results['jl_binary_2nc']['test_acc_mean']
            if acc > best_jl_binary['acc']:
                best_jl_binary = {'k': k, 'acc': acc}
        
        # JL Ortho 2nc
        if 'jl_ortho_2nc' in results:
            acc = results['jl_ortho_2nc']['test_acc_mean']
            if acc > best_jl_ortho['acc']:
                best_jl_ortho = {'k': k, 'acc': acc}
    
    # Compute improvements
    sgc_acc = best_sgc['acc']
    full_improvement = (best_full['acc'] - sgc_acc) * 100
    jl_binary_improvement = (best_jl_binary['acc'] - sgc_acc) * 100
    jl_ortho_improvement = (best_jl_ortho['acc'] - sgc_acc) * 100
    
    table1_data.append({
        'Dataset': dataset,
        'SGC': f"{sgc_acc*100:.2f} (k={best_sgc['k']})",
        'SGC+MLP': f"{best_sgc_mlp['acc']*100:.2f} (k={best_sgc_mlp['k']})",
        'Full+RowNorm': f"{best_full['acc']*100:.2f} (k={best_full['k']})",
        'JL Binary (2nc)': f"{best_jl_binary['acc']*100:.2f} (k={best_jl_binary['k']})",
        'JL Ortho (2nc)': f"{best_jl_ortho['acc']*100:.2f} (k={best_jl_ortho['k']})",
        'Improvement (Full)': format_improvement(full_improvement),
        'Improvement (JL Binary)': format_improvement(jl_binary_improvement),
    })

df_table1 = pd.DataFrame(table1_data)
print('\n' + df_table1.to_string(index=False))

# Save
df_table1.to_csv(f'{OUTPUT_DIR}/table1_best_results.csv', index=False)
df_table1.to_latex(f'{OUTPUT_DIR}/table1_best_results.tex', index=False)
print(f'\n✓ Saved: {OUTPUT_DIR}/table1_best_results.csv')

# ============================================================================
# Table 2: Anti-Smoothing Effect (k=2 vs k=10)
# ============================================================================

print('\n' + '='*80)
print('TABLE 2: ANTI-SMOOTHING EFFECT (k=2 vs k=10)')
print('='*80)

table2_data = []

for dataset, splits, component in DATASETS:
    if dataset not in all_data:
        continue
    
    if 2 not in all_data[dataset] or 10 not in all_data[dataset]:
        continue
    
    results_k2 = all_data[dataset][2]['results']
    results_k10 = all_data[dataset][10]['results']
    
    # SGC trajectory
    sgc_k2 = results_k2['sgc_baseline']['test_acc_mean'] * 100
    sgc_k10 = results_k10['sgc_baseline']['test_acc_mean'] * 100
    sgc_change = sgc_k10 - sgc_k2
    
    # Full trajectory
    full_k2 = results_k2['full_rownorm_mlp']['test_acc_mean'] * 100
    full_k10 = results_k10['full_rownorm_mlp']['test_acc_mean'] * 100
    full_change = full_k10 - full_k2
    
    # Anti-smoothing indicator
    anti_smoothing = '✓' if (sgc_change < 0 and full_change > 0) else '✗'
    
    table2_data.append({
        'Dataset': dataset,
        'SGC @ k=2': f'{sgc_k2:.2f}',
        'SGC @ k=10': f'{sgc_k10:.2f}',
        'SGC Change': format_improvement(sgc_change),
        'Full @ k=2': f'{full_k2:.2f}',
        'Full @ k=10': f'{full_k10:.2f}',
        'Full Change': format_improvement(full_change),
        'Anti-Smoothing': anti_smoothing
    })

df_table2 = pd.DataFrame(table2_data)
print('\n' + df_table2.to_string(index=False))

# Save
df_table2.to_csv(f'{OUTPUT_DIR}/table2_anti_smoothing.csv', index=False)
df_table2.to_latex(f'{OUTPUT_DIR}/table2_anti_smoothing.tex', index=False)
print(f'\n✓ Saved: {OUTPUT_DIR}/table2_anti_smoothing.csv')

# ============================================================================
# Table 3: Parameter Efficiency (k=10)
# ============================================================================

print('\n' + '='*80)
print('TABLE 3: PARAMETER EFFICIENCY (k=10)')
print('='*80)

table3_data = []

for dataset, splits, component in DATASETS:
    if dataset not in all_data or 10 not in all_data[dataset]:
        continue
    
    data = all_data[dataset][10]
    results = data['results']
    param_counts = data['parameter_counts']
    
    methods = [
        ('SGC+MLP', 'sgc_mlp_baseline'),
        ('Full+RowNorm', 'full_rownorm_mlp'),
        ('JL Binary (2nc)', 'jl_binary_2nc'),
        ('JL Ortho (2nc)', 'jl_ortho_2nc'),
    ]
    
    for method_name, method_key in methods:
        if method_key in results and method_key in param_counts:
            acc = results[method_key]['test_acc_mean'] * 100
            params = param_counts[method_key]
            
            table3_data.append({
                'Dataset': dataset,
                'Method': method_name,
                'Accuracy (%)': f'{acc:.2f}',
                'Parameters': f'{params:,}',
                'Acc/Param Ratio': f'{acc/params*1000:.4f}'
            })

df_table3 = pd.DataFrame(table3_data)
print('\n' + df_table3.to_string(index=False))

# Save
df_table3.to_csv(f'{OUTPUT_DIR}/table3_parameter_efficiency.csv', index=False)
df_table3.to_latex(f'{OUTPUT_DIR}/table3_parameter_efficiency.tex', index=False)
print(f'\n✓ Saved: {OUTPUT_DIR}/table3_parameter_efficiency.csv')

# ============================================================================
# Figure 1: K-Value Trajectories (All Datasets)
# ============================================================================

print('\n' + '='*80)
print('GENERATING FIGURE 1: K-VALUE TRAJECTORIES')
print('='*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (dataset, splits, component) in enumerate(DATASETS):
    ax = axes[idx]
    
    if dataset not in all_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(dataset)
        continue
    
    k_vals = sorted(all_data[dataset].keys())
    
    methods = {
        'sgc_baseline': ('SGC', '#1f77b4', 'o'),
        'sgc_mlp_baseline': ('SGC+MLP', '#ff7f0e', 's'),
        'full_rownorm_mlp': ('Full+RowNorm', '#2ca02c', '^'),
        'jl_binary_2nc': ('JL Binary (2nc)', '#d62728', 'v'),
        'jl_ortho_2nc': ('JL Ortho (2nc)', '#9467bd', 'D'),
    }
    
    for method_key, (label, color, marker) in methods.items():
        accs = []
        for k in k_vals:
            if method_key in all_data[dataset][k]['results']:
                accs.append(all_data[dataset][k]['results'][method_key]['test_acc_mean'] * 100)
            else:
                accs.append(None)
        
        if any(a is not None for a in accs):
            ax.plot(k_vals, accs, marker=marker, label=label, color=color, 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Diffusion Steps (k)', fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=10)
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure1_k_trajectories_all.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/figure1_k_trajectories_all.png')
plt.close()

# ============================================================================
# Figure 2: Improvement Heatmap
# ============================================================================

print('\n' + '='*80)
print('GENERATING FIGURE 2: IMPROVEMENT HEATMAP')
print('='*80)

# Prepare data for heatmap
heatmap_data = []
datasets_list = []

for dataset, splits, component in DATASETS:
    if dataset not in all_data:
        continue
    
    datasets_list.append(dataset)
    row = []
    
    for k in K_VALUES:
        if k not in all_data[dataset]:
            row.append(np.nan)
            continue
        
        results = all_data[dataset][k]['results']
        
        if 'full_rownorm_mlp' in results and 'sgc_baseline' in results:
            improvement = (results['full_rownorm_mlp']['test_acc_mean'] - 
                          results['sgc_baseline']['test_acc_mean']) * 100
            row.append(improvement)
        else:
            row.append(np.nan)
    
    heatmap_data.append(row)

# Create heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

heatmap_array = np.array(heatmap_data)
im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)

# Set ticks
ax.set_xticks(np.arange(len(K_VALUES)))
ax.set_yticks(np.arange(len(datasets_list)))
ax.set_xticklabels(K_VALUES)
ax.set_yticklabels(datasets_list)

# Add values
for i in range(len(datasets_list)):
    for j in range(len(K_VALUES)):
        if not np.isnan(heatmap_array[i, j]):
            text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

ax.set_xlabel('Diffusion Steps (k)', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)
ax.set_title('Improvement over SGC Baseline (pp)', fontsize=14, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Improvement (pp)', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure2_improvement_heatmap.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/figure2_improvement_heatmap.png')
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================

print('\n' + '='*80)
print('SUMMARY STATISTICS')
print('='*80)

total_experiments = len([d for d in all_data.values() if d])
total_k_values = sum(len(d) for d in all_data.values())

print(f'\nDatasets analyzed: {total_experiments}')
print(f'Total k-values: {total_k_values}')

# Count successes
successes = 0
failures = 0

for dataset in all_data:
    if 10 in all_data[dataset]:
        results = all_data[dataset][10]['results']
        if 'full_rownorm_mlp' in results and 'sgc_baseline' in results:
            improvement = (results['full_rownorm_mlp']['test_acc_mean'] - 
                          results['sgc_baseline']['test_acc_mean']) * 100
            if improvement > 0:
                successes += 1
            else:
                failures += 1

print(f'\nSuccess rate (improvement > 0 at k=10): {successes}/{successes+failures} ({successes/(successes+failures)*100:.1f}%)')

# Anti-smoothing count
anti_smoothing_count = df_table2[df_table2['Anti-Smoothing'] == '✓'].shape[0]
total_count = df_table2.shape[0]

print(f'Anti-smoothing effect observed: {anti_smoothing_count}/{total_count} datasets')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print(f'\nAll outputs saved to: {OUTPUT_DIR}/')
print('\nGenerated files:')
print('  - table1_best_results.csv / .tex')
print('  - table2_anti_smoothing.csv / .tex')
print('  - table3_parameter_efficiency.csv / .tex')
print('  - figure1_k_trajectories_all.png')
print('  - figure2_improvement_heatmap.png')
print('\n' + '='*80)
