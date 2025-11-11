"""
Analyze SGC Comparison Results Across All Datasets and K Values
================================================================

This script:
1. Reads all JSON result files from results/investigation_sgc_comparison/
2. Aggregates results across datasets and K values
3. Generates comprehensive tables and visualizations
4. Creates a final summary report

Usage:
    python scripts/analyze_sgc_results.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_DIR = 'results/investigation_sgc_comparison'
OUTPUT_DIR = 'results/investigation_sgc_comparison/summary'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

print('='*70)
print('ANALYZING SGC COMPARISON RESULTS')
print('='*70)

# ============================================================================
# 1. Load All Results
# ============================================================================
print('\n[1/5] Loading all result files...')

all_results = []

# Find all JSON result files
result_files = glob.glob(f'{RESULTS_DIR}/*/k*/metrics/results.json')
print(f'Found {len(result_files)} result files')

for filepath in result_files:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    except Exception as e:
        print(f'  Warning: Could not load {filepath}: {e}')

if len(all_results) == 0:
    print('ERROR: No result files found!')
    print(f'Looking in: {RESULTS_DIR}')
    exit(1)

print(f'Successfully loaded {len(all_results)} results')

# ============================================================================
# 2. Organize Results by Dataset and K
# ============================================================================
print('\n[2/5] Organizing results...')

# Create nested dictionary: dataset -> k -> results
organized = {}
for result in all_results:
    dataset = result['dataset']
    k = result['k']
    
    if dataset not in organized:
        organized[dataset] = {}
    
    organized[dataset][k] = result

datasets = sorted(organized.keys())
print(f'Datasets: {", ".join(datasets)}')

# ============================================================================
# 3. Create Summary Tables
# ============================================================================
print('\n[3/5] Creating summary tables...')

# Table 1: Best K per dataset
print('\nTable 1: Optimal K Values per Dataset')
print('='*70)

best_k_data = []
for dataset in datasets:
    k_values = sorted(organized[dataset].keys())
    
    # Find best K for SGC
    sgc_accs = [(k, organized[dataset][k]['sgc_baseline']['test_acc_mean']) for k in k_values]
    best_sgc_k, best_sgc_acc = max(sgc_accs, key=lambda x: x[1])
    
    # Find best K for RowNorm
    rownorm_accs = [(k, organized[dataset][k]['rownorm_variant']['test_acc_mean']) for k in k_values]
    best_rownorm_k, best_rownorm_acc = max(rownorm_accs, key=lambda x: x[1])
    
    # Best improvement
    improvements = [(k, organized[dataset][k]['improvement']['absolute_pp']) for k in k_values]
    best_imp_k, best_imp = max(improvements, key=lambda x: x[1])
    
    best_k_data.append({
        'Dataset': dataset,
        'Best K (SGC)': best_sgc_k,
        'SGC Acc': f"{best_sgc_acc*100:.2f}%",
        'Best K (RowNorm)': best_rownorm_k,
        'RowNorm Acc': f"{best_rownorm_acc*100:.2f}%",
        'Best Improvement': f"{best_imp:+.2f}pp",
        'K at Best Imp': best_imp_k
    })

df_best_k = pd.DataFrame(best_k_data)
print(df_best_k.to_string(index=False))

# Save to CSV
df_best_k.to_csv(f'{OUTPUT_DIR}/optimal_k_per_dataset.csv', index=False)
print(f'\n✓ Saved: {OUTPUT_DIR}/optimal_k_per_dataset.csv')

# Table 2: Complete results matrix
print('\n\nTable 2: Complete Results Matrix')
print('='*70)

complete_data = []
for dataset in datasets:
    k_values = sorted(organized[dataset].keys())
    for k in k_values:
        result = organized[dataset][k]
        complete_data.append({
            'Dataset': dataset,
            'K': k,
            'SGC': f"{result['sgc_baseline']['test_acc_mean']*100:.2f}±{result['sgc_baseline']['test_acc_std']*100:.2f}",
            'RowNorm': f"{result['rownorm_variant']['test_acc_mean']*100:.2f}±{result['rownorm_variant']['test_acc_std']*100:.2f}",
            'Δ (pp)': f"{result['improvement']['absolute_pp']:+.2f}",
            'Δ (%)': f"{result['improvement']['relative_percent']:+.2f}"
        })

df_complete = pd.DataFrame(complete_data)
print(df_complete.to_string(index=False))

# Save to CSV
df_complete.to_csv(f'{OUTPUT_DIR}/complete_results.csv', index=False)
print(f'\n✓ Saved: {OUTPUT_DIR}/complete_results.csv')

# Table 3: Summary statistics per dataset (averaged across K)
print('\n\nTable 3: Average Performance per Dataset (across all K)')
print('='*70)

summary_data = []
for dataset in datasets:
    k_values = sorted(organized[dataset].keys())
    
    sgc_accs = [organized[dataset][k]['sgc_baseline']['test_acc_mean']*100 for k in k_values]
    rownorm_accs = [organized[dataset][k]['rownorm_variant']['test_acc_mean']*100 for k in k_values]
    improvements = [organized[dataset][k]['improvement']['absolute_pp'] for k in k_values]
    
    summary_data.append({
        'Dataset': dataset,
        'K Values': f"{min(k_values)}-{max(k_values)}",
        'Avg SGC': f"{np.mean(sgc_accs):.2f}%",
        'Avg RowNorm': f"{np.mean(rownorm_accs):.2f}%",
        'Avg Δ': f"{np.mean(improvements):+.2f}pp",
        'Best Δ': f"{max(improvements):+.2f}pp",
        'Worst Δ': f"{min(improvements):+.2f}pp"
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save to CSV
df_summary.to_csv(f'{OUTPUT_DIR}/summary_per_dataset.csv', index=False)
print(f'\n✓ Saved: {OUTPUT_DIR}/summary_per_dataset.csv')

# ============================================================================
# 4. Generate Visualizations
# ============================================================================
print('\n[4/5] Generating visualizations...')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Plot 1: Heatmap of improvements
print('  Creating heatmap of improvements...')
fig, ax = plt.subplots(figsize=(12, len(datasets) * 0.8))

# Create matrix for heatmap
all_k_values = sorted(set([k for dataset in organized.values() for k in dataset.keys()]))
heatmap_data = []
for dataset in datasets:
    row = []
    for k in all_k_values:
        if k in organized[dataset]:
            row.append(organized[dataset][k]['improvement']['absolute_pp'])
        else:
            row.append(np.nan)
    heatmap_data.append(row)

# Create heatmap
sns.heatmap(heatmap_data, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            xticklabels=[f'K={k}' for k in all_k_values],
            yticklabels=datasets,
            cbar_kws={'label': 'Improvement (pp)'},
            ax=ax,
            vmin=-30, vmax=5)

ax.set_title('SGC Comparison: Improvement Heatmap\n(RowNorm Variant - SGC Baseline)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Propagation Steps (K)', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/improvement_heatmap.png', dpi=150, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR}/plots/improvement_heatmap.png')
plt.close()

# Plot 2: K-value trajectories for each dataset
print('  Creating K-value trajectories...')
n_datasets = len(datasets)
n_cols = 3
n_rows = (n_datasets + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() if n_datasets > 1 else [axes]

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    k_values = sorted(organized[dataset].keys())
    
    sgc_means = [organized[dataset][k]['sgc_baseline']['test_acc_mean']*100 for k in k_values]
    sgc_stds = [organized[dataset][k]['sgc_baseline']['test_acc_std']*100 for k in k_values]
    rownorm_means = [organized[dataset][k]['rownorm_variant']['test_acc_mean']*100 for k in k_values]
    rownorm_stds = [organized[dataset][k]['rownorm_variant']['test_acc_std']*100 for k in k_values]
    
    ax.errorbar(k_values, sgc_means, yerr=sgc_stds, marker='o', linewidth=2,
                capsize=5, label='SGC Baseline', color='#1f77b4')
    ax.errorbar(k_values, rownorm_means, yerr=rownorm_stds, marker='s', linewidth=2,
                capsize=5, label='RowNorm Variant', color='#ff7f0e')
    
    ax.set_xlabel('Propagation Steps (K)', fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=10)
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)

# Hide empty subplots
for idx in range(n_datasets, len(axes)):
    axes[idx].axis('off')

plt.suptitle('SGC Comparison: Accuracy vs. K Across Datasets', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/k_trajectories_all.png', dpi=150, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR}/plots/k_trajectories_all.png')
plt.close()

# Plot 3: Best performance comparison (bar chart)
print('  Creating best performance comparison...')
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.35

sgc_best = [max([organized[dataset][k]['sgc_baseline']['test_acc_mean']*100 
                 for k in organized[dataset].keys()]) for dataset in datasets]
rownorm_best = [max([organized[dataset][k]['rownorm_variant']['test_acc_mean']*100 
                     for k in organized[dataset].keys()]) for dataset in datasets]

bars1 = ax.bar(x - width/2, sgc_best, width, label='SGC Baseline (Best K)', 
               color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, rownorm_best, width, label='RowNorm Variant (Best K)', 
               color='#ff7f0e', alpha=0.8)

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('SGC Comparison: Best Performance per Dataset', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/best_performance_comparison.png', dpi=150, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR}/plots/best_performance_comparison.png')
plt.close()

# Plot 4: Improvement distribution
print('  Creating improvement distribution...')
fig, ax = plt.subplots(figsize=(10, 6))

improvements_by_dataset = {}
for dataset in datasets:
    improvements = [organized[dataset][k]['improvement']['absolute_pp'] 
                   for k in sorted(organized[dataset].keys())]
    improvements_by_dataset[dataset] = improvements

# Box plot
positions = range(1, len(datasets) + 1)
bp = ax.boxplot(improvements_by_dataset.values(), 
                labels=datasets,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Improvement (pp)', fontsize=12)
ax.set_title('Distribution of Improvements Across K Values', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/improvement_distribution.png', dpi=150, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR}/plots/improvement_distribution.png')
plt.close()

# ============================================================================
# 5. Generate Text Report
# ============================================================================
print('\n[5/5] Generating text report...')

report_path = f'{OUTPUT_DIR}/SGC_Comparison_Summary.txt'
with open(report_path, 'w') as f:
    f.write('='*70 + '\n')
    f.write('SGC COMPARISON: COMPREHENSIVE SUMMARY\n')
    f.write('='*70 + '\n\n')
    
    f.write('Research Question:\n')
    f.write('Can restricted eigenvectors + row normalization enhance SGC?\n\n')
    
    f.write('Two Methods Compared:\n')
    f.write('1. SGC Baseline: X → SGC diffusion → Logistic Regression (with bias)\n')
    f.write('2. Our Variant: X → SGC diffusion → Restricted Eigenvectors → ')
    f.write('Row Norm → Logistic Regression (no bias)\n\n')
    
    f.write('='*70 + '\n')
    f.write('KEY FINDINGS\n')
    f.write('='*70 + '\n\n')
    
    # Overall statistics
    all_improvements = [result['improvement']['absolute_pp'] for result in all_results]
    positive_count = sum(1 for imp in all_improvements if imp > 0)
    negative_count = sum(1 for imp in all_improvements if imp < 0)
    
    f.write(f'Total Experiments: {len(all_results)}\n')
    f.write(f'Datasets Tested: {len(datasets)}\n')
    f.write(f'K Values Tested per Dataset: {len(all_k_values)}\n\n')
    
    f.write(f'Improvements:\n')
    f.write(f'  Positive (RowNorm better): {positive_count}/{len(all_improvements)} ')
    f.write(f'({positive_count/len(all_improvements)*100:.1f}%)\n')
    f.write(f'  Negative (SGC better): {negative_count}/{len(all_improvements)} ')
    f.write(f'({negative_count/len(all_improvements)*100:.1f}%)\n\n')
    
    f.write(f'Average Improvement: {np.mean(all_improvements):+.2f}pp\n')
    f.write(f'Median Improvement: {np.median(all_improvements):+.2f}pp\n')
    f.write(f'Best Improvement: {max(all_improvements):+.2f}pp\n')
    f.write(f'Worst Improvement: {min(all_improvements):+.2f}pp\n\n')
    
    # Per-dataset summary
    f.write('='*70 + '\n')
    f.write('PER-DATASET SUMMARY\n')
    f.write('='*70 + '\n\n')
    
    for dataset in datasets:
        f.write(f'{dataset.upper()}\n')
        f.write('-'*70 + '\n')
        
        k_values = sorted(organized[dataset].keys())
        f.write(f'K values tested: {k_values}\n\n')
        
        # Best results
        sgc_accs = [(k, organized[dataset][k]['sgc_baseline']['test_acc_mean']*100) 
                   for k in k_values]
        best_sgc_k, best_sgc_acc = max(sgc_accs, key=lambda x: x[1])
        
        rownorm_accs = [(k, organized[dataset][k]['rownorm_variant']['test_acc_mean']*100) 
                       for k in k_values]
        best_rownorm_k, best_rownorm_acc = max(rownorm_accs, key=lambda x: x[1])
        
        improvements = [(k, organized[dataset][k]['improvement']['absolute_pp']) 
                       for k in k_values]
        best_imp_k, best_imp = max(improvements, key=lambda x: x[1])
        
        f.write(f'Best SGC: {best_sgc_acc:.2f}% at K={best_sgc_k}\n')
        f.write(f'Best RowNorm: {best_rownorm_acc:.2f}% at K={best_rownorm_k}\n')
        f.write(f'Best Improvement: {best_imp:+.2f}pp at K={best_imp_k}\n')
        f.write(f'Average Improvement: {np.mean([imp for _, imp in improvements]):+.2f}pp\n')
        
        # Verdict
        if best_imp > 2:
            verdict = 'RowNorm variant shows improvement'
        elif best_imp > -2:
            verdict = 'Methods are comparable'
        else:
            verdict = 'SGC baseline is superior'
        f.write(f'Verdict: {verdict}\n\n')
    
    # Technical notes
    f.write('='*70 + '\n')
    f.write('TECHNICAL NOTES\n')
    f.write('='*70 + '\n\n')
    
    f.write('Hyperparameters:\n')
    f.write('  Learning Rate: 0.2 (SGC setting)\n')
    f.write('  Weight Decay: 5e-4\n')
    f.write('  Epochs: 100\n')
    f.write('  Optimizer: Adam\n\n')
    
    f.write('Data Splits:\n')
    f.write('  Random 60/20/20 splits\n')
    f.write('  5 splits × 5 seeds = 25 runs per configuration\n\n')
    
    f.write('SGC Diffusion:\n')
    f.write('  Operator: A_hat = D^(-1/2) (A + I) D^(-1/2)\n')
    f.write('  Propagation: X_tilde = A_hat^K @ X\n\n')
    
    f.write('Restricted Eigenvectors:\n')
    f.write('  Computed from diffused features X_tilde\n')
    f.write('  Solves: L_r @ v = λ D_r @ v\n')
    f.write('  D-orthonormality verified\n\n')
    
    # Conclusions
    f.write('='*70 + '\n')
    f.write('CONCLUSIONS\n')
    f.write('='*70 + '\n\n')
    
    if np.mean(all_improvements) < -5:
        f.write('The RowNorm variant DOES NOT enhance SGC. On average, it significantly\n')
        f.write('underperforms the SGC baseline across datasets and K values.\n\n')
        f.write('Possible reasons:\n')
        f.write('1. Row normalization removes magnitude information\n')
        f.write('2. No bias term constrains decision boundaries\n')
        f.write('3. Single-layer classifier lacks nonlinear capacity\n')
        f.write('4. Restricted eigenvectors from diffused features may not provide\n')
        f.write('   optimal basis for linear classification\n\n')
        f.write('Recommendation: SGC\'s simple approach is superior for this task.\n')
    elif np.mean(all_improvements) > 2:
        f.write('The RowNorm variant shows promise and improves upon SGC baseline.\n')
        f.write('Further investigation warranted.\n')
    else:
        f.write('Results are mixed. Performance depends on dataset characteristics\n')
        f.write('and choice of K. Further analysis needed.\n')

print(f'✓ Saved: {report_path}')

# ============================================================================
# Final Summary
# ============================================================================
print('\n' + '='*70)
print('ANALYSIS COMPLETE')
print('='*70)
print(f'Processed {len(all_results)} experiments across {len(datasets)} datasets')
print(f'\nOutput saved to: {OUTPUT_DIR}/')
print('\nGenerated files:')
print('  Tables:')
print('    - optimal_k_per_dataset.csv')
print('    - complete_results.csv')
print('    - summary_per_dataset.csv')
print('  Plots:')
print('    - improvement_heatmap.png')
print('    - k_trajectories_all.png')
print('    - best_performance_comparison.png')
print('    - improvement_distribution.png')
print('  Report:')
print('    - SGC_Comparison_Summary.txt')
print('='*70)
