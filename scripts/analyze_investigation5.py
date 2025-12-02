"""
===================================================================================
ANALYSIS SCRIPT FOR INVESTIGATION 5: MULTI-SCALE FEATURE FUSION
===================================================================================

This script analyzes results from Investigation 5, which tests:
1. SGC Baseline (k=2)
2. Branch 1 Only (SGC k=2 + MLP)
3. Branch 2 Only (Restricted k=10 + RowNorm + MLP)
4. Dual-Branch (Branch 1 + Branch 2)
5. Triple-Branch (Branch 1 + Branch 2 augmented with log-magnitude)

Generates:
- Summary tables (CSV and LaTeX)
- Performance comparison figures
- Fusion gain analysis
- Comprehensive report

Author: Mohammad
Date: December 2025
===================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Investigation 5 Analysis')
parser.add_argument('--results_dir', type=str, 
                   default='results/investigation5_multiscale_fusion',
                   help='Base directory containing results')
parser.add_argument('--split_type', type=str, choices=['fixed', 'random', 'both'],
                   default='both', help='Which split type to analyze')
args = parser.parse_args()

RESULTS_BASE = args.results_dir
SPLIT_TYPE = args.split_type
OUTPUT_DIR = Path(RESULTS_BASE) / f'analysis_{SPLIT_TYPE}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style('whitegrid')

print('='*80)
print('INVESTIGATION 5: MULTI-SCALE FEATURE FUSION - ANALYSIS')
print('='*80)
print(f'Results directory: {RESULTS_BASE}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Output directory: {OUTPUT_DIR}')
print('='*80)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_results(base_dir):
    """
    Load results from all datasets and configurations
    
    Expected structure:
    results/investigation5_multiscale_fusion/
    ├── dataset_fixed_lcc/
    │   └── k2_k10/
    │       └── metrics/
    │           └── results.json
    ├── dataset_random_lcc/
    │   └── k2_k10/
    │       └── metrics/
    │           └── results.json
    """
    data = {}
    base_path = Path(base_dir)
    
    print('\n[1/5] Loading results...')
    
    if not base_path.exists():
        print(f'ERROR: Results directory not found: {base_path}')
        sys.exit(1)
    
    # Find all result files
    result_files = list(base_path.glob('**/results.json'))
    
    if len(result_files) == 0:
        print(f'ERROR: No results.json files found in {base_path}')
        sys.exit(1)
    
    print(f'Found {len(result_files)} result files')
    
    for result_file in result_files:
        try:
            # Parse path to extract dataset, split_type, k values
            # Example: dataset_fixed_lcc/k2_k10/metrics/results.json
            parts = result_file.parts
            
            # Find dataset_split_component
            dataset_split_comp = None
            k_config = None
            for i, part in enumerate(parts):
                if '_lcc' in part or '_full' in part:
                    dataset_split_comp = part
                    if i + 1 < len(parts) and parts[i+1].startswith('k'):
                        k_config = parts[i+1]
                    break
            
            if not dataset_split_comp or not k_config:
                print(f'Warning: Could not parse path: {result_file}')
                continue
            
            # Parse dataset_split_comp
            # Format: dataset_splittype_component
            parts_dsc = dataset_split_comp.rsplit('_', 2)
            if len(parts_dsc) < 3:
                print(f'Warning: Could not parse dataset_split_comp: {dataset_split_comp}')
                continue
            
            dataset = parts_dsc[0]
            split_type = parts_dsc[1]
            component = parts_dsc[2]
            
            # Parse k values from k_config (e.g., "k2_k10")
            k_parts = k_config.split('_')
            k_low = int(k_parts[0][1:]) if len(k_parts) > 0 else 2
            k_high = int(k_parts[1][1:]) if len(k_parts) > 1 else 10
            
            # Load results
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            key = (dataset, split_type, k_low, k_high)
            data[key] = results
            
            print(f'  ✓ Loaded: {dataset} ({split_type}, k_low={k_low}, k_high={k_high})')
            
        except Exception as e:
            print(f'Warning: Failed to load {result_file}: {e}')
            continue
    
    if len(data) == 0:
        print('ERROR: No valid results loaded')
        sys.exit(1)
    
    print(f'\n✓ Loaded {len(data)} experiments total')
    return data

all_results = load_all_results(RESULTS_BASE)

# ============================================================================
# CREATE SUMMARY DATAFRAME
# ============================================================================

print('\n[2/5] Creating summary tables...')

summary_data = []

for (dataset, split_type, k_low, k_high), results in all_results.items():
    exp_results = results.get('results', {})
    analysis = results.get('analysis', {})
    metadata = results.get('metadata', {})
    
    row = {
        'Dataset': dataset,
        'Split': split_type,
        'k_low': k_low,
        'k_high': k_high,
        # Method performances - access through 'aggregated' key
        'SGC Baseline': exp_results.get('sgc_baseline', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Branch 1 Only': exp_results.get('branch1_only', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Branch 2 Only': exp_results.get('branch2_only', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Dual-Branch': exp_results.get('dual_branch', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Triple-Branch': exp_results.get('triple_branch', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        # Standard deviations - also through 'aggregated'
        'SGC Std': exp_results.get('sgc_baseline', {}).get('aggregated', {}).get('test_acc_std', np.nan) * 100,
        'B1 Std': exp_results.get('branch1_only', {}).get('aggregated', {}).get('test_acc_std', np.nan) * 100,
        'B2 Std': exp_results.get('branch2_only', {}).get('aggregated', {}).get('test_acc_std', np.nan) * 100,
        'Dual Std': exp_results.get('dual_branch', {}).get('aggregated', {}).get('test_acc_std', np.nan) * 100,
        'Triple Std': exp_results.get('triple_branch', {}).get('aggregated', {}).get('test_acc_std', np.nan) * 100,
        # Analysis metrics
        'Best Single': analysis.get('best_single_branch', np.nan),
        'Dual Gain': analysis.get('dual_fusion_gain', np.nan),
        'Triple Gain': analysis.get('triple_fusion_gain', np.nan),
        'Magnitude Contribution': analysis.get('magnitude_contribution', np.nan),
        # Metadata
        'Num Nodes': metadata.get('num_nodes', np.nan),
        'Num Classes': metadata.get('num_classes', np.nan),
        'd_sgc': metadata.get('d_sgc', np.nan),
        'd_eigenvec': metadata.get('d_eigenvec', np.nan),
    }
    
    summary_data.append(row)

df = pd.DataFrame(summary_data)

# Sort by dataset and split type
df = df.sort_values(['Dataset', 'Split'])

# Filter by split type if requested
if SPLIT_TYPE != 'both':
    df = df[df['Split'] == SPLIT_TYPE]

# Save summary table
summary_csv = OUTPUT_DIR / 'summary_table.csv'
df.to_csv(summary_csv, index=False, float_format='%.2f')
print(f'✓ Saved: {summary_csv}')

# Create LaTeX table (main results only)
df_latex = df[['Dataset', 'Split', 'SGC Baseline', 'Branch 1 Only', 'Branch 2 Only', 
               'Dual-Branch', 'Triple-Branch']].copy()
summary_tex = OUTPUT_DIR / 'summary_table.tex'
with open(summary_tex, 'w') as f:
    f.write(df_latex.to_latex(index=False, float_format='%.2f'))
print(f'✓ Saved: {summary_tex}')

# ============================================================================
# FUSION GAIN ANALYSIS
# ============================================================================

print('\n[3/5] Analyzing fusion gains...')

# Categorize results
df['Dual Outcome'] = df['Dual Gain'].apply(
    lambda x: 'Helps (>0.5pp)' if x > 0.5 else ('Neutral' if x > -0.5 else 'Hurts (<-0.5pp)')
)

df['Triple Outcome'] = df['Triple Gain'].apply(
    lambda x: 'Helps (>0.5pp)' if x > 0.5 else ('Neutral' if x > -0.5 else 'Hurts (<-0.5pp)')
)

df['Magnitude Effect'] = df['Magnitude Contribution'].apply(
    lambda x: 'Positive (>0.5pp)' if x > 0.5 else ('Neutral' if x > -0.5 else 'Negative (<-0.5pp)')
)

# Create detailed comparison table
comparison_df = df[['Dataset', 'Split', 'Best Single', 'Dual-Branch', 'Triple-Branch',
                   'Dual Gain', 'Triple Gain', 'Magnitude Contribution',
                   'Dual Outcome', 'Triple Outcome', 'Magnitude Effect']].copy()

comparison_file = OUTPUT_DIR / 'fusion_gain_analysis.csv'
comparison_df.to_csv(comparison_file, index=False, float_format='%.2f')
print(f'✓ Saved: {comparison_file}')

# Print summary statistics
print('\nFusion Gain Statistics:')
print('='*60)
print(f'\nDual-Branch:')
print(f'  Helps:   {(df["Dual Outcome"] == "Helps (>0.5pp)").sum()} / {len(df)} datasets')
print(f'  Neutral: {(df["Dual Outcome"] == "Neutral").sum()} / {len(df)} datasets')
print(f'  Hurts:   {(df["Dual Outcome"] == "Hurts (<-0.5pp)").sum()} / {len(df)} datasets')
print(f'  Mean gain: {df["Dual Gain"].mean():+.2f}pp')
print(f'  Median gain: {df["Dual Gain"].median():+.2f}pp')

print(f'\nTriple-Branch:')
print(f'  Helps:   {(df["Triple Outcome"] == "Helps (>0.5pp)").sum()} / {len(df)} datasets')
print(f'  Neutral: {(df["Triple Outcome"] == "Neutral").sum()} / {len(df)} datasets')
print(f'  Hurts:   {(df["Triple Outcome"] == "Hurts (<-0.5pp)").sum()} / {len(df)} datasets')
print(f'  Mean gain: {df["Triple Gain"].mean():+.2f}pp')
print(f'  Median gain: {df["Triple Gain"].median():+.2f}pp')

print(f'\nMagnitude Contribution (Triple - Dual):')
print(f'  Positive: {(df["Magnitude Effect"] == "Positive (>0.5pp)").sum()} / {len(df)} datasets')
print(f'  Neutral:  {(df["Magnitude Effect"] == "Neutral").sum()} / {len(df)} datasets')
print(f'  Negative: {(df["Magnitude Effect"] == "Negative (<-0.5pp)").sum()} / {len(df)} datasets')
print(f'  Mean contribution: {df["Magnitude Contribution"].mean():+.2f}pp')
print(f'  Median contribution: {df["Magnitude Contribution"].median():+.2f}pp')

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print('\n[4/5] Generating visualizations...')

# Figure 1: Method comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))

datasets = df['Dataset'].values
x = np.arange(len(datasets))
width = 0.15

methods = ['SGC Baseline', 'Branch 1 Only', 'Branch 2 Only', 'Dual-Branch', 'Triple-Branch']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (method, color) in enumerate(zip(methods, colors)):
    offset = (i - 2) * width
    values = df[method].values
    ax.bar(x + offset, values, width, label=method, alpha=0.8, color=color)

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Investigation 5: Method Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n({s})" for d, s in zip(df['Dataset'], df['Split'])], 
                    rotation=45, ha='right', fontsize=9)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig1_file = OUTPUT_DIR / 'method_comparison.png'
plt.savefig(fig1_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig1_file}')
plt.close()

# Figure 2: Fusion gains comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Dual-branch gains
colors_dual = ['green' if x > 0.5 else 'gray' if x > -0.5 else 'red' 
               for x in df['Dual Gain'].values]
ax1.barh(x, df['Dual Gain'].values, color=colors_dual, alpha=0.7)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.axvline(x=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.set_xlabel('Dual Fusion Gain (pp)', fontsize=11)
ax1.set_title('Dual-Branch vs Best Single Branch', fontsize=12, fontweight='bold')
ax1.set_yticks(x)
ax1.set_yticklabels([f"{d} ({s})" for d, s in zip(df['Dataset'], df['Split'])], fontsize=8)
ax1.grid(True, alpha=0.3, axis='x')

# Triple-branch gains
colors_triple = ['green' if x > 0.5 else 'gray' if x > -0.5 else 'red' 
                 for x in df['Triple Gain'].values]
ax2.barh(x, df['Triple Gain'].values, color=colors_triple, alpha=0.7)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.axvline(x=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.set_xlabel('Triple Fusion Gain (pp)', fontsize=11)
ax2.set_title('Triple-Branch vs Best Single Branch', fontsize=12, fontweight='bold')
ax2.set_yticks(x)
ax2.set_yticklabels([f"{d} ({s})" for d, s in zip(df['Dataset'], df['Split'])], fontsize=8)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig2_file = OUTPUT_DIR / 'fusion_gains.png'
plt.savefig(fig2_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig2_file}')
plt.close()

# Figure 3: Magnitude contribution
fig, ax = plt.subplots(figsize=(10, 6))

colors_mag = ['green' if x > 0.5 else 'gray' if x > -0.5 else 'red' 
              for x in df['Magnitude Contribution'].values]
ax.barh(x, df['Magnitude Contribution'].values, color=colors_mag, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Magnitude Contribution (Triple - Dual, pp)', fontsize=11)
ax.set_title('Log-Magnitude Augmentation Effect', fontsize=12, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels([f"{d} ({s})" for d, s in zip(df['Dataset'], df['Split'])], fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig3_file = OUTPUT_DIR / 'magnitude_contribution.png'
plt.savefig(fig3_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig3_file}')
plt.close()

# Figure 4: Scatter plot - Dual vs Triple gain
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['Dual Gain'], df['Triple Gain'], s=100, alpha=0.6)

# Add diagonal line (dual = triple)
min_val = min(df['Dual Gain'].min(), df['Triple Gain'].min())
max_val = max(df['Dual Gain'].max(), df['Triple Gain'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Dual = Triple')

# Add zero lines
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Annotate points
for idx, row in df.iterrows():
    ax.annotate(f"{row['Dataset']}\n({row['Split']})", 
                (row['Dual Gain'], row['Triple Gain']),
                fontsize=7, ha='center', alpha=0.7)

ax.set_xlabel('Dual-Branch Gain (pp)', fontsize=12)
ax.set_ylabel('Triple-Branch Gain (pp)', fontsize=12)
ax.set_title('Dual-Branch vs Triple-Branch Fusion Gains', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4_file = OUTPUT_DIR / 'dual_vs_triple_scatter.png'
plt.savefig(fig4_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig4_file}')
plt.close()

# Figure 5: Performance heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Create heatmap data
heatmap_data = df[methods].values.T
dataset_labels = [f"{d} ({s})" for d, s in zip(df['Dataset'], df['Split'])]

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(dataset_labels)))
ax.set_yticks(np.arange(len(methods)))
ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(methods, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Test Accuracy (%)', fontsize=11)

# Add text annotations
for i in range(len(methods)):
    for j in range(len(dataset_labels)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Performance Heatmap: All Methods', fontsize=14, fontweight='bold')

plt.tight_layout()
fig5_file = OUTPUT_DIR / 'performance_heatmap.png'
plt.savefig(fig5_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig5_file}')
plt.close()

# ============================================================================
# GENERATE COMPREHENSIVE REPORT
# ============================================================================

print('\n[5/5] Generating comprehensive report...')

report_file = OUTPUT_DIR / 'investigation5_comprehensive_report.txt'

with open(report_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('INVESTIGATION 5: MULTI-SCALE FEATURE FUSION\n')
    f.write('Comprehensive Analysis Report\n')
    f.write('='*80 + '\n\n')
    
    f.write('RESEARCH QUESTION\n')
    f.write('-'*80 + '\n')
    f.write('Does combining low-diffusion features (k=2) with high-diffusion restricted\n')
    f.write('eigenvectors (k=10) improve performance? Does augmenting with log-magnitude\n')
    f.write('(from Investigation 3) provide additional benefit?\n\n')
    
    f.write('METHODS TESTED\n')
    f.write('-'*80 + '\n')
    f.write('1. SGC Baseline: Logistic regression on SGC k=2\n')
    f.write('2. Branch 1 Only: SGC k=2 + MLP\n')
    f.write('3. Branch 2 Only: Restricted k=10 + RowNorm + MLP\n')
    f.write('4. Dual-Branch: Fusion of Branch 1 + Branch 2\n')
    f.write('5. Triple-Branch: Branch 1 + Branch 2 (augmented with log-magnitude)\n\n')
    
    f.write('SUMMARY STATISTICS\n')
    f.write('-'*80 + '\n')
    f.write(f'Total experiments: {len(df)}\n')
    f.write(f'Datasets: {df["Dataset"].nunique()}\n')
    f.write(f'Split types: {df["Split"].unique()}\n\n')
    
    f.write('DUAL-BRANCH RESULTS\n')
    f.write('-'*80 + '\n')
    f.write(f'Success rate (>0.5pp gain): {(df["Dual Gain"] > 0.5).sum()} / {len(df)} = {(df["Dual Gain"] > 0.5).sum()/len(df)*100:.1f}%\n')
    f.write(f'Mean gain: {df["Dual Gain"].mean():+.2f}pp (std: {df["Dual Gain"].std():.2f}pp)\n')
    f.write(f'Median gain: {df["Dual Gain"].median():+.2f}pp\n')
    f.write(f'Range: [{df["Dual Gain"].min():.2f}pp, {df["Dual Gain"].max():.2f}pp]\n\n')
    
    # Best dual cases
    best_dual = df.nlargest(3, 'Dual Gain')
    f.write('Top 3 Dual-Branch gains:\n')
    for idx, row in best_dual.iterrows():
        f.write(f'  {row["Dataset"]} ({row["Split"]}): {row["Dual Gain"]:+.2f}pp\n')
    f.write('\n')
    
    f.write('TRIPLE-BRANCH RESULTS\n')
    f.write('-'*80 + '\n')
    f.write(f'Success rate (>0.5pp gain): {(df["Triple Gain"] > 0.5).sum()} / {len(df)} = {(df["Triple Gain"] > 0.5).sum()/len(df)*100:.1f}%\n')
    f.write(f'Mean gain: {df["Triple Gain"].mean():+.2f}pp (std: {df["Triple Gain"].std():.2f}pp)\n')
    f.write(f'Median gain: {df["Triple Gain"].median():+.2f}pp\n')
    f.write(f'Range: [{df["Triple Gain"].min():.2f}pp, {df["Triple Gain"].max():.2f}pp]\n\n')
    
    # Best triple cases
    best_triple = df.nlargest(3, 'Triple Gain')
    f.write('Top 3 Triple-Branch gains:\n')
    for idx, row in best_triple.iterrows():
        f.write(f'  {row["Dataset"]} ({row["Split"]}): {row["Triple Gain"]:+.2f}pp\n')
    f.write('\n')
    
    f.write('MAGNITUDE CONTRIBUTION (Triple - Dual)\n')
    f.write('-'*80 + '\n')
    f.write(f'Positive effect (>0.5pp): {(df["Magnitude Contribution"] > 0.5).sum()} / {len(df)} = {(df["Magnitude Contribution"] > 0.5).sum()/len(df)*100:.1f}%\n')
    f.write(f'Neutral effect (-0.5 to +0.5pp): {((df["Magnitude Contribution"] >= -0.5) & (df["Magnitude Contribution"] <= 0.5)).sum()} / {len(df)}\n')
    f.write(f'Negative effect (<-0.5pp): {(df["Magnitude Contribution"] < -0.5).sum()} / {len(df)} = {(df["Magnitude Contribution"] < -0.5).sum()/len(df)*100:.1f}%\n')
    f.write(f'Mean contribution: {df["Magnitude Contribution"].mean():+.2f}pp (std: {df["Magnitude Contribution"].std():.2f}pp)\n')
    f.write(f'Median contribution: {df["Magnitude Contribution"].median():+.2f}pp\n\n')
    
    # Best magnitude contributions
    best_mag = df.nlargest(3, 'Magnitude Contribution')
    f.write('Top 3 Magnitude contributions:\n')
    for idx, row in best_mag.iterrows():
        f.write(f'  {row["Dataset"]} ({row["Split"]}): {row["Magnitude Contribution"]:+.2f}pp\n')
    f.write('\n')
    
    # Worst magnitude contributions
    worst_mag = df.nsmallest(3, 'Magnitude Contribution')
    f.write('Bottom 3 Magnitude contributions (where it hurt):\n')
    for idx, row in worst_mag.iterrows():
        f.write(f'  {row["Dataset"]} ({row["Split"]}): {row["Magnitude Contribution"]:+.2f}pp\n')
    f.write('\n')
    
    f.write('DETAILED RESULTS TABLE\n')
    f.write('-'*80 + '\n')
    summary_str = df[['Dataset', 'Split', 'SGC Baseline', 'Branch 1 Only', 
                      'Branch 2 Only', 'Dual-Branch', 'Triple-Branch',
                      'Dual Gain', 'Triple Gain', 'Magnitude Contribution']].to_string(index=False)
    f.write(summary_str)
    f.write('\n\n')
    
    f.write('KEY FINDINGS\n')
    f.write('-'*80 + '\n')
    
    # Finding 1: Overall effectiveness
    dual_success_rate = (df["Dual Gain"] > 0.5).sum() / len(df) * 100
    triple_success_rate = (df["Triple Gain"] > 0.5).sum() / len(df) * 100
    
    f.write(f'1. Multi-scale fusion (Dual-Branch) helps in {dual_success_rate:.0f}% of cases\n')
    if dual_success_rate > 50:
        f.write('   → Multi-scale fusion is generally beneficial\n')
    else:
        f.write('   → Multi-scale fusion has limited benefit\n')
    f.write('\n')
    
    f.write(f'2. Log-magnitude augmentation (Triple-Branch) helps in {triple_success_rate:.0f}% of cases\n')
    mag_positive = (df["Magnitude Contribution"] > 0.5).sum() / len(df) * 100
    if mag_positive > 50:
        f.write(f'   → Magnitude augmentation is beneficial ({mag_positive:.0f}% positive effect)\n')
    elif mag_positive > 30:
        f.write(f'   → Magnitude augmentation has mixed results ({mag_positive:.0f}% positive)\n')
    else:
        f.write(f'   → Magnitude augmentation rarely helps ({mag_positive:.0f}% positive)\n')
    f.write('\n')
    
    # Finding 3: Best performing method
    best_method_counts = {
        'SGC Baseline': (df['SGC Baseline'] == df[methods].max(axis=1)).sum(),
        'Branch 1 Only': (df['Branch 1 Only'] == df[methods].max(axis=1)).sum(),
        'Branch 2 Only': (df['Branch 2 Only'] == df[methods].max(axis=1)).sum(),
        'Dual-Branch': (df['Dual-Branch'] == df[methods].max(axis=1)).sum(),
        'Triple-Branch': (df['Triple-Branch'] == df[methods].max(axis=1)).sum(),
    }
    best_method = max(best_method_counts.items(), key=lambda x: x[1])
    f.write(f'3. Best performing method overall: {best_method[0]} ({best_method[1]}/{len(df)} datasets)\n')
    f.write('\n')
    
    f.write('CONCLUSION\n')
    f.write('-'*80 + '\n')
    f.write('Investigation 5 tested multi-scale feature fusion combining low-diffusion\n')
    f.write('(k=2) and high-diffusion (k=10) features with optional log-magnitude\n')
    f.write('augmentation. Results show:\n\n')
    
    if dual_success_rate > 50:
        f.write('- Multi-scale fusion (Dual-Branch) is generally effective\n')
    else:
        f.write('- Multi-scale fusion (Dual-Branch) has limited benefit\n')
    
    if mag_positive > 50:
        f.write('- Log-magnitude augmentation provides additional value\n')
    elif mag_positive > 30:
        f.write('- Log-magnitude augmentation has dataset-dependent benefit\n')
    else:
        f.write('- Log-magnitude augmentation rarely improves results\n')
    
    f.write(f'\nBest overall method: {best_method[0]}\n')
    f.write('\n')
    f.write('='*80 + '\n')

print(f'✓ Saved: {report_file}')

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print('='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print()
print('Generated files:')
print(f'  Summary table (CSV): {summary_csv}')
print(f'  Summary table (LaTeX): {summary_tex}')
print(f'  Fusion gain analysis: {comparison_file}')
print(f'  Figure 1 (Method comparison): {fig1_file}')
print(f'  Figure 2 (Fusion gains): {fig2_file}')
print(f'  Figure 3 (Magnitude contribution): {fig3_file}')
print(f'  Figure 4 (Dual vs Triple scatter): {fig4_file}')
print(f'  Figure 5 (Performance heatmap): {fig5_file}')
print(f'  Comprehensive report: {report_file}')
print()
print('Next step: Review the report and share with Yiannis!')
print()