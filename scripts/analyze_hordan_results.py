"""
Analyze Hordan Framework Results
================================

Loads the JSON results from investigation_hordan_analysis.py and produces:
1. Summary tables (console + LaTeX)
2. Cross-dataset comparison
3. Correlation analysis
4. Publication-ready visualizations

Usage:
    python analyze_hordan_results.py
    python analyze_hordan_results.py --json_path results/hordan_analysis/hordan_analysis_results.json
    python analyze_hordan_results.py --output_dir results/hordan_analysis/analysis
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Analyze Hordan Framework Results')
parser.add_argument('--json_path', type=str, 
                   default='results/hordan_analysis/hordan_analysis_results.json',
                   help='Path to results JSON file')
parser.add_argument('--output_dir', type=str, 
                   default='results/hordan_analysis/analysis',
                   help='Output directory for analysis')
args = parser.parse_args()

JSON_PATH = args.json_path
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*80)
print('HORDAN FRAMEWORK RESULTS ANALYSIS')
print('='*80)

# ============================================================================
# Load Results
# ============================================================================

print(f'\nLoading results from: {JSON_PATH}')

with open(JSON_PATH, 'r') as f:
    all_results = json.load(f)

print(f'Loaded {len(all_results)} dataset-k combinations')

# ============================================================================
# Extract Data into DataFrames
# ============================================================================

# Main results table
rows = []
for key, result in all_results.items():
    dataset = result['dataset']
    k = result['k_diffusion']
    
    # Metadata
    num_nodes = result['metadata'].get('num_nodes_lcc', 0)
    num_features = result['metadata'].get('num_features', 0)
    num_classes = result['metadata'].get('num_classes', 0)
    d_eff = result['metadata'].get('d_effective', 0)
    
    # Sparsity (threshold abs_1e-6)
    sparsity_features = result['sparsity']['features']['thresholds']['abs_1e-6']['sparsity_ratio']
    sparsity_diffused = result['sparsity']['X_diffused']['thresholds']['abs_1e-6']['sparsity_ratio']
    sparsity_U = result['sparsity']['U']['thresholds']['abs_1e-6']['sparsity_ratio']
    
    ref_nodes_features = result['sparsity']['features']['thresholds']['abs_1e-6']['rows_with_no_zeros_pct']
    ref_nodes_diffused = result['sparsity']['X_diffused']['thresholds']['abs_1e-6']['rows_with_no_zeros_pct']
    ref_nodes_U = result['sparsity']['U']['thresholds']['abs_1e-6']['rows_with_no_zeros_pct']
    
    # Magnitude correlation
    fisher_features = result['magnitude_correlation']['features']['separability_metrics']['fisher_ratio']
    fisher_diffused = result['magnitude_correlation']['X_diffused']['separability_metrics']['fisher_ratio']
    fisher_U = result['magnitude_correlation']['U']['separability_metrics']['fisher_ratio']
    
    nmi_features = result['magnitude_correlation']['features']['separability_metrics']['normalized_mutual_info']
    nmi_diffused = result['magnitude_correlation']['X_diffused']['separability_metrics']['normalized_mutual_info']
    nmi_U = result['magnitude_correlation']['U']['separability_metrics']['normalized_mutual_info']
    
    # Equivariant fix test
    fix = result['equivariant_fix']
    sgc_std = fix['results']['sgc_standard']['test_acc_mean']
    restr_std = fix['results']['restricted_standard']['test_acc_mean']
    restr_rn = fix['results']['restricted_rownorm']['test_acc_mean']
    restr_logmag = fix['results']['restricted_magnitude_preserving']['test_acc_mean']
    
    sgc_std_se = fix['results']['sgc_standard']['test_acc_std']
    restr_std_se = fix['results']['restricted_standard']['test_acc_std']
    restr_rn_se = fix['results']['restricted_rownorm']['test_acc_std']
    restr_logmag_se = fix['results']['restricted_magnitude_preserving']['test_acc_std']
    
    # Gaps
    part_a_gap = fix['gaps']['part_a_gap']
    part_b_improvement = fix['gaps']['part_b_improvement']
    hordan_fix = fix['gaps']['magnitude_preserving_vs_rownorm']
    total_vs_baseline = fix['gaps']['total_vs_baseline']
    
    rows.append({
        'dataset': dataset,
        'k': k,
        'key': key,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
        'd_eff': d_eff,
        # Sparsity
        'sparsity_features': sparsity_features,
        'sparsity_diffused': sparsity_diffused,
        'sparsity_U': sparsity_U,
        'ref_nodes_features': ref_nodes_features,
        'ref_nodes_diffused': ref_nodes_diffused,
        'ref_nodes_U': ref_nodes_U,
        # Fisher ratio
        'fisher_features': fisher_features,
        'fisher_diffused': fisher_diffused,
        'fisher_U': fisher_U,
        # NMI
        'nmi_features': nmi_features,
        'nmi_diffused': nmi_diffused,
        'nmi_U': nmi_U,
        # Accuracies
        'sgc_std': sgc_std,
        'restr_std': restr_std,
        'restr_rn': restr_rn,
        'restr_logmag': restr_logmag,
        'sgc_std_se': sgc_std_se,
        'restr_std_se': restr_std_se,
        'restr_rn_se': restr_rn_se,
        'restr_logmag_se': restr_logmag_se,
        # Gaps
        'part_a_gap': part_a_gap,
        'part_b_improvement': part_b_improvement,
        'hordan_fix': hordan_fix,
        'total_vs_baseline': total_vs_baseline,
    })

df = pd.DataFrame(rows)
print(f'\nDataFrame shape: {df.shape}')

# ============================================================================
# Summary Table 1: Main Results
# ============================================================================

print('\n' + '='*80)
print('TABLE 1: EQUIVARIANT FIX TEST RESULTS')
print('='*80)

table1_cols = ['key', 'sgc_std', 'restr_std', 'restr_rn', 'restr_logmag', 
               'part_a_gap', 'part_b_improvement', 'hordan_fix']

table1 = df[table1_cols].copy()
table1['sgc_std'] = (table1['sgc_std'] * 100).round(2)
table1['restr_std'] = (table1['restr_std'] * 100).round(2)
table1['restr_rn'] = (table1['restr_rn'] * 100).round(2)
table1['restr_logmag'] = (table1['restr_logmag'] * 100).round(2)
table1['part_a_gap'] = (table1['part_a_gap'] * 100).round(2)
table1['part_b_improvement'] = (table1['part_b_improvement'] * 100).round(2)
table1['hordan_fix'] = (table1['hordan_fix'] * 100).round(2)

table1.columns = ['Dataset_k', 'SGC+Std', 'Restr+Std', 'Restr+RN', 'Restr+LogMag',
                  'Part A Gap', 'Part B Impr', 'Hordan Fix']

print(table1.to_string(index=False))

# ============================================================================
# Summary Table 2: Sparsity Analysis
# ============================================================================

print('\n' + '='*80)
print('TABLE 2: SPARSITY ANALYSIS')
print('='*80)

table2_cols = ['key', 'sparsity_features', 'sparsity_diffused', 'sparsity_U',
               'ref_nodes_features', 'ref_nodes_diffused', 'ref_nodes_U']

table2 = df[table2_cols].copy()
table2['sparsity_features'] = (table2['sparsity_features'] * 100).round(2)
table2['sparsity_diffused'] = (table2['sparsity_diffused'] * 100).round(2)
table2['sparsity_U'] = (table2['sparsity_U'] * 100).round(2)
table2['ref_nodes_features'] = table2['ref_nodes_features'].round(1)
table2['ref_nodes_diffused'] = table2['ref_nodes_diffused'].round(1)
table2['ref_nodes_U'] = table2['ref_nodes_U'].round(1)

table2.columns = ['Dataset_k', 'Sparse_feat%', 'Sparse_diff%', 'Sparse_U%',
                  'RefNode_feat%', 'RefNode_diff%', 'RefNode_U%']

print(table2.to_string(index=False))

# ============================================================================
# Summary Table 3: Magnitude-Class Correlation
# ============================================================================

print('\n' + '='*80)
print('TABLE 3: MAGNITUDE-CLASS CORRELATION (Fisher Ratio)')
print('='*80)

table3_cols = ['key', 'fisher_features', 'fisher_diffused', 'fisher_U', 'hordan_fix']

table3 = df[table3_cols].copy()
table3['fisher_features'] = table3['fisher_features'].round(4)
table3['fisher_diffused'] = table3['fisher_diffused'].round(4)
table3['fisher_U'] = table3['fisher_U'].round(4)
table3['hordan_fix'] = (table3['hordan_fix'] * 100).round(2)

table3.columns = ['Dataset_k', 'Fisher_feat', 'Fisher_diff', 'Fisher_U', 'Hordan_Fix_pp']

print(table3.to_string(index=False))

# ============================================================================
# Correlation Analysis
# ============================================================================

print('\n' + '='*80)
print('CORRELATION ANALYSIS')
print('='*80)

# Key question: Does Fisher_U predict Hordan Fix?
corr_fisher_hordan, p_fisher = stats.pearsonr(df['fisher_U'], df['hordan_fix'])
corr_sparsity_hordan, p_sparsity = stats.pearsonr(df['sparsity_U'], df['hordan_fix'])
corr_refnodes_hordan, p_refnodes = stats.pearsonr(df['ref_nodes_U'], df['hordan_fix'])

print(f'\nCorrelation with Hordan Fix (LogMag - RowNorm):')
print(f'  Fisher_U:     r = {corr_fisher_hordan:+.4f}  (p = {p_fisher:.4f})')
print(f'  Sparsity_U:   r = {corr_sparsity_hordan:+.4f}  (p = {p_sparsity:.4f})')
print(f'  RefNodes_U:   r = {corr_refnodes_hordan:+.4f}  (p = {p_refnodes:.4f})')

# Does Fisher_U predict Part B improvement?
corr_fisher_partb, p_fisher_partb = stats.pearsonr(df['fisher_U'], df['part_b_improvement'])
print(f'\nCorrelation with Part B (RowNorm - StandardMLP):')
print(f'  Fisher_U:     r = {corr_fisher_partb:+.4f}  (p = {p_fisher_partb:.4f})')

# Interpretation
print('\nInterpretation:')
if corr_fisher_hordan > 0.3:
    print('  → Higher magnitude-class correlation in U → Hordan fix helps MORE')
elif corr_fisher_hordan < -0.3:
    print('  → Higher magnitude-class correlation in U → Hordan fix helps LESS (redundancy!)')
else:
    print('  → No strong relationship between Fisher_U and Hordan fix')

# ============================================================================
# Pattern Analysis: When does Hordan fix help?
# ============================================================================

print('\n' + '='*80)
print('PATTERN ANALYSIS: WHEN DOES HORDAN FIX HELP?')
print('='*80)

# Split by whether Hordan fix helped
df['hordan_helped'] = df['hordan_fix'] > 0

helped = df[df['hordan_helped']]
hurt = df[~df['hordan_helped']]

print(f'\nDatasets where Hordan fix HELPED ({len(helped)}):')
if len(helped) > 0:
    for _, row in helped.iterrows():
        print(f'  {row["key"]}: {row["hordan_fix"]*100:+.2f}pp (Fisher_U={row["fisher_U"]:.4f})')
else:
    print('  None')

print(f'\nDatasets where Hordan fix HURT ({len(hurt)}):')
if len(hurt) > 0:
    for _, row in hurt.iterrows():
        print(f'  {row["key"]}: {row["hordan_fix"]*100:+.2f}pp (Fisher_U={row["fisher_U"]:.4f})')
else:
    print('  None')

# Compare means
if len(helped) > 0 and len(hurt) > 0:
    print(f'\nMean Fisher_U when helped: {helped["fisher_U"].mean():.4f}')
    print(f'Mean Fisher_U when hurt:   {hurt["fisher_U"].mean():.4f}')
    print(f'Mean Sparsity_U when helped: {helped["sparsity_U"].mean()*100:.2f}%')
    print(f'Mean Sparsity_U when hurt:   {hurt["sparsity_U"].mean()*100:.2f}%')

# ============================================================================
# Generate LaTeX Tables
# ============================================================================

print('\n' + '='*80)
print('LATEX TABLE: MAIN RESULTS')
print('='*80)

latex_table = """
\\begin{table}[h]
\\centering
\\caption{Hordan Framework Analysis: Equivariant Fix Test Results}
\\label{tab:hordan_results}
\\begin{tabular}{l|cccc|ccc}
\\toprule
Dataset & SGC+Std & Restr+Std & Restr+RN & Restr+LogMag & Part A & Part B & Hordan Fix \\\\
\\midrule
"""

for _, row in df.iterrows():
    latex_table += f"{row['key'].replace('_', '\\_')} & "
    latex_table += f"{row['sgc_std']*100:.1f} & "
    latex_table += f"{row['restr_std']*100:.1f} & "
    latex_table += f"{row['restr_rn']*100:.1f} & "
    latex_table += f"{row['restr_logmag']*100:.1f} & "
    latex_table += f"{row['part_a_gap']*100:+.1f} & "
    latex_table += f"{row['part_b_improvement']*100:+.1f} & "
    latex_table += f"{row['hordan_fix']*100:+.1f} \\\\\n"

latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

print(latex_table)

# Save LaTeX table
with open(f'{OUTPUT_DIR}/hordan_results_table.tex', 'w') as f:
    f.write(latex_table)
print(f'\n✓ Saved: {OUTPUT_DIR}/hordan_results_table.tex')

# ============================================================================
# Visualizations
# ============================================================================

print('\n' + '='*80)
print('GENERATING VISUALIZATIONS')
print('='*80)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Plot 1: Hordan Fix vs Fisher_U
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scatter: Fisher_U vs Hordan Fix
ax1 = axes[0]
ax1.scatter(df['fisher_U'], df['hordan_fix'] * 100, s=100, alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Fisher Ratio (U)', fontsize=12)
ax1.set_ylabel('Hordan Fix (pp)', fontsize=12)
ax1.set_title(f'Magnitude Correlation vs Hordan Fix\nr = {corr_fisher_hordan:.3f}', fontsize=12)

# Add labels
for _, row in df.iterrows():
    ax1.annotate(row['key'].split('_')[0], 
                (row['fisher_U'], row['hordan_fix'] * 100),
                fontsize=8, alpha=0.7)

# Scatter: Sparsity_U vs Hordan Fix
ax2 = axes[1]
ax2.scatter(df['sparsity_U'] * 100, df['hordan_fix'] * 100, s=100, alpha=0.7)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Sparsity of U (%)', fontsize=12)
ax2.set_ylabel('Hordan Fix (pp)', fontsize=12)
ax2.set_title(f'Sparsity vs Hordan Fix\nr = {corr_sparsity_hordan:.3f}', fontsize=12)

for _, row in df.iterrows():
    ax2.annotate(row['key'].split('_')[0], 
                (row['sparsity_U'] * 100, row['hordan_fix'] * 100),
                fontsize=8, alpha=0.7)

# Bar: Part A, Part B, Hordan Fix by dataset
ax3 = axes[2]
x = np.arange(len(df))
width = 0.25

ax3.bar(x - width, df['part_a_gap'] * 100, width, label='Part A Gap', color='#1f77b4')
ax3.bar(x, df['part_b_improvement'] * 100, width, label='Part B Impr', color='#2ca02c')
ax3.bar(x + width, df['hordan_fix'] * 100, width, label='Hordan Fix', color='#d62728')

ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.set_xlabel('Dataset', fontsize=12)
ax3.set_ylabel('Improvement (pp)', fontsize=12)
ax3.set_title('Part A, Part B, and Hordan Fix', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels([k.split('_')[0] + '\n' + k.split('_')[1] for k in df['key']], 
                    fontsize=8, rotation=0)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hordan_analysis_summary.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/hordan_analysis_summary.png')
plt.close()

# Plot 2: Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(df))
width = 0.2

bars1 = ax.bar(x - 1.5*width, df['sgc_std'] * 100, width, label='SGC+Std', color='#1f77b4')
bars2 = ax.bar(x - 0.5*width, df['restr_std'] * 100, width, label='Restr+Std', color='#ff7f0e')
bars3 = ax.bar(x + 0.5*width, df['restr_rn'] * 100, width, label='Restr+RN', color='#2ca02c')
bars4 = ax.bar(x + 1.5*width, df['restr_logmag'] * 100, width, label='Restr+LogMag', color='#d62728')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Hordan Framework: Method Comparison Across Datasets', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df['key'], rotation=45, ha='right', fontsize=9)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/accuracy_comparison.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/accuracy_comparison.png')
plt.close()

# Plot 3: Heatmap of all metrics
fig, ax = plt.subplots(figsize=(12, 8))

heatmap_cols = ['sparsity_U', 'ref_nodes_U', 'fisher_U', 'nmi_U', 
                'part_a_gap', 'part_b_improvement', 'hordan_fix']
heatmap_data = df[heatmap_cols].copy()

# Normalize for visualization
heatmap_data['sparsity_U'] = heatmap_data['sparsity_U'] * 100
heatmap_data['part_a_gap'] = heatmap_data['part_a_gap'] * 100
heatmap_data['part_b_improvement'] = heatmap_data['part_b_improvement'] * 100
heatmap_data['hordan_fix'] = heatmap_data['hordan_fix'] * 100

heatmap_data.index = df['key']
heatmap_data.columns = ['Sparsity_U%', 'RefNodes_U%', 'Fisher_U', 'NMI_U',
                        'Part_A_pp', 'Part_B_pp', 'Hordan_pp']

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, cbar_kws={'label': 'Value'})
ax.set_title('Hordan Framework: All Metrics Heatmap', fontsize=14)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metrics_heatmap.png', dpi=150, bbox_inches='tight')
print(f'✓ Saved: {OUTPUT_DIR}/metrics_heatmap.png')
plt.close()

# ============================================================================
# Save Full DataFrame
# ============================================================================

df.to_csv(f'{OUTPUT_DIR}/hordan_results_full.csv', index=False)
print(f'✓ Saved: {OUTPUT_DIR}/hordan_results_full.csv')

# ============================================================================
# Final Summary
# ============================================================================

print('\n' + '='*80)
print('FINAL SUMMARY')
print('='*80)

print(f'''
Total dataset-k combinations analyzed: {len(df)}

Key Findings:

1. HORDAN FIX EFFECTIVENESS:
   - Helped (LogMag > RN):     {len(helped)} cases
   - Hurt (LogMag < RN):       {len(hurt)} cases
   - Mean improvement:         {df['hordan_fix'].mean()*100:+.2f}pp

2. CORRELATIONS WITH HORDAN FIX:
   - Fisher_U correlation:     r = {corr_fisher_hordan:+.4f} (p = {p_fisher:.4f})
   - Sparsity_U correlation:   r = {corr_sparsity_hordan:+.4f} (p = {p_sparsity:.4f})
   - RefNodes_U correlation:   r = {corr_refnodes_hordan:+.4f} (p = {p_refnodes:.4f})

3. OVERALL PATTERNS:
   - Part A Gap (basis sensitivity): Mean = {df['part_a_gap'].mean()*100:+.2f}pp
   - Part B Improvement (RowNorm):   Mean = {df['part_b_improvement'].mean()*100:+.2f}pp
   - Best method overall: {"Restr+RN" if df['restr_rn'].mean() > df['restr_logmag'].mean() else "Restr+LogMag"}

4. INTERPRETATION:
''')

if corr_fisher_hordan < -0.2:
    print('   The NEGATIVE correlation between Fisher_U and Hordan Fix suggests')
    print('   that when magnitude IS predictive, adding it explicitly creates REDUNDANCY.')
    print('   This supports the Investigation 5 findings about redundancy.')
elif corr_fisher_hordan > 0.2:
    print('   The POSITIVE correlation between Fisher_U and Hordan Fix suggests')
    print('   that magnitude preservation helps when magnitude carries class info.')
    print('   This supports Hordan\'s theoretical framework.')
else:
    print('   No strong relationship found between magnitude correlation and Hordan Fix.')
    print('   Other factors may dominate (gradient dynamics, optimization landscape).')

print(f'''
All outputs saved to: {OUTPUT_DIR}/
  - hordan_results_table.tex (LaTeX table)
  - hordan_results_full.csv (full data)
  - hordan_analysis_summary.png
  - accuracy_comparison.png
  - metrics_heatmap.png
''')

print('='*80)
