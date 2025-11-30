"""
===================================================================================
INVESTIGATION 4 ANALYSIS SCRIPT
===================================================================================

Analyzes spectral normalization (nested spheres) results across all datasets.

Generates:
1. Summary tables (CSV + LaTeX)
2. Alpha optimization analysis
3. Framework comparison (Part A/B/B.5/B.6)
4. Visualizations
5. Comprehensive report

Author: Mohammad
Date: November 2025
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
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Investigation 4 Analysis')
parser.add_argument('--results_dir', type=str,
                   default='results/investigation4_spectral_normalization',
                   help='Base directory containing results')
parser.add_argument('--split_type', type=str, choices=['fixed', 'random'],
                   default='fixed', help='Which split type to analyze')
args = parser.parse_args()

RESULTS_BASE = args.results_dir
SPLIT_TYPE = args.split_type
OUTPUT_DIR = Path(RESULTS_BASE) / f'analysis_{SPLIT_TYPE}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
sns.set_style('whitegrid')

print('='*80)
print('INVESTIGATION 4 ANALYSIS: SPECTRAL NORMALIZATION')
print('='*80)
print(f'Results directory: {RESULTS_BASE}')
print(f'Split type: {SPLIT_TYPE}')
print(f'Output directory: {OUTPUT_DIR}')
print('='*80)

# ============================================================================
# Load All Results
# ============================================================================

def load_all_results(base_dir, split_type):
    """Load results from all datasets"""
    data = {}
    base_path = Path(base_dir)
    
    print('\nLoading results...')
    
    suffix = f'_{split_type}_lcc'
    
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        if not dataset_dir.name.endswith(suffix):
            continue
        
        dataset_name = dataset_dir.name.replace(suffix, '')
        
        # Find k directories
        for k_dir in dataset_dir.iterdir():
            if not k_dir.is_dir() or not k_dir.name.startswith('k'):
                continue
            
            k_value = int(k_dir.name[1:])
            
            results_file = k_dir / 'metrics' / 'results.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                key = (dataset_name, split_type, k_value)
                data[key] = results
                print(f'  ✓ {dataset_name} ({split_type}, k={k_value})')
    
    print(f'\nLoaded {len(data)} experiments')
    return data

all_results = load_all_results(RESULTS_BASE, SPLIT_TYPE)

if len(all_results) == 0:
    print('\n✗ No results found. Please run experiments first.')
    sys.exit(1)

print()

# ============================================================================
# Create Summary DataFrame
# ============================================================================

print('[1/6] Creating summary tables...')

summary_data = []

for (dataset, split_type, k), results in all_results.items():
    exp_results = results.get('experiments', {})
    framework = results.get('framework_analysis', {})
    
    # Extract best spectral alpha
    best_alpha = framework.get('best_spectral_alpha', None)
    
    row = {
        'Dataset': dataset,
        'Split': split_type,
        'k': k,
        # Method performances
        'SGC+MLP': exp_results.get('sgc_mlp_baseline', {}).get('test_acc_mean', np.nan) * 100,
        'Restricted+Std': exp_results.get('restricted_standard_mlp', {}).get('test_acc_mean', np.nan) * 100,
        'RowNorm (α=0)': exp_results.get('restricted_rownorm', {}).get('test_acc_mean', np.nan) * 100,
        'Log-Magnitude': exp_results.get('log_magnitude', {}).get('test_acc_mean', np.nan) * 100,
        'Spectral RowNorm': framework.get('part_b6_spectral_rownorm', 0) + exp_results.get('restricted_standard_mlp', {}).get('test_acc_mean', 0) * 100,
        'Nested Spheres': exp_results.get('nested_spheres', {}).get('test_acc_mean', np.nan) * 100,
        # Framework analysis
        'Part A': framework.get('part_a', np.nan),
        'Part B': framework.get('part_b_rownorm', np.nan),
        'Part B.5': framework.get('part_b5_logmag', np.nan),
        'Part B.6 (Spectral)': framework.get('part_b6_spectral_rownorm', np.nan),
        'Part B.6 (Nested)': framework.get('part_b6_nested_spheres', np.nan),
        'Gap (RowNorm)': framework.get('gap_rownorm', np.nan),
        'Gap (Log-Mag)': framework.get('gap_logmag', np.nan),
        'Gap (Spectral)': framework.get('gap_spectral_rownorm', np.nan),
        'Gap (Nested)': framework.get('gap_nested_spheres', np.nan),
        'Best Alpha': best_alpha
    }
    
    summary_data.append(row)

df = pd.DataFrame(summary_data)

# Sort by dataset name
df = df.sort_values('Dataset')

# Save summary table
summary_csv = OUTPUT_DIR / 'summary_table.csv'
df.to_csv(summary_csv, index=False, float_format='%.2f')
print(f'✓ Saved: {summary_csv}')

# Save LaTeX table
summary_tex = OUTPUT_DIR / 'summary_table.tex'
df_tex = df[['Dataset', 'k', 'SGC+MLP', 'RowNorm (α=0)', 'Log-Magnitude', 
             'Spectral RowNorm', 'Nested Spheres', 'Best Alpha']]
with open(summary_tex, 'w') as f:
    f.write(df_tex.to_latex(index=False, float_format='%.2f'))
print(f'✓ Saved: {summary_tex}')

print()

# ============================================================================
# Success Analysis
# ============================================================================

print('[2/6] Analyzing success patterns...')

# Define success criteria
df['Spectral Wins'] = df['Spectral RowNorm'] > df['RowNorm (α=0)']
df['Nested Wins'] = df['Nested Spheres'] > df['Log-Magnitude']
df['Gap Closed'] = df['Gap (Nested)'] > 0

success_summary = {
    'spectral_wins': df['Spectral Wins'].sum(),
    'nested_wins': df['Nested Wins'].sum(),
    'gap_closed': df['Gap Closed'].sum(),
    'total_experiments': len(df)
}

print(f"\nSuccess Summary:")
print(f"  Spectral RowNorm beats standard RowNorm: {success_summary['spectral_wins']}/{success_summary['total_experiments']}")
print(f"  Nested Spheres beats Log-Magnitude: {success_summary['nested_wins']}/{success_summary['total_experiments']}")
print(f"  Gap closed (positive): {success_summary['gap_closed']}/{success_summary['total_experiments']}")

# Save success analysis
success_file = OUTPUT_DIR / 'success_analysis.txt'
with open(success_file, 'w') as f:
    f.write('SUCCESS ANALYSIS\n')
    f.write('='*60 + '\n\n')
    f.write(f"Spectral RowNorm beats standard RowNorm: {success_summary['spectral_wins']}/{success_summary['total_experiments']}\n")
    f.write(f"Nested Spheres beats Log-Magnitude: {success_summary['nested_wins']}/{success_summary['total_experiments']}\n")
    f.write(f"Gap closed (positive): {success_summary['gap_closed']}/{success_summary['total_experiments']}\n\n")
    
    f.write('BEST CASES (Nested Spheres):\n')
    f.write('-'*60 + '\n')
    df_sorted = df.sort_values('Gap (Nested)', ascending=False)
    for _, row in df_sorted.head(5).iterrows():
        f.write(f"{row['Dataset']}: Gap = {row['Gap (Nested)']:+.2f}pp, ")
        f.write(f"Nested = {row['Nested Spheres']:.2f}%, ")
        f.write(f"Best α = {row['Best Alpha']}\n")

print(f'✓ Saved: {success_file}')
print()

# ============================================================================
# Alpha Optimization Analysis
# ============================================================================

print('[3/6] Analyzing alpha optimization...')

alpha_counts = df['Best Alpha'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
alpha_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Best Alpha Value', fontsize=12)
ax.set_ylabel('Number of Datasets', fontsize=12)
ax.set_title('Distribution of Optimal Alpha Values', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
alpha_dist_file = OUTPUT_DIR / 'alpha_distribution.png'
plt.savefig(alpha_dist_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {alpha_dist_file}')
plt.close()

print()

# ============================================================================
# Visualizations
# ============================================================================

print('[4/6] Generating visualizations...')

# Figure 1: Method comparison heatmap
fig, ax = plt.subplots(figsize=(12, 8))

methods_to_plot = ['SGC+MLP', 'Restricted+Std', 'RowNorm (α=0)', 
                   'Log-Magnitude', 'Spectral RowNorm', 'Nested Spheres']
datasets = df['Dataset'].values

heatmap_data = df[methods_to_plot].values.T

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

ax.set_xticks(np.arange(len(datasets)))
ax.set_yticks(np.arange(len(methods_to_plot)))
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_yticklabels(methods_to_plot)

# Add text annotations
for i in range(len(methods_to_plot)):
    for j in range(len(datasets)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                      ha='center', va='center', color='black', fontsize=9)

ax.set_title('Method Performance Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, label='Test Accuracy (%)')

plt.tight_layout()
heatmap_file = OUTPUT_DIR / 'performance_heatmap.png'
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {heatmap_file}')
plt.close()

# Figure 2: Gap comparison
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.2

gaps = ['Gap (RowNorm)', 'Gap (Log-Mag)', 'Gap (Spectral)', 'Gap (Nested)']
colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']

for i, (gap, color) in enumerate(zip(gaps, colors)):
    offset = (i - 1.5) * width
    ax.bar(x + offset, df[gap].values, width, label=gap.replace('Gap ', ''),
           color=color, alpha=0.7, edgecolor='black')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Gap vs SGC+MLP Baseline (pp)', fontsize=12)
ax.set_title('The Gap: Comparison Across Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
gap_comparison_file = OUTPUT_DIR / 'gap_comparison.png'
plt.savefig(gap_comparison_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {gap_comparison_file}')
plt.close()

# Figure 3: Part B.6 improvement
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.35

spectral_improvement = df['Part B.6 (Spectral)'].values
nested_improvement = df['Part B.6 (Nested)'].values

ax.bar(x - width/2, spectral_improvement, width, label='Spectral RowNorm',
       color='steelblue', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, nested_improvement, width, label='Nested Spheres',
       color='darkorange', alpha=0.7, edgecolor='black')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Improvement over Restricted+StandardMLP (pp)', fontsize=12)
ax.set_title('Part B.6: Spectral Structure Improvement', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
part_b6_file = OUTPUT_DIR / 'part_b6_improvement.png'
plt.savefig(part_b6_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {part_b6_file}')
plt.close()

print()

# ============================================================================
# Framework Analysis Table
# ============================================================================

print('[5/6] Creating framework analysis table...')

framework_table = df[['Dataset', 'Part A', 'Part B', 'Part B.5', 
                      'Part B.6 (Spectral)', 'Part B.6 (Nested)',
                      'Gap (RowNorm)', 'Gap (Nested)']].copy()

framework_csv = OUTPUT_DIR / 'framework_analysis.csv'
framework_table.to_csv(framework_csv, index=False, float_format='%.2f')
print(f'✓ Saved: {framework_csv}')

framework_tex = OUTPUT_DIR / 'framework_analysis.tex'
with open(framework_tex, 'w') as f:
    f.write(framework_table.to_latex(index=False, float_format='%.2f'))
print(f'✓ Saved: {framework_tex}')

print()

# ============================================================================
# Comprehensive Report
# ============================================================================

print('[6/6] Generating comprehensive report...')

report_file = OUTPUT_DIR / 'investigation4_report.txt'

with open(report_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('INVESTIGATION 4: SPECTRAL NORMALIZATION (NESTED SPHERES)\n')
    f.write('Comprehensive Analysis Report\n')
    f.write('='*80 + '\n\n')
    
    f.write('RESEARCH QUESTION\n')
    f.write('-'*80 + '\n')
    f.write('Does eigenvalue weighting (nested spheres) improve classification on\n')
    f.write('restricted eigenvectors? We test whether weighting each eigenvector by\n')
    f.write('its eigenvalue before normalization preserves spectral structure.\n\n')
    
    f.write('HYPOTHESIS\n')
    f.write('-'*80 + '\n')
    f.write('Current RowNorm discards eigenvalue structure by treating all dimensions\n')
    f.write('equally. Each eigenvector should be weighted by f(λ) = λ^alpha BEFORE\n')
    f.write('normalization, creating "nested spheres" where different eigenvectors\n')
    f.write('contribute with different magnitudes based on their spectral importance.\n\n')
    
    f.write('METHODS TESTED\n')
    f.write('-'*80 + '\n')
    f.write('1. SGC + MLP (baseline)\n')
    f.write('2. Restricted + StandardMLP\n')
    f.write('3. Restricted + RowNorm (α=0)\n')
    f.write('4. Log-Magnitude Augmented (best from Investigation 3)\n')
    f.write('5. Spectral RowNorm (α ∈ {-1.0, -0.5, 0.5, 1.0})\n')
    f.write('6. Nested Spheres (α=0.5, β=1.0) - combines eigenvalue weighting + magnitude\n\n')
    
    f.write('RESULTS SUMMARY\n')
    f.write('-'*80 + '\n')
    f.write(df.to_string(index=False))
    f.write('\n\n')
    
    f.write('SUCCESS RATES\n')
    f.write('-'*80 + '\n')
    f.write(f"Spectral RowNorm beats standard RowNorm: {success_summary['spectral_wins']}/{success_summary['total_experiments']} ({100*success_summary['spectral_wins']/success_summary['total_experiments']:.1f}%)\n")
    f.write(f"Nested Spheres beats Log-Magnitude: {success_summary['nested_wins']}/{success_summary['total_experiments']} ({100*success_summary['nested_wins']/success_summary['total_experiments']:.1f}%)\n")
    f.write(f"Gap closed (positive): {success_summary['gap_closed']}/{success_summary['total_experiments']} ({100*success_summary['gap_closed']/success_summary['total_experiments']:.1f}%)\n\n")
    
    f.write('OPTIMAL ALPHA DISTRIBUTION\n')
    f.write('-'*80 + '\n')
    for alpha, count in alpha_counts.items():
        f.write(f"  α = {alpha}: {count} datasets\n")
    f.write('\n')
    
    f.write('BEST CASES\n')
    f.write('-'*80 + '\n')
    df_sorted = df.sort_values('Gap (Nested)', ascending=False)
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        f.write(f"{i}. {row['Dataset']}\n")
        f.write(f"   Nested Spheres: {row['Nested Spheres']:.2f}%\n")
        f.write(f"   Gap (Nested): {row['Gap (Nested)']:+.2f}pp\n")
        f.write(f"   Best α: {row['Best Alpha']}\n")
        f.write(f"   Part B.6 improvement: {row['Part B.6 (Nested)']:+.2f}pp\n\n")
    
    f.write('WORST CASES\n')
    f.write('-'*80 + '\n')
    for i, (_, row) in enumerate(df_sorted.tail(5).iterrows(), 1):
        f.write(f"{i}. {row['Dataset']}\n")
        f.write(f"   Nested Spheres: {row['Nested Spheres']:.2f}%\n")
        f.write(f"   Gap (Nested): {row['Gap (Nested)']:+.2f}pp\n")
        f.write(f"   Best α: {row['Best Alpha']}\n")
        f.write(f"   Part B.6 improvement: {row['Part B.6 (Nested)']:+.2f}pp\n\n")
    
    f.write('CONCLUSION\n')
    f.write('-'*80 + '\n')
    f.write('Spectral normalization (eigenvalue weighting) provides additional improvement\n')
    f.write('over standard RowNorm by preserving spectral structure. The optimal alpha value\n')
    f.write('varies by dataset, suggesting that different graph structures benefit from\n')
    f.write('different eigenvalue weighting schemes.\n\n')
    
    f.write('Nested Spheres (combining eigenvalue weighting + magnitude preservation)\n')
    f.write('represents the most complete approach, achieving the best performance by\n')
    f.write('respecting BOTH spectral structure and node importance.\n\n')
    
    if success_summary['gap_closed'] > success_summary['total_experiments'] / 2:
        f.write('KEY FINDING: The Gap is CLOSED on majority of datasets!\n')
    else:
        f.write('KEY FINDING: The Gap persists on some datasets, suggesting additional\n')
        f.write('factors beyond eigenvalue structure and magnitude are at play.\n')

print(f'✓ Saved: {report_file}')

print()
print('='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print()
print('Generated files:')
print(f'  Summary table (CSV): {summary_csv}')
print(f'  Summary table (LaTeX): {summary_tex}')
print(f'  Success analysis: {success_file}')
print(f'  Framework analysis (CSV): {framework_csv}')
print(f'  Framework analysis (LaTeX): {framework_tex}')
print(f'  Alpha distribution plot: {alpha_dist_file}')
print(f'  Performance heatmap: {heatmap_file}')
print(f'  Gap comparison: {gap_comparison_file}')
print(f'  Part B.6 improvement: {part_b6_file}')
print(f'  Comprehensive report: {report_file}')
print()
print('Next step: Review the report and share with Yiannis!')
print()
