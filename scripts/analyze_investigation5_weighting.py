"""
===================================================================================
INVESTIGATION 5 ANALYSIS SCRIPT: RADIUS-WEIGHTED TRAINING
===================================================================================

Comprehensive analysis of radius-weighted training experiments.

Tests hypothesis:
  - SIMPLE methods (RowNorm) → Radius weighting should HELP
  - COMPLEX methods (Log-Mag, Nested) → Radius weighting should HURT (redundancy)

Generates:
  1. Hypothesis validation tables
  2. Comparison with Investigation 4 baselines
  3. Framework analysis (Part A/B/B.5/B.6/B.7)
  4. Publication-ready visualizations
  5. Per-dataset comprehensive reports

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
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Investigation 5 Analysis')
parser.add_argument('--results_dir', type=str, 
                   default='results/investigation5_radius_weighted',
                   help='Base directory containing results')
parser.add_argument('--split_type', type=str, choices=['fixed', 'random'],
                   default='fixed', help='Which split type to analyze')
args = parser.parse_args()

RESULTS_BASE = args.results_dir
SPLIT_TYPE = args.split_type
OUTPUT_DIR = os.path.join(RESULTS_BASE, f'analysis_{SPLIT_TYPE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style('whitegrid')

print('='*80)
print('INVESTIGATION 5 ANALYSIS: RADIUS-WEIGHTED TRAINING')
print('='*80)
print(f'Split type: {SPLIT_TYPE}')
print(f'Output: {OUTPUT_DIR}')
print('='*80)

# ============================================================================
# LOAD DATA
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
        
        # Look for k10 directory (main experiments at k=10)
        k_dir = dataset_dir / 'k10'
        if not k_dir.exists():
            continue
        
        results_file = k_dir / 'metrics' / 'results.json'
        if not results_file.exists():
            print(f'  ⚠️  No results.json for {dataset_name}')
            continue
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        data[dataset_name] = results
        print(f'  ✓ Loaded: {dataset_name}')
    
    if len(data) == 0:
        print(f'\n❌ ERROR: No results found in {base_dir}')
        print(f'Expected structure: {base_dir}/<dataset>_{split_type}_lcc/k10/metrics/results.json')
        sys.exit(1)
    
    print(f'\n✓ Loaded {len(data)} datasets')
    return data

all_results = load_all_results(RESULTS_BASE, SPLIT_TYPE)

# ============================================================================
# CREATE SUMMARY DATAFRAME
# ============================================================================

print('\n[1/6] Creating summary tables...')

summary_data = []

for dataset, results in all_results.items():
    experiments = results.get('experiments', {})
    framework = results.get('framework_analysis', {})
    
    # Extract accuracies
    row = {
        'Dataset': dataset,
        # Oversmoothing references (k=2)
        'SGC (k=2)': experiments.get('sgc_k2', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'SGC+MLP (k=2)': experiments.get('sgc_mlp_k2', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        # SGC baselines (k=10)
        'SGC Baseline': experiments.get('sgc_baseline', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'SGC+MLP': experiments.get('sgc_mlp', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        # Part A/B references
        'Restricted+Std': experiments.get('restricted_standard_mlp', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        # SIMPLE method baselines
        'RowNorm': experiments.get('rownorm_baseline', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'RowNorm+Radius': experiments.get('rownorm_radius_weighted', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        # COMPLEX method baselines
        'Log-Magnitude': experiments.get('logmag_baseline', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Log-Magnitude+Radius': experiments.get('logmag_radius_weighted', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Nested Spheres': experiments.get('nested_baseline', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        'Nested+Radius': experiments.get('nested_radius_weighted', {}).get('aggregated', {}).get('test_acc_mean', np.nan) * 100,
        # Framework components
        'Oversmoothing SGC': framework.get('oversmoothing_sgc', np.nan),
        'Oversmoothing MLP': framework.get('oversmoothing_mlp', np.nan),
        'Part A': framework.get('part_a', np.nan),
        'Part B': framework.get('part_b', np.nan),
        'Part B.5': framework.get('part_b5', np.nan),
        'Part B.6': framework.get('part_b6', np.nan),
        'Part B.7 (RowNorm)': framework.get('part_b7_rownorm', np.nan),
        'Part B.7 (Log-Mag)': framework.get('part_b7_logmag', np.nan),
        'Part B.7 (Nested)': framework.get('part_b7_nested', np.nan),
        # Gaps
        'Gap (RowNorm)': framework.get('gap_rownorm', np.nan),
        'Gap (RowNorm+Radius)': framework.get('gap_rownorm_radius', np.nan),
        'Gap (Log-Mag)': framework.get('gap_logmag', np.nan),
        'Gap (Log-Mag+Radius)': framework.get('gap_logmag_radius', np.nan),
        'Gap (Nested)': framework.get('gap_nested', np.nan),
        'Gap (Nested+Radius)': framework.get('gap_nested_radius', np.nan),
    }
    
    summary_data.append(row)

df = pd.DataFrame(summary_data)

# Sort by dataset name for consistency
df = df.sort_values('Dataset')

# Save summary CSV
summary_csv = Path(OUTPUT_DIR) / 'summary_table.csv'
df.to_csv(summary_csv, index=False, float_format='%.2f')
print(f'✓ Saved: {summary_csv}')

# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

print('\n[2/6] Testing radius weighting hypothesis...')

hypothesis_results = []

for _, row in df.iterrows():
    dataset = row['Dataset']
    
    # Part B.7 effects
    rownorm_effect = row['Part B.7 (RowNorm)']
    logmag_effect = row['Part B.7 (Log-Mag)']
    nested_effect = row['Part B.7 (Nested)']
    
    # Hypothesis tests
    simple_helps = rownorm_effect > 0.05  # SIMPLE should benefit
    logmag_hurts = logmag_effect < -0.05  # COMPLEX should hurt
    nested_hurts = nested_effect < -0.05
    
    # Overall hypothesis
    hypothesis_confirmed = simple_helps and (logmag_hurts or nested_hurts)
    
    hypothesis_results.append({
        'Dataset': dataset,
        'RowNorm Effect': rownorm_effect,
        'Log-Mag Effect': logmag_effect,
        'Nested Effect': nested_effect,
        'SIMPLE Helps?': '✓' if simple_helps else '✗',
        'Log-Mag Hurts?': '✓' if logmag_hurts else '✗',
        'Nested Hurts?': '✓' if nested_hurts else '✗',
        'Hypothesis': '✓✓' if hypothesis_confirmed else '~' if simple_helps else '✗✗'
    })

hypothesis_df = pd.DataFrame(hypothesis_results)

# Calculate success rates
total_datasets = len(hypothesis_df)
simple_success = (hypothesis_df['SIMPLE Helps?'] == '✓').sum()
logmag_success = (hypothesis_df['Log-Mag Hurts?'] == '✓').sum()
nested_success = (hypothesis_df['Nested Hurts?'] == '✓').sum()
full_hypothesis = (hypothesis_df['Hypothesis'] == '✓✓').sum()

print(f'\nHypothesis Testing Results:')
print(f'  SIMPLE helps: {simple_success}/{total_datasets} ({simple_success/total_datasets*100:.1f}%)')
print(f'  Log-Mag hurts: {logmag_success}/{total_datasets} ({logmag_success/total_datasets*100:.1f}%)')
print(f'  Nested hurts: {nested_success}/{total_datasets} ({nested_success/total_datasets*100:.1f}%)')
print(f'  Full hypothesis: {full_hypothesis}/{total_datasets} ({full_hypothesis/total_datasets*100:.1f}%)')

hypothesis_csv = Path(OUTPUT_DIR) / 'hypothesis_testing.csv'
hypothesis_df.to_csv(hypothesis_csv, index=False, float_format='%.2f')
print(f'✓ Saved: {hypothesis_csv}')

# Save LaTeX table
hypothesis_tex = Path(OUTPUT_DIR) / 'hypothesis_testing.tex'
with open(hypothesis_tex, 'w') as f:
    f.write(hypothesis_df.to_latex(index=False, float_format='%.2f'))
print(f'✓ Saved: {hypothesis_tex}')

# ============================================================================
# FRAMEWORK COMPARISON (Part B.7 vs Part B.6)
# ============================================================================

print('\n[3/6] Comparing with Investigation 4 baselines...')

comparison_data = []

for _, row in df.iterrows():
    comparison_data.append({
        'Dataset': row['Dataset'],
        'Part B (RowNorm)': row['Part B'],
        'Part B.5 (Log-Mag)': row['Part B.5'],
        'Part B.6 (Nested)': row['Part B.6'],
        'Part B.7 (RowNorm+Radius)': row['Part B'] + row['Part B.7 (RowNorm)'],
        'Part B.7 (Log-Mag+Radius)': row['Part B.5'] + row['Part B.7 (Log-Mag)'],
        'Part B.7 (Nested+Radius)': row['Part B.6'] + row['Part B.7 (Nested)'],
        'Radius Effect (RowNorm)': row['Part B.7 (RowNorm)'],
        'Radius Effect (Log-Mag)': row['Part B.7 (Log-Mag)'],
        'Radius Effect (Nested)': row['Part B.7 (Nested)'],
    })

comparison_df = pd.DataFrame(comparison_data)

comparison_csv = Path(OUTPUT_DIR) / 'framework_comparison.csv'
comparison_df.to_csv(comparison_csv, index=False, float_format='%.2f')
print(f'✓ Saved: {comparison_csv}')

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print('\n[4/6] Generating visualizations...')

# Figure 1: Hypothesis validation (side-by-side bars)
fig, ax = plt.subplots(figsize=(14, 7))

datasets = df['Dataset'].values
x = np.arange(len(datasets))
width = 0.25

bars1 = ax.bar(x - width, df['Part B.7 (RowNorm)'].values, width, 
               label='RowNorm (SIMPLE)', alpha=0.8, color='#2ca02c')
bars2 = ax.bar(x, df['Part B.7 (Log-Mag)'].values, width,
               label='Log-Magnitude (COMPLEX)', alpha=0.8, color='#ff7f0e')
bars3 = ax.bar(x + width, df['Part B.7 (Nested)'].values, width,
               label='Nested Spheres (COMPLEX)', alpha=0.8, color='#d62728')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (+0.05pp)')
ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (-0.05pp)')

ax.set_ylabel('Radius Weighting Effect (pp)', fontsize=12)
ax.set_title(f'Investigation 5: Radius Weighting Effect by Method\n{SPLIT_TYPE.capitalize()} Splits', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig1_file = Path(OUTPUT_DIR) / 'hypothesis_validation.png'
plt.savefig(fig1_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig1_file}')
plt.close()

# Figure 2: Method performance heatmap
fig, ax = plt.subplots(figsize=(12, 8))

methods = ['RowNorm', 'RowNorm+Radius', 'Log-Magnitude', 'Log-Magnitude+Radius', 
           'Nested Spheres', 'Nested+Radius']
heatmap_data = df[methods].T

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=70,
            xticklabels=datasets, yticklabels=methods, ax=ax,
            cbar_kws={'label': 'Test Accuracy (%)'})

ax.set_title(f'Investigation 5: Method Performance Heatmap\n{SPLIT_TYPE.capitalize()} Splits',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Method', fontsize=12)

plt.tight_layout()
fig2_file = Path(OUTPUT_DIR) / 'performance_heatmap.png'
plt.savefig(fig2_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig2_file}')
plt.close()

# Figure 3: Framework progression (Part A → B → B.5 → B.6 → B.7)
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axes = axes.flatten()

for idx, (_, row) in enumerate(df.iterrows()):
    if idx >= len(axes):
        break
    
    ax = axes[idx]
    dataset = row['Dataset']
    
    # Framework progression
    stages = ['Part A', 'Part B', 'Part B.5', 'Part B.6', 
              'B.7\n(RowNorm)', 'B.7\n(Log-Mag)', 'B.7\n(Nested)']
    values = [
        row['Part A'],
        row['Part B'],
        row['Part B.5'],
        row['Part B.6'],
        row['Part B.7 (RowNorm)'],
        row['Part B.7 (Log-Mag)'],
        row['Part B.7 (Nested)']
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    bars = ax.bar(range(len(stages)), values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(dataset, fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                   fontsize=7)

# Hide unused subplots
for idx in range(len(df), len(axes)):
    axes[idx].axis('off')

fig.text(0.04, 0.5, 'Effect (pp)', va='center', rotation='vertical', fontsize=12)
fig.suptitle(f'Investigation 5: Framework Progression Across Datasets\n{SPLIT_TYPE.capitalize()} Splits',
             fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0.05, 0, 1, 0.96])
fig3_file = Path(OUTPUT_DIR) / 'framework_progression.png'
plt.savefig(fig3_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig3_file}')
plt.close()

# Figure 4: Oversmoothing analysis
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(datasets))
width = 0.35

ax.bar(x - width/2, df['Oversmoothing SGC'].values, width,
       label='SGC (Logistic)', alpha=0.8, color='#1f77b4')
ax.bar(x + width/2, df['Oversmoothing MLP'].values, width,
       label='SGC+MLP', alpha=0.8, color='#ff7f0e')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Oversmoothing Effect (k=2 → k=10, pp)', fontsize=12)
ax.set_title(f'Investigation 5: Oversmoothing Analysis\n{SPLIT_TYPE.capitalize()} Splits\nNegative = Performance degradation at higher k',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig4_file = Path(OUTPUT_DIR) / 'oversmoothing_analysis.png'
plt.savefig(fig4_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig4_file}')
plt.close()

# ============================================================================
# COMPREHENSIVE REPORT
# ============================================================================

print('\n[5/6] Generating comprehensive report...')

report_file = Path(OUTPUT_DIR) / 'comprehensive_report.txt'

with open(report_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('INVESTIGATION 5: RADIUS-WEIGHTED TRAINING\n')
    f.write('Comprehensive Analysis Report\n')
    f.write('='*80 + '\n\n')
    
    f.write(f'Split Type: {SPLIT_TYPE.capitalize()}\n')
    f.write(f'Datasets Analyzed: {len(df)}\n\n')
    
    f.write('HYPOTHESIS\n')
    f.write('-'*80 + '\n')
    f.write('Radius weighting adds confidence signal based on node centrality.\n')
    f.write('Expected behavior:\n')
    f.write('  1. SIMPLE methods (RowNorm) lack magnitude info → should BENEFIT\n')
    f.write('  2. COMPLEX methods (Log-Mag, Nested) already use magnitude → should HURT (redundancy)\n\n')
    
    f.write('HYPOTHESIS TESTING RESULTS\n')
    f.write('-'*80 + '\n')
    f.write(f'SIMPLE methods benefit: {simple_success}/{total_datasets} ({simple_success/total_datasets*100:.1f}%)\n')
    f.write(f'Log-Magnitude hurts: {logmag_success}/{total_datasets} ({logmag_success/total_datasets*100:.1f}%)\n')
    f.write(f'Nested Spheres hurts: {nested_success}/{total_datasets} ({nested_success/total_datasets*100:.1f}%)\n')
    f.write(f'Full hypothesis confirmed: {full_hypothesis}/{total_datasets} ({full_hypothesis/total_datasets*100:.1f}%)\n\n')
    
    if full_hypothesis / total_datasets >= 0.7:
        f.write('✓✓ HYPOTHESIS STRONGLY SUPPORTED\n')
    elif full_hypothesis / total_datasets >= 0.5:
        f.write('~ HYPOTHESIS PARTIALLY SUPPORTED\n')
    else:
        f.write('✗✗ HYPOTHESIS NOT SUPPORTED\n')
    f.write('\n')
    
    f.write('PER-DATASET ANALYSIS\n')
    f.write('-'*80 + '\n\n')
    
    for _, row in hypothesis_df.iterrows():
        dataset = row['Dataset']
        f.write(f'{dataset.upper()}\n')
        f.write(f'  RowNorm effect: {row["RowNorm Effect"]:+.2f}pp ({row["SIMPLE Helps?"]})\n')
        f.write(f'  Log-Mag effect: {row["Log-Mag Effect"]:+.2f}pp ({row["Log-Mag Hurts?"]})\n')
        f.write(f'  Nested effect:  {row["Nested Effect"]:+.2f}pp ({row["Nested Hurts?"]})\n')
        f.write(f'  Hypothesis: {row["Hypothesis"]}\n\n')
    
    f.write('FRAMEWORK SUMMARY\n')
    f.write('-'*80 + '\n')
    f.write('Part A: Basis sensitivity (SGC+MLP → Restricted+StandardMLP)\n')
    f.write('Part B: RowNorm effect (Restricted+StandardMLP → Restricted+RowNorm)\n')
    f.write('Part B.5: Magnitude preservation (Log-Magnitude augmentation)\n')
    f.write('Part B.6: Eigenvalue structure (Nested Spheres)\n')
    f.write('Part B.7: Radius weighting effect (NEW)\n\n')
    
    f.write('Mean effects across datasets:\n')
    f.write(f'  Part A: {df["Part A"].mean():+.2f}pp ± {df["Part A"].std():.2f}pp\n')
    f.write(f'  Part B: {df["Part B"].mean():+.2f}pp ± {df["Part B"].std():.2f}pp\n')
    f.write(f'  Part B.5: {df["Part B.5"].mean():+.2f}pp ± {df["Part B.5"].std():.2f}pp\n')
    f.write(f'  Part B.6: {df["Part B.6"].mean():+.2f}pp ± {df["Part B.6"].std():.2f}pp\n')
    f.write(f'  Part B.7 (RowNorm): {df["Part B.7 (RowNorm)"].mean():+.2f}pp ± {df["Part B.7 (RowNorm)"].std():.2f}pp\n')
    f.write(f'  Part B.7 (Log-Mag): {df["Part B.7 (Log-Mag)"].mean():+.2f}pp ± {df["Part B.7 (Log-Mag)"].std():.2f}pp\n')
    f.write(f'  Part B.7 (Nested): {df["Part B.7 (Nested)"].mean():+.2f}pp ± {df["Part B.7 (Nested)"].std():.2f}pp\n\n')
    
    f.write('OVERSMOOTHING ANALYSIS\n')
    f.write('-'*80 + '\n')
    f.write('Comparing k=2 vs k=10 performance:\n\n')
    
    severe_oversmoothing = ((df['Oversmoothing SGC'] < -5) | (df['Oversmoothing MLP'] < -5)).sum()
    f.write(f'Datasets with severe oversmoothing (>5pp degradation): {severe_oversmoothing}/{total_datasets}\n')
    f.write(f'Mean SGC oversmoothing: {df["Oversmoothing SGC"].mean():+.2f}pp\n')
    f.write(f'Mean MLP oversmoothing: {df["Oversmoothing MLP"].mean():+.2f}pp\n\n')
    
    f.write('CONCLUSION\n')
    f.write('-'*80 + '\n')
    f.write('Radius weighting represents a confidence-based training approach that:\n')
    if simple_success / total_datasets >= 0.7:
        f.write('  ✓ Consistently helps SIMPLE methods (no magnitude feature)\n')
    else:
        f.write('  ~ Inconsistently helps SIMPLE methods\n')
    
    if (logmag_success + nested_success) / (2 * total_datasets) >= 0.5:
        f.write('  ✓ Often hurts COMPLEX methods (redundancy confirmed)\n')
    else:
        f.write('  ✗ Does not consistently hurt COMPLEX methods (redundancy not confirmed)\n')
    
    f.write('\n')
    f.write('='*80 + '\n')

print(f'✓ Saved: {report_file}')

# ============================================================================
# PER-DATASET DETAILED REPORTS
# ============================================================================

print('\n[6/6] Generating per-dataset reports...')

per_dataset_dir = Path(OUTPUT_DIR) / 'per_dataset'
per_dataset_dir.mkdir(exist_ok=True)

for dataset, results in all_results.items():
    dataset_file = per_dataset_dir / f'{dataset}_report.txt'
    
    experiments = results.get('experiments', {})
    framework = results.get('framework_analysis', {})
    
    with open(dataset_file, 'w') as f:
        f.write('='*80 + '\n')
        f.write(f'{dataset.upper()}\n')
        f.write('Investigation 5: Radius-Weighted Training\n')
        f.write('='*80 + '\n\n')
        
        f.write('EXPERIMENT RESULTS\n')
        f.write('-'*80 + '\n\n')
        
        exp_names = {
            'sgc_k2': 'SGC Baseline (k=2)',
            'sgc_mlp_k2': 'SGC+MLP (k=2)',
            'sgc_baseline': 'SGC Baseline (k=10)',
            'sgc_mlp': 'SGC+MLP (k=10)',
            'restricted_standard_mlp': 'Restricted + StandardMLP',
            'rownorm_baseline': 'Restricted + RowNorm',
            'rownorm_radius_weighted': 'Restricted + RowNorm + Radius',
            'logmag_baseline': 'Restricted + Log-Magnitude',
            'logmag_radius_weighted': 'Restricted + Log-Magnitude + Radius',
            'nested_baseline': 'Restricted + Nested Spheres',
            'nested_radius_weighted': 'Restricted + Nested Spheres + Radius',
        }
        
        for exp_key, exp_name in exp_names.items():
            if exp_key in experiments:
                agg = experiments[exp_key].get('aggregated', {})
                metadata = experiments[exp_key].get('metadata', {})
                
                acc_mean = agg.get('test_acc_mean', np.nan) * 100
                acc_std = agg.get('test_acc_std', np.nan) * 100
                
                f.write(f'{exp_name}\n')
                f.write(f'  Test Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%\n')
                
                if 'd_effective' in metadata:
                    f.write(f'  Effective dimension: {metadata["d_effective"]}\n')
                if 'ortho_error' in metadata:
                    f.write(f'  D-orthonormality error: {metadata["ortho_error"]:.2e}\n')
                if 'radius_info' in metadata:
                    r_info = metadata['radius_info']
                    f.write(f'  Radius: μ={r_info["mean"]:.4f}, σ={r_info["std"]:.4f}\n')
                if 'weight_info' in metadata:
                    w_info = metadata['weight_info']
                    f.write(f'  Weight distribution: μ={w_info["mean"]:.4f}, σ={w_info["std"]:.4f}\n')
                
                f.write('\n')
        
        f.write('FRAMEWORK ANALYSIS\n')
        f.write('-'*80 + '\n\n')
        
        f.write(f'Oversmoothing SGC: {framework.get("oversmoothing_sgc", np.nan):+.2f}pp (k=2 → k=10)\n')
        f.write(f'Oversmoothing MLP: {framework.get("oversmoothing_mlp", np.nan):+.2f}pp (k=2 → k=10)\n\n')
        
        f.write(f'Part A (Basis Sensitivity): {framework.get("part_a", np.nan):+.2f}pp\n')
        f.write(f'Part B (RowNorm): {framework.get("part_b", np.nan):+.2f}pp\n')
        f.write(f'Part B.5 (Log-Magnitude): {framework.get("part_b5", np.nan):+.2f}pp\n')
        f.write(f'Part B.6 (Nested Spheres): {framework.get("part_b6", np.nan):+.2f}pp\n\n')
        
        f.write('Part B.7 (Radius Weighting Effects):\n')
        rownorm_effect = framework.get("part_b7_rownorm", np.nan)
        logmag_effect = framework.get("part_b7_logmag", np.nan)
        nested_effect = framework.get("part_b7_nested", np.nan)
        
        f.write(f'  RowNorm: {rownorm_effect:+.2f}pp ')
        if rownorm_effect > 0.1:
            f.write('(✓ HELPS - adds new info)\n')
        elif rownorm_effect < -0.1:
            f.write('(✗ HURTS unexpectedly)\n')
        else:
            f.write('(~ No effect)\n')
        
        f.write(f'  Log-Magnitude: {logmag_effect:+.2f}pp ')
        if logmag_effect < -0.05:
            f.write('(✓ HURTS - redundancy confirmed)\n')
        elif logmag_effect > 0.1:
            f.write('(✗ HELPS - contradicts hypothesis)\n')
        else:
            f.write('(~ No effect)\n')
        
        f.write(f'  Nested Spheres: {nested_effect:+.2f}pp ')
        if nested_effect < -0.05:
            f.write('(✓ HURTS - redundancy confirmed)\n')
        elif nested_effect > 0.1:
            f.write('(✗ HELPS - contradicts hypothesis)\n')
        else:
            f.write('(~ No effect)\n')
        
        f.write('\n')
        f.write('='*80 + '\n')
    
    print(f'  ✓ {dataset}')

print(f'\n✓ Saved {len(all_results)} per-dataset reports in {per_dataset_dir}')

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
print(f'  Hypothesis testing: {hypothesis_csv}')
print(f'  Hypothesis testing (LaTeX): {hypothesis_tex}')
print(f'  Framework comparison: {comparison_csv}')
print(f'  Figure 1 (Hypothesis validation): {fig1_file}')
print(f'  Figure 2 (Performance heatmap): {fig2_file}')
print(f'  Figure 3 (Framework progression): {fig3_file}')
print(f'  Figure 4 (Oversmoothing): {fig4_file}')
print(f'  Comprehensive report: {report_file}')
print(f'  Per-dataset reports: {per_dataset_dir}/')
print()
print('='*80)
print('KEY FINDINGS')
print('='*80)
print(f'✓ SIMPLE methods benefit: {simple_success}/{total_datasets} ({simple_success/total_datasets*100:.1f}%)')
print(f'✓ COMPLEX methods hurt: {(logmag_success + nested_success)/(2*total_datasets)*100:.1f}% average')
print(f'✓ Full hypothesis: {full_hypothesis}/{total_datasets} ({full_hypothesis/total_datasets*100:.1f}%)')

if full_hypothesis / total_datasets >= 0.7:
    print('\n→ HYPOTHESIS STRONGLY SUPPORTED')
    print('   Radius weighting helps when magnitude is NOT already a feature')
    print('   Redundancy confirmed for methods that already use magnitude')
elif full_hypothesis / total_datasets >= 0.5:
    print('\n→ HYPOTHESIS PARTIALLY SUPPORTED')
    print('   Mixed results suggest dataset-dependent behavior')
else:
    print('\n→ HYPOTHESIS NOT SUPPORTED')
    print('   Radius weighting shows inconsistent pattern')

print()
print('Next steps: Review comprehensive report and per-dataset analyses')
print('='*80)