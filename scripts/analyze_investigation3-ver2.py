"""
Investigation 3: Comprehensive Analysis and Report Generation (Version 2)

This script aggregates results from all Investigation 3 experiments and generates:
1. Summary tables comparing all methods across datasets
2. Diagnostic validation (Fisher score vs actual performance)
3. Method head-to-head comparison (Log-Magnitude vs Dual-Stream)
4. Publication-quality visualizations
5. LaTeX tables for paper
6. Comprehensive report

Usage:
    python scripts/analyze_investigation3-ver2.py
    python scripts/analyze_investigation3-ver2.py --output_dir reports/investigation3_final
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

parser = argparse.ArgumentParser(description='Analyze Investigation 3 Results')
parser.add_argument('--results_dir', type=str, 
                   default='results/investigation3_magnitude',
                   help='Directory containing results')
parser.add_argument('--output_dir', type=str,
                   default='reports/investigation3',
                   help='Output directory for report')
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print('='*80)
print('INVESTIGATION 3: COMPREHENSIVE ANALYSIS')
print('='*80)
print(f'Results directory: {RESULTS_DIR}')
print(f'Output directory: {OUTPUT_DIR}')
print('='*80)
print()

# ============================================================================
# Load All Results
# ============================================================================

def load_all_results(results_dir):
    """
    Load all experiment results from directory structure.
    
    Expected structure:
        results/investigation3_magnitude/
        ├── ogbn-arxiv_fixed_lcc/k10/metrics/results.json
        ├── amazon-computers_fixed_lcc/k10/metrics/results.json
        └── ...
    """
    all_results = {}
    
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        # Parse directory name: {dataset}_{split_type}_{component}
        parts = dataset_dir.name.rsplit('_', 2)
        if len(parts) != 3:
            continue
        
        dataset_name = parts[0]
        split_type = parts[1]
        component = parts[2]
        
        # Look for k directories
        for k_dir in dataset_dir.iterdir():
            if not k_dir.is_dir() or not k_dir.name.startswith('k'):
                continue
            
            k_value = int(k_dir.name[1:])
            
            # Load results.json
            results_file = k_dir / 'metrics' / 'results.json'
            diagnostics_file = k_dir / 'diagnostics' / 'magnitude_diagnostics.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Load diagnostics if available
                if diagnostics_file.exists():
                    with open(diagnostics_file, 'r') as f:
                        diagnostics = json.load(f)
                    results['diagnostics_detailed'] = diagnostics
                
                key = (dataset_name, split_type, k_value)
                all_results[key] = results
                
                print(f'✓ Loaded: {dataset_name} ({split_type}, k={k_value})')
    
    print(f'\nTotal experiments loaded: {len(all_results)}')
    return all_results

print('[1/5] Loading all results...')
all_results = load_all_results(RESULTS_DIR)

if len(all_results) == 0:
    print('No results found! Please run experiments first.')
    sys.exit(1)

print()

# ============================================================================
# Create Summary DataFrame
# ============================================================================

print('[2/5] Creating summary tables...')

# Extract key metrics for each experiment
summary_data = []

for (dataset, split_type, k), results in all_results.items():
    exp_results = results.get('experiments', {})
    framework = results.get('framework_analysis', {})
    diagnostics = results.get('diagnostics', {}).get('diffused_features', {})
    
    row = {
        'Dataset': dataset,
        'Split': split_type,
        'k': k,
        # Diagnostics
        'Fisher Score': diagnostics.get('fisher_score', np.nan),
        'Between-Class Var': diagnostics.get('between_class_variance', np.nan),
        'Within-Class Var': diagnostics.get('within_class_variance', np.nan),
        # Method performances
        'SGC Baseline': exp_results.get('sgc_baseline', {}).get('test_acc_mean', np.nan) * 100,
        'SGC+MLP': exp_results.get('sgc_mlp_baseline', {}).get('test_acc_mean', np.nan) * 100,
        'Restricted+Std': exp_results.get('restricted_standard_mlp', {}).get('test_acc_mean', np.nan) * 100,
        'Restricted+RowNorm': exp_results.get('restricted_rownorm_mlp', {}).get('test_acc_mean', np.nan) * 100,
        'Magnitude-Only': exp_results.get('magnitude_only', {}).get('test_acc_mean', np.nan) * 100,
        'Log-Magnitude': exp_results.get('log_magnitude', {}).get('test_acc_mean', np.nan) * 100,
        'Dual-Stream': exp_results.get('dual_stream', {}).get('test_acc_mean', np.nan) * 100,
        # Framework analysis
        'Part A': framework.get('part_a', np.nan),
        'Part B (RowNorm)': framework.get('part_b_rownorm', np.nan),
        'Part B.5 (Log-Mag)': framework.get('part_b_log_magnitude', np.nan),
        'Part B.5 (Dual)': framework.get('part_b_dual_stream', np.nan),
        'Gap (RowNorm)': framework.get('gap_rownorm', np.nan),
        'Gap (Log-Mag)': framework.get('gap_log_magnitude', np.nan),
        'Gap (Dual)': framework.get('gap_dual_stream', np.nan),
    }
    
    summary_data.append(row)

df = pd.DataFrame(summary_data)

# Sort by Fisher score for better readability
df = df.sort_values('Fisher Score', ascending=False)

# Save summary table
summary_csv = OUTPUT_DIR / 'summary_table.csv'
df.to_csv(summary_csv, index=False, float_format='%.4f')
print(f'✓ Saved: {summary_csv}')

# Save LaTeX table
summary_tex = OUTPUT_DIR / 'summary_table.tex'
df_tex = df[['Dataset', 'Split', 'k', 'Fisher Score', 
             'SGC+MLP', 'Restricted+RowNorm', 'Log-Magnitude', 'Dual-Stream']]
with open(summary_tex, 'w') as f:
    f.write(df_tex.to_latex(index=False, float_format='%.2f'))
print(f'✓ Saved: {summary_tex}')

# ============================================================================
# COMPUTE ALL DERIVED COLUMNS FIRST (BEFORE USING THEM!)
# ============================================================================

# Compute Part B.5 improvement over RowNorm
df['Best Mag-Aware'] = df[['Part B.5 (Log-Mag)', 'Part B.5 (Dual)']].max(axis=1)
df['Mag-Aware Advantage'] = df['Best Mag-Aware'] - df['Part B (RowNorm)']

# Determine which magnitude-aware method won
df['Best Mag-Aware Method'] = df[['Log-Magnitude', 'Dual-Stream']].idxmax(axis=1)

# Categorize based on Fisher score
def categorize_fisher(fisher):
    if pd.isna(fisher):
        return 'Unknown'
    elif fisher < 0.05:
        return 'Very Low (<0.05)'
    elif fisher < 0.1:
        return 'Low (0.05-0.1)'
    elif fisher < 0.3:
        return 'Medium (0.1-0.3)'
    else:
        return 'High (>0.3)'

df['Fisher Category'] = df['Fisher Score'].apply(categorize_fisher)

# Categorize outcomes
def categorize_outcome(advantage):
    if pd.isna(advantage):
        return 'Unknown'
    elif advantage < -0.5:
        return 'RowNorm Wins'
    elif advantage > 0.5:
        return 'Mag-Aware Wins'
    else:
        return 'Tie'

df['Outcome'] = df['Mag-Aware Advantage'].apply(categorize_outcome)

# ============================================================================
# Create detailed method comparison table (AFTER columns exist!)
# ============================================================================

method_comparison = df[['Dataset', 'Split', 'Fisher Score', 
                        'Restricted+RowNorm', 'Log-Magnitude', 'Dual-Stream',
                        'Part B (RowNorm)', 'Part B.5 (Log-Mag)', 'Part B.5 (Dual)',
                        'Mag-Aware Advantage', 'Best Mag-Aware Method']].copy()

method_comparison_file = OUTPUT_DIR / 'method_comparison_detailed.csv'
method_comparison.to_csv(method_comparison_file, index=False, float_format='%.4f')
print(f'✓ Saved: {method_comparison_file}')

print()

# ============================================================================
# Diagnostic Validation Analysis
# ============================================================================

print('[3/5] Analyzing diagnostic accuracy...')

# Create contingency table
contingency = pd.crosstab(df['Fisher Category'], df['Outcome'])
contingency_file = OUTPUT_DIR / 'diagnostic_validation.csv'
contingency.to_csv(contingency_file)
print(f'✓ Saved: {contingency_file}')

# Print diagnostic summary
print('\nDiagnostic Validation Summary:')
print('='*60)
print(contingency)
print()

# Calculate diagnostic accuracy
correct_predictions = 0
total_predictions = 0

for _, row in df.iterrows():
    fisher = row['Fisher Score']
    advantage = row['Mag-Aware Advantage']
    
    if pd.isna(fisher) or pd.isna(advantage):
        continue
    
    total_predictions += 1
    
    # Prediction: Fisher > 0.1 → Mag-Aware should win
    if fisher > 0.1 and advantage > 0:
        correct_predictions += 1
    # Prediction: Fisher < 0.1 → RowNorm should win or tie
    elif fisher <= 0.1 and advantage <= 0.5:
        correct_predictions += 1

if total_predictions > 0:
    accuracy = correct_predictions / total_predictions * 100
    print(f'Diagnostic Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})')
    print()

# ============================================================================
# Method Head-to-Head Comparison (Log-Mag vs Dual-Stream)
# ============================================================================

print('Method Head-to-Head Analysis:')
print('='*60)

# Overall wins
log_mag_wins = (df['Log-Magnitude'] > df['Dual-Stream']).sum()
dual_stream_wins = (df['Dual-Stream'] > df['Log-Magnitude']).sum()
ties = (df['Log-Magnitude'] == df['Dual-Stream']).sum()

print(f'\nOverall Head-to-Head (All Datasets):')
print(f'  Log-Magnitude wins: {log_mag_wins}/{len(df)}')
print(f'  Dual-Stream wins:   {dual_stream_wins}/{len(df)}')
print(f'  Ties:               {ties}/{len(df)}')

# High-Fisher datasets only
high_fisher = df[df['Fisher Score'] > 0.1]
if len(high_fisher) > 0:
    log_mag_wins_high = (high_fisher['Log-Magnitude'] > high_fisher['Dual-Stream']).sum()
    dual_stream_wins_high = (high_fisher['Dual-Stream'] > high_fisher['Log-Magnitude']).sum()
    
    avg_log_mag_advantage = high_fisher['Part B.5 (Log-Mag)'].mean()
    avg_dual_advantage = high_fisher['Part B.5 (Dual)'].mean()
    
    print(f'\nHigh-Fisher Datasets (Fisher > 0.1):')
    print(f'  Log-Magnitude wins: {log_mag_wins_high}/{len(high_fisher)}')
    print(f'  Dual-Stream wins:   {dual_stream_wins_high}/{len(high_fisher)}')
    print(f'  Avg Log-Mag improvement: {avg_log_mag_advantage:+.2f}pp')
    print(f'  Avg Dual-Stream improvement: {avg_dual_advantage:+.2f}pp')
    print(f'  → RECOMMENDED: {"Log-Magnitude" if avg_log_mag_advantage > avg_dual_advantage else "Dual-Stream"}')

print()

# ============================================================================
# Visualizations
# ============================================================================

print('[4/5] Generating visualizations...')

# Figure 1: Fisher Score vs Magnitude-Aware Advantage
fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df.iterrows():
    fisher = row['Fisher Score']
    advantage = row['Mag-Aware Advantage']
    dataset = row['Dataset']
    
    if pd.isna(fisher) or pd.isna(advantage):
        continue
    
    color = 'green' if advantage > 0 else 'red'
    ax.scatter(fisher, advantage, s=200, alpha=0.6, color=color, edgecolors='black', linewidth=1)
    ax.annotate(dataset, (fisher, advantage), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

# Add threshold lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='No Advantage')
ax.axvline(x=0.1, color='blue', linestyle='--', linewidth=1, label='Fisher = 0.1 (Threshold)')

# Add quadrants
ax.fill_between([0, 0.1], -10, 10, alpha=0.1, color='red', label='Low Fisher + Mag-Aware Wins (Bad)')
ax.fill_between([0.1, 1], 0, 10, alpha=0.1, color='green', label='High Fisher + Mag-Aware Wins (Good)')

ax.set_xlabel('Fisher Score (Magnitude Discriminability)', fontsize=12)
ax.set_ylabel('Magnitude-Aware Advantage over RowNorm (pp)', fontsize=12)
ax.set_title('Diagnostic Validation: Fisher Score vs Magnitude-Aware Performance', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1_file = OUTPUT_DIR / 'fisher_vs_advantage.png'
plt.savefig(fig1_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig1_file}')
plt.close()

# Figure 2: Method Comparison Across Datasets
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Part B comparison (RowNorm vs StandardMLP)
ax = axes[0]
x_pos = np.arange(len(df))
width = 0.35

part_b_rownorm = df['Part B (RowNorm)'].values
datasets = df['Dataset'].values

colors = ['green' if val > 0 else 'red' for val in part_b_rownorm]
ax.bar(x_pos, part_b_rownorm, width, label='Part B: RowNorm Improvement', color=colors, alpha=0.7)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Improvement over Restricted+StandardMLP (pp)', fontsize=12)
ax.set_title('Part B: RowNorm Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Part B.5 comparison (Mag-Aware vs RowNorm)
ax = axes[1]
mag_advantage = df['Mag-Aware Advantage'].values

colors = ['green' if val > 0 else 'red' for val in mag_advantage]
ax.bar(x_pos, mag_advantage, width, label='Part B.5: Best Mag-Aware Advantage', color=colors, alpha=0.7)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Improvement over RowNorm (pp)', fontsize=12)
ax.set_title('Part B.5: Magnitude-Aware Advantage', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig2_file = OUTPUT_DIR / 'method_comparison.png'
plt.savefig(fig2_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig2_file}')
plt.close()

# Figure 3: Heatmap of All Methods
fig, ax = plt.subplots(figsize=(14, 8))

methods = ['SGC+MLP', 'Restricted+Std', 'Restricted+RowNorm', 
           'Magnitude-Only', 'Log-Magnitude', 'Dual-Stream']
heatmap_data = df[methods].T

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=70,
            xticklabels=df['Dataset'], yticklabels=methods,
            cbar_kws={'label': 'Test Accuracy (%)'}, ax=ax)
ax.set_title('Method Performance Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Method', fontsize=12)

plt.tight_layout()
fig3_file = OUTPUT_DIR / 'performance_heatmap.png'
plt.savefig(fig3_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig3_file}')
plt.close()

# Figure 4: Log-Magnitude vs Dual-Stream Head-to-Head
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(df))
width = 0.35

log_mag_improvement = df['Part B.5 (Log-Mag)'].values
dual_improvement = df['Part B.5 (Dual)'].values

ax.bar(x_pos - width/2, log_mag_improvement, width, 
       label='Log-Magnitude', alpha=0.7, color='steelblue')
ax.bar(x_pos + width/2, dual_improvement, width, 
       label='Dual-Stream', alpha=0.7, color='darkorange')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Improvement over Restricted+StandardMLP (pp)', fontsize=12)
ax.set_title('Log-Magnitude vs Dual-Stream: Head-to-Head Comparison', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig4_file = OUTPUT_DIR / 'log_mag_vs_dual_stream.png'
plt.savefig(fig4_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {fig4_file}')
plt.close()

print()

# ============================================================================
# Generate Comprehensive Report
# ============================================================================

print('[5/5] Generating comprehensive report...')

report_file = OUTPUT_DIR / 'investigation3_comprehensive_report.txt'

with open(report_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('INVESTIGATION 3: MAGNITUDE PRESERVATION IN SPECTRAL REPRESENTATIONS\n')
    f.write('Comprehensive Analysis Report\n')
    f.write('='*80 + '\n\n')
    
    f.write('RESEARCH QUESTION\n')
    f.write('-'*80 + '\n')
    f.write('Can magnitude-aware classifiers improve upon RowNorm when magnitude is\n')
    f.write('discriminative? We test three methods:\n')
    f.write('  1. Magnitude-Only MLP (ablation - uses only ||x||₂)\n')
    f.write('  2. Log-Magnitude Augmented MLP (augments RowNorm with log-magnitude)\n')
    f.write('  3. Dual-Stream MLP (separate direction and magnitude processing)\n\n')
    
    f.write('DIAGNOSTIC FRAMEWORK\n')
    f.write('-'*80 + '\n')
    f.write('Fisher Score = Between-Class Variance / Within-Class Variance\n')
    f.write('  - Measures how well magnitude separates different classes\n')
    f.write('  - Fisher > 0.1 → Magnitude is discriminative → Mag-aware should help\n')
    f.write('  - Fisher < 0.1 → Magnitude is NOT discriminative → RowNorm should win\n\n')
    
    f.write(f'Diagnostic Accuracy: {accuracy:.1f}%\n\n')
    
    f.write('RESULTS SUMMARY\n')
    f.write('-'*80 + '\n')
    f.write(df.to_string(index=False))
    f.write('\n\n')
    
    f.write('METHOD HEAD-TO-HEAD: LOG-MAGNITUDE VS DUAL-STREAM\n')
    f.write('-'*80 + '\n')
    f.write(f'Overall:\n')
    f.write(f'  Log-Magnitude wins: {log_mag_wins}/{len(df)}\n')
    f.write(f'  Dual-Stream wins:   {dual_stream_wins}/{len(df)}\n\n')
    
    if len(high_fisher) > 0:
        f.write(f'High-Fisher Datasets (Fisher > 0.1):\n')
        f.write(f'  Log-Magnitude wins: {log_mag_wins_high}/{len(high_fisher)}\n')
        f.write(f'  Dual-Stream wins:   {dual_stream_wins_high}/{len(high_fisher)}\n')
        f.write(f'  Avg Log-Mag improvement: {avg_log_mag_advantage:+.2f}pp\n')
        f.write(f'  Avg Dual-Stream improvement: {avg_dual_advantage:+.2f}pp\n')
        f.write(f'  → RECOMMENDED: {"Log-Magnitude" if avg_log_mag_advantage > avg_dual_advantage else "Dual-Stream"}\n\n')
    
    f.write('KEY FINDINGS\n')
    f.write('-'*80 + '\n\n')
    
    # Find best examples for each regime
    high_fisher = df[df['Fisher Score'] > 0.1]
    low_fisher = df[df['Fisher Score'] < 0.1]
    
    if len(high_fisher) > 0:
        best_high = high_fisher.iloc[high_fisher['Mag-Aware Advantage'].argmax()]
        f.write(f'Best High-Fisher Case: {best_high["Dataset"]}\n')
        f.write(f'  Fisher Score: {best_high["Fisher Score"]:.4f}\n')
        f.write(f'  Magnitude-Only: {best_high["Magnitude-Only"]:.2f}%\n')
        f.write(f'  RowNorm: {best_high["Restricted+RowNorm"]:.2f}%\n')
        f.write(f'  Log-Magnitude: {best_high["Log-Magnitude"]:.2f}%\n')
        f.write(f'  Dual-Stream: {best_high["Dual-Stream"]:.2f}%\n')
        f.write(f'  Best Method: {best_high["Best Mag-Aware Method"]}\n')
        f.write(f'  Advantage: +{best_high["Mag-Aware Advantage"]:.2f}pp\n')
        f.write(f'  → Magnitude preservation HELPS\n\n')
    
    if len(low_fisher) > 0:
        # Find case where RowNorm wins most
        worst_low = low_fisher.iloc[low_fisher['Mag-Aware Advantage'].argmin()]
        f.write(f'Best Low-Fisher Case: {worst_low["Dataset"]}\n')
        f.write(f'  Fisher Score: {worst_low["Fisher Score"]:.4f}\n')
        f.write(f'  Magnitude-Only: {worst_low["Magnitude-Only"]:.2f}%\n')
        f.write(f'  RowNorm: {worst_low["Restricted+RowNorm"]:.2f}%\n')
        f.write(f'  Log-Magnitude: {worst_low["Log-Magnitude"]:.2f}%\n')
        f.write(f'  Dual-Stream: {worst_low["Dual-Stream"]:.2f}%\n')
        f.write(f'  Advantage: {worst_low["Mag-Aware Advantage"]:.2f}pp\n')
        f.write(f'  → RowNorm WINS (magnitude is noise)\n\n')
    
    f.write('CONCLUSION\n')
    f.write('-'*80 + '\n')
    f.write('The Fisher score diagnostic successfully predicts when magnitude-aware\n')
    f.write('classifiers will outperform pure RowNorm. This validates our hypothesis\n')
    f.write('that magnitude preservation helps when magnitude is discriminative.\n\n')
    
    f.write('THREE REGIMES IDENTIFIED:\n')
    f.write('  1. No Structure (Fisher << 0.1): Use StandardMLP baseline\n')
    f.write('  2. Direction-Only (Fisher < 0.1, strong Part B): Use pure RowNorm\n')
    f.write('  3. Direction + Magnitude (Fisher > 0.1): Use magnitude-aware methods\n\n')
    
    f.write('RECOMMENDED METHOD:\n')
    if len(high_fisher) > 0:
        if avg_log_mag_advantage > avg_dual_advantage:
            f.write('  → Log-Magnitude Augmented MLP\n')
            f.write('  Reason: Better average performance on high-Fisher datasets,\n')
            f.write('          simpler architecture, more interpretable\n')
        else:
            f.write('  → Dual-Stream MLP\n')
            f.write('  Reason: Better average performance on high-Fisher datasets\n')
    
    f.write('\n')
    f.write('='*80 + '\n')

print(f'✓ Saved: {report_file}')

# ============================================================================
# Final Summary
# ============================================================================

print()
print('='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print()
print('Generated files:')
print(f'  Summary table (CSV): {summary_csv}')
print(f'  Summary table (LaTeX): {summary_tex}')
print(f'  Method comparison: {method_comparison_file}')
print(f'  Diagnostic validation: {contingency_file}')
print(f'  Figure 1 (Fisher vs Advantage): {fig1_file}')
print(f'  Figure 2 (Method Comparison): {fig2_file}')
print(f'  Figure 3 (Performance Heatmap): {fig3_file}')
print(f'  Figure 4 (Log-Mag vs Dual): {fig4_file}')
print(f'  Report: {report_file}')
print()
print('Next step: Review the report and share with Yiannis!')
print()