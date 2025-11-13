"""
Correlate Eigenvalue Properties with Performance
================================================

Purpose: Link eigenvalue distribution properties to:
1. Baseline restricted eigenvector performance
2. Diffusion benefit
3. RowNorm success/failure

This answers: "WHEN does our method work based on eigenvalue properties?"

Usage:
    python correlate_eigenvalues_performance.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

# Load eigenvalue analysis results
EIGENVALUE_FILE = 'results/eigenvalue_analysis/data/eigenvalue_analysis_complete.json'

# Known baseline performance (from your experiments)
# These are from Investigation 2 - baseline restricted eigenvector + RowNorm
BASELINE_PERFORMANCE = {
    'ogbn-arxiv': 55.74,      # From Diffused_Engineered.pdf Table 1
    'cora': 75.61,
    'citeseer': 68.08,
    'pubmed': 39.98,
    'wikics': 73.52,
    'amazon-computers': 81.04,
    'coauthor-cs': 84.22,
    'coauthor-physics': 90.30
}

# Diffusion improvements (from Diffused_Engineered.pdf Table 1)
DIFFUSION_IMPROVEMENTS = {
    'ogbn-arxiv': 13.83,
    'cora': 7.08,
    'citeseer': 0.12,
    'pubmed': 43.33,
    'wikics': 1.96,
    'amazon-computers': 4.95,
    'coauthor-cs': -2.54,
    'coauthor-physics': 1.83
}

OUTPUT_DIR = 'results/eigenvalue_analysis'
os.makedirs(f'{OUTPUT_DIR}/correlations', exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

print('='*80)
print('EIGENVALUE-PERFORMANCE CORRELATION ANALYSIS')
print('='*80)

# ============================================================================
# Load Data
# ============================================================================

print('\n[1/3] Loading eigenvalue analysis results...')

if not os.path.exists(EIGENVALUE_FILE):
    print(f'❌ Error: {EIGENVALUE_FILE} not found!')
    print('Run analyze_eigenvalue_distributions.py first.')
    exit(1)

with open(EIGENVALUE_FILE, 'r') as f:
    eigenvalue_data = json.load(f)

print(f'✓ Loaded data for {len(eigenvalue_data)} datasets')

# ============================================================================
# Prepare Correlation Data
# ============================================================================

print('\n[2/3] Preparing correlation data...')

correlation_data = []

for dataset, data in eigenvalue_data.items():
    if dataset not in BASELINE_PERFORMANCE:
        print(f'⚠️  Skipping {dataset}: no performance data')
        continue
    
    baseline_stats = data['baseline']['statistics']
    
    row = {
        'dataset': dataset,
        'baseline_accuracy': BASELINE_PERFORMANCE[dataset],
        'diffusion_improvement': DIFFUSION_IMPROVEMENTS.get(dataset, 0),
        'condition_number': data['baseline']['condition'],
        'eigenvalue_spread': baseline_stats['std'],
        'eigenvalue_mean': baseline_stats['mean'],
        'eigenvalue_skewness': baseline_stats['skewness'],
        'eigenvalue_gap_ratio': baseline_stats['gap_ratio'],
        'num_features': data['metadata']['num_features'],
        'effective_rank': data['baseline']['rank']
    }
    
    # Add log-transformed condition (better for correlation)
    row['log_condition'] = np.log10(row['condition_number']) if row['condition_number'] > 0 else 0
    
    # Compute relative improvement
    if row['baseline_accuracy'] > 0:
        row['relative_improvement'] = (row['diffusion_improvement'] / row['baseline_accuracy']) * 100
    else:
        row['relative_improvement'] = 0
    
    correlation_data.append(row)

df = pd.DataFrame(correlation_data)

print(f'✓ Prepared correlation data for {len(df)} datasets')

# Save correlation table
df.to_csv(f'{OUTPUT_DIR}/correlations/correlation_data.csv', index=False)
print(f'✓ Saved: {OUTPUT_DIR}/correlations/correlation_data.csv')

# ============================================================================
# Correlation Analysis
# ============================================================================

print('\n[3/3] Computing correlations...')

# Compute Pearson correlations
correlations = {}

target_vars = ['baseline_accuracy', 'diffusion_improvement', 'relative_improvement']
predictor_vars = ['condition_number', 'log_condition', 'eigenvalue_spread', 
                  'eigenvalue_skewness', 'eigenvalue_gap_ratio']

print('\nPEARSON CORRELATIONS:')
print('-'*80)

for target in target_vars:
    print(f'\n{target.upper().replace("_", " ")}:')
    correlations[target] = {}
    
    for predictor in predictor_vars:
        if predictor in df.columns and target in df.columns:
            r, p = stats.pearsonr(df[predictor], df[target])
            correlations[target][predictor] = {'r': r, 'p': p}
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f'  {predictor:25s}: r = {r:+.3f}, p = {p:.4f} {sig}')

# Save correlations
with open(f'{OUTPUT_DIR}/correlations/correlation_coefficients.json', 'w') as f:
    json.dump(correlations, f, indent=2)

# ============================================================================
# Visualization
# ============================================================================

print('\nCreating visualizations...')

# Figure 1: Key Correlations (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Condition Number vs Baseline Accuracy
ax = axes[0, 0]
ax.scatter(df['log_condition'], df['baseline_accuracy'], s=100, alpha=0.7, edgecolor='black')
for i, row in df.iterrows():
    ax.annotate(row['dataset'], (row['log_condition'], row['baseline_accuracy']),
               fontsize=8, ha='center', va='bottom')

# Add trend line
z = np.polyfit(df['log_condition'], df['baseline_accuracy'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['log_condition'].min(), df['log_condition'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

r, pval = stats.pearsonr(df['log_condition'], df['baseline_accuracy'])
ax.set_xlabel('log₁₀(Condition Number)', fontsize=11)
ax.set_ylabel('Baseline Accuracy (%)', fontsize=11)
ax.set_title(f'Condition vs Baseline Performance\nr = {r:.3f}, p = {pval:.4f}', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 2: Baseline Accuracy vs Diffusion Improvement
ax = axes[0, 1]
ax.scatter(df['baseline_accuracy'], df['diffusion_improvement'], s=100, alpha=0.7, edgecolor='black')
for i, row in df.iterrows():
    ax.annotate(row['dataset'], (row['baseline_accuracy'], row['diffusion_improvement']),
               fontsize=8, ha='center', va='bottom')

# Add trend line
z = np.polyfit(df['baseline_accuracy'], df['diffusion_improvement'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['baseline_accuracy'].min(), df['baseline_accuracy'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

r, pval = stats.pearsonr(df['baseline_accuracy'], df['diffusion_improvement'])
ax.set_xlabel('Baseline Accuracy (%)', fontsize=11)
ax.set_ylabel('Diffusion Improvement (pp)', fontsize=11)
ax.set_title(f'Baseline vs Diffusion Benefit\nr = {r:.3f}, p = {pval:.4f}', fontsize=12)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.grid(True, alpha=0.3)

# Plot 3: Eigenvalue Spread vs Performance
ax = axes[1, 0]
ax.scatter(df['eigenvalue_spread'], df['baseline_accuracy'], s=100, alpha=0.7, edgecolor='black')
for i, row in df.iterrows():
    ax.annotate(row['dataset'], (row['eigenvalue_spread'], row['baseline_accuracy']),
               fontsize=8, ha='center', va='bottom')

r, pval = stats.pearsonr(df['eigenvalue_spread'], df['baseline_accuracy'])
ax.set_xlabel('Eigenvalue Spread (std)', fontsize=11)
ax.set_ylabel('Baseline Accuracy (%)', fontsize=11)
ax.set_title(f'Eigenvalue Spread vs Performance\nr = {r:.3f}, p = {pval:.4f}', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 4: Condition vs Relative Improvement
ax = axes[1, 1]
ax.scatter(df['log_condition'], df['relative_improvement'], s=100, alpha=0.7, edgecolor='black')
for i, row in df.iterrows():
    ax.annotate(row['dataset'], (row['log_condition'], row['relative_improvement']),
               fontsize=8, ha='center', va='bottom')

# Add trend line
z = np.polyfit(df['log_condition'], df['relative_improvement'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['log_condition'].min(), df['log_condition'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

r, pval = stats.pearsonr(df['log_condition'], df['relative_improvement'])
ax.set_xlabel('log₁₀(Condition Number)', fontsize=11)
ax.set_ylabel('Relative Improvement (%)', fontsize=11)
ax.set_title(f'Condition vs Relative Benefit\nr = {r:.3f}, p = {pval:.4f}', fontsize=12)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.grid(True, alpha=0.3)

plt.suptitle('Eigenvalue Properties vs Performance', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlations/key_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'✓ Saved: {OUTPUT_DIR}/correlations/key_correlations.png')

# Figure 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Prepare correlation matrix
corr_vars = ['baseline_accuracy', 'diffusion_improvement', 'log_condition', 
             'eigenvalue_spread', 'eigenvalue_skewness', 'num_features']
corr_labels = ['Baseline\nAccuracy', 'Diffusion\nImprovement', 'log(Condition)', 
               'Eigenvalue\nSpread', 'Eigenvalue\nSkewness', 'Feature\nDimension']

corr_matrix = df[corr_vars].corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={'label': 'Pearson r'},
            xticklabels=corr_labels, yticklabels=corr_labels, ax=ax,
            vmin=-1, vmax=1)

ax.set_title('Correlation Matrix: Eigenvalue Properties vs Performance', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlations/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'✓ Saved: {OUTPUT_DIR}/correlations/correlation_heatmap.png')

# Figure 3: Dataset Categorization
fig, ax = plt.subplots(figsize=(12, 8))

# Categorize datasets by condition number
def categorize_condition(cond):
    if cond < 100:
        return 'Well-Conditioned\n(κ < 100)'
    elif cond < 1000:
        return 'Moderate\n(100 ≤ κ < 1000)'
    else:
        return 'Ill-Conditioned\n(κ ≥ 1000)'

df['category'] = df['condition_number'].apply(categorize_condition)

categories = ['Well-Conditioned\n(κ < 100)', 'Moderate\n(100 ≤ κ < 1000)', 'Ill-Conditioned\n(κ ≥ 1000)']
colors = ['green', 'orange', 'red']

for i, cat in enumerate(categories):
    cat_data = df[df['category'] == cat]
    ax.scatter(cat_data['baseline_accuracy'], cat_data['diffusion_improvement'],
              s=150, alpha=0.7, edgecolor='black', color=colors[i], label=cat)
    
    for _, row in cat_data.iterrows():
        ax.annotate(row['dataset'], (row['baseline_accuracy'], row['diffusion_improvement']),
                   fontsize=9, ha='center', va='bottom')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=70, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Good baseline (70%)')

ax.set_xlabel('Baseline Accuracy (%)', fontsize=12)
ax.set_ylabel('Diffusion Improvement (pp)', fontsize=12)
ax.set_title('Dataset Categorization by Condition Number', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlations/dataset_categorization.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'✓ Saved: {OUTPUT_DIR}/correlations/dataset_categorization.png')

# ============================================================================
# Generate Summary Report
# ============================================================================

print('\nGenerating summary report...')

report = []
report.append('='*80)
report.append('EIGENVALUE-PERFORMANCE CORRELATION SUMMARY')
report.append('='*80)
report.append('')

# Key findings
report.append('KEY FINDINGS:')
report.append('-'*80)
report.append('')

# Finding 1: Condition number
r_cond_base = correlations['baseline_accuracy']['log_condition']['r']
p_cond_base = correlations['baseline_accuracy']['log_condition']['p']
report.append(f'1. CONDITION NUMBER vs BASELINE ACCURACY:')
report.append(f'   r = {r_cond_base:.3f}, p = {p_cond_base:.4f}')
if r_cond_base < -0.5 and p_cond_base < 0.05:
    report.append(f'   ✓ Strong negative correlation: Higher condition → worse baseline')
elif abs(r_cond_base) > 0.3:
    report.append(f'   • Moderate correlation detected')
else:
    report.append(f'   • Weak correlation')
report.append('')

# Finding 2: Baseline vs improvement
r_base_imp = correlations['diffusion_improvement']['baseline_accuracy']['r']
p_base_imp = correlations['diffusion_improvement']['baseline_accuracy']['p']
report.append(f'2. BASELINE ACCURACY vs DIFFUSION IMPROVEMENT:')
report.append(f'   r = {r_base_imp:.3f}, p = {p_base_imp:.4f}')
if r_base_imp < -0.5 and p_base_imp < 0.05:
    report.append(f'   ✓ Strong negative correlation: Poor baseline → larger improvement')
elif abs(r_base_imp) > 0.3:
    report.append(f'   • Moderate correlation detected')
else:
    report.append(f'   • Weak correlation')
report.append('')

# Dataset breakdown
report.append('DATASET BREAKDOWN BY CONDITION:')
report.append('-'*80)
report.append('')

for cat in categories:
    cat_data = df[df['category'] == cat]
    if len(cat_data) > 0:
        report.append(f'{cat}:')
        report.append(f'  Datasets: {", ".join(cat_data["dataset"].tolist())}')
        report.append(f'  Avg baseline: {cat_data["baseline_accuracy"].mean():.2f}%')
        report.append(f'  Avg diffusion benefit: {cat_data["diffusion_improvement"].mean():.2f}pp')
        report.append('')

# Interpretation
report.append('INTERPRETATION:')
report.append('-'*80)
report.append('')

if r_cond_base < -0.5 and r_base_imp < -0.5:
    report.append('✓ Clear pattern: Ill-conditioned eigenproblems → poor baseline → large diffusion benefit')
    report.append('')
    report.append('This suggests:')
    report.append('  1. Condition number predicts baseline restricted eigenvector quality')
    report.append('  2. Diffusion acts as eigenvalue regularization')
    report.append('  3. When eigenvalues are clustered (high κ), diffusion spreads them out')
    report.append('  4. Well-separated eigenvalues → diverse eigenvector directions → RowNorm can work')
else:
    report.append('• Pattern is more complex than simple condition number correlation')
    report.append('• Other factors (feature semantics, graph structure) may dominate')

report.append('')
report.append('='*80)

# Save report
report_text = '\n'.join(report)
with open(f'{OUTPUT_DIR}/correlations/summary_report.txt', 'w') as f:
    f.write(report_text)

print(f'✓ Saved: {OUTPUT_DIR}/correlations/summary_report.txt')

# Print report
print('\n' + report_text)

print('\n' + '='*80)
print('CORRELATION ANALYSIS COMPLETE')
print('='*80)
print(f'Results saved to: {OUTPUT_DIR}/correlations/')
print('='*80)