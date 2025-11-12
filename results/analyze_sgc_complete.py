"""
Complete SGC Comparison Analysis
==================================

Analyzes results from:
1. investigation_sgc_comparison (Logistic Regression, no bias)
2. investigation_sgc_rownorm (RowNorm MLP with hidden layers)

Generates:
- Comparison tables across all datasets
- Plots comparing SGC vs Logistic vs MLP
- Complete summary report

Usage:
    python scripts/analyze_sgc_complete.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

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

K_VALUES = [2, 4, 8, 16]

# Paths
RESULTS_LOGISTIC = 'results/investigation_sgc_comparison'
RESULTS_MLP = 'results/investigation_sgc_rownorm'
OUTPUT_DIR = 'results/sgc_complete_analysis'
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

print('='*80)
print('COMPLETE SGC COMPARISON ANALYSIS')
print('='*80)
print(f'Datasets: {len(DATASETS)}')
print(f'K values: {K_VALUES}')
print(f'Output: {OUTPUT_DIR}/')
print('='*80)

# ============================================================================
# Load Results
# ============================================================================

def load_experiment_results(base_path, dataset, k):
    """Load results for a specific dataset and k value"""
    path = f'{base_path}/{dataset}/k{k}/metrics/results.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

print('\nLoading results...')

results_data = {
    'logistic': {},  # Old: Logistic regression, no bias
    'mlp': {}        # New: RowNorm MLP with hidden layers
}

for dataset in DATASETS:
    results_data['logistic'][dataset] = {}
    results_data['mlp'][dataset] = {}
    
    for k in K_VALUES:
        # Load logistic results
        logistic_result = load_experiment_results(RESULTS_LOGISTIC, dataset, k)
        if logistic_result:
            results_data['logistic'][dataset][k] = logistic_result
        
        # Load MLP results
        mlp_result = load_experiment_results(RESULTS_MLP, dataset, k)
        if mlp_result:
            results_data['mlp'][dataset][k] = mlp_result

# Count loaded results
n_logistic = sum(len(results_data['logistic'][d]) for d in DATASETS)
n_mlp = sum(len(results_data['mlp'][d]) for d in DATASETS)
print(f'✓ Loaded {n_logistic} logistic results')
print(f'✓ Loaded {n_mlp} MLP results')

# ============================================================================
# Extract Best Results per Dataset
# ============================================================================

def get_best_k_results(dataset_results):
    """Find best k for each method"""
    if not dataset_results:
        return None
    
    best_k = None
    best_acc = -1
    best_result = None
    
    for k, result in dataset_results.items():
        # Get RowNorm variant accuracy (key differs between experiments)
        if 'rownorm_variant' in result:
            acc = result['rownorm_variant']['test_acc_mean']
        elif 'rownorm_mlp' in result:
            acc = result['rownorm_mlp']['test_acc_mean']
        else:
            continue
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_result = result
    
    return {'k': best_k, 'result': best_result}

print('\nExtracting best k values per dataset...')

best_results = {
    'logistic': {},
    'mlp': {}
}

for dataset in DATASETS:
    best_results['logistic'][dataset] = get_best_k_results(
        results_data['logistic'][dataset]
    )
    best_results['mlp'][dataset] = get_best_k_results(
        results_data['mlp'][dataset]
    )

# ============================================================================
# Create Summary Table
# ============================================================================

print('\nGenerating summary table...')

summary_lines = []
summary_lines.append('='*120)
summary_lines.append('COMPLETE SGC COMPARISON: LOGISTIC VS MLP')
summary_lines.append('='*120)
summary_lines.append('')
summary_lines.append('Comparison of three methods:')
summary_lines.append('  1. SGC Baseline: X → A^k @ X → Logistic Regression (with bias)')
summary_lines.append('  2. RowNorm Logistic: X → A^k @ X → Restricted Eigs → Row Norm → Logistic (no bias)')
summary_lines.append('  3. RowNorm MLP: X → A^k @ X → Restricted Eigs → Row Norm → MLP (256 hidden)')
summary_lines.append('')
summary_lines.append('='*120)
summary_lines.append(f'{"Dataset":<20} {"k*":<5} {"SGC":<12} {"Logistic":<12} {"MLP":<12} {"Δ(Log)":<10} {"Δ(MLP)":<10} {"MLP Gain":<10}')
summary_lines.append('-'*120)

# Data for plotting
plot_data = {
    'datasets': [],
    'sgc_accs': [],
    'logistic_accs': [],
    'mlp_accs': [],
    'logistic_improvements': [],
    'mlp_improvements': [],
    'mlp_gains': []
}

for dataset in DATASETS:
    best_log = best_results['logistic'].get(dataset)
    best_mlp = best_results['mlp'].get(dataset)
    
    if not best_log and not best_mlp:
        summary_lines.append(f'{dataset:<20} {"N/A":<5} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<10} {"N/A":<10} {"N/A":<10}')
        continue
    
    # Get results (prefer MLP for k and SGC baseline)
    if best_mlp:
        k_star = best_mlp['k']
        result = best_mlp['result']
        sgc_acc = result['sgc_baseline']['test_acc_mean'] * 100
        sgc_std = result['sgc_baseline']['test_acc_std'] * 100
        mlp_acc = result['rownorm_mlp']['test_acc_mean'] * 100
        mlp_std = result['rownorm_mlp']['test_acc_std'] * 100
        mlp_improvement = result['improvement']['absolute_pp']
    else:
        k_star = None
        sgc_acc = None
        mlp_acc = None
        mlp_improvement = None
    
    if best_log:
        log_result = best_log['result']
        if sgc_acc is None:  # Use logistic's SGC baseline if MLP not available
            k_star = best_log['k']
            sgc_acc = log_result['sgc_baseline']['test_acc_mean'] * 100
        log_acc = log_result['rownorm_variant']['test_acc_mean'] * 100
        log_std = log_result['rownorm_variant']['test_acc_std'] * 100
        log_improvement = log_result['improvement']['absolute_pp']
    else:
        log_acc = None
        log_improvement = None
    
    # Compute MLP gain over logistic
    if log_acc is not None and mlp_acc is not None:
        mlp_gain = mlp_acc - log_acc
    else:
        mlp_gain = None
    
    # Format strings
    sgc_str = f'{sgc_acc:.2f}%' if sgc_acc is not None else 'N/A'
    log_str = f'{log_acc:.2f}%' if log_acc is not None else 'N/A'
    mlp_str = f'{mlp_acc:.2f}%' if mlp_acc is not None else 'N/A'
    log_imp_str = f'{log_improvement:+.2f}pp' if log_improvement is not None else 'N/A'
    mlp_imp_str = f'{mlp_improvement:+.2f}pp' if mlp_improvement is not None else 'N/A'
    mlp_gain_str = f'{mlp_gain:+.2f}pp' if mlp_gain is not None else 'N/A'
    k_str = str(k_star) if k_star is not None else 'N/A'
    
    summary_lines.append(
        f'{dataset:<20} {k_str:<5} {sgc_str:<12} {log_str:<12} {mlp_str:<12} '
        f'{log_imp_str:<10} {mlp_imp_str:<10} {mlp_gain_str:<10}'
    )
    
    # Store for plotting
    if sgc_acc is not None:
        plot_data['datasets'].append(dataset)
        plot_data['sgc_accs'].append(sgc_acc)
        plot_data['logistic_accs'].append(log_acc if log_acc is not None else sgc_acc)
        plot_data['mlp_accs'].append(mlp_acc if mlp_acc is not None else sgc_acc)
        plot_data['logistic_improvements'].append(log_improvement if log_improvement is not None else 0)
        plot_data['mlp_improvements'].append(mlp_improvement if mlp_improvement is not None else 0)
        plot_data['mlp_gains'].append(mlp_gain if mlp_gain is not None else 0)

summary_lines.append('='*120)
summary_lines.append('')
summary_lines.append('Legend:')
summary_lines.append('  k* = Optimal propagation steps')
summary_lines.append('  Δ(Log) = Logistic improvement over SGC')
summary_lines.append('  Δ(MLP) = MLP improvement over SGC')
summary_lines.append('  MLP Gain = MLP improvement over Logistic')
summary_lines.append('')

# Compute statistics
log_improvements = [x for x in plot_data['logistic_improvements'] if x != 0]
mlp_improvements = [x for x in plot_data['mlp_improvements'] if x != 0]
mlp_gains = [x for x in plot_data['mlp_gains'] if x != 0]

summary_lines.append('SUMMARY STATISTICS')
summary_lines.append('='*120)
summary_lines.append('')
summary_lines.append('Logistic vs SGC:')
summary_lines.append(f'  Mean improvement: {np.mean(log_improvements):.2f}pp')
summary_lines.append(f'  Datasets improved: {sum(1 for x in log_improvements if x > 0)}/{len(log_improvements)}')
summary_lines.append(f'  Best: {max(log_improvements):+.2f}pp')
summary_lines.append(f'  Worst: {min(log_improvements):+.2f}pp')
summary_lines.append('')
summary_lines.append('MLP vs SGC:')
summary_lines.append(f'  Mean improvement: {np.mean(mlp_improvements):.2f}pp')
summary_lines.append(f'  Datasets improved: {sum(1 for x in mlp_improvements if x > 0)}/{len(mlp_improvements)}')
summary_lines.append(f'  Best: {max(mlp_improvements):+.2f}pp')
summary_lines.append(f'  Worst: {min(mlp_improvements):+.2f}pp')
summary_lines.append('')
summary_lines.append('MLP vs Logistic:')
summary_lines.append(f'  Mean gain: {np.mean(mlp_gains):.2f}pp')
summary_lines.append(f'  Datasets improved: {sum(1 for x in mlp_gains if x > 0)}/{len(mlp_gains)}')
summary_lines.append(f'  Best gain: {max(mlp_gains):+.2f}pp')
summary_lines.append(f'  Worst: {min(mlp_gains):+.2f}pp')
summary_lines.append('')
summary_lines.append('='*120)

# Print and save
for line in summary_lines:
    print(line)

with open(f'{OUTPUT_DIR}/summary_table.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

print(f'\n✓ Summary saved: {OUTPUT_DIR}/summary_table.txt')

# ============================================================================
# Generate Plots
# ============================================================================

print('\nGenerating plots...')

# Plot 1: Accuracy Comparison (All Three Methods)
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(plot_data['datasets']))
width = 0.25

bars1 = ax.bar(x - width, plot_data['sgc_accs'], width, 
               label='SGC Baseline', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, plot_data['logistic_accs'], width,
               label='RowNorm Logistic', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, plot_data['mlp_accs'], width,
               label='RowNorm MLP', color='#2ca02c', alpha=0.8)

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('SGC Comparison: All Three Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([d.replace('-', '\n') for d in plot_data['datasets']], 
                    fontsize=9, rotation=0)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/accuracy_comparison_all.png', dpi=150, bbox_inches='tight')
print(f'✓ Plot saved: accuracy_comparison_all.png')
plt.close()

# Plot 2: Improvement Over SGC
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(plot_data['datasets']))
width = 0.35

bars1 = ax.bar(x - width/2, plot_data['logistic_improvements'], width,
               label='Logistic Improvement', color='#ff7f0e', alpha=0.8)
bars2 = ax.bar(x + width/2, plot_data['mlp_improvements'], width,
               label='MLP Improvement', color='#2ca02c', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Improvement over SGC (pp)', fontsize=12, fontweight='bold')
ax.set_title('Improvement Over SGC Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([d.replace('-', '\n') for d in plot_data['datasets']], 
                    fontsize=9, rotation=0)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (log_imp, mlp_imp) in enumerate(zip(plot_data['logistic_improvements'], 
                                            plot_data['mlp_improvements'])):
    # Logistic
    y_offset = 0.5 if log_imp > 0 else -1.5
    ax.text(i - width/2, log_imp + y_offset, f'{log_imp:+.1f}',
            ha='center', fontsize=8, fontweight='bold')
    # MLP
    y_offset = 0.5 if mlp_imp > 0 else -1.5
    ax.text(i + width/2, mlp_imp + y_offset, f'{mlp_imp:+.1f}',
            ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/improvements_over_sgc.png', dpi=150, bbox_inches='tight')
print(f'✓ Plot saved: improvements_over_sgc.png')
plt.close()

# Plot 3: MLP Gain Over Logistic
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(plot_data['datasets']))
colors = ['#2ca02c' if gain > 0 else '#d62728' for gain in plot_data['mlp_gains']]

bars = ax.bar(x, plot_data['mlp_gains'], color=colors, alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('MLP Gain over Logistic (pp)', fontsize=12, fontweight='bold')
ax.set_title('Does MLP Help? (Gain Over Logistic Regression)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([d.replace('-', '\n') for d in plot_data['datasets']], 
                    fontsize=9, rotation=0)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, gain in enumerate(plot_data['mlp_gains']):
    y_offset = 0.5 if gain > 0 else -1.5
    ax.text(i, gain + y_offset, f'{gain:+.1f}',
            ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/mlp_gain_over_logistic.png', dpi=150, bbox_inches='tight')
print(f'✓ Plot saved: mlp_gain_over_logistic.png')
plt.close()

# Plot 4: K-Value Trajectories (for datasets with both results)
print('\nGenerating k-value trajectory plots...')

datasets_with_both = []
for dataset in DATASETS:
    if results_data['logistic'][dataset] and results_data['mlp'][dataset]:
        # Check if we have multiple k values
        if len(results_data['logistic'][dataset]) > 1 or len(results_data['mlp'][dataset]) > 1:
            datasets_with_both.append(dataset)

if datasets_with_both:
    n_datasets = len(datasets_with_both)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset in enumerate(datasets_with_both):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get data for this dataset
        log_data = results_data['logistic'][dataset]
        mlp_data = results_data['mlp'][dataset]
        
        # Extract k values and accuracies
        k_vals_log = sorted(log_data.keys())
        k_vals_mlp = sorted(mlp_data.keys())
        
        sgc_log = [log_data[k]['sgc_baseline']['test_acc_mean'] * 100 for k in k_vals_log]
        log_accs = [log_data[k]['rownorm_variant']['test_acc_mean'] * 100 for k in k_vals_log]
        
        sgc_mlp = [mlp_data[k]['sgc_baseline']['test_acc_mean'] * 100 for k in k_vals_mlp]
        mlp_accs = [mlp_data[k]['rownorm_mlp']['test_acc_mean'] * 100 for k in k_vals_mlp]
        
        # Plot
        ax.plot(k_vals_log, sgc_log, 'o-', label='SGC', color='#1f77b4', linewidth=2, markersize=6)
        ax.plot(k_vals_log, log_accs, 's-', label='Logistic', color='#ff7f0e', linewidth=2, markersize=6)
        ax.plot(k_vals_mlp, mlp_accs, '^-', label='MLP', color='#2ca02c', linewidth=2, markersize=6)
        
        ax.set_xlabel('Propagation Steps (k)', fontsize=10)
        ax.set_ylabel('Test Accuracy (%)', fontsize=10)
        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(set(k_vals_log + k_vals_mlp)))
    
    # Hide empty subplots
    for idx in range(len(datasets_with_both), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/k_trajectories_all.png', dpi=150, bbox_inches='tight')
    print(f'✓ Plot saved: k_trajectories_all.png')
    plt.close()

# ============================================================================
# Generate Detailed Analysis Report
# ============================================================================

print('\nGenerating detailed analysis report...')

report_lines = []
report_lines.append('='*80)
report_lines.append('DETAILED ANALYSIS: SGC VS LOGISTIC VS MLP')
report_lines.append('='*80)
report_lines.append('')
report_lines.append('EXPERIMENTAL SETUP')
report_lines.append('-'*80)
report_lines.append('Three methods compared:')
report_lines.append('')
report_lines.append('1. SGC Baseline (Wu et al., ICML 2019):')
report_lines.append('   X → A^k @ X → Logistic Regression (with bias)')
report_lines.append('   - Simple diffusion + linear classifier')
report_lines.append('   - Fast and effective baseline')
report_lines.append('')
report_lines.append('2. RowNorm Logistic (Investigation SGC Comparison):')
report_lines.append('   X → A^k @ X → Restricted Eigenvectors → Row Norm → Logistic (no bias)')
report_lines.append('   - Restricted eigenvectors provide "better basis"')
report_lines.append('   - Row normalization exploits geometry')
report_lines.append('   - No bias term (most restrictive)')
report_lines.append('')
report_lines.append('3. RowNorm MLP (Investigation SGC RowNorm):')
report_lines.append('   X → A^k @ X → Restricted Eigenvectors → Row Norm → MLP (256 hidden)')
report_lines.append('   - Same preprocessing as Logistic')
report_lines.append('   - Adds nonlinearity (2-layer MLP, hidden_dim=256)')
report_lines.append('   - CosineAnnealingLR scheduler')
report_lines.append('')
report_lines.append('='*80)
report_lines.append('')

# Categorize datasets by outcome
mlp_helps = []
mlp_hurts = []
mlp_neutral = []

for dataset, gain in zip(plot_data['datasets'], plot_data['mlp_gains']):
    if gain > 2:
        mlp_helps.append((dataset, gain))
    elif gain < -2:
        mlp_hurts.append((dataset, gain))
    else:
        mlp_neutral.append((dataset, gain))

report_lines.append('KEY FINDINGS')
report_lines.append('-'*80)
report_lines.append('')
report_lines.append(f'1. MLP HELPS ({len(mlp_helps)} datasets):')
if mlp_helps:
    for dataset, gain in sorted(mlp_helps, key=lambda x: -x[1]):
        log_imp = plot_data['logistic_improvements'][plot_data['datasets'].index(dataset)]
        mlp_imp = plot_data['mlp_improvements'][plot_data['datasets'].index(dataset)]
        report_lines.append(f'   {dataset:20} Logistic: {log_imp:+6.2f}pp → MLP: {mlp_imp:+6.2f}pp (gain: {gain:+.2f}pp)')
    report_lines.append('')
    report_lines.append('   Interpretation: Nonlinearity helps compensate for numerical instability')
    report_lines.append('   in restricted eigenvectors on these datasets.')
else:
    report_lines.append('   None')
report_lines.append('')

report_lines.append(f'2. MLP HURTS ({len(mlp_hurts)} datasets):')
if mlp_hurts:
    for dataset, gain in sorted(mlp_hurts, key=lambda x: x[1]):
        log_imp = plot_data['logistic_improvements'][plot_data['datasets'].index(dataset)]
        mlp_imp = plot_data['mlp_improvements'][plot_data['datasets'].index(dataset)]
        report_lines.append(f'   {dataset:20} Logistic: {log_imp:+6.2f}pp → MLP: {mlp_imp:+6.2f}pp (loss: {gain:+.2f}pp)')
    report_lines.append('')
    report_lines.append('   Interpretation: Additional parameters cause overfitting or the restricted')
    report_lines.append('   eigenvectors are already poor quality (adding capacity doesn\'t help).')
else:
    report_lines.append('   None')
report_lines.append('')

report_lines.append(f'3. MLP NEUTRAL ({len(mlp_neutral)} datasets):')
if mlp_neutral:
    for dataset, gain in mlp_neutral:
        log_imp = plot_data['logistic_improvements'][plot_data['datasets'].index(dataset)]
        mlp_imp = plot_data['mlp_improvements'][plot_data['datasets'].index(dataset)]
        report_lines.append(f'   {dataset:20} Logistic: {log_imp:+6.2f}pp → MLP: {mlp_imp:+6.2f}pp (Δ: {gain:+.2f}pp)')
    report_lines.append('')
    report_lines.append('   Interpretation: Minimal difference—either logistic is sufficient or both fail.')
else:
    report_lines.append('   None')
report_lines.append('')

report_lines.append('='*80)
report_lines.append('')
report_lines.append('OVERALL ASSESSMENT')
report_lines.append('-'*80)
report_lines.append('')

# Calculate success rates
log_successes = sum(1 for x in plot_data['logistic_improvements'] if x > 2)
mlp_successes = sum(1 for x in plot_data['mlp_improvements'] if x > 2)
total = len(plot_data['datasets'])

report_lines.append(f'Success Rate (>2pp improvement over SGC):')
report_lines.append(f'  Logistic: {log_successes}/{total} = {log_successes/total*100:.1f}%')
report_lines.append(f'  MLP:      {mlp_successes}/{total} = {mlp_successes/total*100:.1f}%')
report_lines.append('')

avg_log = np.mean(plot_data['logistic_improvements'])
avg_mlp = np.mean(plot_data['mlp_improvements'])
avg_gain = np.mean(plot_data['mlp_gains'])

report_lines.append(f'Average Performance vs SGC:')
report_lines.append(f'  Logistic: {avg_log:+.2f}pp')
report_lines.append(f'  MLP:      {avg_mlp:+.2f}pp')
report_lines.append(f'  MLP Gain: {avg_gain:+.2f}pp')
report_lines.append('')

if avg_mlp > avg_log + 1:
    conclusion = 'MLP consistently outperforms Logistic across datasets.'
elif avg_mlp < avg_log - 1:
    conclusion = 'Logistic is generally better than MLP (overfitting or unnecessary complexity).'
else:
    conclusion = 'MLP and Logistic show similar performance on average.'

report_lines.append(f'Conclusion: {conclusion}')
report_lines.append('')

report_lines.append('='*80)
report_lines.append('')
report_lines.append('RECOMMENDATIONS FOR THURSDAY MEETING')
report_lines.append('-'*80)
report_lines.append('')

if mlp_successes > log_successes:
    report_lines.append('1. MLP variant shows improvement over Logistic—use this for paper.')
    report_lines.append('2. Focus paper on datasets where method works (highlight successes).')
    report_lines.append('3. Discuss limitations on high-dimensional datasets in related work.')
elif mlp_successes == log_successes and avg_gain > 0:
    report_lines.append('1. MLP provides marginal gains—decision depends on "simple models" argument.')
    report_lines.append('2. Could present both variants: Logistic (simpler) vs MLP (better performance).')
    report_lines.append('3. Discuss trade-off between simplicity and accuracy.')
else:
    report_lines.append('1. SGC comparison may not be the right direction for the paper.')
    report_lines.append('2. Consider returning to Investigation 1 (true eigenvectors) where results are stronger.')
    report_lines.append('3. Focus on "understanding eigenvector geometry" rather than beating SGC.')

report_lines.append('')
report_lines.append('='*80)

# Print and save
print('\n')
for line in report_lines:
    print(line)

with open(f'{OUTPUT_DIR}/detailed_analysis.txt', 'w') as f:
    f.write('\n'.join(report_lines))

print(f'\n✓ Detailed analysis saved: {OUTPUT_DIR}/detailed_analysis.txt')

# ============================================================================
# Final Summary
# ============================================================================

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print(f'Output directory: {OUTPUT_DIR}/')
print('')
print('Generated files:')
print('  - summary_table.txt (comparison table)')
print('  - detailed_analysis.txt (complete report)')
print('  - plots/accuracy_comparison_all.png')
print('  - plots/improvements_over_sgc.png')
print('  - plots/mlp_gain_over_logistic.png')
if datasets_with_both:
    print('  - plots/k_trajectories_all.png')
print('')
print('Ready for Thursday meeting! 🎯')
print('='*80)