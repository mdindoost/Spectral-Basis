"""
Eigenvalue Distribution Analysis
=================================

Purpose: Investigate the relationship between eigenvalue distribution properties
and RowNorm MLP performance across all experiments.

Research Question:
Does eigenvalue distribution (spread, condition number, participation ratio)
correlate with RowNorm MLP performance advantage over Standard MLP?

Metrics Computed:
- Eigenvalue Spread (Δ): λ_max - λ_min
- Condition Number (κ): λ_max / λ_min  
- Participation Ratio (PR): (Σλ_i)² / Σλ_i²

IMPORTANT: Eigenvalue Sources
-----------------------------
Investigation 1: True eigenvectors from graph Laplacian (compute: eigh(L, D))
Investigation 2a: Restricted eigenvectors from X (compute: restricted_eigenproblem(X, L, D))
Investigation 2b: Random subspaces (compute: restricted_eigenproblem(R, L, D))

If eigenvalues are not in your JSON files, you'll need to:
1. Run this script once to identify missing eigenvalues
2. Compute them using the helper functions below
3. Cache them (manually_add_eigenvalues_to_cache())
4. Rerun this script

Usage:
    # First run (will identify missing eigenvalues)
    python experiments/analyze_eigenvalue_distributions.py
    
    # If eigenvalues missing, compute them with helper script generated
    python results/eigenvalue_analysis/compute_missing_eigenvalues.py
    
    # Then rerun analysis
    python experiments/analyze_eigenvalue_distributions.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Output directory
OUTPUT_DIR = 'results/eigenvalue_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

# Path to store/load eigenvalues cache (if eigenvalues need to be recomputed)
EIGENVALUE_CACHE_DIR = f'{OUTPUT_DIR}/eigenvalue_cache'
os.makedirs(EIGENVALUE_CACHE_DIR, exist_ok=True)

# Experiments to analyze - UPDATED WITH ACTUAL PATHS
EXPERIMENTS = {
    'inv1_true_eigs_fixed': {
        'name': 'True Eigenvectors (fixed)',
        'path_pattern': 'results/investigation1_v2/{dataset}/fixed-splits/metrics/results_aggregated.json',
        'split_type': 'fixed',
        'datasets': ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed']
    },
    'inv1_true_eigs_random': {
        'name': 'True Eigenvectors (random)',
        'path_pattern': 'results/investigation1_v2/{dataset}/random-splits/metrics/results_aggregated.json',
        'split_type': 'random',
        'datasets': ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed']
    },
    'inv2a_restricted_fixed': {
        'name': 'Restricted Eigenvectors (fixed)',
        'path_pattern': 'results/investigation2_directions_AB/{dataset}/fixed_splits/metrics/results_aggregated.json',
        'split_type': 'fixed',
        'datasets': ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 
                     'amazon-computers', 'coauthor-cs', 'coauthor-physics', 'wikics']
    },
    'inv2a_restricted_random': {
        'name': 'Restricted Eigenvectors (random)',
        'path_pattern': 'results/investigation2_directions_AB/{dataset}/random_splits/metrics/results_aggregated.json',
        'split_type': 'random',
        'datasets': ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 
                     'amazon-computers', 'coauthor-cs', 'coauthor-physics', 'wikics']
    },
    'inv2b_random_subspace_fixed': {
        'name': 'Random Subspaces (fixed)',
        'path_pattern': 'results/investigation2_random_subspaces/{dataset}/fixed_splits/metrics/results_complete.json',
        'split_type': 'fixed',
        'datasets': ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 
                     'amazon-computers', 'coauthor-cs', 'coauthor-physics', 'wikics']
    }
}

# Visual styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'True Eigenvectors': '#2E7D32',
    'Restricted Eigenvectors': '#C62828',
    'Diffused Random (k=16)': '#1565C0',
    'Diffused Random (k=32)': '#6A1B9A',
    'Diffused Engineered': '#F57C00'
}

print('='*80)
print('EIGENVALUE DISTRIBUTION ANALYSIS')
print('='*80)
print(f'Output directory: {OUTPUT_DIR}')
print(f'Experiments: {len(EXPERIMENTS)}')

# Debug mode: Test on one configuration first
DEBUG_MODE = False  # Set to True to test on one dataset first
if DEBUG_MODE:
    print('\n⚠ DEBUG MODE: Testing on ogbn-arxiv only')
    for exp_config in EXPERIMENTS.values():
        exp_config['datasets'] = ['ogbn-arxiv']

print('='*80)

# ============================================================================
# Helper Functions
# ============================================================================

def manually_add_eigenvalues_to_cache(experiment_id, dataset, eigenvalues):
    """
    Manually add eigenvalues to cache.
    
    Usage:
        eigenvalues = compute_my_eigenvalues(...)
        manually_add_eigenvalues_to_cache('inv1_true_eigs_fixed', 'ogbn-arxiv', eigenvalues)
    """
    cache_file = f'{EIGENVALUE_CACHE_DIR}/{experiment_id}_{dataset}.npy'
    np.save(cache_file, eigenvalues)
    print(f"✓ Cached {len(eigenvalues)} eigenvalues for {experiment_id}/{dataset}")
    return cache_file

def compute_eigenvalue_metrics(eigenvalues, threshold=1e-10):
    """
    Compute eigenvalue distribution metrics.
    
    Args:
        eigenvalues: numpy array of eigenvalues
        threshold: minimum eigenvalue to consider (excludes near-zero from components)
    
    Returns:
        dict with spread, condition, participation_ratio, n_valid, n_dropped
    """
    # Filter near-zero eigenvalues (from disconnected components)
    valid_eigs = eigenvalues[eigenvalues > threshold]
    dropped = len(eigenvalues) - len(valid_eigs)
    
    if len(valid_eigs) == 0:
        return {
            'spread': np.nan,
            'condition': np.nan,
            'participation_ratio': np.nan,
            'n_eigenvalues_total': len(eigenvalues),
            'n_eigenvalues_valid': 0,
            'n_eigenvalues_dropped': dropped,
            'lambda_min': np.nan,
            'lambda_max': np.nan
        }
    
    lambda_min = valid_eigs.min()
    lambda_max = valid_eigs.max()
    
    # Eigenvalue spread: Δ = λ_max - λ_min
    spread = lambda_max - lambda_min
    
    # Condition number: κ = λ_max / λ_min
    condition = lambda_max / lambda_min if lambda_min > 0 else np.inf
    
    # Participation ratio: PR = (Σλ_i)² / Σλ_i²
    sum_eigs = valid_eigs.sum()
    sum_sq_eigs = (valid_eigs**2).sum()
    participation_ratio = (sum_eigs**2) / sum_sq_eigs if sum_sq_eigs > 0 else np.nan
    
    return {
        'spread': float(spread),
        'condition': float(condition),
        'participation_ratio': float(participation_ratio),
        'n_eigenvalues_total': len(eigenvalues),
        'n_eigenvalues_valid': len(valid_eigs),
        'n_eigenvalues_dropped': dropped,
        'lambda_min': float(lambda_min),
        'lambda_max': float(lambda_max)
    }

def load_eigenvalues_from_experiment(dataset, experiment_config, experiment_id):
    """
    Load eigenvalues from saved experiment results or cache.
    
    Strategy:
    1. Check cache first (eigenvalue_cache/{experiment_id}_{dataset}.npy)
    2. Try to load from JSON if saved
    3. If not available, return None (will need manual computation)
    """
    # Check cache first
    cache_file = f'{EIGENVALUE_CACHE_DIR}/{experiment_id}_{dataset}.npy'
    if os.path.exists(cache_file):
        try:
            eigenvalues = np.load(cache_file)
            return eigenvalues
        except:
            pass
    
    # Try to load from JSON
    path = experiment_config['path_pattern'].format(dataset=dataset)
    
    if not os.path.exists(path):
        print(f"  ⚠ File not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Search for eigenvalues in various possible locations
        eigenvalues = None
        
        # Common patterns
        search_keys = [
            'eigenvalues',
            'restricted_eigenvalues', 
            'graph_eigenvalues',
            'lambda',
            'eigs'
        ]
        
        # Try direct access
        for key in search_keys:
            if key in data:
                eigenvalues = np.array(data[key])
                break
        
        # Try nested access
        if eigenvalues is None:
            for parent_key in data:
                if isinstance(data[parent_key], dict):
                    for key in search_keys:
                        if key in data[parent_key]:
                            eigenvalues = np.array(data[parent_key][key])
                            break
                    if eigenvalues is not None:
                        break
        
        # If found, cache it
        if eigenvalues is not None:
            np.save(cache_file, eigenvalues)
            return eigenvalues
        
        # Eigenvalues not found in JSON
        print(f"  ⚠ Eigenvalues not found in {path}")
        print(f"     You'll need to compute them separately")
        return None
        
    except Exception as e:
        print(f"  ✗ Error loading {path}: {e}")
        return None

def load_performance_from_experiment(dataset, experiment_config):
    """
    Load RowNorm and Standard MLP performance from saved results.
    
    Expected structure in results_aggregated.json:
    {
        "a_raw_std": {"test_acc_mean": 0.55, ...},
        "b_restricted_std": {"test_acc_mean": 0.54, ...},
        "c_restricted_rownorm": {"test_acc_mean": 0.47, ...}
    }
    """
    path = experiment_config['path_pattern'].format(dataset=dataset)
    
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        performance = {}
        
        # Investigation 1 & 2a structure (directions AB)
        if 'c_restricted_rownorm' in data:
            # Experiment (c): RowNorm MLP on restricted eigenvectors
            performance['rownorm'] = data['c_restricted_rownorm'].get('test_acc_mean', 
                                    data['c_restricted_rownorm'].get('test_acc', None))
        
        # Try different keys for Standard MLP
        if 'a_raw_std' in data:
            # Experiment (a): Standard MLP on raw features X
            performance['standard'] = data['a_raw_std'].get('test_acc_mean',
                                     data['a_raw_std'].get('test_acc', None))
        elif 'b_restricted_std' in data:
            # Experiment (b): Standard MLP on restricted eigenvectors
            performance['standard'] = data['b_restricted_std'].get('test_acc_mean',
                                     data['b_restricted_std'].get('test_acc', None))
        
        # Investigation 2b/2c structure (diffused features)
        if not performance and 'graph_diffused' in data:
            perf_data = data['graph_diffused']
            if 'c_restricted_rownorm' in perf_data:
                performance['rownorm'] = perf_data['c_restricted_rownorm'].get('mean', None)
            if 'a_raw_std' in perf_data:
                performance['standard'] = perf_data['a_raw_std'].get('mean', None)
        
        # Try flattened structure
        if not performance:
            for key in ['rownorm_test_acc', 'rownorm_acc', 'test_acc_rownorm']:
                if key in data:
                    performance['rownorm'] = data[key]
                    break
            
            for key in ['standard_test_acc', 'standard_acc', 'test_acc_standard']:
                if key in data:
                    performance['standard'] = data[key]
                    break
        
        # Validate
        if not performance or 'rownorm' not in performance or 'standard' not in performance:
            print(f"  ⚠ Incomplete performance data in: {path}")
            print(f"     Found keys: {list(data.keys())}")
            return None
        
        return performance
        
    except Exception as e:
        print(f"  ✗ Error loading performance from {path}: {e}")
        return None

# ============================================================================
# Main Analysis
# ============================================================================

print('\n[1/4] Loading data and computing metrics...\n')

results = []
failed = []
missing_eigenvalues = []

for exp_id, exp_config in EXPERIMENTS.items():
    print(f"{'='*80}")
    print(f"Experiment: {exp_config['name']}")
    print(f"{'='*80}")
    
    for dataset in exp_config['datasets']:
        print(f"\n  Processing {dataset}...", end=' ')
        
        try:
            # Load eigenvalues
            eigenvalues = load_eigenvalues_from_experiment(dataset, exp_config, exp_id)
            if eigenvalues is None:
                missing_eigenvalues.append((exp_id, dataset))
                failed.append((exp_id, dataset, "eigenvalues not available"))
                print("✗ (no eigenvalues)")
                continue
            
            # Load performance
            performance = load_performance_from_experiment(dataset, exp_config)
            if performance is None:
                failed.append((exp_id, dataset, "performance not found"))
                print("✗ (no performance)")
                continue
            
            # Compute metrics
            metrics = compute_eigenvalue_metrics(eigenvalues)
            
            # Compute RowNorm advantage
            if 'rownorm' in performance and 'standard' in performance:
                rownorm_advantage = (performance['rownorm'] - performance['standard']) * 100
            else:
                rownorm_advantage = np.nan
            
            # Store result
            result = {
                'experiment_id': exp_id,
                'experiment_name': exp_config['name'],
                'dataset': dataset,
                'split_type': exp_config.get('split_type', 'unknown'),
                'rownorm_acc': performance.get('rownorm', np.nan),
                'standard_acc': performance.get('standard', np.nan),
                'rownorm_advantage': rownorm_advantage,
                **metrics
            }
            
            results.append(result)
            
            print(f"✓ κ={metrics['condition']:.2e}, Δ={metrics['spread']:.4e}, " + 
                  f"RowNorm adv={rownorm_advantage:+.2f}%")
            
        except Exception as e:
            failed.append((exp_id, dataset, str(e)))
            print(f"✗ Error: {e}")

# Convert to DataFrame
df = pd.DataFrame(results)

print(f"\n{'='*80}")
print(f"Data collection complete: {len(results)} configurations, {len(failed)} failed")
print(f"{'='*80}")

if len(missing_eigenvalues) > 0:
    print(f"\n⚠ WARNING: {len(missing_eigenvalues)} configurations missing eigenvalues")
    print("You need to compute eigenvalues separately. See helper script below.")
    print("\nMissing eigenvalues for:")
    for exp_id, dataset in missing_eigenvalues[:10]:  # Show first 10
        print(f"  • {exp_id}/{dataset}")
    if len(missing_eigenvalues) > 10:
        print(f"  ... and {len(missing_eigenvalues) - 10} more")

if len(failed) > 0:
    print("\nAll failed configurations:")
    for exp_id, dataset, reason in failed:
        print(f"  • {exp_id}/{dataset}: {reason}")

# Save raw data
df.to_csv(f'{OUTPUT_DIR}/eigenvalue_metrics_raw.csv', index=False)
print(f"\n✓ Raw data saved: {OUTPUT_DIR}/eigenvalue_metrics_raw.csv")

# ============================================================================
# Statistical Analysis
# ============================================================================

print(f"\n{'='*80}")
print('[2/4] Statistical Analysis')
print(f"{'='*80}")

metrics_to_analyze = ['spread', 'condition', 'participation_ratio']
correlation_results = {}

for metric in metrics_to_analyze:
    print(f"\n{metric.upper().replace('_', ' ')}")
    print("-" * 60)
    
    # Filter valid data
    valid_data = df[[metric, 'rownorm_advantage']].dropna()
    
    if len(valid_data) < 3:
        print(f"  ⚠ Insufficient data (N={len(valid_data)})")
        continue
    
    # Compute correlations
    r_pearson, p_pearson = pearsonr(valid_data[metric], valid_data['rownorm_advantage'])
    r_spearman, p_spearman = spearmanr(valid_data[metric], valid_data['rownorm_advantage'])
    
    # Store results
    correlation_results[metric] = {
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_rho': r_spearman,
        'spearman_p': p_spearman,
        'n_samples': len(valid_data)
    }
    
    print(f"  Pearson correlation:  r = {r_pearson:+.3f}, p = {p_pearson:.4f}")
    print(f"  Spearman correlation: ρ = {r_spearman:+.3f}, p = {p_spearman:.4f}")
    print(f"  Sample size: N = {len(valid_data)}")
    
    # Interpretation
    if abs(r_pearson) > 0.7 and p_pearson < 0.01:
        print(f"  → Strong correlation (|r| > 0.7, p < 0.01)")
    elif abs(r_pearson) > 0.4 and p_pearson < 0.05:
        print(f"  → Moderate correlation (|r| > 0.4, p < 0.05)")
    else:
        print(f"  → Weak/no significant correlation")

# Per-experiment analysis
print(f"\n{'='*80}")
print("PER-EXPERIMENT CORRELATION ANALYSIS")
print(f"{'='*80}")

for exp_name in df['experiment_name'].unique():
    exp_data = df[df['experiment_name'] == exp_name]
    
    print(f"\n{exp_name}")
    print("-" * 60)
    
    for metric in metrics_to_analyze:
        valid = exp_data[[metric, 'rownorm_advantage']].dropna()
        
        if len(valid) < 3:
            print(f"  {metric}: Insufficient data (N={len(valid)})")
            continue
        
        r, p = pearsonr(valid[metric], valid['rownorm_advantage'])
        print(f"  {metric}: r = {r:+.3f}, p = {p:.4f}, N = {len(valid)}")

# Save correlation results
with open(f'{OUTPUT_DIR}/correlation_results.json', 'w') as f:
    json.dump(correlation_results, f, indent=2)

# ============================================================================
# Visualizations
# ============================================================================

print(f"\n{'='*80}")
print('[3/4] Generating visualizations...')
print(f"{'='*80}")

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Plot 1: Correlation scatter plots
print("\n  Creating scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (ax, metric) in enumerate(zip(axes, metrics_to_analyze)):
    # Filter valid data
    valid_df = df[[metric, 'rownorm_advantage', 'experiment_name', 'dataset']].dropna()
    
    if len(valid_df) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        continue
    
    # Plot points by experiment type
    for exp_name in valid_df['experiment_name'].unique():
        subset = valid_df[valid_df['experiment_name'] == exp_name]
        color = COLORS.get(exp_name, '#666666')
        
        ax.scatter(subset[metric], subset['rownorm_advantage'], 
                  label=exp_name, color=color, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        
        # Add dataset labels for interesting points
        for _, row in subset.iterrows():
            if abs(row['rownorm_advantage']) > 20 or row['condition'] > 50:  # Label outliers
                ax.annotate(row['dataset'], 
                           (row[metric], row['rownorm_advantage']),
                           fontsize=8, alpha=0.7, xytext=(5, 5),
                           textcoords='offset points')
    
    # Add trend line (overall)
    if len(valid_df) > 2:
        z = np.polyfit(valid_df[metric], valid_df['rownorm_advantage'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_df[metric].min(), valid_df[metric].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    # Add correlation text
    if metric in correlation_results:
        r = correlation_results[metric]['pearson_r']
        p_val = correlation_results[metric]['pearson_p']
        ax.text(0.05, 0.95, f'r = {r:+.3f}\np = {p_val:.4f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel('RowNorm Advantage (%)', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

plt.suptitle('Eigenvalue Distribution vs. RowNorm Performance', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/correlation_scatter.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/plots/correlation_scatter.png")
plt.close()

# Plot 2: Box plots by experiment
print("  Creating box plots...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, metric in zip(axes, metrics_to_analyze):
    valid_df = df[[metric, 'experiment_name']].dropna()
    
    if len(valid_df) == 0:
        continue
    
    # Create box plot
    exp_names = valid_df['experiment_name'].unique()
    data_to_plot = [valid_df[valid_df['experiment_name'] == exp][metric].values 
                    for exp in exp_names]
    
    bp = ax.boxplot(data_to_plot, labels=exp_names, patch_artist=True)
    
    # Color boxes
    for patch, exp_name in zip(bp['boxes'], exp_names):
        patch.set_facecolor(COLORS.get(exp_name, '#CCCCCC'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Eigenvalue Metrics by Experiment Type', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/boxplot_by_experiment.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/plots/boxplot_by_experiment.png")
plt.close()

# Plot 3: Heatmap of correlations
print("  Creating correlation heatmap...")
corr_matrix = df[['spread', 'condition', 'participation_ratio', 
                   'rownorm_advantage', 'rownorm_acc']].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

plt.title('Correlation Matrix: Eigenvalue Metrics vs. Performance', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/plots/correlation_heatmap.png")
plt.close()

# Plot 4: Dataset comparison table
print("  Creating dataset comparison table...")
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Prepare summary table
summary_data = []
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    
    # Average across experiments
    summary_data.append([
        dataset,
        f"{dataset_data['condition'].mean():.2f}",
        f"{dataset_data['spread'].mean():.4f}",
        f"{dataset_data['participation_ratio'].mean():.1f}",
        f"{dataset_data['rownorm_advantage'].mean():+.2f}%",
        f"{dataset_data['n_eigenvalues_valid'].iloc[0]}" if len(dataset_data) > 0 else "N/A"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Dataset', 'Condition (κ)', 'Spread (Δ)', 'Part. Ratio', 
                          'RowNorm Adv.', 'Valid Eigs'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows alternately
for i in range(1, len(summary_data) + 1):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

plt.title('Dataset Summary: Eigenvalue Metrics and Performance', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(f'{OUTPUT_DIR}/plots/dataset_summary_table.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/plots/dataset_summary_table.png")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================

print(f"\n{'='*80}")
print('[4/4] Generating summary report...')
print(f"{'='*80}")

report = f"""
EIGENVALUE DISTRIBUTION ANALYSIS - SUMMARY REPORT
{'='*80}

OVERVIEW
--------
Total configurations analyzed: {len(results)}
Failed configurations: {len(failed)}
Datasets: {df['dataset'].nunique()}
Experiments: {df['experiment_name'].nunique()}

CORRELATION ANALYSIS
-------------------
"""

for metric, results_dict in correlation_results.items():
    r = results_dict['pearson_r']
    p = results_dict['pearson_p']
    n = results_dict['n_samples']
    
    interpretation = "Strong" if abs(r) > 0.7 and p < 0.01 else \
                    "Moderate" if abs(r) > 0.4 and p < 0.05 else "Weak/None"
    
    report += f"""
{metric.upper().replace('_', ' ')}:
  Pearson r = {r:+.3f} (p = {p:.4f})
  N = {n}
  Interpretation: {interpretation} correlation
"""

report += f"""
{'='*80}

KEY FINDINGS
-----------
"""

# Find extreme cases
max_condition = df.loc[df['condition'].idxmax()] if len(df) > 0 else None
min_condition = df.loc[df['condition'].idxmin()] if len(df) > 0 else None
max_advantage = df.loc[df['rownorm_advantage'].idxmax()] if len(df) > 0 else None
min_advantage = df.loc[df['rownorm_advantage'].idxmin()] if len(df) > 0 else None

if max_condition is not None:
    report += f"""
Highest condition number:
  {max_condition['dataset']} ({max_condition['experiment_name']})
  κ = {max_condition['condition']:.2f}
  RowNorm advantage = {max_condition['rownorm_advantage']:+.2f}%

Lowest condition number:
  {min_condition['dataset']} ({min_condition['experiment_name']})
  κ = {min_condition['condition']:.2f}
  RowNorm advantage = {min_condition['rownorm_advantage']:+.2f}%

Best RowNorm performance:
  {max_advantage['dataset']} ({max_advantage['experiment_name']})
  RowNorm advantage = {max_advantage['rownorm_advantage']:+.2f}%
  κ = {max_advantage['condition']:.2f}

Worst RowNorm performance:
  {min_advantage['dataset']} ({min_advantage['experiment_name']})
  RowNorm advantage = {min_advantage['rownorm_advantage']:+.2f}%
  κ = {min_advantage['condition']:.2f}

{'='*80}

FILES GENERATED
--------------
- {OUTPUT_DIR}/eigenvalue_metrics_raw.csv
- {OUTPUT_DIR}/correlation_results.json
- {OUTPUT_DIR}/plots/correlation_scatter.png
- {OUTPUT_DIR}/plots/boxplot_by_experiment.png
- {OUTPUT_DIR}/plots/correlation_heatmap.png
- {OUTPUT_DIR}/plots/dataset_summary_table.png
- {OUTPUT_DIR}/summary_report.txt

{'='*80}
"""

# Save report
with open(f'{OUTPUT_DIR}/summary_report.txt', 'w') as f:
    f.write(report)

print(report)
print(f"✓ Summary report saved: {OUTPUT_DIR}/summary_report.txt")

print(f"\n{'='*80}")
print('ANALYSIS COMPLETE')
print(f"{'='*80}")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"Review plots in: {OUTPUT_DIR}/plots/")
print(f"{'='*80}\n")