"""
Generate Report and Plots for SGC Truncated Experiments
========================================================

Reads JSON results and creates comprehensive visualizations and summary report.

Usage:
    # For a single configuration
    python generate_sgc_truncated_report.py cora random whole
    
    # For all configurations in a dataset
    python generate_sgc_truncated_report.py cora --all
    
    # For multiple datasets
    python generate_sgc_truncated_report.py cora citeseer ogbn-arxiv --all
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Generate SGC Truncated Reports')
parser.add_argument('datasets', nargs='+', help='Dataset name(s)')
parser.add_argument('--split', type=str, default='random', choices=['random', 'fixed'],
                   help='Split type (ignored if --all)')
parser.add_argument('--component', type=str, default='whole', choices=['whole', 'lcc'],
                   help='Component type (ignored if --all)')
parser.add_argument('--all', action='store_true',
                   help='Generate reports for all configurations found')

args = parser.parse_args()

# ============================================================================
# Helper Functions
# ============================================================================

def load_results(dataset_name, split_type, component_type):
    """Load all k results for a configuration"""
    base_path = Path(f'results/investigation_sgc_truncated_fixed/{dataset_name}_{split_type}_{component_type}')
    
    if not base_path.exists():
        return None
    
    results = {}
    
    # Find all k directories
    for k_dir in sorted(base_path.glob('k*')):
        json_path = k_dir / 'metrics' / 'results.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                k_val = data['k_diffusion']
                results[k_val] = data
    
    return results if results else None

def find_all_configurations(dataset_name):
    """Find all available configurations for a dataset"""
    base_path = Path(f'results/investigation_sgc_truncated_fixed')
    
    configs = []
    pattern = f'{dataset_name}_*_*'
    
    for config_dir in base_path.glob(pattern):
        parts = config_dir.name.split('_')
        if len(parts) >= 3:
            split_type = parts[-2]
            component_type = parts[-1]
            configs.append((split_type, component_type))
    
    return configs

# ============================================================================
# Plotting Functions
# ============================================================================

def create_k_trajectory_plot(results, output_path):
    """Create line plot: Accuracy vs k for each method"""
    
    k_values = sorted(results.keys())
    
    methods = [
        ('sgc_baseline', 'SGC Baseline', 'black', '--', 'o'),
        ('full_logistic', 'Full (all k) Logistic', '#1f77b4', '-', 'o'),
        ('full_mlp', 'Full (all k) MLP', '#ff7f0e', '-', 's'),
        ('nclasses_logistic', 'k=nc Logistic', '#2ca02c', '-', '^'),
        ('nclasses_mlp', 'k=nc MLP', '#d62728', '-', 'v'),
        ('2nclasses_logistic', 'k=2nc Logistic', '#9467bd', '-', 'D'),
        ('2nclasses_mlp', 'k=2nc MLP', '#8c564b', '-', 'P')
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    for method_name, label, color, linestyle, marker in methods:
        accs = []
        stds = []
        for k in k_values:
            acc = results[k]['results'][method_name]['test_acc_mean'] * 100
            std = results[k]['results'][method_name]['test_acc_std'] * 100
            accs.append(acc)
            stds.append(std)
        
        accs = np.array(accs)
        stds = np.array(stds)
        
        ax.plot(k_values, accs, label=label, color=color, linestyle=linestyle, 
                marker=marker, markersize=8, linewidth=2)
        ax.fill_between(k_values, accs - stds, accs + stds, alpha=0.2, color=color)
    
    ax.set_xlabel('Diffusion Steps (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy vs Diffusion Parameter k', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {output_path}')

def create_truncation_comparison_plot(results, output_path):
    """Create bar chart comparing truncation levels at each k"""
    
    k_values = sorted(results.keys())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(k_values))
    width = 0.2
    
    # Logistic Regression
    sgc_accs = [results[k]['results']['sgc_baseline']['test_acc_mean'] * 100 for k in k_values]
    full_log = [results[k]['results']['full_logistic']['test_acc_mean'] * 100 for k in k_values]
    nc_log = [results[k]['results']['nclasses_logistic']['test_acc_mean'] * 100 for k in k_values]
    nc2_log = [results[k]['results']['2nclasses_logistic']['test_acc_mean'] * 100 for k in k_values]
    
    ax1.bar(x - 1.5*width, sgc_accs, width, label='SGC Baseline', color='black', alpha=0.7)
    ax1.bar(x - 0.5*width, full_log, width, label='Full (all k)', color='#1f77b4')
    ax1.bar(x + 0.5*width, nc_log, width, label='k=nc', color='#ff7f0e')
    ax1.bar(x + 1.5*width, nc2_log, width, label='k=2nc', color='#2ca02c')
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Truncation Impact: Logistic Regression', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=np.mean(sgc_accs), color='red', linestyle='--', linewidth=1.5, 
                label='Avg SGC', alpha=0.7)
    
    # MLP
    full_mlp = [results[k]['results']['full_mlp']['test_acc_mean'] * 100 for k in k_values]
    nc_mlp = [results[k]['results']['nclasses_mlp']['test_acc_mean'] * 100 for k in k_values]
    nc2_mlp = [results[k]['results']['2nclasses_mlp']['test_acc_mean'] * 100 for k in k_values]
    
    ax2.bar(x - 1.5*width, sgc_accs, width, label='SGC Baseline', color='black', alpha=0.7)
    ax2.bar(x - 0.5*width, full_mlp, width, label='Full (all k)', color='#1f77b4')
    ax2.bar(x + 0.5*width, nc_mlp, width, label='k=nc', color='#ff7f0e')
    ax2.bar(x + 1.5*width, nc2_mlp, width, label='k=2nc', color='#2ca02c')
    
    ax2.set_xlabel('Diffusion Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Truncation Impact: RowNorm MLP', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k={k}' for k in k_values])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=np.mean(sgc_accs), color='red', linestyle='--', linewidth=1.5, 
                label='Avg SGC', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {output_path}')

def create_heatmap(results, output_path):
    """Create heatmap of all results"""
    
    k_values = sorted(results.keys())
    
    methods = [
        'SGC Baseline',
        'Full Logistic',
        'Full MLP',
        'k=nc Logistic',
        'k=nc MLP',
        'k=2nc Logistic',
        'k=2nc MLP'
    ]
    
    method_keys = [
        'sgc_baseline',
        'full_logistic',
        'full_mlp',
        'nclasses_logistic',
        'nclasses_mlp',
        '2nclasses_logistic',
        '2nclasses_mlp'
    ]
    
    data = []
    for method_key in method_keys:
        row = []
        for k in k_values:
            acc = results[k]['results'][method_key]['test_acc_mean'] * 100
            row.append(acc)
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=data.min(), vmax=data.max())
    
    # Set ticks
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_yticklabels(methods)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(k_values)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Test Accuracy Heatmap Across All Methods', fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {output_path}')

def create_improvement_plot(results, output_path):
    """Create bar chart of improvements over SGC baseline"""
    
    k_values = sorted(results.keys())
    
    # Get optimal k for each method
    methods = {
        'Full Logistic': 'full_logistic',
        'Full MLP': 'full_mlp',
        'k=nc Logistic': 'nclasses_logistic',
        'k=nc MLP': 'nclasses_mlp',
        'k=2nc Logistic': '2nclasses_logistic',
        'k=2nc MLP': '2nclasses_mlp'
    }
    
    best_improvements = {}
    best_k = {}
    
    for method_name, method_key in methods.items():
        improvements = []
        for k in k_values:
            imp = results[k]['improvements'][method_key]
            improvements.append(imp)
        
        best_idx = np.argmax(improvements)
        best_improvements[method_name] = improvements[best_idx]
        best_k[method_name] = k_values[best_idx]
    
    # Sort by improvement
    sorted_methods = sorted(best_improvements.items(), key=lambda x: x[1], reverse=True)
    method_names = [x[0] for x in sorted_methods]
    improvements = [x[1] for x in sorted_methods]
    k_labels = [f'k={best_k[name]}' for name in method_names]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(method_names, improvements, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, imp, k_label) in enumerate(zip(bars, improvements, k_labels)):
        x_pos = imp + (1 if imp > 0 else -1)
        ax.text(x_pos, i, f'{imp:+.1f}pp ({k_label})', 
                va='center', ha='left' if imp > 0 else 'right', fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Improvement over SGC Baseline (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Best Improvement for Each Method (at optimal k)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {output_path}')

# ============================================================================
# Report Generation
# ============================================================================

def generate_summary_table(results, dataset_name, split_type, component_type):
    """Generate text summary table"""
    
    k_values = sorted(results.keys())
    
    # Get metadata
    sample_result = results[k_values[0]]
    num_components = sample_result['num_components']
    num_splits = sample_result['num_splits']
    num_seeds = sample_result['num_seeds']
    
    summary = []
    summary.append('='*80)
    summary.append('SGC TRUNCATED EXPERIMENTS: SUMMARY REPORT')
    summary.append('='*80)
    summary.append(f'Dataset: {dataset_name}')
    summary.append(f'Configuration: {split_type} splits, {component_type} graph')
    summary.append(f'Connected Components: {num_components}')
    summary.append(f'Diffusion k values: {k_values}')
    summary.append(f'Statistical Rigor: {num_splits} splits × {num_seeds} seeds = {num_splits * num_seeds} runs per experiment')
    summary.append('='*80)
    summary.append('')
    
    # Best performance for each method
    summary.append('BEST PERFORMANCE (across all k values):')
    summary.append('-'*80)
    
    methods = [
        ('SGC Baseline', 'sgc_baseline'),
        ('Full + Logistic', 'full_logistic'),
        ('Full + MLP', 'full_mlp'),
        ('k=nc + Logistic', 'nclasses_logistic'),
        ('k=nc + MLP', 'nclasses_mlp'),
        ('k=2nc + Logistic', '2nclasses_logistic'),
        ('k=2nc + MLP', '2nclasses_mlp')
    ]
    
    for method_name, method_key in methods:
        best_acc = -1
        best_k = None
        best_imp = None
        
        for k in k_values:
            acc = results[k]['results'][method_key]['test_acc_mean'] * 100
            if acc > best_acc:
                best_acc = acc
                best_k = k
                if method_key != 'sgc_baseline':
                    best_imp = results[k]['improvements'][method_key]
        
        if method_key == 'sgc_baseline':
            summary.append(f'{method_name:25s}: {best_acc:6.2f}% (k={best_k})')
        else:
            summary.append(f'{method_name:25s}: {best_acc:6.2f}% (k={best_k}, {best_imp:+.2f}pp)')
    
    summary.append('')
    summary.append('='*80)
    summary.append('RESULTS BY DIFFUSION PARAMETER k:')
    summary.append('='*80)
    
    for k in k_values:
        summary.append(f'\nk = {k}:')
        summary.append('-'*40)
        
        sgc_acc = results[k]['results']['sgc_baseline']['test_acc_mean'] * 100
        summary.append(f'  SGC Baseline: {sgc_acc:.2f}%')
        summary.append('')
        
        for method_name, method_key in methods[1:]:  # Skip SGC baseline
            acc = results[k]['results'][method_key]['test_acc_mean'] * 100
            std = results[k]['results'][method_key]['test_acc_std'] * 100
            imp = results[k]['improvements'][method_key]
            
            summary.append(f'  {method_name:25s}: {acc:6.2f}% ± {std:5.2f}% ({imp:+6.2f}pp)')
    
    summary.append('')
    summary.append('='*80)
    summary.append('KEY FINDINGS:')
    summary.append('='*80)
    
    # Analyze results
    full_mlp_imps = [results[k]['improvements']['full_mlp'] for k in k_values]
    best_full_mlp = max(full_mlp_imps)
    
    if best_full_mlp > 2:
        summary.append(f'✓ Full MLP beats SGC baseline (best: +{best_full_mlp:.2f}pp)')
    else:
        summary.append(f'✗ Full MLP does not consistently beat SGC baseline (best: {best_full_mlp:+.2f}pp)')
    
    # Check truncation effect
    nc_mlp_imps = [results[k]['improvements']['nclasses_mlp'] for k in k_values]
    nc2_mlp_imps = [results[k]['improvements']['2nclasses_mlp'] for k in k_values]
    
    if max(nc_mlp_imps) < -5 and max(nc2_mlp_imps) < -5:
        summary.append('✗ Truncation severely degrades performance (all truncated < -5pp)')
    elif max(nc2_mlp_imps) > 0:
        summary.append(f'✓ k=2nc truncation shows promise (best: +{max(nc2_mlp_imps):.2f}pp)')
    else:
        summary.append('~ Mixed truncation results')
    
    summary.append('')
    summary.append('='*80)
    
    return '\n'.join(summary)

# ============================================================================
# Main Execution
# ============================================================================

def process_configuration(dataset_name, split_type, component_type):
    """Process a single configuration"""
    
    print(f'\n{"="*80}')
    print(f'Processing: {dataset_name} ({split_type} splits, {component_type} graph)')
    print(f'{"="*80}')
    
    # Load results
    results = load_results(dataset_name, split_type, component_type)
    
    if results is None:
        print(f'✗ No results found for {dataset_name}_{split_type}_{component_type}')
        return False
    
    print(f'✓ Loaded results for k values: {sorted(results.keys())}')
    
    # Create output directory
    output_dir = Path(f'results/investigation_sgc_truncated_fixed/{dataset_name}_{split_type}_{component_type}/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print('\nGenerating plots...')
    
    create_k_trajectory_plot(results, output_dir / 'k_trajectory.png')
    create_truncation_comparison_plot(results, output_dir / 'truncation_comparison.png')
    create_heatmap(results, output_dir / 'accuracy_heatmap.png')
    create_improvement_plot(results, output_dir / 'improvements_over_sgc.png')
    
    # Generate summary report
    print('\nGenerating summary report...')
    summary_text = generate_summary_table(results, dataset_name, split_type, component_type)
    
    summary_path = output_dir.parent / 'SUMMARY_REPORT.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f'✓ Saved: {summary_path}')
    
    # Print summary to console
    print('\n' + summary_text)
    
    print(f'\n✓ All outputs saved to: {output_dir.parent}/')
    
    return True

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    
    success_count = 0
    total_count = 0
    
    for dataset in args.datasets:
        if args.all:
            # Find all configurations for this dataset
            configs = find_all_configurations(dataset)
            
            if not configs:
                print(f'\n✗ No configurations found for dataset: {dataset}')
                continue
            
            print(f'\nFound {len(configs)} configuration(s) for {dataset}:')
            for split_type, component_type in configs:
                print(f'  - {split_type} splits, {component_type} graph')
            
            # Process each configuration
            for split_type, component_type in configs:
                total_count += 1
                if process_configuration(dataset, split_type, component_type):
                    success_count += 1
        else:
            # Process single configuration
            total_count += 1
            if process_configuration(dataset, args.split, args.component):
                success_count += 1
    
    # Final summary
    print(f'\n{"="*80}')
    print(f'COMPLETE: {success_count}/{total_count} configurations processed successfully')
    print(f'{"="*80}')