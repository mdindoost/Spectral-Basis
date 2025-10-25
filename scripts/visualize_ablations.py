"""
Extended Ablation Visualization
================================

Creates publication-quality figures for extended ablation results.

Usage:
    python visualize_ablations.py [dataset] [--split-type fixed|random]
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

EXPERIMENT_NAMES = {
    'standard_X_scaled': '(a) X\n+Scaled',
    'standard_V_scaled': '(b) U\n+Scaled',
    'standard_V_unscaled': "(b') U\n+Unscaled",
    'standard_V_weighted': "(b'') U\n+Weighted",
    'rownorm_V': '(c) U\n+RowNorm'
}

EXPERIMENT_COLORS = {
    'standard_X_scaled': '#1f77b4',      # Blue
    'standard_V_scaled': '#ff7f0e',      # Orange
    'standard_V_unscaled': '#d62728',    # Red
    'standard_V_weighted': '#9467bd',    # Purple
    'rownorm_V': '#2ca02c'               # Green
}

def load_results(dataset, split_type='fixed_splits'):
    """Load results for a dataset"""
    base_path = f'results/investigation2_extended_ablations/{dataset}/{split_type}'
    metrics_file = f'{base_path}/metrics/results_aggregated.json'
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"No results found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_comprehensive_figure(dataset, split_type='fixed_splits', save_path=None):
    """Create comprehensive 6-panel figure"""
    results = load_results(dataset, split_type)
    agg = results['aggregated_results']
    pca = results['pca_results']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Test Accuracy Bar Chart (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    exp_keys = list(EXPERIMENT_NAMES.keys())
    means = [agg[k]['test_acc']['mean'] * 100 for k in exp_keys]
    stds = [agg[k]['test_acc']['std'] * 100 for k in exp_keys]
    colors = [EXPERIMENT_COLORS[k] for k in exp_keys]
    names = [EXPERIMENT_NAMES[k] for k in exp_keys]
    
    x = np.arange(len(exp_keys))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Test Accuracy Comparison', fontsize=13, fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Baseline (a)')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + std + 0.3,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.legend(fontsize=9)
    
    # Panel 2: Effect Size Decomposition (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    
    a_mean = agg['standard_X_scaled']['test_acc']['mean']
    b_mean = agg['standard_V_scaled']['test_acc']['mean']
    bp_mean = agg['standard_V_unscaled']['test_acc']['mean']
    bpp_mean = agg['standard_V_weighted']['test_acc']['mean']
    c_mean = agg['rownorm_V']['test_acc']['mean']
    
    effect_names = ['Dir B\n(a→b)', 'Scaling\n(b→b\')', 'Eigenval\n(b→b\'\')', 'Dir A\n(b→c)']
    effects = [
        (b_mean - a_mean) * 100,
        (bp_mean - b_mean) * 100,
        (bpp_mean - b_mean) * 100,
        (c_mean - b_mean) * 100
    ]
    effect_colors = ['#ff7f0e' if e < 0 else '#2ca02c' for e in effects]
    
    bars = ax2.bar(range(len(effect_names)), effects, color=effect_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_ylabel('Effect Size (pp)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Effect Decomposition', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xticks(range(len(effect_names)))
    ax2.set_xticklabels(effect_names, fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, eff in zip(bars, effects):
        y_pos = bar.get_height() + (0.15 if eff > 0 else -0.15)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{eff:+.2f}', ha='center',
                va='bottom' if eff > 0 else 'top', fontsize=10, fontweight='bold')
    
    # Panel 3: PCA Eigenvalue Spectrum (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    eigvals_X = np.array(pca['X']['eigenvalues'])
    eigvals_U = np.array(pca['U']['eigenvalues'])
    k_plot = min(50, len(eigvals_X))
    
    ax3.semilogy(range(1, k_plot+1), eigvals_X[:k_plot], 'o-', label='X (raw)', linewidth=2, markersize=4, alpha=0.8)
    ax3.semilogy(range(1, k_plot+1), eigvals_U[:k_plot], 's-', label='U (restricted)', linewidth=2, markersize=4, alpha=0.8)
    ax3.set_xlabel('Principal Component Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Eigenvalue (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('C. PCA Eigenvalue Spectrum', fontsize=13, fontweight='bold', loc='left')
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 4: Cumulative Variance (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    cumsum_X = np.array(pca['X']['eigenvalue_cumsum'])
    cumsum_U = np.array(pca['U']['eigenvalue_cumsum'])
    
    ax4.plot(range(1, len(cumsum_X[:k_plot])+1), cumsum_X[:k_plot], linewidth=2.5, label='X (raw)', alpha=0.8)
    ax4.plot(range(1, len(cumsum_U[:k_plot])+1), cumsum_U[:k_plot], linewidth=2.5, label='U (restricted)', alpha=0.8)
    ax4.axhline(0.9, color='red', linestyle='--', alpha=0.5, linewidth=2, label='90% variance')
    ax4.axhline(0.99, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='99% variance')
    ax4.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax4.set_title('D. Cumulative Variance', fontsize=13, fontweight='bold', loc='left')
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim([0, 1.05])
    
    # Panel 5: Geometric Properties Comparison (middle-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    
    X_props = pca['X']
    U_props = pca['U']
    
    properties = ['Eff. Rank\nRatio', 'log₁₀(Cond.)', 'Isotropy\n(CV)']
    x_vals = [
        X_props['effective_rank_ratio'],
        np.log10(X_props['condition_number']),
        X_props['coefficient_variation']
    ]
    u_vals = [
        U_props['effective_rank_ratio'],
        np.log10(U_props['condition_number']),
        U_props['coefficient_variation']
    ]
    
    x_pos = np.arange(len(properties))
    width = 0.35
    ax5.bar(x_pos - width/2, x_vals, width, label='X (raw)', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.bar(x_pos + width/2, u_vals, width, label='U (restricted)', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax5.set_title('E. Geometric Properties', fontsize=13, fontweight='bold', loc='left')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(properties, fontsize=9)
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel 6: Training Dynamics (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    for exp_key in exp_keys:
        val_accs_mean = np.array(agg[exp_key]['val_accs']['mean'])
        val_accs_std = np.array(agg[exp_key]['val_accs']['std'])
        epochs = range(len(val_accs_mean))
        
        ax6.plot(epochs, val_accs_mean, label=EXPERIMENT_NAMES[exp_key], 
                linewidth=2, alpha=0.8, color=EXPERIMENT_COLORS[exp_key])
        ax6.fill_between(epochs, val_accs_mean - val_accs_std, val_accs_mean + val_accs_std,
                        alpha=0.15, color=EXPERIMENT_COLORS[exp_key])
    
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax6.set_title('F. Training Dynamics', fontsize=13, fontweight='bold', loc='left')
    ax6.legend(fontsize=8, framealpha=0.9, loc='lower right')
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 7: Convergence Metrics Table (bottom, spans 3 columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Experiment', 'Test Acc', 'Speed-90', 'Speed-95', 'Speed-99', 'AUC', 'Conv. Rate']
    
    for exp_key in exp_keys:
        test_acc = agg[exp_key]['test_acc']
        conv = agg[exp_key]['convergence_metrics']
        
        row = [
            EXPERIMENT_NAMES[exp_key].replace('\n', ' '),
            f"{test_acc['mean']*100:.2f}±{test_acc['std']*100:.2f}%",
            f"{conv['speed_to_90']['mean']:.1f}",
            f"{conv['speed_to_95']['mean']:.1f}",
            f"{conv['speed_to_99']['mean']:.1f}",
            f"{conv['auc']['mean']:.3f}",
            f"{conv['convergence_rate']['mean']:.4f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax7.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style headers
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)+1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax7.text(0.5, 0.95, 'G. Convergence Metrics Summary', 
            ha='center', va='top', fontsize=13, fontweight='bold', transform=ax7.transAxes)
    
    # Main title
    fig.suptitle(f'Extended Ablation Analysis: {dataset.upper()} ({split_type})',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def create_cross_dataset_comparison(split_type='fixed_splits', save_path=None):
    """Create figure comparing effects across datasets"""
    datasets = ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed',
                'wikics', 'amazon-photo', 'amazon-computers',
                'coauthor-cs', 'coauthor-physics']
    
    results_list = []
    valid_datasets = []
    
    for dataset in datasets:
        try:
            results = load_results(dataset, split_type)
            results_list.append(results)
            valid_datasets.append(dataset)
        except FileNotFoundError:
            print(f"Warning: No results for {dataset}")
    
    if len(valid_datasets) == 0:
        print("Error: No results found for any dataset")
        return
    
    # Extract effects
    dir_b_effects = []
    scaling_effects = []
    eigenval_effects = []
    dir_a_effects = []
    
    for results in results_list:
        agg = results['aggregated_results']
        a = agg['standard_X_scaled']['test_acc']['mean']
        b = agg['standard_V_scaled']['test_acc']['mean']
        bp = agg['standard_V_unscaled']['test_acc']['mean']
        bpp = agg['standard_V_weighted']['test_acc']['mean']
        c = agg['rownorm_V']['test_acc']['mean']
        
        dir_b_effects.append((b - a) * 100)
        scaling_effects.append((bp - b) * 100)
        eigenval_effects.append((bpp - b) * 100)
        dir_a_effects.append((c - b) * 100)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Cross-Dataset Effect Comparison ({split_type})', fontsize=16, fontweight='bold')
    
    # Plot each effect
    effect_data = [
        (dir_b_effects, 'Direction B: Basis Change (a→b)', axes[0, 0]),
        (scaling_effects, "Ablation 1: Remove Scaling (b→b')", axes[0, 1]),
        (eigenval_effects, "Ablation 2: Add Eigenvalues (b→b'')", axes[1, 0]),
        (dir_a_effects, 'Direction A: Model Change (b→c)', axes[1, 1])
    ]
    
    for effects, title, ax in effect_data:
        colors = ['#ff7f0e' if e < 0 else '#2ca02c' for e in effects]
        bars = ax.barh(valid_datasets, effects, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Effect Size (pp)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, eff, dataset in zip(bars, effects, valid_datasets):
            x_pos = bar.get_width() + (0.3 if eff > 0 else -0.3)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{eff:+.1f}', ha='left' if eff > 0 else 'right',
                   va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cross-dataset figure saved: {save_path}")
    else:
        plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize extended ablation results')
    parser.add_argument('dataset', nargs='?', default='ogbn-arxiv')
    parser.add_argument('--split-type', choices=['fixed', 'random'], default='fixed')
    parser.add_argument('--cross-dataset', action='store_true', help='Create cross-dataset comparison')
    parser.add_argument('--output', default=None, help='Output file path')
    
    args = parser.parse_args()
    
    split_type = 'fixed_splits' if args.split_type == 'fixed' else 'random_splits'
    
    if args.cross_dataset:
        save_path = args.output or f'results/cross_dataset_ablations_{args.split_type}.png'
        create_cross_dataset_comparison(split_type, save_path)
    else:
        save_path = args.output or f'results/investigation2_extended_ablations/{args.dataset}/{split_type}/plots/comprehensive_figure.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        create_comprehensive_figure(args.dataset, split_type, save_path)

if __name__ == '__main__':
    main()
