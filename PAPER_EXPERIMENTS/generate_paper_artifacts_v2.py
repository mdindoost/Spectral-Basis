#!/usr/bin/env python3
"""
COMPLETE PAPER ARTIFACTS GENERATOR (STANDARD CONVENTION)
=========================================================

Generates ALL tables and figures for the revised paper with standard convention.

New Paper Structure:
  Section 3: Basis Sensitivity + Over-smoothing (NEW 3.3)
  Section 4: Whitening Mechanism (enhanced)
  Section 5: Recovery Limitations (REWRITTEN)
  Section 6: Optimization Dynamics

Usage:
    python generate_paper_artifacts_complete.py --verify    # Check data exists
    python generate_paper_artifacts_complete.py --all       # Generate everything
    python generate_paper_artifacts_complete.py --section3  # Generate Section 3 only
    python generate_paper_artifacts_complete.py --section5  # Generate Section 5 only

Author: Mohammad (with Claude assistance)
Date: 2026-02-22
"""

import os
import sys
import json
import argparse
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'wikics',
    'amazon-computers', 'amazon-photo', 'coauthor-cs', 'coauthor-physics'
]

K_VALUES = [1, 2, 4, 6, 8, 10, 12, 20, 30]

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'PAPER_RESULTS'
OUTPUT_DIR = SCRIPT_DIR / 'PAPER_OUTPUT'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# ============================================================================
# Helper Functions
# ============================================================================

def load_results(dataset, split_type='fixed', component='lcc', k=10):
    """Load results.json for a dataset"""
    path = RESULTS_DIR / f'{dataset}_{split_type}_{component}' / f'k{k}' / 'metrics' / 'results.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_k_sensitivity(dataset, split_type='fixed', component='lcc'):
    """Load exp8_k_sensitivity.json"""
    path = RESULTS_DIR / f'{dataset}_{split_type}_{component}' / 'analytics' / 'exp8_k_sensitivity.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_spectral_analysis(dataset, split_type='fixed', component='lcc', k=10):
    """Load exp2_spectral_analysis.json"""
    path = RESULTS_DIR / f'{dataset}_{split_type}_{component}' / 'analytics' / f'exp2_spectral_analysis_k{k}.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_dynamics(dataset, split_type='fixed', component='lcc', k=10):
    """Load all training_curves/*.json files"""
    pattern = str(RESULTS_DIR / f'{dataset}_{split_type}_{component}' / f'k{k}' / 'training_curves' / 'split*_dynamics.json')
    files = glob(pattern)
    
    if not files:
        return None
    
    all_data = []
    for f in files:
        with open(f) as fp:
            all_data.append(json.load(fp))
    
    return all_data

def latex_table_header(columns, caption, label):
    """Generate LaTeX table header"""
    ncols = len(columns)
    header = f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' + 'c' * (ncols - 1)}}}
\\toprule
"""
    header += ' & '.join(f'\\textbf{{{c}}}' for c in columns) + ' \\\\\n'
    header += '\\midrule\n'
    return header

def latex_table_footer():
    """Generate LaTeX table footer"""
    return """\\bottomrule
\\end{tabular}
\\end{table}
"""

# ============================================================================
# VERIFICATION: Check All Required Data Exists
# ============================================================================

def verify_data_completeness():
    """Verify all required experimental data exists"""
    print("="*80)
    print("DATA COMPLETENESS VERIFICATION")
    print("="*80)
    
    missing = []
    datasets_found = set()
    
    for ds in DATASETS:
        print(f"\n{ds}:")
        
        # Check k=10 results (minimum requirement)
        results_k10 = load_results(ds, 'fixed', 'lcc', 10)
        if results_k10 is None:
            print(f"  ❌ Missing k=10 results")
            missing.append(f"{ds}: k=10 results")
        else:
            print(f"  ✓ k=10 results found")
            datasets_found.add(ds)
        
        # Check k-sensitivity data
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            print(f"  ⚠ Missing k-sensitivity analysis")
        else:
            k_found = len(k_sens['k_sensitivity'])
            print(f"  ✓ k-sensitivity: {k_found} k values")
        
        # Check spectral analysis
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        if spec is None:
            print(f"  ⚠ Missing spectral analysis")
        else:
            print(f"  ✓ Spectral analysis found")
        
        # Check dynamics
        dyn = load_dynamics(ds, 'fixed', 'lcc', 10)
        if dyn is None:
            print(f"  ⚠ Missing dynamics data")
        else:
            print(f"  ✓ Dynamics: {len(dyn)} splits")
    
    print("\n" + "="*80)
    print(f"SUMMARY: {len(datasets_found)}/{len(DATASETS)} datasets have minimum required data")
    
    if missing:
        print(f"\nMISSING DATA:")
        for m in missing:
            print(f"  - {m}")
        print(f"\n⚠ Run master_training.py and master_analytics.py for missing datasets")
        return False
    else:
        print(f"\n✓ ALL REQUIRED DATA PRESENT - Ready to generate artifacts!")
        return True

# ============================================================================
# SECTION 3: BASIS SENSITIVITY + OVER-SMOOTHING
# ============================================================================

def generate_table_3_1_part_a():
    """Table 3.1: Part A at k=10 (Fixed Splits)"""
    print('Generating Table 3.1: Part A (Fixed Splits, k=10)...')
    
    rows = []
    for ds in DATASETS:
        data = load_results(ds, 'fixed', 'lcc', 10)
        if data is None:
            continue
        
        part_a = data['framework_analysis']['part_a_pp']
        
        # Get std from experiments
        sgc = data['experiments']['sgc_mlp_baseline']
        restr = data['experiments']['restricted_standard_mlp']
        
        # Error propagation for difference
        std = np.sqrt((sgc['test_acc_std']*100)**2 + (restr['test_acc_std']*100)**2)
        
        rows.append({
            'dataset': ds,
            'part_a': part_a,
            'std': std
        })
    
    # Generate LaTeX
    latex = latex_table_header(
        ['Dataset', 'Part A (pp)', 'Std (pp)'],
        'Part A: Basis Sensitivity (Fixed Splits, k=10)',
        'tab:part_a_fixed'
    )
    
    for row in rows:
        latex += f"{row['dataset']:<20} & {row['part_a']:>+7.2f} & {row['std']:>6.2f} \\\\\n"
    
    latex += latex_table_footer()
    
    output_path = TABLES_DIR / 'table_3_1_part_a_fixed.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f'  ✓ Saved: {output_path}')

def generate_figure_3_1_part_a_barchart():
    """Figure 3.1: Part A Bar Chart"""
    print('Generating Figure 3.1: Part A Bar Chart...')
    
    part_a_values = []
    dataset_names = []
    
    for ds in DATASETS:
        data = load_results(ds, 'fixed', 'lcc', 10)
        if data is None:
            continue
        
        part_a_values.append(data['framework_analysis']['part_a_pp'])
        dataset_names.append(ds)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in part_a_values]
    bars = ax.bar(dataset_names, part_a_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_ylabel('Part A (pp)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax.set_title('Basis Sensitivity Across Datasets (k=10, Fixed Splits)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'figure_3_1_part_a_barchart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: {output_path}')

def generate_figure_3_2_part_a_vs_k():
    """Figure 3.2: Part A vs k (CRITICAL - Shows over-smoothing!)"""
    print('Generating Figure 3.2: Part A vs k (Over-smoothing Analysis)...')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette for 9 datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))
    
    crossover_datasets = []
    
    for idx, ds in enumerate(DATASETS):
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            continue
        
        k_vals = []
        part_a_vals = []
        
        for row in k_sens['k_sensitivity']:
            k_vals.append(row['k'])
            part_a_vals.append(row['part_a'])
        
        # Check for crossover (any sign change, including through zero, in either direction)
        has_crossover = False
        crossover_k = None
        for i in range(len(part_a_vals)-1):
            if part_a_vals[i] * part_a_vals[i+1] < 0:
                has_crossover = True
                crossover_k = (k_vals[i] + k_vals[i+1]) / 2
                crossover_datasets.append((ds, crossover_k))
                break
        
        # Plot with thicker line if crossover exists
        linewidth = 3 if has_crossover else 2
        linestyle = '-' if has_crossover else '--'
        ax.plot(k_vals, part_a_vals, 'o-', linewidth=linewidth, linestyle=linestyle,
               label=ds, markersize=7, color=colors[idx])
        
        # Mark crossover point
        if has_crossover and crossover_k:
            ax.scatter([crossover_k], [0], s=200, marker='X', 
                      color=colors[idx], edgecolor='black', linewidth=2, zorder=10)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Diffusion Steps (k)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Part A (pp)', fontsize=18, fontweight='bold')
    ax.set_title('Basis Sensitivity vs Diffusion Depth: Over-Smoothing Transition', 
                fontsize=20, fontweight='bold', pad=20)
    ax.legend(fontsize=11, ncol=2, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # Read ylim AFTER tight_layout so limits are final; place labels at 20/80% of k-range
    fig.canvas.draw()
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    x_left  = xlo + 0.15 * (xhi - xlo)
    x_right = xlo + 0.80 * (xhi - xlo)
    ax.text(x_left,  ylo + 0.85 * (yhi - ylo), 'SGC Dominates\n(Low Smoothing)',
            fontsize=13, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(x_right, ylo + 0.10 * (yhi - ylo), 'Restricted Eigenvectors Win\n(Over-Smoothing Zone)',
            fontsize=13, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    output_path = FIGURES_DIR / 'figure_3_2_part_a_vs_k.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: {output_path}')
    if crossover_datasets:
        print(f'  ✓ Crossover detected: {", ".join([f"{d} (k≈{k:.0f})" for d, k in crossover_datasets])}')

def generate_table_3_3_crossover_analysis():
    """Table 3.3: Over-Smoothing Crossover Analysis"""
    print('Generating Table 3.3: Crossover Point Analysis...')
    
    rows = []
    
    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            continue
        
        k_vals = []
        part_a_vals = []
        
        for row in k_sens['k_sensitivity']:
            k_vals.append(row['k'])
            part_a_vals.append(row['part_a'])
        
        # Find crossover — any sign change (consistent with Fig 3.2 detection)
        crossover_range = None
        for i in range(len(part_a_vals)-1):
            if part_a_vals[i] * part_a_vals[i+1] < 0:
                crossover_range = f'{k_vals[i]}-{k_vals[i+1]}'
                break
        
        rows.append({
            'dataset': ds,
            'part_a_low_k': part_a_vals[0] if part_a_vals else None,  # k=1
            'part_a_high_k': part_a_vals[-1] if part_a_vals else None,  # k=30
            'crossover': crossover_range or 'No crossover',
            'trend': 'Crossover' if crossover_range else 'Monotonic'
        })
    
    # Generate LaTeX
    latex = latex_table_header(
        ['Dataset', 'Part A (k=1)', 'Part A (k=30)', 'Crossover Range', 'Pattern'],
        'Over-Smoothing Transition Analysis',
        'tab:crossover_analysis'
    )
    
    for row in rows:
        if row['part_a_low_k'] is not None:
            latex += (f"{row['dataset']:<20} & {row['part_a_low_k']:>+7.2f}pp & "
                     f"{row['part_a_high_k']:>+7.2f}pp & {row['crossover']:<15} & "
                     f"{row['trend']} \\\\\n")
    
    latex += latex_table_footer()
    
    output_path = TABLES_DIR / 'table_3_3_crossover_analysis.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# SECTION 4: WHITENING MECHANISM
# ============================================================================

def generate_figure_4_1_singular_values():
    """Figure 4.1: Singular Value Comparison (U vs X_diffused)"""
    print('Generating Figure 4.1: Singular Value Spectra...')
    
    datasets_to_show = ['cora', 'citeseer', 'amazon-computers']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, ds in enumerate(datasets_to_show):
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        if spec is None:
            continue
        
        sv_U = spec['analysis']['singular_values_U_top20']
        sv_X = spec['analysis']['singular_values_X_top20']
        
        ax = axes[idx]
        ax.plot(range(1, len(sv_U)+1), sv_U, 'o-', label='U (Restricted Eigenvectors)', 
               linewidth=2.5, markersize=7, color='#1f77b4')
        ax.plot(range(1, len(sv_X)+1), sv_X, 's-', label='X (SGC Diffused)', 
               linewidth=2.5, markersize=7, color='#ff7f0e')
        
        ax.set_yscale('log')
        ax.set_title(f'{ds.title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Singular Value Index', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Singular Value (log scale)', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')
    
    plt.suptitle('Singular Value Spectra: Eigenvectors vs Diffused Features', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'figure_4_1_singular_values.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: {output_path}')

def generate_table_4_1_spectral_properties():
    """Table 4.1: Spectral Properties (Condition numbers, variance ratios)"""
    print('Generating Table 4.1: Spectral Properties...')
    
    rows = []
    
    for ds in DATASETS:
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        if spec is None:
            continue
        
        analysis = spec['analysis']
        var_analysis = analysis['variance_analysis']
        
        rows.append({
            'dataset': ds,
            'cond_U': analysis['condition_number_U'],
            'cond_X': analysis['condition_number_X'],
            'var_ratio_U': var_analysis['U_variance_ratio'],
            'var_ratio_X': var_analysis['X_variance_ratio'],
            'var_reduction': var_analysis['variance_ratio_reduction']
        })
    
    # Generate LaTeX
    latex = latex_table_header(
        ['Dataset', 'κ(U)', 'κ(X)', 'Var Ratio U', 'Var Ratio X', 'Reduction'],
        'Spectral Properties: Condition Numbers and Variance Analysis (k=10)',
        'tab:spectral_properties'
    )
    
    for row in rows:
        latex += (f"{row['dataset']:<20} & {row['cond_U']:>8.2f} & {row['cond_X']:>8.2f} & "
                 f"{row['var_ratio_U']:>8.2f} & {row['var_ratio_X']:>8.2f} & "
                 f"{row['var_reduction']:>8.2f}× \\\\\n")
    
    latex += latex_table_footer()
    
    output_path = TABLES_DIR / 'table_4_1_spectral_properties.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# SECTION 5: RECOVERY LIMITATIONS (NEW - REWRITTEN)
# ============================================================================

def generate_table_5_1_magnitude_cascade():
    """Table 5.1: Magnitude Recovery Cascade (CRITICAL)"""
    print('Generating Table 5.1: Magnitude Recovery Cascade...')
    
    rows = []
    
    for ds in DATASETS:
        data = load_results(ds, 'fixed', 'lcc', 10)
        if data is None:
            continue
        
        exps = data['experiments']
        
        std_acc = exps['restricted_standard_mlp']['test_acc_mean'] * 100
        rn_acc = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100
        
        part_b = rn_acc - std_acc
        
        # Magnitude-aware methods — check key presence, not accuracy > 0
        logmag_present = 'log_magnitude' in exps
        dual_present   = 'dual_stream' in exps
        logmag_acc = exps['log_magnitude']['test_acc_mean'] * 100 if logmag_present else None
        dual_acc   = exps['dual_stream']['test_acc_mean'] * 100   if dual_present   else None

        part_b5_logmag = (logmag_acc - rn_acc) if logmag_acc is not None else None
        part_b5_dual   = (dual_acc   - rn_acc) if dual_acc   is not None else None

        # Recovery rate: how much logmag recovers when RowNorm hurt (part_b < -0.1 pp)
        # Note: recovery_rate can be negative if logmag also hurts (worse than RowNorm)
        if part_b < -0.1 and part_b5_logmag is not None:
            recovery_rate = part_b5_logmag / abs(part_b) * 100
        else:
            recovery_rate = None

        rows.append({
            'dataset': ds,
            'std': std_acc,
            'rownorm': rn_acc,
            'part_b': part_b,
            'logmag': logmag_acc,
            'part_b5_logmag': part_b5_logmag,
            'dual': dual_acc,
            'part_b5_dual': part_b5_dual,
            'recovery_rate': recovery_rate
        })
    
    # Generate LaTeX
    latex = latex_table_header(
        ['Dataset', 'Std', 'RowNorm', 'Part B', 'Log-Mag', 'ΔLog-Mag', 'Recovery'],
        'Magnitude Recovery Cascade: Std → RowNorm → Log-Magnitude (k=10)',
        'tab:magnitude_cascade'
    )
    
    for row in rows:
        recovery_str = f"{row['recovery_rate']:.0f}\\%" if row['recovery_rate'] is not None else 'N/A'
        logmag_delta = f"{row['part_b5_logmag']:+.2f}pp" if row['part_b5_logmag'] is not None else 'N/A'
        logmag_str   = f"{row['logmag']:.2f}\\%"         if row['logmag'] is not None else 'N/A'
        latex += (f"{row['dataset']:<20} & {row['std']:>6.2f}\\% & {row['rownorm']:>6.2f}\\% & "
                 f"{row['part_b']:>+7.2f}pp & "
                 f"{logmag_str:>8} & "
                 f"{logmag_delta:>8} & {recovery_str:>8} \\\\\n")
    
    latex += latex_table_footer()
    
    output_path = TABLES_DIR / 'table_5_1_magnitude_cascade.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f'  ✓ Saved: {output_path}')

def generate_figure_5_1_recovery_cascade():
    """Figure 5.1: Recovery Cascade Visualization"""
    print('Generating Figure 5.1: Recovery Cascade Visualization...')
    
    datasets_to_show = ['cora', 'citeseer', 'amazon-computers']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for idx, ds in enumerate(datasets_to_show):
        data = load_results(ds, 'fixed', 'lcc', 10)
        if data is None:
            continue
        
        exps = data['experiments']
        
        std_acc = exps['restricted_standard_mlp']['test_acc_mean'] * 100
        rn_acc  = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100
        logmag_acc = (exps['log_magnitude']['test_acc_mean'] * 100
                      if 'log_magnitude' in exps else None)
        dual_acc   = (exps['dual_stream']['test_acc_mean'] * 100
                      if 'dual_stream' in exps else None)

        # Only include methods that were actually run
        method_names = ['Std', 'RowNorm']
        accs = [std_acc, rn_acc]
        method_colors = ['gray', '#d62728' if rn_acc < std_acc else '#2ca02c']
        if logmag_acc is not None:
            method_names.append('Log-Mag')
            accs.append(logmag_acc)
            method_colors.append('#2ca02c' if logmag_acc > rn_acc else 'orange')
        if dual_acc is not None:
            method_names.append('Dual')
            accs.append(dual_acc)
            method_colors.append('#2ca02c' if dual_acc > rn_acc else 'orange')

        ax = axes[idx]

        bars = ax.bar(method_names, accs, color=method_colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # Annotate deltas (only for methods present)
        ax.annotate(f'{rn_acc - std_acc:+.1f}pp',
                    xy=(0.5, (std_acc + rn_acc) / 2), fontsize=11, ha='center',
                    fontweight='bold')
        if logmag_acc is not None:
            lm_idx = method_names.index('Log-Mag')
            ax.annotate(f'{logmag_acc - rn_acc:+.1f}pp',
                        xy=(lm_idx - 0.5, (rn_acc + logmag_acc) / 2), fontsize=11,
                        ha='center', fontweight='bold')

        ax.set_title(f'{ds.title()}', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
        # Use only present (non-None) acc values for ylim so a missing method (acc=0)
        # doesn't force y-axis to start at 0.
        if accs:
            ax.set_ylim(min(accs) * 0.95, max(accs) * 1.05)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=11)
    
    plt.suptitle('Magnitude Recovery Cascade: StandardMLP → RowNorm → Magnitude-Aware', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'figure_5_1_recovery_cascade.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: {output_path}')

def generate_table_5_2_spectral_normalization():
    """Table 5.2: Spectral Normalization α Sweep"""
    print('Generating Table 5.2: Spectral Normalization Analysis...')
    
    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    rows = []
    
    for ds in DATASETS:
        data = load_results(ds, 'fixed', 'lcc', 10)
        if data is None:
            continue
        
        exps = data['experiments']
        rn_baseline = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100

        # Collect test_acc for all alpha values (for display only — not for selection).
        alpha_accs = {}
        for alpha in alphas:
            key = f'spectral_rownorm_alpha{alpha}'
            if key in exps:
                alpha_accs[alpha] = exps[key]['test_acc_mean'] * 100

        # best_alpha was selected on val_acc during training (MAJOR-1 fix) and is
        # stored in framework_analysis.  Read it here to avoid re-selecting on test.
        fa         = data.get('framework_analysis', {})
        best_alpha = fa.get('best_spectral_alpha')          # chosen on val set
        best_acc   = alpha_accs.get(best_alpha) if best_alpha is not None else None
        improvement = (best_acc - rn_baseline)  if best_acc  is not None else None
        
        rows.append({
            'dataset': ds,
            'rn_baseline': rn_baseline,
            'best_alpha': best_alpha,
            'best_acc': best_acc if best_alpha is not None else None,
            'improvement': improvement,
            'alpha_accs': alpha_accs
        })
    
    # Generate LaTeX
    latex = latex_table_header(
        ['Dataset', 'RowNorm', 'Best α', 'Best Acc', 'Δ vs RowNorm'],
        'Spectral Normalization: Optimal α Selection (k=10)',
        'tab:spectral_normalization'
    )
    
    for row in rows:
        best_alpha_str = f"{row['best_alpha']:.1f}" if row['best_alpha'] is not None else 'N/A'
        best_acc_str    = f"{row['best_acc']:.2f}\\%"       if row['best_acc']    is not None else 'N/A'
        improvement_str = f"{row['improvement']:+.2f}pp"    if row['improvement'] is not None else 'N/A'
        
        latex += (f"{row['dataset']:<20} & {row['rn_baseline']:>6.2f}\\% & "
                 f"{best_alpha_str:>6} & {best_acc_str:>8} & {improvement_str:>10} \\\\\n")
    
    latex += latex_table_footer()
    
    output_path = TABLES_DIR / 'table_5_2_spectral_normalization.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# SECTION 6: OPTIMIZATION DYNAMICS
# ============================================================================

def _aggregate_dynamics(all_dynamics, method_key):
    """Aggregate val_acc and train_loss curves across all splits/seeds for one method."""
    val_accs, train_losses = [], []
    for split_data in all_dynamics:
        for record in split_data:
            if record.get('method') == method_key:
                val_accs.append(record.get('val_acc', []))
                train_losses.append(record.get('train_loss', []))
    if not val_accs:
        return None
    min_len      = min(len(c) for c in val_accs)
    min_len_loss = min(len(c) for c in train_losses)
    va  = np.array([c[:min_len]      for c in val_accs])
    tl  = np.array([c[:min_len_loss] for c in train_losses])
    return {
        'val_acc_mean':    va.mean(axis=0),
        'val_acc_std':     va.std(axis=0),
        'train_loss_mean': tl.mean(axis=0),
        'train_loss_std':  tl.std(axis=0),
    }


def generate_figure_6_1_training_curves():
    """Figure 6.1: Training curves for every dataset that has dynamics data.

    Produces:
      - figure_6_1_{dataset}_training_curves.pdf  (one per dataset)
      - figure_6_1_all_training_curves.pdf         (combined multi-panel)
    """
    METHOD_KEYS = {
        'sgc_mlp_baseline':       ('SGC+MLP',             '#1f77b4'),
        'restricted_standard_mlp':('Restricted+Std',       '#ff7f0e'),
        'restricted_rownorm_mlp': ('Restricted+RowNorm',   '#2ca02c'),
    }

    datasets_with_data = []
    for ds in DATASETS:
        all_dynamics = load_dynamics(ds, 'fixed', 'lcc', 10)
        if all_dynamics is None:
            print(f'  ⚠ No dynamics data for {ds}, skipping')
            continue
        datasets_with_data.append((ds, all_dynamics))

    if not datasets_with_data:
        print('  ⚠ No dynamics data found for any dataset')
        return

    # ── Per-dataset figures ───────────────────────────────────────────────────
    for ds, all_dynamics in datasets_with_data:
        print(f'  Generating training curves: {ds}...')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for method_key, (label, color) in METHOD_KEYS.items():
            stats = _aggregate_dynamics(all_dynamics, method_key)
            if stats is None:
                continue
            ep_acc  = range(len(stats['val_acc_mean']))
            ep_loss = range(len(stats['train_loss_mean']))

            ax1.plot(ep_acc, stats['val_acc_mean'], linewidth=2.5, label=label, color=color)
            ax1.fill_between(ep_acc,
                             stats['val_acc_mean'] - stats['val_acc_std'],
                             stats['val_acc_mean'] + stats['val_acc_std'],
                             alpha=0.2, color=color)

            ax2.plot(ep_loss, stats['train_loss_mean'], linewidth=2.5, label=label, color=color)
            ax2.fill_between(ep_loss,
                             stats['train_loss_mean'] - stats['train_loss_std'],
                             stats['train_loss_mean'] + stats['train_loss_std'],
                             alpha=0.2, color=color)

        ds_title = ds.title()
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        ax1.set_title(f'{ds_title}: Validation Accuracy', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
        ax2.set_title(f'{ds_title}: Training Loss', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        out = FIGURES_DIR / f'figure_6_1_{ds}_training_curves.pdf'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    ✓ Saved: {out}')

    # ── Combined multi-panel figure (val_acc only, one row per dataset) ───────
    n = len(datasets_with_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (ds, all_dynamics) in zip(axes, datasets_with_data):
        for method_key, (label, color) in METHOD_KEYS.items():
            stats = _aggregate_dynamics(all_dynamics, method_key)
            if stats is None:
                continue
            ep = range(len(stats['val_acc_mean']))
            ax.plot(ep, stats['val_acc_mean'], linewidth=2, label=label, color=color)
            ax.fill_between(ep,
                            stats['val_acc_mean'] - stats['val_acc_std'],
                            stats['val_acc_mean'] + stats['val_acc_std'],
                            alpha=0.15, color=color)
        ax.set_title(ds.title(), fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.08))
    plt.suptitle('Optimization Dynamics: Validation Accuracy (k=10, Fixed Splits)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_all = FIGURES_DIR / 'figure_6_1_all_training_curves.pdf'
    plt.savefig(out_all, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved combined: {out_all}')


def generate_table_6_1_convergence_speed():
    """Table 6.1: Convergence Speed Metrics — all datasets with dynamics data."""
    print('Generating Table 6.1: Convergence Speed (all datasets)...')

    METHOD_KEYS = {
        'sgc_mlp_baseline':        'SGC+MLP',
        'restricted_standard_mlp': 'Restricted+Std',
        'restricted_rownorm_mlp':  'Restricted+RowNorm',
    }

    def _agg_speed(all_dynamics, method_key):
        speed_90, speed_95, speed_99, aucs = [], [], [], []
        for split_data in all_dynamics:
            for record in split_data:
                if record.get('method') == method_key:
                    # speed_to_* is None when the model never hit the threshold —
                    # exclude those runs from the mean rather than biasing with 0.
                    v90 = record.get('speed_to_90_pct_of_peak')
                    v95 = record.get('speed_to_95_pct_of_peak')
                    v99 = record.get('speed_to_99_pct_of_peak')
                    if v90 is not None: speed_90.append(v90)
                    if v95 is not None: speed_95.append(v95)
                    if v99 is not None: speed_99.append(v99)
                    aucs.append(record.get('auc_normalized', 0) or 0)
        if not speed_90:
            return None
        return {
            'speed_90_mean': np.mean(speed_90), 'speed_90_std': np.std(speed_90),
            'speed_95_mean': np.mean(speed_95), 'speed_95_std': np.std(speed_95),
            'speed_99_mean': np.mean(speed_99), 'speed_99_std': np.std(speed_99),
            'auc_mean':      np.mean(aucs),      'auc_std':      np.std(aucs),
        }

    latex = latex_table_header(
        ['Dataset', 'Method', 'Speed to 90\\%', 'Speed to 95\\%', 'Speed to 99\\%', 'AUC'],
        'Convergence Speed Metrics (Fixed Splits, k=10, \\% of own peak accuracy)',
        'tab:convergence_speed'
    )

    found_any = False
    for ds in DATASETS:
        all_dynamics = load_dynamics(ds, 'fixed', 'lcc', 10)
        if all_dynamics is None:
            continue

        ds_rows = []
        for method_key, method_name in METHOD_KEYS.items():
            stats = _agg_speed(all_dynamics, method_key)
            if stats is None:
                continue
            ds_rows.append((method_name, stats))

        if not ds_rows:
            continue

        if found_any:
            latex += '\\midrule\n'
        found_any = True

        for i, (method_name, stats) in enumerate(ds_rows):
            ds_cell = ds.title() if i == 0 else ''
            latex += (f"{ds_cell:<20} & {method_name:<25} & "
                      f"{stats['speed_90_mean']:.0f}$\\pm${stats['speed_90_std']:.0f} & "
                      f"{stats['speed_95_mean']:.0f}$\\pm${stats['speed_95_std']:.0f} & "
                      f"{stats['speed_99_mean']:.0f}$\\pm${stats['speed_99_std']:.0f} & "
                      f"{stats['auc_mean']:.3f}$\\pm${stats['auc_std']:.3f} \\\\\n")

    if not found_any:
        print('  ⚠ No dynamics data found for any dataset')
        return

    latex += latex_table_footer()

    output_path = TABLES_DIR / 'table_6_1_convergence_speed.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# Main Execution
# ============================================================================

def generate_section_3():
    """Generate all Section 3 artifacts"""
    print("\n" + "="*80)
    print("SECTION 3: BASIS SENSITIVITY + OVER-SMOOTHING")
    print("="*80)
    generate_table_3_1_part_a()
    generate_figure_3_1_part_a_barchart()
    generate_figure_3_2_part_a_vs_k()  # CRITICAL
    generate_table_3_3_crossover_analysis()

def generate_section_4():
    """Generate all Section 4 artifacts"""
    print("\n" + "="*80)
    print("SECTION 4: WHITENING MECHANISM")
    print("="*80)
    generate_figure_4_1_singular_values()
    generate_table_4_1_spectral_properties()

def generate_section_5():
    """Generate all Section 5 artifacts"""
    print("\n" + "="*80)
    print("SECTION 5: RECOVERY LIMITATIONS")
    print("="*80)
    generate_table_5_1_magnitude_cascade()  # CRITICAL
    generate_figure_5_1_recovery_cascade()
    generate_table_5_2_spectral_normalization()

def generate_section_6():
    """Generate all Section 6 artifacts"""
    print("\n" + "="*80)
    print("SECTION 6: OPTIMIZATION DYNAMICS")
    print("="*80)
    generate_figure_6_1_training_curves()
    generate_table_6_1_convergence_speed()

def main():
    parser = argparse.ArgumentParser(description='Generate all paper artifacts')
    parser.add_argument('--verify', action='store_true', help='Verify data completeness')
    parser.add_argument('--all', action='store_true', help='Generate all artifacts')
    parser.add_argument('--section3', action='store_true', help='Generate Section 3 only')
    parser.add_argument('--section4', action='store_true', help='Generate Section 4 only')
    parser.add_argument('--section5', action='store_true', help='Generate Section 5 only')
    parser.add_argument('--section6', action='store_true', help='Generate Section 6 only')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.verify:
        complete = verify_data_completeness()
        if not complete:
            print("\n⚠ WARNING: Some data is missing. Artifacts may be incomplete.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    if args.all or args.section3:
        generate_section_3()
    
    if args.all or args.section4:
        generate_section_4()
    
    if args.all or args.section5:
        generate_section_5()
    
    if args.all or args.section6:
        generate_section_6()
    
    print("\n" + "="*80)
    print("ARTIFACT GENERATION COMPLETE")
    print("="*80)
    print(f"Tables:  {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("\n✓ All artifacts ready for paper assembly!")

if __name__ == '__main__':
    main()