#!/usr/bin/env python3
"""
PAPER ARTIFACTS GENERATOR
==========================

Generates all tables and figures for the paper. Results are read from
PAPER_RESULTS/ (produced by master_training.py and master_analytics.py)
and written to PAPER_OUTPUT/tables/ and PAPER_OUTPUT/figures/.

Paper Structure:
  Section 3: Basis Sensitivity + Over-smoothing
    - Table 3.1: Part A (Fixed Splits, k=10)
    - Table 3.2: Part A (Random Splits, k=10)
    - Table 3.3: Crossover Point Analysis (fixed + random)
    - Table Exp8: k-Sensitivity summary (fixed + random)
    - Figure 3.1: Part A Bar Chart (fixed + random)
    - Figure 3.2: Part A vs k (Over-smoothing Transition)

  Section 4: Whitening Mechanism
    - Figure 4.1: Singular Value Spectra (all datasets, 3x3)
    - Table 4.1: Spectral Properties

  Section 5: Recovery Limitations
    - Table 5.1: Magnitude Recovery Cascade
    - Table 5.2: Spectral Normalization Analysis
    - Figure 5.1: Recovery Cascade (all datasets, 3x3)

  Section 6: Optimization Dynamics
    - Figure 6.1: Training Curves (per dataset + combined)
    - Table 6.1: Convergence Speed (fixed + random)

Usage:
    python generate_paper_artifacts_v2.py --all        # Generate everything
    python generate_paper_artifacts_v2.py --section3   # Section 3 only
    python generate_paper_artifacts_v2.py --section4   # Section 4 only
    python generate_paper_artifacts_v2.py --section5   # Section 5 only
    python generate_paper_artifacts_v2.py --section6   # Section 6 only


"""

import os
import sys
import json
import math
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

def generate_table_3_2_part_a_random():
    """Table 3.2: Part A at k=10 (Random Splits)"""
    print('Generating Table 3.2: Part A (Random Splits, k=10)...')

    rows = []
    for ds in DATASETS:
        data = load_results(ds, 'random', 'lcc', 10)
        if data is None:
            continue

        part_a = data['framework_analysis']['part_a_pp']

        sgc = data['experiments']['sgc_mlp_baseline']
        restr = data['experiments']['restricted_standard_mlp']

        std = np.sqrt((sgc['test_acc_std']*100)**2 + (restr['test_acc_std']*100)**2)

        rows.append({
            'dataset': ds,
            'part_a': part_a,
            'std': std
        })

    latex = latex_table_header(
        ['Dataset', 'Part A (pp)', 'Std (pp)'],
        'Part A: Basis Sensitivity (Random Splits, k=10)',
        'tab:part_a_random'
    )

    for row in rows:
        latex += f"{row['dataset']:<20} & {row['part_a']:>+7.2f} & {row['std']:>6.2f} \\\\\n"

    latex += latex_table_footer()

    output_path = TABLES_DIR / 'table_3_2_part_a_random.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'  ✓ Saved: {output_path}')


def generate_figure_3_1_part_a_barchart(split_type='fixed'):
    """Figure 3.1: Part A Bar Chart for a given split type"""
    print(f'Generating Figure 3.1: Part A Bar Chart ({split_type})...')

    part_a_values = []
    dataset_names = []

    for ds in DATASETS:
        data = load_results(ds, split_type, 'lcc', 10)
        if data is None:
            continue
        part_a_values.append(data['framework_analysis']['part_a_pp'])
        dataset_names.append(ds)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#d62728' if v < 0 else '#2ca02c' for v in part_a_values]
    ax.bar(dataset_names, part_a_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_ylabel('Part A (pp)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax.set_title(f'Basis Sensitivity Across Datasets (k=10, {split_type.title()} Splits)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    output_path = FIGURES_DIR / f'figure_3_1_part_a_barchart_{split_type}.pdf'
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

def generate_table_3_3_crossover_analysis(split_type='fixed'):
    """Table 3.3: Over-Smoothing Crossover Analysis (fixed or random splits)."""
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    print(f'Generating Table 3.3: Crossover Point Analysis ({split_label})...')

    rows = []

    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, split_type, 'lcc')
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
            'part_a_low_k':  part_a_vals[0]  if part_a_vals else None,  # k=1
            'part_a_high_k': part_a_vals[-1] if part_a_vals else None,  # k=30
            'crossover': crossover_range or 'No crossover',
            'trend': 'Crossover' if crossover_range else 'Monotonic'
        })

    # Generate LaTeX
    latex = latex_table_header(
        ['Dataset', 'Part A (k=1)', 'Part A (k=30)', 'Crossover Range', 'Pattern'],
        f'Over-Smoothing Transition Analysis ({split_label})',
        f'tab:crossover_analysis_{split_type}'
    )

    for row in rows:
        if row['part_a_low_k'] is not None:
            latex += (f"{row['dataset']:<20} & {row['part_a_low_k']:>+7.2f}pp & "
                     f"{row['part_a_high_k']:>+7.2f}pp & {row['crossover']:<15} & "
                     f"{row['trend']} \\\\\n")

    latex += latex_table_footer()

    output_path = TABLES_DIR / f'table_3_3_crossover_analysis_{split_type}.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# SECTION 3 ADDITIONS: Part B vs k  +  Fisher
# ============================================================================

def generate_figure_3_3_part_b_vs_k():
    """Figure 3.3: Part B vs k — recovery by RowNorm across diffusion depths."""
    print('Generating Figure 3.3: Part B vs k...')

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))

    for idx, ds in enumerate(DATASETS):
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            continue
        k_vals     = [r['k']      for r in k_sens['k_sensitivity'] if r.get('part_b') is not None]
        part_b_vals= [r['part_b'] for r in k_sens['k_sensitivity'] if r.get('part_b') is not None]
        if not k_vals:
            continue
        ax.plot(k_vals, part_b_vals, 'o-', linewidth=2, markersize=6,
                label=ds, color=colors[idx])

    ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Diffusion Steps (k)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Part B (pp)', fontsize=16, fontweight='bold')
    ax.set_title('RowNorm Recovery vs Diffusion Depth (Part B)',
                 fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_3_3_part_b_vs_k.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {output_path}')


def generate_table_3_4_fisher_vs_k(split_type='fixed'):
    """Table 3.4: Fisher score of ||U_i|| at key k values (fixed or random splits)."""
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    print(f'Generating Table 3.4: Fisher vs k ({split_label})...')

    K_SHOW = [1, 2, 4, 6, 8, 10, 12, 20, 30]
    rows = []

    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, split_type, 'lcc')
        if k_sens is None:
            continue
        by_k = {r['k']: r for r in k_sens['k_sensitivity']}
        fisher_at_k = {}
        for k in K_SHOW:
            r = by_k.get(k)
            fisher_at_k[k] = r.get('fisher_score') if r else None
        # Fisher on X_diffused at k=10 for comparison
        r10 = by_k.get(10, {})
        fisher_x = r10.get('fisher_score_X_diffused')
        rows.append({'dataset': ds, 'fisher_at_k': fisher_at_k, 'fisher_x': fisher_x})

    if not rows:
        print(f'  ! No data for split_type={split_type}')
        return

    k_headers = ' & '.join([f'$k={k}$' for k in K_SHOW])
    col_spec = 'l' + 'r' * (len(K_SHOW) + 1)  # +1 for Fisher_X column
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{5pt}')
    lines.append(rf'\caption{{Fisher score of $\|U_i\|$ vs diffusion depth ({split_label}). '
                 r'Fisher $=$ between-class / within-class variance of row norms, '
                 r'computed on training nodes only. '
                 r'Higher values indicate row norms carry more class-discriminative information. '
                 r'$\text{Fisher}_X$ is the same metric on raw diffused features at $k=10$.}')
    lines.append(rf'\label{{tab:fisher_vs_k_{split_type}}}')
    lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')
    lines.append(rf'Dataset & {k_headers} & Fisher$_X$@10 \\')
    lines.append(r'\midrule')
    for row in rows:
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        cells = []
        for k in K_SHOW:
            v = row['fisher_at_k'].get(k)
            cells.append(f'{v:.4f}' if v is not None else r'\textemdash')
        fx = row['fisher_x']
        fx_str = f'{fx:.4f}' if fx is not None else r'\textemdash'
        lines.append(rf'{ds_label} & {" & ".join(cells)} & {fx_str} \\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / f'table_3_4_fisher_vs_k_{split_type}.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


def generate_figure_3_4_fisher_vs_k():
    """Figure 3.4: Fisher score of ||U_i|| vs k — all datasets."""
    print('Generating Figure 3.4: Fisher vs k...')

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))

    for idx, ds in enumerate(DATASETS):
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            continue
        k_vals     = [r['k']            for r in k_sens['k_sensitivity']
                      if r.get('fisher_score') is not None]
        fisher_vals= [r['fisher_score'] for r in k_sens['k_sensitivity']
                      if r.get('fisher_score') is not None]
        if not k_vals:
            continue
        ax.plot(k_vals, fisher_vals, 'o-', linewidth=2, markersize=6,
                label=ds, color=colors[idx])

    ax.set_xlabel('Diffusion Steps (k)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Fisher Score', fontsize=16, fontweight='bold')
    ax.set_title('Fisher Score of $\\|U_i\\|$ vs Diffusion Depth',
                 fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_3_4_fisher_vs_k.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {output_path}')


def generate_figure_3_5_fisher_vs_partb_scatter():
    """Figure 3.5: Fisher score vs Part B recovery at k=10 (cross-dataset scatter).

    Validates: "Fisher diagnostic predicts when magnitude-aware methods help."
    - x-axis: Fisher score of ||U_i|| at k=10 (fixed splits)
    - y-axis: Part B recovery (pp) at k=10
    - Pearson r and p-value shown in annotation box
    - Vertical dashed line at median Fisher separates high/low magnitude regimes
    - Horizontal dashed line at y=0 marks RowNorm neutral point
    """
    from scipy.stats import pearsonr

    print('Generating Figure 3.5: Fisher vs Part B Scatter...')

    points = []
    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            continue
        by_k = {r['k']: r for r in k_sens['k_sensitivity']}
        r10 = by_k.get(10, {})
        fisher = r10.get('fisher_score')
        part_b = r10.get('part_b')
        if fisher is None or part_b is None:
            print(f'  ⚠ Missing fisher or part_b at k=10 for {ds}, skipping')
            continue
        points.append({'dataset': ds, 'fisher': fisher, 'part_b': part_b})

    if len(points) < 3:
        print(f'  ⚠ Only {len(points)} datasets have both fisher and part_b — skipping scatter')
        return

    fishers = np.array([p['fisher'] for p in points])
    part_bs = np.array([p['part_b'] for p in points])
    r, pval = pearsonr(fishers, part_bs)

    # Threshold: median Fisher score
    threshold = float(np.median(fishers))

    fig, ax = plt.subplots(figsize=(9, 7))

    for p in points:
        color = '#2ca02c' if p['part_b'] >= 0 else '#d62728'
        ax.scatter(p['fisher'], p['part_b'], s=120, color=color,
                   edgecolor='black', linewidth=1.2, zorder=5)
        ax.annotate(p['dataset'],
                    xy=(p['fisher'], p['part_b']),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=9, ha='left')

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6,
               label='Part B = 0 (RowNorm neutral)')
    ax.axvline(x=threshold, color='steelblue', linestyle='--', linewidth=1.8,
               alpha=0.8, label=f'Median Fisher = {threshold:.4f}')

    # Pearson annotation
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
    ax.text(0.97, 0.97,
            f'Pearson r = {r:+.3f}\np = {pval:.3f} ({sig})\nn = {len(points)} datasets',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    # Quadrant labels
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.text(threshold + 0.001 * (xhi - xlo), yhi * 0.92,
            'High Fisher\n(magnitude informative)',
            fontsize=8, color='steelblue', ha='left', style='italic')
    ax.text(xlo + 0.01 * (xhi - xlo), yhi * 0.92,
            'Low Fisher\n(magnitude uninformative)',
            fontsize=8, color='gray', ha='left', style='italic')

    ax.set_xlabel('Fisher Score of $\\|U_i\\|$ (k=10)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Part B: RowNorm Recovery (pp, k=10)', fontsize=14, fontweight='bold')
    ax.set_title('Fisher Diagnostic vs RowNorm Recovery\n(Fixed Splits, k=10)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_3_5_fisher_vs_partb_scatter.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {output_path}')
    print(f'  ✓ Pearson r = {r:+.3f}, p = {pval:.4f}, n = {len(points)}, '
          f'Fisher threshold (median) = {threshold:.4f}')


def generate_table_3_5_fisher_partb_data():
    """Table 3.5: Raw data table — Fisher score vs Part B at k=10, all datasets.

    Columns: Dataset | Fisher Score | Part B (pp) | Class (High/Low Fisher)
    Footer row: Pearson r and p-value.
    Threshold = median Fisher score across datasets.
    """
    from scipy.stats import pearsonr

    print('Generating Table 3.5: Fisher vs Part B Data Table...')

    points = []
    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, 'fixed', 'lcc')
        if k_sens is None:
            continue
        by_k = {r['k']: r for r in k_sens['k_sensitivity']}
        r10 = by_k.get(10, {})
        fisher = r10.get('fisher_score')
        part_b = r10.get('part_b')
        if fisher is None or part_b is None:
            print(f'  ⚠ Missing fisher or part_b at k=10 for {ds}, skipping')
            continue
        points.append({'dataset': ds, 'fisher': fisher, 'part_b': part_b})

    if len(points) < 3:
        print(f'  ⚠ Only {len(points)} datasets with data — skipping table')
        return

    fishers = np.array([p['fisher'] for p in points])
    part_bs = np.array([p['part_b'] for p in points])
    r, pval = pearsonr(fishers, part_bs)
    threshold = float(np.median(fishers))

    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{6pt}')
    lines.append(r'\caption{Fisher score of $\|U_i\|$ and RowNorm recovery (Part B) at $k=10$ '
                 r'(fixed splits). Classification uses median Fisher as threshold. '
                 r'Part B $> 0$: RowNorm helps; Part B $< 0$: RowNorm hurts. '
                 r'Pearson $r$ tests linear association between Fisher score and recovery.}')
    lines.append(r'\label{tab:fisher_partb_data}')
    lines.append(r'\begin{tabular}{lrrr}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Fisher Score & Part B (pp) & Fisher Class \\')
    lines.append(r'\midrule')

    for p in points:
        ds_label = p['dataset'].replace('-', r'\mbox{-}')
        cls = 'High' if p['fisher'] >= threshold else 'Low'
        cls_str = rf'\textbf{{{cls}}}' if cls == 'High' else cls
        lines.append(rf"{ds_label} & {p['fisher']:.4f} & {p['part_b']:+.2f} & {cls_str} \\")

    lines.append(r'\midrule')
    lines.append(rf'\multicolumn{{4}}{{l}}{{Median Fisher threshold = {threshold:.4f}; '
                 rf'Pearson $r = {r:+.3f}$, $p = {pval:.3f}$ ({sig}), $n = {len(points)}$}} \\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / 'table_3_5_fisher_partb_data.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')
    print(f'  ✓ Pearson r = {r:+.3f}, p = {pval:.4f} ({sig}), threshold = {threshold:.4f}')


def generate_table_3_6_part_b(split_type='fixed'):
    """Table 3.6 / 3.7: Part B at k=10 — RowNorm recovery across all 9 datasets.

    Columns: Dataset | R+Std (%) | R+RowNorm (%) | Part B (pp) | Std of Part B
    Negative Part B means RowNorm hurts — reported as a finding, not suppressed.
    """
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    tnum = '3.6' if split_type == 'fixed' else '3.7'
    print(f'Generating Table {tnum}: Part B ({split_label}, k=10)...')

    rows = []
    for ds in DATASETS:
        data = load_results(ds, split_type, 'lcc', 10)
        if data is None:
            continue
        exps = data['experiments']
        fa   = data['framework_analysis']

        std_mean = exps['restricted_standard_mlp']['test_acc_mean'] * 100
        std_std  = exps['restricted_standard_mlp']['test_acc_std']  * 100
        rn_mean  = exps['restricted_rownorm_mlp']['test_acc_mean']  * 100
        rn_std   = exps['restricted_rownorm_mlp']['test_acc_std']   * 100
        part_b   = fa['part_b_pp']

        # Error propagation: std of the difference
        part_b_std = float(np.sqrt(std_std ** 2 + rn_std ** 2))

        rows.append({
            'dataset':    ds,
            'std_mean':   std_mean,
            'rn_mean':    rn_mean,
            'part_b':     part_b,
            'part_b_std': part_b_std,
        })

    if not rows:
        print(f'  ⚠ No data for split_type={split_type}')
        return

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{6pt}')
    lines.append(rf'\caption{{Part B: RowNorm recovery at $k=10$ ({split_label}). '
                 r'Part B $=$ Acc(Restricted+RowNorm) $-$ Acc(Restricted+Std). '
                 r'Negative values mean RowNorm \emph{hurts} --- this is a finding, not a failure. '
                 r'Std is propagated from individual method standard deviations across seeds.}')
    lines.append(rf'\label{{tab:part_b_{split_type}}}')
    lines.append(r'\begin{tabular}{lrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & R+Std (\%) & R+RowNorm (\%) & Part B (pp) & Std (pp) \\')
    lines.append(r'\midrule')

    for row in rows:
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        # Bold Part B if magnitude > 1pp in either direction
        pb_fmt = f"{row['part_b']:+.2f}"
        if abs(row['part_b']) > 1.0:
            pb_str = rf'\textbf{{{pb_fmt}}}'
        else:
            pb_str = pb_fmt
        lines.append(rf"{ds_label} & {row['std_mean']:.2f} & {row['rn_mean']:.2f} & "
                     rf"{pb_str} & {row['part_b_std']:.2f} \\")

    lines.append(r'\bottomrule\end{tabular}\end{table}')

    fname = f'table_3_6_part_b_fixed.tex' if split_type == 'fixed' else 'table_3_7_part_b_random.tex'
    output_path = TABLES_DIR / fname
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


# ============================================================================
# SECTION 4: WHITENING MECHANISM
# ============================================================================

def generate_figure_4_1_singular_values():
    """Figure 4.1: Singular Value Comparison (U vs X_diffused) — all datasets, 3×3 grid"""
    print('Generating Figure 4.1: Singular Value Spectra...')

    ncols = 3
    nrows = math.ceil(len(DATASETS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes_flat = axes.flatten()

    plotted = 0
    for idx, ds in enumerate(DATASETS):
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        ax = axes_flat[idx]

        if spec is None:
            ax.set_visible(False)
            continue

        sv_U = spec['analysis']['singular_values_U_top20']
        sv_X = spec['analysis']['singular_values_X_top20']

        ax.plot(range(1, len(sv_U)+1), sv_U, 'o-', label='U (Restricted Eigenvectors)',
                linewidth=2.5, markersize=7, color='#1f77b4')
        ax.plot(range(1, len(sv_X)+1), sv_X, 's-', label='X (SGC Diffused)',
                linewidth=2.5, markersize=7, color='#ff7f0e')

        ax.set_yscale('log')
        ax.set_title(ds, fontsize=13, fontweight='bold')
        ax.set_xlabel('Singular Value Index', fontsize=11)
        if idx % ncols == 0:
            ax.set_ylabel('Singular Value (log scale)', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')
        plotted += 1

    # Hide unused subplots
    for idx in range(len(DATASETS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Singular Value Spectra: Eigenvectors vs Diffused Features',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_4_1_singular_values.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  ✓ Saved: {output_path} ({plotted}/{len(DATASETS)} datasets)')

def generate_table_4_0_dataset_statistics():
    """Table 4.0: Dataset statistics — nodes, edges, avg degree, classes,
    features, d_eff, and edge homophily.

    Homophily = fraction of edges connecting same-class nodes (global graph property).
    d_eff = effective feature dimension after QR decomposition of X_diffused at k=10.
    Loaded from exp2_spectral_analysis_k10.json (requires master_analytics.py run).
    """
    print('Generating Table 4.0: Dataset Statistics...')

    rows = []
    for ds in DATASETS:
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        res  = load_results(ds, 'fixed', 'lcc', 10)
        if spec is None or res is None:
            continue

        meta = res.get('metadata', {})
        a    = spec['analysis']

        rows.append({
            'dataset':     ds,
            'num_nodes':   meta.get('num_nodes'),
            'num_edges':   a.get('num_edges'),
            'avg_degree':  a.get('avg_degree'),
            'num_classes': meta.get('num_classes'),
            'num_features':meta.get('num_features'),
            'd_eff':       meta.get('d_effective') or spec.get('d_effective'),
            'homophily':   a.get('homophily'),   # None if analytics not yet rerun
        })

    if not rows:
        print('  ⚠ No data found')
        return

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{5pt}')
    lines.append(r'\caption{Dataset statistics after LCC extraction (fixed splits, $k=10$). '
                 r'$d_{\mathrm{eff}}$ = rank of $\hat{A}^k X$ after QR decomposition. '
                 r'Homophily $h$ = fraction of edges connecting same-class nodes; '
                 r'$h{=}1$ is perfectly homophilous, $h{=}0$ is heterophilous. '
                 r'\textemdash\ = homophily not yet computed (re-run master\_analytics.py).}')
    lines.append(r'\label{tab:dataset_statistics}')
    lines.append(r'\begin{tabular}{lrrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Nodes & Edges & Avg Deg & Classes & Features & $d_{\mathrm{eff}}$ & Homophily \\')
    lines.append(r'\midrule')

    for row in rows:
        ds_label  = row['dataset'].replace('-', r'\mbox{-}')
        nodes_str = f"{row['num_nodes']:,}"   if row['num_nodes']   is not None else r'\textemdash'
        edges_str = f"{row['num_edges']:,}"   if row['num_edges']   is not None else r'\textemdash'
        deg_str   = f"{row['avg_degree']:.2f}" if row['avg_degree'] is not None else r'\textemdash'
        cls_str   = str(row['num_classes'])   if row['num_classes'] is not None else r'\textemdash'
        feat_str  = f"{row['num_features']:,}" if row['num_features'] is not None else r'\textemdash'
        deff_str  = f"{row['d_eff']:,}"        if row['d_eff']       is not None else r'\textemdash'
        hom_str   = f"{row['homophily']:.3f}"  if row['homophily']   is not None else r'\textemdash'
        lines.append(rf'{ds_label} & {nodes_str} & {edges_str} & {deg_str} & '
                     rf'{cls_str} & {feat_str} & {deff_str} & {hom_str} \\')

    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / 'table_4_0_dataset_statistics.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


def generate_table_4_1_spectral_properties():
    """Table 4.1: Spectral Properties (Condition numbers, variance ratios)"""
    print('Generating Table 4.1: Spectral Properties...')

    DS_DISPLAY = {
        'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed',
        'wikics': 'WikiCS', 'amazon-computers': 'Amazon-Computers',
        'amazon-photo': 'Amazon-Photo', 'coauthor-cs': 'Coauthor-CS',
        'coauthor-physics': 'Coauthor-Physics', 'ogbn-arxiv': 'ogbn-arxiv',
    }

    def fmt_val(v, times=False):
        """Format a float: $\infty$ if inf/nan, scientific notation if >= 1e6, else 2 d.p."""
        suffix = r'$\times$' if times else ''
        if math.isinf(v) or math.isnan(v):
            return r'$\infty$' + suffix
        if abs(v) >= 1e6:
            exp = int(math.floor(math.log10(abs(v))))
            mantissa = v / 10 ** exp
            return rf'${mantissa:.2f}\times10^{{{exp}}}${suffix}'
        return f'{v:.2f}{suffix}'

    rows = []
    has_inf_X = False

    for ds in DATASETS:
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        if spec is None:
            continue
        analysis = spec['analysis']
        var_analysis = analysis['variance_analysis']
        cond_X = analysis['condition_number_X']
        if math.isinf(cond_X) or math.isnan(cond_X):
            has_inf_X = True
        rows.append({
            'dataset':     DS_DISPLAY.get(ds, ds),
            'cond_U':      analysis['condition_number_U'],
            'cond_X':      cond_X,
            'var_ratio_U': var_analysis['U_variance_ratio'],
            'var_ratio_X': var_analysis['X_variance_ratio'],
            'var_reduction': var_analysis['variance_ratio_reduction'],
        })

    caption = (
        r'Spectral properties: condition numbers and variance analysis ($k=10$). '
        r'$\kappa(\cdot)=\sigma_{\max}/\sigma_{\min}$ is the condition number; '
        r'$\infty$ indicates a rank-deficient matrix ($\sigma_{\min}=0$). '
        r'Var Ratio $=$ ratio of largest to smallest per-dimension variance. '
        r'Reduction $=$ Var Ratio$_X$ / Var Ratio$_U$.'
    )

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(rf'\caption{{{caption}}}')
    lines.append(r'\label{tab:spectral_properties}')
    lines.append(r'\begin{tabular}{lccccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Dataset} & $\kappa(U)$ & $\kappa(X)$ & \textbf{Var Ratio U} & \textbf{Var Ratio X} & \textbf{Reduction} \\')
    lines.append(r'\midrule')

    for row in rows:
        cond_X_val = row['cond_X']
        inf_row = math.isinf(cond_X_val) or math.isnan(cond_X_val)
        dagger = r'\rlap{$^\dagger$}' if inf_row else ''
        lines.append(
            f"{row['dataset']:<20} & {fmt_val(row['cond_U'])} & {fmt_val(cond_X_val)} & "
            f"{fmt_val(row['var_ratio_U'])} & {fmt_val(row['var_ratio_X'])} & "
            f"{fmt_val(row['var_reduction'], times=True)}{dagger} \\\\"
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    if has_inf_X:
        lines.append(r'\begin{tablenotes}\small')
        lines.append(
            r'\item[$\dagger$] Datasets with $\kappa(X)=\infty$ have rank-deficient feature matrices '
            r'($\sigma_{\min}=0$), making variance-based quantities involving $X$ unreliable. '
            r'The corresponding $\kappa(U)$ values remain finite, confirming that the Rayleigh-Ritz '
            r'projection regularises the degenerate directions.'
        )
        lines.append(r'\end{tablenotes}')

    lines.append(r'\end{table}')

    latex = '\n'.join(lines) + '\n'

    output_path = TABLES_DIR / 'table_4_1_spectral_properties.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# SECTION 4 ADDITIONS: Separability
# ============================================================================

def generate_table_4_2_separability():
    """Table 4.2: Class separability in the U (eigenvector) space at k=10."""
    print('Generating Table 4.2: Spectral Separability...')

    rows = []
    for ds in DATASETS:
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        if spec is None:
            continue
        a = spec['analysis']
        rows.append({
            'dataset':       ds,
            'separability':  a['spectral_separability'],
            'inter_dist':    a['inter_class_centroid_dist'],
            'intra_dist':    a['intra_class_mean_dist'],
        })

    if not rows:
        print('  ! No spectral analysis data found.')
        return

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{6pt}')
    lines.append(r'\caption{Class separability in the row-normalised eigenvector space $U$ '
                 r'at $k=10$ (fixed splits, training nodes only). '
                 r'Separability $=$ inter-class centroid distance / intra-class mean pairwise distance. '
                 r'Values $<1$ indicate class overlap; higher is better.}')
    lines.append(r'\label{tab:separability}')
    lines.append(r'\begin{tabular}{lrrr}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Separability & Inter-class dist & Intra-class dist \\')
    lines.append(r'\midrule')
    for row in rows:
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        lines.append(rf"{ds_label} & {row['separability']:.4f} & "
                     rf"{row['inter_dist']:.4f} & {row['intra_dist']:.4f} \\")
    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / 'table_4_2_separability.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


# ============================================================================
# SECTION 4 ADDITIONS: Subspace Alignment (Exp 9)
# ============================================================================

def generate_table_4_3_subspace_alignment():
    """Table 4.3: Subspace alignment summary at k=10 alongside Part A and Part B.

    Mean/Max/Min canonical correlation between col(U) and top-d_eff subspace of
    X_diffused. High mean CC means Rayleigh-Ritz preserves dominant directions;
    the table shows whether alignment predicts the Part A gap.
    """
    print('Generating Table 4.3: Subspace Alignment Summary (k=10)...')

    rows = []
    for ds in DATASETS:
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        res  = load_results(ds, 'fixed', 'lcc', 10)
        if spec is None or res is None:
            continue
        sa = spec.get('subspace_alignment')
        if sa is None:
            print(f'  ⚠ No subspace_alignment for {ds} — re-run master_analytics.py')
            continue
        fa = res.get('framework_analysis', {})
        rows.append({
            'dataset': ds,
            'mean_cc': sa['mean_canonical_correlation'],
            'max_cc':  sa['max_canonical_correlation'],
            'min_cc':  sa['min_canonical_correlation'],
            'part_a':  fa.get('part_a_pp'),
            'part_b':  fa.get('part_b_pp'),
        })

    if not rows:
        print('  ⚠ No data — run master_analytics.py for all datasets first')
        return

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{5pt}')
    lines.append(r'\caption{Subspace alignment at $k=10$ (fixed splits): canonical '
                 r'correlations between $\operatorname{col}(U)$ and the top-$d_{\mathrm{eff}}$ '
                 r'left singular vectors of $X_{\mathrm{diff}}$. '
                 r'Values near 1 mean Rayleigh-Ritz preserves dominant directions; '
                 r'near 0 means the subspaces are orthogonal. '
                 r'High mean CC with large Part A gap (e.g.\ CiteSeer) shows that '
                 r'directional preservation does \emph{not} explain the accuracy drop --- '
                 r'magnitude distortion is the key factor.}')
    lines.append(r'\label{tab:subspace_alignment}')
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Mean CC & Max CC & Min CC & Part A (pp) & Part B (pp) \\')
    lines.append(r'\midrule')
    for row in rows:
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        pa_str = f"{row['part_a']:+.2f}" if row['part_a'] is not None else r'\textemdash'
        pb_str = f"{row['part_b']:+.2f}" if row['part_b'] is not None else r'\textemdash'
        lines.append(rf"{ds_label} & {row['mean_cc']:.4f} & {row['max_cc']:.4f} & "
                     rf"{row['min_cc']:.4f} & {pa_str} & {pb_str} \\")
    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / 'table_4_3_subspace_alignment.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


def generate_figure_4_2_alignment_vs_gap():
    """Figure 4.2: Scatter — mean canonical correlation (x) vs Part A gap (y), k=10.

    Tests the hypothesis: lower alignment → larger Part A gap.
    The CiteSeer counter-example (mean_cc ≈ 0.98, Part A ≈ +36pp) already suggests
    this hypothesis is false. The figure shows the actual relationship empirically,
    with a regression line and R² to quantify the (likely weak) association.
    """
    print('Generating Figure 4.2: Alignment vs Part A Gap Scatter...')

    points = []
    for ds in DATASETS:
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        res  = load_results(ds, 'fixed', 'lcc', 10)
        if spec is None or res is None:
            continue
        sa = spec.get('subspace_alignment')
        if sa is None:
            continue
        part_a = res['framework_analysis'].get('part_a_pp')
        if part_a is None:
            continue
        points.append({
            'dataset': ds,
            'mean_cc': sa['mean_canonical_correlation'],
            'part_a':  part_a,
        })

    if len(points) < 3:
        print(f'  ⚠ Only {len(points)} points — skipping scatter')
        return

    xs = np.array([p['mean_cc'] for p in points])
    ys = np.array([p['part_a']  for p in points])

    # Linear regression
    coeffs   = np.polyfit(xs, ys, 1)
    y_pred   = np.polyval(coeffs, xs)
    ss_res   = float(np.sum((ys - y_pred) ** 2))
    ss_tot   = float(np.sum((ys - ys.mean()) ** 2))
    r_sq     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    x_line   = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 100)
    y_line   = np.polyval(coeffs, x_line)

    fig, ax = plt.subplots(figsize=(9, 7))

    for p in points:
        color = '#d62728' if p['part_a'] > 0 else '#2ca02c'
        ax.scatter(p['mean_cc'], p['part_a'], s=130, color=color,
                   edgecolor='black', linewidth=1.2, zorder=5)
        ax.annotate(p['dataset'],
                    xy=(p['mean_cc'], p['part_a']),
                    xytext=(6, 4), textcoords='offset points', fontsize=9)

    ax.plot(x_line, y_line, '--', color='steelblue', linewidth=1.8,
            label=f'Linear fit  $R^2={r_sq:.3f}$')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.5)

    ax.text(0.03, 0.97,
            r'High mean CC + large Part A (e.g.\ CiteSeer)' '\n'
            r'shows alignment $\neq$ predictive of accuracy gap.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.85))

    ax.set_xlabel('Mean Canonical Correlation (col$(U)$ vs col$(X_{\\mathrm{diff}}$))',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Part A (pp)', fontsize=13, fontweight='bold')
    ax.set_title('Subspace Alignment vs Basis Sensitivity Gap\n(k=10, Fixed Splits)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_4_2_alignment_vs_gap.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {output_path}  (R²={r_sq:.3f}, n={len(points)})')


def generate_figure_4_3_cc_spectra():
    """Figure 4.3: Canonical correlation spectra — 3×3 grid, one dataset per panel.

    X-axis: normalised eigenvector index (0→1, i.e. fraction of d_eff).
    Y-axis: canonical correlation (0→1).
    Sorted descending so the drop-off shape is visible.
    Horizontal dashed line at 0.8 marks the high-preservation threshold.

    The key story: most datasets have high CC across the bulk of directions,
    with a tail that drops toward zero. The tail size (low-CC directions) varies
    across datasets and does NOT track Part A — confirming that directional loss
    is not the mechanism behind the accuracy gap.
    """
    print('Generating Figure 4.3: Canonical Correlation Spectra (3×3)...')

    ncols = 3
    nrows = math.ceil(len(DATASETS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = axes.flatten()

    plotted = 0
    for idx, ds in enumerate(DATASETS):
        ax   = axes_flat[idx]
        spec = load_spectral_analysis(ds, 'fixed', 'lcc', 10)
        if spec is None:
            ax.set_visible(False)
            continue
        sa = spec.get('subspace_alignment')
        if sa is None:
            ax.text(0.5, 0.5, 'No alignment data\n(re-run master_analytics.py)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(ds, fontsize=12, fontweight='bold')
            continue

        corrs = np.array(sa['canonical_correlations'])   # already sorted descending
        d_eff = len(corrs)
        xs    = np.linspace(0, 1, d_eff)                 # normalised index

        # Fill under curve: above 0.8 = green, below 0.8 = red
        ax.fill_between(xs, corrs, 0.8,
                        where=(corrs >= 0.8), alpha=0.25, color='#2ca02c',
                        label='CC ≥ 0.8')
        ax.fill_between(xs, corrs, 0,
                        where=(corrs < 0.8), alpha=0.25, color='#d62728',
                        label='CC < 0.8')
        ax.plot(xs, corrs, linewidth=1.5, color='#1f77b4')
        ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)

        mean_cc = sa['mean_canonical_correlation']
        part_a  = spec.get('analysis', {})   # not stored here — load separately

        ax.set_title(f'{ds}\nmean CC={mean_cc:.3f}, $d_{{\\mathrm{{eff}}}}$={d_eff}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Normalised index (fraction of $d_{\\mathrm{eff}}$)', fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel('Canonical Correlation', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle=':')
        if idx == 0:
            ax.legend(fontsize=8, loc='lower left')
        plotted += 1

    for idx in range(len(DATASETS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Canonical Correlation Spectra: col$(U)$ vs col$(X_{\\mathrm{diff}})$\n'
                 '(k=10, Fixed Splits — sorted descending)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_4_3_cc_spectra.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {output_path} ({plotted}/{len(DATASETS)} datasets)')


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
        ['Dataset', 'Std', 'RowNorm', 'Part B', 'Log-Mag', 'ΔLog-Mag', 'Dual', 'ΔDual'],
        'Magnitude Recovery Cascade: Std → RowNorm → Log-Magnitude / Dual-Stream (k=10)',
        'tab:magnitude_cascade'
    )

    for row in rows:
        logmag_delta = f"{row['part_b5_logmag']:+.2f}pp" if row['part_b5_logmag'] is not None else 'N/A'
        logmag_str   = f"{row['logmag']:.2f}\\%"         if row['logmag']          is not None else 'N/A'
        dual_str     = f"{row['dual']:.2f}\\%"           if row['dual']            is not None else 'N/A'
        dual_delta   = f"{row['part_b5_dual']:+.2f}pp"   if row['part_b5_dual']    is not None else 'N/A'
        latex += (f"{row['dataset']:<20} & {row['std']:>6.2f}\\% & {row['rownorm']:>6.2f}\\% & "
                  f"{row['part_b']:>+7.2f}pp & "
                  f"{logmag_str:>8} & {logmag_delta:>8} & "
                  f"{dual_str:>8} & {dual_delta:>8} \\\\\n")

    latex += latex_table_footer()
    
    output_path = TABLES_DIR / 'table_5_1_magnitude_cascade.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f'  ✓ Saved: {output_path}')

def generate_figure_5_1_recovery_cascade():
    """Figure 5.1: Recovery Cascade Visualization — all datasets, 3×3 grid"""
    print('Generating Figure 5.1: Recovery Cascade Visualization...')

    ncols = 3
    nrows = math.ceil(len(DATASETS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes_flat = axes.flatten()

    plotted = 0
    for idx, ds in enumerate(DATASETS):
        data = load_results(ds, 'fixed', 'lcc', 10)
        ax = axes_flat[idx]

        if data is None:
            ax.set_visible(False)
            continue

        exps = data['experiments']

        std_acc    = exps['restricted_standard_mlp']['test_acc_mean'] * 100
        rn_acc     = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100
        logmag_acc = (exps['log_magnitude']['test_acc_mean'] * 100
                      if 'log_magnitude' in exps else None)
        dual_acc   = (exps['dual_stream']['test_acc_mean'] * 100
                      if 'dual_stream' in exps else None)

        method_names  = ['Std', 'RowNorm']
        accs          = [std_acc, rn_acc]
        method_colors = ['gray', '#d62728' if rn_acc < std_acc else '#2ca02c']
        if logmag_acc is not None:
            method_names.append('Log-Mag')
            accs.append(logmag_acc)
            method_colors.append('#2ca02c' if logmag_acc > rn_acc else 'orange')
        if dual_acc is not None:
            method_names.append('Dual')
            accs.append(dual_acc)
            method_colors.append('#2ca02c' if dual_acc > rn_acc else 'orange')

        ax.bar(method_names, accs, color=method_colors, alpha=0.8,
               edgecolor='black', linewidth=1.5)

        ax.annotate(f'{rn_acc - std_acc:+.1f}pp',
                    xy=(0.5, (std_acc + rn_acc) / 2), fontsize=10, ha='center',
                    fontweight='bold')
        if logmag_acc is not None:
            lm_idx = method_names.index('Log-Mag')
            ax.annotate(f'{logmag_acc - rn_acc:+.1f}pp',
                        xy=(lm_idx - 0.5, (rn_acc + logmag_acc) / 2), fontsize=10,
                        ha='center', fontweight='bold')

        ax.set_title(ds, fontsize=13, fontweight='bold')
        col = idx % ncols
        if col == 0:
            ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        if accs:
            ax.set_ylim(min(accs) * 0.95, max(accs) * 1.05)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=10)
        plotted += 1

    # Hide any unused subplots in the last row
    for idx in range(len(DATASETS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Magnitude Recovery Cascade: StandardMLP → RowNorm → Magnitude-Aware',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_5_1_recovery_cascade.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  ✓ Saved: {output_path} ({plotted}/{len(DATASETS)} datasets)')

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
# SECTION 5 ADDITIONS: Full α sweep  +  NestedSpheres
# ============================================================================

def generate_table_5_3_alpha_sweep_full(split_type='fixed'):
    """Table 5.3: Full spectral α sweep — all 5 α values per dataset."""
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    print(f'Generating Table 5.3: Full Alpha Sweep ({split_label})...')

    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    rows = []

    for ds in DATASETS:
        data = load_results(ds, split_type, 'lcc', 10)
        if data is None:
            continue
        exps = data['experiments']
        fa   = data.get('framework_analysis', {})
        rn_baseline  = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100
        best_alpha   = fa.get('best_spectral_alpha')
        alpha_accs   = {}
        for alpha in alphas:
            key = f'spectral_rownorm_alpha{alpha}'
            if key in exps:
                alpha_accs[alpha] = exps[key]['test_acc_mean'] * 100
        rows.append({'dataset': ds, 'rn_baseline': rn_baseline,
                     'alpha_accs': alpha_accs, 'best_alpha': best_alpha})

    if not rows:
        print(f'  ! No data for split_type={split_type}')
        return

    alpha_headers = ' & '.join([f'$\\alpha={a:+.1f}$' for a in alphas])
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{4pt}')
    lines.append(rf'\caption{{Full spectral $\alpha$ sweep: test accuracy (\%) for each '
                 rf'$\alpha \in \{{-1.0, -0.5, 0.0, 0.5, 1.0\}}$ at $k=10$ ({split_label}). '
                 r'Bold = best $\alpha$ selected on validation set. '
                 r'RowNorm baseline shown for reference.}')
    lines.append(rf'\label{{tab:alpha_sweep_full_{split_type}}}')
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(rf'Dataset & RowNorm & {alpha_headers} \\')
    lines.append(r'\midrule')
    for row in rows:
        cells = []
        for alpha in alphas:
            v = row['alpha_accs'].get(alpha)
            if v is None:
                cells.append(r'\textemdash')
            elif alpha == row['best_alpha']:
                cells.append(rf'\textbf{{{v:.2f}}}')
            else:
                cells.append(f'{v:.2f}')
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        lines.append(rf"{ds_label} & {row['rn_baseline']:.2f} & {' & '.join(cells)} \\")
    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / f'table_5_3_alpha_sweep_full_{split_type}.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


def generate_figure_5_2_alpha_sweep():
    """Figure 5.2: Spectral α sweep — accuracy vs α per dataset, 3×3 grid."""
    print('Generating Figure 5.2: Alpha Sweep...')

    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ncols  = 3
    nrows  = math.ceil(len(DATASETS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes_flat  = axes.flatten()

    plotted = 0
    for idx, ds in enumerate(DATASETS):
        data = load_results(ds, 'fixed', 'lcc', 10)
        ax   = axes_flat[idx]
        if data is None:
            ax.set_visible(False)
            continue

        exps       = data['experiments']
        fa         = data.get('framework_analysis', {})
        rn_acc     = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100
        best_alpha = fa.get('best_spectral_alpha')

        accs = []
        for alpha in alphas:
            key = f'spectral_rownorm_alpha{alpha}'
            accs.append(exps[key]['test_acc_mean'] * 100 if key in exps else None)

        valid_alphas = [a for a, v in zip(alphas, accs) if v is not None]
        valid_accs   = [v for v in accs if v is not None]

        colors = ['#d62728' if a == best_alpha else '#1f77b4' for a in valid_alphas]
        ax.bar([str(a) for a in valid_alphas], valid_accs,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.axhline(y=rn_acc, color='gray', linestyle='--', linewidth=1.5,
                   label=f'RowNorm {rn_acc:.1f}%')
        ax.set_title(ds, fontsize=13, fontweight='bold')
        ax.set_xlabel('α', fontsize=11)
        if idx % ncols == 0:
            ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        if valid_accs:
            all_vals = valid_accs + [rn_acc]
            ax.set_ylim(min(all_vals) * 0.97, max(all_vals) * 1.03)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plotted += 1

    for idx in range(len(DATASETS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Spectral $\\alpha$ Sweep: SpectralRowNormMLP at $k=10$\n'
                 '(red = best $\\alpha$ on val set; dashed = RowNorm baseline)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_5_2_alpha_sweep.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {output_path} ({plotted}/{len(DATASETS)} datasets)')


def generate_table_5_4_nested_spheres():
    """Table 5.4: NestedSpheres best result vs RowNorm and best SpectralRowNorm."""
    print('Generating Table 5.4: NestedSpheres Results...')

    rows = []
    for ds in DATASETS:
        data = load_results(ds, 'fixed', 'lcc', 10)
        if data is None:
            continue
        exps = data['experiments']
        fa   = data.get('framework_analysis', {})

        rn_acc          = exps['restricted_rownorm_mlp']['test_acc_mean'] * 100
        best_spec_acc   = fa.get('best_spectral_acc_pct')
        best_spec_alpha = fa.get('best_spectral_alpha')
        best_ns_key     = fa.get('best_nested_spheres_key')
        best_ns_acc     = fa.get('best_nested_spheres_acc_pct')
        part_b6_nested  = fa.get('part_b6_nested_pp')

        # Parse α and β from key like "nested_spheres_a-1.0_b0.0"
        ab_str = r'\textemdash'
        if best_ns_key:
            try:
                parts   = best_ns_key.replace('nested_spheres_', '').split('_b')
                a_val   = parts[0].replace('a', '')
                b_val   = parts[1]
                ab_str  = f'({float(a_val):+.1f}, {float(b_val):+.1f})'
            except Exception:
                ab_str  = best_ns_key

        rows.append({
            'dataset':        ds,
            'rn_acc':         rn_acc,
            'best_spec_acc':  best_spec_acc,
            'best_spec_alpha':best_spec_alpha,
            'best_ns_acc':    best_ns_acc,
            'part_b6_nested': part_b6_nested,
            'ab_str':         ab_str,
        })

    if not rows:
        print('  ! No data found.')
        return

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{5pt}')
    lines.append(r'\caption{NestedSpheres (Exp 6) vs SpectralRowNorm (Exp 5) at $k=10$ '
                 r'(fixed splits). Best $(\alpha,\beta)$ pair selected on validation set from '
                 r'a $5\times5$ grid. $\Delta$ is vs RowNorm baseline.}')
    lines.append(r'\label{tab:nested_spheres}')
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & RowNorm & Best Spectral & Best $\alpha$ & '
                 r'Best Nested & Best $(\alpha,\beta)$ & $\Delta$Nested \\')
    lines.append(r'\midrule')
    for row in rows:
        ds_label  = row['dataset'].replace('-', r'\mbox{-}')
        spec_str  = f"{row['best_spec_acc']:.2f}\\%" if row['best_spec_acc']  is not None else r'\textemdash'
        alpha_str = f"{row['best_spec_alpha']:+.1f}"  if row['best_spec_alpha'] is not None else r'\textemdash'
        ns_str    = f"{row['best_ns_acc']:.2f}\\%"   if row['best_ns_acc']    is not None else r'\textemdash'
        delta_str = f"{row['part_b6_nested']:+.2f}pp" if row['part_b6_nested'] is not None else r'\textemdash'
        lines.append(rf"{ds_label} & {row['rn_acc']:.2f}\% & {spec_str} & {alpha_str} & "
                     rf"{ns_str} & {row['ab_str']} & {delta_str} \\")
    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / 'table_5_4_nested_spheres.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
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

    # ── Combined multi-panel figure (val_acc only, grid layout) ──────────────
    n = len(datasets_with_data)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=False)
    axes_flat = axes.flatten() if n > 1 else [axes]
    # Hide unused subplot slots
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for ax, (ds, all_dynamics) in zip(axes_flat, datasets_with_data):
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

    axes_flat[0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.08))
    plt.suptitle('Optimization Dynamics: Validation Accuracy (k=10, Fixed Splits)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_all = FIGURES_DIR / 'figure_6_1_all_training_curves.pdf'
    plt.savefig(out_all, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved combined: {out_all}')


def generate_table_6_1_convergence_speed(split_type='fixed'):
    """Table 6.1: Convergence Speed Metrics — all datasets with dynamics data."""
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    print(f'Generating Table 6.1: Convergence Speed ({split_label})...')

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
        f'Convergence Speed Metrics ({split_label}, k=10, \\% of own peak accuracy)',
        f'tab:convergence_speed_{split_type}'
    )

    found_any = False
    for ds in DATASETS:
        all_dynamics = load_dynamics(ds, split_type, 'lcc', 10)
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

    output_path = TABLES_DIR / f'table_6_1_convergence_speed_{split_type}.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f'  ✓ Saved: {output_path}')

# ============================================================================
# Main Execution
# ============================================================================

def generate_table_exp8_k_sensitivity(split_type='fixed'):
    """Exp 8 summary table: Part A / Part B / Gap at key k values.

    Columns: Dataset | Part A at k=1,4,10,20,30 | Part B@10 | Gap@10 | Crossover k | Fisher@10
    One row per dataset. Saved as table_exp8_k_sensitivity_{split_type}.tex
    """
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    print(f'Generating Exp 8 Summary Table: k-Sensitivity ({split_label})...')

    K_SHOW = [1, 4, 10, 20, 30]

    rows = []
    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, split_type, 'lcc')
        if k_sens is None:
            continue

        by_k = {r['k']: r for r in k_sens['k_sensitivity']}

        part_a_at_k = {}
        for k in K_SHOW:
            r = by_k.get(k)
            part_a_at_k[k] = r['part_a'] if r and r.get('part_a') is not None else None

        r10     = by_k.get(10, {})
        fisher  = r10.get('fisher_score')
        part_b  = r10.get('part_b')
        rem_gap = r10.get('rem_gap')

        sorted_rows = sorted(k_sens['k_sensitivity'], key=lambda r: r['k'])
        crossover_k = None
        for i in range(len(sorted_rows) - 1):
            a0 = sorted_rows[i].get('part_a')
            a1 = sorted_rows[i+1].get('part_a')
            if a0 is not None and a1 is not None and a0 * a1 < 0:
                crossover_k = (sorted_rows[i]['k'] + sorted_rows[i+1]['k']) / 2
                break

        rows.append({
            'dataset':     ds,
            'part_a_at_k': part_a_at_k,
            'part_b':      part_b,
            'rem_gap':     rem_gap,
            'fisher':      fisher,
            'crossover_k': crossover_k,
        })

    if not rows:
        print(f'  ! No data found for split_type={split_type}')
        return

    col_spec  = 'l' + 'r' * len(K_SHOW) + 'rrrr'
    k_headers = ' & '.join([f'$k={k}$' for k in K_SHOW])

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{4pt}')
    lines.append(rf'\caption{{Exp 8 k-sensitivity ({split_label}). '
                 r'Part A (pp) at key $k$ values; Part B, Remaining Gap, Fisher at $k=10$. '
                 r'Negative Part A means restricted eigenvectors outperform SGC. '
                 r'Crossover $k$ = depth where Part A changes sign.}')
    lines.append(rf'\label{{tab:exp8_k_sensitivity_{split_type}}}')
    lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')
    lines.append(rf'\multicolumn{{1}}{{l}}{{}} & '
                 rf'\multicolumn{{{len(K_SHOW)}}}{{c}}{{Part A (pp)}} & '
                 rf'\multicolumn{{4}}{{c}}{{At $k=10$}} \\')
    lines.append(rf'\cmidrule(lr){{2-{1+len(K_SHOW)}}} '
                 rf'\cmidrule(lr){{{2+len(K_SHOW)}-{5+len(K_SHOW)}}}')
    lines.append(rf'Dataset & {k_headers} & Part B & Gap & Crossover $k$ & Fisher \\')
    lines.append(r'\midrule')

    for row in rows:
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        cells = []
        for k in K_SHOW:
            v = row['part_a_at_k'].get(k)
            cells.append(f'{v:+.2f}' if v is not None else r'\textemdash')
        pb_str  = f"{row['part_b']:+.2f}"  if row['part_b']  is not None else r'\textemdash'
        gap_str = f"{row['rem_gap']:+.2f}" if row['rem_gap'] is not None else r'\textemdash'
        ck_str  = f"{row['crossover_k']:.0f}" if row['crossover_k'] is not None else r'\textemdash'
        fi_str  = f"{row['fisher']:.4f}"   if row['fisher']  is not None else r'\textemdash'
        lines.append(rf'{ds_label} & {" & ".join(cells)} & {pb_str} & {gap_str} & {ck_str} & {fi_str} \\')

    lines.append(r'\bottomrule\end{tabular}\end{table}')

    output_path = TABLES_DIR / f'table_exp8_k_sensitivity_{split_type}.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


def generate_table_exp8_per_dataset(split_type='fixed'):
    """Per-dataset k-sensitivity tables: one table per dataset, all k rows,
    showing raw accuracies (SGC+MLP, R+Std, R+RN), Part A, Part B, Rem. Gap, Fisher, n_runs.
    Saved as table_exp8_per_dataset_{ds}_{split_type}.tex
    """
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    print(f'Generating Exp 8 Per-Dataset Tables ({split_label})...')

    DS_DISPLAY = {
        'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed',
        'wikics': 'WikiCS', 'amazon-computers': 'Amazon-Computers',
        'amazon-photo': 'Amazon-Photo', 'coauthor-cs': 'Coauthor-CS',
        'coauthor-physics': 'Coauthor-Physics', 'ogbn-arxiv': 'ogbn-arxiv',
    }

    for ds in DATASETS:
        k_sens = load_k_sensitivity(ds, split_type, 'lcc')
        if k_sens is None:
            print(f'  ! No exp8 data for {ds} ({split_type}), skipping.')
            continue

        ds_label = DS_DISPLAY.get(ds, ds)
        sorted_rows = sorted(k_sens['k_sensitivity'], key=lambda r: r['k'])

        lines = []
        lines.append(r'\begin{table}[t]')
        lines.append(r'\centering\small\setlength{\tabcolsep}{5pt}')
        lines.append(
            rf'\caption{{k-sensitivity for \textbf{{{ds_label}}} ({split_label}). '
            r'Accuracies are mean test accuracy (\%). '
            r'Part~A $=$ SGC+MLP $-$ R+Std; Part~B $=$ R+RN $-$ R+Std; '
            r'Rem.~Gap $=$ Part~A $-$ Part~B. '
            r'Fisher $=$ Fisher score of $\|U_i\|$ row norms on training nodes.}'
        )
        lines.append(rf'\label{{tab:exp8_per_dataset_{ds}_{split_type}}}')
        lines.append(r'\begin{tabular}{rrrrrrrrrr}')
        lines.append(r'\toprule')
        lines.append(
            r'$k$ & SGC+MLP & R+Std & R+RN & Part~A & Part~B & Rem.~Gap & Fisher & $n_{\mathrm{runs}}$ \\'
        )
        lines.append(r'\midrule')

        for r in sorted_rows:
            k        = r.get('k', '')
            sgc      = r.get('sgc_mlp')
            r_std    = r.get('restr_std')
            r_rn     = r.get('restr_rn')
            part_a   = r.get('part_a')
            part_b   = r.get('part_b')
            rem_gap  = r.get('rem_gap')
            fisher   = r.get('fisher_score')
            n_runs   = r.get('n_runs', r'\textemdash')

            def pct(v):
                return f'{v:.2f}\\%' if v is not None else r'\textemdash'
            def signed(v):
                return f'{v:+.3f}' if v is not None else r'\textemdash'
            def flt4(v):
                return f'{v:.5f}' if v is not None else r'\textemdash'

            lines.append(
                f'{k} & {pct(sgc)} & {pct(r_std)} & {pct(r_rn)} & '
                f'{signed(part_a)} & {signed(part_b)} & {signed(rem_gap)} & '
                f'{flt4(fisher)} & {n_runs} \\\\'
            )

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        fname = f'table_exp8_per_dataset_{ds}_{split_type}.tex'
        output_path = TABLES_DIR / fname
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'  ✓ Saved: {output_path}')


def generate_table_3_8_part_a_k_sweep(split_type='fixed'):
    """Table 3.8 / 3.9: Part A across all k values, all 9 datasets.

    Reads directly from results.json (not from exp8_k_sensitivity.json) so the
    full k sweep is available for every dataset without re-running master_analytics.
    Columns: Dataset | Part A at k=1,2,4,6,8,10,12,20,30
    Crossover k* marked with a dagger symbol where Part A changes sign.
    """
    split_label = 'Fixed Splits' if split_type == 'fixed' else 'Random Splits'
    tnum = '3.8' if split_type == 'fixed' else '3.9'
    print(f'Generating Table {tnum}: Part A k-sweep ({split_label})...')

    rows = []
    for ds in DATASETS:
        part_a_at_k = {}
        crossover_range = None
        prev_k, prev_pa = None, None

        for k in K_VALUES:
            data = load_results(ds, split_type, 'lcc', k)
            if data is None:
                part_a_at_k[k] = None
                continue
            pa = data['framework_analysis'].get('part_a_pp')
            part_a_at_k[k] = pa

            # detect crossover
            if crossover_range is None and prev_pa is not None and pa is not None:
                if prev_pa * pa < 0:
                    crossover_range = (prev_k, k)
            prev_k, prev_pa = k, pa

        rows.append({
            'dataset':        ds,
            'part_a_at_k':    part_a_at_k,
            'crossover_range': crossover_range,
        })

    if not rows:
        print(f'  ⚠ No data for split_type={split_type}')
        return

    k_headers = ' & '.join([f'$k={k}$' for k in K_VALUES])

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering\scriptsize\setlength{\tabcolsep}{3pt}')
    lines.append(rf'\caption{{Part A (pp) across all $k$ values ({split_label}). '
                 r'Part A $=$ Acc(SGC+MLP) $-$ Acc(Restricted+Std). '
                 r'Positive: SGC wins; negative: eigenvectors win (over-smoothing regime). '
                 r'$\dagger$ marks the interval where Part A changes sign (crossover $k^*$).}')
    lines.append(rf'\label{{tab:part_a_k_sweep_{split_type}}}')
    lines.append(rf'\begin{{tabular}}{{l{"r" * len(K_VALUES)}r}}')
    lines.append(r'\toprule')
    lines.append(rf'Dataset & {k_headers} & Crossover $k^*$ \\')
    lines.append(r'\midrule')

    for row in rows:
        ds_label = row['dataset'].replace('-', r'\mbox{-}')
        cells = []
        cr = row['crossover_range']
        for k in K_VALUES:
            v = row['part_a_at_k'].get(k)
            if v is None:
                cells.append(r'\textemdash')
            else:
                # mark the first k after crossover with a dagger
                cell = f'{v:+.1f}'
                if cr is not None and k == cr[1]:
                    cell = rf'{cell}$\dagger$'
                cells.append(cell)
        ck_str = (f'{cr[0]}--{cr[1]}' if cr is not None else r'\textemdash')
        lines.append(rf'{ds_label} & {" & ".join(cells)} & {ck_str} \\')

    lines.append(r'\bottomrule\end{tabular}\end{table}')

    fname = f'table_3_8_part_a_k_sweep_fixed.tex' if split_type == 'fixed' else 'table_3_9_part_a_k_sweep_random.tex'
    output_path = TABLES_DIR / fname
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Saved: {output_path}')


def generate_section_3():
    """Generate all Section 3 artifacts"""
    print("\n" + "="*80)
    print("SECTION 3: BASIS SENSITIVITY + OVER-SMOOTHING")
    print("="*80)
    generate_table_3_1_part_a()
    generate_table_3_2_part_a_random()
    generate_table_3_6_part_b('fixed')
    generate_table_3_6_part_b('random')
    generate_table_3_8_part_a_k_sweep('fixed')
    generate_table_3_8_part_a_k_sweep('random')
    generate_figure_3_1_part_a_barchart('fixed')
    generate_figure_3_1_part_a_barchart('random')
    generate_figure_3_2_part_a_vs_k()
    generate_figure_3_3_part_b_vs_k()
    generate_table_3_3_crossover_analysis('fixed')
    generate_table_3_3_crossover_analysis('random')
    generate_table_3_4_fisher_vs_k('fixed')
    generate_table_3_4_fisher_vs_k('random')
    generate_figure_3_4_fisher_vs_k()
    generate_figure_3_5_fisher_vs_partb_scatter()
    generate_table_3_5_fisher_partb_data()
    generate_table_exp8_k_sensitivity('fixed')
    generate_table_exp8_k_sensitivity('random')
    generate_table_exp8_per_dataset('fixed')
    generate_table_exp8_per_dataset('random')

def generate_section_4():
    """Generate all Section 4 artifacts"""
    print("\n" + "="*80)
    print("SECTION 4: WHITENING MECHANISM")
    print("="*80)
    generate_table_4_0_dataset_statistics()
    generate_figure_4_1_singular_values()
    generate_table_4_1_spectral_properties()
    generate_table_4_2_separability()
    generate_table_4_3_subspace_alignment()
    generate_figure_4_2_alignment_vs_gap()
    generate_figure_4_3_cc_spectra()

def generate_section_5():
    """Generate all Section 5 artifacts"""
    print("\n" + "="*80)
    print("SECTION 5: RECOVERY LIMITATIONS")
    print("="*80)
    generate_table_5_1_magnitude_cascade()
    generate_figure_5_1_recovery_cascade()
    generate_table_5_2_spectral_normalization()
    generate_table_5_3_alpha_sweep_full('fixed')
    generate_table_5_3_alpha_sweep_full('random')
    generate_figure_5_2_alpha_sweep()
    generate_table_5_4_nested_spheres()

def generate_section_6():
    """Generate all Section 6 artifacts"""
    print("\n" + "="*80)
    print("SECTION 6: OPTIMIZATION DYNAMICS")
    print("="*80)
    generate_figure_6_1_training_curves()
    generate_table_6_1_convergence_speed('fixed')
    generate_table_6_1_convergence_speed('random')

def main():
    parser = argparse.ArgumentParser(description='Generate all paper artifacts')
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