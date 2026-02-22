#!/usr/bin/env python3
"""
PAPER ARTIFACTS GENERATOR
=========================
Generates all tables and figures from master_training.py / master_analytics.py results.

Usage:
    python generate_paper_artifacts.py --tables      # Generate all LaTeX tables
    python generate_paper_artifacts.py --figures     # Generate all PDF figures
    python generate_paper_artifacts.py --all         # Generate both
    python generate_paper_artifacts.py --verify      # Verify against Section 5 corrections

Prerequisites:
    - Completed master_training.py runs for all datasets
    - Completed master_analytics.py runs for all datasets
    - Results in PAPER_RESULTS/ directory
"""

import os
import sys
import json
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'wikics',
    'amazon-computers', 'amazon-photo', 'coauthor-cs', 'coauthor-physics'
]

# Issue 1 fix: anchor paths to script location, not cwd
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'PAPER_RESULTS'
OUTPUT_DIR  = SCRIPT_DIR / 'PAPER_OUTPUT'
TABLES_DIR  = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# ============================================================================
# Helper Functions
# ============================================================================

def load_results(dataset, split_type='fixed', component='lcc', k=10):
    """Load results.json for a dataset"""
    path = RESULTS_DIR / f'{dataset}_{split_type}_{component}' / f'k{k}' / 'metrics' / 'results.json'
    if not path.exists():
        print(f'WARNING: Missing {path}')
        return None
    with open(path) as f:
        return json.load(f)

def load_fisher(dataset, split_type='fixed', component='lcc', k=10):
    """Load fisher_diagnostic.json"""
    path = RESULTS_DIR / f'{dataset}_{split_type}_{component}' / f'k{k}' / 'diagnostics' / 'fisher_diagnostic.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_dynamics(dataset, split_type='fixed', component='lcc', k=10):
    """Load all training_curves/*.json files.

    Returns a list of lists: outer list = one entry per file (split),
    inner list = flat list of per-seed records, each a dict with keys:
      seed, split, method, val_acc, train_acc, train_loss, val_loss,
      speed_to_90/95/99, auc_normalized, convergence_rate, checkpoint_val_accs
    """
    pattern = str(RESULTS_DIR / f'{dataset}_{split_type}_{component}' / f'k{k}' / 'training_curves' / 'split*_dynamics.json')
    files = sorted(glob(pattern))

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
# TABLE GENERATORS
# ============================================================================

def generate_table_3_1():
    """Table 3.1: Part A (Fixed Splits, k=10)"""
    print('Generating Table 3.1: Part A (Fixed Splits)...')

    rows = []
    for ds in DATASETS:
        data = load_results(ds, split_type='fixed', k=10)
        if data is None:
            continue

        part_a = data['framework_analysis']['part_a_pp']

        sgc   = data['experiments']['sgc_mlp_baseline']
        restr = data['experiments']['restricted_standard_mlp']

        # Error propagation for difference
        std = np.sqrt((sgc['test_acc_std']*100)**2 + (restr['test_acc_std']*100)**2)

        rows.append({'dataset': ds, 'part_a': part_a, 'std': std})

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

    print(f'  Saved: {output_path}')
    return latex


def generate_table_5_1():
    """Table 5.1: Part B (Fixed Splits, k=10)"""
    print('Generating Table 5.1: Part B (Fixed Splits)...')

    rows = []
    for ds in DATASETS:
        data = load_results(ds, split_type='fixed', k=10)
        if data is None:
            continue

        fa     = data['framework_analysis']
        part_a = fa['part_a_pp']
        part_b = fa['part_b_pp']
        gap    = fa['remaining_gap_pp']

        # Issue 4 fix: recovery only meaningful when Part A > 0 (SGC beats Restricted+Std)
        if part_a > 0.1:
            recovery_pct = (part_b / part_a) * 100
        else:
            recovery_pct = None  # N/A — no gap to recover

        rows.append({
            'dataset':  ds,
            'part_a':   part_a,
            'part_b':   part_b,
            'gap':      gap,
            'recovery': recovery_pct
        })

    latex = latex_table_header(
        ['Dataset', 'Part A', 'Part B', 'Gap', 'Recovery'],
        'Part B: RowNorm Recovery (Fixed Splits, k=10)',
        'tab:part_b_fixed'
    )

    for row in rows:
        recovery_str = f"{row['recovery']:>6.1f}\\%" if row['recovery'] is not None else 'N/A'
        latex += (f"{row['dataset']:<20} & {row['part_a']:>+7.2f}pp & "
                  f"{row['part_b']:>+7.2f}pp & {row['gap']:>+7.2f}pp & "
                  f"{recovery_str} \\\\\n")

    latex += latex_table_footer()

    output_path = TABLES_DIR / 'table_5_1_part_b_fixed.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'  Saved: {output_path}')
    return latex


def generate_table_5_3():
    """Table 5.3: Fisher Diagnostic"""
    print('Generating Table 5.3: Fisher Diagnostic...')

    threshold = 0.05
    rows = []

    for ds in DATASETS:
        fisher_data  = load_fisher(ds, split_type='fixed', k=10)
        results_data = load_results(ds, split_type='fixed', k=10)

        if fisher_data is None or results_data is None:
            continue

        fisher_score = fisher_data['fisher_score']
        part_b       = results_data['framework_analysis']['part_b_pp']

        if fisher_score > threshold:
            prediction = 'Minimal/Hurt'
            correct    = (part_b < 2.0)
        else:
            prediction = 'Help'
            correct    = (part_b > 2.0)

        rows.append({
            'dataset':    ds,
            'fisher':     fisher_score,
            'part_b':     part_b,
            'prediction': prediction,
            'correct':    correct
        })

    latex = latex_table_header(
        ['Dataset', 'Fisher', 'Part B', 'Prediction', 'Correct?'],
        'Fisher Diagnostic (Fixed Splits, k=10)',
        'tab:fisher_diagnostic'
    )

    for row in rows:
        check  = '\\checkmark' if row['correct'] else '\\texttimes'
        latex += (f"{row['dataset']:<20} & {row['fisher']:.5f} & "
                  f"{row['part_b']:>+7.2f}pp & {row['prediction']:<12} & {check} \\\\\n")

    n_correct = sum(1 for r in rows if r['correct'])
    n_total   = len(rows)
    latex += '\\midrule\n'
    latex += f"\\textbf{{Overall Accuracy}} & & & & {n_correct}/{n_total} ({n_correct/n_total*100:.1f}\\%) \\\\\n"

    latex += latex_table_footer()

    output_path = TABLES_DIR / 'table_5_3_fisher_diagnostic.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'  Saved: {output_path}')
    return latex


def generate_table_6_1():
    """Table 6.1: CiteSeer Convergence Speed"""
    print('Generating Table 6.1: CiteSeer Convergence Speed...')

    all_dynamics = load_dynamics('citeseer', split_type='fixed', k=10)
    if all_dynamics is None:
        print('  ERROR: No dynamics data for CiteSeer')
        return None

    methods = {
        'sgc_mlp_baseline':       'SGC+MLP',
        'restricted_standard_mlp': 'Restricted+Std',
        'restricted_rownorm_mlp':  'Restricted+RowNorm'
    }

    # Issue 2 fix: dynamics is a list of lists; each inner list is flat records
    # filtered by record['method']
    aggregated = {}
    for method_key, method_name in methods.items():
        speed_90, speed_95, speed_99, aucs = [], [], [], []

        for split_data in all_dynamics:       # one file (one split) per entry
            for record in split_data:         # flat list of per-seed records
                if record['method'] != method_key:
                    continue
                speed_90.append(record.get('speed_to_90') or 0)
                speed_95.append(record.get('speed_to_95') or 0)
                speed_99.append(record.get('speed_to_99') or 0)
                aucs.append(record.get('auc_normalized') or 0)

        aggregated[method_name] = {
            'speed_90_mean': np.mean(speed_90) if speed_90 else float('nan'),
            'speed_90_std':  np.std(speed_90)  if speed_90 else float('nan'),
            'speed_95_mean': np.mean(speed_95) if speed_95 else float('nan'),
            'speed_95_std':  np.std(speed_95)  if speed_95 else float('nan'),
            'speed_99_mean': np.mean(speed_99) if speed_99 else float('nan'),
            'speed_99_std':  np.std(speed_99)  if speed_99 else float('nan'),
            'auc_mean':      np.mean(aucs)      if aucs      else float('nan'),
            'auc_std':       np.std(aucs)       if aucs      else float('nan'),
        }

    latex = latex_table_header(
        ['Method', 'Speed to 90\\%', 'Speed to 95\\%', 'Speed to 99\\%', 'AUC'],
        'CiteSeer: Convergence Speed (Fixed Splits, k=10)',
        'tab:citeseer_convergence'
    )

    for method_name, stats in aggregated.items():
        latex += (f"{method_name:<25} & "
                  f"{stats['speed_90_mean']:.0f} $\\pm$ {stats['speed_90_std']:.0f} & "
                  f"{stats['speed_95_mean']:.0f} $\\pm$ {stats['speed_95_std']:.0f} & "
                  f"{stats['speed_99_mean']:.0f} $\\pm$ {stats['speed_99_std']:.0f} & "
                  f"{stats['auc_mean']:.3f} $\\pm$ {stats['auc_std']:.3f} \\\\\n")

    latex += latex_table_footer()

    output_path = TABLES_DIR / 'table_6_1_citeseer_convergence.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'  Saved: {output_path}')
    return latex


# ============================================================================
# FIGURE GENERATORS
# ============================================================================

def generate_figure_3_1():
    """Figure 3.1: Part A Bar Chart"""
    print('Generating Figure 3.1: Part A Bar Chart...')

    part_a_values = []
    dataset_names = []

    for ds in DATASETS:
        data = load_results(ds, split_type='fixed', k=10)
        if data is None:
            continue
        part_a_values.append(data['framework_analysis']['part_a_pp'])
        dataset_names.append(ds)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red' if v < 0 else 'green' for v in part_a_values]
    ax.bar(dataset_names, part_a_values, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_ylabel('Part A (pp)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_title('Basis Sensitivity Across Datasets (k=10, Fixed Splits)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_3_1_part_a_barchart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  Saved: {output_path}')


def generate_figure_6_2():
    """Figure 6.2: CiteSeer Epoch-Level Validation Curves"""
    # Issue 3 fix: use epoch-level val_acc (what is actually saved).
    # batch_level_val_acc does not exist in the saved data.
    print('Generating Figure 6.2: CiteSeer Epoch-Level Validation Curves...')

    all_dynamics = load_dynamics('citeseer', split_type='fixed', k=10)
    if all_dynamics is None:
        print('  ERROR: No dynamics data for CiteSeer')
        return

    methods = {
        'sgc_mlp_baseline':        ('SGC+MLP',               'blue'),
        'restricted_standard_mlp': ('Restricted+StandardMLP', 'red'),
        'restricted_rownorm_mlp':  ('Restricted+RowNorm',     'green')
    }

    # Issue 2 fix: correct flat-list traversal filtered by record['method']
    epoch_curves = {m: [] for m in methods}

    for split_data in all_dynamics:
        for record in split_data:
            method_key = record['method']
            if method_key in epoch_curves:
                epoch_curves[method_key].append(record['val_acc'])

    fig, ax = plt.subplots(figsize=(12, 7))

    for method_key, (label, color) in methods.items():
        curves = epoch_curves[method_key]
        if not curves:
            continue

        # Truncate all curves to the shortest run (early stopping varies per seed)
        min_len = min(len(c) for c in curves)
        arr     = np.array([c[:min_len] for c in curves])
        epochs  = np.arange(1, min_len + 1)
        mean    = arr.mean(axis=0)
        std     = arr.std(axis=0)

        ax.plot(epochs, mean, linewidth=2.5, label=label, color=color)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('CiteSeer: Epoch-Level Validation Curves (Fixed Split, k=10)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'figure_6_2_citeseer_epoch_curves.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  Saved: {output_path}')


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_section5_numbers():
    """Verify experimental results match Section 5 corrections"""
    print('\n' + '='*80)
    print('VERIFICATION: Comparing to Section 5 Corrected Values')
    print('='*80)

    expected_fixed = {
        'amazon-computers': {'part_a':   5.22, 'part_b':  0.68},
        'amazon-photo':     {'part_a':  -3.11, 'part_b':  2.38},
        'citeseer':         {'part_a': -30.26, 'part_b': 37.74},
        'coauthor-cs':      {'part_a': -10.58, 'part_b': -0.38},
        'coauthor-physics': {'part_a':  -7.37, 'part_b':  4.62},
        'cora':             {'part_a': -39.83, 'part_b': 23.41},
        'ogbn-arxiv':       {'part_a':  -0.01, 'part_b': -0.52},
        'pubmed':           {'part_a': -29.14, 'part_b':  3.72},
        'wikics':           {'part_a':  -7.40, 'part_b': -3.12},
    }

    tolerance  = 2.0
    mismatches = []

    for ds, expected in expected_fixed.items():
        data = load_results(ds, split_type='fixed', k=10)
        if data is None:
            print(f'  ⚠ {ds:20} MISSING DATA')
            continue

        fa       = data['framework_analysis']
        actual_a = fa['part_a_pp']
        actual_b = fa['part_b_pp']

        diff_a = abs(actual_a - expected['part_a'])
        diff_b = abs(actual_b - expected['part_b'])

        if diff_a > tolerance or diff_b > tolerance:
            print(f'  ❌ {ds:20} MISMATCH!')
            print(f'     Part A: expected {expected["part_a"]:+7.2f}, got {actual_a:+7.2f} (diff={diff_a:.2f}pp)')
            print(f'     Part B: expected {expected["part_b"]:+7.2f}, got {actual_b:+7.2f} (diff={diff_b:.2f}pp)')
            mismatches.append(ds)
        else:
            print(f'  ✓ {ds:20} MATCH (diff_a={diff_a:.2f}, diff_b={diff_b:.2f})')

    print('\n' + '-'*80)
    if mismatches:
        print(f'RESULT: {len(mismatches)}/{len(expected_fixed)} datasets have mismatches')
        print(f'Mismatched: {", ".join(mismatches)}')
    else:
        print(f'RESULT: ALL {len(expected_fixed)} datasets match within ±{tolerance}pp tolerance ✓')
    print('='*80 + '\n')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper tables and figures')
    parser.add_argument('--tables',  action='store_true', help='Generate all tables')
    parser.add_argument('--figures', action='store_true', help='Generate all figures')
    parser.add_argument('--verify',  action='store_true', help='Verify against Section 5')
    parser.add_argument('--all',     action='store_true', help='Generate tables + figures + verify')

    args = parser.parse_args()

    if not any([args.tables, args.figures, args.verify, args.all]):
        parser.print_help()
        return

    # Issue 5 fix: create output directories here, not at module load time
    if args.all or args.tables:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if args.all or args.figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.verify:
        verify_section5_numbers()

    if args.all or args.tables:
        print('\n' + '='*80)
        print('GENERATING TABLES')
        print('='*80)
        generate_table_3_1()
        generate_table_5_1()
        generate_table_5_3()
        generate_table_6_1()
        # Add more table generators here...
        print(f'\n✓ Tables saved to: {TABLES_DIR}')

    if args.all or args.figures:
        print('\n' + '='*80)
        print('GENERATING FIGURES')
        print('='*80)
        generate_figure_3_1()
        generate_figure_6_2()
        # Add more figure generators here...
        print(f'\n✓ Figures saved to: {FIGURES_DIR}')

    print('\n' + '='*80)
    print('PAPER ARTIFACTS GENERATION COMPLETE')
    print('='*80)
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
