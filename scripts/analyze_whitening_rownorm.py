"""
Analysis: Does RowNorm Fix Standard Whitening?
===============================================

This script analyzes results from investigation_whitening_rownorm.py

Separation of concerns:
- experiments/investigation_whitening_rownorm.py: Runs experiments, saves raw data
- scripts/analyze_whitening_rownorm.py: Reads results, interprets, draws conclusions

Usage:
    # Analyze all results
    python scripts/analyze_whitening_rownorm.py
    
    # Analyze specific dataset
    python scripts/analyze_whitening_rownorm.py --dataset cora
    
    # Analyze specific k value
    python scripts/analyze_whitening_rownorm.py --k 2
    
    # Generate LaTeX tables
    python scripts/analyze_whitening_rownorm.py --latex

Author: Mohammad Dindoost
Date: December 2024
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Analyze whitening + RowNorm experiments')
parser.add_argument('--results_dir', type=str, default='results/investigation_whitening_rownorm',
                   help='Directory containing experiment results')
parser.add_argument('--dataset', type=str, default=None,
                   help='Filter by dataset name')
parser.add_argument('--k', type=int, default=None,
                   help='Filter by diffusion k value')
parser.add_argument('--splits', type=str, default=None, choices=['fixed', 'random'],
                   help='Filter by split type (fixed or random)')
parser.add_argument('--latex', action='store_true',
                   help='Generate LaTeX tables')
parser.add_argument('--verbose', action='store_true',
                   help='Print detailed analysis')
args = parser.parse_args()

# ============================================================================
# Load Results
# ============================================================================

def load_all_results(results_dir, dataset_filter=None, k_filter=None, splits_filter=None):
    """Load all results.json files from the results directory."""
    results = []
    
    # Find all results.json files
    pattern = os.path.join(results_dir, '**/metrics/results.json')
    files = glob.glob(pattern, recursive=True)
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            metadata = data.get('metadata', {})
            dataset = metadata.get('dataset', 'unknown')
            k = metadata.get('k_diffusion', 0)
            split_type = metadata.get('split_type', 'unknown')
            
            # Apply filters
            if dataset_filter and dataset != dataset_filter:
                continue
            if k_filter is not None and k != k_filter:
                continue
            if splits_filter and split_type != splits_filter:
                continue
            
            results.append({
                'filepath': filepath,
                'dataset': dataset,
                'k': k,
                'split_type': split_type,
                'metadata': metadata,
                'final_results': data.get('final_results', {}),
                'whitening_analysis': data.get('whitening_analysis', {}),
                'gram_matrix_analysis': data.get('gram_matrix_analysis', {}),
            })
            
        except Exception as e:
            print(f'Warning: Could not load {filepath}: {e}')
    
    return results

# ============================================================================
# Analysis Functions
# ============================================================================

def compute_derived_metrics(result):
    """Compute whitening damage and RowNorm recovery metrics."""
    final = result['final_results']
    
    metrics = {
        'dataset': result['dataset'],
        'k': result['k'],
        'split_type': result.get('split_type', 'unknown'),
    }
    
    # Get baseline accuracies
    orig_mlp = final.get('original_MLP', {}).get('test_acc_mean', 0) * 100
    orig_mlp_rn = final.get('original_MLP+RowNorm', {}).get('test_acc_mean', 0) * 100
    
    metrics['original_MLP'] = orig_mlp
    metrics['original_MLP+RowNorm'] = orig_mlp_rn
    metrics['original_RowNorm_effect'] = orig_mlp_rn - orig_mlp
    
    # Analyze each whitening method
    whitening_methods = ['pca_whiten', 'zca_whiten', 'full_zca_whiten', 'rayleigh_ritz']
    
    for method in whitening_methods:
        mlp_key = f'{method}_MLP'
        mlp_rn_key = f'{method}_MLP+RowNorm'
        
        if mlp_key in final and mlp_rn_key in final:
            mlp_acc = final[mlp_key]['test_acc_mean'] * 100
            mlp_rn_acc = final[mlp_rn_key]['test_acc_mean'] * 100
            
            # Whitening damage = Original - Whitened (positive = damage)
            damage = orig_mlp - mlp_acc
            
            # RowNorm recovery = WithRowNorm - WithoutRowNorm
            recovery = mlp_rn_acc - mlp_acc
            
            # Recovery rate = recovery / damage (if damage > 0)
            recovery_rate = (recovery / damage * 100) if damage > 0 else None
            
            metrics[f'{method}_MLP'] = mlp_acc
            metrics[f'{method}_MLP+RowNorm'] = mlp_rn_acc
            metrics[f'{method}_damage'] = damage
            metrics[f'{method}_recovery'] = recovery
            metrics[f'{method}_recovery_rate'] = recovery_rate
    
    return metrics

def categorize_whitening_type(method):
    """Categorize whitening method by type."""
    if method in ['pca_whiten', 'zca_whiten']:
        return 'train-whitening'
    elif method == 'full_zca_whiten':
        return 'full-whitening (Wadia)'
    elif method == 'rayleigh_ritz':
        return 'spectral (Rayleigh-Ritz)'
    else:
        return 'unknown'

# ============================================================================
# Printing Functions
# ============================================================================

def print_summary_table(all_metrics):
    """Print summary table of all results."""
    
    print('\n' + '='*110)
    print('SUMMARY TABLE: MLP Accuracy (%)')
    print('='*110)
    
    # Header
    print(f'{"Dataset":<12} {"Split":<6} {"k":>3} {"Original":>10} {"PCA":>10} {"ZCA":>10} {"FullZCA":>10} {"R-R":>10}')
    print('-'*110)
    
    for m in sorted(all_metrics, key=lambda x: (x['dataset'], x['split_type'], x['k'])):
        print(f'{m["dataset"]:<12} {m["split_type"]:<6} {m["k"]:>3} '
              f'{m.get("original_MLP", 0):>10.1f} '
              f'{m.get("pca_whiten_MLP", 0):>10.1f} '
              f'{m.get("zca_whiten_MLP", 0):>10.1f} '
              f'{m.get("full_zca_whiten_MLP", 0):>10.1f} '
              f'{m.get("rayleigh_ritz_MLP", 0):>10.1f}')

def print_damage_table(all_metrics):
    """Print whitening damage table."""
    
    print('\n' + '='*110)
    print('WHITENING DAMAGE: Original MLP - Whitened MLP (positive = damage)')
    print('='*110)
    
    print(f'{"Dataset":<12} {"Split":<6} {"k":>3} {"PCA":>10} {"ZCA":>10} {"FullZCA":>10} {"R-R":>10}')
    print('-'*110)
    
    for m in sorted(all_metrics, key=lambda x: (x['dataset'], x['split_type'], x['k'])):
        print(f'{m["dataset"]:<12} {m["split_type"]:<6} {m["k"]:>3} '
              f'{m.get("pca_whiten_damage", 0):>+10.1f} '
              f'{m.get("zca_whiten_damage", 0):>+10.1f} '
              f'{m.get("full_zca_whiten_damage", 0):>+10.1f} '
              f'{m.get("rayleigh_ritz_damage", 0):>+10.1f}')

def print_recovery_table(all_metrics):
    """Print RowNorm recovery table."""
    
    print('\n' + '='*120)
    print('ROWNORM RECOVERY: MLP+RowNorm - MLP (positive = RowNorm helps)')
    print('='*120)
    
    print(f'{"Dataset":<12} {"Split":<6} {"k":>3} {"Original":>10} {"PCA":>10} {"ZCA":>10} {"FullZCA":>10} {"R-R":>10}')
    print('-'*120)
    
    for m in sorted(all_metrics, key=lambda x: (x['dataset'], x['split_type'], x['k'])):
        print(f'{m["dataset"]:<12} {m["split_type"]:<6} {m["k"]:>3} '
              f'{m.get("original_RowNorm_effect", 0):>+10.1f} '
              f'{m.get("pca_whiten_recovery", 0):>+10.1f} '
              f'{m.get("zca_whiten_recovery", 0):>+10.1f} '
              f'{m.get("full_zca_whiten_recovery", 0):>+10.1f} '
              f'{m.get("rayleigh_ritz_recovery", 0):>+10.1f}')

def print_key_findings(all_metrics):
    """Print key findings and conclusions."""
    
    print('\n' + '='*100)
    print('KEY FINDINGS')
    print('='*100)
    
    # Aggregate statistics
    methods = ['pca_whiten', 'zca_whiten', 'full_zca_whiten', 'rayleigh_ritz']
    
    for method in methods:
        damage_key = f'{method}_damage'
        recovery_key = f'{method}_recovery'
        
        damages = [m[damage_key] for m in all_metrics if damage_key in m]
        recoveries = [m[recovery_key] for m in all_metrics if recovery_key in m]
        
        if damages and recoveries:
            avg_damage = np.mean(damages)
            avg_recovery = np.mean(recoveries)
            n_damaged = sum(1 for d in damages if d > 0)
            n_recovered = sum(1 for r in recoveries if r > 0)
            
            category = categorize_whitening_type(method)
            
            print(f'\n{method} ({category}):')
            print(f'  Avg damage:   {avg_damage:+.1f}pp (damaged in {n_damaged}/{len(damages)} cases)')
            print(f'  Avg recovery: {avg_recovery:+.1f}pp (RowNorm helps in {n_recovered}/{len(recoveries)} cases)')

def print_koutis_conclusion(all_metrics):
    """Print conclusion specifically addressing Professor Koutis's question."""
    
    print('\n' + '='*100)
    print('CONCLUSION: DOES ROWNORM FIX WADIA\'S WHITENING?')
    print('='*100)
    print('\nProfessor Koutis asked: "Does RowNorm fix the whitening proposed in [Wadia et al.]?"')
    print('Wadia\'s whitening = full_zca_whiten (computed from ALL data)\n')
    
    # Get full_zca metrics
    full_zca_damages = [m['full_zca_whiten_damage'] for m in all_metrics if 'full_zca_whiten_damage' in m]
    full_zca_recoveries = [m['full_zca_whiten_recovery'] for m in all_metrics if 'full_zca_whiten_recovery' in m]
    
    if full_zca_damages and full_zca_recoveries:
        avg_damage = np.mean(full_zca_damages)
        avg_recovery = np.mean(full_zca_recoveries)
        n_damaged = sum(1 for d in full_zca_damages if d > 0)
        n_recovered = sum(1 for r in full_zca_recoveries if r > 0)
        
        print(f'Full ZCA Whitening (Wadia\'s method):')
        print(f'  - Damages MLP in {n_damaged}/{len(full_zca_damages)} cases (avg: {avg_damage:+.1f}pp)')
        print(f'  - RowNorm helps in {n_recovered}/{len(full_zca_recoveries)} cases (avg: {avg_recovery:+.1f}pp)')
        
        if avg_damage > 5 and avg_recovery > 3:
            print(f'\n★ ANSWER: YES, RowNorm provides partial recovery ({avg_recovery:.1f}pp avg)')
            print(f'  However, recovery is incomplete (only {avg_recovery/avg_damage*100:.0f}% of damage recovered)')
        elif avg_recovery > 0:
            print(f'\n★ ANSWER: PARTIAL - RowNorm provides small recovery ({avg_recovery:.1f}pp avg)')
        else:
            print(f'\n★ ANSWER: NO - RowNorm does not help Wadia\'s whitening ({avg_recovery:.1f}pp avg)')
    
    # Compare with train-whitening
    print('\n' + '-'*50)
    print('Comparison with Train-Whitening (PCA/ZCA from train set only):')
    
    pca_damages = [m['pca_whiten_damage'] for m in all_metrics if 'pca_whiten_damage' in m]
    zca_damages = [m['zca_whiten_damage'] for m in all_metrics if 'zca_whiten_damage' in m]
    
    if pca_damages:
        avg_pca_damage = np.mean(pca_damages)
        n_pca_damaged = sum(1 for d in pca_damages if d > 0)
        print(f'  PCA: Damages in {n_pca_damaged}/{len(pca_damages)} cases (avg: {avg_pca_damage:+.1f}pp)')
        
    if zca_damages:
        avg_zca_damage = np.mean(zca_damages)
        n_zca_damaged = sum(1 for d in zca_damages if d > 0)
        print(f'  ZCA: Damages in {n_zca_damaged}/{len(zca_damages)} cases (avg: {avg_zca_damage:+.1f}pp)')
    
    if pca_damages and np.mean(pca_damages) < 0:
        print('\n  Note: Train-whitening often IMPROVES MLP (negative damage).')
        print('  This is different from Wadia\'s full-data whitening.')

def print_wadia_verification_table(results):
    """
    Print Wadia criterion verification summary table.

    This addresses Professor Koutis's request to verify post-whitening geometry:
    - Is the Gram matrix K = XX^T approaching identity after whitening?
    - What is the effective rank?
    """

    print('\n' + '='*130)
    print('WADIA CRITERION VERIFICATION (Professor Koutis\'s Request)')
    print('='*130)
    print('\nKey: K = XX^T (Gram matrix) determines generalization per Wadia et al.')
    print('     Wadia predicts: n ≤ d → K should become identity → collapse')
    print('     K ≈ I means mean|eig-1| < 0.1')
    print()

    # Header
    print(f'{"Dataset":<15} {"k":>3} {"n":>7} {"d":>7} {"Regime":<12} '
          f'{"K_orig":>10} {"K_fullzca":>10} {"K≈I?":>6} {"EffRank":>8} {"Damage":>8}')
    print('-'*130)

    for r in sorted(results, key=lambda x: (x['dataset'], x['k'])):
        gram = r.get('gram_matrix_analysis', {})
        metadata = r.get('metadata', {})
        final = r.get('final_results', {})

        # Get dimensions
        n = metadata.get('num_nodes', 0)
        d = metadata.get('num_features', 0)
        k = r.get('k', 0)
        regime = 'n≤d' if n <= d else 'n>d'

        # Get Gram matrix statistics
        gram_orig = gram.get('original', {})
        gram_fullzca = gram.get('full_zca', {})

        K_orig_dist = gram_orig.get('K_dist_from_1_mean', float('nan'))
        K_fullzca_dist = gram_fullzca.get('K_dist_from_1_mean', float('nan'))
        K_is_identity = gram_fullzca.get('K_is_approximately_identity', False)
        eff_rank = gram_fullzca.get('K_n_nonzero_eig', 0)

        # Get Full ZCA damage
        orig_mlp = final.get('original_MLP', {}).get('test_acc_mean', 0) * 100
        fullzca_mlp = final.get('full_zca_whiten_MLP', {}).get('test_acc_mean', 0) * 100
        damage = orig_mlp - fullzca_mlp if orig_mlp > 0 and fullzca_mlp > 0 else float('nan')

        # Format K≈I indicator
        k_identity_str = '✓ YES' if K_is_identity else '✗ NO'

        print(f'{r["dataset"]:<15} {k:>3} {n:>7} {d:>7} {regime:<12} '
              f'{K_orig_dist:>10.4f} {K_fullzca_dist:>10.4f} {k_identity_str:>6} '
              f'{eff_rank:>8} {damage:>+8.1f}')

    # Summary statistics
    print('-'*130)

    # Count cases where K ≈ I
    n_identity = sum(1 for r in results
                     if r.get('gram_matrix_analysis', {}).get('full_zca', {}).get('K_is_approximately_identity', False))
    n_total = len(results)

    # Count cases in n ≤ d regime
    n_wadia_regime = sum(1 for r in results
                         if r.get('metadata', {}).get('num_nodes', 0) <= r.get('metadata', {}).get('num_features', 0))

    print(f'\nSummary:')
    print(f'  - K ≈ Identity after Full ZCA: {n_identity}/{n_total} cases')
    print(f'  - Cases in Wadia regime (n ≤ d): {n_wadia_regime}/{n_total}')

    # Check if whitening is working
    avg_K_dist = np.mean([r.get('gram_matrix_analysis', {}).get('full_zca', {}).get('K_dist_from_1_mean', float('nan'))
                         for r in results if 'gram_matrix_analysis' in r and 'full_zca' in r.get('gram_matrix_analysis', {})])

    if not np.isnan(avg_K_dist):
        print(f'  - Average K mean|eig-1| after Full ZCA: {avg_K_dist:.4f}')
        if avg_K_dist < 0.1:
            print(f'  → Whitening IS working (K approaching identity)')
            print(f'  → If collapse not observed, likely due to optimizer (Adam vs GD)')
        else:
            print(f'  → Whitening NOT complete (K not identity)')
            print(f'  → May need smaller eps or data is severely rank-deficient')


def print_wadia_regime_analysis(results):
    """Analyze results separately for n>d and n≤d regimes."""

    print('\n' + '='*100)
    print('WADIA REGIME ANALYSIS: n>d vs n≤d')
    print('='*100)

    # Separate by regime
    n_greater_d = [r for r in results
                   if r.get('metadata', {}).get('num_nodes', 0) > r.get('metadata', {}).get('num_features', 0)]
    n_leq_d = [r for r in results
               if r.get('metadata', {}).get('num_nodes', 0) <= r.get('metadata', {}).get('num_features', 0)]

    for regime_name, regime_results in [('n > d (structure retained)', n_greater_d),
                                         ('n ≤ d (collapse expected)', n_leq_d)]:
        if not regime_results:
            continue

        print(f'\n{regime_name}: {len(regime_results)} experiments')
        print('-'*60)

        # Compute average Full ZCA damage
        damages = []
        recoveries = []
        for r in regime_results:
            final = r.get('final_results', {})
            orig_mlp = final.get('original_MLP', {}).get('test_acc_mean', 0) * 100
            fullzca_mlp = final.get('full_zca_whiten_MLP', {}).get('test_acc_mean', 0) * 100
            fullzca_mlp_rn = final.get('full_zca_whiten_MLP+RowNorm', {}).get('test_acc_mean', 0) * 100

            if orig_mlp > 0 and fullzca_mlp > 0:
                damages.append(orig_mlp - fullzca_mlp)
                recoveries.append(fullzca_mlp_rn - fullzca_mlp)

        if damages:
            avg_damage = np.mean(damages)
            avg_recovery = np.mean(recoveries)
            n_damaged = sum(1 for d in damages if d > 0)
            n_recovered = sum(1 for r in recoveries if r > 0)

            print(f'  Full ZCA (Wadia) damage:   {avg_damage:+.1f}pp avg ({n_damaged}/{len(damages)} cases damaged)')
            print(f'  RowNorm recovery:          {avg_recovery:+.1f}pp avg ({n_recovered}/{len(recoveries)} cases helped)')

            if avg_damage > 0 and avg_recovery > 0:
                print(f'  Recovery rate:             {avg_recovery/avg_damage*100:.1f}% of damage recovered')


def generate_latex_table(all_metrics):
    """Generate LaTeX table for paper."""
    
    print('\n' + '='*100)
    print('LATEX TABLE')
    print('='*100)
    
    print(r'''
\begin{table}[h]
\centering
\caption{RowNorm Recovery on Different Whitening Methods}
\begin{tabular}{lllrrrrr}
\toprule
Dataset & Split & $k$ & Original & PCA & ZCA & Full ZCA & R-R \\
\midrule''')
    
    for m in sorted(all_metrics, key=lambda x: (x['dataset'], x['split_type'], x['k'])):
        print(f'{m["dataset"]} & {m["split_type"]} & {m["k"]} & '
              f'{m.get("original_RowNorm_effect", 0):+.1f} & '
              f'{m.get("pca_whiten_recovery", 0):+.1f} & '
              f'{m.get("zca_whiten_recovery", 0):+.1f} & '
              f'{m.get("full_zca_whiten_recovery", 0):+.1f} & '
              f'{m.get("rayleigh_ritz_recovery", 0):+.1f} \\\\')
    
    print(r'''\bottomrule
\end{tabular}
\label{tab:rownorm_recovery}
\end{table}
''')

# ============================================================================
# Main
# ============================================================================

print('='*100)
print('ANALYSIS: Does RowNorm Fix Standard Whitening?')
print('='*100)

# Load results
print(f'\nLoading results from: {args.results_dir}')
if args.splits:
    print(f'Filtering by splits: {args.splits}')
results = load_all_results(args.results_dir, args.dataset, args.k, args.splits)

if not results:
    print(f'No results found in {args.results_dir}')
    print('Run investigation_whitening_rownorm.py first to generate results.')
    sys.exit(1)

print(f'Found {len(results)} result files')

# Check for mixed splits
split_types = set(r.get('split_type', 'unknown') for r in results)
if len(split_types) > 1 and args.splits is None:
    print(f'\n⚠️  WARNING: Results contain mixed split types: {split_types}')
    print('   Consider filtering with --splits fixed or --splits random')
    print('   to analyze each split type separately.\n')

# Compute derived metrics
all_metrics = [compute_derived_metrics(r) for r in results]

# Print tables
print_summary_table(all_metrics)
print_damage_table(all_metrics)
print_recovery_table(all_metrics)
print_key_findings(all_metrics)
print_koutis_conclusion(all_metrics)

# NEW: Wadia verification tables
print_wadia_verification_table(results)
print_wadia_regime_analysis(results)

if args.latex:
    generate_latex_table(all_metrics)

print('\n' + '='*100)
print('ANALYSIS COMPLETE')
print('='*100)