"""
Summarize Random Subspace Results Across All Datasets
======================================================

Generates comprehensive comparison tables showing:
1. RowNorm effect on engineered vs random features
2. Baseline accuracy: engineered vs random
3. Variance across random subspaces
4. Cross-dataset patterns

Supports both fixed_splits and random_splits results.

Usage:
    # Default: look for fixed_splits results
    python scripts/summarize_random_subspaces.py
    
    # Look for random_splits results
    python scripts/summarize_random_subspaces.py --random-splits
"""

import os
import sys
import json
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Check for --random-splits flag
USE_RANDOM_SPLITS = '--random-splits' in sys.argv
split_type = 'random_splits' if USE_RANDOM_SPLITS else 'fixed_splits'

DATASETS = [
    'ogbn-arxiv',
    'cora', 
    'citeseer',
    'pubmed',
    'wikics',
    'amazon-photo',
    'amazon-computers',
    'coauthor-cs',
    'coauthor-physics'
]

results_base = 'results/investigation2_random_subspaces'

# ============================================================================
# Load Results
# ============================================================================

def load_results(dataset_name):
    """Load results for a dataset"""
    path = f'{results_base}/{dataset_name}/{split_type}/metrics/results_complete.json'
    
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

print('='*140)
print(f'RANDOM SUBSPACE EXPERIMENT: CROSS-DATASET SUMMARY ({split_type.upper()})')
print('='*140)

results_data = []

for dataset in DATASETS:
    result = load_results(dataset)
    if result is None:
        print(f'⚠️  Missing: {dataset} ({split_type})')
        continue
    
    # Extract key metrics
    orig = result['original_features']
    rand = result['aggregated_random']
    comp = result['comparisons']
    
    data = {
        'dataset': dataset,
        'split_type': result['split_type'],
        'dimension': result['dimension'],
        'num_subspaces': result['num_random_subspaces'],
        'num_splits': result.get('num_data_splits', 1),
        
        # Original X results
        'orig_a': orig['a_X_std']['test_acc_mean'],
        'orig_a_std': orig['a_X_std']['test_acc_std'],
        'orig_b': orig['b_U_std']['test_acc_mean'],
        'orig_c': orig['c_U_row']['test_acc_mean'],
        'orig_dir_A': comp['direction_A_original'],
        'orig_dir_B': comp['direction_B_original'],
        
        # Random X_r results
        'rand_a': rand['a_X_std']['mean_across_subspaces'],
        'rand_a_std': rand['a_X_std']['std_across_subspaces'],
        'rand_b': rand['b_U_std']['mean_across_subspaces'],
        'rand_c': rand['c_U_row']['mean_across_subspaces'],
        'rand_dir_A': comp['direction_A_random'],
        'rand_dir_B': comp['direction_B_random'],
        
        # Key comparisons
        'baseline_diff': comp['baseline_diff'],
        'dir_A_improvement': comp['direction_A_improvement'],
        'variance_a': rand['a_X_std']['std_across_subspaces'],
        'variance_b': rand['b_U_std']['std_across_subspaces'],
        'variance_c': rand['c_U_row']['std_across_subspaces']
    }
    
    results_data.append(data)

if not results_data:
    print(f'\nNo results found for {split_type}!')
    print(f'Please run experiments first:')
    if USE_RANDOM_SPLITS:
        print(f'  python experiments/investigation2_random_subspaces.py <dataset> --random-splits')
    else:
        print(f'  python experiments/investigation2_random_subspaces.py <dataset>')
    exit(1)

print(f'\nFound results for {len(results_data)}/{len(DATASETS)} datasets')

# ============================================================================
# Table 1: Complete Results Comparison
# ============================================================================

print('\n' + '='*140)
print('TABLE 1: ENGINEERED vs RANDOM FEATURES - COMPLETE COMPARISON')
print('='*140)
print(f"{'Dataset':<20} {'Dim':<6} {'Subsp':<6} {'Source':<12} {'(a) Baseline':<16} {'Dir B':<10} {'Dir A':<10}")
print('-'*140)

for r in results_data:
    # Original features
    print(f"{r['dataset']:<20} {r['dimension']:<6} {r['num_subspaces']:<6} {'Original X':<12} "
          f"{r['orig_a']*100:5.2f}±{r['orig_a_std']*100:4.2f}%  "
          f"{r['orig_dir_B']:+7.1f}%  {r['orig_dir_A']:+7.1f}%")
    
    # Random features
    print(f"{'':<20} {'':<6} {'':<6} {'Random X_r':<12} "
          f"{r['rand_a']*100:5.2f}±{r['rand_a_std']*100:4.2f}%  "
          f"{r['rand_dir_B']:+7.1f}%  {r['rand_dir_A']:+7.1f}%")
    
    # Difference
    baseline_diff = r['baseline_diff'] * 100
    dir_A_diff = r['dir_A_improvement']
    print(f"{'':<20} {'':<6} {'':<6} {'Difference':<12} "
          f"{baseline_diff:+5.2f}pp        "
          f"{'':>7}   {dir_A_diff:+7.1f}pp")
    print()

print('='*140)
print('\nLegend:')
print('  Dim = Feature dimension')
print('  Subsp = Number of random subspaces tested')
print('  (a) Baseline = Experiment (a): Standard MLP with StandardScaler')
print('  Dir B = Direction B: Basis sensitivity (a→b)')
print('  Dir A = Direction A: RowNorm effect (b→c)')
print('  pp = percentage points')

# ============================================================================
# Table 2: Key Findings
# ============================================================================

print('\n' + '='*140)
print('TABLE 2: KEY FINDINGS')
print('='*140)

# Finding 1: RowNorm improvement
print('\n1. DIRECTION A: Does RowNorm work better on random features?')
print('-'*80)
print(f"{'Dataset':<20} {'Orig X':<12} {'Rand X_r':<12} {'Improvement':<15} {'Verdict':<30}")
print('-'*80)

rownorm_better = []
rownorm_much_better = []
for r in results_data:
    improvement = r['dir_A_improvement']
    
    if improvement > 10:
        verdict = '✓✓ MUCH BETTER'
        rownorm_much_better.append(r['dataset'])
        rownorm_better.append(r['dataset'])
    elif improvement > 5:
        verdict = '✓ Significantly better'
        rownorm_better.append(r['dataset'])
    elif improvement > 2:
        verdict = '✓ Better'
        rownorm_better.append(r['dataset'])
    elif abs(improvement) < 2:
        verdict = '≈ Similar'
    else:
        verdict = '✗ Worse'
    
    print(f"{r['dataset']:<20} {r['orig_dir_A']:+6.1f}%    {r['rand_dir_A']:+6.1f}%    "
          f"{improvement:+6.1f}pp       {verdict:<30}")

print(f'\nDatasets where random helps RowNorm: {len(rownorm_better)}/{len(results_data)}')
if rownorm_better:
    print(f'  {", ".join(rownorm_better)}')
if rownorm_much_better:
    print(f'\nDatasets with dramatic improvement (>10pp): {", ".join(rownorm_much_better)}')

# Finding 2: Random vs engineered baseline
print('\n2. BASELINE COMPARISON: Can random features match engineered?')
print('-'*80)
print(f"{'Dataset':<20} {'Engineered':<15} {'Random':<15} {'Difference':<15} {'Verdict':<30}")
print('-'*80)

random_better = []
random_similar = []
for r in results_data:
    diff = r['baseline_diff'] * 100
    
    if diff > 2:
        verdict = '🤯 Random better!'
        random_better.append(r['dataset'])
    elif abs(diff) < 2:
        verdict = '≈ Similar'
        random_similar.append(r['dataset'])
    else:
        verdict = '✓ Engineered better'
    
    print(f"{r['dataset']:<20} {r['orig_a']*100:5.2f}%        {r['rand_a']*100:5.2f}%        "
          f"{diff:+5.2f}pp        {verdict:<30}")

print(f'\nRandom outperforms engineered: {len(random_better)}/{len(results_data)}')
if random_better:
    print(f'  {", ".join(random_better)}')

print(f'\nRandom matches engineered (±2pp): {len(random_similar)}/{len(results_data)}')
if random_similar:
    print(f'  {", ".join(random_similar)}')

# Finding 3: Variance analysis
print('\n3. VARIANCE ACROSS RANDOM SUBSPACES')
print('-'*80)
print(f"{'Dataset':<20} {'Exp (a)':<12} {'Exp (b)':<12} {'Exp (c)':<12} {'Stability':<20}")
print('-'*80)

high_variance = []
for r in results_data:
    var_a = r['variance_a']
    var_b = r['variance_b']
    var_c = r['variance_c']
    max_var = max(var_a, var_b, var_c)
    
    if max_var > 0.05:
        stability = '⚠️  High variance'
        high_variance.append(r['dataset'])
    elif max_var > 0.02:
        stability = '≈ Moderate'
    else:
        stability = '✓ Stable'
    
    print(f"{r['dataset']:<20} {var_a:.4f}      {var_b:.4f}      {var_c:.4f}      {stability:<20}")

if high_variance:
    print(f'\nDatasets with high variance (>0.05): {", ".join(high_variance)}')
    print('→ Specific random subspace choice matters for these datasets')

print('='*140)

# ============================================================================
# Summary Statistics
# ============================================================================

print('\n' + '='*140)
print('SUMMARY STATISTICS')
print('='*140)

# Direction A improvement
dir_A_improvements = [r['dir_A_improvement'] for r in results_data]
print(f'\nDirection A (RowNorm Effect) Improvement (Random - Original):')
print(f'  Mean: {np.mean(dir_A_improvements):+.1f}pp')
print(f'  Std:  {np.std(dir_A_improvements):.1f}pp')
print(f'  Range: [{np.min(dir_A_improvements):+.1f}, {np.max(dir_A_improvements):+.1f}]pp')

positive_improvements = [x for x in dir_A_improvements if x > 2]
print(f'  Datasets with >2pp improvement: {len(positive_improvements)}/{len(results_data)}')

large_improvements = [x for x in dir_A_improvements if x > 10]
if large_improvements:
    print(f'  Datasets with >10pp improvement: {len(large_improvements)}/{len(results_data)}')

# Baseline differences
baseline_diffs = [r['baseline_diff'] * 100 for r in results_data]
print(f'\nBaseline Accuracy Difference (Random - Engineered):')
print(f'  Mean: {np.mean(baseline_diffs):+.1f}pp')
print(f'  Std:  {np.std(baseline_diffs):.1f}pp')
print(f'  Range: [{np.min(baseline_diffs):+.1f}, {np.max(baseline_diffs):+.1f}]pp')

close_to_zero = [x for x in baseline_diffs if abs(x) < 5]
print(f'  Datasets within 5pp: {len(close_to_zero)}/{len(results_data)}')

# Average variances
avg_var_a = np.mean([r['variance_a'] for r in results_data])
avg_var_b = np.mean([r['variance_b'] for r in results_data])
avg_var_c = np.mean([r['variance_c'] for r in results_data])

print(f'\nAverage Variance Across Random Subspaces:')
print(f'  Experiment (a): {avg_var_a:.4f}')
print(f'  Experiment (b): {avg_var_b:.4f}')
print(f'  Experiment (c): {avg_var_c:.4f}')

print('\n' + '='*140)
print('INTERPRETATION GUIDE')
print('='*140)

print('\nOutcome 1: RowNorm works on random but not engineered')
if np.mean(dir_A_improvements) > 5:
    print(f'  ✓ CONFIRMED: Average improvement = {np.mean(dir_A_improvements):.1f}pp')
    print('  → Problem is with engineered feature conditioning')
    print('  → Random Gaussian features are better suited for restricted eigenvectors')
    print(f'  → {len(positive_improvements)}/{len(results_data)} datasets show this pattern')
else:
    print(f'  ✗ NOT CONFIRMED: Average improvement = {np.mean(dir_A_improvements):.1f}pp')
    print('  → Random features do not systematically help RowNorm')

print('\nOutcome 2: Random features match/exceed engineered')
if np.mean(baseline_diffs) > -2:
    print(f'  ✓ CONFIRMED: Average difference = {np.mean(baseline_diffs):.1f}pp')
    print('  → Graph structure (via random subspaces) captures task-relevant information')
    if len(random_better) > 0:
        print(f'  → Random actually OUTPERFORMS on {len(random_better)} datasets!')
    print('  → Feature engineering may be less critical than expected')
else:
    print(f'  ✗ NOT CONFIRMED: Average difference = {np.mean(baseline_diffs):.1f}pp')
    print('  → Engineered features provide significant value beyond graph structure')

print('\nOutcome 3: High variance across subspaces')
if avg_var_a > 0.03:
    print(f'  ✓ CONFIRMED: Average variance = {avg_var_a:.4f}')
    print('  → Specific random subspace matters')
    print(f'  → {len(high_variance)} datasets show high variance')
    print('  → Need to understand which directions are favorable')
else:
    print(f'  ✗ NOT CONFIRMED: Average variance = {avg_var_a:.4f}')
    print('  → Results consistent across random subspaces')
    print('  → Random subspace choice doesn\'t matter much')

print('\n' + '='*140)
print('DATASET-SPECIFIC PATTERNS')
print('='*140)

# Group by domain
domains = {
    'Citation Networks': ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed'],
    'Wikipedia': ['wikics'],
    'Co-purchase': ['amazon-photo', 'amazon-computers'],
    'Co-authorship': ['coauthor-cs', 'coauthor-physics']
}

for domain_name, domain_datasets in domains.items():
    domain_results = [r for r in results_data if r['dataset'] in domain_datasets]
    if not domain_results:
        continue
    
    print(f'\n{domain_name}:')
    avg_dir_A = np.mean([r['dir_A_improvement'] for r in domain_results])
    avg_baseline = np.mean([r['baseline_diff'] * 100 for r in domain_results])
    
    print(f'  Avg Direction A improvement: {avg_dir_A:+.1f}pp')
    print(f'  Avg baseline difference: {avg_baseline:+.1f}pp')
    
    if avg_dir_A > 5:
        print(f'  → Random features help RowNorm in this domain')
    if abs(avg_baseline) < 2:
        print(f'  → Random features match engineered in this domain')

print('\n' + '='*140)
print('NEXT STEPS')
print('='*140)

print('\n1. If RowNorm improves significantly:')
print('   → Investigate why random features have better conditioning')
print('   → Compare covariance structure of X vs X_r')
print('   → Consider random projection preprocessing for engineered features')

print('\n2. If random features match engineered:')
print('   → This is a major finding! Graph structure is sufficient')
print('   → Feature engineering may be over-fitting')
print('   → Consider unsupervised graph-based feature generation')

print('\n3. If high variance across subspaces:')
print('   → Analyze which random directions work best')
print('   → Look for patterns in successful subspaces')
print('   → May need guided random projection (not purely Gaussian)')

print('\n4. Domain-specific findings:')
if len(rownorm_better) > 0:
    print(f'   → Focus on domains where random helps: {", ".join(set([r["dataset"] for r in results_data if r["dataset"] in rownorm_better]))}')
if len(random_better) > 0:
    print(f'   → Investigate why random outperforms on: {", ".join(random_better)}')

print('\n' + '='*140)
print(f'Summary complete for {split_type}')
print('='*140)