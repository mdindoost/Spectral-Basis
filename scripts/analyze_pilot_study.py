"""
===================================================================================
ANALYZE PILOT STUDY RESULTS
===================================================================================

Analyzes pilot study results to determine:
1. Which radius definitions work best
2. Which weight ranges are most effective
3. Whether learned weights (Model B) help
4. Decision on whether to proceed with full experiments

Decision Criteria:
  - >1.0pp improvement on Cora/ogbn-arxiv → Full study
  - <0.5pp improvement → Abandon, focus on publication
  - One specific approach works → Full study with that approach only
===================================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# Load Results
# ============================================================================

results_dir = Path('results/pilot_study_radius')
results_file = results_dir / 'pilot_results_final.json'

if not results_file.exists():
    print(f'❌ Results file not found: {results_file}')
    print('Please run pilot_study_radius_alternatives.py first.')
    exit(1)

with open(results_file, 'r') as f:
    all_results = json.load(f)

print('='*80)
print('PILOT STUDY ANALYSIS')
print('='*80)
print(f'Loaded {len(all_results)} experiments\n')

# ============================================================================
# Organize Results
# ============================================================================

# Convert to DataFrame
df = pd.DataFrame(all_results)

print('DATA INSPECTION:')
print('='*80)
print(f'Columns: {df.columns.tolist()}')
print(f'\nRadius types: {df["radius_type"].unique()}')
print(f'Weight ranges (first 5): {df["weight_range"].head().tolist()}')
print(f'Weight range types: {df["weight_range"].apply(type).unique()}')
print()

# Convert weight_range to tuple if it's a list or string
def normalize_weight_range(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, str):
        # Parse string like "(0.3, 1.0)" or "[0.3, 1.0]"
        import ast
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return tuple(parsed)
        except:
            pass
    return x

df['weight_range_normalized'] = df['weight_range'].apply(normalize_weight_range)

# Get baseline results (Investigation 5 config: raw radius, [0.3, 1.0])
baseline_mask = (
    (df['radius_type'] == 'raw') & 
    (df['weight_range_normalized'] == (0.3, 1.0))
)

print(f'Baseline mask matches: {baseline_mask.sum()} / {len(df)} experiments')

baseline_df = df[baseline_mask].copy()

if len(baseline_df) == 0:
    print('\n⚠️  WARNING: No baseline results found!')
    print('Attempting alternative filter...')
    
    # Try without weight range filter (in case all have same range)
    baseline_mask_alt = (df['radius_type'] == 'raw')
    baseline_df = df[baseline_mask_alt].copy()
    
    if len(baseline_df) > 0:
        # Group by dataset/method and take first occurrence
        baseline_df = baseline_df.groupby(['dataset', 'method']).first().reset_index()
        print(f'✓ Found {len(baseline_df)} baseline results using alternative filter')

print('\nBASELINE RESULTS (Investigation 5 Configuration)')
print('='*80)
if len(baseline_df) > 0:
    print(baseline_df[['dataset', 'method', 'test_acc_mean', 'test_acc_std']].to_string(index=False))
else:
    print('❌ NO BASELINE RESULTS FOUND - Cannot compute improvements!')
    print('\nDEBUG INFO:')
    print(df[['dataset', 'method', 'radius_type', 'weight_range']].head(10))
    print('\nPlease check the experimental results format.')
    exit(1)
print()

# ============================================================================
# Analysis 1: Alternative Radius Definitions
# ============================================================================

print('\nANALYSIS 1: ALTERNATIVE RADIUS DEFINITIONS')
print('='*80)

radius_types = ['raw', 'eigenvalue', 'degree', 'diffused']
datasets = df['dataset'].unique()
methods = df['method'].unique()

improvements = []

for dataset in datasets:
    for method in methods:
        # Get baseline for this dataset/method
        baseline = baseline_df[
            (baseline_df['dataset'] == dataset) & 
            (baseline_df['method'] == method)
        ]
        
        if len(baseline) == 0:
            continue
            
        baseline_acc = baseline.iloc[0]['test_acc_mean']
        
        # Test each alternative radius
        for radius_type in ['eigenvalue', 'degree', 'diffused']:
            alt = df[
                (df['dataset'] == dataset) & 
                (df['method'] == method) & 
                (df['radius_type'] == radius_type) &
                (df['weight_range_normalized'] == (0.3, 1.0))
            ]
            
            if len(alt) == 0:
                continue
            
            alt_acc = alt.iloc[0]['test_acc_mean']
            improvement = alt_acc - baseline_acc
            
            improvements.append({
                'dataset': dataset,
                'method': method,
                'radius_type': radius_type,
                'baseline_acc': baseline_acc,
                'alternative_acc': alt_acc,
                'improvement': improvement
            })

improvements_df = pd.DataFrame(improvements)

if len(improvements_df) == 0:
    print('\n⚠️  No improvements computed (no alternative radius experiments found)')
else:
    # Show top improvements
    print('\nTOP 10 IMPROVEMENTS:')
    top_improvements = improvements_df.nlargest(min(10, len(improvements_df)), 'improvement')
    print(top_improvements[['dataset', 'method', 'radius_type', 'improvement']].to_string(index=False))

    # Check decision criteria for failures (Cora and ogbn-arxiv)
    failure_datasets = ['cora', 'ogbn-arxiv']
    failure_improvements = improvements_df[improvements_df['dataset'].isin(failure_datasets)]

    print(f'\nIMPROVEMENTS ON FAILURE DATASETS (Cora, ogbn-arxiv):')
    for dataset in failure_datasets:
        dataset_impr = failure_improvements[failure_improvements['dataset'] == dataset]
        if len(dataset_impr) > 0:
            max_impr = dataset_impr['improvement'].max()
            best_row = dataset_impr[dataset_impr['improvement'] == max_impr].iloc[0]
            print(f'{dataset}: Best = {max_impr:+.2f}pp ({best_row["method"]}, {best_row["radius_type"]})')
        else:
            print(f'{dataset}: No results found')

# ============================================================================
# Analysis 2: Aggressive Weight Ranges
# ============================================================================

print('\n\nANALYSIS 2: AGGRESSIVE WEIGHT RANGES')
print('='*80)

weight_range_improvements = []

for dataset in datasets:
    # Get baseline (RowNorm, raw, [0.3, 1.0])
    baseline = baseline_df[
        (baseline_df['dataset'] == dataset) & 
        (baseline_df['method'] == 'rownorm')
    ]
    
    if len(baseline) == 0:
        continue
    
    baseline_acc = baseline.iloc[0]['test_acc_mean']
    
    # Test aggressive ranges
    for weight_range in [(0.1, 1.0), (0.05, 1.0)]:
        alt = df[
            (df['dataset'] == dataset) & 
            (df['method'] == 'rownorm') & 
            (df['radius_type'] == 'raw') &
            (df['weight_range_normalized'] == weight_range)
        ]
        
        if len(alt) == 0:
            continue
        
        alt_acc = alt.iloc[0]['test_acc_mean']
        improvement = alt_acc - baseline_acc
        
        weight_range_improvements.append({
            'dataset': dataset,
            'weight_range': weight_range,
            'baseline_acc': baseline_acc,
            'aggressive_acc': alt_acc,
            'improvement': improvement
        })

weight_range_df = pd.DataFrame(weight_range_improvements)
print('\nAGGRESSIVE WEIGHT RANGE RESULTS:')
print(weight_range_df[['dataset', 'weight_range', 'improvement']].to_string(index=False))

# ============================================================================
# Analysis 3: Learned Weights (Model B)
# ============================================================================

print('\n\nANALYSIS 3: LEARNED WEIGHTS (MODEL B)')
print('='*80)

learned_improvements = []

for dataset in datasets:
    # Get baseline (RowNorm, raw, [0.3, 1.0])
    baseline = baseline_df[
        (baseline_df['dataset'] == dataset) & 
        (baseline_df['method'] == 'rownorm')
    ]
    
    if len(baseline) == 0:
        continue
    
    baseline_acc = baseline.iloc[0]['test_acc_mean']
    
    # Test learned weights
    learned = df[
        (df['dataset'] == dataset) & 
        (df['radius_type'] == 'learned')
    ]
    
    if len(learned) == 0:
        continue
    
    learned_acc = learned.iloc[0]['test_acc_mean']
    improvement = learned_acc - baseline_acc
    
    learned_improvements.append({
        'dataset': dataset,
        'baseline_acc': baseline_acc,
        'learned_acc': learned_acc,
        'improvement': improvement
    })

learned_df = pd.DataFrame(learned_improvements)
print('\nLEARNED WEIGHTS RESULTS:')
print(learned_df[['dataset', 'improvement']].to_string(index=False))

# ============================================================================
# DECISION CRITERIA
# ============================================================================

print('\n\n' + '='*80)
print('DECISION CRITERIA EVALUATION')
print('='*80)

# Criterion 1: >1.0pp improvement on Cora/ogbn-arxiv
max_improvement_cora = 0
max_improvement_arxiv = 0

if len(improvements_df) > 0:
    failure_improvements = improvements_df[improvements_df['dataset'].isin(['cora', 'ogbn-arxiv'])]
    
    if len(failure_improvements) > 0:
        cora_data = failure_improvements[failure_improvements['dataset'] == 'cora']
        arxiv_data = failure_improvements[failure_improvements['dataset'] == 'ogbn-arxiv']
        
        if len(cora_data) > 0:
            max_improvement_cora = cora_data['improvement'].max()
        if len(arxiv_data) > 0:
            max_improvement_arxiv = arxiv_data['improvement'].max()

print(f'\n1. Maximum improvement on failure datasets:')
print(f'   Cora: {max_improvement_cora:+.2f}pp')
print(f'   ogbn-arxiv: {max_improvement_arxiv:+.2f}pp')

proceed_with_full_study = (max_improvement_cora > 1.0) or (max_improvement_arxiv > 1.0)

if proceed_with_full_study:
    print(f'   ✅ DECISION: Proceed with FULL STUDY')
    print(f'   At least one failure dataset shows >1.0pp improvement')
else:
    print(f'   ❌ DECISION: Consider ABANDONING radius alternatives')
    print(f'   Both failure datasets show <1.0pp improvement')

# Criterion 2: Best overall approach
print(f'\n2. Best overall approach:')

# Concatenate all improvement dataframes
all_improvements_list = []
if len(improvements_df) > 0:
    all_improvements_list.append(improvements_df[['dataset', 'improvement']])
if len(weight_range_df) > 0:
    all_improvements_list.append(weight_range_df[['dataset', 'improvement']])
if len(learned_df) > 0:
    all_improvements_list.append(learned_df[['dataset', 'improvement']])

if len(all_improvements_list) > 0:
    all_improvements = pd.concat(all_improvements_list, ignore_index=True)

    mean_improvement = all_improvements.groupby('dataset')['improvement'].mean()
    print(f'\n   Mean improvement by dataset:')
    for dataset, impr in mean_improvement.items():
        print(f'   {dataset}: {impr:+.2f}pp')

    overall_mean = all_improvements['improvement'].mean()
    print(f'\n   Overall mean improvement: {overall_mean:+.2f}pp')
else:
    overall_mean = 0
    print(f'\n   No improvement data available')
    print(f'\n   Overall mean improvement: 0.00pp')

if abs(overall_mean) < 0.5:
    print(f'   ❌ Effect size too small (<0.5pp)')
    print(f'   RECOMMENDATION: Focus on publication with Investigations 1-4')
elif overall_mean > 0:
    print(f'   ✓ Positive effect detected')
    print(f'   Consider selective full study on promising configurations')
else:
    print(f'   ❌ Negative overall effect')
    print(f'   RECOMMENDATION: Abandon radius alternatives')

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print('\n\nGenerating visualizations...')

# Figure 1: Heatmap of improvements by radius type
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, method in enumerate(['rownorm', 'logmag', 'nested']):
    pivot = improvements_df[improvements_df['method'] == method].pivot(
        index='dataset', columns='radius_type', values='improvement'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=axes[idx], cbar_kws={'label': 'Improvement (pp)'})
    axes[idx].set_title(f'{method.upper()}: Radius Type Improvements')
    axes[idx].set_xlabel('Radius Type')
    axes[idx].set_ylabel('Dataset')

plt.tight_layout()
plt.savefig(results_dir / 'radius_type_heatmap.png', dpi=300, bbox_inches='tight')
print(f'✅ Saved: {results_dir}/radius_type_heatmap.png')

# Figure 2: Bar chart of best improvements per dataset
fig, ax = plt.subplots(figsize=(12, 6))

best_per_dataset = all_improvements.groupby('dataset')['improvement'].max().sort_values()
colors = ['green' if x > 0 else 'red' for x in best_per_dataset.values]

best_per_dataset.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.axvline(x=1.0, color='green', linestyle=':', linewidth=1, label='Success threshold (+1.0pp)')
ax.axvline(x=-1.0, color='red', linestyle=':', linewidth=1)
ax.set_xlabel('Best Improvement (pp)', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)
ax.set_title('Best Improvement per Dataset (Any Configuration)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_dir / 'best_improvements.png', dpi=300, bbox_inches='tight')
print(f'✅ Saved: {results_dir}/best_improvements.png')

# ============================================================================
# SUMMARY REPORT
# ============================================================================

summary_file = results_dir / 'pilot_study_summary.txt'

with open(summary_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('PILOT STUDY SUMMARY REPORT\n')
    f.write('='*80 + '\n\n')
    
    f.write('DECISION CRITERIA:\n')
    f.write('-'*80 + '\n')
    f.write(f'Maximum improvement on Cora: {max_improvement_cora:+.2f}pp\n')
    f.write(f'Maximum improvement on ogbn-arxiv: {max_improvement_arxiv:+.2f}pp\n')
    f.write(f'Overall mean improvement: {overall_mean:+.2f}pp\n\n')
    
    if proceed_with_full_study and len(improvements_df) > 0:
        f.write('✅ RECOMMENDATION: Proceed with full study\n')
        f.write('   At least one failure dataset shows >1.0pp improvement\n\n')
        
        # Identify best configuration
        best_config = improvements_df.nlargest(1, 'improvement').iloc[0]
        f.write('BEST CONFIGURATION:\n')
        f.write(f'   Dataset: {best_config["dataset"]}\n')
        f.write(f'   Method: {best_config["method"]}\n')
        f.write(f'   Radius: {best_config["radius_type"]}\n')
        f.write(f'   Improvement: {best_config["improvement"]:+.2f}pp\n\n')
        
        f.write('RECOMMENDED FULL STUDY:\n')
        f.write(f'   Focus on: {best_config["radius_type"]} radius definition\n')
        f.write('   Test on: All 9 datasets from Investigation 5\n')
        f.write('   Weight range: [0.3, 1.0] (current)\n')
    else:
        f.write('❌ RECOMMENDATION: Abandon radius alternatives\n')
        f.write('   Effects too small (<1.0pp on failures, <0.5pp overall)\n')
        f.write('   Focus on: Publication with Investigations 1-4\n')
        f.write('   Include: Investigation 5 as negative result\n\n')
        
        f.write('PUBLICATION FRAMING:\n')
        f.write('   - Four major components identified (Parts A-B.6)\n')
        f.write('   - Radius-based confidence weighting tested but ineffective\n')
        f.write('   - Validates that not all spectral properties are discriminative\n')

print(f'✅ Saved: {summary_file}')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print(f'\nReview {summary_file} for full recommendations')