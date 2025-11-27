"""
===================================================================================
PART A/B ANALYSIS SCRIPT - For Yiannis's Framework (FIXED VERSION)
===================================================================================

This script analyzes results according to the two-part framework:

PART A: BASIS SENSITIVITY
- Compare: SGC+MLP vs Restricted+StandardMLP
- Goal: Show that basis choice matters (even with same span)
- Success: ANY difference (positive or negative)

PART B: SPECTRAL-OPTIMALITY  
- Compare: Restricted+StandardMLP vs Restricted+RowNorm
- Goal: Show that RowNorm improves over StandardMLP on eigenvectors
- Success: RowNorm > StandardMLP

THE GAP:
- Compare: SGC+MLP vs Restricted+RowNorm
- Shows how much of Part A is "fixed" by Part B
===================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Part A/B Framework Analysis')
parser.add_argument('--results_dir', type=str, 
                   default='results/investigation_sgc_antiSmoothing',
                   help='Base directory containing results')
parser.add_argument('--split_type', type=str, choices=['fixed', 'random'],
                   default='fixed', help='Which split type to analyze')
args = parser.parse_args()

RESULTS_BASE = args.results_dir
SPLIT_TYPE = args.split_type
OUTPUT_DIR = os.path.join(RESULTS_BASE, f'part_ab_analysis_{SPLIT_TYPE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style('whitegrid')

print('='*80)
print('PART A/B FRAMEWORK ANALYSIS')
print('='*80)
print(f'Split type: {SPLIT_TYPE}')
print(f'Output: {OUTPUT_DIR}')
print('='*80)

# ============================================================================
# LOAD DATA - FIXED TO MATCH ACTUAL DIRECTORY STRUCTURE
# ============================================================================

def load_all_results(base_dir, split_type):
    """
    Load results from all datasets
    
    Expected structure:
    results/investigation_sgc_antiSmoothing/
    ├── ogbn-arxiv_fixed_lcc/
    │   ├── k2/
    │   │   └── metrics/
    │   │       └── results.json
    │   ├── k4/
    │   └── k10/
    ├── ogbn-arxiv_random_lcc/
    └── ...
    """
    data = {}
    base_path = Path(base_dir)
    
    print('\nLoading results...')
    print('RESULTS_BASE = ', base_dir)
    print('SPLIT_TYPE = ', split_type)
    
    # Look for directories matching pattern: <dataset>_<split_type>_lcc
    suffix = f'_{split_type}_lcc'
    
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dir_name = dataset_dir.name
        
        # Check if this directory matches the split type we want
        if not dir_name.endswith(suffix):
            continue
        
        # Extract dataset name (remove suffix)
        dataset_name = dir_name[:-len(suffix)]
        
        # Load results from all k directories
        dataset_results = {}
        
        for k_dir in dataset_dir.iterdir():
            if not k_dir.is_dir() or not k_dir.name.startswith('k'):
                continue
            
            # Extract k value
            try:
                k = int(k_dir.name[1:])  # Remove 'k' prefix
            except ValueError:
                continue
            
            # Load results.json
            results_file = k_dir / 'metrics' / 'results.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    dataset_results[k] = json.load(f)
        
        if dataset_results:
            data[dataset_name] = dataset_results
            print(f'✓ Loaded: {dataset_name} ({len(dataset_results)} k values)')
    
    return data

print('\nScanning directory structure...')
all_data = load_all_results(RESULTS_BASE, SPLIT_TYPE)
print(f'Total datasets: {len(all_data)}')

if len(all_data) == 0:
    print('\n⚠️  No data found!')
    print(f'\nExpected directory structure:')
    print(f'  {RESULTS_BASE}/')
    print(f'    ├── <dataset>_{SPLIT_TYPE}_lcc/')
    print(f'    │   ├── k2/')
    print(f'    │   │   └── metrics/')
    print(f'    │   │       └── results.json')
    print(f'    │   ├── k4/')
    print(f'    │   └── ...')
    print(f'\nActual directories found:')
    base_path = Path(RESULTS_BASE)
    if base_path.exists():
        for d in sorted(base_path.iterdir()):
            if d.is_dir():
                print(f'  - {d.name}')
    else:
        print(f'  ⚠️  Base directory does not exist: {RESULTS_BASE}')
    sys.exit(1)

# ============================================================================
# PART A: BASIS SENSITIVITY ANALYSIS
# ============================================================================

def analyze_part_a(data, output_dir):
    """
    PART A: BASIS SENSITIVITY
    
    Compare: SGC+MLP vs Restricted+StandardMLP
    Goal: Show basis choice matters (even with identical span)
    """
    print('\n' + '='*80)
    print('PART A: BASIS SENSITIVITY ANALYSIS')
    print('='*80)
    
    results = []
    
    for dataset in sorted(data.keys()):
        for k, k_data in data[dataset].items():
            results_dict = k_data.get('results', {})
            
            sgc_mlp = results_dict.get('sgc_mlp_baseline', {}).get('test_acc_mean', np.nan)
            restricted_std = results_dict.get('restricted_standard_mlp', {}).get('test_acc_mean', np.nan)
            
            if not np.isnan(sgc_mlp) and not np.isnan(restricted_std):
                difference = (restricted_std - sgc_mlp) * 100
                abs_difference = abs(difference)
                
                results.append({
                    'Dataset': dataset,
                    'k': k,
                    'SGC+MLP': f'{sgc_mlp*100:.2f}%',
                    'Restricted+StandardMLP': f'{restricted_std*100:.2f}%',
                    'Difference (pp)': f'{difference:+.2f}',
                    'Abs Difference': abs_difference,
                    'Sensitivity': 'Yes' if abs_difference > 0.5 else 'Marginal'
                })
    
    if not results:
        print('⚠️  No Part A results found (missing restricted_standard_mlp)')
        return None
    
    df = pd.DataFrame(results)
    
    # Summary at k=10
    df_k10 = df[df['k'] == 10].copy()
    
    if len(df_k10) == 0:
        print('⚠️  No k=10 results found, using max k available')
        max_k = df['k'].max()
        df_k10 = df[df['k'] == max_k].copy()
    
    print(f'\n--- Part A Results (k={df_k10.iloc[0]["k"]}) ---')
    print(df_k10[['Dataset', 'SGC+MLP', 'Restricted+StandardMLP', 'Difference (pp)', 'Sensitivity']].to_string(index=False))
    
    # Save
    df_k10.to_csv(os.path.join(output_dir, 'part_a_basis_sensitivity.csv'), index=False)
    
    # Statistics
    sensitivity_count = (df_k10['Sensitivity'] == 'Yes').sum()
    total = len(df_k10)
    
    print(f'\nSENSITIVITY STATISTICS:')
    print(f'  Datasets showing basis sensitivity: {sensitivity_count}/{total} ({sensitivity_count/total*100:.1f}%)')
    print(f'  Mean absolute difference: {df_k10["Abs Difference"].mean():.2f}pp')
    print(f'  Max absolute difference: {df_k10["Abs Difference"].max():.2f}pp')
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = df_k10['Dataset'].values
    sgc_scores = [float(x.strip('%')) for x in df_k10['SGC+MLP'].values]
    restricted_scores = [float(x.strip('%')) for x in df_k10['Restricted+StandardMLP'].values]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, sgc_scores, width, label='SGC+MLP', alpha=0.8)
    ax.bar(x + width/2, restricted_scores, width, label='Restricted+StandardMLP', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Part A: Basis Sensitivity (k={df_k10.iloc[0]["k"]})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part_a_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\n✓ Saved: {output_dir}/part_a_basis_sensitivity.csv')
    print(f'✓ Saved: {output_dir}/part_a_comparison.png')
    
    return df_k10

# ============================================================================
# PART B: SPECTRAL-OPTIMALITY ANALYSIS
# ============================================================================

def analyze_part_b(data, output_dir):
    """
    PART B: SPECTRAL-OPTIMALITY
    
    Compare: Restricted+StandardMLP vs Restricted+RowNorm
    Goal: Show RowNorm improves over StandardMLP on eigenvectors
    """
    print('\n' + '='*80)
    print('PART B: SPECTRAL-OPTIMALITY ANALYSIS')
    print('='*80)
    
    results = []
    
    for dataset in sorted(data.keys()):
        for k, k_data in data[dataset].items():
            results_dict = k_data.get('results', {})
            
            restricted_std = results_dict.get('restricted_standard_mlp', {}).get('test_acc_mean', np.nan)
            restricted_rownorm = results_dict.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            
            if not np.isnan(restricted_std) and not np.isnan(restricted_rownorm):
                improvement = (restricted_rownorm - restricted_std) * 100
                
                results.append({
                    'Dataset': dataset,
                    'k': k,
                    'Restricted+StandardMLP': f'{restricted_std*100:.2f}%',
                    'Restricted+RowNorm': f'{restricted_rownorm*100:.2f}%',
                    'Improvement (pp)': f'{improvement:+.2f}',
                    'RowNorm Helps?': 'Yes' if improvement > 0.5 else ('Marginal' if improvement > -0.5 else 'No')
                })
    
    if not results:
        print('⚠️  No Part B results found')
        return None
    
    df = pd.DataFrame(results)
    
    # Summary at k=10
    df_k10 = df[df['k'] == 10].copy()
    
    if len(df_k10) == 0:
        print('⚠️  No k=10 results found, using max k available')
        max_k = df['k'].max()
        df_k10 = df[df['k'] == max_k].copy()
    
    print(f'\n--- Part B Results (k={df_k10.iloc[0]["k"]}) ---')
    print(df_k10[['Dataset', 'Restricted+StandardMLP', 'Restricted+RowNorm', 'Improvement (pp)', 'RowNorm Helps?']].to_string(index=False))
    
    # Save
    df_k10.to_csv(os.path.join(output_dir, 'part_b_rownorm_improvement.csv'), index=False)
    
    # Statistics
    helps_count = (df_k10['RowNorm Helps?'] == 'Yes').sum()
    total = len(df_k10)
    improvements = [float(x.strip('%').replace('+', '')) for x in df_k10['Improvement (pp)'].values]
    
    print(f'\nROWNORM IMPROVEMENT STATISTICS:')
    print(f'  Datasets where RowNorm helps: {helps_count}/{total} ({helps_count/total*100:.1f}%)')
    print(f'  Mean improvement: {np.mean(improvements):+.2f}pp')
    print(f'  Positive improvements: {sum(1 for x in improvements if x > 0)}/{total}')
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = df_k10['Dataset'].values
    std_scores = [float(x.strip('%')) for x in df_k10['Restricted+StandardMLP'].values]
    rownorm_scores = [float(x.strip('%')) for x in df_k10['Restricted+RowNorm'].values]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, std_scores, width, label='Restricted+StandardMLP', alpha=0.8)
    ax.bar(x + width/2, rownorm_scores, width, label='Restricted+RowNorm', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Part B: RowNorm Improvement on Eigenvectors (k={df_k10.iloc[0]["k"]})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part_b_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\n✓ Saved: {output_dir}/part_b_rownorm_improvement.csv')
    print(f'✓ Saved: {output_dir}/part_b_comparison.png')
    
    return df_k10

# ============================================================================
# THE GAP ANALYSIS
# ============================================================================

def analyze_the_gap(data, output_dir):
    """
    THE GAP: How much does Part B fix Part A?
    
    Compare: SGC+MLP (baseline) vs Restricted+RowNorm (our best)
    """
    print('\n' + '='*80)
    print('THE GAP: HOW MUCH DOES PART B FIX PART A?')
    print('='*80)
    
    results = []
    
    for dataset in sorted(data.keys()):
        for k, k_data in data[dataset].items():
            results_dict = k_data.get('results', {})
            
            sgc_mlp = results_dict.get('sgc_mlp_baseline', {}).get('test_acc_mean', np.nan)
            restricted_std = results_dict.get('restricted_standard_mlp', {}).get('test_acc_mean', np.nan)
            restricted_rownorm = results_dict.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            
            if not np.isnan(sgc_mlp) and not np.isnan(restricted_std) and not np.isnan(restricted_rownorm):
                # Part A: How much worse is Restricted+Std vs SGC+MLP?
                part_a_gap = (restricted_std - sgc_mlp) * 100
                
                # Part B: How much better is Restricted+RowNorm vs Restricted+Std?
                part_b_improvement = (restricted_rownorm - restricted_std) * 100
                
                # The Gap: How much better/worse is Restricted+RowNorm vs SGC+MLP?
                final_gap = (restricted_rownorm - sgc_mlp) * 100
                
                # How much of Part A is fixed?
                if part_a_gap < 0:  # If restricted was worse
                    if part_b_improvement > 0:
                        pct_fixed = min((part_b_improvement / abs(part_a_gap)) * 100, 100)
                    else:
                        pct_fixed = 0
                else:
                    pct_fixed = 100  # Already better in Part A
                
                results.append({
                    'Dataset': dataset,
                    'k': k,
                    'SGC+MLP': f'{sgc_mlp*100:.2f}%',
                    'Restricted+StandardMLP': f'{restricted_std*100:.2f}%',
                    'Restricted+RowNorm': f'{restricted_rownorm*100:.2f}%',
                    'Part A Gap (pp)': f'{part_a_gap:+.2f}',
                    'Part B Improvement (pp)': f'{part_b_improvement:+.2f}',
                    'Final Gap (pp)': f'{final_gap:+.2f}',
                    'Pct Fixed': f'{pct_fixed:.1f}%',
                    'Status': 'Fully Fixed' if final_gap > 0 else 'Partially Fixed' if pct_fixed > 50 else 'Not Fixed'
                })
    
    if not results:
        print('⚠️  No Gap analysis results found')
        return None
    
    df = pd.DataFrame(results)
    
    # Summary at k=10
    df_k10 = df[df['k'] == 10].copy()
    
    if len(df_k10) == 0:
        print('⚠️  No k=10 results found, using max k available')
        max_k = df['k'].max()
        df_k10 = df[df['k'] == max_k].copy()
    
    print(f'\n--- The Gap Analysis (k={df_k10.iloc[0]["k"]}) ---')
    print(df_k10[['Dataset', 'SGC+MLP', 'Restricted+RowNorm', 'Final Gap (pp)', 'Pct Fixed', 'Status']].to_string(index=False))
    
    # Save
    df_k10.to_csv(os.path.join(output_dir, 'the_gap_analysis.csv'), index=False)
    
    # Statistics
    fully_fixed = (df_k10['Status'] == 'Fully Fixed').sum()
    partially_fixed = (df_k10['Status'] == 'Partially Fixed').sum()
    not_fixed = (df_k10['Status'] == 'Not Fixed').sum()
    total = len(df_k10)
    
    print(f'\nGAP CLOSURE STATISTICS:')
    print(f'  Fully fixed (beat SGC+MLP): {fully_fixed}/{total} ({fully_fixed/total*100:.1f}%)')
    print(f'  Partially fixed (>50% closed): {partially_fixed}/{total} ({partially_fixed/total*100:.1f}%)')
    print(f'  Not fixed (<50% closed): {not_fixed}/{total} ({not_fixed/total*100:.1f}%)')
    
    # Visualization: Three-method comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets = df_k10['Dataset'].values
    sgc_scores = [float(x.strip('%')) for x in df_k10['SGC+MLP'].values]
    std_scores = [float(x.strip('%')) for x in df_k10['Restricted+StandardMLP'].values]
    rownorm_scores = [float(x.strip('%')) for x in df_k10['Restricted+RowNorm'].values]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    ax.bar(x - width, sgc_scores, width, label='SGC+MLP (Baseline)', alpha=0.8, color='steelblue')
    ax.bar(x, std_scores, width, label='Restricted+StandardMLP (Part A)', alpha=0.8, color='coral')
    ax.bar(x + width, rownorm_scores, width, label='Restricted+RowNorm (Part B)', alpha=0.8, color='seagreen')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'The Gap: Full Method Comparison (k={df_k10.iloc[0]["k"]})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'the_gap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\n✓ Saved: {output_dir}/the_gap_analysis.csv')
    print(f'✓ Saved: {output_dir}/the_gap_comparison.png')
    
    return df_k10

# ============================================================================
# COMBINED VISUALIZATION
# ============================================================================

def create_combined_figure(part_a_df, part_b_df, gap_df, output_dir):
    """Create a single figure showing all three analyses"""
    
    if part_a_df is None or part_b_df is None or gap_df is None:
        print('\n⚠️  Skipping combined figure (missing data)')
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = part_a_df['Dataset'].values
    x = np.arange(len(datasets))
    
    # Part A
    sgc_scores = [float(x.strip('%')) for x in part_a_df['SGC+MLP'].values]
    restricted_scores = [float(x.strip('%')) for x in part_a_df['Restricted+StandardMLP'].values]
    
    axes[0].bar(x - 0.2, sgc_scores, 0.4, label='SGC+MLP', alpha=0.8)
    axes[0].bar(x + 0.2, restricted_scores, 0.4, label='Restricted+Std', alpha=0.8)
    axes[0].set_title('Part A: Basis Sensitivity', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Part B
    std_scores = [float(x.strip('%')) for x in part_b_df['Restricted+StandardMLP'].values]
    rownorm_scores = [float(x.strip('%')) for x in part_b_df['Restricted+RowNorm'].values]
    
    axes[1].bar(x - 0.2, std_scores, 0.4, label='Restricted+Std', alpha=0.8)
    axes[1].bar(x + 0.2, rownorm_scores, 0.4, label='Restricted+RowNorm', alpha=0.8)
    axes[1].set_title('Part B: RowNorm Improvement', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # The Gap
    sgc_scores_gap = [float(x.strip('%')) for x in gap_df['SGC+MLP'].values]
    rownorm_scores_gap = [float(x.strip('%')) for x in gap_df['Restricted+RowNorm'].values]
    
    axes[2].bar(x - 0.2, sgc_scores_gap, 0.4, label='SGC+MLP', alpha=0.8)
    axes[2].bar(x + 0.2, rownorm_scores_gap, 0.4, label='Restricted+RowNorm', alpha=0.8)
    axes[2].set_title('The Gap: Final Comparison', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets, rotation=45, ha='right')
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Part A/B Framework Analysis (k={part_a_df.iloc[0]["k"]}, {SPLIT_TYPE} splits)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_part_ab_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\n✓ Saved: {output_dir}/combined_part_ab_analysis.png')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if len(all_data) > 0:
    part_a_df = analyze_part_a(all_data, OUTPUT_DIR)
    part_b_df = analyze_part_b(all_data, OUTPUT_DIR)
    gap_df = analyze_the_gap(all_data, OUTPUT_DIR)
    
    create_combined_figure(part_a_df, part_b_df, gap_df, OUTPUT_DIR)
    
    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print(f'\nAll results saved to: {OUTPUT_DIR}/')
    print('\nKey files:')
    print('  - part_a_basis_sensitivity.csv')
    print('  - part_b_rownorm_improvement.csv')
    print('  - the_gap_analysis.csv')
    print('  - combined_part_ab_analysis.png')
else:
    print('\n⚠️  No data found!')