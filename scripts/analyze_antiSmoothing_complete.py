"""
===================================================================================
COMPLETE SGC GLOBAL ANTI-SMOOTHING ANALYSIS
===================================================================================

Comprehensive analysis covering:
1. All JL compression ratios (nc, 2nc, 4nc) 
2. Both projection types (Binary, Orthogonal)
3. Anti-smoothing effect validation
4. Parameter efficiency analysis
5. Per-dataset detailed breakdowns
6. Method comparison and recommendations
7. Failure mode identification
8. Cross-dataset pattern analysis

Author: Mohammad Davari
Date: November 2024
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
from collections import defaultdict
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Complete Anti-Smoothing Analysis')
parser.add_argument('--results_dir', type=str, 
                   default='results/investigation_sgc_antiSmoothing',
                   help='Base directory containing results')
args = parser.parse_args()

RESULTS_BASE = args.results_dir
OUTPUT_DIR = os.path.join(RESULTS_BASE, 'summary')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_percentage(value, default='N/A'):
    """Safely format percentage values"""
    if value is None or np.isnan(value):
        return default
    return f'{value*100:.2f}%'

def safe_improvement(val1, val2, default='N/A'):
    """Safely calculate improvement"""
    if val1 is None or val2 is None or np.isnan(val1) or np.isnan(val2):
        return default
    return f'{(val1 - val2)*100:+.2f}pp'

def format_with_k(value, k, default='N/A'):
    """Format value with k indicator"""
    if value is None or np.isnan(value):
        return default
    return f'{value*100:.2f}% (k={k})'

# ============================================================================
# PHASE 1: AUTO-DISCOVERY
# ============================================================================

def discover_experiments(base_dir):
    """
    Discover all experiments across datasets, splits, components, and k values.
    
    Returns:
        experiments: dict[split_type][dataset][k] = {results, parameter_counts}
    """
    print('='*80)
    print('AUTO-DISCOVERY ANALYSIS')
    print('='*80)
    print(f'Scanning: {base_dir}')
    print('='*80)
    
    experiments = {'fixed': {}, 'random': {}}
    
    # Scan directory structure
    if not os.path.exists(base_dir):
        print(f'ERROR: Results directory not found: {base_dir}')
        return experiments
    
    print('\nDiscovering experiments...\n')
    
    for dataset_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, dataset_folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Parse folder name: dataset_splits_component (e.g., "ogbn-arxiv_fixed_lcc")
        parts = dataset_folder.rsplit('_', 2)
        if len(parts) != 3:
            continue
        
        dataset, split_type, component = parts
        
        if split_type not in ['fixed', 'random']:
            continue
        
        # Find all k values
        k_values = []
        for k_folder in os.listdir(folder_path):
            if k_folder.startswith('k'):
                try:
                    k = int(k_folder[1:])
                    k_values.append(k)
                except ValueError:
                    continue
        
        if k_values:
            if dataset not in experiments[split_type]:
                experiments[split_type][dataset] = {}
            
            for k in sorted(k_values):
                results_file = os.path.join(folder_path, f'k{k}', 'metrics', 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        experiments[split_type][dataset][k] = data
    
    # Print discovery summary
    print('Found experiments:')
    fixed_datasets = list(experiments['fixed'].keys())
    random_datasets = list(experiments['random'].keys())
    print(f'  Fixed splits:  {len(fixed_datasets)} datasets')
    print(f'  Random splits: {len(random_datasets)} datasets')
    
    if fixed_datasets:
        print('\nFixed split datasets:')
        for dataset in sorted(fixed_datasets):
            k_vals = sorted(experiments['fixed'][dataset].keys())
            print(f'  - {dataset:20} k={k_vals}')
    
    if random_datasets:
        print('\nRandom split datasets:')
        for dataset in sorted(random_datasets):
            k_vals = sorted(experiments['random'][dataset].keys())
            print(f'  - {dataset:20} k={k_vals}')
    
    both = set(fixed_datasets) & set(random_datasets)
    if both:
        print(f'\nDatasets with BOTH split types: {sorted(both)}')
    
    return experiments

# ============================================================================
# PHASE 2: TABLE 1 - BEST RESULTS (COMPLETE WITH ALL JL VARIANTS)
# ============================================================================

def generate_table1_complete(data, split_type, output_dir):
    """
    Generate comprehensive Table 1 showing best results for ALL methods including
    all JL compression ratios (nc, 2nc, 4nc) for both binary and orthogonal.
    """
    print('\n' + '='*80)
    print(f'TABLE 1: BEST RESULTS PER DATASET (ALL METHODS) - {split_type.upper()}')
    print('='*80)
    
    # Define all methods to track
    method_configs = {
        'SGC': 'sgc_baseline',
        'SGC+MLP': 'sgc_mlp_baseline',
        'Full+RowNorm': 'full_rownorm_mlp',
        'JL Binary (nc)': 'jl_binary_nc',
        'JL Binary (2nc)': 'jl_binary_2nc',
        'JL Binary (4nc)': 'jl_binary_4nc',
        'JL Ortho (nc)': 'jl_ortho_nc',
        'JL Ortho (2nc)': 'jl_ortho_2nc',
        'JL Ortho (4nc)': 'jl_ortho_4nc'
    }
    
    table_data = []
    
    for dataset in sorted(data.keys()):
        if not data[dataset]:
            continue
        
        row = {'Dataset': dataset}
        
        # Find best k and accuracy for each method
        for display_name, method_key in method_configs.items():
            best_k = None
            best_acc = -1
            
            for k in data[dataset]:
                results = data[dataset][k].get('results', {})
                if method_key in results:
                    acc = results[method_key].get('test_acc_mean', np.nan)
                    if not np.isnan(acc) and acc > best_acc:
                        best_acc = acc
                        best_k = k
            
            if best_k is not None and not np.isnan(best_acc):
                row[display_name] = format_with_k(best_acc, best_k)
            else:
                row[display_name] = 'N/A'
        
        # Calculate improvements
        sgc_acc = None
        full_acc = None
        
        for k in data[dataset]:
            results = data[dataset][k].get('results', {})
            if 'sgc_baseline' in results:
                acc = results['sgc_baseline'].get('test_acc_mean', np.nan)
                if not np.isnan(acc) and (sgc_acc is None or acc > sgc_acc):
                    sgc_acc = acc
            if 'full_rownorm_mlp' in results:
                acc = results['full_rownorm_mlp'].get('test_acc_mean', np.nan)
                if not np.isnan(acc) and (full_acc is None or acc > full_acc):
                    full_acc = acc
        
        row['Improvement (Full)'] = safe_improvement(full_acc, sgc_acc)
        
        table_data.append(row)
    
    if not table_data:
        print('⚠ No data found for table generation')
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'table1_best_results_complete.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Saved: {csv_path}')
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, 'table1_best_results_complete.tex')
    with open(tex_path, 'w') as f:
        latex_str = df.to_latex(index=False, escape=False, column_format='l' + 'c'*(len(df.columns)-1))
        f.write(latex_str)
    print(f'✓ Saved: {tex_path}')
    
    # Print to console (abbreviated for readability)
    print('\n' + df[['Dataset', 'SGC', 'SGC+MLP', 'Full+RowNorm', 
                      'JL Binary (2nc)', 'JL Ortho (2nc)', 'Improvement (Full)']].to_string(index=False))
    
    return df

# ============================================================================
# PHASE 3: TABLE 1B - JL COMPRESSION COMPARISON
# ============================================================================

def generate_table1b_jl_compression(data, split_type, output_dir):
    """
    Compare nc vs 2nc vs 4nc for both binary and orthogonal projections.
    Shows how compression ratio affects accuracy.
    """
    print('\n' + '='*80)
    print(f'TABLE 1B: JL COMPRESSION RATIO COMPARISON - {split_type.upper()}')
    print('='*80)
    
    # Analyze at k=10 (highest diffusion)
    k_target = 10
    
    binary_data = []
    ortho_data = []
    
    for dataset in sorted(data.keys()):
        if k_target not in data[dataset]:
            continue
        
        results = data[dataset][k_target].get('results', {})
        params = data[dataset][k_target].get('parameter_counts', {})
        
        # Binary projections
        binary_row = {'Dataset': dataset, 'k': k_target}
        for ratio in ['nc', '2nc', '4nc']:
            key = f'jl_binary_{ratio}'
            if key in results:
                acc = results[key].get('test_acc_mean', np.nan) * 100
                param_count = params.get(key, 0)
                binary_row[f'{ratio}'] = f'{acc:.2f}%'
                binary_row[f'{ratio} params'] = f'{param_count:,}'
            else:
                binary_row[f'{ratio}'] = 'N/A'
                binary_row[f'{ratio} params'] = 'N/A'
        
        if any(binary_row[f'{r}'] != 'N/A' for r in ['nc', '2nc', '4nc']):
            binary_data.append(binary_row)
        
        # Orthogonal projections
        ortho_row = {'Dataset': dataset, 'k': k_target}
        for ratio in ['nc', '2nc', '4nc']:
            key = f'jl_ortho_{ratio}'
            if key in results:
                acc = results[key].get('test_acc_mean', np.nan) * 100
                param_count = params.get(key, 0)
                ortho_row[f'{ratio}'] = f'{acc:.2f}%'
                ortho_row[f'{ratio} params'] = f'{param_count:,}'
            else:
                ortho_row[f'{ratio}'] = 'N/A'
                ortho_row[f'{ratio} params'] = 'N/A'
        
        if any(ortho_row[f'{r}'] != 'N/A' for r in ['nc', '2nc', '4nc']):
            ortho_data.append(ortho_row)
    
    # Create DataFrames
    if binary_data:
        df_binary = pd.DataFrame(binary_data)
        print('\n--- Binary JL Projections (k=10) ---')
        print(df_binary[['Dataset', 'nc', '2nc', '4nc']].to_string(index=False))
        df_binary.to_csv(os.path.join(output_dir, 'table1b_jl_binary_compression.csv'), index=False)
        
        # Save LaTeX
        with open(os.path.join(output_dir, 'table1b_jl_binary_compression.tex'), 'w') as f:
            f.write(df_binary[['Dataset', 'nc', '2nc', '4nc']].to_latex(index=False, escape=False))
    
    if ortho_data:
        df_ortho = pd.DataFrame(ortho_data)
        print('\n--- Orthogonal JL Projections (k=10) ---')
        print(df_ortho[['Dataset', 'nc', '2nc', '4nc']].to_string(index=False))
        df_ortho.to_csv(os.path.join(output_dir, 'table1b_jl_ortho_compression.csv'), index=False)
        
        # Save LaTeX
        with open(os.path.join(output_dir, 'table1b_jl_ortho_compression.tex'), 'w') as f:
            f.write(df_ortho[['Dataset', 'nc', '2nc', '4nc']].to_latex(index=False, escape=False))
    
    return df_binary if binary_data else None, df_ortho if ortho_data else None

# ============================================================================
# PHASE 4: TABLE 2 - ANTI-SMOOTHING EFFECT
# ============================================================================

def generate_table2_anti_smoothing(data, split_type, output_dir):
    """
    Validate anti-smoothing hypothesis by comparing k=2 vs k=10.
    Expected: SGC degrades (oversmoothing), our method improves (anti-smoothing).
    """
    print('\n' + '='*80)
    print(f'TABLE 2: ANTI-SMOOTHING EFFECT (k=2 vs k=10) - {split_type.upper()}')
    print('='*80)
    
    table_data = []
    anti_smoothing_count = 0
    
    for dataset in sorted(data.keys()):
        if 2 not in data[dataset] or 10 not in data[dataset]:
            continue
        
        # Extract k=2 and k=10 results
        results_k2 = data[dataset][2].get('results', {})
        results_k10 = data[dataset][10].get('results', {})
        
        sgc_k2 = results_k2.get('sgc_baseline', {}).get('test_acc_mean', np.nan) * 100
        sgc_k10 = results_k10.get('sgc_baseline', {}).get('test_acc_mean', np.nan) * 100
        full_k2 = results_k2.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan) * 100
        full_k10 = results_k10.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan) * 100
        
        if not all(np.isnan([sgc_k2, sgc_k10, full_k2, full_k10])):
            sgc_change = sgc_k10 - sgc_k2
            full_change = full_k10 - full_k2
            
            # Anti-smoothing: SGC degrades AND our method improves
            anti_smoothing = '✓' if (sgc_change < 0 and full_change > 0) else '✗'
            if anti_smoothing == '✓':
                anti_smoothing_count += 1
            
            table_data.append({
                'Dataset': dataset,
                'SGC @ k=2': f'{sgc_k2:.2f}%',
                'SGC @ k=10': f'{sgc_k10:.2f}%',
                'SGC Change': f'{sgc_change:+.2f}pp',
                'Full @ k=2': f'{full_k2:.2f}%',
                'Full @ k=10': f'{full_k10:.2f}%',
                'Full Change': f'{full_change:+.2f}pp',
                'Anti-Smoothing': anti_smoothing
            })
    
    if not table_data:
        print('⚠ No data found with both k=2 and k=10')
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'table2_anti_smoothing.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Saved: {csv_path}')
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, 'table2_anti_smoothing.tex')
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f'✓ Saved: {tex_path}')
    
    # Print to console
    print('\n' + df.to_string(index=False))
    
    print(f'\n✓ Anti-smoothing effect observed in {anti_smoothing_count}/{len(table_data)} datasets')
    
    return df

# ============================================================================
# PHASE 5: TABLE 3 - PARAMETER EFFICIENCY (ALL VARIANTS)
# ============================================================================

def generate_table3_parameter_efficiency(data, split_type, output_dir):
    """
    Complete parameter efficiency analysis including ALL JL variants at k=10.
    Shows accuracy per 10K parameters for fair comparison.
    """
    print('\n' + '='*80)
    print(f'TABLE 3: PARAMETER EFFICIENCY (k=10, ALL VARIANTS) - {split_type.upper()}')
    print('='*80)
    
    k_target = 10
    table_data = []
    
    for dataset in sorted(data.keys()):
        if k_target not in data[dataset]:
            continue
        
        results = data[dataset][k_target].get('results', {})
        params = data[dataset][k_target].get('parameter_counts', {})
        
        # All methods including all JL variants
        methods = {
            'SGC+MLP': 'sgc_mlp_baseline',
            'Full+RowNorm': 'full_rownorm_mlp',
            'JL Binary (nc)': 'jl_binary_nc',
            'JL Binary (2nc)': 'jl_binary_2nc',
            'JL Binary (4nc)': 'jl_binary_4nc',
            'JL Ortho (nc)': 'jl_ortho_nc',
            'JL Ortho (2nc)': 'jl_ortho_2nc',
            'JL Ortho (4nc)': 'jl_ortho_4nc'
        }
        
        for display_name, method_key in methods.items():
            if method_key in results:
                acc = results[method_key].get('test_acc_mean', np.nan) * 100
                param_count = params.get(method_key, 0)
                
                if not np.isnan(acc) and param_count > 0:
                    acc_per_10k = acc / (param_count / 10000)
                    
                    table_data.append({
                        'Dataset': dataset,
                        'Method': display_name,
                        'Accuracy (%)': f'{acc:.2f}',
                        'Parameters': f'{param_count:,}',
                        'Acc/10K Params': f'{acc_per_10k:.4f}'
                    })
    
    if not table_data:
        print('⚠ No data found for k=10')
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'table3_parameter_efficiency_complete.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Saved: {csv_path}')
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, 'table3_parameter_efficiency_complete.tex')
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f'✓ Saved: {tex_path}')
    
    # Print summary (abbreviated)
    print('\n' + df.to_string(index=False))
    
    return df

# ============================================================================
# PHASE 6: TABLE 4 - BEST JL CONFIGURATION PER DATASET
# ============================================================================

def generate_table4_best_jl_config(data, split_type, output_dir):
    """
    Identify the best JL configuration (projection type + compression ratio) 
    for each dataset across all k values.
    """
    print('\n' + '='*80)
    print(f'TABLE 4: BEST JL CONFIGURATION PER DATASET - {split_type.upper()}')
    print('='*80)
    
    table_data = []
    
    for dataset in sorted(data.keys()):
        if not data[dataset]:
            continue
        
        best_config = None
        best_acc = -1
        best_k = None
        best_params = 0
        
        # Search across all k values and JL variants
        for k in data[dataset]:
            results = data[dataset][k].get('results', {})
            params = data[dataset][k].get('parameter_counts', {})
            
            for ratio in ['nc', '2nc', '4nc']:
                for proj_type in ['binary', 'ortho']:
                    key = f'jl_{proj_type}_{ratio}'
                    if key in results:
                        acc = results[key].get('test_acc_mean', np.nan)
                        if not np.isnan(acc) and acc > best_acc:
                            best_acc = acc
                            best_config = f'{proj_type.capitalize()} {ratio}'
                            best_k = k
                            best_params = params.get(key, 0)
        
        if best_config:
            # Compare to SGC and Full+RowNorm baselines
            sgc_best = -1
            full_best = -1
            
            for k in data[dataset]:
                results = data[dataset][k].get('results', {})
                if 'sgc_baseline' in results:
                    acc = results['sgc_baseline'].get('test_acc_mean', np.nan)
                    if not np.isnan(acc) and acc > sgc_best:
                        sgc_best = acc
                if 'full_rownorm_mlp' in results:
                    acc = results['full_rownorm_mlp'].get('test_acc_mean', np.nan)
                    if not np.isnan(acc) and acc > full_best:
                        full_best = acc
            
            table_data.append({
                'Dataset': dataset,
                'Best JL Config': best_config,
                'k': best_k,
                'Accuracy': f'{best_acc*100:.2f}%',
                'Parameters': f'{best_params:,}',
                'vs SGC': safe_improvement(best_acc, sgc_best),
                'vs Full': safe_improvement(best_acc, full_best),
                'Beats SGC': '✓' if best_acc > sgc_best else '✗',
                'Beats Full': '✓' if best_acc > full_best else '✗'
            })
    
    if not table_data:
        print('⚠ No JL projection data found')
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'table4_best_jl_config.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Saved: {csv_path}')
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, 'table4_best_jl_config.tex')
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f'✓ Saved: {tex_path}')
    
    # Print to console
    print('\n' + df.to_string(index=False))
    
    # Summary statistics
    beats_sgc = sum(1 for row in table_data if row['Beats SGC'] == '✓')
    beats_full = sum(1 for row in table_data if row['Beats Full'] == '✓')
    print(f'\n✓ Best JL config beats SGC: {beats_sgc}/{len(table_data)} datasets')
    print(f'✓ Best JL config beats Full: {beats_full}/{len(table_data)} datasets')
    
    return df

# ============================================================================
# PHASE 7: TABLE 5 - BINARY VS ORTHOGONAL COMPARISON
# ============================================================================

def generate_table5_binary_vs_ortho(data, split_type, output_dir):
    """
    Direct comparison between binary and orthogonal JL projections 
    at same compression ratios.
    """
    print('\n' + '='*80)
    print(f'TABLE 5: BINARY VS ORTHOGONAL PROJECTION COMPARISON - {split_type.upper()}')
    print('='*80)
    
    k_target = 10
    table_data = []
    
    for dataset in sorted(data.keys()):
        if k_target not in data[dataset]:
            continue
        
        results = data[dataset][k_target].get('results', {})
        
        for ratio in ['nc', '2nc', '4nc']:
            binary_key = f'jl_binary_{ratio}'
            ortho_key = f'jl_ortho_{ratio}'
            
            if binary_key in results and ortho_key in results:
                binary_acc = results[binary_key].get('test_acc_mean', np.nan) * 100
                ortho_acc = results[ortho_key].get('test_acc_mean', np.nan) * 100
                
                if not np.isnan(binary_acc) and not np.isnan(ortho_acc):
                    diff = ortho_acc - binary_acc
                    winner = 'Ortho' if diff > 0.5 else 'Binary' if diff < -0.5 else 'Tie'
                    
                    table_data.append({
                        'Dataset': dataset,
                        'Ratio': ratio,
                        'Binary': f'{binary_acc:.2f}%',
                        'Orthogonal': f'{ortho_acc:.2f}%',
                        'Difference': f'{diff:+.2f}pp',
                        'Winner': winner
                    })
    
    if not table_data:
        print('⚠ No comparison data found')
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'table5_binary_vs_ortho.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Saved: {csv_path}')
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, 'table5_binary_vs_ortho.tex')
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f'✓ Saved: {tex_path}')
    
    # Print to console
    print('\n' + df.to_string(index=False))
    
    # Summary statistics
    binary_wins = sum(1 for row in table_data if row['Winner'] == 'Binary')
    ortho_wins = sum(1 for row in table_data if row['Winner'] == 'Ortho')
    ties = sum(1 for row in table_data if row['Winner'] == 'Tie')
    
    print(f'\n✓ Binary wins: {binary_wins}/{len(table_data)}')
    print(f'✓ Orthogonal wins: {ortho_wins}/{len(table_data)}')
    print(f'✓ Ties: {ties}/{len(table_data)}')
    
    return df

# ============================================================================
# PHASE 8: TABLE 6 - FAILURE MODE ANALYSIS
# ============================================================================

def generate_table6_failure_analysis(data, split_type, output_dir):
    """
    Identify and analyze cases where Full+RowNorm fails to beat SGC baseline.
    """
    print('\n' + '='*80)
    print(f'TABLE 6: FAILURE MODE ANALYSIS - {split_type.upper()}')
    print('='*80)
    
    table_data = []
    
    for dataset in sorted(data.keys()):
        if not data[dataset]:
            continue
        
        # Find best SGC and Full across all k
        best_sgc = -1
        best_sgc_k = None
        best_full = -1
        best_full_k = None
        
        for k in data[dataset]:
            results = data[dataset][k].get('results', {})
            
            if 'sgc_baseline' in results:
                acc = results['sgc_baseline'].get('test_acc_mean', np.nan)
                if not np.isnan(acc) and acc > best_sgc:
                    best_sgc = acc
                    best_sgc_k = k
            
            if 'full_rownorm_mlp' in results:
                acc = results['full_rownorm_mlp'].get('test_acc_mean', np.nan)
                if not np.isnan(acc) and acc > best_full:
                    best_full = acc
                    best_full_k = k
        
        if best_sgc >= 0 and best_full >= 0:
            improvement = (best_full - best_sgc) * 100
            
            # Classify failure severity
            if improvement < 0:
                if improvement < -10:
                    severity = 'Critical'
                elif improvement < -5:
                    severity = 'Major'
                else:
                    severity = 'Minor'
                
                table_data.append({
                    'Dataset': dataset,
                    'Best SGC': f'{best_sgc*100:.2f}% (k={best_sgc_k})',
                    'Best Full': f'{best_full*100:.2f}% (k={best_full_k})',
                    'Gap': f'{improvement:.2f}pp',
                    'Severity': severity
                })
    
    if not table_data:
        print('✓ No failures detected - method outperforms SGC on all datasets!')
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'table6_failure_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f'✓ Saved: {csv_path}')
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, 'table6_failure_analysis.tex')
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f'✓ Saved: {tex_path}')
    
    # Print to console
    print('\n' + df.to_string(index=False))
    
    # Summary
    critical = sum(1 for row in table_data if row['Severity'] == 'Critical')
    major = sum(1 for row in table_data if row['Severity'] == 'Major')
    minor = sum(1 for row in table_data if row['Severity'] == 'Minor')
    
    print(f'\n⚠ Failure summary:')
    print(f'  Critical (< -10pp): {critical}')
    print(f'  Major (-10pp to -5pp): {major}')
    print(f'  Minor (-5pp to 0pp): {minor}')
    
    return df

# ============================================================================
# PHASE 9: FIGURE 1 - K-VALUE TRAJECTORIES (ALL METHODS)
# ============================================================================

def generate_figure1_k_trajectories(data, split_type, output_dir):
    """
    Plot accuracy trajectories across k values for all methods.
    Shows SGC oversmoothing vs our anti-smoothing effect.
    """
    print('\n' + '='*80)
    print(f'FIGURE 1: K-VALUE TRAJECTORIES - {split_type.upper()}')
    print('='*80)
    
    # Separate plots for each dataset
    datasets = sorted(data.keys())
    if not datasets:
        print('⚠ No data to plot')
        return
    
    # Create grid of subplots
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Methods to plot
    methods = {
        'SGC': ('sgc_baseline', 'red', 's', '-'),
        'SGC+MLP': ('sgc_mlp_baseline', 'orange', 'D', '--'),
        'Full+RowNorm': ('full_rownorm_mlp', 'blue', 'o', '-'),
        'JL Binary (2nc)': ('jl_binary_2nc', 'green', '^', '-.'),
        'JL Ortho (2nc)': ('jl_ortho_2nc', 'purple', 'v', ':')
    }
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Collect data for this dataset
        for method_name, (method_key, color, marker, linestyle) in methods.items():
            k_values = []
            accuracies = []
            
            for k in sorted(data[dataset].keys()):
                results = data[dataset][k].get('results', {})
                if method_key in results:
                    acc = results[method_key].get('test_acc_mean', np.nan)
                    if not np.isnan(acc):
                        k_values.append(k)
                        accuracies.append(acc * 100)
            
            if k_values:
                ax.plot(k_values, accuracies, marker=marker, linestyle=linestyle, 
                       color=color, label=method_name, linewidth=2, markersize=6)
        
        ax.set_xlabel('k (Diffusion Steps)', fontsize=10)
        ax.set_ylabel('Test Accuracy (%)', fontsize=10)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure1_k_trajectories.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {fig_path}')

# ============================================================================
# PHASE 10: FIGURE 2 - IMPROVEMENT HEATMAP
# ============================================================================

def generate_figure2_improvement_heatmap(data, split_type, output_dir):
    """
    Heatmap showing improvement of Full+RowNorm over SGC across datasets and k values.
    """
    print('\n' + '='*80)
    print(f'FIGURE 2: IMPROVEMENT HEATMAP - {split_type.upper()}')
    print('='*80)
    
    # Collect improvement data
    datasets = sorted(data.keys())
    all_k = set()
    for dataset in datasets:
        all_k.update(data[dataset].keys())
    k_values = sorted(all_k)
    
    # Build matrix: datasets x k_values
    improvement_matrix = []
    
    for dataset in datasets:
        row = []
        for k in k_values:
            if k in data[dataset]:
                results = data[dataset][k].get('results', {})
                sgc_acc = results.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
                full_acc = results.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
                
                if not np.isnan(sgc_acc) and not np.isnan(full_acc):
                    improvement = (full_acc - sgc_acc) * 100
                    row.append(improvement)
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)
        improvement_matrix.append(row)
    
    improvement_matrix = np.array(improvement_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, len(datasets) * 0.6))
    
    # Use diverging colormap centered at 0
    vmax = max(abs(np.nanmin(improvement_matrix)), abs(np.nanmax(improvement_matrix)))
    vmin = -vmax
    
    sns.heatmap(improvement_matrix, 
                xticklabels=[f'k={k}' for k in k_values],
                yticklabels=datasets,
                cmap='RdYlGn',  # Red (negative) to Green (positive)
                center=0,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt='.1f',
                cbar_kws={'label': 'Improvement (pp)'},
                ax=ax)
    
    ax.set_title(f'Full+RowNorm Improvement over SGC ({split_type} splits)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Diffusion Steps', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure2_improvement_heatmap.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {fig_path}')

# ============================================================================
# PHASE 11: FIGURE 3 - JL COMPRESSION COMPARISON
# ============================================================================

def generate_figure3_jl_compression(data, split_type, output_dir):
    """
    Compare nc vs 2nc vs 4nc for both binary and orthogonal at k=10.
    """
    print('\n' + '='*80)
    print(f'FIGURE 3: JL COMPRESSION RATIO COMPARISON - {split_type.upper()}')
    print('='*80)
    
    k_target = 10
    datasets = sorted(data.keys())
    
    # Collect data
    binary_nc = []
    binary_2nc = []
    binary_4nc = []
    ortho_nc = []
    ortho_2nc = []
    ortho_4nc = []
    valid_datasets = []
    
    for dataset in datasets:
        if k_target not in data[dataset]:
            continue
        
        results = data[dataset][k_target].get('results', {})
        
        # Check if we have all variants
        has_all = all(f'jl_binary_{r}' in results and f'jl_ortho_{r}' in results 
                     for r in ['nc', '2nc', '4nc'])
        
        if has_all:
            binary_nc.append(results['jl_binary_nc']['test_acc_mean'] * 100)
            binary_2nc.append(results['jl_binary_2nc']['test_acc_mean'] * 100)
            binary_4nc.append(results['jl_binary_4nc']['test_acc_mean'] * 100)
            ortho_nc.append(results['jl_ortho_nc']['test_acc_mean'] * 100)
            ortho_2nc.append(results['jl_ortho_2nc']['test_acc_mean'] * 100)
            ortho_4nc.append(results['jl_ortho_4nc']['test_acc_mean'] * 100)
            valid_datasets.append(dataset)
    
    if not valid_datasets:
        print('⚠ No complete JL compression data found')
        return
    
    # Create grouped bar chart
    x = np.arange(len(valid_datasets))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - 2.5*width, binary_nc, width, label='Binary nc', color='#1f77b4')
    ax.bar(x - 1.5*width, binary_2nc, width, label='Binary 2nc', color='#ff7f0e')
    ax.bar(x - 0.5*width, binary_4nc, width, label='Binary 4nc', color='#2ca02c')
    ax.bar(x + 0.5*width, ortho_nc, width, label='Ortho nc', color='#d62728')
    ax.bar(x + 1.5*width, ortho_2nc, width, label='Ortho 2nc', color='#9467bd')
    ax.bar(x + 2.5*width, ortho_4nc, width, label='Ortho 4nc', color='#8c564b')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'JL Compression Ratio Comparison at k=10 ({split_type} splits)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_datasets, rotation=45, ha='right')
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure3_jl_compression_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {fig_path}')

# ============================================================================
# PHASE 12: FIGURE 4 - PARAMETER EFFICIENCY SCATTER
# ============================================================================

def generate_figure4_parameter_efficiency(data, split_type, output_dir):
    """
    Scatter plot: Accuracy vs Parameters for all methods at k=10.
    Shows which methods are on the Pareto frontier.
    """
    print('\n' + '='*80)
    print(f'FIGURE 4: PARAMETER EFFICIENCY SCATTER - {split_type.upper()}')
    print('='*80)
    
    k_target = 10
    
    # Collect data points
    plot_data = []
    
    methods = {
        'SGC+MLP': ('sgc_mlp_baseline', 'orange', 's'),
        'Full+RowNorm': ('full_rownorm_mlp', 'blue', 'o'),
        'JL Binary (nc)': ('jl_binary_nc', 'green', '^'),
        'JL Binary (2nc)': ('jl_binary_2nc', 'green', 'v'),
        'JL Binary (4nc)': ('jl_binary_4nc', 'green', '<'),
        'JL Ortho (nc)': ('jl_ortho_nc', 'purple', '^'),
        'JL Ortho (2nc)': ('jl_ortho_2nc', 'purple', 'v'),
        'JL Ortho (4nc)': ('jl_ortho_4nc', 'purple', '<')
    }
    
    for dataset in sorted(data.keys()):
        if k_target not in data[dataset]:
            continue
        
        results = data[dataset][k_target].get('results', {})
        params = data[dataset][k_target].get('parameter_counts', {})
        
        for method_name, (method_key, color, marker) in methods.items():
            if method_key in results:
                acc = results[method_key].get('test_acc_mean', np.nan) * 100
                param_count = params.get(method_key, 0)
                
                if not np.isnan(acc) and param_count > 0:
                    plot_data.append({
                        'method': method_name,
                        'dataset': dataset,
                        'accuracy': acc,
                        'parameters': param_count,
                        'color': color,
                        'marker': marker
                    })
    
    if not plot_data:
        print('⚠ No parameter efficiency data found')
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot by method
    for method_name in methods.keys():
        method_points = [p for p in plot_data if p['method'] == method_name]
        if method_points:
            x = [p['parameters'] for p in method_points]
            y = [p['accuracy'] for p in method_points]
            color = method_points[0]['color']
            marker = method_points[0]['marker']
            
            ax.scatter(x, y, s=100, alpha=0.7, color=color, marker=marker, 
                      label=method_name, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Parameter Efficiency at k=10 ({split_type} splits)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure4_parameter_efficiency.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {fig_path}')

# ============================================================================
# PHASE 13: FIGURE 5 - ANTI-SMOOTHING EFFECT VISUALIZATION
# ============================================================================

def generate_figure5_anti_smoothing_effect(data, split_type, output_dir):
    """
    Visual comparison of SGC degradation vs Full improvement from k=2 to k=10.
    """
    print('\n' + '='*80)
    print(f'FIGURE 5: ANTI-SMOOTHING EFFECT VISUALIZATION - {split_type.upper()}')
    print('='*80)
    
    # Collect data
    datasets_list = []
    sgc_changes = []
    full_changes = []
    
    for dataset in sorted(data.keys()):
        if 2 not in data[dataset] or 10 not in data[dataset]:
            continue
        
        results_k2 = data[dataset][2].get('results', {})
        results_k10 = data[dataset][10].get('results', {})
        
        sgc_k2 = results_k2.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
        sgc_k10 = results_k10.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
        full_k2 = results_k2.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
        full_k10 = results_k10.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
        
        if not any(np.isnan([sgc_k2, sgc_k10, full_k2, full_k10])):
            datasets_list.append(dataset)
            sgc_changes.append((sgc_k10 - sgc_k2) * 100)
            full_changes.append((full_k10 - full_k2) * 100)
    
    if not datasets_list:
        print('⚠ No data with both k=2 and k=10')
        return
    
    # Create side-by-side bar chart
    x = np.arange(len(datasets_list))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, sgc_changes, width, label='SGC (Oversmoothing)', 
                   color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, full_changes, width, label='Full+RowNorm (Anti-Smoothing)', 
                   color='green', alpha=0.7)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Color bars based on positive/negative
    for bar, change in zip(bars1, sgc_changes):
        if change < 0:
            bar.set_color('darkred')
        else:
            bar.set_color('lightcoral')
    
    for bar, change in zip(bars2, full_changes):
        if change > 0:
            bar.set_color('darkgreen')
        else:
            bar.set_color('lightgreen')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Accuracy Change (pp)', fontsize=12)
    ax.set_title(f'Anti-Smoothing Effect: k=2 → k=10 ({split_type} splits)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_list, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations for anti-smoothing cases
    for i, (sgc_change, full_change) in enumerate(zip(sgc_changes, full_changes)):
        if sgc_change < 0 and full_change > 0:
            ax.text(i, max(sgc_change, full_change) + 1, '✓', 
                   ha='center', va='bottom', fontsize=16, color='green', fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure5_anti_smoothing_effect.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {fig_path}')

# ============================================================================
# PHASE 14: FIGURE 6 - METHOD COMPARISON PER DATASET
# ============================================================================

def generate_figure6_method_comparison(data, split_type, output_dir):
    """
    Bar charts showing all methods at best k for each dataset.
    """
    print('\n' + '='*80)
    print(f'FIGURE 6: METHOD COMPARISON PER DATASET - {split_type.upper()}')
    print('='*80)
    
    datasets = sorted(data.keys())
    if not datasets:
        print('⚠ No data to plot')
        return
    
    # Methods to compare
    methods = [
        'sgc_baseline',
        'sgc_mlp_baseline',
        'full_rownorm_mlp',
        'jl_binary_2nc',
        'jl_ortho_2nc'
    ]
    
    method_labels = [
        'SGC',
        'SGC+MLP',
        'Full+RowNorm',
        'JL Binary (2nc)',
        'JL Ortho (2nc)'
    ]
    
    # Create grid
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Find best k for each method
        best_accs = []
        
        for method_key in methods:
            best_acc = -1
            for k in data[dataset]:
                results = data[dataset][k].get('results', {})
                if method_key in results:
                    acc = results[method_key].get('test_acc_mean', np.nan)
                    if not np.isnan(acc) and acc > best_acc:
                        best_acc = acc
            
            best_accs.append(best_acc * 100 if best_acc >= 0 else 0)
        
        # Create bar chart
        x = np.arange(len(method_labels))
        bars = ax.bar(x, best_accs, color=['red', 'orange', 'blue', 'green', 'purple'], 
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Highlight best method
        max_idx = np.argmax(best_accs)
        bars[max_idx].set_edgecolor('gold')
        bars[max_idx].set_linewidth(3)
        
        ax.set_ylabel('Test Accuracy (%)', fontsize=10)
        ax.set_title(f'{dataset}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Best Accuracy Comparison ({split_type} splits)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure6_method_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: {fig_path}')

# ============================================================================
# PHASE 15: PER-DATASET DETAILED REPORTS
# ============================================================================

def generate_per_dataset_reports(data, split_type, output_dir):
    """
    Generate detailed text reports for each dataset.
    """
    print('\n' + '='*80)
    print(f'GENERATING PER-DATASET DETAILED REPORTS - {split_type.upper()}')
    print('='*80)
    
    reports_dir = os.path.join(output_dir, 'detailed_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    for dataset in sorted(data.keys()):
        if not data[dataset]:
            continue
        
        report_path = os.path.join(reports_dir, f'{dataset}_detailed.txt')
        
        with open(report_path, 'w') as f:
            f.write('='*80 + '\n')
            f.write(f'DETAILED ANALYSIS: {dataset.upper()} ({split_type} splits)\n')
            f.write('='*80 + '\n\n')
            
            # 1. Performance trajectory
            f.write('1. PERFORMANCE TRAJECTORY\n')
            f.write('-'*80 + '\n')
            f.write(f'{"k":>3} | {"SGC":>7} | {"SGC+MLP":>9} | {"Full":>7} | {"JL(2nc)":>9} | {"Improvement":>12}\n')
            f.write('-'*80 + '\n')
            
            k_values = sorted(data[dataset].keys())
            for k in k_values:
                results = data[dataset][k].get('results', {})
                sgc = results.get('sgc_baseline', {}).get('test_acc_mean', np.nan) * 100
                sgc_mlp = results.get('sgc_mlp_baseline', {}).get('test_acc_mean', np.nan) * 100
                full = results.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan) * 100
                jl2nc = results.get('jl_binary_2nc', {}).get('test_acc_mean', np.nan) * 100
                
                improvement = full - sgc if not np.isnan(full) and not np.isnan(sgc) else np.nan
                
                sgc_str = f'{sgc:.2f}%' if not np.isnan(sgc) else 'N/A'
                sgc_mlp_str = f'{sgc_mlp:.2f}%' if not np.isnan(sgc_mlp) else 'N/A'
                full_str = f'{full:.2f}%' if not np.isnan(full) else 'N/A'
                jl2nc_str = f'{jl2nc:.2f}%' if not np.isnan(jl2nc) else 'N/A'
                imp_str = f'{improvement:+.2f}pp' if not np.isnan(improvement) else 'N/A'
                
                f.write(f'{k:>3} | {sgc_str:>7} | {sgc_mlp_str:>9} | {full_str:>7} | {jl2nc_str:>9} | {imp_str:>12}\n')
            
            # 2. Anti-smoothing analysis
            if len(k_values) >= 2:
                k_min, k_max = k_values[0], k_values[-1]
                
                sgc_min = data[dataset][k_min]['results'].get('sgc_baseline', {}).get('test_acc_mean', np.nan) * 100
                sgc_max = data[dataset][k_max]['results'].get('sgc_baseline', {}).get('test_acc_mean', np.nan) * 100
                full_min = data[dataset][k_min]['results'].get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan) * 100
                full_max = data[dataset][k_max]['results'].get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan) * 100
                
                f.write('\n2. ANTI-SMOOTHING EFFECT\n')
                f.write('-'*80 + '\n')
                
                if not any(np.isnan([sgc_min, sgc_max, full_min, full_max])):
                    f.write(f'SGC:        {sgc_min:.2f}% (k={k_min}) → {sgc_max:.2f}% (k={k_max}) [{sgc_max-sgc_min:+.2f}pp]\n')
                    f.write(f'Our method: {full_min:.2f}% (k={k_min}) → {full_max:.2f}% (k={k_max}) [{full_max-full_min:+.2f}pp]\n')
                    
                    if (sgc_max - sgc_min) < 0 and (full_max - full_min) > 0:
                        f.write(f'\n✓ ANTI-SMOOTHING CONFIRMED: SGC degrades while our method improves!\n')
                    else:
                        f.write(f'\n✗ Anti-smoothing pattern not observed\n')
            
            # 3. Best configuration
            f.write('\n3. BEST CONFIGURATION\n')
            f.write('-'*80 + '\n')
            
            best_k = None
            best_acc = -1
            for k in k_values:
                acc = data[dataset][k]['results'].get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
                if not np.isnan(acc) and acc > best_acc:
                    best_acc = acc
                    best_k = k
            
            if best_k:
                f.write(f'Best k: {best_k}\n')
                f.write(f'Best accuracy: {best_acc*100:.2f}%\n')
                
                # Parameter counts
                if 'parameter_counts' in data[dataset][best_k]:
                    params = data[dataset][best_k]['parameter_counts']
                    f.write(f'\nParameter counts at k={best_k}:\n')
                    for method, count in sorted(params.items()):
                        if method in data[dataset][best_k]['results']:
                            acc = data[dataset][best_k]['results'][method]['test_acc_mean'] * 100
                            f.write(f'  {method:30}: {count:>10,} params → {acc:>6.2f}%\n')
            
            # 4. JL compression analysis
            if best_k and best_k in data[dataset]:
                results = data[dataset][best_k]['results']
                
                f.write(f'\n4. JL COMPRESSION COMPARISON (k={best_k})\n')
                f.write('-'*80 + '\n')
                
                for proj_type in ['binary', 'ortho']:
                    f.write(f'\n{proj_type.capitalize()} projections:\n')
                    for ratio in ['nc', '2nc', '4nc']:
                        key = f'jl_{proj_type}_{ratio}'
                        if key in results:
                            acc = results[key]['test_acc_mean'] * 100
                            f.write(f'  {ratio:4}: {acc:6.2f}%\n')
            
            f.write('\n' + '='*80 + '\n')
        
        print(f'✓ Saved: {report_path}')

# ============================================================================
# PHASE 16: COMBINED ANALYSIS (FIXED + RANDOM)
# ============================================================================

def generate_combined_analysis(experiments, output_dir):
    """
    Compare results between fixed and random splits for datasets with both.
    """
    print('\n' + '='*80)
    print('COMBINED ANALYSIS (FIXED VS RANDOM SPLITS)')
    print('='*80)
    
    combined_dir = os.path.join(output_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    # Find datasets with both split types
    fixed_datasets = set(experiments['fixed'].keys())
    random_datasets = set(experiments['random'].keys())
    both = sorted(fixed_datasets & random_datasets)
    
    if not both:
        print('⚠ No datasets with both split types')
        return
    
    print(f'\nDatasets with both splits: {len(both)}')
    
    # Compare best results
    comparison_data = []
    
    for dataset in both:
        # Fixed splits
        fixed_best_sgc = -1
        fixed_best_full = -1
        
        for k in experiments['fixed'][dataset]:
            results = experiments['fixed'][dataset][k].get('results', {})
            sgc_acc = results.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
            full_acc = results.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            
            if not np.isnan(sgc_acc) and sgc_acc > fixed_best_sgc:
                fixed_best_sgc = sgc_acc
            if not np.isnan(full_acc) and full_acc > fixed_best_full:
                fixed_best_full = full_acc
        
        # Random splits
        random_best_sgc = -1
        random_best_full = -1
        
        for k in experiments['random'][dataset]:
            results = experiments['random'][dataset][k].get('results', {})
            sgc_acc = results.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
            full_acc = results.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            
            if not np.isnan(sgc_acc) and sgc_acc > random_best_sgc:
                random_best_sgc = sgc_acc
            if not np.isnan(full_acc) and full_acc > random_best_full:
                random_best_full = full_acc
        
        comparison_data.append({
            'Dataset': dataset,
            'Fixed SGC': safe_percentage(fixed_best_sgc),
            'Fixed Full': safe_percentage(fixed_best_full),
            'Fixed Imp': safe_improvement(fixed_best_full, fixed_best_sgc),
            'Random SGC': safe_percentage(random_best_sgc),
            'Random Full': safe_percentage(random_best_full),
            'Random Imp': safe_improvement(random_best_full, random_best_sgc)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save
    csv_path = os.path.join(combined_dir, 'fixed_vs_random_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n✓ Saved: {csv_path}')
    
    # Print
    print('\n' + df.to_string(index=False))

# ============================================================================
# PHASE 17: SUMMARY STATISTICS
# ============================================================================

def generate_summary_statistics(data, split_type, output_dir):
    """
    Generate comprehensive summary statistics.
    """
    print('\n' + '='*80)
    print(f'SUMMARY STATISTICS - {split_type.upper()}')
    print('='*80)
    
    total_datasets = len(data)
    
    # Success rate (improvement > 0 at max k)
    successes = 0
    for dataset in data:
        k_values = list(data[dataset].keys())
        if k_values:
            k_max = max(k_values)
            results = data[dataset][k_max].get('results', {})
            
            sgc_acc = results.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
            full_acc = results.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            
            if not np.isnan(sgc_acc) and not np.isnan(full_acc) and full_acc > sgc_acc:
                successes += 1
    
    # Anti-smoothing count
    anti_smoothing_count = 0
    for dataset in data:
        if 2 in data[dataset] and 10 in data[dataset]:
            results_k2 = data[dataset][2].get('results', {})
            results_k10 = data[dataset][10].get('results', {})
            
            sgc_k2 = results_k2.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
            sgc_k10 = results_k10.get('sgc_baseline', {}).get('test_acc_mean', np.nan)
            full_k2 = results_k2.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            full_k10 = results_k10.get('full_rownorm_mlp', {}).get('test_acc_mean', np.nan)
            
            if (not np.isnan(sgc_k2) and not np.isnan(sgc_k10) and 
                not np.isnan(full_k2) and not np.isnan(full_k10)):
                if sgc_k10 < sgc_k2 and full_k10 > full_k2:
                    anti_smoothing_count += 1
    
    print(f'Total datasets: {total_datasets}')
    print(f'Success rate (improvement > 0 at max k): {successes}/{total_datasets} ({successes/total_datasets*100:.1f}%)')
    print(f'Anti-smoothing effect observed: {anti_smoothing_count} datasets')
    
    # Save to file
    stats_path = os.path.join(output_dir, 'summary_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write(f'SUMMARY STATISTICS - {split_type.upper()}\n')
        f.write('='*80 + '\n\n')
        f.write(f'Total datasets: {total_datasets}\n')
        f.write(f'Success rate: {successes}/{total_datasets} ({successes/total_datasets*100:.1f}%)\n')
        f.write(f'Anti-smoothing effect: {anti_smoothing_count} datasets\n')
    
    print(f'✓ Saved: {stats_path}')

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis():
    """
    Run complete analysis pipeline for all split types.
    """
    print('\n')
    print('='*80)
    print('COMPLETE SGC GLOBAL ANTI-SMOOTHING ANALYSIS')
    print('='*80)
    print(f'Results directory: {RESULTS_BASE}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('='*80)
    
    # Phase 1: Auto-discovery
    experiments = discover_experiments(RESULTS_BASE)
    
    if not experiments['fixed'] and not experiments['random']:
        print('\n❌ ERROR: No experiments found!')
        print(f'Check that results exist in: {RESULTS_BASE}')
        return
    
    # Phase 2-17: Run analysis for each split type
    for split_type in ['fixed', 'random']:
        if not experiments[split_type]:
            continue
        
        print('\n' + '='*80)
        print(f'ANALYZING {split_type.upper()} SPLITS')
        print('='*80)
        
        split_dir = os.path.join(OUTPUT_DIR, split_type)
        os.makedirs(split_dir, exist_ok=True)
        
        data = experiments[split_type]
        
        # Tables
        generate_table1_complete(data, split_type, split_dir)
        generate_table1b_jl_compression(data, split_type, split_dir)
        generate_table2_anti_smoothing(data, split_type, split_dir)
        generate_table3_parameter_efficiency(data, split_type, split_dir)
        generate_table4_best_jl_config(data, split_type, split_dir)
        generate_table5_binary_vs_ortho(data, split_type, split_dir)
        generate_table6_failure_analysis(data, split_type, split_dir)
        
        # Figures
        generate_figure1_k_trajectories(data, split_type, split_dir)
        generate_figure2_improvement_heatmap(data, split_type, split_dir)
        generate_figure3_jl_compression(data, split_type, split_dir)
        generate_figure4_parameter_efficiency(data, split_type, split_dir)
        generate_figure5_anti_smoothing_effect(data, split_type, split_dir)
        generate_figure6_method_comparison(data, split_type, split_dir)
        
        # Per-dataset reports
        generate_per_dataset_reports(data, split_type, split_dir)
        
        # Summary statistics
        generate_summary_statistics(data, split_type, split_dir)
    
    # Combined analysis
    if experiments['fixed'] and experiments['random']:
        generate_combined_analysis(experiments, OUTPUT_DIR)
    
    # Final summary
    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print(f'\nAll outputs saved to: {OUTPUT_DIR}/')
    print('\nGenerated files:')
    print('  Tables (1-6):')
    print('    - table1_best_results_complete.csv/.tex')
    print('    - table1b_jl_binary_compression.csv/.tex')
    print('    - table1b_jl_ortho_compression.csv/.tex')
    print('    - table2_anti_smoothing.csv/.tex')
    print('    - table3_parameter_efficiency_complete.csv/.tex')
    print('    - table4_best_jl_config.csv/.tex')
    print('    - table5_binary_vs_ortho.csv/.tex')
    print('    - table6_failure_analysis.csv/.tex')
    print('  Figures (1-6):')
    print('    - figure1_k_trajectories.png')
    print('    - figure2_improvement_heatmap.png')
    print('    - figure3_jl_compression_comparison.png')
    print('    - figure4_parameter_efficiency.png')
    print('    - figure5_anti_smoothing_effect.png')
    print('    - figure6_method_comparison.png')
    print('  Reports:')
    print('    - detailed_reports/[dataset]_detailed.txt (per dataset)')
    print('    - summary_statistics.txt')
    print('    - combined/fixed_vs_random_comparison.csv')
    print('='*80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    run_complete_analysis()
