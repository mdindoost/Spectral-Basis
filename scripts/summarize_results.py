"""
Summarize results across all datasets for Investigation 2 Directions A&B
"""

import os
import json
import numpy as np
from pathlib import Path

def load_results(dataset_name, split_type='fixed_splits'):
    """Load aggregated results for a dataset"""
    path = f'results/investigation2_directions_AB/{dataset_name}/{split_type}/metrics/results_aggregated.json'
    
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

def print_summary_table():
    """Print comprehensive summary table across all datasets"""
    
    datasets = [
        ('ogbn-arxiv', 'OGB'),
        ('cora', 'Planetoid'),
        ('citeseer', 'Planetoid'),
        ('pubmed', 'Planetoid'),
        ('wikics', 'WikiCS'),
        ('amazon-photo', 'Amazon'),
        ('amazon-computers', 'Amazon'),
        ('coauthor-cs', 'Coauthor'),
        ('coauthor-physics', 'Coauthor')
    ]
    
    results_data = []
    
    for dataset_name, source in datasets:
        results = load_results(dataset_name)
        
        if results is None:
            print(f"⚠️  No results found for {dataset_name}")
            continue
        
        models = results['models']
        a = models['standard_X_scaled']['test_accuracy']
        b = models['standard_V_scaled']['test_accuracy']
        c = models['rownorm_V']['test_accuracy']
        
        # Calculate effects
        basis_effect = (b['mean'] - a['mean']) / a['mean'] * 100 if a['mean'] > 0 else 0
        model_effect = (c['mean'] - b['mean']) / b['mean'] * 100 if b['mean'] > 0 else 0
        
        results_data.append({
            'dataset': dataset_name,
            'source': source,
            'd_raw': results['d_raw'],
            'd_effective': results['d_effective'],
            'rank_deficient': results['rank_deficiency'],
            # 'ortho_error': results['orthonormality_error'],
            'ortho_error': results.get('orthonormality_error', 0.0),
            'acc_a': a['mean'],
            'std_a': a['std'],
            'acc_b': b['mean'],
            'std_b': b['std'],
            'acc_c': c['mean'],
            'std_c': c['std'],
            'basis_effect': basis_effect,
            'model_effect': model_effect
        })
    
    if not results_data:
        print("No results found. Please run experiments first.")
        return
    
    # Print main results table
    print('\n' + '='*140)
    print('INVESTIGATION 2 DIRECTIONS A&B: SUMMARY ACROSS ALL DATASETS')
    print('='*140)
    print(f"{'Dataset':<20} {'Source':<12} {'Dim':<8} {'(a) X Std':<14} {'(b) V Std':<14} {'(c) V Row':<14} {'Basis Δ':<10} {'Model Δ':<10}")
    print('-'*140)
    
    for r in results_data:
        dataset_str = r['dataset']
        if r['rank_deficient']:
            dataset_str += ' ⚠️'
        
        dim_str = f"{r['d_effective']}"
        if r['d_raw'] != r['d_effective']:
            dim_str += f"/{r['d_raw']}"
        
        print(f"{dataset_str:<20} {r['source']:<12} {dim_str:<8} "
              f"{r['acc_a']*100:5.2f}±{r['std_a']*100:4.2f}%  "
              f"{r['acc_b']*100:5.2f}±{r['std_b']*100:4.2f}%  "
              f"{r['acc_c']*100:5.2f}±{r['std_c']*100:4.2f}%  "
              f"{r['basis_effect']:+6.1f}%   "
              f"{r['model_effect']:+6.1f}%")
    
    print('='*140)
    print('\nLegend:')
    print('  (a) X Std   = Standard MLP on raw features X with StandardScaler')
    print('  (b) V Std   = Standard MLP on restricted eigenvectors V with StandardScaler')
    print('  (c) V Row   = RowNorm MLP on restricted eigenvectors V (no scaling)')
    print('  Basis Δ     = (b-a)/a × 100% [Direction B: basis sensitivity]')
    print('  Model Δ     = (c-b)/b × 100% [Direction A: RowNorm effectiveness]')
    print('  ⚠️          = Rank deficiency detected')
    
    # Analyze patterns
    print('\n' + '='*140)
    print('PATTERN ANALYSIS')
    print('='*140)
    
    # Basis sensitivity pattern
    basis_effects = [r['basis_effect'] for r in results_data]
    avg_basis = np.mean(np.abs(basis_effects))
    strong_basis = [r for r in results_data if abs(r['basis_effect']) > 2]
    
    print(f'\nDirection B (Basis Sensitivity):')
    print(f'  Average absolute effect: {avg_basis:.1f}%')
    print(f'  Datasets with strong sensitivity (>2%): {len(strong_basis)}/{len(results_data)}')
    if strong_basis:
        print('  Strong sensitivity in:')
        for r in strong_basis:
            print(f'    • {r["dataset"]}: {r["basis_effect"]:+.1f}%')
    
    # Model effectiveness pattern
    model_effects = [r['model_effect'] for r in results_data]
    avg_model = np.mean(model_effects)
    rownorm_wins = [r for r in results_data if r['model_effect'] > 1]
    
    print(f'\nDirection A (RowNorm Effectiveness):')
    print(f'  Average model effect: {avg_model:+.1f}%')
    print(f'  Datasets where RowNorm helps (>1%): {len(rownorm_wins)}/{len(results_data)}')
    if rownorm_wins:
        print('  RowNorm advantage in:')
        for r in rownorm_wins:
            print(f'    • {r["dataset"]}: {r["model_effect"]:+.1f}%')
    
    # Rank deficiency correlation
    rank_deficient = [r for r in results_data if r['rank_deficient']]
    if rank_deficient:
        print(f'\nRank Deficiency Analysis:')
        print(f'  Datasets with rank deficiency: {len(rank_deficient)}/{len(results_data)}')
        for r in rank_deficient:
            print(f'    • {r["dataset"]}: {r["d_raw"]} → {r["d_effective"]} dimensions')
            print(f'      Orthonormality error: {r["ortho_error"]:.2e}')
    
    # Dataset size correlation
    print(f'\nDataset Size Correlation:')
    print('  (Smaller datasets tend to show more overfitting, less clear patterns)')
    
    print('\n' + '='*140)

def print_convergence_comparison():
    """Print convergence speed comparison (now includes 99%)"""
    
    datasets = ['ogbn-arxiv', 'wikics', 'amazon-photo', 'amazon-computers', 
                'coauthor-cs', 'coauthor-physics']
    
    print('\n' + '='*120)
    print('CONVERGENCE SPEED COMPARISON')
    print('='*120)
    # Added 99% column
    print(f"{'Dataset':<20} {'Model':<25} {'90%':<10} {'95%':<10} {'99%':<10} {'AUC':<10}")
    print('-'*120)
    
    def safe_mean(conv_block, key):
        try:
            return conv_block.get(key, {}).get('mean', None)
        except Exception:
            return None
    
    def fmt_num(x, width=6, prec=1):
        if x is None:
            return f"{'n/a':>{width}}"
        return f"{x:>{width}.{prec}f}"
    
    for dataset_name in datasets:
        results = load_results(dataset_name)
        if results is None:
            continue
        
        models_data = [
            ('Standard MLP on X', 'standard_X_scaled'),
            ('Standard MLP on V', 'standard_V_scaled'),
            ('RowNorm MLP on V', 'rownorm_V')
        ]
        
        for model_name, model_key in models_data:
            conv = results['models'][model_key]['convergence_metrics']
            
            s90 = safe_mean(conv, 'speed_to_90')
            s95 = safe_mean(conv, 'speed_to_95')
            s99 = safe_mean(conv, 'speed_to_99')  # NEW
            auc = safe_mean(conv, 'auc')
            
            print(f"{dataset_name:<20} {model_name:<25} "
                  f"{fmt_num(s90)}    {fmt_num(s95)}    {fmt_num(s99)}    {fmt_num(auc, width=6, prec=3)}")
        
        print('-'*120)
    
    print('='*120)
    print('Note: Lower epochs-to-threshold is better (faster convergence)')
    print('      AUC = Area Under validation Curve (normalized), higher is better')

if __name__ == '__main__':
    print('\n' + '='*140)
    print(' '*40 + 'INVESTIGATION 2: EXTENDED RESULTS SUMMARY')
    print('='*140)
    
    print_summary_table()
    print_convergence_comparison()
    
    print('\n' + '='*140)
    print('For detailed per-dataset results, see:')
    print('  results/investigation2_directions_AB/<dataset>/fixed_splits/metrics/results_aggregated.json')
    print('='*140)
