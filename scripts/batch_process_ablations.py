"""
Batch Processor for Extended Ablation Analysis
===============================================

Analyzes all available datasets and generates comprehensive reports.

Usage:
    python batch_process_ablations.py [--split-type fixed|random]
"""

import os
import sys
import subprocess
from pathlib import Path
import json

DATASETS = [
    'ogbn-arxiv', 'cora', 'citeseer', 'pubmed',
    'wikics', 'amazon-photo', 'amazon-computers',
    'coauthor-cs', 'coauthor-physics'
]

def check_results_exist(dataset, split_type='fixed_splits'):
    """Check if results exist for a dataset"""
    path = f'results/investigation2_extended_ablations/{dataset}/{split_type}/metrics/results_aggregated.json'
    return os.path.exists(path)

def generate_individual_reports(split_type='fixed_splits'):
    """Generate individual reports for each dataset"""
    print("="*70)
    print(f"GENERATING INDIVIDUAL REPORTS ({split_type})")
    print("="*70)
    
    reports_dir = f'results/extended_ablation_reports/{split_type}'
    os.makedirs(reports_dir, exist_ok=True)
    
    available_datasets = []
    for dataset in DATASETS:
        if check_results_exist(dataset, split_type):
            available_datasets.append(dataset)
            print(f"\n[{len(available_datasets)}/{len(DATASETS)}] Processing {dataset}...")
            
            # Generate text report
            output_file = f'{reports_dir}/{dataset}_report.txt'
            cmd = f'python analyze_extended_ablations.py {dataset} --split-type {split_type.replace("_splits", "")} --output {output_file}'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"  ✓ Report saved: {output_file}")
            
            # Generate figure
            print(f"  → Generating figure...")
            cmd = f'python visualize_ablations.py {dataset} --split-type {split_type.replace("_splits", "")}'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"  ✓ Figure generated")
        else:
            print(f"⊘ Skipping {dataset} (no results found)")
    
    print(f"\n✓ Processed {len(available_datasets)}/{len(DATASETS)} datasets")
    return available_datasets

def generate_cross_dataset_report(split_type='fixed_splits'):
    """Generate cross-dataset comparison report"""
    print("\n" + "="*70)
    print(f"GENERATING CROSS-DATASET COMPARISON ({split_type})")
    print("="*70)
    
    # Generate cross-dataset figure
    print("\n→ Creating cross-dataset comparison figure...")
    output_fig = f'results/extended_ablation_reports/{split_type}/cross_dataset_comparison.png'
    cmd = f'python visualize_ablations.py --cross-dataset --split-type {split_type.replace("_splits", "")} --output {output_fig}'
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    print(f"✓ Figure saved: {output_fig}")
    
    # Generate cross-dataset text analysis
    print("\n→ Running cross-dataset analysis...")
    cmd = f'python analyze_extended_ablations.py --all --split-type {split_type.replace("_splits", "")}'
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    
    output_file = f'results/extended_ablation_reports/{split_type}/cross_dataset_analysis.txt'
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    print(f"✓ Analysis saved: {output_file}")

def create_summary_document(split_type='fixed_splits'):
    """Create a master summary document"""
    print("\n" + "="*70)
    print(f"CREATING MASTER SUMMARY ({split_type})")
    print("="*70)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("EXTENDED ABLATION STUDY - MASTER SUMMARY")
    summary_lines.append(f"Split Type: {split_type}")
    summary_lines.append("="*80)
    summary_lines.append("")
    
    summary_lines.append("EXPERIMENTS CONDUCTED:")
    summary_lines.append("-"*80)
    summary_lines.append("(a)   X → StandardScaler → Standard MLP         [Baseline]")
    summary_lines.append("(b)   U → StandardScaler → Standard MLP         [Direction B: Basis change]")
    summary_lines.append("(b')  U → No scaling → Standard MLP             [Ablation 1: Scaling effect]")
    summary_lines.append("(b'') U_weighted → StandardScaler → Standard MLP [Ablation 2: Eigenvalue effect]")
    summary_lines.append("(c)   U → RowNorm MLP                           [Direction A: Model change]")
    summary_lines.append("")
    
    summary_lines.append("KEY COMPARISONS:")
    summary_lines.append("-"*80)
    summary_lines.append("• Direction B (a→b): Pure basis sensitivity (same model, same preprocessing)")
    summary_lines.append("• Ablation 1 (b→b'): Does StandardScaler destroy U's geometry?")
    summary_lines.append("• Ablation 2 (b→b''): Do restricted eigenvalues carry information?")
    summary_lines.append("• Direction A (b→c): Does RowNorm work on restricted eigenvectors?")
    summary_lines.append("")
    
    # Load and summarize all available results
    summary_lines.append("RESULTS BY DATASET:")
    summary_lines.append("-"*80)
    
    for dataset in DATASETS:
        if not check_results_exist(dataset, split_type):
            continue
        
        path = f'results/investigation2_extended_ablations/{dataset}/{split_type}/metrics/results_aggregated.json'
        with open(path, 'r') as f:
            results = json.load(f)
        
        agg = results['aggregated_results']
        a = agg['standard_X_scaled']['test_acc']['mean'] * 100
        b = agg['standard_V_scaled']['test_acc']['mean'] * 100
        bp = agg['standard_V_unscaled']['test_acc']['mean'] * 100
        bpp = agg['standard_V_weighted']['test_acc']['mean'] * 100
        c = agg['rownorm_V']['test_acc']['mean'] * 100
        
        summary_lines.append(f"\n{dataset.upper()}:")
        summary_lines.append(f"  Baseline (a):        {a:.2f}%")
        summary_lines.append(f"  Direction B (a→b):   {(b-a):+.2f} pp  ({(b-a)/a*100:+.1f}%)")
        summary_lines.append(f"  Scaling (b→b'):      {(bp-b):+.2f} pp  ({(bp-b)/b*100:+.1f}%)")
        summary_lines.append(f"  Eigenval (b→b''):    {(bpp-b):+.2f} pp  ({(bpp-b)/b*100:+.1f}%)")
        summary_lines.append(f"  Direction A (b→c):   {(c-b):+.2f} pp  ({(c-b)/b*100:+.1f}%)")
        
        # Reliability
        train_samples = results['train_samples']
        params = results['parameters_per_model']
        ratio = params / train_samples
        if ratio > 100:
            summary_lines.append(f"  ⚠️  UNRELIABLE (param/sample: {ratio:.0f}:1)")
        elif ratio > 10:
            summary_lines.append(f"  ⚠️  Moderate reliability (param/sample: {ratio:.1f}:1)")
        else:
            summary_lines.append(f"  ✓ High reliability (param/sample: {ratio:.1f}:1)")
    
    summary_lines.append("")
    summary_lines.append("="*80)
    summary_lines.append("END OF SUMMARY")
    summary_lines.append("="*80)
    
    # Save summary
    output_file = f'results/extended_ablation_reports/{split_type}/MASTER_SUMMARY.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Master summary saved: {output_file}")
    
    # Also print to console
    print("\n" + '\n'.join(summary_lines))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process extended ablation results')
    parser.add_argument('--split-type', choices=['fixed', 'random'], default='fixed')
    
    args = parser.parse_args()
    
    split_type = 'fixed_splits' if args.split_type == 'fixed' else 'random_splits'
    
    print("\n" + "="*70)
    print("EXTENDED ABLATION BATCH PROCESSOR")
    print("="*70)
    print(f"Split type: {split_type}")
    print("="*70)
    
    # Step 1: Generate individual reports
    available_datasets = generate_individual_reports(split_type)
    
    if len(available_datasets) == 0:
        print("\n❌ No results found! Please run experiments first.")
        return
    
    # Step 2: Generate cross-dataset comparison
    if len(available_datasets) > 1:
        generate_cross_dataset_report(split_type)
    else:
        print("\n⊘ Skipping cross-dataset comparison (only 1 dataset available)")
    
    # Step 3: Create master summary
    create_summary_document(split_type)
    
    print("\n" + "="*70)
    print("✓ BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: results/extended_ablation_reports/{split_type}/")
    print("\nGenerated files:")
    print(f"  • Individual reports: {len(available_datasets)} files")
    print(f"  • Individual figures: {len(available_datasets)} files")
    if len(available_datasets) > 1:
        print("  • Cross-dataset comparison figure")
        print("  • Cross-dataset analysis report")
    print("  • Master summary document")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
