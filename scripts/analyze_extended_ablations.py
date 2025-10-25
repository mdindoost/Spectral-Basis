"""
Extended Ablation Results Analyzer
===================================

Analyzes results from investigation2_extended_ablations_v2.py
Generates comprehensive reports, tables, and interpretations.

Usage:
    python analyze_extended_ablations.py [dataset]

Examples:
    python analyze_extended_ablations.py ogbn-arxiv
    python analyze_extended_ablations.py  # defaults to ogbn-arxiv if available, else first existing
    python analyze_extended_ablations.py --all  # Analyze all datasets that exist
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    'ogbn-arxiv', 'cora', 'citeseer', 'pubmed',
    'wikics', 'amazon-photo', 'amazon-computers',
    'coauthor-cs', 'coauthor-physics'
]

# All experiments are expected under random splits only.
SPLIT_DIR = 'random_splits'

# Required experiment keys (must match your JSON "models" keys)
REQUIRED_EXPS = [
    'standard_X_scaled',
    'standard_V_scaled',
    'standard_V_unscaled',
    'standard_V_weighted',
    'rownorm_V',
]

EXPERIMENT_NAMES = {
    'standard_X_scaled': '(a) X + Scaled',
    'standard_V_scaled': '(b) U + Scaled',
    'standard_V_unscaled': "(b') U + Unscaled",
    'standard_V_weighted': "(b'') U + Weighted",
    'rownorm_V': '(c) U + RowNorm'
}

COMPARISON_NAMES = {
    'direction_B': 'Direction B: Basis Change (a→b)',
    'ablation_scaling': "Ablation 1: Remove Scaling (b→b')",
    'ablation_eigenval': "Ablation 2: Add Eigenvalues (b→b'')",
    'direction_A': 'Direction A: Model Change (b→c)'
}

# ============================================================================
# Helper Functions
# ============================================================================

def results_path(dataset: str) -> str:
    """Return the expected aggregated metrics path for a dataset."""
    return f'results/investigation2_extended_ablations/{dataset}/{SPLIT_DIR}/metrics/results_aggregated.json'

def load_results(dataset):
    """Load aggregated results for a dataset (random splits only)."""
    metrics_file = results_path(dataset)
    if not os.path.exists(metrics_file):
        return None
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    return data

def format_pct(value, decimals=1):
    """Format percentage with sign"""
    return f"{value:+.{decimals}f}%"

def format_pp(value, decimals=2):
    """Format percentage points with sign"""
    return f"{value:+.{decimals}f} pp"

def fmt_mean_std(mean, std):
    """Format 'mean ± std' in %, robust if std is None."""
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean*100:.2f}"
    return f"{mean*100:.2f} ± {std*100:.2f}"

# ============================================================================
# Analysis Functions
# ============================================================================

class ExtendedAblationAnalyzer:
    """
    Supports two JSON schemas:
    NEW (your file):
      {
        "dataset": "...",
        "split_type": "random_splits",
        "models": {
          "<exp_key>": {
            "test_accuracy": {"mean": float, "std": float, ...},
            ...
          },
          ...
        },
        "pca_analysis": { "X": {...}, "U": {...}, "comparison": {...} }
      }

    LEGACY (fallback support):
      {
        "aggregated_results": {
          "<exp_key>": { "test_acc": {"mean": float, "std": float} }
        },
        "pca_results": { "X": {...}, "U": {...}, "comparison": {...} }
      }
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.split_type = SPLIT_DIR  # kept for report text
        raw = load_results(dataset)
        if raw is None:
            raise FileNotFoundError(f"No results file at {results_path(dataset)}")

        self.schema = self._detect_schema(raw)
        if self.schema == 'new':
            self._init_from_new_schema(raw)
        elif self.schema == 'legacy':
            self._init_from_legacy_schema(raw)
        else:
            raise KeyError("Unrecognized results schema. Expected keys 'models' or 'aggregated_results'.")

        # Validate required experiments exist
        self.missing_exps = [k for k in REQUIRED_EXPS if k not in self._exp_acc]
        if self.missing_exps:
            print(f"Warning: {self.dataset} missing experiment(s): {', '.join(self.missing_exps)}")

    def _detect_schema(self, raw):
        if 'models' in raw and 'pca_analysis' in raw:
            return 'new'
        if 'aggregated_results' in raw and 'pca_results' in raw:
            return 'legacy'
        if 'models' in raw:
            return 'new'
        if 'aggregated_results' in raw:
            return 'legacy'
        return None

    def _init_from_new_schema(self, raw):
        # Map experiments to (mean, std)
        exp_acc = {}
        models = raw.get('models', {})
        for k, v in models.items():
            ta = v.get('test_accuracy', {})
            mean = ta.get('mean', None)
            std = ta.get('std', None)
            if mean is not None:
                exp_acc[k] = (mean, std)
        self._exp_acc = exp_acc

        # PCA summary
        self.pca = raw.get('pca_analysis', {})

    def _init_from_legacy_schema(self, raw):
        exp_acc = {}
        agg = raw.get('aggregated_results', {})
        for k, v in agg.items():
            ta = v.get('test_acc', {})
            mean = ta.get('mean', None)
            std = ta.get('std', None)
            if mean is not None:
                exp_acc[k] = (mean, std)
        self._exp_acc = exp_acc

        self.pca = raw.get('pca_results', {})

    def get_test_acc(self, exp_key):
        """Get test accuracy mean and std for an experiment; raise if missing."""
        if exp_key not in self._exp_acc:
            raise KeyError(f"Experiment '{exp_key}' not found for dataset '{self.dataset}'.")
        return self._exp_acc[exp_key]
    
    def _safe_get_acc(self, exp_key):
        """Return (mean,std) or (None,None) without throwing."""
        return self._exp_acc.get(exp_key, (None, None))

    def compute_comparisons(self):
        """Compute all key comparisons. Skips comparisons if needed baselines are missing."""
        required_for_all = [
            'standard_X_scaled', 'standard_V_scaled',
            'standard_V_unscaled', 'standard_V_weighted', 'rownorm_V'
        ]
        for k in required_for_all:
            if k not in self._exp_acc:
                raise KeyError(f"Cannot compute comparisons: missing '{k}' in dataset '{self.dataset}'.")

        a_mean, a_std = self.get_test_acc('standard_X_scaled')
        b_mean, b_std = self.get_test_acc('standard_V_scaled')
        bp_mean, bp_std = self.get_test_acc('standard_V_unscaled')
        bpp_mean, bpp_std = self.get_test_acc('standard_V_weighted')
        c_mean, c_std = self.get_test_acc('rownorm_V')
        
        comparisons = {}
        
        # Direction B: a vs b (basis change)
        diff = b_mean - a_mean
        pct = (diff / a_mean * 100) if a_mean and a_mean > 0 else 0.0
        comparisons['direction_B'] = {
            'diff_pp': diff,
            'diff_pct': pct,
            'baseline': a_mean,
            'comparison': b_mean,
            'baseline_std': a_std,
            'comparison_std': b_std
        }
        
        # Ablation 1: b vs b' (scaling effect)
        diff = bp_mean - b_mean
        pct = (diff / b_mean * 100) if b_mean and b_mean > 0 else 0.0
        comparisons['ablation_scaling'] = {
            'diff_pp': diff,
            'diff_pct': pct,
            'baseline': b_mean,
            'comparison': bp_mean,
            'baseline_std': b_std,
            'comparison_std': bp_std
        }
        
        # Ablation 2: b vs b'' (eigenvalue effect)
        diff = bpp_mean - b_mean
        pct = (diff / b_mean * 100) if b_mean and b_mean > 0 else 0.0
        comparisons['ablation_eigenval'] = {
            'diff_pp': diff,
            'diff_pct': pct,
            'baseline': b_mean,
            'comparison': bpp_mean,
            'baseline_std': b_std,
            'comparison_std': bpp_std
        }
        
        # Direction A: b vs c (model change)
        diff = c_mean - b_mean
        pct = (diff / b_mean * 100) if b_mean and b_mean > 0 else 0.0
        comparisons['direction_A'] = {
            'diff_pp': diff,
            'diff_pct': pct,
            'baseline': b_mean,
            'comparison': c_mean,
            'baseline_std': b_std,
            'comparison_std': c_std   # <-- FIXED (it was missing previously)
        }
        
        return comparisons
    
    def interpret_results(self):
        """Generate interpretation of results. If required comps missing, return message."""
        try:
            comps = self.compute_comparisons()
        except KeyError as e:
            return f"Interpretation unavailable: {e}"
        interpretations = []
        
        # Direction B
        db = comps['direction_B']
        if db['diff_pct'] < -1:
            interpretations.append(f"✗ Basis change hurts performance ({format_pct(db['diff_pct'])})")
            interpretations.append("  → Despite span(X) = span(U), basis matters!")
        elif db['diff_pct'] > 1:
            interpretations.append(f"✓ Basis change helps ({format_pct(db['diff_pct'])})")
            interpretations.append("  → Restricted eigenvectors are better coordinate system")
        else:
            interpretations.append("○ Minimal basis effect")
        
        # Ablation 1: Scaling
        a1 = comps['ablation_scaling']
        if a1['diff_pct'] > 1:
            interpretations.append(f"\n✓ Removing StandardScaler HELPS ({format_pct(a1['diff_pct'])})")
            interpretations.append("  → StandardScaler destroys D-orthonormality")
            interpretations.append("  → U's geometry matters for performance")
        elif a1['diff_pct'] < -1:
            interpretations.append(f"\n✗ StandardScaler is HELPING ({format_pct(abs(a1['diff_pct']))})")
            interpretations.append("  → Zero-mean/unit-variance is more important than D-orthonormality")
        else:
            interpretations.append("\n○ StandardScaler has minimal effect")
        
        # Ablation 2: Eigenvalues
        a2 = comps['ablation_eigenval']
        if a2['diff_pct'] > 1:
            interpretations.append(f"\n✓ Eigenvalue weighting HELPS ({format_pct(a2['diff_pct'])})")
            interpretations.append("  → Restricted eigenvalues carry learnable structure")
        elif a2['diff_pct'] < -1:
            interpretations.append(f"\n✗ Eigenvalue weighting HURTS ({format_pct(abs(a2['diff_pct']))})")
            interpretations.append("  → Weighting amplifies noise in low eigenvalues")
        else:
            interpretations.append("\n○ Eigenvalues have minimal effect")
            interpretations.append("  → Information is in basis, not scaling")
        
        # Direction A
        da = comps['direction_A']
        if da['diff_pct'] > 1:
            interpretations.append(f"\n✓ RowNorm advantage persists ({format_pct(da['diff_pct'])})")
        elif da['diff_pct'] < -1:
            interpretations.append(f"\n✗ RowNorm FAILS on restricted eigenvectors ({format_pct(da['diff_pct'])})")
            interpretations.append("  → RowNorm requires true spectral structure")
        else:
            interpretations.append("\n○ Minimal model difference")
        
        return "\n".join(interpretations)
    
    def generate_summary_table(self):
        """Generate summary table of all experiments that exist"""
        rows = []
        for exp_key, exp_name in EXPERIMENT_NAMES.items():
            mean_std = self._exp_acc.get(exp_key, None)
            if not mean_std:
                rows.append({
                    'Experiment': exp_name,
                    'Test Acc (%)': 'N/A',
                    'Mean': np.nan,
                    'Std': np.nan
                })
                continue
            mean, std = mean_std
            rows.append({
                'Experiment': exp_name,
                'Test Acc (%)': fmt_mean_std(mean, std),
                'Mean': (mean * 100) if mean is not None else np.nan,
                'Std': (std * 100) if std is not None else np.nan
            })
        df = pd.DataFrame(rows)
        return df
    
    def generate_comparison_table(self):
        """Generate table of key comparisons (where possible)"""
        try:
            comps = self.compute_comparisons()
        except KeyError as e:
            return pd.DataFrame([{'Comparison': 'N/A', 'Effect (pp)': 'N/A', 'Effect (%)': 'N/A',
                                  'Baseline': 'N/A', 'After': f'Unavailable: {e}'}])
        rows = []
        for comp_key, comp_name in COMPARISON_NAMES.items():
            comp = comps[comp_key]
            baseline_str = fmt_mean_std(comp.get('baseline'), comp.get('baseline_std'))
            after_str    = fmt_mean_std(comp.get('comparison'), comp.get('comparison_std'))
            rows.append({
                'Comparison': comp_name,
                'Effect (pp)': format_pp(comp['diff_pp'] * 100),
                'Effect (%)': format_pct(comp['diff_pct']),
                'Baseline': baseline_str,
                'After': after_str
            })
        df = pd.DataFrame(rows)
        return df
    
    def generate_pca_summary(self):
        """Summarize PCA geometric analysis, if present"""
        if not self.pca:
            return "PCA GEOMETRIC ANALYSIS\n" + "=" * 60 + "\nNo PCA data available."
        X = self.pca.get('X', {})
        U = self.pca.get('U', {})
        comp = self.pca.get('comparison', {})
        
        summary = []
        summary.append("PCA GEOMETRIC ANALYSIS")
        summary.append("=" * 60)
        if X and U:
            if 'effective_rank' in X and 'effective_rank' in U:
                summary.append(f"Effective Rank:")
                xr = X.get('effective_rank', float('nan'))
                ur = U.get('effective_rank', float('nan'))
                xrr = X.get('effective_rank_ratio', float('nan'))
                urr = U.get('effective_rank_ratio', float('nan'))
                cr = comp.get('effective_rank_ratio', float('nan'))
                summary.append(f"  X: {xr:.1f} ({xrr*100:.1f}%)")
                summary.append(f"  U: {ur:.1f} ({urr*100:.1f}%)")
                summary.append(f"  Ratio (U/X): {cr:.3f}")
            if 'condition_number' in X and 'condition_number' in U:
                summary.append(f"\nCondition Number:")
                xc = X.get('condition_number', float('nan'))
                uc = U.get('condition_number', float('nan'))
                cnr = comp.get('condition_number_ratio', float('nan'))
                summary.append(f"  X: {xc:.2e}")
                summary.append(f"  U: {uc:.2e}")
                summary.append(f"  Ratio (U/X): {cnr:.3f}")
                if isinstance(cnr, (int, float)):
                    if cnr < 0.5:
                        summary.append("  ✓ U is BETTER conditioned (paradox if performs worse!)")
                    elif cnr > 2.0:
                        summary.append("  ✗ U is WORSE conditioned (may explain degradation)")
            if 'coefficient_variation' in X and 'coefficient_variation' in U:
                summary.append(f"\nIsotropy (Coefficient of Variation):")
                xv = X.get('coefficient_variation', float('nan'))
                uv = U.get('coefficient_variation', float('nan'))
                iso = comp.get('isotropy_ratio', float('nan'))
                summary.append(f"  X: {xv:.3f}")
                summary.append(f"  U: {uv:.3f}")
                summary.append(f"  Ratio (U/X): {iso:.3f}")
                if isinstance(iso, (int, float)):
                    if iso < 0.7:
                        summary.append("  ✓ U is MORE isotropic (more uniform variance)")
                    elif iso > 1.4:
                        summary.append("  ✗ U is LESS isotropic (more skewed variance)")
        else:
            summary.append("No PCA sections 'X' and 'U' found.")
        return "\n".join(summary)
    
    def generate_full_report(self):
        """Generate comprehensive text report"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"EXTENDED ABLATION ANALYSIS: {self.dataset.upper()}")
        lines.append(f"Split Type: {self.split_type}")
        lines.append("=" * 70)
        
        # Summary table
        lines.append("\nEXPERIMENT RESULTS:")
        lines.append("-" * 70)
        df_summary = self.generate_summary_table()
        lines.append(df_summary.to_string(index=False))
        
        # Comparison table
        lines.append("\n\nKEY COMPARISONS:")
        lines.append("-" * 70)
        df_comp = self.generate_comparison_table()
        lines.append(df_comp.to_string(index=False))
        
        # Interpretations
        lines.append("\n\nINTERPRETATION:")
        lines.append("-" * 70)
        lines.append(self.interpret_results())
        
        # PCA summary
        lines.append("\n\n" + self.generate_pca_summary())
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

# ============================================================================
# Cross-Dataset Analysis
# ============================================================================

class CrossDatasetAnalyzer:
    def __init__(self):
        self.split_type = SPLIT_DIR
        self.analyzers = {}
        for dataset in DATASETS:
            raw = load_results(dataset)
            if raw is None:
                print(f"Warning: No results for {dataset} ({SPLIT_DIR})")
                continue
            try:
                analyzer = ExtendedAblationAnalyzer(dataset)
            except Exception as e:
                print(f"Warning: Skipping {dataset}: {e}")
                continue
            if not analyzer._exp_acc:
                print(f"Warning: Skipping {dataset}: no experiment accuracies found.")
                continue
            self.analyzers[dataset] = analyzer
    
    def generate_cross_dataset_table(self):
        """Generate table comparing all datasets that exist"""
        rows = []
        for dataset, analyzer in self.analyzers.items():
            try:
                comps = analyzer.compute_comparisons()
            except KeyError:
                comps = None
            a_mean, _ = analyzer._safe_get_acc('standard_X_scaled')
            row = {
                'Dataset': dataset,
                'Baseline (a)': f'{a_mean*100:.1f}%' if a_mean is not None else 'N/A',
                'Dir B (pp)': 'N/A',
                "Scaling (b→b')": 'N/A',
                "Eigenval (b→b'')": 'N/A',
                'Dir A (pp)': 'N/A',
            }
            if comps is not None:
                row['Dir B (pp)'] = format_pp(comps['direction_B']['diff_pp'] * 100, 1)
                row["Scaling (b→b')"] = format_pp(comps['ablation_scaling']['diff_pp'] * 100, 1)
                row["Eigenval (b→b'')"] = format_pp(comps['ablation_eigenval']['diff_pp'] * 100, 1)
                row['Dir A (pp)'] = format_pp(comps['direction_A']['diff_pp'] * 100, 1)
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=['Dataset','Baseline (a)','Dir B (pp)',"Scaling (b→b')","Eigenval (b→b'')",'Dir A (pp)'])
        df = pd.DataFrame(rows)
        return df
    
    def identify_patterns(self):
        """Identify cross-dataset patterns"""
        if not self.analyzers:
            return "No datasets with results found. Nothing to analyze."
        
        dir_b_effects, scaling_effects, eigenval_effects, dir_a_effects = [], [], [], []
        for analyzer in self.analyzers.values():
            try:
                comps = analyzer.compute_comparisons()
            except KeyError:
                continue
            dir_b_effects.append(comps['direction_B']['diff_pp'])
            scaling_effects.append(comps['ablation_scaling']['diff_pp'])
            eigenval_effects.append(comps['ablation_eigenval']['diff_pp'])
            dir_a_effects.append(comps['direction_A']['diff_pp'])
        
        if not dir_b_effects:
            return "Insufficient comparable datasets to identify patterns."
        
        msg = []
        msg.append("CROSS-DATASET PATTERNS:")
        msg.append("=" * 60)
        msg.append(f"Average Direction B effect: {np.mean(dir_b_effects)*100:.2f} pp")
        msg.append(f"Average Scaling effect: {np.mean(scaling_effects)*100:.2f} pp")
        msg.append(f"Average Eigenvalue effect: {np.mean(eigenval_effects)*100:.2f} pp")
        msg.append(f"Average Direction A effect: {np.mean(dir_a_effects)*100:.2f} pp")
        scaling_helps = sum(1 for e in scaling_effects if e > 0.01)
        scaling_hurts = sum(1 for e in scaling_effects if e < -0.01)
        eigenval_helps = sum(1 for e in eigenval_effects if e > 0.01)
        msg.append(f"\nScaling effect: Helps on {scaling_helps}/{len(scaling_effects)} datasets")
        msg.append(f"Scaling effect: Hurts on {scaling_hurts}/{len(scaling_effects)} datasets")
        msg.append(f"Eigenvalue effect: Helps on {eigenval_helps}/{len(eigenval_effects)} datasets")
        return "\n".join(msg)

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze extended ablation results (random splits only)')
    parser.add_argument('dataset', nargs='?', default='ogbn-arxiv', help='Dataset name')
    parser.add_argument('--all', action='store_true', help='Analyze all datasets that exist')
    parser.add_argument('--output', default=None, help='Output file for report (single-dataset mode)')
    
    args = parser.parse_args()
    
    if args.all:
        print("Analyzing all datasets that exist (random splits)...")
        cross = CrossDatasetAnalyzer()
        if not cross.analyzers:
            print("No datasets with results found under random splits.")
            sys.exit(0)
        
        print("\n" + "="*70)
        print("CROSS-DATASET COMPARISON")
        print("="*70)
        df = cross.generate_cross_dataset_table()
        print(df.to_string(index=False))
        
        print("\n" + cross.identify_patterns())
        return
    
    # Single dataset mode: prefer requested dataset; if missing, warn and continue with the first existing one.
    requested = args.dataset
    data = load_results(requested)
    chosen_dataset = requested
    
    if data is None:
        print(f"Warning: No results for requested dataset '{requested}' ({SPLIT_DIR}). Searching for an existing dataset...")
        fallback = None
        for ds in DATASETS:
            if load_results(ds) is not None:
                fallback = ds
                break
        if fallback is None:
            print("No datasets with results found under random splits. Exiting.")
            sys.exit(0)
        print(f"Continuing with existing dataset: '{fallback}'.")
        chosen_dataset = fallback
    
    try:
        analyzer = ExtendedAblationAnalyzer(chosen_dataset)
    except Exception as e:
        print(f"Failed to analyze dataset '{chosen_dataset}': {e}")
        sys.exit(1)

    report = analyzer.generate_full_report()
    print(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {args.output}")

if __name__ == '__main__':
    main()
