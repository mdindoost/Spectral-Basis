"""
===================================================================================
MASTER ANALYTICS
===================================================================================

CPU-only analytics over results produced by master_training.py.

  Exp 2:  Spectral / covariance analysis at ALL k values
          - Condition number of U and X_diffused
          - Singular value spectra of U and X_diffused
          - Intra-class vs inter-class spectral separation (on row-normalised U)
          - Variance analysis: per-feature variance before/after whitening,
            variance ratio reduction
          - Graph properties: num_edges, avg_degree

  Exp 8:  k-sensitivity analysis
          - Part A / Part B / Fisher score vs k (loaded from master_training.py output)
          - Includes Fisher at all k values (not just k=10)
          - Summary tables and per-dataset CSV

Prerequisites:
  Run master_training.py for all desired (dataset, split, component, k) configurations
  before running this script.

Usage:
  cd /PATH/TO/Spectral-Basis
  PYTHON=/PATH/TO/Spectral-Basis/venv/bin/python

  # Full analytics (all default k values)
  $PYTHON PAPER_EXPERIMENTS/master_analytics.py citeseer --splits fixed

  # Specific k values only
  $PYTHON PAPER_EXPERIMENTS/master_analytics.py citeseer --splits fixed --k_values 10

  # Run all 9 datasets
  for ds in amazon-computers amazon-photo citeseer coauthor-cs coauthor-physics \
             cora ogbn-arxiv pubmed wikics; do
      $PYTHON PAPER_EXPERIMENTS/master_analytics.py $ds --splits fixed
      $PYTHON PAPER_EXPERIMENTS/master_analytics.py $ds --splits random
  done

Arguments:
  dataset       Dataset name (must match a master_training.py run)
  --k_values    k values to analyse (default: 1 2 4 6 8 10 12 20 30)
  --splits      fixed | random (default: fixed)
  --component   lcc | whole (default: lcc)

Output:
  PAPER_RESULTS/{dataset}_{split}_{component}/analytics/exp2_spectral_analysis_k{k}.json
      (one file per k value)
  PAPER_RESULTS/{dataset}_{split}_{component}/analytics/exp8_k_sensitivity.json
  PAPER_RESULTS/{dataset}_{split}_{component}/analytics/exp8_k_sensitivity.csv

Author: Mohammad
===================================================================================
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx

# Import all utilities from src/utils.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_SRC_DIR   = os.path.join(_REPO_ROOT, 'src')
_DATA_DIR  = os.path.join(_REPO_ROOT, 'dataset')
sys.path.insert(0, _SRC_DIR)

# Patch for OGB compatibility (must come before utils import which triggers torch)
try:
    import torch
    _orig = torch.load
    def _patched(*a, **kw):
        if 'weights_only' not in kw:
            kw['weights_only'] = False
        return _orig(*a, **kw)
    torch.load = _patched
except ImportError:
    pass

from graph_utils import (
    load_dataset,
    build_graph_matrices, get_largest_connected_component_nx, extract_subgraph,
    compute_sgc_normalized_adjacency, sgc_precompute, compute_restricted_eigenvectors,
)


# ============================================================================
# Exp 2: Spectral / Covariance Analysis
# ============================================================================

def compute_spectral_analysis(U, X_diffused, eigenvalues, labels, train_idx,
                               num_classes, adj):
    """Compute spectral structure metrics for both U (eigenvectors) and X_diffused.

    Characterises the whitening effect: how U compares to raw diffused features.
    All heavy computations use train_idx only to avoid data leakage.

    Returns a dict of scalar/list results (no large matrices stored).
    """
    n, d_eff = U.shape
    d_raw    = X_diffused.shape[1]

    # ── U: singular values, condition number, variance (training nodes only) ──
    # Use U[train_idx] so cond_U and cond_X are computed on the same number of rows
    # and are directly comparable (both n_train x d matrices).
    # For fixed splits (n_train << d_eff), U_tr is wide: rank <= n_train, so the
    # smallest d_eff - n_train singular values are 0. Using sv_U[-1] would always give
    # inf. Instead use the rank-truncated condition number: max/min of non-zero sv only.
    U_tr     = U[train_idx]
    sv_U     = np.linalg.svd(U_tr, compute_uv=False)
    sv_U_pos = sv_U[sv_U > 1e-12]
    cond_U   = float(sv_U_pos[0] / sv_U_pos[-1]) if len(sv_U_pos) > 1 else float('inf')
    tv_U     = float(np.sum(sv_U ** 2))
    expl_U   = [float(np.sum(sv_U[:i+1]**2) / tv_U * 100) for i in range(min(d_eff, 20))]

    var_U    = np.var(U_tr, axis=0)
    var_ratio_U = float(var_U.max() / (var_U.min() + 1e-12))

    # ── X_diffused: singular values, condition number, variance ────────────
    X_tr     = X_diffused[train_idx]
    sv_X     = np.linalg.svd(X_tr, compute_uv=False)
    cond_X   = float(sv_X[0] / sv_X[-1]) if sv_X[-1] > 1e-12 else float('inf')
    tv_X     = float(np.sum(sv_X ** 2))
    expl_X   = [float(np.sum(sv_X[:i+1]**2) / tv_X * 100) for i in range(min(d_raw, 20))]

    var_X    = np.var(X_tr, axis=0)
    var_ratio_X = float(var_X.max() / (var_X.min() + 1e-12))

    # ── Variance analysis summary ──────────────────────────────────────────
    variance_analysis = {
        'U_variance_mean':          float(var_U.mean()),
        'U_variance_std':           float(var_U.std()),
        'U_variance_max':           float(var_U.max()),
        'U_variance_min':           float(var_U.min()),
        'U_variance_ratio':         var_ratio_U,
        'X_variance_mean':          float(var_X.mean()),
        'X_variance_std':           float(var_X.std()),
        'X_variance_max':           float(var_X.max()),
        'X_variance_min':           float(var_X.min()),
        'X_variance_ratio':         var_ratio_X,
        'variance_ratio_reduction': float(var_ratio_X / (var_ratio_U + 1e-12))
    }

    # ── Separability on row-normalised U (training set) ───────────────────
    U_norm  = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
    U_n_tr  = U_norm[train_idx]
    y_tr    = labels[train_idx]

    centroids = []
    for c in range(num_classes):
        mask = (y_tr == c)
        centroids.append(U_n_tr[mask].mean(axis=0) if mask.sum() > 0
                         else np.zeros(d_eff))
    centroids = np.array(centroids)

    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    inter_class_centroid_dist = float(np.mean(inter_dists)) if inter_dists else 0.0

    intra_dists = []
    for c in range(num_classes):
        mask = (y_tr == c)
        U_c  = U_n_tr[mask]
        if len(U_c) > 200:
            idx = np.random.choice(len(U_c), 200, replace=False)
            U_c = U_c[idx]
        for i in range(len(U_c)):
            for j in range(i + 1, min(i + 50, len(U_c))):
                intra_dists.append(float(np.linalg.norm(U_c[i] - U_c[j])))
    intra_class_mean_dist = float(np.mean(intra_dists)) if intra_dists else 0.0

    # spectral_separability = inter-class centroid distance / intra-class mean pairwise distance,
    # computed on row-normalised U restricted to training nodes.
    # Values <1 mean classes overlap (intra > inter), values >1 mean classes are well-separated.
    # On most real datasets this is <1 (overlap), so HIGHER is better but typical values are small.
    spectral_separability = (inter_class_centroid_dist / intra_class_mean_dist
                              if intra_class_mean_dist > 1e-10 else 0.0)

    # ── Eigenvalue statistics ──────────────────────────────────────────────
    eig_stats = {
        'min':         float(eigenvalues.min()),
        'max':         float(eigenvalues.max()),
        'mean':        float(eigenvalues.mean()),
        'std':         float(eigenvalues.std()),
        'd_effective': int(len(eigenvalues))
    }

    # ── Span verification ──────────────────────────────────────────────────
    # WARNING: This metric is MEANINGLESS for most datasets.
    # When d_eff >= n_train (e.g. Cora: d_eff=1427, n_train=140), U_tr has more
    # columns than rows, so pinv(U_tr) is a right-inverse and U_tr @ pinv(U_tr) = I.
    # This forces span_error = 0 always, regardless of whether U actually spans X.
    # DO NOT report this value in the paper or use it as a validation metric.
    # The correct check (project U onto col(X_diffused) using all nodes) is
    # guaranteed near-zero by Rayleigh-Ritz construction anyway.
    try:
        X_tr_centered = X_tr - X_tr.mean(axis=0)
        U_pinv        = np.linalg.pinv(U_tr)          # d_eff × n_train
        X_proj        = (U_tr @ U_pinv) @ X_tr_centered
        span_error    = float(
            np.linalg.norm(X_tr_centered - X_proj, 'fro') /
            (np.linalg.norm(X_tr_centered, 'fro') + 1e-12)
        )
    except Exception:
        span_error = None

    # ── Graph properties ───────────────────────────────────────────────────
    # adj.nnz includes self-loop entries; subtract them before halving.
    diag_nnz   = int(adj.diagonal().sum())
    num_edges  = int((adj.nnz - diag_nnz) // 2)
    avg_degree = float((adj.nnz - diag_nnz) / adj.shape[0])

    return {
        # U analysis
        'condition_number_U':             cond_U,
        'singular_values_U_top20':        [float(s) for s in sv_U[:20]],
        'singular_values_U_bottom5':      [float(s) for s in sv_U[-5:]],
        'explained_variance_U_pct_top20': expl_U,
        # X_diffused analysis
        'condition_number_X':             cond_X,
        'singular_values_X_top20':        [float(s) for s in sv_X[:20]],
        'explained_variance_X_pct_top20': expl_X,
        # Separability
        'intra_class_mean_dist':          intra_class_mean_dist,
        'inter_class_centroid_dist':      inter_class_centroid_dist,
        'spectral_separability':          float(spectral_separability),
        # Variance analysis
        'variance_analysis':              variance_analysis,
        # Eigenvalue stats
        'eigenvalue_stats':               eig_stats,
        # Span verification (should be ~0: U spans same space as X_diffused)
        'span_verification_error':        span_error,
        # Graph properties
        'num_edges':                      num_edges,
        'avg_degree':                     avg_degree,
    }


# ============================================================================
# Exp 8: k-Sensitivity Loading
# ============================================================================

def load_k_results(paper_results_dir, k):
    """Load results.json for a given k from master_training.py output."""
    path = os.path.join(paper_results_dir, f'k{k}', 'metrics', 'results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_fisher_k(paper_results_dir, k):
    """Load fisher_diagnostic.json for a given k."""
    path = os.path.join(paper_results_dir, f'k{k}', 'diagnostics', 'fisher_diagnostic.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_acc(exp_results, method_key):
    """Extract test_acc_mean × 100 from aggregated results dict."""
    if method_key not in exp_results:
        return None
    val = exp_results[method_key].get('test_acc_mean')
    if val is None:
        return None
    return round(float(val) * 100, 4)


# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description='Master Analytics (Exp 2 and Exp 8)')
parser.add_argument('dataset', type=str,
                    help='Dataset name (must match a master_training.py run)')
parser.add_argument('--k_values', type=int, nargs='+',
                    default=[1, 2, 4, 6, 8, 10, 12, 20, 30],
                    help='k values to analyse (default: 1 2 4 6 8 10 12 20 30)')
parser.add_argument('--splits', type=str, choices=['fixed', 'random'], default='fixed',
                    help='Split type (must match master_training.py run)')
parser.add_argument('--component', type=str, choices=['lcc', 'whole'], default='lcc',
                    help='Graph component (must match master_training.py run)')
args = parser.parse_args()

DATASET_NAME   = args.dataset
K_VALUES       = sorted(args.k_values)
SPLIT_TYPE     = args.splits
COMPONENT_TYPE = args.component

PAPER_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'PAPER_RESULTS',
    f'{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}'
)
ANALYTICS_DIR = os.path.join(PAPER_RESULTS_DIR, 'analytics')
os.makedirs(ANALYTICS_DIR, exist_ok=True)

print('=' * 80)
print('MASTER ANALYTICS (Exp 2 + Exp 8)')
print('=' * 80)
print(f'Dataset:       {DATASET_NAME}')
print(f'Split type:    {SPLIT_TYPE}')
print(f'Component:     {COMPONENT_TYPE}')
print(f'k values:      {K_VALUES}')
print(f'Results dir:   {PAPER_RESULTS_DIR}')
print('=' * 80)

# ============================================================================
# Load dataset and graph (needed for Exp 2 spectral analysis)
# ============================================================================

print('\n[Step 1] Loading dataset and graph...')
(edge_index, X_raw, labels, num_nodes, num_classes,
 train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(DATASET_NAME, root=_DATA_DIR)
print(f'Nodes: {num_nodes:,}, Features: {X_raw.shape[1]}, Classes: {num_classes}')

# Build graph matrices: adj, L=Laplacian, D=degree (correct order from src/utils.py)
adj, L, D = build_graph_matrices(edge_index, num_nodes)

if COMPONENT_TYPE == 'lcc':
    print('\n[Step 2] Extracting LCC...')
    lcc_mask = get_largest_connected_component_nx(adj)
    split_idx_orig_dict = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig
    }
    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_orig_dict
    )
    # adj from extract_subgraph already has self-loops — strip before rebuilding
    # so build_graph_matrices adds them exactly once (same fix as master_training.py).
    adj_no_loops = adj - sp.diags(adj.diagonal())
    adj_coo      = adj_no_loops.tocoo()
    edge_idx_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj, L, D    = build_graph_matrices(edge_idx_lcc, adj.shape[0])
    num_components = 0
else:
    split_idx = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig
    }
    G_tmp = nx.from_scipy_sparse_array(adj)
    num_components = len(list(nx.connected_components(G_tmp))) - 1

num_nodes   = X_raw.shape[0]
num_classes = len(np.unique(labels))

# For spectral analysis, use the same train_idx that master_training.py used for split 0.
# For random splits, master_training.py generates: np.random.seed(0), shuffle, take first 60%.
# Using fixed split's train_idx here when SPLIT_TYPE='random' would analyze the wrong nodes.
if SPLIT_TYPE == 'random':
    # Replicate master_training.py split 0 exactly (same legacy API: seed=0, shuffle, 60%)
    np.random.seed(0)
    _indices = np.arange(num_nodes)
    np.random.shuffle(_indices)
    train_idx = _indices[:int(0.6 * num_nodes)]
else:
    train_idx = split_idx['train_idx']

features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()
A_sgc          = compute_sgc_normalized_adjacency(adj)
print(f'After LCC: nodes={num_nodes:,}, classes={num_classes}')

# ============================================================================
# Exp 2: Spectral Analysis at ALL k values
# ============================================================================

print(f'\n{"="*80}')
print('EXP 2: SPECTRAL ANALYSIS (all k values)')
print(f'{"="*80}')

np.random.seed(42)  # reproducible subsampling in intra-class distance

for k in K_VALUES:
    print(f'\n  k={k}:')

    # Load pre-computed matrices saved by master_training.py (split 0)
    npz_path = os.path.join(PAPER_RESULTS_DIR, f'k{k}', 'matrices', 'split0_matrices.npz')
    if os.path.exists(npz_path):
        data       = np.load(npz_path)
        U          = data['U']
        X_diffused = data['X_diffused']
        eigs       = data['eigenvalues']
        train_idx  = data['train_idx']
        d_eff      = U.shape[1]
        ortho_err  = None   # already verified during training
        print(f'    Loaded matrices from {npz_path}')
    else:
        # Fallback: recompute (slower — run master_training.py first)
        print(f'    WARNING: no cached matrices found, recomputing...')
        X_diffused = sgc_precompute(features_dense.copy(), A_sgc, k)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(
            X_diffused, L, D, num_components
        )
        # train_idx is already set above (split 0 random or fixed), do not override here

    print(f'    d_effective={d_eff}, ortho_error={ortho_err}')

    analysis = compute_spectral_analysis(
        U, X_diffused, eigs, labels, train_idx, num_classes, adj
    )

    sp_err = analysis['span_verification_error']
    print(f'    Condition U={analysis["condition_number_U"]:.4f}  '
          f'Condition X={analysis["condition_number_X"]:.4f}  '
          f'Separability={analysis["spectral_separability"]:.4f}  '
          f'SpanErr={sp_err:.2e}' if sp_err is not None else
          f'    Condition U={analysis["condition_number_U"]:.4f}  '
          f'Condition X={analysis["condition_number_X"]:.4f}  '
          f'Separability={analysis["spectral_separability"]:.4f}')
    va = analysis['variance_analysis']
    print(f'    Var ratio U={va["U_variance_ratio"]:.2f}  '
          f'Var ratio X={va["X_variance_ratio"]:.2f}  '
          f'Reduction={va["variance_ratio_reduction"]:.2f}x')

    exp2_output = {
        'dataset':        DATASET_NAME,
        'split_type':     SPLIT_TYPE,
        'component_type': COMPONENT_TYPE,
        'k':              k,
        'd_effective':    int(d_eff),
        'ortho_error':    float(ortho_err) if ortho_err is not None else None,
        'analysis':       analysis
    }

    exp2_path = os.path.join(ANALYTICS_DIR, f'exp2_spectral_analysis_k{k}.json')
    with open(exp2_path, 'w') as f:
        json.dump(exp2_output, f, indent=2)
    print(f'    Saved: {exp2_path}')

# ============================================================================
# Exp 8: k-Sensitivity Analysis
# ============================================================================

print(f'\n{"="*80}')
print('EXP 8: k-SENSITIVITY ANALYSIS')
print(f'{"="*80}')

k_sensitivity_rows = []

for k in K_VALUES:
    result = load_k_results(PAPER_RESULTS_DIR, k)
    if result is None:
        print(f'  k={k}: No results found (run master_training.py for this k first)')
        continue

    exps = result.get('experiments', {})
    fa   = result.get('framework_analysis', {})

    sgc_mlp   = extract_acc(exps, 'sgc_mlp_baseline')
    restr_std = extract_acc(exps, 'restricted_standard_mlp')
    restr_rn  = extract_acc(exps, 'restricted_rownorm_mlp')

    part_a  = round(float(fa['part_a_pp']),        4) if fa.get('part_a_pp')        is not None else None
    part_b  = round(float(fa['part_b_pp']),        4) if fa.get('part_b_pp')        is not None else None
    rem_gap = round(float(fa['remaining_gap_pp']), 4) if fa.get('remaining_gap_pp') is not None else None

    row = {
        'k':         k,
        'sgc_mlp':   sgc_mlp,
        'restr_std': restr_std,
        'restr_rn':  restr_rn,
        'part_a':    part_a,
        'part_b':    part_b,
        'rem_gap':   rem_gap,
        'n_runs':    result.get('total_runs_per_method', None)
    }

    # Fisher loaded from diagnostics (available at all k)
    fisher_data = load_fisher_k(PAPER_RESULTS_DIR, k)
    if fisher_data:
        # fisher_score = mean across all splits (M-4 fix promoted mean into this key;
        # 'mean_fisher_score' was removed — do not read it).
        row['fisher_score'] = fisher_data.get('fisher_score')
        # Also grab fisher_score_X_diffused if present (from per_split[0])
        per_split = fisher_data.get('per_split', [])
        if per_split:
            row['fisher_score_X_diffused'] = per_split[0].get('fisher_score_X_diffused')

    # k=10: extended methods
    if k == 10:
        row['log_magnitude']  = extract_acc(exps, 'log_magnitude')
        row['dual_stream']    = extract_acc(exps, 'dual_stream')
        for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            row[f'spec_alpha{alpha}'] = extract_acc(exps, f'spectral_rownorm_alpha{alpha}')
        row['best_spectral_alpha']     = fa.get('best_spectral_alpha')
        row['part_b6']                 = fa.get('part_b6_pp')
        row['part_b5_logmag']          = fa.get('part_b5_logmag_pp')
        row['part_b5_dual']            = fa.get('part_b5_dual_pp')
        row['best_nested_spheres_key'] = fa.get('best_nested_spheres_key')
        row['part_b6_nested']          = fa.get('part_b6_nested_pp')

    k_sensitivity_rows.append(row)

# Print k-sensitivity table
if k_sensitivity_rows:
    print(f'\n  {"k":>4}  {"SGC+MLP":>9}  {"R+Std":>9}  {"R+RN":>9}  '
          f'{"Part A":>9}  {"Part B":>9}  {"Rem.Gap":>9}  {"Fisher":>9}  {"n_runs":>7}')
    print(f'  {"-"*95}')
    for row in k_sensitivity_rows:
        def f(v): return f'{v:+.3f}' if v is not None else '    N/A'
        def p(v): return f'{v:.2f}%'  if v is not None else '    N/A'
        def fp(v): return f'{v:.5f}'  if v is not None else '    N/A'
        print(f'  {row["k"]:>4}  {p(row["sgc_mlp"]):>9}  {p(row["restr_std"]):>9}  '
              f'{p(row["restr_rn"]):>9}  {f(row["part_a"]):>9}  {f(row["part_b"]):>9}  '
              f'{f(row["rem_gap"]):>9}  {fp(row.get("fisher_score")):>9}  '
              f'{str(row["n_runs"]):>7}')

# Save k-sensitivity JSON
exp8_output = {
    'dataset':        DATASET_NAME,
    'split_type':     SPLIT_TYPE,
    'component_type': COMPONENT_TYPE,
    'k_values_found': [row['k'] for row in k_sensitivity_rows],
    'k_sensitivity':  k_sensitivity_rows
}

exp8_json_path = os.path.join(ANALYTICS_DIR, 'exp8_k_sensitivity.json')
with open(exp8_json_path, 'w') as f:
    json.dump(exp8_output, f, indent=2)
print(f'\n  Saved JSON: {exp8_json_path}')

# Save k-sensitivity CSV
if k_sensitivity_rows:
    all_cols = [
        'k', 'sgc_mlp', 'restr_std', 'restr_rn', 'part_a', 'part_b', 'rem_gap', 'n_runs',
        'fisher_score', 'fisher_score_X_diffused',
        'log_magnitude', 'dual_stream',
        'spec_alpha-1.0', 'spec_alpha-0.5', 'spec_alpha0.0', 'spec_alpha0.5', 'spec_alpha1.0',
        'best_spectral_alpha', 'part_b6', 'part_b5_logmag', 'part_b5_dual',
        'best_nested_spheres_key', 'part_b6_nested'
    ]
    exp8_csv_path = os.path.join(ANALYTICS_DIR, 'exp8_k_sensitivity.csv')
    with open(exp8_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        for row in k_sensitivity_rows:
            writer.writerow(row)
    print(f'  Saved CSV:  {exp8_csv_path}')

print(f'\n{"="*80}')
print('MASTER ANALYTICS COMPLETE')
print(f'{"="*80}')
print(f'Analytics saved to: {ANALYTICS_DIR}')
