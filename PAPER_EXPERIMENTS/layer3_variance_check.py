"""
Theoretical verification: column-variance vs eigenvalue ordering after D-orthonormalization.

Predictions tested:
  A: Spearman rho(lambda_j, col_var_j) — should be negative (higher lambda => lower var)
  B: CV(col_vars) Cora > CV(col_vars) Amazon-Computers  (concentrated vs dispersed)
  C: Fraction of columns with col_var < 1e-4 — should be high for Cora at k=10

Datasets: Cora (fixed split), Amazon-Computers (fixed split), both at k=10.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.graph_utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
)

K = 10
DATASETS = ['cora', 'amazon-computers']
OUT_DIR = os.path.join(os.path.dirname(__file__), 'PAPER_OUTPUT')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def load_and_compute(dataset_name):
    """
    Load dataset, extract LCC, diffuse at k=K, compute U and eigenvalues.
    Returns U, eigenvalues, train_idx (LCC-remapped).
    """
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} ...")
    edge_index, features, labels, num_nodes, num_classes, train_idx, val_idx, test_idx = \
        load_dataset(dataset_name, root='./dataset')

    # Build graph matrices (includes self-loops)
    adj, L, D = build_graph_matrices(edge_index, num_nodes)

    # Extract LCC
    lcc_mask = get_largest_connected_component_nx(adj)
    split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    adj_lcc, features_lcc, labels_lcc, split_lcc = extract_subgraph(
        adj, features, labels, lcc_mask, split_idx)

    train_idx_lcc = split_lcc['train']
    n_lcc = adj_lcc.shape[0]
    print(f"  LCC nodes: {n_lcc}, train nodes: {len(train_idx_lcc)}")

    # Rebuild L and D for LCC
    _, L_lcc, D_lcc = build_graph_matrices(
        np.array(adj_lcc.nonzero()), n_lcc)
    # Note: build_graph_matrices expects (2, E) edge_index
    # adj_lcc.nonzero() returns (row_arr, col_arr) — reshape to (2, E)
    rows, cols = adj_lcc.nonzero()
    # Remove self-loops that were already in adj (build_graph_matrices adds them)
    mask_no_self = rows != cols
    edge_index_lcc = np.stack([rows[mask_no_self], cols[mask_no_self]], axis=0)
    _, L_lcc, D_lcc = build_graph_matrices(edge_index_lcc, n_lcc)

    # SGC diffusion
    A_hat = compute_sgc_normalized_adjacency(adj_lcc)
    X_diff = sgc_precompute(features_lcc.astype(np.float64), A_hat, K)
    print(f"  X_diff shape: {X_diff.shape}")

    # Compute restricted eigenvectors
    U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(
        X_diff, L_lcc, D_lcc, num_components=0)
    print(f"  d_eff: {d_eff}, ortho_error: {ortho_error:.2e}")
    print(f"  eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")

    return U, eigenvalues, train_idx_lcc, d_eff

# ─────────────────────────────────────────────────────────────────────────────
def analyze(dataset_name, U, eigenvalues, train_idx, d_eff):
    print(f"\n--- Analysis: {dataset_name} k={K} ---")

    U_train = U[train_idx]                           # (n_train, d_eff)
    col_vars = U_train.var(axis=0)                   # (d_eff,)

    lam = eigenvalues                                # already sorted ascending

    # Guard: remove zero eigenvalues from correlation (constant columns)
    valid = lam > 1e-12
    rho, pval = stats.spearmanr(lam[valid], col_vars[valid])

    cv = col_vars.std() / (col_vars.mean() + 1e-30)

    frac_near_const = np.mean(col_vars < 1e-4)
    n_near_const = np.sum(col_vars < 1e-4)

    print(f"  n_train:                   {len(train_idx)}")
    print(f"  d_eff:                     {d_eff}")
    print(f"  Spearman rho(lam, colvar): {rho:.4f}  (p={pval:.2e})")
    print(f"  CV of col_vars:            {cv:.4f}")
    print(f"  col_var range:             [{col_vars.min():.2e}, {col_vars.max():.2e}]")
    print(f"  col_var mean:              {col_vars.mean():.2e}")
    print(f"  col_var median:            {np.median(col_vars):.2e}")
    print(f"  Frac cols < 1e-4:          {frac_near_const:.4f}  ({n_near_const}/{d_eff})")
    print(f"  Declining trend present:   {'YES' if rho < -0.1 else 'NO'}")
    print(f"  lam_max/lam_min:           {lam[valid].max()/lam[valid].min():.1f}x")

    # Check 1/lambda^2 fit quality
    # Predict col_var ~ C / lam^2, estimate C by median ratio
    lam_pos = lam[valid & (col_vars > 0)]
    cv_pos  = col_vars[valid & (col_vars > 0)]
    C_hat   = np.median(cv_pos * lam_pos**2)
    predicted = C_hat / lam_pos**2
    # Spearman between actual and predicted (both sorted by lam, same ordering)
    rho_fit, _ = stats.spearmanr(cv_pos, predicted)
    residual_log = np.std(np.log(cv_pos) - np.log(predicted))
    print(f"  1/lam^2 fit: rho={rho_fit:.4f}, log-residual std={residual_log:.2f}")
    print(f"  1/lam^2 fits well:         {'YES' if residual_log < 0.5 else 'NO'}")

    return dict(
        rho=rho, pval=pval, cv=cv,
        frac_near_const=frac_near_const, n_near_const=n_near_const,
        col_vars=col_vars, lam=lam, valid=valid,
        lam_pos=lam_pos, cv_pos=cv_pos, C_hat=C_hat,
        residual_log=residual_log
    )

# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter(dataset_name, res):
    lam    = res['lam']
    cv     = res['col_vars']
    valid  = res['valid']
    C_hat  = res['C_hat']
    lam_pos = res['lam_pos']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter: col_var vs lambda
    ax.scatter(lam[valid], cv[valid], s=3, alpha=0.4, color='steelblue',
               label='col_var (train nodes)')

    # Near-constant columns (col_var < 1e-4)
    near = (cv < 1e-4) & valid
    ax.scatter(lam[near], cv[near], s=6, alpha=0.7, color='red',
               label=f'col_var < 1e-4 ({near.sum()} cols)')

    # 1/lambda^2 reference
    lam_ref = np.linspace(lam[valid].min(), lam[valid].max(), 300)
    ax.plot(lam_ref, C_hat / lam_ref**2, 'k--', lw=1.5,
            label=r'$C / \lambda^2$ reference')

    ax.set_yscale('log')
    ax.set_xlabel(r'Eigenvalue $\lambda_j$ (normalized Laplacian)', fontsize=11)
    ax.set_ylabel(r'$\mathrm{Var}[U_{\cdot j}]$ over train nodes (log scale)', fontsize=11)
    ax.set_title(
        f'{dataset_name.capitalize()} k={K}: Column variance vs eigenvalue\n'
        f'Spearman ρ={res["rho"]:.3f}, CV={res["cv"]:.3f}, '
        f'frac<1e-4={res["frac_near_const"]:.3f}',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fname = os.path.join(OUT_DIR,
        f'layer3_variance_{dataset_name.replace("-", "_")}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    results = {}
    for ds in DATASETS:
        U, eigenvalues, train_idx, d_eff = load_and_compute(ds)
        res = analyze(ds, U, eigenvalues, train_idx, d_eff)
        plot_scatter(ds, res)
        results[ds] = res

    # ── Prediction B: compare CVs ──────────────────────────────────────────
    print("\n" + "="*60)
    print("PREDICTION B SUMMARY")
    cv_cora  = results['cora']['cv']
    cv_amzn  = results['amazon-computers']['cv']
    print(f"  CV(col_vars) Cora:              {cv_cora:.4f}")
    print(f"  CV(col_vars) Amazon-Computers:  {cv_amzn:.4f}")
    print(f"  Prediction B (Cora CV > Amazon CV): "
          f"{'CONFIRMED' if cv_cora > cv_amzn else 'VIOLATED'}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY TABLE")
    print(f"{'Metric':<45} {'Cora':>12} {'Amazon-Comp':>14}")
    print("-"*72)
    for metric, key in [
        ("Spearman rho(lambda, col_var)", 'rho'),
        ("CV(col_vars)", 'cv'),
        ("Frac cols with col_var < 1e-4", 'frac_near_const'),
    ]:
        vc = results['cora'][key]
        va = results['amazon-computers'][key]
        print(f"  {metric:<43} {vc:>12.4f} {va:>14.4f}")

    print(f"\n  {'Declining trend present':<43} "
          f"{'YES' if results['cora']['rho'] < -0.1 else 'NO':>12} "
          f"{'YES' if results['amazon-computers']['rho'] < -0.1 else 'NO':>14}")
    print(f"  {'1/lambda^2 fits':<43} "
          f"{'YES' if results['cora']['residual_log'] < 0.5 else 'NO':>12} "
          f"{'YES' if results['amazon-computers']['residual_log'] < 0.5 else 'NO':>14}")

if __name__ == '__main__':
    main()
