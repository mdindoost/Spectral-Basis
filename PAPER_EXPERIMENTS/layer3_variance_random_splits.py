"""
Layer 3 variance check — Cora k=10, random splits (seeds 0-4) vs fixed split.
Also computes the same metrics for X_diff to compare variance disparity.

Split logic mirrors master_training.py exactly:
    np.random.seed(s_idx); np.random.shuffle(indices); 60/20/20
    where indices = np.arange(n_lcc_nodes)
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

K           = 10
DATASET     = 'cora'
NUM_SPLITS  = 5
OUT_DIR     = os.path.join(os.path.dirname(__file__), 'PAPER_OUTPUT')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset and build LCC + diffuse once
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {DATASET} ...")
edge_index, features, labels, num_nodes, num_classes, \
    train_idx_orig, val_idx_orig, test_idx_orig = load_dataset(DATASET, root='./dataset')

adj, L, D = build_graph_matrices(edge_index, num_nodes)
lcc_mask   = get_largest_connected_component_nx(adj)
split_idx  = {'train': train_idx_orig, 'val': val_idx_orig, 'test': test_idx_orig}

adj_lcc, features_lcc, labels_lcc, split_lcc = extract_subgraph(
    adj, features, labels, lcc_mask, split_idx)

n_lcc = adj_lcc.shape[0]
print(f"LCC nodes: {n_lcc}")

# Rebuild L and D for LCC (strip self-loops that build_graph_matrices will re-add)
rows, cols = adj_lcc.nonzero()
mask_no_self = rows != cols
edge_index_lcc = np.stack([rows[mask_no_self], cols[mask_no_self]], axis=0)
_, L_lcc, D_lcc = build_graph_matrices(edge_index_lcc, n_lcc)

# Diffuse
A_hat  = compute_sgc_normalized_adjacency(adj_lcc)
X_diff = sgc_precompute(features_lcc.astype(np.float64), A_hat, K)
print(f"X_diff shape: {X_diff.shape}")

# Compute U and eigenvalues once (split-independent)
U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(
    X_diff, L_lcc, D_lcc, num_components=0)
print(f"d_eff: {d_eff}, ortho_error: {ortho_error:.2e}")
print(f"eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")

# Fixed split train_idx (from LCC-remapped split)
fixed_train_idx = split_lcc['train']
print(f"Fixed split n_train: {len(fixed_train_idx)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Variance analysis function
# ─────────────────────────────────────────────────────────────────────────────
def col_var_metrics(mat, train_idx, eigenvalues=None, label=""):
    """
    Compute column-variance metrics for mat[train_idx].
    mat: (n, d) array — either U or X_diff
    eigenvalues: if provided, compute Spearman rho(lambda, col_var)
    """
    cv_arr = mat[train_idx].var(axis=0)        # (d,)

    mean_cv = cv_arr.mean()
    std_cv  = cv_arr.std()
    cv_coef = std_cv / (mean_cv + 1e-30)       # coefficient of variation

    frac_near = np.mean(cv_arr < 1e-4)
    n_near    = np.sum(cv_arr < 1e-4)

    rho = None
    pval = None
    if eigenvalues is not None:
        valid = eigenvalues > 1e-12
        rho, pval = stats.spearmanr(eigenvalues[valid], cv_arr[valid])

    return dict(
        n_train   = len(train_idx),
        col_vars  = cv_arr,
        mean      = mean_cv,
        std       = std_cv,
        cv        = cv_coef,
        cv_range  = (cv_arr.min(), cv_arr.max()),
        frac_near = frac_near,
        n_near    = n_near,
        rho       = rho,
        pval      = pval,
    )

def print_row(label, res):
    rho_str = f"{res['rho']:+.4f} (p={res['pval']:.1e})" if res['rho'] is not None else "  N/A"
    print(f"  {label:<30} n_train={res['n_train']:5d}  "
          f"CV={res['cv']:.4f}  "
          f"frac<1e-4={res['frac_near']:.4f} ({res['n_near']}/{len(res['col_vars'])})  "
          f"rho={rho_str}  "
          f"range=[{res['cv_range'][0]:.2e},{res['cv_range'][1]:.2e}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fixed split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("FIXED SPLIT")
fixed_U    = col_var_metrics(U,      fixed_train_idx, eigenvalues, "U (fixed)")
fixed_Xd   = col_var_metrics(X_diff, fixed_train_idx, None,        "X_diff (fixed)")
print_row("U [fixed split]",      fixed_U)
print_row("X_diff [fixed split]", fixed_Xd)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Random splits (seeds 0–4), mirroring master_training.py exactly
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("RANDOM SPLITS (seeds 0–4)")

rand_U_results  = []
rand_Xd_results = []

for s_idx in range(NUM_SPLITS):
    np.random.seed(s_idx)
    indices = np.arange(n_lcc)
    np.random.shuffle(indices)
    n_tr = int(0.6 * n_lcc)
    n_va = int(0.2 * n_lcc)
    tr   = indices[:n_tr]

    res_U  = col_var_metrics(U,      tr, eigenvalues, f"U seed={s_idx}")
    res_Xd = col_var_metrics(X_diff, tr, None,        f"X_diff seed={s_idx}")

    rand_U_results.append(res_U)
    rand_Xd_results.append(res_Xd)

    print_row(f"U      [seed={s_idx}]", res_U)
    print_row(f"X_diff [seed={s_idx}]", res_Xd)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregate random-split stats
# ─────────────────────────────────────────────────────────────────────────────
rand_cv_U    = np.array([r['cv']        for r in rand_U_results])
rand_frac_U  = np.array([r['frac_near'] for r in rand_U_results])
rand_rho_U   = np.array([r['rho']       for r in rand_U_results])

rand_cv_Xd   = np.array([r['cv']        for r in rand_Xd_results])
rand_frac_Xd = np.array([r['frac_near'] for r in rand_Xd_results])

print("="*90)
print("AGGREGATE (random splits, mean ± std over 5 seeds)")
print(f"  U      CV:         {rand_cv_U.mean():.4f} ± {rand_cv_U.std():.4f}")
print(f"  U      frac<1e-4:  {rand_frac_U.mean():.4f} ± {rand_frac_U.std():.4f}")
print(f"  U      Spearman ρ: {rand_rho_U.mean():.4f} ± {rand_rho_U.std():.4f}")
print(f"  X_diff CV:         {rand_cv_Xd.mean():.4f} ± {rand_cv_Xd.std():.4f}")
print(f"  X_diff frac<1e-4:  {rand_frac_Xd.mean():.4f} ± {rand_frac_Xd.std():.4f}")

print("\n" + "="*90)
print("DIRECT COMPARISON TABLE  (Cora k=10)")
print(f"{'':35} {'CV(col_vars)':>14} {'frac<1e-4':>12} {'Spearman ρ':>12} {'n_train':>9}")
print("-"*85)
print(f"  {'U  — fixed split':<33} {fixed_U['cv']:>14.4f} "
      f"{fixed_U['frac_near']:>12.4f} {fixed_U['rho']:>+12.4f} {fixed_U['n_train']:>9}")
print(f"  {'U  — random (mean±std)':<33} "
      f"{rand_cv_U.mean():>14.4f}±{rand_cv_U.std():.4f} "
      f"{rand_frac_U.mean():>12.4f}±{rand_frac_U.std():.4f} "
      f"{rand_rho_U.mean():>+12.4f}±{rand_rho_U.std():.4f} "
      f"{'~1491':>9}")
print(f"  {'X_diff — fixed split':<33} {fixed_Xd['cv']:>14.4f} "
      f"{fixed_Xd['frac_near']:>12.4f} {'N/A':>12} {fixed_Xd['n_train']:>9}")
print(f"  {'X_diff — random (mean±std)':<33} "
      f"{rand_cv_Xd.mean():>14.4f}±{rand_cv_Xd.std():.4f} "
      f"{rand_frac_Xd.mean():>12.4f}±{rand_frac_Xd.std():.4f} "
      f"{'N/A':>12} "
      f"{'~1491':>9}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Scatter plot: fixed vs one random split (seed=0), U and X_diff side-by-side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f'Cora k={K}: Column variance vs eigenvalue (U) and vs feature index (X_diff)',
             fontsize=12)

lam = eigenvalues
valid = lam > 1e-12

def scatter_panel(ax, lam_or_idx, col_vars, title, xlabel, rho=None, C_hat=None):
    near = col_vars < 1e-4
    ax.scatter(lam_or_idx[~near], col_vars[~near], s=3, alpha=0.4,
               color='steelblue', label='col_var ≥ 1e-4')
    ax.scatter(lam_or_idx[near], col_vars[near], s=4, alpha=0.5,
               color='red', label=f'col_var < 1e-4 ({near.sum()})')
    if C_hat is not None:
        lam_ref = np.linspace(lam_or_idx.min(), lam_or_idx.max(), 300)
        ax.plot(lam_ref, C_hat / lam_ref**2, 'k--', lw=1.2, label=r'$C/\lambda^2$')
    ax.set_yscale('log')
    rho_str = f", ρ={rho:+.3f}" if rho is not None else ""
    ax.set_title(f'{title}{rho_str}', fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('col_var (log)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# U, fixed split
cv_U_fixed = fixed_U['col_vars']
C_hat_fixed = np.median(cv_U_fixed[valid] * lam[valid]**2)
scatter_panel(axes[0,0], lam[valid], cv_U_fixed[valid],
              f'U — fixed split (n_train={len(fixed_train_idx)})',
              r'$\lambda_j$', rho=fixed_U['rho'], C_hat=C_hat_fixed)

# U, random split seed=0
cv_U_rand0 = rand_U_results[0]['col_vars']
tr0_n = rand_U_results[0]['n_train']
C_hat_rand0 = np.median(cv_U_rand0[valid] * lam[valid]**2)
scatter_panel(axes[0,1], lam[valid], cv_U_rand0[valid],
              f'U — random split seed=0 (n_train={tr0_n})',
              r'$\lambda_j$', rho=rand_U_results[0]['rho'], C_hat=C_hat_rand0)

# X_diff, fixed split — sort by feature variance for readability
cv_Xd_fixed = fixed_Xd['col_vars']
feat_idx_fixed = np.argsort(cv_Xd_fixed)     # sorted by increasing variance
scatter_panel(axes[1,0], np.arange(len(cv_Xd_fixed)), cv_Xd_fixed[feat_idx_fixed],
              f'X_diff — fixed split (n_train={len(fixed_train_idx)})\nCV={fixed_Xd["cv"]:.3f}',
              'Feature index (sorted by col_var)')

# X_diff, random split seed=0
cv_Xd_rand0 = rand_Xd_results[0]['col_vars']
tr0_Xd_n    = rand_Xd_results[0]['n_train']
feat_idx_rand0 = np.argsort(cv_Xd_rand0)
scatter_panel(axes[1,1], np.arange(len(cv_Xd_rand0)), cv_Xd_rand0[feat_idx_rand0],
              f'X_diff — random split seed=0 (n_train={tr0_Xd_n})\nCV={rand_Xd_results[0]["cv"]:.3f}',
              'Feature index (sorted by col_var)')

fig.tight_layout()
fname = os.path.join(OUT_DIR, 'layer3_variance_cora_random_vs_fixed.png')
fig.savefig(fname, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Key interpretive checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("INTERPRETIVE CHECKS")

cv_ratio = fixed_U['cv'] / rand_cv_U.mean()
print(f"  CV ratio fixed/random (U):        {cv_ratio:.3f}x")
print(f"  frac<1e-4 ratio fixed/random (U): {fixed_U['frac_near'] / rand_frac_U.mean():.3f}x")

u_vs_xd_cv_fixed  = fixed_U['cv']  / fixed_Xd['cv']
u_vs_xd_cv_random = rand_cv_U.mean() / rand_cv_Xd.mean()
print(f"  CV(U) / CV(X_diff) fixed split:   {u_vs_xd_cv_fixed:.3f}x  "
      f"({'U has MORE disparity' if u_vs_xd_cv_fixed>1 else 'X_diff has MORE disparity'})")
print(f"  CV(U) / CV(X_diff) random split:  {u_vs_xd_cv_random:.3f}x  "
      f"({'U has MORE disparity' if u_vs_xd_cv_random>1 else 'X_diff has MORE disparity'})")

u_frac_vs_xd_fixed  = fixed_U['frac_near']  / (fixed_Xd['frac_near'] + 1e-10)
u_frac_vs_xd_random = rand_frac_U.mean() / (rand_frac_Xd.mean() + 1e-10)
print(f"  frac<1e-4(U) / frac<1e-4(X_diff) fixed:  {u_frac_vs_xd_fixed:.3f}x")
print(f"  frac<1e-4(U) / frac<1e-4(X_diff) random: {u_frac_vs_xd_random:.3f}x")
