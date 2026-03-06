"""
Check: does X_diff column variance align with per-column Fisher discriminability?

Per-column Fisher score for column j (over training nodes only):
    F_j = between_class_var_j / within_class_var_j

    between_class_var_j = sum_c (n_c/N) * (mu_cj - mu_j)^2
    within_class_var_j  = sum_c (n_c/N) * Var[column j | class c]

If Spearman rho(col_var_j, F_j) >> 0 for X_diff: variance ordering aligns with discriminativity.
If Spearman rho(col_var_j, F_j) ~0 for U:         variance ordering does NOT align.

Dataset: Cora k=10, random splits (seeds 0-4).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.graph_utils import (
    load_dataset, build_graph_matrices,
    get_largest_connected_component_nx, extract_subgraph,
    compute_sgc_normalized_adjacency, sgc_precompute,
    compute_restricted_eigenvectors,
)

K          = 10
DATASET    = 'cora'
NUM_SPLITS = 5
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'PAPER_OUTPUT')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Build LCC + diffuse + compute U  (split-independent)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {DATASET} ...")
(edge_index, features, labels, num_nodes, num_classes,
 train_orig, val_orig, test_orig) = load_dataset(DATASET, root='./dataset')

adj, L, D = build_graph_matrices(edge_index, num_nodes)
lcc_mask  = get_largest_connected_component_nx(adj)
split_idx = {'train': train_orig, 'val': val_orig, 'test': test_orig}

adj_lcc, feat_lcc, labs_lcc, split_lcc = extract_subgraph(
    adj, features, labels, lcc_mask, split_idx)
n_lcc = adj_lcc.shape[0]
print(f"LCC nodes: {n_lcc}")

rows, cols = adj_lcc.nonzero()
no_self = rows != cols
_, L_lcc, D_lcc = build_graph_matrices(
    np.stack([rows[no_self], cols[no_self]], axis=0), n_lcc)

A_hat  = compute_sgc_normalized_adjacency(adj_lcc)
X_diff = sgc_precompute(feat_lcc.astype(np.float64), A_hat, K)

U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(
    X_diff, L_lcc, D_lcc, num_components=0)
print(f"X_diff shape: {X_diff.shape}, d_eff: {d_eff}, ortho_error: {ortho_error:.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# Per-column Fisher score  (vectorised, over training nodes only)
# ─────────────────────────────────────────────────────────────────────────────
def per_column_fisher(mat, train_idx, labels_all):
    """
    mat:        (n_nodes, d)  full matrix
    train_idx:  indices into mat (and labels_all)
    labels_all: (n_nodes,) integer class labels

    Returns:
        fisher: (d,)  F_j = between_var_j / within_var_j  (NaN if within_var=0)
        col_vars: (d,) empirical variance over train_idx
    """
    M   = mat[train_idx]          # (n_tr, d)
    y   = labels_all[train_idx]   # (n_tr,)
    N   = len(train_idx)
    mu  = M.mean(axis=0)          # (d,) global mean

    between = np.zeros(M.shape[1])
    within  = np.zeros(M.shape[1])

    for c in np.unique(y):
        mask = (y == c)
        n_c  = mask.sum()
        Mc   = M[mask]              # (n_c, d)
        mu_c = Mc.mean(axis=0)      # (d,)
        w    = n_c / N
        between += w * (mu_c - mu) ** 2
        within  += w * Mc.var(axis=0)   # biased var inside class

    # Avoid division by zero: if within==0 but between>0 → infinite Fisher
    # Use a small floor to get finite ratio; flag separately
    within_safe = np.where(within == 0, np.nan, within)
    fisher = between / within_safe    # (d,)  NaN where within=0

    col_vars = M.var(axis=0)
    return fisher, col_vars

# ─────────────────────────────────────────────────────────────────────────────
# Run over 5 random splits
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
header = (f"{'seed':>5}  {'rho(var,F) Xd':>14}  {'p Xd':>9}  "
          f"{'rho(var,F) U':>13}  {'p U':>9}  "
          f"{'rho(lam,F_U)':>13}  {'p':>9}  "
          f"{'n_valid_Xd':>11}  {'n_valid_U':>10}")
print(header)
print("-"*90)

results = []

for s_idx in range(NUM_SPLITS):
    np.random.seed(s_idx)
    idx = np.arange(n_lcc)
    np.random.shuffle(idx)
    n_tr = int(0.6 * n_lcc)
    tr   = idx[:n_tr]

    # --- X_diff ---
    F_xd, cv_xd = per_column_fisher(X_diff, tr, labs_lcc)
    valid_xd  = np.isfinite(F_xd) & (cv_xd > 0)
    rho_xd, p_xd = stats.spearmanr(cv_xd[valid_xd], F_xd[valid_xd])

    # --- U ---
    F_u, cv_u = per_column_fisher(U, tr, labs_lcc)
    valid_u   = np.isfinite(F_u) & (cv_u > 0)
    rho_u, p_u = stats.spearmanr(cv_u[valid_u], F_u[valid_u])

    # --- rho(lambda, Fisher_U) --- does eigenvalue predict Fisher of U?
    lam_valid = eigenvalues[valid_u]
    rho_lf, p_lf = stats.spearmanr(lam_valid, F_u[valid_u])

    print(f"  {s_idx:>3}  {rho_xd:>+14.4f}  {p_xd:>9.2e}  "
          f"{rho_u:>+13.4f}  {p_u:>9.2e}  "
          f"{rho_lf:>+13.4f}  {p_lf:>9.2e}  "
          f"{valid_xd.sum():>11}  {valid_u.sum():>10}")

    results.append(dict(
        rho_xd=rho_xd, p_xd=p_xd,
        rho_u=rho_u,   p_u=p_u,
        rho_lf=rho_lf, p_lf=p_lf,
        cv_xd=cv_xd, F_xd=F_xd, valid_xd=valid_xd,
        cv_u=cv_u,   F_u=F_u,   valid_u=valid_u,
        tr=tr,
    ))

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────
rho_xd_arr = np.array([r['rho_xd'] for r in results])
rho_u_arr  = np.array([r['rho_u']  for r in results])
rho_lf_arr = np.array([r['rho_lf'] for r in results])

print("="*90)
print("AGGREGATE (mean ± std over 5 seeds)")
print(f"  rho(col_var, Fisher)  X_diff:   {rho_xd_arr.mean():+.4f} ± {rho_xd_arr.std():.4f}")
print(f"  rho(col_var, Fisher)  U:        {rho_u_arr.mean():+.4f}  ± {rho_u_arr.std():.4f}")
print(f"  rho(lambda_j, Fisher_U):        {rho_lf_arr.mean():+.4f}  ± {rho_lf_arr.std():.4f}")

diff = rho_xd_arr.mean() - rho_u_arr.mean()
print(f"\n  Alignment gap (rho_Xd - rho_U):  {diff:+.4f}")
if diff > 0.3:
    print("  → CONFIRMED: X_diff variance strongly aligns with discriminativity; U does not.")
elif diff > 0.1:
    print("  → PARTIAL: X_diff has better alignment than U, but gap is modest.")
else:
    print("  → NOT CONFIRMED: alignment gap is small.")

# ─────────────────────────────────────────────────────────────────────────────
# Scatter plots for seed 0
# ─────────────────────────────────────────────────────────────────────────────
r0 = results[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Cora k={K} — seed=0 (n_train={len(r0["tr"])})', fontsize=12)

def scatter_fisher(ax, col_vars, fisher, valid, title, rho):
    cv  = col_vars[valid]
    f   = fisher[valid]
    # colour by log-Fisher quantile
    q   = np.clip(np.log10(f + 1e-10), -3, 3)
    sc  = ax.scatter(cv, f, s=4, alpha=0.4, c=q, cmap='RdYlGn')
    plt.colorbar(sc, ax=ax, label='log10(Fisher)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('col_var (log)', fontsize=9)
    ax.set_ylabel('Fisher score (log)', fontsize=9)
    ax.set_title(f'{title}\nSpearman ρ = {rho:+.3f}', fontsize=10)
    ax.grid(True, alpha=0.3)

scatter_fisher(axes[0], r0['cv_xd'], r0['F_xd'], r0['valid_xd'],
               'X_diff: col_var vs Fisher', r0['rho_xd'])
scatter_fisher(axes[1], r0['cv_u'],  r0['F_u'],  r0['valid_u'],
               'U: col_var vs Fisher',      r0['rho_u'])

# Panel 3: lambda vs Fisher(U)
valid_u = r0['valid_u']
ax3 = axes[2]
q3  = np.clip(np.log10(r0['F_u'][valid_u] + 1e-10), -3, 3)
sc3 = ax3.scatter(eigenvalues[valid_u], r0['F_u'][valid_u],
                  s=4, alpha=0.4, c=q3, cmap='RdYlGn')
plt.colorbar(sc3, ax=ax3, label='log10(Fisher)')
ax3.set_yscale('log')
ax3.set_xlabel(r'$\lambda_j$', fontsize=9)
ax3.set_ylabel('Fisher score (log)', fontsize=9)
ax3.set_title(f'U: eigenvalue vs Fisher\nSpearman ρ = {r0["rho_lf"]:+.3f}', fontsize=10)
ax3.grid(True, alpha=0.3)

fig.tight_layout()
fname = os.path.join(OUT_DIR, 'layer3_fisher_alignment_cora.png')
fig.savefig(fname, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Print top-10 most discriminative columns in each space
# ─────────────────────────────────────────────────────────────────────────────
r0_cv_xd = r0['cv_xd']
r0_F_xd  = r0['F_xd']
r0_cv_u  = r0['cv_u']
r0_F_u   = r0['F_u']

print("\n--- Top-10 X_diff columns by Fisher score (seed 0) ---")
top_xd = np.argsort(np.where(np.isfinite(r0_F_xd), r0_F_xd, -1))[::-1][:10]
print(f"  {'col':>5}  {'Fisher':>10}  {'col_var':>12}  {'rank_by_var':>12}")
cv_rank_xd = stats.rankdata(-r0_cv_xd)
for c in top_xd:
    print(f"  {c:>5}  {r0_F_xd[c]:>10.4f}  {r0_cv_xd[c]:>12.2e}  {cv_rank_xd[c]:>12.0f}")

print("\n--- Top-10 U columns by Fisher score (seed 0) ---")
top_u = np.argsort(np.where(np.isfinite(r0_F_u), r0_F_u, -1))[::-1][:10]
print(f"  {'col':>5}  {'Fisher':>10}  {'col_var':>12}  {'lambda':>10}  {'rank_by_var':>12}")
cv_rank_u = stats.rankdata(-r0_cv_u)
for c in top_u:
    print(f"  {c:>5}  {r0_F_u[c]:>10.4f}  {r0_cv_u[c]:>12.2e}  "
          f"{eigenvalues[c]:>10.4f}  {cv_rank_u[c]:>12.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Fraction of columns where Fisher > median(Fisher) AND col_var > median(col_var)
# (alignment quadrant check)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Quadrant alignment check (seed 0) ---")
for label, cv_arr, F_arr, valid in [
    ('X_diff', r0['cv_xd'], r0['F_xd'], r0['valid_xd']),
    ('U',      r0['cv_u'],  r0['F_u'],  r0['valid_u']),
]:
    cv_v = cv_arr[valid]
    F_v  = F_arr[valid]
    med_cv = np.median(cv_v)
    med_F  = np.median(F_v)
    hh = np.mean((cv_v > med_cv) & (F_v > med_F))  # high-var & high-Fisher
    ll = np.mean((cv_v < med_cv) & (F_v < med_F))  # low-var & low-Fisher
    hl = np.mean((cv_v > med_cv) & (F_v < med_F))  # high-var but low-Fisher
    lh = np.mean((cv_v < med_cv) & (F_v > med_F))  # low-var but high-Fisher
    print(f"  {label}: high-var∩high-F={hh:.3f}  low-var∩low-F={ll:.3f}  "
          f"(aligned={hh+ll:.3f})  misaligned={hl+lh:.3f}")
